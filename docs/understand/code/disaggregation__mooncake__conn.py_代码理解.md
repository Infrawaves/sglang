# `python/sglang/srt/disaggregation/mooncake/conn.py` 代码理解

- 文件路径：`python/sglang/srt/disaggregation/mooncake/conn.py`
- 行数：4160 行（7 个类，74 个顶层函数和方法）
- 所属模块：PD 分离的 Mooncake 传输后端，在 `common/conn.py` 公共层之上实现真正的 RDMA 搬运。

## 1. 整体功能

这个文件把 `common/conn.py` 的四个公共类具体化为 Mooncake 版本：`MooncakeKVManager`、`MooncakeKVSender`、`MooncakeKVReceiver`、`MooncakeKVBootstrapServer`（最后一个没有新增内容）。字节搬运通过 Mooncake transfer engine 的 `batch_transfer_sync(session_id, src_addrs, dst_addrs, lengths)` 完成，源地址和目的地址都是"缓冲区基址 + 页号 × 每页字节数"。

**KV manager 的线程结构。**
- prefill 侧：若干个传输队列（`transfer_queues`，`FastQueue`），每个队列一个 `transfer_worker` 线程和一个线程池；一个 bootstrap 线程（`start_prefill_thread` 内的闭包）接收 decode 的注册消息、元数据和 ABORT；可选的失效 session 探测线程。
- decode 侧：一个 decode 线程（`start_decode_thread` 内的闭包）接收 prefill 的状态消息、AUX_DATA、CHUNK_READY、STAGING_REQ、ABORT_ACK；一个心跳线程（来自公共层）。

**消息与数据结构。**
- decode 注册消息（`build_receiver_registration_message`，房间号字段为 `"None"`）携带 decode 的所有缓冲区指针、`dst_tp_rank`、`dst_attn_tp_size`、每页字节数、层 ID、staging 信息，以及 `dst_dcp_size`、`dst_dcp_rank`、`dcp_kv_layout`、`draft_rows_dcp_collapsed`。prefill 解析成 `KVArgsRegisterInfo`，存入 `decode_kv_args_table[session_id]`。
- decode 元数据消息（`build_receiver_metadata_message`）携带某个房间的目标页号、aux 索引、状态页号、`required_dst_info_num`、`decode_prefix_len`。prefill 解析成 `TransferInfo`，存入 `transfer_infos[room][session_id]`。收齐 `required_dst_info_num` 个目标后，房间进入 `WaitingForInput`。

**传输分支（`transfer_worker`，3045 行）。** 对每个真实目标：
1. 若注册信息 `requires_dcp_relayout` 为 True（prefill DCP=1、decode DCP>1）：走 `send_kvcache_dcp`，按 token 行重排（`dcp_kv_layout == "page"` 时按页）。DSA index-K 随 KV 一起按行搬（`dsa_index_k_dcp_entries`）。
2. 否则若是 MLA / hybrid-MLA，或两侧 attn TP 相同：走 `send_kvcache`，按页整页复制（`_send_kvcache_generic`）。两侧 DCP 相同（含都为 1）就走这里。
3. 否则若开启 staging：`_do_staging_transfer`（gather→批量 RDMA→decode scatter）。
4. 否则：`send_kvcache_slice`，按 token、按头切片逐段传输。

最后一块还要通过 `maybe_send_extra` 发送状态组件（MAMBA/KDA、SWA、DSA、DSA_TAIL、QSA、MiniMax 等），再通过 `send_aux` 发送 metadata，最后 `conclude_transfer` 通知 decode。

**状态机与缓冲区安全。** 状态推进沿用公共层的 `KVPoll`。Mooncake 额外要求 `requires_transfer_drain = True`：每个入队块持有一份源缓冲区租约（`TransferLifetimeTracker`），ABORT 到达后关闭房间准入，等所有在途块结束后才发 ABORT_ACK；传输失败的房间被"毒化"，不发 ACK，decode 只能等释放超时。

**与 CP/DCP/TP/PP 的关系。**
- CP：`_get_dsa_cache_transfer_skip_flags` 调用公共层 `_should_skip_cp_replicated_state_transfer`，非 CP-TP sharing 下只让 CP rank 0 发送复制的状态。
- DCP：`_prepare_decode_registration` 调用 `requires_dcp_relayout` 决定走哪条路径；`send_kvcache_dcp` 用 `build_dcp_token_transfer_plan`（`pos % dcp == rank` 的行）或 `build_dcp_page_transfer_plan`（每隔 dcp 页取一页）选择要发给该 DCP rank 的数据。
- TP：KV 按 `self.attn_tp_size` 与 `dst_attn_tp_size` 比较选择路径；状态（MAMBA/KDA）按 `self.state_tp_size`（`linear_attn_tp_size()`，CP-TP sharing 下是整个 TP 组）比较，决定 `_send_mamba_state` 还是 `_send_mamba_state_slice`。
- PP：层对层配对优先用两侧注册的层 ID（`build_transfer_entry_pairs`），只有 `pp_size == 1` 才允许按位置回退（`allow_positional_fallback=self.pp_size == 1`）；没有层 ID 时用公共层的 `get_mla_kv_ptrs_with_pp`/`get_mha_kv_ptrs_with_pp` 切片。staging 在 PP>1 时用层 ID 配对 K/V 槽位。MiniMax 状态和 draft peer gather 不支持 PP>1。

## 2. 主要依赖与调用方

**依赖：**
- `disaggregation/common/conn.py`：四个公共基类、`AbortNotification`、`ABORT_TAG`/`ABORT_ACK_TAG`、`KVTransferError`。
- `disaggregation/common/utils.py`：`TransferKVChunk`、`FastQueue`、`group_concurrent_contiguous`（把连续页号合并成块）、`build_dcp_token_transfer_plan`、`build_dcp_page_transfer_plan`、`expand_dcp_page_rows`、`pack_int_lists`/`unpack_int_lists`、`AuxDataCodec`。
- `disaggregation/common/staging_handler.py`、`staging_buffer.py`：staging 模式（异构 TP 的 MHA）。
- `disaggregation/common/dcp_pack.py`：DCP 行重排时把源行打包到连续缓冲区（`try_pack_dcp_src`、`allocate_pack_buffers`）。
- `disaggregation/mooncake/draft_peer_gather.py`：page 布局 DCP 下从同组 attn-TP 对端收集草稿 KV 头。
- `disaggregation/utils.py`：`build_transfer_entry_pairs`、`is_elided_entry`、`resolve_dcp_dst_entry_indices`、`compute_mamba_state_slice_byte_blocks`、`should_send_replicated_state`、`filter_kv_indices_for_shard_rank`、DSA tail 相关函数。
- `distributed/parallel_state.py`：`get_mooncake_transfer_engine`。
- `runtime_context.py`：`get_parallel()`、`get_memory()`、`get_spec()` 等。

**调用方（grep 确认）：**
- `disaggregation/mooncake/__init__.py`：导出四个 Mooncake 类。
- `disaggregation/utils.py` 的 `get_kv_class`（887 行附近）：`--disaggregation-transfer-backend mooncake` 时返回这些类。
- 运行时由 `disaggregation/prefill.py`、`disaggregation/decode.py` 通过 `get_kv_class` 间接构造 manager、sender、receiver；`managers/disagg_service.py` 启动 bootstrap server。

## 3. 类与函数说明

### 3.1 模块级对象

| 名称 | 行号 | 作用 |
|---|---|---|
| `FAILED_SESSION_RECOVERIES` | 104 | Prometheus 计数器，记录通过探测解除黑名单的 session 数。 |
| `FailedSessionProbe` | 111 | 一个被拉黑 session 的探测计划（上次被请求时间、下次探测时间、当前间隔）。 |
| `TransferInfo` | 121 | prefill 侧保存的某房间、某 decode session 的目标信息：目标页号、aux 索引、状态页号、`required_dst_info_num`、是否 dummy、`decode_prefix_len`、注册 epoch、staging 信息。 |
| `TransferInfo.from_zmq` | 139 | 解析 decode 元数据消息；页号和 aux 都为空时视为 dummy。 |
| `KVArgsRegisterInfo` | 173 | prefill 侧保存的 decode 注册信息：所有缓冲区指针和每页字节数、层 ID、`dst_tp_rank`、`dst_attn_tp_size`、`dst_dcp_size/rank`、`dcp_kv_layout`、`requires_dcp_relayout`、`dcp_token_item_lens`、`draft_peer_ranks` 等。 |
| `KVArgsRegisterInfo.from_zmq` | 206 | 按固定帧位置解析注册消息，后加的字段按长度判断，兼容旧版本。 |
| `_uses_flat_entry_layout` | 265 | 判断一组 entry 是否按"每个 entry 一个区域"处理（MLA、强制 flat、有层 ID、单区域 SWA），否则按 MHA 的前半 K、后半 V 处理。传输和注册时校验共用。 |

### 3.2 `MooncakeKVManager`（288 行）

类属性：`supports_role_switch`、`supports_deferred_decode_kv_release`、`requires_transfer_drain` 都为 True；`AUX_DATA_HEADER = b"AUX_DATA"`。

**初始化与内存注册**

| 名称 | 行号 | 作用 |
|---|---|---|
| `__init__` | 301 | 校验自定义内存池；调用公共层初始化；强制开启延迟 KV 释放；prefill 创建租约追踪器、session 失败表、可选 draft peer gather、传输队列/线程池/worker 线程、bootstrap 线程和探测线程；decode 初始化 staging 分配器并启动 decode 线程。 |
| `init_engine` | 457 | 获取全局 Mooncake transfer engine。 |
| `_registerable_regions` | 460 | 汇总 KV、aux、状态的 `(ptr, len)`，去重并跳过长度为 0 的区域（Mooncake 不接受 0 长度）。 |
| `register_buffer_to_engine` | 492 | 批量注册内存区域。 |
| `deregister_buffer_to_engine` | 498 | 批量注销内存区域并清空连接池。 |
| `teardown` | 508 | 角色切换时停止所有线程、关闭 socket 和 ZMQ context、释放 pack 缓冲区、注销内存；不释放 KV 池本身。 |

**staging 与 pack 缓冲区**

| 名称 | 行号 | 作用 |
|---|---|---|
| `register_staging_room_bootstrap` | 616 | decode 侧记录 staging 房间的 bootstrap 信息和接收方。 |
| `set_kv_buffer_tensors` | 620 | 保存 K/V 缓冲区张量和 staging 槽位层 ID，供 staging gather 使用。 |
| `_register_staging_memory` | 636 | 向引擎注册一段 staging 内存。 |
| `_register_pack_buffers` | 639 | 注册 DCP pack 缓冲区，失败时回滚已注册的部分。 |
| `_release_pack_buffers` | 656 | 逐个注销 pack 缓冲区，注销失败的保留所有权。 |
| `_init_staging_buffers` | 675 | prefill 侧为每个队列分配 staging 缓冲区。 |
| `_init_staging_allocator` | 689 | decode 侧创建 staging 分配器。 |
| `_try_create_staging_strategy` | 701 | 有 K/V 张量时创建 `PrefillStagingStrategy`。 |
| `_do_staging_transfer` | 710 | 执行一块的 staging 传输：未就绪时短暂等待并重新入队；块超过 staging 环时抛 `StagingTransferRejected`；成功后发 CHUNK_READY。 |
| `_prefetch_staging_reqs` | 768 | 存在 attn TP 不同的目标时，提前向 decode 请求 staging 空间。 |
| `send_kvcache_staged` | 800 | 按头切片把所有层 gather 到本地 staging 缓冲区，一次（或 PP>1 时按层配对）RDMA 写到 decode staging 区。 |
| `_transfer_data` | 927 | 把 `(src, dst, len)` 列表交给 `batch_transfer_sync`，空列表返回 0。 |

**注册校验**

| 名称 | 行号 | 作用 |
|---|---|---|
| `_state_stride_mismatch` | 936 | 检查某状态组件每对 entry 的每项字节数两侧一致（按层 ID 或按 PP 切片配对），返回第一个不一致的描述。 |
| `_publish_peer_registration` | 1024 | 先校验 decode 的状态布局并记录结论，再把注册信息写入 `decode_kv_args_table`，同时解除该 session 的黑名单。 |
| `_validate_peer_state_layout` | 1048 | 对所有通用状态组件调用 `_state_stride_mismatch`；DFLASH 且状态 TP 与 attn TP 不同时额外要求草稿 KV 是可复制的单 entry 布局。 |

**KV 传输**

| 名称 | 行号 | 作用 |
|---|---|---|
| `_send_kvcache_generic` | 1160 | 通用的按页传输：合并连续页，按 flat 或 MHA K/V 布局得到每层 `(src_ptr, dst_ptr, item_len)`，再一次性或按批调用引擎；支持 host/device 两套目标页号和每层校验。 |
| `_validate_envelope_kv_layout` | 1422 | 开启 unified memory 时要求两侧 attn TP 相同且 KV 只有一个区域、每项字节数相同。 |
| `_await_transfer_futures` | 1468 | 等待所有提交的传输结束（即使有失败），返回第一个非零返回码或抛出第一个异常。 |
| `send_kvcache` | 1491 | 主 KV 池整页传输的入口，处理 HiSparse 的 device 目标页，然后调用 `_send_kvcache_generic`。 |
| `_dsa_index_k_dcp_params` | 1542 | 把 DSA index-K 的行（fp8 key 和 scale 两段）换算成各自宽度的单位下标，必要时打包，生成传输参数。 |
| `dsa_index_k_dcp_entries` | 1598 | DCP 重排时找出需要随 KV 一起按行搬的 DSA index-K 缓冲区对和每页字节数；跳过复用上一层 top-k 的空层。 |
| `send_kvcache_dcp` | 1646 | DCP1→DCPn 的重排传输：page 布局且无草稿时按页选取；否则用 token 计划选出 `pos % dcp == rank` 的目标行，可打包；处理 DSA index-K、草稿 KV（同宽直接搬、不同宽按头切片或从对端收集）。 |
| `_send_gathered_draft_windows` | 1987 | 把从本 rank 和对端收集的草稿行分窗口打包到 pack 缓冲区再写到 decode。 |
| `send_kvcache_slice` | 2038 | 异构 TP 的 MHA：对每层、每页、每 token 计算头切片的源/目的地址，逐段传输。 |

**aux 数据**

| 名称 | 行号 | 作用 |
|---|---|---|
| `send_aux` | 2171 | RDMA 写 metadata 缓冲区；NVLINK 自定义池或设置了 `SGLANG_MOONCAKE_SEND_AUX_TCP` 时改走 TCP。 |
| `send_aux_tcp` | 2199 | 把 aux 序列化后通过 ZMQ 发给 decode。 |
| `send_aux_data_to_endpoint` | 2224 | 编码并发送一条 AUX_DATA 消息。 |
| `_handle_aux_data` | 2247 | decode 侧把 AUX_DATA 反序列化写回本地 metadata 缓冲区。 |

**状态组件传输**

| 名称 | 行号 | 作用 |
|---|---|---|
| `_get_dsa_cache_transfer_skip_flags` | 2267 | 返回 `(skip_kv, skip_state)`：CP 复制状态只让 rank 0 发；hybrid-MLA 在 prefill attn TP > decode attn TP 时，组内非首 rank 跳过 KV。 |
| `_resolve_draft_peer_ranks` | 2297 | page 布局 DCP、带草稿、hybrid-MLA、prefill attn TP > decode attn TP 时，确定本 rank 要代为收集哪些对端的草稿头，并分配 pack 缓冲区；要求 `pp_size == 1` 且 `attn_cp_size == 1`。 |
| `_wait_for_draft_peers` | 2381 | 等待对端写完该块的草稿 KV，超时抛 `DraftPeerNotReady`。 |
| `_is_generic_kvcache_state_type` | 2413 | 列出走 `_send_kvcache_generic` 按页传输的状态类型（SWA、DSA、QSA、SWA_RING、DRAFT_RING 等）。 |
| `_requires_exact_state_index_match` | 2430 | 列出页号必须两侧数量完全相同、不能截断的状态类型。 |
| `_prepare_sharded_dsa_indices` | 2441 | KV sharding 下只保留本 shard 拥有的 DSA 页，并保持源/目的对齐。 |
| `maybe_send_extra` | 2456 | 最后一块时逐个状态组件发送：DCP 重排时跳过 DSA（已随 KV 发送）；MAMBA 按 `state_tp_size` 与 decode attn TP 是否相同选择整份或切片；DSA_TAIL 按槽位；通用类型按页（含 SWA 头切片、QSA 复制检查、DRAFT_RING 过滤 0 行）；MiniMax 要求同 TP、PP=1。 |
| `_send_slot_state` | 2792 | 按 PP 切片 DSA tail 的目标指针，构造块并传输。 |
| `_send_mamba_state` | 2829 | 两侧状态 TP 相同时，按层配对整份复制一个请求槽位的 Mamba/KDA 状态，并校验每槽字节数一致。 |
| `_send_mamba_state_slice` | 2866 | 两侧状态 TP 不同时，用 `compute_mamba_state_slice_byte_blocks` 按头（GDN conv 按子块）切片传输；缺维度信息时退回整份复制。 |
| `_send_swa_state_slice` | 2975 | 非 MLA SWA 状态在异构 TP 下用公共层 `_get_mha_head_slice_blocks` 生成块并并行传输。 |

**传输生命周期**

| 名称 | 行号 | 作用 |
|---|---|---|
| `_poison_unsettled_send` | 3018 | 传输返回失败时毒化房间，阻止发出排空 ACK（底层批次可能仍在写）。 |
| `_release_transfer_lease` | 3028 | 归还一块的源缓冲区租约。 |
| `_maybe_ack_drained_abort` | 3033 | 在房间已关闭且租约排空后才调用父类发出 ACK。 |
| `transfer_worker` | 3045 | 传输线程主循环：取块、计数、等待前向事件、跳过已失败房间，按第 1 节的分支对每个目标发 KV、状态和 aux，收齐目标后结束房间，最后清理房间记录。 |

**控制消息与线程**

| 名称 | 行号 | 作用 |
|---|---|---|
| `_handle_abort_notification` | 3541 | 识别 ABORT 消息并分发到延迟或旧版处理。 |
| `_handle_deferred_abort_notification` | 3558 | 关闭房间准入、标记 Failed、登记 ACK 目标并在已排空时立即 ACK。 |
| `_handle_legacy_abort_notification` | 3580 | 旧格式：标记 Failed 并立即回 ABORT_ACK。 |
| `_prepare_decode_registration` | 3615 | 解析注册消息，计算 `requires_dcp_relayout`；需要重排时准备 `dcp_token_item_lens`、token 布局下分配 pack 缓冲区、解析草稿对端。 |
| `start_prefill_thread` | 3641 | 启动 bootstrap 线程：处理 WATERMARK、STAGING_RSP、ABORT、注册消息（带注册 epoch 和拒绝记录）和元数据消息（收齐后置 WaitingForInput）。 |
| `start_decode_thread` | 3751 | 启动 decode 线程（处理 AUX_DATA、CHUNK_READY、STAGING_REQ、ABORT_ACK、状态消息）和心跳线程。 |
| `add_transfer_request` | 3792 | 把一块 KV 包成 `TransferKVChunk`，按目标 session 端口之和选择队列，先取租约再入队；dummy rank 没有 `transfer_infos` 时直接返回。 |
| `mark_prefill_forward_synced` | 3864 | 通知 draft peer gather 本次前向已同步。 |
| `get_session_id` | 3868 | 返回本引擎的 session ID。 |
| `_unblacklist_session` | 3871 | 从失败集合、失败计数和探测表中移除 session。 |
| `_send_probe` | 3877 | 调用引擎探测一个 session，异常返回 -1。 |
| `_run_one_probe_pass` | 3884 | 对到期的黑名单 session 做探测（指数退避），长时间无人请求的直接移除。 |
| `_failed_session_probe_loop` | 3935 | 探测线程主循环。 |
| `build_receiver_registration_message` | 3945 | decode 侧编码注册消息（指针、层 ID、TP rank、attn TP、每页字节、DCP 大小与 rank、DCP 布局、staging 信息）。注释说明不带 PP rank，因为 decode PP 只能等于 prefill PP 或为 1。 |
| `build_receiver_metadata_message` | 4021 | decode 侧编码某房间的元数据，dummy 目标发空页号。 |

### 3.3 `MooncakeKVSender`（4051 行）

| 名称 | 行号 | 作用 |
|---|---|---|
| `__init__` | 4054 | 调用父类初始化；如果房间的租约已经被提前到达的 ABORT 关闭，就立即 abort。 |
| `is_source_release_safe` | 4073 | 源缓冲区租约是否已排空。 |
| `failure_exception` | 4076 | 抛出失败异常但不清理传输记录（等调度器在 TP/CP 达成排空共识后再 `clear`）。 |
| `clear` | 4090 | 关闭租约，未排空时抛错；排空后发 ACK、清除租约，再调用父类清理。 |
| `send` | 4105 | 加 trace 装饰后调用父类 `send`。 |
| `_record_transfer_indices` | 4119 | KV sharding 下只按本 shard 的 DSA 页计入指标。 |

### 3.4 `MooncakeKVReceiver`（4141 行）与 `MooncakeKVBootstrapServer`（4159 行）

| 名称 | 行号 | 作用 |
|---|---|---|
| `MooncakeKVReceiver.failure_exception` | 4142 | 抛出失败异常但不清理（decode 队列在目标页释放栅栏之后才 `clear`）。 |
| `MooncakeKVBootstrapServer` | 4159 | 直接继承公共 bootstrap server，无新增逻辑。 |

## 4. 与 PCP+DCP 方案 A 的关系

计划 §4.10 对本文件的要求是："prefill 是 CP8+DCP8、decode 是 DCP8 时走按页传输；KDA 状态继续按头 r 对 r 传输。"对照代码，这两条在现有路径上基本已经成立，主要工作是补校验和处理方案 A 下"真实 attn TP=1、注册 TP=8"带来的不一致。

### 4.1 现有代码已经覆盖的部分（推断）

- **相同 DCP 按页传输。** `_prepare_decode_registration`（3615）调用 `requires_dcp_relayout`：两侧 DCP 都是 8 且 rank 相等时返回 False，`transfer_worker`（3197–3329 行）不进 `send_kvcache_dcp`，而是因为 `is_hybrid_mla_backend` 走 `send_kvcache` → `_send_kvcache_generic`，用 `req.dst_kv_indices[kv_chunk.index_slice]` 整页复制。前提是两侧页号是同一套 DCP 虚拟页编号，且每个物理页的行按同一 owner 规则存放（推断，需要和 `mem_cache/kv/alloc/id_space.py` 对照确认）。
- **DSA index-K 按页传输。** 不重排时 `maybe_send_extra`（2471–2477 行）不会跳过 DSA，DSA 按通用状态类型走 `_send_kvcache_generic` 按页复制，并经过 `_state_stride_mismatch` 校验每页字节数。K3.1 的 index-K 在 DCP 下也按 token 分片（`set_dsa_shards_index_k`，计划 §4.7），两侧几何一致时可以直接用（推断）。
- **KDA 状态按头 r 对 r。** K3.1 的 KDA 状态类型是 `StateType.MAMBA`（`disaggregation/utils.py` 1678–1680 行：hybrid-MLA = MLA + MAMBA）。`maybe_send_extra`（2539–2578 行）比较 `self.state_tp_size` 和 `dst_attn_tp_size`：CP-TP sharing 下 `state_tp_size` 是 TP 组宽度 8，decode TP8 时相等，走 `_send_mamba_state` 整份复制，CP rank r 发给 decode rank r。公共层 `_should_skip_cp_replicated_state_transfer` 在 CP-TP sharing 下返回 False，所以每个 CP rank 都会发。
- **DCP1→DCPn 按行传输**（计划的背景描述）对应 `send_kvcache_dcp`（1646）+ `build_dcp_token_transfer_plan`，`dcp_kv_layout == "page"` 时用 `build_dcp_page_transfer_plan`。方案 A 的 prefill 是 DCP8，不会进入这条路径。

### 4.2 需要改或需要确认的函数

| 位置 | 内容 |
|---|---|
| `_prepare_decode_registration`（3615） | 同 DCP 的分支目前没有任何 KV 几何校验。重排路径会调用 `prepare_dcp_token_item_lens` 比较每 token 字节数，按页路径却只按 prefill 的 `kv_item_lens` 计算目的地址（`_send_kvcache_generic` 1329–1331 行）。方案 A 下如果 prefill 的 KV 池没有真正按 DCP 切分（比如仍然每页存完整的行），按页复制会静默写错。建议在 `dst_dcp_size > 1` 且不重排时，校验 `dst_kv_item_lens == kv_args.kv_item_lens`，并校验两侧 `dcp_kv_layout` 都是 `token`（计划未提及）。 |
| `transfer_worker`（3197–3329） | 分支选择不需要改。需要确认 `kv_chunk.index_slice` 的切分落在 DCP 虚拟页边界上，否则 `req.dst_kv_indices[index_slice]` 和 prefill 页号会错位（对应计划 §4.1 的 chunked prefill 对齐约束，但计划没有把它和 PD 传输联系起来）。 |
| `_get_dsa_cache_transfer_skip_flags`（2267） | 用的是真实的 `self.attn_tp_size`。方案 A prefill 的 attn TP 是 1（头复制），注册的却是 8。`1 > dst_attn_tp_size` 永远不成立，所以 `skip_kv` 为 False，每个 CP rank 都发 KV，结果正确，但这是"碰巧对"（推断）。如果改成用注册的 TP 宽度，要同步检查这里。 |
| `maybe_send_extra`（2604–2608） | `has_heterogeneous_attn_tp = self.attn_tp_size(1) != dst_attn_tp_size(8)` 在方案 A 下恒为 True。对 hybrid-MLA 它只影响 QSA 状态，K3.1 不涉及。但说明本文件到处混用"真实 attn TP"和"注册 TP"，加新分支时要注意（计划未提及）。 |
| `_validate_envelope_kv_layout`（1422） | 开启 `--enable-unified-memory` 时，`self.attn_tp_size(1) != dst_attn_tp_size(8)` 会直接报错。CP-TP sharing 本身就有这个问题，方案 A 继承了它（计划未提及）。 |
| `_prefetch_staging_reqs`（768） | 判断是否需要 staging 用的是 `dst_attn_tp_size != self.attn_tp_size`，方案 A 下恒为 True。只有设置了 `kv_buffer_tensors` 才会真正预取，MLA 通常不设（推断）。建议在方案 A 下禁用 staging，或者显式校验。 |
| `_validate_peer_state_layout`（1058–1102） | DFLASH 条件里的 `self.state_tp_size != self.attn_tp_size` 在 CP-TP sharing 下为 True（8 ≠ 1），会触发草稿布局检查。计划把 DFLASH 放在 P4 单独设计，这里是要改的入口之一。 |
| `build_receiver_registration_message`（3945） | decode 已经发送 `dst_dcp_size/rank` 和 `dcp_kv_layout`，prefill 能据此判断。不需要改。 |

### 4.3 计划可能遗漏的点

1. **方案 A prefill 只能对接 DCP 完全相同的 decode。** 公共层 `requires_dcp_relayout` 对 DCP8→DCP1、DCP8→DCP4 都会报错，报错发生在 prefill 的 bootstrap 线程处理注册消息时（`start_prefill_thread` 3672–3685 行，把 session 记入 `rejected_sessions`），然后该 session 的每个请求都以 "Rejected Mooncake KV registration" 失败。decode 侧拿到的 `PrefillServerInfo` 里没有 prefill 的 DCP 大小，无法提前拒绝。计划说 decode 是 DCP8，但没有写明"其他 decode 拓扑不受支持"，也没有说明报错方式（计划未提及）。
2. **草稿 KV（MTP/DFLASH）在按页路径上的布局。** 两侧 DCP 相同时，草稿 entry 也走 `send_kvcache` 按页复制。decode 的草稿池可能是 DCP-collapsed（`draft_rows_dcp_collapsed`），也可能按虚拟页放大（`prepare_dcp_token_item_lens` 里的 `dst_page_scale`）。prefill 的草稿池在方案 A 下用哪种布局，需要和 decode 一致，否则按页复制会错位（计划未提及）。
3. **`decode_prefix_len` 在按页路径上不参与对齐检查。** 重排路径会检查 `decode_prefix_len` 对齐到 `page_size × dcp`（`build_dcp_token_transfer_plan`），按页路径没有这个检查。方案 A 下 decode 命中前缀时，两侧前缀长度和页偏移需要一致（推断）。
4. **`pp_size > 1`。** 本文件的 PP 处理依赖层 ID 配对，只有 PP=1 才允许按位置回退；`_resolve_draft_peer_ranks` 明确要求 `pp_size == 1` 且 `attn_cp_size == 1`；MiniMax 状态不支持 PP>1。计划把 PP 列为非目标，但如果以后放开，这些都是硬限制。
5. **dummy 目标。** decode TP < prefill 注册 TP 时，MLA 只有一个 prefill rank 发真实数据，其他 rank 收到 dummy 元数据后直接 Success（`transfer_worker` 3467–3471 行）。方案 A 每个 prefill rank 只存 1/8 的 KV，任何一个 rank 被标成 dummy 都会丢数据。所以方案 A 必须要求 decode TP 等于 prefill CP（=DCP）大小，计划应写明这条约束（计划未提及）。
