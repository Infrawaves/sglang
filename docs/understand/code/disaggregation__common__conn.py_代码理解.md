# `python/sglang/srt/disaggregation/common/conn.py` 代码理解

- 文件路径：`python/sglang/srt/disaggregation/common/conn.py`
- 行数：4016 行（14 个类，150 个顶层函数和方法）
- 所属模块：PD 分离（prefill/decode disaggregation）的传输后端公共层，Mooncake、NIXL、Mori、Ascend 等后端都继承这里的类。

## 1. 整体功能

这个文件实现 PD 分离里与具体传输引擎无关的部分：rank 之间怎么配对、请求状态怎么流转、控制消息怎么收发。真正搬运字节的 RDMA 调用留给子类（例如 `mooncake/conn.py`）。

**四个核心类。**
- `CommonKVManager`：每个进程一个，prefill 和 decode 共用同一个类、按 `disaggregation_mode` 走不同初始化分支。它持有 ZMQ PULL socket（`rank_port`）、每个 `bootstrap_room` 的状态表 `request_status`、失败原因表 `failure_records`，以及 decode 侧的连接池、拓扑缓存和心跳线程。
- `CommonKVSender`：prefill 侧每个请求一个，负责把 KV 页号分块交给 manager 的传输队列（`add_transfer_request` 由子类实现），并轮询状态。
- `CommonKVReceiver`：decode 侧每个请求一个，负责握手（向 bootstrap server 查询 prefill rank 的 IP/端口，注册 KV 缓冲区指针），然后发送目标页号（`send_metadata`），再轮询状态。
- `CommonKVBootstrapServer`：prefill 实例上的 aiohttp HTTP 服务，保存每个 prefill rank 的地址（按 DP/CP/TP/PP 四级索引），提供 `/route`、`/register_dp_rank`、`/query_dp_ranks`、`/health`。

**传输流程。**
1. prefill 每个 rank 在 `CommonKVManager.__init__` 里调用 `register_to_bootstrap`，用 HTTP PUT 把自己的拓扑和 `rank_ip/rank_port` 登记到 bootstrap server。
2. decode 侧 `try_ensure_parallel_info` 用 `/route?...=-1` 拉取 prefill 的整体拓扑（`PrefillServerInfo`），做一致性检查，再由 `_resolve_rank_mapping` 算出本 decode rank 要连接哪些 prefill TP/CP/PP rank。
3. `CommonKVReceiver.init` 对每个目标 CP rank 起一个共享的 setup 任务（`_fetch_and_register_key`），查询具体 rank 地址并发送注册消息（`build_receiver_registration_message`，内容由子类定义）。
4. decode 分配好 KV 页后调用 `send_metadata`，prefill 侧收到后把房间状态推进到 `WaitingForInput`；prefill 前向完成后 `CommonKVSender.send` 分块提交传输；最后一块传完后 prefill 用 `send_kv_status_message` 通知 decode。
5. decode 侧 `apply_prefill_status` 收齐 `required_prefill_response_num` 个 prefill rank 的 Success 后，把房间标为 Success。

**状态机（`KVPoll`）。** `Bootstrapping → WaitingForInput → Transferring → Success`，`Failed` 是终态。`update_status` 用 `max(current, status)` 推进，并保证 Failed 不会被覆盖、已清理的房间不会被非起始状态"复活"。`conclude_transfer` 是 prefill 侧结束一个房间的统一入口，会把已记录的失败原因优先于 Success。另有一套"延迟 abort ACK"机制（`AbortNotification`/`AbortAck`、`_deferred_*` 字段），保证 decode 在 prefill 的写入排空之前不释放目标页。

**与 CP/DCP/TP/PP 的 rank 映射。**
- CP：`enable_all_cp_ranks_for_transfer`（约 444 行）决定 prefill 是否所有 CP rank 都发送；`is_dummy_cp_rank`（约 494 行）在只让 CP rank 0 发送时把其他 CP rank 标成"假参与者"，它们直接进入 `WaitingForInput`，最后一块时直接置 Success。
- CP-TP group sharing：`register_to_bootstrap`（1940–1942 行）把 `(attn_cp_size, attn_cp_rank)` 当成 TP 维度注册，CP 维度写成 `(1, 0)`。所以 decode 看到的是一个 TP=cp_size、CP=1 的 prefill，CP rank r 就按"CP=1 prefill 的 TP rank r"配对。此时 `enable_all_cp_ranks_for_transfer` 和 `is_dummy_cp_rank` 都为 False，每个 CP rank 都是真实发送方。
- DCP：`requires_dcp_relayout`（591 行）只接受两种拓扑：两侧 DCP 大小相同（要求 DCP rank 一一相等，不需要重排），或 prefill DCP=1、decode DCP>1 且为 MLA/hybrid-MLA（需要按 token 行重排）。其他组合直接报错。decode 侧 `try_ensure_parallel_info`（1753–1767 行）要求 decode DCP>1 时 prefill 的 `attn_cp_size == 1` 且未开 KV sharding。
- TP：`_resolve_rank_mapping` 处理 decode TP 等于、大于、小于 prefill TP 三种情况，MLA 在 decode TP < prefill TP 时只从一个 rank 取真数据，其余 rank 发"dummy"元数据。
- PP：decode PP 必须等于 prefill PP 或为 1；为 1 时要从所有 prefill PP rank 拉取，`required_prefill_response_num` 乘以 `pp_size`。`get_mha_kv_ptrs_with_pp`、`get_mla_kv_ptrs_with_pp`、`_mla_slice_ptrs_for_pp` 负责把 prefill 一个 PP 阶段的本地层对应到 decode 全模型的层指针上。`pp_size > 1` 的 prefill 用 `PPConsensusStore` 存状态，`check_status_pp_consensus` 取所有 PP 阶段的最小状态。

## 2. 主要依赖与调用方

**依赖：**
- `disaggregation/base/conn.py`：`BaseKVManager`/`BaseKVSender`/`BaseKVReceiver`/`BaseKVBootstrapServer` 基类，`KVArgs`、`KVPoll`、`StateType`、`KVTransferMetric`。
- `disaggregation/common/bootstrap.py`：`DeferredBootstrap`（`prefill_complete` 分配策略下的延迟握手）。
- `disaggregation/common/transfer_lifetime.py`：`TransferLifetimeTracker`（源缓冲区租约，只在 Mooncake prefill 上设置）。
- `disaggregation/pp_consensus_store.py`：`PPConsensusStore`。
- `disaggregation/utils.py`：`filter_kv_indices_for_cp_rank`、`filter_kv_indices_for_shard_rank`、`MetadataBuffers`、`get_dsv41_spec_layout`。
- `disaggregation/common/dcp_pack.py`（`init_dcp_pack_buffers`）、`disaggregation/common/staging_buffer.py`（头切片计算）。
- `runtime_context.py`：`get_parallel()` 提供 `attn_tp/cp/dcp/dp` 的大小和 rank、`enable_cp_tp_group_sharing`、`enable_dsa_cache_layer_split`、`enable_kv_cache_sharding`；`linear_attn_tp_size()` 给出状态（如 KDA/GDN）的 TP 宽度。
- `mem_cache/page_interleave.py`：`get_kv_shard_group`。
- 第三方：`zmq`、`aiohttp`、`requests`、`msgspec`、`torch.distributed`。

**调用方（grep 确认）：**
- `disaggregation/mooncake/conn.py`、`nixl/conn.py`、`mori/conn.py`、`ascend/conn.py`：继承四个公共类。
- `disaggregation/prefill.py`：导入 `CommonKVManager`。
- `disaggregation/decode.py`：导入 `CommonKVManager`、`CommonKVReceiver`、`ParallelInfoState`（驱动 `try_ensure_parallel_info` 和握手）。
- `disaggregation/utils.py`：只在 `TYPE_CHECKING` 下导入公共类做类型标注；`get_kv_class` 按后端返回子类（如 Mooncake）。
- `managers/disagg_service.py`：启动 `CommonKVBootstrapServer`（或后端子类）。
- `disaggregation/role_switch.py`：用 `CommonKVReceiver.close_all_sockets` 做 P/D 角色切换。

## 3. 类与函数说明

### 3.1 模块级函数与常量

| 名称 | 行号 | 作用 |
|---|---|---|
| `ABORT_TAG` / `ABORT_ACK_TAG` / `_ANY_GENERATION` | 83–87 | ABORT 与 ABORT_ACK 消息的首帧标签；`_ANY_GENERATION` 是"不限定 generation"的哨兵对象。 |
| `_DECODE_BOOTSTRAP_WORKERS` 等常量 | 98–101 | decode 握手线程池大小（32）、每个 prefill 同时运行的 setup 任务上限（2）、拓扑查询线程数（4），注释说明都是随意取值。 |
| `_bootstrap_http_timeout` | 104 | 从环境变量返回 bootstrap HTTP 请求的 (连接超时, 读超时)。 |
| `_get_bootstrap_session` | 134 | 按 `bootstrap_addr` 返回线程本地的 keep-alive `requests.Session`，避免高并发下耗尽临时端口。 |

### 3.2 辅助数据类型

| 名称 | 行号 | 作用 |
|---|---|---|
| `ParallelInfoState` | 112 | `try_ensure_parallel_info` 一次调用的结果：`FAILED`/`READY`/`PENDING`。 |
| `_KeySetupResult` | 121 | 一个 setup 任务的结果，`bootstrap_infos` 和 `error` 二选一。 |
| `_SetupJob` | 127 | 排队中的 setup 任务：`(bootstrap_key, epoch)`、执行函数和共享 future。 |
| `KVTransferError` | 151 | 传输失败异常，带 `bootstrap_room`、原因和"是否来自其他 rank"。 |
| `KVTransferError.__init__` | 152 | 保存房间号、失败原因和 `is_from_another_rank`。 |
| `KVTransferError.__str__` | 163 | 格式化为 `KVTransferError(bootstrap_room=...): reason`。 |
| `PrefillServerInfo` | 168 | decode 从 bootstrap server 拿到的 prefill 拓扑（attn TP/CP、DP、PP、page_size、dtype、layout、各种开关），以及 decode 侧算好后回填的 rank 映射字段。 |
| `PrefillServerInfo.__post_init__` | 201 | 把 JSON 里的字段统一转成 int/str/bool。 |
| `PrefillRankInfo` | 219 | 单个 prefill rank 的 `rank_ip`/`rank_port`。 |
| `PrefillRankInfo.__post_init__` | 223 | 类型转换。 |
| `AckTarget` | 228 | 延迟 ACK 的目标 `(ip, port, generation)`。 |
| `AbortNotification` | 234 | decode→prefill 的 ABORT 消息（房间号，可选回执地址和 generation）。 |
| `AbortNotification.from_zmq` | 241 | 解析 ZMQ 多帧消息，容忍旧版本缺字段，格式错误时返回 None 或降级。 |
| `AbortNotification.to_zmq` | 275 | 编码成多帧消息。 |
| `AbortNotification.deferred_ack_target` | 288 | 有完整回执地址和 generation 时返回 `AckTarget`，否则 None（不需要延迟 ACK）。 |
| `AbortAck` | 302 | prefill→decode 的 ABORT_ACK（房间、prefill 唯一 rank、generation）。 |
| `AbortAck.from_zmq` | 308 | 解析 ABORT_ACK，格式错误返回 None。 |
| `AbortAck.to_zmq` | 322 | 编码 ABORT_ACK。 |
| `DeferredAbortAckState` | 333 | decode 侧每个房间当前 generation 及已 ACK 的 prefill rank 集合。 |

### 3.3 `CommonKVManager`（338 行）

类属性（339–368 行）定义了子类可覆盖的开关：`kv_status_msg_tag`/`kv_status_msg_carries_reason`（状态消息的线格式）、`reports_transfer_latency`、`kv_shard_rank/size`、`DEFAULT_PREFILL_FAILURE_REASON` 等。

**初始化与 rank 身份**

| 名称 | 行号 | 作用 |
|---|---|---|
| `__init__` | 370 | 读取 attn TP/CP/DCP/DP/PP 的大小和 rank、`state_tp_size`；计算 `enable_all_cp_ranks_for_transfer`（444）；绑定 ZMQ PULL socket；prefill 分支计算 `is_dummy_cp_rank`（494）、同步 bootstrap 端口并 `register_to_bootstrap`，PP>1 时用 `PPConsensusStore`；decode 分支初始化连接池、心跳参数、超时和握手状态。 |
| `prepare_send_state_indices` | 568 | 发送前处理状态页号的钩子，默认原样返回。 |
| `_should_skip_cp_replicated_state_transfer` | 573 | prefill CP>1、非 rank 0、且未开层切分 / CP-TP sharing / KV sharding 时返回 True，表示这个 rank 不再重复发送 CP 复制的状态。 |
| `requires_dcp_relayout` | 591 | 判断与某个 decode DCP rank 配对是否需要按行重排：同 DCP 大小时要求 rank 相等、返回 False；prefill DCP=1→decode DCP>1（MLA）返回 True；其他组合抛错。 |
| `prepare_dcp_token_item_lens` | 611 | 校验 decode 注册的每个 KV entry 的 token 字节宽度与 prefill 一致（草稿 entry 可能按 `dcp_size` 放大），返回每个 entry 每 token 字节数。 |
| `_register_staging_memory` | 645 | 向传输引擎注册一段内存，基类未实现。 |
| `_register_pack_buffers` | 650 | 逐个注册 DCP pack 缓冲区。 |
| `_init_dcp_pack_buffers_once` | 654 | 第一次需要 DCP 行重排时，按最大 prefill token 数为每个传输队列分配并注册 pack 缓冲区。 |

**状态与状态消息**

| 名称 | 行号 | 作用 |
|---|---|---|
| `check_status` | 676 | 返回房间状态。 |
| `check_status_pp_consensus` | 679 | PP>1 时收集各阶段状态，有缺失返回 Bootstrapping，否则取最小值。 |
| `check_peer_aux_buffer_count` | 685 | decode 注册的 metadata 缓冲区个数与本地不同时打一次错误日志。 |
| `build_receiver_registration_message` | 704 | decode 注册消息的编码，由子类实现。 |
| `build_receiver_metadata_message` | 707 | decode 元数据（页号、aux、状态页号、前缀长度、是否 dummy）的编码，由子类实现。 |
| `validate_aux_buffers` | 721 | 开启自定义 info 传输时，校验两侧 metadata 缓冲区布局兼容，不兼容抛 `ValueError`。 |
| `update_status` | 753 | 状态推进的唯一入口：不存在的房间只接受 Bootstrapping/WaitingForInput 作为起始状态并清理旧的延迟 ACK 状态；Failed 是终态；其余取 max。 |
| `_notify_bootstrap` | 784 | 延迟分配模式下，把 `DeferredBootstrap` 产生的通知（就绪或失败）发给 decode。 |
| `_handle_bootstrap_message` | 797 | 延迟分配模式下消费 BOOTSTRAP/ABORT 控制消息；ABORT 仍返回 False 让后端继续处理。 |
| `record_failure` | 820 | 加锁记录房间失败原因。 |
| `_room_notify_targets` | 824 | 返回房间所有非 dummy decode 端点，用于广播状态。 |
| `_encode_kv_status_message` | 841 | 编码 `[room, status, prefill_rank(, reason)]`，按需加 tag 帧。 |
| `parse_kv_status_message` | 859 | decode 侧解析 prefill 状态消息，不是状态消息时返回 None。 |
| `send_kv_status_message` | 886 | 把状态推送给一组 decode 端点，发送失败只记日志。 |
| `send_chunk_ready` | 912 | staging 模式下通知 decode 某块 RDMA 已完成（CHUNK_READY）。 |
| `handle_chunk_ready` | 937 | decode 侧处理 CHUNK_READY，交给 staging handler 做 scatter。 |
| `conclude_transfer` | 957 | prefill 结束房间的统一入口：有已记录失败则把 Success 改成 Failed，更新状态并通知 decode；延迟分配下 Success 推迟到 `sender.clear()` 发布。 |
| `conclude_failure` | 1008 | `conclude_transfer(status=Failed)` 的简写。 |
| `apply_prefill_status` | 1023 | decode 侧处理某个 prefill rank 的状态：Success 计入 `prefill_response_tracker`，收齐后置 Success；Failed 记录原因并置 Failed。 |

**延迟 abort ACK（decode 侧与 prefill 侧）**

| 名称 | 行号 | 作用 |
|---|---|---|
| `register_deferred_abort_room` | 1081 | decode 为房间开一个新 generation 并返回其编号。 |
| `arm_deferred_abort_room` | 1091 | 复用或新建 generation；要求排空时，旧接收方的 generation 未释放前返回 None。 |
| `is_deferred_abort_generation_armed` | 1111 | 判断某 generation 是否是房间当前的 generation。 |
| `note_abort_ack` | 1121 | 只为当前 generation 记录某 prefill rank 的 ACK。 |
| `handle_abort_ack_message` | 1129 | decode 侧消费 ABORT_ACK 消息。 |
| `is_abort_release_safe` | 1147 | 已收到的 ACK 数 ≥ 需要的数量时返回 True，表示可以释放目标页。 |
| `clear_deferred_abort_state` | 1170 | 清掉房间的 ACK 状态（指定 generation 时只清自己的）。 |
| `_prefill_unique_rank` | 1180 | prefill 发送方的唯一编号 `attn_tp_rank*(pp*cp) + pp_rank*cp + attn_cp_rank`，用于状态消息和 ACK。 |
| `_send_abort_ack` | 1188 | 向一个 decode 目标发送 ABORT_ACK，失败只记 debug。 |
| `_maybe_ack_drained_abort` | 1200 | 房间没有在途块且未被"毒化"时，发出所有挂起的 ACK。 |
| `_send_abort_acks` | 1211 | 逐个目标发送 ACK（多个 decode rank 共享一个房间时扇出）。 |
| `register_deferred_ack_target` | 1219 | 登记一个等待排空后发送的 ACK 目标（按 decode 端点去重）。 |
| `poison_deferred_ack_room` | 1237 | 后端无法证明写入已停止时，丢弃 ACK 目标并标记房间，decode 只能等超时。 |

**指标**

| 名称 | 行号 | 作用 |
|---|---|---|
| `get_kv_replica_factor` | 1242 | 返回 KV 被复制到几个 decode 目标（未解析时返回 1 并告警）。 |
| `resolve_kv_replica_factor` | 1251 | MLA 时用 `required_dst_info_num` 作为复制因子。 |

**worker 接收与 decode 握手**

| 名称 | 行号 | 作用 |
|---|---|---|
| `_make_worker_recv` | 1266 | 构造 worker 线程用的接收函数：延迟分配时先过滤 bootstrap 消息；开启角色切换时改成带超时的 poll。 |
| `_ensure_prefill_recompute_executor` | 1294 | 懒创建 PD 重新引导（retract rebootstrap）用的线程池。 |
| `_init_decode_handshake_state` | 1316 | 初始化 decode 握手所需的线程池、拓扑 future、epoch、房间 generation、setup 任务表和积压队列。 |
| `parallel_info_epoch` | 1350 | 返回某 prefill 地址当前的 epoch（被剔除时递增）。 |
| `register_receiver_room` | 1354 | 新接收方成为房间所有者：记录 generation、初始化响应追踪、置 Bootstrapping。 |
| `release_receiver_room` | 1365 | generation 仍匹配时清理房间的所有 decode 侧状态。 |
| `submit_dp_rank_query` | 1378 | 在线程池里批量查询房间对应的 prefill DP rank。 |
| `get_or_start_key_setup` | 1387 | 某 `bootstrap_key` 已缓存就直接返回地址列表，否则返回（或新建）共享的 setup future。 |
| `_pump_setup_backlog` | 1417 | 在每个 prefill 的并发上限内，把积压的 setup 任务提交到线程池。 |
| `_run_setup_job` | 1434 | 执行一个 setup 任务，结果写入 future，然后继续调度积压任务。 |
| `publish_bootstrap_infos` | 1451 | epoch 未变时把注册成功的地址列表写入 `connection_pool`。 |

**PD 重新引导（retraction rebootstrap）**

| 名称 | 行号 | 作用 |
|---|---|---|
| `_get_prefill_recompute_session` | 1465 | 线程本地的 HTTP session。 |
| `_resolve_rebootstrap_prefill_url` | 1474 | 用 bootstrap 地址的 host 加 prefill 自报的 HTTP 端口拼出 `/generate` 的 base URL。 |
| `submit_prefill_recompute` | 1492 | 只由 attn TP/CP/PP 都为 0 的 decode rank 向原 prefill 发 `/generate`，让它用当前权重重算被撤回请求的前缀 KV。 |
| `_fail_prefill_recompute` | 1544 | 重算失败时 abort 接收方并写入描述性原因；传输已成功的房间不再改动。 |
| `_run_prefill_recompute` | 1569 | 实际发 POST，HTTP 错误或异常时走失败路径。 |

**decode 侧拓扑获取与 rank 映射**

| 名称 | 行号 | 作用 |
|---|---|---|
| `_fetch_parallel_info_payload` | 1599 | 静态方法，在线程池里 GET `/route?...=-1&include_dcp_page_support=1`，返回原始 JSON。 |
| `has_parallel_info` | 1621 | 是否已缓存某 prefill 的拓扑。 |
| `try_ensure_parallel_info` | 1625 | 非阻塞推进拓扑获取；拿到后校验 page_size、dtype、layout、DSpark、MiniMax、MXFP8、DCP 约束（DCP>1 要求 prefill CP=1 且无 KV sharding，page 布局要求 `supports_dcp_page`），再调用 `_resolve_rank_mapping` 并缓存。 |
| `_resolve_rank_mapping` | 1791 | 计算 `target_tp_rank(s)`、`target_cp_ranks`、`target_pp_ranks`、`required_dst_info_num`、`required_prefill_response_num`，写回 `PrefillServerInfo`。断言 decode CP=1。 |

**prefill 侧注册**

| 名称 | 行号 | 作用 |
|---|---|---|
| `_sync_bootstrap_port_across_nodes` | 1891 | 多节点 prefill 时把 world rank 0 的 bootstrap 端口广播给所有 rank。 |
| `register_to_bootstrap` | 1921 | 组装本 rank 的拓扑 payload 并 PUT 到 `/route`；CP-TP group sharing 下把 CP 维度当 TP 维度注册（1940–1942）；`supports_dcp_page` 只在 Mooncake 且本地 DCP=1 时为 True；Rust server + DP 时改为 all-gather 后写本地注册表。 |
| `_register_topology_row` | 1999 | 带指数退避重试（最多 5 次）的 HTTP PUT。 |

**ZMQ 发送**

| 名称 | 行号 | 作用 |
|---|---|---|
| `_connect` | 2028 | 按端点缓存 PUSH socket，并用 monitor socket 检测断连后重建。 |
| `_send_multipart_locked` | 2073 | 每个端点一把锁，串行发送多帧消息。 |

**层指针与 PP 映射**

| 名称 | 行号 | 作用 |
|---|---|---|
| `get_mha_kv_ptrs_with_pp` | 2082 | MHA 的 K/V 指针拆分，并把 prefill 本阶段的层对应到 decode 的层范围（含 decode 有草稿 KV 的情况）。 |
| `_get_mha_head_slice_blocks` | 2115 | SWA 等 MHA 状态在两侧 attn TP 不同时，按 token 和头计算每块的源/目的地址；PP>1 或非 NHD 布局直接报错。 |
| `get_mla_kv_ptrs_with_pp` | 2217 | MLA 指针的 PP 切片：长度相同直接返回；压缩 MLA 走 `_mla_slice_ptrs_for_pp`；普通 MLA 取 `[start_layer, start_layer+n)` 并追加草稿 entry。 |
| `_mla_kv_entry_span_with_pp` | 2254 | 返回本 PP 阶段在全模型 entry 列表里的 `[start, end)`。 |
| `_mla_slice_ptrs_for_pp` | 2261 | DeepSeek-V4 压缩 MLA（C4/C128、SWA、indexer）的分组 PP 切片，并检测两侧单池/双池布局不一致。 |

**心跳与节点失效**

| 名称 | 行号 | 作用 |
|---|---|---|
| `_start_heartbeat_checker_thread` | 2362 | 启动 decode 心跳线程，定期 GET 每个 prefill 的 `/health`，连续失败达到阈值时调用 `_handle_node_failure`。 |
| `_on_heartbeat_success` | 2420 | 心跳成功钩子，默认空。 |
| `_handle_node_failure` | 2424 | 清理该 prefill 的连接池条目和拓扑缓存，递增 epoch，把它负责的未完成房间全部置 Failed，并断开过期 socket。 |

### 3.4 `CommonKVSender`（2476 行）

| 名称 | 行号 | 作用 |
|---|---|---|
| `__init__` | 2477 | 初始化指标和 trace；延迟分配时打开 `DeferredBootstrap` 房间；`is_dummy_cp_rank` 的 rank 直接置 `WaitingForInput` 并返回；多 DP 时登记或校验 prefill DP rank；记录 `init_time` 作为 bootstrap 超时起点。 |
| `_register_prefill_dp_rank` | 2553 | POST `/register_dp_rank`，把房间与本 rank 的 DP rank 关联。 |
| `init` | 2569 | 记录本请求总页数和 aux 索引。 |
| `mark_prefill_complete` | 2576 | 延迟分配模式下标记 prefill 前向完成，通知 decode 可以分配。 |
| `pop_decode_prefix_len` | 2587 | 取出并删除 decode 报告的前缀长度。 |
| `get_max_transfer_tokens` | 2590 | 有需要 DCP 行重排的真实目标时，返回 pack 缓冲区可容纳的最大 token 数，用于限制分块大小。 |
| `should_send_kv_chunk` | 2605 | 页数 > 0 或最后一块时才发送。 |
| `get_transfer_metric` | 2608 | 用 KV 和状态字节数乘复制因子得到总传输字节。 |
| `_record_transfer_indices` | 2616 | 累计已提交的 KV 页数和各状态组件的字节数。 |
| `_prepare_send_indices` | 2633 | 推进 `curr_idx`、判断是否最后一块；KV sharding 时只留本 shard 的页；所有 CP rank 发送时按 CP rank 过滤页；dummy CP rank 跳过，最后一块直接置 Success。 |
| `send` | 2677 | 预处理页号后调用 `kv_mgr.add_transfer_request`（子类实现），最后一块才附带 aux 和状态页号。 |
| `poll` | 2717 | 用 `check_status` 轮询。 |
| `poll_pp_consensus` | 2720 | 用 PP 共识状态轮询。 |
| `_poll_with_status` | 2723 | 统一轮询逻辑：staging 块未传完时把 Success 报成 Transferring；检查 bootstrap 超时；终态时记录延迟和结束 trace。 |
| `_check_bootstrap_timeout` | 2759 | Bootstrapping 超过 `bootstrap_timeout` 时记录失败并置 Failed。 |
| `clear` | 2783 | 清理房间在 manager 中的状态、尝试发出排空 ACK、清理 staging 预取记录；延迟分配下在这里发布最终状态。 |
| `failure_exception` | 2820 | 清理后抛出带失败原因的 `KVTransferError`。 |
| `abort` | 2832 | 以 "Aborted by AbortReq." 结束房间。 |

### 3.5 `CommonKVReceiver`（2844 行）

类属性 `_ctx`、`_socket_cache`、`_socket_locks`、`_global_lock`（2849–2853）是所有接收方共享的 ZMQ PUSH socket 缓存。

| 名称 | 行号 | 作用 |
|---|---|---|
| `__init__` | 2855 | 初始化握手字段和截止时间；延迟分配下重复房间直接失败；需要排空且房间仍被旧接收方持有时本接收方失败；否则调用 `register_receiver_room` 成为房间所有者。 |
| `init` | 2921 | 读取缓存的 `PrefillServerInfo` 和 rank 映射，校验分配策略，判断是否需要 staging，为每个目标 CP rank 生成 setup 部件并推进一次。 |
| `_resolve_setup_part` | 2981 | 拼出 `bootstrap_key = addr_dp_cp_tp`，从 manager 拿缓存或共享 setup future。 |
| `_fetch_and_register_key` | 2999 | 在线程池里对每个目标 TP rank、每个目标 PP rank 查询地址，MLA 时把非选中 TP rank 标成 dummy，然后发送注册消息并发布到连接池。 |
| `_advance_setup` | 3057 | 所有部件完成后汇总 `bootstrap_infos`；`early` 策略置 WaitingForInput，`prefill_complete` 策略向每个 prefill 发 BOOTSTRAP。 |
| `_owns_room_locked` | 3104 | 当前 generation 是否仍是房间所有者。 |
| `_setup_is_current_locked` | 3110 | setup 未取消、仍是所有者、epoch 未变且房间仍为 Bootstrapping。 |
| `_fail_owned_room_locked` | 3120 | 仍是所有者时记录原因并置 Failed（保留最早的原因）。 |
| `_fail_setup` | 3130 | 握手失败处理，可选择让缓存的地址失效。 |
| `invalidate_cached_bootstrap_infos` | 3138 | 从 `connection_pool` 删除本接收方使用过的条目。 |
| `_get_bootstrap_info_from_server` | 3145 | GET `/route` 查询某 DP/CP/TP/PP 的 prefill rank 地址，并附上 `pp_rank`。 |
| `query_prefill_dp_ranks` | 3168 | 静态方法，POST `/query_dp_ranks` 批量查询房间的 prefill DP rank。 |
| `_connect` | 3191 | 类方法，按端点缓存 PUSH socket（带发送超时）。 |
| `disconnect_endpoint` | 3209 | 类方法，关闭并移除一个端点的 socket。 |
| `close_all_sockets` | 3222 | 类方法，角色切换时关闭所有缓存 socket，保留 `_ctx`。 |
| `_connect_to_bootstrap_server` | 3247 | 根据 `rank_ip/rank_port` 取 socket 和锁。 |
| `_register_kv_args` | 3254 | 向一组 prefill rank 发送注册消息。 |
| `_send_bootstrap_message` | 3268 | 发送一条控制消息；`send_metadata` 失败时记录失败并置 Failed，注册失败只返回 False。 |
| `send_metadata` | 3299 | 对每个 prefill rank 编码并发送目标页号等元数据，dummy rank 发空页号；记录 `init_time`。 |
| `poll` | 3348 | 加生命周期锁后调用 `_poll`。 |
| `_poll` | 3352 | 终态直接返回；Bootstrapping 时推进握手或检查 prefill_complete 等待超时；WaitingForInput 时检查传输等待超时。 |
| `_poll_bootstrapping` | 3379 | 推进握手，超过 decode bootstrap 截止时间则 abort 房间。 |
| `_check_waiting_timeout` | 3404 | `send_metadata` 后超过 `waiting_timeout` 则 abort 房间。 |
| `_abort_room` | 3421 | 仍是所有者时置 Failed，并按后端是否需要排空选择 `retry_abort` 或 `ensure_abort_notified`。 |
| `clear` | 3439 | 加锁后调用 `_clear`。 |
| `_clear` | 3443 | 必要时补发 ABORT，固定 `conclude_state`，清理延迟 ACK 状态并释放房间。 |
| `failure_exception` | 3483 | 抛出带原因的 `KVTransferError`；未拥有房间的接收方用本地原因。 |
| `abort` | 3498 | 以 "Aborted by AbortReq." abort 房间。 |
| `ensure_abort_notified` | 3505 | 只通知一次 ABORT。 |
| `retry_abort` | 3510 | 再发一次 ABORT，返回是否所有目标都接受。 |
| `_try_send_cached_abort` | 3522 | 类方法，非阻塞地在共享 socket 上发 ABORT（必要时新建 socket），拿不到锁就返回 False。 |
| `_send_abort_notification` | 3577 | 先为本接收方武装 ACK generation，再向所有 prefill rank 发 ABORT；需要排空的后端用非阻塞方式并校验 epoch。 |

### 3.6 `CommonKVBootstrapServer`（3671 行）

| 名称 | 行号 | 作用 |
|---|---|---|
| `__init__` | 3672 | 初始化拓扑字段、`prefill_port_table[dp][cp][tp][pp]`、`room_to_dp_rank`，注册路由并在后台线程启动服务。 |
| `run` | 3712 | 启动服务线程。 |
| `_is_ready` | 3715 | 已注册 rank 数 ≥ `dp*cp*tp*pp` 时就绪。 |
| `_setup_routes` | 3729 | 注册 `/route`、`/register_dp_rank`、`/query_dp_ranks`、`/health`。 |
| `_handle_health_check` | 3735 | 返回 200 OK。 |
| `_handle_route` | 3738 | 按 HTTP 方法分发到 PUT/GET 处理函数。 |
| `_handle_route_put` | 3749 | 处理 prefill rank 注册：第一次注册时固定全局拓扑字段，校验 DSpark 布局和分配策略一致，把地址写进四级表。 |
| `_handle_route_get` | 3855 | 全 -1 参数时返回 `PrefillServerInfo`（按 `include_dcp_page_support` 决定是否带 `supports_dcp_page`）；否则返回指定 rank 的地址，未就绪 503，找不到 404。 |
| `_handle_register_dp_rank` | 3933 | 记录房间对应的 DP rank 和时间戳。 |
| `_handle_query_dp_ranks` | 3945 | 批量返回房间的 DP rank。 |
| `_cleanup_expired_entries` | 3956 | 后台协程，定期删除过期的 `room_to_dp_rank` 条目。 |
| `_run_server` | 3974 | 在新事件循环里启动 aiohttp 服务和清理任务，退出时清理。 |
| `close` | 4006 | 停止事件循环并等待线程退出。 |
| `poll` | 4016 | 占位方法，满足基类接口。 |

## 4. 与 PCP+DCP 方案 A 的关系

计划 §4.10 对本文件的要求是：在 `440-499` 行新增"prefill 是 CP8+DCP8"的身份，按 DCP rank 发送本卡存的那份 KV，修改 `is_dummy_cp_rank`、`enable_all_cp_ranks_for_transfer` 和 bootstrap 注册。对照代码，现有逻辑已经覆盖了大部分，需要改的地方比计划写的少，但有几处计划没有提到的约束。

### 4.1 已经满足、可能不需要改的部分（推断）

- **身份注册已经是"CP rank r = TP rank r"。** K3.1 的 PCP 只能走 CP-TP group sharing（计划 §3.3 第 1 条）。这种情况下 `enable_all_cp_ranks_for_transfer`（444 行）和 `is_dummy_cp_rank`（494 行）都已经是 False，每个 CP rank 都是真实发送方；`register_to_bootstrap`（1940–1942 行）已经把 CP 维度注册成 TP 维度。方案 A 下 `dcp_rank = attn_cp_rank`（计划 §4.2），所以 prefill CP rank r 同时也是 DCP rank r，正好对应 decode 的 TP rank r / DCP rank r。这两个判断本身不需要改。
- **DCP rank 校验会自动生效。** 两侧 DCP 都是 8 时，`requires_dcp_relayout`（591 行）走"同大小"分支：要求 `self.dcp_rank == dst_dcp_rank`，返回 False（不重排，按页传输）。只要 §4.2 把 prefill 的 `dcp_rank` 改成 `attn_cp_rank`，而 decode 的 `target_tp_rank = engine_rank % attn_tp_size`（1812 行）把 decode rank r 配到注册为 TP rank r 的 prefill CP rank r，这个校验就成立。
- **decode 侧的 DCP 前置检查能通过。** `try_ensure_parallel_info` 要求 decode DCP>1 时 `info.attn_cp_size == 1`（1758 行）。由于 CP-TP sharing 下 prefill 注册的 CP 是 1，这个检查能通过。但它依赖注册时的维度交换，见 4.3 第 1 条。
- **KDA 状态按头 r 对 r。** `_should_skip_cp_replicated_state_transfer`（573 行）在 CP-TP sharing 下返回 False，所以每个 CP rank 都发送自己的状态。KDA 头按整个 TP 组切分，CP rank r 持有第 r 份头，配对 decode TP rank r，这是现有行为。

### 4.2 需要改或需要确认的函数

| 位置 | 内容 |
|---|---|
| `register_to_bootstrap`（1921） | 目前 payload 里没有 prefill 的 `dcp_size`/`dcp_kv_layout`。建议加上，让 decode 在 `try_ensure_parallel_info` 阶段就能判断拓扑是否兼容，而不是等到 prefill 收到注册消息时在 `requires_dcp_relayout` 里报错（计划未提及）。 |
| `PrefillServerInfo`（168）和 `_handle_route_put`/`_handle_route_get`（3749/3855） | 若新增 `dcp_size` 字段，要同步到数据类、服务端存储和 GET 返回；同时照 `supports_dcp_page` 的做法兼容旧版 decode（旧 decode 会拒绝未知字段，见 3906–3908 行）（计划未提及）。 |
| `try_ensure_parallel_info`（1753–1767） | decode DCP>1 时的检查只看 prefill `attn_cp_size`、KV sharding，不看 prefill DCP。方案 A 下应该补一条：prefill DCP>1 时要求 `prefill_dcp == decode_dcp` 且 `dcp_kv_layout` 都是 `token`（计划未提及）。 |
| `requires_dcp_relayout`（591） | 方案 A prefill（DCP8）对接 decode DCP1（普通 TP8 decode）或 DCP≠8 的 decode 时会抛 `Unsupported PD DCP topology`。也就是说，方案 A 的 prefill 只能对接 DCP 完全相同的 decode。如果要支持 DCP8→DCP1，需要新增"从 8 个 prefill rank 各取一部分行拼成完整页"的反向重排，这是新功能（计划未提及）。 |
| `register_to_bootstrap` 的 `supports_dcp_page`（1964–1967） | 只在 prefill DCP=1 时为 True。方案 A prefill 是 DCP8，所以 decode 用 `--dcp-kv-layout page` 时会在 1769 行被拒绝。这与计划"不支持 page 布局"一致，但应当在参数校验里给出更明确的报错。 |
| `_prefill_unique_rank`（1180） | 用真实的 `attn_tp_rank`（方案 A 下为 0）和 `attn_cp_rank` 计算，8 个 CP rank 仍然各不相同，可以不改（推断）。 |

### 4.3 计划可能遗漏的点

1. **方案 A 依赖 CP-TP group sharing 的维度交换。** decode DCP 检查（1758 行）和 rank 配对都依赖 prefill 注册时把 CP 当成 TP。如果以后把方案 A 推广到不开 CP-TP sharing 的模型，prefill 会注册 `attn_cp_size=8`，decode DCP 检查会直接拒绝，`_resolve_rank_mapping` 也会去拉所有 CP rank（1861–1872 行）。计划 §4.10 应当写明方案 A 的 PD 只支持 CP-TP sharing（计划未提及）。
2. **decode 侧不能是方案 A 实例。** `_resolve_rank_mapping` 断言 decode `attn_cp_size == 1`（1852 行）。如果有人把开了 PCP+DCP 的统一实例当 decode 用，会直接断言失败。需要在参数校验里拒绝"decode 模式 + enable-prefill-cp"，或者放宽断言（计划未提及）。
3. **页对齐和前缀长度。** 同 DCP 按页传输时，prefill 和 decode 的页号都是同一套"虚拟页"（`page_size × dcp` 个 token）的编号（推断）。`try_ensure_parallel_info` 只比较 `page_size`（1662 行），没有比较 DCP 放大后的页大小。两侧 DCP 不同时 page_size 相同也会被放行，直到注册阶段才报错，见 4.2。
4. **`prefill_complete` 分配策略。** `CommonKVReceiver.init` 要求 prefill 的 TP/CP/PP 都是 1（2945–2948 行）。CP-TP sharing 注册成 TP8，所以方案 A 下不能用 `prefill_complete`。这是现有限制，计划没有列出。
5. **`pp_size > 1`。** 计划把 PP 列为非目标。本文件里 PP 相关逻辑（`_resolve_rank_mapping` 的 PP 映射、`get_mla_kv_ptrs_with_pp`、`check_status_pp_consensus`）与 DCP 无交叉。但 `_get_mha_head_slice_blocks` 在 PP>1 时直接报错，Mooncake 的 `_resolve_draft_peer_ranks` 要求 `pp_size == 1` 且 `attn_cp_size == 1`（见 Mooncake 文档）。如果以后做 PP，需要一起检查。
