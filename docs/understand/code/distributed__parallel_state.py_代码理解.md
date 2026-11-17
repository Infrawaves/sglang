# `python/sglang/srt/distributed/parallel_state.py` 代码理解

- 文件路径：`python/sglang/srt/distributed/parallel_state.py`
- 行数：3550
- 所属模块：分布式通信层。负责建立 WORLD/TP/PP/attention TP/attention CP/DCP/MoE 等进程组，并封装组内的各种集合通信。

## 1. 整体功能

这个文件改编自 vLLM 和 Megatron-LM 的 `parallel_state.py`，接管 PyTorch 的分布式环境。典型流程写在文件头注释里（第 9-24 行）：先调用 `init_distributed_environment` 建 `torch.distributed` 默认组和 WORLD 组，再调用 `initialize_model_parallel` 建所有模型并行子组，进程退出时调用 `destroy_model_parallel` 和 `destroy_distributed_environment` 拆除。

核心概念是 `GroupCoordinator`（第 260 行）。它包装一组 rank 的两个 `ProcessGroup`：`device_group`（NCCL/HCCL 等，用于 GPU 张量）和 `cpu_group`（gloo，用于对象和 CPU 张量）。它按张量大小、是否在 CUDA graph 捕获中、是否在 `torch.compile` 中，把 all-reduce、all-gather、reduce-scatter、all-to-all 等操作分派给 custom allreduce、quick allreduce、PyNccl、MSCCL++、Torch SymmMem、FlashInfer、PCIe-IPC、aiter 等不同实现。为了让 Dynamo 把集合通信当作不透明调用，文件把几个入口注册成 custom op（第 190-257 行），op 内部用组名从 `_groups` 弱引用表里查回 `GroupCoordinator`。

各组以模块级全局变量保存（`_WORLD`、`_TP`、`_ATTN_TP`、`_SHARED_EXPERTS_TP`、`_ATTN_CP`、`_DCP`、`_MOE_DP`、`_MOE_EP`、`_MOE_TP`、`_PP`、`_SELF_PP`，第 2153、2229-2233、2260-2262、2280-2281 行）。`initialize_model_parallel` 建完组后，用 `get_parallel().override_permanently(...)` 把组对象写进 `runtime_context` 的 parallel 上下文（第 2986-2999 行）。大多数 `get_*_group()` 读的是 `get_parallel()` 上的字段，而不是全局变量；只有 `get_dcp_group_no_assert` 和 `get_self_pp_group` 直接读全局变量。

文件末尾（第 3477-3541 行）给一批旧的 getter 套上 `_warn_if_called_from_outside`：从 `sglang.srt.distributed.` 包外调用时，第一次会发 `DeprecationWarning`，提示改读 `get_parallel().<字段>`。

rank 的布局约定是 `[PP, attention DP, CP, attention TP]`，attention TP 变化最快（见第 2833 行注释和 `runtime_context.derive_attention_ranks`）。TP 组是连续的 rank 段；attention CP 组在每个 TP 组内以 `attn_tp_size` 为步长取 rank；DCP 组是 TP 组内长度为 `dcp_size` 的连续切片。

## 2. 主要依赖与调用方

依赖：
- `sglang.srt.runtime_context`：`get_parallel()` 提供拓扑宽度（`tp_size`、`attn_cp_size`、`attn_dcp_size` 等）并接收建好的组；`derive_parallel_widths` 推导 `attn_tp_size`、`moe_tp_size`、`attn_dcp_size`。
- `sglang.srt.distributed.device_communicators.*`：`PyNcclCommunicator`、`dispatch_custom_allreduce`、`QuickAllReduce`、`PyMscclppCommunicator`、`TorchSymmMemCommunicator`、`PcieIpcCommunicator`、`HpuCommunicator`、`XpuCommunicator`、`MessageQueue`（`shm_broadcast`）。
- `sglang.srt.distributed.utils`：`all_gather_single`、`reduce_scatter_single`、`set_global_tcp_store`。
- `sglang.srt.utils.custom_op.register_custom_op`、`compilation_config.register_split_op`。
- `sglang.srt.layers.flashinfer_comm_fusion`（FlashInfer allreduce，懒加载）。
- `sglang.srt.environ.envs`（`SGLANG_USE_MESSAGE_QUEUE_BROADCASTER`、`SGLANG_ENABLE_DETERMINISTIC_INFERENCE`、`SGLANG_K3_SP_COLLECTIVE` 等）。

调用方（grep 确认）：
- `distributed/bootstrap.py:320-338`（`_init_parallel_groups`）：依次调用 `init_distributed_environment`、`initialize_model_parallel`、`_tag_groups_for_flashinfer_allreduce_only`，之后 `initialize_dp_attention`。这是推理进程的主入口。
- `disaggregation/encoder/server.py:591/618`、`weight_cache/daemon.py:225/234`：其它进程类型的初始化。
- DCP 组的使用方通过 `get_parallel().dcp_group` 读取（共约 25 处），例如 `layers/dcp/comm.py:176/180/289`、`layers/attention/dsa/dsa_dcp.py:138/147`、`layers/attention/dsa/dsa_dcp_sharded.py:171/285/479`、`layers/attention/triton_backend.py:2023/2432`、`layers/attention/xpu_backend.py:261`、`hardware_backend/npu/attention/dcp.py:54/60`。
- `dcp_rank` 并不是在这个文件里决定的：`runtime_context.py:1610-1612` 在 publish 阶段把 `dcp_rank` 设为 `tp_rank % dcp_size`，`attn_dcp_rank` 等于它；DCP 关闭时 `attn_dcp_rank=0`（`runtime_context.py:1585-1586`）。`attn_cp_rank` 在 `runtime_context.py:126` 由 `(tp_rank // attn_tp_size) % attn_cp_size` 得出。

## 3. 类与函数说明

### 3.1 模块级辅助函数与数据类

| 名称 | 行号 | 作用 |
|---|---|---|
| `get_torch_distributed_pg_options` | 110 | 只在 NPU 上返回 HCCL 的 `ProcessGroupHCCL.Options`（设置 `hccl_buffer_size`，取 `DEEPEP_HCCL_BUFFSIZE`/`HCCL_BUFFSIZE`，默认 200）；并且只对默认组或组名含 `moe` 的组生效，其它情况返回 `None`。 |
| `GraphCaptureContext` | 128-132 | `@dataclass`，只含 CUDA graph 捕获所用的 `stream`。类型标注用 `torch.Stream`，避免 import 时调用 `torch.get_device_module()` 导致 fork 不安全。 |
| `P2PWork` | 135-138 | `@dataclass`，保存异步点对点发送的 `work` 句柄和 `payload` 张量，防止张量在发送完成前被回收。 |
| `_split_tensor_dict` | 141 | 把字典拆成元数据列表（张量替换成 `TensorMetadata(device 类型, dtype, size)`）和张量列表，供 `broadcast/send/recv_tensor_dict` 使用。只记录设备类型，不记录设备号。 |
| `_get_unique_name` | 170 | 给组名加计数后缀生成唯一名（`tp` → `tp:0`、`tp:1`），作为 custom op 查找组的键。 |
| `_register_group` | 186 | 把 `GroupCoordinator` 以弱引用登记到 `_groups[unique_name]`。 |
| `inplace_all_reduce` | 192 | custom op + split op：按组名查组，调用 `_all_reduce_in_place`。组不存在时断言失败，组已销毁时抛 `ValueError`。 |
| `outplace_all_reduce` | 201 | custom op：按组名查组，调用 `_all_reduce_out_place(tensor, method)` 返回新张量。 |
| `flashinfer_allreduce` | 212 | custom op：调用组的 `_flashinfer_allreduce`。是否可用由调用前的 `_can_use_flashinfer_allreduce` 判断，op 本身没有回退。 |
| `reg_all_gather_into_tensor` | 228 | custom op：调用组的 `_all_gather_into_tensor`，让 all-gather 对 Dynamo 不透明。 |
| `reg_reduce_scatter_tensor` | 239 | custom op：调用组的 `_reduce_scatter_tensor`。 |
| `reg_all_to_all_single` | 250 | custom op：调用组的 `_all_to_all_single`。 |

### 3.2 `GroupCoordinator`（第 260 行）

**构造与基础属性**

| 名称 | 行号 | 作用 |
|---|---|---|
| `__init__` | 301 | 对 `group_ranks` 里的每个 rank 列表都调用 `torch.distributed.new_group` 建 device 组和 gloo cpu 组（`new_group` 是集合操作，所有 rank 都要参与每个组的创建），只保留本 rank 所在的那组，并记录 `ranks`、`world_size`、`rank_in_group`。mooncake 后端会构造 `MooncakeBackendOptions` 和 `active_ranks`。`rank_offset > 0` 时把组内 rank 平移到全局 rank 空间（扩容加入的 rank）。随后按开关创建 PyNccl、MSCCL++、PCIe-IPC、custom allreduce、quick allreduce（仅 HIP）、Torch SymmMem、HPU/XPU 通信器和共享内存 `MessageQueue` 广播器；custom allreduce 初始化失败只打 warning。 |
| `__repr__` | 579 | 输出 ranks、rank、local_rank、组对象、唯一名、world_size、rank_in_group。 |
| `first_rank`（property） | 587 | 组内第一个进程的全局 rank。 |
| `last_rank`（property） | 592 | 组内最后一个进程的全局 rank。 |
| `is_first_rank`（property） | 597 | 当前进程是否是组内第一个。 |
| `is_last_rank`（property） | 602 | 当前进程是否是组内最后一个。 |
| `next_rank`（property） | 607 | 环上下一个进程的全局 rank（PP 用）。 |
| `prev_rank`（property） | 614 | 环上上一个进程的全局 rank。 |
| `get_available_pynccl_comm` | 620 | 只在 PyNccl 通信器存在且 `available` 时返回它，否则返回 `None`。 |
| `require_pynccl_comm` | 627 | 同上，但不可用时抛 `RuntimeError`，用于 `all_gatherv`、`reduce_scatterv` 这类只能走 PyNccl 的操作。 |
| `graph_capture` | 637 | 上下文管理器：准备捕获 stream（先 `wait_stream` 当前流），进入 custom allreduce 的 `capture()`，并在捕获期间启用 PyNccl 和 MSCCL++（它们在 eager 模式下默认关闭）。 |

**all-reduce 系列**

| 名称 | 行号 | 作用 |
|---|---|---|
| `all_reduce` | 699 | 对外入口。world_size 为 1 时直接返回；CPU 张量走 shm allreduce 或 `torch.distributed`；HPU/XPU 走各自通信器（XPU 经 `inplace_all_reduce` 保持对 Dynamo 不透明）；`torch.compile` 中先试 FlashInfer，其次没有加速通信器时用 inplace op，否则用 `outplace_all_reduce(method="auto")` 把选择推迟到运行时；eager 下对称内存 + PyNccl 时原地 all-reduce，然后依次尝试 FlashInfer 和 `_resolve_outplace_all_reduce_method`，都不适用时走 inplace。 |
| `quant_all_reduce` | 809 | 仅 NPU：先用 `npu_dynamic_quant` 量化成 int8，all-gather 量化值和 scale，再在全精度下求和。没有 `torch_npu` 时退回普通 inplace all-reduce。 |
| `fused_allreduce_rmsnorm` | 838 | ROCm 专用：尝试用 custom allreduce 通信器做融合的 all-reduce + RMSNorm；根据 token 数和读量（`_FUSED_AR_RMS_1STAGE_MAX_TOKENS`、`_FUSED_AR_RMS_1STAGE_MAX_READ_BYTES`）或环境变量选 1-stage。不可用时返回 `None` 让调用方回退。 |
| `fused_allreduce_rmsnorm_quant_per_group` | 898 | ROCm gfx95 专用：融合 all-reduce + RMSNorm + 按组 FP8 量化。对 hidden 维、总字节数、world_size=6 等做前置检查，任何不满足或异常都返回 `None`。 |
| `_resolve_outplace_all_reduce_method` | 975 | 按优先级选择 out-of-place 方法名：`ca` → `pcie_ipc` → `qr` → `pymscclpp` → `torch_symm_mem` →（piecewise CUDA graph 中）`pynccl`；都不满足返回 `None`。 |
| `_can_use_flashinfer_allreduce` | 1024 | 组被打上 `_fi_workspace_hint` 时，调用 `flashinfer_comm_fusion.can_use_flashinfer_allreduce` 判断，校验 world_size 和组键与 workspace 一致。 |
| `_flashinfer_allreduce` | 1038 | 调用 FlashInfer allreduce 实现，按 hint 选 attention TP 或 MoE workspace。 |
| `_all_reduce_out_place` | 1047 | 按方法名执行 out-of-place all-reduce。`auto` 时现场解析；解析为 `pymscclpp` 时先 clone（它实际原地写）；解析为 `None` 时优先强制用 PyNccl，否则 clone 后原地 all-reduce。 |
| `_all_reduce_in_place` | 1094 | 原地 all-reduce：PyNccl → Torch SymmMem → `torch.distributed.all_reduce`。 |

**reduce-scatter / all-to-all**

| 名称 | 行号 | 作用 |
|---|---|---|
| `reduce_scatter_along_dim` | 1108 | 沿任意维 reduce-scatter：把目标维移到第 0 维并 contiguous，要求能被 world_size 整除，结果再移回原维。 |
| `_reduce_scatter_tensor` | 1145 | 底层 reduce-scatter：PyNccl（含对称内存检查）或 `reduce_scatter_single`。 |
| `reduce_scatter_tensor` | 1163 | 对外入口。确定性推理模式下用 all-reduce + 切片保证归约顺序与接收 rank 无关；CPU 直接走底层；ROCm 先试 aiter 自定义实现；否则走 `reg_reduce_scatter_tensor` custom op。 |
| `_has_aiter_custom_reduce_scatter` | 1182 | 检查 custom allreduce 通信器是否带 aiter 的 `reduce_scatter` 能力。 |
| `_maybe_aiter_reduce_scatter` | 1191 | ROCm 且 `SGLANG_DP_USE_REDUCE_SCATTER` 开启时，用 aiter 做等分块 reduce-scatter；按是否在 CUDA graph 捕获中选择 registered/unregistered 缓冲；真实 graph warmup 时只把输出清零。返回是否已处理。 |
| `_all_to_all_single` | 1232 | 底层 all-to-all：优先 PyNccl（保持可被 CUDA graph 捕获，注释说明是给 DCP a2a 后端用的），否则 `torch.distributed.all_to_all_single`。 |
| `all_to_all_single` | 1240 | 对外入口：world_size 为 1 时直接拷贝，否则调用 `reg_all_to_all_single`。 |
| `reduce_scatter` | 1246 | 列表形式的 reduce-scatter，直接调 `torch.distributed.reduce_scatter`。 |
| `_deterministic_reduce_scatterv` | 1255 | 确定性模式下的变长 reduce-scatter：all-reduce 后按 `sizes` 前缀和切出本 rank 的片段。 |
| `reduce_scatterv` | 1282 | 变长 reduce-scatter（每 rank 行数可不同），必须有 PyNccl；确定性模式转给上面的函数。 |

**all-gather / gather / broadcast**

| 名称 | 行号 | 作用 |
|---|---|---|
| `_all_gather_into_tensor` | 1314 | 底层 all-gather：ROCm 上先试 aiter custom all-gather（`SGLANG_USE_AITER_AG`），其次 MSCCL++，再次 PyNccl，最后 `all_gather_single`。 |
| `_has_aiter_custom_all_gather` | 1370 | 检查 aiter custom all-gather 能力；确定性集合通信开启时返回 False。 |
| `_deterministic_collectives_enabled`（staticmethod） | 1383 | 有 `SGLANG_USE_1STAGE_ALLREDUCE` 时取它的值，否则取 `SGLANG_ENABLE_DETERMINISTIC_INFERENCE`。 |
| `all_gather_into_tensor` | 1388 | 对外入口：CPU 直接调底层，其它设备经 `reg_all_gather_into_tensor` custom op（避免 Dynamo 改写破坏 XPU graph 捕获）。 |
| `all_gather` | 1399 | 沿任意维 all-gather，按拼接方式分配输出（stack 方式与 `torch.compile` 不兼容）；支持传 `output_tensor_list` 走 `torch.distributed.all_gather`；HPU 和 CPU shm 有专门路径。 |
| `all_gatherv` | 1466 | 变长 all-gather，支持单张量或张量列表、可传预分配 `output`（仅单张量）；用 PyNccl 的 `group_start/group_end` 批量发出；各 rank 大小相同时把 `sizes` 置空以允许对称内存。 |
| `gather` | 1534 | 把各 rank 张量收集到 `dst`（组内 rank）并沿 `dim` 拼接，非 dst 返回 `None`；XPU 走专门实现。 |
| `broadcast` | 1569 | 从组内 `src` 广播张量，优先 PyNccl。 |
| `broadcast_object` | 1589 | 广播 Python 对象；有 `mq_broadcaster` 时走共享内存消息队列（只支持 `src=0`），否则用 cpu 组的 `broadcast_object_list`。 |
| `broadcast_object_list` | 1613 | 在 device 组上广播对象列表（`group` 参数实际未使用）。 |
| `all_gather_object` | 1630 | 在 cpu 组上 all-gather 任意对象。 |

**点对点与张量字典**

| 名称 | 行号 | 作用 |
|---|---|---|
| `send_object` | 1635 | 把对象 pickle 成 uint8 张量，经 cpu 组先发大小再发内容；`async_send` 时返回 `P2PWork` 列表。不允许发给自己。 |
| `recv_object` | 1687 | 与 `send_object` 配对：先 `irecv` 大小，再 `irecv` 内容并反序列化。 |
| `broadcast_tensor_dict` | 1724 | 广播张量字典：元数据经 `broadcast_object`，张量按 CPU/GPU 分别在 cpu 组或 device 组上异步广播；空张量跳过。 |
| `send_tensor_dict` | 1806 | 把张量字典发给 `dst`（默认下一个 rank，PP 用）。传 `all_gather_group` 时只发本 rank 的切片，由接收端再 all-gather。 |
| `recv_tensor_dict` | 1861 | 与上面配对，默认从上一个 rank 接收；按需在 `all_gather_group` 上把切片 all-gather 回完整张量。 |
| `send_recv_tensor_dict` | 1921 | 同时发送和接收张量字典，全部用 `batch_isend_irecv` 一次提交，避免 NPU/HCCL 上 `isend` 阻塞导致的死锁。分四步：交换元数据大小、交换元数据、批量交换张量、按需 all-gather。 |
| `barrier` | 2076 | 在 cpu 组上做 barrier（NCCL barrier 会偷偷建 GPU 张量，易弄错设备）。 |
| `send` | 2085 | 发送张量到 `dst`，优先 PyNccl。 |
| `recv` | 2097 | 按给定 size/dtype 分配张量并从 `src` 接收，优先 PyNccl。 |
| `destroy` | 2113 | 先销毁 PCIe-IPC 通信器（它的 teardown 要在原组上做集合操作），再销毁 device/cpu 组，置空各通信器，MSCCL++ 调 `destroy()`。 |

### 3.3 组的获取、设置和建立

| 名称 | 行号 | 作用 |
|---|---|---|
| `_group_coordinator_device` | 2135 | 根据平台返回本进程设备：CUDA/HIP（`SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS` 时固定 `cuda:0`）、NPU、XPU、MUSA、out-of-tree 平台，否则 CPU。 |
| `get_world_group` | 2156 | 返回 `get_parallel().world_group`。 |
| `init_world_group` | 2160 | 建 WORLD 组，关闭所有加速通信器。 |
| `init_model_parallel_group` | 2183 | 建模型并行子组的统一工厂。未显式指定时，custom allreduce / MSCCL++ / SymmMem 取模块全局开关；PyNccl 在 NPU、XPU、mooncake 后端或平台不支持时关闭；HPU/XPU 通信器总是请求开启。 |
| `get_tp_group` | 2236 | 返回 `get_parallel().tp_group`。第 2278 行别名 `get_tensor_model_parallel_group`。 |
| `get_attn_tp_group` | 2240 | 返回 attention TP 组。 |
| `get_shared_experts_tp_group` | 2244 | 返回共享专家 TP 组。 |
| `get_attn_cp_group` | 2248 | 返回 attention CP 组。 |
| `get_dcp_group_no_assert` | 2252 | 直接返回全局 `_DCP`，DCP 未开启时为 `None`。 |
| `get_dcp_group` | 2256 | 返回 `get_parallel().dcp_group`；DCP 未开启时这个字段没有被写入，读取会报错（第 2984-2985 行注释）。 |
| `get_moe_dp_group` | 2265 | 返回 MoE DP 组。 |
| `get_moe_ep_group` | 2269 | 返回 MoE EP 组。 |
| `get_moe_tp_group` | 2273 | 返回 MoE TP 组。 |
| `get_self_pp_group` | 2284 | 返回每个 rank 单独成组的 PP 组（单层 draft 用），未初始化时断言失败。 |
| `get_pp_group` | 2289 | 返回 PP 组。第 2294 行别名 `get_pipeline_model_parallel_group`。 |
| `get_mooncake_transfer_engine` | 2297 | 懒加载并返回共享的 Mooncake transfer engine（PD 的 mooncake 后端和 mooncake_store 用）。 |
| `graph_capture` | 2310 | 模块级上下文管理器：依次进入 TP 组和 PP 组的 `graph_capture`，再对 `_DCP`、`_ATTN_TP`、`_SHARED_EXPERTS_TP`、`_MOE_EP`、`_MOE_TP` 中按 `id` 去重后未进入过的组进入其 `graph_capture`。 |
| `set_custom_all_reduce` | 2345 | 设置全局 `_ENABLE_CUSTOM_ALL_REDUCE`。 |
| `set_mscclpp` | 2350 | 设置全局 `_ENABLE_MSCCLPP`。 |
| `set_torch_symm_mem_all_reduce` | 2355 | 设置全局 `_ENABLE_TORCH_SYMM_MEM_ALL_REDUCE`。 |
| `set_flashinfer_allreduce_only` | 2360 | 设置全局 `_ENABLE_FLASHINFER_ALLREDUCE_ONLY`。 |
| `_tag_groups_for_flashinfer_allreduce_only` | 2365 | 给拥有 FlashInfer workspace 的组打 `_fi_workspace_hint`：MoE workspace 只打在 `_MOE_EP`（EP>1 时）或 `_MOE_TP` 之一，attention workspace 打在 `_ATTN_TP`；故意不打 `_TP`，因为 DP attention 下 `_TP` 比 workspace 的 peer 多，会静默算错。 |
| `get_default_distributed_backend` | 2396 | 设备是当前平台时返回平台的后端名，否则查 `_DEVICE_TO_DISTRIBUTED_BACKEND`，默认 `gloo`。 |
| `_create_global_tcp_store` | 2407 | 创建全局 `TCPStore`（NIXL 用）。master IP 依次取 `MASTER_ADDR`、`dist_init_addr`，都没有时由 rank 0 广播本机 IP；动态成员模式下不等待 worker。失败只打 warning。 |
| `init_distributed_environment` | 2476 | 如果 `torch.distributed` 未初始化：校验并设置超时（也保存为子组超时 `_MODEL_PARALLEL_GROUP_TIMEOUT`），mooncake 后端构造选项，调用 `init_process_group`，`moe_a2a_backend == "nixl"` 时建 TCPStore。然后推断 `local_rank`，建 `_WORLD`（已存在时校验 world_size），并写入 `get_parallel()`。 |
| `initialize_model_parallel` | 2574 | 按 `get_parallel()` 上的宽度建所有模型并行组，详见 3.4。 |
| `create_custom_parallel_group` | 3002 | 按调用方给出的完整 rank 列表建一个自定义 `ProcessGroup`（默认 gloo）。一般路径用 world 级 `all_gather_object` 收集所有 rank 的配置去重后逐个 `new_group`；NPU 上的标准连续分区直接本地推导，避免在 HCCL 默认组上分配缓冲。所有 rank 必须传相同的完整列表，否则会走不同路径而死锁。 |
| `model_parallel_is_initialized` | 3075 | `_TP` 和 `_PP` 都已建好时返回 True。 |
| `patch_pipeline_parallel_group` | 3085 | 上下文管理器：临时把 `_PP` 和 `get_parallel()` 的 `pp_size/pp_rank/pp_group` 换成给定组（投机解码 draft 用），不可嵌套。 |
| `patch_tensor_parallel_group` | 3111 | 上下文管理器：临时替换 `_TP` 和上下文里的 TP 宽度/rank/组。`owns_attention=True` 时同时把 attention TP 设成该组，attention DP、attention CP、MoE DP/EP 宽度设为 1（`attn_cp_group=None`），并在收窄前算好 DP gather slot。不可嵌套。 |

### 3.4 `initialize_model_parallel`（第 2574-2999 行）详解

**读取宽度（第 2637-2645 行）。** 从 `get_parallel()` 读 `tp_size`、`moe_ep_size`、`pp_size`、`attn_dp_size`、`attn_cp_size`、`moe_dp_size`、`attn_dcp_size`、`shared_experts_tp_size`。这些值在 publish 阶段已由 `runtime_context.parallel_widths_of` 算好，本函数不读 server args。

**前置检查（第 2648-2677 行）。**
- `world_size` 必须等于 `tp_size * pp_size`。扩容加入的 rank（`recovered_rank` 或 `rank_offset > 0`）直接用 `tp*pp` 当 world_size。
- `attn_dcp_size >= 1`；DCP > 1 时平台必须是 HIP、CUDA、NPU 或 XPU。
- `tp_size % dcp_size == 0`。

**TP 组（第 2679-2703 行）。** 每 `tp_size` 个连续 rank 组成一个 TP 组；共 `world_size // tp_size`（即 `pp_size`）个。TP 组可以开共享内存消息队列广播器。

**DCP 组（第 2705-2732 行）。** 只在 `dcp_size > 1` 时建：对每个 TP 组按 `dcp_size` 连续切片，`tp_group[start:start+dcp_size]`。例如 TP8、DCP4 时得到 [0..3] 和 [4..7]；TP8、DCP8 时得到 [0..7]，成员和 TP 组一样，但仍然是一个新的 `GroupCoordinator`（独立的 NCCL 通信器、custom allreduce、PyNccl 等）。DCP 组显式关闭 quick allreduce，因为 DCP 只用 all-gather/reduce-scatter、all-to-all 和 FP32 all-reduce，quick allreduce 只会白占 workspace。DCP 组在 attention CP 组**之前**建，而且建法与 CP 无关。

**推导宽度（第 2734-2747 行）。** 调用 `derive_parallel_widths`，其中 `dcp_enabled=_DCP is not None`，得到 `attn_tp_size = tp_size // attn_dp_size // attn_cp_size` 和 `moe_tp_size = tp_size // moe_ep_size // moe_dp_size`。

**attention CP 组（第 2749-2781 行）。** 如果 `attn_cp_size == tp_size`（第 2753 行），直接复用 `_TP` 对象，不新建组。否则对每个 TP 组、每个 attention DP 副本、每个 `attn_tp_idx`，取 `range(st, en, attn_tp_size)`：同一 DP 副本内、attention TP 下标相同的 rank 构成一个 CP 组。例如 TP8、CP2、DP1 时 `attn_tp_size=4`，CP 组为 [0,4]、[1,5]、[2,6]、[3,7]。

**attention TP 组（第 2785-2822 行）。** `attn_tp_size == tp_size` 时复用 `_TP`；否则每个 (CP, DP) 组合对应一段长度为 `attn_tp_size` 的连续 rank。PyNccl 在 `SYNC_TOKEN_IDS_ACROSS_TP` 或对称内存时开启；`SGLANG_K3_SP_COLLECTIVE` 开启时强制 custom allreduce（K3 的 SP collective 需要）；SymmMem 关闭。

**共享专家 TP 组（第 2824-2852 行）。** 只在 `shared_experts_tp_size > 1` 时建。等于 `attn_tp_size` 时复用 `_ATTN_TP`；否则把 world 按 `shared_size` 连续切片，允许跨 interleave PCP 的 CP rank，但不跨 DP 副本或 PP 段（宽度已在参数校验时保证整除 attention TP × CP）。

**MoE DP 组（第 2858-2885 行）。** `attn_cp_size > moe_dp_size` 时直接复用 `_ATTN_CP`，借用 DP 的 gather/scatter 实现 CP rank 间在 MoE 前共享 token；`moe_dp_size == tp_size` 时复用 `_TP`；否则以 `moe_tp_size * moe_ep_size` 为步长取 rank。publish 阶段 `runtime_context.py:1603-1604` 相应把 `moe_dp_rank` 设成 `attn_cp_rank`。

**MoE EP 组（第 2887-2915 行）。** `moe_ep_size == tp_size` 且不是 NPU 时复用 `_TP`；否则按 (MoE DP, MoE TP) 下标、以 `moe_tp_size` 为步长取 rank，关闭 PyNccl 和 custom allreduce。

**MoE TP 组（第 2917-2945 行）。** `moe_tp_size == tp_size` 时复用 `_TP`；否则每 `moe_tp_size` 个连续 rank 一组。

**PP 组与 self PP 组（第 2947-2982 行）。** PP 组以 `world_size // pp_size` 为步长取 rank；`_SELF_PP` 是每个 rank 单独一组，供单层 draft 使用（因为 `new_group` 是集合操作，所有 rank 都要创建全部单例组）。

**写回上下文（第 2984-2999 行）。** 把非 `None` 的组写入 `get_parallel()`。DCP 关闭时不写 `dcp_group`，读取会报错。注意 `runtime_context._WIDTH_AND_GROUP`（`runtime_context.py:255-261`）只校验 TP/PP/attention TP/attention CP/MoE EP 的组宽，**不校验 DCP 组宽度，也不校验 `dcp_rank`**（`_RANK_AND_WIDTH` 里没有 DCP）。

### 3.5 rank/宽度 getter、销毁与其它

| 名称 | 行号 | 作用 |
|---|---|---|
| `get_world_size` | 3169 | WORLD 组的 world_size。 |
| `get_world_rank` | 3174 | 本进程在 WORLD 组内的 rank。 |
| `get_tensor_model_parallel_world_size` | 3179 | TP 组大小。 |
| `get_dcp_world_size` | 3184 | DCP 组大小，DCP 未开启时会报错。 |
| `get_dcp_rank` | 3188 | 本进程在 DCP 组内的 rank（`rank_in_group`），DCP 未开启时会报错。 |
| `get_tensor_model_parallel_rank` | 3192 | TP 组内 rank。 |
| `get_attn_tensor_model_parallel_world_size` | 3198 | attention TP 组大小。 |
| `get_attn_tensor_model_parallel_rank` | 3203 | attention TP 组内 rank。 |
| `get_attn_context_model_parallel_world_size` | 3209 | attention CP 组大小。 |
| `get_attn_context_model_parallel_rank` | 3214 | attention CP 组内 rank。 |
| `get_pipeline_model_parallel_world_size` | 3219 | PP 组大小。 |
| `get_pipeline_model_parallel_rank` | 3224 | PP 组内 rank。 |
| `get_moe_data_parallel_world_size` | 3230 | MoE DP 组大小。 |
| `get_moe_data_parallel_rank` | 3235 | MoE DP 组内 rank。 |
| `get_moe_expert_parallel_world_size` | 3241 | MoE EP 组大小。 |
| `get_moe_expert_parallel_rank` | 3246 | MoE EP 组内 rank。 |
| `get_moe_tensor_parallel_world_size` | 3252 | MoE TP 组大小。 |
| `get_moe_tensor_parallel_rank` | 3257 | MoE TP 组内 rank。 |
| `destroy_model_parallel` | 3262 | 重置 bootstrap 标志和上下文、清理 DWDP manager，然后逐个销毁组并置空全局变量。对别名做了处理：`_SHARED_EXPERTS_TP` 是 `_ATTN_TP` 或 `_TP` 时不单独销毁；`_MOE_DP` 先于 `_ATTN_CP` 处理，且是 `_ATTN_CP` 或 `_TP` 的别名时不销毁。`_DCP`、`_MOE_EP`、`_MOE_TP`、`_ATTN_CP`、`_ATTN_TP` 没有别名检查，复用 `_TP` 时会对同一对象再调一次 `destroy()`（因为 `destroy` 会把 device/cpu 组置 `None`，组本身不会被重复销毁，推断无害）。 |
| `destroy_distributed_environment` | 3324 | 销毁 `_WORLD`、清空子组超时，并调用 `torch.distributed.destroy_process_group()`。 |
| `abort_distributed_environment` | 3334 | 关停路径用：调用 `_abort_process_group()` 在本地丢弃所有通信器，不做集合操作（对端可能已退出）；旧版 torch 没有这个接口时直接返回。 |
| `cleanup_dist_env_and_memory` | 3354 | 依次销毁模型并行组和分布式环境，可选关闭 Ray，然后 `gc.collect()` 并清空各设备缓存。 |
| `in_the_same_node_as` | 3381 | 集合操作：source rank 建一段共享内存写入魔数，其余 rank 尝试打开并比对，以此判断每个 rank 是否与 source 在同一节点。必须用非 NCCL 组调用。 |
| `monkey_patch_vllm_parallel_state` | 3456 | 把 vLLM 的 `get_pp_group/get_tp_group/get_world_group` 替换成 sglang 的（`reverse=True` 时还原），用于借用 vLLM 代码时共享组。未安装 vLLM 时什么都不做。 |
| `_warn_if_called_from_outside` | 3515 | 装饰器工厂：被包装的 getter 从 `sglang.srt.distributed.` 包外首次调用时发 `DeprecationWarning`，提示改用 `get_parallel().<replacement>`。第 3537-3541 行对 `_CONTEXT_NAME_OF` 里的 getter 批量套用；`__all__`（第 3546 行起）导出时排除这些旧 getter。 |

## 4. 与 PCP+DCP 方案 A 的关系

**计划中的修改。** 计划 4.2 节要求把第 2705-2728 行的 DCP 组构建改为：方案 A 下 DCP 组直接复用 `_ATTN_CP`，不再从 TP 组切片；配套地在 `runtime_context.py:1610-1611` 把 `dcp_rank` 设为 `attn_cp_rank`，并在 `derive_parallel_widths` 中加标志保证 `attn_dcp_size == attn_cp_size`。计划 3.2 节引用的第 2753 行就是 `attn_cp_size == tp_size` 时 `_ATTN_CP = _TP` 的分支。

**对目标配置的实际影响。** 在计划的典型配置（`tp=8, attn_cp=8, dcp=8`，attention DP=1）下，现有代码已经得到相同的成员和 rank：
- 现有 DCP 组是 TP 组的连续切片 `[0..7]`，成员与 `_ATTN_CP`（即 `_TP`）完全相同。
- `attn_tp_size = 8 // 1 // 8 = 1`，所以 `attn_cp_rank = (tp_rank // 1) % 8 = tp_rank`，而现有 `dcp_rank = tp_rank % 8 = tp_rank`，两者相等。

因此方案 A 在这一层的改动只是把"成员相同的两个 `GroupCoordinator`"合并成同一个对象，省掉一套 NCCL 通信器和 custom allreduce/PyNccl 资源，并把"DCP 组 = CP 组"写成显式约束。只要方案 A 要求 `dcp_size == attn_cp_size` 且 attention TP 为 1（即 `attn_cp_size == tp_size`、attention DP=1），rank 推导就不需要改；只有将来放开 `attn_cp_size < tp_size`（方案 B，非目标）时，`tp_rank % dcp_size` 和 `attn_cp_rank` 才会不一致（例如 TP8、CP2 时 CP 组是 [0,4]，而 DCP 切片是 [0,1]）。

**修改时需要注意的点。**
1. **构建顺序（计划未提及）。** `_DCP` 现在在 `_ATTN_CP` 之前建（第 2705 行 vs 第 2749 行），而且 `derive_parallel_widths(..., dcp_enabled=_DCP is not None)`（第 2745 行）依赖 `_DCP` 是否已建。改成复用 `_ATTN_CP` 时，要把 DCP 的赋值挪到 attention CP 组之后，并把 `dcp_enabled` 改成按 `decode_context_parallel_size > 1` 判断，否则 `attn_dcp_size` 会被推导成 1。
2. **复用后的通信器配置不同（计划未提及）。** 现有 DCP 组显式 `use_quick_allreduce=False`（第 2722 行）。复用 `_ATTN_CP`（即 `_TP`）后，DCP 的 all-reduce/all-gather/all-to-all 会走 TP 组的通信器配置（custom allreduce、quick allreduce、消息队列广播器等）。功能上应当没问题，但 DCP 的 FP32 all-reduce 和 all-to-all 会经过 `_resolve_outplace_all_reduce_method` 的选择逻辑，可能命中 custom allreduce 而不是 PyNccl（推断，需要验证数值和 CUDA graph 下的行为）。`_all_to_all_single` 依赖 PyNccl 才能被 graph 捕获，TP 组默认开启 PyNccl，这一点满足。
3. **FlashInfer workspace 标记（计划未提及）。** `_tag_groups_for_flashinfer_allreduce_only` 会给 `_ATTN_TP` 打 `attn_tp` 标记。attention TP=1 时 `_ATTN_TP` 是 1 卡组，不受影响；但如果 `_MOE_EP` 或 `_MOE_TP` 是 `_TP`，`_TP` 会被打上 `moe` 标记，此时 DCP 复用 `_TP` 后，DCP 路径上的 `all_reduce` 也可能走 FlashInfer MoE workspace。该 workspace 的 peer 与 `_TP` 一致，结果正确（推断），但需要确认 DCP 的 FP32 张量是否满足 `can_use_flashinfer_allreduce` 的条件，不满足时会自动回退。
4. **销毁时的别名（计划未提及）。** `destroy_model_parallel` 对 `_DCP` 没有别名判断（第 3293-3296 行），而 `_TP` 在它之前已经 `destroy()`。复用后会对同一对象调用两次 `destroy()`。`destroy` 把 device/cpu 组置 `None`，所以不会重复销毁 ProcessGroup，但 `pymscclpp_comm.destroy()` 没有置空，开启 MSCCL++ 时会被调用两次（推断）。建议按 `_MOE_DP` 的写法加 `is not _ATTN_CP and is not _TP` 判断。
5. **graph_capture 去重已就绪。** 模块级 `graph_capture`（第 2329-2333 行）按 `id` 去重，`_DCP is _TP` 时不会重复进入捕获上下文。
6. **缺少 DCP 一致性校验（计划未提及）。** `runtime_context._WIDTH_AND_GROUP` 和 `_RANK_AND_WIDTH` 都不包含 DCP，因此 DCP 组宽度与 `attn_dcp_size`、`dcp_rank` 与 `get_parallel().dcp_group.rank_in_group` 是否一致，现在没有任何运行时检查。计划第 6 节的单元测试要求"`dcp_rank == attn_cp_rank`"，建议同时把 `("dcp_size", "dcp_group")` 和 `dcp_rank` 加进校验，或在方案 A 分支里断言 `_DCP is _ATTN_CP`。
7. **文档字符串过时（计划未提及）。** `initialize_model_parallel` 的 docstring（第 2594-2597 行）说 DCP "currently only supported on the AMD HIP platform"，但代码（第 2664-2672 行）已允许 HIP、CUDA、NPU、XPU。
8. **`patch_tensor_parallel_group` 不处理 DCP（计划未提及）。** `owns_attention=True` 时它把 `attn_cp_size` 设 1、`attn_cp_group=None`，但不动 `dcp_group`/`dcp_rank`/`attn_dcp_size`。方案 A 中 DCP 与 CP 绑定后，draft 模型在这个上下文里仍会看到 DCP 开启而 CP 关闭。计划把投机解码放在 P4，届时需要考虑。
9. **MoE DP 别名不受影响。** 方案 A 下 `attn_cp_size(8) > moe_dp_size` 的情况会让 `_MOE_DP = _ATTN_CP`（第 2860-2864 行），这条路径与 DCP 无关，复用后三者（`_TP`、`_ATTN_CP`、`_DCP`）以及可能的 `_MOE_DP` 都是同一对象。
