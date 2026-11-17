# `python/sglang/srt/arg_groups/parallel_hook.py` 代码理解

- 文件路径：`python/sglang/srt/arg_groups/parallel_hook.py`
- 行数：841
- 所属模块：参数解析（resolution）阶段的并行相关 hook：attention DP 旧写法迁移、prefill CP、DCP、数据并行、DWDP、Elastic EP、EPLB/分发算法、共享专家 TP 等的校验和默认值推导。

## 1. 整体功能

模块 docstring 写的是"context parallelism 和 decode context parallelism 的参数解析"，实际内容覆盖了大部分并行相关的 hook。每个 `handle_*` / `validate_*` 函数接收 `server_args`，通过 `resolving_view(server_args)`（当前正在解析的视图，能看到前面 hook 声明的值）读取配置，通过 `declare_resolution(server_args, <来源名>, **字段)` 声明新值，不直接修改 `server_args`。`resolved_view` 用于读取"已经声明后的结果"（例如 `attn_cp_size`、`ep_size` 这类由其他 pass 推导的字段）。

这些 hook 在 `arg_groups/pipeline.py` 的 `run_resolution_pipeline` 里通过 `run_hook` 按固定顺序调用（`run_hook` 允许插件按名字替换某个 hook，名字登记在 `arg_groups/resolution_hooks.py`）。与本文件相关的调用顺序：

1. `handle_deprecated_dp_attention`（pipeline 第 51 行）：最先运行，任何 handler 读取 DP 布局之前。
2. `validate_prefill_cp_platform`（第 111 行）：在模型路径解析和 dummy 模型提前返回之前。
3. `handle_decode_context_parallelism`（第 180 行）：在 CUDA graph 配置（202）、模型相关调整（`handle_model_specific_adjustments`，238，K3.1 的 model override 在这里收集）之前。
4. `handle_dwdp`（第 200 行）：必须在 `handle_cuda_graph_config` 和 `handle_data_parallelism` 之前。
5. `handle_data_parallelism`（第 287 行）：在 `handle_hicache`（285）之后。
6. `handle_context_parallelism`（第 291 行）：之后若开启 prefill CP 再跑 `apply_prefill_cp_cuda_graph_default`（293），再跑 `handle_kv_cache_sharding`（297）。
7. `handle_shared_experts_tp`、`handle_eplb_and_dispatch`、`handle_expert_distribution_metrics`、`handle_elastic_ep`（第 310-313 行）：MoE 配置之后。

需要注意：DCP 的校验（第 3 步）早于 prefill CP 的校验（第 6 步），也早于 model override。所以 `handle_decode_context_parallelism` 看到的 `enable_prefill_cp` 是用户输入值，`attn_cp_size` 也还没经过 CP 相关推导。

## 2. 主要依赖与调用方

依赖：
- `arg_groups/cp_tp_group_sharing_hook.resolve_cp_tp_group_sharing`：`handle_context_parallelism` 内调用。
- `arg_groups/model_override_base`：`model_config_of`、`resolving_view`、`resolved_view`。
- `arg_groups/overrides`：`declare_resolution`、`run_post_process_pass` 以及若干 post-process pass（`_dcp_comm_backend_default`、`_data_parallelism_defaults`、`_tp_lm_head_all_to_all_default`、`_dp_lm_head_validation`）、`should_report_expert_balancedness`。
- `runtime_context`：`attn_dp_enabled_of`、`derive_attn_tp_size`、`get_platform`。
- `model_executor/cuda_graph_config`：`Backend`、`Phase`、`with_phase`。
- 延迟导入：`layers/cp/base.init_cp_strategy` / `is_cp_sharded_padded`、`arg_groups/cuda_graph_hook.generate_prefill_cuda_graph_batch_sizes`、`arg_groups/validation_hook.validate_ib_devices`、`elastic_ep/topology.physical_ep_size_to_dp_size`。

调用方（grep 确认）：
- `arg_groups/pipeline.py`：上面列出的所有公开 hook。
- `arg_groups/resolution_hooks.py:16, 29, 41, 74, 81`：登记 hook 名字。
- 单元测试：`test/registered/unit/server_args/test_server_args.py`（`handle_context_parallelism`、`handle_deprecated_dp_attention`、`handle_decode_context_parallelism`）、`test_platform_prefill_cp_deprecation.py`、`test/registered/unit/test_model_overrides.py`、`test/registered/unit/layer_boundary/test_short_conv_boundary.py`（`_boundary_parallelism_overrides`）。

## 3. 类与函数说明

文件里没有类，只有模块级函数。

### 3.1 CP / DCP 相关

| 名称 | 行号 | 作用 |
|---|---|---|
| `handle_deprecated_dp_attention` | 41 | 把旧写法 `--dp-size N --enable-dp-attention` 转成 `--attn-dp-size N --dp-size 1`（仅当 `attn_dp_size == 1` 时），并把 `enable_dp_attention` 声明为 False。 |
| `_boundary_parallelism_overrides` | 66 | 按模型类型做层边界相关的检查：Nemotron-H 拒绝 `attn_cp_size > 1`；LongCat-Flash 拒绝 prefill CP 加 `attn_cp_size > 1`；Nemotron/LongCat/Kimi-K3 系列/Inkling 开了 `enable_attn_tp_input_scattered` 时告警并返回关闭它的声明。 |
| `handle_context_parallelism` | 97 | prefill CP 的主校验。先经 `run_hook` 再跑一次 `validate_prefill_cp_platform`（让插件替换也生效）；非 INSTANCE connector 时读模型架构，声明 `_boundary_parallelism_overrides` 的结果，并做架构限制：HIP 上 prefill CP 只支持 `DeepseekV4ForCausalLM`，DeepSeek V3.2 不支持 zigzag，MiMo V2 只支持 zigzag 且只支持文本；然后调用 `resolve_cp_tp_group_sharing`。之后要求开启 prefill CP 时必须给 `--cp-strategy`；检查 `tp_size` 能被 `attn_cp_size`、`attn_dp_size * attn_cp_size` 整除；`moe_dp_size > 1` 时检查 EP/PP 约束；`attn_cp_size != moe_dp_size` 时要求 `moe_dp_size == 1`。最后调用 `init_cp_strategy` 绑定本进程的 CP 策略单例；若 `enable_unified_memory` 且策略是 KV 分片补齐（`is_cp_sharded_padded`）则报错。 |
| `handle_decode_context_parallelism` | 257 | DCP 的校验。先跑 `_dcp_comm_backend_default`（未设置时：`dcp_size <= 1` 选 `ag_rs`，支持 FlashInfer MNNVL 时选 `fi_a2a`，CUDA/HIP/XPU 选 `a2a`，否则 `ag_rs`）。然后检查 `dcp_size >= 1`；**第 266-267 行禁止 `dcp_size > 1` 与 `enable_prefill_cp` 同时开启**；`a2a`/`fi_a2a` 要求 `dcp_size > 1`；`fi_a2a` 要求 CUDA 平台；`dcp_replicate_q_proj` 要求 `dcp_size > 1` 且通信后端是 `a2a`/`fi_a2a`。 |
| `validate_prefill_cp_platform` | 834 | MUSA 上开启 prefill CP 时报错（已废弃）。在 pipeline 中较早运行，`handle_context_parallelism` 内再经 `run_hook` 调一次。 |

### 3.2 数据并行 / NCCL

| 名称 | 行号 | 作用 |
|---|---|---|
| `handle_data_parallelism` | 293 | 检查 `attn_dp_size >= 1`，并禁止 `dp_size > 1` 和 `attn_dp_size > 1` 同时使用；跑 `_data_parallelism_defaults`；对 `mm_enable_dp_encoder` 打日志。开启 attention DP 时：`schedule_conservativeness` 乘 0.3、Rust server 下开启本地控制广播、`chunked_prefill_size` 除以 `attn_dp_size`，并把 prefill CUDA graph 的 `max_bs`（和未锁定时的 `bs` 列表）钳到新的 `chunked_prefill_size`。最后跑 LM-head 相关的两个 pass，并在可能捕获跨卡集合通信时关闭 NCCL graph buffer 注册。 |
| `_graph_may_capture_collectives` | 393 | 判断是否有任何 CUDA graph 可能包含跨卡集合通信：`tp/ep/dp` 都为 1 时 False；`cuda_graph_config` 为 None 时 True；否则只要任一阶段的 graph 后端不是 DISABLED 就是 True。docstring 说明这是保守的规避，根因未定。 |
| `_disable_nccl_graph_buffer_registration` | 419 | `os.environ.setdefault("NCCL_GRAPH_REGISTER", "0")`；用户显式设成非 0 时打告警。必须在 scheduler 创建 NCCL communicator 之前运行。 |

### 3.3 MoE / 专家并行相关

| 名称 | 行号 | 作用 |
|---|---|---|
| `handle_shared_experts_tp` | 213 | 设置了 `--shared-experts-tp-size` 时检查：必须是 `attn_tp_size * attn_cp_size`（attention 副本大小）的正因子；模型必须是 Kimi-K3 / K3.1；`moe_a2a_backend` 必须是 `deepep`/`megamoe`/`mooncake`/`ascend_fuseep`/`mori` 之一。 |
| `handle_dwdp` | 450 | `dwdp_size > 1` 时：断言 `dwdp_size == tp_size`、PD 模式为 null 或 prefill、未开 EPLB、无投机解码、`pp_size == 1`、未开 two-batch overlap；然后声明 `attn_dp_size=dwdp_size`、`dp_size=1`、本地控制广播、DP LM head、`moe_dense_tp_size=1`、`ep_size=dwdp_size`、`moe_dp_size=1`、`moe_a2a_backend=none`、`disable_cuda_graph=True`，并设置 `SGLANG_SCHEDULER_SKIP_ALL_GATHER`。 |
| `handle_elastic_ep` | 536 | Elastic EP 的校验：EPLB 算法默认改为 `elasticity_aware`；`pp_size == 1`；mooncake 后端校验 IB 设备；`ep_join_mode`、`ep_join_rank_offset`、`max_ep_size` 的组合约束。`max_ep_size > tp_size`（运行时扩容）时还要求：attention 拓扑合法、`attn_tp_size > 1` 时 `moe_dense_tp_size == 1`、超时为正、单 tokenizer worker、不用 Ray、mooncake 后端、round_robin 负载均衡、`attn_tp_size % dcp_size == 0`、decode graph 只能是 full/disabled、prefill graph 关闭、开启 attention DP 和 DP LM head、`attn_cp_size == 1`、`moe_dp_size == 1`、`ep_size == tp_size`、`moe_a2a_backend == nixl`。 |
| `handle_eplb_and_dispatch` | 764 | 开 EPLB 且未设 recorder 模式时默认 `stat`；开 EPLB 或非 trivial 初始专家位置且未设分发算法时，`moe_a2a_backend == none` 选 `dynamic`，否则 `static`；无 a2a 后端时禁止 `static`/`lp`（各卡选的副本会不一致）；开 EPLB 且非扩容加入时断言 `ep_size > 1`。 |
| `handle_expert_distribution_metrics` | 808 | 需要上报专家均衡度且未设 recorder 模式时默认 `stat`；recorder buffer 大小未设时取 `eplb_rebalance_num_iterations`，否则在开了 recorder 时默认 1000。 |

## 4. 与 PCP+DCP 方案 A 的关系

计划 §4.1 的第一项就是这个文件：把 `handle_decode_context_parallelism` 第 266-267 行的互斥检查改成条件放行，条件是 `dcp_size == attn_cp_size`、`cp_strategy == interleave`、`dcp_kv_layout == token`，zigzag 时明确报错。

**需要注意的点：**
- **调用顺序**：`handle_decode_context_parallelism` 在 pipeline 第 180 行运行，`handle_context_parallelism`（含 `resolve_cp_tp_group_sharing`）在第 291 行。在第 180 行时，`cp_strategy` 是用户输入值（可能为 None，"必须设置 `--cp-strategy`"的检查在第 154 行，晚于 DCP 检查）；`attn_cp_size` 也只是 `resolving_view` 下的原值。新条件如果写在第 266 行，要能处理 `cp_strategy is None` 的情况，或者把方案 A 的组合校验挪到 `handle_context_parallelism` 之后统一做（推断）。（计划未提及）
- **`dcp_kv_layout == token` 条件**：`pd_disaggregation_hook.validate_dcp_kv_layout`（pipeline 第 387 行，最后阶段）已经要求 `--dcp-kv-layout page` 只能用在 PD decode 服务器上，而 K3.1 的 CP-TP group sharing 又拒绝 `disaggregation-mode decode`。所以统一部署和 PD prefill 实例下 page 布局本来就会被拒，这条显式检查主要起提示作用；如果写在第 266 行，读到的也是最终值之前的视图（推断）。（计划未提及）
- **`handle_context_parallelism` 里的 `init_cp_strategy`（199-204）**：方案 A 若用新的 CP 策略类（按位置取模的 interleave）或通过 `register_cp_kv_layout` 注册布局，这里是策略对象绑定的地方，需要能根据"DCP 已开启"选择新策略，或在策略内部读取 DCP 配置（推断）。（计划未提及）
- **`is_cp_sharded_padded` 与 unified memory（205-210）**：方案 A 的新策略不应设 `cp_kv_sharded=True`，否则会触发这条报错，并与 DCP 的 owner 切分叠加。K3.1 本身在 model override 里拒绝 `--enable-unified-memory`。
- **整除检查（160-167）**：方案 A 要求 `attn_cp_size == tp_size`（CP-TP group sharing 已检查），此时 `attn_tp_size = 1`。现有整除检查都能通过。
- **`handle_shared_experts_tp`（213）**：attention 副本大小 = `attn_tp_size * attn_cp_size`，方案 A 下为 `tp_size`，和 DCP 无关，不受影响。
- **`handle_elastic_ep` 第 702 行** 要求 `attn_tp_size % dcp_size == 0`。方案 A 下 `attn_tp_size == 1`，这条在 `dcp_size > 1` 时会失败，但扩容路径本身要求 `attn_cp_size == 1`（745），与方案 A 不会同时出现，不需要改。
- **`_dcp_comm_backend_default`**（overrides.py:1605）在第 258 行运行，`is_fi_a2a_supported` 按 `dcp_size/tp_size/pp_size/nnodes` 判断，和 CP 无关。方案 A 下 decode 合并的通信后端选择沿用现有逻辑即可（推断）。
