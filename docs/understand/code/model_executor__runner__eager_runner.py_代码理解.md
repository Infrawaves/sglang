# `python/sglang/srt/model_executor/runner/eager_runner.py` 代码理解

- 文件路径：`python/sglang/srt/model_executor/runner/eager_runner.py`
- 行数：454
- 所属模块：模型执行层（`model_executor/runner/`），不使用 CUDA Graph 时执行一次前向的 runner。

## 1. 整体功能

`EagerRunner` 是 `BaseCudaGraphRunner` 的"eager 对偶"：当一个 batch 不能或不走 CUDA Graph 重放时，`ModelRunner` 把它交给 `EagerRunner` 直接调用模型的 `forward`。它负责三件事：

1. **输入缓冲**：构造时按"最大 batch × 每请求 token 数"和 prefill 上限，建一套固定最大尺寸的输入 registry（`build_eager_registry`）。每次前向把 live batch 复制进这套静态缓冲再切片使用（`load_batch`）。这套 registry 也是后续 CUDA Graph runner 共享输入池的基础，所以要先于 graph runner 构建。
2. **按 forward mode 分发**：`execute` 把 batch 分到 decode、idle、extend 三条路径。没有 CP 时，MIXED batch 会改写成 EXTEND。
3. **extend 路径上的 CP/DCP 准备**：extend 路径是 PCP 和 DCP 元数据准备的汇合点。先判断 `is_cp_active`，是则调用 `prepare_cp_forward` 构建 `attn_cp_metadata`；然后在 DCP 开启时调用模型的 `prepare_context_parallel_metadata_for_dcp` 构建 `attn_dcp_metadata`；最后调用 attention backend 的 `init_forward_metadata`。CP 生效时走 `_execute_extend_cp`：在模型边界切分输入、运行模型主体、gather hidden states 后再算 logits。

调用时机：`ModelRunner` 在 forward 时根据 `can_run_graph` 决定走 graph runner 还是 `EagerRunner`；`EagerRunner.can_run_graph` 永远返回 `False`。

## 2. 主要依赖与调用方

**依赖：**
- `layers/cp/utils.py`：`is_cp_active`、`prepare_cp_forward`、`cp_shard_model_inputs`、`cp_gather_after_forward`、`get_cp_strategy`。
- `layers/cp/padding.py`：`get_cp_padding_align_size`（`__init__` 中对齐 token 上限）。
- `model_executor/cuda_graph_buffer_registry.py`：`build_eager_registry`，提供 `fill_from` / `extract_buffer`。
- `model_executor/runner/base_runner.py`：基类 `BaseRunner`（`model_context`、`warmup`、`_alloc_dummy_decode_buffers`、`attn_tp_size` 等）。
- `runtime_context`：`get_parallel()`（`attn_dcp_size`、`enable_cp_tp_group_sharing`）、`get_spec()`、`get_exec()`。
- `kernels/ops/attention/kv_indices.create_chunked_prefix_cache_kv_indices`：作为参数传给 DCP 元数据构建函数。

**调用方（grep 确认）：**
- `model_executor/model_runner_components/cuda_graph_setup.py:415`：`EagerRunner(model_runner, run_warmup=not defer_distributed_setup)` 构造实例。
- `model_executor/model_runner.py:201, 2183`：导入并用 `isinstance(..., EagerRunner)` 判断当前 prefill runner 是否是 eager。
- 模型侧被它调用的钩子：`prepare_context_parallel_metadata_for_dcp`（`models/kimi_k3.py:3810, 4618`、`models/deepseek_v2.py:3594`、`models/kimi_k25.py:803`、`models/kimi_linear.py:926`），`prepare_cp_inputs`（GLM、Kimi-K3），`prepare_forward_batch`（如 Moss-VL）。

## 3. 类与函数说明

### `EagerRunner(BaseRunner)`（第 77 行）

| 名称 | 行号 | 作用 |
|---|---|---|
| `__init__` | 78 | 计算每请求 token 数（投机解码按 draft token / topk / multi-layer eagle 取最大，dLLM 用 `block_size`），计算 `max_bs`（frozen-KV MTP 再乘 topk，并按 `get_eager_max_batch_size` 对齐）和 `max_num_token`（取 KV 池容量、prefill 上限与 `max_bs × tokens_per_req` 的最大值；需要 MLP sync 时再按 `attn_tp_size` 和 CP padding 对齐）。随后用 `build_eager_registry` 建输入缓冲，并在 `run_warmup=True` 时调用 `warmup()`。 |
| `_autotune_buffers` | 151 | 为 warmup 时 flashinfer MoE autotune 准备 decode 形状的 dummy 缓冲（`bs × num_tokens_per_req`），保证 autotune 选出的 tactic 和真实 decode 形状一致。返回 `(buffers, max_bs)`。 |
| `can_run_graph` | 171 | 恒返回 `False`，防止调用方把 eager batch 路由进 graph 重放分支。 |
| `load_batch` | 176 | 把 live batch 复制进固定最大尺寸的静态缓冲并按本 batch 形状切片返回。设置 `SGLANG_EAGER_INPUT_NO_COPY` 时直接返回浅拷贝（`dataclasses.replace`）。 |
| `execute` | 206 | 入口。在 `model_context()` 中按 forward mode 分发；非 NPU 且没有 CP 策略时把 MIXED 改成 EXTEND（CP 下保留 MIXED 以跳过 zigzag 切分）。不支持的 mode 抛 `ValueError`。 |
| `_execute_decode` | 225 | decode 路径：`load_batch`，必要时调用模型 `prepare_forward_batch` 和 `attn_backend.init_forward_metadata`，然后在 `device_timer_ctx` 和 `forward_context` 中调用 `model.forward`。 |
| `_execute_extend` | 254 | extend 路径（含 target verify、draft extend v2）。顺序是：`load_batch` → `is_cp_active` 为真时 `prepare_cp_forward`（第 264-266 行）→ 在需要初始化元数据、CP 生效或 target verify 时：若 `attn_dcp_size > 1`、非 target verify、模型有 `prepare_context_parallel_metadata_for_dcp` 且 `extend_prefix_lens` 非空，则构建 `attn_dcp_metadata`（第 279-309 行）；再调用 `prepare_forward_batch`、`init_forward_metadata`、`resolve_prefill_shared_read_ends` 和 `maybe_publish_prefill_shared_read_done`。CP 不生效时清空 `attn_cp_metadata`。最后：CP 生效且（不是 CP-TP group sharing，或模型提供 `prepare_cp_inputs`）时走 `_execute_extend_cp`，否则直接 `model.forward`。 |
| `_execute_extend_cp` | 356 | CP extend：通过模型的 `prepare_cp_inputs`（或默认用 embedding 和 positions）得到输入；在 `cp_shard_model_inputs` 上下文中把 embedding、positions、input_ids 切到本 rank，运行模型主体 `model.model`（多模态包装时再取一层 `.model`）；非 PP 最后一级直接返回 proxy；否则用 `cp_gather_after_forward` gather hidden states（以及 DSpark 的 aux hidden states），处理 DSV4 返回的 `(hidden, hidden_before_norm)` 二元组，最后调用 `logits_processor`。 |
| `_execute_idle` | 432 | idle 路径：batch_size > 0（DP attention 的 MLP sync padding）时 `load_batch` 并初始化元数据；否则把 `attn_backend.forward_metadata` 置空，避免 SWA 在 req_pool 上的 use-after-free。然后调用 `model.forward`。 |

## 4. 与 PCP+DCP 方案 A 的关系

计划 §4.5 把本文件第 264-300 行列为修改点："调整 `prepare_cp_forward` 和 `prepare_context_parallel_metadata_for_dcp` 的调用顺序和依赖关系"。读代码后的具体说明：

1. **两份元数据目前互相独立。** `prepare_cp_forward`（第 266 行）只构建 `attn_cp_metadata`，并把 `out_cache_loc` 截到 `sum(extend_seq_lens_cpu)`（`layers/cp/utils.py:303-304`）。它不把 `out_cache_loc` 缩成本 rank 的行，也不修改 `seq_lens`、`extend_prefix_lens`。DCP 元数据（第 295-309 行）读的是这些未切分的字段。方案 A 要做的"合并元数据"（计划 §4.5 第一行）需要在这里拿到 `attn_cp_metadata` 的本卡行信息，所以 DCP 元数据构建要依赖 CP 元数据，顺序必须是 CP 在前（现在已经是这个顺序）。
2. **`out_cache_loc` 与本地写入。** 方案 A 要求"本卡算的 token 直接写本地"（计划 §3.4 第 2 步）。现在 CP 下 `out_cache_loc` 仍是整批的 slot，K 是先 gather 成整批再按完整 `out_cache_loc` 写入的。如果改成只写本卡行，需要在这里（或 `prepare_cp_forward` 里）得到与本卡行一一对应的 `out_cache_loc` 子集。计划 §4.3 把这一点放在 `layers/cp/utils.py:270-304`，两处需要一起改。
3. **`attn_dcp_metadata` 的生成条件。** 第 280-292 行的条件只排除 target verify，不看 CP 是否生效，所以方案 A 下 CP extend 会同时得到两份元数据，这一点不用改。第 281 行和第 289 行都检查 `is_target_verify()`，第 288-291 行那段在第 281 行已排除的前提下永远为假（推断，属冗余条件，不影响功能）。
4. **`_execute_extend_cp` 的进入条件。** 第 339-342 行：CP-TP group sharing 下只有提供 `prepare_cp_inputs` 的模型（GLM、Kimi-K3）才走 `_execute_extend_cp`。K3.1 属于这一类，所以方案 A 的"按位置取模"切分会经过 `cp_shard_model_inputs` 和 `cp_gather_after_forward`，这两个函数要和 `layers/cp/interleave.py` 的新行布局一致（计划 §4.3）。
5. **MIXED 模式。** `execute`（第 211 行）在 CP 策略存在时保留 MIXED。方案 A 下 decode 尾巴混在 extend 里时，按位置取模的切分要能处理 1-token 的 extend 段（计划未提及）。
6. **`__init__` 的 token 上限对齐。** 第 120-124 行在 `require_mlp_sync()` 时按 `attn_tp_size` 和 CP padding 对齐 `max_num_token`。方案 A 新切分如果改变 padding 规则（每请求单独 padding），`get_cp_padding_align_size` 的结果和这里的静态缓冲上限要同步核对（计划未提及）。

**关于冻结文件的提醒：** 计划 §4.5 说本文件是"冻结的核心路径，修改前要先读 `large-class-style` skill"。仓库规则 `.claude/rules/modify-component-must-read.md` 和 `.claude/skills/large-class-style/SKILL.md` §1.2 目前只把 `model_executor/model_runner.py` 列为冻结文件，`eager_runner.py` 不在列表里。不过它属于模型执行核心路径，按计划要求修改前仍建议先读该 skill，遵循"编排函数只做构造、连接、委托、协调"的风格：新的合并元数据逻辑应放到 `layers/cp/` 或 `layers/dcp/` 的协作者里，这里只做调用。
