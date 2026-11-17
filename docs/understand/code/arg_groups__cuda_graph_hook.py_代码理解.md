# `python/sglang/srt/arg_groups/cuda_graph_hook.py` 代码理解

- 文件路径：`python/sglang/srt/arg_groups/cuda_graph_hook.py`
- 行数：610
- 所属模块：参数解析阶段的 CUDA graph 配置：把各种 CLI 开关合并成 `CudaGraphConfig`（分 decode、prefill 两个阶段），按模型、并行方式、PD 角色自动开关 graph，并生成捕获用的 batch size 列表。

## 1. 整体功能

`CudaGraphConfig`（`model_executor/cuda_graph_config.py`）按阶段（`Phase.DECODE`、`Phase.PREFILL`）保存后端（`Backend.DISABLED`/`FULL`/`BREAKABLE`）和 `max_bs`、`bs`、`max_seq_len`、`max_context_size` 等参数。prefill 阶段的 `bs` 表示捕获的 token 数。"breakable"（BCG）是分段捕获的 prefill graph。

`parse_cuda_graph_config` 按优先级合并来源：显式 JSON > 每阶段便捷开关 > 旧的全局开关 > 默认值，并把非默认来源的 `(phase, key)` 记到 `server_args._cuda_graph_config_locked`。后续自动关闭逻辑都尊重这个锁：用户显式指定了 prefill 后端时，不再自动改。

在 pipeline 中的位置：
- 第 195-197 行：`apply_inkling_prefill_cuda_graph_default`、`apply_muse_glimmer_prefill_cuda_graph_max_bs_default`、`apply_unlimited_ocr_v2_cuda_graph_default`，必须在解析 CUDA graph 配置之前（声明式 model override 生效太晚）。
- 第 202 行：`handle_cuda_graph_config`，在 `handle_dwdp` 之后、平台 hook 和 `handle_gpu_memory_settings`（231，生成 batch size 列表）之前。
- 第 258 行：`disable_prefill_cuda_graph_for_deepseek_trtllm_mla`，attention 后端解析之后。
- 第 293 行：`apply_prefill_cp_cuda_graph_default`，只在开启 prefill CP 时，`handle_context_parallelism` 之后运行。
- 第 379 行：`finalize_cuda_graph_prefill_max_context`，在 page size 和所有 capability 调整之后。

## 2. 主要依赖与调用方

依赖：`arg_groups/model_override_base`（`attention_backends_of`、`model_config_of`、`resolved_view`、`resolving_view`）、`arg_groups/overrides.declare_resolution`、`model_executor/cuda_graph_config`、`connector.ConnectorType`；延迟导入 `layers/cp/bcg.supports_prefill_cp_bcg`、`configs/model_config` 的 `is_deepseek_v4`、`uses_kda_attention`、`UNLIMITED_OCR_V2_ARCH`。

调用方（grep 确认）：
- `arg_groups/pipeline.py:185-202, 258, 293, 379`。
- `generate_prefill_cuda_graph_batch_sizes`：`arg_groups/parallel_hook.py:370`（DP 下重新钳制）、`arg_groups/model_hook.py:996`、`arg_groups/memory_hook.py:76`。
- `generate_decode_cuda_graph_batch_sizes`、`generate_cpu_graph_batch_sizes`：`arg_groups/memory_hook.py:74-75, 205`。

## 3. 类与函数说明

文件里没有类。

### 3.1 配置解析与校验

| 名称 | 行号 | 作用 |
|---|---|---|
| `parse_cuda_graph_config` | 32 | 从默认配置出发，依次应用 `disable_cuda_graph`、`disable_prefill/decode_cuda_graph`、各阶段的 `cuda_graph_backend_*`/`max_bs`/`bs`/`max_seq_len`/`prefill_max_context` 便捷开关，最后应用显式 JSON；声明 `cuda_graph_config` 并写入 `_cuda_graph_config_locked`。 |
| `handle_cuda_graph_config` | 426 | pipeline 入口：解析 → 兼容性自动关闭 → DeepEP 调整 → PD 角色 → 校验；最终 prefill 后端为 FULL 时告警（实验性）。 |
| `validate_cuda_graph_config` | 444 | 每个阶段的后端必须在 `ALLOWED_BACKENDS_PER_PHASE` 中，否则报错。 |

### 3.2 自动开关规则

| 名称 | 行号 | 作用 |
|---|---|---|
| `apply_prefill_cp_cuda_graph_default` | 104 | 开启 prefill CP 时运行。prefill 后端未被锁定且 `supports_prefill_cp_bcg` 为真时，把 prefill 后端设为 BREAKABLE，并重新跑兼容性检查、PD 角色和 DeepEP 调整（这样其他不兼容规则仍会生效）；最终不是 BREAKABLE 时告警。`supports_prefill_cp_bcg`（`layers/cp/bcg/__init__.py:42`）要求 `attn_cp_size == tp_size`，且（策略, 后端）是 (zigzag, trtllm_mha) 或 (interleave, dsa)。 |
| `apply_cuda_graph_compatibility` | 136 | prefill 后端被锁定时直接返回。PP > 1 且默认 BREAKABLE 时关闭 prefill graph；否则按后端分别调用 BCG 或 FULL 的不兼容检查。 |
| `disable_breakable_cudagraph_if_incompatible` | 170 | BCG 的不兼容规则列表，命中第一条就关闭 prefill graph：KDA 混合线性 attention（GLM-5 Next 除外）；不支持 CP BCG 的 DeepSeek-V4；**`attn_cp_size > 1` 但不支持 CP BCG（208-212，CP all_gather 回放尺寸不匹配，延后告警所以 name 为 None）**；**`dcp_size > 1`（213-216，捕获时构造的 dummy extend 前向里 `attn_dcp_metadata=None`）**；two-batch overlap；未验证的 a2a 后端（只允许 none/deepep/megamoe/flashinfer）；不在白名单的多模态模型。 |
| `disable_full_prefill_cudagraph_if_incompatible` | 258 | FULL prefill graph 的规则列表，目前为空。 |
| `disable_prefill_cuda_graph_for_deepseek_trtllm_mla` | 279 | 未锁定且 prefill graph 开启时，`DeepseekV3ForCausalLM` 加 `trtllm_mla` prefill 后端则关闭 prefill graph（捕获下 trtllm_mla 会回退到 FlashAttention，性能变差）。 |
| `apply_deepep_adjustments` | 315 | a2a 后端为 DeepEP 且 prefill 为 BREAKABLE 时，把 prefill bucket 向上对齐到 8 的倍数（非 8 倍数可能让 DeepEP 捕获挂起）。 |
| `apply_cuda_graph_disaggregation_roles` | 591 | PD prefill 实例默认关闭 decode graph，PD decode 实例默认关闭 prefill graph（都尊重锁）。 |

### 3.3 模型相关默认值

| 名称 | 行号 | 作用 |
|---|---|---|
| `apply_inkling_prefill_cuda_graph_default` | 349 | Inkling 系列在用户未指定时把 `cuda_graph_backend_prefill` 设为 FULL。 |
| `apply_muse_glimmer_prefill_cuda_graph_max_bs_default` | 375 | MuseGlimmer 系列默认 `cuda_graph_max_bs_prefill=512`。 |
| `apply_unlimited_ocr_v2_cuda_graph_default` | 392 | Unlimited-OCR v2 默认关闭 decode graph（告警说明需显式开启）和 prefill graph（block table 宽度随请求变化）。 |

### 3.4 尺寸与 batch size 列表

| 名称 | 行号 | 作用 |
|---|---|---|
| `_resolve_max_context_size` | 457 | 把请求的 prefill graph 最大上下文向上对齐到 `page_size`；非正数报错，对齐后超过模型上下文长度报错。 |
| `finalize_cuda_graph_prefill_max_context` | 484 | 设置了 `max_context_size` 时，按最终 `page_size` 对齐并声明回配置。 |
| `generate_prefill_cuda_graph_batch_sizes` | 514 | 生成 prefill 捕获的 token 数列表：4-32 步长 4，48-256 步长 16，288-512 步长 32，576-1024 步长 64，1280-4096 步长 256，之后步长 512，截到 `max_bs`。 |
| `generate_decode_cuda_graph_batch_sizes` | 534 | 生成 decode 捕获 batch size：关闭 padding 时 1..max_bs；普通情况 1,2,4,8,12 再按 8/16/32 步长；投机解码时小 batch 更密；保证包含 `max_bs`。 |
| `generate_cpu_graph_batch_sizes` | 569 | 按 `torch_compile_max_bs` 生成 CPU graph 的 batch size 列表。 |

## 4. 与 PCP+DCP 方案 A 的关系

计划 §4.8 提到 prefill breakable graph（`prefill_cuda_graph_runner.py:736-740`、本文件 `:208-212`）需要重新计算 CP buffer 和 DCP gather buffer 的静态尺寸，P1、P2 阶段先禁用。

**需要注意的点：**
- **P1/P2 禁用已经是现有行为**：计划引用的 208-212 是"CP 不支持 BCG"规则。K3.1 + interleave + DSA 满足 `supports_prefill_cp_bcg` 的组合（`(INTERLEAVE, DSA)` 在白名单里，最终还要看 `get_cp_bcg_input_type(...).supports`），这条规则不会命中。真正会在方案 A 下关闭 BCG 的是紧接着的 213-216 行 `dcp_size > 1` 规则。流程是：`apply_prefill_cp_cuda_graph_default` 先把后端设为 BREAKABLE，再调 `apply_cuda_graph_compatibility`，被 DCP 规则关掉，然后第 130-133 行打"BCG 未启用"告警。所以 P1/P2 不需要改代码就会禁用 BCG。（计划引用的行号不是起作用的那条规则）
- **锁会绕过规则**：用户显式传 `--cuda-graph-backend-prefill breakable` 时，`apply_cuda_graph_compatibility` 第 143 行直接返回，DCP 规则不生效，BCG 会在方案 A 下被强行开启。现有 TP+DCP 也有这个问题，但方案 A 下 CP BCG 的静态 buffer 不适配新的行布局，可能静默算错。P1/P2 建议在 CP 或 DCP 的校验里显式拒绝"方案 A + 显式 breakable"（推断）。（计划未提及）
- **P3 要开启 BCG 时**：需要修改 213-216 的 DCP 规则，让它在方案 A 下放行；同时 `supports_prefill_cp_bcg` 对应的 interleave BCG 输入类型（`layers/cp/bcg/interleave_bcg.py`）要按新的按位置取模布局计算静态尺寸（计划 §4.3 已提到）。
- **decode graph**：本文件对 decode graph 没有 CP/DCP 相关的规则。计划 §4.8 的 decode graph 修改（cp-decode-attn-tp 的切片缓存）在 runner 里，不在这里。
- **`apply_cuda_graph_disaggregation_roles`**：PD prefill 实例默认关闭 decode graph。方案 A 在 PD prefill 实例上不做 decode，这条无影响。
- **`generate_prefill_cuda_graph_batch_sizes` 的 bucket**：表示的是整批 token 数。方案 A 下每卡 token 数不再是 `bucket / N`（见 `mega_moe_hook` 文档中关于每请求 padding 的说明），P3 开 BCG 时，CP BCG 的每卡静态尺寸需要按新规则计算（推断）。
