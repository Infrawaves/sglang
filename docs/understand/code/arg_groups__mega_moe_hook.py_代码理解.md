# `python/sglang/srt/arg_groups/mega_moe_hook.py` 代码理解

- 文件路径：`python/sglang/srt/arg_groups/mega_moe_hook.py`
- 行数：324
- 所属模块：参数解析阶段的 MegaMoE（`--moe-a2a-backend megamoe`）处理：别名转换、平台兼容检查，以及每卡 token 预算的校验。

## 1. 整体功能

MegaMoE 是一个融合的 MoE all-to-all 后端，在 CUDA 上用 DeepGEMM kernel 加对称内存，在 ROCm 上可以选 aiter MegaMoEv2（FlyDSL）。它的工作区按"每卡最多 token 数"预先分配，大小由环境变量 `SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK` 决定。部分架构（`MEGA_MOE_NO_FALLBACK_ARCHS`，包括 K3.1）和 NVFP4 专家权重在加载时被重排成 mega 布局，没有回退到普通 fused MoE 的路径，所以必须在启动时保证任何一次前向的每卡 token 数都不超过这个上限。

在 pipeline 中有两个入口：
- `handle_mega_moe`（pipeline 第 55 行）：几乎最先运行，只依赖 `handle_deprecated_dp_attention`。做别名转换和平台/overlap 兼容检查。
- `validate_mega_moe_token_budget_for_model`（pipeline 第 339 行）：在投机解码 hook（320）、boundary reduction（325）和 FlashInfer A2A 预算校验（333）之后运行，注释（第 77 行）说明这样 decode/verify 的 token 上限已经确定。此时 `chunked_prefill_size`（DP 已经除过）、CUDA graph 配置、`attn_cp_size` 都已解析完。

## 2. 主要依赖与调用方

依赖：`arg_groups/model_override_base`（`model_config_of`、`resolving_view`）、`arg_groups/overrides.declare_resolution`、`connector.ConnectorType`、`environ.envs`（`SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK`、`SGLANG_AMD_USE_FLYDSL_MEGA_MOE`、`SGLANG_AMD_FLYDSL_MEGA_MOE_DECODE_MTPR`、`SGLANG_AMD_FLYDSL_MEGA_MOE_MTPR`）、`runtime_context`（`attn_dp_enabled_of`、`get_platform`）、`utils.common`（`ceil_align`、`parse_connector_type`）。

调用方（grep 确认）：
- `arg_groups/pipeline.py:55, 339`；`arg_groups/resolution_hooks.py:17` 登记 `handle_mega_moe`。
- 单元测试：`test/registered/unit/server_args/test_server_args.py:687-784`（`validate_mega_moe_token_budget`、`mega_moe_needs_token_budget`、`mega_moe_decode_tokens_per_rank`）。
- 在 `python/` 下没有其他模块导入这个文件。

## 3. 类与函数说明

文件里没有类。

| 名称 | 行号 | 作用 |
|---|---|---|
| `handle_mega_moe` | 19 | pipeline 入口：依次调用 `handle_moe_runner_backend_alias` 和 `check_mega_moe_compat`。 |
| `check_mega_moe_compat` | 24 | `moe_a2a_backend == "megamoe"` 时检查：不能开 two-batch overlap 或 single-batch overlap（融合 kernel 没有 dispatch/combine 的拆分）；平台必须是 CUDA SM90/SM100 或 HIP，否则报错。 |
| `MEGA_MOE_NO_FALLBACK_ARCHS` | 49 | 在 megamoe 下没有 fused-MoE 回退的架构集合，含 `DeepseekV4ForCausalLM`、`KimiK31ForConditionalGeneration`、若干 Qwen MoE 等。 |
| `mega_moe_needs_token_budget` | 65 | 架构在 `MEGA_MOE_NO_FALLBACK_ARCHS` 中，或专家是 NVFP4（`quantization` 为 `modelopt_fp4`/`minimax_m3_nvfp4`，或 `quant_algo` 含 `FP4`）时返回 True，表示需要严格校验 token 预算。 |
| `validate_mega_moe_token_budget_for_model` | 76 | pipeline 入口：megamoe 且非 encoder-only 时，先跑 `validate_flydsl_decode_capacity`；INSTANCE connector 跳过；按模型架构和量化判断需要预算时调用 `validate_mega_moe_token_budget`。 |
| `mega_moe_decode_tokens_per_rank` | 92 | 估算 decode 每卡最大 token 数：decode CUDA graph 的 `max_bs`，若设置了 `max_running_requests` 则取它与"每 DP rank 请求数按 `tp/attn_dp` 向上对齐"的较小值，再乘每请求 token 数（投机解码时是 draft token 数）。 |
| `flydsl_decode_tokens_per_rank` | 107 | FlyDSL 用：`max_running_requests` 未设时返回 None；否则请求数（attention DP 时除以 `attn_dp_size`）乘每请求行数（投机解码时取 draft token 数与 top-k 的较大值）。 |
| `validate_flydsl_decode_capacity` | 128 | 仅 HIP 且开启 FlyDSL MegaMoE、且单独设置了 decode 容量（`0 < DECODE_MTPR < MTPR`）时检查。PD prefill 跳过；`attn_cp_size > 1` 报错；`max_running_requests` 未设报错；decode 每卡行数超过 `DECODE_MTPR` 报错。 |
| `validate_mega_moe_token_budget` | 159 | 核心预算校验，HIP 跳过。非 PD prefill 时校验 decode：取 `mega_moe_decode_tokens_per_rank`，K3.1 还要和 eager 下的请求上限比较取大值，超过环境变量上限报错。PD decode 直接返回；PP + 动态 chunking 跳过；要求 chunked prefill 开启。Kimi 系列把 prefill token 数取 `chunked_prefill_size`、`max_prefill_tokens`、prefill graph 的 `max_bs`/`bs` 中的最大值，并要求不超过 `上限 × ep_size`。然后按切分方式算每卡预算：**prefill CP 时（239-245）每卡 = ceil(prefill_tokens / attn_cp_size)，Kimi 要求按 `attn_cp_size` 对齐**；attention DP 时用 DP 已除过的 chunk、按 attention TP 对齐；Kimi 且 attention TP > 1（SP-MoE）时每卡 = ceil(prefill_tokens / attn_tp_size)；其他情况每卡 = prefill_tokens、按 attention TP 对齐。对齐后的每卡预算超过上限时报错。 |
| `handle_moe_runner_backend_alias` | 307 | `--moe-runner-backend megamoe` 是 `--moe-a2a-backend megamoe` 的别名：声明 `moe_runner_backend=auto`、`moe_a2a_backend=megamoe`，原 a2a 后端不是 none/megamoe 时告警。 |

## 4. 与 PCP+DCP 方案 A 的关系

计划 §4.1 "新增约束" 一项提到：chunked prefill 大小要同时满足 CP 对齐、DCP 页对齐和 MegaMoE 上限（`mega_moe_hook.py:239-245`）。§4.9 也把 chunked prefill 对齐列为调度相关的修改。

**需要注意的点：**
- **每卡 token 数的估算方式会变**：第 243-245 行假设 prefill CP 下每卡最多 `ceil(prefill_tokens / attn_cp_size)` 个 token，这对应现有 interleave"整批拼接后按下标取模"的切分。方案 A 按每个请求的 `(prefix_len + j) % N` 分配，每个请求单独补齐，一批里有 B 个请求时，单卡最多可能拿到约 `prefill_tokens / N + B` 个 token（推断）。如果新的 `build_metadata` 按每请求 padding，这里的上限要加上请求数相关的余量，或者用 `max_running_requests`/最大 batch 请求数做上界。（计划未提及）
- **对齐**：Kimi 下 `token_alignment = attn_cp_size`（242）。计划还要求 DCP 页对齐（`page_size % dcp == 0`）。这两个对齐条件作用的对象不同：MegaMoE 这里是每卡 token 数的对齐，DCP 页对齐约束的是前缀长度和 page size。新增约束时应分开写，不要混在这个函数里（推断）。
- **Kimi 的 EP 总容量检查（227-237）** 与 CP、DCP 无关，方案 A 下不变。
- **MoE 输入行数（推断）**：计划 §3.4 说 MoE 沿用 CP-TP group sharing 的做法（先 gather 行，再 reduce-scatter 回本卡）。如果 MegaMoE 在 gather 后的完整行上运行，每卡送进 MegaMoE 的 token 数就不是 `prefill_tokens / N`。现有 K3.1 CP-TP group sharing 已经在用这个校验，方案 A 不改 MoE 路径的话不会引入新问题，但需要确认现有校验对 CP-TP sharing 的 MoE 路径本身是对的。
- **decode 预算**：方案 A 的 decode 走 DCP，CP 不生效，每卡 decode token 数和 TP8+DCP8 一样，`mega_moe_decode_tokens_per_rank` 不需要改。
- **`validate_flydsl_decode_capacity` 第 138 行** 在 HIP FlyDSL 下拒绝 `attn_cp_size > 1`，方案 A 只在 CUDA 上做，不受影响。
