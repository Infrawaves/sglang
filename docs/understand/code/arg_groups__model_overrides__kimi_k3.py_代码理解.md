# `python/sglang/srt/arg_groups/model_overrides/kimi_k3.py` 代码理解

- 文件路径：`python/sglang/srt/arg_groups/model_overrides/kimi_k3.py`
- 行数：297
- 所属模块：Kimi-K3 / Kimi-K3.1 的模型级参数覆盖（model override）：按架构声明默认的 attention 后端、MoE runner、DCP 相关默认值，并拒绝不支持的组合。

## 1. 整体功能

这个文件里的 provider 用 `_register_for(<架构名>...)` 注册到 `model_override_base._MODEL_OVERRIDE_FNS`。约定（`register_model_override` 的 docstring）是：provider 接收 `(server_args, hf_config)`，不能修改它们，返回 `{字段: 值}` 字典，可以是空字典；遇到不支持的组合直接抛异常。

调用时机：pipeline 第 238 行的 `handle_model_specific_adjustments`（`arg_groups/model_hook.py:241-245`）调用 `collect_model_override_declarations(model_arch, server_args, hf_config)` 收集这些声明，经 `validate_declarations` 后追加到 `server_args._resolved_overrides`，后续 hook 通过 `resolved_view` 看到这些值。这一步晚于 `handle_decode_context_parallelism`（180）和 `handle_cuda_graph_config`（202），早于 `handle_attention_backend_compatibility`（255）、`handle_hicache`（285）和 `handle_context_parallelism`（291）。

同一个架构可以注册多个 provider。K3.1（`KimiK31ForConditionalGeneration`）上会运行三个：本文件的 `_kimi_k31_overrides`、`_kimi_k3_moe_runner_overrides`，以及 `model_overrides/deepseek_v2.py:39` 的 `_deepseek_family_overrides`（该列表包含 K3.1）。

核心内容：
- **K3（`KimiK3ForConditionalGeneration`）**：不支持 prefill CP；DCP 下选择 decode 后端 `cutedsl_mla`（配 `trtllm_mla` prefill）或 `aiter`，并处理 DSPARK/EAGLE3 的 verify 模式；非 DCP 时在 SM100/SM103 上默认 `trtllm_mla`。
- **K3.1（`KimiK31ForConditionalGeneration`）**：DCP 检查（拒绝 prefill CP、HiSparse、非 mooncake 的 PD 等）、DFLASH/DSPARK 的 verify 模式、无损默认值（FlashMLA prefill token packing）。
- **两者共用**：SM100/103/107 上 MXFP4 打包权重默认 `moe_runner_backend=flashinfer_mxfp4`。

## 2. 主要依赖与调用方

依赖：
- `arg_groups/model_override_base`：`_register_for`、`resolving_view`、`attention_backends_of`（返回 `(prefill, decode)` 后端，拆分字段为空时回落到 `attention_backend`）、`is_attention_backend_not_set`、`_dspark_verify_on_decode_backend`、`_is_mxfp4_pack_quantized`。
- `runtime_context.get_platform`。
- 延迟导入：`flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla`（检查签名）、`layers/attention/aiter_mla_gluon._gluon_fn`、`speculative/ragged_verify`、`environ.envs`。

调用方：
- `arg_groups/model_overrides/__init__.py` 导入这个模块完成注册；`arg_groups/overrides.collect_model_override_declarations`（overrides.py:271）实际调用。
- `model_overrides/deepseek_v2.py:105` 对 K3.1 跳过 `_dsa_dcp_overrides`，注释写明 K3.1 的 DCP 由 `_kimi_k31_dcp_overrides` 处理。
- 单元测试：`test/registered/unit/test_model_overrides.py:4446-4470`（`_kimi_k31_dcp_overrides`）、`:4599-4608`（`_kimi_k31_lossless_defaults`）。

## 3. 类与函数说明

文件里没有类。

| 名称 | 行号 | 作用 |
|---|---|---|
| `_require_kimi_k3_cutedsl_dcp_support` | 20 | 检查 FlashInfer 的 `trtllm_batch_decode_with_kv_cache_mla` 能否导入，且签名里有 `enable_dcp` 参数；否则 `RuntimeError`，提示需要 FlashInfer 0.6.17 以上。K3 DCP 用 `cutedsl_mla` 解码时调用。 |
| `_require_kimi_k3_aiter_gluon_dcp_support` | 41 | `_gluon_fn()` 为 None（aiter gluon MLA kernel 不可用）时 `RuntimeError`。K3 DCP 用 `aiter` 解码时调用。 |
| `_kimi_k3_overrides` | 52 | 注册给 K3。开启 prefill CP 直接报错（提示 K3.1 支持）。`dcp_size > 1` 时：开了 `enable_symm_mem` 就关掉（decode graph 正确性问题）；DSPARK 要求 ragged verify 为 static（或 compact 且开启 `SGLANG_DSPARK_ENABLE_COMPACT_DP_TIER`，aiter 下 compact 不允许）；DSPARK/EAGLE3 设 `speculative_attention_mode=decode`；NPU 上固定 `dcp_comm_backend=a2a`、`dcp_replicate_q_proj=False` 后返回；decode 后端为 `cutedsl_mla` 或未设时检查 FlashInfer 并设 prefill `trtllm_mla` / decode `cutedsl_mla`；为 `aiter` 时检查 gluon kernel，prefill 除非是 triton 否则设为 aiter；其他后端 `AssertionError`；最后合并 `_dcp_replicate_q_proj_default`。非 DCP 时只在 SM100/SM103 上处理：非 DSPARK 且后端未设，默认 prefill/decode 都是 `trtllm_mla`；DSPARK 按 `q_len`（`speculative_num_draft_tokens`，或 block size + 1，默认 8）判断 decode 后端能否承担 verify，能则设 `speculative_attention_mode=decode`，否则告警。 |
| `_kimi_k31_overrides` | 191 | 注册给 K3.1。`dcp_size > 1` 时合并 `_kimi_k31_dcp_overrides`；开启 `enable_unified_memory` 报错；再合并 `_kimi_k31_lossless_defaults`。 |
| `_kimi_k31_lossless_defaults` | 202 | `enable_flashmla_prefill_token_packing` 未设时默认 True（docstring：只改变稀疏 key 求和顺序，按支持的布局自行门控，确定性模式下不生效）。 |
| `_kimi_k31_dcp_overrides` | 214 | K3.1 DCP 校验。要求 `dcp_size` 是 2 的幂且整除 `tp_size`；unsupported 字典拒绝：DSPARK/DFLASH 以外的投机解码、DFLASH draft attention 后端不是 `cutedsl_mla`、PD 模式下传输后端不是 mooncake、HiSparse、**prefill CP（`enable_prefill_cp` 或 `attn_cp_size > 1`，第 240-241 行）**。然后合并 `_dcp_replicate_q_proj_default`；DFLASH 设 draft 后端 `cutedsl_mla` 和 `speculative_attention_mode=decode`；DSPARK 要求 ragged verify 为 static，并设 `speculative_attention_mode=decode`。 |
| `_dcp_replicate_q_proj_default` | 270 | `dcp_replicate_q_proj` 未设且通信后端为 `a2a`/`fi_a2a` 时默认 True：用复制的 bf16 attention 权重在本地算全部头的 Q，省掉每层 Q 的 all-gather。 |
| `_kimi_k3_moe_runner_overrides` | 280 | 注册给 K3 和 K3.1。`moe_runner_backend` 为 `auto`、平台是 SM100/103/107 且权重是 MXFP4 打包量化时，默认 `flashinfer_mxfp4`。 |

## 4. 与 PCP+DCP 方案 A 的关系

计划 §4.1 要求把 `_kimi_k31_dcp_overrides`（214-260）改为放行 prefill CP，同时保留和补充对 DSA 后端、spec 算法、PD 后端的约束。

**需要注意的点：**
- **第 240-241 行的拒绝条件有两部分**：`enable_prefill_cp` 和 `attn_cp_size > 1`。方案 A 两者都会成立。放行时应该收紧成"方案 A 的组合"（开启 prefill CP、`attn_cp_size == dcp_size == tp_size`、interleave），而不是删掉整项，否则 `attn_cp_size > 1` 但不开 prefill CP 的配置（例如只用 cp-decode-attn-tp 的情况）也会被放行（推断）。
- **执行顺序**：这个 provider 在 pipeline 第 238 行运行，早于 `handle_context_parallelism`（291）里的 `resolve_cp_tp_group_sharing`。在这里读到的 `attn_cp_size` 是用户输入值，或者是同一轮里 `_deepseek_family_overrides` 声明的值之前的值（同一架构的多个 provider 都读 `resolving_view`，彼此看不到对方本轮的声明，推断）。所以方案 A 的完整拓扑检查更适合放在 CP-TP group sharing 里做，这里只放行（推断）。（计划未提及）
- **`model_overrides/deepseek_v2.py` 的 `_deepseek_family_overrides` 也作用于 K3.1**：开启 prefill CP 时，它会设置 `attn_dp_size`（`context_parallel_attn_dp_size`）、`dp_size=1`、`attn_cp_size`，并断言 `tp_size <= 8`；DCP 时检查 `index_kpool > 1` 与 `dcp_size > 1` 不兼容，并设置 page size 相关约束。方案 A 需要确认这些声明和 DCP 同时出现时是否一致，计划的修改清单没有列出这个文件。（计划未提及）
- **DSA 后端**：K3.1 DCP 这里没有设置 `dsa_prefill_backend` / `dsa_decode_backend`（`deepseek_v2.py:105` 对 K3.1 跳过 `_dsa_dcp_overrides`，后者会给其他 DSA 模型设 `trtllm`）。测试 docstring（`test_model_overrides.py:4524`）说明 K3.1 两个 DSA 阶段默认都是 `flashmla_kv`。计划 P1 建议先用 FlashMLA，与默认一致；如果第一期只支持一个后端，需要在这里新增"方案 A 下 DSA 后端必须是 X"的检查（推断）。
- **投机解码**：这里允许 DSPARK 和 DFLASH，而 CP-TP group sharing 下 K3.1 只允许 DFLASH。计划把 DFLASH/DSPARK 放到 P4，第一期方案 A 应在这里显式拒绝投机解码（推断）。
- **PD 后端**：现有的"PD 只能用 mooncake"保留。方案 A 下 prefill 实例的 KV 传输身份会变（计划 §4.10，P4），第一期可以额外拒绝方案 A + PD（推断）。
- **`_dcp_replicate_q_proj_default`（270）**：方案 A 下 MLA 权重本来就是完整复制的（attention TP = 1），全部头的 Q 在本地就能算出来。`dcp_replicate_q_proj` 的语义（去掉头维度的 Q all-gather）在 attention TP = 1 时可能是空操作，需要确认它在方案 A 下不会触发额外的权重处理（推断）。（计划未提及）
- **`_kimi_k31_lossless_defaults`**：FlashMLA prefill token packing 默认开启，docstring 说它"按支持的布局自行门控"。方案 A 改变了每卡的 query 行布局（按位置取模），需要确认 token packing 的门控能识别新布局，或在方案 A 下关闭（推断）。（计划未提及）
- `enable_unified_memory` 在 K3.1 上已被拒绝（196-197），与方案 A 无关。
