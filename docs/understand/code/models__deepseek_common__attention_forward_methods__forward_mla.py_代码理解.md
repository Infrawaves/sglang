# `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py` 代码理解

- 文件路径：`python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py`
- 行数：1295
- 所属模块：DeepSeek 系列 MLA attention 的"absorb"前向实现，以 mixin 形式混入 `DeepseekV2AttentionMLA`（Kimi-K3.1 的 `KimiK3MLAAttention` 继承自它）。

## 1. 整体功能

MLA（Multi-head Latent Attention）把 K/V 压缩成一个 latent 向量（`kv_lora_rank` 维）加一段 RoPE 部分。absorb 写法把 `kv_b_proj` 的 K 部分（`w_kc`）吸收到 Q 上，把 V 部分（`w_vc`）放到 attention 之后，这样 attention 直接在 latent 空间上计算。

本文件的 `DeepseekMLAForwardMixin` 把一次 absorb 前向分成两段：

1. **`forward_absorb_prepare`**：从 attention TP 上下文取出 `q_a`/latent，做 layernorm；算 `q_b_proj` 得到全部本地头的 Q；对 DSA 模型运行 indexer 得到 top-k；做 `q_nope × w_kc` 的 absorb BMM（有多种实现：bf16、fp8、DeepGEMM、GGUF、born-fp8、和 attention 融合）；做 RoPE；在 CP 生效时 gather KV；在 DCP 下 decode 时 all-gather Q，extend 时 gather KV。
2. **`forward_absorb_core`**：调用 attention（普通 `attn_mqa` 或 DCP decode 的 `attn_mqa_for_dcp_decode`）。DCP decode 时按 LSE 合并各 rank 的部分输出（a2a、fi_a2a、push-combine 或 all-gather + reduce-scatter）。然后用 `w_vc` 做 V 投影，应用输出门，最后过 `o_proj`。

与 PCP/DCP 相关的分支：
- **PCP（interleave）**：`DSAInterleaveCPIndexerOverlap`（来自 `layers/cp/interleave.py`）在 DSA + interleave extend 时接管 indexer 运行和 Q/KV 准备；否则 `prepare_kv_for_attention`（来自 `dsa_backend.py`）在 CP 生效时把本卡 KV gather 成完整 KV。
- **DCP**：`is_dcp_mla_decode_phase` 判断是否走"本地 KV 分片 + LSE 合并"；对 Kimi-K3.1（`dsa_shards_index_k()` 为真），extend 不走这条路，而是由 DSA backend 自己 gather 完整 KV。

## 2. 主要依赖与调用方

**依赖：**
- `layers/dcp`：`all_gather_kv_cache_for_mla_extend`、`all_gather_q_for_mla_decode`、`cp_lse_ag_out_rs_mla`、`dcp_a2a_lse_reduce`、`dcp_push_combine_fits`、`dcp_push_combine_reduce`；`layers/dcp/layout.dsa_shards_index_k`。
- `layers/cp/interleave`：`DSAInterleaveCPIndexerOverlap`、`is_interleave_extend`。
- `layers/attention/dsa_backend`：`prepare_kv_for_attention`、`uses_raw_fp8_dsa_kv_layout`。
- `layers/layer_boundary.get_attn_tp_context`（取 `fetch_qkv_latent()`）。
- 各类 GEMM/量化 kernel：`bmm_fp8`、`per_tensor_quant_mla_fp8`、`per_token_group_quant_mla_deep_gemm_masked_fp8`、`absorbed_bmm_concat_cast_q_fp8`、`bf16_gemm_dispatch`、`deep_gemm_wrapper`。
- breakable CUDA graph 工具：`eager_on_graph`、`is_in_breakable_cuda_graph`、`is_in_prefill_graph_scope`、`get_forward_batch`。

**调用方（grep 确认）：**
- `models/deepseek_v2.py:2498, 2581`：`DeepseekV2AttentionMLA` 调用 `forward_absorb_prepare` 和 `forward_absorb_core`。`KimiK3MLAAttention`（`models/kimi_k3.py:2745`）继承它。
- `models/dots3_common/modeling.py:923, 940`：同样的两段调用。
- `hardware_backend/npu/attention/mla_preprocess.py`：NPU 有自己的 prepare 变体。
- 模块属性 `self.dcp_size` 在 `deepseek_v2.py:2115` 由 `mla_attention_dcp_size(config, dcp_size=attn_dcp_size, ...)` 设置；`self.num_local_heads = num_heads // attn_tp_size`（第 2120 行），cp-decode-attn-tp 生效时临时改成 `num_heads // decode_tp_size`（第 2361 行）。

## 3. 类与函数说明

### 3.1 模块级函数与类

| 名称 | 行号 | 作用 |
|---|---|---|
| `is_kv_b_lora_active` | 80 | `kv_b_proj` 上是否挂了 LoRA。 |
| `MlaBmmFusionPlan` | 85 | 冻结 dataclass：absorb BMM 与 attention 融合时预分配的缓冲（`q_nope_t`、BMM 输出缓冲及其转置视图、attention 输出缓冲）。 |
| `_select_local_dcp_heads_for_autotune` | 92 | MoE autotune 的 dummy 前向里，不做 DCP 通信，直接按 `attn_dcp_rank` 截取本 rank 的头。 |
| `is_mla_dcp_lse_base_on_e` | 100 | DCP 合并时 LSE 是自然对数还是以 2 为底。DSA 下：K3.1 的分片 DSA 用 FlashMLA 时是自然对数，trtllm decode 是以 2 为底；`flashmla`、`cutedsl_mla`、`aiter` 是自然对数。 |
| `bcg_dcp_extend_kv_gather` | 109 | 被 `@eager_on_graph(True)` 包装：在 breakable graph 下作为 eager 断点执行。组装 DCP extend 用的完整 KV 缓冲：无前缀时直接把本轮 `k_nope`/`k_pe` 写入 `dcp_kv_buffer`；有前缀时调用 `all_gather_kv_cache_for_mla_extend` 从各 DCP rank gather 前缀 KV。prefill graph 下从 `get_forward_batch()` 取 live batch。 |
| `should_defer_dsa_cp_kv_gather` | 144 | 兼容谓词（ROCm MLA 路径导入）：DSA prefill CP 且融合 RoPE 时返回 `True`。 |
| `_apply_attention_output_gate` | 153 | 调用模型提供的输出门钩子（`apply_attention_output_gate` 或 `_apply_gated`），都没有时抛 `RuntimeError`。 |
| `_finish_absorb_output` | 165 | 应用输出门后过 `o_proj`；如果模块定义了 `next_skip_topk`，同时返回 top-k 供下一层复用。 |

### 3.2 `DeepseekMLAForwardMixin`（第 176 行）

类属性：`needs_unfused_mla_qkv`（子类需要显式 absorbed Q 时设为 `True`，会关闭 BMM 融合和 born-fp8）、`dsa_decode_absorb_under_indexer`（decode capture 时把 `q_b_proj`、absorb 和 KV 写入放到 indexer 下的 alt stream）、`_kv_cache_saved_early`（prepare 阶段已写 KV 的标记）。

| 名称 | 行号 | 作用 |
|---|---|---|
| `_prepare_dcp_v_weights` | 187 | 在 DCP 组内 all-gather `w_vc`，得到所有 DCP rank 的头的 V 权重，用于"先做 V 投影再 LSE 合并"。结果缓存在 `_dcp_v_weights`；权重重载时原地 `copy_`，保证已捕获的 graph 地址不变。 |
| `is_dcp_mla_decode_phase` | 197 | 是否走 DCP 的"本地 KV + LSE 合并"路径。`dcp_size == 1` 返回 `False`；decode、target verify、draft extend v2 返回 `True`；DSA 模型的 extend 只有在 `dsa_shards_index_k()` 为假时才返回 `True`（K3.1 的分片 DSA prefill 改为对 gather 后的完整 KV 做 attention）。 |
| `is_dcp_q_replicate_phase` | 213 | `--dcp-replicate-q-proj` 是否生效：decode 和 target verify 生效；HIP 上 DSA extend 也生效。用本地复制的 `q_b_proj` 直接算出全部头的 Q，省掉 Q 的 all-gather。 |
| `init_mla_forward` | 225 | 读取 `flashinfer_mla_disable_ragged` 配置。 |
| `should_run_indexer` | 230 | 本层是否运行自己的 indexer：非 `skip_topk` 层运行；`skip_topk` 层复用上一层的 top-k；NextN 层在没有上一层 top-k 时运行。 |
| `_can_fuse_bmm_into_attention` | 251 | absorb BMM 能否与 attention 融合成一个 split op。要求：非 unfused、非 GGUF、`dcp_size == 1`、在 prefill graph 的 DSA split-op 表面内、DSA 模型、非 DeepGEMM、无 LoRA、无 sink、`w_kc` 不是 fp8、后端在 `FORWARD_ABSORB_CORE_ATTENTION_BACKENDS` 中。 |
| `_q_b_proj_maybe_replicated` | 282 | 不复制时走普通 `q_b_proj_forward`；复制时走量化的 `q_b_proj_replicated_forward`，或用 gather 后的 bf16 权重直接算出 `num_local_heads × dcp_size` 个头。 |
| `_decode_absorb_on_alt_stream` | 298 | 只有默认 bf16 absorb（无 born-fp8、无 LoRA、无 GGUF、无 DeepGEMM、`w_kc` 为 bf16）才能挪到 alt stream。 |
| `_split_q_nope_pe` | 313 | 把 Q 切成 nope/rope 两段，从 latent 取出 `k_pe`。 |
| `_make_mla_bmm_fusion_plan` | 322 | 为 BMM+attention 融合预分配缓冲，返回 `MlaBmmFusionPlan`。 |
| `_q8kv8_born_fp8_q_backend` | 348 | 判断 born-fp8 Q 快速路径能否运行（absorb BMM、拼接和 fp8 转换合成一个 kernel，直接写入 q8kv8 缓冲）。条件很多，其中包括 `dcp_size == 1`、非 graph、非 capture、kernel 形状约束、后端 `q8kv8_born_fp8_q_eligible` 为真。满足时返回 backend，否则返回 `None`。 |
| `run_mla_indexer` | 414 | 调用 `self.indexer(...)` 运行 DSA indexer，返回 top-k。 |
| `prepare_mla_qkv` | 433 | 非融合 RoPE 路径上对 `q_pe`、`k_pe` 做 RoPE。子类覆写时如果会修改 `q`，必须设置 `needs_unfused_mla_qkv`。 |
| `forward_absorb_prepare` | 450 | absorb 前半段（约 470 行）。主要步骤：取出 Q/latent 并做 layernorm（capture 下用 alt stream 并行）；判断 CP indexer overlap（第 472-480 行）、BMM 融合、Q 复制；decode capture 下在 alt stream 上把 `q_b_proj`/absorb/KV 写入与 indexer 重叠（第 533-610 行）；否则顺序执行 `q_b_proj`、可选的 Q8KV8 q-prep overlap、indexer（CP overlap 时走 `cp_overlap.run_indexer`，第 664-683 行）；无 `q_lora_rank` 的模型走 `q_proj`；按条件选择 absorb 实现（第 708-810 行）；做 RoPE；born-fp8 时写入 fp8 Q 并返回哨兵张量；CP overlap 时调用 `cp_overlap.prepare_qkv`，否则 `prepare_kv_for_attention`（第 856-872 行）；DCP 下 decode 阶段 all-gather Q（未复制时），extend 且非 DSA 时调用 `bcg_dcp_extend_kv_gather`（第 875-896 行）。返回元组交给 `forward_absorb_core`。 |
| `forward_absorb_core` | 918 | absorb 后半段。按后端调用 attention：融合计划走 `_eager_bmm_attention`；DCP decode 阶段走 `attn_mqa_for_dcp_decode`（返回输出和 LSE）；其他走 `attn_mqa`；不在 `FORWARD_ABSORB_CORE_ATTENTION_BACKENDS` 的后端先拼接 Q/K。DCP decode 阶段把输出看成 `num_local_heads × dcp_size` 个头，按 `dcp_comm_backend` 合并：a2a/fi_a2a 下满足条件时先用 `_prepare_dcp_v_weights` 做 V 预投影再 push-combine，否则 `dcp_a2a_lse_reduce`；其他后端用 `cp_lse_ag_out_rs_mla`。合并后每 rank 剩 `num_local_heads` 个头。最后做 V 投影（`project_mla_values`）、LoRA 修正、输出门和 `o_proj`。 |
| `project_mla_values` | 1150 | 用 `w_vc` 把 latent 输出展开成 V（GGUF、DeepGEMM、fp8、MUSA、默认 bf16 BMM 几种实现）。 |
| `_fuse_rope_for_trtllm_mla` | 1241 | 是否跳过单独的 RoPE，改为在 attention kernel 里融合 RoPE+量化：DSA/NSA 后端在 raw-fp8 KV 布局且 KV 为 fp8 时为真；`trtllm_mla`/`cutedsl_mla` 在 decode/verify 且数据类型为 fp8 时为真。 |
| `_eager_bmm_attention` | 1266 | 被 `@eager_on_graph` 包装：先做 absorb BMM，再调用 `attn_mqa._eager_attention.run_eager`，作为 graph 内的一个 eager split op。 |

## 4. 与 PCP+DCP 方案 A 的关系

计划 §4.5 和 §4.6 提到本文件三处：阶段判断 `is_dcp_mla_decode_phase`（第 197-211 行）、`bcg_dcp_extend_kv_gather`（第 109 行）的分支，以及 DCP 合并（计划写 `:1030-1100`，实际在 `forward_absorb_core` 第 1021-1108 行）。

### 4.1 prefill（CP extend）

1. **K3.1 不经过 `bcg_dcp_extend_kv_gather`（计划可能有误）。** 第 884 行 `if not self.use_dsa` 表明 DSA 模型的 DCP extend 不调用它，K3.1 的 extend KV gather 在 DSA backend 内部完成（`dsa_backend.py` 的 `dcp_gather_kv_cache`，计划 §4.5 已列出）。对 K3.1 来说，方案 A 不需要修改 `bcg_dcp_extend_kv_gather`，只有以后支持非 DSA MLA 模型时才需要。
2. **`is_dcp_mla_decode_phase` 对 K3.1 extend 已经返回 `False`。** 第 210 行在 `dsa_shards_index_k()` 为真时排除了 extend，所以 CP extend 下 K3.1 已经走"对 gather 后完整 KV 做 attention"的路径。方案 A 可能不需要改这个判断（推断）；需要改的是 attention 内部让 query 只覆盖本卡行。
3. **CP 和 DCP 两条 KV 路径会同时出现。** CP 生效时，第 856-872 行会调用 `cp_overlap.prepare_qkv` 或 `prepare_kv_for_attention`，后者会调用 CP 策略的 `materialize_full_mla_kv`，把本轮新 token 的 KV gather 成完整一份。DSA backend 在 DCP 下又会 gather 整个序列的 KV。方案 A 要求只写本地、不先 gather 再写（计划 §4.4），所以需要在这两处关掉 CP 的 KV gather，避免重复通信和重复写入。入口就是第 472-480 行 `cp_overlap` 的构造条件和第 866 行 `prepare_kv_for_attention` 的调用（计划在 §4.4 提到 `materialize_full_mla_kv`，但没有指出本文件这两个调用点）。
4. **indexer 的 CP overlap。** 第 666-675 行 CP 生效时改用 `cp_overlap.run_indexer`，它会传入 `k_nope` 和 `latent_cache`，在 indexer 运行期间重叠 KV gather。方案 A 下这个重叠对象要换成 DCP 的整序列 gather，或者先关掉 overlap 走普通 `run_mla_indexer`（计划未提及）。
5. **DCP 下的快速路径都已关闭。** `_can_fuse_bmm_into_attention`（第 258 行）和 `_q8kv8_born_fp8_q_backend`（第 392 行）在 `dcp_size > 1` 时直接返回，方案 A 下不会进入这些路径。

### 4.2 decode

1. **头数语义。** DCP decode 路径假定每卡先算出 `num_local_heads` 个头的 Q，再 all-gather 成 `num_local_heads × dcp_size` 个头（第 878 行），合并后回到 `num_local_heads` 个头（第 1024-1027、1086、1113 行）。方案 A 下 attention TP 为 1，`num_local_heads` 等于全部头数 H。如果不做处理，第 878 行会把 Q gather 成 `H × 8` 个头，第 1024 行的 view 也不对（推断）。
   - 选 **D2**（cp-decode-attn-tp）时，`deepseek_v2.py:2361` 在 decode 时把 `num_local_heads` 临时改成 `H / decode_tp_size`，同时按 CP rank 切分 `q_b_proj`、`o_proj`、`w_kc`、`w_vc`。这样本文件的 DCP decode 逻辑可以保持不变：Q gather 回 H 个头，合并后每卡 H/8 个头，正好对上切过的 `o_proj`。这支持计划把 D2 作为推荐方案。
   - 选 **D1** 时，Q 已经是全部头，需要跳过第 878 行的 all-gather（类似 `q_replicate_active` 的效果），并在合并后再 all-gather 头。
2. **V 预投影权重缓存。** `_prepare_dcp_v_weights`（第 187 行）在第一次 decode 时 gather `w_vc` 并缓存到 `_dcp_v_weights`。D2 下这次调用发生在 `maybe_use_decode_attn_tp` 上下文内，`w_vc` 已被切成 H/8 个头，gather 后得到 H 个头，结果正确（推断）。要注意缓存一旦建立就不再随上下文变化，权重更新时由第 194 行原地覆盖，必须保证覆盖时同样处在切片后的上下文中（计划 §4.6 提到 `dcp_v_preproject` 要按切片后的头准备权重，与此一致）。
3. **`dcp_group` 与 `attn_dcp_rank`。** 第 1067、1081、1097、1105 行使用 `get_parallel().dcp_group`，`_select_local_dcp_heads_for_autotune` 使用 `attn_dcp_rank`。方案 A 下这两者分别要等于 attention CP 组和 `attn_cp_rank`（计划 §4.2），本文件不需要改。
4. **合并后的头布局。** 计划 §4.6 要求确认合并后的头布局和 cp-decode-attn-tp 的切片一致。从代码看，`dcp_a2a_lse_reduce` / `cp_lse_ag_out_rs_mla` 把第 r 段头留给 DCP rank r（推断，依据 `_select_local_dcp_heads_for_autotune` 用 `rank * num_local_heads` 截取）。cp-decode-attn-tp 也要按 CP rank r 取第 r 段头，两者在 `dcp_rank == attn_cp_rank` 时一致。
