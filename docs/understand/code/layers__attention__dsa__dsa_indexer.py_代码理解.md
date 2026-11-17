# `python/sglang/srt/layers/attention/dsa/dsa_indexer.py` 代码理解

- 文件路径：`python/sglang/srt/layers/attention/dsa/dsa_indexer.py`
- 行数：2594
- 所属模块：DSA（DeepSeek Sparse Attention）稀疏 attention 的 indexer 层，为每个 query 选出 top-k 个 key 位置，供后面的稀疏 MLA attention 使用。

## 1. 整体功能

DSA 在每个 MLA 层前加一个轻量 indexer：
- **Q 侧**：`wq_b(q_lora)` 得到 `n_heads × head_dim` 的 indexer query；
- **K 侧**：`wk(x)` 得到单头的 indexer key，经过 `k_norm`、RoPE（Kimi-K3.1 是 NoPE，`skip_rope=True`）和可选的 Hadamard 旋转后，量化成 fp8 写入 indexer K cache（每 token `head_dim + 4` 字节，含 scale）；
- **打分**：每个 query 头对历史 key 做 fp8 MQA logits，用 `weights_proj(x)` 得到的 head gate 加权求和，再由 `metadata.topk_transform` 选出 top-k（`index_topk`）。

`Indexer.forward_cuda` 是主入口，按场景分支：
- idle：返回全 -1；
- 序列长度不超过 `index_topk` 时只写 K cache，不算 logits（`_forward_cuda_k_only`）；
- ROCm gfx950 融合 decode、AITER 融合 fp8 写入、CUDA 融合 Q/K 写入（`_fused_q_prepare_and_store`）；
- prefill 在 PCG/BCG graph 下整个 indexer 作为一个 eager split op；
- 一般路径：`_get_q_k_bf16` 算 Q/K → 量化 Q → `_store_index_k_cache` 写 K → 算 head gate → decode/verify 走 `_get_topk_paged`（按页表读 K cache），prefill 走 `_get_topk_ragged`（把每个请求的 K 拼成 ragged 布局）。

与并行相关的逻辑：
- **PCP**：`_get_q_k_bf16` 在 CP 生效时把本卡算出的 key 通过 CP 策略的 `materialize_full_indexer_k_cache` gather 成完整一份再写 cache。
- **DCP（Kimi-K3.1）**：`dsa_shards_index_k()` 为真时 indexer K 按 token 分片存储，只有 CUDA 融合 store kernel 支持按 owner 写入（`_store_index_k_cache`）；prefill 时 `_dcp_gather_index_k` 从各 rank gather 整个序列的 K；decode 时 `_get_topk_paged` 先算本地 logits，再用 `dcp_global_topk` 合成全局 top-k。
- **TP 行切分（K3.1）**：prefill 行数不少于 1024 时，把 logits 和 top-k 的行切给 attention TP 各 rank 计算，再 all-gather（`_indexer_tp_split`）。CP 下不启用。

## 2. 主要依赖与调用方

**依赖：**
- `layers/attention/dsa/dsa_dcp_sharded`：`dcp_gather_request_rows`、`dcp_global_topk`、`dcp_all_visible`；`dsa_dcp`：`dcp_split_rows`、`dcp_all_gather_rows_`（ROCm 行切分）。
- `layers/dcp/layout`：`dsa_shards_index_k`、`get_dcp_lens`。
- `layers/cp/base.get_cp_strategy`、`layers/cp/utils.is_cp_active`。
- `layers/attention/dsa/dsa_prefill_cuda_graph`：`bcg_dsa_indexer_prefill_split`、`pcg_dsa_indexer_prefill_split`。
- kernel：`fused_store_index_k_cache`、`fused_k_indexer_norm_rope(_store)`、`fused_q_indexer_rope_first_quant`、`indexer_q_quant_head_gate`、DeepGEMM 的 fp8 MQA logits、CuTe DSL / AITER paged MQA logits 等。
- `forward_context.get_attn_backend().get_indexer_metadata(...)` 提供 `BaseIndexerMetadata`（页表、序列长度、`ks`/`ke`、`topk_transform` 等）。

**调用方（grep 确认）：**
- `models/deepseek_v2.py:73`、`models/dots3_common/modeling.py:56` 导入 `Indexer` 并在 MLA attention 中构造；`forward_mla.py` 的 `run_mla_indexer` 调用 `self.indexer(...)`。
- `layers/cp/interleave.py` 的 `DSAInterleaveCPIndexerOverlap` 在 CP interleave extend 下直接调用 indexer 的 CUDA 入口。
- `layers/attention/dsa/dsa_indexer_kpool.py` 从本文件导入部分函数；`layers/attention/dsv4/compressor_v2.py:278` 导入 `rotate_activation`。

## 3. 类与函数说明

### 3.1 模块级函数

| 名称 | 行号 | 作用 |
|---|---|---|
| `_decline_fused` | 112 | gfx950 融合 indexer 拒绝某次调用时，每种原因只记录一次日志。 |
| `broadcast_indexer_topk_from_rank0_` | 213 | 仅 CUDA：注册为 custom op 和 split op 的 top-k 广播，供 piecewise graph 使用。 |
| `_broadcast_indexer_topk_from_rank0_impl` | 217 | 在 attention TP 组内从 rank 0 广播 top-k；capture 中且有 PyNCCL 时用 PyNCCL，否则用进程组广播。 |
| `_broadcast_indexer_topk_from_rank0` | 237 | 只在设置 `SGLANG_DSA_TOPK_BROADCAST` 时广播最终 top-k，保证各 rank 选择一致。 |
| `_make_eager_idle_topk_result` | 252 | eager idle rank 返回形状正确的全 -1 top-k，保证后续 MLP/EP 集合通信形状一致。 |
| `_index_k_write_targets` | 265 | 返回 indexer K 的写入目标 `(buffer, slots, rows)`；开启 KV cache sharding 时由 shard pool 给出本卡要写的行。 |
| `_index_k_dcp` | 273 | K3.1 按 token 分片 indexer K 时返回 `(attn_dcp_size, attn_dcp_rank)`，否则返回 `(1, 0)`（其他模型在 DCP 下 indexer K 仍复制）。 |
| `rotate_activation` | 282 | Hadamard 变换（按平台选实现），要求最后一维是 2 的幂。 |

### 3.2 `Indexer(DSANPUIndexerMixin, BaseFusedOp)`（第 297 行）：构造与 head gate

| 名称 | 行号 | 作用 |
|---|---|---|
| `_mqa_logits_free_mem_fraction` | 306 | 静态方法：MQA logits 可用的空闲显存比例。 |
| `__init__` | 309 | 读取 config（rope 维度、Hadamard、bf16 head gate、k_norm 类型）；决定是否用 CUDA indexer 融合（要求非 neox 且有 RoPE）和 AITER 融合 fp8 写入；创建 `wq_b`、`wk`/`weights_proj` 或合并的 `wk_weights_proj`、`k_norm`、`rotary_emb`；K3.1 设置 `tp_split_min_rows = 1024`；`dsa_shards_index_k()` 为真且配置了 `index_init_tokens`/`index_local_tokens` 时抛 `ValueError`；解析 paged MQA logits 后端；判断 gfx950 融合 indexer 能否启用并预分配工作区。 |
| `_with_real_sm_count` | 597 | PP 非最后一级有一个 recv 占用 SM 时，把 DeepGEMM 可用 SM 数减 1。 |
| `_indexer_cos_sin_cache` | 612 | 属性：RoPE 的 cos/sin cache。 |
| `prepare_merged_k_gate` | 615 | 加载权重后、graph capture 前调用：在 CUDA、bf16 head gate、NoPE、非融合、权重为 bf16 且 tiny_gemm 可用时，把 `wk` 和 `weights_proj` 的权重拼成一个 `[wk | weights_proj]` 张量，decode 时一次 GEMM 同时得到 key 和 head gate。 |
| `_project_head_gates` | 637 | 计算 head gate `weights_proj(x)`；CUDA 上直接 `torch.mm`（非 bf16 gate 时输出 fp32）；AITER 下取 3 元组中的 bf16 张量。 |
| `_project_and_scale_head_gates` | 656 | 按 `bf16_head_gate` 选择 bf16 或 fp32 版本，结果乘 `n_heads^-0.5`。 |
| `_get_logits_head_gate` | 663 | 计算最终 logits 权重：head gate × q_scale × softmax_scale。 |
| `_bf16_head_gate` | 672 | bf16 head gate，故意不用 `torch.compile`，以保留中间的 bf16 舍入。 |
| `_project_and_scale_head_gates_fp32` | 679 | `torch.compile` 版本的 fp32 head gate。 |
| `_get_logits_head_gate_fp32` | 687 | `torch.compile` 版本：head gate 乘 q_scale 和 softmax_scale。 |
| `_apply_q_scale_and_softmax_scale` | 696 | `torch.compile`：对已算出的 gate 乘 q_scale 和 softmax_scale。 |
| `_scale_head_gates` | 702 | `torch.compile`：对融合 GEMM 得到的原始 gate 做全部缩放。 |
| `_fused_k_weights` | 706 | 合并 GEMM 后拆出 key 和 gate。 |
| `_k_gate_gemm` | 709 | 合并 GEMM：融合模式用 `wk_weights_proj`；否则行数不超过 16 用 `tiny_gemm_bf16`，更多用 `torch.mm`。 |
| `_split_k_weights` | 718 | 把合并输出按 `[head_dim, n_heads]` 切开。 |

### 3.3 ROCm 融合路径

| 名称 | 行号 | 作用 |
|---|---|---|
| `_aiter_fused_fp8_active` | 722 | AITER 融合 fp8 写入是否可用：CP 生效或有 `attn_cp_metadata` 时不可用，层切分池也不可用。 |
| `_aiter_fused_fp8_rope_caches` | 730 | 按 (device, dtype) 缓存 AITER 用的 cos/sin 表。 |
| `_aiter_fused_fp8_prepare_and_store` | 746 | 一个 AITER kernel 完成 Q 的 RoPE 与 fp8 量化、head gate 缩放、K 的 norm/RoPE/量化写 cache，返回 `(q_fp8, weights)`。 |
| `_maybe_rotate` | 811 | CUDA/HIP 融合写入不做 Hadamard，回退路径也跳过 Hadamard 以保持格式一致；否则调用 `rotate_activation`。 |
| `_gfx950_fused_decode` | 822 | gfx950 融合 decode indexer（4 个 kernel）。任何条件不满足（非 decode/verify、强制非融合 top-k、CP 元数据存在、无 bf16 输入、LoRA、页大小不符、层切分池、行数超限、页表形状不符等）就返回 `None`，让调用方走标准路径。 |

### 3.4 Q/K 计算与 K cache 写入

| 名称 | 行号 | 作用 |
|---|---|---|
| `_should_skip_logits_computation` | 964 | 是否可以不算 logits：prefill 时最长序列不超过 `index_topk`；decode capture 时看 capture 变体是否为 `DSA_DENSE`；eager decode 只在 ROCm 上按长度判断。 |
| `_get_q_k_bf16` | 999 | 计算 bf16 的 indexer Q 和 K：`wq_b`、`wk`（或融合 GEMM）、`k_norm`、RoPE、Hadamard。dual stream 时 Q/K 分在两条流上；CP 生效时在 alt stream 上调用 `get_cp_strategy().materialize_full_indexer_k_cache` 把 key gather 成完整一份（第 1075-1095 行）。返回 `(query, key, weights_raw)`。 |
| `_get_k_bf16` | 1098 | 只算 K（k-only 路径），用一个不与 key 共享内存的 dummy query 做 RoPE。 |
| `_fused_k_prepare_and_store` | 1136 | 融合的 K norm+RoPE+写 cache。NoPE 时直接 `k_norm` 后交给 `_store_index_k_cache`；`dsa_shards_index_k()` 为真时不走融合 store，改走 `_store_index_k_cache` 的 owner 规则；否则可用时调用 `fused_k_indexer_norm_rope_store`。 |
| `_fused_q_prepare_and_store` | 1200 | 融合路径的 Q/K：合并 GEMM 得到 key 和 gate，写 K cache，`wq_b` 得到 Q 后 `_fused_q_quant`。有 alt stream 时两段 GEMM 和两段融合 kernel 分别重叠。 |
| `_fused_q_quant` | 1271 | 融合的 Q 量化与 head gate：NoPE 用 `indexer_q_quant_head_gate`，否则用 `fused_q_indexer_rope_first_quant`。 |
| `_update_rope_guarded` | 1287 | 把 RoPE 结果写回原张量；源和目标是同一块内存时跳过（AMD 原地 RoPE）。 |
| `_get_index_k_read_buffer` | 1296 | 读路径的 indexer K 缓冲：层切分池用 owner 广播后的缓冲，否则用本地缓冲。 |
| `_store_index_k_cache` | 2125 | 写 indexer K cache。CUDA 融合 store 可用时：DCP 分片（`_index_k_dcp()` 的 size > 1）时调用带 `dcp_size`/`dcp_rank` 的 `fused_store_index_k_cache`，由 kernel 把放大后的写入位置映射到本 rank 的行并跳过其他 rank 的 token；否则按 `_index_k_write_targets` 写。不能用融合 store 时断言 `dcp_size == 1`，再走 AITER 或 `act_quant + set_index_k_scale_buffer` 回退路径。 |

### 3.5 打分与 top-k

| 名称 | 行号 | 作用 |
|---|---|---|
| `_pad_heads_for_deep_gemm` | 1305 | 头数少于 32 时把 Q 和权重 pad 到 32 头，满足 DeepGEMM 的对齐要求。 |
| `_mask_init_and_local_tokens` | 1316 | 把每行开头 `num_init_tokens` 个和末尾 `num_local_tokens` 个位置的 logits 设为 +inf，强制选中。 |
| `_get_topk_paged` | 1350 | decode / target verify / draft extend v2 的 top-k：按页表从 K cache 读 key，选择 AITER、XPU、CuTe DSL、DeepGEMM native 或 split 实现计算 paged MQA logits。DCP 分片时断言只用于 decode/verify，logits 和长度都是本 rank 的，再用 `dcp_global_topk` 合成全局 top-k；否则 `topk_transform`。最后为 padding 行补 -1。内部闭包 `_chunked_fp8_paged_mqa_logits`（第 1481 行）在 batch 大于 SM 数时按 SM 数分块计算。 |
| `_get_mqa_logits_budget_bytes` | 1628 | 每个设备测一次 MQA logits 的显存预算并缓存；capture 中用静态值且不缓存。 |
| `_get_topk_ragged` | 1644 | prefill（无投机）的 top-k：取逻辑页表；`dsa_shards_index_k()` 为真时用 `_dcp_gather_index_k` 拿整个序列的 K，否则从本地 cache 读（第 1708-1731 行）；ROCm 的 DCP 复制模式按 1/W 行切分；否则可能用 `_indexer_tp_split` 按 TP 切行；显存不够时按行分块。每块算 fp8 MQA logits、mask、`topk_transform`（ragged 用全局 `topk_indices_offset`，paged 用每 token 的 cu_seqlens）。最后在切分时 all-gather 结果。 |
| `_dcp_gather_index_k` | 1925 | prefill DCP：先按 `get_dcp_lens` 读本 rank 存的那部分 K（页表是本 rank 的页 id），把 fp8 K 和 scale 拼成字节行，用 `dcp_gather_request_rows` 按请求从所有 rank gather，得到和 DCP=1 时 `get_index_k_scale_buffer` 相同的布局，这样后面的 ragged 打分不用改。 |
| `_indexer_tp_split` | 1953 | 决定是否按 attention TP 切分 prefill 行：需要 CUDA、K3.1、非 prefill CP、非 CP 生效、非 capture、TP > 1、行数达到阈值且 PyNCCL 可用。返回 `(group, lo, hi, sizes)` 或 `None`。 |
| `_indexer_tp_split_bounds` | 1991 | 按每行的 KV 长度（打分代价）平衡地切出 TP 个连续行区间，纯 CPU 计算；输入不一致时退回按行数均分。 |

### 3.6 前向入口

| 名称 | 行号 | 作用 |
|---|---|---|
| `_forward_cuda_k_only` | 2017 | 只写 K cache、不算 logits 的快速路径（eager 和 graph split op 共用）。写完后：不需要 indices 时返回 `None`；DCP 分片的 decode/verify 用 `dcp_all_visible` 返回所有可见位置；否则用全零 dummy logits 走 `topk_transform`。`topk_result` 非空时原地填入（graph 的静态缓冲）。 |
| `forward_native` | 2227 | 抛 `NotImplementedError`：indexer 没有纯 torch 实现。 |
| `forward_xpu` | 2232 | 转发给 `forward_cuda`。 |
| `topk_output_width` | 2246 | 属性：top-k 输出宽度，即 `index_topk`。 |
| `forward_cuda` | 2249 | 主入口，分支见第 1 节。注意：prefill CP（`dsa_enable_prefill_cp`）下不走 k-only 快速路径；K3.1 的合并 GEMM 路径（`_wk_gate_weight`）只在 decode/verify 且 CP 未生效时使用；graph 内 prefill 必须走 split op，否则断言失败；`seq_lens` 为空时返回全 -1。 |

## 4. 与 PCP+DCP 方案 A 的关系

计划 §4.5 列出本文件的 `_dcp_gather_index_k`（计划写 `:1930`，函数定义在第 1925 行）和"打分入口 `:1700-1730`"（`_get_topk_ragged` 中第 1701-1731 行），要求"indexer 的 query 用本卡的行，KV 用 gather 来的完整序列，causal 按全局位置计算"。

### 4.1 prefill：方案 A 下 K3.1 走的路径

K3.1 的 indexer 是 NoPE（`skip_rope=True`），所以 `use_dsa_indexer_fusion` 为假（第 347-352 行要求有 RoPE）。prefill 时 `_wk_gate_weight` 分支也不走（只用于 decode/verify），因此 CP extend 会进入 `forward_cuda` 的一般分支（第 2443 行起）：

1. `_get_q_k_bf16`：本卡行算 Q、K，CP 生效时把 K gather 成整批新 token（第 1075-1095 行）。
2. `_store_index_k_cache`：用 gather 后的完整 K 和 `forward_batch.out_cache_loc` 写入，DCP 分片下融合 store kernel 只保留本 rank 拥有的 token。
3. `_get_topk_ragged` → `_dcp_gather_index_k`：从 cache 读本 rank 那份（前缀加本轮），all-gather 成整个序列，再对本卡的 query 行打分。

对照计划：

- **"只写本地"需要改第 1094-1095 行的 CP gather 和写入用的 `out_cache_loc`。** 现在 `prepare_cp_forward` 不把 `out_cache_loc` 缩成本卡行（`layers/cp/utils.py:303-304`），所以 key 必须先 gather 成整批才能和 `out_cache_loc` 对齐。方案 A 去掉 gather 时，要同时把写入用的 `out_cache_loc` 换成本卡行对应的子集。注意按位置取模切分后，本卡算的 token 正好是它拥有的 token，融合 store 的 owner 过滤会全部保留。
- **保留现有 gather 作为 P1 的过渡（计划未提及，推断）。** 如果暂时保留第 1095 行的 CP gather，融合 store 的 owner 过滤本身就能保证每个位置只由 owner 写入，存储结果也正确，只是多一次新 token 的 gather。P1 先求正确性时可以考虑这样做，减少一处改动。
- **打分元数据必须描述本卡行。** `_get_topk_ragged` 用 `q_fp8` 的本地行，同时依赖 `metadata` 的 `ks`/`ke`、`get_seqlens_expanded()`、`get_token_to_batch_idx()`、`attn_metadata.topk_indices_offset`，并在第 1807、1826 行断言这些长度和 logits 行数一致。"按全局位置做 causal"实际上要在 DSA backend 的 `get_indexer_metadata` 里实现：`ks`/`ke` 要对应本卡行的全局位置，并落在 `_dcp_gather_index_k` 返回的"DCP=1 布局"上。本文件的打分代码本身可能不用改（推断）。
- **`_index_k_dcp`（第 273 行）和 `_dcp_gather_index_k`（第 1934 行）读取 `attn_dcp_rank`。** 方案 A 要求它等于 `attn_cp_rank`（计划 §4.2），否则 owner 过滤与 CP 切分对不上，结果会静默算错。
- **TP 行切分自动关闭。** `_indexer_tp_split` 在 `dsa_enable_prefill_cp` 或 CP 生效时返回 `None`（第 1961 行），方案 A 下 prefill 不会再切行，不需要改。
- **k-only 快速路径。** `forward_cuda` 第 2313 行在 prefill CP 下不走 `_forward_cuda_k_only`。方案 A 下这一点保持不变即可；如果以后放开，`_forward_cuda_k_only` 的 extend 分支用的是 dummy logits + `topk_transform`，它的行数要和本卡 query 行一致（计划未提及）。
- **graph 下的 prefill。** `is_graph_dsa_split_op_surface` 分支（第 2380 行）会把整个 indexer 交给 `bcg_dsa_indexer_prefill_split` / `pcg_dsa_indexer_prefill_split`。计划 P1、P2 先禁用 prefill graph，所以这条路径暂时不涉及；以后开启时 `dsa_prefill_cuda_graph.py` 也要支持方案 A 的元数据（计划 §4.8 只提到静态尺寸）。

### 4.2 decode

decode 时 CP 不生效，`_get_topk_paged` 的 DCP 路径（本地 logits + `dcp_global_topk`）和 `_forward_cuda_k_only` 的 `dcp_all_visible` 都按现有 DCP 方式工作，本文件不需要为方案 A 改动。前提同样是 `attn_dcp_rank == attn_cp_rank`，并且 indexer K 的页表是本 rank 的页 id。

### 4.3 其他约束

- `__init__` 第 470-474 行：DCP 分片 indexer K 时不支持 `index_init_tokens` / `index_local_tokens`，方案 A 继承这个限制。
- 第 2189 行：DCP 分片下只有 CUDA 融合 store 能写 indexer K，非 CUDA 平台和 fp8 fnuz 平台不支持方案 A。
- AITER 融合写入（`_aiter_fused_fp8_active`）和 gfx950 融合 decode 在 CP 下都会关闭，只影响 ROCm。
