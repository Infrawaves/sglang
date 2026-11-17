# `python/sglang/srt/layers/attention/dsa_backend.py` 代码理解

- 文件路径：`python/sglang/srt/layers/attention/dsa_backend.py`
- 行数：5236
- 所属模块：DeepSeek Sparse Attention（DSA）的 attention 后端，供 DeepSeek-V3.2 / GLM-5 / Kimi-K3.1 等 DSA 模型的 MLA 层使用。

## 1. 整体功能

### 1.1 在 sglang 里的职责

`DeepseekSparseAttnBackend` 是一个 `AttentionBackend` 子类，由 `attention_registry.py:184-186` 在 `--attention-backend dsa` 时创建。它负责三件事：

1. 每个 forward 前构建一份 `DSAMetadata`（序列长度、page table、DSA 稀疏长度、indexer 需要的 `ks/ke`、FlashMLA/DeepGEMM 调度元数据等），所有层共用。
2. 每层的 `forward_extend` / `forward_decode` 中：写 KV cache，把 indexer 给出的 top-k 下标转换成 KV slot 表，然后按选定的实现调用具体 kernel。
3. CUDA graph 的捕获和回放：预分配静态缓冲区，回放前原地刷新元数据。

稀疏选择本身（indexer 打分和 top-k）不在本文件，在 `layers/attention/dsa/dsa_indexer.py`；本文件通过 `get_indexer_metadata()` 把元数据交给 indexer，再消费它输出的 `topk_indices`。

### 1.2 元数据

- `DSAMetadata`（`:371`，frozen dataclass）：主元数据。分三组字段：
  - 稠密字段：`cache_seqlens_int32`、`cu_seqlens_q/k`、`max_seq_len_q/k`、`page_table_1`（page_size=1 的 slot 表，融合 decode graph 下可以为 `None`）、`real_page_table`（按真实页编号）。
  - DSA 字段：`dsa_cache_seqlens_int32`（每个 query 行的有效 KV 长度，截断到 `index_topk`）、`dsa_cu_seqlens_q/k`、`dsa_seqlens_expanded`（每个 query 行未截断的 causal 长度）、`dsa_extend_seq_lens_list`。
  - prefill 专用：`page_table_1_flattened`（按请求拼接的 slot 序列）、`topk_indices_offset`（RAGGED top-k 的每 token 偏移）、`dcp_local_kv_rows`（DCP prefill 下本卡持有的 KV 行）、`indexer_k_start_end`、`indexer_seq_lens(_cpu)`、`token_to_batch_idx`；以及 KPool 相关的 `pooled_*`、`kpool_*_plan`。
- `DSAFlashMLAMetadata`（`:353`）：FlashMLA 的 tile 调度元数据和 `num_splits`。
- `_RowPlan`（`:221`）：query 行数和元数据行数不一致时（DP/MLP-sync padding、CP 物理分片 padding）的对齐计划。

### 1.3 各阶段流程

- **构建元数据**：`init_forward_metadata`（`:1448`）按 forward mode 分四支：
  - decode/idle：每请求 1 个 query 行。
  - target verify：每请求 `speculative_num_draft_tokens` 行，page table 按行 `repeat_interleave`。
  - draft extend v2：类似 target verify。
  - extend（prefill）：`seqlens_expanded` 为每个新 token 的 causal 长度；若 `can_dsa_prefill_cp_interleave` 成立，用 CP 策略的 `shard_local_tokens` / `shard_per_request` 只保留本 CP rank 的 token 和请求（`bs_idx`）；DCP（`attn_dcp_size > 1`）下总是构建 `page_table_1_flattened` 和 `dcp_local_kv_rows`。
  - 最后统一算 `dsa_cache_seqlens_int32`（再用 `pad_dsa_cache_seqlens` 补到 padding 行数）、DeepGEMM paged MQA 调度、top-k v2 plan、FlashMLA 元数据，并交给 KPool mixin 补字段。
- **prefill（`forward_extend`，`:2538`）**：先选实现（target verify / draft extend 用 decode 实现）。trtllm/prims_ts 直接进 `_forward_trtllm`。其他实现先 `store_mla` 写 KV；若 `use_mha`（短序列的 MHA one-shot）走 `_forward_standard_mha`；否则读 KV cache，DCP 下（非 target verify）用 `dcp_gather_kv_cache` 拿整批请求的完整 KV，再把 top-k 转成 slot 表（fused top-k 直接用；RAGGED 加 `topk_indices_offset`；PAGED 用 `transform_index_page_table_prefill`），最后按实现分派。
- **decode（`forward_decode`，`:2961`）**：trtllm/prims_ts 进 `_forward_trtllm`；其他实现写 KV、转换 top-k（HiSparse 换入 / fused / `transform_index_page_table_decode`），再分派。DCP decode 返回 `(o, lse)` 部分结果，由模型侧做 LSE 合并。
- **target verify**：在 `forward_extend` 中处理，但用 decode 实现；DCP 下与 decode 一样只看本卡 KV、返回部分结果（`:2636-2639`、`:2909`、`:4453`）。

### 1.4 子实现（`_DSA_IMPL_T`，`:469`）

| 实现 | 入口 | 说明 |
|---|---|---|
| `flashmla_sparse` | `_forward_flashmla_sparse` | FlashMLA 稀疏 prefill/decode，bf16 KV（fp8 池先反量化），支持 attn sink |
| `flashmla_sparse_q8` | `_forward_flashmla_sparse_q8kv8` | SM90 FP8 q×kv 稀疏 prefill，只做 RAGGED extend |
| `flashmla_kv` | `_forward_flashmla_kv` | FlashMLA 读 fp8 分页 KV cache，支持 DCP 部分结果 |
| `flashinfer_sparse_mla` | `_forward_flashinfer_sparse_mla` | SM120 FlashInfer 稀疏 MLA |
| `triton_sparse_mla` | `_forward_triton_sparse_mla` | Triton 稀疏 prefill（可选 union 优化） |
| `fa3` | `_forward_fa3` | FA3，每个 query 当作单 token 序列 |
| `tilelang` | `_forward_tilelang` | TileLang 稀疏 kernel |
| `triton` | `_forward_triton_decode` / `_forward_triton_dcp` | Triton split-K decode；ROCm 上有 replicated-index-K DCP 版 |
| `trtllm` | `_forward_trtllm` | FlashInfer trtllm-gen 稀疏 MLA，原始 FP8 KV 布局（576 B/token） |
| `prims_ts` | `_forward_prims_ts`（经 `_forward_trtllm`） | 与 trtllm 同一路由，kernel 换成 Prims-TS |
| `aiter` | `_forward_aiter` / `_forward_aiter_extend` | ROCm aiter `mla_decode_fwd` |
| `intel_xpu` | `_forward_intel_xpu_*` | XPU 的稀疏 decode、稠密/稀疏 prefill |
| `flashmla_auto` | `set_dsa_prefill_impl` | 配置值，运行时按 KV dtype 和 KV/Q 比例选 `flashmla_sparse` 或 `flashmla_kv` |

`uses_raw_fp8_dsa_kv_layout`（`:484`）标记 trtllm/prims_ts 使用不带 tile scale 的原始 FP8 布局，KV 池配置（`kv_cache_configurator.py:3612`）和模型前向据此决定格式。

### 1.5 CUDA graph 支持

- `init_cuda_graph_state`（`:1878`）预分配 `decode_cuda_graph_metadata`：`cache_seqlens`、`cu_seqlens_q/k`、宽 `page_table` 或紧凑 `real_page_table`（`dsa_drop_wide_page_table` 成立时省掉宽表），以及 trtllm/flashmla_kv 的 head-padding q 缓冲和 multi-CTA 计数器。
- 捕获时 `_apply_cuda_graph_metadata` 发现 `bs` 未缓存，就调用 `_build_forward_metadata_cuda_graph` 为该 bs 建一份 `DSAMetadata`；回放时同一函数用 `copy_` 原地刷新（优先用 mixin 里的融合 kernel `_fused_decode_metadata` / `_fused_verify_metadata` / `_fused_draft_extend_metadata`），并在图外刷新 DeepGEMM 调度和 top-k v2 plan。
- MTP 多步 draft 用 `DeepseekSparseAttnMultiStepBackend`：一次 `_precompute_replay_metadata`，再拷贝到每步的后端（≥4 步时用 `fused_metadata_copy_multi_cuda` 一次拷三个）。
- prefill graph（breakable / piecewise）下 `set_dsa_prefill_impl` 强制关闭 MHA one-shot；`_plan_live_rows` 允许 breakable graph 下元数据行多于 q 行。

## 2. 主要依赖与调用方

### 2.1 关键依赖

| 依赖 | 用途 |
|---|---|
| `dsa/dsa_metadata_manager.py`（`DSAMetadataManagementMixin`） | 融合元数据 kernel、回放拷贝、sibling 拷贝 |
| `dsa/dsa_backend_kpool.py`（`DeepseekSparseAttnBackendKPoolMixin`） | index KPool（`dsa_index_kpool > 1`）的元数据和尾部后端选择 |
| `dsa/dsa_backend_mtp_precompute.py` | `PrecomputedMetadata`、`_precompute_replay_metadata`、`compute_cu_seqlens` |
| `dsa/dsa_dcp_sharded.py` | `dcp_gather_request_rows`、`dcp_localize_compact`、DCP top-k 合并初始化 |
| `dsa/dsa_dcp.py` | replicated-index-K DCP：`localize_dcp_page_table`、`dcp_trtllm_sparse_attention` |
| `dsa/dsa_topk_backend.py` | `DSATopKBackend`、`TopkTransformMethod`（RAGGED / PAGED） |
| `dsa/utils.py` | `dsa_use_prefill_cp`、`can_dsa_prefill_cp_interleave`、`is_dsa_enable_prefill_cp`、`pad_dsa_cache_seqlens` 等 |
| `layers/cp/`（`get_cp_strategy`、`is_cp_active`、`cp_local_rope_positions`、`cp_live_store_rows`） | prefill CP 的切分、KV gather |
| `layers/dcp/layout.py` | `dsa_shards_index_k`、`get_dcp_lens`、`filter_dcp_local_chunk_kv_indices` |
| `sglang.kernels.ops.attention.*`、`sgl_kernel.flash_mla`、`flashinfer`、`aiter` | 具体 kernel |

### 2.2 调用方（grep 确认）

- `layers/attention/attention_registry.py:184-186`：创建 `DeepseekSparseAttnBackend`。
- `speculative/draft_utils.py:208-214`：创建 `DeepseekSparseAttnMultiStepBackend`。
- `hardware_backend/sail/attention/dsa_backend.py:45,97`：Sail 硬件子类。
- `models/deepseek_common/attention_forward_methods/forward_mla.py:17-19,573,866,1251`：`prepare_kv_for_attention`、`uses_raw_fp8_dsa_kv_layout`。
- `models/deepseek_common/attention_forward_methods/forward_mha.py:12,262,699`：MHA 路径在 DCP 下调用 `backend.dcp_gather_kv_cache`。
- `layers/cp/interleave.py:454-469,490-529`：`DSAInterleaveCPIndexerOverlap` 调用 `uses_raw_fp8_dsa_kv_layout` 和 `prepare_kv_for_attention`。
- `model_executor/runner/prefill_cuda_graph_runner.py:345-349`：判断是否 DSA 后端。
- `mem_cache/kv_cache_configurator.py:3612-3626`：决定 KV 池布局。
- 只引用类型的：`dots_hybrid_backend.py`、`dsa/kpool_plan.py`、`dsa/dsa_indexer_metadata.py`、`dsa/dsa_backend_kpool.py`、`dsa/dsa_metadata_manager.py`；测试 `test/kits/dsa_metadata_kit.py`。

## 3. 类与函数说明

本文件没有 triton kernel（kernel 都从 `sglang.kernels.ops` 导入）。

### 3.1 模块级函数与常量

| 名称 | 行号 | 作用 |
|---|---|---|
| `prepare_kv_for_attention` | 151 | 处于 DSA prefill CP 且 CP 生效、且未要求延迟时，调用 CP 策略的 `materialize_full_mla_kv` 把本轮新 token 的 `k_nope/k_pe` all-gather 成完整一份；否则原样返回。 |
| `_to_2d_context_lens` | 209 | 把 context 长度规范成 `(N_total, 1)`，避免 DeepGEMM `fp8_paged_mqa_logits` 死锁。 |
| `_plan_live_rows` | 233 | 比较 q 行数和元数据行数，返回 `_RowPlan`；元数据多于 q 时只在 CP 元数据、DSA padding 或 breakable graph 下允许，否则断言失败。 |
| `_apply_q_row_plan` | 265 | 按计划裁掉多余 q 行，或给缺的行补零 q 和 -1 top-k。 |
| `_restore_live_rows` | 294 | 把输出还原到调用方的 q 行数：去掉元数据 padding 行，或给被裁掉的 q padding 行补零。 |
| `_trtllm_padded_heads` | 310 | trtllm-gen 不接受 64<heads<128（K3.1 DCP8 为 96），补到 128。 |
| `_adjacent_absorbed_q` | 315 | 若 `q_nope` 和 `q_rope` 是同一连续缓冲的两半，返回拼好的零拷贝视图，否则 `None`。 |
| `_flashmla_kv_padded_heads` | 334 | SM100 且 kernel 支持原生头数时不补，否则补到 64 或 128。 |
| `_compiled_cat` | 449 | `torch.compile` 包装的 `torch.cat`。 |
| `_cat` | 453 | 拼接 nope/rope 两个 3 维张量，标记动态维后调用 `_compiled_cat`。 |
| `_DSA_IMPL_T` | 469 | 实现名的类型别名。 |
| `uses_raw_fp8_dsa_kv_layout` | 484 | trtllm 和 prims_ts 返回 True，表示使用原始 576 B/token 的 FP8 KV 布局。 |
| `_TRITON_SPARSE_MLA_MAX_HEADS` | 491 | triton_sparse_mla 验证过的每卡最大头数（32）。 |
| `_validate_triton_sparse_mla_backend` | 494 | 校验 triton_sparse_mla：非 ROCm、SM90+、头数 ≤32、不支持 KPool、union 合法。 |
| `_validate_dcp_backend` | 539 | sharded DCP 校验：只支持 K3.1、要求 fused top-k 和 top-k v2、prefill/decode 实现必须同为 `flashmla_kv`、`trtllm` 或 `prims_ts`。 |
| `_validate_prims_ts_dcp` | 570 | prims_ts 在 DCP 下只支持 sharded index-K（LSE 以 e 为底）。 |
| `_resolve_dsa_triton_union` | 581 | 确定性推理下关闭 triton union。 |
| `_validate_flashmla_sparse_q8_backend` | 596 | 校验 flashmla_sparse_q8：KV dtype、NoPE、SM90、不能和 mixed chunk / HiSparse 组合、不能作 decode 后端。 |

### 3.2 数据类

| 名称 | 行号 | 作用 |
|---|---|---|
| `_RowPlan` | 221 | 记录 `live_rows`、`q_pad_rows`、`meta_pad_rows`。 |
| `DSAFlashMLAMetadata` | 353 | FlashMLA 调度元数据容器。 |
| `DSAFlashMLAMetadata.slice` | 359 | 按 slice 截 `num_splits`，调度元数据共享。 |
| `DSAFlashMLAMetadata.copy_` | 365 | 原地拷贝另一份元数据（graph 回放用）。 |
| `DSAMetadata` | 371 | 每次 forward 的主元数据，字段见 §1.2。 |

### 3.3 `DeepseekSparseAttnBackend`：初始化与内部缓冲

| 名称 | 行号 | 作用 |
|---|---|---|
| 类属性（`extend_dummy_seqs_capped_by_req_pool` 等） | 675-686 | 声明 extend padding 行规划、不需要 CPU seq_lens、multi-CTA 计数器默认 None。 |
| `_call_flash_mla_sparse_fwd` | 688 | 延迟导入并调用 `flash_mla_sparse_fwd`（便于子类替换）。 |
| `_call_flash_mla_with_kvcache` | 697 | 调用 sgl_kernel 的 `flash_mla_with_kvcache`。 |
| `_call_get_mla_metadata` | 702 | 调用 sgl_kernel 的 `get_mla_metadata`。 |
| `__init__` | 707 | 读模型和并行配置：计算 `attn_dcp_size/rank`（仅 sharded DCP）、`dcp_size/rank`（replicated index-K DCP）、`real_page_span`、`local_num_q_heads` 与 `num_q_heads = local × attn_dcp_size`；选择实现并校验；初始化 prims_ts、trtllm/flashmla q 缓冲、q8kv8 缓冲、aiter 元数据、workspace、DCP top-k 合并缓冲。 |
| `_make_aiter_dsa_decode_metadata_buffer` | 1094 | 按 aiter 给的尺寸分配 6 个工作元数据张量。 |
| `_ensure_aiter_dsa_decode_metadata_buffer` | 1140 | 容量或 dtype 不符时重新分配 aiter 元数据缓冲。 |
| `_prepare_aiter_dsa_decode_metadata` | 1177 | 调用 `get_mla_metadata_v1` 填充 aiter 持久化调度元数据，返回 kwargs。 |
| `_pad_trtllm_sparse_page_table` | 1231 | 把 top-k 宽度补到 4 的倍数（-1 填充）。 |
| `get_device_int32_arange` | 1373 | 返回缓存的 int32 arange，不够时按 2 的幂扩容。 |
| `update_verify_buffers_to_fill_after_draft` | 1381 | 空实现（接口要求）。 |
| `_graph_page_table_width` | 1384 | graph 回放扫描 `req_to_token` 的列数：有宽表用宽表宽度，否则用 `req_to_token` 宽度。 |
| `_transform_table_1_to_real` | 1406 | 把 page_size=1 的 slot 表按 `real_page_span` 步长采样并整除，得到页号表。 |

### 3.4 元数据构建

| 名称 | 行号 | 作用 |
|---|---|---|
| `_build_paged_mqa_schedule_2d_ctx_lens` | 1246 | 生成 DeepGEMM paged MQA 的 2D context 长度；target verify 的 DG-native 用 `[B, next_n]`；DCP 下换成本卡持有的长度。 |
| `_dcp_local_ctx_lens` | 1280 | sharded DCP 下用 `get_dcp_lens` 把长度换成本 rank 持有的 index-K 份额。 |
| `_refresh_paged_mqa_schedule_metadata` | 1288 | 重新计算调度元数据并原地 `copy_` 到已有缓冲。 |
| `_build_topk_v2_plan` | 1301 | 启用 top-k v2 时，为每次 forward 预计算 cluster 阈值 plan。 |
| `_refresh_topk_v2_plan` | 1317 | graph 回放时原地刷新 top-k v2 plan。 |
| `_get_fused_topk_page_table` | 1330 | fused top-k 下 top-k 输出就是 slot 表；不支持的 top-k 后端报错。 |
| `prepare_mtp_seed_indices` | 1340 | 把 prefill RAGGED top-k（拼接后的逻辑下标）转换成物理 slot，作为下一步 draft decode 的种子。 |
| `dcp_gather_kv_cache` | 1392 | DCP prefill：取本卡 `dcp_local_kv_rows` 对应的 KV 行，用 `dcp_gather_request_rows` 按 `forward_batch.seq_lens_cpu` 从所有 rank 收齐，得到"请求接请求、位置有序"的完整 KV，补齐到整页。 |
| `init_forward_metadata_out_graph` | 1416 | graph 外的元数据入口：调用 `_apply_cuda_graph_metadata` 后再 `init_kernel_metadata`。 |
| `requires_eager_metadata` | 1438 | 属性；子类覆盖了 `init_kernel_metadata` 时为 True。 |
| `init_kernel_metadata` | 1444 | 空钩子，供子类刷新自定义 kernel 状态。 |
| `init_forward_metadata` | 1448 | eager 路径主入口，按 forward mode 构建 `DSAMetadata`（见 §1.3），包括 CP 切分、kv-shard 翻译、DCP 的 `page_table_1_flattened` 和 `dcp_local_kv_rows`。 |
| `_cal_indexer_k_start_end` | 1801 | 为 prefill 的每个 token 计算 indexer 在拼接 K 上的 `[ks, ke)` 和 `token_to_batch_idx`；CP 下 `k_offset` 只累加被选中的请求，再用 `shard_local_tokens` 取本 rank 的 token。 |
| `get_cuda_graph_seq_len_fill_value` | 4852 | graph padding 的 seq_len 填充值 1。 |
| `set_dsa_prefill_impl` | 4856 | 每批决定是否用 MHA one-shot（graph 内、prefill CP、HiSparse 下都关闭），以及 `flashmla_auto` 下选 `flashmla_sparse` 还是 `flashmla_kv`。 |
| `get_topk_transform_method` | 4918 | fp8 KV 或 q8 且实现为 flashmla_sparse 类、EXTEND 时用 RAGGED；sharded DCP 的 EXTEND 也用 RAGGED；其他 PAGED。 |
| `get_indexer_metadata` | 4948 | 打包 `DSAIndexerMetadata` 给 indexer（元数据、top-k 方式、调度、是否强制非融合、`dcp_size`）。 |
| `_compute_flashmla_metadata` | 4971 | 调用 `get_mla_metadata` 计算 FlashMLA tile 调度。 |

### 3.5 CUDA graph

| 名称 | 行号 | 作用 |
|---|---|---|
| `init_cuda_graph_state` | 1878 | 决定 `dsa_drop_wide_page_table`，预分配 graph 静态缓冲和 q padding 缓冲，扩容 multi-CTA 计数器。 |
| `_multi_ctas_kv_counter_for` | 1985 | 返回足够大的 trtllm multi-CTA 计数器；捕获中扩容视为错误。 |
| `_ensure_multi_ctas_kv_counter_capacity` | 2001 | 捕获前把计数器扩到所需行数。 |
| `_build_forward_metadata_cuda_graph` | 2014 | 捕获时为某个 bs 构建并缓存 decode / target verify / draft extend 的 `DSAMetadata`。 |
| `_apply_cuda_graph_metadata` | 2168 | 捕获与回放共用：未缓存则构建；否则按 mode 原地刷新各字段（融合 kernel 或逐项 `copy_`），刷新调度、top-k plan、KPool、FlashMLA 元数据。 |
| `init_forward_metadata_replay_cuda_graph_from_precomputed` | 2490 | MTP 快速路径：从预计算元数据拷贝，再刷新 DeepGEMM 调度和 top-k plan。 |

### 3.6 前向入口

| 名称 | 行号 | 作用 |
|---|---|---|
| `forward_extend` | 2538 | prefill / target verify / draft extend 入口，流程见 §1.3。DCP 下非 target verify 先 `dcp_gather_kv_cache`。 |
| `forward_decode` | 2961 | decode 入口，流程见 §1.3。`flashmla_kv` 在 sharded DCP 下 `dcp_partial=True`。 |

### 3.7 各 kernel 实现

| 名称 | 行号 | 作用 |
|---|---|---|
| `_forward_fa3` | 3178 | FA3 `flash_attn_with_kvcache`，每个 query 为单 token 序列；先裁掉尾部元数据 padding。 |
| `_forward_flashmla_sparse` | 3246 | 头数补到 64（Hopper）或 128（Blackwell），处理 attn sink padding，调用 `flash_mla_sparse_fwd`。 |
| `q8kv8_born_fp8_q_eligible` | 3318 | 判断本批是否一定走 q8kv8 kernel（可让模型直接产出 fp8 q）；prefill CP 下返回 False。 |
| `q8kv8_acquire_born_q_buffer` | 3356 | 返回只增不减的 fp8 q 缓冲（pad 头保持为零）。 |
| `q8kv8_stash_born_q` | 3382 | 记录本层已写好 fp8 q（token 数、layer id），上一份未消费则报错。 |
| `q8kv8_born_q_sentinel` | 3392 | 返回填 NaN 的 bf16 占位 q，误读时输出 NaN。 |
| `_pad_topk_width_buffered` | 3409 | 用持久缓冲把 top-k 宽度补到指定倍数（-1）。 |
| `_forward_flashmla_sparse_q8kv8` | 3448 | SM90 FP8 稀疏 prefill：构造 fp8 q（born / pad / 融合拼接），把 KV gather/cast 到带 top-k 个零行的 fp8 缓冲，调用 `sparse_mla_q8kv8_prefill_fwd`。 |
| `_forward_triton_sparse_mla` | 3682 | 调用 Triton `sparse_mla_prefill`，捕获中关闭 union。 |
| `_forward_flashinfer_sparse_mla` | 3717 | 调用 SM120 的 `flashinfer_sparse_mla_forward`。 |
| `_flashmla_kv_q` | 3746 | 为 flashmla_kv 构造 absorbed q：能零拷贝就零拷贝，否则写入预置零的 head-padding 缓冲。 |
| `_forward_flashmla_kv` | 3781 | FlashMLA 读 fp8 分页 KV；`dcp_partial` 时用 `dcp_localize_compact` 把全局 slot 压成本卡 slot，按 `topk_length` 现建调度，返回 `(o, lse)`；否则按 `_RowPlan` 对齐行。 |
| `_dcp_flashmla_sched_meta` | 3882 | 静态方法；DCP 部分结果每次调用用 DeepSeek-V4 的 windowed 调度生成 `FlashMLASchedMeta`。 |
| `_forward_standard_mha` | 3906 | MHA one-shot：SM100 用 trtllm ragged attention，其他用 `flash_attn_varlen_func`；padding 行清零。 |
| `_forward_tilelang` | 3981 | top-k 宽度补到 64 的倍数后调用 `tilelang_sparse_fwd`。 |
| `_forward_triton_decode` | 4014 | Triton split-K 稀疏 decode，按 stream 缓存 workspace。 |
| `_forward_triton_dcp` | 4039 | ROCm replicated-index-K DCP：`localize_dcp_page_table` 得到本卡 slot 后做 split-K decode，返回 `(out, lse)`。 |
| `_forward_intel_xpu_sparse_decode` | 4085 | XPU：把选中的 KV gather 成虚拟页，再用 `flash_mla_decode`。 |
| `_forward_intel_xpu_dense_prefill` | 4151 | XPU 稠密 causal prefill（`flash_mla_prefill`）。 |
| `_forward_intel_xpu_sparse_prefill` | 4202 | XPU HiSparse 下按翻译后的 slot 做稀疏 prefill。 |
| `_forward_aiter` | 4236 | ROCm decode：头数不足 16 时复制头，提取有效 KV 下标后调用 `mla_decode_fwd`。 |
| `_forward_aiter_extend` | 4324 | ROCm extend：每个 token 作为一个 query 行调用 `mla_decode_fwd`。 |
| `_forward_trtllm` | 4422 | trtllm / prims_ts 统一入口：构造 fp8 q（融合或量化+RoPE）；prefill CP 下用局部 RoPE 位置、all-gather fp8 KV 后存活行写入；写 KV；sharded DCP prefill 用 `dcp_gather_kv_cache`；转换 top-k；DCP 部分结果用 `dcp_localize_compact`；replicated DCP 用 `localize_dcp_page_table`；最后调 `dcp_trtllm_sparse_attention` 或 `trtllm_batch_decode_with_kv_cache_mla`，或转给 `_forward_prims_ts`。 |
| `_forward_prims_ts` | 4757 | 调用 `PrimsTSSparseMLA.forward`；decode 把长度截到 top-k 宽度；需要时返回 `(out, lse)`。 |
| `_trtllm_q_fp8` | 4799 | 把 q_nope/q_rope 融合拼接并 cast 成 fp8，必要时写入 head-padding 缓冲。 |
| `_pad_topk_indices` | 4831 | top-k 行数少于 q 行数时补 -1 行。 |

### 3.8 `DeepseekSparseAttnMultiStepBackend`

| 名称 | 行号 | 作用 |
|---|---|---|
| `DeepseekSparseAttnMultiStepBackend` | 4997 | MTP 多步 draft decode 的容器，每步一个 `DeepseekSparseAttnBackend`。 |
| `__init__` | 5002 | 为 `speculative_num_steps - 1` 步各建一个后端。 |
| `init_forward_metadata` | 5024 | 逐步调用子后端的 `init_forward_metadata`。 |
| `init_cuda_graph_state` | 5028 | 逐步初始化 graph 状态。 |
| `init_forward_metadata_out_graph` | 5032 | 捕获时用 DECODE 视图逐个构建；回放时走 `_init_forward_metadata_replay`，再刷新需要 eager 元数据的子后端。 |
| `_init_forward_metadata_replay` | 5061 | 预计算一次元数据，再拷贝到各子后端：KPool 融合时 sibling 拷贝；步数 >3 时用 `fused_metadata_copy_multi_cuda` 拷前三个，失败回退逐个拷贝。 |
| `init_forward_metadata_in_graph` | 5234 | 逐步调用子后端的 `init_forward_metadata_in_graph`（定义在 `base_attn_backend.py:205`）。 |

## 4. 与 PCP+DCP 方案 A 的关系

### 4.1 处理 prefill CP 的位置

| 位置 | 行号 | 现有行为 |
|---|---|---|
| `prepare_kv_for_attention` | 151-173 | `dsa_use_prefill_cp` 且 `is_cp_active` 时 all-gather 本轮新 token 的 MLA KV（`interleave.py:401` 的 `materialize_full_mla_kv`），之后 `forward_extend` 把完整新 token KV 写入每张卡。 |
| `_plan_live_rows` | 250-252 | 有 `attn_cp_metadata` 时允许元数据行多于 q 行。 |
| `__init__` 的 `flashmla_prefill_token_packing` | 811 | `attn_dcp_size > 1` 且开 prefill CP 时禁用 token packing（注释说 CP 布局没验证过）。 |
| `init_forward_metadata` extend 分支 | 1620-1635 | `can_dsa_prefill_cp_interleave` 时，用 `shard_local_tokens` 截 `seqlens_expanded`，用 `shard_per_request` 得到本 rank 的 `extend_seq_lens` 和 `bs_idx`，再按 `bs_idx` 过滤 `indexer_seq_lens`、`cache_seqlens_int32`、`page_table`。 |
| `init_forward_metadata` | 1645-1652 | 有 `bs_idx` 时 `cu_seqlens_q` 用本 rank 的 extend 长度。 |
| `_cal_indexer_k_start_end` | 1801-1876 | `k_offset` 只累加被选中的请求；最后按 token 切分。 |
| `pad_dsa_cache_seqlens` 调用 | 1716 | CP 下补到 `per_rank_actual_token`。 |
| `q8kv8_born_fp8_q_eligible` | 3344 | prefill CP 下不用 born-fp8 q。 |
| `_forward_trtllm` | 4508-4556 | prefill CP 下用 `cp_local_rope_positions`；量化后 `all_gather_dsa_trtllm_fp8_kv` 收齐新 token 的 fp8 KV，`cp_live_store_rows` 截到存活行，再写入 `out_cache_loc[:n_store]`。 |
| `set_dsa_prefill_impl` | 4893 | prefill CP 下关闭 MHA one-shot。 |

### 4.2 处理 DCP 的位置

| 位置 | 行号 | 现有行为 |
|---|---|---|
| `__init__` | 740-768 | `sharded_dcp = dsa_shards_index_k()`（K3.1 调用过 `set_dsa_shards_index_k()` 且 `attn_dcp_size > 1`）时设置 `attn_dcp_size/rank`，`dcp_size` 置 1；非 sharded 模型反过来。 |
| `__init__` | 755-758 | `num_q_heads = local_num_q_heads × attn_dcp_size`，因为 DCP decode 的 q 从 DCP 组 gather 而来。这个值决定 `trtllm_num_q_heads`、`flashmla_kv_num_q_heads`、`num_dcp_q_heads`、prims_ts 支持检查。 |
| `__init__` | 960-977 | `_validate_dcp_backend`，并初始化 DCP 候选 all-gather 和 top-k 合并缓冲。 |
| `_build_paged_mqa_schedule_2d_ctx_lens` / `_dcp_local_ctx_lens` | 1246-1286 | indexer 只对本卡的 index-K 打分，长度换成本卡份额。 |
| `dcp_gather_kv_cache` | 1392-1404 | DCP prefill 的整序列 KV gather。 |
| `init_forward_metadata` | 1662-1676, 1690, 1775-1784 | DCP 下总是构建 `page_table_1_flattened`，跳过容量检查（slot 是放宽后的编号），`dcp_local_kv_rows = filter_dcp_local_chunk_kv_indices(...) // attn_dcp_size`。 |
| `forward_extend` | 2636-2639 | 非 target verify 时读 gather 后的 KV。 |
| `_forward_flashmla_kv` | 3827-3843, 3874-3879 | `dcp_partial` 时本地化 slot、现建调度、返回 `(o, lse)`。 |
| `_forward_trtllm` | 4452-4456, 4574-4575, 4643-4646, 4720-4745 | sharded DCP：prefill gather KV；decode/verify 部分结果。 |
| `get_topk_transform_method` | 4940-4943 | DCP EXTEND 用 RAGGED，与 gather 后的布局一致。 |
| `_apply_cuda_graph_metadata` | 2263 | DCP 下不用 target verify 的 DG-native `[B, next_n]` 写入。 |

### 4.3 方案 A 需要的修改

方案 A 的 prefill 要求"本卡 query 行（来自 CP）+ 整序列 KV 行（来自 DCP gather）"。对本文件来说：

1. **`prepare_kv_for_attention` 和 `_forward_trtllm` 的 CP gather 要关掉。** 方案 A 下新 token 的 KV 只写本地（计划 §3.4 第 2 步）。`prepare_kv_for_attention` 应在 DCP 生效时直接返回本地 `k_nope/k_pe`；`_forward_trtllm` 的 `all_gather_dsa_trtllm_fp8_kv` 分支（`:4545-4556`）同样要跳过，`cache_loc` 改成本卡 token 的位置。
2. **`init_forward_metadata` 里两套长度要分开。** query 侧用 CP 切分后的 `extend_seq_lens`、`bs_idx`、`seqlens_expanded`；KV 侧的 `page_table_1_flattened`、`dcp_local_kv_rows`、`topk_indices_offset` 要覆盖 gather 后的完整布局。现在三者都从按 `bs_idx` 过滤后的 `indexer_seq_lens_cpu`、`cu_seqlens_k` 算出。
3. **`dcp_gather_kv_cache` 的 `seq_lens_cpu` 要和本地行一致。** 现在传入的是未过滤的 `forward_batch.seq_lens_cpu`，而本地行来自过滤后的请求。方案 A 下应传所有 rank 一致的完整请求列表（collective 要求各 rank 形状一致），并相应地对完整列表构建本地行。
4. **`_cal_indexer_k_start_end`** 的 `ks/ke` 和 `k_offset` 要按 gather 后的完整请求布局计算，按全局位置做 causal（与计划 §4.5 的 indexer 修改配套）。
5. **`get_topk_transform_method`**：`ForwardMode.EXTEND` 的判断需要确认是否覆盖 CP extend 的所有 mode（`is_context_parallel_extend` 可能包括 MIXED，推断）。
6. **行对齐**：CP 物理分片 padding 会让 `dsa_cache_seqlens_int32` 多出零长度行，`_plan_live_rows` 已允许这种情况；方案 A 新增 gather 后不应改变 query 行数，这部分可沿用（推断）。

### 4.4 计划可能遗漏的点

- **（计划未提及）`num_q_heads` 的计算在方案 A 下会错。** `__init__:755-758` 用 `num_attention_heads // attn_tp_size × attn_dcp_size`。现在 DCP 组在 attention TP 组里，所以乘以 `attn_dcp_size` 正好恢复全部头（K3.1 DCP8 为 96）。方案 A 下 `attn_tp_size = 1`、头是复制的，这个公式会得到 96 × 8 = 768，进而让 `trtllm_num_q_heads`、`flashmla_kv_num_q_heads`、`num_dcp_q_heads`（multi-CTA 计数器尺寸）、prims_ts 支持检查都出错。方案 A 下 decode q 本地已经是全部头，不需要 gather（计划 §3.5 第 1 步），应改为不乘 `attn_dcp_size`。
- **（计划未提及）`_forward_trtllm` 中的 CP KV gather 和写入（`:4508-4556`）。** 计划 §4.4 只列了 `interleave.py` 的 `materialize_full_mla_kv` 等函数，但 trtllm/prims_ts 路径在 dsa_backend 内部自己调用 `all_gather_dsa_trtllm_fp8_kv` 和 `cp_live_store_rows`，写入的是 gather 后的全部新 token。如果 P1 选 trtllm 或 prims_ts，这里必须改。
- **（计划未提及）`dcp_gather_kv_cache` 的请求列表不一致问题**（见 §4.3 第 3 条）。计划 §4.5 只说"支持 query 只覆盖本卡的行"，但真正的问题在 KV 侧：本地行按过滤后的请求构建，gather 却按全部请求的长度。CP 下某些请求在某个 rank 上没有 query 行（`bs_idx` 不含它），各 rank 的本地行就对不上，结果会出错或 collective 挂起（推断）。`dsa_indexer.py` 的 `_dcp_gather_index_k` 有同样的依赖。
- **（计划未提及）`topk_indices_offset` 和 `cu_seqlens_k`** 在 CP 下按过滤后的请求算，DCP 的 RAGGED top-k 需要它们指向 gather 后的完整布局。K3.1 走 fused top-k，偏移可能由 indexer 内部决定（推断），需要一起核对。
- **（计划可能有误）P1 的后端选择。** 计划 §5 和 §8.3 写"建议先选 FlashMLA（默认）"。但 `_validate_dcp_backend`（`:539-567`）要求 DCP 下 prefill 和 decode 实现必须同为 `flashmla_kv`、`trtllm` 或 `prims_ts`，`flashmla_sparse` 或 `flashmla_auto` 都会被拒绝。计划应明确写成 `flashmla_kv`。
- **（计划未提及）`flashmla_prefill_token_packing`** 在 `attn_dcp_size > 1` 且开 prefill CP 时被关闭（`:811`），方案 A 下 SM100 的 prefill 会失去这个优化，P3 的性能评估要考虑。
- **（计划未提及）`dcp_size` 与 `attn_dcp_size` 两套字段。** sharded DCP（K3.1）下 `self.dcp_size` 被置 1，`attn_dcp_size` 才是实际 DCP 大小；而 `filter_dcp_local_chunk_kv_indices` 和 `all_gather_kv_cache_for_dcp`（`dcp/comm.py:361`）读 `get_parallel().dcp_rank/dcp_size`。`runtime_context.py:1610-1612` 里 `attn_dcp_rank` 由 `dcp_rank` 派生，所以计划 §4.2 只改 `:1611` 一处时两个字段会一致；但 `dcp_size`（`parallel.dcp_size`）、`attn_dcp_size` 和 `attn_cp_size` 三者在方案 A 下也必须相等，建议在参数校验里断言，并在单元测试里覆盖 `filter_dcp_local_chunk_kv_indices` 与 `dcp_gather_request_rows` 使用同一 rank。
