# `python/sglang/srt/layers/cp/utils.py` 代码理解

- 文件路径：`python/sglang/srt/layers/cp/utils.py`
- 行数：518
- 所属模块：PCP 的公共导入入口和运行时辅助函数，包括 CP 前向准备、切分/gather 包装、CP 分片 KV 存储的容量换算、DSA cache 按层切分的辅助函数。

## 1. 整体功能

这个文件有三类内容：

1. **导入门面**：从 `base.py`、`interleave.py`、`zigzag.py` 重新导出策略类和元数据类型（`__all__`，第 491-518 行），调用方大多从这里 import。
2. **CP 前向的运行时辅助**：`is_cp_active` 判断当前 batch 是否走 CP；`prepare_cp_forward` 在模型前向前构建 `attn_cp_metadata` 并截断 `out_cache_loc`；一组 `cp_shard_*` / `cp_gather_*` 函数是对当前策略对应方法的带断言包装；`cp_shard_model_inputs` 是一个 context manager，在前向期间切分输入、退出时恢复 `ForwardBatch` 上被临时替换的字段。
3. **CP 分片 KV 存储（`cp_kv_sharded`）的容量换算**：`get_cp_sharded_padded*`、`from_cp_sharded*` 系列，把全局 token 数 / cache 下标和本地存储之间按"组 = `cp_size × page_size`"换算。这是 `--enable-kv-cache-sharding`（计划 §1.3 提到的页交错 KV 分片）用的，和 DCP 是两套机制。
4. **DSA cache 按层切分（`--enable-dsa-cache-layer-split`）**：`is_glm_dsa_cache_layer_split_enabled` 等函数，让每个 CP rank 只存一部分层的 DSA KV。

调用时机：`model_executor/runner/eager_runner.py:266` 在 CP 激活时调用 `prepare_cp_forward`，`eager_runner.py:377` 用 `cp_shard_model_inputs` 包住模型前向；BCG 路径在 `layers/cp/bcg/bcg.py:191, 231, 318` 调用。

## 2. 主要依赖与调用方

依赖：`layers/cp/base.py`、`layers/cp/interleave.py`、`layers/cp/zigzag.py`、`layers/cp/padding.py`（`pad_logical_token_to_physical`）、`layers/moe/utils.py`（`get_moe_a2a_backend`）、`runtime_context`（`get_parallel`、`get_schedule`、`uses_mla_backend`）、`layers/dp_attention.py`（`set_local_dp_buffer_len`）。

关键调用方（grep 确认）：
- `prepare_cp_forward`：`model_executor/runner/eager_runner.py:266`、`layers/cp/bcg/bcg.py:191`。
- `cp_shard_model_inputs`：`eager_runner.py:377`。
- `cp_live_store_rows`：`layers/attention/dsa_backend.py:4551`、`layers/attention/dsa/dsa_prefill_cuda_graph.py:183`。
- `cp_local_rope_positions`：`dsa_backend.py:4511`。
- `cp_materialize_global_token_order`：`qwen_sparse_attn_backend.py:138`、`qsa/qsa_indexer.py:646`、`deepseek_v4_backend.py:3656`、`dsv4/compressor.py`。
- `cp_gather_full_sequence_states`：`layers/layer_boundary/ops.py:192`、`dsa/dsa_npu_indexer.py:211`。
- `cp_gather_after_forward`：`layers/cp/bcg/bcg.py:318, 328`、`layers/attention/index_topk_share.py:59`。
- `get_cp_sharded_padded_cache_transfer_indices`：`disaggregation/prefill.py:2073`、`flashattention_backend.py:839, 1202`。
- `get_cp_sharded_padded_chunk_tokens`：`mem_cache/prefill_budget.py:296`。
- `from_cp_sharded`：`managers/scheduler.py:2879, 2972`、`scheduler_components/pool_stats_observer.py`。
- `get_layer_owner` / `get_layer_shard_range`：`mem_cache/dsa_cache_layer_split.py`；`get_glm_dsa_cp_layer_shard_info`：`model_executor/pool_configurator.py:672`、`mem_cache/kv_cache_configurator.py:2301`。

## 3. 类与函数说明

### 3.1 CP 分片 KV 存储的容量换算（`cp_kv_sharded` 布局）

这些函数都通过 `get_cp_strategy().cp_size` 和 `get_schedule().page_size` 计算，"组"指 `cp_size × page_size` 个全局 token。

| 名称 | 行号 | 作用 |
|---|---|---|
| `get_cp_sharded_padded` | 47 | 全局 KV 数除以 `cp_size`，得到本地存储数。 |
| `from_cp_sharded` | 53 | 本地 KV 数乘以 `cp_size`，换回全局数（调度器统计容量用）。 |
| `get_cp_sharded_padded_cache_indices` | 59 | 从 cache 行下标恢复本地完整页的行下标：每组取一个代表行，映射到本地页后展开成 `page_size` 行；代表行 ≤ 0（sentinel）时保持原值。 |
| `from_cp_sharded_expanded_indices` | 78 | 反向：把本地完整页展开到每个 CP 列，按 cache 行顺序输出全局行下标。 |
| `_ceil_group_count` | 92 | 向上取整的组数。 |
| `get_cp_sharded_padded_allocation_tokens` | 97 | 给定前缀和新增 token 数，计算新分配的全局容量（按组对齐的差值）。 |
| `get_cp_sharded_padded_cache_transfer_indices` | 108 | 把 cache 行投影到本地行，保留形状、dtype 和 sentinel（≤ 0 不变）。PD 传输和 FlashAttention 后端用。 |
| `get_cp_sharded_padded_allocation_lengths` | 118 | 请求端点（长度）换成本地预留容量：组数 × `page_size`。 |
| `get_cp_sharded_padded_allocation_size` | 126 | 根据 CPU 上的前缀长度和 extend 长度，统计本次新增的本地行数。 |
| `get_cp_sharded_padded_chunk_tokens` | 138 | 把 chunk 大小对齐到完整的新组，同时保留前缀所在组的剩余尾部。 |
| `get_cp_sharded_padded_available_tokens` | 148 | 把空闲容量换成可用 token 数，在 chunk headroom 扣除之前使用；最多允许一个不完整组的尾部。负值原样返回。 |

### 3.2 DSA cache 按层切分

| 名称 | 行号 | 作用 |
|---|---|---|
| `is_glm_dsa_cache_layer_split_enabled` | 167 | 非 draft worker、开启 `enable_dsa_cache_layer_split`、MLA 后端且是 DSA 模型时为 True。 |
| `get_glm_dsa_cp_layer_shard_info` | 183 | 返回 `(attn_cp_rank, attn_cp_size)`；功能关闭或 CP 只有 1 个 rank 时返回 `(None, 1)`。 |
| `get_glm_dsa_layer_split_effective_num_layers` | 198 | 按层切分时每个 rank 实际要分配的层数：`ceil(num_layers / shard_size) + 1`（多 1 层作为读取远端层的 scratch）。 |
| `get_layer_shard_range` | 216 | rank 拥有的连续层区间 `[start, end)`，前 `total % shard_size` 个 rank 多一层。 |
| `get_layer_owner` | 231 | 找到某个本地层下标属于哪个 rank，找不到时 `ValueError`。 |

### 3.3 CP 判断与前向准备

| 名称 | 行号 | 作用 |
|---|---|---|
| `is_cp_active` | 243 | 当前 batch 是否走 CP：`forward_mode.is_context_parallel_extend()` 为真、策略存在、有 `input_ids`，并且 `strategy.can_apply(len(input_ids), forward_batch)` 为真。decode 下恒为 False。 |
| `is_mla_cp_enabled` | 260 | 配置了 CP 且使用 MLA 后端。 |
| `is_mla_cp_active` | 265 | `is_mla_cp_enabled()` 且 `is_cp_active(forward_batch)`。 |
| `prepare_cp_forward` | 270 | 断言 CP 激活；`attn_cp_metadata` 为空时调用 `strategy.build_metadata`（`num_tokens` 取 extend 长度之和），再 `pad_logical_token_to_physical` 补齐。未开启 CP-TP group sharing 且有 `global_num_tokens_cpu` 时，把 DP buffer 长度设为各 rank 物理行数之和。最后把 `out_cache_loc` 截断到 `num_tokens`（去掉尾部 padding）。注意 `out_cache_loc` 仍是**全部** token 的写入位置。 |
| `cp_split_before_forward` | 307 | 断言后同时切分 embedding 和 positions，返回二元组。BCG 路径使用。 |
| `cp_shard_hidden_states` | 322 | 带断言地调用 `strategy.shard_hidden_states`。 |
| `cp_gather_full_sequence_states` | 331 | 带断言地调用 `strategy.gather_hidden_states`，还原完整逻辑顺序。 |
| `cp_shard_position_ids` | 341 | 带断言地调用 `strategy.shard_position_ids`。 |
| `cp_local_rope_positions` | 350 | 从全局 positions 切出与本地 K 分片逐行对应的本地位置（包括 padding 行），行数不一致时断言失败。`forward_batch.positions` 在所有模式下都保持全局。 |
| `cp_live_store_rows` | 367 | CP all-gather 之后要写入 KV cache 的行数：`total_seq_lens`（真实 token）。断言 gather 行数和 `out_cache_loc` 行数都不小于它，padding 必须在 `out_cache_loc` 尾部。 |
| `cp_interleave_input_ids` | 387 | 生成 rank-major 的 interleave 顺序 input ids。MoE a2a 后端不是 none 时直接切分本地分片；否则补零到物理总行数后 `view(-1, attn_cp_size).T.flatten()`。 |
| `cp_gather_after_forward` | 398 | 模型出口的 gather。对 tuple 逐项 gather，且对 `(hidden, None)` 这种二元组只返回第一项（MiMo 的文本分支）。 |
| `cp_materialize_global_token_order` | 421 | 调 `strategy.gather_kv_cache`，把 CP 张量还原到全局 token 顺序。 |
| `cp_shard_model_inputs` | 432 | context manager：切分 hidden states、positions、input ids；若给了完整 input ids，还临时设置 `forward_batch.input_ids_global`；若 `spec_info.hidden_states` 行数等于完整 hidden，也临时切分。退出时恢复 `spec_info.hidden_states` 和 `input_ids_global`（原来没有就删除），保证 logits 处理仍拿到全批元数据。 |
| `_to_int_list` | 483 | 把张量或序列转成 `list[int]`，None 原样返回。 |

## 4. 与 PCP+DCP 方案 A 的关系

**计划提到的修改：**
- `prepare_cp_forward`（270-304，计划 §4.3 写作 `:270-304`）：截断 `out_cache_loc` 后，要确认写入位置和本卡计算的 token 一一对应。现在 `out_cache_loc[:num_tokens]` 是全部 token 的写入位置，后面靠 gather 后写全部行（见 `cp_live_store_rows`）。方案 A 若"只写本地"，需要用新布局从中选出本卡的那 1/N 行。
- `prepare_cp_forward` 调用 `build_metadata` 时只传了 `seq_lens_cpu` 和 `extend_seq_lens_cpu`，没有传前缀长度。按位置取模需要每个请求的 `prefix_len`（可由 `seq_lens - extend_seq_lens` 得到，或读 `extend_prefix_lens_cpu`），所以 `build_metadata` 的签名或调用要跟着改（推断）。（计划未明确提及）

**计划未提及的相关点：**
- `cp_live_store_rows`（367）：DSA 后端 gather 后用它决定写多少行，断言"写入行数 = `total_seq_lens`、padding 在 `out_cache_loc` 尾部"。方案 A 改为只写本地行后，这个函数和 `dsa_backend.py:4551`、`dsa_prefill_cuda_graph.py:183` 的调用点要一起改。
- `cp_local_rope_positions`（350）从全局 positions 切本地位置，依赖 `shard_position_ids`；切分规则改了以后它自动跟随，但它断言行数等于本地 K 分片，新 padding 规则下要保持一致。
- `cp_interleave_input_ids`（387）用 `view(-1, cp).T.flatten()` 生成 rank-major 顺序，等价于拼接下标取模，需要适配新布局（MoE 和 BCG 都用它）。
- `is_cp_active`（243）在 decode 恒为 False，这与计划 §3.5"decode 走 DCP decode"一致。但它也依赖 `can_apply`：总 extend 长度 < `cp_size` 的 extend batch 会退回非 CP 路径，方案 A 需要保证这种 batch 走普通 DCP extend 时仍然正确（attn_tp=1、权重全复制）。
- 本文件的 `get_cp_sharded_padded*` 系列属于 `--enable-kv-cache-sharding`（页交错 KV 分片，`is_cp_sharded_padded`），和 DCP 的 owner 规则不同；方案 A 应保证两者不同时开启（推断，计划 §1.3 只说它是替代方案）。
