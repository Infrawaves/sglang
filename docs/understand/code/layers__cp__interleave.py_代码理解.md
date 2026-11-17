# `python/sglang/srt/layers/cp/interleave.py` 代码理解

- 文件路径：`python/sglang/srt/layers/cp/interleave.py`
- 行数：604
- 所属模块：prefill context parallel（PCP）的 interleave 切分策略实现，包括元数据、切分/还原、KV 与 indexer K 的 all-gather，以及 DSA indexer 与 KV gather 的重叠调度。

## 1. 整体功能

这个文件实现 `--cp-strategy interleave`。在 EXTEND/MIXED 的 CP 前向里，每个 CP rank 只负责一部分 token：对整批 token 拼接后的下标 `i`，rank `r` 负责 `i % cp_size == r` 的 token（模块 docstring 第 17-22 行）。注意这里的下标是**整批拼接后的下标**，不是每个请求内的序列位置。

主要数据流：

1. `prepare_cp_forward`（`layers/cp/utils.py`）调用 `InterleaveCPStrategy.build_metadata` 得到 `InterleaveContextParallelMetadata`，再由 `pad_logical_token_to_physical`（`layers/cp/padding.py`）把每个 rank 的物理行数对齐到 `cp_size` 的倍数。对齐后 `per_rank_logical_token` 保存真实行数，`per_rank_actual_token` 变成统一的物理行数。
2. 模型入口用 `shard_hidden_states` / `shard_position_ids` 取本 rank 的行，并在尾部补零到物理行数。
3. DSA/MLA 层里，本 rank 的 latent KV 和 indexer K 通过 `gather_kv_cache` all-gather 成完整 token 顺序，每张卡都写一份完整 KV（现有 PCP 的"KV 复制"）。`DSAInterleaveCPIndexerOverlap` 和 `run_indexer_with_kv_gather` 把 KV gather 放到旁路 stream，和 indexer 计算重叠。
4. KDA 层前后用 `cp_interleave_to_sequence_order` / `cp_sequence_to_interleave_order` 在"rank-major 的 interleave 行"和"序列顺序"之间重排。
5. 模型出口用 `gather_hidden_states` 还原完整 token 顺序。

all-gather 结果是 rank-major 布局：`[rank0 的所有行, rank1 的所有行, ...]`，每段长度为物理行数。还原到 token 顺序有三条路径：元数据带 `gather_index` 时用 `index_select`；各 rank 行数相等时用 Triton kernel `interleave_rows` 或 reshape/transpose；否则构造 `(i % cp) * physical_len + i // cp` 的下标再 `index_select`。

元数据的 `gather_index` / `local_index` 字段在这个策略本身的 `build_metadata` 里不会填写，目前只有 `layers/attention/deepseek_v4_backend.py:2112-2170`（`_late_layer_tail_cp_layout`）按 `token_indices % cp_size` 自建 owner 布局时使用。

## 2. 主要依赖与调用方

依赖：
- `layers/cp/base.py`：`ContextParallelStrategy` 基类、`BaseContextParallelMetadata`、`get_cp_strategy`。
- `layers/cp/padding.py`：`pad_local_rows`。
- `layers/dp_attention.py`：`attn_cp_all_gather_into_tensor`、`is_allocation_symmetric`。
- `distributed/device_communicators/pynccl_allocator.py`：`use_symmetric_memory`。
- `sglang.kernels.ops.memory.interleave_rows`（Triton 行重排）、`sglang.kernels.ops.attention.dsa.cp_split.dsa_cp_interleave_q_seqs`。
- `layers/cp/bcg/interleave_bcg.py`：breakable CUDA graph 下的 `bcg_indexer_with_kv`。

调用方（grep 确认）：
- `layers/cp/base.py:326`：`init_cp_strategy` 构造 `InterleaveCPStrategy`。
- `models/kimi_k3.py:2670, 2721`、`models/glm5_next.py:651, 677`：KDA 前后的两个重排函数。
- `layers/layer_boundary/prepare.py:571`、`models/deepseek_common/moe_transport.py:164`：`attn_cp_interleave_gather`。
- `models/deepseek_common/attention_forward_methods/forward_mla.py:473`：`DSAInterleaveCPIndexerOverlap`。
- `layers/attention/dsa_backend.py:168`：`materialize_full_mla_kv`；`dsa_backend.py:1624`：`shard_per_request`。
- `layers/attention/dsa/dsa_indexer.py:1084, 1095`：`materialize_full_indexer_k_cache`。
- `model_executor/forward_batch_info.py:1389`：`moe_num_token_non_padded`。
- `layers/attention/deepseek_v4_backend.py:3102, 3647`、`layers/cp/interleave_kpool.py:18`：`interleave_rows_per_request`。
- `hardware_backend/npu/attention/ascend_backend.py:1429` 等：`local_q_indices`。

## 3. 类与函数说明

### 3.1 顶层函数

| 名称 | 行号 | 作用 |
|---|---|---|
| `attn_cp_interleave_gather` | 57 | 在 attention CP 组上把各 rank 等长的 padded 分片按 rank 顺序 all-gather（不还原 token 顺序）。输出大小按实际分片行数 × `attn_cp_size` 计算，不用 DP scratch 长度；输出 buffer 在对称内存里分配。层边界和 MoE 传输用它拿到 rank-major 的完整行。 |
| `is_interleave_extend` | 88 | 判断当前 batch 是 CP extend 且元数据类型是 `InterleaveContextParallelMetadata`。两个重排函数和 overlap 类用它决定是否生效。 |
| `cp_interleave_to_sequence_order` | 94 | KDA 前把 rank-major 的 interleave 行还原成序列顺序。有 `gather_index` 时走 `index_select`；否则用 `graph_num_tokens`（图模式）或 `total_seq_lens`（eager）截断，GPU 上用 `interleave_rows`，CPU 上 reshape/transpose。 |
| `_use_interleave_rows` | 130 | 判断能否用 Triton kernel `interleave_rows`：张量不在 CPU 且连续。 |
| `cp_sequence_to_interleave_order` | 135 | KDA 后把序列顺序还原成 rank-major 的 padded 行（长度 `gathered_rows`，和层边界的 reduce-scatter 对齐），本身不通信。有 `gather_index` 时用 `index_copy_` 散射回去。 |
| `interleave_rows_per_request` | 418 | 给定每个请求的 extend 长度，计算 rank `cp_rank` 在每个请求里持有多少行，规则是"全局拼接下标 ≡ cp_rank (mod cp_size)"。 |
| `indexer_with_cp_kv` | 539 | indexer + KV gather 的入口：在 breakable CUDA graph 内时预分配输出并调用 `bcg_indexer_with_kv`（eager break），否则直接调 `run_indexer_with_kv_gather`。 |
| `run_indexer_with_kv_gather` | 560 | 在 `dsa_cp_kv` 旁路 stream 上启动 KV all-gather（FP8 先 view 成 uint8 传输），同时在主 stream 跑 `indexer.forward_cuda`（`use_cuda_graph=False`）。`finally` 里主 stream 等待旁路 stream，并对结果 `record_stream`，保证后续 attention 读到完整 KV。 |

### 3.2 `InterleaveContextParallelMetadata`（第 74 行，`@dataclass`，继承 `BaseContextParallelMetadata`）

| 字段 | 含义 |
|---|---|
| `per_rank_actual_token` | 每个 rank 的物理行数（padding 后统一为同一值）。 |
| `max_rank_len` | 每个 rank 的最大行数列表。 |
| `per_rank_logical_token` | padding 前每个 rank 的真实行数，由 `pad_logical_token_to_physical` 填写。 |
| `gather_index` | rank-major 的 gather 结果到 token 顺序的映射（第 79 行注释："Tail row -> packed all-gather slot"）。 |
| `local_index` | 本 rank 的本地元数据行（含 padding 行）。 |
| `moe_local_token_count` | MoE dispatch 用的本地有效行数（0 维 int32 张量），首次访问时惰性创建，BCG 下指向固定地址的 buffer。 |
| `graph_num_tokens` | 图模式下序列相关段使用的容量；eager 的 cache 操作和 gather 仍用 `total_seq_lens`。 |

### 3.3 `InterleaveCPStrategy`（第 163 行，`name = "interleave"`，`kind = INTERLEAVE`）

| 方法 | 行号 | 作用 |
|---|---|---|
| `moe_num_token_non_padded` | 167 | 返回本 rank 的有效行数张量（优先 `per_rank_logical_token`），用于在 MoE all-to-all 前屏蔽 CP padding。前提是有效行是每个本地分片的前缀。 |
| `can_apply` | 183 | 仅在 CP extend 模式下、总 extend 长度 > 0、≥ `cp_size` 且 `cp_size > 1` 时返回 True。否则这一批不走 CP。 |
| `build_metadata` | 190 | 把 `num_tokens` 与 extend 长度之和的差额（padding）加到最后一个请求上，再按总长度 `divmod(total, cp_size)` 算每个 rank 的行数：前 `extra` 个 rank 多一行。只看总长度，不看每个请求的前缀长度。 |
| `shard_hidden_states` | 217 | 对 `x[:total_seq_lens]` 调 `_interleave_shard`，再 `pad_local_rows` 补到物理行数。 |
| `shard_position_ids` | 222 | 同上，对 positions 做切分和补齐。 |
| `_interleave_shard` | 227 | 实际切分：list/tuple 用 `range(cp_rank, len, cp_size)`；张量长度能整除时 `view(-1, cp_size)[:, cp_rank]`，不能整除时用 `arange(cp_rank, tokens, cp_size)` 取行。规则是拼接下标取模。 |
| `local_q_indices` | 244 | 返回 `arange(cp_rank, num_tokens, cp_size)`，即本 rank 的 query 在全局拼接顺序里的下标。NPU 后端使用。 |
| `shard_local_tokens` | 252 | `_interleave_shard` 的公开包装。 |
| `shard_per_request` | 255 | 计算每个请求在本 rank 上的 query 行数：用 `extra_seq` 把上一请求的余数带到下一个请求，等价于拼接下标取模。CPU 结果用 Python 算，GPU 结果由 `dsa_cp_interleave_q_seqs` kernel 生成（图安全）。返回 `(q_lens_cpu, q_lens, bs_idx_cpu, bs_idx)`，去掉 q_len 为 0 的请求。 |
| `gather_hidden_states` | 281 | 调 `_gather_interleaved_tensor`。 |
| `gather_kv_cache` | 286 | 调 `_gather_interleaved_tensor`（与 hidden states 共用实现）。 |
| `_gather_interleaved_tensor` | 291 | 核心 all-gather：检查元数据存在、`total_seq_lens >= 0`、本地行数不少于逻辑行数（否则 `RuntimeError`）；本地行数已等于物理行数且连续时省掉补零拷贝；在对称内存里分配输出后 all-gather；最后按 `gather_index` / 等长快速路径 / 通用下标三种方式还原 token 顺序。 |
| `get_supported_attention_backend` | 357 | 只返回 `[CPAttentionBackendKind.DSA]`，即 interleave 只支持 DSA 后端。 |
| `materialize_full_indexer_k_cache` | 360 | 把本地 indexer K gather 成完整 token 顺序。 |
| `run_attention` | 365 | 空实现，返回 None。这个钩子只给 FlashAttention/zigzag 用，DSA 后端自己跑 attention。 |
| `all_gather_dsa_trtllm_fp8_kv` | 378 | TRT-LLM FP8 路径：把 `k` 和 `k_rope` 拼接后按 uint8 传输，gather 后再 view 回原 dtype 并切开。 |
| `materialize_full_kv` | 389 | 直接 `NotImplementedError`：interleave 不支持稠密 K/V。 |
| `materialize_full_mla_kv` | 401 | 把 `k_nope`、`k_rope` 拼成 latent，gather 成完整顺序后再拆开返回，供 DSA 后端写 KV cache。 |

### 3.4 `DSAInterleaveCPIndexerOverlap`（第 430 行）

每次 MLA 前向创建一个实例，负责安排 indexer 和 KV gather 的重叠。无 RoPE 的模型在原位置启动 indexer；有 RoPE 的模型推迟到本地 KV 完成 RoPE 之后。

| 方法 | 行号 | 作用 |
|---|---|---|
| `__init__` | 438 | 保存 attention 模块和 forward_batch，初始化 `_kv`（gather 后的完整 KV）和 `_deferred`（推迟的 indexer 参数）。 |
| `can_fuse_bmm`（property） | 444 | 有 RoPE 且处于 interleave extend 时返回 False：推迟的 indexer 需要在 attention 前拿到已物化的 absorbed Q。 |
| `_can_overlap` | 452 | 只有 CUDA、interleave extend、indexer 有 `topk_output_width` 时才重叠；若需要 TRT-LLM 融合 RoPE，还要求 DSA prefill 实现使用 raw FP8 KV 布局且不是 MHA。 |
| `run_indexer` | 473 | 不能重叠时直接调 indexer；有 RoPE 时保存参数并返回 None（推迟）；否则拼本地 KV，调 `indexer_with_cp_kv` 同时得到 top-k 和完整 KV。 |
| `prepare_qkv` | 489 | 若之前推迟了 indexer，先（可选地）用 `mla_quantize_and_rope_for_fp8` 对本地 Q/K 做量化和 RoPE，再拼本地 KV 调 `indexer_with_cp_kv`。之后若已有 `_kv` 就从中拆出完整 `k_nope`/`k_pe`，否则走 `prepare_kv_for_attention`。返回 `(q_nope, q_pe, k_nope, k_pe, topk)`。 |

## 4. 与 PCP+DCP 方案 A 的关系

计划 §3.3、§4.3、§4.4 都直接针对这个文件。

**需要修改的函数（计划提到）：**
- `build_metadata`（190）：现在只按总长度平均分配，需要改为按每个请求的 `(prefix_len_req + j) % N` 计算每个 rank 的行数、每个请求的偏移和 padding。
- `_interleave_shard`（227）、`shard_hidden_states`（217）、`shard_position_ids`（222）：现在对 `x[:total_seq_lens]` 做 `view(-1, cp_size)`，即拼接下标取模。方案 A 要按位置取模。
- `gather_hidden_states` / `gather_kv_cache` / `_gather_interleaved_tensor`（281-355）：等长快速路径和通用下标路径都假设拼接下标取模，必须改成按新的映射还原。
- `local_q_indices`（244）、`moe_num_token_non_padded`（167）：前者的 `arange(cp_rank, ..., cp_size)` 同样是拼接下标；后者假设有效行是本地分片的前缀，新布局下要确认这个假设仍成立。
- `cp_interleave_to_sequence_order` / `cp_sequence_to_interleave_order`（94、135）：KDA 用的重排要适配新的行布局。
- `materialize_full_mla_kv`（401）、`materialize_full_indexer_k_cache`（360）、`run_indexer_with_kv_gather`（560）：计划要求方案 A 下只写本地 token，不再先 gather 再写。

**计划未提及、读代码发现的相关点：**
- `shard_per_request`（255）和 `interleave_rows_per_request`（418）也按拼接下标计算每个请求的 query 行数，前者被 `dsa_backend.py:1624` 用来构造 DSA 的 per-request q 长度，后者被 DSV4 和 `interleave_kpool.py` 使用。改切分规则后这两个函数必须一起改，否则 DSA 的 q_lens 与实际行不一致。GPU 侧还依赖 Triton kernel `dsa_cp_interleave_q_seqs`，它也要换成按位置取模的版本。（计划未提及）
- 元数据已经有 `gather_index` / `local_index` 字段，`_gather_interleaved_tensor` 和两个 KDA 重排函数都已支持"有 `gather_index` 就用 `index_select`/`index_copy_`"。`deepseek_v4_backend.py:2112-2170` 的 `_late_layer_tail_cp_layout` 已经按 `owner_rank = token_indices % cp_size` 构造过这种布局，可作为按位置取模布局的参考实现。但 `shard_hidden_states`、`shard_position_ids`、`local_q_indices`、`moe_num_token_non_padded` 目前不读 `local_index`，需要补上。（计划未提及）
- 若元数据带 `gather_index`，`cp_sequence_to_interleave_order` 跳过 Triton 快速路径，改用 `new_zeros + index_copy_`，有额外开销（推断，需在 P3 评估）。
- `can_apply`（183）要求总 extend 长度 ≥ `cp_size`；不满足时这一批不走 CP，会以全 token 的方式进入 DCP extend 路径。方案 A 要保证这条回退路径在 attn_tp=1、DCP=CP 组时仍正确。（计划未提及）
- 关于"只写本地"：现有路径 gather 后每卡写完整 KV；而 `UnitIds.translate_write`（`mem_cache/kv/alloc/id_space.py:166`）在 DCP 下会把非本卡 owner 的写入映射到 sink 0。也就是说，即使沿用"先 gather 再写全部行"，存储结果按 owner 规则也是正确的，只是多写了被丢弃的行（推断）。另外 DSA 的每个 query 本来就需要所有新 token 的 KV，新 token 的 all-gather 无论切分方式如何都要做。由此推断，"计算者 = owner"主要省的是 KV 写入和让新旧 KV 用一次 DCP gather 完成，而不是省掉新 token 的通信。这会影响计划 §3.3 对 zigzag 的判断（Z1 的 all-to-all 也许可以被"PCP gather 新 token + DCP gather 前缀 + translate_write 过滤"代替），建议评审时确认。（计划未提及，推断）
