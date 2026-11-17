# PCP + DCP 融合（方案 A）：Prefill Attention 修改方案

> 状态：草案，待人工评审
> 代码基线：`yhz/test/v0.5.23-k31` 分支（行号在该分支上核对过）
> 关联文档：`PCP_DCP_方案A_计划二.md` §3.4、§4.5；`方案A计划问题汇总.md` §5；`PCP_DCP_融合_CP切分策略.md`；`PCP_DCP_融合_KV写入策略.md`
> 文中的代码片段只是改法示意，**没有运行验证**。标"（推断）"的条目确定性较低。

## 0. 结论

对计划二 §4.5 列出的 7 个文件逐一核对代码后，结论如下：

| 文件 | 计划二的要求 | 核对后的结论 |
|---|---|---|
| `layers/attention/dsa_backend.py` | ①②③ | **①② 必须改，P1 的核心改动在这里**；③ 不需要（§1.4） |
| `layers/attention/dsa/dsa_indexer.py` | 长度和 backend 一致 | **必须改**：`_dcp_gather_index_k` 改用完整的请求列表（§2） |
| `forward_mla.py` | 关闭 CP 的 KV gather 和 overlap | **P1 不改**。这些 gather 是 P1 写 KV 必须的（§3） |
| `models/kimi_k3.py` 的 `prepare_context_parallel_metadata_for_dcp` | 签名加 CP 元数据 | **不改**。K3.1 的 DSA 路径不使用它生成的元数据（§4） |
| `model_executor/runner/eager_runner.py` | 传入 CP 元数据 | **不改**（§4） |
| `layers/dcp/metadata.py`、`layers/cp/base.py` | 新增合并后的元数据 | **不改**。需要的字段直接加在 `DSAMetadata` 上（§5） |
| `prefill_cuda_graph_runner.py` | P3 保留"有前缀命中就走 eager" | 第一期不改。这条规则在 1509-1512 行（§6） |

**P1 唯一必须修的问题：** CP 生效时，DSA 后端会从 query 侧的元数据中删掉"本卡没有 query 行的请求"。但 DCP 的 KV 和 indexer K gather 是一次跨卡的集合通信，要求所有卡用**同一份**完整的请求列表。现在 KV 侧沿用了按卡过滤之后的列表，所以各卡的 gather 输入形状不同，可能悄悄算错，也可能让 collective 挂住。

---

## 1. `layers/attention/dsa_backend.py`

### 1.1 问题出在哪：CP 会按卡删除请求

DSA 后端在 `init_forward_metadata` 的 extend 分支（1591 行起）构建 prefill 元数据。CP 生效时（1622-1638 行）：

```python
if can_dsa_prefill_cp_interleave(forward_batch):
    strategy = get_cp_strategy()
    seqlens_expanded = strategy.shard_local_tokens(seqlens_expanded)
    extend_seq_lens_cpu, extend_seq_lens, bs_idx_cpu, bs_idx = (
        strategy.shard_per_request(extend_seq_lens_cpu, extend_seq_lens)
    )
    indexer_seq_lens_cpu = indexer_seq_lens_cpu[bs_idx_cpu]      # 1631
    indexer_seq_lens = indexer_seq_lens[bs_idx]
    cache_seqlens_int32 = cache_seqlens_int32[bs_idx]
    cu_seqlens_k = compute_cu_seqlens(cache_seqlens_int32)
    ...
    page_table = page_table[bs_idx, :max_seqlen_k]
```

`shard_per_request`（`layers/cp/interleave.py:255-279`）只保留在本卡有 query 行的请求（273-274 行：`q_len > 0`）。之后的 `page_table`、`indexer_seq_lens_cpu`、`cache_seqlens_int32` 和 `cu_seqlens_k` 都是**按卡过滤过的**。

**举例：** N=8，一批有两个请求，本轮新 token 数是 `[3, 9]`。按现有的 interleave（拼接下标取模，`shard_per_request` 在请求之间传递余数）：
- 请求 0 的 3 个 token 分给 rank 0、1、2；
- rank 3–7 上请求 0 没有 query 行，会被过滤掉；
- 所以 rank 0–2 的列表是 `[请求 0, 请求 1]`，rank 3–7 的列表只有 `[请求 1]`。

开启 PCP 但不开 DCP 时，这样过滤没有问题：KV 在每张卡上是完整的，各卡只需要读自己 query 用得到的部分。

### 1.2 ① `dcp_gather_kv_cache` 的输入必须用完整请求列表（必须改，P1）

**位置：** `dcp_gather_kv_cache`（1394-1405 行）：
```python
rows = kv_cache.view(torch.uint8).view(-1, self.kv_cache_dim)
gathered = dcp_gather_request_rows(
    rows[metadata.dcp_local_kv_rows], forward_batch.seq_lens_cpu
)
```
它在 prefill attention 读 KV 前被调用（2638-2641 行，trtllm 路径是 4576-4577 行）。

**为什么会出错：**
- `metadata.dcp_local_kv_rows` 在 1777-1786 行生成，用的是**过滤后**的 `page_table_1_flattened` 和 `indexer_seq_lens_cpu`：
  ```python
  dcp_local_kv_rows=(
      filter_dcp_local_chunk_kv_indices(
          page_table_1_flattened,
          torch.zeros_like(indexer_seq_lens_cpu),
          indexer_seq_lens_cpu,
      ) // self.attn_dcp_size
      if self.attn_dcp_size > 1 and page_table_1_flattened is not None else None
  ),
  ```
  `page_table_1_flattened` 本身也是在 1672-1681 行从过滤后的 `page_table` 拼出来的。
- 但 gather 的长度参数用的是 `forward_batch.seq_lens_cpu`，也就是**全部**请求。
- `dcp_gather_request_rows` 内部调用 `all_gather_kv_cache_for_dcp`（`layers/dcp/comm.py:361-450`）。它按长度列表为每个请求补齐，拼出大小为 `sum(padded_lens)` 的 buffer，在 DCP 组里做 all-gather，再按同一份长度列表重排（419-450 行）。**这要求每张卡的输入行和长度列表一一对应，而且所有卡用的是同一份长度列表。**

用 §1.1 的例子：rank 3 的本地行里没有请求 0，长度列表里却有请求 0，于是请求 1 的行被错当成请求 0 的。rank 0 的本地行和长度列表都包含两个请求。各卡交给 all-gather 的数据内容和顺序都不一样，结果是 KV 被错位重排，attention **悄悄算错**。如果各卡补齐后的总长度也不同，`all_gather_into_tensor` 还可能直接报错或者挂住。

**怎么改：** 构建 KV 侧元数据时，保留一份过滤前的数据：
1. 在 1622 行（CP 分支）之前保存：
   ```python
   dcp_page_table, dcp_seq_lens_cpu = page_table, indexer_seq_lens_cpu   # full request list
   ```
2. 1672-1681 行：`self.attn_dcp_size > 1` 时，`page_table_1_flattened` 改用 `dcp_page_table` 和 `dcp_seq_lens_cpu` 来拼。
3. 1777-1786 行：`dcp_local_kv_rows` 同样改用 `dcp_page_table` 和 `dcp_seq_lens_cpu`。
4. `dcp_gather_kv_cache` 的 gather 长度继续用 `forward_batch.seq_lens_cpu`。这时它和本地行一一对应，所有卡也都一样。

`page_table_1_flattened` 还有其他使用者，例如 1355-1372 行把 top-k 转成物理 slot，以及 RAGGED top-k 的变换。改成完整列表之前要逐一确认它们在 DCP 下期望的是哪种布局（推断：DCP 下这个表只被用来生成 `dcp_local_kv_rows` 和 RAGGED 布局）。如果有使用者必须用过滤后的版本，就新增一个字段 `dcp_page_table_1_flattened`，不要直接覆盖原来的。

**另一种思路（不推荐）：** DCP 下让 `shard_per_request` 不删除 query 数为 0 的请求，这样所有列表天然都是完整的。但现在两处代码（`shard_per_request` 第 273-274 行和 kernel 第 28 行）都显式过滤了 0，说明下游 kernel 很可能不接受长度为 0 的 query 段，风险更大。

### 1.3 ② query 侧的偏移要指向完整的 gather 布局（必须改，P1）

gather 之后，KV 和 indexer K 都按"**所有请求**依次排列"的布局存放：请求 b 的 KV 从 `full_cu[b] = sum(seq_lens[:b])` 开始。query 侧只描述本卡的行，但查 KV 用的偏移必须基于这个完整布局。

| 位置 | 现状 | 要改成 |
|---|---|---|
| `topk_indices_offset`，1694-1699 行 | `repeat_interleave(cu_seqlens_k[:-1], extend_seq_lens)`，其中 `cu_seqlens_k` 来自过滤后的 `cache_seqlens_int32`（1632 行） | DCP 下用完整的 `cache_seqlens` 计算 `full_cu`，再取 `full_cu[bs_idx]`，按本卡每个请求的 query 数 `extend_seq_lens` 展开 |
| `_cal_indexer_k_start_end`，1803-1880 行 | 1866-1868 行：`if bs_idx is None or i in bs_idx: k_offset += seq_len`，只累加本卡保留的请求 | DCP 下**每个请求都要累加** `seq_len`，因为 indexer K 的 gather 包含全部请求。ks/ke 先对全部 token 计算，再用 `shard_local_tokens` 取出本卡的行（1875-1877 行），这一步不用改 |
| `cu_seqlens_k`、`max_seqlen_k`，1632-1637 行 | 由过滤后的列表计算 | 核对它们在 DCP prefill 的稀疏 attention 中是否被使用。如果被使用，也要改成完整布局（推断：flashmla_kv 稀疏 prefill 用 top-k 下标直接访问 gather 后的 buffer，主要依赖 `topk_indices_offset`） |

`get_indexer_metadata`（4950-4970 行）只是把 `forward_metadata` 打包交给 indexer，**本身不用改**。indexer 读到的 ks/ke（`get_indexer_kvcache_range`）和 `topk_indices_offset`，就是上表修改后的结果。

`dsa_indexer.py` 里有两条断言，正好可以作为这一节改动的检查点：1808 行的 `assert logits.shape[1] == k_offset`，以及 1826 行对 `topk_indices_offset` 长度的检查。其中 `k_offset = k_fp8.shape[0]` 是 gather 后的行数（1737 行）。

### 1.4 ③ `_forward_trtllm` 不需要加 guard（修正计划二）

`_forward_trtllm`（4424 行起）的 4547-4560 行在 prefill CP 下调用 `all_gather_dsa_trtllm_fp8_kv`，gather 新 token 的 KV，然后用 `cp_live_store_rows` 截取写入行。这和 `materialize_full_mla_kv` 的作用一样，都只是为了写入：每张卡都写全部新 token，再由 owner 过滤保证每个 slot 只落在一张卡上（见 `PCP_DCP_融合_KV写入策略.md` §1）。在 P1 下这是正确的，不需要加 guard。

另外，参数校验阶段已经在方案 A 下拒绝了 flashmla_kv 以外的后端（`arg_groups/model_overrides/kimi_k3.py`），所以第一期根本不会进入 trtllm 路径。4576-4577 行的 `dcp_gather_kv_cache` 会随着 §1.2 的修改一起变正确。

---

## 2. `layers/attention/dsa/dsa_indexer.py`：`_dcp_gather_index_k` 改用完整列表（必须改，P1）

**位置：** `_get_topk_ragged`（1644 行起）的 1713-1716 行：
```python
if dsa_shards_index_k():
    k_fp8, k_scale = self._dcp_gather_index_k(
        layer_id, metadata, block_tables, indexer_seq_lens_cpu
    )
```
其中 `block_tables` 来自 1686-1689 行的 `metadata.get_logical_page_table()`，`indexer_seq_lens_cpu` 来自 1723 行的 `metadata.get_indexer_seq_len_cpu()`，**两者都是按卡过滤后的**。

`_dcp_gather_index_k`（1925-1951 行）读取本卡的 index K，然后调用：
```python
gathered = dcp_gather_request_rows(torch.cat([k_bytes, s_bytes], dim=-1), seq_lens_cpu)
```

**为什么会出错：** 这里的本地行和长度列表虽然彼此一致，但**各卡的列表不同**：§1.1 的例子里，rank 0 有两个请求，rank 3 只有一个。`all_gather_kv_cache_for_dcp` 要求所有卡使用同一份长度列表，否则补齐后的总长度和重排结果都会对不上，问题和 §1.2 一样。

**怎么改：** 在 `DSAMetadata`（372 行起）里加两个字段，用来记录 DCP gather 使用的完整布局，例如 `dcp_logical_page_table` 和 `dcp_seq_lens_cpu`，在 §1.2 第 1 步保存的时候一起填好。然后：
- 1713-1716 行：DCP 下传入这两个完整字段，不再传 `block_tables` 和 `indexer_seq_lens_cpu`；
- 1925-1951 行的 `_dcp_gather_index_k` 本身不用改，只是输入换成完整列表。之后 `k_offset = k_fp8.shape[0]`（1737 行）就等于所有请求的长度之和，和 §1.3 改过的 ks/ke 一致。

**两条断言保持不变，作为检查点：**
- 1807 行：`assert logits.shape[0] == len(seq_lens_expanded)`，检查 query 侧的行数；
- 1808 行：`assert logits.shape[1] == k_offset`，检查 KV 侧的行数；
- 1826 行：`topk_indices_offset` 的长度至少要等于 `q_offset`。

计划二写的 1807 行和 1826 行是对的，`_dcp_gather_index_k` 的实际位置是 1925 行。

---

## 3. `forward_mla.py`：P1 不改（修正计划二）

计划二要求在方案 A 下关闭三处：
- `cp_overlap` 的构造（478-486 行）；
- indexer 的 CP overlap 调用（671-680 行，`cp_overlap.run_indexer`）；
- `prepare_kv_for_attention` 里的 CP gather（862-877 行，即 `cp_overlap.prepare_qkv` 和 `prepare_kv_for_attention`）。

**读代码后的结论是：P1 不能关闭这三处。**

1. **这些 gather 是 P1 写 KV 的手段。** 现有切分下，计算某个 token 的卡和它的 owner 卡不是同一张（`PCP_DCP_融合_CP切分策略.md` §0.1）。只有先 gather，每张卡才能拿到全部新 token 的 KV，再交给 owner 过滤写入（`PCP_DCP_融合_KV写入策略.md` §3.2）。如果在 P1 关掉 gather，owner 卡就拿不到别的卡算出来的 token，KV 会丢失。
2. **它们不会造成 attention 重复读取。** DCP prefill 的 attention 从 cache 读完整序列（2638-2641 行的 `dcp_gather_kv_cache`），不使用 CP gather 得到的新 token KV。所以"CP 和 DCP 两条 KV 路径同时存在"只是多做一次通信，不会让结果重复或出错。gather 出来的 `k_nope`/`k_pe` 只被 `store_mla` 用于写入（推断：P3 去掉 gather 前，要逐一核对它们之后的使用点）。
3. **overlap 在 P1 仍然有用。** `run_indexer_with_kv_gather`（`interleave.py:560-604`）把 KV gather 放在侧流上，和 indexer 的计算并行，能减少 P1 多出来的那次通信的开销。

**写入和读取的先后顺序已经有保证：**
- MLA KV：先 `store_mla`（2609-2618 行），再 `dcp_gather_kv_cache`（2641 行），两者在同一个 stream 上；
- index K：先 `_store_index_k_cache`（`dsa_indexer.py:2457-2470`，双流分支在 2463 行等待），之后才打分。

**`bcg_dcp_extend_kv_gather`（109 行）和 `is_dcp_mla_decode_phase`（197-211 行）不改**，这一点和计划二一致：
- 884 行的条件是 `if not self.use_dsa`，K3.1 不会走 `bcg_dcp_extend_kv_gather`；
- 210 行在 `dsa_shards_index_k()` 为真时把 extend 排除在外，所以 prefill 不会被当成 decode 处理。

**P3** 改成"只写本地"时，才在这三处跳过 gather，具体做法见 `PCP_DCP_融合_KV写入策略.md` §3.3。

---

## 4. `models/kimi_k3.py` 和 `eager_runner.py`：不改（修正计划二）

**计划二的设想：** 给 `prepare_context_parallel_metadata_for_dcp` 加 CP 元数据参数（`kimi_k3.py` 3810 行在语言模型里，4618-4632 行在多模态外壳里），并在 `eager_runner.py` 的两个调用点（280-300 行，以及 `prefill_cuda_graph_runner.py:1320`）传进去。

**读代码后的结论是不需要：**
- 这个方法最终调用 `layers/dcp/planner.py` 的 `prepare_decode_context_parallel_metadata`，生成 `attn_dcp_metadata`，包括 `dcp_kv_buffer`、`dcp_kv_indices`、`dcp_local_prefix_kv_indices`。
- 这些字段的唯一使用者是 `bcg_dcp_extend_kv_gather`（`forward_mla.py:109-137`），而 DSA 模型不会调用它（884 行）。
- K3.1 prefill 真正用到的 KV 侧元数据，是 DSA 后端自己在 `init_forward_metadata` 里构建的 `dcp_local_kv_rows` 等字段。这些字段可以直接从 `forward_batch` 读到完整的请求信息，§1 已经在这里完成了修改。

所以 query 侧和 KV 侧的"合并"在 DSA 后端内部就能完成，不需要跨层传递 CP 元数据。`eager_runner.py` 的调用顺序（先 `prepare_cp_forward`，再 DCP 元数据）本来就对，**也不用改**。

**已有的改动：** `PCP_DCP_融合_KV写入策略.md` §8 在 planner 里加了前缀对齐检查，它只在 prefill CP 加 DCP 时生效，和这一节的结论不冲突。

**附带的问题（不属于方案 A）：** K3.1 在 DCP 下每次 prefill 都会经 planner 分配一个 `dcp_kv_buffer`，大小是 `seq_lens_sum × kv_dim`（planner 119-126 行），但这个 buffer 没人使用。TP8+DCP8 也存在同样的浪费。可以另外考虑让 DSA 模型跳过它。

---

## 5. `layers/dcp/metadata.py` 和 `layers/cp/base.py`：不新增合并元数据（修正计划二）

**计划二的设想：** 新增一个"合并元数据"类型，同时描述 query 侧的本卡行（来自 CP）和 KV 侧的完整行（来自 DCP）。

**结论是不需要：** 按 §4 的分析，K3.1 的合并在 DSA 后端内部完成，只需要给 `DSAMetadata`（`dsa_backend.py:372` 起）加 §2 提到的 `dcp_logical_page_table` 和 `dcp_seq_lens_cpu` 两个字段。`DSAMetadata` 现在已经有 `dcp_local_kv_rows` 这个 DCP 字段（426 行），新字段跟着加在旁边即可。

- `DecodeContextParallelMetadata`（`layers/dcp/metadata.py`）**不改**。
- `BaseContextParallelMetadata` 和 CP 策略（`layers/cp/base.py`）**不改**。

这样也避开了计划二担心的几个问题：msgspec 不能继承 dataclass、元数据不能设 `frozen`、可能和上游的 `DecodeContextParallelStrategy` 重构冲突。`DSAMetadata` 本身是 dataclass，按仓库规则（`.claude/rules/no-dataclasses.md`），已有的 dataclass 可以继续沿用，新增字段时不用顺带迁移。

---

## 6. `model_executor/runner/prefill_cuda_graph_runner.py`：第一期不改

**相关代码：**
- 736-740 行：`self._dcp_extend_active = get_parallel().attn_dcp_size > 1 and hasattr(model, "prepare_context_parallel_metadata_for_dcp")`；
- **1509-1512 行**（计划二写的是 1511 行）：
  ```python
  # Under DCP only fresh prefills replay; a prefix hit needs the cross-rank
  # prefix KV all-gather with per-batch shapes, which stays eager.
  if self._dcp_extend_active and prefix_lens is not None and any(prefix_lens):
      return False
  ```

**为什么第一期不改：** 参数校验阶段已经在方案 A 下拒绝了 prefill CUDA graph（`cuda_graph_hook.py`），默认规则也会关掉 BCG，所以第一期不会进入这个 runner 的回放。

**P3 如果要开启 graph，需要注意：**
- 1509-1512 行要保留。有前缀命中时，KV gather 的形状随批次变化，没法回放。
- 没有前缀的 prefill 也要做跨卡的 KV gather，因为 DCP 下本轮新 token 的 KV 分散在各张卡上。§1.2 改完后，gather 的长度和**完整请求列表**一一对应，graph 的静态 buffer 要按完整列表的最大长度来分配。
- §1.1 的按卡过滤会让各卡的 query 元数据形状不同，graph 的静态形状必须能覆盖每张卡的情况。

---

## 7. 修改清单汇总

| # | 文件 | 行号 | 改动 | 阶段 |
|---|---|---|---|---|
| 1 | `layers/attention/dsa_backend.py` | 1622 行之前 | 在 CP 过滤之前保存完整的 `page_table` 和 `indexer_seq_lens_cpu` | P1 |
| 2 | `layers/attention/dsa_backend.py` | 1672-1681、1777-1786 | DCP 下 `page_table_1_flattened` 和 `dcp_local_kv_rows` 改用完整列表。如果有使用者需要过滤后的版本，就新增字段 | P1 |
| 3 | `layers/attention/dsa_backend.py` | 1694-1699 | `topk_indices_offset` 改用 `full_cu[bs_idx]` | P1 |
| 4 | `layers/attention/dsa_backend.py` | 1866-1868 | `_cal_indexer_k_start_end` 在 DCP 下累加所有请求的 `seq_len` | P1 |
| 5 | `layers/attention/dsa_backend.py` | 372 行起（`DSAMetadata`） | 新增 `dcp_logical_page_table`、`dcp_seq_lens_cpu` | P1 |
| 6 | `layers/attention/dsa/dsa_indexer.py` | 1713-1716 | `_dcp_gather_index_k` 改用完整列表作为输入 | P1 |
| 7 | `layers/attention/dsa_backend.py` | 1632-1637 | 核对 `cu_seqlens_k`、`max_seqlen_k` 在 DCP 稀疏 prefill 中的用法（推断） | P1（核对） |
| — | `dsa_backend.py` 的 `_forward_trtllm` | 4547-4560 | **不改**（§1.4） | — |
| — | `forward_mla.py` | 478-486、671-680、862-877 | **P1 不改**；P3 跳过 gather | P3 |
| — | `kimi_k3.py`、`eager_runner.py` | 3810、4618；280-300 | **不改**（§4） | — |
| — | `layers/dcp/metadata.py`、`layers/cp/base.py` | — | **不改**（§5） | — |
| — | `prefill_cuda_graph_runner.py` | 1509-1512 | 第一期不改 | P3 |

## 8. 建议的测试

| 测试 | 类型 | 说明 |
|---|---|---|
| **请求被过滤时的一致性** | GPU，8 卡 | 构造让部分 rank 拿不到 query 的批次，例如 N=8 时新 token 数为 `[3, 9]`。方案 A 和 TP8+DCP8 用同一批输入，逐层比较 attention 输出和 logits。**这是验证 §1–§2 最直接的用例**。修改前应该能复现出错误或挂住 |
| 元数据的形状不变量 | CPU（mock 一个 DSA 后端） | 对任意一种请求过滤结果，`dcp_local_kv_rows` 的长度等于 `sum(get_dcp_lens(full_seq_lens, N, rank))`；`topk_indices_offset` 的值等于 `full_cu[bs_idx]` 展开后的结果；`ks.max() <= sum(full_seq_lens)` |
| 带前缀、多请求的批次 | GPU | 覆盖前缀长度不同（都是 64 的倍数）、多个请求、chunked prefill 的情况 |
| 回归 | GPU | 不开 DCP 的 TP8+PCP8，以及不开 PCP 的 TP8+DCP8，行为都和修改前一致。§1 的每处改动都只在 `attn_dcp_size > 1` 且请求被过滤时才生效 |

## 9. P1 已实现的改动和测试运行方式

第 7 节修改清单中的第 1–6 项已经实现，第 7 项是核对项，没有改代码：

| # | 文件 | 改动 |
|---|---|---|
| 1、2 | `layers/attention/dsa_backend.py` | 新增 `_DcpFullBatch`（msgspec.Struct）。开启 DCP 和 interleave CP 时，在 CP 删掉"本卡没有 query 行的请求"之前，保存完整批次的 `page_table`、`seq_lens`、`seq_lens_cpu`。`dcp_local_kv_rows` 改由 `_DcpFullBatch.local_kv_rows()` 按完整批次计算 |
| 3 | 同上 | RAGGED 的 `topk_indices_offset` 改为 `compute_cu_seqlens(全部请求的 seq_lens)[:-1][bs_idx]`，指向 gather 后完整布局里的位置 |
| 4 | 同上 | `_cal_indexer_k_start_end` 新增参数 `whole_batch_keys`。设为 True 时，本卡不保留的请求也会计入 `k_offset` |
| 5 | 同上 | `DSAMetadata` 新增 `dcp_real_page_table`、`dcp_seq_lens`、`dcp_seq_lens_cpu` 三个字段 |
| 6 | `layers/attention/dsa/dsa_indexer.py`、`dsa_indexer_metadata.py` | indexer metadata 新增 `get_dcp_whole_batch()`，基类默认返回 None，DSA 实现返回上面三个字段。`_dcp_gather_index_k` 改为直接接收 `(block_tables, seq_lens, seq_lens_cpu)`，调用时优先使用完整批次，拿不到时回退到原来的参数 |

**对原有路径没有影响**：上面所有分支只在 `dcp_full is not None` 时生效，也就是同时开启 DCP 和 interleave CP，并且 CP 生效的时候。纯 TP8+DCP8，以及只开 PCP 不开 DCP 的配置，走的都是原来的代码。

**新增测试**：`test/registered/unit/layers/attention/test_dcp_page_layout_cpu.py` 中的 `TestDcpWholeBatchRows`（CPU，`base-a-test-cpu`）。测试构造两个请求，其中请求 0 有 4 个 token：1 个已缓存，3 个是新 token。这 3 个新 token 的 query 只落在 rank 0–2 上，但位置 3 的 KV 归 rank 3 所有。测试验证每个 rank 交出的 KV 行都按完整批次的顺序排列，覆盖它拥有的所有位置，而且 rank 3 一定会交出位置 3 那一行。

在装好 sglang 的环境里，从仓库根目录运行：

```bash
python -m pytest test/registered/unit/layers/attention/test_dcp_page_layout_cpu.py -k TestDcpWholeBatchRows
```

做回归时，把这个文件里原有的用例一起跑，再加上 indexer 相关的 DSA 单元测试：

```bash
python -m pytest test/registered/unit/layers/attention/test_dcp_page_layout_cpu.py
python -m pytest test/registered/unit/layers/attention/test_dsa_backend_dp_padding.py
```

**还没做的事**：
- 第 8 节建议的 GPU 一致性测试：用 `[3, 9]` 这类会让部分 rank 拿不到 query 的批次，对比方案 A 和 TP8+DCP8 每一层的输出。这个测试需要 8 卡环境，而且方案 A 的 decode 要到 P2 才实现，目前没法端到端运行。
- 第 7 项：核对 `cu_seqlens_k` 和 `max_seqlen_k` 在 DCP 稀疏 prefill 中的用法。

上述测试都还没有实际运行过。开发机上只有 Python 3.9，没有装 torch，只做过语法解析检查。
