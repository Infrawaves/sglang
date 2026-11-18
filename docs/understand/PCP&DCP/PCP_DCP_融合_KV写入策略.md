# PCP + DCP 融合（方案 A）：KV 写入策略修改方案

> 状态：P1、P3 已实现（§8、§9），待 GPU 验证
> 代码基线：`yhz/test/v0.5.23-k31` 分支（行号已在该分支上核对过）
> 关联文档：`PCP_DCP_方案A_计划二.md` §3.4、§4.4；`方案A计划问题汇总.md` §4；`PCP_DCP_融合_CP切分策略.md`
> 文中的代码片段只是改法示意，**没有运行验证**。标"（推断）"的条目确定性较低。

## 0. 结论

方案 A 下，每个 token 的 MLA latent KV 和 indexer K 只能写在一张卡上，也就是它的 owner 卡。KV 写入分两个阶段推进：

| 阶段 | 写入方式 | 正确性靠什么保证 | 需要改的代码 |
|---|---|---|---|
| **P1** | 沿用现有 PCP 的做法：先在 CP 组内 gather 本轮新 token 的 KV，**每张卡都拿着全部新 token 和完整的 `out_cache_loc` 去写** | 写入核心对放大后的 slot 做 owner 过滤，不归本卡的 slot 一律丢弃 | **不改代码**，只加单元测试和断言 |
| **P3** | 去掉新 token 的 gather，**每张卡只写自己算出来的 token** | 计算卡就是 owner，这需要先改成按位置取模的切分（`PCP_DCP_融合_CP切分策略.md`） | 3 个 gather 入口、2 个写入调用点，并新增一个只给 KV 写入用的 `out_cache_loc` 子集 |

P1 能做到不改代码，依赖 owner 过滤，下面第 1 节逐层说明。P3 是性能优化，每层可以省掉一次新 token 的 KV all-gather。

---

## 1. `mem_cache/kv/alloc/id_space.py`：不改，补测试（P1）

### 1.1 为什么不用改：写入路径已经做了 owner 过滤

**MLA latent KV 的写入链路：**
1. DSA 后端用 `forward_batch.out_cache_loc` 调用 `store_mla`（`layers/attention/dsa_backend.py:2609-2618`、`3015-3024`、`4573`）。
2. `KVLayerRouter.store_mla` 调用 `_write_loc`（`mem_cache/kv/router.py:108-149`），用这个子池的 `IdSpace.write` 把虚拟写入位置转成物理行号。
3. DCP 下 `IdSpace.write` 是 `UnitIds.translate_write`（`id_space.py:211-222` 的 `token_space(widened_writes=True)`）。
4. `translate_write`（`id_space.py:163-185`）用 `attn_dcp_size` 和 `attn_dcp_rank` 调用 `translate_token_ids`。
5. `translate_token_ids_kernel`（`python/sglang/kernels/ops/memory/virtual_slot.py:199-262`）的第 249-252 行：
   ```python
   keep = mask & (loc >= 0)
   if DCP_SIZE > 1:
       keep = keep & ((loc % DCP_SIZE) == DCP_RANK)
       loc = loc // DCP_SIZE
   ...
   tl.store(out_ptr + out_off, tl.where(keep, ids, 0), mask=in_range)
   ```
   不归本卡的 slot 会被转成 0。0 是 padding sink，所有写入 kernel 都会跳过它。CPU 路径（`virtual_slot.py:323-` 的纯 torch 参考实现）的逻辑相同。

所以 P1 下 8 张卡都拿着同样的 16384 个新 token 和同一个 `out_cache_loc` 去写，每个 slot 只会被它的 owner 卡真正写入。这是 DCP 写入的现有机制，开启 PCP 不影响它。

### 1.2 owner 规则为什么等价于"按位置取模"

kernel 判断 owner 用的是**放大 slot 号** `loc % dcp`，不是 token 在序列里的位置。二者相等，依赖下面两点：
- DCP 下分配器的页大小放大为 `page_size × dcp`（`mem_cache/allocation_sizing.py:22-25` 的 `get_alloc_page_size`，以及 `kv/alloc/chain.py:207-210`）。每个放大页的起始 slot 都是 `page_size × dcp` 的倍数。
- 一个请求的 token 在页内连续分配，而且第 0 个 token 落在页起点。所以位置 `p` 的 slot 满足 `slot ≡ p (mod page_size × dcp)`，进而 `slot % dcp == p % dcp`。

DSA 后端读本卡 KV 时也依赖同一个等式：`dcp_local_kv_rows`（`dsa_backend.py:1777-1786`）用 `filter_dcp_local_chunk_kv_indices` 按位置挑出 `p % dcp == rank` 的行。这个等式由现有的分配器保证，方案 A 没有改动分配器，所以它仍然成立。

### 1.3 要加什么测试（P1）

**为什么要测：** P1 的正确性完全依赖"每个 slot 只被一张卡写入，而且一定被写入"。这个性质以前只在"每张卡都算全部 token"的普通 DCP 下被验证过。方案 A 的场景不一样：写入数据来自 CP gather，各卡拿到的 `out_cache_loc` 必须完全相同。一旦有人改了 CP 路径，例如让 `out_cache_loc` 按卡变化，就可能出现漏写或重复写，而且不会报错。

**测试内容（CPU，放在 `test/registered/unit/mem_cache/` 下）：**
```python
# published_topology(tp_size=N, dcp_size=N)；UnitIds(device="cpu") 绑定恒等映射
loc = 放大 slot 序列（覆盖多页、页内偏移、-1 padding）
kept = []
for rank in range(N):
    with get_parallel().override(attn_dcp_rank=rank, dcp_rank=rank):
        kept.append(unit_ids.translate_write(loc) != 0)
# 1) 对每个有效 slot，恰好一个 rank 的 kept 为真（不漏写，不重复写）
# 2) 为真的那个 rank == slot % N
# 3) -1 padding 在所有 rank 上都映射为 0
```
`translate_token_ids` 在 CPU 上走纯 torch 实现（`virtual_slot.py:323-`），所以这个测试可以在 `base-a-test-cpu` 里运行。

---

## 2. `layers/attention/dsa/dsa_indexer.py`：P1 不改；P3 去掉 gather

### 2.1 现状：indexer K 怎么写入

`_get_q_k_bf16` 在 CP 生效时，把本卡的 key gather 成整批新 token 的 key：
- 双流分支（1073-1086 行）：在 `alt_stream` 上调用 `materialize_full_indexer_k_cache`（1081-1083 行）。
- 普通分支（1093-1095 行）：
  ```python
  # allgather+rerrange
  if is_cp_active(forward_batch):
      key = get_cp_strategy().materialize_full_indexer_k_cache(key, forward_batch)
  ```
之后 `forward_cuda` 在 2457-2470 行调用 `_store_index_k_cache(forward_batch, layer_id, key=key, ...)`。没有显式传 `out_cache_loc`，所以用的是 `forward_batch.out_cache_loc`（2143-2144 行）。

### 2.2 P1 为什么不改：融合 store 本身做 owner 过滤

`_store_index_k_cache`（2125 行起）在 2152 行读取 `_index_k_dcp()`（273-279 行）。K3.1 在 DCP 下返回 `(attn_dcp_size, attn_dcp_rank)`。然后 2163-2172 行：
```python
if dcp_size > 1:
    # K3.1 shards index-K by token: the kernel keeps this rank's rows
    fused_store_index_k_cache(key, buf, out_cache_loc, pool.index_page_size,
                              dcp_size=dcp_size, dcp_rank=dcp_rank)
```
`fused_store_index_k_cache`（`python/sglang/kernels/ops/kvcache/dsa_fused_store_index_cache.py:76-86`）的文档写明：`out_cache_loc` 里是放大后的 slot，kernel 只写本卡的 slot，写入行号是 `slot // dcp_size`。这和 MLA KV 的 owner 规则一致。

所以 P1 下 gather 得到的全部新 token key，加上完整的 `out_cache_loc`，经过 kernel 过滤后，每个 slot 只在 owner 卡上落盘。**P1 不改代码。**

**P1 要核对的两点：**
1. **只能走融合 store。** 2185-2188 行有一条断言："DSA index-K writes under DCP need the CUDA fused store"。所以非 CUDA 平台和 fp8 fnuz 平台不支持方案 A。参数校验阶段应该把它们拒掉；现有的 `fi_a2a` 只能在 CUDA 上用，间接满足这个要求（推断）。
2. **双流分支的写入顺序。** 2457-2463 行在 `alt_stream` 上写 indexer K，2463 行 `current_stream.wait_stream(self.alt_stream)` 之后才继续执行。打分阶段的 `_dcp_gather_index_k`（1925 行）要从 cache 读本卡的行，所以必须等写入完成。现有代码已经保证了这个顺序。

### 2.3 P3 怎么改

**目的：** 去掉每层一次的新 token key all-gather，每张卡只写自己算出来的 key。

**前提：** 已经完成按位置取模的切分，本卡的行正好是它拥有的 token；`prepare_cp_forward` 提供了本卡行对应的写入位置（见 §4）。

1. **1081-1083 行和 1094-1095 行：** DCP 生效时跳过 `materialize_full_indexer_k_cache`，直接返回本卡的 key：
   ```python
   if is_cp_active(forward_batch) and not dsa_shards_index_k():
       key = get_cp_strategy().materialize_full_indexer_k_cache(key, forward_batch)
   ```
   这里用 `dsa_shards_index_k()` 判断，等价于 K3.1 加 DCP。其他模型的 PCP 行为不变。

   **为什么可以跳过：** DCP 下打分不读这个 gather 出来的 key，而是从 cache 读整个序列的 indexer K（`_get_topk_ragged` → `_dcp_gather_index_k`，1713-1716 行、1925-1951 行）。所以 gather 出来的 key 只用于写入（推断：需要核对 2380 行的 graph split 分支，以及 k-only 快速路径有没有其他使用者）。
2. **2457-2470 行：** 调用 `_store_index_k_cache` 时显式传入本卡行的写入位置：
   ```python
   self._store_index_k_cache(forward_batch=forward_batch, layer_id=layer_id, key=key,
                             act_quant=act_quant,
                             out_cache_loc=forward_batch.cp_local_out_cache_loc)
   ```
   `_store_index_k_cache` 本来就有 `out_cache_loc` 参数（2132 行），不用改它的签名。本卡行全部归本卡所有，kernel 会把它们全部写入。
3. **断言：** 本卡 key 的行数要等于 `cp_local_out_cache_loc.numel()`，并且写入位置满足 `loc % dcp == rank`。这一条用来防止切分规则和写入位置对不上，导致悄悄漏写。

---

## 3. `layers/cp/interleave.py` 中三个 gather 入口：P1 保留，P3 改成只写本地

### 3.1 三个函数各自的作用

| 函数 | 行号 | 被谁调用 | 作用 |
|---|---|---|---|
| `materialize_full_indexer_k_cache` | 360-363 | `dsa_indexer.py:1082、1095` | 把本卡的 indexer key gather 成整批 |
| `materialize_full_mla_kv` | 401-415 | `dsa_backend.py:152-174` 的 `prepare_kv_for_attention`（被 `forward_mla.py:866` 调用），以及 `DSAInterleaveCPIndexerOverlap.prepare_qkv` 第 529-535 行 | 把本卡的 MLA latent KV（`k_nope`、`k_rope`）gather 成整批 |
| `run_indexer_with_kv_gather` | 560-604 | `indexer_with_cp_kv`（539-557 行），由 `DSAInterleaveCPIndexerOverlap.run_indexer`（473-487 行）和 `prepare_qkv`（514-523 行）调用 | 在侧流上 gather MLA KV，和 indexer 的计算重叠执行，返回 `kv_full` |

这三个函数都经过 `gather_kv_cache` → `_gather_interleaved_tensor`（286-355 行）。gather 结果按序列顺序排列，之后由 `store_mla`（`dsa_backend.py:2609-2618` 等）用完整的 `forward_batch.out_cache_loc` 写入。

### 3.2 P1 为什么保留

现有 PCP 的语义是"KV 在 CP 组内复制，每张卡写完整一份"，这三个函数就是为此设计的。方案 A 下，写入核心的 owner 过滤（§1.1）会把非 owner 的写入丢掉，结果变成"每个 slot 只在 owner 上落盘"，正确性没有问题。代价是每层多做一次新 token 的 KV all-gather，每卡每层大约 16384 × 576 B ≈ 9 MB 的 FP8 latent（推断）。这个开销在 P3 处理。

**P1 要注意：** 计划二 §3.4 要求"关闭 CP 和 DCP 的重复 KV 路径"。这里指的是 **attention 读 KV** 时不要同时用 CP gather 出来的新 token KV 和 DCP 整序列 gather 出来的 KV。**写入用的 gather 在 P1 要保留**，因为 P1 的写入依赖它把全部新 token 带到每张卡上。

不过严格来说，P1 下只要计算卡算出的 token 能写到它的 owner 上就够了。但现有切分规则下计算卡和 owner 不一致（`PCP_DCP_融合_CP切分策略.md` §0.1），所以只能靠 gather 把数据带到 owner 卡上。

### 3.3 P3 怎么改

**前提：** 同 §2.3。

1. **`materialize_full_mla_kv`（401-415 行）**：DCP 生效时不做 gather，原样返回本卡的 `k_nope`、`k_rope`：
   ```python
   def materialize_full_mla_kv(self, forward_batch, layer, k_nope, k_rope):
       if forward_batch.attn_cp_metadata.gather_index is not None and get_parallel().dcp_enabled:
           return k_nope, k_rope        # each rank stores only the tokens it owns
       ...  # 原逻辑
   ```
   更推荐在调用方（`dsa_backend.py:161-166` 的 `prepare_kv_for_attention`）里判断，直接跳过调用。这样 `materialize_full_mla_kv` 的语义保持不变，仍然表示"得到完整的 KV"。
2. **`materialize_full_indexer_k_cache`（360-363 行）**：在调用方 `dsa_indexer.py` 里跳过（§2.3 第 1 步），这个函数本身不改。
3. **`run_indexer_with_kv_gather` / `indexer_with_cp_kv`（539-604 行）**：DCP 生效时不应该再进入"KV gather 与 indexer 重叠"的路径。改法是让 `DSAInterleaveCPIndexerOverlap._can_overlap`（452-471 行）在 `dsa_shards_index_k()` 为真时返回 False，这样 `run_indexer` 会走 476 行的普通 `attention.indexer(...)`，`prepare_qkv` 会走 528-535 行，再配合第 1 步跳过 gather。注意 forward_mla 的文档提到，`cp_overlap` 在 `forward_mla.py:472-480` 构造，666-675 行调用 `run_indexer`。
4. **写入位置**：`dsa_backend.py` 里调用 `store_mla` 的 3 处（2609-2618、3015-3024、4573 行），在 DCP 加 PCP 时要改用 `forward_batch.cp_local_out_cache_loc`（§4）。其中 4573 行在 `_forward_trtllm` 里，那里还有一段自己的 CP KV gather（4508-4556 行的 `all_gather_dsa_trtllm_fp8_kv` 和 `cp_live_store_rows`），也要一起跳过。第一期限定只用 `flashmla_kv`，所以 trtllm 这条路径可以放到以后。

**为什么 P3 能安全地去掉 gather：** DCP 下 prefill 的 attention 和打分都从 cache 读**整个序列**的 KV 和 indexer K（`dsa_backend.py:1392` 的 `dcp_gather_kv_cache`，`dsa_indexer.py:1925` 的 `_dcp_gather_index_k`）。本轮新 token 的 KV 先写进各自 owner 的 cache，再由 DCP gather 收齐。每张卡读的只是自己 cache 里的行，写和读在同一个 stream 上，前后顺序有保证，不会读到其他卡还没写完的数据。所以 CP gather 得到的新 token KV 只在写入时用到，attention 并不需要它（推断：需要在 P3 中逐个核对 `k_nope`/`k_rope` 返回之后的所有使用点）。

---

## 4. `layers/cp/utils.py`：P3 新增本卡行的写入位置

**位置：** `prepare_cp_forward` 第 303-304 行：
```python
if getattr(forward_batch, "out_cache_loc", None) is not None:
    forward_batch.out_cache_loc = forward_batch.out_cache_loc[:num_tokens]
```
这里只截掉了尾部的 padding，没有按卡切分。

**P1：** 不改。写入用的是完整的 `out_cache_loc`，配合 gather 后的完整 KV。

**P3：** 在这里新增一个字段，**不要覆盖 `out_cache_loc`**。logits、投机解码、mamba 状态跟踪等路径还在读完整的 `out_cache_loc`（推断）：
```python
metadata = forward_batch.attn_cp_metadata
if forward_batch.out_cache_loc is not None and metadata.gather_index is not None:
    forward_batch.cp_local_out_cache_loc = forward_batch.out_cache_loc.index_select(
        0, metadata.local_index
    )
```
`local_index` 由按位置取模的 `build_metadata` 生成（`PCP_DCP_融合_CP切分策略.md` §1.1）。新字段要加在 `ForwardBatch` 上（`model_executor/forward_batch_info.py`）。仓库规则 `.claude/rules/forward-batch-init-new-purity.md` 要求 `init_new` 不修改 `ScheduleBatch`。这个字段在 `prepare_cp_forward` 里写到 `ForwardBatch` 上，不碰 `ScheduleBatch`，所以不违反这条规则。

---

## 5. `layers/dcp/planner.py`：不改；前缀对齐由现有调度保证

### 5.1 第 117 行要求什么

`prepare_decode_context_parallel_metadata` 的 113-118 行：
```python
# Prefix lengths are dcp_size-aligned (widened allocator page), so no nonzero().
runner_ids = get_attn_backend().runner_ids
dcp_local_prefix_kv_indices = runner_ids.translate_dcp_read_ids(
    dcp_prefix_kv_indices[parallel.dcp_rank :: parallel.dcp_size]
)
```
`dcp_prefix_kv_indices` 是**所有请求的前缀**按顺序拼起来的 slot 列表，然后用步长 `dcp_size` 取出本卡的行。只有**每个请求的前缀长度都是 `dcp_size` 的倍数**，后一个请求的起点相对 rank 的偏移才不会错位，步长切片才能正好取到 `slot % dcp == rank` 的那些行。

### 5.2 为什么不用改：对齐由调度器保证，PCP 不影响调度器

前缀来自两个地方：
1. **radix cache 命中的前缀**：按页匹配，页大小是 `page_size`。K3.1 强制 `page_size=64`。
2. **chunked prefill 前几个 chunk 留下的前缀**：`managers/schedule_policy.py:1790-1797` 在非 exact-fill 分支里把 chunk 的结束位置对齐到页：
   ```python
   extend_len = chunk_tokens_limit // self.page_size * self.page_size
   ...
   end = (prefix_len + extend_len) // self.page_size * self.page_size
   extend_len = end - prefix_len
   ```
   exact-fill 分支（1779-1789 行）不对齐页，但它只在 gfx95 上开启（87-94 行的 `_use_exact_chunk_fill`），而且有 mamba 前缀共享时会关闭（703-713 行）。K3.1 是 KDA（mamba 类）模型，跑在 CUDA 上，所以不会走这个分支。

所以前缀长度总是 64 的倍数。只要 `64 % dcp == 0`（dcp 取 2 的幂且不超过 64；K3.1 的 DCP 校验要求 dcp 是 2 的幂），每个请求的前缀就都是 dcp 的倍数。方案 A 只改 CP 的计算切分，**不改调度器、分配器和 radix cache**，这个性质保持不变。

**还要补充一点（推断）：** 对 K3.1 来说，planner 生成的 `dcp_kv_buffer` 和 `dcp_local_prefix_kv_indices`，只被 `bcg_dcp_extend_kv_gather` 使用（`forward_mla.py:109-137`）。而 DSA 模型不会调用这个函数（`forward_mla.py:884` 的条件是 `if not self.use_dsa`）。K3.1 实际读本卡 KV 用的是 `dsa_backend.py:1777-1786` 的 `dcp_local_kv_rows`，它按请求分别计算，不依赖前缀对齐。所以第 117 行的对齐要求在 K3.1 上主要影响元数据能否正确构建，不影响 attention 的结果。

### 5.3 建议加的防护（P1，可选）

- 在 `planner.py` 第 113 行加断言，把隐含前提写成代码：
  ```python
  assert all(int(p) % parallel.dcp_size == 0 for p in extend_prefix_lens_cpu), (
      "DCP prefix lengths must be dcp_size-aligned"
  )
  ```
  这里遍历 CPU 列表，开销很小，不会引入 GPU 同步。
- 在参数校验里显式要求 `page_size % dcp_size == 0`。HiCache 主机池在 `mem_cache/pool_host/base.py:242` 已经有同样的断言，但没开 HiCache 时不会检查到。

---

## 6. 修改清单汇总

| # | 文件 | 行号 | 改动 | 阶段 |
|---|---|---|---|---|
| 1 | `mem_cache/kv/alloc/id_space.py` | 163-185 | **不改**；新增 CPU 单元测试：每个 slot 恰好被一个 rank 写入，而且这个 rank 就是 `slot % N`（§1.3） | P1 |
| 2 | `layers/attention/dsa/dsa_indexer.py` | 1081-1083、1094-1095 | P1 不改；P3 在 DCP 生效时跳过 `materialize_full_indexer_k_cache` | P3，已实现 |
| 3 | `layers/attention/dsa/dsa_indexer.py` | 2457-2470 | P3 调用 `_store_index_k_cache` 时传入 `cp_local_out_cache_loc`，并断言行数一致 | P3，已实现 |
| 4 | `layers/attention/dsa_backend.py` | 161-166 | P3 在 `prepare_kv_for_attention` 里跳过 `materialize_full_mla_kv` | P3，已实现 |
| 5 | `layers/cp/interleave.py` | 452-471 | P3 让 `_can_overlap` 在 `dsa_shards_index_k()` 为真时返回 False | P3，已实现 |
| 6 | `layers/attention/dsa_backend.py` | 2609-2618、3015-3024、4573 | P3 的 `store_mla` 改用 `cp_local_out_cache_loc`；trtllm 路径以后再改 | P3，已实现（§9.2 第 3、4 条） |
| 7 | `layers/cp/utils.py` + `model_executor/forward_batch_info.py` | 303-304 | P3 新增 `cp_local_out_cache_loc`，不覆盖 `out_cache_loc` | P3，已实现（字段放在 CP metadata 上，§9.2 第 1 条） |
| 8 | `layers/dcp/planner.py` | 113 | **不改逻辑**；可选加前缀对齐断言 | P1（可选） |
| 9 | 参数校验 | — | 可选：要求 `page_size % dcp_size == 0` | P1（可选） |
| — | `layers/cp/interleave.py` 的三个 gather 入口 | 360-363、401-415、539-604 | **函数本身不改**，P3 在调用方跳过 | — |

## 7. 建议的测试

| 阶段 | 测试 | 类型 |
|---|---|---|
| P1 | §1.3：用 `translate_write` 验证每个 slot 恰好被一个 rank 写入 | CPU 单元测试 |
| P1 | 方案 A 和 TP8+DCP8 用同一批输入（带前缀、多请求）做 prefill，再把每张卡 cache 里的 MLA KV 和 indexer K 按 owner 拼回完整序列，两者逐字节比较 | GPU 集成测试 |
| P3 | 去掉 gather 前后，上一条 GPU 测试的 cache 内容完全一致 | GPU 集成测试 |
| P3 | `cp_local_out_cache_loc` 的每个元素都满足 `loc % N == rank`，所有 rank 合起来正好覆盖全部有效 slot | CPU 单元测试 |

## 8. P1 已实现的改动和测试运行方式

修改清单第 6 节里属于 P1 的第 1、8、9 项已经实现：

| # | 文件 | 改动 |
|---|---|---|
| 1 | `test/registered/unit/mem_cache/test_full_loc_fast_path.py` | 新增 `TestDcpWriteOwnership`：取 DCP=4，每页 4 行，虚拟页 1 映射到物理页 3。4 个 rank 用同一组放大后的 slot（16–31，再加一个 -1 padding）调用 `translate_write`，验证每个 slot 只有一个 rank 写入，而且这个 rank 是 `slot % 4`；写入的行号是 `12 + (slot-16)//4`；padding 在所有 rank 上都映射到空槽 0。`id_space.py` 本身没有改 |
| 8+9 | `python/sglang/srt/layers/dcp/planner.py` | 新增 `_check_prefill_cp_with_dcp_alignment`，在 `prepare_decode_context_parallel_metadata` 开头调用，**只在 `attn_cp_size > 1` 时生效**。它断言两点：`page_size % dcp_size == 0`；每个请求的前缀长度都是 `dcp_size` 的倍数 |
| — | `test/registered/unit/layers/attention/test_dcp_page_layout_cpu.py` | 新增 `TestPrefillCpWithDcpAlignment`：前缀对齐时通过；前缀不对齐时报错，报错信息会列出不对齐的前缀长度；page_size 不能被 dcp 整除时报错 |

和第 5.3 节的设想相比，实现上有两处不同：

1. **断言只在开启 prefill CP 加 DCP 时生效。** gfx95 上的 exact chunk fill 不会把 chunk 结尾对齐到页，所以 AMD 上已有的 DSA+DCP 部署可能出现前缀不对齐。这些部署并不使用 planner 的结果，如果对所有 DCP 都做这个断言，会让它们直接崩溃。
2. **第 9 项放在运行时检查，没有放在参数校验里。** K3.1 的 `page_size=64` 是由 `arg_groups/overrides.py` 里的声明式 override 设置的，我没能确认它和各个 hook 谁先执行。如果在 hook 里检查，读到的可能还是没被 override 的默认值，导致误报。planner 运行时通过 `get_schedule().page_size` 读到的是最终值。

两个测试文件都注册在 `base-a-test-cpu`。在装好 sglang 的环境里，从仓库根目录运行：

```bash
python -m pytest test/registered/unit/mem_cache/test_full_loc_fast_path.py -k TestDcpWriteOwnership
python -m pytest test/registered/unit/layers/attention/test_dcp_page_layout_cpu.py -k TestPrefillCpWithDcpAlignment
```

回归时，两个文件里原有的用例也要一起跑：

```bash
python -m pytest test/registered/unit/mem_cache/test_full_loc_fast_path.py test/registered/unit/layers/attention/test_dcp_page_layout_cpu.py
```

这些测试还没有实际运行过。开发机上只有 Python 3.9，没有装 torch，只做过语法解析检查。

## 9. P3 已实现的改动

前提是按位置取模的切分已经实现（`PCP_DCP_融合_CP切分策略.md` §11）。所有分支都用 `writes_owned_kv_only(forward_batch)` 判断（`layers/cp/interleave.py:104-110`）：当前是 interleave 的 CP extend，并且 metadata 里有 `local_sequence_rows`。只有 prefill CP + DCP 才会满足这个条件。

### 9.1 改动清单

| 文件 | 位置 | 改动 |
|---|---|---|
| `layers/cp/utils.py` | `prepare_cp_forward`，305-315 行 | 按位置切分时，把 `out_cache_loc.index_select(0, local_sequence_rows)` 存到 `metadata.local_out_cache_loc`。`forward_batch.out_cache_loc` 本身不改 |
| `layers/cp/interleave.py` | 113-131 行 | `owned_kv_write(forward_batch, *rows)`：返回本卡写入位置，以及每个本卡张量的有效前缀（本卡分片被补齐到物理长度，尾部是 padding）。断言写入位置的行数等于本卡的逻辑行数 |
| `layers/attention/dsa_backend.py` | `prepare_kv_for_attention`，154-179 行 | 按位置切分时直接返回本卡的 `k_nope`、`k_pe`，不调用 `materialize_full_mla_kv` |
| `layers/attention/dsa_backend.py` | `forward_extend`，2679-2680 行 | `store_mla` 前，按位置切分时用 `owned_kv_write` 换成本卡写入位置和有效行 |
| `layers/attention/dsa_backend.py` | `_forward_trtllm`，4639-4643 行 | 不走 fused-rope 分支时同样改写本卡行（见 §9.2 第 4 条） |
| `layers/cp/interleave.py` | `_can_overlap`，593 行起 | 按位置切分时返回 False，不再走"KV gather 与 indexer 重叠"的路径 |
| `layers/attention/dsa/dsa_indexer.py` | `_get_q_k_bf16`，1076-1102 行 | 两处 `materialize_full_indexer_k_cache` 在按位置切分时都跳过，返回本卡的 key |
| `layers/attention/dsa/dsa_indexer.py` | `forward_cuda`，2464-2490 行 | 两处 extend 路径的 `_store_index_k_cache` 改传 `owned_kv_write` 得到的写入位置和本卡 key |

### 9.2 和 §2–§4 设想的不同之处

1. **写入位置放在 CP metadata 上，不加 `ForwardBatch` 字段。** 字段名是 `InterleaveContextParallelMetadata.local_out_cache_loc`。它和 `local_sequence_rows` 同时生成、同时失效：`eager_runner.py:328` 和 `bcg.py:190` 在每次前向后把 `attn_cp_metadata` 置空，这个字段跟着一起清掉，不会残留到下一次前向。加在 `ForwardBatch` 上就需要另外处理清理。
2. **判断条件用 `writes_owned_kv_only`，不用 `dsa_shards_index_k()`。** 两者在方案 A 下等价，但前者直接表达"本卡只算自己拥有的 token"，这才是能跳过 gather 的原因。MLA KV、indexer K 和 overlap 三处共用这一个判断。
3. **`dsa_backend.py` 的 3075 行（文档原写 3015-3024）没有改。** 它在 `forward_decode` 里，decode 不走 CP 切分。
4. **trtllm 路径也做了最小处理。** `_forward_trtllm` 的 fused-rope 分支自己做 gather（`all_gather_dsa_trtllm_fp8_kv`），写完整的 `out_cache_loc`，结果仍然正确，所以不改。但另一个分支原本依赖 `prepare_kv_for_attention` 已经 gather 过 KV，现在 KV 是本卡的，所以这个分支也改成写本卡行，否则行数和写入位置会对不上。方案 A 第一期只用 flashmla_kv，这条路径不会执行到。

### 9.3 为什么去掉 gather 后结果不变（逐项核对）

§3.3 末尾标了"推断：需要逐个核对 `k_nope`/`k_rope` 返回之后的所有使用点"。实现时核对的结果：

| 使用点 | 位置 | 结论 |
|---|---|---|
| MLA 的 `store_mla` 之后 | `dsa_backend.py:2705-2709` | DCP 下 MLA 路径从 cache 读 KV，再用 `dcp_gather_kv_cache` 收齐整条序列，不使用传入的 `k` |
| MHA_ONE_SHOT 分支 | `dsa_backend.py:2688-2701`、4953-4967 | 会直接使用 `k`，但开启 prefill CP 时 `use_mha` 恒为 False（`not is_dsa_enable_prefill_cp()`） |
| DCP extend 的 KV gather | `forward_mla.py:884-891` | DSA 模型不调用 `bcg_dcp_extend_kv_gather` |
| decode 提前写 KV | `forward_mla.py:540-611` | 只在 decode/verify 下执行 |
| indexer 打分 | `dsa_indexer.py:1719-1727` | DCP 下从 cache 读 `_dcp_gather_index_k`，不使用 gather 出来的 key |
| indexer 的 k-only 快速路径 | `dsa_indexer.py:2325` | 开启 prefill CP 时不走（`not self.dsa_enable_prefill_cp`） |
| indexer 的融合 store 路径 | `dsa_indexer.py:2373-2390` | CP 生效时不走（`attn_cp_metadata is None` / `not is_cp_active`） |

### 9.4 测试

CPU 测试在 `test/registered/unit/cp/test_interleave_dcp_owner_layout.py`。其中 `test_local_kv_writes_land_on_the_owner` 对应 §7 的 P3 CPU 测试：在"slot 和位置模 N 同余"（§1.2）的前提下，各卡的写入位置都满足 `slot % N == rank`，并且所有卡合起来正好覆盖全部 slot。运行方式见 `PCP_DCP_融合_CP切分策略.md` §11.3。这些测试还没有运行过。

§7 的 P3 GPU 测试（去掉 gather 前后 cache 内容逐字节一致）还没有做。
