# PCP + DCP 融合（方案 A）：CP 切分策略修改方案

> 状态：草案，待人工评审
> 代码基线：`yhz/test/v0.5.23-k31` 分支（行号在该分支上核对过）
> 关联文档：`PCP_DCP_方案A_计划二.md` §3.3、§4.3；`方案A计划问题汇总.md` §3、§4
> 文中代码片段只是改法示意，**没有运行验证**。标"（推断）"的条目确定性较低。

## 0. 结论

### 0.1 为什么要改切分规则

现有的 interleave 把整批 token 拼起来，按拼接后的下标取模：拼接下标为 `i` 的 token 交给 CP rank `i % N`。DCP 的 owner 规则按**序列位置**取模，位置为 `p` 的 token 的 KV 存在 `p % N` 号卡上。只要请求有前缀（P 个已缓存 token），或者一批里有多个请求，同一个 token 的计算卡和 KV owner 就会对不上：

```
单个请求，前缀 P=3，本轮 E=8 个新 token，N=4
拼接下标 i :   0  1  2  3  4  5  6  7
序列位置 p :   3  4  5  6  7  8  9 10
计算卡 i%4 :   0  1  2  3  0  1  2  3      <- 现有 interleave
owner  p%4 :   3  0  1  2  3  0  1  2      <- DCP owner 规则
```

计划二 §3.3 打算新增"按位置取模"的切分：rank r 负责 `(prefix_len + j) % N == r` 的 token，让计算卡就是 owner。

### 0.2 但 P1 不需要做这件事（修正计划二）

读完代码后，我的结论是 **P1 可以保留现有切分，不改切分规则**：

1. **KV 写入的正确性不依赖切分。** P1 写 KV 时沿用"先 gather 新 token 的 KV，再写全部行"的做法（`interleave.py:401-415` 的 `materialize_full_mla_kv`，`dsa_indexer.py:1094-1095`）。写入时 `UnitIds.translate_write`（`mem_cache/kv/alloc/id_space.py:163-185`）会把不归本卡的 slot 映射到 sink 0，indexer K 的融合 store 也按 owner 过滤。所以无论计算卡是谁，每个位置的 KV 最终只落在 owner 卡上。
2. **attention 读 KV 不依赖切分。** DCP prefill 用 `dcp_gather_request_rows` 从各个 owner 卡收齐**整条序列**的 KV（`dsa_backend.py:1392`、`dsa_indexer.py:1925`）。query 侧只用 CP 元数据描述本卡的行。两边各自一致即可。
3. **只有 P3 的"只写本地"需要切分规则一致。** 只有去掉新 token 的 gather、让每张卡只写自己算出的 KV 时，才要求计算卡就是 owner。

所以建议：**按位置取模的切分放到 P3，和"只写本地"一起做。** 下面第 1–7 节写的是 P3 要改的具体内容；第 8 节列出 P1 仍然要做的事情。

还有一处要注意：参数校验阶段已经按"每个请求单独补齐"放大了 MegaMoE 的 token 预算（`mega_moe_hook.py`）。P1 如果保留现有切分，这个预算会偏保守，但不会出错。

### 0.3 读代码发现：要改的地方比计划二表格写的少

`InterleaveContextParallelMetadata` 已经有 `gather_index` 和 `local_index` 字段（`interleave.py:79-81`），而且以下函数在 `gather_index` 不为 None 时已经会走任意排列的分支：

| 函数 | 行号 | `gather_index` 分支 |
|---|---|---|
| `cp_interleave_to_sequence_order`（KDA 进入时重排） | 94-127 | 109-110 行：`index_select(0, gather_index)` |
| `cp_sequence_to_interleave_order`（KDA 出来时重排） | 135-160 | 152-153 行：`index_copy_(0, gather_index, ...)` |
| `_gather_interleaved_tensor`（`gather_hidden_states` / `gather_kv_cache` 的实现） | 291-355 | 332-333 行：`index_select(0, gather_index)` |
| `moe_num_token_non_padded` | 167-181 | 只读 `per_rank_logical_token[cp_rank]`，要求有效行在分片前部。新布局满足这一点 |

层边界的 gather 是按 rank 顺序拼接等长分片（`layer_boundary/prepare.py:554-571` 的 `_then_attn_cp_gather` 调用 `attn_cp_interleave_gather`）；取回时用 `tensor_split` 按 rank 切分（`layer_boundary/ops.py:428-437` 的 `attn_cp_take_back_output`，以及 157 行的 `attn_cp_interleave_reduce_scatter`）。这些操作**和分片内部的 token 顺序无关**。所以只要 `build_metadata` 填好 `gather_index`，并且每个 rank 的分片补齐到相同长度，上面这些函数和层边界都不用改。

DeepSeek-V4 已经有按 owner 构造 `gather_index` 的现成实现：`layers/attention/deepseek_v4_backend.py:2112-2170` 的 `_late_layer_tail_cp_layout`。它的 owner 是 `token_indices % cp_size`，方案 A 只需把 owner 换成 `positions % cp_size`。

---

## 1. `layers/cp/interleave.py`

### 1.1 `build_metadata`（190-215 行）：生成按位置的布局（必须改，P3）

**为什么要改：** 现在只按 token 总数均分（205-208 行：`divmod(total_seq_lens, cp_size)`），不生成 `gather_index` 和 `local_index`。按位置取模后，每个 rank 的 token 数取决于各请求的前缀长度，必须在这里算出来。

**怎么改：** 参数里已经有 `seqs_len`（每个请求的 `seq_len`）和 `extend_seqs_len`，前缀长度就是 `seqs_len[i] - extend_seqs_len[i]`，**不需要改签名**。调用方 `layers/cp/utils.py:284-289` 已经传了这两个参数。DCP 生效时走新分支：

```python
def build_metadata(self, num_tokens, seqs_len, extend_seqs_len=None):
    ...  # 196-204 行原样保留，得到 extend_seqs_len 和 total_seq_lens
    if get_parallel().dcp_enabled and seqs_len is not None:
        return self._build_dcp_owner_metadata(seqs_len, extend_seqs_len, total_seq_lens)
    ...  # 205-215 行：原有均分逻辑

def _build_dcp_owner_metadata(self, seqs_len, extend_lens, total):
    # Owner of a token is its sequence position mod N: the DCP KV owner.
    n, rank = self.cp_size, self.cp_rank
    owner = torch.cat([
        (torch.arange(e) + (s - e)) % n for s, e in zip(seqs_len, extend_lens)
    ])  # CPU，长度 total
    counts = torch.bincount(owner, minlength=n).tolist()
    # The stride must equal the physical shard length pad_logical_token_to_physical
    # computes next, or gather_index points into the wrong slots.
    align = get_cp_padding_align_size()
    stride = (max(counts) + align - 1) // align * align
    order = torch.argsort(owner, stable=True)
    starts = torch.tensor([sum(counts[:r]) for r in range(n)])
    slot = torch.empty_like(owner)
    slot[order] = torch.arange(total) - starts[owner[order]]
    device = get_device().device
    return InterleaveContextParallelMetadata(
        per_rank_actual_token=counts,           # logical; padded right after
        max_rank_len=[max(counts)] * n,
        total_seq_lens=total,
        bs=len(extend_lens),
        gather_index=(owner * stride + slot).to(device),          # sequence row -> rank-major slot
        local_index=(owner == rank).nonzero().squeeze(1).to(device),  # this rank's sequence rows
    )
```

需要注意的地方：

- **`gather_index` 的步长必须等于物理分片长度。** 调用方 `prepare_cp_forward` 在 `build_metadata` 之后，会马上调用 `pad_logical_token_to_physical`（`utils.py:290`，实现见 `layers/cp/padding.py:34-43`）。这个函数把 `per_rank_actual_token` 当作逻辑长度，按 `get_cp_padding_align_size()` 对齐成物理长度。所以 `build_metadata` 里 `per_rank_actual_token` 必须填**逻辑**长度，`gather_index` 的步长则要按**对齐后的物理**长度算，两边都对齐到同一个值，否则 `gather_index` 会指到错误的槽位。
- **padding token 的归属。** 200-202 行把 padding token 追加到最后一个请求。它们的位置接在这个请求后面，owner 按同样的规则计算。
- **`local_index` 只放真实行，不放 padding 行。** 这和 V4 的 `local_metadata_rows` 不同（V4 把 padding 行也放进去了）。本卡分片的 padding 由 `pad_local_rows` 补上（`padding.py:46-60`）。
- **与 V4 字段语义的关系。** V4 的 `local_index` 被 `deepseek_v4_backend.py:1744` 当作元数据行使用。K3.1 不经过这条路径，但两边对同名字段的语义要统一，或者给方案 A 换一个字段名（推断）。

### 1.2 `shard_hidden_states`、`shard_position_ids`（217-225 行）：按 `local_index` 取行（必须改，P3）

**为什么要改：** 这两个函数现在都调用 `_interleave_shard`，按拼接下标每隔 N 行取一行。

**怎么改：**
```python
def shard_hidden_states(self, x, forward_batch):
    metadata = forward_batch.attn_cp_metadata
    x = x[: metadata.total_seq_lens]
    local_x = (
        x.index_select(0, metadata.local_index)
        if metadata.gather_index is not None
        else self._interleave_shard(x)
    )
    return pad_local_rows(local_x, metadata, dim=0)
```
`shard_position_ids` 改法相同。这样 `cp_shard_hidden_states`、`cp_shard_position_ids`、`cp_shard_model_inputs`（`utils.py:322-349、432-`），以及 a2a 路径下的 `cp_interleave_input_ids`（`utils.py:389-390`），都会跟着生效。

### 1.3 `_interleave_shard`、`shard_local_tokens`（227-242、252-253 行）：增加 metadata 参数（必须改，P3）

**为什么要改：** `shard_local_tokens(input_)` 只接收一个张量，拿不到 `local_index`。它在 `dsa_backend.py:1624` 被用来切分 `seqlens_expanded`，也就是每个 query 能看到的 KV 长度，用于 indexer 的 causal 判断。这个张量必须和本卡的 query 行一一对应。

**怎么改：** 签名改成 `shard_local_tokens(self, input_, forward_batch)`。`gather_index` 存在时返回 `input_.index_select(0, metadata.local_index)`，否则仍走 `_interleave_shard`。`_interleave_shard` 本身保留，作为非 DCP 的路径。

### 1.4 `local_q_indices`（244-250 行）：返回 `local_index`（建议改，P3）

**为什么要改：** 现在返回 `arange(cp_rank, num_tokens, cp_size)`。K3.1 的 CUDA 路径不调用它，调用方只有 NPU 的 `ascend_backend.py:1429` 和 `ascend_dsv4_backend.py:1161`。为了让同一个策略对外的语义保持一致，建议 `gather_index` 存在时返回 `metadata.local_index`。

### 1.5 `shard_per_request`（255-279 行）：按位置计算每个请求的 query 数（必须改，P3）

**为什么要改：** 现在的计算在请求之间传递余数（266-272 行的 `extra_seq`），本质上还是按拼接下标切分。它返回的 `q_lens`、`bs_idx` 会被 `dsa_backend.py:1626` 用来裁剪 `indexer_seq_lens`、`cache_seqlens`、`page_table`，并生成 `cu_seqlens_q`（1627-1645 行）。

**怎么改：** 增加前缀长度参数，按位置计算请求 i 落在 rank r 上的 token 数：
```
count_i = (P_i + E_i - 1 - r) // N - (P_i - 1 - r) // N        # Python 的向下取整
```
```python
def shard_per_request(self, extend_seqs_cpu, extend_seqs, prefix_lens_cpu, prefix_lens):
    n, r = self.cp_size, self.cp_rank
    q_lens_cpu = [
        (p + e - 1 - r) // n - (p - 1 - r) // n
        for p, e in zip(prefix_lens_cpu, extend_seqs_cpu)
    ]
    ...  # 273-279 行：过滤 0，调用 kernel（kernel 也要传 prefix_lens）
```
只有 DCP 生效时走这个分支。非 DCP 时保持原来的余数传递逻辑，否则会改变现有 PCP 的行为。

### 1.6 不需要改的函数

| 函数 | 行号 | 原因 |
|---|---|---|
| `cp_interleave_to_sequence_order` / `cp_sequence_to_interleave_order` | 94-160 | 已经有 `gather_index` 分支 |
| `moe_num_token_non_padded` | 167-181 | 只读逻辑行数；按位置布局下，有效行仍在分片前部 |
| `can_apply` | 183-188 | 条件只看总长度。按位置切分后某个 rank 可能拿到 0 行，`_gather_interleaved_tensor` 315-316 行和 `pad_local_rows` 能处理这种情况（推断） |
| `gather_hidden_states` / `gather_kv_cache` / `_gather_interleaved_tensor` | 281-355 | 332-333 行已经按 `gather_index` 取行 |
| `materialize_full_mla_kv` / `materialize_full_indexer_k_cache` / `all_gather_dsa_trtllm_fp8_kv` | 360-415 | 都经过 `gather_kv_cache`。P3 改成"只写本地"时另行处理，见 §3.2 |
| `interleave_rows_per_request` | 418-427 | 调用方是 `deepseek_v4_backend.py`（3102、3647 行）和 `interleave_kpool.py`。后者只在 DSA `index_kpool > 1` 时使用，K3.1 的 kpool 是 1。方案 A 第一阶段不经过这里，建议在 DCP 生效时加断言，防止被误用 |
| `DSAInterleaveCPIndexerOverlap` / `run_indexer_with_kv_gather` | 430-604 | 都通过 `gather_kv_cache` 处理 KV |

---

## 2. `kernels/ops/attention/dsa/cp_split.py`（`dsa_cp_interleave_q_seqs`，必须改，P3）

**位置：** `python/sglang/kernels/ops/attention/dsa/cp_split.py`，kernel 在 11-32 行，封装函数在 35-47 行。

**为什么要改：** 它是 `shard_per_request` 在 GPU 上的实现，同样在请求之间传递 `extra_seq`（21、26、32 行），也是按拼接下标计算。它存在的原因是保证这次切分可以被 CUDA graph 捕获（`interleave.py:260` 的 docstring："keep the split graph-safe"）。

**怎么改：** 新增一个 kernel，例如 `_dsa_cp_dcp_q_seqs_kernel`，多接收一个 `prefix_seqs_ptr`，按位置公式计算：
```python
p = tl.load(prefix_seqs_ptr + bs).to(tl.int32)
e = tl.load(in_seqs_ptr + bs).to(tl.int32)
# Triton integer division truncates toward zero; offset by cp_size so both
# numerators stay non-negative (p - 1 - cp_rank >= -cp_size).
cur_seq = (p + e - 1 - cp_rank + cp_size) // cp_size - (p - 1 - cp_rank + cp_size) // cp_size
```
**注意除法：** Triton 和 C 一样，整数除法向零取整，不是向下取整。`p - 1 - r` 可能是负数，例如 `p=0, r=0`。所以分子要先加上 `cp_size`，保证非负，才能和 §1.5 的 Python 公式结果一致。原有 kernel 保留，供非 DCP 路径使用。

**调用方 `layers/attention/dsa_backend.py:1622-1636` 一起改：** 把 `forward_batch.extend_prefix_lens_cpu` 和 `forward_batch.extend_prefix_lens` 传进 `shard_per_request`。`shard_local_tokens(seqlens_expanded)` 改成 `shard_local_tokens(seqlens_expanded, forward_batch)`。计划二 §4.3 的表格没有列出这个调用点。

---

## 3. `layers/cp/utils.py`

### 3.1 `cp_interleave_input_ids`（387-395 行）：非 a2a 路径要按 `gather_index` 重排（必须改，P3）

**为什么要改：** K3.1 的 MoE 默认不走 a2a。这时会进入 392-395 行：
```python
padded_input_ids[: input_ids.shape[0]] = input_ids
return padded_input_ids.view(-1, get_parallel().attn_cp_size).T.flatten()
```
这是按拼接下标生成的"rank 优先"顺序，结果写到 `forward_batch.input_ids_global`（`utils.py:455-457`）。按位置取模后，这个顺序和层边界 gather 出来的行顺序对不上，后续用 `input_ids_global` 的地方（例如按行查表的 MoE 路径，推断）会对错行。

**怎么改：** `gather_index` 存在时，复用 `cp_sequence_to_interleave_order`。它在 152-153 行已经实现了"按 `gather_index` 填进 rank 优先的 padding 布局"：
```python
metadata = forward_batch.attn_cp_metadata
if metadata.gather_index is not None:
    from sglang.srt.layers.cp.interleave import cp_sequence_to_interleave_order
    return cp_sequence_to_interleave_order(
        input_ids[: metadata.total_seq_lens], forward_batch,
        sum(metadata.per_rank_actual_token),
    )
```

### 3.2 `prepare_cp_forward`（270-304 行）：P1 不改；P3 改写入位置

- 284-290 行调用 `build_metadata` 并做 padding。§1.1 的改动不需要改这里的调用。
- 303-304 行 `out_cache_loc = out_cache_loc[:num_tokens]` 只截掉尾部 padding，**不按卡切分**。
  - **P1：** 不改。写入时 gather 全部新 token 的 KV，再配合完整的 `out_cache_loc`，由 owner 过滤保证正确。
  - **P3（只写本地）：** 写入时要用本卡行对应的子集 `out_cache_loc.index_select(0, local_index)`。同时去掉 `materialize_full_mla_kv` 和 `dsa_indexer.py:1094-1095` 里新 token 的 gather。这时计算卡就是 owner，`translate_write` 不会再丢弃任何写入。注意不要直接覆盖 `forward_batch.out_cache_loc`，因为 logits、spec 等其他路径还在读完整的 `out_cache_loc`。应该新增一个只给 KV 写入用的字段（推断）。

---

## 4. `layers/cp/base.py`：不用 `register_cp_kv_layout`（修正计划二）

**计划二的设想：** 用 `register_cp_kv_layout`（255-263 行）注册方案 A 的布局，复用 `init_cp_strategy`（311-320 行）"布局要求特定 `--cp-strategy`"的检查，借此拒绝 zigzag。

**读代码后的结论是不采用，原因有三：**

1. **需要用户传 `--cp-kv-layout`。** 只有 `cfg.cp_kv_layout is not None` 时才走注册表（311 行）。这个值来自 CLI 的 `--cp-kv-layout`，它的 choices 是 `arg_groups/choices.py:261` 的 `CP_KV_LAYOUT_CHOICES`，会在 `register_cp_kv_layout` 时动态追加。要让 argparse 接受这个值，注册必须在构建参数解析器之前完成。现在仓库里没有任何地方调用 `register_cp_kv_layout`，导入顺序也没有保证（推断）。
2. **会和 `--enable-kv-cache-sharding` 冲突。** `arg_groups/kv_shard_hook.py:176-180` 规定：设置了 `cp_kv_layout` 就不能开 KV sharding。这个约束本身没问题，但 `cp_kv_layout` 原本的语义是"prefill CP 自己管 KV 存储"（配合 `cp_kv_sharded`，`base.py:101、368-371`）。方案 A 的 KV 存储由 DCP 分配器负责，语义不同，混用容易让读代码的人误解。
3. **zigzag 已经被拒绝了。** 参数校验阶段已经拒绝方案 A 加 zigzag：`parallel_hook.py` 的 `_validate_prefill_cp_with_dcp`，以及 `cp_tp_group_sharing_hook.py` 对 K3.1 要求 interleave。不需要再靠注册表检查一遍。

**推荐做法：** 不新增策略类，直接在 `InterleaveCPStrategy.build_metadata` 里按 `get_parallel().dcp_enabled` 走 §1.1 的分支，其余函数用 `metadata.gather_index is not None` 判断。这样 `base.py` 不用改。

如果更想要显式的子类（例如 `InterleaveDcpOwnerCPStrategy`），可以在 `init_cp_strategy` 里新增一个 `dcp_size` 参数，在 324-328 行选择子类。这样两个调用方都要改：`base.py:359-364` 的懒加载要传 `parallel.dcp_size`，`arg_groups/parallel_hook.py:215` 要传 `cfg.dcp_size`。

---

## 5. `layers/layer_boundary/`：不改，只需验证

**计划二的设想：** 层边界的 gather 和 padding 要适配新的行布局。

**读代码后的结论是不需要改：**
- gather 按 rank 顺序拼接**等长**分片（`prepare.py:554-571` → `interleave.py:57-71`）。
- 取回用 `tensor_split(attn_cp_size)[attn_cp_rank]`（`ops.py:428-437`）；reduce-scatter 用 `attn_cp_interleave_reduce_scatter`（`ops.py:157-`）。
- 每个 rank 的有效行数由 `_cp_shard_token_rows`（`layout.py:168-172`）读取 `per_rank_logical_token`。

这些都只依赖"每个 rank 一段、长度相同、有效行在前"，**和分片内部的 token 顺序无关**。需要知道序列顺序的地方（KDA、KV gather、最后的 hidden gather）都在 interleave.py 里，而且已经有 `gather_index` 分支。

同理，`models/kimi_k3.py` 的 `_forward_cp_shard_rows`（1977-2005 行）只依赖"按 rank 顺序 gather、分片等长"，不用改。

**需要的验证：** 用随机的前缀长度和多请求的批次，对比方案 A 和 TP8 每一层 KDA 输入的序列顺序（`cp_interleave_to_sequence_order` 的输出）是否完全一致。

---

## 6. `model_executor/runner/eager_runner.py`：不改，只核对

### 6.1 `__init__` 的 `max_num_token`（120-124 行）

```python
if require_mlp_sync():
    max_num_token = ceil_align(max_num_token, self.attn_tp_size)
    max_num_token = ceil_align(max_num_token, get_cp_padding_align_size())
```
这里按"整批输入的行数"给 eager 注册表分配 buffer（`build_eager_registry`）。切分前的输入是完整的 token 序列，长度不受切分规则影响。切分后的分片和 gather 结果都在运行时用 `new_empty` / `new_zeros` 动态分配（`interleave.py:67-69、323、329`），不占用这个上限。所以**不用改**（推断）。

按位置切分时，所有 rank 的物理分片总长 `N × stride` 可能大于 token 总数，最多多出大约 `N × (请求数 + align)`。如果以后有静态 buffer 按 `sum(per_rank_actual_token)` 分配（例如 prefill graph，见 §7），需要另外核对。

### 6.2 MIXED 模式（211 行）

`execute` 在存在 CP 策略时保留 MIXED 模式，不把它改成 EXTEND。计划二担心 MIXED 里 1 个 token 的 extend 段的处理。但 K3.1 走 CP-TP group sharing 时，`cp_tp_group_sharing_hook.py` 的 unsupported 列表**已经拒绝了 `--enable-mixed-chunk`**，所以方案 A 不会产生 MIXED 批次，**不用改**。

如果以后放开 mixed chunk：decode 尾巴是前缀为 `seq_len - 1` 的 1 token extend。按位置取模时，它的 owner 是 `(seq_len - 1) % N`，正好和 DCP decode 的 owner 规则一致，§1.1 的公式可以直接处理。

---

## 7. `layers/cp/bcg/interleave_bcg.py`：第一期不改（P3 再看）

参数校验阶段已经在方案 A 下拒绝 prefill CUDA graph（`cuda_graph_hook.py`），这个文件第一期不会被执行到。

P3 如果开启 graph，有两处要处理：
- 129 行设置 `metadata.graph_num_tokens`，138 行调用 `cp_interleave_input_ids`。graph 的静态形状要能容纳按位置切分后可能多出来的 padding（§6.1）。
- `gather_index` 和 `local_index` 是每次前向都不同的张量，graph 回放前要写进静态 buffer，不能在捕获时固定下来。

---

## 8. P1 必须做的事（保留现有切分时）

| 事项 | 位置 | 说明 |
|---|---|---|
| query 侧的元数据按本卡行构建，KV 侧按完整请求列表构建 | `dsa_backend.py:1392`（`dcp_gather_kv_cache`）、`dsa_indexer.py:1925`（`_dcp_gather_index_k`）、1595-1660 行的元数据 | 这不属于切分规则本身。CP 下有的请求在某个 rank 上没有 query 行（`shard_per_request` 会把它过滤掉），但 KV gather 需要所有请求的完整长度，否则各卡输入形状不一致（`方案A计划问题汇总.md` §5.1） |
| 关闭重复的 CP KV gather（可选） | `forward_mla.py:472-480、866`，`dsa_backend.py:4508-4556` | P1 的写入依赖 gather + owner 过滤，所以**保留**新 token 的 gather。但 attention 读 KV 时要统一走 DCP 的整序列 gather，避免两份 KV 混用 |
| 断言 | `interleave.py` 的 `build_metadata` | 断言 `sum(per_rank_logical_token) == total_seq_lens`，以及每个 rank 的行数不超过 MegaMoE 预算 |

---

## 9. 修改清单汇总

| # | 文件 | 行号 | 改动 | 阶段 |
|---|---|---|---|---|
| 1 | `layers/cp/interleave.py` | 190-215 | `build_metadata` 增加按位置分支，生成 `gather_index`/`local_index`；步长和物理分片长度对齐 | P3 |
| 2 | `layers/cp/interleave.py` | 217-225 | `shard_hidden_states`/`shard_position_ids` 按 `local_index` 取行 | P3 |
| 3 | `layers/cp/interleave.py` | 227-253 | `shard_local_tokens` 增加 `forward_batch` 参数 | P3 |
| 4 | `layers/cp/interleave.py` | 244-250 | `local_q_indices` 返回 `local_index` | P3（建议） |
| 5 | `layers/cp/interleave.py` | 255-279 | `shard_per_request` 增加前缀长度，按位置计算 | P3 |
| 6 | `kernels/ops/attention/dsa/cp_split.py` | 11-47 | 新增按位置计算的 kernel，注意向零取整的问题 | P3 |
| 7 | `layers/attention/dsa_backend.py` | 1622-1636 | 调用方传入前缀长度和 `forward_batch` | P3 |
| 8 | `layers/cp/utils.py` | 387-395 | `cp_interleave_input_ids` 按 `gather_index` 重排 | P3 |
| 9 | `layers/cp/utils.py` | 303-304 | P3 新增"本卡写入位置"，不覆盖 `out_cache_loc` | P3 |
| 10 | `layers/cp/interleave.py` | 418-427 | `interleave_rows_per_request` 在 DCP 生效时断言 | P3 |
| — | `layers/cp/base.py` | 255-332 | **不改**，不用 `register_cp_kv_layout`（§4） | — |
| — | `layers/layer_boundary/`、`kimi_k3.py:1977` | — | **不改**，只做验证（§5） | — |
| — | `model_executor/runner/eager_runner.py` | 120-124、211 | **不改**，只核对（§6） | — |
| — | `layers/cp/bcg/interleave_bcg.py` | 129、138 | 第一期不改（§7） | P3 |

## 10. 建议的测试（CPU，P3 实现时添加）

1. **owner 一致性：** 用随机的前缀长度、多个请求和 padding，验证 `build_metadata` 生成的每个 rank 的 `local_index` 对应的位置都满足 `pos % N == rank`。同时验证 `gather_index` 是一个双射：从序列行到 rank 优先的槽位，并且所有槽位都落在 `N × stride` 范围内。
2. **往返一致：** `cp_sequence_to_interleave_order` 后再做 `cp_interleave_to_sequence_order`，结果等于原序列。用 `shard_hidden_states` 的结果模拟每个 rank 的分片，按 rank 拼接后，用 `gather_index` 取出来，要等于原序列。
3. **每请求计数：** §1.5 的 Python 公式和 §2 的 kernel 公式，在 `p` 为 0、`p < r` 这类边界值上结果一致。CPU 上可以直接比较 Python 公式和逐 token 计数的结果。
4. **非 DCP 不变：** 关闭 DCP 时，`build_metadata`、`shard_per_request` 和 `cp_interleave_input_ids` 的输出和修改前完全相同。
