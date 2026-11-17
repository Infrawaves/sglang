# `python/sglang/srt/mem_cache/kv/alloc/id_space.py` 代码理解

- 文件路径：`python/sglang/srt/mem_cache/kv/alloc/id_space.py`
- 行数：362
- 所属模块：KV 池的虚拟/物理 id 映射。定义一个子池怎样给自己的单元（页或 slot）命名，以及一次迭代的读写 id 怎样经过这张映射表；包括 DCP 下写入位置的 owner 过滤。

## 1. 整体功能

KV 池的每个子池（全注意力子池、滑动窗口子池、recurrent state 子池）都有两套 id：

- **虚拟 id**：请求表 `req_to_token` 和 radix tree 持有的 id，单元在 band 里移动时保持不变；
- **物理 id**：单元当前在 band 里的位置，kernel 用它索引每层的 KV 视图。

`UnitIds` 保存两张表 `virtual_to_physical` / `physical_to_virtual`（单元 0 是 padding sink，每张表最后一行是 -1 sentinel）。token id = `unit * unit_tokens + offset`，翻译时只改 unit、保留 offset。

`IdSpace` 是给消费方用的只读描述：`write` 把本次迭代的虚拟 token id 映射到子池 id；`read_v2p` 是读取用的页表（None 表示读取保持虚拟，由消费方自己处理，DCP 就是这种情况）；`table` 总是子池的页表。

**DCP 相关的关键约定**（第 107-110、155-157、175-177 行注释）：
- DCP 下分配器的页是"放大页"（`page_size × dcp_size` 个虚拟 token），`unit_tokens` 仍是一个单元占用的**物理**行数。
- 读取用的 id 已经是 DCP 折叠后的（`widened // dcp_size`），所以 `translate` 用 `dcp_size=1`。
- 写入用的 `out_cache_loc` 没有折叠，仍带 owner 规则：`loc % dcp_size` 是 owner rank，`loc // dcp_size` 是行。`translate_write` 把不属于本 rank 的 id 映射到 0（所有写 kernel 都会跳过的 sink）。实际计算在 Triton kernel `translate_token_ids`（`sglang/kernels/ops/memory/virtual_slot.py`）里。

文件里还有两部分与 DCP 关系较远：`WindowRead` / `window_read` 描述滑动窗口层每个请求读哪一段 KV 以及 verify tree 的 mask；`ReplicatedTokenIds` 给 draft 模型提供"覆盖 owner DCP 页里每个逻辑 token"的坐标。

## 2. 主要依赖与调用方

依赖：`msgspec`、`torch.profiler.record_function`、`sglang.kernels.ops.memory.virtual_slot`（`bind_inplace`、`translate_token_ids`、`expand_page_mapping`）、`runtime_context.get_parallel`（读 `attn_dcp_size`、`attn_dcp_rank`）。

调用方（grep 确认）：
- `UnitIds` 由 `mem_cache/kv/alloc/band.py:356`、`mem_cache/kv/alloc/chain.py:232, 252` 构造，并传入 `shards_under_dcp`。
- `token_space`：`chain.py:652, 659`（`id_spaces`），owner 子池和每个窗口子池都用 `widened_writes=True`。
- `slot_space`：`chain.py:671`（`state_id_spaces`）。
- `translate_write`：`chain.py:681`（`translate_write_loc`）、`mem_cache/kv/memory/snapshot.py:26`。
- `state_id_space`：`mem_cache/kv/alloc/runner_ids.py:291`、`mem_cache/allocator/base.py:105`。
- `ReplicatedTokenIds`：`mem_cache/kv/alloc/runner_ids.py:194, 212`。
- `window_read`：`layers/attention/flashinfer_backend.py:2074`。

## 3. 类与函数说明

### 3.1 `IdSpaceKind`（第 35 行，`str, Enum`）

两个常用子池名：`FULL = "full"`（全注意力层）、`SLIDING_WINDOW = "swa"`（混合池的滑动窗口层）。id space 按子池名索引，所以任何子池名都可以用在需要 kind 的地方。

### 3.2 `IdSpace`（第 47 行，`msgspec.Struct, frozen=True`）

字段：`key`（共享同一子池的 runner 共享 key 和计划里的 id）、`write`、`read_v2p`、`table`、`prepare_read`（读取前的准备回调）。

| 方法 | 行号 | 作用 |
|---|---|---|
| `read_mapping` | 64 | 若有 `prepare_read` 先调用，再返回 `read_v2p`。 |
| `ids` | 69 | 把虚拟 id 换成子池 id：`write` 为 None 时原样返回，否则调用 `write`。 |

### 3.3 `state_id_space`（第 74 行）

从 `spaces` 字典取 recurrent state 子池的 id space；没有时返回一个 `key=(sub_pool, None)`、不做映射的 `IdSpace`。

### 3.4 `UnitIds`（第 82 行）

| 方法 | 行号 | 作用 |
|---|---|---|
| `__init__` | 97 | 断言 `num_virtual_ids > 0`；保存 `unit_tokens`（物理行数）、`shards_under_dcp`；分配两张填 -1 的 int64 表，长度分别为 `num_virtual_ids + 1`、`num_physical_units + 1`。 |
| `reset` | 120 | 两张表清成 -1，再把 0 号设为 0（padding sink）、最后一行设为 -1。 |
| `bind` | 129 | 用 `bind_inplace` kernel 把单元粒度的虚拟 id 和物理 id 配对，同时写两张表。 |
| `is_bound` | 139 | 虚拟单元在范围内且已映射（不为 -1）时返回 True。会触发一次 `.item()` 同步。 |
| `translate` | 144 | 读取路径：虚拟 token id → 物理 token id，`physical = v2p[t // unit_tokens] * unit_tokens + t % unit_tokens`，固定 `dcp_size=1`（输入已是 DCP 折叠后的 id）。未映射、越界或负数映射到 0。支持 `out=` 原地写入，CUDA graph 捕获时必须用。输出 int64。 |
| `translate_write` | 166 | 写入路径：放大后的虚拟写位置（`out_cache_loc`）→ 物理 id。`shards_under_dcp` 为 True 时用 `attn_dcp_size` 和 `attn_dcp_rank`，不属于本 rank 的位置映射到 0。 |
| `_translate` | 190 | 统一调用 `translate_token_ids(loc, v2p, page_size=unit_tokens, dcp_size, dcp_rank, out, out_width)`。 |
| `token_space` | 210 | 构造 token 子池的 `IdSpace`：`widened_writes` 决定写入用 `translate_write` 还是 `translate`；`defer_reads` 为 True 时 `read_v2p=None`（DCP 在构建读取时自己选本 rank 的份额）；`table` 总是 `virtual_to_physical`。 |
| `slot_space` | 224 | 构造 state 子池的 `IdSpace`：每单元一个 slot，写入就是 `table[slots].to(int32)`，读取用同一张表。 |

### 3.5 `WindowRead`（第 237 行，`msgspec.Struct, frozen=True, kw_only=True`）与 `window_read`

字段：`start`（每个请求读取的第一个位置）、`lens`（读取的 token 数）、`window`（层窗口大小）、`window_left`（kernel 规划时用的窗口，-1 表示读的是前缀、由调用方自己 mask）。

| 名称 | 行号 | 作用 |
|---|---|---|
| `WindowRead.tree_mask` | 254 | 把 verify tree 的 mask（每个请求 `width` 行、`seq_len + width` 列）重新布局到这次窗口读取上。因为 tree 里每行 query 的位置是它在树里的深度而不是行号，窗口被折进 mask 里，kernel 不再带窗口。每行只能看到自己窗口内的前缀列和树上祖先。图 replay 的填充行用 `clamp` 防止越界。 |
| `window_read` | 303 | 构造 `WindowRead`。`prefix_only=True`：只读缓存前缀的最后 `window` 个 token（`window_left=-1`）；否则读第一个 query 前的 `window` 个 token 到最后一个 token，`lens = min(seq_lens, window + seq_lens - prefix_lens)`，由 kernel 按每个 query 的窗口 mask。 |

### 3.6 `ReplicatedTokenIds`（第 329 行）

draft 模型在 owner DCP 页上需要覆盖每个逻辑 token 的坐标。

| 方法 | 行号 | 作用 |
|---|---|---|
| `__init__` | 332 | 保存 owner 的 `UnitIds` 和放大倍数 `factor`，分配展开后的映射表（长度 `(owner 表长 - 1) * factor + 1`）。 |
| `refresh` | 341 | 用 `expand_page_mapping` 把 owner 的页表按 `factor` 展开到 `expanded`。 |
| `write` | 346 | 用 `page_size = owner.unit_tokens * factor`、`dcp_size=1` 翻译写入 id，即每个 rank 都写（复制）。 |
| `space` | 355 | 返回 `IdSpace`，key 为 `(key, "replicated", factor)`，读取前先 `refresh`。 |

## 4. 与 PCP+DCP 方案 A 的关系

计划 §4.4 认为 `translate_write` 现有的 owner 映射"应该可以直接用（推断），需要加单元测试确认"。读代码后的补充：

- **owner 规则是基于写入位置的，不是基于序列位置。** `translate_write` 用 `loc % attn_dcp_size` 判断 owner。只有当 `out_cache_loc` 在放大页内的偏移和序列位置对齐时（即 `loc % dcp == pos % dcp`），它才等价于"位置 p 存在 `p % N` 号卡"。分配器按放大页（`page_size × dcp`）分配、前缀按放大页对齐时这一点成立（推断，见 `layers/dcp/planner.py:113` 的注释）。方案 A 的 chunked prefill 边界也必须满足这个对齐，否则 CP 按位置分配的 token 和 `translate_write` 认定的 owner 会错开，产生计划 §7 所说的静默数值错误。单元测试应直接覆盖"chunk 起点不是 `dcp` 的倍数"这种情况。
- **rank 来源。** `translate_write` 读的是 `attn_dcp_rank` / `attn_dcp_size`，而 `runtime_context.py:1611-1612` 里 `attn_dcp_rank` 由 `dcp_rank`（`tp_rank % dcp_size`）派生。计划 §4.2 把 `dcp_rank` 改为 `attn_cp_rank` 后，这里会自动跟随。另外，`runtime_context.py:126` 中 `attn_cp_rank = (tp_rank // attn_tp_size) % attn_cp_size`，在方案 A 典型配置（`attn_tp_size = 1`、`attn_cp_size = dcp_size = tp_size`）下数值上等于 `tp_rank % dcp_size`，所以这一改动在典型配置下不改变 rank 值，主要是语义上的统一（推断）。（计划未提及）
- **"只写本地"不是正确性的必要条件。** 因为非 owner 的写入会被映射到 sink 0，即使沿用现有 PCP"gather 后每卡写全部新 token"的做法，存储下来的 KV 也只落在 owner 卡上（推断）。计划 §4.4 的"只写本地 token"更多是省写入带宽。这也意味着 P1 可以先保留"gather 后写全部行"，用 `translate_write` 保证 owner 存储，再逐步优化。（计划未提及）
- `token_space` 的 `defer_reads`（210）：DCP 下读取保持虚拟，由 DCP 自己选本 rank 份额（`layers/dcp/planner.py` 的 `translate_dcp_read_ids`）。方案 A 的 prefill 要对整个序列从各 owner gather KV，走的也是这条"读取保持虚拟、由 DCP 选份额"的路径，不需要改这里。
- `ReplicatedTokenIds.write` 用 `dcp_size=1` 让每个 rank 都写，用于 draft。计划把投机解码放在 P4，届时要确认 draft 在 CP+DCP 下用的是哪种 id space。
