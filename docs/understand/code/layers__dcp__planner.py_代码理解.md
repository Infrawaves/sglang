# `python/sglang/srt/layers/dcp/planner.py` 代码理解

- 文件路径：`python/sglang/srt/layers/dcp/planner.py`
- 行数：197
- 所属模块：decode context parallel（DCP）的每次前向元数据构建：extend 阶段的 KV gather buffer 与下标，decode 阶段把 KV 下标切到本 DCP rank。

## 1. 整体功能

模块 docstring 说明这两个函数来自 PR #14194，计划在 P2 阶段变成 `DecodeContextParallelStrategy` 的方法，现在先以函数形式保留行为。

DCP 下每张卡只存 `pos % dcp_size == dcp_rank` 的 KV。两个函数分别服务两个阶段：

1. **`prepare_decode_context_parallel_metadata`（extend/prefill）**：为 DCP extend 构造一个完整的 KV buffer（`dcp_kv_buffer`，`seq_lens_sum` 行）和它的下标。每个请求在 buffer 里的 KV 先放前缀、再放本轮 extend token（`create_dcp_kv_indices` kernel 写的下标：前缀部分是 `prefix_start + offset`，extend 部分是 `extend_prefix_lens_sum + extend_start + offset`）。也就是说 buffer 前半段是所有请求的前缀，后半段是所有请求的 extend token。同时算出本 rank 拥有的那部分前缀在本地 KV 池里的物理下标 `dcp_local_prefix_kv_indices`，供后面从各 owner all-gather 前缀 KV。第 49-53 行注释描述了 buffer 布局（按 rank 分组的前缀 + extend token）。
2. **`plan_dcp_decode_metadata`（decode）**：把通用的 `kv_lens` / `kv_indptr` / `kv_indices` 原地改写成本 rank 的份额：长度变成本地长度，`kv_indices` 前 `total_local_len` 个元素是本 rank 拥有的 id（仍是放大后的虚拟 id），返回 `total_local_len` 给调用方去做 `translate_dcp_read_ids`。

调用时机：
- extend：模型的 `prepare_context_parallel_metadata_for_dcp`（`models/kimi_k3.py:3810-3830`、`models/deepseek_v2.py:3610`、`models/kimi_linear.py:940`）调用前者；eager runner（计划 §4.5 提到 `eager_runner.py:264-300`）和 `prefill_cuda_graph_runner.py:739, 1320` 再把结果挂到 `forward_batch.attn_dcp_metadata`。之后 `forward_mla.py:109` 的 `bcg_dcp_extend_kv_gather` 用这些元数据把本地 extend KV 写进 buffer 尾部并 all-gather 前缀。
- decode：只有 `layers/attention/aiter_backend.py:2774-2808`（ROCm）和 `flashinfer_mla_backend.py`（通过 `layout.py` 的 `update_local_kv_lens_for_dcp`）走这条路径（grep 结果）。K3.1 的 DSA 后端 decode 用自己的 DCP 实现（`dsa_dcp_sharded.py` 等，推断）。

## 2. 主要依赖与调用方

依赖：
- `sglang.kernels.ops.attention.dcp_kernels`：Triton kernel `create_dcp_kv_indices`、`update_kv_lens_and_indices`。
- `layers/dcp/layout.py`：`update_local_kv_lens_for_dcp`（原地把长度换成 `len // N + (rank < len % N)`）。
- `layers/dcp/metadata.py`：`DecodeContextParallelMetadata`。
- `model_executor/forward_context.get_attn_backend`：取 `runner_ids.translate_dcp_read_ids`。
- `runtime_context`：`get_device`、`get_parallel`（读 `dcp_enabled`、`dcp_size`、`dcp_rank`）。

调用方（grep 确认）：`models/kimi_k3.py:3826`、`models/deepseek_v2.py:3610`、`models/kimi_linear.py:940`（前者）；`layers/attention/aiter_backend.py:2788, 2799, 2808`（后者）。

## 3. 类与函数说明

| 名称 | 行号 | 作用 |
|---|---|---|
| `prepare_decode_context_parallel_metadata` | 33 | DCP 未开启时返回 None。否则：(1) 用调用方传入的 `create_chunked_prefix_cache_kv_indices_fn` kernel 从 `req_to_token` 收集所有请求前缀的虚拟 KV id，得到 `dcp_prefix_kv_indices`（长度为前缀总长）；(2) 用 `create_dcp_kv_indices` 生成 `dcp_kv_indptr`（按 `seq_lens` 累加）和 `dcp_kv_indices`（前缀在前、extend 在后）；(3) 取 `dcp_prefix_kv_indices[dcp_rank :: dcp_size]`，经 `translate_dcp_read_ids` 换成本地物理 id，得到 `dcp_local_prefix_kv_indices`；(4) 分配 `seq_lens_sum` 行的 `dcp_kv_buffer`；返回 `DecodeContextParallelMetadata`。第 113 行注释说明：前缀长度按 `dcp_size` 对齐（分配器页被放大），所以可以直接用步长切片，不需要 `nonzero()`。 |
| `plan_dcp_decode_metadata` | 137 | 复制 `kv_lens` 并用 `update_local_kv_lens_for_dcp` 换成本地长度；非 replay 时从张量求最大和总长度，replay（`init_metadata_replay=True`）时从 `fast_decode_kwargs["kv_len_arr_cpu"]` 读（避免 GPU 同步）；用 `update_kv_lens_and_indices` kernel（grid 为 `(bs, ceil(max_local_len / 128))`）挑出 `loc % dcp_size == dcp_rank` 的 id 并压紧；最后原地写回 `kv_indices[:total_local_len]`、`kv_lens`、`kv_indptr`，返回 `total_local_len`。 |

## 4. 与 PCP+DCP 方案 A 的关系

计划 §4.4 指出：前缀长度按 DCP 页对齐的假设（第 113 行）要在 CP 的 chunk 边界上同样满足。补充如下：

- **第 117 行的步长切片依赖每个请求的前缀长度都是 `dcp_size` 的倍数。** `dcp_prefix_kv_indices` 是所有请求前缀拼接的结果，`[dcp_rank :: dcp_size]` 是在拼接后的数组上取步长。只有每个请求的前缀长度都是 `dcp_size` 的倍数，拼接下标的余数才等于请求内位置的余数。这和 interleave 现有的"拼接下标取模"问题是同一类问题。方案 A 下 chunked prefill 的每个 chunk 结束点就是下一个 chunk 的前缀长度，所以 chunk 大小必须是 `dcp_size`（实际是放大页 `page_size × dcp`）的倍数，最后一个 chunk 除外。radix cache 命中的前缀也要按放大页对齐（计划 §4.9 已提到）。
- **extend 部分的 buffer 现在由本地 KV 直接填充。** `forward_mla.py:122-125`：没有前缀时，`bcg_dcp_extend_kv_gather` 直接把本地 `k_nope`/`k_pe` 写进整个 `dcp_kv_buffer`；有前缀时 `all_gather_kv_cache_for_mla_extend` 也把本地 extend KV 写进 buffer 尾部。这依赖"DCP extend 下每张卡都算全部 extend token"。方案 A 下每张卡只算 1/N 的 extend token，buffer 的 extend 部分必须改由 CP gather（或从各 owner gather）填充，否则缺行（推断）。计划 §3.4 第 3 步说"对整个序列从各 owner all-gather"，但没有点明这里的 buffer 布局和填充逻辑要改。（计划未提及）
- **buffer 布局与 query 行的对应。** `dcp_kv_buffer` 的布局是"所有请求前缀 + 所有请求 extend token"，而 CP 下本卡的 query 是按位置取模挑出的 1/N 行。DSA indexer 和稀疏 MLA 按 query 的全局位置做 causal 时，需要从 query 的（请求、位置）映射到这个 buffer 里的行，计划 §4.5 的"合并元数据"要包含这个映射。
- **rank 来源。** 这里用的是 `parallel.dcp_rank`，计划 §4.2 把它改成 `attn_cp_rank` 后自动生效。方案 A 典型配置下两者数值本来就相等（见 `id_space.py` 文档的说明，推断）。
- **`plan_dcp_decode_metadata` 与方案 A 无直接关系。** decode 时 CP 不生效，DCP decode 走原路径；而且这个函数目前只被 ROCm 的 aiter 后端调用，K3.1 的 DSA decode 不经过它（推断）。
