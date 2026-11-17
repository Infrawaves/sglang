# `python/sglang/srt/layers/cp/bcg/interleave_bcg.py` 代码理解

- 文件路径：`python/sglang/srt/layers/cp/bcg/interleave_bcg.py`
- 行数：149
- 所属模块：interleave PCP 在 prefill breakable CUDA graph（BCG）下的适配层，服务 DSA/KDA 混合模型（GLM-5.x、Kimi-K3.1）的文本 prefill。

## 1. 整体功能

prefill BCG 把 prefill 前向录成固定行容量的 CUDA graph，但每次 replay 的真实 token 数、前缀状态和写入位置都会变。这个文件解决两件事：

1. **静态 buffer 和元数据准备**：`InterleaveCPBCGInput` 继承通用的 `PrefillCPBCGInput`（`layers/cp/bcg/bcg.py`），额外分配按 `attn_cp_size²` 对齐的 `input_ids_global` buffer，以及一个固定地址的 `moe_local_token_count` 标量 buffer。`prepare` 在每次前向时把真实的本 rank 行数写进这个 buffer，并恢复全局的 `positions` 和 `out_cache_loc`。
2. **indexer + KV gather 的 eager break**：`bcg_indexer_with_kv` 是用 `eager_on_graph(True)` 包装的 `_indexer_with_kv`，在图里作为一个 eager 断点执行。它每次 replay 时从 forward context 重新读取 batch，按真实行数调用 `run_indexer_with_kv_gather`，再把结果拷进图的静态输出 buffer，尾部补 0（KV）或 -1（top-k）。

模块 docstring 的要点：KPool 使用的完整序列位置（全局）和 DSA RoPE 使用的本地位置要分开保存。

## 2. 主要依赖与调用方

依赖：
- `layers/cp/bcg/bcg.py`：`PrefillCPBCGInput`。
- `layers/cp/interleave.py`：`run_indexer_with_kv_gather`。
- `layers/cp/utils.py`：`cp_interleave_input_ids`。
- `model_executor/runner_backend_utils/breakable_cuda_graph`：`eager_on_graph`；`tc_piecewise_cuda_graph`：`get_tc_piecewise_forward_context`。
- `arg_groups/model_override_base.py`：`resolving_view`、`resolved_view`、`model_config_of`。

调用方（grep 确认）：
- `layers/cp/bcg/__init__.py:32-34`：按策略返回 `InterleaveCPBCGInput`。
- `layers/cp/interleave.py:543`：`indexer_with_cp_kv` 在图内调用 `bcg_indexer_with_kv`。
- `layers/attention/dsa/dsa_prefill_cuda_graph.py:205-207`、`layers/attention/dsa/kpool_prefill_cuda_graph.py:28-30`：`cp_interleave_indexer_rows`。

## 3. 类与函数说明

### 3.1 顶层函数与变量

| 名称 | 行号 | 作用 |
|---|---|---|
| `_indexer_with_kv` | 26 | 图内 eager 断点的实际实现。从 forward context 取当前 batch，用 `cp_interleave_indexer_rows` 得到本 rank 的物理行数 `n` 和逻辑行数 `logical`，校验 `0 <= logical <= n <= 各输入行数`（否则 `ValueError`）。调用 `run_indexer_with_kv_gather` 后校验 KV 行数等于 `sum(extend_seq_lens_cpu)` 且不超过输出容量，top-k 形状合法，然后写入 `kv_output` / `output`，尾部分别置 0 和 -1。 |
| `bcg_indexer_with_kv` | 58 | `eager_on_graph(True)(_indexer_with_kv)`：把上面的函数注册为 BCG 的 eager 段。 |
| `cp_interleave_indexer_rows` | 145 | 返回 `(per_rank_actual_token[rank], logical[rank])`，即本 rank 的物理行数和逻辑行数；没有 `per_rank_logical_token` 时逻辑行数退回物理行数。 |

### 3.2 `InterleaveCPBCGInput`（第 61 行，继承 `PrefillCPBCGInput`）

| 方法/属性 | 行号 | 作用 |
|---|---|---|
| `create`（classmethod） | 63 | 在父类结果上加两个静态 buffer：`input_ids_global`（容量为 `max_num_tokens` 向上对齐到 `attn_cp_size²`，因为每个本地分片先按 CP 对齐再做 rank-major 的 MoE all-gather）和 0 维 `moe_local_token_count`（replay 时从同一地址读真实行数）。 |
| `allows_replay` | 77 | batch 非空、不含多模态输入、`prefix_lens` 不为 None 时才允许 replay。 |
| `_SUPPORTED_ARCHITECTURES` | 82 | 支持的架构：`Glm5NextForConditionalGeneration`、`GlmMoeDsaForCausalLM`、`KimiK31ForConditionalGeneration`。 |
| `supports`（staticmethod） | 89 | 判断能否启用：`dp_size == 1`；`pp_size == 1` 或是 K3.1；K3.1 时 `ep_size == 1` 或 MegaMoE EP 覆盖整个 CP 组（`ep == tp == attn_cp`、`moe_a2a_backend == "megamoe"`、`moe_dp_size == 1`）；`attn_cp_size > 1`；架构在白名单内。 |
| `_max_local_tokens` | 112 | 本 rank 的最大本地行数：总 extend 长度对 `attn_cp_size` 向上取整。 |
| `model_positions` | 116 | 返回 `self.positions[: self.live_local_tokens]`，即传给模型（DSA RoPE）的本地位置。 |
| `prepare` | 119 | 先保存全局 `positions` 和 `out_cache_loc`，调父类 `prepare`（会调用 `prepare_cp_forward`）；然后设置 `graph_num_tokens = static_num_tokens`，把本 rank 逻辑行数写进 `moe_local_token_count` 并挂到元数据上；恢复全局 `out_cache_loc` 和 `positions`（KPool 压缩需要全局连续位置）；最后用 `cp_interleave_input_ids` 生成 interleave 顺序的 input ids 拷进 `input_ids_global`。 |

## 4. 与 PCP+DCP 方案 A 的关系

计划 §4.3 要求这个目录"和 interleave.py 保持一致；第一期可以先关闭 prefill graph"，§4.8 也写明 prefill BCG 在 P1、P2 阶段先禁用。

**如果第一期禁用 BCG**：最直接的做法是在 `supports`（89）里对方案 A（`dcp_size > 1` 且开启 prefill CP）返回 False（推断，计划写的是在 `cuda_graph_hook.py` / `prefill_cuda_graph_runner.py` 处理，这里是另一个可选的拦截点）。

**以后启用 BCG 时需要改的地方：**
- `_max_local_tokens`（112）：现在按总长度 `ceil(total / N)` 估计本地最大行数。按位置取模后，每个请求的前缀余数不同，单个 rank 的行数上限变成 `sum(ceil(L_req / N))`，最多比现在多 `bs - 1` 行（推断）。静态容量要按新上限算。
- `create`（63）里 `input_ids_global` 的 `attn_cp_size²` 对齐来自"每个本地分片 CP 对齐 + rank-major all-gather"，新布局下若 padding 规则改变，需要重新确认。
- `prepare`（119）调用的 `cp_interleave_input_ids`（`layers/cp/utils.py:387`）在 `moe_a2a_backend` 为 none 时用 `view(-1, cp).T.flatten()` 生成 rank-major 顺序，同样是拼接下标取模，需要和新布局一致。（计划未提及）
- `_indexer_with_kv`（26）校验 KV 行数等于 `sum(extend_seq_lens_cpu)`。方案 A 下 attention 需要 gather 整个序列（前缀 + 新 token）的 KV，若复用这个函数，行数检查和 `kv_output` 容量都要改成包含前缀（推断）。（计划未提及）
- `prepare` 恢复全局 `out_cache_loc`（134）。方案 A 若改为只写本卡 token，需要确认 BCG 下写 KV 时从全局 `out_cache_loc` 里选出本卡行的方式与 eager 一致。
