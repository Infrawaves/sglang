# PCP + DCP 融合（方案 A）：prefill CUDA graph 兼容

> 代码基线：`yhz/test/v0.5.23-k31` 分支（行号在该分支上核对过）
> 范围：K3.1 方案 A（`--tp-size N --enable-prefill-cp --attn-cp-size N --cp-strategy interleave --dcp-size N`）的 prefill breakable CUDA graph（BCG）
> 结论均来自读代码；本文的修改只做了 CPU 级别验证，**还没有在 GPU 上运行过**。

## 0. 背景：BCG 的约束

prefill 使用 breakable CUDA graph：一次 forward 被切成若干**捕获段**，中间穿插 **eager break**。

- 捕获段：投影 GEMM、MoE、层间 all-gather / reduce-scatter、KDA 的投影和重排等。replay 时 kernel 读取的是**捕获时的显存地址**，形状也固定为**捕获时的形状**。
- eager break：attention（`layers/radix_attention.py:191` 的 `@eager_on_graph`）、DSA indexer（`layers/cp/bcg/interleave_bcg.py` 的 `bcg_indexer_with_kv`）、KDA 的递归计算。每次 replay 都现场执行，读取当前 batch 的数据。

所以规则是：**捕获段里只能用地址固定、形状固定的张量；每个 batch 都会变的东西，要么放进 eager break，要么先拷进固定地址的 buffer。**

只开 PCP 时，K3.1 已经可以用 BCG：token 按 `i % cp_size` 平均切分，重排是固定形状的 reshape、transpose 或 `interleave_rows`。开了 DCP 之后，P3 改成按位置切分（`_build_position_owner_metadata`，`layers/cp/interleave.py`），问题就出在这里。

## 1. 问题

### 1.1 参数校验直接拒绝

- `arg_groups/cuda_graph_hook.py:143-155`：显式指定 prefill graph 后端时，PCP+DCP 直接报错。这就是部署时遇到的 `does not support a prefill CUDA graph yet`。
- `cuda_graph_hook.py:224`：自动规则在 `dcp_size > 1` 时关闭 BCG，注释是 *"Capture builds a dummy extend forward with attn_dcp_metadata=None"*。但 prefill graph runner 已经有 `_dcp_extend_active` 路径（`model_executor/runner/prefill_cuda_graph_runner.py:738`、`:1343`、`:1375`）：捕获时规划 DCP 元数据，replay 时用当前 batch 重新规划，所以这条注释描述的问题已经被处理了。

### 1.2 KDA 层的重排用了每个 batch 都会变的 `gather_index`（结果错误，最严重）

按位置切分时，每个 batch 都会新建 `gather_index = owner * stride + slot`。KDA 层的 `forward_pre_o_proj`（`models/kimi_k3.py:2675-2732`）在递归计算前后各调用一次：

```python
# cp_interleave_to_sequence_order
return hidden_states.index_select(0, metadata.gather_index)
# cp_sequence_to_interleave_order
return padded.index_copy_(0, metadata.gather_index, hidden_states)
```

这两步都在**捕获段**里。69 个 KDA 层，一次 forward 共 138 次：

- **地址失效**：graph 记住的是捕获时那个 `gather_index` 的地址。replay 前，`PrefillCPBCGInput.prepare` 会调用 `prepare_cp_forward` 重建元数据（`layers/cp/bcg/bcg.py:194`），新的 `gather_index` 在别的地址。replay 读的是旧地址，那块内存可能已经被释放或复用。结果是 token 顺序错乱、KDA 状态算错，**不会报错**。
- **长度不固定**：`gather_index` 的长度是这个 batch 的 `total_seq_lens`，而 graph 里 `index_select` 的输出长度固定为捕获时的长度。

### 1.3 replay 后 `gather_index` 的行步长不对（结果错误）

改代码时又发现一处。`PrefillCPBCGInput.prepare` 在 replay 时，会把每个 rank 的物理行数改成捕获时的 `captured_local_tokens`（`bcg.py` 中的 `metadata.per_rank_actual_token = [captured_local_tokens] * cp_size`），所以 rank-major 张量的行步长变成了 `captured`。但 `gather_index` 是按 eager 时的步长 `physical_rank_len(counts)` 算出来的。两个步长不同时（live batch 比 bucket 小时一般都不同），下面几处都会读错行：

- KDA 的重排（捕获段）；
- eager break 里的 KV gather（`run_indexer_with_kv_gather` → `gather_kv_cache`）；
- replay 之后输出 hidden state 的 gather（`execute_prefill_cp_bcg` → `cp_gather_after_forward`）；
- `InterleaveCPBCGInput.prepare` 里 input id 的重排（`cp_interleave_input_ids`，不使用 MegaMoE 时）。

### 1.4 bucket 大小按平均切分估算，按位置切分时会偏小（崩溃）

replay 前选 bucket 的依据是“每个 rank 最多分到多少行”。interleave 原来的估算是 `ceil(总 token 数 / cp)`（`interleave_bcg.py` 的 `_max_local_tokens`）。按位置切分时，命中前缀缓存的请求前缀按 page（64）对齐，新增 token 都从 rank 0 开始，短请求会集中落在编号小的 rank 上。

例如：20 个请求，每个都命中前缀、只新增 9 个 token，cp=8。

- 估算：ceil(180 / 8) = 23，按 8 对齐后是 24 行；
- 实际：rank 0 分到 20 × 2 = **40 行**。

这样会选中一个太小的 bucket，`prepare` 检查到 `live_physical_tokens > captured_local_tokens`，抛出 `RuntimeError`，不会退回 eager。`58000d63a2`（有 rank 分不到 query 时不走 CP）只处理 0 行的情况，这种分布不均的 batch 仍然会走 CP。

### 1.5 不需要改的部分

- **KV 写入**：`owned_kv_write` 在 `dsa_backend.py` 里的调用（attention 的 eager break）和在 `dsa_indexer.py` 里的调用（`bcg_indexer_with_kv` 这个 eager break），每次都读当前 batch 的 `local_out_cache_loc`。
- **DCP 下的 index-K gather**（`_dcp_gather_index_k`）、DSA metadata（`bs_idx` 过滤、page table 拼接）：都在 eager break 里，或者在 replay 前现场构造。
- **MoE 和层间通信**：`prepare` 已经把每个 rank 的物理行数统一成 `captured`，形状固定。
- **捕获用的虚拟 batch**：`capture_prepare` 带了 `seq_lens_cpu`，`extend_prefix_lens_cpu=[0]*bs`（`prefill_cuda_graph_runner.py:1727`、`:1751`），开 DCP 时会走按位置切分，满足 `build_metadata` 的断言。

## 2. 修改方案和实际改动

### 2.1 重排改用固定地址的 buffer（解决 1.2）

- `InterleaveContextParallelMetadata` 新增字段 `graph_gather_index`（`layers/cp/interleave.py`）。
- 新增 `_sequence_gather_index(metadata)`：设置了 `graph_gather_index` 就返回它，否则返回 `gather_index`。`cp_interleave_to_sequence_order` 和 `cp_sequence_to_interleave_order` 改为调用它。**只有这两个在捕获段里的重排读 `graph_gather_index`**；eager break 里的 gather 继续读 live 的 `gather_index`，长度等于 `total_seq_lens`，`_indexer_with_kv` 对 KV 行数的检查才能通过。
- `InterleaveCPBCGInput.create`：开 DCP 时，按最大 bucket 分配一个 int64 buffer `graph_gather_index`。
- `InterleaveCPBCGInput.prepare`：在 input id 重排之后，用 `fill_graph_gather_index` 填写 `graph_gather_index[:static_num_tokens]`：
  - 前 `total` 个是 live 的 `gather_index`；
  - 后面的指向**各不相同的补齐行**，也就是每个 rank 在 live 行数之后、`captured` 之前的那些空行。这样捕获段里的 `index_copy_` 不会重复写同一行，`index_select` 读到的也只是补齐行。这和只开 PCP 时，graph 输出 `graph_num_tokens` 行、尾部是补齐行的约定一致。
  - 补齐行不够时抛 `RuntimeError`。捕获时补齐行一定够：捕获 batch 的总长度就是 `static_num_tokens`，所以 `cp × captured ≥ static_num_tokens`。

### 2.2 replay 时修正 `gather_index` 的步长（解决 1.3）

`InterleaveCPBCGInput.prepare` 调用 `super().prepare` 之后，马上把 live 的 `gather_index` 换成捕获时的步长：

```python
metadata.gather_index = restride_gather_index(
    metadata.gather_index,
    old_stride=physical_rank_len(metadata.per_rank_logical_token),
    new_stride=metadata.per_rank_actual_token[0],   # replay 时等于 captured
)
```

`restride_gather_index` 把 `owner * old + slot` 改写成 `owner * new + slot`。因为 `slot < counts[owner] ≤ old`，这个改写不会出错。捕获时两个步长相同，直接返回原张量。修正放在 input id 重排之前，所以 1.3 列出的几处用的都是修正后的值。

### 2.3 bucket 大小按实际分配计算（解决 1.4）

- 新增 `position_layout_max_rank_rows(extend_lens, prefix_lens, size)`：用 `position_owned_rows_per_request` 计算每个 rank 分到的行数，取最大值。没有 prefix 信息时，返回上界 `Σ ceil(extend / size)`。
- `InterleaveCPBCGInput._max_local_tokens` 在开 DCP 时改用这个函数。
- `PrefillCPBCGInput._max_local_tokens`、`required_local_tokens`、`select_replay_bucket_for_batch` 增加 `prefix_lens` 参数，默认 `None`，已有调用方不受影响。runner 的两个调用点（`can_run` 和 `load_batch`）传入 `forward_batch.extend_prefix_lens_cpu`。zigzag 的 `_max_local_tokens` 只是同步了参数签名。
- 找不到放得下的 bucket 时，`can_run` 返回 False，这个 batch 走 eager，不会再走到 `prepare` 里的 `RuntimeError`。估算只依赖 CPU 上的长度，每个 rank 结果一样，所以所有 rank 一致地选择走 graph 还是走 eager。

### 2.4 放开参数校验（解决 1.1）

- `cuda_graph_hook.py` 中显式指定后端的分支：PCP+DCP 只允许 `breakable`，并且要求 `tp == attn_cp == dcp`；`full` 仍然拒绝，因为 full graph 下 attention 也会被捕获，没法现场读当前 batch 的数据。
- `cuda_graph_hook.py:224` 的 `dcp_size > 1` 自动规则**保持不变**。所以 `apply_prefill_cp_cuda_graph_default` 自动打开 BCG 之后，又会被这条规则关掉。第一版是**手动开启**：只有显式指定 `--cuda-graph-backend-prefill breakable` 才会生效。GPU 验证通过后，再考虑改成默认开启。

### 2.5 改动文件

| 文件 | 改动 |
|---|---|
| `python/sglang/srt/layers/cp/interleave.py` | 新字段 `graph_gather_index`；`_sequence_gather_index`；两个重排函数改为调用它 |
| `python/sglang/srt/layers/cp/bcg/interleave_bcg.py` | `graph_gather_index` buffer；`prepare` 里修正步长并填写 graph index；`_max_local_tokens` 按位置计算；新增 `position_layout_max_rank_rows`、`restride_gather_index`、`fill_graph_gather_index` |
| `python/sglang/srt/layers/cp/bcg/bcg.py` | `_max_local_tokens`、`required_local_tokens`、`select_replay_bucket_for_batch` 增加 `prefix_lens` |
| `python/sglang/srt/layers/cp/bcg/zigzag_bcg.py` | 同步参数签名 |
| `python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py` | 两个选 bucket 的调用点传入 `prefix_lens` |
| `python/sglang/srt/arg_groups/cuda_graph_hook.py` | 显式指定后端时，方案 A 放行 `breakable` |
| `test/registered/unit/cp/test_interleave_dcp_owner_layout.py` | 新增 `TestGraphGatherIndex`、`TestGraphBucketRowsUnderDcp` |
| `test/registered/unit/server_args/test_prefill_cp_with_dcp_args.py` | `TestPrefillCudaGraphWithPrefillCpAndDcp` 改为：`breakable` 放行，`full` 拒绝，宽度不相等拒绝 |

## 3. 显存影响

| 项目 | 增加量 | 说明 |
|---|---|---|
| `graph_gather_index` buffer | int64 × 最大 bucket，16384 token 时每个 rank 约 128 KB | 可以忽略 |
| 重排后的中间张量 | 无 | graph 里本来就固定为 bucket 长度，只开 PCP 时也一样 |
| 补齐行 | 无 | 指向 rank-major 张量里已有的空行 |
| 按实际分配选 bucket | 无 | 只改变选哪个已捕获的 bucket，代价是多一些补齐计算，或者退回 eager |

需要注意的是**开启 prefill graph 本身**的显存：graph 内存池、每个 bucket 一份的 DCP KV 收集 buffer（`_dcp_kv_buffers`）、输入 buffer。Maas 实测 K3 DSPARK prefill（不开 PCP）的 graph 内存池在 TP0 上约 8.09 GiB；PCP 下每个 rank 的本地行数是 1/8，按理会小很多，但没有实测。K3.1 每张卡留给 KV 的只有约 29 GiB，建议先只捕获少量短 extend 的 bucket（`--cuda-graph-bs-prefill`），再看 graph 捕获的 `mem usage` 和 `KV Cache is allocated ... #tokens` 这两行日志。

如果以后希望分布不均的 batch 也能用 graph，就要在捕获时给每个 bucket 留更多本地行。按位置切分时，每个 rank 的行数 ≤ 总 token 数 / cp + 请求数，所以最多多出“请求数”这么多行。按 hidden 7168、BF16 估算，每个 rank 多 64 行，rank-major 张量约多 7 MB。这一点本次没有做。

## 4. 验证情况

已经做的：

- 所有改动文件通过 `py_compile` 语法检查。
- 把 `position_layout_max_rank_rows`、`restride_gather_index`、`fill_graph_gather_index` 单独抽出来运行（真实 torch）：
  - 2000 个随机 batch：行数估算和逐个位置统计的结果一致；不带 prefix 时的上界不小于真实值；
  - 500 个随机按位置切分的布局：修正步长后按 captured 宽度读取，能还原序列顺序；graph index 的补齐行互不重复，读到的都是补齐值，`index_copy_` 能写回 live 行；补齐行不够时报错；
  - “20 个请求 × 9 个 token”的例子算出 40 行（原来的估算是 23）。

还没有做的：

1. **在开发机上运行 CPU 单元测试**：
   ```bash
   python -m pytest test/registered/unit/cp/test_interleave_dcp_owner_layout.py -v
   python -m pytest test/registered/unit/server_args/test_prefill_cp_with_dcp_args.py -k PrefillCudaGraph -v
   ```
2. **GPU 上对比 eager 和 BCG 的输出**：同一批请求，分别用 `--cuda-graph-backend-prefill disabled` 和 `breakable`，比较 logits 或者 greedy 输出。要覆盖单个长请求、多个短请求、命中前缀、分布不均几种 batch。
3. **确认 KDA 递归会屏蔽补齐行**：只开 PCP 时，这个约定由“KDA 用最新的序列表屏蔽补齐行”保证（见 `cp_interleave_to_sequence_order` 的说明），按位置切分的路径没有单独验证过。
4. **精度**：用 sgl-eval 跑 gsm8k，确认 BCG 和 eager 的分数一致。
5. **graph 显存**：确认捕获的 `mem usage`，必要时减少捕获的 bucket，或者调低 `--mem-fraction-static`。
