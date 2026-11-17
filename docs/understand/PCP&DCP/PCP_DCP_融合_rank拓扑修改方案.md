# PCP + DCP 融合（方案 A）：rank 拓扑修改方案

> 状态：草案，待人工评审
> 代码基线：`yhz/test/v0.5.23-k31` 分支（行号在该分支上核对过）
> 关联文档：`PCP_DCP_方案A_计划二.md` §4.2、`方案A计划问题汇总.md` §2、`PCP_DCP_融合_参数校验修改方案.md`
> 下文的代码片段只是改法示意，**没有运行验证**。标"（推断）"的条目确定性更低。

## 0. 结论

方案 A 的组合是 `tp_size == attn_cp_size == dcp_size == N`、attention TP 为 1。参数校验部分已经实现，见 `PCP_DCP_融合_参数校验修改方案.md`。在这个前提下：

1. **rank 和组的数值本来就对，公式不用改。** 现有代码算出的 `dcp_rank`、`attn_dcp_rank`、DCP 组成员，和"DCP 组 = CP 组"要求的完全一样（§1.1、§2.2）。`runtime_context.py` 和 `parallel_state.py` 只需要**补校验、改文档**，防止以后有人改了其中一边，另一边却没跟着改，导致结果静默算错。
2. **读代码时发现两类必须修改的问题**（计划二里写得不够具体）：
   - **启动时就会崩溃**：`ModelConfig.get_num_kv_heads(attn_tp_size=1, dcp_size=8)` 会算出 `1 // 8 = 0`，接着做除数，抛出 ZeroDivisionError。K3.1 构建 KV 池时会走到这里（§3.2）。
   - **头数放大 8 倍**：有 4 处用"`num_attention_heads // attn_tp_size × attn_dcp_size`"计算 decode 时的 Q 头数。attention TP 为 1 时，结果是 8H，正确值是 H（§3.3）。
3. 全仓 `attn_dcp_size` / `attn_dcp_rank` 的使用点实际是 **172 处，分布在 47 个文件**（计划二写的约 157 处是估算）。按语义分成 5 类（§3）：
   - 需要修改的：上面两类问题涉及的 6 个位置；
   - 只依赖 rank 和宽度、预计不用改、只需加测试的：约 110 处；
   - 其余：K3.1 不经过的路径，或者只是定义和文档。

---

## 1. `python/sglang/srt/runtime_context.py`

### 1.1 rank 推导：数值不变，不改公式

**位置：** `_configure_context`，第 1609-1612 行：
```python
# DCP groups are contiguous slices of a TP group.
if parallel.dcp_enabled:
    placement["dcp_rank"] = placement["tp_rank"] % parallel.dcp_size
placement["attn_dcp_rank"] = placement.get("dcp_rank", 0)
```

**为什么方案 A 下数值本来就对：** 第 126 行 `derive_spawn_ranks` 里有：
```
attn_cp_rank = (tp_rank // attn_tp_size) % attn_cp_size
```
方案 A 下 `attn_tp_size = 1`，`attn_cp_size = tp_size`，所以 `attn_cp_rank = tp_rank`。同时 `dcp_rank = tp_rank % dcp_size = tp_rank`，因为 `dcp_size == tp_size`。两者相等。

**怎么改：** 不改公式，只改注释，写明方案 A 依赖这个等式：
```python
# DCP groups are contiguous slices of a TP group. Under prefill CP with DCP
# the slice is the whole TP group, so dcp_rank == attn_cp_rank.
```
等式本身由 §1.3 新增的校验来保证。

### 1.2 宽度推导：不改

**位置：** `derive_parallel_widths`，第 137-157 行。其中第 156 行 `"attn_dcp_size": dcp_size if dcp_enabled else 1`。

**为什么不改：** 方案 A 下 `attn_dcp_size = dcp_size = attn_cp_size`，已经满足要求。计划二曾考虑在这里加一个"DCP 复用 CP 组"的标志。但这个函数有两个调用入口：publish 时经第 160 行 `parallel_widths_of` 调用；建组时在 `parallel_state.py:2738` 调用。加标志就得两处同时传，收益不大。所以不加。

### 1.3 `_validate_parallel`：补上 DCP 的校验（必须改）

**为什么要改：** `_validate_parallel`（第 266-349 行）在 publish（第 1614 行）、`override_permanently`（第 420 行）和 `override`（第 439 行）时都会执行。它目前检查 TP、PP、attention、MoE 的 rank 和宽度，**完全不检查 DCP**：
- 第 244-251 行的 `_RANK_AND_WIDTH` 里没有 `("attn_dcp_rank", "attn_dcp_size")`；
- 第 255-261 行的 `_WIDTH_AND_GROUP` 里没有 `("dcp_size", "dcp_group")`。

方案 A 依赖 `dcp_rank == attn_cp_rank`，而且 DCP 组和 CP 组要是同一批卡。只要有一边被改了，比如以后 DCP 组改成跨步构建，或者 CP rank 的布局变了，就会静默算错：owner 过滤和 CP 切分对不上，KV 写到错误的卡上，但不报错。

**怎么改：**
1. 第 244-251 行加一项：
   ```python
   ("attn_dcp_rank", "attn_dcp_size"),
   ```
   DCP 关闭时，第 1586 行会把 `attn_dcp_rank` 设为 0，`attn_dcp_size` 为 1，这一项仍然成立。
2. 第 255-261 行加一项：
   ```python
   ("dcp_size", "dcp_group"),
   ```
   DCP 关闭时 `dcp_group` 没有设置，读取会抛异常，第 336-337 行会跳过这一项。
3. 在第 328 行（layout 检查）之后，加上方案 A 的一致性检查：
   ```python
   cp_dcp_terms = ("dcp_size", "attn_cp_size", "attn_tp_size", "dcp_rank", "attn_cp_rank")
   dcp_w, cp_w, tp_w, dcp_r, cp_r = (read(n) for n in cp_dcp_terms)
   prefill_cp_with_dcp = (
       _UNREADABLE not in (dcp_w, cp_w) and dcp_w > 1 and cp_w > 1
   )
   if prefill_cp_with_dcp:
       if dcp_w != cp_w or tp_w not in (_UNREADABLE, 1):
           problems.append(
               "prefill CP with DCP needs dcp_size == attn_cp_size and attn_tp_size == 1\n"
               f"  dcp_size={dcp_w}, attn_cp_size={cp_w}, attn_tp_size={tp_w}"
           )
       if _UNREADABLE not in (dcp_r, cp_r) and dcp_r != cp_r:
           problems.append(
               "prefill CP with DCP needs dcp_rank == attn_cp_rank\n"
               f"  dcp_rank={dcp_r}, attn_cp_rank={cp_r}"
           )
   ```
   `attn_cp_size > 1` 在当前代码里只有开启 prefill CP 时才会出现。参数校验已经禁止其他 CP 和 DCP 的组合，所以这里用宽度判断就够了。
4. 组成员的一致性要等组都建好才能检查，放在 `parallel_state.py`（§2.3）。

### 1.4 字段文档：语义更新（建议改）

**位置：** `python/sglang/srt/arg_groups/fields/parallel.py`
- 第 367-370 行，`attn_dcp_size` 的 doc 是 "Decode context-parallel width inside the attention TP group."
- 第 403-408 行，`attn_dcp_rank` 的 doc 是 "Decode context-parallel rank inside the attention TP group, ..."

**为什么要改：** 方案 A 下 attention TP 为 1，DCP 组在 attention CP 组里，原来的描述不再成立。读代码的人如果照字面理解，可能会写出依赖"attention TP 组内"假设的代码。§3.3 的头数问题就是这种假设造成的。

**怎么改：**
```python
doc="Decode context-parallel width: inside the attention TP group, or the "
"attention CP group itself when prefill CP runs with DCP.",
```
`attn_dcp_rank` 做同样的修改。

---

## 2. `python/sglang/srt/distributed/parallel_state.py`

### 2.1 过时的 docstring（必须改，纯文档）

**位置：** `initialize_model_parallel` 的 docstring，第 2594-2597 行：
```
attn_dcp_size: ... Must be a divisor of `tp_size` and is currently only
    supported on the AMD HIP platform.
```
**为什么要改：** 第 2664-2672 行的代码已经支持 HIP、CUDA、NPU 和 XPU。"only supported on the AMD HIP platform"是错的。

**怎么改：**
```
attn_dcp_size: ... Must be a divisor of `tp_size`. Supported on CUDA, HIP,
    NPU and XPU. With prefill CP the DCP group is the attention-CP group.
```

### 2.2 DCP 组的构建：不改，也**不要**改成复用 `_ATTN_CP`

**位置：** 第 2705-2728 行：
```python
if decode_context_parallel_size > 1:
    dcp_group_ranks = []
    for tp_group in group_ranks:
        for start in range(0, len(tp_group), decode_context_parallel_size):
            dcp_group_ranks.append(tp_group[start : start + decode_context_parallel_size])
    _DCP = init_model_parallel_group(..., use_quick_allreduce=False, group_name="dcp", ...)
```
方案 A 下 `dcp_size == tp_size`，所以切出来的就是整个 TP 组，成员和 `_ATTN_CP` 相同。第 2753-2754 行在 `attn_cp_size == tp_size` 时直接让 `_ATTN_CP = _TP`。

**为什么不改成 `_DCP = _ATTN_CP`：**

| 问题 | 位置 |
|---|---|
| `_DCP` 在 `_ATTN_CP` 之前构建（第 2706 行 vs 第 2748 行），复用就要调整构建顺序 | 2706、2748 |
| 第 2745 行的 `dcp_enabled=_DCP is not None` 依赖这个顺序 | 2745 |
| `_DCP` 显式设置了 `use_quick_allreduce=False`。原因见注释：QuickAllReduce 不服务 DCP 用到的集合通信，却会在每个 rank 上预留很大的 workspace。复用 `_TP` 后这个设置就丢了 | 2719-2722 |
| `destroy_model_parallel` 没有判断别名，会对同一个对象调用两次 `destroy()`。`GroupCoordinator.destroy`（第 2113-2132 行）大部分字段销毁后会置 None，基本是幂等的，但 `pymscclpp_comm.destroy()` 没有置 None | 3293-3296、3315-3317 |

保持独立的组对象，成员相同，只需要加一条断言（§2.3）。这样改动最小，也不影响现有 TP 组的通信器配置。

### 2.3 组成员一致性断言（必须改）

**位置：** 在第 2771 行附近，`_ATTN_CP` 构建完成之后插入：
```python
if decode_context_parallel_size > 1 and attn_cp_size > 1:
    # Prefill CP with DCP: the DCP group must be the attention-CP group.
    assert sorted(_DCP.ranks) == sorted(_ATTN_CP.ranks), (
        f"DCP group {_DCP.ranks} differs from attention-CP group {_ATTN_CP.ranks}"
    )
```
`GroupCoordinator` 有 `ranks` 属性，`layers/layer_boundary/layout.py` 的 `_same_ranks` 就是这样比较的。

**为什么要加：** §1.3 的校验只比较 rank 数值，这里比较组的实际成员。两者一起，才能完整锁住"DCP 组 = CP 组"这个前提。

### 2.4 `patch_tensor_parallel_group`：P1/P2 不改，P4 要改

**位置：** 第 3111-3166 行。`owns_attention=True` 时，会把 `attn_cp_size` 设为 1、`attn_cp_group` 设为 None（第 3141-3156 行），但 **DCP 相关的值都不变**（`dcp_size`、`dcp_rank`、`dcp_group`、`attn_dcp_size`、`attn_dcp_rank`）。

**为什么 P1/P2 不改：** 这个函数只在投机解码的 draft 运行时调用。参数校验已经在方案 A 下拒绝了投机解码。

**P4 要怎么改：** draft 按普通 TP 运行，不加入 DCP 组。所以 narrowing 时要同时设置 `attn_dcp_size=1`、`attn_dcp_rank=0`；或者沿用现有 draft 路径的约定，例如 `dsa_draft_shards_under_dcp`，决定 draft 是否在 DCP 下分片。到 P4 时和 DFLASH 一起设计。

### 2.5 不需要改的部分

- 第 2673-2677 行：`tp % dcp == 0` 的检查保留。
- 第 2980-2997 行：把组记录到 context 里（`"dcp_group": _DCP`）。
- 第 2330 行：`graph_capture` 按 `id()` 去重，DCP 组和 CP 组是两个对象，各自进入一次，不影响结果。

---

## 3. 全仓 `attn_dcp_size` / `attn_dcp_rank` 的使用点

统计命令：`grep -rn 'attn_dcp_size\|attn_dcp_rank' --include='*.py' python/sglang`，共 **172 行，47 个文件**。

判断标准：使用点**只依赖 DCP 的 rank 和宽度**（owner 规则 `pos % N`、放大的页、gather/合并的组大小）时，方案 A 下数值不变，结果正确；使用点**把 DCP 宽度和 attention TP 宽度或头数组合在一起**时，隐含了"DCP 在 attention TP 组内"的假设，方案 A 下会出错。

### 3.1 分类汇总

| 类别 | 处理 | 文件（行数） |
|---|---|---|
| **A. 启动崩溃** | 必须改 | `configs/model_config.py`（`get_num_kv_heads` 本身没有直接引用这两个字段，但它们以参数形式传进来）；调用点 `mem_cache/kv_cache_configurator.py:2711-2714` |
| **B. 头数 × dcp** | 必须改（P2，和 D1/D2 的选择相关） | `layers/attention/dsa_backend.py:755-758、770`；`models/deepseek_v2.py:2262-2264`（通过 2116 行的 `self.dcp_size` 间接用到）；`models/deepseek_common/attention_forward_methods/forward_mla.py:93-97` |
| **C. 只依赖 rank 和宽度** | 不改，加测试兜底 | `dsa_dcp_sharded.py`（9）、`dsa_backend.py` 其余（约 24）、`memory_pool.py`（8）、`hybrid_pool_assembler.py`（7）、`kv/alloc/*`（`runner_ids` 3、`chain` 3、`band` 3、`id_space` 2、`interim` 1）、`pool_configurator.py`（4）、`cache_controller.py`（3）、`linker_pool_assembler.py`（2）、`mooncake_direct_linker.py`（2）、`layers/dcp/layout.py`（2）、`dsa_indexer.py`（2）、`kernels/ops/kvcache/mla_buffer.py`（2）、`dflash_worker_v2.py`（2）、`allocation_sizing.py`、`draft_kv_budget.py`、`kimi_k3.py`、`kimi_k31_verify_dsa_binding.py`、`eager_runner.py`、`prefill_cuda_graph_runner.py`、`flashmla_backend.py`、`cutedsl_mla_backend.py`（各 1），以及 `kv_cache_configurator.py` 里不属于 A 类的那部分（id 空间放大，2866-2868、2952-2954 行等） |
| **D. K3.1 不经过的路径** | 第一期不改 | `ascend_backend.py`（7）、`memory_pool_npu.py`（3）、`deepseek_v2_attention_mla_npu.py`（2）、`dsa_npu_indexer.py`（1）、`aiter_backend.py`（4）、`triton_backend.py`（4）、`flashinfer_backend.py`（3）、`xpu_backend.py`（2）、`trtllm_mla_backend.py`（6）、`flashinfer_mla_backend.py`（1）、`qwen3_5.py`（1）、`forward_batch_info.py`（2，只在 HIP 下生效）、`disaggregation/encoder/server.py`（1）、`test/kits/*`（3）；以及 `kv_cache_configurator.py` 里 ascend、SWA、MHA、unified 池的 `get_num_kv_heads` 调用 |
| **E. 定义和文档** | 见 §1.4、§2.1 | `runtime_context.py`（7）、`arg_groups/fields/parallel.py`（3）、`parallel_state.py`（2） |
| **F. PD 传输** | P4 | `disaggregation/common/conn.py:416-417` |

### 3.2 A 类：`get_num_kv_heads` 除以零（必须改，P1）

**位置：** `python/sglang/srt/configs/model_config.py`，第 1845-1856 行：
```python
def get_num_kv_heads(self, tensor_parallel_size: int, dcp_size: int = 1) -> int:
    """Number of KV heads per GPU.

    DCP ranks replicate KV, so heads shard across ``tp // dcp`` groups. ...
    """
    total_num_kv_heads = self.get_total_num_kv_heads()
    if self.is_draft_model:
        dcp_size = 1
    kv_tensor_parallel_size = tensor_parallel_size // dcp_size
    return max(1, total_num_kv_heads // kv_tensor_parallel_size)
```
**调用点（K3.1 必定经过）：** `mem_cache/kv_cache_configurator.py`，`_build_hybrid_linear_kv_pool`（第 2599 行起）的第 2711-2714 行：
```python
head_num=self.model_config.get_num_kv_heads(
    get_parallel().attn_tp_size, get_parallel().attn_dcp_size
),
```
K3.1 是 MLA 加 KDA 的混合模型，会走这个函数（`pool_class = HybridLinearKVPool`，`use_mla=True`）。这次调用不受任何分支条件限制，一定会执行。

**为什么会崩：** 方案 A 下 `attn_tp_size=1`、`attn_dcp_size=8`，`kv_tensor_parallel_size = 1 // 8 = 0`。下一行 `total_num_kv_heads // 0` 抛出 ZeroDivisionError，**KV 池无法构建，服务启动失败**。

**怎么改：** 这个函数的语义是"DCP rank 之间复制 KV，KV 头在 `tp // dcp` 组之间切分"。DCP 比 attention TP 宽时，组数应当取 1，也就是每张卡持有全部 KV 头：
```python
kv_tensor_parallel_size = max(1, tensor_parallel_size // dcp_size)
```
同时更新 docstring，写明 "DCP wider than attention TP (prefill CP with DCP) keeps every KV head"。

对现有配置的影响：TP8+DCP8 时 `8 // 8 = 1`，结果不变。只有 `tp < dcp` 时行为才会变化，而这种组合以前直接崩溃，不存在可用的旧行为。

**同类调用点（K3.1 不经过，修改后自然受益）：** `kv_cache_configurator.py:1109、1237、1286、2166、2278、2509、2559、2586、2803`，以及 `model_executor/pool_configurator.py:549、560`。

### 3.3 B 类：decode Q 头数 × dcp（必须改，P2）

这几处都默认"每卡头数 = H / attn_tp，DCP 组内 all-gather 后得到 H / attn_tp × dcp"。方案 A 下 attention TP 为 1，按这个算法会得到 8H。

| 位置 | 现有代码 | 方案 A 下的结果 |
|---|---|---|
| `layers/attention/dsa_backend.py:755-758` | `local_num_q_heads = num_attention_heads // attn_tp_size`；`num_q_heads = local_num_q_heads * self.attn_dcp_size` | `num_q_heads = 8H`。它被 trtllm 和 flashmla_kv 的头数、prims_ts 的支持检查使用 |
| `layers/attention/dsa_backend.py:770` | `num_dcp_q_heads = _trtllm_padded_heads(num_q_heads * self.dcp_size)`。K3.1 下 `self.dcp_size` 是 1（第 764 行） | 8H，multi-CTA 计数器按 8H 分配 |
| `models/deepseek_v2.py:2262-2264` | `attn_mqa_for_dcp_decode = RadixAttention(self.num_local_heads * self.dcp_size, ...)`；其中 `self.dcp_size` 来自第 2115-2117 行的 `mla_attention_dcp_size(dcp_size=attn_dcp_size)` | 8H 个头的 decode attention 层 |
| `models/deepseek_common/attention_forward_methods/forward_mla.py:93-97` | `_select_local_dcp_heads_for_autotune`：`attn_output.narrow(1, rank * num_local_heads, num_local_heads)` | 如果在切片上下文之外调用，`num_local_heads = H`，narrow 会越界（推断） |

**正确的值：** 不管选 D1 还是 D2，decode 时 kernel 看到的 Q 头数都应该是 **H**。
- **D2**（cp-decode-attn-tp 切片）：decode 时每卡 `num_local_heads = H / N`，DCP 组内 gather 后得到 H。
- **D1**（不切片）：每卡本来就有 H 个头，跳过 Q 的 all-gather，也是 H。

**怎么改（推荐统一成一个 helper）：** 在 `runtime_context.py` 里新增一个函数，返回 decode 时每卡负责的头数切分份数：
```python
def decode_attn_head_shards() -> int:
    """How many ranks one decode step splits attention heads over before the
    DCP gather: attention TP, times attention CP when prefill CP runs with DCP
    and decode slices the replicated attention (cp decode attention TP)."""
    parallel = get_parallel()
    if parallel.attn_cp_size > 1 and parallel.attn_dcp_size > 1:
        return parallel.attn_tp_size * parallel.attn_cp_size
    return parallel.attn_tp_size
```
然后把上面几处改成：
- `dsa_backend.py:755-758`：prefill 的 `local_num_q_heads` 保持 `H // attn_tp_size`，也就是 H。decode 的 `num_q_heads` 改成 `H // decode_attn_head_shards() * attn_dcp_size`，结果是 H。两个值要分开存，因为这个文件的注释也写明了 "prefill attends with the local heads"。
- `dsa_backend.py:770`：跟着 `num_q_heads` 修正。
- `deepseek_v2.py:2262-2264`：`attn_mqa_for_dcp_decode` 的头数改成 `num_heads // decode_attn_head_shards() * dcp_size`。
- `forward_mla.py:93-97`：narrow 的宽度改成 decode 切片后的头数，或者断言它只在切片上下文里被调用。

选 D1 时，helper 返回 `attn_tp_size`，同时要去掉 `× attn_dcp_size`（Q 不再 gather）。具体实现等 P2 实验出结果后再定。这一节先确定一个原则：**这几处必须共用同一个来源，不能各自推导。**

**K3.1 主路径不经过、但同样有这个模式的地方**（P4，或扩展到其他模型时再处理）：
- `trtllm_mla_backend.py:296-300` 的 `num_decode_q_heads = num_q_heads * attn_dcp_size`，以及继承它的 `cutedsl_mla_backend.py:147-148`。K3.1 的 DFLASH draft 用 cutedsl_mla，P4 开启投机解码时要一起改。
- `triton_backend.py:105、270`、`flashinfer_backend.py:416、1626、1933`：通过 `get_num_kv_heads(attn_tp, attn_dcp)` 计算，§3.2 修改后自然正确。

### 3.4 C 类：只依赖 rank 和宽度（不改，加测试）

这些使用点用 `attn_dcp_size` 和 `attn_dcp_rank` 来做以下几件事：
- owner 过滤：`pos % dcp == rank`（`dsa_dcp_sharded.py:394-395、420-421`，`layers/dcp/layout.py`）；
- 放大的页：`page_size × dcp`（`layout.py:41`，`allocation_sizing.py:25`，`dsa_dcp_sharded.py:458、525`，`dsa_backend.py:743`）；
- 放大的 id 空间：`kv/alloc/*`，`kv_cache_configurator.py:2866-2868、2952-2954`；
- 长度切分：`get_dcp_lens`（`dsa_backend.py:1284`，`dsa_indexer.py:1934`）；
- 组大小断言：`dsa_dcp_sharded.py:287` 的 `assert group.world_size == parallel.attn_dcp_size`；
- 主机池和存储的 rank（`hybrid_pool_assembler.py`、`cache_controller.py`、`mooncake_direct_linker.py`、`mla_buffer.py`）。

它们都不涉及 attention TP，所以只要 §1.3 和 §2.3 保证了 `attn_dcp_rank == attn_cp_rank`、DCP 组 == CP 组，方案 A 下结果就正确（推断）。

**还要注意一点：** `dsa_dcp_sharded.py:287-288` 用 `parallel.dcp_group` 建 `DcpTopKMerge`。这个组和 CP 组是不同的对象、相同的成员（§2.2）。multicast 的 handle 绑定的是 DCP 组，和 CP 组的通信互相独立，不冲突（推断）。

### 3.5 F 类：PD 传输（P4）

`disaggregation/common/conn.py:416-417` 设置 `self.dcp_size = parallel.attn_dcp_size`，`self.dcp_rank = parallel.attn_dcp_rank`。方案 A 下这两个值正确，但还有两个问题：一是注册信息里没有 prefill 的 `dcp_size`；二是代码混用了真实的 attention TP（1）和注册时的 TP（8）。这两点见 `方案A计划问题汇总.md` §10。第一期拒绝 PD 分离，所以不改。

---

## 4. 修改清单汇总

| # | 文件 | 行号 | 改动 | 必要性 | 阶段 |
|---|---|---|---|---|---|
| 1 | `runtime_context.py` | 1609-1612 | 只改注释，公式不变 | 建议 | P1 |
| 2 | `runtime_context.py` | 244-261、328 行之后 | `_validate_parallel` 补 DCP 的 rank、宽度、组检查，以及方案 A 的一致性检查 | **必须** | P1 |
| 3 | `runtime_context.py` | 新增 | `decode_attn_head_shards()` helper | **必须** | P2 |
| 4 | `arg_groups/fields/parallel.py` | 367-370、403-408 | 更新 `attn_dcp_size`/`attn_dcp_rank` 的 doc | 建议 | P1 |
| 5 | `distributed/parallel_state.py` | 2594-2597 | 修正过时的 docstring | 必须（文档） | P1 |
| 6 | `distributed/parallel_state.py` | 约 2771 行之后 | 断言 `_DCP.ranks == _ATTN_CP.ranks` | **必须** | P1 |
| 7 | `distributed/parallel_state.py` | 2705-2728 | 不改，不复用 `_ATTN_CP` | — | — |
| 8 | `distributed/parallel_state.py` | 3111-3166 | `patch_tensor_parallel_group` 补 DCP narrowing | 必须 | P4 |
| 9 | `configs/model_config.py` | 1855 | `kv_tensor_parallel_size = max(1, tp // dcp)` | **必须（否则启动失败）** | P1 |
| 10 | `layers/attention/dsa_backend.py` | 755-758、770 | decode Q 头数改用 helper | **必须** | P2 |
| 11 | `models/deepseek_v2.py` | 2262-2264 | `attn_mqa_for_dcp_decode` 的头数改用 helper | **必须** | P2 |
| 12 | `models/deepseek_common/attention_forward_methods/forward_mla.py` | 93-97 | autotune 的 narrow 宽度 | 必须 | P2 |
| 13 | C 类约 110 处 | — | 不改，加测试兜底 | — | P1 |
| 14 | `trtllm_mla_backend.py`、`cutedsl_mla_backend.py` | 296-300、147-148 | decode 头数改用 helper | 必须 | P4 |

## 5. 建议的测试

在 `test/registered/unit/` 下添加 CPU 测试：

1. **`get_num_kv_heads` 的边界**（`configs/` 下）：
   - `(tp=1, dcp=8)` 返回全部 KV 头，不抛 ZeroDivisionError；
   - `(tp=8, dcp=8)`、`(tp=8, dcp=1)` 的结果和修改前一致，作为回归保护。
2. **`_validate_parallel`**（`server_args/` 或 `distributed/` 下）：构造 `dcp_size=attn_cp_size=8, attn_tp_size=1` 的 parallel context，验证以下 3 种情况会报错：
   - `dcp_rank != attn_cp_rank`；
   - `dcp_size != attn_cp_size`；
   - `attn_tp_size > 1`。
   同时验证 DCP 关闭时，`attn_dcp_rank=0, attn_dcp_size=1` 能通过。
3. **rank 推导的不变量**：对 tp=cp=dcp=8 的每个 `world_rank`，用 `derive_spawn_ranks` 加 `_configure_context` 的逻辑算出的 `dcp_rank`，都等于 `attn_cp_rank`。
4. **`decode_attn_head_shards`**：TP8+DCP8 返回 8，方案 A 返回 8，TP8 不开 DCP 返回 8，TP2+PCP2 不开 DCP 返回 1（attention TP 宽度，PCP 不切 decode 的头）。验证在 TP8+DCP8 和方案 A 下，`num_q_heads` 都是 H。
5. **组成员断言**（需要多进程，放到 GPU 测试里）：方案 A 下 `_DCP.ranks == _ATTN_CP.ranks`。

## 6. 已实现的测试和运行方式

第 1、2、4 项已经作为 CPU 测试加进了现有的测试文件：

- `test/registered/unit/configs/test_model_config_shapes.py` 的 `TestNumKvHeadsUnderDcp`：
  - `(tp=1, dcp=8)` 不再出现除零错误；
  - `(8,8)`、`(8,4)`、`(8,1)` 的结果和修改前一致。
- `test/registered/unit/test_runtime_context.py` 的 `TestTheTopologyIdentities`：
  - 同时开启 prefill CP 和 DCP、rank 一致时校验通过，`decode_attn_head_shards()` 从 2 变为 4；
  - `dcp_rank` 和 CP rank 不一致时报错；
  - DCP 比 CP 窄时报错；
  - `attn_dcp_rank` 越界时报错。

这两个文件都注册在 `base-a-test-cpu`。在装好 sglang 的环境里，从仓库根目录运行：

```bash
python -m pytest test/registered/unit/test_runtime_context.py test/registered/unit/configs/test_model_config_shapes.py
```

只跑本次新增的用例：

```bash
python -m pytest test/registered/unit/configs/test_model_config_shapes.py -k TestNumKvHeadsUnderDcp
python -m pytest test/registered/unit/test_runtime_context.py -k "prefill_cp_with_dcp or attention_dcp_rank or group_built"
```

第 3、5 项（rank 推导的不变量、组成员断言）还没有实现。
