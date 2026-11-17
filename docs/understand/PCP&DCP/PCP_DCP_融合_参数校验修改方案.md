# PCP + DCP 融合（方案 A）：参数校验修改方案

> 状态：草案，待人工评审
> 代码基线：`infrawaves/dev/v0.5.23-k31` 分支（行号在该分支上核对过）
> 关联文档：`PCP_DCP_方案A_计划二.md` §4.1、`docs/understand/方案A计划问题汇总.md` §1
> 下文的代码片段只是改法示意，**没有运行验证**。标"（推断）"的条目确定性更低。

## 0. 范围与前提

本文只涉及 `python/sglang/srt/arg_groups/` 下的 7 个参数校验文件。它们的作用是**允许方案 A 的组合启动，同时把第一期不支持的组合拦下来**，不改变运行时行为。方案 A 要真正跑对，还需要计划二 §4.2–4.7 的运行时改动。

方案 A 的组合定义如下：

```
--tp-size N --enable-prefill-cp --attn-cp-size N --cp-strategy interleave --dcp-size N
（dcp_size == attn_cp_size == tp_size，attention TP = 1，--dcp-kv-layout token）
```

### 0.1 hook 的执行顺序

这些 hook 在 `arg_groups/pipeline.py` 中按下面的顺序执行。几处改法受这个顺序影响：

```
55   handle_mega_moe
180  handle_decode_context_parallelism       ← parallel_hook.py:257
202  handle_cuda_graph_config                ← cuda_graph_hook.py
238  handle_model_specific_adjustments       ← 模型 override（kimi_k3.py / deepseek_v2.py）
285  handle_hicache                          ← hicache_hook.py
291  handle_context_parallelism              ← parallel_hook.py:97，内部调用 cp_tp_group_sharing_hook
339  validate_mega_moe_token_budget_for_model ← mega_moe_hook.py
```

对改法的影响：
- 第 180 行的 DCP hook 运行时，`--cp-strategy` 可能还没设置，模型 override 也还没执行。
- 第 285 行的 `handle_hicache` 运行时，`enable_cp_tp_group_sharing` 还没声明。

---

## 1. `arg_groups/parallel_hook.py`

### 为什么要改
这里直接禁止 PCP 和 DCP 同时开启，方案 A 首先要放开这个限制。

### 改第几行
第 266-267 行，在 `handle_decode_context_parallelism`（第 257 行起）中：
```python
if cfg.dcp_size > 1 and cfg.enable_prefill_cp:
    raise ValueError("--dcp-size > 1 does not support --enable-prefill-cp.")
```

### 怎么改
把直接报错改成"只放行方案 A 的组合"。这个 hook 在流水线第 180 行执行，此时 `cp_strategy` 可能还是 None（"必须设置 `--cp-strategy`"的检查在第 154 行，属于更晚执行的 `handle_context_parallelism`），模型 override 也还没跑。所以这里只检查命令行上已经能确定的宽度：
```python
if cfg.dcp_size > 1 and cfg.enable_prefill_cp:
    if not (cfg.dcp_size == cfg.attn_cp_size == cfg.tp_size):
        raise ValueError(
            "--dcp-size with --enable-prefill-cp requires "
            "--dcp-size == --attn-cp-size == --tp-size (scheme A).")
    if cfg.cp_strategy not in (None, "interleave"):
        raise ValueError("Scheme A supports only --cp-strategy interleave.")
    if cfg.dcp_kv_layout != "token":
        raise ValueError("Scheme A requires --dcp-kv-layout token.")
```

### 同一个文件里的第二处修改
在 `handle_context_parallelism` 第 160-167 行的 `if view.attn_cp_size > 1:` 分支里，补上方案 A 的最终断言。这时 attention 各维度的宽度已经推导出来：
```python
if cfg.dcp_size > 1 and cfg.enable_prefill_cp:
    assert view.attn_tp_size == 1 and view.attn_dcp_size == view.attn_cp_size
```

---

## 2. `arg_groups/cp_tp_group_sharing_hook.py`

### 为什么要改
K3.1 的 prefill CP 只能走 CP-TP group sharing。这里的 unsupported 列表也拒绝 DCP，即使第 1 节放开了限制，到这里仍会报错。

### 改第几行
第 77-82 行的 `unsupported` 字典，其中第 79 行：
```python
"dcp-size > 1": cfg.dcp_size > 1,
```
这个字典由三个模型共用（`Glm5NextForConditionalGeneration`、`Qwen4ExpForConditionalGeneration`、`KimiK31ForConditionalGeneration`）。直接删掉这一行，另外两个模型也会放开 DCP。

### 怎么改
只对 K3.1 去掉这一项，并加上 `dcp == attn_cp` 的检查。改动放在第 107-109 行的 K3.1 分支里：
```python
elif arch == "KimiK31ForConditionalGeneration":
    unsupported["disaggregation-mode decode"] = cfg.disaggregation_mode == "decode"
    unsupported.pop("dcp-size > 1")                     # 方案 A 放行
    unsupported["dcp-size != attn-cp-size"] = (
        cfg.dcp_size > 1 and cfg.dcp_size != view.attn_cp_size)
```
这个函数在流水线第 291 行执行，第 39-50 行已经确认过 `attn_cp_size == tp_size` 和 `cp_strategy == interleave`。所以模型级的最终校验放在这里最合适。

---

## 3. `arg_groups/model_overrides/kimi_k3.py`

### 为什么要改
只要 `dcp_size > 1`，`_kimi_k31_overrides`（第 190-199 行）就会调用 `_kimi_k31_dcp_overrides`，而后者也拒绝 prefill CP。

### 改第几行
`_kimi_k31_dcp_overrides` 位于第 214-268 行，unsupported 字典中的第 240-241 行：
```python
"prefill context parallel": cfg.enable_prefill_cp
or (cfg.attn_cp_size or 1) > 1,
```

### 怎么改
1. **把第 240-241 行收紧，不要删除。** 只放行方案 A 的组合，其他组合继续拒绝：
   ```python
   scheme_a = cfg.enable_prefill_cp and cfg.dcp_size == cfg.attn_cp_size == cfg.tp_size
   ...
   "prefill context parallel other than scheme A":
       (cfg.enable_prefill_cp or (cfg.attn_cp_size or 1) > 1) and not scheme_a,
   ```
2. **第一期在方案 A 下额外拒绝以下组合**（加进同一个字典）：
   - 投机解码：`scheme_a and cfg.speculative_algorithm is not None`。第 225-226 行原本允许 DSPARK 和 DFLASH，但方案 A 下都没验证过。
   - PD 分离：`scheme_a and cfg.disaggregation_mode != "null"`。
   - 后端不是 `flashmla_kv`：`scheme_a and (cfg.dsa_prefill_backend, cfg.dsa_decode_backend) != ("flashmla_kv", "flashmla_kv")`。DSA 后端的 `_validate_dcp_backend`（`dsa_backend.py:539-567`）只要求两侧相同，可选 flashmla_kv、trtllm 或 prims_ts；P1 再收窄到 flashmla_kv 一种。
3. **复核第 270-276 行的 `_dcp_replicate_q_proj_default`。** 方案 A 下 attention TP=1，Q 投影本来就是复制的。`dcp_replicate_q_proj=True` 在这种情况下是什么语义、有什么效果，需要重新确认（推断）。P1 先不改代码，P2 再验证。

---

## 4. `arg_groups/model_overrides/deepseek_v2.py`

### 为什么列进来
`_deepseek_family_overrides` 的注册列表（第 39-55 行）里有 `KimiK31ForConditionalGeneration`，所以它对 K3.1 也会执行，并且在 prefill CP 下会改写并行参数。第一版计划漏了这个文件。

### 相关行
- 第 104-106 行：K3.1 跳过 `_dsa_dcp_overrides`，注释是 "K3.1 DCP: _kimi_k31_dcp_overrides"。
- 第 111-140 行，`if cfg.enable_prefill_cp:` 分支：
  - 第 115-119 行：用 `context_parallel_attn_dp_size` 设置 `attn_dp_size`，并设 `dp_size=1`；
  - 第 131-133 行：`assert cfg.tp_size <= 8`，不允许跨机器做 CP；
  - 第 137-141 行：interleave 且 `attn_cp_size > 1` 时，保留用户给的 `attn_cp_size`。
- 第 168-172 行：`index_kpool > 1 and dcp_size > 1` 时报错。

### 怎么改
**预计不用改代码，但必须核对。** 在方案 A 的典型配置（tp=8、interleave、attn_cp=8、没有 attention DP）下：
- `attn_dp_size` 会被设为 1，`attn_cp_size` 保持 8；
- `tp_size <= 8` 成立；
- 如果 K3.1 的 `index_kpool` 是 1（推断），第 168 行不会报错。

需要记录的约束：
1. 第 131 行的断言决定了**方案 A 只能单机运行**（tp ≤ 8）。这一点应写进计划的限制里。
2. 第 112 行的 warning 写着 "only been verified on Hopper"，而 K3.1 主要跑在 Blackwell 上。
3. 如果核对后发现 K3.1 的 `index_kpool > 1`，要在这里给方案 A 加一条明确的报错。

---

## 5. `arg_groups/hicache_hook.py`

### 为什么要改
计划规定第一期方案 A 不支持 HiCache。但 K3.1 跳过了 `_dsa_dcp_overrides`，现有代码在 DCP 下是允许 HiCache 的，所以要新加一条拒绝规则。

### 改第几行
`resolve_hicache_dcp_compatibility`（第 258-299 行），由 `handle_hicache` 在第 110 行调用。在流水线里它于第 285 行执行，早于 `handle_context_parallelism`（第 291 行）。所以这时 `enable_cp_tp_group_sharing` **还没声明**，只能用 `enable_prefill_cp`、`attn_cp_size`、`dcp_size` 来判断。

### 怎么改
在第 261 行的提前返回之后插入：
```python
if cfg.enable_prefill_cp and cfg.attn_cp_size > 1:
    raise NotImplementedError(
        "HiCache is not supported with --dcp-size and --enable-prefill-cp (scheme A) yet.")
```
同一个文件里其他和 DCP、CP 相关的检查保持不变：
- 第 125 行：per-node 检查；
- 第 162 行：draft node merge 检查；
- 第 214 行：storage page size 检查；
- 第 350 行：key scheme 检查；
- 第 419 行：Mooncake 布局检查。

---

## 6. `arg_groups/cuda_graph_hook.py`

### 为什么要改
方案 A 第一期不支持 prefill CUDA Graph。默认情况下 graph 会被自动关闭，但用户显式指定 backend 时会绕过这个关闭，而 full graph 模式完全没有拦截。

### 相关行
- 第 213-216 行：breakable 的规则 `"decode context parallel (dcp_size > 1)"`，会自动关闭 BCG。这才是 K3.1 实际命中的规则。计划二引用的 208-212 行是 CP 的规则，K3.1 在它的白名单里，不会命中。
- 第 143-144 行：在 `apply_cuda_graph_compatibility` 中，用户显式设置了 `--cuda-graph-backend-prefill`（`(Phase.PREFILL, "backend") in server_args._cuda_graph_config_locked`）时会**直接 return**，第 213-216 行的规则就不会执行。
- 第 258-276 行：`disable_full_prefill_cudagraph_if_incompatible` 的规则列表是**空的**。full graph 加 DCP 时 `_dcp_kv_buffers` 不会被捕获，回放时可能 KeyError（推断）。

### 怎么改
在第 143 行的"locked 提前返回"**之前**插入一条不受用户选择影响的检查：
```python
if (cfg.dcp_size > 1 and cfg.enable_prefill_cp
        and cfg.cuda_graph_config.prefill.backend != Backend.DISABLED):
    if (Phase.PREFILL, "backend") in server_args._cuda_graph_config_locked:
        raise ValueError("Scheme A does not support prefill CUDA graph yet; "
                         "use --cuda-graph-backend-prefill disabled.")
```
用户没有锁定 backend 时，仍由第 213-216 行的规则自动关闭 graph。也可以把同样的条件放进 `validate_cuda_graph_config`（第 444 行）。

---

## 7. `arg_groups/mega_moe_hook.py`

### 为什么要改
用 MegaMoE 时，每卡的 token 数不能超过 `SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK`。方案 A 改成每个请求单独切分、单独补齐后，单卡的最大 token 数会比现在的估算大。

### 改第几行
第 239-245 行，在 `validate_mega_moe_token_budget_for_model` 中（流水线第 339 行）：
```python
if cfg.enable_prefill_cp:
    token_partition_size = cfg.attn_cp_size
    token_alignment = token_partition_size if kimi else 1
    local_chunked_prefill_size = (prefill_tokens + token_partition_size - 1) // token_partition_size
```
现在的估算 `ceil(tokens / N)` 对应现有的 interleave：整批按拼接下标取模，各卡最多相差 1 个 token。方案 A 按位置取模后，每个请求在每张卡上最多多 1 个 token，所以单卡最多约 `ceil(tokens/N) + 请求数` 个（推断）。

### 怎么改
方案 A 下加上请求数带来的余量。请求数的上限取 `min(cfg.max_running_requests, prefill_tokens)`，需要确认在这一阶段 `max_running_requests` 已经确定。保守做法是直接用 `prefill_tokens`：
```python
if cfg.enable_prefill_cp:
    ...
    if cfg.dcp_size > 1:   # scheme A: per-request position-mod split
        max_reqs = cfg.max_running_requests or prefill_tokens
        local_chunked_prefill_size += min(max_reqs, prefill_tokens)
```
另外，P1 的运行时切分代码（`layers/cp/interleave.py`）也要加断言，保证每卡行数不超过这个上限。参数阶段的估算和运行时的实际值，任何一边有偏差都可能导致越界。

---

## 8. 汇总

| 文件 | 行号 | 改动类型 | 必要性 |
|---|---|---|---|
| `parallel_hook.py` | 266-267；160-167 | 互斥改为条件放行；加最终断言 | 必须 |
| `cp_tp_group_sharing_hook.py` | 79（在 107-109 行的 K3.1 分支里处理） | 只对 K3.1 去掉 `dcp-size > 1`；加 `dcp == attn_cp` 检查 | 必须 |
| `model_overrides/kimi_k3.py` | 240-241（在 214-268 行内） | 收紧成只放行方案 A；拒绝投机解码、PD 和非 flashmla_kv 后端 | 必须 |
| `model_overrides/deepseek_v2.py` | 104-141、168-172 | 预计不改，只核对；记录"只能单机"的限制 | 核对 |
| `hicache_hook.py` | 258-299（在第 261 行之后插入） | 新增：方案 A 拒绝 HiCache | 必须（第一期） |
| `cuda_graph_hook.py` | 第 143 行之前插入；规则在 213-216、258-276 | 显式指定 prefill graph 时报错 | 必须（第一期） |
| `mega_moe_hook.py` | 239-245 | token 预算加上每请求补齐的余量 | 用 MegaMoE 时必须 |

## 9. 建议的测试

在 `test/registered/unit/` 下新增参数校验的单元测试，覆盖以下情况：

**应当能启动（不报错）：**
- tp=attn_cp=dcp=8，interleave，flashmla_kv，K3.1。

**应当报错：**
- `dcp_size != attn_cp_size`；
- `--cp-strategy zigzag`；
- `--dcp-kv-layout page`；
- 方案 A 加 HiCache；
- 方案 A 加显式指定的 breakable 或 full prefill graph；
- 方案 A 加投机解码（DFLASH、DSPARK、EAGLE）；
- 方案 A 加 PD 分离；
- 方案 A 使用 trtllm 或 prims_ts 后端；
- GLM5-Next 和 Qwen4-Exp 开启 DCP 加 prefill CP（应继续被拒绝）。

**应当自动处理：**
- 没有显式指定 prefill graph 时，自动关闭 breakable graph。
