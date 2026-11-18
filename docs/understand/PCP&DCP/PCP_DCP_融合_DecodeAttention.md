# PCP + DCP 融合：Decode Attention 修改方案（计划二 §4.6，P2）

本文对应 `PCP_DCP_方案A_计划二.md` §4.6 和 `方案A计划问题汇总.md` 6.1–6.8。
前提仍是方案 A：`tp_size == attn_cp_size == dcp_size = N`，`attn_tp_size = 1`，切分方式只用 interleave，模型只支持 Kimi-K3.1。
行号对应当前分支 `yhz/test/v0.5.23-k31`，即 PrefillAttention P1 改动之后的代码。

---

## 1. 问题本身：方案 A 下 decode 的头数对不上

不开 PCP、只开 DCP 时（例如 TP8 + DCP8，attn_tp = 8），每张卡只有 `H/8` 个头的 Q 投影权重。decode 流程如下：

1. 本卡算出 `H/8` 个头的 Q；
2. 在 DCP 组内 all-gather Q，得到 `H` 个头（`forward_mla.py:884`）；
3. 用本卡持有的 1/8 KV 对 `H` 个头做局部注意力，得到 `H` 个头的部分结果和 LSE（`attn_mqa_for_dcp_decode`，`forward_mla.py:979`）；
4. a2a 或 ag_rs 按 LSE 合并，每张卡拿回自己那 `H/8` 个头（`forward_mla.py:1029-1115`）；
5. 用本卡的 `H/8` 列 o_proj 算出部分和，再在 TP 组内求和。

方案 A 下 attn_tp = 1，每张卡都持有**全部 `H` 个头**的 attention 权重，这是 PCP 的要求：prefill 时头复制、token 切分。如果原样走上面的流程，第 2 步会 gather 出 `8H` 个头，和 attention 后端的头数、合并后的 view、o_proj 的输入宽度全部对不上。

计划二给出三种处理方式：

| 方案 | 做法 | 主要代价 |
|---|---|---|
| D1 | 不 gather Q，每张卡直接用 `H` 个头算局部注意力，合并后再 all-gather 头 | 合并后多一次 all-gather；o_proj 全量计算 N 次 |
| D2 | 复用 `cp_decode_attn_tp`：decode 时把复制的 attention 权重按 CP rank 切成 `H/N`，后面完全走普通 DCP 流程 | 需要让 K3 接入这个机制（本文主体） |
| D3 | 只切 o_proj/g_proj，Q 用全量 `H` 头，不 gather | 需要改冻结文件 `model_runner.py`；q_b_proj 和 w_kc 的计算量是 N 倍 |

**本文推荐 D2**，原因如下：

- rank 拓扑阶段已经按 D2 改了头数公式（见 §2），D1 需要把它们改回去；
- D2 下 forward_mla 的 decode 主流程（Q gather、局部注意力、LSE 合并、view 回 `num_local_heads`）**一行都不用改**。切片后 `num_local_heads = H/N`，和普通 DCP 下的含义完全相同；
- 延迟求和、权重切片缓存、CUDA graph 下的指针稳定性，在 DeepSeek-V4 / GLM-5 上都已经验证过。

D3 本质上和 `--dcp-replicate-q-proj` 是同一个思路：每张卡算全量头的 Q，省掉 Q all-gather。方案 A 下每张卡本来就有全量 Q 权重，实现 D3 时把 qrep 权重直接指向本地权重即可，不需要 gather，可以作为 P3 的性能优化（§8）。

---

## 2. 已经完成的部分（rank 拓扑阶段，按 D2 实现）

| 位置 | 内容 |
|---|---|
| `runtime_context.py:514-525` | 新增 `decode_attn_head_shards()`：attn_cp > 1 且 attn_dcp > 1 时返回 `attn_tp * attn_cp`，否则返回 `attn_tp` |
| `layers/attention/dsa_backend.py:786` | `self.num_q_heads = num_heads // decode_attn_head_shards() * self.attn_dcp_size`，方案 A 下等于 `H` |
| `models/deepseek_v2.py:2266-2267` | `attn_mqa_for_dcp_decode` 的头数用同一公式，方案 A 下等于 `H` |
| `forward_mla.py:92` `_select_local_dcp_heads_for_autotune` | 加了越界断言 |

这些改动**默认 decode 时一定会切片**。如果 decode 没有切片，attention 后端只接受 `H` 个头，而 Q gather 之后是 `8H` 个头，会直接报 shape 错误。所以 §3.1 必须把 `--enable-cp-decode-attn-tp` 设为方案 A 的必选项。

---

## 3. `layers/cp/cp_decode_attn_tp.py` 及相关校验（汇总 6.2、6.6）

### 3.1 白名单：把 K3.1 加进去

**原因**：`--enable-cp-decode-attn-tp` 在 `arg_groups/model_hook.py:202-212` 被白名单拦住，白名单定义在 `cp_decode_attn_tp.py:29` 的 `CP_DECODE_ATTN_TP_SUPPORTED_ARCHS`，目前只有 DeepSeek-V4 和 GLM-5.x。K3.1 不在里面，开这个参数直接报错。

**怎么改**（`cp_decode_attn_tp.py:29` 的元组里加一项）：

```python
    # Kimi-K3.1 under prefill CP with DCP (attn_tp_size=1)
    "KimiK31ForConditionalGeneration",
```

**同时在参数校验里设为必选**。`arg_groups/parallel_hook.py` 的 `_validate_prefill_cp_with_dcp`（参数校验阶段新增）末尾追加以下校验：

```python
    if not cfg.enable_cp_decode_attn_tp:
        raise ValueError(
            "Prefill CP with DCP requires --enable-cp-decode-attn-tp: decode "
            "slices the replicated attention per CP rank before the DCP gather."
        )
```

不用自动打开的做法，原因是这个参数会连带禁掉在线权重更新（`configure_cp_decode_attn_tp`，`cp_decode_attn_tp.py:73-90`），应该让用户显式选择。

`can_defer_cp_decode_attn_tp()`（`cp_decode_attn_tp.py:42-62`）还要求以下条件：CUDA、关闭 layernorm SP、关闭 LoRA、关闭 TBO、关闭 quant communication。只要有一条不满足，K3 的 attention 层就会退回内联求和，但 K3 的内联求和在方案 A 下是坏的（见 §4.3）。因此这几条也要在 `_validate_prefill_cp_with_dcp` 里报错，或者放进 `arg_groups/model_overrides/kimi_k3.py` 的 `prefill_cp_dcp` 规则里，和已有的拒绝项放在一起：

```python
        "enable-layernorm-sp": cfg.enable_layernorm_sp,
        "enable-lora": cfg.enable_lora,
        "enable-two-batch-overlap": cfg.enable_two_batch_overlap,
        "enable-quant-communications": cfg.enable_quant_communications,
```

（字段名以 `fields/` 中的实际名字为准，写代码时核对一遍。）

### 3.2 生效范围：**不要收窄**（修正计划二）

计划二原本建议把 `set_decode_attn_tp`（`cp_decode_attn_tp.py:114-122`）的生效范围收窄到和 `is_dcp_mla_decode_phase`（`forward_mla.py:203-216`）一致。仔细读代码后，结论是**不应该收窄**，原因有两条：

1. **它必须和 layer boundary 的求和方式严格一致。** `set_decode_attn_tp` 的条件是 `not is_cp_active(fb) and not dsa_use_prefill_cp(fb)`，也就是"这个 batch 不走 CP 切分"。boundary 那边是 `layers/layer_boundary/construction.py:225-237` 的 `variant_for`：batch 按 CP 切分时返回 `CONTEXT_PARALLEL`，否则返回 `ORDINARY`。只有 `ORDINARY` 才会按声明的 `ordinary_tp_group` 做 TP 求和。两边判断的是同一件事，`deepseek_v2.py:2915-2920` 还有一个断言专门检查两者是否一致。如果只收窄权重切片这一侧，非 CP 的短 extend 就会出现"权重没切、每张卡都算出完整结果，boundary 却按 ORDINARY 再求和一次"，结果被放大 N 倍。
2. **不收窄在语义上也成立。** 不走 CP 的 extend（太短没被切分的 prefill，或者 target verify）切片之后，每张卡算 `H/N` 个头，KV 通过 DCP gather 得到全量（K3.1 的 DSA 后端自己 gather，`forward_mla.py:889-891` 的注释），等价于一次普通的 TP=N attention。后面的 o_proj 部分和交给 boundary 在 TP 上求和，结果正确。

**怎么改**：`set_decode_attn_tp` 不改。改为 §6.2 中的一项验证：确认非 CP extend 在 `H/N` 个头下，DSA prefill kernel 的 padding 正常。

### 3.3 `radix_attn` 只改了 `attn_mqa`：不需要改

`maybe_use_decode_attn_tp`（`cp_decode_attn_tp.py:306-309`）只把 `radix_attn.tp_q_head_num` 除以 N，调用方传入的是 `self.attn_mqa`（`deepseek_v2.py:2359`）。

- decode 和 verify 用的是 `attn_mqa_for_dcp_decode`（`forward_mla.py:979`），它的头数已经按 §2 的公式建成 `H`，正好等于"切片后 `H/N` 再 gather N 份"，不需要临时覆盖；
- 非 CP extend 用的是 `attn_mqa`，被覆盖成 `H/N`，和切片后的 Q 一致。

所以这一项**不用改**，在文档里记一下即可。

---

## 4. `models/kimi_k3.py`（汇总 6.1、6.3、6.4、6.5、6.8）

### 4.1 `_run_self_attn`（3312 行）：进入切片上下文（6.1）

**原因**：`DeepseekV2DecoderLayer` 在调用 self_attn 时套了一层 `with self.self_attn.maybe_use_decode_attn_tp(forward_batch):`（`deepseek_v2.py:2915`）。K3 的 decoder 层有自己的 `_run_self_attn`（`kimi_k3.py:3312-3345`），里面直接调用 `self.self_attn(...)`（3339 行），从不进入这个上下文。即使开了参数，decode 时权重也不会被切。K3 的 decoder 层同时承载 KDA 层和 MLA 层，KDA 层的 self_attn 没有这个方法。

**怎么改**：在 3339 行的 `result = self.self_attn(...)` 外面套上上下文，KDA 层用 `nullcontext()`（3312 行附近已经 import 了）：

```python
        decode_attn_tp = (
            self.self_attn.maybe_use_decode_attn_tp(forward_batch)
            if self.self_attn.defer_cp_decode_attn_tp
            else nullcontext()
        )
        with decode_attn_tp:
            if self.self_attn.defer_cp_decode_attn_tp:
                assert get_cp_decode_attn_tp_ctx().use_decode_attn_tp == (
                    self.attn_boundary.plan.variant_for(forward_batch)
                    is BatchVariant.ORDINARY
                ), "CP decode weight dispatch disagrees with the boundary sum"
            result = self.self_attn(
                hidden_states=hidden_states,
                positions=positions,
                forward_batch=forward_batch,
                zero_allocator=zero_allocator,
            )
```

- `defer_cp_decode_attn_tp` 在 `DeepseekV2AttentionMLA.__init__`（`deepseek_v2.py:2112-2117`）里计算，KimiK3MLAAttention 继承了这个属性。KDA 的 attention 类需要在 `__init__` 里补一行 `self.defer_cp_decode_attn_tp = False`，不能用 `getattr` 兜底（项目规则）。
- 断言照搬 `deepseek_v2.py:2916-2920`，用来保证 §3.2 说的两侧一致。
- 只在 `defer_cp_decode_attn_tp` 为 True 时进入上下文。为 False 时说明权重不切，必须保持原样。
- 3329-3337 行的 `prepare_qkv_latent` 放在上下文之外没有问题：q_a / kv_a 投影是复制的，不切。真正被切的 `q_b_proj` 在 self_attn 内部执行。

### 4.2 `g_proj` 要一起切（6.3）

**原因**：K3.1 的 MLA 有一个输出门 `g_proj`（`kimi_k3.py:2867-2876`），类型是 `ColumnParallelLinear(parallel_group="attn_tp")`。attn_tp = 1 时它是全量的，输出宽度 `H * v_head_dim`。在 `_gated_o_proj_forward`（2895-2900 行）里，门的输出和 attention 输出逐元素相乘，然后送进 o_proj。切片后 attention 输出只有 `H/N` 个头，门必须同步切成本 rank 的 `H/N` 个头，否则形状不匹配。门在 `_compute_output_gate`（2969-2977 行）中计算，调用点在 o_proj 内部，因此一定在切片上下文之内，包括走 alt stream 的情况。

基类的 `maybe_use_decode_attn_tp`（`deepseek_v2.py:2344-2370`）把要切的模块写死成 `[self.q_b_proj, self.o_proj]`，K3 没有办法追加模块。

**怎么改**：

1. `deepseek_v2.py:2357` 把模块列表抽成一个可覆盖的方法（本文件的改动见 §5.2）：

   ```python
   def _decode_attn_tp_modules(self) -> list:
       return [self.q_b_proj, self.o_proj]
   ```

   2357 行改为 `self._decode_attn_tp_modules(),`。

2. `kimi_k3.py` 的 `KimiK3MLAAttention`（2745 行起）中覆盖这个方法：

   ```python
   def _decode_attn_tp_modules(self) -> list:
       modules = super()._decode_attn_tp_modules()
       if self.use_output_gate:
           # The gate multiplies this rank's sliced heads before o_proj.
           modules.append(self.g_proj)
       return modules
   ```

`g_proj` 是 ColumnParallel，`_get_linear_attrs`（`cp_decode_attn_tp.py:211-238`）会沿 dim 0 切它的 weight 和 scale，并把 `output_size_per_partition` 除以 N。切出来的是第 `attn_cp_rank` 段，也就是本 rank 的那 `H/N` 个头，和 a2a 合并后拿回的头正好对应（§7 有顺序论证）。

`kimi_k3.py:2349` 是另一个类（非 MLA 的 dense attention），里面也有 `self.g_proj`，不涉及。

### 4.3 `_stage_facts` 补上延迟求和的声明（6.4、6.8）

**原因**：

- K3 的 MLA 以 `reduce_results=False` 构造（`kimi_k3.py:2790-2791`）。方案 A 加上 `--enable-cp-decode-attn-tp` 之后，`can_defer_cp_decode_attn_tp()` 为 True，于是 `defer_cp_decode_attn_tp` 为 True（`deepseek_v2.py:2112-2117`）。进入上下文后，`RowParallelLinear.use_decode_attn_tp = not defer = False`（`cp_decode_attn_tp.py:299`），o_proj 不做内联求和，这个求和必须由 stage boundary 来完成。
- DeepSeek-V2 在 `stage_facts`（`deepseek_v2.py:2792-2801`）里声明 `ordinary_tp_group=SumGroup.TP`，boundary 才知道 ORDINARY 批次要在 TP 组上求和。K3 的 `_stage_facts`（`kimi_k3.py:3084-3140`）只给 KDA 层声明了 `tp_group`（3131-3135 行），MLA 层没有任何声明。结果是每张卡只拿到自己 `H/N` 个头的 o_proj 部分和，后面直接当成完整结果用，**输出是错的，而且不会报错**。
- 退回内联求和也救不了：`kimi_k3.py:2862-2865` 在没开 all-reduce fusion 时设置了 `o_proj.use_dp_attention_reduce = True`，`linear.py:1810-1816` 于是在 **attn_tp 组**上 all_reduce。方案 A 下 attn_tp 组只有 1 张卡，等于没有求和。这也是 §3.1 要求 `can_defer` 的条件必须全部满足的原因。

**怎么改**：

1. `kimi_k3.py` 3131 行的 KDA 分支之后，加 MLA 层的声明：

   ```python
    if not config.is_kda_layer(layer_idx) and can_defer_cp_decode_attn_tp():
        # Decode slices the replicated MLA per CP rank (prefill CP with DCP);
        # the stage boundary sums the o_proj partials over TP.
        attn_ops["ordinary_tp_group"] = SumGroup.TP
   ```

   K3.1 的 MLA 层都有 q_lora_rank，并且都是 DSA，所以和 deepseek_v2 的条件等价。如果 K3（非 3.1）的配置也能走到这里，需要再加上 `is_deepseek_dsa(config)` 和 `q_lora_rank is not None` 两个条件，和 `deepseek_v2.py:2797-2799` 保持一致。

2. decoder 层 `__init__` 中，3281-3287 行 `_stage_facts(...)` 之后，照搬 `deepseek_v2.py:2836-2842` 的一致性检查和 o_proj 配置：

   ```python
        defer_decode_sum = self.self_attn.defer_cp_decode_attn_tp
        if (attn.ordinary_tp_group is not None) != defer_decode_sum:
            raise ValueError(
                "CP decode attention TP deferral disagrees with the stage declaration"
            )
        if defer_decode_sum:
            configure_cp_decode_attn_tp(self.self_attn.o_proj, defer_all_reduce=True)
   ```

   `configure_cp_decode_attn_tp` 会检查 `reduce_results=False`，并在 o_proj 上打一个"禁止在线更新权重"的标记（`cp_decode_attn_tp.py:73-90`）。

3. 需要确认的一点：`_ffn_input_fusions`（3290 行起）的 all-reduce fusion 分支用的是 `SumGroup.ATTN_TP`。方案 A 下 attn_tp = 1，`all_reduce_fusion` 已经关掉（2840-2843 行），走不到这个分支。写代码时在 decoder 层 init 里加断言 `not (defer_decode_sum and self.all_reduce_fusion)`，防止将来有人打开 fusion。

### 4.4 `post_load_weights`（4101-4102 行）中 `_prepare_dcp_v_weights` 的调用（6.5）

**原因**：`_prepare_dcp_v_weights`（`forward_mla.py:193-201`）把本卡的 `w_vc` 在 DCP 组内 all-gather，缓存为 `_dcp_v_weights`，供 project-before-merge 使用（`forward_mla.py:1067-1078`）。它有两个调用点：

| 调用点 | 何时执行 | 方案 A + D2 下 |
|---|---|---|
| `forward_mla.py:1067-1074` | 第一次 decode 时惰性创建（不在 graph capture 中） | 在切片上下文内执行：`w_vc` 已切成 `H/N` 头，gather 后得到 `H` 头，**正确** |
| `kimi_k3.py:4101-4102` | 权重重新加载时，已有缓存才刷新 | 在切片上下文外执行：`w_vc` 是全量 `H` 头，gather 后得到 `NH` 头，`cached.copy_` 形状不匹配，**报错** |

第二个调用点只在**权重重新加载**时执行，而 `--enable-cp-decode-attn-tp` 本来就禁止在线权重更新：`configure_cp_decode_attn_tp` 在 o_proj 上设置了 `_derived_weight_cache_error`，切片缓存按 `id(obj)` 存，不会失效（`cp_decode_attn_tp.py:180-196`）。所以正常流程走不到这里，计划二说的"必须挪位置"不成立。

**怎么改**：不挪位置，只加一个保护，让这种情况明确报错，而不是表现为一次莫名其妙的 shape 错误。4101 行改为：

```python
            if getattr(self_attn, "_dcp_v_weights", None) is not None:
                # Decode slices w_vc per CP rank under prefill CP with DCP;
                # a reload here would gather the unsliced heads.
                assert not self_attn.defer_cp_decode_attn_tp, (
                    "weight reload is not supported with CP decode attention TP"
                )
                self_attn._prepare_dcp_v_weights(get_parallel().dcp_group)
```

（4101 行原有的 `getattr` 不是本次新增，不顺手改。）

还有一个隐含条件需要在测试中确认：第一次 decode 必须发生在 CUDA graph capture **之前**，也就是 warmup 是 eager 执行的。如果第一次 decode 就在 capture 中，`weights` 为 None，会退回不做 V 投影的 a2a，结果仍然正确，只是性能差一些。

### 4.5 头数相关（6.4）

`KimiK3MLAAttention` 没有自己算 decode 头数，全部继承自 deepseek_v2，已经在 §2 中改好，这里**不用改**。

---

## 5. `models/deepseek_v2.py`（汇总 6.2、6.4）

### 5.1 `attn_mqa_for_dcp_decode` 的头数：已完成

`deepseek_v2.py:2266-2267` 已改成 `num_heads // decode_attn_head_shards() * self.dcp_size`。方案 A 下等于 `H/N * N = H`，等于切片后 Q gather 回来的头数。普通 DCP 下 `decode_attn_head_shards() == attn_tp`，和原来一样。

### 5.2 `maybe_use_decode_attn_tp`（2344-2370 行）对 K3 子类生效

**原因**：方法本身是继承下来的，问题有两个：K3 没有调用它（§4.1），模块列表不能扩展（§4.2）。切片的内容如下：

| 被切对象 | 维度 | 切后 |
|---|---|---|
| `q_b_proj`（Column） | dim 0 | 本 rank 的 `H/N` 头 |
| `o_proj`（Row） | dim 1 | 本 rank 的 `H/N` 头 × v_head_dim |
| `w_kc` / `w_vc` / `w_scale_k` / `w_scale_v` | dim 0（头） | `H/N` 头 |
| `num_local_heads` | | `H/N` |

这些和 K3.1 的权重布局一致：`w_kc`、`w_vc` 在 `kimi_k3.py:4096-4100` 由 kv_b_proj 拆分得到，第 0 维是头。

**怎么改**：只抽出 `_decode_attn_tp_modules()`（§4.2），2357 行改为调用它，其他不动。

### 5.3 不需要改的地方

- `defer_cp_decode_attn_tp`（2112-2117 行）的条件对 K3.1 自然成立，不改；
- `stage_facts`（2792-2801 行）是 deepseek_v2 自己的声明，K3 用的是自己的 `_stage_facts`，改动在 §4.3。

---

## 6. `layers/attention/dsa_backend.py`（汇总 6.4）

### 6.1 `num_q_heads`（786 行）：已完成

方案 A 下等于 `H`，供 decode 的 flashmla_kv / trtllm 等后端使用。K3.1 是 sharded DCP，后端内部 `dcp_size` 被设为 1（787-796 行），所以 798 行的 `num_dcp_q_heads = padded(num_q_heads * 1)` 也是 `H`，不会被重复放大。

### 6.2 `local_num_q_heads`（783 行）：需要验证

**原因**：`local_num_q_heads = H // attn_tp_size`，方案 A 下等于 `H`。它的用途有两个：

- 844 行：一个 prefill 快路径的准入条件 `1 <= local_num_q_heads <= 32`；
- 893-895 行：`flashmla_kv_num_q_heads_prefill = padded(local_num_q_heads)`，用于 prefill 的 Q padding 缓冲区。

CP 切分的 prefill 用全部 `H` 个头，和这个值一致。但根据 §3.2，**不走 CP 的短 extend 会被切成 `H/N` 头**，这时 Q 实际只有 `H/N` 头，而 padding 目标仍按 `H` 计算。

**怎么改**：先确认 prefill kernel 怎么处理"实际头数小于 padding 头数"：

- 如果是"补零到目标头数，输出再截回实际头数"，那么 `H/N` 也能正常工作，只是多算了一些 pad 头，这一项不用改；
- 如果断言两者相等或按 `local_num_q_heads` 做 view，就要在非 CP extend 时按实际 `q.shape[1]` 计算 padding，或者新增一个 `decode_tp_local_num_q_heads = H // decode_attn_head_shards()`，在切片生效时使用。

这一点只能在 GPU 上跑一个"短 prompt，不触发 CP 切分"的请求来确认，列进 §9 的 GPU 验证项。

---

## 7. `forward_mla.py`（汇总 6.4、6.7）

### 7.1 D2 下：主流程不改，只确认头的顺序一致

decode 时的数据流（N = 8，H 为总头数）：

| 步骤 | 位置 | 形状 | 说明 |
|---|---|---|---|
| q_b_proj（已切） | | `[B, H/8, ...]` | 本 rank 第 `attn_cp_rank` 段的头 |
| Q all-gather | 884 行 | `[B, H, ...]` | 第 `dcp_rank * H/8 + j` 个头来自 dcp_rank 号卡 |
| 局部注意力 | 979 行 | `[B, H, lora]` + LSE | 本卡 1/8 KV |
| view | 1030-1034 行 | `num_local_heads * dcp_size = H` | 和上面一致 |
| a2a / push-combine 合并 | 1084、1100 行 | `[B, H/8, ...]` | 每张卡拿回第 `dcp_rank` 段 |
| view 回 | 1092、1119 行 | `num_local_heads = H/8` | |
| g_proj、o_proj（已切） | | 第 `attn_cp_rank` 段 | 部分和交给 boundary 在 TP 上求和 |

只有当**切片用的 rank（`attn_cp_rank`）和 DCP 通信用的 rank（`dcp_rank`）是同一个编号**时，头的顺序才会对上。方案 A 下 `attn_tp = 1`，所以 `attn_cp_rank = tp_rank % N`，`dcp_rank = tp_rank % N`，两者相等。rank 拓扑阶段已经在 `parallel_state.py` 中断言 DCP 组和 CP 组的成员相同。还缺一条"组内编号相同"的断言，可以加在 `runtime_context.py` 的 `_validate_parallel` 方案 A 分支里：

```python
            # Decode slices heads by CP rank and the DCP merge returns them by
            # DCP rank; the two must name the same shard.
            assert self.attn_cp_rank == self.attn_dcp_rank
```

（编号相同是从公式推出来的，属于"推导性质"，可以在 `test_runtime_context.py` 的 `TestTheTopologyIdentities` 中加一个用例。）

forward_mla 中**不需要改的行**：884-887（Q gather）、1029-1034（view）、1036-1041（autotune narrow，已加断言）、1067-1078（V 权重惰性 gather，见 §4.4）、1084-1115（合并）、1116-1119（view 回）。

### 7.2 `--dcp-replicate-q-proj` 必须关闭（新发现）

**原因**：K3.1 在 a2a / fi_a2a 后端下会默认打开 `dcp_replicate_q_proj`（`arg_groups/model_overrides/kimi_k3.py:261`、`285-291`）。打开后：

- `model_runner.py:1215-1242` 把每层的 `q_b_proj`、`w_kc`、`w_vc` 在 DCP 组内 all-gather，得到"全量头"的副本 `*_qrep`。方案 A 下本地已经是全量 `H` 头，gather 后是 `NH` 头，形状错误，显存也多占 N 倍；
- 1229-1231 行还会把本地 `q_b_proj.weight` 改成指向副本中第 r 段，D2 切片时就会在这段已经被改过的权重上再切一次；
- decode 时 `q_replicate_active` 为 True（`forward_mla.py:494-498`），跳过 Q gather，直接用 `*_qrep` 计算，和切片后的 `num_local_heads` 对不上。

`model_runner.py` 是冻结文件，不应该在这里加分支。

**怎么改**：在 `arg_groups/model_overrides/kimi_k3.py` 的方案 A 规则里处理：

1. `prefill_cp_dcp` 的拒绝项中加 `"dcp-replicate-q-proj": cfg.dcp_replicate_q_proj is True`；
2. 261 行改为方案 A 下不应用这个默认值：

   ```python
    overrides = (
        {"dcp_replicate_q_proj": False}
        if cfg.enable_prefill_cp
        else _dcp_replicate_q_proj_default(cfg, "Kimi-K3.1")
    )
   ```

   方案 A 下 PCP 和 DCP 同时打开，用 `enable_prefill_cp` 区分即可。写代码时用实际字段名，和参数校验阶段的 `prefill_cp_dcp` 规则保持一致。

   这一项放在参数校验文件里，单元测试也加在 `test_prefill_cp_with_dcp_args.py` 中：断言方案 A + fi_a2a 时 `dcp_replicate_q_proj` 被解析为 False，显式传 True 时报错。

### 7.3 D1 下需要改的地方（备选，不推荐）

如果 §9 的 D2 实验失败，改走 D1，需要做以下改动：

1. 回退 §2 的三处头数公式，方案 A 下 `num_q_heads` 和 `attn_mqa_for_dcp_decode` 改为 `H`（不乘 dcp_size）；
2. `forward_mla.py:882-887`：方案 A 下跳过 Q gather，条件写成 `if not q_replicate_active and not prefill_cp_with_dcp:`；
3. `forward_mla.py:1029-1115`：view 改为 `num_local_heads`（即 `H`），合并后每张卡得到 `H/N` 头，再在 DCP 组内 all-gather 回 `H` 头，然后走全量 o_proj。o_proj 的输出是完整结果，boundary **不能**再求和，也就是 §4.3 的声明不加，并且要把 `use_dp_attention_reduce` 的路径确认为空操作。

D1 每层 decode 多一次 `[B, H, v]` 的 all-gather，o_proj 在 N 张卡上重复计算，通信和计算都比 D2 多。

---

## 8. P3 可选优化：D3（复用本地全量 Q 权重）

方案 A 下每张卡都有全量 Q 权重，"每张卡算全量头的 Q、不做 Q all-gather"不需要任何 gather。做法是在切片上下文中**不切** `q_b_proj` 和 `w_kc`，只切 `w_vc`、`o_proj`、`g_proj`，同时让 forward_mla 走 `q_replicate_active` 分支，并把 `*_qrep` 指向本地权重。

- 优点：每层 decode 省掉一次 Q all-gather（`[B, H/N, 576]` × N）；
- 代价：q_b_proj 和 w_kc 的计算量是 N 倍；需要在 `model_runner.py:1215-1242` 增加"本地已是全量头，直接引用"的分支，这是冻结文件，要按 `large-class-style` 的规则处理。

decode batch 较小时，计算增加的部分可以忽略，是否值得做由 P3 的 profile 决定。

---

## 9. 测试与验证

### 9.1 CPU 单元测试（可在 `test/registered/unit/` 中加）

| 用例 | 文件 | 防止的问题 |
|---|---|---|
| 方案 A 下 `attn_cp_rank == attn_dcp_rank` | `test_runtime_context.py` `TestTheTopologyIdentities` | 有人改 rank 公式后头的顺序错位 |
| 方案 A + fi_a2a 解析出 `dcp_replicate_q_proj=False`，显式 True 报错 | `test_prefill_cp_with_dcp_args.py` | K3.1 默认值又把 replicate 打开 |
| 方案 A 不带 `--enable-cp-decode-attn-tp` 报错 | `test_prefill_cp_with_dcp_args.py` | 漏开参数导致 decode 头数不匹配 |
| K3.1 在 `CP_DECODE_ATTN_TP_SUPPORTED_ARCHS` 中 | 同上，通过参数解析通过来间接验证 | 白名单遗漏 |

`_stage_facts` 的声明和 decoder 层 init 中的一致性检查，需要构造 config 和 layer，CPU 上不方便测；decoder 层 init 自带 ValueError，GPU 冒烟测试时会直接暴露。

运行命令（和前几份文档一样，需要装有 torch 的环境）：

```bash
cd /Users/yhz/sglang

# 参数校验：整个文件（包含本阶段新增的用例）
python -m pytest test/registered/unit/server_args/test_prefill_cp_with_dcp_args.py -q

# 只跑本阶段新增的用例
python -m pytest test/registered/unit/server_args/test_prefill_cp_with_dcp_args.py -q -k \
  "cp_decode_attn_tp or replicated_q_proj or inline_decode_sum"

# rank 编号一致性（attn_cp_rank == attn_dcp_rank），rank 拓扑阶段已有
python -m pytest test/registered/unit/test_runtime_context.py -q -k TopologyIdentities
```

本阶段新增的用例（都在 `test_prefill_cp_with_dcp_args.py` 中）：

| 用例 | 所在类 | 检查内容 |
|---|---|---|
| `test_cp_decode_attn_tp_is_required` | `TestDecodeContextParallelismGate` | 不带 `--enable-cp-decode-attn-tp` 报错 |
| `test_kimi_k31_may_enable_cp_decode_attn_tp` | `TestDecodeContextParallelismGate` | K3.1 在 `CP_DECODE_ATTN_TP_SUPPORTED_ARCHS` 中 |
| `test_replicated_q_proj_stays_off` | `TestKimiK31PrefillCpWithDcpOverrides` | 方案 A + fi_a2a 解析出 `dcp_replicate_q_proj=False`，显式 True 报错 |
| `test_plain_dcp_keeps_the_replicated_q_proj_default` | `TestKimiK31PrefillCpWithDcpOverrides` | 不开 PCP 的普通 DCP 仍默认打开 replicate |
| `test_features_that_force_the_inline_decode_sum_are_rejected` | `TestKimiK31PrefillCpWithDcpOverrides` | 拒绝 layernorm SP、LoRA（含 `lora_paths`）、TBO、quant communication |

### 9.2 GPU 验证（P2 的 D2 可行性实验）

1. **2 卡单层对比**：Kimi-K3.1 截成 1 个 MLA 层加 1 个 KDA 层，分别用 TP2 + DCP2（不开 PCP）和方案 A（TP2 = PCP2 = DCP2，开 `--enable-cp-decode-attn-tp`）跑同一批 decode，比较每层输出，误差应在 bf16 噪声范围内；
2. **短 prompt 不触发 CP 切分**：验证 §6.2 的 `local_num_q_heads` padding 问题；
3. **CUDA graph 打开和关闭各跑一次**：验证切片缓存的指针稳定，以及 `_dcp_v_weights` 在 capture 前已经创建（§4.4）；
4. **8 卡整模型精度**：按 `prefer-sgl-eval` 规则用 sgl-eval 跑 gsm8k，和 TP8 + DCP8 的基线比较，采样参数使用 K3.1 的推荐设置。

---

## 10. 修改清单汇总

| 文件 | 位置 | 改动 | 状态 |
|---|---|---|---|
| `layers/cp/cp_decode_attn_tp.py` | 29 行 | 白名单加 `KimiK31ForConditionalGeneration` | 已完成 |
| `layers/cp/cp_decode_attn_tp.py` | 114-122 行 | 生效范围**不收窄**（修正计划二） | 不改 |
| `arg_groups/parallel_hook.py` | `_validate_prefill_cp_with_dcp` | 必须开 `--enable-cp-decode-attn-tp`（layernorm SP、LoRA、TBO、quant comm 的拒绝项放在 `model_overrides/kimi_k3.py`） | 已完成 |
| `arg_groups/model_overrides/kimi_k3.py` | 261、285-291 行和 `prefill_cp_dcp` 规则 | 方案 A 下 `dcp_replicate_q_proj=False`，显式 True 报错 | 已完成 |
| `models/kimi_k3.py` | `_run_self_attn` 3339 行 | 套 `maybe_use_decode_attn_tp` 和一致性断言；KDA 类补 `defer_cp_decode_attn_tp = False` | 已完成 |
| `models/kimi_k3.py` | `KimiK3MLAAttention` | 覆盖 `_decode_attn_tp_modules`，加入 `g_proj` | 已完成 |
| `models/kimi_k3.py` | `_stage_facts` 3131 行后 | MLA 层声明 `ordinary_tp_group=SumGroup.TP` | 已完成 |
| `models/kimi_k3.py` | decoder 层 init 3281 行后 | 声明一致性检查和 `configure_cp_decode_attn_tp` | 已完成 |
| `models/kimi_k3.py` | 4101-4102 行 | 不挪位置，加"不支持重新加载"的断言 | 已完成 |
| `models/deepseek_v2.py` | 2357 行 | 模块列表抽成 `_decode_attn_tp_modules()` | 已完成 |
| `models/deepseek_v2.py` | 2266-2267 行 | `attn_mqa_for_dcp_decode` 头数 | 已完成 |
| `layers/attention/dsa_backend.py` | 786 行 | `num_q_heads` | 已完成 |
| `layers/attention/dsa_backend.py` | 783、844、893-895 行 | `local_num_q_heads` 在非 CP extend 下的 padding | GPU 验证后再定 |
| `runtime_context.py` | `_validate_parallel` 方案 A 分支 | 断言 `attn_cp_rank == attn_dcp_rank` | 已完成（rank 拓扑阶段已有，344-349 行，测试见 `test_runtime_context.py` `test_prefill_cp_with_dcp_refuses_a_dcp_rank_off_the_cp_rank`） |
| `forward_mla.py` | 884、1029-1119 行 | D2 下不改 | 不改 |
| `forward_mla.py` | 92 行 | autotune narrow 断言 | 已完成 |

和计划二不同的结论：

1. `set_decode_attn_tp` 不收窄，否则和 boundary 的求和方式不一致；
2. `_prepare_dcp_v_weights` 不挪位置，它的正常调用点本来就在切片上下文内；
3. 新增两项：必须开 `--enable-cp-decode-attn-tp`，必须关 `--dcp-replicate-q-proj`。
