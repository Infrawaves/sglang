# PCP + DCP 融合：显存核算（计划二 §4.7，P1）

本文对应 `PCP_DCP_方案A_计划二.md` §4.7 和 `方案A计划问题汇总.md` 9.1–9.4。
前提仍是方案 A：`tp_size == attn_cp_size == dcp_size = N`，`attn_tp_size = 1`，切分方式只用 interleave，模型只支持 Kimi-K3.1。
行号对应当前分支 `yhz/test/v0.5.23-k31`，即 DecodeAttention D2 改动之后的代码。

---

## 1. 结论

计划二 §4.7 的两行都是"确认"项。逐项读代码后，结论是 **P1 不需要改代码**：

| 文件 | 计划要确认的点 | 结论 |
|---|---|---|
| `model_executor/pool_configurator.py` | 452-462 行按 `attn_dcp_size` 计算每 token 字节数，indexer 在 `dsa_shards_index_k()` 为真时不乘 dcp | 成立，不改（§3） |
| `mem_cache/kv_cache_configurator.py` | 走 `_build_hybrid_linear_kv_pool` | 成立，不改（§4.1） |
| 同上 | `dcp_enabled` 为真，不会选成逐 token 分配器（2937 行） | 成立，不改（§4.2） |
| 同上 | KDA 状态池按整个 TP 组的头切分（每卡 12 个头） | 成立，不改（§4.3） |

另外发现一项计划没有提到的显存开销：D2 的权重切片缓存在 KV 池分好之后才分配，没有计入核算（§5）。它属于 P2，量级小，建议先记录、不改代码。

---

## 2. 为什么大部分不用改：方案 A 和 TP8 + DCP8 在显存核算上的差别

显存核算的代码已经在普通 DCP（TP8 + DCP8，attn_tp = 8）下验证过。方案 A 和它相比，只有下面三个参数不同。所以只需要逐个检查：显存核算里有没有哪一项依赖这三个参数。

| 参数 | TP8 + DCP8 | 方案 A | 显存核算里的使用位置 |
|---|---|---|---|
| `attn_tp_size` | 8 | 1 | `pool_configurator.py:403`，只用于 MHA 分支；MLA 分支不用（§3.2） |
| `enable_prefill_cp` / `attn_cp_size` | False / 1 | True / 8 | 只通过 GLM 的层切分进入核算，方案 A 下不会开启（§3.3） |
| `enable_cp_tp_group_sharing` | False | True | 决定 KDA 状态按哪个组切头（§4.3） |

`attn_dcp_size`、`dcp_enabled`、`page_size` 在两种部署下相同，用到它们的地方（每 token 字节数、分配器的 id 空间、页大小）结果也相同。

---

## 3. `model_executor/pool_configurator.py`：不改

### 3.1 K3.1 用的是 `DefaultPoolConfigurator`

`create_memory_pool_configurator`（1818-1828 行）只在 `is_hybrid_swa` 为真时选 SWA 的 configurator。K3.1 的 KDA 层是线性注意力，不是滑动窗口，`is_hybrid_swa` 为假（`configs/model_config.py:1318-1322`），所以走 `DefaultPoolConfigurator`（248 行起）。

它在 258-264 行用 `mambaish.full_attention_layer_ids` 统计层数，只算 MLA 层，KDA 层不计入 KV 的每 token 字节数。KDA 状态单独按请求数计算（§4.3）。

### 3.2 MLA 每 token 字节数和 TP 无关

`_compute_cell_size`（378 行起）的 MLA 分支（407 行起）：

```
cell_size = calculate_mla_kv_cache_dim(...) * kv_size * effective_num_layers
```

MLA 的 latent KV 只有一个"头"，所有 rank 存的内容相同，所以这里没有除以 `tp_size`。403 行读的 `attn_tp_size` 只在 MHA 分支（549、560 行的 `get_num_kv_heads`）用到。因此方案 A 把 `attn_tp_size` 从 8 改成 1，不影响 K3.1 的结果。

这里算的是**每个物理行**的字节数。DCP 下每张卡只存 1/N 的 token，分配器把 id 空间放大 N 倍（§4.2），所以每 token 字节数本身不需要乘或除 `dcp_size`。这一点和 TP8 + DCP8 完全相同。

### 3.3 GLM 的层切分不会生效

396-400 行的 `effective_num_layers` 会经过 `get_glm_dsa_layer_split_effective_num_layers`（`layers/cp/utils.py:198-213`）。这是代码里唯一读 `attn_cp_size` 的显存项：开启时每个 CP rank 只存 `ceil(L / N) + 1` 层。

它的开关是 `--enable-dsa-cache-layer-split`（`layers/cp/utils.py:167-180`）。方案 A 下这个参数一定是关的：

- `arg_groups/model_hook.py:376-390` 要求它只能用在 PD prefill 节点上，而方案 A 第一期拒绝 PD；
- `arg_groups/cp_tp_group_sharing_hook.py:78` 在 CP-TP group sharing 下直接拒绝它，K3.1 + 方案 A 一定会开启 sharing（§4.3）。

所以 `effective_num_layers == num_layers`，和 TP8 + DCP8 一样。

### 3.4 452-462 行：indexer 不乘 dcp，前提成立

```python
            if is_deepseek_dsa(model_config.hf_config):
                ...
                indexer_cell_size = self._compute_dsa_indexer_cell_size(...)
                # Index-K spans the widened slot space unless K3.1 shards it
                if not kvc.is_draft_worker and not dsa_shards_index_k():
                    indexer_cell_size *= dcp_size
                cell_size += indexer_cell_size
```

`dsa_shards_index_k()`（`layers/dcp/layout.py:35-36`）返回 `_dsa_shards_index_k and attn_dcp_size > 1`。要让它为真，需要满足两个条件：

1. **标记已经设置。** `KimiK3MLAAttention.__init__` 在 `attn_dcp_size > 1` 时调用 `set_dsa_shards_index_k()`（`models/kimi_k3.py:2786-2790`）。这个条件只看 `attn_dcp_size`，不看 attn_tp 和 CP，方案 A 下 `attn_dcp_size = 8`，一定会设置。
2. **设置发生在核算之前。** 模型在 `model_runner.py:757` 的 `load_model()` 中构造，KV 核算在 1044 行的 `kv_cache_configurator.configure(...)` 中进行，后者再调用 `create_memory_pool_configurator`（`kv_cache_configurator.py:3225`）。顺序是先构造模型，再核算。

核算结果和实际分配一致：`mem_cache/memory_pool.py:5390-5396` 在 `dsa_shards_index_k()` 为真时把 index-K 缓冲区大小设为 `size`（物理行数），不放大 N 倍。核算和分配读的是同一个函数，不会出现一边乘 N、另一边不乘的情况。

**结论：不改。**

### 3.5 和投机解码有关的分支：方案 A 下走不到

290-376 行（EAGLE / DFLASH 草稿模型的显存）方案 A 下都不会执行，因为 `model_overrides/kimi_k3.py` 的 `prefill_cp_dcp` 规则拒绝了投机解码。

---

## 4. `mem_cache/kv_cache_configurator.py`：不改

### 4.1 K3.1 走 `_build_hybrid_linear_kv_pool`（汇总 9.2）

`_build_token_to_kv_pool`（1738 行起）的分支顺序：

| 行 | 条件 | K3.1 |
|---|---|---|
| 1818 | `use_mla_backend and is_hybrid_swa` | 不满足（不是 SWA） |
| 1826 | `use_mla_backend and is_dsa_model and not self.mambaish_config` | 不满足（有 KDA，`mambaish_config` 非空） |
| 1860-1862 | `self.mambaish_config` | **满足**，调用 `_build_hybrid_linear_kv_pool` |

所以 `_build_dsa_kv_pool`（2289 行）里的 GLM CP 层切分和 `dcp_replicated` 逻辑对 K3.1 都不起作用。

`_build_hybrid_linear_kv_pool`（2599-2728 行）中和并行有关的参数只有两个：

- `page_size=self.pool_page_size`（2709 行）。`pool_page_size = page_size * loc_space_scale`（685-686 行），主模型的 `loc_space_scale` 恒为 1（669-682 行，只有草稿模型才大于 1）；
- `head_num=get_num_kv_heads(attn_tp_size, attn_dcp_size)`（2712-2714 行）。MLA 池按 `kv_lora_rank + qk_rope_head_dim` 分配，不用这个头数。

两者在方案 A 和 TP8 + DCP8 下结果相同。**结论：不改。**

### 4.2 不会选成逐 token 分配器（汇总 9.4）

2936-2945 行：

```python
                    elif (
                        get_schedule().page_size == 1 and not get_parallel().dcp_enabled
                    ):
                        token_to_kv_pool_allocator = TokenToKVPoolAllocator(...)
                    else:
                        token_to_kv_pool_allocator = paged_allocator_cls(
                            sizes.max_total_num_tokens * get_parallel().attn_dcp_size,
                            page_size=get_schedule().page_size
                            * get_parallel().attn_dcp_size,
                            ...
```

`dcp_enabled` 在 `runtime_context.py:169` 定义为 `cfg.dcp_size > 1`，方案 A 下 `dcp_size = 8`，所以一定为真，走 2946-2959 行的 paged 分配器：id 空间是物理行数的 N 倍，页大小是 `page_size * N`。这和 TP8 + DCP8 相同。

汇总 9.4 提到的 `page_size % dcp == 0` 约束来自 `mem_cache/pool_host/base.py:242`，只在 HiCache 的 host 池中生效。方案 A 第一期拒绝 HiCache，这条约束不影响 P1。

**结论：不改。**

### 4.3 KDA 状态池按 TP 组切头（汇总 9.3）

KDA 状态池每个请求的大小来自 `config.mamba2_cache_params.mamba_cache_per_req`（`_handle_max_mamba_cache`，3411 行起，3438 行读取）。`mamba2_cache_params` 定义在 `configs/kimi_linear.py:219-228`：

```python
        shape = KimiLinearStateShape.create(
            tp_world_size=linear_attn_tp_size(),
            num_heads=self.linear_attn_config["num_heads"],
            ...
```

`linear_attn_tp_size()`（`runtime_context.py:504-506`）返回 `linear_attn_parallel_group()` 的宽度，后者（497-501 行）在 `enable_cp_tp_group_sharing` 为真时返回 `"tp"`，否则返回 `"attn_tp"`。

方案 A 下 `enable_cp_tp_group_sharing` 一定为真：`arg_groups/cp_tp_group_sharing_hook.py` 的 `resolve_cp_tp_group_sharing` 在 K3.1 开启 prefill CP 且 `attn_cp_size == tp_size` 时，无条件声明 `enable_cp_tp_group_sharing=True`（文件末尾的 `declare_resolution`），没有关闭它的参数。同一个 hook 还检查了 `num_heads % tp_size == 0`。

所以 `linear_attn_tp_size() = tp_size = 8`，每卡 `96 / 8 = 12` 个头。KDA 层本身也用同一个函数切头（`models/kimi_k3.py:2076-2078` 的 `head_group`、`head_tp_size`、`head_tp_rank`），状态池的大小和层实际使用的形状一致。

`mamba2_cache_params` 是 property，每次调用时按当前已发布的并行配置计算。核算发生在并行配置发布之后，所以读到的是方案 A 的值，而不是参数解析阶段的默认值。

**结论：不改。** 如果 `attn_tp_size = 1` 被错误地用在这里，每卡会按 96 个头分配，状态池会大 8 倍，而且会和 KDA 层的形状对不上。当前代码不会出现这种情况。

### 4.4 其他会影响核算、但方案 A 下不生效的项

| 项 | 位置 | 方案 A 下 |
|---|---|---|
| KV cache sharding 的 scratch | `pool_configurator.py:282-285` `compute_page_shard_scratch_bytes` | `--enable-kv-cache-sharding` 下 DSA 要求 trtllm + Mooncake（`arg_groups/kv_shard_hook.py:33-62`），方案 A 第一期只用 flashmla_kv 且拒绝 PD，不会开启 |
| HiSparse | `pool_configurator.py:303-310`、`kv_cache_configurator.py:2660-2673` | 方案 A 拒绝 HiSparse |
| 投机解码的草稿池 | `kv_cache_configurator.py:641-683` | 方案 A 拒绝投机解码 |

---

## 5. 计划未提到的一项：D2 权重切片缓存没有计入核算（P2）

### 5.1 现象

D2 的 `CpDecodeAttnTpContext._activate`（`layers/cp/cp_decode_attn_tp.py:170-199`）在**第一次** decode 时创建切片，并缓存到 `_slice_cache`，之后一直保留，以保证 CUDA graph 的指针稳定。第一次 decode 发生在 warmup 或 CUDA graph capture 时，晚于 KV 池分配（`model_runner.py:1044`）。所以这部分显存不在 KV 核算中，只能挤占 `1 - mem_fraction_static` 的余量。

### 5.2 哪些切片会占新显存

`_slice`（125-155 行）的行为：

| 切片 | 维度 | 是否新分配 | 大小（每个 MLA 层） |
|---|---|---|---|
| `q_b_proj.weight`、`g_proj.weight`（Column） | dim 0 | 否，`narrow` 返回视图 | 0 |
| `w_kc`、`w_vc`、`w_scale_k`、`w_scale_v` | dim 0 | 否，视图 | 0 |
| `o_proj.weight`（Row） | dim 1 | **是**，155 行 `.contiguous()` | 原 o_proj 的 `1/N`，即 `hidden_size * H * v_head_dim / N * 元素字节数` |
| `o_proj` 的 block scale | dim 1 | 是 | 很小 |
| int32 打包的 UE8M0 scale（任意维度） | 0 / 1 | 是，140-154 行重新排布 | 很小 |

总量约为"所有 MLA 层的 o_proj 权重之和 / N"。

另一项 `_dcp_v_weights`（`forward_mla.py:1067-1074`，第一次 decode 时惰性 gather `w_vc`）在方案 A 下 gather 回 `H` 个头，大小和 TP8 + DCP8 下相同，不是新增开销。

### 5.3 建议

- **不改代码。** DeepSeek-V4 和 GLM-5 用的是同一套机制，同样没有计入核算，量级相对 KV 池很小。
- 在 §6 的 GPU 验证中记录 CUDA graph capture 前后的空闲显存，确认这部分被 `mem_fraction_static` 的余量吸收。如果实测接近余量上限，再考虑在 `DefaultPoolConfigurator` 的 `_fixed_overhead_bytes`（282-285 行）中预留这部分，或者在核算前预先创建切片。

### 5.4 预期内的 KV 容量下降（不是 bug）

方案 A 下 attn_tp = 1，每张卡持有全部 attention 权重（q_a、q_b、kv_a、kv_b、o_proj、g_proj），这些权重是 TP8 下的 N 倍。这部分在 `load_model()` 之后、核算之前由 `_profile_available_bytes`（`kv_cache_configurator.py:3010`）实测扣除，自动计入。结果是同样的 `mem_fraction_static` 下，方案 A 的 `max_total_num_tokens` 会比 TP8 + DCP8 少。

TP8 + DCP8 在 fi_a2a 后端下默认开启 `dcp_replicate_q_proj`，每卡额外持有一份全量的 `q_b_proj` 副本。方案 A 关掉了它（DecodeAttention §7.2），所以 `q_b_proj` 这一项两边大致相当，差别主要来自其他 attention 权重。

---

## 6. 验证

P1 不改代码，只需要在 GPU 上确认核算结果。

1. **启动日志对比。** 方案 A 和 TP8 + DCP8 用同一个 `--mem-fraction-static` 启动，比较日志中的 `max_total_num_tokens` 和 KDA 状态池大小（`max_mamba_cache_size`）：
   - KV 行数：方案 A 应该更少，差值约等于"多出来的 attention 权重 / 每 token 字节数"（§5.4）；
   - KDA 状态：每请求字节数应该相同（两边都是每卡 12 个头）。
2. **index-K 缓冲区大小。** 在方案 A 下打印一个 MLA 层的 index-K 缓冲区第 0 维，应该等于 `max_total_num_tokens`（物理行数），不应该是它的 N 倍（§3.4）。
3. **长上下文压测。** 用接近 `max_total_num_tokens * N` 的总 token 数跑满 KV 池，确认不 OOM，并且分配器的 id 空间正好用满。
4. **P2 补充。** 打开 `--enable-cp-decode-attn-tp` 和 CUDA graph，记录 graph capture 前后的空闲显存，确认 §5 的切片缓存被余量吸收。

---

## 7. 修改清单汇总

| 文件 | 位置 | 改动 | 状态 |
|---|---|---|---|
| `model_executor/pool_configurator.py` | 452-462 行 | indexer 不乘 dcp 的前提成立（`set_dsa_shards_index_k` 在模型构造时设置，早于核算） | 不改 |
| `model_executor/pool_configurator.py` | 396-403 行 | `attn_tp_size` 只用于 MHA；GLM 层切分方案 A 下不开启 | 不改 |
| `mem_cache/kv_cache_configurator.py` | 1860-1862、2599-2728 行 | K3.1 走 `_build_hybrid_linear_kv_pool`，参数和 TP8 + DCP8 相同 | 不改 |
| `mem_cache/kv_cache_configurator.py` | 2936-2959 行 | `dcp_enabled` 为真，走 paged 分配器，id 空间放大 N 倍 | 不改 |
| `configs/kimi_linear.py` | 219-228 行 | KDA 状态按 `linear_attn_tp_size() = tp_size` 切头，每卡 12 头 | 不改 |
| `layers/cp/cp_decode_attn_tp.py` | 125-199 行 | o_proj 切片缓存没有计入核算（P2） | 记录，GPU 验证后再定 |

和计划二不同的结论：

1. §4.7 的两项都只需要确认，不需要改代码；
2. 新增一项 P2 的观察项：D2 的 o_proj 切片缓存在 KV 池分配之后才创建。
