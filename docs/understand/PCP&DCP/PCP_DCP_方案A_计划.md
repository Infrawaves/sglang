# PCP 与 DCP 兼容：方案 A（DCP 组 = CP 组）计划文档

> 状态：草案，待人工评审
> 代码基线：`infrawaves/dev/v0.5.23-k31` 分支
> 首要目标模型：Kimi-K3.1（`KimiK31ForConditionalGeneration`）
> 文中带"（推断）"的内容是读代码得出的判断，还没有运行验证。

---

## 1. 背景

### 1.1 现状

当前代码禁止同一个实例同时开启 prefill CP（PCP）和 decode CP（DCP）：

```python
# python/sglang/srt/arg_groups/parallel_hook.py:266-267
if cfg.dcp_size > 1 and cfg.enable_prefill_cp:
    raise ValueError("--dcp-size > 1 does not support --enable-prefill-cp.")
```

Kimi-K3.1 还有两处同样的限制：
- `arg_groups/cp_tp_group_sharing_hook.py`：unsupported 列表里有 `"dcp-size > 1"`。
- `arg_groups/model_overrides/kimi_k3.py:240-241`：K3.1 的 DCP 检查拒绝 `"prefill context parallel"`。

### 1.2 为什么互斥

在同一组卡上，两者对 attention 的三个方面做了相反的假设：

| 方面 | PCP | DCP |
|---|---|---|
| attention 头 | 复制（`attn_tp_size = tp/dp/cp`，cp=tp 时为 1） | 切分（DCP 组是 TP 组的连续切片） |
| token 计算 | 切分（interleave 或 zigzag） | 不切（每卡都算全部 token） |
| KV 存储 | 复制（gather 后每卡都写完整的一份） | 按 owner 切分（位置 p 只存在 `p % N` 号卡上） |

如果绕过检查同时打开，会出现以下问题：
- PCP 的 attention 读不到完整的 KV，静默算错；
- DCP 的 extend buffer 形状不匹配，直接报错；
- decode 时 LSE 合并后的头数和 o_proj 需要的头数对不上。

### 1.3 现有的替代方案

- **PD 分离**：Prefill 实例用 PCP，Decode 实例用 DCP，现在已经能用。
- **`--enable-kv-cache-sharding`**（`arg_groups/kv_shard_hook.py`）：在 PCP 下把 KV 按页切分存储，仅限 PD 的 Prefill 实例。K3.1 目前不在它支持的后端组合里。

---

## 2. 目标与非目标

### 2.1 目标

在**同一个实例、同一组卡**上，同时得到：
1. PCP 的 prefill 加速：每张卡只算 1/N 的 query token。
2. DCP 的 KV 显存节省：MLA 和 indexer 的 KV 每张卡只存 1/N。

典型配置：`--tp-size 8 --enable-prefill-cp --attn-cp-size 8 --cp-strategy interleave --dcp-size 8`。

### 2.2 非目标（第一期不做）

- 方案 B（DCP 嵌套在 attention TP 内，`tp = cp × attn_tp`）。
- `attn_cp_size != dcp_size` 的组合。
- `--dcp-kv-layout page`。
- zigzag 策略：方案 A 只支持按位置取模的 interleave（原因见 §3.3）。以后扩展到 GQA/MHA 模型（FlashAttention / TRT-LLM MHA 后端）时，可以用"算完后按 owner 做 all-to-all 重分发"的方式（§3.3 的 Z1）支持 zigzag。
- GQA 和 MHA 模型。
- attention DP、PP 和这个方案的组合。
- HiCache L3、DFLASH 和 DSPARK 投机解码（放到 P4）。

---

## 3. 设计

### 3.1 核心原则

> **负责计算某个 token 的卡，就是存这个 token 的 KV 的卡。**

### 3.2 拓扑

| 组 | 方案 A 下的定义 | 8 卡例子 |
|---|---|---|
| TP 组 | 不变 | [0..7] |
| attention CP 组 | `attn_cp_size == tp_size`，复用 TP 组（`parallel_state.py:2753`） | [0..7] |
| DCP 组 | **直接等于 attention CP 组** | [0..7] |
| `dcp_rank` | **等于 `attn_cp_rank`**（现在是 `tp_rank % dcp_size`） | r |
| attention TP | 1（头复制） | — |
| KDA 头切分 | 整个 TP 组（CP-TP group sharing 下的现有行为） | 每卡 12 个头 |

### 3.3 token 切分：从"拼接下标取模"改成"按位置取模"

- **现在**：interleave 按整批 token 拼接后的下标 `i % cp` 切分（`layers/cp/interleave.py` 的 `_interleave_shard` 对 `x[:total_seq_lens]` 做 `view(-1, cp_size)`）。
- **DCP 的 owner 规则**：按序列位置 `pos % dcp` 分配。
- **两者错位**：如果请求有前缀 P，chunk 里第 j 个 token 的位置是 P+j，两种规则相差 `P % N`；一批里有多个请求时，错位会累积。
- **新规则**：每个请求单独计算，CP rank r 负责 `(prefix_len_req + j) % N == r` 的那些 token。

这样，CP rank r 算出来的 token，正好就是 DCP 规则下由它存储的 token。

#### 切分方式的选择：方案 A 只支持 interleave

方案 A 只支持按位置取模的 interleave，`--cp-strategy zigzag` 在参数校验阶段直接报错。原因有三个：

1. **K3.1 本身就要求 interleave。** K3.1 的 prefill CP 只能走 CP-TP group sharing，它要求 `--cp-strategy interleave`（`arg_groups/cp_tp_group_sharing_hook.py` 的 `_INTERLEAVE_MODELS`）。K3.1 的 MLA 层是 DSA 稀疏 attention，而 interleave 也只支持 DSA 后端（`layers/cp/interleave.py:357`）。DSA 为每个 query 单独选 key，所以 query 在序列里不连续也没有关系。
2. **只有按位置取模的 interleave 能做到"算哪里就存哪里"。** DCP（`--dcp-kv-layout token`）的 owner 规则是 `pos % N`。interleave 改成按位置取模后，CP rank r 计算的 token 和它要存的 token 完全一致，算完直接写到本地，不需要重分发。
3. **zigzag 和 owner 规则对不上。** zigzag 下 rank r 负责第 r 块和第 2N−1−r 块，每块是连续的一段 token，其中大部分的 owner 是别的卡。要兼容只有两种办法：
   - **Z1：算完后按 owner 重分发。** 用一次 all-to-all，把新 token 的 KV 发给 owner 卡写入。技术上可行，每卡发出约 (N−1)/N 的本地新 token KV。代价是多一次通信，以及额外实现这个重分发步骤。
   - **Z2：让 DCP 的 owner 规则按 zigzag 的块来定。** 不可行。DCP 的 owner 必须只和位置有关，跨 chunk、跨 decode step 保持不变；decode 读取、PD 传输、HiCache 和 radix cache 都依赖这个固定的规则。

**和现有 interleave 的区别：** 现有实现按**整批拼接后的下标**取模（`_interleave_shard` 对 `x[:total_seq_lens]` 做 `view(-1, cp_size)`），和 DCP 的规则同样对不上。所以方案 A 不能直接复用现有的 interleave，要新增按位置取模的版本（见 §4.3）。

**参数约束：** 方案 A 要求 `--cp-strategy interleave` 且 `--dcp-kv-layout token`。如果以后支持 `--dcp-kv-layout page`（owner 按 64 个 token 的页取模），interleave 也要相应改成以页为单位轮流分配。

### 3.4 Prefill 数据流（以 DSA/MLA 层为例）

```
1. 本卡对自己负责的 ~1/N token 计算 Q（全部头）、latent KV、indexer K
2. 直接写入本地 KV cache：本卡算出的 token 本来就归本卡存，不需要 gather 再写
3. 对整个序列（前缀 + 本轮新 token），从各 owner 卡 all-gather KV 行和 indexer K 行
   （复用 dcp_gather_request_rows）
4. indexer：本卡的 query 对完整的 indexer K 打分，选出 top-k（按 query 的全局位置做 causal）
5. 稀疏 MLA：本卡的 query、全部头，对 gather 来的完整 KV 计算，结果精确
6. o_proj 是复制的，直接得到本卡这些行的完整输出
```

KDA 层和 MoE 层沿用 CP-TP group sharing 现在的做法（KDA gather 完整序列后按头切分；MoE 先 gather 行，再 reduce-scatter 回本卡）。只需要配合新的行布局修改重排函数。

### 3.5 Decode 数据流

decode 时 CP 不生效（`is_cp_active` 只在 EXTEND/MIXED 模式下成立），所以走 DCP decode：

```
1. 每张卡在本地算出全部头的 Q（权重是复制的，相当于 dcp_replicate_q_proj）
2. 新 token 的 KV 只写到 owner 卡
3. indexer：先选本地 top-k，再 all-gather 候选，合并成全局 top-k（dcp_global_topk / DcpTopKMerge），
   再筛出本卡的 slot（dcp_localize_compact）
4. 对本卡选中的 KV 做稀疏 attention，得到部分输出和 LSE
5. 按 LSE 合并：fi_a2a / push-combine，合并后每张卡拿到 1/N 的头
6. o_proj：见下面的冲突处理
```

**冲突处理：** DCP 合并后每张卡只有 1/N 的头，但方案 A 下 o_proj 是完整复制的。

| 选项 | 做法 | 优缺点 |
|---|---|---|
| D1 | 合并后再 all-gather 一次头，用完整的 o_proj | 实现简单；多一次通信，o_proj 计算是冗余的 |
| **D2（推荐）** | 复用 `--enable-cp-decode-attn-tp`（`layers/cp/cp_decode_attn_tp.py`）：decode 时把复制的 o_proj 按 CP rank 切头，再 all-reduce | 正好对上 DCP 合并的结果；需要把 K3.1 加入白名单 `CP_DECODE_ATTN_TP_SUPPORTED_ARCHS`，并适配 K3 的输出门 `g_proj` |

### 3.6 预期的资源变化（相对 TP8+PCP8 / TP8+DCP8，推断）

| 方面 | TP8+PCP8 | TP8+DCP8 | 方案 A |
|---|---|---|---|
| MLA KV 每卡 | 18.9 KB/token | 约 2.4 KB/token | 约 2.4 KB/token |
| MLA 权重 | 完整复制 | Q 侧复制，V 和 o_proj 按头切 | 完整复制（decode 时可以按头切片使用） |
| prefill 每卡处理的 query token | 1/8 | 全部 | 1/8 |
| prefill 时 DSA 层的 KV 通信 | gather 新 token | gather 整个序列 | gather 整个序列 |
| decode | 和 TP8 相同（冗余计算） | 本地 1/8 的 KV + LSE 合并 | 和 TP8+DCP8 相同 |

---

## 4. 代码修改清单

### 4.1 参数和校验（`python/sglang/srt/arg_groups/`）

| 文件 | 修改 |
|---|---|
| `parallel_hook.py:266-267` | 改成条件放行：`dcp_size == attn_cp_size`，`cp_strategy == interleave`，`dcp_kv_layout == token`。`cp_strategy == zigzag` 时明确报错，提示方案 A 只支持 interleave |
| `cp_tp_group_sharing_hook.py` | 从 unsupported 列表里去掉 `dcp-size > 1`，并新增 `dcp_size != attn_cp_size` 时报错 |
| `model_overrides/kimi_k3.py:214-260` | K3.1 DCP 检查改为放行 prefill CP，同时保留和补充对 DSA 后端、spec 算法、PD 后端的约束 |
| 新增约束 | `page_size % dcp == 0`；chunked prefill 大小满足 CP 对齐、DCP 页对齐和 MegaMoE 上限（`mega_moe_hook.py:239-245`） |
| `hicache_hook.py` | 第一期显式拒绝方案 A 和 HiCache 的组合 |

### 4.2 rank 拓扑

| 文件 | 修改 |
|---|---|
| `distributed/parallel_state.py:2705-2728` | 方案 A 下，DCP 组直接复用 `_ATTN_CP`，不再从 TP 组切片 |
| `runtime_context.py:1610-1611` | 方案 A 下 `dcp_rank = attn_cp_rank` |
| `runtime_context.py:137-157`（`derive_parallel_widths`） | 加入"DCP 复用 CP 组"的标志，保证 `attn_dcp_size` 和 `attn_cp_size` 一致 |

### 4.3 CP 切分策略

| 文件 | 修改 |
|---|---|
| `layers/cp/interleave.py` | 新增按位置对齐的切分：`build_metadata` 记录每个请求的偏移、`per_rank_actual_token` 和 padding；同步修改 `shard_hidden_states`、`shard_position_ids`、`gather_hidden_states`、`gather_kv_cache`、`local_q_indices`、`moe_num_token_non_padded` |
| `layers/cp/interleave.py` 里的 `cp_interleave_to_sequence_order` / `cp_sequence_to_interleave_order` | 适配新的行布局（KDA 用，`kimi_k3.py:2667-2722`） |
| `layers/cp/bcg/`（例如 `interleave_bcg.py`） | 和上面保持一致；第一期可以先关闭 prefill graph |
| `layers/cp/utils.py:270-304`（`prepare_cp_forward`） | 截断 `out_cache_loc` 后，确认写入位置和本卡计算的 token 一一对应 |

### 4.4 KV 写入和读取

| 文件 | 修改 |
|---|---|
| `layers/cp/interleave.py` 的 `materialize_full_mla_kv`、`materialize_full_indexer_k_cache`、`run_indexer_with_kv_gather` | 方案 A 下只写本地 token，不再先 gather 再写 |
| `mem_cache/kv/alloc/id_space.py`（`translate_write`） | 现有的 owner 映射应该可以直接用（推断），需要加单元测试确认 |
| `layers/dcp/planner.py` | 前缀长度按 DCP 页对齐的假设（`:113`），要在 CP 的 chunk 边界上同样满足 |

### 4.5 prefill attention

| 文件 | 修改 |
|---|---|
| `layers/dcp/metadata.py` 和 `layers/cp/base.py` | 新增一个合并后的元数据，同时描述"本卡的 query 行（来自 CP）"和"完整的 KV 行（来自 DCP）"。用 `msgspec.Struct` 定义，并和上游计划的 `DecodeContextParallelStrategy` 重构保持一致 |
| `model_executor/runner/eager_runner.py:264-300` | 调整 `prepare_cp_forward` 和 `prepare_context_parallel_metadata_for_dcp` 的调用顺序和依赖关系。**这是冻结的核心路径，修改前要先读 `large-class-style` skill** |
| `layers/attention/dsa_backend.py`（`dcp_gather_kv_cache`，`:1392`） | 支持 query 只覆盖本卡的行 |
| `layers/attention/dsa/dsa_indexer.py`（`_dcp_gather_index_k`，`:1930`；打分入口 `:1700-1730`） | indexer 的 query 用本卡的行，KV 用 gather 来的完整序列，causal 按全局位置计算 |
| `models/deepseek_common/attention_forward_methods/forward_mla.py` | 修改阶段判断（`is_dcp_mla_decode_phase`，`:200-211`）和 `bcg_dcp_extend_kv_gather`（`:109`）的分支，在 CP extend 下走新路径 |

### 4.6 decode attention

| 文件 | 修改 |
|---|---|
| `layers/cp/cp_decode_attn_tp.py` | 把 `KimiK31ForConditionalGeneration` 加入 `CP_DECODE_ATTN_TP_SUPPORTED_ARCHS`；适配 K3 的 o_proj 包装和输出门 `g_proj` |
| `models/kimi_k3.py`（`KimiK3MLAAttention`，`:2745-3013`） | decode 时按 CP rank 切头使用 o_proj 和 g_proj；DCP 的 V 预投影（`dcp_v_preproject`）要按切片后的头准备权重 |
| `forward_mla.py:1030-1100`（DCP 合并） | 确认合并后的头布局和 cp-decode-attn-tp 的切片一致 |

### 4.7 模型代码（`models/kimi_k3.py`）

| 位置 | 修改 |
|---|---|
| `KimiK3MoE._forward_cp_shard_rows`（`:1977`） | 适配新的行布局 |
| `_stage_facts` 和层边界（`layers/layer_boundary/`） | 确认 gather 和 reduce-scatter 用的是新的行布局 |
| `KimiK3MLAAttention.__init__`（`:2770-2776`） | 方案 A 下照常调用 `set_dsa_shards_index_k()` |

### 4.8 CUDA Graph（`model_executor/runner/`）

| 位置 | 修改 |
|---|---|
| decode graph | 加上 cp-decode-attn-tp 的切片缓存；遵守 `configure_cp_decode_attn_tp` 对权重更新的限制 |
| prefill breakable graph（`prefill_cuda_graph_runner.py:736-740`，`arg_groups/cuda_graph_hook.py:208-212`） | 重新计算 CP buffer 和 DCP gather buffer 的静态尺寸；**P1 和 P2 阶段先禁用** |

### 4.9 调度和显存核算

| 位置 | 修改 |
|---|---|
| `mem_cache/kv_cache_configurator.py` | KV 池容量按 DCP 切分后计算 |
| radix cache | 页对齐按放大后的页（`page_size × dcp`） |
| chunked prefill | 同时满足 CP、DCP 和 MegaMoE 的对齐要求 |

### 4.10 PD 分离（P4）

| 位置 | 修改 |
|---|---|
| `disaggregation/common/conn.py:440-499` | 新增"prefill 是 CP8+DCP8"的身份：按 DCP rank 发送自己存的那份 KV；修改 `is_dummy_cp_rank`、`enable_all_cp_ranks_for_transfer` 的判断和 bootstrap 注册 |
| `disaggregation/mooncake/conn.py` | prefill 是 CP8+DCP8、decode 是 DCP8 时走按页传输；KDA 状态继续按头 r 对 r 传输 |

### 4.11 HiCache 和投机解码（P4）

| 位置 | 修改 |
|---|---|
| `mem_cache/hicache_storage.py:69-77` | 判断条件从 `tp_rank % dcp_size == dcp_rank` 改成按 CP rank 判断 |
| `mem_cache/pool_host/base.py:242` | 保留 `page_size % dcp == 0` |
| DFLASH | CP-TP sharing 下草稿用普通 TP，DCP 下草稿的 attention 要求 `cutedsl_mla`，两者组合需要单独设计 |

---

## 5. 阶段划分

| 阶段 | 内容 | 产出 | 验收标准 |
|---|---|---|---|
| **P0：确认收益** | 不改代码；在 B300 8 卡上实测 TP8 和 TP8+PCP8 的 TTFT（8K、32K、128K 输入，关闭 radix cache）；确认是否有统一部署的需求 | 压测报告 | **决定继续还是停止** |
| P1：prefill 正确性 | 4.1–4.5、4.7；只用 eager 模式、统一部署；只支持 K3.1 加一个 DSA 后端（建议先选 FlashMLA） | 原型分支 | 每层 hidden 和 logits 与 TP8 对比在容差内 |
| P2：decode | 4.6 | 原型分支 | sgl-eval 精度和 TP8+DCP8 一致 |
| P3：性能 | 4.8、4.9；通信和计算重叠 | 基准测试报告 | TTFT 优于 TP8+DCP8；TPOT 和 KV 容量与 TP8+DCP8 持平 |
| P4：周边功能 | 4.10、4.11 | 端到端测试 | PD 传输和 HiCache 正确 |

粗略工作量：一到两名熟悉这块代码的工程师，P1–P3 约 6–8 周，P4 另算（推断）。

---

## 6. 测试计划

1. **单元测试**
   - DCP 组等于 CP 组，`dcp_rank == attn_cp_rank`。
   - 按位置对齐的切分和还原：覆盖带前缀、多个请求、前缀长度不是 N 的倍数、padding 等情况。
   - KV 的 owner 写入：每个位置只有一张卡写入，并且是本卡计算出的 token。
2. **数值等价**：同一批输入，对比 TP8 和方案 A 的每层输出和 logits。分别测试有前缀命中和没有命中的情况。
3. **精度**：按 `.claude/rules/prefer-sgl-eval.md` 的规范，用 sgl-eval 跑 gsm8k 和一个长上下文任务，采样参数用 K3.1 推荐值，和 TP8+DCP8 对比。报告时写明 harness 和参数。
4. **性能**：同一台机器上对比 TP8+DCP8、TP8+PCP8 和方案 A 的 TTFT（p50/p90）、TPOT 和 `max_total_num_tokens`。
5. **回归**：现有 CP 和 DCP 的测试（`test/registered/cp/`、DCP 相关的单元测试）要全部通过。

---

## 7. 风险

| 风险 | 说明 | 应对 |
|---|---|---|
| **收益不足** | 粗估 PCP 在 K3.1 上省下的复制计算大约占每卡 prefill 时间的 10%，扣掉额外开销后 TTFT 净收益可能只有个位数百分比，而且上下文越长越小 | 由 P0 决定是否继续 |
| prefill 通信变多 | 方案 A 下 attention 要 gather **整个序列**的 KV（包括前缀），而现在的 PCP 只 gather 新 token。前缀越长，代价越大 | P3 评估通信和计算的重叠；或者只在前缀较短时启用 |
| 两套元数据合并 | `attn_cp_metadata` 和 `attn_dcp_metadata` 现在是有意分开的（`layers/dcp/metadata.py` 的注释） | 和上游的 `DecodeContextParallelStrategy` 重构协调 |
| 改动面大 | 涉及冻结的核心文件和多个后端 | 按阶段推进，P1 只支持一个后端 |
| 和上游的冲突 | 这是 Infrawaves 的分支 | 先和上游讨论 RFC，或者等上游重构完成后再做 |
| 静默的数值错误 | owner 规则对不上时，结果会算错但不报错 | P1 加上不变量检查（例如每个 slot 只被写一次、gather 后的行数等于序列长度） |

---

## 8. 待决策的问题

1. 第一期的目标场景是**统一部署**，还是 **PD 的 Prefill 实例**？如果是 PD 的 Prefill 实例，扩展 `--enable-kv-cache-sharding` 可能更划算。
2. decode 时的 o_proj 选 D1（all-gather 头）还是 D2（cp-decode-attn-tp）？
3. P1 用哪个 DSA 后端：FlashMLA（默认）、TRT-LLM 还是 prims_ts（Maas 线上用的）？
4. 是否先和上游沟通 RFC，再在 Infrawaves 分支上做原型？
5. P0 的 TTFT 收益达到多少，才值得继续做？例如 ≥ 15%。

---

## 9. 参考代码位置汇总

（以下路径都在 `python/sglang/srt/` 下）

- 互斥检查：`arg_groups/parallel_hook.py:266`；`arg_groups/cp_tp_group_sharing_hook.py`；`arg_groups/model_overrides/kimi_k3.py:214-260`
- 组和 rank：`distributed/parallel_state.py:2673-2771`；`runtime_context.py:73, 137-157, 1610-1611`
- PCP：`layers/cp/interleave.py`、`layers/cp/utils.py`、`layers/cp/base.py`、`layers/cp/cp_decode_attn_tp.py`
- DCP：`layers/dcp/`（`comm.py`、`planner.py`、`layout.py`、`metadata.py`）；`mem_cache/kv/alloc/id_space.py`
- DSA 在 DCP 下的实现：`layers/attention/dsa/dsa_dcp_sharded.py`、`layers/attention/dsa_backend.py`、`layers/attention/dsa/dsa_indexer.py`
- MLA 前向：`models/deepseek_common/attention_forward_methods/forward_mla.py`
- K3.1 模型：`models/kimi_k3.py`
- 运行器：`model_executor/runner/eager_runner.py`、`model_executor/runner/prefill_cuda_graph_runner.py`
- PD 传输：`disaggregation/common/conn.py`、`disaggregation/mooncake/conn.py`
- 已有的 KV 分片功能：`arg_groups/kv_shard_hook.py`、`layers/attention/kv_shard_hooks.py`、`mem_cache/page_interleave_pool.py`
