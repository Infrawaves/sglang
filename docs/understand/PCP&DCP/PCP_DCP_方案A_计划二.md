# PCP 与 DCP 兼容：方案 A（DCP 组 = CP 组）计划文档 · 第二版

> 状态：草案，待人工评审
> 代码基线：`infrawaves/dev/v0.5.23-k31` 分支
> 首要目标模型：Kimi-K3.1（`KimiK31ForConditionalGeneration`）
> 相对第一版（`PCP_DCP_方案A_计划.md`）的修订依据：`docs/understand/` 下 27 份代码理解文档，以及 `docs/understand/方案A计划问题汇总.md`（下文简称"汇总"，引用格式为"汇总 x.y"）
> 文中带"（推断）"的条目是读代码得出的，没有运行验证。

---

## 0. 相对第一版的主要变化

| 方面 | 第一版 | 第二版 |
|---|---|---|
| rank 拓扑（§4.2） | 修改 `dcp_rank` 公式，DCP 组复用 `_ATTN_CP` | 在方案 A 约束下数值本来就相同。**第一期不改公式**，只加断言和 `_validate_parallel` 校验 |
| KV 写入（§4.4） | 改成只写本地 | **P1 保留"gather 后写全部行"**，依靠 owner 过滤保证正确性；P3 再优化成只写本地。补上之前漏掉的 `_forward_trtllm` 和 `forward_mla.py` 的调用点 |
| prefill attention（§4.5） | 改 `bcg_dcp_extend_kv_gather`、`is_dcp_mla_decode_phase` | K3.1 不经过这两处，**不改**。新的重点是：query 侧按本卡行构建，KV 侧按完整请求列表构建，避免形状不一致 |
| decode（§4.6） | 推荐 D2，把 K3.1 加入 cp-decode-attn-tp 的白名单 | 只加白名单不够，还有 8 处要处理。**P2 先做 D2 可行性实验**，再决定选 D1 还是 D2 |
| 参数校验（§4.1） | 3 个文件 | 补上 `model_overrides/deepseek_v2.py`；明确 hook 的执行顺序；P1 后端写明 `flashmla_kv` |
| CUDA Graph（§4.8） | `cuda_graph_hook.py:208-212` | 起作用的是 213-216 行；还要拒绝显式指定的 breakable 和 full prefill graph |
| 显存核算（§4.9） | `kv_cache_configurator.py` | 每 token 字节数在 `pool_configurator.py` 计算；K3.1 走 hybrid linear 池 |
| PD（§4.10） | 修改 rank 身份和注册 | rank 身份基本已经满足。重点改成：DCP 大小校验、布局校验、页对齐、混用的 attn TP |
| 性能目标（P3） | TPOT 和 TP8+DCP8 持平 | 方案 A 会失去融合 all-reduce、SP-MoE 和 token packing。**目标要下调或改成单独评估** |

---

## 1. 背景（简要）

当前三处代码禁止 PCP 和 DCP 同时开启：
- `arg_groups/parallel_hook.py:266-267`
- `arg_groups/cp_tp_group_sharing_hook.py:79`（`"dcp-size > 1"`）
- `arg_groups/model_overrides/kimi_k3.py:240-241`（`"prefill context parallel"`）

互斥的原因是两者在同一组卡上对 attention 头、token 计算和 KV 存储做了相反的假设：PCP 是头复制、token 切分、KV 复制；DCP 是头切分、token 不切、KV 按 owner 切分。详见第一版 §1.2。

现有的替代方案：
- PD 分离，Prefill 用 PCP，Decode 用 DCP；
- `--enable-kv-cache-sharding`，只能用在 PD 的 Prefill 实例上，而且 K3.1 目前不支持。

---

## 2. 目标与非目标

### 2.1 目标

在同一个实例、同一组卡上，同时得到：
- PCP 的 prefill 加速：每卡只算 1/N 的 query token；
- DCP 的 KV 显存节省：MLA 和 indexer 的 KV 每卡只存 1/N。

典型配置：`--tp-size 8 --enable-prefill-cp --attn-cp-size 8 --cp-strategy interleave --dcp-size 8 --dsa-prefill-backend flashmla_kv --dsa-decode-backend flashmla_kv`。

### 2.2 非目标（第一期不做）

| 不做的项 | 原因或后续 |
|---|---|
| 方案 B（DCP 嵌套在 attention TP 内） | 卡数需求大，KV 仍然复制 cp 份 |
| `attn_cp_size != dcp_size`、`attn_cp_size != tp_size` | CP-TP sharing 要求 `attn_cp_size == tp_size` |
| zigzag 策略 | K3.1 的 DSA 只支持 interleave（汇总 4.2：真正的原因是 DSA 只支持 interleave，不是 owner 规则对不上） |
| `--dcp-kv-layout page` | 已经被现有逻辑间接排除（汇总 1.7） |
| GQA、MHA、非 DSA 的 MLA 模型 | 它们依赖 `bcg_dcp_extend_kv_gather` 等不同路径（汇总 5.5） |
| attention DP、PP | PD 传输和 MTP 在多处要求 PP=1（汇总 10.8） |
| prefill CUDA Graph（breakable 和 full） | P1/P2 显式拒绝，P3 再评估 |
| HiCache（L2 和 L3） | **第一期整体拒绝**，P4 再做（统一汇总 1.9 指出的前后不一致） |
| 投机解码（DFLASH、DSPARK、MTP/EAGLE） | 放到 P4；MTP draft 在 DCP 下现在就会报错（汇总 1.11） |
| PD 分离 | 放到 P4；第一期只支持统一部署 |
| unified memory、`prefill_complete` 分配策略 | 和方案 A 不兼容（汇总 10.6、10.8） |

---

## 3. 设计

### 3.1 核心原则（修订）

> **计算某个 token 的卡，就是存这个 token 的 KV 的卡（最终目标）。在 P1 阶段，只要求存储结果正确，可以先多做一次 gather。**

### 3.2 拓扑

在 `tp = attn_cp = dcp = N`、attn_tp=1 的约束下：
- `_ATTN_CP` 复用 `_TP`（`parallel_state.py:2753`）；
- 现有 DCP 组按 `dcp_size` 切 TP 组得到的，也是整个 TP 组；
- `dcp_rank = tp_rank % dcp_size = tp_rank = attn_cp_rank`。

所以**拓扑不需要修改**，只需要断言这个约束成立（汇总 2.1）。

### 3.3 token 切分：按位置取模的 interleave

- **现状**：按整批拼接后的下标取模，和 DCP 的 owner 规则 `pos % N` 错开 `prefix_len % N`。
- **新规则**：每个请求单独切分，CP rank r 负责 `(prefix_len + j) % N == r` 的 token。每个请求单独补齐，补齐后有效行排在每卡分片的前部（MegaMoE 的要求，汇总 3.7）。
- **要一起修改的拼接下标逻辑**（汇总 3.1）：
  - `_interleave_shard`、`build_metadata`
  - `shard_per_request` 和它调用的 Triton kernel `dsa_cp_interleave_q_seqs`（`dsa_backend.py:1624` 调用）
  - `interleave_rows_per_request`、`cp_interleave_input_ids`
  - KDA 用的重排：在 `build_metadata` 里填好 `gather_index`/`local_index`，现有的任意排列分支可以直接复用（汇总 3.3）
- **参考实现**：`deepseek_v4_backend.py:2112-2170` 的 `_late_layer_tail_cp_layout`（汇总 3.2）。
- **扩展点**：通过 `register_cp_kv_layout` 注册方案 A 的布局，它会检查 `--cp-strategy`，正好能实现"zigzag 报错"。

> 第一版说"P1 必须做到只写本地"。读代码后发现，`translate_write` 会把非 owner 的写入映射到 sink 0，indexer K 的融合 store 也会按 owner 过滤，所以保留现有 gather 后写全部行的做法，结果也正确（汇总 4.1）。**按位置取模的切分仍然是必须的**，原因是：
> 1. P3 要做"只写本地"，前提是本卡算的 token 正好是它拥有的 token。
> 2. decode 时新 token 的 owner 由位置决定，prefill 的布局要和它一致，才能统一推理元数据。
> 如果 P0 的结论是只做 P1 和 P2，可以评估是否保留现有切分（推断）。

### 3.4 prefill 数据流（DSA/MLA 层，P1 版本）

```
1. 本卡对自己负责的 ~1/N token 计算 Q（全部头）、latent KV、indexer K
2. 写 KV：P1 保留 CP 的 gather（新 token），写入时由 owner 过滤，每个位置只落在 owner 卡上
          P3 改成直接写本地（要求 out_cache_loc 换成本卡行的子集）
3. 对整个序列（前缀 + 本轮），从各 owner 卡 all-gather KV 行和 indexer K 行（dcp_gather_request_rows）
   —— KV 侧的元数据按"完整请求列表"构建
4. indexer：本卡 query 行对完整的 indexer K 打分
   —— query 侧的元数据（ks/ke、topk_indices_offset、cu_seqlens_k）指向 gather 后的完整布局，按全局位置做 causal
5. 稀疏 MLA：本卡 query、全部头，对 gather 来的完整 KV 计算（flashmla_kv）
6. o_proj 是复制的，直接得到本卡这些行的完整输出
```

**为了避免重复通信，P1 要关闭的 CP KV gather 调用点：**
- `forward_mla.py:472-480`：`cp_overlap` 的构造
- `forward_mla.py:866`：`prepare_kv_for_attention`
- `dsa_backend.py:4508-4556`：`_forward_trtllm` 自带的 CP KV gather（P1 不用 trtllm，但要加 guard）
- `forward_mla.py:666-675`：indexer 的 CP overlap，先关闭

KDA 层和 MoE 层沿用 CP-TP sharing 现在的做法，只需要让层边界的 gather 和 padding 适配新的行布局（汇总 3.3、3.4）。

### 3.5 decode 数据流与 o_proj 头数冲突

decode 时 CP 不生效，走 DCP decode：本地 Q → indexer 先选本地 top-k、再合并成全局 top-k → 对本卡选中的 KV 做稀疏 attention → LSE 合并 → o_proj。

**冲突**：方案 A 下 attn_tp=1，`num_local_heads = H`。DCP decode 路径假定每卡先算 H/N 个头，再 all-gather 成 H 个头，合并后又回到 H/N 个头。

| 方案 | 做法 | 需要处理的点 |
|---|---|---|
| **D1** | Q 已经是全部头：跳过 `forward_mla.py:878` 的 Q all-gather（效果类似 `q_replicate_active`）；合并后再 all-gather 一次头，用完整的 o_proj | 多一次头 all-gather，o_proj 计算是冗余的。还要修正多处 `× dcp_size` 的头数（汇总 6.4） |
| **D2** | decode 时用 `maybe_use_decode_attn_tp` 把 `q_b_proj`、`w_kc`、`w_vc`、`o_proj`、`g_proj` 和头数按 CP rank 切成 H/N。这样 Q 先算 H/N 个头，再 gather 回 H 个，合并后剩 H/N 个，正好对上切过的 o_proj | 见下面 6 项 |

**D2 必须同时满足的 6 项（汇总 6.1–6.8）：**
1. K3 的 `_run_self_attn`（`kimi_k3.py:3312`）没有调用 `maybe_use_decode_attn_tp`，要加上。
2. `g_proj` 要加入切片列表，decode graph 下的输出门侧流也要覆盖。
3. K3 的 `_stage_facts` 要补上 `ordinary_tp_group=SumGroup.TP` 等延迟求和的声明（参考 `deepseek_v2.py:2790-2840`）。不补的话，o_proj 的部分和不会被求和，结果静默出错。
4. `_prepare_dcp_v_weights` 必须在切片后的上下文中创建缓存，权重更新时也在同样的上下文中覆盖（`kimi_k3.py:4101-4102` 的 `post_load_weights` 调用点要移动或改写）。
5. 修正 `dsa_backend.py:755-758` 的 `num_q_heads` 公式，以及 `attn_mqa_for_dcp_decode` 的头数（`deepseek_v2.py:2261`）。
6. `set_decode_attn_tp` 的生效范围包括 target verify 和短 extend，要和 `is_dcp_mla_decode_phase` 对齐。

**对 D2 的分歧**：cp_decode_attn_tp 的文档认为 DCP decode 要求每卡对全部头计算，所以需要一种只切 o_proj 的新模式；forward_mla 的文档认为切片后再 gather 就能对上。**P2 第一步先做实验：在 2 卡上用一层模型对比 TP2+DCP2 和方案 A + D2 的 attention 输出，再决定选 D1、D2，还是新增一种只切 o_proj 的模式（D3）。**

---

## 4. 需要修改的源代码清单

路径都相对于 `python/sglang/srt/`。"阶段"一列表示最早在哪个阶段修改。每个文件的详细说明见 `docs/understand/<文件名>_代码理解.md`。

### 4.1 参数和校验

| 文件 | 修改内容 | 目的 | 阶段 |
|---|---|---|---|
| `arg_groups/parallel_hook.py` | 第 266-267 行改成条件放行：`enable_prefill_cp and dcp_size > 1` 时，要求 `dcp_size == attn_cp_size == tp_size`，策略为 interleave 或尚未设置。`cp_strategy` 的最终检查放到 `handle_context_parallelism`（第 291 行之后） | 解除互斥，同时只允许方案 A 的组合。这样写是因为 hook 的执行顺序：第 266 行运行时 `cp_strategy` 可能还没设置（汇总 1.1） | P1 |
| `arg_groups/cp_tp_group_sharing_hook.py` | 第 79 行的 `"dcp-size > 1"` 改成只对 K3.1 放行，并要求 `dcp_size == attn_cp_size` | 不能让 GLM-5 Next 和 Qwen4-Exp 跟着放行 DCP（汇总 1.2） | P1 |
| `arg_groups/model_overrides/kimi_k3.py` | `_kimi_k31_dcp_overrides`（214-260 行）把 `"prefill context parallel"` 收紧为只允许方案 A 的组合；新增：后端必须是 `flashmla_kv`、拒绝投机解码、拒绝 PD 分离（第一期） | 只放行已验证的组合（汇总 1.3、1.5） | P1 |
| `arg_groups/model_overrides/deepseek_v2.py`（**新增**） | 检查 `_deepseek_family_overrides` 对 K3.1 的 CP/DCP 声明、`tp_size <= 8` 的断言和 `index_kpool` 检查，在方案 A 下是否成立 | 这个文件同样作用于 K3.1，第一版漏了（汇总 1.4） | P1 |
| `arg_groups/hicache_hook.py` | 新写一条规则：方案 A 加任何 HiCache 都报错。只能用 `enable_prefill_cp`、`attn_cp_size` 和 `dcp_size` 判断，因为这时 `enable_cp_tp_group_sharing` 还没声明 | K3.1 现在在 DCP 下允许 HiCache，不能依赖现有检查（汇总 1.9、1.1） | P1 |
| `arg_groups/cuda_graph_hook.py` | 保留 213-216 行默认关闭 BCG 的规则；新增：方案 A 下用户显式指定 breakable 或 full prefill graph 时报错 | 显式指定会跳过自动关闭的规则；full graph 加 DCP 可能 KeyError（汇总 1.10、8.1、8.2） | P1 |
| `arg_groups/mega_moe_hook.py` | 方案 A 下每卡 token 预算改成按"每请求单独补齐"估算，上限约 `ceil(tokens/N) + 请求数` | 现有估算可能偏小（汇总 1.8） | P1 |
| 新增断言（放在 `parallel_hook.py` 或 `runtime_context.py`） | `dcp_size == attn_dcp_size == attn_cp_size == tp_size`、`attn_tp_size == 1`、`dcp_rank == attn_cp_rank`；拒绝 MTP draft、unified memory 和 `prefill_complete` | 第一期不改拓扑公式，只用断言兜底（汇总 1.6、2.1） | P1 |

### 4.2 rank 拓扑

| 文件 | 修改内容 | 目的 | 阶段 |
|---|---|---|---|
| `runtime_context.py` | 在 `_validate_parallel` 里加上 DCP 的 rank、宽度和组一致性校验；公式不变。在 `attn_dcp_size` 字段文档里写明方案 A 下的含义 | 汇总 2.1、2.4、2.5 | P1 |
| `distributed/parallel_state.py` | **第一期不改**组的构建。只更新过时的 docstring，并在 `initialize_model_parallel` 里断言方案 A 下 `_DCP` 的成员等于 `_ATTN_CP` | 如果复用组对象，会带来构建顺序、`use_quick_allreduce`、重复 destroy 等问题（汇总 2.2），断言就足够 | P1 |
| （评估）全仓 `attn_dcp_size` / `attn_dcp_rank` 的约 157 处使用点 | 逐个确认 MLA 路径没有假设"attention TP 组内的 DCP" | 汇总 2.5 | P1 |

### 4.3 CP 切分策略

| 文件 | 修改内容 | 目的 | 阶段 |
|---|---|---|---|
| `layers/cp/interleave.py` | 新增按位置取模的切分：`build_metadata` 记录每个请求的偏移、`gather_index`/`local_index`、每请求的补齐；同步修改 `_interleave_shard`、`shard_hidden_states`、`shard_position_ids`、`gather_hidden_states`、`gather_kv_cache`、`local_q_indices`、`shard_per_request`、`interleave_rows_per_request`、`moe_num_token_non_padded` | 让本卡计算的 token 和 owner 规则一致（§3.3） | P1 |
| `layers/cp/base.py` | 通过 `register_cp_kv_layout` 注册方案 A 的布局 | 复用现有的扩展点和 strategy 检查（汇总 3.2） | P1 |
| `layers/cp/utils.py` | 修改 `cp_interleave_input_ids`；P3 时在 `prepare_cp_forward` 里得到本卡行对应的 `out_cache_loc` 子集 | 汇总 3.1、4.3 | P1 / P3 |
| `kernels/` 下的 `dsa_cp_interleave_q_seqs`（**新增**） | 按位置取模计算每个请求的 query 行 | 原实现按拼接下标计算（汇总 3.1） | P1 |
| `layers/layer_boundary/`（层边界的 gather 和 padding） | 适配新的行布局和每请求补齐 | `_forward_cp_shard_rows` 和 KDA 重排本身可能不用改，要改的是边界（汇总 3.3、3.4） | P1 |
| `model_executor/runner/eager_runner.py` | 核对 `__init__`（120-124 行）的 `max_num_token` 对齐；处理 MIXED 模式下 1 个 token 的 extend 段 | 汇总 3.5、3.6 | P1 |
| `layers/cp/bcg/interleave_bcg.py` | 第一期不改，prefill graph 被拒绝 | — | P3 |

### 4.4 KV 写入

| 文件 | 修改内容 | 目的 | 阶段 |
|---|---|---|---|
| `mem_cache/kv/alloc/id_space.py` | **不改**；加单元测试，确认 `translate_write` 在方案 A 下每个位置只写一次 | P1 依赖它的 owner 过滤（汇总 4.1） | P1 |
| `layers/attention/dsa/dsa_indexer.py` | P1：保留第 1094-1095 行的 CP gather，确认融合 store 的 owner 过滤正确；P3：去掉 gather，写入时用本卡行的 `out_cache_loc` | 汇总 4.1、4.3 | P1 / P3 |
| `layers/cp/interleave.py` 的 `materialize_full_mla_kv` / `materialize_full_indexer_k_cache` / `run_indexer_with_kv_gather` | P1 保留；P3 改成只写本地 | 汇总 4.1 | P3 |
| `layers/dcp/planner.py` | 不改；保证第 117 行要求的"每请求前缀长度是 dcp 的倍数"，也就是 chunk 边界按放大页对齐 | 汇总 4.6 | P1 |

### 4.5 prefill attention

| 文件 | 修改内容 | 目的 | 阶段 |
|---|---|---|---|
| `layers/attention/dsa_backend.py` | ① `dcp_gather_kv_cache`（1392 行）的本地行和 gather 长度，都按**完整请求列表**构建，不按 CP 过滤；② `topk_indices_offset`、`cu_seqlens_k`、`_cal_indexer_k_start_end` 和 `get_indexer_metadata` 的 `ks`/`ke`：query 侧按本卡行，指向 gather 后的完整布局；③ `_forward_trtllm`（4508-4556 行）加 guard，方案 A 下不做 CP 的 KV gather | 避免各卡形状不一致导致算错或挂起（汇总 5.1、5.2、5.3、4.4） | P1 |
| `layers/attention/dsa/dsa_indexer.py` | `_dcp_gather_index_k`（1925 行）的长度和 `dsa_backend.py` 保持一致；保留第 1807、1826 行的长度断言作为检查点 | 汇总 5.1、5.3 | P1 |
| `models/deepseek_common/attention_forward_methods/forward_mla.py` | 方案 A 下关闭第 472-480 行 `cp_overlap` 的构造、第 866 行 `prepare_kv_for_attention` 里的 CP gather，以及第 666-675 行 indexer 的 CP overlap。**不改** `bcg_dcp_extend_kv_gather` 和 `is_dcp_mla_decode_phase` | 避免 CP 和 DCP 两条 KV 路径重复通信（汇总 4.5、5.4、5.5、5.6） | P1 |
| `models/kimi_k3.py` 的 `prepare_context_parallel_metadata_for_dcp`（两层：语言模型和多模态外壳） | 签名加上 CP 元数据，或者加上 `forward_batch` | 合并后的元数据需要本卡行的信息（汇总 5.7） | P1 |
| `model_executor/runner/eager_runner.py` | 两处调用 `prepare_context_parallel_metadata_for_dcp` 的地方传入 CP 元数据；顺序已经是 CP 在前，不用改。新逻辑放进 `layers/cp/` 或 `layers/dcp/` 的协作者里，这里只做调用 | 汇总 5.7、5.8、5.9 | P1 |
| `layers/dcp/metadata.py` 和 `layers/cp/base.py` | 新增合并后的元数据，用 `msgspec.Struct` 定义，**不能设 `frozen=True`**，也不能继承现有的 dataclass | 汇总 3.8 | P1 |
| `model_executor/runner/prefill_cuda_graph_runner.py` | 第一期不改。如果 P3 开启 graph，要保留第 1511 行"有前缀命中就走 eager"的规则 | 汇总 8.3 | P3 |

### 4.6 decode attention

| 文件 | 修改内容 | 目的 | 阶段 |
|---|---|---|---|
| `layers/cp/cp_decode_attn_tp.py` | D2：把 K3.1 加入白名单，并把生效范围收窄到和 `is_dcp_mla_decode_phase` 一致；或者 D3：新增只切 o_proj 和 `g_proj` 的模式 | §3.5（汇总 6.2、6.6） | P2 |
| `models/kimi_k3.py` | `_run_self_attn`（3312 行）加 `maybe_use_decode_attn_tp`；`g_proj` 加入切片；`_stage_facts` 补上延迟求和的声明；修改 `post_load_weights`（4101-4102 行）中 `_prepare_dcp_v_weights` 的调用位置 | 汇总 6.1、6.3、6.4、6.5、6.8 | P2 |
| `models/deepseek_v2.py`（**新增**） | 修正 `attn_mqa_for_dcp_decode`（2261 行）在 attn_tp=1 时的头数；确认 `maybe_use_decode_attn_tp`（2341-2366 行）对 K3 子类生效 | 汇总 6.2、6.4 | P2 |
| `layers/attention/dsa_backend.py` | 修正 `__init__`（755-758 行）的 `num_q_heads` 等头数公式 | 汇总 6.4 | P2 |
| `models/deepseek_common/attention_forward_methods/forward_mla.py` | D1 下跳过第 878 行的 Q all-gather，并在合并后 all-gather 头；D2 下确认合并后的头布局和切片一致 | 汇总 6.4、6.7 | P2 |

### 4.7 显存核算

| 文件 | 修改内容 | 目的 | 阶段 |
|---|---|---|---|
| `model_executor/pool_configurator.py`（**新增**，替代第一版的 `kv_cache_configurator.py`） | 确认第 452-462 行在方案 A 下按 `attn_dcp_size` 计算每 token 字节数（DSA indexer 在 `dsa_shards_index_k()` 为真时不乘 dcp） | 汇总 9.1 | P1 |
| `mem_cache/kv_cache_configurator.py` | 确认 `_build_hybrid_linear_kv_pool` 路径正确；`dcp_enabled` 为真，所以不会选成逐 token 分配器（2937 行）；KDA 状态池按整个 TP 组的头切分（每卡 12 个头） | 汇总 9.2、9.3、9.4 | P1 |

### 4.8 PD 分离（P4）

| 文件 | 修改内容 | 目的 | 阶段 |
|---|---|---|---|
| `disaggregation/common/conn.py` | rank 身份**基本不用改**（汇总 10.1）；注册信息和 `PrefillServerInfo` 加上 `dcp_size`/`dcp_kv_layout`，在 `try_ensure_parallel_info` 里校验；修正混用"真实 attention TP（1）"和"注册 TP（8）"的三处（`_validate_envelope_kv_layout`、`_prefetch_staging_reqs`、`_validate_peer_state_layout`） | decode 的 DCP 大小不同时能提前拒绝，而不是让整个会话失败（汇总 10.2、10.6） | P4 |
| `disaggregation/mooncake/conn.py` | 按页传输时校验两边的 `kv_item_lens`、`token` 布局，以及 `index_slice` 和 `decode_prefix_len` 是否按 DCP 虚拟页对齐；校验 draft KV 的布局 | 防止数据写错位置而不报错（汇总 10.3、10.7） | P4 |

### 4.9 HiCache（P4）

| 文件 | 修改内容 | 目的 | 阶段 |
|---|---|---|---|
| `mem_cache/hicache_storage.py` | 第 68-73 行和 `is_storage_writer`（88-98 行）改成按 CP rank 的语义（典型配置下数值不变）；决定 `_init_config_suffix`（852-876 行）在方案 A 下是否去掉 CP 段；只支持普通 key scheme | 汇总 11.1–11.3 | P4 |
| Mooncake 等 L3 后端 | 检查 `dcp_rank`、`is_storage_writer` 和 `l3_page_span` 的用法 | 汇总 11.4 | P4 |
| `mem_cache/pool_host/base.py` | 不改（`dcp_rank` 由调用方传入） | 汇总 11.5 | — |

### 4.10 不需要修改、但要加测试兜底的文件

| 文件 | 原因 |
|---|---|
| `layers/dcp/planner.py`、`layers/dcp/metadata.py`（除合并元数据以外） | DCP decode 逻辑不变 |
| `models/kimi_k3.py` 的 `_forward_cp_shard_rows`、KDA 重排调用点 | 和行的位置无关，或者已经支持任意排列（推断） |
| `forward_mla.py` 的 `bcg_dcp_extend_kv_gather`、`is_dcp_mla_decode_phase` | K3.1 不经过它们，或者判断已经正确 |
| `mem_cache/kv/alloc/id_space.py` | owner 过滤已经满足要求 |

---

## 5. 阶段划分

| 阶段 | 内容 | 验收标准 |
|---|---|---|
| **P0：确认收益** | 不改代码。在 B300 8 卡上，对比 TP8、TP8+PCP8、TP8+DCP8 的 TTFT（8K、32K、128K 输入，关闭 radix cache）和 TPOT；确认业务是否需要统一部署 | 方案 A 的预期收益 = TP8+PCP8 相对 TP8 的 TTFT 收益（扣除方案 A 额外的整序列 gather），同时 KV 容量接近 TP8+DCP8。**收益低于约定阈值（例如 15%）就停止** |
| **P1：prefill 正确性** | §4.1–4.5、§4.7；只用 eager 模式、统一部署、K3.1、`flashmla_kv`；KV 写入保留 gather + owner 过滤 | 每层 hidden 和 logits 与 TP8+DCP8 对比在容差内；覆盖带前缀、多请求、前缀长度非 N 倍数、MIXED 模式 |
| **P2：decode** | 先做 D2 可行性实验（§3.5），再按结论实施 §4.6 | sgl-eval（gsm8k 和一个长上下文任务，K3.1 推荐的采样参数）精度和 TP8+DCP8 一致，报告时写明 harness 和参数 |
| **P3：性能** | 只写本地（去掉新 token 的 gather）；评估是否开启 prefill graph；通信和计算重叠 | TTFT 优于 TP8+DCP8。**TPOT 单独评估**，不再要求和 TP8+DCP8 持平（见风险 R12、R13） |
| **P4：周边功能** | §4.8、§4.9；投机解码（DFLASH、MTP） | PD、HiCache、投机解码的端到端测试 |

粗略工作量（推断）：P1 约 3–4 周，P2 约 2–3 周（含实验），P3 约 2 周，P4 另算。比第一版估计的 6–8 周略多，主要增加在 decode 的改动和元数据合并。

---

## 6. 测试计划

1. **参数校验**：方案 A 的合法组合能启动；以下组合要报错：zigzag、`attn_cp != dcp`、HiCache、显式指定 prefill graph、投机解码、PD、非 `flashmla_kv` 后端。GLM-5 Next 和 Qwen4-Exp 仍然要拒绝 DCP。
2. **单元测试**：
   - 断言 `dcp_rank == attn_cp_rank`，`_DCP` 的成员等于 `_ATTN_CP`；
   - 按位置取模的切分和还原，覆盖带前缀、多请求、前缀长度非 N 倍数、1 个 token 的 extend、padding；
   - `translate_write` 和 indexer 的融合 store 在方案 A 下每个位置只写一次；
   - DCP gather 的输入形状在各卡一致（对应风险 R2）。
3. **数值等价**：同一批输入，对比 TP8+DCP8 和方案 A 每层的输出和 logits，分别测有前缀命中和没有命中的情况。
4. **decode 实验（P2 前）**：2 卡、一层模型，对比 TP2+DCP2 和方案 A + D1/D2 的 attention 输出与 o_proj 的输出。
5. **精度**：按 `.claude/rules/prefer-sgl-eval.md` 的规范用 sgl-eval，只和相同 harness、相同参数的结果比较。
6. **性能**：TTFT p50/p90、TPOT、`max_total_num_tokens`，对比 TP8、TP8+PCP8、TP8+DCP8 和方案 A。
7. **回归**：`test/registered/cp/` 和 DCP 相关的单元测试全部通过。

---

## 7. 风险清单

按"影响 × 可能性"排序。"类型"一列：正确性 / 稳定性 / 性能 / 工程。

| # | 风险 | 类型 | 说明 | 应对 | 来源 |
|---|---|---|---|---|---|
| R1 | **decode 时 o_proj 的部分和没有被求和** | 正确性（静默错误） | K3 的 `_stage_facts` 没有声明延迟求和；方案 A 下 `defer_cp_decode_attn_tp` 会为真 | 补上声明；用 P2 的数值实验确认 | 汇总 6.3 |
| R2 | **prefill 时 DCP gather 的形状在各卡不一致** | 正确性 / 稳定性（挂起） | 本地行按 CP 过滤后的请求算，gather 长度按全部请求算 | KV 侧元数据按完整请求列表构建；加形状断言 | 汇总 5.1、5.2 |
| R3 | **多处头数放大 8 倍** | 正确性（报错或静默错误） | `num_q_heads`、`attn_mqa_for_dcp_decode`、Q all-gather、`_prepare_dcp_v_weights` | 在 P2 逐项修正，并加形状断言 | 汇总 6.4、6.5 |
| R4 | **D2 方案本身可能不成立** | 正确性 / 进度 | 两份文档结论不一致 | P2 先做实验；备选 D1 或 D3 | 汇总 6.2 |
| R5 | owner 规则对不上，导致静默算错 | 正确性 | `attn_dcp_rank` 必须等于 `attn_cp_rank`；按位置取模的切分要正确 | 断言、不变量检查（每个 slot 只写一次，gather 行数等于序列长度） | 汇总 2.1、5.10 |
| R6 | CP 和 DCP 两条 KV 路径重复通信或重复写入 | 性能 / 正确性 | `forward_mla.py` 和 `_forward_trtllm` 里的 CP KV gather 没有关掉 | 按 §3.4 关闭，并加 guard | 汇总 4.4、4.5 |
| R7 | **收益不足** | 工程 | TTFT 的净收益可能只有个位数百分比，而且上下文越长越小 | 由 P0 决定是否继续 | 第一版 §7 |
| R8 | prefill 时整序列 KV gather 的代价随前缀增长 | 性能 | 方案 A 要 gather 包括前缀在内的整个序列，现在的 PCP 只 gather 新 token | P3 做通信重叠；评估只在短前缀时启用 | 第一版 §7 |
| R9 | 显式指定的 prefill graph 绕过自动关闭规则 | 稳定性 | breakable 或 full graph 加 DCP 可能 KeyError | P1 显式报错 | 汇总 1.10、8.2 |
| R10 | hook 执行顺序导致放行判断失效 | 正确性（误放行或误拒绝） | 第 266 行运行时 `cp_strategy` 未设置；`handle_hicache` 运行时 `enable_cp_tp_group_sharing` 未声明 | 判断条件只用最早可用的字段；在后续 hook 里再做最终检查 | 汇总 1.1 |
| R11 | 误放行其他模型 | 正确性 | `cp_tp_group_sharing_hook.py:79` 是三个模型共用的 | 只对 K3.1 放行 | 汇总 1.2 |
| R12 | **方案 A 会失去融合 all-reduce 和 SP-MoE** | 性能 | 它们要求 attn_tp > 1 | P3 单独评估 TPOT；考虑为方案 A 补充融合（可选） | 汇总 7.1 |
| R13 | 方案 A 会失去 SM100 的 prefill token packing | 性能 | DCP 加 prefill CP 时会关闭（`dsa_backend.py:811`） | 计入 P3 的 TTFT 评估 | 汇总 7.2 |
| R14 | MegaMoE 的 token 上限估算偏小 | 稳定性（运行时越界） | 每请求单独补齐后，单卡最多约 `tokens/N + 请求数` | 修改估算，加运行时断言 | 汇总 1.8 |
| R15 | KDA 状态池按错误的头数计算 | 显存（OOM 或浪费） | 必须按整个 TP 组切分（每卡 12 个头），不能按 attn_tp=1 | 加单元测试，核对启动日志 | 汇总 9.3 |
| R16 | chunk 边界没有按放大页对齐 | 正确性 | `planner.py:117` 和 PD 的按页传输都依赖这个对齐 | 调度时强制对齐，加断言 | 汇总 4.6、10.7 |
| R17 | `deepseek_v2.py` 的 family override 和方案 A 冲突 | 稳定性（启动失败） | `tp_size <= 8` 的断言、`index_kpool` 检查 | P1 先在这个文件上验证 | 汇总 1.4 |
| R18 | `attn_dcp_*` 约 157 处使用点隐含"attention TP 组内"的假设 | 正确性 | 字段语义和方案 A 不符 | P1 逐处排查 | 汇总 2.5 |
| R19 | PD 分离时 decode 的 DCP 大小不匹配，整个会话失败 | 稳定性（P4） | 注册信息里没有 `dcp_size` | P4 加字段并提前校验；第一期拒绝 PD | 汇总 10.2 |
| R20 | PD 按页传输把数据写错位置，不报错 | 正确性（P4） | 缺少布局和对齐校验 | P4 加校验 | 汇总 10.3、10.7 |
| R21 | 混用 attention TP（1）和注册 TP（8） | 稳定性（P4） | unified memory、staging、DFLASH 布局校验会出错 | P4 逐处修正；第一期拒绝这些功能 | 汇总 10.6 |
| R22 | HiCache 的 L3 key 不能和 TP8+DCP8 共享 | 运维（P4） | 后缀同时带 CP 段和 DCP 段 | P4 决定后缀规则 | 汇总 11.2 |
| R23 | MTP draft 在 DCP 下直接报错 | 功能（P4） | `_build_mtp_kv_shard_pool` | 第一期拒绝；P4 再处理 | 汇总 1.11 |
| R24 | 两套元数据的合并和上游重构冲突 | 工程 | 上游计划做 `DecodeContextParallelStrategy` | 先和上游沟通 RFC；合并后的元数据设计尽量向上游靠拢 | 第一版 §7 |
| R25 | 改动面大，以后合并上游很困难 | 工程 | 涉及 20 多个文件、多个后端 | 按阶段推进；P1 只支持一个后端；新逻辑放进协作者，核心文件只加调用 | 第一版 §7 |
| R26 | 非 CUDA 平台和 fp8 fnuz 平台不支持 | 功能 | DCP 分片的 indexer K 只能由 CUDA 融合 store 写入 | 参数校验阶段拒绝 | 汇总 5.10 |

---

## 8. 待决策的问题

1. P0 的收益阈值定为多少？如果达不到，是否改为扩展 `--enable-kv-cache-sharding`，或者只用 PD 分离？
2. decode 选 D1、D2 还是 D3？（P2 实验后决定）
3. P1 是否接受"保留 gather + owner 过滤"的过渡做法？
4. 方案 A 是否需要支持 PD 分离？如果需要，Decode 实例的 DCP 是否必须和 Prefill 相同？
5. HiCache 的 L3 key 是否需要和 TP8+DCP8 部署共享？
6. 是否先向上游提交 RFC？

---

## 9. 参考

- 第一版计划：`PCP_DCP_方案A_计划.md`
- 问题汇总：`docs/understand/方案A计划问题汇总.md`
- 代码理解文档清单：`docs/understand/README.md`（27 份，按"路径中 `/` 换成 `__` 再加 `_代码理解.md`"命名）
