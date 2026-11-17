# `python/sglang/srt/layers/cp/cp_decode_attn_tp.py` 代码理解

- 文件路径：`python/sglang/srt/layers/cp/cp_decode_attn_tp.py`
- 行数：319
- 所属模块：`--enable-cp-decode-attn-tp` 的实现。PCP 下 attention 权重是复制的（attn_tp_size=1），decode 时按 CP rank 临时切出一份权重，让 decode 的 attention 像普通 TP 一样按头切分。

## 1. 整体功能

PCP 为了让每张卡都能算全部头，把 attention 线性层设成复制（`attn_tp_size = 1`）。decode 阶段 CP 不生效，如果什么都不做，每张卡都会冗余地算全部头。这个文件提供一个 context manager，在 decode 前向期间：

1. 把传入的 `ColumnParallelLinear`（沿输出维 dim 0）和 `RowParallelLinear`（沿输入维 dim 1）的权重和 scale 替换成本 CP rank 的切片，并把 `output_size_per_partition` / `input_size_per_partition` 缩小为 1/N；
2. 对模型额外传入的张量属性（如 MLA 的 `w_kc`、`w_vc`）同样切片；
3. 把 `RadixAttention.tp_q_head_num` 缩小为 1/N；
4. 对 `RowParallelLinear` 设置 `use_decode_attn_tp`，让它在行并行 GEMM 后做 inline all-reduce；或者在 `defer_all_reduce=True` 时把求和交给层边界。

退出时全部恢复。切片结果按 `(id(obj), attr_name)` 缓存，避免重复计算，也保证 CUDA graph 里的指针稳定。因为缓存的切片和录好的 graph 都引用旧权重，`configure_cp_decode_attn_tp` 会在构造时给 o_proj 设置 `_derived_weight_cache_error`，拒绝在线权重更新。

切片规则是连续切块：第 r 个 rank 取 `[r * chunk, (r + 1) * chunk)`（`_slice`，第 137 行）。

## 2. 主要依赖与调用方

依赖：`layers/attention/dsa/utils.py`（`dsa_use_prefill_cp`）、`layers/cp/utils.py`（`is_cp_active`）、`layers/radix_attention.py`、`layers/linear.py`（`ColumnParallelLinear`、`RowParallelLinear`）、`runtime_context`；可选依赖 `deep_gemm.utils.get_tma_aligned_size`、`aiter.ops.shuffle`、`layers/quantization/fp8_utils.unshuffle_aiter_fp8_weight`。

调用方（grep 确认）：
- `CP_DECODE_ATTN_TP_SUPPORTED_ARCHS`：`arg_groups/model_hook.py:203-211`，开启 `--enable-cp-decode-attn-tp` 时检查模型架构，不在白名单就 `ValueError`。
- `can_defer_cp_decode_attn_tp`：`models/deepseek_v2.py:2113, 2796`、`model_executor/runner/base_cuda_graph_runner.py:90`。
- `configure_cp_decode_attn_tp`：`models/deepseek_v2.py:2226`（MLA 的 o_proj）、`deepseek_v2.py:2839`。
- `get_cp_decode_attn_tp_ctx().maybe_use_decode_attn_tp`：`models/deepseek_v2.py:2341-2366`（`DeepseekV2AttentionMLA.maybe_use_decode_attn_tp`，传入 `[q_b_proj, o_proj]`、`w_kc`/`w_vc`/`w_scale_k`/`w_scale_v` 和 `attn_mqa`）、`deepseek_v2.py:2912`；`models/deepseek_v4.py:1348`、`deepseek_v4_platform_stage.py`、`xing4_0.py:377`。
- `KimiK3MLAAttention`（`models/kimi_k3.py:2745`）继承 `DeepseekV2AttentionMLA`，所以继承了上面的 `maybe_use_decode_attn_tp` 和 o_proj 的 `configure_cp_decode_attn_tp` 调用；但 K3.1 不在白名单，CLI 校验会先拒绝。

## 3. 类与函数说明

### 3.1 模块级

| 名称 | 行号 | 作用 |
|---|---|---|
| `CP_DECODE_ATTN_TP_SUPPORTED_ARCHS` | 29 | 白名单：DeepSeek-V4（含 NextN、DSpark）和 GLM-5.x DSA（`GlmMoeDsaForCausalLM` 及 NextN）。注释说明只有 CP 下 attention 线性层以 tp_size=1 复制的模型才能用，这里是唯一的架构字符串来源。 |
| `_global_cp_decode_attn_tp_ctx` | 39 | 上下文单例。 |
| `can_defer_cp_decode_attn_tp` | 42 | 构造时决定 DSA 层能否把 decode 的求和推迟到 stage 边界：要求开启该功能和 prefill CP、`attn_cp_size == tp_size > 1`、`attn_tp_size == 1`、`attn_dp_size == 1`、未开 layernorm SP、CUDA 设备、无 LoRA、无 two-batch overlap、无量化通信。 |
| `get_cp_decode_attn_tp_ctx` | 65 | 返回（必要时创建）`CpDecodeAttnTpContext` 单例。 |
| `configure_cp_decode_attn_tp` | 73 | 构造时校验 o_proj：`defer_all_reduce=True` 但 `reduce_results=True` 时 `ValueError`。功能开启且 `attn_cp_size > 1` 时设置 `_derived_weight_cache_error`，拒绝在线权重更新（缓存的切片和 CUDA graph 会保留旧权重）。 |

### 3.2 `CpDecodeAttnTpContext`（第 93 行）

| 方法 | 行号 | 作用 |
|---|---|---|
| `__init__` | 96 | 功能开启且 `attn_cp_size > 1` 时，`decode_tp_rank = attn_cp_rank`、`decode_tp_size = attn_cp_size`；否则两者为 None。初始化 `use_decode_attn_tp = False` 和 `_slice_cache`。 |
| `is_enabled`（property） | 111 | `decode_tp_size` 不为 None 且 > 1。 |
| `set_decode_attn_tp` | 114 | 决定这次前向是否切片：只要不是 CP 激活的 prefill（`is_cp_active`）且不是 DSA prefill CP（`dsa_use_prefill_cp`），就启用。注意这包括 decode 以外的其他模式（如 target verify、未走 CP 的短 extend）。 |
| `_slice` | 124 | 沿 dim 0 或 1 取本 rank 的连续块；维度不合法或不能整除 `decode_tp_size` 时 `ValueError`。对 DeepGEMM UE8M0 打包的 int32 二维 scale，重新按 TMA 对齐的 MN-major 布局拷贝一份（不解包、不重新量化）。dim 1 的切片会 `contiguous()`。 |
| `_slice_aiter_bpreshuffled_weight` | 156 | AITER B-preshuffle 的权重：先反 shuffle 回逻辑布局，沿 K（dim 1）切片，再 shuffle 回去。 |
| `_activate` | 169 | 把 `obj.attr_name` 替换成切片：属性为 None 时跳过；不能沿 `dim` 切片时 `ValueError`；第一次调用计算切片并缓存 `(原张量, 切片, 是否 Parameter)`；Parameter 改 `.data`，普通张量用 `setattr`。 |
| `_restore` | 204 | 从缓存取回原张量，恢复属性。 |
| `_get_linear_attrs` | 216 | 返回线性层需要切的 `(obj, attr, dim)` 列表：`RowParallelLinear` 用 dim 1，`ColumnParallelLinear` 用 dim 0，其他类型返回空。也包括 `weight_scale_inv` / `weight_scale`；对行并行的 UE8M0 打包 scale，要求 `weight.shape[1]` 能被 `decode_tp_size * 512` 整除，否则 `ValueError`。 |
| `maybe_use_decode_attn_tp` | 246 | context manager 主体。先 `set_decode_attn_tp`，不启用就直接 yield。启用时依次切片 `modules` 里的线性层并缩小 partition 大小、设置 `RowParallelLinear.use_decode_attn_tp = not defer_all_reduce`（`defer_all_reduce` 且 `reduce_results=True` 时 `ValueError`）、切片 `tensor_attrs`、缩小 `radix_attn.tp_q_head_num`；`finally` 里按相反顺序全部恢复。 |

## 4. 与 PCP+DCP 方案 A 的关系

计划 §3.5 的冲突处理推荐 D2：复用 `--enable-cp-decode-attn-tp`，decode 时把复制的 o_proj 按 CP rank 切头再 all-reduce；§4.6 要求把 `KimiK31ForConditionalGeneration` 加入 `CP_DECODE_ATTN_TP_SUPPORTED_ARCHS`，并适配 K3 的 o_proj 包装和输出门 `g_proj`。

**读代码发现的问题（计划未提及）：**

1. **现有调用会切 Q 头，和 DCP decode 冲突。** K3.1 的 `KimiK3MLAAttention` 继承的 `DeepseekV2AttentionMLA.maybe_use_decode_attn_tp`（`models/deepseek_v2.py:2341-2366`）不只切 o_proj，还切 `q_b_proj`、`w_kc`、`w_vc`、`w_scale_k`/`w_scale_v`，并把 `attn_mqa.tp_q_head_num` 和 `num_local_heads` 缩为 1/N。这是"按头切 attention"。DCP decode 要求每张卡对**全部头**、本地 1/N 的 KV 算部分输出，再按 LSE 合并后每张卡拿到 1/N 的头（计划 §3.5 第 1、4、5 步）。如果直接把 K3.1 加入白名单，每张卡只算 1/N 的头，LSE 合并的输入就不完整，结果错误（推断）。方案 A 需要一个"只切 o_proj（和 `g_proj`），不切 Q/`w_kc`/RadixAttention 头数"的变体，或者给 `maybe_use_decode_attn_tp` 加一个只切输出侧的模式。计划里"正好对上 DCP 合并的结果"这句话只对 o_proj 成立。
2. **`w_vc` 和 V 预投影。** DCP 合并前可能先用 `w_vc` 做 V 预投影（`forward_mla.py:1043-1069` 的 `dcp_v_preproject` 分支，用 `_prepare_dcp_v_weights(dcp_group)` 准备权重）。这条路径的 `w_vc` 按 DCP 组准备；如果 `maybe_use_decode_attn_tp` 同时切了 `w_vc`，两者会叠加。计划 §4.6 提到"`dcp_v_preproject` 要按切片后的头准备权重"，这里补充：要明确只由一方负责切 `w_vc`。
3. **头的切片顺序。** `_slice`（137）取连续块 `[r * chunk, (r + 1) * chunk)`。DCP 的 a2a / push-combine 合并后每个 rank 拿到哪些头，需要确认也是按 `dcp_rank` 的连续块（推断，未逐行核对 `dcp_a2a_lse_reduce`），且方案 A 下 `dcp_rank == attn_cp_rank`，否则 o_proj 切片和合并结果的头对不上。计划 §4.6 "确认合并后的头布局和 cp-decode-attn-tp 的切片一致"说的就是这一点。
4. **生效范围不只是 decode。** `set_decode_attn_tp`（114）在"不是 CP prefill"的所有前向上都启用，包括 target verify、draft extend，以及总长度小于 `cp_size` 而退回非 CP 的短 extend。而 `is_dcp_mla_decode_phase`（`forward_mla.py:196-211`）在 decode、target verify、draft extend v2 上走 DCP 合并。两者的覆盖范围需要逐个模式对齐；尤其短 extend 回退时，K3.1 的 DCP extend 走"gather KV 后直接 attention"（`dsa_shards_index_k()` 为真），不经过 LSE 合并，此时不应切 o_proj（推断）。
5. **`can_defer_cp_decode_attn_tp`（42）的条件** 要求 `attn_cp_size == tp_size`、`attn_tp_size == 1`，方案 A 的典型配置满足。但 K3 自己的 o_proj 包装（`kimi_k3.py` 里的对称内存 all-reduce 融合 `_k3_symm_o_proj_out`、输出门 `_gated_o_proj_forward`、SP-MoE 路由 `_route_sp_o_proj_output`）都会替换 `o_proj.forward`，`_activate` 只替换权重属性，不影响这些包装，但包装里按 `o_proj.weight.shape[0]` / `output_size` 分配输出 buffer 的逻辑要确认在切片后仍正确（推断）。
6. **`g_proj`。** K3 的 `g_proj` 是 `ColumnParallelLinear`（`kimi_k3.py` 中 `KimiK3MLAAttention.__init__`），`_get_linear_attrs` 支持按 dim 0 切它；只需在 K3 的 `maybe_use_decode_attn_tp` 里把它加入 `modules`。输出门在 o_proj 之前、TP 本地头空间里乘上，切片后门的头和合并后的头必须同序。
