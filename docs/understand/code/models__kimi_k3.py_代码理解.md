# `python/sglang/srt/models/kimi_k3.py` 代码理解

- 文件路径：`python/sglang/srt/models/kimi_k3.py`
- 行数：4725 行（9 个类、27 个顶层函数、106 个方法）
- 所属模块：`sglang.srt.models`，Kimi-K3 / Kimi-K3.1 的模型实现（多模态外壳 + KimiLinear 风格文本主干），通过 `EntryClass` 注册 `KimiK3ForConditionalGeneration`、`KimiK31ForConditionalGeneration`、`KimiK3LinearForCausalLM` 三个架构。

---

## 1. 整体功能

### 1.1 层级结构

```
KimiK3ForConditionalGeneration / KimiK31ForConditionalGeneration（多模态外壳）
├── vision_tower: KimiK3VisionTower（MoonViT3d，来自 kimi_k3_vl.py，复制、按图片做 DP）
├── mm_projector: KimiK3MultiModalProjector
└── language_model: KimiK3LinearForCausalLM（文本 CausalLM）
    ├── model: KimiK3LinearModel（backbone）
    │   ├── embed_tokens（VocabParallelEmbedding，PP 首段）
    │   ├── attn_bank: AttnBank（Attention Residual 的块残差库）
    │   ├── layers[i]: KimiK3DecoderLayer
    │   │   ├── self_attn: KimiK3DeltaAttention（KDA 层）或 KimiK3MLAAttention（MLA / DSA 层）
    │   │   ├── mlp: KimiK3MoE（Latent MoE）或 KimiK3MLP（dense，前 first_k_dense_replace 层）
    │   │   ├── input_layernorm / post_attention_layernorm
    │   │   └── attn_boundary / ffn_boundary（layer_boundary 的阶段边界，负责残差、norm、集合通信）
    │   └── norm + _final_read（AttnBankOutputRead 或普通 RMSNorm，PP 末段）
    ├── lm_head（ParallelLMHead）
    └── logits_processor
```

文件开头的注释列出了 K3 相对 `kimi_linear.py` 的特性：Attention Residual（`attn_res_block_size`）、Latent MoE（`routed_expert_hidden_size`）、SiTU 激活、MLA 输出门（`mla_use_output_gate`）、KDA 全秩门（`use_full_rank_gate`）。K3.1 额外在 MLA 层上带 DSA indexer（`index_topk`），结构为 93 层：69 层 KDA + 24 层 Gated Sparse-MLA，1 层 dense + 92 层 Latent MoE。

### 1.2 各模块职责

- **`KimiK3MLP`**：`gate_up_proj`（列并行）+ SiLU/SiTU + `down_proj`（行并行，`reduce_results=False`，求和交给阶段边界）。既用于 dense 层，也作为 MoE 的 shared experts。
- **`KimiK3MoE`**：路由门 `MoEGate`（fp32 logits）+ `TopK` + 路由专家（`get_moe_impl_class`）。Latent MoE 下先用 `routed_expert_down_proj` 把 hidden 压到 `moe_hidden_size`，专家在潜空间里算，再经 `routed_expert_norm` 和 `routed_expert_up_proj` 回到 hidden。shared experts 用原始 hidden。前向有三条主路径：
  - `_forward_fused`：纯 TP（无 a2a）时，把 shared gate_up、router gate、latent down 三个权重合并成一次 GEMM（`_front_w`），潜空间和 shared 的 TP 部分和放进同一块对称内存一次 all-reduce；可选 TP 切分的 front（`_front_shard`）和 K3 自己的融合 AR+RMSNorm（`k3_ar_fusion`）。
  - `_forward_unfused`：EP a2a（DeepEP / MegaMoE / Mooncake 等）或不满足融合条件时，三个 GEMM 分开算；shared experts 可以在侧流上和路由 a2a 重叠（SBO），或走 shared-expert TP 子组的 AG/RS，或走权重流（`SharedWeightStream`）。
  - `_forward_cp_shard_rows`：CP-TP group sharing 下的 CP extend，输入是按 CP rank 顺序 gather 来的全部行，输出只返回本 rank 的 CP 行（reduce-scatter）。
- **`KimiK3DeltaAttention`**（KDA）：线性注意力。投影 q/k/v/g（全秩门下合并为 `fused_qkvg_proj`）、beta（`b_proj`）、遗忘门（`f_a_proj` → `f_b_proj`），交给 `RadixLinearAttention` 做短卷积 + 递推，再用 `FusedRMSNormGated`（`o_norm`）和 `o_proj`。头按 `linear_attn_parallel_group()` 切分：普通情况是 attention-TP 组，CP-TP group sharing 下是整个 TP 组。
- **`KimiK3MLAAttention`**：继承 `DeepseekV2AttentionMLA`。在其上增加：输出门 `g_proj`（通过实例级包装 `o_proj.forward` 在 o_proj 前乘 `sigmoid(gate)`）、对称内存 o_proj 输出（`all_reduce_fusion`）、SP-MoE 的 o_proj 输出路由、GGUF 拆分 K/V 的加载、`forward_pre_o_proj`（跳过 o_proj 把门控后的输入交回调用方）。K3.1 的 DSA 层在 DCP 下调用 `set_dsa_shards_index_k()`。类属性 `dsa_decode_absorb_under_indexer = True`、`dcp_v_preproject = True` 打开 `forward_mla.py` 中的两条优化。
- **`KimiK3DecoderLayer`**：选择 attention 和 FFN 类型，构造 Attention Residual 的 `res_norm` / `res_proj`，用 `_declare_stages` 生成 `attn_boundary` / `ffn_boundary`。`forward` 依次执行 boundary.prepare → attention → boundary.finish → FFN boundary.prepare → MLP/MoE → finish。
- **`KimiK3LinearModel`**：embedding、3 条侧流、`AttnBank`、`make_pp_layers` 建层（同时传入 `_stage_facts` 让框架预知每层的阶段声明）、最终读出；`forward` 负责 PP 衔接、bank 的打开/关闭、DSpark/DFLASH 的辅助隐状态捕获。
- **`KimiK3LinearForCausalLM`**：注册 K3.1 的 TGV GEMM tactic、补 `packed_modules_mapping`、构造 backbone/lm_head；提供 CP 和 DCP 的模型钩子（`prepare_cp_inputs`、`prepare_context_parallel_metadata_for_dcp`）；权重加载与后处理。
- **`KimiK3ForConditionalGeneration`**：多模态外壳，视觉塔 + 投影 + 文本模型；图像特征物化（本地预处理、延迟 GPU/CPU 预处理）、视觉 DP；把 CP/DCP 钩子和 PP/aux 属性转发给 `language_model`。`KimiK31ForConditionalGeneration` 是空子类，只用来给 K3.1 一个独立的架构名。

### 1.3 并行方式在本文件中的体现

- **TP**：线性层通过 `parallel_group`（`"tp"` / `"attn_tp"` / `"replicated"` / `"shared_experts_tp"`）选组。dense MLP 默认整个 TP 组（`_dense_mlp_parallel_group`，NPU 可选 attn_tp）。MLA 的头和 `g_proj` 按 attention-TP 切分。各层 o_proj / down_proj 都是 `reduce_results=False`，求和由阶段边界完成；`_fuses_attn_all_reduce` 为真时（attn_tp == tp > 1 且有 attn residual 且 `k3_ar_fusion` 可用），o_proj 写进对称缓冲区，由 `_k3_all_reduce_add` 把 all-reduce 和残差加融合。
- **EP**：`_uses_ep_a2a()` 判断 MoE a2a 后端。K3 不支持在线 EPLB 和冗余专家（`KimiK3MoE.__init__` 直接报错）。MegaMoE（`_forward_mega_experts`）对所有 batch 都走 deep_gemm 融合 a2a，要求 `SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK` 覆盖每 rank 行数。
- **SP-MoE**：EP a2a + attention TP > 1 时（`_shards_moe_rows`），MoE 层只处理本 rank 的 attention-TP 行切片。o_proj 输出通过 `_route_sp_o_proj_output` 写进 `k3_sp_collective` 的对称缓冲区，由 `_k3_reduce_scatter_add` 做 reduce-scatter + 残差加；`_carries_bank_slices` 决定 attention-residual bank 是否也保持在切片上（`SGLANG_K3_SP_ATTN_RES`）。
- **PCP（prefill CP，CP-TP group sharing）**：attention CP 组复用 TP 组，attention TP 宽度为 1（MLA 头复制）。本文件里的体现：
  - KDA 头改为在整个 TP 组上切分（`linear_attn_parallel_group()=="tp"`），`_stage_facts` 给 KDA 层声明 `tp_group=SumGroup.TP`；`KimiK3DeltaAttention.forward_pre_o_proj` 在递推前用 `cp_interleave_to_sequence_order` 把 rank-major 的 interleave 行恢复成序列顺序，算完再用 `cp_sequence_to_interleave_order` 还原。
  - Latent MoE 层声明 `output_on_cp_shards`，`KimiK3DecoderLayer.forward` 在 `ffn_boundary.outputs_on_cp_shards()` 为真时调用 `mlp(..., cp_shard_rows=True)`，走 `_forward_cp_shard_rows`。
  - MegaMoE 下 CP extend 用 `get_cp_strategy().moe_num_token_non_padded()` 屏蔽 padding 行。
  - `_latent_moe_writes_stream` 在 `attn_cp_size > 1` 时为假（CP extend 的 MoE 输出行不是本 rank 的残差行）。
  - `prepare_cp_inputs`：runner 切分 token 之前先完成 embedding（含多模态注入）。
- **DCP（decode CP）**：`KimiK3MLAAttention.__init__` 在 `attn_dcp_size > 1` 且有 `index_topk` 时调用 `set_dsa_shards_index_k()`（indexer K 也按 token 切分，全局 top-k 合并）；`dcp_v_preproject = True` 让 DCP LSE 合并前先做 V 投影；`post_load_weights` 在已有 `_dcp_v_weights` 时重新准备 V 权重；`prepare_context_parallel_metadata_for_dcp` 直接调用 `layers/dcp/planner.py` 的 `prepare_decode_context_parallel_metadata`（NPU 返回 None）。
- **PP**：`make_pp_layers` 只建本段的层；`KimiK3LinearModel.forward` 通过 `pp_proxy_tensors` 传递 hidden、`attn_res_bank` 和 `dspark_hidden_states`；`load_weights` 跳过不属于本段的层。
- **DP attention**：`_dp_attention`、`get_embedding_tp_kwargs()`、idle batch 在 `_run_self_attn` 直接透传；`_latent_moe_writes_stream` 在 DP attention 且非 EP a2a 时为假。

### 1.4 权重加载

`KimiK3ForConditionalGeneration.load_weights` 先用 `hf_to_sglang_mapper` 改名（`language_model.layers.` → `language_model.model.layers.`，`block_sparse_moe` → `mlp`），在流式迭代器里直接加载视觉权重，其余去掉 `language_model.` 前缀交给 `KimiK3LinearForCausalLM.load_weights`。encoder-only 模式也必须把迭代器跑完，因为视觉权重是迭代时的副作用。

`KimiK3LinearForCausalLM.load_weights` 的规则：跳过 PP 段外和截断配置外的层；`weight_packed` 改名为 `weight`；MLA 的 `q_a_proj` 和 `kv_a_proj_with_mqa` 手动拷进 `fused_qkv_a_proj_with_mqa` 的对应行（block-FP8 scale 按 128 块偏移）；`stacked_params_mapping` 处理 dense `gate_up`、KDA 的 `fused_qkvg_proj`（全秩门）或 `fused_qkvbfg_a_proj` / `fused_fg_b_proj`（低秩门）、`qkv_proj`、`qkv_conv1d`，并且只对 KDA 层生效；专家权重用 `_EXPERT_WEIGHT_NAME` 正则匹配 `FusedMoE.make_expert_params_mapping`；ModelOpt FP8_PB_WO 的 `weight_scale` 映射到 `weight_scale_inv`；`b_proj.weight_scale_inv` 所有 rank 共享一个块直接拷贝。最后 `_check_dsa_indexer_loaded` 断言每个 DSA 层正好 5 个 indexer 参数都已加载。

`post_load_weights`（也会被 Dummy/Sharded 加载器的钩子单独调用）：把 MLA 的 `kv_b_proj` 拆成 `w_kc` / `w_vc`（含 ROCm 每通道 FP8 折叠、GGUF Q4_0/Q2_K 校验）；预热 attn-res 的组合权重 `get_cw`；合并 MoE front 权重、预分配 MegaMoE 缓冲区、把 correction bias 转 fp32；KDA 合并 `[f_a | b]`、准备融合 decode kernel 参数；DSA indexer 的 `prepare_merged_k_gate()`；设置 shared-expert 权重流；预编译 KDA prefill kernel。这些都必须在 KV cache 定容和 CUDA graph 捕获之前完成。

---

## 2. 主要依赖与调用方

### 2.1 关键依赖

| 依赖 | 用途 |
|---|---|
| `models/deepseek_v2.py`：`DeepseekV2AttentionMLA`、`MoEGate` | MLA 主体（含 DSA indexer、DCP decode、absorb 等）和 MoE 路由门 |
| `models/deepseek_common/attention_forward_methods/` | `AttnForwardMethod`；`forward_mla.py` 读取 `dsa_decode_absorb_under_indexer`、`dcp_v_preproject`、`_prepare_dcp_v_weights` |
| `models/kimi_k3_vl.py` | `KimiK3VisionTower`、`KimiK3MultiModalProjector` |
| `layers/layer_boundary/`（`declare_attn`、`declare_ffn`、`append_stages`、`SumGroup`、`ReadoutFusion`、`AttnBank*`、`residual`） | 层间残差、norm、TP/CP/DP 集合通信的声明式边界 |
| `layers/cp/interleave.py` | `cp_interleave_to_sequence_order`、`cp_sequence_to_interleave_order`、`is_interleave_extend` |
| `layers/cp/base.py` | `get_cp_strategy()`（MegaMoE 的 CP padding 计数，局部 import） |
| `layers/dcp/layout.py`、`layers/dcp/planner.py` | `set_dsa_shards_index_k`、`prepare_decode_context_parallel_metadata` |
| `layers/communication/k3_ar_fusion.py`、`k3_sp_collective.py`、`k3_shared_weight_stream.py` | K3 专用的融合 all-reduce/RS/AG、对称缓冲区、shared-expert 权重流 |
| `layers/attn_residual.py` | `aggregate_stream`、`get_cw` |
| `layers/moe/`（`get_moe_impl_class`、`FusedMoE`、`TopK`、`route_quant_handoff`、`mega_moe`） | 路由专家、top-k、MegaMoE |
| `layers/radix_linear_attention.py` | KDA 的 `RadixLinearAttention` |
| `kernels/ops/...`（`moe_front`、`kda_fused_decode`、`add3`、`mla_output_gate`、`cutedsl_bf16_gemm`、`kimi_k3_tiny_gemm` 等） | 各类融合 kernel |
| `runtime_context.py` | `get_parallel()`、`linear_attn_parallel_group/size/rank` 等 |

### 2.2 调用方（grep 确认）

| 调用方 | 用到的内容 |
|---|---|
| 模型注册（`EntryClass`，`kimi_k3.py:4721`） | 三个入口类 |
| `srt/multimodal/processors/kimi_k3.py:32` | `KimiK3ForConditionalGeneration`、`KimiK31ForConditionalGeneration` |
| `srt/model_executor/runner/eager_runner.py:283-296` | `prepare_context_parallel_metadata_for_dcp`（DCP extend 元数据） |
| `srt/model_executor/runner/eager_runner.py:341-364` | `prepare_cp_inputs`（CP-TP group sharing 下由 runner 接管 CP 边界） |
| `srt/model_executor/runner/prefill_cuda_graph_runner.py:739, 1320` | `prepare_context_parallel_metadata_for_dcp` |
| `arg_groups/cp_tp_group_sharing_hook.py`、`arg_groups/model_overrides/kimi_k3.py`、`layers/cp/bcg/interleave_bcg.py:85` 等 | 以架构名字符串 `KimiK31ForConditionalGeneration` 判断能力（不 import 本文件） |
| 测试：`test/registered/unit/layers/test_kimi_attention_parallel_groups.py`、`unit/models/test_kimi_k3_*.py`、`unit/models/test_kimi_k25.py`、`kernels/ops/moe/test_moe_front.py`、`unit/model_executor/test_prefill_cuda_graph_runner_helpers.py` 等 | 直接 import 本文件的类和辅助函数 |

---

## 3. 类与函数说明

### 3.1 模块级常量与顶层辅助函数

模块级常量：`_K31_TGV_M_RANGES`（189）、`_K31_GB300_TGV_M_RANGES`（204）、`_K31_GB300_BFA_SIDE_TGV_M_RANGES`（218）是 K3.1 dense GEMM 的 TGV tactic 表；`_SHARDED_FRONT_MIN_TOKENS/MAX_TOKENS`（335-336）是 TP 切分 MoE front 的行数窗口（18-256）；`_EXPERT_WEIGHT_NAME`（179）是专家权重名正则。

| 名称 | 行号 | 作用 |
|---|---|---|
| `_k31_gb300_tuning` | 223 | 判断是否为 GB300（SM103）上的 K3.1 且 `SGLANG_K31_DECODE_GEMM_TACTICS` 开启，决定是否用 GB300 专用 tactic。 |
| `_k3_side_stream_max_tokens` | 233 | 返回 attention 侧流 GEMM（MLA `g_proj`、KDA `[f_a\|b]`）与主流重叠的最大 batch：非 Blackwell 64，GB300 K3.1 为 288，其他 128。 |
| `_k3_bfa_side_gemm` | 244 | KDA 侧流上的 `[f_a\|b]` GEMM：有 tactic 用 cutedsl TGV，否则用 `kimi_k3_tiny_gemm`。 |
| `_cdiv` | 258 | 向上取整除法。 |
| `_uses_modelopt_fp8_pb_wo` | 262 | 判断某前缀的模块在 ModelOpt 配置里是否为 `FP8_PB_WO` 量化。 |
| `_uses_split_gguf_kv_b` | 269 | 判断量化配置是否把 MLA 的 K/V 存成两个独立 GGUF 张量。 |
| `_maybe_map_fp8_pb_scale_name` | 276 | 把 `.weight_scale` 映射到参数表里存在的 `.weight_scale_inv`。 |
| `_get_k3_dense_weight` | 285 | 返回模块的稠密权重；有 block-FP8 scale 时先反量化。 |
| `_k3_bf16_gemm` | 298 | 直接对原始合并权重做 GEMM，同时尊重 `--bf16-gemm-backend cutedsl` 的 TGV 选择，支持 `out=` 和不同输出 dtype。 |
| `_is_unquantized_mergeable` | 345 | 一组权重是否都是同一种 bf16/fp16，可以安全拼接（不丢 scale）。 |
| `_merge_weights_as_views` | 354 | 沿 dim 0 拼接多个模块的权重，把各模块 `.weight` 改指向合并缓冲区的视图（额外显存约为 0），可按行数 pad 对齐。 |
| `_lead_with_rows_copy` | 376 | 新建一个缓冲区：前面放 `merged` 中某段行的拷贝，后面是完整 `merged`，并把模块权重重新指向其中。用于 TP 切分 front。 |
| `_dense_mlp_parallel_group` | 393 | dense MLP 默认用的并行组：`enable_dense_mlp_attn_tp` 且 DP attention 时为 `"attn_tp"`，否则 `"tp"`。 |
| `_add3` | 461 | 计算 `a + b [+ c]`；`c` 存在且 kernel 覆盖时用一次 3 路 JIT kernel，结果与两次加法逐位一致。 |
| `_o_proj_takes_output` | 486 | o_proj 的量化方法是否支持 `apply_into`（写入调用方提供的输出）。 |
| `_k3_all_reduce_add` | 492 | o_proj 输出在多播缓冲区时，做 K3 融合 all-reduce + 残差加；否则返回 None 让边界走默认路径。 |
| `_k3_reduce_scatter_add` | 500 | SP-MoE 下用 K3 调优的 reduce-scatter 把 o_proj 部分和散到本 rank 切片并加残差。 |
| `_route_sp_o_proj_output` | 506 | SP-MoE 且 `k3_sp_collective` 开启时，包装 `o_proj.forward`：当该行数需要对称 RS 时把输出写进持久对称缓冲区并登记。内部闭包 `_sp_o_proj_forward`（517）就是这个包装。 |
| `_k3_symm_o_proj_out` | 535 | 为 o_proj 的 TP 部分和分配 `k3_ar_fusion` 对称缓冲区切片。 |
| `_uses_latent_moe` | 543 | `routed_expert_hidden_size` 不为 None 即使用 Latent MoE。 |
| `_uses_ep_a2a` | 549 | MoE a2a 后端是否为 EP a2a 类（MegaMoE、FlashInfer MegaMoE、DeepEP、Mooncake、Ascend FuseEP、MORI）。 |
| `_is_moe_layer` | 3014 | 层号是否在 dense 前缀之后且符合 `moe_layer_freq`。 |
| `_shards_moe_rows` | 3025 | 是否 SP-MoE：EP a2a 且 attention-TP 组大于 1。 |
| `_fuses_attn_all_reduce` | 3031 | 是否把 o_proj 的 all-reduce 与残差加融合：attn_tp == tp > 1、有 attn residual、`k3_ar_fusion` 可用。 |
| `_carries_bank_slices` | 3043 | SP-MoE 连续层是否让 stream 和 attn-residual bank 保持在 attention-TP 切片上；要求从第一个 MoE 层起全是 MoE、PP=1，且投机算法不是 EAGLE3/DFLASH（DSpark 例外）。 |
| `_latent_moe_writes_stream` | 3068 | 该层 FFN 是否自己写下一条 stream（在尾部加残差）：Latent MoE、有 attn residual、不是"DP attention 且非 EP a2a"、且 `attn_cp_size == 1`。 |
| `_stage_facts` | 3084 | 生成某层的 attention / FFN 阶段声明（`declare_attn` / `declare_ffn`）：bank 读写、SP-MoE 的 tuned all-gather、KDA 在 TP 组上求和（`SumGroup.TP`）、FFN 是否 sparse、dense TP 宽度、`output_complete`、CP-TP group sharing 下的 `output_on_cp_shards`。既被层自身调用，也被 `make_pp_layers` 预先调用。 |

### 3.2 `KimiK3MLP`（407）

| 名称 | 行号 | 作用 |
|---|---|---|
| `__init__` | 410 | 构造 `gate_up_proj`（合并列并行）和 `down_proj`（行并行、不在层内求和），按 `hidden_act` 选择 SiLU 或 SiTU；未给并行组时用 `_dense_mlp_parallel_group()`。 |
| `forward` | 454 | gate_up → 激活 → down_proj，返回 TP 部分和。 |

### 3.3 `KimiK3MoE`（565）

| 名称 | 行号 | 作用 |
|---|---|---|
| `__init__` | 568 | 拒绝 EPLB/冗余专家；建 `MoEGate`、路由专家、`TopK`（必要时强制 STANDARD 输出格式）、MegaMoE 校验（SiTU beta=4.0/25.0）、shared experts（按 replicated / shared-expert TP / attn_tp / tp 选组）、SBO 与侧流通信标志、权重流占位、Latent 的 down/norm/up 投影，以及 `fuse_ar_norm`、`_gemm_ag_up_eligible` 静态判定。 |
| `_merge_front_weights` | 863 | 加载后合并 front 权重：无 a2a 时合并 `[shared gate_up, gate, latent down]`，开 `SGLANG_K3_FUSED_FRONT` 时合并 `[gate, latent down]`；清缓存属性；可切分时建 `_front_block_w` 并设置禁止在线权重更新的错误信息。 |
| `_routed_needs_reduce` | 928 | （cached）路由输出是否还需 TP all-reduce：TP > 1 且无 a2a。 |
| `_eligible_for_fused_front` | 932 | （cached）能否走 `_forward_fused`：Latent、有 shared、三路合并权重已建、无 a2a、shared down 为稠密 bf16/fp16。 |
| `_front_fp32` | 949 | （cached）front GEMM 是否以 fp32 输出，保证路由读精确 logits（非 HIP、可融合、权重 bf16）。 |
| `_front_shard` | 962 | （cached）TP 切分 front 时本 rank 负责的 `[router \| latent]` 行段 `(起始行, 行数)`，不能整除或不对齐 16B 时为 None。 |
| `_sharded_front_covers` | 982 | 当前 token 数是否落在切分 front 的窗口内且 all-gather 缓冲区装得下。 |
| `_mega_moe_buffer` | 992 | 获取（或创建）EP 组上的 MegaMoE 对称缓冲区。 |
| `_forward_mega_experts` | 1010 | 用 deep_gemm MegaMoE 一次完成 dispatch + 分组 GEMM + SiTU + combine；按 `valid_token_count` 屏蔽 padding 行（mxf4 路径），断言行数不超过每 rank 上限。 |
| `_latent_norm` | 1120 | 有 `routed_expert_norm` 时对潜空间输出做 RMSNorm。 |
| `_routing_contract_ok` | 1126 | （cached）融合 kernel 能否自己产出 top-k（sigmoid、带 bias 的非分组 top-k、无 waterfill、非 fake dispatch、非确定性推理、`moe_front` 可用）。 |
| `_ep_front_eligible` | 1166 | （cached）EP 合并 front（gate + latent down 一次 GEMM）是否可用。 |
| `_ep_front` | 1179 | EP 合并 front：策略为 `merged_fp32` 且形状被覆盖时，用 `moe_front.fused_front` 得到 top-k 和 routed_input；否则返回 None。 |
| `_ep_front_overlap` | 1231 | 策略为 `overlap` 时，在侧流上算 fp32 gate + top-k，同时主流算 latent down，返回前汇合。 |
| `_reduce_shared` | 1263 | 对 TP 切分的 shared experts 输出做 TP all-reduce。 |
| `_reduce_latent_and_shared` | 1268 | 对 `[latent \| shared]` 扁平缓冲区做一次 TP all-reduce。 |
| `_reduce_latent` | 1272 | 非融合路径：需要时先在潜空间 all-reduce，再做 latent norm（顺序不能反）。 |
| `_uses_shared_experts_side_comm` | 1280 | （static）shared-expert 组是否为独立通信器（不与 SP 的 RS/AG 共用），是才能把 AG/RS 放侧流。 |
| `_can_stream_shared_weights` | 1285 | 是否能用权重流替代 shared AG/RS：CUDA、有 shared-expert TP 通信、SBO、两个线性层都是未量化 bf16。 |
| `prefetch_shared_weights` | 1305 | 在 attention 之前预取本层 shared 权重的 staging 拷贝（只是提示）。 |
| `_gather_shared_expert_inputs` | 1314 | 分配输出并在 shared-expert TP 子组内 all-gather 输入行。 |
| `_alloc_shared_gather_output` | 1328 | 分配 AG 输出：侧流通信时用普通内存，否则用对称内存（注释说明了 NCCL 2.29 的性能原因）。 |
| `_gather_shared_expert_inputs_into` | 1339 | 在给定缓冲区上执行子组 all-gather。 |
| `_reduce_scatter_shared_experts` | 1348 | 子组 reduce-scatter，把 shared 部分和还原到本 rank 行。 |
| `_shared_experts_pynccl` | 1362 | 侧流通信时强制启用 shared 组的 pynccl，避免 eager 下 c10d 的 CPU 开销。 |
| `_forward_shared_experts` | 1372 | 同步执行 shared experts：无子组通信直接算，否则 AG → MLP → RS。 |
| `_can_overlap_shared_experts_npu` | 1385 | NPU 细粒度双流重叠条件（DeepEP + `MaybeTboDeepEPDispatcher`，不在可断图/分段图里）。 |
| `_forward_unfused` | 1413 | 非融合 front 路径：front（`_ep_front` / `_ep_front_overlap` / gate+topk）、shared experts 的多种重叠方式、路由专家（MegaMoE 或 `self.experts`）、latent reduce/norm/up、尾部合并 shared 与 `prefix_sum`。内部闭包：`wait_shared_gather`（1464，路由专家前等侧流 AG）、`issue_shared_reduce_scatter`（1470，路由专家入队后发侧流 RS）、`issue_shared`（1485，按权重流/侧流通信/SBO/同步四种方式发起 shared）、`run_experts`（1543，含 NPU 的 pre/post dispatch hook）、`wait_and_finalize_shared_experts`（1592，汇合并在旧路径补 RS）。 |
| `_moe_front_needs_dense_bf16` | 1687 | （cached）routed_input 是否必须整理成稠密 bf16（只有 SM100 trtllm-gen mxfp4 和特定 aiter A8W4 路径可直接读跨步/fp32 切片）。 |
| `_route_quant_fuse_eligible` | 1727 | （cached）是否为 trtllm-gen SiTU mxfp8 激活 runner，需要通过 `route_quant_handoff` 暂存输入做融合 route+quant。 |
| `_forward_routed` | 1742 | 算 top-k 并在零拷贝上下文里让专家输出直接写入 `latent`，否则再拷贝。 |
| `_forward_routed_deferred` | 1754 | 跳过算子内 finalize 的路由专家，返回三元组供融合 finalize+AR 使用。 |
| `_forward_shared` | 1767 | 对 front 产出的 gate_up 做激活和 shared down GEMM，结果写入 `shared_output`。 |
| `_get_fused_norm_params` | 1780 | 返回 latent norm 的权重和 eps，供融合 AR+norm。 |
| `_fused_front` | 1785 | 用合并权重一次 GEMM 得到 `(gate_up, router_logits, routed_input)`，按需转稠密 bf16。 |
| `_sharded_front_gemm` | 1807 | 切分 front：一次 GEMM 得到本 rank 的 `[block \| gate_up]`（fp32）。 |
| `_sharded_front_gather` | 1821 | all-gather 各 rank 的 block，拼出完整 router logits 和 routed_input。 |
| `_forward_fused` | 1837 | 融合 front 管线：front → `[latent \| shared]` 对称缓冲区 → 侧流 shared + AR 与主流路由专家重叠 → 融合 AR+norm（可含 deferred finalize、gemm_ag up）→ up_proj → `_add3`。无侧流时一次集合通信覆盖整块缓冲区。 |
| `_forward_cp_shard_rows` | 1977 | CP-TP group sharing 的 CP extend：输入是 attention-CP 组 rank-major gather 的全部行（要求能被 CP size 整除）。可融合时对全部行做 front、路由、shared，然后对 latent 和 shared 各做一次 reduce-scatter 到本 rank 行，再 norm + up + 相加；不可融合时完整 forward 后 `tensor_split` 取本 rank 段。 |
| `forward` | 2007 | 入口：`cp_shard_rows=True` 走 `_forward_cp_shard_rows`；否则可融合走 `_forward_fused`，其余走 `_forward_unfused`。`prefix_sum` 是 Latent MoE 自己写 stream 时要加的残差。 |

### 3.4 `KimiK3DeltaAttention`（2037）

| 名称 | 行号 | 作用 |
|---|---|---|
| `__init__` | 2040 | 按 `linear_attn_parallel_group()` 决定头切分组；按全秩门 / 低秩融合 / 非融合三种布局建投影；建 `dt_bias`、`qkv_conv1d`、`A_log`（自定义 loader）、`o_norm`、`o_proj`（`reduce_results=False`）、`RadixLinearAttention`；SP-MoE 时包装 o_proj。内部闭包 `_a_log_weight_loader`（2279）兼容旧 4-D 和 K3 1-D 的 `A_log`，按头切分加载。 |
| `forward_qkvbfg` | 2344 | 非融合布局：分别算 qkv、beta、遗忘门（f_a→f_b）、输出门（全秩 g 或低秩 g_a→g_b）。 |
| `_merge_bfa_weights` | 2354 | 加载后把 `f_a_proj` 和 `b_proj` 合并成一个 `[f_a \| b]` 权重（pad 到 8 的倍数；block-FP8 时先反量化），并设置 GB300 侧流 TGV 范围；ROCm 可能改走整段 in-proj 合并。 |
| `_merge_kda_inproj_weights_hip` | 2406 | ROCm：把 `[q,k,v,g \| f_a \| b]` 合并成一个缓冲区，`_bfa_w` 为尾部视图，供小 batch 一次 GEMM。 |
| `_may_fuse_kda_inproj` | 2439 | ROCm in-proj 整段合并的前置条件（环境变量、全秩门、权重未量化且同 dtype/宽度）。 |
| `_prepare_fused_decode` | 2466 | 加载后为融合 KDA decode kernel 准备静态参数（转置的 fp32 卷积权重、bias、A_log、o_norm 权重）并挂到 `self.attn`；ROCm 走 gfx950/aiter 版本；形状不符则保持非融合。 |
| `forward_qkvbfg_fused` | 2580 | 融合布局下的投影：全秩门时 `fused_qkvg_proj` + `[f_a\|b]` 小 GEMM（ROCm 整段合并、CUDA graph 捕获时侧流重叠、普通三种分支）；低秩门时 `fused_qkvbfg_a_proj` + 批量 `fused_fg_b_proj`。`defer_f_b` 时把 f_b 交给融合 decode kernel。 |
| `forward_pre_o_proj` | 2659 | KDA 主体：先 `cp_interleave_to_sequence_order` 把 interleave 行恢复为序列顺序，投影、整理形状，把输出门暂存给融合 decode kernel（attempt-and-verify），调用 `self.attn`，未被融合消费时做 `o_norm`，最后 `cp_sequence_to_interleave_order` 还原到边界的行布局（返回 `gathered_rows` 行）。 |
| `forward` | 2725 | `forward_pre_o_proj` + `o_proj`；`all_reduce_fusion` 时 o_proj 写进对称缓冲区。 |

### 3.5 `KimiK3MLAAttention`（2745，继承 `DeepseekV2AttentionMLA`）

类属性：`dsa_decode_absorb_under_indexer = True`（DSA decode 捕获时把 q_b_proj、absorb 和 KV 写入放到 indexer 下的侧流）、`dcp_v_preproject = True`（DCP a2a/push-combine 前先做 V 投影）。

| 名称 | 行号 | 作用 |
|---|---|---|
| `__init__` | 2752 | DCP 下（`index_topk` 存在且 `attn_dcp_size > 1`）调用 `set_dsa_shards_index_k()`（2771-2775）；调用父类构造（`skip_rope=True`、`reduce_results=False`）；GGUF 拆分 K/V 时替换 a 投影并注册 `k_b`/`v_b` GGUF 参数；`all_reduce_fusion` 时用闭包 `_symm_o_proj_forward`（2853）让 o_proj 写对称缓冲区，否则设置 `use_dp_attention_reduce`；有输出门时建 `g_proj`（`parallel_group="attn_tp"`）并用闭包 `_gated_o_proj_forward`（2894）包装 o_proj：先算门并乘 `sigmoid(gate)`（可用融合 kernel），`_return_pre_o_proj` 时跳过 o_proj 返回门控后的输入；SP-MoE 时再包一层 `_route_sp_o_proj_output`。 |
| `_split_kv_b_weight_loader` | 2938 | （static）GGUF 拆分 K/V 参数的 loader：记录 weight_type 或物化未初始化参数后拷贝。 |
| `dispatch_attn_forward_method` | 2949 | 调用父类选择前向方法；GGUF 拆分 K/V 时强制 `AttnForwardMethod.MLA`。 |
| `_fork_output_gate` | 2955 | CUDA graph 捕获、非可断图、batch 不超过 `_gate_bs_limit` 时提前让侧流等待主流，准备在侧流上算门。 |
| `_compute_output_gate` | 2969 | 计算 `g_proj(hidden_states)`；有挂起侧流时在侧流算并让主流等待。 |
| `forward` | 2979 | 记下门的输入并 fork 侧流，然后调用父类 MLA 前向（o_proj 在父类深处被调用，门由包装函数应用）。 |
| `forward_pre_o_proj` | 2994 | 设置 `_return_pre_o_proj` 后跑一次 `forward`，拿回门控后、o_proj 之前的张量；不可重入，o_proj 被调用两次或未调用都报错。 |

### 3.6 `KimiK3DecoderLayer`（3157）

| 名称 | 行号 | 作用 |
|---|---|---|
| `__init__` | 3160 | 判断 MoE / SP-MoE / all_reduce_fusion；按 `is_kda_layer` 建 KDA 或 MLA（侧流 [2] 两者共用）；以 attention 实际的 `all_reduce_fusion` 为准；建 MoE 或 dense MLP、两个 RMSNorm、attention residual 的 norm/proj 和块写入标志；计算 `_ffn_writes_stream` 并声明阶段。 |
| `_declare_stages` | 3261 | 构造本层的 `AttnBankState` 残差操作，调用 `_stage_facts` 得到声明，用 `append_stages` 生成 `attn_boundary` / `ffn_boundary`。 |
| `_ffn_input_fusions` | 3289 | 返回 FFN 读出前完成 attention 求和的融合方式：融合 all-reduce（`_k3_all_reduce_add`）或 SP-MoE 的 reduce-scatter（`_k3_reduce_scatter_add`）。 |
| `_prefetch_shared_weights` | 3302 | MoE 层在 attention 前预取 shared 权重（SP-MoE 时行数除以 attn_tp_size）。 |
| `_run_self_attn` | 3312 | idle batch 直接透传；预取 shared 权重；有 `prepare_qkv_latent` 时设置 attention-TP 上下文；调用 `self_attn` 并取第一个返回值。 |
| `forward` | 3352 | attn_boundary.prepare → `_run_self_attn` → finish → ffn_boundary.prepare → MLP/MoE（MegaMoE 带 `valid_token_count`，CP extend 时取自 CP 策略；Latent MoE 写 stream 时带 `prefix_sum`；`outputs_on_cp_shards` 时 `cp_shard_rows=True`）→ ffn_boundary.finish。 |

### 3.7 `KimiK3LinearModel`（3401）

| 名称 | 行号 | 作用 |
|---|---|---|
| `__init__` | 3404 | 建 embedding（PP 首段）、3 条侧流（HIP 关闭）、`AttnBank`、最终 norm 与 `AttnBankOutputRead`（PP 末段），用 `make_pp_layers` 建层并传入 `_stage_facts`。 |
| `forward` | 3503 | PP 首段 embedding 并启动残差批次，后段从 proxy 恢复；打开 bank；逐层前向并按需捕获 DSpark/DFLASH 辅助隐状态；非末段打包 hidden、bank、aux 到 proxy；末段做 `final_norm` 并关闭 bank。 |
| `packs_aux_hidden_states` | 3631 | （property）PP=1 且需要捕获时把辅助隐状态打包成一个张量。 |
| `_dspark_capture_stream` | 3637 | bank 路径下计算某层之后的预 norm 混合 stream（用下一层或输出侧的 res_proj/res_norm 聚合），bank 被切片时再 all-gather。 |

### 3.8 `KimiK3LinearForCausalLM`（3674）

类属性 `packed_modules_mapping = {"fused_qkvg_proj": [...]}`，让 ModelSlim 能按原始模块名解析量化方案。

| 名称 | 行号 | 作用 |
|---|---|---|
| `get_model_config_for_expert_location` | 3678 | （static）返回 EPLB 用的层数、逻辑专家数、组数。 |
| `__init__` | 3694 | K3.1 注册 TGV m 范围（GB300 额外一套）；把本类的 packed mapping 合并进量化配置；建 backbone、lm_head、logits_processor。 |
| `get_input_embeddings` | 3747 | 返回 `embed_tokens`。 |
| `get_pp_proxy_dspark_hidden_size` | 3750 | 前面 PP 段捕获的 DSpark 层数 × hidden，作为 proxy 张量宽度。 |
| `get_aux_hidden_states_width` | 3756 | 打包模式下辅助隐状态的总宽度。 |
| `set_dspark_layers_to_capture` | 3761 | 设置 DSpark 捕获层（必须显式给出）。 |
| `set_dflash_layers_to_capture` | 3769 | DFLASH 复用 DSpark 的捕获点（不做 +1 偏移）。 |
| `forward` | 3775 | 调 backbone，末段解包 aux 后调用 `logits_processor`。 |
| `prepare_cp_inputs` | 3801 | CP-TP group sharing 的 prefill CP：PP 首段对全部 token 做 embedding，返回 `(input_embeds, positions, kwargs)` 供 runner 切分。 |
| `prepare_context_parallel_metadata_for_dcp` | 3810 | DCP extend 元数据：NPU 返回 None，否则原样转发给 `prepare_decode_context_parallel_metadata`。 |
| `load_weights` | 3840 | 见 §1.4：名称映射、MLA a 投影手动合并、stacked 参数、专家参数、scale 名映射，最后校验 indexer 并调用 `post_load_weights`。 |
| `_check_dsa_indexer_loaded` | 4038 | 断言每个 DSA 层构造了 5 个 indexer 参数并全部加载。 |
| `post_load_weights` | 4059 | 见 §1.4：拆 `kv_b_proj` 为 `w_kc`/`w_vc`，已有 `_dcp_v_weights` 时调用 `_prepare_dcp_v_weights(dcp_group)`（4101-4102），设置 `w_scale`；预热 `get_cw`（内部闭包 `_warm_cw`，4125）；合并 MoE front、MegaMoE 缓冲区、bias 转 fp32；KDA 权重合并与融合 decode 准备；indexer `prepare_merged_k_gate`；权重流；预编译 KDA prefill kernel。 |
| `_setup_shared_weight_stream` | 4189 | 按 PD 角色和环境变量决定是否启用 shared-expert 权重流；检查条件和同节点约束后建 `SharedWeightStream` 并给每个 MoE 层分配 slot。 |

### 3.9 `KimiK3ForConditionalGeneration`（4249）

类属性：`supports_cuda_vmm_feature_transport`、`packed_modules_mapping`（融合模块 → 检查点分片名）、`encoder_media_processor_config`（nvjpeg 解码）、`encoder_only_safetensors_weight_prefixes`、`hf_to_sglang_mapper`。

| 名称 | 行号 | 作用 |
|---|---|---|
| `get_model_config_for_expert_location` | 4253 | （static）转发到文本配置的 EPLB 配置。 |
| `__init__` | 4297 | 建视觉塔和投影（视觉 DP）；非 encoder-only 时按量化描述是否带 `language_model.` 前缀决定前缀，建 `KimiK3LinearForCausalLM`。 |
| `model` | 4334 | （property）返回 `language_model`。 |
| `__setattr__` | 4337 | 忽略对 `model` 的赋值，避免覆盖上面的 property。 |
| `post_load_weights` | 4342 | 转发给 `language_model.post_load_weights()`。 |
| `precompile_kernels_after_loading` | 4347 | 预编译视觉 RoPE 和视觉 FA4 kernel（language_only 时跳过）。 |
| `get_input_embeddings` | 4355 | 返回文本 embedding；encoder-only 时报错。 |
| `lm_head` | 4363 | （property）返回文本 lm_head。 |
| `pp_group` | 4372 | （property）转发 PP 组（prefill CP 的 gather/logits 尾部会用外壳）。 |
| `capture_aux_hidden_states` | 4376 | （property）转发 aux 捕获标志。 |
| `logits_processor` | 4380 | （property）转发 logits_processor。 |
| `get_pp_proxy_dspark_hidden_size` | 4383 | 转发；无文本模型时返回 0。 |
| `set_dflash_layers_to_capture` | 4388 | 转发；encoder-only 报错。 |
| `set_dspark_layers_to_capture` | 4395 | 转发；encoder-only 报错。 |
| `get_aux_hidden_states_width` | 4402 | 转发；无文本模型时返回 0。 |
| `preprocess_mm_for_encoder` | 4407 | EPD 编码端的逐图预处理入口，只支持图像，调用 `prepare_kimi_k3_encoder_inputs`。 |
| `get_image_feature` | 4433 | 校验每个 item 只含一张图，收集 grid；视觉 DP 时用 `run_dp_sharded_mrope_vision_model`，否则本地跑视觉塔，最后过投影。内部闭包 `materialize_item_features`（4454）只物化本 rank 负责的图：本地已预处理、延迟 GPU/CPU 预处理（校验 grid 与 patch 数）或直接取特征张量。 |
| `pad_input_ids` | 4603 | 用 `pad_mm_tokens` 填充多模态占位 token。 |
| `start_layer` | 4607 | （property）文本模型的起始层。 |
| `end_layer` | 4613 | （property）文本模型的结束层。 |
| `prepare_context_parallel_metadata_for_dcp` | 4618 | 转发给 `language_model` 同名方法。 |
| `prepare_cp_inputs` | 4646 | runner 切分 token 前完成多模态 embedding 注入（`prepare_mm_inputs`），返回 `(input_embeds, positions, kwargs)`。 |
| `forward` | 4660 | 通过 `general_mm_embed_routine` 完成多模态嵌入并调用文本模型。 |
| `load_weights` | 4680 | 应用名称映射，流式加载视觉权重，其余交给文本模型（内部生成器 `stream_language_weights`，4691）；encoder-only 时也要耗尽迭代器。 |

### 3.10 `KimiK31ForConditionalGeneration`（4717）

空子类（`pass`），继承 `KimiK3ForConditionalGeneration` 的全部行为，只提供 K3.1 的架构名。各处 hook 和白名单都按这个字符串判断 K3.1。

---

## 4. 与 PCP+DCP 方案 A 的关系

方案 A 下的拓扑（计划 §3.2）：TP 组 = attention CP 组 = DCP 组，attention TP = 1（MLA 头复制），KDA 头在整个 TP 组上切分，`dcp_rank = attn_cp_rank`，token 按"请求内位置取模"分配给 CP rank。

### 4.1 计划 §4.6 / §4.7 涉及的位置

| 位置 | 计划要求 | 读代码后的说明与注意点 |
|---|---|---|
| `KimiK3MoE._forward_cp_shard_rows`（1977） | 适配新的行布局 | 这个方法只依赖两点：输入是 attention-CP 组上 **rank-major** gather 的行，并且每个 rank 的分片长度相同（`assert num_tokens % size == 0`，用 `reduce_scatter_tensor` / `tensor_split(size)[rank]` 取回本 rank 行）。MoE 是逐行计算，与行在序列里的位置无关。所以只要新切分仍然"每 rank 补齐到相同长度、按 rank 顺序 gather"，这个方法可能不用改（推断），要改的是边界的 gather/padding。需要注意：按位置取模后，每个请求在各 rank 上的 token 数最多相差 1，一批多个请求时 rank 间差值会累积，padding 会变多；可融合路径会把 padding 行也送进路由专家，没有屏蔽（现有行为）。 |
| `_stage_facts`（3084）和层边界 | 确认 gather / reduce-scatter 使用新行布局 | 本文件只声明语义（KDA 的 `tp_group=SumGroup.TP`、MoE 的 `output_on_cp_shards`），实际的行搬运在 `layers/layer_boundary/` 和 CP 策略里。`_latent_moe_writes_stream` 在 `attn_cp_size > 1` 时恒为假，方案 A 下保持不变。 |
| KDA 的 `forward_pre_o_proj`（2667-2723） | 适配 `cp_interleave_to_sequence_order` / `cp_sequence_to_interleave_order` | 这两个函数在 `layers/cp/interleave.py:94-160`。它们的 reshape/transpose 分支和 `interleave_rows` kernel 分支都假设"拼接下标 `i % cp`"；但已有一个 `metadata.gather_index` 分支，支持任意排列（`index_select` / `index_copy_`）。方案 A 可以在 `build_metadata` 里填好 `gather_index`，这样 `kimi_k3.py` 不需要改（推断）。同时要保证 graph 模式下 `graph_num_tokens` 的语义一致。 |
| `KimiK3MLAAttention.__init__` 的 `set_dsa_shards_index_k()`（2771-2775） | 方案 A 下照常调用 | 调用条件是构造时 `get_parallel().attn_dcp_size > 1`，设置的是模块级全局变量。因此 §4.2 必须保证 `attn_dcp_size` 在模型构造前就已经按"DCP 复用 CP 组"推导好。打开后 `dsa_index_k_page_tokens` 返回 `page_size × dcp`，`forward_mla.py` 的 DCP 合并也走 K3.1 专用的 strided 打包；prefill CP 的 indexer K 物化（§4.4）必须按切分后的 index-K 布局读取。 |
| `KimiK3MLAAttention` 输出门 `g_proj` 和 o_proj 包装（2866-2933） | decode 时按 CP rank 切头使用 o_proj 和 g_proj | `g_proj` 是 `ColumnParallelLinear(parallel_group="attn_tp")`，方案 A 下 attn_tp=1，所以是完整复制的（64 头 × `v_head_dim`）。D2 下 decode 时注意力输出只剩 1/N 的头，`g_proj` 必须和 o_proj 一起按 dim 0 切片，否则 `x * sigmoid(gate)` 形状不匹配。门在 `_gated_o_proj_forward` 里计算，可能在侧流 `_gate_alt_stream` 上运行（`_compute_output_gate`），切片上下文必须覆盖整个 MLA forward。o_proj 包装只替换实例的 `forward`，切片改的是 `weight` 和 `input_size_per_partition` 属性，两者应能共存（推断）。`_k3_symm_o_proj_out` 用 `o_proj.weight.shape[0]`（输出维），不受 dim 1 切片影响。 |
| `dcp_v_preproject`（2750）与 `post_load_weights` 的 `_prepare_dcp_v_weights`（4101-4102） | V 预投影按切片后的头准备权重 | `_prepare_dcp_v_weights`（`forward_mla.py:187-195`）对 `w_vc` 做 `dcp_group.all_gather(dim=0)`，前提是每个 rank 只持有 DCP 组内 1/N 的头。方案 A 下 `w_vc` 本来就有全部头，在 `post_load_weights` 里（不在 decode 切片上下文中）调用会得到 N 倍的头，后续 `cached.copy_` 也会形状不符。`forward_mla.py:1061-1068` 的惰性调用如果发生在 cp-decode-attn-tp 的切片上下文内，拿到的是切片后的 `w_vc`，gather 结果才正确（推断）。需要让两处调用都用切片后的 `w_vc`，或者在"DCP 组 = CP 组"时直接用本地完整 `w_vc`。 |
| `prepare_cp_inputs`（3801、4646） | 计划未列为修改点 | 两个方法只负责在切分前完成 embedding，不涉及切分规则；方案 A 下预计不用改。 |
| `prepare_context_parallel_metadata_for_dcp`（3810、4618） | §4.5 调整 runner 的调用顺序 | 这两个方法只是把参数转发给 planner，签名里没有 `forward_batch`，也没有 CP 元数据。如果 §4.5 的"合并元数据"需要知道本卡 query 行，就要么改两层签名（以及 `eager_runner.py:296`、`prefill_cuda_graph_runner.py:1320` 两个调用方），要么由 runner 在返回后补充 CP 信息。计划只提到 runner，没有提到这两个模型方法的签名（计划未提及）。 |

### 4.2 计划可能遗漏的点

1. **K3 没有调用 `maybe_use_decode_attn_tp`（计划未提及）。** D2 依赖 `DeepseekV2AttentionMLA.maybe_use_decode_attn_tp`（`deepseek_v2.py:2341`）在 decode 时切片 `q_b_proj`、`o_proj`、`w_kc`、`w_vc`、`attn_mqa` 并修改 `num_local_heads`。DeepSeek-V2 的 decoder 层在 `deepseek_v2.py:2912` 用这个上下文包住 attention；`KimiK3DecoderLayer._run_self_attn`（3312）直接调用 `self.self_attn(...)`，没有这个上下文。只把 `KimiK31ForConditionalGeneration` 加进 `CP_DECODE_ATTN_TP_SUPPORTED_ARCHS` 不会产生任何切片效果，还需要在 `_run_self_attn` 里对 MLA 层加上下文，并把 `g_proj` 加进切片列表（可以在 `KimiK3MLAAttention` 里覆盖 `maybe_use_decode_attn_tp`）。
2. **延迟求和的阶段声明（计划未提及）。** K3 以 `reduce_results=False` 构造父类，所以一旦 `can_defer_cp_decode_attn_tp()` 为真（方案 A 拓扑满足其中 attn_cp==tp、attn_tp==1 等条件），父类的 `defer_cp_decode_attn_tp` 会被置为真，decode 的 CP rank 求和就交给阶段边界。DeepSeek-V2 在 `stage_facts` 里声明 `ordinary_tp_group=SumGroup.TP`，并在 `_build_stages` 里校验两者一致（`deepseek_v2.py:2790-2840`）。K3 的 `_stage_facts` 只给 KDA 层声明 `tp_group`，MLA 层没有声明 `ordinary_tp_group`，也没有一致性校验。不补上的话，decode 时 o_proj 的部分和不会被求和，结果会静默出错（推断）。
3. **DCP decode 的头数按构造时的 `num_local_heads` 定（计划未提及）。** `deepseek_v2.py:2261-2264` 用 `num_local_heads * dcp_size` 构造 `attn_mqa_for_dcp_decode`。方案 A 构造时 attn_tp=1，`num_local_heads=64`，会得到 512 头；D2 切片后每卡应为 8 头，gather 后应为 64 头。这个问题在父类里，但由 K3 继承触发，需要和 §4.6 一起处理（推断）。
4. **方案 A 会让 K3 的融合 all-reduce 失效，影响 P3 的 TPOT 目标（计划未提及）。** `_fuses_attn_all_reduce`（3031）要求 `attn_tp_size > 1`。TP8+DCP8 下 attn_tp=8，MLA 和 KDA 的 o_proj 都走 `k3_ar_fusion` 融合 AR+残差加；方案 A 下 attn_tp=1，所有层的 `all_reduce_fusion` 都为假，KDA 的 TP 求和和 D2 的 decode 求和都回到普通边界路径。计划 §5 的 P3 验收标准是"TPOT 与 TP8+DCP8 持平"，需要把这部分开销算进去，或者让融合 AR 支持"attention CP 组上求和"。同理，`_shards_moe_rows` / SP-MoE 依赖 attn_tp > 1，方案 A 下也不可用。
5. **MegaMoE 的 padding 计数（计划 §4.3 已列，补充注意点）。** `KimiK3DecoderLayer.forward`（3368-3379）在 CP extend 下用 `moe_num_token_non_padded()`，它返回 `per_rank_logical_token[cp_rank]`，并假设本 rank 的有效行是分片的前缀（`interleave.py:167-181`）。新切分必须保持"有效行在前、padding 在后"，否则 `_forward_mega_experts` 的 `valid_rows` 掩码会屏蔽错误的行。
6. **KDA 状态与 DCP 无关，但 KDA decode 的融合 kernel 依赖本地头数。** 方案 A 下 KDA 仍在 TP 组上切头（每卡 12 头），与现有 CP-TP group sharing 一致；`_prepare_fused_decode` 的 `SUPPORTED_HEADS` 检查不受影响。PD 传输里 KDA 状态"按头 r 对 r"（计划 §4.10）的前提也成立。
7. **GGUF 拆分 K/V 路径。** `dispatch_attn_forward_method`（2949）在 `_kimi_split_gguf_kv_b` 时强制非 absorb 的 MLA，`post_load_weights` 也跳过 `w_kc`/`w_vc` 的生成，DCP 的 V 预投影和 D2 的 `w_kc`/`w_vc` 切片都不适用。建议在 §4.1 的参数校验里显式拒绝这个组合（推断）。
8. **CUDA graph 下的输出门侧流。** `_fork_output_gate` 只在捕获模式且 batch ≤ `_gate_bs_limit` 时用侧流计算 `g_proj`。D2 切片后的 `g_proj` 权重要进入切片缓存，并在 decode graph 捕获时固定地址（计划 §4.8 提到切片缓存，但没有点名 `g_proj` 和侧流）。

### 4.3 行号核对

计划引用的 `kimi_k3.py` 行号与当前文件一致或基本一致：`KimiK3MLAAttention` 2745-3011（类结束于 3011，3014 是下一个顶层函数）；`set_dsa_shards_index_k` 调用在 2771-2775；`_forward_cp_shard_rows` 在 1977；KDA 重排在 2667-2723。
