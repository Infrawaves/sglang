# `python/sglang/srt/mem_cache/kv_cache_configurator.py` 代码理解

- 文件路径：`python/sglang/srt/mem_cache/kv_cache_configurator.py`
- 行数：3662
- 所属模块：内存缓存层，负责在模型加载后测算显存预算并创建请求表、KV 池和 KV 分配器。

## 1. 整体功能

`KVCacheConfigurator` 是 `ModelRunner` 初始化 KV cache 的入口（`model_runner.py:698` 构造，`:1044` 调 `configure`）。它做三件事：

1. **测预算**：`_profile_available_bytes` 用当前空闲显存减去非静态余量（`pre_model_load_memory * (1 - mem_fraction_static)`）和多模态预留；混合 mamba/KDA 模型再由 `_handle_max_mamba_cache` 先切走状态池的份额。
2. **算尺寸**：`config_from_budget` 把字节预算交给 `model_executor/pool_configurator.py`（每 token 字节数在那里计算），得到 `max_total_num_tokens` 等，再施加用户上限、PP 同步；`resolve_max_num_reqs` 算出每个 DP worker 的最大并发请求数。unified memory 模式走 `_unified_config_from_budget`。
3. **建池**：`configure` → `_derive_pool_sizes` → `_init_pools`，依次创建 `ReqToTokenPool`（普通 / 混合 mamba / PD decode 版本）、`token_to_kv_pool`（按模型类型分派到 MHA、MLA、DSA、SWA、混合线性、DSV4、Ascend、MiniMax 等十几种构造函数）、`token_to_kv_pool_allocator`。

DCP 相关的核心概念：
- **物理行与逻辑 token**：`max_total_num_tokens` 是**每张卡的物理行数**。DCP 下分配器的 id 空间被放大 `attn_dcp_size` 倍（`_build_token_to_kv_pool_allocator` 2952-2954 行：容量 `max_total_num_tokens * attn_dcp_size`，页大小 `page_size * attn_dcp_size`），这就是"放大页"。调度器看到的逻辑容量由 `logical_token_capacity` 换算。
- **`loc_space_scale`**：draft worker 的 id 空间相对 target 的倍数。target 恒为 1；不走 DCP 的 draft 为 `dcp_size // execution_width`；DCP 折叠的 MLA draft 为 1。
- **混合模型的划分**：mamba/KDA 状态池先按 `mamba_full_memory_ratio / (1 + ratio)` 拿走一部分预算，剩余部分给 KV 池。

## 2. 主要依赖与调用方

依赖：
- `model_executor/pool_configurator.py`：`create_memory_pool_configurator`、`MemoryPoolConfig`，每 token 字节数（含 DCP 下 indexer 的放大）在这里计算。
- `mem_cache/memory_pool.py`：`MLATokenToKVPool`、`DSATokenToKVPool`、`HybridLinearKVPool`、`HybridReqToTokenPool`、`ReqToTokenPool` 等池类。
- `mem_cache/allocator/`：`PagedTokenToKVPoolAllocator`、`SWATokenToKVPoolAllocator`、`HiSparseTokenToKVPoolAllocator`、`PageInterleavePoolAllocator` 等。
- `mem_cache/page_interleave*.py`：`--enable-kv-cache-sharding` 的页切分池。
- `mem_cache/kv/`（plan、sizing、build、decl）：unified memory 路径。
- `configs/model_config.py`、`configs/hybrid_arch.py`：`is_deepseek_dsa`、`is_kimi_k31`、`mla_attention_dcp_size`、`mambaish_config`、`hybrid_kda_config` 等。
- `runtime_context`：`get_parallel().attn_dcp_size`、`attn_tp_size`、`dcp_enabled` 等。

调用方（grep 确认）：
- `model_executor/model_runner.py:108, 698, 1044, 1640, 1656`：构造、`configure`、`logical_token_capacity`、`hybrid_swa_token_capacity`。
- `model_executor/pool_configurator.py:110` 起多处：接收 `kvc: KVCacheConfigurator`，读取其属性（`loc_space_scale`、`dflash_draft_rows_dcp_collapsed` 等）并调用 `calculate_mla_kv_cache_dim`。
- `layers/moe/paged_experts/sizing.py:70`：使用 `mm_runtime_reservation_gb`。
- `hardware_backend/npu/sparsity_driven_kv_offload/config.py:99`：导入本模块函数。

## 3. 类与函数说明

### 3.1 模块级函数与常量

| 名称 | 行号 | 作用 |
|---|---|---|
| `_should_elide_dsa_index_k` | 136 | 是否给共享 top-k 的 DSA 层分配 0 行 index-K（紧凑布局）；NPU 的 PD 或 HiCache、HiSparse、draft worker、unified external linker 下返回 False。 |
| `_get_dsv4_compress_state_dtypes` | 156 | 从 `SGLANG_DSV4_COMPRESS_STATE_DTYPE` 解析 DSV4 压缩状态 dtype（fp32 或 bf16），其他值报 `ValueError`。 |
| `_KV_POOL_BASE_FOR_KIND` | 170 | 常量：平台自定义池类必须继承的基类（mha/mla/dsa）。 |
| `replayssm_ring_memory_charge` | 177 | 计算 ReplaySSM 环形缓冲的显存：按请求计费时返回固定字节（`max(1, max_running//dp)+1` 个槽），否则按每个 mamba 槽计费。 |
| `_pd_prefill_skips_mamba_spec_state` | 196 | PD prefill（非角色切换、非 unified、非 NPU）不跑 TARGET_VERIFY，可跳过投机验证用的 mamba 快照。 |
| `unified_fp8_for_dsv4_pool` | 208 | DSV4 池是否用 unified fp8 布局；DSpark draft 保持 bf16。 |
| `_should_enable_lazy_compaction` | 218 | unified 池的 lazy compaction 默认开启，`SGLANG_DISABLE_LAZY_COMPACTION` 可关闭。 |
| `mm_runtime_reservation_gb` | 226 | 多模态模型在 KV 预算中预留的显存（嵌入缓存加 cuda_ipc/vmm 特征传输池）。 |
| `MAMBA_CACHE_*` 常量 | 251-253 | mamba 池与最大并发请求数的基础比例（3）、skip decode lock 时的减量和 no_buffer 的补偿量。 |
| `_pp_local_per_request_bytes` | 256 | PP 下把线性状态的全局每请求字节数换算成本 stage 的份额；各层字节不均匀时报 `ValueError`。 |
| `runner_state` | 292 | 决定 runner 绑定到 KV 池路由器的循环状态来源：draft 自有深度用自己的表，unified 分配器用分配器的 state，否则共享 target 的。 |
| `mla_draft_rows_are_dcp_collapsed` | 357 | 判断 draft 的 MLA 行是否像 target 一样按 DCP 折叠（物理行）：要求 DCP>1、MLA、非 mamba、非 SWA、非 DSA、非 DSV4、非 FP4、非 out-of-tree、非 ascend。 |
| `host_only_mamba_checkpoint` | 381 | `SGLANG_OPT_MAMBA_HOST_ONLY_CHECKPOINT` 开关；要求开启 hierarchical cache，且不能与 int8 mamba checkpoint 同用，否则报 `ValueError`。 |
| `calculate_mla_kv_cache_dim` | 3596 | 计算 MLA/DSA 每行宽度：非 DSA 为 `kv_lora_rank + qk_rope_head_dim`；DSA 下按本角色的 prefill/decode backend 判断是否用 raw fp8 布局（TRT-LLM、Prims-TS、TileLang、HIP 部分 kernel），否则 fp8 时用带 scale 的 `fp8_ds_mla_row_width`。K3.1 在 HIP 上保持带 scale 的布局。 |

### 3.2 数据结构（`msgspec.Struct`，无方法）

| 名称 | 行号 | 作用 |
|---|---|---|
| `KVCacheConfigResult` | 308 | `configure` 的返回值：token 容量、并发上限、full/SWA 容量、三个池对象和 `MemoryPoolConfig`。 |
| `_InitializedPools` | 319 | `_init_pools` 的返回值：请求表、KV 池、分配器和可选的循环状态。 |
| `_UnifiedSWAHeadGeometry` | 327 | unified 混合 SWA 主机池的 full / SWA 头几何。 |
| `_PoolSizes` | 342 | 建池用的所有尺寸：token 容量、DSV4 的 c4/c128 容量和状态池大小与 dtype、unified 字节数。 |

### 3.3 `KVCacheConfigurator`（404 行，`@dataclass(slots=True, kw_only=True)`）——属性与初始化

| 名称 | 行号 | 作用 |
|---|---|---|
| `draft_attn_tp_size`（property） | 452 | draft 的 attention TP 宽度，优先取 `draft_parallel_layout`。 |
| `__post_init__` | 457 | 计算 DFLASH draft 的每 token 字节和是否 DCP 折叠；解析 `mambaish_config`、`hybrid_gdn_config`、`hybrid_kda_config`；`_resolve_kv_shard`；记录 draft 是否自有深度；NPU modelslim 的缓存布局。 |
| `hybrid_swa_token_capacity` | 494 | 混合 SWA 的可用 token 容量：unified 模式取分配器 `size_full`（受 `max_total_tokens` 限制），否则取 full 或 SWA 容量。 |
| `logical_token_capacity` | 511 | 把每卡物理行换算成请求 token 容量：SWA 不放大；其他为 `max_total_num_tokens * attn_dcp_size // loc_space_scale`。 |
| `_resolve_kv_shard` | 522 | 解析 `--enable-kv-cache-sharding` 的页切分身份（rank、size、group），关闭时为 `(None, 1)`；out-of-tree 平台拒绝。 |
| `_build_fp4_quant_method` | 541 | FP4 KV 时构造量化方法并加载 scale，否则 None。 |
| `_build_mha_quant_method` | 557 | MHA 量化方法：CPU fp8 用 `cpu_fp8_e4m3`，否则走 FP4。 |
| `configure` | 562 | 主入口：校验 CPU fp8 和 page-major draft 限制；draft 用 target 传来的 `memory_pool_config`，target 调 `_resolve_memory_pool_config`；派生尺寸、建池、绑定 state 路由；SWA 请求环时用分配器的真实 SWA 容量；返回 `KVCacheConfigResult`。 |
| `draft_mla_rows_are_physical`（property） | 641 | draft worker 且 DCP>1 且满足 `mla_draft_rows_are_dcp_collapsed` 时为真，表示 draft 按 `loc // dcp_size` 读写 owner 行。 |
| `loc_space_scale`（property） | 669 | 见第 1 节；target 和 DCP 折叠 draft 为 1，其他 draft 为 `dcp_size // mla_attention_dcp_size(is_nextn=True)`（MHA draft 执行宽度为 1）。 |
| `pool_page_size`（property） | 685 | 池的页大小 = `page_size * loc_space_scale`；target 等于 `page_size`。 |
| `_derive_pool_sizes` | 688 | 从 `MemoryPoolConfig` 得到 `_PoolSizes`：容量乘以 `loc_space_scale`；draft 不持有 DSV4 c4/c128/状态池（置 0）；DSV4 读取压缩状态 dtype。 |

### 3.4 建池总流程

| 名称 | 行号 | 作用 |
|---|---|---|
| `_init_pools` | 747 | 建池主流程：unified 模式直接走 `_init_unified_pools`；draft 共享 unified 分配器时尝试 fused 绑定，否则按虚拟 id 空间（DCP 折叠时按放大页对齐后除以 dcp）调整尺寸；HiSparse draft 跟随 target 的逻辑空间；DFLASH draft 环按请求行数定尺寸；创建或克隆请求表；`runner_state`；校验 prefill-only 限制；`_build_token_to_kv_pool`；KV 分片时要求池为 `PageInterleaveKVPoolMixin`；draft 池的逻辑容量必须覆盖虚拟 id 空间（assert）；`_build_token_to_kv_pool_allocator`；prefill-only 但池不是 NoOp 时报 `RuntimeError`。 |
| `_fused_draft_from_target_buffer` | 1005 | EAGLE / DFLASH draft 从 target 存储计划中取 fused placement；计划未声明或 dtype 不一致时报 `ValueError`。 |
| `_validate_prefill_only_disable_kv_cache_pool_family` | 1489 | `--prefill-only-disable-kv-cache` 只支持普通 MHA；其他池族（含 MLA、DSA、混合线性）报 `RuntimeError`。 |

### 3.5 unified memory 路径

| 名称 | 行号 | 作用 |
|---|---|---|
| `_unified_window_and_state` | 1034 | unified 计划是否包含窗口子池（混合 SWA，非 DSV4）和状态子池（mamba 类）。 |
| `_validate_replay_spec_algorithm` | 1043 | 开启 linear ReplaySSM spec 且算法为 DSPARK/DFLASH 时要求 KDA 模型，否则 `ValueError`。 |
| `_kv_layer_decls` | 1057 | 生成 unified 池的逐层声明：full 层几何（MLA 或 MHA，MHA 头数用 `get_num_kv_heads(attn_tp, attn_dcp)`）、窗口层几何、状态几何（含 ReplaySSM 与 PLE 扩展）；既无窗口也无状态时报错；KDA ReplaySSM spec 不支持 PD。 |
| `_init_unified_pools` | 1206 | 用已有的 plan 和 sized 结果调用 `build_kv_storage` 构造统一字节缓冲；未先 sizing 时报 `RuntimeError`。 |
| `_unified_request_rows` | 1260 | 请求表在 context 之外的额外行数，以及 PD decode 的预分配行数。 |
| `_unified_swa_head_geometry` | 1282 | 计算混合 SWA 的 full / SWA 头几何（NPU 压缩路径从 hf text config 取）。 |
| `_unified_host_names` | 1308 | 本 runner unified 计划中的子池名。 |
| `_fused_draft_decision` | 1314 | 决定 draft KV 是否融合进 target 的 unified 池：DCP 下 Triton draft 不支持，prefill/decode 的 DCP 宽度不一致也拒绝；按 draft 自己的 dtype 规则计算区域 dtype，调用 `place_fused_draft`。 |
| `fused_entry_bytes` | 1443 | 某子池 fused 后的每 token 字节（host + draft + pad），没有融合时为 None。 |
| `_fused_draft_for_pool_factory` | 1461 | 把 fused 决策交给池工厂并打日志；声明非法时报 `ValueError`。 |
| `_unified_config_from_budget` | 3269 | unified 模式的尺寸求解：首次构建 storage plan，组装 `KVSizingInputs`（含状态份额、窗口比例、状态槽策略），`size_plan` 求解，施加 token 约束和 PP 同步后重解，返回 `MemoryPoolConfig`。 |
| `_unified_budget_for_full_tokens` | 3380 | 用户设置 `max_total_tokens` 时，二分查找得到恰好够这么多 FULL token 的最小字节预算。 |

### 3.6 请求表（ReqToTokenPool）

| 名称 | 行号 | 作用 |
|---|---|---|
| `_req_pool_rows` | 1529 | 请求行数；Unlimited-OCR v2 文档切分时翻倍。 |
| `_build_req_to_token_pool` | 1535 | 分派：PD decode 或角色切换时用 decode 版本（混合 mamba 用 `HybridMambaDecodeReqToTokenPool`），否则混合 mamba 用 `HybridReqToTokenPool`，其余用默认表。 |
| `_get_mamba_layer_ids_for_req_pool` | 1567 | 本 PP stage 上的 mamba 层 id。 |
| `_get_ple_req_pool_kwargs` | 1576 | Qwen4-exp 的 short conv / ngram 参数，其他模型为空。 |
| `_build_hybrid_mamba_decode_req_pool` | 1599 | PD decode 的混合 mamba 请求表，带预分配、ReplaySSM 参数。 |
| `_build_decode_req_pool` | 1644 | PD decode 的普通请求表（NPU DSV4 用专用类）。 |
| `_build_hybrid_req_pool` | 1667 | 混合 mamba 请求表：状态池大小 `max_mamba_cache_size`，PD prefill 时不分配投机快照。 |
| `_build_default_req_pool` | 1714 | 默认请求表（NPU DSV4 用扩展类）。 |

### 3.7 KV 池构造（token_to_kv_pool）

| 名称 | 行号 | 作用 |
|---|---|---|
| `_build_token_to_kv_pool` | 1738 | KV 池分派：MTP 分片 draft → DSV4 → Ascend 系列 → MLA+SWA → **DSA（要求非 mamba）** → MLA/MLA-FP4 → 其他（SWA、MiniMax、OCR v2、**混合线性**、MHA）。 |
| `_build_mtp_kv_shard_pool` | 1887 | MTP draft 共享 target 的页切分分配器时，按 target 几何建独立 KV 的 PageInterleave 池；只支持 EAGLE MTP，**DCP>1 时报错**，几何、dtype、页大小、容量不一致都报 `ValueError`。 |
| `_build_dsv4_kv_pool` | 2018 | DeepSeek-V4 池（GPU 或 NPU 子类），GPU 要求 `page_size == 256`；target 注册请求行分配时的状态重置回调。 |
| `_page_major_enabled` | 2117 | 是否启用 page-major 布局；unified 模式下的 draft 关闭。 |
| `_resolve_kv_pool_class` | 2125 | 取平台自定义池类，必须继承对应基类，否则 `TypeError`。 |
| `_build_ascend_swa_kv_pool` | 2137 | Ascend 的 SWA 池。 |
| `_build_ascend_minimax_sparse_kv_pool` | 2177 | Ascend 的 MiniMax 稀疏池。 |
| `_build_ascend_mla_kv_pool` | 2208 | Ascend 的 MLA 池；DCP target 的 indexer 大小为 `max_total_num_tokens * dcp_size`（全局 slot 空间）。 |
| `_build_ascend_mha_kv_pool` | 2268 | Ascend 的 MHA 池。 |
| `_build_dsa_kv_pool` | 2289 | 纯 DSA（非混合）池：可选页切分、HiSparse、GLM DSA 的 CP 层切分（`LayerSplitDSATokenToKVPool`）；`loc_space_scale > 1` 时 `dcp_replicated=True`；K3.1 时 `kv_scale_ue8m0=True`。 |
| `_build_hybrid_mla_swa_kv_pool` | 2371 | MLA/DSA full 层加 MLA SWA 层的 `SWAKVPool`。 |
| `_build_mla_fp4_kv_pool` | 2426 | FP4 MLA 池。 |
| `_build_mla_kv_pool` | 2441 | 普通 MLA 池，页切分时换成 `PageInterleaveMLATokenToKVPool`。 |
| `_build_hybrid_swa_kv_pool` | 2470 | MHA 混合 SWA 池；draft 需要全容量时 SWA 大小取 full 容量。 |
| `_build_minimax_sparse_kv_pool` | 2522 | MiniMax 稀疏索引池（可选 NVFP4 原生栈、HiSparse）。 |
| `_build_unlimited_ocr_v2_kv_pool` | 2575 | Unlimited-OCR v2 的主 K/V 加单头 index K 池。 |
| `_build_hybrid_linear_kv_pool` | 2599 | **混合线性（mamba/KDA）模型的 KV 池**：只为 full attention 层建 KV（draft 只建 1 层）；MLA 时加 DSA 参数（`use_dsa`、index 维度、`kv_scale_ue8m0`（K3.1）、`skip_topk_layers`）；可选 HiSparse、QSA；状态池通过 `mamba_pool` 传入。 |
| `_uses_dflash_draft_kv_ring` | 2730 | DFLASH draft 是否使用按请求的 KV 环。 |
| `_hybrid_full_attention_pool_class` | 2737 | 混合模型 full 层的池类：DSA、NPU MLA/MHA、MLA、MXFP8 或 MHA。 |
| `_build_mha_kv_pool` | 2765 | MHA 池（MXFP8、NoOp、页切分变体），头数为 `get_num_kv_heads(attn_tp, attn_dcp)`。 |

### 3.8 分配器

| 名称 | 行号 | 作用 |
|---|---|---|
| `_build_token_to_kv_pool_allocator` | 2817 | target 新建分配器：NPU 的 SWA/分页分配器（分页时容量和页大小都乘 `attn_dcp_size`）；DSV4 非分页 SWA 用普通分页；纯 SWA、SWA、HiSparse、页切分；`page_size == 1` 且未开 DCP 用逐 token 分配器；**其余用分页分配器，容量 `max_total_num_tokens * attn_dcp_size`、页大小 `page_size * attn_dcp_size`**（2947-2959）。draft 共享 target 分配器，只为 SWA 注册映射（虚拟 id 时用恒等映射）。 |

### 3.9 显存测算与尺寸求解

| 名称 | 行号 | 作用 |
|---|---|---|
| `_profile_available_bytes` | 3010 | 先 `gc.collect()` 再测空闲显存（多卡时取 world 内最小值）；减去非静态余量（混合 mamba 且 post-capture KV 时至少保留捕获激活余量）和多模态预留；混合 mamba（非 unified）调用 `_handle_max_mamba_cache`；剩余不大于 0 时报 `ValueError` 并给出建议的 `mem_fraction_static`。 |
| `_calculate_mamba_ratio` | 3082 | 每个请求占用的 mamba 状态槽数：无 radix 或无状态 decode 树为 1；host-only checkpoint 为 1 加 ping-pong 槽；否则基础 3（skip decode lock 时 2）加 extra buffer 的跟踪槽。 |
| `_apply_token_constraints` | 3123 | 施加 `max_total_tokens` 上限并在 PP rank 间取最小值。 |
| `resolve_max_num_reqs` | 3152 | 每个 DP worker 的最大并发：用户值或估算值（2048-4096），不超过 `token_capacity // 2`，混合 mamba 下再受 `max_mamba_cache_size // ratio` 限制；为 0 时报 `RuntimeError`。 |
| `_resolve_memory_pool_config` | 3209 | 测预算 → `config_from_budget` → 算并发 → `finalize_with_max_running_requests`；unified target 直接返回。 |
| `config_from_budget` | 3230 | 字节预算 → `MemoryPoolConfig`：调用 pool_configurator 的 `calculate_pool_sizes`，施加约束和 `cap_tokens`，容量变化时用 `calculate_pool_sizes_from_max_tokens` 重算；约束不允许让容量变大（assert）。 |
| `_handle_max_mamba_cache` | 3411 | 为 mamba/KDA 状态池定尺寸并从剩余显存中扣除：按 PP stage 换算每请求字节（可能用 fused entry 字节）；ReplaySSM 环的计费；三种来源决定 `max_mamba_cache_size`（用户指定、无 radix 时等于最大并发、按 `mamba_full_memory_ratio` 自动求解，投机解码时联合求解中间状态）；结果不大于 0 时报 `RuntimeError`；返回扣除状态池后的剩余 GB。 |

覆盖统计：模块级函数 12 个，数据结构类 4 个，`KVCacheConfigurator` 方法 63 个（含 property）。

## 4. 与 PCP+DCP 方案 A 的关系

计划 §4.9 写的是"`mem_cache/kv_cache_configurator.py`：KV 池容量按 DCP 切分后计算"，§4.1 要求 `page_size % dcp == 0`，§4.9 要求 radix cache 页按 `page_size × dcp` 对齐。读代码后的情况如下。

**K3.1 在本文件中走的路径：**
K3.1 是 KDA 加 DSA 的混合模型，`mambaish_config`（经 `hybrid_kda_config` → `kimi_linear_config`）不为 None。因此 `_build_token_to_kv_pool` 1826 行的 `_build_dsa_kv_pool` 分支（条件含 `not self.mambaish_config`）**不会命中**，实际走 1861 行的 `_build_hybrid_linear_kv_pool`，以 `use_dsa=True`、`kv_scale_ue8m0=True` 创建 `HybridLinearKVPool`，KDA 状态池通过 `mamba_pool=state.pool` 接入。`_build_dsa_kv_pool` 里的 GLM CP 层切分（`get_glm_dsa_cp_layer_shard_info`）和 `dcp_replicated` 对 K3.1 不生效。分配器走 2947-2959 行的分页分支，容量和页大小都乘 `attn_dcp_size`。

**方案 A 下需要关注的点：**

1. **DCP 宽度来源。** 本文件所有 DCP 换算都读 `get_parallel().attn_dcp_size`（`logical_token_capacity`、`loc_space_scale`、分配器放大、`mla_draft_rows_are_dcp_collapsed`）。只要计划 §4.2 让 `attn_dcp_size == attn_cp_size`，这些代码本身应该不用改（推断）。需要测试确认方案 A 下 `get_parallel().dcp_enabled`（2937 行）也为真，否则 `page_size == 1` 时会选错成逐 token 分配器。
2. **放大页对齐。** 分配器页大小是 `page_size * attn_dcp_size`，池的页大小是 `page_size`（target 的 `loc_space_scale == 1`）。本文件不要求 `page_size % dcp == 0`；该约束来自 `mem_cache/pool_host/base.py:242`（HiCache 主机池）。方案 A 的 chunked prefill 和 radix 对齐要以分配器的放大页为准。
3. **mamba/KDA 状态池与 KV 池的划分。** `_handle_max_mamba_cache` 按 `mamba_full_memory_ratio` 切分预算，与 DCP 无关；KDA 状态不随 DCP 切分。方案 A 下每卡 KDA 状态大小取决于 `mamba2_cache_params` 里的每卡头数（`num_k_heads_per_tp`）。CP-TP group sharing 下 KDA 头按整个 TP 组切分（每卡 12 个头），要确认 `mamba_cache_per_req` 是按这个切分计算的，而不是按 `attn_tp_size = 1` 计算（推断，计划未提及）。
4. **MLA 头数与 `attn_tp_size`。** MHA 类池的头数用 `get_num_kv_heads(attn_tp_size, attn_dcp_size)`；K3.1 的 MLA 池不用头数，所以方案 A 的 `attn_tp_size = 1` 不影响 KV 池形状。
5. **投机解码 draft。** K3.1 MTP draft 的 `loc_space_scale = dcp_size // mla_attention_dcp_size(..., is_nextn=True)`，`_derive_pool_sizes` 用它放大 draft 容量。`_build_mtp_kv_shard_pool` 在 DCP>1 时直接报错（1910 行），说明页切分 MTP 与 DCP 不能共存。计划把 DFLASH/DSPARK 放到 P4，但 MTP(EAGLE) draft 的池尺寸在方案 A 下同样需要验证（计划未提及）。
6. **`calculate_mla_kv_cache_dim`（3596）。** 非 PD 模式下只要 prefill 或 decode 有一侧是 raw fp8 后端，整个池就用 raw 行宽。方案 A 的 prefill 由 PCP 路径执行、decode 由 DCP 路径执行，计划 §8 第 3 问选择的 DSA 后端（FlashMLA / TRT-LLM / prims_ts）会决定行宽，两侧的读写 kernel 必须接受同一种布局。

**计划可能不准确的地方：**

- **"KV 池容量按 DCP 切分后计算"的实际位置不在本文件。** 每 token 字节数由 `model_executor/pool_configurator.py` 计算：MLA latent 行按每卡物理行计费，DSA indexer 在 DCP 下乘 `dcp_size`，除非 `dsa_shards_index_k()` 为真（`pool_configurator.py:452-462`，注释"Index-K spans the widened slot space unless K3.1 shards it"）。K3.1 的 indexer 是否切分取决于 `set_dsa_shards_index_k()`（计划 §4.7 提到），它的结果会直接改变这里的计费。所以 §4.9 的修改主要在 `pool_configurator.py`，本文件只负责把结果放大成分配器的 id 空间。
- 本文件 `_validate_prefill_only_disable_kv_cache_pool_family` 的报错文案写明"no context-parallel attention"，但只针对 `--prefill-only-disable-kv-cache`，与方案 A 无关。
