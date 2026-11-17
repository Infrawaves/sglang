# `python/sglang/srt/arg_groups/hicache_hook.py` 代码理解

- 文件路径：`python/sglang/srt/arg_groups/hicache_hook.py`
- 行数：516
- 所属模块：参数解析阶段的分层 KV 缓存（HiCache：L1 设备、L2 主机、L3 存储）相关参数的规范化与校验，包括外部 linker、主机内存模式、存储 key 方案、布局与 I/O 后端兼容性，以及 HiCache 与 DCP 的兼容性。

## 1. 整体功能

HiCache 有多个相互约束的参数：主机内存布局 `hicache_mem_layout`（`layer_first`、`page_first`、`page_first_direct`、`page_unified`）、I/O 后端 `hicache_io_backend`（`kernel`、`direct` 等）、存储后端 `hicache_storage_backend`（`mooncake`、`file`、`npu_memcache` 等）、主机内存模式 `hicache_host_memory_mode`（`cache` 或 `buffer_only`）、存储 key 方案等。这个文件把用户输入规范化成一个合法组合：直接冲突时报错，可以自动改写的就通过 `declare_resolution` 改写（例如 direct I/O 时把 `page_first` 改成 `page_first_direct`）并打告警。

在 pipeline 中有两个入口：
- `handle_hicache_ratio_default`（pipeline 第 83 行）：很早运行，在 dummy 模型提前返回之前，保证直接使用 `HostKVCache` 的单元测试和 dummy 启动不会看到为 None 的 ratio。
- `handle_hicache`（pipeline 第 285 行）：在 attention 后端解析（255）、page size 默认值（269）、`handle_prefill_only_disable_kv_cache`（283）之后，在 `handle_data_parallelism`（287）和 `handle_context_parallelism`（291）之前。

`handle_hicache` 的流程：外部 linker 配置校验（开启 linker 时做完互斥检查直接返回）→ 若 HiCache、decode 卸载、decode 的 host_pool 撤回备份都未启用则返回 → 主机内存模式校验 → key 方案 → 布局与 I/O 兼容 → 存储与布局兼容 → unified memory 与布局 → draft node merge 布局 → decode host 接收阈值与 `layer_first` 的一致性 → DCP 兼容性。

## 2. 主要依赖与调用方

依赖：
- `arg_groups/model_override_base`：`resolving_view`、`use_mla_backend`（DCP 兼容检查要求 MLA）。
- `arg_groups/overrides.declare_resolution`。
- 延迟导入：`mem_cache/unified_cache/linker_config.UnifiedCacheLinkerConfig`、`mem_cache/hybrid_cache/hybrid_cache_controller.HybridCacheController`、`mem_cache/storage/mooncake_store/mooncake_store.MooncakeStoreConfig`、`mem_cache/hicache_draft_node_merge.DRAFT_NODE_MERGE_LAYOUTS`。

调用方（grep 确认）：
- `arg_groups/pipeline.py:78-83, 285`；`arg_groups/resolution_hooks.py` 登记。
- 单元测试：`test/registered/unit/server_args/test_server_args.py`、`test/registered/unit/mem_cache/test_hicache_unified_key_scheme.py`、`test_decode_retraction_backup.py`、`test_external_linker_factory.py`。

## 3. 类与函数说明

文件里没有类。

| 名称 | 行号 | 作用 |
|---|---|---|
| `handle_hicache` | 15 | 主入口，流程见第 1 节。外部 linker 部分：配置必须是 dict；dynamic 后端时解析配置；`enable_linker_mla_dedup` 需要 Mooncake linker；开启 linker 时禁止 PP、关闭 radix cache 或自定义 radix 后端、HiCache、存储后端，检查后直接返回。decode host 接收阈值 `< 1` 时把布局固定为 `layer_first`（共享主机池要和 KV 传输的逐层 buffer 对齐），最后再检查布局是否仍是 `layer_first`。 |
| `validate_mla_hicache_per_node` | 113 | `--enable-mla-hicache-per-node` 的约束：不支持 PP、PD decode、**`dcp_size > 1`**（注释：当前传输约定下 DCP 页按 rank 切分）；Mooncake 存储下不能用 `standalone_storage=True`。 |
| `validate_hicache_draft_node_merge` | 149 | `--hicache-draft-node-merge` 的约束：需要 per-node MLA HiCache、Mooncake 存储、**`attn_cp_size == 1`**（注释：未验证 CP 各 rank 对同一页是否有相同的主机下标）、write-through 类写策略。 |
| `validate_hicache_draft_node_merge_layout` | 175 | 布局必须在 `DRAFT_NODE_MERGE_LAYOUTS` 中，提前拒绝构建 draft 主机池时才会发现的错误。 |
| `validate_hicache_storage_page_size` | 190 | `hicache_storage_page_size` 必须是 `page_size` 的正整数倍；倍数大于 1 时还要求倍数整除 128（一个 L3 批次 128 页，组不能跨批），且必须开启 HiCache、使用 Mooncake、不用 PP、**不用 DCP**、不是 PD decode、不是 `buffer_only`。 |
| `handle_hicache_ratio_default` | 229 | 检查 `hicache_host_memory_fraction` 在 (0, 1]；非 decode 服务器且未给 `hicache_ratio` 时，`buffer_only` 默认 1.2，否则 2.0；用户显式给了 ratio 或 size 时把 fraction 置 None（关闭自动大小）。 |
| `resolve_hicache_dcp_compatibility` | 258 | `dcp_size > 1` 且开启 HiCache 时：L3 存储加 `buffer_only` 报 `NotImplementedError`；投机解码只允许 DSPARK、EAGLE3、DFLASH；不支持 LMCache、HiSparse；只支持 MLA 模型（DCP 下标换算在 `MLATokenToKVPoolHost` 里）。通过后打日志：主机池使用放大的逻辑 slot 计数，在传输边界做每 rank 的物理换算。 |
| `resolve_hicache_key_scheme` | 302 | 非 `unified` key 方案时：`page_unified` 布局加存储后端报错，head/layer 分区参数报错。`unified` 方案时：分区值必须为正，需要存储后端且只支持 `file`，不支持投机解码，**不支持 DCP**（DCP rank 持有子页切片，需要 token 粒度的 key）；强制布局为 `page_unified`、I/O 后端为 `kernel`。 |
| `resolve_layout_io_compatibility` | 381 | `page_unified` 必须配 `kernel` I/O；`page_first_direct` 配 `kernel` 时改用 `direct`；`page_first` 配 `direct` 时改成 `page_first_direct`。 |
| `resolve_storage_layout_compatibility` | 417 | Mooncake 加 DCP 时直接返回（DCP MLA 页支持 Mooncake 多 buffer 传输，保持请求的布局）；否则 Mooncake/NPU MemCache 不支持 `layer_first`，按 I/O 后端改成 `page_first_direct` 或 `page_first`。 |
| `resolve_unified_memory_layout_io` | 448 | unified memory 下 `page_first_direct` 改成 `page_first` + `kernel` I/O（unified 池的设备行是跨步的，direct 拷贝假设行是紧密排列的）。 |
| `validate_hicache_host_memory_mode` | 467 | 模式必须是 `cache` 或 `buffer_only`；非 decode 服务器必须有主机池大小。`buffer_only` 需要存储后端，`write_back` 自动改为 `write_through`，且不支持 PD decode 实例。 |

## 4. 与 PCP+DCP 方案 A 的关系

计划 §4.1 要求在这个文件里"第一期显式拒绝方案 A 和 HiCache 的组合"；§4.11（P4）再处理 `mem_cache/hicache_storage.py:69-77` 里按 `tp_rank % dcp_size == dcp_rank` 判断的逻辑。

**需要注意的点：**
- **拒绝条件放在哪里**：最自然的位置是 `resolve_hicache_dcp_compatibility`（258），它已经只在 `dcp_size > 1` 且开启 HiCache 时运行。拒绝条件可以写成 `enable_prefill_cp` 为真（方案 A 是唯一允许 DCP 和 prefill CP 同时出现的情况）。
- **调用顺序**：`handle_hicache` 在 pipeline 第 285 行运行，早于 `handle_context_parallelism`（291）。这时 `enable_cp_tp_group_sharing` 还没被声明，不能用它判断是否处于方案 A；只能用 `enable_prefill_cp` 和 `dcp_size`（推断）。（计划未提及）
- **拒绝范围**：计划 §2.2 写的是"HiCache L3 放到 P4"，§4.1 写的是"第一期显式拒绝方案 A 和 HiCache 的组合"。`resolve_hicache_dcp_compatibility` 处理的是 L2（设备到主机）路径，L3 另有存储层逻辑。第一期拒绝整个 HiCache（L2 + L3）还是只拒绝 L3，计划两处的说法需要统一。另外 `disaggregation_decode_enable_offload_kvcache` 和 decode 的 host_pool 撤回备份也会进入 `handle_hicache` 的主流程，方案 A 不在 decode 节点上，影响不大（推断）。
- **K3.1 DCP 目前允许 HiCache**：`model_overrides/deepseek_v2.py` 的 `_dsa_dcp_overrides` 会对其他 DSA 模型拒绝 HiCache + DCP，但对 K3.1 跳过（`deepseek_v2.py:105`）。所以 K3.1 在 TP8+DCP8 下 HiCache 是可用的，方案 A 的拒绝需要在这里新增，不能依赖已有检查。
- **已经存在的 CP/DCP 限制**：`validate_hicache_draft_node_merge` 拒绝 `attn_cp_size > 1`，`validate_mla_hicache_per_node`、`validate_hicache_storage_page_size`、`resolve_hicache_key_scheme` 拒绝 `dcp_size > 1`。这些在方案 A 下继续生效。
- **P4 时的考虑**：`resolve_hicache_dcp_compatibility` 的日志说明主机池使用"放大的逻辑 slot 计数 + 传输边界的每 rank 物理换算"。方案 A 下 `dcp_rank = attn_cp_rank`，owner 规则不变（`pos % N`），主机池的换算本身可以复用；需要改的是计划 §4.11 提到的 rank 判断（推断）。
