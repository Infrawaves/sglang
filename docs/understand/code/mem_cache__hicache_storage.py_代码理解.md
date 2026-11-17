# `python/sglang/srt/mem_cache/hicache_storage.py` 代码理解

- 文件路径：`python/sglang/srt/mem_cache/hicache_storage.py`
- 行数：1333
- 所属模块：分层 KV 缓存（HiCache）的 L3 存储层，定义存储后端的抽象接口、配置、传输描述符，以及一个本地文件后端 `HiCacheFile`。

## 1. 整体功能

HiCache 有三级：L1 是 GPU 上的 KV 池，L2 是主机内存池（`HostKVCache`），L3 是外部存储（文件、Mooncake、NIXL 等）。本文件定义 L2 与 L3 之间的接口：

- **配置与身份**：`HiCacheStorageConfig` 描述当前 rank 在 TP/PP/CP/DCP/DP 中的位置，决定存储 key 的后缀和"谁负责写"。DCP 开启时，它校验 DCP 组是 TP 组内的连续切片，且逻辑页大小能被 `dcp_size` 整除。
- **传输描述**：`PoolTransfer`、`PoolTransferResult`、`PoolHitPolicy`、`PoolName` 等描述多池（KV、indexer、Mamba、SWA、DSV4 压缩池、draft 池等）的 v2 批量接口。
- **L3 布局合并**：`merge_l3_layouts` 等函数在多 scheduler、多 PP stage 之间合并"L3 对象布局"回复。
- **后端抽象**：`HiCacheStorage` 是所有后端的基类，提供 v1/v2 批量读写和存在性查询接口。
- **测试注入**：`HiCacheStorageFailureInjector` 包装任意后端，按环境变量注入随机延迟和失败。
- **文件后端**：`HiCacheFile` 把每页存成一个 `.bin` 文件，key 后缀编码模型名、TP/PP/CP/DCP 身份；支持 unified key scheme（与拓扑无关的分块 key）、元数据缓存和 LRU 淘汰。

数据流：`HiCacheController`（`managers/cache_controller.py`）构造 `HiCacheStorageConfig`，通过 `storage/backend_factory.py` 创建后端，再注册主机池；之后预取和回写通过 `batch_exists_v2` / `batch_get_v2` / `batch_set_v2` 按页进行。

## 2. 主要依赖与调用方

**依赖：** `sglang.srt.environ.envs`（测试注入和文件后端相关环境变量）、`mem_cache/pool_host`（`HostKVCache`，仅类型）、`mem_cache/storage/file/lru_file_evictor.LRUFileEvictor`（在 `HiCacheFile.__init__` 中延迟导入，避免循环依赖）。

**调用方（grep 确认）：**
- `managers/cache_controller.py:904`：构造 `HiCacheStorageConfig`，其中 `dcp_rank=get_parallel().attn_dcp_rank`，`attn_cp_rank` 来自 `attn_cp_group` 中的 rank（`get_attn_cp_rank_and_size`，第 478 行）。
- `mem_cache/storage/backend_factory.py:96, 227`：注册 `"file"` 后端为 `HiCacheFile`，并用 `_maybe_wrap_hicache_storage_failure_injector` 包装。
- `managers/data_parallel_controller.py`、`entrypoints/http_server.py`、`entrypoints/engine.py`：`merge_l3_layouts`、`merge_scheduler_info_layouts`、`disabled_l3_layout`。
- `mem_cache/hybrid_cache/hybrid_cache_controller.py`、`mem_cache/unified_radix_cache.py`、`mem_cache/pool_host/group.py`：`merge_source_transfers`、`count_pool_hits`、`PoolTransfer`。
- 其他存储后端（`storage/mooncake_store`、`storage/nixl`、`storage/shm`、`storage/umbp` 等）继承 `HiCacheStorage` 并使用 `l3_page_span`。

## 3. 类与函数说明

### 3.1 配置与数据结构

| 名称 | 行号 | 作用 |
|---|---|---|
| `STORAGE_BATCH_SIZE` | 24 | 单次批量存储 IO 的最大页数（128）。 |
| `HiCacheStorageConfig` | 28 | rank 身份与存储选项的 dataclass：TP/PP/CP/DP 的 rank 和 size、是否 MLA、unified 后缀与分块范围、`dcp_size`/`dcp_rank`、`logical_page_size`、`rank_replicated`、`storage_page_group` 等。 |
| `HiCacheStorageConfig.__post_init__` | 61 | 校验 DCP 参数：`dcp_rank` 在范围内；`dcp_size > 1` 时要求是 MLA 模型、`tp_size % dcp_size == 0` 且 `tp_rank % dcp_size == dcp_rank`（DCP 组是 TP 内的连续切片），并要求 `logical_page_size` 为正且能被 `dcp_size` 整除。不满足时抛 `ValueError`。 |
| `HiCacheStorageConfig.is_rank_replicated` | 82 | 属性：是否各 rank 存同一份整页。未显式设置 `rank_replicated` 时取 `is_mla_model`。 |
| `HiCacheStorageConfig.is_storage_writer` | 88 | 属性：当前 rank 是否负责写共享存储对象。unified key 下只有 CP rank 0 写；否则非副本模型每个 rank 都写，副本模型只有 `tp_rank == dcp_rank` 的 rank 写（DCP 关闭时即 TP rank 0，DCP 开启时是第一个 DCP 组的每个 shard）。 |
| `HiCacheStorageExtraInfo` | 102 | 调用附加信息：前缀页 hash（`prefix_keys`）、`start_page`（分组页时需要）、`storage_plans`（存在性查询返回的分段链）。 |
| `PoolName` | 113 | 池名枚举（KV、MAMBA、SWA、INDEXER、DSV4 各压缩池、DRAFT 系列等），用作 `PoolTransfer` 标识。 |
| `PoolName.__str__` | 148 | 返回枚举值字符串。 |
| `PoolHitPolicy` | 152 | 附加池的命中策略：`ALL_PAGES`（前缀内每页都要有，如 DSA indexer）、`TRAILING_PAGES`（只要求末尾若干页，如 Mamba/SWA）、`BLOCK_GROUPS`（按块组判断，不限制 KV 命中）。 |
| `PoolTransfer` | 169 | v2 接口的单池传输描述符：池名、host/device 索引、key 列表、命中策略、要加载的节点、`indices_from_pool`（sidecar 复用其他池的索引）、块组几何参数。 |
| `merge_source_transfers` | 189 | 把同一个池的多个独立传输合并成一个（拼接索引和 key），供从该池派生索引的 sidecar 使用；只有一个时原样返回，没有时返回 `None`。 |
| `SidecarPoolSpec` | 218 | 冻结 dataclass：sidecar 池名、复用索引的源池名和命中策略。 |
| `PoolTransferResult` | 227 | 存在性查询或传输结果：`kv_hit_pages`、各附加池命中页数、各 rank 可恢复的前缀长度集合、分段链。 |
| `PoolTransferResult.empty` | 243 | 返回全零结果。 |
| `PoolTransferResult.update_kv_hit_pages` | 246 | 跨批累积 KV 命中页数（取最大值）。 |
| `PoolTransferResult.update_extra_pool_hit_pages` | 250 | 记录各附加池的实际成功页数。 |

### 3.2 布局工具函数

| 名称 | 行号 | 作用 |
|---|---|---|
| `l3_page_span` | 259 | 一个 L3 key 覆盖的逻辑 slot 数：`page_size × dcp_size`。DCP 下一个 key 对应放大后的逻辑页，每个 rank 各存一页物理页。 |
| `count_pool_hits` | 267 | 把每池的成功列表转成"从头开始连续成功的页数"。 |
| `disabled_l3_layout` | 274 | 构造一个 `enabled=False` 的布局回复，附带原因。 |
| `_l3_layout_pp_rank` | 286 | 从布局回复里取 `pp_rank`。 |
| `_l3_layout_stage_independent` | 291 | 去掉 `pp_rank` 和 `pools` 后的部分，即各 PP stage 必须一致的字段。 |
| `_l3_layout_unavailable` | 300 | 把布局标为不可用，清空 pools 并写入原因。 |
| `merge_l3_layouts` | 310 | 合并多个 scheduler 的布局回复：副本之间必须完全一致；不同 PP stage 的 pools 按 stage 顺序拼接，合并后 `pp_rank` 设为 `None`；stage 不全或字段不一致时返回禁用布局。全为 `None` 时返回 `None`（留给旧版引擎）。 |
| `merge_scheduler_info_layouts` | 351 | 把合并后的布局写回每个 scheduler 启动信息，保证任取一个都有代表性。 |

### 3.3 `HiCacheStorage`（第 359 行，抽象基类）

| 名称 | 行号 | 作用 |
|---|---|---|
| `register_mem_pool_host` | 366 | 记录锚点主机池。 |
| `register_mem_host_pool_v2` | 369 | 按池名记录主机池到 `registered_pools`。 |
| `on_host_pools_registered` | 374 | 首轮主机池注册完成后的钩子，默认空；后端可在这里完成一次性注册，抛异常会使 attach 失败。 |
| `batch_exists_v2` | 384 | 按池命中策略查询最长可用前缀，返回 `PoolTransferResult`。基类抛 `NotImplementedError`。 |
| `batch_get_v2` | 417 | 按 `PoolTransfer` 从存储读到主机内存，返回每池的成功列表。基类未实现。 |
| `batch_set_v2` | 428 | 按 `PoolTransfer` 从主机内存写到存储。基类未实现。 |
| `batch_get_v1` | 439 | v1 批量读接口，基类为空实现。 |
| `batch_set_v1` | 451 | v1 批量写接口，基类为空实现。 |
| `get` | 464 | 抽象方法：读单个 key。 |
| `batch_get` | 478 | 抽象方法：批量读（标注待废弃）。 |
| `set` | 491 | 抽象方法：写单个 key。 |
| `batch_set` | 506 | 抽象方法：批量写（标注待废弃）。 |
| `exists` | 520 | 抽象方法：key 是否存在。 |
| `batch_exists` | 528 | 默认逐个调用 `exists`，返回从头连续存在的 key 数。 |
| `clear` | 541 | 默认空操作。 |
| `get_stats` | 544 | 默认返回 `None`。 |

### 3.4 `MetadataCache`（第 548 行）

带 TTL 的线程安全"文件存在"缓存，`HiCacheFile` 用它减少 `stat` 调用。

| 名称 | 行号 | 作用 |
|---|---|---|
| `__init__` | 549 | 设置 TTL、字典和锁。 |
| `add` | 555 | 首次加入时记录单调时钟时间戳。 |
| `remove` | 560 | 删除 key。 |
| `contains` | 564 | 查询；TTL 为 -1 表示永不过期，过期则删除并返回 `False`。 |
| `clear` | 575 | 清空。 |

### 3.5 测试注入

| 名称 | 行号 | 作用 |
|---|---|---|
| `_maybe_wrap_hicache_storage_failure_injector` | 580 | 若设置了 `SGLANG_TEST_HICACHE_BACKEND_RANDOM_DELAY` 或 `SGLANG_TEST_HICACHE_BACKEND_RANDOM_FAILURE`，用 `HiCacheStorageFailureInjector` 包装后端，否则原样返回。 |
| `HiCacheStorageFailureInjector` | 602 | 代理类：每次存储 API 调用前注入随机延迟，并按概率返回失败结果。 |
| `HiCacheStorageFailureInjector.__init__` | 610 | 保存被代理后端、延迟上限、失败概率。 |
| `._thread_local_random` | 620 | 返回线程本地的 `random.Random`，避免各 TP/PP rank 因相同种子产生相同的随机序列。 |
| `._inject_delay` | 630 | 延迟上限大于 0 时随机 sleep。 |
| `._inject_failure` | 637 | 按概率返回 `True` 表示本次调用失败。 |
| `register_mem_pool_host`、`register_mem_host_pool_v2`、`on_host_pools_registered` | 645、648、651 | 直接转发给后端。 |
| `batch_exists_v2` | 654 | 注入后转发；失败时返回 `PoolTransferResult.empty()`。 |
| `batch_get_v2`、`batch_set_v2` | 666、680 | 注入后转发；失败时每池返回全 `False`。 |
| `batch_get_v1`、`batch_set_v1` | 694、706 | 注入后转发；失败时返回全 `False`。 |
| `get`、`batch_get` | 718、730 | 注入后转发；失败时返回 `None` / 全 `None`。 |
| `set`、`batch_set` | 742、755 | 注入后转发；失败时返回 `False`。 |
| `exists`、`batch_exists` | 768、775 | 注入后转发；失败时返回 `False` / 0。 |
| `clear`、`get_stats`、`close` | 784、787、790 | 直接转发。 |

### 3.6 `HiCacheFile`（第 794 行，本地文件后端）

| 名称 | 行号 | 作用 |
|---|---|---|
| `_init_config_suffix` | 795 | 计算 key 后缀。unified 模式：用 `unified_suffixes` 作为分块坐标，要求非空且唯一、`dcp_size == 1`、分块范围无缝覆盖整页。普通模式：后缀由模型名、PP（`_{pp_size}_{pp_rank}`）、CP（`_cp{attn_cp_rank}_{attn_cp_size}`）组成；`tp_config_suffix` 再加 TP rank/size；副本模型的 `config_suffix` 不含 TP rank；`dcp_size > 1` 时两者都追加 `_tp{tp}_dcp{dcp_rank}_{dcp_size}_page{logical_page_size}`。 |
| `__init__` | 878 | 确定存储目录（环境变量、extra_config 或默认 `/tmp/hicache`），计算后缀，TP rank 0 且 CP rank 0 时创建目录，按配置启用 `MetadataCache` 并扫描已有文件，最后创建 `LRUFileEvictor`，`owned_suffixes` 决定本 rank 负责淘汰哪些文件。 |
| `_validate_unified_page_bytes` | 945 | unified 多分块时校验分块总字节数等于页字节数。 |
| `register_mem_pool_host` | 953 | 校验页字节数后调用基类注册。 |
| `register_mem_host_pool_v2` | 960 | unified 模式只允许 KV 池（否则 `NotImplementedError`），校验后注册。 |
| `_page_chunks` | 972 | 把一页按 unified 分块范围切成 `(key+suffix, 字节切片)` 列表。 |
| `_get_suffixed_key` | 981 | 给 key 加后缀；Mamba 池用 `tp_config_suffix`（Mamba 状态按 TP 切分），其他用 `config_suffix`。 |
| `_get_component_key` | 987 | 非 KV 池先加池名（`_log_key`）再加后缀。 |
| `_get_component_keys` | 992 | 返回一个逻辑页对应的所有文件 stem；unified 模式下每个分块一个，且不支持 sidecar 池。 |
| `_scan_existing_files_to_metadata_cache` | 1001 | 启动时扫描目录，把属于本 rank/模型的 `.bin` 文件加入元数据缓存。 |
| `get` | 1016 | 读一页。普通模式直接读文件；unified 模式先读到临时缓冲，所有分块成功后再整体拷贝，避免暴露残缺页。 |
| `_get_file` | 1035 | 读单个文件：校验文件大小、`readinto` 读满，成功后刷新 LRU 和元数据缓存；`OSError` 时记录警告并返回 `None`。 |
| `batch_get` | 1055 | 逐个调用 `get`。 |
| `set` | 1068 | 写一页；unified 模式逐块写，最后用 `exists` 确认所有块都在（防止写后一块时淘汰了前一块）。 |
| `_set_file` | 1084 | 写单个文件：已存在则只刷新 LRU；否则向 evictor 预留空间，写临时文件后 `os.replace` 原子替换，失败时回滚预留并删除临时文件。 |
| `batch_set` | 1123 | 逐个调用 `set`，遇到失败立即返回 `False`。 |
| `exists` | 1135 | 一个逻辑页的所有组件文件都存在才返回 `True`。 |
| `_file_exists` | 1138 | 单文件存在性：普通模式先查元数据缓存；unified 模式额外校验文件大小；结果回写到元数据缓存。 |
| `_collect_existing_component_keys` | 1168 | 为一批 key 和附加池收集实际存在的文件名集合。 |
| `batch_exists_v2` | 1209 | 计算最长连续 KV 前缀，再按各附加池的 `ALL_PAGES` / `TRAILING_PAGES` 策略求可恢复前缀集合的交集；不支持的策略（如 `BLOCK_GROUPS`）抛 `ValueError`。 |
| `_log_key` | 1260 | KV 池返回原 key，其他池返回 `{key}.{pool_name}`。 |
| `_read_page` | 1263 | 从存储读一页到主机池的指定页偏移。 |
| `_write_page` | 1272 | 把主机池指定页写到存储。 |
| `_batch_io_v2` | 1280 | v2 批量读写的公共实现：按 `l3_page_span` 校验 `host_indices` 长度等于 `len(keys) × page_span`，不匹配时该池全部失败；否则对每个 key 取对应页起点调用 `op_fn`。 |
| `batch_get_v2`、`batch_set_v2` | 1306、1313 | 分别以 `_read_page`、`_write_page` 调用 `_batch_io_v2`。 |
| `clear` | 1320 | 删除目录下所有文件，清空 evictor 和元数据缓存。 |

## 4. 与 PCP+DCP 方案 A 的关系

计划 §4.11（P4 阶段）提到本文件第 69-77 行："判断条件从 `tp_rank % dcp_size == dcp_rank` 改成按 CP rank 判断"。计划 §4.1 同时要求第一期在 `hicache_hook.py` 中显式拒绝方案 A 与 HiCache 的组合，所以本文件在 P1-P3 不需要改。

读代码后的补充：

1. **第 68-73 行的校验在典型配置下其实能通过。** 方案 A 的典型配置是 `tp = attn_cp = dcp = 8`。此时 `attn_cp` 组复用 TP 组，`dcp_rank = attn_cp_rank`（计划 §4.2），而 `attn_cp_rank` 等于 `tp_rank`（推断，单实例、无 DP/PP 时成立），所以 `tp_rank % 8 == dcp_rank` 成立。要改这条校验，主要是为了表达"DCP 组就是 CP 组"的语义，以及覆盖 `attn_cp_size < tp_size` 的情况（第一期不支持）。计划的描述可以更准确一些。
2. **`is_storage_writer`（第 88-98 行）。** 方案 A 下 `tp_rank == dcp_rank` 对所有 8 个 rank 都成立，于是每个 rank 都是自己那份 shard 的写者，这正是 DCP 按 owner 存储的需要（推断）。如果以后 `attn_cp_rank != tp_rank`，这里的 `tp_rank == dcp_rank` 也要一起改（计划未提及）。
3. **`HiCacheFile._init_config_suffix` 的 key 后缀（第 852-876 行，计划未提及）。** 现在 `attn_cp_size > 1` 时后缀会加 `_cp{rank}_{size}`，`dcp_size > 1` 时再加 `_tp{tp}_dcp{rank}_{size}_page{P}`。方案 A 两者同时生效，后缀会同时含 CP 和 DCP 段。功能上每个 rank 的 key 仍然唯一，但方案 A 的 KV 布局和纯 TP8+DCP8 相同（都是按 `pos % N` 存储），带上 CP 段后两种部署无法共享 L3 文件。是否在方案 A 下省掉 CP 段，需要单独决定。
4. **unified key scheme 与 DCP 互斥。** 第 808-809 行在 unified 模式下要求 `dcp_size == 1`，`cache_controller._build_unified_plan` 也拒绝 DCP。方案 A 开启 DCP，因此只能用普通 key scheme（计划未提及）。
5. **`l3_page_span`（第 259 行）和 `_batch_io_v2`（第 1285 行）** 按 `page_size × dcp_size` 计算每个 key 覆盖的逻辑 slot。方案 A 的主机池仍按 DCP 放大页工作，这部分无需修改（推断）。
6. **其他后端。** Mooncake 等后端也使用 `HiCacheStorageConfig` 的 `dcp_rank`、`is_storage_writer` 和 `l3_page_span`。P4 修改时需要一起检查这些后端，不能只改本文件。
