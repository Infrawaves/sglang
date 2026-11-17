# `python/sglang/srt/mem_cache/pool_host/base.py` 代码理解

- 文件路径：`python/sglang/srt/mem_cache/pool_host/base.py`
- 行数：684
- 所属模块：HiCache 的 L2 主机内存池，定义所有主机 KV 池的抽象基类 `HostKVCache`，以及主机内存预算和 page_unified 回写暂存相关的工具函数。

## 1. 整体功能

HiCache 把 GPU 上的 KV（L1）换出到主机内存（L2），必要时再写到外部存储（L3）。`HostKVCache` 是 L2 主机池的公共基类，具体子类（MHA、MLA、DSA indexer、Mamba、DSV4 状态等）实现缓冲布局和 device↔host 传输 kernel。基类负责：

- **容量与页几何**：根据 `--hicache-size`（GB）或 `--hicache-ratio`，或者调用方给定的 `host_token_capacity`，算出主机池 token 数和页数。DCP 下传入的 `page_size` 是放大后的逻辑页（`page_size × dcp_size`），基类把它除以 `dcp_size` 得到每个 rank 的物理页大小。
- **主机内存预算**：按 cgroup 限制、本机 rank 数和 10 GB 预留计算可用主机内存，超出则报错。
- **slot 分配**：在"逻辑 slot 空间"（`size × dcp_size`）上做 alloc/free，带重复分配和重复释放检测。
- **DCP 页寻址**：`_storage_page_index` 把 L3 key 对应的逻辑页起点转换成本 rank 的物理 slot。
- **层切分**：device 池按层切分时，只为本 rank 拥有的层分配主机空间。
- **page_unified 回写**：准备 device 端暂存缓冲，供 JIT kernel 先重排再做 D2H 拷贝。

## 2. 主要依赖与调用方

**依赖：**
- `kernels/ops/kvcache/hicache.can_use_page_unified_write_back_jit_kernel`：判断回写 JIT kernel 能否构建。
- `mem_cache/host_memory.available_host_memory_bytes`：读取主机可用内存（含 cgroup）。
- `mem_cache/pool_host/common`：`HostTensorAllocator`、`get_allocator_from_storage`、`_cuda_host_unregister`。
- `mem_cache/pool_host/page_unified.PageUnifiedLayout`：page_unified 布局的几何描述。
- `runtime_context.get_parallel`：`launch_world_size`、`nnodes`、`pp_group`。

**调用方（grep 确认）：**
- 子类：`pool_host/mha.py`（`MHATokenToKVPoolHost`、`MHATokenToKOnlyPoolHost`）、`pool_host/mla.py`、`pool_host/dsa.py`（`DSAIndexerPoolHost`）、`pool_host/mamba.py`（`MambaPoolHost`，同时用 `sync_fixed_hicache_size`）、`pool_host/unified.py`、`pool_host/minimax_m3_nvfp4.py`、`memory_pool_host.py`（`DeepSeekV4StateHostPool`）。
- `mem_cache/hicache_auto_size.py`、`mem_cache/mla_host_node_leader.py`：`host_memory_budget_scope`、`ranks_per_host`。
- `mem_cache/kv_cache_builder.py`：`prepare_page_unified_write_back_staging`。
- `mem_cache/pool_host/group.py`、`mem_cache/l2_transfer.py`：`shared_host_layout_domains`。
- `mem_cache/hybrid_cache/hybrid_pool_assembler.py`、`mem_cache/kv/alloc/id_space.py` 等：构造主机池时传入 `dcp_size` 和 `dcp_rank=parallel.attn_dcp_rank`。

## 3. 类与函数说明

### 3.1 模块级常量与函数

| 名称 | 行号 | 作用 |
|---|---|---|
| `HICACHE_HOST_MEMORY_RESERVE_BYTES` | 33 | 计算主机池大小时给系统和其他进程预留的内存（10 GB）。 |
| `_WRITE_BACK_STAGING_PAGE_CHUNK` | 35 | 回写暂存缓冲的最大页数（64）。 |
| `prepare_page_unified_write_back_staging` | 38 | 为 page_unified 布局准备或复用 device 端暂存缓冲。非 CUDA/HIP 或 JIT kernel 不支持该行宽时抛 `ValueError`；复用已有缓冲时校验 dtype、device、连续性和形状，不一致也抛 `ValueError`。 |
| `_host_memory_budget` | 76 | `ContextVar`，保存 `host_memory_budget_scope` 内的预算快照。 |
| `uses_shared_host_layout` | 81 | 类型守卫：对象是 `HostKVCache` 且有 `shared_allocation_domain` 时返回 `True`。 |
| `shared_host_layout_domains` | 88 | 从一组主机池里收集去重后的共享分配域。 |
| `host_memory_budget_scope` | 102 | 上下文管理器：在作用域内用同一个内存快照为所有池记账，避免多次采样 psutil。 |
| `ranks_per_host` | 111 | 每台机器上的 launch rank 数（`launch_world_size // nnodes`），分布式未初始化时返回 1。不发起集合通信。 |
| `host_memory_budget_bytes` | 127 | 当前 rank 可申请的主机内存。在预算作用域内从快照扣除并返回扣除前的值；否则用可用内存减去预留，再除以每机 rank 数。 |
| `sync_fixed_hicache_size` | 154 | 固定 `--hicache-size` 时，各 PP stage 每 token 字节数不同，用 PP 组内 all-reduce MIN 统一主机 token 容量。 |
| `synchronized` | 196 | 装饰器：在 `self.lock` 下执行方法。 |

### 3.2 `HostKVCache`（第 205 行，抽象基类）

类属性默认值：`dcp_size = 1`、`dcp_rank = 0`、`shared_allocation_domain = None`、`stores_page_envelope = False`、`storage_format_tag = None`、`host_buffer_read_only = False`、`_strided_device_row_bytes = None`（第 359 行）。

| 名称 | 行号 | 作用 |
|---|---|---|
| `__init__` | 215 | 记录 dtype、布局、设备等参数。断言 `page_size % dcp_size == 0`（第 242 行），把物理页大小设为 `page_size // dcp_size`。容量：给定 `host_token_capacity` 时要求为正且按物理页对齐；否则按 `host_size`（GB，经 `sync_fixed_hicache_size`）或 `device_capacity_tokens × host_to_device_ratio` 计算，并多留一个 padding 页。非 dummy 时分配缓冲，最后创建锁并 `clear()`。 |
| `_allocate_host_buffer` | 286 | 主机池比 device 池小时给出警告；按 `host_memory_budget_bytes` 检查内存（只读的 follower 不重复记账），不足时抛 `ValueError`；打印分配信息（含 MTP draft 层数）；调用子类 `init_kv_buffer`。 |
| `destroy` | 337 | 进程退出前在用户态解除 pinned 缓冲的 `cudaHostRegister`，避免 SIGKILL 回收时长时间卡住。可重复调用。 |
| `_init_device_row_stride` | 361 | 记录 device KV 缓冲的行步长（token-major 的 unified 池行步长不等于行宽）。多个缓冲步长不一致或混用 packed 与 strided 时抛 `NotImplementedError`。 |
| `device_rows_packed` | 388 | 属性：device 行是否紧密排列。 |
| `device_row_stride_bytes` | 392 | 属性：相邻 device KV 行之间的字节数。 |
| `_require_packed_device_rows` | 396 | 某个 kernel 要求 packed 行而 device 行是 strided 时抛 `NotImplementedError`。 |
| `_page_ids` | 405 | 从页对齐的 token 索引中每隔 `page_size` 取一个，得到页 id 列表。 |
| `_init_page_unified_staging` | 414 | 按页数上限初始化回写暂存缓冲，并打开 `can_use_write_back_jit`。 |
| `get_size_per_token` | 432 | 抽象方法：每 token 字节数。 |
| `_is_device_layer_sharded` | 435 | device 池是否按层切分。 |
| `_device_owned_layer_range` | 439 | 本 rank 存储的连续本地层范围 `[start, end)`；不切分时为全部层。 |
| `_effective_host_layer_num` | 449 | 主机池实际为本 rank 分配的层数（按 `layer_shard_size` 向上取整）。 |
| `_is_device_layer_owned` | 457 | 某层是否归本 rank。 |
| `_host_layer_index` | 461 | device 层号转主机池层下标。 |
| `_owned_device_layer_ids` | 465 | 本 rank 拥有的层号列表。 |
| `init_kv_buffer` | 470 | 抽象方法：分配主机 KV 缓冲。 |
| `load_to_device_per_layer` | 474 | 抽象方法：把某层 KV 从主机加载到 device。 |
| `backup_from_device_all_layer` | 490 | 抽象方法：把所有层 KV 从 device 备份到主机。 |
| `get_page_buffer_element_size` | 498 | 通过子类的 page buffer meta 得到一个存储元素的字节数；逻辑锚点池返回 `None`。 |
| `prepare_transfer_indices` | 509 | 返回在一次 L2 提交期间保持稳定的索引；普通池直接返回，共享压缩池会覆写。 |
| `peer_write_indices` | 521 | 传输对端会写入的 slot；普通池直接返回输入。 |
| `set_peer_write_gate` | 529 | 安装"对端可能正在写"的判定函数；普通池忽略。 |
| `backup_from_device_all_layer_physical` | 534 | 物理索引版本的备份，默认转发给 `backup_from_device_all_layer`。 |
| `load_to_device_per_layer_physical` | 541 | 物理索引版本的加载，默认转发给 `load_to_device_per_layer`。 |
| `get_data_page` | 561 | 抽象方法：取一页数据（L3 写出用）。 |
| `get_dummy_flat_data_page` | 568 | 抽象方法：取一页空数据缓冲（L3 读入用）。 |
| `set_from_flat_data_page` | 576 | 抽象方法：把一页扁平数据写入主机池。 |
| `is_stride_page_aligned` | 582 | 每页步长是否按 4096 字节对齐；基类打警告并返回 `False`（NIXL 文件后端的 O_DIRECT 会退回拷贝模式）。 |
| `clear` | 596 | 加锁重置 `mem_state`、空闲 slot 列表（长度为 `logical_size`）、延迟释放列表和 `slot_used` 标记。 |
| `available_size` | 610 | 空闲 slot 数加待合并的已释放 slot 数。 |
| `_merge_release_slots` | 613 | 把延迟释放的 slot 合并回空闲列表。 |
| `logical_size` | 626 | 属性：radix/controller 看到的 slot 数，等于 `size × dcp_size`（DCP 下 `dcp_size` 个逻辑 slot 共享一行物理存储）。 |
| `logical_page_size` | 631 | 属性：逻辑页大小，等于 `page_size × dcp_size`。 |
| `_storage_page_index` | 635 | 把 L3 key 对应的逻辑页起点换算成本 rank 的物理 slot（`index // dcp_size`）。起点不是逻辑页边界时抛 `ValueError`，越界时抛 `IndexError`。 |
| `alloc` | 650 | 加锁分配 `need_size` 个逻辑 slot，要求按逻辑页对齐；空闲不足返回 `None`；检测重复分配。 |
| `free` | 672 | 加锁释放 slot，检测重复释放，放入延迟释放列表，返回释放数量。 |

## 4. 与 PCP+DCP 方案 A 的关系

计划 §4.11（P4）只提到本文件第 242 行："保留 `page_size % dcp == 0`"。即这里不需要改，只需保证方案 A 下传入的 `page_size` 仍是 DCP 放大后的逻辑页。

读代码后的补充：

1. **`dcp_rank` 的来源。** 本文件只保存调用方传入的 `dcp_rank`，自己不计算。构造主机池的地方（如 `hybrid_pool_assembler.py:1466, 1824`、`kv/alloc/id_space.py:185`）读 `parallel.attn_dcp_rank`。只要计划 §4.2 让 `attn_dcp_rank = attn_cp_rank`，这里自动跟着变（推断）。
2. **逻辑 slot 与 DCP 页对齐。** `alloc` 要求 `need_size % logical_page_size == 0`，`_storage_page_index` 要求 L3 页起点落在逻辑页边界。方案 A 的 radix cache 页对齐按放大后的页（计划 §4.9 的 `page_size × dcp`），和这里一致。
3. **主机容量。** `__init__` 用 `device_capacity_tokens × ratio` 计算 `size`，这里的 `size` 是物理 token 数。方案 A 下 device 池容量按 DCP 切分计算（计划 §4.9 的 `kv_cache_configurator.py`），主机池会随之变化，不需要单独改（推断）。
4. 第一期 `hicache_hook.py` 会拒绝方案 A + HiCache（计划 §4.1），所以本文件在 P1-P3 不会被走到。
