# `python/sglang/srt/runtime_context.py` 代码理解

- 文件路径：`python/sglang/srt/runtime_context.py`
- 行数：2220
- 所属模块：进程级的运行时上下文：已解析配置的命名空间（config bag）、并行拓扑（宽度、rank、通信组）、可变运行时标志、进程资源句柄、每次前向的标志，以及平台事实（`get_platform()`）。

## 1. 整体功能

模块 docstring（第 14-24 行）概括了分工：
- `get_server_args()` 保留原始输入，用于诊断。
- `get_exec()`、`get_schedule()` 等命名空间访问器返回 publish 时从解析结果投影出来的配置（`_ConfigBag`），运行后可以用 `RuntimeContext.override` 修改。
- `get_parallel()` 返回 `ParallelContext`：配置的宽度、推导出的宽度、本进程的 rank 和通信组句柄。读取优先级是 scoped override > 永久 override（stamp）> 已发布配置。
- `get_flags()` 是可变的运行时状态（capture、MoE、DP、SP 标志），`get_resources()` 是进程资源（graph 内存池、stream、buffer 等），`get_forward()` 是每次前向的 scoped 标志。
- `get_platform()` 是机器事实（`is_cuda`、`is_sm100` 等），publish 之前就可以读，支持测试覆盖。

**数据流和调用时机：**
1. 参数解析阶段（`arg_groups/` 下的各个 hook），用 `derive_attn_tp_size`、`attn_dp_enabled_of`、`get_platform` 等纯函数，此时还没有 publish。
2. 每个进程入口调用 `publish(server_args, role=..., ranks=SpawnRanks(...))`。`_configure_context` 先 `resolve_once()`，再 `set_server_args` 投影 config bag（`_build_config_bags` + `_install_derived_leaves`，派生宽度如 `attn_tp_size`、`attn_dcp_size` 在这里通过 `fields/parallel.py` 中 `Derived(fn=...)` 指向本文件的 `*_of` 函数计算出来）。如果给了 `ranks`，用 `derive_spawn_ranks` 和 `_attention_ranks` 算出 `tp_rank`、`pp_rank`、`attn_cp_rank`、`attn_tp_rank`、`attn_dp_rank`、MoE rank、`dcp_rank`、`attn_dcp_rank`，写进 `ParallelContext` 的 stamp，并做拓扑一致性校验。
3. 分布式初始化（`distributed/parallel_state.py:2997`）构建通信组后，通过 `override_permanently` 把 `tp_group`、`attn_cp_group`、`dcp_group` 等写进 stamp。
4. 运行时代码通过 `get_parallel().xxx`、`get_exec().xxx` 读取。

**rank 布局约定**（`derive_attention_ranks` docstring）：TP 组内按 (dp, cp, tp) 排列，tp 变化最快：`tp_rank = (attn_dp_rank * attn_cp_size + attn_cp_rank) * attn_tp_size + attn_tp_rank`。DCP 组是 TP 组的连续切片（第 1609 行注释），`dcp_rank = tp_rank % dcp_size`。

## 2. 主要依赖与调用方

依赖：`msgspec`、`arg_groups/prefill_buffer_ceiling.prefill_buffer_ceiling_of`；延迟导入 `arg_groups/arg_utils`（`namespace_of`、`Derived`）、`arg_groups/fields/parallel.Parallel`、`arg_groups/overrides`（`resolution_result`、`declare_resolution` 等）、`arg_groups/model_override_base`（`resolving_view`、`resolved_view`、`ep_scale_joiner_of` 等）、`server_args.ServerArgs`、`environ.envs`、`utils/common`（平台探测）、`disaggregation/kv_events`。

调用方：`get_parallel`、`get_platform`、`get_exec` 等访问器被整个 `srt` 使用。几个与 CP/DCP 相关的调用点（grep 确认）：
- `arg_groups/fields/parallel.py:349-435`：`Derived(fn="sglang.srt.runtime_context.attn_tp_size_of")` 等，派生宽度的唯一来源。
- `distributed/parallel_state.py:2571, 2997`：构建通信组后写入 `get_parallel()`。
- `distributed/parallel_state.py:2738`：构建通信组时调用 `derive_parallel_widths`（`dcp_enabled=_DCP is not None`），注释说明同一个字典在组建好后被 stamp；`multimodal_gen/runtime/distributed/parallel_state.py:177, 777` 也调用它。
- `layers/dp_attention.py:457`：`derive_attention_ranks`。
- `arg_groups/parallel_hook.py`、`mega_moe_hook.py`：`derive_attn_tp_size`、`attn_dp_enabled_of`、`get_platform`。
- `linear_attn_parallel_group` / `linear_attn_tp_size` / `linear_attn_tp_rank`：`models/kimi_k3.py`、`models/qwen4_exp.py`、`models/glm5_next.py`、`disaggregation/common/conn.py`。
- `draft_attn_parallel_group`：`speculative/draft_parallel.py`；`draft_attn_tp_size`：`mem_cache/kv_cache_configurator.py`。
- `attn_dcp_size` / `attn_dcp_rank`：约 157 处，分布在各个 attention 后端、`layers/dcp/layout.py`、`dsa_dcp_sharded.py`、`dsa_indexer.py`、`kv_cache_configurator.py`、`managers/cache_controller.py`、`disaggregation/common/conn.py` 等。
- `snapshot_context` / `restore_context`：`entrypoints/engine.py`；`assert_published`：`managers/tokenizer_manager.py`、`disaggregation/encoder/server.py`。

## 3. 类与函数说明

### 3.1 并行宽度与 rank 推导（纯函数）

| 名称 | 行号 | 作用 |
|---|---|---|
| `_parallel_config_leaves` | 49 | 返回命名空间以 `parallel` 开头的配置字段名集合（`lru_cache`），publish 前也能用。 |
| `_parallel_fields` | 62 | 配置字段加上 `Parallel` 类里的 `Derived` 名字，是 `ParallelContext.override*` 接受的名字全集。 |
| `derive_attn_tp_size` | 73 | `tp_size // attn_dp_size // attn_cp_size`：TP 组扣掉 attention DP 和 attention CP 之后剩下的宽度。 |
| `derive_attention_ranks` | 79 | 由 `tp_rank` 算 `(attn_tp_rank, attn_dp_rank)`：`tp_rank % attn_tp_size`、`tp_rank // (attn_tp_size * attn_cp_size)`。 |
| `spawn_world_rank` | 95 | `ep_join_rank_offset + tp_size * pp_rank + tp_rank`，用 resolving view，publish 前可调用。 |
| `derive_spawn_ranks` | 106 | 由 WORLD rank 推出 `tp_rank`、`pp_rank`、`attn_cp_rank = (tp_rank // attn_tp_size) % attn_cp_size`、`moe_dp_rank`、`moe_ep_rank`、`moe_tp_rank`，不需要通信组。 |
| `derive_parallel_widths` | 137 | 返回派生宽度字典：`attn_dp_size`、`attn_tp_size`、`moe_ep_size`、`moe_tp_size = tp // ep // moe_dp`、`dcp_enabled`、`attn_dcp_size`（开启 DCP 时等于 `dcp_size`，否则 1）。 |
| `parallel_widths_of` | 160 | 用配置对象调用 `derive_parallel_widths`，`dcp_enabled = dcp_size > 1`。 |
| `launch_world_size_of` | 173 | 初始 WORLD 宽度：`ep_join_rank_offset + tp_size * pp_size`，弹性扩容后不变。 |
| `max_world_size_of` | 181 | WORLD 容量：`max_ep_size` 或启动宽度。 |
| `attn_tp_size_of` | 186 | `Derived` 用：取 `parallel_widths_of` 的 `attn_tp_size`。 |
| `attn_dp_enabled_of` | 191 | `attn_dp_size > 1`，或本进程是弹性 EP 扩容加入者。 |
| `num_dp_ranks_of` | 202 | `dp_size * attn_dp_size`（两者至少一个为 1）。 |
| `attn_dcp_size_of` | 210 | `Derived` 用：取 `attn_dcp_size`。 |
| `moe_ep_size_of` | 215 | `Derived` 用：取 `moe_ep_size`。 |
| `moe_tp_size_of` | 220 | `Derived` 用：取 `moe_tp_size`。 |
| `dcp_enabled_of` | 225 | `Derived` 用：取 `dcp_enabled`。 |
| `SpawnRanks` | 230 | `msgspec.Struct(frozen=True)`，启动器提供的进程位置：`world_rank`、`dp_rank`（None 表示没有 DP controller）、`gpu_id`。 |
| `_RANK_AND_WIDTH` | 244 | 校验用的 (rank, 宽度) 名字对：tp、pp、attn_tp、attn_dp、attn_cp、moe_ep。不含 DCP。 |
| `_WIDTH_AND_GROUP` | 255 | 校验用的 (宽度, 组) 名字对：tp、pp、attn_tp、attn_cp、moe_ep。不含 DCP 和 MoE-DP（注释：MoE-DP 可能复用更宽的 attention CP 组）。 |
| `_UNREADABLE` | 263 | 校验时表示"读不到或不是整数"的哨兵。 |
| `_validate_parallel` | 266 | 拓扑一致性校验：每个 rank 在 [0, 宽度) 内；`tp == attn_tp * attn_dp * attn_cp`；`tp == moe_ep * moe_dp * moe_tp`；`tp_rank` 符合 (dp, cp, tp) 布局；已构建的组的 `world_size` 等于配置宽度。读不到的值跳过，有问题时汇总成一个 `ValueError`。内部闭包 `read` 读取整数值。 |

### 3.2 `ParallelContext`（第 353 行）及相关函数

`__slots__ = ("_overrides", "_stamp", "_config")`：scoped override、永久 stamp、已发布的 `parallel` 配置 bag。

| 名称 | 行号 | 作用 |
|---|---|---|
| `ParallelContext.__init__` | 364 | 初始化三层存储。 |
| `ParallelContext.__getattr__` | 369 | 下划线开头的名字抛 `AttributeError`（防止 pickle/copy 时递归），其他走 `_read`。 |
| `ParallelContext._read` | 377 | 依次查 scoped override、stamp、配置 bag。都没有时：配置字段未发布报 `ValueError`；只有文档的运行时字段（rank、组）报"本进程未写入"；带 `fn` 的派生宽度报"不可用"；未知名字 `AttributeError`。 |
| `ParallelContext.override_permanently` | 408 | 校验名字后写入 stamp，跑 `_validate_parallel`，失败则回滚。publish 写 rank、分布式初始化写组都用它。 |
| `ParallelContext.clear_stamp` | 425 | 清空 stamp（包括 rank）。 |
| `ParallelContext.override` | 430 | context manager，临时覆盖并行值，校验失败回滚，退出时恢复，支持嵌套。 |
| `_derived_widths` | 449 | 返回 `Parallel` 类里所有 `Derived` 声明（含 rank 和组）。 |
| `_install_parallel_properties` | 459 | 为每个派生名字在 `ParallelContext` 上装一个 property（带 docstring），读取仍走 `_read`；模块加载时调用一次（第 474 行）。 |
| `linear_attn_parallel_group` | 477 | 线性 attention 切头用的组：开启 CP-TP group sharing 时为 `"tp"`，否则 `"attn_tp"`。 |
| `linear_attn_tp_size` | 484 | 上面那个组的宽度。 |
| `linear_attn_tp_rank` | 489 | 本 rank 在上面那个组里的下标。 |
| `draft_attn_parallel_group` | 494 | 有 attention 的 draft 模型用的组：`attn_cp_size > 1` 且无 attention DP 时为 `"tp"`（draft 在完整 TP 上跑，不用 CP），否则 `"attn_tp"`。 |
| `draft_attn_tp_size` | 506 | 上面那个组的宽度。 |
| `is_scheduler_io_rank` | 511 | `pp_rank == 0`、`attn_tp_rank == 0`、`attn_cp_rank == 0` 的 rank 负责和 tokenizer/detokenizer 收发，每个 attention DP 组一个。 |

### 3.3 运行时标志与资源

| 名称 | 行号 | 作用 |
|---|---|---|
| `_FlagGroupBase` | 523 | `msgspec.Struct` 基类：写不存在的字段时报错（防拼写错误）；提供事务式 `override`。 |
| `_FlagGroupBase.__setattr__` | 532 | 只允许写 `__struct_fields__` 里的名字。 |
| `_FlagGroupBase.override` | 541 | context manager，先校验全部 key 再写，退出时恢复（测试注入用）。 |
| `CaptureFlags` | 561 | `enable_torch_compile`（publish 时设置，模型不能 compile 时 warmup 清掉）、`disable_dispose_tensor`（decode/spec graph 捕获期间让 `dispose_tensor` 失效）。 |
| `MoeFlags` | 574 | MoE 运行时标志（a2a 后端、runner 后端、DeepEP 模式、TBO/SBO、量化等），由 `initialize_moe_config` 设置，投机解码上下文会临时切换。 |
| `DpFlags` | 602 | attention DP 运行时标志：是否启用、gather 用 world 组、prefill graph 捕获相关的标记、draft scope 的 gather slot、DP gather buffer 的元数据。 |
| `SpFlags` | 631 | LayerNorm 序列并行是否启用。 |
| `Flags` | 639 | 汇总上面四组。 |
| `Resources` | 648 | 进程资源槽：graph 内存池及借用状态、EPLB recorder 和专家位置元数据、LPLB solver、命名 stream、命名 buffer、TBO 事件池、indexer/experts capturer、TCPStore、trace 级别。 |
| `LoRABatchLayout` | 682 | 枚举：`DP_LOCAL`（attention 在 DP 本地 batch 上）、`TP_GLOBAL`（MLP 在 DP gather 后的 batch 上）。 |
| `ForwardFlags` | 689 | 每次前向的 scoped 标志。`_DEFAULTS` 列出所有标志；`_GRAPH_VISIBLE` 里的标志（会在 torch.compile 图里读写）用普通 dict 存，其余用 `ContextVar` 存以隔离线程和嵌套作用域。 |
| `ForwardFlags.__init__` | 747 | 按是否 graph visible 分别建 `_plain` 和 `_vars`。 |
| `ForwardFlags.__getattr__` | 769 | 先查 `_plain`，再查 `ContextVar`，未知名字报错。 |
| `ForwardFlags.__setattr__` | 781 | 禁止属性赋值，只能用 `scoped` 或 `set`。 |
| `ForwardFlags.set` | 787 | 不带作用域的写入，给旧的 setter 用。 |
| `ForwardFlags.scoped` | 796 | context manager，先校验 key，设置后在退出时按相反顺序恢复，异常安全。 |

### 3.4 配置 bag

| 名称 | 行号 | 作用 |
|---|---|---|
| `_ConfigBag` | 820 | 只读的配置命名空间，支持嵌套子 bag。叶子和子 bag 是真正的实例属性，便于 Dynamo 追踪；`_fields`/`_subs` 记录名字。 |
| `_ConfigBag.__init__` | 829 | 初始化 `_path`、`_fields`、`_subs`。 |
| `_ConfigBag.__getattr__` | 834 | 兜底查找（正常读取直接命中 `__dict__`），找不到报错并带路径。 |
| `_ConfigBag.__setattr__` | 847 | 禁止直接写，提示用 `get_context().override` 或 scoped `override`。 |
| `_ConfigBag._set` | 853 | 内部写入：同时更新 `_fields` 和实例属性。 |
| `_ConfigBag._set_sub` | 858 | 注册子 bag。 |
| `_ConfigBag.__contains__` | 863 | 名字是否是本 bag 的叶子。 |
| `_ConfigBag.override` | 867 | context manager，临时覆盖叶子，退出恢复。 |
| `_build_config_bags` | 887 | 按 `namespace_of(ServerArgs)` 把每个字段的解析结果放到对应路径的 bag 里；字段缺失报错；同名叶子和子组冲突报错；最后调 `_install_derived_leaves`。 |
| `_install_derived_leaves` | 928 | 对每个命名空间类里的 `Derived` 声明：没有 `fn` 但有默认值的写默认值；有 `fn` 的导入并用 `resolved_view` 计算后写入。派生宽度（`attn_tp_size`、`attn_dcp_size` 等）在这里算出。 |
| `_resolved_or_field` | 967 | 先取解析结果，没有就取字段值，`server_args` 为 None 时返回默认值。 |

### 3.5 `RuntimeContext`（第 983 行）与 `_ServerArgsOverride`（第 1204 行）

| 名称 | 行号 | 作用 |
|---|---|---|
| `RuntimeContext.__init__` | 999 | 保存 `parallel`，初始化 server_args、config bag、override 日志、publish role，新建 `Flags`、`Resources`、`ForwardFlags`。 |
| `RuntimeContext.get_stream` | 1009 | 取或创建命名 stream（按配置的 device），要在 CUDA graph 捕获之前调用。 |
| `RuntimeContext.set_stream` | 1026 | 安装或替换命名 stream。 |
| `RuntimeContext.get_buffer` | 1032 | 取或用 `factory()` 创建命名的持久 buffer。 |
| `RuntimeContext.server_args`（property） | 1041 | 返回原始 `ServerArgs`，未设置时报 `"Global server args is not set yet!"`。 |
| `RuntimeContext.set_server_args` | 1049 | 设置 `enable_torch_compile` 标志；替换 server_args 并重建 config bag（保留启动器给的 `gpu_id`）；给 `spec` bag 写 `max_speculative_num_draft_tokens`；把 `parallel` bag 接到 `ParallelContext._config`；清空 override 日志和 role。 |
| `RuntimeContext.config_bag` | 1083 | 返回顶层 bag，未发布报错；role 命名空间模式非 off 时检查。 |
| `RuntimeContext.is_config_namespace_published` | 1092 | 命名空间是否存在。 |
| `RuntimeContext._check_role_namespace` | 1097 | record 模式记录 (role, 命名空间) 读取；enforce 模式下 role 未登记或读取不在允许集合里则报错。单独成函数，让默认 off 模式下 Dynamo 可以裁掉分支。 |
| `RuntimeContext.override` | 1121 | 永久修改已解析的配置叶子（不改原始输入）：先校验所有名字和目标 bag，再写入，并把 `(source, fields)` 记入日志。 |
| `RuntimeContext.config_leaf` | 1160 | 按扁平字段名读取解析后的叶子。 |
| `RuntimeContext.overrides_log` | 1178 | 返回 override 日志的浅拷贝。 |
| `RuntimeContext.resolved_server_args_dict` | 1182 | 在 `resolved_dict()` 上叠加运行时 override，用于 `/server_info` 的 `internal_states`。 |
| `RuntimeContext.override_server_args` | 1195 | 返回 `_ServerArgsOverride`，给测试临时发布一个 dummy 配置。 |
| `_ServerArgsOverride.__init__` | 1226 | 保存上下文和字段。普通类而不是生成器，避免 GC 在不确定时刻恢复。 |
| `_ServerArgsOverride.install` | 1231 | 保存当前上下文状态；构造 `ServerArgs(model_path="dummy")` 并 `resolve_once`；校验字段名；真实字段用 `declare_resolution` 声明，其他（下划线私有缓存）直接写；`set_server_args`。 |
| `_ServerArgsOverride.restore` | 1271 | 恢复安装前的全部状态。 |
| `_ServerArgsOverride.__enter__` / `__exit__` | 1288 / 1291 | context manager 接口。 |

### 3.6 全局单例与访问器

| 名称 | 行号 | 作用 |
|---|---|---|
| `_PARALLEL`、`_CONTEXT` | 1295-1296 | 进程级的 `ParallelContext` 和 `RuntimeContext`。 |
| `get_context` | 1299 | 返回 `_CONTEXT`。 |
| `get_parallel` | 1303 | 返回 `_PARALLEL`。 |
| `get_server_args` | 1307 | 返回原始 `ServerArgs`。 |
| `get_flags` / `get_resources` / `get_forward` | 1311 / 1315 / 1319 | 返回对应对象。 |
| `get_device`、`get_model`、`get_exec`、`get_schedule`、`get_memory`、`get_spec`、`get_lora`、`get_mm`、`get_disagg`、`get_serving`、`get_observability` | 1328-1369 | 返回对应顶层 config bag，未发布时 `ValueError`。`parallel` 没有 bag getter，由 `get_parallel()` 直接回答。 |

### 3.7 role 命名空间审计

| 名称 | 行号 | 作用 |
|---|---|---|
| `ROLE_NAMESPACE_SETS` | 1389 | 每个 publish role 允许读取的顶层命名空间，None 表示全部。目前只有 `dp_controller` 有限制。 |
| `_validated_role_ns_mode` | 1409 | 校验 `SGLANG_ROLE_NAMESPACES` 只能是 off/record/enforce，否则报错（避免拼写错误静默关闭检查）。 |
| `_role_ns_mode_from_env` | 1420 | import 时读一次环境变量。 |
| `_ROLE_NS_MODE`、`_RECORDED_NS_READS`、`_RECORD_DUMP_REGISTERED` | 1429-1431 | 模式常量、已记录的读取集合、是否已注册 atexit。 |
| `_is_compiling` | 1434 | `torch.compiler.is_compiling()`，record 模式下在编译追踪中不记录。 |
| `_ensure_record_dump_registered` | 1445 | 第一次记录时注册 atexit 输出。 |
| `_append_role_ns_out` | 1454 | 设置了 `SGLANG_ROLE_NAMESPACES_OUT` 时立即追加写文件（worker 被信号杀掉时 atexit 不会运行）。 |
| `_record_namespace_read` | 1475 | 去重记录一次读取。 |
| `_dump_recorded_namespace_reads` | 1483 | 退出时每个 role 打印一行读取过的命名空间。 |

### 3.8 publish 与上下文生命周期

| 名称 | 行号 | 作用 |
|---|---|---|
| `publish` | 1507 | 为本 OS 进程安装配置，调用 `_configure_context(_CONTEXT, ...)`。允许重复 publish，后者覆盖前者。draft worker 不 publish。没有 `ranks` 时读 rank 需要显式 override，DCP 关闭时 `attn_dcp_rank=0` 例外。 |
| `create_context` | 1536 | 新建一个独立的 `RuntimeContext`（模型自有），不替换进程的活动配置。 |
| `use_context` | 1545 | context manager，临时把 `_CONTEXT`/`_PARALLEL` 换成给定上下文，不隔离并发线程。 |
| `_configure_context` | 1561 | publish 的实现：enforce 模式检查 role；`resolve_once`；`set_server_args`，若丢弃了之前的 override 则告警；记录 role；DCP 关闭时 stamp `attn_dcp_rank=0`；写 `gpu_id`。有 `ranks` 时：`derive_spawn_ranks` 算 rank；`moe_dp_size < attn_cp_size` 时让 `moe_dp_rank = attn_cp_rank`；写 `dp_rank`、`launch_world_rank`；`_attention_ranks` 算 attention rank；**DCP 开启时 `dcp_rank = tp_rank % dcp_size`（1609-1611）**，`attn_dcp_rank = dcp_rank`（1612）；`override_permanently` 后再 `_validate_parallel`。record 模式写 publish 标记。 |
| `_attention_ranks` | 1629 | 用 `derive_attention_ranks` 算 `attn_tp_rank`、`attn_dp_rank`。 |
| `assert_published` | 1639 | 要求当前发布的是同一个 `server_args` 且 role 一致，否则报错并说明原因（不在构造函数里重新 publish，以免丢掉运行时 override）。 |
| `publish_role` | 1666 | 返回上次 publish 的 role。 |
| `get_stream` / `set_stream` / `get_buffer` | 1671 / 1675 / 1679 | 转发到 `_CONTEXT` 的同名方法。 |
| `_GLOBAL_DWDP_MANAGER`、`get_global_dwdp_manager`、`set_global_dwdp_manager` | 1683-1690 | DWDP 管理器的全局槽。 |
| `_group_leaves` | 1695 | 递归取标志组的叶子值（dict/list 做浅拷贝）。 |
| `_restore_leaves` | 1709 | 把叶子值原地写回标志组（dict/list 原地替换内容）。 |
| `snapshot_context` | 1723 | 按 `RuntimeContext.__slots__` 和 `ParallelContext.__slots__` 枚举保存 publish 会替换的全部状态（标志组按叶子保存），加上 DWDP 管理器；启动失败时用来恢复。 |
| `restore_context` | 1752 | 恢复 `snapshot_context` 保存的状态。 |
| `reset_context` | 1769 | 清空发布的配置、parallel stamp、标志、资源、前向标志、DWDP 管理器，用于测试 teardown 和生命周期重置。 |

### 3.9 基于已发布配置的派生访问器

| 名称 | 行号 | 作用 |
|---|---|---|
| `remote_instance_transfer_engine_enabled` | 1786 | 按当前 model 配置（可传 `load_format` 覆盖）判断是否用 transfer engine 加载权重。 |
| `max_prefill_buffer_tokens` | 1796 | prefill buffer 的 token 上限：chunked prefill 大小，PP 动态 chunking 时取 `max_prefill_tokens` 和 1.25 倍 chunk 的较大值，再经 `prefill_buffer_ceiling_of`。 |
| `pre_capture_activation_reserve_mb` | 1815 | CUDA graph 捕获前预留的激活显存（MB）：PD decode 按运行请求数乘 draft token 数，否则按 chunked prefill 或 `max_prefill_tokens`（至少 2048），公式 `512 + tokens * 1.5 + tp * pp / 8 * 1024`，显存大于 60GB 时至少 10GB。 |
| `_PLATFORM_PROBES` / `_PLATFORM_VALUES` | 1842 / 1862 | 平台事实名到 `utils.common` 探测函数名的映射（布尔事实和数值事实）。 |
| `PlatformContext` | 1868 | 机器事实，每个名字对应一个 `lru_cache` 的探测函数，支持覆盖。 |
| `PlatformContext.__init__` | 1878 | 初始化 `_overrides`。 |
| `PlatformContext.__getattr__` | 1881 | 未知名字报错并列出已知名字；有覆盖值返回覆盖值，否则调用探测函数。 |
| `PlatformContext.__setattr__` | 1895 | 禁止赋值，提示用 `override_platform`。 |
| `PlatformContext._install` | 1902 | 写入覆盖值，返回之前的值和新增的键，供恢复用。 |
| `PlatformContext._restore` | 1912 | 恢复覆盖前的状态。 |
| `_PLATFORM`、`get_platform` | 1919 / 1922 | 全局 `PlatformContext` 及其访问器，publish 前可用。 |
| `_PlatformOverride` | 1927 | scoped 平台覆盖。 |
| `_PlatformOverride.__init__` / `install` / `restore` / `__enter__` / `__exit__` | 1932 / 1936 / 1940 / 1945 / 1948 | 保存事实、安装、恢复、context manager 接口。 |
| `_PlatformOverride.__call__` | 1951 | 作为装饰器使用，每次调用新建一个作用域。 |
| `override_platform` | 1969 | 返回 `_PlatformOverride`。 |
| `mamba_cache_chunk_size` | 1977 | mamba 状态的缓存粒度：`max(模型 mamba chunk, page_size)`。 |
| `mamba_checkpoint_grid` | 1985 | `lcm(mamba_cache_chunk_size(), tree_page)`；调用方需传入 radix 树实际分配的页（DCP 会把页放大）。 |
| `mamba_track_grid` | 1993 | 在上面的基础上再和 `mamba_track_interval` 取最小公倍数。 |
| `max_speculative_num_draft_tokens` | 2001 | 当前 draft token 数和 publish 时记录的容量中的较大值（自适应算法可能切换到更长状态）。 |
| `uses_mla_backend` | 2022 | 本进程模型是否走 MLA attention。 |
| `attention_backends` | 2029 | 从 `exec.kernel` bag 取 `(prefill, decode)` 后端对，跟随 publish 后的 override。 |
| `process_model_config` | 2044 | 本进程的 `ModelConfig`。 |
| `reports_expert_balancedness` | 2051 | 专家均衡报告模式不是 off。 |
| `logs_expert_balancedness_to_server_log` | 2059 | 报告模式为 `server_log` 或 `both`。 |
| `exports_expert_balancedness_to_prometheus` | 2064 | 报告模式为 `prometheus` 或 `both`。 |
| `cutedsl_moe_max_num_tokens` | 2069 | 一次前向在单个 DP rank 上经过 CuteDSL MoE 的最大 token 数：`max_prefill_tokens`、TC_PIECEWISE prefill graph 的 `max_bs`、decode `max_bs × 每请求 token 数` 三者最大值；用于 allgather wrapper、MegaMoE、AR fusion buffer 的大小。 |
| `describe_kv_events_publisher` | 2091 | 生成 `/server_info` 里 `kv_events` 的描述（publisher、端点 host/端口基数、topic、block size、DP rank 数、可选的 load 和 replay 端口）；配置缺失、格式错误、publisher 为 null、page size 非法或端点不可路由时返回 None。注释说明 `dp_size` 不随 DCP 放大，因为 DCP 在 rank 内切分。 |

## 4. 与 PCP+DCP 方案 A 的关系

计划 §4.2 列了两处修改：`runtime_context.py:1610-1611` 方案 A 下 `dcp_rank = attn_cp_rank`；`derive_parallel_widths`（137-157）加入"DCP 复用 CP 组"的标志，保证 `attn_dcp_size` 和 `attn_cp_size` 一致。

**需要注意的点：**
- **在方案 A 的约束下，现有公式的结果已经相同**：CP-TP group sharing 要求 `attn_cp_size == tp_size`，方案 A 再要求 `dcp_size == attn_cp_size`，所以 `attn_tp_size = 1`、`attn_dp_size = 1`。这时 `derive_spawn_ranks` 给出 `attn_cp_rank = (tp_rank // 1) % tp_size = tp_rank`，第 1611 行给出 `dcp_rank = tp_rank % tp_size = tp_rank`，两者相等；`attn_dcp_size = dcp_size = attn_cp_size` 也已经一致。`parallel_state.py:2705-2728` 按 `dcp_size` 切 TP 组得到的 DCP 组就是整个 TP 组，和 `_ATTN_CP` 的成员相同（`_ATTN_CP` 在 cp == tp 时复用 TP 组）。`derive_parallel_widths` 有两个调用入口：publish 时经 `parallel_widths_of`（`dcp_enabled = dcp_size > 1`），建组时经 `parallel_state.py:2738`（`dcp_enabled = _DCP is not None`）。如果加"DCP 复用 CP 组"的标志，两个入口都要传。所以 §4.2 这三处在第一期可以只加断言（例如在方案 A 下检查 `dcp_rank == attn_cp_rank`），不必改公式；只有以后支持 `attn_cp_size < tp_size` 的组合时公式才会不同（推断）。（计划未提及）
- **`_validate_parallel` 不校验 DCP**：`_RANK_AND_WIDTH` 和 `_WIDTH_AND_GROUP` 都不含 `dcp_rank`/`dcp_size`/`dcp_group`。计划 §6 的单元测试"DCP 组等于 CP 组，`dcp_rank == attn_cp_rank`"如果想在运行时也保证，可以在这里或 `_configure_context` 里加一条方案 A 专用的不变量检查（推断）。（计划未提及）
- **`attn_dcp_size` 的语义**：`fields/parallel.py:367-370` 的文档写的是"attention TP 组内的 DCP 宽度"，`attn_dcp_rank` 的文档是"attention TP 组内的 DCP rank"。方案 A 下 attention TP 为 1，而 `attn_dcp_size` 为 N，和文档的含义不符。部分 MHA 路径（`kv_cache_configurator.py`、`triton_backend.py`、`flashinfer_backend.py`）调用 `get_num_kv_heads(attn_tp_size, attn_dcp_size)`，方案 A 不支持 MHA，但 MLA 路径里如果有"头数 / attn_tp_size，再按 attn_dcp_size 处理"的假设，需要逐个检查（推断）。如果计划要加"DCP 复用 CP 组"的标志，建议同时更新这两个字段的文档。（计划未提及）
- **`linear_attn_parallel_group`（477）**：开启 CP-TP group sharing 时 KDA 在整个 TP 组上切头，方案 A 下不变，和计划 §3.2 的"KDA 头切分沿用现有行为"一致。
- **`draft_attn_parallel_group`（494）**：CP 下 draft 用完整 TP。DFLASH 的 draft 在 DCP 下要求 `cutedsl_mla` 并按 DCP owner 行读取，方案 A 下两条规则需要合并考虑（计划 §4.11 已提到，P4）。
- **`is_scheduler_io_rank`（511）**：方案 A 下只有 `tp_rank == 0` 满足条件，与现有 PCP 相同，不需要改。
- **`describe_kv_events_publisher`（2091）**：`dp_size` 不随 DCP 放大，方案 A 下 KV 事件的 block size 和页对齐可能要按放大后的页（`page_size × dcp`，计划 §4.9 的 radix cache 一项）重新确认（推断）。
- **`max_prefill_buffer_tokens`、`cutedsl_moe_max_num_tokens`**：按整批 token 数估算 buffer 大小，不考虑 CP 切分。方案 A 下 MoE 的每卡输入行数如果改变（见 `mega_moe_hook` 文档），需要确认这两个上限仍然覆盖（推断）。
