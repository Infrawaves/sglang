# `python/sglang/srt/layers/cp/base.py` 代码理解

- 文件路径：`python/sglang/srt/layers/cp/base.py`
- 行数：390
- 所属模块：PCP 策略的基础类型：策略枚举、attention 后端枚举、元数据基类、策略抽象基类，以及进程级策略单例的初始化与查询。

## 1. 整体功能

模块 docstring（第 15-23 行）说明 CP 策略拆成四个文件：`base.py`（基类、枚举、单例）、`zigzag.py`（zigzag 策略）、`interleave.py`（interleave 策略）、`utils.py`（导入门面）。

核心概念：
- **`ContextParallelStrategy`**：一个 CP 布局的进程级策略对象，定义"怎么切 hidden/positions、怎么 gather hidden/KV、怎么跑 attention、怎么物化完整 KV"这些接口。具体策略只有 `ZigzagCPStrategy` 和 `InterleaveCPStrategy`，另外可以通过 `register_cp_kv_layout` 注册额外的 KV 布局工厂。
- **策略单例 `_STRATEGY`**：三种状态，`None` 表示未初始化，`False` 表示已配置为关闭，否则是策略对象。参数解析阶段由 `arg_groups/parallel_hook.py:197-199` 调 `init_cp_strategy`；worker 子进程里 `_STRATEGY` 重新为 `None`，第一次 `get_cp_strategy()` 时从 `get_parallel()` 读配置惰性初始化。
- **`BaseContextParallelMetadata`**：每次前向的 CP 元数据基类，只有 `total_seq_lens` 和 `bs` 两个字段，放在 `ForwardBatch.attn_cp_metadata` 上。

## 2. 主要依赖与调用方

依赖：`arg_groups/choices.py`（`add_cp_kv_layout_choices`，注册 CLI 可选值）、`runtime_context.get_parallel`；延迟导入 `layers/cp/zigzag.py`、`layers/cp/interleave.py`。

调用方（grep 确认）：
- `init_cp_strategy`：`arg_groups/parallel_hook.py:197-199`，`layers/cp/__init__.py` 导出。
- `get_cp_strategy`：遍布 CP 相关代码（`layers/cp/utils.py`、`interleave.py`、`dsa_indexer.py`、`dsa_backend.py` 等）。
- `disable_cp_strategy`：`speculative/spec_utils.py:1029`，TP-only 的 draft 运行时暂时屏蔽目标模型的 CP 策略。
- `is_cp_sharded_padded`：`disaggregation/prefill.py:2072`、`layers/attention/flashattention_backend.py:837, 1200`、`managers/tp_worker.py:555`。
- `register_cp_kv_layout`：在 `python/sglang` 下 grep 没有找到调用方，目前是预留的扩展点。

## 3. 类与函数说明

### 3.1 枚举

| 名称 | 行号 | 作用 |
|---|---|---|
| `ContextParallelStrategyKind` | 40 | 策略标识：`NONE=0`、`ZIGZAG=1`、`INTERLEAVE=2`。 |
| `ContextParallelStrategyKind.from_string` | 48 | `"zigzag"` / `"interleave"` 转枚举，其他值 `ValueError`。 |
| `ContextParallelStrategyKind.cli_value`（property） | 58 | 枚举转回 CLI 字符串（`"none"`/`"zigzag"`/`"interleave"`），用于报错提示。 |
| `CPAttentionBackendKind` | 66 | CP 策略分发时使用的 attention 后端调用约定：`FLASH_ATTENTION`、`DSA`、`TRTLLM_MHA`、`DSV4`。 |
| `CPAttentionBackendKind.from_string` | 75 | `fa3`/`fa4`/`flashinfer` → `FLASH_ATTENTION`，`dsa`、`trtllm_mha`、`dsv4` 各自对应，其他值 `ValueError`。 |

### 3.2 `BaseContextParallelMetadata`（第 90 行，`@dataclass`）

字段 `total_seq_lens: int = 0`（本批真实 token 总数）、`bs: int = 1`（请求数）。`ZigzagContextParallelMetadata` 和 `InterleaveContextParallelMetadata` 继承它。

### 3.3 `ContextParallelStrategy`（第 96 行，ABC）

类属性：`name`、`kind`、`cp_kv_sharded = False`（是否使用 rank 本地的 KV 存储，`is_cp_sharded_padded` 读它）。

| 方法 | 行号 | 作用 |
|---|---|---|
| `__init__` | 103 | 保存 `cp_size`。 |
| `cp_rank`（property） | 107 | 返回 `get_parallel().attn_cp_rank`，每次实时读取。 |
| `per_layer_attn_cp_comm`（property） | 112 | 返回 `_is_dsa_active()`，目前恒为 False。 |
| `moe_num_token_non_padded` | 115 | 默认返回 `forward_batch.num_token_non_padded`；interleave 覆盖为本 rank 有效行数。 |
| `can_apply`（抽象） | 120 | 当前前向能否用这个策略切分。 |
| `build_metadata`（抽象） | 124 | 构建每次前向的元数据。 |
| `shard_hidden_states`（抽象） | 133 | 把 hidden states 切到本 rank，通常在第一层前。 |
| `shard_position_ids`（抽象） | 137 | 切分每个 token 的位置。 |
| `gather_hidden_states`（抽象） | 141 | gather 各 rank 的 hidden states，通常在最后一层后。 |
| `gather_kv_cache`（抽象） | 150 | gather 各 rank 的 KV，还原完整 token 顺序。 |
| `local_q_indices` | 158 | 默认 `NotImplementedError`；interleave 实现。 |
| `shard_per_request` | 163 | 默认 `NotImplementedError`；按请求切分，返回四元组。 |
| `shard_local_tokens` | 172 | 默认 `NotImplementedError`。 |
| `materialize_full_indexer_k_cache` | 177 | 默认 `NotImplementedError`；DSA indexer K 的 gather。 |
| `all_gather_dsa_trtllm_fp8_kv` | 184 | 默认 `NotImplementedError`；DSA TRT-LLM FP8 KV 的 gather。 |
| `split_before_forward` | 191 | 模板方法：切分 `input_ids` 写到 `forward_batch.cp_input_ids`，切分 positions 覆盖 `forward_batch.positions`，有 `input_embeds` 时返回切分后的 embedding。 |
| `run_attention`（抽象） | 209 | 按后端约定分发 CP attention。 |
| `materialize_full_kv`（抽象） | 220 | 需要时把完整布局的 K/V 写入后端 cache。 |
| `materialize_full_mla_kv`（抽象） | 231 | 物化完整布局的 MLA K/V。 |
| `reindex_attn_metadata` | 240 | 可选的 attention 元数据重写，默认什么都不做。 |

### 3.4 模块级函数

| 名称 | 行号 | 作用 |
|---|---|---|
| `_is_dsa_active` | 245 | 占位实现，恒返回 False（注释说明真正判断需要模型架构）。 |
| `_REGISTERED_CP_KV_LAYOUTS` | 250 | 注册表：KV 布局名 → `(工厂, 兼容的策略 kind)`。 |
| `register_cp_kv_layout` | 255 | 注册一个 KV 布局工厂，并把名字加入 CLI 的 `--cp-kv-layout` 可选值。 |
| `_STRATEGY` | 267 | 策略单例：None 未初始化，False 已关闭。 |
| `disable_cp_strategy` | 271 | context manager：临时把 `_STRATEGY` 置 None，退出时恢复（即使出错）。注释要求同时配合一个关闭 prefill CP 的运行时拓扑作用域，防止惰性初始化重新建出目标策略。 |
| `init_cp_strategy` | 286 | 绑定本进程的策略：未开启 prefill CP 或 `cp_size <= 1` 时缓存 False；给了 `cp_kv_layout` 时查注册表，并检查它要求的策略 kind 与 `--cp-strategy` 一致，否则 `ValueError`；否则按 kind 构造 `ZigzagCPStrategy` 或 `InterleaveCPStrategy`。 |
| `get_cp_strategy` | 335 | 返回策略对象（关闭时返回 None）。未初始化时从 `get_parallel()` 读 `enable_prefill_cp`、`attn_cp_size`、`cp_strategy`、`cp_kv_layout` 并调用 `init_cp_strategy`；读取失败（`AssertionError`/`AttributeError`/`RuntimeError`/`ValueError`）时返回 None，并保持未初始化。 |
| `is_cp_sharded_padded` | 368 | 策略存在且 `cp_kv_sharded` 为 True。 |
| `get_cp_strategy_kind` | 374 | 返回当前策略的 kind，没有策略时返回 `NONE`。 |
| `is_cp_enabled` | 381 | 是否有策略。 |
| `is_zigzag` | 385 | 当前 kind 是否 `ZIGZAG`。 |
| `is_interleave` | 389 | 当前 kind 是否 `INTERLEAVE`。 |

## 4. 与 PCP+DCP 方案 A 的关系

计划 §4.5 要求在 `layers/dcp/metadata.py` 和这个文件里"新增一个合并后的元数据，同时描述本卡的 query 行（来自 CP）和完整的 KV 行（来自 DCP）"，用 `msgspec.Struct` 定义。

**需要注意的点：**
- 现有的 `BaseContextParallelMetadata` 和它的子类都是 `@dataclass`，`InterleaveContextParallelMetadata` 还依赖可变字段（`pad_logical_token_to_physical` 和 `InterleaveCPStrategy.moe_num_token_non_padded` 会原地写字段，BCG 的 `prepare` 也写 `graph_num_tokens`）。新元数据若用 `msgspec.Struct`，不能设为 `frozen=True`，否则这些原地写入会失败；或者要同时改掉这些写入点（推断）。
- 如果新元数据继承 `BaseContextParallelMetadata`，`msgspec.Struct` 不能直接继承 dataclass，需要单独定义或一起迁移（推断）。
- 方案 A 新增"按位置取模"的切分，有两种接入方式：直接修改 `InterleaveCPStrategy`，或注册一个新策略。`register_cp_kv_layout`（255）已经提供了"KV 布局工厂 + 要求的策略 kind"的扩展点：可以注册一个 kind 为 `INTERLEAVE` 的布局，`init_cp_strategy` 会检查 `--cp-strategy interleave`，正好覆盖计划 §3.3 的"zigzag 直接报错"约束。目前这个注册表没有调用方，是否适合用需要评审决定。（计划未提及）
- `cp_kv_sharded`（101）这个类属性表示"rank 本地的 KV 存储"，`is_cp_sharded_padded` 依赖它，影响 `tp_worker.py:555`、`disaggregation/prefill.py:2072` 和 FlashAttention 后端。它对应的是页交错的 KV 分片（`--enable-kv-cache-sharding`），换算公式在 `layers/cp/utils.py`。方案 A 的 KV 由 DCP 的 owner 规则切分，新策略不应设置 `cp_kv_sharded = True`，否则会叠加两套换算（推断）。（计划未提及）
- `cp_rank`（107）读 `attn_cp_rank`。方案 A 把 `dcp_rank` 设为 `attn_cp_rank`，CP 侧不需要改动。
- `disable_cp_strategy`（271）在 TP-only draft 运行时屏蔽 CP。计划把 DFLASH/DSPARK 投机解码放到 P4；如果 draft 关掉 CP 但仍处在 DCP 下，需要单独确认（推断）。
