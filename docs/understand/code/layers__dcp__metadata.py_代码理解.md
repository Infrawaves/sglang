# `python/sglang/srt/layers/dcp/metadata.py` 代码理解

- 文件路径：`python/sglang/srt/layers/dcp/metadata.py`
- 行数：37
- 所属模块：decode context parallel（DCP）每次前向的元数据结构定义。

## 1. 整体功能

这个文件只定义一个数据类 `DecodeContextParallelMetadata`，保存 DCP extend 阶段 KV gather 所需的 buffer 和下标。它由 `layers/dcp/planner.py` 的 `prepare_decode_context_parallel_metadata` 构造，挂在 `ForwardBatch.attn_dcp_metadata` 上，由 `forward_mla.py` 的 `bcg_dcp_extend_kv_gather`、`forward_mha.py` 等读取。

第 23-29 行的注释说明了设计意图：它**有意**不继承 `layers/cp/base.py` 的 `BaseContextParallelMetadata`，原样保留自 PR #14194，存在独立的 `attn_dcp_metadata` 字段里，与 prefill CP 的 `attn_cp_metadata` 分开，不参与 CP 的 `build_metadata` 约定。是否让它改继承 CP 基类，留到 P2 的 `DecodeContextParallelStrategy` 再决定。

## 2. 主要依赖与调用方

依赖：只有 `dataclasses`、`typing.Optional`、`torch`。

调用方（grep 确认）：
- 构造：`layers/dcp/planner.py:127`。
- 导出：`layers/dcp/__init__.py:50`。
- 类型标注：`layers/attention/flashinfer_mla_backend.py:34, 1054, 1102`。
- 读取字段：`models/deepseek_common/attention_forward_methods/forward_mla.py:122-132`（`dcp_extend_prefix_lens_sum`、`dcp_local_prefix_kv_indices`、`dcp_kv_buffer`）、`forward_mha.py:276`、`forward_mla_rocm.py:853`、`forward_mha_rocm.py:239`。

## 3. 类与函数说明

### `DecodeContextParallelMetadata`（第 31 行，`@dataclass`）

文件里没有函数和方法，只有字段：

| 字段 | 行号 | 含义 |
|---|---|---|
| `dcp_kv_indptr` | 33 | 每个请求在 `dcp_kv_buffer` 下标数组里的起止偏移（按 `seq_lens` 累加，长度 bs+1）。 |
| `dcp_kv_buffer` | 34 | 本次 extend 的完整 KV buffer，`seq_lens_sum` 行：前半段是所有请求的前缀 KV（从各 owner gather），后半段是所有请求的 extend token KV。 |
| `dcp_kv_indices` | 35 | 每个请求的 KV 在 `dcp_kv_buffer` 里的行下标（先前缀、后 extend）。 |
| `dcp_local_prefix_kv_indices` | 36 | 本 DCP rank 拥有的前缀 KV 在本地 KV 池里的物理下标，all-gather 前缀时用。 |
| `dcp_extend_prefix_lens_sum` | 37 | 所有请求前缀长度之和；为 0 时 `bcg_dcp_extend_kv_gather` 跳过通信。 |

## 4. 与 PCP+DCP 方案 A 的关系

计划 §4.5 要求在这个文件和 `layers/cp/base.py` 里新增一个合并后的元数据，同时描述"本卡的 query 行（来自 CP）"和"完整的 KV 行（来自 DCP）"，用 `msgspec.Struct` 定义，并和上游的 `DecodeContextParallelStrategy` 重构保持一致。计划 §7 也把"两套元数据合并"列为风险，引用的正是第 23-29 行的注释。

需要注意的点：
- 现有类是 `@dataclass`。按项目规则（`.claude/rules/no-dataclasses.md`）新代码用 `msgspec.Struct`，已有的 dataclass 不强制迁移。合并元数据若新写，用 `msgspec.Struct`；但 CP 侧的 `InterleaveContextParallelMetadata` 会被原地写字段（`pad_logical_token_to_physical`、`moe_num_token_non_padded`、BCG 的 `prepare`），所以不能用 `frozen=True`，除非一起改掉这些写入（推断）。
- 合并元数据至少需要覆盖：本卡 query 行在全局序列中的（请求、位置）；KV buffer 中前缀与 extend 两段的布局（见 `planner.py` 文档）；从 query 行到 buffer 行的 causal 边界；以及哪些 extend 行由本卡计算、需要写本地 KV。现有 `dcp_kv_buffer` 的 extend 段假设本卡算全部 extend token，方案 A 下要由 gather 填充（计划未提及，推断）。
- 第 26-29 行注释明确把"是否和 CP 基类合并"推迟到上游 P2 决定。如果 Infrawaves 分支先合并，后续和上游重构会冲突，这与计划 §7"和上游的冲突"一致。一个更小的做法是保持两个字段独立，只新增一个只读的"CP query 行 → DCP buffer 行"映射，挂在其中一个元数据上（推断，供评审参考）。
