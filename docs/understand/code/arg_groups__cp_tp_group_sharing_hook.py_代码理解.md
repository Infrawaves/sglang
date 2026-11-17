# `python/sglang/srt/arg_groups/cp_tp_group_sharing_hook.py` 代码理解

- 文件路径：`python/sglang/srt/arg_groups/cp_tp_group_sharing_hook.py`
- 行数：141
- 所属模块：参数解析阶段的 CP-TP group sharing 判定与校验：对混合线性 attention 模型，在开启 prefill CP 时决定是否让 CP 组直接等于 TP 组，并声明 `enable_cp_tp_group_sharing=True`。

## 1. 整体功能

CP-TP group sharing 是一种 prefill CP 形态：CP 组就是 TP 组（`attn_cp_size == tp_size`，attention TP 为 1）。注释（第 13-17 行）说明：残差流、attention、indexer 按 CP 切分；MoE 和线性 attention（如 KDA）保留 TP 切分，线性 attention 在递推需要时 gather 整个序列。只有登记在 `_SUPPORTED_MODELS` 里的模型走这条路径，其他模型走普通 prefill CP。

这个文件只有一个函数 `resolve_cp_tp_group_sharing`。它不是独立的 pipeline hook，而是在 `arg_groups/parallel_hook.py:152` 的 `handle_context_parallelism` 中被调用（pipeline 第 291 行），位置在 `handle_model_specific_adjustments`（238）、`handle_hicache`（285）和 `handle_data_parallelism`（287）之后，在投机解码 hook（320）之前。注释（第 53 行）特别说明它早于投机解码默认值和 NEXTN 别名解析。

判定通过后，它声明 `enable_cp_tp_group_sharing=True`。运行时有多处读这个标志：`runtime_context.py:481`（选择 `"tp"` 还是 `"attn_tp"` 作为某些通信组）、`layers/cp/utils.py:294`、`model_executor/runner/eager_runner.py:340`、`models/kimi_k3.py:3152`、`layers/layer_boundary/construction.py:331`、`layers/layer_boundary/factories.py:593`、`disaggregation/common/conn.py:443, 587, 1940`、`speculative/spec_utils.py:1027`（TP-only draft 关闭它）。

## 2. 主要依赖与调用方

依赖：
- `arg_groups/model_override_base`：`resolving_view`、`resolved_view`。
- `arg_groups/overrides.declare_resolution`。
- `configs/hybrid_arch.mambaish_config`：取线性 attention 配置，用于检查头数能否被 TP 整除。
- `environ.envs.SGLANG_DISAGG_STAGING_BUFFER`。
- `model_executor/cuda_graph_config`：`Backend`、`Phase`、`with_phase`（Qwen4-Exp 关闭 prefill graph）。

调用方：`arg_groups/parallel_hook.py:152`（唯一调用点，grep 确认）。

## 3. 类与函数说明

### 3.1 模块级常量

| 名称 | 行号 | 作用 |
|---|---|---|
| `_SUPPORTED_MODELS` | 18 | 支持 CP-TP group sharing 的架构：`Glm5NextForConditionalGeneration`、`Qwen4ExpForConditionalGeneration`、`KimiK31ForConditionalGeneration`。 |
| `_INTERLEAVE_MODELS` | 24 | DSA 层用 interleave CP 的架构（GLM-5 Next、Kimi-K3.1）；其他支持模型（Qwen4-Exp）要求 zigzag。 |

### 3.2 函数

| 名称 | 行号 | 作用 |
|---|---|---|
| `resolve_cp_tp_group_sharing` | 30 | 未开启 prefill CP、`attn_cp_size <= 1` 或架构不在 `_SUPPORTED_MODELS` 时直接返回。否则依次检查：(1) `attn_cp_size == tp_size`（39-43），因为 attention DP 或 attention TP 会让 CP 组比 TP 组窄；(2) `cp_strategy` 必须等于该架构要求的策略（interleave 或 zigzag，44-50）；(3) 投机解码：K3.1 只允许 DFLASH，GLM-5 Next 允许 DFLASH 或 top-k 1 的原生 MTP（EAGLE/NEXTN 且 draft 模型就是目标模型），其他架构不允许（57-75）；(4) unsupported 列表（77-112）：通用项是 `enable_dsa_cache_layer_split`、**`dcp_size > 1`**、`enable_mixed_chunk`、`moe_dp_size > 1`；Qwen4-Exp 额外禁止 EP、a2a 后端、waterfill、PP、PD decode；GLM-5 Next 在 PD 模式下额外禁止 decode 模式、非 mooncake 传输、非 DFLASH 投机、PP、DP、EP；K3.1 额外禁止 PD decode 模式（注释：decode 节点跑 TP + DCP）；(5) PD prefill 下不支持 `SGLANG_DISAGG_STAGING_BUFFER`（113-118），因为 decode 侧把这个 prefill 看成 TP cp_size，而发送端按 attention TP 1 分段；(6) 线性 attention 的头数（`linear_num_key_heads`/`linear_num_value_heads` 或 `linear_attn_config["num_heads"]`）必须能被 `tp_size` 整除（120-133）。全部通过后声明 `enable_cp_tp_group_sharing=True`；Qwen4-Exp 同时关闭 prefill CUDA graph（序列集合通信的大小还依赖 batch）。 |

## 4. 与 PCP+DCP 方案 A 的关系

计划 §4.1 要求：从 unsupported 列表去掉 `"dcp-size > 1"`（第 79 行），并新增 `dcp_size != attn_cp_size` 时报错。

**需要注意的点：**
- **去掉 `dcp-size > 1` 只应对 K3.1 生效**：第 79 行是所有 `_SUPPORTED_MODELS` 共用的项。GLM-5 Next 和 Qwen4-Exp 没有方案 A 的实现，直接删除会让它们也放行 DCP。建议把这一项移到按架构的分支里，只对 K3.1 改为"`dcp_size > 1` 时必须等于 `attn_cp_size`"（推断）。（计划未明确说明作用范围）
- **`attn_cp_size == tp_size` 已经由第 39 行保证**，所以 "`dcp_size == attn_cp_size`" 等价于 "`dcp_size == tp_size`"。K3.1 的 DCP override（`model_overrides/kimi_k3.py:219`）要求 `dcp_size` 是 2 的幂且整除 `tp_size`，两者叠加后实际上要求 `tp_size` 也是 2 的幂（推断）。（计划未提及）
- **`cp_strategy` 检查（49-50）** 对 K3.1 已经强制 interleave。计划 §3.3 "zigzag 直接报错" 在 K3.1 上已由这里覆盖，`parallel_hook.py` 里的 zigzag 报错主要是给提示信息更清楚。
- **投机解码（58-63）**：K3.1 CP-TP group sharing 只允许 DFLASH，而 K3.1 DCP 允许 DSPARK 和 DFLASH 并要求 DFLASH draft 用 `cutedsl_mla`。方案 A 下两者交集只剩 DFLASH（或不开投机），计划把 DFLASH 放到 P4，第一期应在这里或 K3.1 override 里显式拒绝方案 A + 投机解码（推断）。（计划 §4.11 只说需单独设计，未说第一期如何拒绝）
- **K3.1 拒绝 `disaggregation-mode decode`（109）**：方案 A 只在统一部署或 PD prefill 实例上生效，decode 节点仍是普通 TP + DCP。这与计划 §4.10 一致，P4 前不需要改。
- **PD prefill 下的 staging buffer 检查（115-118）**：方案 A 下 prefill 的 KV 按 DCP owner 切分，PD 传输布局会变（计划 §4.10）。这条检查是否仍然足够，要在 P4 时重新看（推断）。
- **线性 attention 头数检查（129-133）**：方案 A 不改 KDA 的 TP 头切分（计划 §3.2），这条保持不变。
- 这个函数运行在 `handle_model_specific_adjustments` 之后，而 K3.1 的 DCP override（拒绝 prefill CP）在 model override 收集阶段就会报错。所以方案 A 的组合必须先在 `kimi_k3.py` 放行，这里的修改才会被执行到。
