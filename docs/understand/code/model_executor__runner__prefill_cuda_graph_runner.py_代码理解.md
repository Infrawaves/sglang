# `python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py` 代码理解

- 文件路径：`python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py`
- 行数：2540
- 所属模块：模型执行层的 CUDA graph 运行器，负责 prefill（EXTEND / MIXED）阶段的图捕获与回放。

## 1. 整体功能

`PrefillCudaGraphRunner` 让 prefill 前向也能走 CUDA graph。具体的捕获/回放机制交给可插拔的 backend（由 `cuda_graph_config.prefill.backend` 决定，经 `resolve_prefill_backend` 选出），本文件负责其余部分：构造假的 `ForwardBatch`、维护静态输入 buffer、初始化或刷新 attention 元数据、裁剪输出。

支持三种 backend（见文件头 docstring，14-36 行）：
- `breakable`（BCG，CUDA 上默认）：分段捕获 transformer body（`layer_model.forward`），attention 元数据和 LM head / logits 尾部在图外按真实 batch 执行，因此支持多请求。
- `full`：每个 token 桶一张完整图，请求轴固定为 `full_prefill_max_req` 个槽位，不足的槽用零长度哨兵填充；还支持 chunked-prefix 变体（按前缀 chunk 数 1/2/4/8/16 捕获）。
- `tc_piecewise`：用 torch.compile 包住整个 `model.forward`，分段图由 torch.compile 的缓存持有。

调用时机：
1. 启动时由 `cuda_graph_setup.py:717` 构造，`__init__` 末尾直接调用 `capture()`，按 token 桶从大到小逐个捕获（`capture_one_shape`）。
2. 每次 prefill 前，`can_run_graph` 判断这个 batch 能否回放；DP attention 下，调度阶段用 `can_replay_locally` 做投票，保证各 DP rank 结论一致。
3. 回放时调用 `execute`：`load_batch` 把真实 batch pad 到桶大小并写入静态 buffer，刷新元数据，然后按 backend 分派到 `execute_prefill_cp_bcg`（CP BCG）、`_execute_body_capture`（BCG/Full）或 `_execute_tc_piecewise`，最后裁掉 padding 行。

与 CP/DCP 相关的两条特殊路径：
- **CP BCG**（`enable_cp_bcg_capture`）：BCG 且 `supports_prefill_cp_bcg` 为真时启用，由 `layers/cp/bcg/bcg.py` 的 `PrefillCPBCGInput` 负责 CP 本地行数（`bucket_local_tokens`）、位置、桶选择。
- **DCP extend**（`_dcp_extend_active`，738 行）：`attn_dcp_size > 1` 且模型实现了 `prepare_context_parallel_metadata_for_dcp` 时启用。捕获时为每个桶保留一个 DCP gather buffer，回放时复用；有前缀命中的 batch 一律走 eager。

## 2. 主要依赖与调用方

依赖：
- `BaseCudaGraphRunner`（`runner/base_cuda_graph_runner.py`）：基类，提供 `warmup`、`model_context`、`_pad_to_bucket`、`tbo_plugin` 等。
- Backend：`BreakableCudaGraphBackend`、`FullCudaGraphBackend`、XPU 的 `FullXPUGraphBackend`、`resolve_prefill_backend`（`runner_backend/`）。
- 缓冲：`PrefillInputBuffers`（`runner_utils/buffers.py`）、`build_prefill_registry` / `CudaGraphBufferRegistry`（`cuda_graph_buffer_registry.py`）。
- CP：`layers/cp/bcg`（`supports_prefill_cp_bcg`、`PrefillCPBCGInput`、`execute_prefill_cp_bcg`、`filter_prefill_cp_bcg_capture_num_tokens`）、`layers/cp/utils.is_cp_active`、`is_mla_cp_enabled`、`is_dsa_enable_prefill_cp`。
- DCP：模型方法 `prepare_context_parallel_metadata_for_dcp`；`create_chunked_prefix_cache_kv_indices` kernel。
- 上下文：`forward_context`、`set_tc_piecewise_forward_context`、`prefill_graph_scope`、`set_forward_batch`、`runtime_context` 的 `get_exec/get_parallel/...`。

调用方（grep 确认）：
- `model_executor/model_runner_components/cuda_graph_setup.py:717`：构造 runner。
- `managers/scheduler_components/dp_attn.py:482, 511`：DP 投票时调用 `can_replay_locally`。
- `layers/cp/bcg/bcg.py:42, 92, 181, 276`：`PrefillCPBCGInput.create(runner)`、`prepare(runner, ...)`、`execute_prefill_cp_bcg(runner, ...)` 反向读取 runner 状态。
- `arg_groups/cuda_graph_hook.py`：配置阶段决定是否启用 BCG（不直接引用本类）。

## 3. 类与函数说明

### 3.1 模块级函数与常量

| 名称 | 行号 | 作用 |
|---|---|---|
| `_full_graph_backend_types` | 114 | 返回 Full 类 backend 的类型元组（CUDA 和 XPU），延迟导入 XPU 模块。 |
| `_is_full_graph_backend` | 122 | 判断 backend 是否为 Full 图。 |
| `_is_full_or_bcg_backend` | 126 | 判断 backend 是否为 Breakable 或 Full，这两者只捕获 transformer body、需要静态 buffer。 |
| `_has_chunked_local_attn` | 132 | 检查 model_runner / model_config / attn_backend 是否有 `attention_chunk_size`；XPU Full 图不支持这种模型，`__init__` 据此报错。 |
| `get_prefill_num_tokens_to_capture` | 200 | 把配置的捕获 token 数对齐到 `get_cuda_graph_max_batch_size` 的 TP/CP 分片宽度并去重排序，避免 capture 和 replay 分片不等导致集合通信错配。 |
| `_chunked_prefix_variant` | 209 | 生成 chunked-prefix 变体标签 `chunked_prefix:{n}`，用作 `ShapeKey.variant_label`。 |
| `_ceil_div` | 213 | 向上取整除法。 |
| `_resolve_transformer_layer_model` | 217 | 穿过 language_model / `.model` 包装找到拥有 `layers` 的内层模块；找不到时抛 `RuntimeError`。 |
| `_build_layer_model_forward_kwargs` | 240 | 按 `layer_model.forward` 签名决定传 `input_embeds` 还是 `inputs_embeds`，以及是否传 `pp_proxy_tensors`。 |
| `_slice_output_rows` | 257 | 递归地把输出里每个张量（含嵌套 tuple/list、`PPProxyTensors`）切到前 `num_tokens` 行；不支持的类型抛 `TypeError`。 |
| `prefill_failure_msg` | 292 | 按 backend 拼接捕获失败提示（BCG / TC_PIECEWISE 各有专门提示）。 |
| `_MAX_PREFILL_CUDA_GRAPH_PADDING_FACTOR` | 194 | 常量 2：pad 后 token 数超过真实数 2 倍就不回放。 |
| `_CHUNKED_PREFIX_VARIANTS` | 197 | 常量 `(1, 2, 4, 8, 16)`：Full 图捕获的前缀 chunk 数。 |
| `_PREFILL_STATIC_FIELDS` | 315 | BCG/Full 静态输入字段名（`seq_lens`、`extend_seq_lens`、`extend_prefix_lens`、`extend_start_loc`、`req_pool_indices`、`orig_seq_lens`），每个是长度 `max_bs` 的 int64 张量。 |

### 3.2 `_ChunkedPrefixCaptureBuffers`（277 行，frozen dataclass，无方法）

Full 图 chunked-prefix 变体共享的张量：`starts`/`starts_cpu`（每个 chunk 的固定起点，分配时填好）、`seq_lens`/`seq_lens_cpu`、`cu_seq_lens`、`kv_indices`（形状 `(max_chunks, prefix_chunk_capacity)`）。

### 3.3 `PrefillCudaGraphRunner`（325 行）——初始化与基础属性

类属性默认值（334-341 行）：`_use_draft_input_embeds`、`_fa4_prefill`、`_backend_can_run_prefill_cuda_graph`、`_backend_vote_prefill_cuda_graph`、`_captured_attn_metadata_max_bs`、`_dcp_extend_active=False`、`dllm_attention`。

| 名称 | 行号 | 作用 |
|---|---|---|
| `_attn_backend_is_dsa`（property） | 344 | attn backend 是否为 `DeepseekSparseAttnBackend`；DSA 在 BCG 下不走 MHA companion，前缀也走稀疏 MLA。 |
| `__init__` | 351 | 读取 prefill 图配置和模型标志；mixed chunk 要求 BCG/Full（否则 assert）；确定 `capture_hidden_mode`；创建 `PrefillInputBuffers` 和 buffer registry（`enable_prefill_cp` 取决于 `is_dsa_enable_prefill_cp() or is_mla_cp_enabled()`）；解析 backend；Full 下设置请求槽、chunked-prefix 变体和 buffer；BCG/Full 下分配 `_prefill_static_buffers`；判定 `enable_cp_bcg_capture` 并按 CP 过滤捕获桶、创建 `PrefillCPBCGInput`；判定 `use_captured_attn_metadata`；解析 `layer_model`；设置 `_dcp_extend_active` 与 `_dcp_kv_buffers`（736-741）；最后 barrier 并 `capture()`。`--cuda-graph-prefill-max-context` 不被 backend 支持或超过 req_to_token 宽度时报 `ValueError`。 |
| `_is_mamba_track_enabled` | 760 | mamba extra buffer 开启且 radix cache 未禁用时，需要 mamba track 槽位。 |
| `_cache_loc_dtype` | 765 | `out_cache_loc` 的 dtype：NPU 为 int32，其余 int64。 |
| `_input_embeds_hidden_size` | 768 | `input_embeds` buffer 宽度，模型可用 `input_embeds_hidden_size` 覆盖 `hidden_size`。 |
| `_next_token_logits_buffer` | 787 | 最后一个 PP rank 上从 `graph_shared_output` 取 logits buffer；行数超出上限返回 None，走 eager 分配。 |
| `_uses_eager_prefill_tail` | 798 | BCG/Full 为真：LM head 和 logits 在图外执行。 |
| `_prefill_logits_buffer_rows` | 801 | 计算 logits buffer 行数；需要 logprob 时要求 eager 尾部，按 DP 全局计数或每请求 `seq_len - start_len` 求和。 |
| `_capture_num_token_non_padded` | 820 | 捕获时把 `num_token_non_padded` 槽填为 `num_tokens`，attn-TP 序列切分时换成本地值。 |
| `_get_layer_model_positions` | 836 | 返回直接调用 `layer_model` 时要用的位置：CP BCG 下优先用 CP 本地位置，其次 mrope 位置，否则普通 `positions`。 |
| `_capture_pp_proxy_tensors` | 855 | 非首个 PP rank 时，从静态 buffer 切出 PP 代理张量。 |
| `_pp_proxy_rows` | 862 | PP 代理行数；CP BCG 下用该桶的 CP 本地 token 数。 |
| `_zero_padded_tail` | 871 | 是否要清零 attention 输出的 padding 尾部：非 CUDA、有 MoE 层或有量化时为真，因为这些算子会读 padding 行。 |
| `_prefill_forward_context`（contextmanager） | 885 | 一次性进入 `set_forward_batch`、`prefill_graph_scope`、`forward_context`、`set_tc_piecewise_forward_context` 四层上下文。 |

### 3.4 前向执行（捕获期）

| 名称 | 行号 | 作用 |
|---|---|---|
| `_run_forward` | 919 | 在 prefill 上下文中跑一次前向：BCG/Full 只调 `layer_model.forward`（带 `@torch.no_grad`，PP 非首 rank 时 `input_ids` 置空）；tc_piecewise 调完整的 `model.forward`。会先设置 DP buffer 长度、`set_is_extend_in_batch(False)`。 |
| `_run_dummy_forward` | 977 | 供 `TcPiecewiseCudaGraphBackend` 的编译 pass 使用：构造假 batch、初始化元数据、跑一次 `_run_forward`。 |
| `run_dummy_multimodal_deepstack_forward` | 988 | 为带 `input_deepstack_embeds` 的模型（Qwen3-VL）预热 deepstack 分支，避免首个图像请求触发同步重编译；不适用时返回 False。 |

### 3.5 资格判断与上下文长度

| 名称 | 行号 | 作用 |
|---|---|---|
| `_has_inactive_dp_rank` | 1059 | DP 下有 rank 本轮 0 token 时返回真（让集合通信一致地走 eager）；MegaMoE 等可容忍 sum-len 且无 rank barrier 的图例外。 |
| `_max_addressable_prefix_len`（static） | 1078 | 可寻址前缀上限：`req_to_token` 宽度与模型 `context_len` 取小。 |
| `_validate_max_context_capacity`（static） | 1090 | `max_context_size` 超过 `req_to_token` 宽度时报 `ValueError`。 |
| `_ignore_max_context_size` | 1099 | 对尚不支持固定上下文的路径（tc_piecewise、CP BCG）打 warning 并把 `max_context_size` 置 None。 |
| `_batch_max_context_len`（static） | 1111 | 取 batch 最大 `seq_len`，优先读 CPU 张量，否则同步读 GPU。 |
| `can_replay_locally` | 1438 | 单 rank 的回放资格判断，`can_run_graph` 和 DP 调度投票共用。拒绝条件包括：CP BCG 不允许、Qwen BCG 多模态、Full+FA4 多模态、bs 超过 Full 槽位或捕获元数据上限、LoRA 不可用、mixed 偏好 eager、有 `input_embeds`/`replace_embeds`、非 CUDA BCG 的 MHA companion 前缀、Full 前缀超出变体、**DCP 下有前缀命中（1511 行）**、target verify、hidden mode 超出捕获模式、logprob 无 eager 尾部、上下文超限、token 数超最大桶、pad 超过 2 倍。 |
| `can_run_graph` | 1540 | forward 时判断：先看 DP 组投票结果和空闲 DP rank，CP BCG 要求 `is_cp_active`，再调 `can_replay_locally`，CP BCG 还要能选出回放桶，最后问 attn backend 的 `can_run_prefill_cuda_graph`。 |

### 3.6 Full 图的 chunked-prefix 支持

| 名称 | 行号 | 作用 |
|---|---|---|
| `_resolve_prefix_chunk_shape`（static） | 1125 | 计算每请求的 chunk 长度和总容量：优先 `full_prefill_prefix_chunk_tokens`，否则 `chunked_prefill_size`，再否则 `max_bs`；先除以槽位数再分配，避免槽位增加放大显存；非正值报 `ValueError`。 |
| `_select_prefix_capture_chunks` | 1147 | 选出能覆盖 `max_prefix_len` 的最小已捕获变体，没有则返回 None。 |
| `_has_uncapturable_chunked_prefix` | 1152 | 前缀超出所有变体时为真（只对 Full 生效）。 |
| `_shape_key` | 1163 | 由桶大小和（Full 下的）前缀变体构造 `ShapeKey`；DP 下用全组最大前缀长度。 |
| `_create_chunked_prefix_buffers` | 1182 | 分配 `_ChunkedPrefixCaptureBuffers`，`starts` 固定为 `c * prefix_chunk_len`。 |
| `_prepare_chunked_prefix_capture` | 1208 | 捕获前缀变体时，把固定容量的 chunk 元数据挂到假 batch 上（所有 chunk 标记 `has_zero_kv`），并调用 backend 的 `prepare_full_cuda_graph_chunked_prefix(in_capture=True)`。 |
| `_populate_chunked_prefix_buffers` | 1251 | 按实际前缀长度刷新每个 chunk 的 `seq_lens`/`cu_seq_lens`，清零后用 kernel 重建 `kv_indices`。 |
| `_prepare_chunked_prefix_replay` | 1294 | 回放时用真实前缀长度（尾部槽补 0）刷新共享 buffer，并调用 backend 的 replay 准备。 |

### 3.7 attention / DCP 元数据

| 名称 | 行号 | 作用 |
|---|---|---|
| `_plan_dcp_metadata` | 1315 | 在临时 `forward_context` 中调用模型的 `prepare_context_parallel_metadata_for_dcp`，传入 `seq_lens`、前缀长度、`req_to_token`、KV buffer 行数、kv dtype 等，返回 DCP extend 元数据（含 `dcp_kv_buffer`）。 |
| `_init_forward_metadata_for_capture` | 1334 | 捕获期元数据初始化（非 Full 路径）：DCP 下先规划零前缀的 DCP 元数据，把其 `dcp_kv_buffer` 存入 `_dcp_kv_buffers[shape_key]`；然后普通 backend 调 `init_forward_metadata`，opt-in backend（如 DSV4）调 BCG 专用入口并缓存元数据对象。 |
| `_prepare_forward_metadata_for_replay` | 1362 | 回放期元数据刷新：DCP 下按真实 batch 重新规划，但换回捕获时的桶大小 buffer；Full 构造请求槽 padding 视图后调 `init_forward_metadata_out_graph`；普通 BCG 调 `init_forward_metadata`（空闲 DP rank 可跳过）；opt-in backend 原地刷新缓存元数据。返回 `SharedReadEnds`。 |
| `_restore_mha_capture_state`（static） | 1432 | 恢复 BCG 分段里缺失的 Python 状态（`mha_one_shot=True` 等），用于非 CUDA 的 MHA companion。 |

### 3.8 捕获流程

| 名称 | 行号 | 作用 |
|---|---|---|
| `_build_capture_spec_info` | 1618 | EAGLE draft 下用静态 hidden state buffer 构造 `EagleDraftInput`，否则 None。 |
| `capture_prepare` | 1627 | 为某个桶构造假 prefill `ForwardBatch`：按 `context_length` 拆成最少的合成请求（Full 补零长度槽），把形状输入写进静态 buffer 并换成其切片，从 registry 取 `input_ids`/`positions`/`out_cache_loc` 等槽，设置 DP 全局 token 数、LoRA、ngram，最后 `prepare_dummy_forward_batch` 和 TBO 插件。请求数超过 `max_bs` 时 assert。 |
| `capture` | 1800 | 在 `model_context()` 中调用 `_capture`。 |
| `_capture` | 1804 | warmup 后在 graph capture 流和 backend 的 capture session 内调用 `_capture_one_stream`，期间置 `capturing_prefill_graph` 标志。 |
| `_capture_one_stream` | 1826 | 按桶从大到小调用 `capture_one_shape`；Full 下每个桶再捕获各前缀变体。 |
| `capture_one_shape` | 1861 | 单个形状的捕获：构造假 batch，`prepare_for_forward`，CP BCG 下 `prefill_cp_bcg_input.prepare(capture=True)`，LoRA 准备，前缀变体准备，元数据初始化（Full 用 `init_forward_metadata_out_graph`，其他用 `_init_forward_metadata_for_capture`）；BCG 下为 mamba 状态池设置 warmup 后清槽的钩子；最后 `backend.capture_one`。 |

### 3.9 回放流程

| 名称 | 行号 | 作用 |
|---|---|---|
| `load_batch` | 1967 | 回放前准备：选桶（CP BCG 用 `select_replay_bucket_for_batch`，选不到报 `RuntimeError`），校验最大上下文，`buffer_registry.fill_from` 写入静态槽，EAGLE3 draft 写 input embeds，构造 `static_forward_batch`（MIXED 当 EXTEND 回放），写入 `_prefill_static_buffers`（Full 下把尾部槽清成哨兵，并清 mamba track 行），刷新 draft hidden state；CP BCG 下对 static batch 做 `prepare(capture=False)` 并以它作为元数据 batch；最后调 `_prepare_forward_metadata_for_replay`。 |
| `_zero_stale_cp_pp_proxy_rows` | 2266 | CP+PP 下，把收到的代理行之后、捕获本地行数之前的旧数据清零。 |
| `_fill_input_embeds_slot` | 2285 | 回放前把实际的 input embeds（多模态合成或 `get_input_embeddings()(input_ids)`）拷进静态槽。 |
| `_pack_qwen_bcg_hc_output` | 2307 | Qwen4-exp BCG 时把 `last_hc_hidden_states` 打包进输出，因为回放不会重放 Python 赋值。 |
| `_restore_qwen_bcg_hc_output` | 2313 | 回放后解包并恢复 `last_hc_hidden_states`。 |
| `_execute_body_capture` | 2318 | BCG/Full 回放：临时把 `layer_model.forward` 替换成"拷 embeds → `backend.replay` → Full 下切行"的闭包，然后 eager 调完整 `model.forward` 执行 LM head/logits；Full 下尾部用原始 batch 的浅拷贝并沿用 body 的 SP 判定。`finally` 中恢复原方法。 |
| `_execute_tc_piecewise` | 2402 | tc_piecewise 回放；不支持固定上下文长度（assert）。 |
| `_trim_logits_output` | 2423 | 把 `LogitsProcessorOutput` 裁到真实行数（Full 的 logits 按请求数，其他按 token 数），投机解码保留 `mm_input_embeds`。 |
| `_finalize_execute_output` | 2463 | 按输出类型分派裁剪：logits、pooler（原样）、PP 代理张量。 |
| `_validate_capture_hidden_mode` | 2475 | 运行时 hidden mode 超过捕获模式时报 `RuntimeError`。 |
| `_wait_for_layer_loads` | 2482 | Full 图绕过池访问器，需要等待逐层加载计数器完成；BCG 直接返回。 |
| `execute` | 2491 | 回放入口：校验 hidden mode → `load_batch` → 前缀变体准备 → 发布共享读完成事件 → 等待层加载 → 分派到 CP BCG / body capture / tc_piecewise → `_finalize_execute_output`。CP BCG 不支持固定上下文（assert）。 |

覆盖统计：模块级函数 11 个，类 2 个，`PrefillCudaGraphRunner` 方法 54 个（含 property 和 staticmethod）。

## 4. 与 PCP+DCP 方案 A 的关系

计划 §4.8 指向本文件 736-740 行（`_dcp_extend_active`），要求重新计算 CP buffer 和 DCP gather buffer 的静态尺寸，并在 P1、P2 阶段先禁用 prefill breakable graph。读代码后的具体情况如下。

**会被方案 A 修改或需要确认的位置：**

1. `__init__` 736-741 行：现在 `_dcp_extend_active` 只看 `attn_dcp_size > 1` 和模型方法，`enable_cp_bcg_capture`（651-667 行）只看 `supports_prefill_cp_bcg`。方案 A 下两者会同时为真，而现有代码从未假设这种组合。P1/P2 阶段最直接的做法是在这里（或 `cuda_graph_hook`）显式拒绝"CP BCG 与 DCP extend 同时开启"。
2. `_plan_dcp_metadata`（1315）：传入的是 `forward_batch.seq_lens`、`extend_seq_lens` 等**全局**请求信息。方案 A 需要合并后的元数据，同时描述本卡的 query 行（CP）和完整 KV 行（DCP）。这里的调用参数要跟随 §4.5 新增的合并元数据接口一起改。
3. `load_batch`（2245-2262）和 `_prepare_forward_metadata_for_replay`（1375-1381）：CP BCG 下元数据是基于经过 `prefill_cp_bcg_input.prepare` 的 `static_forward_batch` 规划的，然后 DCP 分支对同一个 batch 再规划一次 DCP 元数据。两者的先后顺序和依赖关系与 eager 路径（计划 §4.5 提到的 `eager_runner.py:264-300`）需要保持一致。
4. DCP gather buffer 尺寸（`_dcp_kv_buffers`，1341-1346）：现在用零前缀的捕获 batch 规划，buffer 大小等于桶的 token 数，前提是"无前缀时 gather 的行数等于本轮新 token 数"。方案 A 的 prefill 要 gather 整个序列（前缀加新 token），有前缀时尺寸取决于前缀长度，没有静态上界。因此 `can_replay_locally` 1511 行"DCP 下有前缀命中就走 eager"的规则在方案 A 下必须保留。另外 CP BCG 的桶是全局 token 数，而本卡 query 只有 `bucket_local_tokens` 行，gather buffer 应按全局行数、query 侧按本地行数计算（推断）。
5. `_get_layer_model_positions`（836）、`_pp_proxy_rows`（862）以及 `layers/cp/bcg/` 中的 `bucket_local_tokens`：都基于现有 interleave 的"拼接下标取模"布局。按计划 §4.3 改成按位置取模后，每个 rank 的本地行数会随前缀长度变化，捕获期的固定 `bucket_local_tokens` 可能不再成立（推断），这是需要禁用 BCG 的另一个原因。

**计划未提及的相关点：**

- **BCG 默认已经被关闭（计划未提及）。** `arg_groups/cuda_graph_hook.py` 的 `disable_breakable_cudagraph_if_incompatible` 中有两条规则会关掉 K3.1 的 BCG：`dcp_size > 1`（212-216 行，注释写"Capture builds a dummy extend forward with attn_dcp_metadata=None"），以及 KDA 混合线性注意力模型（GLM-5.3 Flash 除外）。所以计划里"P1/P2 先禁用"在默认配置下已经满足。但 `apply_cuda_graph_compatibility` 在用户显式指定 prefill backend 时会直接返回（`_cuda_graph_config_locked`），规则不生效。方案 A 的参数校验应该在 `--cuda-graph-backend-prefill=breakable` 被显式指定时明确报错，而不是依赖默认规则。
- **Full backend 与 DCP（计划未提及）。** 只有 BREAKABLE 有这些自动关闭规则，`disable_full_prefill_cudagraph_if_incompatible` 的规则列表是空的。`capture_one_shape` 在 Full 下走 `init_forward_metadata_out_graph`（1901-1905），不调用 `_init_forward_metadata_for_capture`，所以 `_dcp_kv_buffers` 不会被填充；但回放时 `_prepare_forward_metadata_for_replay` 的 DCP 分支（1375-1381）在 Full 前面执行，会读 `self._dcp_kv_buffers[shape_key]`，这里会出现 `KeyError`（推断，未运行验证）。方案 A 应同时拒绝 Full prefill 图，或者补上 Full 下的 DCP buffer 捕获。
- **KDA 状态槽清理。** `capture_one_shape` 在 BCG 下通过 `post_warmup_hook` 清掉捕获用过的 mamba/KDA 状态槽（1934-1951）。将来方案 A 若启用 BCG，KDA 状态的头切分（整个 TP 组）和 CP 行布局都会影响这里使用的 `req_pool_indices` 映射，需要一起验证（推断）。
- **`num_token_non_padded` 与 MoE。** `_capture_num_token_non_padded` 只在 attn-TP 序列切分时本地化计数。方案 A 下 MoE 先 gather 行再 reduce-scatter，`moe_num_token_non_padded`（计划 §4.3 提到）的计算在 BCG 下由 `PrefillCPBCGInput` 负责，本文件不直接处理。
