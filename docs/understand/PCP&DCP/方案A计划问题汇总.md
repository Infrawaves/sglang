# PCP + DCP 方案 A 计划：代码审读发现的问题汇总

来源：`docs/understand/` 下 27 份代码理解文档的第 4 节，以及各分组 agent 的汇报。审读对象是 `/Users/yhz/sglang/PCP_DCP_方案A_计划.md`。
这些问题都是读代码得出的，**没有运行验证**。标"（推断）"的条目确定性更低。条目按计划章节分组，每条注明出处的理解文档。

---

## 0. 结论摘要

1. **最大的正确性风险在 decode（§3.5 / §4.6）。** D2"复用 cp-decode-attn-tp"不能只加白名单：K3 没有调用 `maybe_use_decode_attn_tp`，`_stage_facts` 缺少延迟求和的声明，多处头数在 attention TP=1 时会放大 8 倍。几份文档对 D2 的看法也不完全一致（见 §5.1），需要先做实验确认。
2. **prefill 的元数据存在形状不一致的风险（§4.5）。** DCP gather 的本地行按 CP 过滤后的请求计算，gather 长度却按全部请求计算，可能算错或让 collective 挂起。
3. **计划高估了几处改动量。** §4.2 的 rank 公式在方案 A 约束下数值本来就相同；"只写本地"不是正确性的前提；`bcg_dcp_extend_kv_gather`、`_forward_cp_shard_rows`、KDA 的重排函数、`is_dcp_mla_decode_phase`、PD 的 rank 身份等可能都不需要改。
4. **计划漏掉了几个文件和调用点。** 例如 `model_overrides/deepseek_v2.py`、`model_executor/pool_configurator.py`、`dsa_backend.py` 的 `_forward_trtllm` 和头数公式、`forward_mla.py` 的 CP KV gather 调用点，以及 interleave 中其他按拼接下标计算的函数。
5. **性能目标可能定得过高（§5 P3）。** 方案 A 下 attention TP=1，K3 的融合 all-reduce、SP-MoE 和 SM100 的 prefill token packing 都会失效。"TPOT 与 TP8+DCP8 持平"未必能达到。

---

## 1. 参数与校验（计划 §4.1）

| # | 问题 | 出处 |
|---|---|---|
| 1.1 | hook 的执行顺序是：`handle_decode_context_parallelism`（pipeline 第 180 行）→ K3.1 model override（238）→ `handle_hicache`（285）→ `handle_context_parallelism`（291）。第 266 行做放行判断时，`cp_strategy` 可能还是 None；`handle_hicache` 运行时，`enable_cp_tp_group_sharing` 也还没声明。放行逻辑只能依据 `enable_prefill_cp`、`attn_cp_size` 和 `dcp_size` 来判断 | arg_groups__parallel_hook.py |
| 1.2 | `cp_tp_group_sharing_hook.py:79` 的 `"dcp-size > 1"` 是三个模型共用的。直接删掉会让 GLM-5 Next 和 Qwen4-Exp 也放行 DCP，应改成只对 K3.1 放行 | arg_groups__cp_tp_group_sharing_hook.py |
| 1.3 | `model_overrides/kimi_k3.py:240-241` 的拒绝条件包含 `enable_prefill_cp` 和 `attn_cp_size > 1` 两部分。应收紧成只允许方案 A 的组合，不能整项删除 | arg_groups__model_overrides__kimi_k3.py |
| 1.4 | **计划未列出这个文件**：`model_overrides/deepseek_v2.py` 的 `_deepseek_family_overrides` 同样作用于 K3.1。开启 prefill CP 时，它会声明 `attn_dp_size`、`attn_cp_size`，并断言 `tp_size <= 8`；开启 DCP 时还会检查 `index_kpool` | arg_groups__model_overrides__kimi_k3.py |
| 1.5 | P1 的后端应写成 `flashmla_kv`，不能只写"FlashMLA"。`_validate_dcp_backend`（`dsa_backend.py:539-567`）要求 DCP 下 prefill 和 decode 都用 `flashmla_kv`、`trtllm` 或 `prims_ts` 中的同一个；`flashmla_sparse` 和 `flashmla_auto` 会被拒绝 | layers__attention__dsa_backend.py |
| 1.6 | 建议在参数校验里断言 `dcp_size == attn_dcp_size == attn_cp_size`。K3.1 下 `self.dcp_size` 被置为 1，而部分 DCP 工具函数读的是 `parallel.dcp_size` | layers__attention__dsa_backend.py |
| 1.7 | `dcp_kv_layout == token` 已经被现有逻辑间接保证：page 布局只能用在 decode 服务器上，而 CP-TP sharing 拒绝 decode 模式 | arg_groups__parallel_hook.py |
| 1.8 | MegaMoE 的每卡 token 预算按 `ceil(tokens / cp)` 估算（`mega_moe_hook.py:243-245`）。方案 A 每个请求单独补齐，单卡最多约 `tokens/N + 请求数` 个 token，现有估算可能偏小（推断） | arg_groups__mega_moe_hook.py |
| 1.9 | HiCache 的说法前后不一致：§2.2 说 HiCache L3 放到 P4，§4.1 说第一期拒绝方案 A 和 HiCache 的组合。另外 K3.1 跳过了 `_dsa_dcp_overrides`（`deepseek_v2.py:105`），所以现在 K3.1 在 DCP 下允许 HiCache。方案 A 的拒绝逻辑需要新写 | arg_groups__hicache_hook.py |
| 1.10 | 用户显式传入 `--cuda-graph-backend-prefill` 时，会跳过自动关闭 BCG 的规则（见 §8）。P1/P2 需要显式报错 | arg_groups__cuda_graph_hook.py |
| 1.11 | DCP 下 MTP draft 的 `_build_mtp_kv_shard_pool` 会直接报错。计划没提到 MTP（EAGLE）draft，只在 P4 写了 DFLASH 和 DSPARK | mem_cache__kv_cache_configurator.py |

## 2. rank 拓扑（计划 §4.2）

| # | 问题 | 出处 |
|---|---|---|
| 2.1 | **改动量被高估。** 在 tp=cp=dcp=8、attn_tp=1 时，`dcp_rank = tp_rank % dcp_size` 和 `attn_cp_rank` 数值相同；`attn_dcp_size` 也已经等于 `attn_cp_size`；现有 DCP 组的成员也和 `_ATTN_CP` 相同。第一期加断言就够了，不需要改公式 | runtime_context.py、layers__cp__*、distributed__parallel_state.py |
| 2.2 | 如果真的要复用组对象：`_DCP` 在 `_ATTN_CP` 之前构建，`dcp_enabled` 依赖这个顺序；复用后会丢掉 DCP 组原来显式设置的 `use_quick_allreduce=False`；`destroy_model_parallel` 没有别名判断，会对同一个对象 destroy 两次；`patch_tensor_parallel_group` 不处理 DCP | distributed__parallel_state.py |
| 2.3 | `derive_parallel_widths` 有两个调用入口：publish 时经 `parallel_widths_of`，建组时在 `parallel_state.py:2738`。如果新增"DCP 复用 CP 组"的标志，两处都要传 | runtime_context.py |
| 2.4 | `_validate_parallel` 完全不校验 DCP 的 rank、宽度和通信组 | runtime_context.py |
| 2.5 | 字段 `attn_dcp_size` / `attn_dcp_rank` 的文档写的是"attention TP 组内"的 DCP（`fields/parallel.py:367-408`），方案 A 下 attention TP 为 1，和这个含义不符。约 157 处使用点里，MLA 路径需要逐个确认（推断） | runtime_context.py |
| 2.6 | `initialize_model_parallel` 的 docstring 还写着 DCP 只支持 HIP，已经过时 | distributed__parallel_state.py |

## 3. CP 切分策略（计划 §3.3、§4.3）

| # | 问题 | 出处 |
|---|---|---|
| 3.1 | 计划漏掉了几处同样按拼接下标计算的函数：`shard_per_request` 和它调用的 Triton kernel `dsa_cp_interleave_q_seqs`（由 `dsa_backend.py:1624` 调用）、`interleave_rows_per_request`、`utils.cp_interleave_input_ids`。这些要和新规则一起修改 | layers__cp__interleave.py、layers__cp__utils.py |
| 3.2 | 有现成的参考实现：`deepseek_v4_backend.py:2112-2170` 的 `_late_layer_tail_cp_layout` 已按 `owner = idx % cp` 构造过布局，元数据里也已经有 `gather_index`/`local_index` 字段。还有一个没人调用的扩展点 `register_cp_kv_layout`，它会检查 `--cp-strategy`，正好能实现"zigzag 报错" | layers__cp__interleave.py、layers__cp__base.py |
| 3.3 | KDA 的两个重排函数已经有支持任意排列的 `gather_index` 分支，在 `build_metadata` 里填好它就可以，`kimi_k3.py` 里的调用点可能不需要改（推断） | models__kimi_k3.py |
| 3.4 | `_forward_cp_shard_rows` 只依赖"按 rank 顺序 gather、每个 rank 补齐到相同长度"，而且 MoE 是逐行计算的，可能不需要改。真正要改的是层边界的 gather 和 padding（推断） | models__kimi_k3.py |
| 3.5 | MIXED 模式下 decode 尾巴混在 extend 里，按位置取模的切分要能处理只有 1 个 token 的 extend 段（计划未提及） | model_executor__runner__eager_runner.py |
| 3.6 | `EagerRunner.__init__`（120-124 行）按 CP padding 对齐 `max_num_token`。每个请求单独补齐后，`get_cp_padding_align_size` 和这里的静态缓冲上限要一起核对（计划未提及） | model_executor__runner__eager_runner.py |
| 3.7 | MegaMoE 要求有效行排在本卡分片的前部，新的 padding 布局要满足这一点 | models__kimi_k3.py |
| 3.8 | 合并后的元数据如果用 `msgspec.Struct`，不能设 `frozen=True`，因为 CP 元数据会被原地修改；`msgspec.Struct` 也不能继承现有的 dataclass 基类 | layers__cp__base.py、layers__dcp__metadata.py |

## 4. KV 写入（计划 §4.4）

| # | 问题 | 出处 |
|---|---|---|
| 4.1 | **"只写本地"不是正确性的前提。** `UnitIds.translate_write` 会把不归本卡的写入映射到 sink 0，indexer K 的融合 store 也会按 owner 过滤。所以 P1 可以保留现在"gather 后写全部行"的做法，结果仍然正确，只是多一次新 token 的 gather，改动更少（推断） | mem_cache__kv__alloc__id_space.py、layers__attention__dsa__dsa_indexer.py |
| 4.2 | 上一条也削弱了计划 §3.3 中"zigzag 必须走 Z1 all-to-all"的论证（推断）。zigzag 仍然不适用于 K3.1，但原因是 DSA 只支持 interleave，不是 owner 规则对不上 | layers__cp__interleave.py |
| 4.3 | 如果要改成只写本地，需要同时把写入用的 `out_cache_loc` 换成本卡行对应的子集。现在 `prepare_cp_forward` 只截取到 `sum(extend)`，不按卡切分（`layers/cp/utils.py:303-304`）。这一点涉及 `eager_runner.py`、`cp/utils.py` 和 `dsa_indexer.py:1094-1095` 三处 | model_executor__runner__eager_runner.py、layers__attention__dsa__dsa_indexer.py |
| 4.4 | **计划漏掉了一处 KV gather**：`dsa_backend.py` 的 `_forward_trtllm`（4508-4556 行）自己会通过 `all_gather_dsa_trtllm_fp8_kv` 和 `cp_live_store_rows` 再做一次 CP 的 KV gather。如果 P1 选 trtllm 或 prims_ts，这里也要处理 | layers__attention__dsa_backend.py |
| 4.5 | **计划漏掉了 `forward_mla.py` 里的调用点**：CP 和 DCP 两条 KV 路径会同时执行。`forward_mla.py` 第 472-480 行构造 `cp_overlap`，第 866 行调用 `prepare_kv_for_attention`，会触发 CP 的 `materialize_full_mla_kv`；DSA 后端在 DCP 下又会 gather 整个序列的 KV。这两处需要关掉 CP 的 KV gather，避免重复通信 | models__deepseek_common__attention_forward_methods__forward_mla.py |
| 4.6 | `planner.py:117` 在所有请求前缀拼接后的数组上按步长切片，要求每个请求的前缀长度都是 dcp 的倍数，也就是 chunk 边界必须按放大页对齐 | layers__dcp__planner.py |

## 5. prefill attention（计划 §4.5）

| # | 问题 | 出处 |
|---|---|---|
| 5.1 | **形状不一致的风险**：`dcp_gather_kv_cache`（`dsa_backend.py:1392`）的本地行 `dcp_local_kv_rows`，来自按 CP 过滤后的 `indexer_seq_lens_cpu`；gather 的长度却用未过滤的 `forward_batch.seq_lens_cpu`。CP 下有些请求在某张卡上没有 query 行，各卡的输入形状就会不一致，可能算错，也可能让 collective 挂起（推断）。`_dcp_gather_index_k` 依赖同一套长度 | layers__attention__dsa_backend.py |
| 5.2 | `topk_indices_offset`、`cu_seqlens_k`、`_cal_indexer_k_start_end` 都按过滤后的请求计算。DCP 的 ragged top-k 需要它们指向 gather 后的完整布局。计划需要写明：**query 侧按本卡行构建，KV 侧按完整请求列表构建** | layers__attention__dsa_backend.py |
| 5.3 | "按全局位置做 causal"实际要在 DSA 后端的 `get_indexer_metadata` 里实现：`ks`/`ke` 要对应本卡行的全局位置。`_get_topk_ragged` 的打分代码可能不需要改；它在第 1807、1826 行有长度断言，可以作为检查点 | layers__attention__dsa__dsa_indexer.py |
| 5.4 | `is_dcp_mla_decode_phase` 在 `dsa_shards_index_k()` 为真时已经排除了 extend（`forward_mla.py:210`），可能不需要改（推断） | models__deepseek_common__attention_forward_methods__forward_mla.py |
| 5.5 | **计划有误**：K3.1 不经过 `bcg_dcp_extend_kv_gather`。`forward_mla.py:884` 的条件是 `if not self.use_dsa`，DSA 模型的 KV gather 在 DSA 后端内部完成。这个函数只和以后支持非 DSA 的 MLA 模型有关 | models__deepseek_common__attention_forward_methods__forward_mla.py |
| 5.6 | indexer 的 CP overlap（`forward_mla.py:666-675` 的 `cp_overlap.run_indexer`）在 indexer 运行期间重叠的是 CP 的 KV gather。方案 A 下要把重叠对象换成 DCP 的整序列 gather，或者先关掉 overlap（计划未提及） | models__deepseek_common__attention_forward_methods__forward_mla.py |
| 5.7 | `prepare_context_parallel_metadata_for_dcp` 的参数里没有 `forward_batch`，也没有 CP 元数据。要构建合并后的元数据，就得改模型里的两层方法和两个 runner 调用方。计划只提到了 runner | models__kimi_k3.py |
| 5.8 | `eager_runner.py` 的调用顺序已经是 CP 元数据在前，这一点不用改。第 288-291 行的条件，在第 281 行已排除的前提下永远为假，属于冗余条件（推断） | model_executor__runner__eager_runner.py |
| 5.9 | 计划称 `eager_runner.py` 是"冻结的核心文件"，这不准确。仓库规则只把 `model_runner.py` 列为冻结文件。不过修改它前仍建议先读 `large-class-style`，把新逻辑放进 `layers/cp/` 或 `layers/dcp/` 的协作者里 | model_executor__runner__eager_runner.py |
| 5.10 | 方案 A 下 indexer 的 TP 行切分会自动关闭（`_indexer_tp_split` 在 CP 生效时返回 None），不需要改。DCP 分片下 indexer K 只能由 CUDA 的融合 store 写入，非 CUDA 平台和 fp8 fnuz 平台不支持方案 A | layers__attention__dsa__dsa_indexer.py |

## 6. decode attention（计划 §3.5、§4.6）

> 这一节几份文档的结论不完全一致，P2 之前需要先用小规模实验确认。

| # | 问题 | 出处 |
|---|---|---|
| 6.1 | **K3 没有调用 `maybe_use_decode_attn_tp`。** DeepSeek-V2 在 `deepseek_v2.py:2912` 用它包住 attention，K3 的 `_run_self_attn`（`kimi_k3.py:3312`）直接调用 `self_attn`。只加白名单不会产生切片效果 | models__kimi_k3.py |
| 6.2 | **`maybe_use_decode_attn_tp` 切的不只是 o_proj**，还有 `q_b_proj`、`w_kc`、`w_vc`、`tp_q_head_num` 和 `num_local_heads`（`deepseek_v2.py:2341-2366`）。<br>两份文档的看法不同：<br>· cp_decode_attn_tp 的文档认为 DCP decode 要求每卡对全部头计算，所以需要一种只切 o_proj 和 `g_proj` 的新模式；<br>· forward_mla 的文档认为切片后 Q 先按 H/8 个头计算，再在 DCP 组内 all-gather 回 H 个头，合并后每卡剩 H/8 个头，正好对上切过的 o_proj，所以 D2 可以成立（推断）。<br>**需要实验确认哪个结论正确。** 另外还要考虑和 `dcp_replicate_q_proj` 的交互 | layers__cp__cp_decode_attn_tp.py、models__deepseek_common__attention_forward_methods__forward_mla.py |
| 6.3 | **K3 的 `_stage_facts` 缺少延迟求和的声明。** K3 构造父类时用 `reduce_results=False`，方案 A 下 `defer_cp_decode_attn_tp` 会变成真；而 DeepSeek-V2 声明了 `ordinary_tp_group=SumGroup.TP`（`deepseek_v2.py:2790-2840`），K3 没有。推断的后果是：decode 时 o_proj 的部分和不会被求和，结果静默出错 | models__kimi_k3.py |
| 6.4 | **头数放大 8 倍的几处地方：**<br>· `dsa_backend.py:755-758` 的 `num_q_heads = num_attention_heads // attn_tp_size × attn_dcp_size`，在方案 A 下得到 768；<br>· 父类 `attn_mqa_for_dcp_decode` 按 `num_local_heads × dcp_size` 计算（`deepseek_v2.py:2261`），得到 512；<br>· `forward_mla.py:878` 的 Q all-gather，在未切片时会得到 H×8 个头；<br>· `_prepare_dcp_v_weights` 如果在 `post_load_weights`（`kimi_k3.py:4101-4102`）里、切片上下文之外调用，`w_vc` 会被 gather 成 512 个头 | layers__attention__dsa_backend.py、models__kimi_k3.py、forward_mla.py |
| 6.5 | `_prepare_dcp_v_weights` 的缓存一旦建立就不再变化。D2 下必须保证它在切片后的上下文中创建，权重更新时也在同样的上下文中覆盖 | models__deepseek_common__attention_forward_methods__forward_mla.py |
| 6.6 | cp-decode-attn-tp 的生效范围比 decode 宽：`set_decode_attn_tp` 在所有非 CP 的前向上都生效，包括 target verify，以及总 extend 长度小于 cp_size 时退回非 CP 的短 extend。这个范围要和 `is_dcp_mla_decode_phase` 对齐 | layers__cp__cp_decode_attn_tp.py |
| 6.7 | DCP 合并后，第 r 段头留在 DCP rank r 上；cp-decode-attn-tp 按 CP rank r 取第 r 段头。在 `dcp_rank == attn_cp_rank` 时两者一致（推断） | models__deepseek_common__attention_forward_methods__forward_mla.py |
| 6.8 | 输出门 `g_proj` 在 decode graph 下走侧流，切片缓存要覆盖这条路径 | models__kimi_k3.py |

## 7. 性能（计划 §5 P3、§7）

| # | 问题 | 出处 |
|---|---|---|
| 7.1 | 方案 A 下 attention TP=1，`_fuses_attn_all_reduce` 要求 attn_tp > 1，所以 K3 的融合 all-reduce 和 SP-MoE 都会失效；TP8+DCP8 下这两项是开启的。"TPOT 与 TP8+DCP8 持平"可能达不到 | models__kimi_k3.py |
| 7.2 | DCP 加上 prefill CP 时，SM100 的 `flashmla_prefill_token_packing` 会被关闭（`dsa_backend.py:811`） | layers__attention__dsa_backend.py |

## 8. CUDA Graph（计划 §4.8）

| # | 问题 | 出处 |
|---|---|---|
| 8.1 | **计划引用的行号不对**：真正关闭 K3.1 prefill BCG 的是 `cuda_graph_hook.py:213-216`（`dcp_size > 1` 的规则），不是 208-212。KDA 模型也有一条关闭规则。所以 P1/P2 默认就会禁用 prefill graph | arg_groups__cuda_graph_hook.py、model_executor__runner__prefill_cuda_graph_runner.py |
| 8.2 | Full prefill graph 加 DCP 在回放时可能出错：捕获时不会填充 `_dcp_kv_buffers`，回放时第 1375-1381 行会读取它，应该会 KeyError（推断）。Full 模式的不兼容规则列表是空的，方案 A 应该同时拒绝 Full | model_executor__runner__prefill_cuda_graph_runner.py |
| 8.3 | CP BCG 和 DCP extend 从没同时开启过。DCP gather buffer 是按零前缀的 batch 捕获的，所以第 1511 行"有前缀命中就走 eager"的规则必须保留 | model_executor__runner__prefill_cuda_graph_runner.py |
| 8.4 | 以后开启 prefill graph 时，`dsa_prefill_cuda_graph.py`（indexer 的 BCG/PCG split）也要支持方案 A 的元数据，计划只提到了静态尺寸 | layers__attention__dsa__dsa_indexer.py |

## 9. 调度与显存核算（计划 §4.9）

| # | 问题 | 出处 |
|---|---|---|
| 9.1 | **计划指向的文件不准**：每 token 字节数在 `model_executor/pool_configurator.py:452-462` 计算，不在 `kv_cache_configurator.py`。后者只负责按 `attn_dcp_size` 放大分配器的 id 空间（2947-2959 行） | mem_cache__kv_cache_configurator.py |
| 9.2 | K3.1 走的是 `_build_hybrid_linear_kv_pool`，不走 `_build_dsa_kv_pool`。DSA 池分支里的 GLM CP 层切分和 `dcp_replicated` 对 K3.1 不起作用 | mem_cache__kv_cache_configurator.py |
| 9.3 | KDA 状态池的大小，要按整个 TP 组的 KDA 头切分来算（每卡 12 个头），不能按 `attn_tp_size = 1` 来算，需要确认（推断） | mem_cache__kv_cache_configurator.py |
| 9.4 | 方案 A 下要确认 `get_parallel().dcp_enabled` 为真，否则 `page_size == 1` 时会选成逐 token 分配器（2937 行）。`page_size % dcp == 0` 这个约束来自 `pool_host/base.py:242`，不来自 `kv_cache_configurator.py` | mem_cache__kv_cache_configurator.py、mem_cache__pool_host__base.py |

## 10. PD 分离（计划 §4.10）

| # | 问题 | 出处 |
|---|---|---|
| 10.1 | **计划高估了改动量**：在 CP-TP sharing 下，`enable_all_cp_ranks_for_transfer` 和 `is_dummy_cp_rank` 已经是 False，每个 CP rank 都会发送；注册时 CP 已被当作 TP 处理；两边 DCP 相同时会走整页传输；KDA 状态在两边都是 TP8 时按 rank 一一对应传输。这几项可能都不需要改 | disaggregation__common__conn.py、disaggregation__mooncake__conn.py |
| 10.2 | **只能配同样 DCP 大小的 decode。** DCP8 发给 DCP1 或 DCP4 的 decode 时，`requires_dcp_relayout` 会在处理注册消息时抛异常，这个会话上的所有请求都会失败。注册信息和 `PrefillServerInfo` 也不包含 prefill 的 `dcp_size`，decode 没法提前拒绝。建议在注册信息里加上 `dcp_size`/`dcp_kv_layout`，并在 `try_ensure_parallel_info` 里校验 | disaggregation__mooncake__conn.py |
| 10.3 | 按页传输时没有校验 KV 布局。如果 prefill 的 KV 池其实没有按 DCP 切分，传输会把数据写到错误的位置，而且不报错。建议在 `dst_dcp_size > 1` 时校验两边的 `kv_item_lens` 相同，并且都用 token 布局 | disaggregation__mooncake__conn.py |
| 10.4 | 方案 A 能通过 PD 校验，是因为 CP 被注册成了 TP（decode 侧第 1758 行要求 prefill CP=1）。以后扩展到不用 CP-TP sharing 的模型时，会在这里被拒绝 | disaggregation__common__conn.py |
| 10.5 | 方案 A 的实例不能当 decode 用：`_resolve_rank_mapping` 断言 decode 侧 `attn_cp_size == 1`（1852 行）。另外 decode 的 TP 必须等于 prefill 的 CP/DCP 大小，否则会把部分 prefill rank 当作 dummy，丢掉 KV | disaggregation__common__conn.py |
| 10.6 | 代码里混用了真实的 attention TP（1）和注册时的 TP（8）。会出问题的地方有：开启 unified memory 时 `_validate_envelope_kv_layout` 抛错；`_prefetch_staging_reqs` 总是判断为需要 staging；DFLASH 分支的 `_validate_peer_state_layout` 被误触发 | disaggregation__common__conn.py |
| 10.7 | 按页传输时不检查 `kv_chunk.index_slice` 和 `decode_prefix_len` 是否按 DCP 虚拟页（`page_size × dcp`）对齐。draft KV（MTP/DFLASH）两边的布局也必须一致 | disaggregation__mooncake__conn.py |
| 10.8 | `prefill_complete` 分配策略要求对端是 TP1/CP1/PP1，不能和方案 A 一起用。PP>1 在多处有硬性限制 | disaggregation__common__conn.py |

## 11. HiCache（计划 §4.11）

| # | 问题 | 出处 |
|---|---|---|
| 11.1 | `hicache_storage.py:68-73` 的校验在典型配置下本来就能通过（`tp_rank == dcp_rank`）。要改它，主要是为了统一语义，以及支持以后 `attn_cp < tp` 的情况；`is_storage_writer`（88-98 行）同理 | mem_cache__hicache_storage.py |
| 11.2 | **计划未提及**：`HiCacheFile._init_config_suffix`（852-876 行）在方案 A 下生成的 key 后缀会同时包含 CP 段和 DCP 段。方案 A 和纯 TP8+DCP8 的 KV 布局其实相同，但带上 CP 段后，两种部署无法共享 L3 文件。需要决定方案 A 下是否省掉 CP 段 | mem_cache__hicache_storage.py |
| 11.3 | unified key scheme 和 DCP 互斥（808-809 行），方案 A 只能用普通的 key scheme | mem_cache__hicache_storage.py |
| 11.4 | Mooncake 等 L3 后端也使用 `HiCacheStorageConfig` 的 `dcp_rank`、`is_storage_writer` 和 `l3_page_span`，P4 修改时要一起检查 | mem_cache__hicache_storage.py |
| 11.5 | `pool_host/base.py` 的 `dcp_rank` 由调用方传入（`attn_dcp_rank`），会自动跟着 §4.2 的改动变化，本文件不需要改 | mem_cache__pool_host__base.py |

---

## 12. 建议对计划做的修订

1. 把 §4.2 从"修改公式"改成"在参数校验里断言三者相等，并补齐 `_validate_parallel` 对 DCP 的校验"。
2. §4.4 改成"P1 保留 gather 后写全部行的做法，依靠 owner 过滤保证正确性；在 P3 再优化成只写本地"，并补上 `_forward_trtllm` 和 `forward_mla.py` 的两个调用点。
3. §4.5 写明"query 侧按本卡行构建，KV 侧按完整请求列表构建"，并补上 `prepare_context_parallel_metadata_for_dcp` 的签名修改。删除对 `bcg_dcp_extend_kv_gather` 的修改要求。
4. §4.6 改成先做一个"D2 可行性实验"，覆盖 6.1–6.8 的检查点，再决定选 D1 还是 D2。
5. §4.1 补上 `model_overrides/deepseek_v2.py`，P1 的后端写明 `flashmla_kv`，并统一 HiCache 的处理时机。
6. §4.8 修正行号，并加上"拒绝显式指定的 breakable 和 full prefill graph"。§4.9 改为指向 `pool_configurator.py`。
7. §4.10 缩小改动范围，重点放在 10.2、10.3、10.6 和 10.7。
8. §5 的 P3 验收标准要考虑 7.1 和 7.2 两项性能损失。
