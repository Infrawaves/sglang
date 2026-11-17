# PCP + DCP 方案 A：涉及修改的代码文件清单

来源：`/Users/yhz/sglang/PCP_DCP_方案A_计划.md` 第 4 节"代码修改清单"。
下表的路径都相对于 `python/sglang/srt/`。每个文件对应一份代码理解文档，命名规则是：把路径里的 `/` 换成 `__`，再加上后缀 `_代码理解.md`。

| # | 代码文件 | 行数 | 计划中的章节 | 理解文档 |
|---|---|---|---|---|
| 1 | `arg_groups/parallel_hook.py` | 841 | §4.1 | [arg_groups__parallel_hook.py_代码理解.md](arg_groups__parallel_hook.py_代码理解.md) |
| 2 | `arg_groups/cp_tp_group_sharing_hook.py` | 141 | §4.1 | [arg_groups__cp_tp_group_sharing_hook.py_代码理解.md](arg_groups__cp_tp_group_sharing_hook.py_代码理解.md) |
| 3 | `arg_groups/model_overrides/kimi_k3.py` | 297 | §4.1 | [arg_groups__model_overrides__kimi_k3.py_代码理解.md](arg_groups__model_overrides__kimi_k3.py_代码理解.md) |
| 4 | `arg_groups/mega_moe_hook.py` | 324 | §4.1 | [arg_groups__mega_moe_hook.py_代码理解.md](arg_groups__mega_moe_hook.py_代码理解.md) |
| 5 | `arg_groups/hicache_hook.py` | 516 | §4.1 | [arg_groups__hicache_hook.py_代码理解.md](arg_groups__hicache_hook.py_代码理解.md) |
| 6 | `arg_groups/cuda_graph_hook.py` | 610 | §4.8 | [arg_groups__cuda_graph_hook.py_代码理解.md](arg_groups__cuda_graph_hook.py_代码理解.md) |
| 7 | `distributed/parallel_state.py` | 3550 | §4.2 | [distributed__parallel_state.py_代码理解.md](distributed__parallel_state.py_代码理解.md) |
| 8 | `runtime_context.py` | 2220 | §4.2 | [runtime_context.py_代码理解.md](runtime_context.py_代码理解.md) |
| 9 | `layers/cp/interleave.py` | 604 | §4.3、§4.4 | [layers__cp__interleave.py_代码理解.md](layers__cp__interleave.py_代码理解.md) |
| 10 | `layers/cp/bcg/interleave_bcg.py` | 149 | §4.3 | [layers__cp__bcg__interleave_bcg.py_代码理解.md](layers__cp__bcg__interleave_bcg.py_代码理解.md) |
| 11 | `layers/cp/utils.py` | 518 | §4.3 | [layers__cp__utils.py_代码理解.md](layers__cp__utils.py_代码理解.md) |
| 12 | `layers/cp/base.py` | 390 | §4.5 | [layers__cp__base.py_代码理解.md](layers__cp__base.py_代码理解.md) |
| 13 | `layers/cp/cp_decode_attn_tp.py` | 319 | §4.6 | [layers__cp__cp_decode_attn_tp.py_代码理解.md](layers__cp__cp_decode_attn_tp.py_代码理解.md) |
| 14 | `mem_cache/kv/alloc/id_space.py` | 362 | §4.4 | [mem_cache__kv__alloc__id_space.py_代码理解.md](mem_cache__kv__alloc__id_space.py_代码理解.md) |
| 15 | `layers/dcp/planner.py` | 197 | §4.4 | [layers__dcp__planner.py_代码理解.md](layers__dcp__planner.py_代码理解.md) |
| 16 | `layers/dcp/metadata.py` | 37 | §4.5 | [layers__dcp__metadata.py_代码理解.md](layers__dcp__metadata.py_代码理解.md) |
| 17 | `model_executor/runner/eager_runner.py` | 454 | §4.5 | [model_executor__runner__eager_runner.py_代码理解.md](model_executor__runner__eager_runner.py_代码理解.md) |
| 18 | `layers/attention/dsa_backend.py` | 5236 | §4.5 | [layers__attention__dsa_backend.py_代码理解.md](layers__attention__dsa_backend.py_代码理解.md) |
| 19 | `layers/attention/dsa/dsa_indexer.py` | 2594 | §4.5 | [layers__attention__dsa__dsa_indexer.py_代码理解.md](layers__attention__dsa__dsa_indexer.py_代码理解.md) |
| 20 | `models/deepseek_common/attention_forward_methods/forward_mla.py` | 1295 | §4.5、§4.6 | [models__deepseek_common__attention_forward_methods__forward_mla.py_代码理解.md](models__deepseek_common__attention_forward_methods__forward_mla.py_代码理解.md) |
| 21 | `models/kimi_k3.py` | 4725 | §4.6、§4.7 | [models__kimi_k3.py_代码理解.md](models__kimi_k3.py_代码理解.md) |
| 22 | `model_executor/runner/prefill_cuda_graph_runner.py` | 2540 | §4.8 | [model_executor__runner__prefill_cuda_graph_runner.py_代码理解.md](model_executor__runner__prefill_cuda_graph_runner.py_代码理解.md) |
| 23 | `mem_cache/kv_cache_configurator.py` | 3662 | §4.9 | [mem_cache__kv_cache_configurator.py_代码理解.md](mem_cache__kv_cache_configurator.py_代码理解.md) |
| 24 | `disaggregation/common/conn.py` | 4016 | §4.10 | [disaggregation__common__conn.py_代码理解.md](disaggregation__common__conn.py_代码理解.md) |
| 25 | `disaggregation/mooncake/conn.py` | 4160 | §4.10 | [disaggregation__mooncake__conn.py_代码理解.md](disaggregation__mooncake__conn.py_代码理解.md) |
| 26 | `mem_cache/hicache_storage.py` | 1333 | §4.11 | [mem_cache__hicache_storage.py_代码理解.md](mem_cache__hicache_storage.py_代码理解.md) |
| 27 | `mem_cache/pool_host/base.py` | 684 | §4.11 | [mem_cache__pool_host__base.py_代码理解.md](mem_cache__pool_host__base.py_代码理解.md) |

所有理解文档第 4 节发现的计划问题，汇总在 [方案A计划问题汇总.md](方案A计划问题汇总.md)。

说明：
- 计划 §4.7 还提到了 `layers/layer_boundary/` 目录，它包含多个文件，没有单独列入本清单。
- 计划 §9 中只作参考、不在 §4 修改清单里的文件（例如 `layers/dcp/comm.py`、`layers/dcp/layout.py`、`layers/attention/dsa/dsa_dcp_sharded.py`），也没有列入。
- 行数取自 `infrawaves/dev/v0.5.23-k31` 分支的当前版本。
