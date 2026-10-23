"""CPU routing regression checks; GPU quantizers and kernel calls are mocked."""

import ast
import contextvars
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple, cast
from unittest.mock import Mock

import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _load(path, names, namespace):
    # Load real dispatch functions and routing types without importing CUDA
    # extensions. Replace their direct dependency imports with fixture globals.
    nodes = [
        n for n in ast.parse(path.read_text()).body if getattr(n, "name", None) in names
    ]
    for node in nodes:
        if isinstance(node, ast.FunctionDef):
            node.body = [n for n in node.body if not isinstance(n, ast.ImportFrom)]
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *nodes,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)


@pytest.fixture
def runner():
    root = Path(__file__).resolve().parents[5] / "python/sglang/srt"
    ns = dict(torch=torch, NamedTuple=NamedTuple, cast=cast)
    _load(
        root / "layers/moe/topk.py",
        {
            "TopKOutputChecker",
            "StandardTopKOutput",
            "StandardTopKOutputPacked",
            "PackedTopKOutput",
            "BypassedTopKOutput",
        },
        ns,
    )
    _load(
        root / "layers/moe/token_dispatcher/standard.py", {"StandardCombineInput"}, ns
    )
    _load(
        root / "layers/moe/moe_runner/flashinfer_trtllm.py",
        {
            "_get_routing_for_flashinfer_routed",
            "fused_experts_none_to_flashinfer_trtllm_fp8",
        },
        ns,
    )
    ns.update(
        FlashinferDispatchOutput=type("FlashinferDispatchOutput", (), {}),
        Fp8QuantizationType=SimpleNamespace(MxFp8=1, DeepSeekFp8=0),
        RoutingMethodType=SimpleNamespace(DeepSeekV3=2, TopK=5),
        _deferred_finalize_enabled=contextvars.ContextVar("defer", default=False),
        use_symmetric_memory=lambda *a, **kw: nullcontext(),
        get_parallel=lambda: SimpleNamespace(tp_group=None),
        is_allocation_symmetric=lambda: False,
        next_power_of_2=lambda n: 1 << (n - 1).bit_length(),
        per_token_group_quant_fp8=Mock(
            return_value=(torch.zeros(2, 4), torch.ones(1, 2))
        ),
        trtllm_fp8_block_scale_routed_moe_out_wrapper=Mock(),
        trtllm_fp8_block_scale_moe_out_wrapper=Mock(),
    )
    quant = SimpleNamespace(
        block_quant=True,
        use_mxfp8=False,
        weight_block_k=128,
        w13_weight_scale_inv=torch.ones(1),
        w2_weight_scale_inv=torch.ones(1),
        w13_weight=None,
        w2_weight=None,
        gemm1_alpha=None,
        gemm1_beta=None,
        gemm1_clamp_limit=None,
        global_num_experts=8,
        intermediate_size=4,
        local_expert_offset=0,
        local_num_experts=8,
        routing_method_type=2,
        activation_type=None,
    )
    config = SimpleNamespace(
        activation="silu", no_combine=False, top_k=2, routed_scaling_factor=1.0
    )
    weights = torch.tensor([[0.25, 0.75], [0.6, 0.4]])
    ids = torch.tensor([[1, 3], [2, 4]], dtype=torch.int32)
    topk = ns["StandardTopKOutput"](weights, ids, None)
    dispatch = SimpleNamespace(hidden_states=torch.zeros(2, 4), topk_output=topk)
    return SimpleNamespace(ns=ns, quant=quant, config=config, dispatch=dispatch)


def test_standard_reuses_precomputed_routing(runner):
    r = runner
    # Packed all-gather outputs can have strided columns.
    topk = r.dispatch.topk_output
    r.dispatch.topk_output = topk._replace(
        topk_ids=topk.topk_ids.repeat(1, 2)[:, ::2],
        topk_weights=topk.topk_weights.repeat(1, 2)[:, ::2],
    )
    result = r.ns["fused_experts_none_to_flashinfer_trtllm_fp8"](
        r.dispatch, r.quant, r.config
    )
    kernel = r.ns["trtllm_fp8_block_scale_routed_moe_out_wrapper"]
    kernel.assert_called_once()
    args = kernel.call_args.kwargs
    torch.testing.assert_close(args["topk_ids"], r.dispatch.topk_output.topk_ids)
    torch.testing.assert_close(
        args["topk_weights"], r.dispatch.topk_output.topk_weights
    )
    assert args["topk_ids"].is_contiguous() and args["topk_weights"].is_contiguous()
    assert args["routing_method_type"] == r.ns["RoutingMethodType"].TopK
    assert args["routing_bias"] is None and args["routed_scaling_factor"] == 1.0
    assert result.hidden_states is args["output"]
    r.ns["trtllm_fp8_block_scale_moe_out_wrapper"].assert_not_called()


def test_bypassed_keeps_logits_kernel(runner):
    r = runner
    logits = torch.zeros(2, 8)
    cfg = SimpleNamespace(
        correction_bias=torch.ones(8), top_k=2, num_expert_group=1, topk_group=1
    )
    r.dispatch.topk_output = r.ns["BypassedTopKOutput"](
        r.dispatch.hidden_states, logits, cfg
    )
    r.ns["fused_experts_none_to_flashinfer_trtllm_fp8"](r.dispatch, r.quant, r.config)
    kernel = r.ns["trtllm_fp8_block_scale_moe_out_wrapper"]
    kernel.assert_called_once()
    assert kernel.call_args.kwargs["routing_logits"] is logits
    assert kernel.call_args.kwargs["routing_bias"] is cfg.correction_bias
    r.ns["trtllm_fp8_block_scale_routed_moe_out_wrapper"].assert_not_called()


def test_per_tensor_rejects_precomputed_routing_before_quantization(runner):
    r = runner
    r.quant.block_quant = False
    with pytest.raises(NotImplementedError, match="per-tensor quantization"):
        r.ns["fused_experts_none_to_flashinfer_trtllm_fp8"](
            r.dispatch, r.quant, r.config
        )
    r.ns["per_token_group_quant_fp8"].assert_not_called()


def test_precomputed_routing_preserves_deferred_finalize_guard(runner):
    r = runner
    r.ns["_deferred_finalize_enabled"].set(True)
    with pytest.raises(RuntimeError, match="FP8 deferred finalize requires"):
        r.ns["fused_experts_none_to_flashinfer_trtllm_fp8"](
            r.dispatch, r.quant, r.config
        )
    r.ns["per_token_group_quant_fp8"].assert_not_called()
