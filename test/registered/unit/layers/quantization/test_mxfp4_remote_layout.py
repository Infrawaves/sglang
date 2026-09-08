# SPDX-License-Identifier: Apache-2.0
"""Metadata parity for MXFP4 MoE remote-instance fast preparation.

K3's MoE resolves to ``Mxfp4MoEMethod``: compressed-tensors detects MXFP4 and
returns it *before* the scheme-based path, so the compressed-tensors scheme
hooks never run for it. These tests pin the ``Mxfp4MoEMethod`` hook to the
layout its regular post-load produces.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from torch import nn

from sglang.srt.layers.quantization.mxfp4 import Mxfp4MoEMethod
from sglang.srt.utils import get_device_sm
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="base-b", runner_config="4-gpu-b200")

NUM_EXPERTS = 2
HIDDEN_SIZE = 256
INTERMEDIATE_SIZE = 128


def _metadata(module: nn.Module):
    return [
        (name, tuple(param.shape), param.dtype)
        for name, param in module.named_parameters()
    ]


def _assert_manifest_is_contiguous(test: unittest.TestCase, module: nn.Module):
    for name, param in module.named_parameters():
        test.assertTrue(param.is_contiguous(), name)
        test.assertGreaterEqual(
            param.untyped_storage().nbytes(),
            param.numel() * param.element_size(),
            name,
        )


def _make_method(*, gate_up_interleaved: bool) -> Mxfp4MoEMethod:
    """Build the method without __init__: it reads global runtime backend state."""
    method = Mxfp4MoEMethod.__new__(Mxfp4MoEMethod)
    nn.Module.__init__(method)
    method.prefix = "test"
    method.topk_indices_dtype = None
    method.with_bias = True
    method.use_triton_kernels = False
    method.use_flashinfer = True
    method.use_marlin = False
    method.use_deep_gemm = False
    method.use_mega_moe = False
    method.flashinfer_mxfp4_moe_precision = "default"
    method._fi_kernel = "trtllm_sm100"
    method.moe_runner_config = SimpleNamespace(
        gemm1_alpha=None,
        gemm1_beta=None,
        gemm1_clamp_limit=None,
        gate_up_interleaved=gate_up_interleaved,
    )
    return method


def _make_layer(method: Mxfp4MoEMethod) -> nn.Module:
    layer = nn.Module()
    layer.num_local_experts = NUM_EXPERTS
    layer.hidden_size = HIDDEN_SIZE
    layer.intermediate_size_per_partition = INTERMEDIATE_SIZE
    layer.moe_runner_config = method.moe_runner_config
    method.create_weights(
        layer,
        num_experts=NUM_EXPERTS,
        hidden_size=HIDDEN_SIZE,
        intermediate_size_per_partition=INTERMEDIATE_SIZE,
        params_dtype=torch.bfloat16,
        with_bias=True,
    )
    layer.to("cuda")
    return layer


@unittest.skipUnless(
    torch.cuda.is_available() and get_device_sm() >= 100,
    "MXFP4 trtllm-gen layout preparation requires Blackwell",
)
class TestMxfp4RemoteLayout(CustomTestCase):
    def _regular_and_remote(self, *, gate_up_interleaved: bool):
        regular_method = _make_method(gate_up_interleaved=gate_up_interleaved)
        regular = _make_layer(regular_method)
        remote_method = _make_method(gate_up_interleaved=gate_up_interleaved)
        remote = _make_layer(remote_method)

        regular_method.process_weights_after_loading(regular)
        remote_method.process_weights_after_loading_for_remote_instance(remote)
        return regular, remote

    def test_moe_layout_and_parameter_order_match_regular_processing(self):
        # K3 Latent MoE is non-interleaved; gpt-oss style is interleaved.
        for gate_up_interleaved in (True, False):
            with self.subTest(gate_up_interleaved=gate_up_interleaved):
                regular, remote = self._regular_and_remote(
                    gate_up_interleaved=gate_up_interleaved
                )
                # Names, shapes, dtypes AND order: the NCCL backend broadcasts
                # parameters positionally, so order is part of the contract.
                self.assertEqual(_metadata(remote), _metadata(regular))
                _assert_manifest_is_contiguous(self, regular)
                _assert_manifest_is_contiguous(self, remote)

    def test_derived_gemm1_params_match_regular_processing(self):
        """The hook computes these rather than leaving them for the transfer.

        Their defaults live in two places now, so a change to the regular path
        that misses the hook would hand clients silently different values.
        """
        regular, remote = self._regular_and_remote(gate_up_interleaved=False)
        for name in ("gemm1_alpha", "gemm1_beta", "gemm1_clamp_limit"):
            torch.testing.assert_close(
                getattr(remote, name), getattr(regular, name), msg=name
            )


class TestMxfp4RemoteLayoutBranchSelection(CustomTestCase):
    """Backends whose repack is not metadata-derivable must not be fast-pathed."""

    def _method_with(self, **flags) -> Mxfp4MoEMethod:
        method = Mxfp4MoEMethod.__new__(Mxfp4MoEMethod)
        nn.Module.__init__(method)
        method.use_flashinfer = flags.get("use_flashinfer", False)
        method._fi_kernel = flags.get("fi_kernel", None)
        method.use_deep_gemm = flags.get("use_deep_gemm", False)
        method.use_mega_moe = flags.get("use_mega_moe", False)
        return method

    def test_non_trtllm_backends_run_the_regular_transform(self):
        cases = (
            {"use_flashinfer": False, "fi_kernel": None},  # marlin / aiter / triton
            {"use_flashinfer": True, "fi_kernel": "cutlass_sm90"},
            {"use_flashinfer": True, "fi_kernel": "cutlass_sm120"},
            # DeepGEMM and MegaMoE are checked BEFORE use_flashinfer in the
            # regular path, and use_mega_moe comes from the a2a backend, so it
            # can be set while the runner is flashinfer_mxfp4 on SM100. Both
            # must still take the regular DeepGEMM transform, or the client
            # builds a trtllm layout for DeepGEMM-packed bytes.
            {
                "use_flashinfer": True,
                "fi_kernel": "trtllm_sm100",
                "use_deep_gemm": True,
            },
            {
                "use_flashinfer": True,
                "fi_kernel": "trtllm_sm100",
                "use_mega_moe": True,
            },
        )
        for flags in cases:
            with self.subTest(**flags):
                method = self._method_with(**flags)
                layer = nn.Module()
                with mock.patch.object(
                    method, "process_weights_after_loading"
                ) as regular:
                    method.process_weights_after_loading_for_remote_instance(layer)
                regular.assert_called_once_with(layer)

    def test_trtllm_branch_skips_the_regular_transform(self):
        method = self._method_with(use_flashinfer=True, fi_kernel="trtllm_sm100")
        layer = nn.Module()
        layer.num_local_experts = 1
        layer.w13_weight = nn.Parameter(
            torch.zeros(1, 4, 2, dtype=torch.uint8), requires_grad=False
        )
        layer.w13_weight_scale = nn.Parameter(
            torch.zeros(1, 4, 1, dtype=torch.uint8), requires_grad=False
        )
        layer.w13_weight_bias = nn.Parameter(
            torch.zeros(1, 4, dtype=torch.bfloat16), requires_grad=False
        )
        layer.w2_weight = nn.Parameter(
            torch.zeros(1, 2, 2, dtype=torch.uint8), requires_grad=False
        )
        layer.w2_weight_scale = nn.Parameter(
            torch.zeros(1, 2, 1, dtype=torch.uint8), requires_grad=False
        )
        layer.w2_weight_bias = nn.Parameter(
            torch.zeros(1, 2, dtype=torch.bfloat16), requires_grad=False
        )
        layer.moe_runner_config = SimpleNamespace(
            gemm1_alpha=None, gemm1_clamp_limit=None
        )

        with mock.patch.object(method, "process_weights_after_loading") as regular:
            method.process_weights_after_loading_for_remote_instance(layer)
        regular.assert_not_called()

        # Scales are reinterpreted in place (uint8 and float8_e4m3fn are both
        # 1 byte); bias widens bf16 -> fp32 so it must be reallocated.
        self.assertEqual(layer.w13_weight_scale.dtype, torch.float8_e4m3fn)
        self.assertEqual(layer.w13_weight_scale.shape, (1, 4, 1))
        self.assertEqual(layer.w13_weight_bias.dtype, torch.float32)
        self.assertEqual(layer.w13_weight_bias.shape, (1, 4))
        self.assertEqual(layer.w2_weight_bias.shape, (1, 2))


if __name__ == "__main__":
    unittest.main()
