import pytest
import torch

from sglang.kernels.ops.attention.mla_kv_pack import pack_mla_kv_fp8
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=15, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.cuda is None,
    reason="Requires an NVIDIA GPU",
)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("heads,rope", [(12, 64), (24, 0)])
@pytest.mark.parametrize("gather,scaled", [(False, False), (True, True)])
def test_pack_projection_views(dtype, heads, rope, gather, scaled):
    torch.manual_seed(0)
    tokens, nope, value = 7, 128, 128
    kv = torch.randn(tokens, heads, nope + value, dtype=dtype, device="cuda")
    k_nope, v = kv.split([nope, value], dim=-1)
    # Positional K is itself a noncontiguous view of the latent projection.
    latent = torch.randn(tokens, 1, 512 + rope, dtype=dtype, device="cuda")
    k_pe = latent[..., 512:] if rope else None
    rows = torch.tensor([6, 0, 3, 3, 1], device="cuda") if gather else None
    k_scale = torch.tensor([0.7], device="cuda") if scaled else None
    v_scale = torch.tensor([1.3], device="cuda") if scaled else None

    k_out, v_out = pack_mla_kv_fp8(
        k_nope, k_pe, v, rows=rows, k_scale=k_scale, v_scale=v_scale
    )
    k_ref = (
        k_nope
        if k_pe is None
        else torch.cat((k_nope, k_pe.expand(-1, heads, -1)), dim=-1)
    )

    def quantize(x, scale):
        x = x.float()
        if scale is not None:
            x = (x * scale.reciprocal()).clamp(-448.0, 448.0)
        x = x.to(torch.float8_e4m3fn)
        return x if rows is None else x[rows]

    assert k_out.is_contiguous() and v_out.is_contiguous()
    torch.testing.assert_close(
        k_out.float(), quantize(k_ref, k_scale).float(), rtol=0, atol=0
    )
    torch.testing.assert_close(
        v_out.float(), quantize(v, v_scale).float(), rtol=0, atol=0
    )


@pytest.mark.parametrize("scale", [None, 0.7])
def test_fp8_rounding_and_overflow(scale):
    values = torch.tensor(
        [-float("inf"), -500, -464, -448, -0.0, 0, 1.0625, 448, 464, 500, float("inf")],
        dtype=torch.float16,
        device="cuda",
    )
    source = values.view(1, 1, -1)
    scale = None if scale is None else torch.tensor([scale], device="cuda")
    k, v = pack_mla_kv_fp8(source, source, source, k_scale=scale, v_scale=scale)
    expected = source.float()
    if scale is not None:
        expected = (expected * scale.reciprocal()).clamp(-448.0, 448.0)
    expected = expected.to(torch.float8_e4m3fn).float()
    torch.testing.assert_close(
        k.float(), torch.cat((expected, expected), -1), rtol=0, atol=0, equal_nan=True
    )
    torch.testing.assert_close(v.float(), expected, rtol=0, atol=0, equal_nan=True)
