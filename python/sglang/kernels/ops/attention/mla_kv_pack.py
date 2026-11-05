# SPDX-License-Identifier: Apache-2.0
"""Fuse MLA K concatenation, V extraction, row packing and FP8 conversion."""

from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def _to_e4m3(value, scale, SCALED: tl.constexpr):
    value = value.to(tl.float32)
    if SCALED:
        # Match static_scaled_fp8_quant: FP32 reciprocal, then saturation.
        value = value * tl.div_rn(1.0, tl.load(scale))
        value = tl.minimum(tl.maximum(value, -448.0), 448.0)
    else:
        # torch.to(e4m3fn) uses nonsaturating RN conversion: 464 rounds to
        # 448, larger magnitudes become NaN. Triton's native cast saturates.
        value = tl.where(tl.abs(value) > 464.0, float("nan"), value)
    # Convert directly from FP32, avoiding an extra FP16 rounding on SM89.
    bits = tl.inline_asm_elementwise(
        "cvt.rn.satfinite.e4m3x2.f32 $0, 0.0, $1;",
        constraints="=h,f",
        args=[value],
        dtype=tl.uint16,
        is_pure=True,
        pack=1,
    )
    return bits.to(tl.uint8).to(tl.float8e4nv, bitcast=True)


@triton.jit
def _pack_mla_kv_fp8_kernel(
    KNope,
    KPe,
    V,
    Rows,
    KScale,
    VScale,
    KOut,
    VOut,
    n_rows,
    HEADS: tl.constexpr,
    NOPE: tl.constexpr,
    ROPE: tl.constexpr,
    VALUE: tl.constexpr,
    KN_ROW: tl.constexpr,
    KN_HEAD: tl.constexpr,
    KN_COL: tl.constexpr,
    KP_ROW: tl.constexpr,
    KP_COL: tl.constexpr,
    V_ROW: tl.constexpr,
    V_HEAD: tl.constexpr,
    V_COL: tl.constexpr,
    GATHER: tl.constexpr,
    SCALE_K: tl.constexpr,
    SCALE_V: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_PE: tl.constexpr,
):
    # Adjacent logical rows are adjacent heads of the projection, giving
    # coalesced loads even though KNope/V are views of interleaved [K,V].
    head_rows = tl.program_id(0).to(tl.int64) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    token = head_rows // HEADS
    head = head_rows % HEADS
    valid = token < n_rows
    if GATHER:
        src = tl.load(Rows + token, mask=valid, other=0).to(tl.int64)
    else:
        src = token
    col = tl.arange(0, BLOCK_D)
    k = tl.load(
        KNope + src[:, None] * KN_ROW + head[:, None] * KN_HEAD + col[None, :] * KN_COL,
        mask=valid[:, None] & (col[None, :] < NOPE),
        other=0,
    )
    v = tl.load(
        V + src[:, None] * V_ROW + head[:, None] * V_HEAD + col[None, :] * V_COL,
        mask=valid[:, None] & (col[None, :] < VALUE),
        other=0,
    )
    k = _to_e4m3(k, KScale, SCALE_K)
    v = _to_e4m3(v, VScale, SCALE_V)
    tl.store(
        KOut + head_rows[:, None] * (NOPE + ROPE) + col[None, :],
        k,
        mask=valid[:, None] & (col[None, :] < NOPE),
    )
    tl.store(
        VOut + head_rows[:, None] * VALUE + col[None, :],
        v,
        mask=valid[:, None] & (col[None, :] < VALUE),
    )
    if ROPE:
        pe_col = tl.arange(0, BLOCK_PE)
        pe = tl.load(
            KPe + src[:, None] * KP_ROW + pe_col[None, :] * KP_COL,
            mask=valid[:, None] & (pe_col[None, :] < ROPE),
            other=0,
        )
        pe = _to_e4m3(pe, KScale, SCALE_K)
        tl.store(
            KOut + head_rows[:, None] * (NOPE + ROPE) + NOPE + pe_col[None, :],
            pe,
            mask=valid[:, None] & (pe_col[None, :] < ROPE),
        )


def pack_mla_kv_fp8(
    k_nope: torch.Tensor,
    k_pe: Optional[torch.Tensor],
    v: torch.Tensor,
    *,
    rows: Optional[torch.Tensor] = None,
    k_scale: Optional[torch.Tensor] = None,
    v_scale: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Write contiguous e4m3 K/V directly from the BF16 projection views.

    ``rows`` indexes source tokens, including repeated causal segments. None
    preserves request order. Scale tensors are provided only for non-unit
    scales; None matches the ordinary PyTorch FP8 cast. No intermediate BF16
    K, contiguous V, or gathered projection is materialized.
    """
    if k_nope.ndim != 3 or v.ndim != 3 or k_nope.shape[:2] != v.shape[:2]:
        raise ValueError("MLA pack requires matching [tokens, heads, dim] K/V")
    if (
        torch.version.cuda is None
        or not k_nope.is_cuda
        or v.device != k_nope.device
        or v.dtype != k_nope.dtype
    ):
        raise ValueError("MLA pack requires matching CUDA K/V tensors")
    if k_nope.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("MLA pack expects BF16/FP16 projection tensors")
    if rows is not None and (
        rows.device != k_nope.device
        or rows.ndim != 1
        or rows.dtype not in (torch.int32, torch.int64)
        or not rows.is_contiguous()
    ):
        raise ValueError("MLA row map must be a contiguous CUDA integer vector")
    for scale in (k_scale, v_scale):
        if scale is not None and (
            scale.device != k_nope.device
            or scale.dtype != torch.float32
            or scale.numel() != 1
        ):
            raise ValueError("MLA FP8 scale must be a scalar CUDA FP32 tensor")
    rope = 0 if k_pe is None else k_pe.shape[-1]
    if k_pe is not None and (
        k_pe.shape != (k_nope.shape[0], 1, rope)
        or k_pe.device != k_nope.device
        or k_pe.dtype != k_nope.dtype
    ):
        raise ValueError("MLA positional K must have shape [tokens, 1, rope_dim]")
    n_rows = k_nope.shape[0] if rows is None else rows.numel()
    heads, nope = k_nope.shape[1:]
    value = v.shape[-1]
    k_out = torch.empty(
        (n_rows, heads, nope + rope), dtype=torch.float8_e4m3fn, device=v.device
    )
    v_out = torch.empty(
        (n_rows, heads, value), dtype=torch.float8_e4m3fn, device=v.device
    )
    if n_rows:
        _pack_mla_kv_fp8_kernel[(triton.cdiv(n_rows * heads, 32),)](
            k_nope,
            k_pe,
            v,
            rows,
            k_scale,
            v_scale,
            k_out,
            v_out,
            n_rows,
            heads,
            nope,
            rope,
            value,
            *k_nope.stride(),
            0 if k_pe is None else k_pe.stride(0),
            0 if k_pe is None else k_pe.stride(-1),
            *v.stride(),
            rows is not None,
            k_scale is not None,
            v_scale is not None,
            32,
            triton.next_power_of_2(max(nope, value)),
            triton.next_power_of_2(max(rope, 1)),
            num_warps=8,
        )
    return k_out, v_out
