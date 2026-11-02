"""Bit-preserving, mixed-dtype row packing for K3's TP-SP input all-gather."""

import torch
import triton
import triton.language as tl


@triton.jit
def _unsigned_ptr(ptr):
    if ptr.dtype.element_ty.primitive_bitwidth == 16:
        return ptr.to(tl.pointer_type(tl.uint16))
    elif ptr.dtype.element_ty.primitive_bitwidth == 32:
        return ptr.to(tl.pointer_type(tl.uint32))
    else:
        return ptr.to(tl.pointer_type(tl.uint64))


@triton.jit
def _copy_packed_kernel(
    tensors,
    packed,
    valid_rows,
    COLS: tl.constexpr,
    STRIDES: tl.constexpr,
    OFFSETS: tl.constexpr,
    GAPS: tl.constexpr,
    BLOCKS_PER_ROW: tl.constexpr,
    PACK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = (tl.program_id(0) // BLOCKS_PER_ROW).to(tl.int64)
    block = tl.program_id(0) % BLOCKS_PER_ROW
    col = block * BLOCK + tl.arange(0, BLOCK)
    for i in tl.static_range(len(COLS)):
        field = _unsigned_ptr(tensors[i])
        source = field + row * STRIDES[i][0] + col * STRIDES[i][1]
        target = (packed + row * OFFSETS[-1] + OFFSETS[i]).to(field.dtype) + col
        if PACK:
            value = tl.load(source, (row < valid_rows) & (col < COLS[i]), other=0)
            tl.store(target, value, col < COLS[i])
            # Only the first column block fills the byte-alignment gap.
            if GAPS[i] > 0:
                if block == 0:
                    byte = tl.arange(0, 8)
                    tl.store(
                        packed + row * OFFSETS[-1] + OFFSETS[i + 1] - GAPS[i] + byte,
                        0,
                        byte < GAPS[i],
                    )
        else:
            value = tl.load(target, col < COLS[i], other=0)
            tl.store(source, value, col < COLS[i])


def _layout(tensors):
    alignment = max(t.element_size() for t in tensors)
    offsets = [0]
    for tensor in tensors:
        width = tensor.shape[1] * tensor.element_size()
        offsets.append(offsets[-1] + triton.cdiv(width, alignment) * alignment)
    return tuple(offsets)


def _copy_packed(tensors, packed, valid_rows, *, pack):
    cols = tuple(t.shape[1] for t in tensors)
    if packed.shape[0] and max(cols):
        offsets = _layout(tensors)
        blocks_per_row = triton.cdiv(max(cols), 1024)
        _copy_packed_kernel[(blocks_per_row * packed.shape[0],)](
            tuple(tensors),
            packed,
            valid_rows,
            cols,
            tuple(t.stride() for t in tensors),
            offsets,
            tuple(
                offsets[i + 1] - offsets[i] - t.shape[1] * t.element_size()
                for i, t in enumerate(tensors)
            ),
            blocks_per_row,
            pack,
            BLOCK=1024,
            num_warps=4,
        )


def pack_moe_inputs(tensors, padded_rows: int) -> torch.Tensor:
    """Pack latent, FP32 weights and integer IDs; zero all communication padding."""
    packed = torch.empty(
        (padded_rows, _layout(tensors)[-1]),
        dtype=torch.uint8,
        device=tensors[0].device,
    )
    _copy_packed(tensors, packed, tensors[0].shape[0], pack=True)
    return packed


def unpack_moe_inputs(packed: torch.Tensor, templates) -> tuple[torch.Tensor, ...]:
    """Unpack gathered rows into contiguous tensors with the templates' dtypes."""
    outputs = tuple(
        torch.empty((packed.shape[0], t.shape[1]), dtype=t.dtype, device=packed.device)
        for t in templates
    )
    _copy_packed(outputs, packed, packed.shape[0], pack=False)
    return outputs
