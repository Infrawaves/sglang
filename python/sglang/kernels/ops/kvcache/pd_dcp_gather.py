from typing import Optional, Sequence, Tuple

import torch
import triton
import triton.language as tl


@triton.jit
def _copy_mla_rows_into_pack_kernel(
    src_metadata,
    row_indices,
    pack,
    num_rows,
    BLOCK_SIZE: tl.constexpr,
):
    segment_id = tl.program_id(0)
    block_id = tl.program_id(1)
    metadata_offset = segment_id * 5
    src = tl.load(src_metadata + metadata_offset).to(pack.dtype)
    row_nbytes = tl.load(src_metadata + metadata_offset + 1)
    pack_offset = tl.load(src_metadata + metadata_offset + 2)
    src_row_stride = tl.load(src_metadata + metadata_offset + 3)
    pack_row_stride = tl.load(src_metadata + metadata_offset + 4)

    offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    layer_nbytes = num_rows * row_nbytes
    mask = offsets < layer_nbytes
    row = offsets // row_nbytes
    byte = offsets % row_nbytes
    src_row = tl.load(row_indices + row, mask=mask, other=0)
    values = tl.load(src + src_row * src_row_stride + byte, mask=mask)
    tl.store(pack + pack_offset + row * pack_row_stride + byte, values, mask=mask)


def _launch_copy_rows(
    metadata: Sequence[int],
    row_indices: torch.Tensor,
    pack: torch.Tensor,
    max_row_nbytes: int,
) -> None:
    num_segments = len(metadata) // 5
    n = int(row_indices.numel())
    src_metadata = torch.tensor(metadata, dtype=torch.int64, device=pack.device)
    grid = (num_segments, triton.cdiv(n * max_row_nbytes, 1024))
    _copy_mla_rows_into_pack_kernel[grid](
        src_metadata,
        row_indices,
        pack,
        n,
        BLOCK_SIZE=1024,
    )


def copy_mla_rows_into_pack(
    kv_data_ptrs: Sequence[int],
    row_indices: torch.Tensor,
    pack: torch.Tensor,
    token_item_lens: Sequence[int],
    src_token_item_lens: Optional[Sequence[int]] = None,
) -> None:
    if src_token_item_lens is None:
        src_token_item_lens = token_item_lens
    if not (len(kv_data_ptrs) == len(token_item_lens) == len(src_token_item_lens)):
        raise ValueError(
            "KV pointers, copy widths, and source strides length mismatch: "
            f"{len(kv_data_ptrs)}, {len(token_item_lens)}, {len(src_token_item_lens)}"
        )
    if not kv_data_ptrs:
        return

    n = int(row_indices.numel())
    metadata = []
    offset = 0
    for ptr, item_len, src_item_len in zip(
        kv_data_ptrs, token_item_lens, src_token_item_lens
    ):
        item_len = int(item_len)
        if item_len <= 0:
            raise ValueError(f"MLA token item length must be positive, got {item_len}")
        # Each entry is a dense [n, item_len] block in the pack.
        metadata.extend((int(ptr), item_len, offset, int(src_item_len), item_len))
        offset += n * item_len

    max_item_len = max(int(item_len) for item_len in token_item_lens)
    _launch_copy_rows(metadata, row_indices, pack, max_item_len)


def copy_rows_into_strided_pack(
    segments: Sequence[Tuple[int, int, int, int, int]],
    row_indices: torch.Tensor,
    pack: torch.Tensor,
) -> None:
    """Copy ``row_nbytes`` of source row ``row_indices[r]`` to
    ``pack + pack_offset + r * pack_row_stride`` for every segment.

    Each segment is ``(src_ptr, row_nbytes, pack_offset, src_row_stride,
    pack_row_stride)`` in bytes; ``src_ptr`` may be a CUDA IPC peer pointer.
    Segments sharing one ``pack_row_stride`` and disjoint ``pack_offset``
    byte ranges interleave several sources into one packed row.
    """
    if not segments:
        return
    metadata = []
    for src_ptr, row_nbytes, pack_offset, src_row_stride, pack_row_stride in segments:
        if row_nbytes <= 0 or pack_row_stride < row_nbytes:
            raise ValueError(
                "Strided pack segment needs 0 < row_nbytes <= pack_row_stride, "
                f"got row_nbytes={row_nbytes}, pack_row_stride={pack_row_stride}"
            )
        metadata.extend(
            (
                int(src_ptr),
                int(row_nbytes),
                int(pack_offset),
                int(src_row_stride),
                int(pack_row_stride),
            )
        )
    max_row_nbytes = max(int(segment[1]) for segment in segments)
    _launch_copy_rows(metadata, row_indices, pack, max_row_nbytes)
