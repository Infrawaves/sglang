from typing import Optional, Sequence

import torch
import triton
import triton.language as tl


@triton.jit
def _copy_mla_rows_into_pack_kernel(
    src_metadata,
    row_indices,
    pack,
    num_rows,
    ROW_BYTES: tl.constexpr,
    COPY_WORDS: tl.constexpr,
    ALIGN_BYTES: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
    start_token=0,
    PAGE_SIZE: tl.constexpr = 0,
    DCP_SIZE: tl.constexpr = 8,
):
    layer_id = tl.program_id(0)
    metadata_offset = layer_id * 3
    element_type: tl.constexpr = tl.uint32 if COPY_WORDS else tl.uint8
    element_bytes: tl.constexpr = 4 if COPY_WORDS else 1
    src_address = tl.multiple_of(tl.load(src_metadata + metadata_offset), ALIGN_BYTES)
    src = src_address.to(tl.pointer_type(element_type))
    if ROW_BYTES:
        row_width = ROW_BYTES // element_bytes
    else:
        row_nbytes = tl.load(src_metadata + metadata_offset + 1)
        row_width = tl.multiple_of(
            row_nbytes // element_bytes, ALIGN_BYTES // element_bytes
        )
    pack_offset = tl.load(src_metadata + metadata_offset + 2)
    if PAGE_SIZE:
        # Paged metadata is static: the third field is a prefix sum of row
        # widths, not a window-dependent byte offset.
        pack_offset *= num_rows
    pack_offset = tl.multiple_of(pack_offset, ALIGN_BYTES)
    dst = (pack + pack_offset).to(tl.pointer_type(element_type))

    # Load one logical index per row, then broadcast it along contiguous
    # columns. Grouping rows also amortizes metadata loads and CTA scheduling.
    local_rows = tl.program_id(1) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    if PAGE_SIZE:
        rank = tl.program_id(2) % DCP_SIZE
        column_block = tl.program_id(2) // DCP_SIZE
        first = (rank - start_token % DCP_SIZE + DCP_SIZE) % DCP_SIZE
        quotient = num_rows // DCP_SIZE
        remainder = num_rows % DCP_SIZE
        count = quotient + (first < remainder).to(tl.int32)
        # Rank-major packed layout; the extra rows form a cyclic interval
        # starting at start_token % DCP_SIZE. This also handles short tails.
        rank_start = start_token % DCP_SIZE
        extra_before = tl.minimum(tl.maximum(rank - rank_start, 0), remainder)
        extra_before += tl.minimum(
            rank, tl.maximum(rank_start + remainder - DCP_SIZE, 0)
        )
        rows = rank * quotient + extra_before + local_rows
        row_mask = local_rows < count
        logical = first + local_rows * DCP_SIZE
        page = tl.load(row_indices + logical // PAGE_SIZE, mask=row_mask, other=0)
        src_rows = page.to(tl.int64) * PAGE_SIZE + logical % PAGE_SIZE
    else:
        column_block = tl.program_id(2)
        rows = local_rows
        row_mask = rows < num_rows
        src_rows = tl.load(row_indices + rows, mask=row_mask, other=0).to(tl.int64)
    cols = column_block * BLOCK_COLS + tl.arange(0, BLOCK_COLS)
    mask = row_mask[:, None] & (cols[None, :] < row_width)
    values = tl.load(
        src + src_rows[:, None] * row_width + cols[None, :], mask=mask, other=0
    )
    tl.store(
        dst + rows[:, None].to(tl.int64) * row_width + cols[None, :],
        values,
        mask=mask,
    )


class PagedMLAGather:
    """Reusable DCP gather with GPU address calculation from physical pages.

    Construct on the stream used for gathers: static layer metadata is uploaded
    once. Source pages describe this window (starting on a physical page), while
    start_token is its position in the request. The result is layer-major, then
    DCP rank-major, exactly like concatenating the old per-rank token plans.
    """

    def __init__(self, kv_data_ptrs, token_item_lens, device, page_size, dcp_size=8):
        if page_size <= 0 or dcp_size <= 0:
            raise ValueError("Page size and DCP size must be positive")
        self.widths = tuple(int(width) for width in token_item_lens)
        self.ptrs = tuple(int(ptr) for ptr in kv_data_ptrs)
        if not self.widths or len(self.ptrs) != len(self.widths):
            raise ValueError("Paged MLA gather requires matching, nonempty layers")
        if any(width <= 0 for width in self.widths):
            raise ValueError("MLA token item lengths must be positive")
        self.page_size = page_size
        self.dcp_size = dcp_size
        self.total_row_bytes = sum(self.widths)
        self.row_bytes = self.widths[0] if len(set(self.widths)) == 1 else 0
        self.align_bytes = 1
        for alignment in (16, 4):
            if all(ptr % alignment == 0 for ptr in self.ptrs) and all(
                width % alignment == 0 for width in self.widths
            ):
                self.align_bytes = alignment
                break
        metadata = []
        self.layer_offsets = []
        offset = 0
        for ptr, width in zip(self.ptrs, self.widths):
            metadata.append((ptr, width, offset))
            self.layer_offsets.append(offset)
            offset += width
        self.metadata = torch.tensor(metadata, dtype=torch.int64, device=device)
        self.launch_shapes = {}
        for alignment in (1, 4, 16):
            max_cols = max(self.widths) // (4 if alignment >= 4 else 1)
            cols = min(256, triton.next_power_of_2(max(1, max_cols)))
            self.launch_shapes[alignment] = (cols, triton.cdiv(max_cols, cols))

    def __call__(self, pages, num_rows, pack, start_token=0):
        if num_rows < 0 or start_token < 0 or start_token % self.page_size:
            raise ValueError("Paged gather requires nonnegative, page-aligned input")
        if num_rows == 0:
            return
        if pages.numel() * self.page_size < num_rows:
            raise ValueError("Insufficient source pages")
        if pages.dtype not in (torch.int32, torch.int64) or not pages.is_contiguous():
            raise ValueError("Source pages must be contiguous int32/int64")
        if pack.dtype != torch.uint8 or not pack.is_contiguous():
            raise ValueError("Paged gather requires a contiguous uint8 pack")
        if pack.numel() < num_rows * self.total_row_bytes:
            raise ValueError("Paged gather pack buffer is too small")
        alignment = self.align_bytes
        if pack.data_ptr() % alignment:
            alignment = 4 if alignment >= 4 and pack.data_ptr() % 4 == 0 else 1
        copy_words = alignment >= 4
        cols, column_blocks = self.launch_shapes[alignment]
        grid = (
            len(self.widths),
            triton.cdiv(triton.cdiv(num_rows, self.dcp_size), 16),
            self.dcp_size * column_blocks,
        )
        _copy_mla_rows_into_pack_kernel[grid](
            self.metadata,
            pages,
            pack,
            num_rows,
            ROW_BYTES=self.row_bytes,
            COPY_WORDS=copy_words,
            ALIGN_BYTES=alignment,
            BLOCK_ROWS=16,
            BLOCK_COLS=cols,
            start_token=start_token,
            PAGE_SIZE=self.page_size,
            DCP_SIZE=self.dcp_size,
            num_warps=4,
        )


def copy_mla_rows_into_pack(
    kv_data_ptrs: Sequence[int],
    row_indices: torch.Tensor,
    pack: torch.Tensor,
    token_item_lens: Sequence[int],
    *,
    src_metadata: Optional[torch.Tensor] = None,
) -> None:
    """Copy arbitrary source rows into a dense, layer-major byte buffer.

    ``src_metadata`` contains (source pointer, row bytes, pack byte offset)
    for each item, consistent with ``kv_data_ptrs`` and ``token_item_lens``.
    Providing it avoids a host-to-device allocation on the window hot path.
    Aligned items are copied as uint32 bit patterns, with no dtype conversion;
    odd widths or unaligned buffers use the same row tiling with byte loads.
    """
    if len(kv_data_ptrs) != len(token_item_lens):
        raise ValueError(
            "kv_data_ptrs and token_item_lens length mismatch: "
            f"{len(kv_data_ptrs)} vs {len(token_item_lens)}"
        )
    if not kv_data_ptrs:
        return

    n = int(row_indices.numel())
    if n == 0:
        return
    widths = tuple(int(item_len) for item_len in token_item_lens)
    if any(width <= 0 for width in widths):
        raise ValueError(f"MLA token item lengths must be positive, got {widths}")
    if pack.dtype != torch.uint8 or not pack.is_contiguous():
        raise ValueError("PD DCP gather requires a contiguous uint8 pack buffer")
    if pack.numel() < n * sum(widths):
        raise ValueError("PD DCP gather pack buffer is too small")
    row_indices = row_indices.contiguous()

    if src_metadata is None:
        metadata = []
        offset = 0
        for ptr, width in zip(kv_data_ptrs, widths):
            metadata.extend((int(ptr), width, offset))
            offset += n * width
        src_metadata = torch.tensor(metadata, dtype=torch.int64, device=pack.device)
    elif src_metadata.numel() != 3 * len(widths):
        raise ValueError("PD DCP gather metadata size mismatch")
    copy_words = (
        pack.data_ptr() % 4 == 0
        and all(int(ptr) % 4 == 0 for ptr in kv_data_ptrs)
        and all(width % 4 == 0 for width in widths)
    )
    # Metadata contains integer addresses, so their alignment is otherwise
    # invisible to Triton. Only promise 16-byte alignment when every layer's
    # source, row stride and packed destination satisfy it (including tails).
    align_bytes = 4 if copy_words else 1
    if (
        copy_words
        and pack.data_ptr() % 16 == 0
        and all(int(ptr) % 16 == 0 for ptr in kv_data_ptrs)
        and all(width % 16 == 0 for width in widths)
    ):
        align_bytes = 16
    max_cols = max(widths) // (4 if copy_words else 1)
    block_cols = min(256, triton.next_power_of_2(max_cols))
    block_rows = 16
    # Specialize the common homogeneous MLA layout (e.g. 576 B / item),
    # while preserving mixed widths in a single launch.
    row_bytes = widths[0] if all(width == widths[0] for width in widths) else 0
    grid = (
        len(widths),
        triton.cdiv(n, block_rows),
        triton.cdiv(max_cols, block_cols),
    )
    _copy_mla_rows_into_pack_kernel[grid](
        src_metadata,
        row_indices,
        pack,
        n,
        ROW_BYTES=row_bytes,
        COPY_WORDS=copy_words,
        ALIGN_BYTES=align_bytes,
        BLOCK_ROWS=block_rows,
        BLOCK_COLS=block_cols,
        num_warps=4,
    )
