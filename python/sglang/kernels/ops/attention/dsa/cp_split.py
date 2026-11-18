"""Interleave CP q-sequence split kernel for DSA prefill.

Migrated from ``sglang.srt.layers.attention.dsa.utils`` (RFC #29630, Phase 2.5).
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _dsa_cp_interleave_q_seqs_kernel(
    in_seqs_ptr,
    out_seqs_ptr,
    bs_idx_ptr,
    tokens: tl.constexpr,
    cp_size: tl.constexpr,
    cp_rank: tl.constexpr,
):
    # Eager metadata uses int32, while prefill graph buffers may use int64.
    extra_seq = tl.full((), 0, in_seqs_ptr.dtype.element_ty)
    bs_idx = 0
    for bs in range(tokens):
        # seq_lens may be int64 under graph capture
        cur_len = tl.load(in_seqs_ptr + bs).to(tl.int32)
        cur_len += extra_seq
        cur_seq = cur_len // cp_size + (cur_len % cp_size > cp_rank)
        if cur_seq > 0:
            tl.store(bs_idx_ptr + bs_idx, bs)
            tl.store(out_seqs_ptr + bs_idx, cur_seq)
            bs_idx += 1
        extra_seq = cur_len - cur_seq * cp_size


def dsa_cp_interleave_q_seqs(
    extend_seqs: torch.Tensor, num_rank_seqs: int, cp_size: int, cp_rank: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """This rank's interleaved share of each request's extend length, as
    ``(q_lens, bs_idx)`` over the ``num_rank_seqs`` requests left non-empty."""
    q_lens = torch.empty(
        (num_rank_seqs,), device=extend_seqs.device, dtype=extend_seqs.dtype
    )
    bs_idx = torch.empty((num_rank_seqs,), device=extend_seqs.device, dtype=torch.int32)
    _dsa_cp_interleave_q_seqs_kernel[(1,)](
        extend_seqs, q_lens, bs_idx, len(extend_seqs), cp_size, cp_rank
    )
    return q_lens, bs_idx


@triton.jit
def _dsa_cp_owner_q_seqs_kernel(
    in_seqs_ptr,
    prefix_seqs_ptr,
    out_seqs_ptr,
    bs_idx_ptr,
    tokens: tl.constexpr,
    cp_size: tl.constexpr,
    cp_rank: tl.constexpr,
):
    bs_idx = 0
    for bs in range(tokens):
        p = tl.load(prefix_seqs_ptr + bs).to(tl.int32)
        e = tl.load(in_seqs_ptr + bs).to(tl.int32)
        # Positions p .. p+e-1 with pos % cp_size == cp_rank. Triton integer
        # division truncates toward zero, so offset both numerators by cp_size
        # to keep them non-negative (p - 1 - cp_rank >= -cp_size).
        cur_seq = (p + e - 1 - cp_rank + cp_size) // cp_size - (
            p - 1 - cp_rank + cp_size
        ) // cp_size
        if cur_seq > 0:
            tl.store(bs_idx_ptr + bs_idx, bs)
            tl.store(out_seqs_ptr + bs_idx, cur_seq)
            bs_idx += 1


def dsa_cp_owner_q_seqs(
    extend_seqs: torch.Tensor,
    prefix_seqs: torch.Tensor,
    num_rank_seqs: int,
    cp_size: int,
    cp_rank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """This rank's share of each request's extend tokens when CP shards by
    sequence position (pos % cp_size == cp_rank, the DCP KV owner), as
    ``(q_lens, bs_idx)`` over the ``num_rank_seqs`` requests left non-empty."""
    q_lens = torch.empty(
        (num_rank_seqs,), device=extend_seqs.device, dtype=extend_seqs.dtype
    )
    bs_idx = torch.empty((num_rank_seqs,), device=extend_seqs.device, dtype=torch.int32)
    _dsa_cp_owner_q_seqs_kernel[(1,)](
        extend_seqs, prefix_seqs, q_lens, bs_idx, len(extend_seqs), cp_size, cp_rank
    )
    return q_lens, bs_idx
