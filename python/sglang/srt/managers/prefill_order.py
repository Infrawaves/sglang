"""Ordering for the ``shortest-prefill-first`` schedule policy.

Ranks queued prefill requests by the prefill work they still need, with
token-based aging so no request is deferred forever. The same key orders
both the waiting queue (``SchedulePolicy.calc_priority``) and, under PD
disaggregation, the bootstrap queue (which request gets a freed metadata
buffer first, ``PrefillBootstrapQueue.pop_bootstrapped``).

Work counts every cache tier: device and host hits from the radix tree, plus
the prefix a finished L3 (storage) prefetch made host-resident. Ignoring the
L3 part would rank a request whose KV is already loaded as cold, push it to
the back, and let host-LRU evict the prefetched span before it runs.

Every input is identical on all TP ranks (radix-tree matches, the rank-agreed
prefetch result, and the scheduler's prefill-token counter), so every rank
computes the same order. Wall-clock time is never used.

This module is pure Python on purpose: no torch, no sglang imports.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple


def total_prefill_len(req) -> int:
    return len(req.origin_input_ids) + len(req.output_ids)


def expected_cached_len(req, prefetched_prefix_len: int = 0) -> int:
    """Tokens of ``req`` expected to be served from cache at admission."""
    matched = len(req.prefix_indices) + req.host_hit_length
    return min(total_prefill_len(req), max(matched, prefetched_prefix_len))


def prefill_work(req, prefetched_prefix_len: int = 0) -> int:
    """Tokens that still have to be prefilled (at least 1)."""
    return max(
        1, total_prefill_len(req) - expected_cached_len(req, prefetched_prefix_len)
    )


def waited_tokens(req, processed_tokens: int) -> int:
    """Prefill tokens the scheduler processed since ``req`` entered the queue.

    Counts from the first queue entry (the bootstrap queue under PD prefill),
    falling back to the waiting-queue entry snapshot used by HRRN.
    """
    arrival = getattr(req, "prefill_arrival_processed_tokens", None)
    if arrival is None:
        arrival = req.arrival_processed_tokens
    return max(0, processed_tokens - arrival)


def shortest_prefill_key(
    req,
    *,
    processed_tokens: int,
    aging_tokens: int,
    prefetched_prefix_len: int = 0,
    deprioritized: bool = False,
) -> Tuple:
    """Sort key: smaller sorts first.

    Requests that waited ``aging_tokens`` or more go first, longest-waiting
    first. Everything else goes shortest remaining work first; requests
    deferred for in-batch prefix sharing go after their tier; ties go to the
    request that waited longer. ``aging_tokens <= 0`` disables aging.
    """
    waited = waited_tokens(req, processed_tokens)
    if aging_tokens > 0 and waited >= aging_tokens:
        return (0, -waited)
    return (1, deprioritized, prefill_work(req, prefetched_prefix_len), -waited)


def sort_shortest_prefill_first(
    reqs: List,
    *,
    processed_tokens: int,
    aging_tokens: int,
    prefetched_prefix_len: Callable[[str], int],
    deprioritized: Optional[set] = None,
) -> None:
    """Stable in-place sort of ``reqs`` by :func:`shortest_prefill_key`."""
    deprioritized = deprioritized or set()
    reqs.sort(
        key=lambda r: shortest_prefill_key(
            r,
            processed_tokens=processed_tokens,
            aging_tokens=aging_tokens,
            prefetched_prefix_len=prefetched_prefix_len(r.rid),
            deprioritized=r.rid in deprioritized,
        )
    )


def shortest_prefill_order(
    reqs: Sequence,
    *,
    processed_tokens: int,
    aging_tokens: int,
    prefetched_prefix_len: Callable[[str], int],
) -> List[int]:
    """Indices of ``reqs`` in shortest-prefill-first order (stable)."""
    keys = [
        shortest_prefill_key(
            r,
            processed_tokens=processed_tokens,
            aging_tokens=aging_tokens,
            prefetched_prefix_len=prefetched_prefix_len(r.rid),
        )
        for r in reqs
    ]
    return sorted(range(len(reqs)), key=keys.__getitem__)
