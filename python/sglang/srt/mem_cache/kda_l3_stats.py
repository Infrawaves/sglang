"""Opt-in L3 KDA (Mamba) state provenance logging.

Classifies every Mamba state a prefill publishes to L3 by where it sits in the
writer's prompt, and logs every L3 hit query with the page whose state it
resolved to. Offline joining the two streams on the page hash attributes each
L3 hit to the class of the state it used.

Classes, in precedence order:
  tail    the request-end state (PD prefill: the completed prompt)
  branch  a state taken at the request's branching point
  head    the state at the end of the first ``SGLANG_KDA_L3_STATS_HEAD_TOKENS``
          tokens of the prompt (absolute position <= the threshold)
  middle  every other chunk-boundary state

Enabled with ``SGLANG_KDA_L3_STATS=1``; off by default. Only attention TP rank 0
logs, and nothing here issues RPCs, collectives or device work.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Any, Optional

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

# Bounds the insert->write handoff map; a node whose write never happens
# (evicted before its write-through ACK) must not grow it forever.
_MAX_PENDING = 65536


def enabled() -> bool:
    return envs.SGLANG_KDA_L3_STATS.get()


def _is_logging_rank() -> bool:
    try:
        from sglang.srt.runtime_context import get_parallel

        return get_parallel().attn_tp_rank == 0
    except Exception:
        return False


def classify(
    position: int, is_request_end: bool, branching_seqlen: Optional[int]
) -> str:
    if is_request_end:
        return "tail"
    if branching_seqlen is not None and position == branching_seqlen:
        return "branch"
    if position <= envs.SGLANG_KDA_L3_STATS_HEAD_TOKENS.get():
        return "head"
    return "middle"


class KdaL3StatsTracker:
    """Carries a state's class from its tree insert to its L3 write.

    The L3 write runs at the write-through ACK, after the insert returned, and
    sees only the node id. The node that holds a Mamba state keeps its id across
    splits (a split creates the new node for the prefix), so the id is a stable
    handoff key; the recorded depth guards against a reused id.
    """

    def __init__(self) -> None:
        self._pending: OrderedDict[Any, tuple] = OrderedDict()
        self._log = _is_logging_rank()

    def on_insert(
        self,
        node_id: Any,
        *,
        rid: str,
        position: int,
        prompt_len: int,
        is_request_end: bool,
        branching_seqlen: Optional[int],
    ) -> None:
        if not self._log or node_id is None or position <= 0:
            return
        cls = classify(position, is_request_end, branching_seqlen)
        self._pending[node_id] = (cls, rid, position, prompt_len)
        self._pending.move_to_end(node_id)
        while len(self._pending) > _MAX_PENDING:
            self._pending.popitem(last=False)

    def on_storage_write(
        self, node_id: Any, mamba_key: Optional[str], node_end: int
    ) -> None:
        if not self._log:
            return
        entry = self._pending.pop(node_id, None)
        if entry is None or mamba_key is None:
            return
        cls, rid, position, prompt_len = entry
        if position != node_end:
            # The id was reused by another node, or the node no longer ends
            # at the state this request inserted.
            return
        logger.info(
            "KDA_L3_WRITE key=%s cls=%s pos=%d prompt_len=%d rid=%s",
            mamba_key,
            cls,
            position,
            prompt_len,
            rid,
        )


def log_hit_query(operation: Any, final_hit_tokens: int, page_size: int) -> None:
    """Log one rank-agreed L3 hit query (attention TP rank 0 only)."""
    if not _is_logging_rank():
        return
    try:
        from sglang.srt.mem_cache.hicache_storage import PoolName

        pool_hits = getattr(operation, "query_pool_hit_pages", None) or {}
        kv_pages = int(pool_hits.get(PoolName.KV, 0) or 0)
        final_pages = final_hit_tokens // page_size
        if kv_pages <= 0 and final_pages <= 0:
            return
        hashes = getattr(operation, "all_hash_values", None) or []
        key = hashes[final_pages - 1] if 0 < final_pages <= len(hashes) else "-"
        start = int(getattr(operation, "start_page", 0) or 0) * page_size
        logger.info(
            "KDA_L3_READ key=%s start=%d hit_end=%d kv_end=%d limited=%d "
            "assume_stored=%d rid=%s",
            key,
            start,
            start + final_pages * page_size,
            start + kv_pages * page_size,
            int(final_pages < kv_pages),
            int(bool(getattr(operation, "assume_stored", False))),
            getattr(operation, "request_id", None),
        )
    except Exception:
        logger.debug("KDA_L3_READ logging failed", exc_info=True)
