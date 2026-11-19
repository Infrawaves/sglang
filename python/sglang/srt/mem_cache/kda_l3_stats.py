"""Opt-in L3 KDA (Mamba) state provenance logging.

Three log lines, joined offline on the page hash (``key``) and the request id:

  KDA_L3_WRITE  a Mamba state's L3 write, logged at the backup ACK with its
                class and whether this instance's put stored it
                (``stored=new``) or found it already present (``exists``).
  KDA_L3_READ   the rank-agreed L3 existence verdict of a prefetch: the hit,
                the contiguous KV prefix and each pool's own stop point.
  KDA_L3_LOAD   what the prefetch actually inserted into the host tree. A
                verdict can still be revoked, capped or discarded, so classes
                are attributed from this line, not from KDA_L3_READ.

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
import threading
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
    """Carries a state's class from its tree insert to its L3 write ACK.

    The L3 write is issued at the write-through ACK, after the insert returned,
    and sees only the node id; its own ACK sees only the operation id. The node that holds a Mamba state keeps its id across
    splits (a split creates the new node for the prefix), so the id is a stable
    handoff key; the recorded depth guards against a reused id.
    """

    def __init__(self) -> None:
        self._pending: OrderedDict[Any, tuple] = OrderedDict()
        # Writes issued to storage, keyed by backup operation id, until the ACK.
        self._issued: OrderedDict[Any, tuple] = OrderedDict()
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

    def wants_write(self, node_id: Any) -> bool:
        """Cheap gate before the caller computes the node depth: only the
        logging rank, and only nodes whose insert recorded a class."""
        return self._log and node_id in self._pending

    def on_storage_write(
        self,
        node_id: Any,
        mamba_key: Optional[str],
        node_end: int,
        operation_id: Any,
    ) -> None:
        """The L3 write was issued; hold its class until the backup ACK."""
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
        self._issued[operation_id] = (mamba_key, cls, rid, position, prompt_len)
        while len(self._issued) > _MAX_PENDING:
            self._issued.popitem(last=False)

    def on_backup_ack(self, operation_id: Any) -> None:
        """The backup finished: log the write with this instance's put result."""
        if not self._log:
            return
        entry = self._issued.pop(operation_id, None)
        if entry is None:
            return
        mamba_key, cls, rid, position, prompt_len = entry
        logger.info(
            "KDA_L3_WRITE key=%s cls=%s pos=%d prompt_len=%d stored=%s rid=%s",
            mamba_key,
            cls,
            position,
            prompt_len,
            pop_put_outcome(mamba_key),
            rid,
        )


# Put outcome of each Mamba state, recorded by the storage backend's write
# thread and read at the backup ACK on the scheduler thread.
_put_outcomes: OrderedDict[str, str] = OrderedDict()
_put_outcomes_lock = threading.Lock()


def put_outcome_recording_active() -> bool:
    return enabled() and _is_logging_rank()


def record_put_outcome(key: str, outcome: str) -> None:
    """``outcome``: new (this put stored it), exists (already in the store,
    nothing written), partial (some of its objects were already there),
    failed (a put returned an error)."""
    with _put_outcomes_lock:
        _put_outcomes[key] = outcome
        _put_outcomes.move_to_end(key)
        while len(_put_outcomes) > _MAX_PENDING:
            _put_outcomes.popitem(last=False)


def pop_put_outcome(key: str) -> str:
    # unknown: the backend does not report outcomes, or the entry was evicted.
    with _put_outcomes_lock:
        return _put_outcomes.pop(key, "unknown")


def log_hit_query(operation: Any, final_hit_tokens: int, page_size: int) -> None:
    """Log one rank-agreed L3 existence verdict (attention TP rank 0 only).

    ``mamba_end`` / ``indexer_end`` are each pool's own stop point. A pool that
    is not part of the query prints ``na``. A queried pool without a verdict
    prints ``?``: Mooncake stops checking pools once no stop point is left and
    reports only nonzero boundaries, and the cross-rank MIN drops a pool any
    rank left out, so a missing verdict does not mean the pool stopped at 0.
    ``limited_by`` is built only from pools with a verdict; ``unknown`` means a
    truncated hit whose cause lies in a pool without one.
    """
    if not _is_logging_rank():
        return
    try:
        from sglang.srt.mem_cache.hicache_storage import PoolName

        assume_stored = bool(getattr(operation, "assume_stored", False))
        pool_hits = getattr(operation, "query_pool_hit_pages", None) or {}
        final_pages = final_hit_tokens // page_size
        # assume_stored skips the existence query, so no pool has a verdict.
        kv_pages = (
            final_pages if assume_stored else int(pool_hits.get(PoolName.KV, 0) or 0)
        )
        if kv_pages <= 0 and final_pages <= 0:
            return
        hashes = getattr(operation, "all_hash_values", None) or []
        key = hashes[final_pages - 1] if 0 < final_pages <= len(hashes) else "-"
        start = int(getattr(operation, "start_page", 0) or 0) * page_size
        kv_end = start + kv_pages * page_size
        queried = {t.name for t in getattr(operation, "pool_transfers", None) or []}

        def pool_end(pool) -> str:
            if assume_stored or pool not in queried:
                return "na"
            if pool not in pool_hits:
                return "?"
            return str(start + int(pool_hits[pool]) * page_size)

        ends = {
            "mamba": pool_end(PoolName.MAMBA),
            "indexer": pool_end(PoolName.INDEXER),
        }
        limited = final_pages < kv_pages
        if limited:
            cut = [n for n, e in ends.items() if e.isdigit() and int(e) < kv_end]
            if cut:
                limited_by = "+".join(cut)
            elif "?" in ends.values():
                limited_by = "unknown"
            else:
                limited_by = "other"
        else:
            limited_by = "-"
        logger.info(
            "KDA_L3_READ key=%s start=%d hit_end=%d kv_end=%d mamba_end=%s "
            "indexer_end=%s limited=%d limited_by=%s assume_stored=%d rid=%s",
            key,
            start,
            start + final_pages * page_size,
            kv_end,
            ends["mamba"],
            ends["indexer"],
            int(limited),
            limited_by,
            int(assume_stored),
            getattr(operation, "request_id", None),
        )
    except Exception:
        logger.debug("KDA_L3_READ logging failed", exc_info=True)


def log_load(operation: Any, rid: Any, page_size: int, dropped: bool) -> None:
    """Log what a prefetch actually inserted into the host tree (TP rank 0).

    ``loaded_end`` is where the restored KDA state sits; ``key`` is that page's
    hash, the join key to KDA_L3_WRITE. A revoked or discarded prefetch never
    reaches here; a dropped host insert is logged with ``result=dropped``.
    """
    if not _is_logging_rank():
        return
    try:
        completed = int(getattr(operation, "completed_tokens", 0) or 0)
        if completed <= 0:
            return
        # Trimming a device-resident head moves storage_start and the hash list
        # together, so both stay aligned with completed_tokens.
        start = int(getattr(operation, "storage_start", 0) or 0)
        hashes = getattr(operation, "hash_value", None) or []
        pages = completed // page_size
        key = hashes[pages - 1] if 0 < pages <= len(hashes) else "-"
        logger.info(
            "KDA_L3_LOAD key=%s start=%d loaded_end=%d result=%s rid=%s",
            key,
            start,
            start + completed,
            "dropped" if dropped else "success",
            rid,
        )
    except Exception:
        logger.debug("KDA_L3_LOAD logging failed", exc_info=True)
