"""Decode-side HiCache storage write in the prefill server's parallel layout.

A decode server writes finished KV and Mamba state so that the next prefill round
reads them exactly like objects it wrote itself: 64-token storage pages, one
sub-page per DCP rank, and Mamba state resliced to the prefill attention TP. The
prefill topology is frozen at the first PD handshake; the backup thread only
looks the resulting plan up. Any unsupported or changed topology disables the
storage write for the whole process, never the decode service.
"""

from __future__ import annotations

import logging
import threading
from typing import Optional, Sequence, Union

import msgspec

logger = logging.getLogger(__name__)


class StorageWriteTarget(msgspec.Struct, frozen=True):
    """Prefill topology the storage objects must match."""

    prefill_attn_tp_size: int
    prefill_page_size: int
    # Prefill attention-TP ranks whose Mamba shards this decode rank writes.
    target_tp_ranks: tuple[int, ...]


class DecodeStorageAnchor(msgspec.Struct, frozen=True):
    """Where a radix node's token span sits in its request's hash chain."""

    # Token ids of every ancestor node from the root down, excluding the node.
    prefix_chunks: tuple[Sequence[int], ...]
    # storage_namespace_seed of the request; None for unnamespaced keys.
    seed: Optional[str]


class MambaStateMeta(msgspec.Struct, frozen=True):
    """One Mamba state tensor as one decode rank holds it, per layer and slot."""

    key_suffix: str
    item_bytes: int
    # Sliceable dim on this rank; 0 marks a tensor replicated across attention TP.
    slice_dim: int
    outer_count: int
    conv_shard_groups: Optional[tuple[int, ...]]


class MambaRankBlocks(msgspec.Struct, frozen=True):
    prefill_tp_rank: int
    # Per tensor: (src_offset, num_bytes) inside one layer item, in prefill order.
    per_tensor: tuple[tuple[tuple[int, int], ...], ...]


class StorageWritePlan(msgspec.Struct, frozen=True):
    sub_page_size: int
    dcp_size: int
    dcp_rank: int
    writes_kv: bool
    mamba_tensors: tuple[MambaStateMeta, ...]
    mamba_ranks: tuple[MambaRankBlocks, ...]


def resolve_target(
    *,
    prefill_attn_tp_size: int,
    prefill_pp_size: int,
    prefill_page_size: Optional[int],
    target_tp_ranks: Optional[Sequence[int]],
    decode_attn_tp_size: int,
) -> Union[StorageWriteTarget, str]:
    """Return the target, or the reason this prefill topology cannot be served."""
    if prefill_pp_size != 1:
        return "prefill_pp"
    if prefill_page_size is None or prefill_page_size <= 0:
        return "prefill_page_size"
    # A prefill rank's Mamba object would need shards from several decode ranks.
    if prefill_attn_tp_size % decode_attn_tp_size != 0:
        return "attn_tp_ratio"
    if not target_tp_ranks:
        return "target_tp_ranks"
    return StorageWriteTarget(
        prefill_attn_tp_size=prefill_attn_tp_size,
        prefill_page_size=prefill_page_size,
        target_tp_ranks=tuple(target_tp_ranks),
    )


class DecodeStorageWriteState:
    """Frozen target plus a one-way disable switch, shared across threads."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._target: Optional[StorageWriteTarget] = None
        self._disabled_reason: Optional[str] = None
        self._disabled_gauge = None

    def offer(self, target: Union[StorageWriteTarget, str]) -> None:
        with self._lock:
            if self._disabled_reason is not None:
                return
            if isinstance(target, str):
                self._disable_locked(target)
            elif self._target is None:
                self._target = target
                logger.info("Decode storage write target frozen: %s", target)
            elif self._target != target:
                logger.error(
                    "Prefill topology %s differs from frozen %s", target, self._target
                )
                self._disable_locked("topology_mismatch")

    def disable(self, reason: str) -> None:
        with self._lock:
            if self._disabled_reason is None:
                self._disable_locked(reason)

    def target(self) -> Optional[StorageWriteTarget]:
        """The frozen target; None before the first handshake or once disabled."""
        with self._lock:
            return None if self._disabled_reason is not None else self._target

    @property
    def disabled_reason(self) -> Optional[str]:
        return self._disabled_reason

    def _disable_locked(self, reason: str) -> None:
        self._disabled_reason = reason
        logger.error(
            "Decode storage write disabled (reason=%s); L2 keeps working.", reason
        )
        self._publish_disabled(reason)

    def _publish_disabled(self, reason: str) -> None:
        try:
            from prometheus_client import Gauge
        except ImportError:
            return
        try:
            if self._disabled_gauge is None:
                self._disabled_gauge = Gauge(
                    "sglang:decode_storage_write_disabled",
                    "1 once decode storage write is disabled, labelled by reason.",
                    labelnames=["reason"],
                    multiprocess_mode="mostrecent",
                )
            self._disabled_gauge.labels(reason=reason).set(1)
        except Exception:
            logger.exception(
                "Failed to publish the decode storage write disabled metric"
            )


_STATE = DecodeStorageWriteState()


def get_decode_storage_write_state() -> DecodeStorageWriteState:
    return _STATE


def offer_prefill_info(*, info, decode_attn_tp_size: int) -> None:
    """Freeze or check a prefill topology; never raises into the PD handshake."""
    try:
        target = resolve_target(
            prefill_attn_tp_size=info.attn_tp_size,
            prefill_pp_size=info.pp_size,
            prefill_page_size=info.page_size,
            target_tp_ranks=info.target_tp_ranks,
            decode_attn_tp_size=decode_attn_tp_size,
        )
    except Exception:
        logger.exception("Failed to resolve the decode storage write target")
        target = "freeze_error"
    _STATE.offer(target)


def sub_page_hashes(
    *,
    prefix_chunks: Sequence[Sequence[int]],
    token_ids: Sequence[int],
    seed: Optional[str],
    sub_page_size: int,
) -> list[str]:
    """Hash ``token_ids`` in prefill pages, chained after all its ancestors.

    Prefill seeds the chain with the request namespace and links every page to
    the previous one, so the result does not depend on how the radix tree split
    the path into nodes.
    """
    from sglang.srt.mem_cache.utils import get_hash_str

    prior = seed
    for chunk in prefix_chunks:
        hashes = get_hash_str(list(chunk), prior, page_size=sub_page_size)
        if hashes:
            prior = hashes[-1]
    return get_hash_str(list(token_ids), prior, page_size=sub_page_size)


def local_sub_page_keys(
    sub_hashes: Sequence[str], *, dcp_size: int, dcp_rank: int
) -> list[str]:
    """Keys of the sub-pages this DCP rank owns, in its physical row order.

    Page layout stripes logical pages starting at rank zero, so rank r holds
    sub-page ``dcp_size * p + r`` of every tree page p.
    """
    return list(sub_hashes[dcp_rank::dcp_size])


def mamba_rank_blocks(
    meta: MambaStateMeta,
    *,
    decode_attn_tp_size: int,
    decode_attn_tp_rank: int,
    prefill_attn_tp_size: int,
    prefill_tp_rank: int,
) -> tuple[tuple[int, int], ...]:
    """(src_offset, num_bytes) blocks forming one prefill rank's layer item."""
    if meta.slice_dim == 0:
        return ((0, meta.item_bytes),)

    from sglang.srt.disaggregation.utils import compute_mamba_state_slice_byte_blocks

    ratio = prefill_attn_tp_size // decode_attn_tp_size
    if meta.slice_dim % ratio != 0 or meta.item_bytes % ratio != 0:
        raise ValueError(f"{meta.key_suffix} does not split by attn TP ratio {ratio}")
    dst_item_bytes = meta.item_bytes // ratio
    blocks = compute_mamba_state_slice_byte_blocks(
        src_item_len=meta.item_bytes,
        dst_item_len=dst_item_bytes,
        src_dim=meta.slice_dim,
        dst_dim=meta.slice_dim // ratio,
        outer_count=meta.outer_count,
        src_attn_tp_size=decode_attn_tp_size,
        dst_attn_tp_size=prefill_attn_tp_size,
        dst_tp_rank_in_group=prefill_tp_rank,
        local_tp_rank_in_group=decode_attn_tp_rank,
        conv_shard_groups=(
            list(meta.conv_shard_groups) if meta.conv_shard_groups else None
        ),
    )
    blocks = sorted(blocks, key=lambda block: block[1])
    expected_dst = 0
    for _, dst_offset, num_bytes in blocks:
        if dst_offset != expected_dst:
            raise ValueError(f"{meta.key_suffix} blocks leave a gap at {dst_offset}")
        expected_dst += num_bytes
    if expected_dst != dst_item_bytes:
        raise ValueError(f"{meta.key_suffix} blocks cover {expected_dst} bytes")
    return tuple((src_offset, num_bytes) for src_offset, _, num_bytes in blocks)


def build_write_plan(
    target: StorageWriteTarget,
    *,
    tree_page_size: int,
    dcp_size: int,
    dcp_rank: int,
    attn_tp_size: int,
    attn_tp_rank: int,
    mamba_tensors: Sequence[MambaStateMeta],
) -> Union[StorageWritePlan, str]:
    """Combine the frozen target with this rank's layout; a str is a disable reason."""
    if tree_page_size != dcp_size * target.prefill_page_size:
        return "page_size"
    # Only DCP == attention TP gives each sub-page exactly one KV writer.
    if dcp_size > 1 and dcp_size != attn_tp_size:
        return "dcp_attn_tp"
    try:
        mamba_ranks = tuple(
            MambaRankBlocks(
                prefill_tp_rank=rank,
                per_tensor=tuple(
                    mamba_rank_blocks(
                        meta,
                        decode_attn_tp_size=attn_tp_size,
                        decode_attn_tp_rank=attn_tp_rank,
                        prefill_attn_tp_size=target.prefill_attn_tp_size,
                        prefill_tp_rank=rank,
                    )
                    for meta in mamba_tensors
                ),
            )
            for rank in target.target_tp_ranks
        )
    except ValueError:
        logger.exception("Mamba state cannot be resliced to the prefill attn TP")
        return "mamba_layout"
    return StorageWritePlan(
        sub_page_size=target.prefill_page_size,
        dcp_size=dcp_size,
        dcp_rank=dcp_rank,
        # MLA KV is replicated across attention TP unless DCP shards it.
        writes_kv=dcp_size > 1 or attn_tp_rank == 0,
        mamba_tensors=tuple(mamba_tensors),
        mamba_ranks=mamba_ranks,
    )


def mamba_put_objects(
    *,
    page_key: str,
    plan: StorageWritePlan,
    slot_bases: Sequence[int],
    num_layers: int,
) -> tuple[list[str], list[list[int]], list[list[int]]]:
    """Keys and host buffer lists of the prefill-layout Mamba objects for one slot.

    ``slot_bases[i]`` is the host address of tensor i's slot; a page_first host
    slot holds every layer back to back, matching the prefill object layout.
    """
    keys: list[str] = []
    ptrs: list[list[int]] = []
    sizes: list[list[int]] = []
    for rank_blocks in plan.mamba_ranks:
        for meta, base, blocks in zip(
            plan.mamba_tensors, slot_bases, rank_blocks.per_tensor
        ):
            key_ptrs: list[int] = []
            key_sizes: list[int] = []
            for layer in range(num_layers):
                layer_base = base + layer * meta.item_bytes
                for src_offset, num_bytes in blocks:
                    ptr = layer_base + src_offset
                    if key_ptrs and key_ptrs[-1] + key_sizes[-1] == ptr:
                        key_sizes[-1] += num_bytes
                    else:
                        key_ptrs.append(ptr)
                        key_sizes.append(num_bytes)
            keys.append(f"{page_key}_{rank_blocks.prefill_tp_rank}_{meta.key_suffix}")
            ptrs.append(key_ptrs)
            sizes.append(key_sizes)
    return keys, ptrs, sizes
