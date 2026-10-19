"""One MLA KV host copy per node, broadcast to local GPUs over NVLink.

MLA latent KV is identical on every attention-TP rank. With this mode only the
first rank on each node (the leader) keeps MLA KV in its HiCache host pool and
loads it from L3; the other local ranks (followers) keep an allocator-only
(#36800 ``is_dummy``) host pool. After the leader's H2D copy, the forward's
layer-wait hook broadcasts the MLA layers (and packed MTP draft layers) to
the followers over NCCL, layer by layer, so later layers keep loading while
earlier ones compute.

PD decode is not supported: its load-back has no forward that waits on the
layers, so nothing would trigger the broadcast. A decode server falls back to
the per-rank host copies.

L3 data, L3 keys and the host page layout are unchanged. KDA/Mamba state is
not touched: it differs per rank and keeps its own host pool on every rank.
"""

from __future__ import annotations

import contextlib
import logging
from typing import Any, Optional

import msgspec
import torch

from sglang.srt.layers.dp_attention import is_dp_attention_enabled
from sglang.srt.mem_cache.mla_host_dedup import (
    MLAHostDedupBroadcaster,
    mla_host_dedup_eligible,
)
from sglang.srt.mem_cache.pool_host.base import (
    host_memory_budget_bytes,
    host_memory_budget_scope,
    ranks_per_host,
)
from sglang.srt.runtime_context import get_disagg, get_memory, get_parallel
from sglang.srt.utils import get_device_module

logger = logging.getLogger(__name__)
device_module = get_device_module()


def split_ranks_by_node(group_ranks: list[int], ranks_on_host: int) -> list[list[int]]:
    """Split a process group's global ranks into per-node groups, in order."""
    nodes: dict[int, list[int]] = {}
    for rank in group_ranks:
        nodes.setdefault(rank // ranks_on_host, []).append(rank)
    return [nodes[node] for node in sorted(nodes)]


class NodeLeaderMLABroadcaster(MLAHostDedupBroadcaster):
    """Broadcast from the first rank of this node's group instead of rank 0."""

    def __init__(
        self,
        device_pool,
        group,
        node_ranks: list[int],
        min_node_size: int,
        cpu_group: Optional[torch.distributed.ProcessGroup] = None,
    ):
        super().__init__(device_pool, group, src_global_rank=node_ranks[0])
        self.node_ranks = node_ranks
        # Smallest node group in the TP group; the same on every rank.
        self.min_node_size = min_node_size
        self.is_src = torch.distributed.get_rank() == node_ranks[0]
        # The TP CPU (gloo) group, for startup barriers across all node groups.
        self.cpu_group = cpu_group

    @classmethod
    def build_for_node(
        cls,
        device_pool,
        tp_group: torch.distributed.ProcessGroup,
        attn_tp_group: Optional[torch.distributed.ProcessGroup],
    ) -> Optional[NodeLeaderMLABroadcaster]:
        """World-collective: every rank must call this, in the same order."""
        from sglang.srt.distributed.parallel_state import create_custom_parallel_group

        base_group = tp_group
        if is_dp_attention_enabled() and attn_tp_group is not None:
            base_group = attn_tp_group
        group_ranks = torch.distributed.get_process_group_ranks(base_group)
        rank = torch.distributed.get_rank()
        node_groups = split_ranks_by_node(list(group_ranks), ranks_per_host())
        node_ranks = next(ranks for ranks in node_groups if rank in ranks)
        # Each rank passes its own node group. The CUDA path all-gathers every
        # rank's group and builds all of them in the same order, so differing
        # lists across nodes are expected here.
        group = create_custom_parallel_group(group_ranks=node_ranks, backend="nccl")
        # Every rank sees every node group, so all ranks decide together: a
        # one-rank node would otherwise leave its rank with a different host
        # capacity from the rest of the TP group.
        min_node_size = min(len(ranks) for ranks in node_groups)
        if min_node_size <= 1:
            return None
        broadcaster = cls(device_pool, group, node_ranks, min_node_size, base_group)
        broadcaster._warmup_group()
        return broadcaster


class _PendingLoad(msgspec.Struct):
    device_indices: torch.Tensor
    finish_event: Any
    prepared: Optional[tuple[torch.Tensor, Optional[torch.Tensor]]] = None
    next_slot: int = 0


class NodeLeaderMLAContext(msgspec.Struct):
    """State of the node-leader path; ``None`` everywhere when it is off."""

    broadcaster: NodeLeaderMLABroadcaster
    # (transfer layer id, MLA device layer id), sorted by transfer layer id.
    mla_layers: list[tuple[int, int]]
    # Packed MTP draft pools; each holds its KV in layer 0.
    draft_pools: tuple[Any, ...] = ()
    pending_loads: dict[int, _PendingLoad] = msgspec.field(default_factory=dict)

    @property
    def is_follower(self) -> bool:
        return not self.broadcaster.is_src

    @property
    def node_size(self) -> int:
        return len(self.broadcaster.node_ranks)

    def host_pool_kwargs(self) -> dict:
        """Every rank sizes the KV allocator for a whole node; only the leader
        backs it with memory. The smallest node sets the size, so all ranks of
        the TP group see the same host capacity."""
        return {
            "is_dummy": self.is_follower,
            "capacity_multiplier": self.broadcaster.min_node_size,
        }

    def leader_budget_scope(self):
        """The host budget is split evenly across local ranks. The leader holds
        the node's only MLA copy, so it may claim the followers' shares too."""
        if self.is_follower:
            return contextlib.nullcontext()
        return host_memory_budget_scope(host_memory_budget_bytes() * self.node_size)

    def wait_until_host_pool_ready(self) -> None:
        """Block until the leader has pinned its MLA host pool.

        Followers build an allocator-only pool with no memory check and would
        reach the next host pool's check while the leader is still pinning, so
        that check would read free memory that is about to be taken. Without
        a budget snapshot (explicit --hicache-ratio/--hicache-size) nothing
        else catches it. Gloo, not NCCL: this is a CPU wait, and pinning
        hundreds of GB can outlast the NCCL watchdog.
        """
        if self.broadcaster.cpu_group is None:
            return
        torch.distributed.barrier(group=self.broadcaster.cpu_group)

    def flush_stale(self, index: int, fence_stream: Any = None) -> None:
        """Finish an unconsumed load left in slot ``index`` before it is reused.

        Called before the new load's H2D waits on ``fence_stream``, so that
        H2D is ordered after this flush and cannot overwrite rows it reads.
        The old load's H2D is done (update_producer asserts it).
        """
        stale = self.pending_loads.pop(index, None)
        if stale is not None:
            self._flush(stale, fence_stream)

    def start_load(
        self, *, index: int, device_indices: torch.Tensor, finish_event: Any
    ) -> None:
        """Queue load ``index``; ``finish_event`` is recorded again after the
        last broadcast, so waiting on the load also waits on the broadcasts."""
        if device_indices.numel() == 0:
            return
        self.pending_loads[index] = _PendingLoad(device_indices, finish_event)

    def _flush(self, load: _PendingLoad, fence_stream: Any) -> None:
        """Broadcast every remaining layer of ``load``.

        Runs on the fence (forward) stream, where the layer-wait hook also
        broadcasts: the NCCL group and the staging buffer are shared, so all
        broadcasts must stay on one stream, after the previous forward and
        before the next one.
        """
        stream_ctx = (
            contextlib.nullcontext()
            if fence_stream is None
            else device_module.stream(fence_stream)
        )
        with stream_ctx:
            self._broadcast_through(load, threshold=None)
            load.finish_event.record()

    def broadcast_ready_layers(self, index: int, threshold: int) -> None:
        """Layer-wait hook, on the forward stream after waiting for transfer
        layer ``threshold``. Transfer layers load in order on one stream, so
        every MLA layer with a transfer id <= threshold is on the leader's GPU.

        ``threshold`` is a transfer layer id: KDA layers and the Mamba state
        copy call this too, so it must not be read as an MLA layer id.
        """
        load = self.pending_loads.get(index)
        if load is None:
            return
        if self._broadcast_through(load, threshold):
            load.finish_event.record()
            del self.pending_loads[index]

    def _broadcast_through(self, load: _PendingLoad, threshold: Optional[int]) -> bool:
        """Broadcast pending MLA layers up to ``threshold`` (all if None).
        Return True once every layer, drafts included, has been broadcast."""
        broadcaster = self.broadcaster
        if load.prepared is None:
            load.prepared = broadcaster.prepare_broadcast(
                self._device_rows(broadcaster.device_pool, load.device_indices),
                device_module.current_stream(),
            )
        slot = load.next_slot
        while slot < len(self.mla_layers) and (
            threshold is None or self.mla_layers[slot][0] <= threshold
        ):
            broadcaster.broadcast_loaded_layer(self.mla_layers[slot][1], load.prepared)
            slot += 1
        load.next_slot = slot
        if slot < len(self.mla_layers):
            return False
        # Draft layers load at transfer ids below every MLA target layer, so
        # they are on the leader's GPU by now.
        for pool in self.draft_pools:
            broadcaster._bcast_layer(
                pool.kv_buffer,
                broadcaster.kv_staging,
                self._device_rows(pool, load.device_indices),
                pool.kv_cache_dim,
                0,
            )
        return True

    def _device_rows(self, device_pool, device_indices: torch.Tensor) -> torch.Tensor:
        # Same controller-ID -> buffer-row translation as L2TransferEngine.
        indices = device_indices.to(self.broadcaster.device)
        translate = device_pool.host_transfer_translate
        if translate is None:
            return indices
        return translate(indices)


def maybe_create_node_leader_context(
    kv_pool,
    full_layer_mapping: dict[int, int],
    tp_group: torch.distributed.ProcessGroup,
    attn_tp_group: Optional[torch.distributed.ProcessGroup],
    draft_pools: tuple[Any, ...] = (),
) -> Optional[NodeLeaderMLAContext]:
    """Create the context before host-pool allocation, or return ``None``."""
    if not get_memory().enable_mla_hicache_per_node:
        return None
    # Rank-independent, so every rank skips the collective below together.
    if not mla_host_dedup_eligible(kv_pool, None) or not all(
        mla_host_dedup_eligible(pool, None) for pool in draft_pools
    ):
        logger.warning(
            "MLA HiCache node leader inactive: needs CUDA and a non-FP4 MLA KV pool."
        )
        return None
    if get_disagg().disaggregation_mode == "decode":
        # Decode load-back has no forward to run the layer-wait hook.
        logger.warning("MLA HiCache node leader inactive: not supported on PD decode.")
        return None
    if get_parallel().dcp_enabled:
        # DCP splits each page across ranks, so local ranks do not hold the
        # same MLA KV. Rank-independent, checked before the collective.
        logger.warning("MLA HiCache node leader inactive: not supported with DCP.")
        return None
    broadcaster = NodeLeaderMLABroadcaster.build_for_node(
        kv_pool, tp_group, attn_tp_group
    )
    if broadcaster is None:
        logger.info("MLA HiCache node leader inactive: one rank on this node.")
        return None
    for pool in draft_pools:
        # Drafts reuse the target's staging buffer.
        assert pool.kv_buffer[0].dtype == broadcaster.kv_staging.dtype
    context = NodeLeaderMLAContext(
        broadcaster=broadcaster,
        mla_layers=sorted(full_layer_mapping.items()),
        draft_pools=tuple(draft_pools),
    )
    logger.info(
        "MLA HiCache node leader enabled: node_ranks=%s, role=%s, mla_layers=%d, "
        "draft_layers=%d",
        broadcaster.node_ranks,
        "follower" if context.is_follower else "leader",
        len(context.mla_layers),
        len(context.draft_pools),
    )
    return context
