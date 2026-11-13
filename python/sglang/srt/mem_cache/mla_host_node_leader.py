"""One MLA KV HiCache host copy per node.

Every rank reopens the node leader's memfd via /proc/<pid>/fd/<fd>, so local ranks
need /proc visibility and permission on it; open or size failures abort
initialization on all ranks. Each rank prefaults a contiguous, page-aligned share
under its own memory policy, and no rank registers or pins before all are ready.
NUMA placement and binding are deployment responsibilities.

Followers keep only dummy slot metadata; the leader restores MLA layers and the
forward layer-wait hook broadcasts them over NCCL. Only the leader does MLA D2H,
L3 reads and Mooncake registration. An MHA draft sidecar can share a node buffer
too (see hicache_draft_node_merge.py).

KDA/Mamba stays per rank unless --enable-kda-hicache-node-aggregate is set: then
each Mamba host buffer is one node-shared memfd laid out as
``[rank0 share | rank1 share | ...]`` with an equal, page-aligned stride. Every
rank keeps a view of its own share, so H2D/D2H and slot allocation are
unchanged; only the leader registers the full buffers with Mooncake and moves
one multi-buffer object per checkpoint spanning every rank's slot.

Requires MADV_POPULATE_WRITE (Linux 5.14+). Launch validation rejects Mooncake
standalone_storage, PD decode and DCP.
"""

from __future__ import annotations

import contextlib
import logging
import mmap
import os
from typing import TYPE_CHECKING, Optional

import msgspec
import torch

from sglang.srt.distributed.parallel_state import create_custom_parallel_group
from sglang.srt.mem_cache.mla_host_dedup import (
    MLAHostDedupBroadcaster,
    mla_host_dedup_eligible,
)
from sglang.srt.mem_cache.pool_host.base import (
    host_memory_budget_bytes,
    host_memory_budget_scope,
    ranks_per_host,
)
from sglang.srt.mem_cache.pool_host.common import HostTensorAllocator
from sglang.srt.mem_cache.storage.mmap.mmap_allocator import create_shm, map_shm
from sglang.srt.runtime_context import get_memory
from sglang.srt.utils import get_device_module

if TYPE_CHECKING:
    from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool

logger = logging.getLogger(__name__)
device_module = get_device_module()


def split_ranks_by_node(group_ranks: list[int], ranks_on_host: int) -> list[list[int]]:
    """Group global ranks by node; assumes contiguous, uniform per-node placement."""
    nodes: dict[int, list[int]] = {}
    for rank in group_ranks:
        nodes.setdefault(rank // ranks_on_host, []).append(rank)
    return [nodes[node] for node in sorted(nodes)]


def synchronize_shared_host_error(
    group: torch.distributed.ProcessGroup,
    error: Exception | None,
    *,
    operation: str,
) -> None:
    """Collective fence over ``group``: every rank must call it,
    passing None on success."""
    errors = [None] * torch.distributed.get_world_size(group=group)
    message = (
        None if error is None else f"rank {torch.distributed.get_rank()}: {error!r}"
    )
    torch.distributed.all_gather_object(errors, message, group=group)
    failures = [message for message in errors if message is not None]
    if failures:
        raise RuntimeError(
            f"MLA HiCache shared host {operation} failed: " + "; ".join(failures)
        ) from error


class NodeSharedHostTensorAllocator(HostTensorAllocator):
    """Reopen one node-local memfd and collectively prefault before pinning."""

    def __init__(
        self,
        *,
        node_ranks: list[int],
        tp_cache_group: torch.distributed.ProcessGroup,
    ):
        super().__init__()
        # Global ranks of this node's sharing group; the first one is the leader.
        self.node_ranks = node_ranks
        self.tp_cache_group = tp_cache_group

    def allocate(self, dims: tuple, dtype: torch.dtype, device: str) -> torch.Tensor:
        """Collective over tp_cache_group;
        every rank calls it once, in the same order."""
        assert device == "cpu", device
        self.dims = dims
        self.dtype = dtype
        node_rank = self.node_ranks.index(torch.distributed.get_rank())
        fd = buffer = error = owner = None
        try:
            if node_rank == 0:
                try:
                    fd = create_shm(dims, dtype)
                    owner = (os.getpid(), fd)
                except Exception as exc:
                    error = exc
            group_ranks = torch.distributed.get_process_group_ranks(self.tp_cache_group)
            owners = [None] * len(group_ranks)
            torch.distributed.all_gather_object(
                owners, owner, group=self.tp_cache_group
            )
            try:
                if error is None:
                    if node_rank != 0:
                        owner = owners[group_ranks.index(self.node_ranks[0])]
                        fd = self._open_shared_fd(owner)
                    buffer = map_shm(
                        fd,
                        dims,
                        dtype,
                        populate_shard=(node_rank, len(self.node_ranks)),
                    )
            except Exception as exc:
                error = exc
            # This fence also keeps the leader's original FD live until all peers
            # have reopened it. No rank may pin before all shards are populated.
            synchronize_shared_host_error(
                self.tp_cache_group, error, operation="prefault"
            )
            return buffer
        except Exception:
            # Do not retain successful mappings in the raised traceback.
            buffer = None
            raise
        finally:
            if fd is not None:
                os.close(fd)

    def _open_shared_fd(self, owner: tuple[int, int] | None) -> int:
        leader_rank = self.node_ranks[0]
        if owner is None:
            raise RuntimeError(
                f"MLA HiCache node leader rank {leader_rank} "
                "did not publish a shared host buffer."
            )
        pid, owner_fd = owner
        path = f"/proc/{pid}/fd/{owner_fd}"
        try:
            return os.open(path, os.O_RDWR)
        except OSError as exc:
            raise RuntimeError(
                f"MLA HiCache cannot reopen node leader rank {leader_rank}'s "
                f"shared host buffer at {path}: {exc}. Ranks on one node must have "
                "permission and /proc visibility to the same leader descriptor."
            ) from exc


class _PendingLoad(msgspec.Struct):
    device_indices: torch.Tensor
    finish_event: torch.cuda.Event
    prepared: Optional[tuple[torch.Tensor, Optional[torch.Tensor]]] = None
    next_slot: int = 0


class NodeLeaderMLABroadcast(msgspec.Struct):
    """Node ranks must issue the same loads in the same order;
    each broadcast is an NCCL collective."""

    broadcaster: MLAHostDedupBroadcaster
    # (transfer layer id, MLA device layer id), sorted by transfer layer id.
    mla_layers: list[tuple[int, int]]
    # Packed MTP draft pools; each holds its KV in layer 0.
    draft_pools: tuple[MLATokenToKVPool, ...] = ()
    pending_loads: dict[int, _PendingLoad] = msgspec.field(default_factory=dict)

    @property
    def is_follower(self) -> bool:
        return not self.broadcaster.is_src

    def flush_stale(
        self, index: int, fence_stream: Optional[torch.cuda.Stream]
    ) -> None:
        """Must run before the new H2D waits on ``fence_stream``, so it cannot
        overwrite rows still being broadcast; the old H2D is already complete."""
        stale = self.pending_loads.pop(index, None)
        if stale is not None:
            self._flush(stale, fence_stream)

    def start_load(
        self,
        *,
        index: int,
        device_indices: torch.Tensor,
        finish_event: torch.cuda.Event,
    ) -> None:
        """``finish_event`` is re-recorded after the last broadcast, so ack waiters
        also wait on the broadcasts."""
        if device_indices.numel() == 0:
            return
        self.pending_loads[index] = _PendingLoad(
            device_indices=device_indices, finish_event=finish_event
        )

    def _flush(
        self, load: _PendingLoad, fence_stream: Optional[torch.cuda.Stream]
    ) -> None:
        """All broadcasts share one NCCL group and staging buffer, so they run on
        the fence (forward) stream, where the layer-wait hook also broadcasts."""
        stream_ctx = (
            contextlib.nullcontext()
            if fence_stream is None
            else device_module.stream(fence_stream)
        )
        with stream_ctx:
            self._broadcast_through(load, threshold=None)
            load.finish_event.record()

    def broadcast_ready_layers(self, index: int, threshold: int) -> None:
        """``threshold`` is a transfer layer id, also used by KDA/Mamba waits;
        transfer layers load in order on one stream,
        so MLA layers with id <= threshold are ready."""
        load = self.pending_loads.get(index)
        if load is None:
            return
        if self._broadcast_through(load, threshold):
            load.finish_event.record()
            del self.pending_loads[index]

    def _broadcast_through(self, load: _PendingLoad, threshold: Optional[int]) -> bool:
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
        # Draft depth d loads at transfer id d; assumes no draft id exceeds the
        # last MLA layer's, so drafts are already on the leader's GPU here.
        for pool in self.draft_pools:
            broadcaster._bcast_layer(
                pool.kv_buffer,
                broadcaster.kv_staging,
                self._device_rows(pool, load.device_indices),
                pool.kv_cache_dim,
                0,
            )
        return True

    def _device_rows(
        self, device_pool: MLATokenToKVPool, device_indices: torch.Tensor
    ) -> torch.Tensor:
        # Same controller-ID -> buffer-row translation as L2TransferEngine.
        indices = device_indices.to(self.broadcaster.device)
        translate = device_pool.host_transfer_translate
        if translate is None:
            return indices
        return translate(indices)


class NodeLeaderMLAContext(msgspec.Struct, frozen=True):
    is_follower: bool
    # Host KV capacity in per-rank shares, identical across the cache group.
    capacity_multiplier: int
    shared_allocator: NodeSharedHostTensorAllocator
    broadcast: NodeLeaderMLABroadcast

    @contextlib.contextmanager
    def leader_budget_scope(self):
        if self.is_follower:
            yield
            return
        with host_memory_budget_scope(
            host_memory_budget_bytes() * self.capacity_multiplier
        ):
            yield


def _build_broadcast(
    *,
    kv_pool: MLATokenToKVPool,
    full_layer_mapping: dict[int, int],
    draft_pools: tuple[MLATokenToKVPool, ...],
    nccl_group: torch.distributed.ProcessGroup,
    node_ranks: list[int],
) -> NodeLeaderMLABroadcast:
    broadcaster = MLAHostDedupBroadcaster(
        device_pool=kv_pool,
        group=nccl_group,
        src_global_rank=node_ranks[0],
    )
    # The constructor picks TP rank 0 as source;
    # per-node groups use each node's first rank.
    broadcaster.is_src = torch.distributed.get_rank() == node_ranks[0]
    broadcaster._warmup_group()
    for pool in draft_pools:
        # Drafts reuse the target's staging buffer.
        assert pool.kv_buffer[0].dtype == broadcaster.kv_staging.dtype
    return NodeLeaderMLABroadcast(
        broadcaster=broadcaster,
        mla_layers=sorted(full_layer_mapping.items()),
        draft_pools=draft_pools,
    )


def maybe_create_node_leader_context(
    *,
    kv_pool: MLATokenToKVPool,
    full_layer_mapping: dict[int, int],
    tp_cache_group: torch.distributed.ProcessGroup,
    draft_pools: tuple[MLATokenToKVPool, ...],
) -> Optional[NodeLeaderMLAContext]:
    """Call on every WORLD rank before host-pool allocation;
    it runs WORLD collectives when enabled."""
    memory = get_memory()
    if not memory.enable_mla_hicache_per_node:
        return None
    # Rank-independent, so every rank raises before the WORLD collectives below.
    if not mla_host_dedup_eligible(kv_pool, None) or not all(
        mla_host_dedup_eligible(pool, None) for pool in draft_pools
    ):
        raise ValueError(
            "--enable-mla-hicache-per-node needs CUDA and a non-FP4 MLA KV pool, "
            "including MTP draft pools."
        )
    group_ranks = torch.distributed.get_process_group_ranks(tp_cache_group)
    node_groups = split_ranks_by_node(group_ranks, ranks_per_host())
    rank = torch.distributed.get_rank()
    node_ranks = next(ranks for ranks in node_groups if rank in ranks)
    # The smallest node decides: every rank in this cache group
    # keeps the same host capacity.
    min_node_size = min(len(ranks) for ranks in node_groups)
    # WORLD-collective: singleton cache groups must join before returning.
    nccl_group = create_custom_parallel_group(group_ranks=node_ranks, backend="nccl")
    if min_node_size <= 1:
        # Every rank returns here together, so each drops only its own handle.
        torch.distributed.destroy_process_group(nccl_group)
        logger.info(
            "MLA HiCache node leader inactive: "
            "the smallest node in this cache group has one rank."
        )
        return None
    broadcast = _build_broadcast(
        kv_pool=kv_pool,
        full_layer_mapping=full_layer_mapping,
        draft_pools=draft_pools,
        nccl_group=nccl_group,
        node_ranks=node_ranks,
    )
    is_follower = rank != node_ranks[0]
    context = NodeLeaderMLAContext(
        is_follower=is_follower,
        capacity_multiplier=min_node_size,
        shared_allocator=NodeSharedHostTensorAllocator(
            node_ranks=node_ranks,
            tp_cache_group=tp_cache_group,
        ),
        broadcast=broadcast,
    )
    logger.info(
        "MLA HiCache node leader enabled: node_ranks=%s, role=%s, "
        "capacity_multiplier=%d, shared shard initialization",
        node_ranks,
        "follower" if is_follower else "leader",
        context.capacity_multiplier,
    )
    return context


# ===== KDA/Mamba node aggregation =====


class _NodeShardedBuffer(msgspec.Struct, frozen=True):
    full: torch.Tensor
    own: torch.Tensor
    stride_bytes: int


class NodeShardedHostTensorAllocator(HostTensorAllocator):
    """Map one node-local memfd per allocation; each rank gets its own share."""

    def __init__(
        self,
        *,
        node_ranks: list[int],
        tp_cache_group: torch.distributed.ProcessGroup,
    ):
        super().__init__()
        # Global ranks of this node; the first one is the leader.
        self.node_ranks = node_ranks
        self.tp_cache_group = tp_cache_group
        self.node_rank = node_ranks.index(torch.distributed.get_rank())
        self._buffers: list[_NodeShardedBuffer] = []

    def allocate(self, dims: tuple, dtype: torch.dtype, device: str) -> torch.Tensor:
        """Collective over tp_cache_group;
        every rank calls it once per buffer, in the same order."""
        assert device == "cpu", device
        self.dims = dims
        self.dtype = dtype
        nbytes = _numel(dims) * dtype.itemsize
        stride = -(-nbytes // mmap.PAGESIZE) * mmap.PAGESIZE
        full_dims = (len(self.node_ranks) * stride,)
        fd = full = error = owner = None
        try:
            if self.node_rank == 0:
                try:
                    fd = create_shm(full_dims, torch.uint8)
                    owner = (os.getpid(), fd, stride)
                except Exception as exc:
                    error = exc
            group_ranks = torch.distributed.get_process_group_ranks(self.tp_cache_group)
            owners = [None] * len(group_ranks)
            torch.distributed.all_gather_object(
                owners, owner, group=self.tp_cache_group
            )
            try:
                if error is None:
                    if self.node_rank != 0:
                        owner = owners[group_ranks.index(self.node_ranks[0])]
                        fd = self._open_shared_fd(owner, stride)
                    # Equal page-aligned shares: each rank faults exactly its own.
                    full = map_shm(
                        fd,
                        full_dims,
                        torch.uint8,
                        populate_shard=(self.node_rank, len(self.node_ranks)),
                    )
            except Exception as exc:
                error = exc
            # Keeps the leader's FD live until every peer reopened it;
            # no rank may pin its share before all shares are populated.
            synchronize_shared_host_error(
                self.tp_cache_group, error, operation="KDA prefault"
            )
            start = self.node_rank * stride
            own = full[start : start + nbytes].view(dtype).view(dims)
            self._buffers.append(
                _NodeShardedBuffer(full=full, own=own, stride_bytes=stride)
            )
            return own
        except Exception:
            full = None
            raise
        finally:
            if fd is not None:
                os.close(fd)

    def full_buffers(self) -> list[torch.Tensor]:
        return [buffer.full for buffer in self._buffers]

    def peer_base_ptrs(self, own_tensor: torch.Tensor) -> list[int]:
        """Base address of every rank's share of ``own_tensor``'s buffer,
        in node-rank order, in this process's address space."""
        ptr = own_tensor.data_ptr()
        for buffer in self._buffers:
            if buffer.own.data_ptr() == ptr:
                base = buffer.full.data_ptr()
                return [
                    base + rank * buffer.stride_bytes
                    for rank in range(len(self.node_ranks))
                ]
        raise ValueError("Tensor was not allocated by this node-sharded allocator.")

    def _open_shared_fd(
        self, owner: Optional[tuple[int, int, int]], stride: int
    ) -> int:
        leader_rank = self.node_ranks[0]
        if owner is None:
            raise RuntimeError(
                f"KDA HiCache node leader rank {leader_rank} "
                "did not publish a shared host buffer."
            )
        pid, owner_fd, leader_stride = owner
        if leader_stride != stride:
            # The leader computes peer addresses from its own stride.
            raise ValueError(
                f"KDA HiCache share stride {stride} differs from node leader "
                f"rank {leader_rank}'s {leader_stride}."
            )
        path = f"/proc/{pid}/fd/{owner_fd}"
        try:
            return os.open(path, os.O_RDWR)
        except OSError as exc:
            raise RuntimeError(
                f"KDA HiCache cannot reopen node leader rank {leader_rank}'s "
                f"shared host buffer at {path}: {exc}. Ranks on one node must have "
                "permission and /proc visibility to the same leader descriptor."
            ) from exc


def _numel(dims: tuple) -> int:
    count = 1
    for dim in dims:
        count *= dim
    return count


class KDANodeAggregateContext(msgspec.Struct, frozen=True):
    node_rank: int
    node_size: int
    node_index: int
    num_nodes: int
    # One Gloo group per storage thread so their collectives never interleave.
    backup_group: torch.distributed.ProcessGroup
    prefetch_group: torch.distributed.ProcessGroup
    allocator: NodeShardedHostTensorAllocator

    @property
    def is_leader(self) -> bool:
        return self.node_rank == 0


def maybe_create_kda_node_aggregate_context(
    *, tp_cache_group: torch.distributed.ProcessGroup
) -> Optional[KDANodeAggregateContext]:
    """Call on every WORLD rank before Mamba host allocation;
    it runs WORLD collectives when enabled."""
    if not get_memory().enable_kda_hicache_node_aggregate:
        return None
    group_ranks = torch.distributed.get_process_group_ranks(tp_cache_group)
    node_groups = split_ranks_by_node(group_ranks, ranks_per_host())
    rank = torch.distributed.get_rank()
    node_index, node_ranks = next(
        (index, sorted(ranks))
        for index, ranks in enumerate(node_groups)
        if rank in ranks
    )
    # WORLD-collective: every rank must join before any rank returns.
    backup_group = create_custom_parallel_group(group_ranks=node_ranks, backend="gloo")
    prefetch_group = create_custom_parallel_group(
        group_ranks=node_ranks, backend="gloo"
    )
    if min(len(ranks) for ranks in node_groups) <= 1:
        # Every rank returns here together, so each drops only its own handles.
        torch.distributed.destroy_process_group(prefetch_group)
        torch.distributed.destroy_process_group(backup_group)
        logger.info(
            "KDA HiCache node aggregation inactive: "
            "the smallest node in this cache group has one rank."
        )
        return None
    context = KDANodeAggregateContext(
        node_rank=node_ranks.index(rank),
        node_size=len(node_ranks),
        node_index=node_index,
        num_nodes=len(node_groups),
        backup_group=backup_group,
        prefetch_group=prefetch_group,
        allocator=NodeShardedHostTensorAllocator(
            node_ranks=node_ranks, tp_cache_group=tp_cache_group
        ),
    )
    logger.info(
        "KDA HiCache node aggregation enabled: node_ranks=%s, role=%s, node %d/%d",
        node_ranks,
        "leader" if context.is_leader else "follower",
        node_index,
        len(node_groups),
    )
    return context
