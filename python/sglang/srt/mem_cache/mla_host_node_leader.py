"""One MLA KV HiCache host copy per node, shared by the local GPUs.

MLA latent KV is identical on every attention-TP rank. With this mode only the
first rank on each node (the leader) writes MLA KV to host and reads it from
L3; the other local ranks (followers) skip D2H backup, L3 prefetch into host
and Mooncake registration. SGLANG_MLA_HICACHE_PER_NODE_MODE picks how
followers get the KV:

  broadcast: followers keep an allocator-only (#36800 ``is_dummy``) host pool.
    After the leader's H2D copy, the forward's layer-wait hook broadcasts the
    MLA layers (and packed MTP draft layers) to the followers over NCCL, layer
    by layer, so later layers keep loading while earlier ones compute.
  shm: the leader allocates the pool in a memfd and sends the fd to the
    followers over an abstract Unix socket (SCM_RIGHTS). Every rank maps and
    pins the same pages and does its own H2D.

In both modes the leader's pool is interleaved over the NUMA nodes its local
ranks are bound to when it can reach them all (leader_host_shares).

PD decode is not supported: broadcast needs a forward that waits on the
layers, and shm is not validated there. A decode server falls back to
per-rank host copies.

L3 data, L3 keys and the host page layout are unchanged. KDA/Mamba state is
not touched: it differs per rank and keeps its own host pool on every rank.
"""

from __future__ import annotations

import contextlib
import logging
import os
import socket
import uuid
from typing import Any, Optional

import msgspec
import torch

from sglang.srt.environ import envs
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
from sglang.srt.mem_cache.pool_host.common import HostTensorAllocator
from sglang.srt.mem_cache.storage.mmap.mmap_allocator import alloc_shm, map_shm
from sglang.srt.runtime_context import (
    get_device,
    get_disagg,
    get_memory,
    get_parallel,
    get_server_args,
)
from sglang.srt.utils import get_device_module

logger = logging.getLogger(__name__)
device_module = get_device_module()


def local_numa_info() -> tuple[int, Optional[frozenset[int]]]:
    """(NUMA node this rank is bound to or -1, NUMA nodes it may allocate on).

    The node mirrors the launcher (numa_utils.configure_subprocess and the
    scheduler's numa_bind_to_node): both bind to get_numa_node_if_available()
    for the rank's GPU. The allowed set is the cpuset limit, or None if unknown.
    """
    from sglang.srt.utils.numa_utils import (
        get_mems_allowed,
        get_numa_node_if_available,
    )

    gpu_id = get_device().gpu_id
    if gpu_id is None:
        return -1, None
    try:
        node = get_numa_node_if_available(get_server_args(), gpu_id)
    except Exception:
        logger.warning("MLA HiCache node leader: NUMA lookup failed", exc_info=True)
        return -1, None
    return (-1 if node is None else int(node)), get_mems_allowed()


def leader_host_shares(
    node_ranks: list[int],
    numa_of: dict[int, int],
    leader_mems_allowed: Optional[frozenset[int]],
) -> tuple[int, tuple[int, ...]]:
    """(per-rank host shares the leader can hold, NUMA nodes to interleave on).

    Each rank's share sits on the NUMA node it is bound to. The leader holds
    them all by interleaving its pool evenly over the group's nodes; even
    interleave puts the same amount on every node, so a node holding fewer
    ranks caps it at (fewest ranks on a node) x (node count). When the group
    is on one node, or no rank is bound, nothing needs interleaving. When the
    leader cannot reach every node (cpuset) or binding is mixed, it falls back
    to the ranks on its own node, which membind keeps it to.
    """
    nodes = [numa_of[rank] for rank in node_ranks]
    span = sorted(set(nodes))
    if len(span) == 1:
        return len(node_ranks), ()
    if -1 not in span and (
        leader_mems_allowed is not None and set(span) <= leader_mems_allowed
    ):
        return min(nodes.count(node) for node in span) * len(span), tuple(span)
    leader_numa = numa_of[node_ranks[0]]
    return sum(1 for node in nodes if node == leader_numa), ()


def split_ranks_by_node(group_ranks: list[int], ranks_on_host: int) -> list[list[int]]:
    """Split a process group's global ranks into per-node groups, in order."""
    nodes: dict[int, list[int]] = {}
    for rank in group_ranks:
        nodes.setdefault(rank // ranks_on_host, []).append(rank)
    return [nodes[node] for node in sorted(nodes)]


class NodeSharedHostTensorAllocator(HostTensorAllocator):
    """The leader allocates a memfd and sends it to every follower; a follower
    maps the fd it receives. Each side pins its own mapping afterwards."""

    def __init__(
        self,
        *,
        is_follower: bool,
        peer_sockets: list[socket.socket],
    ):
        super().__init__()
        self.is_follower = is_follower
        # Leader: one connection per follower. Follower: the leader connection.
        self.peer_sockets = peer_sockets

    def allocate(self, dims: tuple, dtype: torch.dtype, device: str) -> torch.Tensor:
        assert device == "cpu", device
        self.dims = dims
        self.dtype = dtype
        try:
            if self.is_follower:
                return self._receive(dims, dtype)
            # Faulted in this thread, so leader_budget_scope's interleave applies.
            tensor, fd, _ = alloc_shm(dims, dtype)
            try:
                for sock in self.peer_sockets:
                    socket.send_fds(sock, [b"\0"], [fd])
                return tensor
            finally:
                # The mapping owns the pages after this one-buffer exchange.
                os.close(fd)
        finally:
            for sock in self.peer_sockets:
                sock.close()
            self.peer_sockets.clear()

    def _receive(self, dims: tuple, dtype: torch.dtype) -> torch.Tensor:
        # Blocks until the leader has faulted in the pool, which can take minutes.
        _, fds, _, _ = socket.recv_fds(self.peer_sockets[0], 1, 1)
        if not fds:
            raise RuntimeError(
                "MLA HiCache node leader closed its socket before sending the "
                "shared host buffer; see the leader's log."
            )
        try:
            return map_shm(fds[0], dims, dtype)
        finally:
            for fd in fds:
                os.close(fd)


def _socket_address(token: str) -> str:
    # Abstract namespace: nothing on disk, visible only in this network namespace.
    return f"\0sglang-mla-hicache-{token}"


def _connect_node(
    *,
    is_follower: bool,
    node_ranks: list[int],
    tp_cache_group: torch.distributed.ProcessGroup,
) -> list[socket.socket]:
    """Collective over ``tp_cache_group``: connect each follower to its node
    leader. Bind/connect errors are aggregated before accepting peers."""
    rank = torch.distributed.get_rank()
    server = None
    token = None
    error = None
    peer_sockets = []
    try:
        if not is_follower:
            # Any failure must still reach the all_reduce below, or followers hang.
            try:
                token = uuid.uuid4().hex
                server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                server.bind(_socket_address(token))
                server.listen(len(node_ranks))
            except Exception as e:
                error = f"leader setup failed ({e!r})"

        group_ranks = torch.distributed.get_process_group_ranks(tp_cache_group)
        tokens = [None] * len(group_ranks)
        torch.distributed.all_gather_object(tokens, token, group=tp_cache_group)

        if is_follower:
            leader_token = tokens[group_ranks.index(node_ranks[0])]
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                sock.connect(_socket_address(leader_token))
            except OSError as e:
                error = f"cannot reach rank {node_ranks[0]}'s socket ({e})"
            peer_sockets.append(sock)

        ok = torch.tensor([0 if error else 1], dtype=torch.int32)
        torch.distributed.all_reduce(
            ok, op=torch.distributed.ReduceOp.MIN, group=tp_cache_group
        )
        if error is not None:
            logger.error("MLA HiCache node leader: rank %d: %s.", rank, error)
        if ok.item() == 0:
            raise RuntimeError(
                "--enable-mla-hicache-per-node: host-local setup failed on some rank "
                "(see the error logs). Ranks on one host must share a network "
                "namespace (one pod)."
            )
        if server is not None:
            # Every follower connected before the all_reduce, so none of these blocks.
            for _ in node_ranks[1:]:
                peer_sockets.append(server.accept()[0])
        return peer_sockets
    except Exception:
        for sock in peer_sockets:
            sock.close()
        raise
    finally:
        if server is not None:
            server.close()


class NodeLeaderMLABroadcaster(MLAHostDedupBroadcaster):
    """Broadcast from the first rank of this node's group instead of rank 0."""

    def __init__(
        self,
        device_pool,
        group,
        node_ranks: list[int],
        cpu_group: Optional[torch.distributed.ProcessGroup] = None,
    ):
        super().__init__(device_pool, group, src_global_rank=node_ranks[0])
        self.is_src = torch.distributed.get_rank() == node_ranks[0]
        # The TP CPU (gloo) group, for startup barriers across all node groups.
        self.cpu_group = cpu_group


class _PendingLoad(msgspec.Struct):
    device_indices: torch.Tensor
    finish_event: Any
    prepared: Optional[tuple[torch.Tensor, Optional[torch.Tensor]]] = None
    next_slot: int = 0


class NodeLeaderMLABroadcast(msgspec.Struct):
    """Broadcast-mode state; the controller's ``mla_dedup_context``."""

    broadcaster: NodeLeaderMLABroadcaster
    # (transfer layer id, MLA device layer id), sorted by transfer layer id.
    mla_layers: list[tuple[int, int]]
    # Packed MTP draft pools; each holds its KV in layer 0.
    draft_pools: tuple[Any, ...] = ()
    pending_loads: dict[int, _PendingLoad] = msgspec.field(default_factory=dict)

    @property
    def is_follower(self) -> bool:
        return not self.broadcaster.is_src

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


class NodeLeaderMLAContext(msgspec.Struct, frozen=True):
    """State of the per-node MLA host copy; ``None`` everywhere when it is off.
    Exactly one of ``broadcast`` and ``shared_allocator`` is set."""

    is_follower: bool
    # Host KV capacity in per-rank shares: the smallest leader_host_shares in
    # the TP group, so every rank sizes its allocator the same.
    capacity_multiplier: int
    # NUMA nodes the leader interleaves its MLA host pool over; empty when its
    # node group sits on one node or cannot be interleaved.
    interleave_nodes: tuple[int, ...]
    broadcast: Optional[NodeLeaderMLABroadcast]
    shared_allocator: Optional[NodeSharedHostTensorAllocator]

    def host_pool_kwargs(self) -> dict:
        """Every rank sizes the KV pool for capacity_multiplier ranks, so all
        ranks of the TP group see the same host capacity."""
        return {
            "capacity_multiplier": self.capacity_multiplier,
            "is_dummy": self.is_follower and self.broadcast is not None,
            "host_allocator": self.shared_allocator,
            "host_buffer_read_only": self.is_follower,
        }

    @contextlib.contextmanager
    def leader_budget_scope(self):
        """Scope for building this rank's MLA host pool.

        The host budget is split evenly across local ranks; the node's one MLA
        copy may claim capacity_multiplier ranks' shares. The leader faults
        its pool in here, so interleaving it over the ranks' NUMA nodes keeps
        each node at its own ranks' share.
        """
        if self.is_follower:
            yield
            return
        with contextlib.ExitStack() as stack:
            stack.enter_context(
                host_memory_budget_scope(
                    host_memory_budget_bytes() * self.capacity_multiplier
                )
            )
            if self.interleave_nodes:
                from sglang.srt.utils.numa_utils import interleave_memory

                stack.enter_context(interleave_memory(self.interleave_nodes))
            yield


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
    # Rank-independent, so every rank skips the collectives below together.
    if not mla_host_dedup_eligible(kv_pool, None) or not all(
        mla_host_dedup_eligible(pool, None) for pool in draft_pools
    ):
        logger.warning(
            "MLA HiCache node leader inactive: needs CUDA and a non-FP4 MLA KV pool."
        )
        return None
    if get_disagg().disaggregation_mode == "decode":
        # Broadcast needs a forward to run the layer-wait hook; shm is unvalidated.
        logger.warning("MLA HiCache node leader inactive: not supported on PD decode.")
        return None
    if get_parallel().dcp_enabled:
        # DCP splits each page across ranks, so local ranks do not hold the
        # same MLA KV. Rank-independent, checked before the collective.
        logger.warning("MLA HiCache node leader inactive: not supported with DCP.")
        return None
    from sglang.srt.distributed.parallel_state import get_world_group

    mode = envs.SGLANG_MLA_HICACHE_PER_NODE_MODE.get()
    # PP=1 and the rank-independent gates above put every WORLD rank here.
    # Agreement must precede create_custom_parallel_group, even for size-one TP.
    world_group = get_world_group().cpu_group
    modes = [None] * torch.distributed.get_world_size(group=world_group)
    torch.distributed.all_gather_object(modes, mode, group=world_group)
    if any(value != mode for value in modes):
        raise ValueError(
            "SGLANG_MLA_HICACHE_PER_NODE_MODE must agree across WORLD ranks; "
            f"got {modes!r}."
        )
    base_group = tp_group
    if is_dp_attention_enabled() and attn_tp_group is not None:
        base_group = attn_tp_group
    group_ranks = list(torch.distributed.get_process_group_ranks(base_group))
    node_groups = split_ranks_by_node(group_ranks, ranks_per_host())
    rank = torch.distributed.get_rank()
    node_ranks = next(ranks for ranks in node_groups if rank in ranks)
    nccl_group = None
    if mode == "broadcast":
        from sglang.srt.distributed.parallel_state import (
            create_custom_parallel_group,
        )

        # World-collective, so it runs before the size check below. Each rank
        # passes its own node group; the CUDA path all-gathers every rank's
        # group and builds all of them in the same order.
        nccl_group = create_custom_parallel_group(
            group_ranks=node_ranks, backend="nccl"
        )
    # Every rank sees every node group, so all ranks decide together: a
    # one-rank node would otherwise leave its rank with a different host
    # capacity from the rest of the TP group.
    min_node_size = min(len(ranks) for ranks in node_groups)
    if min_node_size <= 1:
        logger.info("MLA HiCache node leader inactive: one rank on this node.")
        return None
    # Only this rank knows its NUMA binding; all_gather_object orders results
    # by group rank. Runs in both modes, before any mode-specific collective.
    infos: list[Optional[tuple[int, Optional[frozenset[int]]]]] = [None] * len(
        group_ranks
    )
    torch.distributed.all_gather_object(infos, local_numa_info(), group=base_group)
    numa_of = {r: info[0] for r, info in zip(group_ranks, infos)}
    allowed_of = {r: info[1] for r, info in zip(group_ranks, infos)}
    shares = {
        tuple(ranks): leader_host_shares(ranks, numa_of, allowed_of[ranks[0]])
        for ranks in node_groups
    }
    capacity_multiplier = min(count for count, _ in shares.values())
    interleave_nodes = shares[tuple(node_ranks)][1]
    is_follower = rank != node_ranks[0]
    broadcast = shared_allocator = None
    if mode == "broadcast":
        broadcaster = NodeLeaderMLABroadcaster(
            device_pool=kv_pool,
            group=nccl_group,
            node_ranks=node_ranks,
            cpu_group=base_group,
        )
        broadcaster._warmup_group()
        for pool in draft_pools:
            # Drafts reuse the target's staging buffer.
            assert pool.kv_buffer[0].dtype == broadcaster.kv_staging.dtype
        broadcast = NodeLeaderMLABroadcast(
            broadcaster=broadcaster,
            mla_layers=sorted(full_layer_mapping.items()),
            draft_pools=tuple(draft_pools),
        )
    else:
        peer_sockets = _connect_node(
            is_follower=is_follower,
            node_ranks=node_ranks,
            tp_cache_group=base_group,
        )
        shared_allocator = NodeSharedHostTensorAllocator(
            is_follower=is_follower, peer_sockets=peer_sockets
        )
    layer_counts = (
        ""
        if broadcast is None
        else f", mla_layers={len(broadcast.mla_layers)}, "
        f"draft_layers={len(broadcast.draft_pools)}"
    )
    logger.info(
        "MLA HiCache node leader enabled: mode=%s, node_ranks=%s, role=%s, "
        "capacity_multiplier=%d, interleave_nodes=%s%s",
        mode,
        node_ranks,
        "follower" if is_follower else "leader",
        capacity_multiplier,
        list(interleave_nodes),
        layer_counts,
    )
    return NodeLeaderMLAContext(
        is_follower=is_follower,
        capacity_multiplier=capacity_multiplier,
        interleave_nodes=interleave_nodes,
        broadcast=broadcast,
        shared_allocator=shared_allocator,
    )
