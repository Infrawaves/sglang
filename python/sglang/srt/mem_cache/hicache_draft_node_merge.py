"""One Mooncake object per page per node for MHA draft KV.

Each node's ranks share one memfd draft host buffer of shape
(node_size, 2, page_num, layer, page_size, head, dim): rank r owns, pins and
D2H/H2Ds slice r exactly like a private pool. The node's last rank
(``leader_mha``) does every L3 exists/get/put for the node, scatter-gathering
the 2 * node_size K/V segments of a page into one object; other ranks do no
draft L3 IO and report every page present, so the cache-group all_reduce(MIN)
on hits, prefetch completion and backup acks takes the leader's result.

Relies on every rank of a node holding the same host indices for a page, the
same assumption the node-leader MLA pool makes.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import msgspec
import torch

from sglang.srt.mem_cache.mla_host_node_leader import (
    NodeSharedHostTensorAllocator,
    split_ranks_by_node,
    synchronize_shared_host_error,
)
from sglang.srt.mem_cache.pool_host.base import ranks_per_host

if TYPE_CHECKING:
    from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost

logger = logging.getLogger(__name__)


class DraftNodeMerge(msgspec.Struct, frozen=True):
    # Global ranks of this node within the cache group, in rank order.
    node_ranks: list[int]
    # This rank's slot in node_ranks, also its slice of the node buffer.
    node_slot: int
    node_idx: int
    num_nodes: int
    tp_size: int

    @property
    def is_leader(self) -> bool:
        # The last rank, so draft L3 IO stays off the MLA leader's NIC.
        return self.node_slot == len(self.node_ranks) - 1

    @property
    def node_size(self) -> int:
        return len(self.node_ranks)

    @property
    def key_suffix(self) -> str:
        # Object size and head ranges depend on TP and node split, so they key.
        return f"tp{self.tp_size}_n{self.node_idx}of{self.num_nodes}"


class DraftNodeShareAllocator(NodeSharedHostTensorAllocator):
    """Map one node buffer and hand back this rank's slice.

    The base fence guarantees every slice is prefaulted before any rank pins
    its own slice; the leader keeps ``node_buffer`` for Mooncake registration.
    """

    def __init__(
        self,
        *,
        merge: DraftNodeMerge,
        tp_cache_group: torch.distributed.ProcessGroup,
    ):
        super().__init__(node_ranks=merge.node_ranks, tp_cache_group=tp_cache_group)
        self.merge = merge
        self.node_buffer = None
        # Whether this rank has joined allocate's collectives (or aborted them).
        self.entered = False

    def allocate(self, dims: tuple, dtype: torch.dtype, device: str) -> torch.Tensor:
        self.entered = True
        node_buffer = super().allocate(
            (self.merge.node_size, *dims), dtype=dtype, device=device
        )
        self.dims = dims
        self.node_buffer = node_buffer
        return node_buffer[self.merge.node_slot]

    def abort(self, error: Exception) -> None:
        """For a rank that failed before ``allocate``: run the same collectives
        with no buffer so peers raise instead of waiting forever."""
        if self.entered:
            return
        self.entered = True
        group_ranks = torch.distributed.get_process_group_ranks(self.tp_cache_group)
        torch.distributed.all_gather_object(
            [None] * len(group_ranks), None, group=self.tp_cache_group
        )
        synchronize_shared_host_error(self.tp_cache_group, error, operation="prefault")


def maybe_create_draft_node_merge(
    *,
    enabled: bool,
    tp_cache_group: torch.distributed.ProcessGroup,
) -> DraftNodeMerge | None:
    """Collective-free, so every rank of the cache group decides alike."""
    if not enabled:
        return None
    group_ranks = torch.distributed.get_process_group_ranks(tp_cache_group)
    node_groups = split_ranks_by_node(group_ranks, ranks_per_host())
    if min(len(ranks) for ranks in node_groups) <= 1:
        raise ValueError(
            "--hicache-draft-node-merge needs at least two ranks on every node "
            "of the cache group."
        )
    rank = torch.distributed.get_rank()
    node_idx, node_ranks = next(
        (idx, ranks) for idx, ranks in enumerate(node_groups) if rank in ranks
    )
    return DraftNodeMerge(
        node_ranks=node_ranks,
        node_slot=node_ranks.index(rank),
        node_idx=node_idx,
        num_nodes=len(node_groups),
        tp_size=len(group_ranks),
    )


def node_page_buffer_meta(
    host_pool: MHATokenToKVPoolHost, merge: DraftNodeMerge, host_indices
) -> tuple[list[list[int]], list[list[int]]]:
    """Per page: [r0.K, r0.V, r1.K, r1.V, ...] segments of the node buffer.

    Slices are equal-sized and contiguous, so a peer's segment is this rank's
    segment shifted by the slice stride.
    """
    ptr_list, size_list = host_pool.get_page_buffer_meta(host_indices)
    stride = host_pool.kv_buffer.nbytes
    offsets = [(slot - merge.node_slot) * stride for slot in range(merge.node_size)]
    node_ptrs, node_sizes = [], []
    # get_page_buffer_meta yields (K, V) per page for page_first_direct.
    for page in range(0, len(ptr_list), 2):
        k_ptr, v_ptr = ptr_list[page], ptr_list[page + 1]
        node_ptrs.append([p + off for off in offsets for p in (k_ptr, v_ptr)])
        node_sizes.append(
            [size for _ in offsets for size in size_list[page : page + 2]]
        )
    return node_ptrs, node_sizes


def log_draft_node_merge(host_pool: MHATokenToKVPoolHost, merge: DraftNodeMerge):
    page_bytes = host_pool.kv_buffer[:, 0].nbytes
    logger.info(
        "draft node merge: role=%s node_ranks=%s key_suffix=%s object_bytes=%d",
        "leader_mha" if merge.is_leader else "follower",
        merge.node_ranks,
        merge.key_suffix,
        page_bytes * merge.node_size,
    )
