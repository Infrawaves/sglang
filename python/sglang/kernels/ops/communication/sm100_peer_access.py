"""Peer validation shared by the SM100 GEMM + RS and AG + GEMM contexts."""

from __future__ import annotations

import torch
import torch.distributed as dist


def check_sm100_peer_access(group, device: torch.device) -> None:
    """Validate the CPU ``group`` the way custom all-reduce v2 admits it: the
    native peer check on one host, one NVLink fabric clique across hosts."""
    from sglang.srt.distributed.device_communicators.custom_all_reduce_utils import (
        is_one_nvlink_clique,
    )
    from sglang.srt.distributed.parallel_state import in_the_same_node_as

    if all(in_the_same_node_as(group, source_rank=0)):
        rank_device_uuids = [None] * dist.get_world_size(group=group)
        dist.all_gather_object(
            rank_device_uuids,
            torch.ops.mega_ops.sm100_device_uuid(device.index),
            group=group,
        )
        torch.ops.mega_ops.sm100_check_peer_access(device.index, rank_device_uuids)
    # sm100_check_peer_access resolves peers among this host's GPUs only; across
    # hosts, symmetric memory maps peers through the fabric (GB300 TP8 = 2x4).
    elif not is_one_nvlink_clique(group, device):
        raise RuntimeError(
            "SM100 fused collective peers span hosts outside one NVLink fabric clique"
        )
