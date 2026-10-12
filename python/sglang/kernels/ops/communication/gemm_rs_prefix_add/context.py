"""Persistent SM100 BF16 GEMM + reduce-scatter + prefix addition."""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

from sglang.kernels.ops.communication.inkling_all_reduce import STATE_SIZE, flags_numel

from .tile_order import build_tile_m_order
from .workspace import WorkspaceLayout


class GemmRSPrefixAdd:
    """Fused row-parallel Linear followed by local-shard prefix addition.

    ``input`` is BF16 [M,K], ``weight`` is BF16 [N,K], prefix is BF16 [M,N] or
    [M/P,N] and output is BF16 [M/P,N]. ``dispatch`` carries ``mode`` (``pull``:
    NVLS reduction, BF16 add; ``push``: ordered FP32 reduction/add, one BF16
    cast), the compiled ``kernel`` and the producer ``tile_order`` (``common``:
    same order on every rank; ``rotated``: start from each source rank's shard).
    Both modes share one symmetric arena sized for ``configs``.
    """

    @torch.inference_mode(False)
    def __init__(self, group, rank, world_size, max_m, n, device, configs):
        self.group, self.group_name = group, group.group_name
        self.rank, self.world_size = rank, world_size
        self.n, self.device = n, device
        self.layout = WorkspaceLayout(max_m, n, world_size, tuple(configs))
        self._prepared_kernel_grids = {}

        with torch.cuda.device(device):
            rank_device_uuids = [None] * world_size
            dist.all_gather_object(
                rank_device_uuids,
                torch.ops.mega_ops.sm100_device_uuid(device.index),
                group=group,
            )
            torch.ops.mega_ops.sm100_check_peer_access(device.index, rank_device_uuids)

            # One symmetric arena: [push receive planes | shared GEMM partial].
            partial_offset = self.layout.recv_data_numel
            self._symmetric_workspace = symm_mem.empty(
                self.layout.arena_numel, device=device, dtype=torch.bfloat16
            )
            self._push_receive_buffer = self._symmetric_workspace[:partial_offset]
            self._gemm_partial_output = self._symmetric_workspace[
                partial_offset : partial_offset + self.layout.partial_numel
            ]
            self._push_receive_buffer.zero_()
            payload_handle = symm_mem.rendezvous(
                self._symmetric_workspace, self.group_name
            )
            self._payload_peer_ptrs = payload_handle.buffer_ptrs_dev
            self._partial_multicast_ptr = (
                payload_handle.multicast_ptr
                + partial_offset * self._symmetric_workspace.element_size()
                if payload_handle.multicast_ptr
                else 0
            )

            # Pull publishes one ready plane per destination rank; push reads
            # only the first plane.
            self._tile_ready_flags = symm_mem.empty(
                self.layout.ready_numel * world_size, device=device, dtype=torch.uint32
            )
            self._tile_ready_flags.zero_()
            ready_handle = symm_mem.rendezvous(self._tile_ready_flags, self.group_name)
            self._ready_peer_ptrs = ready_handle.buffer_ptrs_dev
            self._ready_multicast_ptr = ready_handle.multicast_ptr
            self.supports_pull = bool(
                self._partial_multicast_ptr and self._ready_multicast_ptr
            )

            self._barrier_flags = symm_mem.empty(
                flags_numel(world_size), device=device, dtype=torch.uint32
            )
            self._barrier_flags.zero_()
            barrier_handle = symm_mem.rendezvous(self._barrier_flags, self.group_name)
            self._barrier_peer_ptrs = barrier_handle.buffer_ptrs_dev
            self._barrier_state = torch.zeros(
                STATE_SIZE, device=device, dtype=torch.uint32
            )
            # Publish zeroed flags and push receive planes before any peer can
            # launch. Later calls clear them on the device.
            torch.cuda.current_stream(device).synchronize()
            dist.barrier(group=group)

    @torch.inference_mode(False)
    def prepare(self, m, k, dispatch):
        key = (m, k, dispatch)
        if key in self._prepared_kernel_grids:
            return
        if dispatch.mode == "pull" and not self.supports_pull:
            raise RuntimeError("SM100 fused pull requires NVLS payload and ready flags")
        config = dispatch.kernel
        if not self.layout.can_fuse(m, k, config):
            raise ValueError(f"cannot fuse M={m} K={k} with {config.key}")
        tile_m_order = build_tile_m_order(
            m // self.world_size,
            self.world_size,
            config.tile_m,
            None if dispatch.tile_order == "common" else self.rank,
        )
        # Also primes the native per-stream occupancy cache on this stream.
        resident_capacity = torch.ops.mega_ops.sm100_resident_cluster_capacity(
            self._gemm_partial_output,
            config.cluster_hidden,
            dispatch.mode == "push",
            config.tile_m,
            config.tile_n,
        )
        self._prepared_kernel_grids[key] = (
            torch.tensor(tile_m_order, device=self.device, dtype=torch.int32),
            min(resident_capacity, self.layout.cluster_work(m, config)),
        )

    def forward(self, input, weight, prefix, output, dispatch):
        m, k = input.shape
        tile_m_order, num_clusters = self._prepared_kernel_grids[m, k, dispatch]
        config = dispatch.kernel
        shared_kernel_args = (
            self._tile_ready_flags,
            self._ready_peer_ptrs,
            self._barrier_state,
            self._barrier_flags,
            self._barrier_peer_ptrs,
            tile_m_order,
            self.rank,
            self.world_size,
            self.layout.tile_n_capacity,
            num_clusters,
            config.cluster_hidden,
            config.tile_m,
            config.tile_n,
        )
        if dispatch.mode == "pull":
            torch.ops.mega_ops.sm100_gemm_rs_prefix_add_pull(
                input,
                weight,
                prefix,
                output,
                self._gemm_partial_output,
                self._partial_multicast_ptr,
                self._ready_multicast_ptr,
                *shared_kernel_args,
            )
        else:
            torch.ops.mega_ops.sm100_gemm_rs_prefix_add_push(
                input,
                weight,
                prefix,
                output,
                self._gemm_partial_output,
                self._push_receive_buffer,
                self._payload_peer_ptrs,
                self.layout.plane_stride_elements,
                *shared_kernel_args,
            )
        return output

    __call__ = forward
