"""Copy-engine all-gather overlapped with a persistent SM100 BF16 GEMM."""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

from sglang.kernels.ops.communication.sm100_peer_access import (
    check_sm100_peer_access,
)
from sglang.srt.utils.nvtx_utils import operations_nvtx_range

from .workspace import WorkspaceLayout

# Matches kBarrierFlagWords / kBarrierStateWords in sm100_ag_gemm_comm.cuh.
_BARRIER_FLAG_WORDS = 2
_BARRIER_STATE_WORDS = 1


class AgGemm:
    """All-gather token shards and multiply by a column-parallel weight.

    ``local_input`` is BF16 [M/P,K], ``weight`` is BF16 [N,K] and output is BF16
    [M,N] in rank-major row order. Shards are packed by the live M, so
    ``gathered_input(m)`` is the contiguous all-gathered [M,K] input for the
    other consumers of the same rows. Writing ``local_input_shard(m)`` directly
    skips the kernel's local staging copy. One symmetric arena sized for
    ``max_m`` serves every N and every kernel in ``configs``.
    """

    @torch.inference_mode(False)
    def __init__(self, group, rank, world_size, max_m, k, device, configs):
        self.group, self.group_name = group, group.group_name
        self.rank, self.world_size = rank, world_size
        self.k, self.device = k, device
        self.layout = WorkspaceLayout(max_m, k, world_size, tuple(configs))
        self._prepared_kernel_grids = {}

        with torch.cuda.device(device):
            check_sm100_peer_access(group, device)

            self._symmetric_workspace = symm_mem.empty(
                self.layout.gathered_input_numel, device=device, dtype=torch.bfloat16
            )
            payload_handle = symm_mem.rendezvous(
                self._symmetric_workspace, self.group_name
            )
            # The host launcher issues the copy-engine peer pulls from these.
            self._payload_peer_ptrs = tuple(
                int(ptr) for ptr in payload_handle.buffer_ptrs
            )

            self._barrier_flags = symm_mem.empty(
                _BARRIER_FLAG_WORDS, device=device, dtype=torch.uint32
            )
            self._barrier_flags.zero_()
            barrier_handle = symm_mem.rendezvous(self._barrier_flags, self.group_name)
            self._barrier_peer_ptrs = torch.tensor(
                [int(ptr) for ptr in barrier_handle.buffer_ptrs],
                device=device,
                dtype=torch.int64,
            )
            self._shard_ready_flags = torch.zeros(
                world_size, device=device, dtype=torch.uint32
            )
            self._barrier_state = torch.zeros(
                _BARRIER_STATE_WORDS, device=device, dtype=torch.uint32
            )

            # The copy stream must be non-blocking and distinct from compute.
            self._copy_stream = torch.cuda.Stream(device=device)
            self._copy_start_event = torch.cuda.Event()
            self._copy_complete_event = torch.cuda.Event()
            # Events are created lazily; record both once so the raw handles
            # passed to the host launcher are valid.
            initialization_stream = torch.cuda.current_stream(device)
            self._copy_start_event.record(initialization_stream)
            self._copy_stream.wait_event(self._copy_start_event)
            self._copy_complete_event.record(self._copy_stream)
            initialization_stream.wait_event(self._copy_complete_event)
            self._copy_handles = (
                self._copy_stream.cuda_stream,
                self._copy_start_event.cuda_event,
                self._copy_complete_event.cuda_event,
            )
            # Publish zeroed flags before any peer can launch. Epochs advance
            # on the device afterwards and are never cleared.
            initialization_stream.synchronize()
            dist.barrier(group=group)

    def input_shards(self, m):
        # [P, M/P, K]: the kernel's source stride is the live shard size.
        num_shard_tokens = m // self.world_size
        return self._symmetric_workspace[: m * self.k].view(
            self.world_size, num_shard_tokens, self.k
        )

    def local_input_shard(self, m):
        return self.input_shards(m)[self.rank]

    def gathered_input(self, m):
        return self._symmetric_workspace[: m * self.k].view(m, self.k)

    @torch.inference_mode(False)
    def prepare(self, m, n, dispatch):
        key = (m, n, dispatch)
        if key in self._prepared_kernel_grids:
            return
        config = dispatch.kernel
        if not self.layout.can_fuse(m, n, config):
            raise ValueError(f"cannot fuse M={m} N={n} with {config.key}")
        # Also primes the native per-stream occupancy cache on this stream.
        resident_capacity = torch.ops.mega_ops.sm100_ag_gemm_resident_cluster_capacity(
            self._symmetric_workspace,
            config.tile_m,
            config.cluster_hidden,
            self._copy_handles[0],
        )
        self._prepared_kernel_grids[key] = min(
            resident_capacity, self.layout.cluster_work(m, n, config)
        )

    def forward(self, local_input, weight, output, dispatch):
        num_shard_tokens = local_input.shape[0]
        m, n = num_shard_tokens * self.world_size, weight.shape[0]
        num_clusters = self._prepared_kernel_grids[m, n, dispatch]
        config = dispatch.kernel
        # label the launch for torch profiler and Nsight Systems traces.
        with operations_nvtx_range("mega_ops::sm100_ag_gemm_pull"):
            torch.ops.mega_ops.sm100_ag_gemm_pull(
                local_input,
                weight,
                output,
                self.input_shards(m),
                self._payload_peer_ptrs,
                self._shard_ready_flags,
                self._barrier_state,
                self._barrier_flags,
                self._barrier_peer_ptrs,
                self.rank,
                self.world_size,
                config.tile_m,
                config.cluster_hidden,
                num_clusters,
                *self._copy_handles,
            )
        return output

    __call__ = forward
