# Copyright 2023-2025 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Same-node EPLB expert-weight moves over CUDA IPC.

A rebalance moves routed-expert weights between ranks.  Inside a node the
old updater still sent them through NCCL ``batch_isend_irecv``, which lazily
builds a P2P channel for every newly used peer pair and stalls both sides on
matched send/recv.  Here every rank instead *pulls* the experts it needs
straight out of a peer's weight storage with ``cudaMemcpyAsync`` over NVLink:

1. once per process, each rank exports a ``cudaIpcMemHandle`` for the
   allocation behind every routed-expert tensor and all ranks exchange them;
   peers map the handles into their own context (no extra context on the
   source GPU, unlike rebuilding a torch IPC tensor);
2. per layer, the destination copies ``peer_weight[src_slot]`` into its
   temp buffer, exactly where an NCCL recv would have landed;
3. a one-element all_reduce on the device stream acts as a GPU-side barrier:
   no rank overwrites its weights (buffer -> weight copy) until every peer's
   pulls of the *old* weights have finished.

Any failure (expandable segments, no P2P access, a multi-node flat topology,
etc.) is agreed on collectively and every rank falls back to NCCL, so ranks
never disagree on which collectives they issue.
"""

from __future__ import annotations

import ctypes
import logging
import socket
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.distributed

logger = logging.getLogger(__name__)

_CUDA_IPC_HANDLE_SIZE = 64
_CUDA_MEMCPY_DEFAULT = 4


# ----------------------------------------------------------------------------
# Dense flat views
# ----------------------------------------------------------------------------


def dense_flat_view(tensor: torch.Tensor) -> Optional[torch.Tensor]:
    """1-D view covering exactly the bytes of a dense (gap-free) tensor.

    A per-expert slice of a MegaMoE scale tensor is MN-major (column-major in
    the last two dims) but still dense.  Such a slice is not ``contiguous()``,
    which is what NCCL warns about, and it cannot be handed to a byte copy
    as-is.  When the slice is dense its storage span is exactly ``numel``
    elements starting at ``data_ptr``, so a flat 1-D view of that span is a
    faithful byte image of it.  Returns None for tensors with gaps.
    """
    if tensor.is_contiguous():
        return tensor.reshape(-1)
    dims = sorted(
        (stride, size)
        for size, stride in zip(tensor.shape, tensor.stride())
        if size != 1
    )
    expected_stride = 1
    for stride, size in dims:
        if stride != expected_stride:
            return None
        expected_stride *= size
    return tensor.as_strided(
        (tensor.numel(),), (1,), storage_offset=tensor.storage_offset()
    )


def p2p_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Tensor to hand to NCCL P2P: a dense flat view when one exists.

    Both peers slice the same layout (temp buffers are ``empty_like`` the
    weights), so flat views on both sides move identical bytes and silence
    the non-contiguous P2P warning.
    """
    flat = dense_flat_view(tensor)
    return flat if flat is not None else tensor


# ----------------------------------------------------------------------------
# CUDA runtime / driver bindings (ctypes; no compilation needed)
# ----------------------------------------------------------------------------


class _CudaApi:
    """The handful of runtime/driver calls this module needs."""

    def __init__(self):
        from sglang.srt.distributed.device_communicators.cuda_wrapper import (
            CudaRTLibrary,
            cudaIpcMemHandle_t,
        )

        self._rt = CudaRTLibrary()
        self.handle_t = cudaIpcMemHandle_t
        lib = self._rt.lib
        self._memcpy_async = lib.cudaMemcpyAsync
        self._memcpy_async.restype = ctypes.c_int
        self._memcpy_async.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        self._driver = ctypes.CDLL("libcuda.so.1")
        self._get_address_range = self._driver.cuMemGetAddressRange_v2
        self._get_address_range.restype = ctypes.c_int
        self._get_address_range.argtypes = [
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.c_uint64,
        ]

    def allocation_base(self, ptr: int) -> Tuple[int, int]:
        base = ctypes.c_uint64()
        size = ctypes.c_size_t()
        result = self._get_address_range(
            ctypes.byref(base), ctypes.byref(size), ctypes.c_uint64(ptr)
        )
        if result != 0:
            raise RuntimeError(f"cuMemGetAddressRange failed with {result}")
        return base.value, size.value

    def export_handle(self, base: int) -> bytes:
        handle = self._rt.cudaIpcGetMemHandle(ctypes.c_void_p(base))
        return bytes(ctypes.string_at(ctypes.addressof(handle), _CUDA_IPC_HANDLE_SIZE))

    def open_handle(self, handle_bytes: bytes) -> int:
        handle = self.handle_t()
        ctypes.memmove(ctypes.addressof(handle), handle_bytes, _CUDA_IPC_HANDLE_SIZE)
        return self._rt.cudaIpcOpenMemHandle(handle).value

    def memcpy_async(self, dst: int, src: int, nbytes: int, stream: int) -> None:
        result = self._memcpy_async(
            ctypes.c_void_p(dst),
            ctypes.c_void_p(src),
            ctypes.c_size_t(nbytes),
            _CUDA_MEMCPY_DEFAULT,
            ctypes.c_void_p(stream),
        )
        if result != 0:
            raise RuntimeError(
                f"cudaMemcpyAsync failed: {self._rt.cudaGetErrorString(result)}"
            )


# ----------------------------------------------------------------------------
# Peer weight table
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class _TensorExport:
    """How a peer locates one local tensor: allocation handle + offset."""

    handle: bytes
    offset: int  # bytes from the allocation base to tensor.data_ptr()
    shape: Tuple[int, ...]
    stride: Tuple[int, ...]
    dtype: str
    expert_nbytes: int  # bytes of one dense per-expert slice
    expert_stride_bytes: int  # bytes between consecutive physical experts


def _tensor_signature(tensor: torch.Tensor) -> Tuple:
    return tuple(tensor.shape), tuple(tensor.stride()), str(tensor.dtype)


class IpcPeerWeights:
    """Mapped views of every same-node rank's routed-expert weights.

    Built once (collectively) and reused by every rebalance; weight storage
    is persistent after load, so the mapped addresses stay valid.  Mapped
    handles are never closed (they live as long as the weights).
    """

    def __init__(
        self,
        routed_experts_weights_of_layer: Dict[int, List[torch.Tensor]],
        *,
        rank: int,
        world_size: int,
        num_gpu_per_node: int,
    ):
        self._rank = rank
        self._world_size = world_size
        self._num_gpu_per_node = num_gpu_per_node
        self._node_id = rank // num_gpu_per_node
        self._api: Optional[_CudaApi] = None
        # (layer_id, tensor_index) -> peer rank -> mapped base address of the
        # peer's tensor (data_ptr equivalent in this process).
        self._peer_ptr: Dict[Tuple[int, int], Dict[int, int]] = {}
        self._expert_nbytes: Dict[Tuple[int, int], int] = {}
        self._expert_stride_bytes: Dict[Tuple[int, int], int] = {}
        self._local_ptrs = self._local_ptr_signature(routed_experts_weights_of_layer)
        self._opened: Dict[bytes, int] = {}
        self.enabled = self._build(routed_experts_weights_of_layer)

    @staticmethod
    def _local_ptr_signature(weights_of_layer) -> Tuple:
        return tuple(
            (layer_id, tuple(t.data_ptr() for t in tensors))
            for layer_id, tensors in sorted(weights_of_layer.items())
        )

    def matches(self, weights_of_layer) -> bool:
        return self._local_ptr_signature(weights_of_layer) == self._local_ptrs

    # -- construction -------------------------------------------------------

    def _export_local(self, weights_of_layer) -> Dict[Tuple[int, int], _TensorExport]:
        exports = {}
        handle_of_base: Dict[int, bytes] = {}
        for layer_id, tensors in weights_of_layer.items():
            for tensor_index, tensor in enumerate(tensors):
                if not tensor.is_cuda or tensor.shape[0] == 0:
                    raise RuntimeError("routed expert tensor is not a CUDA tensor")
                expert_slice = tensor[0]
                if dense_flat_view(expert_slice) is None:
                    raise RuntimeError(
                        f"layer {layer_id} tensor {tensor_index} has a non-dense "
                        f"per-expert layout {tuple(tensor.stride())}"
                    )
                ptr = tensor.data_ptr()
                base, _ = self._api.allocation_base(ptr)
                if base not in handle_of_base:
                    handle_of_base[base] = self._api.export_handle(base)
                exports[(layer_id, tensor_index)] = _TensorExport(
                    handle=handle_of_base[base],
                    offset=ptr - base,
                    shape=tuple(tensor.shape),
                    stride=tuple(tensor.stride()),
                    dtype=str(tensor.dtype),
                    expert_nbytes=expert_slice.numel() * tensor.element_size(),
                    expert_stride_bytes=tensor.stride(0) * tensor.element_size(),
                )
        return exports

    def _build(self, weights_of_layer) -> bool:
        local_ok = True
        exports: Dict[Tuple[int, int], _TensorExport] = {}
        try:
            self._api = _CudaApi()
            exports = self._export_local(weights_of_layer)
        except Exception as e:  # pragma: no cover - environment dependent
            logger.warning(f"[EPLB IPC] cannot export local expert weights: {e}")
            local_ok = False

        payload = (socket.gethostname(), local_ok, exports)
        gathered: List = [None] * self._world_size
        torch.distributed.all_gather_object(
            gathered, payload, group=_cpu_group_or_none()
        )

        if local_ok:
            try:
                self._open_peers(gathered, weights_of_layer)
            except Exception as e:  # pragma: no cover - environment dependent
                logger.warning(f"[EPLB IPC] cannot map peer expert weights: {e}")
                local_ok = False

        # Every rank must agree, otherwise ranks would issue different
        # collectives during the update.
        flag = torch.tensor(
            [1 if local_ok else 0],
            dtype=torch.int32,
            device=torch.cuda.current_device(),
        )
        torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MIN)
        enabled = bool(flag.item())
        if self._rank == 0:
            logger.info(
                "[EPLB IPC] same-node expert weight transfer "
                + ("enabled" if enabled else "disabled, falling back to NCCL P2P")
            )
        return enabled

    def _open_peers(self, gathered, weights_of_layer) -> None:
        my_host = gathered[self._rank][0]
        for peer in self._same_node_peers():
            host, peer_ok, peer_exports = gathered[peer]
            if host != my_host:
                raise RuntimeError(
                    f"rank {peer} is on {host}, not {my_host}; topology "
                    "groups it into this node"
                )
            if not peer_ok:
                raise RuntimeError(f"rank {peer} could not export its weights")
            for (layer_id, tensor_index), export in peer_exports.items():
                local = weights_of_layer[layer_id][tensor_index]
                if (export.shape, export.stride, export.dtype) != _tensor_signature(
                    local
                ):
                    raise RuntimeError(
                        f"layer {layer_id} tensor {tensor_index} layout differs "
                        f"on rank {peer}"
                    )
                if export.handle not in self._opened:
                    self._opened[export.handle] = self._api.open_handle(export.handle)
                key = (layer_id, tensor_index)
                self._peer_ptr.setdefault(key, {})[peer] = (
                    self._opened[export.handle] + export.offset
                )
                self._expert_nbytes[key] = export.expert_nbytes
                self._expert_stride_bytes[key] = export.expert_stride_bytes

    def _same_node_peers(self) -> List[int]:
        start = self._node_id * self._num_gpu_per_node
        return [
            r
            for r in range(start, min(start + self._num_gpu_per_node, self._world_size))
            if r != self._rank
        ]

    def is_same_node(self, peer: int) -> bool:
        return peer // self._num_gpu_per_node == self._node_id

    # -- transfer -----------------------------------------------------------

    def pull(
        self,
        *,
        layer_id: int,
        dst_tensors: Sequence[torch.Tensor],
        src_rank: int,
        src_local_index: int,
    ) -> int:
        """Copy one physical expert of ``src_rank`` into ``dst_tensors``.

        ``dst_tensors[i]`` is the destination slice for tensor ``i`` (a temp
        buffer row, same layout as the source).  Returns bytes moved.
        """
        stream = torch.cuda.current_stream().cuda_stream
        moved = 0
        for tensor_index, dst in enumerate(dst_tensors):
            key = (layer_id, tensor_index)
            nbytes = self._expert_nbytes[key]
            flat_dst = dense_flat_view(dst)
            assert (
                flat_dst is not None
                and flat_dst.numel() * flat_dst.element_size() == nbytes
            ), f"{key=} destination slice layout does not match the export"
            # A dense slice with positive strides starts at its data_ptr, so
            # the peer's expert k occupies [base + k * stride0, + nbytes).
            src_ptr = (
                self._peer_ptr[key][src_rank]
                + src_local_index * self._expert_stride_bytes[key]
            )
            self._api.memcpy_async(flat_dst.data_ptr(), src_ptr, nbytes, stream)
            moved += nbytes
        return moved


def _cpu_group_or_none():
    try:
        from sglang.srt.runtime_context import get_parallel

        return get_parallel().world_group.cpu_group
    except Exception:
        return None


def gpu_barrier() -> None:
    """Device-side barrier: a 1-element all_reduce on the current stream.

    The collective cannot complete on any rank before every rank has
    reached it, and each rank enqueues it after its IPC pulls, so once it
    retires on this stream every peer's reads of our old weights are done.
    """
    torch.distributed.all_reduce(
        torch.zeros(1, dtype=torch.int32, device=torch.cuda.current_device())
    )
