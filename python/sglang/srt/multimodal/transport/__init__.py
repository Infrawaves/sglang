"""GPU transports for multimodal feature tensors."""

from typing import Literal

import torch

from sglang.srt.runtime_context import get_parallel

TensorTransportMode = Literal["cuda_ipc", "auto", "default"]


def to_compact_cpu(tensor: torch.Tensor) -> torch.Tensor:
    """Return a CPU tensor that owns exactly its own bytes, for inline pickling.

    Pickling a tensor serializes its whole backing storage, not just the
    elements it views. A per-item feature split from one batched tensor (for
    example a CPU preprocessor's ``pixel_values.split(...)``) therefore carries
    the entire batch, once per item: N items pickle N copies of the batch, and
    unpickling rebuilds N separate storages. ``tensor.cpu()`` does not help for
    a CPU tensor -- it returns the same view -- so every path that hands a
    tensor to pickle compacts it here first. Tensors that already own their
    storage are returned unchanged, without a copy.
    """
    tensor = tensor.cpu()
    if (
        tensor.is_contiguous()
        and tensor.storage_offset() == 0
        and tensor.untyped_storage().nbytes() == tensor.numel() * tensor.element_size()
    ):
        return tensor
    return tensor.clone(memory_format=torch.contiguous_format)


def determine_tensor_transport_mode() -> TensorTransportMode:
    """Select tensor transport from node topology, not rendezvous configuration.

    A rendezvous address may be present for single-node TP. Conversely, Ray can
    inject the address only into scheduler actors for a multi-node deployment,
    and external launchers may use an environment-based rendezvous instead.
    """
    if get_parallel().nnodes > 1:
        # CUDA IPC and POSIX shared memory are local to one node.
        return "default"
    return "cuda_ipc"
