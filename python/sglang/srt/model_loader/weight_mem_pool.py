# SPDX-License-Identifier: Apache-2.0
"""A fabric-capable CUDA memory pool for model weights.

MNNVL serves a weight by exporting a fabric handle for the allocation holding
it, which the driver only grants for memory from ``cuMemCreate``. torch's
caching allocator hands out ``cudaMalloc`` memory, so
``cuMemRetainAllocationHandle`` refuses it -- and mooncake answers that refusal
with success after registering nothing, leaving the seed advertising weights it
cannot serve.

Strictly only a seed needs this, but chained loading makes a client the next
seed, so the knob is per process rather than per role.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Optional

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

NVLINK_WEIGHT_MEM_POOL = "NVLINK"
_SUPPORTED_POOLS = (NVLINK_WEIGHT_MEM_POOL,)

_MOONCAKE_VERSION_HINT = "mooncake-transfer-engine >= 0.3.3.post2"

# The pool owns the address reservations its weights are mapped into, so it has
# to outlive every tensor drawn from it -- collecting it would unmap memory the
# model is still reading.
_lock = threading.Lock()
_pool: Optional[Any] = None


def weight_mem_pool_requested() -> bool:
    """Whether the weights are to come from a fabric-capable pool.

    Answerable before the weights exist, which is what callers deciding on
    transports need: endpoints are built before load_model.
    """
    return envs.SGLANG_REMOTE_INSTANCE_WEIGHT_MEM_POOL.get() is not None


def maybe_init_weight_mem_pool() -> Optional[Any]:
    """Return a ``cuMemCreate``-backed MemPool for weights, or None if unasked.

    Raises when a pool was asked for and cannot be built: falling back to the
    default allocator would reproduce the silent failure this exists to prevent.
    """
    requested = envs.SGLANG_REMOTE_INSTANCE_WEIGHT_MEM_POOL.get()
    if requested is None:
        return None

    pool_type = requested.strip().upper()
    if pool_type not in _SUPPORTED_POOLS:
        raise ValueError(
            f"SGLANG_REMOTE_INSTANCE_WEIGHT_MEM_POOL={requested!r} is not one of "
            f"{_SUPPORTED_POOLS}."
        )

    global _pool
    with _lock:
        if _pool is None:
            _pool = _build_nvlink_mem_pool()
        return _pool


def _build_nvlink_mem_pool() -> Any:
    import torch

    try:
        from mooncake.allocator import NVLinkAllocator
    except ImportError as e:
        raise RuntimeError(
            "SGLANG_REMOTE_INSTANCE_WEIGHT_MEM_POOL=NVLINK needs mooncake's "
            f"fabric allocator, which this build does not expose ({e}). Install "
            f"{_MOONCAKE_VERSION_HINT}, or unset the variable to load weights "
            "over nvlink_intra (same host) or the NIC."
        ) from e

    # The current device, and taken as no argument: torch allocates on it, so a
    # pool built for any other would back weights that land elsewhere.
    device = torch.device("cuda", torch.cuda.current_device())
    _assert_fabric_allocation_works(device)

    allocator = NVLinkAllocator.get_allocator(device)
    pool = torch.cuda.MemPool(allocator.allocator())
    logger.info(
        "Allocating weights from mooncake's fabric memory pool on %s so MNNVL "
        "can export handles for them.",
        device,
    )
    return pool


def _assert_fabric_allocation_works(device: Any) -> None:
    """Fail before the pool is built if this GPU cannot back it.

    Deliberately not mooncake's own ``NVLinkAllocator.detect_mem_backend()``:
    that probe calls cuMemCreate with a hardcoded 4096 bytes, and the driver
    requires a multiple of the allocation granularity (2 MiB on GB300), so it
    answers USE_CUDAMALLOC on hardware where fabric allocation in fact works.
    mooncake's allocator itself rounds up correctly, so only its probe is
    unreliable -- this one rounds, and it also gives a fuller reason on failure.

    Takes the device rather than selecting one: ModelRunner.__init__ has already
    set it for this rank.
    """
    from sglang.srt.utils.cuda_vmm_utils import (
        allocation_handle_type_name,
        get_device_allocation_handle_type,
        is_gpu_fabric_ready,
    )

    fabric_ready = is_gpu_fabric_ready(device)
    try:
        from cuda.bindings import driver as drv

        wanted = drv.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC
        handle_type = get_device_allocation_handle_type(device.index)
    except Exception as e:
        raise RuntimeError(
            "SGLANG_REMOTE_INSTANCE_WEIGHT_MEM_POOL=NVLINK could not probe CUDA "
            f"VMM allocation support on {device} ({e}). Unset the variable to "
            "load weights over nvlink_intra (same host) or the NIC."
        ) from e

    if handle_type != wanted:
        raise RuntimeError(
            unusable_fabric_reason(
                device=device,
                handle_type_name=allocation_handle_type_name(handle_type),
                fabric_ready=fabric_ready,
            )
        )


def unusable_fabric_reason(
    *, device: Any, handle_type_name: str, fabric_ready: bool
) -> str:
    """Why this GPU cannot back the pool, in terms the operator can act on.

    The two causes need different fixes and look identical from the error alone:
    a GPU in a clique that still cannot allocate FABRIC is almost always an IMEX
    channel the process cannot see, while one that never joined has no MNNVL
    domain to join.
    """
    if fabric_ready:
        cause = (
            "This GPU has joined an NVLink fabric clique, so the IMEX channel is "
            "most likely not visible to this process: check that "
            "/dev/nvidia-caps-imex-channels/channel0 exists and, in a container, "
            "is passed through."
        )
    else:
        cause = (
            "This GPU has not joined an NVLink fabric clique, so there is no "
            "MNNVL domain here (x86 HGX nodes have none)."
        )
    return (
        "SGLANG_REMOTE_INSTANCE_WEIGHT_MEM_POOL=NVLINK needs a GPU that can "
        "allocate fabric-exportable memory, but the best allocation handle type "
        f"on {device} is {handle_type_name}. {cause} Unset the variable to load "
        "weights over nvlink_intra (same host) or the NIC."
    )
