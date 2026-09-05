# SPDX-License-Identifier: Apache-2.0
"""Weights on FABRIC-exportable memory, so MNNVL can serve them.

mooncake's NVLink fabric transport serves a buffer by retaining the allocation
behind the pointer and exporting that as a fabric handle -- both cuMemCreate-only.
The torch caching allocator's default path is cudaMalloc, so a seed's weights
cannot be exported, and MNNVL is the only NVLink transport that reaches another
node. See ``nvlink_fabric_utils`` for how that failure presents.

This moves the weights onto a CUDA VMM reservation whose chunks are each one
``cuMemCreate(CU_MEM_HANDLE_TYPE_FABRIC)`` mapping.

Two invariants are what let mooncake serve them unmodified:

* **One registration per mapping.** ``registerLocalMemory`` calls
  ``cuMemGetAddressRange`` once and publishes *that* range, ignoring the length
  it was handed, so one registration spanning several mappings would publish
  only the first.
* **No tensor spans two mappings.** A client resolves a remote pointer by
  finding the single published buffer that wholly contains it, so a straddling
  tensor is unreachable even with both halves registered. Each arena extent is
  exactly one mapping and ``BumpArenaStub.malloc`` first-fits *within* one
  extent, which gives this for free -- provided a chunk is never smaller than
  the largest weight, which ``plan_chunk_bytes`` guarantees.

Migration is chunk-at-a-time on purpose. A seed also serves inference, so it
cannot hold a second copy of the weights: committing everything up front would
peak at twice the weight bytes. Filling one chunk, releasing the originals it
covers, then opening the next peaks at one chunk instead.
"""

from __future__ import annotations

import bisect
import logging
from typing import Optional

import torch

from sglang.srt.utils.cuda_vmm_utils import (
    BumpArenaStub,
    VmmReservation,
    align_up,
    allocation_handle_type_name,
    get_device_granularity,
    make_device_allocation_prop,
)

logger = logging.getLogger(__name__)

# Peak overhead during migration is one chunk, so this floor trades startup
# headroom against the number of mooncake registrations. A model whose largest
# weight exceeds it gets chunks sized to that weight instead.
_MIN_CHUNK_BYTES = 512 << 20

# torch's caching allocator asks the pluggable allocator for a rounded-up
# segment, not the tensor size (CUDACachingAllocator get_allocation_size). The
# bump allocator hands out whole requests, so packing has to be planned against
# these, or a chunk we believe has room returns nullptr and torch raises OOM
# in the middle of a migration.
_SMALL_SIZE = 1 << 20
_SMALL_BUFFER = 2 << 20
_MIN_LARGE_ALLOC = 10 << 20
_LARGE_BUFFER = 20 << 20
_ROUND_LARGE = 2 << 20


class FabricArenaUnavailable(RuntimeError):
    """These weights cannot be placed on memory MNNVL would be able to serve."""


def allocator_request_bytes(nbytes: int) -> int:
    """What torch's caching allocator asks the pluggable allocator for."""
    if nbytes <= _SMALL_SIZE:
        return _SMALL_BUFFER
    if nbytes < _MIN_LARGE_ALLOC:
        return _LARGE_BUFFER
    return align_up(nbytes, _ROUND_LARGE)


def plan_chunk_bytes(largest_weight_bytes: int, granularity: int) -> int:
    """Chunk size that can hold any single weight whole.

    A weight larger than a chunk could not be mapped by one handle, and a client
    cannot resolve a tensor that spans two published buffers -- so the largest
    weight, not a constant, sets the floor.
    """
    return align_up(
        max(_MIN_CHUNK_BYTES, allocator_request_bytes(largest_weight_bytes)),
        granularity,
    )


def plan_chunk_count(sizes: list[int], chunk_bytes: int) -> int:
    """How many chunks the migration will actually consume.

    Simulated rather than estimated from the tensor sum, because the rounding in
    ``allocator_request_bytes`` is not a small factor: every tensor in the
    1--10 MiB band costs a 20 MiB segment, so a model with many small biases and
    per-expert scales can need an order of magnitude more than its bytes. A
    reservation guessed low fails inside the migration, which is the one place
    that cannot be rolled back -- so this mirrors the real first-fit walk.
    """
    chunks = 1
    used = 0
    for nbytes in sizes:
        request = allocator_request_bytes(nbytes)
        if used + request > chunk_bytes:
            chunks += 1
            used = 0
        used += request
    return chunks


class WeightFabricArena:
    """A VA reservation handed out one FABRIC mapping at a time.

    ``mappings`` is the registration list: one ``(addr, length)`` per
    ``cuMemCreate``, which is exactly the granularity mooncake publishes at.
    """

    def __init__(self, device_id: int, sizes: list[int]) -> None:
        self.device_id = int(device_id)
        self.mappings: list[tuple[int, int]] = []
        self._closed = False

        with torch.cuda.device(self.device_id):
            # gpuDirectRDMACapable because a seed serves the same weights over the
            # NIC to peers outside its clique. Without it those pages may refuse
            # ibv_reg_mr, which would trade a working RDMA endpoint for MNNVL.
            prop = make_device_allocation_prop(self.device_id, gpu_direct_rdma=True)
            handle_name = allocation_handle_type_name(prop.requestedHandleTypes)
            if handle_name != "FABRIC":
                # POSIX_FD maps fine locally and is unexportable to the one peer
                # that matters, so refuse rather than build a seed whose MNNVL
                # offer fails at the client's first transfer.
                raise FabricArenaUnavailable(
                    f"GPU {self.device_id} allocates VMM memory as {handle_name}, "
                    f"not FABRIC; MNNVL could not export it."
                )

            self.granularity = get_device_granularity(self.device_id)
            self.chunk_bytes = plan_chunk_bytes(max(sizes), self.granularity)
            # One spare chunk beyond the simulated walk: the real migration frees
            # nothing back, so the two agree, but running out mid-migration is
            # unrecoverable and unmapped VA is free.
            chunks = plan_chunk_count(sizes, self.chunk_bytes) + 1
            reserve = chunks * self.chunk_bytes
            self._reservation = VmmReservation(
                reserve, prop, self.device_id, alignment=self.granularity
            )
            self.reserved_bytes = reserve
            self._next_offset = 0

        self._stub = BumpArenaStub()
        # Packing inside a chunk needs no granularity alignment -- only the
        # mapping does -- and aligning every allocation to 2 MiB would round each
        # of a model's many small tensors up to a full page.
        self._stub.set_align(512)
        self.pool = torch.cuda.MemPool(self._stub.allocator, no_split=True)
        logger.info(
            "WeightFabricArena[%s]: device=%d reserved_va=%.1f GiB chunk=%d MiB "
            "granularity=%d KiB handle_type=FABRIC",
            self._stub.sfx,
            self.device_id,
            reserve / (1024**3),
            self.chunk_bytes >> 20,
            self.granularity // 1024,
        )

    @property
    def committed_bytes(self) -> int:
        return sum(length for _, length in self.mappings)

    @property
    def chunk_used_bytes(self) -> int:
        """Bytes handed out of the currently installed chunk."""
        return self._stub.cursor_bytes

    def has_room_for(self, nbytes: int) -> bool:
        return (
            self.chunk_used_bytes + allocator_request_bytes(nbytes) <= self.chunk_bytes
        )

    def open_chunk(self) -> tuple[int, int]:
        """Back the next chunk and point the allocator at it alone.

        ``set_extents`` resets every bump cursor, which is why one chunk is
        installed at a time: a fresh chunk has nothing to preserve, while
        re-installing a partly-filled one would hand out live addresses again.
        """
        if self._closed:
            raise RuntimeError("WeightFabricArena.open_chunk after close")
        if self._next_offset + self.chunk_bytes > self.reserved_bytes:
            raise FabricArenaUnavailable(
                f"arena exhausted: {self._next_offset + self.chunk_bytes} bytes "
                f"needed past a {self.reserved_bytes}-byte reservation"
            )
        offset = self._next_offset
        with torch.cuda.device(self.device_id):
            self._reservation.map(offset, self.chunk_bytes, retain_handle=True)
        self._next_offset += self.chunk_bytes
        base = self._reservation.base + offset
        self._stub.set_extents([(base, self.chunk_bytes)])
        self.mappings.append((base, self.chunk_bytes))
        return base, self.chunk_bytes

    def assert_inside_current_chunk(self, name: str, addr: int, nbytes: int) -> None:
        """A tensor outside the open mapping is the one silent failure here.

        It would be published in the manifest, be covered by no registered
        buffer, and surface only as the client's "Requested address ... not
        found!" -- the very failure this arena exists to remove.
        """
        base, length = self.mappings[-1]
        if not (base <= addr and addr + nbytes <= base + length):
            raise FabricArenaUnavailable(
                f"weight {name!r} landed at {addr:#x}+{nbytes} outside the open "
                f"mapping [{base:#x}, {base + length:#x}); the allocator asked "
                f"for more than has_room_for predicted."
            )

    def warn_if_anything_was_released(self) -> None:
        """A freed arena block is leaked address space, not reusable memory.

        ``free`` is a no-op that only counts, so anything released -- an
        ``empty_cache`` reaching into this pool, a migrated tensor dropped on the
        floor -- silently shrinks what the chunk can still hold.
        """
        freed = self._stub.freed_bytes
        if freed:
            logger.warning(
                "WeightFabricArena[%s]: %d bytes were released back to the bump "
                "allocator, which cannot reuse them; the arena holds that much "
                "less than its reservation suggests.",
                self._stub.sfx,
                freed,
            )

    def close(self) -> None:
        """Only safe once nothing points into the arena -- it unmaps the weights."""
        if self._closed:
            return
        self._closed = True
        try:
            torch.cuda.synchronize()
        except Exception as e:  # pragma: no cover
            logger.warning("WeightFabricArena.close synchronize failed: %s", e)
        self._reservation.close()


def device_can_back_fabric_arena(device_id: int) -> bool:
    """Whether this device's VMM allocations are FABRIC-exportable.

    Asked before the weights exist, to decide whether offering MNNVL is worth
    attempting at all. Distinct from
    ``caching_allocator_memory_is_fabric_exportable``, which asks about the
    *default allocator's* memory -- the thing an arena exists to sidestep.

    Probes the same prop the arena will build, ``gpuDirectRDMACapable`` included,
    so a device that advertises FABRIC but not that combination is not counted.
    """
    try:
        if not torch.cuda.is_available():
            return False
        with torch.cuda.device(device_id):
            prop = make_device_allocation_prop(device_id, gpu_direct_rdma=True)
            return allocation_handle_type_name(prop.requestedHandleTypes) == "FABRIC"
    except Exception as e:
        logger.debug("Cannot probe FABRIC VMM support on GPU %s (%s).", device_id, e)
        return False


def names_outside_arena(
    manifest: dict[str, tuple[int, int, int]], arena: WeightFabricArena
) -> list[str]:
    """Manifest entries no registered mapping covers.

    Migration deduplicates by storage, so two distinct Parameters that happen to
    share one allocation leave only the survivor rebound; the other keeps the
    original storage alive by reference and stays outside the arena. Publishing
    it would reproduce the failure this arena removes, so the caller withdraws
    MNNVL and serves over the NIC instead.
    """
    mappings = sorted(arena.mappings)
    bases = [base for base, _ in mappings]
    outside = []
    for name, (pointer, numel, element_size) in manifest.items():
        nbytes = numel * element_size
        index = bisect.bisect_right(bases, pointer) - 1
        if index < 0:
            outside.append(name)
            continue
        base, length = mappings[index]
        if pointer + nbytes > base + length:
            outside.append(name)
    return outside


def migratable_parameters(model: torch.nn.Module) -> list[tuple[str, torch.Tensor]]:
    """Parameters worth moving, largest first, one entry per storage.

    Largest first so a chunk's tail is filled by tensors small enough to fit it
    rather than skipped by one that never will. Aliases share storage, so moving
    one and rebinding only that name would leave the others on freed memory.
    """
    by_storage: dict[int, tuple[str, torch.Tensor]] = {}
    for name, param in model.named_parameters(remove_duplicate=False):
        if param.device.type != "cuda":
            continue
        by_storage.setdefault(param.data_ptr(), (name, param))
    return sorted(
        by_storage.values(),
        key=lambda entry: entry[1].numel() * entry[1].element_size(),
        reverse=True,
    )


def reject_unmigratable(params: list[tuple[str, torch.Tensor]]) -> None:
    """Refuse before moving anything, because there is no way back.

    Once a parameter is rebound onto the arena, the arena cannot be closed
    without freeing memory the model is using. So everything that would make the
    migration incomplete has to be a refusal taken up front -- a partial arena
    publishes a manifest whose missing weights fail at the client.
    """
    for name, param in params:
        if not param.is_contiguous():
            raise FabricArenaUnavailable(
                f"weight {name!r} is not contiguous; copying it into the arena "
                f"would not reproduce its layout."
            )


def migrate_weights_to_fabric_arena(
    model: torch.nn.Module, device_id: int
) -> Optional[WeightFabricArena]:
    """Move ``model``'s weights onto FABRIC memory, in place.

    Returns the arena, whose ``mappings`` are what to register with mooncake, or
    None when these weights cannot be placed on such memory -- in which case
    nothing has been moved and the caller should not offer MNNVL.

    Rebinds ``param.data``, so this must run before anything captures a weight
    pointer: a CUDA graph recorded against the old storage would read freed
    memory, as would a module that cached ``.data`` rather than the Parameter.
    """
    params = migratable_parameters(model)
    if not params:
        return None
    sizes = [p.numel() * p.element_size() for _, p in params]

    try:
        reject_unmigratable(params)
        arena = WeightFabricArena(device_id, sizes)
        # Inside the recoverable block: the first cuMemCreate is where a device
        # that advertises FABRIC but cannot honour gpuDirectRDMACapable fails, and
        # nothing has been rebound yet, so that can still degrade to the NIC.
        arena.open_chunk()
    except FabricArenaUnavailable as e:
        logger.info("Not placing weights on FABRIC memory: %s", e)
        return None

    # Past this point a failure cannot be undone: parameters already rebound
    # point into the arena, so closing it would free live weights. Let the
    # exception reach startup instead -- a leaked reservation in a dying process
    # is the cheaper outcome.
    for (name, param), nbytes in zip(params, sizes):
        if not arena.has_room_for(nbytes):
            # The originals covered so far are dead; hand their pages back before
            # backing more, or the peak is the sum of both.
            torch.cuda.empty_cache()
            arena.open_chunk()
        with torch.cuda.use_mem_pool(arena.pool):
            replacement = torch.empty_like(param.data)
        arena.assert_inside_current_chunk(name, replacement.data_ptr(), nbytes)
        replacement.copy_(param.data)
        param.data = replacement
    torch.cuda.empty_cache()
    arena.warn_if_anything_was_released()

    logger.info(
        "Placed %.2f GiB of weights on %d FABRIC mapping(s) of %d MiB (%.2f GiB "
        "committed); MNNVL can serve them.",
        sum(sizes) / (1 << 30),
        len(arena.mappings),
        arena.chunk_bytes >> 20,
        arena.committed_bytes / (1 << 30),
    )
    return arena
