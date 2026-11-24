"""Producer-stream snapshots and bounded, packed D2H for CUDA inference."""

from __future__ import annotations

import logging
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from math import prod

import torch

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PackedRows:
    fields: tuple
    sizes: tuple[int, ...]
    nbytes: int

    def views(self, storage):
        if storage.dtype != torch.uint8 or storage.shape != (self.nbytes,):
            raise ValueError("Invalid snapshot storage")
        return {
            name: part.view(dtype).view(shape)
            for (name, dtype, shape), part in zip(
                self.fields, storage.split(self.sizes), strict=True
            )
        }

    def gather(self, storage, *, sources, owned, physical, positions, tokens, hidden):
        fields = self.views(storage)
        fields["physical"].copy_(physical)
        fields["owned"].copy_(owned)
        if positions is not None:
            fields["positions"].copy_(positions)
            fields["tokens"].copy_(tokens)
        if hidden is not None:
            fields["hidden"].copy_(hidden.detach())
        # Layer-major storage gives index_select a contiguous destination.
        for source, dest in zip(sources, fields["kv"].unbind(0), strict=True):
            source_bytes = source[:, 0, :].view(torch.uint8)
            if source_bytes.shape[1] != dest.shape[1]:
                raise ValueError("MLA row width changed")
            torch.index_select(source_bytes, 0, physical, out=dest)


@lru_cache(maxsize=64)
def packed_rows(rows, layers, row_bytes, hidden_width, verify):
    # Descending element size keeps the typed views naturally aligned.
    fields = [("physical", torch.int64, (rows,))]
    if verify:
        fields.extend(
            [("positions", torch.int64, (rows,)), ("tokens", torch.int64, (rows,))]
        )
    if hidden_width:
        fields.append(("hidden", torch.bfloat16, (rows, hidden_width)))
    fields.extend(
        [("kv", torch.uint8, (layers, rows, row_bytes)), ("owned", torch.bool, (rows,))]
    )
    sizes = tuple(prod(shape) * dtype.itemsize for _, dtype, shape in fields)
    return PackedRows(tuple(fields), sizes, sum(sizes))


class CommitGate:
    """Two-barrier join: dispatch to consumer only once both DMA and commit resolve.

    Avoids blocking the single consumer thread waiting on scheduler-side commit
    resolution, eliminating head-of-line stalls, spurious timeouts, and shutdown
    hangs.
    """

    def __init__(self, deliver, fail):
        self._lock = threading.Lock()
        self._deliver = deliver
        self._fail = fail
        self._dma = None
        self._has_commit = False
        self._commit_payload = None
        self._done = False

    def on_dma(self, slot, nbytes, copied, dispatch_fn):
        to_fire = None
        with self._lock:
            if self._done:
                return
            self._dma = (slot, nbytes, copied, dispatch_fn)
            if self._has_commit:
                self._done = True
                to_fire = (self._dma, self._commit_payload)
        if to_fire is not None:
            self._fire(*to_fire)

    def on_commit(self, payload):
        to_fire = None
        with self._lock:
            if self._done:
                return
            self._has_commit = True
            self._commit_payload = payload
            if self._dma is not None:
                self._done = True
                to_fire = (self._dma, payload)
        if to_fire is not None:
            self._fire(*to_fire)

    def on_dma_failed(self, reason):
        with self._lock:
            if self._done:
                return
            self._done = True
        self._fail(reason)

    def _fire(self, dma, payload):
        slot, nbytes, _copied, dispatch_fn = dma
        deliver = self._deliver
        fail = self._fail
        dispatch_fn(
            slot,
            nbytes,
            lambda host: deliver(host, payload),
            fail,
        )


class SnapshotPool:
    """A slot belongs to one gather, DMA and consumer invocation, in that order."""

    def __init__(self, device, slot_bytes, slots, gpu_bytes, pinned_bytes):
        device = torch.device(device)
        if device.type != "cuda":
            raise ValueError("Snapshot offload requires CUDA")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        self.device = device
        self.copy_stream = torch.cuda.Stream(device=device)
        stride = (slot_bytes + 7) // 8 * 8
        # Allow a second slot's worth per in-flight gather for mapping scratch.
        count = min(slots, gpu_bytes // (2 * stride), pinned_bytes // stride)
        if count <= 0:
            raise ValueError("Insufficient GPU or pinned memory for one snapshot")
        self.slot_count = count
        self.slot_bytes = stride
        self.free = queue.LifoQueue(count)
        gpu = torch.empty((count, stride), dtype=torch.uint8, device=device)
        host = torch.empty(
            (count, stride), dtype=torch.uint8, device="cpu", pin_memory=True
        )
        for gpu_slot, host_slot in zip(gpu.unbind(), host.unbind(), strict=True):
            self.free.put_nowait((gpu_slot, host_slot, torch.cuda.Event()))
        torch.cuda.current_stream(device).synchronize()
        self.reads: list[tuple[torch.cuda.Event, torch.cuda.Stream]] = []
        self.lock = threading.RLock()
        self.gathering = False
        self.quarantine = []
        self.closed = False
        self.copy_worker = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="snapshot-d2h"
        )
        self.consumer = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="snapshot-consumer"
        )
        for worker in (self.copy_worker, self.consumer):
            worker.submit(torch.cuda.set_device, device).result()

    def _prune_reads_locked(self):
        self.reads = [(event, st) for event, st in self.reads if not event.query()]

    def reading(self):
        """Pin migratable KV storage only until its last source read finishes."""
        with self.lock:
            self._prune_reads_locked()
            return self.gathering or bool(self.reads)

    def fence_reuse(self, stream=None):
        target = torch.cuda.current_stream(self.device) if stream is None else stream
        with self.lock:
            self._prune_reads_locked()
            events = [event for event, _ in self.reads]
        for event in events:
            target.wait_event(event)

    def guard_release(self, allocator, *, forward_stream=None):
        """Protect KV pages being freed against non-stream-ordered writers (RDMA,
        async HiCache H2D).

        If every outstanding gather ran on ``forward_stream`` and the allocator
        supports staged reuse guards, attach the latest gather event with zero
        host sync; otherwise wait for in-flight gathers before pages return to
        the free list.
        """
        self.fence_reuse()
        with self.lock:
            self._prune_reads_locked()
            if not self.reads:
                return
            pending = list(self.reads)
        if (
            forward_stream is not None
            and allocator is not None
            and getattr(allocator, "supports_reuse_guard", False)
            and getattr(allocator, "need_sort", False)
            and hasattr(allocator, "set_reuse_guard_event")
            and all(st is forward_stream for _, st in pending)
        ):
            allocator.set_reuse_guard_event(pending[-1][0])
            return
        for event, _ in pending:
            event.synchronize()
        with self.lock:
            self._prune_reads_locked()

    def submit(self, nbytes, fill, deliver, fail, *, stream=None, gate=None):
        with self.lock:
            if self.closed:
                closed = True
            else:
                closed = False
        if closed:
            fail("offload closed")
            return False
        try:
            slot = self.free.get_nowait()
        except queue.Empty:
            fail("snapshot slots exhausted")
            return False
        gpu, _, _ = slot
        caller_stream = torch.cuda.current_stream(self.device)
        gather_stream = caller_stream if stream is None else stream
        try:
            if nbytes > gpu.numel():
                raise ValueError("Snapshot exceeds slot capacity")
            if gather_stream is not caller_stream:
                gather_stream.wait_stream(caller_stream)
            gathered = torch.cuda.Event()
            with self.lock:
                self._prune_reads_locked()
                self.gathering = True
                if gather_stream is not caller_stream:
                    with torch.cuda.stream(gather_stream):
                        fill(gpu[:nbytes], gather_stream)
                else:
                    fill(gpu[:nbytes], gather_stream)
                gathered.record(gather_stream)
                self.reads.append((gathered, gather_stream))
                self.gathering = False
            if gather_stream is not caller_stream:
                caller_stream.wait_event(gathered)
        except Exception:
            # A gather may already have launched. An unknown completion is fatal
            # to serving: source KV/Graph memory could otherwise be overwritten.
            try:
                gather_stream.synchronize()
            except Exception:
                with self.lock:
                    self.closed = True
                    self.quarantine.append((slot, fill))
                raise
            with self.lock:
                self.gathering = False
            self.free.put_nowait(slot)
            fail("snapshot gather failed")
            logger.exception("Snapshot gather failed")
            return False
        try:
            self.copy_worker.submit(
                self._copy, slot, nbytes, gathered, deliver, fail, gate
            )
        except Exception:
            try:
                gathered.synchronize()
            except Exception:
                with self.lock:
                    self.closed = True
                    self.quarantine.append(slot)
                raise
            with self.lock:
                self._prune_reads_locked()
                is_closed = self.closed
            self.free.put_nowait(slot)
            fail("offload closed" if is_closed else "D2H submission failed")
            logger.exception("Snapshot D2H worker submission failed")
            return False
        return True

    def _dispatch_deliver(self, slot, nbytes, deliver, fail):
        try:
            self.consumer.submit(self._deliver, slot, nbytes, deliver, fail)
        except Exception:
            _, _, copied = slot
            try:
                copied.synchronize()
            except Exception:
                with self.lock:
                    self.closed = True
                    self.quarantine.append(slot)
                fail("D2H completion unknown")
                logger.exception("Quarantined snapshot slot on consumer dispatch")
                return
            self.free.put_nowait(slot)
            fail("offload closed")

    @torch.inference_mode()
    def _copy(self, slot, nbytes, gathered, deliver, fail, gate=None):
        gpu, host, copied = slot
        try:
            with torch.cuda.stream(self.copy_stream):
                self.copy_stream.wait_event(gathered)
                host[:nbytes].copy_(gpu[:nbytes], non_blocking=True)
                copied.record(self.copy_stream)
            if gate is not None:
                gate.on_dma(slot, nbytes, copied, self._dispatch_deliver)
            else:
                self._dispatch_deliver(slot, nbytes, deliver, fail)
        except Exception:
            if gate is not None:
                gate.on_dma_failed("D2H setup failed")
            else:
                fail("D2H setup failed")
            logger.exception("Snapshot D2H submission failed")
            try:
                gathered.synchronize()
                self.copy_stream.synchronize()
            except Exception:
                with self.lock:
                    self.closed = True
                    self.quarantine.append(slot)
                return
            self.free.put_nowait(slot)

    def _deliver(self, slot, nbytes, deliver, fail):
        _, host, copied = slot
        try:
            copied.synchronize()
        except Exception:
            with self.lock:
                self.closed = True
                self.quarantine.append(slot)
            fail("D2H completion unknown")
            logger.exception("Quarantined snapshot slot")
            return
        try:
            deliver(host[:nbytes])
        except Exception as exc:
            fail(str(exc) or type(exc).__name__)
            logger.exception("Snapshot consumer failed")
        finally:
            self.free.put_nowait(slot)

    def submit_callback(self, fn):
        """Run a lightweight lifecycle callback on the consumer thread."""
        try:
            self.consumer.submit(fn)
            return True
        except Exception:
            return False

    def close(self):
        with self.lock:
            self.closed = True
        try:
            self.copy_worker.shutdown(wait=True)
            self.consumer.shutdown(wait=True)
            with self.lock:
                quarantined = bool(self.quarantine)
            if quarantined:
                raise RuntimeError("Snapshot completion unknown on shutdown")
        finally:
            with self.lock:
                self.free = queue.LifoQueue()
                self.reads.clear()
