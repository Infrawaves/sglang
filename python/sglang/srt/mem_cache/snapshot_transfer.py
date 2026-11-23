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
        self.free = queue.LifoQueue(count)
        gpu = torch.empty((count, stride), dtype=torch.uint8, device=device)
        host = torch.empty(
            (count, stride), dtype=torch.uint8, device="cpu", pin_memory=True
        )
        for gpu_slot, host_slot in zip(gpu.unbind(), host.unbind(), strict=True):
            self.free.put_nowait((gpu_slot, host_slot, torch.cuda.Event()))
        torch.cuda.current_stream(device).synchronize()
        self.reads = []
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

    def reading(self):
        """Pin migratable KV storage only until its last source read finishes."""
        with self.lock:
            self.reads = [event for event in self.reads if not event.query()]
            return self.gathering or bool(self.reads)

    def fence_reuse(self):
        stream = torch.cuda.current_stream(self.device)
        with self.lock:
            self.reads = [event for event in self.reads if not event.query()]
            for event in self.reads:
                stream.wait_event(event)

    def submit(self, nbytes, fill, deliver, fail):
        if self.closed:
            fail("offload closed")
            return False
        try:
            slot = self.free.get_nowait()
        except queue.Empty:
            fail("snapshot slots exhausted")
            return False
        gpu, _, _ = slot
        stream = torch.cuda.current_stream(self.device)
        try:
            if nbytes > gpu.numel():
                raise ValueError("Snapshot exceeds slot capacity")
            gathered = torch.cuda.Event()
            with self.lock:
                self.reads = [event for event in self.reads if not event.query()]
                self.gathering = True
                fill(gpu[:nbytes], stream)
                gathered.record(stream)
                self.reads.append(gathered)
                self.gathering = False
            self.copy_worker.submit(self._copy, slot, nbytes, gathered, deliver, fail)
        except Exception:
            # A gather may already have launched. An unknown completion is fatal
            # to serving: source KV/Graph memory could otherwise be overwritten.
            try:
                stream.synchronize()
            except Exception:
                self.closed = True
                self.quarantine.append((slot, fill))
                raise
            self.free.put_nowait(slot)
            self.gathering = False
            fail("snapshot gather failed")
            logger.exception("Snapshot gather failed")
            return False
        return True

    @torch.inference_mode()
    def _copy(self, slot, nbytes, gathered, deliver, fail):
        gpu, host, copied = slot
        try:
            with torch.cuda.stream(self.copy_stream):
                self.copy_stream.wait_event(gathered)
                host[:nbytes].copy_(gpu[:nbytes], non_blocking=True)
                copied.record(self.copy_stream)
            self.consumer.submit(self._deliver, slot, nbytes, deliver, fail)
        except Exception:
            fail("D2H setup failed")
            logger.exception("Snapshot D2H submission failed")
            try:
                gathered.synchronize()
                self.copy_stream.synchronize()
            except Exception:
                self.closed = True
                self.quarantine.append(slot)
                return
            self.free.put_nowait(slot)

    def _deliver(self, slot, nbytes, deliver, fail):
        _, host, copied = slot
        try:
            copied.synchronize()
        except Exception:
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

    def close(self):
        self.closed = True
        self.copy_worker.shutdown(wait=True)
        self.consumer.shutdown(wait=True)
        if self.quarantine:
            raise RuntimeError("Snapshot completion unknown on shutdown")
        self.free = queue.LifoQueue()
        self.reads.clear()
