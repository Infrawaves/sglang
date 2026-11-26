"""Side-stream compact MLA KV gather and zero-HBM async hidden D2H pools."""

from __future__ import annotations

import logging
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:
    triton = None
    tl = None
    _HAS_TRITON = False


def _same_stream(a, b) -> bool:
    if a is b:
        return True
    ha = getattr(a, "cuda_stream", None)
    hb = getattr(b, "cuda_stream", None)
    return ha is not None and ha == hb


if _HAS_TRITON:

    @triton.jit
    def _compact_kv_gather_kernel(
        owned_ptr,
        phys_ptr,
        layer_ptrs_ptr,
        layer_strides_ptr,
        meta_ptr,
        out_idx_ptr,
        out_phys_ptr,
        out_kv_ptr,
        n_tokens,
        n_layers,
        row_bytes,
        max_owned,
        BLOCK_B: tl.constexpr,
    ):
        t = tl.program_id(0)
        if t >= n_tokens:
            return
        is_own = tl.load(owned_ptr + t)
        if is_own == 0:
            return
        slot = tl.atomic_add(meta_ptr, 1)
        if slot >= max_owned:
            tl.store(meta_ptr + 1, 1)
            return
        slot_i64 = slot.to(tl.int64)
        phys = tl.load(phys_ptr + t).to(tl.int64)
        tl.store(out_idx_ptr + slot_i64, t.to(tl.int64))
        tl.store(out_phys_ptr + slot_i64, phys)
        for layer_idx in range(0, n_layers):
            src_base = tl.load(layer_ptrs_ptr + layer_idx).to(
                tl.pointer_type(tl.uint8)
            )
            src_stride = tl.load(layer_strides_ptr + layer_idx).to(tl.int64)
            src_row = src_base + phys * src_stride
            dst_row = out_kv_ptr + (slot_i64 * n_layers + layer_idx) * row_bytes
            for off in range(0, row_bytes, BLOCK_B):
                b = off + tl.arange(0, BLOCK_B)
                mask = b < row_bytes
                val = tl.load(src_row + b, mask=mask)
                tl.store(dst_row + b, val, mask=mask)


@dataclass(frozen=True)
class CompactKVPlan:
    """Aligned layout for one compacted owned-only KV snapshot slot.

    Memory layout inside a contiguous uint8 buffer:
      - meta:     [4] int32          -> (n_owned, overflow_flag, 0, 0) (16B aligned)
      - orig_idx: [max_owned] int64  -> relative token offset in [0, chunk_len)
      - physical: [max_owned] int64  -> translated physical KV slot ID
      - kv:       [max_owned, n_layers, row_bytes] uint8 (row-major!)
    """

    max_owned: int
    n_layers: int
    row_bytes: int
    meta_off: int
    idx_off: int
    phys_off: int
    kv_off: int
    nbytes: int

    @classmethod
    def create(cls, max_owned: int, n_layers: int, row_bytes: int) -> "CompactKVPlan":
        meta_off = 0
        idx_off = 16
        phys_off = idx_off + max_owned * 8
        kv_off = phys_off + max_owned * 8
        nbytes = kv_off + max_owned * n_layers * row_bytes
        return cls(
            max_owned=max_owned,
            n_layers=n_layers,
            row_bytes=row_bytes,
            meta_off=meta_off,
            idx_off=idx_off,
            phys_off=phys_off,
            kv_off=kv_off,
            nbytes=nbytes,
        )

    def views(self, storage: torch.Tensor) -> dict[str, torch.Tensor]:
        if storage.dtype != torch.uint8 or storage.numel() < self.nbytes:
            raise ValueError("Invalid compact KV snapshot storage")
        meta = storage[self.meta_off : self.meta_off + 16].view(torch.int32)
        orig_idx = storage[
            self.idx_off : self.idx_off + self.max_owned * 8
        ].view(torch.int64)
        physical = storage[
            self.phys_off : self.phys_off + self.max_owned * 8
        ].view(torch.int64)
        kv = storage[
            self.kv_off : self.kv_off + self.max_owned * self.n_layers * self.row_bytes
        ].view(self.max_owned, self.n_layers, self.row_bytes)
        return {
            "meta": meta,
            "orig_idx": orig_idx,
            "physical": physical,
            "kv": kv,
        }

    def gather(
        self,
        storage: torch.Tensor,
        *,
        sources: tuple[torch.Tensor, ...],
        owned: torch.Tensor,
        physical: torch.Tensor,
        layer_ptrs_gpu: torch.Tensor | None = None,
        layer_strides_gpu: torch.Tensor | None = None,
    ) -> None:
        fields = self.views(storage)
        meta = fields["meta"]
        meta.zero_()
        n_tokens = int(owned.numel())
        if n_tokens == 0:
            return
        if (
            storage.is_cuda
            and _HAS_TRITON
            and layer_ptrs_gpu is not None
            and layer_strides_gpu is not None
        ):
            owned_u8 = owned.to(torch.uint8).contiguous()
            phys_i64 = physical.to(torch.int64).contiguous()
            _compact_kv_gather_kernel[(n_tokens,)](
                owned_u8,
                phys_i64,
                layer_ptrs_gpu,
                layer_strides_gpu,
                meta,
                fields["orig_idx"],
                fields["physical"],
                fields["kv"],
                n_tokens,
                self.n_layers,
                self.row_bytes,
                self.max_owned,
                BLOCK_B=256,
            )
            return

        # CPU / non-Triton fallback (used by CPU unit tests; mirrors exact binary layout).
        owned_idx = torch.nonzero(owned, as_tuple=False).reshape(-1)
        count = int(owned_idx.numel())
        if count > self.max_owned:
            meta[0] = count
            meta[1] = 1
            return
        meta[0] = count
        meta[1] = 0
        if count == 0:
            return
        sel_phys = physical[owned_idx].to(torch.int64)
        fields["orig_idx"][:count].copy_(owned_idx.to(torch.int64))
        fields["physical"][:count].copy_(sel_phys)
        for layer_idx, source in enumerate(sources):
            src_bytes = source[:, 0, :].view(torch.uint8)
            if src_bytes.shape[1] != self.row_bytes:
                raise ValueError("MLA row width changed")
            fields["kv"][:count, layer_idx, :].copy_(
                torch.index_select(src_bytes, 0, sel_phys)
            )


class KVSnapshotPool:
    """Bounded side-stream pool for one-shot compact 1/8 owned KV gather + async D2H."""

    def __init__(
        self,
        device,
        slot_bytes: int,
        slots: int,
        gpu_bytes: int,
        pinned_bytes: int,
        *,
        consumer: ThreadPoolExecutor | None = None,
    ) -> None:
        device = torch.device(device)
        if device.type != "cuda":
            raise ValueError("KVSnapshotPool requires CUDA")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        self.device = device
        self.kv_stream = torch.cuda.Stream(device=device)
        stride = (slot_bytes + 7) // 8 * 8
        count = min(slots, gpu_bytes // stride, pinned_bytes // stride)
        if count <= 0:
            raise ValueError("Insufficient GPU or pinned memory for one KV snapshot slot")
        self.slot_count = count
        self.slot_bytes = stride
        self.free: queue.LifoQueue = queue.LifoQueue(count)
        gpu = torch.empty((count, stride), dtype=torch.uint8, device=device)
        host = torch.empty(
            (count, stride), dtype=torch.uint8, device="cpu", pin_memory=True
        )
        for gpu_slot, host_slot in zip(gpu.unbind(), host.unbind(), strict=True):
            self.free.put_nowait((gpu_slot, host_slot, torch.cuda.Event()))
        self.reads: list[tuple[torch.cuda.Event, torch.cuda.Stream]] = []
        self.lock = threading.RLock()
        self.gathering = False
        self.quarantine: list = []
        self.closed = False
        self._owns_consumer = consumer is None
        self.consumer = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="dflash-offload")
            if consumer is None
            else consumer
        )
        if self._owns_consumer:
            self.consumer.submit(torch.cuda.set_device, device).result()

    def _prune_reads_locked(self) -> None:
        self.reads = [(ev, st) for ev, st in self.reads if not ev.query()]

    def reading(self) -> bool:
        """Pin migratable KV storage while any side-stream KV gather is in flight."""
        with self.lock:
            self._prune_reads_locked()
            return self.gathering or bool(self.reads)

    def guard_release(self, allocator) -> None:
        """Protect released KV pages against reuse until side-stream gathers finish.

        Only attaches the latest in-flight gather event when it has not already
        completed. Uses ``allocator.set_reuse_guard_event()`` when supported so
        there is zero host synchronization on the release path.
        """
        with self.lock:
            self._prune_reads_locked()
            if not self.reads:
                return
            pending = list(self.reads)
        latest_ev = pending[-1][0]
        if latest_ev.query():
            with self.lock:
                self._prune_reads_locked()
            return
        if (
            allocator is not None
            and getattr(allocator, "supports_reuse_guard", False)
            and getattr(allocator, "need_sort", False)
            and hasattr(allocator, "set_reuse_guard_event")
            and all(_same_stream(st, self.kv_stream) for _, st in pending)
        ):
            allocator.set_reuse_guard_event(latest_ev)
            return
        for event, _ in pending:
            event.synchronize()
        with self.lock:
            self._prune_reads_locked()

    @torch.inference_mode()
    def submit(
        self,
        nbytes: int,
        fill,
        deliver,
        fail,
        *,
        wait_streams=(),
    ) -> bool:
        with self.lock:
            if self.closed:
                fail("offload closed")
                return False
        try:
            slot = self.free.get_nowait()
        except queue.Empty:
            fail("kv snapshot slots exhausted")
            return False
        gpu, host, copied = slot
        streams_to_wait = tuple(s for s in wait_streams if s is not None)
        if not streams_to_wait:
            streams_to_wait = (torch.cuda.current_stream(self.device),)
        try:
            if nbytes > gpu.numel():
                raise ValueError("KV snapshot exceeds slot capacity")
            gathered = torch.cuda.Event()
            with self.lock:
                self._prune_reads_locked()
                self.gathering = True
                for st in streams_to_wait:
                    if not _same_stream(self.kv_stream, st):
                        self.kv_stream.wait_stream(st)
                with torch.cuda.stream(self.kv_stream):
                    fill(gpu[:nbytes], self.kv_stream)
                    gathered.record(self.kv_stream)
                    host[:nbytes].copy_(gpu[:nbytes], non_blocking=True)
                    copied.record(self.kv_stream)
                self.reads = [
                    (ev, st)
                    for ev, st in self.reads
                    if not _same_stream(st, self.kv_stream)
                ]
                self.reads.append((gathered, self.kv_stream))
                self.gathering = False
        except Exception:
            try:
                self.kv_stream.synchronize()
            except Exception:
                with self.lock:
                    self.closed = True
                    self.quarantine.append(slot)
                fail("kv snapshot gather failed")
                logger.exception("Quarantined KV snapshot slot after gather error")
                return False
            with self.lock:
                self.gathering = False
            self.free.put_nowait(slot)
            fail("kv snapshot gather failed")
            logger.exception("KV snapshot gather failed")
            return False

        try:
            self.consumer.submit(self._deliver, slot, nbytes, deliver, fail)
        except Exception:
            try:
                copied.synchronize()
            except Exception:
                with self.lock:
                    self.closed = True
                    self.quarantine.append(slot)
                fail("D2H completion unknown")
                return False
            self.free.put_nowait(slot)
            fail("offload closed")
            return False
        return True

    def _deliver(self, slot, nbytes: int, deliver, fail) -> None:
        _, host, copied = slot
        try:
            copied.synchronize()
        except Exception:
            with self.lock:
                self.closed = True
                self.quarantine.append(slot)
            fail("D2H completion unknown")
            logger.exception("Quarantined KV snapshot slot")
            return
        try:
            deliver(host[:nbytes])
        except Exception as exc:
            fail(str(exc) or type(exc).__name__)
            logger.exception("KV snapshot consumer failed")
        finally:
            self.free.put_nowait(slot)

    def submit_callback(self, fn) -> bool:
        try:
            self.consumer.submit(fn)
            return True
        except Exception:
            return False

    def close(self) -> None:
        with self.lock:
            self.closed = True
        try:
            if self._owns_consumer:
                self.consumer.shutdown(wait=True)
        finally:
            with self.lock:
                self.free = queue.LifoQueue()
                self.reads.clear()


class HiddenD2HPool:
    """Zero-HBM async D2H pool that copies directly from offload_verify_hidden_buffer
    on ``copy_stream`` in parallel with accept and draft i+1.
    """

    def __init__(
        self,
        device,
        max_rows: int,
        hidden_width: int,
        slots: int,
        pinned_bytes: int,
        *,
        consumer: ThreadPoolExecutor,
    ) -> None:
        device = torch.device(device)
        if device.type != "cuda":
            raise ValueError("HiddenD2HPool requires CUDA")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        self.device = device
        self.max_rows = max(1, int(max_rows))
        self.hidden_width = int(hidden_width)
        self.copy_stream = torch.cuda.Stream(device=device)
        row_bytes = self.hidden_width * 2 + 16  # bf16 hidden + int64 pos + int64 tok
        slot_bytes = self.max_rows * row_bytes
        count = min(max(1, int(slots)), pinned_bytes // max(1, slot_bytes))
        if count <= 0:
            raise ValueError("Insufficient pinned memory for one hidden D2H slot")
        self.slot_count = count
        self.slot_bytes = slot_bytes
        self.free: queue.LifoQueue = queue.LifoQueue(count)
        for _ in range(count):
            h_buf = torch.empty(
                (self.max_rows, self.hidden_width),
                dtype=torch.bfloat16,
                device="cpu",
                pin_memory=True,
            )
            p_buf = torch.empty(
                (self.max_rows * 8,), dtype=torch.uint8, device="cpu", pin_memory=True
            )
            t_buf = torch.empty(
                (self.max_rows * 8,), dtype=torch.uint8, device="cpu", pin_memory=True
            )
            self.free.put_nowait(
                (h_buf, p_buf, t_buf, torch.cuda.Event(), torch.cuda.Event())
            )
        self.consumer = consumer
        self.lock = threading.RLock()
        self.closed = False
        self._last_meta_done: torch.cuda.Event | None = None
        self._last_d2h_done: torch.cuda.Event | None = None

    def wait_meta_ready(self, forward_stream=None) -> None:
        """GPU-only wait before draft i+1 overwrites static positions / draft_tokens buffers."""
        ev = self._last_meta_done
        if ev is None:
            return
        if ev.query():
            self._last_meta_done = None
            return
        target = (
            torch.cuda.current_stream(self.device)
            if forward_stream is None
            else forward_stream
        )
        target.wait_event(ev)

    def wait_buffer_ready(self, forward_stream=None) -> None:
        """GPU-only wait before verify i+1 overwrites offload_verify_hidden_buffer."""
        ev = self._last_d2h_done
        if ev is None:
            return
        if ev.query():
            self._last_d2h_done = None
            return
        target = (
            torch.cuda.current_stream(self.device)
            if forward_stream is None
            else forward_stream
        )
        target.wait_event(ev)

    @torch.inference_mode()
    def submit(
        self,
        gpu_hidden: torch.Tensor,
        gpu_pos: torch.Tensor,
        gpu_tok: torch.Tensor,
        deliver,
        fail,
        *,
        forward_stream=None,
    ) -> bool:
        with self.lock:
            if self.closed:
                fail("offload closed")
                return False
        n_rows = int(gpu_hidden.shape[0])
        if n_rows <= 0:
            return True
        if n_rows > self.max_rows:
            fail(f"verify hidden rows {n_rows} exceed pool max_rows {self.max_rows}")
            return False
        try:
            slot = self.free.get_nowait()
        except queue.Empty:
            fail("hidden D2H slots exhausted")
            return False

        h_buf, p_buf, t_buf, meta_ev, copied = slot
        p_dtype = gpu_pos.dtype
        t_dtype = gpu_tok.dtype
        p_view = p_buf[: n_rows * gpu_pos.element_size()].view(p_dtype)
        t_view = t_buf[: n_rows * gpu_tok.element_size()].view(t_dtype)
        producer_stream = (
            torch.cuda.current_stream(self.device)
            if forward_stream is None
            else forward_stream
        )
        try:
            # Copy tiny positions + tokens FIRST so meta_ev completes in ~2us
            # while accept i is running, then copy the larger BF16 hidden slice
            # in full parallel with accept i + draft i+1.
            self.copy_stream.wait_stream(producer_stream)
            for src in (gpu_hidden, gpu_pos, gpu_tok):
                src.record_stream(self.copy_stream)
            with torch.cuda.stream(self.copy_stream):
                p_view.copy_(gpu_pos, non_blocking=True)
                t_view.copy_(gpu_tok, non_blocking=True)
                meta_ev.record(self.copy_stream)
                h_buf[:n_rows].copy_(gpu_hidden, non_blocking=True)
                copied.record(self.copy_stream)
            self._last_meta_done = meta_ev
            self._last_d2h_done = copied
        except Exception:
            self.free.put_nowait(slot)
            fail("hidden D2H submission failed")
            logger.exception("Hidden D2H submission failed")
            return False

        try:
            self.consumer.submit(
                self._deliver, slot, n_rows, p_dtype, t_dtype, deliver, fail
            )
        except Exception:
            try:
                copied.synchronize()
            except Exception:
                pass
            self.free.put_nowait(slot)
            fail("offload closed")
            return False
        return True

    def _deliver(self, slot, n_rows: int, p_dtype, t_dtype, deliver, fail) -> None:
        h_buf, p_buf, t_buf, _, copied = slot
        try:
            copied.synchronize()
            p_elem = torch.empty((), dtype=p_dtype).element_size()
            t_elem = torch.empty((), dtype=t_dtype).element_size()
            p_cpu = p_buf[: n_rows * p_elem].view(p_dtype).to(torch.int64)
            t_cpu = t_buf[: n_rows * t_elem].view(t_dtype).to(torch.int64)
            deliver(p_cpu, t_cpu, h_buf[:n_rows])
        except Exception as exc:
            fail(str(exc) or type(exc).__name__)
            logger.exception("Hidden D2H consumer failed")
        finally:
            self.free.put_nowait(slot)

    def close(self) -> None:
        with self.lock:
            self.closed = True
            self._last_meta_done = None
            self._last_d2h_done = None
            self.free = queue.LifoQueue()
