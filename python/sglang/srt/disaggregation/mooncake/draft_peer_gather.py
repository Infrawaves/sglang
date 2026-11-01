"""Prefill-side peer gather of DSPARK draft KV for page-layout PD DCP.

With a hybrid-MLA target and prefill attn-TP = k x decode attn-TP, only one
prefill rank per decode rank sends KV, but each prefill rank holds 1/k of the
decode rank's draft heads. The sender gathers the full decode draft row from
its node-local peers over CUDA IPC.

Every attn-TP rank numbers the chunks it hands to the transfer queue (enqueue
seq) and publishes in shared memory a ready watermark: the largest seq whose
chunks, and all earlier ones, have their source draft KV written on its GPU.
A chunk without a wait event (normal send, non-overlap early send) is ready at
enqueue; an overlap early send waits on the prior forward and becomes ready
when the scheduler syncs that forward's result (``mark_forward_synced``). The
sender gathers a chunk only once every peer's watermark covers its seq.

Peers must enqueue the same chunk at the same seq; like the rest of PD this
relies on TP ranks making identical scheduling decisions.
"""

from __future__ import annotations

import ctypes
import logging
import socket
import threading
import time
import uuid
from multiprocessing import shared_memory
from typing import Callable, Dict, List, Optional, Sequence, Tuple
from unittest.mock import patch

import msgspec
import numpy as np
import numpy.typing as npt
import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

# Arbitrary; peers normally finish their own copy_done well within a forward.
_PEER_WAIT_POLL_S = 0.0005
# One aligned uint64 ready_seq: a single-copy-atomic store, no cross-word order.
SEQ_SHM_NBYTES = 8


def _ready_view(shm: shared_memory.SharedMemory) -> npt.NDArray[np.uint64]:
    return np.ndarray((1,), dtype=np.uint64, buffer=shm.buf)


class DraftPeerNotReady(RuntimeError):
    """Peer draft KV for a chunk was not ready before its gather was submitted.

    Raised only after every earlier native transfer of the send returned and
    before the peer gather or its RDMA started, so normal chunk retirement may
    release the transfer lease.
    """


class _PeerWaitStats:
    """Debug: per-window peer-wait distribution, logged by whichever transfer
    worker closes the window."""

    def __init__(self, *, local_rank: int, interval_s: float):
        self._local_rank = local_rank
        self._interval_s = interval_s
        self._lock = threading.Lock()
        self._window_start = time.monotonic()
        self._waits_ms: List[float] = []
        self._sleeps = 0

    def record(self, *, waited_s: float, sleeps: int) -> None:
        now = time.monotonic()
        with self._lock:
            self._waits_ms.append(waited_s * 1e3)
            self._sleeps += sleeps
            if now - self._window_start < self._interval_s:
                return
            waits = np.asarray(self._waits_ms)
            sleeps_total = self._sleeps
            window_s = now - self._window_start
            self._waits_ms = []
            self._sleeps = 0
            self._window_start = now
        p50, p90, p99 = np.percentile(waits, [50, 90, 99])
        logger.info(
            "PD DCP draft peer wait (attn-TP rank %d, %.0fs): chunks=%d "
            "ready_on_first_check=%.1f%% wait_ms p50=%.3f p90=%.3f p99=%.3f "
            "max=%.3f sleeps=%d",
            self._local_rank,
            window_s,
            waits.size,
            100.0 * float(np.mean(waits == 0.0)),
            p50,
            p90,
            p99,
            float(waits.max()),
            sleeps_total,
        )


class DraftPeerRecord(msgspec.Struct, frozen=True):
    """What one prefill attn-TP rank publishes to its peers at init."""

    hostname: str
    seq_shm_name: str
    # One CUDA IPC handle per distinct base allocation backing the draft pool.
    ipc_handles: List[bytes]
    # Per draft entry: (index into ipc_handles, byte offset from that base).
    entry_locations: List[Tuple[int, int]]
    # Set instead of the handles when this rank cannot export its draft pool.
    export_error: Optional[str] = None


def _export_draft_ipc(
    draft_ptrs: Sequence[int],
) -> Tuple[List[bytes], List[Tuple[int, int]]]:
    from cuda.bindings import driver as drv

    from sglang.srt.distributed.device_communicators.cuda_wrapper import (
        CudaRTLibrary,
    )

    lib = CudaRTLibrary()
    handles: List[bytes] = []
    base_to_handle: Dict[int, int] = {}
    locations: List[Tuple[int, int]] = []
    for ptr in draft_ptrs:
        err, base, _ = drv.cuMemGetAddressRange(int(ptr))
        if err != drv.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"cuMemGetAddressRange failed: {err}")
        base = int(base)
        if base not in base_to_handle:
            handle = lib.cudaIpcGetMemHandle(ctypes.c_void_p(base))
            base_to_handle[base] = len(handles)
            handles.append(bytes(handle))
        locations.append((base_to_handle[base], int(ptr) - base))
    return handles, locations


def _open_draft_ipc(record: DraftPeerRecord, gpu_id: int) -> List[int]:
    from sglang.srt.distributed.device_communicators.cuda_wrapper import (
        CudaRTLibrary,
        cudaIpcMemHandle_t,
    )

    # Opens map onto the calling thread's current device.
    torch.cuda.set_device(gpu_id)
    lib = CudaRTLibrary()
    bases = []
    for raw in record.ipc_handles:
        handle = cudaIpcMemHandle_t.from_buffer_copy(raw)
        # Opens with cudaIpcMemLazyEnablePeerAccess.
        bases.append(int(lib.cudaIpcOpenMemHandle(handle).value))
    return [bases[index] + offset for index, offset in record.entry_locations]


def _attach_seq_shm(name: str) -> shared_memory.SharedMemory:
    # Python tracks attached segments as if owned (stackoverflow.com/q/62748654);
    # the creating rank unlinks it. Same workaround as shm_broadcast.py.
    with patch("multiprocessing.resource_tracker.register", lambda *_, **__: None):
        return shared_memory.SharedMemory(name=name)


class DraftPeerGather:
    """Owns this rank's chunk seq watermark and the peers' mapped draft pointers."""

    def __init__(
        self,
        *,
        local_rank: int,
        local_draft_ptrs: Sequence[int],
        records: Sequence[DraftPeerRecord],
        seq_shm: shared_memory.SharedMemory,
        gpu_id: int,
    ):
        self.local_rank = local_rank
        self.records = list(records)
        self.gpu_id = gpu_id
        self._seq_shm = seq_shm
        self._local_ready = _ready_view(seq_shm)
        self._enqueued_seq = 0
        self._has_pending = False
        self._draft_ptrs: Dict[int, List[int]] = {
            local_rank: [int(ptr) for ptr in local_draft_ptrs]
        }
        self._peer_shms: Dict[int, shared_memory.SharedMemory] = {}
        self._peer_ready: Dict[int, npt.NDArray[np.uint64]] = {
            local_rank: self._local_ready
        }
        interval_s = envs.SGLANG_DISAGG_DRAFT_PEER_WAIT_STATS_INTERVAL.get()
        self._wait_stats = (
            _PeerWaitStats(local_rank=local_rank, interval_s=interval_s)
            if interval_s > 0
            else None
        )

    def enqueue_chunk(self, *, wait_event: Optional[torch.cuda.Event]) -> int:
        """Number one chunk handed to the transfer queue; returns its seq.

        Scheduler thread only. ``wait_event`` is the event the transfer worker
        waits on before reading the chunk's local KV. If it has not completed,
        the chunk and every later one stay unpublished until
        ``mark_forward_synced``; it is queried only when nothing is pending,
        since publishing must not skip an earlier unpublished chunk.
        """
        seq = self._enqueued_seq + 1
        self._enqueued_seq = seq
        if not self._has_pending:
            if wait_event is None or wait_event.query():
                self._local_ready[0] = seq
            else:
                self._has_pending = True
        return seq

    def mark_forward_synced(self) -> None:
        """Publish every enqueued chunk; the caller has synced all forwards
        those chunks read.

        Only the prefill result path calls this. An in-place engine pause
        (``pause_generation(mode="in_place")``) stops result processing, so an
        overlap early send queued just before it stays unpublished and its
        room fails after SGLANG_DISAGGREGATION_WAITING_TIMEOUT; not handled.
        """
        self._local_ready[0] = self._enqueued_seq
        self._has_pending = False

    def open_peers(self, peer_ranks: Sequence[int]) -> None:
        """Map ``peer_ranks``' draft pools and seq shm; raise if any cannot be."""
        host = self.records[self.local_rank].hostname
        for rank in peer_ranks:
            if rank in self._peer_ready:
                continue
            record = self.records[rank]
            if record.hostname != host:
                raise RuntimeError(
                    "PD DCP draft peer gather needs node-local peers, but attn-TP "
                    f"rank {rank} is on {record.hostname} and rank "
                    f"{self.local_rank} is on {host}"
                )
            if record.export_error is not None:
                raise RuntimeError(
                    f"PD DCP draft peer gather cannot map attn-TP rank {rank}'s "
                    f"draft KV over CUDA IPC: {record.export_error}"
                )
            # Attach the shm first: a failed attach must not leave an opened
            # IPC handle that a later registration would open again.
            shm = _attach_seq_shm(record.seq_shm_name)
            self._draft_ptrs[rank] = _open_draft_ipc(record, self.gpu_id)
            self._peer_shms[rank] = shm
            self._peer_ready[rank] = _ready_view(shm)

    def draft_ptrs(self, rank: int) -> List[int]:
        return self._draft_ptrs[rank]

    def wait_for_peers(
        self,
        *,
        peer_ranks: Sequence[int],
        seq: int,
        timeout_s: float,
        is_cancelled: Callable[[], bool],
    ) -> bool:
        """Block until every peer's draft KV for chunk ``seq`` is written;
        False on timeout/cancel.

        Skips this rank: the transfer worker already waited on the chunk's own
        event before reading local KV.
        """
        start = time.monotonic()
        deadline = start + timeout_s
        readies = [
            self._peer_ready[rank] for rank in peer_ranks if rank != self.local_rank
        ]
        sleeps = 0
        while True:
            if all(int(ready[0]) >= seq for ready in readies):
                if self._wait_stats is not None:
                    # Ready on the first check counts as exactly 0.
                    self._wait_stats.record(
                        waited_s=time.monotonic() - start if sleeps else 0.0,
                        sleeps=sleeps,
                    )
                return True
            if is_cancelled() or time.monotonic() >= deadline:
                return False
            time.sleep(_PEER_WAIT_POLL_S)
            sleeps += 1

    def close(self) -> None:
        for shm in self._peer_shms.values():
            shm.close()
        self._peer_shms.clear()
        self._seq_shm.close()
        self._seq_shm.unlink()


def init_draft_peer_gather(
    *,
    attn_tp_group,
    local_rank: int,
    draft_ptrs: Sequence[int],
    gpu_id: int,
) -> DraftPeerGather:
    """Collective over the attn-TP CPU group: publish this rank's record."""
    # Fresh shm pages are zero-filled: ready_seq 0.
    seq_shm = shared_memory.SharedMemory(
        name=f"sgl_dpg_{uuid.uuid4().hex[:16]}", create=True, size=SEQ_SHM_NBYTES
    )
    try:
        handles, locations = _export_draft_ipc(draft_ptrs)
        export_error = None
    except Exception as exc:
        handles, locations = [], []
        export_error = f"{type(exc).__name__}: {exc}"
    record = DraftPeerRecord(
        hostname=socket.gethostname(),
        seq_shm_name=seq_shm.name,
        ipc_handles=handles,
        entry_locations=locations,
        export_error=export_error,
    )
    records = attn_tp_group.all_gather_object(record)
    return DraftPeerGather(
        local_rank=local_rank,
        local_draft_ptrs=draft_ptrs,
        records=records,
        seq_shm=seq_shm,
        gpu_id=gpu_id,
    )


def init_draft_pack_buffers(
    *,
    register_fn: Callable[[int, int], None],
    count: int,
    gpu_id: int,
    max_tokens: int,
    draft_row_nbytes: int,
) -> list:
    """One registered pack buffer per transfer queue, ``max_tokens`` decode rows."""
    from sglang.srt.disaggregation.common.staging_buffer import StagingBuffer
    from sglang.srt.disaggregation.common.staging_handler import (
        _get_custom_mem_pool,
    )

    device = f"cuda:{gpu_id}"
    custom_mem_pool, _ = _get_custom_mem_pool(device)
    size_bytes = max_tokens * draft_row_nbytes
    buffers = []
    for _ in range(count):
        buf = StagingBuffer(size_bytes, device, gpu_id, custom_mem_pool=custom_mem_pool)
        register_fn(buf.get_ptr(), buf.get_size())
        buffers.append(buf)
    return buffers


def _launch_gather(
    segments: Sequence[Tuple[int, int, int, int, int]],
    row_indices: npt.NDArray[np.int64],
    pack_buffer,
) -> None:
    from sglang.kernels.ops.kvcache.pd_dcp_gather import copy_rows_into_strided_pack

    pack = pack_buffer.buffer
    rows = torch.as_tensor(row_indices, device=pack.device, dtype=torch.int64)
    gather_stream = pack_buffer.get_gather_stream()
    gather_stream.wait_stream(torch.cuda.default_stream(pack.device))
    with torch.cuda.stream(gather_stream):
        copy_rows_into_strided_pack(segments, rows, pack)
    gather_stream.synchronize()


def gather_draft_rows_into_pack(
    *,
    pack_buffer,
    entry_source_ptrs: Sequence[Sequence[int]],
    src_widths: Sequence[int],
    dst_widths: Sequence[int],
    src_rows: npt.NDArray[np.int64],
) -> List[int]:
    """Pack ``src_rows`` of every draft entry, entry-major, one decode row each.

    Entry ``e``'s block is ``[len(src_rows), dst_widths[e]]`` bytes; source ``j``
    of the entry fills bytes ``[j * src_widths[e], (j + 1) * src_widths[e])`` of
    each row. Returns each entry's packed base pointer.
    """
    n = int(src_rows.size)
    segments = []
    packed_ptrs = []
    offset = 0
    for sources, src_width, dst_width in zip(entry_source_ptrs, src_widths, dst_widths):
        if len(sources) * src_width != dst_width:
            raise ValueError(
                f"PD DCP draft gather: {len(sources)} sources x {src_width} B "
                f"do not fill a {dst_width} B decode row"
            )
        packed_ptrs.append(pack_buffer.get_ptr() + offset)
        for j, src_ptr in enumerate(sources):
            segments.append(
                (src_ptr, src_width, offset + j * src_width, src_width, dst_width)
            )
        offset += n * dst_width
    if offset > pack_buffer.get_size():
        raise ValueError(
            f"PD DCP draft gather window needs {offset} B, "
            f"pack buffer has {pack_buffer.get_size()} B"
        )
    _launch_gather(segments, src_rows, pack_buffer)
    return packed_ptrs
