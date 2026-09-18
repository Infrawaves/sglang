"""One coordinator, two pack buffers, and one bounded pool per prefill PP rank."""

from __future__ import annotations

import concurrent.futures
import logging
import threading
import time
from collections import deque
from contextlib import nullcontext
from dataclasses import dataclass, field

import torch

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.common.dcp_window_pack import DCPWindowPack, WindowPeer
from sglang.srt.disaggregation.common.dcp_window_plan import DCPDestinationPlan

logger = logging.getLogger(__name__)


@dataclass
class ChunkProgress:
    windows: int = 0
    gathers: int = 0
    sends: int = 0
    tokens: int = 0
    bytes: int = 0


@dataclass
class PendingChunk:
    chunk: object
    cursor: int = 0
    queued_at: float = field(default_factory=time.perf_counter)
    progress: ChunkProgress = field(default_factory=ChunkProgress)


@dataclass
class CancelledChunks:
    room: int
    chunks: deque


@dataclass(eq=False)
class Window:
    chunk: object
    start: int
    end: int
    chunk_done: bool
    buffer: object
    peers: list
    shards: list
    queued_at: float
    ready: torch.cuda.Event | None = None
    progress: ChunkProgress = field(default_factory=ChunkProgress)


class MooncakeWindowScheduler:
    def __init__(self, manager, workers, initial_mb):
        self.manager = manager
        self.pack = DCPWindowPack(manager.kv_args, manager.engine, initial_mb)
        self.peer_cache = {}
        self.registration_cache = {}
        self.pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="mooncake-window-send"
        )
        self.condition = threading.Condition(threading.RLock())
        self.pending = {}
        self.rooms = deque()
        self.accepting = True
        self.fatal_error = None
        self.retained_windows = []
        self.retained_cancellation = None
        self.futures = []
        self.last_received = set()
        self.next_page = {}
        self.quarantined_rooms = set()
        self.rejected_producers = {}
        # Includes ABORT received before room metadata. A late registration
        # must never restart writes after we have acknowledged that abort.
        self.aborted_rooms = set()
        self.closed = False
        self.thread = threading.Thread(
            target=self.run, name="mooncake-window-pack", daemon=True
        )

    def start(self):
        self.thread.start()

    def safe_to_release(self, room):
        with self.condition:
            return (
                room not in self.quarantined_rooms
                and self.manager._staging_outstanding.get(room, 0) == 0
                and all(
                    event.query() for event in self.rejected_producers.get(room, ())
                )
            )

    def idle(self):
        with self.condition:
            return (
                not self.fatal_error
                and not any(self.manager._staging_outstanding.values())
                and all(self.safe_to_release(room) for room in self.rejected_producers)
            )

    def submit(self, chunk):
        with self.condition:
            if not self.accepting or self.fatal_error:
                self.rejected_producers[chunk.room] = chunk.wait_event or ()
                self.manager.record_failure(
                    chunk.room,
                    f"DCP window scheduler unavailable: {self.fatal_error or 'closing'}",
                )
                self.manager.update_status(chunk.room, KVPoll.Failed)
                return
            if chunk.room in self.aborted_rooms or self.manager.request_status.get(
                chunk.room
            ) in (None, KVPoll.Failed):
                return
            if chunk.room in self.last_received:
                raise ValueError("DCP chunk submitted after last chunk")
            if (chunk.index_slice.start or 0) != self.next_page.get(chunk.room, 0):
                raise ValueError("DCP chunks must arrive in source page order")
            self.next_page[chunk.room] = (chunk.index_slice.start or 0) + len(
                chunk.prefill_kv_indices
            )
            if chunk.is_last_chunk:
                self.last_received.add(chunk.room)
            if chunk.room not in self.pending:
                self.pending[chunk.room] = deque()
                self.rooms.append(chunk.room)
            self.pending[chunk.room].append(PendingChunk(chunk))
            self.manager._staging_outstanding[chunk.room] += 1
            self.condition.notify()

    def abort(self, room, endpoint=None):
        # Serialize with submission and final Success. Dequeued windows keep
        # their outstanding reference until their GPU/NIC work has completed.
        with self.condition:
            self.aborted_rooms.add(room)
            self.manager.update_status(room, KVPoll.Failed)
            if endpoint is not None:
                self.manager.register_deferred_ack_target(room, *endpoint)
            self.condition.notify()
            self.manager._maybe_ack_drained_abort(room)

    def _peers(self, chunk):
        manager = self.manager
        cached = self.peer_cache.get(chunk.room)
        if cached is not None:
            with manager.session_lock:
                for peer in cached:
                    req = peer.request
                    session = req.mooncake_session_id
                    if session in manager.failed_sessions:
                        raise RuntimeError("DCP window destination session has failed")
                    if (
                        manager.decode_kv_args_table.get(session)
                        is not peer.registration
                    ):
                        raise RuntimeError(
                            "DCP registration changed during an active room"
                        )
                    if req.dst_kv_indices is not peer.page_table:
                        raise RuntimeError(
                            "DCP destination pages changed during an active room"
                        )
            return cached
        requests = list(manager.transfer_infos[chunk.room].values())
        peers = []
        for req in requests:
            if req.is_dummy:
                continue
            reg = manager.decode_kv_args_table[req.mooncake_session_id]
            registered = self.registration_cache.get(req.mooncake_session_id)
            if registered is None or registered[0] is not reg:
                indices = manager._validate_window_peer(reg)
                registered = (reg, tuple(reg.dst_kv_ptrs[i] for i in indices))
                self.registration_cache[req.mooncake_session_id] = registered
            if req.decode_prefix_len or req.dst_device_kv_indices is not None:
                raise ValueError(
                    "DCP window pack does not support decode prefix reuse or HiSparse"
                )
            with manager.session_lock:
                if req.mooncake_session_id in manager.failed_sessions:
                    raise RuntimeError("DCP window destination session has failed")
            peers.append(
                WindowPeer(
                    req,
                    reg,
                    req.dst_kv_indices,
                    DCPDestinationPlan(req.dst_kv_indices, manager.kv_args.page_size),
                    registered[1],
                )
            )
        if {peer.registration.dst_dcp_rank for peer in peers} != set(range(8)):
            raise ValueError("DCP window requires all eight destination ranks")
        if len(peers) != 8:
            raise ValueError("DCP window requires exactly one destination per DCP rank")
        peers.sort(key=lambda peer: peer.registration.dst_dcp_rank)
        self.peer_cache[chunk.room] = peers
        return peers

    def _take(self):
        with self.condition:
            while self.rooms:
                room = self.rooms.popleft()
                chunks = self.pending[room]
                if self.manager.request_status.get(room) in (None, KVPoll.Failed):
                    del self.pending[room]
                    return CancelledChunks(room, chunks)
                pending = chunks[0]
                chunk = pending.chunk
                start = pending.cursor
                if self.pack.buffers:
                    end, done = self.pack.window_end(chunk, start)
                else:
                    end, done = 0, True
                pending.cursor = end
                # Each reserved window owns a separate reference. Dropping an
                # aborted remainder must not release the same chunk's active
                # gather/RDMA window (including a prefetched last window).
                self.manager._staging_outstanding[room] += 1
                if done:
                    chunks.popleft()
                    self.manager._staging_outstanding[room] -= 1
                if chunks:
                    self.rooms.append(room)
                else:
                    del self.pending[room]
                return chunk, start, end, done, pending.queued_at, pending.progress
        return None

    def _prepare(self, buf):
        item = self._take()
        while isinstance(item, CancelledChunks):
            self.retained_cancellation = item
            # Even skipped gathers retain their source producers. Never wait
            # on CUDA while holding the lock used by scheduler-thread poll().
            for pending in item.chunks:
                for event in pending.chunk.wait_event or ():
                    event.synchronize()
            with self.condition:
                self.manager._staging_outstanding[item.room] -= len(item.chunks)
                self.manager._maybe_ack_drained_abort(item.room)
                self.retained_cancellation = None
            item = self._take()
        if item is None:
            return None
        chunk, start, end, done, queued_at, progress = item
        window = Window(
            chunk, start, end, done, buf, [], [], queued_at, progress=progress
        )
        # Keep partial gathers alive even if enqueue/JIT raises before ready
        # has been recorded. Fatal errors retain this list until process exit.
        self.retained_windows.append(window)
        with self._trace("DCP_PREPARE", window):
            window.peers = self._peers(chunk)
            if buf is not None and window.peers and end > start:
                window.ready, window.shards = self.pack.gather(
                    buf, chunk, start, end, window.peers
                )
            else:
                window.buffer = None
        return window

    def _trace(self, name, window):
        if not self.pack.enable_nvtx:
            return nullcontext()
        base = (window.chunk.index_slice.start or 0) * self.manager.kv_args.page_size
        return torch.cuda.nvtx.range(
            f"{name} pp={self.manager.pp_rank} room={window.chunk.room} "
            f"tokens={base + window.start}:{base + window.end}"
        )

    def _send_shard(self, window, shard, blocks):
        # Pending pool tasks can be skipped after abort. A running sync API
        # call cannot be cancelled; its outstanding window retains all pages.
        with self.condition:
            if self.manager.request_status.get(window.chunk.room) == KVPoll.Failed:
                return 0
        with self.manager.session_lock:
            if shard.request.mooncake_session_id in self.manager.failed_sessions:
                raise RuntimeError("DCP window destination session has failed")
        started = time.perf_counter() if self.pack.log_details else None
        trace = nullcontext()
        if self.pack.enable_nvtx:
            base = (
                window.chunk.index_slice.start or 0
            ) * self.manager.kv_args.page_size
            trace = torch.cuda.nvtx.range(
                f"DCP_SEND pp={self.manager.pp_rank} room={window.chunk.room} "
                f"buf={window.buffer.payload.get_ptr():x} "
                f"tokens={base + window.start}:{base + window.end} "
                f"rank={shard.registration.dst_dcp_rank} blocks={len(blocks)}"
            )
        with trace:
            ret = self.manager._transfer_data(shard.request.mooncake_session_id, blocks)
        if self.pack.log_details or ret != 0:
            logger.warning(
                "DCP window room=%s rank=%d descriptors=%d send_ms=%.3f ret=%s",
                window.chunk.room,
                shard.registration.dst_dcp_rank,
                len(blocks),
                (time.perf_counter() - started) * 1000 if started is not None else -1,
                ret,
            )
        return ret

    def _launch(self, window):
        buf = window.buffer
        if buf is not None:
            with self._trace("DCP_WAIT_GATHER", window):
                window.ready.synchronize()
            buf.state = "READY"
            if self.pack.log_details:
                logger.warning(
                    "DCP window room=%s gather_ms=%.3f queued_to_ready_ms=%.3f",
                    window.chunk.room,
                    buf.gather_started.elapsed_time(buf.ready),
                    (time.perf_counter() - window.queued_at) * 1000,
                )
        elif window.chunk.wait_event is not None:
            for event in window.chunk.wait_event:
                event.synchronize()
        else:
            raise RuntimeError("DCP window chunk has no producer event")
        self.futures = []
        with self._trace("DCP_SUBMIT_SEND", window):
            for shard in window.shards:
                if shard.blocks:
                    self.futures.append(
                        self.pool.submit(self._send_shard, window, shard, shard.blocks)
                    )
        if buf is not None:
            buf.state = "SENDING"
        return self.futures

    def _wait(self, futures):
        error = None
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                if result != 0:
                    error = RuntimeError(
                        f"Mooncake window transfer returned {result}; drain is unknown"
                    )
            except concurrent.futures.CancelledError:
                continue
            except Exception as exc:
                error = exc
            if error is not None:
                for pending in futures:
                    pending.cancel()
        if error is not None:
            raise error

    def _finish(self, window):
        manager = self.manager
        chunk = window.chunk
        if window.chunk_done and chunk.is_last_chunk:
            # Coordinator calls the existing state path directly. Its per-layer
            # tasks may use this pool; no pool worker waits on child pool tasks.
            for peer in window.peers:
                req, reg = peer.request, peer.registration
                if manager.request_status.get(chunk.room) == KVPoll.Failed:
                    break
                if chunk.state_indices:
                    if (
                        manager.maybe_send_extra(
                            req, chunk.state_indices, self.pool, reg
                        )
                        != 0
                    ):
                        raise RuntimeError(
                            "DCP window state transfer failed; drain is unknown"
                        )
                if (
                    manager.send_aux(req, chunk.prefill_aux_index, reg.dst_aux_ptrs)
                    != 0
                ):
                    raise RuntimeError(
                        "DCP window aux transfer failed; drain is unknown"
                    )
        with self.condition:
            if window.chunk_done:
                if (
                    chunk.is_last_chunk
                    and manager.request_status.get(chunk.room) != KVPoll.Failed
                ):
                    if manager._staging_outstanding[chunk.room] != 1:
                        raise RuntimeError(
                            "Last DCP chunk completed before earlier chunks"
                        )
                    manager.update_status(chunk.room, KVPoll.Success)
                    for peer in window.peers:
                        req = peer.request
                        manager.sync_status_to_decode_endpoint(
                            req.endpoint,
                            req.dst_port,
                            chunk.room,
                            KVPoll.Success,
                            manager._prefill_unique_rank(),
                        )
            if window.buffer is not None:
                window.buffer.state = "FREE"
            self.retained_windows.remove(window)
            manager._staging_outstanding[chunk.room] -= 1
            manager._maybe_ack_drained_abort(chunk.room)
        progress = window.progress
        progress.windows += 1
        progress.gathers += int(window.buffer is not None)
        progress.sends += sum(bool(shard.blocks) for shard in window.shards)
        progress.tokens += window.end - window.start
        progress.bytes += (window.end - window.start) * sum(self.pack.token_bytes)
        if (
            window.chunk_done
            and manager.request_status.get(chunk.room) != KVPoll.Failed
        ):
            logger.warning(
                "DCP window chunk room=%s windows=%d gathers=%d sends=%d tokens=%d bytes=%d ret=0",
                chunk.room,
                progress.windows,
                progress.gathers,
                progress.sends,
                progress.tokens,
                progress.bytes,
            )
        if self.pack.log_details:
            logger.warning(
                "DCP window room=%s tokens=%d bytes=%d elapsed_ms=%.3f",
                chunk.room,
                window.end - window.start,
                (window.end - window.start) * sum(self.pack.token_bytes),
                (time.perf_counter() - window.queued_at) * 1000,
            )

    def _quarantine(self, error):
        with self.condition:
            self.fatal_error = str(error)
            self.accepting = False
            # Do not decrement outstanding or send drain ACK. A timeout can
            # return while the transport still accesses source/destination.
            for room, count in list(self.manager._staging_outstanding.items()):
                if count <= 0:
                    continue
                self.quarantined_rooms.add(room)
                self.manager.record_failure(room, f"DCP window quarantined: {error}")
                self.manager.update_status(room, KVPoll.Failed)
                for req in list(self.manager.transfer_infos.get(room, {}).values()):
                    if not req.is_dummy:
                        try:
                            self.manager.sync_status_to_decode_endpoint(
                                req.endpoint,
                                req.dst_port,
                                room,
                                KVPoll.Failed,
                                self.manager._prefill_unique_rank(),
                            )
                        except Exception:
                            logger.exception(
                                "Failed to notify quarantined DCP room %s", room
                            )
        logger.critical(
            "DCP window PP%d quarantined; buffers and request pages retained, "
            "no drain ACK will be sent. Stop transports and coordinate both peers "
            "before recovery: %s",
            self.manager.pp_rank,
            error,
        )

    def run(self):
        torch.cuda.set_device(self.manager.kv_args.gpu_id)
        current = None
        index = 0
        try:
            while True:
                if current is None:
                    buf = self.pack.buffers[index] if self.pack.buffers else None
                    current = self._prepare(buf)
                    if current is None:
                        with self.condition:
                            if not self.accepting and not self.rooms:
                                return
                            if not self.rooms:
                                self.condition.wait(timeout=0.1)
                        continue
                self._launch(current)
                index = 1 - index
                buf = self.pack.buffers[index] if self.pack.buffers else None
                following = self._prepare(buf)
                with self._trace("DCP_WAIT_SEND", current):
                    self._wait(self.futures)
                self._finish(current)
                current = following
        except Exception as error:
            # A preparation/descriptor error may occur while A is still in
            # flight. Observe every submitted future before stopping the pool.
            try:
                self._wait(self.futures)
            except Exception:
                pass
            self._quarantine(error)

    def close(self):
        if self.closed:
            return
        with self.condition:
            self.accepting = False
            self.condition.notify()
        self.thread.join()
        self.pool.shutdown(wait=True, cancel_futures=True)
        if self.fatal_error:
            raise RuntimeError(
                f"Cannot release quarantined DCP buffers: {self.fatal_error}"
            )
        self.pack.close()
        self.closed = True

    def forget(self, room):
        with self.condition:
            if not self.safe_to_release(room):
                raise RuntimeError("Cannot clear an undrained DCP window room")
            if self.manager.request_status.get(room) == KVPoll.Failed:
                self.aborted_rooms.add(room)
            self.last_received.discard(room)
            self.next_page.pop(room, None)
            self.rejected_producers.pop(room, None)
            self.peer_cache.pop(room, None)
            self.manager._staging_outstanding.pop(room, None)
