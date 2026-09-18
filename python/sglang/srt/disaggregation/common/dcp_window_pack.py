"""Fixed-capacity MLA window workspaces. Only the Mooncake coordinator writes them."""

from __future__ import annotations

import gc
import logging
from contextlib import nullcontext
from dataclasses import dataclass

import torch

from sglang.kernels.ops.kvcache.pd_dcp_gather import PagedMLAGather
from sglang.srt.environ import envs
from sglang.srt.disaggregation.common.dcp_pack import dcp_pack_buffer_bytes
from sglang.srt.disaggregation.common.staging_buffer import StagingBuffer
from sglang.srt.disaggregation.common.dcp_window_plan import (
    DCPDestinationPlan,
    dcp_rank_layout,
)

logger = logging.getLogger(__name__)

# A failed unregister is not permission to free memory still known to the NIC.
_UNSAFE_REGISTRATIONS = []


@dataclass
class WindowPeer:
    request: object
    registration: object
    page_table: object
    destination: DCPDestinationPlan
    dst_ptrs: tuple


@dataclass
class WindowShard:
    request: object
    registration: object
    row_offset: int
    blocks: list


class WindowBuffer:
    def __init__(self, size, max_tokens, kv_args, pool, log_details=False):
        device = f"cuda:{kv_args.gpu_id}"
        self.payload = StagingBuffer(size, device, kv_args.gpu_id, pool)
        max_pages = (max_tokens + kv_args.page_size - 1) // kv_args.page_size
        self.pages = torch.empty(max_pages, dtype=torch.int32, device=device)
        self.host_pages = torch.empty(max_pages, dtype=torch.int32, pin_memory=True)
        self.host_pages_array = self.host_pages.numpy()
        self.gather_started = (
            torch.cuda.Event(enable_timing=True) if log_details else None
        )
        self.ready = torch.cuda.Event(enable_timing=log_details)
        self.state = "FREE"
        self.registered = False


class DCPWindowPack:
    def __init__(self, kv_args, engine, initial_mb):
        from sglang.srt.disaggregation.common.staging_handler import (
            _get_custom_mem_pool,
        )

        if initial_mb not in (32, 64, 128):
            raise ValueError("DCP pack buffer must be 32, 64 or 128 MiB")
        self.kv_args = kv_args
        self.engine = engine
        self.enable_nvtx = envs.SGLANG_MOONCAKE_DCP_NVTX.get()
        self.log_details = envs.SGLANG_MOONCAKE_DCP_LOG_DETAILS.get()
        self.buffers = []
        self.token_bytes = [n // kv_args.page_size for n in kv_args.kv_item_lens]
        self.virtual_page = kv_args.page_size * 8
        # Also checks divisibility/positive geometry before allocating anything.
        dcp_pack_buffer_bytes(kv_args.kv_item_lens, kv_args.page_size, 1, 8)
        if len(self.token_bytes) != len(kv_args.kv_data_ptrs):
            raise ValueError("DCP pack source geometry length mismatch")
        self.max_tokens = self.virtual_page
        self.stream = torch.cuda.Stream(device=kv_args.gpu_id)
        if not self.token_bytes:
            return
        self.source_page_limit = min(
            length // width
            for length, width in zip(kv_args.kv_data_lens, kv_args.kv_item_lens)
        )
        pool, allocator = _get_custom_mem_pool(f"cuda:{kv_args.gpu_id}")
        for size_mb in (128, 64, 32):
            if size_mb > initial_mb:
                continue
            size = size_mb * 1024**2
            self.max_tokens = (
                size // (sum(self.token_bytes) * self.virtual_page) * self.virtual_page
            )
            if self.max_tokens == 0:
                raise ValueError("DCP pack buffer cannot hold one virtual page")
            try:
                for _ in range(2):
                    self.buffers.append(
                        WindowBuffer(
                            size, self.max_tokens, kv_args, pool, self.log_details
                        )
                    )
            except torch.cuda.OutOfMemoryError:
                self.buffers.clear()
                # Leave the except block before collection: the traceback can
                # retain a partially constructed workspace, including buffer A.
            else:
                break
            gc.collect()
            torch.cuda.empty_cache()
            if size_mb == 32:
                raise RuntimeError(
                    "DCP window pack could not allocate two 32 MiB buffers"
                )
            logger.warning(
                "DCP window pack PP%d OOM at 2 x %d MiB; retrying 2 x %d MiB",
                kv_args.pp_rank,
                size_mb,
                size_mb // 2,
            )

        with torch.cuda.stream(self.stream):
            self.gather_plan = PagedMLAGather(
                kv_args.kv_data_ptrs,
                self.token_bytes,
                f"cuda:{kv_args.gpu_id}",
                kv_args.page_size,
            )
        try:
            for buf in self.buffers:
                # Treat even a failed registration as potentially partial.
                buf.registered = True
                if engine.batch_register([buf.payload.get_ptr()], [size]) != 0:
                    raise RuntimeError("Mooncake DCP window registration failed")
        except Exception:
            self.close()
            raise
        logger.warning(
            "DCP window pack PP%d: requested=%d MiB actual=2 x %d MiB "
            "allocator=%s registered=2 token_bytes=%s window_tokens=%d",
            kv_args.pp_rank,
            initial_mb,
            size_mb,
            allocator,
            self.token_bytes,
            self.max_tokens,
        )

    def close(self):
        try:
            self.stream.synchronize()
        except Exception:
            _UNSAFE_REGISTRATIONS.extend(self.buffers)
            raise
        failed = False
        for buf in self.buffers:
            if buf.registered:
                try:
                    ret = self.engine.batch_deregister([buf.payload.get_ptr()])
                except Exception:
                    ret = -1
                if ret != 0:
                    _UNSAFE_REGISTRATIONS.append(buf)
                    failed = True
                else:
                    buf.registered = False
        if failed:
            raise RuntimeError("DCP pack unregister failed; GPU storage retained")
        self.buffers.clear()

    def window_end(self, chunk, cursor):
        total = chunk.num_kv_tokens
        if total is None:
            total = len(chunk.prefill_kv_indices) * self.kv_args.page_size
        if not 0 <= total <= len(chunk.prefill_kv_indices) * self.kv_args.page_size:
            raise ValueError("DCP chunk token count exceeds source pages")
        if not 0 <= cursor <= total or (chunk.index_slice.start or 0) < 0:
            raise ValueError("Invalid DCP window cursor or source offset")
        start = (chunk.index_slice.start or 0) * self.kv_args.page_size + cursor
        end = min(total, cursor + self.max_tokens - start % self.virtual_page)
        return end, end == total

    def gather(self, buf, chunk, cursor, end, peers):
        """Return a CPU-waitable ready event and each rank's packed row range."""
        if buf.state != "FREE":
            raise RuntimeError("DCP pack attempted to overwrite a busy buffer")
        buf.state = "GATHERING"
        page_size = self.kv_args.page_size
        if cursor % page_size:
            raise ValueError("DCP window cursor must start on a physical page")
        pages = chunk.prefill_kv_indices[
            cursor // page_size : (end + page_size - 1) // page_size
        ]
        if pages.size and (
            int(pages.min()) < 0
            or int(pages.max()) >= min(self.source_page_limit, 2**31)
        ):
            raise ValueError("DCP source page outside registered KV memory")
        n = end - cursor
        if not 0 < n <= self.max_tokens:
            raise ValueError("Invalid DCP gather window size")
        start_token = (chunk.index_slice.start or 0) * page_size + cursor
        buf.host_pages_array[: pages.size] = pages
        if chunk.wait_event is None:
            raise RuntimeError("DCP window requires a source KV producer event")
        with torch.cuda.stream(self.stream):
            # The page table is CPU-owned and independent of KV production.
            # Upload it before waiting for the GPU writers of the source KV.
            buf.pages[: pages.size].copy_(
                buf.host_pages[: pages.size], non_blocking=True
            )
            for event in chunk.wait_event:
                self.stream.wait_event(event)
            if buf.gather_started is not None:
                buf.gather_started.record(self.stream)
            trace = nullcontext()
            if self.enable_nvtx:
                base = (chunk.index_slice.start or 0) * page_size
                trace = torch.cuda.nvtx.range(
                    f"DCP_GATHER pp={self.kv_args.pp_rank} room={chunk.room} "
                    f"buf={buf.payload.get_ptr():x} tokens={base + cursor}:{base + end}"
                )
            with trace:
                self.gather_plan(
                    buf.pages[: pages.size], n, buf.payload.buffer, start_token
                )
            buf.ready.record(self.stream)
        # Descriptor construction needs addresses, not the gathered contents.
        # Do it while the kernel is queued/running, before the caller waits.
        trace = (
            torch.cuda.nvtx.range(
                f"DCP_DESCRIPTORS pp={self.kv_args.pp_rank} room={chunk.room} tokens={start_token}:{start_token + n}"
            )
            if self.enable_nvtx
            else nullcontext()
        )
        with trace:
            shards = self.prepare_shards(buf, n, start_token, peers)
        return buf.ready, shards

    def prepare_shards(self, buf, num_rows, start_token, peers):
        layout = dcp_rank_layout(num_rows, start_token)
        shards = []
        base = buf.payload.get_ptr()
        for peer in peers:
            rank = peer.registration.dst_dcp_rank
            first, count, row_offset = layout[rank]
            runs = peer.destination.runs((start_token + first) // 8, count)
            blocks = []
            for width, prefix, dst in zip(
                self.token_bytes, self.gather_plan.layer_offsets, peer.dst_ptrs
            ):
                src = base + num_rows * prefix + row_offset * width
                blocks.extend(
                    (src + offset * width, dst + row * width, length * width)
                    for offset, row, length in runs
                )
            shards.append(
                WindowShard(peer.request, peer.registration, row_offset, blocks)
            )
        return shards
