"""Fixed-capacity MLA window workspaces. Only the Mooncake coordinator writes them."""

from __future__ import annotations

import gc
import logging
from dataclasses import dataclass

import numpy as np
import torch

from sglang.kernels.ops.kvcache.pd_dcp_gather import copy_mla_rows_into_pack
from sglang.srt.disaggregation.common.dcp_pack import dcp_pack_buffer_bytes
from sglang.srt.disaggregation.common.staging_buffer import StagingBuffer
from sglang.srt.disaggregation.common.utils import build_dcp_token_transfer_plan

logger = logging.getLogger(__name__)

# A failed unregister is not permission to free memory still known to the NIC.
_UNSAFE_REGISTRATIONS = []


@dataclass
class WindowShard:
    request: object
    registration: object
    dst_rows: np.ndarray
    row_offset: int


class WindowBuffer:
    def __init__(self, size, max_tokens, kv_args, pool):
        device = f"cuda:{kv_args.gpu_id}"
        self.payload = StagingBuffer(size, device, kv_args.gpu_id, pool)
        self.rows = torch.empty(max_tokens, dtype=torch.int64, device=device)
        self.host_rows = torch.empty(max_tokens, dtype=torch.int64, pin_memory=True)
        self.metadata = torch.empty(
            len(kv_args.kv_data_ptrs) * 3, dtype=torch.int64, device=device
        )
        self.host_metadata = torch.empty_like(
            self.metadata, device="cpu", pin_memory=True
        )
        self.gather_started = torch.cuda.Event(enable_timing=True)
        self.ready = torch.cuda.Event(enable_timing=True)
        self.state = "FREE"
        self.registered = False


class DCPWindowPack:
    def __init__(self, kv_args, engine, initial_mb):
        from sglang.srt.disaggregation.common.staging_handler import (
            _get_custom_mem_pool,
        )

        if initial_mb not in (64, 128, 256, 512):
            raise ValueError("DCP pack buffer must be 64, 128, 256 or 512 MiB")
        self.kv_args = kv_args
        self.engine = engine
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
        pool, allocator = _get_custom_mem_pool(f"cuda:{kv_args.gpu_id}")
        for size_mb in (512, 256, 128, 64):
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
                        WindowBuffer(size, self.max_tokens, kv_args, pool)
                    )
            except torch.cuda.OutOfMemoryError:
                self.buffers.clear()
                # Leave the except block before collection: the traceback can
                # retain a partially constructed workspace, including buffer A.
            else:
                break
            gc.collect()
            torch.cuda.empty_cache()
            if size_mb == 64:
                raise RuntimeError(
                    "DCP window pack could not allocate two 64 MiB buffers"
                )
            logger.warning(
                "DCP window pack PP%d OOM at 2 x %d MiB; retrying 2 x %d MiB",
                kv_args.pp_rank,
                size_mb,
                size_mb // 2,
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
        logger.info(
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
            np.any(pages < 0)
            or any(
                (int(pages.max()) + 1) * item_len > length
                for item_len, length in zip(
                    self.kv_args.kv_item_lens, self.kv_args.kv_data_lens
                )
            )
        ):
            raise ValueError("DCP source page outside registered KV memory")
        shards = []
        n = 0
        for req, reg in sorted(peers, key=lambda peer: peer[1].dst_dcp_rank):
            if np.any(req.dst_kv_indices < 0):
                raise ValueError("Negative DCP destination page")
            plan = build_dcp_token_transfer_plan(
                pages,
                req.dst_kv_indices,
                physical_page_size=page_size,
                dcp_size=8,
                dcp_rank=reg.dst_dcp_rank,
                src_page_offset=(chunk.index_slice.start or 0) + cursor // page_size,
                decode_prefix_len=req.decode_prefix_len or 0,
                num_kv_tokens=end - cursor,
            )
            count = len(plan.src_token_indices)
            buf.host_rows.numpy()[n : n + count] = plan.src_token_indices
            shards.append(WindowShard(req, reg, plan.dst_token_indices, n))
            n += count
        if n != end - cursor or n > self.max_tokens:
            raise ValueError("DCP peers do not partition the window exactly once")
        metadata = buf.host_metadata.numpy().reshape(-1, 3)
        offset = 0
        for i, (ptr, width) in enumerate(
            zip(self.kv_args.kv_data_ptrs, self.token_bytes)
        ):
            metadata[i] = (ptr, width, offset)
            offset += n * width
        if chunk.wait_event is None:
            raise RuntimeError("DCP window requires a source KV producer event")
        with torch.cuda.stream(self.stream):
            for event in chunk.wait_event:
                self.stream.wait_event(event)
            buf.gather_started.record(self.stream)
            buf.rows[:n].copy_(buf.host_rows[:n], non_blocking=True)
            buf.metadata.copy_(buf.host_metadata, non_blocking=True)
            copy_mla_rows_into_pack(
                self.kv_args.kv_data_ptrs,
                buf.rows[:n],
                buf.payload.buffer,
                self.token_bytes,
                src_metadata=buf.metadata,
            )
            buf.ready.record(self.stream)
        return buf.ready, shards

    def blocks(self, buf, shard, num_rows, dst_ptrs):
        """Coalesce destination slots; packed source slots are consecutive."""
        rows = shard.dst_rows
        if not len(rows):
            return []
        starts = np.r_[0, np.flatnonzero(np.diff(rows) != 1) + 1]
        ends = np.r_[starts[1:], len(rows)]
        blocks = []
        layer_base = buf.payload.get_ptr()
        for width, dst in zip(self.token_bytes, dst_ptrs):
            src = layer_base + shard.row_offset * width
            blocks.extend(
                (
                    src + int(start) * width,
                    dst + int(rows[start]) * width,
                    int(end - start) * width,
                )
                for start, end in zip(starts, ends)
            )
            layer_base += num_rows * width
        return blocks
