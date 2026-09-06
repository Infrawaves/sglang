"""Opt-in, CPU-only breadcrumbs for a disaggregated Decode hang.

No tensor values are read here. Request output_ids are already Python data.
Host call return does NOT imply GPU completion. Context is captured at forward
submission and explicitly restored for delayed sampling in the overlap loop.
"""

from __future__ import annotations

import functools
import hashlib
import itertools
import json
import logging
import os
import socket
import sys
import time
from contextlib import contextmanager
from contextvars import ContextVar
from logging.handlers import RotatingFileHandler
from pathlib import Path

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)
_context = ContextVar("decode_hang_context", default={})
_writer = None
_writer_pid = None
_failed = False
_event_seq = itertools.count(1)
_sample_seq = itertools.count(1)
_min_seq = itertools.count(1)


@functools.lru_cache(maxsize=1)
def enabled():
    # Startup-only flag; avoid repeated environment parsing in the hot path.
    return envs.SGLANG_DEBUG_DECODE_HANG.get()


class _TraceHandler(RotatingFileHandler):
    def handleError(self, record):
        # Disable tracing on a full/unwritable disk, without failing inference.
        raise


def _identity():
    fields = {"host": socket.gethostname(), "pid": os.getpid()}
    try:
        from sglang.srt.runtime_context import get_parallel

        parallel = get_parallel()
        fields.update(
            tp_rank=parallel.tp_rank,
            pp_rank=parallel.pp_rank,
            dp_rank=parallel.attn_dp_rank,
        )
    except Exception:
        fields["tp_rank"] = None
    return fields


def emit(event, **fields):
    global _writer, _writer_pid, _failed
    if not enabled() or _failed:
        return
    try:
        if _writer is None or _writer_pid != os.getpid():
            if _writer is not None:
                _writer.close()
            directory = Path(envs.SGLANG_DEBUG_DECODE_HANG_DIR.get())
            directory.mkdir(parents=True, exist_ok=True)
            name = f"decode-hang-{socket.gethostname()}-{os.getpid()}-{time.time_ns()}.jsonl"
            _writer = _TraceHandler(
                directory / name,
                maxBytes=max(1, envs.SGLANG_DEBUG_DECODE_HANG_MAX_MB.get()) * 1024**2,
                backupCount=max(1, envs.SGLANG_DEBUG_DECODE_HANG_BACKUPS.get()),
                encoding="utf-8",
            )
            _writer.setFormatter(logging.Formatter("%(message)s"))
            _writer_pid = os.getpid()
            torch = sys.modules.get("torch")
            source_root = Path(__file__).resolve().parents[1]
            source_hashes = {}
            for relative in (
                "observability/decode_hang.py",
                "layers/sampler.py",
                "disaggregation/decode.py",
            ):
                try:
                    source_hashes[relative] = hashlib.sha256(
                        (source_root / relative).read_bytes()
                    ).hexdigest()
                except OSError:
                    source_hashes[relative] = None
            emit(
                "trace_start",
                source_sha256=source_hashes,
                schema=1,
                row_columns=[
                    "rid",
                    "output_len",
                    "last_token",
                    "finished_reason",
                    "retracted",
                    "demoted",
                ],
                torch_version=str(getattr(torch, "__version__", "unavailable")),
                source_file=__file__,
            )
        record = {
            "schema": 1,
            **_identity(),
            "event_seq": next(_event_seq),
            "time_ns": time.time_ns(),
            "monotonic_ns": time.monotonic_ns(),
            **_context.get(),
            "event": event,
            **fields,
        }
        # Reject unexpected objects; never use default=str/repr on a tensor.
        message = json.dumps(record, ensure_ascii=True, separators=(",", ":"))
        _writer.handle(
            logging.LogRecord(__name__, logging.INFO, "", 0, message, (), None)
        )
    except Exception:
        _failed = True
        logger.warning(
            "Decode hang tracing disabled after a logging error", exc_info=True
        )


def request_rows(reqs):
    """Ordered compact rows; no prompts, tensor transfers, or extra finished() calls."""
    rows = []
    for req in reqs:
        ids = req.output_ids
        last = ids[-1] if ids else None
        rows.append(
            [
                req.rid,
                len(ids),
                last if isinstance(last, int) else None,
                (
                    type(req.finished_reason).__name__
                    if req.finished_reason is not None
                    else None
                ),
                bool(req.is_retracted),
                bool(req.is_demoted),
            ]
        )
    return rows


def trace_requests(event, reqs, **fields):
    if enabled() and reqs:
        emit(event, rows=request_rows(reqs), **fields)


def _batch_context(batch, forward_iter=None):
    rids = [req.rid for req in batch.reqs]
    return {
        "forward_iter": batch.forward_iter if forward_iter is None else forward_iter,
        "forward_mode": batch.forward_mode.name,
        "rids_hash": hashlib.sha256(
            json.dumps(rids, separators=(",", ":")).encode()
        ).hexdigest(),
    }


@contextmanager
def _use_context(fields):
    token = _context.set(fields)
    try:
        yield
    finally:
        _context.reset(token)


def _is_decode(owner):
    return (
        getattr(owner.disaggregation_mode, "value", owner.disaggregation_mode)
        == "decode"
    )


def bind_delayed_sample(func, context):
    # Copy only immutable metadata, never a mutable ScheduleBatch or Req list.
    context = dict(context)

    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        with _use_context(context):
            emit("delayed_sample_enter")
            result = func(*args, **kwargs)
            emit("delayed_sample_host_return")
            return result

    return wrapped


def trace_run_batch(func):
    @functools.wraps(func)
    def wrapped(self, batch, *args, **kwargs):
        if not enabled() or not _is_decode(self):
            return func(self, batch, *args, **kwargs)
        # run_batch assigns this forward_iter before submitting any work.
        context = _batch_context(batch, self.forward_ct + 1)
        with _use_context(context):
            trace_requests("batch_submit", batch.reqs)
            result = func(self, batch, *args, **kwargs)
            delayed = getattr(result, "delay_sample_func", None)
            # A prebuilt batch may return an inner idle batch's result; retain
            # the inner wrapper/context rather than wrapping it a second time.
            if delayed is not None and not getattr(
                delayed, "_decode_hang_bound", False
            ):
                result.delay_sample_func = bind_delayed_sample(delayed, context)
                result.delay_sample_func._decode_hang_bound = True
            emit(
                "batch_host_return",
                delayed_sample=delayed is not None,
                can_run_cuda_graph=getattr(result, "can_run_cuda_graph", None),
            )
            return result

    return wrapped


def trace_process_result(func):
    @functools.wraps(func)
    def wrapped(self, batch, result, *args, **kwargs):
        if not enabled() or not _is_decode(self):
            return func(self, batch, result, *args, **kwargs)
        with _use_context(_batch_context(batch)):
            trace_requests("result_process_enter", batch.reqs)
            ret = func(self, batch, result, *args, **kwargs)
            trace_requests("result_process_return", batch.reqs)
            return ret

    return wrapped


def trace_schedule(func):
    @functools.wraps(func)
    def wrapped(self, running_batch, *args, **kwargs):
        if not enabled() or not (running_batch.reqs or self.waiting_queue):
            return func(self, running_batch, *args, **kwargs)
        emit(
            "schedule_enter",
            after_forward_iter=self.forward_ct,
            running=request_rows(running_batch.reqs),
            waiting=request_rows(self.waiting_queue),
        )
        plan = func(self, running_batch, *args, **kwargs)
        emit(
            "schedule_return",
            after_forward_iter=self.forward_ct,
            selected=request_rows(plan.batch_to_run.reqs) if plan.batch_to_run else [],
            waiting=request_rows(self.waiting_queue),
        )
        return plan

    return wrapped


def group_facts(group):
    # All these are local host metadata reads; no collective or CUDA operation.
    import torch.distributed as dist

    fields = {}
    for key, read in (
        ("pg_name", lambda: group.group_name),
        ("pg_backend", lambda: str(dist.get_backend(group))),
        ("pg_ranks", lambda: dist.get_process_group_ranks(group)),
        ("pg_seq", lambda: group._get_sequence_number_for_group()),
    ):
        try:
            fields[key] = read()
        except Exception:
            fields[key] = None
    # pg_seq is PyTorch's host submission sequence, NOT the NCCL device opCount.
    return fields


def token_sync_enter(tensor, group, *, sync_enabled, env_enabled, has_grammar):
    if not enabled():
        return None
    ticket = {
        "sample_seq": next(_sample_seq),
        "min_seq": next(_min_seq) if sync_enabled else None,
        "sync_enabled": sync_enabled,
        "sync_env": env_enabled,
        "has_grammar": has_grammar,
        "numel": tensor.numel(),
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        **group_facts(group),
    }
    emit("token_sync_enter" if sync_enabled else "token_sync_skip", **ticket)
    return ticket


def token_sync_return(ticket):
    if ticket is not None:
        emit("token_sync_host_return", **ticket)
