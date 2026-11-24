"""CPU contract for an MLA cache snapshot and a linear DFLASH verify chain."""

from __future__ import annotations

import threading
from concurrent.futures import Future
from dataclasses import dataclass, field

import torch


@dataclass
class OffloadAttempt:
    """One local producer's handle. P and D must use the same capture/attempt IDs.

    verified_pages names full prompt page indices for which the consumer already
    holds validated bytes for this owner, layout, and attempt. An empty set
    requests the entire prompt. A consumer must discard an attempt with an error.
    """

    capture_id: str
    attempt_id: str
    verified_pages: frozenset[int] = frozenset()
    _error: str | None = field(default=None, init=False, repr=False)
    _lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False, compare=False
    )
    _pending: int = field(default=0, init=False, repr=False, compare=False)
    _released: bool = field(default=False, init=False, repr=False, compare=False)
    _completed: bool = field(default=False, init=False, repr=False, compare=False)

    @property
    def error(self) -> str | None:
        with self._lock:
            return self._error

    def fail(self, reason: str) -> None:
        with self._lock:
            if self._error is None:
                self._error = reason

    def acquire_packet(self) -> None:
        with self._lock:
            self._pending += 1

    def release_packet(self) -> bool:
        """Decrement pending packets; return True once the attempt is ready to complete."""
        with self._lock:
            if self._pending > 0:
                self._pending -= 1
            if self._released and self._pending == 0 and not self._completed:
                self._completed = True
                return True
            return False

    def mark_released(self) -> bool:
        """Mark producer admission closed for this request; return True if completable now."""
        with self._lock:
            self._released = True
            if self._pending == 0 and not self._completed:
                self._completed = True
                return True
            return False


@dataclass(frozen=True)
class MLALayout:
    layers: tuple[int, ...]
    row_bytes: int
    dtype: torch.dtype
    latent_dim: int
    rope_dim: int
    block_scaled_fp8: bool
    power_of_two_scales: bool
    dcp_size: int = 1
    page_size: int = 1
    scale_block_size: int | None = None
    format_version: int = 1


@dataclass(frozen=True)
class OffloadedRows:
    """Raw cache bytes are [owned row, layer, byte]. The consumer borrows tensors
    until its callback returns. Labels and final hidden states use the full
    verify positions, including predecessor rows whose KV lives on another rank.
    """

    role: str
    owner: int
    layout: MLALayout
    positions: torch.Tensor
    input_ids: torch.Tensor
    owned: torch.Tensor
    physical_ids: torch.Tensor
    raw_kv: torch.Tensor
    next_token_ids: torch.Tensor | None
    final_hidden: torch.Tensor | None


def prompt_spans(length, verified_pages, page_size, chunk_rows):
    if any(
        type(i) is not int or not 0 <= i < length // page_size for i in verified_pages
    ):
        raise ValueError(
            "A verified prompt page must be complete and within the prompt"
        )
    begin = 0
    for end_page in sorted(verified_pages):
        end = end_page * page_size
        for offset in range(begin, end, chunk_rows):
            yield offset, min(offset + chunk_rows, end)
        begin = end + page_size
    for offset in range(begin, length, chunk_rows):
        yield offset, min(offset + chunk_rows, length)


def visible_len(req):
    return (
        len(req.output_ids)
        if req.finished_len is None
        else min(req.finished_len, len(req.output_ids))
    )


class VerifyCommit:
    """Resolve only after the scheduler applies acceptance, grammar and stop."""

    def __init__(self, req, attempt):
        self.req = req
        self.attempt = attempt
        self.path = Future()
        self.start = 0
        self.anchor = None

    def before(self):
        req = self.req
        self.start = visible_len(req)
        self.anchor = (
            req.output_ids[self.start - 1] if self.start else req.origin_input_ids[-1]
        )

    def resolve(self):
        req = self.req
        end = visible_len(req)
        anchor = (
            req.output_ids[self.start - 1]
            if 0 < self.start <= end
            else req.origin_input_ids[-1]
        )
        reason = (
            None
            if req.finished_reason is None
            else req.finished_reason.to_json()["type"]
        )
        if (
            req.is_retracted
            or req.retraction_count
            or reason not in (None, "stop", "length")
            or end < self.start
            or anchor != self.anchor
        ):
            self.attempt.fail("retracted, aborted or rewritten reply")
            result = None
        else:
            added = tuple(req.output_ids[self.start : end])
            first = len(req.origin_input_ids) + self.start - 1
            result = (first, (self.anchor, *added[:-1]) if added else (), added)
        if not self.path.done():
            self.path.set_result(result)
        return result

    def after(self):
        self.resolve()

    def abort(self, reason: str):
        self.attempt.fail(reason)
        if not self.path.done():
            self.path.set_result(None)


class VerifyBatchCommit:
    """Resolve a packed decode verify chunk and release its event-driven CommitGate."""

    def __init__(self, commits: list[VerifyCommit], gate, on_done=None):
        self.commits = commits
        self.gate = gate
        self._on_done = on_done
        self._lock = threading.Lock()
        self._resolved = False

    def before(self):
        for commit in self.commits:
            commit.before()

    def after(self):
        with self._lock:
            if self._resolved:
                return
            self._resolved = True
        paths = tuple(commit.resolve() for commit in self.commits)
        if self._on_done is not None:
            self._on_done(self)
        self.gate.on_commit(paths)

    def abort(self, reason: str):
        with self._lock:
            if self._resolved:
                return
            self._resolved = True
        for commit in self.commits:
            commit.abort(reason)
        if self._on_done is not None:
            self._on_done(self)
        self.gate.on_commit(None)


def committed_slice(path, prompt_len, positions, inputs):
    if path is None:
        return slice(0, 0), ()
    first, predecessors, labels = path
    n = len(labels)
    if n > len(positions) or positions[:n].tolist() != list(range(first, first + n)):
        raise ValueError("Verify predecessor position mismatch")
    if tuple(inputs[:n].tolist()) != predecessors:
        raise ValueError("Verify predecessor token mismatch")
    # A teacher row at h predicts token h+1. P supplied the first reply token.
    begin = int(first < prompt_len)
    return slice(begin, n), labels[begin:]
