"""CPU contract for Decode-only DFLASH MLA KV + Teacher hidden snapshot offload."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

import torch


@dataclass
class OffloadAttempt:
    """One local producer's handle for a sampled decode request.

    verified_pages optionally names complete prompt page indices that the consumer
    already holds (e.g. from HiCache/Mooncake) and wants skipped during KV export.
    An empty set exports the entire prompt + reply KV directly from Decode.
    A consumer must discard any attempt whose ``error`` is not None.
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
    format_version: int = 2
    dp_size: int = 1


@dataclass(frozen=True)
class OffloadedRows:
    """One delivered packet of KV rows or final teacher hidden rows on CPU.

    - KV packets (emitted once per span on request release across all DCP owners):
      ``positions``, ``input_ids``, ``owned`` (all True), ``physical_ids``, and
      ``raw_kv`` ``[N_owned, n_layers, row_bytes]`` contain only rows owned by
      ``owner``; ``next_token_ids`` and ``final_hidden`` are None.
    - Teacher hidden packets (emitted once on request release by the DP group's
      hidden owner ``attn_dcp_rank == 0``):
      ``positions``, ``input_ids``, ``next_token_ids``, and ``final_hidden``
      ``[N_reply, hidden_size]`` BF16 cover all committed decode teacher rows
      (``prompt_len .. prompt_len + output_len - 2``); ``raw_kv`` is empty.

    The consumer borrows tensors until its callback returns.
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
    dp_rank: int = 0


def prompt_spans(length: int, verified_pages, page_size: int, chunk_rows: int):
    """Yield (start, end) spans covering [0, length) while skipping verified complete pages."""
    if any(
        type(i) is not int or not 0 <= i < length // page_size for i in verified_pages
    ):
        raise ValueError(
            "A verified page must be complete and within the sequence"
        )
    begin = 0
    for end_page in sorted(verified_pages):
        end = end_page * page_size
        for offset in range(begin, end, chunk_rows):
            yield offset, min(offset + chunk_rows, end)
        begin = end + page_size
    for offset in range(begin, length, chunk_rows):
        yield offset, min(offset + chunk_rows, length)


def visible_len(req) -> int:
    return (
        len(req.output_ids)
        if req.finished_len is None
        else min(req.finished_len, len(req.output_ids))
    )


class RequestHiddenBuffer:
    """Position-indexed CPU buffer for a single sampled decode request.

    Owned exclusively by the single FIFO consumer thread. Every target-verify
    step writes (position, input_id, bf16_hidden) at ``offset = pos - prompt_len``;
    later verify steps automatically overwrite rejected speculative draft slots.
    At request release, ``extract()`` trims to ``[prompt_len, prompt_len + out_len - 1)``
    and verifies every position was captured with matching predecessor token ID.
    """

    def __init__(self, prompt_len: int, hidden_width: int, init_cap: int = 64) -> None:
        self.prompt_len = int(prompt_len)
        self.hidden_width = int(hidden_width)
        self.cap = 0
        self.nbytes = 0
        self.hidden: torch.Tensor | None = None
        self.tokens: torch.Tensor | None = None
        self.present: torch.Tensor | None = None
        self._init_cap = max(16, int(init_cap))

    def _grow(self, min_cap: int) -> int:
        new_cap = max(self._init_cap, self.cap * 2 if self.cap else self._init_cap)
        while new_cap < min_cap:
            new_cap *= 2
        new_hidden = torch.empty(
            (new_cap, self.hidden_width), dtype=torch.bfloat16, device="cpu"
        )
        new_tokens = torch.empty((new_cap,), dtype=torch.int64, device="cpu")
        new_present = torch.zeros((new_cap,), dtype=torch.bool, device="cpu")
        if self.cap > 0 and self.hidden is not None:
            new_hidden[: self.cap].copy_(self.hidden)
            new_tokens[: self.cap].copy_(self.tokens)
            new_present[: self.cap].copy_(self.present)
        old_bytes = self.nbytes
        self.hidden = new_hidden
        self.tokens = new_tokens
        self.present = new_present
        self.cap = new_cap
        self.nbytes = (
            new_hidden.numel() * 2 + new_tokens.numel() * 8 + new_present.numel()
        )
        return self.nbytes - old_bytes

    def write(
        self,
        positions: torch.Tensor,
        tokens: torch.Tensor,
        hidden: torch.Tensor,
        *,
        budget_remaining: int,
    ) -> int:
        """Overwrite positions >= prompt_len. Returns delta bytes allocated, or raises MemoryError."""
        pos_list = positions.tolist()
        tok_list = tokens.tolist()
        max_off = -1
        for p in pos_list:
            off = p - self.prompt_len
            if off >= 0 and off > max_off:
                max_off = off
        if max_off < 0:
            return 0
        delta_bytes = 0
        if max_off + 1 > self.cap:
            est_cap = max(self._init_cap, self.cap * 2 if self.cap else self._init_cap)
            while est_cap < max_off + 1:
                est_cap *= 2
            row_bytes = self.hidden_width * 2 + 9
            est_delta = est_cap * row_bytes - self.nbytes
            if est_delta > budget_remaining:
                raise MemoryError("host hidden buffer budget exceeded")
            delta_bytes = self._grow(max_off + 1)
        assert self.hidden is not None and self.tokens is not None and self.present is not None
        for idx, (p, t) in enumerate(zip(pos_list, tok_list, strict=True)):
            off = p - self.prompt_len
            if off < 0:
                continue
            self.hidden[off].copy_(hidden[idx])
            self.tokens[off] = t
            self.present[off] = True
        return delta_bytes

    def extract(self, output_ids: tuple[int, ...]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """Return (positions, input_ids, next_token_ids, final_hidden) for committed reply rows."""
        n = max(0, len(output_ids) - 1)
        if n == 0:
            return None
        if self.cap < n or self.present is None or self.tokens is None or self.hidden is None:
            raise ValueError(f"missing teacher hidden rows: have cap={self.cap}, need {n}")
        if not bool(self.present[:n].all().item()):
            raise ValueError("incomplete teacher hidden coverage across reply positions")
        stored = self.tokens[:n].tolist()
        expected_in = list(output_ids[:n])
        if stored != expected_in:
            raise ValueError("teacher hidden predecessor token mismatch")
        positions = torch.arange(
            self.prompt_len, self.prompt_len + n, dtype=torch.int64, device="cpu"
        )
        input_ids = torch.tensor(expected_in, dtype=torch.int64, device="cpu")
        next_token_ids = torch.tensor(output_ids[1 : n + 1], dtype=torch.int64, device="cpu")
        return positions, input_ids, next_token_ids, self.hidden[:n]
