"""Conditional token bans: mask one token right after any of a few others.

A request may carry a rule ``(banned, prevs)`` (see
``sampling_params.set_request_conditional_token_ban``). On every row whose
previous token is one of ``prevs`` the logit of ``banned`` is set to ``-inf``,
so no two-token sequence ``prev banned`` can be generated.

Kimi K3 uses it when thinking is off, with ``(think, (<|open|>, <|close|>))``.
The prompt then ends with the response channel already open and the think
channel is not part of the turn, yet the model may still reason in the response
channel and write ``<|close|>think<|sep|>``, closing a think channel that was
never opened, or open one with ``<|open|>think<|sep|>``. Without the name after
either control token it can only use the other channels, as on a normal answer.
The token itself stays free everywhere else, so text that contains the word is
not affected.

A rule holds for the whole request and needs no per-request state: the mask
depends only on the previous token, which every sampling site already has.

The batch state is immutable. Filtering and merging return a new object,
because a forward that is still running keeps the one it started with.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple

import torch

from sglang.srt.sampling.sampling_params import (
    MAX_CONDITIONAL_TOKEN_BAN_PREVS,
    get_request_conditional_token_ban,
)
from sglang.srt.utils.common import is_pin_memory_available

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req

logger = logging.getLogger(__name__)

# (banned, prevs)
Rule = Tuple[int, Tuple[int, ...]]

_NEVER = -1  # padding of the previous-token slots; no token has this id

_skips: Dict[str, int] = {}


def _warn_skipped(key: str, message: str) -> None:
    """A skipped mask is a silent loss of the ban, so it is logged on its 1st,
    10th, 100th, ... occurrence, with the count so far."""
    count = _skips.get(key, 0) + 1
    _skips[key] = count
    if count == 10 ** (len(str(count)) - 1):
        logger.warning("%s (occurrence %d)", message, count)


def batch_token_ban(sampling_info) -> Optional[ConditionalTokenBan]:
    """The batch's ban, or None when no request in it carries a rule."""
    ban = getattr(sampling_info, "conditional_token_ban", None)
    return ban if isinstance(ban, ConditionalTokenBan) else None


def last_input_token_ids(forward_batch) -> Optional[torch.Tensor]:
    """The token each sampled row continues, (bs,), or None when not known here.

    Only a plain decode step is covered. A prefill samples the token after the
    prompt, and K3's thinking-off prompt ends in ``<|sep|>``, which no rule
    lists. The exception is a retracted request: its re-prefill continues its
    last output token and runs that one token without the mask."""
    input_ids = forward_batch.input_ids
    if (
        input_ids is None
        or not forward_batch.forward_mode.is_decode()
        or getattr(forward_batch, "input_ids_from_graph", False)
    ):
        return None
    return input_ids


@dataclasses.dataclass(frozen=True, eq=False)
class ConditionalTokenBan:
    # Host mirror, one entry per batch row; None on a row without a rule. At
    # least one row has a rule.
    rules: Tuple[Optional[Rule], ...]
    # Largest banned id, to check against the logits width.
    max_banned_id: int
    # Device, int64. prev_ids is (bs, MAX_CONDITIONAL_TOKEN_BAN_PREVS): the
    # previous tokens that trigger the mask, padded with _NEVER (every slot on a
    # row without a rule). banned_ids is (bs,): the token to mask (0 on a row
    # without a rule, where nothing triggers).
    prev_ids: torch.Tensor
    banned_ids: torch.Tensor

    def __len__(self) -> int:
        return len(self.rules)

    @classmethod
    def from_reqs(cls, reqs: Sequence[Req], device) -> Optional[ConditionalTokenBan]:
        """State for a batch, or None when no request carries a rule."""
        rules = tuple(
            get_request_conditional_token_ban(
                getattr(req.sampling_params, "custom_params", None)
            )
            for req in reqs
        )
        if not any(rules):
            return None
        prev_ids: List[List[int]] = []
        banned_ids: List[int] = []
        for rule in rules:
            banned, prevs = rule if rule is not None else (0, ())
            padding = MAX_CONDITIONAL_TOKEN_BAN_PREVS - len(prevs)
            prev_ids.append(list(prevs) + [_NEVER] * padding)
            banned_ids.append(banned)
        pin = is_pin_memory_available(device)
        return cls._assemble(
            rules,
            torch.tensor(prev_ids, dtype=torch.int64, pin_memory=pin).to(
                device, non_blocking=True
            ),
            torch.tensor(banned_ids, dtype=torch.int64, pin_memory=pin).to(
                device, non_blocking=True
            ),
        )

    @classmethod
    def _assemble(
        cls,
        rules: Tuple[Optional[Rule], ...],
        prev_ids: torch.Tensor,
        banned_ids: torch.Tensor,
    ) -> ConditionalTokenBan:
        return cls(
            rules=rules,
            max_banned_id=max(rule[0] for rule in rules if rule is not None),
            prev_ids=prev_ids,
            banned_ids=banned_ids,
        )

    def filtered(
        self, keep_indices: Sequence[int], keep_indices_device: torch.Tensor
    ) -> Optional[ConditionalTokenBan]:
        rules = tuple(self.rules[i] for i in keep_indices)
        if not any(rules):
            return None
        return self._assemble(
            rules,
            self.prev_ids[keep_indices_device],
            self.banned_ids[keep_indices_device],
        )

    @classmethod
    def merged(
        cls,
        lhs: Optional[ConditionalTokenBan],
        rhs: Optional[ConditionalTokenBan],
        lhs_len: int,
        rhs_len: int,
        device,
    ) -> Optional[ConditionalTokenBan]:
        """State of ``lhs`` rows followed by ``rhs`` rows; either may be None."""
        if lhs is None and rhs is None:
            return None

        def parts(side: Optional[ConditionalTokenBan], size: int):
            if side is not None:
                return side.rules, side.prev_ids, side.banned_ids
            return (
                (None,) * size,
                torch.full(
                    (size, MAX_CONDITIONAL_TOKEN_BAN_PREVS),
                    _NEVER,
                    dtype=torch.int64,
                    device=device,
                ),
                torch.zeros((size,), dtype=torch.int64, device=device),
            )

        l_rules, l_prev, l_banned = parts(lhs, lhs_len)
        r_rules, r_prev, r_banned = parts(rhs, rhs_len)
        return cls._assemble(
            l_rules + r_rules,
            torch.cat([l_prev, r_prev]),
            torch.cat([l_banned, r_banned]),
        )

    def _usable(self, logits: torch.Tensor, rows: int) -> bool:
        if logits.ndim != 2 or logits.shape[0] != rows:
            _warn_skipped(
                "rows",
                "conditional token ban skipped: logits rows "
                f"{tuple(logits.shape)} do not match the batch ({rows} rows).",
            )
            return False
        if self.max_banned_id >= logits.shape[1]:
            _warn_skipped(
                "vocab",
                "conditional token ban skipped: banned token id "
                f"{self.max_banned_id} is outside the logits width {logits.shape[1]}.",
            )
            return False
        return True

    def _hit(self, prev_tokens: torch.Tensor) -> torch.Tensor:
        """Whether each previous token triggers its row's mask; shaped like
        ``prev_tokens``, whose first dimension is the batch."""
        prev_ids = self.prev_ids
        if prev_tokens.ndim == 2:
            prev_ids = prev_ids.unsqueeze(1)
        # A negative id is a placeholder or padding, never a token.
        return (prev_tokens.unsqueeze(-1) == prev_ids).any(dim=-1) & (prev_tokens >= 0)

    @staticmethod
    def _mask(logits: torch.Tensor, hit: torch.Tensor, banned: torch.Tensor) -> None:
        # Fixed shapes only: boolean indexing would sync with the device.
        column = banned.unsqueeze(1)
        current = logits.gather(1, column)
        logits.scatter_(1, column, current.masked_fill(hit.unsqueeze(1), float("-inf")))

    def apply(self, logits: torch.Tensor, prev_tokens: torch.Tensor) -> None:
        """Mask in place for one token per request.

        ``logits`` is (bs, vocab); ``prev_tokens`` is (bs,), the token each row
        continues."""
        bs = len(self.rules)
        if prev_tokens.ndim != 1 or prev_tokens.shape[0] != bs:
            _warn_skipped(
                "prev",
                "conditional token ban skipped: previous tokens "
                f"{tuple(prev_tokens.shape)} do not match the batch ({bs} rows).",
            )
            return
        if not self._usable(logits, bs):
            return
        self._mask(logits, self._hit(prev_tokens), self.banned_ids)

    def apply_chain(self, logits: torch.Tensor, chain_tokens: torch.Tensor) -> None:
        """Mask in place for a speculative verify block.

        ``chain_tokens`` is (bs, n): the last committed token followed by the
        drafted ones. ``logits`` is (bs * n, vocab), and row ``i * n + j``
        continues ``chain_tokens[i, j]``."""
        bs = len(self.rules)
        if chain_tokens.ndim != 2 or chain_tokens.shape[0] != bs:
            _warn_skipped(
                "chain",
                "conditional token ban skipped: chain tokens "
                f"{tuple(chain_tokens.shape)} do not match the batch ({bs} rows).",
            )
            return
        n = chain_tokens.shape[1]
        if not self._usable(logits, bs * n):
            return
        banned = self.banned_ids.unsqueeze(1).expand(bs, n).reshape(-1)
        self._mask(logits, self._hit(chain_tokens).reshape(-1), banned)
