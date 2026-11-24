"""Conditional token bans: mask one token right after another one.

A request may carry a rule ``(prev, banned, opener)`` (see
``sampling_params.set_request_conditional_token_ban``). On every row whose
previous token is ``prev`` the logit of ``banned`` is set to ``-inf``, so the
two-token sequence ``prev banned`` cannot be generated.

Kimi K3 uses it when thinking is off. The prompt then ends with the response
channel already open, yet the model may still reason there and write
``<|close|>think<|sep|>``, closing a think channel that was never opened. With
``(<|close|>, think, <|open|>)`` the model can only close the channel that is
open, as it does on a normal answer.

A request whose first output token is ``opener`` opened a channel of its own
(``<|open|>think<|sep|>`` at the very start, which the reasoning parser reads as
reasoning) and needs the close marker, so the rule is off for that request. The
state therefore depends only on the request's first output token: it never
changes once that token is known, and can be rebuilt from the request at any
time (retraction, PD handoff).

The batch state is immutable. Filtering, merging and refreshing return a new
object, because a forward that is still running keeps the one it started with.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import TYPE_CHECKING, List, Optional, Sequence, Tuple

import torch

from sglang.srt.sampling.sampling_params import get_request_conditional_token_ban
from sglang.srt.utils.common import is_pin_memory_available

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req

logger = logging.getLogger(__name__)

Rule = Tuple[int, int, int]

# Per-row state.
OFF = 0  # no rule, or the request opened a channel of its own
ARMED = 1  # the rule applies
PENDING = 2  # the rule is set, but the first output token is not known yet

_NEVER = -1  # prev id of a row that is not armed; no token has this id

_warned: set = set()


def _warn_once(key: str, message: str) -> None:
    if key not in _warned:
        _warned.add(key)
        logger.warning(message)


def armed_token_ban(sampling_info) -> Optional[ConditionalTokenBan]:
    """The batch's ban when some row is armed, else None."""
    ban = getattr(sampling_info, "conditional_token_ban", None)
    if isinstance(ban, ConditionalTokenBan) and ban.has_armed:
        return ban
    return None


def settle_pending_token_ban(sampling_info, reqs: Sequence[Req]) -> None:
    """Settle the rows whose first output token arrived since the batch was built.

    A request prefilled on this server has no output when its sampling info is
    built. The scheduler's sampling info gets a new object; a forward that is
    still running keeps its own."""
    ban = getattr(sampling_info, "conditional_token_ban", None)
    if isinstance(ban, ConditionalTokenBan) and ban.has_pending:
        sampling_info.conditional_token_ban = ban.refreshed(reqs)


def last_input_token_ids(forward_batch) -> Optional[torch.Tensor]:
    """The token each sampled row continues, (bs,), or None when not known here.

    Only a plain decode step is covered. A prefill samples a request's first
    token, whose row is still pending; the one armed row a prefill can hold is
    a retracted request, which then runs one token without the mask."""
    input_ids = forward_batch.input_ids
    if (
        input_ids is None
        or not forward_batch.forward_mode.is_decode()
        or getattr(forward_batch, "input_ids_from_graph", False)
    ):
        return None
    return input_ids


def _resolve_state(rule: Optional[Rule], output_ids: Sequence[int]) -> int:
    if rule is None:
        return OFF
    if len(output_ids) == 0:
        return PENDING
    return OFF if output_ids[0] == rule[2] else ARMED


@dataclasses.dataclass(frozen=True)
class ConditionalTokenBan:
    # Host mirror, one entry per batch row.
    rules: Tuple[Optional[Rule], ...]
    states: Tuple[int, ...]
    has_armed: bool
    has_pending: bool
    # Largest banned id of an armed row, to check against the logits width.
    max_banned_id: int
    # Device, (bs,) int64: the previous token that triggers the mask (_NEVER on
    # rows that are not armed) and the token to mask (0 on those rows).
    prev_ids: torch.Tensor
    banned_ids: torch.Tensor

    def __len__(self) -> int:
        return len(self.states)

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
        states = tuple(
            _resolve_state(rule, req.output_ids) for rule, req in zip(rules, reqs)
        )
        return cls._build(rules, states, device)

    @classmethod
    def _build(
        cls, rules: Tuple[Optional[Rule], ...], states: Tuple[int, ...], device
    ) -> ConditionalTokenBan:
        if ARMED not in states:
            # Nothing to mask yet (a prefill batch): no host-to-device copy.
            return cls._assemble(
                rules,
                states,
                torch.full((len(states),), _NEVER, dtype=torch.int64, device=device),
                torch.zeros((len(states),), dtype=torch.int64, device=device),
            )
        prev_ids: List[int] = []
        banned_ids: List[int] = []
        for rule, state in zip(rules, states):
            armed = state == ARMED
            prev_ids.append(rule[0] if armed else _NEVER)
            banned_ids.append(rule[1] if armed else 0)
        pin = is_pin_memory_available(device)
        return cls._assemble(
            rules,
            states,
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
        states: Tuple[int, ...],
        prev_ids: torch.Tensor,
        banned_ids: torch.Tensor,
    ) -> ConditionalTokenBan:
        return cls(
            rules=rules,
            states=states,
            has_armed=ARMED in states,
            has_pending=PENDING in states,
            max_banned_id=max(
                (rule[1] for rule, state in zip(rules, states) if state == ARMED),
                default=-1,
            ),
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
            tuple(self.states[i] for i in keep_indices),
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
                return side.rules, side.states, side.prev_ids, side.banned_ids
            return (
                (None,) * size,
                (OFF,) * size,
                torch.full((size,), _NEVER, dtype=torch.int64, device=device),
                torch.zeros((size,), dtype=torch.int64, device=device),
            )

        l_rules, l_states, l_prev, l_banned = parts(lhs, lhs_len)
        r_rules, r_states, r_prev, r_banned = parts(rhs, rhs_len)
        return cls._assemble(
            l_rules + r_rules,
            l_states + r_states,
            torch.cat([l_prev, r_prev]),
            torch.cat([l_banned, r_banned]),
        )

    def refreshed(self, reqs: Sequence[Req]) -> ConditionalTokenBan:
        """Settle pending rows whose first output token is known by now."""
        if not self.has_pending or len(reqs) != len(self.states):
            return self
        states = tuple(
            _resolve_state(rule, req.output_ids) if state == PENDING else state
            for rule, state, req in zip(self.rules, self.states, reqs)
        )
        if states == self.states:
            return self
        return self._build(self.rules, states, self.prev_ids.device)

    def _usable(self, logits: torch.Tensor, rows: int) -> bool:
        if logits.ndim != 2 or logits.shape[0] != rows:
            _warn_once(
                "rows",
                "conditional token ban skipped: logits rows "
                f"{tuple(logits.shape)} do not match the batch ({rows} rows).",
            )
            return False
        if self.max_banned_id >= logits.shape[1]:
            _warn_once(
                "vocab",
                "conditional token ban skipped: banned token id "
                f"{self.max_banned_id} is outside the logits width {logits.shape[1]}.",
            )
            return False
        return True

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
        if not self.has_armed:
            return
        bs = len(self.states)
        if prev_tokens.ndim != 1 or prev_tokens.shape[0] != bs:
            _warn_once(
                "prev",
                "conditional token ban skipped: previous tokens "
                f"{tuple(prev_tokens.shape)} do not match the batch ({bs} rows).",
            )
            return
        if not self._usable(logits, bs):
            return
        self._mask(logits, prev_tokens == self.prev_ids, self.banned_ids)

    def apply_chain(self, logits: torch.Tensor, chain_tokens: torch.Tensor) -> None:
        """Mask in place for a speculative verify block.

        ``chain_tokens`` is (bs, n): the last committed token followed by the
        drafted ones. ``logits`` is (bs * n, vocab), and row ``i * n + j``
        continues ``chain_tokens[i, j]``."""
        if not self.has_armed:
            return
        bs = len(self.states)
        if chain_tokens.ndim != 2 or chain_tokens.shape[0] != bs:
            _warn_once(
                "chain",
                "conditional token ban skipped: chain tokens "
                f"{tuple(chain_tokens.shape)} do not match the batch ({bs} rows).",
            )
            return
        n = chain_tokens.shape[1]
        if not self._usable(logits, bs * n):
            return
        hit = (chain_tokens == self.prev_ids.unsqueeze(1)).reshape(-1)
        banned = self.banned_ids.unsqueeze(1).expand(bs, n).reshape(-1)
        self._mask(logits, hit, banned)
