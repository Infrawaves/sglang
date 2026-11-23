"""Map K3 parser output back to generated tokens, without re-tokenizing it."""

import logging
import math
import re
from bisect import bisect_right
from typing import Any, Optional, Sequence

from sglang.srt.function_call.kimik3_format import (
    MESSAGE_CLOSE,
    RESPONSE_CLOSE,
    RESPONSE_OPEN,
    THINK_CLOSE,
    THINK_OPEN,
)

_CHANNEL_MARKERS = re.compile(
    b"|".join(
        re.escape(s.encode())
        for s in (
            THINK_OPEN,
            THINK_CLOSE,
            RESPONSE_OPEN,
            RESPONSE_CLOSE,
            MESSAGE_CLOSE,
        )
    )
)

logger = logging.getLogger(__name__)

# internal_content[].logprobs placeholder for a position whose selected logprob
# is unknown (pending, missing under speculative decoding, truncated). Never
# 0.0, which a consumer would read as probability 1.
MISSING_LOGPROB = -9999.0


def place_event_values(
    values: Sequence[Any],
    segments: Optional[Sequence[Sequence[int]]],
    first_position: int,
) -> list[tuple[int, Any]]:
    """``(output position, value)`` for one stream event's logprob values.

    ``segments`` is the event's ``output_token_logprobs_segments``: one
    ``[start, count]`` per (coalesced) sub-event, in the order the values are
    concatenated, so gaps between sub-events stay in place. Without it (a
    non-incremental slice) the values start at ``first_position``.
    """
    if segments is None:
        segments = [(first_position, len(values))]
    placed = []
    for start, count in segments:
        for j in range(count):
            if len(placed) >= len(values):
                raise ValueError("logprob segments exceed the event's entries")
            placed.append((start + j, values[len(placed)]))
    if len(placed) != len(values):
        raise ValueError("logprob segments do not cover the event's entries")
    return placed


def align_token_logprobs(
    token_ids: Sequence[int], entries: Sequence[Optional[Sequence[Any]]]
) -> list[float]:
    """Selected logprob for each id in ``token_ids``.

    ``entries[i]`` is the engine's ``(logprob, token_id, text)`` triple at the
    output position of ``token_ids[i]``, or None. A position without an entry,
    whose entry was sampled for another token id, or whose value is not a
    finite number gets MISSING_LOGPROB.
    """
    logprobs = []
    for i, token_id in enumerate(token_ids):
        value = MISSING_LOGPROB
        if i < len(entries):
            entry = entries[i]
            if entry is not None and len(entry) >= 2 and entry[1] == token_id:
                try:
                    logprob = float(entry[0])
                except (TypeError, ValueError):
                    logprob = math.nan
                if math.isfinite(logprob):
                    value = logprob
        logprobs.append(value)
    return logprobs


class KimiTokenProvenance:
    """Generated tokens behind each K3 increment, for choices[].internal_content.

    One token can back more than one frame: a token whose bytes span two
    increments, reasoning replayed as content, and a tool call's slot-open
    frame and its argument continuation (both built from one parser fragment).
    Its id and selected logprob then repeat in each of those frames, because
    every non-empty increment frame must carry non-empty token_ids. Summing
    internal_content logprobs over frames therefore over-counts such a token;
    a sequence logprob has to come from the deduplicated token sequence.
    """

    def __init__(self, tokenizer, *, with_positions=False):
        # Both the HF Kimi tokenizer and SGLang's tiktoken wrapper expose the
        # actual Encoding. Its byte API preserves partial UTF-8 tokens.
        encoding = getattr(tokenizer, "model", None)
        if not callable(getattr(encoding, "decode_single_token_bytes", None)):
            encoding = getattr(tokenizer, "tokenizer", None)
        self.decode = getattr(encoding, "decode_single_token_bytes", None)
        if not callable(self.decode):
            raise ValueError("K3 internal_content requires a tiktoken byte decoder")
        self.raw = bytearray()
        self.ids = []
        self.logprobs = []
        self.ends = []
        self.cursor = 0
        self.reasoning_ranges = []
        # Engine entries received before their token id, by output position.
        self._early_entries = {}
        # Output positions behind the last take(), for choices[].logprobs.
        self.last_positions = []
        # Add token_positions (generated positions) to each taken entry.
        self.with_positions = with_positions

    def append(self, token_ids, logprobs=None):
        """Record generated tokens; ``logprobs`` is aligned with ``token_ids``
        (None, or a missing position, records MISSING_LOGPROB unless an entry
        for that position was already given to add_entries)."""
        for i, token_id in enumerate(token_ids):
            value = self.decode(token_id)
            if not isinstance(value, bytes) or not value:
                raise ValueError("Invalid token bytes for K3 internal_content")
            position = len(self.ids)
            self.raw.extend(value)
            self.ids.append(token_id)
            if logprobs is not None and i < len(logprobs):
                logprob = logprobs[i]
            elif position in self._early_entries:
                entry = self._early_entries.pop(position)
                logprob = align_token_logprobs([token_id], [entry])[0]
            else:
                logprob = MISSING_LOGPROB
            self.logprobs.append(logprob)
            self.ends.append(len(self.raw))

    def add_entries(self, placed):
        """Record engine ``(logprob, token_id, text)`` entries by output
        position. An entry can arrive in a later event than its token id;
        frames taken after it carry the value."""
        for position, entry in placed:
            if position < len(self.ids):
                self.logprobs[position] = align_token_logprobs(
                    [self.ids[position]], [entry]
                )[0]
            else:
                self._early_entries[position] = entry

    def _ranges(self, source, strip_channels):
        value = source.encode("utf-8")
        if not value:
            return []
        tail = bytes(self.raw[self.cursor :])
        segments = []
        offset = 0
        for marker in _CHANNEL_MARKERS.finditer(tail) if strip_channels else []:
            segments.append((self.cursor + offset, self.cursor + marker.start()))
            offset = marker.end()
        segments.append((self.cursor + offset, len(self.raw)))
        visible = b"".join(self.raw[a:b] for a, b in segments)
        start = visible.find(value)
        if start < 0:
            raise ValueError("K3 increment has no matching generated token source")
        end = start + len(value)
        ranges = []
        position = 0
        for a, b in segments:
            lo, hi = max(start, position), min(end, position + b - a)
            if lo < hi:
                ranges.append((a + lo - position, a + hi - position))
            position += b - a
        self.cursor = ranges[-1][1]
        return ranges

    def take(self, source, *, text=False, reasoning=False, replay_chars=0):
        ranges = []
        if replay_chars:
            prefix = source[:replay_chars].encode("utf-8")
            remaining = len(prefix)
            for a, b in self.reasoning_ranges:
                if remaining <= 0:
                    break
                end = min(b, a + remaining)
                ranges.append((a, end))
                remaining -= end - a
            if remaining or b"".join(self.raw[a:b] for a, b in ranges) != prefix:
                raise ValueError("K3 reasoning replay does not match its token source")
            source = source[replay_chars:]
        ranges.extend(self._ranges(source, strip_channels=text))
        if reasoning:
            self.reasoning_ranges.extend(ranges)
        # Deduplicate token *positions*, not IDs: repeated generated tokens are
        # distinct. A token spanning two increments legitimately appears in both.
        positions = []
        for a, b in ranges:
            index = bisect_right(self.ends, a)
            while index < len(self.ids) and (self.ends[index - 1] if index else 0) < b:
                if not positions or positions[-1] != index:
                    positions.append(index)
                index += 1
        if not positions:
            raise ValueError("K3 non-empty increment has no generated tokens")
        self.last_positions = positions
        entry = {
            "token_ids": [self.ids[i] for i in positions],
            "logprobs": [self.logprobs[i] for i in positions],
        }
        if self.with_positions:
            entry["token_positions"] = list(positions)
        return [entry]


class KimiStandardLogprobs:
    """choices[].logprobs content entries of one K3 choice, released by output
    position.

    A frame built from generated positions ``P`` claims every entry not sent
    yet up to ``max(P)``, in order, so entries of tokens with no visible text
    (tool markers, whitespace between XTML elements, a stripped stop token)
    ride on the next frame. Each entry goes out exactly once. An entry the
    engine has not delivered yet stops the claim; later frames or the end
    frame send it once it arrives. The end frame drains the rest, and
    positions that never received an entry are dropped with one warning.
    """

    def __init__(self, rid: str = "", index: int = 0):
        self._items = {}
        self._next = 0
        self._known = 0
        self._label = f"rid={rid} choice={index}"
        self.degraded = False

    def note_ids(self, count: int) -> None:
        """``count`` generated token ids are known for this choice."""
        self._known = max(self._known, count)

    def add(self, placed) -> None:
        """Record ``(output position, entry)`` pairs; repeats are ignored."""
        for position, item in placed:
            if position >= self._next:
                self._items.setdefault(position, item)

    def degrade(self, reason: str) -> None:
        """Release entries without positions from here on (claim(None))."""
        if not self.degraded:
            self.degraded = True
            logger.warning(
                "K3 logprobs for %s are no longer attributed per frame: %s",
                self._label,
                reason,
            )

    def claim(self, positions: Optional[Sequence[int]]) -> list:
        """Entries for a frame built from ``positions``; None (degraded)
        claims everything received so far."""
        if positions is None:
            end = max(self._known, max(self._items, default=-1) + 1)
        else:
            end = max(positions) + 1 if positions else self._next
        out = []
        while self._next < end and self._next in self._items:
            out.append(self._items.pop(self._next))
            self._next += 1
        return out

    def drain(self) -> list:
        """Entries for the end frame: everything not sent yet."""
        end = max(self._known, max(self._items, default=-1) + 1)
        out = []
        missing = 0
        while self._next < end:
            if self._next in self._items:
                out.append(self._items.pop(self._next))
            else:
                missing += 1
            self._next += 1
        if missing:
            logger.warning(
                "K3 logprobs for %s: %d generated positions never received an "
                "engine entry",
                self._label,
                missing,
            )
        return out
