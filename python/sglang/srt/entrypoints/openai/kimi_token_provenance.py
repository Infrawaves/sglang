"""Map K3 parser output back to generated tokens, without re-tokenizing it."""

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

# internal_content[].logprobs placeholder for a position whose selected logprob
# is unknown (pending, missing under speculative decoding, truncated). Never
# 0.0, which a consumer would read as probability 1.
MISSING_LOGPROB = -9999.0


def align_token_logprobs(
    token_ids: Sequence[int], entries: Sequence[Optional[Sequence[Any]]]
) -> list[float]:
    """Selected logprob for each id in ``token_ids``.

    ``entries`` are the engine's ``(logprob, token_id, text)`` triples,
    right-aligned with ``token_ids`` (the last entry belongs to the last id).
    A position without an entry, whose entry was sampled for another token id,
    or whose value is not a finite number gets MISSING_LOGPROB.
    """
    offset = len(entries) - len(token_ids)
    logprobs = []
    for i, token_id in enumerate(token_ids):
        value = MISSING_LOGPROB
        j = offset + i
        if 0 <= j < len(entries):
            entry = entries[j]
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

    def __init__(self, tokenizer):
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

    def append(self, token_ids, logprobs=None):
        """Record generated tokens; ``logprobs`` is aligned with ``token_ids``
        (None, or a missing position, records MISSING_LOGPROB)."""
        for i, token_id in enumerate(token_ids):
            value = self.decode(token_id)
            if not isinstance(value, bytes) or not value:
                raise ValueError("Invalid token bytes for K3 internal_content")
            self.raw.extend(value)
            self.ids.append(token_id)
            self.logprobs.append(
                logprobs[i]
                if logprobs is not None and i < len(logprobs)
                else MISSING_LOGPROB
            )
            self.ends.append(len(self.raw))

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
        return [
            {
                "token_ids": [self.ids[i] for i in positions],
                "logprobs": [self.logprobs[i] for i in positions],
            }
        ]
