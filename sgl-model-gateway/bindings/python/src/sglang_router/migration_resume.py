"""Resume /generate requests that a retiring decode instance handed off.

A decode instance paused with `pause_generation(mode="migrate")` finishes its
running requests with `meta_info.finish_reason.type == "migrated"`. The reason
carries `origin_input_ids` and `output_ids`; resubmitting their concatenation to
another prefill/decode pair continues the generation, and that prefill reads the
stored KV back from the shared storage backend. This module holds the pure
pieces: recognizing the hand-off, building the resubmitted request, and
rewriting the resumed outputs so the client sees one continuous request.

Supported: single (non-batch) /generate requests with cumulative outputs, the
server default. Not supported: --incremental-streaming-output, batch requests,
OpenAI-compatible endpoints.
"""

from typing import Dict, List, Optional

import orjson

MIGRATED = "migrated"
# A request that keeps landing on retiring instances is given up on.
MAX_RESUMES = 4

_PER_PAIR_FIELDS = ("bootstrap_host", "bootstrap_port", "bootstrap_room", "text")


def migrated_finish(ret) -> Optional[Dict]:
    """The hand-off finish reason of a /generate output, or None."""
    if not isinstance(ret, dict):
        return None
    finish = (ret.get("meta_info") or {}).get("finish_reason")
    if isinstance(finish, dict) and finish.get("type") == MIGRATED:
        return finish
    return None


def resumable(request: Dict) -> bool:
    """Whether this LB can resume `request` (single, not batch) after a hand-off:
    one output per request, so a parallel-sampled request is not resumable."""
    params = request.get("sampling_params")
    n = params.get("n", 1) if isinstance(params, dict) else 1
    return n in (None, 1)


def build_resume_request(original: Dict, finish: Dict, attempt: int) -> Dict:
    """The request that continues `original` after its decode handed it off.

    The prompt becomes every token so far, so the token budget shrinks by the
    tokens already generated. extra_key / cache_salt are kept: the stored KV is
    keyed by them. Bootstrap fields are dropped for the caller to set for the
    new pair."""
    output_ids = list(finish["output_ids"])
    request = {k: v for k, v in original.items() if k not in _PER_PAIR_FIELDS}
    request["input_ids"] = list(finish["origin_input_ids"]) + output_ids
    params = dict(original.get("sampling_params") or {})
    generated = len(output_ids)
    if params.get("max_new_tokens") is not None:
        params["max_new_tokens"] = max(params["max_new_tokens"] - generated, 1)
    if params.get("min_new_tokens"):
        params["min_new_tokens"] = max(params["min_new_tokens"] - generated, 0)
    request["sampling_params"] = params
    # The resumed request may itself be handed off again.
    request["allow_migration"] = True
    if original.get("rid") is not None:
        # The retiring instance may still hold the old rid.
        request["rid"] = f"{original['rid']}-migrated-{attempt}"
    return request


class ResumeStitcher:
    """Rewrites resumed outputs as continuations of the original request."""

    def __init__(self, *, request_id, prompt_tokens: int) -> None:
        self._request_id = request_id
        self._prompt_tokens = prompt_tokens
        self._text = ""
        self._output_ids: List[int] = []
        self._completion_tokens = 0
        self._migrations: List[Dict] = []

    @classmethod
    def for_first_hand_off(cls, ret: Dict, finish: Dict) -> "ResumeStitcher":
        return cls(
            request_id=ret["meta_info"].get("id"),
            prompt_tokens=len(finish["origin_input_ids"]),
        )

    def absorb(self, ret: Dict, finish: Dict) -> None:
        """Record a segment's last output, as its instance sent it."""
        self._text += ret.get("text") or ""
        # A resumed segment's output_ids are its own tokens only; its prompt
        # already holds the earlier ones.
        self._output_ids += list(finish["output_ids"])
        self._completion_tokens += ret["meta_info"].get("completion_tokens") or 0
        self._migrations.append(
            {"kv_flushed_tokens": finish.get("kv_flushed_tokens", 0)}
        )

    def rewrite(self, ret: Dict) -> Dict:
        """An output of the current segment, as part of the whole request."""
        out = dict(ret)
        out["text"] = self._text + (ret.get("text") or "")
        if "output_ids" in ret:
            out["output_ids"] = self._output_ids + list(ret["output_ids"] or [])
        meta = dict(ret.get("meta_info") or {})
        meta["id"] = self._request_id
        meta["prompt_tokens"] = self._prompt_tokens
        meta["completion_tokens"] = self._completion_tokens + (
            meta.get("completion_tokens") or 0
        )
        # cached_tokens stays the resumed segment's own: how much of the
        # resubmitted prompt the storage backend served.
        meta["migrations"] = list(self._migrations)
        out["meta_info"] = meta
        return out


def as_progress(ret: Dict) -> Dict:
    """A hand-off output stripped of its finish, sent on as a progress chunk."""
    out = dict(ret)
    meta = dict(out.get("meta_info") or {})
    meta["finish_reason"] = None
    out["meta_info"] = meta
    return out


def parse_sse_data(event: bytes) -> Optional[Dict]:
    """The JSON payload of one `data: {...}` SSE event, or None."""
    line = event.strip()
    if not line.startswith(b"data:"):
        return None
    body = line[len(b"data:") :].strip()
    if not body or body == b"[DONE]":
        return None
    return orjson.loads(body)


def is_done_event(event: bytes) -> bool:
    return event.strip() == b"data: [DONE]"


def to_sse(payload: Dict) -> bytes:
    return b"data: " + orjson.dumps(payload) + b"\n\n"


class SseEventSplitter:
    """Splits a byte stream into complete `\\n\\n`-terminated SSE events.

    The hand-off event carries the whole token history and can be megabytes,
    larger than one read chunk."""

    def __init__(self) -> None:
        self._buffer = bytearray()

    def feed(self, chunk: bytes) -> List[bytes]:
        self._buffer += chunk
        events = []
        start = 0
        while (end := self._buffer.find(b"\n\n", start)) >= 0:
            events.append(bytes(self._buffer[start : end + 2]))
            start = end + 2
        del self._buffer[:start]
        return events

    def flush(self) -> List[bytes]:
        rest = bytes(self._buffer)
        self._buffer.clear()
        return [rest] if rest.strip() else []
