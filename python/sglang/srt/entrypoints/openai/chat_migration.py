"""Chat-side pieces of decode hot migration (pause_generation mode="migrate").

A retiring decode finishes each running request with finish_reason type
"migrated". For a chat request, serving_chat turns that into one hand-off event
(`sglang_migration`) for the router instead of a finish chunk. The router
resubmits the original body with `migration_resume`; the new decode prompts with
`origin_input_ids + output_ids`, replays `output_ids` through its response
parsers without emitting anything, and streams only what follows.

The event and the resume field are a wire contract with the HTTP PD router of
sgl-model-gateway; their shapes must match its side.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import orjson

from sglang.srt.entrypoints.openai.protocol import MigrationResume
from sglang.srt.managers.incremental_detokenize import PROMPT_TAIL_LEN

logger = logging.getLogger(__name__)

MIGRATED_FINISH_TYPE = "migrated"

Decode = Callable[[List[int]], str]

_REPLACEMENT_CHAR = "\ufffd"
# A UTF-8 character has at most 4 bytes, hence at most 4 byte-level tokens.
_MAX_PARTIAL_CHAR_TOKENS = 4


def resume_input_ids(resume: MigrationResume) -> List[int]:
    """The prompt of a resumed request; its prefill computes the same ids."""
    return list(resume.origin_input_ids) + list(resume.output_ids)


def decode_output_text(
    decode: Decode, *, prompt_ids: Sequence[int], output_ids: Sequence[int]
) -> str:
    """`output_ids` as the detokenizer streams them: decoded after the same
    prompt tail it uses as context (incremental_detokenize.PROMPT_TAIL_LEN)."""
    tail = list(prompt_ids[-PROMPT_TAIL_LEN:]) if PROMPT_TAIL_LEN else []
    return decode(tail + list(output_ids))[len(decode(tail)) :]


def complete_char_output_len(
    decode: Decode, *, prompt_ids: Sequence[int], output_ids: Sequence[int]
) -> int:
    """The longest prefix of `output_ids` that does not end inside a character.

    The detokenizer holds back the bytes of an unfinished character, so the
    client has not seen it; handing off only the prefix lets the next decode
    regenerate the character whole instead of losing it. Tokens are kept when
    dropping them would also drop text that may already have been streamed."""
    window = PROMPT_TAIL_LEN + _MAX_PARTIAL_CHAR_TOKENS
    ids = list(prompt_ids[-window:]) + list(output_ids)
    num_output = len(output_ids)
    for drop in range(min(_MAX_PARTIAL_CHAR_TOKENS, num_output) + 1):
        end = len(ids) - drop
        if not decode(ids[max(0, end - window) : end]).endswith(_REPLACEMENT_CHAR):
            break
    else:
        return num_output
    if drop == 0:
        return num_output
    context = ids[max(0, end - window) : end]
    dropped_text = decode(context + ids[end:])[len(decode(context)) :]
    if dropped_text.strip(_REPLACEMENT_CHAR):
        return num_output
    return num_output - drop


def trim_unfinished_char(text: str) -> str:
    """`text` without the replacement characters of an unfinished character."""
    return text.rstrip(_REPLACEMENT_CHAR)


def contains_token_sequence(ids: Sequence[int], sequence: Sequence[int]) -> bool:
    target, width = list(sequence), len(sequence)
    if not width:
        return False
    return any(
        list(ids[i : i + width]) == target for i in range(len(ids) - width + 1)
    )


def handoff_split(
    finish_reason: Dict[str, Any], resume: Optional[MigrationResume]
) -> Tuple[List[int], List[int]]:
    """(origin_input_ids, output_ids) of the whole chat request at a hand-off.

    A resumed request's engine prompt is the original prompt plus the earlier
    output, so its own split is folded back onto the original one."""
    engine_origin = finish_reason["origin_input_ids"]
    engine_output = finish_reason["output_ids"]
    if resume is None:
        return list(engine_origin), list(engine_output)
    if len(engine_origin) != len(resume.origin_input_ids) + len(resume.output_ids):
        logger.warning(
            "Hand-off of a resumed request: engine prompt has %d tokens, "
            "migration_resume accounts for %d.",
            len(engine_origin),
            len(resume.origin_input_ids) + len(resume.output_ids),
        )
    return list(resume.origin_input_ids), list(resume.output_ids) + list(
        engine_output
    )


def handoff_payload(
    *,
    index: int,
    origin_input_ids: List[int],
    output_ids: List[int],
    kv_flushed_tokens: int,
    reasoning_tokens: int,
) -> Dict[str, Any]:
    return {
        "index": index,
        "origin_input_ids": origin_input_ids,
        "output_ids": output_ids,
        "kv_flushed_tokens": kv_flushed_tokens,
        "reasoning_tokens": reasoning_tokens,
    }


def handoff_body(
    *, object_type: str, response_id: str, created: int, model: str, migration: Dict
) -> Dict[str, Any]:
    """A response carrying no choice, only the hand-off for the router."""
    return {
        "id": response_id,
        "object": object_type,
        "created": created,
        "model": model,
        "choices": [],
        "sglang_migration": migration,
    }


def handoff_sse(*, chunk_id: str, created: int, model: str, migration: Dict) -> str:
    body = handoff_body(
        object_type="chat.completion.chunk",
        response_id=chunk_id,
        created=created,
        model=model,
        migration=migration,
    )
    return "data: " + orjson.dumps(body).decode() + "\n\n"


def with_resumed_prompt(item: Dict[str, Any], *, origin_len: int) -> Dict[str, Any]:
    """An engine output of a resumed request, with its usage counted against
    the original prompt: the earlier output in the engine prompt is completion
    the router already counted."""
    meta_info = dict(item["meta_info"])
    meta_info["prompt_tokens"] = origin_len
    meta_info["cached_tokens"] = min(meta_info.get("cached_tokens") or 0, origin_len)
    return {**item, "meta_info": meta_info}
