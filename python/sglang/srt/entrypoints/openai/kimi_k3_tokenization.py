"""Shared Kimi-K3 prompt preparation for serving and router CPU workers.

Run ``python -u -m sglang.srt.entrypoints.openai.kimi_k3_tokenization`` for
the router's newline-delimited JSON protocol. The first line configures the
tokenizer; subsequent lines contain complete Chat Completions requests.
No model, scheduler, or TokenizerManager is constructed by this worker.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import sys
from collections.abc import Callable
from contextlib import redirect_stdout
from typing import Any

logger = logging.getLogger(__name__)

KIMI_K3_IMAGE_PLACEHOLDER = "<|kimi_image_placeholder|>"
KIMI_K3_IMAGE_PLACEHOLDER_ESCAPED = "<| kimi_image_placeholder |>"
KIMI_K3_MAX_VIDEO_INPUTS = 2


def truncate_kimi_k3_videos(messages: list[dict]) -> None:
    """Keep only the latest two K3 video parts across the whole request.

    K3 receives the complete conversation on every chat request, so the
    ordering here naturally covers both same-turn and multi-turn inputs.
    Removing old parts before encoding keeps the rendered placeholders and
    the extracted ``video_data`` list aligned.
    """
    video_count = sum(
        1
        for message in messages
        for part in (message.get("content") or [])
        if isinstance(part, dict) and part.get("type") == "video_url"
    )
    to_drop = max(0, video_count - KIMI_K3_MAX_VIDEO_INPUTS)
    if not to_drop:
        return

    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        kept = []
        for part in content:
            is_video = isinstance(part, dict) and part.get("type") == "video_url"
            if is_video and to_drop:
                to_drop -= 1
                continue
            kept.append(part)
        message["content"] = kept


def neutralize_kimi_k3_image_placeholder(text: str) -> str:
    return text.replace(KIMI_K3_IMAGE_PLACEHOLDER, KIMI_K3_IMAGE_PLACEHOLDER_ESCAPED)


def neutralize_kimi_k3_image_placeholder_value(value: Any) -> Any:
    if isinstance(value, str):
        return neutralize_kimi_k3_image_placeholder(value)
    if isinstance(value, list):
        return [neutralize_kimi_k3_image_placeholder_value(item) for item in value]
    if isinstance(value, dict):
        return {
            key: neutralize_kimi_k3_image_placeholder_value(item)
            for key, item in value.items()
        }
    return value


def handle_last_assistant_message(messages: list[dict], request: Any):
    """Preserve serving's string-only assistant continuation behavior."""
    assistant_prefix = None
    if messages and messages[-1].get("role") == "assistant":
        last_content = messages[-1].get("content")
        if isinstance(last_content, str):
            if request.continue_final_message:
                assistant_prefix = last_content
                messages = messages[:-1]
            else:
                messages[-1] = {"role": "user", "content": last_content}
    return messages, assistant_prefix


def append_assistant_prefix(tokenizer: Any, prompt_ids: list[int], prefix: str):
    encoded = tokenizer.encode(prefix)
    if encoded and encoded[0] == tokenizer.bos_token_id:
        encoded = encoded[1:]
    return prompt_ids + encoded


def _message_tools(message: Any) -> list:
    # Only generic system/developer messages carry tools; the user-only
    # message model has no `tools` field.
    if message.role not in ("system", "developer"):
        return []
    return getattr(message, "tools", None) or []


def effective_tools(request: Any) -> list:
    tools = list(request.tools or [])
    for message in request.messages:
        tools.extend(_message_tools(message))
    return tools


def prepare_kimi_k3_messages(messages: list[dict], request: Any):
    """Prepare text and media placeholders identically in both entry points."""
    truncate_kimi_k3_videos(messages)
    media_count = 0
    for index, message in enumerate(messages):
        content = message.get("content")
        if isinstance(content, list):
            parts = []
            for part in content:
                if not isinstance(part, dict):
                    continue
                part_type = part.get("type")
                if part_type in ("text", "input_text"):
                    parts.append(
                        {
                            "type": "text",
                            "text": neutralize_kimi_k3_image_placeholder(part["text"]),
                        }
                    )
                elif part_type in ("image_url", "input_image"):
                    image = part.get("image_url") or {}
                    if isinstance(image, str):
                        image = {"url": image, "detail": part.get("detail")}
                    parts.append({"type": "image_url", "image_url": image})
                    media_count += 1
                elif part_type == "video_url":
                    video = part.get("video_url") or {}
                    if isinstance(video, str):
                        video = {"url": video}
                    # K3's chat encoder only recognizes image parts. Emit one
                    # structural media placeholder here; the multimodal
                    # processor expands it into timestamped video chunks.
                    parts.append({"type": "image_url", "image_url": video})
                    media_count += 1
            message["content"] = parts
        elif isinstance(content, str):
            message["content"] = neutralize_kimi_k3_image_placeholder(content)
        elif content is None:
            message["content"] = ""

        if message.get("role") == "assistant":
            for key in ("reasoning_content", "reasoning"):
                if key in message:
                    message[key] = neutralize_kimi_k3_image_placeholder_value(
                        message[key]
                    )
            for tool_call in message.get("tool_calls") or []:
                function = (
                    tool_call.get("function") if isinstance(tool_call, dict) else None
                )
                if isinstance(function, dict) and "arguments" in function:
                    function["arguments"] = neutralize_kimi_k3_image_placeholder_value(
                        function["arguments"]
                    )

        source_tools = _message_tools(request.messages[index])
        if source_tools:
            message["tools"] = [
                tool.model_dump(exclude_unset=True, by_alias=True)
                for tool in source_tools
            ]
        if message.get("role") == "developer":
            message["role"] = "system"

    assistant_prefix = None
    if request.continue_final_message:
        messages, assistant_prefix = handle_last_assistant_message(messages, request)
    return messages, media_count, assistant_prefix


def encode_kimi_k3_messages(
    tokenizer: Any,
    messages: list[dict],
    request: Any,
    *,
    warn_unsupported_effort: Callable[[Any], None] | None = None,
) -> list[int]:
    """Encode already validated/normalized messages using the model renderer."""
    messages, media_count, assistant_prefix = prepare_kimi_k3_messages(
        messages, request
    )
    template_kwargs = dict(request.chat_template_kwargs or {})
    template_kwargs.pop("tokenize", None)
    template_kwargs.pop("return_dict", None)
    template_kwargs.pop("image_prompts", None)
    if media_count:
        template_kwargs["image_prompts"] = ["<|media_pad|>"] * media_count

    if (
        request.reasoning_effort in ("low", "high", "max")
        and "thinking_effort" not in template_kwargs
    ):
        template_kwargs["thinking_effort"] = request.reasoning_effort
    elif (
        request.reasoning_effort not in (None, "none", "low", "high", "max")
        and warn_unsupported_effort is not None
    ):
        warn_unsupported_effort(request.reasoning_effort)

    allowed_tools = getattr(request.tool_choice, "allowed_tools", None)
    if (
        effective_tools(request)
        and isinstance(request.tool_choice, str)
        and request.tool_choice in ("required", "none")
    ):
        template_kwargs.setdefault("tool_choice", request.tool_choice)
    elif allowed_tools is not None and not allowed_tools.tools:
        template_kwargs.setdefault("tool_choice", "none")
    elif allowed_tools is not None and allowed_tools.mode == "required":
        template_kwargs.setdefault("tool_choice", "required")
    if request.response_format is not None:
        template_kwargs.setdefault(
            "response_format",
            request.response_format.model_dump(exclude_unset=True, by_alias=True),
        )

    request_tools = (
        [tool.model_dump(exclude_unset=True, by_alias=True) for tool in request.tools]
        if request.tools
        else None
    )
    prompt_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        tools=request_tools,
        return_dict=False,
        **template_kwargs,
    )
    if assistant_prefix:
        prompt_ids = append_assistant_prefix(tokenizer, prompt_ids, assistant_prefix)
    return prompt_ids


def normalize_assistant_tool_call_arguments(
    message: dict, *, strict: bool = True, loads: Callable = json.loads
) -> None:
    """Normalize tool history; malformed arguments are retained for K3."""
    if message.get("role") != "assistant" or not isinstance(
        message.get("tool_calls"), list
    ):
        return
    for item in message["tool_calls"]:
        function = item.get("function") if isinstance(item, dict) else None
        if not isinstance(function, dict):
            continue
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            try:
                parsed = parse_tool_call_arguments(arguments, loads=loads)
            except ValueError:
                if strict:
                    raise
            else:
                function["arguments"] = parsed


def parse_tool_call_arguments(arguments: str, *, loads: Callable = json.loads) -> dict:
    try:
        parsed = loads(arguments)
    except ValueError as exc:
        raise ValueError(
            "Assistant tool call function.arguments must be valid JSON."
        ) from exc
    if not isinstance(parsed, dict):
        raise ValueError(  # noqa: TRY004 - preserve serving error contract
            "Assistant tool call function.arguments must be a JSON object."
        )
    return parsed


def validate_request(body: Any, *, allow_images: bool = True) -> None:
    """Reject unsupported content before validation can discard extra fields.

    Image URLs are retained verbatim. Loading pixels, checking image bytes,
    and expanding the single image placeholder remain worker operations.
    """
    if not isinstance(body, dict):
        raise ValueError("The chat request must be a JSON object")  # noqa: TRY004
    if body.get("input_ids") is not None:
        raise ValueError("Kimi-K3 encoder expects messages without input_ids")
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a nonempty array")
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("Each message must be a JSON object")  # noqa: TRY004
        content = message.get("content")
        if content is None or isinstance(content, str):
            continue
        if not isinstance(content, list):
            raise ValueError(  # noqa: TRY004 - request failures use ValueError
                "Message content must be text or an array of content parts"
            )
        for part in content:
            if not isinstance(part, dict):
                raise ValueError("Each content part must be a JSON object")  # noqa: TRY004
            if part.get("type") == "text" and isinstance(part.get("text"), str):
                continue
            if allow_images and part.get("type") == "image_url":
                image = part.get("image_url")
                if (
                    not isinstance(image, dict)
                    or not isinstance(image.get("url"), str)
                    or not image["url"].strip()
                ):
                    raise ValueError("image_url must contain a nonempty string url")
                if part.get("modalities") not in (None, "image", "multi-images"):
                    raise ValueError("Router Kimi-K3 encoding does not support video")
                continue
            supported = "text and image_url" if allow_images else "text"
            raise ValueError(
                f"Router Kimi-K3 encoding only supports {supported} content"
            )


def validate_text_request(body: Any) -> None:
    """Compatibility entry point for callers requiring strictly text input."""
    validate_request(body, allow_images=False)


def encode_request(tokenizer: Any, body: dict, defaults: dict | None = None):
    """Encode text/images with the serving entry point's request normalization."""
    import orjson
    from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest
    from sglang.srt.function_call.utils import normalize_json_schema_types

    validate_request(body)
    request = ChatCompletionRequest.model_validate(copy.deepcopy(body))
    # Serving validates and rewrites schemas before rendering. The resulting
    # normalized types are part of the prompt even when tool_choice is "none".
    for tool in effective_tools(request):
        if tool.function.parameters is not None:
            normalize_json_schema_types(tool.function.parameters)
    effort = (
        request.chat_template_kwargs.pop("reasoning_effort", None)
        if request.chat_template_kwargs
        else None
    )
    if effort is not None:
        request.reasoning_effort = effort
    if defaults:
        kwargs = dict(request.chat_template_kwargs or {})
        for key, value in defaults.items():
            kwargs.setdefault(key, value)
        request.chat_template_kwargs = kwargs
        if (
            kwargs.get("reasoning_effort") is not None
            and request.reasoning_effort is None
        ):
            request.reasoning_effort = kwargs["reasoning_effort"]
    messages = [message.model_dump() for message in request.messages]
    for message in messages:
        normalize_assistant_tool_call_arguments(
            message, strict=False, loads=orjson.loads
        )
    return encode_kimi_k3_messages(tokenizer, messages, request)


def encode_text_request(tokenizer: Any, body: dict, defaults: dict | None = None):
    """Compatibility entry point that preserves the original text-only guard."""
    validate_text_request(body)
    return encode_request(tokenizer, body, defaults)


def _validate_token_ids(token_ids: Any) -> list[int]:
    if (
        not isinstance(token_ids, list)
        or not token_ids
        or any(
            type(token_id) is not int or not 0 <= token_id <= 0xFFFFFFFF
            for token_id in token_ids
        )
    ):
        raise RuntimeError("Tokenizer returned invalid or empty input_ids")
    return token_ids


def _load_tokenizer(config: dict):
    from sglang.srt.utils.hf_transformers.tokenizer import get_tokenizer

    tokenizer_path = config.get("tokenizer_path")
    if not isinstance(tokenizer_path, str) or not tokenizer_path:
        raise ValueError("Initialization requires tokenizer_path")
    defaults = config.get("default_chat_template_kwargs") or {}
    if not isinstance(defaults, dict):
        raise TypeError("default_chat_template_kwargs must be a JSON object")
    tokenizer = get_tokenizer(
        tokenizer_path,
        trust_remote_code=config.get("trust_remote_code", False),
        revision=config.get("revision"),
        tokenizer_revision=config.get("revision"),
    )
    # Import/renderer failures must fail router startup, before any requests
    # can be dispatched to P/D with incompatible or missing token IDs.
    _validate_token_ids(
        encode_request(
            tokenizer,
            {"messages": [{"role": "user", "content": "Hello"}]},
            defaults,
        )
    )
    return tokenizer, defaults


def run_worker(
    input_stream=None, output_stream=None, *, load_tokenizer=None, encode=None
):
    """Serve one JSON response per line; request failures keep the worker alive."""
    input_stream = input_stream or sys.stdin
    output_stream = output_stream or sys.stdout
    load_tokenizer = load_tokenizer or _load_tokenizer
    encode = encode or encode_request

    def send(response):
        output_stream.write(json.dumps(response, separators=(",", ":")) + "\n")
        output_stream.flush()

    try:
        with redirect_stdout(sys.stderr):
            config = json.loads(input_stream.readline())
            if not isinstance(config, dict):
                raise TypeError("Initialization must be a JSON object")
            tokenizer, defaults = load_tokenizer(config)
    except Exception as exc:
        logger.exception("Kimi-K3 router tokenizer initialization failed")
        send({"error": str(exc), "error_type": "internal"})
        return 1
    send({"ready": True})
    for line in input_stream:
        try:
            with redirect_stdout(sys.stderr):
                body = json.loads(line)
                token_ids = _validate_token_ids(encode(tokenizer, body, defaults))
            response = {"input_ids": token_ids}
        except ValueError as exc:
            response = {"error": str(exc), "error_type": "request"}
        except Exception as exc:
            logger.exception("Kimi-K3 router tokenizer failed")
            response = {"error": str(exc), "error_type": "internal"}
        send(response)
    return 0


def _protocol_stdout():
    """Reserve the real stdout for protocol lines.

    ``redirect_stdout`` only covers Python-level writes. Native code (Rust
    tokenizers, C extensions in remote code) can write to fd 1 directly, which
    would corrupt the newline-delimited protocol. Move fd 1 to stderr and keep
    a private duplicate of the original stdout for responses.
    """
    sys.stdout.flush()
    protocol_fd = os.dup(1)
    os.dup2(2, 1)
    return os.fdopen(protocol_fd, "w", encoding="utf-8", buffering=1)


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    raise SystemExit(run_worker(output_stream=_protocol_stdout()))
