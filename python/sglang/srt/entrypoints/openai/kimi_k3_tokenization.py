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
import re
import sys
from collections.abc import Callable
from contextlib import redirect_stdout
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest

logger = logging.getLogger(__name__)

KIMI_K3_IMAGE_PLACEHOLDER = "<|kimi_image_placeholder|>"
KIMI_K3_IMAGE_PLACEHOLDER_ESCAPED = "<| kimi_image_placeholder |>"
KIMI_K3_MAX_VIDEO_INPUTS = 2


# K3 exposes three thinking tiers and K3.1 adds "medium". The OpenAI tiers map
# onto the nearest one so a standard client is neither rejected nor silently
# upgraded to the strongest tier; a float is an sglang extension K3 cannot
# express at all.
_KIMI_K3_EFFORT_TIERS = {
    "minimal": "low",
    "low": "low",
    "medium": "high",
    "high": "high",
    "xhigh": "max",
    "max": "max",
}
_KIMI_K31_EFFORT_TIERS = {**_KIMI_K3_EFFORT_TIERS, "medium": "medium"}


def kimi_k3_thinking_effort(
    reasoning_effort: Any, tiers: dict = _KIMI_K3_EFFORT_TIERS
) -> Optional[str]:
    """Map a request's reasoning effort onto K3's thinking_effort, or None."""
    if reasoning_effort is None or reasoning_effort == "none":
        return None
    if isinstance(reasoning_effort, str) and reasoning_effort in tiers:
        return tiers[reasoning_effort]
    supported = ", ".join(repr(t) for t in dict.fromkeys(tiers.values()))
    raise ValueError(
        f"Kimi K3 supports reasoning_effort {supported} (the other OpenAI tiers "
        f"map onto them); got {reasoning_effort!r}."
    )


def kimi_k3_effort_tiers(model_type: str | None) -> dict:
    """Select the checkpoint's native thinking tiers."""
    return _KIMI_K31_EFFORT_TIERS if model_type == "kimi_k31" else _KIMI_K3_EFFORT_TIERS


def truncate_kimi_k3_videos(messages: list[dict[str, Any]]) -> None:
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


_KIMI_K3_TOOL_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_-]{0,255}")


def validate_kimi_k3_message_tools(request: ChatCompletionRequest) -> str | None:
    """Message-level tool rules of the Kimi-K3 encoding, which renders such
    tools only from a system message without content."""
    for index in request._message_tools_indices:
        message = request.messages[index]
        if message.role not in ("system", "developer"):
            return (
                f"messages[{index}]: only a system or developer message can "
                f"declare tools, got role {message.role!r}."
            )
        if message.content:
            return f"messages[{index}]: a message that declares tools must have no content."
        for tool in message.tools:
            # `type` defaults to "function", so a missing one must be caught here.
            if tool.type != "function" or "type" not in tool.model_fields_set:
                return f"messages[{index}]: tool 'type' must be 'function'."
            name = tool.function.name
            if not _KIMI_K3_TOOL_NAME_RE.fullmatch(name):
                return (
                    f"messages[{index}]: invalid tool name {name!r}: it must start "
                    "with a letter or '_', contain only letters, digits, '_' or "
                    "'-', and be at most 256 characters."
                )
    return None


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


def prepare_kimi_k3_messages(
    messages: list[dict[str, Any]],
    request: ChatCompletionRequest,
) -> tuple[list[dict[str, Any]], int, str | None]:
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
    messages: list[dict[str, Any]],
    request: Any,
    *,
    model_type: str | None = None,
    append_prefix: Callable[[list[int], str], list[int]] | None = None,
) -> list[int]:
    """Render validated messages identically for serving and router workers."""
    messages, media_count, assistant_prefix = prepare_kimi_k3_messages(
        messages, request
    )
    template_kwargs = dict(request.chat_template_kwargs or {})
    template_kwargs.pop("tokenize", None)
    template_kwargs.pop("return_dict", None)
    template_kwargs.pop("image_prompts", None)
    template_kwargs.pop("add_generation_prompt", None)
    for key in ("conversation", "tools"):
        if key in template_kwargs:
            raise ValueError(
                f"chat_template_kwargs cannot set {key!r}; "
                "use the request fields instead."
            )
    if media_count:
        template_kwargs["image_prompts"] = ["<|media_pad|>"] * media_count

    tiers = kimi_k3_effort_tiers(model_type)
    effort = template_kwargs.get("thinking_effort")
    if effort is not None and effort not in tiers.values():
        supported = ", ".join(repr(t) for t in dict.fromkeys(tiers.values()))
        raise ValueError(
            f"chat_template_kwargs.thinking_effort must be one of "
            f"{supported}; got {effort!r}."
        )
    if (
        "thinking_effort" not in template_kwargs
        and template_kwargs.get("thinking") is not False
    ):
        effort = kimi_k3_thinking_effort(request.reasoning_effort, tiers)
        if effort is not None:
            template_kwargs["thinking_effort"] = effort

    declared_tools = effective_tools(request)
    allowed_tools = getattr(request.tool_choice, "allowed_tools", None)
    # The encoder renders tool_choice="none" with or without tools, so a
    # client's explicit "none" reaches it either way; the default "none"
    # of a request without tools does not.
    passes_tool_choice = (request.tool_choice == "required" and declared_tools) or (
        request.tool_choice == "none"
        and (declared_tools or getattr(request, "_tool_choice_explicit", False))
    )
    if passes_tool_choice:
        template_kwargs.setdefault("tool_choice", request.tool_choice)
    elif allowed_tools is not None and not allowed_tools.tools:
        # The text-only grammar can still admit malformed tool markers.
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
        prompt_ids = (
            append_prefix(prompt_ids, assistant_prefix)
            if append_prefix is not None
            else append_assistant_prefix(tokenizer, prompt_ids, assistant_prefix)
        )
    return prompt_ids


def validate_request(body: Any, *, allow_images: bool = True) -> None:
    """Reject unsupported content before validation can discard extra fields.

    Image URLs are retained verbatim. Loading pixels, checking image bytes,
    and expanding the single image placeholder remain worker operations.
    """
    if not isinstance(body, dict):
        raise ValueError("The chat request must be a JSON object")  # noqa: TRY004
    if any(
        body.get(field) is not None
        for field in ("input_ids", "kimi_k3_input_ids", "kimi_k3_media_input_ids")
    ):
        raise ValueError(
            "Kimi-K3 encoder expects messages without precomputed input_ids"
        )
    # Request templates are gated by the workers' --trust-request-chat-template.
    kwargs = body.get("chat_template_kwargs")
    if body.get("chat_template") is not None or (
        isinstance(kwargs, dict) and kwargs.get("chat_template") is not None
    ):
        raise ValueError("Router Kimi-K3 encoding does not run request chat templates")
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


def encode_request(
    tokenizer: Any,
    body: dict,
    defaults: dict | None = None,
    *,
    model_type: str | None = None,
):
    """Encode text/images with the serving entry point's request normalization."""
    from sglang.srt.entrypoints.openai.protocol import (
        ChatCompletionRequest,
        ThinkingParam,
    )
    from sglang.srt.function_call.utils import normalize_json_schema_types

    validate_request(body)
    request = ChatCompletionRequest.model_validate(copy.deepcopy(body))
    error = validate_kimi_k3_message_tools(request)
    if error:
        raise ValueError(error)
    if isinstance(request.thinking, dict):
        ThinkingParam.model_validate(request.thinking)
    kimi_k3_thinking_effort(request.reasoning_effort, kimi_k3_effort_tiers(model_type))
    for message_index, message in enumerate(request.messages):
        for tool_index, tool_call in enumerate(
            getattr(message, "tool_calls", None) or []
        ):
            if tool_call.function.name is None or not tool_call.function.name.strip():
                raise ValueError(
                    f"messages[{message_index}].tool_calls[{tool_index}]"
                    ".function.name must be a non-empty string."
                )
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
    # K3's renderer preserves nested argument JSON bytes; do not parse them.
    return encode_kimi_k3_messages(tokenizer, messages, request, model_type=model_type)


def encode_text_request(
    tokenizer: Any,
    body: dict,
    defaults: dict | None = None,
    *,
    model_type: str | None = None,
):
    """Compatibility entry point that preserves the original text-only guard."""
    validate_text_request(body)
    return encode_request(tokenizer, body, defaults, model_type=model_type)


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
    from sglang.srt.entrypoints.openai.chat_encoding import resolve_chat_encoding_spec
    from sglang.srt.utils.hf_transformers.config import get_config
    from sglang.srt.utils.hf_transformers.tokenizer import get_tokenizer

    tokenizer_path = config.get("tokenizer_path")
    if not isinstance(tokenizer_path, str) or not tokenizer_path:
        raise ValueError("Initialization requires tokenizer_path")
    defaults = config.get("default_chat_template_kwargs") or {}
    if not isinstance(defaults, dict):
        raise TypeError("default_chat_template_kwargs must be a JSON object")
    try:
        hf_config = get_config(
            tokenizer_path,
            trust_remote_code=config.get("trust_remote_code", False),
            revision=config.get("revision"),
        )
    except Exception as exc:
        raise ValueError(
            "Failed to load Kimi-K3 model configuration from "
            f"tokenizer_path={tokenizer_path!r}, revision={config.get('revision')!r}. "
            "Use a model/tokenizer directory or HF repository containing a "
            "loadable config.json; tokenizer-only directories are insufficient. "
            f"Original error: {exc}"
        ) from exc
    # Resolve from the checkpoint before loading its tokenizer. A generic HF
    # tokenizer can pass the warmup while rendering incompatible K3 semantics.
    if resolve_chat_encoding_spec(hf_config=hf_config, tokenizer=None) != "kimi_k3":
        raise ValueError(
            "Kimi-K3 chat encoding requires a Kimi-K3 model configuration at "
            f"tokenizer_path: {tokenizer_path}"
        )
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
            model_type=hf_config.model_type,
        )
    )
    return tokenizer, defaults, hf_config.model_type


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
            tokenizer, defaults, model_type = load_tokenizer(config)
    except Exception as exc:
        logger.exception("Kimi-K3 router tokenizer initialization failed")
        send({"error": str(exc), "error_type": "internal"})
        return 1
    send({"ready": True})
    for line in input_stream:
        try:
            with redirect_stdout(sys.stderr):
                body = json.loads(line)
                token_ids = _validate_token_ids(
                    encode(tokenizer, body, defaults, model_type=model_type)
                )
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
