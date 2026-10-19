import ast
import json
import pickle
import unittest
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from fastapi import HTTPException, Request
from pydantic import ValidationError

from sglang.srt.entrypoints.anthropic.protocol import AnthropicMessagesRequest
from sglang.srt.entrypoints.anthropic.serving import (
    AnthropicServing,
    _anthropic_usage_from_openai,
)
from sglang.srt.entrypoints.openai.protocol import (
    ChatCompletionResponse,
    MessageProcessingResult,
    PromptTokensDetails,
)
from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
from sglang.srt.entrypoints.openai.usage_processor import UsageProcessor
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


SNAPSHOT = {
    "ttl_seconds": 3600,
    "read_tokens": 16,
    "write_tokens": 16,
}


def _stream_chunk(delta=None, usage=None, finish_reason=None):
    return (
        "data: "
        + json.dumps(
            {
                "id": "test",
                "created": 0,
                "model": "test",
                "choices": [
                    {
                        "index": 0,
                        "delta": delta or {},
                        "finish_reason": finish_reason,
                    }
                ],
                "usage": usage,
            }
        )
        + "\n\n"
    )


class _FakeChat:
    def __init__(self, frames=(), *, cache_enabled=True):
        self.frames = frames
        self.cache_enabled = cache_enabled
        self.tokenizer_manager = SimpleNamespace(
            tokenizer=SimpleNamespace(chat_template=None),
            create_abort_task=lambda request: None,
        )

    def _validate_request(self, request):
        return None

    def _convert_to_internal_request(self, request, raw_request):
        options = request.prompt_cache_options
        ttl_seconds = None
        if self.cache_enabled:
            ttl_seconds = 3600 if options is not None and options.ttl == "1h" else 300
        return SimpleNamespace(cache_ttl_seconds=ttl_seconds), request

    async def _run_request_conversion(self, request, raw_request):
        return self._convert_to_internal_request(request, raw_request)

    async def _handle_non_streaming_request(
        self, adapted_request, request, raw_request
    ):
        usage = UsageProcessor.calculate_token_usage(
            prompt_tokens=43,
            completion_tokens=2,
            cache_usage={
                **SNAPSHOT,
                "ttl_seconds": adapted_request.cache_ttl_seconds,
            },
        )
        return ChatCompletionResponse(
            id="cache-non-stream",
            model="test",
            choices=[
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "hi"},
                    "finish_reason": "stop",
                }
            ],
            usage=usage,
        )

    async def _generate_chat_stream(self, *args):
        for frame in self.frames:
            yield frame


class _UnavailableChat(_FakeChat):
    async def _generate_chat_stream(self, *args):
        raise HTTPException(
            status_code=503,
            detail="Cache accounting is temporarily unavailable",
        )
        yield  # Make this an async generator, like the production handler.

    async def _handle_non_streaming_request(self, *args):
        raise HTTPException(
            status_code=503,
            detail="Cache accounting is temporarily unavailable",
        )


def _manager_identity_resolver():
    # Load only the actual trust boundary, without importing scheduler/GPU code.
    source = (
        Path(__file__).resolve().parents[5]
        / "python/sglang/srt/managers/tokenizer_manager.py"
    )
    tree = ast.parse(source.read_text())
    manager = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "TokenizerManager"
    )
    method = next(
        node
        for node in manager.body
        if getattr(node, "name", None) == "_resolve_cache_identity"
    )
    namespace = {
        "GenerateReqInput": GenerateReqInput,
        "DisaggregationMode": SimpleNamespace(PREFILL="prefill"),
    }
    code = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            method,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(code), str(source), "exec"), namespace)
    return namespace["_resolve_cache_identity"]


class TestAnthropicCacheIdentity(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.generated_identities = []
        self.adapted_requests = []
        self.stream_counts = (1, 2)
        self.output_text = "hi"
        self.output_prompt_tokens = 43
        self.physical_cached_tokens = 32
        self.accounting_read_tokens = 16
        self.accounting_write_tokens = 16
        self.physical_fallback = False
        self.resolve_identity = _manager_identity_resolver()
        self.manager = SimpleNamespace(disaggregation_mode="decode")

        async def run_in_request_preprocessor(func, *args):
            return func(*args)

        async def generate_request(adapted, raw_request):
            # The real manager preserves the adapter's hash over a shared
            # Bearer credential, or clears untrusted/missing identities.
            self.resolve_identity(self.manager, adapted, raw_request)
            self.generated_identities.append(adapted.api_key)
            self.adapted_requests.append(adapted)
            counts = self.stream_counts if adapted.stream else (2,)
            for count in counts:
                yield {
                    "text": self.output_text[:count],
                    "meta_info": {
                        "id": "anthropic-cache-identity",
                        "prompt_tokens": self.output_prompt_tokens,
                        "completion_tokens": count,
                        "reasoning_tokens": 0,
                        "cached_tokens": self.physical_cached_tokens,
                        "weight_version": "identity-test-version",
                        "finish_reason": (
                            {"type": "stop"} if count == counts[-1] else None
                        ),
                        **(
                            {
                                "cache_usage": {
                                    "ttl_seconds": adapted.cache_ttl_seconds,
                                    "read_tokens": (
                                        self.accounting_read_tokens
                                        if adapted.api_key
                                        and not self.physical_fallback
                                        else self.physical_cached_tokens
                                    ),
                                    "write_tokens": (
                                        self.accounting_write_tokens
                                        if adapted.api_key
                                        and not self.physical_fallback
                                        else 0
                                    ),
                                }
                            }
                            if adapted.cache_ttl_seconds is not None
                            else {}
                        ),
                    },
                }

        self.chat = object.__new__(OpenAIServingChat)
        self.chat.tokenizer_manager = SimpleNamespace(
            tokenizer=SimpleNamespace(chat_template=None),
            model_config=SimpleNamespace(is_multimodal=False),
            process_request_preprocessor=None,
            run_in_request_preprocessor=run_in_request_preprocessor,
            generate_request=generate_request,
            create_abort_task=lambda request: None,
        )
        self.chat.template_manager = SimpleNamespace(reasoning_config=None)
        self.chat.chat_encoding_spec = None
        self.chat.reasoning_parser = None
        self.chat.tool_call_parser = None
        self.chat.is_gpt_oss = False
        self.chat.allowed_custom_labels = None
        self.chat.default_sampling_params = {}
        self.chat._prompt_text_round_trip_is_lossy = False
        self.chat._process_messages = Mock(
            return_value=MessageProcessingResult(
                prompt="hello",
                prompt_ids=list(range(43)),
                image_data=None,
                video_data=None,
                audio_data=None,
                modalities=[],
                stop=[],
            )
        )
        self.config = SimpleNamespace(
            enable_cache_accounting=True,
            enable_cache_report=False,
            incremental_streaming_output=False,
            stream_response_default_include_usage=False,
            return_input_ids=False,
            return_output_ids=False,
            allow_auto_truncate=False,
        )
        self.disagg = SimpleNamespace(disaggregation_mode="decode")
        for name, value in (
            ("get_serving", self.config),
            ("get_disagg", self.disagg),
            ("get_model", SimpleNamespace(context_length=4096)),
        ):
            patcher = patch(
                "sglang.srt.entrypoints.openai.serving_chat." + name,
                return_value=value,
            )
            patcher.start()
            self.addCleanup(patcher.stop)
        self.serving = AnthropicServing(self.chat)

    def request(self, *, stream, user=None, options=None, multimodal=False):
        return AnthropicMessagesRequest(
            model="test",
            messages=[
                {
                    "role": "user",
                    "content": (
                        [
                            {"type": "text", "text": "hello"},
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": "ZmFrZQ==",
                                },
                            },
                        ]
                        if multimodal
                        else "hello"
                    ),
                }
            ],
            max_tokens=2,
            stream=stream,
            cache_user_id=user,
            prompt_cache_options=options,
        )

    def raw_request(self, identity=None):
        return Request(
            {
                "type": "http",
                "headers": [],
                "state": {} if identity is None else {"api_key": identity},
            }
        )

    async def consume(self, response, stream):
        self.assertEqual(response.status_code, 200)
        if stream:
            events = [
                json.loads(frame.split("data: ", 1)[1])
                async for frame in response.body_iterator
            ]
            self.assertEqual(events[-1]["type"], "message_stop")
            self.assertFalse(any(event["type"] == "error" for event in events))
            return next(
                event["message"]["usage"]
                for event in events
                if event["type"] == "message_start"
            )
        return json.loads(response.body)["usage"]

    def configure_multimodal_output(self):
        # Exercise the real protocol bridge with simulated final engine
        # metadata, without decoding media or running a GPU model here.
        self.chat.tokenizer_manager.model_config.is_multimodal = True
        self.chat._process_messages.return_value = MessageProcessingResult(
            prompt="hello <image>",
            prompt_ids=list(range(20)),
            image_data=["fixture-image"],
            video_data=None,
            audio_data=None,
            modalities=["image"],
            stop=[],
        )
        self.output_prompt_tokens = 770
        self.physical_cached_tokens = 333
        self.accounting_read_tokens = 256
        self.accounting_write_tokens = 512

    async def test_multimodal_default_and_selected_ttls_keep_final_input_usage(self):
        self.configure_multimodal_output()
        for options in (None, {"ttl": "5m"}, {"ttl": "1h"}):
            for stream in (False, True):
                with self.subTest(options=options, stream=stream):
                    request = self.request(
                        stream=stream,
                        user="multimodal-user",
                        options=options,
                        multimodal=True,
                    )
                    response = await self.serving.handle_messages(
                        request, self.raw_request()
                    )
                    self.assertEqual(response.status_code, 200)
                    for tier in ("5m", "1h"):
                        self.assertNotIn(
                            f"Msh-Usage-Cache-Write-Tokens-{tier}", response.headers
                        )
                    adapted = self.adapted_requests[-1]
                    self.assertTrue(adapted.contains_mm_input())
                    self.assertEqual(adapted.image_data, ["fixture-image"])
                    self.assertEqual(adapted.text, "hello <image>")
                    self.assertEqual(
                        adapted.cache_ttl_seconds,
                        3600 if options and options["ttl"] == "1h" else 300,
                    )
                    self.assertEqual(adapted.cache_required, options is not None)
                    if stream:
                        events = [
                            json.loads(frame.split("data: ", 1)[1])
                            async for frame in response.body_iterator
                        ]
                        self.assertFalse(
                            any(event["type"] == "error" for event in events)
                        )
                        start = next(
                            event
                            for event in events
                            if event["type"] == "message_start"
                        )
                        usage = start["message"]["usage"]
                        self.assertEqual(usage["output_tokens"], 0)
                        text = "".join(
                            event["delta"].get("text", "")
                            for event in events
                            if event["type"] == "content_block_delta"
                        )
                        self.assertEqual(text, "hi")
                        final = next(
                            event
                            for event in events
                            if event["type"] == "message_delta"
                        )
                        self.assertEqual(final["usage"], {"output_tokens": 2})
                        self.assertEqual(events[-1]["type"], "message_stop")
                    else:
                        body = json.loads(response.body)
                        self.assertEqual(body["content"][0]["text"], "hi")
                        usage = body["usage"]
                        self.assertEqual(usage["output_tokens"], 2)
                    self.assertEqual(usage["input_tokens"], 2)
                    self.assertEqual(usage["cache_read_input_tokens"], 256)
                    self.assertEqual(usage["cache_creation_input_tokens"], 512)
                    self.assertEqual(
                        usage["cache_creation"],
                        {
                            "ephemeral_5m_input_tokens": (
                                512 if adapted.cache_ttl_seconds == 300 else 0
                            ),
                            "ephemeral_1h_input_tokens": (
                                512 if adapted.cache_ttl_seconds == 3600 else 0
                            ),
                        },
                    )

    async def test_multimodal_missing_identity_and_redis_failure_use_physical_read(
        self,
    ):
        self.configure_multimodal_output()
        for source in ("missing_identity", "redis_failure"):
            self.physical_fallback = source == "redis_failure"
            for options in (None, {"ttl": "5m"}, {"ttl": "1h"}):
                for stream in (False, True):
                    with self.subTest(source=source, options=options, stream=stream):
                        response = await self.serving.handle_messages(
                            self.request(
                                stream=stream,
                                user=(
                                    "multimodal-user"
                                    if self.physical_fallback
                                    else None
                                ),
                                options=options,
                                multimodal=True,
                            ),
                            self.raw_request(),
                        )
                        usage = await self.consume(response, stream)
                        adapted = self.adapted_requests[-1]
                        self.assertTrue(adapted.contains_mm_input())
                        self.assertEqual(
                            adapted.api_key is None, source == "missing_identity"
                        )
                        self.assertEqual(usage["input_tokens"], 437)
                        self.assertEqual(usage["cache_read_input_tokens"], 333)
                        self.assertEqual(usage["cache_creation_input_tokens"], 0)
                        self.assertEqual(
                            usage["cache_creation"],
                            {
                                "ephemeral_5m_input_tokens": 0,
                                "ephemeral_1h_input_tokens": 0,
                            },
                        )

    def test_schema_matches_chat_identity_validation(self):
        for value in ("", " \t\n", 123, {}, True):
            with self.subTest(value=value), self.assertRaises(ValidationError):
                self.request(stream=False, user=value)
        request = self.request(stream=False, user="用户-🧪")
        converted = self.serving._convert_to_chat_completion_request(request)
        self.assertEqual(converted.cache_user_id, "用户-🧪")
        self.assertNotIn("用户-🧪", repr(request))

    async def test_body_identity_reaches_real_manager_in_both_response_modes(self):
        for stream in (False, True):
            for options in (None, {"ttl": "5m"}, {"ttl": "1h"}):
                with self.subTest(stream=stream, options=options):
                    raw = self.raw_request()
                    request = self.request(
                        stream=stream, user="gateway-body-user", options=options
                    )
                    response = await self.serving.handle_messages(request, raw)
                    usage = await self.consume(response, stream)
                    expected = (
                        "cache-user-id:" + sha256(b"gateway-body-user").hexdigest()
                    )
                    self.assertEqual(raw.state.cache_identity, expected)
                    self.assertEqual(self.generated_identities[-1], expected)
                    self.assertEqual(usage["cache_read_input_tokens"], 16)
                    self.assertEqual(usage["cache_creation_input_tokens"], 16)

    async def test_body_users_override_shared_verified_bearer(self):
        shared = "api-key:" + sha256(b"shared-gateway-key").hexdigest()
        for stream in (False, True):
            identities = []
            for user in ("gateway-user-a", "gateway-user-b"):
                raw = self.raw_request(shared)
                response = await self.serving.handle_messages(
                    self.request(stream=stream, user=user), raw
                )
                await self.consume(response, stream)
                expected = "cache-user-id:" + sha256(user.encode()).hexdigest()
                self.assertEqual(raw.state.api_key, shared)
                self.assertEqual(raw.state.cache_identity, expected)
                self.assertEqual(self.generated_identities[-1], expected)
                identities.append(expected)
            self.assertNotEqual(*identities)

    async def test_real_openai_stream_keeps_anthropic_input_and_output_usage(
        self,
    ):
        self.stream_counts = (3, 5)
        request = self.request(
            stream=True, user="gateway-role-user", options={"ttl": "1h"}
        )
        request.max_tokens = 5
        response = await self.serving.handle_messages(request, self.raw_request())
        self.assertEqual(response.status_code, 200)
        events = [
            json.loads(frame.split("data: ", 1)[1])
            async for frame in response.body_iterator
        ]
        starts = [event for event in events if event["type"] == "message_start"]
        self.assertEqual(len(starts), 1)
        self.assertEqual(
            starts[0]["message"]["usage"],
            {
                "input_tokens": 11,
                "output_tokens": 0,
                "cache_read_input_tokens": 16,
                "cache_creation_input_tokens": 16,
                "cache_creation": {
                    "ephemeral_5m_input_tokens": 0,
                    "ephemeral_1h_input_tokens": 16,
                },
            },
        )
        final = next(event for event in events if event["type"] == "message_delta")
        self.assertEqual(final["usage"], {"output_tokens": 5})
        self.assertEqual(events[-1]["type"], "message_stop")

    async def test_empty_finished_stream_keeps_cache_and_physical_usage(self):
        self.output_text = ""
        for user, completion_tokens in (
            ("empty-output-user", 1),
            (None, 1),
            (None, 0),
        ):
            self.stream_counts = (completion_tokens,)
            for ttl in ("5m", "1h"):
                with self.subTest(
                    user=user, completion_tokens=completion_tokens, ttl=ttl
                ):
                    response = await self.serving.handle_messages(
                        self.request(stream=True, user=user, options={"ttl": ttl}),
                        self.raw_request(),
                    )
                    self.assertEqual(response.status_code, 200)
                    events = [
                        json.loads(frame.split("data: ", 1)[1])
                        async for frame in response.body_iterator
                    ]
                    self.assertFalse(any(event["type"] == "error" for event in events))
                    starts = [
                        event for event in events if event["type"] == "message_start"
                    ]
                    self.assertEqual(len(starts), 1)
                    read_tokens, write_tokens = (16, 16) if user else (32, 0)
                    self.assertEqual(
                        starts[0]["message"]["usage"],
                        {
                            "input_tokens": 43 - read_tokens - write_tokens,
                            "output_tokens": 0,
                            "cache_read_input_tokens": read_tokens,
                            "cache_creation_input_tokens": write_tokens,
                            "cache_creation": {
                                "ephemeral_5m_input_tokens": (
                                    write_tokens if ttl == "5m" else 0
                                ),
                                "ephemeral_1h_input_tokens": (
                                    write_tokens if ttl == "1h" else 0
                                ),
                            },
                        },
                    )
                    final = next(
                        event for event in events if event["type"] == "message_delta"
                    )
                    self.assertEqual(
                        final["usage"], {"output_tokens": completion_tokens}
                    )
                    self.assertEqual(events[-1]["type"], "message_stop")

    async def test_child_conversion_result_is_published_on_parent_request(self):
        class ChildPreprocessor:
            calls = 0

            async def convert(preprocessor, handler, request, raw_request):
                preprocessor.calls += 1
                child_request = Request(
                    {
                        "type": "http",
                        "headers": [],
                        "state": dict(raw_request.state._state),
                    }
                )
                converted = handler._convert_to_internal_request(request, child_request)
                child_request.state.cache_identity = "child-state-is-not-parent-state"
                return pickle.loads(pickle.dumps(converted))

        preprocessor = ChildPreprocessor()
        self.chat.tokenizer_manager.process_request_preprocessor = preprocessor
        # Exercise the actual parent dispatch branch after a serialized child
        # result, without creating model workers or OS processes in this unit test.
        with patch.dict(
            self.chat._run_request_conversion.__func__.__globals__,
            {"ProcessRequestPreprocessor": ChildPreprocessor},
        ):
            for stream in (False, True):
                for user in ("process-user", None):
                    raw = self.raw_request(
                        "shared-verified-key-digest" if user else None
                    )
                    response = await self.serving.handle_messages(
                        self.request(stream=stream, user=user), raw
                    )
                    usage = await self.consume(response, stream)
                    if user:
                        expected = (
                            "cache-user-id:" + sha256(b"process-user").hexdigest()
                        )
                        self.assertEqual(raw.state.cache_identity, expected)
                        self.assertEqual(self.generated_identities[-1], expected)
                    else:
                        self.assertFalse(hasattr(raw.state, "cache_identity"))
                        self.assertIsNone(self.generated_identities[-1])
                        self.assertEqual(usage["cache_read_input_tokens"], 32)
                        self.assertEqual(usage["cache_creation_input_tokens"], 0)
        self.assertEqual(preprocessor.calls, 4)

    async def test_absent_identity_serves_physical_usage_with_no_creation(self):
        for stream in (False, True):
            for options in (None, {"ttl": "5m"}, {"ttl": "1h"}):
                with self.subTest(stream=stream, options=options):
                    raw = self.raw_request()
                    response = await self.serving.handle_messages(
                        self.request(stream=stream, options=options), raw
                    )
                    usage = await self.consume(response, stream)
                    self.assertEqual(usage["cache_read_input_tokens"], 32)
                    self.assertEqual(usage["cache_creation_input_tokens"], 0)
                    self.assertEqual(usage["input_tokens"], 11)
                    self.assertEqual(
                        usage["cache_creation"],
                        {
                            "ephemeral_5m_input_tokens": 0,
                            "ephemeral_1h_input_tokens": 0,
                        },
                    )
                    self.assertFalse(hasattr(raw.state, "cache_identity"))
                    self.assertIsNone(self.generated_identities[-1])
        self.assertEqual(len(self.generated_identities), 6)

    async def test_verified_bearer_fallback_is_preserved(self):
        identity = "api-key:" + sha256(b"verified-direct-key").hexdigest()
        for stream in (False, True):
            raw = self.raw_request(identity)
            response = await self.serving.handle_messages(
                self.request(stream=stream), raw
            )
            await self.consume(response, stream)
            self.assertEqual(raw.state.cache_identity, identity)
            self.assertEqual(self.generated_identities[-1], identity)

    async def test_disabled_and_prefill_requests_do_not_publish_identity(self):
        for role, enabled in (("decode", False), ("prefill", True)):
            self.disagg.disaggregation_mode = role
            self.config.enable_cache_accounting = enabled
            for stream in (False, True):
                raw = self.raw_request()
                response = await self.serving.handle_messages(
                    self.request(stream=stream, user="unused-body-user"), raw
                )
                await self.consume(response, stream)
                self.assertFalse(hasattr(raw.state, "cache_identity"))
                self.assertIsNone(self.generated_identities[-1])


class TestAnthropicCacheUsage(unittest.IsolatedAsyncioTestCase):
    def request(self, **kwargs):
        return AnthropicMessagesRequest(
            model="test",
            messages=[{"role": "user", "content": "hello"}],
            max_tokens=2,
            **kwargs,
        )

    def usage(self, ttl="1h"):
        snapshot = {
            **SNAPSHOT,
            "ttl_seconds": 300 if ttl == "5m" else 3600,
        }
        return UsageProcessor.calculate_token_usage(
            prompt_tokens=43,
            completion_tokens=2,
            cached_tokens=PromptTokensDetails(cached_tokens=40),
            cache_usage=snapshot,
        )

    def test_top_level_prompt_cache_options_is_preserved(self):
        serving = AnthropicServing(_FakeChat())
        for ttl in ("5m", "1h"):
            request = self.request(
                prompt_cache_options={"mode": "implicit", "ttl": ttl}
            )
            chat_request = serving._convert_to_chat_completion_request(request)
            self.assertEqual(chat_request.prompt_cache_options.mode, "implicit")
            self.assertEqual(chat_request.prompt_cache_options.ttl, ttl)

    def test_missing_ttl_defaults_to_five_minutes_during_conversion(self):
        request = self.request(prompt_cache_options={"mode": "implicit"})
        self.assertEqual(request.prompt_cache_options.ttl, "5m")
        chat_request = AnthropicServing(
            _FakeChat()
        )._convert_to_chat_completion_request(request)
        self.assertEqual(chat_request.prompt_cache_options.mode, "implicit")
        self.assertEqual(chat_request.prompt_cache_options.ttl, "5m")

    def test_missing_options_defer_to_server_default(self):
        request = self.request()
        self.assertIsNone(request.prompt_cache_options)
        chat_request = AnthropicServing(
            _FakeChat()
        )._convert_to_chat_completion_request(request)
        self.assertIsNone(chat_request.prompt_cache_options)

    def test_cache_snapshot_takes_precedence_over_legacy_metrics(self):
        result = _anthropic_usage_from_openai(
            self.usage(), include_input=True, include_output=True
        )
        self.assertEqual(result.input_tokens, 11)
        self.assertEqual(result.cache_read_input_tokens, 16)
        self.assertEqual(result.cache_creation_input_tokens, 16)
        self.assertEqual(result.cache_creation.ephemeral_5m_input_tokens, 0)
        self.assertEqual(result.cache_creation.ephemeral_1h_input_tokens, 16)
        self.assertEqual(
            result.input_tokens
            + result.cache_read_input_tokens
            + result.cache_creation_input_tokens,
            43,
        )

    def test_legacy_usage_keeps_physical_mapping(self):
        usage = self.usage()
        usage.cache_usage = None
        usage.prompt_tokens_details = PromptTokensDetails(cached_tokens=40)
        result = _anthropic_usage_from_openai(
            usage, include_input=True, include_output=True
        )
        self.assertEqual(result.input_tokens, 3)
        self.assertEqual(result.cache_read_input_tokens, 40)
        self.assertNotIn("cache_creation", result.model_dump(exclude_none=True))
        self.assertNotIn(
            "cache_creation_input_tokens", result.model_dump(exclude_none=True)
        )

    def test_non_streaming_conversion_carries_both_read_and_write(self):
        for ttl in ("5m", "1h"):
            with self.subTest(ttl=ttl):
                response = ChatCompletionResponse(
                    id="cache-non-stream",
                    model="test",
                    choices=[
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "hi"},
                            "finish_reason": "stop",
                        }
                    ],
                    usage=self.usage(ttl),
                )
                result = AnthropicServing(_FakeChat())._convert_response(response)
                self.assertEqual(result.usage.input_tokens, 11)
                self.assertEqual(result.usage.cache_read_input_tokens, 16)
                self.assertEqual(result.usage.cache_creation_input_tokens, 16)
                self.assertEqual(
                    result.usage.cache_creation.model_dump(),
                    {
                        "ephemeral_5m_input_tokens": 16 if ttl == "5m" else 0,
                        "ephemeral_1h_input_tokens": 16 if ttl == "1h" else 0,
                    },
                )

    def test_input_tokens_are_derived_and_overreported_cache_is_rejected(self):
        usage = self.usage()
        self.assertEqual(
            usage.cache_usage.model_dump(),
            {"ttl_seconds": 3600, "read_tokens": 16, "write_tokens": 16},
        )
        result = _anthropic_usage_from_openai(
            usage.model_copy(update={"prompt_tokens": 32}),
            include_input=True,
            include_output=True,
        )
        self.assertEqual(result.input_tokens, 0)
        with self.assertRaisesRegex(ValueError, "exceed prompt_tokens"):
            _anthropic_usage_from_openai(
                usage.model_copy(update={"prompt_tokens": 31}),
                include_input=True,
                include_output=True,
            )

    async def test_stream_message_start_has_snapshot_and_delta_only_output(self):
        for ttl in ("5m", "1h"):
            with self.subTest(ttl=ttl):
                public_usage = self.usage(ttl).model_dump()
                self.assertNotIn("cache_usage", public_usage)
                frames = [
                    _stream_chunk({"role": "assistant", "content": ""}),
                    _stream_chunk({"content": "hi"}, public_usage),
                    _stream_chunk(finish_reason="stop"),
                    "data: "
                    + json.dumps(
                        {
                            "id": "test",
                            "created": 0,
                            "model": "test",
                            "choices": [],
                            "usage": self.usage(ttl).model_dump(),
                        }
                    )
                    + "\n\n",
                    "data: [DONE]\n\n",
                ]
                serving = AnthropicServing(_FakeChat(frames))
                request = self.request(
                    stream=True, prompt_cache_options={"mode": "implicit", "ttl": ttl}
                )
                chat_request = serving._convert_to_chat_completion_request(request)
                adapted_request, chat_request = (
                    serving.openai_serving_chat._convert_to_internal_request(
                        chat_request, None
                    )
                )
                events = [
                    json.loads(frame.split("data: ", 1)[1])
                    async for frame in serving._generate_anthropic_stream(
                        adapted_request, chat_request, request, None
                    )
                ]
                start = next(
                    event for event in events if event["type"] == "message_start"
                )
                final = next(
                    event for event in events if event["type"] == "message_delta"
                )
                self.assertEqual(start["message"]["usage"]["input_tokens"], 11)
                self.assertEqual(start["message"]["usage"]["output_tokens"], 0)
                self.assertEqual(
                    start["message"]["usage"]["cache_read_input_tokens"], 16
                )
                self.assertEqual(
                    start["message"]["usage"]["cache_creation_input_tokens"], 16
                )
                self.assertEqual(
                    start["message"]["usage"]["cache_creation"],
                    {
                        "ephemeral_5m_input_tokens": 16 if ttl == "5m" else 0,
                        "ephemeral_1h_input_tokens": 16 if ttl == "1h" else 0,
                    },
                )
                self.assertEqual(final["usage"], {"output_tokens": 2})

    async def test_empty_finished_stream_without_usage_still_closes(self):
        serving = AnthropicServing(
            _FakeChat(
                [
                    _stream_chunk({"role": "assistant", "content": ""}),
                    _stream_chunk(finish_reason="stop"),
                    "data: [DONE]\n\n",
                ]
            )
        )
        response = await serving.handle_messages(self.request(stream=True), None)
        self.assertEqual(response.status_code, 200)
        events = [
            json.loads(frame.split("data: ", 1)[1])
            async for frame in response.body_iterator
        ]
        self.assertFalse(any(event["type"] == "error" for event in events))
        starts = [event for event in events if event["type"] == "message_start"]
        self.assertEqual(len(starts), 1)
        self.assertEqual(
            starts[0]["message"]["usage"], {"input_tokens": 0, "output_tokens": 0}
        )
        final = next(event for event in events if event["type"] == "message_delta")
        self.assertEqual(final["usage"], {"output_tokens": 0})
        self.assertEqual(events[-1]["type"], "message_stop")

    async def test_missing_options_default_to_five_minutes_for_both_response_modes(
        self,
    ):
        for stream in (True, False):
            with self.subTest(stream=stream):
                frames = [
                    _stream_chunk({"role": "assistant", "content": ""}),
                    _stream_chunk(
                        {"content": "hi"},
                        self.usage("5m").model_dump(),
                    ),
                    _stream_chunk(finish_reason="stop"),
                    "data: [DONE]\n\n",
                ]
                serving = AnthropicServing(_FakeChat(frames))
                response = await serving.handle_messages(
                    self.request(stream=stream), None
                )
                self.assertEqual(response.status_code, 200)
                if stream:
                    events = [
                        json.loads(frame.split("data: ", 1)[1])
                        async for frame in response.body_iterator
                    ]
                    usage = next(
                        event["message"]["usage"]
                        for event in events
                        if event["type"] == "message_start"
                    )
                else:
                    usage = json.loads(response.body)["usage"]
                self.assertEqual(usage["input_tokens"], 11)
                self.assertEqual(usage["cache_read_input_tokens"], 16)
                self.assertEqual(usage["cache_creation_input_tokens"], 16)
                self.assertEqual(
                    usage["cache_creation"],
                    {"ephemeral_5m_input_tokens": 16, "ephemeral_1h_input_tokens": 0},
                )

    async def test_accounting_failure_returns_http_503_before_stream_headers(self):
        for stream, options in (
            (stream, options)
            for stream in (True, False)
            for options in (None, {"mode": "implicit"})
        ):
            with self.subTest(stream=stream, options=options):
                serving = AnthropicServing(_UnavailableChat())
                request = self.request(
                    stream=stream,
                    **(
                        {"prompt_cache_options": options} if options is not None else {}
                    ),
                )
                response = await serving.handle_messages(request, None)
                self.assertEqual(response.status_code, 503)
                self.assertEqual(
                    json.loads(response.body)["error"]["type"], "overloaded_error"
                )


if __name__ == "__main__":
    unittest.main()
