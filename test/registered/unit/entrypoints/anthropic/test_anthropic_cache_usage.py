import json
import unittest
from types import SimpleNamespace

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from fastapi import HTTPException

from sglang.srt.entrypoints.anthropic.protocol import AnthropicMessagesRequest
from sglang.srt.entrypoints.anthropic.serving import (
    AnthropicServing,
    _anthropic_usage_from_openai,
)
from sglang.srt.entrypoints.openai.protocol import (
    ChatCompletionResponse,
    PromptTokensDetails,
)
from sglang.srt.entrypoints.openai.usage_processor import UsageProcessor
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


SNAPSHOT = {
    "read_tokens": 16,
    "creation_5m_tokens": 0,
    "creation_1h_tokens": 16,
    "normal_input_tokens": 11,
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

    async def _handle_non_streaming_request(
        self, adapted_request, request, raw_request
    ):
        one_hour = adapted_request.cache_ttl_seconds == 3600
        usage = UsageProcessor.calculate_token_usage(
            prompt_tokens=43,
            completion_tokens=2,
            cache_usage={
                **SNAPSHOT,
                "creation_5m_tokens": 0 if one_hour else 16,
                "creation_1h_tokens": 16 if one_hour else 0,
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
            "creation_5m_tokens": 16 if ttl == "5m" else 0,
            "creation_1h_tokens": 16 if ttl == "1h" else 0,
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
            usage=self.usage(),
        )
        result = AnthropicServing(_FakeChat())._convert_response(response)
        self.assertEqual(result.usage.cache_read_input_tokens, 16)
        self.assertEqual(result.usage.cache_creation_input_tokens, 16)

    async def test_stream_message_start_has_snapshot_and_delta_only_output(self):
        for ttl, delta in (
            (ttl, delta)
            for ttl in ("5m", "1h")
            for delta in ({"role": "assistant", "content": ""}, {"content": "hi"})
        ):
            with self.subTest(ttl=ttl, delta=delta):
                public_usage = self.usage(ttl).model_dump()
                self.assertNotIn("cache_usage", public_usage)
                frames = [
                    _stream_chunk(delta, public_usage),
                    _stream_chunk(finish_reason="stop"),
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

    async def test_missing_options_default_to_five_minutes_for_both_response_modes(
        self,
    ):
        for stream in (True, False):
            with self.subTest(stream=stream):
                frames = [
                    _stream_chunk(
                        {"role": "assistant", "content": ""},
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
