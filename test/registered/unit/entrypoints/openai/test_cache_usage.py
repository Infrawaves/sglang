import json
import unittest
from hashlib import sha256
from http import HTTPStatus
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from fastapi import HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import ValidationError

from sglang.srt.entrypoints.openai.protocol import (
    CacheUsage,
    ChatCompletionRequest,
    MessageProcessingResult,
    PromptCacheOptions,
    PromptTokensDetails,
    UsageInfo,
)
from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
from sglang.srt.entrypoints.openai.usage_processor import UsageProcessor
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


API_KEY_IDENTITY = "api-key:" + sha256(b"verified-key").hexdigest()


SNAPSHOT = {
    "ttl_seconds": 300,
    "read_tokens": 16,
    "write_tokens": 16,
}


class TestCacheUsage(unittest.TestCase):
    def test_write_headers_include_both_tiers_and_zero(self):
        for snapshot in (
            CacheUsage(ttl_seconds=300, write_tokens=700),
            CacheUsage(ttl_seconds=3600, write_tokens=700),
            CacheUsage(ttl_seconds=300, read_tokens=200),
            CacheUsage(ttl_seconds=3600, read_tokens=200),
        ):
            with self.subTest(snapshot=snapshot):
                self.assertEqual(
                    snapshot.response_headers(),
                    {
                        "Msh-Usage-Cache-Write-Tokens-5m": str(
                            snapshot.write_tokens if snapshot.ttl_seconds == 300 else 0
                        ),
                        "Msh-Usage-Cache-Write-Tokens-1h": str(
                            snapshot.write_tokens if snapshot.ttl_seconds == 3600 else 0
                        ),
                    },
                )

    def test_cache_snapshot_and_legacy_metrics_are_independent(self):
        snapshot = {
            "ttl_seconds": 300,
            "read_tokens": 200,
            "write_tokens": 700,
        }
        result = UsageProcessor.calculate_response_usage(
            [
                {
                    "meta_info": {
                        "prompt_tokens": 1000,
                        "completion_tokens": 100,
                        "cached_tokens": 32,
                        "cache_usage": snapshot,
                    }
                }
            ],
            enable_cache_report=True,
        )
        self.assertEqual(
            result.model_dump(exclude={"completion_tokens_details"}),
            {
                "prompt_tokens": 1000,
                "completion_tokens": 100,
                "total_tokens": 1100,
                "prompt_tokens_details": {
                    "cached_tokens": 200,
                    "cache_write_tokens": 700,
                },
            },
        )
        self.assertEqual(result.cache_usage.model_dump(), snapshot)

    def test_cache_usage_does_not_require_legacy_cache_report(self):
        result = UsageProcessor.calculate_response_usage(
            [{"meta_info": {"prompt_tokens": 43, "cache_usage": SNAPSHOT}}]
        )
        self.assertEqual(
            result.prompt_tokens_details.model_dump(),
            {"cached_tokens": 16, "cache_write_tokens": 16},
        )
        self.assertEqual(result.cache_usage.model_dump(), SNAPSHOT)
        tail_only = UsageProcessor.calculate_token_usage(
            prompt_tokens=11,
            completion_tokens=1,
            cache_usage={"ttl_seconds": 300},
        )
        self.assertEqual(
            tail_only.model_dump()["prompt_tokens_details"],
            {"cached_tokens": 0, "cache_write_tokens": 0},
        )

    def test_streaming_snapshots_do_not_accumulate_per_chunk(self):
        for completion_tokens in (1, 2, 3):
            result = UsageProcessor.calculate_streaming_usage(
                {0: 43},
                {0: 0},
                {0: completion_tokens},
                {0: 32},
                1,
                cache_usage={0: SNAPSHOT},
            )
            self.assertEqual(result.cache_usage.model_dump(), SNAPSHOT)
            self.assertEqual(result.completion_tokens, completion_tokens)

    def test_continuous_usage_carries_snapshot(self):
        result = UsageProcessor.calculate_token_usage(
            prompt_tokens=43, completion_tokens=1, cache_usage=SNAPSHOT
        )
        self.assertEqual(result.cache_usage.model_dump(), SNAPSHOT)
        self.assertNotIn("cache_usage", result.model_dump())
        self.assertEqual(
            result.model_dump()["prompt_tokens_details"],
            {"cached_tokens": 16, "cache_write_tokens": 16},
        )

    def test_cache_details_are_copied_before_overriding_legacy_counters(self):
        original = PromptTokensDetails(cached_tokens=32, image_tokens=5)
        result = UsageProcessor.calculate_token_usage(
            prompt_tokens=43,
            completion_tokens=2,
            cached_tokens=original,
            image_tokens=7,
            cache_usage=SNAPSHOT,
        )
        self.assertEqual(
            original.model_dump(), {"cached_tokens": 32, "image_tokens": 5}
        )
        self.assertEqual(
            result.prompt_tokens_details.model_dump(),
            {"cached_tokens": 16, "cache_write_tokens": 16, "image_tokens": 7},
        )

    def test_usage_requires_supported_ttl_and_nonnegative_counts(self):
        for snapshot in (
            {},
            {"ttl_seconds": 60},
            {"ttl_seconds": 300.0},
            {"ttl_seconds": "300"},
            {"ttl_seconds": 300, "read_tokens": -1},
            {"ttl_seconds": 3600, "write_tokens": -1},
            {"ttl_seconds": 300, "read_tokens": True},
            {"ttl_seconds": 3600, "write_tokens": 1.0},
            {
                "ttl_seconds": 300,
                "creation_5m_tokens": 16,
                "creation_1h_tokens": 0,
                "normal_input_tokens": 11,
            },
        ):
            with self.subTest(snapshot=snapshot), self.assertRaises(ValidationError):
                CacheUsage.model_validate(snapshot)

    def test_cache_counts_may_leave_uncached_input_but_not_exceed_prompt(self):
        for prompt_tokens in (32, 43):
            with self.subTest(prompt_tokens=prompt_tokens):
                result = UsageProcessor.calculate_token_usage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=1,
                    cache_usage=SNAPSHOT,
                )
                self.assertEqual(result.prompt_tokens, prompt_tokens)
                self.assertEqual(result.cache_usage.model_dump(), SNAPSHOT)
        with self.assertRaisesRegex(ValueError, "must not exceed prompt_tokens"):
            UsageProcessor.calculate_token_usage(
                prompt_tokens=31, completion_tokens=1, cache_usage=SNAPSHOT
            )
        empty = UsageProcessor.calculate_token_usage(
            prompt_tokens=0, completion_tokens=0, cache_usage={"ttl_seconds": 3600}
        )
        self.assertEqual(empty.prompt_tokens_details.cache_write_tokens, 0)

    def test_aggregation_sums_only_counts_and_preserves_request_ttl(self):
        for ttl_seconds in (300, 3600):
            snapshots = [
                {"ttl_seconds": ttl_seconds, "read_tokens": 10, "write_tokens": 20},
                {"ttl_seconds": ttl_seconds, "read_tokens": 30, "write_tokens": 40},
            ]
            responses = [
                {"meta_info": {"prompt_tokens": 100, "cache_usage": snapshot}}
                for snapshot in snapshots
            ]
            with self.subTest(ttl_seconds=ttl_seconds):
                for result in (
                    UsageProcessor.calculate_response_usage(responses),
                    UsageProcessor.calculate_streaming_usage(
                        {0: 100, 1: 100},
                        {},
                        {},
                        {},
                        1,
                        cache_usage=dict(enumerate(snapshots)),
                    ),
                ):
                    self.assertEqual(
                        result.cache_usage.model_dump(),
                        {
                            "ttl_seconds": ttl_seconds,
                            "read_tokens": 40,
                            "write_tokens": 60,
                        },
                    )
                    self.assertEqual(result.prompt_tokens, 200)
                    self.assertEqual(result.prompt_tokens_details.cached_tokens, 40)
                    self.assertEqual(
                        result.prompt_tokens_details.cache_write_tokens, 60
                    )

    def test_aggregation_rejects_mixed_request_ttls(self):
        snapshots = [{"ttl_seconds": 300}, {"ttl_seconds": 3600}]
        with self.assertRaisesRegex(ValueError, "same ttl_seconds"):
            UsageProcessor.calculate_response_usage(
                [
                    {"meta_info": {"prompt_tokens": 1, "cache_usage": snapshot}}
                    for snapshot in snapshots
                ]
            )
        with self.assertRaisesRegex(ValueError, "same ttl_seconds"):
            UsageProcessor.calculate_streaming_usage(
                {0: 1, 1: 1},
                {},
                {},
                {},
                1,
                cache_usage=dict(enumerate(snapshots)),
            )

    def test_streaming_choices_use_recorded_input_and_sum_output_counts(self):
        result = UsageProcessor.calculate_streaming_usage(
            prompt_tokens={0: 100, 1: 101},
            reasoning_tokens={0: 2, 1: 4},
            completion_tokens={0: 3, 1: 5},
            cached_tokens={0: 66, 1: 88},
            n_choices=2,
            enable_cache_report=True,
            cache_usage={
                0: {"ttl_seconds": 300, "read_tokens": 7, "write_tokens": 4},
                1: {"ttl_seconds": 300, "read_tokens": 8, "write_tokens": 5},
            },
        )
        self.assertEqual(result.prompt_tokens, 100)
        self.assertEqual(result.completion_tokens, 8)
        self.assertEqual(result.total_tokens, 108)
        self.assertEqual(result.completion_tokens_details.reasoning_tokens, 6)
        self.assertEqual(
            result.prompt_tokens_details.model_dump(),
            {"cached_tokens": 7, "cache_write_tokens": 4},
        )
        self.assertEqual(
            result.cache_usage.model_dump(),
            {"ttl_seconds": 300, "read_tokens": 7, "write_tokens": 4},
        )

    def test_old_wire_format_omits_cache_field(self):
        self.assertNotIn("cache_usage", UsageInfo().model_dump())
        result = UsageProcessor.calculate_streaming_usage({0: 4}, {}, {}, {}, 1)
        self.assertNotIn("cache_usage", result.model_dump())
        legacy = UsageProcessor.calculate_token_usage(
            prompt_tokens=43,
            completion_tokens=2,
            cached_tokens=PromptTokensDetails(cached_tokens=32),
        )
        self.assertEqual(
            legacy.model_dump()["prompt_tokens_details"], {"cached_tokens": 32}
        )


class TestCacheRequest(unittest.TestCase):
    def setUp(self):
        self.chat = object.__new__(OpenAIServingChat)
        self.chat.tokenizer_manager = SimpleNamespace(
            model_config=SimpleNamespace(is_multimodal=False)
        )
        self.config = SimpleNamespace(enable_cache_accounting=True)
        self.config_patch = patch(
            "sglang.srt.entrypoints.openai.serving_chat.get_serving",
            return_value=self.config,
        )
        self.config_patch.start()
        self.addCleanup(self.config_patch.stop)
        self.disagg = SimpleNamespace(disaggregation_mode="decode")
        self.disagg_patch = patch(
            "sglang.srt.entrypoints.openai.serving_chat.get_disagg",
            return_value=self.disagg,
        )
        self.disagg_patch.start()
        self.addCleanup(self.disagg_patch.stop)

    def request(self, **kwargs):
        return ChatCompletionRequest(
            messages=[{"role": "user", "content": "hello"}],
            prompt_cache_options={"mode": "implicit"},
            **kwargs,
        )

    def trusted_request(self):
        return SimpleNamespace(state=SimpleNamespace(api_key=API_KEY_IDENTITY))

    def test_ttl_defaults_and_validation(self):
        self.assertEqual(PromptCacheOptions().mode, "implicit")
        self.assertEqual(PromptCacheOptions(mode="implicit").ttl, "5m")
        self.assertNotIn("cache_control", ChatCompletionRequest.model_fields)
        self.assertIsNone(
            ChatCompletionRequest(
                messages=[{"role": "user", "content": "hello"}]
            ).prompt_cache_options
        )
        for invalid in ({"ttl": "10m"}, {"mode": "explicit"}, {"api_key": "x"}):
            with self.subTest(invalid=invalid), self.assertRaises(ValidationError):
                PromptCacheOptions(**invalid)

    def test_trusted_identity_applies_to_explicit_and_default_options(self):
        request = self.request(user="spoofed", api_key="spoofed")
        self.assertIsNone(self.chat._cache_request_params(request, None)["api_key"])
        params = self.chat._cache_request_params(request, self.trusted_request())
        self.assertEqual(
            params,
            {
                "api_key": API_KEY_IDENTITY,
                "cache_required": True,
                "cache_ttl_seconds": 300,
            },
        )
        request.prompt_cache_options = PromptCacheOptions(ttl="1h")
        self.assertEqual(
            self.chat._cache_request_params(request, self.trusted_request())[
                "cache_ttl_seconds"
            ],
            3600,
        )
        request.prompt_cache_options = None
        self.assertIsNone(self.chat._cache_request_params(request, None)["api_key"])
        self.assertEqual(
            self.chat._cache_request_params(request, self.trusted_request()),
            {
                "api_key": API_KEY_IDENTITY,
                "cache_required": False,
                "cache_ttl_seconds": 300,
            },
        )

    def test_missing_identity_keeps_ttl_for_physical_usage_only(self):
        for options, ttl_seconds in (
            (None, 300),
            (PromptCacheOptions(), 300),
            (PromptCacheOptions(ttl="1h"), 3600),
        ):
            for identity in (None, "", " \t", 123):
                with self.subTest(options=options, identity=identity):
                    request = self.request(
                        user="untrusted-openai-user", api_key="body-key"
                    )
                    request.prompt_cache_options = options
                    raw = SimpleNamespace(state=SimpleNamespace(api_key=identity))
                    self.assertEqual(
                        self.chat._cache_request_params(request, raw),
                        {
                            "api_key": None,
                            "cache_required": options is not None,
                            "cache_ttl_seconds": ttl_seconds,
                        },
                    )

    def test_disabled_server_rejects_explicit_options(self):
        raw_request = self.trusted_request()
        self.config.enable_cache_accounting = False
        omitted = self.request()
        omitted.prompt_cache_options = None
        self.assertEqual(self.chat._cache_request_params(omitted, None), {})
        with self.assertRaisesRegex(ValueError, "not enabled"):
            self.chat._cache_request_params(self.request(), raw_request)

    def test_gateway_user_identity_works_without_engine_authentication(self):
        request = self.request(cache_user_id="gateway-user-a")
        identity = "cache-user-id:" + sha256(b"gateway-user-a").hexdigest()
        for options, ttl in (
            (None, 300),
            (PromptCacheOptions(), 300),
            (PromptCacheOptions(ttl="1h"), 3600),
        ):
            with self.subTest(options=options):
                request.prompt_cache_options = options
                params = self.chat._cache_request_params(request, None)
                self.assertEqual(params["api_key"], identity)
                self.assertEqual(params["cache_ttl_seconds"], ttl)
                self.assertEqual(params["cache_required"], options is not None)

    def test_gateway_users_are_isolated_under_one_engine_credential(self):
        identities = [
            self.chat._cache_request_params(
                self.request(cache_user_id=user), self.trusted_request()
            )["api_key"]
            for user in ("gateway-user-a", "gateway-user-b", "gateway-user-a")
        ]
        self.assertEqual(identities[0], identities[2])
        self.assertNotEqual(identities[0], identities[1])
        self.assertNotIn(API_KEY_IDENTITY, identities)

    def test_gateway_identity_validation_and_redacted_repr(self):
        for invalid in ("", " \t\n", 123, [], {}):
            with self.subTest(identity=invalid), self.assertRaises(ValidationError):
                self.request(cache_user_id=invalid)
        request = self.request(cache_user_id="private-gateway-user")
        self.assertNotIn("private-gateway-user", repr(request))
        self.assertEqual(request.model_dump()["cache_user_id"], "private-gateway-user")

    def test_request_shape_is_left_to_tokenizer_manager(self):
        # The chat layer no longer rejects shapes; TokenizerManager decides.
        raw_request = self.trusted_request()
        self.chat.tokenizer_manager.model_config.is_multimodal = True
        for overrides in (
            {"n": 2},
            {"session_id": "s"},
            {"session_params": {}},
            {"custom_params": {"beam_width": 2}},
        ):
            with self.subTest(overrides=overrides):
                params = self.chat._cache_request_params(
                    self.request(**overrides), raw_request
                )
                self.assertEqual(params["cache_ttl_seconds"], 300)

    def test_prefill_ignores_options_before_config_and_identity_checks(self):
        self.disagg.disaggregation_mode = "prefill"
        request = self.request()
        for enabled in (False, True):
            self.config.enable_cache_accounting = enabled
            for options in (None, PromptCacheOptions(), PromptCacheOptions(ttl="1h")):
                with self.subTest(enabled=enabled, options=options):
                    request.prompt_cache_options = options
                    self.assertEqual(self.chat._cache_request_params(request, None), {})

    def test_cache_k3_count_uses_complete_model_input(self):
        self.chat.chat_encoding_spec = "kimi_k3"
        self.assertEqual(self.chat._reported_prompt_tokens({"prompt_tokens": 43}), 40)
        self.assertEqual(
            self.chat._reported_prompt_tokens(
                {"prompt_tokens": 43, "cache_usage": SNAPSHOT}
            ),
            43,
        )


class TestCacheStream(unittest.IsolatedAsyncioTestCase):
    async def test_empty_completion_omits_role_usage_and_keeps_final_usage(self):
        async def generate_request(*args):
            yield {
                "text": "",
                "meta_info": {
                    "id": "cache-empty",
                    "prompt_tokens": 43,
                    "completion_tokens": 1,
                    "cached_tokens": 32,
                    "cache_usage": SNAPSHOT,
                    "finish_reason": {"type": "stop"},
                },
            }

        chat = object.__new__(OpenAIServingChat)
        chat.tokenizer_manager = SimpleNamespace(generate_request=generate_request)
        chat.chat_encoding_spec = None
        chat.reasoning_parser = None
        chat.tool_call_parser = None
        chat._tool_call_parsing_active = lambda request: False
        chat._should_return_input_ids = lambda request: False
        chat._should_return_output_ids = lambda request: False
        request = ChatCompletionRequest(
            messages=[{"role": "user", "content": "hello"}],
            stream=True,
            stream_options={"include_usage": True, "continuous_usage_stats": True},
        )
        config = SimpleNamespace(
            enable_cache_report=False,
            incremental_streaming_output=False,
            stream_response_default_include_usage=False,
        )
        with patch(
            "sglang.srt.entrypoints.openai.serving_chat.get_serving",
            return_value=config,
        ):
            frames = [
                frame async for frame in chat._generate_chat_stream(None, request, None)
            ]
        chunks = [json.loads(frame[6:]) for frame in frames if "[DONE]" not in frame]
        self.assertEqual(chunks[0]["choices"][0]["delta"]["role"], "assistant")
        self.assertNotIn("usage", chunks[0])
        self.assertNotIn("cache_usage", chunks[-1]["usage"])
        self.assertEqual(
            chunks[-1]["usage"]["prompt_tokens_details"],
            {"cached_tokens": 16, "cache_write_tokens": 16},
        )
        self.assertEqual(chunks[-1]["usage"]["prompt_tokens"], 43)
        self.assertEqual(chunks[-1]["usage"]["completion_tokens"], 1)


class TestOpenAIChatCachePath(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.adapted_requests = []
        self.fail_accounting = False
        self.snapshot_override = None
        self.abort_first = False
        self.output_prompt_tokens = 43
        self.physical_cached_tokens = 32
        self.accounting_read_tokens = 16
        self.accounting_write_tokens = 16

        async def run_in_request_preprocessor(func, *args):
            return func(*args)

        async def generate_request(adapted_request, raw_request):
            self.adapted_requests.append(adapted_request)
            if self.fail_accounting:
                raise HTTPException(
                    status_code=503,
                    detail="Cache accounting is temporarily unavailable",
                )
            snapshot = {
                **SNAPSHOT,
                "ttl_seconds": adapted_request.cache_ttl_seconds,
                "read_tokens": self.accounting_read_tokens,
                "write_tokens": self.accounting_write_tokens,
            }
            if adapted_request.api_key is None:
                snapshot.update(read_tokens=self.physical_cached_tokens, write_tokens=0)
            if self.snapshot_override is not None:
                snapshot = dict(self.snapshot_override)
            zero_output = adapted_request.sampling_params.get("max_new_tokens") == 0
            counts = (
                (0,) if zero_output else ((1, 2) if adapted_request.stream else (2,))
            )
            for count in counts:
                output = {
                    "text": "hi"[:count],
                    "meta_info": {
                        "id": "chat-cache-path",
                        "prompt_tokens": self.output_prompt_tokens,
                        "completion_tokens": count,
                        "cached_tokens": self.physical_cached_tokens,
                        **(
                            {"cache_usage": snapshot}
                            if adapted_request.cache_ttl_seconds is not None
                            else {}
                        ),
                        "weight_version": "test-version",
                        "finish_reason": (
                            {
                                "type": "abort",
                                "status_code": HTTPStatus.SERVICE_UNAVAILABLE,
                                "message": "Generation unavailable",
                            }
                            if self.abort_first
                            else (
                                {"type": "length"}
                                if zero_output
                                else ({"type": "stop"} if count == 2 else None)
                            )
                        ),
                    },
                }
                n_choices = adapted_request.sampling_params.get("n", 1)
                if n_choices == 1:
                    yield output
                elif adapted_request.stream:
                    for index in range(n_choices):
                        yield {**output, "index": index}
                else:
                    yield [dict(output) for _ in range(n_choices)]

        self.chat = object.__new__(OpenAIServingChat)
        self.chat.tokenizer_manager = SimpleNamespace(
            model_config=SimpleNamespace(is_multimodal=False),
            request_logger=SimpleNamespace(log_requests=False),
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
        # Rendering is independent of cache accounting. Supply a deterministic
        # encoding while exercising the real internal conversion and handlers.
        self.chat._process_messages = Mock(
            return_value=MessageProcessingResult(
                prompt="hello",
                prompt_ids=list(range(43)),
                image_data=None,
                audio_data=None,
                video_data=None,
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
        self.config_patch = patch(
            "sglang.srt.entrypoints.openai.serving_chat.get_serving",
            return_value=self.config,
        )
        self.config_patch.start()
        self.addCleanup(self.config_patch.stop)
        self.model_patch = patch(
            "sglang.srt.entrypoints.openai.serving_chat.get_model",
            return_value=SimpleNamespace(context_length=4096),
        )
        self.model_patch.start()
        self.addCleanup(self.model_patch.stop)
        self.disagg = SimpleNamespace(disaggregation_mode="decode")
        self.disagg_patch = patch(
            "sglang.srt.entrypoints.openai.serving_chat.get_disagg",
            return_value=self.disagg,
        )
        self.disagg_patch.start()
        self.addCleanup(self.disagg_patch.stop)

    def request(self, ttl, stream=False, *, omit_options=False, multimodal=False):
        options = {"mode": "implicit"}
        if ttl is not None:
            options["ttl"] = ttl
        return ChatCompletionRequest(
            model="test-model",
            messages=[
                {
                    "role": "user",
                    "content": (
                        [
                            {"type": "text", "text": "hello"},
                            {
                                "type": "image_url",
                                "image_url": {"url": "data:image/png;base64,ZmFrZQ=="},
                            },
                        ]
                        if multimodal
                        else "hello"
                    ),
                }
            ],
            max_completion_tokens=2,
            **({} if omit_options else {"prompt_cache_options": options}),
            user="untrusted-body-user",
            stream=stream,
            stream_options={"include_usage": True} if stream else None,
        )

    def raw_request(self):
        return Request(
            {
                "type": "http",
                "headers": [],
                "state": {"api_key": API_KEY_IDENTITY},
            }
        )

    def assert_cache_usage(self, usage, ttl):
        self.assertEqual(usage["prompt_tokens"], 43)
        self.assertEqual(usage["completion_tokens"], 2)
        self.assertEqual(usage["total_tokens"], 45)
        self.assertNotIn("cache_usage", usage)
        self.assertEqual(
            usage["prompt_tokens_details"],
            {"cached_tokens": 16, "cache_write_tokens": 16},
        )
        adapted_request = self.adapted_requests[-1]
        self.assertEqual(adapted_request.api_key, API_KEY_IDENTITY)
        self.assertEqual(
            adapted_request.cache_ttl_seconds, 300 if ttl in (None, "5m") else 3600
        )
        self.assertEqual(list(adapted_request.input_ids), list(range(43)))

    def assert_no_usage(self, chunk):
        self.assertEqual(chunk["choices"][0]["delta"]["role"], "assistant")
        self.assertEqual(chunk["choices"][0]["delta"].get("content"), "")
        self.assertNotIn("usage", chunk)

    def set_stream_outputs(self, outputs):
        self.stream_yielded_indices = []

        async def generate_request(adapted_request, raw_request):
            self.adapted_requests.append(adapted_request)
            for output in outputs:
                self.stream_yielded_indices.append(output.get("index", 0))
                yield output

        self.chat.tokenizer_manager.generate_request = generate_request

    def stream_output(
        self,
        text,
        completion,
        *,
        index=0,
        prompt=43,
        snapshot=SNAPSHOT,
        reasoning=0,
        finish=None,
    ):
        return {
            "index": index,
            "text": text,
            "meta_info": {
                "id": "cache-role-input",
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "reasoning_tokens": reasoning,
                "cached_tokens": snapshot["read_tokens"] if snapshot else 0,
                "finish_reason": finish,
                **({"cache_usage": dict(snapshot)} if snapshot is not None else {}),
            },
        }

    def configure_multimodal_output(self):
        # This API fixture simulates final Scheduler metadata. Media processing
        # and content/page hashing are exercised by the core/manager tests.
        self.chat.tokenizer_manager.model_config.is_multimodal = True
        self.chat._process_messages.return_value = MessageProcessingResult(
            prompt="hello <image>",
            prompt_ids=list(range(20)),
            image_data=["fixture-image"],
            audio_data=None,
            video_data=None,
            modalities=["image"],
            stop=[],
        )
        self.output_prompt_tokens = 770
        self.physical_cached_tokens = 333
        self.accounting_read_tokens = 256
        self.accounting_write_tokens = 512

    async def test_multimodal_final_input_usage_for_default_and_selected_ttls(self):
        self.configure_multimodal_output()
        for ttl, omitted in ((None, True), ("5m", False), ("1h", False)):
            for stream, include_usage, continuous in (
                (False, False, False),
                (True, True, False),
                (True, True, True),
                (True, False, False),
            ):
                with self.subTest(
                    ttl=ttl,
                    stream=stream,
                    include_usage=include_usage,
                    continuous=continuous,
                ):
                    request = self.request(
                        ttl, stream=stream, omit_options=omitted, multimodal=True
                    )
                    request.cache_user_id = "multimodal-gateway-user"
                    if stream:
                        request.stream_options.include_usage = include_usage
                        request.stream_options.continuous_usage_stats = continuous
                    response = await self.chat.handle_request(
                        request, Request({"type": "http", "headers": [], "state": {}})
                    )
                    adapted = self.adapted_requests[-1]
                    self.assertTrue(adapted.contains_mm_input())
                    self.assertEqual(adapted.image_data, ["fixture-image"])
                    self.assertEqual(adapted.text, "hello <image>")
                    self.assertEqual(
                        adapted.cache_ttl_seconds, 3600 if ttl == "1h" else 300
                    )
                    self.assertEqual(adapted.cache_required, not omitted)
                    expected_headers = {
                        "Msh-Usage-Cache-Write-Tokens-5m": (
                            "0" if ttl == "1h" else "512"
                        ),
                        "Msh-Usage-Cache-Write-Tokens-1h": (
                            "512" if ttl == "1h" else "0"
                        ),
                    }
                    if stream:
                        self.assertEqual(response.status_code, 200)
                        # Headers must be available before consuming the body.
                        for header, expected in expected_headers.items():
                            self.assertEqual(response.headers[header], expected)
                        frames = [frame async for frame in response.body_iterator]
                        self.assertEqual(frames[-1], "data: [DONE]\n\n")
                        chunks = [json.loads(frame[6:]) for frame in frames[:-1]]
                        self.assert_no_usage(chunks[0])
                        content = "".join(
                            choice["delta"].get("content") or ""
                            for chunk in chunks
                            for choice in chunk["choices"]
                        )
                        self.assertEqual(content, "hi")
                        tails = [chunk for chunk in chunks if chunk["choices"] == []]
                        self.assertEqual(len(tails), int(include_usage))
                        if not include_usage:
                            self.assertTrue(
                                all(chunk.get("usage") is None for chunk in chunks[1:])
                            )
                            continue
                        usage = tails[0]["usage"]
                        if continuous:
                            for chunk in chunks:
                                if chunk.get("usage") is not None:
                                    self.assertEqual(
                                        chunk["usage"]["prompt_tokens"], 770
                                    )
                                    self.assertEqual(
                                        chunk["usage"]["prompt_tokens_details"],
                                        {
                                            "cached_tokens": 256,
                                            "cache_write_tokens": 512,
                                        },
                                    )
                    else:
                        self.assertEqual(response.choices[0].message.content, "hi")
                        self.assertEqual(
                            response.usage.cache_usage.response_headers(),
                            expected_headers,
                        )
                        usage = response.model_dump()["usage"]
                    self.assertEqual(usage["prompt_tokens"], 770)
                    self.assertEqual(usage["completion_tokens"], 2)
                    self.assertEqual(usage["total_tokens"], 772)
                    self.assertEqual(
                        usage["prompt_tokens_details"],
                        {"cached_tokens": 256, "cache_write_tokens": 512},
                    )
                    self.assertNotIn("cache_usage", usage)

    async def test_multimodal_physical_fallback_keeps_usage_and_headers_consistent(
        self,
    ):
        self.configure_multimodal_output()
        for source in ("missing_identity", "redis_failure"):
            for ttl, omitted in ((None, True), ("5m", False), ("1h", False)):
                for stream in (False, True):
                    with self.subTest(source=source, ttl=ttl, stream=stream):
                        request = self.request(
                            ttl, stream=stream, omit_options=omitted, multimodal=True
                        )
                        if source == "redis_failure":
                            request.cache_user_id = "multimodal-gateway-user"
                            self.snapshot_override = {
                                "ttl_seconds": 3600 if ttl == "1h" else 300,
                                "read_tokens": 333,
                                "write_tokens": 0,
                            }
                        else:
                            self.snapshot_override = None
                        raw = Request({"type": "http", "headers": [], "state": {}})
                        response = await self.chat.handle_request(request, raw)
                        self.assertTrue(self.adapted_requests[-1].contains_mm_input())
                        self.assertEqual(
                            self.adapted_requests[-1].api_key is None,
                            source == "missing_identity",
                        )
                        if stream:
                            self.assertEqual(response.status_code, 200)
                            for tier in ("5m", "1h"):
                                self.assertEqual(
                                    response.headers[
                                        f"Msh-Usage-Cache-Write-Tokens-{tier}"
                                    ],
                                    "0",
                                )
                            frames = [frame async for frame in response.body_iterator]
                            self.assert_no_usage(json.loads(frames[0][6:]))
                            usage = json.loads(frames[-2][6:])["usage"]
                            self.assertEqual(frames[-1], "data: [DONE]\n\n")
                        else:
                            self.assertEqual(
                                response.usage.cache_usage.response_headers(),
                                {
                                    "Msh-Usage-Cache-Write-Tokens-5m": "0",
                                    "Msh-Usage-Cache-Write-Tokens-1h": "0",
                                },
                            )
                            usage = response.model_dump()["usage"]
                        self.assertEqual(usage["prompt_tokens"], 770)
                        self.assertEqual(usage["completion_tokens"], 2)
                        self.assertEqual(usage["total_tokens"], 772)
                        self.assertEqual(
                            usage["prompt_tokens_details"],
                            {
                                "cached_tokens": 333,
                                "cache_write_tokens": 0,
                            },
                        )

    async def test_non_streaming_response_carries_cache_usage_for_both_ttls(self):
        for ttl in (None, "5m", "1h"):
            with self.subTest(ttl=ttl):
                response = await self.chat.handle_request(
                    self.request(ttl), self.raw_request()
                )
                self.assert_cache_usage(response.model_dump()["usage"], ttl)
                self.assertEqual(response.choices[0].message.content, "hi")
                self.assertIsNotNone(response.usage.cache_usage)

    async def test_gateway_identity_reaches_parent_request_state_without_api_key(self):
        for stream in (False, True):
            with self.subTest(stream=stream):
                request = self.request("5m", stream=stream)
                request.cache_user_id = "gateway-body-user"
                raw_request = Request({"type": "http", "headers": [], "state": {}})
                response = await self.chat.handle_request(request, raw_request)
                expected = "cache-user-id:" + sha256(b"gateway-body-user").hexdigest()
                self.assertEqual(raw_request.state.cache_identity, expected)
                self.assertEqual(self.adapted_requests[-1].api_key, expected)
                if stream:
                    async for _ in response.body_iterator:
                        pass
                else:
                    self.assertEqual(response.usage.prompt_tokens, 43)

    async def test_missing_identity_serves_physical_usage_with_zero_write_headers(self):
        for stream in (False, True):
            for ttl, omitted in ((None, True), ("5m", False), ("1h", False)):
                with self.subTest(stream=stream, ttl=ttl, omitted=omitted):
                    request = self.request(ttl, stream=stream, omit_options=omitted)
                    raw = Request({"type": "http", "headers": [], "state": {}})
                    response = await self.chat.handle_request(request, raw)
                    self.assertFalse(hasattr(raw.state, "cache_identity"))
                    self.assertIsNone(self.adapted_requests[-1].api_key)
                    self.assertEqual(
                        self.adapted_requests[-1].cache_ttl_seconds,
                        3600 if ttl == "1h" else 300,
                    )
                    if stream:
                        self.assertEqual(response.status_code, 200)
                        for tier in ("5m", "1h"):
                            self.assertEqual(
                                response.headers[
                                    f"Msh-Usage-Cache-Write-Tokens-{tier}"
                                ],
                                "0",
                            )
                        frames = [frame async for frame in response.body_iterator]
                        usage = json.loads(frames[-2][6:])["usage"]
                    else:
                        self.assertEqual(
                            response.usage.cache_usage.response_headers(),
                            {
                                "Msh-Usage-Cache-Write-Tokens-5m": "0",
                                "Msh-Usage-Cache-Write-Tokens-1h": "0",
                            },
                        )
                        usage = response.model_dump()["usage"]
                    self.assertEqual(
                        usage["prompt_tokens_details"],
                        {"cached_tokens": 32, "cache_write_tokens": 0},
                    )
                    self.assertEqual(usage["prompt_tokens"], 43)
                    self.assertEqual(usage["total_tokens"], 45)

    async def test_missing_identity_parallel_sampling_keeps_inference(self):
        for stream in (False, True):
            for omitted in (False, True):
                with self.subTest(stream=stream, omitted=omitted):
                    request = self.request("5m", stream=stream, omit_options=omitted)
                    request.n = 2
                    raw = Request({"type": "http", "headers": [], "state": {}})
                    response = await self.chat.handle_request(request, raw)
                    self.assertFalse(hasattr(raw.state, "cache_identity"))
                    self.assertIsNone(self.adapted_requests[-1].api_key)
                    if stream:
                        frames = [frame async for frame in response.body_iterator]
                        usage = json.loads(frames[-2][6:])["usage"]
                    else:
                        self.assertEqual(len(response.choices), 2)
                        usage = response.model_dump()["usage"]
                    self.assertEqual(usage["prompt_tokens"], 43)
                    self.assertEqual(usage["completion_tokens"], 4)
                    self.assertEqual(usage["total_tokens"], 47)
                    self.assertEqual(
                        usage["prompt_tokens_details"],
                        {"cached_tokens": 32, "cache_write_tokens": 0},
                    )

    async def test_missing_identity_stream_headers_do_not_require_usage_chunks(self):
        request = self.request(None, stream=True, omit_options=True)
        request.stream_options = None
        raw = Request({"type": "http", "headers": [], "state": {}})
        response = await self.chat.handle_request(request, raw)
        self.assertEqual(response.status_code, 200)
        for tier in ("5m", "1h"):
            self.assertEqual(
                response.headers[f"Msh-Usage-Cache-Write-Tokens-{tier}"], "0"
            )
        frames = [frame async for frame in response.body_iterator]
        self.assertEqual(frames[-1], "data: [DONE]\n\n")
        self.assert_no_usage(json.loads(frames[0][6:]))
        for frame in frames[1:-1]:
            self.assertIsNone(json.loads(frame[6:]).get("usage"))
        self.assertFalse(hasattr(raw.state, "cache_identity"))
        self.assertIsNone(self.adapted_requests[-1].api_key)

    async def test_missing_identity_zero_output_keeps_physical_snapshot(self):
        for stream, include_usage in ((False, True), (True, True), (True, False)):
            for ttl in ("5m", "1h"):
                with self.subTest(stream=stream, include_usage=include_usage, ttl=ttl):
                    request = self.request(ttl, stream=stream)
                    request.max_completion_tokens = 0
                    request.max_tokens = 0
                    if stream:
                        request.stream_options.include_usage = include_usage
                    raw = Request({"type": "http", "headers": [], "state": {}})
                    response = await self.chat.handle_request(request, raw)
                    adapted = self.adapted_requests[-1]
                    self.assertIsNone(adapted.api_key)
                    self.assertTrue(adapted.cache_required)
                    self.assertEqual(adapted.sampling_params["max_new_tokens"], 0)
                    self.assertFalse(hasattr(raw.state, "cache_identity"))
                    expected_headers = {
                        "Msh-Usage-Cache-Write-Tokens-5m": "0",
                        "Msh-Usage-Cache-Write-Tokens-1h": "0",
                    }
                    if stream:
                        self.assertEqual(response.status_code, 200)
                        for header, value in expected_headers.items():
                            self.assertEqual(response.headers[header], value)
                        frames = [frame async for frame in response.body_iterator]
                        self.assertEqual(frames[-1], "data: [DONE]\n\n")
                        chunks = [json.loads(frame[6:]) for frame in frames[:-1]]
                        self.assert_no_usage(chunks[0])
                        self.assertEqual(
                            chunks[0]["choices"][0]["delta"]["role"], "assistant"
                        )
                        self.assertFalse(any("error" in chunk for chunk in chunks))
                        self.assertTrue(
                            any(
                                choice.get("finish_reason") == "length"
                                for chunk in chunks
                                for choice in chunk.get("choices", [])
                            )
                        )
                        usage_chunks = [
                            chunk
                            for chunk in chunks
                            if chunk.get("usage") and chunk.get("choices") == []
                        ]
                        self.assertEqual(len(usage_chunks), int(include_usage))
                        if not include_usage:
                            continue
                        self.assertEqual(usage_chunks[0]["choices"], [])
                        usage = usage_chunks[0]["usage"]
                    else:
                        self.assertEqual(response.choices[0].message.content, "")
                        self.assertEqual(
                            response.usage.cache_usage.response_headers(),
                            expected_headers,
                        )
                        usage = response.model_dump()["usage"]
                    self.assertEqual(usage["prompt_tokens"], 43)
                    self.assertEqual(usage["completion_tokens"], 0)
                    self.assertEqual(usage["total_tokens"], 43)
                    self.assertEqual(
                        usage["prompt_tokens_details"],
                        {"cached_tokens": 32, "cache_write_tokens": 0},
                    )

    async def test_omitted_options_default_to_5m_for_both_response_modes(self):
        for stream in (False, True):
            with self.subTest(stream=stream):
                request = self.request(None, stream=stream, omit_options=True)
                self.assertIsNone(request.prompt_cache_options)
                response = await self.chat.handle_request(request, self.raw_request())
                if stream:
                    frames = [frame async for frame in response.body_iterator]
                    usage = json.loads(frames[-2][6:])["usage"]
                else:
                    usage = response.model_dump()["usage"]
                self.assert_cache_usage(usage, None)

    async def test_disabled_server_with_omitted_options_keeps_legacy_response(self):
        self.config.enable_cache_accounting = False
        response = await self.chat.handle_request(
            self.request(None, omit_options=True),
            Request({"type": "http", "headers": []}),
        )
        self.assertIsNone(response.usage.cache_usage)
        self.assertIsNone(response.usage.prompt_tokens_details)
        self.assertIsNone(self.adapted_requests[-1].api_key)
        self.assertIsNone(self.adapted_requests[-1].cache_ttl_seconds)

    async def test_prefill_ignores_explicit_options_for_both_response_modes(self):
        self.disagg.disaggregation_mode = "prefill"
        self.config.enable_cache_accounting = False
        for stream in (False, True):
            with self.subTest(stream=stream):
                response = await self.chat.handle_request(
                    self.request("1h", stream=stream),
                    Request({"type": "http", "headers": []}),
                )
                if stream:
                    frames = [frame async for frame in response.body_iterator]
                    usage = json.loads(frames[-2][6:])["usage"]
                else:
                    usage = response.model_dump()["usage"]
                self.assertIsNone(usage.get("prompt_tokens_details"))
                self.assertIsNone(self.adapted_requests[-1].api_key)
                self.assertIsNone(self.adapted_requests[-1].cache_ttl_seconds)

    async def test_debug_meta_info_does_not_expose_internal_cache_snapshot(self):
        request = self.request("5m")
        request.return_meta_info = True
        response = await self.chat.handle_request(request, self.raw_request())
        meta_info = response.model_dump()["choices"][0]["meta_info"]
        self.assertNotIn("cache_usage", meta_info)
        self.assertEqual(meta_info["cached_tokens"], 32)
        self.assert_cache_usage(response.model_dump()["usage"], "5m")

    async def test_streaming_include_usage_finishes_with_cache_snapshot(self):
        for ttl, continuous in (
            (None, False),
            ("5m", False),
            ("1h", False),
            ("5m", True),
            ("1h", True),
            (None, True),
        ):
            with self.subTest(ttl=ttl, continuous=continuous):
                request = self.request(ttl, stream=True)
                request.stream_options.continuous_usage_stats = continuous
                response = await self.chat.handle_request(request, self.raw_request())
                self.assertIsInstance(response, StreamingResponse)
                self.assertEqual(
                    response.headers["Msh-Usage-Cache-Write-Tokens-5m"],
                    "0" if ttl == "1h" else "16",
                )
                self.assertEqual(
                    response.headers["Msh-Usage-Cache-Write-Tokens-1h"],
                    "16" if ttl == "1h" else "0",
                )
                frames = [frame async for frame in response.body_iterator]
                self.assertEqual(frames[-1], "data: [DONE]\n\n")
                final = json.loads(frames[-2][6:])
                self.assert_no_usage(json.loads(frames[0][6:]))
                self.assertEqual(final["choices"], [])
                self.assert_cache_usage(final["usage"], ttl)
                self.assertEqual(
                    sum(
                        int(response.headers[f"Msh-Usage-Cache-Write-Tokens-{tier}"])
                        for tier in ("5m", "1h")
                    ),
                    final["usage"]["prompt_tokens_details"]["cache_write_tokens"],
                )
                if continuous:
                    usages = [
                        chunk["usage"]
                        for frame in frames[:-1]
                        if (chunk := json.loads(frame[6:])).get("usage") is not None
                    ]
                    self.assertGreater(len(usages), 1)
                    for usage in usages:
                        self.assertNotIn("cache_usage", usage)
                        self.assertEqual(
                            usage["prompt_tokens_details"],
                            {"cached_tokens": 16, "cache_write_tokens": 16},
                        )

    async def test_stream_headers_do_not_require_usage_chunks(self):
        for ttl in ("5m", "1h"):
            with self.subTest(ttl=ttl):
                request = self.request(ttl, stream=True)
                request.stream_options = None
                response = await self.chat.handle_request(request, self.raw_request())
                # The headers must exist before consuming any response body.
                self.assertEqual(
                    response.headers["Msh-Usage-Cache-Write-Tokens-5m"],
                    "16" if ttl == "5m" else "0",
                )
                self.assertEqual(
                    response.headers["Msh-Usage-Cache-Write-Tokens-1h"],
                    "16" if ttl == "1h" else "0",
                )
                frames = [frame async for frame in response.body_iterator]
                self.assert_no_usage(json.loads(frames[0][6:]))
                for frame in frames[1:-1]:
                    self.assertIsNone(json.loads(frame[6:]).get("usage"))

    async def test_physical_fallback_stream_has_zero_write_headers_and_usage(self):
        self.snapshot_override = {
            "ttl_seconds": 3600,
            "read_tokens": 32,
            "write_tokens": 0,
        }
        for include_usage in (False, True):
            with self.subTest(include_usage=include_usage):
                request = self.request("1h", stream=True)
                request.stream_options.include_usage = include_usage
                response = await self.chat.handle_request(request, self.raw_request())
                self.assertEqual(
                    response.headers["Msh-Usage-Cache-Write-Tokens-5m"], "0"
                )
                self.assertEqual(
                    response.headers["Msh-Usage-Cache-Write-Tokens-1h"], "0"
                )
                frames = [frame async for frame in response.body_iterator]
                self.assert_no_usage(json.loads(frames[0][6:]))
                if include_usage:
                    usage = json.loads(frames[-2][6:])["usage"]
                    self.assertEqual(
                        usage["prompt_tokens_details"],
                        {"cached_tokens": 32, "cache_write_tokens": 0},
                    )
                    self.assertEqual(usage["prompt_tokens"], 43)
                    self.assertNotIn("cache_usage", usage)
                else:
                    for frame in frames[1:-1]:
                        self.assertIsNone(json.loads(frame[6:]).get("usage"))

    async def test_stream_headers_use_a_separate_snapshot_for_each_request(self):
        responses = [
            await self.chat.handle_request(
                self.request(ttl, stream=True), self.raw_request()
            )
            for ttl in ("5m", "1h")
        ]
        self.assertEqual(responses[0].headers["Msh-Usage-Cache-Write-Tokens-5m"], "16")
        self.assertEqual(responses[0].headers["Msh-Usage-Cache-Write-Tokens-1h"], "0")
        self.assertEqual(responses[1].headers["Msh-Usage-Cache-Write-Tokens-5m"], "0")
        self.assertEqual(responses[1].headers["Msh-Usage-Cache-Write-Tokens-1h"], "16")
        for response in responses:
            _ = [frame async for frame in response.body_iterator]

    async def test_streaming_thinking_counts_reach_final_usage(
        self,
    ):
        for include_usage in (False, True):
            for continuous in (False, True):
                with self.subTest(include_usage=include_usage, continuous=continuous):
                    self.set_stream_outputs(
                        [
                            self.stream_output("hel", 3, reasoning=2),
                            self.stream_output(
                                "hello", 5, reasoning=4, finish={"type": "stop"}
                            ),
                        ]
                    )
                    request = self.request("5m", stream=True)
                    request.max_completion_tokens = 5
                    request.stream_options.include_usage = include_usage
                    request.stream_options.continuous_usage_stats = continuous
                    response = await self.chat.handle_request(
                        request, self.raw_request()
                    )
                    frames = [frame async for frame in response.body_iterator]
                    chunks = [json.loads(frame[6:]) for frame in frames[:-1]]
                    self.assert_no_usage(chunks[0])
                    self.assertEqual(
                        response.headers["Msh-Usage-Cache-Write-Tokens-5m"], "16"
                    )
                    tails = [
                        chunk
                        for chunk in chunks
                        if chunk.get("choices") == [] and chunk.get("usage")
                    ]
                    self.assertEqual(len(tails), int(include_usage))
                    if include_usage:
                        final = tails[0]["usage"]
                        self.assertEqual(final["prompt_tokens"], 43)
                        self.assertEqual(final["completion_tokens"], 5)
                        self.assertEqual(final["total_tokens"], 48)
                        self.assertEqual(
                            final["completion_tokens_details"]["reasoning_tokens"], 4
                        )
                        self.assertEqual(
                            final["prompt_tokens_details"],
                            {"cached_tokens": 16, "cache_write_tokens": 16},
                        )

    async def test_default_short_prompt_has_no_usage_without_tail(self):
        self.chat._process_messages.return_value = MessageProcessingResult(
            prompt="short",
            prompt_ids=list(range(11)),
            image_data=None,
            audio_data=None,
            video_data=None,
            modalities=[],
            stop=[],
        )
        self.set_stream_outputs(
            [
                self.stream_output(
                    "hi",
                    2,
                    prompt=11,
                    snapshot={"ttl_seconds": 300, "read_tokens": 0, "write_tokens": 0},
                    finish={"type": "stop"},
                )
            ]
        )
        request = self.request(None, stream=True, omit_options=True)
        request.stream_options = None
        response = await self.chat.handle_request(request, self.raw_request())
        frames = [frame async for frame in response.body_iterator]
        chunks = [json.loads(frame[6:]) for frame in frames[:-1]]
        self.assert_no_usage(chunks[0])
        self.assertEqual([chunk for chunk in chunks if chunk.get("usage")], [])
        for tier in ("5m", "1h"):
            self.assertEqual(
                response.headers[f"Msh-Usage-Cache-Write-Tokens-{tier}"], "0"
            )
        self.assertEqual(frames[-1], "data: [DONE]\n\n")

    async def test_parallel_first_system_abort_has_no_role_usage_or_cache_headers(
        self,
    ):
        snapshot = {"ttl_seconds": 300, "read_tokens": 9, "write_tokens": 0}
        self.set_stream_outputs(
            [
                self.stream_output(
                    "b",
                    1,
                    index=1,
                    snapshot=snapshot,
                    finish={
                        "type": "abort",
                        "status_code": HTTPStatus.SERVICE_UNAVAILABLE,
                        "message": "Generation unavailable",
                    },
                ),
            ]
        )
        request = self.request("5m", stream=True)
        request.n = 2
        response = await self.chat.handle_request(
            request, Request({"type": "http", "headers": [], "state": {}})
        )
        self.assertNotIn("Msh-Usage-Cache-Write-Tokens-5m", response.headers)
        self.assertNotIn("Msh-Usage-Cache-Write-Tokens-1h", response.headers)
        frames = [frame async for frame in response.body_iterator]
        chunks = [json.loads(frame[6:]) for frame in frames[:-1]]
        self.assertIn("error", chunks[0])
        self.assertFalse(
            any(
                choice.get("delta", {}).get("role")
                for chunk in chunks
                for choice in chunk.get("choices", [])
            )
        )

    async def test_parallel_stream_without_snapshot_keeps_legacy_arrival_order(self):
        self.config.enable_cache_accounting = False
        self.set_stream_outputs(
            [
                self.stream_output(
                    "b", 1, index=1, snapshot=None, finish={"type": "stop"}
                ),
                self.stream_output(
                    "a", 1, index=0, snapshot=None, finish={"type": "stop"}
                ),
            ]
        )
        request = self.request(None, stream=True, omit_options=True)
        request.n = 2
        response = await self.chat.handle_request(
            request, Request({"type": "http", "headers": [], "state": {}})
        )
        frames = [frame async for frame in response.body_iterator]
        first = json.loads(frames[0][6:])
        self.assertEqual(first["choices"][0]["index"], 1)
        for frame in frames[:-1]:
            chunk = json.loads(frame[6:])
            if any(
                choice.get("delta", {}).get("role") == "assistant"
                for choice in chunk.get("choices", [])
            ):
                self.assert_no_usage(chunk)
        self.assertNotIn("Msh-Usage-Cache-Write-Tokens-5m", response.headers)

    async def test_disabled_stream_keeps_cache_headers_absent(self):
        self.config.enable_cache_accounting = False
        response = await self.chat.handle_request(
            self.request(None, stream=True, omit_options=True),
            Request({"type": "http", "headers": []}),
        )
        self.assertNotIn("Msh-Usage-Cache-Write-Tokens-5m", response.headers)
        self.assertNotIn("Msh-Usage-Cache-Write-Tokens-1h", response.headers)
        frames = [frame async for frame in response.body_iterator]
        self.assertIsNone(json.loads(frames[0][6:]).get("usage"))

    async def test_first_error_event_does_not_expose_cache_write_headers(self):
        self.abort_first = True
        response = await self.chat.handle_request(
            self.request("1h", stream=True), self.raw_request()
        )
        self.assertNotIn("Msh-Usage-Cache-Write-Tokens-5m", response.headers)
        self.assertNotIn("Msh-Usage-Cache-Write-Tokens-1h", response.headers)
        frames = [frame async for frame in response.body_iterator]
        self.assertIn("error", json.loads(frames[0][6:]))

    async def test_backend_failure_returns_http_503_for_both_response_modes(self):
        self.fail_accounting = True
        for stream in (False, True):
            with self.subTest(stream=stream):
                response = await self.chat.handle_request(
                    self.request("5m", stream=stream), self.raw_request()
                )
                self.assertEqual(response.status_code, 503)
                self.assertEqual(json.loads(response.body)["code"], 503)


if __name__ == "__main__":
    unittest.main()
