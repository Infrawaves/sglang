"""Decode hot migration on /v1/chat/completions: the hand-off event a retiring
decode sends instead of a finish, and the resumed request that continues it."""

from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()  # must precede any import that pulls in sgl_kernel

import asyncio
import json
import unittest
from typing import Optional
from unittest.mock import Mock, patch

from fastapi import Request
from fastapi.responses import StreamingResponse

from sglang.srt.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    MessageProcessingResult,
)
from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
from sglang.srt.function_call.kimik3_format import (
    THINK_CLOSE,
    TOOLS_CLOSE,
    TOOLS_OPEN,
)
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

ORIGIN = [1, 2, 3]
_WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
    },
}


def _byte_decode(ids, **_kwargs):
    """A byte-level tokenizer: every token is one UTF-8 byte."""
    return bytes(ids).decode("utf-8", errors="replace")


class _MockTokenizerManager:
    def __init__(self, *, tool_call_parser=None, reasoning_parser=None):
        self.model_config = Mock(is_multimodal=False)
        self.model_config.get_default_sampling_params.return_value = {}
        hf_config = Mock()
        hf_config.architectures = ["LlamaForCausalLM"]
        hf_config.to_dict.return_value = {}
        self.model_config.hf_config = hf_config
        self.server_args = Mock(
            model_path="dummy",
            revision=None,
            enable_cache_report=False,
            tool_call_parser=tool_call_parser,
            reasoning_parser=reasoning_parser,
            stream_response_default_include_usage=False,
            default_chat_template_kwargs=None,
            grammar_backend="xgrammar",
        )
        self.model_path = "dummy"
        self.served_model_name = "x"
        self.tokenizer = Mock()
        self.tokenizer.encode.return_value = [1, 2, 3, 4, 5]
        self.tokenizer.decode.side_effect = _byte_decode
        self.tokenizer.apply_chat_template.return_value = list(ORIGIN)
        self.tokenizer.chat_template = None
        self.tokenizer.bos_token_id = 1
        self.create_abort_task = Mock()
        self.request_logger = Mock(log_requests=False, log_requests_level=0)
        self.process_request_preprocessor = None
        self.generate_request = None

    async def run_in_request_preprocessor(self, func, *args, **kwargs):
        return func(*args, **kwargs)

    def config_value(self, name: str):
        return getattr(self.server_args, name)


class _MockTemplateManager:
    def __init__(self):
        self.chat_template_name: Optional[str] = None
        self.jinja_template_content_format: Optional[str] = "string"
        self.completion_template_name: Optional[str] = None
        self.reasoning_config = None
        self.force_reasoning = False
        self.jinja_template_may_reorder_tool_results = False
        self.python_chat_template = None


def _engine_events(data: bytes, *, finish, incremental: bool):
    """Engine outputs for byte tokens `data`, a few tokens per output: text
    stops before an unfinished character until the final output."""
    events, sent = [], ""
    ends = list(range(3, len(data), 3)) + [len(data)]
    for i, end in enumerate(ends):
        last = i == len(ends) - 1
        text = data[:end].decode("utf-8", errors="replace")
        if not last:
            text = text.rstrip("\ufffd")
        events.append(
            {
                "text": text[len(sent) :] if incremental else text,
                "output_ids": [],
                "index": 0,
                "meta_info": {
                    "id": "rid",
                    "prompt_tokens": 10,
                    "completion_tokens": end,
                    "reasoning_tokens": 0,
                    "cached_tokens": 0,
                    "finish_reason": finish if last else None,
                },
            }
        )
        sent = text
    return events


def _migrated(output_ids, origin=ORIGIN):
    return {
        "type": "migrated",
        "origin_input_ids": list(origin),
        "output_ids": list(output_ids),
        "kv_flushed_tokens": 4,
    }


def _frames(chunks):
    return [
        json.loads(chunk[len("data: ") :])
        for chunk in chunks
        if chunk.startswith("data: ") and chunk.strip() != "data: [DONE]"
    ]


def _deltas(frames):
    """What a client assembles from a stream, the role frame aside."""
    reasoning, content, finish = "", "", None
    names, arguments = {}, {}
    for frame in frames:
        for choice in frame.get("choices", []):
            delta = choice.get("delta") or {}
            reasoning += delta.get("reasoning_content") or ""
            content += delta.get("content") or ""
            for call in delta.get("tool_calls") or []:
                function = call.get("function") or {}
                # Ids are random per call, so only the name is compared.
                if function.get("name"):
                    names[call["index"]] = function["name"]
                arguments[call["index"]] = arguments.get(call["index"], "") + (
                    function.get("arguments") or ""
                )
            finish = choice.get("finish_reason") or finish
    return {
        "reasoning": reasoning,
        "content": content,
        "names": names,
        "arguments": arguments,
        "finish": finish,
    }


class _MigrationChatCase(CustomTestCase):
    INCREMENTAL = False
    TOOL_CALL_PARSER = None
    REASONING_PARSER = None

    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)
        publish(
            ServerArgs(
                model_path="dummy",
                incremental_streaming_output=self.INCREMENTAL,
                default_chat_template_kwargs=None,
                tool_call_parser=self.TOOL_CALL_PARSER,
                reasoning_parser=self.REASONING_PARSER,
            ),
            role="tokenizer",
        )
        self.tm = _MockTokenizerManager(
            tool_call_parser=self.TOOL_CALL_PARSER,
            reasoning_parser=self.REASONING_PARSER,
        )
        self.chat = OpenAIServingChat(self.tm, _MockTemplateManager())
        self.raw_request = Request({"type": "http", "headers": []})
        self.internal_requests = []

    def _request(self, **kwargs):
        kwargs.setdefault("model", "x")
        kwargs.setdefault("messages", [{"role": "user", "content": "Hi?"}])
        kwargs.setdefault("stream", True)
        kwargs.setdefault("allow_migration", True)
        return ChatCompletionRequest(**kwargs)

    def _serve(self, request, events):
        """Serve `request` end to end with the engine replaced by `events`."""

        async def generate(internal_request, raw_request):
            self.internal_requests.append(internal_request)
            if request.stream:
                for event in events:
                    yield event
            else:
                yield events[-1]

        async def serve():
            self.tm.generate_request = generate
            response = await self.chat.handle_request(request, self.raw_request)
            if isinstance(response, StreamingResponse):
                return [chunk async for chunk in response.body_iterator]
            return response

        return asyncio.run(serve())

    def _segment(self, request, data: bytes, finish):
        return self._serve(
            request, _engine_events(data, finish=finish, incremental=self.INCREMENTAL)
        )

    def _assert_split_streams_like_one(self, text: str, cut: int, **request_kwargs):
        """Handing off after `cut` bytes and resuming elsewhere must give the
        client exactly what one uninterrupted stream gives."""
        data = text.encode("utf-8")
        baseline = _deltas(
            _frames(
                self._segment(
                    self._request(**request_kwargs), data, {"type": "stop"}
                )
            )
        )

        first = _frames(
            self._segment(
                self._request(**request_kwargs), data[:cut], _migrated(data[:cut])
            )
        )
        handoff = first[-1]["sglang_migration"]
        handed_off = bytes(handoff["output_ids"])
        self.assertTrue(data.startswith(handed_off))
        resumed = self._request(
            migration_resume={
                "origin_input_ids": handoff["origin_input_ids"],
                "output_ids": handoff["output_ids"],
                "attempt": 1,
            },
            **request_kwargs,
        )
        second = _frames(
            self._segment(resumed, data[len(handed_off) :], {"type": "stop"})
        )
        self.assertEqual(
            self.internal_requests[-1].input_ids, ORIGIN + list(handed_off)
        )

        parts = [_deltas(first[:-1]), _deltas(second)]
        self.assertEqual(
            parts[0]["reasoning"] + parts[1]["reasoning"], baseline["reasoning"]
        )
        self.assertEqual(parts[0]["content"] + parts[1]["content"], baseline["content"])
        self.assertEqual({**parts[0]["names"], **parts[1]["names"]}, baseline["names"])
        merged_arguments = {
            index: parts[0]["arguments"].get(index, "")
            + parts[1]["arguments"].get(index, "")
            for index in baseline["arguments"]
        }
        self.assertEqual(merged_arguments, baseline["arguments"])
        self.assertIsNone(parts[0]["finish"])
        self.assertEqual(parts[1]["finish"], baseline["finish"])


class TestChatHandOff(_MigrationChatCase):
    def test_stream_ends_with_one_hand_off_event_instead_of_a_finish(self):
        """The router recognizes the hand-off by this event alone; a finish
        chunk or usage before it would reach the client as the end."""
        request = self._request(stream_options={"include_usage": True})
        chunks = self._segment(request, b"Hello", _migrated(b"Hello"))

        self.assertEqual(chunks[-1], "data: [DONE]\n\n")
        self.assertEqual(sum("[DONE]" in chunk for chunk in chunks), 1)
        frames = _frames(chunks)
        event = frames[-1]
        self.assertEqual(event["object"], "chat.completion.chunk")
        self.assertEqual(event["choices"], [])
        self.assertEqual(
            event["sglang_migration"],
            {
                "index": 0,
                "origin_input_ids": ORIGIN,
                "output_ids": list(b"Hello"),
                "kv_flushed_tokens": 4,
                "reasoning_tokens": 0,
            },
        )
        self.assertEqual(sum("sglang_migration" in frame for frame in frames), 1)
        self.assertFalse(any(frame.get("usage") for frame in frames))
        self.assertEqual(_deltas(frames[:-1])["content"], "Hello")
        self.assertIsNone(_deltas(frames)["finish"])

    def test_non_stream_returns_the_hand_off_instead_of_a_message(self):
        """A "migrated" finish_reason fails the response model's validation."""
        response = self._segment(
            self._request(stream=False), b"Hello", _migrated(b"Hello")
        )
        body = json.loads(response.body)
        self.assertEqual(body["object"], "chat.completion")
        self.assertEqual(body["choices"], [])
        self.assertEqual(body["sglang_migration"]["output_ids"], list(b"Hello"))
        self.assertEqual(body["sglang_migration"]["origin_input_ids"], ORIGIN)

    def test_resumed_request_hands_off_the_original_split(self):
        """The engine prompt of a resumed request already holds the earlier
        output; the router must get the original prompt and all output."""
        request = self._request(
            stream=False,
            migration_resume={"origin_input_ids": ORIGIN, "output_ids": list(b"He")},
        )
        response = self._segment(
            request, b"llo", _migrated(b"llo", origin=ORIGIN + list(b"He"))
        )
        migration = json.loads(response.body)["sglang_migration"]
        self.assertEqual(migration["origin_input_ids"], ORIGIN)
        self.assertEqual(migration["output_ids"], list(b"Hello"))

    def test_hand_off_inside_a_character_hands_off_only_whole_characters(self):
        """The client never saw the unfinished character; the next decode must
        generate it whole rather than continue after its first bytes."""
        data = "ab中".encode("utf-8")[:-1]
        frames = _frames(self._segment(self._request(), data, _migrated(data)))
        self.assertEqual(frames[-1]["sglang_migration"]["output_ids"], list(b"ab"))
        self.assertEqual(_deltas(frames[:-1])["content"], "ab")

    def test_split_plain_stream_matches_one_stream(self):
        text = "Hello, 世界! Bye."
        for cut in (1, 5, 9, 10, len(text.encode()) - 1):
            with self.subTest(cut=cut):
                self._assert_split_streams_like_one(text, cut)

    def test_resumed_usage_counts_the_original_prompt_and_this_segment(self):
        """The router adds the handed-off output to completion_tokens; counting
        it again here, or as prompt, would bill it twice."""
        request = self._request(
            stream_options={"include_usage": True},
            migration_resume={"origin_input_ids": ORIGIN, "output_ids": list(b"He")},
        )
        events = _engine_events(
            b"llo", finish={"type": "stop"}, incremental=self.INCREMENTAL
        )
        events[-1]["meta_info"].update(prompt_tokens=5, cached_tokens=5)
        usage = [f for f in _frames(self._serve(request, events)) if f.get("usage")]
        self.assertEqual(usage[-1]["usage"]["prompt_tokens"], len(ORIGIN))
        self.assertEqual(usage[-1]["usage"]["completion_tokens"], 3)

    def test_hand_off_reports_reasoning_tokens_of_every_segment(self):
        """The router cannot tell how many earlier tokens were reasoning once
        a segment reports none, so the hand-off carries the running total."""
        request = self._request(
            migration_resume={
                "origin_input_ids": ORIGIN,
                "output_ids": list(b"He"),
                "reasoning_tokens": 2,
            },
        )
        events = _engine_events(
            b"llo",
            finish=_migrated(b"llo", origin=ORIGIN + list(b"He")),
            incremental=self.INCREMENTAL,
        )
        events[-1]["meta_info"]["reasoning_tokens"] = 1
        frames = _frames(self._serve(request, events))
        self.assertEqual(frames[-1]["sglang_migration"]["reasoning_tokens"], 3)

    def test_resumed_stream_sends_no_second_role_frame(self):
        """The client already got the assistant role from the first segment."""
        request = self._request(
            migration_resume={"origin_input_ids": ORIGIN, "output_ids": list(b"He")},
        )
        frames = _frames(self._segment(request, b"llo", {"type": "stop"}))
        roles = [
            choice["delta"].get("role")
            for frame in frames
            for choice in frame.get("choices", [])
            if choice.get("delta", {}).get("role")
        ]
        self.assertEqual(roles, [])


class TestChatHandOffIncremental(TestChatHandOff):
    INCREMENTAL = True


class TestResumedRequestConversion(_MigrationChatCase):
    def _convert(self, request, *, require_reasoning=False, end_ids=None):
        processed = MessageProcessingResult(
            prompt="",
            prompt_ids=list(ORIGIN),
            image_data=None,
            audio_data=None,
            video_data=None,
            modalities=[],
            stop=[],
            require_reasoning=require_reasoning,
            reasoning_end_token_ids=end_ids,
        )
        with patch.object(self.chat, "_process_messages", return_value=processed):
            adapted, _ = self.chat._convert_to_internal_request(request)
        return adapted

    def test_prompt_is_origin_plus_output(self):
        """Prefill and decode both build this from the same body; any other
        prompt misses the KV the retiring decode stored."""
        resume = {"origin_input_ids": [7, 8], "output_ids": [9]}
        adapted = self._convert(self._request(migration_resume=resume))
        self.assertEqual(adapted.input_ids, [7, 8, 9])
        self.assertIsNone(adapted.text)

    def test_allow_migration_only_for_a_single_choice(self):
        """A hand-off event carries one choice; n > 1 would lose the others."""
        for kwargs, expected in (
            ({}, True),
            ({"n": 2}, False),
            ({"allow_migration": False}, False),
        ):
            with self.subTest(**kwargs):
                adapted = self._convert(self._request(stream=False, **kwargs))
                self.assertIs(adapted.allow_migration, expected)

    def test_resume_past_the_reasoning_does_not_wait_for_it_again(self):
        """The engine ignores stops until it sees the reasoning terminator in
        the output; a terminator already in the handed-off output is not."""
        end_ids = [124]
        for output_ids, expected in (([1, 124, 2], False), ([1, 2], True)):
            with self.subTest(output_ids=output_ids):
                resume = {"origin_input_ids": ORIGIN, "output_ids": output_ids}
                adapted = self._convert(
                    self._request(migration_resume=resume),
                    require_reasoning=True,
                    end_ids=end_ids,
                )
                self.assertIs(adapted.require_reasoning, expected)

    def test_resume_with_several_choices_is_rejected(self):
        resume = {"origin_input_ids": ORIGIN, "output_ids": [4]}
        with self.assertRaises(ValueError):
            self._convert(self._request(n=2, migration_resume=resume))


class TestKimiK3HandOff(_MigrationChatCase):
    """Replay through the kimi_k3 reasoning and tool-call parsers: whatever
    they held back at the hand-off must come out once, after the resume."""

    TOOL_CALL_PARSER = "kimi_k3"
    REASONING_PARSER = "kimi_k3"

    _CALL = (
        TOOLS_OPEN
        + '<|open|>call tool="get_weather" index="1"<|sep|>'
        + '<|open|>argument key="city" type="string"<|sep|>'
        + "Paris 中心"
        + "<|close|>argument<|sep|>"
        + "<|close|>call<|sep|>"
        + TOOLS_CLOSE
    )

    def _cuts(self, text, *anchors):
        data = text.encode("utf-8")
        cuts = {1, len(data) - 1}
        for anchor, offset in anchors:
            cuts.add(data.index(anchor.encode("utf-8")) + offset)
        return sorted(cut for cut in cuts if 0 < cut < len(data))

    def test_hand_off_inside_reasoning_and_its_close_marker(self):
        text = "Let me think about it." + THINK_CLOSE + "The answer is 42."
        for cut in self._cuts(text, ("think about", 3), (THINK_CLOSE, 4)):
            with self.subTest(cut=cut):
                self._assert_split_streams_like_one(text, cut)

    def test_hand_off_inside_a_tool_call_and_a_character(self):
        text = "Checking." + THINK_CLOSE + self._CALL
        for cut in self._cuts(
            text,
            (TOOLS_OPEN, 3),
            ('tool="get_weather"', 5),
            ("Paris", 2),
            ("中", 1),
            ("<|close|>call", 2),
        ):
            with self.subTest(cut=cut):
                self._assert_split_streams_like_one(text, cut, tools=[_WEATHER_TOOL])


class TestKimiK3HandOffIncremental(TestKimiK3HandOff):
    INCREMENTAL = True


if __name__ == "__main__":
    unittest.main(verbosity=2)
