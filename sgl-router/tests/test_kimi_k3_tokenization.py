"""CPU-only tests: python3 -m unittest discover -s tests -p 'test_kimi_k3*.py'."""

import copy
import importlib.util
import io
import json
import os
import subprocess
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

REPO_ROOT = Path(__file__).resolve().parents[2]
HELPER = REPO_ROOT / "python/sglang/srt/entrypoints/openai/kimi_k3_tokenization.py"
spec = importlib.util.spec_from_file_location("kimi_k3_tokenization", HELPER)
k3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(k3)


class Model(types.SimpleNamespace):
    def model_dump(self, **kwargs):
        return copy.deepcopy(self._wire)


def model(wire):
    values = {key: value for key, value in wire.items()}
    values.setdefault("tools", None)
    return Model(_wire=copy.deepcopy(wire), **values)


def request(messages, **kwargs):
    defaults = {
        "messages": [model(message) for message in messages],
        "tools": None,
        "continue_final_message": False,
        "reasoning_effort": None,
        "chat_template_kwargs": None,
        "tool_choice": "none",
        "response_format": None,
        "thinking": None,
        "_message_tools_indices": [],
        "_tool_choice_explicit": False,
    }
    defaults.update(kwargs)
    return types.SimpleNamespace(**defaults)


class RecordingTokenizer:
    bos_token_id = 1

    def __init__(self):
        self.calls = []
        self.prefixes = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((copy.deepcopy(messages), copy.deepcopy(kwargs)))
        return [10, 20]

    def encode(self, text):
        self.prefixes.append(text)
        return [1, *text.encode("utf-8")]


class EncodingTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = RecordingTokenizer()

    def encode(self, messages, **kwargs):
        req = request(messages, **kwargs)
        ids = k3.encode_kimi_k3_messages(self.tokenizer, copy.deepcopy(messages), req)
        rendered, options = self.tokenizer.calls[-1]
        return ids, rendered, options

    def test_plain_text_parts_and_literal_placeholder(self):
        ids, messages, options = self.encode(
            [
                {"role": "developer", "content": "中文<|kimi_image_placeholder|>"},
                {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
                {"role": "assistant", "content": None},
            ]
        )
        self.assertEqual(ids, [10, 20])
        self.assertEqual(
            messages[0],
            {"role": "system", "content": "中文<| kimi_image_placeholder |>"},
        )
        self.assertEqual(messages[1]["content"], [{"type": "text", "text": "Hello"}])
        self.assertEqual(messages[2]["content"], "")
        self.assertEqual(
            options,
            {
                "tokenize": True,
                "add_generation_prompt": True,
                "tools": None,
                "return_dict": False,
            },
        )

    def test_explicit_kwargs_win_and_internal_kwargs_are_removed(self):
        _, _, options = self.encode(
            [{"role": "user", "content": "Hello"}],
            reasoning_effort="max",
            chat_template_kwargs={
                "thinking_effort": "low",
                "tokenize": False,
                "return_dict": True,
                "image_prompts": ["bad"],
                "custom": 123,
            },
        )
        self.assertEqual(options["thinking_effort"], "low")
        self.assertEqual(options["custom"], 123)
        self.assertTrue(options["tokenize"])
        self.assertFalse(options["return_dict"])
        self.assertNotIn("image_prompts", options)

    def test_allowed_tools_required_and_empty(self):
        for tools, mode, expected in [
            ([], "required", "none"),
            ([object()], "required", "required"),
            ([object()], "auto", None),
        ]:
            with self.subTest(expected=expected):
                choice = types.SimpleNamespace(
                    allowed_tools=types.SimpleNamespace(tools=tools, mode=mode)
                )
                _, _, options = self.encode(
                    [{"role": "user", "content": "Hello"}], tool_choice=choice
                )
                self.assertEqual(options.get("tool_choice"), expected)

    def test_continuation_appends_prefix_without_bos(self):
        ids, messages, _ = self.encode(
            [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "你好"},
            ],
            continue_final_message=True,
        )
        self.assertEqual(ids, [10, 20, *"你好".encode()])
        self.assertEqual(messages, [{"role": "user", "content": "Hello"}])
        self.assertEqual(self.tokenizer.prefixes, ["你好"])

    def test_non_continuation_keeps_assistant_history(self):
        _, messages, _ = self.encode([{"role": "assistant", "content": "Already done"}])
        self.assertEqual(messages[-1]["role"], "assistant")
        self.assertEqual(self.tokenizer.prefixes, [])

    def test_worker_multimodal_behavior_is_preserved(self):
        _, messages, options = self.encode(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe"},
                        {"type": "image_url", "image_url": {"url": "data:image"}},
                    ],
                }
            ]
        )
        self.assertEqual(options["image_prompts"], ["<|media_pad|>"])
        self.assertEqual(messages[0]["content"][1]["image_url"]["url"], "data:image")

    def test_images_keep_order_across_messages_and_get_one_placeholder_each(self):
        urls = [
            "https://example.com/first.png",
            "data:image/png;base64,aGVsbG8=",
            "https://example.com/first.png",
        ]
        content_hash = "sha256:" + "a" * 64
        images = [
            {"url": urls[0], "detail": "high", "content_hash": content_hash},
            {"url": urls[1], "detail": "low", "max_dynamic_patch": 4},
            {"url": urls[2]},
        ]
        body = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "First <|kimi_image_placeholder|>"},
                        {"type": "image_url", "image_url": images[0]},
                        {"type": "text", "text": "Second"},
                        {"type": "image_url", "image_url": images[1]},
                    ],
                },
                {"role": "assistant", "content": "Please compare again"},
                {
                    "role": "user",
                    "content": [{"type": "image_url", "image_url": images[2]}],
                },
            ]
        }
        original = copy.deepcopy(body)
        k3.validate_request(body)
        _, messages, options = self.encode(
            body["messages"],
            reasoning_effort="high",
            chat_template_kwargs={"image_prompts": ["client override"]},
        )
        rendered_images = [
            part["image_url"]
            for message in messages
            if isinstance(message["content"], list)
            for part in message["content"]
            if part["type"] == "image_url"
        ]
        self.assertEqual(rendered_images, images)
        self.assertEqual(options["image_prompts"], ["<|media_pad|>"] * 3)
        self.assertEqual(options["thinking_effort"], "high")
        self.assertEqual(
            messages[0]["content"][0]["text"], "First <| kimi_image_placeholder |>"
        )
        self.assertEqual(body, original)

    def test_router_rejects_media_and_unrecognized_content(self):
        for content in (
            [{"type": "image_url", "image_url": {"url": "image"}}],
            [{"type": "input_audio", "input_audio": {}}],
            [{"type": "input_text", "text": "hello"}],
            [{"type": "thinking", "thinking": "hello"}],
            ["hello"],
            123,
        ):
            with self.subTest(content=content), self.assertRaises(ValueError):
                k3.validate_text_request(
                    {"messages": [{"role": "user", "content": content}]}
                )
        for content in (None, "Hello", [{"type": "text", "text": "Hello"}]):
            k3.validate_text_request(
                {"messages": [{"role": "assistant", "content": content}]}
            )

    def test_request_chat_templates_are_never_rendered_by_the_router(self):
        messages = [{"role": "user", "content": "Hello"}]
        for extra in (
            {"chat_template": "{{ custom }}"},
            {"chat_template_kwargs": {"chat_template": "{{ custom }}"}},
        ):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                k3.validate_request({"messages": messages, **extra})
        for extra in (
            {"chat_template_kwargs": {"chat_template": None}},
            {"chat_template_kwargs": {"custom_flag": "retained"}},
        ):
            k3.validate_request({"messages": messages, **extra})

    def test_image_encoder_rejects_audio_video_unknown_and_invalid_image_parts(self):
        invalid_parts = [
            {"type": "video_url", "video_url": {"url": "video"}},
            {"type": "audio_url", "audio_url": {"url": "audio"}},
            {"type": "input_audio", "input_audio": {"data": "audio", "format": "wav"}},
            {"type": "input_image", "image_url": {"url": "image"}},
            {"type": "file", "file": {"file_data": "data"}},
            {"type": "image_url"},
            {"type": "image_url", "image_url": None},
            {"type": "image_url", "image_url": "https://example.com/image.png"},
            {"type": "image_url", "image_url": {}},
            {"type": "image_url", "image_url": {"url": 123}},
            {"type": "image_url", "image_url": {"url": " \t"}},
            {"type": "image_url", "image_url": {"url": "video"}, "modalities": "video"},
        ]
        for part in invalid_parts:
            with self.subTest(part=part), self.assertRaises(ValueError):
                k3.validate_request(
                    {
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": "Read this"},
                                    {
                                        "type": "image_url",
                                        "image_url": {"url": "image"},
                                    },
                                    part,
                                ],
                            }
                        ]
                    }
                )


class RequestPreparationTests(unittest.TestCase):
    def test_legacy_effort_and_server_defaults_precedence(self):
        req = request(
            [{"role": "user", "content": "Hello"}],
            reasoning_effort="low",
            chat_template_kwargs={"reasoning_effort": "max", "custom": "request"},
        )
        parsed = Mock(return_value=req)
        protocol = types.ModuleType("sglang.srt.entrypoints.openai.protocol")
        protocol.ChatCompletionRequest = types.SimpleNamespace(model_validate=parsed)
        protocol.ThinkingParam = types.SimpleNamespace(
            model_validate=lambda value: value
        )
        utils = types.ModuleType("sglang.srt.function_call.utils")
        utils.normalize_json_schema_types = Mock()
        orjson = types.ModuleType("orjson")
        orjson.loads = json.loads
        tokenizer = RecordingTokenizer()
        body = {"messages": [{"role": "user", "content": "Hello"}]}
        with patch.dict(
            sys.modules,
            {protocol.__name__: protocol, utils.__name__: utils, "orjson": orjson},
        ):
            k3.encode_text_request(
                tokenizer,
                body,
                {"reasoning_effort": "high", "custom": "server", "server_only": True},
            )
        options = tokenizer.calls[0][1]
        self.assertEqual(options["thinking_effort"], "max")
        self.assertEqual(options["reasoning_effort"], "high")
        self.assertEqual(options["custom"], "request")
        self.assertTrue(options["server_only"])
        self.assertEqual(body, {"messages": [{"role": "user", "content": "Hello"}]})

    def test_loader_pins_revision_and_warms_renderer_before_ready(self):
        tokenizer = RecordingTokenizer()
        get_tokenizer = Mock(return_value=tokenizer)
        module = types.ModuleType("sglang.srt.utils.hf_transformers.tokenizer")
        module.get_tokenizer = get_tokenizer
        config_module = types.ModuleType("sglang.srt.utils.hf_transformers.config")
        hf_config = types.SimpleNamespace(
            architectures=["KimiK31ForConditionalGeneration"], model_type="kimi_k31"
        )
        config_module.get_config = Mock(return_value=hf_config)
        encoding_module = types.ModuleType(
            "sglang.srt.entrypoints.openai.chat_encoding"
        )
        encoding_module.resolve_chat_encoding_spec = Mock(return_value="kimi_k3")
        defaults = {"thinking": False}
        warmup = Mock(return_value=[10, 20])
        with (
            patch.dict(
                sys.modules,
                {
                    module.__name__: module,
                    config_module.__name__: config_module,
                    encoding_module.__name__: encoding_module,
                },
            ),
            patch.object(k3, "encode_request", warmup),
        ):
            actual_tokenizer, actual_defaults, actual_model_type = k3._load_tokenizer(
                {
                    "tokenizer_path": "/models/k3",
                    "revision": "abc123",
                    "trust_remote_code": True,
                    "default_chat_template_kwargs": defaults,
                }
            )
        config_module.get_config.assert_called_once_with(
            "/models/k3", trust_remote_code=True, revision="abc123"
        )
        encoding_module.resolve_chat_encoding_spec.assert_called_once_with(
            hf_config=hf_config, tokenizer=None
        )
        get_tokenizer.assert_called_once_with(
            "/models/k3",
            trust_remote_code=True,
            revision="abc123",
            tokenizer_revision="abc123",
        )
        self.assertIs(actual_tokenizer, tokenizer)
        self.assertEqual(actual_defaults, defaults)
        self.assertEqual(actual_model_type, "kimi_k31")
        warmup.assert_called_once_with(
            tokenizer,
            {"messages": [{"role": "user", "content": "Hello"}]},
            defaults,
            model_type="kimi_k31",
        )

    def test_loader_rejects_other_encoders_before_tokenizer_or_warmup(self):
        for spec in (None, "dsv4"):
            with self.subTest(spec=spec):
                config_module = types.ModuleType(
                    "sglang.srt.utils.hf_transformers.config"
                )
                hf_config = types.SimpleNamespace(architectures=["OtherModel"])
                config_module.get_config = Mock(return_value=hf_config)
                encoding_module = types.ModuleType(
                    "sglang.srt.entrypoints.openai.chat_encoding"
                )
                encoding_module.resolve_chat_encoding_spec = Mock(return_value=spec)
                tokenizer_module = types.ModuleType(
                    "sglang.srt.utils.hf_transformers.tokenizer"
                )
                tokenizer_module.get_tokenizer = Mock()
                with (
                    patch.dict(
                        sys.modules,
                        {
                            config_module.__name__: config_module,
                            encoding_module.__name__: encoding_module,
                            tokenizer_module.__name__: tokenizer_module,
                        },
                    ),
                    patch.object(k3, "encode_request") as warmup,
                    self.assertRaisesRegex(ValueError, "Kimi-K3 model configuration"),
                ):
                    k3._load_tokenizer({"tokenizer_path": "/models/other"})
                encoding_module.resolve_chat_encoding_spec.assert_called_once_with(
                    hf_config=hf_config, tokenizer=None
                )
                tokenizer_module.get_tokenizer.assert_not_called()
                warmup.assert_not_called()

    def test_loader_explains_missing_model_configuration(self):
        config_module = types.ModuleType("sglang.srt.utils.hf_transformers.config")
        cause = OSError("config.json not found")
        config_module.get_config = Mock(side_effect=cause)
        encoding_module = types.ModuleType(
            "sglang.srt.entrypoints.openai.chat_encoding"
        )
        encoding_module.resolve_chat_encoding_spec = Mock()
        tokenizer_module = types.ModuleType(
            "sglang.srt.utils.hf_transformers.tokenizer"
        )
        tokenizer_module.get_tokenizer = Mock()
        with (
            patch.dict(
                sys.modules,
                {
                    config_module.__name__: config_module,
                    encoding_module.__name__: encoding_module,
                    tokenizer_module.__name__: tokenizer_module,
                },
            ),
            self.assertRaises(ValueError) as caught,
        ):
            k3._load_tokenizer(
                {"tokenizer_path": "/models/tokenizer-only", "revision": "abc123"}
            )
        self.assertIs(caught.exception.__cause__, cause)
        for context in (
            "/models/tokenizer-only",
            "abc123",
            "loadable config.json",
            "tokenizer-only directories are insufficient",
            "config.json not found",
        ):
            self.assertIn(context, str(caught.exception))
        encoding_module.resolve_chat_encoding_spec.assert_not_called()
        tokenizer_module.get_tokenizer.assert_not_called()


class RealSchemaImageTests(unittest.TestCase):
    """Also run against the actual request schema in a SGLang environment."""

    @classmethod
    def setUpClass(cls):
        try:
            import orjson  # noqa: F401

            from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest
            from sglang.srt.function_call.utils import (  # noqa: F401
                normalize_json_schema_types,
            )
        except ModuleNotFoundError as exc:
            if os.environ.get("SGLANG_REQUIRE_KIMI_K3_SCHEMA_TESTS") == "1":
                raise
            raise unittest.SkipTest(
                f"SGLang Python dependencies unavailable: {exc}"
            ) from exc
        cls.request_class = ChatCompletionRequest

    def test_image_request_matches_validated_worker_renderer_input(self):
        tool = {
            "type": "function",
            "function": {"name": "read_image", "parameters": {"type": "object"}},
        }
        images = [
            {
                "url": "https://example.com/image.png",
                "detail": "high",
                "content_hash": "sha256:" + "AB" * 32,
            },
            {"url": "data:image/png;base64,aGVsbG8=", "detail": "low"},
        ]
        body = {
            "messages": [
                {"role": "developer", "content": "", "tools": [tool]},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "First <|kimi_image_placeholder|>"},
                        {"type": "image_url", "image_url": images[0]},
                        {"type": "text", "text": "Second"},
                        {"type": "image_url", "image_url": images[1]},
                    ],
                },
                {"role": "assistant", "content": "Comparison: "},
            ],
            "reasoning_effort": "high",
            "tool_choice": "required",
            "continue_final_message": True,
        }
        original = copy.deepcopy(body)
        tokenizer = RecordingTokenizer()
        router_ids = k3.encode_request(tokenizer, body)
        forwarded = {**copy.deepcopy(body), "kimi_k3_media_input_ids": router_ids}
        validated = self.request_class.model_validate(forwarded)
        self.assertIsNone(validated.input_ids)
        self.assertEqual(validated.kimi_k3_media_input_ids, router_ids)
        baseline = RecordingTokenizer()
        worker_ids = k3.encode_kimi_k3_messages(
            baseline,
            [message.model_dump() for message in validated.messages],
            validated,
        )
        self.assertEqual(router_ids, worker_ids)
        self.assertEqual(tokenizer.calls, baseline.calls)
        rendered, options = tokenizer.calls[0]
        actual_images = [
            part["image_url"]
            for part in rendered[1]["content"]
            if part["type"] == "image_url"
        ]
        self.assertEqual(
            [image["url"] for image in actual_images],
            [image["url"] for image in images],
        )
        self.assertEqual(actual_images[0]["content_hash"], "sha256:" + "ab" * 32)
        self.assertEqual(options["image_prompts"], ["<|media_pad|>"] * 2)
        self.assertEqual(options["thinking_effort"], "high")
        self.assertEqual(options["tool_choice"], "required")
        self.assertEqual(tokenizer.prefixes, ["Comparison: "])
        self.assertEqual(body, original)

    def test_k31_preserves_effort_and_history_argument_bytes(self):
        """A K3.1 medium tier and nested JSON spacing must survive offloading."""
        for arguments in ('{"options": { "x" : 1.00 }}', "not-json {literal}"):
            with self.subTest(arguments=arguments):
                tokenizer = RecordingTokenizer()
                body = {
                    "messages": [
                        {"role": "user", "content": "hello"},
                        {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call-1",
                                    "type": "function",
                                    "function": {"name": "f", "arguments": arguments},
                                }
                            ],
                        },
                        {"role": "tool", "tool_call_id": "call-1", "content": "done"},
                    ],
                    "reasoning_effort": "medium",
                    "tool_choice": "none",
                }
                ids = k3.encode_request(tokenizer, body, model_type="kimi_k31")
                forwarded = self.request_class.model_validate(
                    {**body, "kimi_k3_input_ids": ids}
                )
                self.assertIsNone(forwarded.input_ids)
                self.assertEqual(forwarded.kimi_k3_input_ids, ids)
                rendered, kwargs = tokenizer.calls[-1]
                self.assertEqual(
                    rendered[1]["tool_calls"][0]["function"]["arguments"], arguments
                )
                self.assertEqual(kwargs["thinking_effort"], "medium")
                self.assertEqual(kwargs["tool_choice"], "none")
                k3.encode_request(tokenizer, body, model_type="kimi_k3")
                self.assertEqual(tokenizer.calls[-1][1]["thinking_effort"], "high")

    def test_k31_controls_and_template_argument_validation(self):
        messages = [{"role": "user", "content": "hello"}]
        for control, effort, thinking in (
            ({"reasoning_effort": "minimal"}, "low", True),
            ({"reasoning_effort": "xhigh"}, "max", True),
            ({"thinking": {"type": "disabled", "effort": "high"}}, None, False),
            (
                {
                    "thinking": {"effort": "medium"},
                    "chat_template_kwargs": {"thinking_effort": "low"},
                },
                "low",
                True,
            ),
        ):
            with self.subTest(control=control):
                tokenizer = RecordingTokenizer()
                k3.encode_request(
                    tokenizer, {"messages": messages, **control}, model_type="kimi_k31"
                )
                kwargs = tokenizer.calls[-1][1]
                self.assertEqual(kwargs.get("thinking_effort"), effort)
                self.assertEqual(kwargs["thinking"], thinking)
        for kwargs in (
            {"tools": []},
            {"conversation": []},
            {"thinking_effort": "invalid"},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                k3.encode_request(
                    RecordingTokenizer(),
                    {"messages": messages, "chat_template_kwargs": kwargs},
                    model_type="kimi_k31",
                )
        for field in ("input_ids", "kimi_k3_input_ids", "kimi_k3_media_input_ids"):
            with self.subTest(field=field), self.assertRaises(ValueError):
                k3.encode_request(
                    RecordingTokenizer(), {"messages": messages, field: [1]}
                )

    def test_invalid_image_schema_never_reaches_renderer(self):
        for image in (
            {"url": "https://example.com/image.png", "detail": "invalid"},
            {"url": "https://example.com/image.png", "content_hash": "invalid"},
            {"url": "https://example.com/image.png", "max_dynamic_patch": "invalid"},
        ):
            with self.subTest(image=image):
                tokenizer = RecordingTokenizer()
                with self.assertRaises(ValueError):
                    k3.encode_request(
                        tokenizer,
                        {
                            "messages": [
                                {
                                    "role": "user",
                                    "content": [
                                        {"type": "image_url", "image_url": image}
                                    ],
                                }
                            ]
                        },
                    )
                self.assertEqual(tokenizer.calls, [])


class ProtocolTests(unittest.TestCase):
    def run_protocol(self, lines, *, encoder=None, loader=None):
        output = io.StringIO()
        result = k3.run_worker(
            io.StringIO(
                "\n".join(
                    json.dumps(line) if not isinstance(line, str) else line
                    for line in lines
                )
                + "\n"
            ),
            output,
            load_tokenizer=loader or (lambda _: (RecordingTokenizer(), {}, "kimi_k31")),
            encode=encoder or (lambda *args, **kwargs: [42]),
        )
        return result, [json.loads(line) for line in output.getvalue().splitlines()]

    def test_protocol_recovers_after_invalid_request(self):
        result, replies = self.run_protocol([{}, "{bad", {}])
        self.assertEqual(result, 0)
        self.assertEqual(replies[0], {"ready": True})
        self.assertEqual(replies[1]["error_type"], "request")
        self.assertEqual(replies[2], {"input_ids": [42]})

    def test_runtime_failure_is_internal(self):
        def fail(*args, **kwargs):
            raise RuntimeError("tokenizer unavailable")

        with self.assertLogs(k3.logger, level="ERROR"):
            _, replies = self.run_protocol([{}, {}], encoder=fail)
        self.assertEqual(
            replies[1], {"error": "tokenizer unavailable", "error_type": "internal"}
        )

    def test_init_failure_has_no_ready(self):
        def fail(*args, **kwargs):
            raise RuntimeError("load failed")

        with self.assertLogs(k3.logger, level="ERROR"):
            result, replies = self.run_protocol([{}], loader=fail)
        self.assertEqual(result, 1)
        self.assertEqual(replies, [{"error": "load failed", "error_type": "internal"}])

    def test_stdout_from_loader_and_encoder_is_not_protocol(self):
        def loader(config):
            print("remote tokenizer loading")
            return object(), {}, "kimi_k31"

        def encoder(*args, **kwargs):
            print("remote tokenizer rendering")
            return [0, 0xFFFFFFFF]

        errors = io.StringIO()
        with patch.object(sys, "stderr", errors):
            _, replies = self.run_protocol([{}, {}], loader=loader, encoder=encoder)
        self.assertEqual(replies, [{"ready": True}, {"input_ids": [0, 0xFFFFFFFF]}])
        self.assertIn("remote tokenizer loading", errors.getvalue())
        self.assertIn("remote tokenizer rendering", errors.getvalue())

    def test_invalid_ids_are_internal_errors(self):
        for ids in ([], [True], [-1], [0x100000000], [1.0], {"input_ids": [1]}):
            with self.subTest(ids=ids), self.assertLogs(k3.logger, level="ERROR"):
                _, replies = self.run_protocol(
                    [{}, {}], encoder=lambda *args, ids=ids, **kwargs: ids
                )
                self.assertEqual(replies[1]["error_type"], "internal")


class ProtocolStdoutTests(unittest.TestCase):
    def test_native_fd1_writes_do_not_reach_protocol_stream(self):
        # Native code writing to fd 1 must land on stderr, not the protocol.
        script = f"""
import importlib.util, os
spec = importlib.util.spec_from_file_location("k3", {str(HELPER)!r})
k3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(k3)
out = k3._protocol_stdout()
os.write(1, b"native noise\\n")
print("python noise", flush=True)
out.write('{{"ready":true}}\\n')
out.flush()
"""
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=60,
            check=True,
        )
        self.assertEqual(result.stdout, '{"ready":true}\n')
        self.assertIn("native noise", result.stderr)
        self.assertIn("python noise", result.stderr)


if __name__ == "__main__":
    unittest.main()
