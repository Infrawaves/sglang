"""Kimi K3 strict tool schema verdicts and grammar at the serving entry points.

Full deployment acceptance uses tests/tool_call_json_schema in
MoonshotAI/Kimi-Vendor-Verifier at 7b938c38bbd89c2404a9ed5e7cb40da8079206b4.
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from utils import engine_chunk, make_serving
from xgrammar import GrammarCompiler, GrammarMatcher, TokenizerInfo

from sglang.srt.entrypoints.anthropic.protocol import (
    AnthropicCountTokensRequest,
    AnthropicMessagesRequest,
)
from sglang.srt.entrypoints.anthropic.serving import AnthropicServing
from sglang.srt.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    Function,
    ResponsesRequest,
    TokenizeRequest,
    TokenizeResponse,
    Tool,
)
from sglang.srt.entrypoints.openai.serving_tokenize import OpenAIServingTokenize
from sglang.srt.environ import ToolStrictLevel, envs
from sglang.srt.function_call import kimi_k3_tool_schema
from sglang.srt.function_call.kimi_k3_tool_schema import (
    _WalleSchemaResolver,
    _WalleVerdictCache,
)
from sglang.srt.function_call.kimik3_format import (
    ARGUMENT_CLOSE,
    CALL_CLOSE,
    RESPONSE_CLOSE,
    TOOLS_CLOSE,
    TOOLS_OPEN,
)
from sglang.srt.function_call.walle_pool import (
    WalleBusyError,
    WalleCrashError,
    WalleReply,
    WalleWorkerPool,
)
from sglang.srt.runtime_context import reset_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.kimi_k3_tool_schema_utils import (
    run_on_tool_schema_loop,
    walle_installed,
)
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


# The required self-reference has no finite instance.
_REJECTED = {
    "type": "object",
    "properties": {"n": {"$ref": "#/$defs/N"}},
    "required": ["n"],
    "$defs": {
        "N": {
            "type": "object",
            "properties": {"n": {"$ref": "#/$defs/N"}},
            "required": ["n"],
        }
    },
}
# Walle drops the undeclared required name so the grammar can compile.
# Non-string values keep a raw-text argument from absorbing its successor.
_SERVED = {
    "type": "object",
    "properties": {"b": {"type": "integer"}, "a": {"type": "boolean"}},
    "required": ["a", "undeclared"],
}
_REJECTION = "tool 'f': Invalid tool parameters schema:"


def _object_schema():
    return {
        "type": "object",
        "properties": {"a": {"type": "string"}},
        "required": ["a", "undeclared"],
    }


def _tool(parameters, *, name="f", strict=True):
    return Tool(function=Function(name=name, parameters=parameters, strict=strict))


def _argument(key, kind, value):
    return f'<|open|>argument key="{key}" type="{kind}"<|sep|>{value}{ARGUMENT_CLOSE}'


def _call(*arguments, tool="f"):
    return (
        f'{TOOLS_OPEN}<|open|>call tool="{tool}" index="1"<|sep|>'
        + "".join(arguments)
        + CALL_CLOSE
        + TOOLS_CLOSE
    )


_DECLARED_ORDER = _call(
    _argument("b", "number", "1"), _argument("a", "boolean", "true")
)
_SORTED_ORDER = _call(_argument("a", "boolean", "true"), _argument("b", "number", "1"))

# Bytes plus <|close|> as one special token, as K3 emits it.
_CLOSE_TOKEN = "<|close|>"
_COMPILER = GrammarCompiler(
    TokenizerInfo([bytes([i]) for i in range(256)] + [_CLOSE_TOKEN.encode()])
)


def _token_accepts(compiled, text):
    token_ids, start = [], 0
    while (index := text.find(_CLOSE_TOKEN, start)) != -1:
        token_ids.extend(text[start:index].encode())
        token_ids.append(256)
        start = index + len(_CLOSE_TOKEN)
    token_ids.extend(text[start:].encode())
    matcher = GrammarMatcher(compiled)
    return all(matcher.accept_token(t) for t in token_ids) and matcher.is_completed()


_run = run_on_tool_schema_loop


class _ToolSchemaServingTest(CustomTestCase):
    def setUp(self):
        super().setUp()
        reset_context()
        self.addCleanup(reset_context)
        self.serving = make_serving(tool_call_parser="kimi_k3")
        self.serving.default_chat_template_kwargs = {}
        self.serving.template_manager.chat_template_name = None
        manager = self.serving.tokenizer_manager
        manager.tokenizer.apply_chat_template.return_value = [1, 2]
        manager.process_request_preprocessor = None
        manager.request_logger = Mock(log_requests=False, log_requests_level=0)

        async def run_in_request_preprocessor(func, *args, **kwargs):
            return func(*args, **kwargs)

        manager.run_in_request_preprocessor = run_in_request_preprocessor
        self.internal_request = None

        async def generate(internal_request, raw_request=None):
            self.internal_request = internal_request
            chunk = engine_chunk("ok", finish=True)
            chunk["meta_info"]["weight_version"] = "default"
            yield chunk

        manager.generate_request = generate

    def _assert_grammar(self, *, accepts=(_DECLARED_ORDER,), rejects=(_SORTED_ORDER,)):
        compiled = _COMPILER.compile_structural_tag(
            self.internal_request.sampling_params["structural_tag"]
        )
        # With thinking off the prompt leaves the response channel open.
        prefixes = ("", RESPONSE_CLOSE)
        for call in accepts:
            self.assertTrue(
                any(_token_accepts(compiled, p + call) for p in prefixes), call
            )
        for call in rejects:
            self.assertFalse(
                any(_token_accepts(compiled, p + call) for p in prefixes), call
            )

    # Also the gRPC bridge's path.
    def _chat(self, *tools):
        request = ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "Call f."}],
            tools=list(tools),
            tool_choice="required",
        )
        raw_request = Mock(headers={})
        return _run(self.serving.handle_request(request, raw_request))

    def _responses(self, parameters):
        request = ResponsesRequest(
            model="x",
            input="Call f.",
            store=False,
            tools=[
                {
                    "type": "function",
                    "name": "f",
                    "parameters": parameters,
                    "strict": True,
                }
            ],
            tool_choice="required",
        )
        return _run(self.serving.create_responses(request))

    def _anthropic_body(self, parameters, **fields):
        return {
            "model": "x",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "Call f."}],
            "tools": [{"name": "f", "input_schema": parameters, "strict": True}],
            "tool_choice": {"type": "any"},
            **fields,
        }

    def _tokenize(self, parameters):
        tokenize = OpenAIServingTokenize(self.serving.tokenizer_manager)
        tokenize.chat_serving = self.serving
        self.serving.tokenizer_manager.tokenizer.model_max_length = 4096
        request = TokenizeRequest.model_validate(
            {
                "model": "x",
                "messages": [{"role": "user", "content": "Call f."}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "f",
                            "parameters": parameters,
                            "strict": True,
                        },
                    }
                ],
            }
        )
        return _run(tokenize._handle_non_streaming_request(request, request, None))


@unittest.skipUnless(walle_installed(), "walle is not installed")
class TestKimiK3ToolSchemaEntryPoints(_ToolSchemaServingTest):
    def test_chat(self):
        response = self._chat(_tool(_REJECTED))
        self.assertEqual(response.status_code, 400)
        self.assertIn(_REJECTION, response.body.decode())
        self.assertIsNone(self.internal_request)

        self._chat(_tool(_SERVED))
        self._assert_grammar()

    def test_responses(self):
        response = self._responses(_REJECTED)
        self.assertEqual(response.status_code, 400)
        self.assertIn(_REJECTION, response.body.decode())
        self.assertIsNone(self.internal_request)

        self._responses(_SERVED)
        self._assert_grammar()

    def test_anthropic_messages(self):
        anthropic = AnthropicServing(self.serving)
        for stream in (False, True):
            with self.subTest(stream=stream):
                request = AnthropicMessagesRequest.model_validate(
                    self._anthropic_body(_REJECTED, stream=stream)
                )
                response = _run(anthropic.handle_messages(request, Mock(headers={})))
                self.assertEqual(response.status_code, 400)
                payload = json.loads(response.body)
                self.assertEqual(payload["error"]["type"], "invalid_request_error")
                self.assertIn(_REJECTION, payload["error"]["message"])
        self.assertIsNone(self.internal_request)

        request = AnthropicMessagesRequest.model_validate(self._anthropic_body(_SERVED))
        _run(anthropic.handle_messages(request, Mock(headers={})))
        self._assert_grammar()

    def test_anthropic_count_tokens(self):
        """The prompt includes the client's tool schema even when generation rejects it."""
        anthropic = AnthropicServing(self.serving)
        tokenizer = self.serving.tokenizer_manager.tokenizer
        for parameters in (_REJECTED, _SERVED):
            with self.subTest(parameters=parameters):
                body = self._anthropic_body(parameters)
                body.pop("max_tokens")
                request = AnthropicCountTokensRequest.model_validate(body)
                response = _run(
                    anthropic.handle_count_tokens(request, Mock(headers={}))
                )
                self.assertEqual(response.status_code, 200)
                self.assertEqual(json.loads(response.body), {"input_tokens": 2})
                (tool,) = tokenizer.apply_chat_template.call_args.kwargs["tools"]
                self.assertEqual(tool["function"]["parameters"], parameters)

    def test_tokenize(self):
        """Tokenize renders through the tool grammar builder as well."""
        rejected = self._tokenize(_REJECTED)
        self.assertEqual(rejected.status_code, 400)
        self.assertIn(_REJECTION, json.loads(rejected.body)["message"])
        served = self._tokenize(_SERVED)
        self.assertIsInstance(served, TokenizeResponse)
        self.assertEqual(served.tokens, [1, 2])

    def test_parameter_strict_level_enforces_non_strict_tools(self):
        """Parameter-level strictness also validates and constrains a strict=False tool."""
        with envs.SGLANG_TOOL_STRICT_LEVEL.override(ToolStrictLevel.PARAMETER):
            response = self._chat(_tool(_REJECTED, strict=False))
            self.assertIsNone(
                self.internal_request, "Rejected schema reached generation"
            )
            self.assertEqual(response.status_code, 400)
            self.assertIn(_REJECTION, response.body.decode())

            self._chat(_tool(_SERVED, strict=False))
            self._assert_grammar()

    def test_each_strict_tool_uses_its_own_canonical_schema(self):
        self._chat(_tool(_SERVED), _tool(_object_schema(), name="g"))
        boolean = _argument("a", "boolean", "true")
        string = _argument("a", "string", "hello")
        self._assert_grammar(
            accepts=(_call(boolean), _call(string, tool="g")),
            rejects=(_call(string), _call(boolean, tool="g")),
        )

    def test_other_tool_parsers_constrain_the_raw_schema(self):
        """K3 drops const during normalization. K2 still enforces it on the wire."""
        parameters = {
            "type": "object",
            "properties": {"n": {"type": "integer", "const": 7}},
            "required": ["n"],
        }
        self._chat(_tool(parameters))
        self._assert_grammar(
            accepts=(_call(_argument("n", "number", "8")),), rejects=()
        )

        self.serving.tool_call_parser = "kimi_k2"
        self._chat(_tool(parameters))

        def k2_call(value):
            return (
                "<|tool_calls_section_begin|><|tool_call_begin|>functions.f:0"
                f'<|tool_call_argument_begin|>{{"n":{value}}}'
                "<|tool_call_end|><|tool_calls_section_end|>"
            )

        self._assert_grammar(accepts=(k2_call(7),), rejects=(k2_call(8),))


class TestKimiK3ToolSchemaErrors(_ToolSchemaServingTest):
    def setUp(self):
        # These cases inject worker failures after startup.
        with patch("sglang.srt.entrypoints.openai.serving_chat.check_walle_installed"):
            super().setUp()

    def _every_route(self):
        """Status code and body of every entry point for the served schema."""
        anthropic = AnthropicServing(self.serving)
        count_body = self._anthropic_body(_SERVED)
        count_body.pop("max_tokens")

        def messages(stream):
            request = AnthropicMessagesRequest.model_validate(
                self._anthropic_body(_SERVED, stream=stream)
            )
            return _run(anthropic.handle_messages(request, Mock(headers={})))

        responses = {
            "chat": self._chat(_tool(_SERVED)),
            "responses": self._responses(_SERVED),
            "anthropic": messages(False),
            "anthropic_stream": messages(True),
            "count_tokens": _run(
                anthropic.handle_count_tokens(
                    AnthropicCountTokensRequest.model_validate(count_body),
                    Mock(headers={}),
                )
            ),
            "tokenize": self._tokenize(_SERVED),
        }
        return {
            route: (response.status_code, json.loads(response.body))
            for route, response in responses.items()
        }

    def _mock_walle_pool(self):
        pool = Mock(spec=WalleWorkerPool)

        async def make_resolver():
            return _WalleSchemaResolver(pool=pool, cache=_WalleVerdictCache(1 << 20))

        patcher = patch.object(
            kimi_k3_tool_schema,
            "_process_walle_resolver",
            return_value=_run(make_resolver()),
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        return pool

    def test_empty_parameters_are_valid_but_a_reduced_schema_is_rejected(self):
        pool = self._mock_walle_pool()
        pool.canonical.return_value = WalleReply(canonical="{}")
        self._chat(_tool({}))
        self._assert_grammar(accepts=(_call(),), rejects=())

        response = self._chat(_tool(_REJECTED))
        self.assertEqual(response.status_code, 400)
        self.assertIn(
            "walle reduced it to {}", json.loads(response.body)["error"]["message"]
        )

    def test_cached_rejection_precedes_queue_timeout(self):
        """A known rejection returns 400 even when a preceding cache miss would time out."""
        pool = self._mock_walle_pool()
        pool.canonical.return_value = WalleReply(
            canonical="{}", warning="infinite recursion"
        )
        self.assertEqual(self._chat(_tool(_REJECTED, name="seed")).status_code, 400)

        pool.canonical.side_effect = WalleBusyError(
            "no schema validator became free within 2 s"
        )
        b = _tool(_SERVED, name="B")
        response = self._chat(
            _tool(_SERVED, name="ignored", strict=False), b, _tool(_REJECTED, name="A")
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(
            json.loads(response.body)["error"]["message"],
            "tool 'A': Invalid tool parameters schema: infinite recursion",
        )

        self.assertEqual(self._chat(b).status_code, 503)
        self.assertIsNone(self.internal_request)

    def test_repeated_worker_crashes_return_503_until_recovery(self):
        """After the pool exhausts its retry, the same schema can succeed on a later request."""
        pool = self._mock_walle_pool()
        pool.canonical.side_effect = WalleCrashError(
            "the schema validator crashed on this schema"
        )
        response = self._chat(_tool(_SERVED))
        self.assertEqual(response.status_code, 503)
        self.assertIsNone(self.internal_request)

        pool.canonical.side_effect = None
        pool.canonical.return_value = WalleReply(
            canonical=json.dumps({**_SERVED, "required": ["a"]})
        )
        self._chat(_tool(_SERVED))
        self._assert_grammar()

    def test_busy_walle_is_503_not_400(self):
        """No idle walle worker within the queue timeout is overload, not a bad
        schema: the client should retry the same request."""
        busy = WalleBusyError("no schema validator became free within 2 s")
        with patch.object(_WalleSchemaResolver, "verdicts", side_effect=busy):
            routes = self._every_route()
        self.assertEqual(routes.pop("count_tokens"), (200, {"input_tokens": 2}))
        self.assertEqual(
            {route: status for route, (status, _) in routes.items()},
            dict.fromkeys(routes, 503),
        )
        self.assertEqual(routes["anthropic"][1]["error"]["type"], "overloaded_error")
        self.assertEqual(routes["responses"][1]["error"]["type"], "server_error")
        self.assertIsNone(self.internal_request)

    def test_walle_failing_to_start_is_500_not_400(self):
        broken = RuntimeError("walle worker 1 failed to start")
        with patch.object(_WalleSchemaResolver, "verdicts", side_effect=broken):
            routes = self._every_route()
        self.assertEqual(routes.pop("count_tokens"), (200, {"input_tokens": 2}))
        self.assertEqual(
            {route: status for route, (status, _) in routes.items()},
            dict.fromkeys(routes, 500),
        )
        self.assertEqual(routes["anthropic"][1]["error"]["type"], "api_error")
        self.assertIsNone(self.internal_request)

    def test_malformed_schema_keeps_its_metaschema_error(self):
        """The metaschema check runs first: walle never sees such a schema, and
        the client gets the error naming the malformed keyword."""
        malformed = {"type": "object", "properties": {"a": {"type": 5}}}
        unreachable = AssertionError("walle asked about a malformed schema")
        anthropic = AnthropicServing(self.serving)
        with patch.object(_WalleSchemaResolver, "verdicts", side_effect=unreachable):
            routes = {
                "chat": self._chat(_tool(malformed)),
                "anthropic": _run(
                    anthropic.handle_messages(
                        AnthropicMessagesRequest.model_validate(
                            self._anthropic_body(malformed)
                        ),
                        Mock(headers={}),
                    )
                ),
                "tokenize": self._tokenize(malformed),
            }
        for route, response in routes.items():
            with self.subTest(route=route):
                self.assertEqual(response.status_code, 400)
                body = json.loads(response.body)
                self.assertIn(
                    "invalid 'parameters' schema",
                    body.get("message") or body["error"]["message"],
                )


class TestKimiK3WalleStartup(CustomTestCase):
    def test_only_k3_requires_the_pinned_walle_and_native_library(self):
        """Other Walle releases can silently loosen strict tool constraints."""
        reset_context()
        self.addCleanup(reset_context)
        for version, error in (
            ("0.1.16", r"requires Walle v0[.]1[.]19.*0[.]1[.]16"),
            ("0.1.20", r"requires Walle v0[.]1[.]19.*0[.]1[.]20"),
            # The pinned release without its native library fails in the probe worker.
            ("0.1.19", r"(?s)requires Walle v0[.]1[.]19.*libwalle[.]so"),
        ):
            with (
                self.subTest(version=version),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                (root / "walle").mkdir()
                (root / "walle/__init__.py").write_text(
                    'raise OSError("libwalle.so: cannot open shared object file")\n'
                )
                metadata = root / f"walle-{version}.dist-info"
                metadata.mkdir()
                (metadata / "METADATA").write_text(f"Name: walle\nVersion: {version}\n")
                pythonpath = os.pathsep.join(
                    filter(None, (directory, os.environ.get("PYTHONPATH")))
                )
                with (
                    patch.object(sys, "path", [directory, *sys.path]),
                    patch.dict(os.environ, {"PYTHONPATH": pythonpath}),
                ):
                    # A probe that passed earlier in this process is remembered.
                    kimi_k3_tool_schema._check_walle_worker_starts.cache_clear()
                    make_serving()
                    with self.assertRaisesRegex(RuntimeError, error):
                        make_serving(tool_call_parser="kimi_k3")


if __name__ == "__main__":
    unittest.main()
