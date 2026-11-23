import json
import sys
from unittest.mock import Mock

import pytest
import xgrammar as xgr
from xgrammar.testing import _is_grammar_accept_string

from sglang.srt.constrained.base_grammar_backend import InvalidGrammarObject
from sglang.srt.constrained.xgrammar_backend import XGrammarGrammarBackend
from sglang.srt.entrypoints.openai.protocol import (
    AllowedToolChoice,
    ChatCompletionRequest,
    Function,
    Tool,
    ToolChoice,
    ToolChoiceFuncName,
)
from sglang.srt.environ import ToolStrictLevel, envs
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.function_call.kimik3_detector import KimiK3Detector
from sglang.srt.function_call.kimik3_format import (
    ARGUMENT_CLOSE,
    CALL_CLOSE,
    MESSAGE_CLOSE,
    RESPONSE_CLOSE,
    RESPONSE_OPEN,
    THINK_CLOSE,
    TOOLS_CLOSE,
    TOOLS_OPEN,
)
from sglang.srt.function_call.kimik3_structural_tag import (
    get_kimik3_auto_tool_call_structural_tag,
    get_kimik3_json_response_structural_tag,
    get_kimik3_structural_tag,
)
from sglang.srt.function_call.tool_schema_guard import (
    ToolSchemaError,
    check_tool_schema_terminates,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=7, suite="base-a-test-cpu")

_CLOSE_TOKEN = "<|close|>"
_CLOSE_TOKEN_ID = 256
_TOKENIZER_INFO = xgr.TokenizerInfo(
    [bytes([token_id]) for token_id in range(256)] + [_CLOSE_TOKEN.encode()]
)
_TOKEN_COMPILER = xgr.GrammarCompiler(_TOKENIZER_INFO, cache_enabled=True)


def _tool(name="weather", strict=True):
    return Tool(
        type="function",
        function=Function(
            name=name,
            strict=strict,
            parameters={
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "pattern": "[A-Z][A-Za-z ]+",
                    },
                    "days": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 10,
                    },
                    "unit": {
                        "type": ["string", "null"],
                        "enum": ["celsius", "fahrenheit", None],
                    },
                    "metadata": {
                        "type": "object",
                        "properties": {"source": {"type": "string"}},
                        "required": ["source"],
                        "additionalProperties": False,
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["city", "days"],
                "additionalProperties": False,
            },
        ),
    )


def _allowed_choice(mode="auto", names=("weather",)):
    return AllowedToolChoice.model_validate(
        {
            "type": "allowed_tools",
            "allowed_tools": {
                "mode": mode,
                "tools": [
                    {"type": "function", "function": {"name": name}} for name in names
                ],
            },
        }
    )


def _argument(key, argument_type, value):
    return (
        f'<|open|>argument key="{key}" type="{argument_type}"<|sep|>'
        f"{value}{ARGUMENT_CLOSE}"
    )


def _call(name, index, *arguments):
    return (
        f'<|open|>call tool="{name}" index="{index}"<|sep|>'
        + "".join(arguments)
        + CALL_CLOSE
    )


def _tools_section(*calls):
    return TOOLS_OPEN + "".join(calls) + TOOLS_CLOSE


def _grammar(
    tools,
    tool_choice="auto",
    thinking_mode=False,
    parallel_tool_calls=True,
    response_channel_open=False,
):
    structural_tag = get_kimik3_structural_tag(
        tools,
        tool_choice=tool_choice,
        thinking_mode=thinking_mode,
        parallel_tool_calls=parallel_tool_calls,
        response_channel_open=response_channel_open,
    )
    return xgr.Grammar.from_structural_tag(structural_tag)


def _accepts(grammar, value):
    return _is_grammar_accept_string(grammar, value)


def _encode_with_close_token(value):
    token_ids = []
    start = 0
    while (index := value.find(_CLOSE_TOKEN, start)) != -1:
        token_ids.extend(value[start:index].encode())
        token_ids.append(_CLOSE_TOKEN_ID)
        start = index + len(_CLOSE_TOKEN)
    token_ids.extend(value[start:].encode())
    return token_ids


def _token_accepts(structural_tag, value):
    compiled = _TOKEN_COMPILER.compile_structural_tag(structural_tag)
    matcher = xgr.GrammarMatcher(compiled)
    for token_id in _encode_with_close_token(value):
        if not matcher.accept_token(token_id):
            return False
    return matcher.is_completed()


def _valid_weather_call(index=1):
    return _call(
        "weather",
        index,
        _argument("city", "string", "San Francisco"),
        _argument("days", "number", "3"),
        _argument("unit", "string", "celsius"),
        _argument("metadata", "object", '{"source":"forecast"}'),
        _argument("tags", "array", '["coastal","windy"]'),
    )


def test_strict_schema_accepts_native_xtml_values():
    grammar = _grammar([_tool()], tool_choice="required")

    assert _accepts(grammar, _tools_section(_valid_weather_call()))
    assert _accepts(
        grammar,
        _tools_section(
            _call(
                "weather",
                1,
                _argument("city", "string", "Paris"),
                _argument("days", "number", "1"),
            )
        ),
    )


@pytest.mark.parametrize(
    "arguments",
    [
        (_argument("city", "string", "paris"), _argument("days", "number", "3")),
        (
            _argument("city", "string", "Paris"),
            _argument("days", "number", "11"),
        ),
        (
            _argument("city", "number", "3"),
            _argument("days", "number", "3"),
        ),
        (
            _argument("city", "string", "Paris"),
            _argument("days", "number", "3"),
            _argument("unit", "string", "kelvin"),
        ),
        (
            _argument("city", "string", "Paris"),
            _argument("days", "number", "3"),
            _argument("tags", "array", "[coastal]"),
        ),
        (
            _argument("city", "string", "Paris"),
            _argument("days", "number", "3"),
            _argument("unknown", "string", "value"),
        ),
    ],
)
def test_strict_schema_rejects_invalid_parameters(arguments):
    grammar = _grammar([_tool()], tool_choice="required")

    assert not _accepts(grammar, _tools_section(_call("weather", 1, *arguments)))


def test_required_allows_response_prefix_but_requires_tools():
    grammar = _grammar([_tool()], tool_choice="required")
    response = "<|open|>response<|sep|>Checking.<|close|>response<|sep|>"

    assert _accepts(grammar, response + _tools_section(_valid_weather_call()))
    assert not _accepts(grammar, response)


def test_required_inside_an_open_response_channel_must_close_it_first():
    """With reasoning off the generation prompt leaves the response channel open.

    Free text there is a trap: the model answers in that channel, never reaches
    the tools section the grammar still requires, and so cannot stop either --
    generation runs to max_tokens with no tool call.
    """
    grammar = _grammar([_tool()], tool_choice="required", response_channel_open=True)
    call = _tools_section(_valid_weather_call())

    assert _accepts(grammar, RESPONSE_CLOSE + call)
    assert not _accepts(grammar, "Checking." + RESPONSE_CLOSE + call)
    assert not _accepts(grammar, call)
    assert not _accepts(grammar, RESPONSE_CLOSE)


def test_required_without_an_open_response_channel_is_unchanged():
    """Reasoning on: the model closes think itself, and a complete response block
    before the call stays legitimate."""
    grammar = _grammar([_tool()], tool_choice="required", response_channel_open=False)
    response = "<|open|>response<|sep|>Checking.<|close|>response<|sep|>"
    call = _tools_section(_valid_weather_call())

    assert _accepts(grammar, response + call)
    assert _accepts(grammar, call)
    assert not _accepts(grammar, response)


def test_required_rejects_message_close_before_tools():
    grammar = _grammar([_tool()], tool_choice="required")
    response = "<|open|>response<|sep|>Hi there!<|close|>response<|sep|>"

    assert not _accepts(grammar, response + MESSAGE_CLOSE)
    assert not _accepts(
        grammar, response + MESSAGE_CLOSE + _tools_section(_valid_weather_call())
    )
    assert _accepts(
        grammar, response + _tools_section(_valid_weather_call()) + MESSAGE_CLOSE
    )
    assert not _accepts(grammar, response + "<|close|>messagemessage<|sep|>")
    for near_miss in (
        "<|close|>Message<|sep|>",
        "<|close|>_message<|sep|>",
        TOOLS_CLOSE,
    ):
        assert not _accepts(grammar, response + near_miss)
    assert not _accepts(grammar, "<|open|>response<|sep|>Hi" + TOOLS_CLOSE)


_JSON_ANSWER_SCHEMA = {
    "type": "object",
    "properties": {"a": {"type": "integer"}},
    "required": ["a"],
    "additionalProperties": False,
}


def _json_response_grammar(**kwargs):
    return xgr.Grammar.from_structural_tag(
        get_kimik3_json_response_structural_tag(_JSON_ANSWER_SCHEMA, **kwargs)
    )


def test_json_response_is_written_as_the_response_block():
    """A bare JSON grammar masks the markers around K3's answer; the model, kept
    from opening the block, drifts onto whitespace or into a string it cannot
    leave and generates until max_tokens."""
    grammar = _json_response_grammar()
    answer = '{"a": 1}'

    # After reasoning the model opens the block itself.
    assert _accepts(grammar, RESPONSE_OPEN + answer + RESPONSE_CLOSE + MESSAGE_CLOSE)
    assert _accepts(grammar, RESPONSE_OPEN + answer + RESPONSE_CLOSE)
    # Without reasoning the generation prompt already opened it.
    assert _accepts(grammar, answer + RESPONSE_CLOSE)
    assert not _accepts(grammar, RESPONSE_OPEN + '{"a": "x"}' + RESPONSE_CLOSE)
    assert not _accepts(grammar, RESPONSE_OPEN + answer)
    assert not _accepts(grammar, RESPONSE_OPEN + answer + RESPONSE_CLOSE + "a")


def test_json_response_frames_reasoning_and_caps_whitespace():
    answer = RESPONSE_OPEN + '{"a": 1}' + RESPONSE_CLOSE

    with_reasoning = _json_response_grammar(thinking_mode=True)
    assert _accepts(with_reasoning, "Plan the answer." + THINK_CLOSE + answer)
    assert not _accepts(with_reasoning, answer)

    capped = _json_response_grammar(max_whitespace_cnt=4)
    assert _accepts(capped, RESPONSE_OPEN + '{"a":  1}' + RESPONSE_CLOSE)
    assert not _accepts(
        capped, RESPONSE_OPEN + '{"a":' + " " * 8 + "1}" + RESPONSE_CLOSE
    )


def test_tool_choice_none_rejects_tools_section():
    grammar = _grammar([_tool()], tool_choice="none")
    plain = "<|open|>response<|sep|>Sunny.<|close|>response<|sep|>" + MESSAGE_CLOSE

    assert _accepts(grammar, plain)
    assert not _accepts(grammar, _tools_section(_valid_weather_call()) + MESSAGE_CLOSE)
    assert not _accepts(grammar, "<|open|>tools<|close|>message<|sep|>")
    assert not _accepts(grammar, "<|sep|><|close|>tools<|sep|>" + MESSAGE_CLOSE)
    assert not _accepts(grammar, "")
    for blank in ("", " ", "\n\n"):
        empty = f"<|open|>response<|sep|>{blank}<|close|>response<|sep|>"
        assert not _accepts(grammar, empty + MESSAGE_CLOSE)
    for text in ("é", "👋 hi", "<b>bold</b>", "x"):
        response = f"<|open|>response<|sep|>{text}<|close|>response<|sep|>"
        assert _accepts(grammar, response)
    assert not _accepts(grammar, "<|open|>response<|sep|>x<|close|>Message<|sep|>")
    assert _accepts(
        grammar, "<|open|>response<|sep|>- **a** é\n<b><|close|>response<|sep|>"
    )


def test_auto_allows_plain_response_or_multiple_tool_calls():
    grammar = _grammar([_tool(), _tool("forecast")])
    plain = (
        "<|open|>response<|sep|>No tool needed."
        "<|close|>response<|sep|><|close|>message<|sep|>"
    )
    calls = _tools_section(
        _valid_weather_call(),
        _call(
            "forecast",
            2,
            _argument("city", "string", "London"),
            _argument("days", "number", "2"),
        ),
    )

    assert _accepts(grammar, plain)
    assert _accepts(grammar, calls)


def test_named_tool_choice_forces_only_the_selected_tool():
    grammar = _grammar(
        [_tool(), _tool("forecast")],
        tool_choice=ToolChoice(function=ToolChoiceFuncName(name="forecast")),
    )
    forecast_call = _call(
        "forecast",
        1,
        _argument("city", "string", "London"),
        _argument("days", "number", "2"),
    )

    assert _accepts(grammar, _tools_section(forecast_call))
    assert not _accepts(grammar, _tools_section(_valid_weather_call()))


@pytest.mark.parametrize("mode", ["auto", "required"])
def test_allowed_tools_constrains_names_without_making_parameters_strict(mode):
    tools = [_tool(name, strict=False) for name in ("weather", "forecast", "search")]
    request = ChatCompletionRequest(
        messages=[{"role": "user", "content": "Weather?"}],
        tools=tools,
        tool_choice={
            "type": "allowed_tools",
            "allowed_tools": {
                "mode": mode,
                "tools": [
                    {"type": "function", "function": {"name": name}}
                    for name in ("weather", "forecast")
                ],
            },
        },
    )
    with envs.SGLANG_TOOL_STRICT_LEVEL.override(ToolStrictLevel.OFF):
        constraint = FunctionCallParser(
            tools=tools, tool_call_parser="kimi_k3"
        ).get_structure_constraint(request.tool_choice)
    assert constraint is not None
    assert constraint[0] == "structural_tag"
    grammar = xgr.Grammar.from_structural_tag(constraint[1])
    loose_args = _argument("unknown", "string", "not in the schema")
    assert _accepts(grammar, _tools_section(_call("weather", 1, loose_args)))
    assert _accepts(
        grammar,
        _tools_section(_call("weather", 1), _call("forecast", 2)),
    )
    assert not _accepts(grammar, _tools_section(_call("search", 1)))
    assert not _accepts(
        grammar,
        _tools_section(_call("weather", 1), _call("search", 2)),
    )
    assert _accepts(grammar, "No tool needed.") == (mode == "auto")
    assert all(not tool.function.strict for tool in tools)


@pytest.mark.parametrize("mode", ["auto", "required"])
@pytest.mark.parametrize("thinking_mode", [False, True])
def test_empty_allowed_tools_only_accepts_text(mode, thinking_mode):
    parser = FunctionCallParser(tools=[_tool()], tool_call_parser="kimi_k3")
    constraint = parser.get_structure_constraint(
        _allowed_choice(mode=mode, names=()), thinking_mode=thinking_mode
    )
    grammar = xgr.Grammar.from_structural_tag(constraint[1])
    prefix = "Thinking." + THINK_CLOSE if thinking_mode else ""
    assert _accepts(grammar, prefix + "No tool needed.")
    assert not _accepts(grammar, prefix + _tools_section(_valid_weather_call()))
    assert not _accepts(grammar, prefix + _valid_weather_call())


@pytest.mark.parametrize("mode", ["auto", "required"])
@pytest.mark.parametrize("parallel", [False, True])
def test_allowed_tools_token_matching_preserves_parallel_and_strict(mode, parallel):
    tools = [_tool(), _tool("search")]
    with envs.SGLANG_TOOL_STRICT_LEVEL.override(ToolStrictLevel.OFF):
        constraint = FunctionCallParser(
            tools=tools, tool_call_parser="kimi_k3"
        ).get_structure_constraint(
            _allowed_choice(mode=mode), parallel_tool_calls=parallel, thinking_mode=True
        )
    tag = constraint[1]
    prefix = "Thinking." + THINK_CLOSE
    assert _token_accepts(tag, prefix + _tools_section(_valid_weather_call()))
    assert not _token_accepts(tag, prefix + _tools_section(_call("weather", 1)))
    assert (
        _token_accepts(
            tag, prefix + _tools_section(_valid_weather_call(), _valid_weather_call(2))
        )
        == parallel
    )
    matcher = xgr.GrammarMatcher(_TOKEN_COMPILER.compile_structural_tag(tag))
    assert matcher.accept_string(prefix + TOOLS_OPEN + '<|open|>call tool="')
    assert not matcher.accept_token(ord("s"))


def test_allowed_tools_honors_parameter_strict_level_without_mutating_declarations():
    tools = [_tool(strict=False)]
    with envs.SGLANG_TOOL_STRICT_LEVEL.override(ToolStrictLevel.PARAMETER):
        constraint = FunctionCallParser(
            tools=tools, tool_call_parser="kimi_k3"
        ).get_structure_constraint(_allowed_choice())
    assert _token_accepts(constraint[1], _tools_section(_valid_weather_call()))
    assert not _token_accepts(constraint[1], _tools_section(_call("weather", 1)))
    assert not tools[0].function.strict


def test_allowed_tools_compile_failure_is_an_invalid_grammar_not_unconstrained():
    tool = _tool()
    tool.function.parameters = {
        "type": "object",
        "properties": {"value": {"$ref": "#/$defs/missing"}},
        "required": ["value"],
    }
    constraint = FunctionCallParser(
        tools=[tool], tool_call_parser="kimi_k3"
    ).get_structure_constraint(_allowed_choice())
    request = ChatCompletionRequest(
        messages=[{"role": "user", "content": "Hello"}], tool_choice=_allowed_choice()
    )
    params = request.to_sampling_params([], {}, constraint)
    backend = XGrammarGrammarBackend(
        tokenizer=Mock(init_xgrammar=lambda: (_TOKENIZER_INFO, None)), vocab_size=257
    )
    try:
        result = backend.dispatch_structural_tag(params["structural_tag"])
        assert isinstance(result, InvalidGrammarObject)
        assert "missing" in result.error_message
    finally:
        backend.executor.shutdown(wait=True)


def test_function_call_parser_uses_native_tag_for_named_tool_choice():
    tool_choice = ToolChoice(function=ToolChoiceFuncName(name="forecast"))
    constraint = FunctionCallParser(
        [_tool(), _tool("forecast")], "kimi_k3"
    ).get_structure_constraint(tool_choice)
    assert constraint is not None
    grammar = xgr.Grammar.from_structural_tag(constraint[1])
    forecast_call = _call(
        "forecast",
        1,
        _argument("city", "string", "London"),
        _argument("days", "number", "2"),
    )

    assert _accepts(grammar, _tools_section(forecast_call))
    assert not _accepts(grammar, _tools_section(_valid_weather_call()))


def test_non_strict_tool_keeps_xtml_structure_and_loose_parameters():
    grammar = _grammar([_tool(strict=False)], tool_choice="required")
    call = _call(
        "weather",
        1,
        _argument("custom", "array", '["x",1]'),
        _argument("other", "string", "raw text"),
    )

    assert _accepts(grammar, _tools_section(call))
    assert not _accepts(grammar, _tools_section("unstructured"))


def test_strict_schema_supports_refs_and_mixed_unions():
    tool = Tool(
        type="function",
        function=Function(
            name="convert",
            strict=True,
            parameters={
                "type": "object",
                "$defs": {
                    "mode": {
                        "type": "string",
                        "enum": ["fast", "safe"],
                    }
                },
                "properties": {
                    "mode": {"$ref": "#/$defs/mode"},
                    "value": {
                        "anyOf": [
                            {"type": "string", "enum": ["auto"]},
                            {
                                "type": "integer",
                                "minimum": 2,
                                "maximum": 3,
                            },
                        ]
                    },
                },
                "required": ["mode", "value"],
                "additionalProperties": False,
            },
        ),
    )
    grammar = _grammar([tool], tool_choice="required")

    assert _accepts(
        grammar,
        _tools_section(
            _call(
                "convert",
                1,
                _argument("mode", "string", "fast"),
                _argument("value", "number", "2"),
            )
        ),
    )
    assert not _accepts(
        grammar,
        _tools_section(
            _call(
                "convert",
                1,
                _argument("mode", "string", "unsafe"),
                _argument("value", "number", "1"),
            )
        ),
    )


def _value_tool(value_schema, **extra):
    return Tool(
        type="function",
        function=Function(
            name="emit",
            strict=True,
            parameters={
                "type": "object",
                "required": ["value"],
                "additionalProperties": False,
                "properties": {"value": value_schema},
                **extra,
            },
        ),
    )


def _accepts_value(grammar, value):
    return _accepts(
        grammar,
        _tools_section(_call("emit", 1, _argument("value", "object", value))),
    )


def test_strict_schema_resolves_refs_into_the_root_structure():
    """Each argument compiles as its own schema, so refs into the root's
    properties (kvv TestReferences) must still resolve there."""
    sibling = _grammar(
        [
            _value_tool(
                {
                    "type": "object",
                    "properties": {
                        "a": {"type": "string"},
                        "b": {"$ref": "#/properties/value/properties/a"},
                    },
                }
            )
        ],
        tool_choice="required",
    )
    assert _accepts_value(sibling, '{"a":"x","b":"y"}')
    assert not _accepts_value(sibling, '{"b":1}')

    items = _grammar(
        [
            _value_tool(
                {
                    "type": "object",
                    "properties": {
                        "list": {"type": "array", "items": {"type": "string"}},
                        "one": {"$ref": "#/properties/value/properties/list/items"},
                    },
                }
            )
        ],
        tool_choice="required",
    )
    assert _accepts_value(items, '{"list":["a"],"one":"b"}')
    assert not _accepts_value(items, '{"one":2}')

    # A ref chain through both a definition pool and the root structure.
    chained = _grammar(
        [
            _value_tool(
                {
                    "type": "object",
                    "properties": {
                        "a": {"$ref": "#/$defs/S"},
                        "b": {"type": "integer"},
                        "c": {"$ref": "#/properties/value/properties/b"},
                    },
                },
                **{"$defs": {"S": {"type": "string"}}},
            )
        ],
        tool_choice="required",
    )
    assert _accepts_value(chained, '{"a":"s","b":1,"c":2}')
    assert not _accepts_value(chained, '{"c":"two"}')


def test_strict_schema_resolves_recursive_root_structure_refs():
    tree = {
        "type": "object",
        "properties": {
            "kids": {
                "type": "array",
                "items": {"$ref": "#/properties/value"},
            }
        },
        "additionalProperties": False,
    }
    grammar = _grammar([_value_tool(tree)], tool_choice="required")
    assert _accepts_value(grammar, '{"kids":[{"kids":[]},{}]}')
    assert not _accepts_value(grammar, '{"kids":[{"kids":[1]}]}')


def test_strict_schema_handles_number_enums_and_all_of_integer_constraints():
    tool = Tool(
        type="function",
        function=Function(
            name="score",
            strict=True,
            parameters={
                "type": "object",
                "properties": {
                    "value": {
                        "type": "number",
                        "enum": [1, 1.5],
                    },
                    "count": {
                        "allOf": [
                            {"type": "number"},
                            {"type": "integer", "minimum": 1},
                        ]
                    },
                },
                "required": ["value", "count"],
                "additionalProperties": False,
            },
        ),
    )
    grammar = _grammar([tool], tool_choice="required")

    assert _accepts(
        grammar,
        _tools_section(
            _call(
                "score",
                1,
                _argument("value", "number", "1"),
                _argument("count", "number", "2"),
            )
        ),
    )
    assert not _accepts(
        grammar,
        _tools_section(
            _call(
                "score",
                1,
                _argument("value", "number", "2"),
                _argument("count", "number", "1.5"),
            )
        ),
    )


def test_strict_schema_handles_one_sided_negative_integer_minimum():
    tool = Tool(
        type="function",
        function=Function(
            name="submit",
            strict=True,
            parameters={
                "type": "object",
                "properties": {
                    "value": {
                        "type": "integer",
                        "minimum": -1000000,
                    }
                },
                "required": ["value"],
                "additionalProperties": False,
            },
        ),
    )
    grammar = _grammar([tool], tool_choice="required")

    for value in ("-1000000", "-999999", "-1", "0", "1000001"):
        assert _accepts(
            grammar,
            _tools_section(_call("submit", 1, _argument("value", "number", value))),
        )
    for value in ("-", "-1000001", "1.5"):
        assert not _accepts(
            grammar,
            _tools_section(_call("submit", 1, _argument("value", "number", value))),
        )


def test_strict_schema_enforces_non_ascii_property_names():
    """An emoji or CJK property name (kvv TestEnforcerCases) keeps its object
    constraint; models emit such names as raw UTF-8."""
    key = "abc\U0001f60a"
    value_schema = {
        "type": "object",
        "properties": {key: {"type": "string"}},
        "required": [key],
        "additionalProperties": False,
    }
    grammar = _grammar([_value_tool(value_schema)], tool_choice="required")
    assert _accepts_value(grammar, json.dumps({key: "x"}, ensure_ascii=False))
    assert not _accepts_value(grammar, "{}")
    assert not _accepts_value(grammar, json.dumps({key: 1}, ensure_ascii=False))


def test_strict_root_admits_undeclared_arguments_only_when_declared_open():
    schema = {
        "type": "object",
        "properties": {"count": {"type": "number"}},
        "required": ["count"],
    }
    call = _kvv_call(
        _argument("count", "number", "1"), _argument("confidence", "number", "0.9")
    )

    assert not _accepts(_kvv_grammar(schema), call)
    assert _accepts(_kvv_grammar({**schema, "additionalProperties": True}), call)


def test_dynamic_argument_key_compiles_without_xgrammar_unicode_warning(capfd):
    tool = Tool(
        type="function",
        function=Function(
            name="annotate",
            strict=True,
            parameters={
                "type": "object",
                "properties": {},
            },
        ),
    )
    grammar = _grammar([tool], tool_choice="required")

    assert _accepts(
        grammar,
        _tools_section(_call("annotate", 1, _argument("置信度", "number", "0.9"))),
    )
    assert "Negative Character class" not in capfd.readouterr().err


def test_strict_empty_object_accepts_no_arguments_only():
    tool = Tool(
        type="function",
        function=Function(
            name="ping",
            strict=True,
            parameters={
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        ),
    )
    grammar = _grammar([tool], tool_choice="required")

    assert _accepts(grammar, _tools_section(_call("ping", 1)))
    assert not _accepts(
        grammar,
        _tools_section(_call("ping", 1, _argument("unexpected", "string", "value"))),
    )


def test_tool_strict_level_controls_native_tag_parameter_schema():
    invalid_call = _tools_section(
        _call(
            "weather",
            264,
            _argument("city", "string", "paris"),
            _argument("days", "number", "99"),
        )
    )
    empty_call = _tools_section(_call("weather", 264))

    with envs.SGLANG_TOOL_STRICT_LEVEL.override(ToolStrictLevel.OFF.value):
        constraint = FunctionCallParser(
            [_tool(strict=False)], "kimi_k3"
        ).get_structure_constraint("auto")
        assert constraint is not None
        assert _token_accepts(constraint[1], invalid_call)
        assert not _token_accepts(constraint[1], empty_call)

    with envs.SGLANG_TOOL_STRICT_LEVEL.override(ToolStrictLevel.FUNCTION.value):
        constraint = FunctionCallParser(
            [_tool(strict=False)], "kimi_k3"
        ).get_structure_constraint("auto")
        assert constraint is not None
        grammar = xgr.Grammar.from_structural_tag(constraint[1])
        assert _accepts(grammar, invalid_call)
        assert _accepts(grammar, empty_call)

    with envs.SGLANG_TOOL_STRICT_LEVEL.override(ToolStrictLevel.PARAMETER.value):
        constraint = FunctionCallParser(
            [_tool(strict=False)], "kimi_k3"
        ).get_structure_constraint("auto")
        assert constraint is not None
        assert not _accepts(
            xgr.Grammar.from_structural_tag(constraint[1]), invalid_call
        )


def test_auto_hook_constrains_all_calls_and_requires_nonempty_values():
    structural_tag = get_kimik3_auto_tool_call_structural_tag([_tool(strict=False)])
    assert structural_tag is not None
    first = _call(
        "weather",
        3,
        _argument("city", "string", "Paris"),
        _argument("days", "number", "3"),
    )
    second = _call(
        "weather",
        264,
        _argument("city", "string", "London"),
        _argument("days", "number", "2"),
    )

    assert _token_accepts(structural_tag, _tools_section(first, second))
    assert not _token_accepts(
        structural_tag,
        _tools_section(first, _call("weather", 264)),
    )
    assert not _token_accepts(
        structural_tag,
        _tools_section(
            _call(
                "weather",
                3,
                _argument("city", "string", ""),
                _argument("days", "number", "3"),
            )
        ),
    )
    assert not _token_accepts(structural_tag, _tools_section(_call("weather", 3)))


def test_auto_hook_rejects_unknown_or_unclosed_calls():
    structural_tag = get_kimik3_auto_tool_call_structural_tag([_tool(strict=False)])
    assert structural_tag is not None
    call = _call(
        "weather",
        49,
        _argument("city", "string", "Paris"),
        _argument("days", "number", "3"),
    )
    unknown = _call(
        "forecast",
        49,
        _argument("city", "string", "Paris"),
        _argument("days", "number", "3"),
    )

    assert not _token_accepts(structural_tag, _tools_section(unknown))
    assert not _token_accepts(
        structural_tag, TOOLS_OPEN + call.removesuffix(CALL_CLOSE)
    )
    assert not _token_accepts(structural_tag, TOOLS_OPEN + call)


def test_auto_hook_does_not_swallow_parser_visible_closes():
    structural_tag = get_kimik3_auto_tool_call_structural_tag([_tool(strict=False)])
    assert structural_tag is not None
    output = (
        TOOLS_OPEN
        + '<|open|>call tool="weather" index="23"<|sep|>'
        + _argument("city", "string", "Paris")
        + '<|open|>argument key="days" type="number"<|sep|>'
        + ARGUMENT_CLOSE
        + CALL_CLOSE
        + "3"
        + ARGUMENT_CLOSE
        + CALL_CLOSE
        + TOOLS_CLOSE
    )

    parsed = KimiK3Detector().detect_and_parse(output, [_tool(strict=False)])

    assert json.loads(parsed.calls[0].parameters) == {"city": "Paris", "days": ""}
    assert not _token_accepts(structural_tag, output)


@pytest.mark.parametrize(
    "tool_strict_level",
    [
        ToolStrictLevel.OFF,
        ToolStrictLevel.FUNCTION,
        ToolStrictLevel.PARAMETER,
    ],
)
def test_parallel_tool_calls_false_rejects_second_call(tool_strict_level):
    with envs.SGLANG_TOOL_STRICT_LEVEL.override(tool_strict_level.value):
        constraint = FunctionCallParser(
            [_tool(strict=False)], "kimi_k3"
        ).get_structure_constraint("auto", parallel_tool_calls=False)
    assert constraint is not None
    first = _call(
        "weather",
        165,
        _argument("city", "string", "Paris"),
        _argument("days", "number", "3"),
    )
    second = _call(
        "weather",
        166,
        _argument("city", "string", "London"),
        _argument("days", "number", "2"),
    )

    assert _token_accepts(constraint[1], _tools_section(first))
    assert not _token_accepts(constraint[1], _tools_section(first, second))
    assert not _token_accepts(
        constraint[1], _tools_section(first) + _tools_section(second)
    )


def test_parallel_tool_calls_true_constrains_every_call():
    grammar = _grammar(
        [_tool(strict=False)],
        parallel_tool_calls=True,
    )
    first = _call(
        "weather",
        7,
        _argument("city", "string", "Paris"),
    )
    second = _call(
        "weather",
        19,
        _argument("other", "array", '["loose"]'),
    )

    assert _accepts(grammar, _tools_section(first, second))


def test_parameter_level_constrains_every_parallel_call():
    with envs.SGLANG_TOOL_STRICT_LEVEL.override(ToolStrictLevel.PARAMETER.value):
        constraint = FunctionCallParser(
            [_tool(strict=False)], "kimi_k3"
        ).get_structure_constraint("auto")
    assert constraint is not None
    grammar = xgr.Grammar.from_structural_tag(constraint[1])
    first = _valid_weather_call(index=3)
    valid_second = _call(
        "weather",
        49,
        _argument("city", "string", "London"),
        _argument("days", "number", "2"),
    )
    invalid_second = _call(
        "weather",
        49,
        _argument("city", "string", "london"),
        _argument("days", "number", "99"),
    )

    assert _accepts(grammar, _tools_section(first, valid_second))
    assert not _accepts(grammar, _tools_section(first, invalid_second))


def test_strict_tool_without_parameters_compiles_to_empty_arguments():
    """SGLANG_TOOL_STRICT_LEVEL=2 marks every tool strict, including tools
    that declare no parameters; the grammar build must not fail for them."""
    tool = Tool(type="function", function=Function(name="ping", strict=True))
    grammar = _grammar([tool], tool_choice="required")

    assert _accepts(grammar, _tools_section(_call("ping", 1)))
    assert not _accepts(
        grammar,
        _tools_section(_call("ping", 1, _argument("x", "string", "y"))),
    )


def test_parameter_level_keeps_constraint_with_no_parameter_tool():
    """A single no-parameter tool must not poison the whole request: a build
    error here was swallowed and required fell back to a JSON-only grammar
    the K3 parser cannot read."""
    tools = [
        _tool(strict=False),
        Tool(type="function", function=Function(name="ping")),
    ]
    with envs.SGLANG_TOOL_STRICT_LEVEL.override(ToolStrictLevel.PARAMETER.value):
        constraint = FunctionCallParser(tools, "kimi_k3").get_structure_constraint(
            "required"
        )

    assert constraint is not None
    assert constraint[0] == "structural_tag"


def test_all_of_number_branches_do_not_narrow_to_integer():
    """allOf with only number branches was intersected down to integer,
    silently dropping non-integer enum values."""
    tool = Tool(
        type="function",
        function=Function(
            name="scale",
            strict=True,
            parameters={
                "type": "object",
                "properties": {
                    "factor": {"allOf": [{"type": "number"}], "enum": [1.5, 2]},
                },
                "required": ["factor"],
                "additionalProperties": False,
            },
        ),
    )
    grammar = _grammar([tool], tool_choice="required")

    assert _accepts(
        grammar,
        _tools_section(_call("scale", 1, _argument("factor", "number", "1.5"))),
    )
    assert not _accepts(
        grammar,
        _tools_section(_call("scale", 1, _argument("factor", "number", "3"))),
    )


def test_auto_hook_serializes_into_sampling_parameters():
    constraint = FunctionCallParser(
        [_tool(strict=False)], "kimi_k3"
    ).get_structure_constraint("auto")
    assert constraint is not None
    request = ChatCompletionRequest(
        model="test",
        messages=[{"role": "user", "content": "Weather?"}],
        max_completion_tokens=16,
    )

    sampling_params = request.to_sampling_params(
        stop=[],
        model_generation_config={},
        tool_call_constraint=constraint,
    )

    serialized = json.loads(sampling_params["structural_tag"])
    assert serialized["type"] == "structural_tag"
    assert serialized["format"]["type"] == "triggered_tags"


def test_auto_hook_forces_one_typed_property_when_none_are_required():
    tool = Tool(
        type="function",
        function=Function(
            name="search",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer"},
                },
            },
        ),
    )
    structural_tag = get_kimik3_auto_tool_call_structural_tag([tool])
    assert structural_tag is not None

    assert _token_accepts(
        structural_tag,
        _tools_section(_call("search", 1, _argument("limit", "number", "3"))),
    )
    assert not _token_accepts(structural_tag, _tools_section(_call("search", 1)))


def test_auto_hook_keeps_structure_for_ambiguous_required_argument_type():
    tool = Tool(
        type="function",
        function=Function(
            name="lookup",
            parameters={
                "type": "object",
                "properties": {
                    "key": {"type": ["string", "integer"]},
                },
                "required": ["key"],
            },
        ),
    )

    structural_tag = get_kimik3_auto_tool_call_structural_tag([tool])
    assert structural_tag is not None
    grammar = xgr.Grammar.from_structural_tag(structural_tag)

    assert _accepts(
        grammar,
        _tools_section(_call("lookup", 9254, _argument("key", "number", "3"))),
    )
    assert not _accepts(
        grammar,
        TOOLS_OPEN + _call("lookup", 9254, _argument("key", "number", "3")),
    )


def test_parameter_level_applies_to_other_model_native_tags():
    with envs.SGLANG_TOOL_STRICT_LEVEL.override(ToolStrictLevel.PARAMETER.value):
        constraint = FunctionCallParser(
            [_tool(strict=False)], "kimi_k2"
        ).get_structure_constraint("auto")

    assert constraint is not None
    serialized = constraint[1].model_dump_json()
    assert '"properties"' in serialized
    assert '"city"' in serialized


def test_reasoning_prefix_is_owned_by_exactly_one_layer():
    tool = _tool()
    wrapped_by_xgrammar = _grammar([tool], tool_choice="required", thinking_mode=True)
    post_reasoning_only = _grammar([tool], tool_choice="required", thinking_mode=False)
    output = "reasoning" + THINK_CLOSE + _tools_section(_valid_weather_call())

    assert _accepts(wrapped_by_xgrammar, output)
    assert not _accepts(post_reasoning_only, output)
    assert _accepts(post_reasoning_only, _tools_section(_valid_weather_call()))


def test_output_may_start_inside_the_prefilled_response():
    # Without thinking the prompt ends with the response open marker.
    call = _tools_section(_valid_weather_call())
    required = _grammar([_tool()], tool_choice="required")
    assert _accepts(
        required, "Hi there!<|close|>response<|sep|>" + call + MESSAGE_CLOSE
    )
    assert not _accepts(required, "Hi there!<|close|>response<|sep|>" + MESSAGE_CLOSE)
    assert not _accepts(
        required, "Hi<|close|>response<|sep|>" + MESSAGE_CLOSE + call + MESSAGE_CLOSE
    )

    none = _grammar([_tool()], tool_choice="none")
    assert _accepts(none, "Sunny.<|close|>response<|sep|>" + MESSAGE_CLOSE)
    assert not _accepts(none, "<|close|>response<|sep|>" + MESSAGE_CLOSE)
    assert not _accepts(none, "Sunny.<|close|>response<|sep|>" + call)


def test_single_call_auto_admits_the_turn_close():
    grammar = _grammar([_tool()], parallel_tool_calls=False)
    assert _accepts(grammar, _tools_section(_valid_weather_call()) + MESSAGE_CLOSE)


# Strict schemas below come from the Kimi Vendor Verifier's walle cases.
_N_REQUIRED_SELF = {
    "type": "object",
    "properties": {"next": {"$ref": "#/$defs/N"}},
    "required": ["next"],
}


def _kvv_grammar(parameters):
    tool = Tool(
        type="function",
        function=Function(name="kvv", strict=True, parameters=parameters),
    )
    return _grammar([tool], tool_choice="required")


def _kvv_call(*arguments):
    return _tools_section(_call("kvv", 1, *arguments))


@pytest.mark.parametrize(
    "schema",
    [
        {
            "type": "object",
            "properties": {"node": {"$ref": "#/$defs/N"}},
            "$defs": {"N": _N_REQUIRED_SELF},
        },
        {
            "type": "object",
            "properties": {"root": {"$ref": "#/$defs/A"}},
            "required": ["root"],
            "$defs": {
                "A": {
                    "type": "object",
                    "properties": {"b": {"$ref": "#/$defs/B"}},
                    "required": ["b"],
                },
                "B": {
                    "type": "object",
                    "properties": {"a": {"$ref": "#/$defs/A"}},
                    "required": ["a"],
                },
            },
        },
        {
            "type": "object",
            "properties": {"root": {"$ref": "#/$defs/L1"}},
            "required": ["root"],
            "$defs": {
                "L1": {
                    "type": "object",
                    "properties": {"level2": {"$ref": "#/$defs/L2"}},
                    "required": ["level2"],
                },
                "L2": {"type": "array", "minItems": 1, "items": {"$ref": "#/$defs/L1"}},
            },
        },
    ],
)
def test_strict_schema_without_finite_instance_is_rejected(schema):
    """The verifier expects a 400 that names the schema; XGrammar compiles these
    and a forced call can only run to max_tokens."""
    with pytest.raises(ToolSchemaError, match="parameters schema"):
        check_tool_schema_terminates(schema)


@pytest.mark.parametrize(
    "schema",
    [
        {
            "type": "object",
            "properties": {"node": {"$ref": "#/$defs/N"}},
            "$defs": {
                "N": {
                    "type": "object",
                    "properties": {"next": {"$ref": "#/$defs/N"}},
                }
            },
        },
        {
            "type": "object",
            "properties": {"root": {"$ref": "#/$defs/L1"}},
            "required": ["root"],
            "$defs": {
                "L1": {
                    "type": "object",
                    "properties": {"level2": {"$ref": "#/$defs/L2"}},
                    "required": ["level2"],
                },
                "L2": {"type": "array", "items": {"$ref": "#/$defs/L1"}},
            },
        },
        # Walle strips keywords it does not support before checking refs.
        {
            "type": "object",
            "properties": {"a": {"not": {"$ref": "#/$defs/N"}}},
            "$defs": {"N": _N_REQUIRED_SELF},
        },
    ],
)
def test_strict_schema_with_finite_instance_is_accepted(schema):
    check_tool_schema_terminates(schema)


def test_required_names_without_a_schema_keep_the_call_constrained():
    """Walle prunes required names that have no schema; they must not leave the
    call unconstrained."""
    grammar = _kvv_grammar(
        {
            "type": "object",
            "properties": {"a": {"type": "string"}},
            "required": ["a", "b"],
        }
    )
    assert _accepts(grammar, _kvv_call(_argument("a", "string", "x")))
    assert not _accepts(grammar, _kvv_call())


@pytest.mark.parametrize(
    "schema",
    [
        {"type": "string", "minLength": 10, "maxLength": 2},
        {"type": "integer", "minimum": 10, "maximum": 2},
        {"type": "array", "minItems": 10, "maxItems": 2, "items": {"type": "string"}},
    ],
)
def test_contradicting_bounds_compile_as_unconstrained(schema):
    """Walle degrades min > max to {}, which XGrammar would refuse to compile."""
    grammar = _kvv_grammar({"type": "object", "properties": {"a": schema}})
    assert _accepts(grammar, _kvv_call(_argument("a", "string", "x")))


def test_enum_without_a_value_of_its_type_keeps_the_type():
    grammar = _kvv_grammar(
        {"type": "object", "properties": {"v": {"type": ["null"], "enum": [False]}}}
    )
    assert _accepts(grammar, _kvv_call(_argument("v", "null", "null")))
    assert not _accepts(grammar, _kvv_call(_argument("v", "boolean", "false")))


def test_declared_keys_do_not_escape_through_additional_properties():
    grammar = _kvv_grammar(
        {
            "type": "object",
            "properties": {"a": {"type": "string"}, "abc": {"type": "integer"}},
            "additionalProperties": True,
        }
    )
    assert not _accepts(grammar, _kvv_call(_argument("a", "number", "1")))
    assert not _accepts(grammar, _kvv_call(_argument("abc", "string", "x")))
    for undeclared in ("ab", "abcd", "b"):
        assert _accepts(grammar, _kvv_call(_argument(undeclared, "number", "1")))


def test_single_anyof_branch_keeps_the_stricter_parent_bound():
    grammar = _kvv_grammar(
        {
            "type": "object",
            "properties": {
                "v": {"minLength": 20, "anyOf": [{"type": "string", "minLength": 10}]}
            },
        }
    )
    assert not _accepts(grammar, _kvv_call(_argument("v", "string", "a" * 19)))
    assert _accepts(grammar, _kvv_call(_argument("v", "string", "a" * 20)))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
