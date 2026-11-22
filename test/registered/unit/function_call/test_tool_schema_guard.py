"""Walle-compatible guards for strict K3 tool schemas.

Cases mirror MoonshotAI/walle's validator tests as shipped by the Kimi Vendor
Verifier (testdata/walle_validator_cases/go_validator_cases.jsonl).
"""

import pytest
import xgrammar as xgr

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.kimik3_format import (
    ARGUMENT_CLOSE,
    CALL_CLOSE,
    TOOLS_CLOSE,
    TOOLS_OPEN,
)
from sglang.srt.function_call.kimik3_structural_tag import (
    _attribute_key_regex,
    get_kimik3_structural_tag,
)
from sglang.srt.function_call.tool_schema_guard import (
    ToolSchemaError,
    canonicalize_tool_schema,
    check_tool_schema_terminates,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

_CLOSE_TOKEN = "<|close|>"
_CLOSE_TOKEN_ID = 256
_TOKEN_COMPILER = xgr.GrammarCompiler(
    xgr.TokenizerInfo(
        [bytes([token_id]) for token_id in range(256)] + [_CLOSE_TOKEN.encode()]
    ),
    cache_enabled=False,
)
_NAME = "kvv_walle_case"


def _structural_tag(parameters):
    tool = Tool(function=Function(name=_NAME, parameters=parameters, strict=True))
    return get_kimik3_structural_tag([tool], tool_choice="required")


def _tools_call(*arguments):
    body = "".join(
        f'<|open|>argument key="{key}" type="{xtml_type}"<|sep|>{value}{ARGUMENT_CLOSE}'
        for key, xtml_type, value in arguments
    )
    return (
        f'{TOOLS_OPEN}<|open|>call tool="{_NAME}" index="1"<|sep|>'
        f"{body}{CALL_CLOSE}{TOOLS_CLOSE}"
    )


def _token_accepts(structural_tag, value):
    token_ids = []
    start = 0
    while (index := value.find(_CLOSE_TOKEN, start)) != -1:
        token_ids.extend(value[start:index].encode())
        token_ids.append(_CLOSE_TOKEN_ID)
        start = index + len(_CLOSE_TOKEN)
    token_ids.extend(value[start:].encode())
    matcher = xgr.GrammarMatcher(_TOKEN_COMPILER.compile_structural_tag(structural_tag))
    return all(matcher.accept_token(t) for t in token_ids) and matcher.is_completed()


_N_SELF = {
    "type": "object",
    "properties": {"next": {"$ref": "#/$defs/N"}},
    "required": ["next"],
}


@pytest.mark.parametrize(
    "schema",
    [
        # recursion_required_self_reject: unreachable from required, still rejected
        {
            "type": "object",
            "properties": {"node": {"$ref": "#/$defs/N"}},
            "$defs": {"N": _N_SELF},
        },
        # recursion_mutual_required_reject
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
        # recursion_min_items_reject
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
        # recursion_hidden_branch_reject
        {
            "type": "object",
            "properties": {"label": {"type": "string"}, "loop": {"$ref": "#/$defs/N"}},
            "required": ["label", "loop"],
            "$defs": {"N": _N_SELF},
        },
    ],
)
def test_non_terminating_ref_is_rejected(schema):
    with pytest.raises(ToolSchemaError, match="parameters schema"):
        check_tool_schema_terminates(schema)


@pytest.mark.parametrize(
    "schema",
    [
        # recursion_optional_self
        {
            "type": "object",
            "properties": {"node": {"$ref": "#/$defs/N"}},
            "$defs": {
                "N": {"type": "object", "properties": {"next": {"$ref": "#/$defs/N"}}}
            },
        },
        # recursion_required_array_empty: the empty array terminates
        {
            "type": "object",
            "properties": {"children": {"type": "array", "items": {"$ref": "#"}}},
            "required": ["children"],
        },
        # a self-reference behind anyOf with a terminating branch
        {
            "type": "object",
            "properties": {"n": {"$ref": "#/$defs/T"}},
            "required": ["n"],
            "$defs": {
                "T": {
                    "anyOf": [
                        {"type": "null"},
                        {
                            "type": "object",
                            "properties": {"next": {"$ref": "#/$defs/T"}},
                            "required": ["next"],
                        },
                    ]
                }
            },
        },
        # an unresolvable ref is left to the other validators
        {"type": "object", "properties": {"x": {"$ref": "#/$defs/missing"}}},
    ],
)
def test_terminating_ref_is_accepted(schema):
    check_tool_schema_terminates(schema)


def test_canonicalize_matches_walle():
    assert canonicalize_tool_schema(
        {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a", "b", ""]}
    ) == {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]}
    assert canonicalize_tool_schema({"type": "string", "required": ["a"]}) == {
        "type": "string"
    }
    assert canonicalize_tool_schema({"type": "object", "required": ["a"]}) == {
        "type": "object"
    }
    assert canonicalize_tool_schema(
        {"type": "string", "minLength": 10, "maxLength": 2}
    ) == {}
    assert canonicalize_tool_schema({"type": "string", "enum": ["a", 1, "b"]}) == {
        "type": "string",
        "enum": ["a", "b"],
    }
    assert canonicalize_tool_schema({"type": ["null"], "enum": [False]}) == {
        "type": ["null"]
    }
    # Literal values are data, never rewritten.
    assert canonicalize_tool_schema(
        {"const": {"required": ["x"], "minLength": 9, "maxLength": 1}}
    ) == {"const": {"required": ["x"], "minLength": 9, "maxLength": 1}}


def test_undeclared_required_keeps_the_constraint():
    # Used to raise inside the builder; get_structure_constraint swallowed it and
    # the forced call ran unconstrained (KVV required_undeclared_canonical).
    tag = _structural_tag(
        {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "startTime": {"type": "string"},
                "endTime": {"type": "string"},
            },
            "required": ["start_time", "end_time"],
        }
    )
    assert _token_accepts(tag, _tools_call(("startTime", "string", "t0")))
    assert not _token_accepts(tag, _tools_call(("start_time", "string", "t0")))


def test_declared_key_cannot_escape_through_additional_properties():
    # KVV anyof_branch_type_conflict: `v` may only be a non-empty string.
    tag = _structural_tag(
        {
            "type": "object",
            "properties": {
                "v": {"type": "string", "anyOf": [{"minLength": 1}, {"type": "integer"}]}
            },
        }
    )
    assert _token_accepts(tag, _tools_call(("v", "string", "x")))
    assert not _token_accepts(tag, _tools_call(("v", "number", "0")))
    # Undeclared keys, including ones extending a declared name, stay open.
    assert _token_accepts(tag, _tools_call(("vv", "number", "0")))
    assert _token_accepts(tag, _tools_call(("zzz", "number", "0")))


@pytest.mark.xfail(
    strict=True,
    reason="pre-existing: a strict string value admits the close token, so it can "
    "carry further argument markup (a repeated or mistyped key) inside itself",
)
def test_string_value_cannot_smuggle_arguments():
    tag = _structural_tag(
        {
            "type": "object",
            "additionalProperties": False,
            "properties": {"v": {"type": "string"}, "n": {"type": "integer"}},
        }
    )
    assert not _token_accepts(
        tag, _tools_call(("v", "string", "x"), ("v", "number", "0"))
    )


def test_contradicting_bounds_compile_as_unconstrained():
    tag = _structural_tag(
        {
            "type": "object",
            "properties": {"a": {"type": "integer", "minimum": 10, "maximum": 2}},
            "required": ["a"],
        }
    )
    assert _token_accepts(tag, _tools_call(("a", "string", "anything")))


@pytest.mark.parametrize(
    "declared",
    [["v"], ["ab", "abc", "b"], ["a.b", "a-b", "x[0]", "^", "\\"]],
)
def test_attribute_key_regex_is_the_complement(declared):
    grammar = xgr.Grammar.from_regex(_attribute_key_regex(declared))
    from xgrammar.testing import _is_grammar_accept_string

    for name in declared:
        assert not _is_grammar_accept_string(grammar, name)
    for name in ["z", "vv", "a", "abcd", "ba", "a.bc", "x[0]]", "^^", "\\\\"]:
        if name not in declared:
            assert _is_grammar_accept_string(grammar, name), name
