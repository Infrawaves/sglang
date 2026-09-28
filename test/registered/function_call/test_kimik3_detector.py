import json
import sys

import pytest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.core_types import ToolCallItem
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.function_call.kimik3_detector import KimiK3Detector
from sglang.srt.function_call.kimik3_format import (
    ARGUMENT_CLOSE,
    CALL_CLOSE,
    MESSAGE_CLOSE,
    RESPONSE_CLOSE,
    RESPONSE_OPEN,
    TOOLS_CLOSE,
    TOOLS_OPEN,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=7, suite="base-a-test-cpu")


def _make_tool(name: str) -> Tool:
    return Tool(
        type="function",
        function=Function(
            name=name,
            description=f"{name} tool",
            parameters={
                "type": "object",
                "properties": {"code": {"type": "string"}},
            },
        ),
    )


def _call_block(tool: str, index: int, args: dict[str, tuple[str, str]]) -> str:
    parts = [f'<|open|>call tool="{tool}" index="{index}"<|sep|>']
    for key, (arg_type, value) in args.items():
        parts.append(
            f'<|open|>argument key="{key}" type="{arg_type}"<|sep|>'
            f"{value}<|close|>argument<|sep|>"
        )
    parts.append("<|close|>call<|sep|>")
    return "".join(parts)


def _chunks(text: str, size: int) -> list[str]:
    return [text[index : index + size] for index in range(0, len(text), size)]


def _stream(
    detector: KimiK3Detector, chunks: list[str], tools: list[Tool]
) -> tuple[str, list[ToolCallItem]]:
    text = ""
    calls = []
    for chunk in chunks:
        result = detector.parse_streaming_increment(chunk, tools)
        text += result.normal_text
        for call in result.calls:
            if call.name is not None:
                assert call.tool_index == len(calls)
                calls.append(call.model_copy())
            else:
                calls[call.tool_index].parameters += call.parameters
    return text, calls


def test_detect_and_parse_single_call() -> None:
    detector = KimiK3Detector()
    tools = [_make_tool("python")]
    text = (
        f"{RESPONSE_OPEN}Let me run it.{RESPONSE_CLOSE}{TOOLS_OPEN}"
        + _call_block(
            "python",
            1,
            {"code": ("string", "print(1)"), "opts": ("object", '{"a": 1}')},
        )
        + TOOLS_CLOSE
    )
    result = detector.detect_and_parse(text, tools)
    assert result.normal_text == "Let me run it."
    assert len(result.calls) == 1
    assert result.calls[0].name == "python"
    assert json.loads(result.calls[0].parameters) == {
        "code": "print(1)",
        "opts": {"a": 1},
    }


def test_detect_and_parse_no_tools_channel() -> None:
    detector = KimiK3Detector()
    result = detector.detect_and_parse(
        f"{RESPONSE_OPEN}hi there{RESPONSE_CLOSE}{MESSAGE_CLOSE}",
        [_make_tool("python")],
    )
    assert result.normal_text == "hi there"
    assert result.calls == []


def test_detect_and_parse_multiple_calls() -> None:
    detector = KimiK3Detector()
    text = (
        TOOLS_OPEN
        + _call_block("python", 1, {"code": ("string", "a")})
        + _call_block("python", 2, {"code": ("string", "b")})
        + TOOLS_CLOSE
    )
    result = detector.detect_and_parse(text, [_make_tool("python")])
    assert [call.tool_index for call in result.calls] == [0, 1]
    assert json.loads(result.calls[1].parameters) == {"code": "b"}


def test_detect_and_parse_unclosed_tools_section() -> None:
    detector = KimiK3Detector()
    text = TOOLS_OPEN + _call_block("python", 1, {"code": ("string", "x")})
    result = detector.detect_and_parse(text, [_make_tool("python")])
    assert len(result.calls) == 1
    assert json.loads(result.calls[0].parameters) == {"code": "x"}


def test_attr_unescaping_and_raw_string_args() -> None:
    detector = KimiK3Detector()
    text = (
        f"{TOOLS_OPEN}"
        '<|open|>call tool="a&amp;b" index="1"<|sep|>'
        '<|open|>argument key="q" type="string"<|sep|>'
        "say &quot;hi&quot;<|close|>argument<|sep|>"
        "<|close|>call<|sep|>"
        f"{TOOLS_CLOSE}"
    )
    result = detector.detect_and_parse(text, [_make_tool("python")])
    assert result.calls[0].name == "a&b"
    assert json.loads(result.calls[0].parameters) == {"q": "say &quot;hi&quot;"}


def test_non_string_arg_json_decoding() -> None:
    detector = KimiK3Detector()
    text = (
        TOOLS_OPEN
        + _call_block(
            "python",
            1,
            {
                "n": ("number", "42"),
                "flag": ("boolean", "true"),
                "bad": ("object", "{not json"),
            },
        )
        + TOOLS_CLOSE
    )
    result = detector.detect_and_parse(text, [_make_tool("python")])
    assert json.loads(result.calls[0].parameters) == {
        "n": 42,
        "flag": True,
        "bad": "{not json",
    }


@pytest.mark.parametrize("chunk_size", [1, 7, 23, 4096])
def test_streaming_split_markers(chunk_size: int) -> None:
    """Chunking must not change escaping, value types, or literal marker prefixes."""
    detector = KimiK3Detector()
    tools = [_make_tool("python")]
    code = 'print("雪")\\path\n\r\t\b\f\x00\x1f<|close|>argumenX &quot;'
    text = (
        f"{RESPONSE_OPEN}Hello!{RESPONSE_CLOSE}{TOOLS_OPEN}"
        + _call_block(
            "python",
            27,
            {
                "code": ("string", code),
                "empty": ("string", ""),
                "literal": ("string", "null"),
                "q&amp;&quot;": ("string", "raw &amp;"),
                "number": ("number", "1e2"),
                "null": ("null", "null"),
                "flag": ("boolean", "false"),
                "opts": ("object", '{"a":[true,null,"x"]}'),
                "array": ("array", '[1,"two"]'),
                "bad": ("number", "1e"),
            },
        )
        + TOOLS_CLOSE
    )
    normal_text, calls = _stream(detector, _chunks(text, chunk_size), tools)
    assert normal_text == "Hello!"
    assert len(calls) == 1
    assert calls[0].name == "python"
    expected = {
        "code": code,
        "empty": "",
        "literal": "null",
        'q&"': "raw &amp;",
        "number": 100.0,
        "null": None,
        "flag": False,
        "opts": {"a": [True, None, "x"]},
        "array": [1, "two"],
        "bad": "1e",
    }
    assert json.loads(calls[0].parameters) == expected
    assert detector.detect_and_parse(text, tools).calls == calls


@pytest.mark.parametrize("chunk_size", [7, 4096])
def test_streaming_multiple_calls(chunk_size: int) -> None:
    detector = KimiK3Detector()
    tools = [_make_tool("python"), _make_tool("finish")]
    text = (
        TOOLS_OPEN
        + _call_block("python", 27, {"code": ("string", "a")})
        + _call_block("python", 28, {"code": ("string", "b")})
        + _call_block("finish", 29, {})
        + TOOLS_CLOSE
    )
    _, calls = _stream(detector, _chunks(text, chunk_size), tools)
    assert [call.tool_index for call in calls] == [0, 1, 2]
    assert [call.name for call in calls] == ["python", "python", "finish"]
    assert [json.loads(call.parameters) for call in calls] == [
        {"code": "a"},
        {"code": "b"},
        {},
    ]


def test_streaming_emits_name_and_string_before_closing_markers() -> None:
    """Long tool arguments must reach clients while the model is still writing them."""
    parser = FunctionCallParser([_make_tool("python")], "kimi_k3")
    _, calls = parser.parse_stream_chunk(
        TOOLS_OPEN + '<|open|>call tool="python" index="27"<|sep|>'
    )
    assert [(call.name, call.tool_index) for call in calls] == [("python", 0)]
    _, header = parser.parse_stream_chunk(
        '<|open|>argument key="code" type="string"<|sep|>'
    )
    calls.extend(header)
    value = 'line("雪")\\\n' * 128
    for part in _chunks(value, 16):
        normal_text, deltas = parser.parse_stream_chunk(part)
        assert normal_text == ""
        assert deltas
        assert all(call.name is None and call.tool_index == 0 for call in deltas)
        calls.extend(deltas)
    assert "".join(call.parameters for call in calls) == (
        '{"code": ' + json.dumps(value, ensure_ascii=False)[:-1]
    )
    _, end = parser.parse_stream_chunk(ARGUMENT_CLOSE + CALL_CLOSE + TOOLS_CLOSE)
    calls.extend(end)
    assert json.loads("".join(call.parameters for call in calls)) == {"code": value}
    assert parser.parse_stream_end() == ("", [])


@pytest.mark.parametrize(
    "arg_type, parts, expected",
    [
        ("number", ["1", "e", "2"], 100.0),
        ("null", ["n", "ul", "l"], None),
        ("boolean", ["tr", "ue"], True),
        ("array", ["[1", ",2]"], [1, 2]),
        ("object", ['{"a":', "false}"], {"a": False}),
        ("number", ["1", "e"], "1e"),
    ],
)
def test_streaming_non_string_waits_for_complete_value(arg_type, parts, expected):
    """Partial numbers/JSON cannot be emitted before type conversion or fallback."""
    parser = FunctionCallParser([_make_tool("python")], "kimi_k3")
    _, calls = parser.parse_stream_chunk(
        TOOLS_OPEN + '<|open|>call tool="python" index="1"<|sep|>'
        f'<|open|>argument key="value" type="{arg_type}"<|sep|>'
    )
    for part in parts:
        assert parser.parse_stream_chunk(part) == ("", [])
    _, end = parser.parse_stream_chunk(ARGUMENT_CLOSE + CALL_CLOSE)
    calls.extend(end)
    assert json.loads("".join(call.parameters for call in calls)) == {"value": expected}


def test_streaming_plain_text_only() -> None:
    detector = KimiK3Detector()
    text, calls = _stream(
        detector, ["just a ", "plain ", "reply"], [_make_tool("python")]
    )
    assert text == "just a plain reply"
    assert calls == []


def test_streaming_bookkeeping_for_serving_layer() -> None:
    detector = KimiK3Detector()
    tools = [_make_tool("python")]
    text = (
        TOOLS_OPEN + _call_block("python", 1, {"code": ("string", "a")}) + TOOLS_CLOSE
    )
    _stream(detector, _chunks(text, 9), tools)
    assert detector.current_tool_id == 0
    assert detector.prev_tool_call_arr[0] == {
        "name": "python",
        "arguments": {"code": "a"},
    }
    assert json.loads(detector.streamed_args_for_tool[0]) == {"code": "a"}


@pytest.mark.parametrize(
    "argument", ["", '<|open|>argument key="code" type="string"<|sep|>partial<|cl']
)
def test_stream_end_reports_truncated_tools_section(caplog, argument) -> None:
    """Truncated calls are reported without fabricating closing JSON or leaking markers."""
    detector = KimiK3Detector()
    tools = [_make_tool("python")]
    truncated = TOOLS_OPEN + '<|open|>call tool="python" index="1"<|sep|>' + argument
    text, calls = _stream(detector, _chunks(truncated, 7), tools)
    assert text == ""
    assert calls[0].name == "python"
    assert calls[0].parameters == ('{"code": "partial' if argument else "{")
    assert detector.prev_tool_call_arr == []
    with caplog.at_level("WARNING", logger="sglang.srt.function_call.kimik3_detector"):
        result = detector.finish(tools)
    assert result.calls == []
    assert TOOLS_OPEN not in (result.normal_text or "")
    assert "no complete tool call" in caplog.text


def test_stream_end_releases_held_back_text() -> None:
    detector = KimiK3Detector()
    tools = [_make_tool("python")]
    text, _ = _stream(detector, ["all done", "<"], tools)
    assert text == "all done"
    result = detector.finish(tools)
    assert text + (result.normal_text or "") == "all done<"


def test_stream_end_drops_truncated_marker() -> None:
    detector = KimiK3Detector()
    tools = [_make_tool("python")]
    text, _ = _stream(detector, ["all done", "<|open|>"], tools)
    result = detector.finish(tools)
    assert text + (result.normal_text or "") == "all done"


def test_detector_capabilities_and_registration() -> None:
    detector = KimiK3Detector()
    assert detector.supports_structural_tag()
    assert not detector.parses_required_natively()
    parser = FunctionCallParser([_make_tool("python")], "kimi_k3")
    assert isinstance(parser.detector, KimiK3Detector)
    assert parser.get_structure_constraint("required") is not None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
