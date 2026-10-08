import json
import logging
import re
from typing import List, Literal, Optional, Union

from xgrammar import StructuralTag

from sglang.srt.entrypoints.openai.protocol import AllowedToolChoice, Tool, ToolChoice
from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.core_types import (
    StreamingParseResult,
    ToolCallItem,
    _GetInfoFunc,
)
from sglang.srt.function_call.kimik3_format import (
    ARGUMENT_CLOSE,
    CALL_CLOSE,
    MESSAGE_CLOSE,
    RESPONSE_CLOSE,
    RESPONSE_OPEN,
    TOOLS_CLOSE,
    TOOLS_OPEN,
    partial_suffix_len,
    strip_partial_marker_suffix,
    strip_response_wrappers,
)
from sglang.srt.function_call.kimik3_structural_tag import (
    get_kimik3_auto_tool_call_structural_tag,
    get_kimik3_structural_tag,
)

logger = logging.getLogger(__name__)

_CALL_RE = re.compile(
    r"<\|open\|>call\s+(?P<attrs>(?:(?!<\|sep\|>).)*?)<\|sep\|>"
    r"(?P<body>.*?)<\|close\|>call<\|sep\|>",
    re.DOTALL,
)
_ARG_RE = re.compile(
    r"<\|open\|>argument\s+(?P<attrs>(?:(?!<\|sep\|>).)*?)<\|sep\|>"
    r"(?P<val>.*?)<\|close\|>argument<\|sep\|>",
    re.DOTALL,
)
_CALL_HEADER_RE = re.compile(
    r"<\|open\|>call\s+(?P<attrs>(?:(?!<\|sep\|>).)*?)<\|sep\|>", re.DOTALL
)
_ARG_HEADER_RE = re.compile(
    r"<\|open\|>argument\s+(?P<attrs>(?:(?!<\|sep\|>).)*?)<\|sep\|>", re.DOTALL
)
_ATTR_RE = re.compile(r'(?P<k>\w+)="(?P<v>[^"]*)"')


def _unescape_attr(value: str) -> str:
    return value.replace("&quot;", '"').replace("&amp;", "&")


def _parse_attrs(attrs: str) -> dict:
    return {m["k"]: _unescape_attr(m["v"]) for m in _ATTR_RE.finditer(attrs)}


class KimiK3Detector(BaseFormatDetector):
    """Detector for the Kimi K3 XTML tool-call format.

    K3 emits tool calls in a ``tools`` channel built from dedicated special
    tokens; the plain reply lives in a preceding ``response`` channel:

    ```
    <|open|>response<|sep|>text<|close|>response<|sep|>
    <|open|>tools<|sep|>
      <|open|>call tool="name" index="1"<|sep|>
        <|open|>argument key="k" type="string"<|sep|>raw text<|close|>argument<|sep|>
      <|close|>call<|sep|>
    <|close|>tools<|sep|>
    ```

    ``type="string"`` argument values are raw text; other types are
    JSON-decoded. Attribute values reverse the template's ``&amp;``/``&quot;``
    escaping.
    """

    def __init__(self):
        super().__init__()
        self.bot_token = TOOLS_OPEN
        self.eot_token = TOOLS_CLOSE
        self._cursor = 0
        self._call_start: int | None = None
        self._argument_is_string: bool | None = None

    def has_tool_call(self, text: str) -> bool:
        return self.bot_token in text

    def supports_structural_tag(self) -> bool:
        return True

    def parses_required_natively(self) -> bool:
        return False

    def structure_info(self) -> _GetInfoFunc:
        raise NotImplementedError(
            "Kimi K3 uses its model-native structural tag implementation"
        )

    def get_auto_tool_call_structural_tag(
        self,
        tools: Union[List[Tool], None] = None,
        thinking_mode: bool = False,
        parallel_tool_calls: bool = True,
    ) -> Optional[StructuralTag]:
        return get_kimik3_auto_tool_call_structural_tag(
            tools or [],
            thinking_mode=thinking_mode,
            parallel_tool_calls=parallel_tool_calls,
        )

    def get_structural_tag(
        self,
        tools: Union[List[Tool], None] = None,
        tool_choice: Union[
            ToolChoice, AllowedToolChoice, Literal["auto", "required"]
        ] = "auto",
        thinking_mode: bool = False,
        parallel_tool_calls: bool = True,
    ) -> StructuralTag:
        return get_kimik3_structural_tag(
            tools=tools or [],
            tool_choice=tool_choice,
            thinking_mode=thinking_mode,
            parallel_tool_calls=parallel_tool_calls,
        )

    def _decode_call(self, attrs: str, body: str) -> dict | None:
        call_attrs = _parse_attrs(attrs)
        tool_name = call_attrs.get("tool", "")
        if not tool_name:
            return None
        arguments = {}
        for arg in _ARG_RE.finditer(body):
            arg_attrs = _parse_attrs(arg["attrs"])
            key = arg_attrs.get("key", "")
            arg_type = arg_attrs.get("type", "string")
            raw_value = arg["val"]
            if arg_type == "string":
                arguments[key] = raw_value
            else:
                try:
                    arguments[key] = json.loads(raw_value)
                except json.JSONDecodeError:
                    arguments[key] = raw_value
        return {
            "name": tool_name,
            "arguments": json.dumps(arguments, ensure_ascii=False),
        }

    def _parse_calls(self, section: str) -> List[dict]:
        return [
            call
            for m in _CALL_RE.finditer(section)
            if (call := self._decode_call(m["attrs"], m["body"])) is not None
        ]

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        open_idx = text.find(self.bot_token)
        if open_idx == -1:
            return StreamingParseResult(normal_text=strip_response_wrappers(text))
        # Computed outside the try so the error path can reuse it instead of
        # falling back to raw text, which would ship the XTML tools markup to
        # the client.
        before = strip_response_wrappers(text[:open_idx])
        try:
            section_start = open_idx + len(self.bot_token)
            close_idx = text.find(self.eot_token, section_start)
            section = (
                text[section_start:]
                if close_idx == -1
                else text[section_start:close_idx]
            )
            calls = [
                ToolCallItem(
                    tool_index=i,
                    name=call["name"],
                    parameters=call["arguments"],
                )
                for i, call in enumerate(self._parse_calls(section))
            ]
            return StreamingParseResult(normal_text=before, calls=calls)
        except Exception as e:
            logger.error("Error in Kimi K3 detect_and_parse: %s", e, exc_info=True)
            return StreamingParseResult(normal_text=before)

    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        self._buffer += new_text
        try:
            open_idx = self._buffer.find(self.bot_token)
            if open_idx == -1:
                return StreamingParseResult(normal_text=self._emit_normal_text())

            normal_text = self._emit_normal_text(limit=open_idx)
            self._cursor = max(self._cursor, open_idx + len(self.bot_token))
            return StreamingParseResult(
                normal_text=normal_text, calls=self._stream_calls()
            )
        except Exception as e:
            logger.error(
                "Error in Kimi K3 parse_streaming_increment: %s", e, exc_info=True
            )
            # _cursor indexes into _buffer, so it must be reset with it;
            # otherwise every later _emit_normal_text sees limit <= _cursor
            # and silently drops the rest of the response.
            self._buffer = ""
            self._cursor = 0
            self._call_start = None
            self._argument_is_string = None
            return StreamingParseResult()

    def _append_stream_call(
        self, calls: List[ToolCallItem], parameters: str, *, name: str | None = None
    ) -> None:
        self.streamed_args_for_tool[self.current_tool_id] += parameters
        if calls and calls[-1].tool_index == self.current_tool_id:
            calls[-1].parameters += parameters
        else:
            calls.append(
                ToolCallItem(
                    tool_index=self.current_tool_id, name=name, parameters=parameters
                )
            )

    def _stream_calls(self) -> List[ToolCallItem]:
        calls = []
        while True:
            if self._call_start is None:
                header = _CALL_HEADER_RE.search(self._buffer, self._cursor)
                if header is None:
                    break
                name = _parse_attrs(header["attrs"]).get("tool", "")
                if not name:
                    end = self._buffer.find(CALL_CLOSE, header.end())
                    if end == -1:
                        break
                    self._cursor = end + len(CALL_CLOSE)
                    continue
                self._call_start = header.start()
                self._cursor = header.end()
                self.current_tool_id += 1
                self.streamed_args_for_tool.append("")
                self._append_stream_call(calls, "{", name=name)

            call_end = self._buffer.find(CALL_CLOSE, self._cursor)
            if self._argument_is_string is not None:
                if self._stream_argument(calls, call_end=call_end):
                    continue
                if call_end == -1:
                    break

            if self._argument_is_string is None:
                header = _ARG_HEADER_RE.search(
                    self._buffer,
                    self._cursor,
                    call_end if call_end != -1 else len(self._buffer),
                )
                if header is not None:
                    attrs = _parse_attrs(header["attrs"])
                    self._argument_is_string = attrs.get("type", "string") == "string"
                    prefix = (
                        ""
                        if self.streamed_args_for_tool[self.current_tool_id] == "{"
                        else ", "
                    )
                    prefix += (
                        json.dumps(attrs.get("key", ""), ensure_ascii=False) + ": "
                    )
                    if self._argument_is_string:
                        prefix += '"'
                    self._append_stream_call(calls, prefix)
                    self._cursor = header.end()
                    continue
                if call_end == -1:
                    break
                self._append_stream_call(calls, "}")

            # Keep call-close markers visible even inside malformed loose arguments.
            match = _CALL_RE.match(self._buffer, self._call_start)
            call = self._decode_call(match["attrs"], match["body"])
            self.prev_tool_call_arr.append(
                {"name": call["name"], "arguments": json.loads(call["arguments"])}
            )
            self._cursor = call_end + len(CALL_CLOSE)
            self._call_start = None
            self._argument_is_string = None
        return calls

    def _stream_argument(self, calls: List[ToolCallItem], *, call_end: int) -> bool:
        end = self._buffer.find(ARGUMENT_CLOSE, self._cursor)
        complete = end != -1 and (call_end == -1 or end < call_end)
        if not complete:
            if not self._argument_is_string:
                return False
            end = call_end
            if end == -1:
                end = len(self._buffer) - partial_suffix_len(
                    self._buffer[self._cursor :], [ARGUMENT_CLOSE, CALL_CLOSE]
                )

        value = self._buffer[self._cursor : end]
        if self._argument_is_string:
            # Escape only new raw text; a JSON string's escaped prefix is stable.
            delta = json.dumps(value, ensure_ascii=False)[1:-1]
            if complete:
                delta += '"'
        else:
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                pass
            delta = json.dumps(value, ensure_ascii=False)
        if delta:
            self._append_stream_call(calls, delta)
        self._cursor = end
        if complete:
            self._cursor += len(ARGUMENT_CLOSE)
            self._argument_is_string = None
        return complete

    def finish(self, tools: List[Tool]) -> StreamingParseResult:
        open_idx = self._buffer.find(self.bot_token)
        if open_idx != -1:
            section = self._buffer[open_idx + len(self.bot_token) :]
            if not self._parse_calls(section):
                logger.warning(
                    "Kimi K3 tools section ended with no complete tool call "
                    "(%d buffered chars)",
                    len(section),
                )
            return StreamingParseResult()
        pending = self._emit_normal_text(limit=len(self._buffer))
        return StreamingParseResult(normal_text=strip_partial_marker_suffix(pending))

    def _emit_normal_text(self, limit: int | None = None) -> str:
        if limit is None:
            holdback = partial_suffix_len(
                self._buffer,
                [self.bot_token, RESPONSE_OPEN, RESPONSE_CLOSE, MESSAGE_CLOSE],
            )
            limit = len(self._buffer) - holdback
        if limit <= self._cursor:
            return ""
        pending = self._buffer[self._cursor : limit]
        for marker in (RESPONSE_OPEN, RESPONSE_CLOSE, MESSAGE_CLOSE):
            if marker in pending:
                pending = pending.replace(marker, "")
        self._cursor = limit
        return pending
