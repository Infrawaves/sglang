"""Server-side stop token sequences: `Req.update_finish_state` finishes where a
sequence starts, excluding it, under single-token and speculative commits; a
sequence split across steps, or spanning output committed before the first
check, still matches; and streaming holds back a partial sequence. Client stop
strings and regexes end at the tool-call opener; server stop strings do not.
Pure CPU."""

import unittest
from array import array
from itertools import product

from sglang.srt.function_call.kimik3_format import TOOLS_CLOSE as TOOLS_CLOSE_TEXT
from sglang.srt.managers.req_types import FINISH_MATCHED_TOKEN_SEQUENCE
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.sampling.sampling_params import (
    SamplingParams,
    append_request_stop_token_sequence,
    get_request_stop_token_sequences,
    set_request_tool_calls_start,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

CLOSE, THINK, SEP, OPEN, TOOLS = 901, 910, 902, 900, 911
THINK_CLOSE = [CLOSE, THINK, SEP]
TOOLS_OPEN = [OPEN, TOOLS, SEP]
TOOLS_CLOSE = [CLOSE, TOOLS, SEP]
SHANGHAI, SHANG, HAI, EOS = 20, 21, 22, 99


class _FakeTokenizer:
    eos_token_id = -1
    additional_stop_token_ids = None
    text = {
        CLOSE: "<|close|>",
        THINK: "think",
        SEP: "<|sep|>",
        OPEN: "<|open|>",
        TOOLS: "tools",
        SHANGHAI: "Shanghai",
        SHANG: "Shang",
        HAI: "hai",
    }

    def decode(self, ids):
        return "".join(self.text.get(int(i), "x") for i in ids)

    def encode(self, text, add_special_tokens=False):
        return list(range(len(text)))


def _make_req(*sequences, require_reasoning=False, server_stops=None, **stops):
    params = {}
    for sequence in sequences:
        append_request_stop_token_sequence(params, sequence)
    # Passing server_stops also sends TOOLS_OPEN as the tool-call opener.
    if server_stops is not None:
        set_request_tool_calls_start(
            params, token_ids=TOOLS_OPEN, server_stop_strs=server_stops
        )
    sp = SamplingParams(
        max_new_tokens=1000, custom_params=params.get("custom_params"), **stops
    )
    sp.normalize(tokenizer=_FakeTokenizer())
    req = Req(
        rid="t",
        origin_input_text="",
        origin_input_ids=array("q", [0]),
        sampling_params=sp,
        require_reasoning=require_reasoning,
        eos_token_ids={EOS},
        vocab_size=10_000,
    )
    req.tokenizer = _FakeTokenizer()
    req.output_ids = array("q", [])
    return req


def _commit(req, tokens):
    req.output_ids.extend(tokens)
    if req.require_reasoning:
        req.update_reasoning_tokens(list(tokens), THINK_CLOSE)
    req.update_finish_state(new_accepted_len=len(tokens))


class TestStopTokenSequence(unittest.TestCase):
    def test_transport_round_trip(self):
        params = {}
        append_request_stop_token_sequence(params, THINK_CLOSE)
        append_request_stop_token_sequence(params, TOOLS_OPEN)
        append_request_stop_token_sequence(params, THINK_CLOSE)
        self.assertEqual(
            get_request_stop_token_sequences(params["custom_params"]),
            [tuple(THINK_CLOSE), tuple(TOOLS_OPEN)],
        )

    def test_single_token_commits_exclude_the_sequence(self):
        req = _make_req(THINK_CLOSE)
        for token in [10, 11, CLOSE, THINK]:
            _commit(req, [token])
            self.assertFalse(req.finished())
        _commit(req, [SEP])
        self.assertIsInstance(req.finished_reason, FINISH_MATCHED_TOKEN_SEQUENCE)
        self.assertEqual(req.finished_reason.matched, "<|close|>think<|sep|>")
        self.assertEqual(list(req.output_ids_through_stop), [10, 11])
        self.assertTrue(req.finished_reason.to_json()["stop_excluded"])

    def test_speculative_commit_drops_tokens_after_the_sequence(self):
        req = _make_req(THINK_CLOSE, TOOLS_OPEN)
        _commit(req, [10, OPEN, TOOLS, SEP, 12, 13])
        self.assertEqual(list(req.output_ids_through_stop), [10])

    def test_broken_sequence_does_not_stop(self):
        req = _make_req(THINK_CLOSE)
        _commit(req, [CLOSE, 20, SEP, CLOSE, THINK, 21])
        self.assertFalse(req.finished())

    def test_output_before_the_first_check_counts(self):
        req = _make_req(THINK_CLOSE)
        req.output_ids.extend([10, CLOSE])  # e.g. the prefill's first token in PD
        _commit(req, [THINK, SEP])
        self.assertEqual(list(req.output_ids_through_stop), [10])

    def test_streaming_holds_back_a_partial_sequence(self):
        req = _make_req(TOOLS_OPEN)
        _commit(req, [10])
        self.assertFalse(req.is_matching_stop_token_sequence())
        _commit(req, [OPEN])
        self.assertTrue(req.is_matching_stop_token_sequence())
        _commit(req, [20])
        self.assertFalse(req.is_matching_stop_token_sequence())

    def test_request_without_sequences(self):
        req = _make_req()
        _commit(req, [CLOSE, THINK, SEP])
        self.assertFalse(req.finished())
        self.assertFalse(req.is_matching_stop_token_sequence())


class TestClientStopsEndAtToolCalls(unittest.TestCase):
    def test_stop_in_tool_call_arguments_keeps_the_call(self):
        """A client stop in a tool call's arguments must neither end the request
        mid-call nor hold the arguments back, with thinking on or off."""
        for require_reasoning, stops in product(
            (True, False), ({"stop": "Shanghai"}, {"stop_regex": "Shang.ai"})
        ):
            with self.subTest(require_reasoning=require_reasoning, stops=stops):
                req = _make_req(
                    require_reasoning=require_reasoning, server_stops=[], **stops
                )
                reasoning = [10, *THINK_CLOSE] if require_reasoning else []
                call = [*TOOLS_OPEN, 12, SHANG, HAI, *TOOLS_CLOSE]
                for token in [*reasoning, 11, *call]:
                    _commit(req, [token])
                    self.assertFalse(req.finished())
                    self.assertFalse(req.check_match_stop_str_prefix())
                _commit(req, [EOS])
                self.assertEqual(
                    req.finished_reason.to_json(), {"type": "stop", "matched": EOS}
                )

    def test_speculative_run_checks_only_the_content_before_the_call(self):
        """A run accepting content and a tool call together stops at a client
        stop in the content and ignores one in the arguments."""
        req = _make_req(server_stops=[], stop="Shanghai")
        _commit(req, [10, SHANGHAI, 11, *TOOLS_OPEN, SHANGHAI])
        self.assertEqual(
            req.finished_reason.to_json(), {"type": "stop", "matched": "Shanghai"}
        )
        self.assertEqual(list(req.output_ids_through_stop), [10, SHANGHAI])

        req = _make_req(server_stops=[], stop="Shanghai")
        _commit(req, [10, 11, *TOOLS_OPEN, SHANGHAI, 12])
        self.assertFalse(req.finished())

    def test_server_stop_still_ends_the_call(self):
        """A server stop such as the tools close still ends the request inside
        tool calls, and streaming holds back its partial marker, while a client
        stop in the arguments does not end it."""
        req = _make_req(
            server_stops=[TOOLS_CLOSE_TEXT], stop=["Shanghai", TOOLS_CLOSE_TEXT]
        )
        for token in [11, *TOOLS_OPEN, 12, SHANG, HAI]:
            _commit(req, [token])
            self.assertFalse(req.finished())
        _commit(req, [CLOSE])
        self.assertTrue(req.check_match_stop_str_prefix())
        _commit(req, [TOOLS, SEP])
        self.assertEqual(
            req.finished_reason.to_json(),
            {"type": "stop", "matched": TOOLS_CLOSE_TEXT},
        )
        self.assertEqual(
            list(req.output_ids_through_stop),
            [11, *TOOLS_OPEN, 12, SHANG, HAI, *TOOLS_CLOSE],
        )

    def test_opener_inside_reasoning_keeps_client_stops_on_the_answer(self):
        """An opener written inside reasoning must not disable client stops on
        the answer after reasoning."""
        req = _make_req(require_reasoning=True, server_stops=[], stop="Shanghai")
        for token in [10, *TOOLS_OPEN, 11, *THINK_CLOSE, 12]:
            _commit(req, [token])
            self.assertFalse(req.finished())
        _commit(req, [SHANGHAI])
        self.assertEqual(
            req.finished_reason.to_json(), {"type": "stop", "matched": "Shanghai"}
        )


if __name__ == "__main__":
    unittest.main()
