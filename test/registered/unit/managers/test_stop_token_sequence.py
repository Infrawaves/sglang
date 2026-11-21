"""Server-side stop token sequences: `Req.update_finish_state` finishes where a
sequence starts, excluding it, under single-token and speculative commits; a
sequence split across steps, or spanning output committed before the first
check, still matches; and streaming holds back a partial sequence. Pure CPU."""

import unittest
from array import array

from sglang.srt.managers.req_types import FINISH_MATCHED_TOKEN_SEQUENCE
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.sampling.sampling_params import (
    SamplingParams,
    append_request_stop_token_sequence,
    get_request_stop_token_sequences,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

CLOSE, THINK, SEP, OPEN, TOOLS = 901, 910, 902, 900, 911
THINK_CLOSE = [CLOSE, THINK, SEP]
TOOLS_OPEN = [OPEN, TOOLS, SEP]


class _FakeTokenizer:
    eos_token_id = -1
    additional_stop_token_ids = None
    text = {
        CLOSE: "<|close|>",
        THINK: "think",
        SEP: "<|sep|>",
        OPEN: "<|open|>",
        TOOLS: "tools",
    }

    def decode(self, ids):
        return "".join(self.text.get(int(i), "x") for i in ids)


def _make_req(*sequences):
    params = {}
    for sequence in sequences:
        append_request_stop_token_sequence(params, sequence)
    sp = SamplingParams(max_new_tokens=1000, custom_params=params.get("custom_params"))
    sp.normalize(tokenizer=None)
    req = Req(
        rid="t",
        origin_input_text="",
        origin_input_ids=array("q", [0]),
        sampling_params=sp,
        vocab_size=10_000,
    )
    req.tokenizer = _FakeTokenizer()
    req.output_ids = array("q", [])
    return req


def _commit(req, tokens):
    req.output_ids.extend(tokens)
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


if __name__ == "__main__":
    unittest.main()
