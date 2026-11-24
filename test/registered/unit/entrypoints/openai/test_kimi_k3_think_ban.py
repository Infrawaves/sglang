"""Kimi K3 with thinking off: the serving layer attaches the conditional token
ban ``(think, (<|open|>, <|close|>))`` to the request, and only then. Pure CPU."""

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

maybe_stub_sgl_kernel()  # must precede any import that pulls in sgl_kernel

import unittest
from types import SimpleNamespace

from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
from sglang.srt.environ import envs
from sglang.srt.sampling.sampling_params import (
    REQUEST_CONDITIONAL_TOKEN_BAN_KEY,
    get_request_conditional_token_ban,
    get_request_stop_token_sequences,
)

OPEN, CLOSE, SEP, THINK, RESPONSE = 900, 901, 902, 910, 911
THINKING_OFF = {"thinking": False}


class _FakeTokenizer:
    """Encodes K3 markers as a special token, one name token and <|sep|>."""

    special = {"<|open|>": OPEN, "<|close|>": CLOSE, "<|sep|>": SEP}
    names = {"think": [THINK], "response": [RESPONSE]}

    def __init__(self):
        self.encode_calls = 0

    def encode(self, text, add_special_tokens=False):
        self.encode_calls += 1
        head, rest = text[: text.index("|>") + 2], text[text.index("|>") + 2 :]
        name, tail = rest[: rest.index("<|")], rest[rest.index("<|") :]
        return [self.special[head], *self.names[name], self.special[tail]]

    def convert_tokens_to_ids(self, token):
        return self.special[token]


class _SplitNameTokenizer(_FakeTokenizer):
    """Spells the channel name with two tokens: still a special-token marker a
    stop can match, but with no single name token to mask."""

    names = {"think": [THINK, THINK + 1], "response": [RESPONSE]}


class _PlainTextTokenizer(_FakeTokenizer):
    """Spells the markers with ordinary tokens, as a tokenizer without them would."""

    def encode(self, text, add_special_tokens=False):
        self.encode_calls += 1
        return list(range(len(text)))


def _serving(tokenizer):
    serving = object.__new__(OpenAIServingChat)
    serving.tokenizer_manager = SimpleNamespace(tokenizer=tokenizer)
    serving._kimi_k3_marker_ids = {}
    return serving


class TestKimiK3ThinkBan(CustomTestCase):
    def test_thinking_off_attaches_the_rule(self):
        serving = _serving(_FakeTokenizer())
        sampling_params = {}
        serving._ban_kimi_k3_think_channel_without_thinking(
            sampling_params, THINKING_OFF
        )
        self.assertEqual(
            get_request_conditional_token_ban(sampling_params["custom_params"]),
            (THINK, (OPEN, CLOSE)),
        )

    def test_thinking_on_or_unset_attaches_nothing(self):
        serving = _serving(_FakeTokenizer())
        for kwargs in (None, {}, {"thinking": True}, {"thinking_effort": "low"}):
            with self.subTest(kwargs=kwargs):
                sampling_params = {}
                serving._ban_kimi_k3_think_channel_without_thinking(
                    sampling_params, kwargs
                )
                self.assertEqual(sampling_params, {})
        # The markers are not even encoded for these requests.
        self.assertEqual(serving.tokenizer_manager.tokenizer.encode_calls, 0)

    def test_a_client_cannot_supply_the_rule(self):
        serving = _serving(_FakeTokenizer())
        sampling_params = {
            "custom_params": {REQUEST_CONDITIONAL_TOKEN_BAN_KEY: [1, 2, 3], "mine": 1}
        }
        serving._ban_kimi_k3_think_channel_without_thinking(
            sampling_params, {"thinking": True}
        )
        self.assertEqual(sampling_params["custom_params"], {"mine": 1})

        sampling_params["custom_params"][REQUEST_CONDITIONAL_TOKEN_BAN_KEY] = [1, 2]
        serving._ban_kimi_k3_think_channel_without_thinking(
            sampling_params, THINKING_OFF
        )
        self.assertEqual(
            sampling_params["custom_params"],
            {"mine": 1, REQUEST_CONDITIONAL_TOKEN_BAN_KEY: [THINK, OPEN, CLOSE]},
        )

    def test_the_markers_are_encoded_once_per_server(self):
        tokenizer = _FakeTokenizer()
        serving = _serving(tokenizer)
        for _ in range(3):
            serving._ban_kimi_k3_think_channel_without_thinking({}, THINKING_OFF)
        self.assertEqual(tokenizer.encode_calls, 2)  # think open and think close

    def test_the_think_close_stop_stays_as_the_fallback(self):
        tokenizer = _FakeTokenizer()
        serving = _serving(tokenizer)
        sampling_params = {}
        serving._stop_kimi_k3_reasoning_without_thinking(sampling_params, THINKING_OFF)
        serving._ban_kimi_k3_think_channel_without_thinking(
            sampling_params, THINKING_OFF
        )
        custom_params = sampling_params["custom_params"]
        self.assertEqual(
            get_request_stop_token_sequences(custom_params), [(CLOSE, THINK, SEP)]
        )
        self.assertEqual(
            get_request_conditional_token_ban(custom_params), (THINK, (OPEN, CLOSE))
        )
        # Both read the same cached marker ids.
        self.assertEqual(tokenizer.encode_calls, 2)

    def test_a_tokenizer_without_the_special_tokens_disables_the_ban(self):
        serving = _serving(_PlainTextTokenizer())
        sampling_params = {}
        with self.assertLogs(
            "sglang.srt.entrypoints.openai.serving_chat", level="WARNING"
        ):
            serving._ban_kimi_k3_think_channel_without_thinking(
                sampling_params, THINKING_OFF
            )
        self.assertEqual(sampling_params, {})
        self.assertIsNone(serving._kimi_k3_think_ban_rule())

    def test_a_name_of_two_tokens_disables_the_ban_only(self):
        serving = _serving(_SplitNameTokenizer())
        sampling_params = {}
        with self.assertLogs(
            "sglang.srt.entrypoints.openai.serving_chat", level="WARNING"
        ) as logs:
            serving._ban_kimi_k3_think_channel_without_thinking(
                sampling_params, THINKING_OFF
            )
            # Warned once, not once per request.
            serving._ban_kimi_k3_think_channel_without_thinking(
                sampling_params, THINKING_OFF
            )
        self.assertEqual(len(logs.records), 1)
        self.assertEqual(sampling_params, {})
        # The stop still finds a special-token sequence to match.
        serving._stop_kimi_k3_reasoning_without_thinking(sampling_params, THINKING_OFF)
        self.assertEqual(
            get_request_stop_token_sequences(sampling_params["custom_params"]),
            [(CLOSE, THINK, THINK + 1, SEP)],
        )

    def test_the_switch_turns_it_off(self):
        serving = _serving(_FakeTokenizer())
        sampling_params = {}
        with envs.SGLANG_K3_THINKING_OFF_BAN_THINK.override(False):
            serving._ban_kimi_k3_think_channel_without_thinking(
                sampling_params, THINKING_OFF
            )
        self.assertEqual(sampling_params, {})


if __name__ == "__main__":
    unittest.main()
