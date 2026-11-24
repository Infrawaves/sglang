"""Kimi K3 with thinking off: the serving layer attaches the conditional token
ban ``[<|close|>, think, <|open|>]`` to the request, and only then. Pure CPU."""

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
)

OPEN, CLOSE, SEP, THINK, RESPONSE = 900, 901, 902, 910, 911


class _FakeTokenizer:
    """Encodes K3 markers as a special token, one name token and <|sep|>."""

    special = {"<|open|>": OPEN, "<|close|>": CLOSE, "<|sep|>": SEP}
    names = {"think": THINK, "response": RESPONSE}

    def __init__(self):
        self.encode_calls = 0

    def encode(self, text, add_special_tokens=False):
        self.encode_calls += 1
        head, rest = text[: text.index("|>") + 2], text[text.index("|>") + 2 :]
        name, tail = rest[: rest.index("<|")], rest[rest.index("<|") :]
        return [self.special[head], self.names[name], self.special[tail]]

    def convert_tokens_to_ids(self, token):
        return self.special[token]


class _PlainTextTokenizer(_FakeTokenizer):
    """Spells the markers with ordinary tokens, as a tokenizer without them would."""

    def encode(self, text, add_special_tokens=False):
        self.encode_calls += 1
        return list(range(len(text)))


def _serving(tokenizer):
    serving = object.__new__(OpenAIServingChat)
    serving.tokenizer_manager = SimpleNamespace(tokenizer=tokenizer)
    return serving


class TestKimiK3ThinkCloseBan(CustomTestCase):
    def test_thinking_off_attaches_the_rule(self):
        serving = _serving(_FakeTokenizer())
        sampling_params = {}
        serving._ban_kimi_k3_think_close_without_thinking(
            sampling_params, {"thinking": False}
        )
        self.assertEqual(
            get_request_conditional_token_ban(sampling_params["custom_params"]),
            (CLOSE, THINK, OPEN),
        )

    def test_thinking_on_or_unset_attaches_nothing(self):
        serving = _serving(_FakeTokenizer())
        for kwargs in (None, {}, {"thinking": True}, {"thinking_effort": "low"}):
            with self.subTest(kwargs=kwargs):
                sampling_params = {}
                serving._ban_kimi_k3_think_close_without_thinking(
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
        serving._ban_kimi_k3_think_close_without_thinking(
            sampling_params, {"thinking": True}
        )
        self.assertEqual(sampling_params["custom_params"], {"mine": 1})

        serving._ban_kimi_k3_think_close_without_thinking(
            sampling_params, {"thinking": False}
        )
        self.assertEqual(
            sampling_params["custom_params"],
            {"mine": 1, REQUEST_CONDITIONAL_TOKEN_BAN_KEY: [CLOSE, THINK, OPEN]},
        )

    def test_the_markers_are_encoded_once_per_server(self):
        tokenizer = _FakeTokenizer()
        serving = _serving(tokenizer)
        for _ in range(3):
            serving._ban_kimi_k3_think_close_without_thinking({}, {"thinking": False})
        self.assertEqual(tokenizer.encode_calls, 2)  # think close and think open

    def test_a_tokenizer_without_the_special_tokens_disables_the_ban(self):
        serving = _serving(_PlainTextTokenizer())
        sampling_params = {}
        with self.assertLogs(
            "sglang.srt.entrypoints.openai.serving_chat", level="WARNING"
        ):
            serving._ban_kimi_k3_think_close_without_thinking(
                sampling_params, {"thinking": False}
            )
        self.assertEqual(sampling_params, {})
        self.assertIsNone(serving._kimi_k3_think_close_ban_rule())

    def test_the_switch_turns_it_off(self):
        serving = _serving(_FakeTokenizer())
        sampling_params = {}
        with envs.SGLANG_K3_THINKING_OFF_BAN_THINK_CLOSE.override(False):
            serving._ban_kimi_k3_think_close_without_thinking(
                sampling_params, {"thinking": False}
            )
        self.assertEqual(sampling_params, {})


if __name__ == "__main__":
    unittest.main()
