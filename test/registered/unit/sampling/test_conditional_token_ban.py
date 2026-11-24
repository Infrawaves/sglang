"""Conditional token bans: the request transport, the per-batch state (build,
filter, merge, settle) and the two masks (one token per request, and a
speculative verify chain). Pure CPU, no server, no model loading."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.sampling.conditional_token_ban import (
    ARMED,
    OFF,
    PENDING,
    ConditionalTokenBan,
    armed_token_ban,
    last_input_token_ids,
    settle_pending_token_ban,
)
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.sampling.sampling_params import (
    REQUEST_CONDITIONAL_TOKEN_BAN_KEY,
    TOP_K_ALL,
    get_request_conditional_token_ban,
    set_request_conditional_token_ban,
)
from sglang.test.test_utils import CustomTestCase

VOCAB = 16
DEVICE = "cpu"
# <|close|>, think, <|open|>, plus two ordinary tokens.
CLOSE, THINK, OPEN, RESPONSE, WORD = 9, 4, 8, 5, 1
RULE = [CLOSE, THINK, OPEN]
NEG_INF = float("-inf")


def _req(output_ids=(), rule=RULE):
    custom_params = None if rule is None else {REQUEST_CONDITIONAL_TOKEN_BAN_KEY: rule}
    return SimpleNamespace(
        sampling_params=SimpleNamespace(custom_params=custom_params),
        output_ids=list(output_ids),
    )


def _ban(*reqs):
    return ConditionalTokenBan.from_reqs(list(reqs), DEVICE)


def _masked(logits):
    """(row, column) pairs that hold -inf."""
    return sorted(map(tuple, torch.isinf(logits).nonzero().tolist()))


def _info(batch_size, **overrides):
    defaults = dict(
        temperatures=torch.ones(batch_size, 1),
        top_ps=torch.ones(batch_size),
        top_ks=torch.full((batch_size,), TOP_K_ALL, dtype=torch.int32),
        min_ps=torch.zeros(batch_size),
        is_all_greedy=False,
        is_any_greedy=False,
        need_top_p_sampling=False,
        need_top_k_sampling=False,
        need_min_p_sampling=False,
        vocab_size=VOCAB,
        device=DEVICE,
        penalizer_orchestrator=MagicMock(is_required=False),
    )
    defaults.update(overrides)
    return SamplingBatchInfo(**defaults)


class TestRequestTransport(CustomTestCase):
    def test_set_then_get_round_trips(self):
        sampling_params = {"custom_params": {"mine": 1}}
        set_request_conditional_token_ban(sampling_params, RULE)
        self.assertEqual(
            sampling_params["custom_params"],
            {"mine": 1, REQUEST_CONDITIONAL_TOKEN_BAN_KEY: RULE},
        )
        self.assertEqual(
            get_request_conditional_token_ban(sampling_params["custom_params"]),
            (CLOSE, THINK, OPEN),
        )

    def test_set_does_not_mutate_the_callers_dict(self):
        original = {"mine": 1}
        sampling_params = {"custom_params": original}
        set_request_conditional_token_ban(sampling_params, RULE)
        self.assertEqual(original, {"mine": 1})

    def test_none_removes_a_client_supplied_rule(self):
        sampling_params = {
            "custom_params": {"mine": 1, REQUEST_CONDITIONAL_TOKEN_BAN_KEY: [1, 2, 3]}
        }
        set_request_conditional_token_ban(sampling_params, None)
        self.assertEqual(sampling_params["custom_params"], {"mine": 1})

        only_rule = {"custom_params": {REQUEST_CONDITIONAL_TOKEN_BAN_KEY: [1, 2, 3]}}
        set_request_conditional_token_ban(only_rule, None)
        self.assertIsNone(only_rule["custom_params"])

    def test_none_leaves_requests_without_a_rule_untouched(self):
        for sampling_params in ({}, {"custom_params": None}, {"custom_params": {}}):
            before = dict(sampling_params)
            set_request_conditional_token_ban(sampling_params, None)
            self.assertEqual(sampling_params, before)

    def test_set_rejects_a_malformed_rule(self):
        for rule in ([CLOSE, THINK], [CLOSE, THINK, -1], [CLOSE, THINK, "8"]):
            with self.subTest(rule=rule), self.assertRaises(ValueError):
                set_request_conditional_token_ban({}, rule)

    def test_get_validates(self):
        key = REQUEST_CONDITIONAL_TOKEN_BAN_KEY
        self.assertIsNone(get_request_conditional_token_ban(None))
        self.assertIsNone(get_request_conditional_token_ban({}))
        for rule in ([CLOSE, THINK], "9,4,8", [CLOSE, THINK, True], [CLOSE, -1, OPEN]):
            with self.subTest(rule=rule):
                self.assertIsNone(get_request_conditional_token_ban({key: rule}))
                with self.assertRaises(ValueError):
                    get_request_conditional_token_ban({key: rule}, strict=True)
        # Bounded by the vocabulary when it is known.
        self.assertIsNone(get_request_conditional_token_ban({key: RULE}, vocab_size=9))
        with self.assertRaises(ValueError):
            get_request_conditional_token_ban({key: RULE}, vocab_size=9, strict=True)
        self.assertEqual(
            get_request_conditional_token_ban({key: RULE}, vocab_size=10),
            (CLOSE, THINK, OPEN),
        )


class TestState(CustomTestCase):
    def test_no_rule_in_the_batch_builds_nothing(self):
        self.assertIsNone(_ban(_req(rule=None), _req([WORD], rule=None)))
        self.assertIsNone(ConditionalTokenBan.from_reqs([], DEVICE))

    def test_requests_without_custom_params_are_tolerated(self):
        bare = SimpleNamespace(sampling_params=SimpleNamespace(), output_ids=[])
        self.assertIsNone(ConditionalTokenBan.from_reqs([bare], DEVICE))

    def test_state_follows_the_first_output_token(self):
        ban = _ban(
            _req(rule=None),  # no rule
            _req([]),  # first token not known yet
            _req([WORD, CLOSE]),  # an ordinary answer
            _req([OPEN, THINK]),  # opened a channel of its own
        )
        self.assertEqual(ban.states, (OFF, PENDING, ARMED, OFF))
        self.assertTrue(ban.has_armed)
        self.assertTrue(ban.has_pending)
        self.assertEqual(ban.prev_ids.tolist(), [-1, -1, CLOSE, -1])
        self.assertEqual(ban.banned_ids.tolist(), [0, 0, THINK, 0])
        self.assertEqual(ban.max_banned_id, THINK)

    def test_pending_only_batch_needs_no_mask(self):
        ban = _ban(_req([]), _req([]))
        self.assertFalse(ban.has_armed)
        self.assertEqual(ban.prev_ids.tolist(), [-1, -1])
        self.assertIsNone(armed_token_ban(SimpleNamespace(conditional_token_ban=ban)))

    def test_settle_resolves_pending_rows_into_a_new_object(self):
        first, second, third = _req([]), _req([]), _req([])
        ban = _ban(first, second, third)
        info = SimpleNamespace(conditional_token_ban=ban)

        settle_pending_token_ban(info, [first, second, third])
        self.assertIs(info.conditional_token_ban, ban)  # nothing arrived yet

        first.output_ids.append(WORD)
        second.output_ids.append(OPEN)
        settle_pending_token_ban(info, [first, second, third])
        settled = info.conditional_token_ban
        self.assertIsNot(settled, ban)
        self.assertEqual(settled.states, (ARMED, OFF, PENDING))
        self.assertEqual(settled.prev_ids.tolist(), [CLOSE, -1, -1])
        # A forward that started earlier still holds the old object, unchanged.
        self.assertEqual(ban.states, (PENDING, PENDING, PENDING))
        self.assertEqual(ban.prev_ids.tolist(), [-1, -1, -1])

    def test_settle_ignores_a_batch_of_another_size_and_foreign_objects(self):
        req = _req([])
        ban = _ban(req)
        req.output_ids.append(WORD)
        self.assertIs(ban.refreshed([req, _req([WORD])]), ban)

        settle_pending_token_ban(None, [req])
        mock_info = MagicMock()
        settle_pending_token_ban(mock_info, [req])
        mock_info.conditional_token_ban.refreshed.assert_not_called()
        self.assertIsNone(armed_token_ban(mock_info))
        self.assertIsNone(armed_token_ban(None))

    def test_filter_keeps_rows_aligned(self):
        ban = _ban(_req([WORD]), _req(rule=None), _req([OPEN]), _req([WORD]))
        kept = ban.filtered([3, 1], torch.tensor([3, 1]))
        self.assertEqual(kept.states, (ARMED, OFF))
        self.assertEqual(kept.prev_ids.tolist(), [CLOSE, -1])
        self.assertEqual(len(ban), 4)  # the source is untouched
        self.assertIsNone(ban.filtered([1], torch.tensor([1])))

    def test_merge_pads_the_side_without_rules(self):
        left = _ban(_req([WORD]))
        right = _ban(_req([]), _req([OPEN]))
        merged = ConditionalTokenBan.merged(left, right, 1, 2, DEVICE)
        self.assertEqual(merged.states, (ARMED, PENDING, OFF))
        self.assertEqual(merged.prev_ids.tolist(), [CLOSE, -1, -1])

        padded_left = ConditionalTokenBan.merged(None, left, 2, 1, DEVICE)
        self.assertEqual(padded_left.states, (OFF, OFF, ARMED))
        self.assertEqual(padded_left.rules[:2], (None, None))
        self.assertEqual(padded_left.banned_ids.tolist(), [0, 0, THINK])

        padded_right = ConditionalTokenBan.merged(left, None, 1, 2, DEVICE)
        self.assertEqual(padded_right.states, (ARMED, OFF, OFF))
        self.assertIsNone(ConditionalTokenBan.merged(None, None, 1, 2, DEVICE))


class TestMask(CustomTestCase):
    def test_masks_only_the_banned_token_after_the_trigger(self):
        # rows: armed + trigger, armed + other token, off + trigger, pending + trigger
        ban = _ban(_req([WORD]), _req([WORD]), _req([OPEN]), _req([]))
        logits = torch.zeros(4, VOCAB)
        ban.apply(logits, torch.tensor([CLOSE, WORD, CLOSE, CLOSE]))
        self.assertEqual(_masked(logits), [(0, THINK)])

    def test_other_logits_keep_their_values_and_dtype(self):
        ban = _ban(_req([WORD]))
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                logits = torch.arange(VOCAB, dtype=dtype).unsqueeze(0).clone()
                expected = logits.clone()
                expected[0, THINK] = NEG_INF
                ban.apply(logits, torch.tensor([CLOSE], dtype=torch.int32))
                self.assertEqual(logits.dtype, dtype)
                self.assertTrue(torch.equal(logits, expected))

    def test_argmax_and_sampling_never_return_the_banned_token(self):
        ban = _ban(_req([WORD]))
        logits = torch.zeros(1, VOCAB)
        logits[0, THINK] = 50.0
        logits[0, RESPONSE] = 1.0
        ban.apply(logits, torch.tensor([CLOSE]))
        self.assertEqual(int(logits.argmax(dim=-1)), RESPONSE)
        self.assertEqual(float(torch.softmax(logits, dim=-1)[0, THINK]), 0.0)

    def test_chain_masks_each_position_that_follows_the_trigger(self):
        # Request 0 is armed, request 1 opened its own channel.
        ban = _ban(_req([WORD]), _req([OPEN]))
        chain = torch.tensor(
            [
                [CLOSE, WORD, WORD, CLOSE],  # anchor and the last draft trigger
                [CLOSE, CLOSE, CLOSE, CLOSE],  # off: never masked
            ]
        )
        logits = torch.zeros(2 * 4, VOCAB)
        ban.apply_chain(logits, chain)
        # Row i * n + j continues chain[i, j].
        self.assertEqual(_masked(logits), [(0, THINK), (3, THINK)])

    def test_chain_uses_each_requests_own_rule(self):
        other = [WORD, RESPONSE, OPEN]  # after WORD, mask RESPONSE
        ban = _ban(_req([WORD]), _req([THINK], rule=other))
        chain = torch.tensor([[CLOSE, WORD], [CLOSE, WORD]])
        logits = torch.zeros(4, VOCAB)
        ban.apply_chain(logits, chain)
        self.assertEqual(_masked(logits), [(0, THINK), (3, RESPONSE)])

    def test_mismatched_shapes_are_skipped(self):
        ban = _ban(_req([WORD]), _req([WORD]))
        for logits, prev in (
            (torch.zeros(3, VOCAB), torch.tensor([CLOSE, CLOSE])),  # rows
            (torch.zeros(2, VOCAB), torch.tensor([CLOSE])),  # previous tokens
            (torch.zeros(2, THINK), torch.tensor([CLOSE, CLOSE])),  # vocab too narrow
        ):
            ban.apply(logits, prev)
            self.assertEqual(_masked(logits), [])
        logits = torch.zeros(5, VOCAB)  # not bs * n
        ban.apply_chain(logits, torch.tensor([[CLOSE, CLOSE], [CLOSE, CLOSE]]))
        self.assertEqual(_masked(logits), [])
        logits = torch.zeros(6, VOCAB)  # chain of another batch size
        ban.apply_chain(logits, torch.tensor([[CLOSE, CLOSE]] * 3))
        self.assertEqual(_masked(logits), [])


class TestLastInputTokenIds(CustomTestCase):
    def _batch(self, mode, **overrides):
        fields = dict(
            input_ids=torch.tensor([CLOSE, WORD]),
            forward_mode=mode,
            input_ids_from_graph=False,
        )
        fields.update(overrides)
        return SimpleNamespace(**fields)

    def test_decode_continues_its_input_tokens(self):
        batch = self._batch(ForwardMode.DECODE)
        self.assertIs(last_input_token_ids(batch), batch.input_ids)

    def test_unknown_inputs_return_none(self):
        self.assertIsNone(last_input_token_ids(self._batch(ForwardMode.EXTEND)))
        self.assertIsNone(last_input_token_ids(self._batch(ForwardMode.MIXED)))
        self.assertIsNone(
            last_input_token_ids(self._batch(ForwardMode.DECODE, input_ids=None))
        )
        self.assertIsNone(
            last_input_token_ids(
                self._batch(ForwardMode.DECODE, input_ids_from_graph=True)
            )
        )


class TestSamplingBatchInfo(CustomTestCase):
    def test_filter_and_merge_carry_the_ban(self):
        before = _ban(_req([WORD]), _req([]), _req([OPEN]))
        info = _info(3, conditional_token_ban=before)
        info.filter_batch([0, 1], torch.tensor([0, 1]))
        self.assertEqual(info.conditional_token_ban.states, (ARMED, PENDING))
        self.assertEqual(before.states, (ARMED, PENDING, OFF))

        other = _info(2)  # a batch without rules
        info.merge_batch(other)
        self.assertEqual(len(info), 4)
        merged = info.conditional_token_ban
        self.assertEqual(merged.states, (ARMED, PENDING, OFF, OFF))
        self.assertEqual(merged.prev_ids.tolist(), [CLOSE, -1, -1, -1])

        info.filter_batch([2, 3], torch.tensor([2, 3]))
        self.assertIsNone(info.conditional_token_ban)

    def test_batches_without_rules_stay_without(self):
        info = _info(2)
        info.merge_batch(_info(1))
        self.assertIsNone(info.conditional_token_ban)
        info.filter_batch([0], torch.tensor([0]))
        self.assertIsNone(info.conditional_token_ban)

    def test_only_plain_and_dflash_sampling_take_the_ban(self):
        from sglang.srt.sampling import sampling_batch_info

        reqs = [_req([WORD])]
        for algorithm, taken in (
            (None, True),
            ("DFLASH", True),
            ("dflash", True),
            ("DSPARK", False),
            ("EAGLE", False),
        ):
            spec = SimpleNamespace(speculative_algorithm=algorithm)
            with self.subTest(algorithm=algorithm):
                with patch.object(sampling_batch_info, "get_spec", return_value=spec):
                    ban = sampling_batch_info._conditional_token_ban_from_reqs(
                        reqs, DEVICE
                    )
                self.assertEqual(ban is not None, taken)

    def test_no_rule_never_reads_the_speculative_config(self):
        from sglang.srt.sampling import sampling_batch_info

        with patch.object(sampling_batch_info, "get_spec") as get_spec:
            self.assertIsNone(
                sampling_batch_info._conditional_token_ban_from_reqs(
                    [_req(rule=None)], DEVICE
                )
            )
        get_spec.assert_not_called()


if __name__ == "__main__":
    unittest.main()
