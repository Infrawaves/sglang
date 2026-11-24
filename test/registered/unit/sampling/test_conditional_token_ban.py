"""Conditional token bans: the request transport, the per-batch state (build,
filter, merge) and the two masks (one token per request, and a speculative
verify chain). Pure CPU, no server, no model loading."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.sampling import conditional_token_ban
from sglang.srt.sampling.conditional_token_ban import (
    ConditionalTokenBan,
    batch_token_ban,
    last_input_token_ids,
)
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.sampling.sampling_params import (
    MAX_CONDITIONAL_TOKEN_BAN_PREVS,
    REQUEST_CONDITIONAL_TOKEN_BAN_KEY,
    TOP_K_ALL,
    clear_request_conditional_token_ban,
    get_request_conditional_token_ban,
    set_request_conditional_token_ban,
)
from sglang.test.test_utils import CustomTestCase

VOCAB = 16
DEVICE = "cpu"
# think, <|open|>, <|close|>, plus two ordinary tokens.
THINK, OPEN, CLOSE, RESPONSE, WORD = 4, 8, 9, 5, 1
# After <|open|> or <|close|>, mask think.
RULE = [THINK, OPEN, CLOSE]
PARSED = (THINK, (OPEN, CLOSE))
NEG_INF = float("-inf")


def _prevs(*token_ids):
    """One row of prev_ids: the triggers, padded."""
    return list(token_ids) + [-1] * (MAX_CONDITIONAL_TOKEN_BAN_PREVS - len(token_ids))


def _req(rule=RULE, output_ids=()):
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
        set_request_conditional_token_ban(
            sampling_params, banned=THINK, after=(OPEN, CLOSE)
        )
        self.assertEqual(
            sampling_params["custom_params"],
            {"mine": 1, REQUEST_CONDITIONAL_TOKEN_BAN_KEY: RULE},
        )
        self.assertEqual(
            get_request_conditional_token_ban(sampling_params["custom_params"]),
            PARSED,
        )

    def test_one_previous_token_is_enough(self):
        sampling_params = {}
        set_request_conditional_token_ban(sampling_params, banned=THINK, after=[CLOSE])
        self.assertEqual(
            get_request_conditional_token_ban(sampling_params["custom_params"]),
            (THINK, (CLOSE,)),
        )

    def test_set_does_not_mutate_the_callers_dict(self):
        original = {"mine": 1}
        sampling_params = {"custom_params": original}
        set_request_conditional_token_ban(
            sampling_params, banned=THINK, after=(OPEN, CLOSE)
        )
        self.assertEqual(original, {"mine": 1})

    def test_clear_removes_a_client_supplied_rule(self):
        sampling_params = {
            "custom_params": {"mine": 1, REQUEST_CONDITIONAL_TOKEN_BAN_KEY: [1, 2, 3]}
        }
        clear_request_conditional_token_ban(sampling_params)
        self.assertEqual(sampling_params["custom_params"], {"mine": 1})

        only_rule = {"custom_params": {REQUEST_CONDITIONAL_TOKEN_BAN_KEY: [1, 2, 3]}}
        clear_request_conditional_token_ban(only_rule)
        self.assertIsNone(only_rule["custom_params"])

    def test_clear_leaves_requests_without_a_rule_untouched(self):
        for sampling_params in ({}, {"custom_params": None}, {"custom_params": {}}):
            before = dict(sampling_params)
            clear_request_conditional_token_ban(sampling_params)
            self.assertEqual(sampling_params, before)

    def test_set_rejects_a_malformed_rule(self):
        too_many = list(range(MAX_CONDITIONAL_TOKEN_BAN_PREVS + 1))
        for banned, after in (
            (THINK, []),
            (THINK, too_many),
            (THINK, [CLOSE, -1]),
            (THINK, [CLOSE, "8"]),
            (-1, [CLOSE]),
            (True, [CLOSE]),
        ):
            with self.subTest(banned=banned, after=after):
                with self.assertRaises(ValueError):
                    set_request_conditional_token_ban({}, banned=banned, after=after)

    def test_get_validates(self):
        key = REQUEST_CONDITIONAL_TOKEN_BAN_KEY
        self.assertIsNone(get_request_conditional_token_ban(None))
        self.assertIsNone(get_request_conditional_token_ban({}))
        too_long = list(range(MAX_CONDITIONAL_TOKEN_BAN_PREVS + 2))
        for rule in ([THINK], "4,8,9", [THINK, OPEN, True], [THINK, -1], too_long):
            with self.subTest(rule=rule):
                self.assertIsNone(get_request_conditional_token_ban({key: rule}))
                with self.assertRaises(ValueError):
                    get_request_conditional_token_ban({key: rule}, strict=True)
        # Bounded by the vocabulary when it is known.
        self.assertIsNone(get_request_conditional_token_ban({key: RULE}, vocab_size=9))
        with self.assertRaises(ValueError):
            get_request_conditional_token_ban({key: RULE}, vocab_size=9, strict=True)
        self.assertEqual(
            get_request_conditional_token_ban({key: RULE}, vocab_size=10), PARSED
        )


class TestState(CustomTestCase):
    def test_no_rule_in_the_batch_builds_nothing(self):
        self.assertIsNone(_ban(_req(rule=None), _req(rule=None)))
        self.assertIsNone(ConditionalTokenBan.from_reqs([], DEVICE))

    def test_requests_without_custom_params_are_tolerated(self):
        bare = SimpleNamespace(sampling_params=SimpleNamespace(), output_ids=[])
        self.assertIsNone(ConditionalTokenBan.from_reqs([bare], DEVICE))

    def test_each_row_holds_its_own_rule(self):
        ban = _ban(_req(rule=None), _req(), _req(rule=[RESPONSE, WORD]))
        self.assertEqual(len(ban), 3)
        self.assertEqual(ban.rules, (None, PARSED, (RESPONSE, (WORD,))))
        self.assertEqual(
            ban.prev_ids.tolist(), [_prevs(), _prevs(OPEN, CLOSE), _prevs(WORD)]
        )
        self.assertEqual(ban.banned_ids.tolist(), [0, THINK, RESPONSE])
        self.assertEqual(ban.max_banned_id, RESPONSE)

    def test_the_rule_does_not_depend_on_the_output_so_far(self):
        # No output yet, an ordinary answer, and an output that began with <|open|>.
        ban = _ban(_req(), _req(output_ids=[WORD]), _req(output_ids=[OPEN, THINK]))
        self.assertEqual(ban.rules, (PARSED, PARSED, PARSED))
        logits = torch.zeros(3, VOCAB)
        ban.apply(logits, torch.tensor([CLOSE, CLOSE, CLOSE]))
        self.assertEqual(_masked(logits), [(0, THINK), (1, THINK), (2, THINK)])

    def test_batch_token_ban_reads_the_sampling_info(self):
        ban = _ban(_req())
        self.assertIs(batch_token_ban(SimpleNamespace(conditional_token_ban=ban)), ban)
        self.assertIsNone(batch_token_ban(SimpleNamespace(conditional_token_ban=None)))
        self.assertIsNone(batch_token_ban(SimpleNamespace()))
        self.assertIsNone(batch_token_ban(MagicMock()))
        self.assertIsNone(batch_token_ban(None))

    def test_filter_keeps_rows_aligned(self):
        ban = _ban(_req(), _req(rule=None), _req(rule=[RESPONSE, WORD]), _req())
        kept = ban.filtered([2, 1], torch.tensor([2, 1]))
        self.assertEqual(kept.rules, ((RESPONSE, (WORD,)), None))
        self.assertEqual(kept.prev_ids.tolist(), [_prevs(WORD), _prevs()])
        self.assertEqual(kept.banned_ids.tolist(), [RESPONSE, 0])
        self.assertEqual(kept.max_banned_id, RESPONSE)
        self.assertEqual(len(ban), 4)  # the source is untouched
        self.assertIsNone(ban.filtered([1], torch.tensor([1])))

    def test_merge_pads_the_side_without_rules(self):
        left = _ban(_req())
        right = _ban(_req(rule=None), _req(rule=[RESPONSE, WORD]))
        merged = ConditionalTokenBan.merged(left, right, 1, 2, DEVICE)
        self.assertEqual(merged.rules, (PARSED, None, (RESPONSE, (WORD,))))
        self.assertEqual(
            merged.prev_ids.tolist(), [_prevs(OPEN, CLOSE), _prevs(), _prevs(WORD)]
        )

        padded_left = ConditionalTokenBan.merged(None, left, 2, 1, DEVICE)
        self.assertEqual(padded_left.rules, (None, None, PARSED))
        self.assertEqual(padded_left.banned_ids.tolist(), [0, 0, THINK])
        self.assertEqual(
            padded_left.prev_ids.tolist(), [_prevs(), _prevs(), _prevs(OPEN, CLOSE)]
        )

        padded_right = ConditionalTokenBan.merged(left, None, 1, 2, DEVICE)
        self.assertEqual(padded_right.rules, (PARSED, None, None))
        self.assertIsNone(ConditionalTokenBan.merged(None, None, 1, 2, DEVICE))


class TestMask(CustomTestCase):
    def test_masks_only_the_banned_token_after_either_trigger(self):
        # rows: after <|open|>, after <|close|>, after another token, no rule
        ban = _ban(_req(), _req(), _req(), _req(rule=None))
        logits = torch.zeros(4, VOCAB)
        ban.apply(logits, torch.tensor([OPEN, CLOSE, WORD, CLOSE]))
        self.assertEqual(_masked(logits), [(0, THINK), (1, THINK)])

    def test_a_negative_previous_token_never_triggers(self):
        # A placeholder id equals the padding of the trigger slots.
        ban = _ban(_req(rule=None), _req())
        logits = torch.zeros(2, VOCAB)
        ban.apply(logits, torch.tensor([-1, -1]))
        self.assertEqual(_masked(logits), [])
        chain_logits = torch.zeros(4, VOCAB)
        ban.apply_chain(chain_logits, torch.tensor([[-1, CLOSE], [-1, CLOSE]]))
        self.assertEqual(_masked(chain_logits), [(3, THINK)])

    def test_other_logits_keep_their_values_and_dtype(self):
        ban = _ban(_req())
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                logits = torch.arange(VOCAB, dtype=dtype).unsqueeze(0).clone()
                expected = logits.clone()
                expected[0, THINK] = NEG_INF
                ban.apply(logits, torch.tensor([CLOSE], dtype=torch.int32))
                self.assertEqual(logits.dtype, dtype)
                self.assertTrue(torch.equal(logits, expected))

    def test_argmax_and_sampling_never_return_the_banned_token(self):
        ban = _ban(_req())
        for trigger in (OPEN, CLOSE):
            with self.subTest(trigger=trigger):
                logits = torch.zeros(1, VOCAB)
                logits[0, THINK] = 50.0
                logits[0, RESPONSE] = 1.0
                ban.apply(logits, torch.tensor([trigger]))
                self.assertEqual(int(logits.argmax(dim=-1)), RESPONSE)
                self.assertEqual(float(torch.softmax(logits, dim=-1)[0, THINK]), 0.0)

    def test_chain_masks_each_position_that_follows_a_trigger(self):
        # Request 0 has the rule, request 1 has none.
        ban = _ban(_req(), _req(rule=None))
        chain = torch.tensor(
            [
                [CLOSE, WORD, OPEN, CLOSE],  # anchor, then two drafted triggers
                [CLOSE, OPEN, CLOSE, OPEN],  # no rule: never masked
            ]
        )
        logits = torch.zeros(2 * 4, VOCAB)
        ban.apply_chain(logits, chain)
        # Row i * n + j continues chain[i, j].
        self.assertEqual(_masked(logits), [(0, THINK), (2, THINK), (3, THINK)])

    def test_chain_uses_each_requests_own_rule(self):
        ban = _ban(_req(), _req(rule=[RESPONSE, WORD]))  # after WORD, mask RESPONSE
        chain = torch.tensor([[CLOSE, WORD], [CLOSE, WORD]])
        logits = torch.zeros(4, VOCAB)
        ban.apply_chain(logits, chain)
        self.assertEqual(_masked(logits), [(0, THINK), (3, RESPONSE)])

    def test_mismatched_shapes_are_skipped(self):
        ban = _ban(_req(), _req())
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

    def test_a_skipped_mask_keeps_being_reported(self):
        ban = _ban(_req())
        wrong_rows = torch.zeros(2, VOCAB)
        with patch.dict(conditional_token_ban._skips, clear=True):
            with self.assertLogs(conditional_token_ban.logger, level="WARNING") as logs:
                for _ in range(100):
                    ban.apply(wrong_rows, torch.tensor([CLOSE]))
        # The 1st, 10th and 100th skip, each with the count so far.
        self.assertEqual(len(logs.records), 3)
        self.assertIn("(occurrence 100)", logs.records[-1].getMessage())


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
        before = _ban(_req(), _req(rule=None), _req())
        info = _info(3, conditional_token_ban=before)
        info.filter_batch([0, 1], torch.tensor([0, 1]))
        self.assertEqual(info.conditional_token_ban.rules, (PARSED, None))
        self.assertEqual(len(before), 3)  # the source is untouched

        other = _info(2)  # a batch without rules
        info.merge_batch(other)
        self.assertEqual(len(info), 4)
        merged = info.conditional_token_ban
        self.assertEqual(merged.rules, (PARSED, None, None, None))
        self.assertEqual(
            merged.prev_ids.tolist(),
            [_prevs(OPEN, CLOSE), _prevs(), _prevs(), _prevs()],
        )

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

        reqs = [_req()]
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
