import unittest
from types import SimpleNamespace

from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestClampMaxNewTokens(CustomTestCase):
    """`--clamp-max-new-tokens` clamps an over-budget output but never the input."""

    def _manager(self, *, clamp, auto_truncate=False):
        manager = TokenizerManager.__new__(TokenizerManager)
        manager.context_len = 10
        manager.num_reserved_tokens = 0
        manager.allow_auto_truncate = auto_truncate
        manager.clamp_max_new_tokens = clamp
        manager.validate_total_tokens = True
        manager.is_generation = True
        manager.model_config = SimpleNamespace(
            joint_head_config=None, pointer_head=None
        )
        return manager

    @staticmethod
    def _req(input_len, max_new_tokens):
        return GenerateReqInput(
            input_ids=list(range(1, input_len + 1)),
            sampling_params={"max_new_tokens": max_new_tokens},
        )

    def test_rejects_over_budget_output_by_default(self):
        req = self._req(3, 100)
        with self.assertRaisesRegex(ValueError, "You requested a total of 103"):
            self._manager(clamp=False)._validate_one_request(req, req.input_ids)

    def test_clamps_over_budget_output(self):
        req = self._req(3, 100)
        self._manager(clamp=True)._validate_one_request(req, req.input_ids)
        self.assertEqual(req.sampling_params["max_new_tokens"], 7)

    def test_still_rejects_over_long_input(self):
        req = self._req(10, 5)
        with self.assertRaisesRegex(ValueError, "longer than the model's context"):
            self._manager(clamp=True)._validate_one_request(req, req.input_ids)
        self.assertEqual(len(req.input_ids), 10, "input must not be truncated")


if __name__ == "__main__":
    unittest.main()
