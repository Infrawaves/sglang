"""Keep XGrammar's matcher and Python history semantics aligned on rollback."""

import unittest

import torch
from xgrammar import GrammarCompiler, GrammarMatcher, TokenizerInfo

from sglang.srt.constrained.xgrammar_backend import XGrammarGrammar
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestXGrammarRollback(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        info = TokenizerInfo(["a", "b", "<eos>"], stop_token_ids=[2])
        cls.compiled = GrammarCompiler(info).compile_grammar('root ::= "a"* "b"')

    def _grammar(self, history_len=32):
        grammar = XGrammarGrammar(
            GrammarMatcher(self.compiled),
            3,
            self.compiled,
            None,
            grammar_stats=None,
        )
        for _ in range(history_len):
            grammar.accept_token(0)
        return grammar

    def _mask(self, grammar):
        mask = torch.full((1, 1), -1, dtype=torch.int32)
        grammar.fill_vocab_mask(mask, 0)
        return mask.tolist()

    def test_long_history_and_terminated_matcher_rollback(self):
        grammar = self._grammar(1024)
        for count in (1, 7):
            expected = grammar.accepted_tokens[:-count]
            grammar.rollback(count)
            self.assertEqual(grammar.accepted_tokens, expected)
            self.assertEqual(grammar.current_token, 0)
        for token in (1, 2):
            grammar.accept_token(token)
        self.assertTrue(grammar.is_terminated())
        grammar.rollback(2)
        self.assertFalse(grammar.is_terminated())
        self.assertEqual(grammar.accepted_tokens, [0] * 1016)
        self.assertEqual(grammar.current_token, 2)  # Existing wrapper behavior.
        grammar.accept_token(1)
        grammar.accept_token(2)
        self.assertTrue(grammar.is_terminated())

    def test_zero_and_negative_keep_existing_slice_semantics(self):
        for count in (0, -1):
            with self.subTest(count=count):
                grammar = self._grammar()
                before_mask = self._mask(grammar)
                expected = grammar.accepted_tokens[:-count]
                grammar.rollback(count)
                self.assertEqual(grammar.accepted_tokens, expected)
                self.assertEqual(self._mask(grammar), before_mask)
                self.assertEqual(grammar.current_token, 0)
                grammar.accept_token(1)
                grammar.accept_token(2)
                self.assertTrue(grammar.is_terminated())

    def test_matcher_failure_preserves_python_history(self):
        for history_len, count in ((2, 3), (32, 33)):
            with self.subTest(history_len=history_len, count=count):
                grammar = self._grammar(history_len)
                before = grammar.accepted_tokens[:]
                before_mask = self._mask(grammar)
                with self.assertRaises(RuntimeError):
                    grammar.rollback(count)
                self.assertEqual(grammar.accepted_tokens, before)
                self.assertEqual(grammar.current_token, 0)
                self.assertEqual(self._mask(grammar), before_mask)
                grammar.accept_token(1)
                grammar.accept_token(2)
                self.assertTrue(grammar.is_terminated())


if __name__ == "__main__":
    unittest.main()
