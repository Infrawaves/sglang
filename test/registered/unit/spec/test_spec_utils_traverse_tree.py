"""Unit tests for the spec_utils grammar bitmask traversals (tree and chain).

xgrammar 0.2.0 tightened its FFI binding and rejects 0-d tensors where Python
ints are expected, so every token and row index handed to the grammar must be an int.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch
from xgrammar import GrammarCompiler, GrammarMatcher, TokenizerInfo

from sglang.srt.constrained.reasoner_grammar_backend import ReasonerGrammarObject
from sglang.srt.constrained.xgrammar_backend import (
    MAX_ROLLBACK_TOKENS,
    XGrammarGrammar,
)
from sglang.srt.speculative.spec_utils import (
    GrammarChain,
    generate_chain_token_bitmask,
    generate_token_bitmask,
    traverse_tree,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestTraverseTreePassesIntsToGrammar(CustomTestCase):
    def _record_grammar(self, bitmask):
        """A grammar mock that records every call argument and rejects torch tensors."""
        grammar = MagicMock()
        grammar.is_terminated.return_value = False
        grammar.allocate_vocab_mask.return_value = bitmask
        accept_calls = []
        fill_calls = []

        def record_accept(token):
            if isinstance(token, torch.Tensor):
                raise TypeError(f"accept_token got torch.Tensor: {token!r}")
            accept_calls.append(token)

        def record_fill(bitmask, idx):
            if isinstance(idx, torch.Tensor):
                raise TypeError(f"fill_vocab_mask got torch.Tensor idx: {idx!r}")
            fill_calls.append(idx)

        grammar.accept_token.side_effect = record_accept
        grammar.fill_vocab_mask.side_effect = record_fill
        grammar.rollback.return_value = None
        return grammar, accept_calls, fill_calls

    def _run_chain(self, verify_ids_2d, grammar):
        """The chain path DFLASH/DSPARK feed, on a single-request batch."""
        verify_ids_cpu = GrammarChain.from_device(verify_ids_2d).resolve()
        generate_chain_token_bitmask(
            reqs=[SimpleNamespace(grammar=grammar)],
            verify_ids_cpu=verify_ids_cpu,
            vocab_size=128,
        )

    def test_branching_tree_passes_ints(self):
        # Binary tree exercises both child recursion and sibling recursion:
        #   0 -+- 1
        #      +- 2 --- 3
        retrieve_next_token = torch.tensor([1, -1, 3, -1], dtype=torch.int32)
        retrieve_next_sibling = torch.tensor([-1, 2, -1, -1], dtype=torch.int32)
        draft_tokens = torch.tensor([100, 11, 22, 33], dtype=torch.int64)
        # all bits set: every draft token passes the parent's bitmask check
        bitmask = torch.full((4, 4), -1, dtype=torch.int32)

        grammar, accept_calls, fill_calls = self._record_grammar(bitmask)
        traverse_tree(
            retrieve_next_token,
            retrieve_next_sibling,
            draft_tokens,
            grammar,
            bitmask,
        )

        self.assertEqual(set(accept_calls), {11, 22, 33})
        self.assertEqual(set(fill_calls), {0, 1, 2, 3})
        for token in accept_calls:
            self.assertIsInstance(token, int)
        for idx in fill_calls:
            self.assertIsInstance(idx, int)

    def test_linear_chain_visits_all_positions_in_order(self):
        # Column 0 is the already-committed token; the rest are drafts in chain order.
        bitmask = torch.full((4, 4), -1, dtype=torch.int32)  # all allowed
        grammar, accept_calls, fill_calls = self._record_grammar(bitmask)
        self._run_chain(torch.tensor([[100, 11, 22, 33]]), grammar)

        # Root (col 0) is never accepted; every draft token is, in chain order.
        self.assertEqual(accept_calls, [11, 22, 33])
        self.assertEqual(fill_calls, [0, 1, 2, 3])
        for token in accept_calls:
            self.assertIsInstance(token, int)
        for idx in fill_calls:
            self.assertIsInstance(idx, int)

    def test_linear_chain_stops_at_grammar_reject(self):
        # A draft token the grammar disallows must stop the descent: no accept/fill
        # for that node or anything after it, so the mask rows past it stay unfilled
        # and only the already-filled prefix can be committed.
        bitmask = torch.full((4, 4), -1, dtype=torch.int32)  # all allowed
        # Disallow token id 7 (position 2) in position 1's mask (its parent).
        bitmask[1, 7 // 32] &= ~(1 << (7 % 32))
        grammar, accept_calls, fill_calls = self._record_grammar(bitmask)
        self._run_chain(torch.tensor([[100, 5, 7, 9]]), grammar)

        # Position 1 accepted+filled; position 2 rejected -> 2 and 3 skipped.
        self.assertEqual(accept_calls, [5])
        self.assertEqual(fill_calls, [0, 1])

    def test_linear_chain_stops_at_id_past_vocab(self):
        # A vocab that fills whole mask words leaves no word for id == vocab_size,
        # so the id check must stop the chain before the parent row is read.
        bitmask = torch.full((4, 1), -1, dtype=torch.int32)
        grammar, accept_calls, fill_calls = self._record_grammar(bitmask)
        generate_chain_token_bitmask(
            reqs=[SimpleNamespace(grammar=grammar)],
            verify_ids_cpu=torch.tensor([[100, 5, 32, 9]]),
            vocab_size=32,
        )

        self.assertEqual(accept_calls, [5])
        self.assertEqual(fill_calls, [0, 1])


# Synthetic vocab: no tokenizer download. Grammar: "a" "b"* "c", then <eos>.
_EOS, _A, _B, _C, _X, _THINK_END = range(6)
_VOCAB = ["<eos>", "a", "b", "c", "x", "</think>"]
_VOCAB_SIZE = len(_VOCAB)
_CHAIN_LEN = 6


class TestChainBitmaskMatchesTraverseTree(CustomTestCase):
    """Chain masks and final grammar state match the degenerate tree."""

    @classmethod
    def setUpClass(cls):
        info = TokenizerInfo(_VOCAB, stop_token_ids=[_EOS])
        cls.compiled = GrammarCompiler(info).compile_grammar('root ::= "a" "b"* "c"')

    def _xgrammar(self, prefix):
        grammar = XGrammarGrammar(
            GrammarMatcher(self.compiled, max_rollback_tokens=MAX_ROLLBACK_TOKENS),
            _VOCAB_SIZE,
            self.compiled,
            override_stop_tokens=None,
        )
        for token in prefix:
            grammar.accept_token(token)
        return grammar

    def _reasoner(self, prefix):
        grammar = ReasonerGrammarObject(self._xgrammar([]), think_end_ids=[_THINK_END])
        grammar.maybe_init_reasoning(True)
        for token in prefix:
            grammar.accept_token(token)
        return grammar

    def _batch(self):
        """One request per failure mode, plus a request without a grammar."""
        rows = [
            # Draft x is rejected by its parent row at position 3.
            (self._xgrammar([_A]), [_A, _B, _B, _X, _B, _C]),
            # <eos> at position 3 terminates the grammar; the rest is skipped.
            (self._xgrammar([_A]), [_A, _B, _C, _EOS, _B, _B]),
            # A draft id past vocab_size (and past the mask words) stops the chain.
            (self._xgrammar([_A]), [_A, _B, 1000, _B, _C, _EOS]),
            # Thinking rows stay unfilled; only tokens after </think> reach xgrammar.
            (self._reasoner([_X]), [_X, _X, _THINK_END, _A, _B, _C]),
            (None, [_X, _X, _X, _X, _X, _X]),
        ]
        reqs = [SimpleNamespace(grammar=grammar) for grammar, _ in rows]
        verify_ids = torch.tensor([ids for _, ids in rows], dtype=torch.int64)
        return reqs, verify_ids

    def _state(self, grammar):
        if grammar is None:
            return None
        inner = (
            grammar.grammar if isinstance(grammar, ReasonerGrammarObject) else grammar
        )
        next_mask = torch.full((1, 1), -1, dtype=torch.int32)
        inner.fill_vocab_mask(next_mask, 0)
        state = {
            "accepted_tokens": list(inner.accepted_tokens),
            "terminated": inner.is_terminated(),
            "next_mask": next_mask.tolist(),
        }
        if isinstance(grammar, ReasonerGrammarObject):
            state["reasoner"] = (
                grammar.tokens_in_think,
                grammar.tokens_after_end,
                grammar._matched_think_end_tokens,
                list(grammar._thinking_match_history),
            )
        return state

    def test_chain_matches_degenerate_tree(self):
        tree_reqs, verify_ids = self._batch()
        chain_reqs, _ = self._batch()
        bs = verify_ids.shape[0]
        next_token = torch.full((bs, _CHAIN_LEN), -1, dtype=torch.int64)
        next_token[:, :-1] = torch.arange(1, _CHAIN_LEN, dtype=torch.int64)
        next_sibling = torch.full((bs, _CHAIN_LEN), -1, dtype=torch.int64)

        tree_mask, _ = generate_token_bitmask(
            tree_reqs, next_token, next_sibling, verify_ids, _VOCAB_SIZE
        )
        chain_mask, _ = generate_chain_token_bitmask(
            reqs=chain_reqs,
            verify_ids_cpu=GrammarChain.from_device(verify_ids).resolve(),
            vocab_size=_VOCAB_SIZE,
        )

        self.assertTrue(torch.equal(chain_mask, tree_mask))
        filled = (chain_mask != -1).any(dim=1).view(bs, _CHAIN_LEN)
        filled_positions = [
            [pos for pos in range(_CHAIN_LEN) if filled[i, pos]] for i in range(bs)
        ]
        self.assertEqual(
            filled_positions, [[0, 1, 2], [0, 1, 2], [0, 1], [2, 3, 4, 5], []]
        )
        for tree_req, chain_req in zip(tree_reqs, chain_reqs):
            self.assertEqual(
                self._state(chain_req.grammar), self._state(tree_req.grammar)
            )


if __name__ == "__main__":
    unittest.main()
