"""Regression tests for speculative grammar traversal and rollback.

XGrammar requires Python ints at its FFI boundary. Compare the optimized host
traversal against the original Tensor-based traversal for masks, matcher state,
reasoning transitions, and CPU sampling results.
"""

import copy
import unittest
from unittest.mock import MagicMock

import torch
from xgrammar import GrammarCompiler, GrammarMatcher, TokenizerInfo

from sglang.srt.constrained.reasoner_grammar_backend import ReasonerGrammarObject
from sglang.srt.constrained.xgrammar_backend import XGrammarGrammar
from sglang.srt.speculative.spec_utils import GrammarTree, traverse_tree
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestTraverseTreePassesIntsToGrammar(unittest.TestCase):
    def _record_grammar(self):
        """A grammar mock that records every call argument and rejects torch tensors."""
        grammar = MagicMock()
        grammar.is_terminated.return_value = False
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

    def _chain(self, verify_ids_2d):
        """Row 0 of the links chain-verify algorithms actually feed traverse_tree."""
        links = GrammarTree.from_linear_chain(verify_ids_2d).resolve()
        return tuple(t[0] for t in links)

    def test_branching_tree_passes_ints(self):
        # Binary tree exercises both child recursion and sibling recursion:
        #   0 ─┬─ 1
        #      └─ 2 ─── 3
        retrieve_next_token = torch.tensor([1, -1, 3, -1], dtype=torch.int32)
        retrieve_next_sibling = torch.tensor([-1, 2, -1, -1], dtype=torch.int32)
        draft_tokens = torch.tensor([100, 11, 22, 33], dtype=torch.int64)
        # all bits set: every draft token passes the parent's bitmask check
        bitmask = torch.full((4, 4), -1, dtype=torch.int32)

        grammar, accept_calls, fill_calls = self._record_grammar()
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
        # Chain-verify algorithms (DFLASH/DSPARK) have no branching, so their tree
        # degenerates to 0 -- 1 -- 2 -- 3 with column 0 the already-committed token.
        rnt, rns, draft_tokens = self._chain(torch.tensor([[100, 11, 22, 33]]))
        self.assertEqual(rnt.tolist(), [1, 2, 3, -1])
        self.assertEqual(rns.tolist(), [-1, -1, -1, -1])
        bitmask = torch.full((4, 4), -1, dtype=torch.int32)  # all allowed

        grammar, accept_calls, fill_calls = self._record_grammar()
        traverse_tree(rnt, rns, draft_tokens, grammar, bitmask)

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
        rnt, rns, draft_tokens = self._chain(torch.tensor([[100, 5, 7, 9]]))
        bitmask = torch.full((4, 4), -1, dtype=torch.int32)  # all allowed
        # Disallow token id 7 (draft_tokens[2]) in node 1's mask (its parent).
        bitmask[1, 7 // 32] &= ~(1 << (7 % 32))

        grammar, accept_calls, fill_calls = self._record_grammar()
        traverse_tree(rnt, rns, draft_tokens, grammar, bitmask)

        # Node 1 accepted+filled; node 2 rejected -> node 2 and node 3 skipped.
        self.assertEqual(accept_calls, [5])
        self.assertEqual(fill_calls, [0, 1])


def _tensor_reference(next_token, next_sibling, tokens, grammar, mask, vocab_size):
    """Frozen pre-optimization traversal, including its scalar Tensor operations."""

    def dfs(curr, parent):
        if curr == 0:
            accepted = True
        else:
            token = tokens[curr]
            accepted = (
                False
                if vocab_size and token >= vocab_size
                else (mask[parent][token // 32] & (1 << (token % 32))) != 0
            )
        if accepted:
            if curr != 0:
                grammar.accept_token(int(tokens[curr]))
            if not grammar.is_terminated():
                grammar.fill_vocab_mask(mask, curr)
                if next_token[curr] != -1:
                    dfs(int(next_token[curr]), curr)
            if curr != 0:
                grammar.rollback(1)
        if next_sibling[curr] != -1:
            dfs(int(next_sibling[curr]), parent)

    dfs(0, -1)


def _wrapper_state(grammar):
    # Compare all Python wrapper fields; native matcher state is also checked by
    # its masks, termination, and subsequent real commits.
    state = {}
    for name, value in vars(grammar).items():
        if name in ("matcher", "ctx"):
            continue
        if name == "grammar" and value is not None:
            value = _wrapper_state(value)
        elif name == "_think_end_matcher":
            value = vars(value)
        state[name] = copy.deepcopy(value)
    return state


class _RecordedGrammar:
    def __init__(self, grammar):
        self.grammar = grammar
        self.events = []
        self.rows = []

    def __getattr__(self, name):
        method = getattr(self.grammar, name)
        if name not in ("accept_token", "rollback", "fill_vocab_mask", "is_terminated"):
            return method

        def call(*args):
            result = method(*args)
            if name == "fill_vocab_mask":
                mask, row = args
                detail = (row, mask[row].tolist())
                self.rows.append(row)
            else:
                detail = (args, result)
            self.events.append((name, detail, _wrapper_state(self.grammar)))
            return result

        return call


class _SliceRollbackGrammar(XGrammarGrammar):
    def rollback(self, count):
        self.matcher.rollback(count)
        self.accepted_tokens = self.accepted_tokens[:-count]


class TestTraverseTreeRealMatcherParity(unittest.TestCase):
    VOCAB_SIZE = 67
    A, B, C, EOS = 31, 63, 66, 64

    @classmethod
    def setUpClass(cls):
        # Signed bit 31 in two words, plus the last valid token in a partial word.
        vocab = ["!"] * cls.VOCAB_SIZE
        for token, text in [
            (cls.A, "a"),
            (cls.B, "b"),
            (cls.C, "c"),
            (cls.EOS, "<eos>"),
        ]:
            vocab[token] = text
        info = TokenizerInfo(vocab, stop_token_ids=[cls.EOS])
        compiler = GrammarCompiler(info)
        cls.compiled = compiler.compile_grammar('root ::= "abc"')
        cls.choice_compiled = compiler.compile_grammar('root ::= ("a" | "b")+ "c"')

    def _grammar(self, reasoner=False, terminated=False, choice=False, reference=False):
        compiled = self.choice_compiled if choice else self.compiled
        wrapper = _SliceRollbackGrammar if reference else XGrammarGrammar
        grammar = wrapper(
            GrammarMatcher(compiled),
            self.VOCAB_SIZE,
            compiled,
            None,
            grammar_stats=None,
        )
        if reasoner:
            grammar = ReasonerGrammarObject(grammar, think_end_ids=[1, 2])
            grammar.maybe_init_reasoning(True)
            grammar.accept_token(1)  # First half of a multi-token reasoning end.
        if terminated:
            for token in [self.A, self.B, self.C, self.EOS]:
                grammar.accept_token(token)
        return grammar

    def _compare(
        self,
        links,
        tokens,
        *,
        dtype=torch.int64,
        vocab_size=67,
        noncontiguous=False,
        reasoner=False,
        terminated=False,
        choice=False,
        commit=(),
    ):
        def tensor(values):
            value = torch.tensor(values, dtype=dtype)
            if noncontiguous:
                storage = torch.zeros(2 * len(values), dtype=dtype)
                storage[::2] = value
                value = storage[::2]
            return value

        inputs = tuple(tensor(x) for x in (*links, tokens))
        originals = tuple(x.clone() for x in inputs)
        results = []
        for fn in (_tensor_reference, traverse_tree):
            grammar = self._grammar(
                reasoner=reasoner,
                terminated=terminated,
                choice=choice,
                reference=fn is _tensor_reference,
            )
            recorder = _RecordedGrammar(grammar)
            mask = torch.full((len(tokens), 3), -1, dtype=torch.int32)
            fn(*inputs, recorder, mask, vocab_size)
            # Actual CPU mask application + fixed logits/RNG detects changes in
            # the sampled distributions, including unconstrained thinking rows.
            logits = torch.linspace(-2, 2, self.VOCAB_SIZE).repeat(len(tokens), 1)
            grammar.apply_vocab_mask(logits, mask)
            rows = recorder.rows
            probs = torch.softmax(logits[rows] / 0.7, dim=-1)
            sampled = (
                torch.multinomial(
                    probs, 1, generator=torch.Generator().manual_seed(731)
                )
                if rows
                else torch.empty((0, 1), dtype=torch.int64)
            )
            after_traversal = _wrapper_state(grammar)
            for token in commit:
                grammar.accept_token(token)
            probe = torch.full((1, 3), -1, dtype=torch.int32)
            if not grammar.is_terminated():
                grammar.fill_vocab_mask(probe, 0)
            results.append(
                dict(
                    events=recorder.events,
                    mask=mask.tolist(),
                    samples=sampled.tolist(),
                    state_after_traversal=after_traversal,
                    state_after_commit=_wrapper_state(grammar),
                    terminated=grammar.is_terminated(),
                    next_mask=probe.tolist(),
                    logits=logits[rows].tolist(),
                    probs=probs.tolist(),
                    greedy=logits[rows].argmax(dim=-1).tolist(),
                )
            )
        self.assertEqual(results[0], results[1])
        for before, after in zip(originals, inputs):
            self.assertTrue(torch.equal(before, after))
        return results[1]

    def test_signed_words_tail_and_noncontiguous_inputs(self):
        for dtype in (torch.int32, torch.int64):
            for vocab_size in (None, 0, self.VOCAB_SIZE):
                with self.subTest(dtype=dtype, vocab_size=vocab_size):
                    result = self._compare(
                        ([1, 2, 3, 4, -1], [-1] * 5),
                        [999, self.A, self.B, self.C, self.EOS],
                        dtype=dtype,
                        vocab_size=vocab_size,
                        noncontiguous=True,
                        commit=[self.A, self.B, self.C, self.EOS],
                    )
                    self.assertEqual(result["mask"][0][0], -(1 << 31))
                    self.assertEqual(result["mask"][1][1], -(1 << 31))
                    self.assertTrue(result["terminated"])

    def test_nontrivial_probabilities_greedy_and_seeded_sampling(self):
        result = self._compare(
            ([1, 2, 3, 4, -1], [-1] * 5),
            [999, self.A, self.B, self.C, self.EOS],
            choice=True,
            commit=[self.A, self.B, self.C, self.EOS],
        )
        self.assertGreater(sum(prob > 0 for prob in result["probs"][0]), 1)
        self.assertTrue(result["terminated"])

    def test_rejected_and_out_of_vocab_siblings(self):
        for rejected in (1, self.VOCAB_SIZE):
            with self.subTest(rejected=rejected):
                self._compare(
                    ([1, 4, 3, 4, -1], [-1, 2, -1, -1, -1]),
                    [999, rejected, self.A, self.B, self.C],
                    commit=[self.A, self.B, self.C],
                )

    def test_terminated_root_does_not_accept_or_fill(self):
        result = self._compare(([1, -1], [-1, -1]), [999, self.A], terminated=True)
        self.assertEqual([event[0] for event in result["events"]], ["is_terminated"])
        self.assertEqual(result["mask"], [[-1] * 3] * 2)

    def test_reasoner_partial_marker_speculation_and_real_commit(self):
        # Complete the end marker, traverse constrained output and EOS, then
        # unwind across the thinking/generation boundary before a real commit.
        result = self._compare(
            ([1, 2, 3, 4, 5, -1], [-1] * 6),
            [1, 2, self.A, self.B, self.C, self.EOS],
            reasoner=True,
            commit=[2, self.A, self.B, self.C, self.EOS],
        )
        state = result["state_after_traversal"]
        self.assertEqual((state["tokens_in_think"], state["tokens_after_end"]), (0, -1))
        self.assertEqual(state["_matched_think_end_tokens"], 1)
        self.assertEqual(state["grammar"]["accepted_tokens"], [])
        self.assertTrue(result["terminated"])


if __name__ == "__main__":
    unittest.main()
