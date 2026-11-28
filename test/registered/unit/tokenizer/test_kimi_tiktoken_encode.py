"""Unit tests for kimi_tiktoken_encode on a stand-in Kimi TikTokenTokenizer.

The stand-in re-implements the checkpoint's three methods with the same
behavior over a small byte-level ``tiktoken.Encoding``; the source-hash table
is pointed at it for the duration of each test.
"""

import copy
import multiprocessing
import pickle
import random
import sys
import threading
import time
import types
import unittest
from collections import namedtuple
from unittest import mock

import tiktoken

from sglang.srt.environ import envs
from sglang.srt.tokenizer import kimi_tiktoken_encode as kte
from sglang.srt.utils.patch_tokenizer import unpatch_tokenizer
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=60, suite="base-a-test-cpu")

M = 25_000
C = 400_000
WS = [chr(c) for c in range(0x110000) if chr(c).isspace()]

Segment = namedtuple("Segment", ["text", "allow_special"])


def reference_split(s, max_consecutive_slice_len):
    """Per-character loop with the checkpoint's semantics."""
    run_len = 0
    run_is_space = s[0].isspace() if s else False
    start = 0
    for i, ch in enumerate(s):
        is_space = ch.isspace()
        if is_space != run_is_space:
            run_len, run_is_space = 1, is_space
        else:
            run_len += 1
            if run_len > max_consecutive_slice_len:
                yield s[start:i]
                start, run_len = i, 1
    yield s[start:]


class TikTokenTokenizer:
    def __init__(self, model):
        self.model = model

    def _encode_text_piece(self, text, allow_special_tokens=True):
        ids = []
        for i in range(0, len(text), 400_000):
            for substr in self._split_whitespaces_or_nonwhitespaces(
                text[i : i + 400_000], 25_000
            ):
                if allow_special_tokens:
                    ids.extend(self.model.encode(substr, allowed_special="all"))
                else:
                    ids.extend(self.model.encode(substr, disallowed_special=()))
        return ids

    def encode(self, text, allow_special_tokens=True):
        return self._encode_text_piece(text, allow_special_tokens=allow_special_tokens)

    @staticmethod
    def _split_whitespaces_or_nonwhitespaces(s, max_consecutive_slice_len):
        return reference_split(s, max_consecutive_slice_len)

    def _encode_chat_segments(self, segments):
        ids = []
        for segment in segments:
            ids.extend(
                self._encode_text_piece(
                    segment.text, allow_special_tokens=segment.allow_special
                )
            )
        return ids


# The install filter wants a tokenization_kimi module; pickle resolves the
# class through it.
_FAKE_MODULE = "sglang_test_fake_tokenization_kimi"
TikTokenTokenizer.__module__ = _FAKE_MODULE
sys.modules.setdefault(_FAKE_MODULE, types.ModuleType(_FAKE_MODULE))
sys.modules[_FAKE_MODULE].TikTokenTokenizer = TikTokenTokenizer
sys.modules[_FAKE_MODULE].Segment = Segment


def _make_model():
    ranks = {bytes([i]): i for i in range(256)}
    for word in (b"lo", b"rem", b" ip", b"sum", b"  ", b"\n\n"):
        ranks[word] = len(ranks)
    n = len(ranks)
    return tiktoken.Encoding(
        name="kimi-test",
        pat_str=r"""'s|'t| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
        mergeable_ranks=ranks,
        special_tokens={"<|im_end|>": n, "<|im_user|>": n + 1},
    )


MODEL = _make_model()


def _verified_hashes(cls):
    split = vars(cls)["_split_whitespaces_or_nonwhitespaces"].__func__
    return {
        "_split_whitespaces_or_nonwhitespaces": kte._source_sha256(split),
        "_encode_text_piece": kte._source_sha256(cls._encode_text_piece),
        "_encode_chat_segments": kte._source_sha256(cls._encode_chat_segments),
    }


def _install(tokenizer, fast_scan=True, threads=1):
    with (
        envs.SGLANG_KIMI_TOKENIZER_FAST_SCAN.override(fast_scan),
        envs.SGLANG_KIMI_TOKENIZER_ENCODE_THREADS.override(threads),
    ):
        return kte.install_kimi_tiktoken_encode(tokenizer)


def _segments_ids_in_child(payload, queue):
    tokenizer, segments = pickle.loads(payload)
    queue.put((tokenizer._encode_chat_segments(segments), sorted(kte._pools)))


def _long_segments():
    words = "lorem ipsum <|im_end|> dolor " * 3000  # ~87k chars
    return [
        Segment("<|im_user|>", True),
        Segment(words, False),
        Segment("<|im_end|>" + words, True),
        Segment("x" * (M + 7) + " " * (2 * M + 1) + words, False),
        Segment("<|im_end|>", True),
    ]


class _Base(CustomTestCase):
    def setUp(self):
        patcher = mock.patch.dict(
            kte._VERIFIED_SOURCE_SHA256, _verified_hashes(TikTokenTokenizer)
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(kte.shutdown_encode_pools)

    def original(self):
        return TikTokenTokenizer(MODEL)


class TestFastSplit(_Base):
    def assertSameSlices(self, s, m=M):
        self.assertEqual(list(kte.fast_split(s, m)), list(reference_split(s, m)))

    def test_empty_and_short(self):
        for s in ("", "a", " ", "a" * M, " " * M):
            self.assertSameSlices(s)
        self.assertEqual(list(kte.fast_split("", M)), [""])

    def test_run_lengths_around_the_limit(self):
        for n in (M - 1, M, M + 1, 2 * M - 1, 2 * M, 2 * M + 1, 3 * M + 2):
            for ch in ("a", " ", "\u3000", "\U0001f600"):
                self.assertSameSlices(ch * n)
                self.assertSameSlices("b " + ch * n + " c")
        self.assertEqual(len(list(kte.fast_split("a" * (M + 1), M))), 2)
        self.assertEqual(len(list(kte.fast_split("a" * (2 * M), M))), 2)
        self.assertEqual(len(list(kte.fast_split("a" * (2 * M + 1), M))), 3)

    def test_all_unicode_whitespace_and_surrogates_at_400k(self):
        ws = "".join(WS)
        cases = [
            (ws * (C // len(ws) + 1))[:C],
            ("\ud800" * (M + 1) + ws * 1000 + "\udfff" + "a" * M)[:C],
            ("x" * (C - M - 3) + " " * (M + 5))[:C],
            "\u200b" * (M + 1) + "\x85" * (M + 1) + "\x1c\x1d\x1e\x1f" * M,
        ]
        for s in cases:
            self.assertSameSlices(s)

    def test_random_small_limits(self):
        rng = random.Random(0)
        alphabet = ["a", "\u4e2d", "\u200b", "\ud800", "\U00010000"] + WS
        for _ in range(300):
            s = "".join(
                rng.choice(alphabet) * rng.randint(1, 12)
                for _ in range(rng.randint(0, 40))
            )
            for m in (1, 2, 3, 7):
                self.assertSameSlices(s, m)

    def test_non_positive_limit_raises(self):
        for m in (0, -1):
            with self.assertRaises(ValueError):
                list(kte.fast_split("a" * 10, m))


class TestInstall(_Base):
    def test_installs_on_instance_only(self):
        tok, other = self.original(), self.original()
        _install(tok)
        self.assertIs(vars(tok)["_split_whitespaces_or_nonwhitespaces"], kte.fast_split)
        self.assertNotIn("_encode_chat_segments", vars(tok))
        self.assertNotIn("_split_whitespaces_or_nonwhitespaces", vars(other))
        self.assertIsInstance(
            vars(TikTokenTokenizer)["_split_whitespaces_or_nonwhitespaces"],
            staticmethod,
        )

    def test_switches_off_skips_probe(self):
        tok = self.original()
        with mock.patch.object(kte, "_unsupported_reason") as probe:
            _install(tok, fast_scan=False, threads=1)
        probe.assert_not_called()
        self.assertEqual(vars(tok), {"model": MODEL})

    def test_repeated_install_is_idempotent(self):
        tok = self.original()
        _install(tok, threads=2)
        first = dict(vars(tok))
        with self.assertNoLogs(kte.logger, level="WARNING"):
            _install(tok, threads=4)
        self.assertEqual(vars(tok), first)
        self.assertEqual(tok._encode_chat_segments.threads, 2)

    def test_unverified_source_keeps_original(self):
        def _encode_text_piece(self, text, allow_special_tokens=True):
            return self.model.encode(text, allowed_special="all")

        changed = type(
            "TikTokenTokenizer",
            (TikTokenTokenizer,),
            {"_encode_text_piece": _encode_text_piece, "__module__": _FAKE_MODULE},
        )
        tok = changed(MODEL)
        with self.assertLogs(kte.logger, level="WARNING") as logs:
            _install(tok, threads=2)
        self.assertIn("_encode_text_piece source sha256", logs.output[0])
        self.assertEqual(vars(tok), {"model": MODEL})

    def test_existing_instance_override_keeps_original(self):
        tok = self.original()
        own = lambda s, m: iter([s])  # noqa: E731
        tok._split_whitespaces_or_nonwhitespaces = own
        with self.assertLogs(kte.logger, level="WARNING"):
            _install(tok)
        self.assertIs(tok._split_whitespaces_or_nonwhitespaces, own)
        self.assertNotIn(kte._INSTALLED_ATTR, vars(tok))

    def test_non_tiktoken_model_keeps_original(self):
        tok = TikTokenTokenizer(object())
        with self.assertLogs(kte.logger, level="WARNING"):
            _install(tok)
        self.assertNotIn(kte._INSTALLED_ATTR, vars(tok))

    def test_other_tokenizer_untouched(self):
        other = type("OtherTokenizer", (TikTokenTokenizer,), {})(MODEL)
        with self.assertNoLogs(kte.logger, level="WARNING"):
            _install(other, threads=4)
        self.assertEqual(vars(other), {"model": MODEL})

    def test_bad_thread_count_falls_back_to_one(self):
        tok = self.original()
        with self.assertLogs(kte.logger, level="WARNING"):
            _install(tok, threads=0)
        self.assertNotIn("_encode_chat_segments", vars(tok))
        self.assertIn("_split_whitespaces_or_nonwhitespaces", vars(tok))

    def test_unpatch_removes_only_installed_attrs(self):
        tok = self.original()
        tok.unrelated = 1
        _install(tok, threads=2)
        unpatch_tokenizer(tok)
        self.assertEqual(vars(tok), {"model": MODEL, "unrelated": 1})


class TestParallelEncode(_Base):
    def test_ids_match_original_for_every_mode(self):
        segments = _long_segments()
        text = "".join(s.text for s in segments)
        ref = self.original()
        want_segments = ref._encode_chat_segments(segments)
        want_text = {a: ref.encode(text, allow_special_tokens=a) for a in (True, False)}
        self.assertNotEqual(want_text[True], want_text[False])
        for fast_scan, threads in ((True, 1), (False, 2), (True, 2), (True, 4)):
            tok = _install(self.original(), fast_scan=fast_scan, threads=threads)
            self.assertEqual(tok._encode_chat_segments(segments), want_segments)
            for allow, want in want_text.items():
                self.assertEqual(tok.encode(text, allow_special_tokens=allow), want)

    def test_segments_longer_than_a_chunk(self):
        long = ("lorem ipsum " * (C // 12 + 3000)) + "<|im_end|>"
        segments = [Segment(long, True), Segment(long, False)]
        want = self.original()._encode_chat_segments(segments)
        tok = _install(self.original(), threads=4)
        self.assertEqual(tok._encode_chat_segments(segments), want)

    def _pool_submits(self, segments):
        tok = _install(self.original(), threads=2)
        want = self.original()._encode_chat_segments(segments)
        with mock.patch.object(kte, "_encode_pool", wraps=kte._encode_pool) as pool:
            self.assertEqual(tok._encode_chat_segments(segments), want)
        return pool.call_count

    def test_threshold(self):
        big = kte._PARALLEL_MIN_PIECE_CHARS
        # Under 2 x threshold in total: the class's own method.
        self.assertEqual(self._pool_submits([Segment("a b " * 100, False)]), 0)
        # One long slice plus short ones: inline.
        one = [Segment("a" * big, False)] + [Segment("x y", False)] * (big // 3 + 1)
        self.assertEqual(self._pool_submits(one), 0)
        # Two slices at the threshold: pooled.
        two = [Segment("a" * big, False), Segment("b" * big, True)]
        self.assertEqual(self._pool_submits(two), 1)
        # One run cut into two long slices at 25k: pooled.
        self.assertEqual(self._pool_submits([Segment("c" * (M + big), False)]), 1)

    def test_output_order_when_pieces_finish_out_of_order(self):
        tok = _install(self.original(), threads=4)
        segments = [Segment(ch * (M - 1), False) for ch in "abcdef"]
        want = self.original()._encode_chat_segments(segments)
        threads = set()

        class SlowFirst:
            def encode(self, text, **kwargs):
                threads.add(threading.current_thread().name)
                if text[0] == "a":
                    time.sleep(0.2)
                return MODEL.encode(text, **kwargs)

        tok.model = SlowFirst()
        self.assertEqual(tok._encode_chat_segments(segments), want)
        self.assertTrue(any(n.startswith("kimi-tiktoken-encode") for n in threads))

    def test_worker_error_propagates_and_pool_recovers(self):
        tok = _install(self.original(), threads=2)
        segments = [Segment("a" * M, False), Segment("b" * M, False)]

        class Failing:
            def encode(self, text, **kwargs):
                if text[0] == "b":
                    raise RuntimeError("boom")
                return MODEL.encode(text, **kwargs)

        tok.model = Failing()
        with self.assertRaisesRegex(RuntimeError, "boom"):
            tok._encode_chat_segments(segments)
        tok.model = MODEL
        want = self.original()._encode_chat_segments(segments)
        self.assertEqual(tok._encode_chat_segments(segments), want)
        kte.shutdown_encode_pools()
        self.assertEqual(kte._pools, {})
        self.assertEqual(tok._encode_chat_segments(segments), want)

    def test_pickle_and_deepcopy_roundtrip(self):
        tok = _install(self.original(), threads=2)
        segments = _long_segments()
        want = tok._encode_chat_segments(segments)  # pool now exists
        self.assertTrue(kte._pools)
        for clone in (pickle.loads(pickle.dumps(tok)), copy.deepcopy(tok)):
            self.assertIsNot(clone, tok)
            self.assertIs(clone._encode_chat_segments.tokenizer, clone)
            self.assertIs(clone._split_whitespaces_or_nonwhitespaces, kte.fast_split)
            self.assertEqual(clone._encode_chat_segments(segments), want)
            unpatch_tokenizer(clone)
            self.assertEqual(vars(clone).keys(), {"model"})

    def _run_child(self, method, tok, segments):
        ctx = multiprocessing.get_context(method)
        queue = ctx.Queue()
        proc = ctx.Process(
            target=_segments_ids_in_child,
            args=(pickle.dumps((tok, segments)), queue),
        )
        proc.start()
        ids, pools_after = queue.get(timeout=120)
        proc.join(timeout=60)
        self.assertEqual(proc.exitcode, 0)
        return ids, pools_after

    def test_spawn_and_fork_children_use_their_own_pool(self):
        tok = _install(self.original(), threads=2)
        segments = _long_segments()
        want = tok._encode_chat_segments(segments)
        parent_pool = kte._pools[2]
        for method in ("spawn", "fork"):
            ids, pools_after = self._run_child(method, tok, segments)
            self.assertEqual(ids, want)
            self.assertEqual(list(pools_after), [2])
        self.assertIs(kte._pools[2], parent_pool)


if __name__ == "__main__":
    unittest.main()
