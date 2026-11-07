# SPDX-License-Identifier: Apache-2.0
"""CPU identity tests independent of the model-serving import tree."""

import copy
import importlib.util
import random
import sys
import unittest
from array import array
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

_SOURCE = (
    Path(__file__).resolve().parents[4]
    / "python/sglang/srt/cache_accounting/multimodal.py"
)
_SPEC = importlib.util.spec_from_file_location("_cache_multimodal_test", _SOURCE)
media = importlib.util.module_from_spec(_SPEC)
with patch.dict(sys.modules, {_SPEC.name: media}):
    _SPEC.loader.exec_module(media)

if "sglang.test.ci.ci_register" in sys.modules:
    from sglang.test.ci.ci_register import register_cpu_ci
else:

    def register_cpu_ci(**kwargs):
        pass


register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def item(feature=b"pixels", **kwargs):
    defaults = dict(
        modality="image",
        feature=feature,
        precomputed_embeddings=None,
        model_specific_data={"grid": [1, 2, 3]},
        offsets=[(1, 2)],
        hash=1,
        pad_value=1000001,
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def prepare(*items):
    output = SimpleNamespace(mm_items=list(items))
    media.prepare_media_digests(output)
    return output


class TestMediaDigests(unittest.TestCase):
    def test_same_content_survives_different_pad_and_hash_values(self):
        first = item(hash=17, pad_value=1000001)
        second = item(hash=999, pad_value=1000002)
        prepare(first, second)
        self.assertEqual(
            first.model_specific_data["cache_accounting_digest"],
            second.model_specific_data["cache_accounting_digest"],
        )

    def test_content_modality_and_metadata_change_digest(self):
        inputs = [
            item(),
            item(b"different"),
            item(modality="video"),
            item(modality="audio"),
            item(model_specific_data={"grid": [1, 3, 2]}),
        ]
        prepare(*inputs)
        self.assertEqual(
            len({i.model_specific_data["cache_accounting_digest"] for i in inputs}),
            len(inputs),
        )

    def test_embeddings_are_supported_and_differ_from_features(self):
        feature = item(b"same")
        embedding = item(None, precomputed_embeddings=b"same")
        prepare(feature, embedding)
        self.assertNotEqual(
            feature.model_specific_data["cache_accounting_digest"],
            embedding.model_specific_data["cache_accounting_digest"],
        )

    def test_trusted_artifact_does_not_need_or_hash_features(self):
        identity = media.CacheArtifactIdentity("sha256:" + "a" * 64)
        first = item(
            None, model_specific_data={"cache_accounting_artifact_key": identity}
        )
        second = item(
            object(), model_specific_data={"cache_accounting_artifact_key": identity}
        )
        prepare(first, second)
        self.assertEqual(
            first.model_specific_data["cache_accounting_digest"],
            second.model_specific_data["cache_accounting_digest"],
        )

    def test_external_artifact_assertions_and_old_digest_are_not_trusted(self):
        first = item(
            b"first",
            model_specific_data={
                "cache_accounting_artifact_key": "sha256:" + "a" * 64,
                "cache_accounting_digest": "f" * 64,
            },
        )
        second = item(
            b"second",
            model_specific_data={
                "cache_accounting_artifact_key": {"artifact_key": "sha256:" + "a" * 64},
                "cache_accounting_digest": "f" * 64,
            },
        )
        prepare(first, second)
        self.assertNotEqual(
            first.model_specific_data["cache_accounting_digest"],
            second.model_specific_data["cache_accounting_digest"],
        )
        with self.assertRaises(ValueError):
            prepare(
                item(
                    None,
                    model_specific_data={
                        "cache_accounting_artifact_key": "sha256:" + "a" * 64
                    },
                )
            )

    def test_preparation_is_atomic_and_rejects_missing_or_ambiguous_features(self):
        first = item()
        for invalid in (
            item(None),
            item(b"x", precomputed_embeddings=b"y"),
            item(object()),
        ):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                prepare(first, invalid)
            self.assertNotIn("cache_accounting_digest", first.model_specific_data)

    def test_failed_recomputation_clears_all_stale_media_digests(self):
        first = item()
        second = item(None)
        for current in (first, second):
            current.model_specific_data["cache_accounting_digest"] = "f" * 64
        with self.assertRaises(ValueError):
            prepare(first, second)
        for current in (first, second):
            self.assertNotIn("cache_accounting_digest", current.model_specific_data)
        self.assertEqual(
            media.freeze_cache_input_metadata(
                [1, 2, 3, 4], SimpleNamespace(mm_items=[first])
            ),
            {"error": "invalid_multimodal_metadata"},
        )

    def test_numpy_shape_dtype_and_bytes_are_all_part_of_identity(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("NumPy unavailable")
        inputs = [
            item(np.array([1, 2], dtype="uint8")),
            item(np.array([[1, 2]], dtype="uint8")),
            item(np.array([1, 2], dtype="int8")),
            item(np.array([1, 3], dtype="uint8")),
        ]
        prepare(*inputs)
        self.assertEqual(
            len({i.model_specific_data["cache_accounting_digest"] for i in inputs}), 4
        )
        with self.assertRaises(ValueError):
            prepare(item(np.array([object()], dtype=object)))

    def test_scalar_and_zero_length_numpy_features_are_supported(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("NumPy unavailable")
        inputs = [
            item(np.array(1, dtype="uint8")),
            item(np.array([1], dtype="uint8")),
            item(np.empty((0, 2), dtype="uint8")),
        ]
        prepare(*inputs)
        self.assertEqual(
            len({i.model_specific_data["cache_accounting_digest"] for i in inputs}), 3
        )

    def test_unknown_transport_or_metadata_fails_closed(self):
        for candidate in (
            item(object()),
            item(model_specific_data={"unknown": object()}),
        ):
            with self.assertRaises(ValueError):
                prepare(candidate)

    def test_deferred_preprocessing_settings_are_part_of_identity(self):
        @dataclass(frozen=True)
        class Deferred:
            resize: int

        first = item(
            model_specific_data={"kimi_k3_deferred_preprocessing": Deferred(10)}
        )
        second = item(
            model_specific_data={"kimi_k3_deferred_preprocessing": Deferred(20)}
        )
        prepare(first, second)
        self.assertNotEqual(
            first.model_specific_data["cache_accounting_digest"],
            second.model_specific_data["cache_accounting_digest"],
        )

    def test_transport_control_flags_do_not_change_identity(self):
        first = item()
        second = item(
            model_specific_data={
                "grid": [1, 2, 3],
                "_sglang_defer_cuda_ipc_feature_reconstruction": True,
            }
        )
        prepare(first, second)
        self.assertEqual(
            first.model_specific_data["cache_accounting_digest"],
            second.model_specific_data["cache_accounting_digest"],
        )

    def test_shared_memory_transport_hashes_payload_and_not_handle(self):
        def must_not_materialize(self):
            raise AssertionError("Accounting must not consume SHM ownership")

        transport = type(
            "ShmPointerMMData",
            (),
            {
                "__module__": "sglang.srt.managers.mm_utils",
                "tensor": b"pixels",
                "materialize": must_not_materialize,
            },
        )()
        first = item()
        second = item(transport)
        prepare(first, second)
        self.assertEqual(
            first.model_specific_data["cache_accounting_digest"],
            second.model_specific_data["cache_accounting_digest"],
        )

    def test_cuda_transport_identity_never_acknowledges_a_worker_lease(self):
        def must_not_reconstruct(self, *args, **kwargs):
            raise AssertionError("Accounting must not acknowledge a CUDA IPC consumer")

        transport = type(
            "CudaIpcTensorTransportProxy",
            (),
            {
                "__module__": "sglang.srt.multimodal.transport.cuda_ipc",
                "reconstruct_tensor": None,
                "reconstruct_on_target_device": must_not_reconstruct,
            },
        )()
        with self.assertRaises(ValueError):
            prepare(item(transport))
        transport.reconstruct_tensor = b"pixels"
        plain, ready = item(), item(transport)
        prepare(plain, ready)
        self.assertEqual(
            plain.model_specific_data["cache_accounting_digest"],
            ready.model_specific_data["cache_accounting_digest"],
        )

    def test_producer_shared_memory_is_read_without_consuming_the_owner(self):
        from multiprocessing import shared_memory

        def must_not_consume(self):
            raise AssertionError("Accounting must not consume SHM ownership")

        owner = shared_memory.SharedMemory(create=True, size=3)
        try:
            owner.buf[:3] = b"abc"
            cls = type(
                "ShmPointerMMData",
                (),
                {
                    "__module__": "sglang.srt.managers.mm_utils",
                    "tensor": None,
                    "shape": (3,),
                    "dtype": "torch.uint8",
                    "shm_name": owner.name,
                    "materialize": must_not_consume,
                    "close_and_unlink": must_not_consume,
                },
            )
            transport = cls()
            first = item(transport)
            prepare(first)
            digest = first.model_specific_data["cache_accounting_digest"]
            prepare(first)
            self.assertEqual(
                first.model_specific_data["cache_accounting_digest"], digest
            )
            self.assertIsNone(transport.tensor)
            owner.buf[0] = ord("z")
            prepare(first)
            self.assertNotEqual(
                first.model_specific_data["cache_accounting_digest"], digest
            )
            # Another owner-side access proves the temporary reader did not unlink.
            check = shared_memory.SharedMemory(name=owner.name, create=False)
            check.close()
            transport.shape = (owner.size + 1,)
            with self.assertRaises(ValueError):
                prepare(first)
        finally:
            owner.close()
            owner.unlink()

    def test_torch_feature_shape_dtype_and_bytes_are_hashed(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch unavailable")
        inputs = [
            item(torch.tensor([1, 2], dtype=torch.uint8)),
            item(torch.tensor([[1, 2]], dtype=torch.uint8)),
            item(torch.tensor([1, 2], dtype=torch.int8)),
            item(torch.tensor([1, 3], dtype=torch.uint8)),
        ]
        prepare(*inputs)
        self.assertEqual(
            len({i.model_specific_data["cache_accounting_digest"] for i in inputs}), 4
        )

    def test_engine_torch_subclass_is_hashed_as_tensor(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch unavailable")

        class EngineTensor(torch.Tensor):
            pass

        EngineTensor.__module__ = "sglang.srt.managers.mm_utils"
        tensor = torch.tensor([1, 2, 3], dtype=torch.uint8)
        first = item(tensor)
        second = item(tensor.as_subclass(EngineTensor))
        prepare(first, second)
        self.assertEqual(
            first.model_specific_data["cache_accounting_digest"],
            second.model_specific_data["cache_accounting_digest"],
        )


def _freeze_metadata_scalar_reference(input_ids, mm_inputs):
    """Preserve the pre-vectorization contract without calling production helpers."""
    try:
        ids = array("q", input_ids)
        items = getattr(mm_inputs, "mm_items", None)
        if not items:
            raise ValueError("Missing media items")
        spans = []
        pad_values = set()
        for current in items:
            digest = current.model_specific_data["cache_accounting_digest"]
            if isinstance(digest, str) and digest.startswith("sha256:"):
                digest = digest[7:]
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(c not in "0123456789abcdefABCDEF" for c in digest)
            ):
                raise ValueError("Invalid digest")
            digest = bytes.fromhex(digest).hex()
            offsets = getattr(current, "offsets", None)
            if not offsets:
                raise ValueError("Missing offsets")
            pad = getattr(current, "pad_value", None)
            if type(pad) is not int:
                raise ValueError("Invalid pad type")
            pad_values.add(pad)
            for start, inclusive_end in offsets:
                if (
                    type(start) is not int
                    or type(inclusive_end) is not int
                    or not 0 <= start <= inclusive_end < len(ids)
                ):
                    raise ValueError("Invalid offsets")
                if any(ids[index] != pad for index in range(start, inclusive_end + 1)):
                    raise ValueError("Media interval contains other tokens")
                spans.append((start, inclusive_end + 1, digest))
        spans.sort(key=lambda span: (span[0], span[1]))
        previous_end = 0
        for start, end, _ in spans:
            if start < previous_end:
                raise ValueError("Overlapping intervals")
            previous_end = end
        span_index = 0
        for index, token in enumerate(ids):
            while span_index < len(spans) and index >= spans[span_index][1]:
                span_index += 1
            is_media = span_index < len(spans) and index >= spans[span_index][0]
            if not is_media and (token in pad_values or not 0 <= token <= 0xFFFFFFFF):
                raise ValueError("Invalid uncovered token")
        return {"input_ids": ids, "media_spans": spans}
    except Exception:
        return {"error": "invalid_multimodal_metadata"}


class TestFreezeMetadata(unittest.TestCase):
    @staticmethod
    def _metadata_item(offsets, pad, digest="a" * 64):
        return item(
            offsets=offsets,
            pad_value=pad,
            model_specific_data={"cache_accounting_digest": digest},
        )

    def _assert_matches_scalar(self, tokens, items, *, valid):
        inputs = SimpleNamespace(mm_items=items)
        expected = _freeze_metadata_scalar_reference(tokens, inputs)
        self.assertEqual("error" not in expected, valid)
        actual = media.freeze_cache_input_metadata(tokens, inputs)
        self.assertEqual(actual, expected)
        if valid:
            self.assertIsInstance(actual["input_ids"], array)
            self.assertEqual(actual["input_ids"].typecode, "q")
        return actual

    def test_freeze_copies_final_ids_and_converts_inclusive_offsets(self):
        output = prepare(item(offsets=[(1, 2), (4, 4)]))
        tokens = array("q", [7, 1000001, 1000001, 8, 1000001])
        frozen = media.freeze_cache_input_metadata(tokens, output)
        digest = output.mm_items[0].model_specific_data["cache_accounting_digest"]
        self.assertEqual(frozen["media_spans"], [(1, 3, digest), (4, 5, digest)])
        tokens[0] = 9
        self.assertEqual(frozen["input_ids"][0], 7)
        output.mm_items[0].feature = None
        self.assertEqual(
            media.freeze_cache_input_metadata(frozen["input_ids"], output), frozen
        )

    def test_freeze_rejects_missing_digest_offsets_bounds_or_overlap(self):
        candidates = [SimpleNamespace(mm_items=[]), SimpleNamespace(mm_items=[item()])]
        for offsets in (
            None,
            [],
            [(0, 5)],
            [(-1, 0)],
            [(True, 1)],
            [(2, 1)],
            [(0, 1), (1, 2)],
        ):
            candidates.append(prepare(item(offsets=offsets)))
        for output in candidates:
            with self.subTest(output=output):
                self.assertEqual(
                    media.freeze_cache_input_metadata([1, 2, 3], output),
                    {"error": "invalid_multimodal_metadata"},
                )

    def test_freeze_does_not_swallow_keyboard_interrupt(self):
        class Interrupted:
            @property
            def mm_items(self):
                raise KeyboardInterrupt

        with self.assertRaises(KeyboardInterrupt):
            media.freeze_cache_input_metadata([1], Interrupted())

    def test_freeze_rejects_pad_tokens_not_covered_by_offsets(self):
        output = prepare(item(offsets=[(1, 1)]))
        self.assertEqual(
            media.freeze_cache_input_metadata([7, 1000001, 1000001], output),
            {"error": "invalid_multimodal_metadata"},
        )

    def test_freeze_rejects_offsets_that_would_hide_real_text(self):
        output = prepare(item(offsets=[(1, 3)]))
        self.assertEqual(
            media.freeze_cache_input_metadata([7, 1000001, 1000001, 42], output),
            {"error": "invalid_multimodal_metadata"},
        )

    def test_freeze_rejects_absent_final_pad_identity(self):
        output = prepare(item(pad_value=None))
        self.assertEqual(
            media.freeze_cache_input_metadata([7, 99, 99], output),
            {"error": "invalid_multimodal_metadata"},
        )

    def test_freeze_uses_final_expansion_length_and_large_media_ids(self):
        output = prepare(item(offsets=[(1, 2)], pad_value=1 << 40))
        frozen = media.freeze_cache_input_metadata([7, 1 << 40, 1 << 40, 8], output)
        self.assertEqual(len(frozen["input_ids"]), 4)
        self.assertEqual(frozen["media_spans"][0][:2], (1, 3))

    def test_signed_int64_media_and_uint32_text_boundaries(self):
        for pad in (-(1 << 63), -1, 0, 1, 0xFFFFFFFF, 1 << 32, 1 << 40, (1 << 63) - 1):
            with self.subTest(pad=pad):
                text = [value for value in (0, 1, 0xFFFFFFFF) if value != pad]
                self._assert_matches_scalar(
                    text + [pad],
                    [self._metadata_item([(len(text), len(text))], pad)],
                    valid=True,
                )
        for token in (-(1 << 63), -1, 1 << 32, (1 << 63) - 1):
            with self.subTest(uncovered=token):
                self._assert_matches_scalar(
                    [7, token], [self._metadata_item([(0, 0)], 7)], valid=False
                )
        # Neither input conversion nor NumPy scalar comparison may wrap these.
        for pad in (-(1 << 63) - 1, 1 << 63, (1 << 64) + 7):
            for token in (pad, -(1 << 63), (1 << 63) - 1, 7):
                with self.subTest(oversized_pad=pad, token=token):
                    self._assert_matches_scalar(
                        [token], [self._metadata_item([(0, 0)], pad)], valid=False
                    )

    def test_offsets_and_pad_require_builtin_ints(self):
        for start, end in ((False, 0), (0, False), (0.0, 0), (0, 0.0), ("0", 0)):
            with self.subTest(start=start, end=end):
                self._assert_matches_scalar(
                    [7], [self._metadata_item([(start, end)], 7)], valid=False
                )
        for pad, token in ((True, 1), (False, 0), (7.0, 7), ("7", 7)):
            with self.subTest(pad=pad):
                self._assert_matches_scalar(
                    [token], [self._metadata_item([(0, 0)], pad)], valid=False
                )
        # Token conversion deliberately has array('q') semantics, unlike offsets.
        self._assert_matches_scalar(
            [False, True, 7], [self._metadata_item([(2, 2)], 7)], valid=True
        )

    def test_numpy_integer_tokens_but_not_offsets_or_pad(self):
        self._assert_matches_scalar(
            np.array([0, 7, 0xFFFFFFFF], dtype=np.int64),
            [self._metadata_item([(1, 1)], 7)],
            valid=True,
        )
        for scalar in (np.int64, np.uint64, np.int32):
            for offsets, pad in (
                ([(scalar(0), 0)], 7),
                ([(0, scalar(0))], 7),
                ([(0, 0)], scalar(7)),
            ):
                with self.subTest(scalar=scalar, offsets=offsets, pad=pad):
                    self._assert_matches_scalar(
                        [7], [self._metadata_item(offsets, pad)], valid=False
                    )
        self._assert_matches_scalar(
            np.array([7.0]), [self._metadata_item([(0, 0)], 7)], valid=False
        )
        self._assert_matches_scalar(
            np.array([0xFFFFFFFF, 7], dtype=np.uint64),
            [self._metadata_item([(1, 1)], 7)],
            valid=True,
        )
        self._assert_matches_scalar(
            np.array([1 << 63], dtype=np.uint64),
            [self._metadata_item([(0, 0)], 1 << 63)],
            valid=False,
        )

    def test_float_tokens_are_not_silently_converted(self):
        for tokens in ([7.0], [7.5], [float("nan")], [float("inf")], array("d", [7.0])):
            with self.subTest(tokens=tokens):
                self._assert_matches_scalar(
                    tokens, [self._metadata_item([(0, 0)], 7)], valid=False
                )

    def test_adjacent_unsorted_spans_keep_distinct_digests_for_same_pad(self):
        first = self._metadata_item([(3, 3), (0, 0)], -1, "sha256:" + "AB" * 32)
        second = self._metadata_item([(1, 2)], -1, "cd" * 32)
        result = self._assert_matches_scalar([-1] * 4, [second, first], valid=True)
        self.assertEqual(
            result["media_spans"],
            [(0, 1, "ab" * 32), (1, 3, "cd" * 32), (3, 4, "ab" * 32)],
        )
        for offsets in ([(0, 1), (0, 1)], [(0, 1), (1, 2)], [(0, 2), (1, 1)]):
            with self.subTest(offsets=offsets):
                self._assert_matches_scalar(
                    [7] * 3, [self._metadata_item(offsets, 7)], valid=False
                )
        self._assert_matches_scalar(
            [7],
            [
                self._metadata_item([(0, 0)], 7),
                self._metadata_item([(0, 0)], 7, "b" * 64),
            ],
            valid=False,
        )

    def test_digest_validation_and_canonicalization_match_scalar(self):
        for digest in ("AB" * 32, "sha256:" + "AB" * 32):
            result = self._assert_matches_scalar(
                [7], [self._metadata_item([(0, 0)], 7, digest)], valid=True
            )
            self.assertEqual(result["media_spans"], [(0, 1, "ab" * 32)])
        for digest in (None, b"a" * 64, "a" * 63, "a" * 65, "g" * 64, "sha256:"):
            with self.subTest(digest=digest):
                self._assert_matches_scalar(
                    [7], [self._metadata_item([(0, 0)], 7, digest)], valid=False
                )

    def test_invalid_tokens_in_incomplete_cache_page_are_still_rejected(self):
        # Accounting uses 256-token pages, but the frozen snapshot validates all ids.
        for length in (1, 255, 256, 257, 511, 513):
            tokens = [3] * length
            tokens[0] = 7
            current = self._metadata_item([(0, 0)], 7)
            self._assert_matches_scalar(tokens, [current], valid=True)
            for invalid_tail in (-1, 7, 1 << 32):
                with self.subTest(length=length, tail=invalid_tail):
                    self._assert_matches_scalar(
                        tokens + [invalid_tail], [current], valid=False
                    )

    def test_frozen_array_and_spans_are_independent_snapshots(self):
        tokens = array("q", [3, -1, -1, 4])
        current = self._metadata_item([(1, 2)], -1)
        result = self._assert_matches_scalar(tokens, [current], valid=True)
        self.assertIsNot(result["input_ids"], tokens)
        result["input_ids"][0] = 9
        self.assertEqual(tokens[0], 3)
        tokens[3] = 10
        current.offsets[:] = [(0, 3)]
        current.model_specific_data["cache_accounting_digest"] = "b" * 64
        self.assertEqual(list(result["input_ids"]), [9, -1, -1, 4])
        self.assertEqual(result["media_spans"], [(1, 3, "a" * 64)])

    def test_deterministic_valid_and_invalid_cases_match_scalar_reference(self):
        rng = random.Random(20261103)
        pad_candidates = (
            -(1 << 63),
            -1,
            0,
            1,
            0xFFFFFFFF,
            1 << 32,
            1 << 40,
            (1 << 63) - 1,
        )
        for case in range(256):
            length = rng.randint(1, 768)
            span_count = rng.randint(1, min(length, 8))
            boundaries = (
                [0] + sorted(rng.sample(range(1, length), span_count - 1)) + [length]
            )
            item_count = rng.randint(1, span_count)
            owners = list(range(item_count)) + [
                rng.randrange(item_count) for _ in range(span_count - item_count)
            ]
            rng.shuffle(owners)
            pads = [rng.choice(pad_candidates) for _ in range(item_count)]
            text_choices = [
                token for token in (0, 1, 2, 3, 42, 0xFFFFFFFF) if token not in pads
            ]
            tokens = [rng.choice(text_choices) for _ in range(length)]
            items = [
                self._metadata_item([], pad, f"{rng.getrandbits(256):064x}")
                for pad in pads
            ]
            for begin, end, owner in zip(boundaries, boundaries[1:], owners):
                start = rng.randrange(begin, end)
                stop = rng.randrange(start, end)
                items[owner].offsets.append((start, stop))
                tokens[start : stop + 1] = [pads[owner]] * (stop - start + 1)
            for current in items:
                rng.shuffle(current.offsets)
            rng.shuffle(items)
            with self.subTest(case=case, variant="valid"):
                source = array("q", tokens) if case % 2 else tokens
                self._assert_matches_scalar(source, items, valid=True)

            changed = list(tokens)
            start, _ = items[0].offsets[0]
            changed[start] = 0 if items[0].pad_value != 0 else 1
            duplicate = copy.deepcopy(items)
            duplicate[0].offsets.append(duplicate[0].offsets[0])
            out_of_bounds = copy.deepcopy(items)
            out_of_bounds[0].offsets[0] = (start, length)
            invalid_cases = (
                ("covered-token-mismatch", changed, items),
                ("uncovered-pad", tokens + [items[0].pad_value], items),
                ("uncovered-uint32-overflow", tokens + [1 << 32], items),
                ("duplicate-span", tokens, duplicate),
                ("out-of-bounds", tokens, out_of_bounds),
            )
            for variant, invalid_tokens, invalid_items in invalid_cases:
                with self.subTest(case=case, variant=variant):
                    self._assert_matches_scalar(
                        invalid_tokens, invalid_items, valid=False
                    )

    def test_thousands_of_offsets_on_one_item_match_scalar(self):
        pad = 1 << 40
        for span_count in (1024, 2048):
            tokens = [11] * (4 * span_count + 1)
            offsets = [(4 * index + 1, 4 * index + 2) for index in range(span_count)]
            for start, end in offsets:
                tokens[start : end + 1] = [pad, pad]
            random.Random(span_count).shuffle(offsets)
            current = self._metadata_item(offsets, pad)
            with self.subTest(span_count=span_count, variant="valid"):
                result = self._assert_matches_scalar(tokens, [current], valid=True)
                self.assertEqual(
                    result["media_spans"],
                    [
                        (4 * index + 1, 4 * index + 3, "a" * 64)
                        for index in range(span_count)
                    ],
                )
            wrong_media = list(tokens)
            wrong_media[4 * (span_count // 2) + 2] = 11
            uncovered_pad = list(tokens)
            uncovered_pad[4 * (span_count // 2)] = pad
            for variant, changed in (
                ("wrong-media", wrong_media),
                ("text-pad", uncovered_pad),
            ):
                with self.subTest(span_count=span_count, variant=variant):
                    self._assert_matches_scalar(changed, [current], valid=False)

    def test_thousands_of_shuffled_items_keep_each_pad_and_digest(self):
        for span_count in (1024, 2048):
            tokens = [11] * (4 * span_count + 1)
            items = []
            expected_spans = []
            for index in range(span_count):
                pad = 10000 + index
                digest = f"{index + 1:064x}"
                start = 4 * index + 1
                tokens[start : start + 2] = [pad, pad]
                items.append(self._metadata_item([(start, start + 1)], pad, digest))
                expected_spans.append((start, start + 2, digest))
            random.Random(span_count).shuffle(items)
            with self.subTest(span_count=span_count, variant="valid"):
                result = self._assert_matches_scalar(
                    array("q", tokens), items, valid=True
                )
                self.assertEqual(result["media_spans"], expected_spans)
            # Both values are valid media pads elsewhere. A global membership
            # check alone must not accept a token assigned to the wrong span.
            wrong_media = list(tokens)
            wrong_media[4 * (span_count // 2) + 1] = 10000
            uncovered_pad = list(tokens)
            uncovered_pad[-1] = 10000 + span_count - 1
            for variant, changed in (
                ("another-item-pad", wrong_media),
                ("text-pad", uncovered_pad),
            ):
                with self.subTest(span_count=span_count, variant=variant):
                    self._assert_matches_scalar(changed, items, valid=False)

    def test_adjacent_span_boundaries_keep_sorted_pad_and_digest_association(self):
        pads = [-1, -1, 0, 0xFFFFFFFF, 1 << 40, 1 << 40, 7, -(1 << 63), (1 << 63) - 1]
        tokens = []
        items = []
        expected_spans = []
        for index, pad in enumerate(pads):
            start = len(tokens)
            tokens.extend([pad] * (index % 3 + 1))
            digest = f"{index + 1:064x}"
            items.append(self._metadata_item([(start, len(tokens) - 1)], pad, digest))
            expected_spans.append((start, len(tokens), digest))
        random.Random(20261104).shuffle(items)
        result = self._assert_matches_scalar(tokens, items, valid=True)
        self.assertEqual(result["media_spans"], expected_spans)
        # Swapping just across an adjacent boundary cannot be hidden by the
        # covered mask: the expected pad changes exactly at that boundary.
        for index in range(1, len(pads)):
            if pads[index - 1] == pads[index]:
                continue
            boundary = expected_spans[index][0]
            for position, wrong_pad in (
                (boundary - 1, pads[index]),
                (boundary, pads[index - 1]),
            ):
                with self.subTest(boundary=boundary, position=position):
                    changed = list(tokens)
                    changed[position] = wrong_pad
                    self._assert_matches_scalar(changed, items, valid=False)

    def test_iterator_inputs_preserve_scalar_empty_iterator_semantics(self):
        # Iterators are truthy even when empty. Use fresh instances for each
        # call so the oracle cannot consume the production function's input.
        cases = [
            ([], lambda: iter(()), True),
            ([0, 0xFFFFFFFF], lambda: iter(()), True),
            ([-1], lambda: iter(()), False),
            ([3], lambda: [], False),
            ([3], lambda: [self._metadata_item(iter(()), 7)], True),
            ([7], lambda: [self._metadata_item(iter(()), 7)], False),
            ([3], lambda: [self._metadata_item([], 7)], False),
            ([3], lambda: [self._metadata_item(iter(()), 1 << 80)], True),
            ([3], lambda: [self._metadata_item(iter(()), True)], False),
            ([3], lambda: [self._metadata_item(iter(()), 7, "bad")], False),
            ([3, 7], lambda: iter([self._metadata_item(iter([(1, 1)]), 7)]), True),
        ]
        for index, (tokens, make_items, valid) in enumerate(cases):
            with self.subTest(case=index):
                expected = _freeze_metadata_scalar_reference(
                    tokens, SimpleNamespace(mm_items=make_items())
                )
                self.assertEqual("error" not in expected, valid)
                actual = media.freeze_cache_input_metadata(
                    tokens, SimpleNamespace(mm_items=make_items())
                )
                self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
