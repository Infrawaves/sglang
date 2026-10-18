# SPDX-License-Identifier: Apache-2.0
"""CPU identity tests independent of the model-serving import tree."""

import importlib.util
import sys
import unittest
from array import array
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

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


class TestFreezeMetadata(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
