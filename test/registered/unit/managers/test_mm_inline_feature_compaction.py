"""Inline (pickled) multimodal features must not carry their batch's storage.

A CPU preprocessor that splits one batched ``pixel_values`` into per-item
features hands out views of a single storage. Pickle serializes a tensor's
whole storage, so N such views pickle -- and unpickle -- N copies of the batch.
Every path that pickles features with the request must compact them first.
"""

import pickle
import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.io_struct import TokenizedEmbeddingReqInput  # noqa: E402
from sglang.srt.managers.mm_utils import (  # noqa: E402
    ShmPointerMMData,
    _wrap_shm_or_inline,
    wrap_shm_features,
)
from sglang.srt.managers.schedule_batch import (  # noqa: E402
    Modality,
    MultimodalDataItem,
    MultimodalProcessorOutput,
)
from sglang.srt.multimodal.transport import to_compact_cpu  # noqa: E402
from sglang.srt.sampling.sampling_params import SamplingParams  # noqa: E402

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_ITEMS = 8
_ROWS_PER_ITEM = 64
_ROW = 588


def _split_batch():
    batch = torch.arange(_ITEMS * _ROWS_PER_ITEM * _ROW, dtype=torch.float32).view(
        _ITEMS * _ROWS_PER_ITEM, _ROW
    )
    return batch, list(batch.split([_ROWS_PER_ITEM] * _ITEMS))


def _owns_exactly_its_bytes(tensor: torch.Tensor) -> bool:
    return (
        tensor.storage_offset() == 0
        and tensor.untyped_storage().nbytes() == tensor.numel() * tensor.element_size()
    )


def _request(features, embeddings=None) -> TokenizedEmbeddingReqInput:
    return TokenizedEmbeddingReqInput(
        rid="vlm-request",
        input_text="",
        input_ids=array("q", [1]),
        mm_inputs=MultimodalProcessorOutput(
            mm_items=[
                MultimodalDataItem(
                    modality=Modality.IMAGE,
                    feature=feature,
                    precomputed_embeddings=embeddings,
                )
                for feature in features
            ]
        ),
        token_type_ids=None,
        sampling_params=SamplingParams(max_new_tokens=1),
    )


def _wrap(req, default_transport: bool):
    with (
        patch(
            "sglang.srt.managers.mm_utils._get_is_default_transport",
            return_value=default_transport,
        ),
        patch(
            "sglang.srt.managers.mm_utils.get_serving",
            return_value=SimpleNamespace(skip_tokenizer_init=False),
        ),
    ):
        return wrap_shm_features(req)


class TestToCompactCpu(CustomTestCase):
    def test_view_of_a_batch_is_copied_to_its_own_storage(self):
        batch, views = _split_batch()
        view = views[3]
        self.assertFalse(_owns_exactly_its_bytes(view))

        compact = to_compact_cpu(view)

        self.assertTrue(_owns_exactly_its_bytes(compact))
        torch.testing.assert_close(compact, view)
        self.assertNotEqual(compact.data_ptr(), view.data_ptr())

    def test_tensor_that_owns_its_storage_is_not_copied(self):
        tensor = torch.randn(16, 4)
        self.assertIs(to_compact_cpu(tensor), tensor)

    def test_non_contiguous_view_is_made_contiguous(self):
        tensor = torch.randn(8, 6).t()
        compact = to_compact_cpu(tensor)
        self.assertTrue(compact.is_contiguous())
        torch.testing.assert_close(compact, tensor)


class TestInlineFeatureCompaction(CustomTestCase):
    def test_default_transport_compacts_split_features(self):
        """Multi-node (default) transport pickles features with the request."""
        batch, views = _split_batch()
        req = _wrap(_request(views, embeddings=views[0]), default_transport=True)

        for item, view in zip(req.mm_inputs.mm_items, views):
            self.assertIsInstance(item.feature, torch.Tensor)
            self.assertTrue(_owns_exactly_its_bytes(item.feature))
            torch.testing.assert_close(item.feature, view)
        self.assertTrue(
            _owns_exactly_its_bytes(req.mm_inputs.mm_items[0].precomputed_embeddings)
        )

    def test_default_transport_pickles_each_item_once(self):
        """The regression: N split views used to pickle N copies of the batch."""
        batch, views = _split_batch()
        batch_bytes = batch.numel() * batch.element_size()
        before = len(pickle.dumps(views, protocol=pickle.HIGHEST_PROTOCOL))
        self.assertGreater(before, (_ITEMS - 1) * batch_bytes)

        req = _wrap(_request(views), default_transport=True)
        features = [item.feature for item in req.mm_inputs.mm_items]
        after = len(pickle.dumps(features, protocol=pickle.HIGHEST_PROTOCOL))

        self.assertLess(after, 2 * batch_bytes)
        restored = pickle.loads(pickle.dumps(features))
        for tensor, view in zip(restored, views):
            torch.testing.assert_close(tensor, view)

    def test_default_transport_leaves_list_features_compact(self):
        batch, views = _split_batch()
        req = _wrap(_request([views[:2], tuple(views[2:4])]), default_transport=True)

        as_list, as_tuple = (item.feature for item in req.mm_inputs.mm_items)
        self.assertIsInstance(as_list, list)
        self.assertIsInstance(as_tuple, tuple)
        for tensor in (*as_list, *as_tuple):
            self.assertTrue(_owns_exactly_its_bytes(tensor))

    def test_shm_transport_is_unchanged(self):
        """Single-node SHM already copies only the view's own bytes."""
        batch, views = _split_batch()
        req = _wrap(_request(views[:2]), default_transport=False)
        try:
            for item in req.mm_inputs.mm_items:
                self.assertIsInstance(item.feature, ShmPointerMMData)
                self.assertEqual(tuple(item.feature.shape), tuple(views[0].shape))
        finally:
            for item in req.mm_inputs.mm_items:
                if isinstance(item.feature, ShmPointerMMData):
                    item.feature.close_and_unlink()

    def test_shm_allocation_failure_falls_back_to_a_compact_tensor(self):
        """A full /dev/shm sends the feature inline, so it must be compacted."""
        batch, views = _split_batch()
        with patch(
            "sglang.srt.managers.mm_utils.ShmPointerMMData",
            side_effect=OSError(28, "No space left on device"),
        ):
            fallback = _wrap_shm_or_inline(views[5])

        self.assertIsInstance(fallback, torch.Tensor)
        self.assertTrue(_owns_exactly_its_bytes(fallback))
        torch.testing.assert_close(fallback, views[5])


if __name__ == "__main__":
    unittest.main()
