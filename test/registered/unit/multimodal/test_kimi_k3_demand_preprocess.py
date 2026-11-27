"""K3 demand image preprocessing: header metadata, needed images, readiness.

The metadata path must compose exactly the prompt the full path composes
(input_ids, offsets, grids, identities, pads) so the scheduler can match a
prefix before any image is decoded, and materialization must produce the
full path's items for the images it is asked for.
"""

import asyncio
import base64
import io
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from PIL import Image

from sglang.srt.multimodal.demand_preprocess import (
    ImageDemandState,
    ImageDemandStatus,
    needed_image_indices,
)
from sglang.srt.multimodal.processors.kimi_k3 import KimiK3ImageProcessor
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import ImageData
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=12, suite="base-a-test-cpu")

_IMAGE_TOKEN_ID = 42


class _Tokenizer:
    def encode(self, text, allowed_special=None):
        if text.startswith("<|media_begin|>image "):
            return [10, 11]
        if text == "<|media_end|>":
            return [14]
        return []


class _HFProcessor:
    def __init__(self):
        self.tokenizer = _Tokenizer()
        self.image_processor = SimpleNamespace()
        self.media_processor = SimpleNamespace(
            media_proc_cfg={
                "patch_size": 14,
                "merge_kernel_size": 2,
                "in_patch_limit": 16384,
                "patch_limit_on_one_side": 256,
                "fixed_output_tokens": None,
                "image_mean": [0.5, 0.5, 0.5],
                "image_std": [0.5, 0.5, 0.5],
                "transparent_bg_config": None,
            }
        )


def _png(width: int, height: int, seed: int) -> bytes:
    rng = np.random.default_rng(seed)
    pixels = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
    encoded = io.BytesIO()
    Image.fromarray(pixels).save(encoded, format="PNG")
    return encoded.getvalue()


def _image_data(payload: bytes) -> ImageData:
    return ImageData(url="data:image/png;base64," + base64.b64encode(payload).decode())


def _request():
    return SimpleNamespace(
        audio_data=None, video_data=None, mm_content_hashes=None, mm_hashes=None
    )


@pytest.fixture
def processor():
    hf_config = SimpleNamespace(
        media_placeholder_token_id=_IMAGE_TOKEN_ID,
        to_dict=lambda: {
            "model_type": "kimi_k3",
            "architectures": ["KimiK3ForConditionalGeneration"],
        },
    )
    server_args = ServerArgs(
        model_path="dummy",
        mm_feature_transport="cpu",
        mm_process_config={},
        mm_io_worker_num=1,
        mm_processor_worker_num=0,
        tokenizer_worker_num=1,
        mm_preprocess_cache_size_mb=16,
    )
    publish(server_args, role="tokenizer")
    try:
        processor = KimiK3ImageProcessor(
            hf_config=hf_config,
            server_args=server_args,
            _processor=_HFProcessor(),
            transport_mode=None,
        )
        try:
            # Deferred GPU preprocessing keeps decoded pixels for the model;
            # nothing here touches a GPU.
            with patch(
                "sglang.srt.multimodal.processors.kimi_k3.is_cuda", return_value=True
            ):
                yield processor
        finally:
            processor.shutdown()
    finally:
        reset_context()


def _grid(item):
    return item.model_specific_data["image_grid_thw"].tolist()


def test_metadata_composes_the_full_path_prompt_and_items(processor):
    payloads = [_png(28, 28, 0), _png(84, 56, 1)]
    prompt = [1, _IMAGE_TOKEN_ID, 2, 3, _IMAGE_TOKEN_ID, 4]

    meta, demand = asyncio.run(
        processor.process_mm_metadata_async(
            [_image_data(p) for p in payloads], prompt, _request(), ticket=7
        )
    )
    materialized = asyncio.run(processor.materialize_images_async(demand, [0, 1]))
    full = asyncio.run(
        processor.process_mm_data_async(
            [_image_data(p) for p in payloads], prompt, _request()
        )
    )

    assert demand.ticket == 7
    assert meta.input_ids == full.input_ids
    assert demand.offsets == [item.offsets[0] for item in full.mm_items]
    assert len(meta.mm_items) == len(materialized.mm_items) == len(full.mm_items)
    for meta_item, ready, full_item in zip(
        meta.mm_items, materialized.mm_items, full.mm_items
    ):
        assert meta_item.feature is None
        for item in (meta_item, ready):
            assert item.offsets == full_item.offsets
            assert _grid(item) == _grid(full_item)
            assert item.identity == full_item.identity
            assert item.hash == full_item.hash
            assert item.pad_value == full_item.pad_value
            assert item.pad_values == full_item.pad_values
        assert torch.equal(ready.feature, full_item.feature)
        assert ready.model_specific_data.keys() == full_item.model_specific_data.keys()


def test_materialization_returns_only_the_requested_images(processor):
    payloads = [_png(28, 28, 2), _png(56, 28, 3), _png(28, 56, 4)]
    prompt = [1, _IMAGE_TOKEN_ID, 2, _IMAGE_TOKEN_ID, 3, _IMAGE_TOKEN_ID]
    meta, demand = asyncio.run(
        processor.process_mm_metadata_async(
            [_image_data(p) for p in payloads], prompt, _request(), ticket=1
        )
    )

    ready = asyncio.run(processor.materialize_images_async(demand, [2]))

    assert [item.offsets for item in ready.mm_items] == [meta.mm_items[2].offsets]
    assert ready.mm_items[0].identity == meta.mm_items[2].identity
    assert ready.mm_items[0].feature is not None


def test_corrupt_pixels_fail_at_materialization_with_the_full_path_error(processor):
    # A valid header with truncated pixel data: the header probe accepts it,
    # so the rejection must come from materialization, as a 400 ValueError.
    payload = _png(64, 64, 5)
    corrupt = payload[: len(payload) // 2]
    meta_result = asyncio.run(
        processor.process_mm_metadata_async(
            [_image_data(corrupt)], [1, _IMAGE_TOKEN_ID, 2], _request(), ticket=1
        )
    )
    assert meta_result is not None
    _, demand = meta_result

    with pytest.raises(ValueError, match="^Could not decode image"):
        asyncio.run(processor.materialize_images_async(demand, [0]))
    with pytest.raises(ValueError, match="^Could not decode image"):
        asyncio.run(
            processor.process_mm_data_async(
                [_image_data(corrupt)], [1, _IMAGE_TOKEN_ID, 2], _request()
            )
        )


@pytest.mark.parametrize(
    "payload, prompt",
    [
        # Unreadable header: the full path raises its own decode error.
        (b"not an image", [1, _IMAGE_TOKEN_ID, 2]),
        # Placeholder count mismatch: the full path raises its own error.
        (_png(28, 28, 6), [1, _IMAGE_TOKEN_ID, _IMAGE_TOKEN_ID]),
    ],
)
def test_metadata_declines_what_the_full_path_rejects(processor, payload, prompt):
    result = asyncio.run(
        processor.process_mm_metadata_async(
            [_image_data(payload)], prompt, _request(), ticket=1
        )
    )
    assert result is None


def test_metadata_declines_video_requests(processor):
    request = _request()
    request.video_data = ["clip.mp4"]
    assert (
        asyncio.run(
            processor.process_mm_metadata_async(
                [_image_data(_png(28, 28, 7))],
                [1, _IMAGE_TOKEN_ID, _IMAGE_TOKEN_ID],
                request,
                ticket=1,
            )
        )
        is None
    )


# Three images at inclusive spans; tokens [0, prefix_len) are cached.
_OFFSETS = [(3, 6), (10, 13), (20, 29)]


@pytest.mark.parametrize(
    "prefix_len, needed",
    [
        (0, [0, 1, 2]),  # nothing cached: every image is needed
        (6, [0, 1, 2]),  # the last token of image 0 is still extended
        (7, [1, 2]),  # image 0 lies wholly inside the prefix
        (12, [1, 2]),  # the prefix cuts image 1: it is needed whole
        (30, []),  # every image is cached
    ],
)
def test_needed_images_follow_the_prefix_boundary(prefix_len, needed):
    assert needed_image_indices(_OFFSETS, prefix_len) == needed
    assert ImageDemandState(ticket=1, offsets=_OFFSETS).missing(prefix_len) == needed


def test_shrinking_prefix_requests_only_newly_exposed_images():
    state = ImageDemandState(ticket=1, offsets=_OFFSETS)

    assert state.blocks(15)
    assert state.blocked_prefix_len == 15
    assert state.begin_request(15) == (0, [2])
    # Still waiting for image 2: no second round for it.
    assert state.blocks(15)
    assert state.begin_request(15) is None

    # A lost host prefix exposes image 1 (and keeps image 2 in flight).
    assert state.blocks(8)
    assert state.begin_request(8) == (1, [1])
    assert state.status == [
        ImageDemandStatus.NOT_REQUESTED,
        ImageDemandStatus.REQUESTED,
        ImageDemandStatus.REQUESTED,
    ]

    assert state.take_pending(1, 0, [2])
    state.mark_ready([2])
    assert state.take_pending(1, 1, [1])
    state.mark_ready([1])
    assert not state.blocks(8)
    # Image 0 is still never decoded while the prefix covers it.
    assert state.status[0] == ImageDemandStatus.NOT_REQUESTED


def test_stale_and_duplicate_replies_are_rejected():
    state = ImageDemandState(ticket=5, offsets=_OFFSETS)
    materialization_id, indices = state.begin_request(0)

    assert not state.take_pending(4, materialization_id, indices)  # old ticket
    assert not state.take_pending(5, materialization_id + 1, indices)  # no round
    assert not state.take_pending(5, materialization_id, indices[:1])  # other set
    assert state.take_pending(5, materialization_id, indices)
    assert not state.take_pending(5, materialization_id, indices)  # duplicate


def test_failure_blocks_only_admissions_that_need_the_image():
    state = ImageDemandState(ticket=1, offsets=_OFFSETS)
    state.begin_request(0)
    state.mark_ready([0, 1])
    state.mark_failed([2], "Could not decode image: bad", 400)
    state.mark_failed([2], "second error", 500)

    assert state.failed(0) and state.blocks(0)
    assert state.missing(0) == [2]
    assert not state.failed(30) and not state.blocks(30)
    # The first error is the one the client sees.
    assert (state.error_msg, state.status_code) == ("Could not decode image: bad", 400)
