"""Kimi K3 multimodal processor.

GPU image preprocessing dedicated to K3: unlike the K2.5 wrapper it keeps
the alpha channel through the bicubic resize and then composites RGBA
images onto the checkpoint-configured background
(``transparent_bg_config`` with ``transparent_bg_fill_stage ==
"after_resize"`` in preprocessor_config.json), instead of dropping alpha
at load time.
"""

import asyncio
import functools
import io
import logging
import math
import re
import threading
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from PIL import Image

from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalProcessorOutput,
)
from sglang.srt.environ import envs
from sglang.srt.models.kimi_k3 import KimiK3ForConditionalGeneration
from sglang.srt.multimodal.cache import resolve_multimodal_item_hash, snapshot_media
from sglang.srt.multimodal.kimi_k3_image_processing import (
    DEFERRED_PREPROCESSING_KEY,
    KimiK3DeferredPreprocessing,
)
from sglang.srt.multimodal.kimi_k3_image_processing import (
    fill_transparent_bg as _fill_transparent_bg,
)
from sglang.srt.multimodal.kimi_k3_image_processing import (
    to_chw_uint8,
    to_hwc_uint8,
)
from sglang.srt.multimodal.media_artifacts import (
    MediaArtifactCacheMixin,
    MediaArtifactInput,
)
from sglang.srt.multimodal.media_artifacts.kimi_k3 import (
    KimiK3ImagePreprocessArtifact,
    KimiK3PreprocessConfig,
    KimiK3ResizeConfig,
)
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor as SGLangBaseProcessor,
)
from sglang.srt.multimodal.processors.base_processor import (
    MultimodalSpecialTokens,
)
from sglang.srt.multimodal.processors.kimi_common import KimiGridMMDataMixin
from sglang.srt.multimodal.processors.kimi_k25 import (
    KimiGPUProcessorWrapper,
    _get_image_dimensions,
    _gpu_preprocess_images,
    _grid_thw_from_resize_config,
    navit_resize_config,
)
from sglang.srt.multimodal.transport.cuda_ipc import (
    DEFER_CUDA_IPC_FEATURE_RECONSTRUCTION_KEY,
)
from sglang.srt.runtime_context import get_serving
from sglang.srt.utils import is_cuda, load_image

logger = logging.getLogger(__name__)

IMAGE_PREPROCESS_MODES = ("auto", "cpu", "gpu")


def _resolve_image_preprocess_mode() -> str:
    """Read and validate ``SGLANG_K3_IMAGE_PREPROCESS_MODE``.

    An unrecognized value fails here, at processor construction, rather than
    silently falling back to one of the arms -- a typo in the launch command
    would otherwise look exactly like the mode working.
    """
    mode = (envs.SGLANG_K3_IMAGE_PREPROCESS_MODE.get() or "auto").strip().lower()
    if mode not in IMAGE_PREPROCESS_MODES:
        raise ValueError(
            "SGLANG_K3_IMAGE_PREPROCESS_MODE must be one of "
            f"{', '.join(IMAGE_PREPROCESS_MODES)}; got {mode!r}"
        )
    return mode


@dataclass(frozen=True)
class _K3EncodedImage:
    """Raw image bytes plus header geometry, not yet decoded.

    Under the "auto" GPU budget the decode backend is part of the decision:
    nvJPEG decodes straight into GPU memory, so a request routed to the CPU
    must never reach it. This carries a cache miss from snapshot to the point
    where the whole batch's backend is chosen.
    """

    data: bytes
    width: int
    height: int
    channels: int


def _probe_encoded_image(data: bytes) -> Optional[_K3EncodedImage]:
    """Read width/height/alpha from the image header without decoding pixels.

    Returns None for anything PIL cannot identify, so the caller falls back to
    the regular decode path and keeps its error semantics unchanged.
    """
    try:
        with Image.open(io.BytesIO(data)) as image:
            width, height = image.size
            has_alpha = image.mode != "RGB" and (
                "A" in image.getbands() or "transparency" in image.info
            )
    except Exception:
        return None
    return _K3EncodedImage(data, int(width), int(height), 4 if has_alpha else 3)


def _image_geometry(image) -> tuple[int, int, int]:
    """(width, height, channels) for an encoded, PIL, or CHW tensor image."""
    if isinstance(image, _K3EncodedImage):
        return image.width, image.height, image.channels
    width, height = _get_image_dimensions(image)
    if isinstance(image, torch.Tensor):
        channels = 3 if image.dim() == 2 or image.shape[0] == 1 else image.shape[0]
        return int(width), int(height), int(channels)
    has_alpha = image.mode != "RGB" and (
        "A" in image.getbands() or "transparency" in image.info
    )
    return int(width), int(height), 4 if has_alpha else 3


def _estimate_gpu_preprocess_bytes(images, config) -> int:
    """Rough peak GPU bytes for preprocessing ``images`` on the GPU path.

    Per image, source side: the uint8 image moved to the GPU, its copy in the
    same-size ``torch.cat`` batch, and the fp32 copy ``_resize_bicubic_if_needed``
    makes of that whole batch before interpolating -- uncapped on this branch,
    which has no chunked resize -- so 6 * C * W * H. Output side: the fp32
    resized image, its patchified copy and the final concat, 3 * 3 * padded * 4.
    """
    total = 0
    for image in images:
        width, height, channels = _image_geometry(image)
        resize = navit_resize_config(
            width,
            height,
            config.patch_size,
            config.merge_kernel_size,
            config.in_patch_limit,
            config.patch_limit_on_one_side,
            config.fixed_output_tokens,
        )
        padded = (resize["new_width"] + resize["pad_width"]) * (
            resize["new_height"] + resize["pad_height"]
        )
        total += 6 * channels * width * height + 3 * 3 * padded * 4
    return total


def _decode_encoded_image(image, gpu_image_decode):
    """Decode an ``_K3EncodedImage`` with an explicit backend; pass others through."""
    if not isinstance(image, _K3EncodedImage):
        return image
    decoded, _ = load_image(image.data, gpu_image_decode)
    if isinstance(decoded, Image.Image):
        decoded.load()
    return decoded


def _encode_k3_special_tokens(tokenizer, text: str) -> list[int]:
    """Encode K3 control tokens without allowing them to be BPE-split."""
    try:
        return list(tokenizer.encode(text, allowed_special="all"))
    except TypeError:
        # Keep the helper usable with lightweight tokenizer stubs in CPU tests.
        return list(tokenizer.encode(text))


def _expand_k3_image_prompt_token_ids(
    input_ids: Union[List[int], torch.Tensor],
    image_token_id: int,
    image_token_counts: List[int],
    image_sizes: List[tuple[int, int]],
    tokenizer,
) -> torch.Tensor:
    """Expand K3 image placeholders into the checkpoint's media contract.

    K3 requires each image feature span to be enclosed by its original uploaded
    dimensions.  The chat template deliberately emits one ``media_pad`` per
    image; after decode, insert the surrounding control tokens and expand that
    one placeholder to the NaViT feature count.
    """
    if len(image_token_counts) != len(image_sizes):
        raise ValueError("Expected one original size for each K3 image.")

    if isinstance(input_ids, torch.Tensor):
        input_ids = input_ids.detach().flatten().cpu().numpy()
    input_ids = np.asarray(input_ids, dtype=np.int64)

    if input_ids.ndim != 1:
        raise ValueError("Expected a flat K3 prompt token sequence.")

    placeholder_positions = np.flatnonzero(input_ids == image_token_id)
    placeholder_count = len(placeholder_positions)
    if placeholder_count != len(image_token_counts):
        raise ValueError(
            f"Expected {len(image_token_counts)} image placeholder token(s), "
            f"found {placeholder_count}."
        )

    # Splice whole segments instead of a per-token Python loop: this runs on the
    # tokenizer event loop and long prompts (~1M tokens) blocked it for ~0.1 s.
    segments = []
    segment_start = 0
    for image_index, position in enumerate(placeholder_positions):
        segments.append(input_ids[segment_start:position])
        width, height = image_sizes[image_index]
        segments.append(
            _as_int64_array(
                _encode_k3_special_tokens(
                    tokenizer,
                    f"<|media_begin|>image {width}x{height}<|media_content|>",
                )
            )
        )
        segments.append(
            np.full(
                max(image_token_counts[image_index], 0),
                image_token_id,
                dtype=np.int64,
            )
        )
        segments.append(
            _as_int64_array(_encode_k3_special_tokens(tokenizer, "<|media_end|>"))
        )
        segment_start = position + 1
    segments.append(input_ids[segment_start:])

    return torch.from_numpy(np.concatenate(segments)).unsqueeze(0)


def _as_int64_array(token_ids: list[int]) -> np.ndarray:
    return np.asarray(token_ids, dtype=np.int64).reshape(-1)


def _expand_k3_image_prompt_text(
    input_text: str,
    image_token: str,
    image_token_counts: List[int],
    image_sizes: List[tuple[int, int]],
) -> str:
    """Render the K3 media framing for the CPU HF-processor fallback."""
    parts = input_text.split(image_token)
    if len(parts) - 1 != len(image_token_counts):
        raise ValueError(
            f"Expected {len(image_token_counts)} image placeholder(s), "
            f"found {len(parts) - 1}."
        )

    output = [parts[0]]
    for image_token_count, (width, height), suffix in zip(
        image_token_counts, image_sizes, parts[1:]
    ):
        output.extend(
            (
                f"<|media_begin|>image {width}x{height}<|media_content|>",
                image_token * image_token_count,
                "<|media_end|>",
                suffix,
            )
        )
    return "".join(output)


def _k3_to_cuda_chw(image: Union[torch.Tensor, Image.Image]) -> torch.Tensor:
    if isinstance(image, Image.Image):
        return to_chw_uint8(image, device="cuda")

    image = image.cuda()
    if image.dim() == 2:
        image = image.unsqueeze(0)
    if image.shape[0] == 1:
        image = image.repeat(3, 1, 1)
    return image


class KimiK3GPUProcessorWrapper(KimiGPUProcessorWrapper):
    def __init__(self, hf_processor, image_token, image_token_id, config):
        self.preprocess_config = config
        super().__init__(
            hf_processor,
            image_token=image_token,
            image_token_id=image_token_id,
            patch_size=config.patch_size,
            merge_kernel_size=config.merge_kernel_size,
            in_patch_limit=config.in_patch_limit,
            patch_limit_on_one_side=config.patch_limit_on_one_side,
            fixed_output_tokens=config.fixed_output_tokens,
            image_mean=config.image_mean,
            image_std=config.image_std,
        )
        self._transparent_bg_config = config.transparent_bg_config
        self._image_preprocess_mode = _resolve_image_preprocess_mode()
        self._log_preprocess_backend = envs.SGLANG_K3_IMAGE_PREPROCESS_LOG.get()
        self._oom_fallback = envs.SGLANG_K3_IMAGE_PREPROCESS_OOM_FALLBACK.get()

    def _use_gpu(self, images, use_gpu: Optional[bool] = None) -> bool:
        """Whether this call preprocesses on the GPU.

        ``use_gpu`` is the per-request decision "auto" mode made from its GPU
        budget; None means "follow SGLANG_K3_IMAGE_PREPROCESS_MODE".
        """
        if not images or not torch.cuda.is_available():
            return False
        if use_gpu is None:
            return self._image_preprocess_mode != "cpu"
        return use_gpu

    def _after_gpu_oom(self, images) -> list:
        """Free the failed attempt's GPU memory; return CPU-acceptable images.

        Runs after the ``except`` block has exited, so the OOM traceback --
        whose frames still reference the partial batch tensors -- is gone and
        ``empty_cache`` can actually return that memory to the driver, where
        the scheduler on the same GPU needs it. Decoded CUDA tensors (nvJPEG
        output) are copied to the host, since the CPU processor takes PIL.
        """
        torch.cuda.empty_cache()
        logger.warning(
            "Kimi-K3 preprocess: CUDA OOM on the GPU path for %d image(s); "
            "redoing them on the CPU. Lower "
            "SGLANG_K3_IMAGE_PREPROCESS_GPU_BUDGET_MB if this repeats.",
            len(images),
        )
        return [
            (
                Image.fromarray(to_hwc_uint8(image).numpy())
                if isinstance(image, torch.Tensor)
                else image
            )
            for image in images
        ]

    def _log_backend(self, backend: str, images, resize_configs=None) -> None:
        if not self._log_preprocess_backend:
            return
        total_tokens = (
            sum(config["num_tokens"] for config in resize_configs)
            if resize_configs
            else None
        )
        logger.info(
            "Kimi-K3 preprocess: backend=%s mode=%s items=%d visual_tokens=%s",
            backend,
            self._image_preprocess_mode,
            len(images or ()),
            "n/a" if total_tokens is None else total_tokens,
        )

    def preprocess_fingerprint_payload(self):
        return self.preprocess_config

    def _prepare_input_ids(
        self, input_text, resize_configs, original_input_ids, image_sizes
    ):
        image_token_counts = [config["num_tokens"] for config in resize_configs]
        if original_input_ids is None:
            original_input_ids = _encode_k3_special_tokens(
                self._hf_processor.tokenizer, input_text
            )
        return _expand_k3_image_prompt_token_ids(
            original_input_ids,
            self._image_token_id,
            image_token_counts,
            image_sizes,
            self._hf_processor.tokenizer,
        )

    def __call__(self, text=None, images=None, **kwargs):
        images = images or kwargs.pop("images", None)
        original_input_ids = kwargs.pop("sglang_original_input_ids", None)
        use_gpu = kwargs.pop("sglang_use_gpu", None)
        if self._use_gpu(images, use_gpu):
            try:
                return self._gpu_call(text, images, original_input_ids)
            except torch.OutOfMemoryError:
                if not self._oom_fallback:
                    raise
            # Outside the except block: see _after_gpu_oom.
            images = self._after_gpu_oom(images)
        return self._cpu_call(text, images, original_input_ids, **kwargs)

    def _gpu_call(self, text, images, original_input_ids=None):
        input_text = text[0] if isinstance(text, list) else text

        resize_configs = []
        image_sizes = []
        for image in images:
            w, h = _get_image_dimensions(image)
            image_sizes.append((w, h))
            resize_configs.append(
                navit_resize_config(
                    w,
                    h,
                    self._patch_size,
                    self._merge_kernel_size,
                    self._in_patch_limit,
                    self._patch_limit_on_one_side,
                    self._fixed_output_tokens,
                )
            )

        input_ids = self._prepare_input_ids(
            input_text, resize_configs, original_input_ids, image_sizes
        )

        self._log_backend("gpu", images, resize_configs)
        image_scale, image_bias = self._get_gpu_norm_tensors()
        # Shared source-compatible batched pipeline (same as K2.5): RGBA
        # inputs land in their own source-shape groups, and the
        # transparent-background compositing runs on each resized batch
        # before patchify -- identical order to the previous per-image path.
        pixel_values, grid_thws = _gpu_preprocess_images(
            images,
            resize_configs,
            image_scale,
            image_bias,
            self._patch_size,
            to_chw=_k3_to_cuda_chw,
            post_resize=lambda x: _fill_transparent_bg(x, self._transparent_bg_config),
        )

        return {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
            "image_grid_thw": grid_thws,
        }

    def _cpu_call(self, text, images, original_input_ids=None, **kwargs):
        """HF fallback with the same K3 media framing as the GPU path."""
        input_text = text[0] if isinstance(text, list) else text
        if not images:
            return self._hf_processor(text=[input_text], **kwargs)

        image_sizes = [_get_image_dimensions(image) for image in images]
        image_token_counts = [
            self._hf_processor.media_processor.media_tokens_calculator(
                {"type": "image", "image": image}
            )
            for image in images
        ]
        self._log_backend(
            "cpu", images, [{"num_tokens": count} for count in image_token_counts]
        )
        expanded_text = _expand_k3_image_prompt_text(
            input_text,
            self._image_token,
            image_token_counts,
            image_sizes,
        )
        kwargs["medias"] = [{"type": "image", "image": image} for image in images]
        out = self._hf_processor(text=[expanded_text], **kwargs)
        out["input_ids"] = self._prepare_input_ids(
            input_text,
            [{"num_tokens": count} for count in image_token_counts],
            original_input_ids,
            image_sizes,
        )
        grid_thws = out.pop("grid_thws", None)
        if grid_thws is not None:
            out["image_grid_thw"] = grid_thws
        return out

    def prepare_deferred(self, text, images, original_input_ids=None):
        input_text = text[0] if isinstance(text, list) else text
        image_sizes = [_get_image_dimensions(image) for image in images]
        resize_configs = [
            navit_resize_config(
                width,
                height,
                self._patch_size,
                self._merge_kernel_size,
                self._in_patch_limit,
                self._patch_limit_on_one_side,
                self._fixed_output_tokens,
            )
            for width, height in image_sizes
        ]
        input_ids = self._prepare_input_ids(
            input_text, resize_configs, original_input_ids, image_sizes
        )
        # This path only ever defers GPU preprocessing: the caller gates on
        # `_should_defer_gpu_preprocessing` and stages CHW uint8 features.
        deferred_preprocessing = functools.partial(
            KimiK3DeferredPreprocessing,
            backend="gpu",
            image_mean=list(self._image_mean),
            image_std=list(self._image_std),
            transparent_bg_config=self._transparent_bg_config,
        )
        return input_ids, resize_configs, deferred_preprocessing

    def prepare_image_features(self, images, use_gpu: Optional[bool] = None):
        """Prepare prompt-independent, per-image features in one processor call."""
        image_sizes = [_get_image_dimensions(image) for image in images]
        resize_configs = [
            navit_resize_config(
                width,
                height,
                self._patch_size,
                self._merge_kernel_size,
                self._in_patch_limit,
                self._patch_limit_on_one_side,
                self._fixed_output_tokens,
            )
            for width, height in image_sizes
        ]

        pixel_values = None
        if self._use_gpu(images, use_gpu):
            self._log_backend("gpu", images, resize_configs)
            image_scale, image_bias = self._get_gpu_norm_tensors()
            try:
                pixel_values, grid_thws = _gpu_preprocess_images(
                    images,
                    resize_configs,
                    image_scale,
                    image_bias,
                    self._patch_size,
                    to_chw=_k3_to_cuda_chw,
                    post_resize=lambda x: _fill_transparent_bg(
                        x, self._transparent_bg_config
                    ),
                )
            except torch.OutOfMemoryError:
                if not self._oom_fallback:
                    raise
            if pixel_values is None:
                # Outside the except block: see _after_gpu_oom.
                images = self._after_gpu_oom(images)
        if pixel_values is None:
            # `_cpu_call` logs the backend itself.
            # The checkpoint CPU processor couples prompt composition with media
            # preprocessing. A synthetic prompt keeps that API but is discarded;
            # image features and grids are independent of its text.
            output = self._cpu_call(self._image_token * len(images), images)
            pixel_values = output["pixel_values"]
            grid_thws = output["image_grid_thw"]

        grids = [tuple(int(value) for value in grid) for grid in grid_thws.tolist()]
        patch_counts = [math.prod(grid) for grid in grids]
        if sum(patch_counts) != pixel_values.shape[0]:
            raise ValueError(
                "Kimi-K3 processor feature length does not match image grids: "
                f"{pixel_values.shape[0]} != {sum(patch_counts)}"
            )
        return (
            list(pixel_values.split(patch_counts)),
            image_sizes,
            resize_configs,
            grids,
        )


class KimiK3ImageProcessor(
    KimiGridMMDataMixin,
    MediaArtifactCacheMixin,
    SGLangBaseProcessor,
):
    models = [KimiK3ForConditionalGeneration]
    artifact_modality = Modality.IMAGE
    # K3 accuracy is sensitive to the chroma upsampling used for common 4:2:0
    # JPEG inputs. This mode uses interpolated nvJPEG upsampling when the K3
    # image dependency is installed and otherwise falls back to PIL.
    gpu_image_decode = "nvjpeg_fancy"
    prefer_tokenized_input = True
    precompute_hash_before_cpu_transfer = True
    auto_mm_processor_worker_num = 2
    auto_mm_io_worker_num = 16
    auto_mm_preprocess_cache_size_mb = 256
    supports_mm_processor_concurrency = True
    preserve_processor_input_ids = True

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        mm_tokens = MultimodalSpecialTokens(
            image_token="<|media_pad|>",
            image_token_id=hf_config.media_placeholder_token_id,
            image_token_regex=re.compile(r"(?:<\|media_pad\|>)+"),
        ).build(_processor)

        preprocess_config = KimiK3PreprocessConfig.from_media_processor(
            _processor.media_processor
        )

        processor = KimiK3GPUProcessorWrapper(
            _processor,
            image_token=mm_tokens.image_token,
            image_token_id=mm_tokens.image_token_id,
            config=preprocess_config,
        )
        self._image_preprocess_mode = _resolve_image_preprocess_mode()
        # `_load_single_item` is a classmethod and reads `cls.gpu_image_decode`,
        # so an instance attribute would not reach the decode path. Assign on
        # the class instead of hardcoding the value at class-definition time,
        # which would freeze whatever the environment held at import. The mode
        # is process-global, so every instance resolves the same value.
        type(self).gpu_image_decode = (
            False if self._image_preprocess_mode == "cpu" else "nvjpeg_fancy"
        )

        super().__init__(hf_config, server_args, processor, *args, **kwargs)
        self.mm_tokens = mm_tokens

        # "auto" GPU budget, shared by every request this tokenizer worker is
        # preprocessing. Workers in other tokenizer processes share the same
        # GPU, so each gets an equal slice. None = no budget (always GPU).
        budget_mb = envs.SGLANG_K3_IMAGE_PREPROCESS_GPU_BUDGET_MB.get()
        worker_num = max(int(get_serving().tokenizer_worker_num), 1)
        self._gpu_preprocess_budget_bytes = (
            budget_mb * 1024 * 1024 // worker_num
            if self._image_preprocess_mode == "auto" and budget_mb > 0
            else None
        )
        self._gpu_preprocess_inflight_bytes = 0
        # Reserved from processor worker threads and the event loop alike.
        self._gpu_preprocess_lock = threading.Lock()
        self._log_preprocess_backend = envs.SGLANG_K3_IMAGE_PREPROCESS_LOG.get()
        if self._gpu_preprocess_budget_bytes is not None:
            # GPU and CPU features for one image differ slightly, and a request
            # may now land on either; see artifact_feature_hash_is_backend_stable.
            self.artifact_feature_hash_is_backend_stable = False

        logger.info(
            "Kimi-K3 image preprocessing mode=%s (jpeg_decode=%s, "
            "gpu_budget=%s, oom_fallback=%s, feature_transport=%s, "
            "processor_workers=%d, io_workers=%d, preprocess_cache=%s).",
            self._image_preprocess_mode,
            "pil-cpu" if self.gpu_image_decode is False else self.gpu_image_decode,
            (
                "off"
                if self._gpu_preprocess_budget_bytes is None
                else f"{self._gpu_preprocess_budget_bytes / 2**20:.0f}MiB"
            ),
            "on" if envs.SGLANG_K3_IMAGE_PREPROCESS_OOM_FALLBACK.get() else "off",
            self.mm_feature_transport,
            self.mm_processor_worker_num,
            self.mm_io_worker_num,
            "on" if self.mm_preprocess_cache.enabled else "off",
        )

    def _reserve_gpu_preprocess(self, images) -> tuple[Optional[bool], int]:
        """Pick this batch's backend under the "auto" GPU budget.

        Returns ``(use_gpu, reserved_bytes)``. ``use_gpu`` is None when no
        budget is active, meaning "follow the static mode" exactly as before.
        A batch goes to the GPU only if its estimate fits next to everything
        already in flight; it never waits. A batch carrying an already-decoded
        tensor keeps the GPU, since the CPU processor never received tensors
        before this switch existed. The caller must release ``reserved_bytes``.
        """
        if self._gpu_preprocess_budget_bytes is None or not images:
            return None, 0
        if any(isinstance(image, torch.Tensor) for image in images):
            return True, 0
        budget = self._gpu_preprocess_budget_bytes
        estimate = _estimate_gpu_preprocess_bytes(
            images, self._processor.preprocess_config
        )
        with self._gpu_preprocess_lock:
            inflight = self._gpu_preprocess_inflight_bytes
            use_gpu = inflight + estimate <= budget
            if use_gpu:
                self._gpu_preprocess_inflight_bytes += estimate
        if self._log_preprocess_backend:
            logger.info(
                "Kimi-K3 preprocess budget: backend=%s items=%d "
                "estimate=%.0fMiB inflight=%.0fMiB budget=%.0fMiB",
                "gpu" if use_gpu else "cpu",
                len(images),
                estimate / 2**20,
                inflight / 2**20,
                budget / 2**20,
            )
        return use_gpu, estimate if use_gpu else 0

    def _release_gpu_preprocess(self, reserved_bytes: int) -> None:
        if reserved_bytes:
            with self._gpu_preprocess_lock:
                self._gpu_preprocess_inflight_bytes -= reserved_bytes

    async def _decode_images_for_backend(self, images, use_gpu: Optional[bool]):
        """Decode any ``_K3EncodedImage`` on the IO pool with the chosen backend."""
        if not any(isinstance(image, _K3EncodedImage) for image in images):
            return list(images)
        gpu_image_decode = self.gpu_image_decode if use_gpu is not False else False
        loop = asyncio.get_running_loop()
        return list(
            await asyncio.gather(
                *(
                    loop.run_in_executor(
                        self.io_executor,
                        _decode_encoded_image,
                        image,
                        gpu_image_decode,
                    )
                    for image in images
                )
            )
        )

    def decode_media_snapshot(self, snapshot, modality):
        """Leave raw image bytes undecoded while the "auto" budget is active.

        The batch's backend is chosen in ``_run_preprocess_and_build_artifact_batch``
        from header geometry, and only then decoded -- nvJPEG would otherwise
        put the pixels on the GPU before the request could be sent to the CPU.
        Headers PIL cannot read take the regular decode path unchanged.
        """
        if (
            self._gpu_preprocess_budget_bytes is not None
            and modality == Modality.IMAGE
            and isinstance(snapshot.data, (bytes, bytearray))
        ):
            encoded = _probe_encoded_image(bytes(snapshot.data))
            if encoded is not None:
                return encoded
        return super().decode_media_snapshot(snapshot, modality)

    async def _run_preprocess_and_build_artifact_batch(self, entries):
        use_gpu, reserved = self._reserve_gpu_preprocess(
            [entry.media for entry in entries]
        )
        try:
            decoded = await self._decode_images_for_backend(
                [entry.media for entry in entries], use_gpu
            )
            entries = [
                replace(entry, media=media) for entry, media in zip(entries, decoded)
            ]
            if self.mm_processor_executor is None:
                return self.prepare_artifact_batch(entries, use_gpu=use_gpu)
            return await self.mm_processor_executor.run(
                self.prepare_artifact_batch, entries, use_gpu=use_gpu
            )
        finally:
            self._release_gpu_preprocess(reserved)

    def _should_defer_gpu_preprocessing(self, images) -> bool:
        """
        when raw_bytes <= processed_bytes, preprocess first would introduce larger payload, so deferring gpu preprocessing would benefit
        """
        # Both explicit modes opt out of deferral, for opposite reasons.
        # Under "cpu" this gate would otherwise win first -- its own
        # precondition is `mm_feature_transport == "cpu"`, which is exactly
        # the transport a CPU run uses -- and hand the work to the vision-DP
        # owner rank's GPU, so the request would still preprocess on a GPU,
        # just in the model process. Under "gpu" deferral is simply not the
        # eager-GPU baseline being measured.
        if self._image_preprocess_mode != "auto":
            return False
        if (
            not images
            or self.mm_feature_transport != "cpu"
            or not is_cuda()
            or not all(
                isinstance(image, Image.Image)
                or (isinstance(image, torch.Tensor) and image.dtype == torch.uint8)
                for image in images
            )
        ):
            return False

        raw_bytes = 0
        processed_bytes = 0
        config = self._processor.preprocess_config
        patch_size = config.patch_size
        for image in images:
            width, height = _get_image_dimensions(image)
            resize_config = navit_resize_config(
                width,
                height,
                patch_size,
                config.merge_kernel_size,
                config.in_patch_limit,
                config.patch_limit_on_one_side,
                config.fixed_output_tokens,
            )
            if isinstance(image, torch.Tensor):
                channels = (
                    3 if image.dim() == 2 or image.shape[0] == 1 else image.shape[0]
                )
            else:
                channels = (
                    4
                    if image.mode != "RGB"
                    and ("A" in image.getbands() or "transparency" in image.info)
                    else 3
                )
            raw_bytes += channels * width * height
            padded_width = resize_config["new_width"] + resize_config["pad_width"]
            padded_height = resize_config["new_height"] + resize_config["pad_height"]
            processed_bytes += 3 * padded_width * padded_height * torch.float32.itemsize

        return raw_bytes <= processed_bytes

    def _build_deferred_output(self, base_output):
        (
            input_ids,
            resize_configs,
            deferred_preprocessing,
        ) = self._processor.prepare_deferred(
            base_output.input_text,
            base_output.images,
            base_output.input_ids,
        )
        offsets = self.get_mm_items_offset(
            input_ids.flatten(), self.mm_tokens.image_token_id
        )
        if len(offsets) != len(base_output.images):
            raise ValueError("Expected one Kimi-K3 image span for each image")

        items = []
        for image, resize_config, offset in zip(
            base_output.images, resize_configs, offsets
        ):
            grid_thw = _grid_thw_from_resize_config(
                resize_config, self._processor.preprocess_config.patch_size
            )
            item = MultimodalDataItem(
                modality=Modality.IMAGE,
                feature=to_chw_uint8(image),
                offsets=[offset],
                model_specific_data={
                    "image_grid_thw": torch.tensor([grid_thw], dtype=torch.int64),
                    DEFERRED_PREPROCESSING_KEY: deferred_preprocessing(
                        resize_config=resize_config
                    ),
                },
            )
            items.append(item)

        self._precompute_hashes_before_cpu_transfer(items)
        return MultimodalProcessorOutput(
            input_ids=input_ids.flatten().tolist(),
            mm_items=items,
            im_token_id=self.mm_tokens.image_token_id,
        )

    def _make_artifact(
        self,
        *,
        content_digest: str,
        artifact_key: str,
        original_size: tuple[int, int],
        resize_config: dict,
        grid_thw: tuple[int, int, int],
        feature: torch.Tensor,
        deferred: Optional[KimiK3DeferredPreprocessing] = None,
    ) -> KimiK3ImagePreprocessArtifact:
        """Freeze one image's prompt-independent preprocessing result."""
        # Use the same feature-hash contract as MultimodalDataItem.
        feature_hash = resolve_multimodal_item_hash(
            feature=feature, namespace=artifact_key
        )
        if not self.keep_mm_features_on_device and feature.device.type != "cpu":
            feature = feature.cpu()
        return KimiK3ImagePreprocessArtifact(
            content_digest=content_digest,
            artifact_key=artifact_key,
            feature_hash=feature_hash,
            original_size=original_size,
            resize_config=KimiK3ResizeConfig.from_dict(resize_config),
            grid_thw=grid_thw,
            feature=feature,
            deferred=deferred,
        )

    def prepare_artifact_batch(
        self,
        entries: list[MediaArtifactInput],
        *,
        processor=None,
        use_gpu: Optional[bool] = None,
    ) -> list[KimiK3ImagePreprocessArtifact]:
        """Preprocess raw cache misses into reusable per-image cache items.

        Each entry is a confirmed cache miss. It is either processed now or
        stored with the metadata needed for deferred GPU preprocessing.
        ``use_gpu`` is the "auto" budget's decision for this batch (None:
        follow the static mode). When it chose the CPU nothing is deferred,
        since deferral would only move the GPU work to the model process.
        """
        processor = processor or self._processor
        artifacts: list[Optional[KimiK3ImagePreprocessArtifact]] = [None] * len(entries)
        # 1. collect inputs that must be preprocessed now instead of deferred
        eager_entry_indices = []
        eager_images = []

        config = processor.preprocess_config
        for index, entry in enumerate(entries):
            image = entry.media
            if use_gpu is False or not self._should_defer_gpu_preprocessing([image]):
                eager_entry_indices.append(index)
                eager_images.append(image)
                continue

            width, height = _get_image_dimensions(image)
            resize_config = navit_resize_config(
                width,
                height,
                config.patch_size,
                config.merge_kernel_size,
                config.in_patch_limit,
                config.patch_limit_on_one_side,
                config.fixed_output_tokens,
            )
            grid_thw = _grid_thw_from_resize_config(resize_config, config.patch_size)
            feature = to_chw_uint8(image).cpu().contiguous()
            artifacts[index] = self._make_artifact(
                content_digest=entry.content_digest,
                artifact_key=entry.artifact_key,
                original_size=(width, height),
                resize_config=resize_config,
                grid_thw=grid_thw,
                feature=feature,
                deferred=KimiK3DeferredPreprocessing(
                    backend="gpu",
                    image_mean=list(config.image_mean),
                    image_std=list(config.image_std),
                    transparent_bg_config=config.transparent_bg_config,
                    resize_config=resize_config,
                ),
            )

        # 2. preprocess CPU eager inputs as one batch
        if eager_images:
            features, sizes, configs, grids = processor.prepare_image_features(
                eager_images, use_gpu=use_gpu
            )
            for index, feature, size, resize_config, grid in zip(
                eager_entry_indices, features, sizes, configs, grids
            ):
                entry = entries[index]
                artifacts[index] = self._make_artifact(
                    content_digest=entry.content_digest,
                    artifact_key=entry.artifact_key,
                    original_size=size,
                    resize_config=resize_config,
                    grid_thw=grid,
                    feature=feature,
                )

        # 3. return artifacts in the original processor-input order
        if any(artifact is None for artifact in artifacts):
            raise RuntimeError("Kimi-K3 artifact batch did not produce every image")
        return [artifact for artifact in artifacts if artifact is not None]

    def compose_request(
        self,
        input_text,
        artifacts: list[KimiK3ImagePreprocessArtifact],
    ) -> MultimodalProcessorOutput:
        """Compose the current request from its prompt and ordered artifacts.

        ``prepare_media_artifacts`` has already returned one artifact for each
        processor input, either from the preprocess cache or from fresh
        preprocessing. This method expands the current prompt's image tokens
        and converts each artifact into its request-specific
        ``MultimodalDataItem`` with offsets, grid metadata, feature, and feature
        hash. It does not read raw media or access the preprocess cache.
        """
        # 1. rebuild prompt-specific tokens and offsets
        original_ids = (
            input_text
            if isinstance(input_text, (list, torch.Tensor))
            else _encode_k3_special_tokens(self._tokenizer, input_text)
        )
        input_ids = _expand_k3_image_prompt_token_ids(
            original_ids,
            self.mm_tokens.image_token_id,
            [artifact.resize_config.num_tokens for artifact in artifacts],
            [artifact.original_size for artifact in artifacts],
            self._tokenizer,
        ).flatten()
        offsets = self.get_mm_items_offset(input_ids, self.mm_tokens.image_token_id)
        if len(offsets) != len(artifacts):
            raise ValueError("Expected one Kimi-K3 image span for each image")

        # 2. build request-owned items from prompt-independent artifacts
        items = []
        for artifact, offset in zip(artifacts, offsets):
            model_specific_data = {
                "image_grid_thw": torch.tensor([artifact.grid_thw], dtype=torch.int64)
            }
            if artifact.deferred is not None:
                model_specific_data[DEFERRED_PREPROCESSING_KEY] = artifact.deferred
            item = MultimodalDataItem(
                modality=Modality.IMAGE,
                feature=artifact.feature,
                offsets=[offset],
                model_specific_data=model_specific_data,
            )
            item.set_hash(artifact.feature_hash)
            if self.keep_mm_features_on_device and item.feature is not None:
                item.model_specific_data[DEFER_CUDA_IPC_FEATURE_RECONSTRUCTION_KEY] = (
                    True
                )
            items.append(item)

        return MultimodalProcessorOutput(
            input_ids=input_ids.tolist(),
            mm_items=self._prepare_mm_items_for_transport(items),
            im_token_id=self.mm_tokens.image_token_id,
        )

    async def _stage_raw_images(self, image_data) -> Optional[list]:
        """Fetch each raw image once and read its header, without decoding.

        Returns one ``_K3EncodedImage`` (or already-decoded PIL/tensor) per
        input, or None when any input is not plain raw media; the caller then
        keeps the previous path untouched. URLs are downloaded here exactly
        once, and the bytes are reused for the real decode.
        """
        if not image_data or any(self._is_preprocessed_input(i) for i in image_data):
            return None
        loop = asyncio.get_running_loop()
        try:
            snapshots = await asyncio.gather(
                *(
                    loop.run_in_executor(self.io_executor, snapshot_media, item)
                    for item in image_data
                )
            )
        except Exception:
            # Let the regular loader raise its usual, client-facing error.
            return None
        staged = []
        for snapshot in snapshots:
            data = snapshot.data
            if isinstance(data, (bytes, bytearray)):
                encoded = _probe_encoded_image(bytes(data))
                if encoded is None:
                    return None
                staged.append(encoded)
            elif isinstance(data, (Image.Image, torch.Tensor)):
                staged.append(data)
            else:
                return None
        return staged

    async def _process_mm_data_uncached(
        self, image_data, input_text, request_obj, **kwargs
    ):
        """Uncached path; under the "auto" budget, pick the backend first."""
        staged = (
            await self._stage_raw_images(list(image_data or []))
            if self._gpu_preprocess_budget_bytes is not None
            else None
        )
        if staged is None:
            return await self._process_mm_data_uncached_impl(
                image_data, input_text, request_obj, **kwargs
            )

        use_gpu, reserved = self._reserve_gpu_preprocess(staged)
        try:
            if use_gpu:
                # The loader decodes raw bytes with nvJPEG as before.
                image_data = [
                    i.data if isinstance(i, _K3EncodedImage) else i for i in staged
                ]
            else:
                image_data = await self._decode_images_for_backend(staged, False)
            return await self._process_mm_data_uncached_impl(
                image_data, input_text, request_obj, use_gpu=use_gpu, **kwargs
            )
        finally:
            self._release_gpu_preprocess(reserved)

    async def _process_mm_data_uncached_impl(
        self, image_data, input_text, request_obj, use_gpu=None, **kwargs
    ):
        """Compatibility path for precomputed inputs and lightweight test stubs."""
        expected_image_count = len(image_data or [])
        placeholder_count = self.count_image_placeholders(
            input_text, self.mm_tokens.image_token_id
        )
        if placeholder_count is not None:
            base_output = await self.fast_load_mm_data(
                prompt=input_text,
                image_data=image_data,
                multimodal_tokens=self.mm_tokens,
                discard_alpha_channel=False,
                input_ids=input_text,
            )
        else:
            base_output = await self.load_mm_data(
                prompt=input_text,
                image_data=image_data,
                multimodal_tokens=self.mm_tokens,
                discard_alpha_channel=False,
            )
        if len(base_output.images) != expected_image_count:
            raise ValueError(
                "Kimi image placeholders must map one-to-one to image data: "
                f"expected {expected_image_count}, loaded {len(base_output.images)}"
            )
        if use_gpu is not False and self._should_defer_gpu_preprocessing(
            base_output.images
        ):
            return self._build_deferred_output(base_output)
        mm_items, input_ids, _ = await self.process_and_combine_mm_data_async(
            base_output,
            self.mm_tokens,
            sglang_original_input_ids=base_output.input_ids,
            sglang_use_gpu=use_gpu,
        )
        if self.keep_mm_features_on_device:
            for item in mm_items:
                item.model_specific_data[DEFER_CUDA_IPC_FEATURE_RECONSTRUCTION_KEY] = (
                    True
                )
        return MultimodalProcessorOutput(
            input_ids=input_ids.tolist(),
            mm_items=mm_items,
            im_token_id=self.mm_tokens.image_token_id,
        )

    async def process_mm_data_async(
        self,
        image_data: List[Union[str, bytes, Dict]],
        input_text,
        request_obj,
        *args,
        **kwargs,
    ):
        if request_obj.video_data or kwargs.get("audio_data"):
            raise ValueError("Kimi-K3 supports image input only")

        expected_image_count = len(image_data or [])
        placeholder_count = self.count_image_placeholders(
            input_text, self.mm_tokens.image_token_id
        )
        if placeholder_count is not None:
            if placeholder_count != expected_image_count:
                raise ValueError(
                    "Kimi image placeholders must map one-to-one to image data: "
                    f"expected {expected_image_count}, found {placeholder_count} token(s)"
                )
        if (
            any(self._is_preprocessed_input(item) for item in image_data)
            or not self.mm_preprocess_cache.enabled
        ):
            # 1. keep preprocessed inputs and cache-off requests on the legacy path
            return await self._process_mm_data_uncached(
                image_data, input_text, request_obj, **kwargs
            )

        # 2. resolve per-image artifacts before composing the current prompt
        artifacts = await self.prepare_media_artifacts(
            image_data,
            content_hashes=request_obj.mm_content_hashes,
        )
        return self.compose_request(input_text, artifacts)

    def get_mm_data(self, prompt, embeddings, **kwargs):
        img_grid_thw = kwargs.get("img_grid_thw", None)
        output = self._build_kimi_mm_data_from_grids(
            prompt=prompt,
            embeddings=embeddings,
            image_token_id=self.mm_tokens.image_token_id,
            img_grid_thw=img_grid_thw,
        )
        image_sizes = kwargs.get("original_image_sizes")
        if image_sizes is None:
            return output

        counts = [self._num_image_tokens_from_grid(grid) for grid in img_grid_thw]
        if len(image_sizes) != len(counts):
            raise ValueError(
                "Expected one original image size for each K3 encoder grid."
            )
        output.input_ids = (
            _expand_k3_image_prompt_token_ids(
                prompt,
                self.mm_tokens.image_token_id,
                counts,
                [tuple(size) for size in image_sizes],
                self._tokenizer,
            )
            .flatten()
            .tolist()
        )

        search_start = 0
        for item, count in zip(output.mm_items, counts):
            start = output.input_ids.index(self.mm_tokens.image_token_id, search_start)
            item.offsets = [(start, start + count - 1)]
            search_start = start + count
        return output
