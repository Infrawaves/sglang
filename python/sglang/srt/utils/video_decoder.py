"""Unified video decoder: torchcodec preferred, decord as fallback."""

import json
import logging
import os
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    from torchcodec.decoders import VideoDecoder

    _BACKEND = "torchcodec"
except (ImportError, RuntimeError):
    _BACKEND = "decord"


_cuda_backend_enabled: bool | None = None


def _try_cuda_backend() -> bool:
    """Try to enable torchcodec CUDA backend. Caches result after first call."""
    global _cuda_backend_enabled
    if _cuda_backend_enabled is not None:
        return _cuda_backend_enabled
    try:
        from torchcodec.decoders import set_cuda_backend

        set_cuda_backend("beta")
        _cuda_backend_enabled = True
    except Exception:
        _cuda_backend_enabled = False
    return _cuda_backend_enabled


def _is_cpu_engine() -> bool:
    # Lazy import to avoid circular dependency issues and unnecessary imports on module load
    from sglang.srt.utils.common import is_cpu

    return is_cpu()


class VideoDecoderWrapper:
    """Unified video decoder that uses torchcodec when available, decord as fallback.

    All frames are returned in NHWC uint8 numpy format for consistency.
    """

    def __init__(self, source, device: str = "cpu", num_decode_threads: int = 0):
        """source: file path (str) or video bytes.
        device: "cpu" or "cuda". GPU decoding only supported with torchcodec.
        num_decode_threads: number of parallel decoder instances for frame
            extraction (torchcodec only). 0 = auto (capped at 16),
            1 = single decoder. Set > 1 to split frame indices across
            multiple decoders in parallel threads.
        """
        self._source = source
        self._num_decode_threads = num_decode_threads
        self._source_bytes = source if isinstance(source, bytes) else None
        self._source_path = source if isinstance(source, str) else None
        self._tmp_path = None
        self._backend = _BACKEND
        if self._backend == "torchcodec":
            kwargs = {"dimension_order": "NHWC"}
            if device == "cuda" and _try_cuda_backend():
                kwargs["device"] = "cuda"
            self._tc_kwargs = kwargs
            try:
                try:
                    self._decoder = VideoDecoder(source, **kwargs)
                except RuntimeError:
                    if "device" in kwargs:
                        logger.warning(
                            "CUDA video decoding failed, falling back to CPU."
                        )
                        kwargs.pop("device")
                        self._tc_kwargs = kwargs
                        self._decoder = VideoDecoder(source, **kwargs)
                    else:
                        raise
            except json.JSONDecodeError as error:
                from sglang.srt.utils.video_metadata_fallback import (
                    SequentialVideoDecoder,
                    is_nonfinite_fps_json_error,
                )

                if not is_nonfinite_fps_json_error(error):
                    raise
                self._decoder = SequentialVideoDecoder(source)
                self._backend = "pyav"
                logger.warning(
                    "Video metadata fallback: non-finite TorchCodec FPS; "
                    "using bounded sequential PyAV decode "
                    "(frames=%d, sampling_fps=%s, size=%dx%d). "
                    "Original timing is not repaired.",
                    len(self._decoder),
                    self._decoder.avg_fps,
                    self._decoder.width,
                    self._decoder.height,
                )
        else:
            from decord import VideoReader, cpu

            if isinstance(source, bytes):
                import tempfile

                fd, tmp_path = tempfile.mkstemp(suffix=".mp4")
                try:
                    os.write(fd, source)
                finally:
                    os.close(fd)
                self._tmp_path = tmp_path
                self._decoder = VideoReader(tmp_path, ctx=cpu(0))
            else:
                self._decoder = VideoReader(source, ctx=cpu(0))

    def __len__(self):
        return len(self._decoder)

    def __getitem__(self, idx):
        """Return one NHWC uint8 frame (numpy on CPU, tensor on CUDA)."""
        if self._backend == "torchcodec":
            frame = self._decoder[idx]
            data = frame.data if hasattr(frame, "data") else frame
            return data if data.is_cuda else data.numpy()
        elif self._backend == "pyav":
            return self._decoder[idx]
        else:
            from decord import DECORDError

            try:
                frame = self._decoder[idx]
            except DECORDError as error:
                return self._decode_with_pyav_fallback("frame index", [idx], error)[0]
            return frame.asnumpy() if hasattr(frame, "asnumpy") else np.array(frame)

    @property
    def avg_fps(self) -> float:
        if self._backend == "torchcodec":
            return self._decoder.metadata.average_fps
        elif self._backend == "pyav":
            return self._decoder.avg_fps
        else:
            return self._decoder.get_avg_fps()

    @property
    def frame_shape(self) -> tuple[int, int]:
        if self._backend == "torchcodec":
            metadata = self._decoder.metadata
            height = getattr(metadata, "height", None)
            width = getattr(metadata, "width", None)
            if height and width:
                return int(height), int(width)
        elif self._backend == "pyav":
            return int(self._decoder.height), int(self._decoder.width)
        shape = self[0].shape
        return int(shape[-3]), int(shape[-2])

    def get_frames_at(self, indices: list):
        """Return NHWC uint8 frames (numpy on CPU, tensor on CUDA)."""
        if self._backend == "torchcodec":
            batch = self._decoder.get_frames_at(indices)
            return batch.data if batch.data.is_cuda else batch.data.numpy()
        elif self._backend == "pyav":
            return self._decoder.get_frames_at(indices)
        else:
            return self._decord_batch(indices)

    def get_frames_as_tensor(self, indices: list, num_threads: Optional[int] = None):
        """Return frames at given indices as a torch tensor (NHWC, uint8, pinned memory).

        num_threads overrides the instance's num_decode_threads for this call
        only. Pass 1 when the caller is *itself* already running one decode per
        thread on its own pool: this decoder's inner pool would then multiply
        against that outer concurrency (outer_workers x inner_threads decoders
        alive at once, each with its own ffmpeg threads and each entering
        OpenMP regions during colour conversion), which is how a decode fleet
        exhausts the process/thread limit. None keeps the instance's setting.
        """
        import torch

        effective_threads = (
            self._num_decode_threads if num_threads is None else num_threads
        )
        if (
            self._backend == "torchcodec"
            and effective_threads != 1
            and len(indices) > 1
        ):
            resolved_threads = effective_threads
            if resolved_threads <= 0:
                resolved_threads = min(os.cpu_count() or 8, 16)
            resolved_threads = min(resolved_threads, len(indices))
            if resolved_threads > 1:
                return self._parallel_decode(indices, resolved_threads)

        if self._backend == "torchcodec":
            batch = self._decoder.get_frames_at(indices)
            if _is_cpu_engine():
                return batch.data
            return batch.data if batch.data.is_cuda else batch.data.pin_memory()
        elif self._backend == "pyav":
            return torch.from_numpy(self._decoder.get_frames_at(indices)).pin_memory()
        else:
            output = torch.from_numpy(self._decord_batch(indices))
            return output if _is_cpu_engine() else output.pin_memory()

    def _decord_batch(self, indices):
        from decord import DECORDError

        try:
            return self._decoder.get_batch(indices).asnumpy()
        except DECORDError as error:
            return self._decode_with_pyav_fallback("get_batch", indices, error)

    def _decode_with_pyav_fallback(self, operation, indices, error):
        # decord's drain loop feeds an empty frame into its filter graph when
        # the decoder reports a non-EOF error near the end of the stream
        # (dmlc/decord#183), so get_batch and frame reads fail on trailing
        # frames of truncated uploads. Sequential PyAV skips the corrupt
        # packets and returns the frames that still decode.
        from sglang.srt.utils.common import CORRUPT_VIDEO_MESSAGE
        from sglang.srt.utils.video_metadata_fallback import (
            SequentialVideoDecoder,
            VideoFallbackLimitError,
            decord_fallback_limits,
        )

        # decord validated the indices against its own frame count and counts
        # negative ones from that end before decoding.
        num_frames = len(self._decoder)
        indices = [index + num_frames if index < 0 else index for index in indices]
        logger.warning(
            "decord %s failed (%s); re-decoding %d frames "
            "(max index %s of %d) with sequential PyAV.",
            operation,
            str(error).split("\n", 1)[0],
            len(indices),
            max(indices, default=None),
            num_frames,
        )
        try:
            # This decode replaces decord's on the same IO-pool thread, so the
            # pool already bounds its concurrency; the shared fallback slots
            # would turn a busy pool into failed requests.
            fallback = SequentialVideoDecoder(
                self._source,
                limits=decord_fallback_limits(num_frames),
                decode_slots=False,
                skip_corrupt_packets=True,
            )
            try:
                # On a truncated stream decord's frame count can exceed what
                # PyAV decodes. Callers derive timestamps from decord's indices
                # and FPS, so trailing indices take the last decodable frame
                # and the request keeps its sampled frame count.
                last = len(fallback) - 1
                if max(indices, default=0) > last:
                    logger.warning(
                        "PyAV decoded %d of decord's %d frames; indices past "
                        "%d use frame %d.",
                        len(fallback),
                        num_frames,
                        last,
                        last,
                    )
                    indices = [min(index, last) for index in indices]
                return fallback.get_frames_at(indices)
            finally:
                fallback.close()
        except (ImportError, MemoryError):
            raise  # missing backend / OOM are server errors
        except VideoFallbackLimitError as limit_error:
            # The limits scale with what decord accepted, so a hit is a cost
            # of this upload: 400, retried nowhere else.
            logger.warning(
                "PyAV fallback after decord %s stopped at a limit: %s",
                operation,
                limit_error,
            )
            raise ValueError(f"Could not decode video: {limit_error}") from limit_error
        except Exception as fallback_error:
            # PyAV errors and a stream with no decodable frame mean the
            # upload itself is unreadable: 400.
            raise ValueError(CORRUPT_VIDEO_MESSAGE) from fallback_error

    def _parallel_decode(self, indices, num_threads):
        """Decode frames using multiple VideoDecoder instances in parallel threads."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        import torch

        chunks = [list(c) for c in np.array_split(indices, num_threads) if len(c) > 0]
        source = self._source
        kwargs = self._tc_kwargs
        cuda_device = None
        if kwargs.get("device") == "cuda":
            cuda_device = torch.cuda.current_device()

        def _decode_chunk(chunk):
            # CUDA's current device is thread-local.  Without this, decoder
            # workers created by TP rank > 0 silently default to GPU 0.
            if cuda_device is not None:
                torch.cuda.set_device(cuda_device)
            d = VideoDecoder(source, **kwargs)
            return d.get_frames_at(chunk).data

        with ThreadPoolExecutor(max_workers=len(chunks)) as executor:
            future_to_idx = {
                executor.submit(_decode_chunk, chunk): idx
                for idx, chunk in enumerate(chunks)
            }
            results = [None] * len(chunks)
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                results[idx] = future.result()

        output = torch.cat(results, dim=0)
        if _is_cpu_engine():
            return output
        return output if output.is_cuda else output.pin_memory()

    @property
    def source_bytes(self) -> bytes | None:
        """Return raw video bytes if available (needed for audio extraction)."""
        if self._source_bytes is not None:
            return self._source_bytes
        path = self._tmp_path or self._source_path
        if path is not None:
            if os.path.isfile(path):
                with open(path, "rb") as f:
                    return f.read()
        return None

    def close(self):
        if getattr(self, "_backend", None) == "pyav" and getattr(
            self, "_decoder", None
        ) is not None:
            self._decoder.close()
        self._decoder = None
        self._source = None
        self._source_bytes = None
        self._source_path = None
        if getattr(self, "_tmp_path", None) is not None:
            if os.path.exists(self._tmp_path):
                os.unlink(self._tmp_path)
            self._tmp_path = None

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
