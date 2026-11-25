"""Bounded sequential PyAV decoding for videos TorchCodec or decord cannot read."""

import io
import json
import math
import operator
import os
import re
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np


def _positive_setting(name, default):
    value = type(default)(os.environ.get(name, default))
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


_DECODE_SLOTS = threading.BoundedSemaphore(
    _positive_setting("SGLANG_VIDEO_FALLBACK_CONCURRENCY", 2)
)


def is_nonfinite_fps_json_error(error):
    if not isinstance(error, json.JSONDecodeError):
        return False
    if os.environ.get("SGLANG_VIDEO_METADATA_FALLBACK", "1").lower() in (
        "0",
        "false",
        "off",
    ):
        return False
    return bool(
        re.search(r'"averageFps(?:FromHeader)?"\s*:\s*$', error.doc[: error.pos])
        and re.match(r"-?(?:inf|nan)(?=\s*[,}])", error.doc[error.pos :])
    )


class VideoFallbackLimitError(ValueError):
    """A configured bound of the sequential decode was exceeded."""


@dataclass(frozen=True)
class _FallbackLimits:
    # None leaves a dimension bounded only by the video itself.
    max_input_bytes: int | None = 64 * 1024 * 1024
    max_frames: int | None = 2048
    max_frame_pixels: int | None = 4096 * 2160
    max_total_pixels: int | None = 256 * 1024 * 1024
    max_output_bytes: int | None = 64 * 1024 * 1024
    timeout_seconds: float | None = 10.0

    @classmethod
    def from_env(cls):
        defaults = cls()
        return cls(
            **{
                field.name: _positive_setting(
                    f"SGLANG_VIDEO_FALLBACK_{field.name.upper()}",
                    getattr(defaults, field.name),
                )
                for field in fields(cls)
            }
        )


# Per-pass decode budget for re-decoding a video decord already opened. PyAV
# runs single-threaded here, so a pass over N frames gets 10 s plus 0.1 s per
# frame (a 10 frames/s floor): the deadline then tracks the length decord
# accepted and trips on a stalled stream rather than on a long one.
_DECORD_FALLBACK_SECONDS_PER_FRAME = 0.1


def decord_fallback_limits(num_frames):
    """Limits for re-decoding frames of a video decord opened and sized.

    decord read the same source, so its size and resolution need no separate
    cap. get_frames_at allocates len(indices) * H * W * 3 bytes, the same
    array decord's get_batch would have returned for the request.
    """
    return _FallbackLimits(
        max_input_bytes=None,
        max_frames=None,
        max_frame_pixels=None,
        max_total_pixels=None,
        max_output_bytes=None,
        timeout_seconds=_FallbackLimits.timeout_seconds
        + num_frames * _DECORD_FALLBACK_SECONDS_PER_FRAME,
    )


def _positive_rate(value):
    try:
        number = float(value)
    except (TypeError, ValueError, ZeroDivisionError, OverflowError):
        return None
    return number if math.isfinite(number) and number > 0 else None


class SequentialVideoDecoder:
    """Scan for a bounded frame count, then extract by ordinal without seeking.

    This recovers pictures, not the original timing of a malformed video.
    Deadlines are cooperative: they cannot interrupt a blocked native call.

    decode_slots=True shares the process-wide _DECODE_SLOTS with other
    sequential decodes; callers whose concurrency is already bounded elsewhere
    pass False. skip_corrupt_packets=True decodes past packets the decoder
    rejects (see _decode); by default any decode error ends the scan.
    """

    def __init__(
        self, source, limits=None, decode_slots=True, skip_corrupt_packets=False
    ):
        self._limits = _FallbackLimits.from_env() if limits is None else limits
        self._decode_slots = decode_slots
        self._skip_corrupt_packets = skip_corrupt_packets
        self._source = source
        self._closed = False
        if self._limits.max_input_bytes is not None:
            source_size = (
                len(source)
                if isinstance(source, bytes)
                else Path(source).stat().st_size
            )
            if source_size > self._limits.max_input_bytes:
                raise VideoFallbackLimitError(
                    "Video metadata fallback exceeded the encoded-byte limit"
                )

        with self._open() as (container, stream, deadline):
            self.width = stream.codec_context.width
            self.height = stream.codec_context.height
            self._check_dimensions(self.width, self.height)
            if (
                self._limits.max_frames is not None
                and stream.frames > self._limits.max_frames
            ):
                raise VideoFallbackLimitError(
                    "Video metadata fallback exceeded the source-frame limit"
                )
            average_rate = _positive_rate(stream.average_rate)
            duration = (
                _positive_rate(stream.duration * stream.time_base)
                if stream.duration is not None and stream.time_base is not None
                else None
            )
            guessed_rate = _positive_rate(stream.guessed_rate)
            self._frame_count = sum(
                1 for _ in self._frames(container, stream, deadline)
            )

        if self._frame_count == 0:
            raise ValueError("Video metadata fallback decoded no frames")
        self.avg_fps = (
            average_rate
            or (_positive_rate(self._frame_count / duration) if duration else None)
            or guessed_rate
        )
        if self.avg_fps is None:
            raise ValueError("Video metadata fallback found no finite positive FPS")

    @contextmanager
    def _open(self):
        if self._closed:
            raise ValueError("Video metadata fallback decoder is closed")
        import av

        timeout = self._limits.timeout_seconds
        deadline = None if timeout is None else time.monotonic() + timeout
        if self._decode_slots and not _DECODE_SLOTS.acquire(timeout=timeout):
            raise VideoFallbackLimitError(
                "Video metadata fallback timed out waiting for a decode slot"
            )
        try:
            self._check_deadline(deadline)
            source = (
                io.BytesIO(self._source)
                if isinstance(self._source, bytes)
                else self._source
            )
            options = (
                {}
                if self._limits.max_frame_pixels is None
                else {"max_pixels": str(self._limits.max_frame_pixels)}
            )
            with av.open(source, mode="r", options=options) as container:
                if not container.streams.video:
                    raise ValueError("Video metadata fallback found no video stream")
                stream = container.streams.video[0]
                stream.codec_context.thread_count = 1
                stream.codec_context.thread_type = "NONE"
                self._check_deadline(deadline)
                yield container, stream, deadline
        finally:
            if self._decode_slots:
                _DECODE_SLOTS.release()

    def _check_deadline(self, deadline):
        if deadline is not None and time.monotonic() >= deadline:
            raise VideoFallbackLimitError(
                "Video metadata fallback exceeded its decode time budget"
            )

    def _check_dimensions(self, width, height):
        if width <= 0 or height <= 0:
            raise ValueError("Video metadata fallback found empty frame dimensions")
        max_pixels = self._limits.max_frame_pixels
        if max_pixels is not None and width * height > max_pixels:
            raise VideoFallbackLimitError(
                "Video metadata fallback exceeded its frame-dimension limit"
            )

    @staticmethod
    def _decode(container, stream):
        # A truncated upload ends in a partial packet: send_packet rejects it
        # with InvalidDataError, or the demuxer stops at a sample it cannot
        # read. The decoder state survives a rejected packet, so decoding
        # continues with the next one, and a demux error drains the frames
        # already sent before ending the stream.
        from av.error import InvalidDataError

        packets = container.demux(stream)
        while True:
            try:
                packet = next(packets)
            except StopIteration:
                return
            except InvalidDataError:
                packet = None
            try:
                frames = (
                    stream.codec_context.decode(None)
                    if packet is None
                    else packet.decode()
                )
            except InvalidDataError:
                frames = ()
            yield from frames
            if packet is None:
                return

    def _frames(self, container, stream, deadline):
        total_pixels = 0
        decoded = (
            self._decode(container, stream)
            if self._skip_corrupt_packets
            else container.decode(stream)
        )
        for frame_index, frame in enumerate(decoded):
            self._check_deadline(deadline)
            self._check_dimensions(frame.width, frame.height)
            if (frame.width, frame.height) != (self.width, self.height):
                raise ValueError(
                    "Video metadata fallback does not support changing dimensions"
                )
            total_pixels += frame.width * frame.height
            if (
                self._limits.max_frames is not None
                and frame_index >= self._limits.max_frames
            ):
                raise VideoFallbackLimitError(
                    "Video metadata fallback exceeded the source-frame limit"
                )
            if (
                self._limits.max_total_pixels is not None
                and total_pixels > self._limits.max_total_pixels
            ):
                raise VideoFallbackLimitError(
                    "Video metadata fallback exceeded the decoded-pixel limit"
                )
            yield frame
        self._check_deadline(deadline)

    def __len__(self):
        return self._frame_count

    def __getitem__(self, index):
        index = operator.index(index)
        if index < 0:
            index += len(self)
        return self.get_frames_at([index])[0]

    def get_frames_at(self, indices):
        if self._closed:
            raise ValueError("Video metadata fallback decoder is closed")
        indices = [operator.index(index) for index in indices]
        if any(index < 0 or index >= len(self) for index in indices):
            raise IndexError("Video metadata fallback frame index is out of range")
        output_bytes = len(indices) * self.height * self.width * 3
        max_output_bytes = self._limits.max_output_bytes
        if max_output_bytes is not None and output_bytes > max_output_bytes:
            raise VideoFallbackLimitError(
                "Video metadata fallback exceeded the output-byte limit"
            )
        if not indices:
            return np.empty((0, self.height, self.width, 3), dtype=np.uint8)
        positions = {}
        for position, frame_index in enumerate(indices):
            positions.setdefault(frame_index, []).append(position)
        with self._open() as (container, stream, deadline):
            output = np.empty(
                (len(indices), self.height, self.width, 3), dtype=np.uint8
            )
            for frame_index, frame in enumerate(
                self._frames(container, stream, deadline)
            ):
                if frame_index in positions:
                    pixels = frame.to_ndarray(format="rgb24")
                    self._check_deadline(deadline)
                    for position in positions.pop(frame_index):
                        output[position] = pixels
                if not positions:
                    self._check_deadline(deadline)
                    return output
        raise ValueError("Video metadata fallback ended before the requested frame")

    def close(self):
        self._closed = True
        self._source = None
