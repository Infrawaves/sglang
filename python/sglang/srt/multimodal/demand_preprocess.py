"""Demand image preprocessing: metadata first, image features on demand.

An eligible request leaves the tokenizer with prompt tokens, offsets, grids
and identities computed from image headers only; its items carry no feature.
The tokenizer keeps the raw image bytes. Before the scheduler admits the
request's prefill it computes which images the extend range touches, given
the prefix it actually matched, and asks the tokenizer to preprocess only
those. Images lying wholly inside the matched prefix are never decoded.

Message flow (one rid, any number of rounds)::

    tokenizer --TokenizedGenerateReqInput(mm_demand_ticket)--> scheduler
    scheduler --MaterializeImagesReq(ticket, mid, indices)---> tokenizer
    tokenizer --ImagesReadyReq(ticket, mid, indices, items | error)--> scheduler

The tokenizer answers every MaterializeImagesReq exactly once. The scheduler
accepts a reply only when the request is still alive, the ticket matches and
``(mid, indices)`` is the round it is waiting for; anything else is stale and
its transport proxies are released.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple


class ImageDemandStatus(enum.IntEnum):
    NOT_REQUESTED = 0
    REQUESTED = 1
    READY = 2
    FAILED = 3


def needed_image_indices(
    offsets: Sequence[Tuple[int, int]], prefix_len: int
) -> List[int]:
    """Images the extend range starting at ``prefix_len`` touches.

    Offsets are inclusive token spans. An image cut by the prefix boundary is
    needed whole: the encoder embeds whole images, and the extend part reads
    the embedding rows past the boundary.
    """
    return [index for index, (_, end) in enumerate(offsets) if end >= prefix_len]


@dataclass
class ImageDemandState:
    """Scheduler-side readiness of one request's images.

    Every TP rank keeps an identical copy: it is created from the broadcast
    request, advanced only by the scheduling loop (which makes the same
    decisions on every rank) and by broadcast ImagesReadyReq messages.
    """

    ticket: int
    offsets: List[Tuple[int, int]]
    status: List[ImageDemandStatus] = field(default_factory=list)
    # materialization id -> image indices still awaiting that reply
    pending: Dict[int, Tuple[int, ...]] = field(default_factory=dict)
    next_materialization_id: int = 0
    # Prefix length of the admission attempt that last found images missing.
    blocked_prefix_len: int = 0
    error_msg: Optional[str] = None
    status_code: Optional[int] = None

    def __post_init__(self):
        if not self.status:
            self.status = [ImageDemandStatus.NOT_REQUESTED] * len(self.offsets)
        if len(self.status) != len(self.offsets):
            raise ValueError("image demand status must align with image offsets")

    def missing(self, prefix_len: int) -> List[int]:
        """Needed images whose features are not attached yet."""
        return [
            index
            for index in needed_image_indices(self.offsets, prefix_len)
            if self.status[index] != ImageDemandStatus.READY
        ]

    def failed(self, prefix_len: int) -> bool:
        """Whether a needed image can never become ready."""
        return any(
            self.status[index] == ImageDemandStatus.FAILED
            for index in needed_image_indices(self.offsets, prefix_len)
        )

    def blocks(self, prefix_len: int) -> bool:
        """Whether admission at ``prefix_len`` must wait (or fail) for images."""
        if self.failed(prefix_len) or self.missing(prefix_len):
            self.blocked_prefix_len = prefix_len
            return True
        return False

    def begin_request(self, prefix_len: int) -> Optional[Tuple[int, List[int]]]:
        """Open one materialization round for needed, never-requested images.

        Images already requested are not asked for again, so a prefix that
        shrinks between rounds (retract, eviction, failed load-back) only adds
        the newly exposed images.
        """
        indices = [
            index
            for index in needed_image_indices(self.offsets, prefix_len)
            if self.status[index] == ImageDemandStatus.NOT_REQUESTED
        ]
        if not indices:
            return None
        materialization_id = self.next_materialization_id
        self.next_materialization_id += 1
        for index in indices:
            self.status[index] = ImageDemandStatus.REQUESTED
        self.pending[materialization_id] = tuple(indices)
        return materialization_id, indices

    def take_pending(
        self, ticket: int, materialization_id: int, indices: Sequence[int]
    ) -> bool:
        """Claim the round a reply answers; False marks the reply stale."""
        if ticket != self.ticket:
            return False
        expected = self.pending.get(materialization_id)
        if expected is None or expected != tuple(indices):
            return False
        del self.pending[materialization_id]
        return True

    def mark_ready(self, indices: Sequence[int]) -> None:
        for index in indices:
            self.status[index] = ImageDemandStatus.READY

    def mark_failed(
        self, indices: Sequence[int], error_msg: str, status_code: int
    ) -> None:
        for index in indices:
            self.status[index] = ImageDemandStatus.FAILED
        if self.error_msg is None:
            self.error_msg = error_msg
            self.status_code = status_code


@dataclass
class DemandImageSources:
    """Tokenizer-side inputs for later materialization of one request.

    ``sources`` hold the downloaded bytes in the same envelope as the request
    (so the artifact key is unchanged) and are never fetched again.
    ``artifacts`` are the feature-less header-derived artifacts the prompt was
    composed from; a materialized artifact must match them exactly.
    """

    ticket: int
    sources: List[Any]
    content_digests: List[str]
    artifacts: List[Any]
    offsets: List[Tuple[int, int]]
