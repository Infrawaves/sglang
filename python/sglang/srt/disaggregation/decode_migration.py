"""Hand off running decode requests for a planned retirement of the instance.

`pause_generation(mode="migrate")` arms the exporter. Decoding continues until
no request is still receiving KV from prefill, then the scheduler pauses,
detaches the running requests and passes them to `begin`. Each request's
not-yet-stored complete pages are written to the storage backend through the
decode offload manager; once every TP rank's writes for it are acknowledged (or
its deadline passes), the request finishes with `FINISH_MIGRATED` on the normal
output stream. A router resubmits `origin_input_ids + output_ids` to another
prefill/decode pair, whose prefill reads the stored prefix back.

Every rank calls `should_start` and `poll` the same number of times: both run
one all-reduce over the attention-TP group, so ranks agree on when to pause and
which requests to report.
"""

from __future__ import annotations

import logging
import time
from http import HTTPStatus
from typing import TYPE_CHECKING, Callable, Dict, List, Optional

import torch

from sglang.srt.managers.req_types import FINISH_ABORT, FINISH_MIGRATED
from sglang.srt.managers.scheduler_components.request_list import ListKind

if TYPE_CHECKING:
    from sglang.srt.disaggregation.decode_kvcache_offload_manager import (
        DecodeKVCacheOffloadManager,
    )
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.managers.scheduler_components.request_list import ReqList

logger = logging.getLogger(__name__)

# Where TokenizerManager puts GenerateReqInput.max_thinking_tokens.
_THINKING_BUDGET_PARAM = "thinking_budget"

# Lists whose requests the running decode loop still moves forward: KV still
# arriving from prefill, retracted under memory pressure, aborted with KV held
# until the transfer drains, or waiting for a grammar. The paused loop does not
# advance them, so the hand-off waits until they are empty.
_RECEIVE_SIDE_LISTS = (
    ListKind.PD_PREALLOC,
    ListKind.PD_RECEIVING,
    ListKind.PD_RETRACTED,
    ListKind.PD_RELEASING,
    ListKind.GRAMMAR,
    ListKind.CHUNKED,
)


def decode_receive_side_idle(
    *, lists: Dict[ListKind, ReqList], waiting_queue: ReqList
) -> bool:
    """Whether every live request is either decoding or waiting with its KV."""
    if any(lists.get(kind) for kind in _RECEIVE_SIDE_LISTS):
        return False
    return all(req.kv.holds_kv for req in waiting_queue)


def migration_blocker(req: Req) -> Optional[str]:
    """Why `req` cannot be resubmitted as plain tokens, or None if it can.

    A resubmitted request restarts with the generated tokens as its prompt, so
    state that lives only on this instance (grammar matcher, penalty counters,
    logprob history, per-token side outputs, thinking budget, multimodal
    features, adapters) would be lost."""
    if not req.allow_migration:
        return "the router cannot resume this request"
    params = req.sampling_params
    blockers = (
        (req.return_logprob, "logprobs"),
        (req.return_hidden_states_mode.need_capture(), "hidden states"),
        (req.return_routed_experts, "routed experts"),
        (req.return_indexer_topk, "indexer top-k"),
        (req.return_sampling_mask, "a sampling mask"),
        (req.grammar is not None, "structured output"),
        (req.multimodal_inputs is not None, "multimodal input"),
        (req.input_embeds is not None, "input embeddings"),
        (req.custom_logit_processor is not None, "a custom logit processor"),
        (req.session is not None, "a session"),
        (req.lora_id is not None, "a LoRA adapter"),
        (
            params.frequency_penalty != 0.0
            or params.presence_penalty != 0.0
            or params.repetition_penalty != 1.0,
            "sampling penalties",
        ),
        (
            _THINKING_BUDGET_PARAM in (params.custom_params or {}),
            "a thinking budget",
        ),
    )
    for blocked, what in blockers:
        if blocked:
            return f"requests with {what} cannot be migrated"
    return None


class DecodeMigrationExporter:
    """Freeze, store and hand off decode requests; see the module docstring."""

    def __init__(
        self,
        *,
        offload_manager: DecodeKVCacheOffloadManager,
        emit_outputs: Callable[[List[Req]], None],
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._offload = offload_manager
        self._emit_outputs = emit_outputs
        self._clock = clock
        self._tp_group = offload_manager.tp_group
        self._tp_world_size = offload_manager.tp_world_size
        # Set while armed: pause requested, requests not yet detached.
        self._armed_timeout_s: Optional[float] = None
        self._flush_timeout_s = 0.0
        # Detached requests not yet reported, with their deadlines, in a fixed
        # order shared by all ranks.
        self._pending: Dict[Req, float] = {}
        # Pending requests whose flush found the host pool full.
        self._awaiting_host_memory: List[Req] = []
        # The hand-off reason of each sealed pending request, completed with the
        # stored length when it is reported.
        self._reasons: Dict[Req, FINISH_MIGRATED] = {}
        # Reported, but copies or backups still hold device or host memory.
        self._draining = False
        # Whether the armed pause has detached and handed off the requests.
        self._handed_off = False

    def arm(self, *, flush_timeout_s: float) -> None:
        if self.has_pending():
            logger.warning("Decode migration already in progress; ignoring request.")
            return
        self._armed_timeout_s = flush_timeout_s
        self._handed_off = False

    def is_armed(self) -> bool:
        return self._armed_timeout_s is not None

    def is_done(self) -> bool:
        """Whether a migrate pause was issued and every hand-off it started,
        with its copies and backups, has finished."""
        return self._handed_off and not self.is_armed() and not self.has_pending()

    def should_start(self, *, receive_side_idle: bool) -> bool:
        """Whether every rank is armed and has no request receiving KV."""
        if self._armed_timeout_s is None:
            return False
        return self._all_ranks_agree([receive_side_idle])[0]

    def begin(self, reqs: List[Req]) -> None:
        """Start handing off `reqs`, already detached from every request list."""
        assert self._armed_timeout_s is not None
        self._flush_timeout_s = self._armed_timeout_s
        self._armed_timeout_s = None
        self._handed_off = True
        deadline = self._clock() + self._flush_timeout_s
        rejected = []
        for req in reqs:
            reason = migration_blocker(req)
            if reason is not None:
                self._finish(req, FINISH_ABORT(reason, HTTPStatus.SERVICE_UNAVAILABLE))
                rejected.append(req)
                continue
            self._pending[req] = deadline
            self._flush(req)
        if rejected:
            self._emit_outputs(rejected)
        self._draining = True
        logger.info(
            "Decode migration started: %d requests to hand off, %d rejected.",
            len(self._pending),
            len(rejected),
        )

    def poll(self) -> None:
        """Advance copies and backups; report the requests all ranks finished."""
        if not self.has_pending():
            return
        self._offload.check_offload_progress()
        now = self._clock()
        self._retry_flushes(now)
        pending = list(self._pending)
        local = [self._is_ready(req, now) for req in pending]
        local.append(not self._offload.has_pending_transfers())
        agreed = self._all_ranks_agree(local)
        ready = [req for req, ok in zip(pending, agreed) if ok]
        for req in ready:
            del self._pending[req]
            reason = self._reasons.pop(req)
            reason.kv_flushed_tokens = self._offload.storage_confirmed_tokens(req)
        if ready:
            self._emit_outputs(ready)
        if not self._pending and agreed[-1]:
            self._draining = False
            logger.info("Decode migration finished.")

    def has_pending(self) -> bool:
        """Whether requests or their copies and backups are still in flight."""
        return bool(self._pending) or self._draining

    def _flush(self, req: Req) -> None:
        queued = self._offload.flush_for_migration(req)
        if queued < 0:
            self._awaiting_host_memory.append(req)
            return
        self._seal(req)

    def _retry_flushes(self, now: float) -> None:
        waiting, self._awaiting_host_memory = self._awaiting_host_memory, []
        for req in waiting:
            if now >= self._pending[req]:
                logger.warning(
                    "Decode migration of %s gave up waiting for host memory.",
                    req.rid,
                )
                self._seal(req)
            else:
                self._flush(req)

    def _seal(self, req: Req) -> None:
        """Mark `req` finished; its device KV is released once no copy reads it.

        Finishing must wait for a successful flush: an earlier copy's ack
        releases a finished request's device KV, which a later flush still
        needs to read."""
        reason = FINISH_MIGRATED(
            origin_input_ids=req.origin_input_ids,
            output_ids=req.output_ids,
        )
        self._reasons[req] = reason
        self._finish(req, reason)

    def _finish(self, req: Req, reason) -> None:
        req.finish_outside_step(reason)
        # With no copy in flight, no ack will come to release the device KV.
        self._offload.finalize_release_on_finish(req)

    def _is_ready(self, req: Req, now: float) -> bool:
        if req in self._awaiting_host_memory:
            return False
        if self._offload.has_pending_device_copy(req):
            return False
        if self._offload.has_pending_storage_write(req):
            return now >= self._pending[req]
        return True

    def _all_ranks_agree(self, flags: List[bool]) -> List[bool]:
        votes = torch.tensor(flags, dtype=torch.int)
        if self._tp_world_size > 1:
            torch.distributed.all_reduce(
                votes, op=torch.distributed.ReduceOp.MIN, group=self._tp_group
            )
        return [bool(v) for v in votes.tolist()]
