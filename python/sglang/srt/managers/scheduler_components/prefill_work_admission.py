"""Scheduler-owned admission for request-pull prefill workers.

The ledger charges a prompt's KV until the scheduler's safe retirement
boundary, including bootstrap and source KV writes, and its uncached compute
until its last chunk is allocated. Network timeouts can revoke a grant that has
not been consumed; they can never release an executing request.
"""

from __future__ import annotations

import asyncio
import hmac
import logging
import secrets
import time
import uuid
from collections import OrderedDict
from typing import Any, Callable, Optional

import msgspec

logger = logging.getLogger(__name__)


class PrefillCapacity(msgspec.Struct, frozen=True):
    tokens: int
    requests: int
    metadata: int
    # Free plus evictable recurrent-state slots; None without a state pool.
    state_slots: Optional[int] = None


class _Reservation(msgspec.Struct):
    tokens: int
    expires_at: float
    assignment_id: Optional[str] = None
    request_id: Optional[str] = None
    consumed: bool = False
    request: Any = None
    # Push and internal work is unvalidated: input embeds, multimodal pads and
    # session context extend the prompt after intake, so follow the request.
    follow_request: bool = False


# A prefill server decodes no tokens: PrefillBootstrapQueue sets every
# request's max_new_tokens to 1 before admission.
_PREFILL_OUTPUT_TOKENS = 1
# Admission keeps one more token free than it charges: the strict "<" of
# check_prefill, and IGNORE_EOS_RESERVE_TOKENS on the ignore_eos path.
_ADMISSION_SLACK_TOKENS = 1


class PrefillHeadroom(msgspec.Struct, frozen=True, array_like=True):
    """What one more request may still take; TP ranks reduce it field-wise by MIN."""

    # max_tokens minus prompt tokens still to prefill; positive admits one more.
    work_tokens: int
    # Longest prompt whose KV, one output token and state still fit.
    kv_tokens: int
    requests: int


class PrefillWorkAdmission:
    """Serial scheduler ledger; the caller supplies group-consistent snapshots.

    Work counts prompt tokens still to prefill: a whole prompt until the request
    reaches the scheduler, then its uncached remainder, then 0 once its last
    chunk is allocated. KV promises cover whole prompts until retirement,
    because cache hits occupy KV as well. Allocated KV is already absent from
    the allocator snapshot, so only the remaining promise is subtracted.

    A reservation holds a request slot only; bind re-checks every budget, so
    concurrent reservations cannot overcommit.
    """

    def __init__(
        self,
        *,
        max_tokens: int,
        max_requests: int,
        page_size: int = 1,
        request_overhead_tokens: Optional[int] = None,
        safety_margin_tokens: int = 0,
        reservation_ttl: float = 120.0,
        fresh_state_slots: int = 0,
        state_slot_cost: int = 0,
        pending_state_slots: Callable[[Any], int] = lambda req: 0,
        pending_work_tokens: Callable[[Any], Optional[int]] = lambda req: None,
    ):
        if max_tokens <= 0 or max_requests <= 0 or page_size <= 0:
            raise ValueError("Prefill pull requires positive token/request/page limits")
        if safety_margin_tokens < 0 or reservation_ttl <= 0:
            raise ValueError("Invalid prefill pull safety margin or reservation TTL")
        if fresh_state_slots < 0 or state_slot_cost < 0:
            raise ValueError("Invalid prefill pull state slot accounting")
        self.max_tokens = max_tokens
        self.max_requests = max_requests
        self.page_size = page_size
        # The scheduler's per-request allocation rounding reserve
        # (`PrefillAdder.per_req_token_overhead`): one logical page, or one per
        # shard class for a page-interleaved pool.
        self.request_overhead_tokens = (
            page_size if request_overhead_tokens is None else request_overhead_tokens
        )
        self.safety_margin_tokens = safety_margin_tokens
        self.reservation_ttl = reservation_ttl
        # Slots acquire takes for a request that holds none yet.
        self.fresh_state_slots = fresh_state_slots
        # FULL-token equivalents of one state slot in a shared arena; 0 when
        # state and token pools do not compete for the same bytes.
        self.state_slot_cost = state_slot_cost
        self.pending_state_slots = pending_state_slots
        # Prompt tokens a scheduler request still has to prefill; None charges
        # the whole prompt.
        self.pending_work_tokens = pending_work_tokens
        self.reservations: dict[str, _Reservation] = {}
        # Bounded replay history. Old UUIDs are never reissued by the agent.
        self._terminal: OrderedDict[str, str] = OrderedDict()

    def _remember(self, reservation_id: str, state: str) -> None:
        self._terminal[reservation_id] = state
        self._terminal.move_to_end(reservation_id)
        while len(self._terminal) > 8192:
            self._terminal.popitem(last=False)

    @staticmethod
    def _lapsed(reservation: _Reservation, now: float) -> bool:
        # Only an unbound grant expires. Once bound, the router may already
        # have committed it and paired D, so a slow intake (paused engine,
        # tokenizer backlog) must still be admitted; the agent releases a
        # bound grant itself on every exit path.
        return (
            not reservation.consumed
            and reservation.assignment_id is None
            and now >= reservation.expires_at
        )

    def _expire(self, now: float) -> None:
        for reservation_id, reservation in list(self.reservations.items()):
            if self._lapsed(reservation, now):
                del self.reservations[reservation_id]
                self._remember(reservation_id, "expired")

    def _kv_charge(self, tokens: int) -> int:
        # What the scheduler reserves to admit the prompt: its paged length,
        # the per-request rounding reserve and P's single output token.
        paged = -(-tokens // self.page_size) * self.page_size
        return (
            paged
            + self.request_overhead_tokens
            + _PREFILL_OUTPUT_TOKENS
            + _ADMISSION_SLACK_TOKENS
        )

    @staticmethod
    def _prompt_tokens(reservation: _Reservation) -> int:
        req = reservation.request
        if reservation.follow_request and req is not None:
            return max(reservation.tokens, _prompt_len(req))
        return reservation.tokens

    def _work_charge(self, reservation: _Reservation) -> int:
        if reservation.assignment_id is None and not reservation.consumed:
            return 0
        tokens = self._prompt_tokens(reservation)
        req = reservation.request
        pending = None if req is None else self.pending_work_tokens(req)
        return tokens if pending is None else min(pending, tokens)

    def _kv_promise(self, reservation: _Reservation) -> int:
        if reservation.assignment_id is None and not reservation.consumed:
            return 0
        req = reservation.request
        allocated = 0 if req is None else req.kv.kv_allocated_len
        # Prefix-sharing is deliberately not discounted. A pessimistic
        # promise cannot over-admit after an eviction or a cache miss.
        return max(0, self._kv_charge(self._prompt_tokens(reservation)) - allocated)

    def _headroom(
        self,
        capacity: PrefillCapacity,
        *,
        exclude: Optional[str] = None,
        lapsed_at: Optional[float] = None,
    ) -> PrefillHeadroom:
        """Budget left for one more request, ignoring reservation `exclude` and,
        given `lapsed_at`, grants lapsed by then without expiring them."""
        work = 0
        promised_kv = 0
        promised_rows = 0
        promised_metadata = 0
        promised_state = 0
        held = 0
        for reservation_id, reservation in self.reservations.items():
            if reservation_id == exclude or (
                lapsed_at is not None and self._lapsed(reservation, lapsed_at)
            ):
                continue
            held += 1
            req = reservation.request
            work += self._work_charge(reservation)
            promised_kv += self._kv_promise(reservation)
            promised_rows += req is None or req.kv.req_pool_idx is None
            promised_metadata += req is None or req.metadata_buffer_index == -1
            promised_state += (
                self.fresh_state_slots if req is None else self.pending_state_slots(req)
            )
        # The new request's own state slots compete for the token arena as well.
        state_tokens = (promised_state + self.fresh_state_slots) * self.state_slot_cost
        # The longest prompt both scheduler admission paths take. The normal
        # path needs L + output + overhead < free. The ignore_eos path (taken
        # without a radix cache) needs paged(L) + overhead + output + slack <
        # free, so paged(L) <= free - overhead - output - slack - 1. The
        # second is stricter by at most a page; the client picks ignore_eos,
        # so grant only what both admit. A larger grant was committed and
        # paired with D, then stayed in the waiting queue with NO_TOKEN.
        free = capacity.tokens - promised_kv - state_tokens - self.safety_margin_tokens
        largest = (
            free
            - self.request_overhead_tokens
            - _PREFILL_OUTPUT_TOKENS
            - _ADMISSION_SLACK_TOKENS
            - 1
        )
        kv_tokens = largest // self.page_size * self.page_size
        requests = min(
            self.max_requests - held,
            capacity.requests - promised_rows,
            capacity.metadata - promised_metadata,
        )
        if capacity.state_slots is not None and self.fresh_state_slots > 0:
            requests = min(
                requests,
                (capacity.state_slots - promised_state) // self.fresh_state_slots,
            )
        return PrefillHeadroom(
            work_tokens=max(0, self.max_tokens - work),
            kv_tokens=max(0, kv_tokens),
            requests=max(0, requests),
        )

    def headroom(
        self,
        capacity: PrefillCapacity,
        *,
        now: float,
        exclude: Optional[str] = None,
    ) -> PrefillHeadroom:
        """Local headroom after expiring unconsumed reservations."""
        self._expire(now)
        return self._headroom(capacity, exclude=exclude)

    def peek_headroom(self, capacity: PrefillCapacity, *, now: float):
        """Headroom without mutation, using an estimate of the shared clock."""
        return self._headroom(capacity, lapsed_at=now)

    def peek(self, headroom: Optional[PrefillHeadroom]) -> dict:
        # A long poll's size hint. It holds nothing, so an empty router's
        # poll cannot keep a request slot from routers that have work.
        if headroom is None:
            return self._result(False, error="capacity unavailable")
        if (
            headroom.work_tokens <= 0
            or headroom.kv_tokens <= 0
            or headroom.requests <= 0
        ):
            return self._result(False, error="no_capacity")
        return self._result(True, {"max_tokens": headroom.kv_tokens})

    def summary(self) -> dict:
        """Counts for the periodic ledger log; all zero on an idle worker."""
        reservations = self.reservations.values()
        return {
            "reservations": len(self.reservations),
            "reserved": sum(
                r.assignment_id is None and not r.consumed and not r.follow_request
                for r in reservations
            ),
            "bound": sum(
                r.assignment_id is not None and not r.consumed for r in reservations
            ),
            "consumed": sum(r.consumed and not r.follow_request for r in reservations),
            "push_or_internal": sum(r.follow_request for r in reservations),
            "work_tokens": sum(self._work_charge(r) for r in reservations),
            "kv_promised": sum(self._kv_promise(r) for r in reservations),
        }

    @staticmethod
    def _result(success: bool, data=None, error=None) -> dict:
        return {"success": success, "data": data or {}, "error": error}

    def control(
        self,
        *,
        action: str,
        reservation_id: str,
        now: float,
        capacity: Optional[PrefillCapacity] = None,
        capacity_limit: Optional[PrefillHeadroom] = None,
        assignment_id: Optional[str] = None,
        tokens: Optional[int] = None,
        request_id: Optional[str] = None,
    ) -> dict:
        self._expire(now)
        if not reservation_id:
            return self._result(False, error="reservation_id is required")
        reservation = self.reservations.get(reservation_id)

        def limit(exclude: Optional[str]) -> Optional[PrefillHeadroom]:
            # A TP group passes its MIN-reduced headroom; alone, compute it here.
            if capacity_limit is not None:
                return capacity_limit
            if capacity is None:
                return None
            return self._headroom(capacity, exclude=exclude)

        if action == "peek":
            return self.peek(limit(None))
        if action == "reserve":
            if reservation_id in self._terminal:
                return self._result(False, error=self._terminal[reservation_id])
            if reservation is None:
                headroom = limit(None)
                if headroom is None:
                    return self._result(False, error="capacity unavailable")
                if (
                    headroom.work_tokens <= 0
                    or headroom.kv_tokens <= 0
                    or headroom.requests <= 0
                ):
                    return self._result(False, error="no_capacity")
                # The grant is advisory: bind re-checks against current budgets.
                reservation = _Reservation(
                    tokens=headroom.kv_tokens,
                    expires_at=now + self.reservation_ttl,
                )
                self.reservations[reservation_id] = reservation
            if reservation.assignment_id is not None or reservation.consumed:
                return self._result(False, error="reservation already bound")
            return self._result(
                True,
                {
                    "reservation_id": reservation_id,
                    "max_tokens": reservation.tokens,
                    "max_requests": 1,
                    "expires_in_ms": max(0, int((reservation.expires_at - now) * 1000)),
                },
            )
        if action == "query":
            state = self._terminal.get(reservation_id, "unknown")
            if reservation is not None:
                state = (
                    "consumed"
                    if reservation.consumed
                    else ("bound" if reservation.assignment_id else "reserved")
                )
            return self._result(True, {"state": state})
        if action == "release":
            if reservation is not None and reservation.consumed:
                return self._result(False, {"state": "consumed"}, "already_consumed")
            if reservation is not None:
                del self.reservations[reservation_id]
                self._remember(reservation_id, "released")
            return self._result(
                True, {"state": self._terminal.get(reservation_id, "unknown")}
            )
        if action != "bind":
            return self._result(False, error="unknown action")
        if reservation is None:
            return self._result(
                False, error=self._terminal.get(reservation_id, "unknown reservation")
            )
        if not assignment_id or not request_id or tokens is None or tokens <= 0:
            return self._result(
                False,
                error="bind requires assignment_id, request_id and positive tokens",
            )
        if reservation.consumed:
            return self._result(False, error="already_consumed")
        if reservation.assignment_id is not None:
            same = (
                reservation.assignment_id == assignment_id
                and reservation.request_id == request_id
                and reservation.tokens == tokens
            )
            return self._result(
                same, {"state": "bound"}, None if same else "already bound"
            )
        if tokens > reservation.tokens:
            return self._result(False, error="request exceeds reserved tokens")
        # Other claims may have bound since this grant; admit against now.
        headroom = limit(reservation_id)
        if headroom is None:
            return self._result(False, error="capacity unavailable")
        if (
            headroom.work_tokens <= 0
            or headroom.requests <= 0
            or tokens > headroom.kv_tokens
        ):
            return self._result(False, error="no_capacity")
        reservation.tokens = tokens
        reservation.assignment_id = assignment_id
        reservation.request_id = request_id
        return self._result(True, {"state": "bound"})

    def consume(
        self,
        *,
        reservation_id: Optional[str],
        assignment_id: Optional[str],
        request_id: str,
        tokens: int,
        now: Optional[float] = None,
    ) -> Optional[str]:
        # Without `now`, expiry stays where control actions already ran it on a
        # group-agreed clock, so intake needs no per-request clock broadcast.
        if now is not None:
            self._expire(now)
        reservation = self.reservations.get(reservation_id)
        if reservation is None:
            return "Missing or expired prefill pull reservation"
        if reservation.consumed:
            return "Prefill pull reservation was already consumed"
        if (
            not assignment_id
            or reservation.assignment_id != assignment_id
            or reservation.request_id != request_id
            or tokens <= 0
            or tokens > reservation.tokens
        ):
            return "Prefill pull assignment, request identity or token count mismatch"
        reservation.consumed = True
        # Router text costs are conservative bounds. The normal tokenizer
        # supplies the exact prompt length before this atomic handoff.
        reservation.tokens = tokens
        return None

    def attach_request(self, reservation_id: Optional[str], req: Any) -> None:
        reservation = self.reservations.get(reservation_id)
        if reservation is not None and reservation.consumed:
            reservation.request = req

    def admit_internal(
        self, request_id: str, tokens: int, *, kind: str = "internal"
    ) -> str:
        """Charge work that arrives without a reservation: trusted health/warmup
        probes, or push dispatches when push and pull share this worker."""
        # Unique per admission: clients may reuse a rid, and a second request
        # under the same key would silently replace the first one's charge.
        reservation_id = f"{kind}:{request_id}:{uuid.uuid4().hex}"
        self.reservations[reservation_id] = _Reservation(
            tokens=tokens,
            expires_at=float("inf"),
            request_id=request_id,
            consumed=True,
            follow_request=True,
        )
        return reservation_id

    def retire(self, req: Any) -> None:
        # Identity, rather than rid equality, protects another attempt's charge.
        for reservation_id, reservation in list(self.reservations.items()):
            if reservation.request is req:
                del self.reservations[reservation_id]
                self._remember(reservation_id, "retired")
                return

    def finish_intake(self, reservation_id: Optional[str], failed: bool) -> None:
        reservation = self.reservations.get(reservation_id)
        if reservation is None or not reservation.consumed:
            return
        req = reservation.request
        # Validation errors may stream an abort before entering any queue.
        # Once a sender exists, only safe scheduler retirement can release it.
        if req is None or (req.disagg_kv_sender is None and (failed or req.finished())):
            del self.reservations[reservation_id]
            self._remember(reservation_id, "retired")


class SchedulerPrefillWorkAdmission:
    """Narrow runtime adapter: resource snapshots and TP consensus, no network."""

    def __init__(
        self,
        *,
        max_tokens: int,
        max_requests: int,
        safety_margin_tokens: int,
        pool_stats,
        req_pool,
        metadata_allocator,
        page_size,
        cpu_group,
        group_rank,
        group_size,
        token_to_kv_pool_allocator=None,
        tree_cache=None,
        allow_push: bool = False,
        logical_page_size: Optional[int] = None,
    ):
        self.allow_push = allow_push
        stats = pool_stats.get_pool_stats()
        metrics = sorted(u.metrics for u in pool_stats.storage.usage())
        if (
            stats.is_hybrid_swa
            or stats.is_hisparse
            or metrics not in (["full"], ["full", "mamba"])
        ):
            raise ValueError(
                "Prefill pull supports a dense/MLA KV pool with an optional recurrent-state pool; SWA and HiSparse require separate pool admission"
            )
        state = (
            token_to_kv_pool_allocator.state
            if token_to_kv_pool_allocator is not None
            else None
        )
        if stats.is_hybrid_ssm and state is None:
            raise ValueError("Prefill pull needs the allocator's state slots")
        self.has_state_pool = stats.is_hybrid_ssm
        storage_hit_end = (
            (
                lambda req: tree_cache.get_pending_storage_hit_end(
                    req.cache_request_handle
                )
            )
            if tree_cache is not None
            else (lambda req: None)
        )
        self.ledger = PrefillWorkAdmission(
            max_tokens=max_tokens,
            max_requests=max_requests,
            page_size=page_size,
            request_overhead_tokens=_request_overhead_tokens(
                allocator=token_to_kv_pool_allocator,
                logical_page_size=logical_page_size or page_size,
            ),
            safety_margin_tokens=safety_margin_tokens,
            pending_work_tokens=lambda req: pending_prefill_tokens(
                req, storage_hit_end=storage_hit_end
            ),
            **(
                _state_slot_accounting(
                    allocator=token_to_kv_pool_allocator,
                    req_pool=req_pool,
                    tree_cache=tree_cache,
                )
                if self.has_state_pool
                else {}
            ),
        )
        self.pool_stats = pool_stats
        self.req_pool = req_pool
        self.metadata_allocator = metadata_allocator
        self.cpu_group = cpu_group
        self.group_rank = group_rank
        self.group_size = group_size
        # Grants use the first rank's monotonic clock. Hosts have different
        # clock origins, so peeks translate local time using the last broadcast.
        # Before any control call there are no grants whose TTL needs checking.
        self._clock_offset = 0.0
        # Whether this rank logs the ledger (the group's first); resolved on
        # first use, as the process group may not exist yet at construction.
        self._logs_ledger: Optional[bool] = None
        self._logged_at = float("-inf")

    def _now(self) -> float:
        now = time.monotonic()
        if self.group_size > 1:
            import torch

            values = [now]
            torch.distributed.broadcast_object_list(
                values, src=self.group_rank, group=self.cpu_group
            )
            now = values[0]
            # Anchor after receipt: peek can lag by the broadcast latency, but
            # cannot expire a grant early because it waited for the broadcast.
            self._clock_offset = now - time.monotonic()
        return now

    def _capacity(self, paused: bool) -> PrefillCapacity:
        if paused:
            return PrefillCapacity(
                tokens=0,
                requests=0,
                metadata=0,
                state_slots=0 if self.has_state_pool else None,
            )
        stats = self.pool_stats.get_pool_stats()
        return PrefillCapacity(
            tokens=stats.full_available_size + stats.full_evictable_size,
            requests=self.req_pool.available_size(),
            metadata=self.metadata_allocator.available_size(),
            state_slots=(
                stats.mamba_available_size + stats.mamba_evictable_size
                if self.has_state_pool
                else None
            ),
        )

    def _reduce(self, headroom: PrefillHeadroom) -> PrefillHeadroom:
        if self.group_size <= 1:
            return headroom
        import torch

        tensor = torch.tensor(
            msgspec.structs.astuple(headroom), dtype=torch.int64, device="cpu"
        )
        torch.distributed.all_reduce(
            tensor, op=torch.distributed.ReduceOp.MIN, group=self.cpu_group
        )
        return PrefillHeadroom(*tensor.tolist())

    def handle_control(self, req, *, paused: bool):
        from sglang.srt.managers.io_struct import PrefillPullBudgetReqOutput

        if req.action == "peek":
            # Translate into the clock used by expires_at without another
            # broadcast. Peek changes nothing; reserve and bind recheck expiry
            # on the current shared clock before changing the ledger.
            headroom = self._reduce(
                self.ledger.peek_headroom(
                    self._capacity(paused), now=time.monotonic() + self._clock_offset
                )
            )
            self._log_ledger(headroom)
            return PrefillPullBudgetReqOutput(rid=req.rid, **self.ledger.peek(headroom))
        now = self._now()
        capacity = None
        capacity_limit = None
        if req.action in ("reserve", "bind"):
            capacity = self._capacity(paused)
            capacity_limit = self._reduce(
                self.ledger.headroom(
                    capacity,
                    now=now,
                    exclude=req.reservation_id if req.action == "bind" else None,
                )
            )
        result = self.ledger.control(
            action=req.action,
            reservation_id=req.reservation_id,
            assignment_id=req.assignment_id,
            request_id=req.request_id,
            tokens=req.tokens,
            now=now,
            capacity=capacity,
            capacity_limit=capacity_limit,
        )
        return PrefillPullBudgetReqOutput(rid=req.rid, **result)

    def _log_ledger(self, headroom: PrefillHeadroom) -> None:
        # A leaked reservation shows only as a worker that never claims again;
        # this line shows whether an idle worker's ledger is back to zero.
        now = time.monotonic()
        if now - self._logged_at < _LEDGER_LOG_INTERVAL:
            return
        self._logged_at = now
        if self._logs_ledger is None:
            self._logs_ledger = (
                self.group_size <= 1 or _global_rank() == self.group_rank
            )
        if not self._logs_ledger:
            return
        logger.info(
            "Prefill pull ledger: %s, max_tokens=%d, max_requests=%d, headroom=%s",
            self.ledger.summary(),
            self.ledger.max_tokens,
            self.ledger.max_requests,
            msgspec.structs.asdict(headroom),
        )

    def authorize(self, req) -> Optional[str]:
        # Until the scheduler request exists, embeds count one token per row;
        # afterwards the charge follows its final length (see follow_request).
        intake_tokens = max(len(req.input_ids or ()), len(req.input_embeds or ()))
        if req.prefill_pull_internal:
            req.prefill_pull_reservation_id = self.ledger.admit_internal(
                req.rid, intake_tokens
            )
            return None
        # Same test as the tokenizer's (`prepare_prefill_pull_request`): an
        # empty id is no pull identity, so it must not reach consume and 403.
        if self.allow_push and not (
            req.prefill_pull_reservation_id or req.prefill_pull_assignment_id
        ):
            # A push dispatch cannot be refused without failing an already
            # paired D; charge it so pull grants shrink around it.
            req.prefill_pull_reservation_id = self.ledger.admit_internal(
                req.rid, intake_tokens, kind="push"
            )
            return None
        if (
            req.mm_inputs is not None
            or req.input_embeds is not None
            or req.session_id is not None
            or req.session_params is not None
            or req.positional_embed_overrides is not None
            or req.lora_id is not None
            or req.sampling_params.n != 1
            or req.sampling_params.beam_width not in (None, 1)
        ):
            return "Prefill pull supports single text requests without sessions, adapters or beam search"
        return self.ledger.consume(
            reservation_id=req.prefill_pull_reservation_id,
            assignment_id=req.prefill_pull_assignment_id,
            request_id=req.rid,
            tokens=len(req.input_ids or ()),
        )

    def handle_generate_request(
        self, req, *, process_request: Callable[[], None], send_control_output: Callable
    ) -> None:
        """Authorize intake and settle failures without releasing an active sender."""
        from sglang.srt.managers.io_struct import AbortReq

        if error := self.authorize(req):
            send_control_output(
                AbortReq(
                    rid=req.rid,
                    finished_reason={
                        "type": "abort",
                        "status_code": 403,
                        "message": error,
                    },
                ),
                req,
            )
            return
        failed = True
        try:
            process_request()
            failed = False
        finally:
            self.ledger.finish_intake(req.prefill_pull_reservation_id, failed)


# Seconds between ledger log lines; arbitrary, one line a minute per worker.
_LEDGER_LOG_INTERVAL = 60.0


def _global_rank() -> int:
    import torch

    return torch.distributed.get_rank()


def _request_overhead_tokens(*, allocator, logical_page_size: int) -> int:
    """`PrefillAdder.per_req_token_overhead`, from the same inputs."""
    if allocator is None:
        return logical_page_size
    from sglang.srt.mem_cache.allocator.page_interleave import (
        page_interleave_shard_size,
    )

    shard_size = page_interleave_shard_size(allocator)
    return shard_size * allocator.page_size if shard_size > 1 else logical_page_size


def _prompt_len(req) -> int:
    # `seqlen` grows with the first output token once prefill is done, while
    # the KV allocated stays the prompt's: that token is not prefill work.
    return req.seqlen - len(req.output_ids)


def pending_prefill_tokens(
    req, *, storage_hit_end: Callable[[Any], Optional[int]]
) -> int:
    """Prompt tokens `req` still has to prefill, in the unit of the engine's
    `num_waiting_uncached_tokens`: KV allocation advances with each scheduled
    chunk; before the first one, the device/host prefix match, the
    enqueue-time match and the queried L3 hit are discounted (see
    `SchedulerLoadInquirer._cached_prefix_len`)."""
    prompt = _prompt_len(req)
    if req.kv.kv_allocated_len > 0:
        return max(0, prompt - req.kv.kv_allocated_len)
    cached = max(
        req.num_matched_prefix_tokens, req.storage_prefetch_last_match_len or 0
    )
    hit_end = storage_hit_end(req)
    if hit_end is not None:
        cached = max(cached, hit_end)
    return max(0, prompt - min(cached, prompt))


def _state_slot_accounting(*, allocator, req_pool, tree_cache) -> dict:
    """Ledger arguments that price recurrent state like the scheduler's admission."""
    from types import SimpleNamespace

    state = allocator.state

    def pending_state_slots(req) -> int:
        return state.pending_request_slots(req_pool, req)

    # A request that holds no state slot or tracking buffer yet, as acquire sees it.
    fresh = SimpleNamespace(
        kv=SimpleNamespace(holds_mamba=False, mamba_ping_pong_track_buffer=None)
    )
    return dict(
        fresh_state_slots=pending_state_slots(fresh),
        # Nonzero only where state and tokens share one arena (unified pools).
        state_slot_cost=allocator.create_prefill_budget(tree_cache).state_slot_cost,
        pending_state_slots=pending_state_slots,
    )


class PrefillPullBudgetClient:
    """Correlate IPC replies so cancelled calls cannot consume a later reply."""

    def __init__(self, send: Callable):
        self.send = send
        self.pending: dict[str, asyncio.Future] = {}

    async def call(self, **kwargs) -> dict:
        from sglang.srt.managers.io_struct import PrefillPullBudgetReq

        req = PrefillPullBudgetReq(rid=uuid.uuid4().hex, **kwargs)
        future = asyncio.get_running_loop().create_future()
        self.pending[req.rid] = future
        try:
            self.send(req)
            result = await future
            return {
                "success": result.success,
                "data": result.data,
                "error": result.error,
            }
        finally:
            self.pending.pop(req.rid, None)

    def handle_recv(self, result) -> None:
        future = self.pending.get(result.rid)
        if future is not None and not future.done():
            future.set_result(result)


PULL_WARMUP_HEADER = "X-SGLang-Prefill-Pull-Warmup"
# Generated per process: the launcher's warmup runs in this process and posts to
# its own --host, which need not be loopback. Pull requires one tokenizer
# worker, so this process also receives the request.
_PULL_WARMUP_TOKEN = secrets.token_hex(16)


def pull_warmup_headers() -> dict[str, str]:
    """Headers that authorize this process's own startup warmup request."""
    return {PULL_WARMUP_HEADER: _PULL_WARMUP_TOKEN}


def prepare_prefill_pull_request(
    obj, request, *, starting: bool, allow_push: bool = False
) -> bool:
    """Return trusted warmup/health status without adding an HTTP bypass field.

    With `allow_push`, a request carrying no pull identity is a push router's
    dispatch: it passes here and the scheduler charges it to the ledger.
    """
    from sglang.srt.constants import HEALTH_CHECK_RID_PREFIX
    from sglang.srt.disaggregation.mode import FAKE_BOOTSTRAP_HOST
    from sglang.srt.managers.io_struct import GenerateReqInput

    if not isinstance(obj, GenerateReqInput):
        raise ValueError("Prefill pull only supports generation")
    path = request.url.path if request is not None else None
    health = (
        isinstance(obj.rid, str)
        and obj.rid.startswith(HEALTH_CHECK_RID_PREFIX)
        and (request is None or path in ("/health", "/health_generate"))
    )
    supplied = "" if request is None else request.headers.get(PULL_WARMUP_HEADER, "")
    # compare_digest raises TypeError on non-ASCII str; such a value is no token.
    trusted = request is None or (
        supplied.isascii() and hmac.compare_digest(supplied, _PULL_WARMUP_TOKEN)
    )
    # Paused or unhealthy workers are also unready, but must never regain the
    # startup bypass. Only the startup state authorizes the launcher warmup.
    warmup = starting and trusted and obj.bootstrap_host == FAKE_BOOTSTRAP_HOST
    internal = health or warmup
    push = allow_push and not (
        obj.prefill_pull_reservation_id or obj.prefill_pull_assignment_id
    )
    if not internal and not push:
        params = obj.sampling_params or {}
        if (
            not isinstance(params, dict)
            or params.get("n", 1) != 1
            or isinstance(obj.text, list)
            or (obj.input_ids and isinstance(obj.input_ids[0], list))
            or not obj.prefill_pull_reservation_id
            or not obj.prefill_pull_assignment_id
        ):
            raise ValueError(
                "Prefill pull requires an authorized single request from the pull agent"
            )
    return internal
