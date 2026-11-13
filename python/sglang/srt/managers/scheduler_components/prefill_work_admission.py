"""Scheduler-owned admission for request-pull prefill workers.

The ledger charges a prompt until the scheduler's safe retirement boundary,
including bootstrap and source KV writes. Network timeouts can revoke a grant
that has not been consumed; they can never release an executing request.
"""

from __future__ import annotations

import asyncio
import ipaddress
import time
import uuid
from collections import OrderedDict
from typing import Any, Callable, Optional

import msgspec


class PrefillCapacity(msgspec.Struct, frozen=True):
    tokens: int
    requests: int
    metadata: int


class _Reservation(msgspec.Struct):
    tokens: int
    expires_at: float
    assignment_id: Optional[str] = None
    request_id: Optional[str] = None
    consumed: bool = False
    request: Any = None


class PrefillWorkAdmission:
    """Serial scheduler ledger; the caller supplies group-consistent snapshots.

    Work tokens count full prompts conservatively, without predicting cache hits
    or GPU speed. KV promises are page rounded independently of the work cap.
    Actual allocated KV is already absent from the allocator snapshot, so only
    the remaining promised allocation is subtracted from it.
    """

    def __init__(
        self,
        *,
        max_tokens: int,
        max_requests: int,
        page_size: int = 1,
        safety_margin_tokens: int = 0,
        reservation_ttl: float = 120.0,
    ):
        if max_tokens <= 0 or max_requests <= 0 or page_size <= 0:
            raise ValueError("Prefill pull requires positive token/request/page limits")
        if safety_margin_tokens < 0 or reservation_ttl <= 0:
            raise ValueError("Invalid prefill pull safety margin or reservation TTL")
        self.max_tokens = max_tokens
        self.max_requests = max_requests
        self.page_size = page_size
        self.safety_margin_tokens = safety_margin_tokens
        self.reservation_ttl = reservation_ttl
        self.reservations: dict[str, _Reservation] = {}
        # Bounded replay history. Old UUIDs are never reissued by the agent.
        self._terminal: OrderedDict[str, str] = OrderedDict()

    def _remember(self, reservation_id: str, state: str) -> None:
        self._terminal[reservation_id] = state
        self._terminal.move_to_end(reservation_id)
        while len(self._terminal) > 8192:
            self._terminal.popitem(last=False)

    def _expire(self, now: float) -> None:
        for reservation_id, reservation in list(self.reservations.items()):
            if not reservation.consumed and now >= reservation.expires_at:
                del self.reservations[reservation_id]
                self._remember(reservation_id, "expired")

    def _kv_charge(self, tokens: int) -> int:
        # One output token and page rounding remain covered even at a boundary.
        return ((tokens + 1 + self.page_size - 1) // self.page_size) * self.page_size

    def _remaining_capacity(self, capacity: PrefillCapacity) -> tuple[int, int]:
        work = 0
        promised_kv = 0
        promised_rows = 0
        promised_metadata = 0
        for reservation in self.reservations.values():
            work += reservation.tokens
            req = reservation.request
            allocated = 0 if req is None else req.kv.kv_allocated_len
            # Prefix-sharing is deliberately not discounted. A pessimistic
            # promise cannot over-admit after an eviction or a cache miss.
            promised_kv += max(0, self._kv_charge(reservation.tokens) - allocated)
            promised_rows += req is None or req.kv.req_pool_idx is None
            promised_metadata += req is None or req.metadata_buffer_index == -1
        tokens = min(
            self.max_tokens - work,
            capacity.tokens - promised_kv - self.safety_margin_tokens - self.page_size,
        )
        requests = min(
            self.max_requests - len(self.reservations),
            capacity.requests - promised_rows,
            capacity.metadata - promised_metadata,
        )
        return max(0, tokens), max(0, requests)

    def remaining_capacity(
        self, capacity: PrefillCapacity, *, now: float
    ) -> tuple[int, int]:
        """Return local headroom after expiring unconsumed reservations."""
        self._expire(now)
        return self._remaining_capacity(capacity)

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
        capacity_limit: Optional[tuple[int, int]] = None,
        assignment_id: Optional[str] = None,
        tokens: Optional[int] = None,
        request_id: Optional[str] = None,
    ) -> dict:
        self._expire(now)
        if not reservation_id:
            return self._result(False, error="reservation_id is required")
        reservation = self.reservations.get(reservation_id)
        if action == "reserve":
            if reservation_id in self._terminal:
                return self._result(False, error=self._terminal[reservation_id])
            if reservation is None:
                if capacity is None:
                    return self._result(False, error="capacity unavailable")
                available_tokens, available_requests = self._remaining_capacity(
                    capacity
                )
                if capacity_limit is not None:
                    available_tokens = min(available_tokens, capacity_limit[0])
                    available_requests = min(available_requests, capacity_limit[1])
                if available_tokens <= 0 or available_requests <= 0:
                    return self._result(False, error="no_capacity")
                reservation = _Reservation(
                    tokens=available_tokens,
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
        now: float,
    ) -> Optional[str]:
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

    def admit_internal(self, request_id: str, tokens: int) -> str:
        """Account trusted health/warmup work while keeping its startup bypass."""
        reservation_id = f"internal:{request_id}"
        self.reservations[reservation_id] = _Reservation(
            tokens=tokens,
            expires_at=float("inf"),
            request_id=request_id,
            consumed=True,
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
    ):
        stats = pool_stats.get_pool_stats()
        usage = pool_stats.storage.usage()
        if (
            stats.is_hybrid_swa
            or stats.is_hybrid_ssm
            or stats.is_hisparse
            or len(usage) != 1
            or usage[0].metrics != "full"
        ):
            raise ValueError(
                "Prefill pull currently supports a single dense/MLA KV pool; SWA, Mamba and HiSparse require separate pool admission"
            )
        self.ledger = PrefillWorkAdmission(
            max_tokens=max_tokens,
            max_requests=max_requests,
            page_size=page_size,
            safety_margin_tokens=safety_margin_tokens,
        )
        self.pool_stats = pool_stats
        self.req_pool = req_pool
        self.metadata_allocator = metadata_allocator
        self.cpu_group = cpu_group
        self.group_rank = group_rank
        self.group_size = group_size

    def _now(self) -> float:
        now = time.monotonic()
        if self.group_size > 1:
            import torch

            values = [now]
            torch.distributed.broadcast_object_list(
                values, src=self.group_rank, group=self.cpu_group
            )
            now = values[0]
        return now

    def _capacity(self, paused: bool) -> PrefillCapacity:
        stats = self.pool_stats.get_pool_stats()
        values = [
            stats.full_available_size + stats.full_evictable_size,
            self.req_pool.available_size(),
            self.metadata_allocator.available_size(),
        ]
        if paused:
            values = [0, 0, 0]
        return PrefillCapacity(*values)

    def handle_control(self, req, *, paused: bool):
        from sglang.srt.managers.io_struct import PrefillPullBudgetReqOutput

        now = self._now()
        capacity = None
        capacity_limit = None
        if req.action == "reserve":
            capacity = self._capacity(paused)
            capacity_limit = self.ledger.remaining_capacity(capacity, now=now)
            if self.group_size > 1:
                import torch

                tensor = torch.tensor(capacity_limit, dtype=torch.int64, device="cpu")
                torch.distributed.all_reduce(
                    tensor, op=torch.distributed.ReduceOp.MIN, group=self.cpu_group
                )
                capacity_limit = tuple(tensor.tolist())
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

    def authorize(self, req) -> Optional[str]:
        if req.prefill_pull_internal:
            req.prefill_pull_reservation_id = self.ledger.admit_internal(
                req.rid, len(req.input_ids or ())
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
            now=self._now(),
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


def prepare_prefill_pull_request(obj, request, *, starting: bool) -> bool:
    """Return trusted warmup/health status without adding an HTTP bypass field."""
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
    local = request is None
    if request is not None and request.client is not None:
        try:
            local = ipaddress.ip_address(request.client.host).is_loopback
        except ValueError:
            pass
    # Paused or unhealthy workers are also unready, but must never regain the
    # startup bypass. Only the startup state authorizes the launcher warmup.
    warmup = starting and local and obj.bootstrap_host == FAKE_BOOTSTRAP_HOST
    internal = health or warmup
    if not internal:
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
