"""HTTP bridge for scheduler-authorized prefill request pulling.

The scheduler owns capacity and admission. This asynchronous agent owns only
network I/O: reserve capacity, acquire an offer, bind it, obtain a commit, and
execute the ordinary local /generate endpoint. An offer is never executable.

The agent claims from every configured router at once. Routers keep separate
queues and never coordinate: the scheduler's ledger re-checks capacity at each
bind, so concurrent claims from several routers cannot overcommit this worker.
"""

from __future__ import annotations

import asyncio
import contextlib
import ipaddress
import json
import logging
import socket
import ssl
import uuid
from collections.abc import Awaitable, Callable
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import aiohttp
import msgspec

logger = logging.getLogger(__name__)


class PrefillPullAgentConfig(msgspec.Struct, frozen=True):
    # An http:// hostname is re-resolved and every address is a router, which
    # tracks a Kubernetes headless Service; https:// and IPs are used as given.
    router_urls: tuple[str, ...]
    worker_url: str
    local_url: str
    max_inflight: int
    # Claims in flight at once per router: overlaps round trips with the next.
    claim_concurrency: int = 1
    api_key: str | None = None
    router_api_key: str | None = None
    local_ssl: ssl.SSLContext | bool = True
    poll_wait_ms: int = 10000
    retry_interval: float = 0.5
    status_interval: float = 1.0
    control_timeout: float = 15.0
    router_refresh_interval: float = 10.0
    # Advertise a live pull agent even when its scheduler has no capacity.
    # Routers expire this presence after 45 s without an acquire/heartbeat.
    heartbeat_interval: float = 10.0
    # Refreshes a router must be missing from DNS before its claims stop, so a
    # transient lookup gap does not drop outcome reporting.
    router_missing_refreshes: int = 3
    # Seconds a router may stay unreachable before its pending outcomes are
    # dropped, whether or not it is still listed. While it answers, outcomes
    # keep going however long it drains: its timeout ends the client's wait.
    router_unreachable_grace: float = 60.0
    # A router that answers a claim with 4xx (bad key, version skew) will not
    # accept a retry soon; arbitrary, long enough to keep its logs quiet.
    rejected_retry_interval: float = 5.0
    # How often a worker with free slots but no budget asks again. Compute
    # budget frees mid-request (last chunk allocated), which no local event
    # signals; arbitrary, peeks are shared so this is one IPC per interval.
    no_capacity_retry_interval: float = 0.05
    # How often an open long poll checks whether room grew past what it asked
    # for; an assignment ending here checks at once. Arbitrary.
    poll_refresh_interval: float = 0.5
    # Repeated claim failures of one kind log a warning at most this often.
    warning_interval: float = 30.0
    max_response_bytes: int = 4 * 1024 * 1024
    max_offer_bytes: int = 16 * 1024 * 1024


class PullProtocolError(RuntimeError):
    pass


class PullHTTPError(PullProtocolError):
    def __init__(self, action: str, status: int):
        super().__init__(f"Pull {action} returned HTTP {status}")
        self.status = status


def _expandable(url: str) -> bool:
    parts = urlsplit(url)
    if parts.scheme != "http" or not parts.hostname:
        return False
    try:
        ipaddress.ip_address(parts.hostname)
    except ValueError:
        return True
    return False


def _with_host(url: str, address: str) -> str:
    parts = urlsplit(url)
    host = f"[{address}]" if ":" in address else address
    netloc = f"{host}:{parts.port or 80}"
    return urlunsplit((parts.scheme, netloc, parts.path, "", ""))


class PrefillPullAgent:
    def __init__(
        self,
        config: PrefillPullAgentConfig,
        *,
        budget: Callable[..., Awaitable[dict[str, Any]]],
        abort: Callable[[str], Any],
        ready: Callable[[], bool],
    ):
        self.config = config
        self._budget = budget
        self._abort = abort
        self._ready = ready
        self.worker_epoch = uuid.uuid4().hex
        self._session: aiohttp.ClientSession | None = None
        # Router base URL -> its claim loops, listed or draining.
        self._routers: dict[str, list[asyncio.Task]] = {}
        self._heartbeats: dict[str, asyncio.Task] = {}
        # Routers currently in DNS (or configured literally). A router that
        # left keeps its loops while it answers: it may still hold queued
        # requests that only a claim can drain.
        self._listed: set[str] = set()
        # Configured URL -> last resolved router base URLs.
        self._resolved: dict[str, set[str]] = {}
        self._missing: dict[str, int] = {}
        self._refresher: asyncio.Task | None = None
        self._assignments: dict[str, asyncio.Task] = {}
        self._assignment_reservations: dict[str, str] = {}
        self._assignment_routers: dict[str, str] = {}
        # Set (and replaced) when an assignment ends, waking every claim loop
        # that waits for room instead of leaving it asleep for retry_interval.
        self._capacity_freed = asyncio.Event()
        # The latest peek every claim loop shares, in flight or answered at
        # `_peek_answered_at`: one scheduler round trip (a TP collective) per
        # no_capacity_retry_interval however many loops ask.
        self._peek_task: asyncio.Future | None = None
        self._peek_answered_at = 0.0
        self._warned_at: dict[tuple[str, str], float] = {}
        self._draining = False
        self._stopping = False

    async def start(self):
        if self._session is not None:
            raise RuntimeError("Prefill pull agent is already running")
        if self.config.max_inflight < 1:
            raise ValueError("Prefill pull max_inflight must be positive")
        if not 1 <= self.config.claim_concurrency <= 16:
            raise ValueError("Prefill pull claim concurrency must be within 1..16")
        if not self.config.router_urls:
            raise ValueError("Prefill pull needs at least one router URL")
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=None, sock_connect=10),
            connector=aiohttp.TCPConnector(limit=0),
        )
        await self._refresh_routers()
        if any(_expandable(url) for url in self.config.router_urls):
            self._refresher = asyncio.create_task(self._refresh_loop())

    def _can_claim(self) -> bool:
        return not (self._draining or self._stopping) and self._ready()

    async def _stop_claims(self, *, finish_polls: bool):
        self._draining = True
        if self._refresher is not None:
            self._refresher.cancel()
        for task in self._heartbeats.values():
            task.cancel()
        # Wake loops waiting for room, so they see the flag and leave.
        self._capacity_freed.set()
        self._capacity_freed = asyncio.Event()
        loops = [task for tasks in self._routers.values() for task in tasks]
        if finish_polls and loops:
            # A poll the router already answered may carry an offer. Cancelled,
            # its offer sat unanswered until --pull-offer-timeout-ms; finished,
            # _admit_offer rejects it and the router requeues it at once.
            await asyncio.wait(
                loops,
                timeout=self.config.poll_wait_ms / 1000 + self.config.control_timeout,
            )
        tasks = [*loops, *self._heartbeats.values(), self._refresher, self._peek_task]
        tasks = [task for task in tasks if task is not None]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    async def drain(self):
        """Stop new claims and finish accepted work before scheduler shutdown.

        An accept can be in flight before local /generate registers its rid.
        Keep the scheduler and HTTP session alive through execution and outcome
        reporting, even when the tokenizer has no requests yet.
        """
        await self._stop_claims(finish_polls=True)
        await asyncio.shield(
            asyncio.gather(*tuple(self._assignments.values()), return_exceptions=True)
        )

    async def close(self):
        self._stopping = True
        assignments = [
            (
                assignment_id,
                self._assignment_routers[assignment_id],
                self._assignment_reservations[assignment_id],
                task,
            )
            for assignment_id, task in self._assignments.items()
        ]
        await self._stop_claims(finish_polls=False)
        tasks = [task for *_, task in assignments]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        # A task cancelled before its coroutine first runs has no finally block
        # to revoke the grant. Keep shutdown cleanup with the task's owner too.
        await asyncio.gather(
            *(
                self._close_assignment(router, assignment_id, reservation_id)
                for assignment_id, router, reservation_id, _ in assignments
            )
        )
        if self._session is not None:
            await self._session.close()
            self._session = None

    async def _resolve(self, url: str) -> set[str]:
        if not _expandable(url):
            return {url.rstrip("/")}
        parts = urlsplit(url)
        infos = await asyncio.get_running_loop().getaddrinfo(
            parts.hostname, parts.port or 80, type=socket.SOCK_STREAM
        )
        # One family only: a dual-stack name (localhost, or a pod with both A
        # and AAAA records) would otherwise count each router twice and double
        # its claim loops. IPv4 wins when present, as each pod has one.
        v4 = {info[4][0] for info in infos if info[0] == socket.AF_INET}
        addresses = v4 or {info[4][0] for info in infos}
        return {_with_host(url, address).rstrip("/") for address in addresses}

    async def _refresh_routers(self):
        for url in self.config.router_urls:
            try:
                self._resolved[url] = await self._resolve(url)
            except OSError as exc:
                # Keep the last answer: a failed lookup is not a removal.
                logger.warning("Cannot resolve prefill pull router %s: %s", url, exc)
        if self._draining or self._stopping:
            return
        current = set().union(*self._resolved.values())
        for router in current:
            self._missing.pop(router, None)
            self._listed.add(router)
            if router not in self._routers:
                logger.info("Prefill pull claiming from router %s", router)
                self._routers[router] = [
                    asyncio.create_task(self._acquire_loop(router))
                    for _ in range(self.config.claim_concurrency)
                ]
                self._heartbeats[router] = asyncio.create_task(
                    self._heartbeat_loop(router)
                )
        for router in list(self._listed - current):
            self._missing[router] = self._missing.get(router, 0) + 1
            if self._missing[router] >= self.config.router_missing_refreshes:
                await self._drop_router(router)

    async def _drop_router(self, router: str):
        """Mark `router` as gone from DNS. Its claim loops keep going while it
        answers: a router leaving a rolling update drains its queue, and
        requests nobody claims time out at their client. They stop once it
        stays unreachable (see `_unreachable_for`)."""
        logger.info(
            "Prefill pull router %s left DNS; claiming until it stops answering",
            router,
        )
        self._listed.discard(router)
        self._missing.pop(router, None)

    def _retire_router(self, router: str):
        loops = self._routers.pop(router, [])
        heartbeat = self._heartbeats.pop(router, None)
        if heartbeat is not None:
            loops.append(heartbeat)
        current = asyncio.current_task()
        for task in loops:
            if task is not current:
                task.cancel()
        logger.info("Prefill pull router %s is unreachable; stopped claiming", router)

    async def _heartbeat_loop(self, router: str):
        """A zero-budget acquire advertises presence without claiming work.

        Capacity and presence are separate: full workers must remain eligible
        for queued requests, while push-only or disconnected workers must not
        attract cache affinity. This loop never contacts the scheduler.
        """
        unreachable_since = None
        clock = asyncio.get_running_loop()
        while not (self._draining or self._stopping):
            if not self._can_claim():
                await asyncio.sleep(self.config.retry_interval)
                continue
            delay = self.config.heartbeat_interval
            try:
                reply = await self._control(
                    router,
                    "acquire",
                    {
                        "worker_url": self.config.worker_url,
                        "worker_epoch": self.worker_epoch,
                        "dp_rank": 0,
                        "poll_id": uuid.uuid4().hex,
                        "max_tokens": 0,
                        "max_requests": 1,
                        "max_offered": self.config.claim_concurrency,
                        "wait_ms": 0,
                    },
                )
                if reply.get("status") != "empty":
                    raise PullProtocolError("Pull heartbeat must not offer work")
                unreachable_since = None
            except (aiohttp.ClientError, TimeoutError) as exc:
                if unreachable_since is None:
                    unreachable_since = clock.time()
                if (
                    router not in self._listed
                    and clock.time() - unreachable_since
                    >= self.config.router_unreachable_grace
                ):
                    self._retire_router(router)
                    return
                self._warn(router, "heartbeat", "heartbeat failed: %r", exc)
                delay = self.config.retry_interval
            except PullProtocolError as exc:
                unreachable_since = None
                self._warn(router, "heartbeat", "heartbeat refused: %s", exc)
                delay = (
                    self.config.rejected_retry_interval
                    if isinstance(exc, PullHTTPError) and 400 <= exc.status < 500
                    else self.config.retry_interval
                )
            await asyncio.sleep(delay)

    async def _refresh_loop(self):
        while not (self._draining or self._stopping):
            await asyncio.sleep(self.config.router_refresh_interval)
            try:
                await self._refresh_routers()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Prefill pull router refresh failed")

    async def _close_assignment(
        self, router: str, assignment_id: str, reservation_id: str
    ):
        await self._release(reservation_id)
        # Reject also cancels committed assignments; the router must not keep
        # waiting for an outcome from an agent that has shut down. A committed
        # assignment remains non-replayable even if this notification is lost.
        with contextlib.suppress(aiohttp.ClientError, TimeoutError, PullProtocolError):
            await self._control(router, "reject", self._identity(assignment_id))

    def _abort_local(self, request_id: str):
        try:
            self._abort(request_id)
        except Exception:
            logger.exception("Failed to abort prefill pull request %s", request_id)

    def _identity(self, assignment_id: str) -> dict[str, Any]:
        return {
            "assignment_id": assignment_id,
            "worker_url": self.config.worker_url,
            "worker_epoch": self.worker_epoch,
            "dp_rank": 0,
        }

    async def _control(
        self, router: str, action: str, body: dict[str, Any], *, hold_ms: int = 0
    ) -> dict[str, Any]:
        headers = {}
        if self.config.router_api_key:
            headers["Authorization"] = f"Bearer {self.config.router_api_key}"
        async with self._session.post(
            router + "/internal/pull/" + action,
            json=body,
            headers=headers,
            # A long poll legitimately holds the response for hold_ms.
            timeout=aiohttp.ClientTimeout(
                total=hold_ms / 1000 + self.config.control_timeout
            ),
        ) as response:
            payload = await self._read_limited(response, self.config.max_offer_bytes)
            if response.status != 200:
                raise PullHTTPError(action, response.status)
            try:
                result = json.loads(payload)
            except ValueError as exc:
                raise PullProtocolError(f"Pull {action} returned invalid JSON") from exc
            if not isinstance(result, dict):
                raise PullProtocolError(f"Pull {action} returned a non-object")
            return result

    @staticmethod
    async def _read_limited(response, limit: int) -> bytes:
        chunks = bytearray()
        async for chunk in response.content.iter_chunked(65536):
            chunks.extend(chunk)
            if len(chunks) > limit:
                raise PullProtocolError("Prefill pull HTTP response exceeds size limit")
        return bytes(chunks)

    async def _release(self, reservation_id: str):
        # Release revokes unconsumed grants. A consumed grant is retired only
        # by scheduler lifecycle observation, never by an HTTP abort response.
        try:
            await self._budget_call(action="release", reservation_id=reservation_id)
        except Exception:
            logger.exception("Failed to revoke prefill pull reservation")

    async def _budget_call(self, **kwargs):
        return await asyncio.wait_for(
            self._budget(**kwargs), timeout=self.config.control_timeout
        )

    async def _acquire_loop(self, router: str):
        unreachable_since = None
        while not (self._draining or self._stopping):
            # Reserved only once an offer arrives (see _run_assignment): a
            # router with no work must not hold a request slot through its
            # long poll while another router's queue waits for it.
            reservation_id = uuid.uuid4().hex
            try:
                poll = await self._next_poll()
                if not isinstance(poll, dict):
                    await self._wait_for_capacity(poll)
                    continue
                started = asyncio.get_running_loop().time()
                try:
                    offer = await self._acquire_offer(router, poll)
                except (aiohttp.ClientError, TimeoutError) as exc:
                    # Only the router's own failure counts toward giving up on
                    # it; a stalled local scheduler (peek above) or a router
                    # that answers with an error must not retire it.
                    clock = asyncio.get_running_loop().time()
                    if unreachable_since is None:
                        unreachable_since = clock
                    if (
                        router not in self._listed
                        and clock - unreachable_since
                        >= self.config.router_unreachable_grace
                    ):
                        self._retire_router(router)
                        return
                    self._warn(router, "unreachable", "is unreachable: %r", exc)
                    await asyncio.sleep(self.config.retry_interval)
                    continue
                except PullProtocolError as exc:
                    unreachable_since = None
                    self._warn(router, "refused", "refused a claim: %s", exc)
                    rejected = isinstance(exc, PullHTTPError) and exc.status < 500
                    await asyncio.sleep(
                        self.config.rejected_retry_interval
                        if rejected
                        else self.config.retry_interval
                    )
                    continue
                unreachable_since = None
                if offer.get("status") == "empty":
                    await self._after_empty(offer, started=started)
                    continue
                await self._admit_offer(router, reservation_id, offer)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                expected = isinstance(
                    exc, (aiohttp.ClientError, TimeoutError, PullProtocolError)
                )
                self._warn(
                    router,
                    type(exc).__name__,
                    "claim failed: %r",
                    exc,
                    exc_info=not expected,
                )
                await asyncio.sleep(self.config.retry_interval)

    async def _after_empty(self, offer: dict[str, Any], *, started: float):
        if offer.get("held"):
            # The router holds this worker for a starved request it cannot
            # fit yet: only fresher room helps, so peek again soon.
            await self._wait_for_capacity(self.config.no_capacity_retry_interval)
            return
        waited = asyncio.get_running_loop().time() - started
        # A poll that waited out wait_ms already paced itself; sleeping after
        # it left requests arriving meanwhile unclaimed. One answered early
        # (an open breaker, offers in flight) backs off instead of spinning;
        # with poll_wait_ms=0 every empty answer is an early one.
        if (
            self.config.poll_wait_ms == 0
            or waited * 1000 < self.config.poll_wait_ms / 2
        ):
            await asyncio.sleep(self.config.retry_interval)

    async def _next_poll(self) -> dict[str, Any] | float:
        """The next acquire request, or how long to wait for room."""
        if not self._can_claim() or len(self._assignments) >= self.config.max_inflight:
            return self.config.retry_interval
        grant = await self._peek()
        if not self._can_claim() or not grant.get("success"):
            return self.config.no_capacity_retry_interval
        return {
            "worker_url": self.config.worker_url,
            "worker_epoch": self.worker_epoch,
            "dp_rank": 0,
            "poll_id": uuid.uuid4().hex,
            "max_tokens": grant["data"]["max_tokens"],
            "max_requests": 1,
            # Lets the router hold one uncommitted offer per claim loop.
            "max_offered": self.config.claim_concurrency,
            "wait_ms": self.config.poll_wait_ms,
        }

    async def _peek(self) -> dict[str, Any]:
        # Peek holds nothing, so claim loops can share one answer.
        loop = asyncio.get_running_loop()
        task = self._peek_task
        fresh = task is not None and (
            not task.done()
            or (
                not task.cancelled()
                and task.exception() is None
                and loop.time() - self._peek_answered_at
                < self.config.no_capacity_retry_interval
            )
        )
        if not fresh:
            task = asyncio.ensure_future(
                self._budget_call(action="peek", reservation_id=uuid.uuid4().hex)
            )
            task.add_done_callback(self._peek_answered)
            self._peek_task = task
        # Shielded: one loop's cancellation must not fail the others' peek.
        return await asyncio.shield(task)

    def _peek_answered(self, task: asyncio.Future):
        self._peek_answered_at = asyncio.get_running_loop().time()
        # Retrieved here too, in case every loop awaiting it was cancelled.
        if not task.cancelled():
            task.exception()

    async def _wait_for_capacity(self, timeout: float):
        # The timeout covers room freed by anything other than an assignment
        # ending here, such as push requests or the scheduler becoming ready.
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(self._capacity_freed.wait(), timeout)

    def _warn(self, router: str, kind: str, message: str, *args, exc_info=False):
        clock = asyncio.get_running_loop().time()
        key = (router, kind)
        last = self._warned_at.get(key)
        level = logging.DEBUG
        if last is None or clock - last >= self.config.warning_interval:
            self._warned_at[key] = clock
            level = logging.WARNING
        logger.log(
            level,
            "Prefill pull router %s " + message,
            router,
            *args,
            exc_info=exc_info,
        )

    async def _admit_offer(
        self, router: str, reservation_id: str, offer: dict[str, Any]
    ):
        assignment_id = offer.get("assignment_id")
        if offer.get("status") != "offer" or not isinstance(assignment_id, str):
            raise PullProtocolError("Invalid prefill pull offer")
        if assignment_id in self._assignments:
            raise PullProtocolError("Duplicate live prefill pull assignment")
        # Loops for every router passed the pre-poll check while their
        # polls were open; admit only as many offers as there is room.
        # No await between this check and the insertion below.
        if not self._can_claim() or len(self._assignments) >= self.config.max_inflight:
            with contextlib.suppress(
                aiohttp.ClientError, TimeoutError, PullProtocolError
            ):
                await self._control(router, "reject", self._identity(assignment_id))
            return
        task = asyncio.create_task(self._run_assignment(router, reservation_id, offer))
        self._assignments[assignment_id] = task
        self._assignment_reservations[assignment_id] = reservation_id
        self._assignment_routers[assignment_id] = router
        task.add_done_callback(
            lambda task, key=assignment_id: self._assignment_done(key, task)
        )

    async def _acquire_offer(self, router: str, poll: dict[str, Any]) -> dict[str, Any]:
        # A long poll asks for at most the room seen when it began. When room
        # grows, ask again with the same poll ID: an offer the router already
        # made for the old request comes back rather than a second one.
        while True:
            request = asyncio.ensure_future(self._acquire_once(router, poll))
            # Retrieved here too, for when this loop is cancelled meanwhile.
            request.add_done_callback(lambda t: t.cancelled() or t.exception())
            grown = asyncio.ensure_future(self._room_grows(poll["max_tokens"]))
            try:
                await asyncio.wait(
                    {request, grown}, return_when=asyncio.FIRST_COMPLETED
                )
            finally:
                grown.cancel()
                if not request.done():
                    request.cancel()
                await asyncio.wait({request, grown})
            if not request.cancelled():
                return request.result()
            poll = {**poll, "max_tokens": grown.result()}

    async def _room_grows(self, max_tokens: int) -> int:
        """The larger grant once one is available."""
        while True:
            await self._wait_for_capacity(self.config.poll_refresh_interval)
            if not self._can_claim():
                # Draining: let the open poll finish rather than reissue it.
                continue
            try:
                grant = await self._peek()
            except asyncio.CancelledError:
                raise
            except Exception:
                continue
            if grant.get("success") and grant["data"]["max_tokens"] > max_tokens:
                return grant["data"]["max_tokens"]

    async def _acquire_once(self, router: str, poll: dict[str, Any]) -> dict[str, Any]:
        # Retry the same poll, so a lost offer cannot create another
        # assignment for this claim. Abandoning an offer is safe: without
        # accept, the router cannot commit it.
        for attempt in range(3):
            try:
                return await self._control(
                    router, "acquire", poll, hold_ms=poll["wait_ms"]
                )
            except (aiohttp.ClientError, TimeoutError):
                if attempt == 2:
                    raise
        raise AssertionError("unreachable")

    def _assignment_done(self, assignment_id: str, task: asyncio.Task):
        self._assignments.pop(assignment_id, None)
        self._assignment_reservations.pop(assignment_id, None)
        self._assignment_routers.pop(assignment_id, None)
        self._capacity_freed.set()
        self._capacity_freed = asyncio.Event()
        # The woken loops need a fresh answer, not the one from before.
        self._peek_answered_at = float("-inf")
        if not task.cancelled() and task.exception() is not None:
            logger.error("Prefill pull assignment failed: %s", task.exception())

    async def _commit(self, router: str, identity: dict[str, Any]) -> bool:
        try:
            result = await self._control(router, "accept", identity)
        except (aiohttp.ClientError, TimeoutError, PullProtocolError):
            # A timeout does not tell us whether the router committed. Query
            # this assignment; do not execute speculatively or acquire a retry.
            result = await self._control(router, "status", identity)
            if result.get("status") == "offered":
                if not self._can_claim():
                    await self._control(router, "reject", identity)
                    return False
                result = await self._control(router, "accept", identity)
        return result.get("status") == "committed"

    async def _run_assignment(
        self, router: str, reservation_id: str, offer: dict[str, Any]
    ):
        assignment_id = offer["assignment_id"]
        identity = self._identity(assignment_id)
        request_id = offer.get("request_id")
        execution = None
        monitor = None
        committed = False
        # Each release is a scheduler IPC and, with TP > 1, a group collective:
        # send it once, before any local abort, and never for a failed reserve.
        released = False

        async def revoke():
            nonlocal released
            if not released:
                released = True
                await self._release(reservation_id)

        try:
            body = self._execution_body(offer, reservation_id)
            if not self._can_claim():
                released = True
                await self._control(router, "reject", identity)
                return
            # The peek before polling held nothing; another claim may have
            # taken the slot since, in which case the offer goes back.
            granted = await self._budget_call(
                action="reserve", reservation_id=reservation_id
            )
            if not granted.get("success"):
                released = True
                await self._control(router, "reject", identity)
                return
            if not self._can_claim():
                await self._control(router, "reject", identity)
                return
            bound = await self._budget_call(
                action="bind",
                reservation_id=reservation_id,
                assignment_id=assignment_id,
                tokens=offer["input_tokens"],
                request_id=request_id,
            )
            if not bound.get("success") or not self._can_claim():
                await self._control(router, "reject", identity)
                return
            # Once accept is sent, it may commit even if readiness changes
            # while waiting for its ACK. Drain waits for this assignment.
            committed = await self._commit(router, identity)
            if not committed:
                return
            execution = asyncio.create_task(
                self._execute(body, offer.get("headers", {}))
            )
            monitor = asyncio.create_task(self._monitor(router, identity))
            done, _ = await asyncio.wait(
                [execution, monitor], return_when=asyncio.FIRST_COMPLETED
            )
            # A finished execution wins over a monitor that ended in the same
            # round: aborting it would report a successful prefill as 499.
            if execution not in done:
                # Revoke first. A /generate POST still in tokenization must
                # not enter the scheduler after cancellation was observed.
                await revoke()
                self._abort_local(request_id)
                execution.cancel()
                await asyncio.gather(execution, return_exceptions=True)
                status_code, response_body = (
                    499,
                    '{"error":"pull assignment cancelled"}',
                )
            else:
                status_code, response_body = execution.result()
            monitor.cancel()
            await asyncio.gather(monitor, return_exceptions=True)
            # Before reporting, which may retry for long: an intake that failed
            # before consuming the grant should free it now.
            await revoke()
            await self._report_outcome(router, identity, status_code, response_body)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "Prefill pull assignment %s failed: %s",
                assignment_id,
                exc,
                exc_info=True,
            )
            await revoke()
            if committed:
                self._abort_local(request_id)
            # Accept may have committed before a transport failure. Reporting
            # failure resolves that attempt; it never authorizes replay.
            await self._report_outcome(
                router,
                identity,
                502,
                json.dumps({"error": "Prefill pull execution failed"}),
            )
        finally:
            await revoke()
            if execution is not None and not execution.done():
                self._abort_local(request_id)
            for task in (execution, monitor):
                if task is not None:
                    task.cancel()
            await asyncio.gather(
                *(task for task in (execution, monitor) if task is not None),
                return_exceptions=True,
            )

    @staticmethod
    def _execution_body(offer: dict[str, Any], reservation_id: str) -> dict[str, Any]:
        if offer.get("path") != "/generate" or not isinstance(offer.get("body"), dict):
            raise PullProtocolError("Pull only supports the native /generate endpoint")
        request_id = offer.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            raise PullProtocolError("Pull offer is missing its engine request id")
        tokens = offer.get("input_tokens")
        if not isinstance(tokens, int) or isinstance(tokens, bool) or tokens < 1:
            raise PullProtocolError(
                "Pull offer must carry a positive input token count"
            )
        body = dict(offer["body"])
        body.update(
            rid=request_id,
            stream=False,
            routed_dp_rank=0,
            prefill_pull_reservation_id=reservation_id,
            prefill_pull_assignment_id=offer["assignment_id"],
        )
        return body

    async def _execute(self, body: dict[str, Any], offered_headers: dict[str, str]):
        # Do not forward client Authorization, Host, or transfer headers to a
        # local management endpoint. Trace context is the only forwarded data.
        headers = {
            key: value
            for key, value in offered_headers.items()
            if key.lower() in ("traceparent", "tracestate", "baggage")
        }
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        async with self._session.post(
            self.config.local_url.rstrip("/") + "/generate",
            json=body,
            headers=headers,
            ssl=self.config.local_ssl,
        ) as response:
            if 200 <= response.status < 300:
                # The router judges a successful prefill by status alone and
                # the client's output comes from D; logprobs or hidden states
                # can make this body arbitrarily large, so it is not relayed.
                async for _ in response.content.iter_chunked(65536):
                    pass
                return response.status, "{}"
            payload = await self._read_limited(response, self.config.max_response_bytes)
            body = payload.decode("utf-8", errors="replace")
            # Replacement characters can expand malformed upstream bytes. The
            # router's outcome limit applies to the resulting UTF-8 string.
            if len(body.encode("utf-8")) > self.config.max_response_bytes:
                raise PullProtocolError("Decoded prefill response exceeds size limit")
            return response.status, body

    async def _monitor(self, router: str, identity: dict[str, Any]):
        failures = 0
        while True:
            await asyncio.sleep(self.config.status_interval)
            try:
                status = (await self._control(router, "status", identity)).get("status")
                failures = 0
                if status != "committed":
                    return
            except (aiohttp.ClientError, TimeoutError, PullProtocolError):
                failures += 1
                if failures >= 3:
                    # Fail closed during sustained loss of the control plane;
                    # cancellation does not make this attempt safe to replay.
                    return

    async def _report_outcome(self, router: str, identity, status_code: int, body: str):
        # A router that still answers, even one removed from DNS and draining
        # its clients, is owed the outcome. One that stays unreachable is given
        # up on: retrying it forever would pin this in-flight slot.
        clock = asyncio.get_running_loop()
        unreachable_since = None
        terminal = ("completed", "cancelled", "expired")
        while not self._stopping:
            try:
                result = await self._control(
                    router,
                    "outcome",
                    {**identity, "status_code": status_code, "body": body},
                )
                if result.get("status") in terminal:
                    return
                raise PullProtocolError("Router did not acknowledge prefill outcome")
            except (aiohttp.ClientError, TimeoutError, PullProtocolError):
                pass
            await asyncio.sleep(self.config.retry_interval)
            try:
                result = await self._control(router, "status", identity)
                unreachable_since = None
                if result.get("status") in terminal:
                    return
            except (aiohttp.ClientError, TimeoutError, PullProtocolError):
                if unreachable_since is None:
                    unreachable_since = clock.time()
                # Any router, listed or not: a fixed IP or https URL is never
                # removed, so waiting for removal let one dead router pin the
                # in-flight slots every healthy router needs.
                if (
                    clock.time() - unreachable_since
                    >= self.config.router_unreachable_grace
                ):
                    logger.warning(
                        "Dropping prefill pull outcome for unreachable router %s",
                        router,
                    )
                    return
