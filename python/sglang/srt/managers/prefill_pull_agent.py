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
    # Refreshes a router must be missing from DNS before its claims stop, so a
    # transient lookup gap does not drop outcome reporting.
    router_missing_refreshes: int = 3
    # Seconds a router may stay unreachable before its pending outcomes are
    # dropped, whether or not it is still listed. While it answers, outcomes
    # keep going however long it drains: its timeout ends the client's wait.
    router_unreachable_grace: float = 60.0
    max_response_bytes: int = 4 * 1024 * 1024
    max_offer_bytes: int = 16 * 1024 * 1024


class PullProtocolError(RuntimeError):
    pass


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
        # Active router base URL -> its claim loops.
        self._routers: dict[str, list[asyncio.Task]] = {}
        # Configured URL -> last resolved router base URLs.
        self._resolved: dict[str, set[str]] = {}
        self._missing: dict[str, int] = {}
        self._refresher: asyncio.Task | None = None
        self._assignments: dict[str, asyncio.Task] = {}
        self._assignment_reservations: dict[str, str] = {}
        self._assignment_routers: dict[str, str] = {}
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
        loops = [task for tasks in self._routers.values() for task in tasks]
        if self._refresher is not None:
            loops.append(self._refresher)
        for task in loops:
            task.cancel()
        await asyncio.gather(*loops, return_exceptions=True)
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
        return {_with_host(url, info[4][0]).rstrip("/") for info in infos}

    async def _refresh_routers(self):
        for url in self.config.router_urls:
            try:
                self._resolved[url] = await self._resolve(url)
            except OSError as exc:
                # Keep the last answer: a failed lookup is not a removal.
                logger.warning("Cannot resolve prefill pull router %s: %s", url, exc)
        current = set().union(*self._resolved.values())
        for router in current - self._routers.keys():
            logger.info("Prefill pull claiming from router %s", router)
            self._routers[router] = [
                asyncio.create_task(self._acquire_loop(router))
                for _ in range(self.config.claim_concurrency)
            ]
        for router in list(self._routers.keys() - current):
            self._missing[router] = self._missing.get(router, 0) + 1
            if self._missing[router] >= self.config.router_missing_refreshes:
                await self._drop_router(router)
        for router in current:
            self._missing.pop(router, None)

    async def _drop_router(self, router: str):
        """Stop claiming from `router`; outcomes it is owed still go out while
        it answers (see `_report_outcome`)."""
        logger.info("Prefill pull router %s is gone; stopping its claims", router)
        loops = self._routers.pop(router)
        self._missing.pop(router, None)
        for task in loops:
            task.cancel()
        await asyncio.gather(*loops, return_exceptions=True)

    async def _refresh_loop(self):
        while not self._stopping:
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
        self, router: str, action: str, body: dict[str, Any]
    ) -> dict[str, Any]:
        headers = {}
        if self.config.router_api_key:
            headers["Authorization"] = f"Bearer {self.config.router_api_key}"
        async with self._session.post(
            router + "/internal/pull/" + action,
            json=body,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=self.config.control_timeout),
        ) as response:
            payload = await self._read_limited(response, self.config.max_offer_bytes)
            if response.status != 200:
                raise PullProtocolError(
                    f"Pull {action} returned HTTP {response.status}"
                )
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
        while not self._stopping:
            # Reserved only once an offer arrives (see _run_assignment): a
            # router with no work must not hold a request slot through its
            # long poll while another router's queue waits for it.
            reservation_id = uuid.uuid4().hex
            try:
                if (
                    not self._ready()
                    or len(self._assignments) >= self.config.max_inflight
                ):
                    await asyncio.sleep(self.config.retry_interval)
                    continue
                grant = await self._budget_call(
                    action="peek", reservation_id=reservation_id
                )
                if not grant.get("success"):
                    await asyncio.sleep(self.config.retry_interval)
                    continue
                data = grant["data"]
                poll = {
                    "worker_url": self.config.worker_url,
                    "worker_epoch": self.worker_epoch,
                    "dp_rank": 0,
                    "poll_id": uuid.uuid4().hex,
                    "max_tokens": data["max_tokens"],
                    "max_requests": 1,
                    # Lets the router hold one uncommitted offer per claim loop.
                    "max_offered": self.config.claim_concurrency,
                    "wait_ms": self.config.poll_wait_ms,
                }
                # Retry the same poll, so a lost offer cannot create another
                # assignment for this claim. Abandoning an offer is safe:
                # without accept, the router cannot commit it.
                offer = None
                for attempt in range(3):
                    try:
                        offer = await self._control(router, "acquire", poll)
                        break
                    except (aiohttp.ClientError, TimeoutError):
                        if attempt == 2:
                            raise
                if offer.get("status") == "empty":
                    await asyncio.sleep(self.config.retry_interval)
                    continue
                assignment_id = offer.get("assignment_id")
                if offer.get("status") != "offer" or not isinstance(assignment_id, str):
                    raise PullProtocolError("Invalid prefill pull offer")
                if assignment_id in self._assignments:
                    raise PullProtocolError("Duplicate live prefill pull assignment")
                # Loops for every router passed the pre-poll check while their
                # polls were open; admit only as many offers as there is room.
                # No await between this check and the insertion below.
                if len(self._assignments) >= self.config.max_inflight:
                    with contextlib.suppress(
                        aiohttp.ClientError, TimeoutError, PullProtocolError
                    ):
                        await self._control(
                            router, "reject", self._identity(assignment_id)
                        )
                    continue
                task = asyncio.create_task(
                    self._run_assignment(router, reservation_id, offer)
                )
                self._assignments[assignment_id] = task
                self._assignment_reservations[assignment_id] = reservation_id
                self._assignment_routers[assignment_id] = router
                task.add_done_callback(
                    lambda task, key=assignment_id: self._assignment_done(key, task)
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(
                    "Prefill pull acquisition from %s failed: %s",
                    router,
                    exc,
                    exc_info=True,
                )
                await asyncio.sleep(self.config.retry_interval)

    def _assignment_done(self, assignment_id: str, task: asyncio.Task):
        self._assignments.pop(assignment_id, None)
        self._assignment_reservations.pop(assignment_id, None)
        self._assignment_routers.pop(assignment_id, None)
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
        try:
            body = self._execution_body(offer, reservation_id)
            # The peek before polling held nothing; another claim may have
            # taken the slot since, in which case the offer goes back.
            granted = await self._budget_call(
                action="reserve", reservation_id=reservation_id
            )
            if not granted.get("success"):
                await self._control(router, "reject", identity)
                return
            bound = await self._budget_call(
                action="bind",
                reservation_id=reservation_id,
                assignment_id=assignment_id,
                tokens=offer["input_tokens"],
                request_id=request_id,
            )
            if not bound.get("success"):
                await self._control(router, "reject", identity)
                return
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
            if monitor in done:
                # Revoke first. A /generate POST still in tokenization must
                # not enter the scheduler after cancellation was observed.
                await self._release(reservation_id)
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
            await self._release(reservation_id)
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
            await self._release(reservation_id)
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
            await self._release(reservation_id)
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
