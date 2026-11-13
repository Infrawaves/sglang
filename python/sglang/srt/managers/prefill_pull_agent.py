"""HTTP bridge for scheduler-authorized prefill request pulling.

The scheduler owns capacity and admission. This asynchronous agent owns only
network I/O: reserve capacity, acquire an offer, bind it, obtain a commit, and
execute the ordinary local /generate endpoint. An offer is never executable.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import ssl
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

import aiohttp
import msgspec

logger = logging.getLogger(__name__)


class PrefillPullAgentConfig(msgspec.Struct, frozen=True):
    router_url: str
    worker_url: str
    local_url: str
    max_inflight: int
    api_key: str | None = None
    router_api_key: str | None = None
    local_ssl: ssl.SSLContext | bool = True
    poll_wait_ms: int = 10000
    retry_interval: float = 0.5
    status_interval: float = 1.0
    control_timeout: float = 15.0
    max_response_bytes: int = 4 * 1024 * 1024
    max_offer_bytes: int = 16 * 1024 * 1024


class PullProtocolError(RuntimeError):
    pass


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
        self._acquirer: asyncio.Task | None = None
        self._assignments: dict[str, asyncio.Task] = {}
        self._assignment_reservations: dict[str, str] = {}
        self._stopping = False

    async def start(self):
        if self._session is not None:
            raise RuntimeError("Prefill pull agent is already running")
        if self.config.max_inflight < 1:
            raise ValueError("Prefill pull max_inflight must be positive")
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=None, sock_connect=10),
            connector=aiohttp.TCPConnector(limit=self.config.max_inflight * 2 + 4),
        )
        self._acquirer = asyncio.create_task(self._acquire_loop())

    async def close(self):
        self._stopping = True
        assignments = [
            (assignment_id, self._assignment_reservations[assignment_id], task)
            for assignment_id, task in self._assignments.items()
        ]
        if self._acquirer is not None:
            self._acquirer.cancel()
            await asyncio.gather(self._acquirer, return_exceptions=True)
        tasks = [task for _, _, task in assignments]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        # A task cancelled before its coroutine first runs has no finally block
        # to revoke the grant. Keep shutdown cleanup with the task's owner too.
        await asyncio.gather(
            *(
                self._close_assignment(assignment_id, reservation_id)
                for assignment_id, reservation_id, _ in assignments
            )
        )
        if self._session is not None:
            await self._session.close()
            self._session = None

    async def _close_assignment(self, assignment_id: str, reservation_id: str):
        await self._release(reservation_id)
        # Reject also cancels committed assignments; the router must not keep
        # waiting for an outcome from an agent that has shut down. A committed
        # assignment remains non-replayable even if this notification is lost.
        with contextlib.suppress(aiohttp.ClientError, TimeoutError, PullProtocolError):
            await self._control("reject", self._identity(assignment_id))

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

    async def _control(self, action: str, body: dict[str, Any]) -> dict[str, Any]:
        headers = {}
        if self.config.router_api_key:
            headers["Authorization"] = f"Bearer {self.config.router_api_key}"
        async with self._session.post(
            self.config.router_url.rstrip("/") + "/internal/pull/" + action,
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

    async def _acquire_loop(self):
        while not self._stopping:
            reservation_id = uuid.uuid4().hex
            handed_off = False
            reservation_requested = False
            try:
                if (
                    not self._ready()
                    or len(self._assignments) >= self.config.max_inflight
                ):
                    await asyncio.sleep(self.config.retry_interval)
                    continue
                reservation_requested = True
                grant = await self._budget_call(
                    action="reserve", reservation_id=reservation_id
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
                    "wait_ms": self.config.poll_wait_ms,
                }
                # Retry the same poll, so a lost offer cannot create another
                # assignment against this reservation. Abandoning an offer is
                # safe: without accept, the router cannot commit it.
                offer = None
                for attempt in range(3):
                    try:
                        offer = await self._control("acquire", poll)
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
                task = asyncio.create_task(self._run_assignment(reservation_id, offer))
                self._assignments[assignment_id] = task
                self._assignment_reservations[assignment_id] = reservation_id
                task.add_done_callback(
                    lambda task, key=assignment_id: self._assignment_done(key, task)
                )
                handed_off = True
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(
                    "Prefill pull acquisition failed: %s", exc, exc_info=True
                )
                await asyncio.sleep(self.config.retry_interval)
            finally:
                if reservation_requested and not handed_off:
                    await self._release(reservation_id)

    def _assignment_done(self, assignment_id: str, task: asyncio.Task):
        self._assignments.pop(assignment_id, None)
        self._assignment_reservations.pop(assignment_id, None)
        if not task.cancelled() and task.exception() is not None:
            logger.error("Prefill pull assignment failed: %s", task.exception())

    async def _commit(self, identity: dict[str, Any]) -> bool:
        try:
            result = await self._control("accept", identity)
        except (aiohttp.ClientError, TimeoutError, PullProtocolError):
            # A timeout does not tell us whether the router committed. Query
            # this assignment; do not execute speculatively or acquire a retry.
            result = await self._control("status", identity)
            if result.get("status") == "offered":
                result = await self._control("accept", identity)
        return result.get("status") == "committed"

    async def _run_assignment(self, reservation_id: str, offer: dict[str, Any]):
        assignment_id = offer["assignment_id"]
        identity = self._identity(assignment_id)
        request_id = offer.get("request_id")
        execution = None
        monitor = None
        committed = False
        try:
            body = self._execution_body(offer, reservation_id)
            bound = await self._budget_call(
                action="bind",
                reservation_id=reservation_id,
                assignment_id=assignment_id,
                tokens=offer["input_tokens"],
                request_id=request_id,
            )
            if not bound.get("success"):
                await self._control("reject", identity)
                return
            committed = await self._commit(identity)
            if not committed:
                return
            execution = asyncio.create_task(
                self._execute(body, offer.get("headers", {}))
            )
            monitor = asyncio.create_task(self._monitor(identity))
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
            await self._report_outcome(identity, status_code, response_body)
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
                identity, 502, json.dumps({"error": "Prefill pull execution failed"})
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
            payload = await self._read_limited(response, self.config.max_response_bytes)
            body = payload.decode("utf-8", errors="replace")
            # Replacement characters can expand malformed upstream bytes. The
            # router's outcome limit applies to the resulting UTF-8 string.
            if len(body.encode("utf-8")) > self.config.max_response_bytes:
                raise PullProtocolError("Decoded prefill response exceeds size limit")
            return response.status, body

    async def _monitor(self, identity: dict[str, Any]):
        failures = 0
        while True:
            await asyncio.sleep(self.config.status_interval)
            try:
                status = (await self._control("status", identity)).get("status")
                failures = 0
                if status != "committed":
                    return
            except (aiohttp.ClientError, TimeoutError, PullProtocolError):
                failures += 1
                if failures >= 3:
                    # Fail closed during sustained loss of the control plane;
                    # cancellation does not make this attempt safe to replay.
                    return

    async def _report_outcome(self, identity, status_code: int, body: str):
        while not self._stopping:
            try:
                result = await self._control(
                    "outcome", {**identity, "status_code": status_code, "body": body}
                )
                if result.get("status") in ("completed", "cancelled", "expired"):
                    return
                raise PullProtocolError("Router did not acknowledge prefill outcome")
            except (aiohttp.ClientError, TimeoutError, PullProtocolError):
                await asyncio.sleep(self.config.retry_interval)
                with contextlib.suppress(
                    aiohttp.ClientError, TimeoutError, PullProtocolError
                ):
                    result = await self._control("status", identity)
                    if result.get("status") in ("completed", "cancelled", "expired"):
                        return
