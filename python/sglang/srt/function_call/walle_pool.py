"""Walle worker subprocesses (walle_worker.py) served from one event loop.

A schema goes to an idle worker, or waits FIFO for one. A worker that overruns
the execution timeout is killed and replaced. A worker found dead, or dying
mid-schema, is replaced and the schema retried once. This process never loads
libwalle itself.
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
import sys
from collections import deque
from pathlib import Path
from typing import Deque, Optional, Set

import msgspec

logger = logging.getLogger(__name__)

_WORKER_SCRIPT = str(Path(__file__).with_name("walle_worker.py"))
_READY_LINE = b'{"ready": true}\n'
# A reply holds a whole canonical schema; walle rejects inputs over ~120 KB,
# but inlining a $ref can grow the output.
_REPLY_LIMIT_BYTES = 64 << 20
# Arbitrary; a worker measured ~50 ms from spawn to ready.
_WORKER_START_TIMEOUT_S = 30.0
# Arbitrary; a schema walle has not seen takes well under 10 ms. A wait this
# long means every worker was busy.
_QUEUE_WAIT_LOG_S = 0.1


def _worker_argv() -> tuple:
    # -P: the script's directory, sglang's function_call package, must not
    # shadow top-level modules.
    return (sys.executable, "-P", _WORKER_SCRIPT)


def run_walle_worker_once() -> None:
    """Start a worker with no input and require its ready line; raises
    RuntimeError naming why it did not start."""
    try:
        result = subprocess.run(
            _worker_argv(),
            input=b"",
            capture_output=True,
            timeout=_WORKER_START_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            f"a walle worker did not start within {_WORKER_START_TIMEOUT_S:g} s"
        ) from None
    if result.returncode != 0 or result.stdout != _READY_LINE:
        stderr = result.stderr.decode("utf-8", "replace").strip()[-2000:]
        raise RuntimeError(
            f"a walle worker failed to start (exit code {result.returncode}): "
            f"{stderr or result.stdout[:200]!r}"
        )


class WalleBusyError(Exception):
    """No walle worker became idle within the queue timeout."""


class WalleCrashError(Exception):
    """Walle workers died on a schema twice in a row."""


class _ReplyTooLargeError(Exception):
    pass


# The worker is gone, or out of step with its pipe.
_WORKER_LOST_ERRORS = (EOFError, ConnectionError, msgspec.DecodeError)


class WalleReply(msgspec.Struct, frozen=True):
    """One worker answer; ``error`` is set when walle could not produce one."""

    canonical: Optional[str] = None
    warning: Optional[str] = None
    error: Optional[str] = None


class _WalleWorker:
    def __init__(self, process: asyncio.subprocess.Process):
        self.process = process

    @classmethod
    async def start(cls) -> _WalleWorker:
        process = await asyncio.create_subprocess_exec(
            *_worker_argv(),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            limit=_REPLY_LIMIT_BYTES,
        )
        worker = cls(process)
        try:
            async with asyncio.timeout(_WORKER_START_TIMEOUT_S):
                ready = await process.stdout.readline()
        except BaseException:
            worker.kill()
            raise
        if ready != _READY_LINE:
            worker.kill()
            raise RuntimeError(
                f"walle worker {process.pid} failed to start (see its stderr above)"
            )
        return worker

    async def call(self, schema_text: str) -> WalleReply:
        self.process.stdin.write(schema_text.encode("utf-8") + b"\n")
        await self.process.stdin.drain()
        try:
            line = await self.process.stdout.readline()
        except ValueError:
            # The reply overran the stream limit; its rest is still unread.
            raise _ReplyTooLargeError() from None
        if not line:
            raise EOFError(f"walle worker {self.process.pid} exited")
        return msgspec.json.decode(line, type=WalleReply)

    def exited(self) -> bool:
        return self.process.returncode is not None

    def kill(self) -> None:
        try:
            self.process.kill()
        except ProcessLookupError:
            pass


class WalleWorkerPool:
    """Walle workers bound to the running event loop, started on first use."""

    def __init__(
        self, *, num_workers: int, exec_timeout_s: float, queue_timeout_s: float
    ):
        self.exec_timeout_s = exec_timeout_s
        self.queue_timeout_s = queue_timeout_s
        # None is a slot whose worker has not started (or failed to restart).
        self._idle: Deque[Optional[_WalleWorker]] = deque([None] * num_workers)
        self._waiters: Deque[asyncio.Future] = deque()
        self._workers: Set[_WalleWorker] = set()
        self._replacements: Set[asyncio.Task] = set()

    async def canonical(self, schema_text: str) -> WalleReply:
        """Walle's answer for one schema.

        Raises WalleBusyError on queue timeout, and WalleCrashError when a
        fresh worker dies on the schema too."""
        worker = await self._acquire()
        for attempt in (1, 2):
            if worker is None or worker.exited():
                if worker is not None:
                    await self._forget(worker)
                try:
                    worker = await _WalleWorker.start()
                except BaseException:
                    self._release(None)
                    raise
                self._workers.add(worker)
            try:
                async with asyncio.timeout(self.exec_timeout_s):
                    reply = await worker.call(schema_text)
            except TimeoutError:
                self._replace(worker)
                return WalleReply(
                    error=f"too complex to validate within {self.exec_timeout_s:g} s"
                )
            except _ReplyTooLargeError:
                self._replace(worker)
                limit_mb = _REPLY_LIMIT_BYTES / (1 << 20)
                return WalleReply(
                    error=f"walle's canonical form exceeds {limit_mb:g} MB"
                )
            except _WORKER_LOST_ERRORS as exc:
                logger.error(
                    "walle worker %d lost on a schema (attempt %d): %r",
                    worker.process.pid,
                    attempt,
                    exc,
                )
                if attempt == 2:
                    self._replace(worker)
                    raise WalleCrashError(
                        "the schema validator crashed on this schema"
                    ) from None
                await self._forget(worker)
                worker = None
                continue
            except BaseException:
                # Cancelled mid-call: its reply would answer the next schema.
                self._replace(worker)
                raise
            self._release(worker)
            return reply

    def close(self) -> None:
        """Kill every worker; the pool is unusable afterwards."""
        for worker in self._workers:
            worker.kill()
        self._workers.clear()
        for task in self._replacements:
            task.cancel()

    async def _acquire(self) -> Optional[_WalleWorker]:
        if self._idle:
            return self._idle.popleft()
        loop = asyncio.get_running_loop()
        waiter = loop.create_future()
        self._waiters.append(waiter)
        started = loop.time()
        try:
            async with asyncio.timeout(self.queue_timeout_s):
                worker = await waiter
        except BaseException as exc:
            if waiter.done() and not waiter.cancelled():
                # Handed a worker just as this request gave up waiting.
                self._release(waiter.result())
            else:
                waiter.cancel()
            if isinstance(exc, TimeoutError):
                logger.warning(
                    "No walle worker became free within %g s; %d more schemas waiting",
                    self.queue_timeout_s,
                    self._waiting(),
                )
                raise WalleBusyError(
                    f"no schema validator became free within {self.queue_timeout_s:g} s"
                ) from None
            raise
        waited_s = loop.time() - started
        if waited_s >= _QUEUE_WAIT_LOG_S:
            logger.info(
                "Waited %.0f ms for a walle worker; %d more schemas waiting",
                waited_s * 1e3,
                self._waiting(),
            )
        return worker

    def _waiting(self) -> int:
        return sum(not waiter.done() for waiter in self._waiters)

    def _release(self, worker: Optional[_WalleWorker]) -> None:
        while self._waiters:
            waiter = self._waiters.popleft()
            if not waiter.done():
                waiter.set_result(worker)
                return
        self._idle.append(worker)

    async def _forget(self, worker: _WalleWorker) -> None:
        worker.kill()
        self._workers.discard(worker)
        await worker.process.wait()

    def _replace(self, worker: _WalleWorker) -> None:
        worker.kill()
        self._workers.discard(worker)
        task = asyncio.get_running_loop().create_task(self._start_replacement(worker))
        self._replacements.add(task)
        task.add_done_callback(self._replacements.discard)

    async def _start_replacement(self, dead: _WalleWorker) -> None:
        await dead.process.wait()
        replacement = None
        try:
            replacement = await _WalleWorker.start()
            self._workers.add(replacement)
        except Exception:
            logger.exception("Could not start a walle worker; retrying on next use")
        self._release(replacement)
