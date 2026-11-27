"""Verdict and canonical form of Kimi K3 strict tool parameters.

The Kimi Vendor Verifier holds vendors to Moonshot walle's verdicts: a schema
walle rejects gets a 400, and one it accepts is constrained as walle's canonical
form. Worker subprocesses run the official Walle v0.1.19 validator.
"""

from __future__ import annotations

import asyncio
import functools
import hashlib
import importlib.metadata
import json
import logging
import sys
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import msgspec

from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.environ import ToolStrictLevel, envs
from sglang.srt.function_call.walle_pool import (
    WalleCrashError,
    WalleReply,
    WalleWorkerPool,
    run_walle_worker_once,
)
from sglang.srt.observability.req_time_stats import monotonic_time

logger = logging.getLogger(__name__)

_REQUIRED_WALLE_VERSION = "0.1.19"


class KimiK3ToolSchemaInternalError(Exception):
    """A server-side failure to produce a tool's canonical parameters; never
    the client's fault, so never a 4xx."""


def needs_canonical_parameters(tool: Tool) -> bool:
    """Whether the K3 grammar constrains this tool's arguments to its schema."""
    function = tool.function
    return isinstance(function.parameters, dict) and (
        function.strict
        or envs.SGLANG_TOOL_STRICT_LEVEL.get() >= ToolStrictLevel.PARAMETER
    )


def check_walle_installed() -> None:
    """Require the pinned Walle release and a working native library.

    Loads libwalle in a throwaway worker, never here: loading it in this
    process would start a Go runtime that every later fork inherits broken."""
    try:
        installed = importlib.metadata.version("walle")
    except importlib.metadata.PackageNotFoundError:
        installed = None
    if installed is None:
        problem = "the walle Python package is not installed"
    elif installed != _REQUIRED_WALLE_VERSION:
        problem = f"installed version is {installed}"
    else:
        try:
            _check_walle_worker_starts()
            return
        except RuntimeError as exc:
            problem = str(exc)
    raise RuntimeError(
        f"Kimi K3 tool validation requires Walle v{_REQUIRED_WALLE_VERSION}: {problem}. "
        "Install the Walle wheel built with libwalle.so for this platform."
    )


# A library built for another architecture or libc only fails on load.
_check_walle_worker_starts = functools.cache(run_walle_worker_once)


async def resolve_kimi_k3_tool_schemas(
    tools: List[Tool],
) -> Tuple[List[Optional[Dict[str, Any]]], Optional[str]]:
    """Canonical parameters per tool (None where not strict), or a 400 message.

    Raises WalleBusyError on queue timeout or WalleCrashError after a failed retry."""
    canonical: List[Optional[dict]] = [None] * len(tools)
    schemas = {
        index: tool.function.parameters
        for index, tool in enumerate(tools)
        if needs_canonical_parameters(tool)
    }
    if not schemas:
        return canonical, None
    verdicts = await _process_walle_resolver().verdicts(schemas)
    for index, (parameters, rejection) in verdicts.items():
        if rejection is not None:
            return [], f"tool {tools[index].function.name!r}: {rejection}"
        canonical[index] = parameters
    return canonical, None


# Per entry besides its text: the 32-byte key's bytes object, the OrderedDict
# slot and link node, and the verdict Struct; ~188 B by tracemalloc on CPython 3.12.
_CACHE_ENTRY_OVERHEAD_BYTES = 192


class _WalleVerdict(msgspec.Struct, frozen=True):
    """Walle's verdict: the order-restored canonical text, or why it rejected."""

    canonical_text: Optional[str] = None
    rejection: Optional[str] = None

    def size_bytes(self) -> int:
        text = self.canonical_text if self.rejection is None else self.rejection
        return _CACHE_ENTRY_OVERHEAD_BYTES + sys.getsizeof(text)

    def result(self) -> Tuple[Optional[dict], Optional[str]]:
        if self.rejection is not None:
            return None, self.rejection
        return json.loads(self.canonical_text), None


def _walle_verdict(*, reply: WalleReply, parameters: Dict[str, Any]) -> _WalleVerdict:
    if reply.error is not None:
        return _WalleVerdict(rejection=f"Invalid tool parameters schema: {reply.error}")
    if reply.canonical == "{}" and parameters != {}:
        # Walle reduces a schema it cannot repair to {}.
        reason = reply.warning or "walle reduced it to {}"
        return _WalleVerdict(rejection=f"Invalid tool parameters schema: {reason}")
    if reply.warning:
        logger.debug("walle repaired a tool parameters schema: %s", reply.warning)
    return _WalleVerdict(canonical_text=reply.canonical)


class _WalleVerdictCache:
    """LRU of walle verdicts, bounded by the memory its entries hold."""

    def __init__(self, max_bytes: int):
        self.max_bytes = max_bytes
        self.size_bytes = 0
        self._entries: OrderedDict[bytes, _WalleVerdict] = OrderedDict()

    def get(self, key: bytes) -> Optional[_WalleVerdict]:
        verdict = self._entries.get(key)
        if verdict is not None:
            self._entries.move_to_end(key)
        return verdict

    def put(self, *, key: bytes, verdict: _WalleVerdict) -> None:
        size = verdict.size_bytes()
        if size > self.max_bytes or key in self._entries:
            return
        self._entries[key] = verdict
        self.size_bytes += size
        while self.size_bytes > self.max_bytes:
            _, evicted = self._entries.popitem(last=False)
            self.size_bytes -= evicted.size_bytes()


class _WalleSchemaResolver:
    """Cached walle verdicts on one event loop; concurrent asks share one call."""

    def __init__(self, *, pool: WalleWorkerPool, cache: _WalleVerdictCache):
        self.loop = asyncio.get_running_loop()
        self.pool = pool
        self.cache = cache
        self._inflight: Dict[bytes, asyncio.Task] = {}

    async def verdicts(
        self, schemas: Dict[int, Dict[str, Any]]
    ) -> Dict[int, Tuple[Optional[dict], Optional[str]]]:
        """Verdicts by tool index. A cached rejection settles the whole request."""
        found = {}
        missing = {}
        for index, parameters in schemas.items():
            text = json.dumps(parameters, ensure_ascii=False)
            key = hashlib.sha256(text.encode("utf-8")).digest()
            verdict = self.cache.get(key)
            if verdict is None:
                missing[index] = (key, text)
            elif verdict.rejection is not None:
                return {index: verdict.result()}
            else:
                found[index] = verdict

        if missing:
            asked = await asyncio.gather(
                *(
                    self._shared_ask(key=key, text=text, parameters=schemas[index])
                    for index, (key, text) in missing.items()
                )
            )
            found.update(zip(missing, asked))
        return {index: verdict.result() for index, verdict in found.items()}

    async def _shared_ask(
        self, *, key: bytes, text: str, parameters: Dict[str, Any]
    ) -> _WalleVerdict:
        task = self._inflight.get(key)
        if task is None:
            task = self.loop.create_task(
                self._ask_walle(key=key, text=text, parameters=parameters)
            )
            self._inflight[key] = task
            task.add_done_callback(lambda _: self._inflight.pop(key, None))
        # A cancelled request must not cancel the call others may share.
        return await asyncio.shield(task)

    async def _ask_walle(
        self, *, key: bytes, text: str, parameters: Dict[str, Any]
    ) -> _WalleVerdict:
        started = monotonic_time()
        try:
            reply = await self.pool.canonical(text)
        except WalleCrashError as exc:
            logger.info(
                "walle tool schema %s: %s in %.1f ms",
                key.hex()[:16],
                exc,
                (monotonic_time() - started) * 1e3,
            )
            raise
        verdict = _walle_verdict(reply=reply, parameters=parameters)
        self.cache.put(key=key, verdict=verdict)
        # Prefill and decode each ask walle; the hash lines up their verdicts.
        logger.info(
            "walle tool schema %s: %s in %.1f ms",
            key.hex()[:16],
            "accepted" if verdict.rejection is None else verdict.rejection,
            (monotonic_time() - started) * 1e3,
        )
        return verdict


_PROCESS_WALLE_RESOLVER: Optional[_WalleSchemaResolver] = None
_PROCESS_VERDICT_CACHE: Optional[_WalleVerdictCache] = None


def _process_walle_resolver() -> _WalleSchemaResolver:
    """This process's resolver, on the running loop; its workers start lazily."""
    global _PROCESS_WALLE_RESOLVER, _PROCESS_VERDICT_CACHE
    resolver = _PROCESS_WALLE_RESOLVER
    if resolver is not None and resolver.loop is asyncio.get_running_loop():
        return resolver
    if resolver is not None:
        # Workers are bound to the loop that started them.
        resolver.pool.close()
    if _PROCESS_VERDICT_CACHE is None:
        _PROCESS_VERDICT_CACHE = _WalleVerdictCache(
            envs.SGLANG_KIMI_K3_WALLE_CACHE_MB.get() << 20
        )
    resolver = _WalleSchemaResolver(
        pool=WalleWorkerPool(
            num_workers=envs.SGLANG_KIMI_K3_WALLE_WORKERS.get(),
            exec_timeout_s=envs.SGLANG_KIMI_K3_WALLE_TIMEOUT_S.get(),
            queue_timeout_s=envs.SGLANG_KIMI_K3_WALLE_QUEUE_TIMEOUT_S.get(),
        ),
        cache=_PROCESS_VERDICT_CACHE,
    )
    _PROCESS_WALLE_RESOLVER = resolver
    return resolver
