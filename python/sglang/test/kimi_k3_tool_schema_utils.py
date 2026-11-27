"""Test support for Kimi K3 strict tool schemas validated by Walle workers."""

import asyncio
import functools
import importlib.util


def walle_installed() -> bool:
    """An installed wheel must exercise the worker, including native load failures."""
    return importlib.util.find_spec("walle") is not None


@functools.cache
def _tool_schema_loop() -> asyncio.AbstractEventLoop:
    # One loop per process, as walle workers belong to the loop that started
    # them; made on first use, after imports have set the event loop policy.
    return asyncio.new_event_loop()


def run_on_tool_schema_loop(coroutine):
    """Run ``coroutine`` to completion on the process's shared test loop."""
    return _tool_schema_loop().run_until_complete(coroutine)
