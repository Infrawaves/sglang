"""Walle verdicts for Kimi K3 strict tool schemas: the worker pool, the verdict
cache, the verdict rules and the property order restored on walle's output."""

import asyncio
import json
import os
import random
import signal
import tempfile
import time
import unittest
from unittest.mock import patch

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call import walle_pool
from sglang.srt.function_call.kimi_k3_property_order import restore_property_order
from sglang.srt.function_call.kimi_k3_tool_schema import (
    _WalleSchemaResolver,
    _WalleVerdict,
    _WalleVerdictCache,
    resolve_kimi_k3_tool_schemas,
)
from sglang.srt.function_call.walle_pool import (
    WalleBusyError,
    WalleCrashError,
    WalleReply,
    WalleWorkerPool,
    _WalleWorker,
    run_walle_worker_once,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.kimi_k3_tool_schema_utils import (
    run_on_tool_schema_loop,
    walle_installed,
)
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


def _slow_schema():
    """Walle v0.1.19 takes ~2 s on this and accepts it: 16 models, each with 3
    optional direct refs to others, and 10 `required` naming undeclared keys."""
    rng = random.Random(1)
    defs = {}
    for i in range(16):
        properties = {f"f{j}": {"type": "string"} for j in range(4)}
        for r, target in enumerate(rng.sample(range(16), 3)):
            properties[f"r{r}"] = {"$ref": f"#/$defs/M{target}"}
        required = ["f0", "undeclared"] if i < 10 else ["f0"]
        defs[f"M{i}"] = {
            "type": "object",
            "properties": properties,
            "required": required,
        }
    return {
        "type": "object",
        "properties": {"root": {"$ref": "#/$defs/M0"}},
        "required": ["root"],
        "$defs": defs,
    }


_SMALL = '{"type":"object","properties":{"b":{"type":"integer"},"a":{}}}'

# Echoes each schema as its canonical form, and dies on one that says "crash";
# "crash-once" dies only if the marker file named after it does not exist yet.
_FAKE_WORKER = """
import json, os, sys
out = sys.stdout.buffer
out.write(b'{"ready": true}\\n')
out.flush()
for line in sys.stdin.buffer:
    text = line.decode().rstrip("\\n")
    marker = os.environ.get("FAKE_WALLE_MARKER", "")
    if "crash-once" in text and not os.path.exists(marker):
        open(marker, "w").close()
        os._exit(1)
    if "crash-always" in text:
        os._exit(1)
    reply = {"canonical": text, "warning": None, "error": None}
    out.write(json.dumps(reply).encode() + b"\\n")
    out.flush()
"""


async def _close(pool):
    # Let replacements finish starting, so every worker is reaped on this loop.
    await asyncio.gather(*pool._replacements, return_exceptions=True)
    workers = list(pool._workers)
    pool.close()
    await asyncio.gather(*(worker.process.wait() for worker in workers))


async def _wait_for(predicate, timeout_s=10.0):
    deadline = time.monotonic() + timeout_s
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("condition not reached")
        await asyncio.sleep(0.01)


@unittest.skipUnless(walle_installed(), "walle is not installed")
class TestWalleWorkerPool(CustomTestCase):
    def test_round_trip_keeps_non_ascii_and_reports_walle_errors(self):
        async def run():
            pool = WalleWorkerPool(num_workers=1, exec_timeout_s=10, queue_timeout_s=10)
            try:
                reply = await pool.canonical(
                    json.dumps(
                        {"type": "object", "properties": {"城市": {"type": "string"}}},
                        ensure_ascii=False,
                    )
                )
                broken = await pool.canonical("not json")
                after = await pool.canonical(_SMALL)
            finally:
                await _close(pool)
            return reply, broken, after

        reply, broken, after = asyncio.run(run())
        self.assertEqual(
            json.loads(reply.canonical)["properties"], {"城市": {"type": "string"}}
        )
        self.assertIsNone(broken.canonical)
        self.assertIn("JSON", broken.error)
        # An error reply leaves the worker in step with the next schema.
        self.assertEqual(
            json.loads(after.canonical),
            {"type": "object", "properties": {"a": {}, "b": {"type": "integer"}}},
        )

    def test_execution_timeout_kills_the_worker_and_a_replacement_serves_the_queue(
        self,
    ):
        async def run():
            pool = WalleWorkerPool(
                num_workers=1, exec_timeout_s=0.3, queue_timeout_s=10
            )
            try:
                slow = asyncio.create_task(pool.canonical(json.dumps(_slow_schema())))
                await _wait_for(lambda: pool._workers)
                (timed_out,) = pool._workers
                with self.assertLogs(walle_pool.logger, "INFO") as logs:
                    queued = await pool.canonical(_SMALL)
                reply = await slow
                await asyncio.wait_for(timed_out.process.wait(), 5)
            finally:
                await _close(pool)
            return reply, timed_out, queued, logs.output

        reply, timed_out, queued, output = asyncio.run(run())
        self.assertEqual(reply.error, "too complex to validate within 0.3 s")
        self.assertIsNotNone(timed_out.process.returncode)
        self.assertIsNotNone(queued.canonical)
        self.assertTrue(any("Waited" in line for line in output), output)

    def test_idle_worker_killed_from_outside_is_replaced_before_the_next_schema(self):
        """An OOM-killed idle worker must not turn the next schema into a 400."""

        async def run():
            pool = WalleWorkerPool(num_workers=1, exec_timeout_s=10, queue_timeout_s=10)
            try:
                await pool.canonical(_SMALL)
                (killed,) = pool._workers
                os.kill(killed.process.pid, signal.SIGKILL)
                await killed.process.wait()
                return await pool.canonical(_SMALL)
            finally:
                await _close(pool)

        reply = asyncio.run(run())
        self.assertIsNone(reply.error)
        self.assertIsNotNone(reply.canonical)

    def test_reply_over_the_stream_limit_is_a_rejection_and_replaces_the_worker(self):
        async def run():
            pool = WalleWorkerPool(num_workers=1, exec_timeout_s=10, queue_timeout_s=10)
            try:
                await pool.canonical('{"type":"string"}')
                (overrun,) = pool._workers
                big = await pool.canonical(
                    json.dumps(_object(*(f"p{i}" for i in range(40))))
                )
                await asyncio.wait_for(overrun.process.wait(), 5)
                after = await pool.canonical('{"type":"string"}')
            finally:
                await _close(pool)
            return big, after

        with patch.object(walle_pool, "_REPLY_LIMIT_BYTES", 256):
            big, after = asyncio.run(run())
        self.assertTrue(big.error.startswith("walle's canonical form exceeds"))
        self.assertEqual(after.canonical, '{"type":"string"}')

    def test_queue_timeout_is_busy_and_leaves_the_worker_to_the_pool(self):
        """A request that gives up waiting must not keep the worker it was
        about to be handed: the pool still serves once the busy one is done."""

        async def run():
            pool = WalleWorkerPool(
                num_workers=1, exec_timeout_s=0.5, queue_timeout_s=0.1
            )
            try:
                slow = asyncio.create_task(pool.canonical(json.dumps(_slow_schema())))
                await _wait_for(lambda: pool._workers)
                started = time.monotonic()
                with self.assertLogs(walle_pool.logger, "WARNING") as logs:
                    with self.assertRaises(WalleBusyError):
                        await pool.canonical(_SMALL)
                waited = time.monotonic() - started
                self.assertIn(
                    "No walle worker became free within 0.1 s", logs.output[0]
                )
                await slow
                after = await asyncio.wait_for(pool.canonical(_SMALL), 10)
            finally:
                await _close(pool)
            return waited, after

        waited, after = asyncio.run(run())
        # Gave up on the queue, well before the busy worker's 0.5 s timeout.
        self.assertLess(waited, 1.0)
        self.assertIsNotNone(after.canonical)

    def test_worker_exits_when_its_stdin_closes(self):
        """A worker whose parent is gone sees EOF and must not linger."""

        async def run():
            worker = await _WalleWorker.start()
            worker.process.stdin.close()
            return await asyncio.wait_for(worker.process.wait(), 10)

        self.assertEqual(asyncio.run(run()), 0)


class TestWalleStartupCheck(CustomTestCase):
    def test_worker_without_the_ready_line_fails_the_check(self):
        """Anything on stdout before the ready line would answer the first schema."""
        with tempfile.TemporaryDirectory() as directory:
            script = os.path.join(directory, "fake_walle_worker.py")
            with open(script, "w") as file:
                file.write("print('hello')\n")
            with patch.object(walle_pool, "_WORKER_SCRIPT", script):
                with self.assertRaisesRegex(RuntimeError, "failed to start"):
                    run_walle_worker_once()


class TestWorkerCrash(CustomTestCase):
    """A worker dying mid-schema, driven by a stand-in worker script."""

    def setUp(self):
        super().setUp()
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        script = os.path.join(directory.name, "fake_walle_worker.py")
        with open(script, "w") as file:
            file.write(_FAKE_WORKER)
        self.marker = os.path.join(directory.name, "crashed-once")
        for patcher in (
            patch.object(walle_pool, "_WORKER_SCRIPT", script),
            patch.dict(os.environ, {"FAKE_WALLE_MARKER": self.marker}),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_a_crash_is_retried_once_on_a_fresh_worker(self):
        async def run():
            pool = WalleWorkerPool(num_workers=1, exec_timeout_s=10, queue_timeout_s=10)
            try:
                return await pool.canonical('"crash-once"')
            finally:
                await _close(pool)

        reply = asyncio.run(run())
        self.assertTrue(os.path.exists(self.marker))
        self.assertEqual(reply.canonical, '"crash-once"')

    def test_a_second_crash_raises_and_the_pool_still_serves(self):
        async def run():
            pool = WalleWorkerPool(num_workers=1, exec_timeout_s=10, queue_timeout_s=10)
            try:
                with self.assertRaises(WalleCrashError):
                    await pool.canonical('"crash-always"')
                return await pool.canonical('"fine"')
            finally:
                await _close(pool)

        self.assertEqual(asyncio.run(run()).canonical, '"fine"')


class _CountingPool:
    """Stands in for the worker pool, answering each schema unchanged."""

    def __init__(self):
        self.calls = 0

    async def canonical(self, schema_text):
        self.calls += 1
        await asyncio.sleep(0)
        return WalleReply(canonical=schema_text)


class TestWalleVerdictCache(CustomTestCase):
    def test_cache_is_lru_and_bounded_by_bytes(self):
        small = _WalleVerdict(canonical_text="{}")
        cache = _WalleVerdictCache(max_bytes=2 * small.size_bytes())
        cache.put(key=b"a", verdict=small)
        cache.put(key=b"b", verdict=small)
        self.assertEqual(cache.get(b"a"), small)
        cache.put(key=b"c", verdict=small)
        self.assertIsNone(cache.get(b"b"))

        cache.put(key=b"huge", verdict=_WalleVerdict(canonical_text=" " * 4096 + "{}"))
        self.assertIsNone(cache.get(b"huge"))
        self.assertEqual(cache.get(b"a"), small)
        self.assertEqual(cache.get(b"c"), small)

    def test_repeated_and_concurrent_asks_reach_walle_once(self):
        async def run():
            pool = _CountingPool()
            resolver = _WalleSchemaResolver(
                pool=pool, cache=_WalleVerdictCache(1 << 20)
            )
            schema = {"type": "object", "properties": {"b": {}, "a": {}}}
            first, second = await asyncio.gather(
                resolver.verdicts({0: schema}), resolver.verdicts({0: schema})
            )
            third = await resolver.verdicts({0: schema, 1: schema})
            return pool.calls, [*first.values(), *second.values(), *third.values()]

        calls, verdicts = asyncio.run(run())
        self.assertEqual(calls, 1)
        for _, rejection in verdicts:
            self.assertIsNone(rejection)

    @unittest.skipUnless(walle_installed(), "walle is not installed")
    def test_timeout_is_cached_as_a_rejection(self):
        async def run():
            pool = WalleWorkerPool(
                num_workers=1, exec_timeout_s=0.2, queue_timeout_s=10
            )
            resolver = _WalleSchemaResolver(
                pool=pool, cache=_WalleVerdictCache(1 << 20)
            )
            try:
                schemas = {0: _slow_schema()}
                first = await resolver.verdicts(schemas)
                with patch.object(pool, "canonical", side_effect=AssertionError):
                    second = await resolver.verdicts(schemas)
                return first, second
            finally:
                await _close(pool)

        first, second = asyncio.run(run())
        self.assertEqual(
            first,
            {
                0: (
                    None,
                    "Invalid tool parameters schema: too complex to validate within 0.2 s",
                )
            },
        )
        self.assertEqual(second, first)


def _worker_canonical(schema):
    """Walle's canonical form as a worker answers it, declared order restored."""

    async def run():
        pool = WalleWorkerPool(num_workers=1, exec_timeout_s=10, queue_timeout_s=10)
        try:
            return await pool.canonical(json.dumps(schema))
        finally:
            await _close(pool)

    reply = asyncio.run(run())
    assert reply.error is None, reply.error
    assert reply.canonical != "{}", reply.warning
    return json.loads(reply.canonical)


def _object(*names, **extra):
    return {
        "type": "object",
        "properties": {name: {"type": "string"} for name in names},
        **extra,
    }


def _deep_object(depth):
    node = {"type": "string"}
    for _ in range(depth):
        node = {"type": "object", "properties": {"a": node}}
    return node


def _chain_uses(*, length, uses):
    """``uses`` root properties, all one $ref into an alias chain ``length`` long."""
    defs = {f"d{i}": {"$ref": f"#/$defs/d{i + 1}"} for i in range(length)}
    defs[f"d{length}"] = _object("z", "a")
    return {
        "type": "object",
        "properties": {f"p{i}": {"$ref": "#/$defs/d0"} for i in range(uses)},
        "$defs": defs,
    }


def _reverse_orphans(length):
    """Unreferenced defs n0001 -> n0000 and so on, listed against the chain."""
    defs = {"n0000": _object("z", "a")}
    defs.update(
        {f"n{i:04d}": {"$ref": f"#/$defs/n{i - 1:04d}"} for i in range(1, length)}
    )
    return {
        "type": "object",
        "properties": {"x": {"type": "string"}},
        "$defs": dict(sorted(defs.items())),
    }


# Walle drops the string branch: it conflicts with the parent's type.
_DROPPED_BRANCH = {
    "type": "object",
    "properties": {
        "first": _object("a", "b"),
        "v": {
            "type": "object",
            "anyOf": [
                {"type": "string"},
                _object("b", "a"),
                {
                    "type": "object",
                    "properties": {"b": {"type": "integer"}, "a": {"type": "integer"}},
                },
            ],
        },
    },
}


_NESTED_ANYOF = {
    "type": "object",
    "properties": {
        "q": _object("a", "b"),
        "p": {
            "anyOf": [
                {"anyOf": [{"type": "string"}, _object("b", "a", required=["b", "a"])]},
                {"type": "null"},
            ]
        },
    },
    "required": ["p"],
}


def _sorted(schema):
    # Walle's own property order, for a schema it leaves otherwise unchanged.
    return json.loads(json.dumps(schema, sort_keys=True))


class TestPropertyOrderRestorer(CustomTestCase):
    """The restorer alone, on canonical forms written as walle emits them."""

    def test_equal_length_nested_anyof_keeps_the_inner_objects_order(self):
        """The inner object's declaration orders names it shares with another object."""
        restored = restore_property_order(
            canonical=_sorted(_NESTED_ANYOF), original=_NESTED_ANYOF
        )
        inner = restored["properties"]["p"]["anyOf"][0]["anyOf"][1]
        self.assertEqual(list(inner["properties"]), ["b", "a"])

    def test_unmatched_names_fall_back_to_their_first_declaration(self):
        canonical = {"properties": {"a": {}, "b": {}, "c": {}}, "type": "object"}
        original = {
            "type": "object",
            "properties": {"c": {}},
            "$defs": {"X": _object("b", "a")},
        }
        restored = restore_property_order(canonical=canonical, original=original)
        self.assertEqual(list(restored["properties"]), ["c", "b", "a"])

    def test_many_uses_of_a_long_alias_chain(self):
        """All uses of the shared alias chain retain the declared order."""
        schema = _chain_uses(length=2000, uses=1500)
        restored = restore_property_order(canonical=_sorted(schema), original=schema)
        self.assertEqual(list(restored["properties"]), [f"p{i}" for i in range(1500)])
        self.assertEqual(list(restored["$defs"]["d2000"]["properties"]), ["z", "a"])

    def test_unreferenced_defs_chained_against_their_order(self):
        schema = _reverse_orphans(2000)
        restored = restore_property_order(canonical=_sorted(schema), original=schema)
        self.assertEqual(list(restored["$defs"]["n0000"]["properties"]), ["z", "a"])

    def test_branches_after_one_walle_dropped_keep_their_own_order(self):
        canonical = _sorted(_DROPPED_BRANCH)
        canonical["properties"]["v"]["anyOf"] = canonical["properties"]["v"]["anyOf"][
            1:
        ]
        restored = restore_property_order(canonical=canonical, original=_DROPPED_BRANCH)
        self.assertEqual(
            [list(b["properties"]) for b in restored["properties"]["v"]["anyOf"]],
            [["b", "a"], ["b", "a"]],
        )
        self.assertEqual(
            list(restored["properties"]["first"]["properties"]), ["a", "b"]
        )


@unittest.skipUnless(walle_installed(), "walle is not installed")
class TestRestorePropertyOrder(CustomTestCase):
    """Walle sorts properties by name; the K3 grammar wants declaration order."""

    def test_nested_objects_and_items(self):
        schema = _object("zeta", "alpha")
        schema["properties"]["mid"] = {"type": "array", "items": _object("y", "b")}
        restored = _worker_canonical(schema)
        self.assertEqual(list(restored["properties"]), ["zeta", "alpha", "mid"])
        self.assertEqual(
            list(restored["properties"]["mid"]["items"]["properties"]), ["y", "b"]
        )

    def test_defs_keep_their_own_order_where_names_repeat(self):
        # A per-object match: a global first-declaration order would put the
        # second def's names in the first def's order.
        schema = {
            "type": "object",
            "properties": {"p": {"$ref": "#/$defs/P"}, "q": {"$ref": "#/$defs/Q"}},
            "$defs": {"P": _object("x", "a"), "Q": _object("a", "x")},
        }
        restored = _worker_canonical(schema)
        self.assertEqual(list(restored["$defs"]["P"]["properties"]), ["x", "a"])
        self.assertEqual(list(restored["$defs"]["Q"]["properties"]), ["a", "x"])

    def test_hoisted_definitions_take_their_use_sites_order(self):
        """Walle renames the definitions it hoists. Each keeps its own order where
        names repeat, including one that only another hoisted definition uses."""
        outer = {
            "type": "object",
            "properties": {
                "b": {"type": "string"},
                "inner": {"$ref": "#/definitions/Inner"},
                "a": {"type": "string"},
            },
        }
        schema = {
            "type": "object",
            "properties": {
                "q": _object("a", "b", "c", "d"),
                "v": {"$ref": "#/definitions/Outer"},
            },
            "definitions": {"Outer": outer, "Inner": _object("d", "c")},
        }
        restored = _worker_canonical(schema)
        self.assertEqual(
            sorted(list(d["properties"]) for d in restored["$defs"].values()),
            [["b", "inner", "a"], ["d", "c"]],
        )

    def test_ref_branches_when_walle_drops_one(self):
        """Walle drops the branch whose $ref names a string. The $ref it keeps and
        the object after it must each match their own original, not the dropped one."""
        schema = {
            "type": "object",
            "properties": {
                "q": _object("a", "b", "c", "d"),
                "p": {
                    "type": "object",
                    "anyOf": [
                        {"$ref": "#/$defs/S"},
                        {"$ref": "#/$defs/D"},
                        _object("b", "a"),
                    ],
                },
            },
            "$defs": {"S": {"type": "string"}, "D": _object("d", "c")},
        }
        restored = _worker_canonical(schema)
        branches = restored["properties"]["p"]["anyOf"]
        self.assertEqual(list(branches[-1]["properties"]), ["b", "a"])
        self.assertEqual(list(restored["$defs"]["D"]["properties"]), ["d", "c"])

    def test_ref_with_siblings_walle_inlines(self):
        schema = {
            "type": "object",
            "properties": {
                "v": {"$ref": "#/$defs/P", "properties": {"m": {"type": "string"}}}
            },
            "$defs": {"P": _object("q", "p")},
        }
        restored = _worker_canonical(schema)
        self.assertEqual(
            list(restored["properties"]["v"]["properties"]), ["q", "p", "m"]
        )

    def test_anyof_branches_walle_distributes_the_parent_into(self):
        schema = {
            "type": "object",
            "properties": {
                "v": {
                    **_object("k", "c"),
                    "anyOf": [_object("y", "x"), {"required": ["k"]}],
                }
            },
        }
        restored = _worker_canonical(schema)
        branches = restored["properties"]["v"]["anyOf"]
        self.assertEqual(len(branches), 2)
        self.assertEqual(list(branches[0]["properties"]), ["k", "c", "y", "x"])
        self.assertEqual(list(branches[1]["properties"]), ["k", "c"])

    def test_wide_anyof_matches_each_branch_to_its_own(self):
        # walle's 500-branch maximum; each branch declares its names reversed.
        schema = {
            "type": "object",
            "properties": {
                "p": {"anyOf": [_object(f"z{i}", f"a{i}") for i in range(500)]}
            },
        }
        branches = _worker_canonical(schema)["properties"]["p"]["anyOf"]
        self.assertEqual(
            [list(branch["properties"]) for branch in branches],
            [[f"z{i}", f"a{i}"] for i in range(500)],
        )

    def test_nested_anyof_matches_each_branch_to_its_own(self):
        inner = [_object(f"y{i}") for i in range(40)]
        outer = {"type": "object", "properties": {"q": {"anyOf": inner}, "b": {}}}
        schema = {"type": "object", "properties": {"p": {"anyOf": [outer] * 40}}}
        branches = _worker_canonical(schema)["properties"]["p"]["anyOf"]
        self.assertEqual(len(branches), 40)
        for branch in branches:
            self.assertEqual(list(branch["properties"]), ["q", "b"])
            self.assertEqual(
                [list(b["properties"]) for b in branch["properties"]["q"]["anyOf"]],
                [[f"y{i}"] for i in range(40)],
            )

    def test_deep_nesting_under_keywords_walle_drops(self):
        """walle drops not, allOf and extension keywords whole, so it accepts
        them nested far deeper than the interpreter's recursion limit."""
        for keyword in ("not", "allOf", "x-ext"):
            with self.subTest(keyword=keyword):
                deep = _deep_object(3000)
                dropped = {"type": "string"}
                dropped[keyword] = [deep] if keyword == "allOf" else deep
                schema = {"type": "object", "properties": {"s": dropped, "b": {}}}
                tool = Tool(function=Function(name="f", strict=True, parameters=schema))
                canonical, rejection = run_on_tool_schema_loop(
                    resolve_kimi_k3_tool_schemas(tools=[tool])
                )
                self.assertIsNone(rejection)
                self.assertEqual(list(canonical[0]["properties"]), ["s", "b"])


if __name__ == "__main__":
    unittest.main()
