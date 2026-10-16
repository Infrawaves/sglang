# SPDX-License-Identifier: Apache-2.0
"""CPU tests; optional real Redis script tests use REDIS_SERVER_BIN or PATH.

Run directly with Python to avoid importing the GPU-serving dependency tree:
    python test/registered/unit/cache_accounting/test_page_cache.py -v
"""

import asyncio
import importlib.util
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import uuid
from dataclasses import replace
from pathlib import Path
from unittest.mock import AsyncMock, patch

# Load this dependency-free module without importing sglang.__init__, which
# initializes the model-serving environment. This also permits stdlib-only QA.
_SOURCE = (
    Path(__file__).resolve().parents[4]
    / "python/sglang/srt/cache_accounting/page_cache.py"
)
_SPEC = importlib.util.spec_from_file_location("_page_cache_test", _SOURCE)
cache = importlib.util.module_from_spec(_SPEC)
with patch.dict(sys.modules, {_SPEC.name: cache}):
    _SPEC.loader.exec_module(cache)

# CI registration is parsed statically; avoid importing heavyweight sglang.
if "sglang.test.ci.ci_register" in sys.modules:
    from sglang.test.ci.ci_register import register_cpu_ci
else:

    def register_cpu_ci(**kwargs):
        """Runtime no-op when testing without the serving package installed."""


register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def plan(ids, *, ttl=300, page_size=4, **kwargs):
    return cache.build_page_cache_plan(
        ids,
        kwargs.pop("api_key", "api-key-1"),
        kwargs.pop("namespace", "weights-v1/tokenizer-v1/template-v1/adapter-none"),
        page_size,
        ttl,
        **kwargs,
    )


class TestPageCachePlan(unittest.TestCase):
    def test_page_boundaries_and_partial_tail(self):
        for length, expected in [(0, 0), (3, 0), (4, 1), (5, 1), (8, 2)]:
            with self.subTest(length=length):
                result = plan(list(range(length)))
                self.assertEqual(len(result.page_keys), expected)
                self.assertEqual(result.prompt_tokens, length)

    def test_tail_changes_do_not_change_complete_pages(self):
        self.assertEqual(
            plan([1, 2, 3, 4, 5]).page_keys, plan([1, 2, 3, 4, 6]).page_keys
        )

    def test_identical_pages_after_different_history_do_not_match(self):
        original = plan([1, 2, 3, 4, 10, 11, 12, 13])
        other = plan([5, 6, 7, 8, 10, 11, 12, 13])
        self.assertNotEqual(original.page_keys[1], other.page_keys[1])

    def test_extension_shares_only_the_identical_prefix(self):
        shorter = plan(list(range(8)))
        extended = plan(list(range(13)))
        self.assertEqual(shorter.page_keys, extended.page_keys[:2])
        changed = plan([*range(4), 99, *range(5, 13)])
        self.assertEqual(shorter.page_keys[0], changed.page_keys[0])
        self.assertNotEqual(shorter.page_keys[1], changed.page_keys[1])

    def test_api_key_namespace_page_size_and_ttl_are_isolated(self):
        original = plan(list(range(12)))
        for change in [
            {"api_key": "api-key-2"},
            {"namespace": "weights-v2"},
            {"page_size": 3},
            {"ttl": 3600},
        ]:
            with self.subTest(change=change):
                other = plan(list(range(12)), **change)
                self.assertTrue(set(original.page_keys).isdisjoint(other.page_keys))

    def test_all_script_keys_share_a_first_page_hash_tag(self):
        result = plan(list(range(20)))
        tags = {key.split("{", 1)[1].split("}", 1)[0] for key in result.page_keys}
        self.assertEqual(len(tags), 1)
        other = plan([99, *range(1, 20)])
        self.assertNotEqual(
            next(iter(tags)), other.page_keys[0].split("{", 1)[1].split("}", 1)[0]
        )

    def test_token_encoding_is_unambiguous_and_validates_tail(self):
        self.assertNotEqual(
            plan([1, 23], page_size=2).page_keys, plan([12, 3], page_size=2).page_keys
        )
        self.assertTrue(plan([0, 0xFFFFFFFF], page_size=2).page_keys)
        for bad in [-1, 0x100000000, True, 1.0, "1"]:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                plan([0, 1, 2, 3, bad])

    def test_reject_invalid_options(self):
        for change in [
            {"page_size": 0},
            {"page_size": True},
            {"ttl": 600},
            {"api_key": ""},
            {"namespace": ""},
        ]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                plan([1, 2, 3, 4], **change)

    def test_more_than_4096_complete_pages_are_supported(self):
        result = plan(list(range(4097)), page_size=1)
        self.assertEqual(len(result.page_keys), 4097)
        self.assertEqual(result.prompt_tokens, 4097)
        self.assertIsNone(cache._validate_plan(result))


class TestRedisBoundary(unittest.IsolatedAsyncioTestCase):
    async def test_no_complete_page_never_initializes_redis(self):
        store = cache.RedisPageCacheStore("redis://unused")
        with patch.object(
            store, "_client", side_effect=AssertionError("must stay lazy")
        ):
            usage = await store.account(plan([1, 2, 3]))
        self.assertEqual(
            usage.to_dict(),
            {
                "read_tokens": 0,
                "creation_5m_tokens": 0,
                "creation_1h_tokens": 0,
                "normal_input_tokens": 3,
            },
        )
        await store.aclose()

    async def test_redis_error_and_malformed_result_do_not_invent_usage(self):
        for outcome in [
            ConnectionError("offline"),
            [0, 8, 0, 0],
            [0, 4, 0, 1],
            [1, 3, 0, 0],
            [0, 0, 4, 0],
        ]:
            with self.subTest(outcome=outcome):
                store = cache.RedisPageCacheStore("redis://unused")
                client = AsyncMock()
                if isinstance(outcome, Exception):
                    client.eval.side_effect = outcome
                else:
                    client.eval.return_value = outcome
                store._redis = client
                with self.assertRaises(cache.CacheAccountingError):
                    await store.account(plan([1, 2, 3, 4]))
                await store.aclose()

    async def test_timeout_is_unconfirmed(self):
        async def slow_eval(*args):
            await asyncio.sleep(1)

        store = cache.RedisPageCacheStore("redis://unused", timeout_seconds=0.01)
        store._redis = AsyncMock()
        store._redis.eval.side_effect = slow_eval
        with self.assertRaises(cache.CacheAccountingError) as failure:
            await store.account(plan([1, 2, 3, 4]))
        self.assertIsInstance(failure.exception.__cause__, TimeoutError)
        await store.aclose()

    async def test_invalid_plan_is_rejected_before_redis(self):
        original = plan(list(range(8)))
        malformed = replace(original, page_keys=(original.page_keys[0], "wrong-slot"))
        store = cache.RedisPageCacheStore("redis://unused")
        with patch.object(
            store, "_client", side_effect=AssertionError("must not connect")
        ):
            with self.assertRaises(cache.CacheAccountingError):
                await store.account(malformed)

    @unittest.skipUnless(
        importlib.util.find_spec("redis"), "redis Python package required"
    )
    async def test_redis_client_does_not_retry_failed_operations(self):
        from redis.exceptions import ConnectionError, TimeoutError

        store = cache.RedisPageCacheStore("redis://unused")
        retry = store._client().connection_pool.connection_kwargs["retry"]
        for error_type in (ConnectionError, TimeoutError):
            with self.subTest(error_type=error_type):
                attempts = 0

                async def failing_operation():
                    nonlocal attempts
                    attempts += 1
                    raise error_type("unconfirmed")

                async def handle_failure(error):
                    pass

                with self.assertRaises(error_type):
                    await retry.call_with_retry(failing_operation, handle_failure)
                self.assertEqual(attempts, 1)
        await store.aclose()


_REDIS_SERVER = os.environ.get("REDIS_SERVER_BIN") or shutil.which("redis-server")
_HAS_REDIS = importlib.util.find_spec("redis") is not None


@unittest.skipUnless(
    _REDIS_SERVER and _HAS_REDIS, "real redis-server and redis Python package required"
)
class TestRedisScript(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory(prefix="sgl-redis-", dir="/tmp")
        cls.socket_path = str(Path(cls.directory.name) / "redis.sock")
        cls.process = subprocess.Popen(
            [
                _REDIS_SERVER,
                "--port",
                "0",
                "--unixsocket",
                cls.socket_path,
                "--save",
                "",
                "--appendonly",
                "no",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            with socket.socket(socket.AF_UNIX) as probe:
                try:
                    probe.connect(cls.socket_path)
                    return
                except OSError:
                    time.sleep(0.01)
        cls.tearDownClass()
        raise RuntimeError("test Redis failed to start")

    @classmethod
    def tearDownClass(cls):
        cls.process.terminate()
        try:
            cls.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            cls.process.kill()
            cls.process.wait(timeout=5)
        cls.directory.cleanup()

    async def asyncSetUp(self):
        self.store = cache.RedisPageCacheStore(f"unix://{self.socket_path}")
        self.namespace = uuid.uuid4().hex

    async def asyncTearDown(self):
        await self.store.aclose()

    def make_plan(self, count=9, *, ttl=300):
        return plan(
            list(range(count)),
            namespace=self.namespace,
            ttl=ttl,
        )

    async def test_prefix_extension_read_write_and_tail_conserve_tokens(self):
        initial = await self.store.account(self.make_plan(9))
        self.assertEqual(initial, cache.CacheUsage(0, 8, 0, 1))
        extended = await self.store.account(self.make_plan(14))
        self.assertEqual(extended, cache.CacheUsage(8, 4, 0, 2))

    async def test_more_than_4096_pages_create_and_read_conserve_tokens(self):
        initial = plan(list(range(4097)), page_size=1, namespace=self.namespace)
        created = await self.store.account(initial)
        self.assertEqual(created, cache.CacheUsage(0, 4097, 0, 0))
        read = await self.store.account(initial)
        self.assertEqual(read, cache.CacheUsage(4097, 0, 0, 0))

    async def test_ttl_tiers_are_independent_and_hits_renew_selected_tier(self):
        long = self.make_plan(ttl=3600)
        short = self.make_plan(ttl=300)
        self.assertEqual(await self.store.account(long), cache.CacheUsage(0, 0, 8, 1))
        self.assertEqual(await self.store.account(short), cache.CacheUsage(0, 8, 0, 1))
        client = self.store._client()
        await client.expire(short.page_keys[0], 5)
        self.assertEqual(
            await self.store.account(self.make_plan()),
            cache.CacheUsage(8, 0, 0, 1),
        )
        self.assertGreaterEqual(await client.ttl(short.page_keys[0]), 298)
        self.assertGreaterEqual(await client.ttl(long.page_keys[0]), 3598)

    async def test_ttl_tiers_never_read_each_other_in_either_order(self):
        for first_ttl, second_ttl in ((300, 3600), (3600, 300)):
            with self.subTest(first_ttl=first_ttl):
                namespace = f"{self.namespace}:{first_ttl}"
                first = plan(list(range(9)), namespace=namespace, ttl=first_ttl)
                second = plan(
                    list(range(9)),
                    namespace=namespace,
                    ttl=second_ttl,
                )
                expected_first = cache.CacheUsage(
                    0, 8 if first_ttl == 300 else 0, 8 if first_ttl == 3600 else 0, 1
                )
                expected_second = cache.CacheUsage(
                    0, 8 if second_ttl == 300 else 0, 8 if second_ttl == 3600 else 0, 1
                )
                self.assertEqual(await self.store.account(first), expected_first)
                self.assertEqual(await self.store.account(second), expected_second)
                for ttl in (300, 3600):
                    following = plan(list(range(9)), namespace=namespace, ttl=ttl)
                    self.assertEqual(
                        await self.store.account(following),
                        cache.CacheUsage(8, 0, 0, 1),
                    )

    async def test_renewing_either_ttl_tier_leaves_other_tier_expiration_untouched(
        self,
    ):
        client = self.store._client()
        for selected_ttl, other_ttl in ((300, 3600), (3600, 300)):
            with self.subTest(selected_ttl=selected_ttl):
                namespace = f"{self.namespace}:renew:{selected_ttl}"
                selected = plan(list(range(9)), namespace=namespace, ttl=selected_ttl)
                other = plan(list(range(9)), namespace=namespace, ttl=other_ttl)
                await self.store.account(selected)
                await self.store.account(other)
                for key in (*selected.page_keys, *other.page_keys):
                    await client.expire(key, 5)
                following = plan(
                    list(range(9)),
                    namespace=namespace,
                    ttl=selected_ttl,
                )
                self.assertEqual(
                    await self.store.account(following), cache.CacheUsage(8, 0, 0, 1)
                )
                for key in selected.page_keys:
                    self.assertGreaterEqual(await client.ttl(key), selected_ttl - 2)
                for key in other.page_keys:
                    self.assertTrue(0 < await client.ttl(key) <= 5)

    async def test_expired_pages_are_created_again(self):
        first = self.make_plan()
        await self.store.account(first)
        for key in first.page_keys:
            await self.store._client().pexpire(key, 1)
        await asyncio.sleep(0.02)
        self.assertEqual(
            await self.store.account(self.make_plan()),
            cache.CacheUsage(0, 8, 0, 1),
        )

    async def test_only_longest_continuous_prefix_is_read(self):
        initial = self.make_plan(13)
        await self.store.account(initial)
        await self.store._client().delete(initial.page_keys[1])
        self.assertEqual(
            await self.store.account(self.make_plan(13)),
            cache.CacheUsage(4, 8, 0, 1),
        )

    async def test_concurrent_requests_have_one_creator(self):
        results = await asyncio.gather(
            *(self.store.account(self.make_plan()) for _ in range(20))
        )
        self.assertEqual(sum(result.creation_5m_tokens for result in results), 8)
        self.assertEqual(sum(result.read_tokens for result in results), 19 * 8)
        self.assertTrue(all(result.normal_input_tokens == 1 for result in results))

    async def test_invalid_key_type_is_checked_before_any_page_mutation(self):
        original = self.make_plan()
        client = self.store._client()
        await client.hset(original.page_keys[1], "corrupt", "value")
        with self.assertRaises(cache.CacheAccountingError):
            await self.store.account(original)
        self.assertEqual(await client.exists(original.page_keys[0]), 0)


if __name__ == "__main__":
    unittest.main()
