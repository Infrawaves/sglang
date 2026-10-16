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
from unittest.mock import AsyncMock, Mock, patch

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
    def test_fixed_256_block_boundaries(self):
        self.assertEqual(cache.CACHE_BLOCK_SIZE, 256)
        for length, complete_blocks, tail in [
            (0, 0, 0),
            (1, 0, 1),
            (255, 0, 255),
            (256, 1, 0),
            (257, 1, 1),
            (511, 1, 255),
            (512, 2, 0),
            (513, 2, 1),
        ]:
            with self.subTest(length=length):
                result = plan(list(range(length)), page_size=cache.CACHE_BLOCK_SIZE)
                self.assertEqual(len(result.page_keys), complete_blocks)
                self.assertEqual(result.prompt_tokens, length)
                self.assertEqual(result.prompt_tokens % result.page_size, tail)

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
    def setUp(self):
        self.logger = Mock()
        logger_patch = patch.object(cache, "logger", self.logger)
        logger_patch.start()
        self.addCleanup(logger_patch.stop)

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
        self.logger.info.assert_not_called()
        self.logger.warning.assert_not_called()
        await store.aclose()

    async def test_eval_start_and_success_logs_include_rid_and_confirmed_usage(self):
        store = cache.RedisPageCacheStore("redis://unused", timeout_seconds=0.2)
        store._redis = AsyncMock()
        store._redis.eval.return_value = [4, 4, 0, 1]
        result = await store.account(plan(list(range(9))), rid="request-redis-1")
        self.assertEqual(result, cache.CacheUsage(4, 4, 0, 1))
        store._redis.eval.assert_awaited_once()
        self.assertEqual(self.logger.info.call_count, 2)
        start, complete = [
            call.args[0] % call.args[1:] for call in self.logger.info.call_args_list
        ]
        self.assertEqual(
            start,
            "Cache Redis EVAL start rid=request-redis-1 page_count=2 ttl=300 "
            "timeout_ms=200.000",
        )
        self.assertRegex(
            complete,
            r"^Cache Redis EVAL complete rid=request-redis-1 result=success "
            r"page_count=2 ttl=300 read=4 write_5m=4 write_1h=0 normal=1 "
            r"elapsed_ms=\d+\.\d{3} timeout_ms=200\.000$",
        )
        self.logger.warning.assert_not_called()
        await store.aclose()

    async def test_eval_failure_logs_classification_without_sensitive_contents(self):
        errors = [(TimeoutError, "timeout"), (ConnectionError, "error")]
        try:
            from redis.exceptions import TimeoutError as RedisTimeoutError
        except ImportError:
            pass
        else:
            errors.append((RedisTimeoutError, "timeout"))
        original = plan(list(range(9)))
        secrets = (
            "redis://:secret-password@private-redis:6379/0",
            "secret-api-key",
            "secret-input-content",
            original.page_keys[0],
        )
        for error_type, classification in errors:
            with self.subTest(error_type=error_type):
                self.logger.reset_mock()
                store = cache.RedisPageCacheStore("redis://unused", timeout_seconds=0.2)
                store._redis = AsyncMock()
                store._redis.eval.side_effect = error_type(" ".join(secrets))
                with self.assertRaises(cache.CacheAccountingError):
                    await store.account(original, rid="request-redis-2")
                self.logger.info.assert_called_once()
                self.logger.warning.assert_called_once()
                warning = self.logger.warning.call_args
                message = warning.args[0] % warning.args[1:]
                self.assertIn(
                    f"rid=request-redis-2 result={classification} "
                    f"error_type={error_type.__name__}",
                    message,
                )
                self.assertIn("timeout_ms=200.000", message)
                for secret in secrets:
                    self.assertNotIn(secret, message)
                self.assertEqual(warning.kwargs, {})
                await store.aclose()

    async def test_cancellation_propagates_without_failure_or_success_log(self):
        entered = asyncio.Event()

        async def blocked_eval(*args):
            entered.set()
            await asyncio.Event().wait()

        store = cache.RedisPageCacheStore("redis://unused")
        store._redis = AsyncMock()
        store._redis.eval.side_effect = blocked_eval
        task = asyncio.create_task(store.account(plan(list(range(9))), rid="cancelled"))
        await entered.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.logger.info.assert_called_once()
        self.assertIn("EVAL start", self.logger.info.call_args.args[0])
        self.logger.warning.assert_not_called()
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
        self.assertIsInstance(
            failure.exception.__cause__, (TimeoutError, asyncio.TimeoutError)
        )
        self.logger.warning.assert_called_once()
        warning = self.logger.warning.call_args
        self.assertIn("result=timeout", warning.args[0] % warning.args[1:])
        await store.aclose()

    async def test_invalid_plan_is_rejected_before_redis(self):
        original = plan(list(range(8)))
        malformed = replace(original, page_keys=(original.page_keys[0], "wrong-slot"))
        store = cache.RedisPageCacheStore("redis://unused")
        with patch.object(
            store, "_client", side_effect=AssertionError("must not connect")
        ) as client:
            with self.assertRaises(ValueError):
                await store.account(malformed)
        client.assert_not_called()
        self.logger.info.assert_not_called()
        self.logger.warning.assert_not_called()

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

    def make_plan(self, count=9, *, ttl=300, page_size=4):
        return plan(
            list(range(count)),
            namespace=self.namespace,
            ttl=ttl,
            page_size=page_size,
        )

    async def test_fixed_256_block_accounting_and_prefix_extension(self):
        for ttl in (300, 3600):
            for length, read, written, tail in [
                (255, 0, 0, 255),
                (256, 0, 256, 0),
                (257, 256, 0, 1),
                (511, 256, 0, 255),
                (512, 256, 256, 0),
                (513, 512, 0, 1),
            ]:
                with self.subTest(ttl=ttl, length=length):
                    result = await self.store.account(
                        self.make_plan(
                            length, ttl=ttl, page_size=cache.CACHE_BLOCK_SIZE
                        )
                    )
                    self.assertEqual(
                        result,
                        cache.CacheUsage(
                            read,
                            written if ttl == 300 else 0,
                            written if ttl == 3600 else 0,
                            tail,
                        ),
                    )
                    self.assertEqual(sum(result.to_dict().values()), length)

    async def test_real_eval_logs_cold_and_hit_results_with_request_id(self):
        original = self.make_plan(513, ttl=3600, page_size=cache.CACHE_BLOCK_SIZE)
        with patch.object(cache, "logger") as logger:
            cold = await self.store.account(original, rid="real-cold")
            hit = await self.store.account(original, rid="real-hit")
        self.assertEqual(cold, cache.CacheUsage(0, 0, 512, 1))
        self.assertEqual(hit, cache.CacheUsage(512, 0, 0, 1))
        self.assertEqual(logger.info.call_count, 4)
        messages = [call.args[0] % call.args[1:] for call in logger.info.call_args_list]
        self.assertIn("EVAL start rid=real-cold page_count=2 ttl=3600", messages[0])
        self.assertIn("rid=real-cold result=success", messages[1])
        self.assertIn("read=0 write_5m=0 write_1h=512 normal=1", messages[1])
        self.assertIn("rid=real-hit result=success", messages[3])
        self.assertIn("read=512 write_5m=0 write_1h=0 normal=1", messages[3])
        logger.warning.assert_not_called()

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
                first = plan(
                    list(range(513)),
                    namespace=namespace,
                    ttl=first_ttl,
                    page_size=cache.CACHE_BLOCK_SIZE,
                )
                second = plan(
                    list(range(513)),
                    namespace=namespace,
                    ttl=second_ttl,
                    page_size=cache.CACHE_BLOCK_SIZE,
                )
                expected_first = cache.CacheUsage(
                    0,
                    512 if first_ttl == 300 else 0,
                    512 if first_ttl == 3600 else 0,
                    1,
                )
                expected_second = cache.CacheUsage(
                    0,
                    512 if second_ttl == 300 else 0,
                    512 if second_ttl == 3600 else 0,
                    1,
                )
                self.assertEqual(await self.store.account(first), expected_first)
                self.assertEqual(await self.store.account(second), expected_second)
                for ttl in (300, 3600):
                    following = plan(
                        list(range(513)),
                        namespace=namespace,
                        ttl=ttl,
                        page_size=cache.CACHE_BLOCK_SIZE,
                    )
                    self.assertEqual(
                        await self.store.account(following),
                        cache.CacheUsage(512, 0, 0, 1),
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

    async def test_fixed_256_block_failure_does_not_return_usage_or_mutate_pages(self):
        original = self.make_plan(769, page_size=cache.CACHE_BLOCK_SIZE)
        client = self.store._client()
        await client.set(original.page_keys[0], "1", ex=5)
        await client.hset(original.page_keys[1], "corrupt", "value")
        with self.assertRaises(cache.CacheAccountingError):
            await self.store.account(original)
        self.assertTrue(0 < await client.ttl(original.page_keys[0]) <= 5)
        self.assertEqual(await client.hget(original.page_keys[1], "corrupt"), b"value")
        self.assertEqual(await client.exists(original.page_keys[2]), 0)


if __name__ == "__main__":
    unittest.main()
