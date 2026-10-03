"""Unit tests for shortest-prefill-first ordering (pure logic, no GPU)."""

import unittest
from types import SimpleNamespace

from sglang.srt.managers.prefill_order import (
    expected_cached_len,
    prefill_work,
    shortest_prefill_key,
    shortest_prefill_order,
    sort_shortest_prefill_first,
    waited_tokens,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _req(rid, total, matched=0, host=0, arrival=0, prefill_arrival=None):
    return SimpleNamespace(
        rid=rid,
        origin_input_ids=[0] * total,
        output_ids=[],
        prefix_indices=[0] * matched,
        host_hit_length=host,
        arrival_processed_tokens=arrival,
        prefill_arrival_processed_tokens=prefill_arrival,
    )


def _no_prefetch(rid):
    return 0


class TestPrefillWork(CustomTestCase):
    def test_device_and_host_hits_count_as_cached(self):
        r = _req("a", 1000, matched=300, host=500)
        self.assertEqual(expected_cached_len(r), 800)
        self.assertEqual(prefill_work(r), 200)

    def test_l3_prefetch_extends_cached_prefix(self):
        r = _req("a", 1000, matched=64)
        self.assertEqual(expected_cached_len(r, prefetched_prefix_len=960), 960)
        self.assertEqual(prefill_work(r, prefetched_prefix_len=960), 40)

    def test_prefetch_shorter_than_match_is_ignored(self):
        r = _req("a", 1000, matched=900)
        self.assertEqual(expected_cached_len(r, prefetched_prefix_len=500), 900)

    def test_work_is_at_least_one(self):
        r = _req("a", 1000, matched=1000)
        self.assertEqual(prefill_work(r, prefetched_prefix_len=5000), 1)

    def test_waited_counts_from_first_queue_entry(self):
        r = _req("a", 10, arrival=900, prefill_arrival=100)
        self.assertEqual(waited_tokens(r, processed_tokens=1000), 900)
        r2 = _req("b", 10, arrival=900)
        self.assertEqual(waited_tokens(r2, processed_tokens=1000), 100)


class TestShortestPrefillOrder(CustomTestCase):
    def test_l3_warm_request_goes_before_cold(self):
        # Same prompt length; only "warm" had its prefix prefetched from L3.
        cold = _req("cold", 100_000, matched=64)
        warm = _req("warm", 100_000, matched=0)
        prefetched = {"warm": 99_000}.get
        order = shortest_prefill_order(
            [cold, warm],
            processed_tokens=0,
            aging_tokens=0,
            prefetched_prefix_len=lambda rid: prefetched(rid, 0),
        )
        self.assertEqual(order, [1, 0])

    def test_without_l3_hint_warm_looks_cold(self):
        cold = _req("cold", 100_000, matched=64)
        warm = _req("warm", 100_000, matched=0)
        order = shortest_prefill_order(
            [cold, warm],
            processed_tokens=0,
            aging_tokens=0,
            prefetched_prefix_len=_no_prefetch,
        )
        self.assertEqual(order, [0, 1])

    def test_aged_requests_first_longest_waiting_first(self):
        big_old = _req("big_old", 600_000, prefill_arrival=0)
        mid_old = _req("mid_old", 60_000, prefill_arrival=500)
        small_new = _req("small_new", 1_000, prefill_arrival=3_000)
        queue = [small_new, mid_old, big_old]
        sort_shortest_prefill_first(
            queue,
            processed_tokens=3_000,
            aging_tokens=2_000,
            prefetched_prefix_len=_no_prefetch,
        )
        # big_old waited 3000, mid_old 2500 (both aged); small_new waited 0.
        self.assertEqual([r.rid for r in queue], ["big_old", "mid_old", "small_new"])

    def test_aging_disabled(self):
        big_old = _req("big_old", 600_000, prefill_arrival=0)
        small_new = _req("small_new", 1_000, prefill_arrival=10**9)
        queue = [big_old, small_new]
        sort_shortest_prefill_first(
            queue,
            processed_tokens=10**9,
            aging_tokens=0,
            prefetched_prefix_len=_no_prefetch,
        )
        self.assertEqual([r.rid for r in queue], ["small_new", "big_old"])

    def test_deprioritized_after_their_tier(self):
        a = _req("a", 100)
        b = _req("b", 10)
        queue = [a, b]
        sort_shortest_prefill_first(
            queue,
            processed_tokens=0,
            aging_tokens=0,
            prefetched_prefix_len=_no_prefetch,
            deprioritized={"b"},
        )
        self.assertEqual([r.rid for r in queue], ["a", "b"])

    def test_equal_work_prefers_longer_wait_then_arrival(self):
        x = _req("x", 100, prefill_arrival=50)
        y = _req("y", 100, prefill_arrival=10)
        z = _req("z", 100, prefill_arrival=10)
        order = shortest_prefill_order(
            [x, y, z],
            processed_tokens=100,
            aging_tokens=0,
            prefetched_prefix_len=_no_prefetch,
        )
        self.assertEqual(order, [1, 2, 0])

    def test_key_tiers(self):
        r = _req("r", 10, prefill_arrival=0)
        aged = shortest_prefill_key(r, processed_tokens=100, aging_tokens=50)
        fresh = shortest_prefill_key(r, processed_tokens=10, aging_tokens=50)
        self.assertLess(aged, fresh)


if __name__ == "__main__":
    unittest.main()
