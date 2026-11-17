import logging
import os
import unittest
from types import SimpleNamespace
from unittest import mock

from sglang.srt.mem_cache import kda_l3_stats
from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")


class TestClassify(unittest.TestCase):
    def test_precedence(self):
        with mock.patch.dict(os.environ, {"SGLANG_KDA_L3_STATS_HEAD_TOKENS": "16384"}):
            # tail wins over branch and head
            self.assertEqual(kda_l3_stats.classify(16384, True, 16384), "tail")
            self.assertEqual(kda_l3_stats.classify(16384, False, 16384), "branch")
            self.assertEqual(kda_l3_stats.classify(16384, False, None), "head")
            self.assertEqual(kda_l3_stats.classify(15872, False, 20480), "head")
            self.assertEqual(kda_l3_stats.classify(16896, False, None), "middle")
            self.assertEqual(kda_l3_stats.classify(32768, False, 20480), "middle")


class TestTracker(unittest.TestCase):
    def _tracker(self):
        with mock.patch.object(kda_l3_stats, "_is_logging_rank", return_value=True):
            return kda_l3_stats.KdaL3StatsTracker()

    def test_write_logs_class_once(self):
        t = self._tracker()
        t.on_insert(
            7,
            rid="r1",
            position=32768,
            prompt_len=58000,
            is_request_end=False,
            branching_seqlen=None,
        )
        with self.assertLogs(kda_l3_stats.logger, level=logging.INFO) as cm:
            t.on_storage_write(7, "abc", 32768)
        self.assertIn("KDA_L3_WRITE key=abc cls=middle pos=32768", cm.output[0])
        # popped: a second write of the same node logs nothing
        with self.assertNoLogs(kda_l3_stats.logger, level=logging.INFO):
            t.on_storage_write(7, "abc", 32768)

    def test_depth_mismatch_is_dropped(self):
        t = self._tracker()
        t.on_insert(
            7,
            rid="r1",
            position=16384,
            prompt_len=58000,
            is_request_end=False,
            branching_seqlen=None,
        )
        with self.assertNoLogs(kda_l3_stats.logger, level=logging.INFO):
            t.on_storage_write(7, "abc", 20480)

    def test_pending_is_bounded(self):
        t = self._tracker()
        with mock.patch.object(kda_l3_stats, "_MAX_PENDING", 4):
            for i in range(10):
                t.on_insert(
                    i,
                    rid="r",
                    position=512,
                    prompt_len=1024,
                    is_request_end=True,
                    branching_seqlen=None,
                )
        self.assertEqual(list(t._pending), [6, 7, 8, 9])

    def test_non_logging_rank_records_nothing(self):
        with mock.patch.object(kda_l3_stats, "_is_logging_rank", return_value=False):
            t = kda_l3_stats.KdaL3StatsTracker()
        t.on_insert(
            1,
            rid="r",
            position=512,
            prompt_len=1024,
            is_request_end=True,
            branching_seqlen=None,
        )
        self.assertEqual(len(t._pending), 0)


class TestWriteGate(unittest.TestCase):
    """The O(depth) prefix walk runs only on the logging rank for recorded nodes."""

    def _cache(self, log_rank):
        from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
        from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

        with mock.patch.object(kda_l3_stats, "_is_logging_rank", return_value=log_rank):
            tracker = kda_l3_stats.KdaL3StatsTracker()
        cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
        cache.kda_l3_stats = tracker
        cache.tree_core = SimpleNamespace(page_size=512)
        cache.get_prefix_hash_values = mock.Mock(return_value=["p"] * 31)
        spec = SimpleNamespace(
            hash_value=["h"],
            comp_xfers={ComponentType.MAMBA: [SimpleNamespace(keys=["k32"])]},
        )
        return cache, tracker, spec

    def _record(self, tracker, node_id, position):
        tracker._pending[node_id] = ("middle", "r", position, 58000)

    def test_non_logging_rank_skips_walk(self):
        cache, tracker, spec = self._cache(log_rank=False)
        self._record(tracker, 7, 16384)  # even a stray entry must not walk
        cache._record_kda_l3_write(7, spec)
        cache.get_prefix_hash_values.assert_not_called()

    def test_unrecorded_node_skips_walk(self):
        cache, tracker, spec = self._cache(log_rank=True)
        cache._record_kda_l3_write(7, spec)
        cache.get_prefix_hash_values.assert_not_called()

    def test_recorded_node_walks_and_logs(self):
        cache, tracker, spec = self._cache(log_rank=True)
        self._record(tracker, 7, 32 * 512)
        with self.assertLogs(kda_l3_stats.logger, level=logging.INFO) as cm:
            cache._record_kda_l3_write(7, spec)
        cache.get_prefix_hash_values.assert_called_once_with(7)
        self.assertIn("key=k32 cls=middle pos=16384", cm.output[0])


class TestReadLog(unittest.TestCase):
    """start_page=4, page=512: positions are 2048 + pages * 512."""

    def _op(self, kv_pages, mamba=None, indexer=None, pools=("mamba", "indexer")):
        hits = {PoolName.KV: kv_pages}
        if mamba:
            hits[PoolName.MAMBA] = mamba
        if indexer:
            hits[PoolName.INDEXER] = indexer
        names = {"mamba": PoolName.MAMBA, "indexer": PoolName.INDEXER}
        return SimpleNamespace(
            query_pool_hit_pages=hits,
            pool_transfers=[SimpleNamespace(name=names[p]) for p in pools],
            all_hash_values=[f"h{i}" for i in range(1, 11)],
            start_page=4,
            assume_stored=False,
            request_id="rid-x",
        )

    def _line(self, op, final_tokens):
        with mock.patch.object(kda_l3_stats, "_is_logging_rank", return_value=True):
            with self.assertLogs(kda_l3_stats.logger, level=logging.INFO) as cm:
                kda_l3_stats.log_hit_query(op, final_tokens, 512)
        return cm.output[0]

    def test_full_hit(self):
        line = self._line(self._op(8, mamba=8, indexer=8), 8 * 512)
        self.assertIn(
            "key=h8 start=2048 hit_end=6144 kv_end=6144 mamba_end=6144 "
            "indexer_end=6144 limited=0 limited_by=-",
            line,
        )
        self.assertIn("rid=rid-x", line)

    def test_cut_by_mamba(self):
        line = self._line(self._op(8, mamba=3, indexer=8), 3 * 512)
        self.assertIn("hit_end=3584 kv_end=6144 mamba_end=3584 indexer_end=6144", line)
        self.assertIn("limited=1 limited_by=mamba", line)

    def test_cut_by_indexer(self):
        # Indexer page 6 is missing; the state before it sits at page 5.
        line = self._line(self._op(8, mamba=8, indexer=5), 5 * 512)
        self.assertIn("mamba_end=6144 indexer_end=4608 limited=1 limited_by=indexer", line)

    def test_cut_by_both(self):
        line = self._line(self._op(8, mamba=3, indexer=5), 3 * 512)
        self.assertIn("limited_by=mamba+indexer", line)

    def test_pool_without_stop_point_ends_at_start(self):
        # A queried pool with boundary 0 is absent from the verdict dict.
        line = self._line(self._op(8, indexer=8), 0)
        self.assertIn("key=- start=2048 hit_end=2048 kv_end=6144 mamba_end=2048", line)
        self.assertIn("limited_by=mamba", line)

    def test_unqueried_pool_is_na(self):
        line = self._line(self._op(8, mamba=3, pools=("mamba",)), 3 * 512)
        self.assertIn("mamba_end=3584 indexer_end=na limited=1 limited_by=mamba", line)

    def test_cut_by_other_pool(self):
        line = self._line(self._op(8, mamba=8, indexer=8), 5 * 512)
        self.assertIn("limited=1 limited_by=other", line)

    def test_assume_stored_has_no_verdicts(self):
        op = self._op(0)
        op.assume_stored = True
        op.query_pool_hit_pages = {}
        line = self._line(op, 6 * 512)
        self.assertIn(
            "hit_end=5120 kv_end=5120 mamba_end=na indexer_end=na limited=0 "
            "limited_by=- assume_stored=1",
            line,
        )

    def test_miss_not_logged(self):
        with mock.patch.object(kda_l3_stats, "_is_logging_rank", return_value=True):
            with self.assertNoLogs(kda_l3_stats.logger, level=logging.INFO):
                kda_l3_stats.log_hit_query(self._op(0), 0, 512)


if __name__ == "__main__":
    unittest.main()
