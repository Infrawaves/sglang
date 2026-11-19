import logging
import os
import unittest
from types import SimpleNamespace
from unittest import mock

from sglang.srt.mem_cache import kda_l3_stats
from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")


def _logging_rank(value=True):
    return mock.patch.object(kda_l3_stats, "_is_logging_rank", return_value=value)


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
    def setUp(self):
        kda_l3_stats._put_outcomes.clear()

    def _tracker(self):
        with _logging_rank():
            return kda_l3_stats.KdaL3StatsTracker()

    def _insert(self, t, node, position=32768, end=False):
        t.on_insert(
            node,
            rid="r1",
            position=position,
            prompt_len=58000,
            is_request_end=end,
            branching_seqlen=None,
        )

    def test_write_logged_at_ack_with_put_outcome(self):
        t = self._tracker()
        self._insert(t, 7)
        with self.assertNoLogs(kda_l3_stats.logger, level=logging.INFO):
            t.on_storage_write(7, "abc", 32768, operation_id=11)
        kda_l3_stats.record_put_outcome("abc", "new")
        with self.assertLogs(kda_l3_stats.logger, level=logging.INFO) as cm:
            t.on_backup_ack(11)
        self.assertIn(
            "KDA_L3_WRITE key=abc cls=middle pos=32768 prompt_len=58000 stored=new",
            cm.output[0],
        )
        # popped: a second ACK of the same operation logs nothing
        with self.assertNoLogs(kda_l3_stats.logger, level=logging.INFO):
            t.on_backup_ack(11)

    def test_existing_object_is_marked_exists(self):
        t = self._tracker()
        self._insert(t, 7)
        t.on_storage_write(7, "abc", 32768, operation_id=11)
        kda_l3_stats.record_put_outcome("abc", "exists")
        with self.assertLogs(kda_l3_stats.logger, level=logging.INFO) as cm:
            t.on_backup_ack(11)
        self.assertIn("stored=exists", cm.output[0])

    def test_missing_outcome_is_unknown(self):
        t = self._tracker()
        self._insert(t, 7)
        t.on_storage_write(7, "abc", 32768, operation_id=11)
        with self.assertLogs(kda_l3_stats.logger, level=logging.INFO) as cm:
            t.on_backup_ack(11)
        self.assertIn("stored=unknown", cm.output[0])

    def test_depth_mismatch_is_dropped(self):
        t = self._tracker()
        self._insert(t, 7, position=16384)
        t.on_storage_write(7, "abc", 20480, operation_id=11)
        with self.assertNoLogs(kda_l3_stats.logger, level=logging.INFO):
            t.on_backup_ack(11)

    def test_pending_and_issued_are_bounded(self):
        t = self._tracker()
        with mock.patch.object(kda_l3_stats, "_MAX_PENDING", 4):
            for i in range(10):
                self._insert(t, i, position=512, end=True)
            self.assertEqual(list(t._pending), [6, 7, 8, 9])
            for i in range(6, 10):
                t.on_storage_write(i, f"k{i}", 512, operation_id=100 + i)
            for i in range(10):
                self._insert(t, 20 + i, position=512, end=True)
                t.on_storage_write(20 + i, f"k{i}", 512, operation_id=200 + i)
            self.assertEqual(len(t._issued), 4)

    def test_non_logging_rank_records_nothing(self):
        with _logging_rank(False):
            t = kda_l3_stats.KdaL3StatsTracker()
        self._insert(t, 1, position=512, end=True)
        self.assertEqual(len(t._pending), 0)


class TestPutOutcome(unittest.TestCase):
    """Mooncake's set path skips objects that exist; the outcome says which."""

    def setUp(self):
        kda_l3_stats._put_outcomes.clear()

    def _record(self, exist, put):
        from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import (
            MooncakeStore,
        )

        with mock.patch.object(
            kda_l3_stats, "put_outcome_recording_active", return_value=True
        ):
            # two pages, two objects per page (temporal + conv)
            MooncakeStore._record_kda_put_outcomes(["p1", "p2"], exist, put, 2)
        return [kda_l3_stats.pop_put_outcome(k) for k in ("p1", "p2")]

    def test_new_and_exists(self):
        self.assertEqual(self._record([0, 0, 1, 1], [0, 0, 0, 0]), ["new", "exists"])

    def test_partial_and_failed(self):
        self.assertEqual(
            self._record([1, 0, 0, 0], [0, 0, -1, 0]), ["partial", "failed"]
        )

    def test_inactive_records_nothing(self):
        from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import (
            MooncakeStore,
        )

        with mock.patch.object(
            kda_l3_stats, "put_outcome_recording_active", return_value=False
        ):
            MooncakeStore._record_kda_put_outcomes(["p1"], [0, 0], [0, 0], 2)
        self.assertEqual(kda_l3_stats.pop_put_outcome("p1"), "unknown")


class TestWriteGate(unittest.TestCase):
    """The O(depth) prefix walk runs only on the logging rank for recorded nodes."""

    def _cache(self, log_rank):
        from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
        from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

        with _logging_rank(log_rank):
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
        cache._record_kda_l3_write(7, spec, operation_id=1)
        cache.get_prefix_hash_values.assert_not_called()

    def test_unrecorded_node_skips_walk(self):
        cache, tracker, spec = self._cache(log_rank=True)
        cache._record_kda_l3_write(7, spec, operation_id=1)
        cache.get_prefix_hash_values.assert_not_called()

    def test_recorded_node_walks_and_waits_for_ack(self):
        cache, tracker, spec = self._cache(log_rank=True)
        self._record(tracker, 7, 32 * 512)
        cache._record_kda_l3_write(7, spec, operation_id=1)
        cache.get_prefix_hash_values.assert_called_once_with(7)
        self.assertEqual(tracker._issued[1][:2], ("k32", "middle"))


class TestReadLog(unittest.TestCase):
    """start_page=4, page=512: positions are 2048 + pages * 512.

    Mooncake reports a pool only with a nonzero boundary and stops checking
    pools once nothing is restorable, so a missing pool has no verdict.
    """

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
        with _logging_rank():
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

    def test_no_state_in_span_has_no_verdicts(self):
        # No KDA state in the KV span: Mooncake reports neither pool (the
        # indexer is never checked), so neither end is known.
        line = self._line(self._op(8), 0)
        self.assertIn(
            "key=- start=2048 hit_end=2048 kv_end=6144 mamba_end=? indexer_end=? "
            "limited=1 limited_by=unknown",
            line,
        )

    def test_cause_only_from_pools_with_a_verdict(self):
        # A rank left the indexer out, so the MIN dropped its verdict.
        line = self._line(self._op(8, mamba=3), 3 * 512)
        self.assertIn("mamba_end=3584 indexer_end=? limited=1 limited_by=mamba", line)

    def test_unknown_when_short_cause_has_no_verdict(self):
        line = self._line(self._op(8, mamba=8), 5 * 512)
        self.assertIn("indexer_end=? limited=1 limited_by=unknown", line)

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
        with _logging_rank():
            with self.assertNoLogs(kda_l3_stats.logger, level=logging.INFO):
                kda_l3_stats.log_hit_query(self._op(0), 0, 512)


class TestLoadLog(unittest.TestCase):
    """The loaded result, which may be shorter than the query verdict."""

    def _op(self, completed, start=2048):
        return SimpleNamespace(
            completed_tokens=completed,
            storage_start=start,
            hash_value=[f"h{i}" for i in range(1, 11)],
        )

    def test_success_names_the_loaded_state(self):
        with _logging_rank():
            with self.assertLogs(kda_l3_stats.logger, level=logging.INFO) as cm:
                kda_l3_stats.log_load(self._op(3 * 512), "rid-x", 512, dropped=False)
        self.assertIn(
            "KDA_L3_LOAD key=h3 start=2048 loaded_end=3584 result=success rid=rid-x",
            cm.output[0],
        )

    def test_dropped(self):
        with _logging_rank():
            with self.assertLogs(kda_l3_stats.logger, level=logging.INFO) as cm:
                kda_l3_stats.log_load(self._op(512), "rid-x", 512, dropped=True)
        self.assertIn("result=dropped", cm.output[0])

    def test_nothing_loaded_not_logged(self):
        with _logging_rank():
            with self.assertNoLogs(kda_l3_stats.logger, level=logging.INFO):
                kda_l3_stats.log_load(self._op(0), "rid-x", 512, dropped=False)

    def test_non_logging_rank(self):
        with _logging_rank(False):
            with self.assertNoLogs(kda_l3_stats.logger, level=logging.INFO):
                kda_l3_stats.log_load(self._op(512), "rid-x", 512, dropped=False)


if __name__ == "__main__":
    unittest.main()
