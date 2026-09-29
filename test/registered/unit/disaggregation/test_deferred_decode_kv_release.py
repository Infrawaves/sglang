"""Unit tests for the deferred decode-side KV release mechanism.

When a decode request is aborted while its prefill->decode KV transfer may still
be in flight, the decode side holds its KV pages / req-slot instead of freeing
them immediately (which could let the still-in-flight write land on pages already
reused by another request). The pages are released once every prefill rank acks
that its transfer drained (CommonKVManager.is_abort_release_safe). A timeout
must never permit reuse of a destination whose writers have not drained.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.disaggregation import decode as decode_mod
from sglang.srt.disaggregation.common.conn import CommonKVManager
from sglang.srt.disaggregation.decode import DecodeTransferQueue
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _make_manager():
    """A bare CommonKVManager carrying only the deferred-ack state the helpers
    touch (avoids the heavy real __init__)."""
    mgr = CommonKVManager.__new__(CommonKVManager)
    mgr._deferred_abort_ack_tracker = {}
    mgr._deferred_abort_tokens = {}
    mgr._deferred_abort_expected = {}
    return mgr


class TestAbortAckAggregation(CustomTestCase):
    def test_release_safe_only_after_all_required_ranks_ack(self):
        mgr = _make_manager()
        room = 100
        mgr.register_deferred_abort_room(room)
        self.assertFalse(mgr.is_abort_release_safe(room, required_acks=2))

        mgr.note_abort_ack(room, 0)
        self.assertFalse(mgr.is_abort_release_safe(room, required_acks=2))

        mgr.note_abort_ack(room, 1)
        self.assertTrue(mgr.is_abort_release_safe(room, required_acks=2))

    def test_duplicate_rank_ack_does_not_over_count(self):
        mgr = _make_manager()
        room = 101
        mgr.register_deferred_abort_room(room)
        mgr.note_abort_ack(room, 0)
        mgr.note_abort_ack(room, 0)  # same rank twice
        # Two acks arrived but from one rank: not safe for a 2-rank prefill.
        self.assertFalse(mgr.is_abort_release_safe(room, required_acks=2))

    def test_single_rank_fast_path(self):
        mgr = _make_manager()
        room = 102
        mgr.register_deferred_abort_room(room)
        mgr.note_abort_ack(room, 0)
        self.assertTrue(mgr.is_abort_release_safe(room, required_acks=1))

    def test_clear_deferred_abort_state(self):
        mgr = _make_manager()
        room = 103
        mgr.register_deferred_abort_room(room)
        mgr.note_abort_ack(room, 0)
        mgr.clear_deferred_abort_state(room)
        self.assertNotIn(room, mgr._deferred_abort_ack_tracker)
        self.assertFalse(mgr.is_abort_release_safe(room, required_acks=1))

    def test_ack_before_register_is_dropped(self):
        # An ack for a room that isn't actively held must not be recorded (it
        # would otherwise pollute a later request reusing the same room).
        mgr = _make_manager()
        room = 104
        mgr.note_abort_ack(room, 0)  # no register yet
        self.assertNotIn(room, mgr._deferred_abort_ack_tracker)
        self.assertFalse(mgr.is_abort_release_safe(room, required_acks=1))

    def test_late_ack_after_release_does_not_pollute_reused_room(self):
        # Regression for bootstrap_room reuse: req A (room R) releases, then a
        # late ack from A arrives, then req B reuses room R. B must start from a
        # clean slate and not inherit A's ack (which would release B early while
        # its transfer is still in flight -> KV corruption).
        mgr = _make_manager()
        room = 105

        # Req A's accounting was cleared after its destination was retired.
        mgr.register_deferred_abort_room(room)
        mgr.note_abort_ack(room, 0)
        mgr.clear_deferred_abort_state(room)

        # Late ack from A's other rank arrives after release -> dropped.
        mgr.note_abort_ack(room, 1)
        self.assertNotIn(room, mgr._deferred_abort_ack_tracker)

        # Req B reuses room R.
        mgr.register_deferred_abort_room(room)
        # Only B's rank-0 has acked so far; a 2-rank prefill is NOT safe yet.
        mgr.note_abort_ack(room, 0)
        self.assertFalse(mgr.is_abort_release_safe(room, required_acks=2))
        mgr.note_abort_ack(room, 1)
        self.assertTrue(mgr.is_abort_release_safe(room, required_acks=2))

    def test_duplicate_register_preserves_acks(self):
        mgr = _make_manager()
        room = 106
        mgr.register_deferred_abort_room(room)
        mgr.note_abort_ack(room, 0)
        mgr.note_abort_ack(room, 1)
        self.assertTrue(mgr.is_abort_release_safe(room, required_acks=2))
        # Duplicate cancellation must not discard an ACK already received.
        mgr.register_deferred_abort_room(room)
        self.assertTrue(mgr.is_abort_release_safe(room, required_acks=2))

    def test_mooncake_rejects_stale_and_unexpected_writer_acks(self):
        mgr = _make_manager()
        mgr.requires_transfer_drain = True
        room = 107
        mgr.register_deferred_abort_room(room, token="current", expected_ranks={0, 1})
        mgr.note_abort_ack(room, 0, token="previous")
        mgr.note_abort_ack(room, 9, token="current")
        mgr.note_abort_ack(room, 1)
        self.assertFalse(mgr.is_abort_release_safe(room, required_acks=2))
        self.assertEqual(mgr._deferred_abort_ack_tracker[room], set())
        mgr.note_abort_ack(room, 0, token="current")
        mgr.register_deferred_abort_room(room, token="current", expected_ranks={0, 1})
        mgr.note_abort_ack(room, 1, token="current")
        self.assertTrue(mgr.is_abort_release_safe(room, required_acks=2))


class _FakeIdxAllocator:
    def __init__(self):
        self.freed = []

    def free(self, idx):
        self.freed.append(idx)


def _make_queue(timeout=30.0):
    q = DecodeTransferQueue.__new__(DecodeTransferQueue)
    q.queue = []
    q._release_tp_size = 1
    q._deferred_release_error = None
    q._deferred_releases = []
    q._failed_deferred_releases = []
    q.deferred_kv_release_timeout = timeout
    q.enable_staging = False
    q.staging_handler = None
    q.tree_cache = object()
    q.metadata_buffers = SimpleNamespace(bootstrap_room={})
    q.req_to_metadata_buffer_idx_allocator = _FakeIdxAllocator()
    q.scheduler = SimpleNamespace(enable_hisparse=False)
    return q


def _make_decode_req(room, idx, mgr, n_prefill_ranks=1):
    receiver = SimpleNamespace(
        kv_mgr=mgr,
        # One entry per prefill rank the decode notified of the abort; its length
        # is the required drain-ack count (see DecodeTransferQueue._defer_release).
        bootstrap_infos=[{"rank": r} for r in range(n_prefill_ranks)],
        clear=lambda: None,
        retry_abort=Mock(),
    )
    return SimpleNamespace(
        req=SimpleNamespace(rid=f"req-{room}", bootstrap_room=room),
        kv_receiver=receiver,
        metadata_buffer_index=idx,
        prefix_match=None,
        hicache_restored_node=None,
        hicache_load_consumer_index=-1,
        _chunk_events=[],
    )


class TestResolveDeferredReleases(CustomTestCase):
    def test_cleanup_failure_keeps_pool_held_without_retrying_partial_free(self):
        mgr = _make_manager()
        q = _make_queue()
        failed = _make_decode_req(805, 5, mgr, n_prefill_ranks=1)
        healthy = _make_decode_req(806, 6, mgr, n_prefill_ranks=1)
        for req in (failed, healthy):
            room = req.req.bootstrap_room
            mgr.register_deferred_abort_room(room)
            mgr.note_abort_ack(room, 0)
            q._defer_release(req)
        q.enable_staging = True
        q.staging_handler = SimpleNamespace(
            is_staging_room=lambda room: room == 805,
            unregister_decode_req=Mock(side_effect=RuntimeError("scatter not drained")),
        )
        with patch.object(decode_mod, "release_kv_cache") as release:
            with self.assertRaisesRegex(RuntimeError, "failed buffer cleanup"):
                q.resolve_deferred_releases()
            with self.assertRaisesRegex(RuntimeError, "prior cleanup failure"):
                q.resolve_deferred_releases()
        release.assert_called_once_with(healthy.req, q.tree_cache, is_insert=False)
        q.staging_handler.unregister_decode_req.assert_called_once_with(805)
        self.assertEqual(q._failed_deferred_releases, [failed])
        self.assertTrue(q.has_pending_deferred_releases())
        with self.assertRaisesRegex(RuntimeError, "quarantined"):
            q.release_memory_occupation()

    def test_noop_when_nothing_deferred(self):
        q = _make_queue()
        with patch.object(decode_mod, "release_kv_cache") as rel:
            q.resolve_deferred_releases()
        rel.assert_not_called()

    def test_holds_until_drained_then_releases(self):
        mgr = _make_manager()
        room, idx = 200, 7
        q = _make_queue()
        dreq = _make_decode_req(room, idx, mgr, n_prefill_ranks=2)
        # In production the room is armed in abort_request when the ABORT is
        # sent, before the scheduler defers here.
        mgr.register_deferred_abort_room(room)
        q._defer_release(dreq)

        with patch.object(decode_mod, "release_kv_cache") as rel:
            # Not yet acked -> held, not released.
            q.resolve_deferred_releases()
            rel.assert_not_called()
            self.assertEqual(len(q._deferred_releases), 1)

            # One of two ranks acked -> still held.
            mgr.note_abort_ack(room, 0)
            q.resolve_deferred_releases()
            rel.assert_not_called()
            self.assertEqual(len(q._deferred_releases), 1)

            # Both ranks acked -> released exactly once.
            mgr.note_abort_ack(room, 1)
            q.resolve_deferred_releases()
            rel.assert_called_once_with(dreq.req, q.tree_cache, is_insert=False)

        # Held state fully cleaned up.
        self.assertEqual(q._deferred_releases, [])
        self.assertEqual(q.req_to_metadata_buffer_idx_allocator.freed, [idx])
        self.assertEqual(q.metadata_buffers.bootstrap_room[idx], 0)
        self.assertNotIn(room, mgr._deferred_abort_ack_tracker)
        self.assertIsNone(dreq.kv_receiver)

    def test_timeout_keeps_all_destinations_reserved_and_retries_abort(self):
        mgr = _make_manager()
        room, idx = 300, 3
        q = _make_queue(timeout=30.0)
        dreq = _make_decode_req(room, idx, mgr, n_prefill_ranks=1)
        # Force an already-expired deadline (no ack will ever arrive).
        q._deferred_releases.append((dreq, float("-inf"), idx, 1))

        with patch.object(decode_mod, "release_kv_cache") as rel:
            q.resolve_deferred_releases()
            rel.assert_not_called()

        self.assertEqual(len(q._deferred_releases), 1)
        self.assertEqual(q.req_to_metadata_buffer_idx_allocator.freed, [])
        self.assertIsNotNone(dreq.kv_receiver)
        dreq.kv_receiver.retry_abort.assert_called_once()

    def test_partial_cleanup_stops_reentry_without_retrying_frees(self):
        # Finish this common release plan, then stop scheduling if any free
        # failed. Re-entering must not retry a potentially partial free.
        mgr = _make_manager()
        q = _make_queue()
        good = _make_decode_req(700, 1, mgr)
        bad = _make_decode_req(701, 2, mgr)
        # Both remotely drained -> both selected for release.
        for room in (700, 701):
            mgr.register_deferred_abort_room(room)
            mgr.note_abort_ack(room, 0)
        q._deferred_releases.append((bad, float("-inf"), 2, 1))
        q._deferred_releases.append((good, float("-inf"), 1, 1))

        calls = []

        def fake_release(req, tree_cache, is_insert):
            calls.append(req)
            if req is bad.req:
                raise RuntimeError("boom")

        with patch.object(decode_mod, "release_kv_cache", side_effect=fake_release):
            with self.assertRaisesRegex(RuntimeError, "failed buffer cleanup"):
                q.resolve_deferred_releases()
            self.assertEqual(calls, [good.req, bad.req])
            self.assertEqual(q._deferred_releases, [])
            self.assertEqual(q._failed_deferred_releases, [bad])
            self.assertTrue(q.has_pending_deferred_releases())
            with self.assertRaisesRegex(RuntimeError, "prior cleanup failure"):
                q.resolve_deferred_releases()
            self.assertEqual(calls, [good.req, bad.req])

    def test_defer_release_records_deadline_and_idx(self):
        mgr = _make_manager()
        q = _make_queue(timeout=12.5)
        dreq = _make_decode_req(room=400, idx=9, mgr=mgr)
        q._defer_release(dreq)
        self.assertEqual(len(q._deferred_releases), 1)
        held_req, deadline, held_idx, required = q._deferred_releases[0]
        self.assertIs(held_req, dreq)
        self.assertEqual(held_idx, 9)
        self.assertIsInstance(deadline, float)

    def test_unknown_writer_set_never_counts_as_zero_writers(self):
        mgr = _make_manager()
        q = _make_queue()
        dreq = _make_decode_req(800, 4, mgr, n_prefill_ranks=0)
        q._defer_release(dreq)
        self.assertIsNone(q._deferred_releases[0][3])
        with patch.object(decode_mod, "release_kv_cache") as rel:
            q.resolve_deferred_releases()
        rel.assert_not_called()
        self.assertIsNotNone(dreq.kv_receiver)

    def test_repeated_defer_does_not_duplicate_release(self):
        mgr = _make_manager()
        q = _make_queue()
        dreq = _make_decode_req(801, 4, mgr)
        q._defer_release(dreq)
        q._defer_release(dreq)
        self.assertEqual(len(q._deferred_releases), 1)

    def test_local_restore_must_drain_before_cache_lock_and_kv_release(self):
        mgr = _make_manager()
        q = _make_queue()
        dreq = _make_decode_req(802, 4, mgr)
        dreq.hicache_restored_node = object()
        dreq.hicache_restore_lock_receipt = object()
        dreq.hicache_load_consumer_index = 3
        q.tree_cache = SimpleNamespace(
            is_load_back_event_done=Mock(return_value=False),
            dec_lock_ref=Mock(),
        )
        mgr.register_deferred_abort_room(802)
        mgr.note_abort_ack(802, 0)
        q._defer_release(dreq)
        with patch.object(decode_mod, "release_kv_cache") as rel:
            q.resolve_deferred_releases()
            rel.assert_not_called()
            q.tree_cache.dec_lock_ref.assert_not_called()
            self.assertEqual(q.req_to_metadata_buffer_idx_allocator.freed, [])
            q.tree_cache.is_load_back_event_done.return_value = True
            q.resolve_deferred_releases()
            rel.assert_called_once_with(dreq.req, q.tree_cache, is_insert=False)
        self.assertIsNone(dreq.hicache_restored_node)
        self.assertEqual(q.req_to_metadata_buffer_idx_allocator.freed, [4])

    def test_staging_and_hisparse_buffers_release_only_after_all_writers_ack(self):
        mgr = _make_manager()
        q = _make_queue()
        dreq = _make_decode_req(803, 4, mgr, n_prefill_ranks=2)
        events = []
        q.enable_staging = True
        q.staging_handler = SimpleNamespace(
            is_staging_room=lambda room: True,
            unregister_decode_req=lambda room: events.append("staging-drained"),
        )
        q.scheduler = SimpleNamespace(
            enable_hisparse=True,
            hisparse_coordinator=SimpleNamespace(
                request_finished=lambda req: events.append("hisparse-released")
            ),
        )
        mgr.register_deferred_abort_room(803)
        mgr.note_abort_ack(803, 0)
        q._defer_release(dreq)
        with patch.object(
            decode_mod,
            "release_kv_cache",
            side_effect=lambda *a, **kw: events.append("kv-released"),
        ):
            q.resolve_deferred_releases()
            self.assertEqual(events, [])
            mgr.note_abort_ack(803, 1)
            q.resolve_deferred_releases()
        self.assertEqual(
            events, ["staging-drained", "hisparse-released", "kv-released"]
        )

    def test_memory_offload_cannot_drop_quarantined_or_active_transfers(self):
        mgr = _make_manager()
        q = _make_queue()
        dreq = _make_decode_req(804, 4, mgr)
        q._defer_release(dreq)
        with self.assertRaisesRegex(RuntimeError, "quarantined"):
            q.release_memory_occupation()
        self.assertEqual(len(q._deferred_releases), 1)
        q._deferred_releases.clear()
        q.queue.append(dreq)
        with self.assertRaisesRegex(RuntimeError, "KV transfers"):
            q.release_memory_occupation()
        self.assertEqual(q.queue, [dreq])


if __name__ == "__main__":
    unittest.main()
