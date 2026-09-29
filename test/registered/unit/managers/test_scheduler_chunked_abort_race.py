"""Tests for deferred chunked-prefill aborts."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.base import KVPoll  # noqa: E402
from sglang.srt.disaggregation.utils import DisaggregationMode  # noqa: E402
from sglang.srt.managers.io_struct import AbortReq  # noqa: E402
from sglang.srt.managers.scheduler import Scheduler  # noqa: E402

register_cpu_ci(est_time=9, suite="base-a-test-cpu")


class _FakeReq:
    """Minimal stand-in for Req: only the fields the abort paths touch."""

    def __init__(self, rid: str):
        self.rid = rid
        # Mirrors Req.kv; the abort paths read only these two predicates.
        self.kv = SimpleNamespace(holds_kv=True, holds_mamba=False)
        self.to_finish = None
        self._finished = False

    def finished(self):
        return self._finished


def _make_scheduler(pending_req, *, chunked_req, running_reqs) -> Scheduler:
    sched = Scheduler.__new__(Scheduler)
    sched.enable_chunked_prefill_round_robin = False
    sched.chunked_req = chunked_req
    sched._pending_chunked_abort_req = pending_req
    sched.waiting_queue = []
    sched.dllm_config = None
    sched.grammar_manager = Mock()
    sched.disaggregation_mode = None
    sched.enable_hicache_storage = False
    sched.mm_receiver = None
    sched.ps = SimpleNamespace(pp_size=1)
    sched.running_batch = SimpleNamespace(reqs=running_reqs)
    sched.last_batch = None
    return sched


class TestPendingChunkedAbortRace(CustomTestCase):
    def test_req_left_chunked_slot_is_aborted(self):
        req = _FakeReq("zombie_rid")
        sched = _make_scheduler(req, chunked_req=None, running_reqs=[req])

        sched.process_pending_chunked_abort()

        self.assertIsNotNone(req.to_finish, "recorded abort was never applied")
        self.assertIsNone(sched._pending_chunked_abort_req)

    def test_finished_req_only_clears_marker(self):
        req = _FakeReq("done_rid")
        req._finished = True
        sched = _make_scheduler(req, chunked_req=None, running_reqs=[])

        sched.process_pending_chunked_abort()

        self.assertIsNone(req.to_finish)
        self.assertIsNone(sched._pending_chunked_abort_req)


class TestPrefillAbortSourceOwnership(CustomTestCase):
    def make_prefill_scheduler(self):
        req = _FakeReq("prefill-source")
        req.return_logprob = False
        req.metadata_buffer_index = 7
        req.pending_bootstrap = True
        req.bootstrap_room = 123
        req.finished_reason = None
        req.time_stats = Mock()
        req.disagg_kv_sender = Mock()
        req.disagg_kv_sender.poll.return_value = KVPoll.Failed
        req.disagg_kv_sender.is_source_release_safe.return_value = False

        scheduler = _make_scheduler(None, chunked_req=None, running_reqs=[])
        scheduler.disaggregation_mode = DisaggregationMode.PREFILL
        scheduler.disagg_prefill_inflight_queue = []
        scheduler.disagg_prefill_bootstrap_queue = SimpleNamespace(queue=[])
        scheduler.disagg_prefill_pending_chunk_rids = {req.rid}
        scheduler.req_to_metadata_buffer_idx_allocator = Mock()
        scheduler.tree_cache = Mock()
        scheduler.ipc_channels = Mock()
        scheduler.beam_coordinator = Mock()
        scheduler.enable_overlap = False
        scheduler.enable_hierarchical_cache = False
        scheduler.enable_unified_cache_external_linker = False
        scheduler.attn_cp_cpu_group = scheduler.attn_tp_cpu_group = None
        scheduler.scheduler_stage_metrics = None
        scheduler.ps.tp_rank = 0
        scheduler.metrics_reporter = SimpleNamespace(enable_metrics=False)
        scheduler.output_streamer = Mock()
        scheduler.collect_inflight_reqs = Mock(return_value=[])
        return scheduler, req

    @patch("sglang.srt.managers.scheduler.release_kv_cache")
    @patch("sglang.srt.disaggregation.prefill.release_kv_cache")
    def test_waiting_and_chunked_abort_keep_source_until_drained(
        self, release_prefill, release_scheduler
    ):
        for queued in (False, True):
            with self.subTest(queued=queued):
                scheduler, req = self.make_prefill_scheduler()
                if queued:
                    scheduler.waiting_queue = [req]
                    scheduler.abort_request(AbortReq(rid=req.rid))
                else:
                    scheduler._release_chunked_abort(req)

                self.assertEqual(scheduler.disagg_prefill_inflight_queue, [req])
                self.assertEqual(req.metadata_buffer_index, 7)
                self.assertTrue(req.kv.holds_kv)
                release_scheduler.assert_not_called()
                release_prefill.assert_not_called()

                with (
                    patch(
                        "sglang.srt.disaggregation.prefill.poll_and_all_reduce_attn_cp_tp_group",
                        side_effect=lambda senders, *_: [s.poll() for s in senders],
                    ),
                    patch("sglang.srt.disaggregation.prefill.dist.all_reduce"),
                ):
                    self.assertEqual(
                        scheduler.process_disagg_prefill_inflight_queue(), []
                    )
                    release_prefill.assert_not_called()
                    req.disagg_kv_sender.is_source_release_safe.return_value = True
                    self.assertEqual(
                        scheduler.process_disagg_prefill_inflight_queue(), [req]
                    )
                self.assertEqual(req.metadata_buffer_index, -1)
                self.assertEqual(scheduler.disagg_prefill_inflight_queue, [])
                release_prefill.assert_called_once_with(
                    req, scheduler.tree_cache, is_insert=False
                )
                release_prefill.reset_mock()


if __name__ == "__main__":
    unittest.main(verbosity=2)
