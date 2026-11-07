"""Tests for deferred chunked-prefill aborts."""

import unittest
from collections import deque
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import (
    CustomTestCase,
    enter_scope,
    maybe_stub_sgl_kernel,
    published_topology,
)

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.base import KVPoll  # noqa: E402
from sglang.srt.disaggregation.utils import DisaggregationMode  # noqa: E402
from sglang.srt.managers.io_struct import AbortReq  # noqa: E402
from sglang.srt.managers.req_types import FINISH_ABORT  # noqa: E402
from sglang.srt.managers.schedule_batch import Req  # noqa: E402
from sglang.srt.managers.scheduler import Scheduler  # noqa: E402
from sglang.srt.managers.scheduler_components import (  # noqa: E402
    request_retirer as retirer_mod,
)
from sglang.test.scheduler_test_utils import (  # noqa: E402
    empty_request_lists,
    release_only_retirer,
)

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class _FakeReq:
    """Minimal stand-in for Req: only the fields the abort paths touch."""

    request_finish = Req.request_finish
    clear_requested_finish = Req.clear_requested_finish
    finish_outside_step = Req.finish_outside_step

    def __init__(self, rid: str):
        self.rid = rid
        # Mirrors Req.kv; the abort paths read only these fields.
        self.kv = SimpleNamespace(
            holds_kv=True, holds_mamba=False, retraction_backup=None
        )
        self.to_finish = None
        self._finished = False
        self.discard_output_reason = None
        self.return_logprob = False
        self.output_ids = []
        self.weight_version_events = []
        self.prefill_weight_versions = None
        self.time_stats = Mock()
        self.cache_request_handle = None
        self.session = None

    def finished(self):
        return self._finished

    def release_mm_inputs_on_abort(self):
        pass

    def get_mm_token_counts(self):
        return None


def _make_scheduler(pending_req, *, chunked_req, running_reqs) -> Scheduler:
    sched = Scheduler.__new__(Scheduler)
    empty_request_lists(sched)
    sched.enable_continuous_input_polling = False
    sched.result_queue = deque()
    sched.enable_chunked_prefill_round_robin = False
    sched.chunked_req = chunked_req
    sched._pending_chunked_abort_req = pending_req
    sched.disaggregation_mode = None
    sched.enable_hicache_storage = False
    sched.mm_receiver = None
    sched.rust_server = None
    sched.running_batch = SimpleNamespace(reqs=running_reqs)
    sched.last_batch = None
    sched.beam_coordinator = Mock()
    sched.output_streamer = Mock()
    sched.enable_hisparse = False
    return sched


class TestPendingChunkedAbortRace(CustomTestCase):
    def setUp(self):
        enter_scope(self, published_topology())

    def test_req_left_chunked_slot_is_aborted(self):
        req = _FakeReq("zombie_rid")
        sched = _make_scheduler(req, chunked_req=None, running_reqs=[req])

        sched.process_pending_chunked_abort()

        self.assertIsNotNone(req.to_finish, "recorded abort was never applied")
        self.assertIsNone(sched._pending_chunked_abort_req)

    def test_discarded_chunk_keeps_its_abort_reason(self):
        req = _FakeReq("failed_load")
        req.discard_output_reason = FINISH_ABORT("External KV cache load failed", 503)
        sched = _make_scheduler(req, chunked_req=req, running_reqs=[])
        sched.request_retirer = Mock()
        sched.ipc_channels = Mock()

        sched.process_pending_chunked_abort()

        self.assertEqual(req.finished_reason.status_code, 503)
        (abort_req, _), _ = sched.ipc_channels.send_to_tokenizer.send_output.call_args
        self.assertEqual(abort_req.finished_reason, req.discard_output_reason.to_json())

    def test_aborted_chunked_beam_leader_retires_its_group(self):
        req = _FakeReq("beam_leader")
        sched = _make_scheduler(req, chunked_req=req, running_reqs=[])
        sched.request_retirer = release_only_retirer(
            Mock(), beam_coordinator=sched.beam_coordinator
        )
        sched.ipc_channels = Mock()

        with (
            patch.object(sched.request_retirer, "_detach"),
            patch.object(retirer_mod, "release_kv_cache"),
        ):
            sched.process_pending_chunked_abort()

        sched.beam_coordinator.retire_group.assert_called_once_with(req)

    def test_aborted_chunked_req_releases_hisparse_before_kv(self):
        # Each chunk's extend allocates HiSparse spec scratch for the row.
        req = _FakeReq("hisparse_chunked")
        sched = _make_scheduler(req, chunked_req=req, running_reqs=[])
        sched.ipc_channels = Mock()
        calls = []
        hisparse_coordinator = Mock()
        hisparse_coordinator.request_finished.side_effect = lambda req: calls.append(
            ("hisparse", req)
        )
        sched.request_retirer = release_only_retirer(Mock(), hisparse_coordinator)

        with (
            patch.object(sched.request_retirer, "_detach"),
            patch.object(
                retirer_mod,
                "release_kv_cache",
                side_effect=lambda req, *a, **kw: calls.append(("kv", req)),
            ),
        ):
            sched.process_pending_chunked_abort()

        self.assertEqual(calls, [("hisparse", req), ("kv", req)])


class TestPrefillAbortSourceOwnership(CustomTestCase):
    def setUp(self):
        enter_scope(self, published_topology())

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
        scheduler.metrics_reporter = SimpleNamespace(enable_metrics=False)
        scheduler.output_streamer = Mock()
        scheduler.collect_inflight_reqs = Mock(return_value=[])
        scheduler.request_retirer = release_only_retirer(
            scheduler.tree_cache, beam_coordinator=scheduler.beam_coordinator
        )
        return scheduler, req

    @patch.object(retirer_mod, "release_kv_cache")
    def test_waiting_and_chunked_abort_keep_source_until_drained(
        self, release_retirer
    ):
        for queued in (False, True):
            with self.subTest(queued=queued):
                scheduler, req = self.make_prefill_scheduler()
                if queued:
                    scheduler.waiting_queue.join([req])
                    scheduler.abort_request(AbortReq(rid=req.rid))
                else:
                    scheduler.chunked_req = req
                    scheduler._release_chunked_abort(req)
                    self.assertNotIn(req, scheduler.chunked_reqs)

                self.assertEqual(list(scheduler.disagg_prefill_transfer_queue), [req])
                self.assertNotIn(req, scheduler.waiting_queue)
                self.assertEqual(req.metadata_buffer_index, 7)
                self.assertTrue(req.kv.holds_kv)
                self.assertFalse(req.pending_bootstrap)
                release_retirer.assert_not_called()

                with (
                    patch(
                        "sglang.srt.disaggregation.prefill.poll_and_all_reduce_attn_cp_tp_group",
                        side_effect=lambda senders, *_: [s.poll() for s in senders],
                    ),
                    patch("sglang.srt.disaggregation.prefill.dist.all_reduce"),
                ):
                    self.assertEqual(
                        scheduler.process_disagg_prefill_transfer_queue(), []
                    )
                    release_retirer.assert_not_called()
                    req.disagg_kv_sender.is_source_release_safe.return_value = True
                    self.assertEqual(
                        scheduler.process_disagg_prefill_transfer_queue(), [req]
                    )
                self.assertEqual(req.metadata_buffer_index, -1)
                self.assertEqual(len(scheduler.disagg_prefill_transfer_queue), 0)
                release_retirer.assert_called_once_with(
                    req, scheduler.tree_cache, checkpoint=False
                )
                release_retirer.reset_mock()


if __name__ == "__main__":
    unittest.main(verbosity=2)
