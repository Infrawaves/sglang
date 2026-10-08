import threading
import time
import unittest
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.common.conn import CommonKVManager, ParallelInfoState
from sglang.srt.disaggregation.decode import (
    DecodePreallocQueue,
    DecodeTransferQueue,
    HiCacheRestoreResult,
)
from sglang.srt.disaggregation.fake.conn import FakeKVManager, FakeKVReceiver
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.distributed.parallel_state_wrapper import ParallelState
from sglang.srt.managers.schedule_batch import FINISH_ABORT
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.runtime_context import get_context, publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class FakeReceiver:
    def __init__(self):
        self.clear_called = False
        self.conclude_state = None

    def clear(self):
        self.clear_called = True

    def failure_exception(self):
        return None


class TestDecodeQueueCleanup(CustomTestCase):
    def setUp(self):
        # The code under test reads its config from the bags.
        reset_context()
        self.addCleanup(reset_context)
        publish(ServerArgs(model_path="dummy"), role="tokenizer")

    def test_paged_swa_retraction_resume_uses_physical_page_budget(self):
        # resume_retracted_reqs reads the retraction backend off the disagg
        # bag, so the case publishes a config instead of injecting one.
        override = get_context().override_server_args(
            disaggregation_decode_retraction_backup="cpu_tensor"
        )
        override.install()
        self.addCleanup(override.restore)

        page_size = 128
        fill_len = 574
        physical_tokens_per_req = 5 * page_size
        physical_available = 18 * page_size

        reqs = [
            SimpleNamespace(
                rid=f"req-{i}",
                origin_input_ids=[0] * fill_len,
                output_ids=[],
                is_retracted=True,
                retraction_backup=None,
                load_kv_cache=MagicMock(),
            )
            for i in range(4)
        ]

        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue.retracted_queue = reqs.copy()
        queue.num_reserved_decode_tokens = 0
        queue.req_to_token_pool = SimpleNamespace(available_size=lambda: len(reqs))
        queue.token_to_kv_pool_allocator = SimpleNamespace(page_size=page_size)
        queue.tree_cache = MagicMock()
        queue.scheduler = SimpleNamespace(
            sliding_window_size=2047,
            server_args=SimpleNamespace(disable_radix_cache=True),
        )
        queue._uses_swa_tail_prealloc = MagicMock(return_value=True)
        queue._swa_aware_allocatable_token_budgets = MagicMock(
            return_value=(physical_available, physical_available)
        )
        queue._swa_tail_allocatable_token_budget = MagicMock(
            side_effect=lambda **_: physical_available
        )

        def pre_alloc(_req):
            nonlocal physical_available
            self.assertGreaterEqual(physical_available, physical_tokens_per_req)
            physical_available -= physical_tokens_per_req

        queue._pre_alloc = MagicMock(side_effect=pre_alloc)

        resumed = queue.resume_retracted_reqs()

        self.assertEqual(resumed, reqs[:3])
        self.assertEqual(queue.retracted_queue, reqs[3:])
        self.assertEqual(physical_available, 3 * page_size)
        self.assertEqual(queue._pre_alloc.call_count, 3)

    def test_prealloc_abort_clears_receiver_before_removing_request(self):
        receiver = FakeReceiver()
        req = SimpleNamespace(
            rid="abort-prealloc",
            bootstrap_room=42,
            finished_reason=None,
            return_logprob=False,
        )
        decode_req = SimpleNamespace(
            req=req, kv_receiver=receiver, waiting_for_input=True
        )

        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue.pp_size = 1
        queue.tp_rank = 0
        queue.gloo_group = object()
        queue.queue = [decode_req]
        queue.pending_reqs = []
        queue.retracted_queue = []
        queue._resolve_pending_reqs = MagicMock()
        queue._uses_swa_tail_prealloc = MagicMock(return_value=False)
        queue._allocatable_token_budgets = MagicMock(return_value=0)
        queue._hicache_pending_restore_tokens = MagicMock(return_value=0)

        scheduler = MagicMock()
        scheduler.running_batch.reqs = []
        scheduler.enable_priority_scheduling = False
        scheduler.enable_hisparse = False
        scheduler.metrics_reporter.enable_metrics = False
        scheduler.output_streamer = MagicMock()
        queue.scheduler = scheduler

        with patch(
            "sglang.srt.disaggregation.decode.poll_and_all_reduce",
            return_value=[KVPoll.Failed],
        ) as poll:
            preallocated, failed = queue.pop_preallocated()

        poll.assert_called_once_with([receiver], queue.gloo_group)
        self.assertEqual(preallocated, [])
        self.assertEqual(failed, [decode_req])
        self.assertEqual(queue.queue, [])
        self.assertTrue(receiver.clear_called)
        self.assertIsNone(decode_req.kv_receiver)
        self.assertIsInstance(req.finished_reason, FINISH_ABORT)
        scheduler.output_streamer.stream_output.assert_called_once_with(
            [req], req.return_logprob
        )

    def test_prealloc_abort_also_drops_from_pending_reqs(self):
        # Same DecodeRequest lives in both queue and pending_reqs (add() slow
        # path). Aborting must drop it from both, and compare by identity since
        # DecodeRequest's dataclass __eq__ would compare the tensor receiver.
        class BadEqReceiver(FakeReceiver):
            def __eq__(self, other):
                raise TypeError("use identity comparison, not value equality")

            __hash__ = object.__hash__

        receiver = BadEqReceiver()
        req = SimpleNamespace(
            rid="abort-shared",
            finished_reason=FINISH_ABORT("aborted"),
            return_logprob=False,
        )
        decode_req = SimpleNamespace(req=req, kv_receiver=receiver)

        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue.pp_size = 1
        queue.queue = [decode_req]
        queue.pending_reqs = [decode_req]  # same object, dual ownership
        queue.retracted_queue = []
        queue._resolve_pending_reqs = MagicMock()
        queue._update_handshake_waiters = MagicMock()
        queue._uses_swa_tail_prealloc = MagicMock(return_value=False)
        queue._allocatable_token_budgets = MagicMock(return_value=0)
        queue._hicache_pending_restore_tokens = MagicMock(return_value=0)

        scheduler = MagicMock()
        scheduler.running_batch.reqs = []
        scheduler.enable_priority_scheduling = False
        scheduler.enable_hisparse = False
        scheduler.output_streamer = MagicMock()
        queue.scheduler = scheduler

        # Must not raise on the receiver __eq__ above.
        preallocated, failed = queue.pop_preallocated()

        self.assertEqual(preallocated, [])
        self.assertEqual(failed, [decode_req])
        self.assertEqual(queue.queue, [])
        self.assertTrue(all(r is not decode_req for r in queue.pending_reqs))
        self.assertIsNone(decode_req.kv_receiver)

    def test_swa_reclaim_failure_rejects_only_request(self):
        receiver = FakeReceiver()
        req = SimpleNamespace(
            rid="swa-reclaim-failed",
            origin_input_ids=[1, 2, 3],
            output_ids=[],
            finished_reason=None,
            return_logprob=False,
            sampling_params=SimpleNamespace(max_new_tokens=1),
        )
        decode_req = SimpleNamespace(
            req=req,
            kv_receiver=receiver,
            waiting_for_input=True,
            is_rebootstrap=False,
        )

        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue.pp_size = 1
        queue.queue = [decode_req]
        queue.pending_reqs = [decode_req]
        queue.retracted_queue = []
        queue.num_reserved_decode_tokens = 0
        queue._resolve_pending_reqs = MagicMock()
        queue._update_handshake_waiters = MagicMock()
        queue._uses_swa_tail_prealloc = MagicMock(return_value=True)
        queue._swa_aware_allocatable_token_budgets = MagicMock(
            return_value=(1024, 1024)
        )
        queue._prealloc_required_tokens = MagicMock(return_value=(3, 3))
        queue._prealloc_kv_lens = MagicMock(return_value=(3, 3))
        queue._reclaim_swa_tail_capacity = MagicMock(
            return_value=(
                "SWA eviction insufficient: needed=64, available=0, "
                "req=swa-reclaim-failed"
            )
        )
        queue._hicache_pending_restore_tokens = MagicMock(return_value=0)
        queue._pre_alloc = MagicMock()
        queue.req_to_token_pool = MagicMock()
        queue.req_to_token_pool.available_size.return_value = 1
        queue.req_to_metadata_buffer_idx_allocator = MagicMock()
        queue.req_to_metadata_buffer_idx_allocator.available_size.return_value = 1

        scheduler = MagicMock()
        scheduler.running_batch.reqs = []
        scheduler.enable_priority_scheduling = False
        scheduler.enable_hisparse = False
        scheduler.server_args.disaggregation_decode_enable_radix_cache = False
        scheduler.output_streamer = MagicMock()
        queue.scheduler = scheduler

        preallocated, failed = queue.pop_preallocated()

        self.assertEqual(preallocated, [])
        self.assertEqual(failed, [decode_req])
        self.assertEqual(queue.queue, [])
        self.assertEqual(queue.pending_reqs, [])
        self.assertTrue(receiver.clear_called)
        self.assertIsNone(decode_req.kv_receiver)
        self.assertIsInstance(req.finished_reason, FINISH_ABORT)
        queue._pre_alloc.assert_not_called()
        scheduler.output_streamer.stream_output.assert_called_once_with(
            [req], req.return_logprob
        )

    def test_ensure_prefill_info_tolerates_cleared_receiver(self):
        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue._max_ensure_retries = 1
        queue._ensure_retry_interval = 0
        queue._ensure_retry_count = {"127.0.0.1:11500": 0}
        queue._ensure_last_attempt_time = {}
        queue.kv_manager = MagicMock()
        queue.kv_manager.try_ensure_parallel_info.return_value = False
        queue.kv_manager.has_parallel_info.return_value = False

        cleared_req = SimpleNamespace(
            req=SimpleNamespace(rid="cleared"), kv_receiver=None
        )
        addr_to_reqs = {"127.0.0.1:11500": [cleared_req]}

        ready, remaining = queue._ensure_prefill_info(addr_to_reqs)

        self.assertEqual(ready, {})
        self.assertEqual(remaining, [])

    def test_prefetches_prefill_dp_rank_query(self):
        addr = "127.0.0.1:11500"
        executor = MagicMock()
        future_first = Future()
        future_first.set_result({"7": 1})
        future_tail = Future()
        future_tail.set_result({"8": 2})
        executor.submit.side_effect = [future_first, future_tail]

        def decode_req(room):
            return SimpleNamespace(
                req=SimpleNamespace(
                    bootstrap_host="127.0.0.1",
                    bootstrap_port=11500,
                    bootstrap_room=room,
                ),
                kv_receiver=MagicMock(),
            )

        first = decode_req(7)
        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue.pending_reqs = [first]
        queue._prefill_dp_rank_queries = {}
        queue.kv_manager = SimpleNamespace(
            prefill_info_table={addr: object()},
            _ensure_bootstrap_executor=lambda: executor,
        )
        queue._resolve_prefill_dp_rank = MagicMock(return_value=None)
        queue._ensure_prefill_info = lambda groups: (groups, [])

        queue.prefetch_prefill_dp_rank_queries()
        tail = decode_req(8)
        queue.pending_reqs.append(tail)
        with patch(
            "sglang.srt.disaggregation.decode.CommonKVReceiver.query_prefill_dp_ranks",
            return_value={"8": 2},
        ) as query:
            queue._resolve_pending_reqs()

        _, called_addr, called_rooms = executor.submit.call_args_list[0].args
        self.assertEqual((called_addr, called_rooms), (addr, [7]))
        _, called_addr, called_rooms = executor.submit.call_args_list[1].args
        self.assertEqual((called_addr, called_rooms), (addr, [8]))
        query.assert_not_called()
        first.kv_receiver.init.assert_called_once_with(1)
        tail.kv_receiver.init.assert_not_called()
        self.assertEqual(queue.pending_reqs, [tail])

        queue._resolve_pending_reqs()
        tail.kv_receiver.init.assert_called_once_with(2)
        self.assertEqual(queue.pending_reqs, [])

    def _ensure_info_queue(self, addr, state, cached=False):
        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue._max_ensure_retries = 15
        queue._ensure_retry_interval = 1.0
        queue._ensure_retry_interval_max = 4.0
        queue._ensure_retry_count = {}
        queue._ensure_last_attempt_time = {}
        queue.kv_manager = MagicMock()
        queue.kv_manager.try_ensure_parallel_info.return_value = state
        queue.kv_manager.has_parallel_info.return_value = cached
        return queue

    def test_pending_topology_fetch_does_not_consume_retry_budget(self):
        addr = "127.0.0.1:11500"
        queue = self._ensure_info_queue(addr, ParallelInfoState.PENDING)
        req = SimpleNamespace(
            req=SimpleNamespace(rid="pending"), kv_receiver=MagicMock()
        )

        with patch(
            "sglang.srt.disaggregation.decode.time.monotonic", return_value=100.0
        ):
            for _ in range(50):
                ready, remaining = queue._ensure_prefill_info({addr: [req]})

        self.assertEqual(ready, {})
        self.assertEqual(remaining, [req])
        self.assertEqual(queue._ensure_retry_count, {})
        self.assertEqual(queue._ensure_last_attempt_time, {})
        self.assertEqual(
            queue.kv_manager.try_ensure_parallel_info.call_count,
            50,
        )
        req.kv_receiver.abort.assert_not_called()

    def test_completed_pending_fetch_is_consumed_without_retry_delay(self):
        addr = "127.0.0.1:11500"
        queue = self._ensure_info_queue(addr, ParallelInfoState.PENDING)
        queue.kv_manager.try_ensure_parallel_info.side_effect = [
            ParallelInfoState.PENDING,
            ParallelInfoState.READY,
        ]
        req = SimpleNamespace(
            req=SimpleNamespace(rid="pending-ready"), kv_receiver=MagicMock()
        )

        with patch(
            "sglang.srt.disaggregation.decode.time.monotonic", return_value=100.0
        ):
            first_ready, first_remaining = queue._ensure_prefill_info({addr: [req]})
            second_ready, second_remaining = queue._ensure_prefill_info({addr: [req]})

        self.assertEqual((first_ready, first_remaining), ({}, [req]))
        self.assertEqual((second_ready, second_remaining), ({addr: [req]}, []))
        self.assertEqual(
            queue.kv_manager.try_ensure_parallel_info.call_count,
            2,
        )

    def test_failed_topology_fetch_still_consumes_retry_budget(self):
        addr = "127.0.0.1:11500"
        queue = self._ensure_info_queue(addr, ParallelInfoState.FAILED)
        queue._max_ensure_retries = 2
        req = SimpleNamespace(
            req=SimpleNamespace(rid="failed"), kv_receiver=MagicMock()
        )

        ready, remaining = queue._ensure_prefill_info({addr: [req]})
        self.assertEqual((ready, remaining), ({}, [req]))
        self.assertEqual(queue._ensure_retry_count[addr], 1)

        queue._ensure_last_attempt_time = {}
        ready, remaining = queue._ensure_prefill_info({addr: [req]})
        self.assertEqual((ready, remaining), ({}, []))
        req.kv_receiver.abort.assert_called_once()
        self.assertNotIn(addr, queue._ensure_retry_count)

    def test_topology_retry_delay_backs_off_and_caps(self):
        addr = "127.0.0.1:11500"
        queue = self._ensure_info_queue(addr, ParallelInfoState.FAILED)

        delays = []
        for count in range(7):
            queue._ensure_retry_count = {addr: count}
            delays.append(queue._retry_delay_for(addr))

        self.assertEqual(delays, [1.0, 1.0, 2.0, 4.0, 4.0, 4.0, 4.0])

    def test_cached_topology_is_consumed_before_the_pacing_gate(self):
        addr = "127.0.0.1:11500"
        queue = self._ensure_info_queue(addr, ParallelInfoState.FAILED, cached=True)
        queue._ensure_retry_count = {addr: 3}
        queue._ensure_last_attempt_time = {addr: time.monotonic()}
        req = SimpleNamespace(
            req=SimpleNamespace(rid="cached"), kv_receiver=MagicMock()
        )

        ready, remaining = queue._ensure_prefill_info({addr: [req]})

        self.assertEqual(ready, {addr: [req]})
        self.assertEqual(remaining, [])
        queue.kv_manager.try_ensure_parallel_info.assert_not_called()
        self.assertNotIn(addr, queue._ensure_retry_count)
        self.assertNotIn(addr, queue._ensure_last_attempt_time)

    def test_unresolved_dp_rank_query_defers_instead_of_blocking(self):
        addr = "127.0.0.1:11500"
        pending_future = Future()  # deliberately never resolved
        req = SimpleNamespace(
            req=SimpleNamespace(
                bootstrap_host="127.0.0.1", bootstrap_port=11500, bootstrap_room=7
            ),
            kv_receiver=MagicMock(),
        )

        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue.pending_reqs = [req]
        queue._prefill_dp_rank_queries = {addr: ((7,), pending_future)}
        queue._resolve_prefill_dp_rank = MagicMock(return_value=None)
        queue._ensure_prefill_info = lambda groups: (groups, [])

        with patch(
            "sglang.srt.disaggregation.decode.CommonKVReceiver.query_prefill_dp_ranks"
        ) as query:
            queue._resolve_pending_reqs()

        query.assert_not_called()
        req.kv_receiver.init.assert_not_called()
        self.assertEqual(queue.pending_reqs, [req])
        self.assertIs(queue._prefill_dp_rank_queries[addr][1], pending_future)
        self.assertFalse(pending_future.cancelled())

    def test_missing_dp_rank_future_is_submitted_async(self):
        """A newly ingested request must not use the synchronous query fallback."""
        addr = "127.0.0.1:11500"
        req = SimpleNamespace(
            req=SimpleNamespace(
                bootstrap_host="127.0.0.1", bootstrap_port=11500, bootstrap_room=7
            ),
            kv_receiver=MagicMock(),
        )

        executor = MagicMock()
        submitted_future = Future()
        executor.submit.return_value = submitted_future

        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue.pending_reqs = [req]
        queue._prefill_dp_rank_queries = {}
        queue._resolve_prefill_dp_rank = MagicMock(return_value=None)
        queue._ensure_prefill_info = lambda groups: (groups, [])
        queue.kv_manager = MagicMock()
        queue.kv_manager._ensure_bootstrap_executor.return_value = executor

        with patch(
            "sglang.srt.disaggregation.decode.CommonKVReceiver.query_prefill_dp_ranks"
        ) as query:
            queue._resolve_pending_reqs()

        query.assert_not_called()
        executor.submit.assert_called_once_with(
            query,
            addr,
            [7],
        )
        req.kv_receiver.init.assert_not_called()
        self.assertEqual(queue.pending_reqs, [req])

    @patch("sglang.srt.disaggregation.decode.release_kv_cache")
    @patch("sglang.srt.disaggregation.decode.prepare_abort")
    @patch("sglang.srt.disaggregation.decode.poll_and_all_reduce")
    def test_transfer_failure_cleanup_respects_deferred_release_gates(
        self, mock_poll, mock_prepare_abort, mock_release_kv_cache
    ):
        receiver = FakeReceiver()
        req = SimpleNamespace(
            rid="failed-transfer",
            bootstrap_room=7,
            return_logprob=False,
        )
        decode_req = SimpleNamespace(
            req=req,
            kv_receiver=receiver,
            metadata_buffer_index=3,
            hicache_restore_status=HiCacheRestoreResult.READY,
        )

        queue = DecodeTransferQueue.__new__(DecodeTransferQueue)
        queue.queue = [decode_req]
        queue.enable_staging = False
        queue.enable_deferred_kv_release = False
        queue.gloo_group = MagicMock()
        queue.req_to_metadata_buffer_idx_allocator = MagicMock()
        queue.tp_rank = 0
        queue.tree_cache = MagicMock()
        queue.metadata_buffers = SimpleNamespace(bootstrap_room=[None] * 4)
        queue.spec_algorithm = MagicMock()
        queue.spec_algorithm.is_none.return_value = True
        queue._clean_hicache_prefetch_resources = MagicMock()

        scheduler = MagicMock()
        scheduler.enable_decode_hicache = False
        scheduler.enable_hisparse = False
        scheduler.output_streamer = MagicMock()
        scheduler.metrics_reporter.enable_metrics = False
        queue.scheduler = scheduler

        mock_poll.return_value = [KVPoll.Failed]

        transferred = queue.pop_transferred()

        self.assertEqual(transferred, [])
        self.assertEqual(queue.queue, [])
        self.assertTrue(receiver.clear_called)
        self.assertIsNone(decode_req.kv_receiver)
        queue.req_to_metadata_buffer_idx_allocator.free.assert_called_once_with(3)
        scheduler.output_streamer.stream_output.assert_called_once_with(
            [req], req.return_logprob
        )
        mock_prepare_abort.assert_called_once()
        mock_release_kv_cache.assert_called_once_with(
            req, queue.tree_cache, is_insert=False
        )

        receiver = FakeReceiver()
        receiver.kv_mgr = FakeKVManager.__new__(FakeKVManager)
        decode_req.kv_receiver = receiver
        queue.queue = [decode_req]
        queue.enable_deferred_kv_release = True
        queue.req_to_metadata_buffer_idx_allocator.reset_mock()
        mock_release_kv_cache.reset_mock()

        transferred = queue.pop_transferred()

        self.assertEqual(transferred, [])
        self.assertEqual(queue.queue, [])
        self.assertTrue(receiver.clear_called)
        self.assertIsNone(decode_req.kv_receiver)
        queue.req_to_metadata_buffer_idx_allocator.free.assert_called_once_with(3)
        mock_release_kv_cache.assert_called_once_with(
            req, queue.tree_cache, is_insert=False
        )

    def test_fake_receiver_initializes_deferred_release_state(self):
        manager = MagicMock()
        receiver = FakeKVReceiver(manager, "")

        self.assertIs(receiver.kv_mgr, manager)
        self.assertFalse(receiver.abort_notified)

    @patch("sglang.srt.disaggregation.decode.release_kv_cache")
    @patch("sglang.srt.disaggregation.decode.prepare_abort")
    def test_mooncake_failures_hold_destinations_even_without_optional_flag(
        self, mock_prepare_abort, mock_release_kv_cache
    ):
        for poll, restore_status in (
            (KVPoll.Failed, HiCacheRestoreResult.READY),
            (KVPoll.Transferring, HiCacheRestoreResult.FAILED),
        ):
            with self.subTest(poll=poll, restore_status=restore_status):
                receiver = FakeReceiver()
                receiver.kv_mgr = SimpleNamespace(
                    requires_transfer_drain=True,
                    enable_deferred_decode_kv_release=True,
                )
                receiver.bootstrap_infos = [{"rank": 0}, {"rank": 1}]
                receiver.abort_notified = False
                receiver.abort = MagicMock()
                req = SimpleNamespace(
                    rid="failed-transfer", bootstrap_room=7, return_logprob=False
                )
                dreq = SimpleNamespace(
                    req=req,
                    kv_receiver=receiver,
                    metadata_buffer_index=3,
                    hicache_restore_status=restore_status,
                )
                q = DecodeTransferQueue.__new__(DecodeTransferQueue)
                q.queue = [dreq]
                q.enable_staging = False
                q.enable_deferred_kv_release = False
                q.deferred_kv_release_timeout = 30
                q._deferred_releases = []
                q.tp_rank = 0
                q._poll_with_metadata_gate = MagicMock(return_value=[poll])
                q._clean_hicache_prefetch_resources = MagicMock()
                q.req_to_metadata_buffer_idx_allocator = MagicMock()
                q.scheduler = MagicMock()
                q.scheduler.enable_decode_hicache = False
                q.scheduler.enable_hisparse = True
                q.scheduler.metrics_reporter.enable_metrics = False

                self.assertEqual(q.pop_transferred(), [])
                self.assertEqual(q.queue, [])
                self.assertEqual(len(q._deferred_releases), 1)
                self.assertEqual(q._deferred_releases[0][2:], (3, 2))
                self.assertIs(dreq.kv_receiver, receiver)
                self.assertFalse(receiver.clear_called)
                q._clean_hicache_prefetch_resources.assert_not_called()
                q.scheduler.hisparse_coordinator.request_finished.assert_not_called()
                q.req_to_metadata_buffer_idx_allocator.free.assert_not_called()
                mock_release_kv_cache.assert_not_called()
                receiver.abort.assert_called_once()

    def test_retracted_decode_requests_keep_scheduler_non_idle(self):
        # is_fully_idle reads the retraction backend off the disagg bag.
        override = get_context().override_server_args(
            disaggregation_decode_retraction_backup="cpu_tensor"
        )
        override.install()
        self.addCleanup(override.restore)

        scheduler = Scheduler.__new__(Scheduler)
        scheduler.running_batch = MagicMock()
        scheduler.running_batch.is_empty.return_value = True
        scheduler.chunked_req = None
        scheduler.suspended_prefill_queue = []
        scheduler._pending_round_robin_actions = {}
        scheduler._engine_paused = False
        scheduler.dllm_manager = MagicMock()
        scheduler.dllm_manager.any_staging_reqs.return_value = False
        scheduler.last_batch = None
        scheduler.cur_batch_for_debug = None
        scheduler.enable_overlap = False
        scheduler.ps = ParallelState.trivial()
        scheduler.running_mbs = []
        scheduler.waiting_queue = []
        scheduler.grammar_manager = SimpleNamespace(grammar_queue=[])
        scheduler.disaggregation_mode = DisaggregationMode.DECODE
        scheduler.disagg_decode_prealloc_queue = SimpleNamespace(
            queue=[], retracted_queue=[object()], demotion_queue=[]
        )
        scheduler.disagg_decode_transfer_queue = SimpleNamespace(
            queue=[], has_pending_deferred_releases=lambda: False
        )
        scheduler.decode_offload_manager = None
        scheduler.enable_hisparse = False
        scheduler.enable_hierarchical_cache = False

        self.assertFalse(scheduler.is_fully_idle())

        scheduler.disagg_decode_prealloc_queue.retracted_queue.clear()
        self.assertTrue(scheduler.is_fully_idle())
        scheduler.disagg_decode_transfer_queue.has_pending_deferred_releases = lambda: (
            True
        )
        self.assertFalse(scheduler.is_fully_idle())
        # Quarantined buffers have no GPU forward to carry a health response;
        # permit a fresh health request while still blocking cache flush/offload.
        self.assertTrue(scheduler.is_fully_idle(for_health_check=True))


class TestParallelInfoFetchEpoch(CustomTestCase):
    """The topology fetch runs on an executor, so it can outlive the heartbeat's
    decision that the prefill it was querying is dead."""

    ADDR = "prefill:8998"

    def _manager(self, payload=None, fetch=None):
        # CommonKVManager implements both of BaseKVManager's abstract methods, so
        # it can be built without running __init__ (which needs torch.distributed,
        # ZMQ sockets and a live bootstrap server).
        mgr = CommonKVManager.__new__(CommonKVManager)
        mgr.prefill_info_table = {}
        mgr.connection_lock = threading.Lock()
        mgr.connection_pool = {}
        mgr._parallel_info_futures = {}
        mgr._parallel_info_epochs = defaultdict(int)
        mgr.addr_to_rooms_tracker = defaultdict(set)
        mgr.request_status = {}
        mgr.kv_args = SimpleNamespace(page_size=64)
        mgr.kv_cache_dtype_str = "bfloat16"
        mgr.dcp_size = 1
        mgr.is_mla_backend = False
        mgr.is_hybrid_mla_backend = False
        mgr.record_failure = MagicMock()
        mgr.update_status = MagicMock()
        mgr.check_status = MagicMock(return_value=KVPoll.Bootstrapping)
        mgr._resolve_rank_mapping = MagicMock()

        executor = ThreadPoolExecutor(max_workers=2)
        self.addCleanup(executor.shutdown, wait=True)
        mgr._ensure_prefill_recompute_executor = lambda: executor
        mgr._ensure_parallel_info_executor = lambda: executor
        mgr._ensure_bootstrap_executor = lambda: executor

        if fetch is None:
            row = payload if payload is not None else self._payload()
            fetch = MagicMock(return_value=row)
        mgr._fetch_parallel_info_payload = fetch
        return mgr, fetch

    @staticmethod
    def _payload(**overrides):
        row = {
            "attn_tp_size": 8,
            "attn_cp_size": 1,
            "dp_size": 1,
            "pp_size": 1,
            "page_size": 64,
            "kv_cache_dtype": "bfloat16",
            "follow_bootstrap_room": True,
        }
        row.update(overrides)
        return row

    def _drain(self, mgr):
        """Advance the state machine until it leaves PENDING."""
        for _ in range(200):
            state = mgr.try_ensure_parallel_info(self.ADDR)
            if state != ParallelInfoState.PENDING:
                return state
            time.sleep(0.005)
        self.fail("fetch never left PENDING")

    def test_fetch_is_not_awaited_on_the_calling_thread(self):
        entered = threading.Event()
        release = threading.Event()
        self.addCleanup(release.set)

        def blocking_fetch(addr):
            entered.set()
            release.wait(5)
            return self._payload()

        mgr, _ = self._manager(fetch=blocking_fetch)

        state = mgr.try_ensure_parallel_info(self.ADDR)
        self.assertEqual(state, ParallelInfoState.PENDING)
        self.assertTrue(entered.wait(5), "fetch never started")
        self.assertEqual(
            mgr.try_ensure_parallel_info(self.ADDR), ParallelInfoState.PENDING
        )
        self.assertNotIn(self.ADDR, mgr.prefill_info_table)

        release.set()
        self.assertEqual(self._drain(mgr), ParallelInfoState.READY)
        self.assertIn(self.ADDR, mgr.prefill_info_table)

    def test_in_flight_fetch_is_not_duplicated(self):
        entered = threading.Event()
        release = threading.Event()
        self.addCleanup(release.set)
        calls = []

        def blocking_fetch(addr):
            calls.append(addr)
            entered.set()
            release.wait(5)
            return self._payload()

        mgr, _ = self._manager(fetch=blocking_fetch)

        mgr.try_ensure_parallel_info(self.ADDR)
        self.assertTrue(entered.wait(5))
        for _ in range(20):
            mgr.try_ensure_parallel_info(self.ADDR)

        release.set()
        self._drain(mgr)
        self.assertEqual(calls, [self.ADDR], "one GET per addr while in flight")

    def test_eviction_pops_the_in_flight_fetch(self):
        entered = threading.Event()
        release = threading.Event()
        self.addCleanup(release.set)

        def blocking_fetch(addr):
            entered.set()
            release.wait(5)
            return self._payload()

        mgr, _ = self._manager(fetch=blocking_fetch)

        self.assertEqual(
            mgr.try_ensure_parallel_info(self.ADDR), ParallelInfoState.PENDING
        )
        self.assertTrue(entered.wait(5), "fetch never started")

        with patch(
            "sglang.srt.disaggregation.common.conn.CommonKVReceiver.disconnect_endpoint"
        ):
            mgr._handle_node_failure(self.ADDR)

        self.assertNotIn(self.ADDR, mgr._parallel_info_futures)
        self.assertEqual(mgr._parallel_info_epochs[self.ADDR], 1)
        release.set()

    def test_eviction_between_fetch_and_publish_does_not_resurrect_address(self):
        mgr, _ = self._manager()
        evicted = []

        def evict_mid_publish(info):
            if evicted:
                return
            evicted.append(True)
            with patch(
                "sglang.srt.disaggregation.common.conn.CommonKVReceiver.disconnect_endpoint"
            ):
                mgr._handle_node_failure(self.ADDR)

        mgr._resolve_rank_mapping = evict_mid_publish

        state = self._drain(mgr)

        self.assertTrue(evicted, "injection point never ran")
        self.assertEqual(state, ParallelInfoState.FAILED)
        self.assertNotIn(
            self.ADDR,
            mgr.prefill_info_table,
            "a fetch that resolved before eviction republished a dead prefill",
        )

    def test_fetch_after_eviction_publishes_again(self):
        mgr, _ = self._manager()

        with patch(
            "sglang.srt.disaggregation.common.conn.CommonKVReceiver.disconnect_endpoint"
        ):
            mgr._handle_node_failure(self.ADDR)
        self.assertEqual(mgr._parallel_info_epochs[self.ADDR], 1)

        self.assertEqual(self._drain(mgr), ParallelInfoState.READY)
        self.assertIn(self.ADDR, mgr.prefill_info_table)

    def test_cancelled_fetch_is_reported_as_a_failed_attempt(self):
        mgr, _ = self._manager()
        cancelled = Future()
        self.assertTrue(cancelled.cancel())
        mgr._parallel_info_futures[self.ADDR] = (0, cancelled)

        self.assertEqual(
            mgr.try_ensure_parallel_info(self.ADDR), ParallelInfoState.FAILED
        )
        self.assertNotIn(self.ADDR, mgr._parallel_info_futures)

    def test_cache_hit_needs_no_executor(self):
        mgr, fetch = self._manager()
        mgr.prefill_info_table[self.ADDR] = object()

        self.assertTrue(mgr.has_parallel_info(self.ADDR))
        self.assertEqual(
            mgr.try_ensure_parallel_info(self.ADDR), ParallelInfoState.READY
        )
        fetch.assert_not_called()

    def test_failed_fetch_is_retried_by_the_next_call(self):
        mgr, fetch = self._manager(fetch=MagicMock(return_value=None))

        self.assertEqual(self._drain(mgr), ParallelInfoState.FAILED)
        self.assertNotIn(self.ADDR, mgr._parallel_info_futures)

        fetch.return_value = self._payload()
        self.assertEqual(self._drain(mgr), ParallelInfoState.READY)
        self.assertIn(self.ADDR, mgr.prefill_info_table)

    def test_page_size_mismatch_raises_on_the_calling_thread(self):
        mgr, _ = self._manager(payload=self._payload(page_size=32))

        with self.assertRaises(RuntimeError) as ctx:
            self._drain(mgr)
        self.assertIn("Page size mismatch", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
