import contextlib
import threading
import time
import unittest
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from disagg_test_utils import make_decode_kv_manager
from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.base.conn import StateType
from sglang.srt.disaggregation.common.conn import (
    CommonKVManager,
    KVTransferError,
    ParallelInfoState,
)
from sglang.srt.disaggregation.decode import (
    DecodePreallocQueue,
    DecodeRequest,
    DecodeTransferQueue,
    SchedulerDisaggregationDecodeMixin,
    _PendingPublish,
)
from sglang.srt.disaggregation.decode_hicache_mixin import (
    HiCacheRestoreGatedKVReceiver,
    HiCacheRestoreResult,
)
from sglang.srt.disaggregation.utils import ReqToMetadataIdxAllocator
from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import ReqKvInfo
from sglang.srt.managers.scheduler_components.request_list import (
    IndexedReqList,
    ListKind,
)
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.scheduler_test_utils import FakeReq, decode_queue
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=13, suite="base-a-test-cpu")


def _run_as_single_rank(test: unittest.TestCase) -> None:
    # The transfer queue sizes its release collectives from the gloo group;
    # these tests have no process group, so answer as one rank.
    world_size = patch("torch.distributed.get_world_size", return_value=1)
    world_size.start()
    test.addCleanup(world_size.stop)


class FakeReceiver:
    def __init__(self):
        self.clear_called = False
        self.conclude_state = None

    def clear(self):
        self.clear_called = True

    def failure_exception(self):
        return None


class TestDeferredReleaseRecovery(CustomTestCase):
    def _queue(self, *, sent=True, already_notified=False):
        manager = CommonKVManager.__new__(CommonKVManager)
        manager.requires_transfer_drain = True
        manager._deferred_abort_ack_tracker = {}
        manager._deferred_abort_generation = 0
        # The quarantined transfer's armed ABORT generation (#37077).
        self.generation = manager.register_deferred_abort_room(7)
        receiver = SimpleNamespace(
            kv_mgr=manager,
            _abort_generation=self.generation,
            retry_abort=MagicMock(return_value=sent),
            abort_notified=already_notified,
            bootstrap_infos=[{"rank": 0}, {"rank": 1}],
        )
        req = SimpleNamespace(
            req=SimpleNamespace(rid="quarantined", bootstrap_room=7),
            kv_receiver=receiver,
            metadata_buffer_index=3,
            hicache_restore_lock=None,
        )
        queue = DecodeTransferQueue.__new__(DecodeTransferQueue)
        queue.enable_host_receive = False
        queue.scheduler = SimpleNamespace(
            metrics_reporter=SimpleNamespace(enable_metrics=False)
        )
        queue._release_tp_size = 1
        queue._deferred_releases = []
        queue._failed_deferred_releases = []
        queue._deferred_release_error = None
        queue.deferred_kv_release_timeout = 30
        queue.enable_staging = False
        queue._do_release = MagicMock()
        with patch("sglang.srt.disaggregation.decode.time.monotonic", return_value=0):
            queue._defer_release(req)
        return queue, req, manager

    def test_unsent_abort_retries_before_warning_deadline(self):
        queue, req, manager = self._queue()
        with (
            patch("sglang.srt.disaggregation.decode.time.monotonic") as clock,
            patch("sglang.srt.disaggregation.decode.logger") as logger,
        ):
            clock.return_value = 0.049
            queue.resolve_deferred_releases()
            req.kv_receiver.retry_abort.assert_not_called()
            clock.return_value = 0.05
            queue.resolve_deferred_releases()
            req.kv_receiver.retry_abort.assert_called_once_with()
            logger.warning.assert_not_called()
        self.assertAlmostEqual(req._abort_retry_deadline, 30.05)
        self.assertFalse(manager.is_abort_release_safe(7, 2))
        queue._do_release.assert_not_called()

    def test_backpressure_backs_off_without_releasing_buffers(self):
        queue, req, manager = self._queue(sent=False)
        intervals = []
        with (
            patch("sglang.srt.disaggregation.decode.time.monotonic") as clock,
            patch("sglang.srt.disaggregation.decode.logger"),
        ):
            for attempt in range(20):
                now = req._abort_retry_deadline
                clock.return_value = now
                queue.resolve_deferred_releases()
                intervals.append(req._abort_retry_deadline - now)
                self.assertEqual(req.kv_receiver.retry_abort.call_count, attempt + 1)
                # A second scheduler step at the same time cannot busy-loop.
                queue.resolve_deferred_releases()
                self.assertEqual(req.kv_receiver.retry_abort.call_count, attempt + 1)
        self.assertAlmostEqual(intervals[0], 0.1)
        for previous, current in zip(intervals, intervals[1:]):
            self.assertGreaterEqual(current + 1e-9, previous)
        self.assertAlmostEqual(max(intervals), 60.0)
        self.assertTrue(queue.has_pending_deferred_releases())
        self.assertFalse(manager.is_abort_release_safe(7, 2))
        queue._do_release.assert_not_called()

    def test_matching_late_acks_release_during_retry_backoff(self):
        queue, req, manager = self._queue(sent=False)
        receiver = req.kv_receiver
        with patch("sglang.srt.disaggregation.decode.time.monotonic") as clock:
            clock.return_value = 0.05
            queue.resolve_deferred_releases()
            manager.note_abort_ack(7, 0, self.generation + 1)
            manager.note_abort_ack(7, 1, self.generation)
            clock.return_value = 0.06
            queue.resolve_deferred_releases()
            queue._do_release.assert_not_called()
            manager.note_abort_ack(7, 0, self.generation)
            clock.return_value = 0.07
            queue.resolve_deferred_releases()
        receiver.retry_abort.assert_called_once_with()
        queue._do_release.assert_called_once_with(req, 3)
        self.assertFalse(queue.has_pending_deferred_releases())

    def test_accepted_abort_waits_for_ack_and_retries_with_backoff(self):
        queue, req, manager = self._queue(already_notified=True)
        with (
            patch("sglang.srt.disaggregation.decode.time.monotonic") as clock,
            patch("sglang.srt.disaggregation.decode.logger"),
        ):
            clock.return_value = 29.99
            queue.resolve_deferred_releases()
            req.kv_receiver.retry_abort.assert_not_called()
            clock.return_value = 30
            queue.resolve_deferred_releases()
            self.assertEqual(req._abort_retry_deadline, 90)
            clock.return_value = 89
            queue.resolve_deferred_releases()
            req.kv_receiver.retry_abort.assert_called_once_with()
            clock.return_value = 90
            queue.resolve_deferred_releases()
            self.assertEqual(req.kv_receiver.retry_abort.call_count, 2)
            self.assertEqual(req._abort_retry_deadline, 150)
        self.assertFalse(manager.is_abort_release_safe(7, 2))
        queue._do_release.assert_not_called()

    def test_warning_does_not_override_retry_backoff(self):
        queue, req, manager = self._queue(sent=False)
        req._abort_retry_deadline = 120
        with (
            patch("sglang.srt.disaggregation.decode.time.monotonic", return_value=30),
            self.assertLogs(
                "sglang.srt.disaggregation.decode", level="WARNING"
            ) as logs,
        ):
            queue.resolve_deferred_releases()
        self.assertIn("retrying ABORT with backoff", logs.output[0])
        req.kv_receiver.retry_abort.assert_not_called()
        # (start_time, deadline, metadata_idx, required_acks)
        self.assertEqual(queue._deferred_releases[0][1:], (0, 60, 3, 2))
        queue._do_release.assert_not_called()


class TestDecodeQueueCleanup(CustomTestCase):
    def setUp(self):
        _run_as_single_rank(self)
        # The code under test reads its config from the bags.
        reset_context()
        self.addCleanup(reset_context)
        publish(ServerArgs(model_path="dummy"), role="tokenizer")

    def test_ensure_prefill_info_tolerates_cleared_receiver(self):
        # A req whose kv_receiver was already cleared must not crash on .abort().
        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue._max_ensure_retries = 1
        queue._ensure_retry_interval = 0
        queue._ensure_retry_count = {"127.0.0.1:11500": 0}
        queue._ensure_last_attempt_time = {}
        queue.kv_manager = MagicMock()
        queue.kv_manager.try_ensure_parallel_info.return_value = False

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
        future = Future()
        future.set_result({"7": 1})
        executor.submit.return_value = future

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
            _ensure_prefill_recompute_executor=lambda: executor,
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

        _, called_addr, called_rooms = executor.submit.call_args.args
        self.assertEqual((called_addr, called_rooms), (addr, [7]))
        query.assert_called_once_with(addr, [8])
        first.kv_receiver.init.assert_called_once_with(1)
        tail.kv_receiver.init.assert_called_once_with(2)
        self.assertEqual(queue.pending_reqs, [])

    @patch("sglang.srt.disaggregation.decode.release_kv_cache")
    @patch("sglang.srt.disaggregation.decode.prepare_abort")
    def test_mooncake_failures_hold_destinations_even_without_optional_flag(
        self, mock_prepare_abort, mock_release_kv_cache
    ):
        # A restore failure folds into the poll only once transport is terminal
        # (#43562), so Failed is the case that reaches this branch.
        for poll, restore_status in (
            (KVPoll.Failed, HiCacheRestoreResult.READY),
            (KVPoll.Failed, HiCacheRestoreResult.FAILED),
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

                def ensure_abort_notified(*, force_arm=False, receiver=receiver):
                    receiver.abort()
                    receiver.abort_notified = True

                receiver.ensure_abort_notified = ensure_abort_notified
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
                q.enable_host_receive = False
                q.enable_deferred_kv_release = False
                q.deferred_kv_release_timeout = 30
                q.gloo_group = MagicMock()
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
                self.assertEqual(q._deferred_releases[0][3:], (3, 2))
                self.assertIs(dreq.kv_receiver, receiver)
                self.assertFalse(receiver.clear_called)
                q._clean_hicache_prefetch_resources.assert_not_called()
                q.scheduler.hisparse_coordinator.request_finished.assert_not_called()
                q.req_to_metadata_buffer_idx_allocator.free.assert_not_called()
                mock_release_kv_cache.assert_not_called()
                receiver.abort.assert_called_once()

    def test_retracted_decode_requests_keep_scheduler_non_idle(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.running_batch = MagicMock()
        scheduler.running_batch.is_empty.return_value = True
        scheduler.chunked_req = None
        scheduler._engine_paused = False
        scheduler.dllm_manager = MagicMock()
        scheduler.dllm_manager.any_staging_reqs.return_value = False
        scheduler.last_batch = None
        scheduler.cur_batch_for_debug = None
        scheduler.enable_overlap = False
        scheduler.running_mbs = []
        scheduler.waiting_queue = []
        scheduler.grammar_manager = SimpleNamespace(grammar_queue=[])
        scheduler.disaggregation_mode = DisaggregationMode.DECODE
        scheduler.disagg_decode_prealloc_queue = SimpleNamespace(
            queue=[], retracted_queue=[object()]
        )
        scheduler.disagg_decode_transfer_queue = SimpleNamespace(
            queue=[], has_pending_deferred_releases=lambda: False
        )
        scheduler.decode_offload_manager = None
        scheduler.enable_hisparse = False
        scheduler.enable_hierarchical_cache = False
        scheduler.enable_lmcache = False

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

    def test_unresolved_dp_rank_query_defers_instead_of_blocking(self):
        req = self._pending_decode_req(7)
        pending_future = Future()  # deliberately never resolved
        queue = self._dp_rank_queue([req], [pending_future])

        queue._resolve_pending_reqs()
        queue._resolve_pending_reqs()

        queue.kv_manager.submit_dp_rank_query.assert_called_once()
        req.kv_receiver.init.assert_not_called()
        self.assertEqual(queue.pending_reqs, [req])
        self.assertFalse(pending_future.cancelled())

    def test_dp_rank_answer_does_not_cover_a_later_request_reusing_the_room(self):
        """An answer is bound to the request that asked, not to its room: a new
        request that reuses the room is asked about again."""
        old = self._pending_decode_req(7)
        lookup = Future()
        queue = self._dp_rank_queue([old], [lookup, Future()])
        queue._resolve_pending_reqs()

        new = self._pending_decode_req(7)
        queue.pending_reqs = [new]
        lookup.set_result({"7": 1})
        queue._resolve_pending_reqs()

        new.kv_receiver.init.assert_not_called()
        self.assertEqual(queue.pending_reqs, [new])
        self.assertEqual(queue.kv_manager.submit_dp_rank_query.call_count, 2)

    def test_dp_rank_answer_from_before_eviction_is_discarded(self):
        req = self._pending_decode_req(7)
        queue = self._dp_rank_queue([req], [self._answered({"7": 1}), Future()])
        queue.kv_manager.parallel_info_epoch.side_effect = [0, 1, 1]

        queue._resolve_pending_reqs()
        queue._resolve_pending_reqs()

        req.kv_receiver.init.assert_not_called()
        self.assertEqual(queue.pending_reqs, [req])

    def test_steady_arrivals_are_never_aborted_by_the_dp_rank_lookup(self):
        """With requests arriving every cycle, some request is always waiting for
        its room to be registered on the prefill; that must not count as one
        lookup stuck for the whole handshake timeout and abort fresh arrivals."""
        queue = self._dp_rank_queue([], [])
        registered = set()
        in_flight = []

        def submit_dp_rank_query(*, bootstrap_addr, bootstrap_rooms):
            future = Future()
            in_flight.append((future, list(bootstrap_rooms)))
            return future

        queue.kv_manager.submit_dp_rank_query.side_effect = submit_dp_rank_query
        arrived = []
        now = 1000.0
        for room in range(1, 801):
            now += 1.0
            with patch(
                "sglang.srt.disaggregation.decode.time.monotonic", return_value=now
            ):
                registered.update(range(1, room))  # prefill lags by one cycle
                decode_req = self._pending_decode_req(room)
                arrived.append(decode_req)
                queue.pending_reqs.append(decode_req)
                queue.prefetch_prefill_dp_rank_queries()
                queue._resolve_pending_reqs()
            for future, rooms in in_flight:
                future.set_result({str(r): 0 for r in rooms if r in registered})
            in_flight.clear()

        for decode_req in arrived:
            decode_req.kv_receiver.abort.assert_not_called()
        self.assertTrue(all(r.kv_receiver.init.called for r in arrived[:-2]))

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

    def test_missing_dp_rank_future_is_submitted_async(self):
        """A newly ingested request must not use the synchronous query fallback."""
        req = self._pending_decode_req(7)
        queue = self._dp_rank_queue([req], [Future()])

        with patch(
            "sglang.srt.disaggregation.decode.CommonKVReceiver.query_prefill_dp_ranks"
        ) as query:
            queue._resolve_pending_reqs()

        query.assert_not_called()
        queue.kv_manager.submit_dp_rank_query.assert_called_once_with(
            bootstrap_addr="127.0.0.1:11500", bootstrap_rooms=[7]
        )
        req.kv_receiver.init.assert_not_called()
        self.assertEqual(queue.pending_reqs, [req])


class TestHeldDestinationPublication(CustomTestCase):
    def setUp(self):
        super().setUp()
        _run_as_single_rank(self)
        reset_context()
        self.addCleanup(reset_context)
        publish(ServerArgs(model_path="dummy"), role="tokenizer")

    def make_queue(self):
        queue = DecodeTransferQueue.__new__(DecodeTransferQueue)
        queue.gloo_group = object()
        receiver = SimpleNamespace(
            poll=MagicMock(return_value=KVPoll.WaitingForInput),
            send_metadata=MagicMock(),
        )
        event = SimpleNamespace(query=MagicMock(return_value=True))
        entry = SimpleNamespace(
            req=FakeReq(),
            kv_receiver=receiver,
            pending_metadata=(event, ("pages", 3, None), {}),
            is_rebootstrap=False,
        )
        queue.queue = decode_queue(ListKind.PD_RECEIVING, [entry])
        return queue, entry, event, receiver

    def test_waits_for_every_rank_and_sends_once(self):
        queue, entry, event, receiver = self.make_queue()
        with (
            patch("torch.distributed.get_world_size", return_value=2),
            patch("torch.distributed.all_reduce") as reduce,
        ):
            # This rank finished its forward; another rank is still writing.
            reduce.side_effect = lambda flags, **_: flags.zero_()
            queue.publish_pending_metadata()
            receiver.send_metadata.assert_not_called()
            self.assertIsNotNone(entry.pending_metadata)

            # The local forward is now the one that still needs to finish.
            reduce.side_effect = None
            event.query.return_value = False
            queue.publish_pending_metadata()
            receiver.send_metadata.assert_not_called()

            event.query.return_value = True
            queue.publish_pending_metadata()
            queue.publish_pending_metadata()
            receiver.send_metadata.assert_called_once_with("pages", 3, None)
            self.assertIsNone(entry.pending_metadata)

    def test_failed_receiver_keeps_destination_unpublished(self):
        queue, entry, _, receiver = self.make_queue()
        receiver.poll.return_value = KVPoll.Failed
        queue.publish_pending_metadata()
        receiver.send_metadata.assert_not_called()
        self.assertIsNotNone(entry.pending_metadata)

    def test_publishes_while_retracted_requests_wait(self):
        queue, _, _, receiver = self.make_queue()
        queue.resolve_deferred_releases = MagicMock()
        prealloc = SimpleNamespace(
            publish_pending_destinations=MagicMock(),
            resume_retracted_reqs=lambda: [],
            retracted_queue=[object()],
        )
        scheduler = SimpleNamespace(
            scheduler_stage_metrics=None,
            tree_cache=SimpleNamespace(linker=None),
            enable_decode_hicache=False,
            disagg_decode_prealloc_queue=prealloc,
            disagg_decode_transfer_queue=queue,
            waiting_queue=IndexedReqList(ListKind.WAITING),
        )
        SchedulerDisaggregationDecodeMixin.process_decode_queue(scheduler)
        receiver.send_metadata.assert_called_once_with("pages", 3, None)


class _GateReceiver:
    def __init__(self, poll):
        self._poll = poll
        self.require_staging = False

    def poll(self):
        return self._poll

    def clear(self):
        pass

    def failure_exception(self):
        raise KVTransferError(
            7, "Failed due to an unknown reason from another rank", True
        )


def _gated_decode_req(poll, restore_status):
    return SimpleNamespace(
        req=FakeReq(rid="r", bootstrap_room=7, return_logprob=False),
        kv_receiver=_GateReceiver(poll),
        metadata_buffer_index=3,
        hicache_restore_status=restore_status,
        pending_metadata=None,
    )


class TestDecodeHiCacheRestoreGate(CustomTestCase):
    def setUp(self):
        super().setUp()
        _run_as_single_rank(self)

    def test_gate_folds_restore_outcome_into_terminal_poll_only(self):
        cases = [
            (KVPoll.Success, HiCacheRestoreResult.READY, KVPoll.Success),
            (KVPoll.Success, HiCacheRestoreResult.PENDING, KVPoll.Transferring),
            (KVPoll.Success, HiCacheRestoreResult.FAILED, KVPoll.Failed),
            (KVPoll.Transferring, HiCacheRestoreResult.FAILED, KVPoll.Transferring),
            (
                KVPoll.WaitingForInput,
                HiCacheRestoreResult.FAILED,
                KVPoll.WaitingForInput,
            ),
            (KVPoll.Failed, HiCacheRestoreResult.PENDING, KVPoll.Failed),
        ]
        for poll, status, expected in cases:
            with self.subTest(poll=poll, status=status):
                gated = HiCacheRestoreGatedKVReceiver(_gated_decode_req(poll, status))
                self.assertEqual(gated.poll(), expected)

    @patch("sglang.srt.disaggregation.decode.prepare_abort")
    def test_pop_transferred_branches_on_reduced_poll_only(self, mock_prepare_abort):
        def make_queue(decode_req, peer_poll):
            queue = DecodeTransferQueue.__new__(DecodeTransferQueue)
            queue.queue = decode_queue(ListKind.PD_RECEIVING, [decode_req])
            queue.enable_staging = False
            queue.enable_host_receive = False
            queue.enable_deferred_kv_release = False
            queue.gloo_group = MagicMock()
            queue.req_to_metadata_buffer_idx_allocator = MagicMock()
            queue.tp_rank = 0
            queue.tree_cache = MagicMock()
            queue.metadata_buffers = SimpleNamespace(bootstrap_room=[None] * 4)
            queue.metadata_buffers.clear_slot = lambda idx: (
                queue.metadata_buffers.bootstrap_room.__setitem__(idx, 0)
            )
            queue._clean_hicache_prefetch_resources = MagicMock()
            queue._commit_transfer_to_req = MagicMock()
            queue._record_transfer_failure = MagicMock()
            scheduler = MagicMock()
            scheduler.enable_decode_hicache = True
            scheduler.enable_hisparse = False
            scheduler.metrics_reporter.enable_metrics = False
            queue.scheduler = scheduler

            def reduce_with_peer(
                pollers, group, decode_reqs=None, metadata_buffers=None
            ):
                return [min(int(p.poll()), int(peer_poll)) for p in pollers]

            patcher = patch(
                "sglang.srt.disaggregation.decode.poll_and_all_reduce",
                side_effect=reduce_with_peer,
            )
            patcher.start()
            self.addCleanup(patcher.stop)
            return queue

        decode_req = _gated_decode_req(KVPoll.Success, HiCacheRestoreResult.FAILED)
        queue = make_queue(decode_req, peer_poll=KVPoll.Failed)
        self.assertEqual(queue.pop_transferred(), [])
        self.assertEqual(list(queue.queue.items()), [])
        queue._commit_transfer_to_req.assert_not_called()
        queue._record_transfer_failure.assert_called_once_with("HiCache restore failed")
        mock_prepare_abort.assert_called_once()

        mock_prepare_abort.reset_mock()
        decode_req = _gated_decode_req(KVPoll.Success, HiCacheRestoreResult.READY)
        queue = make_queue(decode_req, peer_poll=KVPoll.Failed)
        self.assertEqual(queue.pop_transferred(), [])
        queue._commit_transfer_to_req.assert_not_called()
        queue._record_transfer_failure.assert_called_once_with(
            "Failed due to an unknown reason from another rank"
        )
        mock_prepare_abort.assert_called_once()

        mock_prepare_abort.reset_mock()
        decode_req = _gated_decode_req(KVPoll.Transferring, HiCacheRestoreResult.FAILED)
        queue = make_queue(decode_req, peer_poll=KVPoll.Success)
        self.assertEqual(queue.pop_transferred(), [])
        self.assertEqual(list(queue.queue.items()), [decode_req])
        mock_prepare_abort.assert_not_called()
        queue.scheduler.request_retirer.release.assert_not_called()


class _PollingReceiver:
    def __init__(self, poll=KVPoll.WaitingForInput):
        self.clear_called = False
        self.conclude_state = None
        self._poll = poll
        self.send_metadata = MagicMock()

    def poll(self):
        return self._poll

    def clear(self):
        self.clear_called = True

    def failure_exception(self):
        return None


def _local_reduce(flags, group, op):
    return flags


class TestDeferredDestinationPublish(CustomTestCase):
    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)
        publish(ServerArgs(model_path="dummy"), role="tokenizer")
        for target in ("stream", "current_stream", "Event"):
            patcher = patch(f"sglang.srt.disaggregation.decode.torch.cuda.{target}")
            patcher.start()
            self.addCleanup(patcher.stop)

    def _deferring_queue(self, decode_reqs):
        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue.pp_size = 1
        queue.gloo_group = object()
        queue.queue = decode_queue(ListKind.PD_PREALLOC, decode_reqs)
        queue.pending_reqs = []
        queue.retracted_queue = []
        queue.num_reserved_decode_tokens = 0
        queue._num_published_destinations = 0
        queue._defer_publish = True
        queue._pending_publish = []
        queue._publish_stream = MagicMock()
        queue._resolve_pending_reqs = MagicMock()
        queue._update_handshake_waiters = MagicMock()
        queue._uses_swa_tail_prealloc = MagicMock(return_value=False)
        queue._uses_swa_reservation = MagicMock(return_value=False)
        queue._allocatable_token_budgets = MagicMock(return_value=1000)
        queue._prealloc_reservation_fits = MagicMock(return_value=True)
        queue.req_to_token_pool = MagicMock()
        queue.req_to_token_pool.available_size.return_value = 4
        queue.req_to_token_pool.holds_state_slots = False
        queue.req_to_token_pool.req_to_token = torch.arange(32).reshape(4, 8)

        def pre_alloc(req, *args):
            req.kv.req_pool_idx = 1
            return queue.req_to_token_pool.req_to_token[1, : len(req.origin_input_ids)]

        queue._pre_alloc = MagicMock(side_effect=pre_alloc)
        queue.req_to_metadata_buffer_idx_allocator = ReqToMetadataIdxAllocator(4)
        queue.token_to_kv_pool_allocator = SimpleNamespace(
            page_size=1,
            state=None,
            translate_kv_indices_for_transfer=lambda indices: indices,
        )
        queue.kv_manager = SimpleNamespace(kv_args=SimpleNamespace(state_types=[]))
        queue.transfer_queue = SimpleNamespace(
            queue=decode_queue(ListKind.PD_RECEIVING), enable_staging=False
        )
        queue.tree_cache = MagicMock()
        queue.scheduler = MagicMock(
            enable_priority_scheduling=False,
            enable_hisparse=False,
            enable_lora=False,
            enable_decode_hicache=False,
        )
        queue.scheduler.running_batch.reqs = []
        return queue

    @staticmethod
    def _decode_req(receiver):
        req = FakeReq(
            rid="deferred",
            origin_input_ids=[1] * 5,
            output_ids=[],
            kv=ReqKvInfo(),
            finished_reason=None,
            sampling_params=SimpleNamespace(max_new_tokens=4),
            time_stats=MagicMock(),
        )
        return DecodeRequest(req=req, kv_receiver=receiver, waiting_for_input=True)

    @staticmethod
    def _pending(decode_req):
        decode_req.awaiting_publish = True
        return _PendingPublish(
            allocated=MagicMock(),
            decode_req=decode_req,
            dst_kv_indices=torch.arange(4),
            prefix_len=0,
            total_prefix_len=0,
            fill_len=4,
        )

    def test_deferred_destination_keeps_its_metadata_slot_until_published(self):
        # The transfer queue frees the metadata slot of any request it fails, so
        # a deferred request must own one before joining the queue, and the
        # publish must send that slot instead of taking a second one.
        receiver = _PollingReceiver()
        decode_req = self._decode_req(receiver)
        queue = self._deferring_queue([decode_req])

        self.assertEqual(queue.pop_preallocated(), ([decode_req], []))

        receiver.send_metadata.assert_not_called()
        self.assertTrue(decode_req.awaiting_publish)
        self.assertEqual(decode_req.metadata_buffer_index, 0)
        self.assertEqual(queue.req_to_metadata_buffer_idx_allocator.available_size(), 3)
        entry_time = decode_req.req.time_stats.set_decode_transfer_queue_entry_time
        entry_time.assert_called_once()
        # transfer_queue.extend() takes the request over.
        queue.note_destinations_queued(1)

        allocated = queue._pending_publish[0].allocated
        with patch(
            "sglang.srt.disaggregation.decode.all_reduce_flags",
            side_effect=_local_reduce,
        ):
            queue.publish_pending_destinations()

        queue._publish_stream.wait_event.assert_called_once_with(allocated)
        receiver.send_metadata.assert_called_once()
        entry_time.assert_called_once()
        page_indices, metadata_index, state_indices = (
            receiver.send_metadata.call_args.args
        )
        self.assertEqual(page_indices.tolist(), [8, 9, 10, 11, 12])
        self.assertEqual(metadata_index, 0)
        self.assertEqual(queue.req_to_metadata_buffer_idx_allocator.available_size(), 3)
        self.assertFalse(decode_req.awaiting_publish)
        self.assertEqual(queue._pending_publish, [])
        self.assertFalse(queue.has_published_destinations)

    def test_inline_publish_holds_the_send_until_the_forward_is_done(self):
        """Where the deferred publish is off (staging, HiSparse, no WAR
        barrier), overlap still reuses pages the in-flight forward may write, so
        the inline publish records an event on the forward stream and holds the
        send for the transfer queue."""
        receiver = _PollingReceiver()
        decode_req = self._decode_req(receiver)
        queue = self._deferring_queue([decode_req])
        queue._defer_publish = False
        recorded = []
        ready = SimpleNamespace(record=lambda stream: recorded.append(stream))
        queue.scheduler.enable_overlap = True
        queue.scheduler.device_module = SimpleNamespace(Event=lambda: ready)
        queue.scheduler.forward_stream = object()

        self.assertEqual(queue.pop_preallocated(), ([decode_req], []))

        receiver.send_metadata.assert_not_called()
        self.assertEqual(recorded, [queue.scheduler.forward_stream])
        self.assertFalse(decode_req.awaiting_publish)
        event, args, kwargs = decode_req.pending_metadata
        self.assertIs(event, ready)
        self.assertEqual(args[0].tolist(), [8, 9, 10, 11, 12])
        self.assertEqual(args[1], decode_req.metadata_buffer_index)
        self.assertEqual(kwargs, {"decode_prefix_len": 0})
        self.assertEqual(decode_req.metadata_buffer_index, 0)
        self.assertTrue(queue.has_published_destinations)

    def test_staging_readback_preserves_payload_and_waits_for_both_events(self):
        payloads = []
        original_empty, original_copy = torch.empty, torch.Tensor.copy_
        for master, staging in ((False, True), (True, False), (True, True)):
            receiver = _PollingReceiver()
            receiver.require_staging = True
            entry = self._decode_req(receiver)
            entry.req.bootstrap_room, entry.req.bootstrap_host = 17, "127.0.0.1"
            queue = self._deferring_queue([entry])
            queue._defer_publish = False
            queue.scheduler.enable_overlap = True
            forward_done, readback_done = MagicMock(), MagicMock()
            forward_done.query.return_value = readback_done.query.return_value = False
            queue.scheduler.device_module = SimpleNamespace(Event=lambda: forward_done)
            queue.scheduler.forward_stream = object()
            queue.transfer_queue.enable_staging = staging
            queue.transfer_queue.staging_handler = MagicMock()
            queue.token_to_kv_pool_allocator.page_size = 2
            queue.token_to_kv_pool_allocator.state_ids = lambda _: SimpleNamespace(
                ids=lambda x: x + 100
            )
            queue.req_to_token_pool.req_index_to_mamba_index_mapping = torch.arange(
                4
            ).reshape(4, 1)
            queue.kv_manager.kv_args.state_types = [StateType.MAMBA]

            def pinned_empty(shape, *, dtype, pin_memory):
                return original_empty(shape, dtype=dtype)

            def copy(host, device, *, non_blocking):
                self.assertTrue(non_blocking)
                return original_copy(host, device)

            with (
                envs.SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION.override(master),
                patch(
                    "sglang.srt.disaggregation.decode.torch.cuda.is_available",
                    return_value=True,
                ),
                patch(
                    "sglang.srt.disaggregation.decode.torch.cuda.Event",
                    return_value=readback_done,
                ),
                patch(
                    "sglang.srt.disaggregation.decode.torch.empty",
                    side_effect=pinned_empty,
                ),
                patch.object(torch.Tensor, "copy_", copy),
            ):
                queue.pop_preallocated()
            transfer = DecodeTransferQueue.__new__(DecodeTransferQueue)
            transfer.queue = decode_queue(ListKind.PD_RECEIVING, [entry])
            transfer.gloo_group = object()
            with patch(
                "sglang.srt.disaggregation.decode.all_reduce_flags",
                side_effect=_local_reduce,
            ):
                transfer.publish_pending_metadata()
                receiver.send_metadata.assert_not_called()
                forward_done.query.return_value = True
                transfer.publish_pending_metadata()
                self.assertEqual(
                    receiver.send_metadata.call_count, int(not (master and staging))
                )
                readback_done.query.return_value = True
                transfer.publish_pending_metadata()
            receiver.send_metadata.assert_called_once()
            pages, slot, states = receiver.send_metadata.call_args.args
            payloads.append(
                (
                    pages.tolist(),
                    slot,
                    [[x.tolist() for x in part] for part in states],
                    receiver.send_metadata.call_args.kwargs,
                )
            )
            self.assertIsNone(entry.pending_metadata)
        self.assertEqual(payloads[0], payloads[1])
        self.assertEqual(payloads[1], payloads[2])
        self.assertEqual(
            payloads[1], ([4, 5, 6], 0, [[[101]]], {"decode_prefix_len": 0})
        )

    def test_recompute_waits_for_held_metadata_to_be_sent(self):
        receiver = _PollingReceiver()
        decode_req = self._decode_req(receiver)
        decode_req.is_rebootstrap = True
        queue = self._deferring_queue([decode_req])
        queue._defer_publish = False
        ready = MagicMock()
        ready.query.return_value = False
        queue.scheduler.enable_overlap = True
        queue.scheduler.device_module = SimpleNamespace(Event=lambda: ready)
        queue.scheduler.disagg_decode_prealloc_queue = queue
        queue._submit_prefill_recompute = MagicMock(
            side_effect=lambda _: receiver.send_metadata.assert_called_once()
        )

        queue.pop_preallocated()
        queue._submit_prefill_recompute.assert_not_called()
        transfer = DecodeTransferQueue.__new__(DecodeTransferQueue)
        transfer.queue = decode_queue(ListKind.PD_RECEIVING, [decode_req])
        transfer.gloo_group = object()
        transfer.scheduler = queue.scheduler
        with patch(
            "sglang.srt.disaggregation.decode.all_reduce_flags",
            side_effect=_local_reduce,
        ):
            transfer.publish_pending_metadata()
            queue._submit_prefill_recompute.assert_not_called()
            ready.query.return_value = True
            transfer.publish_pending_metadata()
        queue._submit_prefill_recompute.assert_called_once_with(decode_req)

    def test_deferred_destination_closes_the_move_gate_until_queued(self):
        """A later allocation in the same pop must not let unified-memory compaction
        move a deferred request's pages before its publish reads them back."""
        queue = self._deferring_queue([self._decode_req(_PollingReceiver())])

        queue.pop_preallocated()

        self.assertTrue(queue.has_published_destinations)
        queue.note_destinations_queued(1)
        self.assertFalse(queue.has_published_destinations)

    def test_released_receiver_keeps_the_publish_reduction_aligned(self):
        """A pending request the transfer queue released keeps its slot in the
        all-reduced polls, polls as failed and is not published."""
        released = self._decode_req(None)
        live = self._decode_req(_PollingReceiver())
        queue = self._deferring_queue([])
        queue._pending_publish = [self._pending(released), self._pending(live)]
        queue._publish_destination = MagicMock()

        with patch(
            "sglang.srt.disaggregation.decode.all_reduce_flags",
            side_effect=_local_reduce,
        ) as reduce:
            queue.publish_pending_destinations()

        self.assertEqual(
            reduce.call_args.kwargs["flags"],
            [int(KVPoll.Failed), int(KVPoll.WaitingForInput)],
        )
        queue._publish_destination.assert_called_once()
        self.assertIs(queue._publish_destination.call_args.kwargs["decode_req"], live)
        self.assertTrue(released.awaiting_publish)
        self.assertFalse(live.awaiting_publish)


def _admit_req(**overrides):
    fields = dict(
        grammar=None,
        output_dsa_topk_indices=None,
        kv=SimpleNamespace(retraction_backup=None),
        hidden_states_tensor=object(),
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


class TestAdmissionOverlap(CustomTestCase):
    def test_scope_uses_admitted_requests_and_keeps_host_read_paths_stock(self):
        scheduler = SimpleNamespace(
            enable_overlap=True,
            spec_algorithm=SimpleNamespace(is_eagle=lambda: True),
            enable_hisparse=False,
        )
        disagg = SimpleNamespace(
            disaggregation_decode_host_receive_threshold=1,
            disaggregation_decode_enable_radix_cache=False,
        )
        with patch("sglang.srt.disaggregation.decode.get_disagg", return_value=disagg):
            for enabled in (False, True):
                with envs.SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION.override(enabled):
                    scope = (
                        SchedulerDisaggregationDecodeMixin._nonblocking_admit_in_scope
                    )
                    self.assertEqual(
                        scope(scheduler, SimpleNamespace(reqs=[_admit_req()])), enabled
                    )
                    self.assertFalse(
                        scope(
                            scheduler,
                            SimpleNamespace(reqs=[_admit_req(grammar=object())]),
                        )
                    )

    def test_seed_is_ordered_on_forward_stream_without_schedule_wait(self):
        for nonblocking in (False, True):
            batch = MagicMock()
            batch.is_empty.return_value = False
            calls = []

            @contextlib.contextmanager
            def forward_context():
                calls.append("enter_forward")
                yield
                calls.append("leave_forward")

            scheduler = SimpleNamespace(
                _get_new_prebuilt_batch=lambda _: batch,
                _nonblocking_admit_in_scope=lambda _: nonblocking,
                _verify_kv_checksums=lambda _: calls.append("verify"),
                ngram_embedding_manager=MagicMock(),
                chunked_req=None,
                future_map=object(),
                enable_overlap=True,
                schedule_stream=MagicMock(),
                forward_stream=MagicMock(),
                forward_stream_ctx=forward_context(),
            )
            batch.process_prebuilt.side_effect = lambda _: calls.append("seed")
            result = SchedulerDisaggregationDecodeMixin.get_new_prebuilt_batch(
                scheduler, object()
            )
            self.assertIs(result, batch)
            if nonblocking:
                self.assertEqual(
                    calls, ["verify", "enter_forward", "seed", "leave_forward"]
                )
                scheduler.schedule_stream.wait_stream.assert_not_called()
                scheduler.forward_stream.wait_stream.assert_called_once_with(
                    scheduler.schedule_stream
                )
                batch.req_pool_indices.record_stream.assert_called_once_with(
                    scheduler.forward_stream
                )
                batch.seq_lens.record_stream.assert_called_once_with(
                    scheduler.forward_stream
                )
            else:
                self.assertEqual(calls, ["verify", "seed"])
                scheduler.schedule_stream.wait_stream.assert_called_once_with(
                    scheduler.forward_stream
                )
                scheduler.forward_stream.wait_stream.assert_not_called()



class TestParallelInfoFetchEpoch(CustomTestCase):
    """The topology fetch runs on an executor, so it can outlive the heartbeat's
    decision that the prefill it was querying is dead."""

    ADDR = "prefill:8998"

    def _manager(self, payload=None, fetch=None):
        mgr = make_decode_kv_manager(self)
        mgr._resolve_rank_mapping = MagicMock()
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
