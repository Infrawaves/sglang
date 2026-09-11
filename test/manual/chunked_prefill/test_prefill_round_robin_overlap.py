"""CPU checks of overlap result/abort ordering and Mooncake source ownership.

Uses production method bodies; GPU kernels and TP collectives are stand-ins.
Run with Python 3.10+. No SGLang or torch installation required.
"""

import concurrent.futures
import threading
import unittest
from collections import defaultdict
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

from test_prefill_round_robin_runtime import (
    GLOBALS,
    Batch,
    Harness,
    Prefill,
    Req,
    extract,
)


class Tensor:
    def __init__(self, values):
        self.values = list(values)

    def tolist(self):
        return self.values[:]


TORCH = NS(
    tensor=lambda values, **kwargs: Tensor(values),
    uint8=None,
    distributed=NS(all_reduce=lambda *args, **kwargs: None, ReduceOp=NS(MIN=None)),
)


class OverlapTests(unittest.TestCase):
    def setUp(self):
        self.torch_patch = patch.dict(GLOBALS, torch=TORCH)
        self.torch_patch.start()
        self.addCleanup(self.torch_patch.stop)
        self.s = Harness()
        self.s.enable_overlap = True
        self.s.spec_algorithm = NS(is_eagle=lambda: False)
        self.s.batch_result_processor = NS(
            snapshot_auxiliary_output_starts=lambda *args: None,
            move_logprobs_to_cpu=lambda **kwargs: None,
        )
        self.s.metrics_reporter = Mock()
        self.s.maybe_send_health_check_signal = Mock()

    def result(self, batch):
        batch.spec_info = NS()
        batch.prefill_stats = None
        batch.dp_cooperation_info = None
        return NS(
            logits_output=None,
            next_token_ids=Tensor([1] * len(batch.reqs)),
            extend_input_len_per_req=None,
            extend_logprob_start_len_per_req=None,
            copy_done=Mock(),
            auxiliary_host_output=None,
            routed_experts_output=None,
            indexer_topk_output=None,
            next_draft_input=batch.spec_info,
            can_run_cuda_graph=False,
        )

    def request(self, name, middle=True):
        req = Req(name, 384, prefix=128)
        req.return_logprob = False
        req.metadata_buffer_index = 0
        req.time_stats.set_last_chunked_prefill_finish_time = Mock()
        req.time_stats.set_completion_time = Mock()
        req.inflight_middle_chunks = int(middle)
        req.disagg_kv_sender.is_source_release_safe = Mock(return_value=True)
        return req

    def test_abort_waits_for_result_and_source_then_releases_once(self):
        s = self.s
        a = self.request("a")
        b = self.request("b")
        old, new = Batch([a], a), Batch([b], b)
        result = self.result(old)
        s.result_queue = [(old, result)]
        s.chunked_req = a
        s.abort_request(NS(rid="a", abort_all=False))
        s.process_pending_chunked_abort()
        self.assertIsNone(s.chunked_req)
        self.assertEqual(a.cleanup, [])
        # Next batch launches before A's result is processed.
        s.result_queue.append((new, self.result(new)))
        s.result_queue.pop(0)
        a.disagg_kv_sender.is_source_release_safe.return_value = False
        s.process_batch_result_disagg_prefill(old, result)
        result.copy_done.synchronize.assert_called_once()
        self.assertEqual(a.inflight_middle_chunks, 0)
        s.process_pending_round_robin_actions()
        self.assertEqual(a.cleanup, [])
        a.disagg_kv_sender.is_source_release_safe.return_value = True
        s.process_pending_round_robin_actions()
        s.process_pending_round_robin_actions()
        self.assertEqual(a.cleanup.count("kv"), 1)
        self.assertEqual(s.outputs, [a])
        self.assertEqual(s.result_queue[0][0].reqs, [b])

    def test_abort_finds_final_result_without_current_or_suspended(self):
        s = self.s
        a = self.request("final", middle=False)
        batch = Batch([a])
        result = self.result(batch)
        s.result_queue = [(batch, result)]
        s.abort_request(NS(rid=a.rid, abort_all=False))
        self.assertIn(a, s._pending_round_robin_actions)
        s.process_pending_chunked_abort()
        self.assertEqual(a.cleanup, [])
        s.result_queue.pop(0)
        s.process_batch_result_disagg_prefill(batch, result)
        s.process_pending_round_robin_actions()
        self.assertEqual(s.outputs, [a])
        s.send_kv_chunk.assert_not_called()

    def test_suspended_pending_bootstrap_result_does_not_retry(self):
        s = self.s
        a = self.request("a")
        a.pending_bootstrap = True
        s.suspend(a)
        batch = Batch([a], a)
        retry = Mock()
        s.optimistic_release_and_requeue = retry
        s.process_batch_result_disagg_prefill(batch, self.result(batch))
        retry.assert_not_called()
        self.assertEqual(s.suspended_prefill_queue, [a])
        self.assertEqual(a.cleanup, [])

    def test_retry_is_deferred_and_abort_wins(self):
        s = self.s
        a = self.request("a")
        s.chunked_req = a
        batch = Batch([a], a)
        s.result_queue = [(batch, self.result(batch))]
        retry = Mock()
        s.optimistic_release_and_requeue = retry
        s.defer_round_robin_action(a, "retry")
        s.process_pending_round_robin_actions()
        retry.assert_not_called()
        s.abort_request(NS(rid=a.rid, abort_all=False))
        s.defer_round_robin_action(a, "retry")
        s.result_queue.clear()
        s.process_pending_round_robin_actions()
        retry.assert_not_called()
        self.assertEqual(s.outputs, [a])

    def test_bootstrap_failure_and_retry_use_existing_finish_paths(self):
        for action in ("bootstrap_failure", "retry"):
            with self.subTest(action=action):
                s = self.s
                a = self.request(action)
                s.chunked_req = a
                batch = Batch([a], a)
                result = self.result(batch)
                s.result_queue = [(batch, result)]
                if action == "bootstrap_failure":
                    Prefill.handle_bootstrap_failure(s, a)
                    finish = s.handle_bootstrap_failure = Mock()
                else:
                    Prefill.optimistic_release_and_requeue(s, a)
                    finish = s.optimistic_release_and_requeue = Mock()
                self.assertIsNone(s.chunked_req)
                s.process_pending_round_robin_actions()
                finish.assert_not_called()
                s.result_queue.pop(0)
                s.process_batch_result_disagg_prefill(batch, result)
                s.process_pending_round_robin_actions()
                s.process_pending_round_robin_actions()
                finish.assert_called_once_with(a, defer=False)

    def test_other_rank_can_hold_release(self):
        a = self.request("a")
        self.s.defer_round_robin_action(a, "abort")

        def hold(tensor, **kwargs):
            tensor.values = [0]

        with patch.object(TORCH.distributed, "all_reduce", hold):
            self.s.process_pending_round_robin_actions()
        self.assertEqual(a.cleanup, [])
        self.s.process_pending_round_robin_actions()
        self.assertEqual(self.s.outputs, [a])


MOON_GLOBALS = dict(
    concurrent=concurrent,
    logger=Mock(),
    KVPoll=NS(Failed=0),
    DisaggregationMode=NS(PREFILL="prefill"),
    TraceNullContext=lambda: None,
    TransferKVChunk=lambda **kwargs: NS(staging_counted=False, **kwargs),
)
Mooncake = extract(
    "disaggregation/mooncake/conn.py",
    "MooncakeKVManager",
    [
        "add_transfer_request",
        "_finish_source_transfer",
        "is_source_release_safe",
        "_await_transfer_futures",
        "transfer_worker",
    ],
    MOON_GLOBALS,
)


class SourceDrainTests(unittest.TestCase):
    def manager(self):
        m = Mooncake()
        m.track_source_transfers = True
        m._source_transfers = defaultdict(int)
        m._source_transfers_lock = threading.Lock()
        m.enable_deferred_decode_kv_release = False
        m.disaggregation_mode = "prefill"
        m.request_status = {1: 2}
        m.check_status = lambda room: m.request_status[room]
        m.transfer_infos = {1: {"host:123": None}}
        self.queued = []
        m.transfer_queues = [NS(put=self.queued.append)]
        return m

    def test_queued_and_running_tasks_both_hold_source(self):
        m = self.manager()
        for _ in range(2):
            m.add_transfer_request(1, [], slice(0, 1), False)
        self.assertEqual(len(self.queued), 2)
        self.assertFalse(m.is_source_release_safe(1))
        m._finish_source_transfer(1)
        self.assertFalse(m.is_source_release_safe(1))
        m._finish_source_transfer(1)
        self.assertTrue(m.is_source_release_safe(1))
        self.assertEqual(dict(m._source_transfers), {})

    def test_worker_discards_queued_failed_chunks_before_source_is_free(self):
        m = self.manager()
        m.enable_trace = False
        m.bootstrap_port = 1
        m._staging_outstanding = defaultdict(int)
        for _ in range(2):
            m.add_transfer_request(1, [], slice(0, 1), False)
        m.request_status[1] = 0
        seen = []

        def get():
            seen.append(m.is_source_release_safe(1))
            if self.queued:
                return self.queued.pop(0)
            raise StopIteration("end test worker")

        with self.assertRaisesRegex(RuntimeError, "end test worker"):
            m.transfer_worker(NS(get=get), None)
        self.assertEqual(seen, [False, False, True])
        self.assertEqual(dict(m._source_transfers), {})

    def test_failure_still_waits_for_running_transfer(self):
        m = self.manager()
        failed = concurrent.futures.Future()
        running = concurrent.futures.Future()
        running.set_running_or_notify_cancel()
        failed.set_result(1)
        entered = threading.Event()
        completed = threading.Event()
        result = []

        def ordered(futures):
            yield failed
            entered.set()
            yield running

        def wait():
            result.append(m._await_transfer_futures([failed, running]))
            completed.set()

        with patch.object(concurrent.futures, "as_completed", ordered):
            thread = threading.Thread(target=wait)
            thread.start()
            self.assertTrue(entered.wait(1))
            self.assertFalse(completed.is_set())
            running.set_result(0)
            thread.join(1)
        self.assertFalse(thread.is_alive())
        self.assertEqual(result, [1])


if __name__ == "__main__":
    unittest.main(verbosity=2)
