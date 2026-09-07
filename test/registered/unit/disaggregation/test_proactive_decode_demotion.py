import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.disaggregation.decode import (
    DecodePreallocQueue,
    DemotedRequest,
    SchedulerDisaggregationDecodeMixin,
)
from sglang.srt.managers.schedule_batch import Req, release_req
from sglang.srt.observability.decode_metric_collector import (
    DEFAULT_OUTPUT_LEN_BUCKETS,
    DecodeMetricCollector,
)
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_CPU_TENSOR_DISAGG = SimpleNamespace(
    disaggregation_decode_retraction_backup="cpu_tensor",
    disaggregation_decode_enable_radix_cache=False,
)


class _FakeBatch:
    """Minimal running-batch stand-in recording release_req calls."""

    def __init__(self, reqs, backup_results=None):
        self.reqs = list(reqs)
        self.batch_is_full = True
        self.release_calls = []
        self.backup_results = list(backup_results or [])

    def is_empty(self):
        return not self.reqs

    def batch_size(self):
        return len(self.reqs)

    def release_req(self, index, _, offload_kv, *, is_demoted=False):
        victim = self.reqs[index]
        if is_demoted:
            backup_saved = self.backup_results.pop(0) if self.backup_results else True
            if backup_saved:
                victim.is_demoted = True
        else:
            backup_saved = True
            victim.is_retracted = True
        self.release_calls.append((victim.rid, index, offload_kv, is_demoted))
        return backup_saved

    def filter_batch(self, keep_indices):
        self.reqs = [self.reqs[i] for i in keep_indices]


def _make_demotion_candidate(rid, seqlen, output_len, *, last_demote_output_len=0):
    return SimpleNamespace(
        rid=rid,
        seqlen=seqlen,
        output_ids=[0] * output_len,
        origin_input_ids=[0] * (seqlen - output_len),
        is_retracted=False,
        is_demoted=False,
        last_demote_output_len=last_demote_output_len,
        finished=lambda: False,
        sampling_params=SimpleNamespace(max_new_tokens=128),
        time_stats=SimpleNamespace(set_retract_time=MagicMock()),
    )


def _make_demotion_scheduler(batch, *, budget, enable_metrics=False):
    """Scheduler stub wired to a real DecodePreallocQueue so add_demoted_req
    performs its actual budget deduction."""
    prealloc_queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
    prealloc_queue.demotion_queue = []
    prealloc_queue.demoted_tokens_total = 0
    scheduler = SimpleNamespace(
        running_batch=batch,
        enable_overlap=False,
        remain_cpu_demote_tokens=budget,
        server_args=SimpleNamespace(
            proactive_demotion_max_input_len=10,
            proactive_demotion_min_output_len=11,
            proactive_decode_demotion_cache_usage=0.70,
        ),
        pool_stats_observer=SimpleNamespace(
            get_pool_stats=lambda: SimpleNamespace(get_max_pool_usage=lambda: 0.96)
        ),
        disagg_decode_prealloc_queue=prealloc_queue,
        new_token_ratio_tracker=SimpleNamespace(current=0.0),
        metrics_reporter=SimpleNamespace(
            num_demoted_reqs=0,
            enable_metrics=enable_metrics,
            metrics_collector=MagicMock(),
        ),
        req_to_token_pool=MagicMock(),
        token_to_kv_pool_allocator=MagicMock(),
        tree_cache=MagicMock(),
        hisparse_coordinator=None,
    )
    prealloc_queue.scheduler = scheduler
    # SimpleNamespace stands in for the mixin, so bind the helper the loop calls.
    scheduler._decode_pool_above_demotion_threshold = SchedulerDisaggregationDecodeMixin._decode_pool_above_demotion_threshold.__get__(
        scheduler
    )
    return scheduler


class _PumpTree:
    def __init__(self, *, pending=False):
        self.check_calls = 0
        self.retraction_ssd_backups = {1: object()} if pending else {}

    def check_hicache_events(self):
        self.check_calls += 1


def _make_pump_scheduler(tree, *, enable_decode_hicache):
    scheduler = SimpleNamespace(
        enable_decode_hicache=enable_decode_hicache,
        tree_cache=tree,
        running_batch=None,
        decode_offload_manager=SimpleNamespace(check_offload_progress=lambda: None),
        disagg_decode_transfer_queue=SimpleNamespace(
            resolve_deferred_releases=lambda: None,
            extend=lambda _: None,
            pop_transferred=lambda: [],
        ),
        disagg_decode_prealloc_queue=SimpleNamespace(
            resume_retracted_reqs=lambda: [],
            retracted_queue=[],
            pop_preallocated=lambda: ([], None),
        ),
        resume_demote_reqs=lambda: [],
        waiting_queue=[],
        need_to_proactive_retract_request=lambda: False,
        enable_hisparse=False,
        polling_count=0,
        polling_interval=2,
        scheduler_stage_metrics=None,
    )
    return scheduler


class TestProactiveDecodeDemotion(CustomTestCase):
    def test_decode_queue_pumps_hicache_only_while_retraction_write_pending(self):
        """With decode HiCache off, the SSD path enters the HiCache event
        phase only while a retraction write awaits its ack; an idle SSD
        deployment must not pay a collective per step."""
        ssd = SimpleNamespace(
            disaggregation_decode_retraction_backup="ssd",
            disaggregation_decode_enable_offload_kvcache=False,
            disaggregation_decode_polling_interval=2,
            enable_proactive_decode_demotion=False,
        )
        for pending, expected in ((False, 0), (True, 1)):
            tree = _PumpTree(pending=pending)
            scheduler = _make_pump_scheduler(tree, enable_decode_hicache=False)
            with patch("sglang.srt.disaggregation.decode.get_disagg", return_value=ssd):
                SchedulerDisaggregationDecodeMixin.process_decode_queue(scheduler)
            self.assertEqual(tree.check_calls, expected, f"pending={pending}")

    def test_decode_queue_pumps_hicache_when_enabled(self):
        cpu = SimpleNamespace(
            disaggregation_decode_retraction_backup="cpu_tensor",
            disaggregation_decode_enable_offload_kvcache=False,
            disaggregation_decode_polling_interval=2,
            enable_proactive_decode_demotion=False,
        )
        tree = _PumpTree()
        scheduler = _make_pump_scheduler(tree, enable_decode_hicache=True)

        with patch("sglang.srt.disaggregation.decode.get_disagg", return_value=cpu):
            SchedulerDisaggregationDecodeMixin.process_decode_queue(scheduler)

        self.assertEqual(tree.check_calls, 1)

    def test_decode_queue_skips_hicache_when_neither_source_active(self):
        """Guards the negative branch: with hicache off and a non-ssd backup
        there is nothing to drain, so the gate must not degrade to always-true."""
        cpu = SimpleNamespace(
            disaggregation_decode_retraction_backup="cpu_tensor",
            disaggregation_decode_enable_offload_kvcache=False,
            disaggregation_decode_polling_interval=2,
            enable_proactive_decode_demotion=False,
        )
        tree = _PumpTree()
        scheduler = _make_pump_scheduler(tree, enable_decode_hicache=False)

        with patch("sglang.srt.disaggregation.decode.get_disagg", return_value=cpu):
            SchedulerDisaggregationDecodeMixin.process_decode_queue(scheduler)

        self.assertEqual(tree.check_calls, 0)

    def test_req_tracks_demote_separately_from_retract(self):
        demoted_req = Req(
            "demoted", "", array("q", [1]), SamplingParams(max_new_tokens=8)
        )
        demoted_req.reset_for_retract(is_demoted=True)
        self.assertTrue(demoted_req.is_demoted)
        self.assertFalse(demoted_req.is_retracted)
        self.assertEqual(demoted_req.retraction_count, 0)

        retracted_req = Req(
            "retracted", "", array("q", [1]), SamplingParams(max_new_tokens=8)
        )
        retracted_req.reset_for_retract()
        self.assertFalse(retracted_req.is_demoted)
        self.assertTrue(retracted_req.is_retracted)
        self.assertEqual(retracted_req.retraction_count, 1)

    def test_failed_demoted_backup_preserves_kv_and_request_state(self):
        req = SimpleNamespace(
            finished=lambda: False,
            kv=SimpleNamespace(),
            reset_for_retract=MagicMock(),
        )
        disagg = SimpleNamespace(
            disaggregation_mode="decode",
            disaggregation_decode_retraction_backup="ssd",
        )
        with (
            patch("sglang.srt.managers.schedule_batch.get_disagg", return_value=disagg),
            patch(
                "sglang.srt.managers.schedule_batch.retraction_backup",
                return_value=None,
            ),
            patch("sglang.srt.managers.schedule_batch.release_kv_cache") as release_kv,
        ):
            self.assertFalse(
                release_req(
                    req=req,
                    remaing_req_count=1,
                    req_to_token_pool=MagicMock(),
                    token_to_kv_pool_allocator=MagicMock(),
                    tree_cache=MagicMock(),
                    hisparse_coordinator=None,
                    is_demoted=True,
                )
            )
        release_kv.assert_not_called()
        req.reset_for_retract.assert_not_called()

    def test_collector_uses_generation_metric_buckets(self):
        self.assertEqual(DEFAULT_OUTPUT_LEN_BUCKETS[-1], 1_100_000)
        self.assertEqual(len(DEFAULT_OUTPUT_LEN_BUCKETS), 35)

    def test_collector_window_and_quantiles(self):
        now = [0.0]
        collector = DecodeMetricCollector(
            bucket_bounds=[0, 1, 2, 4, 8, 16, 32],
            clock=lambda: now[0],
        )
        for length in (1, 2, 4, 8, 8, 32, 32, 32, 32, 32):
            collector.observe_output_len(length)
        self.assertIsNone(collector.maybe_update())
        now[0] = 15.0
        self.assertEqual(collector.maybe_update(), (8, 32))
        self.assertIsNone(collector.maybe_update())

        collector.observe_output_len(1)
        now[0] = 30.0
        self.assertEqual(collector.maybe_update(), (1, 1))

    def test_server_args_defaults_and_validation(self):
        args = ServerArgs(model_path="dummy")
        args.resolve_once()
        self.assertFalse(args.enable_proactive_decode_demotion)
        self.assertEqual(args.proactive_decode_demotion_cache_usage, 0.70)
        self.assertEqual(args.proactive_safe_cpu_demote_cache_usage, 0.2)
        self.assertEqual(args.candidate_demotion_output_len_threthold, 2.0)
        self.assertEqual(args.proactive_demotion_max_input_len, 4096)
        self.assertEqual(args.proactive_demotion_min_output_len, 8192)
        self.assertEqual(args.proactive_demotion_recovery_duration, 180.0)

        with self.assertRaises(ValueError):
            ServerArgs(
                model_path="dummy",
                disaggregation_mode="decode",
                proactive_decode_demotion_cache_usage=1.1,
            ).resolve_once()

        with self.assertRaises(ValueError):
            ServerArgs(
                model_path="dummy",
                disaggregation_mode="decode",
                proactive_safe_cpu_demote_cache_usage=0.0,
            ).resolve_once()

        with self.assertRaises(ValueError):
            ServerArgs(
                model_path="dummy",
                disaggregation_mode="decode",
                proactive_demotion_max_input_len=0,
            ).resolve_once()

        with self.assertRaises(ValueError):
            ServerArgs(
                model_path="dummy",
                disaggregation_mode="decode",
                proactive_demotion_min_output_len=0,
            ).resolve_once()

    def _make_demoted_queue(self, req, recovery_duration):
        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue.demotion_queue = [
            DemotedRequest(req=req, demoted_start_time=0.0, demoted_tokens=7)
        ]
        queue.demoted_tokens_total = 7
        queue.scheduler = SimpleNamespace(
            server_args=SimpleNamespace(
                proactive_demotion_recovery_duration=recovery_duration
            ),
            remain_cpu_demote_tokens=0,
        )
        queue.req_to_token_pool = SimpleNamespace(available_size=lambda: 1)
        queue.token_to_kv_pool_allocator = MagicMock()
        queue.tree_cache = MagicMock()
        queue._uses_swa_tail_prealloc = lambda: False
        queue._allocatable_token_budgets = lambda **_: 10
        queue._prealloc_required_tokens = lambda _: (1, 1)
        queue._pre_alloc = MagicMock()
        return queue

    def _make_demoted_queue_with_entries(self, entries, recovery_duration):
        queue = self._make_demoted_queue(entries[0][0], recovery_duration)
        queue.demotion_queue = [
            DemotedRequest(
                req=req,
                demoted_start_time=start_time,
                demoted_tokens=demoted_tokens,
            )
            for req, start_time, demoted_tokens in entries
        ]
        queue.demoted_tokens_total = sum(tokens for _, _, tokens in entries)
        return queue

    def test_demoted_recovery_decision_respects_slots_and_full_budget(self):
        reqs = [SimpleNamespace(is_demoted=True) for _ in range(3)]
        for slots, budget, expected in [
            (0, 10, []),
            (1, 10, [0]),
            (3, 3, []),
            (3, 4, [0]),
            (3, 8, [0, 1]),
        ]:
            with self.subTest(slots=slots, budget=budget):
                queue = self._make_demoted_queue_with_entries(
                    [(req, 0.0, 7) for req in reqs], recovery_duration=0.0
                )
                queue.req_to_token_pool.available_size = lambda: slots
                queue._allocatable_token_budgets = lambda **_: budget
                queue._prealloc_required_tokens = lambda _: (4, 4)

                with patch(
                    "sglang.srt.disaggregation.decode.get_disagg",
                    return_value=_CPU_TENSOR_DISAGG,
                ):
                    self.assertEqual(
                        queue.get_demoted_req_indices_to_resume(), expected
                    )
                queue._pre_alloc.assert_not_called()

    def _make_demoted_swa_queue(self, fill_lens, capacity, reserved_tokens):
        reqs = [
            SimpleNamespace(
                origin_input_ids=[0] * fill_len,
                output_ids=[],
                is_demoted=True,
            )
            for fill_len in fill_lens
        ]
        queue = self._make_demoted_queue_with_entries(
            [(req, 0.0, fill_len) for req, fill_len in zip(reqs, fill_lens)],
            recovery_duration=0.0,
        )
        queue._uses_swa_tail_prealloc = lambda: True
        del queue._allocatable_token_budgets
        del queue._prealloc_required_tokens
        queue.num_reserved_decode_tokens = reserved_tokens
        queue.req_to_token_pool.available_size = lambda: len(reqs)
        queue.token_to_kv_pool_allocator = SimpleNamespace(
            page_size=16,
            size_swa=capacity,
            full_available_size=lambda: 4096,
            swa_available_size=lambda: capacity,
        )
        queue.tree_cache.swa_evictable_size.return_value = 0
        queue.scheduler.server_args.disaggregation_decode_enable_radix_cache = False
        queue.scheduler.running_batch = SimpleNamespace(reqs=[])
        queue.scheduler.waiting_queue = []
        queue.scheduler.last_batch = None
        queue.scheduler.enable_hisparse = False
        queue.scheduler.sliding_window_size = 64
        queue.transfer_queue = SimpleNamespace(queue=[])
        queue.retracted_queue = []
        return queue

    def test_demoted_recovery_accounts_for_pending_swa_pages_and_growth(self):
        for fill_lens, capacity, reserved, disable_radix, expected in [
            ([64, 64], 96, 16, True, [0]),
            ([64, 64], 128, 16, True, [0, 1]),
            ([16, 64], 80, 16, True, [0]),
            ([64, 64], 128, 16, False, [0]),
            ([17, 17, 17], 80, 0, True, [0, 1]),
        ]:
            with self.subTest(
                fill_lens=fill_lens, capacity=capacity, disable_radix=disable_radix
            ):
                queue = self._make_demoted_swa_queue(fill_lens, capacity, reserved)
                with (
                    patch(
                        "sglang.srt.disaggregation.decode.get_memory",
                        return_value=SimpleNamespace(disable_radix_cache=disable_radix),
                    ),
                    patch(
                        "sglang.srt.disaggregation.decode.get_disagg",
                        return_value=_CPU_TENSOR_DISAGG,
                    ),
                ):
                    self.assertEqual(
                        queue.get_demoted_req_indices_to_resume(), expected
                    )
                queue._pre_alloc.assert_not_called()
                self.assertEqual(len(queue.demotion_queue), len(fill_lens))
                self.assertEqual(queue.scheduler.remain_cpu_demote_tokens, 0)

    def test_demoted_recovery_uses_tp_consensus(self):
        for rank, indices in [(0, [1]), (1, [1]), (0, []), (1, [])]:
            with self.subTest(rank=rank, indices=indices):
                first, second = [SimpleNamespace(is_demoted=True) for _ in range(2)]
                queue = self._make_demoted_queue_with_entries(
                    [(first, 10.0, 7), (second, 0.0, 11)], recovery_duration=10.0
                )
                decision = MagicMock(wraps=queue.get_demoted_req_indices_to_resume)
                queue.get_demoted_req_indices_to_resume = decision
                group = SimpleNamespace(
                    rank_in_group=rank,
                    broadcast_object=MagicMock(return_value=indices),
                )
                scheduler = SimpleNamespace(
                    dp_tp_group=group,
                    disagg_decode_prealloc_queue=queue,
                    waiting_queue=[],
                )

                with patch(
                    "sglang.srt.disaggregation.decode.time.monotonic",
                    return_value=15.0 if indices else 5.0,
                ), patch(
                    "sglang.srt.disaggregation.decode.get_disagg",
                    return_value=_CPU_TENSOR_DISAGG,
                ), patch("sglang.srt.disaggregation.decode.retraction_restore") as restore:
                    resumed = SchedulerDisaggregationDecodeMixin.resume_demote_reqs(
                        scheduler
                    )

                group.broadcast_object.assert_called_once_with(
                    indices if rank == 0 else None, src=0
                )
                self.assertEqual(decision.call_count, int(rank == 0))
                self.assertEqual(resumed, [second] if indices else [])
                self.assertEqual(scheduler.waiting_queue, resumed)
                self.assertEqual(
                    [entry.req for entry in queue.demotion_queue],
                    [first] if indices else [first, second],
                )
                self.assertTrue(first.is_demoted)
                self.assertEqual(second.is_demoted, not indices)
                self.assertEqual(
                    queue.scheduler.remain_cpu_demote_tokens, 11 if indices else 0
                )
                self.assertEqual(restore.call_count, len(indices))

    def test_demoted_request_waits_then_restores(self):
        req = SimpleNamespace(is_retracted=False, is_demoted=True)
        queue = self._make_demoted_queue(req, recovery_duration=10.0)
        with patch(
            "sglang.srt.disaggregation.decode.time.monotonic", return_value=5.0
        ):
            indices = queue.get_demoted_req_indices_to_resume()
            self.assertEqual(indices, [])
            self.assertEqual(queue.resume_demote_reqs(indices), [])
        self.assertEqual(len(queue.demotion_queue), 1)
        self.assertEqual(queue.scheduler.remain_cpu_demote_tokens, 0)

        with patch(
            "sglang.srt.disaggregation.decode.time.monotonic", return_value=15.0
        ), patch(
            "sglang.srt.disaggregation.decode.get_disagg",
            return_value=_CPU_TENSOR_DISAGG,
        ), patch("sglang.srt.disaggregation.decode.retraction_restore") as restore:
            indices = queue.get_demoted_req_indices_to_resume()
            self.assertEqual(indices, [0])
            queue._pre_alloc.assert_not_called()
            self.assertTrue(req.is_demoted)
            self.assertEqual(queue.resume_demote_reqs(indices), [req])
        self.assertEqual(queue.demotion_queue, [])
        self.assertFalse(req.is_retracted)
        self.assertFalse(req.is_demoted)
        restore.assert_called_once()
        # Resume returns the demoted tokens to the CPU offload budget.
        self.assertEqual(queue.scheduler.remain_cpu_demote_tokens, 7)

    def test_release_memory_occupation_returns_budget(self):
        """Dropping a demoted CPU backup must return its tokens to the budget,
        or the budget leaks and demotion locks up permanently."""
        req = SimpleNamespace(is_retracted=False, is_demoted=True)
        queue = self._make_demoted_queue(req, recovery_duration=10.0)
        queue.queue = []
        queue.retracted_queue = []
        queue.kv_manager = SimpleNamespace()
        queue._cancel_prefill_dp_rank_queries = lambda: None
        with patch(
            "sglang.srt.disaggregation.decode.get_disagg",
            return_value=_CPU_TENSOR_DISAGG,
        ), patch("sglang.srt.disaggregation.decode.retraction_discard") as discard:
            queue.release_memory_occupation()
        discard.assert_called_once()
        self.assertEqual(queue.demotion_queue, [])
        self.assertEqual(queue.scheduler.remain_cpu_demote_tokens, 7)

    def _make_retract_check_scheduler(self, remain_cpu_demote_tokens):
        return SimpleNamespace(
            decode_metric_collector=SimpleNamespace(maybe_update=lambda: None),
            remain_cpu_demote_tokens=remain_cpu_demote_tokens,
            pool_stats_observer=SimpleNamespace(
                get_pool_stats=lambda: SimpleNamespace(
                    get_max_pool_usage=lambda: 0.96
                )
            ),
            server_args=SimpleNamespace(
                proactive_decode_demotion_cache_usage=0.70,
            ),
        )

    def test_demotion_gated_by_cpu_budget_not_queue(self):
        """The old gate blocked any new wave while the demotion queue was
        non-empty, so a mostly-recovered queue still froze demotion; the gate
        must instead track the remaining CPU token budget."""
        check = SchedulerDisaggregationDecodeMixin.need_to_proactive_retract_request
        with patch(
            "sglang.srt.disaggregation.decode.get_disagg",
            return_value=SimpleNamespace(enable_proactive_decode_demotion=True),
        ) as disagg:
            self.assertFalse(check(self._make_retract_check_scheduler(0)))
            self.assertFalse(check(self._make_retract_check_scheduler(-5)))
            # Budget remaining allows a new wave even mid-recovery.
            self.assertTrue(check(self._make_retract_check_scheduler(100)))
            disagg.return_value.enable_proactive_decode_demotion = False
            self.assertFalse(check(self._make_retract_check_scheduler(100)))

    def test_triggers_without_output_len_quantiles(self):
        """The fixed rule must fire on cache pressure alone; an empty quantile
        window (no completed requests yet) must not suppress demotion."""
        scheduler = self._make_retract_check_scheduler(100)
        scheduler.decode_metric_collector = SimpleNamespace(
            maybe_update=lambda: (None, None)
        )
        with patch(
            "sglang.srt.disaggregation.decode.get_disagg",
            return_value=SimpleNamespace(enable_proactive_decode_demotion=True),
        ):
            self.assertTrue(
                SchedulerDisaggregationDecodeMixin.need_to_proactive_retract_request(
                    scheduler
                )
            )

    def test_proactive_demotion_filters_and_spends_budget(self):
        short = _make_demotion_candidate("short", 10, 5)
        medium = _make_demotion_candidate("medium", 20, 11)
        long = _make_demotion_candidate("long", 30, 30)
        batch = _FakeBatch([long, short, medium])
        scheduler = _make_demotion_scheduler(batch, budget=40, enable_metrics=True)

        with patch(
            "sglang.srt.disaggregation.decode.get_disagg",
            return_value=_CPU_TENSOR_DISAGG,
        ):
            self.assertTrue(
                SchedulerDisaggregationDecodeMixin.proactively_demote_longest_request(
                    scheduler
                )
            )
        demotion_queue = scheduler.disagg_decode_prealloc_queue.demotion_queue
        self.assertEqual(batch.reqs, [short])
        self.assertEqual([entry.req for entry in demotion_queue], [long, medium])
        self.assertEqual(
            [entry.demoted_tokens for entry in demotion_queue], [30, 20]
        )
        # 40 - 30 - 20: the last victim may overshoot the budget by one request.
        self.assertEqual(scheduler.remain_cpu_demote_tokens, -10)
        self.assertFalse(long.is_retracted)
        self.assertTrue(long.is_demoted)
        self.assertFalse(medium.is_retracted)
        self.assertTrue(medium.is_demoted)
        self.assertEqual(
            batch.release_calls,
            [("long", 0, True, True), ("medium", 1, True, True)],
        )
        self.assertEqual(scheduler.metrics_reporter.num_demoted_reqs, 2)
        scheduler.metrics_reporter.metrics_collector.increment_demoted_reqs.assert_called_once_with(
            num_demoted_reqs=2,
            num_demoted_input_tokens=9,
            num_demoted_output_tokens=41,
        )

    def test_demotion_stops_when_budget_exhausted(self):
        """The wave must stop on budget exhaustion even while over-long
        candidates remain; without the budget check it drained every
        candidate and the CPU backup grew past the configured cap."""
        medium = _make_demotion_candidate("medium", 20, 11)
        long = _make_demotion_candidate("long", 30, 30)
        batch = _FakeBatch([long, medium])
        scheduler = _make_demotion_scheduler(batch, budget=25)

        with patch(
            "sglang.srt.disaggregation.decode.get_disagg",
            return_value=_CPU_TENSOR_DISAGG,
        ):
            self.assertTrue(
                SchedulerDisaggregationDecodeMixin.proactively_demote_longest_request(
                    scheduler
                )
            )
        # long (seqlen 30) exhausts the budget of 25; medium stays running.
        demotion_queue = scheduler.disagg_decode_prealloc_queue.demotion_queue
        self.assertEqual([entry.req for entry in demotion_queue], [long])
        self.assertEqual(batch.reqs, [medium])
        self.assertEqual(scheduler.remain_cpu_demote_tokens, -5)
        self.assertFalse(medium.is_demoted)

    def test_redemotion_requires_incremental_output(self):
        """Re-demotion must require min_output_len tokens generated since the
        last demotion; comparing total output length alone re-demoted a
        just-recovered request that had generated almost nothing new."""
        # Total output 30 but only 5 new tokens since the last demotion.
        recovered = _make_demotion_candidate(
            "recovered", 35, 30, last_demote_output_len=25
        )
        fresh = _make_demotion_candidate("fresh", 33, 30)
        batch = _FakeBatch([recovered, fresh])
        scheduler = _make_demotion_scheduler(batch, budget=100)

        with patch(
            "sglang.srt.disaggregation.decode.get_disagg",
            return_value=_CPU_TENSOR_DISAGG,
        ):
            self.assertTrue(
                SchedulerDisaggregationDecodeMixin.proactively_demote_longest_request(
                    scheduler
                )
            )
        demotion_queue = scheduler.disagg_decode_prealloc_queue.demotion_queue
        self.assertEqual([entry.req for entry in demotion_queue], [fresh])
        self.assertEqual(batch.reqs, [recovered])
        self.assertFalse(recovered.is_demoted)
        # Demotion records the output length the next wave must build on.
        self.assertEqual(fresh.last_demote_output_len, 30)

    def test_demotion_queue_cache_usage_matches_budget_spend(self):
        """demotion_queue_cache_usage must equal the budget debited by
        add_demoted_req divided by the pool size, so the gauge tracks the
        proactive_safe_cpu_demote_cache_usage cap without separate accounting."""
        long = _make_demotion_candidate("long", 30, 20)
        medium = _make_demotion_candidate("medium", 20, 12)
        batch = _FakeBatch([long, medium])
        initial_budget = 100
        scheduler = _make_demotion_scheduler(batch, budget=initial_budget)
        queue = scheduler.disagg_decode_prealloc_queue
        queue.max_total_num_tokens = 500

        with patch(
            "sglang.srt.disaggregation.decode.get_disagg",
            return_value=_CPU_TENSOR_DISAGG,
        ):
            self.assertEqual(queue.demotion_queue_cache_usage(), 0.0)
        self.assertEqual(queue.demoted_reqs(), [])

        with patch(
            "sglang.srt.disaggregation.decode.get_disagg",
            return_value=_CPU_TENSOR_DISAGG,
        ):
            self.assertTrue(
                SchedulerDisaggregationDecodeMixin.proactively_demote_longest_request(
                    scheduler
                )
            )

        spent = initial_budget - scheduler.remain_cpu_demote_tokens
        with patch(
            "sglang.srt.disaggregation.decode.get_disagg",
            return_value=_CPU_TENSOR_DISAGG,
        ):
            self.assertEqual(
                queue.demotion_queue_cache_usage(), spent / queue.max_total_num_tokens
            )
        self.assertEqual(queue.demoted_reqs(), [long, medium])

    def test_ssd_failed_backup_keeps_victim_and_ends_wave(self):
        """L2 is exhausted after reclaim, so a smaller candidate would only
        repeat the eviction pass: the wave stops with the victim running."""
        long = _make_demotion_candidate("long", 30, 20)
        short = _make_demotion_candidate("short", 20, 12)
        batch = _FakeBatch([long, short], backup_results=[False, True])
        scheduler = _make_demotion_scheduler(batch, budget=100)
        ssd = SimpleNamespace(disaggregation_decode_retraction_backup="ssd")

        with patch("sglang.srt.disaggregation.decode.get_disagg", return_value=ssd):
            self.assertFalse(
                SchedulerDisaggregationDecodeMixin.proactively_demote_longest_request(
                    scheduler
                )
            )

        self.assertEqual(batch.reqs, [long, short])
        self.assertFalse(long.is_demoted)
        self.assertEqual(scheduler.disagg_decode_prealloc_queue.demoted_reqs(), [])
        self.assertFalse(hasattr(long, "to_finish"))

    def test_ssd_wave_stops_once_usage_drops_below_threshold(self):
        """SSD has no token budget; the wave must stop on the live GPU usage
        reading, or one step would demote every candidate."""
        long = _make_demotion_candidate("long", 30, 20)
        short = _make_demotion_candidate("short", 20, 12)
        batch = _FakeBatch([long, short])
        scheduler = _make_demotion_scheduler(batch, budget=100)
        usage = iter([0.60])
        scheduler.pool_stats_observer = SimpleNamespace(
            get_pool_stats=lambda: SimpleNamespace(
                get_max_pool_usage=lambda: next(usage)
            )
        )
        ssd = SimpleNamespace(disaggregation_decode_retraction_backup="ssd")

        with patch("sglang.srt.disaggregation.decode.get_disagg", return_value=ssd):
            self.assertTrue(
                SchedulerDisaggregationDecodeMixin.proactively_demote_longest_request(
                    scheduler
                )
            )

        self.assertEqual(batch.reqs, [short])
        self.assertEqual(scheduler.disagg_decode_prealloc_queue.demoted_reqs(), [long])

    def test_host_pool_failed_backup_aborts_victim(self):
        """The abort branch must resolve _make_abort_req; it lives in
        scheduler, which imports this module, so an unguarded name here
        raised NameError and killed the scheduler on the first failure."""
        victim = _make_demotion_candidate("victim", 30, 20)
        batch = _FakeBatch([victim], backup_results=[False])
        scheduler = _make_demotion_scheduler(batch, budget=100)
        send_output = MagicMock()
        scheduler.ipc_channels = SimpleNamespace(
            send_to_tokenizer=SimpleNamespace(send_output=send_output)
        )
        host_pool = SimpleNamespace(disaggregation_decode_retraction_backup="host_pool")
        abort_req = object()

        with (
            patch(
                "sglang.srt.disaggregation.decode.get_disagg", return_value=host_pool
            ),
            patch(
                "sglang.srt.managers.scheduler._make_abort_req", return_value=abort_req
            ),
        ):
            self.assertFalse(
                SchedulerDisaggregationDecodeMixin.proactively_demote_longest_request(
                    scheduler
                )
            )

        self.assertEqual(batch.reqs, [])
        self.assertTrue(hasattr(victim, "to_finish"))
        send_output.assert_called_once_with(abort_req, victim)


if __name__ == "__main__":
    unittest.main()
