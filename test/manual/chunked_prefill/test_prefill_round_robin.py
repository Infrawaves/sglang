"""Core round-robin scheduling and overlap cleanup regressions.

Run: python3 test/manual/chunked_prefill/test_prefill_round_robin.py
Requires Python 3.10+. Executes production method bodies with CPU-only doubles;
GPU execution, real TP collectives and RDMA require deployment tests.
"""

import ast
import logging
import os
import unittest
from array import array
from contextlib import nullcontext
from enum import Enum, auto
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[3] / "python/sglang/srt"


def extract(path, class_name, names, namespace, extra=()):
    names = names.split()
    tree = ast.parse((ROOT / path).read_text())
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name
    )
    cls.bases, cls.keywords, cls.decorator_list = [], [], []
    cls.body = [
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names
    ]
    assert {n.name for n in cls.body} == set(names)
    for n in cls.body:
        n.decorator_list = []
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    extras = [
        n
        for n in tree.body
        if (isinstance(n, ast.ClassDef) and n.name in extra)
        or (
            isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id in extra for t in n.targets)
        )
    ]
    exec(
        compile(
            ast.fix_missing_locations(
                ast.Module(body=[future, *extras, cls], type_ignores=[])
            ),
            str(ROOT / path),
            "exec",
        ),
        namespace,
    )
    return namespace[class_name]


ADDER_GLOBALS = {"os": os, "Enum": Enum, "auto": auto}
PrefillAdder = extract(
    "managers/schedule_policy.py",
    "PrefillAdder",
    """ceil_paged_tokens budget_state _update_prefill_budget add_chunked_req
    add_one_req add_one_req_ignore_eos _round_robin_can_admit""",
    ADDER_GLOBALS,
    extra=("AddReqResult", "CLIP_MAX_NEW_TOKENS", "IGNORE_EOS_RESERVE_TOKENS"),
)
Result = ADDER_GLOBALS["AddReqResult"]


class Adder(PrefillAdder):
    def __init__(self, free, quantum):
        self.round_robin_requests = None
        self.free = free
        self.page_size = 64
        self.rem_total_token_offset = self.cur_rem_token_offset = 0
        self.rem_input_tokens = self.rem_chunk_tokens = quantum
        self.new_token_ratio = 1.0
        self.tree_cache = NS(disable=True)
        self.running_batch = NS(reqs=[])
        self.dllm_config = self.prefill_delayer_single_pass = None
        self.prefill_max_requests = self.rem_mamba_slots = None
        self.is_hybrid_swa = False
        self.can_run_list = []
        self.new_chunked_req = self.req_states = None
        self.log_hit_tokens = self.log_input_tokens = 0
        self.reprocessed_log_hit_tokens = self.reprocessed_log_input_tokens = 0

    @property
    def rem_total_tokens(self):
        return self.free - self.rem_total_token_offset

    @property
    def cur_rem_tokens(self):
        return self.free - self.cur_rem_token_offset

    def _mamba_gap_budget_for_req(self, req):
        return 0

    def _lock_node(self, node):
        return nullcontext()

    def _check_prefill_tile_budget(self, length):
        return None

    def _req_inc_lock_ref(self, req):
        pass

    def _account_prefill_cache_admission(self, req, prefix):
        pass


SCHEDULE = NS(
    prefill_max_requests=None, enable_mixed_chunk=False, enable_dynamic_chunking=False
)
PARALLEL = NS(pp_max_micro_batch_size=32)
MODE = NS(PREFILL="prefill", DECODE="decode", NULL="null")


class Batch:
    def __init__(self, reqs=(), chunked_req=None):
        self.reqs = list(reqs)
        self.chunked_req = chunked_req
        self.batch_is_full = False
        self.forward_mode = NS(is_extend=lambda: True)

    @classmethod
    def init_new(cls, reqs, *args, chunked_req=None):
        return cls(reqs, chunked_req)

    def is_empty(self):
        return not self.reqs

    def batch_size(self):
        return len(self.reqs)

    def filter_batch(self, chunked_req_to_exclude):
        self.reqs = [r for r in self.reqs if r not in chunked_req_to_exclude]

    def prepare_for_extend(self):
        pass


def make_adder(page, cache, allocator, running, ratio, budget, chunk, *args, **kwargs):
    adder = Adder(allocator.free, chunk)
    adder.rem_input_tokens = budget
    adder.round_robin_requests = kwargs["round_robin_requests"]
    adder.preempt_list = []
    return adder


def cache_chunk(req, cache, **kwargs):
    req.prefix_indices = range(req.kv.kv_allocated_len)


GLOBALS = dict(
    array=array,
    get_schedule=lambda: SCHEDULE,
    get_parallel=lambda: PARALLEL,
    get_memory=lambda: NS(enable_flexkv=False),
    get_disagg=lambda: NS(
        disaggregation_mode="prefill",
        disaggregation_transfer_backend="mooncake",
        optimistic_prefill_attempts=2,
    ),
    get_exec=lambda: NS(dllm=NS(dllm_algorithm=None)),
    TEST_RETRACT=False,
    PrefillAdder=make_adder,
    AddReqResult=Result,
    ScheduleBatch=Batch,
    PrefillStats=NS(from_adder=lambda *args, **kwargs: None),
    set_time_batch=lambda *args: None,
    DisaggregationMode=MODE,
    maybe_cache_unfinished_req=cache_chunk,
    is_aborted=lambda r: r.aborted or r.to_finish is not None,
    prepare_abort=lambda r, reason: setattr(r, "aborted", True),
    maybe_release_metadata_buffer=lambda req, allocator: req.cleanup.append("metadata"),
    release_kv_cache=lambda req, cache, **kwargs: req.cleanup.append("kv"),
    _make_abort_req=lambda req, **kwargs: req.rid,
    logger=logging.getLogger(__name__),
    FINISH_ABORT=lambda *args: args or "abort",
    HTTPStatus=NS(SERVICE_UNAVAILABLE=503),
    KVPoll=NS(
        Failed="failed",
        WaitingForInput="ready",
        Transferring="transferring",
        Success="success",
    ),
    should_force_retry=lambda r: False,
    poll_and_all_reduce_attn_cp_tp_group=lambda senders, *args: [
        s.poll for s in senders
    ],
)
Scheduler = extract(
    "managers/scheduler.py",
    "Scheduler",
    """_validate_prefill_round_robin enqueue_prefill_ready reset_prefill_ready_seq_if_idle
    iter_round_robin_requests remove_prefill_ready_requests
    _get_new_batch_prefill_raw process_pending_chunked_abort
    _release_chunked_abort abort_request pause_generation _poll_timeout_aborts""",
    GLOBALS,
)
Prefill = extract(
    "disaggregation/prefill.py",
    "SchedulerDisaggregationPrefillMixin",
    """process_prefill_chunk has_bootstrapped_waiting_req resolve_waiting_queue_bootstrap
    has_pending_prefill_result defer_round_robin_action
    process_pending_round_robin_actions process_disagg_prefill_inflight_queue
    process_batch_result_disagg_prefill handle_bootstrap_failure optimistic_release_and_requeue
    _retire_aborted_prefill_result""",
    GLOBALS,
)


class Req:
    def __init__(self, name, length, prefix=0):
        self.full_untruncated_fill_ids = range(length)
        self.origin_input_ids = range(length)
        self.prefix_indices = range(prefix)
        self.output_ids = []
        self.kv = NS(kv_allocated_len=prefix)
        self.sampling_params = NS(max_new_tokens=1, ignore_eos=False)
        self.host_hit_length = 0
        self.retracted_stain = False
        self.last_node = None
        self.extend_range = None
        self.rid = name
        self.prefill_ready_seq = None
        self.inflight_middle_chunks = 0
        self.kv.holds_kv = prefix > 0
        self.kv.holds_mamba = False
        self.beam_group = None
        self.pending_bootstrap = False
        self.aborted = False
        self.cleanup = []
        self.matches = []
        self.disagg_kv_sender = NS(
            abort=lambda: self.cleanup.append("sender"), poll="ready"
        )
        self.time_stats = NS(
            trace_ctx=NS(abort=lambda **kwargs: None),
            set_completion_time=Mock(),
            set_prefill_finished_time=Mock(),
        )
        self.metadata_buffer_index = 0
        self.return_logprob = False
        self.to_finish = None
        self.seqlen = length
        self.kv.cache_protected_len = 0

    def set_extend_range(self, start, end):
        self.extend_range = NS(start=start, end=end, length=end - start)

    def needs_host_load_back(self):
        return False

    def init_next_round_input(self, tree_cache=None):
        self.matches.append(tree_cache)

    def update_finish_state(self):
        self.aborted = True

    def finished(self):
        return self.aborted


class Harness(Scheduler, Prefill):
    def __init__(self, free=4096, rows=32, chunk=128, enabled=True):
        self.enable_chunked_prefill_round_robin = enabled
        self.waiting_queue = []
        self.suspended_prefill_queue = []
        self.chunked_req = None
        self._pending_round_robin_actions = {}
        self._pending_chunked_abort_req = None
        self._prefill_ready_seq = 0
        self.running_batch = Batch()
        self.last_batch = None
        self.result_queue = []
        self.enable_hierarchical_cache = self.enable_unified_cache_external_linker = (
            False
        )
        self.enable_hicache_storage = self.enable_priority_preemption = (
            self.is_hybrid_swa
        ) = False
        self.enable_priority_scheduling = self.enable_lora = self.enable_overlap = False
        self.is_mixed_chunk = False
        self.dllm_config = self.min_free_slots_delayer = self.dynamic_chunk_sizer = None
        self.grammar_manager = NS(
            has_waiting_grammars=lambda: False, abort_requests=lambda req: None
        )
        self.policy = NS(calc_priority=lambda *args, **kwargs: None)
        self.processed_tokens_counter = 0
        self.chunked_prefill_size = chunk
        self.tp_worker = NS(model_runner=NS(attn_backend=NS(), prefill_aware_swa=False))
        self.page_size = 64
        self.tree_cache = NS()
        self.token_to_kv_pool_allocator = NS(free=free)
        self.req_to_token_pool = NS(available_size=lambda: rows)
        self.new_token_ratio_tracker = NS(current=1.0)
        self.max_prefill_tokens = chunk
        self.priority_scheduling_preemption_threshold = 0
        self.max_prefill_bs = self.max_running_requests = 32
        self.truncation_align_size = None
        self.model_config = None
        self.spec_algorithm = NS(is_none=lambda: True)
        self.load_inquirer = NS(_get_num_pending_tokens=lambda **kwargs: 0)
        self.disaggregation_mode = MODE.PREFILL
        self.mm_receiver = None
        self.disagg_prefill_bootstrap_queue = NS(
            queue=[], finalize_bootstrap=self.finalize_bootstrap
        )
        self.disagg_prefill_inflight_queue = []
        self.collect_inflight_reqs = lambda: set()
        self.clear_pending_chunk_send = lambda req: req.cleanup.append("pending_send")
        self.req_to_metadata_buffer_idx_allocator = None
        self._release_aborted_request = lambda rid: None
        self.outputs = []
        self.output_streamer = NS(
            stream_output=lambda reqs, *args: self.outputs.extend(reqs)
        )
        self.ipc_channels = NS(
            send_to_tokenizer=NS(
                send_output=lambda output, req: self.outputs.append(req)
            )
        )
        self.check_bootstrap = lambda req: not req.pending_bootstrap
        self.send_kv_chunk = Mock()
        self.optimistic_release_and_requeue = self.retry
        self.attn_cp_cpu_group = self.attn_tp_cpu_group = None
        self.failed = []
        self.handle_bootstrap_failure = lambda req, **kwargs: self.failed.append(req)

    def finalize_bootstrap(self, req):
        req.pending_bootstrap = False

    def retry(self, req):
        self.remove_prefill_ready_requests((req,))
        req.kv.holds_kv = False
        self.enqueue_prefill_ready([req])

    def get_num_allocatable_reqs(self, *args, **kwargs):
        return self.req_to_token_pool.available_size()

    def suspend(self, req):
        req.prefill_ready_seq = self._prefill_ready_seq
        self._prefill_ready_seq += 1
        self.suspended_prefill_queue.append(req)

    def batch(self):
        return self._get_new_batch_prefill_raw(None, self.running_batch)[0]

    def complete(self, batch):
        for req in batch.reqs:
            req.kv.holds_kv = True
            req.kv.kv_allocated_len = req.extend_range.end
            if req is batch.chunked_req:
                req.inflight_middle_chunks -= 1
        self.last_batch = batch


class RoundRobinTests(unittest.TestCase):
    def test_resume_rejected_without_appending(self):
        original = Adder.add_chunked_req

        def exhausted(adder, req):
            adder.is_hybrid_swa = True
            adder._swa_req_ring = False
            adder.rem_swa_tokens = 0
            return original(adder, req)

        for location in ("mixed", "suspended", "current"):
            with self.subTest(location=location):
                s = Harness()
                req = Req("resume", 384, prefix=64)
                req.set_extend_range(0, 64)
                fresh = Req("fresh", 64)
                if location == "mixed":
                    s.enqueue_prefill_ready([fresh])
                if location == "current":
                    s.chunked_req = req
                else:
                    s.suspend(req)
                seq = req.prefill_ready_seq
                count = req.inflight_middle_chunks
                with patch.object(Adder, "add_chunked_req", exhausted):
                    batch = s.batch()
                if location == "mixed":
                    self.assertEqual(batch.reqs, [fresh])
                else:
                    self.assertIsNone(batch)
                self.assertEqual(req.inflight_middle_chunks, count)
                self.assertEqual(req.prefill_ready_seq, seq)
                self.assertIs(s.chunked_req, req if location == "current" else None)
                self.assertEqual(
                    s.suspended_prefill_queue, [] if location == "current" else [req]
                )
                batch = s.batch()
                self.assertEqual(batch.reqs.count(req), 1)
                self.assertNotIn(req, s.suspended_prefill_queue)
                self.assertEqual(req.inflight_middle_chunks, count + 1)

    def test_suspended_running_timeout(self):
        s = Harness()
        s.ps = NS(pp_size=1)
        expired, recent, unstarted, done = [
            Req(n, 128) for n in ("old", "new", "zero", "done")
        ]
        for req, entry in zip((expired, recent, unstarted, done), (1, 99, 0, 1)):
            req.time_stats.forward_entry_time = entry
        done.finished = lambda: True
        s.suspended_prefill_queue = [expired, recent, unstarted, done]
        waiting = Req("waiting", 128)
        waiting.time_stats.wait_queue_entry_time = 1
        s.waiting_queue = [waiting]
        running_timeout = NS(get=lambda: 10)
        with patch.dict(
            GLOBALS,
            envs=NS(
                SGLANG_REQ_WAITING_TIMEOUT=NS(get=lambda: 0),
                SGLANG_REQ_RUNNING_TIMEOUT=running_timeout,
            ),
            time=NS(perf_counter=lambda: 100),
            AbortReq=NS,
        ):
            self.assertEqual([r.rid for r in s._poll_timeout_aborts()], [expired.rid])
            s.last_batch = Batch([expired])
            aborts = s._poll_timeout_aborts()
            self.assertEqual(len(aborts), 1)
            self.assertEqual(aborts[0].finished_reason["status_code"], 503)
            self.assertEqual(
                aborts[0].abort_message, "Request running timeout reached."
            )
            s.enable_chunked_prefill_round_robin = False
            s.last_batch = None
            self.assertEqual(s._poll_timeout_aborts(), [])
            s.ps.pp_size = 2
            s.running_mbs, s.mbs = [Batch([expired])], [None]
            self.assertEqual([r.rid for r in s._poll_timeout_aborts()], [expired.rid])
            running_timeout.get = lambda: 0
            self.assertEqual(s._poll_timeout_aborts(), [])
            GLOBALS["envs"].SGLANG_REQ_WAITING_TIMEOUT.get = lambda: 10
            self.assertEqual([r.rid for r in s._poll_timeout_aborts()], [waiting.rid])

    def test_round_robin_configuration_limits(self):
        valid = dict(
            enable_chunked_prefill_round_robin=True,
            disaggregation_mode=MODE.PREFILL,
            ps=NS(pp_size=1),
            enable_pdmux=False,
            is_hybrid_swa=False,
            chunked_prefill_size=128,
        )
        Scheduler._validate_prefill_round_robin(NS(**valid))
        for field, value in (
            ("disaggregation_mode", MODE.NULL),
            ("disaggregation_mode", MODE.DECODE),
            ("ps", NS(pp_size=2)),
            ("enable_pdmux", True),
            ("is_hybrid_swa", True),
            ("chunked_prefill_size", None),
        ):
            with self.subTest(field=field, value=value):
                config = dict(valid, **{field: value})
                with self.assertRaises(ValueError):
                    Scheduler._validate_prefill_round_robin(NS(**config))
                config["enable_chunked_prefill_round_robin"] = False
                Scheduler._validate_prefill_round_robin(NS(**config))

    def test_bootstrap_failures_are_removed_in_one_batch(self):
        for overlap in (False, True):
            with self.subTest(overlap=overlap):
                s = Harness()
                s.enable_overlap = overlap
                a, b, live = [Req(name, 128) for name in ("a", "b", "live")]
                s.waiting_queue = [a]
                s.suspended_prefill_queue = [b, live]
                for req in (a, b, live):
                    req.disagg_kv_sender.poll = (
                        "transferring" if req is live else "failed"
                    )
                if overlap:
                    s.handle_bootstrap_failure = lambda req: (
                        Prefill.handle_bootstrap_failure(s, req)
                    )
                remove = s.remove_prefill_ready_requests = Mock(
                    wraps=s.remove_prefill_ready_requests
                )
                s.resolve_waiting_queue_bootstrap()
                remove.assert_called_once_with({a, b})
                self.assertEqual(s.waiting_queue, [])
                self.assertEqual(s.suspended_prefill_queue, [live])
                if overlap:
                    self.assertEqual(
                        s._pending_round_robin_actions,
                        {a: "bootstrap_failure", b: "bootstrap_failure"},
                    )
                else:
                    self.assertEqual(s.failed, [a, b])

    def test_reservation_uses_planned_end_without_double_counting(self):
        other, candidate = Req("other", 257, prefix=64), Req("new", 128)
        adder = Adder(512, 128)
        adder.round_robin_requests = (other, other)
        self.assertFalse(adder._round_robin_can_admit(candidate))
        other.set_extend_range(64, 257)
        adder.can_run_list.append(other)
        adder.rem_total_token_offset = 321
        self.assertTrue(adder._round_robin_can_admit(candidate))
        self.assertEqual(adder.rem_total_token_offset, 321)

    def test_fresh_before_yield_and_uncontested_number(self):
        s = Harness()
        a, b = Req("a", 384), Req("b", 64)
        s.enqueue_prefill_ready([a])
        batch = s.batch()
        s.complete(batch)
        s.process_prefill_chunk(batch, s.running_batch)
        self.assertIs(s.chunked_req, a)
        self.assertEqual(a.prefill_ready_seq, 0)
        s.enqueue_prefill_ready([b])
        s.process_prefill_chunk(None, s.running_batch)
        self.assertLess(b.prefill_ready_seq, a.prefill_ready_seq)
        batch = s.batch()
        self.assertEqual(batch.reqs, [b, a])
        self.assertIs(batch.chunked_req, a)
        self.assertEqual(a.matches, [s.tree_cache, None])
        self.assertEqual(a.inflight_middle_chunks, 1)

    def test_long_long_short_and_no_second_middle(self):
        s = Harness()
        a, b, c = Req("a", 384), Req("b", 384), Req("c", 32)
        s.enqueue_prefill_ready([a, b, c])
        order = []
        for _ in range(3):
            batch = s.batch()
            order += [r.rid for r in batch.reqs]
            self.assertLessEqual(
                sum(r.extend_range.end < r.seqlen for r in batch.reqs), 1
            )
            s.complete(batch)
            s.process_prefill_chunk(batch, s.running_batch)
        self.assertEqual(order, ["a", "b", "c", "a"])
        self.assertEqual(s.waiting_queue, [])

    def test_pressure_or_zero_rows_preserves_fresh_fifo(self):
        for free, rows in ((256, 32), (4096, 0)):
            s = Harness(free=free, rows=rows)
            b, a = Req("b", 200), Req("a", 256, prefix=128)
            s.enqueue_prefill_ready([b])
            s.suspend(a)
            s.running_batch.batch_is_full = True
            batch = s.batch()
            self.assertEqual(batch.reqs, [a])
            self.assertEqual(s.waiting_queue, [b])
            self.assertEqual(b.prefill_ready_seq, 0)
            self.assertIsNone(s.chunked_req)

    def test_final_then_new_middle_commits_on_budget_exhaustion(self):
        s = Harness()
        a, b = Req("a", 128, prefix=64), Req("b", 256)
        s.suspend(a)
        s.enqueue_prefill_ready([b])
        batch = s.batch()
        self.assertEqual(batch.reqs, [a, b])
        self.assertIs(s.chunked_req, b)
        self.assertFalse(s.suspended_prefill_queue)
        self.assertEqual(b.inflight_middle_chunks, 1)

    def test_counter_resets_only_at_empty_boundary(self):
        s = Harness()
        a = Req("a", 256, prefix=64)
        s.suspend(a)
        s.reset_prefill_ready_seq_if_idle()
        self.assertEqual(s._prefill_ready_seq, 1)
        s.remove_prefill_ready_requests((a,))
        s.reset_prefill_ready_seq_if_idle()
        s.enqueue_prefill_ready([a])
        self.assertEqual(a.prefill_ready_seq, 0)

    def test_abort_suspended_keeps_current_and_releases_once(self):
        s = Harness()
        a, b = Req("a", 256, prefix=64), Req("b", 256, prefix=64)
        s.suspend(a)
        s.chunked_req = b
        abort = NS(rid="a", abort_all=False, abort_message=None)
        s.abort_request(abort)
        s.abort_request(abort)
        self.assertFalse(s._pending_round_robin_actions)
        s.process_pending_chunked_abort()
        s.process_pending_chunked_abort()
        self.assertIs(s.chunked_req, b)
        self.assertEqual(a.cleanup, ["pending_send", "sender", "metadata", "kv"])
        self.assertEqual(s.outputs, [a])
        self.assertFalse(s.suspended_prefill_queue)

    def test_disabled_preserves_current_first(self):
        s = Harness(enabled=False)
        a, b = Req("a", 384, prefix=128), Req("b", 32)
        s.chunked_req = a
        s.enqueue_prefill_ready([b])
        self.assertIsNone(b.prefill_ready_seq)
        self.assertEqual(s.batch().reqs, [a])
        s.process_prefill_chunk(None, s.running_batch)
        self.assertIs(s.chunked_req, a)
        self.assertEqual(s.suspended_prefill_queue, [])


class OverlapTests(unittest.TestCase):
    def setUp(self):
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
            next_token_ids=NS(tolist=lambda: [1] * len(batch.reqs)),
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
        req.disagg_kv_sender.abort = Mock()
        return req

    def test_result_terminal_action_is_removed_before_next_admission(self):
        s = self.s
        a = self.request("a", middle=False)
        live = Req("live", 64)
        s.suspend(a)
        s.enqueue_prefill_ready([live])
        a.to_finish = "abort"
        batch = Batch([a], None)
        s.process_batch_result_disagg_prefill(batch, self.result(batch))
        self.assertEqual(s._pending_round_robin_actions[a], "retire")
        self.assertIn(a, s.suspended_prefill_queue)
        self.assertEqual(a.cleanup, [])
        s.process_pending_round_robin_actions()
        self.assertNotIn(a, s.suspended_prefill_queue)
        self.assertEqual(a.cleanup.count("kv"), 1)
        self.assertEqual(s.batch().reqs, [live])

    def test_paused_abort_without_result_releases_immediately(self):
        s = self.s
        req = self.request("parked")
        s.suspend(req)
        s.last_batch = Batch([req])
        s.collect_inflight_reqs = lambda: [req]
        s.pause_generation(NS(mode="in_place"))
        self.assertTrue(s._engine_paused)
        s.abort_request(NS(rid=req.rid, abort_all=False, abort_message="cancelled"))
        self.assertEqual(s._pending_round_robin_actions, {})
        self.assertEqual(s.outputs, [req])
        self.assertEqual(req.cleanup.count("kv"), 1)
        self.assertEqual(req.cleanup.count("metadata"), 1)
        self.assertEqual(req.to_finish, ("cancelled", 503))
        req.disagg_kv_sender.abort.assert_called_once()
        s.abort_request(NS(rid=req.rid, abort_all=False, abort_message="again"))
        self.assertEqual(s.outputs, [req])
        self.assertEqual(req.cleanup.count("kv"), 1)
        self.assertEqual(req.to_finish, ("cancelled", 503))

    def test_abort_waits_for_result_then_releases_once(self):
        s = self.s
        a = self.request("a")
        b = self.request("b")
        old, new = Batch([a], a), Batch([b], b)
        result = self.result(old)
        result.copy_done.synchronize.side_effect = lambda: self.assertEqual(
            a.cleanup, []
        )
        s.result_queue = [(old, result)]
        s.chunked_req = a
        s.abort_request(NS(rid="a", abort_all=False, abort_message=None))
        s.process_pending_chunked_abort()
        self.assertIsNone(s.chunked_req)
        self.assertEqual(a.cleanup, [])
        # Next batch launches before A's result is processed.
        s.result_queue.append((new, self.result(new)))
        s.result_queue.pop(0)
        s.process_batch_result_disagg_prefill(old, result)
        result.copy_done.synchronize.assert_called_once()
        self.assertEqual(a.inflight_middle_chunks, 0)
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
        s.abort_request(NS(rid=a.rid, abort_all=False, abort_message=None))
        self.assertNotIn(a, s._pending_round_robin_actions)
        s.process_pending_chunked_abort()
        self.assertEqual(a.cleanup, [])
        s.result_queue.pop(0)
        s.process_batch_result_disagg_prefill(batch, result)
        s.process_pending_round_robin_actions()
        self.assertEqual(s.outputs, [a])
        s.send_kv_chunk.assert_not_called()

    def test_abort_all_waits_for_last_result_and_preserves_terminal_reason(self):
        s = self.s
        a = self.request("two-results")
        a.inflight_middle_chunks = 2
        batches = [Batch([a], a), Batch([a], a)]
        results = [self.result(batch) for batch in batches]
        s.result_queue = list(zip(batches, results))
        s.suspend(a)
        pending = self.request("already-failed")
        reason = pending.to_finish = "original-failure"
        s.defer_round_robin_action(pending, "bootstrap_failure")
        s.collect_inflight_reqs = lambda: [a, pending]
        abort = NS(rid="", abort_all=True, abort_message="timeout")
        s.abort_request(abort)
        reason_after_first_abort = a.to_finish
        s.abort_request(NS(rid="", abort_all=True, abort_message="second timeout"))
        self.assertIs(a.to_finish, reason_after_first_abort)
        self.assertEqual(a.to_finish, ("timeout", 503))
        self.assertIs(pending.to_finish, reason)
        self.assertEqual(s._pending_round_robin_actions[pending], "bootstrap_failure")
        for i, (batch, result) in enumerate(zip(batches, results)):
            s.result_queue.pop(0)
            s.process_batch_result_disagg_prefill(batch, result)
            result.copy_done.synchronize.assert_called_once()
            self.assertEqual(a.inflight_middle_chunks, 1 - i)
            if i == 0:
                self.assertNotIn(a, s._pending_round_robin_actions)
            else:
                self.assertEqual(s._pending_round_robin_actions[a], "retire")
            self.assertEqual(a.cleanup, [])
        # The existing bootstrap handler is independent of this client's abort.
        s.process_pending_round_robin_actions()
        self.assertEqual(s.outputs, [a])
        self.assertEqual(a.cleanup.count("kv"), 1)
        s.send_kv_chunk.assert_not_called()

    def test_abort_inflight_uses_existing_sender_path(self):
        s = self.s
        a = self.request("inflight", middle=False)
        s.disagg_prefill_inflight_queue = [a]
        s.result_queue = [(Batch([a]), None)]
        s.abort_request(NS(rid=a.rid, abort_all=False, abort_message=None))
        a.disagg_kv_sender.abort.assert_called_once()
        self.assertEqual(s._pending_round_robin_actions, {})
        self.assertEqual(a.cleanup, [])
        self.assertEqual(s.disagg_prefill_inflight_queue, [a])

    def test_bootstrap_yield_uses_native_retry_after_admission(self):
        for overlap in (False, True):
            with self.subTest(overlap=overlap):
                self.setUp()
                s = self.s
                s.enable_overlap = overlap
                s.metrics_reporter.enable_metrics = False
                a, b = self.request("a"), Req("b", 64)
                a.pending_bootstrap = True
                a.prefill_attempt_count = 0
                a.inflight_middle_chunks = 0
                a.prefix_indices = range(0)
                a.kv.kv_allocated_len = 0
                a.kv.holds_kv = False
                a.reset_for_retract = Mock(
                    side_effect=lambda: setattr(a.kv, "holds_kv", False)
                )
                a.time_stats.reset_prefill_retry_time = Mock()
                a.time_stats.set_wait_queue_entry_time = Mock()
                retry = s.optimistic_release_and_requeue = Mock(
                    wraps=lambda req: Prefill.optimistic_release_and_requeue(s, req)
                )
                s.enqueue_prefill_ready([a])
                batch = s.batch()
                self.assertIs(s.chunked_req, a)
                a.kv.holds_kv = True
                a.kv.kv_allocated_len = a.extend_range.end
                # The result owns an independent snapshot, as in the real loop.
                snapshot = Batch(batch.reqs, batch.chunked_req)
                result = self.result(snapshot)
                if overlap:
                    s.result_queue = [(snapshot, result)]
                    result.copy_done.synchronize.side_effect = lambda: (
                        retry.assert_not_called()
                    )
                else:
                    s.process_batch_result_disagg_prefill(snapshot, result)
                s.enqueue_prefill_ready([b])
                s.process_prefill_chunk(batch, s.running_batch)
                self.assertIsNone(s.chunked_req)
                if overlap:
                    retry.assert_not_called()
                    self.assertEqual(a.cleanup, [])
                    next_batch = s.batch()
                    self.assertEqual(next_batch.reqs, [b])
                    s.result_queue.append((next_batch, self.result(next_batch)))
                    old, old_result = s.result_queue.pop(0)
                    s.process_batch_result_disagg_prefill(old, old_result)
                retry.assert_called_once_with(a)
                a.reset_for_retract.assert_called_once()
                self.assertEqual(a.cleanup.count("kv"), 1)
                self.assertFalse(a.kv.holds_kv)
                self.assertEqual(a.start_send_idx, 0)
                self.assertEqual(a.tmp_end_idx, -1)
                self.assertEqual(a.prefill_attempt_count, 1)
                self.assertIs(s.waiting_queue[-1], a)
                self.assertEqual(s.suspended_prefill_queue, [])
                s.send_kv_chunk.assert_not_called()

    def test_pending_bootstrap_without_ready_competitor_keeps_slot(self):
        s = self.s
        a, b = self.request("a"), self.request("b")
        a.pending_bootstrap = b.pending_bootstrap = True
        s.chunked_req = a
        s.enqueue_prefill_ready([b])
        s.process_prefill_chunk(None, s.running_batch)
        self.assertIs(s.chunked_req, a)
        self.assertEqual(s.suspended_prefill_queue, [])
        self.assertEqual(a.cleanup, [])
        s.send_kv_chunk.assert_not_called()

    def test_bootstrap_failure_waits_for_result(self):
        s = self.s
        a = self.request("a")
        a.pending_bootstrap = True
        s.chunked_req = a
        batch = Batch([a], a)
        result = self.result(batch)
        s.result_queue = [(batch, result)]

        def fail_bootstrap(req):
            Prefill.handle_bootstrap_failure(s, req)
            return False

        s.check_bootstrap = fail_bootstrap
        retry = s.optimistic_release_and_requeue = Mock()
        s.process_prefill_chunk(None, s.running_batch)
        self.assertIsNone(s.chunked_req)
        self.assertEqual(s._pending_round_robin_actions[a], "bootstrap_failure")
        finish = s.handle_bootstrap_failure = Mock()
        s.process_pending_round_robin_actions()
        finish.assert_not_called()
        self.assertEqual(a.cleanup, [])
        s.result_queue.pop(0)
        s.process_batch_result_disagg_prefill(batch, result)
        s.process_pending_round_robin_actions()
        s.process_pending_round_robin_actions()
        finish.assert_called_once_with(a, defer=False)
        retry.assert_not_called()

    def test_inflight_failure_uses_native_handler(self):
        s = self.s
        a = self.request("failed", middle=False)
        a.pending_bootstrap = False
        a.bootstrap_host = "fake"
        a.finished_reason = None
        s.disagg_prefill_inflight_queue = [a]
        s.handle_inflight_transfer_failure = Mock()
        # The native TP-reduced failure path also handles a locally ready sender.
        with patch.dict(
            GLOBALS,
            poll_and_all_reduce_attn_cp_tp_group=lambda *args: ["failed"],
            FINISH_ABORT=type("Abort", (), {}),
            FAKE_BOOTSTRAP_HOST="fake",
        ):
            self.assertEqual(s.process_disagg_prefill_inflight_queue(), [a])
        s.handle_inflight_transfer_failure.assert_called_once_with(a)
        a.disagg_kv_sender.abort.assert_not_called()
        self.assertEqual(s.outputs, [a])
        self.assertEqual(s.disagg_prefill_inflight_queue, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
