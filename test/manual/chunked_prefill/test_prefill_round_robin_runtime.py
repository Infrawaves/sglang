"""CPU regression checks of production scheduler methods, with GPU/transport stubs.

Run: python3 test/manual/chunked_prefill/test_prefill_round_robin_runtime.py
This executes complete method bodies; it does not validate GPU kernels or TP transport.
"""

import ast
import logging
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

from prefill_round_robin_budget_check import Adder, Request, Result

ROOT = Path(__file__).resolve().parents[3] / "python/sglang/srt"
SCHEDULE = NS(
    prefill_max_requests=None, enable_mixed_chunk=False, enable_dynamic_chunking=False
)
PARALLEL = NS(pp_max_micro_batch_size=32)
MODE = NS(PREFILL="prefill", DECODE="decode", NULL="null")


def extract(path, class_name, names, namespace):
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
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[future, cls], type_ignores=[])),
            str(ROOT / path),
            "exec",
        ),
        namespace,
    )
    return namespace[class_name]


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
    get_schedule=lambda: SCHEDULE,
    get_parallel=lambda: PARALLEL,
    get_memory=lambda: NS(enable_flexkv=False),
    get_disagg=lambda: NS(disaggregation_mode="prefill"),
    get_exec=lambda: NS(dllm=NS(dllm_algorithm=None)),
    TEST_RETRACT=False,
    PrefillAdder=make_adder,
    AddReqResult=Result,
    ScheduleBatch=Batch,
    PrefillStats=NS(from_adder=lambda *args, **kwargs: None),
    set_time_batch=lambda *args: None,
    DisaggregationMode=MODE,
    maybe_cache_unfinished_req=cache_chunk,
    is_aborted=lambda r: r.aborted,
    prepare_abort=lambda r, reason: setattr(r, "aborted", True),
    maybe_release_metadata_buffer=lambda req, allocator: req.cleanup.append("metadata"),
    release_kv_cache=lambda req, cache, **kwargs: req.cleanup.append("kv"),
    _make_abort_req=lambda req, **kwargs: req.rid,
    logger=logging.getLogger(__name__),
    FINISH_ABORT=lambda: "abort",
    KVPoll=NS(Failed="failed", WaitingForInput="ready"),
    should_force_retry=lambda r: False,
    poll_and_all_reduce_attn_cp_tp_group=lambda senders, *args: [
        s.poll for s in senders
    ],
)
Scheduler = extract(
    "managers/scheduler.py",
    "Scheduler",
    [
        "enqueue_prefill_ready",
        "reset_prefill_ready_seq_if_idle",
        "iter_round_robin_requests",
        "detach_round_robin_request",
        "_get_new_batch_prefill_raw",
        "process_pending_chunked_abort",
        "_release_chunked_abort",
        "abort_request",
        "_validate_prefill_round_robin",
    ],
    GLOBALS,
)
# Discover the existing mixin's name without importing the GPU package.
prefill_tree = ast.parse((ROOT / "disaggregation/prefill.py").read_text())
prefill_class = next(
    n.name
    for n in prefill_tree.body
    if isinstance(n, ast.ClassDef)
    and any(
        isinstance(m, ast.FunctionDef) and m.name == "process_prefill_chunk"
        for m in n.body
    )
)
Prefill = extract(
    "disaggregation/prefill.py",
    prefill_class,
    [
        "process_prefill_chunk",
        "resolve_waiting_queue_bootstrap",
        "has_bootstrapped_waiting_req",
    ],
    GLOBALS,
)


class Req(Request):
    def __init__(self, name, length, prefix=0):
        super().__init__(length, prefix=prefix)
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
        self.time_stats = NS(trace_ctx=NS(abort=lambda **kwargs: None))
        self.to_finish = None
        self.seqlen = length
        self.kv.cache_protected_len = 0

    def init_next_round_input(self, tree_cache=None):
        self.matches.append(tree_cache)

    def finished(self):
        return self.aborted


class Harness(Scheduler, Prefill):
    def __init__(self, free=4096, rows=32, chunk=128, enabled=True):
        self.enable_chunked_prefill_round_robin = enabled
        self.waiting_queue = []
        self.suspended_prefill_queue = []
        self.chunked_req = None
        self._pending_round_robin_aborts = {}
        self._pending_chunked_abort_req = None
        self._prefill_ready_seq = 0
        self.running_batch = Batch()
        self.last_batch = None
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
        self.handle_bootstrap_failure = lambda req: self.failed.append(req)

    def finalize_bootstrap(self, req):
        req.pending_bootstrap = False

    def retry(self, req):
        self.detach_round_robin_request(req)
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

    def test_resume_does_not_consume_fresh_row(self):
        s = Harness(rows=1)
        a, b = Req("a", 96, prefix=64), Req("b", 64)
        s.suspend(a)
        s.enqueue_prefill_ready([b])
        self.assertEqual(s.batch().reqs, [a, b])

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

    def test_empty_budget_keeps_current(self):
        s = Harness()
        a = Req("a", 256, prefix=64)
        s.chunked_req = a
        s.max_prefill_tokens = 0
        self.assertIsNone(s.batch())
        self.assertIs(s.chunked_req, a)

    def test_zero_free_tail_page_continues(self):
        s = Harness(free=0, rows=0)
        a = Req("a", 64, prefix=1)
        s.suspend(a)
        self.assertEqual(s.batch().reqs, [a])
        self.assertIsNone(s.chunked_req)

    def test_counter_resets_only_at_empty_boundary(self):
        s = Harness()
        a = Req("a", 256, prefix=64)
        s.suspend(a)
        s.reset_prefill_ready_seq_if_idle()
        self.assertEqual(s._prefill_ready_seq, 1)
        s.detach_round_robin_request(a)
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
        self.assertEqual(list(s._pending_round_robin_aborts), [a])
        s.process_pending_chunked_abort()
        s.process_pending_chunked_abort()
        self.assertIs(s.chunked_req, b)
        self.assertEqual(a.cleanup, ["pending_send", "sender", "metadata", "kv"])
        self.assertEqual(s.outputs, [a])
        self.assertFalse(s.suspended_prefill_queue)

    def test_abort_all_is_ordered_and_deduplicated(self):
        s = Harness()
        a, b = Req("a", 256, prefix=64), Req("b", 256, prefix=64)
        s.chunked_req = a
        s.suspend(b)
        s.abort_request(NS(rid="", abort_all=True, abort_message=None))
        self.assertEqual(list(s.iter_round_robin_requests()), [a, b])
        s.process_pending_chunked_abort()
        self.assertEqual(s.outputs, [a, b])
        self.assertIsNone(s.chunked_req)

    def test_bootstrap_release_precedes_rr_yield(self):
        s = Harness()
        a, b = Req("a", 256, prefix=64), Req("b", 32)
        a.pending_bootstrap = True
        s.chunked_req = a
        s.enqueue_prefill_ready([b])
        s.process_prefill_chunk(None, s.running_batch)
        self.assertEqual(s.waiting_queue, [b, a])
        self.assertFalse(s.suspended_prefill_queue)
        self.assertFalse(a.kv.holds_kv)
        self.assertIsNone(s.chunked_req)

    def test_bootstrap_polls_suspended_and_detaches_failure(self):
        s = Harness()
        a, b = Req("a", 256, prefix=64), Req("b", 256, prefix=64)
        a.pending_bootstrap = b.pending_bootstrap = True
        a.disagg_kv_sender.poll = "failed"
        s.suspend(a)
        s.suspend(b)
        s.resolve_waiting_queue_bootstrap()
        self.assertEqual(s.failed, [a])
        self.assertEqual(s.suspended_prefill_queue, [b])
        self.assertFalse(b.pending_bootstrap)
        self.assertTrue(s.has_bootstrapped_waiting_req())

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


class StartupAndAccountingTests(unittest.TestCase):
    def test_startup_checks_resolved_config_and_allocator(self):
        s = Harness()
        s.enable_overlap_mlx = s.enable_pdmux = s.enable_unified_memory = False
        s.schedule_policy = "fcfs"
        s.ps = NS(
            pp_size=1,
            tp_size=8,
            attn_tp_size=8,
            attn_cp_size=1,
            attn_dcp_size=1,
            attn_dp_size=1,
        )
        allocator_class = type("PagedTokenToKVPoolAllocator", (), {})
        s.token_to_kv_pool_allocator = allocator_class()
        module = NS(PagedTokenToKVPoolAllocator=allocator_class)
        with patch.dict(
            "sys.modules", {"sglang.srt.mem_cache.allocator.paged": module}
        ):
            s._validate_prefill_round_robin()
            for tp_size in (1, 2, 4, 8, 16):
                for page_size in (1, 16, 32, 64, 128):
                    with self.subTest(tp_size=tp_size, page_size=page_size):
                        s.ps.tp_size = s.ps.attn_tp_size = tp_size
                        s.page_size = page_size
                        s._validate_prefill_round_robin()
            s.ps.tp_size = s.ps.attn_tp_size = 8
            s.page_size = 64
            for target, name, value in (
                (s, "enable_overlap", True),
                (s, "enable_unified_memory", True),
                (s, "enable_lora", True),
                (s, "schedule_policy", "lpm"),
                (s.ps, "attn_tp_size", 4),
                (s.ps, "pp_size", 2),
                (s, "token_to_kv_pool_allocator", object()),
                (SCHEDULE, "enable_dynamic_chunking", True),
            ):
                previous = getattr(target, name)
                try:
                    setattr(target, name, value)
                    with self.assertRaises(ValueError, msg=name):
                        s._validate_prefill_round_robin()
                finally:
                    setattr(target, name, previous)

    def test_ordinary_abort_markers_still_clear_after_completion(self):
        s = Harness(enabled=False)
        a = Req("a", 256, prefix=64)
        a.aborted = True
        s._pending_chunked_abort_req = a
        s.process_pending_chunked_abort()
        self.assertIsNone(s._pending_chunked_abort_req)
        self.assertEqual(s.outputs, [])

    def test_load_and_invariant_count_suspended_once(self):
        s = Harness()
        a, b = Req("a", 256, prefix=64), Req("b", 256, prefix=128)
        s.suspend(a)
        s.chunked_req = b
        s._pending_round_robin_aborts[a] = None
        namespace = dict(
            DisaggregationMode=MODE, ceil_align=lambda n, p: (n + p - 1) // p * p
        )
        load_class = extract(
            "managers/scheduler_components/load_inquirer.py",
            "SchedulerLoadInquirer",
            ["_get_num_pending_tokens", "get_num_waiting_uncached_tokens"],
            namespace,
        )
        load = load_class()
        load.get_round_robin_requests = s.iter_round_robin_requests
        load.get_chunked_req = lambda: b
        load.get_waiting_queue = lambda: []
        load.waiting_queue_prefix_matched = lambda: False
        load.get_recent_cache_hit_rate = lambda: 0
        load.disaggregation_mode = MODE.PREFILL
        self.assertEqual(load._get_num_pending_tokens(), 320)
        self.assertEqual(load._get_num_pending_tokens(chunk_deduct=64), 256)
        self.assertEqual(load.get_num_waiting_uncached_tokens(), 320)
        invariant_class = extract(
            "managers/scheduler_components/invariant_checker.py",
            "SchedulerInvariantChecker",
            ["_get_total_uncached_sizes"],
            namespace,
        )
        checker = invariant_class()
        checker.get_last_batch = lambda: Batch([a, b])
        checker.get_running_batch = lambda: Batch([b])
        checker.get_chunked_req = lambda: b
        checker.get_round_robin_requests = s.iter_round_robin_requests
        checker.page_size = 64
        checker.is_hybrid_swa = False
        self.assertEqual(checker._get_total_uncached_sizes(), (192, 0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
