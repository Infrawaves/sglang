"""Unit tests for srt/disaggregation/decode_migration.py and the decode offload
manager's migration flush."""

import unittest
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from weakref import WeakKeyDictionary as WeakKeyDict

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.decode_kvcache_offload_manager import (
    DecodeKVCacheOffloadManager,
)
from sglang.srt.disaggregation.decode_migration import (
    DecodeMigrationExporter,
    decode_receive_side_idle,
    migration_blocker,
)
from sglang.srt.disaggregation.kv_events import OffloadedState
from sglang.srt.disaggregation.mode import DisaggregationMode
from sglang.srt.managers.req_types import FINISH_ABORT, FINISH_MIGRATED
from sglang.srt.managers.schedule_batch import Req, ReqKvInfo
from sglang.srt.managers.scheduler_components.internal_state import (
    SchedulerInternalStateController,
)
from sglang.srt.managers.scheduler_components.request_list import (
    IndexedReqList,
    ListKind,
)
from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
from sglang.srt.sampling.sampling_params import SamplingParams

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _RecordingAllocator(BaseTokenToKVPoolAllocator):
    def __init__(self, page_size: int):
        super().__init__(
            size=1024,
            page_size=page_size,
            dtype=torch.bfloat16,
            device="cpu",
            kvcache=None,
            need_sort=False,
        )
        self.freed = []

    def clear(self):
        self.freed = []

    def alloc(self, need_size: int):
        raise NotImplementedError

    def free(self, free_index: torch.Tensor):
        self.freed.append(free_index.clone())


def _make_manager(*, page_size: int, stride: int, pool_size: int = 64):
    """A real offload manager over recording pools; the controller is mocked."""
    req_to_token_pool = MagicMock()
    req_to_token_pool.req_to_token = torch.arange(
        pool_size, dtype=torch.int64
    ).unsqueeze(0)
    allocator = _RecordingAllocator(page_size)
    tree_cache = MagicMock()
    tree_cache.protected_size_ = 0
    tree_cache.req_to_token_pool = req_to_token_pool
    tree_cache.token_to_kv_pool_allocator = allocator
    tree_cache.logical_page_size = allocator.page_size
    tree_cache.free_kv_row = lambda owner, ranges: BasePrefixCache.free_kv_row(
        tree_cache, owner, ranges
    )

    manager = object.__new__(DecodeKVCacheOffloadManager)
    manager.req_to_token_pool = req_to_token_pool
    manager.token_to_kv_pool_allocator = allocator
    manager.kv_cache = MagicMock()
    manager._unified_window = False
    manager.page_size = page_size
    manager.offload_stride = stride
    manager.request_counter = 0
    manager.tree_cache = tree_cache
    manager.tp_group = None
    manager.tp_world_size = 1
    manager.offloaded_state = WeakKeyDict()
    manager.offload_inflight = WeakKeyDict()
    manager.storage_confirmed_len = WeakKeyDict()
    manager._storage_chain_broken = WeakKeyDict()
    manager.ongoing_offload = {}
    manager.ongoing_backup = {}
    manager.decode_host_mem_pool = MagicMock()
    manager.cache_controller = MagicMock()
    manager.cache_controller.write.side_effect = lambda device_indices, **_: (
        device_indices.clone()
    )
    return manager, allocator.freed


def _make_req(*, rid="r", prompt_len=8, output_len=1, **req_kwargs) -> Req:
    req_kwargs.setdefault("allow_migration", True)
    req = Req(
        rid=rid,
        origin_input_text="",
        origin_input_ids=list(range(prompt_len)),
        sampling_params=req_kwargs.pop("sampling_params", SamplingParams()),
        **req_kwargs,
    )
    req.output_ids = list(range(100, 100 + output_len))
    seq_len = prompt_len + output_len
    req.kv = ReqKvInfo(
        req_pool_idx=0, kv_committed_len=seq_len, kv_allocated_len=seq_len
    )
    # The prompt pages' hash tail; the hash itself is not under test.
    req.lock = None
    return req


class TestMigrationFlush(CustomTestCase):
    def test_flush_queues_complete_pages_left_by_the_stride(self):
        """With page 4 and stride 16, the per-step path leaves up to 15 decode
        tokens on the device; the migration flush must queue every complete page
        of them and nothing of the partial tail."""
        manager, _ = _make_manager(page_size=4, stride=16)
        # Prompt 8 (two pages, stored by prefill) + 27 decode tokens committed.
        req = _make_req(prompt_len=8, output_len=28)
        manager.offloaded_state[req] = OffloadedState(last_hash="prompt-tail")

        self.assertEqual(manager.offload_kv_cache(req), True)
        self.assertEqual(manager.offloaded_state[req].inc_len, 16)
        self.assertEqual(manager.flush_for_migration(req), 8)
        self.assertEqual(manager.offloaded_state[req].inc_len, 24)
        # 3 tokens of a partial page remain; nothing more to queue.
        self.assertEqual(manager.flush_for_migration(req), 0)

    def test_flush_reports_full_host_pool_without_losing_progress(self):
        """A full host pool must be reported distinctly from 'nothing to write',
        and must not advance the offloaded length, so a retry writes the pages."""
        manager, _ = _make_manager(page_size=4, stride=4)
        req = _make_req(prompt_len=8, output_len=9)
        manager.offloaded_state[req] = OffloadedState(last_hash="prompt-tail")
        manager.cache_controller.write.side_effect = lambda **_: None

        self.assertEqual(manager.flush_for_migration(req), -1)
        self.assertEqual(manager.offloaded_state[req].inc_len, 0)

    def test_partial_storage_write_stops_the_confirmed_span(self):
        """Pages after a failed one cannot be hit (the prefix match stops at the
        gap), so later successful writes must not extend the confirmed span."""
        manager, _ = _make_manager(page_size=4, stride=4)
        req = _make_req()
        manager._record_storage_outcome(req, written=8, requested=8)
        manager._record_storage_outcome(req, written=4, requested=8)
        manager._record_storage_outcome(req, written=8, requested=8)
        self.assertEqual(manager.storage_confirmed_tokens(req), 12)


class _FakeOffload:
    """The offload manager surface the exporter uses, with scripted progress."""

    def __init__(self, *, flush_results=None):
        self.tp_group = None
        self.tp_world_size = 1
        self.flush_results = dict(flush_results or {})
        self.device_copy = set()
        self.storage_write = set()
        self.released = []
        self.confirmed = {}

    def flush_for_migration(self, req):
        results = self.flush_results.get(req, [0])
        return results.pop(0) if len(results) > 1 else results[0]

    def finalize_release_on_finish(self, req):
        if req not in self.device_copy:
            self.released.append(req)

    def check_offload_progress(self):
        pass

    def has_pending_device_copy(self, req):
        return req in self.device_copy

    def has_pending_storage_write(self, req):
        return req in self.storage_write

    def has_pending_transfers(self):
        return bool(self.device_copy or self.storage_write)

    def storage_confirmed_tokens(self, req):
        return self.confirmed.get(req, 0)


class _Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def _exporter(offload, clock=None):
    emitted = []
    exporter = DecodeMigrationExporter(
        offload_manager=offload,
        emit_outputs=lambda reqs: emitted.extend(reqs),
        clock=clock or _Clock(),
    )
    exporter.arm(flush_timeout_s=10.0)
    return exporter, emitted


class TestDecodeMigrationExporter(CustomTestCase):
    def test_request_with_nothing_to_flush_releases_its_kv(self):
        """Regression: release rode on a device-copy ack, so a request whose
        flush queued nothing (e.g. it had only the token from prefill) kept its
        device KV forever. Uses the real offload manager."""
        manager, freed = _make_manager(page_size=4, stride=4)
        req = _make_req(prompt_len=8, output_len=1)
        manager.req_to_token_pool.free.side_effect = lambda r: setattr(
            r.kv, "req_pool_idx", None
        )
        exporter, emitted = _exporter(manager)
        manager.check_offload_progress = lambda: None

        exporter.begin([req])

        self.assertIsInstance(req.finished_reason, FINISH_MIGRATED)
        self.assertEqual(len(freed), 1)
        exporter.poll()
        self.assertEqual(emitted, [req])
        self.assertFalse(exporter.has_pending())

    def test_request_is_reported_only_after_every_rank_stored_it(self):
        """The hand-off tells a router to read the KV back; reporting before a
        peer rank's storage write lands would make that read miss."""
        offload = _FakeOffload()
        req = _make_req()
        exporter, emitted = _exporter(offload)
        peer_votes = {"req": 0}

        def all_reduce(tensor, op, group):
            tensor[0] = min(int(tensor[0]), peer_votes["req"])

        exporter._tp_world_size = 2
        with patch.object(torch.distributed, "all_reduce", all_reduce):
            exporter.begin([req])
            exporter.poll()
            self.assertEqual(emitted, [])
            peer_votes["req"] = 1
            exporter.poll()
        self.assertEqual(emitted, [req])

    def test_host_full_request_stays_unfinished_until_its_flush_is_queued(self):
        """Finishing marks the device KV releasable on the next copy ack; a
        request still waiting for host memory needs that KV for its retry."""
        req = _make_req()
        offload = _FakeOffload(flush_results={req: [-1, 8]})
        exporter, emitted = _exporter(offload)

        exporter.begin([req])
        self.assertIsNone(req.finished_reason)
        self.assertEqual(offload.released, [])

        offload.device_copy.add(req)  # the retried flush is now copying
        exporter.poll()
        self.assertIsInstance(req.finished_reason, FINISH_MIGRATED)
        self.assertEqual(emitted, [])
        offload.device_copy.clear()
        exporter.poll()
        self.assertEqual(emitted, [req])

    def test_timed_out_request_is_reported_but_backups_keep_it_busy(self):
        """After the deadline the request is handed off with what was stored,
        but the instance is not idle while that backup still reads host memory."""
        clock = _Clock()
        req = _make_req()
        offload = _FakeOffload()
        offload.storage_write.add(req)
        offload.confirmed[req] = 8
        exporter, emitted = _exporter(offload, clock)

        exporter.begin([req])
        exporter.poll()
        self.assertEqual(emitted, [])
        clock.now = 11.0
        exporter.poll()
        self.assertEqual(emitted, [req])
        self.assertEqual(req.finished_reason.to_json()["kv_flushed_tokens"], 8)
        self.assertTrue(exporter.has_pending())
        offload.storage_write.clear()
        exporter.poll()
        self.assertFalse(exporter.has_pending())

    def test_request_that_cannot_be_resubmitted_is_aborted_with_503(self):
        req = _make_req()
        req.return_logprob = True
        offload = _FakeOffload()
        exporter, emitted = _exporter(offload)

        exporter.begin([req])

        self.assertIsInstance(req.finished_reason, FINISH_ABORT)
        self.assertEqual(req.finished_reason.status_code, 503)
        self.assertEqual(offload.released, [req])
        self.assertEqual(emitted, [req])

    def test_request_its_router_cannot_resume_is_aborted_with_503(self):
        """A request whose router did not opt in would reach its client as a
        "migrated" finish nobody resumes; it must end with a retryable 503."""
        req = _make_req(allow_migration=False)
        exporter, emitted = _exporter(_FakeOffload())

        exporter.begin([req])

        self.assertIsInstance(req.finished_reason, FINISH_ABORT)
        self.assertEqual(req.finished_reason.status_code, 503)
        self.assertEqual(emitted, [req])

    def test_state_a_resubmission_cannot_carry_blocks_migration(self):
        """Per-token side outputs and a thinking budget restart from zero on
        the new decode, so the client would get silently wrong results."""
        for name, req_kwargs in (
            ("hidden states", {"return_hidden_states": True}),
            ("last hidden state", {"return_hidden_states": "last"}),
            ("routed experts", {"return_routed_experts": True}),
            ("indexer top-k", {"return_indexer_topk": True}),
            ("sampling mask", {"return_sampling_mask": True}),
            (
                "thinking budget",
                {
                    "sampling_params": SamplingParams(
                        custom_params={"thinking_budget": 16}
                    )
                },
            ),
        ):
            with self.subTest(name):
                self.assertIsNotNone(migration_blocker(_make_req(**req_kwargs)))
        self.assertIsNone(migration_blocker(_make_req()))

    def test_done_only_after_a_migrate_pause_finished_its_hand_offs(self):
        """Worker pods of a decode group exit on this; an instance that was
        merely idle must not report it."""
        offload = _FakeOffload()
        exporter = DecodeMigrationExporter(
            offload_manager=offload, emit_outputs=lambda reqs: None, clock=_Clock()
        )
        self.assertFalse(exporter.is_done())
        exporter.arm(flush_timeout_s=10.0)
        self.assertFalse(exporter.is_done())
        req = _make_req()
        offload.storage_write.add(req)
        exporter.begin([req])
        self.assertFalse(exporter.is_done())
        offload.storage_write.clear()
        exporter.poll()
        self.assertTrue(exporter.is_done())
        exporter.arm(flush_timeout_s=10.0)
        self.assertFalse(exporter.is_done())

    def test_finish_reason_carries_the_resubmission_tokens(self):
        """The router has no tokenizer: the tokens must come with the hand-off."""
        reason = FINISH_MIGRATED(origin_input_ids=[1, 2], output_ids=[3])
        reason.kv_flushed_tokens = 4
        self.assertEqual(
            reason.to_json(),
            {
                "type": "migrated",
                "origin_input_ids": [1, 2],
                "output_ids": [3],
                "kv_flushed_tokens": 4,
            },
        )


class TestMigrationDoneState(CustomTestCase):
    """/server_info's migration_done, which a decode group's worker pods exit on."""

    @staticmethod
    def _state(*, exporter, idle):
        scheduler = SimpleNamespace(
            disaggregation_mode=DisaggregationMode.DECODE,
            tp_worker=SimpleNamespace(get_decode_cuda_graph_bs=lambda: [1]),
            is_fully_idle=lambda: idle,
            decode_migration_exporter=exporter,
        )
        controller = SchedulerInternalStateController(scheduler, record_step_time=False)
        state = {"memory_usage": {"graph": defaultdict(float)}}
        controller._add_disaggregation(state)
        return state

    def test_reported_only_once_the_hand_off_finished_and_the_rank_is_idle(self):
        finished = SimpleNamespace(is_done=lambda: True)
        unfinished = SimpleNamespace(is_done=lambda: False)
        for name, exporter, idle, expected in (
            ("never paused or still handing off", unfinished, True, False),
            ("hand-off done", finished, True, True),
            ("hand-off done, rank busy", finished, False, False),
            ("no exporter: not a migration-capable decode", None, True, False),
        ):
            with self.subTest(name):
                state = self._state(exporter=exporter, idle=idle)
                self.assertIs(state["migration_done"], expected)
                self.assertIs(state["fully_idle"], idle)


class TestDecodeReceiveSideIdle(CustomTestCase):
    def test_waits_for_requests_still_receiving_or_without_kv(self):
        """The paused loop no longer polls KV transfers, so pausing with a
        request mid-transfer would leave the instance busy forever."""
        receiving = IndexedReqList(ListKind.PD_RECEIVING)
        waiting = IndexedReqList(ListKind.WAITING)
        lists = {ListKind.PD_RECEIVING: receiving, ListKind.WAITING: waiting}
        self.assertTrue(decode_receive_side_idle(lists=lists, waiting_queue=waiting))

        receiving.join([_make_req(rid="a")])
        self.assertFalse(decode_receive_side_idle(lists=lists, waiting_queue=waiting))
        receiving.take()

        without_kv = _make_req(rid="b")
        without_kv.kv = ReqKvInfo()
        waiting.join([without_kv])
        self.assertFalse(decode_receive_side_idle(lists=lists, waiting_queue=waiting))


if __name__ == "__main__":
    unittest.main()
