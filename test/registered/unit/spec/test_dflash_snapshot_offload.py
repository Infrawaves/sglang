import queue
import threading
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.mem_cache.kv.memory.views import MLAKVView
from sglang.srt.mem_cache.snapshot_transfer import CommitGate, SnapshotPool
from sglang.srt.speculative import dflash_offload as offload
from sglang.srt.speculative.dflash_snapshot_rows import (
    OffloadAttempt,
    VerifyCommit,
    committed_slice,
    prompt_spans,
)
from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci

register_cpu_ci(est_time=4, suite="base-a-test-cpu")
register_cuda_ci(est_time=8, stage="base-b", runner_config="1-gpu-small")


def request(prompt=(11, 12), reply=(), rid="sample"):
    return SimpleNamespace(
        rid=rid,
        origin_input_ids=prompt,
        output_ids=list(reply),
        finished_len=None,
        finished_reason=None,
        is_retracted=False,
        retraction_count=0,
        multimodal_inputs=None,
        input_embeds=None,
        kv=SimpleNamespace(kv_committed_len=len(prompt), req_pool_idx=0),
    )


def test_default_allocates_no_buffers():
    offload.reset()
    assert not offload.wants_teacher()
    assert offload.configure_target_model(SimpleNamespace(is_draft_worker=False)) == 0
    assert offload.initialize(None) is None
    assert offload.runtime is None


def test_configure_target_model_shared_buffer_and_all_rank_budget(monkeypatch):
    offload.reset()
    try:
        offload.register(
            offload.OffloadConfig(
                layers=(1,),
                select=lambda req: None,
                consume=lambda a, r: None,
                gpu_bytes=256,
            )
        )
        monkeypatch.setattr(
            "sglang.srt.runtime_context.get_disagg",
            lambda: SimpleNamespace(disaggregation_mode="decode"),
        )
        rank = {"tp_rank": 0}
        monkeypatch.setattr(
            "sglang.srt.runtime_context.get_parallel",
            lambda: SimpleNamespace(tp_rank=rank["tp_rank"]),
        )

        # Wrapper model (e.g. VLM with language_model.logits_processor) works.
        lp = SimpleNamespace(
            offload_verify_hidden=False, offload_verify_hidden_buffer=None
        )
        runner = SimpleNamespace(
            is_draft_worker=False,
            device="cpu",
            model_config=SimpleNamespace(hidden_size=4, dtype=torch.bfloat16),
            model=SimpleNamespace(language_model=SimpleNamespace(logits_processor=lp)),
            graph_shared_output=SimpleNamespace(max_rows=16),
        )
        # 16 * 4 * 2 = 128 bytes < 256
        assert offload.configure_target_model(runner) == 128
        assert lp.offload_verify_hidden is True
        assert lp.offload_verify_hidden_buffer.shape == (16, 4)
        assert lp.offload_verify_hidden_buffer.dtype == torch.bfloat16
        assert runner.offload_shared_hidden_bytes == 128

        # Non-zero TP rank validates budget too, without enabling hidden output.
        rank["tp_rank"] = 1
        lp1 = SimpleNamespace(
            offload_verify_hidden=False, offload_verify_hidden_buffer=None
        )
        runner1 = SimpleNamespace(
            is_draft_worker=False,
            device="cpu",
            model_config=SimpleNamespace(hidden_size=4, dtype=torch.bfloat16),
            model=SimpleNamespace(logits_processor=lp1),
            graph_shared_output=SimpleNamespace(max_rows=16),
        )
        assert offload.configure_target_model(runner1) == 0
        assert lp1.offload_verify_hidden is False
        assert lp1.offload_verify_hidden_buffer is None
        assert runner1.offload_shared_hidden_bytes == 0

        # Exceeding budget fails on non-zero rank as well (prevents TP barrier hang).
        runner1.graph_shared_output = SimpleNamespace(max_rows=32)  # 256 >= 256
        with pytest.raises(ValueError, match="budget"):
            offload.configure_target_model(runner1)

        # Missing LogitsProcessor fails fast.
        runner_no_lp = SimpleNamespace(
            is_draft_worker=False,
            device="cpu",
            model_config=SimpleNamespace(hidden_size=4, dtype=torch.bfloat16),
            model=SimpleNamespace(),
            graph_shared_output=None,
        )
        with pytest.raises(ValueError, match="LogitsProcessor"):
            offload.configure_target_model(runner_no_lp)
    finally:
        offload.reset()


def test_logits_processor_writes_into_shared_offload_buffer():
    pytest.importorskip("triton")
    from sglang.srt.layers.logits_processor import LogitsMetadata, LogitsProcessor
    from sglang.srt.model_executor.forward_mode import ForwardMode

    lp = object.__new__(LogitsProcessor)
    lp.keep_last_hidden_states = False
    lp.offload_verify_hidden = True
    shared_buf = torch.zeros((6, 4), dtype=torch.bfloat16)
    lp.offload_verify_hidden_buffer = shared_buf

    meta = LogitsMetadata(forward_mode=ForwardMode.TARGET_VERIFY)
    h1 = torch.arange(16, dtype=torch.bfloat16).reshape(4, 4)
    out1 = lp._get_last_hidden_states_to_store(h1, None, meta)
    assert out1.data_ptr() == shared_buf.data_ptr()
    assert torch.equal(out1, h1)

    # Larger eager batch falls back to returning hidden_states directly.
    h_big = torch.ones((8, 4), dtype=torch.bfloat16)
    out_big = lp._get_last_hidden_states_to_store(h_big, None, meta)
    assert out_big is h_big

    # Non-verify mode returns None.
    assert (
        lp._get_last_hidden_states_to_store(
            h1, None, LogitsMetadata(forward_mode=ForwardMode.DECODE)
        )
        is None
    )


def test_prompt_coverage():
    assert list(prompt_spans(11, {0, 3}, 2, 3)) == [(2, 5), (5, 6), (8, 11)]
    with pytest.raises(ValueError, match="complete"):
        list(prompt_spans(11, {5}, 2, 3))


class HostPool:
    """Defer delivery, preserving the snapshot before deliberate source reuse."""

    def __init__(self, *args):
        self.pending = []
        self.callbacks = []
        self.fences = 0
        self.guard_calls = []
        self.submitted_streams = []
        self.submit_count = 0
        self.closed = False

    def fence_reuse(self, stream=None):
        self.fences += 1

    def guard_release(self, allocator, *, forward_stream=None):
        self.fences += 1
        self.guard_calls.append((allocator, forward_stream))

    def submit(self, nbytes, fill, deliver, fail, *, stream=None, gate=None):
        if self.closed:
            fail("offload closed")
            return False
        self.submit_count += 1
        self.submitted_streams.append(stream)
        buf = torch.empty(nbytes, dtype=torch.uint8)
        fill(buf, stream)
        if gate is not None:
            gate.on_dma(
                (None, buf, None),
                nbytes,
                None,
                lambda slot, nb, fn, fl: self.pending.append(lambda: fn(buf)),
            )
        else:
            self.pending.append(lambda: deliver(buf))
        return True

    def submit_callback(self, fn):
        if self.closed:
            return False
        self.callbacks.append(fn)
        return True

    def drain(self):
        while self.pending or self.callbacks:
            pending, self.pending = self.pending, []
            for deliver in pending:
                deliver()
            callbacks, self.callbacks = self.callbacks, []
            for cb in callbacks:
                cb()

    def close(self):
        self.closed = True
        self.drain()


def runtime(
    monkeypatch,
    role,
    attempts,
    packets,
    *,
    on_complete=None,
    decode_batch_reqs=8,
    forward_stream=None,
    allocator=None,
):
    monkeypatch.setattr(offload, "SnapshotPool", HostPool)
    monkeypatch.setattr(torch.Tensor, "record_stream", lambda self, stream: None)
    view = MLAKVView()
    view.dcp_replicated = view.dsa_kv_cache_store_fp8 = False
    view.kv_lora_rank, view.qk_rope_head_dim = 2, 0
    buffers = {
        i: (torch.arange(40).reshape(10, 1, 4) + 100 * i).to(torch.bfloat16)[:, :, :2]
        for i in (1, 3)
    }

    def layer(i):
        return SimpleNamespace(owner=True, view=view, key=lambda: buffers[i])

    router = SimpleNamespace(layer=layer, read=layer)
    lp = SimpleNamespace(offload_verify_hidden=(role == "decode"))
    runner = SimpleNamespace(
        token_to_kv_pool=SimpleNamespace(layer_router=router),
        token_to_kv_pool_allocator=allocator,
        forward_stream=forward_stream,
        runner_ids=SimpleNamespace(translate_dcp_read_ids=lambda ids: ids // 2 + 1),
        req_to_token_pool=SimpleNamespace(req_to_token=torch.tensor([[5, 8, 2, 9, 4]])),
        model_config=SimpleNamespace(hidden_size=4, dtype=torch.bfloat16),
        model=SimpleNamespace(logits_processor=lp),
        offload_shared_hidden_bytes=0,
        device="cpu",
    )
    select_fn = (
        attempts
        if callable(attempts)
        else (lambda req: attempts if req.rid == "sample" else None)
    )
    cfg = offload.OffloadConfig(
        (1, 3),
        select_fn,
        lambda selected, packet: packets.append((selected, packet)),
        on_complete=on_complete,
        prompt_chunk_rows=2,
        decode_batch_reqs=decode_batch_reqs,
    )
    return offload.OffloadRuntime(runner, 4, 2, role, 2, 0, cfg), buffers


def test_prefill_uses_verified_pages_forward_stream_and_on_complete(monkeypatch):
    packets = []
    completed = []
    fake_stream = object()
    fake_alloc = object()
    attempt = OffloadAttempt("capture", "attempt", frozenset({0}))
    rt, buffers = runtime(
        monkeypatch,
        "prefill",
        attempt,
        packets,
        on_complete=completed.append,
        forward_stream=fake_stream,
        allocator=fake_alloc,
    )
    expected_kv = torch.stack([buffers[i][2, 0] for i in (1, 3)]).view(torch.uint8)
    req = request(prompt=(10, 11, 12, 13, 14))
    req.finished_reason = SimpleNamespace(to_json=lambda: {"type": "length"})
    rt.before_release(req)
    for value in buffers.values():
        value.zero_()
    assert rt.pool.fences == 2
    assert rt.pool.submitted_streams == [fake_stream, fake_stream]
    assert rt.pool.guard_calls == [(fake_alloc, fake_stream)]
    assert not packets
    assert not completed
    rt.pool.drain()
    rows = [p for _, p in packets]
    assert [p.positions.tolist() for p in rows] == [[2, 3], [4]]
    assert [p.input_ids.tolist() for p in rows] == [[12, 13], [14]]
    assert [p.physical_ids.tolist() for p in rows] == [[2], [3]]
    assert torch.equal(rows[0].raw_kv[0], expected_kv)
    assert all(p.final_hidden is None and p.next_token_ids is None for p in rows)
    assert rows[0].layout.dcp_size == 2
    assert rows[0].layout.page_size == 2
    assert rows[0].layout.format_version == 1
    assert completed == [attempt]
    assert attempt.error is None


def test_verify_batches_multiple_requests_and_commits_event_driven(monkeypatch):
    a1 = OffloadAttempt("cap1", "att1")
    a2 = OffloadAttempt("cap2", "att2")
    packets = []
    completed = []
    rt, buffers = runtime(
        monkeypatch,
        "decode",
        lambda req: {"r1": a1, "r2": a2}.get(req.rid),
        packets,
        on_complete=completed.append,
        decode_batch_reqs=8,
    )
    expected_kv_r1 = torch.stack([buffers[i][5, 0] for i in (1, 3)]).view(torch.uint8)
    expected_kv_r2 = torch.stack([buffers[i][2, 0] for i in (1, 3)]).view(torch.uint8)
    skipped = request(rid="unsampled")
    req1 = request(prompt=(11, 12), reply=(20,), rid="r1")
    req2 = request(prompt=(31, 32), reply=(40,), rid="r2")
    positions = torch.tensor([0, 0, 0, 0, 2, 3, 4, 5, 2, 3, 4, 5])
    inputs = torch.tensor([0, 0, 0, 0, 20, 42, 77, 88, 40, 55, 66, 77])
    ids = torch.tensor([0, 0, 0, 0, 5, 8, 2, 9, 2, 7, 4, 3])
    hidden = torch.arange(48, dtype=torch.bfloat16).reshape(12, 4)

    commits = rt.snapshot_verify([skipped, req1, req2], positions, inputs, ids, hidden, 4)
    # Both selected requests packed into a single slot / single D2H submission.
    assert len(commits) == 1
    assert rt.pool.submit_count == 1
    for value in (positions, inputs, ids, hidden, *buffers.values()):
        value.zero_()

    # Before commit.after(), nothing is dispatched to consumer yet.
    rt.pool.drain()
    assert not packets

    commits[0].before()
    req1.output_ids.extend([42, 99, 100])
    req1.finished_len = 3
    req1.finished_reason = SimpleNamespace(to_json=lambda: {"type": "stop"})
    req2.output_ids.extend([55])
    # Release req1 during result processing (before commit.after(), matching batch_result_processor).
    rt.before_release(req1)
    assert not completed  # Still has in-flight verify packet!
    commits[0].after()
    rt.pool.drain()

    assert len(packets) == 2
    (att1, p1), (att2, p2) = packets
    assert att1 is a1 and att2 is a2
    assert p1.positions.tolist() == [2, 3]
    assert p1.input_ids.tolist() == [20, 42]
    assert p1.next_token_ids.tolist() == [42, 99]
    assert p1.final_hidden.tolist() == [[16, 17, 18, 19], [20, 21, 22, 23]]
    assert p1.owned.tolist() == [False, True]
    assert p1.physical_ids.tolist() == [5]
    assert torch.equal(p1.raw_kv[0], expected_kv_r1)

    assert p2.positions.tolist() == [2]
    assert p2.input_ids.tolist() == [40]
    assert p2.next_token_ids.tolist() == [55]
    assert p2.owned.tolist() == [True]
    assert p2.physical_ids.tolist() == [2]
    assert torch.equal(p2.raw_kv[0], expected_kv_r2)

    # req1 completed strictly after its packet was delivered; req2 still active.
    assert completed == [a1]
    assert a1.error is None


def test_verify_commit_gate_handles_both_orders_and_shutdown_abort():
    events = []
    dispatched = []

    def dispatch(slot, nbytes, fn, fail):
        dispatched.append((slot, nbytes, fn, fail))

    # Order 1: commit resolves before DMA finishes.
    gate1 = CommitGate(
        lambda host, payload: events.append(("deliver1", payload)),
        lambda reason: events.append(("fail1", reason)),
    )
    gate1.on_commit(("ok",))
    assert not dispatched
    gate1.on_dma("slot1", 16, None, dispatch)
    assert len(dispatched) == 1
    dispatched[0][2](torch.zeros(16, dtype=torch.uint8))
    assert events == [("deliver1", ("ok",))]

    # Order 2: DMA finishes before commit; commit passes None (shutdown/abort).
    gate2 = CommitGate(
        lambda host, payload: events.append(("deliver2", payload)),
        lambda reason: events.append(("fail2", reason)),
    )
    gate2.on_dma("slot2", 8, None, dispatch)
    assert len(dispatched) == 1
    gate2.on_commit(None)
    assert len(dispatched) == 2
    dispatched[1][2](torch.zeros(8, dtype=torch.uint8))
    assert events[-1] == ("deliver2", None)


def test_close_aborts_unresolved_verify_commits_without_blocking(monkeypatch):
    attempt = OffloadAttempt("cap", "att")
    packets = []
    completed = []
    rt, _ = runtime(
        monkeypatch, "decode", attempt, packets, on_complete=completed.append
    )
    req = request(reply=(20,))
    commits = rt.snapshot_verify(
        [req],
        torch.tensor([2, 3, 4, 5]),
        torch.tensor([20, 42, 77, 88]),
        torch.tensor([5, 8, 2, 9]),
        torch.zeros((4, 4), dtype=torch.bfloat16),
        4,
    )
    assert len(commits) == 1
    # Never call commits[0].after() (simulates overlap graceful exit).
    rt.close()
    assert attempt.error == "offload closed"
    assert not packets
    assert completed == [attempt]


def test_guard_release_stages_or_syncs_in_flight_gathers():
    pool = object.__new__(SnapshotPool)
    pool.device = torch.device("cpu")
    pool.lock = threading.RLock()
    pool.gathering = False
    fwd_stream = SimpleNamespace(cuda_stream=1001, waited=[])
    # Distinct Python wrapper with the same underlying cuda_stream handle
    # (mirrors torch.cuda.current_stream() returning a fresh Stream object).
    fwd_wrapper = SimpleNamespace(cuda_stream=1001, waited=[])
    fences = []
    pool.fence_reuse = lambda stream=None: fences.append(stream)

    # Case 1: allocator supports reuse guard and all reads share forward_stream's handle
    # -> stage without host sync AND without stalling schedule_stream via fence_reuse().
    synced = []
    ev1 = SimpleNamespace(query=lambda: False, synchronize=lambda: synced.append("ev1"))
    pool.reads = [(ev1, fwd_wrapper)]
    guarded = []
    alloc_staged = SimpleNamespace(
        supports_reuse_guard=True,
        need_sort=True,
        set_reuse_guard_event=guarded.append,
    )
    pool.guard_release(alloc_staged, forward_stream=fwd_stream)
    assert guarded == [ev1]
    assert synced == []
    assert fences == []

    # Case 2: allocator has no reuse guard -> fences and synchronizes pending gather event.
    done = {"flag": False}
    ev2 = SimpleNamespace(
        query=lambda: done["flag"],
        synchronize=lambda: (synced.append("ev2"), done.__setitem__("flag", True)),
    )
    pool.reads = [(ev2, fwd_stream)]
    alloc_plain = SimpleNamespace(supports_reuse_guard=False, need_sort=False)
    pool.guard_release(alloc_plain, forward_stream=fwd_stream)
    assert synced == ["ev2"]
    assert len(fences) == 1
    assert pool.reads == []


def test_fence_reuse_skips_same_cuda_stream_handle():
    pool = object.__new__(SnapshotPool)
    pool.device = torch.device("cpu")
    pool.lock = threading.RLock()
    ev = SimpleNamespace(query=lambda: False)
    s_same_a = SimpleNamespace(cuda_stream=777, waited=[])
    s_same_b = SimpleNamespace(cuda_stream=777, waited=[])
    s_same_b.wait_event = s_same_b.waited.append
    s_other = SimpleNamespace(cuda_stream=888, waited=[])
    s_other.wait_event = s_other.waited.append
    pool.reads = [(ev, s_same_a)]

    # Same underlying CUDA stream handle: already stream-ordered -> no wait_event.
    pool.fence_reuse(stream=s_same_b)
    assert s_same_b.waited == []

    # Different CUDA stream handle: must wait_event.
    pool.fence_reuse(stream=s_other)
    assert s_other.waited == [ev]


def test_submit_fill_failure_resets_gathering_and_recycles_slot(monkeypatch):
    pool = object.__new__(SnapshotPool)
    pool.device = torch.device("cpu")
    pool.lock = threading.RLock()
    pool.gathering = False
    pool.closed = False
    pool.reads = []
    pool.quarantine = []
    pool.free = queue.LifoQueue()
    slot = (torch.empty(16, dtype=torch.uint8), torch.empty(16, dtype=torch.uint8), None)
    pool.free.put_nowait(slot)
    synced = []
    fake_stream = SimpleNamespace(synchronize=lambda: synced.append(True))
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device=None: fake_stream)
    monkeypatch.setattr(torch.cuda, "Event", lambda: SimpleNamespace())

    errors = []

    def bad_fill(buf, stream):
        assert pool.reading() is True
        raise RuntimeError("boom")

    ok = pool.submit(8, bad_fill, lambda h: None, errors.append)
    assert ok is False
    assert synced == [True]
    assert pool.gathering is False
    assert pool.reading() is False
    assert pool.free.qsize() == 1
    assert errors == ["snapshot gather failed"]


def test_inspect_layout_fp8_packed_and_unpacked_checks():
    packed_view = MLAKVView()
    packed_view.dcp_replicated = False
    packed_view.dsa_kv_cache_store_fp8 = True
    packed_view.kv_scale_ue8m0 = True
    packed_view.kv_lora_rank = 128
    packed_view.qk_rope_head_dim = 16
    # 128 + (128 // 128) * 4 + 16 * 2 = 164 bytes
    packed_buf = torch.empty((4, 1, 164), dtype=torch.uint8)
    router = SimpleNamespace(
        layer=lambda i: SimpleNamespace(owner=True, view=packed_view, key=lambda: packed_buf)
    )
    layout = offload.inspect_layout(router, (0,), dcp_size=8, page_size=64)
    assert layout.block_scaled_fp8 is True
    assert layout.power_of_two_scales is True
    assert layout.scale_block_size == 128
    assert layout.dcp_size == 8
    assert layout.page_size == 64

    # Unpacked FP8 with fp8_cast_on_write=False must be rejected.
    raw_fp8_view = MLAKVView()
    raw_fp8_view.dcp_replicated = False
    raw_fp8_view.dsa_kv_cache_store_fp8 = False
    raw_fp8_view.fp8_cast_on_write = False
    raw_fp8_view.kv_lora_rank = 16
    raw_fp8_view.qk_rope_head_dim = 8
    fp8_buf = torch.empty((4, 1, 24), dtype=torch.float8_e4m3fn)
    router_bad = SimpleNamespace(
        layer=lambda i: SimpleNamespace(owner=True, view=raw_fp8_view, key=lambda: fp8_buf)
    )
    with pytest.raises(ValueError, match="encoding"):
        offload.inspect_layout(router_bad, (0,))


def test_attempt_skips_input_embeds_and_runtime_rejects_unconfigured_target(monkeypatch):
    packets = []
    attempt = OffloadAttempt("cap", "att")
    rt, _ = runtime(monkeypatch, "prefill", attempt, packets)
    req = request()
    req.input_embeds = torch.zeros((2, 4))
    assert rt._attempt(req) is None

    # Decode rank 0 with target model lacking offload_verify_hidden must fail fast.
    view = MLAKVView()
    view.dcp_replicated = view.dsa_kv_cache_store_fp8 = False
    view.kv_lora_rank, view.qk_rope_head_dim = 2, 0
    buf = torch.zeros((4, 1, 2), dtype=torch.bfloat16)
    router = SimpleNamespace(
        layer=lambda i: SimpleNamespace(owner=True, view=view, key=lambda: buf)
    )
    runner = SimpleNamespace(
        token_to_kv_pool=SimpleNamespace(layer_router=router),
        runner_ids=SimpleNamespace(translate_dcp_read_ids=lambda ids: ids),
        req_to_token_pool=SimpleNamespace(req_to_token=torch.zeros((1, 4), dtype=torch.int64)),
        model_config=SimpleNamespace(hidden_size=4, dtype=torch.bfloat16),
        model=SimpleNamespace(logits_processor=SimpleNamespace(offload_verify_hidden=False)),
        device="cpu",
    )
    cfg = offload.OffloadConfig((0,), lambda r: None, lambda a, p: None)
    with pytest.raises(ValueError, match="verify hidden offload enabled"):
        offload.OffloadRuntime(runner, 4, 2, "decode", 1, 0, cfg)


def test_initialize_rejects_page_interleaved_and_cp_sharded(monkeypatch):
    offload.reset()
    try:
        import sys
        import types

        class FakeDFlashWorkerV2:
            pass

        fake_mod = types.ModuleType("sglang.srt.speculative.dflash_worker_v2")
        fake_mod.DFlashWorkerV2 = FakeDFlashWorkerV2
        monkeypatch.setitem(sys.modules, "sglang.srt.speculative.dflash_worker_v2", fake_mod)

        cp_flag = {"sharded": False}
        fake_cp = types.ModuleType("sglang.srt.layers.cp.base")
        fake_cp.is_cp_sharded_padded = lambda: cp_flag["sharded"]
        monkeypatch.setitem(sys.modules, "sglang.srt.layers.cp.base", fake_cp)

        fake_storage = types.ModuleType("sglang.srt.mem_cache.static_kv_storage")
        fake_storage.kv_storage_of = lambda *a, **kw: None
        monkeypatch.setitem(sys.modules, "sglang.srt.mem_cache.static_kv_storage", fake_storage)

        offload.register(
            offload.OffloadConfig((0,), lambda r: None, lambda a, p: None)
        )
        monkeypatch.setattr(
            "sglang.srt.runtime_context.get_parallel",
            lambda: SimpleNamespace(
                pp_size=1, dp_size=1, tp_size=2, attn_dcp_size=2, tp_rank=0, attn_dcp_rank=0
            ),
        )
        fake_sched = SimpleNamespace(
            disaggregation_mode=SimpleNamespace(value="decode"),
            draft_worker=FakeDFlashWorkerV2(),
            kv_shard_widening=2,
        )
        with pytest.raises(ValueError, match="token-interleaved"):
            offload.initialize(fake_sched)

        fake_sched.kv_shard_widening = 1
        cp_flag["sharded"] = True
        with pytest.raises(ValueError, match="token-interleaved"):
            offload.initialize(fake_sched)
    finally:
        offload.reset()


def test_prefill_consumer_error_records_attempt_error_before_on_complete(monkeypatch):
    completed = []
    attempt = OffloadAttempt("cap", "att")

    def boom(selected, packet):
        raise ValueError("disk full")

    monkeypatch.setattr(offload, "SnapshotPool", HostPool)
    monkeypatch.setattr(torch.Tensor, "record_stream", lambda self, stream: None)
    view = MLAKVView()
    view.dcp_replicated = view.dsa_kv_cache_store_fp8 = False
    view.kv_lora_rank, view.qk_rope_head_dim = 2, 0
    buf = torch.zeros((10, 1, 2), dtype=torch.bfloat16)
    router = SimpleNamespace(
        layer=lambda i: SimpleNamespace(owner=True, view=view, key=lambda: buf),
        read=lambda i: SimpleNamespace(owner=True, view=view, key=lambda: buf),
    )
    runner = SimpleNamespace(
        token_to_kv_pool=SimpleNamespace(layer_router=router),
        runner_ids=SimpleNamespace(translate_dcp_read_ids=lambda ids: ids),
        req_to_token_pool=SimpleNamespace(req_to_token=torch.tensor([[1, 2]])),
        model_config=SimpleNamespace(hidden_size=4, dtype=torch.bfloat16),
        device="cpu",
    )
    cfg = offload.OffloadConfig(
        (0,),
        lambda req: attempt,
        boom,
        on_complete=lambda a: completed.append((a, a.error)),
        prompt_chunk_rows=2,
    )
    rt = offload.OffloadRuntime(runner, 4, 2, "prefill", 1, 0, cfg)
    req = request(prompt=(10, 11))
    req.finished_reason = SimpleNamespace(to_json=lambda: {"type": "length"})
    rt.before_release(req)
    rt.pool.drain()
    assert completed == [(attempt, "disk full")]


def test_verify_chunks_selected_requests_by_decode_batch_reqs(monkeypatch):
    attempts = {f"r{i}": OffloadAttempt(f"c{i}", f"a{i}") for i in range(3)}
    packets = []
    rt, _ = runtime(
        monkeypatch,
        "decode",
        lambda req: attempts.get(req.rid),
        packets,
        decode_batch_reqs=2,
    )
    reqs = [request(prompt=(11, 12), reply=(20 + i,), rid=f"r{i}") for i in range(3)]
    positions = torch.tensor([2, 3, 4, 5] * 3)
    inputs = torch.tensor([20, 42, 77, 88, 21, 43, 77, 88, 22, 44, 77, 88])
    ids = torch.tensor([2, 4, 6, 8] * 3)
    hidden = torch.zeros((12, 4), dtype=torch.bfloat16)

    commits = rt.snapshot_verify(reqs, positions, inputs, ids, hidden, 4)
    # 3 selected requests with decode_batch_reqs=2 -> 2 packed chunk submissions (2 + 1).
    assert len(commits) == 2
    assert rt.pool.submit_count == 2
    for c in commits:
        c.before()
    for i, r in enumerate(reqs):
        r.output_ids.append(42 + i)
    for c in commits:
        c.after()
    rt.pool.drain()
    assert len(packets) == 3
    assert [a.capture_id for a, _ in packets] == ["c0", "c1", "c2"]


@pytest.mark.parametrize(
    "kind",
    ["bonus", "first", "no_commit", "wrong_token", "wrong_position", "retracted"],
)
def test_commit_boundaries(kind):
    req = request(reply=() if kind == "first" else (20,))
    attempt = OffloadAttempt("capture", "attempt")
    commit = VerifyCommit(req, attempt)
    commit.before()
    if kind != "no_commit":
        req.output_ids.extend([42, 43, 99])
    if kind == "retracted":
        req.retraction_count += 1
    commit.after()
    path = commit.path.result()
    if kind == "retracted":
        assert attempt.error and path is None
        return
    positions = torch.arange(1 if kind == "first" else 2, 5 if kind == "first" else 6)
    inputs = torch.tensor([12 if kind == "first" else 20, 42, 43, 33])
    if kind in ("wrong_token", "wrong_position"):
        (positions if kind == "wrong_position" else inputs)[1] += 1
        with pytest.raises(ValueError, match="predecessor"):
            committed_slice(path, 2, positions, inputs)
        return
    rows, labels = committed_slice(path, 2, positions, inputs)
    expected = {"first": [43, 99], "bonus": [42, 43, 99], "no_commit": []}[kind]
    assert list(labels) == expected
    assert len(positions[rows]) == len(expected)


@pytest.mark.parametrize("dma_failure", [False, True])
def test_consumer_error_recycles_only_after_confirmed_dma(dma_failure):
    pool = object.__new__(SnapshotPool)
    pool.lock = threading.RLock()
    pool.closed, pool.quarantine, pool.free = False, [], queue.LifoQueue()

    def wait():
        if dma_failure:
            raise RuntimeError("lost event")

    def bad_consumer(buf):
        raise ValueError("consumer error")

    slot = (None, torch.zeros(2), SimpleNamespace(synchronize=wait))
    errors = []
    pool._deliver(slot, 2, bad_consumer, errors.append)
    assert bool(pool.quarantine) == pool.closed == dma_failure
    assert pool.free.empty() == dma_failure
    assert errors == (["D2H completion unknown"] if dma_failure else ["consumer error"])


def test_close_resets_runtime_even_when_pool_raises():
    offload.reset()

    class BrokenRuntime:
        def close(self):
            raise RuntimeError("Snapshot completion unknown on shutdown")

    offload.runtime = BrokenRuntime()
    with pytest.raises(RuntimeError, match="unknown on shutdown"):
        offload.close()
    assert offload.runtime is None


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA Graph and pinned D2H"
)
def test_graph_replay_and_slot_backpressure():
    device = torch.device("cuda", torch.cuda.current_device())
    pool = SnapshotPool(device, 64, 1, 256, 64)
    value = torch.ones(64, device=device, dtype=torch.uint8)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = value + 1
    graph.replay()
    release = threading.Event()
    seen, errors = [], []

    def consume(host):
        if not release.wait(5):
            raise TimeoutError("consumer stalled")
        seen.append(host.clone())

    def snapshot(buf, stream):
        buf.copy_(output)
        output.record_stream(stream)

    try:
        assert pool.submit(64, snapshot, consume, errors.append)
        replay_stream = torch.cuda.Stream(device=device)
        with torch.cuda.stream(replay_stream):
            pool.fence_reuse()
            value.fill_(8)
            graph.replay()
        torch.cuda.current_stream().wait_stream(replay_stream)
        assert not pool.submit(64, snapshot, consume, errors.append)
    finally:
        release.set()
        pool.close()
    assert errors == ["snapshot slots exhausted"]
    assert torch.equal(seen[0], torch.full((64,), 2, dtype=torch.uint8))
    assert torch.all(output == 9)
