"""Unit and CUDA tests for Decode-only v2 DFLASH snapshot offload."""

from __future__ import annotations

import queue
import threading
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.kv.memory.views import MLAKVView
from sglang.srt.mem_cache.snapshot_transfer import (
    CompactKVPlan,
    HiddenD2HPool,
    KVSnapshotPool,
)
from sglang.srt.speculative import dflash_offload as offload
from sglang.srt.speculative.dflash_snapshot_rows import (
    OffloadAttempt,
    RequestHiddenBuffer,
    prompt_spans,
)
from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci

register_cpu_ci(est_time=4, suite="base-a-test-cpu")
register_cuda_ci(est_time=8, stage="base-b", runner_config="1-gpu-small")


def make_req(prompt=(11, 12), reply=(), rid="sample", committed=None):
    if committed is None:
        committed = len(prompt) + max(0, len(reply) - 1)
    return SimpleNamespace(
        rid=rid,
        origin_input_ids=list(prompt),
        output_ids=list(reply),
        finished_len=None,
        finished_reason=None,
        is_retracted=False,
        retraction_count=0,
        multimodal_inputs=None,
        input_embeds=None,
        kv=SimpleNamespace(kv_committed_len=committed, req_pool_idx=0),
    )


def _enable_env(monkeypatch, flag: bool = True):
    monkeypatch.setattr(envs.SGLANG_DFLASH_SNAPSHOT_OFFLOAD, "get", lambda: flag)


def test_explicit_engine_switch_gates_everything(monkeypatch):
    offload.reset()
    try:
        # Even when a plugin registers, switch OFF must keep behavior identical to base.
        _enable_env(monkeypatch, False)
        offload.register(
            offload.OffloadConfig(
                layers=(1,),
                select=lambda req: OffloadAttempt("c", "a"),
                consume=lambda a, r: None,
                gpu_bytes=64 << 20,
            )
        )
        assert not offload.is_enabled()
        assert not offload.wants_teacher()
        assert offload.gpu_budget_reservation_gb() == 0.0
        assert offload.configure_target_model(SimpleNamespace(is_draft_worker=False)) == 0
        assert offload.initialize(None) is None
        assert offload.runtime is None

        # Turn explicit switch ON -> decode reports budget reservation and teacher interest.
        _enable_env(monkeypatch, True)
        monkeypatch.setattr(
            "sglang.srt.runtime_context.get_disagg",
            lambda: SimpleNamespace(disaggregation_mode="decode"),
        )
        monkeypatch.setattr(
            "sglang.srt.runtime_context.get_parallel",
            lambda: SimpleNamespace(
                dp_size=1,
                dp_rank=0,
                attn_dp_size=4,
                attn_dp_rank=2,
                attn_dcp_size=8,
                attn_dcp_rank=0,
                tp_size=32,
                tp_rank=16,
            ),
        )
        assert offload.is_enabled()
        assert offload.wants_teacher() is True
        assert offload.gpu_budget_reservation_gb() == pytest.approx(64 / 1024)
    finally:
        offload.reset()


def test_tp32_attn_dp4_dcp8_hidden_owner_and_failsafe_configure(monkeypatch):
    offload.reset()
    try:
        _enable_env(monkeypatch, True)
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
        # TP32 / attn-DP4 / DCP8: rank 16 is DP group 2's hidden owner (attn_dcp_rank == 0).
        coords = {"tp_rank": 16, "attn_dp_rank": 2, "attn_dcp_rank": 0}
        monkeypatch.setattr(
            "sglang.srt.runtime_context.get_parallel",
            lambda: SimpleNamespace(
                pp_size=1,
                dp_size=1,
                dp_rank=0,
                attn_dp_size=4,
                attn_dp_rank=coords["attn_dp_rank"],
                attn_tp_size=8,
                attn_tp_rank=coords["attn_dcp_rank"],
                attn_dcp_size=8,
                attn_dcp_rank=coords["attn_dcp_rank"],
                tp_size=32,
                tp_rank=coords["tp_rank"],
            ),
        )
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
        # 16 * 4 * 2 = 128 bytes < 256 -> DP2 rank 0 gets shared buffer!
        assert offload.configure_target_model(runner) == 128
        assert lp.offload_verify_hidden is True
        assert lp.offload_verify_hidden_buffer.shape == (16, 4)
        assert runner.offload_shared_hidden_bytes == 128

        # DP2 rank 1 (attn_dcp_rank=1) does NOT enable hidden output.
        coords["tp_rank"] = 17
        coords["attn_dcp_rank"] = 1
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

        # Budget overflow must degrade cleanly (return 0, NEVER raise!).
        runner1.graph_shared_output = SimpleNamespace(max_rows=32)  # 256 >= 256
        assert offload.configure_target_model(runner1) == 0
        assert lp1.offload_verify_hidden is False

        # Missing LogitsProcessor must degrade cleanly (NEVER raise!).
        runner_no_lp = SimpleNamespace(
            is_draft_worker=False,
            device="cpu",
            model_config=SimpleNamespace(hidden_size=4, dtype=torch.bfloat16),
            model=SimpleNamespace(),
            graph_shared_output=None,
        )
        assert offload.configure_target_model(runner_no_lp) == 0

        # Real SGLang DP=1 ParallelContext has dp_rank=None -> must succeed cleanly!
        coords["dp_rank"] = None
        coords["attn_dp_size"] = 1
        coords["attn_dp_rank"] = 0
        coords["attn_dcp_rank"] = 0
        assert offload.wants_teacher() is True
        assert offload.configure_target_model(runner) == 128

        # Mid-serving recapture while runtime is active must close runtime & degrade, NEVER raise!
        closed = []
        offload.runtime = SimpleNamespace(close=lambda: closed.append(True))
        coords["attn_dcp_rank"] = 0
        assert offload.configure_target_model(runner) == 0
        assert closed == [True]
        assert offload.runtime is None
        assert lp.offload_verify_hidden is False
    finally:
        offload.reset()


def test_initialize_degrades_cleanly_on_prefill_or_bad_topology(monkeypatch):
    offload.reset()
    try:
        _enable_env(monkeypatch, True)
        import sys
        import types

        class FakeDFlashWorkerV2:
            block_size = 5

        fake_mod = types.ModuleType("sglang.srt.speculative.dflash_worker_v2")
        fake_mod.DFlashWorkerV2 = FakeDFlashWorkerV2
        monkeypatch.setitem(
            sys.modules, "sglang.srt.speculative.dflash_worker_v2", fake_mod
        )
        fake_cp = types.ModuleType("sglang.srt.layers.cp.base")
        fake_cp.is_cp_sharded_padded = lambda: False
        monkeypatch.setitem(sys.modules, "sglang.srt.layers.cp.base", fake_cp)
        fake_storage = types.ModuleType("sglang.srt.mem_cache.static_kv_storage")
        fake_storage.kv_storage_of = lambda *a, **kw: None
        monkeypatch.setitem(
            sys.modules, "sglang.srt.mem_cache.static_kv_storage", fake_storage
        )

        offload.register(
            offload.OffloadConfig((0,), lambda r: None, lambda a, p: None)
        )
        # Prefill role returns None silently (Decode-only capture).
        prefill_sched = SimpleNamespace(
            disaggregation_mode=SimpleNamespace(value="prefill"),
            draft_worker=FakeDFlashWorkerV2(),
        )
        assert offload.initialize(prefill_sched) is None

        # Unsupported attn_tp != dcp_size degrades cleanly to None (NEVER raises!).
        monkeypatch.setattr(
            "sglang.srt.runtime_context.get_parallel",
            lambda: SimpleNamespace(
                pp_size=1,
                dp_size=1,
                dp_rank=0,
                attn_dp_size=4,
                attn_dp_rank=0,
                attn_cp_size=1,
                attn_tp_size=8,
                attn_tp_rank=0,
                attn_dcp_size=4,
                attn_dcp_rank=0,
                tp_size=32,
                tp_rank=0,
            ),
        )
        decode_sched = SimpleNamespace(
            disaggregation_mode=SimpleNamespace(value="decode"),
            draft_worker=FakeDFlashWorkerV2(),
            kv_shard_widening=1,
        )
        assert offload.initialize(decode_sched) is None
        assert offload.runtime is None
    finally:
        offload.reset()


def test_request_hidden_buffer_overwrites_rejected_drafts_and_enforces_budget():
    buf = RequestHiddenBuffer(prompt_len=10, hidden_width=4, init_cap=4)
    # Step 1 verify: positions [9, 10, 11, 12] (pos 9 is prompt last token -> ignored).
    h1 = torch.arange(16, dtype=torch.bfloat16).reshape(4, 4)
    d1 = buf.write(
        torch.tensor([9, 10, 11, 12]),
        torch.tensor([999, 100, 101, 777]),  # 777 is a rejected draft token at pos 12
        h1,
        budget_remaining=1024,
    )
    assert d1 > 0

    # Step 2 verify: pos 12 was rejected, now verified with true token 102 and new hidden!
    h2 = (torch.arange(16, dtype=torch.bfloat16).reshape(4, 4) + 100)
    buf.write(
        torch.tensor([12, 13, 14, 15]),
        torch.tensor([102, 103, 888, 999]),
        h2,
        budget_remaining=1024,
    )

    # Final accepted output_ids = (100, 101, 102, 103, 104) -> 4 teacher rows at pos 10..13.
    pos, inp, nxt, hid = buf.extract((100, 101, 102, 103, 104))
    assert pos.tolist() == [10, 11, 12, 13]
    assert inp.tolist() == [100, 101, 102, 103]
    assert nxt.tolist() == [101, 102, 103, 104]
    assert torch.equal(hid[0], h1[1])  # pos 10 from step 1
    assert torch.equal(hid[1], h1[2])  # pos 11 from step 1
    assert torch.equal(hid[2], h2[0])  # pos 12 OVERWRITTEN by step 2!
    assert torch.equal(hid[3], h2[1])  # pos 13 from step 2

    # Mismatched predecessor token fails loudly.
    with pytest.raises(ValueError, match="mismatch"):
        buf.extract((100, 101, 999, 103, 104))

    # Exceeding budget raises MemoryError cleanly.
    with pytest.raises(MemoryError, match="budget"):
        buf.write(
            torch.tensor([100]),
            torch.tensor([555]),
            torch.zeros((1, 4), dtype=torch.bfloat16),
            budget_remaining=16,
        )


class HostKVPool:
    def __init__(self, device, slot_bytes, slots, gpu_bytes, pinned_bytes, *, consumer=None):
        self.slot_count = slots
        self.pending = []
        self.guard_calls = []
        self.submit_count = 0
        self.closed = False
        self.quarantine = []
        self.consumer = consumer

    def reading(self):
        return bool(self.pending)

    def guard_release(self, allocator):
        self.guard_calls.append(allocator)

    def submit(self, nbytes, fill, deliver, fail, *, wait_streams=()):
        if self.closed:
            fail("offload closed")
            return False
        self.submit_count += 1
        buf = torch.empty(nbytes, dtype=torch.uint8)
        fill(buf, None)
        self.pending.append(lambda: deliver(buf))
        return True

    def submit_callback(self, fn):
        if self.closed:
            return False
        self.pending.append(fn)
        return True

    def drain(self):
        while self.pending:
            todo, self.pending = self.pending, []
            for fn in todo:
                fn()

    def close(self):
        self.closed = True
        self.drain()


class HostHiddenPool:
    def __init__(self, device, max_rows, hidden_width, slots, pinned_bytes, *, consumer):
        self.slot_count = slots
        self.max_rows = max_rows
        self.meta_wait_calls = 0
        self.wait_calls = 0
        self.submit_slices = []
        self.pending = []
        self.closed = False
        self.consumer = consumer

    def wait_meta_ready(self, forward_stream=None):
        self.meta_wait_calls += 1

    def wait_buffer_ready(self, forward_stream=None):
        self.wait_calls += 1

    def submit(self, gpu_h, gpu_p, gpu_t, deliver, fail, *, forward_stream=None):
        if self.closed:
            fail("offload closed")
            return False
        self.submit_slices.append(int(gpu_h.shape[0]))
        h = gpu_h.clone()
        p = gpu_p.to(torch.int64).clone()
        t = gpu_t.to(torch.int64).clone()
        self.pending.append(lambda: deliver(p, t, h))
        return True

    def drain(self):
        while self.pending:
            todo, self.pending = self.pending, []
            for fn in todo:
                fn()

    def close(self):
        self.closed = True
        self.drain()


def make_runtime(
    monkeypatch,
    attempts,
    packets,
    *,
    on_complete=None,
    dcp_size=2,
    dcp_rank=0,
    dp_size=4,
    dp_rank=2,
    allocator=None,
    req_to_token=None,
):
    monkeypatch.setattr(offload, "KVSnapshotPool", HostKVPool)
    monkeypatch.setattr(offload, "HiddenD2HPool", HostHiddenPool)
    monkeypatch.setattr(torch.Tensor, "record_stream", lambda self, stream: None)
    view = MLAKVView()
    view.dcp_replicated = view.dsa_kv_cache_store_fp8 = False
    view.kv_lora_rank, view.qk_rope_head_dim = 2, 0
    buffers = {
        i: (torch.arange(60).reshape(15, 1, 4) + 100 * i).to(torch.bfloat16)[:, :, :2]
        for i in (1, 3)
    }

    def layer(i):
        return SimpleNamespace(owner=True, view=view, key=lambda: buffers[i])

    router = SimpleNamespace(layer=layer, read=layer)
    lp = SimpleNamespace(offload_verify_hidden=(dcp_rank == 0))
    if req_to_token is None:
        # 6 tokens total: prompt(2) + reply(4) -> IDs [5, 8, 2, 9, 4, 7]
        # Under dcp_size=2, rank=0 owns even IDs at indices 1(id=8), 2(id=2), 4(id=4) -> 3/6 rows!
        req_to_token = torch.tensor([[5, 8, 2, 9, 4, 7]])
    runner = SimpleNamespace(
        token_to_kv_pool=SimpleNamespace(layer_router=router),
        token_to_kv_pool_allocator=allocator,
        forward_stream=None,
        runner_ids=SimpleNamespace(translate_dcp_read_ids=lambda ids: ids // 2 + 1),
        req_to_token_pool=SimpleNamespace(req_to_token=req_to_token),
        model_config=SimpleNamespace(hidden_size=4, dtype=torch.bfloat16),
        model=SimpleNamespace(logits_processor=lp),
        graph_shared_output=SimpleNamespace(max_rows=32),
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
        kv_chunk_rows=4,
    )
    rt = offload.OffloadRuntime(
        runner, 4, 2, dp_size, dp_rank, dcp_size, dcp_rank, cfg
    )
    return rt, buffers


def test_decode_only_sparse_noncontiguous_hidden_overwrite_and_compact_kv(monkeypatch):
    a1 = OffloadAttempt("cap1", "att1")
    a4 = OffloadAttempt("cap4", "att4")
    packets = []
    completed = []
    fake_alloc = object()
    rt, buffers = make_runtime(
        monkeypatch,
        lambda req: {"r1": a1, "r4": a4}.get(req.rid),
        packets,
        on_complete=completed.append,
        dcp_size=2,
        dcp_rank=0,
        dp_size=4,
        dp_rank=2,
        allocator=fake_alloc,
    )
    # Sparse non-contiguous batch of 6 requests: only indices 1 and 4 are selected!
    reqs = [make_req(prompt=(11, 12), reply=(20,), rid=f"r{i}") for i in range(6)]
    width = 4
    positions = torch.tensor([1, 2, 3, 4] * 6, dtype=torch.int32)
    inputs = torch.zeros((6 * width,), dtype=torch.int32)
    # r1 (index 1): pos [1,2,3,4], tokens [12, 20, 42, 999(rejected)]
    inputs[4:8] = torch.tensor([12, 20, 42, 999], dtype=torch.int32)
    # r4 (index 4): pos [1,2,3,4], tokens [12, 20, 55, 66]
    inputs[16:20] = torch.tensor([12, 20, 55, 66], dtype=torch.int32)
    hidden = torch.arange(6 * width * 4, dtype=torch.bfloat16).reshape(24, 4)

    rt.before_step()
    assert rt.hidden_pool.meta_wait_calls == 1
    rt.before_verify()
    assert rt.hidden_pool.wait_calls == 1
    rt.snapshot_verify_hidden(reqs, positions, inputs, hidden, width)
    # Slices pure contiguous view [1*4 : 5*4] = 16 rows (ZERO H2D index_select!).
    assert rt.hidden_pool.submit_slices == [16]

    # Step 2 for r1: overwrites pos 4 (rejected 999 -> true token 43).
    pos2 = torch.tensor([4, 5, 6, 7] * 6, dtype=torch.int32)
    inp2 = torch.zeros((6 * width,), dtype=torch.int32)
    inp2[4:8] = torch.tensor([43, 44, 777, 888], dtype=torch.int32)
    hid2 = hidden + 1000
    rt.snapshot_verify_hidden(reqs, pos2, inp2, hid2, width)
    rt.hidden_pool.drain()

    # Finish r1: prompt=(11,12), reply=(20, 42, 43, 44) -> kv_committed_len=5 (pos 0..4), 3 teacher rows (pos 2,3,4).
    r1 = reqs[1]
    r1.output_ids = [20, 42, 43, 44]
    r1.kv.kv_committed_len = 5
    r1.finished_reason = SimpleNamespace(to_json=lambda: {"type": "stop"})
    expected_kv_pos1 = torch.stack([buffers[i][5, 0] for i in (1, 3)]).view(torch.uint8)
    expected_kv_pos2 = torch.stack([buffers[i][2, 0] for i in (1, 3)]).view(torch.uint8)
    expected_kv_pos4 = torch.stack([buffers[i][3, 0] for i in (1, 3)]).view(torch.uint8)

    rt.before_release(r1)
    # Zero source KV buffers immediately after before_release to prove snapshot preserved bytes.
    for b in buffers.values():
        b.zero_()
    assert rt.kv_pool.guard_calls == [fake_alloc]
    rt.kv_pool.drain()

    # Check r1 emitted compact owned-only KV packets + 1 trimmed final hidden packet!
    r1_pkts = [p for a, p in packets if a is a1]
    kv_pkts = [p for p in r1_pkts if p.final_hidden is None]
    hid_pkts = [p for p in r1_pkts if p.final_hidden is not None]

    # Owned positions in [0, 5) for req_to_token=[5,8,2,9,4] under dcp=2, rank=0: ONLY [1, 2, 4]!
    all_kv_pos = torch.cat([p.positions for p in kv_pkts]).tolist()
    all_kv_tok = torch.cat([p.input_ids for p in kv_pkts]).tolist()
    all_kv_raw = torch.cat([p.raw_kv for p in kv_pkts], dim=0)
    assert all_kv_pos == [1, 2, 4]
    assert all_kv_tok == [12, 20, 43]
    assert all(p.owned.all().item() for p in kv_pkts)
    assert all(p.dp_rank == 2 for p in kv_pkts)
    assert torch.equal(all_kv_raw[0], expected_kv_pos1)
    assert torch.equal(all_kv_raw[1], expected_kv_pos2)
    assert torch.equal(all_kv_raw[2], expected_kv_pos4)

    assert len(hid_pkts) == 1
    hp = hid_pkts[0]
    assert hp.dp_rank == 2 and hp.owner == 0
    assert hp.positions.tolist() == [2, 3, 4]
    assert hp.input_ids.tolist() == [20, 42, 43]
    assert hp.next_token_ids.tolist() == [42, 43, 44]
    # pos 2 & 3 from step 1 (rows 5 & 6 of hidden), pos 4 OVERWRITTEN by step 2 (row 4 of hid2)!
    assert torch.equal(hp.final_hidden[0], hidden[5])
    assert torch.equal(hp.final_hidden[1], hidden[6])
    assert torch.equal(hp.final_hidden[2], hid2[4])
    assert completed == [a1]
    assert a1.error is None
    assert rt._host_hidden_bytes_used >= 0  # r1 buffer freed


def test_non_hidden_owner_rank_skips_verify_hidden_entirely(monkeypatch):
    a = OffloadAttempt("cap", "att")
    packets = []
    rt, _ = make_runtime(monkeypatch, a, packets, dcp_size=8, dcp_rank=3)
    assert rt.hidden_width == 0
    assert rt.hidden_pool is None
    # Verify hooks are pure no-ops on non-hidden-owner ranks.
    rt.before_verify()
    rt.snapshot_verify_hidden([make_req()], None, None, None, 5)
    assert not packets


def test_kv_guard_release_stages_or_syncs():
    pool = object.__new__(KVSnapshotPool)
    pool.device = torch.device("cpu")
    pool.lock = threading.RLock()
    pool.gathering = False
    pool.kv_stream = SimpleNamespace(cuda_stream=2002)

    # Case 1: in-flight gather on kv_stream + staged allocator -> attach guard with 0 sync.
    synced = []
    ev1 = SimpleNamespace(query=lambda: False, synchronize=lambda: synced.append("ev1"))
    pool.reads = [(ev1, SimpleNamespace(cuda_stream=2002))]
    guarded = []
    alloc_staged = SimpleNamespace(
        supports_reuse_guard=True,
        need_sort=True,
        set_reuse_guard_event=guarded.append,
    )
    pool.guard_release(alloc_staged)
    assert guarded == [ev1]
    assert synced == []

    # Case 2: gather already complete -> does NOT even arm reuse_guard!
    ev_done = SimpleNamespace(query=lambda: True, synchronize=lambda: synced.append("done"))
    pool.reads = [(ev_done, SimpleNamespace(cuda_stream=2002))]
    guarded.clear()
    pool.guard_release(alloc_staged)
    assert guarded == []
    assert synced == []

    # Case 3: non-staged allocator -> synchronizes in-flight event.
    done = {"flag": False}
    ev2 = SimpleNamespace(
        query=lambda: done["flag"],
        synchronize=lambda: (synced.append("ev2"), done.__setitem__("flag", True)),
    )
    pool.reads = [(ev2, pool.kv_stream)]
    alloc_plain = SimpleNamespace(supports_reuse_guard=False, need_sort=False)
    pool.guard_release(alloc_plain)
    assert synced == ["ev2"]
    assert pool.reads == []


def test_prompt_spans_and_verified_pages():
    assert list(prompt_spans(11, {0, 3}, 2, 3)) == [(2, 5), (5, 6), (8, 11)]
    with pytest.raises(ValueError, match="complete"):
        list(prompt_spans(11, {5}, 2, 3))


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA + Triton compact KV gather"
)
def test_cuda_triton_compact_kv_gather_and_hidden_d2h():
    device = torch.device("cuda", torch.cuda.current_device())
    n_layers = 5
    row_bytes = 576
    n_tokens = 256
    dcp_size = 8
    rank = 3
    sources = tuple(
        torch.randint(
            1, 250, (512, 1, row_bytes), dtype=torch.uint8, device=device
        )
        for _ in range(n_layers)
    )
    ids = torch.arange(1, n_tokens + 1, dtype=torch.int64, device=device)
    owned = (ids > 0) & (ids.remainder(dcp_size) == rank)
    physical = torch.where(owned, ids // dcp_size + 1, 0).to(torch.int64)
    ptrs = torch.tensor([s.data_ptr() for s in sources], dtype=torch.int64, device=device)
    strides = torch.tensor(
        [s.stride(0) * s.element_size() for s in sources],
        dtype=torch.int64,
        device=device,
    )

    plan = CompactKVPlan.create(max_owned=64, n_layers=n_layers, row_bytes=row_bytes)
    pool = KVSnapshotPool(device, plan.nbytes, 4, 16 << 20, 16 << 20)
    seen = []
    errors = []
    try:
        def fill(storage, stream):
            plan.gather(
                storage,
                sources=sources,
                owned=owned,
                physical=physical,
                layer_ptrs_gpu=ptrs,
                layer_strides_gpu=strides,
            )

        assert pool.submit(plan.nbytes, fill, lambda h: seen.append(h.clone()), errors.append)
    finally:
        pool.close()

    assert not errors
    assert len(seen) == 1
    fields = plan.views(seen[0])
    count = int(fields["meta"][0].item())
    overflow = int(fields["meta"][1].item())
    assert overflow == 0
    assert count == int(owned.sum().item()) == 32
    order = torch.argsort(fields["orig_idx"][:count])
    rel = fields["orig_idx"][:count][order]
    kv = fields["kv"][:count][order]
    expected_idx = torch.nonzero(owned.cpu(), as_tuple=False).reshape(-1)
    assert torch.equal(rel, expected_idx)
    for k, t_idx in enumerate(expected_idx.tolist()):
        phys_id = int(physical[t_idx].item())
        for l_idx in range(n_layers):
            assert torch.equal(kv[k, l_idx], sources[l_idx][phys_id, 0].cpu())

    # Also verify HiddenD2HPool with int32 positions/tokens (same-dtype async DMA, zero host sync).
    from concurrent.futures import ThreadPoolExecutor

    cons = ThreadPoolExecutor(max_workers=1)
    cons.submit(torch.cuda.set_device, device).result()
    h_pool = HiddenD2HPool(device, 32, 16, 2, 4 << 20, consumer=cons)
    got_h = []
    try:
        g_h = torch.randn((10, 16), dtype=torch.bfloat16, device=device)
        g_p = torch.arange(10, 20, dtype=torch.int32, device=device)
        g_t = torch.arange(100, 110, dtype=torch.int32, device=device)
        assert h_pool.submit(
            g_h,
            g_p,
            g_t,
            lambda p, t, h: got_h.append((p.clone(), t.clone(), h.clone())),
            errors.append,
        )
        h_pool.wait_meta_ready()
        h_pool.wait_buffer_ready()
        h_pool.close()
        cons.shutdown(wait=True)
    finally:
        pass
    assert not errors
    assert len(got_h) == 1
    assert got_h[0][0].tolist() == list(range(10, 20))
    assert got_h[0][1].tolist() == list(range(100, 110))
    assert torch.equal(got_h[0][2], g_h.cpu())

    # Verify full OffloadRuntime init with unindexed device="cuda" (real ModelRunner.device!).
    view = MLAKVView()
    view.dcp_replicated = False
    view.dsa_kv_cache_store_fp8 = False
    view.kv_lora_rank = 512
    view.qk_rope_head_dim = 64
    kv_bufs = {
        i: torch.randint(1, 200, (128, 1, 576), dtype=torch.uint8, device=device).view(torch.float8_e4m3fn)
        for i in (3, 23)
    }
    router = SimpleNamespace(
        layer=lambda i: SimpleNamespace(owner=True, view=view, key=lambda i=i: kv_bufs[i]),
        read=lambda i: SimpleNamespace(owner=True, view=view, key=lambda i=i: kv_bufs[i]),
    )
    lp = SimpleNamespace(offload_verify_hidden=True, offload_verify_hidden_buffer=None)
    runner = SimpleNamespace(
        token_to_kv_pool=SimpleNamespace(layer_router=router),
        token_to_kv_pool_allocator=None,
        forward_stream=torch.cuda.current_stream(device),
        runner_ids=SimpleNamespace(translate_dcp_read_ids=lambda x: x // 8 + 1),
        req_to_token_pool=SimpleNamespace(
            req_to_token=torch.arange(1, 65, dtype=torch.int64, device=device).unsqueeze(0)
        ),
        model_config=SimpleNamespace(hidden_size=7168, dtype=torch.bfloat16),
        model=SimpleNamespace(logits_processor=lp),
        graph_shared_output=SimpleNamespace(max_rows=576),
        offload_shared_hidden_bytes=576 * 7168 * 2,
        device="cuda",  # unindexed string, matching real SGLang ModelRunner.device!
    )
    cfg = offload.OffloadConfig(
        (3, 23),
        lambda r: None,
        lambda a, p: None,
        gpu_bytes=64 << 20,
        pinned_bytes=192 << 20,
    )
    rt_cuda = offload.OffloadRuntime(
        runner, 6, 64, 1, 0, 8, 0, cfg, max_running_requests=128
    )
    try:
        assert rt_cuda.kv_pool.slot_count >= 1
        assert rt_cuda.hidden_pool is not None and rt_cuda.hidden_pool.slot_count >= 2
    finally:
        rt_cuda.close()
