import queue
import threading
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.mem_cache.kv.memory.views import MLAKVView
from sglang.srt.mem_cache.snapshot_transfer import SnapshotPool
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


def request(prompt=(11, 12), reply=()):
    return SimpleNamespace(
        rid="sample",
        origin_input_ids=prompt,
        output_ids=list(reply),
        finished_len=None,
        finished_reason=None,
        is_retracted=False,
        retraction_count=0,
        multimodal_inputs=None,
        kv=SimpleNamespace(kv_committed_len=len(prompt), req_pool_idx=0),
    )


def test_default_allocates_no_buffers():
    assert not offload.wants_teacher()
    assert offload.reserve_graph_bytes(None) == 0
    offload.initialize(None)
    assert offload.runtime is None


def test_graph_budget_counts_ragged_shapes_variants_and_warmup(monkeypatch):
    monkeypatch.setattr(offload, "wants_teacher", lambda: True)
    graph = SimpleNamespace(
        model_runner=SimpleNamespace(
            is_draft_worker=False, model_config=SimpleNamespace(hidden_size=4)
        ),
        capture_forward_mode=SimpleNamespace(is_target_verify=lambda: True),
        _capture_sizes=lambda: [1, (3, 8)],
        captured_req_width=4,
        attention_graph_variants=SimpleNamespace(capture_labels=(0, 1)),
        _attention_varies_at=lambda bs: bs == 3,
        record_nolora_graph=True,
        logits_graph_variants=(0, 1),
    )
    expected = 8 * (4 * 4 + 8 * 8 + 8)
    monkeypatch.setattr(offload, "config", SimpleNamespace(gpu_bytes=expected + 1))
    assert offload.reserve_graph_bytes(graph) == expected
    offload.config.gpu_bytes -= 1
    with pytest.raises(ValueError, match="budget"):
        offload.reserve_graph_bytes(graph)


def test_prompt_coverage():
    assert list(prompt_spans(11, {0, 3}, 2, 3)) == [(2, 5), (5, 6), (8, 11)]
    with pytest.raises(ValueError, match="complete"):
        list(prompt_spans(11, {5}, 2, 3))


class HostPool:
    """Defer delivery, preserving the snapshot before deliberate source reuse."""

    def __init__(self, *args):
        self.pending = []
        self.fences = 0

    def fence_reuse(self):
        self.fences += 1

    def submit(self, nbytes, fill, deliver, fail):
        buf = torch.empty(nbytes, dtype=torch.uint8)
        fill(buf, None)
        self.pending.append(lambda: deliver(buf))
        return True

    def drain(self):
        for deliver in self.pending:
            deliver()


def runtime(monkeypatch, role, attempt, packets):
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
    runner = SimpleNamespace(
        token_to_kv_pool=SimpleNamespace(layer_router=router),
        runner_ids=SimpleNamespace(translate_dcp_read_ids=lambda ids: ids // 2 + 1),
        req_to_token_pool=SimpleNamespace(req_to_token=torch.tensor([[5, 8, 2, 9, 4]])),
        model_config=SimpleNamespace(hidden_size=4, dtype=torch.bfloat16),
        decode_cuda_graph_runner=None,
        device="cpu",
    )
    cfg = offload.OffloadConfig(
        (1, 3),
        lambda req: attempt if req.rid == "sample" else None,
        lambda selected, packet: packets.append(packet),
        prompt_chunk_rows=2,
    )
    return offload.OffloadRuntime(runner, 4, 2, role, 2, 0, cfg), buffers


def test_prefill_uses_verified_pages_and_copies_tail_before_release(monkeypatch):
    packets = []
    attempt = OffloadAttempt("capture", "attempt", frozenset({0}))
    rt, buffers = runtime(monkeypatch, "prefill", attempt, packets)
    expected_kv = torch.stack([buffers[i][2, 0] for i in (1, 3)]).view(torch.uint8)
    req = request(prompt=(10, 11, 12, 13, 14))
    req.finished_reason = SimpleNamespace(to_json=lambda: {"type": "length"})
    rt.before_release(req)
    for value in buffers.values():
        value.zero_()
    assert rt.pool.fences == 2
    assert not packets
    rt.pool.drain()
    assert [p.positions.tolist() for p in packets] == [[2, 3], [4]]
    assert [p.input_ids.tolist() for p in packets] == [[12, 13], [14]]
    assert [p.physical_ids.tolist() for p in packets] == [[2], [3]]
    assert torch.equal(packets[0].raw_kv[0], expected_kv)
    assert all(p.final_hidden is None and p.next_token_ids is None for p in packets)


def test_verify_snapshots_source_then_commits_visible_predecessors(monkeypatch):
    attempt = OffloadAttempt("capture", "attempt")
    packets = []
    rt, buffers = runtime(monkeypatch, "decode", attempt, packets)
    expected_kv = torch.stack([buffers[i][5, 0] for i in (1, 3)]).view(torch.uint8)
    skipped, req = request(), request(reply=(20,))
    skipped.rid = "unsampled"
    positions = torch.tensor([0, 0, 0, 0, 2, 3, 4, 5])
    inputs = torch.tensor([0, 0, 0, 0, 20, 42, 77, 88])
    ids = torch.tensor([0, 0, 0, 0, 5, 8, 2, 9])
    hidden = torch.arange(32, dtype=torch.bfloat16).reshape(8, 4)
    (commit,) = rt.snapshot_verify([skipped, req], positions, inputs, ids, hidden, 4)
    for value in (positions, inputs, ids, hidden, *buffers.values()):
        value.zero_()
    commit.before()
    req.output_ids.extend([42, 99, 100])
    req.finished_len = 3
    req.finished_reason = SimpleNamespace(to_json=lambda: {"type": "stop"})
    commit.after()
    rt.pool.drain()
    (p,) = packets
    assert p.positions.tolist() == [2, 3]
    assert p.input_ids.tolist() == [20, 42]
    assert p.next_token_ids.tolist() == [42, 99]
    assert p.final_hidden.tolist() == [[16, 17, 18, 19], [20, 21, 22, 23]]
    assert p.owned.tolist() == [False, True]
    assert p.physical_ids.tolist() == [5]
    assert torch.equal(p.raw_kv[0], expected_kv)


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
