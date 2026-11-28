"""CPU contracts for the SM90 GEMM norm boundary fusion (no CUDA smoke test)."""

import json
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from sglang.srt.environ import envs
from sglang.srt.layers.layer_boundary.fusions import sm90_gemm_norm as fusion
from sglang.srt.layers.layer_boundary.output import UnreducedOutput
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

_HIDDEN = 8


@pytest.fixture(autouse=True)
def _not_capturing(monkeypatch):
    # The CPU build of torch raises from this query instead of returning False.
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)


def _config(architecture, *, num_layers=4):
    return SimpleNamespace(architectures=[architecture], num_hidden_layers=num_layers)


def _batch(m, mode=ForwardMode.EXTEND):
    return SimpleNamespace(
        forward_mode=mode, input_ids=torch.zeros(m, dtype=torch.long)
    )


class _FakeCol:
    def __init__(self, *, max_m, n):
        self.data = torch.full((max_m, _HIDDEN), 7.0)
        self.n = n
        self.calls = []

    def input_buffer(self, m):
        return self.data[:m]

    def forward(self, *, partial, residual, norm_weight, weight):
        self.calls.append((partial, residual))
        return torch.ones(partial.shape[0], self.n)


def _all_gather(full, shard):
    rows = shard.shape[0]
    for rank in range(full.shape[0] // rows):
        full[rank * rows : (rank + 1) * rows] = shard + rank


def _service(*, world_size=2, rank=0, max_m=64, sparse=False, row_config=(128, 128, 1)):
    service = object.__new__(fusion._Service)
    service.group = SimpleNamespace(
        rank_in_group=rank, all_gather_into_tensor=_all_gather
    )
    service.world_size = world_size
    service.max_m = max_m
    service.row = SimpleNamespace(config_for=lambda m: row_config, forward=Mock())
    service.col = _FakeCol(max_m=max_m, n=4)
    service.sparse = sparse
    service.pending_partial = None
    service.pending_row = None
    service.pending_col = None
    return service


def _linear():
    return SimpleNamespace(weight=torch.zeros(4, _HIDDEN))


def _norm():
    return SimpleNamespace(weight=torch.ones(_HIDDEN))


# --- construction gate ---------------------------------------------------------


def test_provider_is_off_without_the_env_var():
    with envs.SGLANG_ENABLE_SM90_GEMM_NORM_FUSION.override(False):
        assert (
            fusion.make_sm90_gemm_norm_fusion(_config("Qwen3ForCausalLM"), layer_id=0)
            is None
        )


@pytest.mark.parametrize(
    "architecture",
    ["MistralForCausalLM", "Qwen2ForCausalLM", "Qwen3ForSequenceClassification"],
)
def test_provider_is_off_for_other_architectures(architecture):
    with envs.SGLANG_ENABLE_SM90_GEMM_NORM_FUSION.override(True):
        assert (
            fusion.make_sm90_gemm_norm_fusion(_config(architecture), layer_id=0) is None
        )


def test_provider_marks_first_and_terminal_layers():
    with envs.SGLANG_ENABLE_SM90_GEMM_NORM_FUSION.override(True):
        first = fusion.make_sm90_gemm_norm_fusion(
            _config("Qwen3ForCausalLM"), layer_id=0
        )
        last = fusion.make_sm90_gemm_norm_fusion(
            _config("Qwen3MoeForCausalLM"), layer_id=3
        )
    assert (first.sparse, first.first, first.terminal) == (False, True, False)
    assert (last.sparse, last.first, last.terminal) == (True, False, True)


def test_install_is_a_no_op_for_other_architectures():
    model = SimpleNamespace(config=_config("MistralForCausalLM"), model=Mock())
    with envs.SGLANG_ENABLE_SM90_GEMM_NORM_FUSION.override(True):
        fusion.install_sm90_gemm_norm_fusion(model)
    assert model.model.method_calls == []


# --- inert provider --------------------------------------------------------------


def test_uninstalled_provider_declines_every_boundary():
    provider = fusion.Sm90GemmNormFusion(sparse=False, first=True, terminal=False)
    plan = SimpleNamespace(norm=_norm())
    (ffn,) = provider.ffn_input(plan)
    (attention,) = provider.attention_input(plan)
    hidden = torch.zeros(4, _HIDDEN)
    assert ffn.run(hidden, hidden, _batch(4)) is None
    owed = UnreducedOutput(partial=hidden, group=Mock())
    assert attention(owed, hidden, _batch(4), None) is None
    assert not provider.can_defer_finalize(plan, _batch(4))
    assert not provider.can_defer_all_reduce(plan, _batch(4))


# --- per-forward decision ---------------------------------------------------------


def test_active_pads_extend_batches_to_the_world_size():
    service = _service(world_size=4, max_m=64)
    assert service.active(_batch(10)) == 12
    assert service.active(_batch(12)) == 12


@pytest.mark.parametrize(
    "mode", [ForwardMode.DECODE, ForwardMode.MIXED, ForwardMode.TARGET_VERIFY]
)
def test_active_is_prefill_only(mode):
    assert _service().active(_batch(8, mode)) is None


def test_active_declines_empty_oversized_and_untiled_batches():
    assert _service().active(_batch(0)) is None
    assert _service(max_m=16).active(_batch(17)) is None
    assert _service(row_config=None).active(_batch(8)) is None


def test_active_never_fuses_inside_cuda_graph_capture(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    assert _service().active(_batch(8)) is None


# --- residual shard -----------------------------------------------------------------


def test_shard_zero_pads_the_last_rank():
    residual = torch.arange(5 * _HIDDEN, dtype=torch.float32).view(5, _HIDDEN)
    first = _service(world_size=2, rank=0).shard(residual, 6)
    last = _service(world_size=2, rank=1).shard(residual, 6)
    torch.testing.assert_close(first, residual[:3])
    torch.testing.assert_close(last[:2], residual[3:5])
    assert torch.count_nonzero(last[2]) == 0


def test_gather_trims_the_padding():
    shard = torch.zeros(3, _HIDDEN)
    full = _service(world_size=2).gather(shard, 5)
    assert full.shape == (5, _HIDDEN)
    torch.testing.assert_close(full[3:5], torch.ones(2, _HIDDEN))


# --- handoffs -------------------------------------------------------------------------


def test_unconsumed_or_mismatched_row_handoffs_raise():
    service = _service()
    row = fusion._RowOutput(
        normed=torch.zeros(2, _HIDDEN), residual=torch.zeros(1, _HIDDEN)
    )
    service.put_row(row)
    with pytest.raises(RuntimeError):
        service.put_row(row)
    with pytest.raises(RuntimeError):
        service.take_row(torch.zeros(2, _HIDDEN))


def test_unconsumed_or_mismatched_partial_handoffs_raise():
    service = _service()
    value = torch.zeros(2, _HIDDEN)
    service.put_partial(value, value)
    with pytest.raises(RuntimeError):
        service.put_partial(value, value)
    with pytest.raises(RuntimeError):
        service.take_partial(torch.zeros(2, _HIDDEN))


def test_column_wrapper_reuses_the_fused_gemm_once_and_only_for_its_input():
    service = _service()
    native = Mock(return_value=("native", None))
    linear = SimpleNamespace(forward=native)
    fusion._install_col_forward(linear, service)
    col = fusion._ColOutput(normed=torch.zeros(2, _HIDDEN), output=torch.ones(2, 4))
    service.put_col(col)
    assert linear.forward(torch.zeros(2, _HIDDEN)) == ("native", None)
    assert linear.forward(col.normed) == (col.output, None)
    assert linear.forward(col.normed) == ("native", None)


# --- boundary candidates -------------------------------------------------------------


def _installed(*, sparse, first, terminal, service):
    provider = fusion.Sm90GemmNormFusion(sparse=sparse, first=first, terminal=terminal)
    provider.install(service, col_linear=_linear())
    return provider


def test_dense_ffn_input_shards_on_layer0_and_gathers_on_the_last_layer():
    service = _service(world_size=2)
    hidden = service.col_input(m=5, m_pad=6)[:5]
    service.put_partial(hidden, service.col_input(m=5, m_pad=6))
    provider = _installed(sparse=False, first=True, terminal=True, service=service)
    (ffn,) = provider.ffn_input(SimpleNamespace(norm=_norm()))
    full_residual = torch.zeros(5, _HIDDEN)

    normed, residual = ffn.run(hidden, full_residual, _batch(5))

    _, kernel_residual = service.col.calls[0]
    assert kernel_residual.shape == (3, _HIDDEN)
    assert residual.shape == (5, _HIDDEN)
    assert normed.shape == (5, _HIDDEN)
    assert service.pending_col.normed is normed
    assert torch.count_nonzero(service.col.data[5]) == 0


def test_dense_attention_input_requires_the_fused_down_proj():
    service = _service()
    provider = _installed(sparse=False, first=False, terminal=False, service=service)
    (attention,) = provider.attention_input(SimpleNamespace(norm=_norm()))
    owed = UnreducedOutput(partial=torch.zeros(4, _HIDDEN), group=Mock())
    with pytest.raises(RuntimeError):
        attention(owed, torch.zeros(2, _HIDDEN), _batch(4), None)
    with pytest.raises(RuntimeError):
        attention(owed, torch.zeros(2, _HIDDEN), _batch(4), torch.zeros(4, _HIDDEN))


def test_moe_attention_input_runs_the_qkv_kernel_on_the_moe_partial():
    service = _service(sparse=True)
    provider = _installed(sparse=True, first=False, terminal=False, service=service)
    (attention,) = provider.attention_input(SimpleNamespace(norm=_norm()))
    partial = torch.full((3, _HIDDEN), 2.0)
    residual = torch.zeros(2, _HIDDEN)

    normed, out_residual = attention(
        UnreducedOutput(partial=partial, group=Mock()), residual, _batch(3), None
    )

    torch.testing.assert_close(service.col.data[:3], partial)
    assert torch.count_nonzero(service.col.data[3]) == 0
    assert out_residual is residual
    assert service.pending_col.normed is normed


# --- row wrapper ----------------------------------------------------------------------


def test_row_wrapper_passes_through_when_inactive():
    service = _service()
    native = Mock(return_value=("native", None))
    linear = SimpleNamespace(forward=native, weight=torch.zeros(_HIDDEN, 4))
    fusion._install_row_forward(
        linear, service, norm=_norm(), shards_residual=False, deferred_sum=True
    )
    batch = _batch(4, ForwardMode.DECODE)
    assert linear.forward(torch.zeros(4, 4), forward_batch=batch) == ("native", None)
    native.assert_called_once()


def test_row_wrapper_refuses_a_sum_the_boundary_does_not_defer(monkeypatch):
    from sglang.srt.layers import moe

    monkeypatch.setattr(moe, "should_skip_mlp_all_reduce", lambda: False)
    service = _service()
    linear = SimpleNamespace(forward=Mock(), weight=torch.zeros(_HIDDEN, 4))
    fusion._install_row_forward(
        linear, service, norm=_norm(), shards_residual=False, deferred_sum=True
    )
    with pytest.raises(RuntimeError):
        linear.forward(torch.zeros(4, 4), forward_batch=_batch(4))


# --- chunk table loading --------------------------------------------------------------


def _stub_mega_ops(monkeypatch):
    def build_common_order(m, tile_m, world_size):
        per_rank = (m // world_size) // tile_m
        return tuple(
            rank * per_rank + round_idx
            for round_idx in range(per_rank)
            for rank in range(world_size)
        )

    pkg = ModuleType("mega_ops")
    pkg.__file__ = "/nonexistent/mega_ops/__init__.py"
    tile_order = ModuleType("mega_ops.tile_order")
    tile_order.build_physical_tile_m_order = build_common_order
    monkeypatch.setitem(sys.modules, "mega_ops", pkg)
    monkeypatch.setitem(sys.modules, "mega_ops.tile_order", tile_order)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: "NVIDIA H200")


def _chunk_service(*, world_size=4, rank=1, max_m=64, row_configs=None):
    service = _service(world_size=world_size, rank=rank, max_m=max_m)
    service.group.device = torch.device("cpu")
    service.row = SimpleNamespace(
        config_for=lambda m: (row_configs or {}).get(m, (128, 256, 2)),
        set_chunk_table=Mock(),
    )
    service.col.set_chunk_table = Mock()
    service.col._physical_tile_m_order_for = Mock(return_value=torch.tensor([99]))
    return service


def test_load_chunk_tables_filters_by_max_m_alignment_and_row_dispatch(
    monkeypatch, tmp_path
):
    _stub_mega_ops(monkeypatch)
    row_dir = tmp_path / "gemm_rs_norm_ag"
    row_dir.mkdir()
    spec = {
        "match": {"tile_shape": [128, 256], "cluster_m": 2},
        "configs": {
            "16": {"chunk_table": {"1": [0, 1, 0, 4, 1, 1, 0, 4]}},
            "18": {"chunk_table": {"1": [0, 1, 0, 4]}},
            "32": {"chunk_table": {"1": [0, 1, 0, 4]}},
            "128": {"chunk_table": {"1": [0, 1, 0, 4]}},
        },
    }
    (row_dir / "world=4,H=5120,K=3200,device_name=NVIDIA_H200.json").write_text(
        json.dumps(spec)
    )
    service = _chunk_service(
        world_size=4, rank=1, max_m=64, row_configs={16: (128, 256, 2), 32: (128, 128, 2)}
    )

    with envs.SGLANG_SM90_GEMM_NORM_CONFIG_DIR.override(str(tmp_path)):
        service._load_chunk_tables(hidden_size=5120, row_k=3200, col_n=6400)

    service.row.set_chunk_table.assert_called_once()
    loaded_m, loaded_table = service.row.set_chunk_table.call_args.args
    assert loaded_m == 16
    assert loaded_table.dtype == torch.int32
    torch.testing.assert_close(
        loaded_table, torch.tensor([[0, 1, 0, 4], [1, 1, 0, 4]], dtype=torch.int32)
    )
    service.col.set_chunk_table.assert_not_called()


def test_load_chunk_tables_switches_col_to_common_order_only_for_loaded_ms(
    monkeypatch, tmp_path
):
    _stub_mega_ops(monkeypatch)
    col_dir = tmp_path / "rs_norm_ag_gemm"
    col_dir.mkdir()
    spec = {
        "match": {"physical_tile_m_order": "common"},
        "configs": {"512": {"chunk_table": {"0": [0, 4, 0, 1]}}},
    }
    (col_dir / "world=2,H=5120,N=6400,device_name=NVIDIA_H200.json").write_text(
        json.dumps(spec)
    )
    service = _chunk_service(world_size=2, rank=0, max_m=512)
    native_order = service.col._physical_tile_m_order_for

    with envs.SGLANG_SM90_GEMM_NORM_CONFIG_DIR.override(str(tmp_path)):
        service._load_chunk_tables(hidden_size=5120, row_k=3200, col_n=6400)

    order_loaded = service.col._physical_tile_m_order_for(512, 128)
    torch.testing.assert_close(
        order_loaded, torch.tensor([0, 2, 1, 3], dtype=torch.int32)
    )
    assert service.col._physical_tile_m_order_for(512, 128) is order_loaded
    native_order.assert_not_called()

    order_unloaded = service.col._physical_tile_m_order_for(256, 128)
    torch.testing.assert_close(order_unloaded, torch.tensor([99]))
    native_order.assert_called_once_with(256, 128)


def test_load_chunk_tables_missing_files_leave_kernels_unchanged(monkeypatch, tmp_path):
    _stub_mega_ops(monkeypatch)
    service = _chunk_service()
    native_order = service.col._physical_tile_m_order_for

    with envs.SGLANG_SM90_GEMM_NORM_CONFIG_DIR.override(str(tmp_path)):
        service._load_chunk_tables(hidden_size=5120, row_k=3200, col_n=6400)

    service.row.set_chunk_table.assert_not_called()
    service.col.set_chunk_table.assert_not_called()
    assert service.col._physical_tile_m_order_for is native_order
