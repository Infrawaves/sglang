"""CPU contracts for the opt-in K3 fused o_proj adapter (no CUDA smoke test)."""

import itertools
import sys
from contextlib import nullcontext
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

# Keep optional CUDA/JIT dependencies out of these CPU adapter tests.
jit_utils = ModuleType("sglang.kernels.jit.utils")
jit_utils.cache_once = lambda function: function
jit_utils.empty_sentinel = Mock()
jit_utils.load_jit = Mock()
jit_utils.make_cpp_args = Mock()
previous_jit_utils = sys.modules.get(jit_utils.__name__)
sys.modules[jit_utils.__name__] = jit_utils
try:
    from sglang.kernels.ops.communication import gemm_rs_prefix_add as ops
finally:
    # Restore only this dependency; keep imported Torch/SGLang modules cached.
    if previous_jit_utils is None:
        sys.modules.pop(jit_utils.__name__)
    else:
        sys.modules[jit_utils.__name__] = previous_jit_utils
from sglang.srt.layers.communication import k3_sp_gemm_fusion as fusion
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

SM100_OPS = (
    "sm100_device_uuid",
    "sm100_check_peer_access",
    "sm100_resident_cluster_capacity",
    "sm100_gemm_rs_prefix_add_pull",
    "sm100_gemm_rs_prefix_add_push",
)
PULL = ops.DispatchConfig("pull", ops.KernelConfig(256, 128, 2), "common")
PUSH = ops.DispatchConfig("push", ops.KernelConfig(128, 128, 2), "common")


class Tensor:
    """Tensor metadata without allocating the production-sized CUDA buffers."""

    addresses = itertools.count(0x100000000, 0x100000000)

    def __init__(self, shape, *, dtype=torch.bfloat16, device="cuda:0", ptr=None):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.device = device
        self.contiguous = True
        self.ptr = next(self.addresses) if ptr is None else ptr

    def is_contiguous(self):
        return self.contiguous

    def data_ptr(self):
        return self.ptr

    def __getitem__(self, rows):
        start, stop, step = rows.indices(self.shape[0])
        assert step == 1
        return Tensor(
            (stop - start,) + self.shape[1:],
            dtype=self.dtype,
            device=self.device,
            ptr=self.ptr + start * self.shape[1] * 2,
        )


@pytest.fixture
def group():
    return SimpleNamespace(
        rank_in_group=1,
        world_size=4,
        device=torch.device("cuda:0"),
        cpu_group=object(),
    )


@pytest.fixture
def dispatch_table():
    # The checked-in GB300 table stays the policy source in these tests.
    return ops.load_dispatch_table(4, 7168, "NVIDIA_GB300")


@pytest.fixture
def runtime(monkeypatch):
    stream = SimpleNamespace(synchronize=Mock())
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *_: stream)
    monkeypatch.setattr(torch.cuda, "device", lambda *_: nullcontext())
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *_: "NVIDIA GB300")
    monkeypatch.setattr(torch, "zeros", lambda shape, **kwargs: Tensor(shape))
    monkeypatch.setattr(torch, "empty", lambda shape, **kwargs: Tensor(shape))
    mega_ops = SimpleNamespace(**{name: Mock() for name in SM100_OPS})
    monkeypatch.setattr(torch, "ops", SimpleNamespace(mega_ops=mega_ops))
    monkeypatch.setattr(
        fusion,
        "get_exec",
        lambda: SimpleNamespace(
            overlap=SimpleNamespace(
                enable_two_batch_overlap=False, enable_single_batch_overlap=False
            ),
            graph=SimpleNamespace(enable_torch_compile=False),
        ),
    )
    monkeypatch.setattr(
        fusion, "get_disagg", lambda: SimpleNamespace(enable_pdmux=False)
    )
    monkeypatch.setattr(
        fusion, "get_lora", lambda: SimpleNamespace(enable_lora=False, lora_paths=None)
    )
    monkeypatch.setattr(fusion, "get_forward", lambda: SimpleNamespace(sp_active=False))
    return stream


@pytest.fixture
def o_proj(monkeypatch, group):
    # Import the precise supported types without importing GPU linear backends.
    linear_module = ModuleType("sglang.srt.layers.linear")
    unquant_module = ModuleType("sglang.srt.layers.quantization.unquant")

    class RowParallelLinear:
        pass

    class UnquantizedLinearMethod:
        pass

    linear_module.RowParallelLinear = RowParallelLinear
    unquant_module.UnquantizedLinearMethod = UnquantizedLinearMethod
    monkeypatch.setitem(sys.modules, linear_module.__name__, linear_module)
    monkeypatch.setitem(sys.modules, unquant_module.__name__, unquant_module)
    o_proj = RowParallelLinear()
    o_proj.quant_method = UnquantizedLinearMethod()
    o_proj.input_is_parallel = True
    o_proj.skip_bias_add = False
    o_proj.bias = None
    o_proj.scheme = None
    o_proj.use_decode_attn_tp = False
    o_proj.tp_size = group.world_size
    o_proj.output_size = 7168
    o_proj.weight = Tensor((7168, 2048))
    return o_proj


class FakeOperator:
    def __init__(self, *args):
        self.args = args
        self.prepare = Mock()
        self.calls = []

    def __call__(self, input, weight, prefix, output, dispatch):
        self.calls.append((input, weight, prefix, output, dispatch))
        return output


@pytest.fixture
def manager(group, dispatch_table, runtime, monkeypatch):
    monkeypatch.setattr(ops, "GemmRSPrefixAdd", FakeOperator)
    manager = fusion.K3SPGemmFusionManager(group, dispatch_table, 7168)
    manager.prepare(16384)
    return manager


def test_o_proj_call_is_nested_and_restored_after_exception():
    assert fusion._CURRENT_O_PROJ_FUSION_CALL.get() is None
    with fusion.use_o_proj_fusion_call(None, 8192) as outer:
        with pytest.raises(ValueError):
            with fusion.use_o_proj_fusion_call(None, 12288) as inner:
                assert fusion._CURRENT_O_PROJ_FUSION_CALL.get() is inner
                raise ValueError("attention failed")
        assert fusion._CURRENT_O_PROJ_FUSION_CALL.get() is outer
    assert fusion._CURRENT_O_PROJ_FUSION_CALL.get() is None


def test_manager_creation_needs_only_registered_ops(group, runtime):
    manager = fusion.create_gemm_fusion_manager(group, 7168)
    assert manager is not None
    assert manager.gemm_rs_operator is None


def test_missing_ops_or_table_keeps_native(group, runtime, caplog):
    del torch.ops.mega_ops.sm100_gemm_rs_prefix_add_push
    assert fusion.create_gemm_fusion_manager(group, 7168) is None
    assert "missing: sm100_gemm_rs_prefix_add_push" in caplog.text
    torch.ops.mega_ops.sm100_gemm_rs_prefix_add_push = Mock()
    group.world_size = 8
    assert fusion.create_gemm_fusion_manager(group, 7168) is None
    assert "has no table for TP8" in caplog.text


@pytest.mark.parametrize("regime", ["pdmux", "tbo", "sbo", "lora", "compile"])
def test_unsupported_execution_regimes_disable_fusion_at_initialization(
    group, runtime, monkeypatch, caplog, regime
):
    execution = SimpleNamespace(
        overlap=SimpleNamespace(
            enable_two_batch_overlap=regime == "tbo",
            enable_single_batch_overlap=regime == "sbo",
        ),
        graph=SimpleNamespace(enable_torch_compile=regime == "compile"),
    )
    monkeypatch.setattr(fusion, "get_exec", lambda: execution)
    monkeypatch.setattr(
        fusion, "get_disagg", lambda: SimpleNamespace(enable_pdmux=regime == "pdmux")
    )
    monkeypatch.setattr(
        fusion,
        "get_lora",
        lambda: SimpleNamespace(enable_lora=regime == "lora", lora_paths=None),
    )
    assert fusion.create_gemm_fusion_manager(group, 7168) is None
    assert "unsupported execution configuration" in caplog.text


def test_prepare_sizes_workspace_for_shapes_under_service_ceiling(
    group, dispatch_table, runtime, monkeypatch
):
    monkeypatch.setattr(ops, "GemmRSPrefixAdd", FakeOperator)
    manager = fusion.K3SPGemmFusionManager(group, dispatch_table, 7168)
    manager.prepare(13000)
    operator = manager.gemm_rs_operator
    assert operator.args[:5] == (group.cpu_group, 1, 4, 12288, 7168)
    assert operator.args[6] == {PULL.kernel}
    assert [call.args for call in operator.prepare.call_args_list] == [
        (8192, 2048, PULL),
        (12288, 2048, PULL),
    ]
    assert manager.zero_prefix_buffer.shape == (3072, 7168)
    runtime.synchronize.assert_called_once_with()


def test_prepare_below_fused_envelope_allocates_nothing(
    group, dispatch_table, runtime, monkeypatch
):
    constructor = Mock()
    monkeypatch.setattr(ops, "GemmRSPrefixAdd", constructor)
    manager = fusion.K3SPGemmFusionManager(group, dispatch_table, 7168)
    manager.prepare(4096)
    constructor.assert_not_called()
    assert manager.gemm_rs_operator is None


@pytest.mark.parametrize("prefix_rows", [None, 8192, 2048])
def test_fused_call_notifies_shard_and_prefix_once(manager, o_proj, prefix_rows):
    attention_output = Tensor((8192, 2048))
    prefix_sum = None if prefix_rows is None else Tensor((prefix_rows, 7168))
    with fusion.use_o_proj_fusion_call(prefix_sum, 8192) as fusion_call:
        output, bias = manager.try_fused_o_proj(o_proj, attention_output)
        assert bias is None
        assert output.shape == (2048, 7168)
        assert fusion_call.fused_output == fusion.FusedOProjOutput(output, 2048)
        assert manager.try_fused_o_proj(o_proj, attention_output) is None
    [(input, weight, prefix, call_output, dispatch)] = manager.gemm_rs_operator.calls
    assert input is attention_output
    assert weight is o_proj.weight
    if prefix_sum is None:
        assert prefix.shape == (2048, 7168)
        assert prefix.data_ptr() == manager.zero_prefix_buffer.data_ptr()
    else:
        assert prefix is prefix_sum
    assert call_output is output
    assert dispatch == PULL


def test_consecutive_layers_keep_separate_model_outputs(manager, o_proj):
    outputs = []
    for _ in range(2):
        with fusion.use_o_proj_fusion_call(None, 8192):
            outputs.append(manager.try_fused_o_proj(o_proj, Tensor((8192, 2048)))[0])
    assert outputs[0].data_ptr() != outputs[1].data_ptr()
    assert len(manager.gemm_rs_operator.calls) == 2


@pytest.mark.parametrize("m", [512, 1024, 2048, 4096, 10000])
def test_baseline_disabled_and_unmeasured_shapes_stay_native(manager, o_proj, m):
    with fusion.use_o_proj_fusion_call(None, m) as fusion_call:
        assert manager.try_fused_o_proj(o_proj, Tensor((m, 2048))) is None
        assert fusion_call.fused_output is None
    assert manager.gemm_rs_operator.calls == []


def test_no_scope_trimmed_padding_and_other_k_stay_native(manager, o_proj):
    assert manager.try_fused_o_proj(o_proj, Tensor((8192, 2048))) is None
    with fusion.use_o_proj_fusion_call(None, 12288):
        assert manager.try_fused_o_proj(o_proj, Tensor((8192, 2048))) is None
    with fusion.use_o_proj_fusion_call(None, 8192):
        assert manager.try_fused_o_proj(o_proj, Tensor((8192, 4096))) is None
    assert manager.gemm_rs_operator.calls == []


def test_supported_o_proj(manager, o_proj):
    assert manager.supports(o_proj)


@pytest.mark.parametrize(
    "name,value",
    [
        ("input_is_parallel", False),
        ("skip_bias_add", True),
        ("bias", object()),
        ("scheme", object()),
        ("tp_size", 8),
        ("output_size", 4096),
        ("quant_method", object()),
    ],
)
def test_extra_linear_semantics_are_unsupported(manager, o_proj, name, value):
    setattr(o_proj, name, value)
    assert not manager.supports(o_proj)


def test_custom_unquantized_method_subclass_is_unsupported(manager, o_proj):
    class ScaledMethod(type(o_proj.quant_method)):
        pass

    o_proj.quant_method = ScaledMethod()
    assert not manager.supports(o_proj)


@pytest.mark.parametrize(
    "kwargs",
    [{"output_tensor": None}, {"skip_all_reduce": True}, {"unexpected_scale": 1.0}],
)
def test_explicit_linear_overrides_stay_native(manager, o_proj, kwargs):
    with fusion.use_o_proj_fusion_call(None, 8192):
        assert manager.try_fused_o_proj(o_proj, Tensor((8192, 2048)), **kwargs) is None
    assert manager.gemm_rs_operator.calls == []


def test_forward_batch_kwarg_is_accepted(manager, o_proj):
    with fusion.use_o_proj_fusion_call(None, 8192):
        assert (
            manager.try_fused_o_proj(o_proj, Tensor((8192, 2048)), forward_batch=None)
            is not None
        )


@pytest.mark.parametrize(
    "problem", ["wrong_dtype", "not_contiguous", "prefix_dtype", "prefix_layout"]
)
def test_incompatible_dtype_or_layout_stays_native(manager, o_proj, problem):
    attention_output = Tensor((8192, 2048))
    prefix_sum = Tensor((2048, 7168))
    if problem == "wrong_dtype":
        attention_output.dtype = torch.float16
    elif problem == "not_contiguous":
        attention_output.contiguous = False
    elif problem == "prefix_dtype":
        prefix_sum.dtype = torch.float32
    else:
        prefix_sum.contiguous = False
    with fusion.use_o_proj_fusion_call(prefix_sum, 8192):
        assert manager.try_fused_o_proj(o_proj, attention_output) is None
    assert manager.gemm_rs_operator.calls == []


def test_megatron_sp_and_decode_attn_tp_stay_native(manager, o_proj, monkeypatch):
    with fusion.use_o_proj_fusion_call(None, 8192):
        monkeypatch.setattr(
            fusion, "get_forward", lambda: SimpleNamespace(sp_active=True)
        )
        assert manager.try_fused_o_proj(o_proj, Tensor((8192, 2048))) is None
        monkeypatch.setattr(
            fusion, "get_forward", lambda: SimpleNamespace(sp_active=False)
        )
        o_proj.use_decode_attn_tp = True
        assert manager.try_fused_o_proj(o_proj, Tensor((8192, 2048))) is None
    assert manager.gemm_rs_operator.calls == []


def test_launch_failure_does_not_claim_completion(manager, o_proj):
    manager.gemm_rs_operator = Mock(side_effect=RuntimeError("kernel launch failed"))
    with fusion.use_o_proj_fusion_call(None, 8192) as fusion_call:
        with pytest.raises(RuntimeError, match="kernel launch failed"):
            manager.try_fused_o_proj(o_proj, Tensor((8192, 2048)))
        assert fusion_call.fused_output is None


def test_one_context_prepares_and_runs_both_modes(group, o_proj, runtime, monkeypatch):
    monkeypatch.setattr(ops, "GemmRSPrefixAdd", FakeOperator)
    table = ops.DispatchTable(2048, {4096: PUSH, 8192: PULL})
    manager = fusion.K3SPGemmFusionManager(group, table, 7168)
    manager.prepare(8192)
    operator = manager.gemm_rs_operator
    assert operator.args[6] == {PUSH.kernel, PULL.kernel}
    assert [call.args[2] for call in operator.prepare.call_args_list] == [PUSH, PULL]
    for m in (4096, 8192, 4096):
        with fusion.use_o_proj_fusion_call(None, m):
            assert manager.try_fused_o_proj(o_proj, Tensor((m, 2048))) is not None
    assert [call[4] for call in operator.calls] == [PUSH, PULL, PUSH]
