"""CPU control-flow checks for K3's optional fused o_proj + RS + prefix.

Execute the production definitions without importing the serving/GPU stack.
Tensor-shaped doubles make these routing tests independent of CUDA; numerical
kernel and full-model smoke tests belong to the separate GPU validation suite.
"""

import ast
import copy
import importlib.util
import sys
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

REPO_ROOT = Path(__file__).resolve().parents[4]
MODEL = REPO_ROOT / "python/sglang/srt/models/kimi_k3.py"
SP = REPO_ROOT / "python/sglang/srt/layers/communication/k3_sp_collective.py"


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


register_cpu_ci = _load_module(
    "k3_sp_gemm_test_ci_register",
    REPO_ROOT / "python/sglang/test/ci/ci_register.py",
).register_cpu_ci
register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _definitions(path, names, namespace, class_name=None):
    """Compile unmodified production function bodies with test dependencies."""
    tree = ast.parse(path.read_text(), filename=str(path))
    body = tree.body
    if class_name is not None:
        body = next(
            node.body
            for node in body
            if isinstance(node, ast.ClassDef) and node.name == class_name
        )
    selected = [
        copy.deepcopy(node)
        for node in body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names
    ]
    found = {node.name for node in selected}
    if found != set(names):
        raise AssertionError(f"Missing production definitions: {set(names) - found}")
    future = ast.parse("from __future__ import annotations").body
    module = ast.fix_missing_locations(
        ast.Module(body=future + selected, type_ignores=[])
    )
    exec(compile(module, str(path), "exec"), namespace)
    return SimpleNamespace(**{name: namespace[name] for name in names})


def _constructor(class_name, base, namespace):
    tree = ast.parse(MODEL.read_text(), filename=str(MODEL))
    source = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    node = copy.deepcopy(source)
    node.bases = [ast.Name(id="TestBase", ctx=ast.Load())]
    node.body = [
        method
        for method in node.body
        if isinstance(method, ast.FunctionDef) and method.name == "__init__"
    ]
    future = ast.parse("from __future__ import annotations").body
    module = ast.fix_missing_locations(
        ast.Module(body=future + [node], type_ignores=[])
    )
    namespace["TestBase"] = base
    exec(compile(module, str(MODEL), "exec"), namespace)
    return namespace[class_name]


class _Tensor:
    """Only shape and slicing are required by the production control flow."""

    def __init__(self, rows, width=8, name="tensor"):
        self.shape = (rows, width)
        self.ndim = 2
        self.name = name
        self.dtype = "bfloat16"
        self.device = "cuda"
        self.writes = []
        self.data = self

    def __getitem__(self, index):
        if not isinstance(index, slice):
            raise AssertionError(f"Unexpected tensor indexing: {index!r}")
        rows = len(range(*index.indices(self.shape[0])))
        return _Tensor(rows, self.shape[1], f"{self.name}[{index}]")

    def __setitem__(self, index, value):
        self.writes.append((index, value))

    def new_zeros(self, *shape):
        return _Tensor(*shape, name=f"zeros({self.name})")

    def unsqueeze(self, *args):
        return self

    def squeeze(self, *args):
        return self

    def unflatten(self, *args):
        return self

    def flatten(self, *args):
        return self


def _forward_batch(*, idle=False, extend=False, real_rows=None):
    return SimpleNamespace(
        forward_mode=SimpleNamespace(
            is_idle=lambda: idle,
            is_extend=lambda: extend,
        ),
        extend_seq_lens_cpu=None if real_rows is None else [real_rows],
    )


class TestKimiK3SPGemmAttentionBoundary(unittest.TestCase):
    def _run_attention(self, *, idle=False, real_rows=8, padded_output=None):
        sp = SimpleNamespace(
            o_proj_output_rows=Mock(side_effect=lambda rows: nullcontext()),
            finish_padded_o_proj_output=Mock(return_value=padded_output),
        )
        definitions = _definitions(
            MODEL,
            ["_run_self_attn"],
            {"k3_sp_collective": sp},
            "KimiK3DecoderLayer",
        )
        owner = SimpleNamespace(
            _trim_padded_attn=True,
            _run_self_attn_inner=Mock(
                return_value=_Tensor(real_rows, name="attention")
            ),
        )
        hidden = _Tensor(8, name="hidden")
        output = definitions._run_self_attn(
            owner,
            hidden,
            _Tensor(8, 1, "positions"),
            _forward_batch(idle=idle, extend=True, real_rows=real_rows),
            object(),
        )
        return output, hidden, owner, sp

    def test_idle_rank_does_not_enter_attention(self):
        output, hidden, owner, sp = self._run_attention(idle=True)
        self.assertIs(output, hidden)
        owner._run_self_attn_inner.assert_not_called()
        sp.o_proj_output_rows.assert_not_called()

    def test_trimmed_attention_reconstructs_full_padded_batch(self):
        output, _, owner, sp = self._run_attention(real_rows=6)
        self.assertEqual(owner._run_self_attn_inner.call_args.args[0].shape, (6, 8))
        self.assertEqual(output.shape, (8, 8))
        self.assertEqual(output.writes[0][0], slice(None, 6))
        sp.o_proj_output_rows.assert_called_once_with(8)

    def test_trimmed_attention_reuses_symmetric_padded_output(self):
        padded_output = _Tensor(8, name="symmetric")
        output, _, _, sp = self._run_attention(real_rows=6, padded_output=padded_output)
        self.assertIs(output, padded_output)
        sp.finish_padded_o_proj_output.assert_called_once()

    def test_unpadded_attention_preserves_output(self):
        output, _, owner, sp = self._run_attention()
        self.assertIs(output, owner._run_self_attn_inner.return_value)
        sp.o_proj_output_rows.assert_not_called()


def _sp_install(manager, **namespace):
    state = SimpleNamespace(gemm_fusion_manager=manager)
    namespace.setdefault("get_o_proj_output_rows", lambda rows: rows)
    namespace.setdefault("requires_symmetric_rs", lambda *args: False)
    return _definitions(
        SP, ["install_o_proj_wrapper"], {"_init_state": lambda: state, **namespace}
    ).install_o_proj_wrapper


class TestKimiK3SPGemmOProjWrapper(unittest.TestCase):
    def _wrapper(self, *, fused=False, symmetric=False, manager=True, supported=True):
        output = _Tensor(2, name="fused")
        fusion_manager = SimpleNamespace(
            supports=Mock(return_value=supported),
            try_fused_o_proj=Mock(return_value=(output, None) if fused else None),
        )
        native = Mock(return_value=(_Tensor(8, name="native"), None))
        layer = SimpleNamespace(forward=native, output_size=8)
        namespace = {
            "requires_symmetric_rs": Mock(return_value=symmetric),
            "get_o_proj_output_buffer": Mock(return_value=_Tensor(8, name="symmetric")),
            "register_o_proj_output": Mock(),
        }
        _sp_install(fusion_manager if manager else None, **namespace)(layer)
        return SimpleNamespace(
            layer=layer,
            native=native,
            manager=fusion_manager,
            namespace=namespace,
        )

    def test_fused_result_replaces_native_gemm(self):
        run = self._wrapper(fused=True)
        x = _Tensor(8)
        output = run.layer.forward(x)
        self.assertIs(output, run.manager.try_fused_o_proj.return_value)
        run.manager.supports.assert_called_once_with(run.layer)
        run.manager.try_fused_o_proj.assert_called_once_with(run.layer, x)
        run.native.assert_not_called()
        run.namespace["requires_symmetric_rs"].assert_not_called()

    def test_baseline_keeps_native_symmetric_output_contract(self):
        run = self._wrapper(symmetric=True)
        x = _Tensor(8)
        output = run.layer.forward(x)
        self.assertIs(output, run.native.return_value)
        run.native.assert_called_once()
        self.assertEqual(run.native.call_args.kwargs["output_tensor"].shape, (8, 8))
        run.namespace["register_o_proj_output"].assert_called_once_with(
            output[0], run.namespace["get_o_proj_output_buffer"].return_value
        )

    def test_disabled_or_unsupported_o_proj_does_not_attempt_fusion(self):
        for kwargs in (dict(manager=False), dict(supported=False)):
            with self.subTest(**kwargs):
                run = self._wrapper(**kwargs)
                x = _Tensor(8)
                run.layer.forward(x)
                run.manager.try_fused_o_proj.assert_not_called()
                run.native.assert_called_once_with(x)

    def test_positional_arguments_stay_native(self):
        run = self._wrapper(fused=True)
        x = _Tensor(8)
        run.layer.forward(x, False)
        run.manager.try_fused_o_proj.assert_not_called()
        run.native.assert_called_once_with(x, False)


class TestKimiK3SPGemmGateOrdering(unittest.TestCase):
    def _dependencies(self):
        events = []
        result = _Tensor(2, name="fused")

        def fused_forward(layer, x, *args, **kwargs):
            events.append(("fused", x))
            return result, None

        manager = SimpleNamespace(
            supports=lambda o_proj: True,
            try_fused_o_proj=Mock(side_effect=fused_forward),
        )
        install = _sp_install(manager)

        class Linear:
            def __init__(self, *args, **kwargs):
                self.output_size = 8
                self.weight = _Tensor(8)
                self.bias = None
                self.forward = Mock(return_value=(_Tensor(8, name="native"), None))

            def __call__(self, *args, **kwargs):
                return self.forward(*args, **kwargs)

        namespace = {
            "_uses_split_gguf_kv_b": lambda config: False,
            "_uses_modelopt_fp8_pb_wo": lambda *args: False,
            "_is_hip": False,
            "ColumnParallelLinear": Linear,
            "MergedColumnParallelLinear": Linear,
            "ReplicatedLinear": Linear,
            "RowParallelLinear": Linear,
            "get_parallel": lambda: SimpleNamespace(
                tp_size=4, attn_tp_size=4, attn_tp_rank=1
            ),
            "k3_sp_collective": SimpleNamespace(
                enabled=lambda: True, install_o_proj_wrapper=install
            ),
            "divide": lambda a, b: a // b,
            "set_weight_attrs": lambda *args: None,
            "sharded_weight_loader": lambda *args: object(),
            "FusedRMSNormGated": Mock(),
            "RadixLinearAttention": Mock(),
            "torch": SimpleNamespace(
                empty=lambda *args, **kwargs: _Tensor(1), float32="float32"
            ),
            "nn": SimpleNamespace(Parameter=lambda value: value),
        }
        config = SimpleNamespace(
            hidden_size=8,
            num_attention_heads=4,
            qk_nope_head_dim=2,
            qk_rope_head_dim=2,
            v_head_dim=2,
            q_lora_rank=2,
            kv_lora_rank=2,
            mla_use_output_gate=True,
            linear_attn_config={
                "head_dim": 2,
                "num_heads": 4,
                "short_conv_kernel_size": 3,
                "use_full_rank_gate": True,
            },
        )
        return namespace, config, Linear, manager, events, result

    def test_mla_constructor_installs_fusion_inside_output_gate(self):
        namespace, config, linear, manager, events, result = self._dependencies()

        class BaseMLA:
            def __init__(self, **kwargs):
                self.o_proj = linear()

        attention = _constructor("KimiK3MLAAttention", BaseMLA, namespace)(
            config, 0, sp_moe=True
        )
        gate_input = _Tensor(8, name="gate_input")
        gate = _Tensor(8, name="gate")
        gated = _Tensor(8, name="gated")
        attention._gate_hidden_states = gate_input
        attention._compute_output_gate = Mock(return_value=gate)

        def apply_gate(x, gate):
            events.append(("gate", x))
            return gated

        gate_module = SimpleNamespace(
            covered=lambda x, gate: True,
            kimi_k3_mla_output_gate=Mock(side_effect=apply_gate),
        )
        with patch.dict(
            sys.modules,
            {
                "sglang.kernels.ops.attention": SimpleNamespace(
                    mla_output_gate=gate_module
                )
            },
        ):
            output = attention.o_proj(_Tensor(8, name="attention_output"))
        self.assertEqual([name for name, _ in events], ["gate", "fused"])
        manager.try_fused_o_proj.assert_called_once_with(attention.o_proj, gated)
        attention._compute_output_gate.assert_called_once_with(gate_input)
        self.assertIs(output[0], result)
        self.assertIsNone(attention._gate_hidden_states)
        self.assertFalse(attention.o_proj.reduce_results)

    def test_kda_constructor_and_forward_apply_norm_gate_before_fusion(self):
        namespace, config, _, manager, events, result = self._dependencies()
        attention = _constructor("KimiK3DeltaAttention", object, namespace)(
            0, config.hidden_size, config, sp_moe=True
        )
        gated = _Tensor(8, name="norm_gated")

        def norm_gate(value, gate):
            events.append(("gate", value))
            return gated

        attention.forward_qkvbfg_fused = Mock(
            return_value=tuple(_Tensor(8) for _ in range(4))
        )
        attention.attn = Mock(return_value=_Tensor(8, name="attention_output"))
        attention.o_norm = Mock(side_effect=norm_gate)
        forward = _definitions(MODEL, ["forward"], {}, "KimiK3DeltaAttention").forward
        batch = SimpleNamespace(forward_mode=SimpleNamespace(is_decode=lambda: True))
        output = forward(attention, _Tensor(8), _Tensor(8, 1), batch, object())
        self.assertEqual([name for name, _ in events], ["gate", "fused"])
        manager.try_fused_o_proj.assert_called_once_with(attention.o_proj, gated)
        self.assertIs(output, result)
        self.assertFalse(attention.o_proj.reduce_results)


class TestKimiK3SPGemmPreparation(unittest.TestCase):
    def _prepare(self, state):
        return _definitions(
            SP, ["prepare_gemm_fusion"], {"_STATE": state}
        ).prepare_gemm_fusion

    def test_prepare_sizes_workspace_for_the_prefill_ceiling(self):
        manager = SimpleNamespace(prepare=Mock())
        with patch.dict(
            sys.modules,
            {
                "sglang.srt.runtime_context": SimpleNamespace(
                    max_prefill_buffer_tokens=lambda: 16384
                )
            },
        ):
            self._prepare(SimpleNamespace(gemm_fusion_manager=manager))()
        manager.prepare.assert_called_once_with(16384)

    def test_prepare_with_fusion_off_is_a_no_op(self):
        for state in (None, SimpleNamespace(gemm_fusion_manager=None)):
            with self.subTest(state=state):
                with patch.dict(sys.modules, {"sglang.srt.runtime_context": None}):
                    self._prepare(state)()

    def test_language_model_hook_only_prepares_attention_residual_models(self):
        prepare = Mock()
        hook = _definitions(
            MODEL,
            ["prepare_before_cuda_graph_capture"],
            {"k3_sp_collective": SimpleNamespace(prepare_gemm_fusion=prepare)},
            "KimiK3LinearForCausalLM",
        ).prepare_before_cuda_graph_capture
        hook(
            SimpleNamespace(config=SimpleNamespace(attn_res_block_size=None)), object()
        )
        prepare.assert_not_called()
        hook(SimpleNamespace(config=SimpleNamespace(attn_res_block_size=4)), object())
        prepare.assert_called_once_with()

    def test_multimodal_hook_delegates_to_the_language_model(self):
        hook = _definitions(
            MODEL,
            ["prepare_before_cuda_graph_capture"],
            {},
            "KimiK3ForConditionalGeneration",
        ).prepare_before_cuda_graph_capture
        language_model = SimpleNamespace(prepare_before_cuda_graph_capture=Mock())
        runner = object()
        hook(SimpleNamespace(language_model=language_model), runner)
        language_model.prepare_before_cuda_graph_capture.assert_called_once_with(runner)
        hook(SimpleNamespace(language_model=None), runner)

    def test_model_forward_does_not_prepare(self):
        class StopBeforeEmbedding(Exception):
            pass

        sp = Mock()
        forward = _definitions(
            MODEL,
            ["forward"],
            {
                "k3_sp_collective": sp,
                "get_parallel": Mock(side_effect=StopBeforeEmbedding),
            },
            "KimiK3LinearModel",
        ).forward
        owner = SimpleNamespace(config=SimpleNamespace(attn_res_block_size=4))
        with self.assertRaises(StopBeforeEmbedding):
            forward(owner, _Tensor(8, 1), _Tensor(8, 1), _forward_batch())
        self.assertEqual(sp.mock_calls, [])


class TestKimiK3SPGemmInitialization(unittest.TestCase):
    def _initialize(
        self,
        *,
        fusion_enabled,
        sp_enabled=True,
        attn_res_enabled=False,
        manager_available=True,
    ):
        class Comm:
            disabled = False
            has_multicast = True
            obj = object()

        group = SimpleNamespace(world_size=4, ca_comm=Comm(), cpu_group=object())
        envs = SimpleNamespace(
            SGLANG_K3_SP_COLLECTIVE=SimpleNamespace(
                is_set=Mock(return_value=True), get=Mock(return_value=sp_enabled)
            ),
            SGLANG_K3_SP_GEMM_FUSION=SimpleNamespace(
                get=Mock(return_value=fusion_enabled)
            ),
            SGLANG_K3_SP_ATTN_RES=SimpleNamespace(
                get=Mock(return_value=attn_res_enabled)
            ),
        )
        factory = Mock(return_value=object() if manager_available else None)
        gather = Mock()
        namespace = {
            "_STATE": None,
            "_INITIALIZED": False,
            "_SUPPORTED_WORLD_SIZES": (4, 8),
            "_HIDDEN_SIZE": 7168,
            "logger": Mock(),
            "envs": envs,
            "torch": SimpleNamespace(
                device=lambda device: device,
                distributed=SimpleNamespace(all_gather_object=gather),
            ),
            "get_exec": lambda: SimpleNamespace(
                moe=SimpleNamespace(moe_a2a_backend="deepep")
            ),
            "get_parallel": lambda: SimpleNamespace(attn_tp_group=group),
        }
        definitions = _definitions(SP, ["_State", "_init_state"], namespace)
        imports = {
            "sglang.srt.distributed.device_communicators.custom_all_reduce_v2": SimpleNamespace(
                CustomAllReduceV2=Comm
            ),
            "sglang.srt.utils.common": SimpleNamespace(get_device_sm=lambda: 103),
            "sglang.kernels.ops.attention": SimpleNamespace(
                attn_res=SimpleNamespace(register_comm=Mock())
            ),
            "sglang.kernels.ops.communication": SimpleNamespace(
                sp_collective=SimpleNamespace(
                    get_dispatch=Mock(return_value=object()), register_comm=Mock()
                )
            ),
            "sglang.srt.layers.communication.k3_sp_gemm_fusion": SimpleNamespace(
                create_gemm_fusion_manager=factory
            ),
        }
        with patch.dict(sys.modules, imports):
            first = definitions._init_state()
            second = definitions._init_state()
        self.assertIs(first, second)
        return SimpleNamespace(
            state=first,
            envs=envs,
            factory=factory,
            gather=gather,
            group=group,
            logger=namespace["logger"],
        )

    def test_opt_in_is_read_once_and_manager_created_once(self):
        run = self._initialize(fusion_enabled=True)
        run.envs.SGLANG_K3_SP_GEMM_FUSION.get.assert_called_once()
        run.factory.assert_called_once_with(run.group, 7168)
        self.assertIs(run.state.gemm_fusion_manager, run.factory.return_value)
        run.gather.assert_not_called()

    def test_flag_off_does_not_probe_abi_or_create_workspace(self):
        run = self._initialize(fusion_enabled=False)
        run.envs.SGLANG_K3_SP_GEMM_FUSION.get.assert_called_once()
        run.factory.assert_not_called()
        self.assertIsNone(run.state.gemm_fusion_manager)
        run.gather.assert_not_called()

    def test_attn_res_priority_skips_gemm_initialization_with_warning(self):
        for fusion_enabled in (False, True):
            with self.subTest(fusion_enabled=fusion_enabled):
                run = self._initialize(
                    fusion_enabled=fusion_enabled, attn_res_enabled=True
                )
                run.factory.assert_not_called()
                self.assertIsNone(run.state.gemm_fusion_manager)
                run.gather.assert_not_called()
                warned = any(
                    "SGLANG_K3_SP_GEMM_FUSION is ignored" in call.args[0]
                    for call in run.logger.warning.call_args_list
                )
                self.assertEqual(warned, fusion_enabled)

    def test_unavailable_manager_does_not_log_fusion_enabled(self):
        run = self._initialize(fusion_enabled=True, manager_available=False)
        run.factory.assert_called_once_with(run.group, 7168)
        self.assertIsNone(run.state.gemm_fusion_manager)
        self.assertFalse(
            any(
                "K3 SP GEMM fusion enabled" in call.args[0]
                for call in run.logger.info.call_args_list
            )
        )

    def test_sp_disabled_never_reads_gemm_flag(self):
        run = self._initialize(fusion_enabled=True, sp_enabled=False)
        self.assertIsNone(run.state)
        run.envs.SGLANG_K3_SP_GEMM_FUSION.get.assert_not_called()
        run.factory.assert_not_called()
        run.gather.assert_not_called()


class TestKimiK3SPGemmStandardResidual(unittest.TestCase):
    def test_standard_residual_keeps_all_reduce_and_full_rows(self):
        # A standard-residual layer has no token-shard contract: even when
        # SP-MoE is configured it must never activate GEMM+RS fusion.
        forward = _definitions(
            MODEL,
            ["forward"],
            {},
            "KimiK3DecoderLayer",
        ).forward
        for has_residual in (False, True):
            with self.subTest(has_residual=has_residual):
                hidden = _Tensor(8, name="hidden")
                residual = _Tensor(8, name="residual") if has_residual else None
                normalized = _Tensor(8, name="normalized")
                projected = _Tensor(8, name="projected")
                reduced = _Tensor(8, name="reduced")
                mlp_input = _Tensor(8, name="mlp_input")
                carried = residual if has_residual else hidden
                owner = SimpleNamespace(
                    input_layernorm=Mock(
                        return_value=(
                            (normalized, carried) if has_residual else normalized
                        )
                    ),
                    _run_self_attn=Mock(return_value=projected),
                    _finish_attn_reduce=Mock(return_value=(reduced, -1, False)),
                    post_attention_layernorm=Mock(return_value=(mlp_input, carried)),
                    mlp=Mock(return_value=_Tensor(8, name="mlp")),
                    _forward_attn_residual=Mock(),
                )
                result = forward(
                    owner,
                    _Tensor(8, 1),
                    hidden,
                    _forward_batch(),
                    residual,
                    None,
                    object(),
                )
                owner._finish_attn_reduce.assert_called_once_with(
                    projected, allow_scatter=False
                )
                owner._forward_attn_residual.assert_not_called()
                self.assertEqual(result[0].shape[0], 8)
                self.assertIs(result[1], carried)
                self.assertFalse(result[2])


class TestKimiK3SPGemmReductionCompletion(unittest.TestCase):
    def _run_residual(
        self,
        *,
        native_fused=False,
        keep_sharded=True,
        write=False,
        gemm_fused=False,
        idle=False,
        input_sharded=False,
        sp_moe=True,
    ):
        group = SimpleNamespace(world_size=4, rank_in_group=1)
        gathered = _Tensor(8, name="gathered")
        o_proj_call = SimpleNamespace(fused_output=None)
        o_proj_call_context = Mock(
            side_effect=lambda *args, **kwargs: nullcontext(o_proj_call)
        )
        namespace = {
            "get_parallel": lambda: SimpleNamespace(attn_tp_group=group),
            "_sp_all_gather_rows": Mock(return_value=gathered),
            "_sp_local_rows": lambda hidden: slice(2, 4),
            "k3_sp_collective": SimpleNamespace(o_proj_fusion_call=o_proj_call_context),
            "nullcontext": nullcontext,
        }
        forward = _definitions(
            MODEL,
            ["_forward_attn_residual"],
            namespace,
            "KimiK3DecoderLayer",
        )._forward_attn_residual
        projected = _Tensor(8, name="o_proj")
        shard = _Tensor(2, name="rs")
        prefix = _Tensor(2 if input_sharded else 8, name="prefix")
        agg2_prefix = _Tensor(2, name="agg2_prefix")
        if gemm_fused:
            o_proj_call.fused_output = SimpleNamespace(output=shard, shard_start_row=2)
        attn_res = SimpleNamespace(
            forward=Mock(
                side_effect=(
                    [] if input_sharded else [(_Tensor(8, name="agg1"), prefix)]
                )
                + [(_Tensor(2, name="agg2"), agg2_prefix)]
            ),
            forward_sp_all_gather=Mock(return_value=(_Tensor(8, name="agg1"), prefix)),
            forward_sp_reduce_scatter=Mock(
                return_value=(shard, agg2_prefix) if native_fused else None
            ),
        )
        owner = SimpleNamespace(
            _sp_moe=sp_moe,
            all_reduce_fusion=False,
            is_block_write_layer=write,
            self_attention_res_proj=object(),
            self_attention_res_norm=object(),
            input_layernorm=object(),
            mlp_res_proj=object(),
            mlp_res_norm=object(),
            post_attention_layernorm=object(),
            _run_self_attn=Mock(return_value=shard if gemm_fused else projected),
            _finish_attn_reduce=Mock(return_value=(shard, 2, True)),
            mlp=Mock(return_value=_Tensor(2, name="moe")),
        )
        result = forward(
            owner,
            _Tensor(8, 1),
            _Tensor(2 if input_sharded else 8, name="hidden"),
            _Tensor(2 if input_sharded else 8, name="incoming_prefix"),
            attn_res,
            _forward_batch(idle=idle),
            object(),
            input_sharded,
            keep_sharded,
        )
        return SimpleNamespace(
            result=result,
            owner=owner,
            attn_res=attn_res,
            prefix=prefix,
            projected=projected,
            shard=shard,
            agg2_prefix=agg2_prefix,
            all_gather=namespace["_sp_all_gather_rows"],
            o_proj_call_context=o_proj_call_context,
        )

    def test_fused_completion_skips_both_rs_paths_but_runs_aggregation(self):
        run = self._run_residual(gemm_fused=True)
        run.attn_res.forward_sp_reduce_scatter.assert_not_called()
        run.owner._finish_attn_reduce.assert_not_called()
        run.o_proj_call_context.assert_called_once_with(run.prefix, 8)
        self.assertEqual(run.attn_res.forward.call_count, 2)
        agg2 = run.attn_res.forward.call_args
        self.assertIs(agg2.args[0], run.shard)
        self.assertIsNone(agg2.args[1])
        self.assertEqual(agg2.kwargs["rows"], slice(2, 4))
        self.assertIs(run.owner.mlp.call_args.kwargs["prefix_sum"], run.agg2_prefix)
        self.assertTrue(run.result[2])

    def test_fused_block_write_consumes_no_prior_prefix(self):
        run = self._run_residual(gemm_fused=True, write=True)
        run.o_proj_call_context.assert_called_once_with(None, 8)
        run.attn_res.forward_sp_reduce_scatter.assert_not_called()
        self.assertEqual(run.attn_res.forward.call_count, 2)

    def test_sharded_input_preserves_local_prefix_and_global_token_count(self):
        run = self._run_residual(gemm_fused=True, input_sharded=True)
        run.attn_res.forward_sp_all_gather.assert_called_once()
        self.assertEqual(run.prefix.shape[0], 2)
        run.o_proj_call_context.assert_called_once_with(run.prefix, 8)
        run.attn_res.forward_sp_reduce_scatter.assert_not_called()
        run.owner._finish_attn_reduce.assert_not_called()
        run.attn_res.forward.assert_called_once()
        agg2 = run.attn_res.forward.call_args
        self.assertIs(agg2.args[0], run.shard)
        self.assertIsNone(agg2.args[1])
        self.assertEqual(agg2.kwargs["rows"], slice(2, 4))

    def test_idle_bypasses_o_proj_fusion_call(self):
        run = self._run_residual(idle=True)
        run.o_proj_call_context.assert_not_called()

    def test_non_sp_attention_bypasses_o_proj_fusion_call(self):
        run = self._run_residual(sp_moe=False)
        run.o_proj_call_context.assert_not_called()
        run.attn_res.forward_sp_reduce_scatter.assert_not_called()
        run.owner._finish_attn_reduce.assert_not_called()
        self.assertEqual(run.attn_res.forward.call_count, 2)
        agg2 = run.attn_res.forward.call_args
        self.assertIs(agg2.args[0], run.projected)
        self.assertIs(agg2.args[1], run.prefix)
        self.assertIsNone(agg2.kwargs["rows"])

    def test_baseline_retains_native_rs_and_prefix_completion(self):
        run = self._run_residual()
        run.attn_res.forward_sp_reduce_scatter.assert_called_once()
        run.owner._finish_attn_reduce.assert_called_once_with(
            run.projected, allow_scatter=True, residual=run.prefix
        )
        self.assertEqual(run.attn_res.forward.call_count, 2)
        agg2 = run.attn_res.forward.call_args
        self.assertIs(agg2.args[0], run.shard)
        self.assertIsNone(agg2.args[1])
        self.assertEqual(agg2.kwargs["rows"], slice(2, 4))
        self.assertTrue(run.result[2])

    def test_native_attn_res_fusion_keeps_its_aggregation_contract(self):
        run = self._run_residual(native_fused=True)
        run.owner._finish_attn_reduce.assert_not_called()
        # The preexisting RS+attn-res fusion really includes Aggregation 2.
        self.assertEqual(run.attn_res.forward.call_count, 1)
        self.assertIs(run.owner.mlp.call_args.kwargs["prefix_sum"], run.agg2_prefix)

    def test_baseline_block_write_passes_no_prefix_to_rs(self):
        run = self._run_residual(write=True)
        self.assertIsNone(run.attn_res.forward_sp_reduce_scatter.call_args.args[1])
        self.assertIsNone(run.owner._finish_attn_reduce.call_args.kwargs["residual"])

    def test_full_output_still_all_gathers_after_moe(self):
        run = self._run_residual(keep_sharded=False)
        run.all_gather.assert_called_once_with(run.owner.mlp.return_value)
        self.assertIs(run.result[0], run.all_gather.return_value)
        self.assertFalse(run.result[2])


if __name__ == "__main__":
    unittest.main()
