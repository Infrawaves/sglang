"""CPU-only contract tests for the arguments of prefill CP with DCP.

Prefill CP with DCP runs one group that is the TP group, the attention-CP group
and the DCP group at once (tp_size == attn_cp_size == dcp_size). Only
Kimi-K3.1 implements it, and its first stage serves aggregated,
non-speculative flashmla_kv without HiCache or a prefill CUDA graph.
"""

from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.arg_groups import parallel_hook
from sglang.srt.arg_groups.cp_tp_group_sharing_hook import resolve_cp_tp_group_sharing
from sglang.srt.arg_groups.cuda_graph_hook import apply_cuda_graph_compatibility
from sglang.srt.arg_groups.hicache_hook import resolve_hicache_dcp_compatibility
from sglang.srt.arg_groups.mega_moe_hook import validate_mega_moe_token_budget
from sglang.srt.arg_groups.model_overrides import kimi_k3 as kimi_k3_overrides
from sglang.srt.arg_groups.overrides import resolution_result
from sglang.srt.environ import envs
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    CudaGraphConfig,
    Phase,
    PhaseConfig,
)
from sglang.srt.runtime_context import override_platform
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

KIMI_K31 = "KimiK31ForConditionalGeneration"

PREFILL_CP_WITH_DCP = dict(
    tp_size=8,
    enable_prefill_cp=True,
    attn_cp_size=8,
    cp_strategy="interleave",
    dcp_size=8,
)


def _args(arch=KIMI_K31, **fields) -> ServerArgs:
    values = dict(model_path="dummy", **PREFILL_CP_WITH_DCP)
    values.update(fields)
    args = ServerArgs(**values)
    args._model_config = SimpleNamespace(
        hf_config=SimpleNamespace(architectures=[arch])
    )
    return args


class TestDecodeContextParallelismGate(CustomTestCase):
    """`handle_decode_context_parallelism` admits prefill CP with DCP only as
    one group, and only for the architectures that implement it."""

    @staticmethod
    def _handle(args):
        with (
            override_platform(is_cuda=True, is_hip=False),
            patch(
                "sglang.srt.arg_groups.overrides.is_fi_a2a_supported",
                return_value=False,
            ),
        ):
            parallel_hook.handle_decode_context_parallelism(args)

    def test_kimi_k31_one_group_is_admitted(self):
        self._handle(_args())

    def test_unset_cp_strategy_is_admitted_before_it_resolves(self):
        self._handle(_args(cp_strategy=None))

    def test_other_architectures_stay_rejected(self):
        # DeepSeek DSA DCP has no prefill-CP rejection of its own.
        for arch in (
            "DeepseekV32ForCausalLM",
            "Glm5NextForConditionalGeneration",
            "KimiK3ForConditionalGeneration",
        ):
            with self.subTest(arch=arch):
                with self.assertRaisesRegex(ValueError, "only supported for"):
                    self._handle(_args(arch=arch))

    def test_widths_must_be_equal(self):
        for fields in (
            dict(attn_cp_size=4),
            dict(dcp_size=4),
            dict(tp_size=16, attn_cp_size=8, dcp_size=8),
        ):
            with self.subTest(**fields):
                with self.assertRaisesRegex(
                    ValueError, "--dcp-size == --attn-cp-size == --tp-size"
                ):
                    self._handle(_args(**fields))

    def test_zigzag_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "interleave"):
            self._handle(_args(cp_strategy="zigzag"))

    def test_page_layout_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "--dcp-kv-layout token"):
            self._handle(_args(dcp_kv_layout="page"))


class TestCpTpGroupSharingWithDcp(CustomTestCase):
    """CP-TP group sharing admits DCP for Kimi-K3.1 alone, and only on the CP
    group itself."""

    @staticmethod
    def _resolve(arch, **fields):
        args = _args(arch=arch, **fields)
        model = SimpleNamespace(hf_config=SimpleNamespace(architectures=[arch]))
        with patch(
            "sglang.srt.arg_groups.cp_tp_group_sharing_hook.mambaish_config",
            return_value=SimpleNamespace(linear_attn_config={"num_heads": 96}),
        ):
            resolve_cp_tp_group_sharing(args, model)
        return args

    def test_kimi_k31_shares_the_group_with_dcp(self):
        args = self._resolve(KIMI_K31)
        self.assertTrue(resolution_result(args, "enable_cp_tp_group_sharing"))

    def test_kimi_k31_dcp_narrower_than_cp_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "dcp-size != attn-cp-size"):
            self._resolve(KIMI_K31, dcp_size=4)

    def test_other_sharing_models_still_reject_dcp(self):
        with self.assertRaisesRegex(ValueError, "--dcp-size > 1 is not supported"):
            self._resolve("Glm5NextForConditionalGeneration")


class TestKimiK31PrefillCpWithDcpOverrides(CustomTestCase):
    """Kimi-K3.1 DCP admits prefill CP only as one group, and its first stage
    rejects speculation, PD and DSA backends other than flashmla_kv."""

    @staticmethod
    def _overrides(**fields):
        cfg = SimpleNamespace(
            dcp_size=8,
            tp_size=8,
            dcp_comm_backend="fi_a2a",
            dcp_replicate_q_proj=None,
            speculative_algorithm=None,
            speculative_draft_attention_backend=None,
            disaggregation_mode="null",
            disaggregation_transfer_backend="mooncake",
            enable_hierarchical_cache=False,
            enable_hisparse=False,
            enable_prefill_cp=True,
            attn_cp_size=8,
            dsa_prefill_backend=None,
            dsa_decode_backend=None,
        )
        for name, value in fields.items():
            setattr(cfg, name, value)
        return kimi_k3_overrides._kimi_k31_dcp_overrides(cfg)

    def test_one_group_is_admitted(self):
        self._overrides()
        self._overrides(
            dsa_prefill_backend="flashmla_kv", dsa_decode_backend="flashmla_kv"
        )

    def test_other_prefill_cp_layouts_stay_rejected(self):
        for fields in (
            dict(attn_cp_size=4),
            dict(enable_prefill_cp=False, attn_cp_size=8),
        ):
            with self.subTest(**fields):
                with self.assertRaisesRegex(ValueError, "prefill context parallel"):
                    self._overrides(**fields)

    def test_speculative_decoding_is_rejected(self):
        for algorithm in ("DFLASH", "DSPARK"):
            with self.subTest(algorithm=algorithm):
                with self.assertRaisesRegex(
                    ValueError, "speculative decoding with prefill context parallel"
                ):
                    self._overrides(speculative_algorithm=algorithm)

    def test_pd_disaggregation_is_rejected(self):
        for mode in ("prefill", "decode"):
            with self.subTest(mode=mode):
                with self.assertRaisesRegex(ValueError, "PD disaggregation"):
                    self._overrides(disaggregation_mode=mode)

    def test_dsa_backends_other_than_flashmla_kv_are_rejected(self):
        for backend in ("trtllm", "prims_ts"):
            with self.subTest(backend=backend):
                with self.assertRaisesRegex(ValueError, "flashmla_kv"):
                    self._overrides(
                        dsa_prefill_backend=backend, dsa_decode_backend=backend
                    )

    def test_dcp_without_prefill_cp_keeps_trtllm_and_speculation(self):
        # The prefill-CP rules must not narrow plain DCP.
        self._overrides(
            enable_prefill_cp=False,
            attn_cp_size=None,
            dsa_prefill_backend="trtllm",
            dsa_decode_backend="trtllm",
            speculative_algorithm="DFLASH",
        )


class TestHiCacheWithPrefillCpAndDcp(CustomTestCase):
    """HiCache under DCP is rejected once prefill CP shards tokens too."""

    @staticmethod
    def _resolve(**fields):
        args = _args(enable_hierarchical_cache=True, **fields)
        with patch(
            "sglang.srt.arg_groups.hicache_hook.use_mla_backend", return_value=True
        ):
            resolve_hicache_dcp_compatibility(args)

    def test_prefill_cp_with_dcp_is_rejected(self):
        with self.assertRaisesRegex(NotImplementedError, "--enable-prefill-cp"):
            self._resolve()

    def test_dcp_without_prefill_cp_is_admitted(self):
        self._resolve(enable_prefill_cp=False, attn_cp_size=1)


class TestPrefillCudaGraphWithPrefillCpAndDcp(CustomTestCase):
    """An explicit prefill graph skips the rules that would disable it, so the
    combination must be rejected outright."""

    @staticmethod
    def _apply(backend, **fields):
        args = _args(
            cuda_graph_config=CudaGraphConfig(prefill=PhaseConfig(backend=backend)),
            **fields,
        )
        args._cuda_graph_config_locked = {(Phase.PREFILL, "backend")}
        apply_cuda_graph_compatibility(args)

    def test_explicit_prefill_graph_is_rejected(self):
        for backend in (Backend.BREAKABLE, Backend.FULL):
            with self.subTest(backend=backend):
                with self.assertRaisesRegex(ValueError, "prefill CUDA graph"):
                    self._apply(backend)

    def test_explicit_disabled_is_admitted(self):
        self._apply(Backend.DISABLED)

    def test_explicit_graph_without_prefill_cp_is_left_alone(self):
        self._apply(Backend.BREAKABLE, enable_prefill_cp=False, attn_cp_size=1)


class TestMegaMoeBudgetWithPrefillCpAndDcp(CustomTestCase):
    """The per-request position split adds up to one row per request to a
    rank: 16384 tokens over 8 ranks with 64 requests need 2048 + 64 rows."""

    @staticmethod
    def _validate(max_tokens_per_rank, **fields):
        args = _args(
            moe_a2a_backend="megamoe",
            ep_size=8,
            chunked_prefill_size=16384,
            max_prefill_tokens=16384,
            max_running_requests=64,
            disaggregation_mode="prefill",
            cuda_graph_config=CudaGraphConfig(
                prefill=PhaseConfig(backend=Backend.DISABLED)
            ),
            **fields,
        )
        with (
            override_platform(is_hip=False),
            envs.SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK.override(
                max_tokens_per_rank
            ),
        ):
            validate_mega_moe_token_budget(args, KIMI_K31)

    def test_budget_covers_one_extra_row_per_request(self):
        self._validate(2112)
        with self.assertRaisesRegex(ValueError, "required_per_rank=2112"):
            self._validate(2111)

    def test_prefill_cp_without_dcp_keeps_the_even_split(self):
        self._validate(2048, dcp_size=1)
