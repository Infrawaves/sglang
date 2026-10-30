"""Kimi-K3 EPLB integration tests, including the MXFP4 MegaMoE path."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from sglang.srt.models import kimi_k3
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from torch import nn

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


def make_module(cls):
    module = cls.__new__(cls)
    nn.Module.__init__(module)
    return module


class TestKimiK3Eplb(CustomTestCase):
    def test_expert_metadata(self):
        for groups in (None, 0, 1, 2):
            with self.subTest(groups=groups):
                config = SimpleNamespace(
                    num_hidden_layers=5,
                    n_routed_experts=4,
                    num_experts=4,
                    num_expert_group=groups,
                )
                for cls, argument in (
                    (kimi_k3.KimiK3LinearForCausalLM, config),
                    (
                        kimi_k3.KimiK3ForConditionalGeneration,
                        SimpleNamespace(text_config=config),
                    ),
                ):
                    metadata = cls.get_model_config_for_expert_location(argument)
                    self.assertEqual(metadata.num_layers, 5)
                    self.assertEqual(metadata.num_logical_experts, 4)
                    self.assertEqual(metadata.num_groups, groups or None)

        config.n_routed_experts = None
        config.num_experts = None
        self.assertIsNone(
            kimi_k3.KimiK3LinearForCausalLM.get_model_config_for_expert_location(
                config
            )
        )
        self.assertIsNone(
            kimi_k3.KimiK3ForConditionalGeneration.get_model_config_for_expert_location(
                SimpleNamespace(text_config=config)
            )
        )

    def test_expert_metrics_metadata_allows_eplb_layout(self):
        config = SimpleNamespace(
            num_hidden_layers=5,
            n_routed_experts=4,
            num_experts=4,
            num_expert_group=None,
        )
        runtime_moe = SimpleNamespace(
            expert_distribution_recorder_mode="stat",
            moe_a2a_backend="megamoe",
            elastic_ep_backend=None,
            enable_eplb=True,
            ep_num_redundant_experts=2,
            init_expert_location="init_by_eplb",
        )
        with (
            patch.object(
                kimi_k3,
                "get_exec",
                return_value=SimpleNamespace(moe=runtime_moe),
            ),
            patch.object(
                kimi_k3,
                "get_parallel",
                return_value=SimpleNamespace(pp_size=1),
            ),
        ):
            metadata = (
                kimi_k3.KimiK3LinearForCausalLM.get_model_config_for_expert_location(
                    config
                )
            )
        self.assertEqual(metadata.num_logical_experts, 4)
        self.assertEqual(metadata.num_layers, 5)

    def test_expert_metrics_metadata_allows_deepep(self):
        config = SimpleNamespace(
            num_hidden_layers=5,
            n_routed_experts=4,
            num_experts=4,
            num_expert_group=None,
        )
        runtime_moe = SimpleNamespace(
            expert_distribution_recorder_mode="stat",
            moe_a2a_backend="deepep",
            deepep_mode="auto",
            elastic_ep_backend=None,
            enable_eplb=True,
            ep_num_redundant_experts=2,
            init_expert_location="trivial",
        )
        with (
            patch.object(
                kimi_k3,
                "get_exec",
                return_value=SimpleNamespace(moe=runtime_moe),
            ),
            patch.object(
                kimi_k3,
                "get_parallel",
                return_value=SimpleNamespace(pp_size=1),
            ),
        ):
            metadata = (
                kimi_k3.KimiK3LinearForCausalLM.get_model_config_for_expert_location(
                    config
                )
            )
        self.assertEqual(metadata.num_logical_experts, 4)
        self.assertEqual(metadata.num_layers, 5)

    def test_expert_metrics_metadata_rejects_deepep_approximate_recorder(self):
        config = SimpleNamespace(
            num_hidden_layers=5,
            n_routed_experts=4,
            num_experts=4,
            num_expert_group=None,
        )
        runtime_moe = SimpleNamespace(
            expert_distribution_recorder_mode="stat_approx",
            moe_a2a_backend="deepep",
            deepep_mode="auto",
            elastic_ep_backend=None,
        )
        with (
            patch.object(
                kimi_k3,
                "get_exec",
                return_value=SimpleNamespace(moe=runtime_moe),
            ),
            patch.object(
                kimi_k3,
                "get_parallel",
                return_value=SimpleNamespace(pp_size=1),
            ),
            self.assertRaises(ValueError),
        ):
            kimi_k3.KimiK3LinearForCausalLM.get_model_config_for_expert_location(
                config
            )

    def test_expert_metrics_metadata_rejects_unsupported_recorder(self):
        config = SimpleNamespace(
            num_hidden_layers=5,
            n_routed_experts=4,
            num_experts=4,
            num_expert_group=None,
        )
        runtime_moe = SimpleNamespace(
            expert_distribution_recorder_mode="per_token",
            moe_a2a_backend="megamoe",
            elastic_ep_backend=None,
        )
        with (
            patch.object(
                kimi_k3,
                "get_exec",
                return_value=SimpleNamespace(moe=runtime_moe),
            ),
            patch.object(
                kimi_k3,
                "get_parallel",
                return_value=SimpleNamespace(pp_size=1),
            ),
            self.assertRaises(ValueError),
        ):
            kimi_k3.KimiK3LinearForCausalLM.get_model_config_for_expert_location(
                config
            )

    def test_fused_remap_ok_follows_dispatch_algorithm_and_kill_switch(self):
        cases = {
            None: True,
            "static": True,
            "dynamic": True,
            "fake": False,
            "lp": False,
        }
        for algorithm, supported in cases.items():
            for remap_enabled in (True, False):
                with self.subTest(algorithm=algorithm, remap=remap_enabled):
                    moe = make_module(kimi_k3.KimiK3MoE)
                    with (
                        patch.object(
                            kimi_k3,
                            "get_exec",
                            return_value=SimpleNamespace(
                                moe=SimpleNamespace(ep_dispatch_algorithm=algorithm)
                            ),
                        ),
                        patch.object(
                            kimi_k3.envs.SGLANG_EPLB_FUSED_FRONT_REMAP,
                            "get",
                            return_value=remap_enabled,
                        ),
                    ):
                        self.assertEqual(
                            moe._fused_remap_ok, supported and remap_enabled
                        )

    def test_weight_views_exclude_non_routed_parameters(self):
        moe = make_module(kimi_k3.KimiK3MoE)
        moe._use_mega_moe = False
        experts = nn.Module()
        experts.num_local_experts = 4
        for name, shape in (
            ("weight", (4, 3, 2)),
            ("scale", (4, 1)),
            ("correction_bias", (4,)),
            ("global_scale", (4,)),
            ("scalar", ()),
            ("other", (3, 2)),
        ):
            experts.register_parameter(
                name, nn.Parameter(torch.ones(shape), requires_grad=False)
            )
        experts.global_scale._sglang_require_global_experts = True
        moe.experts = experts

        weights = moe.get_moe_weights()

        self.assertEqual(len(weights), 2)
        self.assertEqual(weights[0].data_ptr(), experts.weight.data_ptr())
        self.assertEqual(weights[1].data_ptr(), experts.scale.data_ptr())
        weights[0][0, 0, 0] = 7
        self.assertEqual(experts.weight[0, 0, 0].item(), 7)

    def test_mxfp4_megamoe_views_are_eplb_visible(self):
        moe = make_module(kimi_k3.KimiK3MoE)
        moe._use_mega_moe = True
        experts = nn.Module()
        experts.num_local_experts = 2

        w13 = torch.arange(12, dtype=torch.int8).reshape(2, 3, 2)
        w13_scale = torch.arange(12, dtype=torch.uint8).reshape(2, 3, 2)
        w2 = torch.arange(12, dtype=torch.int8).reshape(2, 3, 2)
        w2_scale = torch.arange(12, dtype=torch.uint8).reshape(2, 3, 2)
        for name, value in (
            ("w13_weight", w13),
            ("w13_weight_scale", w13_scale),
            ("w2_weight", w2),
            ("w2_weight_scale", w2_scale),
        ):
            experts.register_parameter(
                name,
                nn.Parameter(torch.zeros_like(value), requires_grad=False),
            )

        # This mirrors Mxfp4MoEMethod.process_weights_after_loading: the
        # transformed MegaMoE tensors are put behind the registered Params.
        experts.w13_weight.data = w13
        experts.w13_weight_scale.data = w13_scale
        experts.w2_weight.data = w2
        experts.w2_weight_scale.data = w2_scale
        experts.mega_l1_weights = (w13, w13_scale)
        experts.mega_l2_weights = (w2, w2_scale)
        moe.experts = experts

        weights = moe.get_moe_weights()

        self.assertEqual(len(weights), 4)
        self.assertEqual(
            [tensor.data_ptr() for tensor in weights],
            [
                experts.w13_weight.data_ptr(),
                experts.w13_weight_scale.data_ptr(),
                experts.w2_weight.data_ptr(),
                experts.w2_weight_scale.data_ptr(),
            ],
        )
        weights[1][1].fill_(23)
        self.assertTrue(torch.equal(experts.mega_l1_weights[1][1], weights[1][1]))

    def test_pp_layer_ids_and_multimodal_forwarding(self):
        moe = make_module(kimi_k3.KimiK3MoE)
        weight = torch.ones(4, 2)
        moe.get_moe_weights = Mock(return_value=[weight])
        text = make_module(kimi_k3.KimiK3LinearForCausalLM)
        text.model = SimpleNamespace(
            start_layer=2,
            end_layer=4,
            layers={
                0: SimpleNamespace(mlp=moe),
                2: SimpleNamespace(mlp=nn.Identity()),
                3: SimpleNamespace(mlp=moe),
                4: SimpleNamespace(mlp=moe),
            },
        )
        text._routed_experts_weights_of_layer = kimi_k3.LazyValue(
            lambda: {
                layer_id: text.model.layers[layer_id].mlp.get_moe_weights()
                for layer_id in range(text.model.start_layer, text.model.end_layer)
                if isinstance(text.model.layers[layer_id].mlp, kimi_k3.KimiK3MoE)
            }
        )
        wrapper = make_module(kimi_k3.KimiK3ForConditionalGeneration)
        wrapper.language_model = text

        result = wrapper.routed_experts_weights_of_layer

        self.assertEqual(list(result), [3])
        self.assertIs(result[3][0], weight)
        moe.get_moe_weights.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
