import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.speculative.dspark_components import dspark_planner as planner_module  # noqa: E402

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestDSparkVerifyBudget(CustomTestCase):
    def _make_tp_planner(self, group):
        parallel = SimpleNamespace(tp_group=group)
        spec = SimpleNamespace(
            speculative_dspark_confidence_sts_path=None,
            speculative_dspark_align_verify_tokens_to_graph_tier=True,
            speculative_dspark_sps_table_path="profiled",
        )
        runner = SimpleNamespace(
            req_to_token_pool=SimpleNamespace(req_generation=torch.ones(1)),
            decode_cuda_graph_runner=SimpleNamespace(
                ragged_verify_mode=True, max_bs=8, capture_num_tokens=[4, 8, 16, 32]
            ),
        )
        with (
            patch.object(planner_module, "get_spec", return_value=spec),
            patch.object(planner_module, "get_parallel", return_value=parallel),
            patch.object(
                planner_module,
                "get_schedule",
                return_value=SimpleNamespace(disable_overlap_schedule=False),
            ),
            patch.object(planner_module, "is_dp_attention_enabled", return_value=False),
            planner_module.envs.SGLANG_RAGGED_VERIFY_MODE.override("compact"),
            patch.object(
                planner_module,
                "build_sps_cost_table",
                return_value=planner_module.SpsCostTable([1, 8], [150.0, 90.0], 32),
            ),
        ):
            return planner_module.DSparkVerifyPlanner(
                draft_model=SimpleNamespace(confidence_head=object()),
                gamma=3,
                model_runner=runner,
                device="cpu",
                tp_rank=group.rank_in_group,
                verify_num_draft_tokens=4,
                tp_sync=Mock(),
            )

    def test_tp_budget_uses_rank_zero_even_when_confidence_readiness_differs(self):
        group = Mock(world_size=2, rank_in_group=0)
        leader = self._make_tp_planner(group)
        peer_group = Mock(world_size=2, rank_in_group=1)
        peer = self._make_tp_planner(peer_group)
        indices = torch.tensor([0])
        ready = SimpleNamespace(confidence=torch.ones(1, 3), generation=indices)
        for rank0_budget in (None, 0, 3):
            with self.subTest(rank0_budget=rank0_budget):
                resolved = None if rank0_budget is None else ready
                group.broadcast_object.side_effect = lambda value: value
                leader._budget_planner.compute_budget = Mock(return_value=rank0_budget)
                result = leader._budget_from_resolved(
                    resolved=resolved, req_pool_indices_cpu=indices
                )
                group.broadcast_object.assert_called_with(rank0_budget)
                peer_group.broadcast_object.return_value = result
                result = peer._budget_from_resolved(
                    resolved=ready if resolved is None else None,
                    req_pool_indices_cpu=indices,
                )
                peer_group.broadcast_object.assert_called_with(None)
                self.assertEqual(result, rank0_budget)

    def test_shared_graph_bucket_fills_budget_after_admission(self):
        planner = self._make_tp_planner(Mock(world_size=1, rank_in_group=0))
        indices = torch.arange(3)
        lens = torch.tensor([4, 2, 2], dtype=torch.int32)
        with (
            patch.object(planner, "_schedule_verify_lens", return_value=lens) as topk,
            patch.object(
                planner_module.RaggedVerifyLayout, "from_verify_lens_device"
            ) as pack,
        ):
            planner.schedule_layout(
                req_pool_indices=indices,
                prefix_lens=indices,
                device=torch.device("cpu"),
                confidence=torch.ones(3, 3),
                budget=1,
                global_num_reqs=4,
                dp_tier_num_tokens=7,
            )
            self.assertEqual(topk.call_args.kwargs["budget"], 5)
            pack.assert_called_once_with(verify_lens=lens, graph_num_tokens=8)

    def test_overflow_uses_eager_before_aligning_or_scheduling_topk(self):
        planner = self._make_tp_planner(Mock(world_size=1, rank_in_group=0))
        planner._dynamic_graph_tier = False  # DP uses a globally pinned tier.
        planner._schedule_verify_lens = Mock()
        indices = torch.tensor([0])
        layout = planner.schedule_layout(
            req_pool_indices=indices,
            prefix_lens=indices,
            device=torch.device("cpu"),
            confidence=torch.ones(1, 3),
            budget=1,
            global_num_reqs=9,
        )
        self.assertIsNone(layout)
        planner._schedule_verify_lens.assert_not_called()


if __name__ == "__main__":
    unittest.main()
