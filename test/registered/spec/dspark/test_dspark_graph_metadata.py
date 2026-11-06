"""Shared graph utility transport in the existing scheduler MLP exchange."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.scheduler_components import dp_attn  # noqa: E402
from sglang.srt.model_executor.forward_batch_info import ForwardMode  # noqa: E402
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def _info(utilities=None, count=3):
    return dp_attn.MLPSyncBatchInfo(
        dp_size=1,
        tp_size=1,
        cp_size=1,
        num_tokens=2,
        num_tokens_for_logprob=2,
        can_run_decode_cuda_graph=True,
        can_run_draft_cuda_graph=True,
        can_run_prefill_cuda_graph=False,
        is_extend_in_batch=False,
        local_can_run_tbo=False,
        local_forward_mode=ForwardMode.DECODE.value,
        spec_graph_candidate_count=count,
        spec_graph_utilities=utilities,
    )


class TestGraphUtilityTransport(CustomTestCase):
    def test_idle_and_prebuilt_have_neutral_fixed_width(self):
        for batch in (
            None,
            SimpleNamespace(forward_mode=ForwardMode.IDLE),
            SimpleNamespace(forward_mode=ForwardMode.PREBUILT),
        ):
            self.assertEqual(dp_attn.local_spec_graph_utilities(batch, 3), [0] * 3)

    def test_active_missing_or_wrong_width_disables_search(self):
        for mode, utilities in (
            (ForwardMode.DECODE, None),
            (ForwardMode.DECODE, [10, 20]),
            (ForwardMode.EXTEND, [10, 20, 30]),
        ):
            batch = SimpleNamespace(
                forward_mode=mode,
                spec_graph_utilities=utilities,
                spec_verify_budget_presync=True,
            )
            self.assertEqual(dp_attn.local_spec_graph_utilities(batch, 3), [-1] * 3)

    def test_decode_curve_requires_this_rounds_presync(self):
        batch = SimpleNamespace(
            forward_mode=ForwardMode.DECODE,
            spec_graph_utilities=[10, 20, 30],
            spec_verify_budget_presync=False,
        )
        self.assertEqual(dp_attn.local_spec_graph_utilities(batch, 3), [-1] * 3)
        batch.spec_verify_budget_presync = True
        self.assertEqual(dp_attn.local_spec_graph_utilities(batch, 3), [10, 20, 30])

    def test_row_width_and_default_legacy_layout(self):
        info = _info([10, 20, 30])
        self.assertEqual(info._get_local_tensor("cpu").tolist()[11:], [10, 20, 30])
        self.assertEqual(info._get_fallback_tensor("cpu").tolist()[11:], [0, 0, 0])
        self.assertEqual(_info(count=0)._get_local_tensor("cpu").numel(), 11)
        self.assertEqual(_info()._get_local_tensor("cpu").tolist()[11:], [-1] * 3)

    def test_single_dp_finalize_retains_curve(self):
        info = _info([10, 20, 30])
        info.finalize_local()
        self.assertEqual(info.global_spec_graph_utilities, [[10, 20, 30]])
        legacy = _info(count=0)
        legacy.finalize_local()
        self.assertIsNone(legacy.global_spec_graph_utilities)

    def test_rank_zero_curve_and_inactive_dp(self):
        # A confidence event may be ready only on one TP rank. Only rank 0 of
        # each DP participates in shared selection; an inactive DP is neutral.
        rows = [
            _info(curve) for curve in ([10, 20, 30], [-1] * 3, [40, 50, 60], [9] * 3)
        ]
        stacked = torch.stack([row._get_local_tensor("cpu") for row in rows])
        info = rows[0]
        info.dp_size, info.tp_size = 2, 2

        def exchange(output, local, group=None):
            output.copy_(stacked.flatten())

        for active, expected in (
            ([1, 1, 1, 1], [[10, 20, 30], [40, 50, 60]]),
            ([1, 1, 0, 0], [[10, 20, 30], [0, 0, 0]]),
        ):
            parallel = SimpleNamespace(
                tp_group=SimpleNamespace(active_ranks_cpu=torch.tensor(active))
            )
            with (
                patch.object(
                    dp_attn, "exchange_sync_rows", side_effect=exchange
                ) as sync,
                patch.object(dp_attn, "get_parallel", return_value=parallel),
            ):
                info.all_gather(device="cpu", group=None)
            sync.assert_called_once()
            self.assertEqual(info.global_spec_graph_utilities, expected)

    def test_gather_clears_prior_selection_and_carries_curve(self):
        info = _info([10, 20, 30])
        info.finalize_local()
        batch = SimpleNamespace(
            spec_shared_graph_tokens=96,
            spec_shared_graph_step_seconds=0.05,
            spec_shared_graph_theta=100.0,
        )
        dp_attn._update_gather_batch(batch, info, require_mlp_tp_gather=True)
        self.assertEqual(batch.global_spec_graph_utilities, [[10, 20, 30]])
        self.assertIsNone(batch.spec_shared_graph_tokens)
        self.assertIsNone(batch.spec_shared_graph_step_seconds)
        self.assertIsNone(batch.spec_shared_graph_theta)
        batch.spec_shared_graph_tokens = 192
        dp_attn._update_gather_batch(
            batch, info, require_mlp_tp_gather=True, skip_global_metadata=True
        )
        self.assertIsNone(batch.global_spec_graph_utilities)
        self.assertIsNone(batch.spec_shared_graph_tokens)

    def test_adapter_finalizes_after_prebuilt_inner_metadata_is_ready(self):
        batch = SimpleNamespace(
            forward_mode=ForwardMode.PREBUILT, inner_idle_batch=None
        )
        finalized = []

        def gather(local_batch, **kwargs):
            self.assertEqual(kwargs["spec_graph_candidate_count"], 3)
            local_batch.inner_idle_batch = SimpleNamespace(
                global_spec_graph_utilities=[[10, 20, 30], [0, 0, 0]]
            )
            return local_batch

        def finalize(gathered):
            finalized.append(gathered)
            self.assertEqual(
                gathered.inner_idle_batch.global_spec_graph_utilities,
                [[10, 20, 30], [0, 0, 0]],
            )

        adapter = dp_attn.SchedulerDPAttnAdapter(
            model_runner=None,
            req_to_token_pool=None,
            token_to_kv_pool_allocator=None,
            tree_cache=None,
            offload_tags=set(),
            model_config=None,
            enable_overlap=True,
            spec_algorithm=SimpleNamespace(
                is_eagle=lambda: False,
                is_standalone=lambda: False,
                is_dspark=lambda: False,
            ),
            get_require_mlp_sync=lambda: True,
            spec_graph_candidate_count=3,
            finalize_spec_graph_budget=finalize,
        )
        with (
            patch.object(dp_attn, "prepare_mlp_sync_batch_raw", side_effect=gather),
            patch.object(dp_attn, "check_cuda_graph_backend", return_value=False),
            patch.object(dp_attn, "require_mlp_tp_gather", return_value=True),
            patch.object(
                dp_attn,
                "get_schedule",
                return_value=SimpleNamespace(disable_overlap_schedule=False),
            ),
            patch.object(
                dp_attn, "get_parallel", return_value=SimpleNamespace(dwdp_size=1)
            ),
        ):
            self.assertIs(adapter.maybe_prepare_mlp_sync_batch(batch), batch)
            self.assertIs(
                adapter.maybe_prepare_mlp_sync_batch(batch, need_sync=False), batch
            )
        self.assertEqual(finalized, [batch])


if __name__ == "__main__":
    unittest.main()
