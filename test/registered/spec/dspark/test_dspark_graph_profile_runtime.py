"""Exact graph profiling controls and scheduler-boundary records."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.kernels.ops.speculative.dspark.dspark_schedule import (  # noqa: E402
    schedule_verify_lens_topk,
)
from sglang.srt.environ import envs  # noqa: E402
from sglang.srt.managers.scheduler import Scheduler  # noqa: E402
from sglang.srt.model_executor.forward_batch_info import ForwardMode  # noqa: E402
from sglang.srt.speculative.dspark_components import dspark_planner  # noqa: E402
from sglang.srt.speculative.dspark_components.dspark_observability import (  # noqa: E402
    DecodeStepObservation,
    DsparkInfoDumper,
    DsparkStepObservers,
)
from sglang.srt.speculative.dspark_components.dspark_planner import (  # noqa: E402
    DSparkScheduleConfig,
    DSparkVerifyPlanner,
)
from sglang.srt.speculative.ragged_verify import RaggedVerifyMode  # noqa: E402
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _profile_planner():
    planner = DSparkVerifyPlanner.__new__(DSparkVerifyPlanner)
    planner.verify_num_draft_tokens = 6
    planner.device = torch.device("cpu")
    planner._schedule_cfg = DSparkScheduleConfig(gamma=5)
    planner._ragged_verify_mode = RaggedVerifyMode.COMPACT
    planner._is_verify_all = True  # No SPS table loaded on the profile server.
    planner._dynamic_graph_tier = False
    planner._align_verify_tokens_to_graph_tier = True
    planner._dp_tier_gather_enabled = True
    planner._budget_planner = Mock()
    planner._shared_graph_search = None
    planner._forced_graph_tokens = None
    planner._profile_graph_shapes = None
    planner.model_runner = SimpleNamespace(
        decode_cuda_graph_runner=SimpleNamespace(
            capture_num_tokens=[6, 12, 24],
            _ragged_slots_by_tokens={6: (1, 2, 4, 6), 12: (2, 4, 8), 24: (4, 8)},
            max_bs=8,
            captured_req_width=6,
            reserve_ragged_padding_slot=True,
            ragged_verify_mode=True,
        )
    )
    return planner


def _profile_batch(bs=2, mode=ForwardMode.DECODE):
    return SimpleNamespace(
        forward_mode=mode,
        batch_size=lambda: bs,
        spec_info=SimpleNamespace(verify_token_budget=None),
        is_extend_in_batch=False,
        can_run_decode_cuda_graph=True,
        global_num_tokens=[2, 1, 0],
        spec_shared_graph_tokens=None,
        spec_shared_graph_step_seconds=None,
        spec_shared_graph_theta=None,
        spec_verify_budget_presync=False,
    )


class TestForcedGraphProfiling(unittest.TestCase):
    def setUp(self):
        for override in (
            envs.SGLANG_DSPARK_ENABLE_SPS_RECORD.override(True),
            envs.SGLANG_SIMULATE_ACC_LEN.override(1.0),
        ):
            override.__enter__()
            self.addCleanup(override.__exit__, None, None, None)

    def test_control_requires_profile_flags_align_and_captured_graph(self):
        planner = _profile_planner()
        for invalid in (True, "12", 0, 7, 12.0):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                planner.validate_forced_graph_tokens(invalid)
        for env in (envs.SGLANG_DSPARK_ENABLE_SPS_RECORD, envs.SGLANG_SIMULATE_ACC_LEN):
            with env.override(0), self.assertRaises(ValueError):
                planner.validate_forced_graph_tokens(12)
        planner._align_verify_tokens_to_graph_tier = False
        with self.assertRaises(ValueError):
            planner.validate_forced_graph_tokens(12)
        planner.set_forced_graph_tokens(None)

    def test_pin_bypasses_missing_table_and_confidence_relay(self):
        planner, batch, future = _profile_planner(), _profile_batch(), Mock()
        planner.set_forced_graph_tokens(12)
        planner.prepare_verify_budget_pre_sync(batch, future)
        future.resolve_confidence_cpu.assert_not_called()
        self.assertTrue(batch.spec_verify_budget_presync)
        self.assertEqual(batch.spec_info.verify_token_budget, 10)
        self.assertEqual(batch.spec_verify_tier_num_tokens, 12)
        planner.finalize_shared_graph_budget(batch)
        planner.prepare_verify_budget(batch, future)
        self.assertEqual(batch.spec_shared_graph_tokens, 12)
        self.assertEqual(batch.spec_info.verify_token_budget, 10)

    def test_busy_idle_and_prebuilt_use_same_pinned_graph(self):
        planner = _profile_planner()
        planner.set_forced_graph_tokens(12)
        for mode in (ForwardMode.DECODE, ForwardMode.IDLE, ForwardMode.PREBUILT):
            batch = _profile_batch(mode=mode)
            if mode is ForwardMode.PREBUILT:
                batch.inner_idle_batch = _profile_batch(mode=ForwardMode.IDLE)
            planner.finalize_shared_graph_budget(batch)
            target = batch.inner_idle_batch if mode is ForwardMode.PREBUILT else batch
            self.assertEqual(target.spec_shared_graph_tokens, 12)

    def test_ramp_batch_cannot_replay_pinned_graph_is_unselected(self):
        planner, batch = _profile_planner(), _profile_batch()
        planner.set_forced_graph_tokens(24)
        batch.spec_shared_graph_tokens = 12
        planner.finalize_shared_graph_budget(batch)
        self.assertIsNone(batch.spec_shared_graph_tokens)

    def test_uninitialized_table_does_not_force_uniform_profile_layout(self):
        planner = _profile_planner()
        planner.set_forced_graph_tokens(6)
        planner._schedule_verify_lens = Mock(return_value=torch.tensor([3, 3]))
        with patch.object(
            dspark_planner.RaggedVerifyLayout, "from_verify_lens_device"
        ) as pack:
            planner.schedule_layout(
                req_pool_indices=torch.arange(2),
                prefix_lens=torch.ones(2),
                device=torch.device("cpu"),
                confidence=torch.ones(2, 5),
                budget=4,
                global_num_reqs=2,
                dp_tier_num_tokens=6,
            )
        planner._schedule_verify_lens.assert_called_once()
        self.assertEqual(pack.call_args.kwargs["graph_num_tokens"], 6)

    def test_metadata_matches_capture_geometry_and_boundary_convention(self):
        metadata = _profile_planner().graph_profile_metadata()
        self.assertEqual(metadata["capture_num_tokens"], [6, 12, 24])
        self.assertEqual(metadata["slots_by_tokens"][12], (2, 4, 8))
        self.assertEqual(metadata["timing"], "scheduler_boundary_v1")
        self.assertTrue(metadata["align"])
        self.assertEqual(metadata["profile_allocation"], "uniform_confidence_topk")

    def test_cached_profile_confidence_fills_zeros_and_nans_then_restores(self):
        planner = _profile_planner()
        planner._tp_sync = Mock()
        planner.set_forced_graph_tokens(6)
        buffer = planner._profile_confidence
        execute = Mock(side_effect=schedule_verify_lens_topk)
        with (
            patch.object(
                dspark_planner,
                "ScheduleVerifyLensTopk",
                SimpleNamespace(execute=execute),
                create=True,
            ),
            patch.object(dspark_planner, "resolve_level", return_value=0, create=True),
            patch.object(
                dspark_planner,
                "InvariantCheckLevel",
                SimpleNamespace(WARN=1),
                create=True,
            ),
            patch.object(
                dspark_planner,
                "SpecTpSyncSite",
                SimpleNamespace(DSPARK_PLAN=0),
                create=True,
            ),
        ):
            for value in (0.0, float("nan")):
                lens = planner._schedule_verify_lens(
                    req_pool_indices=torch.arange(2),
                    prefix_lens=torch.ones(2),
                    device=torch.device("cpu"),
                    confidence=torch.full((2, 5), value),
                    budget=4,
                )
                self.assertEqual(int(lens.sum()), 6)
                self.assertEqual(
                    execute.call_args.kwargs["confidence"].data_ptr(), buffer.data_ptr()
                )
            planner.set_forced_graph_tokens(12)
            self.assertIs(planner._profile_confidence, buffer)
            planner.set_forced_graph_tokens(None)
            zeros = torch.zeros(2, 5)
            lens = planner._schedule_verify_lens(
                req_pool_indices=torch.arange(2),
                prefix_lens=torch.ones(2),
                device=torch.device("cpu"),
                confidence=zeros,
                budget=4,
            )
            self.assertIs(execute.call_args.kwargs["confidence"], zeros)
            self.assertEqual(int(lens.sum()), 2)


class TestGraphProfileRecords(unittest.TestCase):
    def _observation(self, forward_ct):
        return DecodeStepObservation(
            forward_ct=forward_ct,
            bs=2,
            mode="compact",
            budget=10,
            lag_steps=2,
            num_verify_tokens=12,
            verify_tokens_local=12,
            verify_tokens_dp_synced=12,
            verify_tokens_graph_key=12,
            predicted_step_ms=None,
            predicted_theta=None,
            verify_lens=None,
            confidence=None,
            req_pool_indices=None,
            prefix_lens=None,
            draft_tokens=None,
            bonus_tokens=None,
            correct_len=None,
            cap_trim_lens=None,
            commit_lens=None,
            rids=None,
            graph_max_bs=2,
            graph_slots=4,
            graph_num_tokens=12,
            can_run_cuda_graph=True,
        )

    def test_boundary_timestamp_and_shape_stay_with_their_round(self):
        clock = Mock(side_effect=[100.0, 100.1, 100.3])
        dumper = DsparkInfoDumper(
            components={"core", "step_cpu_time"},
            gamma=5,
            verify_num_draft_tokens=6,
            attn_tp_rank=0,
            device=torch.device("cpu"),
            mode_value="compact",
            clock=clock,
        )
        for ct in (10, 11, 12):
            dumper.observe_decode_step(self._observation(ct))
        rows = dumper.dump()["records"]
        self.assertEqual([r["forward_ct"] for r in rows], [10, 11, 12])
        self.assertEqual(
            [r["step_boundary_seconds"] for r in rows], [100.0, 100.1, 100.3]
        )
        self.assertTrue(
            all(
                (r["graph_max_bs"], r["graph_slots"], r["graph_num_tokens"])
                == (2, 4, 12)
                for r in rows
            )
        )
        self.assertTrue(all(r["can_run_cuda_graph"] for r in rows))

    def test_eager_forward_cannot_claim_a_measured_graph_shape(self):
        planner = SimpleNamespace(
            model_runner=SimpleNamespace(decode_cuda_graph_runner=Mock()),
            take_budget_decision=Mock(return_value=None),
            mode_value="compact",
            lag_steps=2,
        )
        runner = planner.model_runner.decode_cuda_graph_runner
        runner._ragged_replay_shape.return_value = (4, 12)
        observer = DsparkStepObservers.__new__(DsparkStepObservers)
        observer._planner = planner
        observer._block_accept_recorder = None
        observer._info_dumper = Mock(enabled=True)
        for graph in (False, True):
            observer.observe_verify_step(
                forward_ct=1,
                reqs=[],
                bs=2,
                proposal_folded=True,
                verify_ids_2d=torch.ones(2, 6),
                target_logits=None,
                layout=SimpleNamespace(graph_num_tokens=12, verify_lens=None),
                confidence=None,
                prefix_lens=None,
                draft_tokens=None,
                draft_block=None,
                sampling_info=None,
                correct_len=None,
                cap_trim_lens=None,
                bonus=None,
                commit_lens=None,
                verify_token_budget=10,
                req_pool_indices=None,
                verify_tier_num_tokens=12,
                dp_tier_num_tokens=12,
                graph_max_bs=2,
                can_run_cuda_graph=graph,
            )
            obs = observer._info_dumper.observe_decode_step.call_args.args[0]
            self.assertEqual(obs.can_run_cuda_graph, graph)
            self.assertEqual(obs.graph_slots, 4 if graph else None)
            self.assertEqual(obs.graph_num_tokens, 12 if graph else None)
        runner._ragged_replay_shape.assert_called_once()


class TestGraphProfileControl(unittest.TestCase):
    def _scheduler(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.spec_algorithm = SimpleNamespace(
            is_dspark=lambda: True, is_none=lambda: False
        )
        scheduler.draft_worker = Mock()
        scheduler.metrics_reporter = SimpleNamespace(
            spec_total_num_forward_ct=0, spec_total_num_accept_tokens=0
        )
        return scheduler

    def test_rejected_graph_control_preserves_prior_state_and_records(self):
        scheduler = self._scheduler()
        scheduler.draft_worker.validate_dspark_forced_graph_tokens.side_effect = (
            ValueError("uncaptured G")
        )
        result = scheduler.set_internal_state(
            SimpleNamespace(
                server_args={
                    "dspark_clear_info_records": True,
                    "dspark_force_graph_tokens": 7,
                }
            )
        )
        self.assertFalse(result.updated)
        scheduler.draft_worker.set_dspark_forced_graph_tokens.assert_not_called()
        scheduler.draft_worker.clear_info_records.assert_not_called()

    def test_valid_control_and_cleanup_route_to_worker(self):
        scheduler = self._scheduler()
        result = scheduler.set_internal_state(
            SimpleNamespace(
                server_args={
                    "dspark_force_graph_tokens": 12,
                    "dspark_force_budget_frac": None,
                    "dspark_clear_info_records": True,
                }
            )
        )
        self.assertTrue(result.updated)
        scheduler.draft_worker.set_dspark_forced_graph_tokens.assert_called_with(12)
        scheduler.draft_worker.set_dspark_forced_budget_frac.assert_called_with(None)
        scheduler.draft_worker.clear_info_records.assert_called_once()
        result = scheduler.set_internal_state(
            SimpleNamespace(server_args={"dspark_force_graph_tokens": None})
        )
        self.assertTrue(result.updated)
        scheduler.draft_worker.set_dspark_forced_graph_tokens.assert_called_with(None)


if __name__ == "__main__":
    unittest.main()
