"""Shared-graph utility, DP decisions, and host planner integration."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.model_executor.forward_batch_info import ForwardMode  # noqa: E402
from sglang.srt.speculative.dspark_components import (  # noqa: E402
    dspark_graph_search,
    dspark_planner,
    dspark_worker_v2,
)
from sglang.srt.speculative.dspark_components.dspark_graph_budget import (  # noqa: E402
    GraphBudgetShapes,
)
from sglang.srt.speculative.dspark_components.dspark_graph_search import (  # noqa: E402
    UTILITY_SCALE,
    SharedGraphBudgetSearch,
)
from sglang.srt.speculative.dspark_components.dspark_planner import (  # noqa: E402
    DSparkScheduleConfig,
    DSparkVerifyPlanner,
    HostConfidenceBudgetPlanner,
    compute_verify_token_budget,
)
from sglang.srt.speculative.dspark_components.dspark_sps import (  # noqa: E402
    SpsAdditiveCostTable,
    SpsCostTable,
    SpsGraphCostSample,
    SpsGraphCostTable,
)
from sglang.srt.speculative.dspark_components.dspark_worker_v2 import (  # noqa: E402
    DSparkWorkerV2,
)
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _search(*, costs=(1.0, 2.0, 3.0), max_bs=6, reserve=False):
    grid = [3, 6, 9]
    shapes = GraphBudgetShapes(
        capture_num_tokens=grid,
        slots_by_tokens={
            g: tuple(range((g + 2) // 3, min(g, max_bs) + 1)) for g in grid
        },
        verify_width=3,
        max_bs=max_bs,
        reserve_padding_slot=reserve,
    )
    table = SpsCostTable(
        sample_batch_tokens=[1] + grid,
        sample_steps_per_sec=[1 / costs[0]] + [1 / cost for cost in costs],
        max_batch_tokens=9,
    )
    return SharedGraphBudgetSearch(shapes=shapes, cost_table=table)


def _batch(bs=2, mode=ForwardMode.DECODE, **overrides):
    attrs = dict(
        spec_info=SimpleNamespace(verify_token_budget=None),
        forward_mode=mode,
        is_extend_in_batch=False,
        can_run_decode_cuda_graph=True,
        batch_size=lambda: bs,
        global_num_tokens=[2, 3, 0],
        global_spec_graph_utilities=None,
        spec_verify_budget_presync=True,
        spec_shared_graph_tokens=None,
        spec_shared_graph_step_seconds=None,
        spec_shared_graph_theta=None,
    )
    attrs.update(overrides)
    return SimpleNamespace(**attrs)


class TestSharedGraphSearch(unittest.TestCase):
    def test_global_optimum_can_be_smaller_than_maximum_local_decision(self):
        search = _search()
        survival = [torch.ones(2, 2), torch.zeros(3, 2)]
        local = [
            compute_verify_token_budget(
                history_survival_probs=row,
                sps_table=search.cost_table,
                cfg=DSparkScheduleConfig(gamma=2),
            )
            for row in survival
        ]
        max_local_tokens = max(
            row.shape[0] + plan.budget for row, plan in zip(survival, local)
        )
        self.assertEqual(max_local_tokens, 5)  # The old path rounds this to G=6.
        utilities = [search.utilities(row, survival_eps=1e-6) for row in survival]
        decision = search.select(batch_sizes=[2, 3], utilities=utilities)
        self.assertEqual(decision.shape.graph_num_tokens, 3)
        self.assertEqual(decision.expected_tokens, 6.0)
        self.assertEqual(sum(row[1] for row in utilities) / UTILITY_SCALE, 9.0)
        self.assertGreater(decision.expected_tokens / decision.step_seconds, 9.0 / 2.0)

    def test_curve_includes_align_gain_and_local_full_width_cap(self):
        search = _search()
        curve = search.utilities(
            torch.tensor([[0.9, 0.1], [0.8, 0.2]]), survival_eps=1e-6
        )
        self.assertAlmostEqual(curve[0] / UTILITY_SCALE, 2.9, places=5)
        self.assertAlmostEqual(curve[1] / UTILITY_SCALE, 4.0, places=5)
        self.assertEqual(curve[1], curve[2])
        self.assertEqual(curve[-1], curve[1])

    def test_idle_rank_has_zero_utility_and_does_not_change_selection(self):
        search = _search()
        active = search.utilities(torch.ones(2, 2), survival_eps=1e-6)
        idle = search.utilities(torch.empty(0, 2), survival_eps=1e-6)
        self.assertEqual(idle, [0] * search.payload_size)
        self.assertEqual(
            search.select(batch_sizes=[2], utilities=[active]),
            search.select(batch_sizes=[0, 2, 0], utilities=[idle, active, idle]),
        )

    def test_equal_scores_choose_smallest_graph_on_every_rank(self):
        search = _search(costs=(1.0, 2.0, 3.0))
        curve = search.utilities(torch.ones(3, 2), survival_eps=1e-6)
        decisions = [
            search.select(batch_sizes=[3, 3], utilities=[curve, curve])
            for _ in range(8)
        ]
        self.assertTrue(all(decision == decisions[0] for decision in decisions))
        self.assertEqual(decisions[0].shape.graph_num_tokens, 3)

    def test_full_width_fallback_counts_candidates_below_epsilon(self):
        search = _search(costs=(1.0, 1.17, 3.0), max_bs=3, reserve=True)
        curve = search.utilities(
            torch.tensor([[0.6, 0.05], [0.4, 0.02]]), survival_eps=0.1
        )
        self.assertAlmostEqual(curve[1] / UTILITY_SCALE, 3.0, places=5)
        self.assertAlmostEqual(curve[-1] / UTILITY_SCALE, 3.07, places=5)
        decision = search.select(batch_sizes=[2], utilities=[curve])
        self.assertTrue(decision.shape.uniform)
        self.assertEqual(decision.shape.graph_num_tokens, 6)
        self.assertAlmostEqual(decision.expected_tokens, 3.07, places=5)

    def test_invalid_peer_disables_shared_selection_for_the_whole_group(self):
        search = _search()
        valid = search.utilities(torch.ones(2, 2), survival_eps=1e-6)
        for invalid in (
            None,
            [],
            [valid],
            [valid, [-1] * search.payload_size],
            [valid, [0]],
        ):
            with self.subTest(invalid=invalid):
                self.assertIsNone(search.select(batch_sizes=[2, 2], utilities=invalid))
        for sizes in ([], [0, 0], [-1, 2], [7, 7]):
            with self.subTest(sizes=sizes):
                self.assertIsNone(
                    search.select(batch_sizes=sizes, utilities=[valid] * len(sizes))
                )

    def test_invalid_survival_produces_invalid_payload(self):
        search = _search()
        for value in (float("nan"), float("inf"), -0.1, 1.1):
            with self.subTest(value=value):
                self.assertEqual(
                    search.utilities(torch.tensor([[value, 0.5]]), survival_eps=1e-6),
                    [-1] * search.payload_size,
                )

    def test_cost_lookup_is_cached_by_global_maximum_batch(self):
        search = _search()
        curve = search.utilities(torch.ones(3, 2), survival_eps=1e-6)
        with patch.object(
            dspark_graph_search,
            "graph_step_time",
            wraps=dspark_graph_search.graph_step_time,
        ) as lookup:
            search.select(batch_sizes=[3, 2], utilities=[curve, curve])
            first_count = lookup.call_count
            self.assertEqual(first_count, 3)
            search.select(batch_sizes=[2, 3], utilities=[curve, curve])
            self.assertEqual(lookup.call_count, first_count)
            search.select(batch_sizes=[2, 2], utilities=[curve, curve])
            self.assertEqual(lookup.call_count, first_count + 2)


class TestSharedGraphStartup(unittest.TestCase):
    def _table(self, *, verify_width=3, grid=None):
        return SpsGraphCostTable(
            verify_width=verify_width,
            capture_num_tokens=[3, 6, 9] if grid is None else grid,
            samples=[
                SpsGraphCostSample(2, 2, 3, 0.1),
                SpsGraphCostSample(2, 2, 6, 0.2),
            ],
        )

    def _planner(self, *, table=None):
        planner = DSparkVerifyPlanner.__new__(DSparkVerifyPlanner)
        planner._budget_planner = SimpleNamespace(
            sps_table=self._table() if table is None else table
        )
        planner._shared_graph_search = None
        planner._dp_tier_gather_enabled = True
        planner._align_verify_tokens_to_graph_tier = True
        planner._is_verify_all = False
        planner._schedule_cfg = DSparkScheduleConfig(gamma=2)
        planner.verify_num_draft_tokens = 3
        planner.model_runner = SimpleNamespace(
            decode_cuda_graph_runner=SimpleNamespace(
                ragged_verify_mode=True,
                capture_num_tokens=[3, 6, 9],
                _ragged_slots_by_tokens={3: (1, 2, 3), 6: (2, 3, 6), 9: (3, 6)},
                captured_req_width=3,
                max_bs=6,
                reserve_ragged_padding_slot=False,
            )
        )
        return planner

    def _initialize(self, planner):
        with patch.object(
            dspark_planner, "get_parallel", return_value=SimpleNamespace(tp_rank=0)
        ):
            return planner.initialize_shared_graph_search()

    def test_graph_table_initializes_and_selects_measured_shape(self):
        planner = self._planner()
        self.assertEqual(self._initialize(planner), 4)
        search = planner._shared_graph_search
        self.assertIs(search.cost_table, planner._budget_planner.sps_table)
        curve = search.utilities(torch.ones(2, 2), survival_eps=1e-6)
        decision = search.select(batch_sizes=[2, 2], utilities=[curve, curve])
        self.assertEqual(decision.shape.graph_num_tokens, 3)
        self.assertEqual(decision.step_seconds, 0.1)

    def test_graph_table_rejects_unavailable_scheduler_prerequisites(self):
        for attr, value, message in (
            ("_dp_tier_gather_enabled", False, "compact DP tier gathering"),
            (
                "_align_verify_tokens_to_graph_tier",
                False,
                "--speculative-dspark-align-verify-tokens-to-graph-tier",
            ),
            ("_is_verify_all", True, "budget scheduling"),
            (
                "_schedule_cfg",
                DSparkScheduleConfig(gamma=2, min_verify_len=0),
                "min_verify_len=1",
            ),
            (
                "_schedule_cfg",
                DSparkScheduleConfig(gamma=2, max_verify_len=2),
                "max_verify_len=verify_num_draft_tokens",
            ),
        ):
            with self.subTest(attr=attr, value=value):
                planner = self._planner()
                setattr(planner, attr, value)
                with self.assertRaisesRegex(ValueError, message):
                    self._initialize(planner)
                self.assertIsNone(planner._shared_graph_search)

    def test_graph_table_rejects_missing_ragged_capture(self):
        for case in ("no_runner", "not_ragged", "no_slot_buckets"):
            with self.subTest(case=case):
                planner = self._planner()
                runner = planner.model_runner.decode_cuda_graph_runner
                if case == "no_runner":
                    planner.model_runner.decode_cuda_graph_runner = None
                elif case == "not_ragged":
                    runner.ragged_verify_mode = False
                else:
                    runner._ragged_slots_by_tokens = {}
                with self.assertRaisesRegex(
                    ValueError, "captured ragged decode CUDA graphs"
                ):
                    self._initialize(planner)
                self.assertIsNone(planner._shared_graph_search)

    def test_graph_table_still_rejects_width_and_capture_grid_mismatch(self):
        for table, message in (
            (self._table(verify_width=4), "verify_width"),
            (self._table(grid=[3, 6, 12]), "capture_num_tokens"),
        ):
            with self.subTest(message=message):
                planner = self._planner(table=table)
                with self.assertRaisesRegex(ValueError, message):
                    self._initialize(planner)
                self.assertIsNone(planner._shared_graph_search)

    def test_legacy_tables_keep_local_budget_fallback(self):
        tables = (
            _search().cost_table,
            SpsAdditiveCostTable(0.1, [1, 6], [0.0, 0.1], [1, 9], [0.0, 0.1]),
        )
        for table in tables:
            for case in ("no_dp_gather", "no_align", "no_runner"):
                with self.subTest(table=type(table).__name__, case=case):
                    planner = self._planner(table=table)
                    if case == "no_dp_gather":
                        planner._dp_tier_gather_enabled = False
                    elif case == "no_align":
                        planner._align_verify_tokens_to_graph_tier = False
                    else:
                        planner.model_runner.decode_cuda_graph_runner = None
                    self.assertEqual(self._initialize(planner), 0)
                    self.assertIsNone(planner._shared_graph_search)
                    decision = compute_verify_token_budget(
                        history_survival_probs=torch.ones(2, 2),
                        sps_table=table,
                        cfg=planner._schedule_cfg,
                    )
                    self.assertIsNotNone(decision.budget)

    def test_no_budget_planner_and_verify_all_keep_search_disabled(self):
        for has_budget_planner in (False, True):
            with self.subTest(has_budget_planner=has_budget_planner):
                planner = self._planner(table=_search().cost_table)
                planner._is_verify_all = True
                if not has_budget_planner:
                    planner._budget_planner = None
                self.assertEqual(self._initialize(planner), 0)
                self.assertIsNone(planner._shared_graph_search)

    def test_valid_graph_table_preserves_transient_step_fallback(self):
        planner = self._planner()
        self._initialize(planner)
        search = planner._shared_graph_search
        curve = search.utilities(torch.ones(2, 2), survival_eps=1e-6)
        for attrs in (
            {"is_extend_in_batch": True},
            {"can_run_decode_cuda_graph": False},
            {"global_spec_graph_utilities": None},
        ):
            with self.subTest(attrs=attrs):
                batch = _batch(
                    global_num_tokens=[2, 2],
                    global_spec_graph_utilities=[curve, curve],
                )
                planner.finalize_shared_graph_budget(batch)
                self.assertEqual(batch.spec_shared_graph_tokens, 3)
                for attr, value in attrs.items():
                    setattr(batch, attr, value)
                planner.finalize_shared_graph_budget(batch)
                self.assertIsNone(batch.spec_shared_graph_tokens)
                self.assertIsNone(batch.spec_shared_graph_theta)
                self.assertIs(planner._shared_graph_search, search)


class TestSharedGraphHostPlanner(unittest.TestCase):
    def _host(self):
        search = _search()
        with dspark_planner.envs.SGLANG_DSPARK_CONFIDENCE_RELAY_LAG_STEPS.override(2):
            host = HostConfidenceBudgetPlanner(
                sps_table=search.cost_table,
                cfg=DSparkScheduleConfig(gamma=2),
                model_runner=SimpleNamespace(),
                relay_lag_steps=2,
            )
        return host, search

    def test_uses_existing_lagged_survival_and_generation_fallback(self):
        host, search = self._host()
        confidence = torch.tensor([[0.5, 0.5], [0.1, 0.1]])
        result = host.compute_budget(
            confidence=confidence,
            generation=torch.tensor([2, 1]),
            current_generation=torch.tensor([2, 2]),
            req_pool_indices_cpu=torch.arange(2),
            graph_search=search,
        )
        self.assertIsNone(result)
        self.assertIsNone(host.last_decision)
        expected = search.utilities(
            torch.tensor([[0.5, 0.25], [1.0, 1.0]]), survival_eps=1e-6
        )
        self.assertEqual(host.last_graph_utilities, expected)
        host.note_non_decode_step()
        self.assertIsNone(host.last_graph_utilities)

    def test_forced_profile_budget_preserves_override_and_suppresses_curve(self):
        host, search = self._host()
        host.forced_budget_frac = 0.5
        host.last_graph_utilities = [123] * search.payload_size
        result = host.compute_budget(
            confidence=torch.ones(2, 2),
            generation=torch.ones(2, dtype=torch.int64),
            current_generation=torch.ones(2, dtype=torch.int64),
            req_pool_indices_cpu=torch.arange(2),
            graph_search=search,
        )
        self.assertEqual(result, 2)
        self.assertEqual(host.last_decision.budget, 2)
        self.assertIsNone(host.last_graph_utilities)

    def test_attention_tp_peer_skips_host_sort_and_shared_search(self):
        planner = DSparkVerifyPlanner.__new__(DSparkVerifyPlanner)
        planner._budget_tp_group = None
        planner._shared_graph_search = _search()
        planner._budget_planner = Mock()
        with patch.object(
            dspark_planner, "get_parallel", return_value=SimpleNamespace(attn_tp_rank=1)
        ):
            result = planner._budget_from_resolved(
                resolved=Mock(), req_pool_indices_cpu=torch.arange(2)
            )
        self.assertIsNone(result)
        planner._budget_planner.compute_budget.assert_not_called()
        planner._budget_planner.note_non_decode_step.assert_called_once()

    def test_attention_tp_peer_skips_confidence_relay_cpu_gather(self):
        planner = DSparkVerifyPlanner.__new__(DSparkVerifyPlanner)
        planner._budget_tp_group = None
        planner._shared_graph_search = _search()
        planner._budget_planner = Mock()
        planner._is_verify_all = False
        planner.verify_num_draft_tokens = 3
        planner._schedule_cfg = DSparkScheduleConfig(gamma=2)
        batch = _batch(req_pool_indices_cpu=torch.arange(2))
        future_map = Mock()
        with patch.object(
            dspark_planner, "get_parallel", return_value=SimpleNamespace(attn_tp_rank=1)
        ):
            planner._compute_local_verify_budget(batch, future_map)
        future_map.resolve_confidence_cpu.assert_not_called()
        planner._budget_planner.compute_budget.assert_not_called()
        self.assertIsNone(batch.spec_info.verify_token_budget)
        self.assertIsNone(batch.spec_graph_utilities)
        self.assertEqual(batch.spec_verify_tier_num_tokens, -1)

    def test_finalization_agrees_for_busy_idle_and_prebuilt_ranks(self):
        planner = DSparkVerifyPlanner.__new__(DSparkVerifyPlanner)
        search = planner._shared_graph_search = _search()
        utilities = [
            search.utilities(torch.ones(2, 2), survival_eps=1e-6),
            search.utilities(torch.zeros(3, 2), survival_eps=1e-6),
            [0] * search.payload_size,
        ]
        batches = [_batch(mode=ForwardMode.DECODE), _batch(bs=0, mode=ForwardMode.IDLE)]
        prebuilt = _batch(
            mode=ForwardMode.PREBUILT,
            inner_idle_batch=_batch(bs=0, mode=ForwardMode.IDLE),
        )
        batches.append(prebuilt)
        for batch in batches:
            inner = (
                batch.inner_idle_batch if batch.forward_mode.is_prebuilt() else batch
            )
            inner.global_spec_graph_utilities = utilities
            planner.finalize_shared_graph_budget(batch)
            self.assertEqual(inner.spec_shared_graph_tokens, 3)
            self.assertEqual(inner.spec_shared_graph_theta, 6.0)

    def test_finalize_clears_stale_decision_for_unsupported_steps(self):
        planner = DSparkVerifyPlanner.__new__(DSparkVerifyPlanner)
        planner._shared_graph_search = _search()
        for attrs in (
            {"is_extend_in_batch": True},
            {"can_run_decode_cuda_graph": False},
            {"forward_mode": ForwardMode.EXTEND},
            {"global_num_tokens": None},
            {"global_spec_graph_utilities": None},
        ):
            with self.subTest(attrs=attrs):
                batch = _batch(
                    spec_shared_graph_tokens=9, spec_shared_graph_theta=99.0, **attrs
                )
                planner.finalize_shared_graph_budget(batch)
                self.assertIsNone(batch.spec_shared_graph_tokens)
                self.assertIsNone(batch.spec_shared_graph_theta)

    def test_shared_graph_installs_aligned_local_budget_and_prediction(self):
        planner = DSparkVerifyPlanner.__new__(DSparkVerifyPlanner)
        planner._budget_planner = Mock()
        planner._dp_tier_gather_enabled = True
        planner.verify_num_draft_tokens = 3
        planner._schedule_cfg = DSparkScheduleConfig(gamma=2)
        for bs, expected in ((1, 2), (2, 4), (3, 3)):
            with self.subTest(bs=bs):
                batch = _batch(
                    bs=bs,
                    spec_shared_graph_tokens=6,
                    spec_shared_graph_step_seconds=0.05,
                    spec_shared_graph_theta=180.0,
                )
                planner.prepare_verify_budget(batch, Mock())
                self.assertEqual(batch.spec_info.verify_token_budget, expected)
                self.assertEqual(batch.spec_verify_tier_num_tokens, bs + expected)
                self.assertEqual(
                    planner._budget_planner.last_decision.graph_num_tokens, 6
                )
                self.assertEqual(
                    planner._budget_planner.last_decision.predicted_step_seconds, 0.05
                )
                self.assertFalse(batch.spec_verify_budget_presync)


class TestSharedGraphWorker(unittest.TestCase):
    def _worker(self):
        worker = DSparkWorkerV2.__new__(DSparkWorkerV2)
        worker._dp_synced_verify = True
        worker._verify_planner = SimpleNamespace(
            dp_tier_gather_enabled=True, is_compact_mode=True
        )
        worker.device = torch.device("cpu")
        worker.verify_num_draft_tokens = 3
        worker.model_runner = Mock()
        return worker

    def test_worker_honors_shared_graph_over_legacy_local_max(self):
        worker = self._worker()
        batch = _batch(
            spec_shared_graph_tokens=6,
            global_spec_verify_tier_num_tokens=[9, 9, 0],
        )
        self.assertEqual(worker._dp_verify_tier_num_tokens(batch), 6)
        batch.spec_shared_graph_tokens = None
        self.assertEqual(worker._dp_verify_tier_num_tokens(batch), 9)

    def test_shared_graph_obeys_existing_replay_gates(self):
        for attr, value in (
            ("global_num_tokens", None),
            ("is_extend_in_batch", True),
            ("can_run_decode_cuda_graph", False),
        ):
            with self.subTest(attr=attr):
                batch = _batch(spec_shared_graph_tokens=6, **{attr: value})
                self.assertIsNone(self._worker()._dp_verify_tier_num_tokens(batch))
        worker = self._worker()
        worker._dp_synced_verify = False
        self.assertIsNone(
            worker._dp_verify_tier_num_tokens(_batch(spec_shared_graph_tokens=6))
        )
        worker._dp_synced_verify = True
        worker._verify_planner.dp_tier_gather_enabled = False
        self.assertIsNone(
            worker._dp_verify_tier_num_tokens(_batch(spec_shared_graph_tokens=6))
        )

    def test_idle_layout_receives_shared_graph_and_global_batch(self):
        worker = self._worker()
        batch = _batch(bs=0, mode=ForwardMode.IDLE, spec_shared_graph_tokens=6)
        with patch.object(
            dspark_worker_v2, "idle_ragged_layout", return_value="layout"
        ) as build:
            self.assertEqual(worker._idle_verify_ragged_layout(batch), "layout")
        self.assertEqual(build.call_args.kwargs["dp_tier_num_tokens"], 6)
        self.assertEqual(build.call_args.kwargs["tier_num_reqs"], 3)


if __name__ == "__main__":
    unittest.main()
