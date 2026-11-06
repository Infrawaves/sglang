"""Actual graph sweeps, shared-window timing and failure cleanup."""

import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from sglang.benchmark import dspark_sps_profiler as profiler
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def profile_geometry():
    return dict(
        capture_num_tokens=[3, 6, 9, 12, 18],
        slots_by_tokens={3: [1, 2, 3], 6: [2, 3, 6], 9: [3, 6], 12: [4, 6], 18: [6]},
        max_bs=6,
        verify_width=3,
        reserve_padding_slot=False,
        align=True,
        timing="scheduler_boundary_v1",
        profile_allocation="uniform_confidence_topk",
        dp_tier_gather_enabled=True,
    )


def graph_rows(*, cell=(2, 2, 3), n=13, increments=None):
    bs, slots, tokens = cell
    stamp = 100.0
    rows = []
    for ct in range(n):
        rows.append(
            profiler.SpsRow(ct, bs, tokens, 0.02, bs, slots, tokens, True, stamp)
        )
        stamp += increments[ct % len(increments)] if increments else 0.02
    return rows


def graph_summary(cell, seconds=0.02):
    bs, slots, tokens = cell
    return dict(
        batch_size_per_rank=bs,
        graph_slots=slots,
        graph_num_tokens=tokens,
        steps_per_sec=1 / seconds,
        timing="scheduler_boundary_v1",
        per_rank_window_step_seconds=[seconds],
        steady_forward_ct=[8, 136],
        num_steady_steps=128,
    )


class TestGraphProfiler(unittest.TestCase):
    def test_load_timeout_covers_long_round_and_must_be_positive(self):
        response = Mock(status_code=200)
        kwargs = dict(
            base_url="http://unused",
            num_requests=1,
            dp_size=1,
            input_len=4,
            max_new_tokens=100,
            temperature=1.0,
            vocab_size=10000,
            rng=random.Random(1),
        )
        with patch.object(profiler.requests, "post", return_value=response) as post:
            thread = profiler.start_load(**kwargs, request_timeout_seconds=660)
            thread.join(timeout=5)
        self.assertFalse(thread.is_alive())
        self.assertEqual(post.call_args.kwargs["timeout"], 660)
        with self.assertRaisesRegex(ValueError, "positive"):
            profiler.start_load(**kwargs, request_timeout_seconds=0)

    def test_boundary_only_records_are_opt_in_for_graph_profiling(self):
        row = profiler.msgspec.to_builtins(graph_rows(n=1)[0])
        row.pop("step_time")
        response = Mock()
        response.json.return_value = {
            "internal_states": [{"dspark_info_record": {"records": [row]}}]
        }
        with patch.object(profiler.requests, "get", return_value=response):
            self.assertEqual(profiler.fetch_rank_rows(base_url="http://unused"), [[]])
            fetched = profiler.fetch_rank_rows(
                base_url="http://unused", include_graph_boundaries=True
            )
        self.assertEqual(fetched[0][0].step_boundary_seconds, 100.0)

    def test_server_metadata_requires_matching_geometry_and_allocation(self):
        def server_info():
            return dict(
                speculative_algorithm="DSPARK",
                tokenizer_path="dummy",
                tp_size=4,
                internal_states=[
                    dict(
                        dp_size=2,
                        effective_max_running_requests_per_dp=6,
                        cuda_graph_config={"decode": {"bs": [1, 2, 6]}},
                        dspark_info_record=dict(
                            mode="compact",
                            components=["core", "step_cpu_time"],
                            simulate_acc_len=1.0,
                            verify_num_draft_tokens=3,
                            graph_profile=profile_geometry(),
                        ),
                    )
                    for _ in range(2)
                ],
            )

        response = Mock()
        info = server_info()
        response.json.return_value = info
        with patch.object(profiler.requests, "get", return_value=response):
            context = profiler.fetch_server_context(
                base_url="http://unused",
                local_tokenizer_path=None,
                allowed_modes=("compact",),
                graph_table=True,
            )
            self.assertEqual(context.dp_size, 2)
            info["internal_states"][1]["dspark_info_record"]["graph_profile"][
                "max_bs"
            ] = 5
            with self.assertRaisesRegex(ValueError, "disagree"):
                profiler.fetch_server_context(
                    base_url="http://unused",
                    local_tokenizer_path=None,
                    allowed_modes=("compact",),
                    graph_table=True,
                )
            info = server_info()
            response.json.return_value = info
            for state in info["internal_states"]:
                state["dspark_info_record"]["graph_profile"].pop("profile_allocation")
            with self.assertRaisesRegex(ValueError, "uniform_confidence_topk"):
                profiler.fetch_server_context(
                    base_url="http://unused",
                    local_tokenizer_path=None,
                    allowed_modes=("compact",),
                    graph_table=True,
                )

    def test_cli_all_passes_graph_mode_and_selected_tokens(self):
        with (
            patch.object(
                profiler.sys,
                "argv",
                [
                    "profiler",
                    "all",
                    "--graph-table",
                    "--graph-tokens",
                    "3",
                    "--batch-size",
                    "2",
                    "--no-plot",
                ],
            ),
            patch.object(profiler, "profile_all") as run,
        ):
            profiler.cli_main()
        self.assertTrue(run.call_args.kwargs["graph_table"])
        self.assertEqual(run.call_args.kwargs["graph_tokens"], [3])
        self.assertEqual(run.call_args.kwargs["batch_sizes"], [2])

    def test_sweep_endpoints_cover_each_shape_without_all_b_cartesian_product(self):
        profile = profile_geometry()
        cells = profiler.build_graph_sweep(graph_profile=profile, batch_sizes=[1, 6])
        shapes = profiler.graph_shapes(profile)
        all_cells = [
            (b, s.slots, s.graph_num_tokens)
            for b in range(1, 7)
            for s in shapes.for_batch(b)
            if s
        ]
        self.assertLess(len(cells), len(all_cells))
        table = profiler.build_graph_table_from_summaries(
            summaries=[graph_summary(cell) for cell in cells], graph_profile=profile
        )
        profiler.validate_graph_table_coverage(
            table=table, graph_profile=profile, max_batch_size=6
        )
        for b, s, g in all_cells:
            self.assertAlmostEqual(
                table.step_time(num_reqs=b, graph_slots=s, graph_num_tokens=g), 0.02
            )

    def test_smoke_subset_keeps_full_capture_grid(self):
        profile = profile_geometry()
        cells = profiler.build_graph_sweep(
            graph_profile=profile, batch_sizes=[2], graph_tokens=[3]
        )
        self.assertEqual(cells, [(2, 2, 3)])
        table = profiler.build_graph_table_from_summaries(
            summaries=[graph_summary(cells[0])], graph_profile=profile
        )
        self.assertEqual(table.capture_num_tokens, profile["capture_num_tokens"])
        profiler.validate_graph_table_coverage(
            table=table,
            graph_profile=profile,
            min_batch_size=2,
            max_batch_size=2,
            graph_tokens=[3],
        )
        with self.assertRaisesRegex(ValueError, "subset"):
            profiler.build_graph_sweep(
                graph_profile=profile, batch_sizes=[2], graph_tokens=[4]
            )

    def test_group_cost_uses_whole_common_window_not_per_step_maxima(self):
        rows = [
            graph_rows(increments=[0.01, 0.03]),
            graph_rows(increments=[0.03, 0.01]),
        ]
        outcome = profiler.postprocess_graph_round(
            rank_rows=rows,
            cell=(2, 2, 3),
            dp_size=2,
            min_steady_steps=4,
            load_info=profiler.LoadInfo(4, 100, 1.0, True),
        )
        self.assertEqual(outcome.steady_forward_ct, [8, 12])
        self.assertEqual(outcome.num_steady_steps, 4)
        self.assertAlmostEqual(1 / outcome.steps_per_sec, 0.02)

    def test_group_cost_uses_slowest_rank_window(self):
        outcome = profiler.postprocess_graph_round(
            rank_rows=[graph_rows(increments=[0.01]), graph_rows(increments=[0.03])],
            cell=(2, 2, 3),
            dp_size=2,
            min_steady_steps=4,
            load_info=profiler.LoadInfo(4, 100, 1.0, True),
        )
        self.assertAlmostEqual(1 / outcome.steps_per_sec, 0.03)

    def test_missing_step_and_unobserved_graph_are_rejected(self):
        good = graph_rows()
        with self.assertRaisesRegex(RuntimeError, "missing step"):
            profiler.graph_aligned_window(
                rank_rows=[good, good[:10] + good[11:]], cell=(2, 2, 3)
            )
        for field, value in (
            ("graph_slots", 3),
            ("can_run_cuda_graph", False),
            ("step_boundary_seconds", None),
        ):
            row = profiler.SpsRow(
                **{**profiler.msgspec.to_builtins(good[10]), field: value}
            )
            with self.subTest(field=field), self.assertRaises(RuntimeError):
                profiler.graph_aligned_window(
                    rank_rows=[good, good[:10] + [row] + good[11:]], cell=(2, 2, 3)
                )

    def test_window_requires_n_plus_one_boundaries(self):
        with self.assertRaisesRegex(RuntimeError, "3 steady intervals"):
            profiler.postprocess_graph_round(
                rank_rows=[graph_rows(n=12)],
                cell=(2, 2, 3),
                dp_size=1,
                min_steady_steps=4,
                load_info=profiler.LoadInfo(2, 100, 1.0, True),
            )

    def test_graph_fit_uses_median_time_and_rejects_legacy_rows(self):
        profile = profile_geometry()
        table = profiler.build_graph_table_from_summaries(
            summaries=[
                graph_summary((2, 2, 3), seconds) for seconds in (0.01, 0.02, 0.08)
            ],
            graph_profile=profile,
        )
        self.assertAlmostEqual(
            table.step_time(num_reqs=2, graph_slots=2, graph_num_tokens=3), 0.02
        )
        with self.assertRaisesRegex(ValueError, "actual graph shape"):
            profiler.build_graph_table_from_summaries(
                summaries=[{"batch_size_per_rank": 2, "steps_per_sec": 50}],
                graph_profile=profile,
            )

    def test_fit_roundtrip_and_missing_planned_cell(self):
        profile = profile_geometry()
        cells = profiler.build_graph_sweep(graph_profile=profile, batch_sizes=[1, 6])
        manifest = dict(
            graph_profile=profile,
            graph_min_batch_size=1,
            graph_max_batch_size=6,
            graph_cells=cells,
        )
        with tempfile.TemporaryDirectory() as tmp:
            out = str(Path(tmp) / "graph.json")
            paths = profiler.out_paths(out=out)
            paths["manifest"].write_text(json.dumps(manifest))
            paths["rounds"].write_text(
                "\n".join(json.dumps(graph_summary(cell)) for cell in cells)
            )
            profiler.fit_profile(
                out=out, max_batch_tokens=None, self_check=True, plot=False
            )
            table = profiler.load_sps_table_from_path(out)
            self.assertIsInstance(table, profiler.SpsGraphCostTable)
            manifest["repeats"] = 2
            paths["manifest"].write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "requested repeats"):
                profiler.fit_profile(
                    out=out, max_batch_tokens=None, self_check=True, plot=False
                )
            manifest["repeats"] = 1
            paths["manifest"].write_text(json.dumps(manifest))
            paths["rounds"].write_text(
                "\n".join(json.dumps(graph_summary(cell)) for cell in cells[1:])
            )
            with self.assertRaisesRegex(ValueError, "incomplete"):
                profiler.fit_profile(
                    out=out, max_batch_tokens=None, self_check=True, plot=False
                )

    def test_round_always_aborts_and_joins_on_bad_records(self):
        context = profiler.ServerContext(
            "http://unused",
            "dummy",
            1,
            1,
            3,
            1.0,
            6,
            float("inf"),
            float("inf"),
            profile_geometry(),
        )
        settings = profiler.RoundSettings(16, 1.0, 4, 0.0, 5.0)
        thread = Mock()
        with (
            patch.object(
                profiler, "should_skip_due_to_max_running_requests", return_value=False
            ),
            patch.object(
                profiler, "should_skip_due_to_token_capacity", return_value=False
            ),
            patch.object(profiler, "set_forced_graph_tokens") as pin,
            patch.object(profiler, "flush_cache"),
            patch.object(profiler, "fetch_rank_rows", side_effect=[[[]], [[]]]),
            patch.object(profiler, "start_load", return_value=thread),
            patch.object(profiler.time, "sleep"),
            patch.object(
                profiler,
                "graph_aligned_window",
                side_effect=RuntimeError("bad records"),
            ),
            patch.object(profiler, "abort_all_requests") as abort,
        ):
            with self.assertRaisesRegex(RuntimeError, "bad records"):
                profiler.run_graph_round(
                    context=context,
                    cell=(2, 2, 3),
                    settings=settings,
                    vocab_size=10000,
                    rng=random.Random(1),
                )
        pin.assert_called_once_with(
            base_url=context.base_url, graph_num_tokens=3, clear_records=True
        )
        abort.assert_called_once()
        thread.join.assert_called_once()

    def test_graph_round_collects_actual_shape_boundaries(self):
        context = profiler.ServerContext(
            "http://unused",
            "dummy",
            1,
            1,
            3,
            1.0,
            6,
            float("inf"),
            float("inf"),
            profile_geometry(),
        )
        settings = profiler.RoundSettings(16, 1.0, 4, 0.0, 5.0)
        thread = Mock()
        thread.is_alive.return_value = False
        rows = [graph_rows()]
        with (
            patch.object(
                profiler, "should_skip_due_to_max_running_requests", return_value=False
            ),
            patch.object(
                profiler, "should_skip_due_to_token_capacity", return_value=False
            ),
            patch.object(profiler, "set_forced_graph_tokens"),
            patch.object(profiler, "flush_cache"),
            patch.object(
                profiler, "fetch_rank_rows", side_effect=[[[]], rows, rows]
            ) as fetch,
            patch.object(profiler, "start_load", return_value=thread) as load,
            patch.object(profiler.time, "sleep"),
            patch.object(profiler, "abort_all_requests"),
        ):
            outcome = profiler.run_graph_round(
                context=context,
                cell=(2, 2, 3),
                settings=settings,
                vocab_size=10000,
                rng=random.Random(1),
            )
        self.assertAlmostEqual(outcome.steps_per_sec, 50.0)
        self.assertEqual(outcome.graph_slots, 2)
        self.assertEqual(
            load.call_args.kwargs["request_timeout_seconds"],
            settings.round_timeout_seconds + profiler.LOAD_JOIN_TIMEOUT_SECONDS,
        )
        self.assertTrue(
            all(
                call.kwargs["include_graph_boundaries"] for call in fetch.call_args_list
            )
        )

    def test_profile_clears_both_overrides_on_interruption(self):
        context = profiler.ServerContext(
            "http://unused",
            "dummy",
            1,
            1,
            3,
            1.0,
            6,
            float("inf"),
            float("inf"),
            profile_geometry(),
        )
        settings = profiler.RoundSettings(16, 1.0, 4, 0.0, 5.0)
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(profiler, "get_tokenizer", return_value=list(range(10000))),
            patch.object(profiler, "fetch_server_context", return_value=context),
            patch.object(profiler, "run_graph_round", side_effect=KeyboardInterrupt),
            patch.object(profiler, "set_forced_graph_tokens") as graph,
            patch.object(profiler, "set_forced_budget_frac") as frac,
            patch.object(profiler, "abort_all_requests") as abort,
        ):
            with self.assertRaises(KeyboardInterrupt):
                profiler.run_graph_profile(
                    base_url=context.base_url,
                    batch_sizes=[2],
                    settings=settings,
                    out=str(Path(tmp) / "out.json"),
                    repeats=1,
                    local_tokenizer_path=None,
                    graph_tokens=[3],
                )
            graph.assert_called_once_with(
                base_url=context.base_url, graph_num_tokens=None
            )
            self.assertEqual(frac.call_count, 2)
            self.assertEqual(abort.call_count, 2)


if __name__ == "__main__":
    unittest.main()
