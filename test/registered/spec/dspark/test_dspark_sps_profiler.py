import random
import unittest
from unittest.mock import MagicMock, patch

from sglang.benchmark import dspark_sps_profiler
from sglang.benchmark.dspark_sps_profiler import (
    LoadInfo,
    ServerContext,
    SpsRow,
    build_request_count_sweep,
    build_table_from_summaries,
    count_aligned_steps,
    postprocess_round,
    resolve_cuda_graph_max_bs,
    round_summary_dict,
    start_load,
    validate_sweep_against_server,
)
from sglang.srt.disaggregation.utils import FAKE_BOOTSTRAP_HOST
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


def make_load_info() -> LoadInfo:
    return LoadInfo(
        num_requests=4, max_new_tokens=1200, wall_seconds=1.0, reached_target=True
    )


def make_rows(
    *,
    num_rows: int = 30,
    num_running_reqs: int = 4,
    num_verify_tokens: int = 32,
    step_time: float = 0.01,
    first_forward_ct: int = 0,
) -> list[SpsRow]:
    return [
        SpsRow(
            forward_ct=first_forward_ct + index,
            num_running_reqs=num_running_reqs,
            num_verify_tokens=num_verify_tokens,
            step_time=step_time,
        )
        for index in range(num_rows)
    ]


def make_context(**overrides) -> ServerContext:
    values = dict(
        base_url="http://localhost:30000",
        tokenizer_path="dummy",
        tp_size=4,
        dp_size=1,
        verify_num_draft_tokens=8,
        simulate_acc_len=1.0,
        cuda_graph_max_bs=128,
        skip_max_running_requests_threshold=float("inf"),
        skip_token_capacity_threshold=float("inf"),
    )
    values.update(overrides)
    return ServerContext(**values)


class TestPostprocessRound(CustomTestCase):
    def test_single_rank_round_builds_probe_from_median_step_time(self):
        outcome = postprocess_round(
            rank_rows=[make_rows(step_time=0.01)],
            batch_size_per_rank=4,
            dp_size=1,
            verify_num_draft_tokens=8,
            min_steady_steps=16,
            load_info=make_load_info(),
        )
        self.assertEqual(outcome.batch_tokens, 32)
        self.assertAlmostEqual(outcome.steps_per_sec, 100.0)
        self.assertEqual(outcome.match_fraction, 1.0)

    def test_round_warmup_steps_are_dropped_from_timing(self):
        slow_head = make_rows(num_rows=8, step_time=0.5, first_forward_ct=0)
        steady_tail = make_rows(num_rows=20, step_time=0.01, first_forward_ct=8)
        outcome = postprocess_round(
            rank_rows=[slow_head + steady_tail],
            batch_size_per_rank=4,
            dp_size=1,
            verify_num_draft_tokens=8,
            min_steady_steps=16,
            load_info=make_load_info(),
        )
        self.assertAlmostEqual(outcome.steps_per_sec, 100.0)

    def test_off_target_batch_rows_are_filtered_out(self):
        ramp = make_rows(num_rows=10, num_running_reqs=2, num_verify_tokens=16)
        steady = make_rows(num_rows=30, first_forward_ct=10, step_time=0.02)
        outcome = postprocess_round(
            rank_rows=[ramp + steady],
            batch_size_per_rank=4,
            dp_size=1,
            verify_num_draft_tokens=8,
            min_steady_steps=16,
            load_info=make_load_info(),
        )
        self.assertAlmostEqual(outcome.steps_per_sec, 50.0)
        self.assertAlmostEqual(outcome.match_fraction, 1.0)

    def test_mid_round_instability_raises(self):
        head = make_rows(num_rows=15)
        gap = make_rows(
            num_rows=40, num_running_reqs=3, num_verify_tokens=24, first_forward_ct=15
        )
        tail = make_rows(num_rows=15, first_forward_ct=55)
        with self.assertRaisesRegex(RuntimeError, "unstable mid-round"):
            postprocess_round(
                rank_rows=[head + gap + tail],
                batch_size_per_rank=4,
                dp_size=1,
                verify_num_draft_tokens=8,
                min_steady_steps=16,
                load_info=make_load_info(),
            )

    def test_round_that_never_stabilizes_raises(self):
        rows = make_rows(num_rows=50, num_running_reqs=3, num_verify_tokens=24)
        rows += make_rows(num_rows=2, first_forward_ct=50)
        with self.assertRaisesRegex(RuntimeError, "never stabilized"):
            postprocess_round(
                rank_rows=[rows],
                batch_size_per_rank=4,
                dp_size=1,
                verify_num_draft_tokens=8,
                min_steady_steps=16,
                load_info=make_load_info(),
            )


class TestPostprocessRoundCrossRank(CustomTestCase):
    def test_two_uniform_ranks_average_their_step_times(self):
        outcome = postprocess_round(
            rank_rows=[make_rows(step_time=0.01), make_rows(step_time=0.03)],
            batch_size_per_rank=4,
            dp_size=2,
            verify_num_draft_tokens=8,
            min_steady_steps=16,
            load_info=make_load_info(),
        )
        self.assertEqual(outcome.batch_size_per_rank, 4)
        self.assertEqual(outcome.batch_tokens, 32)
        self.assertAlmostEqual(outcome.steps_per_sec, 50.0)
        self.assertEqual(len(outcome.per_rank_median_step_time), 2)
        self.assertAlmostEqual(outcome.per_rank_median_step_time[0], 0.01)
        self.assertAlmostEqual(outcome.per_rank_median_step_time[1], 0.03)

    def test_rank_with_no_new_records_raises(self):
        with self.assertRaisesRegex(RuntimeError, "no new decode-step records"):
            postprocess_round(
                rank_rows=[make_rows(), []],
                batch_size_per_rank=4,
                dp_size=2,
                verify_num_draft_tokens=8,
                min_steady_steps=16,
                load_info=make_load_info(),
            )

    def test_disjoint_forward_ct_ranges_raise(self):
        with self.assertRaisesRegex(RuntimeError, "no common forward_ct"):
            postprocess_round(
                rank_rows=[
                    make_rows(first_forward_ct=0),
                    make_rows(first_forward_ct=1000),
                ],
                batch_size_per_rank=4,
                dp_size=2,
                verify_num_draft_tokens=8,
                min_steady_steps=16,
                load_info=make_load_info(),
            )

    def test_rank_below_expected_verify_tokens_raises(self):
        # A rank reporting fewer verify tokens than bs_per_rank * K is not
        # running the uniform static verify; above-expected counts are
        # tolerated (the recorded count is the replayed graph tier).
        with self.assertRaisesRegex(RuntimeError, "num_verify_tokens"):
            postprocess_round(
                rank_rows=[make_rows(), make_rows(num_verify_tokens=24)],
                batch_size_per_rank=4,
                dp_size=2,
                verify_num_draft_tokens=8,
                min_steady_steps=16,
                load_info=make_load_info(),
            )

    def test_rank_count_mismatch_raises(self):
        with self.assertRaisesRegex(RuntimeError, "DP ranks"):
            postprocess_round(
                rank_rows=[make_rows()],
                batch_size_per_rank=4,
                dp_size=2,
                verify_num_draft_tokens=8,
                min_steady_steps=16,
                load_info=make_load_info(),
            )


class TestTableAssembly(CustomTestCase):
    def test_repeats_take_the_median_per_batch_tokens(self):
        rounds = [
            postprocess_round(
                rank_rows=[make_rows(step_time=step_time)],
                batch_size_per_rank=4,
                dp_size=1,
                verify_num_draft_tokens=8,
                min_steady_steps=16,
                load_info=make_load_info(),
            )
            for step_time in (0.01, 0.02, 0.04)
        ]
        table = build_table_from_summaries(
            summaries=[
                round_summary_dict(outcome=outcome, repeat=repeat)
                for repeat, outcome in enumerate(rounds)
            ],
            max_batch_tokens=None,
            offdiag=False,
        )
        self.assertEqual(table.sample_batch_tokens, [32])
        self.assertAlmostEqual(table.sample_steps_per_sec[0], 50.0)


class TestSweepHelpers(CustomTestCase):
    def test_request_count_sweep_tapers_and_hits_the_max(self):
        sweep = build_request_count_sweep(100)
        self.assertEqual(sweep[:4], [1, 2, 4, 8])
        self.assertEqual(sweep[-1], 100)
        self.assertIn(64, sweep)

    def test_sweep_beyond_captured_cuda_graphs_raises(self):
        with self.assertRaisesRegex(ValueError, "cuda graphs"):
            validate_sweep_against_server(
                context=make_context(cuda_graph_max_bs=64),
                batch_sizes=[8, 128],
            )

    def test_sweep_within_captured_cuda_graphs_passes(self):
        validate_sweep_against_server(
            context=make_context(cuda_graph_max_bs=64, dp_size=2),
            batch_sizes=[8, 64],
        )

    def test_resolve_cuda_graph_max_bs_prefers_captured_list(self):
        internal_state = {
            "cuda_graph_config": {"decode": {"bs": [1, 2, 160], "max_bs": 128}}
        }
        self.assertEqual(resolve_cuda_graph_max_bs(internal_state=internal_state), 160)

    def test_resolve_cuda_graph_max_bs_handles_missing_config(self):
        self.assertIsNone(resolve_cuda_graph_max_bs(internal_state={}))


class TestCountAlignedSteps(CustomTestCase):
    def test_off_target_steps_are_not_counted(self):
        rows = make_rows(num_rows=10, num_running_reqs=3)
        self.assertEqual(
            count_aligned_steps(rank_rows=[rows], batch_size_per_rank=4), 0
        )


def run_load_with_response(
    response: MagicMock, *, num_requests: int = 2, dp_size: int = 1
) -> MagicMock:
    with patch.object(
        dspark_sps_profiler.requests, "post", return_value=response
    ) as post:
        thread = start_load(
            base_url="http://localhost:30000",
            num_requests=num_requests,
            dp_size=dp_size,
            input_len=4,
            max_new_tokens=8,
            temperature=1.0,
            vocab_size=4096,
            rng=random.Random(0),
        )
        thread.join(timeout=10)
    return post


class TestStartLoad(CustomTestCase):
    def test_load_payload_targets_the_fake_transfer_path(self):
        """A PD decode server built the KV receiver address from bootstrap_host
        and crashed its scheduler on None; the fake host routes the load through
        the fake transfer path instead."""
        post = run_load_with_response(MagicMock(status_code=200, text="[]"))
        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["bootstrap_host"], FAKE_BOOTSTRAP_HOST)

    def test_rejected_load_logs_status_and_server_message(self):
        """A 400 from /generate used to be dropped silently, so the round waited
        out its timeout with no hint of the cause."""
        response = MagicMock(
            status_code=400,
            text='{"error":{"message":"exceeding --max-batch-outputs=1024."}}',
        )
        with self.assertLogs(dspark_sps_profiler.logger, level="ERROR") as logs:
            run_load_with_response(response)
        self.assertIn("HTTP 400", logs.output[0])
        self.assertIn("--max-batch-outputs=1024", logs.output[0])

    def test_load_pins_an_equal_share_to_every_dp_rank(self):
        """Round-robin dispatch of one batch was shifted by a /health probe
        landing mid-batch, leaving ranks at B-1 / B+1 so the round never
        aligned; each rank now gets its own pinned share."""
        post = run_load_with_response(
            MagicMock(status_code=200, text="[]"), num_requests=8, dp_size=4
        )
        payloads = [call.kwargs["json"] for call in post.call_args_list]
        self.assertEqual(
            sorted(payload["routed_dp_rank"] for payload in payloads), [0, 1, 2, 3]
        )
        self.assertTrue(all(len(payload["input_ids"]) == 2 for payload in payloads))
        prompts = [tuple(ids) for payload in payloads for ids in payload["input_ids"]]
        self.assertEqual(len(set(prompts)), 8)

    def test_single_dp_load_leaves_routing_to_the_server(self):
        post = run_load_with_response(MagicMock(status_code=200, text="[]"))
        self.assertEqual(post.call_count, 1)
        self.assertNotIn("routed_dp_rank", post.call_args.kwargs["json"])


class TestMinSteadySteps(CustomTestCase):
    def test_min_steady_steps_rejects_thin_probes(self):
        with self.assertRaisesRegex(RuntimeError, "never stabilized"):
            postprocess_round(
                rank_rows=[make_rows(num_rows=20)],
                batch_size_per_rank=4,
                dp_size=1,
                verify_num_draft_tokens=8,
                min_steady_steps=32,
                load_info=make_load_info(),
            )


if __name__ == "__main__":
    unittest.main()
