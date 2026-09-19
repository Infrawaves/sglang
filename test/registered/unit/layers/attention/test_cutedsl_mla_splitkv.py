# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""CPU coverage for CuTeDSL MLA split planning and isolated wrapper overrides."""

import importlib.metadata
import importlib.util
import io
import json
import math
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import torch

from sglang.srt.layers.attention.cutedsl_mla_splitkv import (
    create_cutedsl_mla_decode_with_splits,
    plan_cutedsl_mla_splits,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

_FLASHINFER_MODULE = "flashinfer.cute_dsl.attention.monolithic.mla_decode"


class TestCuTeDSLMLASplitPlan(CustomTestCase):
    def test_decode_workspace_includes_padded_heads_and_lse(self):
        # K3 attention TP4 uses H24, but partial output storage pads to 128 heads.
        for splits, workspace in ((1, 0), (4, 134479872), (8, 268959744)):
            with self.subTest(splits=splits):
                self.assertEqual(
                    plan_cutedsl_mla_splits(128, 1, 24, 512, splits),
                    (splits, workspace),
                )

    def test_query_head_tiles_increase_workspace(self):
        # Eight queries times 24 heads require two 128-row tiles.
        self.assertEqual(
            plan_cutedsl_mla_splits(128, 8, 24, 512, 4),
            (4, 268959744),
        )

    def test_short_context_removes_empty_splits(self):
        cases = ((1, 8, 1), (128, 8, 1), (129, 8, 2), (257, 8, 3), (513, 4, 3))
        for max_seq_len, requested, expected in cases:
            with self.subTest(max_seq_len=max_seq_len, requested=requested):
                splits, workspace = plan_cutedsl_mla_splits(
                    128, 1, 24, 512, requested, max_seq_len=max_seq_len
                )
                self.assertEqual(splits, expected)
                if expected == 1:
                    self.assertEqual(workspace, 0)
                else:
                    self.assertEqual(workspace, 33619968 * expected)

    def test_long_context_preserves_requested_splits(self):
        self.assertEqual(
            plan_cutedsl_mla_splits(128, 1, 24, 512, 8, max_seq_len=1048576),
            (8, 268959744),
        )

    def test_invalid_split_count_is_rejected(self):
        for value in (0, -1, 33, True, False, 4.0, "4", None):
            with self.subTest(num_splits=value), self.assertRaises(ValueError):
                plan_cutedsl_mla_splits(128, 1, 24, 512, value)

    def test_invalid_geometry_is_rejected(self):
        valid = dict(batch_size=128, q_len=1, num_heads=24, kv_lora_rank=512)
        for field in valid:
            for value in (0, -1):
                with self.subTest(field=field, value=value):
                    geometry = dict(valid, **{field: value})
                    with self.assertRaises(ValueError):
                        plan_cutedsl_mla_splits(**geometry, num_splits=4)
        for value in (0, -1):
            with self.subTest(max_seq_len=value), self.assertRaises(ValueError):
                plan_cutedsl_mla_splits(**valid, num_splits=4, max_seq_len=value)


class TestCuTeDSLMLAIsolatedWrapper(CustomTestCase):
    def setUp(self):
        super().setUp()
        self.module = ModuleType(_FLASHINFER_MODULE)
        # A real Python wrapper is needed: the override must resolve the helper
        # through its own globals, just as FlashInfer's wrapper does.
        exec(
            """
def _get_split_kv_and_workspace_size(
    B, q_len, H, kv_lora_rank, max_active_blocks,
    max_seq_len=None, occupancy_q_tiles=False,
):
    return 1, 0

def cute_dsl_mla_decode(
    B=128, q_len=1, H=24, kv_lora_rank=512,
    *, max_seq_len=1048576, marker="preserved",
):
    plan = _get_split_kv_and_workspace_size(
        B, q_len, H, kv_lora_rank, 152, max_seq_len, False
    )
    return plan, marker
""",
            self.module.__dict__,
        )
        self.original = self.module.cute_dsl_mla_decode
        self.original_helper = self.module._get_split_kv_and_workspace_size
        modules_patch = patch.dict("sys.modules", {_FLASHINFER_MODULE: self.module})
        modules_patch.start()
        self.addCleanup(modules_patch.stop)
        version_patch = patch(
            "sglang.srt.layers.attention.cutedsl_mla_splitkv.version",
            return_value="0.6.17",
        )
        self.version = version_patch.start()
        self.addCleanup(version_patch.stop)

    def test_two_wrappers_do_not_modify_each_other_or_flashinfer(self):
        four = create_cutedsl_mla_decode_with_splits(4)
        eight = create_cutedsl_mla_decode_with_splits(8)
        self.assertEqual(four(), ((4, 134479872), "preserved"))
        self.assertEqual(eight(), ((8, 268959744), "preserved"))
        self.assertEqual(four(), ((4, 134479872), "preserved"))
        self.assertEqual(self.original(), ((1, 0), "preserved"))
        self.assertIs(self.module.cute_dsl_mla_decode, self.original)
        self.assertIs(
            self.module._get_split_kv_and_workspace_size, self.original_helper
        )
        self.assertIsNot(four.__globals__, self.original.__globals__)
        self.assertIsNot(four.__globals__, eight.__globals__)
        self.version.assert_called_with("flashinfer-python")

    def test_wrapper_preserves_defaults_and_forwards_geometry(self):
        decode = create_cutedsl_mla_decode_with_splits(8)
        self.assertEqual(decode.__defaults__, self.original.__defaults__)
        self.assertEqual(decode.__kwdefaults__, self.original.__kwdefaults__)
        self.assertEqual(
            decode(128, 8, 24, 512, max_seq_len=129, marker="explicit"),
            ((2, 134479872), "explicit"),
        )

    def test_other_flashinfer_versions_are_rejected(self):
        for version in ("0.6.16", "0.6.19", "0.6.17+custom", "0.6.18+custom"):
            with self.subTest(version=version):
                self.version.return_value = version
                with self.assertRaises(RuntimeError):
                    create_cutedsl_mla_decode_with_splits(4)
        self.assertIs(
            self.module._get_split_kv_and_workspace_size, self.original_helper
        )

    def test_supported_flashinfer_versions(self):
        for version in ("0.6.17", "0.6.18"):
            with self.subTest(version=version):
                self.version.return_value = version
                decode = create_cutedsl_mla_decode_with_splits(4)
                self.assertEqual(decode(), ((4, 134479872), "preserved"))
                self.assertEqual(self.original(), ((1, 0), "preserved"))

    def test_missing_flashinfer_distribution_is_rejected(self):
        self.version.side_effect = importlib.metadata.PackageNotFoundError(
            "flashinfer-python"
        )
        with self.assertRaises(importlib.metadata.PackageNotFoundError):
            create_cutedsl_mla_decode_with_splits(4)

    def test_factory_rejects_invalid_splits(self):
        for value in (0, -1, 33, True, 4.0, "4"):
            with self.subTest(num_splits=value), self.assertRaises(ValueError):
                create_cutedsl_mla_decode_with_splits(value)


class TestCuTeDSLMLABenchmark(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        path = (
            Path(__file__).resolve().parents[4] / "manual/bench_cutedsl_mla_splitkv.py"
        )
        spec = importlib.util.spec_from_file_location("bench_cutedsl_mla_splitkv", path)
        cls.bench = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.bench)

    def test_batch_sweep_uses_each_batch_and_preserves_actual_lengths(self):
        args = self.bench.parse_args(
            [
                "--batch-sizes",
                "1",
                "4",
                "128",
                "4",
                "--distributions",
                "uniform64k",
                "mixedmean64k",
                "--max-seq-len",
                "1048576",
            ]
        )
        cases = self.bench.build_cases(args)
        self.assertEqual(len(cases), 6)
        for case, name, lengths in cases:
            with self.subTest(batch=case.batch_size, distribution=name):
                self.assertEqual(case.active_batch_size, case.batch_size)
                self.assertEqual(len(lengths), case.batch_size)
                self.assertEqual(sum(lengths), 65536 * case.batch_size)
                self.assertEqual(case.max_seq_len, 1048576)
                self.assertLess(max(lengths), case.max_seq_len)
                if name == "uniform64k":
                    self.assertEqual(set(lengths), {65536})
        self.assertEqual(
            [case.batch_size for case, _, _ in cases], [1, 1, 4, 4, 128, 128]
        )

    def test_custom_lengths_keep_active_count_separate_from_graph_batch(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "lengths.json"
            lengths = [1, 128, 65536, 1048576]
            path.write_text(json.dumps(lengths))
            args = self.bench.parse_args(
                [
                    "--batch-sizes",
                    "8",
                    "128",
                    "--active-batch-size",
                    "4",
                    "--lengths-json",
                    str(path),
                    "--length-scale",
                    "16",
                    "--max-seq-len",
                    "1048576",
                ]
            )
            cases = self.bench.build_cases(args)
        self.assertEqual(len(cases), 2)
        for case, name, actual in cases:
            with self.subTest(batch=case.batch_size):
                self.assertEqual(name, "custom")
                self.assertEqual(case.active_batch_size, 4)
                self.assertEqual(actual, lengths)
                self.assertLess(len(actual), case.batch_size)

    def test_single_batch_cli_still_supports_true_one_million_context(self):
        args = self.bench.parse_args(
            [
                "--batch-size",
                "1",
                "--distributions",
                "uniform64k",
                "--length-scale",
                "16",
                "--max-seq-len",
                "1048576",
            ]
        )
        [(case, name, lengths)] = self.bench.build_cases(args)
        self.assertEqual((case.batch_size, case.active_batch_size), (1, 1))
        self.assertEqual(name, "uniform64k")
        self.assertEqual(lengths, [1048576])
        self.assertEqual(args.splits, [1, 4, 8, 16, 32])

    def test_invalid_cli_geometry_and_conflicting_batch_flags_are_rejected(self):
        invalid = [
            ["--batch-size", "1", "--batch-sizes", "1", "2"],
            ["--batch-sizes", "0", "8"],
            ["--batch-sizes", "4", "8", "--active-batch-size", "5"],
            ["--active-batch-size", "0"],
            ["--max-seq-len", "0"],
            ["--length-scale", "nan"],
            ["--splits", "1", "33"],
            ["--atol", "-0.1"],
            ["--rtol", "inf"],
        ]
        for argv in invalid:
            with self.subTest(argv=argv), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    self.bench.parse_args(argv)
                self.assertEqual(error.exception.code, 2)

    def test_invalid_custom_lengths_and_too_small_bound_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "lengths.json"
            for payload in ([1], [1, 0], [1, True], [1, 2.0], {"lengths": [1, 2]}):
                with self.subTest(payload=payload):
                    path.write_text(json.dumps(payload))
                    args = self.bench.parse_args(
                        ["--batch-size", "2", "--lengths-json", str(path)]
                    )
                    with self.assertRaises(ValueError):
                        self.bench.build_cases(args)
            path.write_text("[1, 129]")
            args = self.bench.parse_args(
                [
                    "--batch-size",
                    "2",
                    "--lengths-json",
                    str(path),
                    "--max-seq-len",
                    "128",
                ]
            )
            with self.assertRaisesRegex(ValueError, "below an actual KV length"):
                self.bench.build_cases(args)
        args = self.bench.parse_args(["--batch-size", "1", "--max-seq-len", "65535"])
        with self.assertRaisesRegex(ValueError, "below an actual KV length"):
            self.bench.build_cases(args)

    def test_metrics_apply_absolute_plus_relative_tolerance_and_handle_zero(self):
        actual = torch.tensor([0.125, 2.25, 100.5])
        expected = torch.tensor([0.0, 2.0, 100.0])
        result = self.bench.compare_outputs(actual, expected, atol=0.125, rtol=0.005)
        self.assertFalse(result["passed"])
        self.assertEqual(result["mismatch_count"], 1)
        self.assertAlmostEqual(result["mismatch_fraction"], 1 / 3)
        self.assertEqual(result["max_abs_error"], 0.5)
        self.assertAlmostEqual(result["mean_abs_error"], 7 / 24)
        self.assertAlmostEqual(result["rmse"], math.sqrt(7 / 64))
        self.assertEqual(result["max_rel_error"], 0.125)
        self.assertEqual(result["zero_ref_mismatch_count"], 0)
        zeros = self.bench.compare_outputs(
            torch.tensor([0.0, 0.25]), torch.zeros(2), atol=0.125, rtol=0.01
        )
        self.assertEqual(zeros["zero_ref_mismatch_count"], 1)
        self.assertIsNone(zeros["max_rel_error"])
        self.assertFalse(zeros["passed"])
        self.assertTrue(
            self.bench.compare_outputs(expected, expected, 0.0, 0.0)["passed"]
        )

    def test_nonfinite_pairs_fail_and_all_metrics_remain_valid_json(self):
        for actual, expected, finite_pairs, mismatch_count in (
            ([float("inf")], [float("inf")], 0, 1),
            ([float("nan"), 0.0], [0.0, float("nan")], 0, 2),
            ([float("inf"), 1.0], [float("-inf"), 1.0], 1, 1),
        ):
            with self.subTest(actual=actual, expected=expected):
                result = self.bench.compare_outputs(
                    torch.tensor(actual), torch.tensor(expected), 0.002, 0.01
                )
                self.assertFalse(result["passed"])
                self.assertEqual(result["finite_pairs"], finite_pairs)
                self.assertEqual(result["mismatch_count"], mismatch_count)
                self.assertEqual(result["nonfinite_actual"], 1)
                self.assertEqual(result["nonfinite_reference"], 1)
                if not finite_pairs:
                    self.assertIsNone(result["max_abs_error"])
                    self.assertIsNone(result["worst_abs_index"])
                    self.assertIsNone(result["rmse"])
                json.dumps(result, allow_nan=False)

    def test_worst_error_index_identifies_the_correct_batch_and_head(self):
        expected = torch.zeros((3, 1, 2, 4))
        actual = expected.clone()
        actual[0, 0, 0, 1] = 0.25
        actual[2, 0, 1, 3] = -3.0
        result = self.bench.compare_outputs(actual, expected, 0.002, 0.01)
        self.assertEqual(result["worst_abs_index"], [2, 0, 1, 3])
        self.assertEqual(result["worst_abs_actual"], -3.0)
        self.assertEqual(result["worst_abs_reference"], 0.0)
        self.assertEqual(result["max_abs_error"], 3.0)
        self.assertEqual(result["mismatch_count"], 2)
        self.assertEqual(result["numel"], 24)

    def test_unstable_stock_reference_cannot_report_a_candidate_pass(self):
        for passed, reference_valid, expected in (
            (True, True, "pass"),
            (False, True, "fail"),
            (True, False, "invalid_reference"),
            (False, False, "invalid_reference"),
        ):
            with self.subTest(passed=passed, reference_valid=reference_valid):
                checks = {"initial": {"passed": True}, "replay": {"passed": passed}}
                self.assertEqual(
                    self.bench.result_status(checks, reference_valid), expected
                )


if __name__ == "__main__":
    unittest.main()
