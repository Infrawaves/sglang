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
import unittest
from types import ModuleType
from unittest.mock import patch

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
        for version in ("0.6.16", "0.6.18", "0.6.17+custom"):
            with self.subTest(version=version):
                self.version.return_value = version
                with self.assertRaises(RuntimeError):
                    create_cutedsl_mla_decode_with_splits(4)
        self.assertIs(
            self.module._get_split_kv_and_workspace_size, self.original_helper
        )

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


if __name__ == "__main__":
    unittest.main()
