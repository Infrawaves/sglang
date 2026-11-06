"""CPU coverage for shared graph selection against captured replay geometry."""

import unittest

from sglang.srt.speculative.dspark_components.dspark_graph_budget import (
    GraphBudgetShape,
    GraphBudgetShapes,
    full_coverage_graph_slots,
    resolve_ragged_graph_slots,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestGraphBudgetShapes(unittest.TestCase):
    def test_slot_resolver_uses_actual_capture_bucket(self):
        kwargs = dict(
            graph_num_tokens=240,
            max_live_bs=63,
            verify_width=6,
        )
        buckets = {240: (40, 64, 128, 240)}
        self.assertEqual(full_coverage_graph_slots(63, 240, 6), 93)
        self.assertEqual(
            resolve_ragged_graph_slots(buckets, reserve_padding_slot=True, **kwargs),
            128,
        )
        self.assertEqual(
            resolve_ragged_graph_slots(buckets, reserve_padding_slot=False, **kwargs),
            64,
        )

    def test_compact_candidate_aligns_with_replay_shape(self):
        shapes = GraphBudgetShapes(
            capture_num_tokens=[60, 240, 384],
            slots_by_tokens={
                60: (10, 16, 32, 60),
                240: (40, 64, 128, 240),
                384: (64, 128, 256),
            },
            verify_width=6,
            max_bs=256,
            reserve_padding_slot=True,
        )
        actual = shapes.for_batch(63)
        self.assertEqual(
            actual, (None, GraphBudgetShape(128, 240), GraphBudgetShape(128, 384))
        )
        self.assertIs(actual, shapes.for_batch(63))
        self.assertEqual(shapes.for_batch(10), (GraphBudgetShape(32, 60), None, None))

    def test_slot_pressure_only_admits_rounded_full_width_candidate(self):
        shapes = GraphBudgetShapes(
            capture_num_tokens=[240, 378, 384],
            slots_by_tokens={240: (40, 64), 378: (63, 64), 384: (64,)},
            verify_width=6,
            max_bs=64,
            reserve_padding_slot=True,
        )
        self.assertEqual(
            shapes.for_batch(63),
            (None, GraphBudgetShape(63, 378, uniform=True), None),
        )
        self.assertEqual(
            shapes.for_batch(64),
            (None, None, GraphBudgetShape(64, 384, uniform=True)),
        )

    def test_full_width_between_capture_batch_sizes(self):
        shapes = GraphBudgetShapes(
            capture_num_tokens=[240, 384],
            slots_by_tokens={240: (40, 64), 384: (64,)},
            verify_width=6,
            max_bs=64,
            reserve_padding_slot=True,
        )
        self.assertEqual(
            shapes.for_batch(63), (None, GraphBudgetShape(64, 384, uniform=True))
        )

    def test_uncaptured_and_out_of_range_candidates_are_unavailable(self):
        shapes = GraphBudgetShapes(
            capture_num_tokens=[60, 240],
            slots_by_tokens={60: (10, 16, 32, 60)},
            verify_width=6,
            max_bs=64,
            reserve_padding_slot=False,
        )
        self.assertEqual(shapes.for_batch(63), (None, None))
        self.assertEqual(shapes.for_batch(0), (None, None))
        self.assertEqual(shapes.for_batch(65), (None, None))
        self.assertEqual(shapes.for_batch(10), (GraphBudgetShape(10, 60), None))

    def test_rejects_incompatible_capture_keys(self):
        for grid, buckets in (
            ([60, 60], {60: (10,)}),
            ([60], {60: (8,)}),
            ([60], {60: (64,)}),
            ([60], {60: (16, 10)}),
            ([60], {60: (10, 10)}),
        ):
            with self.subTest(grid=grid, buckets=buckets):
                with self.assertRaises(ValueError):
                    GraphBudgetShapes(
                        capture_num_tokens=grid,
                        slots_by_tokens=buckets,
                        verify_width=6,
                        max_bs=64,
                        reserve_padding_slot=True,
                    )

    def test_copies_capture_map_before_caching(self):
        buckets = {60: [10, 16, 32, 60]}
        shapes = GraphBudgetShapes(
            capture_num_tokens=[60],
            slots_by_tokens=buckets,
            verify_width=6,
            max_bs=64,
            reserve_padding_slot=False,
        )
        buckets[60].clear()
        self.assertEqual(shapes.for_batch(10), (GraphBudgetShape(10, 60),))


if __name__ == "__main__":
    unittest.main()
