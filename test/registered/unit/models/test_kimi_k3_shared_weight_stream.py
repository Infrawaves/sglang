"""Shared-expert weight streaming must stage exactly the TP1 shared MLP weights."""

import unittest

import torch

from sglang.srt.layers.communication.k3_shared_weight_stream import slot_copy_plan
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _rank_pool(gate_ups, downs, rank, world):
    """Rank r's weight-stream pool: per layer its MergedColumnParallel gate_up
    shard [gate_r; up_r], then its RowParallel down shard."""
    parts = []
    for gate_up, down in zip(gate_ups, downs):
        full = down.shape[1]
        cols = slice(rank * full // world, (rank + 1) * full // world)
        parts += [gate_up[:full][cols], gate_up[full:][cols], down[:, cols]]
    return torch.cat([p.reshape(-1) for p in parts])


class TestSharedWeightStreamPlan(CustomTestCase):
    def test_staging_matches_tp1_weights(self):
        # Pitched pull copies of every rank's shards must rebuild the TP1
        # gate_up ([gate; up] rows) and down (columns) of the requested slot.
        hidden, full, layers = 8, 12, 3
        for world in (2, 4):
            gate_ups = [torch.randn(2 * full, hidden) for _ in range(layers)]
            downs = [torch.randn(hidden, full) for _ in range(layers)]
            pools = [_rank_pool(gate_ups, downs, r, world) for r in range(world)]
            for slot in range(layers):
                with self.subTest(world=world, slot=slot):
                    staging = {
                        "gate_up": torch.full((2 * full, hidden), float("nan")),
                        "down": torch.full((hidden, full), float("nan")),
                    }
                    plan = slot_copy_plan(
                        slot=slot, world_size=world, shard=full // world, hidden=hidden
                    )
                    # Each op is one CPU launch on a launch-bound path.
                    self.assertEqual(len(plan), 2 * world)
                    for op in plan:
                        dst = staging[op.dst].view(-1)
                        for row in range(op.height):
                            d = op.dst_offset + row * op.dst_pitch
                            s = op.src_offset + row * op.src_pitch
                            dst[d : d + op.width] = pools[op.src_rank][s : s + op.width]
                    torch.testing.assert_close(staging["gate_up"], gate_ups[slot])
                    torch.testing.assert_close(staging["down"], downs[slot])


if __name__ == "__main__":
    unittest.main()
