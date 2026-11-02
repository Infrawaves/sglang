"""Unit tests for the DeepInfer EPLB policy port."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest

import torch

from sglang.srt.eplb.eplb_algorithms import (
    EplbAlgorithm,
    compute_algorithm,
    deepinfer,
    deepseek_vec,
    rebalance_experts,
)
from sglang.srt.eplb.eplb_manager import predicted_rank_balancedness
from sglang.test.test_utils import CustomTestCase


def _skewed_counts(num_steps, num_layers, num_logical, seed=0):
    g = torch.Generator().manual_seed(seed)
    base = torch.distributions.Dirichlet(torch.full((num_logical,), 0.3))
    torch.manual_seed(seed)
    popularity = base.sample((num_layers,))
    bursts = base.sample((num_steps, num_layers))
    rate = 32768 * (0.6 * popularity + 0.4 * bursts)
    return torch.poisson(rate, generator=g)


class TestDeepInferEPLB(CustomTestCase):
    def _check_layout(self, p2l, l2p, count, num_logical, num_physical):
        num_layers = p2l.shape[0]
        self.assertEqual(tuple(p2l.shape), (num_layers, num_physical))
        self.assertTrue((count.sum(-1) == num_physical).all())
        for layer in range(num_layers):
            self.assertTrue(
                torch.equal(
                    torch.bincount(p2l[layer].long(), minlength=num_logical),
                    count[layer].long(),
                )
            )
            for expert in range(num_logical):
                slots = l2p[layer, expert]
                slots = slots[slots >= 0].long()
                self.assertEqual(len(slots), int(count[layer, expert]))
                self.assertTrue((p2l[layer, slots] == expert).all())

    def test_layout_is_consistent(self):
        counts = _skewed_counts(64, 3, 64)
        p2l, l2p, count = deepinfer.rebalance_experts(counts, 72, 9)
        self._check_layout(p2l, l2p, count, 64, 72)

    def test_no_redundancy_still_rebalances_placement(self):
        counts = _skewed_counts(64, 2, 64)
        p2l, l2p, count = deepinfer.rebalance_experts(counts, 64, 8)
        self._check_layout(p2l, l2p, count, 64, 64)
        self.assertTrue((count == 1).all())
        trivial = torch.arange(64).expand(2, -1)
        self.assertGreater(
            predicted_rank_balancedness(counts, p2l, 8),
            predicted_rank_balancedness(counts, trivial, 8),
        )

    def test_accepts_aggregated_counts(self):
        counts = _skewed_counts(16, 2, 32).sum(0)
        p2l, l2p, count = deepinfer.rebalance_experts(counts, 40, 5)
        self._check_layout(p2l, l2p, count, 32, 40)

    def test_beats_deepseek_vec_on_bursty_load(self):
        counts = _skewed_counts(200, 4, 384)
        num_physical, ep = 392, 8
        ours = deepinfer.rebalance_experts(counts, num_physical, num_physical // ep)
        vec = deepseek_vec.rebalance_experts(
            counts, num_physical, num_physical // ep, None, 1, False
        )
        self.assertGreater(
            predicted_rank_balancedness(counts, ours[0], ep),
            predicted_rank_balancedness(counts, vec[0], ep),
        )

    def test_dispatch_through_registry(self):
        self.assertEqual(
            compute_algorithm("deepinfer", num_groups=None, num_nodes=2),
            EplbAlgorithm.deepinfer,
        )
        counts = _skewed_counts(8, 2, 16)
        p2l, _, _ = rebalance_experts(
            tokens_per_expert=counts,
            num_physical_experts=24,
            num_local_physical_experts=3,
            num_groups=None,
            num_nodes=2,
            algorithm=EplbAlgorithm.deepinfer,
        )
        self.assertEqual(tuple(p2l.shape), (2, 24))

    def test_predicted_balancedness_perfect_layout(self):
        counts = torch.ones(4, 1, 8)
        p2l = torch.arange(8).unsqueeze(0)
        self.assertAlmostEqual(predicted_rank_balancedness(counts, p2l, 4), 1.0)


if __name__ == "__main__":
    unittest.main()
