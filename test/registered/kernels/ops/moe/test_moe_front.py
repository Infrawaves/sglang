import unittest

import torch

from sglang.kernels.ops.moe.moe_front import NUM_EXPERTS, TOPK, fused_front
from sglang.kernels.ops.moe.moe_fused_gate import moe_fused_gate
from sglang.srt.utils import get_device_sm
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=15, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

_HIDDEN_SIZE = 7168


class TestMoeFront(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")
        if get_device_sm() < 100:
            raise unittest.SkipTest("Kimi K3 compute kernels require SM100a+")

    def test_moe_front(self):
        torch.manual_seed(4)
        num_tokens, latent_dim = 1, 128
        hidden = (
            torch.randn(
                num_tokens,
                _HIDDEN_SIZE,
                device="cuda",
                dtype=torch.bfloat16,
            )
            / 32
        )
        weight = (
            torch.randn(
                NUM_EXPERTS + latent_dim,
                _HIDDEN_SIZE,
                device="cuda",
                dtype=torch.bfloat16,
            )
            / 32
        )
        bias = torch.randn(NUM_EXPERTS, device="cuda")

        weights, ids, routed = fused_front(
            hidden,
            weight,
            bias,
            latent_dim,
            renormalize=True,
            routed_scaling_factor=2.5,
            apply_routed_scaling_factor_on_output=True,
        )
        merged = torch.mm(hidden, weight.t(), out_dtype=torch.float32)
        ref_weights, ref_ids = moe_fused_gate(
            merged[:, :NUM_EXPERTS],
            bias,
            topk=TOPK,
            scoring_func="sigmoid",
            renormalize=True,
            routed_scaling_factor=2.5,
            apply_routed_scaling_factor_on_output=True,
        )
        order = ids.argsort(dim=-1)
        ref_order = ref_ids.argsort(dim=-1)
        self.assertTrue(
            torch.equal(
                ids.gather(1, order),
                ref_ids.to(torch.int32).gather(1, ref_order),
            )
        )
        torch.testing.assert_close(
            weights.gather(1, order),
            ref_weights.gather(1, ref_order),
            rtol=1e-6,
            atol=0,
        )
        self.assertTrue(
            torch.equal(
                routed,
                merged[:, NUM_EXPERTS:].to(torch.bfloat16),
            )
        )

    def _front_inputs(self, num_tokens, latent_dim=128, seed=7):
        torch.manual_seed(seed)
        hidden = (
            torch.randn(num_tokens, _HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16)
            / 32
        )
        weight = (
            torch.randn(
                NUM_EXPERTS + latent_dim,
                _HIDDEN_SIZE,
                device="cuda",
                dtype=torch.bfloat16,
            )
            / 32
        )
        bias = torch.randn(NUM_EXPERTS, device="cuda")
        return hidden, weight, bias, latent_dim

    def test_static_remap_and_count(self):
        """Static EPLB dispatch applied in the epilogue == gather afterwards,
        and the in-kernel count == scatter_add over non-padded rows."""
        from types import SimpleNamespace

        num_tokens, num_valid = 24, 19
        hidden, weight, bias, latent = self._front_inputs(num_tokens)
        num_physical = NUM_EXPERTS + 64
        for map_dtype in (torch.int64, torch.int32):
            perm = torch.randperm(num_physical, device="cuda")[:NUM_EXPERTS]
            info = SimpleNamespace(
                ep_dispatch_algorithm="static",
                partial_logical_to_rank_dispatch_physical_map=perm.to(map_dtype),
                partial_logical_to_all_physical_map=None,
                partial_logical_to_all_physical_map_num_valid=None,
                num_physical_experts=num_physical,
                rank_invariant=False,
            )
            count = torch.zeros(num_physical, dtype=torch.int32, device="cuda")
            valid = torch.tensor(num_valid, dtype=torch.int32, device="cuda")

            w_ref, i_ref, r_ref = fused_front(hidden, weight, bias, latent)
            w, i, r = fused_front(
                hidden,
                weight,
                bias,
                latent,
                dispatch_info=info,
                physical_count=count,
                num_valid_tokens=valid,
            )
            self.assertTrue(torch.equal(w, w_ref))
            self.assertTrue(torch.equal(r, r_ref))
            expected_ids = perm[i_ref.long()].to(torch.int32)
            self.assertTrue(torch.equal(i, expected_ids))

            expected_count = torch.zeros_like(count)
            expected_count.scatter_add_(
                0,
                expected_ids[:num_valid].reshape(-1).long(),
                torch.ones(num_valid * TOPK, dtype=torch.int32, device="cuda"),
            )
            self.assertTrue(torch.equal(count, expected_count))

    def test_dynamic_remap_picks_valid_replicas(self):
        """Dynamic dispatch must land on one of the logical expert's replicas
        and spread rows over them."""
        from types import SimpleNamespace

        num_tokens = 64
        hidden, weight, bias, latent = self._front_inputs(num_tokens, seed=11)
        num_replicas = 3
        num_physical = NUM_EXPERTS * num_replicas
        # logical e -> physical {e, e + E, e + 2E}; odd experts get 2 replicas.
        all_physical = torch.stack(
            [
                torch.arange(NUM_EXPERTS, device="cuda") + k * NUM_EXPERTS
                for k in range(num_replicas)
            ],
            dim=1,
        )
        num_valid = torch.full((NUM_EXPERTS,), num_replicas, device="cuda")
        num_valid[1::2] = 2
        all_physical[1::2, 2] = -1
        for rank_invariant in (False, True):
            info = SimpleNamespace(
                ep_dispatch_algorithm="dynamic",
                partial_logical_to_rank_dispatch_physical_map=None,
                partial_logical_to_all_physical_map=all_physical,
                partial_logical_to_all_physical_map_num_valid=num_valid,
                num_physical_experts=num_physical,
                rank_invariant=rank_invariant,
            )
            _, i_ref, _ = fused_front(hidden, weight, bias, latent)
            _, i, _ = fused_front(
                hidden, weight, bias, latent, dispatch_info=info, dispatch_salt=5
            )
            logical = i_ref.long()
            replica = (i.long() - logical) // NUM_EXPERTS
            self.assertTrue(torch.all((i.long() - logical) % NUM_EXPERTS == 0))
            self.assertTrue(torch.all(replica >= 0))
            self.assertTrue(torch.all(replica < num_valid[logical]))
            if rank_invariant:
                expected = (
                    torch.arange(num_tokens, device="cuda")[:, None]
                    % num_valid[logical]
                )
                self.assertTrue(torch.equal(replica, expected))
            else:
                # Every replica index is used across the batch.
                self.assertEqual(set(replica.unique().tolist()), {0, 1, 2})


if __name__ == "__main__":
    unittest.main()
