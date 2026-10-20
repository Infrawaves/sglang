"""Correctness coverage for the fused MLA DCP decode metadata kernel."""

import unittest

import torch

from sglang.kernels.ops.attention.dcp_kernels import prepare_dcp_local_lens


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TestPrepareDCPLocalLens(unittest.TestCase):
    def test_matches_torch_reference(self):
        generator = torch.Generator().manual_seed(20260905)
        for batch_size in (1, 7, 64, 257):
            host_lens = torch.randint(
                0, 1_000_000, (batch_size,), generator=generator, dtype=torch.int32
            )
            lens = host_lens.cuda()
            for world_size in (2, 4, 8):
                for rank in range(world_size):
                    actual_lens = torch.empty_like(lens)
                    actual_indptr = torch.empty(
                        batch_size + 1, dtype=torch.int32, device="cuda"
                    )
                    prepare_dcp_local_lens(
                        lens, actual_lens, actual_indptr, rank, world_size
                    )

                    expected_lens = host_lens // world_size + (
                        rank < host_lens % world_size
                    )
                    expected_indptr = torch.zeros(batch_size + 1, dtype=torch.int32)
                    expected_indptr[1:] = expected_lens.cumsum(0)
                    torch.testing.assert_close(actual_lens.cpu(), expected_lens)
                    torch.testing.assert_close(actual_indptr.cpu(), expected_indptr)


if __name__ == "__main__":
    unittest.main()
