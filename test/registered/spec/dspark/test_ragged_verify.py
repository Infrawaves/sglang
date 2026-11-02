import unittest

import torch

from sglang.srt.speculative.ragged_verify import (
    RaggedVerifyLayout,
    build_ragged_target_verify_geometry,
    unpad_dense_verify_rows,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_DEVICE = torch.device("cpu")
_GRID = [8, 16, 24, 32, 64]

# The backend capability checks (supports_ragged_verify_graph) live in
# test_ragged_verify_backend_capability.py: importing the backend modules
# pulls GPU-only wheels, which fail to import on the CPU runners.


class TestRaggedTargetVerifyGeometry(CustomTestCase):
    def test_mixed_verify_lens_geometry(self):
        layout = RaggedVerifyLayout.from_verify_lens(
            verify_lens_cpu=[8, 1, 3], device=_DEVICE, grid=_GRID
        )
        seq_lens = torch.tensor([10, 20, 30], dtype=torch.int32)
        geometry = build_ragged_target_verify_geometry(seq_lens=seq_lens, layout=layout)
        self.assertEqual(geometry.cache_seqlens_int32.tolist(), [18, 21, 33])
        self.assertEqual(geometry.cu_seqlens_q.tolist(), [0, 8, 9, 12])
        self.assertEqual(geometry.cu_seqlens_k.tolist(), [0, 18, 39, 72])
        self.assertEqual(geometry.max_seq_len_q, 8)

    def test_geometry_dtypes_are_int32(self):
        layout = RaggedVerifyLayout.from_verify_lens(
            verify_lens_cpu=[8, 1, 3], device=_DEVICE, grid=_GRID
        )
        seq_lens = torch.tensor([10, 20, 30], dtype=torch.int64)
        geometry = build_ragged_target_verify_geometry(seq_lens=seq_lens, layout=layout)
        self.assertEqual(geometry.cache_seqlens_int32.dtype, torch.int32)
        self.assertEqual(geometry.cu_seqlens_q.dtype, torch.int32)
        self.assertEqual(geometry.cu_seqlens_k.dtype, torch.int32)


class TestPaddedRaggedVerifyGeometry(CustomTestCase):
    def test_padded_layout_grows_bs_and_fills_bucket(self):
        raw = RaggedVerifyLayout.from_verify_lens(
            verify_lens_cpu=[8, 1, 3],
            device=_DEVICE,
            grid=[8, 16, 32, 64],
            graph_num_tokens_floor=24,
        )
        self.assertEqual(raw.graph_num_tokens, 32)
        padded = raw.padded_to_bucket(padded_bs=4)
        self.assertEqual(padded.bs, 4)
        self.assertEqual(padded.verify_lens.tolist(), [8, 1, 3, 20])
        self.assertEqual(padded.qo_indptr_device.tolist(), [0, 8, 9, 12, 32])
        seq_lens = torch.tensor([10, 20, 30, 1], dtype=torch.int32)
        geometry = build_ragged_target_verify_geometry(seq_lens=seq_lens, layout=padded)
        self.assertEqual(geometry.cu_seqlens_q.tolist(), [0, 8, 9, 12, 32])
        self.assertEqual(geometry.cache_seqlens_int32.tolist(), [18, 21, 33, 21])
        self.assertEqual(int(geometry.cu_seqlens_k[-1]), 18 + 21 + 33 + 21)

    def test_padded_layout_decoupled_slots_spread_slack(self):
        raw = RaggedVerifyLayout.from_verify_lens(
            verify_lens_cpu=[8, 1, 3],
            device=_DEVICE,
            grid=[8, 16, 32, 64],
            graph_num_tokens_floor=24,
        )
        padded = raw.padded_to_bucket(padded_bs=6)
        self.assertEqual(padded.bs, 6)
        self.assertEqual(padded.verify_lens.tolist(), [8, 1, 3, 7, 7, 6])
        self.assertEqual(int(padded.qo_indptr_device[-1]), 32)

    def test_padded_layout_budget_tier_below_uniform(self):
        raw = RaggedVerifyLayout.from_verify_lens(
            verify_lens_cpu=[8, 1, 3],
            device=_DEVICE,
            grid=[8, 16, 32, 64],
        )
        self.assertEqual(raw.graph_num_tokens, 16)
        padded = raw.padded_to_bucket(padded_bs=3)
        self.assertEqual(padded.verify_lens.tolist(), [8, 1, 7])
        self.assertEqual(int(padded.qo_indptr_device[-1]), 16)

    def test_padded_layout_zero_len_pad_rows(self):
        raw = RaggedVerifyLayout.from_verify_lens(
            verify_lens_cpu=[8, 8],
            device=_DEVICE,
            grid=[8, 16, 32, 64],
        )
        self.assertEqual(raw.graph_num_tokens, 16)
        padded = raw.padded_to_bucket(padded_bs=8)
        self.assertEqual(padded.verify_lens.tolist(), [8, 8, 0, 0, 0, 0, 0, 0])
        self.assertEqual(int(padded.qo_indptr_device[-1]), 16)


class TestCaptureVerifyLens(CustomTestCase):
    def test_small_tier_one_token_rows(self):
        from sglang.srt.speculative.ragged_verify import build_capture_verify_lens

        lens = build_capture_verify_lens(num_tokens=8, num_slots=8, num_draft_tokens=8)
        self.assertEqual(lens, [1] * 8)

    def test_large_tier_spreads_within_window(self):
        from sglang.srt.speculative.ragged_verify import build_capture_verify_lens

        lens = build_capture_verify_lens(
            num_tokens=1024, num_slots=128, num_draft_tokens=8
        )
        self.assertEqual(sum(lens), 1024)
        self.assertEqual(lens, [8] * 128)

    def test_uneven_tier_rows_stay_legal(self):
        from sglang.srt.speculative.ragged_verify import build_capture_verify_lens

        lens = build_capture_verify_lens(num_tokens=24, num_slots=5, num_draft_tokens=8)
        self.assertEqual(sum(lens), 24)
        self.assertTrue(all(1 <= v <= 8 for v in lens))

    def test_rejects_overpacked_tier(self):
        from sglang.srt.speculative.ragged_verify import build_capture_verify_lens

        with self.assertRaises(ValueError):
            build_capture_verify_lens(num_tokens=64, num_slots=4, num_draft_tokens=8)
        with self.assertRaises(ValueError):
            build_capture_verify_lens(num_tokens=4, num_slots=8, num_draft_tokens=8)


def _pack_reference(dense, *, verify_lens, width, num_tokens, fill):
    """Packed rows of a dense [bs * width, ...] tensor, filled past the tail."""
    out = torch.full((num_tokens, *dense.shape[1:]), fill, dtype=dense.dtype)
    t = 0
    for r, n in enumerate(verify_lens):
        for j in range(n):
            out[t] = dense[r * width + j]
            t += 1
    return out


def _lse_combine(outs, lses):
    """Row-wise LSE merge of per-rank attention partials (natural log)."""
    lse = torch.stack(lses)  # [ranks, rows, heads]
    m = lse.max(dim=0).values
    w = torch.exp(lse - m)
    num = sum(w[i].unsqueeze(-1) * outs[i] for i in range(len(outs)))
    return num / w.sum(dim=0).unsqueeze(-1)


class TestUnpadDenseVerifyRows(CustomTestCase):
    def test_matches_reference_on_random_layouts(self):
        g = torch.Generator().manual_seed(20261101)
        for _ in range(300):
            width = int(torch.randint(2, 9, (1,), generator=g))
            bs = int(torch.randint(1, 17, (1,), generator=g))
            lens = torch.randint(1, width + 1, (bs,), generator=g).tolist()
            num_tokens = sum(lens) + int(torch.randint(0, 13, (1,), generator=g))
            cu = torch.tensor([0] + lens, dtype=torch.int32).cumsum(0).to(torch.int32)
            dense = torch.randn(bs * width, 3, 4, generator=g)
            got = unpad_dense_verify_rows(
                dense,
                cu_seqlens_q=cu,
                seq_lens_q=torch.tensor(lens, dtype=torch.int32),
                num_tokens=num_tokens,
                width=width,
            )
            want = _pack_reference(
                dense, verify_lens=lens, width=width, num_tokens=num_tokens, fill=0.0
            )
            self.assertTrue(torch.equal(got, want))

    def test_layout_indptr_and_lse_shape(self):
        # The backend passes the layout's own qo_indptr and a 2-D lse.
        layout = RaggedVerifyLayout.from_verify_lens(
            verify_lens_cpu=[6, 1, 3], device=_DEVICE, grid=_GRID
        )
        width = 6
        lse = torch.arange(3 * width * 2, dtype=torch.float32).view(3 * width, 2)
        got = unpad_dense_verify_rows(
            lse,
            cu_seqlens_q=layout.qo_indptr_device,
            seq_lens_q=layout.verify_lens,
            num_tokens=layout.graph_num_tokens,
            width=width,
        )
        self.assertEqual(got.shape, (layout.graph_num_tokens, 2))
        want = _pack_reference(
            lse,
            verify_lens=[6, 1, 3],
            width=width,
            num_tokens=layout.graph_num_tokens,
            fill=0.0,
        )
        self.assertTrue(torch.equal(got, want))

    def test_rows_past_clamped_lens_are_filled(self):
        # Capped layouts clamp seq_lens_q below the indptr span; the rows the
        # clamp drops must come back as fill, not as another request's rows.
        dense = torch.arange(2 * 4, dtype=torch.float32).view(8, 1) + 1
        got = unpad_dense_verify_rows(
            dense,
            cu_seqlens_q=torch.tensor([0, 4, 6], dtype=torch.int32),
            seq_lens_q=torch.tensor([2, 2], dtype=torch.int32),
            num_tokens=8,
            width=4,
            fill_value=-7.0,
        )
        self.assertEqual(
            got.view(-1).tolist(), [1.0, 2.0, -7.0, -7.0, 5.0, 6.0, -7.0, -7.0]
        )

    def test_unpad_commutes_with_dcp_lse_combine(self):
        # DCP merges partials row by row, so unpadding each rank's out and lse
        # before the merge equals merging the dense rows and then unpadding.
        g = torch.Generator().manual_seed(7)
        width, lens = 6, [3, 6, 1, 2]
        bs, num_tokens = len(lens), 16
        cu = torch.tensor([0] + lens, dtype=torch.int32).cumsum(0).to(torch.int32)
        seq_lens = torch.tensor(lens, dtype=torch.int32)
        outs = [torch.randn(bs * width, 2, 8, generator=g) for _ in range(4)]
        lses = [torch.randn(bs * width, 2, generator=g) for _ in range(4)]

        def unpad(x):
            return unpad_dense_verify_rows(
                x,
                cu_seqlens_q=cu,
                seq_lens_q=seq_lens,
                num_tokens=num_tokens,
                width=width,
            )

        merged_then_unpadded = unpad(_lse_combine(outs, lses))
        unpadded_then_merged = _lse_combine(
            [unpad(o) for o in outs], [unpad(x) for x in lses]
        )
        real = sum(lens)
        torch.testing.assert_close(
            unpadded_then_merged[:real], merged_then_unpadded[:real]
        )
        # Tier padding rows: out = 0 and lse = 0 on every rank -> finite 0.
        self.assertTrue(torch.isfinite(unpadded_then_merged[real:]).all())
        self.assertEqual(float(unpadded_then_merged[real:].abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
