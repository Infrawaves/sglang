import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.attention.dsa.dsa_indexer import (
    Indexer,
    _make_eager_idle_topk_result,
)
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_INDEXER = "sglang.srt.layers.attention.dsa.dsa_indexer"


class TestDSAIndexerIdle(unittest.TestCase):
    def test_builds_invalid_topk_rows_for_padded_idle_tokens(self):
        result = _make_eager_idle_topk_result(
            torch.empty((8, 16)),
            index_topk=4,
            return_indices=True,
        )

        self.assertEqual(result.shape, (8, 4))
        self.assertEqual(result.dtype, torch.int32)
        self.assertTrue(torch.all(result == -1))

    def test_returns_none_when_indices_are_not_consumed(self):
        self.assertIsNone(
            _make_eager_idle_topk_result(
                torch.empty((8, 16)),
                index_topk=4,
                return_indices=False,
            )
        )

    def test_eager_idle_short_circuits_before_paged_mqa(self):
        indexer = SimpleNamespace(index_topk=4)
        batch = SimpleNamespace(forward_mode=ForwardMode.IDLE)

        with (
            patch(f"{_INDEXER}._is_cuda", True),
            patch(f"{_INDEXER}.get_is_capture_mode", return_value=False),
            patch(
                f"{_INDEXER}._broadcast_indexer_topk_from_rank0",
                side_effect=lambda result: result,
            ),
            patch(
                f"{_INDEXER}.maybe_capture_indexer_topk",
                side_effect=lambda _layer_id, result: result,
            ),
        ):
            result = Indexer.forward_cuda(
                indexer,
                x=torch.empty((8, 16)),
                q_lora=torch.empty((8, 16)),
                positions=torch.empty((8,), dtype=torch.int64),
                forward_batch=batch,
                layer_id=3,
            )

        self.assertEqual(result.shape, (8, 4))
        self.assertTrue(torch.all(result == -1))


class TestDSAIndexerDcpGatherEmptyRank(unittest.TestCase):
    def test_rank_owning_no_index_k_rows_joins_the_gather(self):
        """A prefill shorter than dcp_size leaves some DCP ranks with no
        index-K rows; those ranks must still contribute a [0, W] slab to the
        collective gather instead of raising, or the peer ranks hang in it."""
        head_dim, scale_bytes, seq_len, dcp_size = 128, 4, 4, 8
        seq_lens_cpu = torch.tensor([seq_len], dtype=torch.int32)
        pool = SimpleNamespace(
            get_index_k_scale_buffer=lambda _layer, _lens, _pages, rows, _max: (
                torch.zeros((rows, head_dim), dtype=torch.uint8),
                torch.zeros((rows, scale_bytes), dtype=torch.uint8),
            )
        )
        contributed = []

        def gather(local_rows, lens_cpu):
            contributed.append(tuple(local_rows.shape))
            return torch.zeros(
                (int(lens_cpu.sum()), local_rows.shape[1]), dtype=torch.uint8
            )

        # Token i lives on rank i % dcp_size, so ranks 4..7 own nothing.
        for dcp_rank in range(seq_len, dcp_size):
            parallel = SimpleNamespace(attn_dcp_size=dcp_size, attn_dcp_rank=dcp_rank)
            with (
                patch(f"{_INDEXER}.get_parallel", return_value=parallel),
                patch(f"{_INDEXER}.get_token_to_kv_pool", return_value=pool),
                patch(f"{_INDEXER}.dcp_gather_request_rows", side_effect=gather),
            ):
                k_fp8, k_scale = Indexer._dcp_gather_index_k(
                    0,
                    torch.zeros((1, 1), dtype=torch.int32),
                    seq_lens_cpu.clone(),
                    seq_lens_cpu,
                )
            self.assertEqual(k_fp8.shape, (seq_len, head_dim))
            self.assertEqual(k_scale.shape, (seq_len, scale_bytes))

        self.assertEqual(
            contributed, [(0, head_dim + scale_bytes)] * (dcp_size - seq_len)
        )

    def test_one_token_gather_returns_tma_aligned_rows(self):
        """A one-token prefill (the /health probe sends input_ids=[0]) gathers a
        single row; DeepGEMM's fp8_mqa_logits needs a 16-byte-aligned row
        stride, which a one-row split of the [K | scale] slab does not have."""
        head_dim, scale_bytes, dcp_size = 128, 4, 8
        seq_lens_cpu = torch.tensor([1], dtype=torch.int32)
        pool = SimpleNamespace(
            get_index_k_scale_buffer=lambda _layer, _lens, _pages, rows, _max: (
                torch.zeros((rows, head_dim), dtype=torch.uint8),
                torch.zeros((rows, scale_bytes), dtype=torch.uint8),
            )
        )

        def gather(local_rows, lens_cpu):
            return torch.zeros(
                (int(lens_cpu.sum()), local_rows.shape[1]), dtype=torch.uint8
            )

        parallel = SimpleNamespace(attn_dcp_size=dcp_size, attn_dcp_rank=0)
        with (
            patch(f"{_INDEXER}.get_parallel", return_value=parallel),
            patch(f"{_INDEXER}.get_token_to_kv_pool", return_value=pool),
            patch(f"{_INDEXER}.dcp_gather_request_rows", side_effect=gather),
        ):
            k_fp8, k_scale = Indexer._dcp_gather_index_k(
                0,
                torch.zeros((1, 1), dtype=torch.int32),
                seq_lens_cpu.clone(),
                seq_lens_cpu,
            )
        self.assertEqual(k_fp8.shape, (1, head_dim))
        self.assertEqual(k_fp8.stride(0) * k_fp8.element_size() % 16, 0)
        self.assertEqual(k_scale.stride(), (scale_bytes, 1))


if __name__ == "__main__":
    unittest.main()
