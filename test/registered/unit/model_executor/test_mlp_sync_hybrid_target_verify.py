import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.model_executor import forward_batch_info as fbi
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.runner.eager_runner import EagerRunner
from sglang.srt.speculative.spec_info import SpecInputType
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

NUM_TOKENS_PER_REQ = 4


def _sync(
    forward_mode,
    num_reqs,
    global_num_tokens,
    *,
    verify_width=NUM_TOKENS_PER_REQ,
    attn_tp_size=1,
    padding_mode=fbi.DpPaddingMode.MAX_LEN,
    is_extend_in_batch=True,
):
    """Run the eager DP padding path as rank 0 of a hybrid-SSM target."""
    num_tokens = num_reqs * verify_width
    batch = ForwardBatch(
        forward_mode=forward_mode,
        batch_size=num_reqs,
        input_ids=torch.zeros(num_tokens, dtype=torch.long),
        req_pool_indices=torch.arange(1, num_reqs + 1),
        seq_lens=torch.full((num_reqs,), 9),
        out_cache_loc=torch.arange(1, num_tokens + 1),
        seq_lens_sum=9 * num_reqs,
        positions=torch.arange(num_tokens),
        spec_info=SimpleNamespace(
            spec_input_type=SpecInputType.DFLASH_VERIFY,
            num_tokens_per_req=verify_width,
            is_draft_input=lambda: False,
            ragged_verify_layout=None,
        ),
        is_extend_in_batch=is_extend_in_batch,
        global_num_tokens_cpu=list(global_num_tokens),
        global_num_tokens_for_logprob_cpu=list(global_num_tokens),
        global_num_tokens_gpu=torch.zeros(len(global_num_tokens), dtype=torch.int64),
    )
    model_runner = MagicMock(is_draft_worker=False)
    model_runner.attn_tp_sequence_sharded.return_value = False
    model_runner.attn_backend.get_cpu_graph_seq_len_fill_value.return_value = 1
    no_prefill_graph = SimpleNamespace(
        graph=SimpleNamespace(
            cuda_graph_config=SimpleNamespace(prefill=SimpleNamespace(bs=[]))
        )
    )
    with (
        patch.multiple(
            fbi,
            get_parallel=lambda: SimpleNamespace(attn_tp_size=attn_tp_size),
            get_exec=lambda: no_prefill_graph,
            _elastic_should_preserve_local_token_counts=lambda **kwargs: False,
            dp_slot_in=lambda per_rank: 0,
            set_dp_buffer_len_from_batch=lambda *args: None,
            set_is_extend_in_batch=lambda *args: None,
            _is_cpu=True,
        ),
        patch.object(
            fbi, "mambaish_config", lambda model_config: object(), create=True
        ),
        patch.object(
            fbi.DpPaddingMode,
            "get_dp_padding_mode",
            return_value=padding_mode,
        ),
        patch(
            "sglang.srt.batch_overlap.two_batch_overlap.TboForwardBatchPreparer.prepare"
        ),
    ):
        batch.prepare_mlp_sync_batch(model_runner)
    return batch


class TestMlpSyncHybridTargetVerify(CustomTestCase):
    def test_verify_batch_keeps_its_verify_layout(self):
        """In a DP step padded to MAX_LEN while another rank prefills, a hybrid-SSM
        verify batch keeps its verify layout: one request per draft width."""
        batch = _sync(ForwardMode.TARGET_VERIFY, num_reqs=2, global_num_tokens=[8, 0])
        self.assertEqual(batch.forward_mode, ForwardMode.TARGET_VERIFY)
        self.assertEqual(batch.batch_size, 2)
        self.assertIsNone(batch.extend_seq_lens)

    def test_idle_rank_joins_as_verify_batch(self):
        """An idle rank in the same verify step runs as a verify batch of the
        padded width, not as a fabricated prefill."""
        batch = _sync(ForwardMode.IDLE, num_reqs=0, global_num_tokens=[0, 8])
        self.assertEqual(batch.forward_mode, ForwardMode.TARGET_VERIFY)
        self.assertEqual(batch._original_forward_mode, ForwardMode.IDLE)
        self.assertEqual(batch.batch_size, 8 // NUM_TOKENS_PER_REQ)

    def test_zero_row_verify_rank_keeps_idle_mode(self):
        batch = _sync(
            ForwardMode.IDLE,
            num_reqs=0,
            global_num_tokens=[0, 8, 0, 0],
            verify_width=8,
            attn_tp_size=4,
            padding_mode=fbi.DpPaddingMode.SUM_LEN,
            is_extend_in_batch=False,
        )
        self.assertEqual(batch.forward_mode, ForwardMode.IDLE)
        self.assertEqual(batch.batch_size, 0)
        self.assertEqual(batch.input_ids.numel(), 0)
        self.assertEqual(batch.global_dp_buffer_len, 8)
        runner = EagerRunner.__new__(EagerRunner)
        runner.model_runner = MagicMock(device_timer=None)
        runner.model_runner._pp_kwargs.return_value = {}
        runner.execute(batch)
        runner.model_runner.attn_backend.init_forward_metadata.assert_not_called()
        runner.model_runner.model.forward.assert_called_once_with(
            batch.input_ids, batch.positions, batch
        )

    def test_tp_padding_adds_whole_verify_requests(self):
        for mode, num_reqs, counts in (
            (ForwardMode.TARGET_VERIFY, 1, [6, 0]),
            (ForwardMode.IDLE, 0, [0, 6]),
        ):
            with self.subTest(mode=mode):
                batch = _sync(
                    mode,
                    num_reqs=num_reqs,
                    global_num_tokens=counts,
                    verify_width=6,
                    attn_tp_size=4,
                    is_extend_in_batch=False,
                )
                self.assertEqual(batch.global_num_tokens_cpu, [12, 12])
                self.assertEqual(batch.forward_mode, ForwardMode.TARGET_VERIFY)
                self.assertEqual(batch.batch_size, 2)
                self.assertEqual(batch.input_ids.numel(), batch.batch_size * 6)
                self.assertEqual(
                    batch.req_pool_indices.tolist(),
                    [1] * num_reqs + [0] * (batch.batch_size - num_reqs),
                )
                real_tokens = num_reqs * 6
                self.assertEqual(
                    batch.out_cache_loc.tolist(),
                    list(range(1, real_tokens + 1)) + [0] * (12 - real_tokens),
                )


if __name__ == "__main__":
    unittest.main()
