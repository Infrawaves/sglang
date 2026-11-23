"""PD first-token logprob state carried in the prefill's metadata row.

In PD the prefill samples the first output token, and its metadata row carries
that token's logprob only when the prefill's copy of the request asked for
logprobs. Rows are reused, so decode must not report a previous owner's value
(or the zero fill) as this request's logprob.
"""

import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.disaggregation.decode import DecodeRequest, DecodeTransferQueue
from sglang.srt.disaggregation.utils import (
    FIRST_TOKEN_LOGPROB_ABSENT,
    FIRST_TOKEN_LOGPROB_SLOT,
    FIRST_TOKEN_LOGPROB_UNKNOWN,
    FIRST_TOKEN_LOGPROB_WRITTEN,
    MetadataBuffers,
)
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)
from sglang.srt.managers.scheduler_components.logprob_result_processor import (
    SchedulerLogprobResultProcessor,
)
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _prefill_req(*, return_logprob, logprob=None, token_id=11):
    logprob_state = SimpleNamespace(
        output_token_logprobs_val=[] if logprob is None else [logprob],
        output_token_logprobs_idx=[] if logprob is None else [token_id],
        output_top_logprobs_val=[],
        output_top_logprobs_idx=[],
    )
    return SimpleNamespace(
        metadata_buffer_index=0,
        output_ids=[token_id],
        cached_tokens=0,
        cached_tokens_device=0,
        cached_tokens_host=0,
        cached_tokens_storage=0,
        get_mm_token_counts=lambda: (0, 0, 0),
        return_logprob=return_logprob,
        logprob=logprob_state,
        return_sampling_mask=False,
        hidden_states_tensor=None,
        bootstrap_room=7,
        customized_info=None,
    )


class TestPrefillWritesFirstTokenLogprobState(CustomTestCase):
    def test_every_owner_of_a_row_records_whether_it_wrote_the_logprob(self):
        buf = MetadataBuffers(
            size=2,
            hidden_size=16,
            hidden_states_dtype=torch.float32,
            max_sampling_mask_tokens=128,
        )
        buf.set_buf(_prefill_req(return_logprob=True, logprob=-0.5))
        self.assertEqual(
            buf.cached_tokens[0][FIRST_TOKEN_LOGPROB_SLOT].item(),
            FIRST_TOKEN_LOGPROB_WRITTEN,
        )
        self.assertAlmostEqual(buf.output_token_logprobs_val[0][0].item(), -0.5)

        # The row's next owner computed no logprob: the previous owner's -0.5
        # is still in the row, and the state says it is not this request's.
        buf.set_buf(_prefill_req(return_logprob=False))
        self.assertEqual(
            buf.cached_tokens[0][FIRST_TOKEN_LOGPROB_SLOT].item(),
            FIRST_TOKEN_LOGPROB_ABSENT,
        )
        self.assertAlmostEqual(buf.output_token_logprobs_val[0][0].item(), -0.5)

        # Asked for logprobs but has none to send.
        buf.set_buf(_prefill_req(return_logprob=True))
        self.assertEqual(
            buf.cached_tokens[0][FIRST_TOKEN_LOGPROB_SLOT].item(),
            FIRST_TOKEN_LOGPROB_ABSENT,
        )


class TestDecodeCommitsFirstTokenLogprob(CustomTestCase):
    def setUp(self):
        world_size = patch("torch.distributed.get_world_size", return_value=1)
        world_size.start()
        self.addCleanup(world_size.stop)
        reset_context()
        self.addCleanup(reset_context)
        publish(ServerArgs(model_path="dummy"), role="tokenizer")

    def _commit(self, state):
        params = SamplingParams()
        params.normalize(None)
        req = Req(
            "first-token-logprob",
            "",
            [1, 2, 3],
            params,
            return_logprob=True,
            bootstrap_host="prefill",
            bootstrap_room=7,
            vocab_size=2048,
        )
        processor = SchedulerBatchResultProcessor.__new__(
            SchedulerBatchResultProcessor
        )
        object.__setattr__(
            processor,
            "model_config",
            SimpleNamespace(self_opened_think_start_ids=[]),
        )
        object.__setattr__(
            processor, "logprob_result_processor", SchedulerLogprobResultProcessor
        )
        queue = DecodeTransferQueue.__new__(DecodeTransferQueue)
        queue.scheduler = SimpleNamespace(
            kv_checksum_computer=None, batch_result_processor=processor
        )
        queue.spec_algorithm = SpeculativeAlgorithm.NONE
        cached_tokens = torch.zeros(16, dtype=torch.int32)
        cached_tokens[FIRST_TOKEN_LOGPROB_SLOT] = state
        # The row holds -0.5 for token 12: whatever its last writer left.
        queue.metadata_buffers = SimpleNamespace(
            get_buf=lambda idx: (
                torch.tensor([11]),
                cached_tokens,
                torch.tensor([-0.5]),
                torch.tensor([12]),
                torch.zeros(0),
                torch.zeros(0, dtype=torch.int32),
                *([None] * 7),
                torch.tensor([7]),
            ),
            get_customized_info=lambda idx: None,
        )
        queue._commit_transfer_to_req(
            DecodeRequest(
                req,
                SimpleNamespace(clear=lambda: None, failure_exception=lambda: None),
                metadata_buffer_index=0,
            )
        )
        self.assertEqual(req.output_ids, [11])
        return req.logprob

    def test_written_and_legacy_rows_are_trusted(self):
        for state in (FIRST_TOKEN_LOGPROB_WRITTEN, FIRST_TOKEN_LOGPROB_UNKNOWN):
            with self.subTest(state=state):
                logprob = self._commit(state)
                self.assertEqual(logprob.output_token_logprobs_val, [-0.5])
                self.assertEqual(logprob.output_token_logprobs_idx, [12])

    def test_absent_logprob_is_committed_as_unknown_not_the_stale_row(self):
        logprob = self._commit(FIRST_TOKEN_LOGPROB_ABSENT)
        # One row per output token, so later positions stay aligned.
        self.assertEqual(len(logprob.output_token_logprobs_val), 1)
        self.assertTrue(math.isnan(logprob.output_token_logprobs_val[0]))
        self.assertEqual(logprob.output_token_logprobs_idx, [11])


if __name__ == "__main__":
    unittest.main()
