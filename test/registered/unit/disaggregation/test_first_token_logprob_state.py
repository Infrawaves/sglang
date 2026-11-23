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
    FAKE_BOOTSTRAP_HOST,
    FIRST_TOKEN_LOGPROB_ABSENT,
    FIRST_TOKEN_LOGPROB_SLOT,
    FIRST_TOKEN_LOGPROB_UNKNOWN,
    FIRST_TOKEN_LOGPROB_WRITTEN,
    InvalidDisaggregationMetadata,
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

TOP_CAPACITY = 4


def _buffers():
    return MetadataBuffers(
        size=2,
        hidden_size=16,
        hidden_states_dtype=torch.float32,
        max_sampling_mask_tokens=128,
        max_top_logprobs_num=TOP_CAPACITY,
    )


def _prefill_req(
    *,
    return_logprob,
    logprob=None,
    token_id=11,
    logprob_token_id=None,
    top_logprobs_num=0,
    top=None,
):
    """A prefill request after its first token; ``top`` is ``(values, ids)``
    of the first token's top row, None when no row was computed."""
    if logprob_token_id is None:
        logprob_token_id = token_id
    logprob_state = SimpleNamespace(
        output_token_logprobs_val=[] if logprob is None else [logprob],
        output_token_logprobs_idx=[] if logprob is None else [logprob_token_id],
        output_top_logprobs_val=[] if top is None else [top[0]],
        output_top_logprobs_idx=[] if top is None else [top[1]],
        top_logprobs_num=top_logprobs_num,
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


def _state(buf):
    return buf.cached_tokens[0][FIRST_TOKEN_LOGPROB_SLOT].item()


def _snapshot(buf):
    return {
        name: value.clone()
        for name, value in vars(buf).items()
        if isinstance(value, torch.Tensor)
    }


class TestPrefillWritesFirstTokenLogprobState(CustomTestCase):
    def test_every_owner_of_a_row_records_whether_it_wrote_the_logprob(self):
        buf = _buffers()
        buf.set_buf(_prefill_req(return_logprob=True, logprob=-0.5))
        self.assertEqual(_state(buf), FIRST_TOKEN_LOGPROB_WRITTEN)
        self.assertAlmostEqual(buf.output_token_logprobs_val[0][0].item(), -0.5)

        # The row's next owner sampled the same token but computed no
        # logprob: the previous owner's -0.5 and token ID are still in the
        # row, and only the state says they are not this request's.
        buf.set_buf(_prefill_req(return_logprob=False))
        self.assertEqual(_state(buf), FIRST_TOKEN_LOGPROB_ABSENT)
        self.assertAlmostEqual(buf.output_token_logprobs_val[0][0].item(), -0.5)
        self.assertEqual(buf.output_token_logprobs_idx[0][0].item(), 11)

        # Asked for logprobs but has none to send.
        buf.set_buf(_prefill_req(return_logprob=True))
        self.assertEqual(_state(buf), FIRST_TOKEN_LOGPROB_ABSENT)

    def test_valid_selected_logprob_with_and_without_top_rows_is_written(self):
        buf = _buffers()
        buf.set_buf(_prefill_req(return_logprob=True, logprob=-0.5))
        self.assertEqual(_state(buf), FIRST_TOKEN_LOGPROB_WRITTEN)
        self.assertEqual(buf.output_token_logprobs_idx[0][0].item(), 11)

        buf.set_buf(
            _prefill_req(
                return_logprob=True,
                logprob=-0.25,
                top_logprobs_num=2,
                top=([-0.25, -1.5], [11, 40]),
            )
        )
        self.assertEqual(_state(buf), FIRST_TOKEN_LOGPROB_WRITTEN)
        self.assertEqual(buf.output_top_logprobs_val[0][:2].tolist(), [-0.25, -1.5])
        self.assertEqual(buf.output_top_logprobs_idx[0][:2].tolist(), [11, 40])

    def test_incomplete_selected_metadata_is_rejected_before_any_write(self):
        cases = {
            "value_without_token_id": dict(logprob=-0.5),
            "token_id_of_another_token": dict(logprob=-0.5, logprob_token_id=12),
            "top_requested_but_missing": dict(logprob=-0.5, top_logprobs_num=2),
            "top_row_short": dict(
                logprob=-0.5, top_logprobs_num=2, top=([-0.5], [11])
            ),
            "top_row_long": dict(
                logprob=-0.5, top_logprobs_num=1, top=([-0.5, -1.0], [11, 3])
            ),
            "top_ids_short": dict(
                logprob=-0.5, top_logprobs_num=2, top=([-0.5, -1.0], [11])
            ),
        }
        for name, kwargs in cases.items():
            with self.subTest(name):
                buf = _buffers()
                # A previous owner's WRITTEN row must survive the rejection.
                buf.set_buf(_prefill_req(return_logprob=True, logprob=-3.0))
                req = _prefill_req(return_logprob=True, **kwargs)
                if name == "value_without_token_id":
                    req.logprob.output_token_logprobs_idx = []
                before = _snapshot(buf)
                with self.assertRaises(InvalidDisaggregationMetadata):
                    buf.set_buf(req)
                for tensor_name, value in before.items():
                    self.assertTrue(
                        torch.equal(value, getattr(buf, tensor_name)), tensor_name
                    )


class TestDecodeCommitsFirstTokenLogprob(CustomTestCase):
    def setUp(self):
        world_size = patch("torch.distributed.get_world_size", return_value=1)
        world_size.start()
        self.addCleanup(world_size.stop)
        reset_context()
        self.addCleanup(reset_context)
        publish(ServerArgs(model_path="dummy"), role="tokenizer")

    def _queue(self, metadata_buffers):
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
        queue.metadata_buffers = metadata_buffers
        return queue

    def _req(self, *, top_logprobs_num=0, bootstrap_host="prefill"):
        params = SamplingParams()
        params.normalize(None)
        return Req(
            "first-token-logprob",
            "",
            [1, 2, 3],
            params,
            return_logprob=True,
            top_logprobs_num=top_logprobs_num,
            bootstrap_host=bootstrap_host,
            bootstrap_room=7,
            vocab_size=2048,
        )

    def _commit(self, metadata_buffers, req, *, is_rebootstrap=False):
        self._queue(metadata_buffers)._commit_transfer_to_req(
            DecodeRequest(
                req,
                SimpleNamespace(clear=lambda: None, failure_exception=lambda: None),
                metadata_buffer_index=0,
                is_rebootstrap=is_rebootstrap,
            )
        )
        return req.logprob

    def _row(self, state, *, logprob_token_id=12):
        """A received row: token 11 sampled, logprob rows holding -0.5 for
        ``logprob_token_id`` and a top row, whatever their last writer left."""
        cached_tokens = torch.zeros(16, dtype=torch.int32)
        cached_tokens[FIRST_TOKEN_LOGPROB_SLOT] = state
        return SimpleNamespace(
            get_buf=lambda idx: (
                torch.tensor([11]),
                cached_tokens,
                torch.tensor([-0.5]),
                torch.tensor([logprob_token_id]),
                torch.tensor([-0.5, -2.0]),
                torch.tensor([logprob_token_id, 30], dtype=torch.int32),
                *([None] * 7),
                torch.tensor([7]),
            ),
            get_customized_info=lambda idx: None,
        )

    def _assert_missing(self, logprob, token_id, *, top_requested):
        # One row per output token, so later positions stay aligned.
        self.assertEqual(len(logprob.output_token_logprobs_val), 1)
        self.assertTrue(math.isnan(logprob.output_token_logprobs_val[0]))
        self.assertEqual(logprob.output_token_logprobs_idx, [token_id])
        # No top candidates are made up for a missing token.
        expected_top = [[]] if top_requested else []
        self.assertEqual(logprob.output_top_logprobs_val, expected_top)
        self.assertEqual(logprob.output_top_logprobs_idx, expected_top)

    def test_written_row_is_committed(self):
        req = self._req(top_logprobs_num=2)
        logprob = self._commit(
            self._row(FIRST_TOKEN_LOGPROB_WRITTEN, logprob_token_id=11), req
        )
        self.assertEqual(list(req.output_ids), [11])
        self.assertEqual(logprob.output_token_logprobs_val, [-0.5])
        self.assertEqual(logprob.output_token_logprobs_idx, [11])
        self.assertEqual(logprob.output_top_logprobs_val, [[-0.5, -2.0]])
        self.assertEqual(logprob.output_top_logprobs_idx, [[11, 30]])

    def test_unwritten_rows_are_committed_as_unknown_even_for_the_same_token(self):
        # logprob_token_id=11 is the sampled token, so the stale row would
        # pass the serving layer's token-ID check if decode reported it.
        for state in (
            FIRST_TOKEN_LOGPROB_UNKNOWN,
            FIRST_TOKEN_LOGPROB_ABSENT,
            # Any other value is not a WRITTEN row either.
            7,
        ):
            for top_logprobs_num in (0, 2):
                with self.subTest(state=state, top_logprobs_num=top_logprobs_num):
                    req = self._req(top_logprobs_num=top_logprobs_num)
                    logprob = self._commit(
                        self._row(state, logprob_token_id=11), req
                    )
                    self.assertEqual(list(req.output_ids), [11])
                    self._assert_missing(
                        logprob, 11, top_requested=top_logprobs_num > 0
                    )

    def test_fake_transfer_never_reads_the_row(self):
        req = self._req(top_logprobs_num=2, bootstrap_host=FAKE_BOOTSTRAP_HOST)
        logprob = self._commit(
            self._row(FIRST_TOKEN_LOGPROB_WRITTEN, logprob_token_id=11), req
        )
        self.assertEqual(len(req.output_ids), 1)
        self._assert_missing(logprob, req.output_ids[0], top_requested=True)

    def test_rebootstrap_commits_no_new_logprob(self):
        req = self._req()
        req.output_ids.extend([11, 13])
        logprob = self._commit(
            self._row(FIRST_TOKEN_LOGPROB_WRITTEN, logprob_token_id=11),
            req,
            is_rebootstrap=True,
        )
        self.assertEqual(list(req.output_ids), [11, 13])
        self.assertEqual(logprob.output_token_logprobs_val, [])

    def test_reused_row_end_to_end(self):
        """Prefill rows copied as the transfer does, then committed on decode."""

        def transfer(prefill_req, *, legacy_prefill=False):
            prefill = _buffers()
            # Previous owner of the row: same sampled token, real logprob.
            prefill.set_buf(
                _prefill_req(
                    return_logprob=True,
                    logprob=-0.5,
                    top_logprobs_num=2,
                    top=([-0.5, -2.0], [11, 30]),
                )
            )
            prefill.set_buf(prefill_req)
            if legacy_prefill:
                prefill.cached_tokens[0][FIRST_TOKEN_LOGPROB_SLOT] = 0
            decode = _buffers()
            for name, value in _snapshot(prefill).items():
                getattr(decode, name).copy_(value)
            return decode

        cases = {
            "prefill_computed_no_logprob": (
                _prefill_req(return_logprob=False),
                False,
            ),
            "legacy_prefill_wrote_no_state": (
                _prefill_req(return_logprob=False),
                True,
            ),
        }
        for name, (prefill_req, legacy) in cases.items():
            with self.subTest(name):
                req = self._req(top_logprobs_num=2)
                decode = transfer(prefill_req, legacy_prefill=legacy)
                logprob = self._commit(decode, req)
                self.assertEqual(list(req.output_ids), [11])
                self._assert_missing(logprob, 11, top_requested=True)

        req = self._req(top_logprobs_num=2)
        logprob = self._commit(
            transfer(
                _prefill_req(
                    return_logprob=True,
                    logprob=-0.25,
                    top_logprobs_num=2,
                    top=([-0.25, -1.5], [11, 40]),
                )
            ),
            req,
        )
        self.assertEqual(logprob.output_token_logprobs_val, [-0.25])
        self.assertEqual(logprob.output_token_logprobs_idx, [11])
        self.assertEqual(logprob.output_top_logprobs_val, [[-0.25, -1.5]])
        self.assertEqual(logprob.output_top_logprobs_idx, [[11, 40]])


if __name__ == "__main__":
    unittest.main()
