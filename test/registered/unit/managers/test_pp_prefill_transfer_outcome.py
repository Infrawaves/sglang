"""PP consensus on the outcome of PD prefill KV transfers.

Each PP stage of a disaggregated prefill sends the KV of its own layers. A
client abort that lands after the first stage finished sending fails only the
later stages, so the stages must agree not only on which transfers concluded
but also on whether each one failed.
"""

import unittest
from types import SimpleNamespace

from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _stage(first_rank: bool, succeeded, failed, prev=None):
    """A PP stage whose own senders report `succeeded` / `failed` rids."""
    # The stage's local outcome comes from the source-drain gate, which reports
    # a concluded transfer as Success / Failed and anything else as in flight.
    reqs = [SimpleNamespace(rid=rid) for rid in [*succeeded, *failed]]
    return SimpleNamespace(
        pp_group=SimpleNamespace(is_first_rank=first_rank),
        disagg_prefill_transfer_queue=reqs,
        poll_prefill_transfer_releasable=lambda queue: [
            KVPoll.Success if req.rid in succeeded else KVPoll.Failed
            for req in queue
        ],
        _pp_recv_pyobj_from_prev_stage=lambda: prev,
    )


def _consensus(stage):
    return SchedulerPPMixin._pp_pd_get_prefill_transferred_ids(stage)


class TestPPPrefillTransferOutcomeConsensus(CustomTestCase):
    def test_first_stage_reports_its_own_outcomes(self):
        stage = _stage(first_rank=True, succeeded=["a"], failed=["b"])
        self.assertEqual(_consensus(stage), {"a": False, "b": True})

    def test_failure_on_any_stage_fails_the_transfer_on_every_stage(self):
        # Stage 0 finished sending before the abort; stage 1 had not.
        stage1 = _stage(
            first_rank=False,
            succeeded=["c"],
            failed=["r"],
            prev={"r": False, "c": False},
        )
        self.assertEqual(_consensus(stage1), {"r": True, "c": False})
        # The failure is carried on even where this stage itself succeeded.
        stage2 = _stage(first_rank=False, succeeded=["r"], failed=[], prev={"r": True})
        self.assertEqual(_consensus(stage2), {"r": True})

    def test_only_transfers_concluded_on_every_stage_are_released(self):
        # "a" has not concluded on the previous stages yet.
        stage = _stage(
            first_rank=False, succeeded=["a", "b"], failed=[], prev={"b": False}
        )
        self.assertEqual(_consensus(stage), {"b": False})

    def test_unconcluded_on_this_stage_is_held_back(self):
        # The previous stages concluded "a", this stage's sender is still transferring.
        stage = _stage(first_rank=False, succeeded=[], failed=[], prev={"a": True})
        self.assertEqual(_consensus(stage), {})


if __name__ == "__main__":
    unittest.main()
