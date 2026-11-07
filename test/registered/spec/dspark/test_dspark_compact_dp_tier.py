"""DSpark compact verify: the per-DP verify tier and budget ride the MLP sync.

Covers the scheduler-side half of the DP tier (local report, rank-0 selection,
budget adoption) and the worker's gating. GPU parity (greedy static vs compact)
lives with the GPU suites.
"""

import random
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.scheduler_components import dp_attn  # noqa: E402
from sglang.srt.model_executor.forward_batch_info import ForwardMode  # noqa: E402
from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (  # noqa: E402
    DecodeCudaGraphRunner,
    ragged_verify_slot_buckets,
)
from sglang.srt.speculative.dspark_components import dspark_planner  # noqa: E402
from sglang.srt.speculative.dspark_components.dspark_planner import (  # noqa: E402
    DSparkVerifyPlanner,
    dp_global_verify_tier_num_tokens,
    dp_rank0_verify_budget,
    local_verify_tier_num_tokens,
)
from sglang.srt.speculative.dspark_components.dspark_worker_v2 import (  # noqa: E402
    DSparkWorkerV2,
)
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

_WIDTH = 6  # gamma + 1


def _planner(*, dp_tier: bool = True, verify_all: bool = False, budget=10):
    planner = DSparkVerifyPlanner.__new__(DSparkVerifyPlanner)
    planner.verify_num_draft_tokens = _WIDTH
    planner._schedule_cfg = SimpleNamespace(min_verify_len=1)
    planner._budget_planner = Mock()
    planner._dp_tier_gather_enabled = dp_tier
    planner._is_verify_all = verify_all
    planner._dp_budget_override_count = 0
    planner._budget_from_resolved = Mock(return_value=budget)
    return planner


def _batch(bs: int = 8, mode: ForwardMode = ForwardMode.DECODE):
    return SimpleNamespace(
        spec_info=SimpleNamespace(verify_token_budget="unset"),
        forward_mode=mode,
        is_extend_in_batch=False,
        batch_size=lambda: bs,
        req_pool_indices_cpu=torch.arange(bs),
        spec_verify_tier_num_tokens=-1,
        spec_verify_token_budget_presync=None,
        spec_verify_budget_presync=False,
        global_spec_verify_tier_num_tokens=None,
        global_spec_verify_token_budget_enc=None,
    )


class TestPreSyncBudget(CustomTestCase):
    def test_records_budget_and_local_tier(self):
        planner, batch = _planner(budget=10), _batch(bs=8)
        planner.prepare_verify_budget_pre_sync(batch, Mock())
        self.assertTrue(batch.spec_verify_budget_presync)
        self.assertEqual(batch.spec_info.verify_token_budget, 10)
        self.assertEqual(batch.spec_verify_token_budget_presync, 10)
        self.assertEqual(batch.spec_verify_tier_num_tokens, 8 + 10)

    def test_noop_without_dp_tier(self):
        planner, batch = _planner(dp_tier=False), _batch()
        planner.prepare_verify_budget_pre_sync(batch, Mock())
        self.assertFalse(batch.spec_verify_budget_presync)
        planner._budget_from_resolved.assert_not_called()

    def test_noop_on_non_decode(self):
        for mode in (ForwardMode.EXTEND, ForwardMode.IDLE):
            planner, batch = _planner(), _batch(mode=mode)
            planner.prepare_verify_budget_pre_sync(batch, Mock())
            self.assertFalse(batch.spec_verify_budget_presync)

    def test_verify_all_reports_uniform_tier(self):
        # Verify-all serves the uniform layout, so the reported tier must be
        # bs * width or idle DP ranks would replay a smaller graph.
        planner, batch = _planner(verify_all=True, budget=3), _batch(bs=5)
        planner.prepare_verify_budget_pre_sync(batch, Mock())
        self.assertEqual(batch.spec_verify_tier_num_tokens, 5 * _WIDTH)

    def test_missing_budget_reports_sentinel(self):
        planner, batch = _planner(budget=None), _batch()
        planner.prepare_verify_budget_pre_sync(batch, Mock())
        self.assertEqual(batch.spec_verify_tier_num_tokens, -1)
        self.assertEqual(dp_attn.local_spec_verify_tier(batch), (-1, 0))


class TestRunBatchBudget(CustomTestCase):
    def _gathered(self, batch, *, tiers, budgets_enc):
        batch.global_spec_verify_tier_num_tokens = tiers
        batch.global_spec_verify_token_budget_enc = budgets_enc

    def test_adopts_rank0_budget_and_tier(self):
        planner, batch = _planner(budget=10), _batch(bs=8)
        planner.prepare_verify_budget_pre_sync(batch, Mock())
        # This rank sits in DP 2, whose attention-TP rank 0 saw budget 4.
        self._gathered(batch, tiers=[18, 0, 12, 30], budgets_enc=[11, 0, 5, 23])
        with patch.object(
            dspark_planner, "get_parallel", return_value=SimpleNamespace(attn_dp_rank=2)
        ):
            planner.prepare_verify_budget(batch, Mock())
        self.assertEqual(batch.spec_info.verify_token_budget, 4)
        self.assertEqual(batch.spec_verify_tier_num_tokens, 12)
        self.assertFalse(batch.spec_verify_budget_presync)
        self.assertEqual(planner._dp_budget_override_count, 1)
        planner._budget_planner.note_non_decode_step.assert_called_once()

    def test_rank0_without_budget_wins(self):
        planner, batch = _planner(budget=10), _batch(bs=8)
        planner.prepare_verify_budget_pre_sync(batch, Mock())
        self._gathered(batch, tiers=[-1, 18], budgets_enc=[0, 11])
        with patch.object(
            dspark_planner, "get_parallel", return_value=SimpleNamespace(attn_dp_rank=0)
        ):
            planner.prepare_verify_budget(batch, Mock())
        self.assertIsNone(batch.spec_info.verify_token_budget)
        self.assertEqual(batch.spec_verify_tier_num_tokens, -1)

    def test_matching_budget_is_not_an_override(self):
        planner, batch = _planner(budget=10), _batch(bs=8)
        planner.prepare_verify_budget_pre_sync(batch, Mock())
        self._gathered(batch, tiers=[18, 6], budgets_enc=[11, 1])
        with patch.object(
            dspark_planner, "get_parallel", return_value=SimpleNamespace(attn_dp_rank=0)
        ):
            planner.prepare_verify_budget(batch, Mock())
        self.assertEqual(batch.spec_info.verify_token_budget, 10)
        self.assertEqual(planner._dp_budget_override_count, 0)
        planner._budget_planner.note_non_decode_step.assert_not_called()

    def test_decode_without_pre_sync_is_verify_all(self):
        planner, batch = _planner(budget=10), _batch()
        planner.prepare_verify_budget(batch, Mock())
        planner._budget_from_resolved.assert_not_called()
        self.assertIsNone(batch.spec_info.verify_token_budget)
        self.assertEqual(batch.spec_verify_tier_num_tokens, -1)

    def test_extend_clears_stale_pre_sync_flag(self):
        planner, batch = _planner(), _batch()
        batch.spec_verify_budget_presync = True
        batch.forward_mode = ForwardMode.EXTEND
        planner.prepare_verify_budget(batch, Mock())
        self.assertFalse(batch.spec_verify_budget_presync)
        planner._budget_planner.note_non_decode_step.assert_called_once()

    def test_without_dp_tier_computes_locally(self):
        planner, batch = _planner(dp_tier=False, budget=7), _batch(bs=4)
        planner.prepare_verify_budget(batch, Mock())
        self.assertEqual(batch.spec_info.verify_token_budget, 7)
        self.assertEqual(batch.spec_verify_tier_num_tokens, 4 + 7)


class TestMlpSyncCarriesTier(CustomTestCase):
    def test_local_report(self):
        self.assertEqual(dp_attn.local_spec_verify_tier(None), (0, 0))
        idle = SimpleNamespace(forward_mode=ForwardMode.IDLE)
        self.assertEqual(dp_attn.local_spec_verify_tier(idle), (0, 0))
        not_prepared = SimpleNamespace(forward_mode=ForwardMode.DECODE)
        self.assertEqual(dp_attn.local_spec_verify_tier(not_prepared), (-1, 0))
        prepared = SimpleNamespace(
            forward_mode=ForwardMode.DECODE,
            spec_verify_budget_presync=True,
            spec_verify_token_budget_presync=5,
            spec_verify_tier_num_tokens=13,
        )
        self.assertEqual(dp_attn.local_spec_verify_tier(prepared), (13, 6))
        prepared.spec_verify_token_budget_presync = None
        self.assertEqual(dp_attn.local_spec_verify_tier(prepared), (13, 0))

    def test_budget_encoding_round_trip(self):
        self.assertIsNone(dp_rank0_verify_budget(budget_enc=[0, 7], dp_rank=0))
        self.assertEqual(dp_rank0_verify_budget(budget_enc=[0, 1], dp_rank=1), 0)
        self.assertEqual(dp_rank0_verify_budget(budget_enc=[0, 7], dp_rank=1), 6)

    def test_neutral_row_equals_fallback(self):
        # An idle rank must still look like an inactive slot to the elastic
        # all-reduce gather, which detects missing ranks by an all-zero row.
        info = _info(num_tokens=0, tier=0, budget_enc=0)
        local = info._get_local_tensor(device="cpu")
        fallback = info._get_fallback_tensor(device="cpu")
        self.assertEqual(local.numel(), fallback.numel())
        self.assertEqual(int(local[-2:].abs().sum()), 0)
        self.assertEqual(int(fallback[-2:].abs().sum()), 0)

    def test_gather_reads_each_dp_attention_tp_rank0(self):
        dp_size, attn_tp = 2, 4
        rows = []
        for dp in range(dp_size):
            for tp in range(attn_tp):
                # Ranks 1..3 report something else: only rank 0 may count.
                tier = 18 + dp if tp == 0 else 999
                budget_enc = 11 + dp if tp == 0 else 0
                rows.append(_info(num_tokens=8, tier=tier, budget_enc=budget_enc))
        gathered_info = _all_gather(rows, dp_size=dp_size, attn_tp=attn_tp)
        self.assertEqual(gathered_info.global_spec_verify_tier_num_tokens, [18, 19])
        self.assertEqual(gathered_info.global_spec_verify_token_budget_enc, [11, 12])

        batch = SimpleNamespace()
        dp_attn._update_gather_batch(batch, gathered_info, require_mlp_tp_gather=True)
        self.assertEqual(batch.global_spec_verify_tier_num_tokens, [18, 19])
        self.assertEqual(batch.global_spec_verify_token_budget_enc, [11, 12])


class TestRankAgreementUnderRacyBudgets(CustomTestCase):
    def test_every_rank_ends_on_one_tier_and_dp_local_budget(self):
        # Each rank's host may or may not see the relayed confidence (event
        # query), so local budgets can differ inside a DP. After the MLP sync
        # every rank of a DP must hold the same budget, and every rank of every
        # DP the same global tier.
        rng = random.Random(1101)
        for _ in range(300):
            dp_size, attn_tp = rng.randint(1, 8), rng.choice([1, 2, 4])
            bs_per_dp = [rng.choice([0, rng.randint(1, 64)]) for _ in range(dp_size)]
            batches, rows = [], []
            for dp in range(dp_size):
                for _tp in range(attn_tp):
                    bs = bs_per_dp[dp]
                    if bs == 0:
                        batches.append(None)
                        rows.append(_info(num_tokens=0, tier=0, budget_enc=0))
                        continue
                    budget = rng.choice([None, rng.randint(0, bs * (_WIDTH - 1))])
                    planner, batch = _planner(budget=budget), _batch(bs=bs)
                    planner.prepare_verify_budget_pre_sync(batch, Mock())
                    tier, budget_enc = dp_attn.local_spec_verify_tier(batch)
                    batches.append((planner, batch))
                    rows.append(_info(num_tokens=bs, tier=tier, budget_enc=budget_enc))
            gathered_info = _all_gather(rows, dp_size=dp_size, attn_tp=attn_tp)
            global_tier = dp_global_verify_tier_num_tokens(
                global_tier_num_tokens=gathered_info.global_spec_verify_tier_num_tokens
            )
            for index, entry in enumerate(batches):
                if entry is None:
                    continue
                dp = index // attn_tp
                planner, batch = entry
                dp_attn._update_gather_batch(
                    batch, gathered_info, require_mlp_tp_gather=True
                )
                with patch.object(
                    dspark_planner,
                    "get_parallel",
                    return_value=SimpleNamespace(attn_dp_rank=dp),
                ):
                    planner.prepare_verify_budget(batch, Mock())
                rank0_planner, rank0_batch = batches[dp * attn_tp]
                self.assertEqual(
                    batch.spec_info.verify_token_budget,
                    rank0_batch.spec_info.verify_token_budget,
                )
                self.assertEqual(
                    dp_global_verify_tier_num_tokens(
                        global_tier_num_tokens=batch.global_spec_verify_tier_num_tokens
                    ),
                    global_tier,
                )
            if global_tier is not None:
                # The shared tier covers every DP's packed verify rows.
                for dp in range(dp_size):
                    if bs_per_dp[dp] == 0:
                        continue
                    _, rank0_batch = batches[dp * attn_tp]
                    budget = rank0_batch.spec_info.verify_token_budget
                    self.assertLessEqual(
                        local_verify_tier_num_tokens(
                            bs=bs_per_dp[dp],
                            verify_token_budget=budget,
                            verify_num_draft_tokens=_WIDTH,
                            min_verify_len=1,
                        ),
                        global_tier,
                    )


class TestWorkerDpTierGate(CustomTestCase):
    def _worker(self, *, dp_synced=True, gather=True):
        worker = DSparkWorkerV2.__new__(DSparkWorkerV2)
        worker._dp_synced_verify = dp_synced
        worker._verify_planner = SimpleNamespace(
            dp_tier_gather_enabled=gather, is_compact_mode=True
        )
        return worker

    def _batch(self, **overrides):
        fields = dict(
            global_num_tokens=[8, 0, 5],
            global_spec_verify_tier_num_tokens=[18, 0, 12],
            is_extend_in_batch=False,
            can_run_decode_cuda_graph=True,
        )
        fields.update(overrides)
        return SimpleNamespace(**fields)

    def test_shared_tier_is_the_dp_max(self):
        self.assertEqual(self._worker()._dp_verify_tier_num_tokens(self._batch()), 18)

    def test_off_cases(self):
        cases = [
            (self._worker(dp_synced=False), self._batch()),
            (self._worker(gather=False), self._batch()),
            (self._worker(), self._batch(is_extend_in_batch=True)),
            (self._worker(), self._batch(can_run_decode_cuda_graph=False)),
            (self._worker(), self._batch(global_num_tokens=None)),
            (self._worker(), self._batch(global_spec_verify_tier_num_tokens=[18, -1])),
        ]
        for worker, batch in cases:
            self.assertIsNone(worker._dp_verify_tier_num_tokens(batch))

    def test_eager_fallback_only_needs_uniform_width_for_synced_verify(self):
        batch = self._batch(can_run_decode_cuda_graph=False)
        self.assertTrue(self._worker()._use_uniform_eager_verify(batch))
        self.assertFalse(self._worker(dp_synced=False)._use_uniform_eager_verify(batch))
        self.assertFalse(self._worker()._use_uniform_eager_verify(self._batch()))

    def test_idle_rank_follows_eager_vote(self):
        worker = self._worker()
        batch = self._batch(can_run_decode_cuda_graph=False)
        self.assertIsNone(worker._idle_verify_ragged_layout(batch))

    def test_full_coverage_replay_bounds_every_query_on_busy_and_idle_ranks(self):
        device = torch.device("cpu")
        # The 40-request DP needs 240 tokens. The 63-request DP only verifies
        # its anchors, so 177 tokens must fit in dummy rows on that rank.
        counts = [63, 40, 0]
        batch = self._batch(
            global_num_tokens=counts, global_spec_verify_tier_num_tokens=[63, 240, 0]
        )
        for max_bs, expected_shape in ((256, (128, 240)), (64, (64, 384))):
            with self.subTest(max_bs=max_bs):
                runner = DecodeCudaGraphRunner.__new__(DecodeCudaGraphRunner)
                runner.ragged_verify_mode = True
                runner.reserve_ragged_padding_slot = True
                runner.captured_req_width = _WIDTH
                runner.max_bs = max_bs
                runner.capture_num_tokens = sorted({240, 384, max_bs * _WIDTH})
                runner._ragged_slots_by_tokens = {
                    g: ragged_verify_slot_buckets(g, _WIDTH, max_bs)
                    for g in runner.capture_num_tokens
                }
                worker = self._worker()
                worker.device = device
                worker.verify_num_draft_tokens = _WIDTH
                worker.model_runner = SimpleNamespace(decode_cuda_graph_runner=runner)
                planner = _planner()
                planner.model_runner = worker.model_runner
                planner._ragged_verify_mode = dspark_planner.RaggedVerifyMode.COMPACT
                planner._align_verify_tokens_to_graph_tier = False

                idle = worker._idle_verify_ragged_layout(batch)
                rows = [(idle, counts[0], 1)]
                tier = worker._dp_verify_tier_num_tokens(batch)
                for bs, length in ((63, 1), (40, _WIDTH)):
                    planner._schedule_verify_lens = Mock(
                        return_value=torch.full((bs,), length, dtype=torch.int32)
                    )
                    active = planner.schedule_layout(
                        req_pool_indices=torch.arange(bs),
                        prefix_lens=torch.ones(bs),
                        device=device,
                        confidence=None,
                        budget=bs * (length - 1),
                        global_num_reqs=max(counts),
                        dp_tier_num_tokens=tier,
                    )
                    rows.append((active, bs, length))

                self.assertIn(expected_shape, runner._capture_sizes())
                for layout, bs, length in rows:
                    shape = runner._ragged_replay_shape(layout)
                    self.assertIsNotNone(shape)
                    slots, tokens = shape
                    padded = layout.padded_to_bucket(padded_bs=slots)
                    lengths = padded.qo_indptr_device.diff().tolist()
                    # If the capture cannot fit the slack at G=240, all DPs
                    # switch to full width, including the synthetic idle rows.
                    expected_length = _WIDTH if tokens == 384 else length
                    self.assertEqual(lengths[:bs], [expected_length] * bs)
                    self.assertEqual(len(lengths), slots)
                    self.assertEqual(sum(lengths), tokens)
                    self.assertGreaterEqual(min(lengths), 0)
                    self.assertLessEqual(max(lengths), _WIDTH)
                    self.assertEqual(shape, expected_shape)

    def test_uneven_active_and_idle_ranks_share_graph_dimensions(self):
        worker = self._worker()
        worker.device = torch.device("cpu")
        worker.verify_num_draft_tokens = _WIDTH
        worker.model_runner = SimpleNamespace(
            decode_cuda_graph_runner=SimpleNamespace(
                ragged_verify_mode=True,
                capture_num_tokens=[60, 192, 240, 360],
                max_bs=60,
            )
        )
        planner = _planner()
        planner.model_runner = worker.model_runner
        planner._ragged_verify_mode = dspark_planner.RaggedVerifyMode.COMPACT
        planner._dynamic_graph_tier = False
        planner._align_verify_tokens_to_graph_tier = True
        counts = [57, 60, 0]
        for reserve, tiers, expected_tokens in (
            (False, [171, 239, 0], 240),
            (False, [171, -1, 0], 360),
            # At capacity, a backend that needs a ghost slot must use a
            # full-width layout on every rank, even one with fewer requests.
            (True, [171, 239, 0], 360),
            # The largest DP verifies exactly one anchor per request.
            (True, [57, 60, 0], 60),
        ):
            with self.subTest(reserve=reserve, tiers=tiers):
                worker.model_runner.decode_cuda_graph_runner.reserve_ragged_padding_slot = reserve
                batch = self._batch(
                    global_num_tokens=counts, global_spec_verify_tier_num_tokens=tiers
                )
                shared_tier = worker._dp_verify_tier_num_tokens(batch)
                layouts = [worker._idle_verify_ragged_layout(batch)]
                for bs in counts[:2]:
                    planner._schedule_verify_lens = Mock(
                        return_value=(
                            torch.full(
                                (bs,), 1 if shared_tier == 60 else 3, dtype=torch.int32
                            )
                            if shared_tier is not None
                            else None
                        )
                    )
                    layouts.append(
                        planner.schedule_layout(
                            req_pool_indices=torch.arange(bs),
                            prefix_lens=torch.ones(bs),
                            device=worker.device,
                            confidence=None,
                            budget=2 * bs if shared_tier is not None else None,
                            global_num_reqs=max(counts),
                            dp_tier_num_tokens=shared_tier,
                        )
                    )
                self.assertEqual(
                    {
                        (layout.graph_max_bs, layout.graph_num_tokens)
                        for layout in layouts
                    },
                    {(60, expected_tokens)},
                )
                if expected_tokens == 360:
                    for layout in layouts:
                        self.assertEqual(
                            layout.verify_lens.tolist(), [_WIDTH] * layout.bs
                        )


def _info(*, num_tokens: int, tier: int, budget_enc: int) -> dp_attn.MLPSyncBatchInfo:
    return dp_attn.MLPSyncBatchInfo(
        dp_size=1,
        tp_size=1,
        cp_size=1,
        num_tokens=num_tokens,
        num_tokens_for_logprob=num_tokens,
        can_run_decode_cuda_graph=True,
        can_run_draft_cuda_graph=True,
        can_run_prefill_cuda_graph=False,
        is_extend_in_batch=False,
        local_can_run_tbo=False,
        local_forward_mode=ForwardMode.DECODE.value,
        spec_verify_tier_num_tokens=tier,
        spec_verify_token_budget_enc=budget_enc,
    )


def _all_gather(rows, *, dp_size: int, attn_tp: int) -> dp_attn.MLPSyncBatchInfo:
    """Run MLPSyncBatchInfo.all_gather with every rank's row pre-filled."""
    stacked = torch.stack([row._get_local_tensor(device="cpu") for row in rows])

    def fake_all_gather_single(output, local, group=None):
        output.copy_(stacked.flatten())

    info = rows[0]
    info.dp_size, info.tp_size, info.cp_size = dp_size, attn_tp, 1
    parallel = SimpleNamespace(
        tp_group=SimpleNamespace(
            active_ranks_cpu=torch.ones(dp_size * attn_tp, dtype=torch.int64)
        )
    )
    with (
        patch.object(dp_attn, "all_gather_single", fake_all_gather_single),
        patch.object(dp_attn, "get_parallel", return_value=parallel),
    ):
        info.all_gather(device="cpu", group=None)
    return info


if __name__ == "__main__":
    unittest.main()
