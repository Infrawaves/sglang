"""CPU tests for the EPLB latency changes.

* same-node CUDA-IPC weight moves (planning + barrier ordering, with the
  CUDA copy faked) and the dense flat P2P views;
* expert balancedness gathered once per window;
* DeepEP dispatch-side `stat` counts and the fused-count buffer gating.
"""

import random
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.eplb import expert_location_updater as updater
from sglang.srt.eplb.expert_distribution import (
    _DeepepDispatchCountSinglePassGatherer,
    _ExpertDistributionRecorderReal,
    _SelectExpertsSinglePassGatherer,
    _UtilizationRateAccumulatorMixin,
    compute_gpu_physical_count,
    compute_utilization_rate,
)
from sglang.srt.eplb.ipc_weight_transfer import dense_flat_view, p2p_tensor
from sglang.srt.utils import Withable
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


# ---------------------------------------------------------------------------
# dense flat views
# ---------------------------------------------------------------------------


class TestDenseFlatView(unittest.TestCase):
    def test_contiguous(self):
        t = torch.arange(24).reshape(2, 3, 4)
        flat = dense_flat_view(t[1])
        self.assertTrue(torch.equal(flat, t[1].reshape(-1)))

    def test_mn_major_slice_is_dense_but_not_contiguous(self):
        # [experts, mn, k] stored MN-major per expert (like a MegaMoE scale).
        experts, mn, k = 3, 8, 5
        base = torch.arange(experts * mn * k, dtype=torch.int32)
        t = base.as_strided((experts, mn, k), (mn * k, 1, mn))
        expert = t[2]
        self.assertFalse(expert.is_contiguous())
        flat = dense_flat_view(expert)
        self.assertIsNotNone(flat)
        self.assertEqual(flat.numel(), mn * k)
        self.assertEqual(flat.data_ptr(), expert.data_ptr())
        self.assertTrue(torch.equal(flat, base[2 * mn * k : 3 * mn * k]))

    def test_gapped_slice_is_rejected(self):
        t = torch.zeros(4, 10)[:, :6]
        self.assertIsNone(dense_flat_view(t))
        self.assertIs(p2p_tensor(t), t)


# ---------------------------------------------------------------------------
# rebalance weight moves
# ---------------------------------------------------------------------------


class _FakeP2POp:
    """torch's P2POp validates against a live process group; these tests
    only need the op's fields."""

    def __init__(self, op, tensor, peer):
        self.op = op
        self.tensor = tensor
        self.peer = peer


class _FakeIpcPeers:
    """IpcPeerWeights stand-in: pulls copy from the peers' old weights."""

    def __init__(self, old_weights_of_rank, num_gpu_per_node, rank, events):
        self._old = old_weights_of_rank
        self._num_gpu_per_node = num_gpu_per_node
        self._rank = rank
        self._events = events
        self.pulled_bytes = 0

    def is_same_node(self, peer):
        return peer // self._num_gpu_per_node == self._rank // self._num_gpu_per_node

    def pull(self, *, layer_id, dst_tensors, src_rank, src_local_index):
        self._events.append(("pull", self._rank, src_rank))
        for i, dst in enumerate(dst_tensors):
            dst.copy_(self._old[src_rank][i][src_local_index])
            self.pulled_bytes += dst.numel() * dst.element_size()
        return self.pulled_bytes


def _make_weights(physical_to_logical, rank, num_local):
    """Two tensors per rank whose rows encode the logical expert id."""
    rows = physical_to_logical[rank * num_local : (rank + 1) * num_local]
    w0 = torch.tensor(rows, dtype=torch.float32)[:, None].repeat(1, 3)
    # MN-major second tensor, like a MegaMoE scale.
    w1_base = torch.tensor(rows, dtype=torch.int32)[:, None].repeat(1, 6)
    w1 = w1_base.as_strided((num_local, 2, 3), (6, 1, 2))
    return [w0, w1]


def _run_rebalance(*, num_ranks, num_gpu_per_node, num_local, old_map, new_map):
    events = []
    weights = {r: _make_weights(old_map, r, num_local) for r in range(num_ranks)}
    old_snapshot = {r: [t.clone() for t in ws] for r, ws in weights.items()}
    # clone() of the strided tensor keeps its layout (dense, non-contiguous).
    for r in range(num_ranks):
        old_snapshot[r][1] = (
            old_snapshot[r][1]
            .as_strided(weights[r][1].shape, weights[r][1].stride())
            .copy_(weights[r][1])
        )

    def fake_batch_isend_irecv(ops):
        events.append(("nccl", ops))
        return []

    for rank in range(num_ranks):
        ipc = _FakeIpcPeers(old_snapshot, num_gpu_per_node, rank, events)
        with (
            patch.object(
                updater.torch.distributed, "batch_isend_irecv", fake_batch_isend_irecv
            ),
            patch.object(updater, "P2POp", _FakeP2POp),
            patch.object(
                updater, "gpu_barrier", lambda: events.append(("barrier", rank))
            ),
            patch.object(updater.ElasticEPStateManager, "instance", return_value=None),
        ):
            updater.update_expert_weights_single_layer(
                routed_experts_weights=weights[rank],
                temp_buffers=updater.create_temp_buffers(weights[rank]),
                old_physical_to_logical_map=old_map,
                new_physical_to_logical_map=new_map,
                num_local_physical_experts=num_local,
                num_gpu_per_node=num_gpu_per_node,
                rank=rank,
                world_size=num_ranks,
                ipc_peer_weights=ipc,
                layer_id=0,
            )
    return weights, events


class TestIpcRebalance(unittest.TestCase):
    def _check(self, weights, new_map, num_local):
        for rank, (w0, w1) in weights.items():
            expect = new_map[rank * num_local : (rank + 1) * num_local]
            self.assertEqual(w0[:, 0].tolist(), [float(x) for x in expect])
            # Every element of an expert slice carries its logical id.
            for i in range(num_local):
                self.assertTrue(torch.all(w1[i] == expect[i]), (rank, i))

    def test_single_node_moves_only_use_ipc(self):
        random.seed(0)
        num_ranks, num_local = 4, 4
        num_physical = num_ranks * num_local
        old_map = list(range(num_physical))
        new_map = old_map[:]
        random.shuffle(new_map)
        weights, events = _run_rebalance(
            num_ranks=num_ranks,
            num_gpu_per_node=num_ranks,
            num_local=num_local,
            old_map=old_map,
            new_map=new_map,
        )
        self._check(weights, new_map, num_local)
        self.assertFalse(any(e[0] == "nccl" for e in events))
        # Every rank hits the barrier once per layer, even with no pulls.
        self.assertEqual(sum(e[0] == "barrier" for e in events), num_ranks)

    def test_cross_node_moves_keep_nccl_without_same_node_sends(self):
        # 2 nodes x 2 GPUs (2 experts each): ranks 0,1 = node 0, ranks 2,3 =
        # node 1.  1<->2 moves within node 0 (rank 0 <-> rank 1); 3<->4 moves
        # across nodes (rank 1 <-> rank 2).
        num_ranks, num_gpu_per_node, num_local = 4, 2, 2
        old_map = [0, 1, 2, 3, 4, 5, 6, 7]
        new_map = [0, 2, 1, 4, 3, 5, 6, 7]
        _, events = _run_rebalance(
            num_ranks=num_ranks,
            num_gpu_per_node=num_gpu_per_node,
            num_local=num_local,
            old_map=old_map,
            new_map=new_map,
        )
        nccl_peers = {
            (e[1][0].peer if e[1] else None) for e in events if e[0] == "nccl"
        }
        nccl_ops = [op for e in events if e[0] == "nccl" for op in e[1]]
        pulls = [(e[1], e[2]) for e in events if e[0] == "pull"]
        # Same-node move: pulled over IPC in both directions.
        self.assertIn((0, 1), pulls)
        self.assertIn((1, 0), pulls)
        self.assertTrue(all(r // 2 == src // 2 for r, src in pulls))
        # Cross-node move stays on NCCL, and only between ranks 1 and 2.
        self.assertTrue(nccl_ops)
        self.assertTrue(all(op.peer in (1, 2) for op in nccl_ops), nccl_peers)
        self.assertTrue(all(op.tensor.dim() == 1 for op in nccl_ops))

    def test_flat_views_without_ipc(self):
        """NCCL path: P2P tensors are flat (no non-contiguous warning)."""
        num_ranks, num_local = 2, 2
        old_map = [0, 1, 2, 3]
        new_map = [2, 1, 0, 3]
        captured = []

        def fake_batch_isend_irecv(ops):
            captured.extend(ops)
            return []

        weights = {r: _make_weights(old_map, r, num_local) for r in range(num_ranks)}
        with (
            patch.object(
                updater.torch.distributed, "batch_isend_irecv", fake_batch_isend_irecv
            ),
            patch.object(updater, "P2POp", _FakeP2POp),
            patch.object(updater.ElasticEPStateManager, "instance", return_value=None),
        ):
            for rank in range(num_ranks):
                updater.update_expert_weights_single_layer(
                    routed_experts_weights=weights[rank],
                    temp_buffers=updater.create_temp_buffers(weights[rank]),
                    old_physical_to_logical_map=old_map,
                    new_physical_to_logical_map=new_map,
                    num_local_physical_experts=num_local,
                    num_gpu_per_node=num_ranks,
                    rank=rank,
                    world_size=num_ranks,
                )
        self.assertTrue(captured)
        self.assertTrue(all(op.tensor.dim() == 1 for op in captured))
        self.assertTrue(all(op.tensor.is_contiguous() for op in captured))


# ---------------------------------------------------------------------------
# balancedness window
# ---------------------------------------------------------------------------


def _make_accumulator(interval, *, world_size=2, rank=0, num_layers=3, ep=2):
    acc = object.__new__(_UtilizationRateAccumulatorMixin)
    acc._expert_location_metadata = SimpleNamespace(num_layers=num_layers, ep_size=ep)
    acc._enable = True
    acc._rank = rank
    history = []
    acc._history = SimpleNamespace(values=history, append=history.append)
    acc._reset_server_log_history = True
    acc._metric_heatmap_interval = 0
    acc._metric_heatmap_collection_counter = 0
    acc._balancedness_interval = interval
    acc._balancedness_staging = None
    acc._balancedness_gathered = None
    acc._balancedness_forward_pass_ids = []
    return acc


class TestBalancednessWindow(unittest.TestCase):
    def test_gathers_once_per_window_and_matches_per_pass_values(self):
        torch.manual_seed(0)
        world_size, interval, num_layers, ep, experts_per_gpu = 2, 4, 3, 2, 5
        steps = [
            [
                torch.randint(
                    0, 9, (num_layers, ep * experts_per_gpu), dtype=torch.int32
                )
                for _ in range(world_size)
            ]
            for _ in range(interval)
        ]
        accs = [
            _make_accumulator(
                interval, world_size=world_size, rank=r, num_layers=num_layers, ep=ep
            )
            for r in range(world_size)
        ]
        gather_calls = []

        def fake_all_gather(out, inp):
            # Each rank's staging is visible; emulate the collective.
            gather_calls.append(1)
            for r, acc in enumerate(accs):
                out[r].copy_(acc._balancedness_staging)

        outputs = [None] * interval
        env = SimpleNamespace(
            moe=SimpleNamespace(eplb_min_rebalancing_utilization_threshold=0.5)
        )
        with (
            patch(
                "sglang.srt.eplb.expert_distribution.torch.distributed.all_gather_into_tensor",
                fake_all_gather,
            ),
            patch(
                "sglang.srt.eplb.expert_distribution.torch.distributed.get_world_size",
                return_value=world_size,
            ),
            patch("sglang.srt.eplb.expert_distribution.get_exec", return_value=env),
            patch(
                "sglang.srt.eplb.expert_distribution.get_device_namespace",
                return_value=SimpleNamespace(device="cpu"),
            ),
            patch(
                "sglang.srt.eplb.expert_distribution.logs_expert_balancedness_to_server_log",
                return_value=False,
            ),
        ):
            for step_idx, per_rank in enumerate(steps):
                out = {}
                # Ranks append in lockstep; rank 1 goes first so its staging
                # is complete when the fake collective reads it.
                for r in (1, 0):
                    target = out if r == 0 else {}
                    accs[r]._append_utilization_rate(step_idx, per_rank[r], target)
                outputs[step_idx] = out

        # One all_gather per rank for the whole window, none per pass.
        self.assertEqual(len(gather_calls), world_size)
        self.assertTrue(all("metrics" not in o for o in outputs[:-1]))
        metrics = outputs[-1]["metrics"]
        expected = [
            torch.mean(
                compute_utilization_rate(
                    sum(
                        compute_gpu_physical_count(per_rank[r], num_gpu=ep)
                        for r in range(world_size)
                    )
                )
            ).item()
            for per_rank in steps
        ]
        self.assertEqual(metrics.forward_pass_id, interval - 1)
        self.assertAlmostEqual(
            metrics.eplb_balancedness.item(), sum(expected) / interval, places=5
        )
        for got, want in zip(accs[0]._history.values, expected):
            self.assertAlmostEqual(got, want, places=5)


# ---------------------------------------------------------------------------
# dispatch-side counts / fused count buffer
# ---------------------------------------------------------------------------


def _metadata(num_layers=2, num_physical=8, ep=2):
    return SimpleNamespace(
        num_layers=num_layers,
        num_physical_experts=num_physical,
        num_local_physical_experts=num_physical // ep,
        ep_size=ep,
    )


class TestDispatchSideCounts(unittest.TestCase):
    def _gatherer(self, rank):
        with patch(
            "sglang.srt.eplb.expert_distribution.get_device", return_value="cpu"
        ):
            return _DeepepDispatchCountSinglePassGatherer(_metadata(), rank)

    def test_normal_layout_and_low_latency_share_global_layout(self):
        g0, g1 = self._gatherer(0), self._gatherer(1)
        # Normal: senders report per-global-expert counts.
        g0.on_deepep_dispatch_normal_layout(0, torch.tensor([1, 0, 2, 0, 3, 0, 0, 1]))
        g1.on_deepep_dispatch_normal_layout(0, torch.tensor([0, 1, 0, 0, 0, 2, 1, 0]))
        # Low latency: owners report received counts of their local experts.
        g0.on_deepep_dispatch_low_latency(1, torch.tensor([4, 0, 1, 2]))
        g1.on_deepep_dispatch_low_latency(1, torch.tensor([0, 3, 0, 5]))
        total = (
            g0.collect()["global_physical_count"]
            + g1.collect()["global_physical_count"]
        )
        self.assertEqual(total[0].tolist(), [1, 1, 2, 0, 3, 2, 1, 1])
        self.assertEqual(total[1].tolist(), [4, 0, 1, 2, 0, 3, 0, 5])

    def test_recv_stats_view_aliases_local_slice(self):
        g1 = self._gatherer(1)
        view = g1.local_count_view(0)
        self.assertTrue(view.is_contiguous())
        view += torch.tensor([1, 2, 3, 4], dtype=torch.int32)
        self.assertEqual(
            g1.collect()["global_physical_count"][0].tolist(), [0, 0, 0, 0, 1, 2, 3, 4]
        )


class _FakeDeviceModule:
    capturing = False

    @classmethod
    def is_current_stream_capturing(cls):
        return cls.capturing


class TestFusedCountBufferGating(unittest.TestCase):
    def setUp(self):
        _FakeDeviceModule.capturing = False
        device_patch = patch(
            "sglang.srt.eplb.expert_distribution.torch.get_device_module",
            return_value=_FakeDeviceModule,
        )
        device_patch.start()
        self.addCleanup(device_patch.stop)

    def _recorder(self, gatherer):
        rec = object.__new__(_ExpertDistributionRecorderReal)
        rec._disable_all = False
        rec._recording = True
        rec._record_current_pass = True
        rec._current_layer_idx = Withable()
        rec._current_debug_name = Withable()
        rec._accumulator = SimpleNamespace(
            get_single_pass_gatherer_key=lambda _: "primary"
        )
        rec._single_pass_gatherers = {"primary": gatherer}
        return rec

    def test_select_gatherer_row_when_hook_would_record(self):
        with patch(
            "sglang.srt.eplb.expert_distribution.get_device", return_value="cpu"
        ):
            gatherer = _SelectExpertsSinglePassGatherer(_metadata(), 0)
        rec = self._recorder(gatherer)
        with patch(
            "sglang.srt.eplb.expert_distribution.megamoe_prefill_only_recorder_enabled",
            return_value=False,
        ):
            self.assertIsNone(rec.get_fused_select_count_buffer())  # no layer
            with rec._current_layer_idx.with_value(1):
                row = rec.get_fused_select_count_buffer()
                self.assertEqual(row.data_ptr(), gatherer._data[1].data_ptr())
                rec._recording = False
                self.assertIsNone(rec.get_fused_select_count_buffer())
                # CUDA-graph capture records even while not recording, like
                # the scatter_add hook does.
                _FakeDeviceModule.capturing = True
                self.assertIsNotNone(rec.get_fused_select_count_buffer())

    def test_dispatch_gatherer_owns_counts(self):
        with patch(
            "sglang.srt.eplb.expert_distribution.get_device", return_value="cpu"
        ):
            gatherer = _DeepepDispatchCountSinglePassGatherer(_metadata(), 1)
        rec = self._recorder(gatherer)
        with (
            patch(
                "sglang.srt.eplb.expert_distribution.megamoe_prefill_only_recorder_enabled",
                return_value=False,
            ),
            rec._current_layer_idx.with_value(0),
        ):
            self.assertIsNone(rec.get_fused_select_count_buffer())
            view = rec.get_deepep_low_latency_recv_stats_buffer()
            self.assertEqual(view.data_ptr(), gatherer.local_count_view(0).data_ptr())


if __name__ == "__main__":
    unittest.main()
