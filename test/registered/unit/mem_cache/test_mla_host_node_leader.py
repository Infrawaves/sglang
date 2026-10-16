import contextlib
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.mem_cache.mla_host_node_leader import (
    NodeLeaderMLAContext,
    split_ranks_by_node,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _context(mla_layers, *, is_src=True, node_ranks=(0, 1, 2, 3), draft_pools=()):
    broadcaster = SimpleNamespace(
        is_src=is_src,
        node_ranks=list(node_ranks),
        min_node_size=len(node_ranks),
        device=torch.device("cpu"),
        device_pool=SimpleNamespace(host_transfer_translate=None),
        kv_staging=torch.empty(8),
        prepare_broadcast=mock.Mock(side_effect=lambda idx, stream: (idx, None)),
        broadcast_loaded_layer=mock.Mock(),
        _bcast_layer=mock.Mock(),
    )
    return NodeLeaderMLAContext(
        broadcaster=broadcaster,
        mla_layers=mla_layers,
        draft_pools=draft_pools,
    )


class TestSplitRanksByNode(unittest.TestCase):
    def test_tp8_on_two_nodes(self):
        self.assertEqual(
            split_ranks_by_node(list(range(8)), 4), [[0, 1, 2, 3], [4, 5, 6, 7]]
        )

    def test_group_not_starting_at_zero(self):
        self.assertEqual(split_ranks_by_node([8, 9, 12, 13], 4), [[8, 9], [12, 13]])


class TestNodeLeaderBroadcast(unittest.TestCase):
    def setUp(self):
        self.stream = object()
        self.entered_streams = []

        @contextlib.contextmanager
        def stream(s):
            self.entered_streams.append(s)
            yield

        patcher = mock.patch(
            "sglang.srt.mem_cache.mla_host_node_leader.device_module",
            SimpleNamespace(current_stream=lambda: self.stream, stream=stream),
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_drafts_broadcast_after_last_mla_layer(self):
        draft = SimpleNamespace(
            kv_buffer=[torch.empty(4, 2)], kv_cache_dim=2, host_transfer_translate=None
        )
        ctx = _context([(3, 0), (7, 1)], draft_pools=(draft,))
        finish = mock.Mock()
        ctx.start_load(index=0, device_indices=torch.arange(4), finish_event=finish)
        ctx.broadcast_ready_layers(0, 3)
        ctx.broadcaster._bcast_layer.assert_not_called()
        ctx.broadcast_ready_layers(0, 7)
        ctx.broadcaster._bcast_layer.assert_called_once()
        args = ctx.broadcaster._bcast_layer.call_args.args
        self.assertIs(args[0], draft.kv_buffer)
        self.assertEqual(args[3], 2)
        self.assertEqual(args[4], 0)
        finish.record.assert_called_once()

    def test_broadcast_follows_transfer_layer_ids(self):
        # Kimi-K3-like interleave: MLA layers sit at transfer ids 3, 7, 11.
        ctx = _context([(3, 0), (7, 1), (11, 2)])
        finish = mock.Mock()
        ctx.start_load(index=0, device_indices=torch.arange(4), finish_event=finish)

        # Waits on KDA layers before the first MLA layer broadcast nothing.
        for threshold in (0, 1, 2):
            ctx.broadcast_ready_layers(0, threshold)
        ctx.broadcaster.broadcast_loaded_layer.assert_not_called()

        ctx.broadcast_ready_layers(0, 3)
        ctx.broadcast_ready_layers(0, 5)
        layers = [c.args[0] for c in ctx.broadcaster.broadcast_loaded_layer.call_args_list]
        self.assertEqual(layers, [0])
        finish.record.assert_not_called()

        # A Mamba-state wait on the last layer flushes everything still pending.
        ctx.broadcast_ready_layers(0, 11)
        layers = [c.args[0] for c in ctx.broadcaster.broadcast_loaded_layer.call_args_list]
        self.assertEqual(layers, [0, 1, 2])
        finish.record.assert_called_once()
        self.assertNotIn(0, ctx.pending_loads)

    def test_reused_slot_flushes_unconsumed_load(self):
        ctx = _context([(3, 0), (7, 1)])
        old_finish, fence = mock.Mock(), object()
        ctx.start_load(index=0, device_indices=torch.arange(4), finish_event=old_finish)
        ctx.broadcast_ready_layers(0, 3)  # forward stopped after layer 0
        new_finish = mock.Mock()
        ctx.flush_stale(0, fence)
        ctx.start_load(index=0, device_indices=torch.arange(2), finish_event=new_finish)
        layers = [
            c.args[0] for c in ctx.broadcaster.broadcast_loaded_layer.call_args_list
        ]
        self.assertEqual(layers, [0, 1])  # old load finished, not dropped
        old_finish.record.assert_called_once()
        # Flushed on the forward (fence) stream, like the layer-wait hook.
        self.assertEqual(self.entered_streams, [fence])
        self.assertIs(ctx.pending_loads[0].finish_event, new_finish)
        new_finish.record.assert_not_called()

    def test_each_layer_broadcast_once(self):
        ctx = _context([(3, 0), (7, 1)])
        ctx.start_load(index=1, device_indices=torch.arange(2), finish_event=mock.Mock())
        for threshold in (3, 3, 7, 7):
            ctx.broadcast_ready_layers(1, threshold)
        self.assertEqual(ctx.broadcaster.broadcast_loaded_layer.call_count, 2)
        ctx.broadcaster.prepare_broadcast.assert_called_once()

    def test_unknown_or_empty_load_is_ignored(self):
        ctx = _context([(3, 0)])
        ctx.broadcast_ready_layers(2, 3)
        ctx.start_load(index=0, device_indices=torch.arange(0), finish_event=mock.Mock())
        ctx.broadcast_ready_layers(0, 3)
        ctx.broadcaster.broadcast_loaded_layer.assert_not_called()

    def test_roles_and_host_pool_kwargs(self):
        leader = _context([(3, 0)], is_src=True)
        follower = _context([(3, 0)], is_src=False)
        self.assertFalse(leader.is_follower)
        self.assertTrue(follower.is_follower)
        self.assertEqual(
            leader.host_pool_kwargs(), {"is_dummy": False, "capacity_multiplier": 4}
        )
        self.assertEqual(
            follower.host_pool_kwargs(), {"is_dummy": True, "capacity_multiplier": 4}
        )


if __name__ == "__main__":
    unittest.main()
