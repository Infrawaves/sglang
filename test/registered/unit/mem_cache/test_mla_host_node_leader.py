import contextlib
import socket
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.managers.cache_controller import HiCacheController
from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.srt.mem_cache.mla_host_node_leader import (
    NodeLeaderMLABroadcast,
    NodeLeaderMLAContext,
    NodeSharedHostTensorAllocator,
    split_ranks_by_node,
)
from sglang.srt.mem_cache.pool_host import PoolEntry
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _context(mla_layers, *, is_src=True, draft_pools=()):
    broadcaster = SimpleNamespace(
        is_src=is_src,
        device=torch.device("cpu"),
        device_pool=SimpleNamespace(host_transfer_translate=None),
        kv_staging=torch.empty(8),
        prepare_broadcast=mock.Mock(side_effect=lambda idx, stream: (idx, None)),
        broadcast_loaded_layer=mock.Mock(),
        _bcast_layer=mock.Mock(),
    )
    return NodeLeaderMLABroadcast(
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
        layers = [
            c.args[0] for c in ctx.broadcaster.broadcast_loaded_layer.call_args_list
        ]
        self.assertEqual(layers, [0])
        finish.record.assert_not_called()

        # A Mamba-state wait on the last layer flushes everything still pending.
        ctx.broadcast_ready_layers(0, 11)
        layers = [
            c.args[0] for c in ctx.broadcaster.broadcast_loaded_layer.call_args_list
        ]
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
        ctx.start_load(
            index=1, device_indices=torch.arange(2), finish_event=mock.Mock()
        )
        for threshold in (3, 3, 7, 7):
            ctx.broadcast_ready_layers(1, threshold)
        self.assertEqual(ctx.broadcaster.broadcast_loaded_layer.call_count, 2)
        ctx.broadcaster.prepare_broadcast.assert_called_once()

    def test_unknown_or_empty_load_is_ignored(self):
        ctx = _context([(3, 0)])
        ctx.broadcast_ready_layers(2, 3)
        ctx.start_load(
            index=0, device_indices=torch.arange(0), finish_event=mock.Mock()
        )
        ctx.broadcast_ready_layers(0, 3)
        ctx.broadcaster.broadcast_loaded_layer.assert_not_called()


class TestNodeLeaderModes(CustomTestCase):
    def test_mode_sets_follower_pool_kind(self):
        """Broadcast followers get an allocator-only pool, shm followers map the
        leader's buffer; a real private pool on either would hide stale KV."""
        for broadcast, shared_allocator, is_dummy in (
            (_context([(3, 0)], is_src=False), None, True),
            (None, object(), False),
        ):
            for is_follower in (True, False):
                kwargs = NodeLeaderMLAContext(
                    is_follower=is_follower,
                    capacity_multiplier=4,
                    interleave_nodes=(),
                    broadcast=broadcast,
                    shared_allocator=shared_allocator,
                ).host_pool_kwargs()
                self.assertEqual(kwargs["is_dummy"], is_dummy and is_follower)
                self.assertEqual(kwargs["host_buffer_read_only"], is_follower)


@unittest.skipUnless(sys.platform == "linux", "memfd and SCM_RIGHTS need Linux")
class TestNodeSharedHostTensorAllocator(CustomTestCase):
    def _pair(self):
        leader_sock, follower_sock = socket.socketpair(socket.AF_UNIX)
        self.addCleanup(leader_sock.close)
        self.addCleanup(follower_sock.close)
        leader = NodeSharedHostTensorAllocator(
            is_follower=False, peer_sockets=[leader_sock]
        )
        follower = NodeSharedHostTensorAllocator(
            is_follower=True, peer_sockets=[follower_sock]
        )
        return leader, follower

    def test_follower_maps_the_leaders_pages(self):
        """A follower must see the leader's writes: a private copy would make
        it load stale KV with no error."""
        leader, follower = self._pair()
        dims = (3, 5, 1, 7)
        leader_buf = leader.allocate(dims, torch.bfloat16, "cpu")
        follower_buf = follower.allocate(dims, torch.bfloat16, "cpu")

        self.assertNotEqual(leader_buf.data_ptr(), follower_buf.data_ptr())
        leader_buf[2, 4, 0, 6] = 3.5
        self.assertEqual(follower_buf[2, 4, 0, 6].item(), 3.5)

    def test_follower_rejects_a_differently_sized_buffer(self):
        leader, follower = self._pair()
        leader.allocate((1024, 64), torch.uint8, "cpu")
        with self.assertRaises(ValueError):
            follower.allocate((1024, 128), torch.uint8, "cpu")


class TestReadOnlyAnchorTransfers(CustomTestCase):
    def _controller(
        self, *, mla_host_read_only: bool, mla_dedup=None
    ) -> HybridCacheController:
        kv = PoolEntry(
            name=PoolName.KV,
            host_pool=object(),
            device_pool=object(),
            layer_mapper=lambda layer_id: layer_id,
            is_primary_index_anchor=True,
        )
        mamba = PoolEntry(
            name=PoolName.MAMBA,
            host_pool=object(),
            device_pool=object(),
            layer_mapper=lambda layer_id: layer_id,
        )
        controller = HybridCacheController.__new__(HybridCacheController)
        controller.mla_host_read_only = mla_host_read_only
        controller.mla_dedup = mla_dedup
        controller.transfer_layer_id_max = 2
        controller.mem_pool_host = SimpleNamespace(
            anchor_entry=kv, entry_map={kv.name: kv, mamba.name: mamba}
        )
        return controller

    def test_read_only_anchor_is_loaded_but_never_written(self):
        """A follower must skip D2H into the shared MLA buffer (only the leader
        writes it) yet still H2D from it, unless the leader broadcasts its MLA
        KV; Mamba is per rank both ways."""
        indices = torch.arange(4)
        mamba = PoolTransfer(
            name=PoolName.MAMBA, host_indices=indices, device_indices=indices
        )
        follower = self._controller(mla_host_read_only=True)
        leader = self._controller(mla_host_read_only=False)
        entries = follower.mem_pool_host.entry_map
        kv_pool, mamba_pool = (
            entries[PoolName.KV].host_pool,
            entries[PoolName.MAMBA].host_pool,
        )

        def host_pools(transfers):
            return [t.host_pool for t in transfers]

        self.assertEqual(
            host_pools(follower._l2_transfers(indices, indices, [mamba])),
            [mamba_pool],
        )
        self.assertEqual(
            host_pools(follower._l2_load_transfers(indices, indices, [mamba])),
            [kv_pool, mamba_pool],
        )
        self.assertEqual(len(leader._l2_transfers(indices, indices, [mamba])), 2)
        broadcast_follower = self._controller(
            mla_host_read_only=True, mla_dedup=_context([(3, 0)], is_src=False)
        )
        broadcast_follower.mem_pool_host = follower.mem_pool_host
        self.assertEqual(
            host_pools(
                broadcast_follower._l2_load_transfers(indices, indices, [mamba])
            ),
            [mamba_pool],
        )

    def test_follower_l3_prefetch_never_writes_the_shared_buffer(self):
        """Attaching L3 must leave a follower with the stub page get: the real
        one would write prefetched pages into the leader's shared buffer."""

        def base_attach(self, **kwargs):
            self.page_get_func = self._page_get_zero_copy

        roles = {}
        for read_only in (True, False):
            controller = self._controller(mla_host_read_only=read_only)
            controller.host_memory_mode = "cache"
            controller.pp_group = controller.pp_prefetch_command_group = None
            with mock.patch.object(
                HiCacheController, "attach_storage_backend", base_attach
            ):
                controller.attach_storage_backend("mooncake")
            roles[read_only] = controller.page_get_func

        self.assertEqual(
            roles[True].__func__, HybridCacheController._page_get_shared_by_leader
        )
        self.assertEqual(roles[False].__func__, HiCacheController._page_get_zero_copy)


if __name__ == "__main__":
    unittest.main()
