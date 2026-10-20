import contextlib
import os
import socket
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.environ import envs
from sglang.srt.managers.cache_controller import HiCacheController
from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.srt.mem_cache.memory_pool_host import LogicalHostPool
from sglang.srt.mem_cache.mla_host_node_leader import (
    NodeLeaderMLABroadcast,
    NodeLeaderMLAContext,
    NodeSharedHostTensorAllocator,
    _connect_node,
    maybe_create_node_leader_context,
    split_ranks_by_node,
)
from sglang.srt.mem_cache.pool_host import PoolEntry
from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost
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


class TestNodeLeaderStartup(CustomTestCase):
    @contextlib.contextmanager
    def _startup(self, modes, *, enabled=True):
        module = "sglang.srt.mem_cache.mla_host_node_leader"
        world_cpu = object()

        def gather(output, value, *, group):
            self.assertIs(group, world_cpu)
            output[:] = modes

        with (
            mock.patch(
                f"{module}.get_memory",
                return_value=SimpleNamespace(enable_mla_hicache_per_node=enabled),
            ),
            mock.patch(f"{module}.mla_host_dedup_eligible", return_value=True),
            mock.patch(
                f"{module}.get_disagg",
                return_value=SimpleNamespace(disaggregation_mode="null"),
            ),
            mock.patch(
                f"{module}.get_parallel",
                return_value=SimpleNamespace(dcp_enabled=False),
            ),
            mock.patch(f"{module}.is_dp_attention_enabled", return_value=False),
            mock.patch(f"{module}.ranks_per_host", return_value=2),
            mock.patch(
                "sglang.srt.distributed.parallel_state.get_world_group",
                return_value=SimpleNamespace(cpu_group=world_cpu),
            ),
            mock.patch("torch.distributed.get_world_size", return_value=len(modes)),
            mock.patch(
                "torch.distributed.all_gather_object", side_effect=gather
            ) as agreed,
            mock.patch("torch.distributed.get_rank", return_value=0),
            mock.patch("torch.distributed.get_process_group_ranks", return_value=[0]),
            mock.patch(
                "sglang.srt.distributed.parallel_state.create_custom_parallel_group"
            ) as create,
        ):
            yield agreed, create

    def test_mode_mismatch_across_tp_groups_precedes_branching(self):
        """TP-local agreement must not let different WORLD ranks diverge at NCCL setup."""
        for mode in ("broadcast", "shm"):
            with (
                envs.SGLANG_MLA_HICACHE_PER_NODE_MODE.override(mode),
                self._startup(["broadcast", "broadcast", "shm", "shm"]) as (_, create),
            ):
                with self.assertRaisesRegex(ValueError, "must agree across WORLD"):
                    maybe_create_node_leader_context(object(), {}, object(), None)
                create.assert_not_called()

    def test_flag_off_and_size_one_keep_collective_order(self):
        """Flag-off is inert; size-one ranks still join WORLD setup before returning."""
        for enabled in (False, True):
            for mode in ("broadcast", "shm"):
                with (
                    envs.SGLANG_MLA_HICACHE_PER_NODE_MODE.override(mode),
                    self._startup([mode] * 4, enabled=enabled) as (agreed, create),
                ):
                    result = maybe_create_node_leader_context(
                        object(), {}, object(), None
                    )
                    self.assertIsNone(result)
                    self.assertEqual(agreed.call_count, int(enabled))
                    self.assertEqual(
                        create.call_count, int(enabled and mode == "broadcast")
                    )

    def test_bootstrap_setup_failure_closes_local_sockets(self):
        """Aggregated bind/connect failure must not retain bootstrap sockets."""
        for follower in (False, True):
            sock = mock.Mock()
            sock.bind.side_effect = OSError("bind failed")
            sock.connect.side_effect = OSError("connect failed")
            with (
                mock.patch("socket.socket", return_value=sock),
                mock.patch("torch.distributed.get_rank", return_value=int(follower)),
                mock.patch(
                    "torch.distributed.get_process_group_ranks", return_value=[0, 1]
                ),
                mock.patch("torch.distributed.all_gather_object"),
                mock.patch("torch.distributed.all_reduce"),
            ):
                with self.assertRaisesRegex(RuntimeError, "host-local setup failed"):
                    _connect_node(
                        is_follower=follower, node_ranks=[0, 1], tp_cache_group=object()
                    )
            sock.close.assert_called_once()


class TestNodeSharedHostTensorCleanup(CustomTestCase):
    def test_exchange_closes_fds_and_sockets_on_success_and_failure(self):
        """Failed allocation/send/map must release bootstrap resources as success does."""
        module = "sglang.srt.mem_cache.mla_host_node_leader"
        for failure in (None, "allocate", "send", "map", "receive"):
            for follower in (False, True):
                if (failure in ("allocate", "send") and follower) or (
                    failure in ("map", "receive") and not follower
                ):
                    continue
                with (
                    self.subTest(failure=failure, follower=follower),
                    contextlib.ExitStack() as resources,
                ):
                    read_fd, write_fd = os.pipe()
                    os.close(write_fd)
                    resources.callback(self._close_if_open, read_fd)
                    sock, peer = socket.socketpair()
                    resources.callback(sock.close)
                    resources.callback(peer.close)
                    allocator = NodeSharedHostTensorAllocator(
                        is_follower=follower, peer_sockets=[sock]
                    )
                    tensor = object()
                    with (
                        mock.patch(
                            f"{module}.alloc_shm",
                            return_value=(tensor, read_fd, None),
                            side_effect=OSError("allocate")
                            if failure == "allocate"
                            else None,
                        ),
                        mock.patch(
                            "socket.send_fds",
                            create=True,
                            side_effect=OSError("send") if failure == "send" else None,
                        ),
                        mock.patch(
                            "socket.recv_fds",
                            create=True,
                            return_value=(b"x", [read_fd], None, None),
                            side_effect=OSError("receive")
                            if failure == "receive"
                            else None,
                        ),
                        mock.patch(
                            f"{module}.map_shm",
                            return_value=tensor,
                            side_effect=ValueError("map") if failure == "map" else None,
                        ),
                    ):
                        if failure:
                            with self.assertRaisesRegex((OSError, ValueError), failure):
                                allocator.allocate((1,), torch.uint8, "cpu")
                        else:
                            self.assertIs(
                                allocator.allocate((1,), torch.uint8, "cpu"), tensor
                            )
                    self.assertEqual(sock.fileno(), -1)
                    self.assertEqual(allocator.peer_sockets, [])
                    if failure not in ("allocate", "receive"):
                        with self.assertRaises(OSError):
                            os.fstat(read_fd)

    @staticmethod
    def _close_if_open(fd):
        try:
            os.close(fd)
        except OSError:
            pass


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
        host_pool = MLATokenToKVPoolHost.__new__(MLATokenToKVPoolHost)
        host_pool.host_buffer_read_only = mla_host_read_only
        kv = PoolEntry(
            name=PoolName.KV,
            host_pool=host_pool,
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
        host_group = SimpleNamespace(
            anchor_entry=kv, entry_map={kv.name: kv, mamba.name: mamba}
        )
        with mock.patch.object(HiCacheController, "__init__", return_value=None):
            controller = HybridCacheController(None, host_group, 1, None, mock.Mock())
        controller.mla_dedup = mla_dedup
        controller.transfer_layer_id_max = 2
        controller.mem_pool_host = host_group
        return controller

    def test_logical_anchor_has_no_mla_role(self):
        """A logical anchor has no MLA ownership field and must retain its transfers."""
        host = LogicalHostPool.__new__(LogicalHostPool)
        group = SimpleNamespace(anchor_entry=SimpleNamespace(host_pool=host))
        with mock.patch.object(HiCacheController, "__init__", return_value=None):
            controller = HybridCacheController(None, group, 1, None, mock.Mock())
        self.assertFalse(controller.mla_host_read_only)

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

        follower = roles[True].__self__
        follower.storage_backend = mock.Mock()
        self.assertEqual(roles[True](None, ["page0", "page1"], torch.arange(2)), 2)
        self.assertEqual(follower.storage_backend.mock_calls, [])
        self.assertEqual(roles[False].__func__, HiCacheController._page_get_zero_copy)


if __name__ == "__main__":
    unittest.main()
