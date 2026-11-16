import contextlib
import inspect
import mmap
import multiprocessing
import os
import sys
import tempfile
import unittest
import weakref
from datetime import timedelta
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.managers.cache_controller import HiCacheController
from sglang.srt.mem_cache import mla_host_node_leader
from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.srt.mem_cache.memory_pool_host import LogicalHostPool
from sglang.srt.mem_cache.mla_host_dedup import MLAHostDedupBroadcaster
from sglang.srt.mem_cache.mla_host_node_leader import (
    NodeLeaderMLABroadcast,
    NodeSharedHostTensorAllocator,
    maybe_create_node_leader_context,
    split_ranks_by_node,
    synchronize_shared_host_error,
)
from sglang.srt.mem_cache.pool_host import PoolEntry
from sglang.srt.mem_cache.pool_host.common import (
    ALLOC_MEMORY_FUNCS,
    alloc_with_host_register,
)
from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost
from sglang.srt.mem_cache.unified_cache.storage_attachment import StorageAttachment
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=120, suite="base-a-test-cpu")


def _context(mla_layers, *, is_src=True, draft_pools=()):
    broadcaster = mock.create_autospec(MLAHostDedupBroadcaster, instance=True)
    broadcaster.is_src = is_src
    broadcaster.device = torch.device("cpu")
    broadcaster.device_pool = SimpleNamespace()
    broadcaster.kv_staging = torch.empty(8)
    broadcaster.prepare_broadcast.side_effect = lambda indices, stream: (
        indices,
        None,
    )
    return NodeLeaderMLABroadcast(
        broadcaster=broadcaster,
        mla_layers=mla_layers,
        draft_pools=draft_pools,
    )


class TestSplitRanksByNode(CustomTestCase):
    def test_attention_group_crosses_host_boundaries(self):
        # TP12/DP2, four ranks per host: the second attention group spans 2+4 ranks.
        self.assertEqual(
            split_ranks_by_node(list(range(6, 12)), 4), [[6, 7], [8, 9, 10, 11]]
        )


class TestIndexerSharingEligibility(CustomTestCase):
    def test_assembler_shares_only_the_target_indexer(self):
        from sglang.srt.mem_cache.device_pool_info import IndexKeyLayout
        from sglang.srt.mem_cache.hybrid_cache import hybrid_pool_assembler as assembler

        root = SimpleNamespace(
            layer_shard_enabled=False,
            layer_num=2,
            device="cpu",
            store_dtype=torch.float16,
            start_layer=0,
            end_layer=2,
        )
        host = SimpleNamespace(
            page_num=4, page_size=2, layout="page_first", dcp_size=1, dcp_rank=0
        )
        for name in (PoolName.INDEXER, PoolName.DRAFT_INDEXER):
            for follower in (False, True):
                with self.subTest(name=name, follower=follower):
                    allocator = object()
                    leader = SimpleNamespace(
                        is_follower=follower,
                        dsa_indexer_allocator=mock.Mock(return_value=allocator),
                    )
                    config = SimpleNamespace(
                        pool_name=name,
                        device_to_host_layer={0: 0, 1: 1},
                        buffer_info=SimpleNamespace(
                            key_layout=IndexKeyLayout.PAGED_FP8_KEYS_WITH_SCALES
                        ),
                    )
                    with (
                        mock.patch.object(
                            assembler, "DSAIndexerPoolHost"
                        ) as constructor,
                        mock.patch.object(
                            assembler, "_get_allocator_type", return_value="default"
                        ),
                    ):
                        assembler._build_sidecar_host_pool(
                            config=config,
                            root=root,
                            root_host_pool=host,
                            node_leader=leader,
                        )
                    args = constructor.call_args.kwargs
                    shared = name == PoolName.INDEXER
                    self.assertIs(args["host_allocator"], allocator if shared else None)
                    self.assertEqual(args["is_dummy"], shared and follower)
                    self.assertEqual(args["host_buffer_read_only"], shared and follower)
                    self.assertEqual(args["num_host_pages"], host.page_num)
                    if shared:
                        leader.dsa_indexer_allocator.assert_called_once_with(
                            layer_sharded=False
                        )
                    else:
                        leader.dsa_indexer_allocator.assert_not_called()

    def test_sharing_conditions_and_independent_allocators(self):
        from sglang.srt.mem_cache.mla_host_node_leader import NodeLeaderMLAContext

        shared = NodeSharedHostTensorAllocator(
            node_ranks=[0, 1], tp_cache_group=object()
        )
        for has_indexer in (False, True):
            for has_draft in (False, True):
                for sharded in (False, True):
                    with self.subTest(
                        indexer=has_indexer, draft=has_draft, sharded=sharded
                    ):
                        context = NodeLeaderMLAContext(
                            is_follower=False,
                            capacity_multiplier=2,
                            shared_allocator=shared,
                            broadcast=SimpleNamespace(
                                broadcaster=SimpleNamespace(
                                    idx_bufs=[] if has_indexer else None
                                ),
                                draft_pools=(object(),) if has_draft else (),
                            ),
                        )
                        expected = has_indexer and not has_draft and not sharded
                        self.assertEqual(
                            context.shares_dsa_indexer(layer_sharded=sharded), expected
                        )
                        allocator = context.dsa_indexer_allocator(layer_sharded=sharded)
                        if expected:
                            self.assertIsInstance(
                                allocator, NodeSharedHostTensorAllocator
                            )
                            self.assertIsNot(allocator, shared)
                            self.assertIsNot(
                                allocator,
                                context.dsa_indexer_allocator(layer_sharded=sharded),
                            )
                            self.assertEqual(allocator.node_ranks, shared.node_ranks)
                            self.assertIs(
                                allocator.tp_cache_group, shared.tp_cache_group
                            )
                        else:
                            self.assertIsNone(allocator)


class TestNodeLeaderBroadcast(CustomTestCase):
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

    def test_restore_uses_anchor_translation_but_not_for_packed_draft(self):
        for is_src in (False, True):
            for translated in (False, True):
                with self.subTest(is_src=is_src, translated=translated):
                    from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool

                    draft = SimpleNamespace(
                        kv_buffer=[torch.empty(8, 2)], kv_cache_dim=2
                    )
                    ctx = _context(
                        [(3, 0), (7, 1)], is_src=is_src, draft_pools=(draft,)
                    )
                    ctx.broadcaster.device_pool = DSATokenToKVPool.__new__(
                        DSATokenToKVPool
                    )
                    self.assertFalse(
                        hasattr(ctx.broadcaster.device_pool, "host_transfer_translate")
                    )
                    mapper = (
                        mock.Mock(side_effect=lambda ids: ids + 8)
                        if translated
                        else None
                    )
                    anchor = PoolEntry(
                        name=PoolName.KV,
                        host_pool=SimpleNamespace(host_buffer_read_only=not is_src),
                        device_pool=ctx.broadcaster.device_pool,
                        device_translate=mapper,
                        is_primary_index_anchor=True,
                    )
                    group = SimpleNamespace(
                        anchor_entry=anchor, entry_map={PoolName.KV: anchor}
                    )
                    with mock.patch.object(
                        HiCacheController, "__init__", return_value=None
                    ):
                        HybridCacheController(
                            None, group, 2, None, mock.Mock(), node_leader_broadcast=ctx
                        )
                    self.assertIs(ctx.device_translate, mapper)
                    indices = torch.tensor([2, 3, 0, 1])
                    finish = mock.Mock()
                    ctx.start_load(index=0, device_indices=indices, finish_event=finish)
                    ctx.broadcast_ready_layers(0, 0)
                    ctx.broadcast_ready_layers(0, 7)
                    expected = indices + 8 if translated else indices
                    torch.testing.assert_close(
                        ctx.broadcaster.prepare_broadcast.call_args.args[0], expected
                    )
                    torch.testing.assert_close(
                        ctx.broadcaster._bcast_layer.call_args.args[2], indices
                    )
                    ctx.broadcaster.prepare_broadcast.assert_called_once()
                    if mapper is not None:
                        mapper.assert_called_once()
                    finish.record.assert_called_once()

    def test_drafts_broadcast_after_last_mla_layer(self):
        draft = SimpleNamespace(kv_buffer=[torch.empty(4, 2)], kv_cache_dim=2)
        ctx = _context([(3, 0), (7, 1)], draft_pools=(draft,))
        finish = mock.Mock()
        ctx.start_load(index=0, device_indices=torch.arange(4), finish_event=finish)
        ctx.broadcast_ready_layers(0, 3)
        ctx.broadcaster._bcast_layer.assert_not_called()
        ctx.broadcast_ready_layers(0, 7)
        ctx.broadcaster._bcast_layer.assert_called_once()
        call = ctx.broadcaster._bcast_layer.call_args
        bound = (
            inspect.signature(MLAHostDedupBroadcaster._bcast_layer)
            .bind(ctx.broadcaster, *call.args, **call.kwargs)
            .arguments
        )
        self.assertIs(bound["buf_list"], draft.kv_buffer)
        self.assertEqual(bound["elem"], 2)
        self.assertEqual(bound["layer_id"], 0)
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
        ctx.broadcast_ready_layers(0, 3)
        ctx.broadcast_ready_layers(0, 5)
        layers = [
            c.args[0] for c in ctx.broadcaster.broadcast_loaded_layer.call_args_list
        ]
        self.assertEqual(layers, [0])
        finish.record.assert_not_called()

        # A Mamba-state wait on the last layer flushes everything still pending.
        ctx.broadcast_ready_layers(0, 11)
        ctx.broadcast_ready_layers(0, 11)
        layers = [
            c.args[0] for c in ctx.broadcaster.broadcast_loaded_layer.call_args_list
        ]
        self.assertEqual(layers, [0, 1, 2])
        finish.record.assert_called_once()
        # One prepare per load, not per layer wait (the load.prepared cache).
        ctx.broadcaster.prepare_broadcast.assert_called_once()

    def test_reused_slot_flushes_unconsumed_load(self):
        """Reusing a load slot must finish the old load's remaining layers, not drop them."""
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
        new_finish.record.assert_not_called()
        ctx.broadcast_ready_layers(0, 7)
        self.assertEqual(
            [c.args[0] for c in ctx.broadcaster.broadcast_loaded_layer.call_args_list],
            [0, 1, 0, 1],
        )
        new_finish.record.assert_called_once()

    def test_unknown_or_empty_load_is_ignored(self):
        ctx = _context([(3, 0)])
        ctx.broadcast_ready_layers(2, 3)
        finish = mock.Mock()
        ctx.start_load(index=0, device_indices=torch.arange(0), finish_event=finish)
        ctx.broadcast_ready_layers(0, 3)
        ctx.broadcaster.prepare_broadcast.assert_not_called()
        ctx.broadcaster.broadcast_loaded_layer.assert_not_called()
        finish.record.assert_not_called()


class TestNodeLeaderStartup(CustomTestCase):
    @contextlib.contextmanager
    def _startup(
        self,
        *,
        enabled=True,
        group_ranks=(0,),
        rank=0,
        eligible=True,
    ):
        module = "sglang.srt.mem_cache.mla_host_node_leader"

        def broadcaster(*, src_global_rank, **kwargs):
            return SimpleNamespace(
                is_src=rank == src_global_rank,
                kv_staging=torch.empty(1, dtype=torch.uint8),
                _warmup_group=mock.Mock(),
            )

        with (
            mock.patch(
                f"{module}.get_memory",
                return_value=SimpleNamespace(
                    enable_mla_hicache_per_node=enabled,
                ),
            ),
            mock.patch(f"{module}.mla_host_dedup_eligible", return_value=eligible),
            mock.patch(f"{module}.ranks_per_host", return_value=4),
            mock.patch("torch.distributed.get_rank", return_value=rank),
            mock.patch(
                "torch.distributed.get_process_group_ranks",
                return_value=list(group_ranks),
            ),
            mock.patch(f"{module}.create_custom_parallel_group") as create,
            mock.patch("torch.distributed.destroy_process_group") as destroy,
            mock.patch(
                f"{module}.MLAHostDedupBroadcaster",
                autospec=True,
                side_effect=broadcaster,
            ) as build,
        ):
            self.destroy = destroy
            yield create, build

    def _create(self, full_layer_mapping=None):
        return maybe_create_node_leader_context(
            kv_pool=object(),
            full_layer_mapping=full_layer_mapping or {},
            tp_cache_group=object(),
            draft_pools=(),
        )

    def test_disabled_flag_adds_no_startup_collective(self):
        """Flag off must return before any WORLD-collective NCCL setup."""
        with self._startup(enabled=False, group_ranks=range(4)) as (create, _):
            self.assertIsNone(self._create())
            create.assert_not_called()

    def test_ineligible_pool_rejects_before_world_group_creation(self):
        """A non-CUDA or FP4 MLA pool must raise on every rank before WORLD setup."""
        with self._startup(group_ranks=range(4), eligible=False) as (create, _):
            with self.assertRaisesRegex(ValueError, "non-FP4 MLA KV pool"):
                self._create()
            create.assert_not_called()

    def test_singleton_groups_join_world_setup(self):
        """One-rank-per-node groups must not strand active peers in WORLD NCCL setup."""
        with self._startup(group_ranks=(0,)) as (create, _):
            self.assertIsNone(self._create())
            create.assert_called_once_with(group_ranks=[0], backend="nccl")
            self.destroy.assert_called_once_with(create.return_value)

    def test_common_capacity_and_own_node_ranks(self):
        """Each rank must share with its own node's ranks, with equal cache-group capacity."""
        for ranks, rank, local_rank, capacity, nccl in (
            (range(8), 4, 0, 4, [4, 5, 6, 7]),
            (range(8), 7, 3, 4, [4, 5, 6, 7]),
            (range(6, 12), 8, 0, 2, [8, 9, 10, 11]),
            (range(6, 12), 11, 3, 2, [8, 9, 10, 11]),
        ):
            with (
                self.subTest(rank=rank),
                self._startup(group_ranks=ranks, rank=rank) as (create, _),
            ):
                context = self._create({3: 0})
                self.assertEqual(context.capacity_multiplier, capacity)
                self.assertEqual(context.is_follower, local_rank != 0)
                self.assertEqual(
                    context.shared_allocator.node_ranks,
                    list(range(rank - local_rank, rank - local_rank + 4)),
                )
                self.assertIsNotNone(context.broadcast)
                # The broadcast NCCL group spans only this node's ranks.
                create.assert_called_once_with(group_ranks=nccl, backend="nccl")


class TestNodeSharedHostTensorCleanup(CustomTestCase):
    def test_follower_reopens_own_leader_and_releases_fds(self):
        """Followers reopen their own node leader; acquired FDs and maps always release."""
        module = "sglang.srt.mem_cache.mla_host_node_leader"
        for follower, failure in (
            (False, None),
            (True, None),
            (False, "create"),
            (True, "map"),
            (False, "remote"),
        ):
            with (
                self.subTest(follower=follower, failure=failure),
                tempfile.TemporaryFile() as file,
            ):
                file.truncate(mmap.PAGESIZE)
                # Published by this node's leader (global rank 8, group index 2).
                owner = (12345, file.fileno())
                acquired = []
                reopened = []
                mappings = []
                phases = []
                allocator = NodeSharedHostTensorAllocator(
                    node_ranks=[8, 9],
                    tp_cache_group=object(),
                )

                def acquire(*args):
                    if failure == "create":
                        raise OSError("injected create")
                    fd = os.dup(file.fileno())
                    acquired.append(fd)
                    return fd

                def reopen(published):
                    reopened.append(published)
                    fd = os.dup(file.fileno())
                    acquired.append(fd)
                    return fd

                def gather(output, value, *, group):
                    self.assertIs(group, allocator.tp_cache_group)
                    if not phases:
                        # The first node's descriptor must never be reopened here.
                        output[:] = [(999, 99), None, owner, None]
                        output[3 if follower else 2] = value
                        phases.append("metadata")
                    else:
                        output[:] = [None, None, None, None]
                        output[3 if follower else 2] = value
                        if failure == "remote":
                            output[0] = "rank 2: injected remote prefault"
                        if not follower and acquired:
                            # Even on failure, the leader holds its published FD
                            # until every peer has finished the acquisition fence.
                            os.fstat(acquired[0])
                        phases.append("prefault")

                def mapped(fd, dims, dtype, *, populate_shard):
                    if failure == "map":
                        raise OSError("injected map")
                    mm = mmap.mmap(fd, mmap.PAGESIZE)
                    mappings.append(weakref.ref(mm))
                    return torch.frombuffer(mm, dtype=dtype, count=1)

                with (
                    mock.patch(f"{module}.create_shm", side_effect=acquire),
                    mock.patch.object(allocator, "_open_shared_fd", side_effect=reopen),
                    mock.patch(f"{module}.map_shm", side_effect=mapped),
                    mock.patch(
                        "torch.distributed.get_process_group_ranks",
                        return_value=[2, 3, 8, 9],
                    ),
                    mock.patch("torch.distributed.get_world_size", return_value=4),
                    mock.patch(
                        "torch.distributed.get_rank", return_value=8 + int(follower)
                    ),
                    mock.patch(
                        "torch.distributed.all_gather_object", side_effect=gather
                    ),
                ):
                    error = None
                    try:
                        tensor = allocator.allocate((1,), torch.uint8, "cpu")
                    except RuntimeError as exc:
                        error = exc
                    self.assertEqual(reopened, [owner] if follower else [])
                    self.assertEqual(bool(acquired), failure != "create")
                    self.assertEqual(bool(mappings), failure in (None, "remote"))
                    if failure:
                        self.assertIsNotNone(error)
                        self.assertIn(failure, str(error))
                        # Retaining the traceback must not retain successful maps.
                        self.assertTrue(all(ref() is None for ref in mappings))
                    else:
                        self.assertIsNone(error)
                        self.assertEqual(tensor.tolist(), [0])
                        self.assertIsNotNone(mappings[0]())
                        del tensor
                        self.assertIsNone(mappings[0]())
                    self.assertEqual(phases, ["metadata", "prefault"])
                    for fd in acquired:
                        with self.assertRaises(OSError):
                            os.fstat(fd)


# The shared buffer maps 10 pages, populated 5/5 by the two node-local ranks.
_SHARED_BYTES = 9 * mmap.PAGESIZE + 128
_LEADER_SHARD_BYTES = 5 * mmap.PAGESIZE
# Each worker fills its shard with rank + 1: leader 1s, follower 2s.
_SHARED_MARKER_SUM = _LEADER_SHARD_BYTES + 2 * (_SHARED_BYTES - _LEADER_SHARD_BYTES)


def _build_test_shared_pool(allocator):
    pin_count = 0

    def register(buffer, granularity=None):
        nonlocal pin_count
        if int(buffer.sum()) != _SHARED_MARKER_SUM:
            raise AssertionError("registration preceded shared initialization")
        pin_count += 1

    device_pool = SimpleNamespace(
        layer_num=1,
        layer_shard_enabled=False,
        store_dtype=torch.uint8,
        kv_lora_rank=1,
        qk_rope_head_dim=0,
        size=_SHARED_BYTES - 1,
        start_layer=0,
        end_layer=1,
        device="cpu",
    )
    with (
        mock.patch.dict(ALLOC_MEMORY_FUNCS, {"cpu": alloc_with_host_register}),
        mock.patch(
            "sglang.srt.mem_cache.pool_host.common._cuda_host_register",
            side_effect=register,
        ),
        mock.patch(
            "sglang.srt.mem_cache.pool_host.base.host_memory_budget_bytes",
            return_value=1 << 30,
        ),
    ):
        follower = torch.distributed.get_rank() != allocator.node_ranks[0]
        pool = MLATokenToKVPoolHost(
            device_pool,
            1,
            0,
            1,
            "layer_first",
            is_dummy=follower,
            host_allocator=allocator,
            host_buffer_read_only=follower,
        )
    return pool, pin_count


def _shared_allocator_worker(rank, init_path, failure, events, out):
    blocked, release, leader_fence, returned = events
    world_size = 4 if failure == "other_node" else 2
    node_rank = rank % 2
    node_ranks = [rank - node_rank, rank - node_rank + 1]
    torch.distributed.init_process_group(
        "gloo",
        init_method=f"file://{init_path}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=30),
    )
    try:
        group = torch.distributed.group.WORLD
        allocator = NodeSharedHostTensorAllocator(
            node_ranks=node_ranks,
            tp_cache_group=group,
        )
        dims = (
            _SHARED_BYTES * 2 if failure == "size" and rank == 1 else _SHARED_BYTES,
        )
        original_map = mla_host_node_leader.map_shm
        original_open = allocator._open_shared_fd

        def open_buffer(owner):
            if rank == 1:
                if failure == "lifetime":
                    blocked.set()
                    if not release.wait(30):
                        raise TimeoutError(
                            "test did not release descriptor acquisition"
                        )
                elif failure == "open":
                    # Fail os.open inside the real reopen so its wrapper runs.
                    with mock.patch.object(
                        mla_host_node_leader.os,
                        "open",
                        side_effect=PermissionError("injected open failure"),
                    ):
                        return original_open(owner)
            return original_open(owner)

        def map_buffer(fd, dims, dtype, *, populate_shard):
            if rank == 3 and failure == "other_node":
                raise OSError("injected other_node prefault failure")
            if rank == 1 and failure == "pool":
                blocked.set()
                if not release.wait(30):
                    raise TimeoutError("test did not release prefault")
            buffer = original_map(fd, dims, dtype, populate_shard=populate_shard)
            # Publish one marker per worker before the allocator's completion
            # fence; all returned views must already contain both markers.
            start, end = (
                (0, _LEADER_SHARD_BYTES)
                if node_rank == 0
                else (_LEADER_SHARD_BYTES, _SHARED_BYTES)
            )
            buffer.view(-1)[start:end].fill_(rank + 1)
            return buffer

        def synchronize(group, error, *, operation):
            if rank == 0 and operation == "prefault":
                leader_fence.set()
            synchronize_shared_host_error(group, error, operation=operation)

        with contextlib.ExitStack() as stack:
            stack.enter_context(
                mock.patch.object(allocator, "_open_shared_fd", open_buffer)
            )
            stack.enter_context(
                mock.patch.object(mla_host_node_leader, "map_shm", map_buffer)
            )
            stack.enter_context(
                mock.patch.object(
                    mla_host_node_leader, "synchronize_shared_host_error", synchronize
                )
            )
            if rank == 0 and failure == "create":
                stack.enter_context(
                    mock.patch.object(
                        mla_host_node_leader,
                        "create_shm",
                        side_effect=OSError("injected create failure"),
                    )
                )
            try:
                if failure == "pool":
                    pool, pin_count = _build_test_shared_pool(allocator)
                    returned[rank].set()
                    out.put(
                        (
                            rank,
                            "pool",
                            pin_count,
                            bool(pool.get_contiguous_buf_infos()[0]),
                        )
                    )
                else:
                    buffer = allocator.allocate(dims, torch.uint8, "cpu")
                    returned[rank].set()
                    out.put((rank, "ok", int(buffer.sum())))
            except Exception as exc:
                out.put((rank, "error", str(exc)))
    finally:
        torch.distributed.destroy_process_group()


@unittest.skipUnless(
    sys.platform == "linux", "/proc FD reopening and sharded madvise need Linux"
)
class TestNodeSharedHostTensorAllocator(CustomTestCase):
    def _run(self, failure):
        context = multiprocessing.get_context("spawn")
        world_size = 4 if failure == "other_node" else 2
        events = (
            context.Event(),
            context.Event(),
            context.Event(),
            [context.Event() for _ in range(world_size)],
        )
        out = context.Queue()
        with tempfile.TemporaryDirectory() as directory:
            processes = [
                context.Process(
                    target=_shared_allocator_worker,
                    args=(rank, os.path.join(directory, "gloo"), failure, events, out),
                )
                for rank in range(world_size)
            ]
            for process in processes:
                process.start()
            try:
                if failure in ("pool", "lifetime"):
                    blocked, release, leader_fence, returned = events
                    self.assertTrue(
                        blocked.wait(45),
                        "follower did not reach blocked initialization",
                    )
                    self.assertTrue(leader_fence.wait(45), "leader did not reach fence")
                    self.assertFalse(
                        returned[0].is_set(),
                        "leader returned to its pin caller before peer prefault",
                    )
                    release.set()
                results = [out.get(timeout=45) for _ in processes]
                for process in processes:
                    process.join(timeout=10)
                    self.assertEqual(process.exitcode, 0)
                return sorted(results)
            finally:
                events[1].set()
                for process in processes:
                    if process.is_alive():
                        process.terminate()
                    process.join(timeout=10)
                out.close()
                out.join_thread()

    def test_initialize_before_registration_and_keep_only_runtime_views(self):
        """No rank pins before every share is populated; followers keep no data view."""
        results = self._run("pool")
        self.assertEqual(results, [(0, "pool", 1, True), (1, "pool", 0, False)])

    def test_late_follower_reopens_before_leader_closes_fd(self):
        """The original descriptor must survive until a delayed follower opens it."""
        results = self._run("lifetime")
        self.assertEqual(
            results, [(0, "ok", _SHARED_MARKER_SUM), (1, "ok", _SHARED_MARKER_SUM)]
        )

    def test_setup_failures_reach_every_rank(self):
        """Setup failures must release peers, including another node-local group."""
        for failure in ("create", "open", "size", "other_node"):
            with self.subTest(failure=failure):
                results = self._run(failure)
                self.assertEqual(
                    [result[1] for result in results], ["error"] * len(results)
                )
                needle = (
                    "expected"
                    if failure == "size"
                    else "cannot reopen" if failure == "open" else f"injected {failure}"
                )
                for _, _, message in results:
                    self.assertIn(needle, message)


def _shared_indexer_worker(rank, world_size, init_path, layout, failing_rank, out):
    from sglang.srt.mem_cache.device_pool_info import IndexKeyBufferInfo, IndexKeyLayout
    from sglang.srt.mem_cache.pool_host.dsa import DSAIndexerPoolHost

    torch.distributed.init_process_group(
        "gloo",
        init_method=f"file://{init_path}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=30),
    )
    try:
        leader = rank - rank % 2
        allocator = NodeSharedHostTensorAllocator(
            node_ranks=[leader, leader + 1],
            tp_cache_group=torch.distributed.group.WORLD,
        )
        info = IndexKeyBufferInfo(
            page_size=2,
            compress_ratio=1,
            key_layout=IndexKeyLayout.PAGED_FP8_KEYS_WITH_SCALES,
            buffers=tuple(torch.zeros((4, 264), dtype=torch.uint8) for _ in range(2)),
        )
        original_map = mla_host_node_leader.map_shm
        mappings = []
        owners = []

        def map_buffer(fd, dims, dtype, *, populate_shard):
            buffer = original_map(fd, dims, dtype, populate_shard=populate_shard)
            mappings.append(weakref.ref(buffer))
            owners.append(os.fstat(fd).st_ino)
            # Each node writes a distinct marker before the real prefault fence.
            if rank == leader:
                buffer.fill_(leader + 1)
            return buffer

        with (
            mock.patch.dict(ALLOC_MEMORY_FUNCS, {"cpu": alloc_with_host_register}),
            mock.patch.object(mla_host_node_leader, "map_shm", map_buffer),
            mock.patch(
                "sglang.srt.mem_cache.pool_host.dsa.host_memory_budget_bytes",
                return_value=0 if rank == failing_rank else 1 << 30,
            ) as budget,
            mock.patch.object(
                allocator, "allocate", wraps=allocator.allocate
            ) as allocate,
        ):
            try:
                pool = DSAIndexerPoolHost(
                    info,
                    layer_mapping={0: 0, 1: 1},
                    target_device_layer_num=2,
                    num_host_pages=4,
                    transfer_page_size=2,
                    layout=layout,
                    buffer_device="cpu",
                    dtype=torch.float16,
                    pin_memory=False,
                    is_dummy=rank != leader,
                    host_allocator=allocator,
                    host_buffer_read_only=rank != leader,
                )
            except RuntimeError as exc:
                out.put((rank, "error", str(exc), allocate.call_count))
            else:
                buffer = pool.index_k_with_scale_buffer
                out.put(
                    (
                        rank,
                        "ok",
                        allocator.dims,
                        pool.size,
                        buffer is not None,
                        (
                            None
                            if buffer is None
                            else bool(torch.all(buffer == leader + 1))
                        ),
                        owners,
                        sum(ref() is not None for ref in mappings),
                        budget.call_count,
                        allocate.call_count,
                    )
                )
    finally:
        torch.distributed.destroy_process_group()


@unittest.skipUnless(sys.platform == "linux", "shared memfd requires Linux /proc")
class TestNodeSharedDSAIndexer(CustomTestCase):
    def _run(self, world_size, layout, failing_rank=None):
        context = multiprocessing.get_context("spawn")
        out = context.Queue()
        with tempfile.TemporaryDirectory() as directory:
            processes = [
                context.Process(
                    target=_shared_indexer_worker,
                    args=(
                        rank,
                        world_size,
                        os.path.join(directory, "gloo"),
                        layout,
                        failing_rank,
                        out,
                    ),
                )
                for rank in range(world_size)
            ]
            try:
                for process in processes:
                    process.start()
                results = sorted(out.get(timeout=60) for _ in processes)
                for process in processes:
                    process.join(timeout=10)
                    self.assertEqual(process.exitcode, 0)
                return results
            finally:
                for process in processes:
                    if process.pid is not None:
                        if process.is_alive():
                            process.terminate()
                        process.join(timeout=10)
                out.close()
                out.join_thread()

    def test_one_and_two_nodes_allocate_only_on_leaders(self):
        for world_size in (2, 4):
            for layout in ("layer_first", "page_first"):
                with self.subTest(world_size=world_size, layout=layout):
                    results = self._run(world_size, layout)
                    expected_dims = (
                        (2, 5, 264) if layout == "layer_first" else (5, 2, 1, 264)
                    )
                    for row in results:
                        rank = row[0]
                        self.assertEqual(row[1], "ok", row)
                        self.assertEqual(row[2:4], (expected_dims, 8))
                        self.assertEqual(
                            row[4:6], (rank % 2 == 0, True if rank % 2 == 0 else None)
                        )
                        self.assertEqual(
                            row[7:],
                            (1 if rank % 2 == 0 else 0, 1 if rank % 2 == 0 else 0, 1),
                        )
                        self.assertEqual(row[6], results[rank - rank % 2][6])
                    if world_size == 4:
                        self.assertNotEqual(results[0][6], results[2][6])

    def test_either_leader_budget_failure_reaches_all_nodes_before_allocate(self):
        for leader in (0, 2):
            with self.subTest(failing_leader=leader):
                results = self._run(4, "page_first", failing_rank=leader)
                self.assertEqual([row[0] for row in results], list(range(4)))
                for _, status, message, allocations in results:
                    self.assertEqual(status, "error")
                    self.assertIn("DSA indexer budget", message)
                    self.assertIn(f"rank {leader}:", message)
                    self.assertIn("Not enough host memory", message)
                    self.assertEqual(allocations, 0)


class TestReadOnlyAnchorTransfers(CustomTestCase):
    def _controller(
        self, *, mla_host_read_only: bool, node_leader_broadcast=None
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
        controller.node_leader_broadcast = node_leader_broadcast
        controller.transfer_layer_id_max = 2
        controller.mem_pool_host = host_group
        return controller

    def test_indexer_transfers_and_backup_follow_node_roles(self):
        indices = torch.arange(4)
        for rank, follower in ((0, False), (1, True), (4, False), (5, True)):
            with self.subTest(rank=rank):
                controller = self._controller(
                    mla_host_read_only=follower,
                    node_leader_broadcast=_context([(0, 0)], is_src=not follower),
                )
                entry = PoolEntry(
                    name=PoolName.INDEXER,
                    host_pool=SimpleNamespace(host_buffer_read_only=follower),
                    device_pool=object(),
                    layer_mapper=lambda layer: layer,
                )
                controller.mem_pool_host.entry_map[PoolName.INDEXER] = entry
                transfer = PoolTransfer(
                    name=PoolName.INDEXER,
                    host_indices=indices,
                    device_indices=indices,
                )
                expected = (
                    []
                    if follower
                    else [
                        controller.mem_pool_host.anchor_entry.host_pool,
                        entry.host_pool,
                    ]
                )
                self.assertEqual(
                    [
                        t.host_pool
                        for t in controller._l2_transfers(indices, indices, [transfer])
                    ],
                    expected,
                )
                self.assertEqual(
                    [
                        t.host_pool
                        for t in controller._l2_load_transfers(
                            indices, indices, [transfer]
                        )
                    ],
                    expected,
                )
                controller.storage_config = SimpleNamespace(is_storage_writer=rank == 0)
                controller.draft_node_merge = None
                controller.storage_backend_type = "mooncake"
                self.assertEqual(controller.should_backup(transfer), rank == 0)

    def test_indexer_l3_reads_are_filtered_before_backend_call(self):
        from sglang.srt.mem_cache.hicache_storage import HiCacheStorageExtraInfo

        for follower in (False, True):
            with self.subTest(follower=follower):
                controller = self._controller(mla_host_read_only=follower)
                controller.mem_pool_host.entry_map[PoolName.INDEXER] = SimpleNamespace(
                    host_pool=SimpleNamespace(host_buffer_read_only=follower),
                )
                controller.page_get_func = mock.Mock(return_value=2)
                controller.storage_backend = SimpleNamespace(
                    batch_get_v2=mock.Mock(
                        return_value={PoolName.INDEXER: [True, False]}
                    ),
                )
                transfer = PoolTransfer(name=PoolName.INDEXER)
                hits = controller._page_transfer_kv_batch(
                    SimpleNamespace(request_id="indexer-read"),
                    ["a", "b"],
                    torch.arange(4),
                    HiCacheStorageExtraInfo(),
                    [transfer],
                )
                self.assertEqual(hits, 2 if follower else 1)
                backend = controller.storage_backend.batch_get_v2
                if follower:
                    backend.assert_not_called()
                else:
                    backend.assert_called_once()
                    self.assertEqual(
                        [t.name for t in backend.call_args.args[0]], [PoolName.INDEXER]
                    )
                    self.assertEqual(backend.call_args.args[0][0].keys, ["a", "b"])

    def test_logical_anchor_has_no_mla_role(self):
        """A non-MLA (logical) anchor must default to writable, not probe an MLA-only field."""
        host = LogicalHostPool.__new__(LogicalHostPool)
        group = SimpleNamespace(anchor_entry=SimpleNamespace(host_pool=host))
        with mock.patch.object(HiCacheController, "__init__", return_value=None):
            controller = HybridCacheController(None, group, 1, None, mock.Mock())
        self.assertFalse(controller.mla_host_read_only)

    def test_transfer_plans_preserve_mamba_and_respect_mla_ownership(self):
        """Only leaders back up MLA; broadcast alone replaces follower MLA H2D."""
        indices = torch.arange(4)
        mamba = PoolTransfer(
            name=PoolName.MAMBA, host_indices=indices, device_indices=indices
        )
        for follower, write_mla, load_mla in (
            (False, True, True),
            (True, False, False),
        ):
            with self.subTest(follower=follower):
                controller = self._controller(
                    mla_host_read_only=follower,
                    node_leader_broadcast=_context([(3, 0)], is_src=not follower),
                )
                entries = controller.mem_pool_host.entry_map
                kv_pool = entries[PoolName.KV].host_pool
                mamba_pool = entries[PoolName.MAMBA].host_pool
                writes = controller._l2_transfers(indices, indices, [mamba])
                loads = controller._l2_load_transfers(indices, indices, [mamba])
                self.assertEqual(
                    [t.host_pool for t in writes],
                    [kv_pool, mamba_pool] if write_mla else [mamba_pool],
                )
                self.assertEqual(
                    [t.host_pool for t in loads],
                    [kv_pool, mamba_pool] if load_mla else [mamba_pool],
                )

    def test_follower_l3_prefetch_never_writes_the_shared_buffer(self):
        """Leader prefetch reads storage; followers acknowledge pages without writing them."""

        def base_attach(
            self,
            storage_backend,
            prefetch_threshold=256,
            model_name=None,
            storage_backend_extra_config=None,
        ):
            self.page_get_func = self._page_get_zero_copy

        for read_only in (True, False):
            with self.subTest(read_only=read_only):
                reads = []

                def batch_get(hash_values, host_indices, extra_info=None):
                    reads.extend(hash_values)
                    return [True, False]

                controller = self._controller(mla_host_read_only=read_only)
                controller.host_memory_mode = "cache"
                controller.pp_group = controller.pp_prefetch_command_group = None
                controller.storage_backend = SimpleNamespace(batch_get_v1=batch_get)
                with mock.patch.object(
                    HiCacheController, "attach_storage_backend", base_attach
                ):
                    controller.attach_storage_backend("mooncake")
                count = controller.page_get_func(
                    SimpleNamespace(request_id="prefetch"),
                    ["page0", "page1"],
                    torch.arange(2),
                )
                self.assertEqual(count, 2 if read_only else 1)
                self.assertEqual(reads, [] if read_only else ["page0", "page1"])


class TestNodeLeaderStoragePolicies(CustomTestCase):
    def _cache(self, *, attached=False, backend="mooncake"):
        events = []
        controller = SimpleNamespace(
            mem_pool_host=SimpleNamespace(entries=[]),
            storage_backend_type=backend if attached else None,
            enable_storage=attached,
            write_policy="write_through_selective",
        )

        def attach_backend(storage_backend, **kwargs):
            events.append("attach")
            controller.storage_backend_type = storage_backend
            controller.enable_storage = True

        def detach_backend():
            events.append("detach")
            controller.storage_backend_type = None
            controller.enable_storage = False

        controller.attach_storage_backend = attach_backend
        controller.detach_storage_backend = detach_backend
        controller.clear_storage_backend = lambda: True
        cache = SimpleNamespace(
            cache_controller=controller,
            enable_storage=attached,
            enable_storage_metrics=False,
            write_through_threshold=2,
            is_write_back=False,
            prefetch_stop_policy="best_effort",
            _enable_metrics_flag=False,
            extra_metric_labels={},
            sliding_window_size=None,
            host_memory_mode="cache",
            page_size=64,
            tree_core=SimpleNamespace(backfill_missing_hash_values=lambda: 0),
            drain_storage_control_queues_local=lambda: events.append("drain"),
            ongoing_prefetch={},
            ongoing_backup={},
            _storage_prefetch_hit_remaining_by_reqid={},
            prefetch_loaded_tokens_by_reqid={},
            prefetch_loaded_storage_start_by_reqid={},
            storage_prefetch_retries={},
            storage_existence_cache={},
        )
        cache._storage_attachment = StorageAttachment(cache)
        return cache, controller, events

    def test_per_node_startup_config_clear_and_shutdown_cleanup_remain_available(self):
        """Runtime admission must not block startup publication or atexit's local detach."""
        with mock.patch(
            "sglang.srt.mem_cache.unified_cache.storage_attachment.get_memory",
            return_value=SimpleNamespace(enable_mla_hicache_per_node=True),
        ):
            cache, controller, _ = self._cache()
            controller.enable_storage = True
            controller.storage_backend_type = "mooncake"
            cache._storage_attachment.apply_runtime_config(
                storage_backend="mooncake",
                prefetch_threshold=64,
                prefetch_timeout_base=1.0,
                prefetch_timeout_per_ki_token=0.25,
                hicache_storage_pass_prefix_keys=True,
                enable_storage=True,
                enable_storage_metrics=False,
                extra_metric_labels={},
            )
            self.assertTrue(cache.enable_storage)
            self.assertEqual(cache.prefetch_threshold, 64)
            self.assertTrue(UnifiedRadixCache.clear_storage_backend(cache))
            self.assertTrue(cache.enable_storage)
            UnifiedRadixCache.shutdown(cache)
            self.assertFalse(cache.enable_storage)
            self.assertFalse(controller.enable_storage)


if __name__ == "__main__":
    unittest.main()
