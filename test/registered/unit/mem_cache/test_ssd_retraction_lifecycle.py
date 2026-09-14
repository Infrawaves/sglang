import ctypes
import threading
import unittest
from queue import Queue
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.managers.cache_controller import HiCacheController, StorageOperation
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    StorageOperation as HybridStorageOperation,
)
from sglang.srt.mem_cache.common import (
    RetractionBackup,
    RetractionStorageState,
    release_kv_cache,
)
from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorageExtraInfo,
    PoolName,
    PoolTransfer,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestMooncakeDcpKeyScoping(CustomTestCase):
    def test_storage_config_uses_attention_local_identity(self):
        from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool

        controller = HiCacheController.__new__(HiCacheController)
        controller.mem_pool_device = MLATokenToKVPool.__new__(MLATokenToKVPool)
        controller.mem_pool_host = SimpleNamespace(layout="layer_first")
        controller.enable_storage_metrics = False
        controller.get_attn_cp_rank_and_size = lambda: (0, 1)
        parallel = SimpleNamespace(
            tp_rank=7,
            tp_size=8,
            attn_tp_rank=3,
            attn_tp_size=4,
            pp_rank=0,
            pp_size=1,
            attn_dcp_rank=1,
            attn_dcp_size=2,
            attn_dp_rank=1,
            attn_dp_size=2,
        )
        module = "sglang.srt.managers.cache_controller"
        with (
            patch(f"{module}.get_parallel", return_value=parallel),
            patch(f"{module}.is_dp_attention_enabled", return_value=True),
            patch(f"{module}.get_attention_dp_rank", return_value=1),
        ):
            config = controller._generate_storage_config()
        self.assertEqual((config.tp_rank, config.tp_size), (3, 4))
        self.assertEqual((config.dcp_rank, config.dcp_size), (1, 2))
        self.assertEqual((config.attn_dp_rank, config.attn_dp_size), (1, 2))
        self.assertTrue(config.is_mla_model)

    def test_rank_scoped_write_read_exists_and_cleanup(self):
        from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost
        from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import (
            MooncakeStore,
        )

        for dcp_size, dp_size in ((1, 1), (2, 1), (2, 2)):
            with self.subTest(dcp_size=dcp_size, dp_size=dp_size):
                objects = {}

                def put(keys, pointers, sizes, config):
                    self.assertTrue(config.with_hard_pin)
                    for key, ptrs, lengths in zip(keys, pointers, sizes):
                        self.assertNotIn(key, objects)
                        objects[key] = [
                            ctypes.string_at(ptr, length)
                            for ptr, length in zip(ptrs, lengths)
                        ]
                    return [0] * len(keys)

                def get(keys, pointers, sizes):
                    for key, ptrs, lengths in zip(keys, pointers, sizes):
                        for ptr, length, data in zip(ptrs, lengths, objects[key]):
                            self.assertEqual(length, len(data))
                            ctypes.memmove(ptr, data, length)
                    return [sum(lengths) for lengths in sizes]

                def remove(keys, force):
                    self.assertTrue(force)
                    for key in keys:
                        del objects[key]
                    return [0] * len(keys)

                stores = {}
                keys = ["h0", "h1"]
                for dp_rank in range(dp_size):
                    # Two replicated DCP groups inside each attention-TP group.
                    for tp_rank in range(2 * dcp_size):
                        dcp_rank = tp_rank % dcp_size
                        pool = MLATokenToKVPoolHost.__new__(MLATokenToKVPoolHost)
                        pool.dcp_rank, pool.dcp_size = dcp_rank, dcp_size
                        pool.page_size, pool.size = 2, 8
                        pool.layout, pool.layer_num = "layer_first", 2
                        pool.kv_cache_dim, pool.dtype = 4, torch.float32
                        pool.kv_buffer = torch.zeros((2, 8, 1, 4))
                        payload = torch.arange(32).reshape(2, 4, 1, 4).float()
                        payload += 100 * dp_rank + 50 * dcp_rank
                        pool.kv_buffer[:, 2:6] = payload

                        store = MooncakeStore.__new__(MooncakeStore)
                        store.mem_pool_host = pool
                        store.is_mla_backend = True
                        store.enable_storage_metrics = False
                        store.config_prefix = "model"
                        store._use_group_semantics = False
                        store._replicate_config_cls = SimpleNamespace
                        store.mha_suffix, store.mla_suffix = (
                            MooncakeStore._build_key_suffixes(
                                tp_rank, 0, False, dcp_rank, dcp_size, dp_rank, dp_size
                            )
                        )
                        store.store = SimpleNamespace(
                            batch_put_from_multi_buffers=put,
                            batch_get_into_multi_buffers=get,
                            batch_is_exist=lambda ks: [int(k in objects) for k in ks],
                            batch_remove=remove,
                        )
                        stores[dp_rank, tp_rank] = store
                        if tp_rank < dcp_size:
                            self.assertEqual(
                                store.batch_set_v1(
                                    keys,
                                    torch.arange(2 * dcp_size, 6 * dcp_size),
                                    HiCacheStorageExtraInfo(pin=True),
                                ),
                                [True, True],
                            )
                        self.assertEqual(store.batch_exists(keys), 2)
                        pool.kv_buffer.zero_()
                        self.assertEqual(
                            store.batch_get_v1(
                                keys, torch.arange(4 * dcp_size, 8 * dcp_size)
                            ),
                            [True, True],
                        )
                        torch.testing.assert_close(pool.kv_buffer[:, 4:8], payload)
                        self.assertEqual(pool.kv_buffer[:, :4].count_nonzero(), 0)

                self.assertEqual(len(objects), 2 * dp_size * dcp_size)
                for dp_rank in range(dp_size):
                    for dcp_rank in range(dcp_size):
                        store = stores[dp_rank, dcp_rank]
                        self.assertEqual(
                            store.batch_remove_v2(
                                [PoolTransfer(name=PoolName.KV, keys=keys)]
                            ),
                            [],
                        )
                        self.assertEqual(store.batch_exists(keys), 0)
                    # Deleting one DP replica must leave the next replica intact.
                    for remaining_dp in range(dp_rank + 1, dp_size):
                        self.assertEqual(stores[remaining_dp, 0].batch_exists(keys), 2)
                self.assertFalse(objects)

    def test_dcp1_suffixes_are_byte_compatible(self):
        from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import (
            MooncakeStore,
        )

        self.assertEqual(
            MooncakeStore._build_key_suffixes(2, 1, True, 0, 1, 0, 4),
            ("2_1", "1"),
        )

    def test_dcp_and_dp_suffixes_scope_mla_and_mamba(self):
        from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import (
            MooncakeStore,
        )

        self.assertEqual(
            MooncakeStore._build_key_suffixes(3, 0, False, 1, 4, 0, 1),
            ("3", "dcp1_4"),
        )
        self.assertEqual(
            MooncakeStore._build_key_suffixes(3, 2, True, 1, 4, 1, 2),
            ("3_2_dp1_2", "2_dp1_2_dcp1_4"),
        )

    def test_hybrid_components_use_scoped_suffixes(self):
        from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import (
            MooncakeStore,
        )

        store = MooncakeStore.__new__(MooncakeStore)
        store.mla_suffix = "dp1_2_dcp1_4"
        store.mha_suffix = "3_dp1_2"
        store.is_mla_backend = True
        store.registered_pools = {
            PoolName.KV: SimpleNamespace(),
            PoolName.MAMBA: SimpleNamespace(
                conv_buffer=[object()], temporal_state_elem_size=1
            ),
        }
        kv_keys, _ = store._get_hybrid_page_component_keys(
            ["h0"], PoolTransfer(name=PoolName.KV, keys=["h0"])
        )
        mamba_keys, _ = store._get_hybrid_page_component_keys(
            ["h0"], PoolTransfer(name=PoolName.MAMBA, keys=["h0"])
        )
        self.assertEqual(kv_keys, ["h0_dp1_2_dcp1_4_k"])
        self.assertEqual(mamba_keys, ["h0_3_dp1_2_temporal", "h0_3_dp1_2_conv_0"])

    def test_replicated_writer_is_selected_per_dcp_shard(self):
        from sglang.srt.mem_cache.storage import StorageBackendFactory

        cases = [
            (True, 0, 1, False),
            (True, 1, 1, True),
            (True, 3, 4, False),
            (True, 4, 4, True),
            (False, 5, 4, False),
        ]
        for is_mla, local_tp_rank, dcp_size, expected_skip in cases:
            controller = HiCacheController.__new__(HiCacheController)
            controller.enable_storage = False
            controller._stop_storage_threads = lambda: None
            controller.prefetch_hits_sync_groups = []
            controller.prefetch_completion_sync_groups = []
            controller._destroy_sync_groups = lambda _groups: None
            controller._generate_storage_config = lambda *_args, **_kwargs: (
                SimpleNamespace(
                    is_mla_model=is_mla,
                    tp_rank=local_tp_rank,
                    dcp_size=dcp_size,
                )
            )
            controller.storage_host_pool = object()
            with patch.object(
                StorageBackendFactory,
                "create_backend",
                side_effect=RuntimeError("stop after writer selection"),
            ):
                with self.assertRaisesRegex(RuntimeError, "stop after writer"):
                    controller.attach_storage_backend("mooncake")
            self.assertEqual(controller.backup_skip, expected_skip)


class TestRetractionStorageState(CustomTestCase):
    def test_state_registry_is_typed_and_complete(self):
        self.assertEqual(
            set(RetractionStorageState),
            {
                RetractionStorageState.L2_READY,
                RetractionStorageState.L3_WRITE_PENDING,
                RetractionStorageState.L3_READY,
                RetractionStorageState.L3_WRITE_FAILED,
                RetractionStorageState.DISCARD_PENDING,
                RetractionStorageState.RELEASED,
            },
        )
        self.assertNotEqual(RetractionStorageState.L3_READY, "L3_READY")

    def test_request_and_registry_share_one_mutable_record(self):
        backup = RetractionBackup()
        registry = {7: backup}
        backup.storage_state = RetractionStorageState.L3_WRITE_PENDING
        self.assertIs(registry[7], backup)
        self.assertIs(
            registry[7].storage_state,
            RetractionStorageState.L3_WRITE_PENDING,
        )


class TestRetractionPinForwarding(CustomTestCase):
    """The pin flag must survive the whole write path.

    A demoted request's KV is only protected because the backend is told to
    pin it. The request is dropped from the running batch on a successful
    write, so a flag lost anywhere between write_storage and the backend
    yields an evictable object that nothing detects until the resume read
    fails. Every hop is asserted for that reason.
    """

    def _run_page_backup(self, *, pin: bool):
        controller = HiCacheController.__new__(HiCacheController)
        controller.page_size = 2
        captured = []

        def fake_page_set_func(hashes, host_indices, extra_info):
            captured.append(extra_info)
            return True

        controller.page_set_func = fake_page_set_func

        operation = StorageOperation(
            host_indices=torch.arange(4, dtype=torch.int64),
            token_ids=[1, 2, 3, 4],
            hash_value=["h0", "h1"],
            pin=pin,
        )
        controller._page_backup(operation)
        return operation, captured

    def test_pin_reaches_the_backend_when_requested(self):
        operation, captured = self._run_page_backup(pin=True)
        self.assertTrue(operation.pin)
        self.assertTrue(captured)
        self.assertTrue(
            all(info.pin for info in captured),
            "pin was dropped before reaching the storage backend",
        )

    def test_ordinary_write_through_does_not_pin(self):
        """Negative branch: a write that never asked for a pin must not get
        one, or ordinary hicache traffic would become unevictable."""
        operation, captured = self._run_page_backup(pin=False)
        self.assertFalse(operation.pin)
        self.assertTrue(captured)
        self.assertFalse(any(info.pin for info in captured))


class TestRetractionHardPin(CustomTestCase):
    """The backend contract is a hard pin; a soft pin must not satisfy it."""

    def _make_store(self, *, supports_hard_pin):
        from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import (
            MooncakeStore,
        )

        store = MooncakeStore.__new__(MooncakeStore)
        store._supports_hard_pin = supports_hard_pin
        store._replicate_config_cls = SimpleNamespace
        store._use_group_semantics = False
        store.store = MagicMock()
        store.store.batch_put_from.return_value = [0]
        store.store.batch_remove.return_value = [0, 0]
        store.config_prefix = None
        store.is_mla_backend = False
        store.should_split_heads = False
        store.mha_suffix = "0"
        store.mem_pool_host = SimpleNamespace(kv_buffer=object())
        return store

    def test_supports_pin_requires_hard_pin(self):
        self.assertFalse(self._make_store(supports_hard_pin=False).supports_pin())
        self.assertTrue(self._make_store(supports_hard_pin=True).supports_pin())

    def test_pinned_put_sets_with_hard_pin(self):
        store = self._make_store(supports_hard_pin=True)
        store._put_batch_zero_copy_impl(["k"], [1], [2], pin=True)
        config = store.store.batch_put_from.call_args.args[3]
        self.assertTrue(config.with_hard_pin)

    def test_batch_remove_v2_forces_expands_and_reports_refusals(self):
        """A restore read leaves a lease, so the delete must force and name the
        k/v objects batch_set_v1 wrote. A refused delete comes back by logical
        key so the hard pin can be retried; not-found was never pinned."""
        store = self._make_store(supports_hard_pin=True)
        # h0: k removed, v refused; h1: both not found (a write that never landed).
        store.store.batch_remove.return_value = [0, -703, -704, -704]
        failed = store.batch_remove_v2(
            [PoolTransfer(name=PoolName.KV, keys=["h0", "h1"])]
        )
        keys, force = store.store.batch_remove.call_args.args
        self.assertEqual(keys, ["h0_0_k", "h0_0_v", "h1_0_k", "h1_0_v"])
        self.assertTrue(force)
        self.assertEqual([(t.name, t.keys) for t in failed], [(PoolName.KV, ["h0"])])


class TestRetractionRestoreAdmission(CustomTestCase):
    def _make_cache(self, *, available, reclaimed):
        from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

        cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
        # page_size is a property forwarding to tree_core.
        cache.tree_core = SimpleNamespace(page_size=16)
        free = {"n": available}

        def reclaim(_need):
            free["n"] = reclaimed

        cache.host_pool_group = SimpleNamespace(
            available_size=lambda pool=None: free["n"],
            entry_map={},
        )
        cache._reclaim_retraction_host = reclaim
        return cache

    def _req(self, backup):
        return SimpleNamespace(seqlen=17, kv=SimpleNamespace(retraction_backup=backup))

    def test_pending_write_is_not_admissible(self):
        cache = self._make_cache(available=1000, reclaimed=1000)
        backup = RetractionBackup(
            storage_state=RetractionStorageState.L3_WRITE_PENDING,
            storage_hashes=["h0"],
        )
        self.assertFalse(cache.retraction_restore_admissible(self._req(backup)))

    def test_l3_ready_admits_after_reclaim_only_when_span_fits(self):
        """The staging span is ceil_align(seqlen - 1): the boundary token has
        no KV yet, so seqlen 17 needs exactly 16 slots, not 17 or 32."""
        backup = RetractionBackup(
            storage_state=RetractionStorageState.L3_READY, storage_hashes=["h0"]
        )
        self.assertTrue(
            self._make_cache(available=0, reclaimed=16).retraction_restore_admissible(
                self._req(backup)
            )
        )
        self.assertFalse(
            self._make_cache(available=0, reclaimed=15).retraction_restore_admissible(
                self._req(backup)
            )
        )

    def test_side_pool_admission_uses_logical_page_size(self):
        cache = self._make_cache(available=512, reclaimed=512)
        cache.host_pool_group.entry_map[PoolName.SWA] = SimpleNamespace(
            host_pool=SimpleNamespace(page_size=64, logical_page_size=512),
            host_evict_fn=lambda _need: None,
        )
        backup = RetractionBackup(
            storage_state=RetractionStorageState.L3_READY,
            storage_hashes=["h0"],
            pool_transfers=[
                PoolTransfer(
                    name=PoolName.SWA,
                    keys=["s0", "s1"],
                    host_indices=torch.arange(1024),
                )
            ],
        )
        self.assertFalse(cache.retraction_restore_admissible(self._req(backup)))

    def test_side_pool_storage_transfer_uses_logical_page_size(self):
        from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

        cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
        cache.host_pool_group = SimpleNamespace(
            entry_map={
                PoolName.SWA: SimpleNamespace(
                    host_pool=SimpleNamespace(page_size=64, logical_page_size=512)
                )
            }
        )
        transfer = PoolTransfer(
            name=PoolName.SWA,
            host_indices=torch.arange(1024),
        )
        backup = RetractionBackup(pool_transfers=[transfer])

        transfers = cache._retraction_storage_transfers(backup, ["k0", "k1", "k2"])
        self.assertEqual(transfers[0].keys, ["k1", "k2"])


class TestMooncakeLogicalPageAccounting(CustomTestCase):
    def _make_store(self):
        from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import (
            MooncakeStore,
        )

        store = MooncakeStore.__new__(MooncakeStore)
        store.mem_pool_host = SimpleNamespace(
            page_size=64,
            logical_page_size=512,
            get_page_buffer_meta=lambda indices: ([1], [2]),
        )
        store.registered_pools = {
            PoolName.KV: store.mem_pool_host,
        }
        store.is_mla_backend = False
        store.should_split_heads = False
        store._tag_keys = lambda keys: keys
        store._get_hybrid_page_component_keys = lambda keys, transfer: (keys, 1)
        store._can_use_group_semantics = lambda: False
        store._batch_exist = lambda keys: [1 for _ in keys]
        store._get_batch_zero_copy_impl = lambda keys, ptrs, sizes: [0 for _ in keys]
        store._batch_postprocess = lambda results, **kwargs: results
        return store

    def test_batch_preprocess_uses_logical_page_size(self):
        store = self._make_store()
        store._get_mha_buffer_meta = MagicMock(return_value=(["h"], [1], [2]))
        result = store._batch_preprocess(["h"], torch.arange(512))
        self.assertEqual(result, (["h"], [1], [2]))
        store._get_mha_buffer_meta.assert_called_once()

    def test_batch_io_uses_logical_page_size(self):
        store = self._make_store()
        transfer = PoolTransfer(
            name=PoolName.KV,
            keys=["h"],
            host_indices=torch.arange(512),
        )
        result = store._batch_io_v2([transfer], is_set=False)
        self.assertEqual(result, {PoolName.KV: [0]})


class TestRetractionRestoreCollective(CustomTestCase):
    def _make_cache(self, *, page_hits=1, sidecar_hits=None):
        from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

        cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
        cache.tree_core = SimpleNamespace(enable_storage=True, page_size=2)
        cache.cache_controller = SimpleNamespace(
            page_get_func=MagicMock(return_value=page_hits),
            storage_backend=MagicMock(),
        )
        if sidecar_hits is not None:
            cache.cache_controller.storage_backend.batch_get_v2.return_value = {
                PoolName.SWA: sidecar_hits
            }
        cache.host_pool_group = MagicMock()
        cache.host_pool_group.alloc.return_value = torch.arange(2, dtype=torch.int64)
        cache.host_pool_group.resolve_host_transfers.return_value = [
            PoolTransfer(name=PoolName.SWA, keys=["s0"])
        ]
        cache._reclaim_retraction_host = MagicMock()
        cache._retraction_device_transfers = MagicMock(
            return_value=(
                torch.arange(2, dtype=torch.int64),
                [
                    PoolTransfer(
                        name=PoolName.KV,
                        device_indices=torch.arange(2, dtype=torch.int64),
                    ),
                    PoolTransfer(
                        name=PoolName.SWA,
                        device_indices=torch.arange(2, dtype=torch.int64),
                    ),
                ],
            )
        )
        cache.retraction_restore = MagicMock()
        cache.retraction_ssd_backups = {}
        cache.retraction_ssd_requests = {}
        return cache

    def _req_backup(self, cache):
        backup = RetractionBackup(
            storage_operation_id=17,
            storage_hashes=["h0"],
            pool_transfers=[
                PoolTransfer(name=PoolName.KV, keys=["h0"]),
                PoolTransfer(name=PoolName.SWA, keys=["s0"]),
            ],
            storage_state=RetractionStorageState.L3_READY,
        )
        req = SimpleNamespace(
            rid=3, seqlen=3, kv=SimpleNamespace(retraction_backup=None)
        )
        cache.retraction_ssd_backups[17] = backup
        cache.retraction_ssd_requests[17] = req
        return req, backup

    def _install_all_reduce(self, cache, backup, *, reduced_success):
        calls = []

        def fake_all_reduce(tensor, _op):
            self.assertIs(backup.storage_state, RetractionStorageState.L3_READY)
            calls.append(tensor.clone())
            tensor[0] = int(reduced_success)

        cache._all_reduce = fake_all_reduce
        return calls

    def test_restore_consensus_all_success_commits(self):
        cache = self._make_cache(sidecar_hits=[True])
        req, backup = self._req_backup(cache)
        calls = self._install_all_reduce(cache, backup, reduced_success=True)

        self.assertTrue(cache.retraction_restore_ssd(req, backup))
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].tolist(), [1])
        cache.retraction_restore.assert_called_once()
        self.assertIs(backup.storage_state, RetractionStorageState.RELEASED)
        self.assertNotIn(17, cache.retraction_ssd_backups)
        self.assertNotIn(17, cache.retraction_ssd_requests)

    def test_restore_consensus_asymmetric_read_rolls_back_staging(self):
        cache = self._make_cache(page_hits=0, sidecar_hits=[True])
        req, backup = self._req_backup(cache)
        calls = self._install_all_reduce(cache, backup, reduced_success=False)

        self.assertFalse(cache.retraction_restore_ssd(req, backup))
        self.assertEqual(len(calls), 1)
        cache.host_pool_group.free.assert_called_once()
        cache.host_pool_group.release_transfers.assert_called_once()
        cache.retraction_restore.assert_not_called()
        self.assertIs(backup.storage_state, RetractionStorageState.L3_READY)
        cache.cache_controller.storage_backend.batch_remove_v2.assert_not_called()

    def test_restore_collective_participates_after_early_page_failure(self):
        cache = self._make_cache(page_hits=0, sidecar_hits=[True])
        req, backup = self._req_backup(cache)
        calls = self._install_all_reduce(cache, backup, reduced_success=False)

        self.assertFalse(cache.retraction_restore_ssd(req, backup))
        self.assertEqual(len(calls), 1)


class TestBackupThreadAcksRaisedWrite(CustomTestCase):
    def test_raised_write_is_acked_as_failure(self):
        """A backend exception used to kill the backup thread with no ACK,
        stranding the demoted request in L3_WRITE_PENDING; it must be acked
        as a failed write instead."""
        controller = HybridCacheController.__new__(HybridCacheController)
        controller.backup_queue = Queue()
        controller.ack_backup_queue = Queue()
        controller.storage_stop_event = threading.Event()

        def raise_and_stop(_operation):
            controller.storage_stop_event.set()
            raise RuntimeError("rpc down")

        controller._page_backup = raise_and_stop
        operation = HybridStorageOperation(None, [], hash_value=["h0"])
        operation.completed_tokens = 2
        operation.pool_storage_result.update_extra_pool_hit_pages({"mamba": 1})
        controller.backup_queue.put(operation)

        controller.backup_thread_func()

        self.assertIs(controller.ack_backup_queue.get_nowait(), operation)
        self.assertEqual(operation.completed_tokens, 0)
        self.assertEqual(operation.pool_storage_result.extra_pool_hit_pages, {})


class TestRetractionL3Lifetime(CustomTestCase):
    """L3 objects outlive resume and are deleted once, at the terminal release."""

    def _make_cache(self, *, backup_skip=False):
        from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

        cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
        # enable_storage is a property forwarding to tree_core.
        cache.tree_core = SimpleNamespace(enable_storage=True)
        cache.retraction_l3_orphans = {}
        cache.cache_controller = SimpleNamespace(
            backup_skip=backup_skip,
            should_backup=lambda transfer: True,
            storage_backend=MagicMock(),
        )
        return cache

    def test_keys_accumulate_across_demotions_and_release_once(self):
        """A re-demotion rewrites only its tail plus the earlier partial last
        page; the union of both cycles must go to the backend in one delete."""
        cache = self._make_cache()
        req = SimpleNamespace(kv=SimpleNamespace(retraction_l3_keys=None))
        first = RetractionBackup(
            storage_hashes=["p0", "p1_partial"],
            pool_transfers=[PoolTransfer(name=PoolName.SWA, keys=["p1_partial"])],
        )
        second = RetractionBackup(
            storage_hashes=["p0", "p1", "p2"],
            pool_transfers=[PoolTransfer(name=PoolName.SWA, keys=["p2"])],
        )
        cache._record_retraction_l3(req, first)
        cache._record_retraction_l3(req, second)
        cache.release_retraction_l3(req)

        remove = cache.cache_controller.storage_backend.batch_remove_v2
        (transfers,), _ = remove.call_args
        by_pool = {transfer.name: transfer.keys for transfer in transfers}
        self.assertEqual(by_pool[PoolName.KV], ["p0", "p1", "p1_partial", "p2"])
        self.assertEqual(by_pool[PoolName.SWA], ["p1_partial", "p2"])
        self.assertIsNone(req.kv.retraction_l3_keys)
        cache.release_retraction_l3(req)
        remove.assert_called_once()

    def test_undeletable_keys_ride_along_with_the_next_release(self):
        """A refused delete used to drop the only record of a hard-pinned
        object, leaking it forever; the keys must be kept and retried."""
        cache = self._make_cache()
        remove = cache.cache_controller.storage_backend.batch_remove_v2
        remove.return_value = [PoolTransfer(name=PoolName.KV, keys=["p0"])]
        for hashes in (["p0"], ["p1"]):
            req = SimpleNamespace(kv=SimpleNamespace(retraction_l3_keys=None))
            cache._record_retraction_l3(req, RetractionBackup(storage_hashes=hashes))
            cache.release_retraction_l3(req)
            remove.return_value = []
        # The refused p0 rides along with p1's release and is then forgotten.
        (transfers,), _ = remove.call_args
        self.assertEqual(
            {t.name: t.keys for t in transfers}, {PoolName.KV: ["p0", "p1"]}
        )
        self.assertEqual(cache.retraction_l3_orphans, {})

    def test_non_writer_rank_records_no_primary_keys(self):
        """Replicated MLA KV is written by TP0 only; another rank deleting it
        would race TP0's own restore read."""
        cache = self._make_cache(backup_skip=True)
        req = SimpleNamespace(kv=SimpleNamespace(retraction_l3_keys=None))
        cache._record_retraction_l3(req, RetractionBackup(storage_hashes=["p0"]))
        self.assertEqual(req.kv.retraction_l3_keys, {})

    def test_release_kv_cache_drops_l3_only_on_terminal_release(self):
        """A retraction sets retraction_backup before it releases device KV, so
        a release without one is the request's end and the only delete point."""
        tree_cache = MagicMock()
        kv = SimpleNamespace(
            holds_kv=False,
            is_kv_released=True,
            holds_mamba=False,
            retraction_l3_keys={PoolName.KV: {"p0"}},
            retraction_backup=RetractionBackup(),
        )
        req = SimpleNamespace(kv=kv)

        release_kv_cache(req, tree_cache, is_insert=False)
        tree_cache.release_retraction_l3.assert_not_called()

        kv.retraction_backup = None
        release_kv_cache(req, tree_cache)
        tree_cache.release_retraction_l3.assert_called_once_with(req)


if __name__ == "__main__":
    unittest.main()
