import threading
import unittest
from queue import Queue
from types import SimpleNamespace
from unittest.mock import MagicMock

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
from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


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
