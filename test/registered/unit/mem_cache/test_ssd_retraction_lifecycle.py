import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.srt.managers.cache_controller import HiCacheController, StorageOperation
from sglang.srt.mem_cache.common import (
    RetractionBackup,
    RetractionStorageState,
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

    def test_batch_remove_v2_forces_and_expands_primary_keys(self):
        """A restore read leaves a lease on the objects, so the delete must
        force; and it must name the same k/v objects batch_set_v1 wrote."""
        store = self._make_store(supports_hard_pin=True)
        store.batch_remove_v2([PoolTransfer(name=PoolName.KV, keys=["h0"])])
        keys, force = store.store.batch_remove.call_args.args
        self.assertEqual(keys, ["h0_0_k", "h0_0_v"])
        self.assertTrue(force)


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


if __name__ == "__main__":
    unittest.main()
