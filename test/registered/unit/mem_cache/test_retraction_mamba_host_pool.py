import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.srt.mem_cache.common import (
    RetractionBackup,
    RetractionStorageState,
    retraction_discard,
)
from sglang.srt.mem_cache.hicache_storage import (
    PoolName,
    PoolTransfer,
    PoolTransferResult,
)
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    StorageOperation as HybridStorageOperation,
)
from sglang.srt.mem_cache.memory_pool import (
    HybridLinearKVPool,
    MHATokenToKVPool,
    MLATokenToKVPool,
)
from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import MooncakeStore
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _make_cache(
    *,
    page_size: int,
    mamba: bool,
    swa: bool = False,
    entry_names=(PoolName.KV, PoolName.MAMBA),
    kv_cache=None,
) -> UnifiedRadixCache:
    cache = object.__new__(UnifiedRadixCache)
    # page_size is a property forwarding to tree_core.
    cache.tree_core = SimpleNamespace(page_size=page_size)
    cache.is_mamba_enabled = mamba
    cache.is_swa_enabled = swa
    cache.sidecar_pool_specs = []
    cache.cache_controller = MagicMock()
    cache.host_pool_group = MagicMock()
    cache.host_pool_group.entry_map = {
        name: SimpleNamespace(host_pool=SimpleNamespace(page_size=1, size=8))
        for name in entry_names
    }
    cache.token_to_kv_pool_allocator = SimpleNamespace(get_kvcache=lambda: kv_cache)
    return cache


def _make_req(*, seqlen: int, mamba_slot: int):
    req_to_token = torch.arange(1, 64, dtype=torch.int64).unsqueeze(0)
    req = SimpleNamespace(
        rid="req",
        seqlen=seqlen,
        kv=SimpleNamespace(
            req_pool_idx=0,
            mamba_pool_idx=torch.tensor(mamba_slot),
            holds_mamba=True,
            mamba_needs_clear=True,
            retraction_l3_keys=None,
        ),
    )
    return req, SimpleNamespace(req_to_token=req_to_token)


class TestRetractionMambaTransfers(CustomTestCase):
    def test_mamba_transfer_is_never_padded(self):
        """The KV run is padded to a page because _pre_alloc hands out
        page-aligned contiguous runs; a mamba slot has no such invariant
        (MambaPoolHost.page_size == 1), so padding would fabricate slot ids
        and copy another request's state."""
        cache = _make_cache(page_size=4, mamba=True)
        req, cache.req_to_token_pool = _make_req(seqlen=7, mamba_slot=5)

        full_indices, extra = cache._retraction_device_transfers(req)

        self.assertEqual(len(full_indices), 8)
        self.assertEqual([t.name for t in extra], [PoolName.MAMBA])
        mamba = extra[0]
        self.assertEqual(mamba.device_indices.tolist(), [5])
        self.assertEqual(mamba.device_indices.dtype, torch.int64)
        self.assertIsNone(mamba.host_indices)

    def test_mamba_transfer_requires_a_held_slot(self):
        """A request that released its slot has no state to back up; silently
        emitting a transfer for `None` would surface as a tensor error deep in
        the L2 engine instead of at the call site."""
        cache = _make_cache(page_size=1, mamba=True)
        req, cache.req_to_token_pool = _make_req(seqlen=3, mamba_slot=1)
        req.kv.holds_mamba = False
        with self.assertRaises(AssertionError):
            cache._retraction_device_transfers(req)

    def test_mamba_storage_transfer_takes_the_final_key(self):
        """One host slot with page_size 1 must resolve to the last KV page key,
        the same convention MambaComponent uses for BACKUP_STORAGE; any other
        key would be unreadable by the ordinary mamba prefetch path."""
        cache = _make_cache(page_size=1, mamba=True)
        backup = RetractionBackup(
            pool_transfers=[
                PoolTransfer(name=PoolName.MAMBA, host_indices=torch.tensor([3]))
            ]
        )

        transfers = cache._retraction_storage_transfers(backup, ["h0", "h1", "h2"])

        self.assertEqual(transfers[0].keys, ["h2"])

        too_many = RetractionBackup(
            pool_transfers=[
                PoolTransfer(
                    name=PoolName.MAMBA, host_indices=torch.tensor([0, 1, 2, 3])
                )
            ]
        )
        with self.assertRaises(ValueError):
            cache._retraction_storage_transfers(too_many, ["h0", "h1", "h2"])


class TestSupportsRetractionBackupPoolMatrix(CustomTestCase):
    """Completeness contract: a KV pool class only qualifies when every host
    pool it needs is registered, so adding a pool class without its host
    entry stays red."""

    def test_hybrid_linear_with_mamba_entry_is_supported(self):
        cache = _make_cache(
            page_size=1, mamba=True, kv_cache=object.__new__(HybridLinearKVPool)
        )
        self.assertTrue(cache.supports_retraction_backup())

    def test_hybrid_linear_without_mamba_entry_is_rejected(self):
        cache = _make_cache(
            page_size=1,
            mamba=True,
            entry_names=(PoolName.KV,),
            kv_cache=object.__new__(HybridLinearKVPool),
        )
        self.assertFalse(cache.supports_retraction_backup())

    def test_mamba_plus_swa_is_rejected(self):
        cache = _make_cache(
            page_size=1,
            mamba=True,
            swa=True,
            entry_names=(PoolName.KV, PoolName.SWA, PoolName.MAMBA),
            kv_cache=object.__new__(HybridLinearKVPool),
        )
        self.assertFalse(cache.supports_retraction_backup())

    def test_standalone_mla_is_not_admitted_this_round(self):
        cache = _make_cache(
            page_size=1,
            mamba=False,
            entry_names=(PoolName.KV,),
            kv_cache=object.__new__(MLATokenToKVPool),
        )
        self.assertFalse(cache.supports_retraction_backup())

    def test_mha_and_swa_paths_are_unchanged(self):
        mha = _make_cache(
            page_size=1,
            mamba=False,
            entry_names=(PoolName.KV,),
            kv_cache=object.__new__(MHATokenToKVPool),
        )
        self.assertTrue(mha.supports_retraction_backup())
        swa_missing_entry = _make_cache(
            page_size=1,
            mamba=False,
            swa=True,
            entry_names=(PoolName.KV,),
            kv_cache=object.__new__(SWAKVPool),
        )
        self.assertFalse(swa_missing_entry.supports_retraction_backup())


class TestMambaKeysAreRankScopedOnMooncake(CustomTestCase):
    def test_mamba_objects_carry_the_rank_while_mla_kv_does_not(self):
        """Every TP rank writes its own mamba shard under the same page key
        (should_backup(MAMBA) is True everywhere) and Mooncake skips a put for
        a key that already exists, so a rank-free mamba suffix would let rank
        0's shard silently stand in for every other rank's state."""
        store = object.__new__(MooncakeStore)
        store.mha_suffix = "1"
        store.mla_suffix = ""
        store.registered_pools = {
            PoolName.MAMBA: SimpleNamespace(
                conv_buffer=[object()], temporal_state_elem_size=16
            ),
            PoolName.KV: object(),
        }

        mamba_keys, mamba_multiplier = store._get_hybrid_page_component_keys(
            ["h2"], PoolTransfer(name=PoolName.MAMBA)
        )
        kv_keys, kv_multiplier = store._get_hybrid_page_component_keys(
            ["h2"], PoolTransfer(name=PoolName.KV)
        )

        self.assertEqual(mamba_multiplier, 2)
        self.assertEqual(mamba_keys, ["h2_1_temporal", "h2_1_conv_0"])
        self.assertEqual(kv_multiplier, 1)
        self.assertEqual(kv_keys, ["h2__k"])


class TestRetractionRestoreClearsMambaNeedsClear(CustomTestCase):
    def test_needs_clear_is_false_after_restore(self):
        """HybridReqToTokenPool.alloc arms the deferred clear on the fresh
        slot _pre_alloc hands the resumed request. Nothing on the decode side
        consumes it today, but any future collector would zero the state this
        restore just copied back."""
        cache = _make_cache(page_size=1, mamba=True)
        req, cache.req_to_token_pool = _make_req(seqlen=3, mamba_slot=2)
        controller = cache.cache_controller
        controller._resolve_device_transfers.side_effect = lambda transfers, **_: (
            transfers
        )
        controller._move_op_indices.side_effect = lambda op: (
            op.host_indices,
            op.device_indices,
            op.pool_transfers,
        )
        controller.layer_num = 1
        backup = RetractionBackup(
            host_indices=torch.tensor([7, 8]),
            pool_transfers=[
                PoolTransfer(name=PoolName.MAMBA, host_indices=torch.tensor([4]))
            ],
        )

        cache.retraction_restore(req, backup)

        self.assertFalse(req.kv.mamba_needs_clear)
        controller.l2_transfer_engine.submit_host_to_device.assert_called_once()
        cache.host_pool_group.free.assert_called_once()
        cache.host_pool_group.release_transfers.assert_called_once()


def _make_ssd_cache(*, backup_skip: bool = False) -> UnifiedRadixCache:
    """Stub every collaborator of the SSD lifecycle so the MAMBA transfer can be
    followed from D2H staging through the L3 write, the ACK, the L3 read and
    the terminal delete without a GPU or a storage backend."""
    cache = _make_cache(page_size=2, mamba=True)
    cache.tree_core.enable_storage = True
    cache.host_memory_mode = "cache"
    cache.disable = True
    cache.retraction_ssd_backups = {}
    cache.retraction_ssd_requests = {}
    cache.retraction_l3_orphans = {}

    def resolve(transfers, *, primary_device_indices=None, primary_host_indices=None):
        for transfer in transfers or []:
            if transfer.indices_from_pool is None and transfer.host_indices is None:
                transfer.host_indices = torch.tensor([42], dtype=torch.int64)
        return transfers

    group = cache.host_pool_group
    group.alloc.side_effect = lambda n, **_: torch.arange(n, dtype=torch.int64)
    group.resolve_host_transfers.side_effect = resolve
    group.available_size.return_value = 100

    controller = cache.cache_controller
    controller.backup_skip = backup_skip
    controller.layer_num = 1
    controller.should_backup.return_value = True
    controller.storage_backend.supports_pin.return_value = True
    controller.get_hash_str.side_effect = lambda ids, seed, page_size: [
        f"h{i}" for i in range(len(ids) // page_size)
    ]
    controller.write_storage.return_value = 7
    controller._move_write_operation.side_effect = lambda op: (
        op.host_indices,
        op.device_indices,
        op.pool_transfers,
    )
    controller._move_op_indices.side_effect = lambda op: (
        op.host_indices,
        op.device_indices,
        op.pool_transfers,
    )
    controller._resolve_device_transfers.side_effect = lambda t, **_: t
    controller.page_get_func.side_effect = lambda op, hashes, idx, extra: len(hashes)
    controller.storage_backend.batch_get_v2.return_value = {PoolName.MAMBA: [True]}
    return cache


def _make_ssd_req(*, mamba_slot: int):
    # seqlen 5 -> 4 committed tokens -> 2 pages of size 2 -> keys h0, h1.
    req, pool = _make_req(seqlen=5, mamba_slot=mamba_slot)
    req.origin_input_ids = [1, 2, 3]
    req.output_ids = [4, 5]
    req.kv.retraction_backup = None
    return req, pool


def _write_op(*, completed_tokens: int, mamba_pages: int) -> HybridStorageOperation:
    op = object.__new__(HybridStorageOperation)
    op.completed_tokens = completed_tokens
    op.pool_storage_result = PoolTransferResult(0, {PoolName.MAMBA: mamba_pages})
    return op


class TestMambaSsdLifecycle(CustomTestCase):
    """Bookkeeping across the whole SSD path: the one MAMBA slot must be written,
    counted, read and deleted as its own object, or a rank silently loses its
    recurrent-state shard while the KV pages look complete."""

    def test_mamba_slot_rides_every_storage_call(self):
        cache = _make_ssd_cache()
        req, cache.req_to_token_pool = _make_ssd_req(mamba_slot=3)

        backup = cache.retraction_backup_ssd(req)

        self.assertIs(backup.storage_state, RetractionStorageState.L3_WRITE_PENDING)
        self.assertIs(cache.retraction_ssd_backups[7], backup)
        write_kwargs = cache.cache_controller.write_storage.call_args.kwargs
        self.assertTrue(write_kwargs["pin"])
        self.assertEqual(write_kwargs["hash_value"], ["h0", "h1"])
        (mamba_write,) = write_kwargs["extra_pools"]
        self.assertEqual(mamba_write.name, PoolName.MAMBA)
        self.assertEqual(mamba_write.keys, ["h1"])
        self.assertEqual(mamba_write.host_indices.tolist(), [42])

        # A complete KV write with a missing mamba object must not release L2.
        self.assertFalse(
            cache._retraction_write_succeeded(
                _write_op(completed_tokens=4, mamba_pages=0), backup
            )
        )
        self.assertTrue(
            cache._retraction_write_succeeded(
                _write_op(completed_tokens=4, mamba_pages=1), backup
            )
        )

        cache._finish_retraction_ssd_backup(7, True)
        self.assertIs(backup.storage_state, RetractionStorageState.L3_READY)
        self.assertIsNone(backup.host_indices)
        self.assertIsNone(mamba_write.host_indices)
        self.assertEqual(mamba_write.keys, ["h1"])
        self.assertIs(req.kv.retraction_backup, backup)
        cache.host_pool_group.release_transfers.assert_called_once()

        self.assertTrue(cache.retraction_restore_admissible(req))
        cache.host_pool_group.available_size.assert_any_call(pool=PoolName.MAMBA)

        # _pre_alloc hands the resumed request a fresh slot before restore.
        req.kv.mamba_pool_idx = torch.tensor(6)
        req.kv.mamba_needs_clear = True
        cache.retraction_restore_ssd(req, backup)

        (sidecars,) = cache.cache_controller.storage_backend.batch_get_v2.call_args.args
        self.assertEqual([t.name for t in sidecars], [PoolName.MAMBA])
        self.assertEqual(sidecars[0].keys, ["h1"])
        self.assertEqual(len(sidecars[0].host_indices), 1)
        cache.cache_controller.l2_transfer_engine.submit_host_to_device.assert_called_once()
        self.assertFalse(req.kv.mamba_needs_clear)

        self.assertIs(backup.storage_state, RetractionStorageState.RELEASED)
        self.assertEqual(cache.retraction_ssd_backups, {})
        # The hard-pinned objects outlive the resume so a re-demotion only
        # writes its new tail; the mamba key must be tracked as its own object.
        cache.cache_controller.storage_backend.batch_remove_v2.assert_not_called()
        self.assertEqual(
            req.kv.retraction_l3_keys,
            {PoolName.KV: {"h0", "h1"}, PoolName.MAMBA: {"h1"}},
        )

        cache.release_retraction_l3(req)
        (removed,) = (
            cache.cache_controller.storage_backend.batch_remove_v2.call_args.args
        )
        self.assertEqual(
            {t.name: t.keys for t in removed},
            {PoolName.KV: ["h0", "h1"], PoolName.MAMBA: ["h1"]},
        )
        self.assertIsNone(req.kv.retraction_l3_keys)

    def test_replicated_kv_rank_still_owns_its_mamba_shard(self):
        """On an MLA non-zero TP rank the KV pages are somebody else's write, so
        the rank's vote and its terminal delete must be driven by the mamba
        shard alone."""
        cache = _make_ssd_cache(backup_skip=True)
        req, cache.req_to_token_pool = _make_ssd_req(mamba_slot=3)
        backup = cache.retraction_backup_ssd(req)

        self.assertFalse(
            cache._retraction_write_succeeded(
                _write_op(completed_tokens=0, mamba_pages=0), backup
            )
        )
        self.assertTrue(
            cache._retraction_write_succeeded(
                _write_op(completed_tokens=0, mamba_pages=1), backup
            )
        )

        cache._finish_retraction_ssd_backup(7, True)
        self.assertIs(req.kv.retraction_backup, backup)
        retraction_discard(req, cache, "ssd")

        (removed,) = (
            cache.cache_controller.storage_backend.batch_remove_v2.call_args.args
        )
        self.assertEqual({t.name: t.keys for t in removed}, {PoolName.MAMBA: ["h1"]})
        self.assertIsNone(req.kv.retraction_l3_keys)
        self.assertIsNone(req.kv.retraction_backup)


if __name__ == "__main__":
    unittest.main()
