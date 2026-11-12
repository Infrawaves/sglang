"""--hicache-storage-bundle-pools: KV-derived pools stored inside the KV object.

A byte-copying Mooncake stand-in checks the data layout end to end: each test
writes real host bytes through the store, zeroes the host pools and reads them
back, using the real MLA and DSA indexer host page addressing.
"""

import ctypes
import types
import unittest
from unittest.mock import patch

import torch

from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorageConfig,
    HiCacheStorageExtraInfo,
    PoolName,
    PoolTransfer,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


class _ReplicateConfig:
    def __init__(self):
        self.group_ids = None


class _ByteStore:
    """Copies real bytes between host pointers and a shared key -> bytes map."""

    def __init__(self, objects):
        self.objects = objects

    def setup(self, *args, **kwargs):
        return 0

    def register_buffer(self, *args, **kwargs):
        return 0

    def put(self, key, value, *args):
        return 0

    def is_exist(self, key):
        return 1

    def get(self, key):
        return bytes(4 * 1024)

    @staticmethod
    def _spans(ptrs, sizes):
        if isinstance(ptrs, (list, tuple)):
            return list(zip(ptrs, sizes))
        return [(ptrs, sizes)]

    def batch_is_exist(self, keys):
        return [1 if key in self.objects else 0 for key in keys]

    def _put(self, keys, ptrs, sizes):
        for key, p, s in zip(keys, ptrs, sizes):
            self.objects[key] = b"".join(
                ctypes.string_at(ptr, size) for ptr, size in self._spans(p, s)
            )
        return [0] * len(keys)

    def batch_put_from(self, keys, ptrs, sizes, *args):
        return self._put(keys, ptrs, sizes)

    def batch_put_from_multi_buffers(self, keys, ptrs, sizes, *args):
        return self._put(keys, ptrs, sizes)

    def _get(self, keys, ptrs, sizes):
        results = []
        for key, p, s in zip(keys, ptrs, sizes):
            data = self.objects.get(key)
            spans = self._spans(p, s)
            if data is None or len(data) != sum(size for _, size in spans):
                results.append(-1)
                continue
            offset = 0
            for ptr, size in spans:
                ctypes.memmove(ptr, data[offset : offset + size], size)
                offset += size
            results.append(offset)
        return results

    def batch_get_into(self, keys, ptrs, sizes):
        return self._get(keys, ptrs, sizes)

    def batch_get_into_multi_buffers(self, keys, ptrs, sizes):
        return self._get(keys, ptrs, sizes)


class TestMooncakeBundlePools(CustomTestCase):
    PHYS_PAGE = 2  # tokens per physical host page on one rank
    LAYERS = 2
    KV_DIM = 4
    INDEX_STRIDE = 6  # bytes per indexer layer per page
    NUM_PAGES = 16

    def _mla_host(self, layout, dcp, rank):
        from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost

        host = MLATokenToKVPoolHost.__new__(MLATokenToKVPoolHost)
        host.dcp_size, host.dcp_rank = dcp, rank
        host.layout = layout
        host.page_size = self.PHYS_PAGE
        host.size = self.NUM_PAGES * self.PHYS_PAGE
        host.layer_num = self.LAYERS
        host.kv_cache_dim = self.KV_DIM
        host.dtype = torch.uint8
        host.host_buffer_read_only = False
        host._size_per_token = self.LAYERS * self.KV_DIM
        host.allocator = None
        host.kv_buffer = torch.randint(
            0, 256, (host.size * self.LAYERS * self.KV_DIM,), dtype=torch.uint8
        )
        return host

    def _indexer_host(self, dcp, rank):
        from sglang.srt.mem_cache.pool_host.dsa import DSAIndexerPoolHost

        host = DSAIndexerPoolHost.__new__(DSAIndexerPoolHost)
        host.dcp_size, host.dcp_rank = dcp, rank
        host.layout = "page_first"
        host.page_size = self.PHYS_PAGE
        host.size = self.NUM_PAGES * self.PHYS_PAGE
        host.layer_num = self.LAYERS
        host.indexer_page_stride_size = self.INDEX_STRIDE
        host.indexer_dtype = torch.uint8
        host.host_buffer_read_only = False
        host.allocator = None
        host.index_k_with_scale_buffer = torch.randint(
            0,
            256,
            (self.NUM_PAGES * self.LAYERS * self.INDEX_STRIDE,),
            dtype=torch.uint8,
        )
        return host

    class _MHADraftHost:
        """An MHA draft: every rank writes its own `_draft_k` / `_draft_v`."""

        page_size = 2
        dcp_size = 1

        def __init__(self):
            self.buffer = torch.randint(0, 256, (16 * 2 * 8,), dtype=torch.uint8)
            self.kv_buffer = self.buffer

        def get_hybrid_pool_buffer(self):
            return [self.buffer]

        def get_page_buffer_meta(self, indices):
            ptrs, sizes = [], []
            for i in range(0, len(indices), self.page_size):
                page = int(indices[i]) // self.page_size
                base = self.buffer.data_ptr() + page * 16
                ptrs += [base, base + 8]
                sizes += [8, 8]
            return ptrs, sizes

    def _store(self, kv, sidecars, *, dcp, rank, objects, bundle, group=1):
        cfg = HiCacheStorageConfig(
            tp_rank=rank,
            tp_size=max(dcp, 1),
            pp_rank=0,
            pp_size=1,
            attn_cp_rank=0,
            attn_cp_size=1,
            is_mla_model=True,
            enable_storage_metrics=False,
            is_page_first_layout=True,
            model_name="kimi",
            extra_config={
                "master_server_address": "127.0.0.1:50051",
                "check_server": False,
                "global_segment_size": 1024 * 1024,
            },
            dcp_size=dcp,
            dcp_rank=rank % dcp,
            logical_page_size=self.PHYS_PAGE * dcp if dcp > 1 else None,
            storage_page_group=group,
            storage_bundle_pools=bundle,
        )
        backend = _ByteStore(objects)

        class Store(_ByteStore):
            def __new__(cls):
                return backend

        mooncake = types.ModuleType("mooncake")
        mooncake_store = types.ModuleType("mooncake.store")
        mooncake_store.MooncakeDistributedStore = Store
        mooncake_store.ReplicateConfig = _ReplicateConfig
        with patch.dict(
            "sys.modules", {"mooncake": mooncake, "mooncake.store": mooncake_store}
        ):
            from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import (
                MooncakeStore,
            )

            store = MooncakeStore(cfg)
        store.register_mem_pool_host(kv)
        for name, pool in sidecars.items():
            store.register_mem_host_pool_v2(pool, name)
        store.on_host_pools_registered()
        return store

    def _indices(self, logical_page, first, num):
        return torch.arange(first * logical_page, (first + num) * logical_page)

    @staticmethod
    def _buffer(pool):
        for attr in ("kv_buffer", "index_k_with_scale_buffer", "buffer"):
            buf = getattr(pool, attr, None)
            if buf is not None:
                return buf
        raise AttributeError(type(pool).__name__)

    def _write_then_read(self, store, kv, sidecars, logical_page, pages, *, group):
        """Controller order: sidecar v2 before KV v1 on write, KV v1 before
        sidecar v2 on read. Read pages 2.. into host pages 9.. ."""
        names = list(sidecars)

        def transfers(first, keys):
            return [
                PoolTransfer(
                    name=name,
                    host_indices=self._indices(logical_page, first, len(keys)),
                    keys=keys,
                )
                for name in names
            ]

        info = HiCacheStorageExtraInfo(start_page=1)
        set_v2 = store.batch_set_v2(transfers(1, pages), info)
        self.assertTrue(all(all(v) for v in set_v2.values()), set_v2)
        self.assertEqual(
            store.batch_set_v1(pages, self._indices(logical_page, 1, len(pages)), info),
            [True] * len(pages),
        )
        written = {
            n: self._buffer(p).clone() for n, p in {**sidecars, "kv": kv}.items()
        }
        for pool in (kv, *sidecars.values()):
            self._buffer(pool).zero_()

        keys = pages[1:]
        extra = HiCacheStorageExtraInfo(start_page=2)
        hit = store.batch_exists_v2(keys, transfers(9, keys), extra_info=extra)
        self.assertEqual(hit.kv_hit_pages, len(keys))
        for name in names:
            self.assertEqual(hit.extra_pool_hit_pages[name], len(keys), name)
        extra.storage_plans = hit.storage_plans
        self.assertEqual(
            store.batch_get_v1(keys, self._indices(logical_page, 9, len(keys)), extra),
            [True] * len(keys),
        )
        got = store.batch_get_v2(transfers(9, keys), extra)
        for name in names:
            self.assertEqual(got[name], [True] * len(keys), name)
        return written

    def _assert_pages(self, pool, written, page_bytes, src_first, dst_first, n):
        buf = self._buffer(pool)
        for i in range(n):
            torch.testing.assert_close(
                buf[(dst_first + i) * page_bytes : (dst_first + i + 1) * page_bytes],
                written[
                    (src_first + i) * page_bytes : (src_first + i + 1) * page_bytes
                ],
            )

    def test_dcp4_bundles_kv_indexer_and_mla_draft_into_one_object(self):
        dcp = 4
        logical = self.PHYS_PAGE * dcp
        kv_page = self.LAYERS * self.KV_DIM * self.PHYS_PAGE
        idx_page = self.LAYERS * self.INDEX_STRIDE
        pages = [f"h{i}" for i in range(1, 7)]
        for layout in ("page_first", "layer_first"):
            objects = {}
            for rank in range(dcp):
                with self.subTest(layout=layout, rank=rank):
                    kv = self._mla_host(layout, dcp, rank)
                    sidecars = {
                        PoolName.INDEXER: self._indexer_host(dcp, rank),
                        PoolName.DRAFT: self._mla_host("page_first", dcp, rank),
                    }
                    store = self._store(
                        kv, sidecars, dcp=dcp, rank=rank, objects=objects, bundle=True
                    )
                    self.assertEqual(
                        store.bundled_pools, (PoolName.INDEXER, PoolName.DRAFT)
                    )
                    written = self._write_then_read(
                        store, kv, sidecars, logical, pages, group=1
                    )
                    mine = [k for k in objects if f"_dcp{rank}_" in k]
                    # One object per page per shard, holding all three pools.
                    self.assertEqual(len(mine), 6, mine)
                    self.assertTrue(all(k.endswith("_kb_indexer_draft") for k in mine))
                    self._assert_pages(
                        sidecars[PoolName.INDEXER],
                        written[PoolName.INDEXER],
                        idx_page,
                        2,
                        9,
                        5,
                    )
                    self._assert_pages(
                        sidecars[PoolName.DRAFT],
                        written[PoolName.DRAFT],
                        kv_page,
                        2,
                        9,
                        5,
                    )
                    if layout == "page_first":
                        self._assert_pages(kv, written["kv"], kv_page, 2, 9, 5)
                    else:
                        row = self.KV_DIM * self.PHYS_PAGE
                        layer = kv.size * self.KV_DIM
                        for i in range(5):
                            for l in range(self.LAYERS):
                                torch.testing.assert_close(
                                    kv.kv_buffer[
                                        l * layer
                                        + (9 + i) * row : l * layer
                                        + (10 + i) * row
                                    ],
                                    written["kv"][
                                        l * layer
                                        + (2 + i) * row : l * layer
                                        + (3 + i) * row
                                    ],
                                )

    def test_off_keeps_one_object_per_pool(self):
        objects = {}
        kv = self._mla_host("page_first", 1, 0)
        sidecars = {
            PoolName.INDEXER: self._indexer_host(1, 0),
            PoolName.DRAFT: self._mla_host("page_first", 1, 0),
        }
        store = self._store(kv, sidecars, dcp=1, rank=0, objects=objects, bundle=False)
        self.assertEqual(store.bundled_pools, ())
        self._write_then_read(
            store, kv, sidecars, self.PHYS_PAGE, [f"h{i}" for i in range(1, 7)], group=1
        )
        self.assertEqual(len(objects), 18)
        self.assertTrue(all("_kb_" not in k for k in objects), sorted(objects))

    def test_mha_draft_keeps_its_own_objects(self):
        objects = {}
        kv = self._mla_host("page_first", 1, 0)
        draft = self._MHADraftHost()
        sidecars = {PoolName.INDEXER: self._indexer_host(1, 0), PoolName.DRAFT: draft}
        store = self._store(kv, sidecars, dcp=1, rank=0, objects=objects, bundle=True)
        self.assertEqual(store.bundled_pools, (PoolName.INDEXER,))
        pages = [f"h{i}" for i in range(1, 7)]
        written = self._write_then_read(
            store, kv, sidecars, self.PHYS_PAGE, pages, group=1
        )
        kv_keys = [k for k in objects if "_kb_indexer" in k]
        draft_keys = [k for k in objects if "_draft_" in k]
        self.assertEqual(len(kv_keys), 6)
        self.assertEqual(len(draft_keys), 12)  # _draft_k and _draft_v per page
        self.assertEqual(len(objects), 18)
        self._assert_pages(draft, written[PoolName.DRAFT], 16, 2, 9, 5)

    def test_without_bundle_candidates_the_keys_do_not_change(self):
        """K3-like: MLA KV and an MHA draft, no indexer -> same keys as off."""
        keys = {}
        for bundle in (False, True):
            objects = {}
            kv = self._mla_host("page_first", 1, 0)
            store = self._store(
                kv,
                {PoolName.DRAFT: self._MHADraftHost()},
                dcp=1,
                rank=0,
                objects=objects,
                bundle=bundle,
            )
            self.assertEqual(store.bundled_pools, ())
            store.batch_set_v1(["a", "b"], torch.arange(4))
            keys[bundle] = sorted(objects)
        self.assertEqual(keys[False], keys[True])

    def _node_shared(self, host, kind):
        """Make ``host`` the per-node MLA HiCache ``kind`` ("follower" or
        "leader") view of a node-shared host pool."""
        from sglang.srt.mem_cache.mla_host_node_leader import (
            NodeSharedHostTensorAllocator,
        )

        if kind == "follower":
            host.host_buffer_read_only = True
            if hasattr(host, "kv_buffer"):
                host.kv_buffer = None
            if hasattr(host, "index_k_with_scale_buffer"):
                host.index_k_with_scale_buffer = None
        else:
            host.allocator = NodeSharedHostTensorAllocator.__new__(
                NodeSharedHostTensorAllocator
            )
        return host

    def test_per_node_mla_keeps_a_per_rank_indexer_out_of_the_kv_object(self):
        for kind in ("follower", "leader"):
            with self.subTest(kind=kind):
                kv = self._node_shared(self._mla_host("page_first", 1, 0), kind)
                store = self._store(
                    kv,
                    {PoolName.INDEXER: self._indexer_host(1, 0)},
                    dcp=1,
                    rank=0,
                    objects={},
                    bundle=True,
                )
                self.assertEqual(store.bundled_pools, ())

    def test_per_node_mla_bundles_a_node_shared_indexer(self):
        # Leader and follower resolve the same bundle, so the key suffix the
        # writing leader uses is the one every reading leader queries.
        for kind in ("follower", "leader"):
            with self.subTest(kind=kind):
                kv = self._node_shared(self._mla_host("page_first", 1, 0), kind)
                indexer = self._node_shared(self._indexer_host(1, 0), kind)
                store = self._store(
                    kv,
                    {PoolName.INDEXER: indexer},
                    dcp=1,
                    rank=0,
                    objects={},
                    bundle=True,
                )
                self.assertEqual(store.bundled_pools, (PoolName.INDEXER,))

    def test_per_node_indexer_bytes_round_trip(self):
        for bundle in (False, True):
            for group in (1, 4):
                with self.subTest(bundle=bundle, group=group):
                    objects = {}
                    kv = self._node_shared(self._mla_host("page_first", 1, 0), "leader")
                    indexer = self._node_shared(self._indexer_host(1, 0), "leader")
                    store = self._store(
                        kv,
                        {PoolName.INDEXER: indexer},
                        dcp=1,
                        rank=0,
                        objects=objects,
                        bundle=bundle,
                        group=group,
                    )
                    pages = [f"h{i}" for i in range(1, 7)]
                    written = self._write_then_read(
                        store,
                        kv,
                        {PoolName.INDEXER: indexer},
                        self.PHYS_PAGE,
                        pages,
                        group=group,
                    )
                    kv_objects = 6 if group == 1 else 2
                    self.assertEqual(len(objects), kv_objects + (0 if bundle else 6))
                    self.assertEqual(
                        sum("_kb_indexer" in key for key in objects),
                        kv_objects if bundle else 0,
                    )
                    self._assert_pages(
                        kv,
                        written["kv"],
                        self.LAYERS * self.KV_DIM * self.PHYS_PAGE,
                        2,
                        9,
                        5,
                    )
                    self._assert_pages(
                        indexer,
                        written[PoolName.INDEXER],
                        self.LAYERS * self.INDEX_STRIDE,
                        2,
                        9,
                        5,
                    )

    def test_per_node_follower_indexer_never_accesses_storage_or_buffers(self):
        for bundle in (False, True):
            for group in (1, 4):
                with self.subTest(bundle=bundle, group=group):
                    kv = self._node_shared(
                        self._mla_host("page_first", 1, 0), "follower"
                    )
                    indexer = self._node_shared(self._indexer_host(1, 0), "follower")
                    with patch.object(_ByteStore, "register_buffer") as register:
                        store = self._store(
                            kv,
                            {PoolName.INDEXER: indexer},
                            dcp=1,
                            rank=0,
                            objects={},
                            bundle=bundle,
                            group=group,
                        )
                    register.assert_not_called()
                    self.assertEqual(
                        store.bundled_pools, (PoolName.INDEXER,) if bundle else ()
                    )
                    transfer = PoolTransfer(
                        name=PoolName.INDEXER,
                        keys=["absent"],
                        host_indices=self._indices(self.PHYS_PAGE, 1, 1),
                    )
                    with (
                        patch.object(
                            _ByteStore,
                            "_put",
                            side_effect=AssertionError("follower put"),
                        ),
                        patch.object(
                            _ByteStore,
                            "_get",
                            side_effect=AssertionError("follower get"),
                        ),
                        patch.object(
                            _ByteStore,
                            "batch_is_exist",
                            side_effect=AssertionError("follower exists"),
                        ),
                        patch.object(
                            indexer,
                            "get_page_buffer_meta",
                            side_effect=AssertionError("follower buffer"),
                        ),
                    ):
                        self.assertTrue(
                            store._object_layout(["absent"], transfer).skip_io
                        )
                        self.assertEqual(
                            store.batch_set_v2([transfer])[PoolName.INDEXER], [True]
                        )
                        self.assertEqual(
                            store.batch_get_v2([transfer])[PoolName.INDEXER], [True]
                        )
                        self.assertEqual(
                            store._pools_page_exists(["absent"], [transfer], 1),
                            [[True]],
                        )

    def test_bundle_composes_with_page_groups(self):
        objects = {}
        kv = self._mla_host("page_first", 1, 0)
        sidecars = {
            PoolName.INDEXER: self._indexer_host(1, 0),
            PoolName.DRAFT: self._mla_host("page_first", 1, 0),
        }
        store = self._store(
            kv, sidecars, dcp=1, rank=0, objects=objects, bundle=True, group=4
        )
        pages = [f"h{i}" for i in range(1, 7)]
        written = self._write_then_read(
            store, kv, sidecars, self.PHYS_PAGE, pages, group=4
        )
        # Absolute pages 1..6 -> segments (1,4) and (4,7): two objects in total.
        self.assertEqual(len(objects), 2, sorted(objects))
        self.assertTrue(all("_kb_indexer_draft_g4" in k for k in objects))
        kv_page = self.LAYERS * self.KV_DIM * self.PHYS_PAGE
        self._assert_pages(kv, written["kv"], kv_page, 2, 9, 5)
        self._assert_pages(
            sidecars[PoolName.INDEXER],
            written[PoolName.INDEXER],
            self.LAYERS * self.INDEX_STRIDE,
            2,
            9,
            5,
        )
        self._assert_pages(
            sidecars[PoolName.DRAFT], written[PoolName.DRAFT], kv_page, 2, 9, 5
        )


if __name__ == "__main__":
    unittest.main(verbosity=3)
