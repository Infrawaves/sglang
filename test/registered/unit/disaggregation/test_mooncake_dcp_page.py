"""CPU transport contracts for direct Mooncake DCP page sends."""

import concurrent.futures
import queue
import struct
import threading
import unittest
from collections import defaultdict
from itertools import product
from multiprocessing import shared_memory
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
from disagg_test_utils import CopyTransport

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.common.transfer_lifetime import TransferLifetimeTracker
from sglang.srt.disaggregation.common.utils import pack_int_lists
from sglang.srt.disaggregation.mooncake import draft_peer_gather
from sglang.srt.disaggregation.mooncake.conn import (
    MooncakeKVManager,
    TransferInfo,
)
from sglang.srt.disaggregation.mooncake.draft_peer_gather import (
    SEQ_SHM_NBYTES,
    DraftPeerGather,
    DraftPeerRecord,
)
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


class TestMooncakeDcpPage(CustomTestCase):
    page_size = 4
    widths = (3, 5)

    draft_width = 7
    draft_layer_id = 90

    def _case(self, *, custom_pool=False, draft=False):
        sources = [
            ((np.arange(64 * width) + layer * 41) % 251)
            .astype(np.uint8)
            .reshape(64, width)
            for layer, width in enumerate(self.widths)
        ]
        destinations = [np.full_like(source, 0xEE) for source in sources]
        layer_ids = [11, 17]
        if draft:
            # The decode draft pool pages at page_size * dcp_size and holds
            # every token, so it is larger than the per-rank target pool.
            sources.append(
                ((np.arange(64 * self.draft_width) * 7 + 3) % 251)
                .astype(np.uint8)
                .reshape(64, self.draft_width)
            )
            destinations.append(
                np.full((15 * self.page_size * 3, self.draft_width), 0xEE, np.uint8)
            )
            layer_ids.append(self.draft_layer_id)
        manager = object.__new__(MooncakeKVManager)
        manager.kv_args = SimpleNamespace(
            page_size=self.page_size,
            kv_data_ptrs=[buffer.ctypes.data for buffer in sources],
            kv_item_lens=[buffer[0].nbytes * self.page_size for buffer in sources],
            kv_layer_ids=layer_ids,
            num_draft_entries=int(draft),
        )
        manager.enable_custom_mem_pool = custom_pool
        manager.custom_mem_pool_type = "NVLINK" if custom_pool else None
        manager.max_transfer_batch_indices = 0
        manager.enable_deferred_decode_kv_release = False
        manager.is_mla_backend = True
        manager.is_hybrid_mla_backend = False
        manager.pp_size = 1
        manager.attn_tp_size = 1
        transport = CopyTransport(sources, destinations)
        manager._transfer_data = transport
        return manager, sources, destinations, transport

    def _send(self, manager, destinations, *, executor=None, **kwargs):
        layer_ids = manager.kv_args.kv_layer_ids
        widths = list(self.widths) + [self.draft_width] * (len(layer_ids) - 2)
        return manager.send_kvcache_dcp(
            "session",
            kwargs.pop("source_pages", np.array([7], dtype=np.int32)),
            # Reversed destination registration exercises the existing layer map.
            [buffer.ctypes.data for buffer in reversed(destinations)],
            kwargs.pop("destination_pages", np.array([10], dtype=np.int32)),
            dcp_token_item_lens=widths,
            dst_dcp_kv_layout="page",
            dst_dcp_size=kwargs.pop("dcp_size", 3),
            dst_dcp_rank=kwargs.pop("rank", 0),
            src_page_offset=kwargs.pop("offset", 0),
            decode_prefix_len=kwargs.pop("prefix", 12),
            num_kv_tokens=kwargs.pop("length", 4),
            executor=executor,
            dst_layer_ids=list(reversed(layer_ids)),
            pack_buffer=kwargs.pop("pack_buffer", None),
        )

    def test_fragmented_chunks_copy_owned_pages_including_unused_tail(self):
        source_pages = np.array([7, 1, 9, 3, 12, 4, 8], dtype=np.int32)
        destination_pages = np.array([10, 2, 14], dtype=np.int32)
        for rank, custom_pool in product(range(3), (False, True)):
            with self.subTest(rank=rank, custom_pool=custom_pool):
                manager, sources, destinations, transport = self._case(
                    custom_pool=custom_pool
                )
                with (
                    concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor,
                    patch(
                        "sglang.srt.disaggregation.common.dcp_pack.try_pack_dcp_src",
                        side_effect=AssertionError("page send must not pack"),
                    ),
                ):
                    for offset, count, length in ((0, 2, 8), (2, 3, 12), (5, 2, 5)):
                        self.assertEqual(
                            self._send(
                                manager,
                                destinations,
                                executor=executor,
                                source_pages=source_pages[offset : offset + count],
                                destination_pages=destination_pages,
                                rank=rank,
                                offset=offset,
                                length=length,
                            ),
                            0,
                        )
                        if rank == 2 and offset == 0:
                            self.assertEqual(transport.bytes_sent, 0)

                owned = [i for i in range(28) if ((12 + i) // 4) % 3 == rank]
                for source, destination in zip(sources, destinations):
                    expected = np.full_like(destination, 0xEE)
                    for local, logical in enumerate(owned):
                        source_row = source_pages[logical // 4] * 4 + logical % 4
                        destination_row = destination_pages[local // 4] * 4 + local % 4
                        expected[destination_row] = source[source_row]
                    np.testing.assert_array_equal(destination, expected)
                self.assertEqual(transport.bytes_sent, len(owned) * sum(self.widths))

    def test_draft_entries_receive_every_token_at_the_virtual_page_size(self):
        """DSPARK's replicated draft pool needs every token on every rank. The
        page plan would send it only this rank's pages at the target geometry."""
        source_pages = np.array([7, 1, 9, 3, 12, 4, 8], dtype=np.int32)
        destination_pages = np.array([10, 2, 14], dtype=np.int32)
        chunks = ((0, 2, 8), (2, 3, 12), (5, 2, 5))
        total_tokens = sum(length for _, _, length in chunks)
        for rank, custom_pool in product(range(3), (False, True)):
            with self.subTest(rank=rank, custom_pool=custom_pool):
                manager, sources, destinations, transport = self._case(
                    custom_pool=custom_pool, draft=True
                )
                with (
                    concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor,
                    patch(
                        "sglang.srt.disaggregation.common.dcp_pack.try_pack_dcp_src",
                        side_effect=AssertionError("page send must not pack"),
                    ),
                ):
                    for offset, count, length in chunks:
                        self.assertEqual(
                            self._send(
                                manager,
                                destinations,
                                executor=executor,
                                source_pages=source_pages[offset : offset + count],
                                destination_pages=destination_pages,
                                rank=rank,
                                offset=offset,
                                length=length,
                                # A token-layout peer may have created one.
                                pack_buffer=object(),
                            ),
                            0,
                        )

                owned = [i for i in range(28) if ((12 + i) // 4) % 3 == rank]
                for source, destination in zip(sources[:2], destinations[:2]):
                    expected = np.full_like(destination, 0xEE)
                    for local, logical in enumerate(owned):
                        source_row = source_pages[logical // 4] * 4 + logical % 4
                        destination_row = destination_pages[local // 4] * 4 + local % 4
                        expected[destination_row] = source[source_row]
                    np.testing.assert_array_equal(destination, expected)

                virtual_page = 4 * 3
                expected = np.full_like(destinations[2], 0xEE)
                for token in range(total_tokens):
                    source_row = source_pages[token // 4] * 4 + token % 4
                    destination_row = (
                        destination_pages[token // virtual_page] * virtual_page
                        + token % virtual_page
                    )
                    expected[destination_row] = sources[2][source_row]
                np.testing.assert_array_equal(destinations[2], expected)
                self.assertEqual(
                    transport.bytes_sent,
                    len(owned) * sum(self.widths) + total_tokens * self.draft_width,
                )

    def test_page_send_propagates_transport_failure(self):
        """A failed page transfer must not be reported as a successful handoff."""
        for custom_pool in (False, True):
            with self.subTest(custom_pool=custom_pool):
                manager, _, destinations, _ = self._case(custom_pool=custom_pool)
                with (
                    concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor,
                    patch.object(manager, "_transfer_data", return_value=-9),
                ):
                    self.assertEqual(
                        self._send(manager, destinations, executor=executor), -9
                    )

    def test_generic_send_merges_pages_without_crossing_physical_gaps(self):
        manager, sources, destinations, transport = self._case()
        with patch.object(manager, "_transfer_data", wraps=transport) as transfer:
            self.assertEqual(
                self._send(
                    manager,
                    destinations,
                    source_pages=np.array([10, 13, 11, 14, 3, 15, 4, 12, 5], np.int32),
                    destination_pages=np.array([1, 2, 3, 8, 9], np.int32),
                    dcp_size=2,
                    prefix=0,
                    length=34,
                ),
                0,
            )
        expected = [
            (
                source.ctypes.data + src_page * 4 * width,
                destination.ctypes.data + dst_page * 4 * width,
                page_count * 4 * width,
            )
            for source, destination, width in zip(sources, destinations, self.widths)
            for src_page, dst_page, page_count in ((10, 1, 2), (3, 3, 1), (4, 8, 2))
        ]
        transfer.assert_called_once_with("session", expected)
        self.assertEqual(transport.bytes_sent, 20 * sum(self.widths))


def _host_pack_buffer(size_bytes, *args, **kwargs):
    """Host stand-in for StagingBuffer: same pointer/size contract."""
    buffer = np.zeros(size_bytes, dtype=np.uint8)
    return SimpleNamespace(
        buffer=buffer,
        get_ptr=lambda: buffer.ctypes.data,
        get_size=lambda: size_bytes,
    )


def _numpy_gather(sources, launches):
    """Byte-exact host model of copy_rows_into_strided_pack."""

    def gather(segments, row_indices, pack_buffer):
        launches.append(int(row_indices.size))
        pack = pack_buffer.buffer
        for src_ptr, row_nbytes, pack_offset, src_stride, pack_stride in segments:
            for row, src_row in enumerate(row_indices):
                start = pack_offset + row * pack_stride
                pack[start : start + row_nbytes] = CopyTransport._region(
                    sources, src_ptr + int(src_row) * src_stride, row_nbytes
                )

    return gather


class TestMooncakeDcpPageDraftPeerGather(CustomTestCase):
    """Hybrid-MLA target + DSPARK draft, prefill attn-TP = k x decode attn-TP."""

    page_size = 4
    dcp_size = 2
    target_width = 6
    draft_src_width = 3
    num_draft = 2
    layer_ids = [0, 90, 91]
    source_pages = np.array([9, 2, 14, 5, 11, 0], dtype=np.int32)
    # Decode page ids; the draft pool reads them at the virtual page size.
    destination_pages = np.array([3, 7, 1], dtype=np.int32)
    prefix = 8
    # (page offset, page count, tokens); the second chunk ends on a partial page.
    chunks = ((0, 3, 12), (3, 3, 10))
    num_tokens = 22
    room = 4242

    def _build(self, prefill_tp, decode_tp):
        k = prefill_tp // decode_tp
        rng = np.random.default_rng(prefill_tp * 10 + decode_tp)
        draft_global_width = self.draft_src_width * prefill_tp
        target_src = rng.integers(0, 251, (64, self.target_width), dtype=np.uint8)
        draft_src = [
            rng.integers(0, 251, (64, draft_global_width), dtype=np.uint8)
            for _ in range(self.num_draft)
        ]
        prefill_buffers = [
            [target_src.copy()]
            + [
                draft[
                    :, p * self.draft_src_width : (p + 1) * self.draft_src_width
                ].copy()
                for draft in draft_src
            ]
            for p in range(prefill_tp)
        ]
        dst_width = self.draft_src_width * k
        decode_buffers = [
            [np.full((32, self.target_width), 0xEE, np.uint8)]
            + [np.full((64, dst_width), 0xEE, np.uint8) for _ in range(self.num_draft)]
            for _ in range(decode_tp)
        ]
        seq_shms = [
            shared_memory.SharedMemory(create=True, size=SEQ_SHM_NBYTES)
            for _ in range(prefill_tp)
        ]
        records = [
            DraftPeerRecord(
                hostname="node0",
                seq_shm_name=shm.name,
                ipc_handles=[],
                entry_locations=[],
            )
            for shm in seq_shms
        ]
        ptrs_by_shm = {
            shm.name: [buffer.ctypes.data for buffer in buffers[1:]]
            for shm, buffers in zip(seq_shms, prefill_buffers)
        }
        ipc = patch.object(
            draft_peer_gather,
            "_open_draft_ipc",
            side_effect=lambda record, gpu_id: ptrs_by_shm[record.seq_shm_name],
        )
        ipc.start()
        self.addCleanup(ipc.stop)
        sources = [b for buffers in prefill_buffers for b in buffers]
        destinations = [b for buffers in decode_buffers for b in buffers]
        managers = []
        for p in range(prefill_tp):
            gather = DraftPeerGather(
                local_rank=p,
                local_draft_ptrs=ptrs_by_shm[seq_shms[p].name],
                records=records,
                seq_shm=seq_shms[p],
                gpu_id=0,
            )
            self.addCleanup(gather.close)
            managers.append(
                self._manager(
                    p, prefill_tp, prefill_buffers[p], gather, sources, destinations
                )
            )
        return SimpleNamespace(
            k=k,
            managers=managers,
            target_src=target_src,
            draft_src=draft_src,
            decode_buffers=decode_buffers,
            sources=sources,
        )

    def _manager(self, rank, prefill_tp, buffers, gather, sources, destinations):
        manager = object.__new__(MooncakeKVManager)
        manager.disaggregation_mode = DisaggregationMode.PREFILL
        manager.kv_args = SimpleNamespace(
            engine_rank=rank,
            gpu_id=0,
            page_size=self.page_size,
            kv_data_ptrs=[buffer.ctypes.data for buffer in buffers],
            kv_item_lens=[buffer[0].nbytes * self.page_size for buffer in buffers],
            kv_layer_ids=list(self.layer_ids),
            num_draft_entries=self.num_draft,
            state_types=[],
        )
        manager.server_args = SimpleNamespace(enable_pd_role_switch=False)
        manager.is_mla_backend = False
        manager.is_hybrid_mla_backend = True
        manager.attn_tp_size = prefill_tp
        manager.attn_tp_rank = rank
        manager.attn_cp_size = 1
        manager.attn_cp_rank = 0
        manager.pp_size = 1
        manager.pp_rank = 0
        manager.dcp_size = 1
        manager.dcp_rank = 0
        manager.enable_staging = False
        manager.enable_trace = False
        manager.enable_custom_mem_pool = False
        manager.custom_mem_pool_type = None
        manager.max_transfer_batch_indices = 0
        manager.enable_deferred_decode_kv_release = True
        manager._stopped = False
        manager._worker_threads = []
        manager._transfer_lifetime = TransferLifetimeTracker()
        manager._abort_drain_lock = threading.RLock()
        manager._drain_ack_targets = {}
        manager._dcp_pack_buffers = None
        manager._dcp_pack_max_tokens = None
        manager._draft_peer_gather = gather
        manager._draft_pack_buffers = None
        manager.decode_kv_args_table = {}
        manager.transfer_infos = {}
        manager.request_status = {}
        manager.req_to_decode_prefix_len = {}
        manager.session_lock = threading.Lock()
        manager.session_failures = defaultdict(int)
        manager.failed_sessions = set()
        manager.failure_lock = threading.Lock()
        manager.failure_records = {}
        manager.transfer_queues = [queue.Queue()]
        manager.engine = SimpleNamespace(batch_register=MagicMock())
        manager.server_socket = None
        manager.bootstrap_port = 30000
        manager._send_multipart_locked = MagicMock()
        manager._transfer_data = CopyTransport(sources, destinations)
        return manager

    def _session(self, decode_rank):
        return f"127.0.0.1:{18000 + decode_rank}"

    def _register(self, manager, decode_rank, decode_tp, decode_buffers):
        """Feed one decode registration through the prefill bootstrap thread."""
        k = manager.attn_tp_size // decode_tp
        dst_item_lens = [self.page_size * self.target_width] + [
            self.page_size * self.dcp_size * self.draft_src_width * k
        ] * self.num_draft
        message = [
            b"None",
            b"127.0.0.1",
            str(19000 + decode_rank).encode(),
            self._session(decode_rank).encode(),
            struct.pack(
                f"{len(decode_buffers)}Q", *(b.ctypes.data for b in decode_buffers)
            ),
            b"",
            pack_int_lists([], "Q"),
            str(decode_rank).encode(),
            str(decode_tp).encode(),
            str(dst_item_lens[0]).encode(),
            pack_int_lists([], "I"),
            pack_int_lists([], "I"),
            struct.pack(f"{len(self.layer_ids)}I", *self.layer_ids),
            b"",
            b"",
            b"",
            str(self.dcp_size).encode(),
            str(decode_rank % self.dcp_size).encode(),
            b"",
            struct.pack(f"{len(dst_item_lens)}Q", *dst_item_lens),
            b"page",
        ]
        messages = iter([message])

        def recv():
            message = next(messages, None)
            if message is None:
                manager._stopped = True
            return message

        with (
            patch.object(manager, "_make_worker_recv", return_value=recv),
            patch.object(manager, "_register_staging_memory"),
            patch(
                "sglang.srt.disaggregation.common.staging_handler._get_custom_mem_pool",
                return_value=(None, None),
            ),
            patch(
                "sglang.srt.disaggregation.common.staging_buffer.StagingBuffer",
                side_effect=self._pack_buffer,
            ),
        ):
            manager.start_prefill_thread()
            for thread in manager._worker_threads:
                thread.join(timeout=5)
        self.assertIn(self._session(decode_rank), manager.decode_kv_args_table)

    def _pack_buffer(self, size_bytes, *args, **kwargs):
        buffer = _host_pack_buffer(size_bytes)
        self._cluster.sources.append(buffer.buffer)
        return buffer

    def _enqueue(self, manager, decode_rank, early_send_event=None):
        """Enqueue the two chunks; ``early_send_event`` marks the first as an
        overlap early send waiting on the prior forward."""
        manager.transfer_infos[self.room] = {
            self._session(decode_rank): TransferInfo(
                room=self.room,
                endpoint="127.0.0.1",
                dst_port=19000 + decode_rank,
                mooncake_session_id=self._session(decode_rank),
                dst_kv_indices=self.destination_pages,
                dst_aux_index=0,
                dst_state_indices=[],
                required_dst_info_num=1,
                is_dummy=False,
                decode_prefix_len=self.prefix,
            )
        }
        manager.request_status[self.room] = KVPoll.WaitingForInput
        manager._transfer_lifetime.open(self.room)
        for i, (offset, count, tokens) in enumerate(self.chunks):
            manager.add_transfer_request(
                self.room,
                self.source_pages[offset : offset + count],
                slice(offset, offset + count),
                False,
                num_kv_tokens=tokens,
                wait_event=early_send_event if i == 0 else None,
            )

    def _drain(self, manager):
        chunks = []
        while not manager.transfer_queues[0].empty():
            chunks.append(manager.transfer_queues[0].get())
        work = SimpleNamespace(get=MagicMock(side_effect=chunks + [SystemExit]))
        with self.assertRaises(SystemExit):
            manager.transfer_worker(work, executor=None)

    def _register_all(self, cluster, decode_tp):
        with (
            envs.SGLANG_DISAGG_PREFILL_EARLY_SEND_CACHED_PREFIX.override(True),
            get_context().override_server_args(chunked_prefill_size=5),
        ):
            for p, manager in enumerate(cluster.managers):
                self._register(
                    manager,
                    p // cluster.k,
                    decode_tp,
                    cluster.decode_buffers[p // cluster.k],
                )

    def _run(self, prefill_tp, decode_tp, launches):
        self._cluster = cluster = self._build(prefill_tp, decode_tp)
        self._register_all(cluster, decode_tp)
        for p, manager in enumerate(cluster.managers):
            self._enqueue(manager, p // cluster.k)
        with patch.object(
            draft_peer_gather,
            "_launch_gather",
            side_effect=_numpy_gather(cluster.sources, launches),
        ):
            for manager in cluster.managers:
                self._drain(manager)
        return cluster

    def _assert_decode_rows(self, cluster):
        dst_width = self.draft_src_width * cluster.k
        for d, buffers in enumerate(cluster.decode_buffers):
            np.testing.assert_array_equal(
                buffers[0], self._expected_target(d % self.dcp_size)
            )
            for draft, destination in zip(cluster.draft_src, buffers[1:]):
                np.testing.assert_array_equal(
                    destination, self._expected_draft(draft, d, dst_width)
                )

    def _expected_target(self, dcp_rank):
        expected = np.full((32, self.target_width), 0xEE, np.uint8)
        for page in range(self.source_pages.size):
            logical_page = self.prefix // self.page_size + page
            if logical_page % self.dcp_size != dcp_rank:
                continue
            local_page = page // self.dcp_size
            dst_page = self.destination_pages[local_page]
            src_page = self.source_pages[page]
            expected[dst_page * 4 : dst_page * 4 + 4] = self._cluster.target_src[
                src_page * 4 : src_page * 4 + 4
            ]
        return expected

    def _expected_draft(self, draft, decode_rank, dst_width):
        virtual_page = self.page_size * self.dcp_size
        expected = np.full((64, dst_width), 0xEE, np.uint8)
        for token in range(self.num_tokens):
            src_row = self.source_pages[token // 4] * 4 + token % 4
            dst_row = (
                self.destination_pages[token // virtual_page] * virtual_page
                + token % virtual_page
            )
            expected[dst_row] = draft[
                src_row, decode_rank * dst_width : (decode_rank + 1) * dst_width
            ]
        return expected

    def test_sender_delivers_full_decode_draft_rows(self):
        """Every decode rank's draft heads come from k prefill ranks, of which
        only one sends hybrid-MLA KV; any missing shard leaves stale draft KV."""
        for prefill_tp, decode_tp in ((4, 2), (8, 2)):
            with self.subTest(prefill_tp=prefill_tp, decode_tp=decode_tp):
                launches = []
                cluster = self._run(prefill_tp, decode_tp, launches)
                k = cluster.k
                dst_width = self.draft_src_width * k
                self._assert_decode_rows(cluster)
                target_pages = [3, 3]  # pages owned per dcp rank in this layout
                for p, manager in enumerate(cluster.managers):
                    expected_bytes = 0
                    if p % k == 0:
                        expected_bytes = (
                            target_pages[(p // k) % self.dcp_size]
                            * self.page_size
                            * self.target_width
                            + self.num_tokens * dst_width * self.num_draft
                        )
                    self.assertEqual(manager._transfer_data.bytes_sent, expected_bytes)
                # The 8-row pack window splits each 12/10-token chunk in two.
                self.assertEqual(launches, [8, 4, 8, 2] * decode_tp)

    def test_sender_fails_room_when_peer_never_enqueues(self):
        """A sender must not read a peer's draft KV for a chunk the peer has
        not finished. Target KV needs only local KV and still goes out; on
        timeout the room fails, the session stays healthy and leases drain."""
        self._cluster = cluster = self._build(4, 2)
        self._register_all(cluster, 2)
        sender = cluster.managers[0]
        self._enqueue(sender, 0)
        launches = []
        with (
            envs.SGLANG_DISAGGREGATION_WAITING_TIMEOUT.override(0),
            patch.object(
                draft_peer_gather,
                "_launch_gather",
                side_effect=_numpy_gather(cluster.sources, launches),
            ),
        ):
            self._drain(sender)
        self.assertEqual(launches, [])
        self.assertGreater(sender._transfer_data.bytes_sent, 0)
        for destination in cluster.decode_buffers[0][1:]:
            self.assertTrue((destination == 0xEE).all())
        self.assertEqual(sender.request_status[self.room], KVPoll.Failed)
        self.assertIn("did not finish writing", sender.failure_records[self.room])
        self.assertEqual(sender.failed_sessions, set())
        self.assertTrue(sender._transfer_lifetime.is_drained(self.room))

    def test_target_send_failure_skips_peer_wait_and_keeps_lease(self):
        self._cluster = cluster = self._build(4, 2)
        self._register_all(cluster, 2)
        sender = cluster.managers[0]
        self._enqueue(sender, 0)
        with (
            patch.object(sender, "_transfer_data", return_value=1),
            patch.object(sender, "_wait_for_draft_peers") as wait,
        ):
            self._drain(sender)
        wait.assert_not_called()
        self.assertEqual(sender.request_status[self.room], KVPoll.Failed)
        self.assertEqual(sender.failed_sessions, {self._session(0)})
        self.assertFalse(sender._transfer_lifetime.is_drained(self.room))

    def test_overlap_early_send_waits_for_peer_forward_sync(self):
        """Under overlap an early send is enqueued before the prior forward
        finishes writing its pages; the sender must not gather a peer's draft
        KV for it until that peer's scheduler has synced the forward."""
        self._cluster = cluster = self._build(4, 2)
        self._register_all(cluster, 2)
        for p, manager in enumerate(cluster.managers):
            self._enqueue(manager, p // cluster.k, early_send_event=MagicMock())
        sender = cluster.managers[0]
        launches = []
        polls = []

        def peer_scheduler_syncs_forward(_):
            polls.append((list(launches), sender._transfer_data.bytes_sent))
            for manager in cluster.managers:
                manager.mark_prefill_forward_synced()

        with (
            patch.object(
                draft_peer_gather,
                "_launch_gather",
                side_effect=_numpy_gather(cluster.sources, launches),
            ),
            patch.object(
                draft_peer_gather.time,
                "sleep",
                side_effect=peer_scheduler_syncs_forward,
            ),
        ):
            for manager in cluster.managers:
                self._drain(manager)
        # One poll, after the first chunk's local target KV went out and before
        # any peer draft KV was read.
        self.assertEqual(len(polls), 1)
        launches_at_poll, bytes_at_poll = polls[0]
        self.assertEqual(launches_at_poll, [])
        self.assertGreater(bytes_at_poll, 0)
        self._assert_decode_rows(cluster)

    def test_ready_watermark_stays_contiguous(self):
        """Without a wait event a chunk is ready at enqueue, unless an earlier
        chunk is still waiting on an unsynced forward."""
        cluster = self._build(2, 1)
        sender, peer = (m._draft_peer_gather for m in cluster.managers)
        sender.open_peers([0, 1])
        ready = dict(peer_ranks=[0, 1], timeout_s=0, is_cancelled=lambda: False)

        def enqueue(has_wait_event):
            return peer.enqueue_chunk(has_wait_event=has_wait_event)

        self.assertTrue(sender.wait_for_peers(seq=enqueue(False), **ready))
        pending = enqueue(True)
        after_pending = enqueue(False)
        self.assertFalse(sender.wait_for_peers(seq=pending, **ready))
        self.assertFalse(sender.wait_for_peers(seq=after_pending, **ready))
        peer.mark_forward_synced()
        self.assertTrue(sender.wait_for_peers(seq=after_pending, **ready))
        self.assertTrue(sender.wait_for_peers(seq=enqueue(False), **ready))
        self.assertFalse(
            sender.wait_for_peers(
                peer_ranks=[0, 1], seq=99, timeout_s=60, is_cancelled=lambda: True
            )
        )

    def test_wait_stats_log_reports_first_check_and_sleeps(self):
        with envs.SGLANG_DISAGG_DRAFT_PEER_WAIT_STATS_INTERVAL.override(1e-9):
            cluster = self._build(2, 1)
        sender, peer = (m._draft_peer_gather for m in cluster.managers)
        sender.open_peers([0, 1])
        ready = dict(peer_ranks=[0, 1], timeout_s=60, is_cancelled=lambda: False)
        with self.assertLogs(draft_peer_gather.logger, "INFO") as logs:
            self.assertTrue(
                sender.wait_for_peers(
                    seq=peer.enqueue_chunk(has_wait_event=False), **ready
                )
            )
            pending = peer.enqueue_chunk(has_wait_event=True)
            with patch.object(
                draft_peer_gather.time,
                "sleep",
                side_effect=lambda _: peer.mark_forward_synced(),
            ):
                self.assertTrue(sender.wait_for_peers(seq=pending, **ready))
        first, second = logs.output
        self.assertIn("chunks=1 ready_on_first_check=100.0%", first)
        self.assertIn("sleeps=0", first)
        self.assertIn("chunks=1 ready_on_first_check=0.0%", second)
        self.assertIn("sleeps=1", second)

    def _registration_info(self, dst_attn_tp_size, dst_token_width):
        # Only the draft tail of dst_kv_item_lens is read; one page holds
        # page_size x dcp_size tokens of the replicated draft pool.
        draft_item_len = dst_token_width * self.page_size * self.dcp_size
        return SimpleNamespace(
            requires_dcp_relayout=True,
            dcp_kv_layout="page",
            dst_attn_tp_size=dst_attn_tp_size,
            dst_tp_rank=0,
            dst_dcp_size=self.dcp_size,
            dst_kv_item_lens=[0] + [draft_item_len] * self.num_draft,
        )

    def test_registration_rejects_staging(self):
        cluster = self._build(4, 2)
        cluster.managers[0].enable_staging = True
        info = self._registration_info(2, self.draft_src_width * 2)
        with self.assertRaisesRegex(RuntimeError, "staging"):
            cluster.managers[0]._resolve_draft_peer_ranks(info)

    def test_registration_rejects_shards_that_do_not_fill_a_decode_row(self):
        """Replicated GQA heads: k prefill shards cover less than a decode row."""
        cluster = self._build(4, 2)
        info = self._registration_info(2, self.draft_src_width * 2 + 1)
        with self.assertRaisesRegex(RuntimeError, "fill one decode draft row"):
            cluster.managers[0]._resolve_draft_peer_ranks(info)

    def test_registration_rejects_cross_node_peer(self):
        cluster = self._build(4, 2)
        gather = cluster.managers[0]._draft_peer_gather
        gather.records[1] = DraftPeerRecord(
            hostname="node1",
            seq_shm_name=gather.records[1].seq_shm_name,
            ipc_handles=[],
            entry_locations=[],
        )
        info = self._registration_info(2, self.draft_src_width * 2)
        with self.assertRaisesRegex(RuntimeError, "node-local"):
            cluster.managers[0]._resolve_draft_peer_ranks(info)


if __name__ == "__main__":
    unittest.main()
