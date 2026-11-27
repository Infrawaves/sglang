"""CPU checks for prefill CP with DCP sharding tokens by sequence position.

Under DCP the KV of position p lives on rank p % N, and prefill CP with DCP
runs one group (cp == dcp == N). Each CP rank must compute exactly the tokens
it owns, so it can write its own KV without gathering the new tokens.
"""

import random
import unittest
from collections import Counter
from contextlib import ExitStack, contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.attention.dsa import dsa_prefill_cuda_graph as graph_split
from sglang.srt.layers.cp import base as cp_base
from sglang.srt.layers.cp import interleave
from sglang.srt.layers.cp import padding as cp_padding
from sglang.srt.layers.cp import utils as cp_utils
from sglang.srt.layers.cp.bcg import interleave_bcg
from sglang.srt.layers.cp.bcg.interleave_bcg import (
    InterleaveCPBCGInput,
    fill_graph_gather_index,
    position_layout_max_rank_rows,
    restride_gather_index,
)
from sglang.srt.layers.cp.interleave import (
    InterleaveCPStrategy,
    cp_interleave_to_sequence_order,
    cp_sequence_to_interleave_order,
    owned_kv_write,
    position_owned_rows_per_request,
    writes_owned_kv_only,
)
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

N = 4
# (prefix_len, extend_len) per request: prefixes off and on rank boundaries,
# a request shorter than N, and one that starts mid-cycle.
REQUESTS = [(3, 8), (0, 5), (64, 2), (7, 13)]


@contextmanager
def _topology(*, rank, dcp_size=N):
    parallel = SimpleNamespace(
        attn_cp_size=N, attn_cp_rank=rank, attn_dcp_size=dcp_size
    )
    with ExitStack() as stack:
        for module in (interleave, cp_base):
            stack.enter_context(patch.object(module, "get_parallel", lambda: parallel))
        stack.enter_context(
            patch.object(interleave, "get_device", lambda: SimpleNamespace(device="cpu"))
        )
        # Interleave aligns each shard to the CP width.
        stack.enter_context(
            patch.object(cp_padding, "get_cp_padding_align_size", lambda: N)
        )
        yield


def _forward_batch(rank, requests=REQUESTS, dcp_size=N):
    """A CP extend batch with its metadata built and padded as prepare_cp_forward does."""
    prefix = [p for p, _ in requests]
    extend = [e for _, e in requests]
    with _topology(rank=rank, dcp_size=dcp_size):
        metadata = InterleaveCPStrategy(cp_size=N).build_metadata(
            num_tokens=sum(extend),
            seqs_len=[p + e for p, e in requests],
            extend_seqs_len=extend,
        )
        cp_padding.pad_logical_token_to_physical(metadata)
    positions = torch.cat([torch.arange(p, p + e) for p, e in zip(prefix, extend)])
    batch = SimpleNamespace(
        forward_mode=ForwardMode.EXTEND, attn_cp_metadata=metadata, positions=positions
    )
    return batch, positions


class TestPositionOwnerLayout(CustomTestCase):
    def test_each_rank_holds_exactly_the_positions_it_owns(self):
        seen = []
        for rank in range(N):
            batch, positions = _forward_batch(rank)
            rows = batch.attn_cp_metadata.local_sequence_rows
            self.assertTrue(torch.all(positions[rows] % N == rank))
            # Sequence order within the shard: KV writes and causal masks rely on it.
            self.assertTrue(torch.all(rows[1:] > rows[:-1]))
            self.assertEqual(
                rows.numel(), batch.attn_cp_metadata.per_rank_logical_token[rank]
            )
            seen.append(rows)
        self.assertEqual(
            sorted(torch.cat(seen).tolist()), list(range(positions.numel()))
        )

    def test_gather_index_restores_sequence_order_from_rank_major_shards(self):
        hidden = torch.arange(sum(e for _, e in REQUESTS), dtype=torch.float32)
        shards = []
        for rank in range(N):
            batch, _ = _forward_batch(rank)
            with _topology(rank=rank):
                shards.append(
                    InterleaveCPStrategy(cp_size=N).shard_hidden_states(hidden, batch)
                )
        # Every shard is padded to one physical length, as the boundary gathers them.
        self.assertEqual(len({s.shape[0] for s in shards}), 1)
        gathered = torch.cat(shards)
        metadata = batch.attn_cp_metadata
        self.assertEqual(len(set(metadata.gather_index.tolist())), hidden.numel())
        self.assertLess(int(metadata.gather_index.max()), gathered.shape[0])
        with _topology(rank=N - 1):
            self.assertTrue(
                torch.equal(cp_interleave_to_sequence_order(gathered, batch), hidden)
            )
            self.assertTrue(
                torch.equal(
                    cp_sequence_to_interleave_order(hidden, batch, gathered.shape[0]),
                    gathered,
                )
            )

    def test_per_request_counts_match_the_shard(self):
        prefix = [p for p, _ in REQUESTS]
        extend = [e for _, e in REQUESTS]
        for rank in range(N):
            counts = position_owned_rows_per_request(prefix, extend, rank=rank, size=N)
            expected = [
                sum(1 for pos in range(p, p + e) if pos % N == rank)
                for p, e in REQUESTS
            ]
            self.assertEqual(counts, expected)
            batch, _ = _forward_batch(rank)
            self.assertEqual(
                sum(counts), batch.attn_cp_metadata.per_rank_logical_token[rank]
            )

    def test_input_ids_global_follow_the_boundary_rows(self):
        input_ids = torch.arange(100, 100 + sum(e for _, e in REQUESTS))
        shards = []
        for rank in range(N):
            batch, _ = _forward_batch(rank)
            with _topology(rank=rank):
                shards.append(
                    InterleaveCPStrategy(cp_size=N).shard_hidden_states(input_ids, batch)
                )
        with (
            _topology(rank=0),
            patch.object(cp_utils, "is_cp_active", lambda fb: True),
            patch.object(
                cp_utils,
                "get_moe_a2a_backend",
                lambda: SimpleNamespace(is_none=lambda: True),
            ),
        ):
            global_ids = cp_utils.cp_interleave_input_ids(input_ids, batch)
        self.assertTrue(torch.equal(global_ids, torch.cat(shards)))

    def test_local_kv_writes_land_on_the_owner(self):
        # The DCP allocator places position p at a slot congruent to p mod N.
        slots = torch.cat(
            [1024 * (i + 1) + torch.arange(p, p + e) for i, (p, e) in enumerate(REQUESTS)]
        )
        written = []
        for rank in range(N):
            batch, _ = _forward_batch(rank)
            metadata = batch.attn_cp_metadata
            metadata.local_out_cache_loc = slots.index_select(
                0, metadata.local_sequence_rows
            )
            physical = metadata.per_rank_actual_token[rank]
            k = torch.zeros(physical, 1, 8)
            with _topology(rank=rank):
                self.assertTrue(writes_owned_kv_only(batch))
                loc, k_live = owned_kv_write(batch, k)
            self.assertTrue(torch.all(loc % N == rank))
            self.assertEqual(k_live.shape[0], loc.shape[0])
            written.append(loc)
        self.assertEqual(sorted(torch.cat(written).tolist()), sorted(slots.tolist()))

    def test_graph_indexer_stores_the_local_index_k_on_the_owner(self):
        """The prefill-graph DSA indexer keeps the local (ungathered) index-K under
        PCP+DCP; it must write that shard to the owner's slots, not demand a
        gathered key."""
        slots = torch.arange(sum(e for _, e in REQUESTS))
        for rank in range(N):
            batch, _ = _forward_batch(rank)
            metadata = batch.attn_cp_metadata
            metadata.local_out_cache_loc = slots.index_select(
                0, metadata.local_sequence_rows
            )
            batch.out_cache_loc = slots
            key = torch.randn(metadata.per_rank_actual_token[rank], 8)
            with (
                _topology(rank=rank),
                patch.object(cp_utils, "is_cp_active", lambda fb: True),
            ):
                store_key, store_loc = graph_split._index_k_store_rows(
                    forward_batch=batch,
                    key=key,
                    extend_num_tokens=slots.numel(),
                    in_cp=True,
                )
            live = metadata.per_rank_logical_token[rank]
            self.assertTrue(torch.equal(store_key, key[:live]))
            self.assertTrue(torch.equal(store_loc, metadata.local_out_cache_loc))


class TestGraphGatherIndex(CustomTestCase):
    """A breakable-graph replay widens every shard to the captured rows and keeps
    the captured reorder length; the per-batch gather_index must follow both."""

    @staticmethod
    def _shards(stride):
        """Every rank's positions padded to ``stride`` with -1, laid rank-major."""
        shards = []
        for rank in range(N):
            batch, positions = _forward_batch(rank)
            rows = batch.attn_cp_metadata.local_sequence_rows
            local = positions.index_select(0, rows)
            shards.append(torch.cat([local, torch.full((stride - len(local),), -1)]))
        return torch.cat(shards), batch.attn_cp_metadata, positions

    def test_restrided_index_reads_wider_shards_in_sequence_order(self):
        metadata = _forward_batch(0)[0].attn_cp_metadata
        eager_stride = metadata.per_rank_actual_token[0]
        gathered, metadata, positions = self._shards(eager_stride + N)
        index = restride_gather_index(
            metadata.gather_index, old_stride=eager_stride, new_stride=eager_stride + N
        )
        self.assertTrue(torch.equal(gathered.index_select(0, index), positions))

    def test_padded_index_reads_padding_and_writes_each_slot_once(self):
        metadata = _forward_batch(0)[0].attn_cp_metadata
        eager_stride = metadata.per_rank_actual_token[0]
        stride = eager_stride + N
        gathered, metadata, positions = self._shards(stride)
        live = restride_gather_index(
            metadata.gather_index, old_stride=eager_stride, new_stride=stride
        )
        total = live.shape[0]
        graph_index = torch.empty(total + 5, dtype=torch.int64)
        fill_graph_gather_index(
            graph_index, live, rank_rows=metadata.per_rank_logical_token, stride=stride
        )
        # Distinct slots: the captured index_copy_ never writes one twice.
        self.assertEqual(len(set(graph_index.tolist())), total + 5)
        sequence = gathered.index_select(0, graph_index)
        self.assertTrue(torch.equal(sequence[:total], positions))
        self.assertTrue(torch.all(sequence[total:] == -1))
        restored = torch.zeros_like(gathered).index_copy_(0, graph_index, sequence)
        self.assertTrue(torch.equal(restored[live], gathered[live]))

    def test_bucket_without_enough_padding_slots_is_rejected(self):
        metadata = _forward_batch(0)[0].attn_cp_metadata
        stride = metadata.per_rank_actual_token[0]
        with self.assertRaisesRegex(RuntimeError, "padding slots"):
            fill_graph_gather_index(
                torch.empty(N * stride + 1, dtype=torch.int64),
                metadata.gather_index,
                rank_rows=metadata.per_rank_logical_token,
                stride=stride,
            )


class TestGraphBucketRowsUnderDcp(CustomTestCase):
    """Page-aligned prefixes start every short extend on rank 0, so a rank holds
    more rows than the even split ceil(total / size); a bucket sized by the even
    split raises at replay instead of falling back to eager."""

    def test_short_extends_after_page_aligned_prefixes(self):
        extend, prefix = [9] * 20, [64 * (i + 1) for i in range(20)]
        parallel = SimpleNamespace(attn_cp_size=8, attn_dcp_size=8)
        with patch.object(interleave_bcg, "get_parallel", lambda: parallel):
            rows = InterleaveCPBCGInput._max_local_tokens(None, extend, prefix)
        # Rank 0 holds positions 64k and 64k + 8 of every request; the even
        # split would claim ceil(180 / 8) = 23.
        self.assertEqual(rows, 40)

    def test_rows_match_the_positions_each_rank_owns(self):
        rng = random.Random(0)
        for _ in range(500):
            size = rng.choice((2, 4, 8))
            requests = [
                (rng.randrange(0, 200), rng.randrange(1, 30))
                for _ in range(rng.randrange(1, 6))
            ]
            prefix, extend = zip(*requests)
            owned = Counter((p + i) % size for p, e in requests for i in range(e))
            expected = max(owned.values())
            self.assertEqual(
                position_layout_max_rank_rows(extend, prefix, size), expected
            )
            # Without prefixes the bound must still cover every placement.
            self.assertGreaterEqual(
                position_layout_max_rank_rows(extend, None, size), expected
            )


class TestCpGateUnderDcp(CustomTestCase):
    """A batch of at least N new tokens whose positions miss a rank would leave
    that rank with no query rows; it skips the DCP index-K gather its peers
    wait in, so such a batch must run without CP."""

    def _can_apply(self, requests, dcp_size=N):
        batch = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            extend_seq_lens_cpu=[e for _, e in requests],
            seq_lens_cpu=torch.tensor([p + e for p, e in requests]),
        )
        with _topology(rank=0, dcp_size=dcp_size):
            return InterleaveCPStrategy(cp_size=N).can_apply(
                sum(e for _, e in requests), batch
            )

    def test_batch_missing_a_rank_runs_without_cp(self):
        # Positions 0..1 twice cover ranks 0..1 only; 63..64 covers ranks 3 and 0.
        self.assertFalse(self._can_apply([(0, 2), (0, 2)]))
        self.assertFalse(self._can_apply([(63, 2), (0, 2)]))

    def test_batch_covering_every_rank_keeps_cp(self):
        self.assertTrue(self._can_apply([(0, 2), (2, 2)]))
        self.assertTrue(self._can_apply([(5, N)]))
        self.assertTrue(self._can_apply(REQUESTS))

    def test_without_dcp_only_the_token_count_gates(self):
        self.assertTrue(self._can_apply([(0, 2), (0, 2)], dcp_size=1))


class TestWithoutDcp(CustomTestCase):
    def test_concatenated_index_split_is_unchanged(self):
        total = sum(e for _, e in REQUESTS)
        for rank in range(N):
            batch, _ = _forward_batch(rank, dcp_size=1)
            metadata = batch.attn_cp_metadata
            self.assertIsNone(metadata.gather_index)
            self.assertIsNone(metadata.local_sequence_rows)
            self.assertFalse(writes_owned_kv_only(batch))
            base, extra = divmod(total, N)
            self.assertEqual(
                metadata.per_rank_logical_token,
                [base + (r < extra) for r in range(N)],
            )

    def test_graph_indexer_stores_the_gathered_live_tokens(self):
        total = sum(e for _, e in REQUESTS)
        batch, _ = _forward_batch(0, dcp_size=1)
        batch.out_cache_loc = torch.arange(total + N)
        # A gathered key with bucket padding at its tail.
        key = torch.randn(total + N, 8)
        with (
            _topology(rank=0, dcp_size=1),
            patch.object(cp_utils, "is_cp_active", lambda fb: True),
        ):
            store_key, store_loc = graph_split._index_k_store_rows(
                forward_batch=batch,
                key=key,
                extend_num_tokens=total,
                in_cp=True,
            )
        self.assertTrue(torch.equal(store_key, key[:total]))
        self.assertTrue(torch.equal(store_loc, batch.out_cache_loc[:total]))


if __name__ == "__main__":
    unittest.main()
