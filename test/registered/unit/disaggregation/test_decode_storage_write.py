import ctypes
import unittest

import numpy as np

from sglang.srt.disaggregation.decode_storage_write import (
    DecodeStorageWriteState,
    MambaStateMeta,
    StorageWriteTarget,
    build_write_plan,
    local_skip_count,
    local_sub_page_keys,
    mamba_put_objects,
    prompt_stored_end,
    resolve_target,
    skip_sub_pages,
    sub_page_hashes,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

LAYERS = 3
CONV_ROWS = 3  # Kimi conv state is [K - 1, dim]
CONV_GROUPS = (8, 8, 16)  # q | k | v, each head-sharded on its own
TEMPORAL_HEADS = 8
TEMPORAL_TAIL = (2, 3)


def _full_layers(rng):
    temporal = rng.integers(0, 2**16, (LAYERS, TEMPORAL_HEADS, *TEMPORAL_TAIL))
    conv = rng.integers(0, 2**16, (LAYERS, CONV_ROWS, sum(CONV_GROUPS)))
    return temporal.astype(np.uint16), conv.astype(np.uint16)


def _shard(temporal, conv, tp, rank):
    """One attention-TP rank's Mamba state, as either side's model holds it."""
    heads = TEMPORAL_HEADS // tp
    t = temporal[:, rank * heads : (rank + 1) * heads]
    parts, start = [], 0
    for group in CONV_GROUPS:
        width = group // tp
        parts.append(conv[:, :, start + rank * width : start + (rank + 1) * width])
        start += group
    return np.ascontiguousarray(t), np.ascontiguousarray(np.concatenate(parts, -1))


def _metas(decode_tp):
    return [
        MambaStateMeta(
            key_suffix="temporal",
            item_bytes=(TEMPORAL_HEADS // decode_tp) * int(np.prod(TEMPORAL_TAIL)) * 2,
            slice_dim=TEMPORAL_HEADS // decode_tp,
            outer_count=1,
            conv_shard_groups=None,
        ),
        MambaStateMeta(
            key_suffix="conv_0",
            item_bytes=CONV_ROWS * (sum(CONV_GROUPS) // decode_tp) * 2,
            slice_dim=sum(CONV_GROUPS) // decode_tp,
            outer_count=CONV_ROWS,
            conv_shard_groups=CONV_GROUPS,
        ),
    ]


def _read(ptrs, sizes):
    return b"".join(ctypes.string_at(p, n) for p, n in zip(ptrs, sizes))


class TestMambaReslice(CustomTestCase):
    def test_decode_objects_match_prefill_shards_byte_for_byte(self):
        """A decode rank's Mamba slot, resliced, must equal what each prefill rank
        it covers would have written from its own shard: every layer back to
        back, conv q/k/v sub-blocks sliced independently."""
        rng = np.random.default_rng(0)
        temporal, conv = _full_layers(rng)
        for decode_tp, prefill_tp in ((4, 4), (4, 8), (2, 8)):
            ratio = prefill_tp // decode_tp
            for decode_rank in range(decode_tp):
                targets = list(range(decode_rank * ratio, (decode_rank + 1) * ratio))
                plan = build_write_plan(
                    StorageWriteTarget(
                        prefill_attn_tp_size=prefill_tp,
                        prefill_page_size=64,
                        target_tp_ranks=tuple(targets),
                    ),
                    tree_page_size=64,
                    dcp_size=1,
                    dcp_rank=0,
                    attn_tp_size=decode_tp,
                    attn_tp_rank=decode_rank,
                    mamba_tensors=_metas(decode_tp),
                )
                host = _shard(temporal, conv, decode_tp, decode_rank)
                keys, ptrs, sizes = mamba_put_objects(
                    page_key="h",
                    plan=plan,
                    slot_bases=[arr.ctypes.data for arr in host],
                    num_layers=LAYERS,
                )
                expected = {}
                for rank in targets:
                    t, c = _shard(temporal, conv, prefill_tp, rank)
                    expected[f"h_{rank}_temporal"] = t.tobytes()
                    expected[f"h_{rank}_conv_0"] = c.tobytes()
                got = {k: _read(p, s) for k, p, s in zip(keys, ptrs, sizes)}
                self.assertEqual(got, expected, (decode_tp, prefill_tp, decode_rank))


class TestStorageKeys(CustomTestCase):
    def test_hash_chain_does_not_depend_on_node_split(self):
        """Prefill hashes one continuous 64-token chain from the namespace seed;
        decode must reach the same keys however its tree split the path."""
        tokens = list(range(1, 64 * 12 + 1))
        seed = "ab" * 32
        whole = sub_page_hashes(
            prefix_chunks=(), token_ids=tokens, seed=seed, sub_page_size=64
        )
        split = sub_page_hashes(
            prefix_chunks=(tokens[:256], tokens[256:512]),
            token_ids=tokens[512:],
            seed=seed,
            sub_page_size=64,
        )
        self.assertEqual(split, whole[8:])
        unseeded = sub_page_hashes(
            prefix_chunks=(), token_ids=tokens, seed=None, sub_page_size=64
        )
        self.assertNotEqual(unseeded, whole)


class TestPromptSkip(CustomTestCase):
    def test_dcp_ranks_together_send_exactly_the_pages_past_the_prompt(self):
        """Every rank drops only its own sub-pages below the prompt boundary, so
        the union over DCP ranks is the global sub-pages >= skip, none twice."""
        for dcp in (1, 8):
            for num_sub in (dcp, 3 * dcp):
                for prompt_end in range(0, 64 * (num_sub + 2), 64):
                    for node_start in (0, 64 * dcp):
                        skip = skip_sub_pages(
                            node_start=node_start,
                            prompt_end_abs=prompt_end,
                            sub_page_size=64,
                            num_sub_pages=num_sub,
                        )
                        sent = []
                        for rank in range(dcp):
                            keys = local_sub_page_keys(
                                list(range(num_sub)), dcp_size=dcp, dcp_rank=rank
                            )
                            k = local_skip_count(skip=skip, dcp_size=dcp, dcp_rank=rank)
                            sent += keys[k:]
                        want = [
                            i
                            for i in range(num_sub)
                            if node_start + 64 * i >= prompt_end
                        ]
                        self.assertEqual(sorted(sent), want)

    def test_boundary_never_passes_what_prefill_stored(self):
        """Prefill caches up to a Mamba checkpoint depth (multiple of the chunk
        size); the boundary must not exceed it or the hash chain breaks."""
        self.assertEqual(
            prompt_stored_end(prompt_len=32_100, prefill_page_size=64, grid=64), 32_064
        )
        for grid in (64, 128, 256):
            for prompt_len in range(0, 2_000, 37):
                end = prompt_stored_end(
                    prompt_len=prompt_len, prefill_page_size=64, grid=grid
                )
                self.assertLessEqual(end, prompt_len // grid * grid)


class TestWriteState(CustomTestCase):
    def _target(self, prefill_tp=8):
        return resolve_target(
            prefill_attn_tp_size=prefill_tp,
            prefill_pp_size=1,
            prefill_page_size=64,
            target_tp_ranks=[0, 1],
            decode_attn_tp_size=4,
        )

    def test_unsupported_prefill_topologies_are_rejected(self):
        common = dict(prefill_page_size=64, target_tp_ranks=[0])
        self.assertEqual(
            resolve_target(
                prefill_attn_tp_size=8,
                prefill_pp_size=2,
                decode_attn_tp_size=4,
                **common,
            ),
            "prefill_pp",
        )
        self.assertEqual(
            resolve_target(
                prefill_attn_tp_size=2,
                prefill_pp_size=1,
                decode_attn_tp_size=4,
                **common,
            ),
            "attn_tp_ratio",
        )

    def test_mismatch_disables_for_good(self):
        state = DecodeStorageWriteState()
        state.offer(self._target())
        self.assertIsNotNone(state.target())
        state.offer(self._target())
        self.assertIsNotNone(state.target())
        state.offer(self._target(prefill_tp=16))
        self.assertIsNone(state.target())
        self.assertEqual(state.disabled_reason, "topology_mismatch")
        state.offer(self._target())
        self.assertIsNone(state.target())

    def test_tree_page_must_cover_one_sub_page_per_dcp_rank(self):
        plan = build_write_plan(
            self._target(),
            tree_page_size=64,
            dcp_size=4,
            dcp_rank=0,
            attn_tp_size=4,
            attn_tp_rank=0,
            mamba_tensors=[],
        )
        self.assertEqual(plan, "page_size")

    def test_dcp_narrower_than_attn_tp_is_rejected(self):
        """Two attention-TP ranks sharing a dcp_rank would both write the same
        KV sub-pages; only DCP == attention TP has one writer per key."""
        plan = build_write_plan(
            self._target(),
            tree_page_size=128,
            dcp_size=2,
            dcp_rank=0,
            attn_tp_size=4,
            attn_tp_rank=0,
            mamba_tensors=[],
        )
        self.assertEqual(plan, "dcp_attn_tp")


if __name__ == "__main__":
    unittest.main()
