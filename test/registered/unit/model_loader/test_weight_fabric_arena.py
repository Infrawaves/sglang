"""CPU-only tests for the fabric weight arena's placement arithmetic.

The arena exists so mooncake's MNNVL transport can serve a seed's weights, which
it can only do for cuMemCreate(FABRIC) memory. Two invariants make that work, and
both are arithmetic that fails silently:

* a chunk must hold any single weight whole, or a client cannot resolve it;
* the fit check must model what torch's allocator *asks for*, not the tensor size,
  or a chunk we believe has room returns nullptr mid-migration.

The CUDA paths (reserve, map, export) are covered by the on-device probe instead;
what is testable here is the arithmetic and the bookkeeping around it.
"""

import unittest
from types import SimpleNamespace

from sglang.srt.model_loader.weight_fabric_arena import (
    FabricArenaUnavailable,
    allocator_request_bytes,
    migratable_parameters,
    names_outside_arena,
    plan_chunk_bytes,
    plan_chunk_count,
    reject_unmigratable,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_MiB = 1 << 20


def _tensor(*, ptr, nbytes, contiguous=True, device="cuda"):
    # The functions under test touch only these attributes, so a stand-in keeps
    # this suite CPU-only and independent of a CUDA allocator's real addresses.
    return SimpleNamespace(
        device=SimpleNamespace(type=device),
        data_ptr=lambda: ptr,
        numel=lambda: nbytes,
        element_size=lambda: 1,
        is_contiguous=lambda: contiguous,
    )


def _model(params):
    return SimpleNamespace(named_parameters=lambda remove_duplicate: iter(params))


class TestAllocatorRequestBytes(CustomTestCase):
    """torch's caching allocator asks the pluggable allocator for a rounded-up
    segment, not the tensor. Underestimating here is not a packing inefficiency:
    the bump allocator returns nullptr and torch raises OOM partway through a
    migration that cannot be rolled back.
    """

    def test_small_allocations_round_to_the_small_buffer(self):
        for nbytes in (1, 4096, 1 * _MiB):
            with self.subTest(nbytes=nbytes):
                self.assertEqual(allocator_request_bytes(nbytes), 2 * _MiB)

    def test_the_expensive_band_rounds_to_the_large_buffer(self):
        # Anything in (1 MiB, 10 MiB) costs a 20 MiB segment. This is where the
        # arena wastes most, so the fit check has to know rather than discover it.
        for nbytes in (1 * _MiB + 1, 4 * _MiB, 9 * _MiB):
            with self.subTest(nbytes=nbytes):
                self.assertEqual(allocator_request_bytes(nbytes), 20 * _MiB)

    def test_large_allocations_round_up_to_the_granule(self):
        self.assertEqual(allocator_request_bytes(10 * _MiB), 10 * _MiB)
        self.assertEqual(allocator_request_bytes(10 * _MiB + 1), 12 * _MiB)
        self.assertEqual(allocator_request_bytes(101 * _MiB), 102 * _MiB)

    def test_it_never_returns_less_than_asked(self):
        for nbytes in (1, _MiB - 1, _MiB, _MiB + 1, 10 * _MiB, 337 * _MiB + 7):
            with self.subTest(nbytes=nbytes):
                self.assertGreaterEqual(allocator_request_bytes(nbytes), nbytes)


class TestPlanChunkBytes(CustomTestCase):
    """A weight bigger than a chunk could not be mapped by one handle, and a
    client resolves a remote pointer by finding the one published buffer that
    wholly contains it. So the largest weight, not a constant, sets the floor --
    a fused MoE tensor is routinely larger than any default worth picking.
    """

    def test_a_small_model_gets_the_floor(self):
        self.assertEqual(plan_chunk_bytes(8 * _MiB, 2 * _MiB), 512 * _MiB)

    def test_a_weight_larger_than_the_floor_raises_the_chunk(self):
        chunk = plan_chunk_bytes(3 * 1024 * _MiB, 2 * _MiB)
        self.assertGreaterEqual(chunk, 3 * 1024 * _MiB)

    def test_the_chunk_always_holds_the_largest_weight_as_allocated(self):
        # The allocator's request, not the tensor, is what must fit.
        for largest in (1, 9 * _MiB, 10 * _MiB, 700 * _MiB, 2048 * _MiB + 3):
            with self.subTest(largest=largest):
                chunk = plan_chunk_bytes(largest, 2 * _MiB)
                self.assertGreaterEqual(chunk, allocator_request_bytes(largest))

    def test_the_chunk_is_granularity_aligned(self):
        # An unaligned chunk would make cuMemMap reject the mapping.
        for granularity in (2 * _MiB, 32 * _MiB):
            with self.subTest(granularity=granularity):
                self.assertEqual(
                    plan_chunk_bytes(9000 * _MiB, granularity) % granularity, 0
                )


class TestPlanChunkCount(CustomTestCase):
    """The reservation is sized by simulating the packing, not by scaling the
    tensor sum. A guess low fails inside the migration, which is the one place
    that cannot be rolled back: parameters already rebound point into the arena,
    so it cannot be closed without freeing live weights.
    """

    CHUNK = 512 * _MiB

    def test_a_model_that_fits_needs_one_chunk(self):
        self.assertEqual(plan_chunk_count([64 * _MiB] * 8, self.CHUNK), 1)

    def test_it_opens_a_second_chunk_when_the_first_is_full(self):
        self.assertEqual(plan_chunk_count([256 * _MiB] * 3, self.CHUNK), 2)

    def test_the_expensive_band_is_what_this_exists_for(self):
        # 400 tensors of 1.5 MiB are 600 MiB of weights but 8 GiB of segments,
        # because each costs 20 MiB. Sizing the reservation at a small multiple of
        # the tensor sum -- as this did before -- runs out partway through.
        sizes = [3 * _MiB // 2] * 400
        chunks = plan_chunk_count(sizes, self.CHUNK)
        self.assertEqual(chunks, 16)
        self.assertGreater(chunks * self.CHUNK, sum(sizes) * 4)

    def test_it_matches_the_walk_the_migration_actually_performs(self):
        """The invariant: the plan is never under what the real loop consumes.

        Both are first-fit over ``allocator_request_bytes`` in the same order, so
        this drives an independent walker with the same rule and compares.
        """
        cases = [
            [64 * _MiB] * 8,
            [256 * _MiB] * 3,
            [3 * _MiB // 2] * 400,
            [700 * _MiB, 1 * _MiB, 9 * _MiB, 300 * _MiB],
            sorted([13 * _MiB, 2 * _MiB, 511 * _MiB, 40 * _MiB], reverse=True),
            [1],
        ]
        for sizes in cases:
            with self.subTest(sizes=f"{len(sizes)} tensors, {sum(sizes) >> 20} MiB"):
                chunk = plan_chunk_bytes(max(sizes), 2 * _MiB)
                consumed, used = 1, 0
                for nbytes in sizes:
                    request = allocator_request_bytes(nbytes)
                    if used + request > chunk:
                        consumed += 1
                        used = 0
                    used += request
                self.assertEqual(plan_chunk_count(sizes, chunk), consumed)

    def test_a_single_weight_never_needs_a_second_chunk(self):
        # plan_chunk_bytes sizes the chunk to the largest weight, so the biggest
        # tensor in any model is always placeable -- otherwise no mapping could
        # hold it whole and no client could resolve it.
        for largest in (1, 9 * _MiB, 700 * _MiB, 3000 * _MiB):
            with self.subTest(largest=largest):
                chunk = plan_chunk_bytes(largest, 2 * _MiB)
                self.assertEqual(plan_chunk_count([largest], chunk), 1)


class TestNamesOutsideArena(CustomTestCase):
    """The last line of defence before publishing. Migration deduplicates by
    storage, so two distinct Parameters over one allocation leave the non-survivor
    on the original memory -- in the arena's manifest, covered by no mapping. That
    is exactly the "Requested address ... not found!" the arena exists to remove,
    so it has to be caught here and cost the NIC instead.
    """

    def _arena(self, mappings):
        return SimpleNamespace(mappings=mappings)

    def test_weights_inside_one_mapping_are_accepted(self):
        arena = self._arena([(0x1000, 0x1000)])
        manifest = {"a": (0x1000, 512, 2), "b": (0x1800, 1024, 2)}
        self.assertEqual(names_outside_arena(manifest, arena), [])

    def test_a_weight_before_every_mapping_is_caught(self):
        arena = self._arena([(0x2000, 0x1000)])
        self.assertEqual(names_outside_arena({"a": (0x1000, 8, 1)}, arena), ["a"])

    def test_a_weight_past_its_mappings_end_is_caught(self):
        # Starts inside, ends outside: the client's containment test fails, so a
        # bounds check on the start alone would publish an unreachable weight.
        arena = self._arena([(0x1000, 0x1000)])
        self.assertEqual(names_outside_arena({"a": (0x1F00, 512, 1)}, arena), ["a"])

    def test_a_weight_in_the_gap_between_mappings_is_caught(self):
        # Chunks are separate cuMemCreate mappings and need not be adjacent, so
        # "after mapping 0 and before mapping 1" is a real address.
        arena = self._arena([(0x1000, 0x1000), (0x9000, 0x1000)])
        self.assertEqual(names_outside_arena({"a": (0x5000, 8, 1)}, arena), ["a"])

    def test_it_picks_the_right_mapping_out_of_many(self):
        arena = self._arena([(0x1000, 0x1000), (0x5000, 0x1000), (0x9000, 0x1000)])
        manifest = {
            "first": (0x1000, 8, 1),
            "middle": (0x5800, 8, 1),
            "last": (0x9FF8, 8, 1),
            "past_last": (0xA000, 8, 1),
        }
        self.assertEqual(names_outside_arena(manifest, arena), ["past_last"])

    def test_unsorted_mappings_are_handled(self):
        # mappings is append-ordered by open_chunk, which need not be ascending.
        arena = self._arena([(0x9000, 0x1000), (0x1000, 0x1000)])
        self.assertEqual(names_outside_arena({"a": (0x1000, 8, 1)}, arena), [])

    def test_the_byte_count_comes_from_numel_times_element_size(self):
        # A manifest entry is (ptr, numel, element_size); multiplying wrongly
        # would under-measure a wide dtype and pass a straddling weight.
        arena = self._arena([(0x1000, 0x1000)])
        manifest = {"fits": (0x1000, 2048, 2), "straddles": (0x1000, 4096, 2)}
        self.assertEqual(names_outside_arena(manifest, arena), ["straddles"])


class TestMigratableParameters(CustomTestCase):
    def test_aliases_over_one_storage_yield_one_entry(self):
        # Moving one and rebinding only that name would leave the alias pointing
        # at memory the migration freed.
        shared = _tensor(ptr=0x1000, nbytes=64)
        params = [("a", shared), ("a_alias", shared)]
        self.assertEqual(len(migratable_parameters(_model(params))), 1)

    def test_largest_first(self):
        # So a chunk's tail is filled by tensors small enough to fit it, rather
        # than blocked by one that never will.
        params = [
            ("small", _tensor(ptr=0x1000, nbytes=10)),
            ("big", _tensor(ptr=0x2000, nbytes=1000)),
            ("mid", _tensor(ptr=0x3000, nbytes=100)),
        ]
        got = [name for name, _ in migratable_parameters(_model(params))]
        self.assertEqual(got, ["big", "mid", "small"])

    def test_non_cuda_parameters_are_skipped(self):
        params = [
            ("cpu", _tensor(ptr=0x1000, nbytes=64, device="cpu")),
            ("gpu", _tensor(ptr=0x2000, nbytes=64)),
        ]
        got = [name for name, _ in migratable_parameters(_model(params))]
        self.assertEqual(got, ["gpu"])

    def test_no_cuda_parameters_is_not_an_error(self):
        self.assertEqual(migratable_parameters(_model([])), [])


class TestRejectUnmigratable(CustomTestCase):
    """Refusals have to happen before the first rebind. After it the arena cannot
    be closed without freeing live weights, so there is no way back -- a
    half-migrated model publishes a manifest whose remainder fails at the client.
    """

    def test_a_non_contiguous_weight_is_refused(self):
        params = [("odd", _tensor(ptr=0x1000, nbytes=64, contiguous=False))]
        with self.assertRaises(FabricArenaUnavailable):
            reject_unmigratable(params)

    def test_the_message_names_the_weight(self):
        params = [("layers.0.w", _tensor(ptr=0x1000, nbytes=64, contiguous=False))]
        with self.assertRaisesRegex(FabricArenaUnavailable, "layers.0.w"):
            reject_unmigratable(params)

    def test_ordinary_weights_pass(self):
        params = [("a", _tensor(ptr=0x1000, nbytes=64))]
        reject_unmigratable(params)


if __name__ == "__main__":
    unittest.main()
