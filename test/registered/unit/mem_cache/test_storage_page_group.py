import unittest

from sglang.srt.mem_cache.storage_page_group import group_segments, plan_group_reads
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestGroupedReads(unittest.TestCase):
    def _plan(self, start, end, group, nodes):
        stored = {
            segment
            for lo, hi in nodes
            for segment in group_segments(lo, hi - lo, group)
        }
        calls = []

        def probe(segments):
            calls.append(segments)
            return [seg in stored for seg in segments]

        plan = plan_group_reads(start, end - start, group, probe)
        probed = [seg for call in calls for seg in call]
        self.assertEqual(len(probed), len(set(probed)), "repeated existence probe")
        return plan, calls

    def test_adjacent_split_groups(self):
        for group, end, nodes in (
            (4, 16, [(0, 6), (6, 9), (9, 16)]),
            (2, 128, [(0, 1), (1, 3), (3, 128)]),
        ):
            with self.subTest(group=group):
                plan, _ = self._plan(0, end, group, nodes)
                expected = [
                    seg
                    for lo, hi in nodes
                    for seg in group_segments(lo, hi - lo, group)
                ]
                self.assertEqual(plan.chain(0, end), (expected, end))
                self.assertEqual(plan.restorable(0), [hi for _, hi in expected])

    def test_gap_stops_before_unreachable_suffix(self):
        plan, calls = self._plan(0, 15, 4, [(0, 5), (6, 15)])
        self.assertEqual(plan.chain(0, 15), ([(0, 4), (4, 5)], 5))
        self.assertEqual(plan.restorable(0), [4, 5])
        self.assertFalse(any(lo >= 8 for batch in calls[1:] for lo, _ in batch))

    def test_partial_first_page_can_read_leading_pages_into_scratch(self):
        plan, _ = self._plan(2, 11, 4, [(0, 4), (4, 6), (6, 9), (9, 11)])
        self.assertEqual(
            plan.chain(2, 11), ([(0, 4), (4, 6), (6, 8), (8, 9), (9, 11)], 11)
        )

    def test_all_full_groups_need_one_probe(self):
        plan, calls = self._plan(0, 384, 128, [(0, 384)])
        full = [(0, 128), (128, 256), (256, 384)]
        self.assertEqual(calls, [full])
        self.assertEqual(plan.chain(0, 384), (full, 384))

    def test_separated_node_boundaries_are_probed_in_batches(self):
        # Each odd group is split after its first page. Adding more such groups
        # should increase the key count, not the number of Mooncake round trips.
        for group, max_calls in ((2, 2), (4, 3)):
            with self.subTest(group=group):
                end = 65 * group
                cuts = [0, *(i * group + 1 for i in range(1, 65, 2)), end]
                plan, calls = self._plan(0, end, group, zip(cuts, cuts[1:]))
                self.assertEqual(plan.chain(0, end)[1], end)
                self.assertLessEqual(len(calls), max_calls)

    def test_wide_cold_group_probes_only_reachable_starts(self):
        for end in (127, 1024):
            with self.subTest(end=end):
                plan, calls = self._plan(0, end, 128, [])
                self.assertEqual(plan.restorable(0), [])
                self.assertEqual(calls[-1], [(0, hi) for hi in range(1, 128)])
                self.assertEqual(len(calls), 1 if end == 127 else 2)

    def test_empty_query(self):
        plan, calls = self._plan(4, 4, 2, [])
        self.assertEqual(plan.chain(4, 4), ([], 4))
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
