import unittest

import numpy as np

from temporal_ann.twitter_ann.workload import (
    HOUR_MS,
    centered_interval,
    contains,
    generate_intervals,
    intervals_are_valid,
    split_coverage_ms,
)
from temporal_ann.twitter_ann.bucketed import resolve_intervals
from temporal_ann.twitter_data import YEAR_START_MS


class TwitterVariedWorkloadTests(unittest.TestCase):
    def setUp(self):
        # Widely separated anchors make all fragmentation levels feasible.
        anchors = np.array([100, 1000, 2000, 3000, 4000, 5000, 6000, 7000], dtype=np.int64)
        self.pools = {name: anchors for name in ("burst", "ordinary", "quiet")}
        self.query_time = YEAR_START_MS + 4500 * HOUR_MS + HOUR_MS // 2

    def test_coverage_is_split_exactly_across_predicates(self):
        widths = split_coverage_ms(1, 3)
        self.assertEqual(len(widths), 3)
        self.assertEqual(sum(widths), HOUR_MS)
        self.assertLessEqual(max(widths) - min(widths), 1)

    def test_centered_interval_preserves_width_at_year_boundary(self):
        start, end = centered_interval(YEAR_START_MS, 24 * HOUR_MS)
        self.assertEqual(start, YEAR_START_MS)
        self.assertEqual(end - start, 24 * HOUR_MS)

    def test_aligned_fragmented_predicates_are_disjoint_and_contain_query_time(self):
        intervals = generate_intervals(
            self.query_time, 4, 24, "burst", "aligned", self.pools,
            np.random.default_rng(7),
        )
        self.assertTrue(intervals_are_valid(intervals, 24 * HOUR_MS))
        self.assertTrue(contains(intervals, self.query_time))

    def test_and_predicates_overlap_with_exact_effective_coverage(self):
        intervals = generate_intervals(
            self.query_time, 4, 24, "ordinary", "aligned", self.pools,
            np.random.default_rng(9), "AND",
        )
        self.assertEqual(len(intervals), 4)
        self.assertTrue(intervals_are_valid(intervals, 24 * HOUR_MS, "AND"))
        effective = resolve_intervals(intervals, "AND")
        self.assertEqual(effective[0][1] - effective[0][0], 24 * HOUR_MS)
        self.assertTrue(contains(effective, self.query_time))

    def test_independent_predicates_exclude_query_time(self):
        intervals = generate_intervals(
            self.query_time, 4, 720, "quiet", "independent", self.pools,
            np.random.default_rng(7),
        )
        self.assertTrue(intervals_are_valid(intervals, 720 * HOUR_MS))
        self.assertFalse(contains(intervals, self.query_time))


if __name__ == "__main__":
    unittest.main()
