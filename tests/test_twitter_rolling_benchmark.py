import unittest

import numpy as np

from temporal_ann.twitter_ann.bucketed import resolve_intervals
from temporal_ann.twitter_ann.rolling_benchmark import make_and_intervals, make_intervals


class TwitterRollingBenchmarkTests(unittest.TestCase):
    def test_generated_intervals_are_disjoint_and_preserve_coverage(self):
        rng = np.random.default_rng(19)
        for fragments in (1, 2, 4):
            intervals = make_intervals(0, 1_000_000_000, 24, fragments, rng)
            self.assertEqual(len(intervals), fragments)
            self.assertEqual(sum(end - start for start, end in intervals), 24 * 3_600_000)
            self.assertTrue(all(start < end for start, end in intervals))
            self.assertTrue(all(intervals[i][1] <= intervals[i + 1][0]
                                for i in range(len(intervals) - 1)))

    def test_and_intervals_preserve_intersection_coverage(self):
        intervals = make_and_intervals(
            0, 1_000_000_000, 24, 4, np.random.default_rng(19)
        )
        self.assertEqual(len(intervals), 4)
        effective = resolve_intervals(intervals, "AND")
        self.assertEqual(effective[0][1] - effective[0][0], 24 * 3_600_000)

    def test_rejects_coverage_larger_than_timeline(self):
        with self.assertRaises(ValueError):
            make_intervals(0, 1_000, 1, 1, np.random.default_rng(1))


if __name__ == "__main__":
    unittest.main()
