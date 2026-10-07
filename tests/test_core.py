import unittest

import numpy as np

from temporal_ann.core import Dataset, TemporalPredicate, exact_search
from temporal_ann.methods import FixedTemporalPartition


class PredicateTests(unittest.TestCase):
    def test_union_normalizes_overlapping_intervals(self):
        predicate = TemporalPredicate.from_intervals([(0.1, 0.3), (0.2, 0.5)])
        self.assertEqual(predicate.intervals, ((0.1, 0.5),))

    def test_intersection(self):
        left = TemporalPredicate.from_intervals([(0.1, 0.4), (0.7, 0.9)])
        right = TemporalPredicate.from_intervals([(0.3, 0.8)])
        self.assertEqual(left.intersection(right).intervals, ((0.3, 0.4), (0.7, 0.8)))


class SearchTests(unittest.TestCase):
    def setUp(self):
        self.dataset = Dataset(
            vectors=np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32),
            timestamps=np.array([0.1, 0.3, 0.6, 0.9]),
        )

    def test_exact_search_obeys_predicate(self):
        predicate = TemporalPredicate.from_intervals([(0.2, 0.8)])
        result = exact_search(self.dataset, np.array([1.8], dtype=np.float32), predicate, 2)
        np.testing.assert_array_equal(result.ids, [2, 1])

    def test_fixed_partition_matches_exact(self):
        predicate = TemporalPredicate.from_intervals([(0.0, 0.35), (0.55, 0.7)])
        query = np.array([1.7], dtype=np.float32)
        expected = exact_search(self.dataset, query, predicate, 3)
        actual = FixedTemporalPartition(self.dataset, 2).search(query, predicate, 3)
        np.testing.assert_array_equal(actual.ids, expected.ids)


if __name__ == "__main__":
    unittest.main()
