import unittest

import numpy as np

from temporal_ann.twitter_ann.gateway_oracle import (
    candidate_pool,
    select_greedy_oracle,
    select_random,
    select_temporal_density,
)


class TwitterGatewayOracleTests(unittest.TestCase):
    def test_candidate_pool_and_random_selection_are_equal_budget(self):
        pool = candidate_pool(100, 10)
        np.testing.assert_array_equal(pool, np.arange(9, 100, 10))
        first = select_random(pool, 4, 7)
        second = select_random(pool, 4, 7)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(len(np.unique(first)), 4)
        self.assertTrue(np.all(np.diff(first) > 0))

    def test_temporal_density_selects_one_per_population_block(self):
        pool = np.arange(12, dtype=np.int64)
        timestamps = np.arange(12, dtype=np.int64) * 3_600_000
        # Make IDs 2 and 8 occur in denser hours by assigning surrounding nodes.
        timestamps[0:4] = 2 * 3_600_000
        timestamps[6:11] = 8 * 3_600_000
        selected = select_temporal_density(pool, timestamps, 3)
        self.assertEqual(len(selected), 3)
        self.assertTrue(np.all(np.diff(selected) > 0))

    def test_greedy_oracle_covers_distinct_query_utility(self):
        pool = np.array([10, 20, 30, 40], dtype=np.int64)
        utility = np.array([
            [1.0, 0.1, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.1],
        ], dtype=np.float32)
        selected = select_greedy_oracle(pool, utility, 2)
        np.testing.assert_array_equal(selected, [10, 30])


if __name__ == "__main__":
    unittest.main()
