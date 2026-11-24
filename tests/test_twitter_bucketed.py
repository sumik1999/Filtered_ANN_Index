import tempfile
import unittest
from pathlib import Path

import numpy as np

from temporal_ann.twitter_ann.bucketed import (
    AdaptivePlanner,
    BucketBuildConfig,
    BucketSearchConfig,
    BucketedHNSW,
    PlannerConfig,
    assign_buckets,
    build_bucket_indexes,
    eligible_ids_from_sorted,
    make_boundaries,
    normalize_intervals,
    resolve_intervals,
)


class TwitterBucketedTests(unittest.TestCase):
    def setUp(self):
        self.timestamps = np.array([1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500], dtype=np.int64)
        self.vectors = np.array([[float(i), 0.0] for i in range(8)], dtype=np.float32)

    def config(self, strategy="fixed"):
        return BucketBuildConfig(4, strategy, m=2, ef_construction=20, threads=1,
                                 timeline_start=1000, timeline_end=5000)

    def test_normalizes_interval_union(self):
        self.assertEqual(normalize_intervals([[20, 30], [10, 20], [40, 50]]), [(10, 30), (40, 50)])

    def test_resolves_and_intersection(self):
        self.assertEqual(
            resolve_intervals([[1000, 4000], [1500, 3500], [2000, 4500]], "AND"),
            [(2000, 3500)],
        )
        self.assertEqual(resolve_intervals([[1000, 2000], [3000, 4000]], "AND"), [])

    def test_fixed_and_density_boundaries_assign_every_record_once(self):
        for strategy in ("fixed", "density"):
            boundaries = make_boundaries(self.timestamps, self.config(strategy))
            assignments = assign_buckets(self.timestamps, boundaries)
            self.assertEqual(len(assignments), len(self.timestamps))
            self.assertTrue(np.all((assignments >= 0) & (assignments < 4)))

    def test_density_boundaries_do_not_split_equal_timestamps(self):
        timestamps = np.array([1000, 1000, 2000, 2000], dtype=np.int64)
        config = BucketBuildConfig(2, "density", timeline_start=0, timeline_end=3000)
        assignments = assign_buckets(timestamps, make_boundaries(timestamps, config))
        self.assertEqual(len(set(assignments[timestamps == 1000])), 1)
        self.assertEqual(len(set(assignments[timestamps == 2000])), 1)

    def test_sorted_timestamp_lookup_supports_interval_union(self):
        order = np.argsort(self.timestamps)
        ids = eligible_ids_from_sorted(self.timestamps[order], order, [[1000, 2000], [3000, 4000]])
        np.testing.assert_array_equal(ids, [0, 1, 4, 5])

    def test_bucket_search_routes_once_filters_boundaries_and_reranks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "index"
            build_bucket_indexes(self.vectors, self.timestamps, root, self.config())
            index = BucketedHNSW(root)
            # Both predicates intersect bucket 1; it must still be searched once.
            intervals = [[1900, 2300], [2400, 3100]]
            self.assertEqual(index.selected_buckets(intervals), [0, 1, 2])
            result = index.search(
                np.array([3.1, 0.0], dtype=np.float32), self.timestamps, self.vectors,
                intervals, 3, BucketSearchConfig(20, 10, 1.0),
            )
            np.testing.assert_array_equal(result["ids"], [3, 4, 2])
            self.assertEqual(result["selected_bucket_count"], 3)
            self.assertGreater(result["partial_bucket_count"], 0)

    def test_adaptive_planner_selects_exact_bucket_and_global_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "index"
            build_bucket_indexes(self.vectors, self.timestamps, root, self.config())
            index = BucketedHNSW(root)
            exact = AdaptivePlanner(index, PlannerConfig(2, 0.5, 4))
            self.assertEqual(exact.choose([[1000, 1501]])[0], "exact_pre_filter")
            bucket = AdaptivePlanner(index, PlannerConfig(0, 0.5, 4))
            self.assertEqual(bucket.choose([[1000, 2001]])[0], "bucket_hnsw")
            global_planner = AdaptivePlanner(index, PlannerConfig(0, 0.1, 1))
            self.assertEqual(global_planner.choose([[1000, 5000]])[0], "global_hnsw_post_filter")


if __name__ == "__main__":
    unittest.main()
