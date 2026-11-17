import tempfile
import unittest
from pathlib import Path

import numpy as np

from temporal_ann.twitter_ann.rolling_hnsw import (
    RollingBuildConfig,
    RollingSearchConfig,
    RollingTemporalHNSW,
    build_rolling_index,
)


class TwitterRollingHNSWTests(unittest.TestCase):
    def setUp(self):
        # Source IDs are intentionally not in timestamp order.
        self.timestamps = np.array(
            [5000, 1000, 3000, 2000, 4000, 6000, 8000, 7000], dtype=np.int64
        )
        self.vectors = np.array(
            [[float(i), float(i % 2)] for i in range(len(self.timestamps))],
            dtype=np.float32,
        )
        self.config = RollingBuildConfig(
            vector_count=8,
            semantic_degree=2,
            gateway_degree=2,
            gateway_period=2,
            construction_candidates=4,
            ef_construction=20,
            candidate_index_m=2,
            threads=1,
        )

    def build(self, directory: str) -> RollingTemporalHNSW:
        root = Path(directory) / "rolling"
        build_rolling_index(
            self.vectors, self.timestamps, root, self.config, progress_every=0
        )
        return RollingTemporalHNSW(root, self.vectors)

    def test_builds_chronological_backbone_and_periodic_gateways(self):
        with tempfile.TemporaryDirectory() as directory:
            index = self.build(directory)
            np.testing.assert_array_equal(index.timestamps, np.arange(1000, 9000, 1000))
            np.testing.assert_array_equal(index.global_ids, [1, 3, 2, 4, 0, 5, 7, 6])
            np.testing.assert_array_equal(index.gateway_ids, [1, 3, 5, 7])
            self.assertEqual(int(index.temporal_neighbors[0, 0]), -1)
            self.assertEqual(int(index.temporal_neighbors[0, 1]), 1)
            self.assertEqual(int(index.temporal_neighbors[4, 0]), 3)
            self.assertEqual(int(index.temporal_neighbors[4, 1]), 5)

    def test_exact_fallback_returns_exact_in_range_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            index = self.build(directory)
            result = index.search(
                np.array([3.1, 1.0], dtype=np.float32),
                [[2000, 6000]],
                RollingSearchConfig(k=3, ef_search=4, exact_threshold=10,
                                    gateway_candidates=4, gateway_seeds=2),
            )
            self.assertEqual(result["method"], "exact")
            self.assertEqual(result["eligible_count"], 4)
            for global_id in result["ids"]:
                self.assertGreaterEqual(self.timestamps[global_id], 2000)
                self.assertLess(self.timestamps[global_id], 6000)

    def test_strict_graph_search_never_scores_out_of_range_nodes(self):
        with tempfile.TemporaryDirectory() as directory:
            index = self.build(directory)
            result = index.search(
                np.array([4.0, 0.0], dtype=np.float32),
                [[1000, 7000]],
                RollingSearchConfig(k=2, ef_search=4, exact_threshold=0,
                                    gateway_candidates=4, gateway_seeds=2),
            )
            self.assertEqual(result["method"], "strict_graph")
            self.assertGreater(result["gateway_seeds"], 0)
            for global_id in result["ids"]:
                self.assertGreaterEqual(self.timestamps[global_id], 1000)
                self.assertLess(self.timestamps[global_id], 7000)

    def test_disjoint_intervals_merge_exact_components(self):
        with tempfile.TemporaryDirectory() as directory:
            index = self.build(directory)
            result = index.search(
                np.array([2.5, 0.0], dtype=np.float32),
                [[1000, 3000], [6000, 8000]],
                RollingSearchConfig(k=4, ef_search=4, exact_threshold=10,
                                    gateway_candidates=4, gateway_seeds=2),
            )
            self.assertEqual(result["eligible_count"], 4)
            self.assertEqual(len(result["ids"]), 4)
            for global_id in result["ids"]:
                timestamp = self.timestamps[global_id]
                self.assertTrue(1000 <= timestamp < 3000 or 6000 <= timestamp < 8000)


if __name__ == "__main__":
    unittest.main()
