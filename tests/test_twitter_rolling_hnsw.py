import tempfile
import unittest
from dataclasses import replace
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

            semantic_only = index.search(
                np.array([4.0, 0.0], dtype=np.float32),
                [[1000, 7000]],
                RollingSearchConfig(k=2, ef_search=4, exact_threshold=0,
                                    gateway_candidates=4, gateway_seeds=2,
                                    use_temporal_edges=False, use_gateway_edges=False),
            )
            self.assertEqual(semantic_only["temporal_edges_examined"], 0)
            self.assertEqual(semantic_only["gateway_edges_examined"], 0)

    def test_range_edges_use_separate_left_and_right_budgets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "rolling_range"
            build_rolling_index(
                self.vectors, self.timestamps, root,
                replace(self.config, semantic_degree=1, range_degree_per_side=2),
                progress_every=0,
            )
            index = RollingTemporalHNSW(root, self.vectors)
            self.assertTrue(np.any(index.range_left_neighbors[1:] >= 0))
            self.assertTrue(np.any(index.range_right_neighbors[:-1] >= 0))
            result = index.search(
                np.array([4.0, 0.0], dtype=np.float32), [[1000, 7000]],
                RollingSearchConfig(k=2, ef_search=4, exact_threshold=0,
                                    gateway_candidates=4, gateway_seeds=2,
                                    use_range_edges=True),
            )
            self.assertGreater(result["range_edges_examined"], 0)

    def test_multi_scale_range_edges_have_independent_budgets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "rolling_multi_range"
            build_rolling_index(
                self.vectors, self.timestamps, root,
                replace(self.config, semantic_degree=1, range_degree_per_side=1,
                        range_scales_hours="1,2,4"),
                progress_every=0,
            )
            index = RollingTemporalHNSW(root, self.vectors)
            self.assertEqual(index.range_left_neighbors.shape, (8, 3, 1))
            self.assertEqual(index.range_right_neighbors.shape, (8, 3, 1))
            self.assertEqual(index.manifest["range_scales_hours"], [1, 2, 4])
            self.assertTrue(np.all(np.any(index.range_left_neighbors[1:] >= 0, axis=0)))

    def test_hierarchy_has_semantic_and_protected_temporal_links(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "rolling_hierarchy"
            build_rolling_index(
                self.vectors, self.timestamps, root,
                replace(self.config, hierarchy_enabled=True,
                        hierarchy_semantic_degree=2,
                        hierarchy_level_probability=0.9,
                        hierarchy_max_level=2,
                        hierarchy_construction_candidates=4),
                progress_every=0,
            )
            index = RollingTemporalHNSW(root, self.vectors)
            self.assertGreater(int(np.max(index.levels)), 0)
            self.assertTrue(np.any(index.upper_semantic_neighbors >= 0))
            self.assertTrue(np.any(index.upper_temporal_neighbors >= 0))
            result = index.search(
                np.array([4.0, 0.0], dtype=np.float32), [[1000, 7000]],
                RollingSearchConfig(k=2, ef_search=4, exact_threshold=0,
                                    gateway_candidates=4, gateway_seeds=2,
                                    use_hierarchy=True, upper_ef_search=4),
            )
            self.assertIn("strict_hierarchy", result["method"])
            self.assertGreater(result["upper_entry_level"], 0)
            self.assertGreater(result["upper_nodes_visited"], 0)

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
