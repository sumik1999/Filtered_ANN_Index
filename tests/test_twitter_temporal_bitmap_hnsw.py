import tempfile
import unittest
from pathlib import Path

import numpy as np

from temporal_ann.twitter_ann.temporal_bitmap_hnsw import (
    TemporalBitmapBuildConfig,
    TemporalBitmapHNSW,
    TemporalBitmapSearchConfig,
    _bitmap_intersects,
    build_temporal_bitmap_index,
)


class TwitterTemporalBitmapHNSWTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        generator = np.random.default_rng(12)
        values = generator.normal(size=(24, 8)).astype(np.float32)
        self.vectors = values / np.linalg.norm(values, axis=1, keepdims=True)
        # Source order is deliberately not chronological.
        ordered_timestamps = np.arange(24, dtype=np.int64) * 30 * 60 * 1000
        permutation = generator.permutation(24)
        self.timestamps = ordered_timestamps[permutation]
        self.vectors = self.vectors[permutation]
        self.config = TemporalBitmapBuildConfig(
            vector_count=24,
            local_semantic_degree=2,
            historical_semantic_degree=2,
            future_reciprocal_degree=2,
            local_window_hours=1,
            range_degree_per_side=1,
            range_scales_hours="1,4",
            range_candidate_count=4,
            bridge_degree_per_side=1,
            bridge_scales_epochs="1,2,4",
            epoch_hours=1,
            gateway_period=2,
            construction_candidates=4,
            ef_construction=20,
            candidate_index_m=2,
            threads=1,
        )
        self.path = Path(self.temporary.name) / "bitmap"
        build_temporal_bitmap_index(
            self.vectors, self.timestamps, self.path, self.config,
            progress_every=0,
        )
        self.index = TemporalBitmapHNSW(self.path, self.vectors)

    def tearDown(self):
        self.temporary.cleanup()

    def test_new_nodes_have_historical_edges_and_old_nodes_get_future_edges(self):
        self.assertTrue(np.any(self.index.historical_neighbors[1:] >= 0))
        self.assertTrue(np.any(self.index.future_neighbors[:-1] >= 0))
        for row, neighbors in enumerate(self.index.historical_neighbors):
            self.assertTrue(all(int(value) < row for value in neighbors if value >= 0))
        for row, neighbors in enumerate(self.index.future_neighbors):
            self.assertTrue(all(int(value) > row for value in neighbors if value >= 0))

    def test_local_edges_stay_in_the_same_epoch(self):
        for row, neighbors in enumerate(self.index.local_neighbors):
            for neighbor in neighbors:
                if neighbor >= 0:
                    self.assertEqual(
                        int(self.index.epoch_ids[row]),
                        int(self.index.epoch_ids[int(neighbor)]),
                    )

    def test_temporal_backbone_and_multiscale_bridges_are_bidirectional(self):
        np.testing.assert_array_equal(
            self.index.temporal_neighbors[1:, 0], np.arange(23, dtype=np.int32)
        )
        np.testing.assert_array_equal(
            self.index.temporal_neighbors[:-1, 1], np.arange(1, 24, dtype=np.int32)
        )
        self.assertTrue(np.any(self.index.bridge_left_neighbors >= 0))
        self.assertTrue(np.any(self.index.bridge_right_neighbors >= 0))

    def test_edge_bitmaps_have_no_false_negatives(self):
        groups = {
            "local": self.index.local_neighbors,
            "historical": self.index.historical_neighbors,
            "future": self.index.future_neighbors,
            "temporal": self.index.temporal_neighbors,
            "bridge": np.concatenate((
                self.index.bridge_left_neighbors,
                self.index.bridge_right_neighbors,
            ), axis=2),
        }
        for scale_index, scale in enumerate(self.config.range_scales()):
            groups[f"range_{scale}h"] = np.concatenate((
                self.index.range_left_neighbors[:, scale_index, :],
                self.index.range_right_neighbors[:, scale_index, :],
            ), axis=1)
        for name, adjacency in groups.items():
            group = self.index.group_index[name]
            for row, neighbors in enumerate(adjacency):
                for neighbor in np.asarray(neighbors).reshape(-1):
                    if neighbor < 0:
                        continue
                    epoch = int(self.index.epoch_ids[int(neighbor)])
                    self.assertTrue(
                        _bitmap_intersects(
                            self.index.edge_epoch_bitmaps[row, group], epoch, epoch
                        ),
                        (name, row, int(neighbor), epoch),
                    )

    def test_bitmap_pruning_is_result_equivalent_and_skips_groups(self):
        query = np.asarray(self.vectors[self.index.global_ids[8]], dtype=np.float32)
        intervals = [[3 * 60 * 60 * 1000, 5 * 60 * 60 * 1000]]
        enabled = self.index.search(
            query, intervals,
            TemporalBitmapSearchConfig(
                k=3, ef_search=8, exact_threshold=0,
                gateway_candidates=4, gateway_seeds=2,
                use_bitmap_pruning=True,
            ),
        )
        disabled = self.index.search(
            query, intervals,
            TemporalBitmapSearchConfig(
                k=3, ef_search=8, exact_threshold=0,
                gateway_candidates=4, gateway_seeds=2,
                use_bitmap_pruning=False,
            ),
        )
        np.testing.assert_array_equal(enabled["ids"], disabled["ids"])
        np.testing.assert_allclose(enabled["distances"], disabled["distances"])
        self.assertGreater(enabled["bitmap_groups_skipped"], 0)
        self.assertLessEqual(enabled["edges_examined"], disabled["edges_examined"])

    def test_random_range_queries_preserve_strict_filtering_with_bitmaps(self):
        rng = np.random.default_rng(18)
        for _ in range(30):
            start = int(rng.integers(0, 20)) * 30 * 60 * 1000
            end = start + int(rng.integers(1, 7)) * 30 * 60 * 1000
            result = self.index.search(
                rng.normal(size=8).astype(np.float32), [[start, end]],
                TemporalBitmapSearchConfig(
                    k=4, ef_search=8, exact_threshold=0,
                    gateway_candidates=4, gateway_seeds=2,
                ),
            )
            for global_id in result["ids"]:
                self.assertGreaterEqual(self.timestamps[global_id], start)
                self.assertLess(self.timestamps[global_id], end)


if __name__ == "__main__":
    unittest.main()
