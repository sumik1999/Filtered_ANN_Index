import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from temporal_ann.fixed_hnsw import (
    FixedHNSWConfig,
    FixedHNSWIndex,
    FixedHNSWSearchConfig,
    assign_buckets,
    build_fixed_hnsw,
)


class FixedHNSWTests(unittest.TestCase):
    def setUp(self):
        self.vectors = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [2.0, 0.0],
                [3.0, 0.0],
                [4.0, 0.0],
                [5.0, 0.0],
            ],
            dtype=np.float32,
        )
        self.timestamps = np.array([0.0, 0.25, 0.25, 0.49, 0.5, 0.99])
        self.config = FixedHNSWConfig(
            bucket_count=4, m=4, ef_construction=40, threads=1
        )

    def build(self, directory: Path) -> FixedHNSWIndex:
        build_fixed_hnsw(
            self.vectors, self.timestamps, directory, self.config
        )
        return FixedHNSWIndex(directory)

    def test_assignment_uses_half_open_boundaries_and_preserves_duplicates(self):
        np.testing.assert_array_equal(
            assign_buckets(self.timestamps, self.config), [0, 1, 1, 1, 2, 3]
        )

    def test_out_of_range_timestamps_are_rejected(self):
        with self.assertRaises(ValueError):
            assign_buckets(np.array([0.0, 1.0]), self.config)

    def test_build_persists_manifest_indexes_and_global_id_maps(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "fixed"
            manifest = build_fixed_hnsw(
                self.vectors,
                self.timestamps,
                directory,
                self.config,
                source_metadata={"test": True},
            )
            self.assertEqual(manifest["vector_count"], 6)
            self.assertEqual([x["vector_count"] for x in manifest["buckets"]], [1, 3, 1, 1])
            self.assertTrue((directory / "manifest.json").is_file())
            ids = np.load(directory / "bucket_0001_ids.npy")
            np.testing.assert_array_equal(ids, [1, 2, 3])
            persisted = json.loads((directory / "manifest.json").read_text())
            self.assertGreater(persisted["index_size_bytes"], 0)
            self.assertGreater(persisted["mapping_size_bytes"], 0)

    def test_existing_directory_requires_matching_configuration(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "fixed"
            build_fixed_hnsw(self.vectors, self.timestamps, directory, self.config)
            different = FixedHNSWConfig(
                bucket_count=2, m=4, ef_construction=40, threads=1
            )
            with self.assertRaises(ValueError):
                build_fixed_hnsw(self.vectors, self.timestamps, directory, different)

    def test_bucket_routing_obeys_exact_boundaries_and_deduplicates(self):
        with tempfile.TemporaryDirectory() as temporary:
            index = self.build(Path(temporary) / "fixed")
            self.assertEqual(index.selected_buckets([[0.0, 0.25]]), [0])
            self.assertEqual(index.selected_buckets([[0.25, 0.5]]), [1])
            self.assertEqual(index.selected_buckets([[0.49, 0.51]]), [1, 2])
            self.assertEqual(
                index.selected_buckets([[0.0, 0.3], [0.2, 0.6]]), [0, 1, 2]
            )
            self.assertEqual(index.selected_buckets([[1.1, 1.2]]), [])

    def test_search_filters_boundaries_and_reranks_across_buckets(self):
        with tempfile.TemporaryDirectory() as temporary:
            index = self.build(Path(temporary) / "fixed")
            result = index.search(
                np.array([2.6, 0.0], dtype=np.float32),
                self.timestamps,
                self.vectors,
                [[0.25, 0.5], [0.9, 1.0]],
                k=3,
                search_config=FixedHNSWSearchConfig(ef_search=64, overfetch=10),
            )
            np.testing.assert_array_equal(result["ids"], [3, 2, 1])
            np.testing.assert_allclose(
                result["distances"], [0.16, 0.36, 2.56], rtol=1e-5
            )
            self.assertEqual(result["selected_buckets"], [1, 3])
            self.assertEqual(result["valid_candidates"], 4)

    def test_empty_and_fewer_than_k_results(self):
        with tempfile.TemporaryDirectory() as temporary:
            index = self.build(Path(temporary) / "fixed")
            empty = index.search(
                self.vectors[0],
                self.timestamps,
                self.vectors,
                [[1.1, 1.2]],
                5,
                FixedHNSWSearchConfig(32, 2),
            )
            self.assertEqual(len(empty["ids"]), 0)
            partial = index.search(
                self.vectors[0],
                self.timestamps,
                self.vectors,
                [[0.0, 0.25]],
                5,
                FixedHNSWSearchConfig(32, 2),
            )
            np.testing.assert_array_equal(partial["ids"], [0])


if __name__ == "__main__":
    unittest.main()
