import json
import tempfile
import unittest
from pathlib import Path

import faiss
import numpy as np

from temporal_ann.twitter_ann.hnsw import build_index, search_post_filtered


class TwitterHNSWTests(unittest.TestCase):
    def test_build_persists_index_and_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            vectors = np.eye(4, dtype=np.float32)
            np.save(root / "vectors.npy", vectors)
            index, metadata = build_index(
                root / "vectors.npy", root / "ann" / "index.faiss",
                root / "ann" / "metadata.json", m=2, ef_construction=20,
                threads=1, batch_size=2,
            )
            self.assertEqual(index.ntotal, 4)
            self.assertEqual(metadata["vector_count"], 4)
            self.assertTrue((root / "ann" / "index.faiss").exists())
            persisted = json.loads((root / "ann" / "metadata.json").read_text())
            self.assertEqual(persisted["dimensions"], 4)

    def test_post_filter_uses_half_open_timestamp_window(self):
        vectors = np.array([[0.0, 0.0], [0.1, 0.0], [0.2, 0.0], [2.0, 2.0]], dtype=np.float32)
        index = faiss.IndexHNSWFlat(2, 2, faiss.METRIC_L2)
        index.hnsw.efConstruction = 20
        index.add(vectors)
        result = search_post_filtered(
            index, np.array([0.0, 0.0], dtype=np.float32),
            np.array([1000, 2000, 3000, 4000], dtype=np.int64),
            2000, 3000, k=2, overfetch=2, ef_search=20,
        )
        np.testing.assert_array_equal(result["ids"], [1])
        self.assertEqual(result["valid_candidates"], 1)


if __name__ == "__main__":
    unittest.main()
