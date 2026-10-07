import unittest

import faiss
import numpy as np

from temporal_ann.hnsw_post_filter import candidate_mask, search_post_filtered


class HnswPostFilterTests(unittest.TestCase):
    def test_candidate_mask_supports_interval_union(self):
        timestamps = np.array([0.1, 0.3, 0.6, 0.9])
        ids = np.array([3, 2, -1, 0], dtype=np.int64)
        mask = candidate_mask(ids, timestamps, [[0.0, 0.2], [0.8, 1.0]])
        np.testing.assert_array_equal(mask, [True, False, False, True])

    def test_search_filters_global_hnsw_results(self):
        vectors = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32)
        timestamps = np.array([0.1, 0.3, 0.6, 0.9])
        index = faiss.IndexHNSWFlat(1, 2, faiss.METRIC_L2)
        index.add(vectors)
        result = search_post_filtered(
            index,
            np.array([1.8], dtype=np.float32),
            timestamps,
            [[0.0, 0.4], [0.8, 1.0]],
            k=2,
            overfetch=2,
            ef_search=16,
        )
        np.testing.assert_array_equal(result["ids"], [1, 3])
        self.assertEqual(result["valid_candidates"], 3)


if __name__ == "__main__":
    unittest.main()
