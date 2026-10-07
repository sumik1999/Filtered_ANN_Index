import unittest

import faiss
import numpy as np

from temporal_ann.hnsw_idselector import eligibility_mask, search_with_selector


class HnswIdSelectorTests(unittest.TestCase):
    def test_eligibility_mask_supports_interval_union(self):
        timestamps = np.array([0.1, 0.3, 0.6, 0.9])
        mask = eligibility_mask(timestamps, [[0.0, 0.2], [0.8, 1.0]])
        np.testing.assert_array_equal(mask, [True, False, False, True])

    def test_selector_only_returns_eligible_ids(self):
        vectors = np.arange(10, dtype=np.float32).reshape(-1, 1)
        timestamps = np.arange(10, dtype=np.float64) / 10
        index = faiss.IndexHNSWFlat(1, 4, faiss.METRIC_L2)
        index.add(vectors)
        result = search_with_selector(
            index,
            np.array([5.2], dtype=np.float32),
            timestamps,
            [[0.0, 0.3], [0.8, 1.0]],
            k=4,
            ef_search=32,
        )
        self.assertTrue(set(result["ids"].tolist()).issubset({0, 1, 2, 8, 9}))
        self.assertEqual(result["eligible_count"], 5)
        self.assertEqual(result["bitmap_bytes"], 2)


if __name__ == "__main__":
    unittest.main()
