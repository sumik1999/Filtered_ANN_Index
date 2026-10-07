import unittest

import faiss
import numpy as np

from temporal_ann.faiss_hnsw import validate_index


class FaissHnswTests(unittest.TestCase):
    def test_hnsw_uses_sequential_ids_and_squared_l2(self):
        vectors = np.array([[0.0, 0.0], [2.0, 0.0], [1.0, 0.0]], dtype=np.float32)
        index = faiss.IndexHNSWFlat(2, 2, faiss.METRIC_L2)
        index.add(vectors)
        validate_index(index, dimensions=2, expected_count=3)

        distances, ids = index.search(np.array([[0.9, 0.0]], dtype=np.float32), 3)
        np.testing.assert_array_equal(ids[0], [2, 0, 1])
        np.testing.assert_allclose(distances[0], [0.01, 0.81, 1.21], rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
