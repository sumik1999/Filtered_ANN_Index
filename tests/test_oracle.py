import unittest

import numpy as np

from temporal_ann.oracle import build_workload, exact_filtered_search


class OracleTests(unittest.TestCase):
    def test_workload_has_four_fragments_and_full_timeline_coverage(self):
        workload = build_workload(1000, 10_000, seed=7)
        self.assertEqual(len({item["query_id"] for item in workload}), 1000)

        covered = np.zeros(40, dtype=bool)
        for item in workload:
            self.assertEqual(item["fragment_count"], 4)
            self.assertEqual(len(item["intervals"]), 4)
            width = sum(end - start for start, end in item["intervals"])
            self.assertAlmostEqual(width, 0.10)
            for cell in range(40):
                midpoint = (cell + 0.5) / 40
                if any(start <= midpoint < end for start, end in item["intervals"]):
                    covered[cell] = True
        self.assertTrue(covered.all())

    def test_exact_filtered_search(self):
        database = np.array([[0.0], [3.0], [1.0], [2.0]], dtype=np.float32)
        query = np.array([1.2], dtype=np.float32)
        ids, distances = exact_filtered_search(
            database, query, np.array([0, 1, 2], dtype=np.int64), max_k=2, block_size=2
        )
        np.testing.assert_array_equal(ids, [2, 0])
        np.testing.assert_allclose(distances, [0.04, 1.44], rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
