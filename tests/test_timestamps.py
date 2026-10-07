import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np

from temporal_ann.timestamps import generate


class TimestampGenerationTests(unittest.TestCase):
    def test_generates_reproducible_float64_arrays_in_normalized_range(self):
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            common = {
                "count": 1000,
                "seed": 11,
                "year": 2023,
                "bins": 32,
                "zipf_exponent": 1.2,
            }
            generate(Namespace(**common, output=Path(first)))
            generate(Namespace(**common, output=Path(second)))

            for name in ("uniform", "poisson_clustered", "zipf"):
                left = np.load(Path(first) / f"{name}_seed-11.npy")
                right = np.load(Path(second) / f"{name}_seed-11.npy")
                self.assertEqual(left.dtype, np.float64)
                self.assertEqual(left.shape, (1000,))
                self.assertTrue(np.all((left >= 0.0) & (left < 1.0)))
                np.testing.assert_array_equal(left, right)


if __name__ == "__main__":
    unittest.main()
