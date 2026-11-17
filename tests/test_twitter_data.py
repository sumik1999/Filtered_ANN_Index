import tempfile
import unittest
from pathlib import Path

import duckdb
import numpy as np

from temporal_ann.twitter_data import (
    _completed_embedding_rows,
    build_timestamp_database,
    traffic_summary,
)


class TwitterDataTests(unittest.TestCase):
    def test_traffic_summary_reports_bursts(self):
        summary = traffic_summary(np.array([1, 1, 1, 1, 20], dtype=np.int64))
        self.assertEqual(summary["maximum_per_hour"], 20)
        self.assertEqual(summary["median_per_hour"], 1.0)
        self.assertEqual(summary["peak_to_median"], 20.0)
        self.assertGreater(summary["fano_factor"], 1.0)

    def test_resumable_embeddings_find_first_unwritten_row(self):
        matrix = np.zeros((5, 3), dtype=np.float32)
        matrix[:3, 0] = 1.0
        self.assertEqual(_completed_embedding_rows(matrix, chunk_size=2), 3)
        matrix[3:, 0] = 1.0
        self.assertEqual(_completed_embedding_rows(matrix, chunk_size=2), 5)

    def test_timestamp_database_supports_half_open_ranges(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = root / "records.parquet"
            source = duckdb.connect()
            source.execute(
                "COPY (SELECT * FROM (VALUES (0, 1000), (1, 2000), (2, 3000)) "
                "AS t(vector_id, timestamp_ms)) TO ? (FORMAT PARQUET)",
                [str(records)],
            )
            source.close()
            connection, _ = build_timestamp_database(records, root / "timestamps.duckdb", False)
            ids = connection.execute(
                "SELECT vector_id FROM tweets WHERE timestamp_ms >= ? AND timestamp_ms < ? ORDER BY vector_id",
                [1000, 3000],
            ).fetchnumpy()["vector_id"]
            connection.close()
            np.testing.assert_array_equal(ids, [0, 1])


if __name__ == "__main__":
    unittest.main()
