from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import duckdb
import h5py
import numpy as np

from .oracle import exact_filtered_search


DISTRIBUTIONS = ("uniform", "poisson_clustered", "zipf")
FILTER_SQL = """
SELECT vector_id
FROM sift_timestamps
WHERE (timestamp_{distribution} >= ? AND timestamp_{distribution} < ?)
   OR (timestamp_{distribution} >= ? AND timestamp_{distribution} < ?)
   OR (timestamp_{distribution} >= ? AND timestamp_{distribution} < ?)
   OR (timestamp_{distribution} >= ? AND timestamp_{distribution} < ?)
ORDER BY vector_id
"""


def load_workload(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def sql_parameters(record: dict) -> list[float]:
    if len(record["intervals"]) != 4:
        raise ValueError("exact benchmark expects exactly four intervals")
    return [value for interval in record["intervals"] for value in interval]


def create_database(
    database_path: Path, timestamps_path: Path, rebuild: bool
) -> tuple[duckdb.DuckDBPyConnection, float, bool]:
    if rebuild and database_path.exists():
        database_path.unlink()
    database_path.parent.mkdir(parents=True, exist_ok=True)
    existed = database_path.exists()
    connection = duckdb.connect(str(database_path))
    if existed:
        count = connection.execute("SELECT count(*) FROM sift_timestamps").fetchone()[0]
        if count != 1_000_000:
            raise ValueError(f"SQL timestamp database has unexpected row count: {count}")
        return connection, 0.0, False

    started = time.perf_counter()
    columns: dict[str, np.ndarray] = {
        "vector_id": np.arange(1_000_000, dtype=np.int32)
    }
    timestamp_metadata = json.loads(
        (timestamps_path / "metadata.json").read_text(encoding="utf-8")
    )
    for name in DISTRIBUTIONS:
        filename = timestamp_metadata["distributions"][name]["file"]
        values = np.load(timestamps_path / filename)
        if values.shape != (1_000_000,) or values.dtype != np.float64:
            raise ValueError(f"invalid timestamps for {name}")
        columns[f"timestamp_{name}"] = values

    connection.register("timestamp_source", columns)
    connection.execute(
        """
        CREATE TABLE sift_timestamps AS
        SELECT vector_id, timestamp_uniform, timestamp_poisson_clustered, timestamp_zipf
        FROM timestamp_source
        """
    )
    connection.unregister("timestamp_source")
    for name in DISTRIBUTIONS:
        connection.execute(
            f"CREATE INDEX idx_timestamp_{name} "
            f"ON sift_timestamps(timestamp_{name})"
        )
    connection.execute("CHECKPOINT")
    return connection, time.perf_counter() - started, True


def recall(returned: np.ndarray, expected: np.ndarray, k: int) -> float:
    returned_set = set(returned[:k].tolist())
    expected_ids = expected[:k]
    expected_set = set(expected_ids[expected_ids >= 0].tolist())
    if not expected_set:
        return 1.0 if not returned_set else 0.0
    return len(returned_set & expected_set) / len(expected_set)


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def run(args: argparse.Namespace) -> tuple[list[dict], dict]:
    workload = load_workload(args.workload)
    if not workload:
        raise ValueError("workload is empty")

    connection, build_seconds, database_created = create_database(
        args.database, args.timestamps, args.rebuild_db
    )
    connection.execute(f"SET threads = {args.threads}")

    with h5py.File(args.sift_file, "r") as sift:
        database_vectors = np.asarray(sift["train"], dtype=np.float32)
        all_queries = np.asarray(sift["test"], dtype=np.float32)
    query_vectors = all_queries[[record["query_id"] for record in workload]]

    rows: list[dict] = []
    for distribution in DISTRIBUTIONS:
        ground_truth = np.load(
            args.ground_truth / f"{distribution}_ground_truth.npz"
        )
        expected_query_ids = np.asarray(
            [record["query_id"] for record in workload], dtype=np.int32
        )
        if not np.array_equal(ground_truth["query_ids"], expected_query_ids):
            raise ValueError(f"query IDs do not match {distribution} ground truth")

        sql = FILTER_SQL.format(distribution=distribution)
        # Warm the SQL path without including it in measured query latency.
        connection.execute(sql, sql_parameters(workload[0])).fetchall()

        for index, (record, query) in enumerate(zip(workload, query_vectors)):
            total_started = time.perf_counter_ns()
            sql_started = time.perf_counter_ns()
            result = connection.execute(sql, sql_parameters(record)).fetchnumpy()
            eligible_ids = result["vector_id"].astype(np.int64, copy=False)
            sql_ms = (time.perf_counter_ns() - sql_started) / 1e6

            expected_count = int(ground_truth["qualifying_counts"][index])
            if len(eligible_ids) != expected_count:
                raise RuntimeError(
                    f"SQL filter mismatch for {distribution} workload {index}: "
                    f"{len(eligible_ids)} != {expected_count}"
                )

            vector_started = time.perf_counter_ns()
            ids, distances = exact_filtered_search(
                database_vectors,
                query,
                eligible_ids,
                args.max_k,
                args.block_size,
            )
            vector_ms = (time.perf_counter_ns() - vector_started) / 1e6
            total_ms = (time.perf_counter_ns() - total_started) / 1e6

            expected_ids = ground_truth["neighbor_ids"][index]
            expected_distances = ground_truth["squared_l2_distances"][index]
            if not np.array_equal(ids, expected_ids[: len(ids)]):
                raise RuntimeError(
                    f"exact IDs differ from oracle for {distribution} workload {index}"
                )
            np.testing.assert_array_equal(distances, expected_distances[: len(ids)])

            rows.append(
                {
                    "distribution": distribution,
                    "workload_id": record["workload_id"],
                    "query_id": record["query_id"],
                    "total_width": record["total_width"],
                    "fragments": record["fragment_count"],
                    "qualifying_count": len(eligible_ids),
                    "actual_selectivity": len(eligible_ids) / len(database_vectors),
                    "sql_filter_ms": sql_ms,
                    "exact_l2_ms": vector_ms,
                    "total_latency_ms": total_ms,
                    "recall_at_1": recall(ids, expected_ids, 1),
                    "recall_at_10": recall(ids, expected_ids, 10),
                    "recall_at_50": recall(ids, expected_ids, 50),
                    "recall_at_100": recall(ids, expected_ids, 100),
                    "candidates_examined": len(eligible_ids),
                }
            )
            if (index + 1) % args.progress_every == 0 or index + 1 == len(workload):
                print(
                    f"{distribution}: {index + 1}/{len(workload)} benchmark queries",
                    flush=True,
                )

    connection.execute("CHECKPOINT")
    connection.close()
    summary: dict = {
        "method": "SQL temporal pre-filter plus exact blocked L2 scan",
        "ann_index_used": False,
        "sql_engine": f"DuckDB {duckdb.__version__}",
        "database_file": str(args.database),
        "database_size_bytes": args.database.stat().st_size,
        "database_created_this_run": database_created,
        "database_build_seconds": build_seconds,
        "query_count_per_distribution": len(workload),
        "max_k": args.max_k,
        "distributions": {},
    }
    for distribution in DISTRIBUTIONS:
        members = [row for row in rows if row["distribution"] == distribution]
        summary["distributions"][distribution] = {
            "queries": len(members),
            "mean_selectivity": float(np.mean([x["actual_selectivity"] for x in members])),
            "mean_candidates_examined": float(
                np.mean([x["candidates_examined"] for x in members])
            ),
            "mean_sql_filter_ms": float(np.mean([x["sql_filter_ms"] for x in members])),
            "mean_exact_l2_ms": float(np.mean([x["exact_l2_ms"] for x in members])),
            "mean_total_latency_ms": float(
                np.mean([x["total_latency_ms"] for x in members])
            ),
            "p50_total_latency_ms": percentile(
                [x["total_latency_ms"] for x in members], 50
            ),
            "p95_total_latency_ms": percentile(
                [x["total_latency_ms"] for x in members], 95
            ),
            "mean_recall_at_100": float(
                np.mean([x["recall_at_100"] for x in members])
            ),
        }
    return rows, summary


def write_results(output: Path, rows: list[dict], summary: dict) -> None:
    output.mkdir(parents=True, exist_ok=True)
    with (output / "queries.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark SQL temporal filtering followed by exact SIFT L2 search"
    )
    parser.add_argument(
        "--sift-file", type=Path, default=Path("SIFT_data/sift-128-euclidean.hdf5")
    )
    parser.add_argument(
        "--timestamps", type=Path, default=Path("SIFT_data/timestamps")
    )
    parser.add_argument(
        "--workload", type=Path,
        default=Path("SIFT_data/workloads/temporal_queries.jsonl"),
    )
    parser.add_argument(
        "--ground-truth", type=Path, default=Path("SIFT_data/ground_truth")
    )
    parser.add_argument(
        "--database", type=Path, default=Path("SIFT_data/sql/sift_timestamps.duckdb")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results/sift_exact")
    )
    parser.add_argument("--max-k", type=int, default=100)
    parser.add_argument("--block-size", type=int, default=65_536)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--rebuild-db", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rows, summary = run(args)
    write_results(args.output, rows, summary)
    print(f"Wrote {len(rows)} exact benchmark measurements to {args.output}")


if __name__ == "__main__":
    main()
