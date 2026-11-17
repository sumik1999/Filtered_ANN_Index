from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import numpy as np

from .exact_benchmark import percentile
from .oracle import exact_filtered_search

DATASET_ID = "ExponentialScience/DLT-Tweets"
DATASET_REVISION = "c7d979394488bbd94bd6f493af1775d6b6159176"
YEAR = 2021
YEAR_START_MS = int(datetime(YEAR, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
YEAR_END_MS = int(datetime(YEAR + 1, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def download(args: argparse.Namespace) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise RuntimeError("Install Twitter dependencies with: pip install -e '.[twitter]'") from error

    args.output.mkdir(parents=True, exist_ok=True)
    location = snapshot_download(
        repo_id=DATASET_ID,
        repo_type="dataset",
        revision=args.revision,
        local_dir=args.output,
        allow_patterns=["data/*.parquet", "README.md"],
    )
    print(f"Downloaded {DATASET_ID}@{args.revision} to {location}")
    return Path(location)


def traffic_summary(hourly_counts: np.ndarray) -> dict[str, Any]:
    values = np.asarray(hourly_counts, dtype=np.float64)
    if values.ndim != 1 or not len(values) or np.any(values < 0):
        raise ValueError("hourly_counts must be a non-empty non-negative vector")
    mean = float(values.mean())
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    ordered = np.sort(values)[::-1]
    busiest = max(1, int(np.ceil(0.01 * len(values))))
    return {
        "hours": len(values),
        "mean_per_hour": mean,
        "median_per_hour": median,
        "maximum_per_hour": int(values.max()),
        "coefficient_of_variation": float(values.std() / mean) if mean else 0.0,
        "fano_factor": float(values.var() / mean) if mean else 0.0,
        "peak_to_median": float(values.max() / median) if median else None,
        "busiest_one_percent_share": float(ordered[:busiest].sum() / values.sum()) if values.sum() else 0.0,
        "burst_threshold_median_plus_3mad": median + 3.0 * mad,
    }


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    shards = sorted(args.input.glob("data/*.parquet"))
    if not shards:
        raise FileNotFoundError(f"No Parquet shards found under {args.input / 'data'}")
    if args.max_records <= 0 or args.query_count <= 0:
        raise ValueError("max-records and query-count must be positive")
    if args.query_count >= args.max_records:
        raise ValueError("query-count must be smaller than max-records")

    args.output.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect()
    connection.execute(f"SET threads={args.threads}")
    sources = ", ".join("?" for _ in shards)
    paths = [str(path) for path in shards]
    started = time.perf_counter()

    # The source is already English and deduplicated. The explicit checks make
    # that contract visible and remove conventional retweets and blank posts.
    connection.execute(
        f"""
        CREATE TEMP TABLE candidates AS
        SELECT
            md5(tweet || '|' || CAST(timestamp AS VARCHAR)) AS source_key,
            tweet AS text,
            epoch_ms(CAST(timestamp AS TIMESTAMP)) AS timestamp_ms
        FROM read_parquet([{sources}], union_by_name=true)
        WHERE CAST(year AS INTEGER) = ?
          AND lower(language) = 'en'
          AND length(trim(tweet)) > 0
          AND NOT regexp_matches(trim(tweet), '^RT\\s+@', 'i')
        """,
        [*paths, args.year],
    )
    available = connection.execute("SELECT count(*) FROM candidates").fetchone()[0]
    take = min(args.max_records, available)
    connection.execute(
        "CREATE TEMP TABLE sampled AS SELECT * FROM candidates ORDER BY source_key LIMIT ?",
        [take],
    )
    query_count = min(args.query_count, max(0, take - 1))
    database_count = take - query_count
    if database_count <= 0:
        raise ValueError("not enough source records after filtering")

    query_path = args.output / "query_records.parquet"
    database_path = args.output / "database_records.parquet"
    connection.execute(
        f"""
        COPY (
            SELECT row_number() OVER (ORDER BY source_key) - 1 AS query_id,
                   source_key, text, timestamp_ms
            FROM sampled
            ORDER BY source_key
            LIMIT {query_count}
        ) TO ? (FORMAT PARQUET, COMPRESSION ZSTD)
        """,
        [str(query_path)],
    )
    connection.execute(
        f"""
        COPY (
            SELECT row_number() OVER (ORDER BY source_key) - 1 AS vector_id,
                   source_key, text, timestamp_ms
            FROM (SELECT * FROM sampled ORDER BY source_key OFFSET {query_count})
        ) TO ? (FORMAT PARQUET, COMPRESSION ZSTD)
        """,
        [str(database_path)],
    )

    hourly = connection.execute(
        """
        WITH hours AS (
            SELECT range AS hour_start
            FROM range(?, ?, 3600000)
        ), counts AS (
            SELECT (timestamp_ms - ?) // 3600000 AS hour_number, count(*) AS n
            FROM sampled GROUP BY hour_number
        )
        SELECT coalesce(counts.n, 0)
        FROM hours LEFT JOIN counts
          ON counts.hour_number = (hours.hour_start - ?) // 3600000
        ORDER BY hours.hour_start
        """,
        [YEAR_START_MS, YEAR_END_MS, YEAR_START_MS, YEAR_START_MS],
    ).fetchnumpy()["COALESCE(counts.n, 0)"]
    summary = traffic_summary(hourly)
    metadata = {
        "schema_version": 1,
        "source": {"dataset": DATASET_ID, "revision": args.revision},
        "year": args.year,
        "range": {"start_inclusive": "2021-01-01T00:00:00Z", "end_exclusive": "2022-01-01T00:00:00Z"},
        "selection": {
            "available_after_filtering": available,
            "selected": take,
            "database_records": database_count,
            "query_records": query_count,
            "method": "lowest deterministic MD5 keys; uniform record sample preserving relative traffic density",
            "language": "en",
            "retweets": "records beginning with RT @ removed; source is deduplicated",
        },
        "traffic": summary,
        "construction_seconds": time.perf_counter() - started,
    }
    _write_json(args.output / "metadata.json", metadata)
    connection.close()
    print(f"Prepared {database_count:,} database and {query_count:,} query tweets in {args.output}")
    return metadata


def _completed_embedding_rows(matrix: np.ndarray, chunk_size: int = 100_000) -> int:
    """Find the first unwritten row in a normalized, resumable embedding matrix."""
    for start in range(0, len(matrix), chunk_size):
        norms = np.linalg.norm(matrix[start : start + chunk_size], axis=1)
        incomplete = np.flatnonzero(norms < 0.5)
        if len(incomplete):
            return start + int(incomplete[0])
    return len(matrix)


def _encode_parquet(path: Path, id_column: str, output: Path, model: Any, batch_size: int) -> tuple[int, int]:
    connection = duckdb.connect()
    count = connection.execute("SELECT count(*) FROM read_parquet(?)", [str(path)]).fetchone()[0]
    dimensions = int(model.get_embedding_dimension())
    if output.exists():
        matrix = np.load(output, mmap_mode="r+")
        if matrix.shape != (count, dimensions) or matrix.dtype != np.float32:
            raise ValueError(f"existing {output} has an incompatible shape or dtype")
        written = _completed_embedding_rows(matrix)
        print(f"{output.name}: resuming at {written:,}/{count:,}", flush=True)
    else:
        matrix = np.lib.format.open_memmap(
            output, mode="w+", dtype=np.float32, shape=(count, dimensions)
        )
        written = 0

    cursor = connection.execute(
        f"SELECT {id_column}, text FROM read_parquet(?) WHERE {id_column} >= ? ORDER BY {id_column}",
        [str(path), written],
    )
    next_report = ((written // 100_000) + 1) * 100_000
    while True:
        rows = cursor.fetchmany(batch_size)
        if not rows:
            break
        expected = list(range(written, written + len(rows)))
        if [int(row[0]) for row in rows] != expected:
            raise ValueError(f"{id_column} values are not dense and ordered")
        embeddings = model.encode(
            [row[1] for row in rows],
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32, copy=False)
        if embeddings.shape[1] != dimensions:
            raise ValueError("model returned an unexpected embedding dimension")
        matrix[written : written + len(rows)] = embeddings
        written += len(rows)
        if written >= next_report or written == count:
            matrix.flush()
            print(f"{output.name}: {written:,}/{count:,}", flush=True)
            next_report += 100_000
    connection.close()
    if written != count:
        raise RuntimeError(f"encoded {written} of {count} records")
    matrix.flush()
    del matrix
    return count, dimensions


def embed(args: argparse.Namespace) -> dict[str, Any]:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise RuntimeError("Install Twitter dependencies with: pip install -e '.[twitter]'") from error
    args.output.mkdir(parents=True, exist_ok=True)
    model = SentenceTransformer(args.model, device=args.device)
    started = time.perf_counter()
    database_count, dimensions = _encode_parquet(
        args.input / "database_records.parquet", "vector_id",
        args.output / "database_vectors.npy", model, args.batch_size,
    )
    query_count, query_dimensions = _encode_parquet(
        args.input / "query_records.parquet", "query_id",
        args.output / "query_vectors.npy", model, args.batch_size,
    )
    if dimensions != query_dimensions:
        raise RuntimeError("database and query dimensions differ")
    metadata = {
        "schema_version": 1,
        "model": args.model,
        "normalization": "L2",
        "distance": "squared L2 (equivalent ordering to cosine for normalized vectors)",
        "dtype": "float32",
        "dimensions": dimensions,
        "database_records": database_count,
        "query_records": query_count,
        "device": str(model.device),
        "construction_seconds": time.perf_counter() - started,
    }
    _write_json(args.output / "embedding_metadata.json", metadata)
    return metadata


def build_timestamp_database(records: Path, database: Path, rebuild: bool) -> tuple[duckdb.DuckDBPyConnection, float]:
    if rebuild and database.exists():
        database.unlink()
    database.parent.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect(str(database))
    tables = {row[0] for row in connection.execute("SHOW TABLES").fetchall()}
    if "tweets" in tables:
        return connection, 0.0
    started = time.perf_counter()
    connection.execute(
        "CREATE TABLE tweets AS SELECT vector_id::INTEGER vector_id, timestamp_ms::BIGINT timestamp_ms FROM read_parquet(?)",
        [str(records)],
    )
    connection.execute("CREATE INDEX idx_tweets_timestamp_ms ON tweets(timestamp_ms)")
    connection.execute("CHECKPOINT")
    return connection, time.perf_counter() - started


def benchmark(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    vectors = np.load(args.embeddings / "database_vectors.npy", mmap_mode="r")
    queries = np.load(args.embeddings / "query_vectors.npy", mmap_mode="r")
    connection, build_seconds = build_timestamp_database(
        args.input / "database_records.parquet", args.database, args.rebuild_db
    )
    connection.execute(f"SET threads={args.threads}")
    query_records = connection.execute(
        "SELECT query_id, timestamp_ms FROM read_parquet(?) ORDER BY query_id LIMIT ?",
        [str(args.input / "query_records.parquet"), min(args.query_count, len(queries))],
    ).fetchall()
    durations = [int(value) for value in args.durations_hours.split(",") if value.strip()]
    if not durations or any(value <= 0 for value in durations):
        raise ValueError("durations-hours must contain positive integers")

    hourly_counts = connection.execute(
        "SELECT timestamp_ms // 3600000 AS hour_number, count(*) AS n "
        "FROM tweets GROUP BY hour_number"
    ).fetchall()
    count_values = np.asarray([row[1] for row in hourly_counts], dtype=np.float64)
    median = float(np.median(count_values))
    mad = float(np.median(np.abs(count_values - median)))
    by_hour = {int(hour): int(count) for hour, count in hourly_counts}

    rows: list[dict[str, Any]] = []
    ground_truth_ids: list[np.ndarray] = []
    ground_truth_distances: list[np.ndarray] = []
    sql = "SELECT vector_id FROM tweets WHERE timestamp_ms >= ? AND timestamp_ms < ? ORDER BY vector_id"
    for position, (query_id, timestamp_ms) in enumerate(query_records):
        hour_count = by_hour.get(int(timestamp_ms) // 3600000, 0)
        traffic_class = "burst" if hour_count > median + 3 * mad else ("quiet" if hour_count <= median else "ordinary")
        for duration_hours in durations:
            half = duration_hours * 3_600_000 // 2
            start_ms = max(YEAR_START_MS, int(timestamp_ms) - half)
            end_ms = min(YEAR_END_MS, start_ms + duration_hours * 3_600_000)
            start_ms = max(YEAR_START_MS, end_ms - duration_hours * 3_600_000)
            total_started = time.perf_counter_ns()
            filter_started = time.perf_counter_ns()
            eligible = connection.execute(sql, [start_ms, end_ms]).fetchnumpy()["vector_id"].astype(np.int64, copy=False)
            filter_ms = (time.perf_counter_ns() - filter_started) / 1e6
            exact_started = time.perf_counter_ns()
            ids, distances = exact_filtered_search(
                vectors, queries[query_id], eligible, args.k, args.block_size
            )
            exact_ms = (time.perf_counter_ns() - exact_started) / 1e6
            padded_ids = np.full(args.k, -1, dtype=np.int64)
            padded_distances = np.full(args.k, np.inf, dtype=np.float32)
            padded_ids[: len(ids)] = ids
            padded_distances[: len(distances)] = distances
            ground_truth_ids.append(padded_ids)
            ground_truth_distances.append(padded_distances)
            rows.append({
                "measurement_id": len(rows),
                "query_id": query_id, "traffic_class": traffic_class,
                "query_hour_count": hour_count, "duration_hours": duration_hours,
                "start_ms": start_ms, "end_ms": end_ms, "k": args.k,
                "eligible_count": len(eligible), "actual_selectivity": len(eligible) / len(vectors),
                "result_count": len(ids), "timestamp_filter_ms": filter_ms,
                "exact_l2_ms": exact_ms,
                "total_latency_ms": (time.perf_counter_ns() - total_started) / 1e6,
                "candidates_examined": len(eligible),
            })
        if args.progress_every and (position + 1) % args.progress_every == 0:
            print(f"exact Twitter queries: {position + 1}/{len(query_records)}", flush=True)
    connection.close()
    summary = {
        "method": "DuckDB temporal pre-filter plus exact blocked squared L2",
        "dataset": DATASET_ID, "year": YEAR, "database_records": len(vectors),
        "queries": len(query_records), "measurements": len(rows), "k": args.k,
        "timestamp_database_build_seconds": build_seconds,
        "groups": [],
    }
    for duration in durations:
        for traffic_class in ("burst", "ordinary", "quiet"):
            members = [x for x in rows if x["duration_hours"] == duration and x["traffic_class"] == traffic_class]
            if members:
                summary["groups"].append({
                    "duration_hours": duration, "traffic_class": traffic_class, "queries": len(members),
                    "mean_selectivity": float(np.mean([x["actual_selectivity"] for x in members])),
                    "mean_candidates_examined": float(np.mean([x["candidates_examined"] for x in members])),
                    "mean_filter_ms": float(np.mean([x["timestamp_filter_ms"] for x in members])),
                    "mean_exact_l2_ms": float(np.mean([x["exact_l2_ms"] for x in members])),
                    "mean_total_latency_ms": float(np.mean([x["total_latency_ms"] for x in members])),
                    "p50_total_latency_ms": percentile([x["total_latency_ms"] for x in members], 50),
                    "p95_total_latency_ms": percentile([x["total_latency_ms"] for x in members], 95),
                })
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "queries.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    _write_json(args.output / "summary.json", summary)
    np.savez_compressed(
        args.output / "ground_truth.npz",
        neighbor_ids=np.stack(ground_truth_ids),
        distances=np.stack(ground_truth_distances),
        query_ids=np.asarray([row["query_id"] for row in rows], dtype=np.int64),
        duration_hours=np.asarray([row["duration_hours"] for row in rows], dtype=np.int32),
        start_ms=np.asarray([row["start_ms"] for row in rows], dtype=np.int64),
        end_ms=np.asarray([row["end_ms"] for row in rows], dtype=np.int64),
        eligible_count=np.asarray([row["eligible_count"] for row in rows], dtype=np.int64),
    )
    dataset_metadata = json.loads((args.input / "metadata.json").read_text(encoding="utf-8"))
    embedding_metadata = json.loads(
        (args.embeddings / "embedding_metadata.json").read_text(encoding="utf-8")
    )
    configuration = {
        "schema_version": 1,
        "dataset": dataset_metadata,
        "embeddings": embedding_metadata,
        "baseline": {
            "name": "exact hard temporal pre-filter",
            "temporal_semantics": "hard eligibility only; timestamps do not contribute to ranking distance",
            "predicate": "timestamp_ms >= start_ms AND timestamp_ms < end_ms",
            "predicate_interval": "half-open [start_ms, end_ms)",
            "timestamp_index": "DuckDB ART index on timestamp_ms",
            "ranking": "exact squared L2 over every temporally eligible MiniLM vector",
            "distance_formula": "sum_j((query[j] - candidate[j]) ** 2), j=1..384",
            "approximation_used": False,
            "expected_recall_at_k": 1.0,
            "k": args.k,
            "distance_block_size": args.block_size,
            "durations_hours": durations,
            "query_count": len(query_records),
            "threads": args.threads,
            "traffic_classes": {
                "burst": "query hour count > median + 3*MAD",
                "ordinary": "median < query hour count <= median + 3*MAD",
                "quiet": "query hour count <= median",
            },
        },
    }
    _write_json(args.output / "configuration_report.json", configuration)
    return rows, summary


def download_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download the DLT-Tweets Parquet shards")
    parser.add_argument("--output", type=Path, default=Path("Twitter_data/raw/dlt-tweets"))
    parser.add_argument("--revision", default=DATASET_REVISION)
    return parser


def prepare_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare a deterministic 2021 DLT-Tweets sample")
    parser.add_argument("--input", type=Path, default=Path("Twitter_data/raw/dlt-tweets"))
    parser.add_argument("--output", type=Path, default=Path("Twitter_data/processed"))
    parser.add_argument("--year", type=int, default=YEAR, choices=[YEAR])
    parser.add_argument("--revision", default=DATASET_REVISION)
    parser.add_argument("--max-records", type=int, default=1_000_000)
    parser.add_argument("--query-count", type=int, default=10_000)
    parser.add_argument("--threads", type=int, default=4)
    return parser


def embed_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Encode prepared tweets with MiniLM")
    parser.add_argument("--input", type=Path, default=Path("Twitter_data/processed"))
    parser.add_argument("--output", type=Path, default=Path("Twitter_data/processed"))
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    return parser


def benchmark_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark exact temporal pre-filtering on DLT-Tweets")
    parser.add_argument("--input", type=Path, default=Path("Twitter_data/processed"))
    parser.add_argument("--embeddings", type=Path, default=Path("Twitter_data/processed"))
    parser.add_argument("--database", type=Path, default=Path("Twitter_data/sql/timestamps.duckdb"))
    parser.add_argument("--output", type=Path, default=Path("results/twitter_exact"))
    parser.add_argument("--query-count", type=int, default=1_000)
    parser.add_argument("--durations-hours", default="1,6,24,168,720")
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--block-size", type=int, default=65_536)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--rebuild-db", action="store_true")
    return parser


def download_main() -> None: download(download_parser().parse_args())
def prepare_main() -> None: prepare(prepare_parser().parse_args())
def embed_main() -> None: embed(embed_parser().parse_args())
def benchmark_main() -> None:
    args = benchmark_parser().parse_args(); rows, _ = benchmark(args)
    print(f"Wrote {len(rows):,} measurements to {args.output}")
