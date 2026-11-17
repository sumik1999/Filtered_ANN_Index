from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import duckdb
import faiss
import numpy as np

from ..exact_benchmark import percentile


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_ints(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values or any(item <= 0 for item in values):
        raise ValueError("values must be comma-separated positive integers")
    return values


def load_timestamps(records: Path, cache: Path) -> np.ndarray:
    """Load dense vector-ID-ordered timestamps, creating a reusable NumPy cache."""
    if cache.exists():
        values = np.load(cache, mmap_mode="r")
        if values.ndim != 1 or values.dtype != np.int64:
            raise ValueError(f"invalid timestamp cache: {cache}")
        return values
    cache.parent.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect()
    rows = connection.execute(
        "SELECT vector_id, timestamp_ms FROM read_parquet(?) ORDER BY vector_id",
        [str(records)],
    ).fetchnumpy()
    connection.close()
    ids = rows["vector_id"].astype(np.int64, copy=False)
    if not np.array_equal(ids, np.arange(len(ids), dtype=np.int64)):
        raise ValueError("database vector IDs must be dense and zero-based")
    values = rows["timestamp_ms"].astype(np.int64, copy=False)
    np.save(cache, values, allow_pickle=False)
    return np.load(cache, mmap_mode="r")


def validate_index(index: faiss.Index, vectors: np.ndarray) -> None:
    if index.d != vectors.shape[1] or index.ntotal != len(vectors):
        raise ValueError("HNSW index does not match the embedding matrix")
    if index.metric_type != faiss.METRIC_L2:
        raise ValueError("HNSW index must use squared L2")


def build_index(
    vectors_path: Path,
    output: Path,
    metadata_path: Path,
    *,
    m: int = 32,
    ef_construction: int = 200,
    threads: int = 4,
    batch_size: int = 50_000,
    rebuild: bool = False,
) -> tuple[faiss.Index, dict[str, Any]]:
    """Build a global HNSW index without copying the full embedding matrix."""
    if min(m, ef_construction, threads, batch_size) <= 0:
        raise ValueError("HNSW parameters, threads, and batch size must be positive")
    vectors = np.load(vectors_path, mmap_mode="r")
    if vectors.ndim != 2 or vectors.dtype != np.float32 or not len(vectors):
        raise ValueError("embeddings must be a non-empty float32 matrix")
    if output.exists() and not rebuild:
        index = faiss.read_index(str(output))
        validate_index(index, vectors)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        return index, metadata

    faiss.omp_set_num_threads(threads)
    index = faiss.IndexHNSWFlat(vectors.shape[1], m, faiss.METRIC_L2)
    index.hnsw.efConstruction = ef_construction
    started = time.perf_counter()
    for start in range(0, len(vectors), batch_size):
        end = min(start + batch_size, len(vectors))
        index.add(np.ascontiguousarray(vectors[start:end]))
        print(f"Twitter HNSW: {end:,}/{len(vectors):,}", flush=True)
    construction_seconds = time.perf_counter() - started
    validate_index(index, vectors)
    output.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(output))
    metadata = {
        "schema_version": 1,
        "backend": "FAISS IndexHNSWFlat",
        "metric": "squared L2",
        "source": str(vectors_path),
        "vector_count": len(vectors),
        "dimensions": vectors.shape[1],
        "dtype": "float32",
        "id_mapping": "FAISS label equals database vector_id",
        "M": m,
        "efConstruction": ef_construction,
        "threads": threads,
        "construction_seconds": construction_seconds,
        "index_file": output.name,
        "index_size_bytes": output.stat().st_size,
        "index_sha256": _sha256(output),
    }
    _write_json(metadata_path, metadata)
    return index, metadata


def search_post_filtered(
    index: faiss.Index,
    query: np.ndarray,
    timestamps: np.ndarray,
    start_ms: int,
    end_ms: int,
    k: int,
    overfetch: int,
    ef_search: int,
) -> dict[str, Any]:
    """Search globally, then enforce the half-open timestamp predicate."""
    if k <= 0 or overfetch <= 0 or ef_search <= 0 or start_ms >= end_ms:
        raise ValueError("invalid search parameters")
    query = np.asarray(query, dtype=np.float32)
    if query.shape != (index.d,):
        raise ValueError(f"query must have shape ({index.d},)")
    fetch = min(index.ntotal, k * overfetch)
    effective_ef = max(ef_search, fetch)
    params = faiss.SearchParametersHNSW(efSearch=effective_ef)
    faiss.cvar.hnsw_stats.reset()
    started = time.perf_counter_ns()
    distances, ids = index.search(query.reshape(1, -1), fetch, params=params)
    search_ms = (time.perf_counter_ns() - started) / 1e6
    distance_computations = int(faiss.cvar.hnsw_stats.ndis)

    started = time.perf_counter_ns()
    ids, distances = ids[0], distances[0]
    valid_id = ids >= 0
    mask = np.zeros(len(ids), dtype=bool)
    mask[valid_id] = ((timestamps[ids[valid_id]] >= start_ms) &
                      (timestamps[ids[valid_id]] < end_ms))
    result_ids = ids[mask][:k].astype(np.int64, copy=False)
    result_distances = distances[mask][:k].astype(np.float32, copy=False)
    filter_ms = (time.perf_counter_ns() - started) / 1e6
    return {
        "ids": result_ids,
        "distances": result_distances,
        "candidates_requested": fetch,
        "valid_candidates": int(mask.sum()),
        "effective_ef_search": effective_ef,
        "distance_computations": distance_computations,
        "search_ms": search_ms,
        "filter_ms": filter_ms,
        "total_ms": search_ms + filter_ms,
    }


def _recall(found: np.ndarray, truth: np.ndarray, k: int) -> float:
    expected = truth[truth >= 0][:k]
    if not len(expected):
        return 1.0 if not len(found) else 0.0
    return len(set(map(int, found)) & set(map(int, expected))) / len(expected)


def benchmark(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    vectors = np.load(args.embeddings / "database_vectors.npy", mmap_mode="r")
    queries = np.load(args.embeddings / "query_vectors.npy", mmap_mode="r")
    timestamps = load_timestamps(
        args.input / "database_records.parquet", args.timestamp_cache
    )
    if len(vectors) != len(timestamps):
        raise ValueError("embeddings and timestamps have different row counts")
    index, metadata = build_index(
        args.embeddings / "database_vectors.npy", args.index, args.index_metadata,
        m=args.m, ef_construction=args.ef_construction, threads=args.threads,
        batch_size=args.build_batch_size, rebuild=args.rebuild_index,
    )
    truth = np.load(args.ground_truth)
    required = {"neighbor_ids", "query_ids", "duration_hours", "start_ms", "end_ms"}
    if not required.issubset(truth.files):
        raise ValueError("ground truth is incompatible; rerun benchmark-twitter-exact")
    with args.exact_queries.open(newline="", encoding="utf-8") as handle:
        exact_rows = list(csv.DictReader(handle))
    if len(exact_rows) != len(truth["query_ids"]):
        raise ValueError("exact query rows and ground truth have different lengths")
    for position, row in enumerate(exact_rows):
        if int(row["measurement_id"]) != position:
            raise ValueError("exact query measurements must be dense and ordered")
    ef_values, overfetch_values = parse_ints(args.ef_search), parse_ints(args.overfetch)
    faiss.omp_set_num_threads(args.threads)
    rows: list[dict[str, Any]] = []
    measurement_count = len(truth["query_ids"])
    for ef_search in ef_values:
        for overfetch in overfetch_values:
            for position in range(measurement_count):
                query_id = int(truth["query_ids"][position])
                result = search_post_filtered(
                    index, queries[query_id], timestamps,
                    int(truth["start_ms"][position]), int(truth["end_ms"][position]),
                    args.k, overfetch, ef_search,
                )
                exact_row = exact_rows[position]
                rows.append({
                    "measurement_id": position, "query_id": query_id,
                    "traffic_class": exact_row["traffic_class"],
                    "query_hour_count": int(exact_row["query_hour_count"]),
                    "duration_hours": int(truth["duration_hours"][position]),
                    "eligible_count": int(exact_row["eligible_count"]),
                    "actual_selectivity": float(exact_row["actual_selectivity"]),
                    "k": args.k,
                    "ef_search": ef_search,
                    "effective_ef_search": result["effective_ef_search"],
                    "overfetch": overfetch,
                    "candidates_requested": result["candidates_requested"],
                    "valid_candidates": result["valid_candidates"],
                    "distance_computations": result["distance_computations"],
                    "result_count": len(result["ids"]),
                    "search_ms": result["search_ms"], "filter_ms": result["filter_ms"],
                    "total_latency_ms": result["total_ms"],
                    "recall_at_k": _recall(result["ids"], truth["neighbor_ids"][position], args.k),
                })
                if args.progress_every and (position + 1) % args.progress_every == 0:
                    print(f"Twitter HNSW ef={ef_search} overfetch={overfetch}: "
                          f"{position + 1}/{measurement_count}", flush=True)

    grouped: dict[tuple[int, str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["duration_hours"], row["traffic_class"], row["ef_search"], row["overfetch"])].append(row)
    groups = []
    for (duration, traffic_class, ef_search, overfetch), members in sorted(grouped.items()):
        latencies = [x["total_latency_ms"] for x in members]
        groups.append({
            "duration_hours": duration, "traffic_class": traffic_class,
            "ef_search": ef_search, "overfetch": overfetch,
            "queries": len(members),
            "mean_selectivity": float(np.mean([x["actual_selectivity"] for x in members])),
            "mean_recall_at_k": float(np.mean([x["recall_at_k"] for x in members])),
            "complete_result_rate": float(np.mean([x["result_count"] == args.k for x in members])),
            "mean_valid_candidates": float(np.mean([x["valid_candidates"] for x in members])),
            "mean_total_latency_ms": float(np.mean(latencies)),
            "p50_total_latency_ms": percentile(latencies, 50),
            "p95_total_latency_ms": percentile(latencies, 95),
        })
    summary = {
        "method": "global FAISS HNSW followed by exact timestamp post-filter",
        "temporal_predicate": "half-open [start_ms, end_ms)",
        "query_measurements": measurement_count,
        "index": metadata, "groups": groups,
    }
    return rows, summary


def write_results(output: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    if not rows:
        raise ValueError("no benchmark rows were produced")
    output.mkdir(parents=True, exist_ok=True)
    with (output / "queries.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    _write_json(output / "summary.json", summary)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build global HNSW for DLT-Tweets")
    parser.add_argument("--embeddings", type=Path, default=Path("Twitter_data/processed"))
    parser.add_argument("--index", type=Path, default=Path("Twitter_data/ann/hnsw_m32_efc200.index"))
    parser.add_argument("--metadata", type=Path, default=Path("Twitter_data/ann/hnsw_metadata.json"))
    parser.add_argument("--m", type=int, default=32)
    parser.add_argument("--ef-construction", type=int, default=200)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=50_000)
    parser.add_argument("--rebuild", action="store_true")
    return parser


def benchmark_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark Twitter HNSW post-filtering")
    parser.add_argument("--input", type=Path, default=Path("Twitter_data/processed"))
    parser.add_argument("--embeddings", type=Path, default=Path("Twitter_data/processed"))
    parser.add_argument("--timestamp-cache", type=Path, default=Path("Twitter_data/ann/timestamps_ms.npy"))
    parser.add_argument("--index", type=Path, default=Path("Twitter_data/ann/hnsw_m32_efc200.index"))
    parser.add_argument("--index-metadata", type=Path, default=Path("Twitter_data/ann/hnsw_metadata.json"))
    parser.add_argument("--ground-truth", type=Path, default=Path("results/twitter_exact/ground_truth.npz"))
    parser.add_argument("--exact-queries", type=Path, default=Path("results/twitter_exact/queries.csv"))
    parser.add_argument("--output", type=Path, default=Path("results/twitter_hnsw_post_filter"))
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--ef-search", default="256")
    parser.add_argument("--overfetch", default="10,50,100")
    parser.add_argument("--m", type=int, default=32)
    parser.add_argument("--ef-construction", type=int, default=200)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--build-batch-size", type=int, default=50_000)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--rebuild-index", action="store_true")
    return parser


def build_main() -> None:
    args = build_parser().parse_args()
    _, metadata = build_index(
        args.embeddings / "database_vectors.npy", args.index, args.metadata,
        m=args.m, ef_construction=args.ef_construction, threads=args.threads,
        batch_size=args.batch_size, rebuild=args.rebuild,
    )
    print(json.dumps(metadata, indent=2))


def benchmark_main() -> None:
    args = benchmark_parser().parse_args()
    rows, summary = benchmark(args)
    write_results(args.output, rows, summary)
    print(f"Wrote {len(rows):,} Twitter HNSW measurements to {args.output}")
