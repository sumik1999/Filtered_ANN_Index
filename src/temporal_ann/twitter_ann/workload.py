from __future__ import annotations

import argparse
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import duckdb
import faiss
import numpy as np

from ..twitter_data import YEAR_END_MS, YEAR_START_MS
from .bucketed import resolve_intervals

HOUR_MS = 3_600_000
TRAFFIC_CLASSES = ("burst", "ordinary", "quiet")
DENSITY_CLASSES = ("dense", "medium_dense", "medium_sparse", "sparse")
ALIGNMENTS = ("aligned", "independent")


def split_coverage_ms(total_hours: int, predicate_count: int) -> list[int]:
    if total_hours <= 0 or predicate_count not in (1, 2, 3, 4):
        raise ValueError("coverage must be positive and predicate_count must be 1..4")
    total = total_hours * HOUR_MS
    base, remainder = divmod(total, predicate_count)
    return [base + (1 if position < remainder else 0) for position in range(predicate_count)]


def centered_interval(center_ms: int, width_ms: int) -> tuple[int, int]:
    if width_ms <= 0 or width_ms > YEAR_END_MS - YEAR_START_MS:
        raise ValueError("interval width is outside the dataset timeline")
    start = center_ms - width_ms // 2
    end = start + width_ms
    if start < YEAR_START_MS:
        start, end = YEAR_START_MS, YEAR_START_MS + width_ms
    if end > YEAR_END_MS:
        start, end = YEAR_END_MS - width_ms, YEAR_END_MS
    return int(start), int(end)


def intervals_are_valid(
    intervals: list[tuple[int, int]], expected_width_ms: int, operator: str = "OR"
) -> bool:
    if not intervals or any(start >= end for start, end in intervals):
        return False
    if intervals != sorted(intervals):
        return False
    if intervals[0][0] < YEAR_START_MS or max(end for _, end in intervals) > YEAR_END_MS:
        return False
    normalized_operator = operator.upper()
    if normalized_operator == "OR":
        if any(left[1] >= right[0] for left, right in zip(intervals, intervals[1:])):
            return False
        return sum(end - start for start, end in intervals) == expected_width_ms
    if normalized_operator == "AND":
        effective = resolve_intervals(intervals, "AND")
        return bool(effective) and effective[0][1] - effective[0][0] == expected_width_ms
    return False


def contains(intervals: list[tuple[int, int]], timestamp_ms: int) -> bool:
    return any(start <= timestamp_ms < end for start, end in intervals)


def _and_intervals(base: tuple[int, int], predicate_count: int) -> list[tuple[int, int]]:
    """Create overlapping predicates whose intersection is exactly ``base``."""
    if predicate_count == 1:
        return [base]
    start, end = base
    width = end - start
    extension = max(HOUR_MS, width // max(2, predicate_count))
    intervals = [
        (start, min(YEAR_END_MS, end + extension)),
        (max(YEAR_START_MS, start - extension), end),
    ]
    for position in range(2, predicate_count):
        factor = position
        intervals.append((
            max(YEAR_START_MS, start - factor * extension),
            min(YEAR_END_MS, end + (predicate_count - position) * extension),
        ))
    return sorted(intervals)


def classify_hours(timestamps: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, float]]:
    hour_count = (YEAR_END_MS - YEAR_START_MS) // HOUR_MS
    hour_ids = ((np.asarray(timestamps, dtype=np.int64) - YEAR_START_MS) // HOUR_MS)
    if np.any(hour_ids < 0) or np.any(hour_ids >= hour_count):
        raise ValueError("timestamps fall outside 2021")
    counts = np.bincount(hour_ids, minlength=hour_count)
    median = float(np.median(counts))
    mad = float(np.median(np.abs(counts - median)))
    threshold = median + 3.0 * mad
    pools = {
        "burst": np.flatnonzero(counts > threshold),
        "ordinary": np.flatnonzero((counts > median) & (counts <= threshold)),
        "quiet": np.flatnonzero(counts <= median),
    }
    if any(not len(pool) for pool in pools.values()):
        raise ValueError("at least one traffic class has no anchor hours")
    return counts, pools, {"median": median, "mad": mad, "burst_threshold": threshold}


def _hour_class(hour_id: int, counts: np.ndarray, stats: dict[str, float]) -> str:
    count = int(counts[hour_id])
    if count > stats["burst_threshold"]:
        return "burst"
    if count > stats["median"]:
        return "ordinary"
    return "quiet"


def generate_intervals(
    query_timestamp_ms: int,
    predicate_count: int,
    total_hours: int,
    traffic_class: str,
    alignment: str,
    anchor_pools: dict[str, np.ndarray],
    rng: np.random.Generator,
    operator: str = "OR",
) -> list[tuple[int, int]]:
    """Generate an OR union or an overlapping AND predicate.

    ``total_hours`` is the effective union/intersection width, making OR and
    AND records comparable after their Boolean predicate is resolved.
    """
    if traffic_class not in TRAFFIC_CLASSES or alignment not in ALIGNMENTS:
        raise ValueError("unknown traffic class or alignment")
    operator = operator.upper()
    if operator not in {"OR", "AND"}:
        raise ValueError("operator must be OR or AND")
    if operator == "AND":
        base = generate_intervals(
            query_timestamp_ms, 1, total_hours, traffic_class, alignment,
            anchor_pools, rng, "OR",
        )[0]
        intervals = _and_intervals(base, predicate_count)
        if not intervals_are_valid(intervals, total_hours * HOUR_MS, "AND"):
            raise RuntimeError("generated invalid AND intervals")
        return intervals

    widths = split_coverage_ms(total_hours, predicate_count)
    intervals: list[tuple[int, int]] = []
    remaining = list(widths)
    if alignment == "aligned":
        intervals.append(centered_interval(query_timestamp_ms, remaining.pop(0)))

    anchors = np.asarray(anchor_pools[traffic_class])
    # More than enough deterministic attempts even for four 180-hour fragments.
    attempts = rng.permutation(anchors)
    cursor = 0
    while remaining and cursor < len(attempts):
        hour_id = int(attempts[cursor]); cursor += 1
        center = YEAR_START_MS + hour_id * HOUR_MS + HOUR_MS // 2
        candidate = centered_interval(center, remaining[0])
        proposed = sorted([*intervals, candidate])
        if any(left[1] >= right[0] for left, right in zip(proposed, proposed[1:])):
            continue
        if alignment == "independent" and contains(proposed, query_timestamp_ms):
            continue
        intervals, remaining = proposed, remaining[1:]
    if remaining:
        raise RuntimeError("could not place disjoint temporal predicates")
    expected = total_hours * HOUR_MS
    if not intervals_are_valid(intervals, expected):
        raise RuntimeError("generated invalid intervals")
    if (alignment == "aligned") != contains(intervals, query_timestamp_ms):
        raise RuntimeError("generated intervals violate alignment")
    return intervals


def spatial_metadata(
    database_vectors: np.ndarray,
    query_vectors: np.ndarray,
    index: faiss.Index,
    *,
    neighbors: int,
    ef_search: int,
    cluster_count: int,
    kmeans_sample_size: int,
    kmeans_iterations: int,
    seed: int,
    threads: int,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    if neighbors < 8 or neighbors > index.ntotal:
        raise ValueError("neighbors must be between 8 and the database size")
    if cluster_count <= 1 or cluster_count > len(query_vectors):
        raise ValueError("invalid semantic cluster count")
    faiss.omp_set_num_threads(threads)
    params = faiss.SearchParametersHNSW(efSearch=max(ef_search, neighbors))
    distances, _ = index.search(
        np.ascontiguousarray(query_vectors), neighbors, params=params
    )
    local_density = np.median(distances[:, 7:neighbors], axis=1).astype(np.float32)
    order = np.argsort(local_density, kind="stable")
    density_quartile = np.empty(len(query_vectors), dtype=np.int8)
    for quartile, ids in enumerate(np.array_split(order, 4)):
        density_quartile[ids] = quartile

    rng = np.random.default_rng(seed)
    sample_size = min(kmeans_sample_size, len(database_vectors))
    sample_ids = rng.choice(len(database_vectors), sample_size, replace=False)
    training = np.ascontiguousarray(database_vectors[sample_ids], dtype=np.float32)
    kmeans = faiss.Kmeans(
        database_vectors.shape[1], cluster_count, niter=kmeans_iterations,
        seed=seed, verbose=False, gpu=False,
    )
    kmeans.train(training)
    centroid_index = faiss.IndexFlatL2(database_vectors.shape[1])
    centroid_index.add(kmeans.centroids)
    centroid_distances, cluster_ids = centroid_index.search(
        np.ascontiguousarray(query_vectors), 1
    )
    metadata = {
        "local_density_distance": local_density,
        "density_quartile": density_quartile,
        "semantic_cluster_id": cluster_ids[:, 0].astype(np.int32),
        "centroid_distance": centroid_distances[:, 0].astype(np.float32),
    }
    return metadata, kmeans.centroids.astype(np.float32)


class BalancedQuerySelector:
    def __init__(
        self,
        density_quartiles: np.ndarray,
        cluster_ids: np.ndarray,
        native_classes: np.ndarray,
        seed: int,
    ):
        self.density = density_quartiles
        self.clusters = cluster_ids
        self.native_classes = native_classes
        self.usage = np.zeros(len(cluster_ids), dtype=np.int32)
        self.cluster_usage: Counter[int] = Counter()
        rng = np.random.default_rng(seed)
        self.priority = np.empty(len(cluster_ids), dtype=np.int64)
        permutation = rng.permutation(len(cluster_ids))
        self.priority[permutation] = np.arange(len(cluster_ids))

    def select_many(
        self, density: int, traffic_class: str, alignment: str, count: int
    ) -> list[int]:
        mask = self.density == density
        if alignment == "aligned":
            mask &= self.native_classes == traffic_class
        candidates = np.flatnonzero(mask)
        if not len(candidates):
            raise ValueError(
                f"no query vectors for density={density}, traffic={traffic_class}, "
                f"alignment={alignment}"
            )
        cluster_use = Counter()
        chosen: list[int] = []
        for _ in range(count):
            query_id = int(
                min(
                    candidates,
                    key=lambda value: (
                        self.usage[value],
                        cluster_use[int(self.clusters[value])],
                        self.cluster_usage[int(self.clusters[value])],
                        self.priority[value],
                    ),
                )
            )
            cluster = int(self.clusters[query_id])
            chosen.append(query_id)
            self.usage[query_id] += 1
            cluster_use[cluster] += 1
            self.cluster_usage[cluster] += 1
        return chosen


def _load_query_records(path: Path) -> tuple[np.ndarray, list[str]]:
    connection = duckdb.connect()
    rows = connection.execute(
        "SELECT query_id, timestamp_ms, source_key FROM read_parquet(?) ORDER BY query_id",
        [str(path)],
    ).fetchall()
    connection.close()
    if [int(row[0]) for row in rows] != list(range(len(rows))):
        raise ValueError("query IDs must be dense and ordered")
    return (
        np.asarray([row[1] for row in rows], dtype=np.int64),
        [str(row[2]) for row in rows],
    )


def _eligible_count(sorted_timestamps: np.ndarray, intervals: list[tuple[int, int]]) -> int:
    return sum(
        int(np.searchsorted(sorted_timestamps, end, side="left") -
            np.searchsorted(sorted_timestamps, start, side="left"))
        for start, end in intervals
    )


def generate(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = np.random.default_rng(args.seed)
    database_vectors = np.load(args.embeddings / "database_vectors.npy", mmap_mode="r")
    query_vectors = np.load(args.embeddings / "query_vectors.npy", mmap_mode="r")
    timestamps = np.load(args.timestamps, mmap_mode="r")
    query_timestamps, source_keys = _load_query_records(
        args.records / "query_records.parquet"
    )
    if len(database_vectors) != len(timestamps) or len(query_vectors) != len(query_timestamps):
        raise ValueError("record and embedding counts do not match")
    index = faiss.read_index(str(args.index))
    if index.ntotal != len(database_vectors) or index.d != database_vectors.shape[1]:
        raise ValueError("HNSW index does not match database embeddings")

    counts, anchor_pools, traffic_stats = classify_hours(timestamps)
    query_hours = ((query_timestamps - YEAR_START_MS) // HOUR_MS).astype(np.int64)
    native_classes = np.asarray([
        _hour_class(int(hour), counts, traffic_stats) for hour in query_hours
    ])
    spatial, centroids = spatial_metadata(
        database_vectors, query_vectors, index,
        neighbors=args.neighbors, ef_search=args.spatial_ef_search,
        cluster_count=args.clusters, kmeans_sample_size=args.kmeans_sample_size,
        kmeans_iterations=args.kmeans_iterations, seed=args.seed, threads=args.threads,
    )

    selector = BalancedQuerySelector(
        spatial["density_quartile"], spatial["semantic_cluster_id"],
        native_classes, args.seed,
    )
    durations = [int(value) for value in args.coverage_hours.split(",") if value.strip()]
    if not durations or any(value <= 0 or value > 8760 for value in durations):
        raise ValueError("coverage-hours must contain valid positive durations")
    operators = [value.strip().upper() for value in args.operators.split(",") if value.strip()]
    if not operators or len(set(operators)) != len(operators) or any(
        value not in {"OR", "AND"} for value in operators
    ):
        raise ValueError("operators must be a unique comma-separated subset of OR,AND")
    if args.records_per_cell <= 0 or args.k <= 0:
        raise ValueError("records-per-cell and k must be positive")
    sorted_timestamps = np.sort(np.asarray(timestamps))
    records: list[dict[str, Any]] = []
    cell_counts: Counter[tuple] = Counter()
    factors = itertools.product(
        operators, range(1, 5), durations, TRAFFIC_CLASSES, range(4), ALIGNMENTS
    )
    for operator, predicate_count, total_hours, traffic_class, density, alignment in factors:
        selected = selector.select_many(
            density, traffic_class, alignment, args.records_per_cell
        )
        for query_id in selected:
            intervals = generate_intervals(
                int(query_timestamps[query_id]), predicate_count, total_hours,
                traffic_class, alignment, anchor_pools, rng, operator,
            )
            effective_intervals = resolve_intervals(intervals, operator)
            eligible = _eligible_count(sorted_timestamps, effective_intervals)
            cell = (operator, predicate_count, total_hours, traffic_class, density, alignment)
            cell_counts[cell] += 1
            records.append({
                "schema_version": 2,
                "workload_id": -1,
                "query_id": query_id,
                "query_source_key": source_keys[query_id],
                "query_timestamp_ms": int(query_timestamps[query_id]),
                "query_reuse_number": int(selector.usage[query_id]),
                "k": args.k,
                "metric": "squared_l2",
                "predicate_operator": operator,
                "predicate_count": predicate_count,
                "requested_total_coverage_hours": total_hours,
                "intervals": [[start, end] for start, end in intervals],
                "effective_intervals": [
                    [start, end] for start, end in effective_intervals
                ],
                "anchor_traffic_class": traffic_class,
                "time_vector_alignment": alignment,
                "native_timestamp_eligible": contains(
                    effective_intervals, int(query_timestamps[query_id])
                ),
                "vector_density_class": DENSITY_CLASSES[density],
                "vector_density_quartile": density,
                "local_density_distance": float(spatial["local_density_distance"][query_id]),
                "semantic_cluster_id": int(spatial["semantic_cluster_id"][query_id]),
                "centroid_distance": float(spatial["centroid_distance"][query_id]),
                "eligible_count": eligible,
                "actual_selectivity": eligible / len(database_vectors),
                "has_at_least_k_eligible": eligible >= args.k,
                "seed": args.seed,
            })

    permutation = rng.permutation(len(records))
    records = [records[int(position)] for position in permutation]
    reuse = Counter(record["query_id"] for record in records)
    for workload_id, record in enumerate(records):
        record["workload_id"] = workload_id
        # Final reuse count is more useful than assignment-time count.
        record["query_reuse_number"] = reuse[record["query_id"]]

    expected_cells = (
        len(operators) * 4 * len(durations) * len(TRAFFIC_CLASSES) * 4 * len(ALIGNMENTS)
    )
    if len(cell_counts) != expected_cells or set(cell_counts.values()) != {args.records_per_cell}:
        raise RuntimeError("factorial workload is not balanced")
    summary = {
        "schema_version": 2,
        "records": len(records),
        "factorial_cells": expected_cells,
        "records_per_cell": args.records_per_cell,
        "predicate_operators": dict(Counter(x["predicate_operator"] for x in records)),
        "predicate_counts": dict(Counter(x["predicate_count"] for x in records)),
        "coverage_hours": dict(Counter(x["requested_total_coverage_hours"] for x in records)),
        "traffic_classes": dict(Counter(x["anchor_traffic_class"] for x in records)),
        "vector_density_classes": dict(Counter(x["vector_density_class"] for x in records)),
        "alignments": dict(Counter(x["time_vector_alignment"] for x in records)),
        "semantic_clusters_represented": len(set(x["semantic_cluster_id"] for x in records)),
        "unique_query_vectors": len(reuse),
        "maximum_query_reuse": max(reuse.values()),
        "queries_with_fewer_than_k_eligible": sum(not x["has_at_least_k_eligible"] for x in records),
        "selectivity": {
            "minimum": min(x["actual_selectivity"] for x in records),
            "mean": float(np.mean([x["actual_selectivity"] for x in records])),
            "maximum": max(x["actual_selectivity"] for x in records),
        },
        "traffic_thresholds": traffic_stats,
    }

    for path in (
        args.output, args.summary, args.spatial_metadata, args.centroids, args.config
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n")
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    np.savez_compressed(
        args.spatial_metadata,
        local_density_distance=spatial["local_density_distance"],
        density_quartile=spatial["density_quartile"],
        semantic_cluster_id=spatial["semantic_cluster_id"],
        centroid_distance=spatial["centroid_distance"],
        native_traffic_class=native_classes,
    )
    np.save(args.centroids, centroids, allow_pickle=False)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    args.config.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return records, summary


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate the balanced varied Twitter query workload")
    parser.add_argument("--records", type=Path, default=Path("Twitter_data/processed"))
    parser.add_argument("--embeddings", type=Path, default=Path("Twitter_data/processed"))
    parser.add_argument("--timestamps", type=Path, default=Path("Twitter_data/ann/timestamps_ms.npy"))
    parser.add_argument("--index", type=Path, default=Path("Twitter_data/ann/hnsw_m32_efc200.index"))
    parser.add_argument("--output", type=Path, default=Path("Twitter_data/workloads/temporal_vector_queries_9600.jsonl"))
    parser.add_argument("--summary", type=Path, default=Path("Twitter_data/workloads/temporal_vector_queries_9600_summary.json"))
    parser.add_argument("--spatial-metadata", type=Path, default=Path("Twitter_data/workloads/query_spatial_metadata.npz"))
    parser.add_argument("--centroids", type=Path, default=Path("Twitter_data/workloads/semantic_centroids.npy"))
    parser.add_argument("--config", type=Path, default=Path("Twitter_data/workloads/generation_config.json"))
    parser.add_argument("--coverage-hours", default="1,6,24,168,720")
    parser.add_argument("--operators", default="OR,AND",
                        help="comma-separated Boolean temporal operators")
    parser.add_argument("--records-per-cell", type=int, default=10,
                        help="records per operator-aware factorial cell")
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--neighbors", type=int, default=32)
    parser.add_argument("--spatial-ef-search", type=int, default=512)
    parser.add_argument("--clusters", type=int, default=64)
    parser.add_argument("--kmeans-sample-size", type=int, default=100_000)
    parser.add_argument("--kmeans-iterations", type=int, default=20)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    return parser


def main() -> None:
    args = parser().parse_args()
    records, summary = generate(args)
    print(f"Wrote {len(records):,} balanced queries to {args.output}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
