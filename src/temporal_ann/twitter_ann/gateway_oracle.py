from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from ..exact_benchmark import percentile
from .rolling_benchmark import load_workload
from .rolling_hnsw import RollingSearchConfig, RollingTemporalHNSW


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def candidate_pool(vector_count: int, step: int) -> np.ndarray:
    if vector_count <= 0 or step <= 0:
        raise ValueError("vector_count and candidate step must be positive")
    values = np.arange(step - 1, vector_count, step, dtype=np.int64)
    if not len(values):
        raise ValueError("candidate step exceeds vector count")
    return values


def select_random(pool: np.ndarray, count: int, seed: int) -> np.ndarray:
    if count > len(pool):
        raise ValueError("gateway count exceeds candidate pool")
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(pool, count, replace=False)).astype(np.int64)


def select_temporal_density(
    pool: np.ndarray,
    timestamps: np.ndarray,
    count: int,
) -> np.ndarray:
    """Select one high-traffic representative from each population block."""
    if count > len(pool):
        raise ValueError("gateway count exceeds candidate pool")
    hours, hour_counts = np.unique(
        np.asarray(timestamps, dtype=np.int64) // 3_600_000,
        return_counts=True,
    )
    density = {int(hour): int(value) for hour, value in zip(hours, hour_counts)}
    selected = []
    for block in np.array_split(pool, count):
        middle = float(np.mean(block))
        chosen = max(
            map(int, block),
            key=lambda local_id: (
                density[int(timestamps[local_id]) // 3_600_000],
                -abs(local_id - middle),
            ),
        )
        selected.append(chosen)
    return np.asarray(sorted(selected), dtype=np.int64)


def select_semantic_clusters(
    pool: np.ndarray,
    index: RollingTemporalHNSW,
    count: int,
    seed: int,
    threads: int,
) -> np.ndarray:
    if count > len(pool):
        raise ValueError("gateway count exceeds candidate pool")
    database_ids = index.global_ids[pool]
    candidate_values = np.ascontiguousarray(
        index.vectors[database_ids], dtype=np.float32
    )
    training_count = min(10_000, len(index.global_ids))
    training_local_ids = np.linspace(
        0, len(index.global_ids) - 1, training_count, dtype=np.int64
    )
    training_values = np.ascontiguousarray(
        index.vectors[index.global_ids[training_local_ids]], dtype=np.float32
    )
    faiss.omp_set_num_threads(threads)
    kmeans = faiss.Kmeans(
        training_values.shape[1], count, niter=25, nredo=1,
        verbose=False, seed=seed, spherical=False,
    )
    kmeans.train(training_values)
    candidate_index = faiss.IndexFlatL2(candidate_values.shape[1])
    candidate_index.add(candidate_values)
    fetch = min(32, len(pool))
    _, nearest = candidate_index.search(
        np.ascontiguousarray(kmeans.centroids, dtype=np.float32), fetch
    )
    chosen: list[int] = []
    used: set[int] = set()
    for row in nearest:
        for position in row:
            candidate = int(pool[int(position)])
            if candidate not in used:
                chosen.append(candidate)
                used.add(candidate)
                break
    if len(chosen) < count:
        for candidate in pool:
            value = int(candidate)
            if value not in used:
                chosen.append(value)
                used.add(value)
                if len(chosen) == count:
                    break
    return np.asarray(sorted(chosen), dtype=np.int64)


def gateway_utility_matrix(
    pool: np.ndarray,
    index: RollingTemporalHNSW,
    queries: np.ndarray,
    workload: list[dict[str, Any]],
    positions: list[int],
) -> np.ndarray:
    candidate_timestamps = np.asarray(index.timestamps[pool], dtype=np.int64)
    query_ids = np.asarray([workload[position]["query_id"] for position in positions])
    query_values = np.ascontiguousarray(queries[query_ids], dtype=np.float32)
    candidate_values = np.ascontiguousarray(
        index.vectors[index.global_ids[pool]], dtype=np.float32
    )
    query_norms = np.einsum("ij,ij->i", query_values, query_values)[:, None]
    candidate_norms = np.einsum(
        "ij,ij->i", candidate_values, candidate_values
    )[None, :]
    distances = np.maximum(
        0.0, query_norms + candidate_norms - 2.0 * query_values @ candidate_values.T
    )
    eligible = np.zeros(distances.shape, dtype=bool)
    for row, position in enumerate(positions):
        for start, end in workload[position]["intervals"]:
            eligible[row] |= (
                (candidate_timestamps >= start) & (candidate_timestamps < end)
            )
    eligible_distances = distances[eligible]
    temperature = float(np.median(eligible_distances)) if len(eligible_distances) else 1.0
    temperature = max(temperature, 1e-6)
    # The constant coverage term rewards placing at least one entry in every range.
    return np.where(eligible, 0.05 + np.exp(-distances / temperature), 0.0).astype(
        np.float32
    )


def select_greedy_oracle(
    pool: np.ndarray,
    utility: np.ndarray,
    count: int,
) -> np.ndarray:
    if utility.ndim != 2 or utility.shape[1] != len(pool):
        raise ValueError("utility matrix does not match candidate pool")
    if count > len(pool):
        raise ValueError("gateway count exceeds candidate pool")
    current = np.zeros(utility.shape[0], dtype=np.float32)
    available = np.ones(len(pool), dtype=bool)
    selected: list[int] = []
    for _ in range(count):
        gains = np.maximum(utility - current[:, None], 0.0).sum(axis=0)
        gains[~available] = -1.0
        position = int(np.argmax(gains))
        selected.append(int(pool[position]))
        available[position] = False
        current = np.maximum(current, utility[:, position])
    return np.asarray(sorted(selected), dtype=np.int64)


def _quality(found: np.ndarray, expected: np.ndarray) -> tuple[float, float]:
    truth = expected[expected >= 0]
    found_set, truth_set = set(map(int, found)), set(map(int, truth))
    overlap = len(found_set & truth_set)
    recall = overlap / len(truth_set) if truth_set else float(not found_set)
    precision = overlap / len(found_set) if found_set else float(not truth_set)
    return recall, precision


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = [row["total_latency_ms"] for row in rows]
    return {
        "queries": len(rows),
        "mean_recall_at_k": float(np.mean([row["recall_at_k"] for row in rows])),
        "mean_precision_at_k": float(np.mean([row["precision_at_k"] for row in rows])),
        "mean_total_latency_ms": float(np.mean(latencies)),
        "p50_total_latency_ms": percentile(latencies, 50),
        "p95_total_latency_ms": percentile(latencies, 95),
        "mean_distance_computations": float(np.mean([
            row["distance_computations"] for row in rows
        ])),
        "mean_nodes_visited": float(np.mean([row["nodes_visited"] for row in rows])),
        "mean_gateway_candidates": float(np.mean([
            row["gateway_candidates"] for row in rows
        ])),
        "mean_gateway_seeds": float(np.mean([row["gateway_seeds"] for row in rows])),
    }


def run(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    vectors = np.load(args.vectors, mmap_mode="r")
    queries = np.load(args.queries, mmap_mode="r")
    index = RollingTemporalHNSW(args.index, vectors)
    workload = load_workload(args.workload)
    truth = np.load(args.ground_truth)
    pool = candidate_pool(len(index.global_ids), args.candidate_step)
    gateway_count = args.gateway_count or len(index.gateway_ids)
    graph_positions = [
        position for position, record in enumerate(workload)
        if record["eligible_count"] > args.exact_threshold
    ]
    training_positions = graph_positions[::2]
    evaluation_positions = graph_positions[1::2]
    if not training_positions or not evaluation_positions:
        raise ValueError("workload does not contain enough graph-search queries")

    periodic = np.asarray(index.gateway_ids, dtype=np.int64)
    if len(periodic) != gateway_count:
        raise ValueError("requested gateway count differs from periodic baseline")
    utility = gateway_utility_matrix(
        pool, index, queries, workload, training_positions
    )
    gateway_sets = {
        "periodic": periodic,
        "random": select_random(pool, gateway_count, args.seed),
        "semantic_cluster": select_semantic_clusters(
            pool, index, gateway_count, args.seed, args.threads
        ),
        "temporal_density": select_temporal_density(
            pool, index.timestamps, gateway_count
        ),
        "greedy_oracle": select_greedy_oracle(pool, utility, gateway_count),
    }
    for name, values in gateway_sets.items():
        if len(values) != gateway_count or len(np.unique(values)) != gateway_count:
            raise ValueError(f"{name} did not produce an equal-budget gateway set")
        if np.any(np.diff(values) <= 0):
            raise ValueError(f"{name} gateway IDs are not ordered")

    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output / "gateway_sets.npz",
        **gateway_sets,
        candidate_pool=pool,
        training_workload_ids=np.asarray(training_positions, dtype=np.int64),
        evaluation_workload_ids=np.asarray(evaluation_positions, dtype=np.int64),
    )
    search_config = RollingSearchConfig(
        k=100,
        ef_search=args.ef_search,
        exact_threshold=args.exact_threshold,
        gateway_candidates=args.gateway_candidates,
        gateway_seeds=args.gateway_seeds,
        use_temporal_edges=True,
        use_gateway_edges=False,
        use_range_edges=True,
        use_hierarchy=False,
    )
    # Warm vector pages and adjacency before comparing per-strategy latency.
    warm_record = workload[evaluation_positions[0]]
    warm_query = np.asarray(
        queries[warm_record["query_id"]], dtype=np.float32
    )
    for entry_ids in gateway_sets.values():
        index.search(
            warm_query, warm_record["intervals"], search_config,
            entry_gateway_ids=entry_ids,
        )

    rows: list[dict[str, Any]] = []
    for strategy, entry_ids in gateway_sets.items():
        started = time.perf_counter()
        for completed, position in enumerate(evaluation_positions, 1):
            record = workload[position]
            result = index.search(
                np.asarray(queries[record["query_id"]], dtype=np.float32),
                record["intervals"], search_config,
                entry_gateway_ids=entry_ids,
            )
            recall, precision = _quality(
                result["ids"], truth["neighbor_ids"][position]
            )
            rows.append({
                "workload_id": position,
                "query_id": record["query_id"],
                "strategy": strategy,
                "coverage_hours": record["coverage_hours"],
                "fragment_count": record["fragment_count"],
                "eligible_count": record["eligible_count"],
                "recall_at_k": recall,
                "precision_at_k": precision,
                "result_count": len(result["ids"]),
                "gateway_candidates": result["gateway_candidates"],
                "gateway_seeds": result["gateway_seeds"],
                "distance_computations": result["distance_computations"],
                "nodes_visited": result["nodes_visited"],
                "total_latency_ms": result["total_ms"],
            })
            if args.progress_every and completed % args.progress_every == 0:
                elapsed = time.perf_counter() - started
                print(
                    f"gateway oracle {strategy}: {completed}/{len(evaluation_positions)} "
                    f"elapsed={elapsed:.1f}s",
                    flush=True,
                )

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    slices: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["strategy"]].append(row)
        slices[(row["strategy"], "coverage_hours", row["coverage_hours"])].append(row)
        slices[(row["strategy"], "fragment_count", row["fragment_count"])].append(row)
    summary = {
        "method": "equal-budget offline gateway entry oracle",
        "gateway_edges_enabled": False,
        "configuration": {
            "gateway_count": gateway_count,
            "gateway_degree": index.config.gateway_degree,
            "candidate_step": args.candidate_step,
            "candidate_pool_size": len(pool),
            "seed": args.seed,
            "ef_search": args.ef_search,
            "exact_threshold": args.exact_threshold,
            "gateway_candidates": args.gateway_candidates,
            "gateway_seeds": args.gateway_seeds,
            "training_queries": len(training_positions),
            "held_out_queries": len(evaluation_positions),
        },
        "overall": {name: _aggregate(members) for name, members in grouped.items()},
        "factor_slices": [
            {"strategy": strategy, "dimension": dimension, "value": value,
             **_aggregate(members)}
            for (strategy, dimension, value), members in sorted(slices.items())
        ],
    }
    _write_rows(args.output / "queries.csv", rows)
    _write_json(args.output / "summary.json", summary)
    return rows, summary


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Evaluate equal-budget offline RT-HNSW gateway entry sets")
    value.add_argument("--index", type=Path, default=Path("Twitter_data/ann/rolling_hnsw/flat_100k_multiscale"))
    value.add_argument("--vectors", type=Path, default=Path("Twitter_data/processed/database_vectors.npy"))
    value.add_argument("--queries", type=Path, default=Path("Twitter_data/processed/query_vectors.npy"))
    value.add_argument("--workload", type=Path, default=Path("Twitter_data/workloads/rt_hnsw_100k_queries.jsonl"))
    value.add_argument("--ground-truth", type=Path, default=Path("results/twitter_rt_hnsw_100k/ground_truth.npz"))
    value.add_argument("--output", type=Path, default=Path("results/twitter_rt_hnsw_100k/gateway_oracle"))
    value.add_argument("--gateway-count", type=int, default=0)
    value.add_argument("--candidate-step", type=int, default=64)
    value.add_argument("--seed", type=int, default=29)
    value.add_argument("--threads", type=int, default=4)
    value.add_argument("--ef-search", type=int, default=256)
    value.add_argument("--exact-threshold", type=int, default=5_000)
    value.add_argument("--gateway-candidates", type=int, default=16)
    value.add_argument("--gateway-seeds", type=int, default=4)
    value.add_argument("--progress-every", type=int, default=50)
    return value


def main() -> None:
    args = parser().parse_args()
    _, summary = run(args)
    print(json.dumps(summary["overall"], indent=2))
