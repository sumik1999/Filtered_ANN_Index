from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from ..exact_benchmark import percentile
from ..oracle import exact_filtered_search
from .rolling_hnsw import RollingSearchConfig, RollingTemporalHNSW

HOUR_MS = 3_600_000


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("no rows to write")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_ints(value: str) -> list[int]:
    result = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not result or any(item <= 0 for item in result):
        raise ValueError("expected comma-separated positive integers")
    return result


def _split_amount(total: int, count: int) -> list[int]:
    base, remainder = divmod(total, count)
    return [base + int(position < remainder) for position in range(count)]


def _random_partition(total: int, count: int, rng: np.random.Generator) -> list[int]:
    if total <= 0:
        return [0] * count
    weights = rng.random(count)
    weights /= np.sum(weights)
    values = np.floor(weights * total).astype(np.int64)
    values[0] += total - int(np.sum(values))
    return [int(value) for value in values]


def make_intervals(
    timeline_start: int,
    timeline_end: int,
    coverage_hours: int,
    fragment_count: int,
    rng: np.random.Generator,
) -> list[list[int]]:
    total_width = coverage_hours * HOUR_MS
    span = timeline_end - timeline_start
    if total_width > span:
        raise ValueError("coverage exceeds RT-HNSW prototype timeline")
    widths = _split_amount(total_width, fragment_count)
    gaps = _random_partition(span - total_width, fragment_count + 1, rng)
    cursor = timeline_start + gaps[0]
    intervals: list[list[int]] = []
    for position, width in enumerate(widths):
        intervals.append([cursor, cursor + width])
        cursor += width + gaps[position + 1]
    return intervals


def generate_workload(
    index: RollingTemporalHNSW,
    query_count: int,
    coverage_hours: list[int],
    fragment_counts: list[int],
    seed: int,
) -> list[dict[str, Any]]:
    cells = len(coverage_hours) * len(fragment_counts)
    if query_count < cells or query_count % cells:
        raise ValueError("query_count must be a positive multiple of the workload cell count")
    per_cell = query_count // cells
    rng = np.random.default_rng(seed)
    query_order = rng.permutation(10_000)
    records: list[dict[str, Any]] = []
    timeline_start = int(index.timestamps[0])
    timeline_end = int(index.timestamps[-1]) + 1
    position = 0
    for width in coverage_hours:
        for fragments in fragment_counts:
            for _ in range(per_cell):
                intervals = make_intervals(
                    timeline_start, timeline_end, width, fragments, rng
                )
                components = index._components(intervals)
                eligible_count = sum(right - left for _, _, left, right in components)
                records.append({
                    "workload_id": position,
                    "query_id": int(query_order[position % len(query_order)]),
                    "k": 100,
                    "coverage_hours": width,
                    "fragment_count": fragments,
                    "intervals": intervals,
                    "eligible_count": eligible_count,
                    "actual_selectivity": eligible_count / len(index.global_ids),
                    "seed": seed,
                })
                position += 1
    rng.shuffle(records)
    for workload_id, record in enumerate(records):
        record["workload_id"] = workload_id
    return records


def write_workload(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")


def load_workload(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if [row["workload_id"] for row in rows] != list(range(len(rows))):
        raise ValueError("workload IDs must be dense and ordered")
    return rows


def generate_ground_truth(
    index: RollingTemporalHNSW,
    queries: np.ndarray,
    workload: list[dict[str, Any]],
    output: Path,
    block_size: int = 65_536,
    progress_every: int = 100,
) -> None:
    k = max(record["k"] for record in workload)
    neighbor_ids = np.full((len(workload), k), -1, dtype=np.int64)
    distances = np.full((len(workload), k), np.inf, dtype=np.float32)
    for position, record in enumerate(workload):
        parts = [
            np.arange(left, right, dtype=np.int64)
            for _, _, left, right in index._components(record["intervals"])
        ]
        local_ids = np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)
        eligible_global_ids = np.asarray(index.global_ids[local_ids], dtype=np.int64)
        ids, values = exact_filtered_search(
            index.vectors,
            np.asarray(queries[record["query_id"]], dtype=np.float32),
            eligible_global_ids,
            record["k"],
            block_size,
        )
        neighbor_ids[position, :len(ids)] = ids
        distances[position, :len(values)] = values
        if progress_every and (position + 1) % progress_every == 0:
            print(f"RT-HNSW ground truth: {position + 1}/{len(workload)}", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        neighbor_ids=neighbor_ids,
        distances=distances,
        workload_ids=np.arange(len(workload), dtype=np.int64),
    )


def _quality(found: np.ndarray, truth: np.ndarray) -> tuple[float, float]:
    expected = truth[truth >= 0]
    expected_set = set(map(int, expected))
    found_set = set(map(int, found))
    overlap = len(expected_set & found_set)
    recall = overlap / len(expected_set) if expected_set else float(not found_set)
    precision = overlap / len(found_set) if found_set else float(not expected_set)
    return recall, precision


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = [row["total_latency_ms"] for row in rows]
    return {
        "queries": len(rows),
        "mean_recall_at_k": float(np.mean([row["recall_at_k"] for row in rows])),
        "mean_precision_at_k": float(np.mean([row["precision_at_k"] for row in rows])),
        "complete_result_rate": float(np.mean([row["result_count"] == row["k"] for row in rows])),
        "mean_total_latency_ms": float(np.mean(latencies)),
        "p50_total_latency_ms": percentile(latencies, 50),
        "p95_total_latency_ms": percentile(latencies, 95),
        "mean_distance_computations": float(np.mean([row["distance_computations"] for row in rows])),
        "mean_nodes_visited": float(np.mean([row["nodes_visited"] for row in rows])),
        "mean_temporal_edges_examined": float(np.mean([row["temporal_edges_examined"] for row in rows])),
        "mean_range_edges_examined": float(np.mean([row["range_edges_examined"] for row in rows])),
        "mean_gateway_edges_examined": float(np.mean([row["gateway_edges_examined"] for row in rows])),
        "mean_semantic_edges_examined": float(np.mean([row["semantic_edges_examined"] for row in rows])),
        "mean_upper_entry_level": float(np.mean([row["upper_entry_level"] for row in rows])),
        "mean_upper_nodes_visited": float(np.mean([row["upper_nodes_visited"] for row in rows])),
        "mean_upper_distance_computations": float(np.mean([row["upper_distance_computations"] for row in rows])),
        "mean_upper_semantic_edges_examined": float(np.mean([row["upper_semantic_edges_examined"] for row in rows])),
        "mean_upper_range_edges_examined": float(np.mean([row["upper_range_edges_examined"] for row in rows])),
        "mean_upper_temporal_edges_examined": float(np.mean([row["upper_temporal_edges_examined"] for row in rows])),
        "mean_eligible_count": float(np.mean([row["eligible_count"] for row in rows])),
    }


def benchmark(
    index: RollingTemporalHNSW,
    queries: np.ndarray,
    workload: list[dict[str, Any]],
    truth: np.lib.npyio.NpzFile,
    output: Path,
    *,
    ef_search: int = 128,
    exact_threshold: int = 5_000,
    gateway_candidates: int = 16,
    gateway_seeds: int = 4,
    upper_ef_search: int = 32,
    variant_names: list[str] | None = None,
    progress_every: int = 100,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    variants = {
        "semantic_only": (False, False, False, False),
        "semantic_temporal": (True, False, False, False),
        "semantic_temporal_gateway": (True, True, False, False),
        "semantic_temporal_range": (True, False, True, False),
        "semantic_temporal_gateway_range": (True, True, True, False),
    }
    if index.config.hierarchy_enabled:
        variants["hierarchical_multiscale"] = (True, False, True, True)
    if variant_names:
        unknown = set(variant_names) - set(variants)
        if unknown:
            raise ValueError(f"unknown variants: {sorted(unknown)}")
        variants = {name: variants[name] for name in variant_names}
    rows: list[dict[str, Any]] = []
    for variant, (use_temporal, use_gateway, use_range, use_hierarchy) in variants.items():
        config = RollingSearchConfig(
            k=100,
            ef_search=ef_search,
            exact_threshold=exact_threshold,
            gateway_candidates=gateway_candidates,
            gateway_seeds=gateway_seeds,
            use_temporal_edges=use_temporal,
            use_gateway_edges=use_gateway,
            use_range_edges=use_range,
            use_hierarchy=use_hierarchy,
            upper_ef_search=upper_ef_search,
        )
        variant_started = time.perf_counter()
        for position, record in enumerate(workload):
            result = index.search(
                np.asarray(queries[record["query_id"]], dtype=np.float32),
                record["intervals"],
                config,
            )
            recall, precision = _quality(result["ids"], truth["neighbor_ids"][position])
            rows.append({
                "workload_id": record["workload_id"],
                "query_id": record["query_id"],
                "variant": variant,
                "coverage_hours": record["coverage_hours"],
                "fragment_count": record["fragment_count"],
                "eligible_count": record["eligible_count"],
                "actual_selectivity": record["actual_selectivity"],
                "k": record["k"],
                "execution_method": result["method"],
                "result_count": len(result["ids"]),
                "recall_at_k": recall,
                "precision_at_k": precision,
                "candidate_count": result["candidate_count"],
                "distance_computations": result["distance_computations"],
                "rerank_distance_computations": result["rerank_distance_computations"],
                "nodes_visited": result["nodes_visited"],
                "gateway_candidates": result["gateway_candidates"],
                "gateway_seeds": result["gateway_seeds"],
                "semantic_edges_examined": result["semantic_edges_examined"],
                "range_edges_examined": result["range_edges_examined"],
                "gateway_edges_examined": result["gateway_edges_examined"],
                "temporal_edges_examined": result["temporal_edges_examined"],
                "out_of_range_edges_rejected": result["out_of_range_edges_rejected"],
                "upper_entry_level": result["upper_entry_level"],
                "upper_nodes_visited": result["upper_nodes_visited"],
                "upper_distance_computations": result["upper_distance_computations"],
                "upper_semantic_edges_examined": result["upper_semantic_edges_examined"],
                "upper_range_edges_examined": result["upper_range_edges_examined"],
                "upper_temporal_edges_examined": result["upper_temporal_edges_examined"],
                "upper_out_of_range_edges_rejected": result["upper_out_of_range_edges_rejected"],
                "routing_ms": result["routing_ms"],
                "search_ms": result["search_ms"],
                "rerank_ms": result["rerank_ms"],
                "total_latency_ms": result["total_ms"],
            })
            completed = position + 1
            if progress_every and completed % progress_every == 0:
                elapsed = time.perf_counter() - variant_started
                eta = elapsed * (len(workload) - completed) / completed
                _write_rows(output / "queries.partial.csv", rows)
                print(
                    f"RT-HNSW {variant}: {completed}/{len(workload)} "
                    f"({100 * completed / len(workload):.1f}%), "
                    f"elapsed={elapsed / 60:.1f}m eta={eta / 60:.1f}m",
                    flush=True,
                )

    by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    slices: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_variant[row["variant"]].append(row)
        slices[(row["variant"], "coverage_hours", row["coverage_hours"])].append(row)
        slices[(row["variant"], "fragment_count", row["fragment_count"])].append(row)
    summary = {
        "method": "flat rolling temporal HNSW ablation",
        "measurements": len(rows),
        "queries_per_variant": len(workload),
        "configuration": {
            "ef_search": ef_search,
            "exact_threshold": exact_threshold,
            "gateway_candidates": gateway_candidates,
            "gateway_seeds": gateway_seeds,
            "upper_ef_search": upper_ef_search,
            "variants": variants,
        },
        "overall": {name: _aggregate(members) for name, members in by_variant.items()},
        "factor_slices": [
            {"variant": variant, "dimension": dimension, "value": value, **_aggregate(members)}
            for (variant, dimension, value), members in sorted(slices.items())
        ],
    }
    _write_rows(output / "queries.csv", rows)
    _write_json(output / "summary.json", summary)
    (output / "queries.partial.csv").unlink(missing_ok=True)
    return rows, summary


def generation_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate the matched 100k RT-HNSW workload and ground truth")
    parser.add_argument("--index", type=Path, default=Path("Twitter_data/ann/rolling_hnsw/flat_100k"))
    parser.add_argument("--vectors", type=Path, default=Path("Twitter_data/processed/database_vectors.npy"))
    parser.add_argument("--queries", type=Path, default=Path("Twitter_data/processed/query_vectors.npy"))
    parser.add_argument("--workload", type=Path, default=Path("Twitter_data/workloads/rt_hnsw_100k_queries.jsonl"))
    parser.add_argument("--ground-truth", type=Path, default=Path("results/twitter_rt_hnsw_100k/ground_truth.npz"))
    parser.add_argument("--query-count", type=int, default=900)
    parser.add_argument("--coverage-hours", default="1,6,24,168,720")
    parser.add_argument("--fragment-counts", default="1,2,4")
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--block-size", type=int, default=65_536)
    parser.add_argument("--progress-every", type=int, default=100)
    return parser


def benchmark_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark the flat 100k RT-HNSW ablation")
    parser.add_argument("--index", type=Path, default=Path("Twitter_data/ann/rolling_hnsw/flat_100k"))
    parser.add_argument("--vectors", type=Path, default=Path("Twitter_data/processed/database_vectors.npy"))
    parser.add_argument("--queries", type=Path, default=Path("Twitter_data/processed/query_vectors.npy"))
    parser.add_argument("--workload", type=Path, default=Path("Twitter_data/workloads/rt_hnsw_100k_queries.jsonl"))
    parser.add_argument("--ground-truth", type=Path, default=Path("results/twitter_rt_hnsw_100k/ground_truth.npz"))
    parser.add_argument("--output", type=Path, default=Path("results/twitter_rt_hnsw_100k/ablation"))
    parser.add_argument("--ef-search", type=int, default=128)
    parser.add_argument("--exact-threshold", type=int, default=5_000)
    parser.add_argument("--gateway-candidates", type=int, default=16)
    parser.add_argument("--gateway-seeds", type=int, default=4)
    parser.add_argument("--upper-ef-search", type=int, default=32)
    parser.add_argument("--variants", default="")
    parser.add_argument("--progress-every", type=int, default=100)
    return parser


def generate_main() -> None:
    args = generation_parser().parse_args()
    vectors = np.load(args.vectors, mmap_mode="r")
    queries = np.load(args.queries, mmap_mode="r")
    index = RollingTemporalHNSW(args.index, vectors)
    workload = generate_workload(
        index, args.query_count, parse_ints(args.coverage_hours),
        parse_ints(args.fragment_counts), args.seed,
    )
    write_workload(args.workload, workload)
    generate_ground_truth(
        index, queries, workload, args.ground_truth,
        block_size=args.block_size, progress_every=args.progress_every,
    )
    print(f"Wrote {len(workload)} RT-HNSW queries and exact ground truth")


def benchmark_main() -> None:
    args = benchmark_parser().parse_args()
    vectors = np.load(args.vectors, mmap_mode="r")
    queries = np.load(args.queries, mmap_mode="r")
    index = RollingTemporalHNSW(args.index, vectors)
    workload = load_workload(args.workload)
    truth = np.load(args.ground_truth)
    _, summary = benchmark(
        index, queries, workload, truth, args.output,
        ef_search=args.ef_search,
        exact_threshold=args.exact_threshold,
        gateway_candidates=args.gateway_candidates,
        gateway_seeds=args.gateway_seeds,
        upper_ef_search=args.upper_ef_search,
        variant_names=[value.strip() for value in args.variants.split(",") if value.strip()],
        progress_every=args.progress_every,
    )
    print(json.dumps(summary["overall"], indent=2))
