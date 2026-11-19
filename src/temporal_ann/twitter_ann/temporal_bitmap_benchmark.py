from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from ..exact_benchmark import percentile
from .rolling_benchmark import load_workload
from .temporal_bitmap_hnsw import (
    TemporalBitmapHNSW,
    TemporalBitmapSearchConfig,
)


def _quality(found: np.ndarray, expected: np.ndarray) -> tuple[float, float]:
    truth = expected[expected >= 0]
    found_set, truth_set = set(map(int, found)), set(map(int, truth))
    overlap = len(found_set & truth_set)
    return (
        overlap / len(truth_set) if len(truth_set) else float(not len(found)),
        overlap / len(found_set) if len(found_set) else float(not len(truth)),
    )


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "queries": len(rows),
        "mean_recall_at_k": float(np.mean([row["recall_at_k"] for row in rows])),
        "mean_precision_at_k": float(np.mean([row["precision_at_k"] for row in rows])),
        "mean_total_latency_ms": float(np.mean([row["total_latency_ms"] for row in rows])),
        "p95_total_latency_ms": percentile([row["total_latency_ms"] for row in rows], 95),
        "mean_distance_computations": float(np.mean([row["distance_computations"] for row in rows])),
        "mean_edges_examined": float(np.mean([row["edges_examined"] for row in rows])),
        "mean_bitmap_groups_considered": float(np.mean([row["bitmap_groups_considered"] for row in rows])),
        "mean_bitmap_groups_skipped": float(np.mean([row["bitmap_groups_skipped"] for row in rows])),
        "mean_out_of_range_edges_rejected": float(np.mean([row["out_of_range_edges_rejected"] for row in rows])),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    vectors = np.load(args.vectors, mmap_mode="r")
    queries = np.load(args.queries, mmap_mode="r")
    index = TemporalBitmapHNSW(args.index, vectors)
    workload = load_workload(args.workload)
    truth = np.load(args.ground_truth)
    base = TemporalBitmapSearchConfig(
        k=100, ef_search=args.ef_search, exact_threshold=args.exact_threshold,
        gateway_candidates=args.gateway_candidates, gateway_seeds=args.gateway_seeds,
    )
    variants = {
        "bitmap_pruning": base,
        "no_bitmap_pruning": replace(base, use_bitmap_pruning=False),
    }
    rows = []
    for variant, config in variants.items():
        for position, record in enumerate(workload):
            result = index.search(
                np.asarray(queries[record["query_id"]], dtype=np.float32),
                record["intervals"], config,
            )
            recall, precision = _quality(result["ids"], truth["neighbor_ids"][position])
            rows.append({
                "workload_id": position,
                "variant": variant,
                "coverage_hours": record["coverage_hours"],
                "fragment_count": record["fragment_count"],
                "eligible_count": record["eligible_count"],
                "recall_at_k": recall,
                "precision_at_k": precision,
                "total_latency_ms": result["total_ms"],
                "distance_computations": result["distance_computations"],
                "nodes_visited": result["nodes_visited"],
                "edges_examined": result["edges_examined"],
                "bitmap_groups_considered": result["bitmap_groups_considered"],
                "bitmap_groups_skipped": result["bitmap_groups_skipped"],
                "out_of_range_edges_rejected": result["out_of_range_edges_rejected"],
            })
        print(f"bitmap benchmark completed: {variant}", flush=True)
    grouped = defaultdict(list)
    slices = defaultdict(list)
    for row in rows:
        grouped[row["variant"]].append(row)
        slices[(row["variant"], row["coverage_hours"])].append(row)
    summary = {
        "index": str(args.index),
        "configuration": {
            "ef_search": args.ef_search,
            "exact_threshold": args.exact_threshold,
            "gateway_candidates": args.gateway_candidates,
            "gateway_seeds": args.gateway_seeds,
        },
        "overall": {name: _aggregate(values) for name, values in grouped.items()},
        "coverage_slices": [
            {"variant": variant, "coverage_hours": coverage, **_aggregate(values)}
            for (variant, coverage), values in sorted(slices.items())
        ],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "queries.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Benchmark time-stretched bitmap RT-HNSW")
    value.add_argument("--index", type=Path, default=Path("Twitter_data/ann/rolling_hnsw/bitmap_100k"))
    value.add_argument("--vectors", type=Path, default=Path("Twitter_data/processed/database_vectors.npy"))
    value.add_argument("--queries", type=Path, default=Path("Twitter_data/processed/query_vectors.npy"))
    value.add_argument("--workload", type=Path, default=Path("Twitter_data/workloads/rt_hnsw_100k_queries.jsonl"))
    value.add_argument("--ground-truth", type=Path, default=Path("results/twitter_rt_hnsw_100k/ground_truth.npz"))
    value.add_argument("--output", type=Path, default=Path("results/twitter_rt_hnsw_100k/temporal_bitmap"))
    value.add_argument("--ef-search", type=int, default=256)
    value.add_argument("--exact-threshold", type=int, default=5_000)
    value.add_argument("--gateway-candidates", type=int, default=16)
    value.add_argument("--gateway-seeds", type=int, default=4)
    return value


def main() -> None:
    summary = run(parser().parse_args())
    print(json.dumps(summary["overall"], indent=2))
