from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import faiss
import h5py
import numpy as np

from .exact_benchmark import DISTRIBUTIONS, load_workload, percentile, recall
from .faiss_hnsw import load_or_build_index


def parse_ints(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values or any(item <= 0 for item in values):
        raise ValueError("parameter lists must contain positive integers")
    return values


def candidate_mask(
    candidate_ids: np.ndarray,
    timestamps: np.ndarray,
    intervals: list[list[float]],
) -> np.ndarray:
    valid = candidate_ids >= 0
    candidate_times = np.empty(len(candidate_ids), dtype=np.float64)
    candidate_times[valid] = timestamps[candidate_ids[valid]]
    temporal = np.zeros(len(candidate_ids), dtype=bool)
    for start, end in intervals:
        temporal |= valid & (candidate_times >= start) & (candidate_times < end)
    return temporal


def search_post_filtered(
    index: faiss.Index,
    query: np.ndarray,
    timestamps: np.ndarray,
    intervals: list[list[float]],
    k: int,
    overfetch: int,
    ef_search: int,
) -> dict:
    fetch_count = min(index.ntotal, k * overfetch)
    effective_ef_search = max(ef_search, fetch_count)
    params = faiss.SearchParametersHNSW(efSearch=effective_ef_search)

    faiss.cvar.hnsw_stats.reset()
    search_started = time.perf_counter_ns()
    distances, ids = index.search(query.reshape(1, -1), fetch_count, params=params)
    search_ms = (time.perf_counter_ns() - search_started) / 1e6
    distance_computations = int(faiss.cvar.hnsw_stats.ndis)

    filter_started = time.perf_counter_ns()
    ids = ids[0]
    distances = distances[0]
    valid = candidate_mask(ids, timestamps, intervals)
    result_ids = ids[valid][:k].astype(np.int32, copy=False)
    result_distances = distances[valid][:k].astype(np.float32, copy=False)
    filter_ms = (time.perf_counter_ns() - filter_started) / 1e6
    return {
        "ids": result_ids,
        "distances": result_distances,
        "fetch_count": fetch_count,
        "valid_candidates": int(valid.sum()),
        "distance_computations": distance_computations,
        "effective_ef_search": effective_ef_search,
        "search_ms": search_ms,
        "filter_ms": filter_ms,
        "total_ms": search_ms + filter_ms,
    }


def run(args: argparse.Namespace) -> tuple[list[dict], dict]:
    workload = load_workload(args.workload)
    ef_values = parse_ints(args.ef_search)
    overfetch_values = parse_ints(args.overfetch)
    faiss.omp_set_num_threads(args.threads)
    index, index_metadata = load_or_build_index(
        args.sift_file,
        args.index,
        args.index_metadata,
        args.m,
        args.ef_construction,
        args.threads,
        args.rebuild_index,
    )

    with h5py.File(args.sift_file, "r") as sift:
        all_queries = np.asarray(sift["test"], dtype=np.float32)
    queries = all_queries[[record["query_id"] for record in workload]]
    timestamp_metadata = json.loads(
        (args.timestamps / "metadata.json").read_text(encoding="utf-8")
    )

    rows: list[dict] = []
    for distribution in DISTRIBUTIONS:
        timestamps = np.load(
            args.timestamps
            / timestamp_metadata["distributions"][distribution]["file"],
            mmap_mode="r",
        )
        ground_truth = np.load(
            args.ground_truth / f"{distribution}_ground_truth.npz"
        )
        for ef_search in ef_values:
            for overfetch in overfetch_values:
                for position, (record, query) in enumerate(zip(workload, queries)):
                    result = search_post_filtered(
                        index,
                        query,
                        timestamps,
                        record["intervals"],
                        args.k,
                        overfetch,
                        ef_search,
                    )
                    truth = ground_truth["neighbor_ids"][position]
                    rows.append(
                        {
                            "distribution": distribution,
                            "workload_id": record["workload_id"],
                            "query_id": record["query_id"],
                            "k": args.k,
                            "ef_search": ef_search,
                            "effective_ef_search": result["effective_ef_search"],
                            "overfetch": overfetch,
                            "candidates_requested": result["fetch_count"],
                            "valid_candidates": result["valid_candidates"],
                            "distance_computations": result["distance_computations"],
                            "result_count": len(result["ids"]),
                            "search_ms": result["search_ms"],
                            "filter_ms": result["filter_ms"],
                            "total_latency_ms": result["total_ms"],
                            "recall_at_k": recall(result["ids"], truth, args.k),
                            "actual_selectivity": float(
                                ground_truth["actual_selectivity"][position]
                            ),
                        }
                    )
                    if (position + 1) % args.progress_every == 0:
                        print(
                            f"{distribution} ef={ef_search} overfetch={overfetch}: "
                            f"{position + 1}/{len(workload)}",
                            flush=True,
                        )

    summary = summarize(rows)
    summary.update(
        {
            "method": "global FAISS HNSW post-filtering",
            "index": index_metadata,
            "query_count": len(workload),
            "timestamp_attributes_stored_outside_faiss": True,
        }
    )
    return rows, summary


def summarize(rows: list[dict]) -> dict:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["distribution"], row["ef_search"], row["overfetch"])].append(row)
    groups = []
    for (distribution, ef_search, overfetch), members in grouped.items():
        groups.append(
            {
                "distribution": distribution,
                "requested_ef_search": ef_search,
                "effective_ef_search": members[0]["effective_ef_search"],
                "overfetch": overfetch,
                "queries": len(members),
                "mean_recall_at_k": float(np.mean([x["recall_at_k"] for x in members])),
                "complete_result_rate": float(
                    np.mean([x["result_count"] == x["k"] for x in members])
                ),
                "mean_total_latency_ms": float(
                    np.mean([x["total_latency_ms"] for x in members])
                ),
                "p50_total_latency_ms": percentile(
                    [x["total_latency_ms"] for x in members], 50
                ),
                "p95_total_latency_ms": percentile(
                    [x["total_latency_ms"] for x in members], 95
                ),
                "mean_distance_computations": float(
                    np.mean([x["distance_computations"] for x in members])
                ),
            }
        )
    return {"groups": groups}


def write_results(output: Path, rows: list[dict], summary: dict, args: argparse.Namespace) -> None:
    output.mkdir(parents=True, exist_ok=True)
    with (output / "queries.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    config = {
        "method": "global_hnsw_post_filter",
        "k": args.k,
        "ef_search": parse_ints(args.ef_search),
        "overfetch": parse_ints(args.overfetch),
        "threads": args.threads,
    }
    (output / "config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark global HNSW post-filtering")
    parser.add_argument("--sift-file", type=Path, default=Path("SIFT_data/sift-128-euclidean.hdf5"))
    parser.add_argument("--timestamps", type=Path, default=Path("SIFT_data/timestamps"))
    parser.add_argument("--workload", type=Path, default=Path("SIFT_data/workloads/temporal_queries_10000.jsonl"))
    parser.add_argument("--ground-truth", type=Path, default=Path("SIFT_data/ground_truth_10000"))
    parser.add_argument("--index", type=Path, default=Path("SIFT_data/faiss/hnsw_m32_efc200.index"))
    parser.add_argument("--index-metadata", type=Path, default=Path("SIFT_data/faiss/metadata.json"))
    parser.add_argument("--output", type=Path, default=Path("results/hnsw_post_filter"))
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--ef-search", default="256")
    parser.add_argument("--overfetch", default="10")
    parser.add_argument("--m", type=int, default=32)
    parser.add_argument("--ef-construction", type=int, default=200)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--progress-every", type=int, default=500)
    parser.add_argument("--rebuild-index", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rows, summary = run(args)
    write_results(args.output, rows, summary, args)
    print(f"Wrote {len(rows)} post-filter measurements to {args.output}")


if __name__ == "__main__":
    main()
