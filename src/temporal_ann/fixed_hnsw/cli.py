from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import faiss
import h5py
import numpy as np

from ..exact_benchmark import DISTRIBUTIONS, load_workload, percentile, recall
from ..hnsw_post_filter import parse_ints
from .index import (
    FixedHNSWConfig,
    FixedHNSWIndex,
    FixedHNSWSearchConfig,
    build_fixed_hnsw,
)


def parse_distributions(value: str) -> list[str]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    if values == ["all"]:
        return list(DISTRIBUTIONS)
    unknown = set(values) - set(DISTRIBUTIONS)
    if not values or unknown:
        raise ValueError(f"distributions must be selected from {DISTRIBUTIONS} or 'all'")
    return values


def index_directory(
    root: Path, distribution: str, bucket_count: int, m: int, ef_construction: int
) -> Path:
    return root / distribution / f"buckets_{bucket_count}_m{m}_efc{ef_construction}"


def load_timestamps(directory: Path, distribution: str) -> tuple[np.ndarray, str]:
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    filename = metadata["distributions"][distribution]["file"]
    return np.load(directory / filename, mmap_mode="r"), filename


def run_build(args: argparse.Namespace) -> list[dict]:
    bucket_counts = parse_ints(args.bucket_counts)
    distributions = parse_distributions(args.distributions)
    with h5py.File(args.sift_file, "r") as sift:
        vectors = np.asarray(sift["train"], dtype=np.float32)

    manifests = []
    for distribution in distributions:
        timestamps, timestamp_file = load_timestamps(args.timestamps, distribution)
        for bucket_count in bucket_counts:
            config = FixedHNSWConfig(
                bucket_count=bucket_count,
                m=args.m,
                ef_construction=args.ef_construction,
                threads=args.threads,
                timeline_start=args.timeline_start,
                timeline_end=args.timeline_end,
            )
            output = index_directory(
                args.index_root, distribution, bucket_count, args.m, args.ef_construction
            )
            manifest = build_fixed_hnsw(
                vectors,
                timestamps,
                output,
                config,
                source_metadata={
                    "vectors": str(args.sift_file) + "::train",
                    "timestamps": str(args.timestamps / timestamp_file),
                    "distribution": distribution,
                },
                rebuild=args.rebuild,
            )
            manifests.append(manifest)
            print(
                f"{distribution}: persisted {bucket_count} buckets at {output} "
                f"({manifest['construction_seconds']:.2f}s)",
                flush=True,
            )
    return manifests


def summarize(rows: list[dict]) -> dict:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        key = (
            row["distribution"],
            row["bucket_count"],
            row["ef_search"],
            row["overfetch"],
        )
        grouped[key].append(row)
    groups = []
    for (distribution, buckets, ef_search, overfetch), members in grouped.items():
        groups.append(
            {
                "distribution": distribution,
                "bucket_count": buckets,
                "ef_search": ef_search,
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
                "mean_selected_buckets": float(
                    np.mean([x["selected_bucket_count"] for x in members])
                ),
                "mean_valid_candidates": float(
                    np.mean([x["valid_candidates"] for x in members])
                ),
                "mean_distance_computations": float(
                    np.mean([x["distance_computations"] for x in members])
                ),
            }
        )
    return {"method": "fixed-width temporal FAISS HNSW", "groups": groups}


def run_benchmark(args: argparse.Namespace) -> tuple[list[dict], dict]:
    bucket_counts = parse_ints(args.bucket_counts)
    ef_values = parse_ints(args.ef_search)
    overfetch_values = parse_ints(args.overfetch)
    distributions = parse_distributions(args.distributions)
    workload = load_workload(args.workload)
    faiss.omp_set_num_threads(args.threads)

    with h5py.File(args.sift_file, "r") as sift:
        vectors = np.asarray(sift["train"], dtype=np.float32)
        all_queries = np.asarray(sift["test"], dtype=np.float32)
    queries = all_queries[[record["query_id"] for record in workload]]

    rows: list[dict] = []
    indexes: list[dict] = []
    for distribution in distributions:
        timestamps, _ = load_timestamps(args.timestamps, distribution)
        ground_truth = np.load(args.ground_truth / f"{distribution}_ground_truth.npz")
        for bucket_count in bucket_counts:
            directory = index_directory(
                args.index_root, distribution, bucket_count, args.m, args.ef_construction
            )
            if not directory.exists():
                raise FileNotFoundError(
                    f"missing {directory}; run build-fixed-hnsw with matching parameters"
                )
            index = FixedHNSWIndex(directory)
            index.preload()
            indexes.append(
                {
                    "distribution": distribution,
                    "bucket_count": bucket_count,
                    "directory": str(directory),
                    "index_size_bytes": index.manifest["index_size_bytes"],
                    "mapping_size_bytes": index.manifest["mapping_size_bytes"],
                    "construction_seconds": index.manifest["construction_seconds"],
                }
            )
            for ef_search in ef_values:
                for overfetch in overfetch_values:
                    search_config = FixedHNSWSearchConfig(ef_search, overfetch)
                    for record, query in zip(
                        workload[: args.warmup], queries[: args.warmup]
                    ):
                        index.search(
                            query,
                            timestamps,
                            vectors,
                            record["intervals"],
                            args.k,
                            search_config,
                        )
                    for position, (record, query) in enumerate(zip(workload, queries)):
                        result = index.search(
                            query,
                            timestamps,
                            vectors,
                            record["intervals"],
                            args.k,
                            search_config,
                        )
                        truth = ground_truth["neighbor_ids"][position]
                        rows.append(
                            {
                                "distribution": distribution,
                                "workload_id": record["workload_id"],
                                "query_id": record["query_id"],
                                "k": args.k,
                                "bucket_count": bucket_count,
                                "ef_search": ef_search,
                                "effective_ef_search": result["effective_ef_search"],
                                "overfetch": overfetch,
                                "selected_bucket_count": len(result["selected_buckets"]),
                                "candidates_requested": result["candidates_requested"],
                                "candidates_returned": result["candidates_returned"],
                                "unique_candidates": result["unique_candidates"],
                                "valid_candidates": result["valid_candidates"],
                                "distance_computations": result["distance_computations"],
                                "result_count": len(result["ids"]),
                                "search_ms": result["search_ms"],
                                "filter_ms": result["filter_ms"],
                                "rerank_ms": result["rerank_ms"],
                                "total_latency_ms": result["total_ms"],
                                "recall_at_k": recall(result["ids"], truth, args.k),
                                "actual_selectivity": float(
                                    ground_truth["actual_selectivity"][position]
                                ),
                            }
                        )
                        if args.progress_every and (position + 1) % args.progress_every == 0:
                            print(
                                f"{distribution} buckets={bucket_count} ef={ef_search} "
                                f"overfetch={overfetch}: {position + 1}/{len(workload)}",
                                flush=True,
                            )
    summary = summarize(rows)
    summary["indexes"] = indexes
    summary["query_count"] = len(workload)
    summary["warmup_queries"] = args.warmup
    return rows, summary


def write_results(
    output: Path, rows: list[dict], summary: dict, args: argparse.Namespace
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    with (output / "queries.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    config = {
        "method": "fixed_width_temporal_hnsw",
        "distributions": parse_distributions(args.distributions),
        "bucket_counts": parse_ints(args.bucket_counts),
        "k": args.k,
        "ef_search": parse_ints(args.ef_search),
        "overfetch": parse_ints(args.overfetch),
        "m": args.m,
        "ef_construction": args.ef_construction,
        "threads": args.threads,
        "warmup": args.warmup,
    }
    (output / "config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )


def common_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--sift-file", type=Path, default=Path("SIFT_data/sift-128-euclidean.hdf5")
    )
    parser.add_argument("--timestamps", type=Path, default=Path("SIFT_data/timestamps"))
    parser.add_argument(
        "--index-root", type=Path, default=Path("SIFT_data/fixed_hnsw")
    )
    parser.add_argument("--distributions", default="all")
    parser.add_argument("--bucket-counts", default="32")
    parser.add_argument("--m", type=int, default=32)
    parser.add_argument("--ef-construction", type=int, default=200)
    parser.add_argument("--threads", type=int, default=4)
    return parser


def build_parser() -> argparse.ArgumentParser:
    parser = common_parser("Build persistent fixed-width temporal HNSW indexes")
    parser.add_argument("--timeline-start", type=float, default=0.0)
    parser.add_argument("--timeline-end", type=float, default=1.0)
    parser.add_argument("--rebuild", action="store_true")
    return parser


def benchmark_parser() -> argparse.ArgumentParser:
    parser = common_parser("Benchmark fixed-width temporal HNSW indexes")
    parser.add_argument(
        "--workload",
        type=Path,
        default=Path("SIFT_data/workloads/temporal_queries_10000.jsonl"),
    )
    parser.add_argument(
        "--ground-truth", type=Path, default=Path("SIFT_data/ground_truth_10000")
    )
    parser.add_argument("--output", type=Path, default=Path("results/fixed_hnsw"))
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--ef-search", default="256")
    parser.add_argument("--overfetch", default="2")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--progress-every", type=int, default=500)
    return parser


def build_main() -> None:
    run_build(build_parser().parse_args())


def benchmark_main() -> None:
    args = benchmark_parser().parse_args()
    rows, summary = run_benchmark(args)
    write_results(args.output, rows, summary, args)
    print(f"Wrote {len(rows)} fixed-HNSW measurements to {args.output}")
