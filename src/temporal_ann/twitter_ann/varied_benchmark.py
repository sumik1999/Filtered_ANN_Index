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
from ..oracle import exact_filtered_search
from ..twitter_data import YEAR_END_MS, YEAR_START_MS
from .bucketed import (
    AdaptivePlanner,
    BucketBuildConfig,
    BucketSearchConfig,
    BucketedHNSW,
    PlannerConfig,
    build_bucket_indexes,
    eligible_ids_from_sorted,
    interval_mask,
    normalize_intervals,
    resolve_intervals,
)
from .hnsw import parse_ints


def load_workload(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows or [row["workload_id"] for row in rows] != list(range(len(rows))):
        raise ValueError("workload IDs must be non-empty, dense, and ordered")
    for row in rows:
        operator = str(row.get("predicate_operator", "OR")).upper()
        raw = row["intervals"]
        if len(raw) != row["predicate_count"]:
            raise ValueError(f"workload {row['workload_id']} predicate count mismatch")
        effective = resolve_intervals(raw, operator)
        if operator == "OR" and len(normalize_intervals(raw)) != len(raw):
            raise ValueError(f"workload {row['workload_id']} has overlapping OR predicates")
        if not effective:
            raise ValueError(f"workload {row['workload_id']} has an empty {operator} predicate")
        if ("effective_intervals" in row and
                [list(value) for value in effective] != row["effective_intervals"]):
            raise ValueError(f"workload {row['workload_id']} effective intervals mismatch")
        row["predicate_operator"] = operator
        row["_effective_intervals"] = effective
    return rows


def _intervals(record: dict[str, Any]) -> list[tuple[int, int]]:
    return record.get("_effective_intervals") or resolve_intervals(
        record["intervals"], record.get("predicate_operator", "OR")
    )


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("no benchmark rows")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def _base_row(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "workload_id": record["workload_id"], "query_id": record["query_id"],
        "predicate_operator": record.get("predicate_operator", "OR"),
        "predicate_count": record["predicate_count"],
        "coverage_hours": record["requested_total_coverage_hours"],
        "traffic_class": record["anchor_traffic_class"],
        "vector_density_class": record["vector_density_class"],
        "time_vector_alignment": record["time_vector_alignment"],
        "semantic_cluster_id": record["semantic_cluster_id"],
        "eligible_count": record["eligible_count"],
        "actual_selectivity": record["actual_selectivity"], "k": record["k"],
    }


def recall(found: np.ndarray, expected: np.ndarray, k: int) -> float:
    truth = expected[expected >= 0][:k]
    if not len(truth):
        return 1.0 if not len(found) else 0.0
    return len(set(map(int, found)) & set(map(int, truth))) / len(truth)


def _aggregate(members: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"queries": len(members)}
    for field in (
        "recall_at_k", "total_latency_ms", "candidates_examined", "valid_candidates",
        "selected_bucket_count", "partial_bucket_count", "distance_computations",
        "routing_ms", "search_ms", "filter_ms", "rerank_ms", "planner_ms",
    ):
        if field in members[0]:
            result[f"mean_{field}"] = float(np.mean([row[field] for row in members]))
    if "result_count" in members[0]:
        result["complete_result_rate"] = float(np.mean([row["result_count"] == row["k"] for row in members]))
    if "total_latency_ms" in members[0]:
        values = [row["total_latency_ms"] for row in members]
        result["p50_total_latency_ms"] = percentile(values, 50)
        result["p95_total_latency_ms"] = percentile(values, 95)
    return result


def summarize(rows: list[dict[str, Any]], method: str, configuration: dict[str, Any]) -> dict[str, Any]:
    dimensions = (
        "predicate_operator", "predicate_count", "coverage_hours", "traffic_class",
        "vector_density_class", "time_vector_alignment",
    )
    slices = []
    for dimension in dimensions:
        grouped: dict[Any, list[dict[str, Any]]] = defaultdict(list)
        for row in rows: grouped[row[dimension]].append(row)
        for value, members in sorted(grouped.items(), key=lambda item: str(item[0])):
            slices.append({"dimension": dimension, "value": value, **_aggregate(members)})
    interactions: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        interactions[(row["predicate_count"], row["vector_density_class"])].append(row)
    interaction_rows = [
        {"predicate_count": key[0], "vector_density_class": key[1], **_aggregate(members)}
        for key, members in sorted(interactions.items())
    ]
    configuration_fields = [
        field for field in ("bucket_strategy", "bucket_count", "ef_search", "overfetch", "chosen_method")
        if field in rows[0]
    ]
    configuration_groups = []
    if configuration_fields:
        grouped_configs: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped_configs[tuple(row[field] for field in configuration_fields)].append(row)
        for key, members in sorted(grouped_configs.items(), key=lambda item: str(item[0])):
            configuration_groups.append({
                **dict(zip(configuration_fields, key)), **_aggregate(members)
            })
    return {
        "method": method, "configuration": configuration,
        "measurements": len(rows), "overall": _aggregate(rows),
        "configuration_groups": configuration_groups,
        "factor_slices": slices, "predicate_density_interactions": interaction_rows,
    }


def run_exact(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    workload = load_workload(args.workload)
    vectors = np.load(args.embeddings / "database_vectors.npy", mmap_mode="r")
    queries = np.load(args.embeddings / "query_vectors.npy", mmap_mode="r")
    timestamps = np.load(args.timestamps, mmap_mode="r")
    order = np.argsort(timestamps, kind="stable")
    sorted_times, sorted_ids = timestamps[order], order.astype(np.int64)
    rows, truth_ids, truth_distances = [], [], []
    for position, record in enumerate(workload):
        started = time.perf_counter_ns()
        filter_started = time.perf_counter_ns()
        eligible = eligible_ids_from_sorted(sorted_times, sorted_ids, _intervals(record))
        filter_ms = (time.perf_counter_ns() - filter_started) / 1e6
        search_started = time.perf_counter_ns()
        ids, distances = exact_filtered_search(
            vectors, queries[record["query_id"]], eligible, record["k"], args.block_size
        )
        search_ms = (time.perf_counter_ns() - search_started) / 1e6
        padded_ids = np.full(record["k"], -1, dtype=np.int64)
        padded_distances = np.full(record["k"], np.inf, dtype=np.float32)
        padded_ids[:len(ids)], padded_distances[:len(distances)] = ids, distances
        truth_ids.append(padded_ids); truth_distances.append(padded_distances)
        rows.append({**_base_row(record), "result_count": len(ids),
                     "candidates_examined": len(eligible), "timestamp_filter_ms": filter_ms,
                     "exact_l2_ms": search_ms,
                     "total_latency_ms": (time.perf_counter_ns() - started) / 1e6})
        if args.progress_every and (position + 1) % args.progress_every == 0:
            print(f"varied exact: {position + 1}/{len(workload)}", flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    _write_rows(args.output / "queries.csv", rows)
    np.savez_compressed(args.output / "ground_truth.npz",
                       neighbor_ids=np.stack(truth_ids), distances=np.stack(truth_distances),
                       workload_ids=np.arange(len(workload), dtype=np.int64))
    summary = summarize(rows, "exact temporal pre-filter plus blocked squared L2",
                        {"block_size": args.block_size, "workload": str(args.workload)})
    _write_json(args.output / "summary.json", summary)
    return rows, summary


def _global_search(
    index: faiss.Index, query: np.ndarray, timestamps: np.ndarray,
    intervals: list[list[int]], k: int, overfetch: int, ef_search: int,
) -> dict[str, Any]:
    fetch = min(index.ntotal, k * overfetch); effective = max(fetch, ef_search)
    params = faiss.SearchParametersHNSW(efSearch=effective)
    faiss.cvar.hnsw_stats.reset(); started = time.perf_counter_ns()
    distances, ids = index.search(query.reshape(1, -1), fetch, params=params)
    search_ms = (time.perf_counter_ns() - started) / 1e6
    computations = int(faiss.cvar.hnsw_stats.ndis)
    started = time.perf_counter_ns(); ids, distances = ids[0], distances[0]
    valid_ids = ids >= 0; mask = np.zeros(len(ids), dtype=bool)
    mask[valid_ids] = interval_mask(timestamps[ids[valid_ids]], intervals)
    result_ids = ids[mask][:k].astype(np.int64, copy=False)
    result_distances = distances[mask][:k].astype(np.float32, copy=False)
    filter_ms = (time.perf_counter_ns() - started) / 1e6
    return {"ids": result_ids, "distances": result_distances,
            "candidates_examined": fetch, "valid_candidates": int(mask.sum()),
            "effective_ef_search": effective, "distance_computations": computations,
            "search_ms": search_ms, "filter_ms": filter_ms, "total_ms": search_ms + filter_ms}


def run_global(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    workload = load_workload(args.workload); truth = np.load(args.ground_truth)
    queries = np.load(args.embeddings / "query_vectors.npy", mmap_mode="r")
    timestamps = np.load(args.timestamps, mmap_mode="r")
    index = faiss.read_index(str(args.index)); faiss.omp_set_num_threads(args.threads)
    rows = []
    for ef_search in parse_ints(args.ef_search):
        for overfetch in parse_ints(args.overfetch):
            for position, record in enumerate(workload):
                result = _global_search(index, queries[record["query_id"]], timestamps,
                                        _intervals(record), record["k"], overfetch, ef_search)
                rows.append({**_base_row(record), "ef_search": ef_search,
                             "effective_ef_search": result["effective_ef_search"],
                             "overfetch": overfetch, "result_count": len(result["ids"]),
                             "candidates_examined": result["candidates_examined"],
                             "valid_candidates": result["valid_candidates"],
                             "distance_computations": result["distance_computations"],
                             "search_ms": result["search_ms"], "filter_ms": result["filter_ms"],
                             "total_latency_ms": result["total_ms"],
                             "recall_at_k": recall(result["ids"], truth["neighbor_ids"][position], record["k"])})
            print(f"varied global ef={ef_search} overfetch={overfetch} complete", flush=True)
    _write_rows(args.output / "queries.csv", rows)
    summary = summarize(rows, "global HNSW plus resolved OR/AND temporal post-filter",
                        {"ef_search": parse_ints(args.ef_search), "overfetch": parse_ints(args.overfetch)})
    _write_json(args.output / "summary.json", summary); return rows, summary


def index_directory(root: Path, strategy: str, count: int, m: int, efc: int) -> Path:
    return root / strategy / f"buckets_{count}_m{m}_efc{efc}"


def run_build(args: argparse.Namespace) -> list[dict[str, Any]]:
    vectors = np.load(args.embeddings / "database_vectors.npy", mmap_mode="r")
    timestamps = np.load(args.timestamps, mmap_mode="r")
    manifests = []
    for strategy in [x.strip() for x in args.strategies.split(",") if x.strip()]:
        for count in parse_ints(args.bucket_counts):
            config = BucketBuildConfig(count, strategy, args.m, args.ef_construction,
                                       args.threads, YEAR_START_MS, YEAR_END_MS)
            output = index_directory(args.index_root, strategy, count, args.m, args.ef_construction)
            manifest = build_bucket_indexes(vectors, timestamps, output, config,
                                            source={"dataset": "DLT-Tweets", "embeddings": str(args.embeddings)},
                                            rebuild=args.rebuild)
            manifests.append(manifest)
            print(f"built {strategy} Twitter buckets={count} at {output}", flush=True)
    return manifests


def run_bucket(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    workload = load_workload(args.workload); truth = np.load(args.ground_truth)
    vectors = np.load(args.embeddings / "database_vectors.npy", mmap_mode="r")
    queries = np.load(args.embeddings / "query_vectors.npy", mmap_mode="r")
    timestamps = np.load(args.timestamps, mmap_mode="r"); faiss.omp_set_num_threads(args.threads)
    rows = []
    indexes = []
    for strategy in [x.strip() for x in args.strategies.split(",") if x.strip()]:
        for count in parse_ints(args.bucket_counts):
            directory = index_directory(args.index_root, strategy, count, args.m, args.ef_construction)
            bucket_index = BucketedHNSW(directory); bucket_index.preload()
            indexes.append({
                "strategy": strategy, "bucket_count": count, "directory": str(directory),
                "construction_seconds": bucket_index.manifest["construction_seconds"],
                "index_size_bytes": bucket_index.manifest["index_size_bytes"],
                "mapping_size_bytes": bucket_index.manifest["mapping_size_bytes"],
                "timestamps_size_bytes": bucket_index.manifest["timestamps_size_bytes"],
            })
            for ef_search in parse_ints(args.ef_search):
                for overfetch in parse_ints(args.overfetch):
                    config = BucketSearchConfig(ef_search, overfetch, args.boundary_safety)
                    configuration_started = time.perf_counter()
                    for position, record in enumerate(workload):
                        result = bucket_index.search(queries[record["query_id"]], timestamps, vectors,
                                                     _intervals(record), record["k"], config)
                        rows.append({**_base_row(record), "bucket_strategy": strategy,
                                     "bucket_count": count, "ef_search": ef_search,
                                     "overfetch": overfetch, "result_count": len(result["ids"]),
                                     "selected_bucket_count": result["selected_bucket_count"],
                                     "partial_bucket_count": result["partial_bucket_count"],
                                     "candidates_examined": result["candidates_requested"],
                                     "valid_candidates": result["valid_candidates"],
                                     "distance_computations": result["distance_computations"],
                                     "effective_ef_search": result["effective_ef_search"],
                                     "routing_ms": result["routing_ms"], "search_ms": result["search_ms"],
                                     "filter_ms": result["filter_ms"], "rerank_ms": result["rerank_ms"],
                                     "total_latency_ms": result["total_ms"],
                                     "recall_at_k": recall(result["ids"], truth["neighbor_ids"][position], record["k"])})
                        completed = position + 1
                        if args.progress_every and completed % args.progress_every == 0:
                            elapsed = time.perf_counter() - configuration_started
                            eta = elapsed * (len(workload) - completed) / completed
                            _write_rows(args.output / "queries.partial.csv", rows)
                            print(
                                f"bucket {strategy} count={count} ef={ef_search} overfetch={overfetch}: "
                                f"{completed}/{len(workload)} ({100 * completed / len(workload):.1f}%), "
                                f"elapsed={elapsed / 60:.1f}m eta={eta / 60:.1f}m",
                                flush=True,
                            )
                    print(f"bucket {strategy} count={count} ef={ef_search} overfetch={overfetch} complete", flush=True)
    _write_rows(args.output / "queries.csv", rows)
    (args.output / "queries.partial.csv").unlink(missing_ok=True)
    summary = summarize(rows, "bucketed temporal HNSW", {"strategies": args.strategies,
                        "bucket_counts": parse_ints(args.bucket_counts), "ef_search": parse_ints(args.ef_search),
                        "overfetch": parse_ints(args.overfetch), "boundary_safety": args.boundary_safety,
                        "indexes": indexes})
    _write_json(args.output / "summary.json", summary); return rows, summary


def run_adaptive(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    workload = load_workload(args.workload); truth = np.load(args.ground_truth)
    vectors = np.load(args.embeddings / "database_vectors.npy", mmap_mode="r")
    queries = np.load(args.embeddings / "query_vectors.npy", mmap_mode="r")
    timestamps = np.load(args.timestamps, mmap_mode="r")
    order = np.argsort(timestamps, kind="stable"); sorted_times, sorted_ids = timestamps[order], order.astype(np.int64)
    directory = index_directory(args.index_root, args.strategy, args.bucket_count, args.m, args.ef_construction)
    bucket_index = BucketedHNSW(directory); bucket_index.preload()
    planner_config = PlannerConfig(args.exact_threshold, args.bucket_selectivity_threshold, args.max_buckets)
    planner = AdaptivePlanner(bucket_index, planner_config)
    global_index = faiss.read_index(str(args.global_index)); faiss.omp_set_num_threads(args.threads)
    bucket_config = BucketSearchConfig(args.ef_search, args.overfetch, args.boundary_safety)
    rows = []
    for position, record in enumerate(workload):
        effective_intervals = _intervals(record)
        plan_started = time.perf_counter_ns(); method, estimate = planner.choose(effective_intervals)
        planner_ms = (time.perf_counter_ns() - plan_started) / 1e6
        if method == "exact_pre_filter":
            started = time.perf_counter_ns()
            eligible = eligible_ids_from_sorted(sorted_times, sorted_ids, effective_intervals)
            ids, distances = exact_filtered_search(vectors, queries[record["query_id"]], eligible, record["k"], args.block_size)
            execution_ms = (time.perf_counter_ns() - started) / 1e6; candidates = len(eligible)
        elif method == "bucket_hnsw":
            result = bucket_index.search(queries[record["query_id"]], timestamps, vectors,
                                         effective_intervals, record["k"], bucket_config)
            ids, distances, execution_ms, candidates = result["ids"], result["distances"], result["total_ms"], result["candidates_requested"]
        else:
            result = _global_search(global_index, queries[record["query_id"]], timestamps,
                                    effective_intervals, record["k"], args.overfetch, args.ef_search)
            ids, distances, execution_ms, candidates = result["ids"], result["distances"], result["total_ms"], result["candidates_examined"]
        rows.append({**_base_row(record), "chosen_method": method,
                     "selected_bucket_count": estimate["selected_bucket_count"],
                     "estimated_eligible_count": estimate["estimated_eligible_count"],
                     "planner_ms": planner_ms, "execution_ms": execution_ms,
                     "total_latency_ms": planner_ms + execution_ms, "candidates_examined": candidates,
                     "result_count": len(ids), "recall_at_k": recall(ids, truth["neighbor_ids"][position], record["k"])})
        if args.progress_every and (position + 1) % args.progress_every == 0:
            print(f"adaptive: {position + 1}/{len(workload)}", flush=True)
    _write_rows(args.output / "queries.csv", rows)
    methods = {name: sum(row["chosen_method"] == name for row in rows)
               for name in ("exact_pre_filter", "bucket_hnsw", "global_hnsw_post_filter")}
    summary = summarize(rows, "adaptive exact/bucket/global planner",
                        {**asdict_safe(planner_config), "method_counts": methods})
    _write_json(args.output / "summary.json", summary); return rows, summary


def asdict_safe(config: PlannerConfig) -> dict[str, Any]:
    return {"exact_candidate_threshold": config.exact_candidate_threshold,
            "bucket_selectivity_threshold": config.bucket_selectivity_threshold,
            "max_selected_buckets": config.max_selected_buckets}


def common_workload_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--workload", type=Path, default=Path("Twitter_data/workloads/temporal_vector_queries_9600.jsonl"))
    parser.add_argument("--embeddings", type=Path, default=Path("Twitter_data/processed"))
    parser.add_argument("--timestamps", type=Path, default=Path("Twitter_data/ann/timestamps_ms.npy"))
    parser.add_argument("--progress-every", type=int, default=100)
    return parser


def exact_parser() -> argparse.ArgumentParser:
    parser = common_workload_parser("Build exact ground truth for varied Twitter queries")
    parser.add_argument("--output", type=Path, default=Path("results/twitter_varied_exact"))
    parser.add_argument("--block-size", type=int, default=65_536)
    return parser


def global_parser() -> argparse.ArgumentParser:
    parser = common_workload_parser("Benchmark global HNSW on varied Twitter queries")
    parser.add_argument("--ground-truth", type=Path, default=Path("results/twitter_varied_exact/ground_truth.npz"))
    parser.add_argument("--index", type=Path, default=Path("Twitter_data/ann/hnsw_m32_efc200.index"))
    parser.add_argument("--output", type=Path, default=Path("results/twitter_varied_global_hnsw"))
    parser.add_argument("--ef-search", default="256,512"); parser.add_argument("--overfetch", default="10,50,100")
    parser.add_argument("--threads", type=int, default=4)
    return parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build fixed-width and density-balanced Twitter HNSW buckets")
    parser.add_argument("--embeddings", type=Path, default=Path("Twitter_data/processed"))
    parser.add_argument("--timestamps", type=Path, default=Path("Twitter_data/ann/timestamps_ms.npy"))
    parser.add_argument("--index-root", type=Path, default=Path("Twitter_data/ann/buckets"))
    parser.add_argument("--strategies", default="fixed,density"); parser.add_argument("--bucket-counts", default="32,64,128")
    parser.add_argument("--m", type=int, default=32); parser.add_argument("--ef-construction", type=int, default=200)
    parser.add_argument("--threads", type=int, default=4); parser.add_argument("--rebuild", action="store_true")
    return parser


def bucket_parser() -> argparse.ArgumentParser:
    parser = common_workload_parser("Benchmark bucketed Twitter HNSW")
    parser.add_argument("--ground-truth", type=Path, default=Path("results/twitter_varied_exact/ground_truth.npz"))
    parser.add_argument("--index-root", type=Path, default=Path("Twitter_data/ann/buckets"))
    parser.add_argument("--output", type=Path, default=Path("results/twitter_bucket_hnsw"))
    parser.add_argument("--strategies", default="fixed,density"); parser.add_argument("--bucket-counts", default="32,64,128")
    parser.add_argument("--m", type=int, default=32); parser.add_argument("--ef-construction", type=int, default=200)
    parser.add_argument("--ef-search", default="64,128,256"); parser.add_argument("--overfetch", default="1,2,5,10")
    parser.add_argument("--boundary-safety", type=float, default=1.25); parser.add_argument("--threads", type=int, default=4)
    return parser


def adaptive_parser() -> argparse.ArgumentParser:
    parser = common_workload_parser("Benchmark adaptive Twitter temporal ANN planner")
    parser.add_argument("--ground-truth", type=Path, default=Path("results/twitter_varied_exact/ground_truth.npz"))
    parser.add_argument("--index-root", type=Path, default=Path("Twitter_data/ann/buckets"))
    parser.add_argument("--global-index", type=Path, default=Path("Twitter_data/ann/hnsw_m32_efc200.index"))
    parser.add_argument("--output", type=Path, default=Path("results/twitter_adaptive"))
    parser.add_argument("--strategy", default="density", choices=["fixed", "density"]); parser.add_argument("--bucket-count", type=int, default=64)
    parser.add_argument("--m", type=int, default=32); parser.add_argument("--ef-construction", type=int, default=200)
    parser.add_argument("--ef-search", type=int, default=128); parser.add_argument("--overfetch", type=int, default=5)
    parser.add_argument("--boundary-safety", type=float, default=1.25); parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--exact-threshold", type=int, default=5000); parser.add_argument("--bucket-selectivity-threshold", type=float, default=0.15)
    parser.add_argument("--max-buckets", type=int, default=16); parser.add_argument("--block-size", type=int, default=65_536)
    return parser


def exact_main() -> None: run_exact(exact_parser().parse_args())
def global_main() -> None: run_global(global_parser().parse_args())
def build_main() -> None: run_build(build_parser().parse_args())
def bucket_main() -> None: run_bucket(bucket_parser().parse_args())
def adaptive_main() -> None: run_adaptive(adaptive_parser().parse_args())
