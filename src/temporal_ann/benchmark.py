from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from .core import exact_search
from .data import fragmented_predicate, generate_dataset, query_for_predicate
from .methods import FixedTemporalPartition, GlobalPostFilterProxy, SearchMethod, TemporalPreFilter


def parse_list(value: str, cast: Callable[[str], object]) -> list:
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def recall_at_k(returned: np.ndarray, truth: np.ndarray) -> float:
    if len(truth) == 0:
        return 1.0 if len(returned) == 0 else 0.0
    return len(set(returned.tolist()).intersection(truth.tolist())) / len(truth)


def percentile(values: Iterable[float], quantile: float) -> float:
    return float(np.percentile(np.fromiter(values, dtype=np.float64), quantile))


def run(args: argparse.Namespace) -> tuple[list[dict], list[dict], list[dict]]:
    rng = np.random.default_rng(args.seed)
    rows: list[dict] = []
    builds: list[dict] = []
    datasets: list[dict] = []
    args.data_dir.mkdir(parents=True, exist_ok=True)

    for density in args.densities:
        for correlation in args.correlations:
            dataset_seed = int(rng.integers(0, 2**32 - 1))
            dataset = generate_dataset(
                args.size, args.dimensions, correlation, density, dataset_seed
            )
            dataset_name = (
                f"{density}_corr-{correlation:g}_n-{args.size}_d-{args.dimensions}"
                f"_seed-{dataset_seed}.npz"
            )
            dataset_path = args.data_dir / dataset_name
            np.savez_compressed(
                dataset_path,
                vectors=dataset.vectors,
                timestamps=dataset.timestamps,
            )
            datasets.append(
                {
                    "file": dataset_name,
                    "size": args.size,
                    "dimensions": args.dimensions,
                    "density": density,
                    "correlation": correlation,
                    "seed": dataset_seed,
                }
            )

            factories: list[tuple[str, Callable[[], SearchMethod]]] = [
                ("temporal_pre_filter", lambda: TemporalPreFilter(dataset)),
                (
                    "global_post_filter_proxy",
                    lambda: GlobalPostFilterProxy(dataset, args.overfetch),
                ),
                (
                    "fixed_temporal_partition",
                    lambda: FixedTemporalPartition(dataset, args.buckets),
                ),
            ]
            methods: list[SearchMethod] = []
            for name, factory in factories:
                started = time.perf_counter_ns()
                method = factory()
                build_ms = (time.perf_counter_ns() - started) / 1e6
                methods.append(method)
                builds.append(
                    {
                        "method": name,
                        "density": density,
                        "correlation": correlation,
                        "build_ms": build_ms,
                        "index_size_bytes": method.index_size_bytes,
                    }
                )

            for width in args.widths:
                for fragments in args.fragments:
                    if fragments > 1 and width >= 1.0:
                        continue
                    for k in args.ks:
                        for query_number in range(args.queries):
                            predicate = fragmented_predicate(width, fragments, rng)
                            query = query_for_predicate(dataset, predicate, rng)
                            truth = exact_search(dataset, query, predicate, k)
                            actual_selectivity = truth.candidates_examined / len(dataset.vectors)

                            for method in methods:
                                started = time.perf_counter_ns()
                                result = method.search(query, predicate, k)
                                latency_ms = (time.perf_counter_ns() - started) / 1e6
                                rows.append(
                                    {
                                        "method": method.name,
                                        "density": density,
                                        "correlation": correlation,
                                        "target_width": width,
                                        "actual_selectivity": actual_selectivity,
                                        "fragments": fragments,
                                        "k": k,
                                        "query": query_number,
                                        "latency_ms": latency_ms,
                                        "recall_at_k": recall_at_k(result.ids, truth.ids),
                                        "result_count": len(result.ids),
                                        "candidates_examined": result.candidates_examined,
                                        "index_size_bytes": method.index_size_bytes,
                                        "intervals": json.dumps(predicate.intervals),
                                    }
                                )
    return rows, builds, datasets


def summarize(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    keys = ("method", "density", "correlation", "target_width", "fragments", "k")
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)

    summary: list[dict] = []
    for group, members in grouped.items():
        item = dict(zip(keys, group))
        item.update(
            {
                "queries": len(members),
                "mean_selectivity": float(np.mean([x["actual_selectivity"] for x in members])),
                "mean_recall_at_k": float(np.mean([x["recall_at_k"] for x in members])),
                "mean_latency_ms": float(np.mean([x["latency_ms"] for x in members])),
                "p50_latency_ms": percentile((x["latency_ms"] for x in members), 50),
                "p95_latency_ms": percentile((x["latency_ms"] for x in members), 95),
                "mean_candidates_examined": float(
                    np.mean([x["candidates_examined"] for x in members])
                ),
                "index_size_bytes": members[0]["index_size_bytes"],
            }
        )
        summary.append(item)
    return summary


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark temporal vector search strategies")
    parser.add_argument("--size", type=int, default=10_000)
    parser.add_argument("--dimensions", type=int, default=32)
    parser.add_argument("--queries", type=int, default=10, help="queries per workload cell")
    parser.add_argument("--widths", default="0.01,0.1,0.5")
    parser.add_argument("--fragments", default="1,4")
    parser.add_argument("--ks", default="10,50")
    parser.add_argument("--correlations", default="0.0,0.8")
    parser.add_argument("--densities", default="uniform,clustered")
    parser.add_argument("--buckets", type=int, default=32)
    parser.add_argument("--overfetch", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--data-dir", type=Path, default=Path("synthetic_data"),
        help="directory for generated vector and timestamp datasets",
    )
    parser.add_argument("--output", type=Path, default=Path("results"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.widths = parse_list(args.widths, float)
    args.fragments = parse_list(args.fragments, int)
    args.ks = parse_list(args.ks, int)
    args.correlations = parse_list(args.correlations, float)
    args.densities = parse_list(args.densities, str)

    args.output.mkdir(parents=True, exist_ok=True)
    rows, builds, datasets = run(args)
    summary = summarize(rows)
    write_csv(args.output / "queries.csv", rows)
    write_csv(args.output / "builds.csv", builds)
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (args.data_dir / "manifest.json").write_text(
        json.dumps(datasets, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved {len(datasets)} synthetic datasets to {args.data_dir}")
    print(f"Wrote {len(rows)} query measurements to {args.output}")


if __name__ == "__main__":
    main()
