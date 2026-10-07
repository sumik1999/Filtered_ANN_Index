from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import h5py
import numpy as np

from .core import TemporalPredicate, top_k
from .data import fragmented_predicate


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_workload(
    query_count: int, available_queries: int, seed: int
) -> list[dict]:
    if query_count > available_queries:
        raise ValueError("query_count exceeds the available SIFT test vectors")

    rng = np.random.default_rng(seed)
    query_ids = rng.choice(available_queries, size=query_count, replace=False)
    predicates: list[TemporalPredicate] = []

    # These ten predicates guarantee that the workload union covers [0, 1).
    # Each contains four separated cells, and the ten predicates cover all 40 cells.
    guaranteed = min(10, query_count)
    for offset in range(guaranteed):
        predicates.append(
            TemporalPredicate.from_intervals(
                [
                    ((offset + quarter * 10) / 40, (offset + quarter * 10 + 1) / 40)
                    for quarter in range(4)
                ]
            )
        )

    # Remaining predicates are randomly positioned to avoid alignment bias.
    for _ in range(guaranteed, query_count):
        predicates.append(fragmented_predicate(0.10, 4, rng))

    permutation = rng.permutation(query_count)
    predicates = [predicates[index] for index in permutation]
    return [
        {
            "workload_id": index,
            "query_id": int(query_ids[index]),
            "total_width": 0.10,
            "fragment_count": 4,
            "boolean_operator": "OR",
            "intervals": [list(interval) for interval in predicates[index].intervals],
        }
        for index in range(query_count)
    ]


def exact_filtered_search(
    database: np.ndarray,
    query: np.ndarray,
    eligible_ids: np.ndarray,
    max_k: int,
    block_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    best_ids = np.empty(0, dtype=np.int64)
    best_distances = np.empty(0, dtype=np.float32)

    for start in range(0, len(eligible_ids), block_size):
        ids = eligible_ids[start : start + block_size]
        vectors = database[ids]
        delta = vectors - query
        distances = np.einsum("ij,ij->i", delta, delta, optimize=True)
        combined_ids = np.concatenate((best_ids, ids))
        combined_distances = np.concatenate((best_distances, distances))
        best_ids, best_distances = top_k(combined_ids, combined_distances, max_k)

    return best_ids.astype(np.int32), best_distances.astype(np.float32)


def save_workload(path: Path, workload: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in workload:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")


def generate(args: argparse.Namespace) -> dict:
    args.output.mkdir(parents=True, exist_ok=True)
    args.workload_output.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.sift_file, "r") as sift:
        database_shape = sift["train"].shape
        test_shape = sift["test"].shape
        if database_shape != (1_000_000, 128) or test_shape != (10_000, 128):
            raise ValueError(
                f"unexpected SIFT shapes: train={database_shape}, test={test_shape}"
            )
        workload = build_workload(args.query_count, test_shape[0], args.seed)
        workload_path = args.workload_output
        save_workload(workload_path, workload)

        # Loading once avoids repeated HDF5 random-access overhead during exact scans.
        database = np.asarray(sift["train"], dtype=np.float32)
        all_queries = np.asarray(sift["test"], dtype=np.float32)
        queries = all_queries[[record["query_id"] for record in workload]]

    timestamp_metadata = json.loads(
        (args.timestamps / "metadata.json").read_text(encoding="utf-8")
    )
    distributions = timestamp_metadata["distributions"]
    metadata = {
        "schema_version": 1,
        "status": "building",
        "source": {
            "sift_file": str(args.sift_file),
            "sift_sha256": file_sha256(args.sift_file),
            "database_array": "train",
            "query_array": "test",
        },
        "workload": {
            "file": str(workload_path),
            "sha256": file_sha256(workload_path),
            "query_count": args.query_count,
            "query_selection": "random without replacement from SIFT test, then fixed",
            "seed": args.seed,
            "total_width_per_query": 0.10,
            "fragment_count": 4,
            "fragment_width": 0.025,
            "boolean_operator": "OR",
            "overall_timeline_coverage": "[0, 1)",
        },
        "oracle": {
            "algorithm": "exact filtered scan",
            "distance": "squared Euclidean (squared L2)",
            "max_k": args.max_k,
            "block_size": args.block_size,
            "padding_neighbor_id": -1,
            "padding_distance": "infinity",
        },
        "distributions": {},
    }
    metadata_path = args.output / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    for distribution_name, distribution_info in distributions.items():
        timestamp_path = args.timestamps / distribution_info["file"]
        timestamps = np.load(timestamp_path, mmap_mode="r")
        if timestamps.shape != (len(database),) or timestamps.dtype != np.float64:
            raise ValueError(f"invalid timestamp array: {timestamp_path}")

        neighbor_ids = np.full(
            (args.query_count, args.max_k), -1, dtype=np.int32
        )
        distances = np.full(
            (args.query_count, args.max_k), np.inf, dtype=np.float32
        )
        qualifying_counts = np.empty(args.query_count, dtype=np.int32)
        started = time.perf_counter()

        for index, (record, query) in enumerate(zip(workload, queries)):
            predicate = TemporalPredicate.from_intervals(record["intervals"])
            eligible_ids = np.flatnonzero(predicate.contains(timestamps))
            qualifying_counts[index] = len(eligible_ids)
            ids, exact_distances = exact_filtered_search(
                database, query, eligible_ids, args.max_k, args.block_size
            )
            neighbor_ids[index, : len(ids)] = ids
            distances[index, : len(ids)] = exact_distances
            if (index + 1) % args.progress_every == 0 or index + 1 == args.query_count:
                elapsed = time.perf_counter() - started
                print(
                    f"{distribution_name}: {index + 1}/{args.query_count} "
                    f"queries ({elapsed:.1f}s)",
                    flush=True,
                )

        elapsed = time.perf_counter() - started
        actual_selectivity = qualifying_counts.astype(np.float64) / len(database)
        output_path = args.output / f"{distribution_name}_ground_truth.npz"
        np.savez_compressed(
            output_path,
            query_ids=np.asarray([x["query_id"] for x in workload], dtype=np.int32),
            neighbor_ids=neighbor_ids,
            squared_l2_distances=distances,
            qualifying_counts=qualifying_counts,
            actual_selectivity=actual_selectivity,
        )
        metadata["distributions"][distribution_name] = {
            "timestamp_file": str(timestamp_path),
            "timestamp_sha256": distribution_info["sha256"],
            "ground_truth_file": output_path.name,
            "ground_truth_sha256": file_sha256(output_path),
            "construction_seconds": elapsed,
            "minimum_qualifying_count": int(qualifying_counts.min()),
            "maximum_qualifying_count": int(qualifying_counts.max()),
            "mean_selectivity": float(actual_selectivity.mean()),
        }
        metadata_path.write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )

    metadata["status"] = "complete"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build exact temporal SIFT1M ground truth")
    parser.add_argument(
        "--sift-file",
        type=Path,
        default=Path("SIFT_data/sift-128-euclidean.hdf5"),
    )
    parser.add_argument(
        "--timestamps", type=Path, default=Path("SIFT_data/timestamps")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("SIFT_data/ground_truth")
    )
    parser.add_argument(
        "--workload-output",
        type=Path,
        default=Path("SIFT_data/workloads/temporal_queries.jsonl"),
    )
    parser.add_argument("--query-count", type=int, default=1000)
    parser.add_argument("--max-k", type=int, default=100)
    parser.add_argument("--block-size", type=int, default=65_536)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=7)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    metadata = generate(args)
    print(
        f"Exact oracle status: {metadata['status']}; output: {args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
