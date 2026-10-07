from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import faiss
import h5py
import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_index(index: faiss.Index, dimensions: int, expected_count: int) -> None:
    if index.d != dimensions:
        raise ValueError(f"FAISS dimension mismatch: {index.d} != {dimensions}")
    if index.ntotal != expected_count:
        raise ValueError(f"FAISS vector count mismatch: {index.ntotal} != {expected_count}")
    if index.metric_type != faiss.METRIC_L2:
        raise ValueError("FAISS index must use squared L2 distance")


def build_index(
    sift_file: Path,
    output: Path,
    metadata_path: Path,
    m: int,
    ef_construction: int,
    threads: int,
) -> tuple[faiss.IndexHNSWFlat, dict]:
    if m <= 0 or ef_construction <= 0 or threads <= 0:
        raise ValueError("M, efConstruction, and threads must be positive")

    with h5py.File(sift_file, "r") as sift:
        vectors = np.asarray(sift["train"], dtype=np.float32)
    if vectors.shape != (1_000_000, 128):
        raise ValueError(f"unexpected SIFT database shape: {vectors.shape}")

    faiss.omp_set_num_threads(threads)
    started = time.perf_counter()
    index = faiss.IndexHNSWFlat(vectors.shape[1], m, faiss.METRIC_L2)
    index.hnsw.efConstruction = ef_construction
    index.add(vectors)
    construction_seconds = time.perf_counter() - started
    validate_index(index, 128, 1_000_000)

    output.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(output))
    metadata = {
        "schema_version": 1,
        "backend": "FAISS IndexHNSWFlat",
        "faiss_version": faiss.__version__,
        "source": str(sift_file),
        "vector_count": 1_000_000,
        "dimensions": 128,
        "dtype": "float32",
        "metric": "squared L2",
        "id_mapping": "FAISS label equals SIFT train row ID",
        "M": m,
        "efConstruction": ef_construction,
        "threads": threads,
        "construction_seconds": construction_seconds,
        "index_file": output.name,
        "index_size_bytes": output.stat().st_size,
        "index_sha256": sha256(output),
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return index, metadata


def load_or_build_index(
    sift_file: Path,
    index_path: Path,
    metadata_path: Path,
    m: int,
    ef_construction: int,
    threads: int,
    rebuild: bool = False,
) -> tuple[faiss.IndexHNSWFlat, dict]:
    if rebuild and index_path.exists():
        index_path.unlink()
    if not index_path.exists():
        return build_index(
            sift_file, index_path, metadata_path, m, ef_construction, threads
        )

    started = time.perf_counter()
    index = faiss.read_index(str(index_path))
    load_seconds = time.perf_counter() - started
    validate_index(index, 128, 1_000_000)
    metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.exists()
        else {}
    )
    metadata["last_load_seconds"] = load_seconds
    return index, metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the global FAISS HNSW index")
    parser.add_argument(
        "--sift-file", type=Path, default=Path("SIFT_data/sift-128-euclidean.hdf5")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("SIFT_data/faiss/hnsw_m32_efc200.index")
    )
    parser.add_argument(
        "--metadata", type=Path, default=Path("SIFT_data/faiss/metadata.json")
    )
    parser.add_argument("--m", type=int, default=32)
    parser.add_argument("--ef-construction", type=int, default=200)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--rebuild", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    _, metadata = load_or_build_index(
        args.sift_file,
        args.output,
        args.metadata,
        args.m,
        args.ef_construction,
        args.threads,
        args.rebuild,
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
