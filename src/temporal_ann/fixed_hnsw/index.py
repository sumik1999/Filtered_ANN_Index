from __future__ import annotations

import json
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np


@dataclass(frozen=True)
class FixedHNSWConfig:
    """Construction parameters for fixed-width temporal HNSW indexes."""

    bucket_count: int = 32
    m: int = 32
    ef_construction: int = 200
    threads: int = 4
    timeline_start: float = 0.0
    timeline_end: float = 1.0

    def validate(self) -> None:
        if self.bucket_count <= 0:
            raise ValueError("bucket_count must be positive")
        if self.m <= 0 or self.ef_construction <= 0 or self.threads <= 0:
            raise ValueError("M, efConstruction, and threads must be positive")
        if not self.timeline_start < self.timeline_end:
            raise ValueError("timeline_start must be less than timeline_end")


@dataclass(frozen=True)
class FixedHNSWSearchConfig:
    """Query parameters. overfetch is applied independently to every bucket."""

    ef_search: int = 256
    overfetch: int = 2

    def validate(self) -> None:
        if self.ef_search <= 0 or self.overfetch <= 0:
            raise ValueError("ef_search and overfetch must be positive")


def assign_buckets(timestamps: np.ndarray, config: FixedHNSWConfig) -> np.ndarray:
    """Assign each timestamp to exactly one half-open fixed-width bucket."""
    config.validate()
    values = np.asarray(timestamps, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("timestamps must be one-dimensional")
    if not np.all(np.isfinite(values)):
        raise ValueError("timestamps must be finite")
    if np.any(values < config.timeline_start) or np.any(values >= config.timeline_end):
        raise ValueError("timestamps must lie in [timeline_start, timeline_end)")
    scaled = (values - config.timeline_start) / (
        config.timeline_end - config.timeline_start
    )
    return np.minimum(
        (scaled * config.bucket_count).astype(np.int64), config.bucket_count - 1
    )


def _validate_vectors(vectors: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
    values = np.asarray(vectors, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError("vectors must be a non-empty two-dimensional array")
    if len(values) != len(timestamps):
        raise ValueError("timestamps must contain one value per vector")
    return np.ascontiguousarray(values)


def build_fixed_hnsw(
    vectors: np.ndarray,
    timestamps: np.ndarray,
    output: Path,
    config: FixedHNSWConfig,
    *,
    source_metadata: dict[str, Any] | None = None,
    rebuild: bool = False,
) -> dict[str, Any]:
    """Build and atomically persist one HNSW index per non-empty bucket."""
    config.validate()
    output = Path(output)
    vectors = _validate_vectors(vectors, timestamps)
    bucket_ids = assign_buckets(timestamps, config)

    if output.exists() and not rebuild:
        manifest_path = output / "manifest.json"
        if not manifest_path.exists():
            raise FileExistsError(f"{output} exists without a manifest; use rebuild=True")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("config") != asdict(config):
            raise ValueError("persisted index configuration does not match requested configuration")
        return manifest

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    faiss.omp_set_num_threads(config.threads)
    started = time.perf_counter()
    buckets: list[dict[str, Any]] = []
    try:
        boundaries = np.linspace(
            config.timeline_start,
            config.timeline_end,
            config.bucket_count + 1,
            dtype=np.float64,
        )
        for bucket in range(config.bucket_count):
            global_ids = np.flatnonzero(bucket_ids == bucket).astype(np.int64)
            record: dict[str, Any] = {
                "bucket_id": bucket,
                "start": float(boundaries[bucket]),
                "end": float(boundaries[bucket + 1]),
                "vector_count": int(len(global_ids)),
                "index_file": None,
                "ids_file": None,
                "index_size_bytes": 0,
                "ids_size_bytes": 0,
            }
            if len(global_ids):
                index = faiss.IndexHNSWFlat(vectors.shape[1], config.m, faiss.METRIC_L2)
                index.hnsw.efConstruction = config.ef_construction
                index.add(np.ascontiguousarray(vectors[global_ids]))
                index_name = f"bucket_{bucket:04d}.index"
                ids_name = f"bucket_{bucket:04d}_ids.npy"
                faiss.write_index(index, str(temporary / index_name))
                np.save(temporary / ids_name, global_ids, allow_pickle=False)
                record.update(
                    {
                        "index_file": index_name,
                        "ids_file": ids_name,
                        "index_size_bytes": (temporary / index_name).stat().st_size,
                        "ids_size_bytes": (temporary / ids_name).stat().st_size,
                    }
                )
            buckets.append(record)

        manifest = {
            "schema_version": 1,
            "backend": "FAISS IndexHNSWFlat per fixed-width temporal bucket",
            "faiss_version": faiss.__version__,
            "metric": "squared L2",
            "vector_count": int(len(vectors)),
            "dimensions": int(vectors.shape[1]),
            "dtype": "float32",
            "id_mapping": "bucket-local FAISS label maps through each bucket's ids_file",
            "config": asdict(config),
            "construction_seconds": time.perf_counter() - started,
            "source": source_metadata or {},
            "buckets": buckets,
        }
        manifest["index_size_bytes"] = sum(x["index_size_bytes"] for x in buckets)
        manifest["mapping_size_bytes"] = sum(x["ids_size_bytes"] for x in buckets)
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        if output.exists():
            shutil.rmtree(output)
        temporary.replace(output)
        return manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


class FixedHNSWIndex:
    """A lazily loaded directory of persistent fixed-width HNSW indexes."""

    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self.manifest = json.loads(
            (self.directory / "manifest.json").read_text(encoding="utf-8")
        )
        self.config = FixedHNSWConfig(**self.manifest["config"])
        self.config.validate()
        self.dimensions = int(self.manifest["dimensions"])
        self.vector_count = int(self.manifest["vector_count"])
        self.buckets = self.manifest["buckets"]
        if len(self.buckets) != self.config.bucket_count:
            raise ValueError("manifest bucket count is inconsistent")
        self._cache: dict[int, tuple[faiss.Index, np.ndarray]] = {}

    def selected_buckets(self, intervals: list[list[float]] | tuple[tuple[float, float], ...]) -> list[int]:
        """Return unique bucket IDs intersecting normalized or overlapping intervals."""
        width = (self.config.timeline_end - self.config.timeline_start) / self.config.bucket_count
        selected: set[int] = set()
        for start_value, end_value in intervals:
            start, end = float(start_value), float(end_value)
            if not start < end:
                raise ValueError("each interval must have start < end")
            start = max(start, self.config.timeline_start)
            end = min(end, self.config.timeline_end)
            if start >= end:
                continue
            first = min(
                self.config.bucket_count - 1,
                max(0, int(np.floor((start - self.config.timeline_start) / width))),
            )
            last_point = np.nextafter(end, -np.inf)
            last = min(
                self.config.bucket_count - 1,
                max(0, int(np.floor((last_point - self.config.timeline_start) / width))),
            )
            selected.update(range(first, last + 1))
        return sorted(selected)

    def _load_bucket(self, bucket_id: int) -> tuple[faiss.Index, np.ndarray] | None:
        record = self.buckets[bucket_id]
        if record["vector_count"] == 0:
            return None
        if bucket_id not in self._cache:
            index = faiss.read_index(str(self.directory / record["index_file"]))
            global_ids = np.load(self.directory / record["ids_file"], mmap_mode="r")
            if index.d != self.dimensions or index.ntotal != len(global_ids):
                raise ValueError(f"bucket {bucket_id} index and mapping are inconsistent")
            if index.metric_type != faiss.METRIC_L2:
                raise ValueError(f"bucket {bucket_id} does not use squared L2")
            self._cache[bucket_id] = (index, global_ids)
        return self._cache[bucket_id]

    def preload(self) -> None:
        """Load every non-empty bucket so disk I/O is excluded from query timings."""
        for bucket_id in range(self.config.bucket_count):
            self._load_bucket(bucket_id)

    def search(
        self,
        query: np.ndarray,
        timestamps: np.ndarray,
        vectors: np.ndarray,
        intervals: list[list[float]] | tuple[tuple[float, float], ...],
        k: int,
        search_config: FixedHNSWSearchConfig,
    ) -> dict[str, Any]:
        """Search selected buckets, hard-filter timestamps, deduplicate, and rerank."""
        search_config.validate()
        if k <= 0:
            raise ValueError("k must be positive")
        query = np.asarray(query, dtype=np.float32)
        if query.shape != (self.dimensions,):
            raise ValueError(f"query must have shape ({self.dimensions},)")
        if len(timestamps) != self.vector_count or len(vectors) != self.vector_count:
            raise ValueError("vectors and timestamps must match the persisted index")

        selected = self.selected_buckets(intervals)
        candidate_parts: list[np.ndarray] = []
        requested = returned = distance_computations = 0
        effective_values: list[int] = []
        search_started = time.perf_counter_ns()
        for bucket_id in selected:
            loaded = self._load_bucket(bucket_id)
            if loaded is None:
                continue
            index, global_ids = loaded
            fetch = min(index.ntotal, k * search_config.overfetch)
            effective_ef = max(search_config.ef_search, fetch)
            params = faiss.SearchParametersHNSW(efSearch=effective_ef)
            faiss.cvar.hnsw_stats.reset()
            _, local_ids = index.search(query.reshape(1, -1), fetch, params=params)
            distance_computations += int(faiss.cvar.hnsw_stats.ndis)
            valid_local = local_ids[0][local_ids[0] >= 0]
            candidate_parts.append(np.asarray(global_ids[valid_local], dtype=np.int64))
            requested += fetch
            returned += len(valid_local)
            effective_values.append(effective_ef)
        search_ms = (time.perf_counter_ns() - search_started) / 1e6

        filter_started = time.perf_counter_ns()
        candidates = (
            np.unique(np.concatenate(candidate_parts))
            if candidate_parts
            else np.empty(0, dtype=np.int64)
        )
        temporal = np.zeros(len(candidates), dtype=bool)
        candidate_times = np.asarray(timestamps)[candidates]
        for start, end in intervals:
            temporal |= (candidate_times >= start) & (candidate_times < end)
        valid_ids = candidates[temporal]
        filter_ms = (time.perf_counter_ns() - filter_started) / 1e6

        rerank_started = time.perf_counter_ns()
        if len(valid_ids):
            selected_vectors = np.asarray(vectors)[valid_ids].astype(np.float32, copy=False)
            delta = selected_vectors - query
            distances = np.einsum("ij,ij->i", delta, delta)
            count = min(k, len(valid_ids))
            positions = np.argpartition(distances, count - 1)[:count]
            positions = positions[np.argsort(distances[positions], kind="stable")]
            result_ids = valid_ids[positions]
            result_distances = distances[positions].astype(np.float32, copy=False)
        else:
            result_ids = np.empty(0, dtype=np.int64)
            result_distances = np.empty(0, dtype=np.float32)
        rerank_ms = (time.perf_counter_ns() - rerank_started) / 1e6

        return {
            "ids": result_ids,
            "distances": result_distances,
            "selected_buckets": selected,
            "candidates_requested": requested,
            "candidates_returned": returned,
            "unique_candidates": int(len(candidates)),
            "valid_candidates": int(len(valid_ids)),
            "distance_computations": distance_computations,
            "effective_ef_search": max(effective_values, default=search_config.ef_search),
            "search_ms": search_ms,
            "filter_ms": filter_ms,
            "rerank_ms": rerank_ms,
            "total_ms": search_ms + filter_ms + rerank_ms,
        }
