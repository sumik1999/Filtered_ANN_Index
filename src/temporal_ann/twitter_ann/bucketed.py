from __future__ import annotations

import json
import math
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import faiss
import numpy as np


@dataclass(frozen=True)
class BucketBuildConfig:
    bucket_count: int
    strategy: str = "fixed"
    m: int = 32
    ef_construction: int = 200
    threads: int = 4
    timeline_start: int = 1609459200000
    timeline_end: int = 1640995200000

    def validate(self) -> None:
        if self.bucket_count <= 0 or self.strategy not in {"fixed", "density"}:
            raise ValueError("bucket_count must be positive and strategy fixed or density")
        if min(self.m, self.ef_construction, self.threads) <= 0:
            raise ValueError("HNSW construction parameters must be positive")
        if self.timeline_start >= self.timeline_end:
            raise ValueError("invalid timeline")


@dataclass(frozen=True)
class BucketSearchConfig:
    ef_search: int = 128
    overfetch: int = 2
    boundary_safety: float = 1.0

    def validate(self) -> None:
        if self.ef_search <= 0 or self.overfetch <= 0 or self.boundary_safety < 1.0:
            raise ValueError("invalid bucket search parameters")


@dataclass(frozen=True)
class PlannerConfig:
    exact_candidate_threshold: int = 5_000
    bucket_selectivity_threshold: float = 0.15
    max_selected_buckets: int = 16

    def validate(self) -> None:
        if self.exact_candidate_threshold < 0 or self.max_selected_buckets <= 0:
            raise ValueError("invalid planner thresholds")
        if not 0.0 <= self.bucket_selectivity_threshold <= 1.0:
            raise ValueError("bucket selectivity threshold must be in [0,1]")


def normalize_intervals(intervals: Iterable[Iterable[int]]) -> list[tuple[int, int]]:
    ordered = sorted((int(start), int(end)) for start, end in intervals)
    if any(start >= end for start, end in ordered):
        raise ValueError("every interval must have start < end")
    merged: list[tuple[int, int]] = []
    for start, end in ordered:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def interval_mask(timestamps: np.ndarray, intervals: Iterable[Iterable[int]]) -> np.ndarray:
    values = np.asarray(timestamps)
    mask = np.zeros(len(values), dtype=bool)
    for start, end in normalize_intervals(intervals):
        mask |= (values >= start) & (values < end)
    return mask


def eligible_ids_from_sorted(
    sorted_timestamps: np.ndarray,
    sorted_global_ids: np.ndarray,
    intervals: Iterable[Iterable[int]],
) -> np.ndarray:
    parts = []
    for start, end in normalize_intervals(intervals):
        left = np.searchsorted(sorted_timestamps, start, side="left")
        right = np.searchsorted(sorted_timestamps, end, side="left")
        parts.append(np.asarray(sorted_global_ids[left:right], dtype=np.int64))
    return np.sort(np.concatenate(parts)) if parts else np.empty(0, dtype=np.int64)


def make_boundaries(timestamps: np.ndarray, config: BucketBuildConfig) -> np.ndarray:
    config.validate()
    values = np.asarray(timestamps, dtype=np.int64)
    if values.ndim != 1 or np.any(values < config.timeline_start) or np.any(values >= config.timeline_end):
        raise ValueError("timestamps must be a vector within the configured timeline")
    if config.strategy == "fixed":
        boundaries = np.linspace(
            config.timeline_start, config.timeline_end, config.bucket_count + 1,
            dtype=np.int64,
        )
    else:
        sorted_values = np.sort(values)
        positions = np.linspace(0, len(sorted_values), config.bucket_count + 1, dtype=np.int64)
        boundaries = np.empty(config.bucket_count + 1, dtype=np.int64)
        boundaries[0], boundaries[-1] = config.timeline_start, config.timeline_end
        for position in range(1, config.bucket_count):
            candidate = int(sorted_values[min(positions[position], len(sorted_values) - 1)])
            # Equal timestamps are never split: right-sided assignment places all
            # values equal to this boundary in the following bucket.
            boundaries[position] = candidate
        if np.any(np.diff(boundaries) <= 0):
            raise ValueError("cannot create strictly increasing density boundaries; reduce bucket count")
    if np.any(np.diff(boundaries) <= 0):
        raise ValueError("bucket boundaries are not strictly increasing")
    return boundaries


def assign_buckets(timestamps: np.ndarray, boundaries: np.ndarray) -> np.ndarray:
    values = np.asarray(timestamps, dtype=np.int64)
    result = np.searchsorted(boundaries, values, side="right") - 1
    if np.any(result < 0) or np.any(result >= len(boundaries) - 1):
        raise ValueError("timestamp outside bucket boundaries")
    return result.astype(np.int32)


def _hour_histogram(timestamps: np.ndarray) -> dict[str, int]:
    if not len(timestamps):
        return {}
    hours, counts = np.unique(np.asarray(timestamps, dtype=np.int64) // 3_600_000, return_counts=True)
    return {str(int(hour)): int(count) for hour, count in zip(hours, counts)}


def build_bucket_indexes(
    vectors: np.ndarray,
    timestamps: np.ndarray,
    output: Path,
    config: BucketBuildConfig,
    *,
    source: dict[str, Any] | None = None,
    rebuild: bool = False,
) -> dict[str, Any]:
    config.validate()
    vectors = np.asarray(vectors)
    timestamps = np.asarray(timestamps, dtype=np.int64)
    if vectors.ndim != 2 or vectors.dtype != np.float32 or len(vectors) != len(timestamps):
        raise ValueError("vectors must be float32 and aligned with timestamps")
    output = Path(output)
    if output.exists() and not rebuild:
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        if manifest["config"] != asdict(config):
            raise ValueError("persisted bucket configuration does not match")
        return manifest

    boundaries = make_boundaries(timestamps, config)
    assignments = assign_buckets(timestamps, boundaries)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    faiss.omp_set_num_threads(config.threads)
    started = time.perf_counter()
    records: list[dict[str, Any]] = []
    try:
        for bucket_id in range(config.bucket_count):
            ids = np.flatnonzero(assignments == bucket_id).astype(np.int64)
            bucket_times = np.sort(timestamps[ids])
            ids_name = f"bucket_{bucket_id:04d}_ids.npy"
            times_name = f"bucket_{bucket_id:04d}_timestamps.npy"
            index_name = f"bucket_{bucket_id:04d}.index"
            np.save(temporary / ids_name, ids, allow_pickle=False)
            np.save(temporary / times_name, bucket_times, allow_pickle=False)
            index_size = 0
            if len(ids):
                index = faiss.IndexHNSWFlat(vectors.shape[1], config.m, faiss.METRIC_L2)
                index.hnsw.efConstruction = config.ef_construction
                index.add(np.ascontiguousarray(vectors[ids]))
                faiss.write_index(index, str(temporary / index_name))
                index_size = (temporary / index_name).stat().st_size
            records.append({
                "bucket_id": bucket_id,
                "start": int(boundaries[bucket_id]),
                "end": int(boundaries[bucket_id + 1]),
                "vector_count": len(ids),
                "minimum_timestamp": int(bucket_times[0]) if len(ids) else None,
                "maximum_timestamp": int(bucket_times[-1]) if len(ids) else None,
                "mean_records_per_active_hour": (
                    float(np.mean(list(_hour_histogram(bucket_times).values()))) if len(ids) else 0.0
                ),
                "hour_counts": _hour_histogram(bucket_times),
                "index_file": index_name if len(ids) else None,
                "ids_file": ids_name,
                "timestamps_file": times_name,
                "index_size_bytes": index_size,
                "mapping_size_bytes": (temporary / ids_name).stat().st_size,
                "timestamps_size_bytes": (temporary / times_name).stat().st_size,
            })
        manifest = {
            "schema_version": 1,
            "backend": "FAISS IndexHNSWFlat per temporal bucket",
            "metric": "squared L2",
            "vector_count": len(vectors),
            "dimensions": vectors.shape[1],
            "dtype": "float32",
            "config": asdict(config),
            "boundaries": boundaries.tolist(),
            "source": source or {},
            "construction_seconds": time.perf_counter() - started,
            "buckets": records,
        }
        for field in ("index_size_bytes", "mapping_size_bytes", "timestamps_size_bytes"):
            manifest[field] = sum(record[field] for record in records)
        (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        if output.exists():
            shutil.rmtree(output)
        temporary.replace(output)
        return manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


class BucketedHNSW:
    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self.manifest = json.loads((self.directory / "manifest.json").read_text(encoding="utf-8"))
        self.config = BucketBuildConfig(**self.manifest["config"])
        self.boundaries = np.asarray(self.manifest["boundaries"], dtype=np.int64)
        self.buckets = self.manifest["buckets"]
        self.dimensions = int(self.manifest["dimensions"])
        self.vector_count = int(self.manifest["vector_count"])
        self._cache: dict[int, tuple[faiss.Index, np.ndarray, np.ndarray]] = {}
        self._time_cache: dict[int, np.ndarray] = {}

    def _times(self, bucket_id: int) -> np.ndarray:
        if bucket_id not in self._time_cache:
            record = self.buckets[bucket_id]
            self._time_cache[bucket_id] = np.load(
                self.directory / record["timestamps_file"], mmap_mode="r"
            )
        return self._time_cache[bucket_id]

    def _load(self, bucket_id: int) -> tuple[faiss.Index, np.ndarray, np.ndarray] | None:
        record = self.buckets[bucket_id]
        if not record["vector_count"]:
            return None
        if bucket_id not in self._cache:
            index = faiss.read_index(str(self.directory / record["index_file"]))
            ids = np.load(self.directory / record["ids_file"], mmap_mode="r")
            times = self._times(bucket_id)
            if index.ntotal != len(ids) or len(ids) != len(times):
                raise ValueError(f"bucket {bucket_id} artifacts are inconsistent")
            self._cache[bucket_id] = index, ids, times
        return self._cache[bucket_id]

    def preload(self) -> None:
        for bucket_id in range(len(self.buckets)):
            self._load(bucket_id)

    def selected_buckets(self, intervals: Iterable[Iterable[int]]) -> list[int]:
        selected: set[int] = set()
        for start, end in normalize_intervals(intervals):
            clipped_start = max(start, int(self.boundaries[0]))
            clipped_end = min(end, int(self.boundaries[-1]))
            if clipped_start >= clipped_end:
                continue
            first = int(np.searchsorted(self.boundaries, clipped_start, side="right") - 1)
            last = int(np.searchsorted(self.boundaries, clipped_end - 1, side="right") - 1)
            selected.update(range(max(0, first), min(len(self.buckets) - 1, last) + 1))
        return sorted(selected)

    def bucket_coverage(self, bucket_id: int, intervals: Iterable[Iterable[int]]) -> tuple[bool, int]:
        record = self.buckets[bucket_id]
        normalized = normalize_intervals(intervals)
        full = any(start <= record["start"] and end >= record["end"] for start, end in normalized)
        if not record["vector_count"]:
            return full, 0
        times = self._times(bucket_id)
        count = 0
        for start, end in normalized:
            count += int(np.searchsorted(times, end, side="left") - np.searchsorted(times, start, side="left"))
        return full, count

    def estimate(self, intervals: Iterable[Iterable[int]]) -> dict[str, Any]:
        selected = self.selected_buckets(intervals)
        eligible = selected_population = partial = 0
        details = []
        for bucket_id in selected:
            full, count = self.bucket_coverage(bucket_id, intervals)
            population = int(self.buckets[bucket_id]["vector_count"])
            eligible += count; selected_population += population; partial += int(not full)
            details.append({"bucket_id": bucket_id, "full": full, "eligible": count, "population": population})
        return {
            "selected_buckets": selected,
            "selected_bucket_count": len(selected),
            "partial_bucket_count": partial,
            "estimated_eligible_count": eligible,
            "selected_population": selected_population,
            "estimated_selectivity": eligible / self.vector_count,
            "details": details,
        }

    def search(
        self,
        query: np.ndarray,
        timestamps: np.ndarray,
        vectors: np.ndarray,
        intervals: Iterable[Iterable[int]],
        k: int,
        search_config: BucketSearchConfig,
    ) -> dict[str, Any]:
        search_config.validate()
        if k <= 0:
            raise ValueError("k must be positive")
        query = np.asarray(query, dtype=np.float32)
        normalized = normalize_intervals(intervals)
        route_started = time.perf_counter_ns()
        estimate = self.estimate(normalized)
        routing_ms = (time.perf_counter_ns() - route_started) / 1e6
        candidates: list[np.ndarray] = []
        requested = distance_computations = 0
        effective_values = []
        search_started = time.perf_counter_ns()
        for detail in estimate["details"]:
            bucket_id, population, eligible = detail["bucket_id"], detail["population"], detail["eligible"]
            loaded = self._load(bucket_id)
            if loaded is None or eligible == 0:
                continue
            index, global_ids, _ = loaded
            base = k * search_config.overfetch
            if detail["full"]:
                fetch = min(population, base)
            else:
                fraction = eligible / population
                fetch = min(population, max(base, math.ceil(base * search_config.boundary_safety / fraction)))
            effective = max(search_config.ef_search, fetch)
            params = faiss.SearchParametersHNSW(efSearch=effective)
            faiss.cvar.hnsw_stats.reset()
            _, local_ids = index.search(query.reshape(1, -1), fetch, params=params)
            distance_computations += int(faiss.cvar.hnsw_stats.ndis)
            valid = local_ids[0][local_ids[0] >= 0]
            candidates.append(np.asarray(global_ids[valid], dtype=np.int64))
            requested += fetch; effective_values.append(effective)
        search_ms = (time.perf_counter_ns() - search_started) / 1e6

        filter_started = time.perf_counter_ns()
        unique = np.unique(np.concatenate(candidates)) if candidates else np.empty(0, dtype=np.int64)
        valid_ids = unique[interval_mask(np.asarray(timestamps)[unique], normalized)]
        filter_ms = (time.perf_counter_ns() - filter_started) / 1e6
        rerank_started = time.perf_counter_ns()
        if len(valid_ids):
            delta = np.asarray(vectors)[valid_ids].astype(np.float32, copy=False) - query
            distances = np.einsum("ij,ij->i", delta, delta)
            count = min(k, len(valid_ids))
            positions = np.argpartition(distances, count - 1)[:count]
            positions = positions[np.argsort(distances[positions], kind="stable")]
            result_ids = valid_ids[positions]
            result_distances = distances[positions].astype(np.float32, copy=False)
        else:
            result_ids = np.empty(0, dtype=np.int64); result_distances = np.empty(0, dtype=np.float32)
        rerank_ms = (time.perf_counter_ns() - rerank_started) / 1e6
        return {
            "ids": result_ids, "distances": result_distances,
            **{key: value for key, value in estimate.items() if key != "details"},
            "candidates_requested": requested, "unique_candidates": len(unique),
            "valid_candidates": len(valid_ids), "distance_computations": distance_computations,
            "effective_ef_search": max(effective_values, default=search_config.ef_search),
            "routing_ms": routing_ms, "search_ms": search_ms, "filter_ms": filter_ms,
            "rerank_ms": rerank_ms,
            "total_ms": routing_ms + search_ms + filter_ms + rerank_ms,
        }


class AdaptivePlanner:
    def __init__(self, bucket_index: BucketedHNSW, config: PlannerConfig):
        config.validate(); self.bucket_index = bucket_index; self.config = config

    def choose(self, intervals: Iterable[Iterable[int]]) -> tuple[str, dict[str, Any]]:
        estimate = self.bucket_index.estimate(intervals)
        if estimate["estimated_eligible_count"] <= self.config.exact_candidate_threshold:
            method = "exact_pre_filter"
        elif (estimate["estimated_selectivity"] <= self.config.bucket_selectivity_threshold
              and estimate["selected_bucket_count"] <= self.config.max_selected_buckets):
            method = "bucket_hnsw"
        else:
            method = "global_hnsw_post_filter"
        return method, estimate
