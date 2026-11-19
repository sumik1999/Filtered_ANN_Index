from __future__ import annotations

import argparse
import heapq
import json
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import faiss
import numpy as np

from .bucketed import normalize_intervals


@dataclass(frozen=True)
class RollingBuildConfig:
    vector_count: int = 100_000
    semantic_degree: int = 32
    gateway_degree: int = 8
    gateway_period: int = 512
    construction_candidates: int = 64
    ef_construction: int = 200
    candidate_index_m: int = 32
    threads: int = 4
    range_degree_per_side: int = 0
    range_candidate_count: int = 64
    range_window_hours: int = 720
    range_scales_hours: str = ""
    hierarchy_enabled: bool = False
    hierarchy_semantic_degree: int = 8
    hierarchy_level_probability: float = 0.05
    hierarchy_max_level: int = 6
    hierarchy_seed: int = 23
    hierarchy_construction_candidates: int = 32
    hierarchy_range_degree_per_side: int = 0
    hierarchy_range_candidate_count: int = 32
    hierarchy_range_scales_hours: str = ""

    def range_scales(self) -> tuple[int, ...]:
        if self.range_degree_per_side == 0:
            return ()
        if self.range_scales_hours.strip():
            scales = tuple(sorted({
                int(value.strip())
                for value in self.range_scales_hours.split(",")
                if value.strip()
            }))
            if not scales or any(value <= 0 for value in scales):
                raise ValueError("range scales must be comma-separated positive hours")
            return scales
        return (self.range_window_hours,)

    def hierarchy_range_scales(self) -> tuple[int, ...]:
        if self.hierarchy_range_degree_per_side == 0:
            return ()
        value = self.hierarchy_range_scales_hours.strip()
        if value:
            scales = tuple(sorted({
                int(item.strip()) for item in value.split(",") if item.strip()
            }))
        else:
            scales = self.range_scales()
        if not scales or any(scale <= 0 for scale in scales):
            raise ValueError("hierarchy range scales must be positive hours")
        return scales

    def validate(self) -> None:
        values = (
            self.vector_count, self.semantic_degree, self.gateway_degree,
            self.gateway_period, self.construction_candidates,
            self.ef_construction, self.candidate_index_m, self.threads,
            self.range_candidate_count, self.range_window_hours,
            self.hierarchy_semantic_degree, self.hierarchy_max_level,
            self.hierarchy_construction_candidates,
            self.hierarchy_range_candidate_count,
        )
        if (any(value <= 0 for value in values) or self.range_degree_per_side < 0
                or self.hierarchy_range_degree_per_side < 0):
            raise ValueError("rolling-HNSW parameters are invalid")
        if not 0.0 < self.hierarchy_level_probability < 1.0:
            raise ValueError("hierarchy_level_probability must be in (0,1)")
        if self.construction_candidates < self.semantic_degree:
            raise ValueError("construction_candidates must be at least semantic_degree")
        self.range_scales()
        self.hierarchy_range_scales()


@dataclass(frozen=True)
class RollingSearchConfig:
    k: int = 100
    ef_search: int = 128
    exact_threshold: int = 5_000
    gateway_candidates: int = 16
    gateway_seeds: int = 4
    use_temporal_edges: bool = True
    use_gateway_edges: bool = True
    use_range_edges: bool = True
    use_hierarchy: bool = True
    upper_ef_search: int = 32

    def validate(self) -> None:
        if min(self.k, self.ef_search, self.gateway_candidates,
               self.gateway_seeds, self.upper_ef_search) <= 0:
            raise ValueError("search parameters must be positive")
        if self.ef_search < self.k:
            raise ValueError("ef_search must be at least k")
        if self.exact_threshold < 0:
            raise ValueError("exact_threshold must be non-negative")
        if self.gateway_seeds > self.gateway_candidates:
            raise ValueError("gateway_seeds cannot exceed gateway_candidates")


def _add_bounded_edge(
    neighbors: np.ndarray,
    distances: np.ndarray,
    row: int,
    neighbor: int,
    distance: float,
) -> None:
    current = neighbors[row]
    duplicate = np.flatnonzero(current == neighbor)
    if len(duplicate):
        distances[row, int(duplicate[0])] = min(distances[row, int(duplicate[0])], distance)
        return
    empty = np.flatnonzero(current < 0)
    if len(empty):
        position = int(empty[0])
        current[position] = neighbor
        distances[row, position] = distance
        return
    farthest = int(np.argmax(distances[row]))
    if distance < float(distances[row, farthest]):
        current[farthest] = neighbor
        distances[row, farthest] = distance


def _exact_top_k(
    query: np.ndarray,
    vectors: np.ndarray,
    global_ids: np.ndarray,
    local_ids: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    if not len(local_ids):
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32), 0
    database_ids = global_ids[local_ids]
    values = np.asarray(vectors[database_ids], dtype=np.float32)
    delta = values - query
    distances = np.einsum("ij,ij->i", delta, delta)
    count = min(k, len(local_ids))
    positions = np.argpartition(distances, count - 1)[:count]
    positions = positions[np.argsort(distances[positions], kind="stable")]
    return local_ids[positions], distances[positions].astype(np.float32, copy=False), len(local_ids)


def build_rolling_index(
    vectors: np.ndarray,
    timestamps: np.ndarray,
    output: Path,
    config: RollingBuildConfig,
    *,
    source: dict[str, Any] | None = None,
    rebuild: bool = False,
    progress_every: int = 5_000,
) -> dict[str, Any]:
    """Build a flat rolling temporal graph in chronological insertion order.

    FAISS HNSW is used only as an online semantic candidate generator. Query
    traversal uses the persisted custom semantic, temporal and gateway lists.
    """
    config.validate()
    vectors = np.asarray(vectors)
    timestamps = np.asarray(timestamps, dtype=np.int64)
    if vectors.ndim != 2 or vectors.dtype != np.float32 or not len(vectors):
        raise ValueError("vectors must be a non-empty float32 matrix")
    if timestamps.ndim != 1 or len(timestamps) != len(vectors):
        raise ValueError("timestamps must align with vectors")
    count = min(config.vector_count, len(vectors))
    output = Path(output)
    if output.exists() and not rebuild:
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        persisted_config = RollingBuildConfig(**manifest["config"])
        if persisted_config != config:
            raise ValueError("persisted rolling-HNSW configuration does not match")
        return manifest

    # Stable order gives duplicate timestamps a deterministic insertion sequence.
    global_ids = np.argsort(timestamps, kind="stable")[:count].astype(np.int64, copy=False)
    ordered_timestamps = np.asarray(timestamps[global_ids], dtype=np.int64)
    semantic_neighbors = np.full((count, config.semantic_degree), -1, dtype=np.int32)
    semantic_distances = np.full((count, config.semantic_degree), np.inf, dtype=np.float32)
    temporal_neighbors = np.full((count, 2), -1, dtype=np.int32)
    range_scales = config.range_scales()
    range_shape = (count, len(range_scales), config.range_degree_per_side)
    range_left_neighbors = np.full(range_shape, -1, dtype=np.int32)
    range_right_neighbors = np.full(range_shape, -1, dtype=np.int32)
    range_left_scores = np.full(range_shape, np.inf, dtype=np.float32)
    range_right_scores = np.full(range_shape, np.inf, dtype=np.float32)

    levels = np.zeros(count, dtype=np.int8)
    if config.hierarchy_enabled:
        level_rng = np.random.default_rng(config.hierarchy_seed)
        for local_id in range(count):
            level = 0
            while (level < config.hierarchy_max_level and
                   level_rng.random() < config.hierarchy_level_probability):
                level += 1
            levels[local_id] = level
        upper_shape = (
            count, config.hierarchy_max_level, config.hierarchy_semantic_degree
        )
        upper_temporal_shape = (count, config.hierarchy_max_level, 2)
    else:
        upper_shape = (count, 0, 0)
        upper_temporal_shape = (count, 0, 2)
    upper_semantic_neighbors = np.full(upper_shape, -1, dtype=np.int32)
    upper_semantic_distances = np.full(upper_shape, np.inf, dtype=np.float32)
    upper_temporal_neighbors = np.full(upper_temporal_shape, -1, dtype=np.int32)
    hierarchy_range_scales = config.hierarchy_range_scales()
    if config.hierarchy_enabled and hierarchy_range_scales:
        upper_range_shape = (
            count, config.hierarchy_max_level,
            config.hierarchy_range_degree_per_side,
        )
        level_range_scales = tuple(
            hierarchy_range_scales[min(level, len(hierarchy_range_scales) - 1)]
            for level in range(config.hierarchy_max_level)
        )
    else:
        upper_range_shape = (count, 0, 0)
        level_range_scales = ()
    upper_range_left_neighbors = np.full(upper_range_shape, -1, dtype=np.int32)
    upper_range_right_neighbors = np.full(upper_range_shape, -1, dtype=np.int32)
    upper_range_left_scores = np.full(upper_range_shape, np.inf, dtype=np.float32)
    upper_range_right_scores = np.full(upper_range_shape, np.inf, dtype=np.float32)
    last_at_level = np.full(config.hierarchy_max_level, -1, dtype=np.int32)
    level_inserted_ids: list[list[int]] = [
        [] for _ in range(config.hierarchy_max_level)
    ]

    gateway_rows = np.full(count, -1, dtype=np.int32)
    max_gateways = count // config.gateway_period
    gateway_ids = np.empty(max_gateways, dtype=np.int32)
    gateway_neighbors = np.full((max_gateways, config.gateway_degree), -1, dtype=np.int32)
    gateway_distances = np.full((max_gateways, config.gateway_degree), np.inf, dtype=np.float32)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    faiss.omp_set_num_threads(config.threads)
    candidate_index = faiss.IndexHNSWFlat(
        vectors.shape[1], config.candidate_index_m, faiss.METRIC_L2
    )
    candidate_index.hnsw.efConstruction = config.ef_construction
    upper_indexes: list[tuple[faiss.IndexHNSWFlat, faiss.IndexIDMap2]] = []
    if config.hierarchy_enabled:
        for _ in range(config.hierarchy_max_level):
            base = faiss.IndexHNSWFlat(
                vectors.shape[1], config.hierarchy_semantic_degree, faiss.METRIC_L2
            )
            base.hnsw.efConstruction = config.ef_construction
            upper_indexes.append((base, faiss.IndexIDMap2(base)))
    gateway_count = 0
    started = time.perf_counter()
    try:
        for local_id, global_id in enumerate(global_ids):
            vector = np.ascontiguousarray(vectors[global_id]).reshape(1, -1)
            if local_id:
                fetch = min(local_id, config.construction_candidates)
                params = faiss.SearchParametersHNSW(
                    efSearch=max(config.ef_construction, fetch)
                )
                found_distances, found_ids = candidate_index.search(
                    vector, fetch, params=params
                )
                found_candidates = found_ids[0]
                found_candidate_distances = found_distances[0]
                valid = found_candidates >= 0
                all_candidates = found_candidates[valid]
                all_candidate_distances = found_candidate_distances[valid]
                candidates = all_candidates[:config.semantic_degree]
                candidate_distances = all_candidate_distances[:config.semantic_degree]
                for neighbor, distance in zip(candidates, candidate_distances):
                    neighbor_id = int(neighbor)
                    value = float(distance)
                    _add_bounded_edge(
                        semantic_neighbors, semantic_distances,
                        local_id, neighbor_id, value,
                    )
                    _add_bounded_edge(
                        semantic_neighbors, semantic_distances,
                        neighbor_id, local_id, value,
                    )

                temporal_neighbors[local_id, 0] = local_id - 1
                temporal_neighbors[local_id - 1, 1] = local_id

                if range_scales:
                    current_timestamp = int(ordered_timestamps[local_id])
                    for scale_index, scale_hours in enumerate(range_scales):
                        window_ms = scale_hours * 3_600_000
                        window_start = current_timestamp - window_ms
                        first_recent = int(np.searchsorted(
                            ordered_timestamps[:local_id], window_start, side="left"
                        ))
                        recent_population = local_id - first_recent
                        recent_count = min(config.range_candidate_count, recent_population)
                        if recent_count:
                            recent = np.unique(np.linspace(
                                first_recent, local_id - 1, recent_count, dtype=np.int64
                            ))
                        else:
                            recent = np.empty(0, dtype=np.int64)
                        immediate = np.arange(
                            max(first_recent, local_id - 16), local_id, dtype=np.int64
                        )
                        pool = np.unique(np.concatenate((
                            np.asarray(all_candidates, dtype=np.int64), recent, immediate,
                        )))
                        if not len(pool):
                            continue
                        pool_global_ids = global_ids[pool]
                        delta = np.asarray(vectors[pool_global_ids], dtype=np.float32) - vector[0]
                        range_distances = np.einsum("ij,ij->i", delta, delta)
                        gaps = current_timestamp - ordered_timestamps[pool]
                        coeligibility = np.maximum(0.0, 1.0 - gaps / window_ms)
                        valid_range = coeligibility > 0.0
                        pool = pool[valid_range]
                        range_distances = range_distances[valid_range]
                        coeligibility = coeligibility[valid_range]
                        scores = range_distances / (coeligibility + 1e-6)
                        selected_count = min(config.range_degree_per_side, len(pool))
                        if not selected_count:
                            continue
                        selected = np.argpartition(scores, selected_count - 1)[:selected_count]
                        selected = selected[np.argsort(scores[selected], kind="stable")]
                        left_neighbors = range_left_neighbors[:, scale_index, :]
                        right_neighbors = range_right_neighbors[:, scale_index, :]
                        left_scores = range_left_scores[:, scale_index, :]
                        right_scores = range_right_scores[:, scale_index, :]
                        for selected_position in selected:
                            neighbor_id = int(pool[selected_position])
                            score = float(scores[selected_position])
                            _add_bounded_edge(
                                left_neighbors, left_scores,
                                local_id, neighbor_id, score,
                            )
                            _add_bounded_edge(
                                right_neighbors, right_scores,
                                neighbor_id, local_id, score,
                            )

            if config.hierarchy_enabled and levels[local_id] > 0:
                for level in range(1, int(levels[local_id]) + 1):
                    level_index = level - 1
                    base, id_map = upper_indexes[level_index]
                    upper_candidate_ids = np.empty(0, dtype=np.int64)
                    if id_map.ntotal:
                        base.hnsw.efSearch = max(
                            config.ef_construction,
                            config.hierarchy_construction_candidates,
                        )
                        fetch = min(
                            int(id_map.ntotal), config.hierarchy_construction_candidates
                        )
                        upper_distances, upper_ids = id_map.search(vector, fetch)
                        valid_upper = upper_ids[0] >= 0
                        upper_candidate_ids = np.asarray(
                            upper_ids[0][valid_upper], dtype=np.int64
                        )
                        upper_candidate_distances = upper_distances[0][valid_upper]
                        selected_ids = upper_candidate_ids[
                            :config.hierarchy_semantic_degree
                        ]
                        selected_distances = upper_candidate_distances[
                            :config.hierarchy_semantic_degree
                        ]
                        level_neighbors = upper_semantic_neighbors[:, level_index, :]
                        level_distances = upper_semantic_distances[:, level_index, :]
                        for neighbor, distance in zip(selected_ids, selected_distances):
                            neighbor_id = int(neighbor)
                            value = float(distance)
                            _add_bounded_edge(
                                level_neighbors, level_distances,
                                local_id, neighbor_id, value,
                            )
                            _add_bounded_edge(
                                level_neighbors, level_distances,
                                neighbor_id, local_id, value,
                            )

                    if hierarchy_range_scales and level_inserted_ids[level_index]:
                        scale_hours = level_range_scales[level_index]
                        window_ms = scale_hours * 3_600_000
                        current_timestamp = int(ordered_timestamps[local_id])
                        prior_ids = np.asarray(
                            level_inserted_ids[level_index], dtype=np.int64
                        )
                        prior_timestamps = ordered_timestamps[prior_ids]
                        first_recent = int(np.searchsorted(
                            prior_timestamps, current_timestamp - window_ms,
                            side="left",
                        ))
                        recent_population = len(prior_ids) - first_recent
                        recent_count = min(
                            config.hierarchy_range_candidate_count,
                            recent_population,
                        )
                        if recent_count:
                            positions = np.unique(np.linspace(
                                first_recent, len(prior_ids) - 1,
                                recent_count, dtype=np.int64,
                            ))
                            recent = prior_ids[positions]
                        else:
                            recent = np.empty(0, dtype=np.int64)
                        immediate = prior_ids[max(first_recent, len(prior_ids) - 8):]
                        pool = np.unique(np.concatenate((
                            upper_candidate_ids, recent, immediate,
                        )))
                        if len(pool):
                            pool_global_ids = global_ids[pool]
                            difference = np.asarray(
                                vectors[pool_global_ids], dtype=np.float32
                            ) - vector[0]
                            distances = np.einsum(
                                "ij,ij->i", difference, difference
                            )
                            gaps = current_timestamp - ordered_timestamps[pool]
                            coeligibility = np.maximum(
                                0.0, 1.0 - gaps / window_ms
                            )
                            valid_range = coeligibility > 0.0
                            pool = pool[valid_range]
                            distances = distances[valid_range]
                            coeligibility = coeligibility[valid_range]
                            scores = distances / (coeligibility + 1e-6)
                            selected_count = min(
                                config.hierarchy_range_degree_per_side, len(pool)
                            )
                            if selected_count:
                                selected = np.argpartition(
                                    scores, selected_count - 1
                                )[:selected_count]
                                selected = selected[
                                    np.argsort(scores[selected], kind="stable")
                                ]
                                left_neighbors = upper_range_left_neighbors[
                                    :, level_index, :
                                ]
                                right_neighbors = upper_range_right_neighbors[
                                    :, level_index, :
                                ]
                                left_scores = upper_range_left_scores[
                                    :, level_index, :
                                ]
                                right_scores = upper_range_right_scores[
                                    :, level_index, :
                                ]
                                for position in selected:
                                    neighbor_id = int(pool[position])
                                    score = float(scores[position])
                                    _add_bounded_edge(
                                        left_neighbors, left_scores,
                                        local_id, neighbor_id, score,
                                    )
                                    _add_bounded_edge(
                                        right_neighbors, right_scores,
                                        neighbor_id, local_id, score,
                                    )

                    previous = int(last_at_level[level_index])
                    if previous >= 0:
                        upper_temporal_neighbors[local_id, level_index, 0] = previous
                        upper_temporal_neighbors[previous, level_index, 1] = local_id
                    last_at_level[level_index] = local_id
                    level_inserted_ids[level_index].append(local_id)
                    id_map.add_with_ids(
                        vector, np.asarray([local_id], dtype=np.int64)
                    )

            if (local_id + 1) % config.gateway_period == 0:
                row = gateway_count
                gateway_count += 1
                gateway_ids[row] = local_id
                gateway_rows[local_id] = row
                if row:
                    prior_local_ids = gateway_ids[:row].astype(np.int64, copy=False)
                    prior_global_ids = global_ids[prior_local_ids]
                    delta = np.asarray(vectors[prior_global_ids], dtype=np.float32) - vector[0]
                    distances = np.einsum("ij,ij->i", delta, delta)
                    selected_count = min(config.gateway_degree, row)
                    selected = np.argpartition(distances, selected_count - 1)[:selected_count]
                    selected = selected[np.argsort(distances[selected], kind="stable")]
                    for prior_row in selected:
                        prior_row = int(prior_row)
                        distance = float(distances[prior_row])
                        _add_bounded_edge(
                            gateway_neighbors, gateway_distances,
                            row, prior_row, distance,
                        )
                        _add_bounded_edge(
                            gateway_neighbors, gateway_distances,
                            prior_row, row, distance,
                        )

            candidate_index.add(vector)
            completed = local_id + 1
            if progress_every and completed % progress_every == 0:
                elapsed = time.perf_counter() - started
                eta = elapsed * (count - completed) / completed
                print(
                    f"RT-HNSW build: {completed:,}/{count:,} "
                    f"({100 * completed / count:.1f}%), "
                    f"elapsed={elapsed / 60:.1f}m eta={eta / 60:.1f}m",
                    flush=True,
                )

        files = {
            "global_ids": "global_ids.npy",
            "timestamps": "timestamps.npy",
            "semantic_neighbors": "semantic_neighbors.npy",
            "temporal_neighbors": "temporal_neighbors.npy",
            "range_left_neighbors": "range_left_neighbors.npy",
            "range_right_neighbors": "range_right_neighbors.npy",
            "levels": "levels.npy",
            "upper_semantic_neighbors": "upper_semantic_neighbors.npy",
            "upper_temporal_neighbors": "upper_temporal_neighbors.npy",
            "upper_range_left_neighbors": "upper_range_left_neighbors.npy",
            "upper_range_right_neighbors": "upper_range_right_neighbors.npy",
            "gateway_ids": "gateway_ids.npy",
            "gateway_rows": "gateway_rows.npy",
            "gateway_neighbors": "gateway_neighbors.npy",
        }
        arrays = {
            "global_ids": global_ids,
            "timestamps": ordered_timestamps,
            "semantic_neighbors": semantic_neighbors,
            "temporal_neighbors": temporal_neighbors,
            "range_left_neighbors": range_left_neighbors,
            "range_right_neighbors": range_right_neighbors,
            "levels": levels,
            "upper_semantic_neighbors": upper_semantic_neighbors,
            "upper_temporal_neighbors": upper_temporal_neighbors,
            "upper_range_left_neighbors": upper_range_left_neighbors,
            "upper_range_right_neighbors": upper_range_right_neighbors,
            "gateway_ids": gateway_ids[:gateway_count],
            "gateway_rows": gateway_rows,
            "gateway_neighbors": gateway_neighbors[:gateway_count],
        }
        for name, array in arrays.items():
            np.save(temporary / files[name], array, allow_pickle=False)
        manifest = {
            "schema_version": 1,
            "backend": (
                "hierarchical rolling temporal HNSW prototype"
                if config.hierarchy_enabled
                else "flat rolling temporal HNSW prototype"
            ),
            "candidate_generator": "FAISS IndexHNSWFlat (construction only)",
            "metric": "squared L2",
            "vector_count": count,
            "dimensions": int(vectors.shape[1]),
            "dtype": "float32",
            "ordered_arrival": True,
            "timestamp_order_key": "(timestamp_ms, stable source vector_id order)",
            "config": asdict(config),
            "source": source or {},
            "construction_seconds": time.perf_counter() - started,
            "gateway_count": gateway_count,
            "range_scales_hours": list(range_scales),
            "hierarchy_level_counts": {
                str(level): int(np.sum(levels >= level))
                for level in range(1, config.hierarchy_max_level + 1)
            },
            "hierarchy_range_scale_by_level": {
                str(level + 1): int(scale)
                for level, scale in enumerate(level_range_scales)
            },
            "files": files,
        }
        manifest["artifact_size_bytes"] = sum(
            (temporary / filename).stat().st_size for filename in files.values()
        )
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


class RollingTemporalHNSW:
    def __init__(self, directory: Path, vectors: np.ndarray):
        self.directory = Path(directory)
        self.manifest = json.loads(
            (self.directory / "manifest.json").read_text(encoding="utf-8")
        )
        self.config = RollingBuildConfig(**self.manifest["config"])
        files = self.manifest["files"]
        self.global_ids = np.load(self.directory / files["global_ids"], mmap_mode="r")
        self.timestamps = np.load(self.directory / files["timestamps"], mmap_mode="r")
        self.semantic_neighbors = np.load(
            self.directory / files["semantic_neighbors"], mmap_mode="r"
        )
        self.temporal_neighbors = np.load(
            self.directory / files["temporal_neighbors"], mmap_mode="r"
        )
        if "range_left_neighbors" in files:
            self.range_left_neighbors = np.load(
                self.directory / files["range_left_neighbors"], mmap_mode="r"
            )
            self.range_right_neighbors = np.load(
                self.directory / files["range_right_neighbors"], mmap_mode="r"
            )
        else:
            self.range_left_neighbors = np.empty((len(self.global_ids), 0), dtype=np.int32)
            self.range_right_neighbors = np.empty((len(self.global_ids), 0), dtype=np.int32)
        if "levels" in files:
            self.levels = np.load(self.directory / files["levels"], mmap_mode="r")
            self.upper_semantic_neighbors = np.load(
                self.directory / files["upper_semantic_neighbors"], mmap_mode="r"
            )
            self.upper_temporal_neighbors = np.load(
                self.directory / files["upper_temporal_neighbors"], mmap_mode="r"
            )
            if "upper_range_left_neighbors" in files:
                self.upper_range_left_neighbors = np.load(
                    self.directory / files["upper_range_left_neighbors"],
                    mmap_mode="r",
                )
                self.upper_range_right_neighbors = np.load(
                    self.directory / files["upper_range_right_neighbors"],
                    mmap_mode="r",
                )
            else:
                self.upper_range_left_neighbors = np.empty(
                    (len(self.global_ids), 0, 0), dtype=np.int32
                )
                self.upper_range_right_neighbors = np.empty(
                    (len(self.global_ids), 0, 0), dtype=np.int32
                )
        else:
            self.levels = np.zeros(len(self.global_ids), dtype=np.int8)
            self.upper_semantic_neighbors = np.empty(
                (len(self.global_ids), 0, 0), dtype=np.int32
            )
            self.upper_temporal_neighbors = np.empty(
                (len(self.global_ids), 0, 2), dtype=np.int32
            )
            self.upper_range_left_neighbors = np.empty(
                (len(self.global_ids), 0, 0), dtype=np.int32
            )
            self.upper_range_right_neighbors = np.empty(
                (len(self.global_ids), 0, 0), dtype=np.int32
            )
        self.level_ids = {
            level: np.flatnonzero(self.levels >= level).astype(np.int32)
            for level in range(1, int(np.max(self.levels, initial=0)) + 1)
        }
        self.gateway_ids = np.load(self.directory / files["gateway_ids"], mmap_mode="r")
        self.gateway_rows = np.load(self.directory / files["gateway_rows"], mmap_mode="r")
        self.gateway_neighbors = np.load(
            self.directory / files["gateway_neighbors"], mmap_mode="r"
        )
        self.gateway_timestamps = np.asarray(self.timestamps[self.gateway_ids], dtype=np.int64)
        self.vectors = vectors
        if len(self.global_ids) != self.manifest["vector_count"]:
            raise ValueError("rolling-HNSW artifacts do not match manifest")
        if vectors.ndim != 2 or vectors.dtype != np.float32:
            raise ValueError("vectors must be a float32 matrix")
        if len(self.global_ids) and int(np.max(self.global_ids)) >= len(vectors):
            raise ValueError("rolling-HNSW global IDs exceed vector matrix")
        if np.any(np.diff(self.timestamps) < 0):
            raise ValueError("rolling-HNSW timestamps are not ordered")

    def _components(
        self, intervals: Iterable[Iterable[int]]
    ) -> list[tuple[int, int, int, int]]:
        components = []
        for start, end in normalize_intervals(intervals):
            left = int(np.searchsorted(self.timestamps, start, side="left"))
            right = int(np.searchsorted(self.timestamps, end, side="left"))
            if left < right:
                components.append((start, end, left, right))
        return components

    def _gateway_seed_pairs(
        self,
        query: np.ndarray,
        start: int,
        end: int,
        config: RollingSearchConfig,
    ) -> tuple[list[tuple[float, int]], int]:
        left = int(np.searchsorted(self.gateway_timestamps, start, side="left"))
        right = int(np.searchsorted(self.gateway_timestamps, end, side="left"))
        if left >= right:
            return [], 0
        available = right - left
        sample_count = min(config.gateway_candidates, available)
        positions = np.linspace(left, right - 1, sample_count, dtype=np.int64)
        positions = np.unique(positions)
        local_ids = np.asarray(self.gateway_ids[positions], dtype=np.int64)
        database_ids = self.global_ids[local_ids]
        delta = np.asarray(self.vectors[database_ids], dtype=np.float32) - query
        distances = np.einsum("ij,ij->i", delta, delta)
        seed_count = min(config.gateway_seeds, len(local_ids))
        selected = np.argpartition(distances, seed_count - 1)[:seed_count]
        selected = selected[np.argsort(distances[selected], kind="stable")]
        return [
            (float(distances[position]), int(local_ids[position]))
            for position in selected
        ], len(local_ids)

    def _hierarchical_seed_pairs(
        self,
        query: np.ndarray,
        local_left: int,
        local_right: int,
        config: RollingSearchConfig,
    ) -> tuple[list[tuple[float, int]], dict[str, int]]:
        diagnostics = {
            "upper_entry_level": 0,
            "upper_nodes_visited": 0,
            "upper_semantic_edges_examined": 0,
            "upper_range_edges_examined": 0,
            "upper_temporal_edges_examined": 0,
            "upper_out_of_range_edges_rejected": 0,
            "upper_distance_computations": 0,
            "upper_entry_candidates": 0,
        }
        if not config.use_hierarchy or not self.level_ids:
            return [], diagnostics

        highest_level = 0
        eligible_level_ids = np.empty(0, dtype=np.int32)
        for level in sorted(self.level_ids, reverse=True):
            ids = self.level_ids[level]
            left = int(np.searchsorted(ids, local_left, side="left"))
            right = int(np.searchsorted(ids, local_right, side="left"))
            if left < right:
                highest_level = level
                eligible_level_ids = ids[left:right]
                break
        if not highest_level:
            return [], diagnostics

        sample_count = min(config.gateway_candidates, len(eligible_level_ids))
        positions = np.unique(np.linspace(
            0, len(eligible_level_ids) - 1, sample_count, dtype=np.int64
        ))
        sampled = np.asarray(eligible_level_ids[positions], dtype=np.int64)
        database_ids = self.global_ids[sampled]
        delta = np.asarray(self.vectors[database_ids], dtype=np.float32) - query
        distances = np.einsum("ij,ij->i", delta, delta)
        cache = {
            int(local_id): float(distance)
            for local_id, distance in zip(sampled, distances)
        }
        seed_count = min(config.gateway_seeds, len(sampled))
        selected = np.argpartition(distances, seed_count - 1)[:seed_count]
        pairs = sorted(
            (float(distances[position]), int(sampled[position]))
            for position in selected
        )
        diagnostics["upper_entry_level"] = highest_level
        diagnostics["upper_entry_candidates"] = len(sampled)
        diagnostics["upper_distance_computations"] = len(sampled)

        for level in range(highest_level, 0, -1):
            candidates = list(pairs)
            heapq.heapify(candidates)
            best = [(-distance, local_id) for distance, local_id in pairs]
            heapq.heapify(best)
            visited = {local_id for _, local_id in pairs}
            while candidates:
                current_distance, current = heapq.heappop(candidates)
                if (len(best) >= config.upper_ef_search and
                        current_distance > -best[0][0]):
                    break
                edge_groups: list[tuple[str, np.ndarray]] = []
                if (config.use_range_edges and
                        self.upper_range_left_neighbors.shape[1] >= level):
                    edge_groups.extend((
                        ("range", self.upper_range_left_neighbors[current, level - 1]),
                        ("range", self.upper_range_right_neighbors[current, level - 1]),
                    ))
                edge_groups.extend((
                    ("semantic", self.upper_semantic_neighbors[current, level - 1]),
                    ("temporal", self.upper_temporal_neighbors[current, level - 1]),
                ))
                for edge_type, neighbors in edge_groups:
                    for value in neighbors:
                        neighbor = int(value)
                        if neighbor < 0:
                            continue
                        if edge_type == "semantic":
                            diagnostics["upper_semantic_edges_examined"] += 1
                        elif edge_type == "range":
                            diagnostics["upper_range_edges_examined"] += 1
                        else:
                            diagnostics["upper_temporal_edges_examined"] += 1
                        if neighbor < local_left or neighbor >= local_right:
                            diagnostics["upper_out_of_range_edges_rejected"] += 1
                            continue
                        if neighbor in visited:
                            continue
                        visited.add(neighbor)
                        if neighbor in cache:
                            distance = cache[neighbor]
                        else:
                            global_id = int(self.global_ids[neighbor])
                            difference = np.asarray(
                                self.vectors[global_id], dtype=np.float32
                            ) - query
                            distance = float(np.dot(difference, difference))
                            cache[neighbor] = distance
                            diagnostics["upper_distance_computations"] += 1
                        if (len(best) < config.upper_ef_search or
                                distance < -best[0][0]):
                            heapq.heappush(candidates, (distance, neighbor))
                            heapq.heappush(best, (-distance, neighbor))
                            if len(best) > config.upper_ef_search:
                                heapq.heappop(best)
            diagnostics["upper_nodes_visited"] += len(visited)
            ordered = sorted((-distance, local_id) for distance, local_id in best)
            pairs = ordered[:min(config.gateway_seeds, len(ordered))]
        return pairs, diagnostics

    def _strict_component_search(
        self,
        query: np.ndarray,
        local_left: int,
        local_right: int,
        start: int,
        end: int,
        config: RollingSearchConfig,
    ) -> dict[str, Any]:
        upper_seeds, upper_diagnostics = self._hierarchical_seed_pairs(
            query, local_left, local_right, config
        )
        if upper_seeds:
            seeds = upper_seeds
            gateway_candidates = 0
            component_method = "strict_hierarchy"
        else:
            seeds, gateway_candidates = self._gateway_seed_pairs(
                query, start, end, config
            )
            component_method = "strict_graph"
        if not seeds:
            # This occurs only for a component smaller than the periodic spacing
            # in the intended configuration. Exact search preserves correctness.
            local_ids = np.arange(local_left, local_right, dtype=np.int64)
            ids, distances, computations = _exact_top_k(
                query, self.vectors, self.global_ids, local_ids, config.k
            )
            return {
                "local_ids": ids, "distances": distances,
                "distance_computations": computations,
                "nodes_visited": computations, "gateway_candidates": 0,
                "gateway_seeds": 0, "semantic_edges_examined": 0,
                "range_edges_examined": 0, "gateway_edges_examined": 0,
                "temporal_edges_examined": 0,
                "out_of_range_edges_rejected": 0,
                **upper_diagnostics,
                "component_method": "exact_no_gateway",
            }

        candidates: list[tuple[float, int]] = []
        best: list[tuple[float, int]] = []
        visited: set[int] = set()
        for distance, local_id in seeds:
            if local_id in visited:
                continue
            visited.add(local_id)
            heapq.heappush(candidates, (distance, local_id))
            heapq.heappush(best, (-distance, local_id))

        semantic_examined = range_examined = gateway_examined = temporal_examined = rejected = 0
        distance_computations = (
            gateway_candidates + upper_diagnostics["upper_distance_computations"]
        )
        while candidates:
            current_distance, current = heapq.heappop(candidates)
            if len(best) >= config.ef_search and current_distance > -best[0][0]:
                break

            edge_groups: list[tuple[str, np.ndarray]] = []
            if config.use_range_edges:
                edge_groups.extend((
                    ("range", self.range_left_neighbors[current].reshape(-1)),
                    ("range", self.range_right_neighbors[current].reshape(-1)),
                ))
            edge_groups.append(("semantic", self.semantic_neighbors[current]))
            gateway_row = int(self.gateway_rows[current])
            if config.use_gateway_edges and gateway_row >= 0:
                neighbor_rows = self.gateway_neighbors[gateway_row]
                valid_rows = neighbor_rows[neighbor_rows >= 0]
                edge_groups.append(("gateway", self.gateway_ids[valid_rows]))
            if config.use_temporal_edges:
                edge_groups.append(("temporal", self.temporal_neighbors[current]))

            for edge_type, neighbors in edge_groups:
                for value in neighbors:
                    neighbor = int(value)
                    if neighbor < 0:
                        continue
                    if edge_type == "semantic":
                        semantic_examined += 1
                    elif edge_type == "range":
                        range_examined += 1
                    elif edge_type == "gateway":
                        gateway_examined += 1
                    else:
                        temporal_examined += 1
                    if neighbor < local_left or neighbor >= local_right:
                        rejected += 1
                        continue
                    if neighbor in visited:
                        continue
                    visited.add(neighbor)
                    global_id = int(self.global_ids[neighbor])
                    delta = np.asarray(self.vectors[global_id], dtype=np.float32) - query
                    distance = float(np.dot(delta, delta))
                    distance_computations += 1
                    if len(best) < config.ef_search or distance < -best[0][0]:
                        heapq.heappush(candidates, (distance, neighbor))
                        heapq.heappush(best, (-distance, neighbor))
                        if len(best) > config.ef_search:
                            heapq.heappop(best)

        ordered = sorted([(-distance, local_id) for distance, local_id in best])
        return {
            "local_ids": np.asarray([item[1] for item in ordered], dtype=np.int64),
            "distances": np.asarray([item[0] for item in ordered], dtype=np.float32),
            "distance_computations": distance_computations,
            "nodes_visited": len(visited) + upper_diagnostics["upper_nodes_visited"],
            "gateway_candidates": gateway_candidates,
            "gateway_seeds": len(seeds),
            "semantic_edges_examined": semantic_examined,
            "range_edges_examined": range_examined,
            "gateway_edges_examined": gateway_examined,
            "temporal_edges_examined": temporal_examined,
            "out_of_range_edges_rejected": rejected,
            **upper_diagnostics,
            "component_method": component_method,
        }

    def search(
        self,
        query: np.ndarray,
        intervals: Iterable[Iterable[int]],
        search_config: RollingSearchConfig = RollingSearchConfig(),
    ) -> dict[str, Any]:
        search_config.validate()
        query = np.asarray(query, dtype=np.float32)
        if query.shape != (self.manifest["dimensions"],):
            raise ValueError(f"query must have shape ({self.manifest['dimensions']},)")
        normalized = normalize_intervals(intervals)
        route_started = time.perf_counter_ns()
        components = self._components(normalized)
        eligible_count = sum(right - left for _, _, left, right in components)
        routing_ms = (time.perf_counter_ns() - route_started) / 1e6
        search_started = time.perf_counter_ns()

        diagnostics = {
            "distance_computations": 0, "nodes_visited": 0,
            "gateway_candidates": 0, "gateway_seeds": 0,
            "semantic_edges_examined": 0, "range_edges_examined": 0,
            "gateway_edges_examined": 0, "temporal_edges_examined": 0,
            "out_of_range_edges_rejected": 0,
            "upper_entry_level": 0, "upper_nodes_visited": 0,
            "upper_semantic_edges_examined": 0,
            "upper_range_edges_examined": 0,
            "upper_temporal_edges_examined": 0,
            "upper_out_of_range_edges_rejected": 0,
            "upper_distance_computations": 0,
            "upper_entry_candidates": 0,
        }
        candidate_parts: list[np.ndarray] = []
        methods: list[str] = []
        if eligible_count <= search_config.exact_threshold:
            local_ids = np.concatenate([
                np.arange(left, right, dtype=np.int64)
                for _, _, left, right in components
            ]) if components else np.empty(0, dtype=np.int64)
            ids, _, computations = _exact_top_k(
                query, self.vectors, self.global_ids, local_ids, search_config.k
            )
            candidate_parts.append(ids)
            diagnostics["distance_computations"] = computations
            diagnostics["nodes_visited"] = computations
            methods.append("exact")
        else:
            for start, end, left, right in components:
                component_count = right - left
                if component_count <= search_config.exact_threshold:
                    local_ids = np.arange(left, right, dtype=np.int64)
                    ids, _, computations = _exact_top_k(
                        query, self.vectors, self.global_ids, local_ids, search_config.k
                    )
                    candidate_parts.append(ids)
                    diagnostics["distance_computations"] += computations
                    diagnostics["nodes_visited"] += computations
                    methods.append("exact_component")
                    continue
                result = self._strict_component_search(
                    query, left, right, start, end, search_config
                )
                candidate_parts.append(result["local_ids"])
                methods.append(result["component_method"])
                for field in diagnostics:
                    if field == "upper_entry_level":
                        diagnostics[field] = max(diagnostics[field], result[field])
                    else:
                        diagnostics[field] += result[field]
        search_ms = (time.perf_counter_ns() - search_started) / 1e6

        rerank_started = time.perf_counter_ns()
        candidates = np.unique(np.concatenate(candidate_parts)) if candidate_parts else np.empty(0, dtype=np.int64)
        result_local_ids, result_distances, rerank_computations = _exact_top_k(
            query, self.vectors, self.global_ids, candidates, search_config.k
        )
        rerank_ms = (time.perf_counter_ns() - rerank_started) / 1e6
        result_ids = np.asarray(self.global_ids[result_local_ids], dtype=np.int64)
        # The rerank repeats candidate distances by design and is reported separately.
        return {
            "ids": result_ids,
            "distances": result_distances,
            "method": "+".join(sorted(set(methods))) if methods else "empty",
            "eligible_count": eligible_count,
            "candidate_count": len(candidates),
            **diagnostics,
            "rerank_distance_computations": rerank_computations,
            "routing_ms": routing_ms,
            "search_ms": search_ms,
            "rerank_ms": rerank_ms,
            "total_ms": routing_ms + search_ms + rerank_ms,
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the flat 100k Twitter RT-HNSW prototype")
    parser.add_argument("--embeddings", type=Path, default=Path("Twitter_data/processed/database_vectors.npy"))
    parser.add_argument("--timestamps", type=Path, default=Path("Twitter_data/ann/timestamps_ms.npy"))
    parser.add_argument("--output", type=Path, default=Path("Twitter_data/ann/rolling_hnsw/flat_100k"))
    parser.add_argument("--vector-count", type=int, default=100_000)
    parser.add_argument("--semantic-degree", type=int, default=32)
    parser.add_argument("--gateway-degree", type=int, default=8)
    parser.add_argument("--gateway-period", type=int, default=512)
    parser.add_argument("--construction-candidates", type=int, default=64)
    parser.add_argument("--ef-construction", type=int, default=200)
    parser.add_argument("--candidate-index-m", type=int, default=32)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--range-degree-per-side", type=int, default=0)
    parser.add_argument("--range-candidate-count", type=int, default=64)
    parser.add_argument("--range-window-hours", type=int, default=720)
    parser.add_argument("--range-scales-hours", default="")
    parser.add_argument("--hierarchy", action="store_true")
    parser.add_argument("--hierarchy-semantic-degree", type=int, default=8)
    parser.add_argument("--hierarchy-level-probability", type=float, default=0.05)
    parser.add_argument("--hierarchy-max-level", type=int, default=6)
    parser.add_argument("--hierarchy-seed", type=int, default=23)
    parser.add_argument("--hierarchy-construction-candidates", type=int, default=32)
    parser.add_argument("--hierarchy-range-degree-per-side", type=int, default=0)
    parser.add_argument("--hierarchy-range-candidate-count", type=int, default=32)
    parser.add_argument("--hierarchy-range-scales-hours", default="")
    parser.add_argument("--progress-every", type=int, default=5_000)
    parser.add_argument("--rebuild", action="store_true")
    return parser


def build_main() -> None:
    args = build_parser().parse_args()
    vectors = np.load(args.embeddings, mmap_mode="r")
    timestamps = np.load(args.timestamps, mmap_mode="r")
    config = RollingBuildConfig(
        vector_count=args.vector_count,
        semantic_degree=args.semantic_degree,
        gateway_degree=args.gateway_degree,
        gateway_period=args.gateway_period,
        construction_candidates=args.construction_candidates,
        ef_construction=args.ef_construction,
        candidate_index_m=args.candidate_index_m,
        threads=args.threads,
        range_degree_per_side=args.range_degree_per_side,
        range_candidate_count=args.range_candidate_count,
        range_window_hours=args.range_window_hours,
        range_scales_hours=args.range_scales_hours,
        hierarchy_enabled=args.hierarchy,
        hierarchy_semantic_degree=args.hierarchy_semantic_degree,
        hierarchy_level_probability=args.hierarchy_level_probability,
        hierarchy_max_level=args.hierarchy_max_level,
        hierarchy_seed=args.hierarchy_seed,
        hierarchy_construction_candidates=args.hierarchy_construction_candidates,
        hierarchy_range_degree_per_side=args.hierarchy_range_degree_per_side,
        hierarchy_range_candidate_count=args.hierarchy_range_candidate_count,
        hierarchy_range_scales_hours=args.hierarchy_range_scales_hours,
    )
    manifest = build_rolling_index(
        vectors, timestamps, args.output, config,
        source={"embeddings": str(args.embeddings), "timestamps": str(args.timestamps)},
        rebuild=args.rebuild, progress_every=args.progress_every,
    )
    print(json.dumps(manifest, indent=2))
