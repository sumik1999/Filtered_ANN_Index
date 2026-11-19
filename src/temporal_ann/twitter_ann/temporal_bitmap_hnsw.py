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
from .rolling_hnsw import _add_bounded_edge, _exact_top_k

HOUR_MS = 3_600_000


@dataclass(frozen=True)
class TemporalBitmapBuildConfig:
    vector_count: int = 100_000
    local_semantic_degree: int = 6
    historical_semantic_degree: int = 6
    future_reciprocal_degree: int = 6
    local_window_hours: int = 24
    range_degree_per_side: int = 1
    range_scales_hours: str = "24,168,720"
    range_candidate_count: int = 64
    bridge_degree_per_side: int = 1
    bridge_scales_epochs: str = "1,4,16,64"
    epoch_hours: int = 24
    gateway_period: int = 512
    construction_candidates: int = 64
    ef_construction: int = 200
    candidate_index_m: int = 32
    threads: int = 4

    def range_scales(self) -> tuple[int, ...]:
        return tuple(sorted({int(value) for value in self.range_scales_hours.split(",") if value}))

    def bridge_scales(self) -> tuple[int, ...]:
        return tuple(sorted({int(value) for value in self.bridge_scales_epochs.split(",") if value}))

    def validate(self) -> None:
        positive = (
            self.vector_count, self.local_semantic_degree,
            self.historical_semantic_degree, self.future_reciprocal_degree,
            self.local_window_hours, self.range_degree_per_side,
            self.range_candidate_count, self.bridge_degree_per_side,
            self.epoch_hours, self.gateway_period, self.construction_candidates,
            self.ef_construction, self.candidate_index_m, self.threads,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("temporal bitmap HNSW parameters must be positive")
        if self.construction_candidates < max(
            self.local_semantic_degree, self.historical_semantic_degree
        ):
            raise ValueError("construction candidates are below semantic degree")
        if not self.range_scales() or not self.bridge_scales():
            raise ValueError("range and bridge scales must be nonempty")


@dataclass(frozen=True)
class TemporalBitmapSearchConfig:
    k: int = 100
    ef_search: int = 256
    exact_threshold: int = 5_000
    gateway_candidates: int = 16
    gateway_seeds: int = 4
    use_bitmap_pruning: bool = True
    use_local_edges: bool = True
    use_historical_edges: bool = True
    use_future_edges: bool = True
    use_range_edges: bool = True
    use_bridge_edges: bool = True
    use_temporal_edges: bool = True

    def validate(self) -> None:
        if min(self.k, self.ef_search, self.gateway_candidates, self.gateway_seeds) <= 0:
            raise ValueError("search parameters must be positive")
        if self.ef_search < self.k or self.gateway_seeds > self.gateway_candidates:
            raise ValueError("invalid search beam or gateway count")
        if self.exact_threshold < 0:
            raise ValueError("exact threshold must be nonnegative")


def _grow_edge(
    neighbors: np.ndarray,
    distances: np.ndarray,
    row: int,
    neighbor: int,
    distance: float,
) -> None:
    _add_bounded_edge(neighbors, distances, row, neighbor, distance)


def _set_bitmap(bits: np.ndarray, row: int, group: int, epoch: int) -> None:
    word, offset = divmod(epoch, 64)
    bits[row, group, word] |= np.uint64(1) << np.uint64(offset)


def _bitmap_intersects(bits: np.ndarray, first_epoch: int, last_epoch: int) -> bool:
    if first_epoch > last_epoch or not bits.size:
        return False
    first_epoch = max(0, first_epoch)
    last_epoch = min(bits.size * 64 - 1, last_epoch)
    if first_epoch > last_epoch:
        return False
    first_word, first_offset = divmod(first_epoch, 64)
    last_word, last_offset = divmod(last_epoch, 64)
    if first_word == last_word:
        width = last_offset - first_offset + 1
        mask = np.uint64(((1 << width) - 1) << first_offset)
        return bool(bits[first_word] & mask)
    first_mask = np.uint64(((1 << (64 - first_offset)) - 1) << first_offset)
    if bits[first_word] & first_mask:
        return True
    if last_word > first_word + 1 and np.any(bits[first_word + 1:last_word]):
        return True
    last_mask = np.uint64((1 << (last_offset + 1)) - 1) if last_offset < 63 else np.uint64(2**64 - 1)
    return bool(bits[last_word] & last_mask)


def build_temporal_bitmap_index(
    vectors: np.ndarray,
    timestamps: np.ndarray,
    output: Path,
    config: TemporalBitmapBuildConfig,
    *,
    rebuild: bool = False,
    progress_every: int = 5_000,
) -> dict[str, Any]:
    config.validate()
    vectors = np.asarray(vectors)
    timestamps = np.asarray(timestamps, dtype=np.int64)
    if vectors.ndim != 2 or vectors.dtype != np.float32 or not len(vectors):
        raise ValueError("vectors must be a nonempty float32 matrix")
    if timestamps.shape != (len(vectors),):
        raise ValueError("timestamps must align with vectors")
    count = min(config.vector_count, len(vectors))
    output = Path(output)
    if output.exists() and not rebuild:
        return json.loads((output / "manifest.json").read_text(encoding="utf-8"))

    global_ids = np.argsort(timestamps, kind="stable")[:count].astype(np.int64)
    ordered_timestamps = np.asarray(timestamps[global_ids], dtype=np.int64)
    epoch_ms = config.epoch_hours * HOUR_MS
    epoch_origin = int(ordered_timestamps[0] // epoch_ms * epoch_ms)
    epoch_ids = ((ordered_timestamps - epoch_origin) // epoch_ms).astype(np.int64)
    epoch_count = int(epoch_ids[-1]) + 1
    bitmap_words = (epoch_count + 63) // 64
    range_scales = config.range_scales()
    bridge_scales = config.bridge_scales()
    group_names = ["local", "historical", "future"]
    group_names.extend(f"range_{scale}h" for scale in range_scales)
    group_names.extend(("bridge", "temporal"))
    group_index = {name: index for index, name in enumerate(group_names)}

    local = np.full((count, config.local_semantic_degree), -1, dtype=np.int32)
    local_dist = np.full_like(local, np.inf, dtype=np.float32)
    historical = np.full((count, config.historical_semantic_degree), -1, dtype=np.int32)
    historical_dist = np.full_like(historical, np.inf, dtype=np.float32)
    future = np.full((count, config.future_reciprocal_degree), -1, dtype=np.int32)
    future_dist = np.full_like(future, np.inf, dtype=np.float32)
    temporal = np.full((count, 2), -1, dtype=np.int32)
    range_shape = (count, len(range_scales), config.range_degree_per_side)
    range_left = np.full(range_shape, -1, dtype=np.int32)
    range_right = np.full(range_shape, -1, dtype=np.int32)
    range_left_scores = np.full(range_shape, np.inf, dtype=np.float32)
    range_right_scores = np.full(range_shape, np.inf, dtype=np.float32)
    bridge_shape = (count, len(bridge_scales), config.bridge_degree_per_side)
    bridge_left = np.full(bridge_shape, -1, dtype=np.int32)
    bridge_right = np.full(bridge_shape, -1, dtype=np.int32)
    bridge_left_scores = np.full(bridge_shape, np.inf, dtype=np.float32)
    bridge_right_scores = np.full(bridge_shape, np.inf, dtype=np.float32)

    gateway_ids = np.arange(
        config.gateway_period - 1, count, config.gateway_period, dtype=np.int32
    )
    faiss.omp_set_num_threads(config.threads)
    candidate_index = faiss.IndexHNSWFlat(
        vectors.shape[1], config.candidate_index_m, faiss.METRIC_L2
    )
    candidate_index.hnsw.efConstruction = config.ef_construction
    started = time.perf_counter()

    for local_id, global_id in enumerate(global_ids):
        vector = np.ascontiguousarray(vectors[global_id]).reshape(1, -1)
        all_candidates = np.empty(0, dtype=np.int64)
        all_distances = np.empty(0, dtype=np.float32)
        if local_id:
            fetch = min(local_id, config.construction_candidates)
            params = faiss.SearchParametersHNSW(
                efSearch=max(config.ef_construction, fetch)
            )
            distances, found = candidate_index.search(vector, fetch, params=params)
            valid = found[0] >= 0
            all_candidates = np.asarray(found[0][valid], dtype=np.int64)
            all_distances = np.asarray(distances[0][valid], dtype=np.float32)

            for neighbor, distance in zip(
                all_candidates[:config.historical_semantic_degree],
                all_distances[:config.historical_semantic_degree],
            ):
                neighbor, distance = int(neighbor), float(distance)
                _grow_edge(historical, historical_dist, local_id, neighbor, distance)
                # Future reciprocal capacity is independent and cannot evict the
                # old node's historical or local links.
                _grow_edge(future, future_dist, neighbor, local_id, distance)

            timestamp = int(ordered_timestamps[local_id])
            local_start = int(np.searchsorted(
                ordered_timestamps[:local_id],
                timestamp - config.local_window_hours * HOUR_MS,
                side="left",
            ))
            recent_count = min(config.range_candidate_count, local_id - local_start)
            recent = (
                np.unique(np.linspace(local_start, local_id - 1, recent_count, dtype=np.int64))
                if recent_count else np.empty(0, dtype=np.int64)
            )
            local_pool = np.unique(np.concatenate((
                all_candidates[
                    (all_candidates >= local_start)
                    & (epoch_ids[all_candidates] == epoch_ids[local_id])
                ],
                recent[epoch_ids[recent] == epoch_ids[local_id]],
            )))
            if len(local_pool):
                delta = np.asarray(vectors[global_ids[local_pool]], dtype=np.float32) - vector[0]
                local_distances = np.einsum("ij,ij->i", delta, delta)
                selected_count = min(config.local_semantic_degree, len(local_pool))
                selected = np.argpartition(local_distances, selected_count - 1)[:selected_count]
                selected = selected[np.argsort(local_distances[selected], kind="stable")]
                for position in selected:
                    neighbor, distance = int(local_pool[position]), float(local_distances[position])
                    _grow_edge(local, local_dist, local_id, neighbor, distance)
                    _grow_edge(local, local_dist, neighbor, local_id, distance)

            temporal[local_id, 0] = local_id - 1
            temporal[local_id - 1, 1] = local_id

            for scale_index, scale_hours in enumerate(range_scales):
                window_ms = scale_hours * HOUR_MS
                first = int(np.searchsorted(
                    ordered_timestamps[:local_id], timestamp - window_ms, side="left"
                ))
                population = local_id - first
                sample_count = min(config.range_candidate_count, population)
                sampled = (
                    np.unique(np.linspace(first, local_id - 1, sample_count, dtype=np.int64))
                    if sample_count else np.empty(0, dtype=np.int64)
                )
                immediate = np.arange(max(first, local_id - 16), local_id, dtype=np.int64)
                pool = np.unique(np.concatenate((all_candidates, sampled, immediate)))
                if not len(pool):
                    continue
                delta = np.asarray(vectors[global_ids[pool]], dtype=np.float32) - vector[0]
                vector_distances = np.einsum("ij,ij->i", delta, delta)
                gaps = timestamp - ordered_timestamps[pool]
                coeligibility = np.maximum(0.0, 1.0 - gaps / window_ms)
                valid_range = coeligibility > 0
                pool = pool[valid_range]
                scores = vector_distances[valid_range] / (coeligibility[valid_range] + 1e-6)
                selected_count = min(config.range_degree_per_side, len(pool))
                if selected_count:
                    selected = np.argpartition(scores, selected_count - 1)[:selected_count]
                    selected = selected[np.argsort(scores[selected], kind="stable")]
                    for position in selected:
                        neighbor, score = int(pool[position]), float(scores[position])
                        _grow_edge(
                            range_left[:, scale_index, :], range_left_scores[:, scale_index, :],
                            local_id, neighbor, score,
                        )
                        _grow_edge(
                            range_right[:, scale_index, :], range_right_scores[:, scale_index, :],
                            neighbor, local_id, score,
                        )

            for scale_index, scale_epochs in enumerate(bridge_scales):
                target_time = timestamp - scale_epochs * epoch_ms
                target = int(np.searchsorted(
                    ordered_timestamps[:local_id], target_time, side="right"
                )) - 1
                if target < 0:
                    continue
                candidates = np.arange(
                    max(0, target - config.bridge_degree_per_side + 1), target + 1,
                    dtype=np.int64,
                )
                for neighbor in candidates:
                    gap = float(timestamp - int(ordered_timestamps[neighbor]))
                    _grow_edge(
                        bridge_left[:, scale_index, :], bridge_left_scores[:, scale_index, :],
                        local_id, int(neighbor), gap,
                    )
                    _grow_edge(
                        bridge_right[:, scale_index, :], bridge_right_scores[:, scale_index, :],
                        int(neighbor), local_id, gap,
                    )

        candidate_index.add(vector)
        completed = local_id + 1
        if progress_every and completed % progress_every == 0:
            print(
                f"bitmap RT-HNSW build: {completed:,}/{count:,} "
                f"elapsed={(time.perf_counter() - started):.1f}s",
                flush=True,
            )

    edge_epoch_bitmaps = np.zeros(
        (count, len(group_names), bitmap_words), dtype=np.uint64
    )
    for row in range(count):
        groups: list[tuple[int, np.ndarray]] = [
            (group_index["local"], local[row]),
            (group_index["historical"], historical[row]),
            (group_index["future"], future[row]),
        ]
        groups.extend(
            (group_index[f"range_{scale}h"], np.concatenate((
                range_left[row, index], range_right[row, index]
            )))
            for index, scale in enumerate(range_scales)
        )
        groups.extend((
            (group_index["bridge"], np.concatenate((
                bridge_left[row].reshape(-1), bridge_right[row].reshape(-1)
            ))),
            (group_index["temporal"], temporal[row]),
        ))
        for group, neighbors in groups:
            for neighbor in neighbors:
                if neighbor >= 0:
                    _set_bitmap(
                        edge_epoch_bitmaps, row, group, int(epoch_ids[int(neighbor)])
                    )

    portal_coverage_bitmaps = np.zeros((len(gateway_ids), bitmap_words), dtype=np.uint64)
    for portal_row, local_id in enumerate(gateway_ids):
        _set_bitmap(portal_coverage_bitmaps[:, None, :], portal_row, 0, int(epoch_ids[local_id]))
        portal_coverage_bitmaps[portal_row] |= np.bitwise_or.reduce(
            edge_epoch_bitmaps[local_id], axis=0
        )

    arrays = {
        "global_ids": global_ids,
        "timestamps": ordered_timestamps,
        "epoch_ids": epoch_ids,
        "local_neighbors": local,
        "historical_neighbors": historical,
        "future_neighbors": future,
        "temporal_neighbors": temporal,
        "range_left_neighbors": range_left,
        "range_right_neighbors": range_right,
        "bridge_left_neighbors": bridge_left,
        "bridge_right_neighbors": bridge_right,
        "gateway_ids": gateway_ids,
        "edge_epoch_bitmaps": edge_epoch_bitmaps,
        "portal_coverage_bitmaps": portal_coverage_bitmaps,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        files = {name: f"{name}.npy" for name in arrays}
        for name, array in arrays.items():
            np.save(temporary / files[name], array, allow_pickle=False)
        manifest = {
            "schema_version": 1,
            "backend": "time-stretched temporal bitmap HNSW prototype",
            "metric": "squared L2",
            "vector_count": count,
            "dimensions": int(vectors.shape[1]),
            "config": asdict(config),
            "epoch_origin_ms": epoch_origin,
            "epoch_count": epoch_count,
            "bitmap_words": bitmap_words,
            "edge_group_names": group_names,
            "range_scales_hours": list(range_scales),
            "bridge_scales_epochs": list(bridge_scales),
            "gateway_count": len(gateway_ids),
            "construction_seconds": time.perf_counter() - started,
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


class TemporalBitmapHNSW:
    def __init__(self, directory: Path, vectors: np.ndarray):
        self.directory = Path(directory)
        self.manifest = json.loads((self.directory / "manifest.json").read_text())
        self.config = TemporalBitmapBuildConfig(**self.manifest["config"])
        self.vectors = vectors
        for name, filename in self.manifest["files"].items():
            setattr(self, name, np.load(self.directory / filename, mmap_mode="r"))
        self.group_index = {
            name: index for index, name in enumerate(self.manifest["edge_group_names"])
        }
        self.epoch_origin = int(self.manifest["epoch_origin_ms"])
        self.epoch_ms = self.config.epoch_hours * HOUR_MS
        if len(self.global_ids) != self.manifest["vector_count"]:
            raise ValueError("bitmap graph artifacts do not align")

    def _component_bounds(self, start: int, end: int) -> tuple[int, int]:
        return (
            int(np.searchsorted(self.timestamps, start, side="left")),
            int(np.searchsorted(self.timestamps, end, side="left")),
        )

    def _epoch_range(self, start: int, end: int) -> tuple[int, int]:
        return (
            (start - self.epoch_origin) // self.epoch_ms,
            (end - 1 - self.epoch_origin) // self.epoch_ms,
        )

    def _gateway_seeds(
        self, query: np.ndarray, start: int, end: int,
        config: TemporalBitmapSearchConfig,
    ) -> tuple[list[tuple[float, int]], int, int]:
        timestamps = self.timestamps[self.gateway_ids]
        left = int(np.searchsorted(timestamps, start, side="left"))
        right = int(np.searchsorted(timestamps, end, side="left"))
        if left >= right:
            return [], 0, 0
        first_epoch, last_epoch = self._epoch_range(start, end)
        available_rows = np.arange(left, right, dtype=np.int64)
        bitmap_matches = sum(
            _bitmap_intersects(
                self.portal_coverage_bitmaps[row], first_epoch, last_epoch
            ) for row in available_rows
        )
        count = min(config.gateway_candidates, len(available_rows))
        positions = np.unique(np.linspace(left, right - 1, count, dtype=np.int64))
        local_ids = np.asarray(self.gateway_ids[positions], dtype=np.int64)
        values = np.asarray(self.vectors[self.global_ids[local_ids]], dtype=np.float32)
        delta = values - query
        distances = np.einsum("ij,ij->i", delta, delta)
        seed_count = min(config.gateway_seeds, len(local_ids))
        selected = np.argpartition(distances, seed_count - 1)[:seed_count]
        selected = selected[np.argsort(distances[selected], kind="stable")]
        return [
            (float(distances[position]), int(local_ids[position]))
            for position in selected
        ], len(local_ids), bitmap_matches

    def _strict_component(
        self, query: np.ndarray, start: int, end: int,
        left: int, right: int, config: TemporalBitmapSearchConfig,
    ) -> dict[str, Any]:
        seeds, gateway_candidates, portal_matches = self._gateway_seeds(
            query, start, end, config
        )
        if not seeds:
            local_ids = np.arange(left, right, dtype=np.int64)
            ids, distances, computations = _exact_top_k(
                query, self.vectors, self.global_ids, local_ids, config.k
            )
            return {
                "local_ids": ids, "distances": distances,
                "distance_computations": computations, "nodes_visited": computations,
                "edges_examined": 0, "out_of_range_edges_rejected": 0,
                "bitmap_groups_considered": 0, "bitmap_groups_skipped": 0,
                "gateway_candidates": 0, "gateway_seeds": 0,
                "portal_bitmap_matches": 0, "method": "exact_no_gateway",
            }
        candidates = list(seeds)
        heapq.heapify(candidates)
        best = [(-distance, local_id) for distance, local_id in seeds]
        heapq.heapify(best)
        visited = {local_id for _, local_id in seeds}
        distance_computations = gateway_candidates
        edges_examined = rejected = groups_considered = groups_skipped = 0
        first_epoch, last_epoch = self._epoch_range(start, end)
        while candidates:
            current_distance, current = heapq.heappop(candidates)
            if len(best) >= config.ef_search and current_distance > -best[0][0]:
                break
            groups: list[tuple[int, np.ndarray]] = []
            if config.use_local_edges:
                groups.append((self.group_index["local"], self.local_neighbors[current]))
            if config.use_historical_edges:
                groups.append((self.group_index["historical"], self.historical_neighbors[current]))
            if config.use_future_edges:
                groups.append((self.group_index["future"], self.future_neighbors[current]))
            if config.use_range_edges:
                groups.extend(
                    (self.group_index[f"range_{scale}h"], np.concatenate((
                        self.range_left_neighbors[current, index],
                        self.range_right_neighbors[current, index],
                    ))) for index, scale in enumerate(self.config.range_scales())
                )
            if config.use_bridge_edges:
                groups.append((self.group_index["bridge"], np.concatenate((
                    self.bridge_left_neighbors[current].reshape(-1),
                    self.bridge_right_neighbors[current].reshape(-1),
                ))))
            if config.use_temporal_edges:
                groups.append((self.group_index["temporal"], self.temporal_neighbors[current]))
            for group, neighbors in groups:
                groups_considered += 1
                if config.use_bitmap_pruning and not _bitmap_intersects(
                    self.edge_epoch_bitmaps[current, group], first_epoch, last_epoch
                ):
                    groups_skipped += 1
                    continue
                for value in neighbors:
                    neighbor = int(value)
                    if neighbor < 0:
                        continue
                    edges_examined += 1
                    if neighbor < left or neighbor >= right:
                        rejected += 1
                        continue
                    if neighbor in visited:
                        continue
                    visited.add(neighbor)
                    delta = np.asarray(
                        self.vectors[int(self.global_ids[neighbor])], dtype=np.float32
                    ) - query
                    distance = float(np.dot(delta, delta))
                    distance_computations += 1
                    if len(best) < config.ef_search or distance < -best[0][0]:
                        heapq.heappush(candidates, (distance, neighbor))
                        heapq.heappush(best, (-distance, neighbor))
                        if len(best) > config.ef_search:
                            heapq.heappop(best)
        ordered = sorted((-distance, local_id) for distance, local_id in best)
        return {
            "local_ids": np.asarray([value[1] for value in ordered], dtype=np.int64),
            "distances": np.asarray([value[0] for value in ordered], dtype=np.float32),
            "distance_computations": distance_computations,
            "nodes_visited": len(visited), "edges_examined": edges_examined,
            "out_of_range_edges_rejected": rejected,
            "bitmap_groups_considered": groups_considered,
            "bitmap_groups_skipped": groups_skipped,
            "gateway_candidates": gateway_candidates, "gateway_seeds": len(seeds),
            "portal_bitmap_matches": portal_matches, "method": "strict_bitmap_graph",
        }

    def search(
        self, query: np.ndarray, intervals: Iterable[Iterable[int]],
        config: TemporalBitmapSearchConfig = TemporalBitmapSearchConfig(),
    ) -> dict[str, Any]:
        config.validate()
        query = np.asarray(query, dtype=np.float32)
        if query.shape != (self.manifest["dimensions"],) or not np.all(np.isfinite(query)):
            raise ValueError("query has invalid shape or values")
        components = []
        for start, end in normalize_intervals(intervals):
            left, right = self._component_bounds(start, end)
            if left < right:
                components.append((start, end, left, right))
        eligible_count = sum(right - left for _, _, left, right in components)
        diagnostics = {
            "distance_computations": 0, "nodes_visited": 0,
            "edges_examined": 0, "out_of_range_edges_rejected": 0,
            "bitmap_groups_considered": 0, "bitmap_groups_skipped": 0,
            "gateway_candidates": 0, "gateway_seeds": 0,
            "portal_bitmap_matches": 0,
        }
        candidate_parts = []
        methods = []
        started = time.perf_counter_ns()
        if eligible_count <= config.exact_threshold:
            local_ids = np.concatenate([
                np.arange(left, right, dtype=np.int64)
                for _, _, left, right in components
            ]) if components else np.empty(0, dtype=np.int64)
            ids, _, computations = _exact_top_k(
                query, self.vectors, self.global_ids, local_ids, config.k
            )
            candidate_parts.append(ids)
            diagnostics["distance_computations"] = computations
            diagnostics["nodes_visited"] = computations
            methods.append("exact")
        else:
            for start, end, left, right in components:
                if right - left <= config.exact_threshold:
                    ids, _, computations = _exact_top_k(
                        query, self.vectors, self.global_ids,
                        np.arange(left, right, dtype=np.int64), config.k,
                    )
                    candidate_parts.append(ids)
                    diagnostics["distance_computations"] += computations
                    diagnostics["nodes_visited"] += computations
                    methods.append("exact_component")
                else:
                    result = self._strict_component(
                        query, start, end, left, right, config
                    )
                    candidate_parts.append(result["local_ids"])
                    methods.append(result["method"])
                    for field in diagnostics:
                        diagnostics[field] += result[field]
        candidates = np.unique(np.concatenate(candidate_parts)) if candidate_parts else np.empty(0, dtype=np.int64)
        local_ids, distances, rerank = _exact_top_k(
            query, self.vectors, self.global_ids, candidates, config.k
        )
        total_ms = (time.perf_counter_ns() - started) / 1e6
        return {
            "ids": np.asarray(self.global_ids[local_ids], dtype=np.int64),
            "distances": distances,
            "method": "+".join(sorted(set(methods))) if methods else "empty",
            "eligible_count": eligible_count, "candidate_count": len(candidates),
            **diagnostics, "rerank_distance_computations": rerank,
            "total_ms": total_ms,
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build time-stretched bitmap RT-HNSW")
    parser.add_argument("--embeddings", type=Path, default=Path("Twitter_data/processed/database_vectors.npy"))
    parser.add_argument("--timestamps", type=Path, default=Path("Twitter_data/ann/timestamps_ms.npy"))
    parser.add_argument("--output", type=Path, default=Path("Twitter_data/ann/rolling_hnsw/bitmap_100k"))
    parser.add_argument("--vector-count", type=int, default=100_000)
    parser.add_argument("--epoch-hours", type=int, default=24)
    parser.add_argument("--rebuild", action="store_true")
    return parser


def build_main() -> None:
    args = build_parser().parse_args()
    vectors = np.load(args.embeddings, mmap_mode="r")
    timestamps = np.load(args.timestamps, mmap_mode="r")
    manifest = build_temporal_bitmap_index(
        vectors, timestamps, args.output,
        TemporalBitmapBuildConfig(
            vector_count=args.vector_count, epoch_hours=args.epoch_hours
        ),
        rebuild=args.rebuild,
    )
    print(json.dumps(manifest, indent=2))
