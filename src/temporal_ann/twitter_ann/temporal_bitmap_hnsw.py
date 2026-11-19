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
    # Fixed base degree budget: 4 local + 8 historical + 8 future
    # + 6 range + 4 bridge + 2 temporal = 32.
    local_semantic_degree: int = 4
    historical_semantic_degree: int = 8
    future_reciprocal_degree: int = 8
    local_window_hours: int = 24
    range_degree_per_side: int = 1
    range_scales_hours: str = "24,168,720"
    range_candidate_count: int = 64
    bridge_degree_per_side: int = 1
    bridge_scales_epochs: str = "1,16"
    epoch_hours: int = 24
    gateway_period: int = 512
    portal_semantic_degree: int = 4
    portal_epoch_radius: int = 1
    persist_unpacked_adjacency: bool = False
    persist_incremental_state: bool = False
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
            self.epoch_hours, self.gateway_period, self.portal_semantic_degree,
            self.portal_epoch_radius, self.construction_candidates,
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
    portal_ef_search: int = 16
    use_portal_navigation: bool = True
    use_bitmap_pruning: bool = True
    use_local_edges: bool = True
    use_historical_edges: bool = True
    use_future_edges: bool = True
    use_range_edges: bool = True
    use_bridge_edges: bool = True
    use_temporal_edges: bool = True

    def validate(self) -> None:
        if min(self.k, self.ef_search, self.gateway_candidates,
               self.gateway_seeds, self.portal_ef_search) <= 0:
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


def _pack_epoch_grouped_adjacency(
    epoch_ids: np.ndarray,
    grouped_rows: list[list[np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pack `(node, edge group) -> epoch blocks -> neighbor IDs` as CSR."""
    group_block_offsets = [0]
    block_epochs: list[int] = []
    block_neighbor_offsets = [0]
    packed_neighbors: list[int] = []
    for row_groups in grouped_rows:
        for neighbors in row_groups:
            valid = np.asarray(neighbors, dtype=np.int64).reshape(-1)
            valid = valid[valid >= 0]
            if len(valid):
                order = np.lexsort((valid, epoch_ids[valid]))
                valid = valid[order]
                epochs = epoch_ids[valid]
                starts = np.r_[0, np.flatnonzero(np.diff(epochs)) + 1]
                ends = np.r_[starts[1:], len(valid)]
                for start, end in zip(starts, ends):
                    block_epochs.append(int(epochs[start]))
                    packed_neighbors.extend(map(int, valid[start:end]))
                    block_neighbor_offsets.append(len(packed_neighbors))
            group_block_offsets.append(len(block_epochs))
    return (
        np.asarray(group_block_offsets, dtype=np.uint32),
        np.asarray(block_epochs, dtype=np.int32),
        np.asarray(block_neighbor_offsets, dtype=np.uint32),
        np.asarray(packed_neighbors, dtype=np.int32),
    )


def build_temporal_bitmap_index(
    vectors: np.ndarray,
    timestamps: np.ndarray,
    output: Path,
    config: TemporalBitmapBuildConfig,
    *,
    rebuild: bool = False,
    progress_every: int = 5_000,
    resume_directory: Path | None = None,
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
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("config") != asdict(config):
            raise ValueError("persisted temporal-bitmap configuration differs")
        return manifest

    resume_directory = Path(resume_directory) if resume_directory else None
    if resume_directory is not None:
        if not config.persist_incremental_state:
            raise ValueError("resume requires persist_incremental_state")
        if np.any(np.diff(timestamps[:count]) < 0):
            raise ValueError("resumed timestamps must already be ordered")
        global_ids = np.arange(count, dtype=np.int64)
    else:
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
    start_local_id = 0
    if resume_directory is not None:
        previous_manifest = json.loads(
            (resume_directory / "manifest.json").read_text(encoding="utf-8")
        )
        if previous_manifest["config"] != asdict(config):
            raise ValueError("resumed bitmap index configuration differs")
        previous_files = previous_manifest["files"]
        start_local_id = int(previous_manifest["vector_count"])
        if start_local_id > count:
            raise ValueError("resumed index is larger than input")
        copy_names = {
            "local_neighbors": local,
            "historical_neighbors": historical,
            "future_neighbors": future,
            "temporal_neighbors": temporal,
            "range_left_neighbors": range_left,
            "range_right_neighbors": range_right,
            "bridge_left_neighbors": bridge_left,
            "bridge_right_neighbors": bridge_right,
            "internal_local_distances": local_dist,
            "internal_historical_distances": historical_dist,
            "internal_future_distances": future_dist,
            "internal_range_left_scores": range_left_scores,
            "internal_range_right_scores": range_right_scores,
            "internal_bridge_left_scores": bridge_left_scores,
            "internal_bridge_right_scores": bridge_right_scores,
        }
        for name, target in copy_names.items():
            previous = np.load(resume_directory / previous_files[name], mmap_mode="r")
            target[:start_local_id] = previous
        candidate_index = faiss.read_index(
            str(resume_directory / previous_files["internal_candidate_index"])
        )
        if int(candidate_index.ntotal) != start_local_id:
            raise ValueError("resumed candidate index count differs")
    started = time.perf_counter()

    for local_id in range(start_local_id, count):
        global_id = int(global_ids[local_id])
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
    grouped_rows: list[list[np.ndarray]] = []
    for row in range(count):
        row_groups = [local[row], historical[row], future[row]]
        row_groups.extend(
            np.concatenate((range_left[row, index], range_right[row, index]))
            for index, _ in enumerate(range_scales)
        )
        row_groups.extend((
            np.concatenate((
                bridge_left[row].reshape(-1), bridge_right[row].reshape(-1)
            )),
            temporal[row],
        ))
        grouped_rows.append(row_groups)
        for group, neighbors in enumerate(row_groups):
            for neighbor in neighbors:
                if neighbor >= 0:
                    _set_bitmap(
                        edge_epoch_bitmaps, row, group, int(epoch_ids[int(neighbor)])
                    )
    (
        packed_group_block_offsets,
        packed_block_epochs,
        packed_block_neighbor_offsets,
        packed_neighbors,
    ) = _pack_epoch_grouped_adjacency(epoch_ids, grouped_rows)

    portal_count = len(gateway_ids)
    portal_coverage_bitmaps = np.zeros((portal_count, bitmap_words), dtype=np.uint64)
    portal_semantic_neighbors = np.full(
        (portal_count, config.portal_semantic_degree), -1, dtype=np.int32
    )
    portal_semantic_distances = np.full(
        (portal_count, config.portal_semantic_degree), np.inf, dtype=np.float32
    )
    portal_temporal_neighbors = np.full((portal_count, 2), -1, dtype=np.int32)
    portal_epochs = epoch_ids[gateway_ids]
    for portal_row, local_id in enumerate(gateway_ids):
        _set_bitmap(portal_coverage_bitmaps[:, None, :], portal_row, 0, int(epoch_ids[local_id]))
        portal_coverage_bitmaps[portal_row] |= np.bitwise_or.reduce(
            edge_epoch_bitmaps[local_id], axis=0
        )
        if portal_row:
            portal_temporal_neighbors[portal_row, 0] = portal_row - 1
            portal_temporal_neighbors[portal_row - 1, 1] = portal_row
        candidates = np.flatnonzero(
            (np.abs(portal_epochs - portal_epochs[portal_row]) <= config.portal_epoch_radius)
            & (np.arange(portal_count) < portal_row)
        )
        if len(candidates):
            difference = np.asarray(
                vectors[global_ids[gateway_ids[candidates]]], dtype=np.float32
            ) - np.asarray(vectors[global_ids[local_id]], dtype=np.float32)
            distances = np.einsum("ij,ij->i", difference, difference)
            selected_count = min(config.portal_semantic_degree, len(candidates))
            selected = np.argpartition(distances, selected_count - 1)[:selected_count]
            selected = selected[np.argsort(distances[selected], kind="stable")]
            for position in selected:
                other = int(candidates[position])
                distance = float(distances[position])
                _grow_edge(
                    portal_semantic_neighbors, portal_semantic_distances,
                    portal_row, other, distance,
                )
                _grow_edge(
                    portal_semantic_neighbors, portal_semantic_distances,
                    other, portal_row, distance,
                )
                other_epoch = int(portal_epochs[other])
                _set_bitmap(portal_coverage_bitmaps[:, None, :], portal_row, 0, other_epoch)
                _set_bitmap(
                    portal_coverage_bitmaps[:, None, :], other, 0,
                    int(portal_epochs[portal_row]),
                )
    epoch_portal_offsets = [0]
    epoch_portal_rows: list[int] = []
    for epoch in range(epoch_count):
        epoch_portal_rows.extend(map(int, np.flatnonzero(portal_epochs == epoch)))
        epoch_portal_offsets.append(len(epoch_portal_rows))

    arrays = {
        "global_ids": global_ids,
        "timestamps": ordered_timestamps,
        "epoch_ids": epoch_ids,
        "gateway_ids": gateway_ids,
        "edge_epoch_bitmaps": edge_epoch_bitmaps,
        "packed_group_block_offsets": packed_group_block_offsets,
        "packed_block_epochs": packed_block_epochs,
        "packed_block_neighbor_offsets": packed_block_neighbor_offsets,
        "packed_neighbors": packed_neighbors,
        "portal_coverage_bitmaps": portal_coverage_bitmaps,
        "portal_semantic_neighbors": portal_semantic_neighbors,
        "portal_temporal_neighbors": portal_temporal_neighbors,
        "portal_epochs": portal_epochs,
        "epoch_portal_offsets": np.asarray(epoch_portal_offsets, dtype=np.int64),
        "epoch_portal_rows": np.asarray(epoch_portal_rows, dtype=np.int32),
    }
    if config.persist_unpacked_adjacency or config.persist_incremental_state:
        arrays.update({
            "local_neighbors": local,
            "historical_neighbors": historical,
            "future_neighbors": future,
            "temporal_neighbors": temporal,
            "range_left_neighbors": range_left,
            "range_right_neighbors": range_right,
            "bridge_left_neighbors": bridge_left,
            "bridge_right_neighbors": bridge_right,
        })
    if config.persist_incremental_state:
        arrays.update({
            "internal_local_distances": local_dist,
            "internal_historical_distances": historical_dist,
            "internal_future_distances": future_dist,
            "internal_range_left_scores": range_left_scores,
            "internal_range_right_scores": range_right_scores,
            "internal_bridge_left_scores": bridge_left_scores,
            "internal_bridge_right_scores": bridge_right_scores,
        })
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        files = {name: f"{name}.npy" for name in arrays}
        for name, array in arrays.items():
            np.save(temporary / files[name], array, allow_pickle=False)
        if config.persist_incremental_state:
            files["internal_candidate_index"] = "construction.faiss"
            faiss.write_index(
                candidate_index, str(temporary / files["internal_candidate_index"])
            )
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
            if filename.endswith(".npy"):
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
    ) -> tuple[list[tuple[float, int]], int, int, int, int]:
        first_epoch, last_epoch = self._epoch_range(start, end)
        first_epoch = max(0, first_epoch)
        last_epoch = min(len(self.epoch_portal_offsets) - 2, last_epoch)
        if first_epoch > last_epoch:
            return [], 0, 0, 0, 0
        parts = [
            self.epoch_portal_rows[
                self.epoch_portal_offsets[epoch]:self.epoch_portal_offsets[epoch + 1]
            ]
            for epoch in range(first_epoch, last_epoch + 1)
        ]
        available_rows = np.concatenate(parts) if parts else np.empty(0, dtype=np.int32)
        if len(available_rows):
            local_ids = self.gateway_ids[available_rows]
            exact = (
                (self.timestamps[local_ids] >= start)
                & (self.timestamps[local_ids] < end)
            )
            available_rows = np.asarray(available_rows[exact], dtype=np.int64)
        if not len(available_rows):
            return [], 0, 0, 0, 0
        bitmap_matches = sum(
            _bitmap_intersects(
                self.portal_coverage_bitmaps[row], first_epoch, last_epoch
            ) for row in available_rows
        )
        count = min(config.gateway_candidates, len(available_rows))
        positions = np.unique(np.linspace(0, len(available_rows) - 1, count, dtype=np.int64))
        sampled_rows = available_rows[positions]
        sampled_local_ids = np.asarray(self.gateway_ids[sampled_rows], dtype=np.int64)
        values = np.asarray(
            self.vectors[self.global_ids[sampled_local_ids]], dtype=np.float32
        )
        delta = values - query
        distances = np.einsum("ij,ij->i", delta, delta)
        seed_count = min(config.gateway_seeds, len(sampled_rows))
        selected = np.argpartition(distances, seed_count - 1)[:seed_count]
        portal_candidates = [
            (float(distances[position]), int(sampled_rows[position]))
            for position in selected
        ]
        if not config.use_portal_navigation:
            direct = sorted(portal_candidates)
            return [
                (distance, int(self.gateway_ids[row]))
                for distance, row in direct
            ], len(sampled_rows), bitmap_matches, len(direct), 0
        heapq.heapify(portal_candidates)
        portal_best = [(-distance, row) for distance, row in portal_candidates]
        heapq.heapify(portal_best)
        visited = {row for _, row in portal_candidates}
        cache = {
            int(row): float(distance)
            for row, distance in zip(sampled_rows, distances)
        }
        edges_examined = 0
        while portal_candidates:
            current_distance, current_row = heapq.heappop(portal_candidates)
            if (len(portal_best) >= config.portal_ef_search
                    and current_distance > -portal_best[0][0]):
                break
            neighbors = np.concatenate((
                self.portal_semantic_neighbors[current_row],
                self.portal_temporal_neighbors[current_row],
            ))
            for value in neighbors:
                neighbor_row = int(value)
                if neighbor_row < 0:
                    continue
                edges_examined += 1
                if neighbor_row in visited:
                    continue
                local_id = int(self.gateway_ids[neighbor_row])
                timestamp = int(self.timestamps[local_id])
                if timestamp < start or timestamp >= end:
                    continue
                visited.add(neighbor_row)
                distance = cache.get(neighbor_row)
                if distance is None:
                    difference = np.asarray(
                        self.vectors[int(self.global_ids[local_id])], dtype=np.float32
                    ) - query
                    distance = float(np.dot(difference, difference))
                    cache[neighbor_row] = distance
                if (len(portal_best) < config.portal_ef_search
                        or distance < -portal_best[0][0]):
                    heapq.heappush(portal_candidates, (distance, neighbor_row))
                    heapq.heappush(portal_best, (-distance, neighbor_row))
                    if len(portal_best) > config.portal_ef_search:
                        heapq.heappop(portal_best)
        ordered = sorted((-distance, row) for distance, row in portal_best)
        seeds = [
            (distance, int(self.gateway_ids[row]))
            for distance, row in ordered[:config.gateway_seeds]
        ]
        return seeds, len(sampled_rows), bitmap_matches, len(visited), edges_examined

    def _packed_group_neighbors(
        self,
        node: int,
        group: int,
        first_epoch: int,
        last_epoch: int,
        prune: bool,
    ) -> np.ndarray:
        flat_group = node * len(self.group_index) + group
        block_left = int(self.packed_group_block_offsets[flat_group])
        block_right = int(self.packed_group_block_offsets[flat_group + 1])
        if block_left >= block_right:
            return np.empty(0, dtype=np.int32)
        if prune:
            epochs = self.packed_block_epochs[block_left:block_right]
            selected_left = block_left + int(np.searchsorted(
                epochs, first_epoch, side="left"
            ))
            selected_right = block_left + int(np.searchsorted(
                epochs, last_epoch, side="right"
            ))
            if selected_left >= selected_right:
                return np.empty(0, dtype=np.int32)
            block_left, block_right = selected_left, selected_right
        neighbor_left = int(self.packed_block_neighbor_offsets[block_left])
        neighbor_right = int(self.packed_block_neighbor_offsets[block_right])
        return np.asarray(self.packed_neighbors[neighbor_left:neighbor_right])

    def _strict_component(
        self, query: np.ndarray, start: int, end: int,
        left: int, right: int, config: TemporalBitmapSearchConfig,
    ) -> dict[str, Any]:
        (
            seeds, gateway_candidates, portal_matches,
            portal_nodes_visited, portal_edges_examined,
        ) = self._gateway_seeds(query, start, end, config)
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
                "portal_bitmap_matches": 0, "portal_nodes_visited": 0,
                "portal_edges_examined": 0, "method": "exact_no_gateway",
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
            groups: list[int] = []
            if config.use_local_edges:
                groups.append(self.group_index["local"])
            if config.use_historical_edges:
                groups.append(self.group_index["historical"])
            if config.use_future_edges:
                groups.append(self.group_index["future"])
            if config.use_range_edges:
                groups.extend(
                    self.group_index[f"range_{scale}h"]
                    for scale in self.config.range_scales()
                )
            if config.use_bridge_edges:
                groups.append(self.group_index["bridge"])
            if config.use_temporal_edges:
                groups.append(self.group_index["temporal"])
            for group in groups:
                groups_considered += 1
                neighbors = self._packed_group_neighbors(
                    current, group, first_epoch, last_epoch,
                    config.use_bitmap_pruning,
                )
                if not len(neighbors):
                    groups_skipped += int(config.use_bitmap_pruning)
                    continue
                edges_examined += len(neighbors)
                in_range = neighbors[(neighbors >= left) & (neighbors < right)]
                rejected += len(neighbors) - len(in_range)
                unseen = np.asarray(
                    [int(value) for value in in_range if int(value) not in visited],
                    dtype=np.int64,
                )
                if not len(unseen):
                    continue
                visited.update(map(int, unseen))
                values = np.asarray(
                    self.vectors[self.global_ids[unseen]], dtype=np.float32
                )
                delta = values - query
                distances = np.einsum("ij,ij->i", delta, delta)
                distance_computations += len(unseen)
                for neighbor, distance in zip(unseen, distances):
                    neighbor, distance = int(neighbor), float(distance)
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
            "portal_bitmap_matches": portal_matches,
            "portal_nodes_visited": portal_nodes_visited,
            "portal_edges_examined": portal_edges_examined,
            "method": "strict_bitmap_graph",
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
            "portal_bitmap_matches": 0, "portal_nodes_visited": 0,
            "portal_edges_examined": 0,
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
    parser.add_argument("--output", type=Path, default=Path("Twitter_data/ann/rolling_hnsw/bitmap_100k_v3"))
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
