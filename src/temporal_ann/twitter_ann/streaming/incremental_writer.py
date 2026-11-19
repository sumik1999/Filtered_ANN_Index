from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import faiss
import numpy as np

from ..rolling_hnsw import RollingBuildConfig, RollingTemporalHNSW, _add_bounded_edge
from .ordering import canonical_json, validate_embedded_event


class RollingTemporalHNSWWriter:
    """Durable single-writer append API for the flat multi-scale RT-HNSW.

    Generations are immutable and complete. Graph updates are incremental in
    memory; publication currently rewrites complete NumPy artifacts. This is a
    correctness-first implementation to be replaced by chunked copy-on-write
    storage once crash/replay behavior is established.
    """

    def __init__(self, root: Path):
        self.root = Path(root)
        current = self.root / "CURRENT"
        if not current.exists():
            raise FileNotFoundError(f"no streaming index at {self.root}")
        generation_name = current.read_text(encoding="utf-8").strip()
        if not generation_name:
            raise ValueError("CURRENT does not name a generation")
        self._load_generation(self.root / "generations" / generation_name)
        self._recover_wals()

    @classmethod
    def create(
        cls,
        root: Path,
        config: RollingBuildConfig,
        dimensions: int = 384,
        index_name: str = "twitter-rt-hnsw",
    ) -> "RollingTemporalHNSWWriter":
        config.validate()
        if config.hierarchy_enabled:
            raise ValueError("incremental writer currently supports the flat topology only")
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")
        root = Path(root)
        if (root / "CURRENT").exists():
            writer = cls(root)
            if writer.config != config or writer.dimensions != dimensions:
                raise ValueError("existing streaming index configuration differs")
            return writer
        root.mkdir(parents=True, exist_ok=True)
        (root / "generations").mkdir(exist_ok=True)
        (root / "wal").mkdir(exist_ok=True)
        writer = cls.__new__(cls)
        writer.root = root
        writer.config = config
        writer.dimensions = dimensions
        writer.index_name = index_name
        writer.current_generation = None
        writer.last_sequence = -1
        writer.last_order_key: tuple[int, str] | None = None
        writer.checkpoint: dict[str, Any] = {}
        writer.event_metadata: list[dict[str, Any]] = []
        writer._initialize_empty()
        writer._publish_generation(recovered=False)
        return cls(root)

    def _initialize_empty(self) -> None:
        count = 0
        config = self.config
        scales = config.range_scales()
        self.vectors = np.empty((count, self.dimensions), dtype=np.float32)
        self.timestamps = np.empty(count, dtype=np.int64)
        self.global_ids = np.empty(count, dtype=np.int64)
        self.semantic_neighbors = np.full((count, config.semantic_degree), -1, dtype=np.int32)
        self.semantic_distances = np.full((count, config.semantic_degree), np.inf, dtype=np.float32)
        self.temporal_neighbors = np.full((count, 2), -1, dtype=np.int32)
        shape = (count, len(scales), config.range_degree_per_side)
        self.range_left_neighbors = np.full(shape, -1, dtype=np.int32)
        self.range_right_neighbors = np.full(shape, -1, dtype=np.int32)
        self.range_left_scores = np.full(shape, np.inf, dtype=np.float32)
        self.range_right_scores = np.full(shape, np.inf, dtype=np.float32)
        self.gateway_ids = np.empty(0, dtype=np.int32)
        self.gateway_rows = np.full(count, -1, dtype=np.int32)
        self.gateway_neighbors = np.full((0, config.gateway_degree), -1, dtype=np.int32)
        self.gateway_distances = np.full((0, config.gateway_degree), np.inf, dtype=np.float32)
        self.candidate_index = self._new_candidate_index()

    def _new_candidate_index(self):
        faiss.omp_set_num_threads(self.config.threads)
        index = faiss.IndexHNSWFlat(
            self.dimensions, self.config.candidate_index_m, faiss.METRIC_L2
        )
        index.hnsw.efConstruction = self.config.ef_construction
        return index

    def _load_generation(self, directory: Path) -> None:
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("streaming_schema_version") != 1:
            raise ValueError("unsupported streaming generation")
        self.current_generation = directory.name
        self.config = RollingBuildConfig(**manifest["config"])
        self.dimensions = int(manifest["dimensions"])
        self.index_name = manifest["index_name"]
        self.last_sequence = int(manifest["last_index_sequence"])
        key = manifest.get("last_order_key")
        self.last_order_key = (int(key[0]), str(key[1])) if key else None
        self.checkpoint = manifest.get("checkpoint", {})
        files = manifest["files"]
        for filename, expected in manifest.get("sha256", {}).items():
            digest = hashlib.sha256()
            with (directory / filename).open("rb") as handle:
                while chunk := handle.read(1024 * 1024):
                    digest.update(chunk)
            if digest.hexdigest() != expected:
                raise ValueError(f"streaming artifact checksum mismatch: {filename}")
        for attribute in (
            "vectors", "timestamps", "global_ids", "semantic_neighbors",
            "semantic_distances", "temporal_neighbors", "range_left_neighbors",
            "range_right_neighbors", "range_left_scores", "range_right_scores",
            "gateway_ids", "gateway_rows", "gateway_neighbors", "gateway_distances",
        ):
            setattr(self, attribute, np.load(directory / files[attribute]).copy())
        metadata_path = directory / files["event_metadata"]
        self.event_metadata = [
            json.loads(line) for line in metadata_path.read_text(encoding="utf-8").splitlines()
            if line
        ]
        self.candidate_index = faiss.read_index(str(directory / files["candidate_index"]))
        self._validate_loaded_state()

    def _validate_loaded_state(self) -> None:
        count = len(self.vectors)
        if self.vectors.shape != (count, self.dimensions):
            raise ValueError("invalid persisted vector shape")
        if len(self.timestamps) != count or len(self.event_metadata) != count:
            raise ValueError("streaming generation arrays are not aligned")
        if count != self.last_sequence + 1:
            raise ValueError("streaming generation sequence is not contiguous")
        if np.any(np.diff(self.timestamps) < 0):
            raise ValueError("streaming generation timestamps are not ordered")
        if int(self.candidate_index.ntotal) != count:
            raise ValueError("construction index count differs from graph")

    @property
    def vector_count(self) -> int:
        return len(self.vectors)

    def _prepare_capacity(self, events: list[dict[str, Any]]) -> int:
        old_count = self.vector_count
        target = old_count + len(events)
        if target > self.config.vector_count:
            raise ValueError("streaming index reached configured vector capacity")
        new_vectors = np.asarray([event["embedding"] for event in events], dtype=np.float32)
        self.vectors = np.concatenate((self.vectors, new_vectors), axis=0)
        self.timestamps = np.concatenate((
            self.timestamps,
            np.asarray([event["event_time_ms"] for event in events], dtype=np.int64),
        ))
        self.global_ids = np.arange(target, dtype=np.int64)

        def grow(array: np.ndarray, shape: tuple[int, ...], fill: int | float):
            result = np.full(shape, fill, dtype=array.dtype)
            if len(array):
                result[:old_count] = array
            return result

        self.semantic_neighbors = grow(
            self.semantic_neighbors, (target, self.config.semantic_degree), -1
        )
        self.semantic_distances = grow(
            self.semantic_distances, (target, self.config.semantic_degree), np.inf
        )
        self.temporal_neighbors = grow(self.temporal_neighbors, (target, 2), -1)
        range_shape = (
            target, len(self.config.range_scales()), self.config.range_degree_per_side
        )
        self.range_left_neighbors = grow(self.range_left_neighbors, range_shape, -1)
        self.range_right_neighbors = grow(self.range_right_neighbors, range_shape, -1)
        self.range_left_scores = grow(self.range_left_scores, range_shape, np.inf)
        self.range_right_scores = grow(self.range_right_scores, range_shape, np.inf)
        self.gateway_rows = grow(self.gateway_rows, (target,), -1)

        target_gateways = target // self.config.gateway_period
        old_gateways = len(self.gateway_ids)
        ids = np.empty(target_gateways, dtype=np.int32)
        if old_gateways:
            ids[:old_gateways] = self.gateway_ids
        self.gateway_ids = ids
        gateway_shape = (target_gateways, self.config.gateway_degree)
        gateway_neighbors = np.full(gateway_shape, -1, dtype=np.int32)
        gateway_distances = np.full(gateway_shape, np.inf, dtype=np.float32)
        if old_gateways:
            gateway_neighbors[:old_gateways] = self.gateway_neighbors
            gateway_distances[:old_gateways] = self.gateway_distances
        self.gateway_neighbors = gateway_neighbors
        self.gateway_distances = gateway_distances
        return old_count

    def _insert_one(self, local_id: int) -> None:
        config = self.config
        vector = np.ascontiguousarray(self.vectors[local_id]).reshape(1, -1)
        all_candidates = np.empty(0, dtype=np.int64)
        if local_id:
            fetch = min(local_id, config.construction_candidates)
            params = faiss.SearchParametersHNSW(
                efSearch=max(config.ef_construction, fetch)
            )
            found_distances, found_ids = self.candidate_index.search(
                vector, fetch, params=params
            )
            valid = found_ids[0] >= 0
            all_candidates = np.asarray(found_ids[0][valid], dtype=np.int64)
            distances = found_distances[0][valid]
            for neighbor, distance in zip(
                all_candidates[:config.semantic_degree],
                distances[:config.semantic_degree],
            ):
                neighbor_id, value = int(neighbor), float(distance)
                _add_bounded_edge(
                    self.semantic_neighbors, self.semantic_distances,
                    local_id, neighbor_id, value,
                )
                _add_bounded_edge(
                    self.semantic_neighbors, self.semantic_distances,
                    neighbor_id, local_id, value,
                )
            self.temporal_neighbors[local_id, 0] = local_id - 1
            self.temporal_neighbors[local_id - 1, 1] = local_id

            timestamp = int(self.timestamps[local_id])
            for scale_index, scale_hours in enumerate(config.range_scales()):
                window_ms = scale_hours * 3_600_000
                first = int(np.searchsorted(
                    self.timestamps[:local_id], timestamp - window_ms, side="left"
                ))
                population = local_id - first
                recent_count = min(config.range_candidate_count, population)
                recent = (
                    np.unique(np.linspace(first, local_id - 1, recent_count, dtype=np.int64))
                    if recent_count else np.empty(0, dtype=np.int64)
                )
                immediate = np.arange(max(first, local_id - 16), local_id, dtype=np.int64)
                pool = np.unique(np.concatenate((all_candidates, recent, immediate)))
                if not len(pool):
                    continue
                delta = self.vectors[pool] - vector[0]
                vector_distances = np.einsum("ij,ij->i", delta, delta)
                gaps = timestamp - self.timestamps[pool]
                coeligibility = np.maximum(0.0, 1.0 - gaps / window_ms)
                valid_range = coeligibility > 0.0
                pool = pool[valid_range]
                scores = vector_distances[valid_range] / (
                    coeligibility[valid_range] + 1e-6
                )
                selected_count = min(config.range_degree_per_side, len(pool))
                if not selected_count:
                    continue
                selected = np.argpartition(scores, selected_count - 1)[:selected_count]
                selected = selected[np.argsort(scores[selected], kind="stable")]
                for position in selected:
                    neighbor_id, score = int(pool[position]), float(scores[position])
                    _add_bounded_edge(
                        self.range_left_neighbors[:, scale_index, :],
                        self.range_left_scores[:, scale_index, :],
                        local_id, neighbor_id, score,
                    )
                    _add_bounded_edge(
                        self.range_right_neighbors[:, scale_index, :],
                        self.range_right_scores[:, scale_index, :],
                        neighbor_id, local_id, score,
                    )

        if (local_id + 1) % config.gateway_period == 0:
            row = (local_id + 1) // config.gateway_period - 1
            self.gateway_ids[row] = local_id
            self.gateway_rows[local_id] = row
            if row:
                prior_ids = self.gateway_ids[:row].astype(np.int64, copy=False)
                delta = self.vectors[prior_ids] - vector[0]
                distances = np.einsum("ij,ij->i", delta, delta)
                selected_count = min(config.gateway_degree, row)
                selected = np.argpartition(distances, selected_count - 1)[:selected_count]
                selected = selected[np.argsort(distances[selected], kind="stable")]
                for prior_row in selected:
                    prior_row = int(prior_row)
                    distance = float(distances[prior_row])
                    _add_bounded_edge(
                        self.gateway_neighbors, self.gateway_distances,
                        row, prior_row, distance,
                    )
                    _add_bounded_edge(
                        self.gateway_neighbors, self.gateway_distances,
                        prior_row, row, distance,
                    )
        self.candidate_index.add(vector)

    def _validate_batch(self, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        accepted: list[dict[str, Any]] = []
        expected = self.last_sequence + 1
        prior_key = self.last_order_key
        previous_input_sequence: int | None = None
        for event in events:
            validate_embedded_event(event)
            if event.get("index_name") != self.index_name:
                raise ValueError("event index_name does not match writer")
            sequence = event.get("index_sequence")
            if not isinstance(sequence, int) or isinstance(sequence, bool):
                raise ValueError("ordered event requires an integer index_sequence")
            if previous_input_sequence is not None and sequence <= previous_input_sequence:
                raise ValueError("input batch sequences must be strictly increasing")
            previous_input_sequence = sequence
            if sequence <= self.last_sequence:
                metadata = self.event_metadata[sequence]
                if metadata["content_sha256"] != event["content_sha256"]:
                    raise ValueError("replayed sequence has conflicting content")
                continue
            if sequence != expected:
                raise ValueError(f"index sequence gap: expected {expected}, received {sequence}")
            key = (int(event["event_time_ms"]), str(event["tweet_id"]))
            if prior_key is not None and key < prior_key:
                raise ValueError("ordered event key decreased")
            vector = np.asarray(event["embedding"], dtype=np.float32)
            if vector.shape != (self.dimensions,) or not np.all(np.isfinite(vector)):
                raise ValueError("event embedding is incompatible with index")
            accepted.append(event)
            expected += 1
            prior_key = key
        return accepted

    def append_batch(
        self,
        events: Iterable[dict[str, Any]],
        checkpoint: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        values = [dict(event) for event in events]
        accepted = self._validate_batch(values)
        if not accepted:
            if checkpoint is not None and dict(checkpoint) != self.checkpoint:
                self.checkpoint = dict(checkpoint)
                generation = self._publish_generation(recovered=False)
            else:
                generation = self.current_generation
            return {
                "appended": 0,
                "last_index_sequence": self.last_sequence,
                "generation": generation,
            }
        first, last = accepted[0]["index_sequence"], accepted[-1]["index_sequence"]
        wal_path = self.root / "wal" / f"{first:020d}-{last:020d}.jsonl"
        temporary_wal = wal_path.with_suffix(".tmp")
        with temporary_wal.open("wb") as handle:
            for event in accepted:
                handle.write(canonical_json(event) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_wal, wal_path)
        self._fsync_directory(wal_path.parent)

        try:
            old_count = self._prepare_capacity(accepted)
            for offset, event in enumerate(accepted):
                local_id = old_count + offset
                self._insert_one(local_id)
                self.event_metadata.append({
                    "index_sequence": event["index_sequence"],
                    "tweet_id": event["tweet_id"],
                    "source_event_id": event["source_event_id"],
                    "content_sha256": event["content_sha256"],
                })
                self.last_sequence = int(event["index_sequence"])
                self.last_order_key = (
                    int(event["event_time_ms"]), str(event["tweet_id"])
                )
            if checkpoint is not None:
                self.checkpoint = dict(checkpoint)
            generation = self._publish_generation(recovered=False)
            wal_path.unlink(missing_ok=True)
            self._fsync_directory(wal_path.parent)
            return {
                "appended": len(accepted),
                "last_index_sequence": self.last_sequence,
                "generation": generation,
            }
        except BaseException:
            self._load_generation(self.root / "generations" / self.current_generation)
            raise

    def _recover_wals(self) -> None:
        for path in sorted((self.root / "wal").glob("*.jsonl")):
            events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            accepted = self._validate_batch(events)
            if not accepted:
                path.unlink()
                continue
            # append_batch rewrites the same deterministic WAL and publishes it.
            self.append_batch(accepted, checkpoint={"recovered_from_wal": path.name})

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def _publish_generation(self, recovered: bool) -> str:
        generation_number = 0 if self.current_generation is None else int(self.current_generation) + 1
        generation_name = f"{generation_number:020d}"
        generations = self.root / "generations"
        temporary = Path(tempfile.mkdtemp(prefix=f".{generation_name}-", dir=generations))
        files = {
            name: f"{name}.npy" for name in (
                "vectors", "timestamps", "global_ids", "semantic_neighbors",
                "semantic_distances", "temporal_neighbors", "range_left_neighbors",
                "range_right_neighbors", "range_left_scores", "range_right_scores",
                "gateway_ids", "gateway_rows", "gateway_neighbors", "gateway_distances",
            )
        }
        files.update({
            "levels": "levels.npy",
            "upper_semantic_neighbors": "upper_semantic_neighbors.npy",
            "upper_temporal_neighbors": "upper_temporal_neighbors.npy",
            "upper_range_left_neighbors": "upper_range_left_neighbors.npy",
            "upper_range_right_neighbors": "upper_range_right_neighbors.npy",
            "candidate_index": "construction.faiss",
            "event_metadata": "event_metadata.jsonl",
        })
        arrays = {
            name: getattr(self, name) for name in (
                "vectors", "timestamps", "global_ids", "semantic_neighbors",
                "semantic_distances", "temporal_neighbors", "range_left_neighbors",
                "range_right_neighbors", "range_left_scores", "range_right_scores",
                "gateway_ids", "gateway_rows", "gateway_neighbors", "gateway_distances",
            )
        }
        count = self.vector_count
        arrays.update({
            "levels": np.zeros(count, dtype=np.int8),
            "upper_semantic_neighbors": np.empty((count, 0, 0), dtype=np.int32),
            "upper_temporal_neighbors": np.empty((count, 0, 2), dtype=np.int32),
            "upper_range_left_neighbors": np.empty((count, 0, 0), dtype=np.int32),
            "upper_range_right_neighbors": np.empty((count, 0, 0), dtype=np.int32),
        })
        try:
            for name, array in arrays.items():
                np.save(temporary / files[name], array, allow_pickle=False)
            faiss.write_index(self.candidate_index, str(temporary / files["candidate_index"]))
            metadata_path = temporary / files["event_metadata"]
            with metadata_path.open("wb") as handle:
                for value in self.event_metadata:
                    handle.write(canonical_json(value) + b"\n")
            checksums = {}
            for filename in files.values():
                digest = hashlib.sha256()
                with (temporary / filename).open("rb") as handle:
                    while chunk := handle.read(1024 * 1024):
                        digest.update(chunk)
                checksums[filename] = digest.hexdigest()
            manifest = {
                "schema_version": 1,
                "streaming_schema_version": 1,
                "backend": "flat rolling temporal HNSW incremental generation",
                "index_name": self.index_name,
                "metric": "squared L2",
                "vector_count": count,
                "dimensions": self.dimensions,
                "dtype": "float32",
                "ordered_arrival": True,
                "timestamp_order_key": "(event_time_ms, tweet_id)",
                "config": asdict(self.config),
                "gateway_count": len(self.gateway_ids),
                "range_scales_hours": list(self.config.range_scales()),
                "hierarchy_level_counts": {},
                "hierarchy_range_scale_by_level": {},
                "last_index_sequence": self.last_sequence,
                "last_order_key": list(self.last_order_key) if self.last_order_key else None,
                "checkpoint": self.checkpoint,
                "recovered": recovered,
                "created_at_ms": int(time.time() * 1000),
                "files": files,
                "sha256": checksums,
            }
            manifest["artifact_size_bytes"] = sum(
                (temporary / filename).stat().st_size for filename in files.values()
            )
            manifest_path = temporary / "manifest.json"
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            for path in temporary.iterdir():
                with path.open("rb") as handle:
                    os.fsync(handle.fileno())
            self._fsync_directory(temporary)
            final = generations / generation_name
            temporary.replace(final)
            self._fsync_directory(generations)
            pointer_tmp = self.root / ".CURRENT.tmp"
            with pointer_tmp.open("w", encoding="utf-8") as handle:
                handle.write(generation_name + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(pointer_tmp, self.root / "CURRENT")
            self._fsync_directory(self.root)
            self.current_generation = generation_name
            return generation_name
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise

    def open_snapshot(self) -> RollingTemporalHNSW:
        directory = self.root / "generations" / self.current_generation
        return RollingTemporalHNSW(directory, np.load(directory / "vectors.npy", mmap_mode="r"))
