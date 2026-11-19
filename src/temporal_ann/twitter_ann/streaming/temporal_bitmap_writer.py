from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ..temporal_bitmap_hnsw import (
    TemporalBitmapBuildConfig,
    build_temporal_bitmap_index,
)
from .ordering import canonical_json, validate_embedded_event


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class TemporalBitmapKafkaWriter:
    """Idempotent Kafka micro-batch writer for the temporal-bitmap topology.

    Existing adjacency is loaded from the previous generation and only new
    vectors execute graph insertion. Packed epoch blocks and portal directories
    are regenerated because they are immutable query-side summaries.
    """

    def __init__(
        self,
        root: Path,
        config: TemporalBitmapBuildConfig,
        index_name: str = "twitter-rt-hnsw-bitmap",
    ):
        if not config.persist_incremental_state:
            raise ValueError("bitmap Kafka writer requires persisted incremental state")
        self.root = Path(root)
        self.config = config
        self.index_name = index_name
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "generations").mkdir(exist_ok=True)
        (self.root / "wal").mkdir(exist_ok=True)
        self.current_generation: str | None = None
        self.last_sequence = -1
        self.last_order_key: tuple[int, str] | None = None
        self.event_metadata: list[dict[str, Any]] = []
        self.checkpoint: dict[str, Any] = {}
        current = self.root / "CURRENT"
        if current.exists():
            self._load_current()
        self._recover_wals()

    def _load_current(self) -> None:
        self.current_generation = (self.root / "CURRENT").read_text().strip()
        directory = self.root / "generations" / self.current_generation
        manifest = json.loads((directory / "manifest.json").read_text())
        if manifest["config"] != vars(self.config):
            raise ValueError("bitmap streaming configuration differs")
        if manifest.get("index_name") != self.index_name:
            raise ValueError("bitmap streaming index name differs")
        self.last_sequence = int(manifest["last_index_sequence"])
        key = manifest.get("last_order_key")
        self.last_order_key = (int(key[0]), str(key[1])) if key else None
        self.checkpoint = manifest.get("checkpoint", {})
        metadata_file = manifest["files"]["stream_event_metadata"]
        self.event_metadata = [
            json.loads(line)
            for line in (directory / metadata_file).read_text().splitlines()
            if line
        ]
        if len(self.event_metadata) != self.last_sequence + 1:
            raise ValueError("bitmap streaming metadata is not contiguous")

    @property
    def vector_count(self) -> int:
        return self.last_sequence + 1

    def _validate(self, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        accepted = []
        expected = self.last_sequence + 1
        prior_key = self.last_order_key
        previous_input = None
        for event in events:
            validate_embedded_event(event)
            if event.get("index_name") != self.index_name:
                raise ValueError("event index_name does not match bitmap writer")
            sequence = event.get("index_sequence")
            if not isinstance(sequence, int) or isinstance(sequence, bool):
                raise ValueError("ordered event requires index_sequence")
            if previous_input is not None and sequence <= previous_input:
                raise ValueError("input sequences must be strictly increasing")
            previous_input = sequence
            if sequence <= self.last_sequence:
                if self.event_metadata[sequence]["content_sha256"] != event["content_sha256"]:
                    raise ValueError("replayed sequence has conflicting content")
                continue
            if sequence != expected:
                raise ValueError(f"index sequence gap: expected {expected}, received {sequence}")
            key = (int(event["event_time_ms"]), str(event["tweet_id"]))
            if prior_key is not None and key < prior_key:
                raise ValueError("ordered event key decreased")
            accepted.append(dict(event))
            expected += 1
            prior_key = key
        return accepted

    def append_batch(
        self,
        events: Iterable[dict[str, Any]],
        checkpoint: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        values = [dict(event) for event in events]
        accepted = self._validate(values)
        if not accepted:
            return {
                "appended": 0,
                "last_index_sequence": self.last_sequence,
                "generation": self.current_generation,
            }
        first, last = accepted[0]["index_sequence"], accepted[-1]["index_sequence"]
        wal = self.root / "wal" / f"{first:020d}-{last:020d}.jsonl"
        temporary_wal = wal.with_suffix(".tmp")
        with temporary_wal.open("wb") as handle:
            for event in accepted:
                handle.write(canonical_json(event) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_wal, wal)

        previous_directory = (
            self.root / "generations" / self.current_generation
            if self.current_generation else None
        )
        if previous_directory is None:
            previous_vectors = np.empty((0, 384), dtype=np.float32)
        else:
            previous_manifest = json.loads(
                (previous_directory / "manifest.json").read_text()
            )
            previous_vectors = np.load(
                previous_directory / previous_manifest["files"]["stream_vectors"]
            )
        appended_vectors = np.asarray(
            [event["embedding"] for event in accepted], dtype=np.float32
        )
        vectors = np.concatenate((previous_vectors, appended_vectors), axis=0)
        timestamps = np.asarray(
            [item["event_time_ms"] for item in self.event_metadata]
            + [event["event_time_ms"] for event in accepted],
            dtype=np.int64,
        )
        generation_number = 0 if self.current_generation is None else int(self.current_generation) + 1
        generation_name = f"{generation_number:020d}"
        staging = Path(tempfile.mkdtemp(
            prefix=f".{generation_name}-", dir=self.root / "generations"
        ))
        shutil.rmtree(staging)
        try:
            build_temporal_bitmap_index(
                vectors, timestamps, staging, self.config,
                rebuild=True, progress_every=0,
                resume_directory=previous_directory,
            )
            metadata = self.event_metadata + [
                {
                    "index_sequence": event["index_sequence"],
                    "tweet_id": event["tweet_id"],
                    "source_event_id": event["source_event_id"],
                    "event_time_ms": event["event_time_ms"],
                    "content_sha256": event["content_sha256"],
                }
                for event in accepted
            ]
            np.save(staging / "stream_vectors.npy", vectors, allow_pickle=False)
            with (staging / "stream_event_metadata.jsonl").open("wb") as handle:
                for item in metadata:
                    handle.write(canonical_json(item) + b"\n")
            manifest_path = staging / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["index_name"] = self.index_name
            manifest["last_index_sequence"] = int(accepted[-1]["index_sequence"])
            key = (int(accepted[-1]["event_time_ms"]), str(accepted[-1]["tweet_id"]))
            manifest["last_order_key"] = list(key)
            manifest["checkpoint"] = dict(checkpoint or {})
            manifest["files"]["stream_vectors"] = "stream_vectors.npy"
            manifest["files"]["stream_event_metadata"] = "stream_event_metadata.jsonl"
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
            for path in staging.iterdir():
                with path.open("rb") as handle:
                    os.fsync(handle.fileno())
            _fsync_directory(staging)
            final = self.root / "generations" / generation_name
            staging.replace(final)
            _fsync_directory(final.parent)
            pointer = self.root / ".CURRENT.tmp"
            with pointer.open("w") as handle:
                handle.write(generation_name + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(pointer, self.root / "CURRENT")
            _fsync_directory(self.root)
            self.current_generation = generation_name
            self.last_sequence = int(accepted[-1]["index_sequence"])
            self.last_order_key = key
            self.event_metadata = metadata
            self.checkpoint = dict(checkpoint or {})
            wal.unlink(missing_ok=True)
            return {
                "appended": len(accepted),
                "last_index_sequence": self.last_sequence,
                "generation": generation_name,
            }
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise

    def _recover_wals(self) -> None:
        for wal in sorted((self.root / "wal").glob("*.jsonl")):
            events = [json.loads(line) for line in wal.read_text().splitlines() if line]
            accepted = self._validate(events)
            if not accepted:
                wal.unlink()
            else:
                self.append_batch(
                    accepted, checkpoint={"recovered_from_wal": wal.name}
                )
