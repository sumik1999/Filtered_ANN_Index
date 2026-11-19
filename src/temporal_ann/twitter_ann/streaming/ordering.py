from __future__ import annotations

import copy
import hashlib
import heapq
import json
import math
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np

INT64_MIN = -(2**63)
INT64_MAX = 2**63 - 1


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def content_hash(value: dict[str, Any], excluded: set[str] | None = None) -> str:
    excluded = excluded or {"content_sha256", "index_sequence"}
    payload = {key: item for key, item in value.items() if key not in excluded}
    return hashlib.sha256(canonical_json(payload)).hexdigest()


def validate_raw_event(value: dict[str, Any]) -> dict[str, Any]:
    required = (
        "schema_version", "source_event_id", "tweet_id", "event_time_ms",
        "ingest_time_ms", "text", "language",
    )
    if not isinstance(value, dict) or any(key not in value for key in required):
        raise ValueError("raw event is missing required fields")
    if value["schema_version"] != 1:
        raise ValueError("unsupported raw-event schema version")
    for key, limit in (("source_event_id", 256), ("tweet_id", 128)):
        if not isinstance(value[key], str) or not value[key] or len(value[key]) > limit:
            raise ValueError(f"invalid {key}")
    for key in ("event_time_ms", "ingest_time_ms"):
        if isinstance(value[key], bool) or not isinstance(value[key], int):
            raise ValueError(f"{key} must be an integer")
        if not INT64_MIN <= value[key] <= INT64_MAX:
            raise ValueError(f"{key} is outside signed int64")
    if value["ingest_time_ms"] < 0:
        raise ValueError("ingest_time_ms must be nonnegative")
    if not isinstance(value["text"], str) or not value["text"].strip():
        raise ValueError("text must be nonempty")
    if len(value["text"]) > 65_536:
        raise ValueError("text exceeds maximum length")
    if not isinstance(value["language"], str) or not 2 <= len(value["language"]) <= 16:
        raise ValueError("invalid language")
    return value


def deterministic_embedding(text: str, dimensions: int = 384) -> list[float]:
    """Development-only deterministic normalized embedding."""
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "big", signed=False)
    generator = np.random.default_rng(seed)
    vector = generator.standard_normal(dimensions).astype(np.float32)
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm == 0:
        raise ValueError("failed to create finite embedding")
    return (vector / norm).tolist()


def embed_raw_event(raw: dict[str, Any], index_name: str) -> dict[str, Any]:
    validate_raw_event(raw)
    event = {
        "schema_version": 1,
        "index_name": index_name,
        "source_event_id": raw["source_event_id"],
        "tweet_id": raw["tweet_id"],
        "event_time_ms": raw["event_time_ms"],
        "ingest_time_ms": raw["ingest_time_ms"],
        "embedding_model": "deterministic-sha256-v1",
        "embedding_dimension": 384,
        "embedding": deterministic_embedding(raw["text"]),
    }
    event["content_sha256"] = content_hash(event)
    return event


def validate_embedded_event(value: dict[str, Any]) -> dict[str, Any]:
    required = (
        "schema_version", "index_name", "source_event_id", "tweet_id",
        "event_time_ms", "ingest_time_ms", "embedding_model",
        "embedding_dimension", "embedding", "content_sha256",
    )
    if not isinstance(value, dict) or any(key not in value for key in required):
        raise ValueError("embedded event is missing required fields")
    if value["schema_version"] != 1 or value["embedding_dimension"] != 384:
        raise ValueError("unsupported schema or embedding dimension")
    for key in ("index_name", "source_event_id", "tweet_id", "embedding_model"):
        if not isinstance(value[key], str) or not value[key]:
            raise ValueError(f"invalid {key}")
    for key in ("event_time_ms", "ingest_time_ms"):
        if isinstance(value[key], bool) or not isinstance(value[key], int):
            raise ValueError(f"{key} must be an integer")
        if not INT64_MIN <= value[key] <= INT64_MAX:
            raise ValueError(f"{key} is outside signed int64")
    vector = np.asarray(value["embedding"])
    if vector.shape != (384,) or not np.issubdtype(vector.dtype, np.number):
        raise ValueError("embedding must contain 384 numeric values")
    if not np.all(np.isfinite(vector)):
        raise ValueError("embedding must contain only finite values")
    if (not isinstance(value["content_sha256"], str)
            or len(value["content_sha256"]) != 64
            or any(character not in "0123456789abcdef" for character in value["content_sha256"])):
        raise ValueError("invalid content hash")
    if content_hash(value) != value["content_sha256"]:
        raise ValueError("embedded-event content hash mismatch")
    return value


def ordering_key(event: dict[str, Any]) -> tuple[int, str]:
    return int(event["event_time_ms"]), str(event["tweet_id"])


def buffer_key(event: dict[str, Any]) -> str:
    timestamp, tweet_id = ordering_key(event)
    return f"buffer:{timestamp + 2**63:020d}:{tweet_id}"


@dataclass
class OrdererBatchResult:
    ordered: list[dict[str, Any]] = field(default_factory=list)
    late: list[dict[str, Any]] = field(default_factory=list)
    duplicates: list[dict[str, Any]] = field(default_factory=list)
    added_buffer: dict[str, dict[str, Any]] = field(default_factory=dict)
    removed_buffer: set[str] = field(default_factory=set)
    added_source_hashes: dict[str, str] = field(default_factory=dict)
    added_tweet_hashes: dict[str, str] = field(default_factory=dict)


class EventTimeOrderer:
    """Deterministic bounded-lateness orderer with Kafka-changelogable state."""

    def __init__(self, partitions: Iterable[int], allowed_lateness_ms: int = 600_000):
        if allowed_lateness_ms < 0:
            raise ValueError("allowed lateness must be nonnegative")
        values = sorted(set(map(int, partitions)))
        if not values or values[0] < 0:
            raise ValueError("at least one nonnegative partition is required")
        self.allowed_lateness_ms = int(allowed_lateness_ms)
        self.partition_max: dict[int, int | None] = {value: None for value in values}
        self.idle_partitions: set[int] = set()
        self.buffer: dict[str, dict[str, Any]] = {}
        self.heap: list[tuple[int, str, str]] = []
        self.source_hashes: dict[str, str] = {}
        self.tweet_hashes: dict[str, str] = {}
        self.next_sequence = 0
        self.committed_watermark: int | None = None
        self.last_emitted_key: tuple[int, str] | None = None

    def clone(self) -> "EventTimeOrderer":
        return copy.deepcopy(self)

    def _candidate_watermark(self) -> int | None:
        active = [
            maximum for partition, maximum in self.partition_max.items()
            if partition not in self.idle_partitions and maximum is not None
        ]
        waiting = [
            partition for partition, maximum in self.partition_max.items()
            if partition not in self.idle_partitions and maximum is None
        ]
        if waiting or not active:
            return None
        return min(active) - self.allowed_lateness_ms

    def _advance_watermark(self) -> None:
        candidate = self._candidate_watermark()
        if candidate is not None:
            self.committed_watermark = (
                candidate if self.committed_watermark is None
                else max(self.committed_watermark, candidate)
            )

    def _drain(self, result: OrdererBatchResult, force: bool = False) -> None:
        while self.heap:
            timestamp, tweet_id, key = self.heap[0]
            if not force and (
                self.committed_watermark is None
                or timestamp > self.committed_watermark
            ):
                break
            heapq.heappop(self.heap)
            event = self.buffer.pop(key, None)
            if event is None:
                continue
            current_key = (timestamp, tweet_id)
            if self.last_emitted_key is not None and current_key < self.last_emitted_key:
                raise RuntimeError("orderer attempted to emit a decreasing event key")
            ordered = dict(event)
            ordered["index_sequence"] = self.next_sequence
            ordered["content_sha256"] = content_hash(ordered)
            self.next_sequence += 1
            self.last_emitted_key = current_key
            result.ordered.append(ordered)
            result.removed_buffer.add(key)

    def process(
        self,
        records: Iterable[tuple[int, dict[str, Any]]],
        force_flush: bool = False,
    ) -> OrdererBatchResult:
        result = OrdererBatchResult()
        for partition, event in records:
            partition = int(partition)
            if partition not in self.partition_max:
                raise ValueError(f"unknown input partition {partition}")
            validate_embedded_event(event)
            event_hash = event["content_sha256"]
            source_id = event["source_event_id"]
            tweet_id = event["tweet_id"]
            existing_source = self.source_hashes.get(source_id)
            existing_tweet = self.tweet_hashes.get(tweet_id)
            if existing_source is not None or existing_tweet is not None:
                if ((existing_source is not None and existing_source != event_hash)
                        or (existing_tweet is not None and existing_tweet != event_hash)):
                    raise ValueError("conflicting duplicate source_event_id or tweet_id")
                result.duplicates.append(event)
                continue

            self.source_hashes[source_id] = event_hash
            self.tweet_hashes[tweet_id] = event_hash
            result.added_source_hashes[source_id] = event_hash
            result.added_tweet_hashes[tweet_id] = event_hash
            timestamp = int(event["event_time_ms"])
            maximum = self.partition_max[partition]
            self.partition_max[partition] = timestamp if maximum is None else max(maximum, timestamp)
            self.idle_partitions.discard(partition)

            if (self.committed_watermark is not None
                    and timestamp <= self.committed_watermark):
                late = dict(event)
                late["late_reason"] = "event_time_at_or_before_committed_watermark"
                late["committed_watermark_ms"] = self.committed_watermark
                result.late.append(late)
                self._advance_watermark()
                continue

            key = buffer_key(event)
            if key in self.buffer:
                raise ValueError("two distinct events have the same temporal ordering key")
            stored = dict(event)
            self.buffer[key] = stored
            heapq.heappush(self.heap, (*ordering_key(stored), key))
            result.added_buffer[key] = stored
            self._advance_watermark()
            self._drain(result)

        if force_flush:
            self._drain(result, force=True)
        return result

    def mark_idle(self, partitions: Iterable[int]) -> OrdererBatchResult:
        for partition in partitions:
            partition = int(partition)
            if partition not in self.partition_max:
                raise ValueError(f"unknown input partition {partition}")
            self.idle_partitions.add(partition)
        self._advance_watermark()
        result = OrdererBatchResult()
        self._drain(result)
        return result

    def metadata(self) -> dict[str, Any]:
        return {
            "allowed_lateness_ms": self.allowed_lateness_ms,
            "partition_max": {str(key): value for key, value in self.partition_max.items()},
            "idle_partitions": sorted(self.idle_partitions),
            "next_sequence": self.next_sequence,
            "committed_watermark": self.committed_watermark,
            "last_emitted_key": list(self.last_emitted_key) if self.last_emitted_key else None,
        }

    def changelog_records(self, result: OrdererBatchResult) -> list[tuple[str, bytes | None]]:
        records: list[tuple[str, bytes | None]] = []
        for source_id, value in result.added_source_hashes.items():
            records.append((f"source:{source_id}", value.encode()))
        for tweet_id, value in result.added_tweet_hashes.items():
            records.append((f"tweet:{tweet_id}", value.encode()))
        for key, event in result.added_buffer.items():
            if key not in result.removed_buffer:
                records.append((key, canonical_json(event)))
        records.extend((key, None) for key in result.removed_buffer)
        records.append(("metadata", canonical_json(self.metadata())))
        return records

    def restore_record(self, key: str, value: bytes | None) -> None:
        if key == "metadata":
            if value is None:
                return
            metadata = json.loads(value)
            if metadata["allowed_lateness_ms"] != self.allowed_lateness_ms:
                raise ValueError("changelog lateness configuration differs")
            self.partition_max = {
                int(partition): maximum
                for partition, maximum in metadata["partition_max"].items()
            }
            self.idle_partitions = set(metadata["idle_partitions"])
            self.next_sequence = int(metadata["next_sequence"])
            self.committed_watermark = metadata["committed_watermark"]
            last = metadata["last_emitted_key"]
            self.last_emitted_key = (int(last[0]), str(last[1])) if last else None
        elif key.startswith("source:"):
            source_id = key.removeprefix("source:")
            if value is None:
                self.source_hashes.pop(source_id, None)
            else:
                self.source_hashes[source_id] = value.decode()
        elif key.startswith("tweet:"):
            tweet_id = key.removeprefix("tweet:")
            if value is None:
                self.tweet_hashes.pop(tweet_id, None)
            else:
                self.tweet_hashes[tweet_id] = value.decode()
        elif key.startswith("buffer:"):
            if value is None:
                self.buffer.pop(key, None)
            else:
                self.buffer[key] = json.loads(value)

    def rebuild_heap(self) -> None:
        self.heap = [(*ordering_key(event), key) for key, event in self.buffer.items()]
        heapq.heapify(self.heap)
