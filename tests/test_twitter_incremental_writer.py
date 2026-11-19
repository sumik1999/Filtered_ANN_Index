import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from temporal_ann.twitter_ann.rolling_hnsw import (
    RollingBuildConfig,
    RollingSearchConfig,
    RollingTemporalHNSW,
    build_rolling_index,
)
from temporal_ann.twitter_ann.streaming.incremental_writer import (
    RollingTemporalHNSWWriter,
)
from temporal_ann.twitter_ann.streaming.ordering import canonical_json, content_hash


class TwitterIncrementalWriterTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.config = RollingBuildConfig(
            vector_count=32,
            semantic_degree=4,
            gateway_degree=2,
            gateway_period=3,
            construction_candidates=8,
            ef_construction=30,
            candidate_index_m=4,
            threads=1,
            range_degree_per_side=1,
            range_candidate_count=6,
            range_scales_hours="1,2",
        )
        generator = np.random.default_rng(51)
        vectors = generator.normal(size=(12, 384)).astype(np.float32)
        self.vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        self.events = [self.event(index) for index in range(len(self.vectors))]

    def tearDown(self):
        self.temporary.cleanup()

    def event(self, index):
        event = {
            "schema_version": 1,
            "index_name": "test-index",
            "index_sequence": index,
            "source_event_id": f"source-{index:04d}",
            "tweet_id": f"tweet-{index:04d}",
            "event_time_ms": 1_000 + index * 1_000,
            "ingest_time_ms": 100_000 + index,
            "embedding_model": "test-model",
            "embedding_dimension": 384,
            "embedding": self.vectors[index].tolist(),
        }
        event["content_sha256"] = content_hash(event)
        return event

    def create(self):
        return RollingTemporalHNSWWriter.create(
            self.root / "streaming", self.config,
            dimensions=384, index_name="test-index",
        )

    def test_incremental_batches_match_uninterrupted_offline_topology(self):
        writer = self.create()
        first = writer.append_batch(self.events[:5], checkpoint={"offset": 4})
        self.assertEqual(first["appended"], 5)
        first_generation = writer.current_generation

        writer = RollingTemporalHNSWWriter(self.root / "streaming")
        second = writer.append_batch(self.events[5:], checkpoint={"offset": 11})
        self.assertEqual(second["appended"], 7)
        self.assertEqual(writer.last_sequence, 11)
        self.assertEqual(writer.checkpoint, {"offset": 11})
        self.assertNotEqual(first_generation, writer.current_generation)

        offline_path = self.root / "offline"
        build_rolling_index(
            self.vectors,
            np.asarray([event["event_time_ms"] for event in self.events]),
            offline_path,
            self.config,
            progress_every=0,
        )
        offline = RollingTemporalHNSW(offline_path, self.vectors)
        snapshot = writer.open_snapshot()
        for attribute in (
            "timestamps", "global_ids", "semantic_neighbors",
            "temporal_neighbors", "range_left_neighbors",
            "range_right_neighbors", "gateway_ids", "gateway_rows",
            "gateway_neighbors",
        ):
            np.testing.assert_array_equal(
                getattr(snapshot, attribute), getattr(offline, attribute),
                err_msg=attribute,
            )

    def test_replayed_batch_is_idempotent_and_conflict_is_rejected(self):
        writer = self.create()
        writer.append_batch(self.events[:4])
        result = writer.append_batch(self.events[:4])
        self.assertEqual(result["appended"], 0)
        self.assertEqual(writer.vector_count, 4)

        conflict = dict(self.events[2])
        conflict["embedding_model"] = "different"
        conflict["content_sha256"] = content_hash(conflict)
        with self.assertRaisesRegex(ValueError, "conflicting content"):
            writer.append_batch([conflict])

    def test_replayed_kafka_batch_can_advance_checkpoint_without_duplicate_rows(self):
        writer = self.create()
        writer.append_batch(self.events[:3], checkpoint={"offset": 2})
        generation = writer.current_generation
        result = writer.append_batch(self.events[:3], checkpoint={"offset": 3})
        self.assertEqual(result["appended"], 0)
        self.assertEqual(writer.vector_count, 3)
        self.assertNotEqual(writer.current_generation, generation)
        self.assertEqual(writer.checkpoint, {"offset": 3})

    def test_sequence_gap_and_decreasing_order_key_are_rejected(self):
        writer = self.create()
        with self.assertRaisesRegex(ValueError, "sequence gap"):
            writer.append_batch([self.events[1]])
        writer.append_batch(self.events[:2])
        decreasing = dict(self.events[2])
        decreasing["event_time_ms"] = 500
        decreasing["content_sha256"] = content_hash(decreasing)
        with self.assertRaisesRegex(ValueError, "decreased"):
            writer.append_batch([decreasing])

    def test_orphaned_wal_is_replayed_on_restart(self):
        writer = self.create()
        writer.append_batch(self.events[:3])
        wal = self.root / "streaming" / "wal" / "00000000000000000003-00000000000000000005.jsonl"
        with wal.open("wb") as handle:
            for event in self.events[3:6]:
                handle.write(canonical_json(event) + b"\n")
        recovered = RollingTemporalHNSWWriter(self.root / "streaming")
        self.assertEqual(recovered.last_sequence, 5)
        self.assertEqual(recovered.vector_count, 6)
        self.assertFalse(wal.exists())
        self.assertEqual(recovered.checkpoint["recovered_from_wal"], wal.name)

    def test_corrupted_generation_artifact_fails_checksum_validation(self):
        writer = self.create()
        writer.append_batch(self.events[:3])
        vectors = (
            self.root / "streaming" / "generations" /
            writer.current_generation / "vectors.npy"
        )
        with vectors.open("ab") as handle:
            handle.write(b"corrupt")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            RollingTemporalHNSWWriter(self.root / "streaming")

    def test_published_snapshot_is_queryable_and_old_generation_is_immutable(self):
        writer = self.create()
        writer.append_batch(self.events[:6])
        old_generation = writer.current_generation
        old_manifest = (
            self.root / "streaming" / "generations" / old_generation / "manifest.json"
        ).read_bytes()
        snapshot = writer.open_snapshot()
        result = snapshot.search(
            self.vectors[2], [[1_000, 7_000]],
            RollingSearchConfig(
                k=3, ef_search=4, exact_threshold=100,
                gateway_candidates=4, gateway_seeds=2,
            ),
        )
        self.assertEqual(result["method"], "exact")
        self.assertEqual(int(result["ids"][0]), 2)

        writer.append_batch(self.events[6:8])
        self.assertEqual(
            old_manifest,
            (self.root / "streaming" / "generations" / old_generation / "manifest.json").read_bytes(),
        )
        current = (self.root / "streaming" / "CURRENT").read_text().strip()
        self.assertEqual(current, writer.current_generation)


if __name__ == "__main__":
    unittest.main()
