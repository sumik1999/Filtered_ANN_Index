import tempfile
import unittest
from pathlib import Path

import numpy as np

from temporal_ann.twitter_ann.streaming.live_query import LiveSnapshotManager
from temporal_ann.twitter_ann.streaming.ordering import content_hash
from temporal_ann.twitter_ann.streaming.temporal_bitmap_writer import (
    TemporalBitmapKafkaWriter,
)
from temporal_ann.twitter_ann.temporal_bitmap_hnsw import (
    TemporalBitmapBuildConfig,
    TemporalBitmapHNSW,
    TemporalBitmapSearchConfig,
    build_temporal_bitmap_index,
)


class TwitterTemporalBitmapWriterTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        rng = np.random.default_rng(101)
        values = rng.normal(size=(10, 384)).astype(np.float32)
        self.vectors = values / np.linalg.norm(values, axis=1, keepdims=True)
        self.config = TemporalBitmapBuildConfig(
            vector_count=20,
            local_semantic_degree=2,
            historical_semantic_degree=3,
            future_reciprocal_degree=3,
            local_window_hours=1,
            range_degree_per_side=1,
            range_scales_hours="1,4",
            range_candidate_count=4,
            bridge_degree_per_side=1,
            bridge_scales_epochs="1,4",
            epoch_hours=1,
            gateway_period=2,
            portal_semantic_degree=2,
            portal_epoch_radius=1,
            persist_incremental_state=True,
            construction_candidates=4,
            ef_construction=20,
            candidate_index_m=2,
            threads=1,
        )
        self.events = [self.event(index) for index in range(10)]

    def tearDown(self):
        self.temporary.cleanup()

    def event(self, index):
        event = {
            "schema_version": 1,
            "index_name": "bitmap-stream",
            "index_sequence": index,
            "source_event_id": f"source-{index}",
            "tweet_id": f"tweet-{index:04d}",
            "event_time_ms": 1_000 + index * 30 * 60 * 1000,
            "ingest_time_ms": 100_000 + index,
            "embedding_model": "test",
            "embedding_dimension": 384,
            "embedding": self.vectors[index].tolist(),
        }
        event["content_sha256"] = content_hash(event)
        return event

    def test_kafka_batches_append_without_reinserting_old_nodes(self):
        writer = TemporalBitmapKafkaWriter(
            self.root / "stream", self.config, "bitmap-stream"
        )
        writer.append_batch(self.events[:4], checkpoint={"offset": 3})
        first_generation = writer.current_generation
        writer = TemporalBitmapKafkaWriter(
            self.root / "stream", self.config, "bitmap-stream"
        )
        result = writer.append_batch(self.events[4:], checkpoint={"offset": 9})
        self.assertEqual(result["appended"], 6)
        self.assertEqual(writer.last_sequence, 9)
        self.assertNotEqual(first_generation, writer.current_generation)

        generation = self.root / "stream" / "generations" / writer.current_generation
        stream_vectors = np.load(generation / "stream_vectors.npy", mmap_mode="r")
        incremental = TemporalBitmapHNSW(generation, stream_vectors)

        offline_path = self.root / "offline"
        build_temporal_bitmap_index(
            self.vectors,
            np.asarray([event["event_time_ms"] for event in self.events]),
            offline_path,
            self.config,
            progress_every=0,
        )
        offline = TemporalBitmapHNSW(offline_path, self.vectors)
        manager = LiveSnapshotManager(self.root / "stream", poll_interval=0.01)
        with manager.acquire(min_sequence=9, timeout=0.1) as snapshot:
            result = snapshot.search(
                self.vectors[3], [[0, 20_000_000]],
                TemporalBitmapSearchConfig(
                    k=3, ef_search=4, exact_threshold=100,
                    gateway_candidates=4, gateway_seeds=2,
                ),
            )
            self.assertEqual(result["eligible_count"], 10)
        manager.close()

        for name in (
            "packed_group_block_offsets", "packed_block_epochs",
            "packed_block_neighbor_offsets", "packed_neighbors",
            "edge_epoch_bitmaps", "portal_semantic_neighbors",
            "portal_temporal_neighbors", "epoch_portal_offsets",
            "epoch_portal_rows",
        ):
            np.testing.assert_array_equal(
                getattr(incremental, name), getattr(offline, name), err_msg=name
            )

    def test_replay_is_idempotent_and_gap_is_rejected(self):
        writer = TemporalBitmapKafkaWriter(
            self.root / "stream", self.config, "bitmap-stream"
        )
        writer.append_batch(self.events[:3])
        duplicate = writer.append_batch(self.events[:3])
        self.assertEqual(duplicate["appended"], 0)
        self.assertEqual(writer.vector_count, 3)
        with self.assertRaisesRegex(ValueError, "sequence gap"):
            writer.append_batch([self.events[4]])


if __name__ == "__main__":
    unittest.main()
