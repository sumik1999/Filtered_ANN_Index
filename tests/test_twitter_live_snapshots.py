import json
import tempfile
import threading
import time
import unittest
import urllib.request
from pathlib import Path

import numpy as np

from temporal_ann.twitter_ann.rolling_hnsw import RollingBuildConfig, RollingSearchConfig
from temporal_ann.twitter_ann.streaming.incremental_writer import RollingTemporalHNSWWriter
from temporal_ann.twitter_ann.streaming.live_query import (
    LiveQueryHTTPServer,
    LiveSnapshotManager,
)
from temporal_ann.twitter_ann.streaming.ordering import content_hash


class TwitterLiveSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "streaming"
        self.config = RollingBuildConfig(
            vector_count=50,
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
        generator = np.random.default_rng(92)
        values = generator.normal(size=(10, 384)).astype(np.float32)
        self.vectors = values / np.linalg.norm(values, axis=1, keepdims=True)
        self.events = [self.event(index) for index in range(10)]
        self.writer = RollingTemporalHNSWWriter.create(
            self.root, self.config, dimensions=384, index_name="live-test"
        )

    def tearDown(self):
        self.temporary.cleanup()

    def event(self, index):
        event = {
            "schema_version": 1,
            "index_name": "live-test",
            "index_sequence": index,
            "source_event_id": f"source-{index}",
            "tweet_id": f"tweet-{index:04d}",
            "event_time_ms": 1_000 + index * 1_000,
            "ingest_time_ms": 100_000 + index,
            "embedding_model": "test",
            "embedding_dimension": 384,
            "embedding": self.vectors[index].tolist(),
        }
        event["content_sha256"] = content_hash(event)
        return event

    def search_config(self):
        return RollingSearchConfig(
            k=3, ef_search=4, exact_threshold=100,
            gateway_candidates=4, gateway_seeds=2,
        )

    def test_active_query_keeps_old_generation_while_new_queries_switch(self):
        self.writer.append_batch(self.events[:3])
        manager = LiveSnapshotManager(self.root, poll_interval=0.01)
        old = manager.acquire()
        old_generation = old.generation
        self.assertEqual(old.last_sequence, 2)

        self.writer.append_batch(self.events[3:6])
        self.assertTrue(manager.refresh())
        with manager.acquire() as new:
            self.assertNotEqual(new.generation, old_generation)
            self.assertEqual(new.last_sequence, 5)
            new_result = new.search(
                self.vectors[4], [[1_000, 7_000]], self.search_config()
            )
            self.assertEqual(new_result["eligible_count"], 6)

        old_result = old.search(
            self.vectors[1], [[1_000, 7_000]], self.search_config()
        )
        self.assertEqual(old_result["eligible_count"], 3)
        self.assertIn(old_generation, manager.status()["retired_generations"])

        # Reader lease prevents collection while the old query is active.
        manager.collect_generations(retain=1)
        self.assertTrue((self.root / "generations" / old_generation).exists())
        old.release()
        deleted = manager.collect_generations(retain=1)
        self.assertIn(old_generation, deleted)
        manager.close()

    def test_minimum_sequence_waits_for_background_publication(self):
        self.writer.append_batch(self.events[:2])
        manager = LiveSnapshotManager(self.root, poll_interval=0.01)
        manager.start()

        def publish():
            time.sleep(0.05)
            self.writer.append_batch(self.events[2:5])

        thread = threading.Thread(target=publish)
        thread.start()
        with manager.acquire(min_sequence=4, timeout=2.0) as snapshot:
            self.assertGreaterEqual(snapshot.last_sequence, 4)
            self.assertEqual(snapshot.manifest["vector_count"], 5)
        thread.join()
        manager.close()

    def test_minimum_sequence_timeout_reports_current_freshness(self):
        self.writer.append_batch(self.events[:2])
        manager = LiveSnapshotManager(self.root, poll_interval=0.01)
        with self.assertRaisesRegex(TimeoutError, "current=1"):
            manager.acquire(min_sequence=9, timeout=0.03)
        manager.close()

    def test_queries_never_observe_partial_generation_during_updates(self):
        self.writer.append_batch(self.events[:2])
        manager = LiveSnapshotManager(self.root, poll_interval=0.005)
        manager.start()
        failures = []
        stopping = threading.Event()

        def query_loop():
            try:
                while not stopping.is_set():
                    with manager.acquire(timeout=0.2) as snapshot:
                        result = snapshot.search(
                            self.vectors[0], [[0, 20_000]], self.search_config()
                        )
                        expected = snapshot.last_sequence + 1
                        if result["eligible_count"] != expected:
                            failures.append((snapshot.generation, expected, result["eligible_count"]))
            except Exception as error:  # surfaced in the main test thread
                failures.append(error)

        reader = threading.Thread(target=query_loop)
        reader.start()
        for index in range(2, 8):
            self.writer.append_batch([self.events[index]])
            time.sleep(0.01)
        with manager.acquire(min_sequence=7, timeout=2.0) as snapshot:
            self.assertEqual(snapshot.last_sequence, 7)
        stopping.set()
        reader.join(timeout=2.0)
        self.assertFalse(reader.is_alive())
        self.assertEqual(failures, [])
        manager.close()

    def test_http_service_returns_generation_and_sequence(self):
        self.writer.append_batch(self.events[:6])
        manager = LiveSnapshotManager(self.root, poll_interval=0.01)
        manager.start()
        server = LiveQueryHTTPServer(("127.0.0.1", 0), manager, quiet=True)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        port = server.server_address[1]
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health") as response:
                health = json.loads(response.read())
            self.assertTrue(health["healthy"])
            self.assertEqual(health["last_index_sequence"], 5)

            request = urllib.request.Request(
                f"http://127.0.0.1:{port}/search",
                data=json.dumps({
                    "query": self.vectors[2].tolist(),
                    "intervals": [[1_000, 7_000]],
                    "k": 3,
                    "ef_search": 4,
                    "exact_threshold": 100,
                    "min_sequence": 5,
                    "wait_timeout_ms": 100,
                }).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request) as response:
                result = json.loads(response.read())
            self.assertEqual(result["last_index_sequence"], 5)
            self.assertEqual(result["eligible_count"], 6)
            self.assertEqual(result["ids"][0], 2)
            self.assertIn("generation", result)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()
            manager.close()


if __name__ == "__main__":
    unittest.main()
