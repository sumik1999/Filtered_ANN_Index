import unittest

import numpy as np

from temporal_ann.twitter_ann.streaming.ordering import (
    EventTimeOrderer,
    content_hash,
    deterministic_embedding,
    embed_raw_event,
    validate_embedded_event,
    validate_raw_event,
)


def raw(tweet_id: str, timestamp: int, text: str | None = None):
    return {
        "schema_version": 1,
        "source_event_id": f"source-{tweet_id}",
        "tweet_id": tweet_id,
        "event_time_ms": timestamp,
        "ingest_time_ms": timestamp + 1_000,
        "text": text or f"tweet {tweet_id}",
        "language": "en",
    }


def embedded(tweet_id: str, timestamp: int):
    return embed_raw_event(raw(tweet_id, timestamp), "test-index")


class TwitterKafkaOrderingTests(unittest.TestCase):
    def test_development_embedding_is_deterministic_finite_and_normalized(self):
        first = np.asarray(deterministic_embedding("same tweet"), dtype=np.float32)
        second = np.asarray(deterministic_embedding("same tweet"), dtype=np.float32)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.shape, (384,))
        self.assertTrue(np.all(np.isfinite(first)))
        self.assertAlmostEqual(float(np.linalg.norm(first)), 1.0, places=6)

    def test_validation_and_embedding_contract(self):
        event = raw("7", 10)
        self.assertIs(validate_raw_event(event), event)
        result = embed_raw_event(event, "index")
        self.assertIs(validate_embedded_event(result), result)
        broken = dict(result)
        broken["embedding"] = list(result["embedding"])
        broken["embedding"][0] = float("nan")
        with self.assertRaises(ValueError):
            validate_embedded_event(broken)

    def test_orders_across_partitions_and_allocates_contiguous_sequences(self):
        orderer = EventTimeOrderer([0, 1], allowed_lateness_ms=10)
        result = orderer.process([
            (0, embedded("b", 100)),
            (1, embedded("d", 105)),
            (0, embedded("a", 130)),
            (1, embedded("c", 140)),
        ])
        self.assertEqual(
            [(value["event_time_ms"], value["tweet_id"]) for value in result.ordered],
            [(100, "b"), (105, "d")],
        )
        flushed = orderer.process([], force_flush=True)
        combined = result.ordered + flushed.ordered
        self.assertEqual([value["index_sequence"] for value in combined], list(range(4)))
        self.assertEqual(
            [(value["event_time_ms"], value["tweet_id"]) for value in combined],
            sorted((value["event_time_ms"], value["tweet_id"]) for value in combined),
        )

    def test_ties_are_ordered_by_tweet_id(self):
        orderer = EventTimeOrderer([0], allowed_lateness_ms=1_000)
        result = orderer.process([
            (0, embedded("z", 100)),
            (0, embedded("a", 100)),
            (0, embedded("m", 100)),
        ], force_flush=True)
        self.assertEqual([value["tweet_id"] for value in result.ordered], ["a", "m", "z"])

    def test_event_behind_committed_watermark_is_late(self):
        orderer = EventTimeOrderer([0, 1], allowed_lateness_ms=10)
        orderer.process([(0, embedded("a", 200)), (1, embedded("b", 210))])
        result = orderer.process([(0, embedded("late", 150))])
        self.assertEqual(len(result.late), 1)
        self.assertEqual(result.late[0]["tweet_id"], "late")
        self.assertEqual(result.ordered, [])
        self.assertNotIn("index_sequence", result.late[0])

    def test_duplicates_are_idempotent_and_conflicts_are_rejected(self):
        event = embedded("a", 100)
        orderer = EventTimeOrderer([0], allowed_lateness_ms=100)
        orderer.process([(0, event)])
        duplicate = orderer.process([(0, dict(event))])
        self.assertEqual(len(duplicate.duplicates), 1)
        conflict = dict(event)
        conflict["event_time_ms"] = 101
        conflict["content_sha256"] = content_hash(conflict)
        with self.assertRaises(ValueError):
            orderer.process([(0, conflict)])

    def test_idle_partition_allows_watermark_progress(self):
        orderer = EventTimeOrderer([0, 1], allowed_lateness_ms=0)
        self.assertEqual(orderer.process([(0, embedded("a", 100))]).ordered, [])
        result = orderer.mark_idle([1])
        self.assertEqual([event["tweet_id"] for event in result.ordered], ["a"])

    def test_changelog_reconstructs_buffer_deduplication_and_metadata(self):
        orderer = EventTimeOrderer([0, 1], allowed_lateness_ms=10)
        result = orderer.process([
            (0, embedded("a", 100)),
            (1, embedded("b", 105)),
            (0, embedded("c", 130)),
        ])
        restored = EventTimeOrderer([0, 1], allowed_lateness_ms=10)
        for key, value in orderer.changelog_records(result):
            restored.restore_record(key, value)
        restored.rebuild_heap()
        self.assertEqual(restored.metadata(), orderer.metadata())
        self.assertEqual(restored.buffer, orderer.buffer)
        self.assertEqual(restored.source_hashes, orderer.source_hashes)
        retry = restored.process([(0, embedded("a", 100))])
        self.assertEqual(len(retry.duplicates), 1)

    def test_trial_clone_can_be_discarded_without_advancing_live_state(self):
        live = EventTimeOrderer([0], allowed_lateness_ms=0)
        trial = live.clone()
        result = trial.process([(0, embedded("a", 100))])
        self.assertEqual(len(result.ordered), 1)
        self.assertEqual(live.next_sequence, 0)
        self.assertEqual(live.buffer, {})
        self.assertEqual(trial.next_sequence, 1)


if __name__ == "__main__":
    unittest.main()
