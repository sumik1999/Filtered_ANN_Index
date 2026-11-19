from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from .ordering import (
    EventTimeOrderer,
    OrdererBatchResult,
    canonical_json,
    embed_raw_event,
    validate_embedded_event,
    validate_raw_event,
)

RAW_TOPIC = "twitter.raw.v1"
EMBEDDED_TOPIC = "twitter.embedded.v1"
ORDERED_TOPIC = "twitter.index.ordered.v1"
LATE_TOPIC = "twitter.index.late.v1"
DLQ_TOPIC = "twitter.index.dlq.v1"
STATE_TOPIC = "twitter.orderer.state.v1"
CHECKPOINT_TOPIC = "twitter.index.checkpoints.v1"


def _kafka():
    try:
        from confluent_kafka import Consumer, Producer, TopicPartition
    except ImportError as error:
        raise SystemExit(
            "Kafka commands require: pip install -e '.[kafka]'"
        ) from error
    return Consumer, Producer, TopicPartition


def _producer_config(bootstrap: str, transactional_id: str | None = None) -> dict[str, Any]:
    config: dict[str, Any] = {
        "bootstrap.servers": bootstrap,
        "enable.idempotence": True,
        "acks": "all",
        "compression.type": "zstd",
    }
    if transactional_id:
        config["transactional.id"] = transactional_id
    return config


def _consumer_config(bootstrap: str, group: str) -> dict[str, Any]:
    return {
        "bootstrap.servers": bootstrap,
        "group.id": group,
        "enable.auto.commit": False,
        "auto.offset.reset": "earliest",
        "isolation.level": "read_committed",
    }


def _produce(producer, topic: str, key: str, value: bytes | None) -> None:
    while True:
        try:
            producer.produce(topic, key=key.encode(), value=value)
            return
        except BufferError:
            producer.poll(0.1)


def produce_jsonl(args: argparse.Namespace) -> None:
    _, Producer, _ = _kafka()
    producer = Producer(_producer_config(args.bootstrap))
    source = sys.stdin if str(args.input) == "-" else args.input.open(encoding="utf-8")
    sent = 0
    try:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
                validate_raw_event(event)
            except (ValueError, TypeError, json.JSONDecodeError) as error:
                raise SystemExit(f"invalid event at line {line_number}: {error}") from error
            _produce(producer, RAW_TOPIC, event["tweet_id"], canonical_json(event))
            producer.poll(0)
            sent += 1
    finally:
        if source is not sys.stdin:
            source.close()
    producer.flush()
    print(f"produced {sent} records to {RAW_TOPIC}")


def _dlq_value(stage: str, message, error: Exception) -> bytes:
    return canonical_json({
        "stage": stage,
        "error": str(error),
        "source_topic": message.topic(),
        "source_partition": message.partition(),
        "source_offset": message.offset(),
        "value": message.value().decode("utf-8", errors="replace") if message.value() else None,
        "failed_at_ms": int(time.time() * 1000),
    })


def run_embedder(args: argparse.Namespace) -> None:
    Consumer, Producer, TopicPartition = _kafka()
    consumer = Consumer(_consumer_config(args.bootstrap, args.group))
    producer = Producer(_producer_config(
        args.bootstrap, args.transactional_id or f"rt-hnsw-embedder-{uuid.uuid4()}"
    ))
    producer.init_transactions()
    consumer.subscribe([RAW_TOPIC])
    completed = 0
    try:
        while args.max_records <= 0 or completed < args.max_records:
            message = consumer.poll(args.poll_timeout)
            if message is None:
                continue
            if message.error():
                raise RuntimeError(message.error())
            producer.begin_transaction()
            try:
                raw = json.loads(message.value())
                embedded = embed_raw_event(raw, args.index_name)
                _produce(producer, EMBEDDED_TOPIC, embedded["tweet_id"], canonical_json(embedded))
            except Exception as error:
                _produce(producer, DLQ_TOPIC, message.key().decode() if message.key() else "unknown",
                         _dlq_value("embedding", message, error))
            offsets = [TopicPartition(message.topic(), message.partition(), message.offset() + 1)]
            producer.send_offsets_to_transaction(offsets, consumer.consumer_group_metadata())
            producer.commit_transaction()
            completed += 1
    finally:
        consumer.close()
    print(f"embedded/handled {completed} records")


def _recover_state(bootstrap: str, orderer: EventTimeOrderer) -> None:
    Consumer, _, TopicPartition = _kafka()
    consumer = Consumer({
        **_consumer_config(bootstrap, f"rt-hnsw-state-recovery-{uuid.uuid4()}"),
        "enable.partition.eof": True,
    })
    partition = TopicPartition(STATE_TOPIC, 0)
    low, high = consumer.get_watermark_offsets(partition, timeout=10)
    if high <= low:
        consumer.close()
        return
    consumer.assign([TopicPartition(STATE_TOPIC, 0, low)])
    seen_through = low
    while seen_through < high:
        message = consumer.poll(5.0)
        if message is None:
            continue
        if message.error():
            # Partition EOF is expected once the captured high watermark is read.
            if message.offset() >= high:
                break
            continue
        key = message.key().decode()
        orderer.restore_record(key, message.value())
        seen_through = message.offset() + 1
    consumer.close()
    orderer.rebuild_heap()


def _merge_result(target: OrdererBatchResult, source: OrdererBatchResult) -> None:
    target.ordered.extend(source.ordered)
    target.late.extend(source.late)
    target.duplicates.extend(source.duplicates)
    target.added_buffer.update(source.added_buffer)
    target.removed_buffer.update(source.removed_buffer)
    target.added_source_hashes.update(source.added_source_hashes)
    target.added_tweet_hashes.update(source.added_tweet_hashes)


def run_orderer(args: argparse.Namespace) -> None:
    Consumer, Producer, TopicPartition = _kafka()
    probe = Consumer(_consumer_config(args.bootstrap, f"rt-hnsw-probe-{uuid.uuid4()}"))
    metadata = probe.list_topics(EMBEDDED_TOPIC, timeout=10)
    topic = metadata.topics.get(EMBEDDED_TOPIC)
    if topic is None or topic.error is not None:
        raise RuntimeError(f"cannot inspect {EMBEDDED_TOPIC}: {getattr(topic, 'error', None)}")
    partitions = sorted(topic.partitions)
    probe.close()

    orderer = EventTimeOrderer(partitions, args.allowed_lateness_ms)
    _recover_state(args.bootstrap, orderer)
    if set(orderer.partition_max) != set(partitions):
        raise RuntimeError("embedded topic partition count differs from orderer changelog")

    consumer = Consumer(_consumer_config(args.bootstrap, args.group))
    producer = Producer(_producer_config(
        args.bootstrap, args.transactional_id or f"rt-hnsw-orderer-{args.index_name}"
    ))
    producer.init_transactions()
    consumer.subscribe([EMBEDDED_TOPIC])
    handled = 0
    last_seen_wall_ms = {partition: int(time.time() * 1000) for partition in partitions}
    try:
        while args.max_records <= 0 or handled < args.max_records:
            remaining = args.batch_size if args.max_records <= 0 else min(
                args.batch_size, args.max_records - handled
            )
            messages = consumer.consume(remaining, timeout=args.poll_timeout)
            messages = [message for message in messages if not message.error()]
            if not messages:
                continue
            trial = orderer.clone()
            aggregate = OrdererBatchResult()
            dlq: list[tuple[Any, Exception]] = []
            now_ms = int(time.time() * 1000)
            for message in messages:
                last_seen_wall_ms[message.partition()] = now_ms
                try:
                    event = json.loads(message.value())
                    validate_embedded_event(event)
                    if event["index_name"] != args.index_name:
                        raise ValueError("event index_name does not match orderer")
                    _merge_result(
                        aggregate,
                        trial.process([(message.partition(), event)]),
                    )
                except Exception as error:
                    dlq.append((message, error))
            stale_partitions = [
                partition for partition, seen_ms in last_seen_wall_ms.items()
                if now_ms - seen_ms >= args.idle_timeout_ms
            ]
            if stale_partitions:
                _merge_result(aggregate, trial.mark_idle(stale_partitions))

            producer.begin_transaction()
            try:
                for event in aggregate.ordered:
                    _produce(
                        producer, ORDERED_TOPIC,
                        f"{event['index_sequence']:020d}", canonical_json(event),
                    )
                for event in aggregate.late:
                    _produce(producer, LATE_TOPIC, event["tweet_id"], canonical_json(event))
                for message, error in dlq:
                    key = message.key().decode() if message.key() else "unknown"
                    _produce(producer, DLQ_TOPIC, key, _dlq_value("ordering", message, error))
                for key, value in trial.changelog_records(aggregate):
                    _produce(producer, STATE_TOPIC, key, value)
                latest: dict[tuple[str, int], int] = {}
                for message in messages:
                    identity = (message.topic(), message.partition())
                    latest[identity] = max(latest.get(identity, 0), message.offset() + 1)
                offsets = [
                    TopicPartition(topic_name, partition, offset)
                    for (topic_name, partition), offset in latest.items()
                ]
                producer.send_offsets_to_transaction(
                    offsets, consumer.consumer_group_metadata()
                )
                producer.commit_transaction()
                orderer = trial
                handled += len(messages)
            except Exception:
                producer.abort_transaction()
                raise
    finally:
        consumer.close()
    print(
        f"handled={handled} ordered_sequence={orderer.next_sequence} "
        f"watermark={orderer.committed_watermark} buffered={len(orderer.buffer)}"
    )


def run_index_writer(args: argparse.Namespace) -> None:
    Consumer, Producer, TopicPartition = _kafka()
    import fcntl
    from ..rolling_hnsw import RollingBuildConfig
    from .incremental_writer import RollingTemporalHNSWWriter

    config = RollingBuildConfig(
        vector_count=args.capacity,
        semantic_degree=args.semantic_degree,
        gateway_degree=args.gateway_degree,
        gateway_period=args.gateway_period,
        construction_candidates=args.construction_candidates,
        ef_construction=args.ef_construction,
        candidate_index_m=args.candidate_index_m,
        threads=args.threads,
        range_degree_per_side=args.range_degree_per_side,
        range_candidate_count=args.range_candidate_count,
        range_scales_hours=args.range_scales_hours,
    )
    args.index.mkdir(parents=True, exist_ok=True)
    lease_handle = (args.index / "WRITER.lock").open("a+")
    try:
        fcntl.flock(lease_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        lease_handle.close()
        raise RuntimeError(f"another writer owns {args.index}") from error
    writer = RollingTemporalHNSWWriter.create(
        args.index, config, dimensions=384, index_name=args.index_name
    )
    consumer = Consumer(_consumer_config(args.bootstrap, args.group))
    checkpoint_producer = Producer(_producer_config(args.bootstrap))
    consumer.subscribe([ORDERED_TOPIC])
    handled = 0
    try:
        while args.max_records <= 0 or handled < args.max_records:
            remaining = args.batch_size if args.max_records <= 0 else min(
                args.batch_size, args.max_records - handled
            )
            messages = consumer.consume(remaining, timeout=args.poll_timeout)
            messages = [message for message in messages if not message.error()]
            if not messages:
                continue
            messages.sort(key=lambda message: (message.partition(), message.offset()))
            events = [json.loads(message.value()) for message in messages]
            latest = max(messages, key=lambda message: message.offset())
            checkpoint = {
                "topic": latest.topic(),
                "partition": latest.partition(),
                "offset": latest.offset(),
            }
            result = writer.append_batch(events, checkpoint=checkpoint)
            checkpoint_value = canonical_json({
                "index_name": args.index_name,
                "generation": writer.current_generation,
                "last_index_sequence": writer.last_sequence,
                "vector_count": writer.vector_count,
                "source": checkpoint,
                "published_at_ms": int(time.time() * 1000),
            })
            _produce(checkpoint_producer, CHECKPOINT_TOPIC, args.index_name, checkpoint_value)
            checkpoint_producer.flush()
            consumer.commit(
                offsets=[TopicPartition(
                    latest.topic(), latest.partition(), latest.offset() + 1
                )],
                asynchronous=False,
            )
            handled += len(messages)
            print(
                f"index writer: consumed={handled} appended={result['appended']} "
                f"sequence={writer.last_sequence} generation={writer.current_generation}",
                flush=True,
            )
    finally:
        consumer.close()
        fcntl.flock(lease_handle.fileno(), fcntl.LOCK_UN)
        lease_handle.close()


def producer_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Produce validated Twitter JSONL to Kafka")
    parser.add_argument("--bootstrap", default=os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:29092"))
    parser.add_argument("--input", type=Path, default=Path("-"))
    return parser


def embedder_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run deterministic development embedding worker")
    parser.add_argument("--bootstrap", default=os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:29092"))
    parser.add_argument("--group", default="rt-hnsw-embedder-v1")
    parser.add_argument("--transactional-id", default="")
    parser.add_argument("--index-name", default="twitter-rt-hnsw")
    parser.add_argument("--poll-timeout", type=float, default=1.0)
    parser.add_argument("--max-records", type=int, default=0)
    return parser


def orderer_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Order embedded Twitter events by bounded event time")
    parser.add_argument("--bootstrap", default=os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:29092"))
    parser.add_argument("--group", default="rt-hnsw-orderer-v1")
    parser.add_argument("--transactional-id", default="")
    parser.add_argument("--index-name", default="twitter-rt-hnsw")
    parser.add_argument("--allowed-lateness-ms", type=int, default=600_000)
    parser.add_argument("--idle-timeout-ms", type=int, default=60_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--poll-timeout", type=float, default=1.0)
    parser.add_argument("--max-records", type=int, default=0)
    return parser


def index_writer_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Append ordered Kafka events to durable RT-HNSW generations")
    parser.add_argument("--bootstrap", default=os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:29092"))
    parser.add_argument("--group", default="rt-hnsw-index-writer-v1")
    parser.add_argument("--index-name", default="twitter-rt-hnsw")
    parser.add_argument("--index", type=Path, default=Path("Twitter_data/ann/streaming_rt_hnsw"))
    parser.add_argument("--capacity", type=int, default=10_000_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--poll-timeout", type=float, default=1.0)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--semantic-degree", type=int, default=20)
    parser.add_argument("--gateway-degree", type=int, default=8)
    parser.add_argument("--gateway-period", type=int, default=512)
    parser.add_argument("--construction-candidates", type=int, default=64)
    parser.add_argument("--ef-construction", type=int, default=200)
    parser.add_argument("--candidate-index-m", type=int, default=32)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--range-degree-per-side", type=int, default=2)
    parser.add_argument("--range-candidate-count", type=int, default=64)
    parser.add_argument("--range-scales-hours", default="24,168,720")
    return parser


def producer_main() -> None:
    produce_jsonl(producer_parser().parse_args())


def embedder_main() -> None:
    run_embedder(embedder_parser().parse_args())


def orderer_main() -> None:
    run_orderer(orderer_parser().parse_args())


def index_writer_main() -> None:
    run_index_writer(index_writer_parser().parse_args())
