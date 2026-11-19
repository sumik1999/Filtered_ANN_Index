#!/usr/bin/env bash
set -euo pipefail

BOOTSTRAP_SERVER="${1:-localhost:29092}"
KAFKA_TOPICS="/opt/bitnami/kafka/bin/kafka-topics.sh"
KAFKA_CONFIGS="/opt/bitnami/kafka/bin/kafka-configs.sh"

create_topic() {
  local topic="$1"
  local partitions="$2"
  local retention_ms="$3"
  "$KAFKA_TOPICS" \
    --bootstrap-server "$BOOTSTRAP_SERVER" \
    --create --if-not-exists \
    --topic "$topic" \
    --partitions "$partitions" \
    --replication-factor 1 \
    --config compression.type=producer \
    --config retention.ms="$retention_ms"
}

create_topic twitter.raw.v1 6 604800000
create_topic twitter.embedded.v1 6 604800000
create_topic twitter.index.ordered.v1 1 -1
create_topic twitter.index.late.v1 3 2592000000
create_topic twitter.index.dlq.v1 3 2592000000

"$KAFKA_TOPICS" \
  --bootstrap-server "$BOOTSTRAP_SERVER" \
  --create --if-not-exists \
  --topic twitter.index.checkpoints.v1 \
  --partitions 1 \
  --replication-factor 1 \
  --config cleanup.policy=compact \
  --config min.cleanable.dirty.ratio=0.01

# Preserve the canonical ordered log indefinitely and use compacted checkpoint state.
"$KAFKA_CONFIGS" --bootstrap-server "$BOOTSTRAP_SERVER" \
  --entity-type topics --entity-name twitter.index.ordered.v1 \
  --alter --add-config cleanup.policy=delete,retention.ms=-1

echo "RT-HNSW Kafka topics are ready on ${BOOTSTRAP_SERVER}:"
"$KAFKA_TOPICS" --bootstrap-server "$BOOTSTRAP_SERVER" --list
