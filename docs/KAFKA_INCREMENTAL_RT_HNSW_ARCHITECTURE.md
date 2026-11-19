# Kafka architecture for ordered incremental RT-HNSW updates

## Status and scope

This document defines the streaming architecture for ingesting new Twitter records and publishing ordered, incremental RT-HNSW snapshots. The branch now includes a correctness-first durable single-writer implementation for the flat multi-scale topology; production storage optimization and live query-service integration remain outstanding.

The initial design optimizes for correctness and recoverability rather than maximum write throughput:

- one logical writer owns an RT-HNSW index;
- index sequence is globally ordered by `(event_time_ms, tweet_id)`;
- bounded out-of-order arrivals are reordered before indexing;
- records later than the committed watermark are quarantined rather than corrupting chronological local-ID order;
- query readers observe immutable, atomically published snapshots; and
- every update is replayable and idempotent.

## Why Kafka ordering alone is insufficient

Kafka guarantees ordering only inside one partition, and producer arrival order is not event-time order. Partitioning raw tweets by tweet ID therefore cannot preserve the RT-HNSW invariant:

```text
local_id order == nondecreasing timestamp order
```

The system needs an explicit event-time ordering stage followed by a single-partition ordered log. A single raw partition would preserve arrival order, not event-time order, and would not solve late arrivals.

## Data flow

```text
Twitter source
    |
    v
 twitter.raw.v1 (6 partitions, keyed by tweet_id)
    |
    v
 Validator / deduplicator
    | invalid ----------------------------> twitter.index.dlq.v1
    v
 Embedding workers (parallel)
    |
    v
 twitter.embedded.v1 (6 partitions)
    |
    v
 Event-time orderer + watermark + sequence allocator
    | too late ----------------------------> twitter.index.late.v1
    v
 twitter.index.ordered.v1 (1 partition)
    |
    v
 Durable RT-HNSW single writer
    |
    +--> immutable generation N+1
    +--> atomic CURRENT manifest swap
    +--> twitter.index.checkpoints.v1
    |
    v
 Query processes reload generation N+1
```

## Topic design

| Topic | Partitions | Key | Retention | Purpose |
|---|---:|---|---|---|
| `twitter.raw.v1` | 6 | `tweet_id` | 7 days | Validated source envelope before embedding |
| `twitter.embedded.v1` | 6 | `tweet_id` | 7 days | Embedding-complete records; may be out of order |
| `twitter.index.ordered.v1` | 1 | `index_sequence` | Indefinite | Canonical replay log for one index |
| `twitter.index.late.v1` | 3 | `tweet_id` | 30 days | Events behind the committed watermark |
| `twitter.index.dlq.v1` | 3 | `tweet_id` | 30 days | Invalid schema, embedding or validation failures |
| `twitter.orderer.state.v1` | 1 | state key | Compacted | Transactional orderer buffer, dedupe and watermark changelog |
| `twitter.index.checkpoints.v1` | 1 | `index_name` | Compacted | Published generation and committed sequence metadata |

The ordered topic must not be compacted because it is the authoritative reconstruction log. Compression should be `zstd`, and producers should use `acks=all` with idempotence enabled.

## Event contracts

Development JSON contracts are stored under `schemas/kafka/`. Production should use Avro or Protobuf with Schema Registry and `BACKWARD_TRANSITIVE` compatibility.

### Raw event

Required fields:

```text
schema_version
source_event_id
tweet_id
event_time_ms
ingest_time_ms
text
language
```

`source_event_id` is the idempotency key across source retries. `tweet_id` is the semantic record identity.

### Ordered index event

Required fields:

```text
schema_version
index_name
index_sequence
tweet_id
event_time_ms
ingest_time_ms
embedding_model
embedding_dimension
embedding
source_event_id
content_sha256
```

`index_sequence` is allocated only by the orderer. For one index it starts at zero and is contiguous. The index writer rejects a gap and ignores an exact duplicate sequence only when its content hash matches.

## Validation and deduplication

Before embedding:

1. Validate schema and field sizes.
2. Require finite, signed-int64-compatible timestamps.
3. Normalize text and reject empty text.
4. Deduplicate by `source_event_id` and then `tweet_id` using a persistent state store.
5. Route conflicting duplicate IDs to the DLQ rather than silently replacing indexed content.
6. Record embedding model and normalization policy.
7. Require a finite 384-dimensional `float32` vector after embedding.
8. Normalize the vector exactly as the existing index expects.

Deduplication state must outlive raw-topic retention. A Kafka Streams state store backed by a changelog topic or an external durable key-value store is appropriate.

## Event-time ordering and watermark

The orderer consumes every `twitter.embedded.v1` partition and keeps a persistent min-heap/state store ordered by:

```text
(event_time_ms, tweet_id)
```

For each active input partition `p`, track its largest observed event time `max_event_time[p]`. The safe global watermark is:

```text
global_watermark = min(max_event_time[p] for active p) - allowed_lateness_ms
```

Records at or below the watermark can be emitted in deterministic order. A starting `allowed_lateness_ms` of ten minutes is reasonable for testing but must be chosen from measured embedding and source-delay percentiles.

### Idle partitions

A partition that has received no records for a configured idle timeout must be marked idle; otherwise one quiet partition can stop the watermark forever. When it becomes active again, records older than the committed watermark are late records.

### Tie handling

Tweets can share a millisecond timestamp. `tweet_id` provides deterministic ordering within a timestamp. Once a timestamp has crossed the watermark, a later record with the same or earlier ordering key is late and cannot enter the main graph directly.

### Transactional output

The orderer uses Kafka transactions to atomically:

1. emit an ordered batch;
2. emit late/DLQ records where applicable; and
3. commit embedded-topic consumer offsets.

This gives exactly-once behavior inside Kafka. It does not make the external index update transactional; index idempotence handles that boundary.

## Late-arrival policy

Appending an old timestamp to the current graph would violate chronological local-ID order. The initial policy is therefore explicit:

```text
event_time > committed watermark  -> ordered main log
event_time <= committed watermark -> late-event topic
```

Late records are not dropped. They can be handled by one of three later policies:

1. **Rebuild:** include late events in a new complete generation.
2. **Delta index:** query a small late-event index alongside the main index and merge results.
3. **Bounded repair:** insert into a mutable recent segment before that segment is sealed.

Phase one uses quarantine plus scheduled rebuild. Silent insertion into the main graph is forbidden.

## Single-writer RT-HNSW service

Only one active writer may own an index. A Kafka consumer group with the single ordered partition naturally elects one consumer, but the writer should also use an index-directory lease/fencing token to prevent two processes from publishing generations.

### Required incremental writer API

The current monolithic builder should be refactored behind:

```python
writer = RollingTemporalHNSWWriter(index_directory)
result = writer.append_batch(
    events,
    checkpoint={"topic": topic, "partition": partition, "offset": offset},
)
```

`append_batch` durably writes its WAL, updates the graph, publishes an immutable
generation, advances `CURRENT`, and only then returns to the Kafka consumer.
It must:

1. verify contiguous sequence numbers;
2. verify nondecreasing `(event_time_ms, tweet_id)` keys;
3. reject non-finite or dimensionally incompatible vectors;
4. assign chronological local IDs;
5. update the semantic candidate index;
6. create semantic, temporal, multi-scale and reciprocal edges;
7. update gateway/hierarchy state if enabled; and
8. write changes to a non-current generation.

The FAISS construction-candidate index is mutable state and must be checkpointed with each generation or deterministically rebuilt on restart. Persisting only custom adjacency is insufficient for continued insertion.

## Micro-batch commit protocol

Use contiguous micro-batches, initially 256–1,024 events or a one-second maximum delay.

For batch `[first_sequence, last_sequence]`:

1. Poll records without committing Kafka offsets.
2. Validate the sequence and content hashes.
3. Write a local append WAL containing the complete batch.
4. Flush and `fsync` the WAL.
5. Build generation `N+1` in a staging directory using copy-on-write or append-only files.
6. Persist all adjacency, vectors, timestamps, candidate-index state and metadata.
7. Write a generation manifest containing sequence range, ordered-topic offsets and checksums.
8. `fsync` files and directory entries.
9. Atomically replace the `CURRENT` manifest pointer.
10. Emit the checkpoint record.
11. Commit the Kafka consumer offset.
12. Remove WAL entries older than the published generation.

### Crash behavior

| Failure point | Recovery |
|---|---|
| Before WAL flush | Kafka redelivers the batch |
| After WAL, before publish | Recover or discard staging generation, then replay |
| After publish, before offset commit | Kafka redelivers; writer sees sequence already published and verifies hash before skipping |
| After offset commit | Published generation is already durable |

Kafka and filesystem publication are not one atomic transaction. The published `last_index_sequence` and per-event hash make replay safe.

## Snapshot layout

Recommended index directory:

```text
index/
  CURRENT
  generations/
    0000000000000042/
      manifest.json
      vectors.npy
      timestamps.npy
      global_ids.npy
      semantic_neighbors.npy
      temporal_neighbors.npy
      range_left_neighbors.npy
      range_right_neighbors.npy
      gateway_*.npy
      construction.faiss
      checkpoint.json
  wal/
  leases/
```

For early correctness testing, writing complete generations is acceptable but expensive. The next storage iteration should use immutable base files plus append-only chunks and periodic compaction.

## Query consistency

Query processes never read staging files. They resolve `CURRENT`, open one immutable generation and retain it for the duration of a query.

Publication uses read-copy-update semantics:

1. Writer publishes generation `N+1` atomically.
2. New queries open `N+1`.
3. Queries already using `N` finish normally.
4. Generation `N` is removed only after reader references expire.

The API should return the generation and `last_index_sequence` with diagnostics so clients can request a minimum freshness level.

## Backpressure

The ordered log is the durable buffer. If index construction slows:

- do not drop or reorder records;
- expose consumer lag in records, bytes and event-time seconds;
- pause raw ingestion only when retention or disk limits are threatened;
- increase micro-batch size within an insertion-latency SLO;
- never add more consumers to the same index partition, because one writer is intentional.

If one writer cannot sustain source rate, shard into independent indexes with a query-time merge. Do not increase ordered-topic partitions for one index unless chronological ordering is redesigned around sealed temporal segments.

## Observability

Required metrics:

### Ingestion

- records received, validated, duplicated and rejected;
- embedding latency and failures;
- per-partition lag;
- source-to-index event-time latency.

### Ordering

- current watermark;
- reorder-buffer size and age;
- idle partitions;
- late-record count and lateness distribution;
- sequence gaps and transactional aborts.

### Index writer

- last consumed and published sequence;
- batch size and append latency p50/p95/p99;
- WAL and generation size;
- candidate-generation and edge-update latency;
- checkpoint and publish latency;
- replayed/duplicate records;
- index generation age;
- insertion failures.

### Query service

- generation and sequence served;
- snapshot reload latency;
- query recall regression corpus;
- exact/graph routing rates;
- query latency during publication.

Alerts should cover stalled watermark, nonzero sequence gaps, late-rate spikes, publication failures, growing consumer lag and index/query generation divergence.

## Security and data governance

- TLS and SASL for non-development Kafka.
- Topic ACLs: producers cannot write ordered/checkpoint topics directly.
- Secrets outside source control.
- Retention and deletion policy for tweet text.
- Prefer storing only the normalized embedding and required metadata in the ordered index log if raw text retention is unnecessary.
- Audit embedding-model changes; a model-version change requires a new index or complete rebuild.

## Testing strategy

### Unit tests

- schema validation and finite vectors;
- deterministic tie ordering;
- watermark movement and idle partitions;
- duplicate source events;
- sequence gap/conflict rejection;
- late-event routing.

### Integration tests with Kafka

1. Produce timestamps in random arrival order across all raw partitions.
2. Delay selected embeddings.
3. Verify the ordered topic is monotonic by `(event_time_ms, tweet_id)` with contiguous sequence numbers.
4. Kill the orderer during a transaction and verify no duplicates/gaps.
5. Kill the writer at every micro-batch commit step and verify one published copy of every sequence.
6. Restart from the checkpoint and compare the index byte/content manifest with a clean replay.
7. Query continuously during publication and verify every query uses one complete generation.
8. Inject events behind the watermark and verify they appear only in the late topic.

### Differential index test

Build two indexes from the same canonical event log:

- one using the existing offline builder;
- one using Kafka micro-batch replay.

Require identical chronological IDs, timestamps and protected temporal links. Semantic/range adjacency should be identical when insertion batching does not change candidate-index behavior; otherwise require documented deterministic equivalence and equal query results on a fixed corpus.

## Delivery phases

### Phase 0: local infrastructure

- Start the development Kafka broker from `deploy/kafka/docker-compose.yml`.
- Create topics with `deploy/kafka/create-topics.sh`.
- Validate JSON contracts under `schemas/kafka/`.

### Phase 1: ordered log only

Implemented in this branch:

- validated JSONL producer;
- deterministic normalized development embedding worker;
- bounded event-time watermark orderer;
- contiguous index-sequence allocation;
- duplicate/conflict detection and late-event routing; and
- a transactionally written compacted state changelog for restart recovery.

The Kafka adapter atomically commits ordered/late/DLQ output, orderer state and
embedded-input offsets in one Kafka transaction. The in-memory state is a cache
reconstructed from `twitter.orderer.state.v1` after restart. Broker-level failure
injection remains required before Phase 1 is considered production-complete.

### Phase 2: durable incremental writer

Implemented in this branch for the flat multi-scale topology:

- `RollingTemporalHNSWWriter.create/open` and ordered micro-batch append;
- incremental semantic, temporal, range and gateway edge updates;
- persisted FAISS construction-candidate state;
- append WAL and automatic orphaned-WAL recovery;
- contiguous-sequence and event-time ordering enforcement;
- idempotent replay with content-hash conflict detection;
- immutable checksummed generations and atomic `CURRENT` publication;
- advisory single-writer lease in the Kafka service;
- compacted checkpoint publication after generation commit; and
- differential tests against the uninterrupted offline builder.

The correctness-first publisher currently rewrites complete NumPy artifacts for
each generation. It supports the recommended flat topology only. Chunked
append-only arrays, generation garbage collection, hierarchy updates and broker
failure-injection tests remain before production use.

### Phase 3: live query snapshots

Implemented in this branch:

- background polling and atomic reload of `CURRENT`;
- read-copy-update snapshot switching;
- per-query generation pinning and reference counts;
- cross-process shared reader leases;
- safe old-generation collection using an exclusive lease;
- minimum-index-sequence freshness waits with bounded timeout;
- concurrent query/update isolation tests; and
- a threaded HTTP service exposing `/search`, `/health`, `/status` and
  `/refresh`.

Every response identifies the generation and last index sequence that served
it. Existing queries finish on their original generation while new queries use
the newly published one. Production SLO calibration, authentication, request
rate limiting and optimized chunked generation storage remain outstanding.

### Phase 4: late-data support

- Start with scheduled rebuild.
- Evaluate recent mutable segments or a query-merged late delta index.
- Preserve a strict no-silent-corruption policy.

### Phase 5: production hardening

- Multi-broker Kafka, replication factor three and min ISR two.
- Schema Registry, TLS/SASL and ACLs.
- Immutable chunk storage and compaction.
- Full failure-injection, backup and disaster-recovery testing.

## Initial acceptance criteria

The first end-to-end version is acceptable only when:

1. one million shuffled input events produce a gap-free ordered sequence;
2. no main-index timestamp is lower than its predecessor;
3. duplicates do not create additional vectors;
4. late events are observable and never silently appended;
5. process termination at any commit point recovers without loss or duplicate index rows;
6. clean replay and incremental replay produce equivalent indexes;
7. query readers never observe partial generations; and
8. sustained writer throughput exceeds measured source rate with at least 2x headroom.
