# Time-stretched temporal-bitmap HNSW prototype

## Insertion invariants

Records are inserted in stable `(timestamp, source order)` order. For node `v`:

1. Global HNSW construction search finds older semantic candidates.
2. `v` stores the closest candidates in its historical list.
3. Each selected old node stores `v` in a separate bounded future-reciprocal list.
4. Same-epoch candidates are selected into an independent local semantic list with reciprocal links.
5. Predecessor/successor links are protected.
6. Co-eligibility-scored range edges are created independently at 24, 168 and 720 hours.
7. Deterministic bridges connect backward and reciprocally forward at 1, 4, 16 and 64 epochs.

Future insertion can therefore replace another future link, but cannot evict an old node's local or historical links.

## Bitmap layout

An epoch is 24 hours by default. Each node has one bitmap per edge group:

```text
local
historical
future
range_24h
range_168h
range_720h
bridge
temporal
```

The bitmap contains every epoch represented by a valid destination in that group. It is a conservative summary: a set bit may still produce an exact timestamp rejection at a partial epoch boundary, but a missing bit guarantees that the group contains no eligible destination in that epoch.

Portal coverage is the union of a periodic portal's own epoch and direct-neighbour edge-group bitmaps. Under strict traversal, the portal itself must still lie inside the exact query interval.

## Query algorithm

For each normalized half-open interval:

1. Binary-search exact local-ID bounds.
2. Select and score periodic portals inside those bounds.
3. Convert the interval to a coarse epoch range.
4. Before scanning an edge group, intersect its bitmap with the query epoch range.
5. Skip groups with empty intersection.
6. Apply exact local-ID range checks to every surviving destination before vector distance.
7. Merge, deduplicate and exactly rerank component candidates.

Bitmap pruning cannot alter recall relative to the same graph with pruning disabled when summaries have no false negatives.

## Current scope

Implemented:

- chronological online construction replay;
- protected edge classes by temporal direction;
- multi-scale range and skip-epoch bridges;
- arbitrary-length packed `uint64` epoch bitmaps;
- portal coverage summaries;
- strict single and disjoint interval search;
- exact fallback;
- pruning diagnostics and matched on/off benchmark;
- no-false-negative and strict-filter tests.

Not yet implemented:

- Kafka append support for this experimental topology;
- hierarchy-region coverage bitmaps;
- packed epoch-grouped adjacency blocks;
- deletion/expiration bitmap reference counts;
- C++ bitmap and adjacency traversal;
- bitmap combination with non-temporal metadata.

The persisted 100k experiment uses one 64-bit word because its timeline spans only 38 daily epochs. The implementation supports more words for longer timelines.
