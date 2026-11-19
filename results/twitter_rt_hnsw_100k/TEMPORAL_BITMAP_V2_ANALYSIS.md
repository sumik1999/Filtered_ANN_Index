# Packed time-stretched temporal-bitmap RT-HNSW v2

## Requested mechanisms

All six mechanisms are implemented in this branch.

### 1. Packed epoch-grouped adjacency

Adjacency is persisted as two-level CSR:

```text
(node, edge group) -> sorted epoch blocks -> packed neighbor IDs
```

Files use `uint32` block/neighbor offsets and contain no duplicate fixed-width query adjacency by default. Query traversal binary-searches only epoch blocks intersecting the temporal predicate.

### 2. Vectorized traversal

Selected neighbor IDs are filtered as NumPy arrays and vector distances are evaluated in batches. Heap maintenance remains Python. This validates packed/vectorized semantics, but small per-node arrays make NumPy dispatch expensive; a fused C++ loop is still needed for latency.

### 3. Fixed degree reallocation

The nominal level-0 budget is exactly 32:

| Class | Degree |
|---|---:|
| Same-epoch local semantic | 4 |
| New-to-old broad semantic | 8 |
| Old-to-new future reciprocal | 8 |
| 24/168/720-hour range | 6 |
| 1/16-epoch temporal bridges | 4 |
| Protected predecessor/successor | 2 |
| **Total** | **32** |

Nominal broad semantic capacity is 16 edges, balanced by direction so old regions retain forward navigation as new records arrive. Observed average degree is 24.47: 4.00 local, 8.00 historical, 3.78 future, 4.47 range, 2.22 bridge and 2.00 temporal. Bounded reciprocal replacement leaves future and bridge capacity underfilled, so observed broad semantic degree is 11.78 rather than 16.

### 4. Region-assigned upper portals

Periodic portals are assigned to exact 24-hour epoch regions through CSR epoch directories. Each portal has four semantic links restricted to its epoch and adjacent epochs plus protected chronological portal links. Query entry considers only portals assigned to intersecting epochs, enforces the exact interval, and performs a bounded portal beam before level-0 traversal.

### 5. Kafka incremental writer

`kafka-twitter-index-writer --topology temporal-bitmap` consumes the canonical ordered topic. It persists insertion-only internal adjacency, edge scores and the FAISS candidate generator. On each generation only new vectors execute graph insertion; existing graph nodes are not reinserted. Immutable packed blocks, bitmaps and portal directories are regenerated as query-side summaries, then `CURRENT` advances. Sequence gaps, decreasing event-time keys and conflicting replay are rejected.

Differential tests show a two-batch Kafka build produces the same packed graph and portal topology as uninterrupted chronological construction.

## 100k build

- Build time: 136.7 seconds
- Epochs: 38
- Portals: 195
- Query artifact size: 34.5 MB
- Existing flat multi-scale graph: approximately 15.6 MB

Packed block metadata and eight per-node epoch summaries increase storage even though fixed-width query adjacency is omitted.

## efSearch 256

| Variant | Recall@100 | Mean latency | p95 latency | Distance computations | Edges examined | Rejected edges |
|---|---:|---:|---:|---:|---:|---:|
| Packed epochs + portal navigation | 0.9443 | 19.40 ms | 98.00 ms | 2,868 | 3,575 | 211 |
| Packed, no epoch pruning + portals | 0.9443 | 13.97 ms | 69.22 ms | 2,868 | 5,270 | 1,906 |
| Packed epochs + direct entries | 0.9442 | 19.50 ms | 97.83 ms | 2,875 | 3,587 | 211 |
| Packed, no pruning + direct entries | 0.9442 | 13.92 ms | 68.82 ms | 2,875 | 5,286 | 1,910 |

Epoch-block pruning provides identical recall on the same entry policy. It reduces edge examination by 32.2% and explicit out-of-range rejection by 88.9%. It does not reduce distance computations because strict traversal already rejects those destinations before distance evaluation.

The vectorized Python implementation is slower: packed pruning adds approximately 39% mean latency versus scanning all packed blocks. Per-node arrays are too small to amortize NumPy calls. Packed storage is appropriate for a fused C++ traversal, not the current Python heap loop.

Region portal navigation visits 23.0 portal nodes and examines 71.3 portal edges per query. It improves recall by only 0.0067 percentage points and saves about six distance computations per query. It is active but not material.

## Quality comparison

| Method | Overall recall | 168-hour recall | 720-hour recall |
|---|---:|---:|---:|
| Packed bitmap v2 | 0.9443 | 0.8546 | 0.8667 |
| Existing flat multi-scale | 0.9515 | 0.8699 | 0.8873 |

The fixed-budget directional graph remains below the existing flat multi-scale topology. Protecting eight future reciprocal slots reduces the diversity obtained by ordinary reciprocal semantic pruning, while deterministic bridge capacity does not compensate on broad queries.

## Decision

The implementation successfully validates packed epoch pruning, fixed directional budgets, region portals and Kafka continuation, but it should not replace flat multi-scale RT-HNSW. The useful result is structural: packed epoch blocks eliminate most rejected-edge decoding with no recall change. The next defensible optimization is a compiled traversal over these blocks; further degree or portal tuning in Python is not justified.
