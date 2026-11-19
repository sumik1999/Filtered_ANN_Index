# RT-HNSW query-filter correctness test plan

## Purpose

This plan treats temporal filtering as a correctness boundary, not merely a recall metric. An RT-HNSW query must never return or score a database vector outside the normalized union of its half-open intervals. Approximate recall may vary with topology and beam width; filter correctness may not.

## Core invariants

For normalized intervals \(I\), eligible IDs \(E\), returned IDs \(R\), and every database vector row read during search \(S\):

1. **Strict result isolation:** `R ⊆ E`.
2. **Strict scoring isolation:** `S ⊆ E`.
3. **Half-open semantics:** a timestamp equal to `start` is eligible and one equal to `end` is not.
4. **Union cardinality:** `eligible_count == |E|`; overlap cannot be double-counted.
5. **Normalization invariance:** interval order, duplicates, nesting, overlap and touching boundaries do not change results.
6. **Uniqueness:** returned IDs contain no duplicates after component merging.
7. **Distance ordering:** returned distances are nondecreasing and correspond to the returned IDs.
8. **Exact differential correctness:** whenever exact routing is selected, IDs and distances equal an independent brute-force oracle.
9. **Planner boundary:** an eligible population equal to the exact threshold routes exactly; only larger populations are approximate.
10. **Topology independence:** semantic, temporal, range, gateway and upper-level edge traversal must all preserve invariants 1 and 2.

## Implemented deterministic test index

`tests/test_twitter_rolling_filtering.py` builds a persistent 32-vector index with:

- timestamps deliberately shuffled before construction;
- two vectors at every timestamp;
- semantic, temporal, gateway and multi-scale range edges;
- a two-level hierarchy with upper range edges; and
- a small gateway period that exercises entry selection.

A tracking matrix wraps the database vectors after construction and records every row requested by query execution. This checks the stronger scoring-isolation property rather than only inspecting final results.

## Implemented cases

### Interval boundaries and normalization

- Duplicate records exactly at interval starts and ends.
- Empty interval list.
- Intervals entirely before or after the indexed timeline.
- Full signed-int64 timestamp domain.
- Overlapping, nested, duplicated, touching and reordered intervals.
- Invalid zero-width, reversed and malformed intervals.

### Routing and component behavior

- Eligible count exactly equal to the exact threshold.
- Mixed disjoint query where one component uses exact search and another uses graph search.
- Graph component with no eligible gateway, forcing exact fallback.
- `k` larger than the eligible population.
- Deduplication and global distance ordering after component merge.

### Graph topology

Filter isolation is checked independently for:

1. semantic-only traversal;
2. semantic plus protected temporal traversal;
3. gateway-edge traversal;
4. multi-scale range traversal; and
5. hierarchical semantic/range/temporal traversal.

### Deterministic property tests

Two seeded suites generate 80 interval unions each:

- exact searches are compared ID-for-ID and distance-for-distance with brute force;
- graph searches audit every vector-row access and every returned ID for eligibility.

Seeds are fixed so failures are reproducible in CI.

### Input validation

- Query dimension mismatch remains covered by the base tests.
- NaN and infinity query components are rejected before routing.
- Invalid entry-gateway directories remain covered by gateway tests.

## Production-scale differential test matrix

The next integration suite should use the persisted 100k multi-scale index and exact ground truth. It should sample at least the following matrix:

| Dimension | Values |
|---|---|
| Coverage | empty, one-record, 1h, 6h, 24h, 168h, 720h, full timeline |
| Fragment count | 1, 2, 4, 8, 16 |
| Selectivity | 0, 1, `k-1`, `k`, 4,999, 5,000, 5,001, 10%, 50%, 100% |
| Boundary location | timeline start/end, duplicate timestamps, burst edge, quiet gap |
| `k` | 1, 10, 100, greater than eligible count |
| `efSearch` | `k`, 128, 256, 512, 1024 |
| Topology | semantic, temporal, range, gateway, hierarchy |

Every run must recompute the eligibility mask independently from source timestamps. Recall is measured only after strict-filter invariants pass.

## Metamorphic tests for the 100k index

For each selected query:

1. Permute interval order; output must not change.
2. Duplicate every interval; output must not change.
3. Split `[a,c)` at `b` into `[a,b),[b,c)`; output must not change.
4. Add an interval wholly contained in another; output must not change.
5. Add an empty external region containing no records; output must not change.
6. Compare exact and graph routes around threshold 5,000; both must return only eligible IDs.
7. Disable each edge class in turn; quality may change, filter validity may not.

## Adversarial datasets still required

Small generated indexes should cover:

- all records sharing one timestamp;
- long periods with no records;
- extreme burst density at a boundary;
- identical vectors at different timestamps;
- vectors whose globally closest matches are all outside the query interval;
- gateway spacing larger than every query component;
- hierarchy levels with no eligible upper node;
- one eligible upper node with every upper edge out of range;
- minimum legal graph degrees and saturated fixed-width adjacency;
- stable timestamp ties across insertion batches.

## Persistence and corruption tests

After a successful build:

1. Close and reload the index and rerun a fixed query corpus.
2. Verify byte-identical IDs and numerically equal distances before and after reload.
3. Reject manifests with incompatible vector count or dimensions.
4. Reject unsorted timestamp artifacts.
5. Reject out-of-range neighbour IDs and malformed adjacency shapes during validation.
6. Simulate a truncated artifact and require a clear load failure.

Items 3–6 require stronger load-time artifact validation than currently implemented.

## Performance and regression tests

Correctness tests should not use wall-clock assertions. A separate stable benchmark should record:

- routing, search and rerank latency;
- distance computations;
- nodes visited;
- edge examinations and out-of-range rejection by edge class;
- exact/graph route counts;
- eligible population and fragment count.

A regression is flagged when, at fixed recall and configuration:

- distance computations increase by more than 10%;
- p95 latency increases by more than 15% across repeated warmed runs; or
- custom graph storage increases without an explicit budget change.

## CI tiers

### Per commit

- All small deterministic unit and property tests.
- Fixed seeds.
- Target runtime below five seconds.

### Nightly

- 100k persisted index differential matrix.
- Multiple property-test seeds.
- Persistence/reload checks.
- Warmed performance snapshot.

### Release

- 100k, 250k and full-scale datasets.
- Multiple workload seeds.
- Recall, latency, storage and insertion regression reports.
- Zero tolerated strict-filter or scoring-isolation violations.

## Acceptance rule

A topology is eligible for performance comparison only if it has zero filter violations across unit, randomized, persisted-index and metamorphic suites. Approximate recall regressions may be traded against latency; temporal correctness cannot.
