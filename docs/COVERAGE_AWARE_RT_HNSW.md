# Coverage-Aware Flat Multi-Scale RT-HNSW

## Purpose

This extension tests whether broad-range recall failures in flat multi-scale
RT-HNSW are caused by temporal regions that the similarity-prioritized beam
never enters. It adds a compact temporal block directory and a mandatory
coverage phase without changing the established level-0 graph topology.

The design deliberately separates two objectives:

1. **Temporal coverage:** every temporal block intersecting a graph-searched
   query component receives a minimum expansion budget.
2. **Approximate nearest-neighbor quality:** the remaining global beam follows
   vector distance as before.

Coverage does not imply that every eligible vector is examined or that exact
nearest neighbors are found.

## Preserved behavior

The extension retains:

- stable chronological local IDs;
- semantic, protected temporal, and multi-scale range edges;
- strict half-open filtering before destination distance computation;
- exact fallback for small eligible populations;
- one global graph rather than one ANN index per block; and
- independent handling of disjoint query components.

No transitive min/max reachability metadata is attached to ordinary HNSW
edges. Such metadata becomes broad in a cyclic graph and is difficult to
maintain conservatively.

## Persisted metadata

For fixed population block size `S`, local vector `i` belongs to block
`floor(i / S)`. Each block persists:

- left and right local-ID offsets;
- minimum observed timestamp;
- maximum observed timestamp; and
- up to `g` representative local IDs.

Representatives are chosen from at most 64 evenly sampled block members:

1. first temporal member;
2. last temporal member;
3. sampled vector nearest the sampled centroid; and
4. additional farthest-point representatives.

Ordinary nodes need no additional descriptor array. The metadata cost is
`O(Bg)` for `B = ceil(N/S)` blocks, plus `O(B)` bounds and offsets.

Existing RT-HNSW artifacts that predate these files remain readable. Their
coverage metadata is derived in memory when loaded.

## Invariants

### Complete partition

Every local vector belongs to exactly one block, and block offsets cover
`[0, N)` without gaps or overlap.

### Conservative bounds

For every member `v` of block `Bi`:

```text
block_min[i] <= timestamp(v) <= block_max[i]
```

Bounds may cause extra block discovery but cannot omit an intersecting member.

### Strict component membership

A destination is scored only if its local ID lies inside the current normalized
half-open query component. Coverage expansion is additionally restricted to
its assigned block. Cross-block edges are deferred to the subsequent global
beam.

### Mandatory block coverage

Let `BQ` be the blocks intersecting graph-searched query component `Q`, and let
`EQ` be blocks for which at least one coverage candidate was expanded. With a
positive mandatory expansion count:

```text
EQ = BQ
```

The implementation records `coverage_blocks_considered`,
`coverage_blocks_covered`, and `coverage_expansions` so this invariant is
directly testable.

### Separate termination

Ordinary HNSW beam termination cannot stop the mandatory phase. After coverage
finishes, all scored coverage frontiers are merged into the usual global beam.

## Search algorithm

For each graph-searched component:

1. Locate its first and last fixed-population blocks using block offsets.
2. Reject only blocks whose conservative timestamp bounds are disjoint.
3. Filter each block's representatives to the strict component. Boundary local
   IDs are fallback representatives for partially intersected blocks.
4. Score at most `coverage_gateway_candidates` representatives and choose the
   closest as the block seed.
5. Expand the block-local candidate queue at least
   `coverage_expansions_per_block` times.
6. Merge all scored coverage frontiers into a global candidate heap.
7. Continue strict multi-scale RT-HNSW beam search with the normal `efSearch`
   result capacity.
8. Merge components and exact-rerank the retained candidates.

The prototype counts mandatory work in `distance_computations`; it is not free.
The current comparison therefore reports both recall and computation counts.
A later fixed-computation scheduler can reserve a fraction of a hard total
budget for coverage.

## Diagnostics

Every search reports:

- `coverage_blocks_considered`;
- `coverage_blocks_covered`;
- `coverage_expansions`;
- `coverage_gateway_candidates`;
- `coverage_distance_computations`;
- `coverage_cross_block_edges_skipped`; and
- `visited_block_ids`.

The benchmark maps exact top-k neighbors back to temporal blocks and adds:

- `ground_truth_block_count`;
- `visited_block_count`; and
- `ground_truth_block_coverage`.

This distinguishes two failure modes:

1. **Missed-region failure:** a block containing an exact top-k neighbor was
   never visited. Coverage scheduling has plausible headroom.
2. **Within-region failure:** all ground-truth blocks were visited but the exact
   vectors were missed. More block coverage is unlikely to help.

## Experiment

Build the established equal-nominal-degree multi-scale graph with coverage
metadata:

```bash
build-twitter-rt-hnsw \
  --output Twitter_data/ann/rolling_hnsw/flat_100k_coverage \
  --vector-count 100000 \
  --semantic-degree 20 \
  --range-degree-per-side 2 \
  --range-scales-hours 24,168,720 \
  --coverage-block-size 512 \
  --coverage-gateways-per-block 4
```

Compare the unchanged baseline and coverage scheduler on the same graph:

```bash
benchmark-twitter-rt-hnsw \
  --index Twitter_data/ann/rolling_hnsw/flat_100k_coverage \
  --variants semantic_temporal_range,coverage_multiscale \
  --ef-search 256 \
  --coverage-expansions-per-block 1 \
  --coverage-gateway-candidates 4 \
  --output results/twitter_rt_hnsw_100k/coverage_ef256
```

Sweep:

- block size: 256, 512, 1000, 2048;
- representatives per block: 1, 2, 4, 8;
- mandatory expansions: 1, 2, 5, 10; and
- `efSearch`: 128, 256, 512.

Evaluate 1, 6, 24, 168, and 720-hour coverage separately. Narrow components
will often use exact fallback, so the primary acceptance signal is broad-query
recall versus distance computations and p95 latency.

## Acceptance gate

The scheduler should not replace the recommended flat RT-HNSW baseline unless:

1. broad-range recall improves materially;
2. ground-truth block coverage identifies missed regions in the baseline;
3. the gain survives equal-distance-computation comparison;
4. p95 latency remains acceptable; and
5. strict filtering tests continue to prove that no out-of-range destination is
   scored or returned.

If baseline ground-truth block coverage is already near 1.0, further block
coverage work should be stopped in favor of improving semantic navigation
inside already visited blocks.
