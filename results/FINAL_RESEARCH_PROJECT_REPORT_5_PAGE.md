---
title: "Temporal-Filter-Aware Approximate Nearest-Neighbor Search"
subtitle: "A Comparative Study of Bucket HNSW, Flat RT-HNSW, and Coverage-Bounded RT-HNSW"
author: "Database Project — Final Research Submission"
date: "19 November 2026"
lang: en-US
---

# 1. Research objective and principal findings

## Abstract

This project investigates approximate nearest-neighbor search over Twitter
embeddings subject to timestamp predicates. Conventional HNSW organizes vectors
by semantic proximity but does not ensure that retrieved candidates satisfy a
selective temporal range. Three designs are evaluated: density-balanced bucket
HNSW, flat multi-scale rolling temporal HNSW (RT-HNSW), and coverage-bounded
RT-HNSW.

Bucket HNSW achieved 0.9991 Recall@100 on a 990,000-vector panel, but required
144.18 ms mean and 566.15 ms p95 latency. Flat RT-HNSW was evaluated on a
separate 100,000-vector rolling panel and achieved approximately 0.9515 recall
at `efSearch=256`, while enforcing temporal eligibility before vector-distance
calculation. Coverage-bounded RT-HNSW increased same-`efSearch` recall from
0.95143 to 0.95273 and guaranteed that every intersecting temporal block
received search effort. However, it required 8.65% more distance computations
and performed worse at an equal computation budget.

The project therefore recommends flat multi-scale RT-HNSW as the default
single-graph design. Bucket HNSW remains appropriate when near-exact broad-range
recall is required. Coverage-bounded search should be activated selectively for
queries at risk of missing temporal regions.

## Problem definition

Each database record contains embedding $x_i$, timestamp $t_i$, and stable
identifier $id_i$. Given query vector $q$, result count $k=100$, and intervals
$P=\{[a_j,b_j)\}$, the system must return the nearest vectors satisfying

$$
\exists [a_j,b_j)\in P:\quad a_j\le t_i<b_j.
$$

The filter is authoritative and half-open. In RT-HNSW, an invalid destination
is rejected before its vector is read or its distance is computed.

## Research contributions

1. A density-balanced bucket baseline with exact boundary filtering.
2. A single chronological RT-HNSW graph with protected temporal links and
   independent 24-, 168-, and 720-hour range edges.
3. A coverage-bounded extension with a formal temporal block-coverage
   invariant.
4. An equal-computation evaluation demonstrating that stronger coverage is not
   automatically a more efficient ANN strategy.
5. An ordered Kafka architecture for durable incremental graph updates and
   immutable live query snapshots.

<div class="page-break"></div>

```{=openxml}
<w:p><w:r><w:br w:type="page"/></w:r></w:p>
```

# 2. Flat multi-scale RT-HNSW design

## 2.1 Construction principle

Vectors are inserted in stable chronological order. Consequently, local vector
IDs follow nondecreasing timestamp order, and a temporal predicate can be
translated into one or more contiguous local-ID ranges by binary search.

FAISS HNSW is used only during construction to identify semantic candidates
among vectors that have already arrived. Query execution uses custom persisted
adjacency rather than the FAISS construction graph.

## 2.2 Graph topology

| Edge class | Per-node budget | Purpose |
|---|---:|---|
| Semantic | 20 | Preserve vector-space proximity |
| 24-hour left/right | 2 + 2 | Support narrow temporal neighborhoods |
| 168-hour left/right | 2 + 2 | Support weekly-scale predicates |
| 720-hour left/right | 2 + 2 | Support broad predicates |
| Temporal predecessor/successor | Separate | Guarantee chronological connectivity |

The predecessor and successor edges are protected: semantic pruning cannot
remove them. Therefore, every contiguous temporal range retains a basic path.

Range edges are selected independently at each scale. Their candidate pools
combine semantic candidates, samples from the relevant temporal window, and
immediate chronological neighbors. Candidates are ranked by vector distance
adjusted by temporal co-eligibility. Independent budgets prevent one temporal
scale from consuming all range-aware capacity.

## 2.3 Query mechanism

The search sequence is:

1. Normalize overlapping intervals into a disjoint half-open union.
2. Binary-search the ordered timestamp array to locate eligible local-ID
   components.
3. Count eligible vectors.
4. If at most 5,000 vectors are eligible, perform exact pre-filtered search.
5. Otherwise, choose periodic entry points located inside the component.
6. Run a beam search over semantic, temporal, and multi-scale range edges.
7. Reject an edge destination outside the component before distance evaluation.
8. Search disjoint components independently, deduplicate candidates, and exact
   rerank the retained set.

```text
interval predicate
    → timestamp-to-local-ID routing
    → exact fallback OR in-range graph entries
    → strict multi-scale beam traversal
    → deduplication and exact reranking
```

This architecture differs from post-filtering because the predicate constrains
entry selection and every graph expansion. It differs from bucket HNSW because
all vectors remain in one logical graph.

## 2.4 Design strengths and limitations

Flat RT-HNSW provides strict filtering, compact custom adjacency, and a tunable
quality-cost curve through `efSearch`. Its main limitation is broad-range
recall: at practical beam widths, semantic navigation inside a large eligible
subgraph remains approximate. Significantly late records also require a rebuild,
repair procedure, or separate delta index because direct insertion would break
chronological local-ID order.

<div class="page-break"></div>

```{=openxml}
<w:p><w:r><w:br w:type="page"/></w:r></w:p>
```

# 3. Coverage-bounded RT-HNSW design

## 3.1 Motivation

A normal ANN beam prioritizes vector distance. For a broad temporal predicate,
its best entry points may be concentrated in one part of the timeline. Other
eligible regions can receive no exploration even though they may contain exact
top-$k$ neighbors. Coverage-bounded RT-HNSW tests whether mandatory regional
participation corrects this failure.

The underlying flat graph is unchanged. Coverage is implemented as a compact
routing overlay and a separate scheduling phase.

## 3.2 Temporal block metadata

Chronological local IDs are divided into fixed-population blocks. With block
size 512, the 100,000-vector index contains approximately 196 blocks. Each block
stores:

- first and last local-ID offsets;
- conservative minimum and maximum timestamps;
- first and last temporal vectors;
- an approximate semantic medoid; and
- a farthest-point representative.

The default metadata occupies approximately 8 KB, excluding array headers.
Min/max values describe only a precisely defined block, not everything
transitively reachable through a cyclic HNSW node.

## 3.3 Coverage invariant

Let $B_Q$ be the blocks whose timestamp intervals intersect query component
$Q$. The mandatory phase requires

$$
\forall B_i\in B_Q,\quad expanded(B_i)\ge e,
$$

where $e$ is the configured minimum expansion count. A block can be pruned only
when its conservative bounds are provably disjoint from $Q$.

This is a **block-coverage guarantee**. It does not guarantee that every
eligible vector is examined or that the exact nearest neighbors are found.

## 3.4 Two-stage search

**Stage 1 — mandatory temporal coverage**

1. Discover every block intersecting the strict query component.
2. Remove representatives outside the exact component.
3. Score the remaining representatives against the query.
4. Select the closest representative as the block seed.
5. Perform at least $e$ block-local expansions.
6. Preserve strict filtering before every distance computation.

**Stage 2 — semantic ANN search**

All block frontiers are merged into one global candidate heap. The ordinary
strict RT-HNSW beam then spends its remaining effort according to vector
distance.

```text
query component
    → complete block discovery
    → mandatory block-local queues
    → merged frontier
    → global strict semantic beam
```

The mandatory queue is independent of ordinary HNSW termination. A semantically
promising region cannot cause another intersecting block to be skipped.

## 3.5 Expected behavior

Coverage improves robustness when flat search completely misses a relevant
region. Its cost grows with the number of intersecting blocks. Broad ranges and
larger databases therefore pay the greatest overhead. This trade-off motivates
an equal-computation evaluation rather than comparison at `efSearch` alone.

<div class="page-break"></div>

```{=openxml}
<w:p><w:r><w:br w:type="page"/></w:r></w:p>
```

# 4. Experimental results and comparative interpretation

## 4.1 Evaluation panels

Bucket HNSW and RT-HNSW were evaluated on different panels; their absolute
latencies are not directly comparable.

| Property | Bucket panel | Rolling graph panel |
|---|---:|---:|
| Database vectors | 990,000 | 100,000 |
| Queries | 9,600 | 900 |
| Dimensions | 384 | 384 |
| Temporal widths | 1–720 hours | 1–720 hours |
| Predicate components | 1–4 | 1, 2, 4 |
| Ground truth | Exact eligible L2 | Exact eligible L2 |

## 4.2 Principal metrics

**Bucket HNSW:** 0.99908 Recall@100, 144.18 ms mean latency, 566.15 ms p95,
35,811 mean distance computations, and 1.790 GB HNSW storage. Narrow one-hour
queries average 350.43 ms because boundary buckets require substantial
overfetch. Fragmentation raises mean latency from 42.58 ms for one component to
251.61 ms for four.

**Flat RT-HNSW:**

| efSearch | Recall@100 | Mean / p95 latency | Computations |
|---:|---:|---:|---:|
| 128 | 0.9173 | 4.65 / 18.02 ms | 2,134 |
| 256 | 0.9515 | 7.57 / 33.86 ms | 2,742 |
| 512 | 0.9741 | 12.96 / 61.82 ms | 3,825 |
| 2,048 | 0.9946 | 41.44 / 211.75 ms | 8,352 |

At ef256, recall is 0.8699 for 168-hour and 0.8873 for 720-hour queries. Narrow
queries generally use exact fallback and achieve recall 1.0. Custom graph
storage is approximately 15.6 MB excluding vectors.

## 4.3 Controlled flat-versus-coverage result

| Metric at ef256 | Flat | Coverage-bounded | Change |
|---|---:|---:|---:|
| Recall@100 | 0.95143 | 0.95273 | +0.00130 |
| Graph-query recall | 0.87511 | 0.87846 | +0.00334 |
| Distance computations | 2,741.9 | 2,979.0 | +8.65% |
| Semantic edges examined | 3,914 | 4,624 | +18.14% |
| Ground-truth block coverage | 99.924% | 100% | +0.076% |

At an approximately equal computation budget, flat ef256 achieves 0.95143
recall with 2,741.9 computations, while coverage ef205 achieves only 0.94317
with 2,746.8 computations. Coverage therefore fails the equal-budget promotion
criterion.

Only 3 of 350 graph-routed flat queries missed a block containing an exact
top-100 neighbor. Coverage improved all three, but across all 900 queries only
75 improved, 801 tied, and 24 regressed. The dominant residual error is
within-region navigation, not failure to enter a temporal region.

## 4.4 Comparative conclusion

Bucket HNSW is the near-exact, high-cost design. Flat RT-HNSW is the strongest
single-graph quality-cost compromise. Coverage-bounded RT-HNSW provides a valid
fairness and diagnostic invariant, but should be selective rather than the
default query policy.

<div class="page-break"></div>

```{=openxml}
<w:p><w:r><w:br w:type="page"/></w:r></w:p>
```

# 5. Streaming implementation and final recommendation

## 5.1 Ordered Kafka update architecture

Parallel Kafka partitions preserve arrival order only within each partition;
they do not preserve global event-time order. RT-HNSW therefore uses an
explicit ordering stage:

```text
raw topic (6 partitions)
    → validation and parallel embedding
    → embedded topic (6 partitions)
    → event-time watermark orderer
    → canonical ordered topic (1 partition)
    → single durable RT-HNSW writer
    → immutable generation and atomic CURRENT
    → live snapshot query service
```

The orderer buffers records by `(event_time_ms, tie_key)` and emits them only
when a global watermark proves the order safe. It assigns contiguous
`index_sequence` values. Events behind the committed watermark are routed to a
late topic rather than silently inserted out of order.

## 5.2 Durable publication sequence

For every micro-batch, the writer:

1. validates sequence continuity, event-time order, vector dimensions, finite
   values, embedding model, and content hash;
2. writes and `fsync`s an append WAL;
3. updates semantic, protected temporal, range, reciprocal, and gateway state;
4. persists FAISS construction state and a checksummed staged generation;
5. atomically replaces the `CURRENT` generation pointer; and
6. publishes a checkpoint before committing the Kafka offset.

If Kafka redelivers a published batch, sequence and content hash make replay
idempotent. Query processes pin one immutable generation for the full request:
existing queries finish on generation $N$, while new queries switch to $N+1$.

## 5.3 Final decision sequence

```text
eligible population ≤ 5,000
    → exact timestamp pre-filter and exact scan

broad range requiring near-exact recall
    → density-balanced bucket HNSW

standard filtered ANN query
    → flat multi-scale RT-HNSW

predicted missed-region or temporal-fairness risk
    → selectively enable coverage-bounded search
```

The default third design should remain flat RT-HNSW with 20 semantic edges,
protected predecessor/successor links, two left/right edges at each temporal
scale, exact threshold 5,000, and initial `efSearch=256`.

## 5.4 Novelty and limitations

The project contribution combines protected chronology, independent
multi-scale budgets, strict pre-distance filtering, exact fallback, a tested
coverage invariant, and restart-safe Kafka continuation. It does not claim to
originate filtered graph ANNS generally.

The main limitations are separate evaluation panels, Python traversal, the
100,000-vector RT study, and incomplete broker failure testing. Priority work is
a unified 990,000-vector evaluation, fused C++ traversal, append-only storage,
and a fixed-budget selective coverage trigger.

**Final conclusion:** flat multi-scale RT-HNSW is the principal project outcome:
it is strict, compact, tunable, and compatible with ordered streaming updates.
Bucket HNSW is the high-recall alternative; coverage-bounded RT-HNSW is a
selective diagnostic, not the default policy.
