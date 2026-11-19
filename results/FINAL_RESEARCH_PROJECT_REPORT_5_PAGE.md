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

Each database record contains embedding **xᵢ**, timestamp **tᵢ**, and stable
identifier **idᵢ**. Given query vector **q**, result count **k = 100**, and
interval set **P**, the eligibility condition is:

> **Eligible(i) ⇔ there exists [aⱼ, bⱼ) in P such that aⱼ ≤ tᵢ < bⱼ.**

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

# 2. RT-HNSW and the range-aware extension

## 2.1 Base RT-HNSW organization

RT-HNSW is one graph over all vectors; temporal buckets are not separate ANN
indexes. Before construction, source IDs are stably sorted by timestamp. The
resulting local IDs satisfy the ordering invariant:

> **t(v₀) ≤ t(v₁) ≤ ⋯ ≤ t(v[N−1]).**

Therefore, binary search converts every half-open time interval into a contiguous
local-ID range.

During insertion, a temporary FAISS HNSW searches only prior vectors and
returns up to 64 semantic candidates. Bounded reciprocal lists retain the
closest candidates. The new vector and its predecessor receive protected
backbone links, and every 512th vector becomes a query gateway. FAISS is used
only for construction; queries traverse the persisted custom adjacency.

## 2.2 Range-aware multi-scale topology

The base design primarily connects semantic neighbors and can lose useful edges
when a narrow filter removes their destinations. The final range-aware variant
reallocates the nominal degree budget as follows:

| Edge class | Budget | Function |
|---|---:|---|
| Semantic | 20 | Global vector-space proximity |
| 24-hour left/right | 2 + 2 | Short-range temporal compatibility |
| 168-hour left/right | 2 + 2 | Weekly-scale compatibility |
| 720-hour left/right | 2 + 2 | Broad-range compatibility |
| Protected predecessor/successor | Separate | Guaranteed temporal backbone |

At scale **W**, candidates combine semantic proposals, up to 64 evenly sampled
prior vectors inside **W**, and 16 immediate predecessors. They are ranked by:

> **score(u, vᵢ) = d(u, vᵢ) ÷ [max(0, 1 − (tᵢ − tᵤ) ÷ W) + ε].**

The best two become left edges from **vᵢ** and reciprocal right edges on older
nodes. Separate scale and direction arrays prevent broad or future links from
evicting short-range historical links.

## 2.3 Strict query traversal

Binary search maps each normalized predicate component to
`[local_left, local_right)`. Up to 5,000 eligible vectors use exact search;
otherwise, the closest four of at most 16 in-range gateways seed a minimum
candidate heap and a maximum result heap of size `efSearch`. Every semantic,
temporal, or range destination first passes the local-ID boundary test:

```text
if neighbor < local_left or neighbor >= local_right: reject
else if unseen: compute distance and update the beam
```

## 2.4 Working sequence and design decisions

```text
predicate → local-ID components → exact fallback or gateway seeds
          → strict semantic + temporal + range beam → exact rerank
```

The closest candidate is expanded first; termination occurs when it is farther
than the worst retained beam element. Disjoint components receive independent
beams before deduplication.

**Design decisions:** (1) chronological IDs make time ranges searchable by
binary search; (2) construction-only FAISS supplies mature semantic candidates;
(3) protected temporal links preserve interval connectivity; (4) independent
scale/direction budgets prevent edge-class eviction; and (5) exact fallback
avoids approximation when at most 5,000 vectors are eligible.

The topology improves filtered edge survival but not exact recall. Very late
records require rebuild, repair, or a delta index to preserve chronological IDs.

<div class="page-break"></div>

```{=openxml}
<w:p><w:r><w:br w:type="page"/></w:r></w:p>
```

# 3. Coverage-bounded range-aware RT-HNSW

## 3.1 Purpose and metadata

Range-aware edges improve the chance of navigating inside a filter, but the
single global beam can still concentrate on a few semantically attractive time
regions. The coverage-bounded variant leaves the graph unchanged and adds a
routing overlay that requires every intersecting region to participate.

Chronological IDs form 512-vector blocks (about 196 blocks at 100,000 vectors).
Each block stores ID offsets, conservative timestamp bounds, and four
representatives selected from up to 64 samples: first, last, centroid-nearest,
and farthest-point vectors. The overlay is approximately 8 KB. Bounds describe
a precise block; transitive metadata would expand toward the whole cyclic graph
and provide little pruning value.

## 3.2 Coverage invariant

For query component **Q**, let **B_Q** contain every block whose conservative
interval intersects **Q**. A block is removed only when its maximum timestamp is
below the query start or its minimum is at/above the query end. The mandatory
phase enforces the invariant:

> **For every block Bᵢ in B_Q: expanded(Bᵢ) ≥ e.**

Here, **e** is the configured block-local expansion count. This guarantees
**temporal block coverage**, not examination of every vector and not exact
nearest-neighbor recall.

## 3.3 Two-stage search procedure

**Coverage stage.** For every intersecting block, discard out-of-component
representatives, add strict boundaries for a partial block, score up to **g**
representatives, and expand the closest seed at least **e** times. Invalid
neighbors are rejected; cross-block edges are deferred.

**ANN stage.** Merge local frontiers into one candidate heap, then resume normal
strict range-aware search with the configured `efSearch` capacity.

```text
component → intersecting blocks → representative scoring
          → mandatory local expansions → merged global frontier
          → strict range-aware beam → exact rerank
```

The mandatory queue cannot be stopped by the ordinary HNSW termination rule.
For **R** intersecting blocks, **g** representatives, graph degree **M**,
mandatory effort **e**, vector dimension **d**, and **E** later ANN expansions,
a practical cost model is:

> **Query work = O(R·g·d + R·e·M·d + E·M·d).**

This explains the principal trade-off: coverage is deterministic, but its cost
grows with temporal width and database size. The experiment therefore evaluates
both same-`efSearch` quality and quality at matched distance computations.

## 3.4 Working example and design decisions

If a query overlaps **B₃–B₇**, all five blocks score valid representatives and
expand one seed. Only after merging those five frontiers may the global beam
concentrate on the most promising region.

| Design decision | Rationale |
|---|---|
| Sidecar overlay, unchanged graph | Isolate the effect of scheduling from topology changes |
| Fixed-population blocks | Bound metadata per block despite bursty tweet density |
| Conservative min/max bounds | Permit extra work but never false-negative pruning |
| Boundary plus diverse representatives | Represent temporal extremes and different semantic regions |
| Separate mandatory and ANN queues | Prevent ordinary beam termination from skipping a block |
| Strict in-range expansion | Preserve the project's no-out-of-range-scoring guarantee |

Coverage is a regional-participation contract, not exhaustive search; it is an
optional policy rather than a replacement for the flat graph.

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
existing queries finish on generation **N**, while new queries switch to
**N + 1**.

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
