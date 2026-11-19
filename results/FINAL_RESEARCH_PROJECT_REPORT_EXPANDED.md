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

# 2. Density-balanced bucket HNSW

## 2.1 Construction and purpose

Bucket HNSW restricts the temporal domain before ANN traversal. Equal-width
calendar buckets are poorly suited to bursty Twitter traffic because their
populations differ sharply. The density-balanced design instead stably sorts
all records by `(timestamp, source ID)`, chooses 64 boundaries with approximately
equal populations without splitting equal timestamps, and builds one FAISS
`IndexHNSWFlat` per bucket. Each index uses `M=32` and
`efConstruction=200`; persisted local-to-global maps recover source IDs.

The bucket directory stores each bucket's half-open timestamp bounds and row
count. Thus routing is exact even though search inside a selected bucket is
approximate.

## 2.2 Query execution

Intervals are first sorted and merged into a disjoint union. The directory then
labels every intersecting bucket as **full** or **partial**. A full bucket can be
searched directly because every member satisfies the predicate. For a partial
boundary bucket, overlap selectivity controls overfetch (`2×`, with boundary
safety `1.25`); timestamps are checked exactly after FAISS returns local IDs.
All candidates are translated to global IDs, deduplicated across interval
components, and reranked by exact query distance.

```text
predicate union → bucket directory → full/partial bucket searches
                → exact boundary filter → global-ID deduplication
                → exact candidate rerank → top k
```

The final configuration uses `efSearch=128`. Half-open checks are retained at
partial boundaries, so a record exactly at an interval end is rejected.

## 2.3 Working query example

Consider an illustrative query embedding **q**, `k=3`, and two disjoint time
components **P₁** and **P₂**. In the persisted directory, **P₁** intersects the
tail of bucket **B₁₈**, all of **B₁₉**, and the head of **B₂₀**; **P₂** cuts
through **B₂₅**.

1. Routing produces `{B₁₈ partial, B₁₉ full, B₂₀ partial, B₂₅ partial}`.
2. `B₁₉` runs ordinary HNSW. The other three request enlarged candidate sets.
3. Suppose `B₁₈` returns a close record at the exact end of **P₁**. The
   half-open test removes it before merging. A record appearing through
   overlapping routes is deduplicated by global ID.
4. Exact reranking of the remaining vectors returns
   `(v₃₁₉₈₀, 0.06)`, `(v₃₆₁₀₄, 0.08)`, and `(v₂₉₀₄₄, 0.11)`.

Distances are illustrative squared-L2 values. In the measured experiment,
`k=100`; the sequence is unchanged. The example shows how partition routing
can approach exact results without scanning all 990,000 vectors, while partial
buckets still incur extra work.

## 2.4 Design rationale and limitation

| Decision | Reason |
|---|---|
| Equal-population rather than equal-time buckets | Balance index size under bursty arrival density |
| Preserve equal-timestamp groups | Avoid ambiguous boundary ownership |
| Full versus partial routing | Avoid unnecessary filtering inside fully covered buckets |
| Exact boundary checks | Preserve authoritative half-open predicates |
| Global deduplication and reranking | Produce one correctly ordered result list across indexes |

The design achieved 0.99908 Recall@100, but narrow predicates can occupy only a
small fraction of a large bucket. Fragmented predicates also fan out to more
indexes and partial boundaries. This explains its high and variable query cost.

<div class="page-break"></div>

```{=openxml}
<w:p><w:r><w:br w:type="page"/></w:r></w:p>
```

# 3. RT-HNSW and the range-aware extension

## 3.1 Base RT-HNSW organization

RT-HNSW is one graph over all vectors; temporal buckets are not separate ANN
indexes. Before construction, source IDs are stably sorted by timestamp. The
resulting local IDs satisfy the ordering invariant:

> **t(v₀) ≤ t(v₁) ≤ ⋯ ≤ t(v[N−1]).**

Therefore, binary search converts every half-open time interval into a contiguous
local-ID range.

During insertion, construction-only FAISS searches prior vectors for up to 64
semantic candidates. Bounded reciprocal lists retain the closest; the new
vector and predecessor receive protected backbone links, and every 512th vector
becomes a query gateway.

## 3.2 Range-aware multi-scale topology

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

At scale **W**, semantic proposals, sampled prior vectors inside **W**, and
immediate predecessors are ranked by:

> **score(u, vᵢ) = d(u, vᵢ) ÷ [max(0, 1 − (tᵢ − tᵤ) ÷ W) + ε].**

The best two form left and reciprocal right edges. Separate arrays prevent
broad or future links from evicting short-range historical links.

## 3.3 Strict query traversal

Binary search maps each component to `[local_left, local_right)`. Up to 5,000
eligible vectors use exact search; otherwise, four in-range gateways seed the
candidate and `efSearch` result heaps. Every destination first passes:

```text
if neighbor < local_left or neighbor >= local_right: reject
else if unseen: compute distance and update the beam
```

## 3.4 Working query example and design decisions

Use the same illustrative **q**, `k=3`, and predicate **P₁ ∪ P₂**. Binary search
maps the components to local-ID ranges `[28,700, 32,350)` and
`[35,120, 37,600)`. Their union has 6,130 vectors, so it exceeds the 5,000-vector
exact threshold and enters graph search.

1. The gateway directory identifies at most 16 in-range candidates; the four
   closest valid gateways seed each component's beam.
2. Expanding gateway `v₂₈₈₀₀` follows a 168-hour range edge to `v₂₉₀₄₄` and
   semantic links within **P₁**. Gateway `v₃₅₈₄₀` reaches `v₃₆₁₀₄` in **P₂**.
3. An edge to `v₂₇₉₉₀` fails the local-ID test and is rejected **before** its
   vector row is read. Valid candidates update the `efSearch=256` beam.
4. When the next candidate is farther than the worst retained beam element,
   traversal stops. Exact reranking of visited eligible candidates returns
   `(v₃₆₁₀₄, 0.08)`, `(v₂₉₀₄₄, 0.11)`, and `(v₃₅₄₁₂, 0.14)`.

This illustrative run misses `(v₃₁₉₈₀, 0.06)` because no explored path reaches
its temporal region; that is approximation, not a filter error. Disjoint
components use independent beams before global deduplication and reranking.

**Rationale:** chronological IDs enable binary-search filtering; construction
FAISS supplies semantic candidates; protected links preserve connectivity;
separate budgets prevent edge-class eviction; and exact fallback removes small-
range approximation. Late records require repair, rebuild, or a delta index.

<div class="page-break"></div>

```{=openxml}
<w:p><w:r><w:br w:type="page"/></w:r></w:p>
```

# 4. Coverage-bounded range-aware RT-HNSW

## 4.1 Purpose and metadata

A global beam can concentrate on a few attractive time regions. Coverage leaves
the graph unchanged but requires every intersecting region to participate.
Chronological IDs form 512-vector blocks; each stores ID offsets, conservative
time bounds, and four sampled representatives: first, last, centroid-nearest,
and farthest-point. The 100k overlay is approximately 8 KB.

## 4.2 Coverage invariant

For query component **Q**, let **B_Q** contain every block whose conservative
interval intersects **Q**. A block is removed only when its maximum timestamp is
below the query start or its minimum is at/above the query end. The mandatory
phase enforces the invariant:

> **For every block Bᵢ in B_Q: expanded(Bᵢ) ≥ e.**

Here, **e** is the configured block-local expansion count. This guarantees
**temporal block coverage**, not examination of every vector and not exact
nearest-neighbor recall.

## 4.3 Two-stage search procedure

**Coverage stage.** For every intersecting block, discard out-of-component
representatives, add strict boundaries for a partial block, score up to **g**
representatives, and expand the closest seed at least **e** times. Invalid
neighbors are rejected; cross-block edges are deferred.

**ANN stage.** Merge local frontiers into one candidate heap, then resume normal
strict range-aware search with the configured `efSearch` capacity.

The mandatory queue ignores ordinary HNSW termination. For **R** blocks, **g**
representatives, degree **M**, mandatory effort **e**, dimension **d**, and
**E** later ANN expansions, work is:

> **Query work = O(R·g·d + R·e·M·d + E·M·d).**

Coverage is deterministic, but cost grows with temporal width; therefore it is
also tested at matched distance computations.

## 4.4 Working query example and design decisions

Apply coverage scheduling to the same **q**, `k=3`, and ID components
`[28,700, 32,350)` and `[35,120, 37,600)`. With 512-vector blocks, they
intersect blocks **B₅₆–B₆₃** and **B₆₈–B₇₃**.

1. Conservative min/max tests retain all 14 intersecting blocks. Boundary
   representatives outside the two components are discarded.
2. Up to four valid representatives per block are scored. The nearest seed in
   every block receives at least one local expansion; cross-block edges are
   deferred during this mandatory phase.
3. Expansion in **B₆₂** discovers `v₃₁₉₈₀`, the candidate that the ordinary
   global beam missed. Only after every retained block records one expansion are
   all local frontiers merged.
4. The normal strict beam resumes and exact reranking returns
   `(v₃₁₉₈₀, 0.06)`, `(v₃₆₁₀₄, 0.08)`, and `(v₂₉₀₄₄, 0.11)`.

The example explains the mechanism, not a promise of improvement: coverage
ensures regional participation, while neighbors remain approximate.

**Rationale.** The sidecar isolates scheduling from topology. Fixed-population
blocks bound metadata; conservative bounds cannot prune an eligible block.
Diverse representatives and separate queues prevent premature regional loss,
and strict expansion preserves the no-out-of-range-scoring rule. Coverage is
not exhaustive search.

<div class="page-break"></div>

```{=openxml}
<w:p><w:r><w:br w:type="page"/></w:r></w:p>
```

# 5. Experimental results and comparative interpretation

## 5.1 Evaluation panels

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

## 5.2 Principal metrics

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

## 5.3 Controlled flat-versus-coverage result

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

## 5.4 Comparative conclusion

Bucket HNSW is the near-exact, high-cost design. Flat RT-HNSW is the strongest
single-graph quality-cost compromise. Coverage-bounded RT-HNSW provides a valid
fairness and diagnostic invariant, but should be selective rather than the
default query policy.

<div class="page-break"></div>

```{=openxml}
<w:p><w:r><w:br w:type="page"/></w:r></w:p>
```

# 6. Streaming implementation and final recommendation

## 6.1 Ordered Kafka update architecture

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

## 6.2 Durable publication sequence

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

## 6.3 Final decision sequence

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

## 6.4 Novelty and limitations

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
