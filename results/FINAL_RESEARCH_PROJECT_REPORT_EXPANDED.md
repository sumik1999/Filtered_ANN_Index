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

## 3.1 The design in plain language

RT-HNSW keeps **one graph**, not one index per time bucket. Its local IDs are
stable chronological addresses: source records are sorted by `(timestamp,
source ID)` before insertion, so

> **t(v₀) ≤ t(v₁) ≤ ⋯ ≤ t(v[N−1]).**

A time predicate can therefore be converted by binary search into one or more
contiguous ID ranges. Inside those ranges, the graph needs different kinds of
“roads”: semantic roads toward similar tweets, temporal roads that keep nearby
times connected, and sparse entry points from which a query can begin.

Construction processes vectors in this chronological order. A temporary FAISS
HNSW searches only already inserted vectors and proposes up to 64 semantic
neighbors. It helps build the custom graph but is **not** queried afterward.

## 3.2 Edge types and what each one solves

| Stored edge type | Per-node budget | Meaning and purpose |
|---|---:|---|
| Semantic | 20 | Closest embedding neighbors; provides the main similarity search paths |
| Protected temporal | predecessor + successor | Immediate chronological neighbors; keeps an interval traversable even when semantic links leave it |
| 24-hour range | 2 left + 2 right | Similar nodes likely to coexist in a short time filter |
| 168-hour range | 2 left + 2 right | Weekly-scale alternatives when short links are insufficient |
| 720-hour range | 2 left + 2 right | Longer temporal jumps for broad predicates |
| Gateway shortcut | up to 8 per gateway | Semantic links between sparse periodic gateways; optional during query traversal |

**Semantic edges.** The 20 closest proposals are stored in bounded reciprocal
lists. They answer “where is a more similar vector?” but know nothing about the
query's time range.

**Protected temporal edges.** Every inserted node links backward to its
immediate predecessor, and that predecessor links forward to it. These two
slots are separate from semantic storage and cannot be evicted. They form a
chronological backbone, although following many one-step links can be slow.

**Multi-scale range edges.** The evaluated horizons are 24 hours (one day),
168 hours (one week—not 120 hours), and 720 hours (approximately 30 days).
They are rolling construction horizons, not calendar buckets and not limits on
query width. When node **vᵢ** is inserted, each scale independently considers
prior semantic proposals, sampled nodes from the preceding **W** hours, and
immediate predecessors. Candidates are ranked by:

> **score(u, vᵢ) = d(u, vᵢ) ÷ [max(0, 1 − (tᵢ − tᵤ) ÷ W) + ε].**

The score rewards semantic closeness while penalizing candidates near the edge
of the horizon. At every scale, the best two older candidates become left edges
from **vᵢ**, and bounded reciprocal links become future-facing right edges on
the older nodes. Thus 24-hour edges support short filtered intervals, 168-hour
edges provide weekly jumps, and 720-hour edges avoid long chains during broad
monthly queries. Their arrays and budgets are independent: a monthly edge cannot
evict a daily edge, a future edge, or the protected backbone. During search all
scales may be examined, but a destination outside the exact half-open predicate
is rejected before its vector distance is computed.

**Gateway shortcut edges.** Periodic gateways may also connect to up to eight
semantically nearest earlier gateways, with bounded reciprocal links. This is a
stored shortcut class, distinct from using gateways as query entry points. The
controlled flat-versus-coverage result disables these optional shortcut edges,
so its difference comes from scheduling rather than extra topology.

<div class="page-break"></div>

```{=openxml}
<w:p><w:r><w:br w:type="page"/></w:r></w:p>
```

## 3.3 What a periodic gateway does

A gateway is an ordinary vector marked as a possible **starting point**; it is
not a separate index and does not bypass filtering. Every 512th chronological
node is marked, which spreads entry points through time. Gateway IDs and
timestamps are stored in sorted order.

For each graph-searched predicate component, query seeding is:

1. binary-search the gateway directory to keep only gateways inside the
   component;
2. sample at most 16 of them at evenly spaced temporal positions, preventing a
   dense sub-period from supplying every candidate;
3. compute distance from **q** only for these valid candidates; and
4. place the closest four into the candidate heap and retained-result heap.

Thus “16 gateway candidates” means at most 16 scored possible entries, while
“four gateway seeds” means the four that actually start traversal. If a short
component contains no periodic gateway, the implementation performs exact
component search rather than entering through an out-of-range node.

## 3.4 Strict traversal after seeding

The closest heap item is expanded first. For each of its enabled semantic,
range, protected-temporal, and optional gateway-shortcut edges, the destination
must pass the component boundary test before its vector is read:

```text
if neighbor < local_left or neighbor >= local_right: reject without scoring
else if unseen: score it and update the efSearch beam
```

Search stops when the closest unexpanded candidate is farther than the worst
item retained in the full `efSearch` beam. Distances are exact for examined
vectors, but the set reached through the graph is approximate. Disjoint
predicate components receive independent beams; their candidates are then
merged, deduplicated, and reranked.

## 3.5 Working query from predicate to result

Use the same illustrative **q**, `k=3`, and predicate **P₁ ∪ P₂**. Binary search
maps it to `[28,700, 32,350)` and `[35,120, 37,600)`, containing 6,130 vectors,
so the 5,000-vector exact fallback does not apply.

1. **P₁** contains seven periodic gateways and **P₂** contains five; all are
   eligible candidates because each count is below 16. Their exact distances
   are computed, and up to four per component become seeds.
2. Expanding gateway `v₂₉₁₈₃` follows a 168-hour left edge to `v₂₉₀₄₄` and
   semantic edges inside **P₁**. Gateway `v₃₅₈₃₉` reaches `v₃₆₁₀₄` in **P₂**.
3. An edge to `v₂₇₉₉₀` fails the ID-range test and is rejected before vector
   access. Other valid destinations compete by query distance in the beam.
4. Termination and final reranking return `(v₃₆₁₀₄, 0.08)`,
   `(v₂₉₀₄₄, 0.11)`, and `(v₃₅₄₁₂, 0.14)`.

The illustrative flat search misses `(v₃₁₉₈₀, 0.06)` because its region never
enters the beam. This is an ANN navigation miss, not a temporal-filter error.
Chronological IDs, protected edges, separate edge budgets, and valid gateway
seeds make search robust; they do not make it exact. Late records still require
repair, rebuild, or a delta index to preserve chronological addressing.

<div class="page-break"></div>

```{=openxml}
<w:p><w:r><w:br w:type="page"/></w:r></w:p>
```

# 4. Coverage-bounded range-aware RT-HNSW

## 4.1 What changes—and what does not

The third design keeps the **same graph and edge types**: it adds no block edges
and duplicates no vectors. It changes entry scheduling because one global beam
may concentrate on a few attractive regions. Chronological IDs are divided into
512-vector blocks with conservative time bounds and up to four **coverage
representatives**. These block gateways differ from periodic gateways:

| Entry concept | Scope | Selection | Role |
|---|---|---|---|
| Periodic gateway | One every 512 global IDs | Fixed chronological position | Supplies a few entries for ordinary flat search |
| Coverage representative | Up to four inside every block | First, last, centroid-nearest, and farthest-point samples | Forces every intersecting block to supply a local entry |

First/last cover temporal extremes, centroid-nearest covers the semantic center,
and farthest-point adds diversity. Partial blocks use strict boundary IDs when
stored representatives lie outside the predicate. The 100k sidecar is about
8 KB.

## 4.2 Coverage guarantee

For component **Q**, let **B_Q** be every block whose conservative time interval
intersects **Q**. A block is pruned only when its maximum timestamp is before
**Q**, or its minimum is at/after the half-open end. The mandatory phase enforces:

> **For every block Bᵢ in B_Q: expanded(Bᵢ) ≥ e.**

This means each eligible region receives work. It does **not** mean that every
vector is examined, that all graph paths are followed, or that recall is 1.0.

## 4.3 How block gateways use the graph edges

Each block filters representatives to the strict component, scores at most
four, selects the nearest seed, and expands it through existing semantic,
multi-scale range, and protected temporal edges. During this phase, a destination
must remain inside both the component **and its block**. Cross-block edges are
deferred so one block cannot satisfy another's obligation. Once all blocks are
covered, frontiers merge and ordinary strict traversal may follow eligible
cross-block edges. Optional periodic gateway shortcuts are disabled in both
controlled methods.

For **R** blocks, **g** representatives, graph degree **M**, mandatory effort
**e**, dimension **d**, and **E** later ANN expansions:

> **Query work = O(R·g·d + R·e·M·d + E·M·d).**

The first two terms are the explicit price of guaranteed regional participation.

## 4.4 Working query from blocks to result

For the same components `[28,700, 32,350)` and `[35,120, 37,600)`, 512-vector
partitioning identifies **B₅₆–B₆₃** and **B₆₈–B₇₃**: 14 blocks in total.

1. Conservative bounds retain all 14. Representatives outside the exact
   component are removed; boundary fallbacks keep partial blocks seedable.
2. Up to four representatives are scored in each block. The nearest becomes a
   local seed and receives at least one in-block expansion. Ordinary beam
   termination cannot stop this mandatory phase.
3. In **B₆₂**, a representative's existing range or semantic edge reaches
   `v₃₁₉₈₀`. This is the candidate missed by flat scheduling. Only after all 14
   blocks are marked covered are their frontiers merged.
4. Global strict traversal and reranking return `(v₃₁₉₈₀, 0.06)`,
   `(v₃₆₁₀₄, 0.08)`, and `(v₂₉₀₄₄, 0.11)`.

Coverage can repair a missed region, but extra seeds consume budget and a
nearest vector can remain undiscovered inside a visited block. The sidecar
separates scheduling from topology; conservative bounds and strict edge checks
prevent false-negative pruning and out-of-range scoring.

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
