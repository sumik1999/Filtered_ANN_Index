---
title: "Temporal-Filter-Aware Approximate Nearest-Neighbor Search"
subtitle: "Five-Page Research Project Submission"
author: "Database Project — Final Submission"
date: "19 November 2026"
lang: en-US
---

# 1. Research problem and contribution

## Abstract

This project investigates approximate nearest-neighbor search over Twitter
embeddings subject to one or more timestamp intervals. Ordinary HNSW is arranged
by vector similarity and can spend most of its search budget on records that do
not satisfy a selective temporal predicate. Three filter-aware designs are
studied: **density-balanced bucket HNSW**, **flat multi-scale rolling temporal
HNSW (RT-HNSW)**, and **coverage-aware RT-HNSW**. Bucket HNSW attains 0.9991
Recall@100 on a 990,000-vector panel, but incurs 144.18 ms mean and 566.15 ms p95
latency. Flat RT-HNSW uses one chronological graph and reaches approximately
0.9515 recall at `efSearch=256` on a separate 100,000-vector panel, with strict
pre-distance filtering and exact fallback. Coverage-aware search guarantees
that every intersecting temporal block is explored and raises same-`efSearch`
recall from 0.95143 to 0.95273, but loses at equal computation. The recommended
research design is therefore flat multi-scale RT-HNSW, updated through an
ordered Kafka pipeline and published as immutable query snapshots. Bucket HNSW
remains the high-recall path; coverage is retained only for diagnosed tail-risk
queries.

## Problem definition

For database $D=\{(x_i,t_i,id_i)\}$ and intervals
$P=\{[a_j,b_j)\}$, the eligible set is

$$D_P=\{x_i:\exists[a_j,b_j)\in P,\ a_j\le t_i<b_j\}.$$

The query returns the $k=100$ eligible vectors minimizing squared L2 distance
to query vector $q$. Correctness requires strict half-open filtering; an
ineligible RT-HNSW destination is rejected **before** reading its vector or
computing distance.

## Research questions

1. Can temporal partitioning provide near-exact recall, and at what boundary
   and fragmentation cost?
2. Can one chronologically constructed graph provide a useful strict-filtered
   recall/cost trade-off?
3. Does mandatory temporal-region coverage improve that graph under an equal
   computation budget?
4. Can chronological construction be continued safely from an online stream?

## Main project contributions

- Density-balanced temporal HNSW routing with safe boundaries and global rerank.
- A single flat graph combining 20 semantic edges, protected chronological
  links, and independent 24/168/720-hour range-edge budgets.
- Exact fallback at 5,000 eligible vectors and strict row-access auditing.
- A formal block-coverage extension and an equal-budget negative result.
- Ordered Kafka ingestion with WAL recovery, idempotent replay, immutable
  generations, and live query snapshot pinning.

<div class="page-break"></div>

```{=openxml}
<w:p><w:r><w:br w:type="page"/></w:r></w:p>
```

# 2. Index designs and query sequence

## 2.1 Density-balanced bucket HNSW

The 990,000-vector timeline is divided into 64 approximately equal-population
buckets; each owns an independent FAISS HNSW (`M=32`, `efConstruction=200`).
Density balancing avoids extreme index-size differences caused by bursty tweet
traffic.

**Query sequence**

```text
normalize interval union
 → route to intersecting buckets
 → fully covered bucket: ordinary HNSW
 → partial boundary bucket: overfetch + exact timestamp filter
 → merge and deduplicate global IDs
 → exact squared-L2 rerank
```

The design maximizes recall, but a narrow interval may occupy only a small
fraction of a large bucket. It must then retrieve many candidates to obtain 100
eligible survivors. Multiple OR intervals add boundary buckets and index fanout.

## 2.2 Flat multi-scale RT-HNSW

Vectors are inserted in stable chronological order, making a temporal interval
contiguous in local-ID space. FAISS HNSW is used only to find semantic
construction candidates; queries traverse custom persisted adjacency.

| Edge class | Per-node budget |
|---|---:|
| Semantic | 20 |
| 24-hour left/right | 2 + 2 |
| 168-hour left/right | 2 + 2 |
| 720-hour left/right | 2 + 2 |
| Protected predecessor/successor | Separate |

Range edges use vector proximity adjusted by temporal co-eligibility. Separate
budgets prevent one temporal scale from evicting links useful to another.

**Query sequence**

```text
normalize intervals and binary-search timestamp components
 → eligible count ≤ 5,000: exact eligible scan
 → otherwise select in-range periodic entries
 → strict beam over semantic + temporal + range edges
 → reject invalid destination before vector distance
 → independently search each disjoint component
 → deduplicate and exact-rerank retained candidates
```

## 2.3 Coverage-aware RT-HNSW

The flat graph is unchanged. Local IDs are grouped into 512-vector blocks. Each
block stores conservative min/max timestamps and four representatives: first,
last, approximate medoid, and farthest-point sample.

**Query sequence**

```text
discover every intersecting block
 → score block representatives
 → mandatory block-local expansion
 → merge all block frontiers
 → run the ordinary strict global beam
```

If $B_Q$ is the set of blocks intersecting query component $Q$, the mandatory
phase enforces that every block in $B_Q$ receives an expansion. This guarantees
block participation—not examination of every vector and not exact recall.

<div class="page-break"></div>

```{=openxml}
<w:p><w:r><w:br w:type="page"/></w:r></w:p>
```

# 3. Experimental evaluation

## Evaluation panels

Absolute bucket and RT-HNSW latency must not be directly compared because the
methods were measured on different panels.

| Property | Full bucket panel | Rolling graph panel |
|---|---:|---:|
| Database vectors | 990,000 | 100,000 |
| Queries | 9,600 | 900 |
| Dimensions | 384 | 384 |
| Temporal widths | 1, 6, 24, 168, 720 h | Same |
| Predicate components | 1–4 | 1, 2, 4 |
| Ground truth | Exact eligible L2 | Exact eligible L2 |

## Bucket HNSW results

| Metric | Result |
|---|---:|
| Recall@100 | **0.99908** |
| Mean / p95 latency | 144.18 / 566.15 ms |
| Distance computations | 35,811 |
| Selected / partial buckets | 3.89 / 3.26 |
| Build time | 52.67 s |
| HNSW storage | 1.790 GB |

| Width | Recall | Mean latency |
|---:|---:|---:|
| 1 h | 0.99994 | 350.43 ms |
| 24 h | 0.99972 | 58.27 ms |
| 168 h | 0.99828 | 29.00 ms |
| 720 h | 0.99754 | 32.95 ms |

Fragmentation raises mean latency from 42.58 ms for one interval to 251.61 ms
for four, while recall remains above 0.998. The evidence confirms near-exact
quality but severe boundary and fanout variance.

## Flat RT-HNSW results

| efSearch | Recall | Mean / p95 latency | Computations |
|---:|---:|---:|---:|
| 128 | 0.9173 | 4.65 / 18.02 ms | 2,134 |
| 256 | 0.9515 | 7.57 / 33.86 ms | 2,742 |
| 512 | 0.9741 | 12.96 / 61.82 ms | 3,825 |
| 2,048 | 0.9946 | 41.44 / 211.75 ms | 8,352 |

At ef256, broad recall is 0.8699 for 168 hours and 0.8873 for 720 hours. Narrow
queries generally use exact fallback and reach recall 1.0. Custom graph storage
is approximately 15.6 MB excluding vectors; build time is 133.1 seconds.

## Interpretation

Bucket HNSW is the quality-oriented method. Flat RT-HNSW is the strongest
single-graph method and exposes a continuous beam-width trade-off. Its main
remaining weakness is broad-range recall, not temporal correctness.

<div class="page-break"></div>

```{=openxml}
<w:p><w:r><w:br w:type="page"/></w:r></w:p>
```

# 4. Coverage result and online streaming design

## Coverage-aware result

| Metric at ef256 | Flat | Coverage | Change |
|---|---:|---:|---:|
| Recall@100 | 0.95143 | 0.95273 | +0.00130 |
| Graph-query recall | 0.87511 | 0.87846 | +0.00334 |
| Distance computations | 2,741.9 | 2,979.0 | +8.65% |
| Semantic edges examined | 3,914 | 4,624 | +18.14% |
| Ground-truth block coverage | 99.924% | 100% | +0.076% |

At matched computation, flat ef256 obtains 0.95143 recall with 2,741.9
computations, whereas coverage ef205 obtains only 0.94317 with 2,746.8.
Coverage therefore fails the equal-budget acceptance test.

Only 3 of 350 graph-routed baseline queries missed any block containing an
exact top-100 result. Coverage improved all three, but 75 of all queries
improved, 801 tied, and 24 regressed. The mechanism correctly repairs rare
missed-region failures; most errors occur inside regions that flat RT-HNSW
already visits. It should be optional rather than mandatory.

## Kafka continuation

Parallel arrival order cannot directly define chronological local IDs. The
streaming design therefore separates ingestion from a canonical ordered log.

```text
raw.v1 (6 partitions)
 → parallel validation and embedding
 → embedded.v1 (6 partitions)
 → event-time watermark orderer
 → ordered.v1 (1 partition, contiguous index_sequence)
 → single durable RT-HNSW writer
 → immutable generation + atomic CURRENT
 → live snapshot query service
```

The orderer emits by `(event_time_ms, tie_key)` only after the global watermark
is safe. Late records are quarantined; they are never silently appended behind
the committed watermark.

**Micro-batch durability sequence**

1. Poll without committing the Kafka offset.
2. Validate contiguous sequence, ordering key, vector, model, and content hash.
3. Write and `fsync` an append WAL.
4. Update semantic, temporal, multi-scale, reciprocal, and gateway state.
5. Persist FAISS construction state and a checksummed staged generation.
6. Atomically replace `CURRENT`.
7. Publish checkpoint and commit Kafka offset.

Redelivery after publication is safe because sequence and content hash are
verified. Query processes pin one immutable generation; old queries finish on
$N$ while new queries switch to $N+1$.

Implemented components include the transactional orderer, durable flat writer,
WAL recovery, idempotent replay, immutable generations, live refresh, reader
leases, and freshness waits. Broker-level process-kill tests, append-only
storage, production embedding, and late-event delta indexes remain future work.

<div class="page-break"></div>

```{=openxml}
<w:p><w:r><w:br w:type="page"/></w:r></w:p>
```

# 5. Critical findings and final decision

## Research findings

1. **Temporal partitioning is effective but costly.** Density-balanced bucket
   HNSW gives near-exact recall, validating pre-search routing, but boundary
   overfetch reverses the expected narrow-query advantage and fragmentation
   produces high fanout.
2. **A single temporal graph is viable.** Flat RT-HNSW preserves strict
   correctness, compact adjacency, exact selective search, and tunable broad
   approximation without maintaining 64 indexes.
3. **More coverage is not automatically better ANN.** Mandatory block coverage
   gives a formal invariant and slightly tighter recall distribution, but its
   same-`efSearch` gain disappears under equal computation.
4. **The key residual problem is within-region navigation.** Baseline traversal
   already covers 99.805% of ground-truth blocks on graph queries.
5. **Online construction requires explicit ordering and snapshot publication.**
   Kafka partition order alone cannot preserve event-time local IDs.

## Bounded novelty

The project does not claim to originate filtered graph search. Its contribution
is the implemented combination of protected chronological connectivity,
independent multi-scale range budgets, co-eligibility edge selection, strict
pre-distance filtering, exact fallback, a formally tested coverage scheduler,
and restart-safe Kafka continuation. The negative coverage result is itself a
research contribution because it prevents adding complexity that does not
survive an equal-budget test.

## Final architecture recommendation

```text
small eligible set
 → exact timestamp pre-filter and scan

broad range requiring near-exact quality
 → density-balanced bucket HNSW

normal filtered ANN with tunable cost
 → flat multi-scale RT-HNSW

predicted missed-region or fairness risk
 → selectively enable coverage mode
```

The default third design should remain **flat multi-scale RT-HNSW** with
`efSearch=256`, exact threshold 5,000, 20 semantic edges, and two left/right
range edges at 24, 168, and 720 hours. Its online form should consume one
canonical Kafka sequence and publish immutable snapshots.

## Limitations and immediate future work

- Run all methods on one identical 990,000-vector workload.
- Replace Python traversal with fused C++/pybind11.
- Replace full-generation rewrites with append-only chunks or copy-on-write
  pages.
- Run Kafka transaction-abort and process-kill integration tests.
- Add a late-event delta index and rolling compaction.
- Test a selective coverage trigger under a fixed computation budget.

## Project assessment

| Criterion | Result |
|---|---:|
| Technical depth | 18/20 |
| Experimental rigor | 17/20 |
| Correctness and validation | 14/15 |
| Novelty and analysis | 13/15 |
| Reproducibility | 9/10 |
| Production readiness | 3/5 |
| Other formulation/documentation criteria | 14/15 |
| **Transparent self-assessment** | **88/100** |

**Conclusion:** Flat multi-scale RT-HNSW is the strongest project outcome: it is
strict, compact, tunable, and compatible with durable ordered updates. Bucket
HNSW remains the high-recall alternative, while coverage-aware RT-HNSW is a
valuable diagnostic and selective tail policy rather than the default search
mechanism.

## Selected references

1. Malkov and Yashunin, “Efficient and Robust Approximate Nearest Neighbor
   Search Using HNSW,” *IEEE TPAMI*, 2020.
2. Zuo et al., “SeRF: Segment Graph for Range-Filtering ANNS,” *PACM*, 2024.
3. Engels et al., “Approximate Nearest Neighbor Search with Window Filters,”
   2024.
4. Peng et al., “Dynamic Range-Filtering Approximate Nearest Neighbor Search,”
   *PVLDB*, 2025.
