# Detailed report: four temporal HNSW architectures

## Executive summary

This project evaluates four progressively more filter-aware ways to answer:

```text
top-k nearest vectors subject to timestamp belonging to one or more intervals
```

The four architectures are:

1. **Global HNSW post-filtering** — search one ordinary global HNSW and reject timestamp-invalid results afterward.
2. **Bucket HNSW** — partition the timeline and search only HNSW indexes whose buckets intersect the predicate.
3. **Flat multi-scale RT-HNSW** — maintain one chronological graph with semantic, protected temporal and range-aware edges, then traverse strictly inside the predicate.
4. **Packed temporal-bitmap RT-HNSW** — separate historical/future insertion edges, pack adjacency by epoch, use conservative epoch summaries, route through epoch-assigned portals, and continue construction from Kafka.

The main result is not one universally best method:

- Global post-filtering is fast but loses most filtered neighbours on selective ranges.
- Bucket HNSW delivers nearly exact recall but has high boundary and fragmentation cost.
- Flat RT-HNSW provides the best tested single-graph quality/latency trade-off and remains the recommended research topology.
- The bitmap version safely removes most invalid edge decoding, but its Python/vectorized implementation is slower and its fixed directional degree allocation lowers recall.

## Critical comparability warning

The four methods were not all measured on one identical corpus and workload. Results must be interpreted in two panels.

### Panel A: full Twitter baseline

| Property | Value |
|---|---|
| Database | 990,000 DLT-Tweets vectors |
| Queries | 9,600 balanced queries |
| Dimensions | 384 normalized MiniLM dimensions |
| Predicate widths | 1, 6, 24, 168 and 720 hours |
| Fragmentation | 1–4 OR predicates |
| Ground truth | Exact timestamp pre-filter + exact squared L2 |
| Methods | Global post-filter and density-64 bucket HNSW |

### Panel B: rolling graph experiments

| Property | Value |
|---|---|
| Database | Earliest 100,000 DLT-Tweets vectors |
| Queries | 900 matched queries |
| Dimensions | 384 normalized MiniLM dimensions |
| Predicate widths | 1, 6, 24, 168 and 720 hours |
| Fragmentation | 1, 2 and 4 intervals |
| Exact fallback | At most 5,000 eligible vectors |
| Methods | Flat RT-HNSW and temporal-bitmap RT-HNSW |

Latency, recall and storage numbers should be compared directly only within a panel. A final publication-quality comparison requires all four methods on the same database, workload, implementation language and hardware conditions.

---

# 1. Global HNSW post-filtering

## Design

The index is a conventional global FAISS `IndexHNSWFlat`:

```text
query vector
   ↓
global HNSW search
   ↓
retrieve k × overfetch candidates
   ↓
exact timestamp filtering
   ↓
return surviving top-k
```

The graph knows nothing about timestamps. Timestamp arrays are external metadata joined using the FAISS vector ID.

Representative configuration:

```text
M = 32
efConstruction = 200
efSearch = 256
overfetch = 10
requested k = 100
retrieved candidates = 1,000
```

## Full-data metrics

| Metric | Result |
|---|---:|
| Recall@100 | 0.2347 |
| Mean latency | 3.71 ms |
| p50 latency | 3.86 ms |
| p95 latency | 5.07 ms |
| Candidates returned before filtering | 1,000 |
| Valid candidates after filtering | 29.95 |
| Distance computations | 16,748 |
| Complete-result rate | 0.0833 |
| Build time | 288.6 seconds |
| Index size | 1.790 GB |

A weaker but cheaper `efSearch=128`, overfetch-2 configuration obtains only 0.0696 recall at 0.93 ms.

## Recall by temporal coverage

| Coverage | Recall | Mean latency |
|---:|---:|---:|
| 1 hour | 0.0070 | 3.69 ms |
| 6 hours | 0.0276 | 3.68 ms |
| 24 hours | 0.0669 | 3.74 ms |
| 168 hours | 0.2750 | 3.73 ms |
| 720 hours | 0.7971 | 3.73 ms |

## Why it fails for selective ranges

If selectivity is `s`, only approximately `s × L` of `L` globally retrieved candidates survive filtering. A one-hour query can have selectivity below 0.03%; retrieving 1,000 candidates may leave fewer than one useful record in ordinary traffic.

Increasing `efSearch` alone does not solve this. The final candidate count is controlled mainly by overfetch, and sufficiently large overfetch approaches a global scan.

## Advantages

- Simplest architecture.
- Uses a mature optimized FAISS implementation.
- Best latency when low filtered recall is acceptable.
- Strong for unfiltered and very broad predicates.
- Standard HNSW incremental insertion is available.
- No temporal partition routing or result merge across indexes.

## Shortcomings

- Filtering is disconnected from graph navigation.
- Narrow intervals frequently return fewer than `k` valid results.
- Fixed overfetch cannot handle widely varying selectivity.
- Fragmented predicates do not guide candidate discovery.
- A global semantic neighbour may be useless for every narrow temporal query.

## Appropriate use

Use global post-filtering when the predicate covers a large fraction of the database, latency matters more than recall, or the global HNSW is already available and temporal filtering is uncommon.

---

# 2. Density-balanced bucket HNSW

## Design changes from post-filtering

The timeline is divided into 64 buckets containing approximately equal populations. Every bucket owns a separate FAISS HNSW and local-to-global ID mapping.

```text
query intervals
   ↓
normalize interval union
   ↓
route to intersecting buckets
   ↓
fully covered bucket: ordinary HNSW search
partially covered bucket: overfetch + exact timestamp filtering
   ↓
deduplicate and exact rerank
```

Density balancing is important because fixed-duration buckets would vary greatly across Twitter bursts and quiet periods.

Representative configuration:

```text
64 density-balanced buckets
M = 32
efConstruction = 200
efSearch = 128
overfetch = 2
boundary safety = 1.25
```

## Full-data metrics

| Metric | Result |
|---|---:|
| Recall@100 | 0.9991 |
| Mean latency | 144.18 ms |
| p50 latency | 44.43 ms |
| p95 latency | 566.15 ms |
| Candidates examined | 18,708 |
| Valid candidates | 827.66 |
| Distance computations | 35,811 |
| Buckets selected | 3.89 |
| Partially covered buckets | 3.26 |
| Complete-result rate | 0.9329 |
| Build time | 52.7 seconds |
| HNSW storage | 1.790 GB |
| ID map + timestamp storage | 15.9 MB |

The 0.9329 complete-result rate is the workload maximum because 644 queries have fewer than 100 eligible records.

## Results by temporal coverage

| Coverage | Recall | Mean latency |
|---:|---:|---:|
| 1 hour | 0.9999 | 350.43 ms |
| 6 hours | 0.9999 | 250.23 ms |
| 24 hours | 0.9997 | 58.27 ms |
| 168 hours | 0.9983 | 29.00 ms |
| 720 hours | 0.9975 | 32.95 ms |

The counterintuitive narrow-range latency comes from boundary compensation: a one-hour range intersects only a small fraction of a large population-balanced bucket, so the search requests many bucket candidates to guarantee enough survivors.

## Fragmentation

| OR predicates | Recall | Mean latency |
|---:|---:|---:|
| 1 | 0.9983 | 42.58 ms |
| 2 | 0.9991 | 106.22 ms |
| 3 | 0.9994 | 176.29 ms |
| 4 | 0.9996 | 251.61 ms |

Every extra interval can introduce additional partial boundary buckets. Quality remains high, but routing and candidate work increase sharply.

## Advantages

- Nearly exact recall.
- Temporal routing happens before ANN search.
- Fully covered buckets require no filtering.
- Buckets can be searched in parallel.
- Old buckets can be expired or replaced independently.
- Late data can be routed to an appropriate bucket or delta bucket.
- Uses optimized unmodified FAISS HNSW within each partition.

## Shortcomings

- Boundary buckets dominate narrow predicates.
- Query fanout grows with fragmented predicates.
- Candidate budgeting and merging are complex.
- Small or skewed buckets can have weak graph quality.
- Multiple indexes complicate updates, persistence and memory management.
- Mean latency is 11.6 times higher than exact pre-filtering on the complete workload.

## Appropriate use

Bucket HNSW is useful for broad predicates where exact vector scanning is expensive. In this experiment, it is approximately 29.6% faster than exact search at 720 hours while retaining 0.9975 recall. Narrow ranges should be routed to exact search instead.

---

# 3. Flat multi-scale RT-HNSW

## Design objective

RT-HNSW keeps one chronological graph but makes temporal eligibility part of entry selection, graph topology and traversal.

```text
query interval
   ↓
find in-range entries
   ↓
strict graph traversal
   ↓
semantic + temporal + scale-compatible edges
   ↓
eligible top-k
```

FAISS HNSW is used only during construction to generate semantic candidates. Query traversal uses the custom persisted graph.

## Graph layout

| Edge class | Budget |
|---|---:|
| Ordinary semantic | 20 |
| 24-hour left/right range | 2 + 2 |
| 168-hour left/right range | 2 + 2 |
| 720-hour left/right range | 2 + 2 |
| Nominal semantic/range degree | 32 |
| Protected predecessor/successor | Separate |

Observed semantic-plus-range degree is 29.23.

Every range scale obtains candidates independently from:

- Semantic construction candidates
- Sampled records inside the temporal scale
- Immediate chronological neighbours

Candidates use co-eligibility-weighted vector distance, favoring edges likely to survive the intended temporal filters.

## Strict traversal

A destination is checked against the exact half-open timestamp interval before vector distance calculation. For disjoint predicates, each normalized component is searched independently and results are deduplicated and reranked.

Queries with at most 5,000 eligible vectors use exact search. This avoids approximation overhead and provides perfect narrow-range quality.

## 100k metrics

| efSearch | Recall@100 | Mean latency | p95 latency | Distance computations |
|---:|---:|---:|---:|---:|
| 128 | 0.9173 | 4.65 ms | 18.02 ms | 2,134 |
| 256 | 0.9515 | 7.57 ms | 33.86 ms | 2,742 |
| 512 | 0.9741 | 12.96 ms | 61.82 ms | 3,825 |
| 1024 | 0.9875 | 22.83 ms | 114.56 ms | 5,630 |
| 2048 | 0.9946 | 41.44 ms | 211.75 ms | 8,352 |

Build and storage:

| Metric | Result |
|---|---:|
| Build time | 133.1 seconds |
| Custom graph storage | 15.6 MB |
| Gateways | 195 |
| Vectors included in storage figure | No |

## Broad approximate quality

| Coverage | ef=256 | ef=512 | ef=1024 | ef=2048 |
|---:|---:|---:|---:|---:|
| 168 hours | 0.8699 | 0.9316 | 0.9694 | 0.9872 |
| 720 hours | 0.8873 | 0.9391 | 0.9681 | 0.9860 |

The 1, 6 and 24-hour slices are generally exact. Overall recall therefore combines exact narrow queries and approximate broad queries and should not be interpreted as broad-query recall alone.

## Fragmentation at efSearch 256

| Components | Recall | Mean latency | Distance computations |
|---:|---:|---:|---:|
| 1 | 0.9416 | 4.40 ms | 1,563 |
| 2 | 0.9452 | 7.55 ms | 2,147 |
| 4 | 0.9675 | 10.76 ms | 4,516 |

Fragmentation increases work approximately additively because components are searched independently. Recall rises in this matched workload partly because more components receive independent candidate budgets.

## Project-level novelty

The novel contribution is the combination of:

1. **Protected chronological connectivity** — predecessor/successor links cannot be removed by semantic pruning.
2. **Independent multi-scale range budgets** — 24, 168 and 720-hour topology does not compete for one neighbour list.
3. **Temporal co-eligibility scoring** — construction values an edge according to both vector proximity and expected filter survival.
4. **Strict pre-distance filtering** — invalid destinations are rejected before vector evaluation rather than only before output.
5. **Query-aware in-range entries** — search does not depend on a global entry outside the predicate.
6. **Hybrid exact/graph execution** — selective ranges use exact pre-filtering automatically.
7. **Chronological online construction** — new records are inserted using only previously available records.

These are project-level novelty claims about the implemented combination, not a claim that no related filtered-graph technique exists in the literature.

## How it helps

- It converts the query from “global ANN then filter” into “ANN inside the eligible temporal subgraph.”
- Multi-scale edges improve recall by 1.37 percentage points over the best single-scale graph at efSearch 256.
- Protected temporal links improve the initial semantic graph by 2.36 points at efSearch 128.
- It avoids bucket boundary overfetch and multi-index candidate merging.
- Exact fallback makes selective queries both correct and inexpensive.
- One graph naturally supports chronological streaming insertion.

## Advantages

- One logical index instead of 64 independent HNSWs.
- Strict temporal correctness.
- Tunable recall/latency through `efSearch`.
- Small custom graph storage, excluding vectors.
- Better practical trade-off than single-scale and semantic-only prototypes.
- Natural basis for Kafka-ordered incremental updates and immutable snapshots.

## Shortcomings

- Broad recall remains below 0.99 at practical beam widths.
- Python traversal limits latency and throughput.
- Flat topology has no demonstrated logarithmic scaling.
- Ordered local IDs complicate significantly late arrivals.
- Gateway and upper-hierarchy experiments provided negligible gains.
- Evaluation currently uses only 100,000 vectors.

---

# 4. Packed temporal-bitmap RT-HNSW

## Design objective

The bitmap version asks two additional questions:

1. How can old temporal regions retain connectivity as future reciprocal edges arrive?
2. Can the query avoid decoding adjacency groups that contain no eligible epoch?

It remains one logical chronological graph, but changes both topology and physical adjacency representation.

## Fixed base degree budget

| Edge class | Nominal degree |
|---|---:|
| Same-epoch local semantic | 4 |
| New-to-old historical semantic | 8 |
| Old-to-new future reciprocal semantic | 8 |
| Multi-scale range | 6 |
| 1/16-epoch bridges | 4 |
| Protected predecessor/successor | 2 |
| **Total** | **32** |

Historical and future links have independent lists. A future insertion therefore cannot evict an old node's historical or local links.

Observed degree is only 24.47 because bounded reciprocal replacement leaves future and bridge lists underfilled:

```text
local 4.00
historical 8.00
future 3.78
range 4.47
bridge 2.22
temporal 2.00
```

## Packed epoch-grouped adjacency

Time is divided into 24-hour epochs. Adjacency is persisted as two-level CSR:

```text
(node, edge class)
    -> sorted epoch blocks
        -> packed neighbor IDs
```

The query binary-searches the block epochs and decodes only intersecting blocks. Conservative `uint64` summaries have no false negatives; exact timestamp checks remain authoritative for partial epoch boundaries.

## Region-assigned portals

The 195 portals are assigned to exact epoch regions. Every portal has:

- Four semantic links within its epoch or adjacent epochs
- Protected previous/next portal links
- A coverage bitmap
- An epoch-to-portal CSR directory

The query selects portals only from intersecting regions, filters exact timestamps, and performs a bounded portal beam before level-0 search.

## Streaming changes

This version integrates with the ordered Kafka pipeline:

```bash
kafka-twitter-index-writer --topology temporal-bitmap
```

The writer persists insertion adjacency, edge scores and the FAISS candidate generator. Only new vectors execute graph insertion after restart. Packed blocks, bitmaps and portal directories are regenerated as immutable query summaries. WAL replay, contiguous sequence enforcement, content-hash conflict detection, atomic generation publication and live snapshot loading are supported.

## Four v2 variants at efSearch 256

| Variant | Recall | Mean latency | p95 | Distances | Edges examined | Rejected edges |
|---|---:|---:|---:|---:|---:|---:|
| Packed epochs + portal navigation | 0.9443 | 19.46 ms | 97.63 ms | 2,868 | 3,575 | 211 |
| Packed, no pruning + portals | 0.9443 | 13.85 ms | 68.85 ms | 2,868 | 5,270 | 1,906 |
| Packed epochs + direct entries | 0.9442 | 19.33 ms | 97.72 ms | 2,875 | 3,587 | 211 |
| Packed, no pruning + direct entries | 0.9442 | 13.82 ms | 68.39 ms | 2,875 | 5,286 | 1,910 |

Build and storage:

| Metric | Result |
|---|---:|
| Build time | 136.7 seconds |
| Epochs | 38 |
| Query artifact size | 34.5 MB |
| Portal nodes | 195 |
| Vectors included in storage figure | No |

Broad quality:

| Coverage | Recall |
|---:|---:|
| 168 hours | 0.8546 |
| 720 hours | 0.8667 |

## What bitmap pruning accomplishes

Compared with scanning every packed block under the same portal policy:

- Recall is identical.
- Edge examination decreases by 32.2%.
- Explicit out-of-range rejection decreases by 88.9%.
- Distance computations do not change.
- Python mean latency increases by approximately 40%.

Distance computations are unchanged because strict traversal already rejected those edges before distance calculation. Bitmap blocks save adjacency decoding and timestamp tests, not vector-distance work.

The current implementation makes one or more NumPy calls for very small neighbour groups. That overhead is greater than the work saved. The representation is better suited to a fused C++ loop, SIMD block traversal or disk-based adjacency where avoiding block reads is more valuable.

## Portal result

Portal navigation visits 23.0 portal nodes and examines 71.3 portal edges per query. Compared with direct region entries, it:

- Improves recall by only 0.0067 percentage points.
- Saves approximately six distance computations per query.

The portal layer is active and correct but not materially useful on the current 38-epoch dataset.

## Project-level novelty

Relative to flat RT-HNSW, the bitmap version adds:

1. **Directionally protected semantic budgets** — historical and future reciprocal links cannot evict each other.
2. **Time-stretched skip connectivity** — deterministic temporal bridges supplement the immediate backbone.
3. **Conservative edge-group epoch summaries** — an empty intersection safely proves an adjacency group cannot help.
4. **Packed epoch CSR** — temporal filtering occurs before adjacency decoding, not only before vector distance.
5. **Region-assigned portal navigation** — upper routing is constrained by temporal coverage rather than global semantics alone.
6. **Restart-equivalent Kafka continuation** — micro-batch insertion produces the same packed graph as uninterrupted chronological construction in differential tests.
7. **Immutable live generations** — readers pin one generation while new queries move atomically to the next.

Again, novelty refers to the implemented combination and experimental evaluation, not a universal literature-priority claim.

## How it helps

- Prevents future insertions from consuming every edge useful to historical queries.
- Reduces invalid adjacency decoding for selective ranges.
- Makes temporal block metadata directly usable by memory-mapped or disk-based traversal.
- Provides deterministic routing ownership through epoch regions.
- Extends the graph into a crash-recoverable streaming system rather than only an offline prototype.
- Preserves strict exact filtering after coarse bitmap pruning.

## Current disadvantages

- Recall is 0.72 percentage points below flat RT-HNSW at efSearch 256.
- Broad recall is lower at both 168 and 720 hours.
- Actual future reciprocal occupancy is much lower than its nominal budget.
- Packed metadata more than doubles custom graph storage.
- Python/vectorized traversal is slower than scanning packed adjacency.
- Portal benefit is negligible.
- Packed summaries must currently be regenerated at each Kafka generation.

The bitmap version is therefore an important systems/topology experiment, not the recommended query topology.

---

# Cross-architecture design evolution

| Property | Post-filter | Bucket HNSW | RT-HNSW | Bitmap RT-HNSW |
|---|---|---|---|---|
| Number of logical graphs | 1 | 64 | 1 | 1 |
| Filter used for entry selection | No | Bucket routing | Yes | Yes, epoch regions |
| Filter used in topology | No | Partition boundaries | Multi-scale edges | Directional edges + epochs |
| Filter enforced before distance | No | Partially | Yes | Yes |
| Exact fallback | No | Planner-dependent | Yes | Yes |
| Multi-interval support | Post-filter union | Bucket union | Component search | Component search |
| Protected temporal connectivity | No | Inside bucket only | Yes | Yes |
| Rolling insertion | Standard global insert | Per-bucket insert | Chronological graph | Kafka continuation |
| Late arrival support | Easy append | Route/delta bucket | Not native | Late Kafka topic/delta required |
| Packed temporal adjacency | No | No | No | Yes |
| Region-aware upper routing | No | Bucket directory | Weak gateways | Epoch portals |
| Best demonstrated property | Speed | Recall | Trade-off | Edge-pruning safety |

## Architectural progression

```text
Post-filter:
vector search first, temporal correctness afterward

Bucket HNSW:
temporal partition routing first, vector search second

RT-HNSW:
one graph whose edges and traversal understand temporal eligibility

Bitmap RT-HNSW:
rolling-direction topology plus physical adjacency blocks that understand epochs
```

---

# Which method is useful for which query?

## Narrow and highly selective temporal ranges

Best current choice: exact temporal pre-filtering, used automatically by RT-HNSW below 5,000 eligible vectors.

Global post-filtering has extremely low recall; bucket HNSW overfetches large boundary buckets; graph traversal is unnecessary when the eligible set is already small.

## Broad ranges requiring near-exact recall

Best demonstrated choice: density-balanced bucket HNSW.

At 168–720 hours it retains approximately 0.998 recall with 29–33 ms mean latency on the full dataset.

## Broad ranges with a tunable quality/latency target

Best current research topology: flat multi-scale RT-HNSW.

It offers a continuous beam-width trade-off and avoids multi-index fanout, although broad recall requires large `efSearch` for near-exact quality.

## Unfiltered or weakly filtered search

Best choice: standard global HNSW. Temporal structures add overhead without benefit.

## High-rate ordered streaming with historical-range requirements

Most complete systems prototype: temporal-bitmap RT-HNSW with Kafka continuation.

Its query quality is currently weaker, but it provides explicit insertion-direction protection, WAL recovery, ordered replay, immutable generations and live snapshot switching.

## Many fragmented intervals

- Global post-filtering remains low quality.
- Bucket HNSW quality remains high but fanout becomes expensive.
- RT-HNSW searches each component independently, so work grows with fragment count.
- Bitmap blocks can skip epoch-disjoint adjacency, but current Python overhead prevents a latency gain.

---

# Storage interpretation

Storage figures are not directly equivalent:

- Global and bucket FAISS `IndexHNSWFlat` files include FAISS vector storage.
- RT-HNSW and bitmap figures report custom graph artifacts and exclude the external vector matrix.
- Kafka streaming generations additionally retain insertion state and vectors, so their storage is larger than the read-only query artifact.

A fair storage comparison must report vectors, graph adjacency, mappings, metadata, construction state and retained generations separately.

---

# Final conclusions

## Global post-filter HNSW

The correct baseline for simplicity and speed, but not a viable high-recall solution for selective temporal predicates.

## Bucket HNSW

The strongest quality result. Its weakness is cost variance: narrow boundary queries and fragmented predicates can be much slower than exact search.

## Flat multi-scale RT-HNSW

The strongest current single-graph design. Its key contribution is integrating temporal co-eligibility into graph construction and strict traversal while retaining a hybrid exact path.

## Packed temporal-bitmap RT-HNSW

The most advanced rolling systems prototype. It demonstrates safe epoch-block pruning and restart-equivalent streaming insertion, but does not improve current recall or Python latency. Its packed representation should be evaluated next with a fused C++ traversal rather than further Python tuning.

## Recommended production-style planner

A practical system should not force every query through one method:

```text
small eligible set
    -> exact pre-filter

broad range requiring very high recall
    -> bucket HNSW

moderate/broad range with latency-quality trade-off
    -> flat multi-scale RT-HNSW

unfiltered query
    -> global HNSW
```

The bitmap topology is currently best treated as the streaming and packed-adjacency research branch, not the default query engine.
