# Varied Twitter temporal-vector query dataset

## Goal

Create a reproducible workload that varies both the hard temporal predicate and
the location of the query in the 384-dimensional MiniLM vector space. In this
dataset, "spatial" means embedding/vector space, not geographic location; the
selected DLT-Tweets data does not provide reliable geographic coordinates.

Every workload query returns top-k by squared L2 distance subject to either:

```text
timestamp in interval_1 OR ... OR interval_f
```

or:

```text
timestamp in interval_1 AND ... AND interval_f
```

Timestamps are hard eligibility conditions and never contribute to vector
distance. OR intervals are disjoint; AND intervals overlap and are resolved to
their intersection. All bounds are half-open `[start,end)`.

## Dataset size and balanced design

Generate 9,600 query records from the 10,000 held-out tweet vectors. Use a
balanced factorial design with 10 records in each operator-aware cell:

| Factor | Levels |
|---|---|
| Boolean operator | OR, AND |
| Temporal predicate count (`f`) | 1, 2, 3, 4 |
| Total eligible time coverage | 1, 6, 24, 168, 720 hours |
| Anchor traffic class | burst, ordinary, quiet |
| Vector-space density | dense, medium-dense, medium-sparse, sparse |
| Time/vector alignment | aligned, independent |

This gives `2 * 4 * 5 * 3 * 4 * 2 * 10 = 9,600` records. Query vectors may be reused
across experimental cells, but the generator should minimize reuse and record a
`query_reuse_number`. Repeated vectors are useful because they isolate the
impact of changing temporal predicates.

The generator must use a fixed seed and deterministically shuffle final records.

## Temporal variation

### Predicate count and coverage

`predicate_count` is the number of intervals joined by the selected operator.
Effective temporal coverage is held constant when predicate count changes. For
OR, a 24-hour query contains:

- one 24-hour interval;
- two 12-hour intervals;
- three 8-hour intervals; or
- four 6-hour intervals.

For AND, predicates are overlapping supersets whose intersection is exactly the
requested effective window. For example, `[00:00,36:00) AND [12:00,48:00)` has
the effective interval `[12:00,36:00)`. Disjoint AND ranges are deliberately
excluded because one timestamp cannot satisfy them and the result is empty.
This design compares fragmentation/intersection structure while controlling
effective coverage. Rounding uses integer milliseconds.

### Traffic classes

Classify every hour from database traffic counts:

- `burst`: count greater than `median + 3*MAD`;
- `ordinary`: count greater than median and at most `median + 3*MAD`;
- `quiet`: count at most median.

The class describes each interval's anchor hour. Select anchors uniformly from
the requested class rather than proportionally to tweet volume, so bursts do
not dominate. For multiple predicates, sample distinct anchors and reject
choices whose resulting intervals overlap.

Store both requested anchor class and measured properties such as eligible
count and actual selectivity. Wide intervals can span several traffic regimes,
so measured selectivity is the authoritative value.

### Time/vector alignment

- `aligned`: at least one interval contains the held-out query tweet's native
  timestamp. The query's native hour must match the requested traffic class.
- `independent`: intervals are sampled independently of the query tweet's
  timestamp and must not contain it.

This tests whether semantic content associated with an event is easier to find
inside that event's time range than in unrelated periods.

For fragmented aligned predicates, one interval contains the native timestamp;
the remaining intervals use disjoint anchors from the same requested traffic
class.

## Vector-space variation

### Density score

For each held-out query vector, search the global database HNSW index with a
high search depth and retrieve 32 neighbours. Define:

```text
local_density_distance = median(distance_to_neighbours_8_through_32)
```

Skipping the first few neighbours reduces sensitivity to near-duplicate posts.
Partition queries into four equal-frequency strata by this score:

1. `dense` — smallest distances;
2. `medium_dense`;
3. `medium_sparse`;
4. `sparse` — largest distances.

The approximate score is used only for workload stratification, not correctness
or final recall measurement.

### Semantic coverage

Train deterministic FAISS k-means with 64 centroids over a reproducible sample
of database vectors. Assign held-out queries to their nearest centroid. Within
each vector-density stratum and experimental cell, select query vectors
round-robin across non-empty semantic clusters. This prevents a small number of
cryptocurrency topics from dominating the workload.

Record cluster ID, centroid distance, local-density distance, and density
quartile. Cluster ID is a coverage label, not a query predicate.

## JSONL schema

Store the workload at:

```text
Twitter_data/workloads/temporal_vector_queries_9600.jsonl
```

Example record:

```json
{
  "schema_version": 1,
  "workload_id": 17,
  "query_id": 4821,
  "query_timestamp_ms": 1623456789000,
  "query_reuse_number": 1,
  "k": 100,
  "metric": "squared_l2",
  "predicate_operator": "OR",
  "predicate_count": 3,
  "requested_total_coverage_hours": 24,
  "intervals": [
    [1609804800000, 1609833600000],
    [1622505600000, 1622534400000],
    [1635724800000, 1635753600000]
  ],
  "anchor_traffic_class": "burst",
  "time_vector_alignment": "independent",
  "native_timestamp_eligible": false,
  "vector_density_class": "medium_sparse",
  "local_density_distance": 0.4132,
  "semantic_cluster_id": 21,
  "centroid_distance": 0.2871,
  "eligible_count": 1842,
  "actual_selectivity": 0.0018606,
  "seed": 7
}
```

`eligible_count` and `actual_selectivity` are populated by an exact timestamp
count after interval construction. Workload records with fewer than `k`
eligible database vectors remain valid and are marked with
`has_at_least_k_eligible=false`.

## Companion files

The generator should write:

```text
Twitter_data/workloads/
├── temporal_vector_queries_9600.jsonl
├── temporal_vector_queries_9600_summary.json
├── query_spatial_metadata.npz
├── semantic_centroids.npy
└── generation_config.json
```

The summary must contain counts for every factor and cross-product cell,
selectivity distributions, interval overlap validation, query reuse counts,
semantic-cluster coverage, and the number of queries with fewer than `k`
eligible vectors.

Exact top-k ground truth should be stored separately:

```text
results/twitter_varied_exact/ground_truth.npz
results/twitter_varied_exact/queries.csv
results/twitter_varied_exact/summary.json
```

Ground truth arrays must align by `workload_id` and pad missing neighbours with
ID `-1` and distance `inf`.

## Validation requirements

Before accepting the workload, verify:

1. exactly 9,600 records and 20 records per factorial cell;
2. predicate counts range from one through four;
3. intervals are ordered, disjoint, in the 2021 range, and half-open;
4. summed interval width equals requested total coverage;
5. aligned queries contain their native timestamp;
6. independent queries exclude their native timestamp;
7. stored eligible counts match an exact DuckDB interval-union count;
8. each vector-density quartile contains the same number of records;
9. all 64 non-empty semantic clusters are represented where possible;
10. identical seed and inputs produce byte-identical JSONL output.

## Reporting plan

Report recall, complete-result rate, candidate count, and latency by:

- predicate count;
- total temporal coverage;
- measured selectivity;
- burst/ordinary/quiet class;
- vector-density class;
- aligned versus independent predicates;
- interactions of predicate count with selectivity and vector density.

The primary comparison should keep total temporal coverage fixed while varying
predicate count. This directly measures the cost of one through four temporal
predicates without confusing fragmentation with selectivity.
