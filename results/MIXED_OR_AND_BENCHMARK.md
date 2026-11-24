# Mixed OR/AND temporal-query benchmark

## Semantics

- `OR`: union of disjoint half-open intervals.
- `AND`: intersection of overlapping half-open intervals.
- For a single timestamp attribute, disjoint AND intervals are empty, so the
  generator creates overlapping predicates whose intersection has the requested
  effective coverage.

Both workloads contain 9,600 balanced queries: 4,800 OR and 4,800 AND. Coverage
levels are 1, 6, 24, 168, and 720 hours.

## Density-64 bucket HNSW (990,000 vectors)

Configuration: `M=32`, `efConstruction=200`, `efSearch=128`, overfetch 2,
boundary safety 1.25, `k=100`.

| Operator | Queries | Recall@100 | Mean latency | p95 latency | Candidates | Selected buckets | Partial buckets | Distance computations |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| OR | 4,800 | 0.99903 | 144.39 ms | 567.03 ms | 18,734.9 | 3.88 | 3.25 | 35,867.5 |
| AND | 4,800 | 0.99871 | 42.21 ms | 147.28 ms | 5,971.1 | 2.38 | 1.43 | 16,144.8 |
| Combined | 9,600 | 0.99887 | 93.30 ms | 434.96 ms | 12,353.0 | 3.13 | 2.34 | 26,006.1 |

AND is cheaper here because its effective predicate is one contiguous
intersection, whereas multi-predicate OR queries fan out across disjoint time
regions. Recall remains above 0.9987 for both operators.

Artifacts:

```text
Twitter_data/workloads/temporal_vector_queries_9600_mixed.jsonl
results/twitter_varied_exact_mixed/
results/twitter_bucket_hnsw_density_mixed/
```

## Flat and coverage-guided RT-HNSW (100,000 vectors)

Configuration: 900 queries (450 per operator), `efSearch=256`, exact fallback at
5,000 eligible vectors, `k=100`.

| Method | Operator | Recall@100 | Mean latency | p95 latency | Distance computations | GT block coverage |
|---|---|---:|---:|---:|---:|---:|
| Flat multi-scale | OR | 0.94751 | 7.09 ms | 31.59 ms | 2,689.9 | 0.99993 |
| Flat multi-scale | AND | 0.94249 | 4.12 ms | 10.81 ms | 1,556.3 | 0.99979 |
| Coverage-guided | OR | 0.94989 | 8.34 ms | 37.02 ms | 2,925.5 | 1.00000 |
| Coverage-guided | AND | 0.94467 | 5.40 ms | 16.76 ms | 1,825.7 | 1.00000 |
| Flat multi-scale | Combined | 0.94500 | 5.61 ms | 19.13 ms | 2,123.1 | 0.99986 |
| Coverage-guided | Combined | 0.94728 | 6.87 ms | 24.41 ms | 2,375.6 | 1.00000 |

Coverage improves combined recall by 0.00228 and guarantees ground-truth block
coverage, while adding about 11.9% distance computations. As in earlier runs,
Python latency is prototype-level and operation counters are the more portable
comparison.

Artifacts:

```text
Twitter_data/workloads/rt_hnsw_100k_queries_mixed.jsonl
results/twitter_rt_hnsw_100k/mixed_ground_truth.npz
results/twitter_rt_hnsw_100k/mixed_ef256/
```
