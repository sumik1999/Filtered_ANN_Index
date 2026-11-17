# Twitter temporal ANN comparative analysis

## Evaluation

All methods use the same 9,600-query balanced DLT-Tweets workload, 990,000 database vectors, 384-dimensional normalized MiniLM embeddings, squared L2, hard half-open timestamp predicates, and exact filtered top-100 ground truth.

Compared methods:

1. Exact temporal pre-filter followed by blocked exact L2.
2. Global HNSW post-filter, `efSearch=128`, over-fetch 2 (same nominal settings as bucketed HNSW).
3. Global HNSW post-filter, `efSearch=256`, over-fetch 10 (stronger conventional global baseline).
4. Density-balanced 64-bucket HNSW, `efSearch=128`, over-fetch 2, boundary safety 1.25.

## Overall results

| Method | Recall@100 | Derived precision | Complete rate | Mean latency | p50 | p95 | Mean candidates |
|---|---:|---:|---:|---:|---:|---:|---:|
| Exact pre-filter | 1.0000 | 1.0000 | 0.9329 | 12.39 ms | 1.82 ms | 50.89 ms | 22,230 |
| Global HNSW, ef128/of2 | 0.0696 | 0.5523 | 0.0090 | 0.93 ms | 0.93 ms | 1.40 ms | 200 |
| Global HNSW, ef256/of10 | 0.2347 | 0.7326 | 0.0833 | 3.71 ms | 3.86 ms | 5.07 ms | 1,000 |
| Density-64 bucket HNSW | 0.9991 | 0.9991 | 0.9329 | 144.18 ms | 44.43 ms | 566.15 ms | 18,708 |

Precision is derived per query as `true-positive returned IDs / returned IDs`. Exact and bucketed completeness is capped by 644 workload queries with fewer than 100 eligible records; 0.9329 is therefore the maximum possible complete-result rate.

## Results by temporal coverage

| Coverage | Exact latency | Global of10 recall / latency | Bucket recall / latency |
|---:|---:|---:|---:|
| 1 hour | 0.18 ms | 0.0070 / 3.69 ms | 0.9999 / 350.43 ms |
| 6 hours | 0.61 ms | 0.0276 / 3.68 ms | 0.9999 / 250.23 ms |
| 24 hours | 2.02 ms | 0.0669 / 3.74 ms | 0.9997 / 58.27 ms |
| 168 hours | 12.34 ms | 0.2750 / 3.73 ms | 0.9983 / 29.00 ms |
| 720 hours | 46.81 ms | 0.7971 / 3.73 ms | 0.9975 / 32.95 ms |

## Fragmentation effect

| OR predicates | Exact mean latency | Global of10 recall / latency | Bucket recall / latency |
|---:|---:|---:|---:|
| 1 | 12.02 ms | 0.2392 / 3.53 ms | 0.9983 / 42.58 ms |
| 2 | 12.50 ms | 0.2289 / 3.62 ms | 0.9991 / 106.22 ms |
| 3 | 12.42 ms | 0.2369 / 3.83 ms | 0.9994 / 176.29 ms |
| 4 | 12.62 ms | 0.2337 / 3.87 ms | 0.9996 / 251.61 ms |

## Conclusions

- Global post-filter HNSW is fastest but unusable for selective temporal predicates at the tested budgets. Over-fetch 10 raises recall only to 0.235 overall and completes only 8.33% of queries.
- Density-balanced bucket HNSW solves the quality problem: recall is 0.9991 and it reaches the workload's maximum possible completeness.
- The current bucket boundary allocation is too aggressive for narrow ranges. Inverse-selectivity compensation often requests most of a boundary bucket, producing 350 ms mean latency for one-hour ranges even though exact pre-filtering takes 0.18 ms.
- Fragmentation is expensive for bucket HNSW because each extra OR interval introduces additional partial buckets. Mean latency rises from 42.58 ms for one predicate to 251.61 ms for four.
- Exact pre-filtering is the best overall method for this workload: it is exact and its 12.39 ms mean latency is 11.6 times lower than bucket HNSW.
- Bucket HNSW becomes beneficial for broad predicates: at 720 hours it is about 29.6% faster than exact search while retaining 0.9975 recall.
- These results support the adaptive planner design: use exact pre-filtering for narrow/selective predicates, bucket HNSW for broad ranges where exact vector scanning is expensive, and avoid global post-filtering unless low recall is acceptable or candidate budgeting is made selectivity-aware.
