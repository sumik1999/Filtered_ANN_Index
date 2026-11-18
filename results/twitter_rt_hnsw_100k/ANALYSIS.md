# Flat RT-HNSW 100k prototype: first ablation

## Setup

- Earliest 100,000 DLT-Tweets database vectors in stable timestamp order.
- Timeline: 2021-01-01 through 2021-02-07.
- 384-dimensional normalized MiniLM vectors; squared L2.
- Flat semantic degree: 32.
- Protected predecessor/successor temporal backbone.
- 195 periodic gateways, one every 512 insertions, gateway degree 8.
- Exact fallback at 5,000 eligible records.
- 900 matched held-out queries: 60 queries in every coverage/fragmentation cell.
- Coverage: 1, 6, 24, 168 and 720 hours.
- Fragment counts: 1, 2 and 4.

The exact fallback handles the 1, 6 and 24 hour slices. The 168 and 720 hour slices exercise strict graph traversal.

## Ablation at efSearch=128

| Variant | Recall@100 | Mean latency | p95 latency | Mean distance computations |
|---|---:|---:|---:|---:|
| Semantic only | 0.8496 | 3.34 ms | 11.57 ms | 1,847 |
| Semantic + temporal | 0.8732 | 3.90 ms | 13.89 ms | 2,029 |
| Semantic + temporal + gateway edges | 0.8731 | 3.63 ms | 13.48 ms | 2,029 |

The temporal backbone improves overall recall by 2.36 percentage points, but periodic gateway edges add no measurable recall improvement.

## Recall by coverage for the complete graph

| Coverage | ef=128 | ef=256 | ef=512 |
|---:|---:|---:|---:|
| 1 hour | 1.0000 | 1.0000 | 1.0000 |
| 6 hours | 1.0000 | 1.0000 | 1.0000 |
| 24 hours | 1.0000 | 1.0000 | 1.0000 |
| 168 hours | 0.6272 | 0.7212 | 0.8086 |
| 720 hours | 0.7382 | 0.8241 | 0.8817 |

At efSearch=512, complete-graph mean latency is 10.42 ms overall; the 720-hour slice reaches 31.91 ms and 0.8817 recall. This remains below the intended 0.99 recall target.

Increasing gateway candidates from 16 to 64 and gateway seeds from 4 to 8 at efSearch=256 changes complete-graph recall only from 0.9090 to 0.9095 overall. The limitation is therefore graph navigability, not simply too few initial seeds.

## Conclusions

1. Strict timestamp traversal and exact fallback work correctly.
2. The protected temporal backbone improves reachable search quality, but does not by itself provide efficient high-recall navigation.
3. Current periodic gateway edges are ineffective at the same search width. Adaptive gateways should not be added yet because the periodic gateway mechanism has not established a useful baseline.
4. Increasing beam width improves recall but does not reach 0.99; efSearch alone is not the solution.
5. The next topology experiment should add temporally compatible range-aware semantic edges with separate left/right budgets, then repeat the same ablation at equal total degree.
6. A second useful experiment is upper-level temporal chains in a true hierarchical implementation. The current prototype is intentionally flat.
