# RT-HNSW range-aware edge experiment

## Construction

The experiment adds separate online-maintained range-left and range-right adjacency lists. For each chronologically inserted node, candidates are the union of:

- semantic candidates returned by the construction HNSW;
- evenly sampled older nodes inside a 720-hour temporal window; and
- the 16 immediately preceding nodes.

For candidate edge `(x,y)`, construction calculates:

```text
coeligibility = max(0, 1 - timestamp_gap / 720_hours)
score = squared_l2(x,y) / (coeligibility + 1e-6)
```

The lowest-score candidates enter the new node's left-side range budget and the old node's independently bounded right-side budget. Range edges are persisted separately and are expanded before ordinary semantic edges during strict search.

Two equal-nominal-degree configurations were built:

| Configuration | Semantic budget | Left range budget | Right range budget | Nominal total |
|---|---:|---:|---:|---:|
| Range-4 | 24 | 4 | 4 | 32 |
| Range-8 | 16 | 8 | 8 | 32 |

The original baseline has semantic degree 32 and no range edges. The Range-8 graph actually averages 28.56 semantic-plus-range edges because reciprocal right-side lists do not all fill, so its observed degree is lower than the baseline.

## Complete-graph results

The rows below enable semantic, temporal, gateway and range edges.

| Range budget | efSearch | Recall@100 | Mean latency | p95 latency | Mean distance computations |
|---:|---:|---:|---:|---:|---:|
| 4/side | 128 | 0.8887 | 4.64 ms | 17.31 ms | 2,037 |
| 4/side | 256 | 0.9264 | 6.88 ms | 29.25 ms | 2,558 |
| 4/side | 512 | 0.9544 | 12.25 ms | 55.41 ms | 3,504 |
| 8/side | 128 | 0.9009 | 4.52 ms | 17.77 ms | 2,137 |
| 8/side | 256 | 0.9378 | 7.28 ms | 31.46 ms | 2,751 |
| 8/side | 512 | 0.9645 | 13.28 ms | 59.19 ms | 3,853 |
| 8/side | 1024 | 0.9820 | 23.71 ms | 111.51 ms | 5,712 |

For comparison, the original semantic-32 graph with temporal and gateway edges reaches 0.8731, 0.9090 and 0.9381 recall at efSearch 128, 256 and 512 respectively. At equal nominal degree, Range-8 therefore improves recall by approximately 2.6 to 2.8 percentage points.

## Broad-range results for Range-8

| Coverage | Recall ef=128 | Recall ef=256 | Recall ef=512 | Recall ef=1024 |
|---:|---:|---:|---:|---:|
| 168 hours | 0.7096 | 0.8061 | 0.8858 | 0.9406 |
| 720 hours | 0.7951 | 0.8828 | 0.9367 | 0.9693 |

The 1, 6 and 24-hour slices remain exact because they fall below the 5,000-record exact threshold.

## Findings

1. Temporally compatible range edges materially improve strict induced-subgraph navigation at equal nominal degree.
2. Eight edges per temporal side outperform four per side, despite reducing ordinary semantic degree from 24 to 16.
3. Gateway edges remain neutral: enabling them changes recall by less than measurement noise. Adaptive gateways should still be deferred.
4. Range-aware topology is more effective than merely increasing gateway seed count.
5. The flat graph still does not reach the 0.99 recall target. efSearch 1024 reaches 0.982 overall but broad-range recall remains 0.941–0.969 and tail latency rises substantially.
6. The next experiment should add a true hierarchy or multi-scale range edges rather than further increasing flat-graph beam width.
