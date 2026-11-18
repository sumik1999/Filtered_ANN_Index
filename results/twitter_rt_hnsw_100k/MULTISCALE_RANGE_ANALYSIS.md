# RT-HNSW multi-scale range-edge experiment

## Structure

The multi-scale graph reserves independent left/right range-edge budgets for three query-relevant temporal scales:

```text
24 hours, 168 hours, 720 hours
```

The graph uses:

| Edge class | Budget |
|---|---:|
| Ordinary semantic | 20 |
| 24-hour range-left/right | 2 + 2 |
| 168-hour range-left/right | 2 + 2 |
| 720-hour range-left/right | 2 + 2 |
| Nominal semantic/range total | 32 |

Every scale obtains construction candidates independently from semantic HNSW candidates, evenly sampled records inside that scale, and the immediate chronological neighborhood. Each scale applies its own co-eligibility-weighted distance score.

The observed average semantic-plus-range degree is 29.23, lower than both the nominal budget and the original semantic-32 baseline because reciprocal right-side budgets do not always fill.

## Overall results

The table reports the complete semantic + temporal + gateway + multi-scale range graph.

| efSearch | Recall@100 | Mean latency | p95 latency | Mean distance computations |
|---:|---:|---:|---:|---:|
| 128 | 0.9173 | 4.65 ms | 18.02 ms | 2,134 |
| 256 | 0.9515 | 7.57 ms | 33.86 ms | 2,742 |
| 512 | 0.9741 | 12.96 ms | 61.82 ms | 3,825 |
| 1024 | 0.9875 | 22.83 ms | 114.56 ms | 5,630 |
| 2048 | 0.9946 | 41.44 ms | 211.75 ms | 8,352 |

## Comparison with the best single-scale graph

| efSearch | Single-scale Range-8 | Multi-scale | Improvement |
|---:|---:|---:|---:|
| 128 | 0.9009 | 0.9173 | +0.0164 |
| 256 | 0.9378 | 0.9515 | +0.0137 |
| 512 | 0.9645 | 0.9741 | +0.0096 |
| 1024 | 0.9820 | 0.9875 | +0.0055 |

Multi-scale edges improve recall at every tested beam width while using fewer observed edges than the semantic-32 baseline.

## Broad-range quality

| Coverage | ef=128 | ef=256 | ef=512 | ef=1024 | ef=2048 |
|---:|---:|---:|---:|---:|---:|
| 168 hours | 0.7758 | 0.8699 | 0.9316 | 0.9694 | 0.9872 |
| 720 hours | 0.8107 | 0.8873 | 0.9391 | 0.9681 | 0.9860 |

The overall result exceeds 0.99 at efSearch 2048 because the narrow 1, 6 and 24-hour slices are exact. The approximate 168 and 720-hour slices remain slightly below 0.99 and incur high latency at that beam width.

## Conclusion

1. Multi-scale range budgets are more effective than one 720-hour range budget.
2. The gain is largest at practical lower beam widths.
3. Gateway edges remain neutral and can be omitted from the next topology experiment.
4. Raising flat-graph efSearch eventually reaches high recall but produces unacceptable broad-range tail latency.
5. The next structural step should be a true hierarchy with upper-level semantic navigation and protected chronological links, while retaining the multi-scale base-layer range lists.
