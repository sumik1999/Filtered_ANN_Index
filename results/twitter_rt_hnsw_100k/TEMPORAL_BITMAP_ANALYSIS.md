# Time-stretched temporal-bitmap RT-HNSW

## Implemented topology

The graph is replayed in stable timestamp order. New nodes connect backward into the existing graph while old nodes receive reciprocal future links in an independent budget:

| Edge class | Nominal degree |
|---|---:|
| Same-epoch local semantic | 6 |
| New-to-old historical semantic | 6 |
| Old-to-new protected future reciprocal | 6 |
| Range-aware 24/168/720-hour edges | 6 total |
| Temporal skip bridges at 1/4/16/64 epochs | 8 total |
| Protected predecessor/successor | 2 |

Observed average degrees are 6.00 local, 6.00 historical, 2.81 future, 4.47 range, 3.49 bridge and 2.00 temporal, or about 24.77 total. Reciprocal future and bridge lists intentionally remain underfilled when bounded replacement concentrates arrivals on the same older nodes.

Local, historical and future lists cannot evict one another. This preserves old-range connectivity as future records arrive. The graph contains 195 periodic in-range entry portals. The complete 100,000-node predecessor/successor chain and the direction of every historical/future edge were validated after persistence.

Time is divided into 24-hour epochs. Every node stores a conservative bitmap for each edge group, and every portal stores the union of its own epoch and direct-neighbour epochs. A query builds the intersecting epoch range. An edge group whose bitmap has no intersection is skipped before its adjacency is scanned; exact half-open timestamp checks remain authoritative.

## Build

- Vectors: 100,000
- Epochs: 38
- Bitmap words per edge group: one `uint64`
- Custom artifact size: 22.4 MB
- Build time: 127.1 seconds
- Graph insertion is chronological and incremental in memory; this experiment persists after replay rather than through the Kafka generation writer.

## efSearch 256 results

| Variant | Recall@100 | Mean latency | p95 latency | Edges examined | Out-of-range edges rejected | Distance computations |
|---|---:|---:|---:|---:|---:|---:|
| Bitmap pruning | 0.9491 | 11.06 ms | 53.25 ms | 4,797 | 1,211 | 3,077 |
| Same graph, bitmap disabled | 0.9491 | 9.08 ms | 43.24 ms | 5,256 | 1,670 | 3,077 |

Bitmap pruning is recall-equivalent by construction. It reduces edge examination by 8.7% and explicit out-of-range edge rejection by 27.5%, but does not reduce distance computations: every skipped group contains only coarse-epoch-ineligible nodes that strict filtering would have rejected before distance evaluation.

Python bitmap/group dispatch increases mean latency by 21.9% and p95 by 23.1%. The pruning would need packed adjacency blocks and compiled traversal to turn reduced edge decoding into latency improvement.

## Broad-range recall

| Coverage | Bitmap graph | Existing flat multi-scale graph |
|---:|---:|---:|
| 168 hours | 0.8763 | 0.8699 |
| 720 hours | 0.8693 | 0.8873 |
| Overall | 0.9491 | 0.9515 |

The independent time-directional budgets slightly improve the seven-day slice but regress the thirty-day slice. The tested allocation spends too much degree on local/future/bridge protection and too little on broad semantic/range navigation.

## Decision

The implementation validates three mechanisms:

1. new nodes can connect backward without reciprocal future links evicting historical links;
2. deterministic temporal bridges and the protected backbone keep time regions connected; and
3. conservative edge-group bitmaps safely skip adjacency with zero recall change.

The current configuration is not better than flat multi-scale RT-HNSW overall. Do not replace the current topology. The next bitmap experiment should use packed epoch-grouped adjacency and a fixed degree reallocation, then run traversal in C++ or vectorized blocks. Portal coverage bitmaps add little under strict in-range entry selection and should remain diagnostic unless a region-assigned upper routing layer is introduced.
