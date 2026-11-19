# Equal-budget offline gateway entry oracle

## Setup

- Index: flat 100k multi-scale RT-HNSW.
- Gateway edges: disabled; gateways are entry points only.
- Gateway count: 195 for every strategy.
- Gateway degree budget retained at eight but unused in this experiment.
- Candidate pool: every 64th chronological node, 1,562 candidates.
- Search: efSearch 256, exact threshold 5,000, at most 16 candidates and four seeds per interval component.
- Workload: the 360 broad-range graph queries from the matched 900-query workload.
- Training: alternating 180 broad queries.
- Primary evaluation: the other 180 held-out broad queries.

Strategies:

1. Periodic: the existing every-512th node policy.
2. Random: deterministic equal-count sample from the candidate pool.
3. Semantic cluster: representatives of 195 FAISS k-means clusters trained on 10,000 timeline-spanning vectors.
4. Temporal density: one high-traffic representative from each equal-population temporal block.
5. Greedy oracle: facility-location selection maximizing temporal eligibility plus query/candidate semantic affinity on training queries.

## Held-out results

| Strategy | Recall@100 | Mean latency | p95 latency | Distance computations |
|---|---:|---:|---:|---:|
| Periodic | 0.8826 | 16.24 ms | 33.20 ms | 5,168 |
| Random | 0.8818 | 16.11 ms | 32.94 ms | 5,170 |
| Semantic cluster | 0.8817 | 16.12 ms | 33.22 ms | 5,160 |
| Temporal density | 0.8821 | 16.13 ms | 33.19 ms | 5,173 |
| Greedy oracle | 0.8811 | 16.09 ms | 32.90 ms | 5,157 |

The greedy oracle reduces distance computations by only 0.21% and decreases recall by 0.14 percentage points relative to periodic gateways. None of the alternative entry sets approaches the required +1 recall point or 10% computation reduction.

## Interpretation

1. Entry placement is not the current bottleneck. All equal-budget policies reach essentially the same level-0 search basins.
2. The existing evenly sampled, query-scored in-range entry procedure is robust to gateway-set choice.
3. Multi-scale level-0 topology and beam width dominate recall.
4. Semantic clustering and traffic-density placement do not improve held-out quality.
5. The training-query facility-location oracle does not transfer into measurable search benefit.

## Decision

The adaptive-gateway acceptance gate fails. Online adaptive promotion should remain deferred, and gateway nodes should continue to serve as a simple deterministic entry directory. Further adaptive work is not justified unless the search algorithm or gateway-edge topology changes enough to create measurable oracle headroom.
