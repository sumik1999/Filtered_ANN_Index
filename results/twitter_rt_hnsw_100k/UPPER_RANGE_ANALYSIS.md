# Range-aware upper-level RT-HNSW experiment

## Fixed-storage construction

The original hierarchy reserves eight semantic neighbours per upper-level node. The range-aware hierarchy keeps the same nominal upper degree by replacing half of that budget:

| Upper edge class | Original | Range-aware |
|---|---:|---:|
| Semantic | 8 | 4 |
| Range-left | 0 | 2 |
| Range-right | 0 | 2 |
| Nominal total | 8 | 8 |

Range scales are mapped by level:

```text
level 1 -> 24 hours
level 2 -> 168 hours
level 3+ -> 720 hours
```

Each upper range list is built online from upper semantic construction candidates, sampled same-level nodes inside the assigned temporal scale, and recent same-level nodes. It uses the same co-eligibility-weighted squared-L2 score as the base multi-scale graph.

The level populations remain fixed at 4,880, 245 and 12 nodes. Custom artifact size remains 39.7 MB, matching the semantic-8 hierarchy's allocated storage.

## Results

Default upper efSearch is 32.

| Base efSearch | Semantic-8 hierarchy | Upper-range hierarchy | Gain |
|---:|---:|---:|---:|
| 128 | 0.9194 | 0.9210 | +0.0016 |
| 256 | 0.9525 | 0.9528 | +0.0003 |
| 512 | 0.9745 | 0.9747 | +0.0002 |
| 1024 | 0.9877 | 0.9877 | approximately 0 |

At base efSearch 256 and upper efSearch 64, recall rises to 0.9537, only about 0.0004 above the semantic-8 hierarchy at the same upper beam.

The range-aware upper search examines about 182 upper range edges and performs about 148 upper distance computations per query at upper efSearch 32. The extra topology is active, but it does not materially alter the level-0 basin reached by the search.

## Critical failure points

1. Replacing four semantic links with four range links preserves storage but mostly changes redundant upper connectivity rather than finding new level-0 basins.
2. The highest eligible level averages only 1.12 across the workload; most searches receive little true hierarchical descent.
3. Level 2 and level 3 populations are too sparse for strong query-dependent coverage inside arbitrary intervals.
4. Query-aware level-0 multi-scale search dominates final recall, making small upper entry changes disappear at moderate and large base beam widths.
5. Increasing upper beam width adds work but does not produce a meaningful recall/latency improvement.

## Decision

The fixed-storage upper range experiment is implemented successfully but does not justify further tuning of the current hierarchy. A denser hierarchy would consume either more storage or force smaller per-node degree, and should only be attempted if an offline entry-point oracle demonstrates substantial headroom.

Adaptive gateway implementation should therefore begin with an offline oracle and equal-budget selection experiment. If an oracle gateway set cannot materially improve level-0 search, online adaptive promotion should be abandoned.
