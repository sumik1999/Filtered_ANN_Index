# Hierarchical RT-HNSW experiment

## Implemented structure

The multi-scale base graph is augmented with a geometric hierarchy. Levels are assigned deterministically during chronological insertion with promotion probability 0.05 and seed 23.

| Level | Nodes |
|---:|---:|
| 0 | 100,000 |
| 1 | 4,880 |
| 2 | 245 |
| 3 | 12 |

Every upper level contains:

- eight bounded semantic neighbours selected through an incremental level-specific FAISS HNSW candidate index; and
- protected bidirectional chronological links between consecutive nodes present at that level.

All three chronological level chains were validated. The complete custom graph occupies 39.7 MB excluding vectors, versus approximately 15.6 MB for the flat multi-scale graph.

## Query algorithm

For every non-exact interval:

1. Find the highest hierarchy level containing an in-range node.
2. Sample up to 16 in-range nodes from that level and retain the four closest query-aware entries.
3. Run strict upper-layer beam search from the highest level down to level 1.
4. Reject out-of-range destinations before vector evaluation.
5. Pass the best upper-layer entries to the existing level-0 search.
6. At level 0, search multi-scale range, semantic and protected temporal edges.

The default upper-layer beam width is 32.

## Results

The comparison below uses the same hierarchical index with upper traversal disabled and enabled, isolating query-time hierarchy effects.

| Base efSearch | Flat traversal recall | Hierarchical recall | Hierarchical mean latency | Upper distance computations |
|---:|---:|---:|---:|---:|
| 128 | 0.9173 | 0.9194 | 5.40 ms | 115 |
| 256 | 0.9515 | 0.9525 | 10.63 ms | 115 |
| 512 | 0.9741 | 0.9745 | 13.70 ms | 115 |
| 1024 | 0.9875 | 0.9877 | 23.61 ms | 115 |

At base efSearch 256, varying the upper beam gives:

| Upper efSearch | Recall | Mean latency | Upper distance computations |
|---:|---:|---:|---:|
| 8 | 0.9516 | 7.75 ms | 45 |
| 16 | 0.9519 | 7.67 ms | 72 |
| 32 | 0.9525 | 10.63 ms | 115 |
| 64 | 0.9533 | 8.56 ms | 182 |

Timing differences at this scale include Python traversal noise, but the recall conclusion is stable: upper traversal adds at most about 0.18 percentage points in the tested settings.

## Critical limitations found

1. **Strict filtering removes upper semantic shortcuts.** Globally useful upper neighbours frequently lie outside a query interval and cannot be traversed.
2. **Chronological upper links guarantee reachability, not semantic progress.** They preserve each induced level chain but usually do not move toward the query vector.
3. **The hierarchy is sparse.** Only 245 nodes reach level 2 and 12 reach level 3, so many intervals begin at level 1 or obtain weak high-level coverage.
4. **Upper semantic construction is not range-aware.** It optimizes global vector proximity rather than edge survival inside fixed temporal windows.
5. **Level-0 multi-source entry is already strong.** Query-aware periodic seeds and multi-scale base edges leave little additional work for the hierarchy to improve.
6. **Python heap traversal adds overhead.** A C++ implementation would reduce latency but would not resolve the small recall gain.

## Conclusion

All requested hierarchical mechanisms are implemented and structurally valid, but the first hierarchy does not materially improve performance. The critical failure is not implementation feasibility; it is that ordinary global upper-level semantic edges have low utility under strict interval filtering.

A defensible next experiment would make upper-level edges range-aware at one or more temporal scales, or raise promotion density and compare at a fixed storage budget. Adaptive gateways should still be deferred until an upper topology demonstrates a meaningful navigation benefit.
