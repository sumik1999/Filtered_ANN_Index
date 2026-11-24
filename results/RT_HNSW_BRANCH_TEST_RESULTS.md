# RT-HNSW branch validation

Branch: `RT-HNSW`

## Unit tests

Command:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

Result: **48 tests passed**.

The suite covers chronological temporal edges, strict half-open filtering,
vector-access isolation, exact fallback, disjoint query components, independent
24/168/720-hour range budgets, hierarchy links, gateway overrides, bucket
routing, and the mixed OR/AND predicate resolver.

## Mixed OR/AND query benchmark

Index: flat 100,000-vector multi-scale RT-HNSW
Configuration: `efSearch=256`, `k=100`, exact fallback at 5,000 eligible rows
Workload: 900 queries, split evenly into 450 OR and 450 AND queries.

| Operator | Queries | Recall@100 | Mean computations | Mean nodes visited | Mean range edges | Mean temporal edges |
|---|---:|---:|---:|---:|---:|---:|
| OR | 450 | 0.94751 | 2,689.9 | 2,681.3 | 2,164.5 | 399.1 |
| AND | 450 | 0.94249 | 1,556.3 | 1,551.5 | 1,137.9 | 208.4 |
| Combined | 900 | **0.94500** | 2,123.1 | 2,116.4 | 1,651.2 | 303.8 |

AND predicates are resolved to their overlapping intersection before strict
RT-HNSW traversal. OR predicates remain separate normalized components and are
searched independently before candidate merging and reranking.

Latency is retained in the JSON artifact but is prototype-level because this
branch performs custom traversal in Python. Recall and deterministic operation
counters are the preferred branch comparison.

Detailed output:

```text
results/twitter_rt_hnsw_100k/mixed_ef256_rt_branch/summary.json
```
