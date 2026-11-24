# Coverage-aware RT-HNSW branch validation

Branch: `RT-HNSW-coverage-aware`

## Unit tests

Command:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

Result: **50 tests passed**.

In addition to the base RT-HNSW checks, this branch validates persisted coverage
metadata, strict block-local expansion, mandatory exploration of every
intersecting block, and mixed OR/AND predicate resolution.

## Mixed OR/AND query benchmark

Graph: 100,000-vector multi-scale RT-HNSW
Configuration: `efSearch=256`, `k=100`, exact fallback at 5,000 eligible rows
Workload: 900 queries, split evenly into 450 OR and 450 AND queries.

| Method | Operator | Queries | Recall@100 | Mean computations | GT block coverage |
|---|---|---:|---:|---:|---:|
| Flat multi-scale | OR | 450 | 0.94751 | 2,689.9 | 0.99993 |
| Flat multi-scale | AND | 450 | 0.94249 | 1,556.3 | 0.99979 |
| Coverage-guided | OR | 450 | 0.94989 | 2,925.5 | 1.00000 |
| Coverage-guided | AND | 450 | 0.94467 | 1,825.7 | 1.00000 |
| Flat multi-scale | Combined | 900 | 0.94500 | 2,123.1 | 0.99986 |
| Coverage-guided | Combined | 900 | **0.94728** | 2,375.6 | **1.00000** |

Detailed output:

```text
results/twitter_rt_hnsw_100k/mixed_ef256/summary.json
results/MIXED_OR_AND_BENCHMARK.md
```
