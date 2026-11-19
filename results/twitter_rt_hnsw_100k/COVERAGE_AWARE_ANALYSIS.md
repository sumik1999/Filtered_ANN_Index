# Coverage-Aware RT-HNSW Comparative Analysis

## Decision

The coverage overlay is **structurally more deterministic**, but it is **not a
more efficient or generally better default search mechanism** than flat
multi-scale RT-HNSW.

It achieved its formal invariant on every tested graph query and eliminated the
few cases where the baseline missed a temporal block containing an exact top-k
neighbor. At the same `efSearch`, however, the recall gain was small and came
from additional work. At matched mean distance computations, coverage-aware
search was worse.

Recommendation:

- retain the implementation as an optional diagnostic or tail-risk mode;
- do not replace flat multi-scale RT-HNSW as the recommended third design;
- do not add more coverage hierarchy complexity unless a workload oracle shows
  substantially more missed ground-truth blocks than this Twitter workload.

## Evaluation setup

- Database: first 100,000 chronologically ordered Twitter vectors.
- Workload: 900 matched queries.
- Coverage widths: 1, 6, 24, 168, and 720 hours.
- Fragment counts: 1, 2, and 4.
- Exact fallback: 5,000 eligible vectors.
- Graph: existing flat multi-scale RT-HNSW.
- Semantic degree: 20.
- Range degree: two links per side at 24, 168, and 720 hours.
- Default coverage block size: 512 vectors.
- Representatives per block: four.
- Mandatory expansions per block: one unless otherwise stated.
- Metric: squared L2.
- Filtering: strict half-open filtering before destination distance evaluation.

Primary output:

```text
results/twitter_rt_hnsw_100k/coverage_ef256/
```

Additional sweeps are stored under `coverage_ef128`, `coverage_ef512`,
`coverage_block*_ef256`, `coverage_ef256_e*`, and the equal-computation output
directories.

## Main result at efSearch 256

| Metric | Flat multi-scale baseline | Coverage-aware | Change |
|---|---:|---:|---:|
| Recall@100 | 0.95143 | 0.95273 | +0.00130 |
| Graph-query recall@100 | 0.87511 | 0.87846 | +0.00334 |
| Mean distance computations | 2,741.9 | 2,979.0 | +8.65% |
| Mean nodes visited | 2,733.4 | 2,868.6 | +4.95% |
| Mean semantic edges examined | 3,914.0 | 4,623.9 | +18.14% |
| Mean range edges examined | 2,125.7 | 2,517.4 | +18.43% |
| Ground-truth block coverage | 0.99924 | 1.00000 | +0.00076 |
| Coverage invariant failures | N/A | 0 | — |

The paired recall difference over all 900 queries was +0.00130, with a 95%
paired bootstrap interval of approximately `[+0.00070, +0.00201]`. On the 350
queries that used graph traversal, the mean difference was +0.00334 with an
interval of approximately `[+0.00183, +0.00517]`.

This is a measurable same-`efSearch` gain, but it is not free.

## Equal-computation comparison

The baseline at `efSearch=256` used 2,741.9 mean distance computations.
Coverage search with four representatives was adjusted to `efSearch=205` and
used 2,746.8 computations.

| Configuration | Mean computations | Recall@100 |
|---|---:|---:|
| Baseline ef256 | 2,741.9 | 0.95143 |
| Coverage ef205, four representatives | 2,746.8 | 0.94317 |

At essentially equal computation, coverage lost **0.00827 recall**, or 0.827
percentage points. On graph-routed queries, it lost 0.02126 recall.

A cheaper one-representative mode was also matched against the baseline:

| Configuration | Mean computations | Recall@100 |
|---|---:|---:|
| Baseline ef256 | 2,741.9 | 0.95143 |
| Coverage ef238, one representative | 2,747.9 | 0.94933 |

Even the cheaper form lost 0.00210 recall at matched computation. Therefore the
coverage overlay fails the equal-budget acceptance gate.

## Recall across efSearch

| efSearch | Baseline recall | Coverage recall | Gain | Baseline computations | Coverage computations |
|---:|---:|---:|---:|---:|---:|
| 128 | 0.91726 | 0.91898 | +0.00172 | 2,134.5 | 2,379.1 |
| 256 | 0.95143 | 0.95273 | +0.00130 | 2,741.9 | 2,979.0 |
| 512 | 0.97410 | 0.97440 | +0.00030 | 3,823.8 | 4,046.1 |

The gain shrinks as the ordinary beam becomes stronger. This indicates that
coverage mostly compensates for limited entry breadth rather than improving the
underlying graph.

## Broad-query results

At ef256:

| Coverage width | Baseline recall | Coverage recall | Gain | Baseline computations | Coverage computations |
|---:|---:|---:|---:|---:|---:|
| 168 hours | 0.86989 | 0.87278 | +0.00289 | 5,536.5 | 5,696.0 |
| 720 hours | 0.88728 | 0.89089 | +0.00361 | 4,799.9 | 5,825.8 |

The 720-hour case paid approximately 21.4% more distance computations for a
0.361 percentage-point recall gain.

Narrow 1, 6, and 24-hour cells used exact search and remained at recall 1.0.
They do not provide evidence for or against the coverage scheduler.

## Per-query stability

At ef256, among all 900 queries:

- 75 improved;
- 801 tied;
- 24 became worse;
- best change: +0.20 recall;
- worst change: -0.04 recall.

Among the 350 graph-routed queries:

| Statistic | Baseline | Coverage-aware |
|---|---:|---:|
| Mean recall | 0.87511 | 0.87846 |
| Recall standard deviation | 0.10545 | 0.10182 |
| 5th percentile | 0.6845 | 0.6900 |
| 10th percentile | 0.7400 | 0.7400 |
| Median | 0.9000 | 0.9000 |

Coverage slightly tightened the recall distribution, but it did not provide
monotonic per-query improvement. It is therefore only modestly more stable in
quality terms.

## Why the benefit is small

The new block diagnostics reveal that the baseline already visited nearly every
block containing an exact top-k result:

- overall ground-truth block coverage: 99.924%;
- graph-query ground-truth block coverage: 99.805%;
- graph queries missing any ground-truth block: 3 out of 350.

Coverage improved all three exceptional queries:

| Workload ID | Width | Fragments | Baseline block coverage | Baseline recall | Coverage recall |
|---:|---:|---:|---:|---:|---:|
| 81 | 168 | 1 | 0.53125 | 0.15 | 0.21 |
| 214 | 720 | 4 | 0.82090 | 0.58 | 0.63 |
| 664 | 720 | 2 | 0.96364 | 0.78 | 0.79 |

This confirms that the mechanism solves its intended failure mode. That failure
mode is simply rare on this workload. Most remaining recall loss occurs inside
blocks that baseline traversal already reaches.

## Block-size sensitivity

At ef256 with four representatives and one mandatory expansion:

| Block size | Recall@100 | Mean computations | Blocks considered | GT block coverage |
|---:|---:|---:|---:|---:|
| 256 | 0.95387 | 3,208.8 | 73.82 | 1.0 |
| 512 | 0.95273 | 2,979.0 | 37.28 | 1.0 |
| 1,000 | 0.95203 | 2,861.7 | 19.45 | 1.0 |
| 2,048 | 0.95172 | 2,800.0 | 9.88 | 1.0 |

Smaller blocks produce slightly better recall but rapidly increase mandatory
work. Larger blocks approach baseline cost and baseline quality. The mechanism
is therefore sensitive to block granularity, although the trend is predictable.

## Representative-count sensitivity

At block size 512, ef256, and one expansion:

| Representatives scored per block | Recall@100 | Mean computations |
|---:|---:|---:|
| 1 | 0.95196 | 2,828.8 |
| 2 | 0.95249 | 2,877.4 |
| 4 | 0.95273 | 2,979.0 |

More representatives give diminishing recall gains. One representative is the
best low-overhead diagnostic mode, but it still fails the matched-computation
comparison.

## Mandatory-expansion sensitivity

At block size 512, four representatives, and ef256:

| Expansions per block | Recall@100 | Mean computations | Coverage computations |
|---:|---:|---:|---:|
| 1 | 0.95273 | 2,979.0 | 303.1 |
| 2 | 0.95342 | 3,078.7 | 429.5 |
| 5 | 0.95553 | 3,369.9 | 798.3 |

Additional local exploration increases recall, but this is primarily a larger
search budget. It does not establish superior efficiency.

## Latency interpretation

Latency was measured, but Python traversal, memory-map warming, variant order,
and concurrent sweep execution make small latency differences unreliable. The
coverage implementation also performs many Python-level per-block heap and edge
operations.

The robust conclusion is based on deterministic counters:

- coverage always performs additional representative scoring;
- work grows with the number of intersecting blocks;
- broad intervals pay the highest overhead; and
- matched-computation recall is lower.

No latency speedup should be claimed from these runs. A fused implementation
would reduce absolute latency but would not remove the extra distance work.

## Correctness and structural stability

The mechanism was completely stable with respect to its explicit invariant:

- every considered block was covered in all 900-query sweeps;
- no coverage invariant failure occurred;
- strict filtering tests proved that scored and returned vectors remain inside
  the half-open predicate;
- exact fallback behavior was unchanged; and
- metadata overhead at 100k vectors, block size 512, and four representatives
  is only about 8 KB excluding NumPy headers.

This makes it useful for deterministic temporal fairness, diagnostics, or a
service-level mode that values regional coverage over average efficiency.

## Scaling risk

Coverage cost is proportional to the number of intersecting blocks. At block
size 512:

- 100k vectors require about 196 blocks for a full-range query;
- 990k vectors would require about 1,934 blocks;
- four representatives would require up to 7,736 entry distance computations
  before ordinary graph traversal.

Consequently, the current mandatory-all-block policy becomes less stable in
latency and cost as the index and query width grow.

## Final recommendation

### Is it more stable?

- **Coverage correctness:** yes.
- **Deterministic regional participation:** yes.
- **Recall distribution:** slightly.
- **Per-query recall monotonicity:** no.
- **Equal-budget ANN quality:** no.
- **Broad-query computation and scaling:** worse.
- **Replacement for the third design:** no.

Keep flat multi-scale RT-HNSW as the recommended third architecture. Preserve
coverage-aware search as an optional mode for:

1. diagnosing whether a new workload has temporal-region misses;
2. protecting rare tail queries where regional participation is required; or
3. collecting oracle evidence before changing topology.

A better next experiment is a **selective coverage trigger**: invoke coverage
only when entry seeds are temporally concentrated or when an inexpensive
predictor identifies likely missed-region risk. Mandatory coverage of every
block on every broad query is not justified by this evaluation.
