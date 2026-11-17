# 2021 DLT-Tweets exact baseline configuration

## Dataset

| Setting | Value |
|---|---|
| Source | `ExponentialScience/DLT-Tweets` |
| Source revision | `c7d979394488bbd94bd6f493af1775d6b6159176` |
| Reported license | CC-BY-NC 4.0 |
| Time range | `[2021-01-01T00:00:00Z, 2022-01-01T00:00:00Z)` |
| Language | English |
| Eligible 2021 source records | 13,650,782 |
| Selected records | 1,000,000 |
| Database records | 990,000 |
| Held-out query records | 10,000 |
| Sampling | Lowest deterministic MD5 keys (uniform record sample) |
| Retweets | Conventional records beginning with `RT @` removed |
| Source duplicates | Removed by the source dataset creators |

The sample retains native timestamps and relative temporal density; it is not
balanced per hour or day. This preserves real cryptocurrency-event bursts. It
is a domain-specific Twitter workload, not a platform-wide random sample.

## Measured temporal distribution

The statistics below were calculated over all 8,760 hours of 2021 after
selection.

| Metric | Value |
|---|---:|
| Mean tweets/hour | 114.16 |
| Median tweets/hour | 101 |
| Maximum tweets/hour | 1,009 |
| Coefficient of variation | 0.4973 |
| Fano factor | 28.2349 |
| Peak-to-median ratio | 9.9901 |
| Share in busiest 1% of hours | 3.7088% |
| Burst threshold (`median + 3*MAD`) | 179 tweets/hour |

## Vector representation

| Setting | Value |
|---|---|
| Model | `sentence-transformers/all-MiniLM-L6-v2` |
| Dimensions | 384 |
| Storage type | NumPy `float32` memory-mapped matrix |
| Normalization | L2-normalized vectors |
| Metric | Squared Euclidean distance |

For normalized vectors, squared L2 and cosine similarity produce the same
ordering. Query tweets are held out and cannot retrieve themselves.

## Exact hard temporal pre-filter baseline

Each query consists of a held-out MiniLM vector `q`, a half-open timestamp
range `[start_ms, end_ms)`, and `k`.

### Stage 1: exact hard temporal eligibility

DuckDB executes the equivalent of:

```sql
SELECT vector_id
FROM tweets
WHERE timestamp_ms >= start_ms
  AND timestamp_ms < end_ms
ORDER BY vector_id;
```

An ART index is built on `timestamp_ms`. Every returned ID satisfies the range,
and every database record satisfying the range is eligible. Timestamps are not
appended to the embedding and temporal distance does not affect ranking.

### Stage 2: exact semantic ranking

For every eligible vector `v_i`, the baseline computes:

```text
squared_l2(q, v_i) = Σ(j=1..384) (q[j] - v_i[j])²
```

All eligible vectors are examined in blocks of 65,536 vectors. The algorithm
retains the globally smallest `k` distances across blocks. Blocking limits
temporary memory but does not approximate the result.

Therefore, this baseline returns the true semantic top-k among exactly those
tweets satisfying the time predicate. Its expected recall@k against itself is
1.0.

## Default benchmark configuration

| Setting | Value |
|---|---|
| `k` | 100 |
| Measured queries | 1,000 |
| Window durations | 1, 6, 24, 168, and 720 hours |
| Distance block size | 65,536 |
| SQL/vector threads | 4 |
| Traffic groups | burst, ordinary, quiet |

Per-query output records timestamp-filter latency, exact-distance latency,
total latency, eligible count, actual selectivity, candidates examined, result
count, duration, and traffic class. `results/twitter_exact/configuration_report.json`
is generated with the precise runtime configuration when the benchmark runs.
