# Temporal ANN benchmark

A benchmark harness for evaluating vector search over timestamped objects. It includes workload generation, a correctness oracle, exact reference strategies, a global FAISS HNSW baseline, and persistent fixed-width temporal HNSW indexes.

## Included

- Synthetic timestamped vectors with configurable temporal density and time/vector correlation.
- Normalized interval predicates with union and intersection operations.
- Workloads varying temporal width, interval fragmentation, `k`, density, correlation, and fixed partition granularity.
- Exact ground truth and recall@k measurement.
- Per-query latency, candidates examined, auxiliary index size, and build time.
- Reference temporal pre-filter, exact global post-filter proxy, and fixed temporal partition strategies.

The global post-filter implementation is deliberately named a **proxy**: it computes an exact global ranking and applies ANN-style over-fetch/filter semantics. Its recall and candidate counts are useful, but its latency is not representative of an ANN index. A real ANN backend is the next implementation step.

## Set up

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

The project description remains in `requirements.txt` as requested. Python package dependencies are declared in `pyproject.toml`.

## Run

A quick smoke benchmark:

```bash
temporal-ann-benchmark \
  --size 2000 --dimensions 16 --queries 3 \
  --widths 0.01,0.1 --fragments 1,4 --ks 10
```

Run the default benchmark:

```bash
temporal-ann-benchmark
```

Generated datasets are stored in `synthetic_data/` as compressed `.npz` files containing `vectors` and `timestamps`. Generation parameters and filenames are recorded in `synthetic_data/manifest.json`. Use `--data-dir` to choose another location.

Benchmark results are written to:

- `results/queries.csv`: one row per query and method
- `results/builds.csv`: construction time and auxiliary index size
- `results/summary.json`: grouped mean recall/candidate counts and mean/p50/p95 latency

Use `temporal-ann-benchmark --help` for all controls. For example, partition granularity and post-filter over-fetching are controlled by `--buckets` and `--overfetch`.

## FAISS HNSW post-filter baseline

This branch adds a global `IndexHNSWFlat` using squared L2. Timestamps remain external and are joined through the sequential FAISS/SIFT vector ID. Build the index and run the 10,000-query benchmark with:

```bash
build-faiss-hnsw
benchmark-hnsw-post-filter --ef-search 256 --overfetch 10
```

Use comma-separated values such as `--ef-search 256,512` and `--overfetch 10,20,50` to measure recall/latency trade-offs.

## Fixed-width temporal HNSW

The fixed-partition implementation is in `src/temporal_ann/fixed_hnsw/`. It persists one FAISS HNSW index and one local-to-global ID map per non-empty temporal bucket. Build one or more configurations with:

```bash
build-fixed-hnsw \
  --distributions all \
  --bucket-counts 16,32,64 \
  --m 32 --ef-construction 200 --threads 4
```

Benchmark persisted configurations with configurable search depth and per-bucket over-fetching:

```bash
benchmark-fixed-hnsw \
  --distributions all \
  --bucket-counts 16,32,64 \
  --ef-search 64,128,256 \
  --overfetch 1,2,5 \
  --k 100 --warmup 10
```

Indexes default to `SIFT_data/fixed_hnsw/`, separated by timestamp distribution and construction parameters. Results default to `results/fixed_hnsw/`. Queries route to intersecting buckets, convert local labels to global SIFT IDs, apply exact half-open timestamp filtering, deduplicate candidates, and rerank with exact squared L2.

Use `build-fixed-hnsw --help` and `benchmark-fixed-hnsw --help` for data paths, timeline bounds, rebuild controls, and all other parameters.

## Real-world Twitter baseline

The `Twitter_data/` pipeline uses one million English records from the 2021
`ExponentialScience/DLT-Tweets` corpus. It preserves native timestamps and
real event bursts, removes conventional retweets, holds out 10,000 query
tweets, and encodes text with normalized 384-dimensional MiniLM vectors.

```bash
pip install -e '.[twitter]'
download-twitter-dlt
prepare-twitter-dlt --max-records 1000000 --query-count 10000
embed-twitter-dlt --batch-size 256
benchmark-twitter-exact --query-count 1000
build-twitter-hnsw --m 32 --ef-construction 200 --threads 4
benchmark-twitter-hnsw-post-filter \
  --ef-search 256,512 --overfetch 10,50,100

generate-twitter-varied-workload \
  --records-per-cell 20 --clusters 64 --seed 7
```

The exact baseline uses an indexed DuckDB half-open timestamp range scan,
followed by exact blocked squared-L2 search over eligible memory-mapped vectors.
It also saves reusable top-k ground truth. The global HNSW baseline retrieves
vector-nearest candidates first and then enforces the exact half-open timestamp
predicate; its recall, completeness, graph-distance computations, and latency
are measured against that ground truth. Generated ANN indexes and timestamp
arrays are isolated in `Twitter_data/ann/`. The varied-workload generator
creates 9,600 balanced queries spanning one to four OR predicates, five total
coverage widths, burst/ordinary/quiet periods, aligned/independent time ranges,
four vector-density strata, and 64 semantic clusters. See
`Twitter_data/workloads/QUERY_DATASET_DESIGN.md` and `Twitter_data/README.md`
for provenance, layout, limitations, and controls.

### Varied workload and bucketed Twitter ANN

Build exact ground truth and the matching global-HNSW baseline for the balanced
9,600-query workload:

```bash
benchmark-twitter-varied-exact
benchmark-twitter-varied-global-hnsw --ef-search 256,512 --overfetch 10,50,100
```

Build both fixed-width and density-balanced bucket directories, then benchmark
multi-predicate routing and boundary-aware candidate allocation:

```bash
build-twitter-bucket-hnsw \
  --strategies fixed,density --bucket-counts 32,64,128

benchmark-twitter-bucket-hnsw \
  --strategies fixed,density --bucket-counts 32,64,128 \
  --ef-search 64,128,256 --overfetch 1,2,5,10
```

Each bucket persists a FAISS HNSW index, local-to-global ID map, sorted timestamp
array, population statistics, sparse hourly histogram, and construction
manifest. Queries normalize interval unions, route each bucket once, distinguish
fully and partially covered buckets, apply population-aware boundary over-fetch,
filter exact timestamps, deduplicate, and rerank exact squared-L2 distances.

Run the initial cost-based adaptive planner with a built density-balanced index:

```bash
benchmark-twitter-adaptive \
  --strategy density --bucket-count 64 \
  --exact-threshold 5000 --bucket-selectivity-threshold 0.15 \
  --max-buckets 16
```

The planner records whether every query uses exact pre-filtering, bucket HNSW,
or global HNSW post-filtering.

### Flat Rolling Temporal HNSW prototype

The initial RT-HNSW prototype replays the earliest 100,000 Twitter vectors in
stable timestamp order. FAISS HNSW is used only to generate online semantic
construction candidates; query traversal uses separate persisted semantic,
protected temporal, and periodic-gateway adjacency. Gateways are promoted every
512 records. The search implementation uses strict in-range multi-source graph
traversal and falls back to exact search at 5,000 eligible records.

```bash
build-twitter-rt-hnsw \
  --vector-count 100000 \
  --gateway-period 512 \
  --semantic-degree 32 \
  --ef-construction 200
```

The default output is `Twitter_data/ann/rolling_hnsw/flat_100k/`. Generate a
matched 900-query workload and exact ground truth, then run the semantic-only,
semantic-plus-temporal, and semantic-plus-temporal-plus-gateway ablation:

```bash
generate-twitter-rt-workload
benchmark-twitter-rt-hnsw \
  --ef-search 128 --exact-threshold 5000 \
  --gateway-candidates 16 --gateway-seeds 4
```

The workload balances 1, 6, 24, 168 and 720-hour coverage with one, two and
four disjoint predicates. Build the equal-nominal-degree range-aware experiment
with 16 semantic and eight left/right temporal-compatible edges:

```bash
build-twitter-rt-hnsw \
  --output Twitter_data/ann/rolling_hnsw/flat_100k_range8 \
  --semantic-degree 16 --range-degree-per-side 8 \
  --range-candidate-count 64 --range-window-hours 720

benchmark-twitter-rt-hnsw \
  --index Twitter_data/ann/rolling_hnsw/flat_100k_range8 \
  --ef-search 512 \
  --output results/twitter_rt_hnsw_100k/range8_ef512
```

For independent range budgets at the query-relevant temporal scales, use the
multi-scale equal-nominal-degree configuration:

```bash
build-twitter-rt-hnsw \
  --output Twitter_data/ann/rolling_hnsw/flat_100k_multiscale \
  --semantic-degree 20 --range-degree-per-side 2 \
  --range-scales-hours 24,168,720

benchmark-twitter-rt-hnsw \
  --index Twitter_data/ann/rolling_hnsw/flat_100k_multiscale \
  --ef-search 512 \
  --output results/twitter_rt_hnsw_100k/multiscale_ef512
```

The default workload output is `results/twitter_rt_hnsw_100k/`. This remains a
flat research prototype, not a production FAISS HNSW replacement: it has no
adaptive gateways, late-arrival repair, expiration, or upper-layer temporal
chains yet.

## Test

```bash
python -m unittest discover -s tests
```
