# Temporal ANN benchmark

A research harness for temporal-filtered approximate nearest-neighbor search.
It includes synthetic and Twitter workloads, exact correctness oracles, global
and bucketed FAISS HNSW baselines, rolling temporal HNSW designs, coverage-aware
search, and separate experimental branches for strict-filter testing, Kafka
updates, and temporal bitmaps.

## Included

- Synthetic timestamped vectors with configurable temporal density and time/vector correlation.
- Normalized interval predicates with union and intersection operations.
- Workloads varying temporal width, interval fragmentation, `k`, density, correlation, and fixed partition granularity.
- Exact ground truth and recall@k measurement.
- Per-query latency, candidates examined, auxiliary index size, and build time.
- Reference temporal pre-filter, exact global post-filter proxy, and fixed temporal partition strategies.

The global post-filter implementation is deliberately named a **proxy**: it computes an exact global ranking and applies ANN-style over-fetch/filter semantics. Its recall and candidate counts are useful, but its latency is not representative of an ANN index. A real ANN backend is the next implementation step.

## Branch guide

The repository uses research branches rather than placing every experiment on
`main`. The branches are parallel lineages: the latest coverage report does not
also contain the Kafka, bitmap, or expanded strict-filter test work.

| Branch | Purpose | Relationship / when to use it |
|---|---|---|
| `main` | Original synthetic temporal-ANN harness and exact reference methods | Smallest stable starting point |
| `feature/hnsw-idselector` | FAISS HNSW filtering with `IDSelectorBitmap` | Independent experiment forked from `main` |
| `feature/hnsw-post-filter` | Global HNSW post-filtering, fixed temporal HNSW, and the Twitter baseline | Foundation of the RT-HNSW lineage |
| `RT-HNSW` | Flat RT-HNSW, range and multi-scale edges, hierarchy, upper-range experiment, and gateway oracle | Shared RT-HNSW research base for the later experimental branches |
| `RT-HNSW-filter-tests` | Rigorous half-open-filter, vector-access, differential, and graph-isolation tests | Use to audit strict filtering independently of later topology work |
| `RT-HNSW-kafka-ingestion` | Event-time ordering, transactional Kafka stages, durable incremental writer, immutable generations, and live snapshots | Use for online-update and recovery work |
| `RT-HNSW-temporal-bitmaps` | Packed epoch adjacency, temporal bitmaps, regional portals, and Kafka continuation | Extends the Kafka lineage with the bitmap topology |
| `RT-HNSW-coverage-aware` | Coverage blocks, mandatory regional exploration, evaluations, final reports, and reproducibility guide | Current report branch and recommended branch for the final project artifacts |

A simplified lineage is:

```text
main
├── feature/hnsw-idselector
└── feature/hnsw-post-filter
    └── RT-HNSW
        ├── RT-HNSW-filter-tests
        ├── RT-HNSW-kafka-ingestion
        │   └── RT-HNSW-temporal-bitmaps
        └── RT-HNSW-coverage-aware
```

> **Branch context:** `main` is intentionally the minimal synthetic harness.
> The detailed Twitter, RT-HNSW, coverage, and final-report instructions below
> describe `RT-HNSW-coverage-aware`. Check out that branch before running those
> commands. Kafka and bitmap commands require their respective branches.

Inspect the live local/remote status before choosing a branch:

```bash
git fetch --all --prune
git branch -a -vv
git log --all --graph --decorate --oneline --simplify-by-decoration
git worktree list
```

## Set up

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

The project description remains in `requirements.txt` as requested. Python package dependencies are declared in `pyproject.toml`.

For a complete clean-room reproduction of the Twitter dataset, density-64
bucket experiment, flat and coverage-bounded RT-HNSW experiments, equal-budget
sweeps, validation checks, and optional Kafka pipeline, see
[`REPRODUCIBILITY.md`](REPRODUCIBILITY.md).

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

Build the hierarchical extension with incremental upper-level semantic indexes,
protected chronological chains at every level, and strict in-range upper-layer
entry and descent:

```bash
build-twitter-rt-hnsw \
  --output Twitter_data/ann/rolling_hnsw/hierarchical_100k_multiscale \
  --semantic-degree 20 --range-degree-per-side 2 \
  --range-scales-hours 24,168,720 \
  --hierarchy --hierarchy-semantic-degree 8 \
  --hierarchy-level-probability 0.05 --hierarchy-max-level 6

benchmark-twitter-rt-hnsw \
  --index Twitter_data/ann/rolling_hnsw/hierarchical_100k_multiscale \
  --ef-search 512 --upper-ef-search 32 \
  --output results/twitter_rt_hnsw_100k/hierarchy_ef512
```

The fixed-storage upper range-aware experiment replaces the upper semantic-8
budget with four semantic and two left/right range-aware edges. Levels map to
24, 168 and 720-hour temporal scales:

```bash
build-twitter-rt-hnsw \
  --output Twitter_data/ann/rolling_hnsw/hierarchical_100k_upper_range \
  --semantic-degree 20 --range-degree-per-side 2 \
  --range-scales-hours 24,168,720 --hierarchy \
  --hierarchy-semantic-degree 4 \
  --hierarchy-range-degree-per-side 2 \
  --hierarchy-range-scales-hours 24,168,720
```

Evaluate periodic, random, semantic-cluster, temporal-density and greedy-oracle
entry directories at the same 195-gateway budget, without gateway edges:

```bash
benchmark-twitter-gateway-oracle \
  --index Twitter_data/ann/rolling_hnsw/flat_100k_multiscale \
  --candidate-step 64 --ef-search 256
```

The coverage-aware extension adds compact fixed-population temporal blocks,
conservative min/max timestamps, diverse representatives, and a mandatory
coverage queue. It preserves the existing strict graph and exact fallback while
ensuring every block intersecting a graph-searched component receives the
configured minimum number of expansions:

```bash
build-twitter-rt-hnsw \
  --output Twitter_data/ann/rolling_hnsw/flat_100k_coverage \
  --semantic-degree 20 --range-degree-per-side 2 \
  --range-scales-hours 24,168,720 \
  --coverage-block-size 512 --coverage-gateways-per-block 4

benchmark-twitter-rt-hnsw \
  --index Twitter_data/ann/rolling_hnsw/flat_100k_coverage \
  --variants semantic_temporal_range,coverage_multiscale \
  --ef-search 256 --coverage-expansions-per-block 1 \
  --coverage-gateway-candidates 4 \
  --output results/twitter_rt_hnsw_100k/coverage_ef256
```

The benchmark reports both temporal coverage diagnostics and the fraction of
blocks containing exact top-k neighbors that traversal visited. Block coverage
is not a guarantee of exact nearest-neighbor recall. See
`docs/COVERAGE_AWARE_RT_HNSW.md` for the invariants and experiment protocol.

The default workload output is `results/twitter_rt_hnsw_100k/`. This remains a
research prototype, not a production FAISS HNSW replacement: it has no late-
arrival repair, expiration, or concurrent update support. The gated adaptive
gateway plan and failed oracle acceptance result are documented in
`docs/ADAPTIVE_GATEWAY_PLAN.md`.

## Test

Run the checked-out branch's unit tests and a self-contained synthetic smoke
benchmark:

```bash
scripts/test-current-branch.sh all
```

The smoke benchmark writes only to a temporary directory and removes it on
exit. Individual modes are available:

```bash
scripts/test-current-branch.sh unit
scripts/test-current-branch.sh smoke
```

Set a specific interpreter when needed:

```bash
PYTHON=.venv/bin/python scripts/test-current-branch.sh all
```

The direct unit-test command remains:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

### Test another branch without disturbing this checkout

Use a detached worktree, create its environment, run its version of the tests,
and remove it afterward:

```bash
BRANCH=RT-HNSW-kafka-ingestion
WORKTREE="../temporal-ann-test-${BRANCH}"

git worktree add --detach "$WORKTREE" "$BRANCH"
python3 -m venv "$WORKTREE/.venv"
"$WORKTREE/.venv/bin/python" -m pip install -e "$WORKTREE[twitter,kafka]"
(
  cd "$WORKTREE"
  PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
  PYTHONPATH=src .venv/bin/python -m temporal_ann.benchmark \
    --size 2000 --dimensions 16 --queries 3 \
    --widths 0.01,0.1 --fragments 1,4 --ks 10 \
    --data-dir /tmp/temporal-ann-data \
    --output /tmp/temporal-ann-results
)
git worktree remove "$WORKTREE"
```

Use `-e "$WORKTREE"` for `main` and the early feature branches, or
`-e "$WORKTREE[twitter]"` for Twitter/RT-HNSW branches without Kafka. Docker is
required only for broker-level Kafka integration; unit tests do not require a
running broker. The complete data and experiment reproduction procedure is in
[`REPRODUCIBILITY.md`](REPRODUCIBILITY.md).
