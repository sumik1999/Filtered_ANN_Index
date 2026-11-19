# Reproducing the temporal-filtered HNSW experiments

This README gives a clean, end-to-end procedure for reproducing the results in
the final project report:

- density-balanced bucket HNSW on 990,000 Twitter vectors;
- flat multi-scale RT-HNSW on the earliest 100,000 vectors;
- coverage-bounded RT-HNSW on the same 100,000-vector graph;
- the equal-computation and parameter-sensitivity experiments; and
- the optional Kafka incremental-update prototype.

The commands assume execution from the repository root. Generated data, ANN
indexes, and most result directories are intentionally ignored by Git.

---

## 1. Scope and comparability

The report uses two different evaluation panels. Preserve this separation when
interpreting reproduced results.

| Panel | Database | Queries | Methods |
|---|---:|---:|---|
| Full Twitter | 990,000 | 9,600 | Density-64 bucket HNSW |
| Rolling graph | 100,000 | 900 | Flat and coverage-bounded RT-HNSW |

Do **not** directly compare absolute latency, storage, or recall between panels.
Flat and coverage-bounded RT-HNSW are directly comparable because they use the
same graph, workload, vectors, and exact ground truth.

The workload uses 384-dimensional, L2-normalized
`sentence-transformers/all-MiniLM-L6-v2` embeddings and squared L2 distance.
Temporal predicates are strict half-open intervals `[start_ms, end_ms)`.

---

## 2. Repository versions

### Main experiment branch

```bash
git clone https://github.com/sumik1999/Filtered_ANN_Index.git
cd Filtered_ANN_Index
git fetch --all --tags
git checkout RT-HNSW-coverage-aware
```

Record the exact revision used:

```bash
git rev-parse HEAD
git status --short
```

The coverage implementation entered the branch in commit `b73582b`; the full
coverage analysis was added in `69d9904`. Later commits update only reports and
reproducibility material unless otherwise indicated.

### Kafka branch

The Kafka writer and live snapshot service are on a separate lineage:

```text
RT-HNSW-kafka-ingestion
```

Use a worktree rather than repeatedly switching branches:

```bash
git worktree add ../Filtered_ANN_Index-kafka RT-HNSW-kafka-ingestion
```

Kafka setup is described in Section 13.

---

## 3. Hardware and software requirements

### Required

- Python 3.10 or newer; Python 3.11 or 3.12 is recommended.
- A 64-bit operating system.
- At least 16 GB RAM recommended for the full 990k experiment.
- At least 15 GB free disk space for raw data, vectors, indexes, and results.
- Internet access for the dataset and embedding model.

### Optional

- Apple Silicon MPS, CUDA, or another supported accelerator for embedding.
- Docker for the Kafka experiment.
- `jq` for convenient JSON inspection.

### Reference artifact sizes and times

These values are observations, not strict acceptance criteria:

| Stage/artifact | Reference value |
|---|---:|
| Embedding 1,000,000 selected records on MPS | 986.98 s |
| Global 990k HNSW construction | 288.57 s |
| Density-64 bucket construction | 52.67 s |
| Flat 100k RT-HNSW construction | 133.09 s |
| Global FAISS index | 1.790 GB |
| Density-64 bucket HNSWs | 1.790 GB plus 15.9 MB maps/timestamps |
| Flat RT-HNSW custom graph | 15.6 MB, vectors excluded |

Full benchmark duration depends strongly on CPU, storage, cache state, and
FAISS build.

---

## 4. Python environment

Create an isolated environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e '.[twitter]'
```

Base dependencies are declared in `pyproject.toml`:

- NumPy;
- HDF5 support;
- DuckDB; and
- FAISS CPU.

The `twitter` extra installs Hugging Face and Sentence Transformers support.

Verify imports and capture versions:

```bash
python - <<'PY'
import platform
import duckdb
import faiss
import numpy
print("Python:", platform.python_version())
print("NumPy:", numpy.__version__)
print("DuckDB:", duckdb.__version__)
print("FAISS:", getattr(faiss, "__version__", "unknown"))
PY

python -m pip freeze > reproduction_environment.txt
```

All CLI entry points should now be visible:

```bash
build-twitter-bucket-hnsw --help
build-twitter-rt-hnsw --help
benchmark-twitter-rt-hnsw --help
```

If shell entry points are unavailable, prepend `PYTHONPATH=src` or reinstall
with `python -m pip install -e .`.

---

## 5. Run unit tests first

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

The coverage branch reference run reports 47 passing tests. Important checks
include:

- half-open temporal boundaries;
- deterministic chronological IDs;
- exact fallback equivalence;
- disjoint interval unions;
- persisted range and coverage metadata;
- mandatory block coverage; and
- vector-row auditing proving that strict search does not score out-of-range
  rows.

Stop here if tests fail.

---

## 6. Download and prepare the Twitter data

### Source

The experiment uses:

```text
ExponentialScience/DLT-Tweets
revision c7d979394488bbd94bd6f493af1775d6b6159176
license reported by dataset card: CC-BY-NC 4.0
```

The selected time range is the half-open year 2021. Conventional `RT @...`
records are excluded. Selection uses deterministic MD5 keys, preserving traffic
density while selecting one million records.

### Download

```bash
download-twitter-dlt \
  --revision c7d979394488bbd94bd6f493af1775d6b6159176
```

The download is several gigabytes.

### Prepare deterministic database/query splits

```bash
prepare-twitter-dlt \
  --year 2021 \
  --revision c7d979394488bbd94bd6f493af1775d6b6159176 \
  --max-records 1000000 \
  --query-count 10000 \
  --threads 4
```

Expected split:

```text
990,000 database records
10,000 held-out query records
```

### Embed

CPU:

```bash
embed-twitter-dlt --batch-size 256 --device cpu
```

Apple Silicon example:

```bash
embed-twitter-dlt --batch-size 256 --device mps
```

CUDA example:

```bash
embed-twitter-dlt --batch-size 256 --device cuda
```

The command uses normalized `all-MiniLM-L6-v2` embeddings. Do not change model,
normalization, or dimension if exact reproduction is required.

### Materialize the timestamp cache

The ANN experiments use a memory-mapped int64 timestamp array derived from the
prepared Parquet records:

```bash
python - <<'PY'
from pathlib import Path
from temporal_ann.twitter_ann.hnsw import load_timestamps

values = load_timestamps(
    Path("Twitter_data/processed/database_records.parquet"),
    Path("Twitter_data/ann/timestamps_ms.npy"),
)
print("wrote", len(values), "timestamps")
PY
```

### Verify prepared artifacts

```bash
python - <<'PY'
import json
import numpy as np
from pathlib import Path

root = Path("Twitter_data")
db = np.load(root / "processed/database_vectors.npy", mmap_mode="r")
queries = np.load(root / "processed/query_vectors.npy", mmap_mode="r")
timestamps = np.load(root / "ann/timestamps_ms.npy", mmap_mode="r")
metadata = json.loads((root / "processed/metadata.json").read_text())
embedding = json.loads((root / "processed/embedding_metadata.json").read_text())

assert db.shape == (990_000, 384), db.shape
assert queries.shape == (10_000, 384), queries.shape
assert timestamps.shape == (990_000,), timestamps.shape
assert db.dtype == np.float32
assert timestamps.dtype == np.int64
assert metadata["source"]["revision"] == "c7d979394488bbd94bd6f493af1775d6b6159176"
assert embedding["model"] == "sentence-transformers/all-MiniLM-L6-v2"
assert embedding["normalization"] == "L2"
print("Twitter artifacts are valid")
PY
```

---

## 7. Create an isolated reproduction workspace

The commands below avoid overwriting the original checked-out result artifacts:

```bash
export REPRO="$PWD/reproduction"
mkdir -p \
  "$REPRO/ann" \
  "$REPRO/workloads" \
  "$REPRO/results"
```

Keep four controls fixed for deterministic comparison:

```text
FAISS threads = 4
workload seed = 7 for the full panel
workload seed = 19 for the rolling panel
embedding model and normalization unchanged
stable timestamp ordering unchanged
```

Recommended execution discipline:

```bash
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
```

Run latency benchmarks sequentially, not in parallel. Record whether the first
run is cold or warm, and avoid heavy background workloads. Redirect long-run
logs without hiding failures, for example:

```bash
some-benchmark-command 2>&1 | tee "$REPRO/results/run.log"
```

If the canonical generated artifacts already exist locally, verify them with
the checks below and skip the expensive download/embedding stages. Do not assume
Git contains vectors or indexes; these artifacts are ignored intentionally.

FAISS floating-point and HNSW construction details can vary slightly by version,
CPU, and thread scheduling. Recall and latency should therefore be compared
within documented tolerances rather than by requiring byte-identical indexes.

---

## 8. Reproduce the 990k density-balanced bucket experiment

### 8.1 Build the global HNSW used by workload generation

The balanced workload generator uses a global semantic HNSW to characterize
query/vector relationships.

```bash
build-twitter-hnsw \
  --embeddings Twitter_data/processed \
  --index "$REPRO/ann/hnsw_m32_efc200.index" \
  --metadata "$REPRO/ann/hnsw_metadata.json" \
  --m 32 \
  --ef-construction 200 \
  --threads 4 \
  --batch-size 50000 \
  --rebuild
```

Expected metadata:

```text
vector_count = 990000
dimensions = 384
M = 32
efConstruction = 200
```

### 8.2 Generate the balanced 9,600-query workload

```bash
generate-twitter-varied-workload \
  --records Twitter_data/processed \
  --embeddings Twitter_data/processed \
  --timestamps Twitter_data/ann/timestamps_ms.npy \
  --index "$REPRO/ann/hnsw_m32_efc200.index" \
  --output "$REPRO/workloads/temporal_vector_queries_9600.jsonl" \
  --summary "$REPRO/workloads/temporal_vector_queries_9600_summary.json" \
  --spatial-metadata "$REPRO/workloads/query_spatial_metadata.npz" \
  --centroids "$REPRO/workloads/semantic_centroids.npy" \
  --config "$REPRO/workloads/generation_config.json" \
  --coverage-hours 1,6,24,168,720 \
  --records-per-cell 20 \
  --k 100 \
  --neighbors 32 \
  --spatial-ef-search 512 \
  --clusters 64 \
  --kmeans-sample-size 100000 \
  --kmeans-iterations 20 \
  --threads 4 \
  --seed 7
```

Verify query count:

```bash
wc -l "$REPRO/workloads/temporal_vector_queries_9600.jsonl"
# Expected: 9600
```

### 8.3 Build exact ground truth

```bash
benchmark-twitter-varied-exact \
  --workload "$REPRO/workloads/temporal_vector_queries_9600.jsonl" \
  --embeddings Twitter_data/processed \
  --timestamps Twitter_data/ann/timestamps_ms.npy \
  --output "$REPRO/results/twitter_varied_exact" \
  --block-size 65536 \
  --progress-every 100
```

Required output:

```text
$REPRO/results/twitter_varied_exact/ground_truth.npz
```

### 8.4 Build the density-64 bucket index

```bash
build-twitter-bucket-hnsw \
  --embeddings Twitter_data/processed \
  --timestamps Twitter_data/ann/timestamps_ms.npy \
  --index-root "$REPRO/ann/buckets" \
  --strategies density \
  --bucket-counts 64 \
  --m 32 \
  --ef-construction 200 \
  --threads 4 \
  --rebuild
```

Expected directory:

```text
$REPRO/ann/buckets/density/buckets_64_m32_efc200/
```

### 8.5 Run the final bucket configuration

```bash
benchmark-twitter-bucket-hnsw \
  --workload "$REPRO/workloads/temporal_vector_queries_9600.jsonl" \
  --embeddings Twitter_data/processed \
  --timestamps Twitter_data/ann/timestamps_ms.npy \
  --ground-truth "$REPRO/results/twitter_varied_exact/ground_truth.npz" \
  --index-root "$REPRO/ann/buckets" \
  --output "$REPRO/results/twitter_bucket_hnsw_density64" \
  --strategies density \
  --bucket-counts 64 \
  --m 32 \
  --ef-construction 200 \
  --ef-search 128 \
  --overfetch 2 \
  --boundary-safety 1.25 \
  --threads 4 \
  --progress-every 500
```

### 8.6 Expected bucket results

Reference values from
`results/twitter_bucket_hnsw_density_final/summary.json`:

| Metric | Reference |
|---|---:|
| Recall@100 | 0.99908 |
| Mean latency | 144.18 ms |
| p50 latency | 44.43 ms |
| p95 latency | 566.15 ms |
| Mean distance computations | 35,811 |
| Mean selected buckets | 3.89 |
| Mean partial buckets | 3.26 |
| Complete-result rate | 0.9329 |

Latency can differ substantially across hardware and cache state. Recall should
remain close; investigate differences larger than approximately 0.002 before
accepting the reproduction.

---

## 9. Reproduce the 100k flat multi-scale RT-HNSW index

### 9.1 Build the graph

```bash
build-twitter-rt-hnsw \
  --embeddings Twitter_data/processed/database_vectors.npy \
  --timestamps Twitter_data/ann/timestamps_ms.npy \
  --output "$REPRO/ann/flat_100k_multiscale" \
  --vector-count 100000 \
  --semantic-degree 20 \
  --gateway-degree 8 \
  --gateway-period 512 \
  --construction-candidates 64 \
  --ef-construction 200 \
  --candidate-index-m 32 \
  --threads 4 \
  --range-degree-per-side 2 \
  --range-candidate-count 64 \
  --range-scales-hours 24,168,720 \
  --coverage-block-size 512 \
  --coverage-gateways-per-block 4 \
  --progress-every 5000 \
  --rebuild
```

This single build contains the flat range-aware topology and coverage metadata.
Coverage changes query scheduling, not the underlying graph.

Verify the manifest:

```bash
python - <<'PY'
import json, os
from pathlib import Path
root = Path(os.environ["REPRO"]) / "ann/flat_100k_multiscale"
m = json.loads((root / "manifest.json").read_text())
assert m["vector_count"] == 100_000
assert m["dimensions"] == 384
assert m["config"]["semantic_degree"] == 20
assert m["config"]["range_degree_per_side"] == 2
assert m["range_scales_hours"] == [24, 168, 720]
assert m["config"]["coverage_block_size"] == 512
print("RT-HNSW manifest is valid")
PY
```

### 9.2 Generate the matched 900-query workload and exact oracle

```bash
generate-twitter-rt-workload \
  --index "$REPRO/ann/flat_100k_multiscale" \
  --vectors Twitter_data/processed/database_vectors.npy \
  --queries Twitter_data/processed/query_vectors.npy \
  --workload "$REPRO/workloads/rt_hnsw_100k_queries.jsonl" \
  --ground-truth "$REPRO/results/rt_hnsw_ground_truth.npz" \
  --query-count 900 \
  --coverage-hours 1,6,24,168,720 \
  --fragment-counts 1,2,4 \
  --seed 19 \
  --block-size 65536 \
  --progress-every 100
```

Verify:

```bash
wc -l "$REPRO/workloads/rt_hnsw_100k_queries.jsonl"
# Expected: 900
```

### 9.3 Run the controlled flat-versus-coverage comparison

```bash
benchmark-twitter-rt-hnsw \
  --index "$REPRO/ann/flat_100k_multiscale" \
  --vectors Twitter_data/processed/database_vectors.npy \
  --queries Twitter_data/processed/query_vectors.npy \
  --workload "$REPRO/workloads/rt_hnsw_100k_queries.jsonl" \
  --ground-truth "$REPRO/results/rt_hnsw_ground_truth.npz" \
  --output "$REPRO/results/coverage_ef256" \
  --variants semantic_temporal_range,coverage_multiscale \
  --ef-search 256 \
  --exact-threshold 5000 \
  --gateway-candidates 16 \
  --gateway-seeds 4 \
  --coverage-expansions-per-block 1 \
  --coverage-gateway-candidates 4 \
  --progress-every 100
```

Reference result:

| Metric | Flat | Coverage |
|---|---:|---:|
| Recall@100 | 0.95143 | 0.95273 |
| Distance computations | 2,741.9 | 2,979.0 |
| Ground-truth block coverage | 0.99924 | 1.00000 |
| Coverage blocks considered/covered | 0/0 | 37.284/37.284 |

Every coverage query must satisfy:

```text
coverage_blocks_considered == coverage_blocks_covered
```

### 9.4 Reproduce the established flat RT-HNSW point

The report's approximately 0.9515 result includes the gateway-range variant:

```bash
benchmark-twitter-rt-hnsw \
  --index "$REPRO/ann/flat_100k_multiscale" \
  --vectors Twitter_data/processed/database_vectors.npy \
  --queries Twitter_data/processed/query_vectors.npy \
  --workload "$REPRO/workloads/rt_hnsw_100k_queries.jsonl" \
  --ground-truth "$REPRO/results/rt_hnsw_ground_truth.npz" \
  --output "$REPRO/results/flat_gateway_range_ef256" \
  --variants semantic_temporal_gateway_range \
  --ef-search 256 \
  --exact-threshold 5000 \
  --gateway-candidates 16 \
  --gateway-seeds 4 \
  --progress-every 100
```

Reference Recall@100: approximately `0.95146`.

---

## 10. Reproduce the beam-width and equal-computation experiments

### 10.1 Flat and coverage beam sweep

```bash
for EF in 128 256 512; do
  benchmark-twitter-rt-hnsw \
    --index "$REPRO/ann/flat_100k_multiscale" \
    --vectors Twitter_data/processed/database_vectors.npy \
    --queries Twitter_data/processed/query_vectors.npy \
    --workload "$REPRO/workloads/rt_hnsw_100k_queries.jsonl" \
    --ground-truth "$REPRO/results/rt_hnsw_ground_truth.npz" \
    --output "$REPRO/results/coverage_ef${EF}" \
    --variants semantic_temporal_range,coverage_multiscale \
    --ef-search "$EF" \
    --exact-threshold 5000 \
    --coverage-expansions-per-block 1 \
    --coverage-gateway-candidates 4 \
    --progress-every 100
done
```

Expected recall/computation relationship:

| efSearch | Flat recall | Coverage recall | Flat computations | Coverage computations |
|---:|---:|---:|---:|---:|
| 128 | 0.91726 | 0.91898 | 2,134.5 | 2,379.1 |
| 256 | 0.95143 | 0.95273 | 2,741.9 | 2,979.0 |
| 512 | 0.97410 | 0.97440 | 3,823.8 | 4,046.1 |

### 10.2 Equal-computation coverage point

Flat ef256 is already present in `coverage_ef256`. Run coverage at ef205:

```bash
benchmark-twitter-rt-hnsw \
  --index "$REPRO/ann/flat_100k_multiscale" \
  --vectors Twitter_data/processed/database_vectors.npy \
  --queries Twitter_data/processed/query_vectors.npy \
  --workload "$REPRO/workloads/rt_hnsw_100k_queries.jsonl" \
  --ground-truth "$REPRO/results/rt_hnsw_ground_truth.npz" \
  --output "$REPRO/results/coverage_ef205_equal_compute" \
  --variants coverage_multiscale \
  --ef-search 205 \
  --exact-threshold 5000 \
  --coverage-expansions-per-block 1 \
  --coverage-gateway-candidates 4 \
  --progress-every 100
```

Expected comparison:

| Method | Recall | Computations |
|---|---:|---:|
| Flat ef256 | 0.95143 | 2,741.9 |
| Coverage ef205 | 0.94317 | 2,746.8 |

This is the experiment that rejects coverage as the default equal-budget search
policy.

---

## 11. Reproduce coverage parameter sensitivity

### 11.1 Mandatory expansions

```bash
for E in 1 2 5; do
  benchmark-twitter-rt-hnsw \
    --index "$REPRO/ann/flat_100k_multiscale" \
    --vectors Twitter_data/processed/database_vectors.npy \
    --queries Twitter_data/processed/query_vectors.npy \
    --workload "$REPRO/workloads/rt_hnsw_100k_queries.jsonl" \
    --ground-truth "$REPRO/results/rt_hnsw_ground_truth.npz" \
    --output "$REPRO/results/coverage_ef256_e${E}" \
    --variants coverage_multiscale \
    --ef-search 256 \
    --exact-threshold 5000 \
    --coverage-expansions-per-block "$E" \
    --coverage-gateway-candidates 4 \
    --progress-every 100
done
```

Reference:

| Expansions | Recall | Computations |
|---:|---:|---:|
| 1 | 0.95273 | 2,979.0 |
| 2 | 0.95342 | 3,078.7 |
| 5 | 0.95553 | 3,369.9 |

### 11.2 Representatives scored per block

```bash
for G in 1 2 4; do
  benchmark-twitter-rt-hnsw \
    --index "$REPRO/ann/flat_100k_multiscale" \
    --vectors Twitter_data/processed/database_vectors.npy \
    --queries Twitter_data/processed/query_vectors.npy \
    --workload "$REPRO/workloads/rt_hnsw_100k_queries.jsonl" \
    --ground-truth "$REPRO/results/rt_hnsw_ground_truth.npz" \
    --output "$REPRO/results/coverage_ef256_g${G}" \
    --variants coverage_multiscale \
    --ef-search 256 \
    --exact-threshold 5000 \
    --coverage-expansions-per-block 1 \
    --coverage-gateway-candidates "$G" \
    --progress-every 100
done
```

Reference:

| Representatives | Recall | Computations |
|---:|---:|---:|
| 1 | 0.95196 | 2,828.8 |
| 2 | 0.95249 | 2,877.4 |
| 4 | 0.95273 | 2,979.0 |

### 11.3 Block-size sensitivity

Block size is persisted build metadata. For a strict public-CLI reproduction,
build one index per block size. The semantic/range graph configuration remains
identical.

```bash
for S in 256 512 1000 2048; do
  build-twitter-rt-hnsw \
    --embeddings Twitter_data/processed/database_vectors.npy \
    --timestamps Twitter_data/ann/timestamps_ms.npy \
    --output "$REPRO/ann/flat_100k_block${S}" \
    --vector-count 100000 \
    --semantic-degree 20 \
    --gateway-degree 8 \
    --gateway-period 512 \
    --construction-candidates 64 \
    --ef-construction 200 \
    --candidate-index-m 32 \
    --threads 4 \
    --range-degree-per-side 2 \
    --range-candidate-count 64 \
    --range-scales-hours 24,168,720 \
    --coverage-block-size "$S" \
    --coverage-gateways-per-block 4 \
    --progress-every 5000 \
    --rebuild

  benchmark-twitter-rt-hnsw \
    --index "$REPRO/ann/flat_100k_block${S}" \
    --vectors Twitter_data/processed/database_vectors.npy \
    --queries Twitter_data/processed/query_vectors.npy \
    --workload "$REPRO/workloads/rt_hnsw_100k_queries.jsonl" \
    --ground-truth "$REPRO/results/rt_hnsw_ground_truth.npz" \
    --output "$REPRO/results/coverage_block${S}_ef256" \
    --variants coverage_multiscale \
    --ef-search 256 \
    --exact-threshold 5000 \
    --coverage-expansions-per-block 1 \
    --coverage-gateway-candidates 4 \
    --progress-every 100
done
```

Reference:

| Block size | Recall | Computations | Mean blocks considered |
|---:|---:|---:|---:|
| 256 | 0.95387 | 3,208.8 | 73.82 |
| 512 | 0.95273 | 2,979.0 | 37.28 |
| 1,000 | 0.95203 | 2,861.7 | 19.45 |
| 2,048 | 0.95172 | 2,800.0 | 9.88 |

---

## 12. Validate reproduced summaries automatically

Run this after the primary bucket and ef256 experiments:

```bash
python - <<'PY'
import json
import math
import os
from pathlib import Path

root = Path(os.environ["REPRO"]) / "results"

bucket = json.loads(
    (root / "twitter_bucket_hnsw_density64/summary.json").read_text()
)["overall"]
coverage = json.loads(
    (root / "coverage_ef256/summary.json").read_text()
)["overall"]

flat = coverage["semantic_temporal_range"]
cov = coverage["coverage_multiscale"]

def close(name, value, expected, tolerance):
    if not math.isclose(value, expected, abs_tol=tolerance):
        raise AssertionError(f"{name}: {value} != {expected} ± {tolerance}")
    print(f"PASS {name}: {value}")

close("bucket recall", bucket["mean_recall_at_k"], 0.99908, 0.002)
close("flat recall", flat["mean_recall_at_k"], 0.95143, 0.003)
close("coverage recall", cov["mean_recall_at_k"], 0.95273, 0.003)
close("flat computations", flat["mean_distance_computations"], 2741.9, 75)
close("coverage computations", cov["mean_distance_computations"], 2979.0, 75)
assert cov["mean_coverage_blocks_considered"] == cov["mean_coverage_blocks_covered"]
print("PASS mandatory coverage invariant")
PY
```

Do not assert exact latency. Report the observed hardware, software versions,
thread count, whether caches were warm, and variant execution order.

---

## 13. Optional Kafka online-update reproduction

The Kafka implementation is not on the coverage branch. Enter the worktree
created in Section 2:

```bash
cd ../Filtered_ANN_Index-kafka
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e '.[kafka]'
```

### 13.1 Start the development broker

```bash
docker compose -f deploy/kafka/docker-compose.yml up -d
```

The compose file starts a single KRaft broker and a `kafka-init` service that
creates:

```text
twitter.raw.v1
twitter.embedded.v1
twitter.index.ordered.v1
twitter.index.late.v1
twitter.index.dlq.v1
twitter.orderer.state.v1
twitter.index.checkpoints.v1
```

Check health and topic initialization:

```bash
docker compose -f deploy/kafka/docker-compose.yml ps
docker compose -f deploy/kafka/docker-compose.yml logs kafka-init
```

### 13.2 Run the pipeline

Use separate terminals, each with the Kafka worktree virtual environment active.

Terminal 1 — embedding worker:

```bash
kafka-twitter-embedder --index-name twitter-rt-hnsw
```

Terminal 2 — event-time orderer:

```bash
kafka-twitter-orderer \
  --index-name twitter-rt-hnsw \
  --allowed-lateness-ms 600000
```

Terminal 3 — durable index writer:

```bash
kafka-twitter-index-writer \
  --index-name twitter-rt-hnsw \
  --index Twitter_data/ann/streaming_rt_hnsw
```

Terminal 4 — live query service:

```bash
serve-twitter-rt-hnsw \
  --index Twitter_data/ann/streaming_rt_hnsw \
  --port 8080
```

Terminal 5 — source producer:

```bash
kafka-twitter-producer --input new_tweets.jsonl
```

The development embedder is deterministic but is not a production MiniLM
replacement. For graph equivalence with offline Twitter experiments, replay the
same precomputed embeddings or use exactly the same embedding model and
normalization.

### 13.3 Query the live service

```bash
curl -X POST http://127.0.0.1:8080/search \
  -H 'Content-Type: application/json' \
  -d '{
    "query": [0.0],
    "intervals": [[1609459200000, 1640995200000]],
    "k": 100,
    "ef_search": 256,
    "min_sequence": 0,
    "wait_timeout_ms": 2000
  }'
```

Replace `[0.0]` with one finite 384-dimensional query vector.

Health endpoints:

```bash
curl http://127.0.0.1:8080/health
curl http://127.0.0.1:8080/status
curl -X POST http://127.0.0.1:8080/refresh
```

### 13.4 Stop Kafka

Preserve broker state:

```bash
docker compose -f deploy/kafka/docker-compose.yml down
```

Delete broker state and restart from empty topics:

```bash
docker compose -f deploy/kafka/docker-compose.yml down -v
```

### 13.5 Kafka validation status

Unit and differential tests cover ordering, idempotence, WAL recovery,
immutable publication, and live snapshot pinning. Broker-level transaction
abort and process-kill testing was not completed in the reference environment
because Docker was unavailable. Do not describe the Kafka path as
production-validated until those tests pass on a real broker.

---

## 14. Output files and interpretation

Each benchmark output directory contains:

```text
queries.csv     one row per query and variant
summary.json    aggregate and factor-slice metrics
```

During long runs, a partial CSV may be written and replaced after completion.
Index directories contain a `manifest.json` recording construction parameters,
source paths, vector count, dimensions, timestamp order, file names, build time,
and storage size.

Important fields:

```text
mean_recall_at_k
mean_total_latency_ms
p50_total_latency_ms
p95_total_latency_ms
mean_distance_computations
mean_nodes_visited
mean_coverage_blocks_considered
mean_coverage_blocks_covered
mean_ground_truth_block_coverage
```

The complete-result rate can be below one when fewer than 100 eligible records
exist; this is not necessarily an ANN failure.

---

## 15. Common problems

### `pytest` is unavailable

The project uses standard-library `unittest` for the documented suite:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

### FAISS cannot be imported

Use Python 3.11/3.12 and recreate the environment. Confirm that `faiss-cpu` is
available for the platform.

### Commands are missing after installation

```bash
python -m pip install -e .
hash -r
```

Or run with the environment's explicit scripts:

```bash
.venv/bin/build-twitter-rt-hnsw --help
```

### Existing index configuration mismatch

Builders refuse to reuse an output directory with different persisted
parameters. Use a new output path or pass `--rebuild`.

### Out of memory during embedding or exact search

Reduce embedding batch size. Exact search already uses blocked computation;
reduce `--block-size` if necessary.

### Different latency but similar recall

This is expected across hardware, FAISS versions, cache state, and background
load. Repeat runs after a warm-up, avoid concurrent benchmarks, and report
p50/p95 alongside deterministic work counters.

### Coverage invariant fails

Inspect `queries.csv` and confirm:

- `coverage_enabled` was selected through the `coverage_multiscale` variant;
- block metadata matches the index configuration;
- the exact threshold did not route the query to exact search; and
- intervals are valid half-open ranges.

### Kafka watermark stalls

Check all embedded-topic partitions and idle-partition handling. One active
partition with no advancing event time can hold the global watermark back.

---

## 16. Reproduction checklist

- [ ] Checked out `RT-HNSW-coverage-aware`.
- [ ] Recorded commit, Python, NumPy, FAISS, and DuckDB versions.
- [ ] Passed all unit tests.
- [ ] Used the documented DLT-Tweets revision.
- [ ] Produced 990,000 database and 10,000 held-out query vectors.
- [ ] Preserved MiniLM, 384 dimensions, L2 normalization, and squared L2.
- [ ] Generated the 9,600-query full workload with seed 7.
- [ ] Built exact full-panel ground truth.
- [ ] Built and evaluated density-64 bucket HNSW.
- [ ] Built the 100k multi-scale graph with seed/order/configuration unchanged.
- [ ] Generated the 900-query rolling workload with seed 19.
- [ ] Ran flat and coverage variants on the same graph and ground truth.
- [ ] Verified all considered coverage blocks were covered.
- [ ] Ran the equal-computation coverage experiment.
- [ ] Avoided direct numerical comparison between the 990k and 100k panels.
- [ ] Reported latency environment and cache protocol.
- [ ] Clearly labeled Kafka broker tests as optional/incomplete when not run.

---

## 17. Reference evidence

| Evidence | Repository path |
|---|---|
| Bucket final summary | `results/twitter_bucket_hnsw_density_final/summary.json` |
| Flat/coverage ef256 summary | `results/twitter_rt_hnsw_100k/coverage_ef256/summary.json` |
| Equal-computation summary | `results/twitter_rt_hnsw_100k/coverage_ef205_equal_compute/summary.json` |
| Coverage analysis | `results/twitter_rt_hnsw_100k/COVERAGE_AWARE_ANALYSIS.md` |
| Coverage design | `docs/COVERAGE_AWARE_RT_HNSW.md` |
| Multi-scale analysis | `results/twitter_rt_hnsw_100k/MULTISCALE_RANGE_ANALYSIS.md` |
| Dataset design | `Twitter_data/workloads/QUERY_DATASET_DESIGN.md` |
| Final five-page report | `results/FINAL_RESEARCH_PROJECT_REPORT_5_PAGE.md` |
| Submission DOCX/PDF | `deliverables/research_project_submission/` |
| Kafka architecture | `docs/KAFKA_INCREMENTAL_RT_HNSW_ARCHITECTURE.md` on `RT-HNSW-kafka-ingestion` |

For the final conclusions, the most important reproducibility checks are the
controlled flat-versus-coverage run and the equal-computation run. They establish
that mandatory coverage is structurally correct and repairs rare missed-region
queries, but should not replace flat multi-scale RT-HNSW as the default search
mechanism.
