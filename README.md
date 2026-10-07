# Temporal ANN benchmark

An initial benchmark harness for evaluating vector search over timestamped objects. It currently establishes the workload, correctness oracle, measurements, and reference query strategies before adding a real ANN backend.

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

## Test

```bash
python -m unittest discover -s tests
```
