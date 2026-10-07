# SIFT1M data

Source: [TensorFlow Datasets SIFT1M](https://www.tensorflow.org/datasets/catalog/sift1m)  
Underlying file: <https://ann-benchmarks.com/sift-128-euclidean.hdf5>

The downloaded HDF5 file is intentionally excluded from Git.

## Inspected contents

| HDF5 key | Shape | Type | Meaning |
|---|---:|---|---|
| `train` | 1,000,000 × 128 | `float32` | Database vectors |
| `test` | 10,000 × 128 | `float32` | Query vectors |
| `neighbors` | 10,000 × 100 | `int32` | Exact nearest database-vector IDs |
| `distances` | 10,000 × 100 | `float32` | Corresponding Euclidean distances |

The file is 525,128,288 bytes (500.80 MiB), has a `distance = euclidean` attribute, and contains no timestamp field. Its SHA-256 digest is recorded in `metadata.json`.

The vectors are unnormalized SIFT descriptors. In a sample of the first 10,000 vectors, database values ranged from 0 to 180, with mean 26.709 and standard deviation 36.165. The complete test split ranged from 0 to 184, with mean 26.708 and standard deviation 36.165.

The stored distances were checked against the vectors and are ordinary Euclidean (L2) distances, not squared L2 distances. Neighbor IDs are zero-based indices into `train`.

## Temporal benchmark implications

Only the one million database vectors need synthetic timestamps. The test vectors remain queries. Once a temporal predicate is introduced, the supplied unfiltered neighbors are no longer sufficient as temporal ground truth; exact filtered search must compute ground truth for each predicate.

Timestamp assignments are stored separately under `timestamps/`; the source HDF5 file is not modified. Three reproducible `float64` arrays have been generated over normalized range `[0, 1)`:

1. `uniform_seed-7.npy`: independent uniform timestamps;
2. `poisson_clustered_seed-7.npy`: clustered non-homogeneous Poisson timestamps; and
3. `zipf_seed-7.npy`: finite Zipf timestamps with exponent 1.2.

Array position is the corresponding row ID in `train`, and assignments are independent of vector values. `timestamps/config.json` preserves the generation configuration. `timestamps/metadata.json` records parameters, checksums, summary statistics, and the interpretation of `[0, 1)` as calendar year 2023 UTC. Regenerate them with:

```bash
generate-sift-timestamps
```

Use `generate-sift-timestamps --help` to change the seed, year, bin count, Zipf exponent, or output location. Projection-ranked timestamp assignment can be added later to produce controlled vector/time correlation while preserving each timestamp distribution.

## Exact temporal oracle

`workloads/temporal_queries.jsonl` contains 1,000 fixed SIFT test-query IDs and predicates. Every predicate has total width 0.10, represented by four disjoint half-open intervals of width 0.025 joined by `OR`. Collectively, the workload covers the entire normalized timeline.

`ground_truth/` contains exact filtered top-100 neighbors for each timestamp distribution. Ground-truth distances are squared Euclidean distances. See `ground_truth/config.json` for the fixed configuration and `ground_truth/metadata.json` for checksums, construction times, and selectivity statistics.

Regenerate the oracle with:

```bash
generate-sift-oracle
```

## Exact SQL pre-filter benchmark

The exact baseline stores vector IDs and all three timestamp columns in DuckDB. For every workload record it executes the four temporal intervals as an SQL `OR` filter, then runs a brute-force blocked squared-L2 scan over only the qualifying SIFT vectors. No ANN index is used.

Configuration is recorded in `sql/config.json`; raw and summarized measurements are written under `results/sift_exact/`. Run it with:

```bash
benchmark-sift-exact
```

A full benchmark using all 10,000 SIFT test vectors is also available:

- workload: `workloads/temporal_queries_10000.jsonl`
- oracle: `ground_truth_10000/`
- configuration: `sql/config_10000.json`
- results: `results/sift_exact_10000/`

Re-run the full exact benchmark with:

```bash
benchmark-sift-exact \
  --workload SIFT_data/workloads/temporal_queries_10000.jsonl \
  --ground-truth SIFT_data/ground_truth_10000 \
  --output results/sift_exact_10000
```
