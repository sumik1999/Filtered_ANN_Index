# Twitter temporal ANN data

This directory is the workspace for a one-year, at-most-one-million-record
Twitter/X temporal similarity benchmark. Dataset artifacts are intentionally
ignored by Git; only documentation and empty directory markers are tracked.

## Selected source

**ExponentialScience/DLT-Tweets**

- Dataset: <https://huggingface.co/datasets/ExponentialScience/DLT-Tweets>
- Revision: `c7d979394488bbd94bd6f493af1775d6b6159176`
- License reported by its dataset card: CC-BY-NC 4.0
- Time range: `[2021-01-01T00:00:00Z, 2022-01-01T00:00:00Z)`
- Language: English
- Retweets: conventional `RT @...` records removed
- Maximum selected records: 1,000,000, including 10,000 held-out queries

DLT-Tweets contains text and timestamps, so no Twitter API hydration is needed.
It is deduplicated and usernames have been removed by its creators. Selection
uses the lowest deterministic MD5 keys, giving a uniform record sample that
preserves relative traffic density instead of flattening busy periods.

This is real event-driven Twitter traffic, but it represents cryptocurrency and
blockchain discussion rather than an unbiased platform-wide sample. Results
must be described as a domain-specific real-world workload.

## Layout

```text
Twitter_data/
├── ids/          # downloaded public ID files
├── raw/          # hydrated API JSONL
├── processed/    # vectors, timestamp arrays, split IDs, metadata
├── workloads/    # temporal query predicates
├── sql/          # generated DuckDB timestamp index
├── ann/          # generated timestamp cache and global HNSW index
├── schema.sql    # logical benchmark schema
└── README.md
```

## ANN record design

Each successfully hydrated tweet becomes one immutable object:

```text
x_i = (embedding(text_i), created_at_i)
```

`vector_id` is a dense zero-based row number shared by the vector matrix and
timestamp table. `tweet_id` is retained as a string because Twitter snowflake
IDs do not safely fit every signed integer representation. The timestamp index
contains only fields needed for filtering; text and vectors remain outside
DuckDB.

Proposed processed files:

- `database_vectors.npy`: L2-normalized float32 vectors, shape `(N, D)`.
- `database_timestamps_ms.npy`: UTC Unix milliseconds, int64, shape `(N,)`.
- `database_records.jsonl`: vector ID, tweet ID, timestamp, language, and flags.
- `query_vectors.npy`: held-out query tweet vectors.
- `query_records.jsonl`: held-out query metadata.
- `metadata.json`: provenance, hashes, model version, filtering, and attrition.

Queries must be held out from the database to prevent trivial self-matches.
The default embedding proposal is `sentence-transformers/all-MiniLM-L6-v2`
(384 dimensions), L2 normalized. Squared L2 then has the same ordering as
cosine similarity.

## Exact temporal pre-filter baseline

For each query and half-open time predicate `[start_ms, end_ms)`:

1. Use the DuckDB timestamp index to obtain eligible `vector_id` values.
2. Read those rows from `database_vectors.npy`.
3. Compute exact squared L2 distances in bounded blocks.
4. Retain the exact top-k and report filtering, distance, and total latency.

Workloads will include ordinary, traffic-weighted, burst, and quiet periods at
multiple durations and selectivities. This preserves the distinction between
temporal filtering and vector similarity.

## Run the pipeline

Install optional model/download dependencies:

```bash
pip install -e '.[twitter]'
```

Then run each reproducible stage:

```bash
download-twitter-dlt
prepare-twitter-dlt --max-records 1000000 --query-count 10000
embed-twitter-dlt --batch-size 256
benchmark-twitter-exact --query-count 1000
build-twitter-hnsw --m 32 --ef-construction 200 --threads 4
benchmark-twitter-hnsw-post-filter --ef-search 256,512 --overfetch 10,50,100
generate-twitter-varied-workload --records-per-cell 20 --clusters 64 --seed 7
benchmark-twitter-varied-exact
benchmark-twitter-varied-global-hnsw
build-twitter-bucket-hnsw --strategies fixed,density --bucket-counts 32,64,128
benchmark-twitter-bucket-hnsw --strategies fixed,density --bucket-counts 32,64,128
benchmark-twitter-adaptive --strategy density --bucket-count 64
```

Run the exact command before the HNSW benchmark: it writes
`results/twitter_exact/ground_truth.npz`, which is used to report recall. The
HNSW implementation performs global vector retrieval and then applies the exact
half-open timestamp predicate as a post-filter.

The download is several gigabytes. Encoding nearly one million tweets is much
faster on a CUDA or Apple Silicon accelerator; pass an explicit device with
`--device` if needed. The benchmark defaults to 1, 6, 24, 168, and 720-hour
half-open windows and reports burst, ordinary, and quiet traffic separately.

## Reproducibility and compliance

Do not commit source text or generated embeddings. Preserve the source revision,
selection metadata, model name, dataset citation, and license in reports. Use
this CC-BY-NC dataset only in ways allowed by its license and dataset terms.
