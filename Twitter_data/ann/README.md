# Twitter ANN artifacts

This directory stores generated ANN-specific data for the prepared DLT-Tweets
sample. Large files are ignored by Git.

Generated files:

- `timestamps_ms.npy`: dense timestamp array ordered by `vector_id`.
- `hnsw_m32_efc200.index`: global FAISS `IndexHNSWFlat`.
- `hnsw_metadata.json`: construction parameters, size, timing, and checksum.

The source records and MiniLM vectors remain in `Twitter_data/processed/`.
Exact ground truth is written to `results/twitter_exact/ground_truth.npz` and
post-filter measurements to `results/twitter_hnsw_post_filter/`.
