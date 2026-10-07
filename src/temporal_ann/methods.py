from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .core import Dataset, SearchResult, TemporalPredicate, exact_search, squared_l2, top_k


class SearchMethod(ABC):
    name: str

    @property
    @abstractmethod
    def index_size_bytes(self) -> int:
        """Return auxiliary index bytes, excluding the source dataset."""

    @abstractmethod
    def search(
        self, query: np.ndarray, predicate: TemporalPredicate, k: int
    ) -> SearchResult:
        pass


class TemporalPreFilter(SearchMethod):
    name = "temporal_pre_filter"

    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    @property
    def index_size_bytes(self) -> int:
        return 0

    def search(
        self, query: np.ndarray, predicate: TemporalPredicate, k: int
    ) -> SearchResult:
        return exact_search(self.dataset, query, predicate, k)


class GlobalPostFilterProxy(SearchMethod):
    """Exact global ranking used to benchmark post-filter semantics before an ANN backend."""

    name = "global_post_filter_proxy"

    def __init__(self, dataset: Dataset, overfetch: int):
        if overfetch <= 0:
            raise ValueError("overfetch must be positive")
        self.dataset = dataset
        self.overfetch = overfetch

    @property
    def index_size_bytes(self) -> int:
        return 0

    def search(
        self, query: np.ndarray, predicate: TemporalPredicate, k: int
    ) -> SearchResult:
        distances = squared_l2(self.dataset.vectors, query)
        fetch = min(len(distances), k * self.overfetch)
        if fetch == 0:
            return SearchResult(
                np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32), 0
            )
        candidates = np.argpartition(distances, fetch - 1)[:fetch]
        valid = predicate.contains(self.dataset.timestamps[candidates])
        ids, result_distances = top_k(candidates[valid], distances[candidates[valid]], k)
        return SearchResult(ids, result_distances, fetch)


class FixedTemporalPartition(SearchMethod):
    name = "fixed_temporal_partition"

    def __init__(self, dataset: Dataset, bucket_count: int):
        if bucket_count <= 0:
            raise ValueError("bucket_count must be positive")
        self.dataset = dataset
        self.bucket_count = bucket_count
        bucket_ids = np.minimum(
            (dataset.timestamps * bucket_count).astype(np.int64), bucket_count - 1
        )
        self.order = np.argsort(bucket_ids, kind="stable").astype(np.int64)
        counts = np.bincount(bucket_ids, minlength=bucket_count)
        self.offsets = np.concatenate(([0], np.cumsum(counts))).astype(np.int64)

    @property
    def index_size_bytes(self) -> int:
        return self.order.nbytes + self.offsets.nbytes

    def search(
        self, query: np.ndarray, predicate: TemporalPredicate, k: int
    ) -> SearchResult:
        selected: list[np.ndarray] = []
        for start, end in predicate.intervals:
            first = min(int(np.floor(start * self.bucket_count)), self.bucket_count - 1)
            last = min(int(np.ceil(end * self.bucket_count)) - 1, self.bucket_count - 1)
            for bucket in range(first, last + 1):
                begin, finish = self.offsets[bucket], self.offsets[bucket + 1]
                selected.append(self.order[begin:finish])

        if not selected:
            return SearchResult(
                np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32), 0
            )
        candidates = np.unique(np.concatenate(selected))
        valid_ids = candidates[predicate.contains(self.dataset.timestamps[candidates])]
        distances = squared_l2(self.dataset.vectors[valid_ids], query)
        ids, result_distances = top_k(valid_ids, distances, k)
        return SearchResult(ids, result_distances, len(candidates))
