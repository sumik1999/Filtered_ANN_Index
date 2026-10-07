from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class TemporalPredicate:
    """A normalized union of half-open intervals [start, end)."""

    intervals: tuple[tuple[float, float], ...]

    @classmethod
    def from_intervals(
        cls, intervals: Iterable[tuple[float, float]]
    ) -> "TemporalPredicate":
        ordered = sorted((float(start), float(end)) for start, end in intervals)
        if any(start >= end for start, end in ordered):
            raise ValueError("Each interval must have start < end")

        merged: list[tuple[float, float]] = []
        for start, end in ordered:
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
            else:
                merged.append((start, end))
        return cls(tuple(merged))

    def contains(self, timestamps: np.ndarray) -> np.ndarray:
        result = np.zeros(timestamps.shape, dtype=bool)
        for start, end in self.intervals:
            result |= (timestamps >= start) & (timestamps < end)
        return result

    def union(self, other: "TemporalPredicate") -> "TemporalPredicate":
        return self.from_intervals((*self.intervals, *other.intervals))

    def intersection(self, other: "TemporalPredicate") -> "TemporalPredicate":
        intersections: list[tuple[float, float]] = []
        left = right = 0
        while left < len(self.intervals) and right < len(other.intervals):
            a_start, a_end = self.intervals[left]
            b_start, b_end = other.intervals[right]
            start, end = max(a_start, b_start), min(a_end, b_end)
            if start < end:
                intersections.append((start, end))
            if a_end < b_end:
                left += 1
            else:
                right += 1
        return self.from_intervals(intersections)


@dataclass(frozen=True)
class Dataset:
    vectors: np.ndarray
    timestamps: np.ndarray

    def __post_init__(self) -> None:
        if self.vectors.ndim != 2:
            raise ValueError("vectors must be a two-dimensional array")
        if self.timestamps.shape != (len(self.vectors),):
            raise ValueError("timestamps must have one value per vector")


@dataclass(frozen=True)
class SearchResult:
    ids: np.ndarray
    distances: np.ndarray
    candidates_examined: int


def squared_l2(vectors: np.ndarray, query: np.ndarray) -> np.ndarray:
    delta = vectors - query
    return np.einsum("ij,ij->i", delta, delta)


def top_k(ids: np.ndarray, distances: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    if k <= 0:
        raise ValueError("k must be positive")
    if len(ids) == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)

    count = min(k, len(ids))
    selected = np.argpartition(distances, count - 1)[:count]
    selected = selected[np.argsort(distances[selected], kind="stable")]
    return ids[selected].astype(np.int64, copy=False), distances[selected]


def exact_search(
    dataset: Dataset, query: np.ndarray, predicate: TemporalPredicate, k: int
) -> SearchResult:
    ids = np.flatnonzero(predicate.contains(dataset.timestamps))
    distances = squared_l2(dataset.vectors[ids], query)
    result_ids, result_distances = top_k(ids, distances, k)
    return SearchResult(result_ids, result_distances, len(ids))
