from __future__ import annotations

import numpy as np

from .core import Dataset, TemporalPredicate


def generate_dataset(
    size: int,
    dimensions: int,
    correlation: float,
    density: str,
    seed: int,
) -> Dataset:
    """Generate vectors and timestamps in [0, 1) with controlled dependence."""
    if size <= 0 or dimensions <= 0:
        raise ValueError("size and dimensions must be positive")
    if not 0.0 <= correlation <= 1.0:
        raise ValueError("correlation must be between 0 and 1")

    rng = np.random.default_rng(seed)
    if density == "uniform":
        timestamps = rng.random(size)
    elif density == "clustered":
        centers = np.array([0.15, 0.42, 0.72, 0.9])
        cluster = rng.choice(len(centers), size=size, p=[0.15, 0.35, 0.35, 0.15])
        timestamps = np.clip(centers[cluster] + rng.normal(0.0, 0.035, size), 0.0, 1.0)
        timestamps = np.minimum(timestamps, np.nextafter(1.0, 0.0))
    else:
        raise ValueError("density must be 'uniform' or 'clustered'")

    noise = rng.normal(size=(size, dimensions)).astype(np.float32)
    temporal = np.empty_like(noise)
    frequencies = np.arange(1, dimensions + 1, dtype=np.float32)
    phases = 2.0 * np.pi * timestamps[:, None] * np.ceil(frequencies / 2.0)
    temporal[:, 0::2] = np.sin(phases[:, 0::2])
    temporal[:, 1::2] = np.cos(phases[:, 1::2])

    vectors = np.sqrt(1.0 - correlation**2) * noise + correlation * temporal
    vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    return Dataset(vectors.astype(np.float32), timestamps.astype(np.float64))


def fragmented_predicate(
    total_width: float, fragments: int, rng: np.random.Generator
) -> TemporalPredicate:
    """Create disjoint intervals with the requested combined width."""
    if not 0.0 < total_width <= 1.0:
        raise ValueError("total_width must be in (0, 1]")
    if fragments <= 0:
        raise ValueError("fragments must be positive")
    if total_width == 1.0:
        return TemporalPredicate.from_intervals([(0.0, 1.0)])

    interval_width = total_width / fragments
    free_space = 1.0 - total_width
    gaps = rng.dirichlet(np.ones(fragments + 1)) * free_space
    intervals: list[tuple[float, float]] = []
    cursor = gaps[0]
    for index in range(fragments):
        intervals.append((cursor, cursor + interval_width))
        cursor += interval_width + gaps[index + 1]
    return TemporalPredicate.from_intervals(intervals)


def query_for_predicate(
    dataset: Dataset,
    predicate: TemporalPredicate,
    rng: np.random.Generator,
    noise: float = 0.02,
) -> np.ndarray:
    eligible = np.flatnonzero(predicate.contains(dataset.timestamps))
    if len(eligible):
        query = dataset.vectors[rng.choice(eligible)].copy()
    else:
        query = rng.normal(size=dataset.vectors.shape[1]).astype(np.float32)
    query += rng.normal(0.0, noise, size=query.shape).astype(np.float32)
    query /= max(float(np.linalg.norm(query)), 1e-12)
    return query
