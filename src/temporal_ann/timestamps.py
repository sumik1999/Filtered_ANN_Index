from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def uniform_timestamps(count: int, rng: np.random.Generator) -> np.ndarray:
    return rng.random(count, dtype=np.float64)


def poisson_timestamps(
    count: int, bins: int, rng: np.random.Generator
) -> tuple[np.ndarray, dict]:
    """Sample a clustered non-homogeneous Poisson process, conditioned on count."""
    rates = np.ones(bins, dtype=np.float64)
    regions = (
        (0.10, 0.15, 10.0),
        (0.40, 0.50, 20.0),
        (0.80, 0.83, 15.0),
    )
    for start, end, multiplier in regions:
        first = int(np.floor(start * bins))
        last = int(np.ceil(end * bins))
        rates[first:last] = multiplier

    selected = rng.choice(bins, size=count, p=rates / rates.sum())
    timestamps = (selected + rng.random(count)) / bins
    parameters = {
        "process": "non-homogeneous Poisson process conditioned on total count",
        "baseline_rate": 1.0,
        "high_density_regions": [
            {"start": start, "end": end, "rate_multiplier": multiplier}
            for start, end, multiplier in regions
        ],
    }
    return timestamps.astype(np.float64, copy=False), parameters


def zipf_timestamps(
    count: int, bins: int, exponent: float, rng: np.random.Generator
) -> np.ndarray:
    """Sample from a finite Zipf distribution and jitter within each time bin."""
    ranks = np.arange(1, bins + 1, dtype=np.float64)
    probabilities = np.power(ranks, -exponent)
    probabilities /= probabilities.sum()
    selected = rng.choice(bins, size=count, p=probabilities)
    return ((selected + rng.random(count)) / bins).astype(np.float64, copy=False)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summarize(values: np.ndarray, bins: int) -> dict:
    counts, _ = np.histogram(values, bins=bins, range=(0.0, 1.0))
    quantiles = np.quantile(values, [0.01, 0.25, 0.5, 0.75, 0.99])
    return {
        "count": int(len(values)),
        "dtype": str(values.dtype),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "mean": float(values.mean()),
        "standard_deviation": float(values.std()),
        "quantiles": {
            "0.01": float(quantiles[0]),
            "0.25": float(quantiles[1]),
            "0.50": float(quantiles[2]),
            "0.75": float(quantiles[3]),
            "0.99": float(quantiles[4]),
        },
        "bin_count_min": int(counts.min()),
        "bin_count_max": int(counts.max()),
        "bin_count_coefficient_of_variation": float(counts.std() / counts.mean()),
    }


def generate(args: argparse.Namespace) -> dict:
    if args.count <= 0 or args.bins <= 0:
        raise ValueError("count and bins must be positive")
    if args.zipf_exponent <= 0:
        raise ValueError("zipf exponent must be positive")

    args.output.mkdir(parents=True, exist_ok=True)
    streams = np.random.SeedSequence(args.seed).spawn(3)
    distributions: list[tuple[str, np.ndarray, dict]] = []
    distributions.append(
        ("uniform", uniform_timestamps(args.count, np.random.default_rng(streams[0])), {})
    )
    poisson, poisson_parameters = poisson_timestamps(
        args.count, args.bins, np.random.default_rng(streams[1])
    )
    distributions.append(("poisson_clustered", poisson, poisson_parameters))
    distributions.append(
        (
            "zipf",
            zipf_timestamps(
                args.count,
                args.bins,
                args.zipf_exponent,
                np.random.default_rng(streams[2]),
            ),
            {"exponent": args.zipf_exponent},
        )
    )

    year_start = datetime(args.year, 1, 1, tzinfo=timezone.utc)
    year_end = datetime(args.year + 1, 1, 1, tzinfo=timezone.utc)
    metadata = {
        "schema_version": 1,
        "source_vectors": "../sift-128-euclidean.hdf5::train",
        "assignment": "independent of vector values; array position equals train vector ID",
        "seed": args.seed,
        "normalized_range": {"start_inclusive": 0.0, "end_exclusive": 1.0},
        "calendar_mapping": {
            "start_inclusive": year_start.isoformat(),
            "end_exclusive": year_end.isoformat(),
            "duration_seconds": int((year_end - year_start).total_seconds()),
            "note": "Calendar mapping is metadata only; benchmark files remain normalized float64.",
        },
        "temporal_bins": args.bins,
        "distributions": {},
    }

    for name, values, parameters in distributions:
        if values.dtype != np.float64 or np.any(values < 0.0) or np.any(values >= 1.0):
            raise RuntimeError(f"invalid generated timestamps for {name}")
        path = args.output / f"{name}_seed-{args.seed}.npy"
        np.save(path, values, allow_pickle=False)
        metadata["distributions"][name] = {
            "file": path.name,
            "parameters": parameters,
            "summary": summarize(values, args.bins),
            "sha256": sha256(path),
        }

    (args.output / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate timestamps for SIFT1M")
    parser.add_argument("--count", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--year", type=int, default=2023)
    parser.add_argument("--bins", type=int, default=1024)
    parser.add_argument("--zipf-exponent", type=float, default=1.2)
    parser.add_argument(
        "--output", type=Path, default=Path("SIFT_data/timestamps")
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    metadata = generate(args)
    print(
        f"Generated {len(metadata['distributions'])} timestamp arrays in {args.output}"
    )


if __name__ == "__main__":
    main()
