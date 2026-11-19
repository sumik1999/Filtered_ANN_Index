import tempfile
import unittest
from pathlib import Path

import numpy as np

from temporal_ann.twitter_ann.bucketed import interval_mask, normalize_intervals
from temporal_ann.twitter_ann.rolling_hnsw import (
    RollingBuildConfig,
    RollingSearchConfig,
    RollingTemporalHNSW,
    build_rolling_index,
)


class TrackingMatrix:
    """Array-like wrapper recording every database row read by search."""

    def __init__(self, values: np.ndarray):
        self.values = values
        self.shape = values.shape
        self.dtype = values.dtype
        self.read_ids: list[int] = []

    def reset(self) -> None:
        self.read_ids.clear()

    def __getitem__(self, key):
        row_key = key[0] if isinstance(key, tuple) else key
        rows = np.arange(len(self.values), dtype=np.int64)[row_key]
        self.read_ids.extend(np.asarray(rows).reshape(-1).astype(int).tolist())
        return self.values[key]


class TwitterRollingFilteringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        # Every timestamp occurs twice, exercising stable ordering and half-open
        # boundaries where several records share the endpoint.
        chronological_timestamps = np.repeat(
            np.arange(1_000, 17_000, 1_000, dtype=np.int64), 2
        )
        permutation = np.random.default_rng(17).permutation(
            len(chronological_timestamps)
        )
        cls.timestamps = chronological_timestamps[permutation]
        ids = np.arange(len(cls.timestamps), dtype=np.float32)
        cls.vectors = np.column_stack((
            ids / 32.0,
            np.sin(ids),
            np.cos(ids),
            (ids % 5) / 5.0,
        )).astype(np.float32)
        root = Path(cls.temporary.name) / "filter_index"
        build_rolling_index(
            cls.vectors,
            cls.timestamps,
            root,
            RollingBuildConfig(
                vector_count=len(cls.timestamps),
                semantic_degree=4,
                gateway_degree=3,
                gateway_period=4,
                construction_candidates=8,
                ef_construction=30,
                candidate_index_m=4,
                range_degree_per_side=1,
                range_scales_hours="1,2,4",
                hierarchy_enabled=True,
                hierarchy_level_probability=0.7,
                hierarchy_max_level=2,
                hierarchy_semantic_degree=2,
                hierarchy_construction_candidates=6,
                hierarchy_range_degree_per_side=1,
                hierarchy_range_candidate_count=6,
                hierarchy_range_scales_hours="1,2",
                threads=1,
            ),
            progress_every=0,
        )
        cls.index = RollingTemporalHNSW(root, cls.vectors)
        cls.tracker = TrackingMatrix(cls.vectors)
        cls.index.vectors = cls.tracker

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def exact_config(self, k=7, threshold=10_000):
        return RollingSearchConfig(
            k=k,
            ef_search=max(16, k),
            exact_threshold=threshold,
            gateway_candidates=8,
            gateway_seeds=3,
        )

    def graph_config(self, k=7, hierarchy=True):
        return RollingSearchConfig(
            k=k,
            ef_search=16,
            exact_threshold=0,
            gateway_candidates=8,
            gateway_seeds=3,
            use_temporal_edges=True,
            use_gateway_edges=True,
            use_range_edges=True,
            use_hierarchy=hierarchy,
            upper_ef_search=8,
        )

    def expected_exact(self, query, intervals, k):
        eligible = np.flatnonzero(interval_mask(self.timestamps, intervals))
        differences = self.vectors[eligible] - query
        distances = np.einsum("ij,ij->i", differences, differences)
        order = np.argsort(distances, kind="stable")[:k]
        return eligible[order], distances[order]

    def assert_only_eligible_rows_read(self, intervals):
        mask = interval_mask(self.timestamps, intervals)
        self.assertTrue(
            all(mask[global_id] for global_id in self.tracker.read_ids),
            f"out-of-range vector rows were read: {self.tracker.read_ids}",
        )

    def test_half_open_boundaries_include_start_and_exclude_duplicate_end(self):
        intervals = [[2_000, 4_000]]
        query = np.array([0.3, 0.1, -0.2, 0.4], dtype=np.float32)
        self.tracker.reset()
        result = self.index.search(query, intervals, self.exact_config(k=20))
        returned_timestamps = self.timestamps[result["ids"]]
        self.assertEqual(result["eligible_count"], 4)
        self.assertTrue(np.all(returned_timestamps >= 2_000))
        self.assertTrue(np.all(returned_timestamps < 4_000))
        self.assertEqual(np.count_nonzero(returned_timestamps == 2_000), 2)
        self.assertEqual(np.count_nonzero(returned_timestamps == 4_000), 0)
        self.assert_only_eligible_rows_read(intervals)

    def test_overlapping_touching_nested_and_reordered_intervals_are_invariant(self):
        query = np.array([0.5, -0.2, 0.4, 0.1], dtype=np.float32)
        canonical = [[2_000, 8_000], [11_000, 14_000]]
        redundant = [
            [12_000, 14_000], [4_000, 7_000], [2_000, 5_000],
            [7_000, 8_000], [11_000, 13_000], [2_000, 8_000],
        ]
        self.assertEqual(normalize_intervals(canonical), normalize_intervals(redundant))
        first = self.index.search(query, canonical, self.exact_config(k=12))
        second = self.index.search(query, redundant, self.exact_config(k=12))
        np.testing.assert_array_equal(first["ids"], second["ids"])
        np.testing.assert_allclose(first["distances"], second["distances"])
        self.assertEqual(first["eligible_count"], second["eligible_count"])

    def test_empty_and_outside_timeline_filters_return_no_results(self):
        query = np.zeros(4, dtype=np.float32)
        for intervals in ([], [[-10_000, -1]], [[20_000, 30_000]]):
            self.tracker.reset()
            result = self.index.search(query, intervals, self.graph_config())
            self.assertIn(result["method"], {"empty", "exact"})
            self.assertEqual(result["eligible_count"], 0)
            self.assertEqual(result["candidate_count"], 0)
            self.assertEqual(len(result["ids"]), 0)
            self.assertEqual(self.tracker.read_ids, [])

    def test_full_int64_domain_filter_contains_every_record(self):
        intervals = [[np.iinfo(np.int64).min, np.iinfo(np.int64).max]]
        result = self.index.search(
            np.zeros(4, dtype=np.float32), intervals, self.exact_config(k=100)
        )
        self.assertEqual(result["eligible_count"], len(self.timestamps))
        self.assertEqual(len(result["ids"]), len(self.timestamps))

    def test_exact_threshold_is_inclusive(self):
        query = np.zeros(4, dtype=np.float32)
        intervals = [[1_000, 3_000]]  # Four records.
        exact = self.index.search(query, intervals, self.exact_config(threshold=4))
        graph = self.index.search(query, intervals, self.exact_config(threshold=3))
        self.assertEqual(exact["method"], "exact")
        self.assertNotEqual(graph["method"], "exact")
        self.assertEqual(exact["eligible_count"], graph["eligible_count"])

    def test_mixed_exact_and_graph_components_remain_strict(self):
        # First component has two records; second has eight. The total exceeds
        # the threshold while only the first component uses exact fallback.
        intervals = [[1_000, 2_000], [9_000, 13_000]]
        config = RollingSearchConfig(
            k=8,
            ef_search=12,
            exact_threshold=3,
            gateway_candidates=8,
            gateway_seeds=3,
            use_range_edges=True,
            use_hierarchy=False,
        )
        self.tracker.reset()
        result = self.index.search(
            np.array([0.2, 0.1, 0.0, 0.3], dtype=np.float32), intervals, config
        )
        self.assertIn("exact_component", result["method"])
        self.assertIn("strict_graph", result["method"])
        self.assertEqual(result["eligible_count"], 10)
        self.assert_only_eligible_rows_read(intervals)
        self.assertEqual(len(result["ids"]), len(np.unique(result["ids"])))

    def test_component_without_an_eligible_gateway_falls_back_to_exact(self):
        intervals = [[1_000, 2_000]]
        self.tracker.reset()
        result = self.index.search(
            np.zeros(4, dtype=np.float32),
            intervals,
            self.graph_config(hierarchy=False),
            entry_gateway_ids=np.array([31], dtype=np.int64),
        )
        self.assertEqual(result["method"], "exact_no_gateway")
        self.assertEqual(result["eligible_count"], 2)
        self.assert_only_eligible_rows_read(intervals)

    def test_every_graph_edge_configuration_preserves_filter_isolation(self):
        intervals = [[3_000, 12_000]]
        query = np.array([0.4, -0.3, 0.2, 0.1], dtype=np.float32)
        configurations = [
            {"use_temporal_edges": False, "use_gateway_edges": False,
             "use_range_edges": False, "use_hierarchy": False},
            {"use_temporal_edges": True, "use_gateway_edges": False,
             "use_range_edges": False, "use_hierarchy": False},
            {"use_temporal_edges": True, "use_gateway_edges": True,
             "use_range_edges": False, "use_hierarchy": False},
            {"use_temporal_edges": True, "use_gateway_edges": False,
             "use_range_edges": True, "use_hierarchy": False},
            {"use_temporal_edges": True, "use_gateway_edges": True,
             "use_range_edges": True, "use_hierarchy": True},
        ]
        for options in configurations:
            with self.subTest(options=options):
                config = RollingSearchConfig(
                    k=7, ef_search=16, exact_threshold=0,
                    gateway_candidates=8, gateway_seeds=3,
                    upper_ef_search=8, **options,
                )
                self.tracker.reset()
                result = self.index.search(query, intervals, config)
                mask = interval_mask(self.timestamps, intervals)
                self.assertTrue(all(mask[value] for value in result["ids"]))
                self.assert_only_eligible_rows_read(intervals)

    def test_randomized_exact_queries_match_brute_force_oracle(self):
        rng = np.random.default_rng(81)
        for _ in range(80):
            interval_count = int(rng.integers(0, 6))
            intervals = []
            for _ in range(interval_count):
                start = int(rng.integers(-2, 20)) * 1_000
                width = int(rng.integers(1, 8)) * 1_000
                intervals.append([start, start + width])
            query = rng.normal(size=4).astype(np.float32)
            k = int(rng.integers(1, 15))
            expected_ids, expected_distances = self.expected_exact(query, intervals, k)
            result = self.index.search(query, intervals, self.exact_config(k=k))
            np.testing.assert_array_equal(result["ids"], expected_ids)
            np.testing.assert_allclose(
                result["distances"], expected_distances, rtol=1e-6, atol=1e-6
            )
            self.assertEqual(
                result["eligible_count"],
                int(np.count_nonzero(interval_mask(self.timestamps, intervals))),
            )

    def test_randomized_graph_queries_never_read_or_return_ineligible_vectors(self):
        rng = np.random.default_rng(91)
        for _ in range(80):
            intervals = []
            for _ in range(int(rng.integers(1, 5))):
                start = int(rng.integers(-2, 18)) * 1_000
                width = int(rng.integers(1, 10)) * 1_000
                intervals.append([start, start + width])
            query = rng.normal(size=4).astype(np.float32)
            self.tracker.reset()
            result = self.index.search(query, intervals, self.graph_config(k=9))
            mask = interval_mask(self.timestamps, intervals)
            self.assertEqual(result["eligible_count"], int(np.count_nonzero(mask)))
            self.assertTrue(all(mask[global_id] for global_id in result["ids"]))
            self.assertEqual(len(result["ids"]), len(np.unique(result["ids"])))
            self.assertTrue(np.all(np.diff(result["distances"]) >= -1e-7))
            self.assert_only_eligible_rows_read(intervals)

    def test_invalid_filters_and_nonfinite_queries_are_rejected(self):
        query = np.zeros(4, dtype=np.float32)
        for intervals in ([[1_000, 1_000]], [[2_000, 1_000]], [[1_000]]):
            with self.assertRaises((TypeError, ValueError)):
                self.index.search(query, intervals, self.exact_config())
        for invalid in (
            np.array([np.nan, 0, 0, 0], dtype=np.float32),
            np.array([np.inf, 0, 0, 0], dtype=np.float32),
        ):
            with self.assertRaises(ValueError):
                self.index.search(invalid, [[1_000, 2_000]], self.exact_config())


if __name__ == "__main__":
    unittest.main()
