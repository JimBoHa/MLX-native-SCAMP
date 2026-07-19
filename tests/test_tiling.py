import unittest
from unittest.mock import patch

import mlx.core as mx
import numpy as np

import mlx_native_scamp._metal_1nn as metal_1nn
import mlx_native_scamp._metal_sum as metal_sum
import mlx_native_scamp.core as scamp_core
import pyscamp as mp


class MaxTileSizeTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(2300)
        self.a = rng.normal(size=72).astype(np.float32)
        self.b = rng.normal(size=68).astype(np.float32)
        self.m = 8

    def test_all_profiles_accept_upstream_max_tile_size_kwarg(self):
        options = {
            "gpus": [],
            "precision": "double",
            "max_tile_size": np.int64(1024),
        }
        calls = (
            lambda: mp.selfjoin(self.a, self.m, **options),
            lambda: mp.abjoin(self.a, self.b, self.m, **options),
            lambda: mp.selfjoin_sum(self.a, self.m, **options),
            lambda: mp.abjoin_sum(self.a, self.b, self.m, **options),
            lambda: mp.selfjoin_matrix(
                self.a, self.m, mheight=3, mwidth=4, **options
            ),
            lambda: mp.abjoin_matrix(
                self.a, self.b, self.m, mheight=3, mwidth=4, **options
            ),
            lambda: mp.selfjoin_knn(self.a, self.m, 2, **options),
            lambda: mp.abjoin_knn(self.a, self.b, self.m, 2, **options),
        )

        for call in calls:
            with self.subTest(call=call):
                call()

    def test_max_tile_size_enforces_upstream_constraints(self):
        for value in (-1, 0):
            with self.subTest(value=value), self.assertRaisesRegex(
                ValueError, "value must be greater than 0"
            ):
                mp.selfjoin(self.a, self.m, max_tile_size=value)

        for value in (1, 1023):
            with self.subTest(value=value), self.assertRaisesRegex(
                ValueError, "must be at least 1024"
            ):
                mp.selfjoin(self.a, self.m, max_tile_size=value)

        series = np.arange(513, dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "at least twice m"):
            mp.abjoin(series, series, 513, max_tile_size=1024)
        with (
            patch.object(
                scamp_core, "_default_max_tile_size", return_value=1024
            ),
            self.assertRaisesRegex(ValueError, "at least twice m"),
        ):
            mp.abjoin(series, series, 513)

        boundary = np.arange(512, dtype=np.float32)
        profile, index = mp.abjoin(
            boundary,
            boundary,
            512,
            pearson=True,
            max_tile_size=1024,
            gpus=[],
        )
        self.assertEqual((1,), profile.shape)
        self.assertEqual(profile.shape, index.shape)

    def test_tile_geometry_caps_both_axes_and_advisory_working_set(self):
        budget = 256 * 1024
        with patch.object(
            scamp_core, "_similarity_tile_budget_bytes", return_value=budget
        ):
            rows, columns = scamp_core._portable_tile_shape(
                10_000_000,
                20_000_000,
                64,
                mx.float32,
                1024,
            )

        max_subsequences = 1024 - 64 + 1
        self.assertGreater(rows, 0)
        self.assertGreater(columns, 0)
        self.assertLessEqual(rows, min(scamp_core.BLOCK_ROWS, max_subsequences))
        self.assertLessEqual(columns, max_subsequences)
        self.assertLessEqual(
            scamp_core._estimate_similarity_tile_bytes(
                rows, columns, 64, np.dtype(np.float32).itemsize
            ),
            budget,
        )

    def test_unified_memory_controls_the_automatic_tile_budget(self):
        with patch.object(
            scamp_core,
            "_device_working_set_bytes",
            return_value=512 * scamp_core.MIB,
        ):
            small = scamp_core._similarity_tile_budget_bytes()
        with patch.object(
            scamp_core,
            "_device_working_set_bytes",
            return_value=8 * 1024 * scamp_core.MIB,
        ):
            large = scamp_core._similarity_tile_budget_bytes()

        self.assertEqual(scamp_core.MIN_SIMILARITY_TILE_BUDGET_BYTES, small)
        self.assertEqual(scamp_core.MAX_SIMILARITY_TILE_BUDGET_BYTES, large)

    def test_default_tile_size_matches_upstream_resource_defaults(self):
        with mx.stream(mx.cpu):
            self.assertEqual(
                scamp_core.UPSTREAM_CPU_MAX_TILE_SIZE,
                scamp_core._default_max_tile_size(),
            )
        with mx.stream(mx.gpu):
            self.assertEqual(
                scamp_core.UPSTREAM_METAL_MAX_TILE_SIZE,
                scamp_core._default_max_tile_size(),
            )

    def test_forced_multitile_profiles_match_single_tile_results(self):
        options = {"gpus": [], "precision": "double"}
        calls = {
            "1nn": lambda **extra: mp.abjoin(
                self.a, self.b, self.m, pearson=True, **options, **extra
            ),
            "sum": lambda **extra: mp.abjoin_sum(
                self.a,
                self.b,
                self.m,
                threshold=0.2,
                **options,
                **extra,
            ),
            "matrix": lambda **extra: mp.abjoin_matrix(
                self.a,
                self.b,
                self.m,
                threshold=0.2,
                mheight=4,
                mwidth=5,
                pearson=True,
                **options,
                **extra,
            ),
            "knn": lambda **extra: mp.abjoin_knn(
                self.a,
                self.b,
                self.m,
                3,
                threshold=0.2,
                pearson=True,
                **options,
                **extra,
            ),
        }

        for name, call in calls.items():
            with self.subTest(profile=name):
                expected = call()
                with patch.object(
                    scamp_core,
                    "_similarity_tile_budget_bytes",
                    return_value=64 * 1024,
                ):
                    actual = call(max_tile_size=1024)
                self._assert_profile_equal(expected, actual)

    def test_tiny_budget_selfjoin_profiles_preserve_global_exclusion(self):
        rng = np.random.default_rng(2302)
        series = rng.normal(size=52).astype(np.float64)
        m = 8
        options = {
            "gpus": [],
            "precision": "double",
            "max_tile_size": 1024,
        }
        calls = {
            "1nn": lambda: mp.selfjoin(
                series, m, pearson=True, **options
            ),
            "sum": lambda: mp.selfjoin_sum(
                series, m, threshold=0.1, **options
            ),
            "matrix": lambda: mp.selfjoin_matrix(
                series,
                m,
                threshold=0.0,
                mheight=4,
                mwidth=5,
                pearson=True,
                **options,
            ),
            "knn": lambda: mp.selfjoin_knn(
                series,
                m,
                3,
                threshold=0.0,
                pearson=True,
                **options,
            ),
        }
        subsequences = series.size - m + 1
        with patch.object(
            scamp_core,
            "_similarity_tile_budget_bytes",
            return_value=8 * 1024,
        ):
            tile_rows, tile_columns = scamp_core._portable_tile_shape(
                subsequences,
                subsequences,
                m,
                mx.float64,
                1024,
            )
        self.assertLess(tile_rows, subsequences)
        self.assertLess(tile_columns, subsequences)
        self.assertNotEqual(tile_rows, tile_columns)

        for name, call in calls.items():
            with self.subTest(profile=name):
                with patch.object(
                    scamp_core,
                    "_similarity_tile_budget_bytes",
                    return_value=64 * scamp_core.MIB,
                ):
                    expected = call()
                with patch.object(
                    scamp_core,
                    "_similarity_tile_budget_bytes",
                    return_value=8 * 1024,
                ):
                    actual = call()
                self._assert_profile_equal(expected, actual)

    def test_knn_equal_correlations_prefer_smallest_rows_for_every_tile_shape(self):
        pattern = np.array([0.0, 1.0, 0.0, -1.0], dtype=np.float64)
        a = np.tile(pattern, 12)
        b = np.tile(pattern, 16)
        results = []

        for budget in (64 * scamp_core.MIB, 8 * 1024):
            with patch.object(
                scamp_core,
                "_similarity_tile_budget_bytes",
                return_value=budget,
            ):
                results.append(
                    mp.abjoin_knn(
                        a,
                        b,
                        4,
                        5,
                        threshold=0.99,
                        pearson=True,
                        precision="double",
                        gpus=[],
                        max_tile_size=1024,
                    )
                )

        self._assert_profile_equal(results[0], results[1])
        first_column = [row for col, row, _ in results[0] if col == 0]
        self.assertEqual([0, 4, 8, 12, 16], first_column)
        np.testing.assert_allclose(
            [value for col, _, value in results[0] if col == 0],
            np.ones(5),
        )

    def test_normalization_and_lazy_execution_stay_tile_bounded(self):
        rng = np.random.default_rng(2301)
        a = rng.normal(size=120).astype(np.float32)
        b = rng.normal(size=136).astype(np.float32)
        spans = []
        original_prepare = scamp_core._prepare_series_tile

        def record_prepare(series, start, end, m):
            spans.append((end - start, int(series.values.shape[0])))
            return original_prepare(series, start, end, m)

        with (
            patch.object(
                scamp_core,
                "_similarity_tile_budget_bytes",
                return_value=64 * 1024,
            ),
            patch.object(
                scamp_core,
                "_prepare_series_tile",
                side_effect=record_prepare,
            ),
            patch.object(
                scamp_core,
                "_schedule_reducer_state",
                wraps=scamp_core._schedule_reducer_state,
            ) as schedule,
            patch.object(
                scamp_core.mx,
                "synchronize",
                wraps=scamp_core.mx.synchronize,
            ) as synchronize,
        ):
            profile, index = mp.abjoin(
                a,
                b,
                16,
                pearson=True,
                precision="double",
                gpus=[],
                max_tile_size=1024,
            )

        self.assertEqual((a.size - 16 + 1,), profile.shape)
        self.assertEqual(profile.shape, index.shape)
        self.assertGreater(len(spans), 2)
        self.assertTrue(all(span <= 1024 - 16 + 1 for span, _ in spans))
        self.assertTrue(all(span < total - 16 + 1 for span, total in spans))
        self.assertGreater(schedule.call_count, scamp_core.MAX_IN_FLIGHT_SIMILARITY_TILES)
        self.assertGreater(synchronize.call_count, 0)

    def test_restrictive_explicit_tile_does_not_bypass_limit_on_metal(self):
        series = np.arange(1025, dtype=np.float32)

        with patch.object(metal_1nn, "best_match") as kernel:
            profile, index = mp.selfjoin(
                series,
                3,
                pearson=True,
                precision="single",
                gpus=[0],
                max_tile_size=1024,
            )

        kernel.assert_not_called()
        self.assertEqual((1023,), profile.shape)
        self.assertEqual(profile.shape, index.shape)

    def test_omitted_tile_size_uses_default_ceiling_before_metal_routing(self):
        series = np.arange(1025, dtype=np.float32)
        calls = (
            lambda: mp.selfjoin(
                series,
                8,
                pearson=True,
                precision="single",
                gpus=[0],
            ),
            lambda: mp.selfjoin_sum(
                series,
                8,
                threshold=0.2,
                precision="single",
                gpus=[0],
            ),
        )
        original_prepare = scamp_core._prepare_series_tile

        for call in calls:
            spans = []

            def record_prepare(tiled_series, start, end, m):
                spans.append(end - start)
                return original_prepare(tiled_series, start, end, m)

            with (
                self.subTest(call=call),
                patch.object(
                    scamp_core, "_default_max_tile_size", return_value=1024
                ),
                patch.object(
                    scamp_core,
                    "_metal_sum_is_worthwhile",
                    return_value=True,
                ) as sum_worthwhile,
                patch.object(
                    metal_1nn,
                    "best_match",
                    side_effect=AssertionError("join-wide Metal bypassed ceiling"),
                ) as one_nn_kernel,
                patch.object(
                    metal_sum,
                    "sum_threshold",
                    side_effect=AssertionError("join-wide Metal bypassed ceiling"),
                ) as sum_kernel,
                patch.object(
                    scamp_core,
                    "_prepare_series_tile",
                    side_effect=record_prepare,
                ),
            ):
                output = call()

            one_nn_kernel.assert_not_called()
            sum_kernel.assert_not_called()
            sum_worthwhile.assert_not_called()
            self.assertTrue(spans)
            self.assertLessEqual(max(spans), 1024 - 8 + 1)
            result_size = output[0].size if isinstance(output, tuple) else output.size
            self.assertEqual(series.size - 8 + 1, result_size)

    def _assert_profile_equal(self, expected, actual):
        if isinstance(expected, tuple):
            np.testing.assert_allclose(
                expected[0], actual[0], rtol=1e-5, atol=1e-5, equal_nan=True
            )
            np.testing.assert_array_equal(expected[1], actual[1])
            return
        if isinstance(expected, np.ndarray):
            np.testing.assert_allclose(
                expected, actual, rtol=1e-5, atol=1e-5, equal_nan=True
            )
            return

        self.assertEqual(
            [(col, row) for col, row, _ in expected],
            [(col, row) for col, row, _ in actual],
        )
        np.testing.assert_allclose(
            [value for _, _, value in expected],
            [value for _, _, value in actual],
            rtol=1e-5,
            atol=1e-5,
        )


if __name__ == "__main__":
    unittest.main()
