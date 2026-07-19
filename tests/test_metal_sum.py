import unittest
from unittest import mock

import mlx.core as mx
import numpy as np

import mlx_native_scamp as mp
from mlx_native_scamp import _metal_sum, core
from reference import distance_matrix, reduce_sum_thresh


class MetalSumRoutingTests(unittest.TestCase):
    def test_density_pair_floors_are_inclusive(self):
        cases = (
            (0.05, 12_000),
            (0.18, 25_000),
        )
        for density, side in cases:
            with self.subTest(density=density):
                self.assertTrue(
                    core._metal_sum_workload_is_worthwhile(
                        side, side, 128, density, False
                    )
                )
                self.assertFalse(
                    core._metal_sum_workload_is_worthwhile(
                        side - 1, side, 128, density, False
                    )
                )

        self.assertFalse(
            core._metal_sum_workload_is_worthwhile(
                1_000_000, 1_000_000, 128, 0.180_001, False
            )
        )

    def test_selfjoin_gate_counts_only_eligible_diagonals(self):
        # ceil(65 / 4) excludes 17 diagonals.  4,000 active diagonals contain
        # 8,002,000 comparisons; one fewer contains only 7,998,000.
        self.assertTrue(
            core._metal_sum_workload_is_worthwhile(
                4_017, 4_017, 65, 0.04, True
            )
        )
        self.assertFalse(
            core._metal_sum_workload_is_worthwhile(
                4_016, 4_016, 65, 0.04, True
            )
        )

    def test_short_windows_raise_the_pair_floor(self):
        self.assertTrue(
            core._metal_sum_workload_is_worthwhile(
                12_000, 12_000, 128, 0.0, False
            )
        )
        self.assertFalse(
            core._metal_sum_workload_is_worthwhile(
                12_000, 12_000, 16, 0.0, False
            )
        )
        self.assertTrue(
            core._metal_sum_workload_is_worthwhile(
                17_000, 17_000, 16, 0.0, False
            )
        )

    def test_rectangular_ab_join_stays_portable(self):
        self.assertFalse(
            core._metal_sum_workload_is_worthwhile(
                144_000_000, 1, 128, 0.0, False
            )
        )
        self.assertFalse(
            core._metal_sum_workload_is_worthwhile(
                24_000, 6_000, 128, 0.0, False
            )
        )

    def test_density_sample_rejects_smooth_atomic_heavy_join(self):
        m = 128
        subsequences = 5_500
        size = subsequences + m - 1
        random = np.random.default_rng(14).normal(size=size).astype(np.float32)
        ramp = np.arange(size, dtype=np.float32)

        self.assertTrue(
            core._metal_sum_is_worthwhile(
                random,
                random,
                subsequences,
                subsequences,
                m,
                0.2,
                True,
            )
        )
        self.assertFalse(
            core._metal_sum_is_worthwhile(
                ramp,
                ramp,
                subsequences,
                subsequences,
                m,
                0.2,
                True,
            )
        )

    def test_regression_sized_short_window_joins_stay_portable(self):
        self.assertFalse(
            core._metal_sum_workload_is_worthwhile(
                1_485, 1_485, 16, 0.0, True
            )
        )
        self.assertFalse(
            core._metal_sum_workload_is_worthwhile(
                5_085, 5_085, 16, 0.0, False
            )
        )


@unittest.skipUnless(mx.metal.is_available(), "Metal is unavailable")
class MetalSumThresholdTests(unittest.TestCase):
    def setUp(self):
        self.previous_device = mx.default_device()
        mx.set_default_device(mx.gpu)

    def tearDown(self):
        mx.set_default_device(self.previous_device)

    def test_selfjoin_matches_reference_with_invalid_windows(self):
        rng = np.random.default_rng(331)
        series = rng.normal(size=113).astype(np.float32)
        series[19] = np.nan
        m = 8
        threshold = 0.15
        expected = reduce_sum_thresh(distance_matrix(series, None, m), threshold)

        with (
            mock.patch.object(
                core, "_metal_sum_is_worthwhile", return_value=True
            ),
            mock.patch.object(
                _metal_sum, "sum_threshold", wraps=_metal_sum.sum_threshold
            ) as kernel,
        ):
            actual = mp.selfjoin_sum(
                series,
                m,
                threshold=threshold,
                precision="single",
            )

        kernel.assert_called_once()
        np.testing.assert_allclose(actual, expected, rtol=5e-5, atol=2e-5)

    def test_abjoin_matches_reference(self):
        rng = np.random.default_rng(1204)
        a = rng.normal(size=91).astype(np.float32)
        b = rng.normal(size=107).astype(np.float32)
        b[42] = np.inf
        m = 12
        threshold = 0.25
        matrix = distance_matrix(a, b, m)
        expected = reduce_sum_thresh(matrix, threshold)

        with (
            mx.stream(mx.cpu),
            mock.patch.object(
                core, "_metal_sum_is_worthwhile", return_value=True
            ),
            mock.patch.object(
                _metal_sum, "sum_threshold", wraps=_metal_sum.sum_threshold
            ) as kernel,
        ):
            actual = mp.abjoin_sum(
                a,
                b,
                m,
                threshold=threshold,
                precision="single",
                gpus=[0],
            )

        kernel.assert_called_once()
        np.testing.assert_allclose(actual, expected, rtol=5e-5, atol=2e-5)

    def test_aligned_abjoin_exclusion_matches_reference(self):
        rng = np.random.default_rng(2113)
        a = rng.normal(size=97).astype(np.float32)
        b = rng.normal(size=97).astype(np.float32)
        m = 9
        source_start = 16
        exclusion = (m + 3) // 4
        target_start = source_start + exclusion - 1
        b[target_start : target_start + m] = a[
            source_start : source_start + m
        ]
        matrix = distance_matrix(a, b, m)
        positions = np.arange(matrix.shape[0])
        matrix[
            np.abs(positions[:, None] - positions[None, :]) < exclusion
        ] = -2.0
        threshold = 0.15
        expected = reduce_sum_thresh(matrix, threshold)

        with (
            mock.patch.object(
                core, "_metal_sum_is_worthwhile", return_value=True
            ),
            mock.patch.object(
                _metal_sum, "sum_threshold", wraps=_metal_sum.sum_threshold
            ) as kernel,
        ):
            actual = mp.abjoin_sum(
                a,
                b,
                m,
                threshold=threshold,
                allow_trivial_match=False,
                precision="single",
                gpus=[0],
            )

        kernel.assert_called_once()
        self.assertEqual(exclusion, kernel.call_args.args[-1])
        np.testing.assert_allclose(actual, expected, rtol=5e-5, atol=2e-5)

    def test_threshold_comparison_is_strict(self):
        series = np.arange(32, dtype=np.float32)
        with mock.patch.object(
            core, "_metal_sum_is_worthwhile", return_value=True
        ):
            actual = mp.abjoin_sum(
                series,
                series,
                8,
                threshold=1.0,
                precision="single",
                gpus=[0],
            )
        np.testing.assert_array_equal(actual, np.zeros_like(actual))

    def test_selfjoin_uses_ceil_exclusion_for_non_multiple_of_four_window(self):
        rng = np.random.default_rng(94)
        series = rng.normal(size=67).astype(np.float32)
        m = 5
        matrix = distance_matrix(series, None, m)
        positions = np.arange(matrix.shape[0])
        matrix[np.abs(positions[:, None] - positions[None, :]) < (m + 3) // 4] = -2.0
        expected = reduce_sum_thresh(matrix, 0.0)

        with mock.patch.object(
            core, "_metal_sum_is_worthwhile", return_value=True
        ):
            actual = mp.selfjoin_sum(
                series,
                m,
                threshold=0.0,
                precision="single",
                gpus=[0],
            )

        np.testing.assert_allclose(actual, expected, rtol=5e-5, atol=2e-5)

    def test_high_offset_float32_variation_matches_reference(self):
        rng = np.random.default_rng(409)
        a = (1e7 + rng.integers(-32, 33, size=173)).astype(np.float32)
        b = (1e7 + rng.integers(-32, 33, size=181)).astype(np.float32)
        m = 16
        threshold = 0.2
        expected = reduce_sum_thresh(distance_matrix(a, b, m), threshold)

        with mock.patch.object(
            core, "_metal_sum_is_worthwhile", return_value=True
        ):
            actual = mp.abjoin_sum(
                a,
                b,
                m,
                threshold=threshold,
                precision="single",
                gpus=[0],
            )

        np.testing.assert_allclose(actual, expected, rtol=5e-5, atol=3e-5)

    def test_checkpoints_bound_long_diagonal_drift_across_partial_batches(self):
        series = np.random.default_rng(3).normal(size=2_300).astype(np.float32)
        m = 16
        threshold = 0.1
        expected = mp.selfjoin_sum(
            series,
            m,
            threshold=threshold,
            precision="single",
            gpus=[],
        )
        diagonal_count = series.size - m + 1 - (m + 3) // 4
        self.assertGreater(diagonal_count, _metal_sum.DIAGONALS_PER_PARTIAL)

        with mock.patch.object(
            core, "_metal_sum_is_worthwhile", return_value=True
        ):
            actual = mp.selfjoin_sum(
                series,
                m,
                threshold=threshold,
                precision="single",
                gpus=[0],
            )

        self.assertLess(float(np.max(np.abs(actual - expected))), 5e-4)

    def test_shorter_checkpoints_bound_known_threshold_drift(self):
        series = np.random.default_rng(7_719).normal(size=2_902).astype(np.float32)
        m = 31
        threshold = 0.1
        expected = mp.selfjoin_sum(
            series,
            m,
            threshold=threshold,
            precision="single",
            gpus=[],
        )

        with mock.patch.object(
            core, "_metal_sum_is_worthwhile", return_value=True
        ):
            actual = mp.selfjoin_sum(
                series,
                m,
                threshold=threshold,
                precision="single",
                gpus=[0],
            )

        # Float32 correlations arbitrarily close to the strict threshold can
        # still fall on either side. This regression instead ensures recurrence
        # drift stays below the full ~0.1 contribution lost with 256 steps.
        self.assertLess(float(np.max(np.abs(actual - expected))), 5e-4)

    def test_large_window_uses_only_linear_recurrence_arrays(self):
        rng = np.random.default_rng(1025)
        series = rng.normal(size=10_000).astype(np.float32)
        m = 1024
        captured = {}

        def inspect_preparation(prepared_a, prepared_b, *_args):
            self.assertIs(prepared_a, prepared_b)
            self.assertIsNone(prepared_a.windows)
            self.assertIsNone(prepared_a.valid)
            arrays = (
                prepared_a.recurrence_clean,
                prepared_a.recurrence_means,
                prepared_a.recurrence_inv_norm,
                prepared_a.recurrence_df,
                prepared_a.recurrence_dg,
            )
            captured["elements"] = sum(int(array.size) for array in arrays)
            return np.zeros((prepared_a.subsequences,), dtype=np.float64)

        with (
            mock.patch.object(
                core, "_metal_sum_is_worthwhile", return_value=True
            ),
            mock.patch.object(
                core,
                "_prepare_series",
                side_effect=AssertionError("normalized windows were materialized"),
            ),
            mock.patch.object(
                _metal_sum, "sum_threshold", side_effect=inspect_preparation
            ),
        ):
            profile = mp.selfjoin_sum(
                series,
                m,
                threshold=0.2,
                precision="single",
                gpus=[0],
            )

        self.assertEqual(series.size - m + 1, profile.size)
        self.assertLessEqual(captured["elements"], 5 * series.size)

    def test_small_join_keeps_faster_portable_reducer(self):
        series = np.arange(512, dtype=np.float32)
        with mock.patch.object(_metal_sum, "sum_threshold") as kernel:
            mp.selfjoin_sum(
                series,
                16,
                threshold=0.25,
                precision="single",
                gpus=[0],
            )
        kernel.assert_not_called()

    def test_cpu_and_non_float32_inputs_use_portable_reducer(self):
        float32_series = np.arange(48, dtype=np.float32)
        float64_series = np.arange(48, dtype=np.float64)
        cases = (
            (float32_series, {"gpus": []}),
            (float64_series, {"gpus": [0]}),
        )
        for series, resources in cases:
            with self.subTest(dtype=series.dtype, resources=resources):
                with mock.patch.object(_metal_sum, "sum_threshold") as kernel:
                    mp.selfjoin_sum(
                        series,
                        8,
                        threshold=0.0,
                        precision="single",
                        **resources,
                    )
                kernel.assert_not_called()

    def test_negative_threshold_uses_the_faster_portable_reducer(self):
        series = np.arange(48, dtype=np.float32)
        with mock.patch.object(_metal_sum, "sum_threshold") as kernel:
            mp.selfjoin_sum(
                series,
                8,
                threshold=-0.1,
                precision="single",
                gpus=[0],
            )
        kernel.assert_not_called()

    def test_double_and_ultra_keep_the_cpu_reducer(self):
        series = np.arange(48, dtype=np.float32)
        for precision in ("double", "ultra"):
            with self.subTest(precision=precision):
                with mock.patch.object(_metal_sum, "sum_threshold") as kernel:
                    mp.selfjoin_sum(
                        series,
                        8,
                        threshold=0.0,
                        precision=precision,
                    )
                kernel.assert_not_called()

    def test_kernel_failure_is_not_silently_hidden(self):
        series = np.arange(48, dtype=np.float32)
        with (
            mock.patch.object(
                core, "_metal_sum_is_worthwhile", return_value=True
            ),
            mock.patch.object(_metal_sum, "sum_threshold") as kernel,
        ):
            kernel.side_effect = RuntimeError("custom Metal compilation failed")
            with self.assertRaisesRegex(RuntimeError, "compilation failed"):
                mp.selfjoin_sum(
                    series,
                    8,
                    threshold=0.0,
                    precision="single",
                )


if __name__ == "__main__":
    unittest.main()
