import math
import unittest
from unittest import mock

import numpy as np

from mlx_native_scamp import core


class RecurrencePreparationTests(unittest.TestCase):
    def test_vectorized_statistics_match_compensated_scalar(self):
        rng = np.random.default_rng(1907)
        high_offset = (
            1e7 + rng.integers(-64, 65, size=10_000)
        ).astype(np.float32)
        high_offset = high_offset.astype(np.float64) - float(high_offset[0])
        trend = (
            np.arange(10_000, dtype=np.float32) * np.float32(1e10)
        ).astype(np.float64)
        cases = (
            ("random", rng.normal(size=10_000).astype(np.float32).astype(np.float64)),
            ("high-offset", high_offset),
            ("trend", trend - trend[0]),
        )

        for name, clean in cases:
            with self.subTest(name=name):
                actual = core._rolling_mean_and_norm_sq_vectorized(clean, 128)
                expected = core._rolling_mean_and_norm_sq_scalar(clean, 128)

                self.assertIsNotNone(actual)
                self.assertIsNotNone(expected)
                np.testing.assert_allclose(
                    actual[0], expected[0], rtol=2e-13, atol=2e-13
                )
                np.testing.assert_allclose(
                    actual[1], expected[1], rtol=2e-13, atol=2e-13
                )

    def test_adversarial_inputs_retain_scalar_results(self):
        rng = np.random.default_rng(88)
        piecewise = np.zeros((4096,), dtype=np.float64)
        piecewise[511:1537] = 3.0
        piecewise[2048::17] = -7.0
        dynamic = np.empty((4096,), dtype=np.float64)
        dynamic[::4] = 0.0
        dynamic[1::4] = 1e20
        dynamic[2::4] = -1e20
        dynamic[3::4] = 1.0
        cases = (
            ("near-flat", rng.normal(scale=1e-8, size=4096), 32),
            ("piecewise-flat", piecewise, 64),
            ("dynamic-range", dynamic, 128),
        )

        for name, clean, m in cases:
            with self.subTest(name=name):
                actual = core._rolling_mean_and_norm_sq(clean, m)
                expected = core._rolling_mean_and_norm_sq_scalar(clean, m)

                self.assertIsNotNone(actual)
                self.assertIsNotNone(expected)
                np.testing.assert_allclose(
                    actual[0], expected[0], rtol=1e-11, atol=0.0
                )
                np.testing.assert_allclose(
                    actual[1], expected[1], rtol=1e-11, atol=0.0
                )

    def test_vectorized_preparation_preserves_metal_float32_arrays(self):
        source = (
            1e7
            + np.random.default_rng(701).integers(-32, 33, size=10_000)
        ).astype(np.float32)
        actual = core._prepare_metal_recurrence(source, 128)

        with mock.patch.object(
            core,
            "_rolling_mean_and_norm_sq",
            core._rolling_mean_and_norm_sq_scalar,
        ):
            expected = core._prepare_metal_recurrence(source, 128)

        self.assertIsNotNone(actual)
        self.assertIsNotNone(expected)
        for field in (
            "recurrence_clean",
            "recurrence_means",
            "recurrence_inv_norm",
            "recurrence_df",
            "recurrence_dg",
        ):
            with self.subTest(field=field):
                np.testing.assert_array_equal(
                    np.asarray(getattr(actual, field)),
                    np.asarray(getattr(expected, field)),
                )

    def test_suspicious_cancellation_uses_scalar_fallback(self):
        clean = (
            np.random.default_rng(0)
            .normal(size=512)
            .astype(np.float32)
            .astype(np.float64)
        )
        sentinel = (
            np.array([123.0], dtype=np.float64),
            np.array([456.0], dtype=np.float64),
        )

        self.assertIsNone(core._rolling_mean_and_norm_sq_vectorized(clean, 3))
        with mock.patch.object(
            core,
            "_rolling_mean_and_norm_sq_scalar",
            return_value=sentinel,
        ) as scalar:
            actual = core._rolling_mean_and_norm_sq(clean, 3)

        self.assertIs(actual, sentinel)
        scalar.assert_called_once_with(clean, 3)

    def test_first_outlier_stays_within_preparation_tolerance(self):
        m = 31
        n = 38
        rng = np.random.default_rng(7516)
        baseline = np.float32(1e10)
        source = np.r_[
            np.float32(0.0),
            (
                baseline
                + rng.integers(-256, 257, size=n - 1).astype(np.float32)
                * np.spacing(baseline)
            ).astype(np.float32),
        ]
        clean = source.astype(np.float64)

        statistics = core._rolling_mean_and_norm_sq_vectorized(clean, m)

        self.assertIsNotNone(statistics)
        vector_inv_norm = 1.0 / np.sqrt(statistics[1])
        direct_inv_norm = []
        for start in range(n - m + 1):
            window = clean[start : start + m]
            mean = math.fsum(window) / m
            norm_sq = math.fsum(
                (float(value) - mean) ** 2 for value in window
            )
            direct_inv_norm.append(1.0 / math.sqrt(norm_sq))

        np.testing.assert_allclose(
            vector_inv_norm,
            np.asarray(direct_inv_norm),
            rtol=1.0 / core.ROLLING_STATISTICS_SAFETY_FACTOR,
            atol=0.0,
        )

    def test_exact_flat_series_stays_on_vectorized_path(self):
        clean = np.zeros((100_000,), dtype=np.float64)

        with mock.patch.object(
            core,
            "_rolling_mean_and_norm_sq_scalar",
            side_effect=AssertionError("unexpected scalar fallback"),
        ):
            means, norms_sq = core._rolling_mean_and_norm_sq(clean, 1024)

        np.testing.assert_array_equal(means, 0.0)
        np.testing.assert_array_equal(norms_sq, 0.0)

    def test_checkpoint_work_remains_linear_for_large_windows(self):
        clean = (
            np.random.default_rng(31)
            .normal(size=100_000)
            .astype(np.float32)
            .astype(np.float64)
        )
        m = 8192

        with mock.patch.object(
            core.math, "fsum", wraps=math.fsum
        ) as faithful_sum:
            statistics = core._rolling_mean_and_norm_sq_vectorized(clean, m)

        self.assertIsNotNone(statistics)
        lengths = [len(call.args[0]) for call in faithful_sum.call_args_list]
        self.assertTrue(lengths)
        self.assertTrue(all(length == m for length in lengths))
        # Each checkpoint sums the window and its absolute values. Since the
        # block span is at least m, their aggregate work never forms n-by-m.
        self.assertLessEqual(sum(lengths), 2 * (clean.size + m))

    def test_extreme_nonfinite_statistics_are_rejected(self):
        clean = np.array([0.0, 1e308, -1e308, 1e308], dtype=np.float64)

        with np.errstate(over="ignore", invalid="ignore"):
            self.assertIsNone(
                core._rolling_mean_and_norm_sq_vectorized(clean, 3)
            )


if __name__ == "__main__":
    unittest.main()
