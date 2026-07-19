import unittest
from unittest import mock

import mlx.core as mx
import numpy as np

import mlx_native_scamp as mp
from mlx_native_scamp import _metal_1nn
from reference import distance_matrix, reduce_1nn_index, reduce_sum_thresh


@unittest.skipUnless(mx.metal.is_available(), "Metal is unavailable")
class MetalDiagonal1NNTests(unittest.TestCase):
    def setUp(self):
        self.previous_device = mx.default_device()
        mx.set_default_device(mx.gpu)

    def tearDown(self):
        mx.set_default_device(self.previous_device)

    def test_default_single_metal_matches_reference_with_invalid_windows(self):
        rng = np.random.default_rng(1107)
        series = rng.normal(size=97).astype(np.float32)
        series[17] = np.nan
        m = 8
        expected, expected_index = reduce_1nn_index(distance_matrix(series, None, m))

        with mock.patch.object(_metal_1nn, "best_match", wraps=_metal_1nn.best_match) as kernel:
            actual, actual_index = mp.selfjoin(series, m, pearson=True, precision="single")

        kernel.assert_called_once()
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5, equal_nan=True)
        np.testing.assert_array_equal(actual_index, expected_index)

    def test_explicit_metal_abjoin_matches_exact_indices(self):
        rng = np.random.default_rng(2024)
        a = rng.normal(size=83).astype(np.float32)
        b = rng.normal(size=101).astype(np.float32)
        b[31] = np.inf
        m = 8
        expected, expected_index = reduce_1nn_index(distance_matrix(a, b, m))

        with mx.stream(mx.cpu):
            with mock.patch.object(
                _metal_1nn, "best_match", wraps=_metal_1nn.best_match
            ) as kernel:
                actual, actual_index = mp.abjoin(
                    a,
                    b,
                    m,
                    pearson=True,
                    precision="single",
                    gpus=[0],
                )

        kernel.assert_called_once()
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5, equal_nan=True)
        np.testing.assert_array_equal(actual_index, expected_index)

    def test_non_multiple_of_four_exclusion_matches_reference_on_metal(self):
        rng = np.random.default_rng(1)
        series = np.cumsum(rng.normal(size=37)).astype(np.float32)

        with mock.patch.object(
            _metal_1nn,
            "best_match",
            wraps=_metal_1nn.best_match,
        ) as kernel:
            for m in (3, 5):
                with self.subTest(m=m):
                    dm = distance_matrix(series, None, m)
                    expected_corr, expected_index = reduce_1nn_index(dm)
                    expected_sum = reduce_sum_thresh(dm, -1.0)

                    actual_corr, actual_index = mp.selfjoin(
                        series,
                        m,
                        pearson=True,
                        precision="single",
                        gpus=[0],
                    )
                    actual_sum = mp.selfjoin_sum(
                        series,
                        m,
                        threshold=-1.0,
                        precision="single",
                        gpus=[0],
                    )

                    np.testing.assert_allclose(
                        actual_corr,
                        expected_corr,
                        rtol=2e-5,
                        atol=2e-5,
                        equal_nan=True,
                    )
                    np.testing.assert_array_equal(actual_index, expected_index)
                    np.testing.assert_allclose(
                        actual_sum,
                        expected_sum,
                        rtol=2e-5,
                        atol=2e-5,
                    )

        self.assertEqual(2, kernel.call_count)

    def test_cpu_single_uses_portable_path(self):
        rng = np.random.default_rng(8)
        series = rng.normal(size=64).astype(np.float32)
        expected, expected_index = reduce_1nn_index(distance_matrix(series, None, 8))

        with mock.patch.object(_metal_1nn, "best_match") as kernel:
            actual, actual_index = mp.selfjoin(
                series,
                8,
                pearson=True,
                precision="single",
                gpus=[],
            )

        kernel.assert_not_called()
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)
        np.testing.assert_array_equal(actual_index, expected_index)

    def test_extreme_float32_magnitudes_use_portable_path(self):
        series = np.array(
            [1e20, -1e20, 5e19, -5e19, 8e19, -8e19, 3e19, -3e19],
            dtype=np.float32,
        )
        expected = mp.abjoin(
            series,
            series[::-1].copy(),
            4,
            pearson=True,
            precision="single",
            gpus=[],
        )

        with mock.patch.object(_metal_1nn, "best_match") as kernel:
            actual = mp.abjoin(
                series,
                series[::-1].copy(),
                4,
                pearson=True,
                precision="single",
                gpus=[0],
            )

        kernel.assert_not_called()
        np.testing.assert_allclose(actual[0], expected[0], equal_nan=True)
        np.testing.assert_array_equal(actual[1], expected[1])

    def test_non_float32_high_offset_input_uses_portable_path(self):
        series = np.array(
            [1e8, 1e8 + 1, 1e8 + 2, 1e8 + 4, 1e8 + 8],
            dtype=np.float64,
        )
        expected = mp.abjoin(
            series,
            series,
            4,
            pearson=True,
            precision="single",
            gpus=[],
        )

        with mock.patch.object(_metal_1nn, "best_match") as kernel:
            actual = mp.abjoin(
                series,
                series,
                4,
                pearson=True,
                precision="single",
                gpus=[0],
            )

        kernel.assert_not_called()
        np.testing.assert_allclose(actual[0], expected[0], equal_nan=True)
        np.testing.assert_array_equal(actual[1], expected[1])

    def test_double_and_ultra_keep_the_portable_path(self):
        series = np.arange(32, dtype=np.float32)
        for precision in ("double", "ultra"):
            with self.subTest(precision=precision):
                with mock.patch.object(_metal_1nn, "best_match") as kernel:
                    mp.selfjoin(series, 8, pearson=True, precision=precision)
                kernel.assert_not_called()

    def test_metal_kernel_failure_is_not_silently_hidden(self):
        series = np.arange(32, dtype=np.float32)
        with mock.patch.object(_metal_1nn, "best_match") as kernel:
            kernel.side_effect = RuntimeError("custom Metal compilation failed")
            with self.assertRaisesRegex(RuntimeError, "compilation failed"):
                mp.selfjoin(series, 8, pearson=True, precision="single")


if __name__ == "__main__":
    unittest.main()
