import unittest
from unittest import mock

import mlx.core as mx
import numpy as np

import mlx_native_scamp as mp
from mlx_native_scamp import _metal_matrix


@unittest.skipUnless(mx.metal.is_available(), "Metal is unavailable")
class MetalMatrixSummaryTests(unittest.TestCase):
    def setUp(self):
        self.previous_device = mx.default_device()
        mx.set_default_device(mx.gpu)

    def tearDown(self):
        mx.set_default_device(self.previous_device)

    def test_selfjoin_matches_portable_non_square_summary(self):
        rng = np.random.default_rng(515)
        series = rng.normal(size=137).astype(np.float32)
        expected = mp.selfjoin_matrix(
            series,
            8,
            mheight=11,
            mwidth=13,
            threshold=-0.4,
            pearson=True,
            precision="single",
            gpus=[],
        )

        with mock.patch.object(
            _metal_matrix, "_KERNEL", wraps=_metal_matrix._KERNEL
        ) as kernel:
            actual = mp.selfjoin_matrix(
                series,
                8,
                mheight=11,
                mwidth=13,
                threshold=-0.4,
                pearson=True,
                precision="single",
                gpus=[0],
            )

        kernel.assert_called_once()
        np.testing.assert_allclose(
            actual, expected, rtol=2e-5, atol=2e-5, equal_nan=True
        )

    def test_abjoin_matches_portable_with_invalid_windows(self):
        rng = np.random.default_rng(919)
        a = rng.normal(size=149).astype(np.float32)
        b = rng.normal(size=121).astype(np.float32)
        a[28] = np.nan
        b[77] = np.inf
        expected = mp.abjoin_matrix(
            a,
            b,
            8,
            mheight=9,
            mwidth=7,
            threshold=0.1,
            pearson=False,
            precision="single",
            gpus=[],
        )

        actual = mp.abjoin_matrix(
            a,
            b,
            8,
            mheight=9,
            mwidth=7,
            threshold=0.1,
            pearson=False,
            precision="single",
            gpus=[0],
        )

        np.testing.assert_allclose(
            actual, expected, rtol=2e-5, atol=2e-5, equal_nan=True
        )

    def test_non_float32_input_keeps_portable_path(self):
        series = np.linspace(1e8, 1e8 + 64, 65, dtype=np.float64)
        with mock.patch.object(_metal_matrix, "_KERNEL") as kernel:
            mp.selfjoin_matrix(
                series,
                8,
                mheight=5,
                mwidth=5,
                pearson=True,
                precision="single",
                gpus=[0],
            )
        kernel.assert_not_called()


if __name__ == "__main__":
    unittest.main()
