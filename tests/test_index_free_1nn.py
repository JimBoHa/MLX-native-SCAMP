import unittest
from unittest import mock

import mlx.core as mx
import numpy as np

import mlx_native_scamp as mp
from mlx_native_scamp import _metal_1nn


class IndexFree1NNTests(unittest.TestCase):
    def setUp(self):
        self.previous_device = mx.default_device()

    def tearDown(self):
        mx.set_default_device(self.previous_device)

    def test_cpu_profiles_match_indexed_values(self):
        rng = np.random.default_rng(44)
        a = rng.normal(size=91).astype(np.float32)
        b = rng.normal(size=83).astype(np.float32)

        expected_self = mp.selfjoin(a, 9, pearson=True, gpus=[])[0]
        expected_ab = mp.abjoin(a, b, 9, pearson=False, gpus=[])[0]

        np.testing.assert_array_equal(
            mp.selfjoin_1nn(a, 9, pearson=True, gpus=[]), expected_self
        )
        np.testing.assert_array_equal(
            mp.abjoin_1nn(a, b, 9, pearson=False, gpus=[]), expected_ab
        )

    @unittest.skipUnless(mx.metal.is_available(), "Metal is unavailable")
    def test_metal_profile_omits_index_pass(self):
        mx.set_default_device(mx.gpu)
        rng = np.random.default_rng(81)
        series = rng.normal(size=257).astype(np.float32)
        expected = mp.selfjoin(series, 11, pearson=True, precision="single")[0]

        with mock.patch.object(_metal_1nn, "_INDEX_KERNEL") as index_kernel:
            actual = mp.selfjoin_1nn(
                series,
                11,
                pearson=True,
                precision="single",
            )

        index_kernel.assert_not_called()
        np.testing.assert_array_equal(actual, expected)

    @unittest.skipUnless(mx.metal.is_available(), "Metal is unavailable")
    def test_metal_ab_profile_matches_indexed_values_with_invalid_windows(self):
        mx.set_default_device(mx.gpu)
        rng = np.random.default_rng(19)
        a = rng.normal(size=129).astype(np.float32)
        b = rng.normal(size=117).astype(np.float32)
        b[41] = np.nan
        expected = mp.abjoin(a, b, 8, pearson=False, precision="single")[0]

        actual = mp.abjoin_1nn(a, b, 8, pearson=False, precision="single")

        np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
