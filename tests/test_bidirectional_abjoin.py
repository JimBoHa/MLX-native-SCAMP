import unittest
from unittest import mock

import mlx.core as mx
import numpy as np

import mlx_native_scamp as mp
from mlx_native_scamp import _metal_1nn


class BidirectionalABJoinTests(unittest.TestCase):
    def setUp(self):
        self.previous_device = mx.default_device()

    def tearDown(self):
        mx.set_default_device(self.previous_device)

    def _assert_matches_two_joins(self, a, b, m, **kwargs):
        (profile_a, index_a), (profile_b, index_b) = mp.abjoin_bidirectional(
            a, b, m, **kwargs
        )
        expected_a, expected_index_a = mp.abjoin(a, b, m, **kwargs)
        expected_b, expected_index_b = mp.abjoin(b, a, m, **kwargs)
        np.testing.assert_allclose(
            profile_a, expected_a, rtol=2e-5, atol=2e-5, equal_nan=True
        )
        np.testing.assert_array_equal(index_a, expected_index_a)
        np.testing.assert_allclose(
            profile_b, expected_b, rtol=2e-5, atol=2e-5, equal_nan=True
        )
        np.testing.assert_array_equal(index_b, expected_index_b)

    def test_cpu_reduces_both_axes_with_one_block_traversal(self):
        rng = np.random.default_rng(302)
        a = rng.normal(size=83).astype(np.float32)
        b = rng.normal(size=71).astype(np.float32)
        self._assert_matches_two_joins(a, b, 8, pearson=True, gpus=[])

    def test_cpu_preserves_invalid_windows_and_euclidean_output(self):
        rng = np.random.default_rng(71)
        a = rng.normal(size=67).astype(np.float32)
        b = rng.normal(size=73).astype(np.float32)
        a[18] = np.nan
        b[39] = np.inf
        self._assert_matches_two_joins(a, b, 7, pearson=False, gpus=[])

    @unittest.skipUnless(mx.metal.is_available(), "Metal is unavailable")
    def test_metal_reduces_both_axes_in_one_kernel_pair(self):
        mx.set_default_device(mx.gpu)
        rng = np.random.default_rng(808)
        a = rng.normal(size=193).astype(np.float32)
        b = rng.normal(size=181).astype(np.float32)

        with mock.patch.object(
            _metal_1nn,
            "_BIDIRECTIONAL_PROFILE_KERNEL",
            wraps=_metal_1nn._BIDIRECTIONAL_PROFILE_KERNEL,
        ) as profile_kernel, mock.patch.object(
            _metal_1nn,
            "_BIDIRECTIONAL_INDEX_KERNEL",
            wraps=_metal_1nn._BIDIRECTIONAL_INDEX_KERNEL,
        ) as index_kernel:
            self._assert_matches_two_joins(
                a,
                b,
                9,
                pearson=True,
                precision="single",
            )

        profile_kernel.assert_called_once()
        index_kernel.assert_called_once()

    @unittest.skipUnless(mx.metal.is_available(), "Metal is unavailable")
    def test_metal_uses_smallest_index_for_ties(self):
        mx.set_default_device(mx.gpu)
        a = np.tile(np.array([0, 1, 3, 2], dtype=np.float32), 16)
        b = np.tile(np.array([3, 2, 0, 1], dtype=np.float32), 15)
        self._assert_matches_two_joins(
            a,
            b,
            4,
            pearson=True,
            precision="single",
        )


if __name__ == "__main__":
    unittest.main()
