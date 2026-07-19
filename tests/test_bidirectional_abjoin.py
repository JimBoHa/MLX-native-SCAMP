import unittest
from unittest import mock

import mlx.core as mx
import numpy as np

import mlx_native_scamp as mp
import pyscamp
from mlx_native_scamp import _metal_1nn, core


class _BidirectionalAssertions:
    def _directional_results(self, a, b, m, **kwargs):
        return (
            mp.abjoin(a, b, m, **kwargs),
            mp.abjoin(b, a, m, **kwargs),
        )

    def _assert_matches_directional(self, actual, expected):
        for actual_axis, expected_axis in zip(actual, expected, strict=True):
            actual_profile, actual_index = actual_axis
            expected_profile, expected_index = expected_axis
            np.testing.assert_allclose(
                actual_profile,
                expected_profile,
                rtol=2e-5,
                atol=2e-5,
                equal_nan=True,
            )
            np.testing.assert_array_equal(actual_index, expected_index)


class BidirectionalABJoinTests(_BidirectionalAssertions, unittest.TestCase):
    def test_portable_reduces_both_axes_with_one_block_traversal(self):
        rng = np.random.default_rng(302)
        a = rng.normal(size=611).astype(np.float32)
        b = rng.normal(size=587).astype(np.float32)
        options = {"pearson": True, "gpus": []}
        expected = self._directional_results(a, b, 8, **options)

        with mock.patch.object(
            core, "_iterate_blocks", wraps=core._iterate_blocks
        ) as traversal:
            actual = mp.abjoin_bidirectional(a, b, 8, **options)

        traversal.assert_called_once()
        self._assert_matches_directional(actual, expected)

    def test_portable_preserves_invalid_windows_and_euclidean_output(self):
        rng = np.random.default_rng(71)
        a = rng.normal(size=83).astype(np.float32)
        b = rng.normal(size=91).astype(np.float32)
        m = 7
        a[18] = np.nan
        b[39] = np.inf
        options = {"pearson": False, "precision": "ultra", "gpus": []}

        actual = mp.abjoin_bidirectional(a, b, m, **options)
        expected = self._directional_results(a, b, m, **options)

        self._assert_matches_directional(actual, expected)
        invalid_a = slice(18 - m + 1, 19)
        invalid_b = slice(39 - m + 1, 40)
        self.assertTrue(np.all(np.isnan(actual[0][0][invalid_a])))
        self.assertTrue(np.all(actual[0][1][invalid_a] == -1))
        self.assertTrue(np.all(np.isnan(actual[1][0][invalid_b])))
        self.assertTrue(np.all(actual[1][1][invalid_b] == -1))

    def test_portable_ties_choose_smallest_opposite_axis_index(self):
        pattern = np.array([0.0, 1.0, 3.0, 2.0], dtype=np.float32)
        a = np.tile(pattern, 9)
        b = np.tile(pattern, 10)

        result_a, result_b = mp.abjoin_bidirectional(
            a,
            b,
            4,
            pearson=True,
            precision="double",
            gpus=[],
        )

        np.testing.assert_array_equal(
            result_a[1], np.arange(result_a[1].size) % pattern.size
        )
        np.testing.assert_array_equal(
            result_b[1], np.arange(result_b[1].size) % pattern.size
        )

    def test_aligned_exclusion_matches_both_directional_joins(self):
        rng = np.random.default_rng(1313)
        a = rng.normal(size=89).astype(np.float32)
        b = a.copy()
        m = 9
        exclusion = (m + 3) // 4
        options = {
            "pearson": True,
            "allow_trivial_match": False,
            "gpus": [],
        }

        actual = mp.abjoin_bidirectional(a, b, m, **options)
        expected = self._directional_results(a, b, m, **options)

        self._assert_matches_directional(actual, expected)
        for _, indices in actual:
            positions = np.arange(indices.size)
            self.assertTrue(np.all(indices >= 0))
            self.assertTrue(np.all(np.abs(indices - positions) >= exclusion))

    def test_native_extension_does_not_expand_pyscamp_surface(self):
        self.assertIn("abjoin_bidirectional", mp.__all__)
        self.assertFalse(hasattr(pyscamp, "abjoin_bidirectional"))

        a = np.arange(16, dtype=np.float32)
        b = a[::-1].copy()
        with self.assertRaisesRegex(ValueError, "m must be at least 3"):
            mp.abjoin_bidirectional(a, b, 2)
        with self.assertRaisesRegex(ValueError, "a must be a 1D array"):
            mp.abjoin_bidirectional(a.reshape(4, 4), b, 3)
        with self.assertRaisesRegex(TypeError, "boolean-compatible"):
            mp.abjoin_bidirectional(a, b, 3, allow_trivial_match="false")
        with self.assertRaisesRegex(ValueError, "Concurrent CPU and Metal"):
            mp.abjoin_bidirectional(a, b, 3, gpus=[0], threads=1)


@unittest.skipUnless(mx.metal.is_available(), "Metal is unavailable")
class MetalBidirectionalABJoinTests(
    _BidirectionalAssertions, unittest.TestCase
):
    def setUp(self):
        self.previous_device = mx.default_device()
        mx.set_default_device(mx.gpu)

    def tearDown(self):
        mx.set_default_device(self.previous_device)

    def test_metal_uses_one_bidirectional_kernel_pair(self):
        rng = np.random.default_rng(808)
        a = rng.normal(size=193).astype(np.float32)
        b = rng.normal(size=181).astype(np.float32)
        a[71] = np.nan
        b[114] = np.inf
        options = {
            "pearson": True,
            "precision": "single",
            "gpus": [0],
        }
        expected = self._directional_results(a, b, 9, **options)

        with (
            mock.patch.object(
                _metal_1nn,
                "_BIDIRECTIONAL_PROFILE_KERNEL",
                wraps=_metal_1nn._BIDIRECTIONAL_PROFILE_KERNEL,
            ) as profile_kernel,
            mock.patch.object(
                _metal_1nn,
                "_BIDIRECTIONAL_INDEX_KERNEL",
                wraps=_metal_1nn._BIDIRECTIONAL_INDEX_KERNEL,
            ) as index_kernel,
            mock.patch.object(
                _metal_1nn,
                "_PROFILE_KERNEL",
                wraps=_metal_1nn._PROFILE_KERNEL,
            ) as one_way_profile,
            mock.patch.object(
                _metal_1nn,
                "_INDEX_KERNEL",
                wraps=_metal_1nn._INDEX_KERNEL,
            ) as one_way_index,
        ):
            actual = mp.abjoin_bidirectional(a, b, 9, **options)

        profile_kernel.assert_called_once()
        index_kernel.assert_called_once()
        one_way_profile.assert_not_called()
        one_way_index.assert_not_called()
        self._assert_matches_directional(actual, expected)

    def test_metal_ties_choose_smallest_opposite_axis_index(self):
        pattern = np.array([0.0, 1.0, 3.0, 2.0], dtype=np.float32)
        a = np.tile(pattern, 16)
        b = np.tile(pattern, 15)

        result_a, result_b = mp.abjoin_bidirectional(
            a,
            b,
            4,
            pearson=True,
            precision="single",
            gpus=[0],
        )

        np.testing.assert_array_equal(
            result_a[1], np.arange(result_a[1].size) % pattern.size
        )
        np.testing.assert_array_equal(
            result_b[1], np.arange(result_b[1].size) % pattern.size
        )

    def test_metal_aligned_exclusion_matches_directional_joins(self):
        rng = np.random.default_rng(2727)
        a = rng.normal(size=97).astype(np.float32)
        b = a.copy()
        m = 11
        options = {
            "pearson": True,
            "precision": "single",
            "gpus": [0],
            "allow_trivial_match": False,
        }

        actual = mp.abjoin_bidirectional(a, b, m, **options)
        expected = self._directional_results(a, b, m, **options)

        self._assert_matches_directional(actual, expected)


if __name__ == "__main__":
    unittest.main()
