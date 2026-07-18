import unittest

import mlx.core as mx
import numpy as np

import pyscamp as mp

from reference import corr_to_euclidean, distance_matrix, reduce_1nn_index, reduce_matrix, reduce_sum_thresh


EXPECTED_PUBLIC_CALLABLES = {
    "gpu_supported",
    "selfjoin",
    "abjoin",
    "selfjoin_sum",
    "abjoin_sum",
    "selfjoin_knn",
    "abjoin_knn",
    "selfjoin_matrix",
    "abjoin_matrix",
}


class PyScampCompatTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        self.a = rng.random(256, dtype=np.float32)
        self.b = rng.random(256, dtype=np.float32)
        self.m = 32
        self.dm_self = distance_matrix(self.a, None, self.m)
        self.dm_ab = distance_matrix(self.a, self.b, self.m)

    def _assert_knn_matches_reference(self, matches, dm, threshold):
        self.assertTrue(matches)
        grouped = {}
        for col, row, corr in matches:
            grouped.setdefault(col, []).append((row, corr))
            self.assertGreaterEqual(corr, threshold)
            np.testing.assert_allclose(dm[row, col], corr, rtol=1e-4, atol=1e-4)
        for col, group in grouped.items():
            self.assertLessEqual(len(group), 3)
            best_row = int(np.nanargmax(dm[:, col]))
            self.assertEqual(group[0][0], best_row)

    def test_public_surface_matches_upstream_bindings(self):
        exported = {name for name in EXPECTED_PUBLIC_CALLABLES if hasattr(mp, name)}
        self.assertEqual(EXPECTED_PUBLIC_CALLABLES, exported)
        self.assertEqual("dev", mp.__version__)

    def test_gpu_supported(self):
        self.assertTrue(mp.gpu_supported())

    def test_selfjoin_matches_reference(self):
        valid_dist, valid_idx = reduce_1nn_index(self.dm_self)
        out_dist, out_idx = mp.selfjoin(self.a, self.m, pearson=True)
        np.testing.assert_allclose(valid_dist, out_dist, equal_nan=True, rtol=1e-4, atol=1e-4)
        np.testing.assert_array_equal(valid_idx, out_idx)

    def test_abjoin_matches_reference(self):
        valid_dist, valid_idx = reduce_1nn_index(self.dm_ab)
        out_dist, out_idx = mp.abjoin(self.a, self.b, self.m, pearson=True)
        np.testing.assert_allclose(valid_dist, out_dist, equal_nan=True, rtol=1e-4, atol=1e-4)
        np.testing.assert_array_equal(valid_idx, out_idx)

    def test_selfjoin_euclidean_matches_reference_conversion(self):
        valid_corr, valid_idx = reduce_1nn_index(self.dm_self)
        valid_dist = corr_to_euclidean(valid_corr, self.m)
        out_dist, out_idx = mp.selfjoin(self.a, self.m, pearson=False)
        np.testing.assert_allclose(valid_dist, out_dist, equal_nan=True, rtol=1e-4, atol=1e-4)
        np.testing.assert_array_equal(valid_idx, out_idx)

    def test_selfjoin_sum_matches_reference(self):
        valid = reduce_sum_thresh(self.dm_self, 0.5)
        out = mp.selfjoin_sum(self.a, self.m, threshold=0.5, pearson=True)
        np.testing.assert_allclose(valid, out, rtol=1e-4, atol=1e-4)

    def test_abjoin_sum_matches_reference(self):
        valid = reduce_sum_thresh(self.dm_ab, 0.5)
        out = mp.abjoin_sum(self.a, self.b, self.m, threshold=0.5, pearson=True)
        np.testing.assert_allclose(valid, out, rtol=1e-4, atol=1e-4)

    def test_selfjoin_matrix_matches_reference(self):
        valid = reduce_matrix(self.dm_self, 4, 5, True)
        out = mp.selfjoin_matrix(self.a, self.m, threshold=0.0, mheight=4, mwidth=5, pearson=True)
        np.testing.assert_allclose(valid, out, equal_nan=True, rtol=1e-4, atol=1e-4)

    def test_abjoin_matrix_matches_reference(self):
        valid = reduce_matrix(self.dm_ab, 4, 5, False)
        out = mp.abjoin_matrix(self.a, self.b, self.m, threshold=0.0, mheight=4, mwidth=5, pearson=True)
        np.testing.assert_allclose(valid, out, equal_nan=True, rtol=1e-4, atol=1e-4)

    def test_abjoin_matrix_euclidean_matches_reference_conversion(self):
        valid = corr_to_euclidean(reduce_matrix(self.dm_ab, 4, 5, False), self.m)
        out = mp.abjoin_matrix(self.a, self.b, self.m, threshold=0.0, mheight=4, mwidth=5, pearson=False)
        np.testing.assert_allclose(valid, out, equal_nan=True, rtol=1e-4, atol=1e-4)

    def test_selfjoin_knn_contains_valid_top_matches(self):
        matches = mp.selfjoin_knn(self.a, self.m, 3, threshold=0.2, pearson=True)
        self._assert_knn_matches_reference(matches, self.dm_self, 0.2)

    def test_abjoin_knn_contains_valid_top_matches(self):
        matches = mp.abjoin_knn(self.a, self.b, self.m, 3, threshold=0.2, pearson=True)
        self._assert_knn_matches_reference(matches, self.dm_ab, 0.2)

    def test_nan_windows_are_excluded(self):
        arr = self.a.copy()
        arr[10] = np.nan
        out_dist, out_idx = mp.selfjoin(arr, self.m, pearson=True)
        self.assertTrue(np.isnan(out_dist[:11]).any())
        self.assertIn(-1, out_idx.tolist())

    def test_mlx_array_inputs_are_supported(self):
        valid_dist, valid_idx = reduce_1nn_index(self.dm_ab)
        out_dist, out_idx = mp.abjoin(mx.array(self.a), mx.array(self.b), self.m, pearson=True)
        np.testing.assert_allclose(valid_dist, out_dist, equal_nan=True, rtol=1e-4, atol=1e-4)
        np.testing.assert_array_equal(valid_idx, out_idx)

    def test_compatibility_kwargs_are_accepted(self):
        out_dist, out_idx = mp.selfjoin(
            self.a,
            self.m,
            pearson=True,
            gpus=[],
            threads=2,
            precision="double",
            verbose=True,
        )
        self.assertEqual(out_dist.shape, out_idx.shape)

    def test_invalid_kwargs_raise(self):
        with self.assertRaises(ValueError):
            mp.selfjoin(self.a, self.m, nope=True)

    def test_window_size_accepts_index_compatible_integers(self):
        out_dist, out_idx = mp.selfjoin(self.a, np.int64(self.m), pearson=True)
        self.assertEqual(out_dist.shape, out_idx.shape)

    def test_window_size_rejects_non_integer_values(self):
        for value in (3.0, np.float64(3.0), "3", None):
            with self.subTest(value=value):
                with self.assertRaisesRegex(TypeError, "^m must be an integer$"):
                    mp.selfjoin(self.a, value)

    def test_window_size_must_be_at_least_three_for_self_and_ab_joins(self):
        with self.assertRaisesRegex(ValueError, "^m must be at least 3$"):
            mp.selfjoin(self.a, 2)
        with self.assertRaisesRegex(ValueError, "^m must be at least 3$"):
            mp.abjoin(self.a, self.b, 2)

    def test_window_size_cannot_exceed_input_lengths(self):
        with self.assertRaisesRegex(ValueError, r"^m must be less than or equal to len\(a\)$"):
            mp.selfjoin(self.a[:3], 4)
        with self.assertRaisesRegex(ValueError, r"^m must be less than or equal to len\(a\)$"):
            mp.abjoin(self.a[:3], self.b[:4], 4)
        with self.assertRaisesRegex(ValueError, r"^m must be less than or equal to len\(b\)$"):
            mp.abjoin(self.a[:4], self.b[:3], 4)

    def test_selfjoin_matrix_dimensions_cannot_exceed_subsequence_count(self):
        subsequences = len(self.a) - self.m + 1
        with self.assertRaisesRegex(
            ValueError,
            "^mwidth must be less than or equal to the number of subsequences in a$",
        ):
            mp.selfjoin_matrix(self.a, self.m, mwidth=subsequences + 1, mheight=1)
        with self.assertRaisesRegex(
            ValueError,
            "^mheight must be less than or equal to the number of subsequences in a$",
        ):
            mp.selfjoin_matrix(self.a, self.m, mwidth=1, mheight=subsequences + 1)

    def test_abjoin_matrix_dimensions_use_a_width_and_b_height(self):
        a = self.a[:96]
        b = self.b[:80]
        m = 16
        subsequences_a = len(a) - m + 1
        subsequences_b = len(b) - m + 1
        with self.assertRaisesRegex(
            ValueError,
            "^mwidth must be less than or equal to the number of subsequences in a$",
        ):
            mp.abjoin_matrix(a, b, m, mwidth=subsequences_a + 1, mheight=1)
        with self.assertRaisesRegex(
            ValueError,
            "^mheight must be less than or equal to the number of subsequences in b$",
        ):
            mp.abjoin_matrix(a, b, m, mwidth=1, mheight=subsequences_b + 1)

    def test_matrix_dimensions_may_equal_subsequence_counts(self):
        a = np.arange(10, dtype=np.float32)
        b = np.arange(8, dtype=np.float32)
        self_out = mp.selfjoin_matrix(a, 3, mwidth=8, mheight=8, pearson=True)
        ab_out = mp.abjoin_matrix(a, b, 3, mwidth=8, mheight=6, pearson=True)
        self.assertEqual((8, 8), self_out.shape)
        self.assertEqual((6, 8), ab_out.shape)


if __name__ == "__main__":
    unittest.main()
