import unittest

import numpy as np

import pyscamp as mp

from reference import distance_matrix, reduce_1nn_index, reduce_matrix, reduce_sum_thresh


class PyScampCompatTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        self.a = rng.random(256, dtype=np.float32)
        self.b = rng.random(256, dtype=np.float32)
        self.m = 32

    def test_gpu_supported(self):
        self.assertTrue(mp.gpu_supported())

    def test_selfjoin_matches_reference(self):
        dm = distance_matrix(self.a, None, self.m)
        valid_dist, valid_idx = reduce_1nn_index(dm)
        out_dist, out_idx = mp.selfjoin(self.a, self.m, pearson=True)
        np.testing.assert_allclose(valid_dist, out_dist, equal_nan=True, rtol=1e-4, atol=1e-4)
        np.testing.assert_array_equal(valid_idx, out_idx)

    def test_abjoin_matches_reference(self):
        dm = distance_matrix(self.a, self.b, self.m)
        valid_dist, valid_idx = reduce_1nn_index(dm)
        out_dist, out_idx = mp.abjoin(self.a, self.b, self.m, pearson=True)
        np.testing.assert_allclose(valid_dist, out_dist, equal_nan=True, rtol=1e-4, atol=1e-4)
        np.testing.assert_array_equal(valid_idx, out_idx)

    def test_sum_profile_matches_reference(self):
        dm = distance_matrix(self.a, self.b, self.m)
        valid = reduce_sum_thresh(dm, 0.5)
        out = mp.abjoin_sum(self.a, self.b, self.m, threshold=0.5, pearson=True)
        np.testing.assert_allclose(valid, out, rtol=1e-4, atol=1e-4)

    def test_matrix_summary_matches_reference(self):
        dm = distance_matrix(self.a, None, self.m)
        valid = reduce_matrix(dm, 4, 5, True)
        out = mp.selfjoin_matrix(self.a, self.m, threshold=0.0, mheight=4, mwidth=5, pearson=True)
        np.testing.assert_allclose(valid, out, equal_nan=True, rtol=1e-4, atol=1e-4)

    def test_knn_contains_valid_top_matches(self):
        dm = distance_matrix(self.a, self.b, self.m)
        matches = mp.abjoin_knn(self.a, self.b, self.m, 3, threshold=0.2, pearson=True)
        self.assertTrue(matches)
        grouped = {}
        for col, row, corr in matches:
            grouped.setdefault(col, []).append((row, corr))
            self.assertGreaterEqual(corr, 0.2)
            np.testing.assert_allclose(dm[row, col], corr, rtol=1e-4, atol=1e-4)
        for col, group in grouped.items():
            self.assertLessEqual(len(group), 3)
            best_row = int(np.argmax(dm[:, col]))
            self.assertEqual(group[0][0], best_row)

    def test_nan_windows_are_excluded(self):
        arr = self.a.copy()
        arr[10] = np.nan
        out_dist, out_idx = mp.selfjoin(arr, self.m, pearson=True)
        self.assertTrue(np.isnan(out_dist[:11]).any())
        self.assertIn(-1, out_idx.tolist())

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


if __name__ == "__main__":
    unittest.main()
