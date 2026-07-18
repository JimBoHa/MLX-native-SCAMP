import io
import re
import unittest
from contextlib import redirect_stdout

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
        with redirect_stdout(io.StringIO()):
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

    def test_verbose_reports_each_join_and_profile(self):
        rng = np.random.default_rng(7)
        a = rng.normal(size=10).astype(np.float32)
        b = rng.normal(size=12).astype(np.float32)
        m = 4
        a_subsequences = len(a) - m + 1
        b_subsequences = len(b) - m + 1
        cases = [
            ("selfjoin/1nn", a_subsequences, lambda: mp.selfjoin(a, m, verbose=True)),
            ("abjoin/1nn", b_subsequences, lambda: mp.abjoin(a, b, m, verbose=True)),
            ("selfjoin/sum", a_subsequences, lambda: mp.selfjoin_sum(a, m, verbose=True)),
            ("abjoin/sum", b_subsequences, lambda: mp.abjoin_sum(a, b, m, verbose=True)),
            (
                "selfjoin/matrix",
                a_subsequences,
                lambda: mp.selfjoin_matrix(a, m, mheight=2, mwidth=3, verbose=True),
            ),
            (
                "abjoin/matrix",
                b_subsequences,
                lambda: mp.abjoin_matrix(a, b, m, mheight=2, mwidth=3, verbose=True),
            ),
            ("selfjoin/knn", a_subsequences, lambda: mp.selfjoin_knn(a, m, 2, verbose=True)),
            ("abjoin/knn", b_subsequences, lambda: mp.abjoin_knn(a, b, m, 2, verbose=True)),
        ]

        for label, expected_b_subsequences, run_profile in cases:
            with self.subTest(profile=label):
                stdout = io.StringIO()
                with redirect_stdout(stdout):
                    run_profile()
                lines = stdout.getvalue().splitlines()

                self.assertEqual(2, len(lines))
                start = (
                    f"pyscamp {label} start: "
                    f"a_subsequences={a_subsequences} "
                    f"b_subsequences={expected_b_subsequences} "
                    f"window={m} device="
                )
                self.assertRegex(lines[0], rf"^{re.escape(start)}(?:cpu|gpu|unknown)$")
                self.assertRegex(
                    lines[1],
                    rf"^pyscamp {re.escape(label)} complete: elapsed=\d+\.\d{{6}}s$",
                )

    def test_profiles_are_silent_by_default(self):
        rng = np.random.default_rng(8)
        a = rng.normal(size=10).astype(np.float32)
        m = 4
        profiles = [
            lambda: mp.selfjoin(a, m),
            lambda: mp.selfjoin_sum(a, m),
            lambda: mp.selfjoin_matrix(a, m, mheight=2, mwidth=3),
            lambda: mp.selfjoin_knn(a, m, 2),
        ]

        for run_profile in profiles:
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                run_profile()
            self.assertEqual("", stdout.getvalue())

    def test_invalid_kwargs_raise(self):
        with self.assertRaises(ValueError):
            mp.selfjoin(self.a, self.m, nope=True)


if __name__ == "__main__":
    unittest.main()
