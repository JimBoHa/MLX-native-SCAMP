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

    def _assert_knn_equals_reference(self, matches, dm, k, threshold):
        expected = []
        for col in range(dm.shape[1]):
            for row in np.argsort(dm[:, col])[::-1][:k]:
                corr = float(dm[row, col])
                if corr >= threshold and corr >= -1.0:
                    expected.append((col, int(row), corr))

        self.assertEqual(
            [(col, row) for col, row, _ in expected],
            [(col, row) for col, row, _ in matches],
        )
        np.testing.assert_allclose(
            [corr for _, _, corr in expected],
            [corr for _, _, corr in matches],
            rtol=1e-4,
            atol=1e-4,
        )

    def _aligned_ab_fixture(self):
        rng = np.random.default_rng(17)
        series = rng.standard_normal(64).astype(np.float32)
        m = 33
        floor_exclusion = m // 4
        for index in range(floor_exclusion, floor_exclusion + m):
            series[index] = series[index - floor_exclusion]

        dm = distance_matrix(series, series, m)
        exclusion = (m + 3) // 4
        rows = np.arange(dm.shape[0])[:, None]
        cols = np.arange(dm.shape[1])[None, :]
        dm[np.abs(rows - cols) < exclusion] = -2.0
        return series, m, dm

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

    def test_selfjoin_uses_ceil_exclusion_zone(self):
        arr = np.array([0, 1, 3, 2, 5, 4], dtype=np.float32)
        for m in (3, 5):
            valid_dist, valid_idx = reduce_1nn_index(distance_matrix(arr, None, m))
            out_dist, out_idx = mp.selfjoin(arr, m, pearson=True)
            np.testing.assert_allclose(valid_dist, out_dist, equal_nan=True, rtol=1e-4, atol=1e-4)
            np.testing.assert_array_equal(valid_idx, out_idx)

    def test_abjoin_matches_reference(self):
        valid_dist, valid_idx = reduce_1nn_index(self.dm_ab)
        out_dist, out_idx = mp.abjoin(self.a, self.b, self.m, pearson=True)
        np.testing.assert_allclose(valid_dist, out_dist, equal_nan=True, rtol=1e-4, atol=1e-4)
        np.testing.assert_array_equal(valid_idx, out_idx)

    def test_abjoin_can_exclude_aligned_trivial_matches(self):
        series, m, dm = self._aligned_ab_fixture()
        valid_dist, valid_idx = reduce_1nn_index(dm)

        out_dist, out_idx = mp.abjoin(
            series,
            series,
            m,
            pearson=True,
            allow_trivial_match=False,
        )

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

    def test_abjoin_sum_can_exclude_aligned_trivial_matches(self):
        series, m, dm = self._aligned_ab_fixture()
        valid = reduce_sum_thresh(dm, 0.2)

        out = mp.abjoin_sum(
            series,
            series,
            m,
            threshold=0.2,
            allow_trivial_match=False,
        )

        np.testing.assert_allclose(valid, out, rtol=1e-4, atol=1e-4)

    def test_selfjoin_matrix_matches_reference(self):
        valid = reduce_matrix(self.dm_self, 4, 5, True)
        out = mp.selfjoin_matrix(self.a, self.m, threshold=0.0, mheight=4, mwidth=5, pearson=True)
        np.testing.assert_allclose(valid, out, equal_nan=True, rtol=1e-4, atol=1e-4)

    def test_abjoin_matrix_matches_reference(self):
        valid = reduce_matrix(self.dm_ab, 4, 5, False)
        out = mp.abjoin_matrix(self.a, self.b, self.m, threshold=0.0, mheight=4, mwidth=5, pearson=True)
        np.testing.assert_allclose(valid, out, equal_nan=True, rtol=1e-4, atol=1e-4)

    def test_abjoin_matrix_can_exclude_aligned_trivial_matches(self):
        series, m, dm = self._aligned_ab_fixture()
        matrix_size = dm.shape[0]
        valid = reduce_matrix(dm, matrix_size, matrix_size, False)

        out = mp.abjoin_matrix(
            series,
            series,
            m,
            threshold=-1.0,
            mheight=matrix_size,
            mwidth=matrix_size,
            pearson=True,
            allow_trivial_match=False,
        )

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

    def test_abjoin_knn_can_exclude_aligned_trivial_matches(self):
        series, m, dm = self._aligned_ab_fixture()

        matches = mp.abjoin_knn(
            series,
            series,
            m,
            3,
            threshold=0.2,
            pearson=True,
            allow_trivial_match=False,
        )

        self._assert_knn_equals_reference(matches, dm, 3, 0.2)

    def test_allow_trivial_match_defaults_to_true_for_abjoins(self):
        series = self.a[:64]
        m = 16

        default_dist, default_idx = mp.abjoin(series, series, m, pearson=True)
        explicit_dist, explicit_idx = mp.abjoin(
            series,
            series,
            m,
            pearson=True,
            allow_trivial_match=True,
        )
        np.testing.assert_allclose(default_dist, explicit_dist, equal_nan=True)
        np.testing.assert_array_equal(default_idx, explicit_idx)
        np.testing.assert_allclose(default_dist, np.ones_like(default_dist), rtol=1e-4, atol=1e-4)
        np.testing.assert_array_equal(default_idx, np.arange(default_idx.size, dtype=np.int32))

        default_sum = mp.abjoin_sum(series, series, m, threshold=0.2)
        explicit_sum = mp.abjoin_sum(
            series,
            series,
            m,
            threshold=0.2,
            allow_trivial_match=True,
        )
        np.testing.assert_allclose(default_sum, explicit_sum)

        matrix_kwargs = {"threshold": 0.0, "mheight": 4, "mwidth": 5, "pearson": True}
        default_matrix = mp.abjoin_matrix(series, series, m, **matrix_kwargs)
        explicit_matrix = mp.abjoin_matrix(
            series,
            series,
            m,
            allow_trivial_match=True,
            **matrix_kwargs,
        )
        np.testing.assert_allclose(default_matrix, explicit_matrix, equal_nan=True)

        default_knn = mp.abjoin_knn(series, series, m, 3, threshold=0.2, pearson=True)
        explicit_knn = mp.abjoin_knn(
            series,
            series,
            m,
            3,
            threshold=0.2,
            pearson=True,
            allow_trivial_match=True,
        )
        self.assertEqual(default_knn, explicit_knn)

    def test_allow_trivial_match_is_rejected_by_selfjoins(self):
        expected = (
            "allow_trivial_match is only valid for ab-joins; "
            "self-joins always exclude trivial matches."
        )
        calls = [
            ("selfjoin", lambda value: mp.selfjoin(self.a, self.m, allow_trivial_match=value)),
            (
                "selfjoin_sum",
                lambda value: mp.selfjoin_sum(self.a, self.m, allow_trivial_match=value),
            ),
            (
                "selfjoin_matrix",
                lambda value: mp.selfjoin_matrix(self.a, self.m, allow_trivial_match=value),
            ),
            (
                "selfjoin_knn",
                lambda value: mp.selfjoin_knn(self.a, self.m, 3, allow_trivial_match=value),
            ),
        ]

        for name, call in calls:
            for value in (True, False):
                with self.subTest(profile=name, allow_trivial_match=value):
                    with self.assertRaises(ValueError) as raised:
                        call(value)
                    self.assertEqual(expected, str(raised.exception))

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


if __name__ == "__main__":
    unittest.main()
