import unittest
from unittest.mock import patch

import mlx.core as mx
import numpy as np

import mlx_native_scamp.core as scamp_core
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

    def test_gpu_supported_is_independent_of_default_device(self):
        previous_device = mx.default_device()
        try:
            mx.set_default_device(mx.cpu)
            self.assertEqual(mx.metal.is_available(), mp.gpu_supported())
        finally:
            mx.set_default_device(previous_device)

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

    def test_non_multiple_of_four_exclusion_matches_reference_on_cpu(self):
        rng = np.random.default_rng(1)
        series = np.cumsum(rng.normal(size=37)).astype(np.float32)

        for m in (3, 5):
            with self.subTest(m=m):
                dm = distance_matrix(series, None, m)
                expected_corr, expected_index = reduce_1nn_index(dm)
                expected_sum = reduce_sum_thresh(dm, -1.0)

                actual_corr, actual_index = mp.selfjoin(
                    series,
                    m,
                    pearson=True,
                    precision="double",
                    gpus=[],
                )
                actual_sum = mp.selfjoin_sum(
                    series,
                    m,
                    threshold=-1.0,
                    precision="double",
                    gpus=[],
                )

                np.testing.assert_allclose(
                    actual_corr,
                    expected_corr,
                    rtol=1e-5,
                    atol=1e-5,
                    equal_nan=True,
                )
                np.testing.assert_array_equal(actual_index, expected_index)
                np.testing.assert_allclose(
                    actual_sum,
                    expected_sum,
                    rtol=1e-5,
                    atol=1e-5,
                )

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

    def test_abjoin_knn_excludes_matches_equal_to_threshold(self):
        a = np.array([1, 0, -1, 0], dtype=np.float32)
        b = np.array([0, 1, 0, -1], dtype=np.float32)

        matches = mp.abjoin_knn(a, b, 4, 1, threshold=0.0, pearson=True)

        self.assertEqual([], matches)

    def test_nan_windows_are_excluded(self):
        arr = self.a.copy()
        arr[10] = np.nan
        out_dist, out_idx = mp.selfjoin(arr, self.m, pearson=True)
        self.assertTrue(np.isnan(out_dist[:11]).any())
        self.assertIn(-1, out_idx.tolist())

    def test_normalization_is_stable_for_large_finite_values(self):
        series = np.array([1e20, 2e20, 3e20], dtype=np.float32)

        out_corr, out_idx = mp.abjoin(series, series, 3, pearson=True)
        out_dist, dist_idx = mp.abjoin(series, series, 3, pearson=False)

        np.testing.assert_array_equal(out_corr, np.array([1.0], dtype=np.float32))
        np.testing.assert_array_equal(out_idx, np.array([0], dtype=np.int32))
        np.testing.assert_array_equal(out_dist, np.array([0.0], dtype=np.float32))
        np.testing.assert_array_equal(dist_idx, np.array([0], dtype=np.int32))

    def test_scaling_preserves_flat_subsequence_detection(self):
        series = np.array([1e-8, 2e-8, 3e-8], dtype=np.float32)

        out_corr, out_idx = mp.abjoin(series, series, 3, pearson=True)

        self.assertTrue(np.isnan(out_corr[0]))
        np.testing.assert_array_equal(out_idx, np.array([-1], dtype=np.int32))

    def test_scaling_is_independent_for_each_window(self):
        # Upstream SCAMP keeps all three windows valid. A single series-wide
        # scale makes the first two underflow when the final sample is huge.
        series = np.array([1.0, 2.0, 3.0, 4.0, 1e38], dtype=np.float32)

        out_corr, out_idx = mp.abjoin(series, series, 3, pearson=True)

        np.testing.assert_allclose(out_corr, np.ones(3, dtype=np.float32), atol=1e-6)
        np.testing.assert_array_equal(out_idx, np.array([0, 0, 2], dtype=np.int32))

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

    def test_resource_kwargs_select_expected_stream(self):
        cases = [
            (None, 0, None),
            ([], 0, mx.cpu),
            (None, 2, mx.cpu),
            ([0], 0, mx.gpu),
        ]

        for gpus, threads, expected_device in cases:
            with self.subTest(gpus=gpus, threads=threads):
                stream = scamp_core._select_execution_stream(gpus, threads)
                if expected_device is None:
                    self.assertIsNone(stream)
                else:
                    self.assertEqual(expected_device, stream.device)

    def test_resource_kwargs_reject_unsupported_gpu_requests(self):
        for gpus in ([1], [-1], ["0"]):
            with self.subTest(gpus=gpus):
                with self.assertRaisesRegex(ValueError, "GPU device ID"):
                    mp.selfjoin(self.a, self.m, gpus=gpus)

        for gpus in ([0, 1], [0, 0]):
            with self.subTest(gpus=gpus):
                with self.assertRaisesRegex(ValueError, "multi-GPU"):
                    mp.selfjoin(self.a, self.m, gpus=gpus)

        with self.assertRaisesRegex(ValueError, "Concurrent CPU and Metal"):
            mp.selfjoin(self.a, self.m, gpus=[0], threads=2)

    def test_selected_stream_wraps_graph_construction_and_evaluation(self):
        original_device = mx.default_device()
        original_ensure = scamp_core._ensure_1d_array
        original_profile = scamp_core._best_match_profile
        cases = [
            (mx.gpu, {"gpus": []}, mx.cpu),
            (mx.cpu, {"gpus": [0], "precision": "single"}, mx.gpu),
        ]

        for outer_device, kwargs, selected_device in cases:
            observed_devices = []

            def record_ensure(*args, **call_kwargs):
                observed_devices.append(mx.default_device())
                return original_ensure(*args, **call_kwargs)

            def record_profile(*args, **call_kwargs):
                observed_devices.append(mx.default_device())
                result = original_profile(*args, **call_kwargs)
                observed_devices.append(mx.default_device())
                return result

            with self.subTest(kwargs=kwargs):
                with mx.stream(outer_device):
                    with patch.object(
                        scamp_core,
                        "_ensure_1d_array",
                        side_effect=record_ensure,
                    ), patch.object(
                        scamp_core,
                        "_best_match_profile",
                        side_effect=record_profile,
                    ):
                        mp.selfjoin(self.a[:64], 16, pearson=True, **kwargs)
                    self.assertEqual(outer_device, mx.default_device())

                self.assertGreaterEqual(len(observed_devices), 3)
                self.assertTrue(
                    all(device == selected_device for device in observed_devices)
                )

        self.assertEqual(original_device, mx.default_device())

    def test_cpu_resource_kwargs_preserve_default_and_public_output(self):
        valid_dist, valid_idx = reduce_1nn_index(self.dm_self)
        original_device = mx.default_device()

        with mx.stream(mx.gpu):
            for kwargs in ({"gpus": []}, {"threads": 2}):
                with self.subTest(kwargs=kwargs):
                    out_dist, out_idx = mp.selfjoin(
                        self.a,
                        self.m,
                        pearson=True,
                        **kwargs,
                    )
                    self.assertEqual(mx.gpu, mx.default_device())
                    np.testing.assert_allclose(
                        valid_dist,
                        out_dist,
                        equal_nan=True,
                        rtol=1e-4,
                        atol=1e-4,
                    )
                    np.testing.assert_array_equal(valid_idx, out_idx)

        self.assertEqual(original_device, mx.default_device())

    def test_gpu_resource_kwarg_preserves_default_and_public_output(self):
        valid_dist, valid_idx = reduce_1nn_index(self.dm_ab)
        original_device = mx.default_device()

        with mx.stream(mx.cpu):
            out_dist, out_idx = mp.abjoin(
                self.a,
                self.b,
                self.m,
                pearson=True,
                gpus=[0],
                precision="single",
            )
            self.assertEqual(mx.cpu, mx.default_device())
            np.testing.assert_allclose(
                valid_dist,
                out_dist,
                equal_nan=True,
                rtol=1e-4,
                atol=1e-4,
            )
            np.testing.assert_array_equal(valid_idx, out_idx)

        self.assertEqual(original_device, mx.default_device())

    def test_metal_rejects_float64_precision_modes(self):
        for precision in ("double", "ultra"):
            with self.subTest(precision=precision):
                with self.assertRaisesRegex(ValueError, "Metal does not support float64"):
                    mp.selfjoin(
                        self.a,
                        self.m,
                        gpus=[0],
                        precision=precision,
                    )

    def test_double_and_ultra_preserve_high_offset_variation(self):
        series = np.array([1e8, 1e8 + 1, 1e8 + 2, 1e8 + 4], dtype=np.float64)

        for kwargs in ({}, {"precision": "double"}, {"precision": "ultra"}):
            with self.subTest(kwargs=kwargs):
                out_corr, out_idx = mp.abjoin(series, series, 4, pearson=True, **kwargs)
                np.testing.assert_allclose(out_corr, np.array([1.0], dtype=np.float32), atol=1e-6)
                np.testing.assert_array_equal(out_idx, np.array([0], dtype=np.int32))

        single_corr, single_idx = mp.abjoin(
            series,
            series,
            4,
            pearson=True,
            precision="single",
        )
        self.assertTrue(np.isnan(single_corr[0]))
        np.testing.assert_array_equal(single_idx, np.array([-1], dtype=np.int32))

    def test_removed_mixed_precision_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "single, double, ultra"):
            mp.selfjoin(self.a, self.m, precision="mixed")

    def test_invalid_kwargs_raise(self):
        with self.assertRaises(ValueError):
            mp.selfjoin(self.a, self.m, nope=True)

    def test_nan_threshold_is_rejected(self):
        calls = (
            lambda: mp.selfjoin_sum(self.a, self.m, threshold=np.nan),
            lambda: mp.selfjoin_matrix(self.a, self.m, threshold=np.nan),
            lambda: mp.selfjoin_knn(self.a, self.m, 3, threshold=np.nan),
        )
        for call in calls:
            with self.subTest(call=call), self.assertRaisesRegex(ValueError, "finite"):
                call()


if __name__ == "__main__":
    unittest.main()
