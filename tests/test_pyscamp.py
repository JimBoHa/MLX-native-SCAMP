import unittest
from importlib.metadata import PackageNotFoundError, version as distribution_version
from unittest.mock import patch

import mlx.core as mx
import numpy as np

import mlx_native_scamp
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

    def _assert_knn_equals_reference(self, matches, dm, k, threshold):
        expected = []
        for col in range(dm.shape[1]):
            order = np.argsort(dm[:, col])[::-1][:k]
            for row in order:
                corr = float(dm[row, col])
                if corr > threshold and corr >= -1.0:
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
        a = rng.standard_normal(96).astype(np.float32)
        b = rng.standard_normal(96).astype(np.float32)
        m = 33
        floor_exclusion = m // 4
        source_start = 20
        target_start = source_start + floor_exclusion
        b[target_start : target_start + m] = a[
            source_start : source_start + m
        ]

        unmasked = distance_matrix(a, b, m)
        self.assertEqual(target_start, np.argmax(unmasked[:, source_start]))
        exclusion = (m + 3) // 4
        rows = np.arange(unmasked.shape[0])[:, None]
        cols = np.arange(unmasked.shape[1])[None, :]
        masked = unmasked.copy()
        masked[np.abs(rows - cols) < exclusion] = -2.0
        self.assertNotEqual(target_start, np.argmax(masked[:, source_start]))
        return a, b, m, masked

    def test_public_surface_matches_upstream_bindings(self):
        exported = {name for name in EXPECTED_PUBLIC_CALLABLES if hasattr(mp, name)}
        try:
            installed_version = distribution_version("mlx-native-scamp")
        except PackageNotFoundError:
            installed_version = "dev"
        self.assertEqual(EXPECTED_PUBLIC_CALLABLES, exported)
        self.assertEqual(installed_version, mlx_native_scamp.__version__)
        self.assertEqual(mlx_native_scamp.__version__, mp.__version__)

    def test_source_tree_version_falls_back_to_dev(self):
        missing_distribution = PackageNotFoundError("mlx-native-scamp")
        with patch.object(
            mlx_native_scamp,
            "_distribution_version",
            side_effect=missing_distribution,
        ):
            self.assertEqual("dev", mlx_native_scamp._resolve_version())

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

    def test_aligned_ab_exclusion_matches_all_portable_profiles(self):
        a, b, m, matrix = self._aligned_ab_fixture()
        expected_corr, expected_index = reduce_1nn_index(matrix)
        actual_corr, actual_index = mp.abjoin(
            a,
            b,
            m,
            pearson=True,
            allow_trivial_match=False,
            gpus=[],
        )
        np.testing.assert_allclose(
            actual_corr,
            expected_corr,
            equal_nan=True,
            rtol=1e-4,
            atol=1e-4,
        )
        np.testing.assert_array_equal(actual_index, expected_index)

        threshold = 0.2
        expected_sum = reduce_sum_thresh(matrix, threshold)
        actual_sum = mp.abjoin_sum(
            a,
            b,
            m,
            threshold=threshold,
            allow_trivial_match=False,
            gpus=[],
        )
        np.testing.assert_allclose(
            actual_sum, expected_sum, rtol=1e-4, atol=1e-4
        )

        size = matrix.shape[0]
        expected_matrix = reduce_matrix(matrix, size, size, False)
        actual_matrix = mp.abjoin_matrix(
            a,
            b,
            m,
            threshold=-1.0,
            mheight=size,
            mwidth=size,
            pearson=True,
            allow_trivial_match=False,
            gpus=[],
        )
        np.testing.assert_allclose(
            actual_matrix,
            expected_matrix,
            equal_nan=True,
            rtol=1e-4,
            atol=1e-4,
        )

        matches = mp.abjoin_knn(
            a,
            b,
            m,
            3,
            threshold=threshold,
            pearson=True,
            allow_trivial_match=False,
            gpus=[],
        )
        self._assert_knn_equals_reference(matches, matrix, 3, threshold)

    def test_allow_trivial_match_defaults_true_for_all_ab_profiles(self):
        options = {"pearson": True, "gpus": []}
        default_corr, default_index = mp.abjoin(
            self.a, self.b, self.m, **options
        )
        explicit_corr, explicit_index = mp.abjoin(
            self.a,
            self.b,
            self.m,
            allow_trivial_match=True,
            **options,
        )
        np.testing.assert_allclose(default_corr, explicit_corr, equal_nan=True)
        np.testing.assert_array_equal(default_index, explicit_index)

        default_sum = mp.abjoin_sum(
            self.a, self.b, self.m, threshold=0.2, gpus=[]
        )
        explicit_sum = mp.abjoin_sum(
            self.a,
            self.b,
            self.m,
            threshold=0.2,
            allow_trivial_match=True,
            gpus=[],
        )
        np.testing.assert_allclose(default_sum, explicit_sum)

        matrix_options = {
            "threshold": 0.0,
            "mheight": 4,
            "mwidth": 5,
            "pearson": True,
            "gpus": [],
        }
        default_matrix = mp.abjoin_matrix(
            self.a, self.b, self.m, **matrix_options
        )
        explicit_matrix = mp.abjoin_matrix(
            self.a,
            self.b,
            self.m,
            allow_trivial_match=True,
            **matrix_options,
        )
        np.testing.assert_allclose(
            default_matrix, explicit_matrix, equal_nan=True
        )

        default_knn = mp.abjoin_knn(
            self.a,
            self.b,
            self.m,
            3,
            threshold=0.2,
            pearson=True,
            gpus=[],
        )
        explicit_knn = mp.abjoin_knn(
            self.a,
            self.b,
            self.m,
            3,
            threshold=0.2,
            pearson=True,
            allow_trivial_match=True,
            gpus=[],
        )
        self.assertEqual(default_knn, explicit_knn)

    def test_allow_trivial_match_is_rejected_by_selfjoins(self):
        calls = (
            lambda: mp.selfjoin(
                self.a, self.m, allow_trivial_match=False
            ),
            lambda: mp.selfjoin_sum(
                self.a, self.m, allow_trivial_match=False
            ),
            lambda: mp.selfjoin_matrix(
                self.a, self.m, allow_trivial_match=False
            ),
            lambda: mp.selfjoin_knn(
                self.a, self.m, 3, allow_trivial_match=False
            ),
        )
        for call in calls:
            with self.subTest(call=call), self.assertRaisesRegex(
                ValueError, "only valid for ab-joins"
            ):
                call()

    def test_portable_reducers_clamp_perfect_correlations(self):
        series_by_precision = {
            "single": np.array(
                [-0.5038742, -1.1873481, -0.28324285],
                dtype=np.float32,
            ),
            "double": np.array(
                [-0.7908847696275746, 0.2369731299165827, 0.05437949611686499],
                dtype=np.float64,
            ),
        }

        for precision, series in series_by_precision.items():
            window = len(series)
            expected_distance = {
                1.0: np.array([0.0], dtype=np.float32),
                -1.0: np.array(
                    [2.0 * np.sqrt(window)],
                    dtype=np.float32,
                ),
            }
            options = {"gpus": [], "precision": precision}
            for correlation, other in ((1.0, series), (-1.0, -series)):
                with self.subTest(precision=precision, correlation=correlation):
                    out_corr, out_idx = mp.abjoin(
                        series,
                        other,
                        window,
                        pearson=True,
                        **options,
                    )
                    out_dist, dist_idx = mp.abjoin(
                        series,
                        other,
                        window,
                        pearson=False,
                        **options,
                    )
                    summed = mp.abjoin_sum(
                        series,
                        other,
                        window,
                        threshold=-1.0 if correlation < 0.0 else 0.0,
                        **options,
                    )
                    matrix = mp.abjoin_matrix(
                        series,
                        other,
                        window,
                        threshold=-1.0 if correlation < 0.0 else 0.0,
                        mheight=1,
                        mwidth=1,
                        pearson=True,
                        **options,
                    )
                    matches = mp.abjoin_knn(
                        series,
                        other,
                        window,
                        1,
                        threshold=-1.0 if correlation < 0.0 else 0.0,
                        pearson=True,
                        **options,
                    )

                    np.testing.assert_array_equal(
                        out_corr,
                        np.array([correlation], dtype=np.float32),
                    )
                    np.testing.assert_array_equal(
                        out_idx,
                        np.array([0], dtype=np.int32),
                    )
                    np.testing.assert_array_equal(
                        out_dist,
                        expected_distance[correlation],
                    )
                    np.testing.assert_array_equal(dist_idx, out_idx)
                    np.testing.assert_array_equal(
                        summed,
                        np.array(
                            [1.0 if correlation > 0.0 else 0.0],
                            dtype=np.float64,
                        ),
                    )
                    np.testing.assert_array_equal(
                        matrix,
                        np.array([[correlation]], dtype=np.float32),
                    )
                    self.assertEqual(
                        [(0, 0, 1.0)] if correlation > 0.0 else [],
                        matches,
                    )

    def test_portable_clamping_preserves_exclusion_sentinel(self):
        series = np.arange(8, dtype=np.float32)

        with mx.stream(mx.cpu):
            prepared = scamp_core._prepare_tiled_series(mx.array(series), 4)
            blocks = list(
                scamp_core._iterate_tiled_blocks(
                    prepared,
                    prepared,
                    4,
                    (4 + 3) // 4,
                    tile_rows=prepared.subsequences,
                    tile_columns=prepared.subsequences,
                )
            )

        self.assertEqual(1, len(blocks))
        block = np.asarray(blocks[0].values)
        np.testing.assert_array_equal(
            np.diag(block),
            np.full((prepared.subsequences,), scamp_core.SENTINEL),
        )
        valid = block != scamp_core.SENTINEL
        self.assertTrue(np.all(block[valid] >= -1.0))
        self.assertTrue(np.all(block[valid] <= 1.0))

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

    def test_knn_neighbor_count_must_be_a_positive_integer(self):
        for call in (
            lambda k: mp.selfjoin_knn(self.a, self.m, k),
            lambda k: mp.abjoin_knn(self.a, self.b, self.m, k),
        ):
            with self.subTest(join=call):
                matches = call(np.int64(2))
                self.assertIsInstance(matches, list)
                for invalid in (2.5, "2", None):
                    with (
                        self.subTest(invalid=invalid),
                        patch.object(
                            scamp_core,
                            "_run_profile_with_resources",
                            side_effect=AssertionError("unexpected MLX execution"),
                        ) as run_profile,
                        self.assertRaisesRegex(TypeError, "integer"),
                    ):
                        call(invalid)
                    run_profile.assert_not_called()
                for invalid in (0, -1):
                    with (
                        self.subTest(invalid=invalid),
                        patch.object(
                            scamp_core,
                            "_run_profile_with_resources",
                            side_effect=AssertionError("unexpected MLX execution"),
                        ) as run_profile,
                        self.assertRaisesRegex(ValueError, "greater than 0"),
                    ):
                        call(invalid)
                    run_profile.assert_not_called()

    def test_profiles_schedule_compact_reducer_state_per_block(self):
        rng = np.random.default_rng(7)
        a = rng.normal(size=10).astype(np.float32)
        b = rng.normal(size=14).astype(np.float32)
        m = 4
        block_rows = 3
        columns = len(a) - m + 1
        rows = len(b) - m + 1
        expected_blocks = (rows + block_rows - 1) // block_rows
        profiles = [
            ("1nn", lambda: mp.abjoin(a, b, m, pearson=True), ((columns,), (columns,))),
            ("sum", lambda: mp.abjoin_sum(a, b, m, threshold=0.2), ((columns,),)),
            (
                "matrix",
                lambda: mp.abjoin_matrix(a, b, m, mheight=2, mwidth=3, pearson=True),
                ((2, 3),),
            ),
            (
                "knn",
                lambda: mp.abjoin_knn(a, b, m, 2, threshold=0.2, pearson=True),
                ((2, columns), (2, columns)),
            ),
        ]

        for name, run_profile, expected_shapes in profiles:
            with self.subTest(profile=name):
                with (
                    patch.object(scamp_core, "BLOCK_ROWS", block_rows),
                    patch.object(
                        scamp_core,
                        "_schedule_reducer_state",
                        wraps=scamp_core._schedule_reducer_state,
                    ) as schedule,
                ):
                    run_profile()

                self.assertEqual(expected_blocks, schedule.call_count)
                for call in schedule.call_args_list:
                    actual_shapes = tuple(tuple(state.shape) for state in call.args)
                    self.assertEqual(expected_shapes, actual_shapes)

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

    def test_numpy_compatible_inputs_are_forcecast_and_contiguous(self):
        inputs = (
            range(16),
            np.arange(16, dtype=np.float64),
            np.ascontiguousarray(np.arange(16, dtype=np.float64)[::-1])[::-1],
            np.array([str(value) for value in range(16)]),
            np.array(range(16), dtype=object),
        )
        for precision in ("single", "double"):
            expected_dist, expected_idx = mp.selfjoin(
                np.arange(16, dtype=np.float64),
                4,
                pearson=True,
                precision=precision,
            )
            for values in inputs:
                with self.subTest(
                    precision=precision,
                    input_type=type(values),
                    dtype=getattr(values, "dtype", None),
                ):
                    out_dist, out_idx = mp.selfjoin(
                        values,
                        4,
                        pearson=True,
                        precision=precision,
                    )
                    np.testing.assert_allclose(
                        expected_dist,
                        out_dist,
                        equal_nan=True,
                        rtol=1e-4,
                        atol=1e-4,
                    )
                    np.testing.assert_array_equal(expected_idx, out_idx)

    def test_numpy_coercion_preserves_selected_precision(self):
        numeric_strings = np.array(["100000000", "100000001", "100000002"])

        with mx.stream(mx.cpu):
            single = scamp_core._ensure_1d_array(
                numeric_strings, "a", mx.float32
            )
            double = scamp_core._ensure_1d_array(
                numeric_strings, "a", mx.float64
            )
            mx.eval(single, double)
            single_value = float(np.asarray(single)[1])
            double_value = float(np.asarray(double)[1])

        self.assertEqual(mx.float32, single.dtype)
        self.assertEqual(mx.float64, double.dtype)
        self.assertEqual(100000000.0, single_value)
        self.assertEqual(100000001.0, double_value)

    def test_mlx_inputs_stay_on_the_device_native_conversion_path(self):
        source = mx.array([0.0, 1.0, 2.0], dtype=mx.float32)

        with patch.object(
            scamp_core.np,
            "asarray",
            side_effect=AssertionError("unexpected host conversion"),
        ):
            with mx.stream(mx.cpu):
                converted = scamp_core._ensure_1d_array(source, "a", mx.float64)
                mx.eval(converted)

        self.assertEqual(mx.float64, converted.dtype)
        np.testing.assert_array_equal(np.asarray(converted), np.arange(3))

    def test_non_1d_numpy_compatible_inputs_are_rejected_clearly(self):
        for values in (42, [[0, 1], [2, 3]], np.zeros((2, 2))):
            with self.subTest(values=values):
                with self.assertRaisesRegex(ValueError, "a must be a 1D array"):
                    mp.selfjoin(values, 3)

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

    def test_common_kwarg_types_match_upstream_for_every_profile(self):
        calls = {
            "selfjoin": lambda **kwargs: mp.selfjoin(self.a, self.m, **kwargs),
            "abjoin": lambda **kwargs: mp.abjoin(self.a, self.b, self.m, **kwargs),
            "selfjoin_sum": lambda **kwargs: mp.selfjoin_sum(self.a, self.m, **kwargs),
            "abjoin_sum": lambda **kwargs: mp.abjoin_sum(self.a, self.b, self.m, **kwargs),
            "selfjoin_matrix": lambda **kwargs: mp.selfjoin_matrix(self.a, self.m, **kwargs),
            "abjoin_matrix": lambda **kwargs: mp.abjoin_matrix(self.a, self.b, self.m, **kwargs),
            "selfjoin_knn": lambda **kwargs: mp.selfjoin_knn(self.a, self.m, 2, **kwargs),
            "abjoin_knn": lambda **kwargs: mp.abjoin_knn(self.a, self.b, self.m, 2, **kwargs),
        }
        invalid_kwargs = {
            "precision": 1,
            "pearson": "false",
            "verbose": [],
            "threads": 1.5,
            "gpus": None,
            "max_tile_size": 1024.0,
        }

        for profile, call in calls.items():
            for keyword, value in invalid_kwargs.items():
                with self.subTest(profile=profile, keyword=keyword):
                    with self.assertRaises(TypeError):
                        call(**{keyword: value})
            with self.subTest(profile=profile, keyword="GPU device ID"):
                with self.assertRaisesRegex(TypeError, "GPU device ID"):
                    call(gpus=[0.0])

    def test_profile_specific_kwarg_types_match_upstream(self):
        threshold_calls = {
            "selfjoin_sum": lambda **kwargs: mp.selfjoin_sum(self.a, self.m, **kwargs),
            "abjoin_sum": lambda **kwargs: mp.abjoin_sum(self.a, self.b, self.m, **kwargs),
            "selfjoin_matrix": lambda **kwargs: mp.selfjoin_matrix(self.a, self.m, **kwargs),
            "abjoin_matrix": lambda **kwargs: mp.abjoin_matrix(self.a, self.b, self.m, **kwargs),
            "selfjoin_knn": lambda **kwargs: mp.selfjoin_knn(self.a, self.m, 2, **kwargs),
            "abjoin_knn": lambda **kwargs: mp.abjoin_knn(self.a, self.b, self.m, 2, **kwargs),
        }
        for profile, call in threshold_calls.items():
            with self.subTest(profile=profile, keyword="threshold"):
                with self.assertRaisesRegex(TypeError, "real number"):
                    call(threshold="0.2")

        matrix_calls = {
            "selfjoin_matrix": lambda **kwargs: mp.selfjoin_matrix(self.a, self.m, **kwargs),
            "abjoin_matrix": lambda **kwargs: mp.abjoin_matrix(self.a, self.b, self.m, **kwargs),
        }
        for profile, call in matrix_calls.items():
            for keyword, value in (("mheight", 2.5), ("mwidth", "2")):
                with self.subTest(profile=profile, keyword=keyword):
                    with self.assertRaisesRegex(TypeError, "integer"):
                        call(**{keyword: value})

        ab_calls = {
            "abjoin": lambda **kwargs: mp.abjoin(
                self.a, self.b, self.m, **kwargs
            ),
            "abjoin_sum": lambda **kwargs: mp.abjoin_sum(
                self.a, self.b, self.m, **kwargs
            ),
            "abjoin_matrix": lambda **kwargs: mp.abjoin_matrix(
                self.a, self.b, self.m, **kwargs
            ),
            "abjoin_knn": lambda **kwargs: mp.abjoin_knn(
                self.a, self.b, self.m, 2, **kwargs
            ),
        }
        for profile, call in ab_calls.items():
            with self.subTest(profile=profile, keyword="allow_trivial_match"):
                with self.assertRaisesRegex(TypeError, "boolean-compatible"):
                    call(allow_trivial_match="false")

    def test_numpy_scalar_compatibility_kwargs_are_accepted(self):
        common = {
            "pearson": np.bool_(True),
            "verbose": None,
            "threads": np.int64(1),
            "gpus": (),
            "precision": "double",
        }
        a = self.a[:64]
        b = self.b[:64]
        m = 16
        calls = (
            lambda: mp.selfjoin(a, m, **common),
            lambda: mp.abjoin(a, b, m, **common),
            lambda: mp.selfjoin_sum(a, m, threshold=np.float32(0.2), **common),
            lambda: mp.abjoin_sum(a, b, m, threshold=np.float32(0.2), **common),
            lambda: mp.selfjoin_matrix(
                a,
                m,
                threshold=np.float32(0.2),
                mheight=np.int64(2),
                mwidth=np.int64(3),
                **common,
            ),
            lambda: mp.abjoin_matrix(
                a,
                b,
                m,
                threshold=np.float32(0.2),
                mheight=np.int64(2),
                mwidth=np.int64(3),
                **common,
            ),
            lambda: mp.selfjoin_knn(a, m, 2, threshold=np.float32(0.2), **common),
            lambda: mp.abjoin_knn(a, b, m, 2, threshold=np.float32(0.2), **common),
        )
        for call in calls:
            with self.subTest(call=call):
                call()

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
        for gpus in ([1], [-1]):
            with self.subTest(gpus=gpus):
                with self.assertRaisesRegex(ValueError, "GPU device ID"):
                    mp.selfjoin(self.a, self.m, gpus=gpus)

        with self.assertRaisesRegex(TypeError, "GPU device ID"):
            mp.selfjoin(self.a, self.m, gpus=["0"])

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
