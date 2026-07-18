import unittest
from unittest.mock import patch

import mlx.core as mx
import numpy as np

import mlx_native_scamp.core as scamp_core
import pyscamp as mp
from mlx_native_scamp import core

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

    def _assert_join_results_equal(self, expected, actual):
        if isinstance(expected, tuple):
            np.testing.assert_allclose(
                expected[0], actual[0], equal_nan=True, rtol=1e-5, atol=1e-5
            )
            np.testing.assert_array_equal(expected[1], actual[1])
        else:
            np.testing.assert_allclose(
                np.asarray(expected),
                np.asarray(actual),
                equal_nan=True,
                rtol=1e-5,
                atol=1e-5,
            )

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

    def test_max_tile_size_preserves_all_join_outputs(self):
        rng = np.random.default_rng(7)
        a = rng.random(1200, dtype=np.float32)
        b = rng.random(1175, dtype=np.float32)
        m = 32
        tiled = {"max_tile_size": np.int64(1024)}
        calls = {
            "selfjoin": lambda options: mp.selfjoin(a, m, pearson=True, **options),
            "abjoin": lambda options: mp.abjoin(a, b, m, pearson=True, **options),
            "selfjoin_sum": lambda options: mp.selfjoin_sum(a, m, threshold=0.2, **options),
            "abjoin_sum": lambda options: mp.abjoin_sum(a, b, m, threshold=0.2, **options),
            "selfjoin_matrix": lambda options: mp.selfjoin_matrix(
                a, m, threshold=0.1, mheight=5, mwidth=6, pearson=True, **options
            ),
            "abjoin_matrix": lambda options: mp.abjoin_matrix(
                a, b, m, threshold=0.1, mheight=5, mwidth=6, pearson=True, **options
            ),
            "selfjoin_knn": lambda options: mp.selfjoin_knn(
                a, m, 3, threshold=0.2, pearson=True, **options
            ),
            "abjoin_knn": lambda options: mp.abjoin_knn(
                a, b, m, 3, threshold=0.2, pearson=True, **options
            ),
        }

        for name, call in calls.items():
            with self.subTest(name=name):
                self._assert_join_results_equal(call({}), call(tiled))

    def test_max_tile_size_bounds_both_similarity_dimensions(self):
        rng = np.random.default_rng(8)
        m = 32
        max_tile_size = 1024
        tile_subsequences = max_tile_size - m + 1
        prepared_a = core._prepare_series(
            core._ensure_1d_array(rng.random(1400, dtype=np.float32), "a"), m
        )
        prepared_b = core._prepare_series(
            core._ensure_1d_array(rng.random(1300, dtype=np.float32), "b"), m
        )

        tiles = list(
            core._iterate_blocks(
                prepared_a,
                prepared_b,
                m,
                False,
                block_rows=2048,
                tile_subsequences=tile_subsequences,
            )
        )

        self.assertGreater(len(tiles), 1)
        self.assertEqual(993, tile_subsequences)
        self.assertTrue(
            all(tile.values.shape[0] <= tile_subsequences for tile in tiles)
        )
        self.assertTrue(
            all(tile.values.shape[1] <= tile_subsequences for tile in tiles)
        )

    def test_automatic_tile_size_uses_unified_memory_and_remains_bounded(self):
        rng = np.random.default_rng(9)
        m = 32
        prepared_a = core._prepare_series(
            core._ensure_1d_array(rng.random(2600, dtype=np.float32), "a"), m
        )
        prepared_b = core._prepare_series(
            core._ensure_1d_array(rng.random(2500, dtype=np.float32), "b"), m
        )

        with patch.object(core, "_device_working_set_bytes", return_value=64 * core.MIB):
            low_memory_tile = core._automatic_max_tile_size(
                prepared_a, prepared_b, m
            )
        with patch.object(
            core, "_device_working_set_bytes", return_value=8 * 1024 * core.MIB
        ):
            high_memory_tile = core._automatic_max_tile_size(
                prepared_a, prepared_b, m
            )

        tile_subsequences = core._tile_subsequence_count(low_memory_tile, m)
        self.assertGreater(high_memory_tile, low_memory_tile)
        self.assertGreaterEqual(low_memory_tile, max(1024, 2 * m))
        self.assertLess(tile_subsequences, prepared_a.subsequences)

        tiles = list(
            core._iterate_blocks(
                prepared_a,
                prepared_b,
                m,
                False,
                block_rows=2048,
                tile_subsequences=tile_subsequences,
            )
        )
        self.assertGreater(len(tiles), 1)
        self.assertTrue(
            all(max(tile.values.shape) <= tile_subsequences for tile in tiles)
        )

    def test_omitted_max_tile_size_uses_automatic_policy(self):
        with patch.object(
            core,
            "_automatic_max_tile_size",
            wraps=core._automatic_max_tile_size,
        ) as automatic:
            mp.abjoin(self.a, self.b, self.m, pearson=True)

        automatic.assert_called_once()

    def test_reducer_scheduler_bounds_in_flight_tiles(self):
        with (
            patch.object(core, "_schedule_reducer_state") as schedule,
            patch.object(core.mx, "synchronize") as synchronize,
        ):
            scheduler = core.ReducerScheduler()
            scheduler.schedule("first")
            synchronize.assert_not_called()
            scheduler.schedule("second")
            synchronize.assert_called_once_with()
            scheduler.schedule("third")
            scheduler.finish()

        self.assertEqual(3, schedule.call_count)
        self.assertEqual(2, synchronize.call_count)

    def test_max_tile_size_validation(self):
        with self.assertRaisesRegex(ValueError, "integer"):
            mp.selfjoin(self.a, self.m, max_tile_size=1024.0)
        with self.assertRaisesRegex(ValueError, "integer"):
            mp.selfjoin(self.a, self.m, max_tile_size=None)
        with self.assertRaisesRegex(ValueError, "at least 1024"):
            mp.selfjoin(self.a, self.m, max_tile_size=1023)

        series = np.arange(1200, dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "at least twice m"):
            mp.selfjoin(series, 513, max_tile_size=1024)

        boundary_profile, _ = mp.selfjoin(series[:512], 512, max_tile_size=1024)
        self.assertEqual((1,), boundary_profile.shape)

    def test_invalid_kwargs_raise(self):
        with self.assertRaises(ValueError):
            mp.selfjoin(self.a, self.m, nope=True)


if __name__ == "__main__":
    unittest.main()
