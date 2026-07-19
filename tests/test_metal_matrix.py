import unittest
from unittest import mock

import mlx.core as mx
import numpy as np

import mlx_native_scamp as mp
from mlx_native_scamp import _metal_matrix, core


class MetalMatrixIndexingTests(unittest.TestCase):
    def test_signed_diagonal_boundaries(self):
        int32_max = int(np.iinfo(np.int32).max)

        self.assertTrue(
            _metal_matrix.indexing_is_safe(
                int32_max + 1, int32_max + 1, 1, 1, True, 0
            )
        )
        self.assertFalse(
            _metal_matrix.indexing_is_safe(
                int32_max + 2, int32_max + 2, 1, 1, True, 0
            )
        )
        self.assertTrue(
            _metal_matrix.indexing_is_safe(
                int32_max + 1, 1, 1, 1, False, 0
            )
        )
        self.assertFalse(
            _metal_matrix.indexing_is_safe(
                int32_max + 2, 1, 1, 1, False, 0
            )
        )

    def test_output_cell_and_exclusion_boundaries(self):
        int32_max = int(np.iinfo(np.int32).max)

        self.assertTrue(
            _metal_matrix.indexing_is_safe(
                int32_max + 1,
                int32_max + 1,
                65_535,
                65_537,
                True,
                int32_max,
            )
        )
        self.assertFalse(
            _metal_matrix.indexing_is_safe(
                8, 8, 65_536, 65_536, True, 0
            )
        )
        self.assertFalse(
            _metal_matrix.indexing_is_safe(
                int32_max + 1,
                int32_max + 1,
                1,
                1,
                True,
                int32_max + 1,
            )
        )


@unittest.skipUnless(mx.metal.is_available(), "Metal is unavailable")
class MetalMatrixSummaryTests(unittest.TestCase):
    def setUp(self):
        self.previous_device = mx.default_device()
        mx.set_default_device(mx.gpu)

    def tearDown(self):
        mx.set_default_device(self.previous_device)

    def test_selfjoin_matches_portable_non_square_summary(self):
        rng = np.random.default_rng(515)
        series = rng.normal(size=137).astype(np.float32)
        expected = mp.selfjoin_matrix(
            series,
            8,
            mheight=11,
            mwidth=13,
            threshold=-0.4,
            pearson=True,
            precision="single",
            gpus=[],
        )

        with (
            mock.patch.object(
                core,
                "_prepare_series",
                side_effect=AssertionError(
                    "normalized windows were materialized"
                ),
            ),
            mock.patch.object(
                _metal_matrix,
                "matrix_summary",
                wraps=_metal_matrix.matrix_summary,
            ) as kernel,
        ):
            actual = mp.selfjoin_matrix(
                series,
                8,
                mheight=11,
                mwidth=13,
                threshold=-0.4,
                pearson=True,
                precision="single",
                gpus=[0],
            )

        kernel.assert_called_once()
        self.assertTrue(kernel.call_args.args[5])
        self.assertEqual((8 + 3) // 4, kernel.call_args.args[6])
        np.testing.assert_allclose(
            actual, expected, rtol=2e-5, atol=2e-5, equal_nan=True
        )
        finite = actual[np.isfinite(actual)]
        self.assertTrue(np.all(finite >= -1.0))
        self.assertTrue(np.all(finite <= 1.0))

    def test_selfjoin_full_summary_preserves_upper_triangle(self):
        rng = np.random.default_rng(73)
        series = rng.normal(size=29).astype(np.float32)
        m = 7
        subsequences = series.size - m + 1
        options = {
            "mheight": subsequences,
            "mwidth": subsequences,
            "threshold": -1.0,
            "pearson": True,
            "precision": "single",
        }
        expected = mp.selfjoin_matrix(series, m, gpus=[], **options)
        actual = mp.selfjoin_matrix(series, m, gpus=[0], **options)

        np.testing.assert_allclose(
            actual, expected, rtol=2e-5, atol=2e-5, equal_nan=True
        )
        lower_triangle = np.tril(
            np.ones(actual.shape, dtype=bool),
            k=0,
        )
        self.assertTrue(np.all(np.isnan(actual[lower_triangle])))

    def test_abjoin_matches_portable_across_rectangular_bin_edges(self):
        rng = np.random.default_rng(919)
        a = rng.normal(size=29).astype(np.float32)
        b = rng.normal(size=37).astype(np.float32)
        a[14] = np.nan
        b[23] = np.inf
        options = {
            "mheight": 7,
            "mwidth": 6,
            "threshold": 0.1,
            "pearson": False,
            "precision": "single",
        }
        expected = mp.abjoin_matrix(a, b, 5, gpus=[], **options)
        actual = mp.abjoin_matrix(a, b, 5, gpus=[0], **options)

        np.testing.assert_allclose(
            actual, expected, rtol=2e-5, atol=2e-5, equal_nan=True
        )

    def test_integer_bin_edges_match_portable_ceil_definition(self):
        for subsequences, bins in ((23, 7), (32, 32), (130, 11)):
            with self.subTest(subsequences=subsequences, bins=bins):
                expected = np.ceil(
                    np.arange(bins + 1) * subsequences / bins
                ).astype(np.int64)
                np.testing.assert_array_equal(
                    core._matrix_bin_edges(subsequences, bins),
                    expected,
                )

    def test_all_invalid_bin_stays_nan(self):
        a = np.array([1.0, np.nan, 2.0, 3.0], dtype=np.float32)
        b = np.array([3.0, 1.0, 4.0, 2.0], dtype=np.float32)

        with mock.patch.object(
            _metal_matrix,
            "matrix_summary",
            wraps=_metal_matrix.matrix_summary,
        ) as kernel:
            actual = mp.abjoin_matrix(
                a,
                b,
                4,
                mheight=1,
                mwidth=1,
                threshold=-1.0,
                pearson=True,
                precision="single",
                gpus=[0],
            )

        kernel.assert_called_once()
        self.assertTrue(np.isnan(actual[0, 0]))

    def test_aligned_abjoin_exclusion_matches_portable(self):
        rng = np.random.default_rng(2112)
        a = rng.normal(size=29).astype(np.float32)
        b = a.copy()
        m = 7
        subsequences = a.size - m + 1
        exclusion = (m + 3) // 4
        options = {
            "mheight": subsequences,
            "mwidth": subsequences,
            "threshold": -1.0,
            "pearson": True,
            "precision": "single",
            "allow_trivial_match": False,
        }
        expected = mp.abjoin_matrix(a, b, m, gpus=[], **options)

        with mock.patch.object(
            _metal_matrix,
            "matrix_summary",
            wraps=_metal_matrix.matrix_summary,
        ) as kernel:
            actual = mp.abjoin_matrix(a, b, m, gpus=[0], **options)

        kernel.assert_called_once()
        self.assertFalse(kernel.call_args.args[5])
        self.assertEqual(exclusion, kernel.call_args.args[6])
        np.testing.assert_allclose(
            actual, expected, rtol=2e-5, atol=2e-5, equal_nan=True
        )
        positions = np.arange(subsequences)
        excluded = np.abs(positions[:, None] - positions[None, :]) < exclusion
        self.assertTrue(np.all(np.isnan(actual[excluded])))

    def test_threshold_equality_and_euclidean_conversion(self):
        cases = (
            (
                1.0,
                np.array([-1.0, 1.0, -1.0, 1.0], dtype=np.float32),
                np.array([-1.0, 1.0, -1.0, 1.0], dtype=np.float32),
            ),
            (
                0.0,
                np.array([1.0, 1.0, -1.0, -1.0], dtype=np.float32),
                np.array([1.0, -1.0, 1.0, -1.0], dtype=np.float32),
            ),
            (
                -1.0,
                np.array([-1.0, 1.0, -1.0, 1.0], dtype=np.float32),
                np.array([1.0, -1.0, 1.0, -1.0], dtype=np.float32),
            ),
        )

        for correlation, a, b in cases:
            with self.subTest(correlation=correlation):
                pearson = mp.abjoin_matrix(
                    a,
                    b,
                    4,
                    mheight=1,
                    mwidth=1,
                    threshold=correlation,
                    pearson=True,
                    precision="single",
                    gpus=[0],
                )
                euclidean = mp.abjoin_matrix(
                    a,
                    b,
                    4,
                    mheight=1,
                    mwidth=1,
                    threshold=correlation,
                    pearson=False,
                    precision="single",
                    gpus=[0],
                )

                self.assertAlmostEqual(correlation, float(pearson[0, 0]))
                self.assertAlmostEqual(
                    np.sqrt(8.0 * (1.0 - correlation)),
                    float(euclidean[0, 0]),
                )

    def test_recurrence_storage_remains_linear(self):
        rng = np.random.default_rng(1028)
        series = rng.normal(size=10_000).astype(np.float32)
        m = 1024
        captured = {}

        def inspect_preparation(
            prepared_a,
            prepared_b,
            _m,
            rows,
            cols,
            self_join,
            exclusion,
            _row_edges,
            _col_edges,
        ):
            self.assertIs(prepared_a, prepared_b)
            self.assertTrue(self_join)
            self.assertEqual((m + 3) // 4, exclusion)
            self.assertIsNone(prepared_a.windows)
            self.assertIsNone(prepared_a.valid)
            arrays = (
                prepared_a.recurrence_clean,
                prepared_a.recurrence_means,
                prepared_a.recurrence_inv_norm,
                prepared_a.recurrence_df,
                prepared_a.recurrence_dg,
            )
            captured["elements"] = sum(int(array.size) for array in arrays)
            return np.full((rows, cols), -2.0, dtype=np.float32)

        with (
            mock.patch.object(
                core,
                "_prepare_series",
                side_effect=AssertionError(
                    "normalized windows were materialized"
                ),
            ),
            mock.patch.object(
                _metal_matrix,
                "matrix_summary",
                side_effect=inspect_preparation,
            ),
        ):
            summary = mp.selfjoin_matrix(
                series,
                m,
                mheight=3,
                mwidth=4,
                threshold=-1.0,
                pearson=True,
                precision="single",
                gpus=[0],
            )

        self.assertTrue(np.all(np.isnan(summary)))
        self.assertLessEqual(captured["elements"], 5 * series.size)

    def test_cpu_and_higher_precision_keep_portable_path(self):
        series = np.random.default_rng(28).normal(size=64).astype(np.float32)

        with mock.patch.object(_metal_matrix, "matrix_summary") as kernel:
            mp.selfjoin_matrix(
                series,
                8,
                mheight=5,
                mwidth=5,
                pearson=True,
                precision="single",
                gpus=[],
            )
            for precision in ("double", "ultra"):
                mp.selfjoin_matrix(
                    series,
                    8,
                    mheight=5,
                    mwidth=5,
                    pearson=True,
                    precision=precision,
                )

        kernel.assert_not_called()

    def test_unsafe_kernel_indexing_keeps_portable_path(self):
        series = np.random.default_rng(283).normal(size=64).astype(np.float32)

        with (
            mock.patch.object(
                _metal_matrix, "indexing_is_safe", return_value=False
            ) as indexing_is_safe,
            mock.patch.object(
                core,
                "_prepare_metal_recurrence",
                side_effect=AssertionError(
                    "unsafe matrix indexing reached Metal preparation"
                ),
            ) as metal_preparation,
            mock.patch.object(_metal_matrix, "matrix_summary") as kernel,
        ):
            summary = mp.selfjoin_matrix(
                series,
                8,
                mheight=5,
                mwidth=7,
                threshold=-1.0,
                pearson=True,
                precision="single",
                gpus=[0],
            )

        indexing_is_safe.assert_called_once_with(57, 57, 5, 7, True, 2)
        metal_preparation.assert_not_called()
        kernel.assert_not_called()
        self.assertEqual((5, 7), summary.shape)

    def test_explicit_tile_ceiling_keeps_oversized_abjoin_portable(self):
        rng = np.random.default_rng(281)
        a = rng.normal(size=1000).astype(np.float32)
        b = rng.normal(size=1025).astype(np.float32)

        with (
            mock.patch.object(
                _metal_matrix,
                "matrix_summary",
                side_effect=AssertionError(
                    "join-wide Metal bypassed the explicit ceiling"
                ),
            ) as kernel,
            mock.patch.object(
                core,
                "_prepare_series_tile",
                wraps=core._prepare_series_tile,
            ) as tiled_preparation,
        ):
            summary = mp.abjoin_matrix(
                a,
                b,
                8,
                mheight=5,
                mwidth=7,
                pearson=True,
                precision="single",
                gpus=[0],
                max_tile_size=1024,
            )

        kernel.assert_not_called()
        self.assertGreater(tiled_preparation.call_count, 0)
        self.assertEqual((5, 7), summary.shape)

    def test_default_tile_ceiling_keeps_oversized_selfjoin_portable(self):
        series = np.random.default_rng(282).normal(size=1025).astype(np.float32)

        with (
            mock.patch.object(
                core,
                "_default_max_tile_size",
                return_value=1024,
            ) as default_ceiling,
            mock.patch.object(
                _metal_matrix,
                "matrix_summary",
                side_effect=AssertionError(
                    "join-wide Metal bypassed the default ceiling"
                ),
            ) as kernel,
            mock.patch.object(
                core,
                "_prepare_series_tile",
                wraps=core._prepare_series_tile,
            ) as tiled_preparation,
        ):
            summary = mp.selfjoin_matrix(
                series,
                8,
                mheight=5,
                mwidth=7,
                pearson=True,
                precision="single",
                gpus=[0],
            )

        default_ceiling.assert_called_once_with()
        kernel.assert_not_called()
        self.assertGreater(tiled_preparation.call_count, 0)
        self.assertEqual((5, 7), summary.shape)

    def test_high_offset_non_float32_input_keeps_portable_path(self):
        offsets = np.array(
            [0, 16, 32, 64, 128, 80, 48, 144, 96, 176, 112, 208],
            dtype=np.float64,
        )
        series = 1e8 + offsets
        options = {
            "mheight": 3,
            "mwidth": 3,
            "pearson": True,
            "precision": "single",
        }
        expected = mp.selfjoin_matrix(series, 4, gpus=[], **options)

        with mock.patch.object(_metal_matrix, "matrix_summary") as kernel:
            actual = mp.selfjoin_matrix(series, 4, gpus=[0], **options)

        kernel.assert_not_called()
        np.testing.assert_allclose(actual, expected, equal_nan=True)

    def test_extreme_float32_magnitudes_fall_back_safely(self):
        a = np.array(
            [1e20, -1e20, 5e19, -5e19, 8e19, -8e19, 3e19, -3e19],
            dtype=np.float32,
        )
        b = a[::-1].copy()
        options = {
            "mheight": 2,
            "mwidth": 2,
            "threshold": -1.0,
            "pearson": True,
            "precision": "single",
        }
        expected = mp.abjoin_matrix(a, b, 4, gpus=[], **options)

        with (
            mock.patch.object(_metal_matrix, "matrix_summary") as kernel,
            mock.patch.object(
                core, "_prepare_series", wraps=core._prepare_series
            ) as portable_preparation,
        ):
            actual = mp.abjoin_matrix(a, b, 4, gpus=[0], **options)

        kernel.assert_not_called()
        self.assertEqual(2, portable_preparation.call_count)
        np.testing.assert_allclose(actual, expected, equal_nan=True)


if __name__ == "__main__":
    unittest.main()
