import unittest
from unittest import mock

import mlx.core as mx
import numpy as np
import pyscamp

import mlx_native_scamp as mp
from mlx_native_scamp import _metal_1nn, core


class IndexFree1NNTests(unittest.TestCase):
    def setUp(self):
        self.previous_device = mx.default_device()

    def tearDown(self):
        mx.set_default_device(self.previous_device)

    def test_native_namespace_does_not_expand_strict_pyscamp_surface(self):
        self.assertTrue(callable(mp.selfjoin_1nn))
        self.assertTrue(callable(mp.abjoin_1nn))
        self.assertFalse(hasattr(pyscamp, "selfjoin_1nn"))
        self.assertFalse(hasattr(pyscamp, "abjoin_1nn"))

    def test_cpu_profiles_match_indexed_values_without_index_reduction(self):
        rng = np.random.default_rng(44)
        a = rng.normal(size=91).astype(np.float32)
        b = rng.normal(size=83).astype(np.float32)

        expected_self = mp.selfjoin(a, 9, pearson=True, gpus=[])[0]
        expected_ab = mp.abjoin(a, b, 9, pearson=False, gpus=[])[0]

        with (
            mock.patch.object(
                core,
                "_iterate_tiled_blocks",
                wraps=core._iterate_tiled_blocks,
            ) as tiled_blocks,
            mock.patch.object(
                core.mx,
                "argmax",
                side_effect=AssertionError("value-only reducer requested indices"),
            ) as argmax,
        ):
            actual_self = mp.selfjoin_1nn(a, 9, pearson=True, gpus=[])
            actual_ab = mp.abjoin_1nn(a, b, 9, pearson=False, gpus=[])

        self.assertEqual(2, tiled_blocks.call_count)
        argmax.assert_not_called()
        np.testing.assert_array_equal(actual_self, expected_self)
        np.testing.assert_array_equal(actual_ab, expected_ab)

    def test_aligned_ab_exclusion_matches_indexed_profile(self):
        rng = np.random.default_rng(17)
        a = rng.standard_normal(96).astype(np.float32)
        b = rng.standard_normal(96).astype(np.float32)
        m = 33
        exclusion = (m + 3) // 4
        source_start = 20
        target_start = source_start + exclusion - 1
        b[target_start : target_start + m] = a[
            source_start : source_start + m
        ]
        options = {
            "pearson": True,
            "allow_trivial_match": False,
            "gpus": [],
        }
        expected = mp.abjoin(a, b, m, **options)[0]

        with mock.patch.object(
            core,
            "_iterate_tiled_blocks",
            wraps=core._iterate_tiled_blocks,
        ) as tiled_blocks:
            actual = mp.abjoin_1nn(a, b, m, **options)

        self.assertEqual(exclusion, tiled_blocks.call_args.args[3])
        np.testing.assert_array_equal(actual, expected)

    @unittest.skipUnless(mx.metal.is_available(), "Metal is unavailable")
    def test_restrictive_tiles_bound_value_only_fallback_without_argmax(self):
        rng = np.random.default_rng(2601)
        series = rng.normal(size=1025).astype(np.float32)
        m = 3
        max_tile_size = 1024
        subsequences = series.size - m + 1
        max_tile_subsequences = max_tile_size - m + 1
        expected = mp.selfjoin(
            series,
            m,
            pearson=True,
            precision="single",
            gpus=[0],
            max_tile_size=max_tile_size,
        )[0]

        for explicit_limit in (True, False):
            kwargs = {"max_tile_size": max_tile_size} if explicit_limit else {}
            with (
                self.subTest(explicit_limit=explicit_limit),
                mock.patch.object(
                    core,
                    "_default_max_tile_size",
                    return_value=max_tile_size,
                ),
                mock.patch.object(
                    core,
                    "_iterate_tiled_blocks",
                    wraps=core._iterate_tiled_blocks,
                ) as tiled_blocks,
                mock.patch.object(
                    core.mx,
                    "argmax",
                    side_effect=AssertionError(
                        "value-only reducer requested indices"
                    ),
                ) as argmax,
                mock.patch.object(_metal_1nn, "best_profile") as metal_profile,
                mock.patch.object(_metal_1nn, "_INDEX_KERNEL") as index_kernel,
            ):
                actual = mp.selfjoin_1nn(
                    series,
                    m,
                    pearson=True,
                    precision="single",
                    gpus=[0],
                    **kwargs,
                )

            tiled_blocks.assert_called_once()
            tile_rows = tiled_blocks.call_args.args[4]
            tile_columns = tiled_blocks.call_args.args[5]
            self.assertLess(tile_rows, subsequences)
            self.assertLess(tile_columns, subsequences)
            self.assertLessEqual(tile_rows, max_tile_subsequences)
            self.assertLessEqual(tile_columns, max_tile_subsequences)
            argmax.assert_not_called()
            metal_profile.assert_not_called()
            index_kernel.assert_not_called()
            np.testing.assert_array_equal(actual, expected)

    def test_value_api_reuses_validation_and_resource_contract(self):
        series = np.arange(32, dtype=np.float32)
        with self.assertRaisesRegex(
            ValueError, "allow_trivial_match is only valid for ab-joins"
        ):
            mp.selfjoin_1nn(series, 8, allow_trivial_match=False)
        with self.assertRaisesRegex(ValueError, "Metal does not support float64"):
            mp.abjoin_1nn(
                series,
                series,
                8,
                precision="double",
                gpus=[0],
            )

    @unittest.skipUnless(mx.metal.is_available(), "Metal is unavailable")
    def test_metal_profile_omits_index_pass(self):
        mx.set_default_device(mx.gpu)
        rng = np.random.default_rng(81)
        series = rng.normal(size=257).astype(np.float32)
        expected = mp.selfjoin(series, 11, pearson=True, precision="single")[0]

        with mock.patch.object(_metal_1nn, "_INDEX_KERNEL") as index_kernel:
            actual = mp.selfjoin_1nn(
                series,
                11,
                pearson=True,
                precision="single",
            )

        index_kernel.assert_not_called()
        np.testing.assert_array_equal(actual, expected)

    @unittest.skipUnless(mx.metal.is_available(), "Metal is unavailable")
    def test_metal_aligned_ab_matches_indexed_values_with_invalid_windows(self):
        mx.set_default_device(mx.gpu)
        rng = np.random.default_rng(19)
        a = rng.normal(size=129).astype(np.float32)
        b = rng.normal(size=129).astype(np.float32)
        b[41] = np.nan
        m = 8
        exclusion = (m + 3) // 4
        expected = mp.abjoin(
            a,
            b,
            m,
            pearson=False,
            precision="single",
            allow_trivial_match=False,
        )[0]

        with (
            mock.patch.object(
                _metal_1nn,
                "best_profile",
                wraps=_metal_1nn.best_profile,
            ) as kernel,
            mock.patch.object(_metal_1nn, "_INDEX_KERNEL") as index_kernel,
        ):
            actual = mp.abjoin_1nn(
                a,
                b,
                m,
                pearson=False,
                precision="single",
                allow_trivial_match=False,
            )

        self.assertEqual(exclusion, kernel.call_args.args[-1])
        index_kernel.assert_not_called()
        np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
