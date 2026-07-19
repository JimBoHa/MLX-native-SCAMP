import inspect
import unittest
from dataclasses import fields

import mlx_native_scamp
import pyscamp


STRICT_CALLABLES = {
    "abjoin",
    "abjoin_knn",
    "abjoin_matrix",
    "abjoin_sum",
    "autotune",
    "gpu_supported",
    "selfjoin",
    "selfjoin_knn",
    "selfjoin_matrix",
    "selfjoin_sum",
}

JOIN_CALLABLES = {
    "abjoin",
    "abjoin_1nn",
    "abjoin_bidirectional",
    "abjoin_knn",
    "abjoin_matrix",
    "abjoin_sum",
    "selfjoin",
    "selfjoin_1nn",
    "selfjoin_knn",
    "selfjoin_matrix",
    "selfjoin_sum",
}


class PublicDocumentationTests(unittest.TestCase):
    def _assert_substantive_doc(self, owner, name):
        value = getattr(owner, name)
        self.assertTrue(callable(value), name)
        documentation = inspect.getdoc(value)
        self.assertIsNotNone(documentation, name)
        self.assertGreaterEqual(len(documentation.split()), 20, name)
        return documentation

    def test_public_modules_explain_their_scope(self):
        for module in (pyscamp, mlx_native_scamp):
            with self.subTest(module=module.__name__):
                documentation = inspect.getdoc(module)
                self.assertIsNotNone(documentation)
                self.assertGreaterEqual(len(documentation.split()), 20)

    def test_strict_pyscamp_surface_is_complete_and_documented(self):
        self.assertEqual(STRICT_CALLABLES | {"__version__"}, set(pyscamp.__all__))
        for name in sorted(STRICT_CALLABLES):
            with self.subTest(name=name):
                self._assert_substantive_doc(pyscamp, name)
                self.assertIs(getattr(pyscamp, name), getattr(mlx_native_scamp, name))

    def test_every_native_top_level_callable_is_documented(self):
        for name in sorted(set(mlx_native_scamp.__all__) - {"__version__"}):
            with self.subTest(name=name):
                self._assert_substantive_doc(mlx_native_scamp, name)

    def test_join_docs_cover_every_shared_runtime_control(self):
        controls = {
            "gpus",
            "max_tile_size",
            "pearson",
            "precision",
            "threads",
            "verbose",
        }
        for name in sorted(JOIN_CALLABLES):
            documentation = inspect.getdoc(getattr(mlx_native_scamp, name))
            assert documentation is not None
            with self.subTest(name=name):
                for control in controls:
                    self.assertIn(control, documentation)
                self.assertNotIn("mixed", documentation.lower())
                self.assertIn("Returns", documentation)

    def test_profile_specific_controls_are_documented(self):
        groups = {
            "allow_trivial_match": {
                "abjoin",
                "abjoin_1nn",
                "abjoin_bidirectional",
                "abjoin_knn",
                "abjoin_matrix",
                "abjoin_sum",
            },
            "threshold": {
                "abjoin_knn",
                "abjoin_matrix",
                "abjoin_sum",
                "selfjoin_knn",
                "selfjoin_matrix",
                "selfjoin_sum",
            },
            "mheight": {"abjoin_matrix", "selfjoin_matrix"},
            "mwidth": {"abjoin_matrix", "selfjoin_matrix"},
            "k": {"abjoin_knn", "selfjoin_knn"},
        }
        for keyword, names in groups.items():
            for name in sorted(names):
                with self.subTest(name=name, keyword=keyword):
                    documentation = inspect.getdoc(getattr(mlx_native_scamp, name))
                    self.assertIn(keyword, documentation)

    def test_profile_output_dtypes_are_documented(self):
        groups = {
            "float32": {
                "abjoin",
                "abjoin_1nn",
                "abjoin_bidirectional",
                "abjoin_matrix",
                "selfjoin",
                "selfjoin_1nn",
                "selfjoin_matrix",
            },
            "int32": {"abjoin", "abjoin_bidirectional", "selfjoin"},
            "float64": {
                "abjoin_sum",
                "selfjoin_sum",
            },
        }
        for dtype, names in groups.items():
            for name in sorted(names):
                documentation = inspect.getdoc(getattr(mlx_native_scamp, name))
                with self.subTest(name=name, dtype=dtype):
                    self.assertIn(dtype, documentation)

    def test_public_autotune_types_document_fields_and_properties(self):
        public_types = (
            mlx_native_scamp.AutotunePlan,
            mlx_native_scamp.AutotuneWorkload,
            mlx_native_scamp.CandidateMeasurement,
            mlx_native_scamp.StrategyDescription,
        )
        for public_type in public_types:
            documentation = inspect.getdoc(public_type)
            assert documentation is not None
            for field in fields(public_type):
                with self.subTest(type=public_type.__name__, field=field.name):
                    self.assertIn(field.name, documentation)
        workload_doc = inspect.getdoc(mlx_native_scamp.AutotuneWorkload)
        self.assertIn("dtype_class", workload_doc)
        self.assertIn("key", workload_doc)

    def test_autotune_entry_points_document_selection_and_storage(self):
        for owner, name in (
            (pyscamp, "autotune"),
            (mlx_native_scamp, "run_autotune"),
        ):
            documentation = inspect.getdoc(getattr(owner, name))
            assert documentation is not None
            with self.subTest(owner=owner.__name__, name=name):
                for term in ("devices", "cache_path", "quick", "Metal"):
                    self.assertIn(term, documentation)


if __name__ == "__main__":
    unittest.main()
