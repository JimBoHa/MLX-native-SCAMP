import io
import re
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
from unittest import mock

import mlx.core as mx
import numpy as np

import mlx_native_scamp as native
from mlx_native_scamp import _autotune_cache as tuning_cache
from mlx_native_scamp import core
import pyscamp


class VerbosePublicApiTests(unittest.TestCase):
    def setUp(self):
        generator = np.random.default_rng(20260719)
        self.a = generator.standard_normal(48).astype(np.float32)
        self.b = generator.standard_normal(52).astype(np.float32)
        self.m = 8

    def _cases(self):
        return (
            ("selfjoin", "1nn", lambda **kw: pyscamp.selfjoin(self.a, self.m, **kw)),
            (
                "abjoin",
                "1nn",
                lambda **kw: pyscamp.abjoin(self.a, self.b, self.m, **kw),
            ),
            (
                "selfjoin",
                "sum",
                lambda **kw: pyscamp.selfjoin_sum(
                    self.a, self.m, threshold=0.25, **kw
                ),
            ),
            (
                "abjoin",
                "sum",
                lambda **kw: pyscamp.abjoin_sum(
                    self.a, self.b, self.m, threshold=0.25, **kw
                ),
            ),
            (
                "selfjoin",
                "matrix",
                lambda **kw: pyscamp.selfjoin_matrix(
                    self.a, self.m, mheight=4, mwidth=5, **kw
                ),
            ),
            (
                "abjoin",
                "matrix",
                lambda **kw: pyscamp.abjoin_matrix(
                    self.a,
                    self.b,
                    self.m,
                    mheight=4,
                    mwidth=5,
                    **kw,
                ),
            ),
            (
                "selfjoin",
                "knn",
                lambda **kw: pyscamp.selfjoin_knn(self.a, self.m, 2, **kw),
            ),
            (
                "abjoin",
                "knn",
                lambda **kw: pyscamp.abjoin_knn(
                    self.a, self.b, self.m, 2, **kw
                ),
            ),
            (
                "selfjoin",
                "1nn_value",
                lambda **kw: native.selfjoin_1nn(self.a, self.m, **kw),
            ),
            (
                "abjoin",
                "1nn_value",
                lambda **kw: native.abjoin_1nn(
                    self.a, self.b, self.m, **kw
                ),
            ),
            (
                "abjoin",
                "1nn_bidirectional",
                lambda **kw: native.abjoin_bidirectional(
                    self.a, self.b, self.m, **kw
                ),
            ),
        )

    def test_all_public_profiles_are_silent_by_default(self):
        for join, profile, operation in self._cases():
            with self.subTest(join=join, profile=profile):
                output = io.StringIO()
                with redirect_stdout(output):
                    operation(gpus=[])
                self.assertEqual("", output.getvalue())

    def test_all_public_profiles_report_resolved_cpu_metadata(self):
        expected_a = len(self.a) - self.m + 1
        for join, profile, operation in self._cases():
            with self.subTest(join=join, profile=profile):
                output = io.StringIO()
                with redirect_stdout(output):
                    operation(verbose=True, gpus=[])
                lines = output.getvalue().splitlines()

                self.assertEqual(2, len(lines))
                start = re.match(
                    rf"^mlx-scamp op=(\d+) {join}/{profile} start: (.+)$",
                    lines[0],
                )
                self.assertIsNotNone(start)
                assert start is not None
                operation_id, metadata = start.groups()
                expected_b = (
                    expected_a
                    if join == "selfjoin"
                    else len(self.b) - self.m + 1
                )
                for field in (
                    f"a_subsequences={expected_a}",
                    f"b_subsequences={expected_b}",
                    f"window={self.m}",
                    "precision=double",
                    "dtype=float64",
                    "backend=cpu",
                    "implementation=portable",
                    "max_tile_size=128000",
                ):
                    self.assertIn(field, metadata)
                self.assertRegex(metadata, r"geometry=\d+x\d+$")
                self.assertRegex(
                    lines[1],
                    rf"^mlx-scamp op={operation_id} {join}/{profile} "
                    r"complete: backend=cpu implementation=portable "
                    r"elapsed=\d+\.\d{6}s$",
                )

    def test_verbose_does_not_change_results(self):
        expected = pyscamp.abjoin(
            self.a, self.b, self.m, pearson=True, gpus=[]
        )
        with redirect_stdout(io.StringIO()):
            observed = pyscamp.abjoin(
                self.a,
                self.b,
                self.m,
                pearson=True,
                gpus=[],
                verbose=True,
            )
        np.testing.assert_allclose(
            expected[0], observed[0], equal_nan=True, rtol=0.0, atol=0.0
        )
        np.testing.assert_array_equal(expected[1], observed[1])

    def test_invalid_calls_are_silent_and_do_not_start_a_timer(self):
        output = io.StringIO()
        with mock.patch.object(
            core.time,
            "perf_counter_ns",
            side_effect=AssertionError("timer started"),
        ), redirect_stdout(output):
            with self.assertRaisesRegex(ValueError, r"len\(a\)"):
                pyscamp.selfjoin(self.a[:4], self.m, verbose=True)
        self.assertEqual("", output.getvalue())


class StructuredExecutionEventTests(unittest.TestCase):
    def _decision(self):
        return core._ExecutionDecision(
            join="abjoin",
            profile="1nn",
            subsequences_a=17,
            subsequences_b=19,
            window=8,
            precision="single",
            compute_dtype="float32",
            backend="portable_metal",
            implementation="portable",
            max_tile_size=1024,
            geometry="portable",
            tile_rows=16,
            tile_columns=17,
        )

    def test_structured_events_order_sync_timing_and_operation_id(self):
        events = []
        sequence = []

        def sink(event):
            events.append(event)
            sequence.append(event.phase)

        with mock.patch.object(
            core, "_EXECUTION_EVENT_SINK", side_effect=sink
        ), mock.patch.object(
            core, "_EXECUTION_OPERATION_IDS", iter((41,))
        ), mock.patch.object(
            core.mx, "synchronize", side_effect=lambda: sequence.append("sync")
        ), mock.patch.object(
            core.time, "perf_counter_ns", return_value=2_500
        ):
            result = core._execute_with_reporting(
                self._decision(),
                1_000,
                lambda: sequence.append("execute") or "result",
            )

        self.assertEqual("result", result)
        self.assertEqual(["start", "execute", "sync", "complete"], sequence)
        self.assertEqual(["start", "complete"], [event.phase for event in events])
        self.assertEqual({41}, {event.operation_id for event in events})
        self.assertEqual(1_500, events[-1].elapsed_ns)
        self.assertEqual(self._decision(), events[0].decision)

    def test_reducer_failure_emits_error_without_false_completion(self):
        events = []

        def fail():
            raise RuntimeError("reducer failed")

        with mock.patch.object(
            core, "_EXECUTION_EVENT_SINK", side_effect=events.append
        ), mock.patch.object(
            core, "_EXECUTION_OPERATION_IDS", iter((7,))
        ), mock.patch.object(
            core.time, "perf_counter_ns", return_value=4_000
        ), mock.patch.object(core.mx, "synchronize") as synchronize:
            with self.assertRaisesRegex(RuntimeError, "reducer failed"):
                core._execute_with_reporting(self._decision(), 1_000, fail)

        self.assertEqual(["start", "error"], [event.phase for event in events])
        self.assertEqual("RuntimeError", events[-1].error_type)
        self.assertEqual(3_000, events[-1].elapsed_ns)
        synchronize.assert_not_called()

    def test_concurrent_events_keep_unique_correlated_operation_ids(self):
        events = []

        def execute_one(_index):
            started_ns = core.time.perf_counter_ns()
            return core._execute_with_reporting(
                self._decision(), started_ns, lambda: "result"
            )

        with mock.patch.object(
            core, "_EXECUTION_EVENT_SINK", side_effect=events.append
        ), mock.patch.object(
            core, "_EXECUTION_OPERATION_IDS", iter(range(100, 116))
        ), mock.patch.object(core.mx, "synchronize"):
            with ThreadPoolExecutor(max_workers=8) as pool:
                results = list(pool.map(execute_one, range(16)))

        self.assertEqual(["result"] * 16, results)
        grouped = {}
        for event in events:
            grouped.setdefault(event.operation_id, []).append(event.phase)
        self.assertEqual(set(range(100, 116)), set(grouped))
        self.assertTrue(
            all(phases == ["start", "complete"] for phases in grouped.values())
        )

    def test_false_path_never_constructs_or_formats_reporting(self):
        series = np.arange(32, dtype=np.float32)
        with mock.patch.object(
            core,
            "_execute_with_reporting",
            side_effect=AssertionError("reporter constructed"),
        ), mock.patch.object(
            core.time,
            "perf_counter_ns",
            side_effect=AssertionError("timer started"),
        ), mock.patch.object(
            core,
            "_format_execution_event",
            side_effect=AssertionError("event formatted"),
        ), mock.patch("builtins.print", side_effect=AssertionError("printed")):
            profile, indices = pyscamp.selfjoin(series, 8, gpus=[])
        self.assertEqual(profile.shape, indices.shape)

    def test_stdout_formatter_has_fixed_start_complete_and_error_shapes(self):
        decision = self._decision()
        start = core._format_execution_event(
            core._ExecutionEvent(3, "start", decision)
        )
        complete = core._format_execution_event(
            core._ExecutionEvent(3, "complete", decision, elapsed_ns=1_250_000)
        )
        error = core._format_execution_event(
            core._ExecutionEvent(
                3,
                "error",
                decision,
                elapsed_ns=1_250_000,
                error_type="RuntimeError",
            )
        )
        self.assertIn("geometry=16x17", start)
        self.assertTrue(complete.endswith("elapsed=0.001250s"))
        self.assertTrue(error.endswith("exception=RuntimeError"))


@unittest.skipUnless(mx.metal.is_available(), "Metal is unavailable")
class MetalVerboseRouteTests(unittest.TestCase):
    def setUp(self):
        generator = np.random.default_rng(42)
        self.a = generator.standard_normal(256).astype(np.float32)
        self.b = generator.standard_normal(260).astype(np.float32)
        self.m = 32

    def _start_line(self, operation):
        output = io.StringIO()
        with redirect_stdout(output):
            operation()
        lines = output.getvalue().splitlines()
        self.assertEqual(2, len(lines))
        return lines[0]

    def test_real_custom_and_portable_metal_routes_are_truthful(self):
        cases = (
            (
                "implementation=metal_1nn",
                lambda: pyscamp.selfjoin(
                    self.a,
                    self.m,
                    precision="single",
                    gpus=[0],
                    verbose=True,
                ),
            ),
            (
                "implementation=metal_1nn",
                lambda: native.selfjoin_1nn(
                    self.a,
                    self.m,
                    precision="single",
                    gpus=[0],
                    verbose=True,
                ),
            ),
            (
                "implementation=metal_bidirectional",
                lambda: native.abjoin_bidirectional(
                    self.a,
                    self.b,
                    self.m,
                    precision="single",
                    gpus=[0],
                    verbose=True,
                ),
            ),
            (
                "implementation=metal_matrix",
                lambda: pyscamp.abjoin_matrix(
                    self.a,
                    self.b,
                    self.m,
                    mheight=8,
                    mwidth=8,
                    precision="single",
                    gpus=[0],
                    verbose=True,
                ),
            ),
            (
                "backend=portable_metal implementation=portable",
                lambda: pyscamp.abjoin_knn(
                    self.a,
                    self.b,
                    self.m,
                    2,
                    precision="single",
                    gpus=[0],
                    verbose=True,
                ),
            ),
        )
        for expected, operation in cases:
            with self.subTest(expected=expected):
                line = self._start_line(operation)
                self.assertIn(expected, line)
                if expected.startswith("implementation=metal"):
                    self.assertIn("backend=custom_metal", line)

        with mock.patch.object(
            core, "_metal_sum_is_worthwhile", return_value=True
        ):
            line = self._start_line(
                lambda: pyscamp.selfjoin_sum(
                    self.a,
                    self.m,
                    threshold=0.8,
                    precision="single",
                    gpus=[0],
                    verbose=True,
                )
            )
        self.assertIn("backend=custom_metal implementation=metal_sum", line)

    def test_each_custom_gate_reports_portable_fallback(self):
        fallback_operations = (
            lambda: pyscamp.selfjoin(
                self.a.astype(np.float64),
                self.m,
                precision="single",
                gpus=[0],
                verbose=True,
            ),
            lambda: pyscamp.abjoin_sum(
                self.a,
                self.b,
                self.m,
                threshold=-0.1,
                precision="single",
                gpus=[0],
                verbose=True,
            ),
        )
        for operation in fallback_operations:
            line = self._start_line(operation)
            self.assertIn("backend=portable_metal implementation=portable", line)

        with mock.patch.object(
            core, "_metal_recurrence_is_safe", return_value=False
        ):
            line = self._start_line(
                lambda: pyscamp.selfjoin(
                    self.a,
                    self.m,
                    precision="single",
                    gpus=[0],
                    verbose=True,
                )
            )
        self.assertIn("backend=portable_metal implementation=portable", line)

        with mock.patch.object(
            core, "_metal_sum_is_worthwhile", return_value=False
        ):
            line = self._start_line(
                lambda: pyscamp.selfjoin_sum(
                    self.a,
                    self.m,
                    threshold=0.8,
                    precision="single",
                    gpus=[0],
                    verbose=True,
                )
            )
        self.assertIn("backend=portable_metal implementation=portable", line)

        with mock.patch(
            "mlx_native_scamp._metal_matrix.indexing_is_safe",
            return_value=False,
        ):
            line = self._start_line(
                lambda: pyscamp.abjoin_matrix(
                    self.a,
                    self.b,
                    self.m,
                    mheight=8,
                    mwidth=8,
                    precision="single",
                    gpus=[0],
                    verbose=True,
                )
            )
        self.assertIn("backend=portable_metal implementation=portable", line)

        oversized = np.random.default_rng(9).standard_normal(1050).astype(
            np.float32
        )
        with mock.patch.object(
            core, "_best_match_profile", return_value="profile"
        ):
            line = self._start_line(
                lambda: pyscamp.selfjoin(
                    oversized,
                    self.m,
                    max_tile_size=1024,
                    precision="single",
                    gpus=[0],
                    verbose=True,
                )
            )
        self.assertIn("backend=portable_metal implementation=portable", line)

    def test_tuned_routes_are_forwarded_to_the_execution_reporter(self):
        params = {
            "gpus": None,
            "threads": 0,
            "precision": "single",
            "max_tile_size": None,
            "verbose": True,
        }
        strategies = (
            ("1nn_index:cpu:rows-64", "cpu", False),
            ("1nn_index:portable_metal:rows-64", "portable_metal", False),
            ("1nn_index:metal-diagonal", "portable_metal", True),
        )
        for name, backend, custom in strategies:
            strategy = tuning_cache.STRATEGY_BY_NAME[name]
            with self.subTest(strategy=name), mock.patch.object(
                core, "_implicit_tuning_strategy", return_value=strategy
            ), mock.patch.object(
                core, "_run_profile", return_value="profile"
            ) as run_profile:
                result = core._run_profile_with_resources(
                    params,
                    self.a,
                    None,
                    self.m,
                    pearson=True,
                    profile="1nn",
                )
            self.assertEqual("profile", result)
            options = run_profile.call_args.kwargs
            self.assertTrue(options["verbose"])
            self.assertEqual(backend, options["portable_backend"])
            self.assertEqual(custom, options["use_metal_1nn"])

    def test_cached_recommendations_report_the_route_that_actually_runs(self):
        cases = (
            (
                "1nn_index:cpu:rows-64",
                self.a,
                "backend=cpu implementation=portable",
            ),
            (
                "1nn_index:portable_metal:rows-64",
                self.a,
                "backend=portable_metal implementation=portable",
            ),
            (
                "1nn_index:metal-diagonal",
                self.a.astype(np.float64),
                "backend=portable_metal implementation=portable",
            ),
        )
        for name, series, expected in cases:
            strategy = tuning_cache.STRATEGY_BY_NAME[name]
            with self.subTest(strategy=name), mock.patch.object(
                core, "_implicit_tuning_strategy", return_value=strategy
            ):
                line = self._start_line(
                    lambda: pyscamp.selfjoin(
                        series,
                        self.m,
                        precision="single",
                        verbose=True,
                    )
                )
            self.assertIn(expected, line)


if __name__ == "__main__":
    unittest.main()
