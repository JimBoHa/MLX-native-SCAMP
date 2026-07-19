import inspect
import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest import mock

import mlx.core as mx
import numpy as np

import mlx_native_scamp
import mlx_native_scamp._autotune as tuning
from mlx_native_scamp import _autotune_cache as tuning_cache
from mlx_native_scamp import core as scamp_core
import pyscamp


def _profile_result(index=1):
    return (
        np.array([0.25, np.nan], dtype=np.float32),
        np.array([index, -1], dtype=np.int32),
    )


def _small_workload(
    profile="1nn_index", precision="single", **overrides
):
    values = {
        "name": f"test-{profile}-{precision}",
        "profile": profile,
        "precision": precision,
        "n_a": 64,
        "n_b": 64,
        "m": 16,
        "self_join": True,
    }
    if profile == "sum_thresh":
        values["threshold_density"] = 0.1
    elif profile == "matrix_summary":
        values["matrix_shape"] = (4, 4)
    elif profile == "knn":
        values["k"] = 3
    values.update(overrides)
    return tuning.AutotuneWorkload(**values)


class _CandidateClock:
    def __init__(self, current, durations):
        self.current = current
        self.durations = durations
        self.total = 0
        self.starting = True

    def __call__(self):
        if self.starting:
            self.starting = False
            return self.total
        self.total += self.durations[self.current[0]]
        self.starting = True
        return self.total


class AutotunePlanTests(unittest.TestCase):
    def test_strict_signature_and_namespace_remain_upstream_compatible(self):
        self.assertIs(pyscamp.autotune, mlx_native_scamp.autotune)
        signature = inspect.signature(pyscamp.autotune)
        self.assertEqual(["devices", "cache_path"], list(signature.parameters))
        self.assertIsNone(signature.parameters["devices"].default)
        self.assertEqual("", signature.parameters["cache_path"].default)
        self.assertIn("autotune", pyscamp.__all__)
        for extension in (
            "autotune_plan",
            "run_autotune",
            "strategy_descriptions",
        ):
            self.assertIn(extension, mlx_native_scamp.__all__)
            self.assertNotIn(extension, pyscamp.__all__)

    def test_quick_plan_covers_five_upstream_families_and_precisions(self):
        plan = tuning.autotune_plan("quick")
        observed = {
            (workload.profile, workload.precision)
            for workload in plan.workloads
        }
        expected = {
            (profile, precision)
            for profile in tuning.UPSTREAM_PROFILE_FAMILIES
            for precision in ("single", "double")
        }
        self.assertEqual(expected, observed)
        self.assertEqual(10, len(plan.workloads))
        self.assertEqual(len(plan.workloads), len({row.key for row in plan.workloads}))
        self.assertLessEqual(max(row.n_a for row in plan.workloads), 512)

    def test_full_plan_adds_large_asymmetric_profile_and_native_buckets(self):
        quick = tuning.autotune_plan("quick")
        full = tuning.autotune_plan("full")
        self.assertEqual(quick.workloads, full.workloads[: len(quick.workloads)])
        added = full.workloads[len(quick.workloads) :]
        self.assertTrue(any(max(row.n_a, row.n_b) >= 4096 for row in added))
        self.assertTrue(any(row.n_a != row.n_b for row in added))
        self.assertTrue(
            any(row.threshold_density in {0.01, 0.5} for row in added)
        )
        self.assertTrue(any(row.matrix_shape == (16, 64) for row in added))
        self.assertTrue(any(row.k == 32 for row in added))
        self.assertEqual(
            {"single", "double"},
            {
                row.precision
                for row in added
                if row.profile == "bidirectional_ab"
            },
        )

    def test_invalid_mode_is_rejected_without_running_work(self):
        with self.assertRaisesRegex(ValueError, "quick.*full"):
            tuning.autotune_plan("slow")

    def test_strategy_descriptions_are_typed_and_complete(self):
        descriptions = tuning.strategy_descriptions()
        self.assertEqual(len(tuning.cache.STRATEGIES), len(descriptions))
        self.assertEqual(
            set(tuning.cache.STRATEGIES),
            {description.strategy for description in descriptions},
        )
        self.assertTrue(all(description.summary for description in descriptions))
        self.assertTrue(
            all(
                description.backend in {"cpu", "portable_metal", "custom_metal"}
                for description in descriptions
            )
        )

    def test_candidate_eligibility_is_precision_route_and_family_specific(self):
        double = tuning._eligible_strategies(_small_workload(precision="double"))
        ultra = tuning._eligible_strategies(_small_workload(precision="ultra"))
        single = tuning._eligible_strategies(_small_workload())
        self.assertTrue(double)
        self.assertTrue(ultra)
        self.assertTrue(all(row.route == "cpu" for row in double + ultra))
        self.assertTrue(any(row.route == "metal_1nn" for row in single))
        self.assertTrue(all(row.profile == "1nn_index" for row in single))


class ResultEquivalenceTests(unittest.TestCase):
    def test_array_profile_and_index_families_use_their_own_semantics(self):
        tuning._assert_result_equivalent(
            "1nn_value",
            "single",
            np.array([0.5 + 1e-5, np.nan]),
            np.array([0.5, np.nan]),
        )
        tuning._assert_result_equivalent(
            "1nn_index", "single", _profile_result(), _profile_result()
        )
        with self.assertRaises(AssertionError):
            tuning._assert_result_equivalent(
                "1nn_index", "single", _profile_result(2), _profile_result(1)
            )

    def test_knn_requires_exact_pairs_and_tolerates_small_score_error(self):
        reference = [(1, 7, 0.5), (2, 8, 0.25)]
        tuning._assert_result_equivalent(
            "knn",
            "single",
            [(1, 7, 0.50001), (2, 8, 0.25)],
            reference,
        )
        with self.assertRaises(AssertionError):
            tuning._assert_result_equivalent(
                "knn", "single", [(1, 6, 0.5), (2, 8, 0.25)], reference
            )

    def test_matrix_shape_and_bidirectional_indices_are_checked(self):
        with self.assertRaises(AssertionError):
            tuning._assert_result_equivalent(
                "matrix_summary",
                "single",
                np.zeros((2, 2)),
                np.zeros((4, 1)),
            )
        reference = (_profile_result(1), _profile_result(2))
        tuning._assert_result_equivalent(
            "bidirectional_ab", "single", reference, reference
        )
        with self.assertRaises(AssertionError):
            tuning._assert_result_equivalent(
                "bidirectional_ab",
                "single",
                (_profile_result(1), _profile_result(3)),
                reference,
            )

    def test_malformed_results_are_rejected(self):
        with self.assertRaises(AssertionError):
            tuning._assert_result_equivalent(
                "1nn_index", "single", np.zeros(2), _profile_result()
            )


class AutotuneRunnerTests(unittest.TestCase):
    def test_devices_are_deduplicated_and_nonzero_devices_rejected(self):
        plan = tuning.AutotunePlan(
            "quick", (_small_workload(),), warmups=0, trials=1
        )
        records = []
        ticks = iter(range(0, 10_000, 10))
        with redirect_stdout(StringIO()):
            result = tuning.run_autotune(
                [0, 0, 0],
                "cache.txt",
                plan=plan,
                executor=lambda *_: _profile_result(),
                synchronize=lambda: None,
                clock=lambda: next(ticks),
                record_writer=lambda record, path: records.append((record, path)),
            )
        self.assertEqual(1, result)
        self.assertEqual(1, len(records))
        self.assertEqual("cache.txt", records[0][1])

        for devices in ([1], [-1], [0, 1]):
            with self.subTest(devices=devices):
                with self.assertRaisesRegex(ValueError, "GPU device ID"):
                    tuning.run_autotune(devices, executor=lambda *_: None, plan=plan)
        for devices in (0, "0", [0, 1.5]):
            with self.subTest(devices=devices):
                with self.assertRaisesRegex(TypeError, "devices"):
                    tuning.run_autotune(devices, executor=lambda *_: None, plan=plan)

    def test_warmups_are_outside_synchronized_median_timing(self):
        workload = _small_workload()
        strategy = tuning._eligible_strategies(workload)[0]
        events = []
        ticks = iter((100, 111, 200, 235))

        def executor(*_args):
            events.append("execute")
            return _profile_result()

        def synchronize():
            events.append("sync")

        measurement = tuning._measure_candidate(
            workload,
            strategy,
            _profile_result(),
            executor=executor,
            synchronize=synchronize,
            clock=lambda: next(ticks),
            warmups=1,
            trials=2,
        )
        self.assertEqual((11, 35), measurement.samples_ns)
        self.assertEqual(23, measurement.duration_ns)
        self.assertEqual(
            [
                "execute",
                "sync",
                "sync",
                "execute",
                "sync",
                "sync",
                "execute",
                "sync",
            ],
            events,
        )

    def test_runner_rejects_wrong_candidate_and_uses_resource_tie_break(self):
        workload = _small_workload()
        plan = tuning.AutotunePlan("quick", (workload,), warmups=0, trials=3)
        current = [""]
        durations = {
            strategy.name: (
                50
                if strategy.route == "cpu"
                else 5
                if strategy.route == "portable_metal"
                and dict(strategy.parameters)["portable_row_cap"] in {64, 128}
                else 100
            )
            for strategy in tuning._eligible_strategies(workload)
        }
        wrong = next(
            strategy.name
            for strategy in tuning._eligible_strategies(workload)
            if strategy.route == "portable_metal"
            and dict(strategy.parameters)["portable_row_cap"] == 64
        )
        records = []

        def executor(_workload, strategy):
            current[0] = strategy.name
            return _profile_result(9 if strategy.name == wrong else 1)

        output = StringIO()
        with redirect_stdout(output):
            tuning.run_autotune(
                plan=plan,
                executor=executor,
                synchronize=lambda: None,
                clock=_CandidateClock(current, durations),
                record_writer=lambda record, _path: records.append(record),
            )

        self.assertIn("rejected", output.getvalue())
        self.assertEqual(1, len(records))
        # rows-64 was incorrect, so equal-duration rows-128 beats the more
        # resource-intensive candidates by the deterministic second key.
        self.assertEqual(
            "1nn_index:portable_metal:rows-128", records[0].candidate
        )
        self.assertEqual(5, records[0].duration_ns)

    def test_injected_runner_never_resolves_lazy_core_hook(self):
        plan = tuning.AutotunePlan(
            "quick", (_small_workload(),), warmups=0, trials=1
        )
        ticks = iter(range(0, 10_000, 10))
        with mock.patch.object(
            tuning, "_default_executor", side_effect=AssertionError("imported")
        ), redirect_stdout(StringIO()):
            tuning.run_autotune(
                plan=plan,
                executor=lambda *_: _profile_result(),
                synchronize=lambda: None,
                clock=lambda: next(ticks),
                record_writer=lambda *_: None,
            )

    def test_default_runner_checks_metal_before_resolving_core_hook(self):
        with mock.patch.object(
            tuning.mx.metal, "is_available", return_value=False
        ), mock.patch.object(
            tuning, "_default_executor", side_effect=AssertionError("resolved")
        ):
            with self.assertRaisesRegex(ValueError, "No Metal device"):
                pyscamp.autotune(cache_path="unused")

    def test_explicit_stdout_reports_plan_trials_selection_and_cache(self):
        plan = tuning.AutotunePlan(
            "quick", (_small_workload(),), warmups=0, trials=1
        )
        ticks = iter(range(0, 10_000, 10))
        output = StringIO()
        with redirect_stdout(output):
            tuning.run_autotune(
                plan=plan,
                executor=lambda *_: _profile_result(),
                synchronize=lambda: None,
                clock=lambda: next(ticks),
                record_writer=lambda *_: None,
            )
        text = output.getvalue()
        self.assertIn("plan=quick", text)
        self.assertIn("trial 1/1", text)
        self.assertIn("selected:", text)
        self.assertIn("cache=", text)

    def test_cache_path_type_and_plan_counts_are_validated(self):
        plan = tuning.AutotunePlan(
            "quick", (_small_workload(),), warmups=0, trials=1
        )
        with self.assertRaisesRegex(TypeError, "cache_path"):
            tuning.run_autotune(
                cache_path=None, executor=lambda *_: None, plan=plan
            )
        for invalid in (
            tuning.AutotunePlan("quick", plan.workloads, -1, 1),
            tuning.AutotunePlan("quick", plan.workloads, 0, 0),
        ):
            with self.assertRaises(ValueError):
                tuning.run_autotune(executor=lambda *_: None, plan=invalid)


class AutotuneCoreIntegrationTests(unittest.TestCase):
    def _params(self, **overrides):
        values = {
            "gpus": None,
            "threads": 0,
            "precision": "single",
            "max_tile_size": None,
        }
        values.update(overrides)
        return values

    def test_runtime_key_includes_alignment_and_profile_controls(self):
        params = self._params()
        params["allow_trivial_match"] = False
        a = np.arange(64, dtype=np.float32)
        b = np.arange(48, dtype=np.float32)

        with mock.patch.object(
            tuning_cache, "lookup_record", return_value=None
        ) as lookup:
            strategy = scamp_core._implicit_tuning_strategy(
                params,
                (a, b, 8),
                {"profile": "knn", "k": 7},
            )

        self.assertIsNone(strategy)
        key = lookup.call_args.args[0]
        self.assertEqual("knn", key.profile)
        self.assertEqual("ab", key.join)
        self.assertEqual("aligned", key.alignment)
        self.assertEqual("k:2^3", key.profile_bucket)

    def test_explicit_resources_never_consult_tuning_cache(self):
        series = np.arange(32, dtype=np.float32)
        with mock.patch.object(
            tuning_cache,
            "lookup_record",
            side_effect=AssertionError("cache was consulted"),
        ):
            for params in (
                self._params(gpus=[]),
                self._params(threads=1),
            ):
                self.assertIsNone(
                    scamp_core._implicit_tuning_strategy(
                        params,
                        (series, None, 8),
                        {"profile": "1nn"},
                    )
                )

    def test_cpu_recommendation_preserves_tile_ceiling_and_disables_custom_metal(self):
        strategy = tuning_cache.STRATEGY_BY_NAME[
            "1nn_index:cpu:rows-64"
        ]
        series = np.arange(32, dtype=np.float32)
        with mock.patch.object(
            scamp_core, "_implicit_tuning_strategy", return_value=strategy
        ), mock.patch.object(
            scamp_core, "_run_profile", return_value="profile"
        ) as run_profile:
            result = scamp_core._run_profile_with_resources(
                self._params(max_tile_size=1024),
                series,
                None,
                8,
                profile="1nn",
                pearson=True,
            )

        self.assertEqual("profile", result)
        options = run_profile.call_args.kwargs
        self.assertEqual(1024, options["max_tile_size"])
        self.assertEqual(64, options["portable_row_cap"])
        self.assertFalse(options["use_metal_1nn"])
        self.assertFalse(options["use_metal_matrix"])
        self.assertFalse(options["use_metal_sum"])

    def test_portable_row_cap_remains_bounded_by_current_scheduler(self):
        rows, _ = scamp_core._portable_tile_shape(
            4096, 4096, 8, mx.float32, 8192, row_cap=64
        )
        self.assertLessEqual(rows, 64)
        with self.assertRaisesRegex(ValueError, "row cap"):
            scamp_core._portable_tile_shape(
                4096,
                4096,
                8,
                mx.float32,
                8192,
                row_cap=scamp_core.BLOCK_ROWS + 1,
            )

    def test_cpu_executor_covers_every_result_family(self):
        workloads = (
            _small_workload(),
            _small_workload("1nn_value"),
            _small_workload("sum_thresh"),
            _small_workload(
                "matrix_summary",
                n_b=48,
                self_join=False,
                matrix_shape=(4, 5),
            ),
            _small_workload("knn"),
            _small_workload(
                "bidirectional_ab", n_b=48, self_join=False
            ),
        )
        for workload in workloads:
            with self.subTest(profile=workload.profile):
                strategy = tuning_cache.STRATEGY_BY_NAME[
                    f"{workload.profile}:cpu:rows-64"
                ]
                result = scamp_core._autotune_execute_candidate(
                    workload, strategy
                )
                tuning._snapshot_result(workload.profile, result)

    @unittest.skipUnless(mx.metal.is_available(), "Metal is unavailable")
    def test_default_executor_runs_real_portable_and_custom_candidates(self):
        workload = _small_workload(n_a=32, n_b=32, m=8)
        cpu = tuning_cache.STRATEGY_BY_NAME[
            "1nn_index:cpu:rows-64"
        ]
        portable = tuning_cache.STRATEGY_BY_NAME[
            "1nn_index:portable_metal:rows-64"
        ]
        custom = tuning_cache.STRATEGY_BY_NAME[
            "1nn_index:metal-diagonal"
        ]
        reference = scamp_core._autotune_execute_candidate(workload, cpu)
        for strategy in (portable, custom):
            with self.subTest(strategy=strategy.name):
                candidate = scamp_core._autotune_execute_candidate(
                    workload, strategy
                )
                tuning._assert_result_equivalent(
                    workload.profile,
                    workload.precision,
                    candidate,
                    tuning._snapshot_result(workload.profile, reference),
                )

    @unittest.skipUnless(mx.metal.is_available(), "Metal is unavailable")
    def test_explicit_runner_uses_lazy_real_core_executor(self):
        plan = tuning.AutotunePlan(
            "quick",
            (_small_workload(n_a=32, n_b=32, m=8),),
            warmups=0,
            trials=1,
        )
        records = []
        with redirect_stdout(StringIO()):
            result = tuning.run_autotune(
                plan=plan,
                record_writer=lambda record, _path: records.append(record),
            )

        self.assertEqual(1, result)
        self.assertEqual(1, len(records))
        self.assertEqual(plan.workloads[0].key, records[0].key)


if __name__ == "__main__":
    unittest.main()
