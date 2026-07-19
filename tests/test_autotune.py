import inspect
import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest import mock

import numpy as np

import mlx_native_scamp
import mlx_native_scamp._autotune as tuning
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


if __name__ == "__main__":
    unittest.main()
