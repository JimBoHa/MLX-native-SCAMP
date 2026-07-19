from __future__ import annotations

import operator
import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import mlx.core as mx
import numpy as np

from . import _autotune_cache as cache


AutotuneMode = Literal["quick", "full"]
UPSTREAM_PROFILE_FAMILIES = (
    "1nn_index",
    "1nn_value",
    "sum_thresh",
    "matrix_summary",
    "knn",
)


@dataclass(frozen=True, slots=True)
class AutotuneWorkload:
    """One reproducible workload bucket in an explicit autotune plan."""

    name: str
    profile: str
    precision: str
    n_a: int
    n_b: int
    m: int
    self_join: bool
    aligned: bool = False
    route: str = "auto"
    max_tile_size: int | None = None
    threshold_density: float | None = None
    k: int | None = None
    matrix_shape: tuple[int, int] | None = None

    @property
    def dtype_class(self) -> str:
        return "float32" if self.precision == "single" else "float64"

    @property
    def key(self) -> cache.WorkloadKey:
        return cache.make_workload_key(
            self.profile,
            self.precision,
            self.route,
            self.n_a,
            self.n_b,
            self.m,
            self_join=self.self_join,
            aligned=self.aligned,
            dtype_class=self.dtype_class,
            max_tile_size=self.max_tile_size,
            threshold_density=self.threshold_density,
            k=self.k,
            matrix_shape=self.matrix_shape,
        )


@dataclass(frozen=True, slots=True)
class AutotunePlan:
    """Immutable, inspectable work scheduled by :func:`run_autotune`."""

    mode: AutotuneMode
    workloads: tuple[AutotuneWorkload, ...]
    warmups: int
    trials: int


@dataclass(frozen=True, slots=True)
class StrategyDescription:
    """Typed presentation and tie-break metadata for a cached strategy."""

    strategy: cache.Strategy
    backend: Literal["cpu", "portable_metal", "custom_metal"]
    resource_rank: tuple[int, int]
    summary: str


@dataclass(frozen=True, slots=True)
class CandidateMeasurement:
    """The synchronized duration samples accepted for one candidate."""

    strategy: cache.Strategy
    samples_ns: tuple[int, ...]
    duration_ns: int
    resource_rank: tuple[int, int]


CandidateExecutor = Callable[[AutotuneWorkload, cache.Strategy], Any]
Synchronize = Callable[[], None]
Clock = Callable[[], int]
RecordWriter = Callable[[cache.TuningRecord, str], Any]


def _workload(
    name: str,
    profile: str,
    precision: str,
    *,
    n_a: int,
    n_b: int,
    m: int,
    self_join: bool,
    aligned: bool = False,
    max_tile_size: int | None = None,
    threshold_density: float | None = None,
    k: int | None = None,
    matrix_shape: tuple[int, int] | None = None,
) -> AutotuneWorkload:
    workload = AutotuneWorkload(
        name=name,
        profile=profile,
        precision=precision,
        n_a=n_a,
        n_b=n_b,
        m=m,
        self_join=self_join,
        aligned=aligned,
        max_tile_size=max_tile_size,
        threshold_density=threshold_density,
        k=k,
        matrix_shape=matrix_shape,
    )
    # Constructing the key here makes an invalid built-in plan fail at import-
    # independent plan construction, before an executor or cache is touched.
    workload.key
    return workload


def _quick_workloads() -> tuple[AutotuneWorkload, ...]:
    workloads: list[AutotuneWorkload] = []
    representatives = (
        ("1nn_index", dict(n_a=512, n_b=512, m=64, self_join=True)),
        ("1nn_value", dict(n_a=384, n_b=512, m=48, self_join=False)),
        (
            "sum_thresh",
            dict(
                n_a=512,
                n_b=512,
                m=64,
                self_join=True,
                threshold_density=0.1,
            ),
        ),
        (
            "matrix_summary",
            dict(
                n_a=512,
                n_b=384,
                m=64,
                self_join=False,
                matrix_shape=(16, 16),
            ),
        ),
        (
            "knn",
            dict(n_a=384, n_b=384, m=48, self_join=True, k=4),
        ),
    )
    for profile, parameters in representatives:
        for precision in ("single", "double"):
            workloads.append(
                _workload(
                    f"quick-{profile}-{precision}",
                    profile,
                    precision,
                    **parameters,
                )
            )
    return tuple(workloads)


def _full_workloads() -> tuple[AutotuneWorkload, ...]:
    return _quick_workloads() + (
        _workload(
            "full-large-1nn-index-single",
            "1nn_index",
            "single",
            n_a=4096,
            n_b=4096,
            m=128,
            self_join=True,
        ),
        _workload(
            "full-asymmetric-1nn-value-double",
            "1nn_value",
            "double",
            n_a=2048,
            n_b=512,
            m=96,
            self_join=False,
        ),
        _workload(
            "full-sparse-sum-single",
            "sum_thresh",
            "single",
            n_a=2048,
            n_b=2048,
            m=96,
            self_join=True,
            threshold_density=0.01,
        ),
        _workload(
            "full-dense-sum-double",
            "sum_thresh",
            "double",
            n_a=1024,
            n_b=2048,
            m=64,
            self_join=False,
            threshold_density=0.5,
        ),
        _workload(
            "full-wide-matrix-single",
            "matrix_summary",
            "single",
            n_a=2048,
            n_b=1024,
            m=64,
            self_join=False,
            matrix_shape=(16, 64),
        ),
        _workload(
            "full-large-k-double",
            "knn",
            "double",
            n_a=1024,
            n_b=1024,
            m=64,
            self_join=True,
            k=32,
        ),
        _workload(
            "full-bidirectional-single",
            "bidirectional_ab",
            "single",
            n_a=1536,
            n_b=768,
            m=64,
            self_join=False,
        ),
        _workload(
            "full-bidirectional-double",
            "bidirectional_ab",
            "double",
            n_a=768,
            n_b=1536,
            m=64,
            self_join=False,
        ),
    )


def autotune_plan(mode: AutotuneMode = "quick") -> AutotunePlan:
    """Return the deterministic work plan without running any benchmark."""

    if mode == "quick":
        plan = AutotunePlan(mode, _quick_workloads(), warmups=1, trials=3)
    elif mode == "full":
        plan = AutotunePlan(mode, _full_workloads(), warmups=2, trials=5)
    else:
        raise ValueError("mode must be 'quick' or 'full'")

    keys = [workload.key for workload in plan.workloads]
    if len(keys) != len(set(keys)):
        raise RuntimeError("autotune plan contains duplicate workload keys")
    return plan


def _describe_strategy(strategy: cache.Strategy) -> StrategyDescription:
    parameters = dict(strategy.parameters)
    row_cap = parameters.get("portable_row_cap", 256)
    if strategy.route == "cpu":
        backend: Literal["cpu", "portable_metal", "custom_metal"] = "cpu"
        backend_rank = 0
        label = "MLX CPU"
    elif strategy.route == "portable_metal":
        backend = "portable_metal"
        backend_rank = 1
        label = "portable MLX Metal"
    else:
        backend = "custom_metal"
        backend_rank = 2
        label = f"custom {strategy.route} kernel"
    parameter_text = (
        f", portable_row_cap={row_cap}" if strategy.parameters else ""
    )
    return StrategyDescription(
        strategy=strategy,
        backend=backend,
        resource_rank=(row_cap, backend_rank),
        summary=f"{strategy.profile}: {label}{parameter_text}",
    )


def strategy_descriptions() -> tuple[StrategyDescription, ...]:
    """Describe every versioned cache strategy with typed metadata."""

    return tuple(_describe_strategy(strategy) for strategy in cache.STRATEGIES)


def _snapshot_array(value: Any) -> np.ndarray:
    return np.array(value, copy=True)


def _snapshot_result(profile: str, value: Any) -> Any:
    if profile in {"1nn_value", "sum_thresh", "matrix_summary"}:
        return _snapshot_array(value)
    if profile == "1nn_index":
        profile_values, indices = value
        return _snapshot_array(profile_values), _snapshot_array(indices)
    if profile == "knn":
        return tuple(
            (operator.index(row), operator.index(column), float(score))
            for row, column, score in value
        )
    if profile == "bidirectional_ab":
        (values_a, indices_a), (values_b, indices_b) = value
        return (
            (_snapshot_array(values_a), _snapshot_array(indices_a)),
            (_snapshot_array(values_b), _snapshot_array(indices_b)),
        )
    raise ValueError(f"unknown autotune result family {profile!r}")


def _assert_array_equivalent(
    candidate: Any,
    reference: Any,
    *,
    precision: str,
    exact: bool = False,
) -> None:
    candidate_array = np.asarray(candidate)
    reference_array = np.asarray(reference)
    if candidate_array.shape != reference_array.shape:
        raise AssertionError(
            f"shape {candidate_array.shape} != {reference_array.shape}"
        )
    if exact:
        np.testing.assert_array_equal(candidate_array, reference_array)
        return
    tolerance = 5e-4 if precision == "single" else 1e-9
    np.testing.assert_allclose(
        candidate_array,
        reference_array,
        rtol=tolerance,
        atol=tolerance,
        equal_nan=True,
    )


def _assert_result_equivalent(
    profile: str,
    precision: str,
    candidate: Any,
    reference: Any,
) -> None:
    """Compare candidate results using the semantics of each profile family."""

    try:
        candidate = _snapshot_result(profile, candidate)
    except (TypeError, ValueError, OverflowError) as exc:
        raise AssertionError("result has the wrong structure") from exc

    if profile in {"1nn_value", "sum_thresh", "matrix_summary"}:
        _assert_array_equivalent(candidate, reference, precision=precision)
        return
    if profile == "1nn_index":
        _assert_array_equivalent(candidate[0], reference[0], precision=precision)
        _assert_array_equivalent(
            candidate[1], reference[1], precision=precision, exact=True
        )
        return
    if profile == "knn":
        if len(candidate) != len(reference):
            raise AssertionError(
                f"KNN row count {len(candidate)} != {len(reference)}"
            )
        _assert_array_equivalent(
            [(row, column) for row, column, _ in candidate],
            [(row, column) for row, column, _ in reference],
            precision=precision,
            exact=True,
        )
        _assert_array_equivalent(
            [score for _, _, score in candidate],
            [score for _, _, score in reference],
            precision=precision,
        )
        return
    if profile == "bidirectional_ab":
        for candidate_pair, reference_pair in zip(candidate, reference, strict=True):
            _assert_array_equivalent(
                candidate_pair[0], reference_pair[0], precision=precision
            )
            _assert_array_equivalent(
                candidate_pair[1],
                reference_pair[1],
                precision=precision,
                exact=True,
            )
        return
    raise ValueError(f"unknown autotune result family {profile!r}")


# The explicit runner is defined below the pure plan/comparison layer so this
# module remains importable even before core grows its lazy executor hook.


def _validate_devices(devices: Sequence[int] | None) -> tuple[int, ...]:
    if devices is None:
        return (0,)
    if isinstance(devices, (str, bytes)) or not isinstance(devices, Sequence):
        raise TypeError("devices must be a sequence of integer device IDs or None")

    targets: list[int] = []
    for device in devices:
        try:
            device_id = operator.index(device)
        except TypeError as exc:
            raise TypeError(
                "devices must contain only integer device IDs"
            ) from exc
        if device_id != 0:
            raise ValueError(
                f"Unsupported GPU device ID {device_id!r}; "
                "MLX/Metal exposes only GPU device 0"
            )
        if device_id not in targets:
            targets.append(device_id)
    return tuple(targets or (0,))


def _eligible_strategies(
    workload: AutotuneWorkload,
) -> tuple[cache.Strategy, ...]:
    eligible: list[cache.Strategy] = []
    for strategy in cache.STRATEGIES:
        if strategy.profile != workload.profile:
            continue
        if workload.precision != "single" and strategy.route != "cpu":
            continue
        if workload.route == "cpu" and strategy.route != "cpu":
            continue
        if workload.route == "metal" and strategy.route == "cpu":
            continue
        eligible.append(strategy)
    return tuple(eligible)


def _default_executor(
    workload: AutotuneWorkload, strategy: cache.Strategy
) -> Any:
    """Import the core benchmark hook only after the user starts tuning."""

    try:
        from .core import _autotune_execute_candidate
    except ImportError as exc:  # pragma: no cover - transitional integration
        raise RuntimeError(
            "The installed core does not provide the autotune executor hook"
        ) from exc
    return _autotune_execute_candidate(workload, strategy)


def _measure_candidate(
    workload: AutotuneWorkload,
    strategy: cache.Strategy,
    reference: Any,
    *,
    executor: CandidateExecutor,
    synchronize: Synchronize,
    clock: Clock,
    warmups: int,
    trials: int,
) -> CandidateMeasurement:
    for warmup in range(warmups):
        warm_result = executor(workload, strategy)
        synchronize()
        _assert_result_equivalent(
            workload.profile,
            workload.precision,
            warm_result,
            reference,
        )
        print(f"    warmup {warmup + 1}/{warmups}: correct")

    samples: list[int] = []
    for trial in range(trials):
        synchronize()
        start = clock()
        result = executor(workload, strategy)
        synchronize()
        elapsed = clock() - start
        if elapsed <= 0:
            raise RuntimeError("autotune clock returned a non-positive duration")
        _assert_result_equivalent(
            workload.profile,
            workload.precision,
            result,
            reference,
        )
        samples.append(elapsed)
        print(f"    trial {trial + 1}/{trials}: {elapsed} ns, correct")

    description = _describe_strategy(strategy)
    return CandidateMeasurement(
        strategy=strategy,
        samples_ns=tuple(samples),
        duration_ns=int(statistics.median(samples)),
        resource_rank=description.resource_rank,
    )


def _validate_plan(plan: AutotunePlan) -> None:
    if plan.mode not in {"quick", "full"}:
        raise ValueError("plan mode must be 'quick' or 'full'")
    if type(plan.warmups) is not int or plan.warmups < 0:
        raise ValueError("plan warmups must be a non-negative integer")
    if type(plan.trials) is not int or plan.trials <= 0:
        raise ValueError("plan trials must be a positive integer")
    if not plan.workloads:
        raise ValueError("autotune plan must contain at least one workload")
    keys = [workload.key for workload in plan.workloads]
    if len(keys) != len(set(keys)):
        raise ValueError("autotune plan contains duplicate workload keys")


def run_autotune(
    devices: Sequence[int] | None = None,
    cache_path: str = "",
    *,
    mode: AutotuneMode = "quick",
    executor: CandidateExecutor | None = None,
    synchronize: Synchronize | None = None,
    clock: Clock | None = None,
    record_writer: RecordWriter | None = None,
    plan: AutotunePlan | None = None,
) -> int:
    """Run an explicit MLX tuning plan and cache one winner per workload.

    The injectable executor, synchronizer, clock, and writer keep selection
    logic deterministic and testable without importing the core runtime.
    """

    targets = _validate_devices(devices)
    if not isinstance(cache_path, str):
        raise TypeError("cache_path must be a string")
    selected_plan = autotune_plan(mode) if plan is None else plan
    _validate_plan(selected_plan)

    if executor is None:
        try:
            metal_available = bool(mx.metal.is_available())
        except Exception:
            metal_available = False
        if not metal_available:
            raise ValueError(
                "No Metal device available; pyscamp.autotune() needs Apple Metal"
            )
        executor = _default_executor
    synchronize = mx.synchronize if synchronize is None else synchronize
    clock = time.perf_counter_ns if clock is None else clock
    writer = cache.save_record if record_writer is None else record_writer

    print(
        "MLX SCAMP autotune: "
        f"plan={selected_plan.mode}, devices={list(targets)}, "
        f"workloads={len(selected_plan.workloads)}, "
        f"warmups={selected_plan.warmups}, trials={selected_plan.trials}"
    )
    for workload_index, workload in enumerate(selected_plan.workloads, start=1):
        strategies = _eligible_strategies(workload)
        if not strategies:
            raise RuntimeError(
                f"No eligible autotune candidates for {workload.name}"
            )
        print(
            f"  plan {workload_index}/{len(selected_plan.workloads)}: "
            f"{workload.name} ({len(strategies)} candidates)"
        )

        reference_strategy = min(
            (strategy for strategy in strategies if strategy.route == "cpu"),
            key=lambda strategy: (
                _describe_strategy(strategy).resource_rank,
                strategy.name,
            ),
            default=None,
        )
        if reference_strategy is None:
            raise RuntimeError(
                f"No CPU correctness reference for {workload.name}"
            )
        try:
            reference = executor(workload, reference_strategy)
            synchronize()
            reference = _snapshot_result(workload.profile, reference)
        except Exception as exc:
            raise RuntimeError(
                f"Unable to compute reference for {workload.name}"
            ) from exc

        measurements: list[CandidateMeasurement] = []
        for candidate_index, strategy in enumerate(strategies, start=1):
            print(
                f"    candidate {candidate_index}/{len(strategies)}: "
                f"{_describe_strategy(strategy).summary}"
            )
            try:
                measurement = _measure_candidate(
                    workload,
                    strategy,
                    reference,
                    executor=executor,
                    synchronize=synchronize,
                    clock=clock,
                    warmups=selected_plan.warmups,
                    trials=selected_plan.trials,
                )
            except (AssertionError, RuntimeError, TypeError, ValueError) as exc:
                print(f"      rejected: {exc}")
                continue
            measurements.append(measurement)
            print(f"      median: {measurement.duration_ns} ns")

        if not measurements:
            raise RuntimeError(
                f"Every autotune candidate was rejected for {workload.name}"
            )
        winner = min(
            measurements,
            key=lambda measurement: (
                measurement.duration_ns,
                measurement.resource_rank,
                measurement.strategy.name,
            ),
        )
        record = cache.new_record(
            workload.key,
            winner.strategy.name,
            winner.duration_ns,
            selected_plan.trials,
        )
        writer(record, cache_path)
        print(
            f"    selected: {winner.strategy.name} "
            f"({winner.duration_ns} ns median)"
        )

    print(f"  cache={cache.sidecar_path(cache_path)}")
    return len(targets)


def autotune(devices: Sequence[int] | None = None, cache_path: str = "") -> int:
    """Run the laptop-safe explicit plan with upstream's API signature."""

    return run_autotune(devices, cache_path, mode="quick")
