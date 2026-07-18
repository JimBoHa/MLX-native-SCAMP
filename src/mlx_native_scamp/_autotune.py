from __future__ import annotations

import importlib.metadata
import json
import operator
import os
import platform
import statistics
import subprocess
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np


CACHE_FORMAT = "MLX_SCAMP_AUTOTUNE_V1"
KERNEL_REVISION = "metal-diagonal-1nn-v1"
DEFAULT_INPUT_LENGTH = 4096
DEFAULT_BLOCK_ROWS = 256
DEFAULT_THREADGROUP_WIDTH = 256
BLOCK_ROW_CANDIDATES = (64, 128, 256, 512)
THREADGROUP_CANDIDATES = (32, 64, 128, 256)


@dataclass(frozen=True, slots=True)
class TuningConfig:
    """Launch choices measured for one Apple/MLX software-device tuple."""

    preferred_1nn_backend: str
    preferred_portable_backend: str
    cpu_block_rows: int
    metal_block_rows: int
    metal_threadgroup_width: int
    input_length: int
    timings_ns: dict[str, int]


BenchmarkResult = tuple[np.ndarray, np.ndarray]
Benchmark = Callable[[str, str, int, int, np.ndarray, int], BenchmarkResult]


def _default_cache_path() -> Path:
    override = os.environ.get("SCAMP_AUTOTUNE_CACHE")
    if override is not None:
        return Path(override)
    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache is not None:
        return Path(xdg_cache) / "scamp" / "autotune.txt"
    return Path.home() / ".cache" / "scamp" / "autotune.txt"


def _resolve_cache_path(cache_path: str = "") -> Path:
    return Path(cache_path) if cache_path else _default_cache_path()


@lru_cache(maxsize=1)
def _mlx_version() -> str:
    try:
        return importlib.metadata.version("mlx")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _sysctl_value(name: str) -> str | None:
    try:
        value = subprocess.check_output(
            ["/usr/sbin/sysctl", "-n", name],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return value or None


@lru_cache(maxsize=1)
def _device_information() -> dict[str, Any]:
    try:
        return dict(mx.device_info())
    except (AttributeError, RuntimeError):
        # MLX 0.23.x predates device_info(). Keep the minimum-supported
        # release keyed to the physical Mac rather than collapsing every
        # Apple Silicon generation to a generic arm64 cache record.
        return {
            "device_name": _sysctl_value("machdep.cpu.brand_string")
            or "Apple Metal",
            "architecture": _sysctl_value("hw.model") or "unknown",
            "memory_size": _sysctl_value("hw.memsize") or "unknown",
        }


@lru_cache(maxsize=1)
def _device_key() -> str:
    info = _device_information()
    return "|".join(
        (
            str(info.get("device_name", "Apple Metal")),
            str(info.get("architecture", "unknown")),
            str(info.get("memory_size", "unknown")),
            platform.machine() or "unknown",
            f"mlx-{_mlx_version()}",
            KERNEL_REVISION,
        )
    )


def variant_descriptions() -> tuple[str, ...]:
    """Return the safe launch choices swept by the MLX autotuner."""

    portable = tuple(
        f"portable MLX CPU/Metal: block_rows={rows}"
        for rows in BLOCK_ROW_CANDIDATES
    )
    diagonal = tuple(
        f"Metal diagonal 1NN: threadgroup_width={width}"
        for width in THREADGROUP_CANDIDATES
    )
    return portable + diagonal


def _read_payload(path: Path, *, strict: bool) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as cache_file:
            payload = json.load(cache_file)
    except FileNotFoundError:
        return {"format": CACHE_FORMAT, "records": {}}
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        if strict:
            raise ValueError(f"Unable to read MLX autotune cache {path}: {exc}") from exc
        return {"format": CACHE_FORMAT, "records": {}}

    if not isinstance(payload, dict) or payload.get("format") != CACHE_FORMAT:
        if strict:
            raise ValueError(
                f"Autotune cache {path} is not in {CACHE_FORMAT} format; "
                "choose a different cache_path or remove the incompatible file"
            )
        return {"format": CACHE_FORMAT, "records": {}}
    if not isinstance(payload.get("records"), dict):
        if strict:
            raise ValueError(f"Malformed MLX autotune cache {path}: records must be an object")
        return {"format": CACHE_FORMAT, "records": {}}
    return payload


def _config_from_record(record: Any) -> TuningConfig | None:
    if not isinstance(record, dict):
        return None
    try:
        config = TuningConfig(
            preferred_1nn_backend=str(record["preferred_1nn_backend"]),
            preferred_portable_backend=str(record["preferred_portable_backend"]),
            cpu_block_rows=operator.index(record["cpu_block_rows"]),
            metal_block_rows=operator.index(record["metal_block_rows"]),
            metal_threadgroup_width=operator.index(
                record["metal_threadgroup_width"]
            ),
            input_length=operator.index(record["input_length"]),
            timings_ns={
                str(name): operator.index(duration)
                for name, duration in record["timings_ns"].items()
            },
        )
    except (KeyError, TypeError, AttributeError):
        return None

    if config.preferred_1nn_backend not in {"cpu", "metal"}:
        return None
    if config.preferred_portable_backend not in {"cpu", "metal"}:
        return None
    if config.cpu_block_rows not in BLOCK_ROW_CANDIDATES:
        return None
    if config.metal_block_rows not in BLOCK_ROW_CANDIDATES:
        return None
    if config.metal_threadgroup_width not in THREADGROUP_CANDIDATES:
        return None
    if config.input_length < 256:
        return None
    if any(duration <= 0 for duration in config.timings_ns.values()):
        return None
    return config


@lru_cache(maxsize=8)
def _load_tuning_once(path_string: str, device_key: str) -> TuningConfig | None:
    payload = _read_payload(Path(path_string), strict=False)
    return _config_from_record(payload["records"].get(device_key))


def load_tuning(cache_path: str = "") -> TuningConfig | None:
    """Load a tuning record once per path/device for process-fast lookup."""

    return _load_tuning_once(str(_resolve_cache_path(cache_path)), _device_key())


def _save_tuning(config: TuningConfig, path: Path) -> None:
    payload = _read_payload(path, strict=True)
    payload["records"][_device_key()] = asdict(config)
    parent = path.parent
    if parent != Path(""):
        parent.mkdir(parents=True, exist_ok=True)

    temp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=parent if parent != Path("") else Path("."),
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temp_file:
            temp_name = temp_file.name
            json.dump(payload, temp_file, indent=2, sort_keys=True)
            temp_file.write("\n")
            temp_file.flush()
            os.fsync(temp_file.fileno())
        os.chmod(temp_name, 0o600)
        os.replace(temp_name, path)
        _load_tuning_once.cache_clear()
    finally:
        if temp_name is not None:
            try:
                Path(temp_name).unlink()
            except FileNotFoundError:
                pass


def _parse_environment_int(name: str, default: int, minimum: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value < minimum:
        raise ValueError(f"{name} must be greater than or equal to {minimum}")
    return value


def _validate_devices(devices: Sequence[int] | None) -> list[int]:
    if devices is None:
        return [0]
    if isinstance(devices, (str, bytes)) or not isinstance(devices, Sequence):
        raise TypeError("devices must be a sequence of integer device IDs or None")
    targets: list[int] = []
    for device in devices:
        try:
            device_id = operator.index(device)
        except TypeError as exc:
            raise TypeError("devices must contain only integer device IDs") from exc
        if device_id != 0:
            raise ValueError(
                f"Unsupported GPU device ID {device_id!r}; "
                "MLX/Metal exposes only GPU device 0"
            )
        targets.append(device_id)
    return targets or [0]


def _assert_equivalent(
    candidate: BenchmarkResult, reference: BenchmarkResult, label: str
) -> None:
    candidate_profile, candidate_index = candidate
    reference_profile, reference_index = reference
    if candidate_profile.shape != reference_profile.shape:
        raise RuntimeError(f"Autotune candidate {label} returned the wrong shape")
    try:
        np.testing.assert_allclose(
            candidate_profile,
            reference_profile,
            rtol=2e-4,
            atol=2e-4,
            equal_nan=True,
        )
        np.testing.assert_array_equal(candidate_index, reference_index)
    except AssertionError as exc:
        raise RuntimeError(
            f"Autotune candidate {label} failed its correctness check"
        ) from exc


def _benchmark_trials(
    benchmark: Benchmark,
    reference: BenchmarkResult,
    *,
    workload: str,
    device: str,
    block_rows: int,
    threadgroup_width: int,
    series: np.ndarray,
    window: int,
    warmups: int,
    trials: int,
) -> int:
    label = (
        f"{device}:{workload}:rows={block_rows}:"
        f"threadgroup={threadgroup_width}"
    )
    for _ in range(warmups):
        result = benchmark(
            workload, device, block_rows, threadgroup_width, series, window
        )
        _assert_equivalent(result, reference, label)

    samples: list[int] = []
    for _ in range(trials):
        start = time.perf_counter_ns()
        result = benchmark(
            workload, device, block_rows, threadgroup_width, series, window
        )
        elapsed = time.perf_counter_ns() - start
        _assert_equivalent(result, reference, label)
        samples.append(elapsed)
    return int(statistics.median(samples))


def _run_sweep(
    benchmark: Benchmark,
    *,
    input_length: int,
    warmups: int,
    trials: int,
) -> TuningConfig:
    rng = np.random.default_rng(1729)
    series = (
        rng.standard_normal(input_length).astype(np.float32)
        + np.sin(np.arange(input_length, dtype=np.float32) / 17.0)
    )
    window = min(128, max(16, input_length // 32))
    reference = benchmark(
        "portable_1nn",
        "cpu",
        DEFAULT_BLOCK_ROWS,
        DEFAULT_THREADGROUP_WIDTH,
        series,
        window,
    )

    timings: dict[str, int] = {}
    for device in ("cpu", "metal"):
        for block_rows in BLOCK_ROW_CANDIDATES:
            label = f"{device}.portable.block_rows.{block_rows}"
            timings[label] = _benchmark_trials(
                benchmark,
                reference,
                workload="portable_1nn",
                device=device,
                block_rows=block_rows,
                threadgroup_width=DEFAULT_THREADGROUP_WIDTH,
                series=series,
                window=window,
                warmups=warmups,
                trials=trials,
            )

    for threadgroup_width in THREADGROUP_CANDIDATES:
        label = f"metal.diagonal.threadgroup.{threadgroup_width}"
        timings[label] = _benchmark_trials(
            benchmark,
            reference,
            workload="metal_1nn",
            device="metal",
            block_rows=DEFAULT_BLOCK_ROWS,
            threadgroup_width=threadgroup_width,
            series=series,
            window=window,
            warmups=warmups,
            trials=trials,
        )

    cpu_rows = min(
        BLOCK_ROW_CANDIDATES,
        key=lambda rows: (timings[f"cpu.portable.block_rows.{rows}"], rows),
    )
    metal_rows = min(
        BLOCK_ROW_CANDIDATES,
        key=lambda rows: (timings[f"metal.portable.block_rows.{rows}"], rows),
    )
    metal_threadgroup = min(
        THREADGROUP_CANDIDATES,
        key=lambda width: (
            timings[f"metal.diagonal.threadgroup.{width}"],
            width,
        ),
    )
    cpu_1nn = timings[f"cpu.portable.block_rows.{cpu_rows}"]
    metal_1nn = timings[f"metal.diagonal.threadgroup.{metal_threadgroup}"]
    preferred_1nn_backend = "metal" if metal_1nn <= cpu_1nn else "cpu"
    metal_portable = timings[f"metal.portable.block_rows.{metal_rows}"]
    preferred_portable_backend = "metal" if metal_portable <= cpu_1nn else "cpu"
    return TuningConfig(
        preferred_1nn_backend=preferred_1nn_backend,
        preferred_portable_backend=preferred_portable_backend,
        cpu_block_rows=cpu_rows,
        metal_block_rows=metal_rows,
        metal_threadgroup_width=metal_threadgroup,
        input_length=input_length,
        timings_ns=timings,
    )


def autotune(devices: Sequence[int] | None = None, cache_path: str = "") -> int:
    """Benchmark MLX execution choices and persist the winners for this Mac.

    The public signature and return value match upstream ``pyscamp.autotune``.
    On Apple Silicon, MLX exposes a single Metal device (ID 0); each requested
    target benchmarks Metal against MLX CPU as well as safe launch/block sizes.
    """

    targets = _validate_devices(devices)
    if not isinstance(cache_path, str):
        raise TypeError("cache_path must be a string")
    if not mx.metal.is_available():
        raise ValueError(
            "No Metal device available; pyscamp.autotune() needs Apple Metal"
        )

    input_length = _parse_environment_int(
        "SCAMP_AUTOTUNE_INPUT_LENGTH", DEFAULT_INPUT_LENGTH, 256
    )
    warmups = _parse_environment_int("SCAMP_AUTOTUNE_WARMUP_RUNS", 1, 0)
    trials = _parse_environment_int("MLX_SCAMP_AUTOTUNE_TRIALS", 3, 1)

    # Imported lazily to keep core -> cache lookup imports acyclic.
    from .core import _autotune_benchmark_candidate

    resolved_path = _resolve_cache_path(cache_path)
    for device_id in targets:
        print(
            f"MLX SCAMP autotune: Metal device {device_id}, "
            f"input length {input_length}"
        )
        config = _run_sweep(
            _autotune_benchmark_candidate,
            input_length=input_length,
            warmups=warmups,
            trials=trials,
        )
        _save_tuning(config, resolved_path)
        print(
            "  selected "
            f"1nn_backend={config.preferred_1nn_backend}, "
            f"portable_backend={config.preferred_portable_backend}, "
            f"cpu_block_rows={config.cpu_block_rows}, "
            f"metal_block_rows={config.metal_block_rows}, "
            f"metal_threadgroup={config.metal_threadgroup_width}"
        )
    print(f"  cache={resolved_path}")
    return len(targets)
