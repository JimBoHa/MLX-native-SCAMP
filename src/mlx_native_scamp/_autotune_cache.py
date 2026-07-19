from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import tempfile
import threading
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from functools import lru_cache
from itertools import islice
from pathlib import Path
from typing import Any

import mlx.core as mx

try:
    import fcntl
except ImportError:  # pragma: no cover - macOS and Linux provide fcntl
    fcntl = None


CACHE_FORMAT = "MLX_SCAMP_AUTOTUNE_V2"
SCHEMA_VERSION = 2
CORE_ALGORITHM_REVISION = "tiled-v2-metal-profiles-v1"
CANDIDATE_MANIFEST_VERSION = "2026-07-19-1"
MAX_CACHE_BYTES = 4 * 1024 * 1024
MAX_RECORDS = 4096
MAX_ENVIRONMENTS = 32
MAX_DURATION_NS = 24 * 60 * 60 * 1_000_000_000
PORTABLE_ROW_CAPS = (64, 128, 256)
PROFILE_FAMILIES = frozenset(
    {
        "1nn_index",
        "1nn_value",
        "sum_thresh",
        "matrix_summary",
        "knn",
        "bidirectional_ab",
    }
)
PRECISIONS = frozenset({"single", "double", "ultra"})
ROUTE_POLICIES = frozenset({"auto", "cpu", "metal"})


@dataclass(frozen=True, slots=True)
class Strategy:
    name: str
    profile: str
    route: str
    parameters: tuple[tuple[str, int], ...] = ()


def _strategies() -> tuple[Strategy, ...]:
    rows: list[Strategy] = []
    for profile in sorted(PROFILE_FAMILIES):
        for route in ("cpu", "portable_metal"):
            for row_cap in PORTABLE_ROW_CAPS:
                rows.append(
                    Strategy(
                        f"{profile}:{route}:rows-{row_cap}",
                        profile,
                        route,
                        (("portable_row_cap", row_cap),),
                    )
                )
    rows.extend(
        (
            Strategy("1nn_index:metal-diagonal", "1nn_index", "metal_1nn"),
            Strategy("1nn_value:metal-diagonal", "1nn_value", "metal_1nn"),
            Strategy(
                "bidirectional_ab:metal-diagonal",
                "bidirectional_ab",
                "metal_bidirectional",
            ),
            Strategy("sum_thresh:metal-sparse", "sum_thresh", "metal_sum"),
            Strategy(
                "matrix_summary:metal-diagonal",
                "matrix_summary",
                "metal_matrix",
            ),
        )
    )
    return tuple(rows)


STRATEGIES = _strategies()
STRATEGY_BY_NAME = {strategy.name: strategy for strategy in STRATEGIES}


def candidate_manifest_id() -> str:
    manifest = {
        "version": CANDIDATE_MANIFEST_VERSION,
        "strategies": [asdict(strategy) for strategy in STRATEGIES],
    }
    return _digest(manifest)


@dataclass(frozen=True, slots=True)
class WorkloadKey:
    profile: str
    precision: str
    route: str
    join: str
    work_bucket: str
    window_bucket: str
    aspect_bucket: str
    dtype_class: str
    tile_regime: str
    profile_bucket: str


@dataclass(frozen=True, slots=True)
class TuningRecord:
    key: WorkloadKey
    candidate: str
    duration_ns: int
    trials: int
    environment_id: str
    manifest_id: str
    created_ns: int

    @property
    def parameters(self) -> dict[str, int]:
        strategy = STRATEGY_BY_NAME[self.candidate]
        return dict(strategy.parameters)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _power_bucket(value: int) -> str:
    if value <= 0:
        raise ValueError("bucket values must be positive")
    return f"2^{(value - 1).bit_length()}"


def _aspect_bucket(n_a: int, n_b: int, self_join: bool) -> str:
    if self_join:
        return "self"
    larger = max(n_a, n_b)
    smaller = min(n_a, n_b)
    direction = "a" if n_a >= n_b else "b"
    exponent = max(0, math.ceil(math.log2(larger / smaller)))
    return f"{direction}:2^{exponent}"


def _density_bucket(density: float | None) -> str:
    if density is None:
        return "density:unknown"
    if not math.isfinite(density) or density < 0.0 or density > 1.0:
        raise ValueError("threshold density must be between zero and one")
    for upper in (0.01, 0.05, 0.2, 0.5, 1.0):
        if density <= upper:
            return f"density:<={upper:g}"
    raise AssertionError("unreachable density bucket")


def make_workload_key(
    profile: str,
    precision: str,
    route: str,
    n_a: int,
    n_b: int,
    m: int,
    *,
    self_join: bool,
    dtype_class: str,
    max_tile_size: int | None,
    threshold_density: float | None = None,
    k: int | None = None,
    matrix_shape: tuple[int, int] | None = None,
) -> WorkloadKey:
    if profile not in PROFILE_FAMILIES:
        raise ValueError(f"unknown autotune profile family {profile!r}")
    if precision not in PRECISIONS:
        raise ValueError(f"unknown precision {precision!r}")
    if route not in ROUTE_POLICIES:
        raise ValueError(f"unknown route policy {route!r}")
    if min(n_a, n_b, m) <= 0:
        raise ValueError("subsequence counts and window must be positive")
    if not isinstance(dtype_class, str) or not dtype_class:
        raise ValueError("dtype_class must be a non-empty string")
    if max_tile_size is not None and max_tile_size <= 0:
        raise ValueError("max_tile_size must be positive")

    if profile == "sum_thresh":
        profile_bucket = _density_bucket(threshold_density)
    elif profile == "knn":
        if k is None or k <= 0:
            raise ValueError("knn workloads require a positive k")
        profile_bucket = f"k:{_power_bucket(k)}"
    elif profile == "matrix_summary":
        if matrix_shape is None or min(matrix_shape) <= 0:
            raise ValueError("matrix workloads require a positive shape")
        rows, cols = matrix_shape
        profile_bucket = (
            f"cells:{_power_bucket(rows * cols)};"
            f"shape:{_aspect_bucket(cols, rows, False)}"
        )
    else:
        profile_bucket = "default"

    tile_regime = (
        "default"
        if max_tile_size is None
        else f"explicit:{_power_bucket(max_tile_size)}"
    )
    return WorkloadKey(
        profile=profile,
        precision=precision,
        route=route,
        join="self" if self_join else "ab",
        work_bucket=_power_bucket(n_a * n_b),
        window_bucket=_power_bucket(m),
        aspect_bucket=_aspect_bucket(n_a, n_b, self_join),
        dtype_class=dtype_class,
        tile_regime=tile_regime,
        profile_bucket=profile_bucket,
    )


def _default_upstream_cache_path() -> Path:
    override = os.environ.get("SCAMP_AUTOTUNE_CACHE")
    if override is not None:
        return Path(override)
    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache is not None:
        return Path(xdg_cache) / "scamp" / "autotune.txt"
    return Path.home() / ".cache" / "scamp" / "autotune.txt"


def sidecar_path(cache_path: str = "") -> Path:
    upstream_path = Path(cache_path) if cache_path else _default_upstream_cache_path()
    return upstream_path.with_name(f"{upstream_path.name}.mlx.json")


@lru_cache(maxsize=1)
def _mlx_version() -> str:
    try:
        return importlib.metadata.version("mlx")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _memory_bytes() -> int | None:
    try:
        pages = int(os.sysconf("SC_PHYS_PAGES"))
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, OSError, TypeError, ValueError):
        return None
    value = pages * page_size
    return value if value > 0 else None


@lru_cache(maxsize=1)
def environment_fingerprint() -> dict[str, Any]:
    try:
        device = dict(mx.device_info())
    except Exception:
        device = {}
    stable_device = {
        str(key): value
        for key, value in sorted(device.items())
        if isinstance(value, (str, int, float, bool))
    }
    return {
        "algorithm_revision": CORE_ALGORITHM_REVISION,
        "candidate_manifest": candidate_manifest_id(),
        "mlx_version": _mlx_version(),
        "macos_version": platform.mac_ver()[0] or "unknown",
        "macos_build": platform.version(),
        "machine": platform.machine() or "unknown",
        "processor": platform.processor() or "unknown",
        "memory_bytes": _memory_bytes(),
        "device": stable_device,
    }


@lru_cache(maxsize=1)
def environment_id() -> str:
    return _digest(environment_fingerprint())


def _empty_payload() -> dict[str, Any]:
    return {
        "format": CACHE_FORMAT,
        "schema": SCHEMA_VERSION,
        "environments": {},
        "records": {},
    }


def _read_payload(path: Path) -> dict[str, Any]:
    try:
        if path.stat().st_size > MAX_CACHE_BYTES:
            return _empty_payload()
        raw = path.read_bytes()
    except (FileNotFoundError, OSError):
        return _empty_payload()
    if len(raw) > MAX_CACHE_BYTES:
        return _empty_payload()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError):
        return _empty_payload()
    if not isinstance(payload, dict):
        return _empty_payload()
    if payload.get("format") != CACHE_FORMAT or payload.get("schema") != SCHEMA_VERSION:
        return _empty_payload()
    if not isinstance(payload.get("records"), dict):
        return _empty_payload()
    if not isinstance(payload.get("environments"), dict):
        return _empty_payload()
    return payload


def _workload_key_from_dict(value: Any) -> WorkloadKey | None:
    if not isinstance(value, dict):
        return None
    fields = tuple(WorkloadKey.__dataclass_fields__)
    if set(value) != set(fields):
        return None
    try:
        key = WorkloadKey(**{field: value[field] for field in fields})
    except TypeError:
        return None
    if key.profile not in PROFILE_FAMILIES:
        return None
    if key.precision not in PRECISIONS or key.route not in ROUTE_POLICIES:
        return None
    if key.join not in {"self", "ab"}:
        return None
    if any(
        not isinstance(item, str) or not item or len(item) > 128
        for item in asdict(key).values()
    ):
        return None
    return key


def _record_from_dict(value: Any) -> TuningRecord | None:
    if not isinstance(value, dict):
        return None
    try:
        key = _workload_key_from_dict(value["key"])
        candidate = value["candidate"]
        duration_ns = value["duration_ns"]
        trials = value["trials"]
        record_environment = value["environment_id"]
        manifest = value["manifest_id"]
        created_ns = value["created_ns"]
    except KeyError:
        return None
    if key is None or not isinstance(candidate, str):
        return None
    strategy = STRATEGY_BY_NAME.get(candidate)
    if strategy is None or strategy.profile != key.profile:
        return None
    if key.precision != "single" and strategy.route != "cpu":
        return None
    if key.route == "cpu" and strategy.route != "cpu":
        return None
    if key.route == "metal" and strategy.route == "cpu":
        return None
    if type(duration_ns) is not int or not 0 < duration_ns <= MAX_DURATION_NS:
        return None
    if type(trials) is not int or not 0 < trials <= 1000:
        return None
    if type(created_ns) is not int or created_ns <= 0:
        return None
    if not _is_digest(record_environment) or not _is_digest(manifest):
        return None
    return TuningRecord(
        key,
        candidate,
        duration_ns,
        trials,
        record_environment,
        manifest,
        created_ns,
    )


def _is_digest(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(character in "0123456789abcdef" for character in value)


def record_id(record: TuningRecord) -> str:
    return _digest(
        {
            "environment_id": record.environment_id,
            "manifest_id": record.manifest_id,
            "key": asdict(record.key),
        }
    )


def new_record(
    key: WorkloadKey,
    candidate: str,
    duration_ns: int,
    trials: int,
    *,
    created_ns: int | None = None,
) -> TuningRecord:
    record = TuningRecord(
        key=key,
        candidate=candidate,
        duration_ns=duration_ns,
        trials=trials,
        environment_id=environment_id(),
        manifest_id=candidate_manifest_id(),
        created_ns=time.time_ns() if created_ns is None else created_ns,
    )
    if _record_from_dict(_record_to_dict(record)) is None:
        raise ValueError("invalid autotune record")
    return record


def _record_to_dict(record: TuningRecord) -> dict[str, Any]:
    value = asdict(record)
    value["key"] = asdict(record.key)
    return value


_PROCESS_LOCK = threading.RLock()


@contextmanager
def _locked_cache(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(f"{path.name}.lock")
    with _PROCESS_LOCK:
        with lock_path.open("a+b") as lock_file:
            os.chmod(lock_path, 0o600)
            if fcntl is not None:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                if fcntl is not None:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _write_payload(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = (_canonical_json(payload) + "\n").encode("utf-8")
    if len(encoded) > MAX_CACHE_BYTES:
        raise ValueError("MLX autotune cache exceeds its size limit")
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb", closefd=True) as temp_file:
            descriptor = -1
            temp_file.write(encoded)
            temp_file.flush()
            os.fsync(temp_file.fileno())
        os.replace(temp_name, path)
        os.chmod(path, 0o600)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass


def save_record(record: TuningRecord, cache_path: str = "") -> Path:
    validated = _record_from_dict(_record_to_dict(record))
    if validated is None:
        raise ValueError("invalid autotune record")
    if record.environment_id != environment_id():
        raise ValueError("autotune record belongs to a different environment")
    if record.manifest_id != candidate_manifest_id():
        raise ValueError("autotune record uses a stale candidate manifest")
    path = sidecar_path(cache_path)
    with _locked_cache(path):
        payload = _read_payload(path)
        records = payload["records"]
        valid_records = [
            parsed
            for parsed in (
                _record_from_dict(value)
                for value in islice(records.values(), MAX_RECORDS)
            )
            if parsed is not None
        ]
        valid_records.append(validated)
        valid_records.sort(key=lambda item: item.created_ns, reverse=True)
        newest_by_key: dict[str, TuningRecord] = {}
        for item in valid_records:
            newest_by_key.setdefault(record_id(item), item)
        payload["records"] = {
            record_id(item): _record_to_dict(item)
            for item in islice(newest_by_key.values(), MAX_RECORDS)
        }
        environments = payload["environments"]
        environments[record.environment_id] = environment_fingerprint()
        if len(environments) > MAX_ENVIRONMENTS:
            keep = {item.environment_id for item in valid_records[:MAX_RECORDS]}
            payload["environments"] = {
                key: value
                for key, value in environments.items()
                if key in keep
            }
        _write_payload(path, payload)
    _load_records_once.cache_clear()
    return path


@lru_cache(maxsize=16)
def _load_records_once(
    path_string: str, current_environment: str, current_manifest: str
) -> tuple[TuningRecord, ...]:
    payload = _read_payload(Path(path_string))
    records: list[TuningRecord] = []
    for value in list(payload["records"].values())[:MAX_RECORDS]:
        record = _record_from_dict(value)
        if record is None:
            continue
        if record.environment_id != current_environment:
            continue
        if record.manifest_id != current_manifest:
            continue
        records.append(record)
    return tuple(records)


def load_records(cache_path: str = "") -> tuple[TuningRecord, ...]:
    return _load_records_once(
        str(sidecar_path(cache_path)), environment_id(), candidate_manifest_id()
    )


def lookup_record(key: WorkloadKey, cache_path: str = "") -> TuningRecord | None:
    expected = asdict(key)
    for record in load_records(cache_path):
        if asdict(record.key) == expected:
            return record
    return None


def reset_cache(cache_path: str = "") -> bool:
    path = sidecar_path(cache_path)
    removed = False
    with _locked_cache(path):
        try:
            path.unlink()
            removed = True
        except FileNotFoundError:
            pass
    _load_records_once.cache_clear()
    return removed


def cache_status(cache_path: str = "") -> dict[str, Any]:
    path = sidecar_path(cache_path)
    records = load_records(cache_path)
    return {
        "path": str(path),
        "exists": path.exists(),
        "environment_id": environment_id(),
        "manifest_id": candidate_manifest_id(),
        "records": len(records),
    }
