from __future__ import annotations

import math
import operator
from contextlib import nullcontext
from dataclasses import dataclass
from numbers import Real
from typing import Any

import mlx.core as mx
import numpy as np

from ._exclusion import self_join_exclusion

SENTINEL = -2.0
FLATNESS_EPSILON = 1e-13
VALID_PRECISIONS = {"single", "double", "ultra"}
BLOCK_ROWS = 256
ROLLING_STATISTICS_BLOCK = 4096
ROLLING_STATISTICS_SAFETY_FACTOR = 262144.0
MIB = 1024 * 1024
DEFAULT_UNIFIED_MEMORY_BYTES = 2 * 1024 * MIB
MIN_SIMILARITY_TILE_BUDGET_BYTES = 8 * MIB
MAX_SIMILARITY_TILE_BUDGET_BYTES = 64 * MIB
SIMILARITY_TILE_WORKING_SET_DIVISOR = 64
NORMALIZED_WINDOW_TEMPORARY_FACTOR = 8
SIMILARITY_CELL_TEMPORARY_FACTOR = 6
MAX_IN_FLIGHT_SIMILARITY_TILES = 2
UPSTREAM_CPU_MAX_TILE_SIZE = 128000
UPSTREAM_METAL_MAX_TILE_SIZE = 512000


@dataclass(slots=True)
class PreparedSeries:
    windows: Any | None
    valid: Any | None
    subsequences: int
    recurrence_clean: Any | None = None
    recurrence_means: Any | None = None
    recurrence_inv_norm: Any | None = None
    recurrence_df: Any | None = None
    recurrence_dg: Any | None = None


@dataclass(slots=True)
class TiledSeries:
    values: Any
    subsequences: int


@dataclass(slots=True)
class SimilarityTile:
    row_start: int
    row_end: int
    col_start: int
    col_end: int
    row_indices: Any
    col_indices: Any
    values: Any


@dataclass(slots=True)
class ReducerScheduler:
    """Bound lazy similarity graphs while retaining compact reducer state."""

    pending: int = 0

    def schedule(self, *state: Any) -> None:
        _schedule_reducer_state(*state)
        self.pending += 1
        if self.pending >= MAX_IN_FLIGHT_SIMILARITY_TILES:
            self.finish()

    def finish(self) -> None:
        if self.pending:
            mx.synchronize()
            self.pending = 0


@dataclass(frozen=True, slots=True)
class _ResolvedTuningStrategy:
    strategy: Any
    sum_density: float | None = None

    @property
    def route(self) -> str:
        return self.strategy.route

    @property
    def parameters(self) -> tuple[tuple[str, int], ...]:
        return self.strategy.parameters


def _schedule_reducer_state(*state: Any) -> None:
    """Schedule compact state so MLX can release its similarity block."""
    mx.async_eval(*state)


def _device_working_set_bytes() -> int:
    """Return Apple's recommended unified-memory working set when available."""

    getters = []
    if hasattr(mx, "device_info"):
        getters.append(mx.device_info)
    metal = getattr(mx, "metal", None)
    if not getters and metal is not None and hasattr(metal, "device_info"):
        getters.append(metal.device_info)

    for getter in getters:
        try:
            info = getter()
        except Exception:
            continue
        if not isinstance(info, dict):
            continue
        for key in ("max_recommended_working_set_size", "memory_size"):
            try:
                size = int(info.get(key, 0))
            except (TypeError, ValueError):
                continue
            if size > 0:
                return size
    return DEFAULT_UNIFIED_MEMORY_BYTES


def _similarity_tile_budget_bytes() -> int:
    target = _device_working_set_bytes() // SIMILARITY_TILE_WORKING_SET_DIVISOR
    return max(
        MIN_SIMILARITY_TILE_BUDGET_BYTES,
        min(MAX_SIMILARITY_TILE_BUDGET_BYTES, target),
    )


def _dtype_itemsize(dtype: Any) -> int:
    if dtype == mx.float32:
        return np.dtype(np.float32).itemsize
    if dtype == mx.float64:
        return np.dtype(np.float64).itemsize
    raise TypeError(f"Unsupported MLX compute dtype: {dtype}")


def _default_max_tile_size() -> int:
    try:
        selected_metal = mx.default_device() == mx.gpu
    except Exception:
        selected_metal = False
    upstream_default = (
        UPSTREAM_METAL_MAX_TILE_SIZE
        if selected_metal
        else UPSTREAM_CPU_MAX_TILE_SIZE
    )
    return upstream_default


def _tile_subsequence_count(max_tile_size: int, m: int) -> int:
    """Translate SCAMP's time-series tile length to matrix dimensions."""

    return max_tile_size - m + 1


def _estimate_similarity_tile_bytes(
    row_count: int,
    column_count: int,
    m: int,
    itemsize: int,
) -> int:
    """Estimate transient MLX storage used by one portable similarity tile."""

    axis_bytes = m * itemsize * NORMALIZED_WINDOW_TEMPORARY_FACTOR + 33
    cell_bytes = itemsize * SIMILARITY_CELL_TEMPORARY_FACTOR + 2
    return (row_count + column_count) * axis_bytes + (
        row_count * column_count * cell_bytes
    )


def _portable_tile_shape(
    row_subsequences: int,
    column_subsequences: int,
    m: int,
    dtype: Any,
    max_tile_size: int,
    row_cap: int = BLOCK_ROWS,
) -> tuple[int, int]:
    """Choose bounded row/column spans within SCAMP's tile-size ceiling."""

    if row_cap <= 0 or row_cap > BLOCK_ROWS:
        raise ValueError(f"portable row cap must be between 1 and {BLOCK_ROWS}")
    max_subsequences = _tile_subsequence_count(max_tile_size, m)
    row_limit = min(row_subsequences, max_subsequences, row_cap)
    column_limit = min(column_subsequences, max_subsequences)
    if row_limit <= 0 or column_limit <= 0:
        raise ValueError("portable tile dimensions must be positive")

    budget = _similarity_tile_budget_bytes()
    itemsize = _dtype_itemsize(dtype)
    axis_bytes = m * itemsize * NORMALIZED_WINDOW_TEMPORARY_FACTOR + 33
    cell_bytes = itemsize * SIMILARITY_CELL_TEMPORARY_FACTOR + 2

    # Reserve most of the target for the column windows and the lazy GEMM
    # graph. A one-row/one-column tile is the irreducible lower bound when a
    # single very large window already exceeds the advisory byte target.
    row_window_limit = max(1, budget // max(4 * axis_bytes, 1))
    row_matrix_limit = max(1, math.isqrt(max(1, budget // (4 * cell_bytes))))
    tile_rows = min(row_limit, row_window_limit, row_matrix_limit)
    remaining = max(0, budget - tile_rows * axis_bytes)
    bytes_per_column = axis_bytes + tile_rows * cell_bytes
    tile_columns = min(column_limit, max(1, remaining // bytes_per_column))
    return tile_rows, tile_columns


def gpu_supported() -> bool:
    try:
        return bool(mx.metal.is_available())
    except Exception:
        return False


def _select_execution_stream(gpus: Any, threads: int) -> Any | None:
    if gpus is None:
        if threads > 0:
            return mx.default_stream(mx.cpu)
        return None

    try:
        gpu_ids = list(gpus)
    except TypeError as exc:
        raise ValueError("gpus must be a sequence of GPU device IDs") from exc

    if not gpu_ids:
        return mx.default_stream(mx.cpu)
    if threads > 0:
        raise ValueError(
            "Concurrent CPU and Metal execution is not supported; "
            "specify either gpus=[0] or a positive threads value"
        )
    if len(gpu_ids) > 1:
        raise ValueError(
            "MLX/Metal supports only one GPU; multi-GPU requests are not supported"
        )
    if gpu_ids[0] != 0:
        raise ValueError(
            f"Unsupported GPU device ID {gpu_ids[0]!r}; "
            "MLX/Metal exposes only GPU device 0"
        )
    return mx.default_stream(mx.gpu)


def _ensure_1d_array(values: Any, name: str, dtype: Any) -> Any:
    if isinstance(values, mx.array):
        array = values
        if array.dtype != dtype:
            array = array.astype(dtype)
    else:
        numpy_array = np.asarray(values)
        if numpy_array.ndim != 1:
            raise ValueError(f"{name} must be a 1D array")
        if dtype == mx.float32:
            numpy_dtype = np.float32
        elif dtype == mx.float64:
            numpy_dtype = np.float64
        else:
            raise TypeError(f"Unsupported MLX input dtype: {dtype}")
        contiguous = np.ascontiguousarray(numpy_array, dtype=numpy_dtype)
        array = mx.array(contiguous, dtype=dtype)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1D array")
    return array


def _index_kwarg(value: Any, name: str) -> int:
    try:
        return operator.index(value)
    except TypeError:
        raise TypeError(f"{name} must be an integer") from None


def _normalize_window_size(m: Any) -> int:
    normalized = _index_kwarg(m, "m")
    if normalized < 3:
        raise ValueError("m must be at least 3")
    return normalized


def _bool_kwarg(value: Any, name: str) -> bool:
    if value is None:
        return False
    if not isinstance(value, (Real, np.bool_)):
        raise TypeError(f"{name} must be a boolean-compatible number or None")
    return bool(value)


def _gpu_kwarg(value: Any) -> list[int]:
    if value is None:
        raise TypeError("gpus must be a sequence of integer device IDs")
    try:
        devices = iter(value)
    except TypeError:
        raise TypeError("gpus must be a sequence of integer device IDs") from None
    return [_index_kwarg(device, "GPU device ID") for device in devices]


def _positive_knn_k(value: Any) -> int:
    k = _index_kwarg(value, "k")
    if k <= 0:
        raise ValueError("k must be greater than 0")
    return k


def _parse_common_kwargs(
    kwargs: dict[str, Any],
    allow_matrix: bool = False,
    allow_threshold: bool = False,
    is_ab_join: bool = False,
) -> dict[str, Any]:
    valid_keys = {
        "verbose",
        "precision",
        "pearson",
        "gpus",
        "threads",
        "max_tile_size",
    }
    if allow_threshold:
        valid_keys.add("threshold")
    if allow_matrix:
        valid_keys.update({"mheight", "mwidth"})
    if is_ab_join:
        valid_keys.add("allow_trivial_match")
    elif "allow_trivial_match" in kwargs:
        raise ValueError(
            "allow_trivial_match is only valid for ab-joins; "
            "self-joins always exclude trivial matches."
        )

    unknown = set(kwargs) - valid_keys
    if unknown:
        raise ValueError(f"Invalid keyword argument specified unknown argument: {sorted(unknown)[0]}")

    precision = kwargs.get("precision", "double")
    if not isinstance(precision, str):
        raise TypeError("precision must be a string")
    if precision not in VALID_PRECISIONS:
        raise ValueError("Invalid precision type specified: valid options are single, double, ultra")

    threshold_value = kwargs.get("threshold", 0.0)
    if not isinstance(threshold_value, Real):
        raise TypeError("threshold must be a real number")
    threshold = float(threshold_value)
    if allow_threshold and (not np.isfinite(threshold) or threshold < -1.0 or threshold > 1.0):
        raise ValueError("Invalid threshold specified: value must be finite and between -1 and 1")

    threads = _index_kwarg(kwargs.get("threads", 0), "threads")
    if threads < 0:
        raise ValueError("Invalid number of cpu worker threads specified, must be greater than or equal to 0.")

    max_tile_size = None
    if "max_tile_size" in kwargs:
        max_tile_size = _index_kwarg(kwargs["max_tile_size"], "max_tile_size")
        if max_tile_size <= 0:
            raise ValueError(
                "Invalid max_tile_size specified: value must be greater than 0"
            )
        if max_tile_size < 1024:
            raise ValueError("max_tile_size must be at least 1024")

    params = {
        "pearson": _bool_kwarg(kwargs.get("pearson", False), "pearson"),
        "precision": precision,
        "threshold": threshold,
        "verbose": _bool_kwarg(kwargs.get("verbose", False), "verbose"),
        "threads": threads,
        "gpus": _gpu_kwarg(kwargs["gpus"]) if "gpus" in kwargs else None,
        "max_tile_size": max_tile_size,
    }
    if is_ab_join:
        params["allow_trivial_match"] = _bool_kwarg(
            kwargs.get("allow_trivial_match", True),
            "allow_trivial_match",
        )
    if allow_matrix:
        params["mheight"] = _index_kwarg(kwargs.get("mheight", 50), "mheight")
        params["mwidth"] = _index_kwarg(kwargs.get("mwidth", 50), "mwidth")
        if params["mheight"] <= 0:
            raise ValueError("Invalid matrix height specified: value must be greater than 0")
        if params["mwidth"] <= 0:
            raise ValueError("Invalid matrix width specified: value must be greater than 0")
    return params


def _window_view(x: Any, m: int) -> Any:
    n = int(x.shape[0]) - m + 1
    return mx.as_strided(x, shape=(n, m), strides=(1, 1), offset=0)


def _sliding_valid_mask(finite_mask: Any, m: int) -> Any:
    prefix = mx.cumsum(finite_mask.astype(mx.int32), axis=0)
    padded = mx.concatenate([mx.zeros((1,), dtype=mx.int32), prefix], axis=0)
    window_counts = padded[m:] - padded[:-m]
    return window_counts == m


def _prepare_series(values: Any, m: int) -> PreparedSeries:
    x = values
    finite = mx.isfinite(x)
    clean = mx.where(finite, x, mx.zeros_like(x))
    windows = _window_view(clean, m)
    # Pearson correlation is invariant to an independent positive scale for
    # each window.  Scaling per window prevents overflow without allowing one
    # extreme value elsewhere in the series to underflow ordinary windows.
    scales = mx.max(mx.abs(windows), axis=1, keepdims=True)
    safe_scales = mx.where(scales > 0.0, scales, mx.ones_like(scales))
    scaled_windows = windows / safe_scales
    means = mx.mean(scaled_windows, axis=1, keepdims=True)
    centered = scaled_windows - means
    norms_sq = mx.sum(centered * centered, axis=1)
    scaled_flatness_limit = np.sqrt(FLATNESS_EPSILON) / safe_scales[:, 0]
    valid = _sliding_valid_mask(finite, m) & (mx.sqrt(norms_sq) > scaled_flatness_limit)
    safe_norms_sq = mx.where(valid, norms_sq, mx.ones_like(norms_sq))
    inv_norm = mx.where(valid, 1.0 / mx.sqrt(safe_norms_sq), 0.0)
    normalized = centered * inv_norm[:, None]
    means = means[:, 0]
    df = mx.concatenate(
        [(clean[m:] - clean[:-m]) * 0.5, mx.zeros((1,), dtype=clean.dtype)]
    )
    dg = mx.concatenate(
        [
            (clean[m:] - means[1:]) + (clean[:-m] - means[:-1]),
            mx.zeros((1,), dtype=clean.dtype),
        ]
    )
    return PreparedSeries(
        windows=normalized,
        valid=valid,
        subsequences=int(normalized.shape[0]),
        recurrence_clean=clean,
        recurrence_means=means,
        recurrence_inv_norm=inv_norm,
        recurrence_df=df,
        recurrence_dg=dg,
    )


def _prepare_tiled_series(values: Any, m: int) -> TiledSeries:
    return TiledSeries(
        values=values,
        subsequences=int(values.shape[0]) - m + 1,
    )


def _prepare_series_tile(
    series: TiledSeries,
    start: int,
    end: int,
    m: int,
) -> PreparedSeries:
    segment = series.values[start : end + m - 1]
    prepared = _prepare_series(segment, m)
    # Materialize normalization before constructing the GEMM graph. This
    # prevents the lazy graph from retaining every normalization temporary
    # alongside the similarity-mask temporaries.
    mx.eval(prepared.windows, prepared.valid)
    return prepared


def _tile_ranges(subsequences: int, tile_subsequences: int):
    for start in range(0, subsequences, tile_subsequences):
        yield start, min(subsequences, start + tile_subsequences)


def _iterate_tiled_blocks(
    series_a: TiledSeries,
    series_b: TiledSeries,
    m: int,
    exclusion: int,
    tile_rows: int,
    tile_columns: int,
):
    for col_start, col_end in _tile_ranges(
        series_a.subsequences, tile_columns
    ):
        prepared_a = _prepare_series_tile(series_a, col_start, col_end, m)
        col_indices = mx.arange(col_start, col_end, dtype=mx.int32)
        for row_start, row_end in _tile_ranges(
            series_b.subsequences, tile_rows
        ):
            if (
                series_a is series_b
                and row_start == col_start
                and row_end == col_end
            ):
                prepared_b = prepared_a
            else:
                prepared_b = _prepare_series_tile(
                    series_b, row_start, row_end, m
                )
            row_indices = mx.arange(row_start, row_end, dtype=mx.int32)
            block = prepared_b.windows @ prepared_a.windows.T
            valid_mask = prepared_b.valid[:, None] & prepared_a.valid[None, :]
            sentinel_block = mx.full(block.shape, SENTINEL, dtype=block.dtype)
            block = mx.where(
                valid_mask,
                mx.clip(block, -1.0, 1.0),
                sentinel_block,
            )
            if exclusion > 0:
                diag_mask = (
                    mx.abs(row_indices[:, None] - col_indices[None, :])
                    < exclusion
                )
                block = mx.where(diag_mask, sentinel_block, block)
            yield SimilarityTile(
                row_start=row_start,
                row_end=row_end,
                col_start=col_start,
                col_end=col_end,
                row_indices=row_indices,
                col_indices=col_indices,
                values=block,
            )


def _rolling_mean_and_norm_sq_scalar(
    clean: np.ndarray, m: int
) -> tuple[np.ndarray, np.ndarray] | None:
    """Compute sliding statistics with SCAMP's compensated scalar recurrence.

    The compensated moving sum follows SCAMP's Ogita-style CPU precompute.  A
    remove/add variance recurrence then retains linear time and memory without
    subtracting two large rolling sums of squares.
    """

    subsequences = clean.size - m + 1
    means = np.empty((subsequences,), dtype=np.float64)
    norms_sq = np.empty((subsequences,), dtype=np.float64)

    primary_sum = clean[0]
    correction = 0.0
    for value in clean[1:m]:
        combined = primary_sum + value
        displacement = combined - primary_sum
        correction += (primary_sum - (combined - displacement)) + (
            value - displacement
        )
        primary_sum = combined

    mean = (primary_sum + correction) / m
    differences = clean[:m] - mean
    norm_sq = float(np.sum(differences * differences, dtype=np.float64))
    if not np.isfinite(mean) or not np.isfinite(norm_sq):
        return None
    means[0] = mean
    norms_sq[0] = max(norm_sq, 0.0)

    for start in range(1, subsequences):
        outgoing = clean[start - 1]
        incoming = clean[start + m - 1]
        previous_mean = mean

        combined = primary_sum - outgoing
        displacement = combined - primary_sum
        correction += (primary_sum - (combined - displacement)) - (
            outgoing + displacement
        )
        primary_sum = combined

        combined = primary_sum + incoming
        displacement = combined - primary_sum
        correction += (primary_sum - (combined - displacement)) + (
            incoming - displacement
        )
        primary_sum = combined
        mean = (primary_sum + correction) / m
        update = (incoming - outgoing) * (
            incoming - mean + outgoing - previous_mean
        )
        next_norm_sq = norm_sq + update
        if not np.isfinite(mean) or not np.isfinite(next_norm_sq):
            return None

        # Roundoff can put an exactly flat window infinitesimally below zero.
        # A materially negative value means the recurrence has lost stability;
        # let the caller use the portable normalized-window implementation.
        error_scale = max(abs(norm_sq), abs(update), 1.0)
        error_bound = (
            256.0 * m * np.finfo(np.float64).eps * error_scale
        )
        if next_norm_sq < -error_bound:
            return None
        norm_sq = max(next_norm_sq, 0.0)
        means[start] = mean
        norms_sq[start] = norm_sq

    return means, norms_sq


def _rolling_mean_and_norm_sq_vectorized(
    clean: np.ndarray, m: int
) -> tuple[np.ndarray, np.ndarray] | None:
    """Vectorize stable rolling statistics in bounded, checked blocks.

    Each block starts from a high-accuracy window sum and a directly
    centered norm. NumPy then evaluates the same remove/add mean and variance
    recurrences as the scalar implementation. Blocks span at least one window,
    so checkpoint work remains O(n) even when ``m`` is large. Conservative
    forward-error bounds reject cancellation-sensitive blocks rather than
    letting a fast approximation change the normalized profile.
    """

    subsequences = clean.size - m + 1
    means = np.empty((subsequences,), dtype=np.float64)
    norms_sq = np.empty((subsequences,), dtype=np.float64)
    epsilon = np.finfo(np.float64).eps
    block_span = max(ROLLING_STATISTICS_BLOCK, m)

    for block_start in range(0, subsequences, block_span):
        block_end = min(block_start + block_span, subsequences)
        block_size = block_end - block_start
        window = clean[block_start : block_start + m]

        try:
            primary_sum = math.fsum(window)
            absolute_sum = math.fsum(np.abs(window))
        except (OverflowError, ValueError):
            return None
        mean = primary_sum / m
        differences = window - mean
        squared_differences = differences * differences
        norm_sq = float(np.sum(squared_differences, dtype=np.float64))
        if not np.isfinite(mean) or not np.isfinite(norm_sq):
            return None

        means[block_start] = mean
        norms_sq[block_start] = max(norm_sq, 0.0)
        if block_size == 1:
            continue

        outgoing = clean[block_start : block_end - 1]
        incoming = clean[block_start + m : block_end + m - 1]
        delta = incoming - outgoing
        absolute_incoming = np.abs(incoming)
        absolute_outgoing = np.abs(outgoing)
        delta_round_error = epsilon * (
            absolute_incoming + absolute_outgoing
        )
        cumulative_delta = np.cumsum(delta, dtype=np.float64)
        block_means = means[block_start:block_end]
        block_means[1:] = (primary_sum + cumulative_delta) / m

        second_factor = (
            incoming
            - block_means[1:]
            + outgoing
            - block_means[:-1]
        )
        updates = delta * second_factor
        cumulative_update = np.cumsum(updates, dtype=np.float64)
        block_norms = norms_sq[block_start:block_end]
        block_norms[1:] = norm_sq + cumulative_update

        recurrence_arrays = (
            cumulative_delta,
            block_means,
            updates,
            block_norms,
        )
        if any(not np.all(np.isfinite(array)) for array in recurrence_arrays):
            return None

        steps_epsilon = (block_size - 1) * epsilon
        window_epsilon = m * epsilon
        if steps_epsilon >= 1.0 or window_epsilon >= 1.0:
            return None
        steps_gamma = steps_epsilon / (1.0 - steps_epsilon)
        window_gamma = window_epsilon / (1.0 - window_epsilon)

        cumulative_delta_scale = float(np.sum(np.abs(delta), dtype=np.float64))
        maximum_delta_sum = float(np.max(np.abs(cumulative_delta)))
        sum_error = (
            epsilon * absolute_sum
            + float(np.sum(delta_round_error, dtype=np.float64))
            + steps_gamma * cumulative_delta_scale
            + epsilon * (abs(primary_sum) + maximum_delta_sum)
        )
        mean_error = sum_error / m

        checkpoint_mean_error = (
            epsilon * (absolute_sum + abs(primary_sum)) / m
        )
        difference_error = (
            epsilon * (np.abs(window) + abs(mean))
            + checkpoint_mean_error
        )
        initial_norm_error = (
            (window_gamma + epsilon)
            * float(np.sum(squared_differences, dtype=np.float64))
            + 2.0
            * float(
                np.sum(
                    np.abs(differences) * difference_error,
                    dtype=np.float64,
                )
            )
            + float(np.sum(difference_error * difference_error, dtype=np.float64))
        )
        second_factor_error = (
            2.0 * mean_error
            + 4.0
            * epsilon
            * (
                absolute_incoming
                + absolute_outgoing
                + np.abs(block_means[1:])
                + np.abs(block_means[:-1])
            )
        )
        update_error = (
            np.abs(delta) * second_factor_error
            + delta_round_error * np.abs(second_factor)
            + 2.0 * epsilon * np.abs(updates)
        )
        norm_error = (
            initial_norm_error
            + float(np.sum(update_error, dtype=np.float64))
            + steps_gamma * float(np.sum(np.abs(updates), dtype=np.float64))
            + epsilon
            * (abs(norm_sq) + float(np.max(np.abs(cumulative_update))))
        )
        if not np.isfinite(mean_error) or not np.isfinite(norm_error):
            return None

        # The scalar path clamps tiny negative recurrence noise after every
        # update. Re-run it when vectorized accumulation could cross zero, or
        # when its forward error exceeds the accepted ~3.8e-6 fraction of the
        # window's norm or centering scale.
        if np.any(block_norms < 0.0):
            return None
        positive_norms = block_norms[block_norms > 0.0]
        if positive_norms.size:
            minimum_norm = float(np.min(positive_norms))
            minimum_scale = math.sqrt(minimum_norm / m)
            if (
                ROLLING_STATISTICS_SAFETY_FACTOR * norm_error >= minimum_norm
                or ROLLING_STATISTICS_SAFETY_FACTOR * mean_error
                >= minimum_scale
                or np.any(
                    np.abs(block_norms - FLATNESS_EPSILON)
                    <= ROLLING_STATISTICS_SAFETY_FACTOR * norm_error
                )
            ):
                return None
        if np.any(block_norms == 0.0):
            covered = clean[block_start : block_end + m - 1]
            if not np.all(covered == covered[0]):
                return None

    return means, norms_sq


def _rolling_mean_and_norm_sq(
    clean: np.ndarray, m: int
) -> tuple[np.ndarray, np.ndarray] | None:
    """Use vectorized rolling statistics with an exact scalar fallback."""

    statistics = _rolling_mean_and_norm_sq_vectorized(clean, m)
    if statistics is not None:
        return statistics
    return _rolling_mean_and_norm_sq_scalar(clean, m)


def _prepare_metal_recurrence(values: Any, m: int) -> PreparedSeries | None:
    """Prepare the arrays consumed by the float32 diagonal Metal kernel.

    Input values have already been quantized to float32.  Converting those
    values to float64 *before* subtracting a finite per-series origin preserves
    their remaining variation at large offsets.  Statistics are then computed
    on the CPU in float64 and transferred as five linear-sized float32 arrays.
    """

    source = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(source)
    clean = np.zeros(source.shape, dtype=np.float64)
    if np.any(finite):
        finite_values = source[finite].astype(np.float64)
        clean[finite] = finite_values - finite_values[0]
    if not np.all(np.isfinite(clean)):
        return None

    statistics = _rolling_mean_and_norm_sq(clean, m)
    if statistics is None:
        return None
    means, norms_sq = statistics

    invalid_prefix = np.concatenate(
        [np.zeros((1,), dtype=np.int64), np.cumsum(~finite, dtype=np.int64)]
    )
    if m == 1:
        nonflat = np.zeros(norms_sq.shape, dtype=bool)
    else:
        changes = clean[1:] != clean[:-1]
        change_prefix = np.concatenate(
            [
                np.zeros((1,), dtype=np.int64),
                np.cumsum(changes, dtype=np.int64),
            ]
        )
        nonflat = change_prefix[m - 1 :] - change_prefix[: -(m - 1)] > 0
    # Centering translates values but never scales them, so SCAMP's flatness
    # threshold remains expressed in the original input units.
    valid = (invalid_prefix[m:] - invalid_prefix[:-m] == 0) & (
        norms_sq > FLATNESS_EPSILON
    ) & nonflat
    inv_norm = np.zeros(norms_sq.shape, dtype=np.float64)
    inv_norm[valid] = 1.0 / np.sqrt(norms_sq[valid])

    df = np.zeros(means.shape, dtype=np.float64)
    dg = np.zeros(means.shape, dtype=np.float64)
    if means.size > 1:
        df[:-1] = (clean[m:] - clean[:-m]) * 0.5
        dg[:-1] = (clean[m:] - means[1:]) + (clean[:-m] - means[:-1])

    recurrence = (clean, means, inv_norm, df, dg)
    float32_limit = float(np.finfo(np.float32).max)
    if any(
        not np.all(np.isfinite(array))
        or (array.size and float(np.max(np.abs(array))) > float32_limit)
        for array in recurrence
    ):
        return None

    clean32, means32, inv_norm32, df32, dg32 = (
        mx.array(array.astype(np.float32, copy=False), dtype=mx.float32)
        for array in recurrence
    )
    return PreparedSeries(
        windows=None,
        valid=None,
        subsequences=int(means.size),
        recurrence_clean=clean32,
        recurrence_means=means32,
        recurrence_inv_norm=inv_norm32,
        recurrence_df=df32,
        recurrence_dg=dg32,
    )


def _is_float32_input(values: Any) -> bool:
    if isinstance(values, mx.array):
        return values.dtype == mx.float32
    try:
        return np.asarray(values).dtype == np.dtype(np.float32)
    except (TypeError, ValueError):
        return False


def _metal_recurrence_is_safe(
    prepared_a: PreparedSeries,
    prepared_b: PreparedSeries,
    m: int,
) -> bool:
    recurrence_arrays = (
        prepared_a.recurrence_clean,
        prepared_a.recurrence_means,
        prepared_a.recurrence_inv_norm,
        prepared_a.recurrence_df,
        prepared_a.recurrence_dg,
        prepared_b.recurrence_clean,
        prepared_b.recurrence_means,
        prepared_b.recurrence_inv_norm,
        prepared_b.recurrence_df,
        prepared_b.recurrence_dg,
    )
    if any(value is None or value.dtype != mx.float32 for value in recurrence_arrays):
        return False

    max_a = float(np.asarray(mx.max(mx.abs(prepared_a.recurrence_clean))))
    max_b = float(np.asarray(mx.max(mx.abs(prepared_b.recurrence_clean))))
    longest_diagonal = max(prepared_a.subsequences, prepared_b.subsequences)
    conservative_bound = (
        8.0 * (m + 2 * longest_diagonal) * max_a * max_b
    )
    return np.isfinite(conservative_bound) and (
        conservative_bound <= float(np.finfo(np.float32).max)
    )


def _topk_corr_then_smallest_index(
    values: Any,
    indices: Any,
    k: int,
    *,
    indices_are_sorted: bool = False,
) -> tuple[Any, Any]:
    """Select correlations descending with the smallest row breaking ties."""

    rows = int(values.shape[0])
    k = max(1, min(int(k), rows))
    if not indices_are_sorted:
        index_order = mx.argsort(indices, axis=0)
        values = mx.take_along_axis(values, index_order, axis=0)
        indices = mx.take_along_axis(indices, index_order, axis=0)

    # MLX argsort is stable. Sorting rows first and then sorting negative
    # correlations preserves ascending row order within every equal-value run.
    correlation_order = mx.argsort(-values, axis=0)
    top_order = mx.take(correlation_order, mx.arange(k), axis=0)
    return (
        mx.take_along_axis(values, top_order, axis=0),
        mx.take_along_axis(indices, top_order, axis=0),
    )


def _convert_profile_output(corr: np.ndarray, m: int, pearson: bool) -> np.ndarray:
    if pearson:
        out = corr.astype(np.float32)
        out[out < -1.0] = np.nan
        return out
    out = np.sqrt(np.maximum(2.0 * m * (1.0 - corr), 0.0)).astype(np.float32)
    out[corr < -1.0] = np.nan
    return out


def _convert_match_value(corr: float, m: int, pearson: bool) -> float:
    if corr < -1.0:
        return float("nan")
    if pearson:
        return float(corr)
    return float(np.sqrt(max(2.0 * m * (1.0 - corr), 0.0)))


def _best_match_profile(
    prepared_a: PreparedSeries | TiledSeries,
    prepared_b: PreparedSeries | TiledSeries,
    m: int,
    pearson: bool,
    self_join: bool,
    exclusion: int,
    use_metal_kernel: bool,
    tile_rows: int | None = None,
    tile_columns: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if use_metal_kernel:
        from ._metal_1nn import best_match

        corr_np, idx_np = best_match(
            prepared_a, prepared_b, m, self_join, exclusion
        )
        return _convert_profile_output(corr_np, m, pearson), idx_np

    if not isinstance(prepared_a, TiledSeries) or not isinstance(
        prepared_b, TiledSeries
    ):
        raise TypeError("portable reducers require tiled series inputs")
    if tile_rows is None or tile_columns is None:
        raise TypeError("portable reducers require explicit tile dimensions")

    column_ranges = list(
        _tile_ranges(prepared_a.subsequences, tile_columns)
    )
    best_corr = {
        start: mx.full(
            (end - start,), SENTINEL, dtype=prepared_a.values.dtype
        )
        for start, end in column_ranges
    }
    best_idx = {
        start: mx.full((end - start,), -1, dtype=mx.int32)
        for start, end in column_ranges
    }
    scheduler = ReducerScheduler()
    for tile in _iterate_tiled_blocks(
        prepared_a,
        prepared_b,
        m,
        exclusion,
        tile_rows,
        tile_columns,
    ):
        block_best_corr = mx.max(tile.values, axis=0)
        block_best_idx = mx.argmax(tile.values, axis=0) + tile.row_start
        current_corr = best_corr[tile.col_start]
        current_idx = best_idx[tile.col_start]
        update = block_best_corr > current_corr
        best_corr[tile.col_start] = mx.where(
            update, block_best_corr, current_corr
        )
        best_idx[tile.col_start] = mx.where(
            update, block_best_idx, current_idx
        )
        scheduler.schedule(
            best_corr[tile.col_start], best_idx[tile.col_start]
        )
    scheduler.finish()
    corr_np = np.concatenate(
        [
            np.asarray(best_corr[start]).astype(np.float32)
            for start, _ in column_ranges
        ]
    )
    idx_np = np.concatenate(
        [
            np.asarray(best_idx[start]).astype(np.int32)
            for start, _ in column_ranges
        ]
    )
    idx_np[corr_np < -1.0] = -1
    return _convert_profile_output(corr_np, m, pearson), idx_np


def _bidirectional_best_match_profile(
    prepared_a: PreparedSeries | TiledSeries,
    prepared_b: PreparedSeries | TiledSeries,
    m: int,
    pearson: bool,
    exclusion: int,
    use_metal_kernel: bool,
    tile_rows: int | None = None,
    tile_columns: int | None = None,
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Reduce both axes of one AB-join distance matrix."""

    if use_metal_kernel:
        from ._metal_1nn import bidirectional_best_match

        corr_a, idx_a, corr_b, idx_b = bidirectional_best_match(
            prepared_a, prepared_b, m, exclusion
        )
    else:
        if not isinstance(prepared_a, TiledSeries) or not isinstance(
            prepared_b, TiledSeries
        ):
            raise TypeError("portable reducers require tiled series inputs")
        if tile_rows is None or tile_columns is None:
            raise TypeError(
                "portable reducers require explicit tile dimensions"
            )

        column_ranges = list(
            _tile_ranges(prepared_a.subsequences, tile_columns)
        )
        row_ranges = list(_tile_ranges(prepared_b.subsequences, tile_rows))
        best_corr_a = {
            start: mx.full(
                (end - start,), SENTINEL, dtype=prepared_a.values.dtype
            )
            for start, end in column_ranges
        }
        best_idx_a = {
            start: mx.full((end - start,), -1, dtype=mx.int32)
            for start, end in column_ranges
        }
        best_corr_b = {
            start: mx.full(
                (end - start,), SENTINEL, dtype=prepared_b.values.dtype
            )
            for start, end in row_ranges
        }
        best_idx_b = {
            start: mx.full((end - start,), -1, dtype=mx.int32)
            for start, end in row_ranges
        }
        scheduler = ReducerScheduler()

        # One traversal updates compact state for both global matrix axes.
        # Tiles arrive with columns outermost and both axes in ascending order;
        # argmax chooses the first local match and strict updates retain the
        # smallest global index when equal maxima occur in later tiles.
        for tile in _iterate_tiled_blocks(
            prepared_a,
            prepared_b,
            m,
            exclusion,
            tile_rows,
            tile_columns,
        ):
            block_best_corr_a = mx.max(tile.values, axis=0)
            block_best_idx_a = (
                mx.argmax(tile.values, axis=0) + tile.row_start
            )
            current_corr_a = best_corr_a[tile.col_start]
            current_idx_a = best_idx_a[tile.col_start]
            update_a = block_best_corr_a > current_corr_a
            best_corr_a[tile.col_start] = mx.where(
                update_a, block_best_corr_a, current_corr_a
            )
            best_idx_a[tile.col_start] = mx.where(
                update_a, block_best_idx_a, current_idx_a
            )

            block_best_corr_b = mx.max(tile.values, axis=1)
            block_best_idx_b = (
                mx.argmax(tile.values, axis=1) + tile.col_start
            )
            current_corr_b = best_corr_b[tile.row_start]
            current_idx_b = best_idx_b[tile.row_start]
            update_b = block_best_corr_b > current_corr_b
            best_corr_b[tile.row_start] = mx.where(
                update_b, block_best_corr_b, current_corr_b
            )
            best_idx_b[tile.row_start] = mx.where(
                update_b, block_best_idx_b, current_idx_b
            )
            scheduler.schedule(
                best_corr_a[tile.col_start],
                best_idx_a[tile.col_start],
                best_corr_b[tile.row_start],
                best_idx_b[tile.row_start],
            )

        scheduler.finish()
        corr_a = np.concatenate(
            [
                np.asarray(best_corr_a[start]).astype(np.float32)
                for start, _ in column_ranges
            ]
        )
        idx_a = np.concatenate(
            [
                np.asarray(best_idx_a[start]).astype(np.int32)
                for start, _ in column_ranges
            ]
        )
        corr_b = np.concatenate(
            [
                np.asarray(best_corr_b[start]).astype(np.float32)
                for start, _ in row_ranges
            ]
        )
        idx_b = np.concatenate(
            [
                np.asarray(best_idx_b[start]).astype(np.int32)
                for start, _ in row_ranges
            ]
        )
        idx_a[corr_a < -1.0] = -1
        idx_b[corr_b < -1.0] = -1

    return (
        (_convert_profile_output(corr_a, m, pearson), idx_a),
        (_convert_profile_output(corr_b, m, pearson), idx_b),
    )


def _best_match_values(
    prepared_a: PreparedSeries | TiledSeries,
    prepared_b: PreparedSeries | TiledSeries,
    m: int,
    pearson: bool,
    self_join: bool,
    exclusion: int,
    use_metal_kernel: bool,
    tile_rows: int | None = None,
    tile_columns: int | None = None,
) -> np.ndarray:
    if use_metal_kernel:
        from ._metal_1nn import best_profile

        corr_np = best_profile(
            prepared_a, prepared_b, m, self_join, exclusion
        )
        return _convert_profile_output(corr_np, m, pearson)

    if not isinstance(prepared_a, TiledSeries) or not isinstance(
        prepared_b, TiledSeries
    ):
        raise TypeError("portable reducers require tiled series inputs")
    if tile_rows is None or tile_columns is None:
        raise TypeError("portable reducers require explicit tile dimensions")

    column_ranges = list(
        _tile_ranges(prepared_a.subsequences, tile_columns)
    )
    best_corr = {
        start: mx.full(
            (end - start,), SENTINEL, dtype=prepared_a.values.dtype
        )
        for start, end in column_ranges
    }
    scheduler = ReducerScheduler()
    for tile in _iterate_tiled_blocks(
        prepared_a,
        prepared_b,
        m,
        exclusion,
        tile_rows,
        tile_columns,
    ):
        block_best_corr = mx.max(tile.values, axis=0)
        current_corr = best_corr[tile.col_start]
        update = block_best_corr > current_corr
        best_corr[tile.col_start] = mx.where(
            update, block_best_corr, current_corr
        )
        scheduler.schedule(best_corr[tile.col_start])
    scheduler.finish()
    corr_np = np.concatenate(
        [
            np.asarray(best_corr[start]).astype(np.float32)
            for start, _ in column_ranges
        ]
    )
    return _convert_profile_output(corr_np, m, pearson)


def _sum_threshold_profile(
    prepared_a: PreparedSeries | TiledSeries,
    prepared_b: PreparedSeries | TiledSeries,
    m: int,
    threshold: float,
    self_join: bool,
    exclusion: int,
    use_metal_kernel: bool,
    tile_rows: int | None = None,
    tile_columns: int | None = None,
) -> np.ndarray:
    if use_metal_kernel:
        from ._metal_sum import sum_threshold

        return sum_threshold(
            prepared_a, prepared_b, m, threshold, self_join, exclusion
        )

    if not isinstance(prepared_a, TiledSeries) or not isinstance(
        prepared_b, TiledSeries
    ):
        raise TypeError("portable reducers require tiled series inputs")
    if tile_rows is None or tile_columns is None:
        raise TypeError("portable reducers require explicit tile dimensions")

    column_ranges = list(
        _tile_ranges(prepared_a.subsequences, tile_columns)
    )
    accum = {
        start: mx.zeros((end - start,), dtype=prepared_a.values.dtype)
        for start, end in column_ranges
    }
    scheduler = ReducerScheduler()
    for tile in _iterate_tiled_blocks(
        prepared_a,
        prepared_b,
        m,
        exclusion,
        tile_rows,
        tile_columns,
    ):
        filtered = mx.where(
            tile.values > threshold,
            tile.values,
            mx.zeros_like(tile.values),
        )
        accum[tile.col_start] = accum[tile.col_start] + mx.sum(
            filtered, axis=0
        )
        scheduler.schedule(accum[tile.col_start])
    scheduler.finish()
    return np.concatenate(
        [
            np.asarray(accum[start], dtype=np.float64)
            for start, _ in column_ranges
        ]
    )


def _convert_matrix_summary(
    summary: np.ndarray,
    m: int,
    pearson: bool,
    threshold: float,
) -> np.ndarray:
    summary = np.asarray(summary, dtype=np.float32)
    summary[summary < -1.0] = np.nan
    summary[summary < threshold] = np.nan
    if pearson:
        return summary
    out = np.sqrt(np.maximum(2.0 * m * (1.0 - summary), 0.0)).astype(np.float32)
    out[np.isnan(summary)] = np.nan
    return out


def _matrix_bin_edges(subsequences: int, bins: int) -> np.ndarray:
    """Return SCAMP's ceil-spaced matrix-summary bin boundaries."""

    return np.fromiter(
        (
            (index * subsequences + bins - 1) // bins
            for index in range(bins + 1)
        ),
        dtype=np.int64,
        count=bins + 1,
    )


def _metal_sum_workload_is_worthwhile(
    column_subsequences: int,
    row_subsequences: int,
    m: int,
    qualifying_density: float,
    self_join: bool,
) -> bool:
    """Apply the benchmarked pair floor for a sampled SUM workload."""

    if self_join:
        active_diagonals = max(
            column_subsequences - self_join_exclusion(m), 0
        )
        comparisons = active_diagonals * (active_diagonals + 1) // 2
    else:
        shorter = min(column_subsequences, row_subsequences)
        longer = max(column_subsequences, row_subsequences)
        if shorter <= 0 or longer > 2 * shorter:
            return False
        comparisons = column_subsequences * row_subsequences

    if self_join and qualifying_density <= 0.05:
        minimum_comparisons = 8_000_000
    elif self_join and qualifying_density <= 0.18:
        minimum_comparisons = 12_000_000
    elif self_join and qualifying_density <= 0.35:
        minimum_comparisons = 40_000_000
    elif not self_join and qualifying_density <= 0.05:
        minimum_comparisons = 144_000_000
    elif not self_join and qualifying_density <= 0.18:
        minimum_comparisons = 625_000_000
    else:
        return False

    # Short windows make the portable matrix multiply cheap relative to Metal
    # dispatch and atomic overhead.  Keep those crossovers farther out.
    if m < 32:
        minimum_comparisons *= 2
    elif m < 64:
        minimum_comparisons = minimum_comparisons * 3 // 2
    return comparisons >= minimum_comparisons


def _estimate_metal_sum_density(
    values_a: Any,
    values_b: Any,
    m: int,
    threshold: float,
    self_join: bool,
) -> float | None:
    """Sample the fraction of pairs that would require a Metal atomic add."""

    sample_count = min(256, 262_144 // m)
    if sample_count < 32:
        return None

    source_a = np.asarray(values_a)
    source_b = source_a if self_join else np.asarray(values_b)
    if source_a.ndim != 1 or source_b.ndim != 1:
        return None
    n_a = source_a.size - m + 1
    n_b = source_b.size - m + 1
    if min(n_a, n_b) <= 0:
        return None
    seed = (
        n_a * 0x9E3779B1
        + n_b * 0x85EBCA6B
        + m * 0xC2B2AE35
        + int(self_join)
    ) & ((1 << 64) - 1)
    generator = np.random.default_rng(seed)

    if self_join:
        exclusion = self_join_exclusion(m)
        sampled_rows: list[int] = []
        sampled_columns: list[int] = []
        while len(sampled_rows) < sample_count:
            candidates = sample_count * 2
            first = generator.integers(0, n_a, size=candidates)
            second = generator.integers(0, n_a, size=candidates)
            rows = np.minimum(first, second)
            columns = np.maximum(first, second)
            eligible = columns - rows >= exclusion
            sampled_rows.extend(rows[eligible].tolist())
            sampled_columns.extend(columns[eligible].tolist())
        row_indices = np.asarray(sampled_rows[:sample_count])
        column_indices = np.asarray(sampled_columns[:sample_count])
    else:
        column_indices = generator.integers(0, n_a, size=sample_count)
        row_indices = generator.integers(0, n_b, size=sample_count)

    offsets = np.arange(m)
    a_positions = column_indices[:, None] + offsets
    b_positions = row_indices[:, None] + offsets
    with np.errstate(over="ignore", invalid="ignore"):
        windows_a = np.asarray(
            source_a[a_positions], dtype=np.float32
        ).astype(np.float64)
        windows_b = np.asarray(
            source_b[b_positions], dtype=np.float32
        ).astype(np.float64)
    finite_a = np.isfinite(windows_a)
    finite_b = np.isfinite(windows_b)
    valid = np.all(finite_a, axis=1) & np.all(finite_b, axis=1)
    windows_a[~finite_a] = 0.0
    windows_b[~finite_b] = 0.0
    # Per-window translation is correlation-invariant and avoids cancellation
    # without copying the complete input into a float64 sampling buffer.
    windows_a -= windows_a[:, :1]
    windows_b -= windows_b[:, :1]
    windows_a -= np.mean(windows_a, axis=1, keepdims=True)
    windows_b -= np.mean(windows_b, axis=1, keepdims=True)
    norm_a_sq = np.sum(windows_a * windows_a, axis=1)
    norm_b_sq = np.sum(windows_b * windows_b, axis=1)
    valid &= (norm_a_sq > FLATNESS_EPSILON) & (
        norm_b_sq > FLATNESS_EPSILON
    )
    correlations = np.full((sample_count,), SENTINEL, dtype=np.float64)
    correlations[valid] = np.sum(
        windows_a[valid] * windows_b[valid], axis=1
    ) / np.sqrt(norm_a_sq[valid] * norm_b_sq[valid])
    qualifying = int(np.count_nonzero(correlations > threshold))

    # Five synthetic successes add a small upper margin so sampling near a
    # bucket boundary chooses the more conservative crossover.
    return min(1.0, (qualifying + 5) / sample_count)


def _metal_sum_is_worthwhile(
    values_a: Any,
    values_b: Any,
    column_subsequences: int,
    row_subsequences: int,
    m: int,
    threshold: float,
    self_join: bool,
    qualifying_density: float | None = None,
) -> bool:
    """Estimate whether sparse Metal SUM should beat the portable reducer."""

    if not _metal_sum_workload_is_worthwhile(
        column_subsequences, row_subsequences, m, 0.0, self_join
    ):
        return False
    density = qualifying_density
    if density is None:
        density = _estimate_metal_sum_density(
            values_a, values_b, m, threshold, self_join
        )
    return density is not None and _metal_sum_workload_is_worthwhile(
        column_subsequences, row_subsequences, m, density, self_join
    )


def _matrix_summary(
    prepared_a: PreparedSeries | TiledSeries,
    prepared_b: PreparedSeries | TiledSeries,
    m: int,
    pearson: bool,
    threshold: float,
    rows: int,
    cols: int,
    self_join: bool,
    exclusion: int,
    *,
    use_metal_kernel: bool = False,
    tile_rows: int | None = None,
    tile_columns: int | None = None,
) -> np.ndarray:
    row_edges = _matrix_bin_edges(prepared_b.subsequences, rows)
    col_edges = _matrix_bin_edges(prepared_a.subsequences, cols)
    if use_metal_kernel:
        from ._metal_matrix import matrix_summary

        summary = matrix_summary(
            prepared_a,
            prepared_b,
            m,
            rows,
            cols,
            self_join,
            exclusion,
            row_edges,
            col_edges,
        )
        return _convert_matrix_summary(summary, m, pearson, threshold)

    if not isinstance(prepared_a, TiledSeries) or not isinstance(
        prepared_b, TiledSeries
    ):
        raise TypeError("portable matrix summaries require tiled series inputs")
    if tile_rows is None or tile_columns is None:
        raise TypeError(
            "portable matrix summaries require explicit tile dimensions"
        )
    col_ranges = [(int(col_edges[c]), int(col_edges[c + 1])) for c in range(cols)]
    summary = mx.full((rows, cols), SENTINEL, dtype=prepared_a.values.dtype)
    scheduler = ReducerScheduler()

    for tile in _iterate_tiled_blocks(
        prepared_a,
        prepared_b,
        m,
        exclusion,
        tile_rows,
        tile_columns,
    ):
        block = tile.values
        if self_join:
            upper_mask = tile.row_indices[:, None] <= tile.col_indices[None, :]
            block = mx.where(
                upper_mask,
                block,
                mx.full(block.shape, SENTINEL, dtype=block.dtype),
            )
        block_rows_summary: list[Any] = []
        for r in range(rows):
            rs = int(max(tile.row_start, row_edges[r]))
            re = int(min(tile.row_end, row_edges[r + 1]))
            if rs >= re:
                block_rows_summary.append(
                    mx.full((cols,), SENTINEL, dtype=summary.dtype)
                )
                continue
            row_slice = block[rs - tile.row_start : re - tile.row_start]
            block_cols_summary: list[Any] = []
            for c in range(cols):
                cs = max(tile.col_start, col_ranges[c][0])
                ce = min(tile.col_end, col_ranges[c][1])
                if cs >= ce:
                    block_cols_summary.append(
                        mx.array(SENTINEL, dtype=summary.dtype)
                    )
                    continue
                block_cols_summary.append(
                    mx.max(
                        row_slice[
                            :, cs - tile.col_start : ce - tile.col_start
                        ]
                    )
                )
            block_rows_summary.append(mx.stack(block_cols_summary, axis=0))
        summary = mx.maximum(summary, mx.stack(block_rows_summary, axis=0))
        scheduler.schedule(summary)

    scheduler.finish()
    return _convert_matrix_summary(
        np.asarray(summary, dtype=np.float32),
        m,
        pearson,
        threshold,
    )


def _knn_profile(
    prepared_a: TiledSeries,
    prepared_b: TiledSeries,
    m: int,
    k: int,
    threshold: float,
    pearson: bool,
    exclusion: int,
    tile_rows: int,
    tile_columns: int,
) -> list[tuple[int, int, float]]:
    column_ranges = list(
        _tile_ranges(prepared_a.subsequences, tile_columns)
    )
    best_corr = {
        start: mx.full(
            (k, end - start), SENTINEL, dtype=prepared_a.values.dtype
        )
        for start, end in column_ranges
    }
    best_idx = {
        start: mx.full((k, end - start), -1, dtype=mx.int32)
        for start, end in column_ranges
    }
    scheduler = ReducerScheduler()

    for tile in _iterate_tiled_blocks(
        prepared_a,
        prepared_b,
        m,
        exclusion,
        tile_rows,
        tile_columns,
    ):
        local_indices = mx.broadcast_to(
            tile.row_indices[:, None], tile.values.shape
        )
        local_corr, local_idx = _topk_corr_then_smallest_index(
            tile.values,
            local_indices,
            k,
            indices_are_sorted=True,
        )
        merged_corr = mx.concatenate(
            [best_corr[tile.col_start], local_corr], axis=0
        )
        merged_idx = mx.concatenate(
            [best_idx[tile.col_start], local_idx], axis=0
        )
        (
            best_corr[tile.col_start],
            best_idx[tile.col_start],
        ) = _topk_corr_then_smallest_index(
            merged_corr,
            merged_idx,
            k,
        )
        scheduler.schedule(
            best_corr[tile.col_start], best_idx[tile.col_start]
        )

    scheduler.finish()
    results: list[tuple[int, int, float]] = []
    for col_start, col_end in column_ranges:
        corr_np = np.asarray(best_corr[col_start]).astype(np.float32)
        idx_np = np.asarray(best_idx[col_start]).astype(np.int32)
        for local_col in range(col_end - col_start):
            seen_rows: set[int] = set()
            for rank in range(k):
                corr = corr_np[rank, local_col]
                row = int(idx_np[rank, local_col])
                if (
                    corr <= threshold
                    or corr < -1.0
                    or row < 0
                    or row in seen_rows
                ):
                    continue
                seen_rows.add(row)
                results.append(
                    (
                        col_start + local_col,
                        row,
                        _convert_match_value(corr, m, pearson),
                    )
                )
    return results


def _run_profile(
    a: Any,
    b: Any | None,
    m: int,
    *,
    pearson: bool,
    precision: str,
    threshold: float = 0.0,
    mheight: int = 50,
    mwidth: int = 50,
    allow_trivial_match: bool = True,
    profile: str,
    k: int | None = None,
    max_tile_size: int | None = None,
    use_metal_1nn: bool = False,
    use_metal_matrix: bool = False,
    use_metal_sum: bool = False,
    portable_row_cap: int = BLOCK_ROWS,
    sum_density: float | None = None,
):
    m = _normalize_window_size(m)
    effective_max_tile_size = (
        _default_max_tile_size()
        if max_tile_size is None
        else max_tile_size
    )
    if effective_max_tile_size < 2 * m:
        raise ValueError("max_tile_size must be at least twice m")
    has_b = b is not None
    if profile == "1nn_bidirectional" and not has_b:
        raise ValueError("bidirectional profiles require an AB-join")
    float32_sources = _is_float32_input(a) and (
        not has_b or _is_float32_input(b)
    )
    compute_dtype = mx.float32 if precision == "single" else mx.float64
    # Metal does not provide native float64. The execution resolver places
    # double and ultra on MLX CPU; single can stay on the selected Metal GPU.
    # Upstream's ultra mode changes its sliding recurrence, while this direct
    # normalized-window implementation uses the same float64 path for both.
    series_a = _ensure_1d_array(a, "a", compute_dtype)
    if int(series_a.shape[0]) < m:
        raise ValueError("m must be less than or equal to len(a)")

    series_b = _ensure_1d_array(b, "b", compute_dtype) if has_b else series_a
    if int(series_b.shape[0]) < m:
        raise ValueError("m must be less than or equal to len(b)")

    subsequences_a = int(series_a.shape[0]) - m + 1
    subsequences_b = int(series_b.shape[0]) - m + 1

    if profile == "matrix":
        if mwidth > subsequences_a:
            raise ValueError(
                "mwidth must be less than or equal to the number of subsequences in a"
            )
        if mheight > subsequences_b:
            height_series = "b" if has_b else "a"
            raise ValueError(
                f"mheight must be less than or equal to the number of subsequences in {height_series}"
            )

    self_join = not has_b
    join_fits_tile = (
        int(series_a.shape[0]) <= effective_max_tile_size
        and int(series_b.shape[0]) <= effective_max_tile_size
    )
    exclusion = (
        self_join_exclusion(m)
        if self_join or not allow_trivial_match
        else 0
    )

    sum_recurrence = (
        profile == "sum"
        and use_metal_sum
        and float32_sources
        and join_fits_tile
        and threshold >= 0.0
        and _metal_sum_is_worthwhile(
            a,
            a if self_join else b,
            subsequences_a,
            subsequences_b,
            m,
            threshold,
            self_join,
            sum_density,
        )
    )
    matrix_recurrence = (
        profile == "matrix"
        and use_metal_matrix
        and float32_sources
        and join_fits_tile
    )
    if matrix_recurrence:
        from ._metal_matrix import indexing_is_safe

        matrix_recurrence = indexing_is_safe(
            subsequences_a,
            subsequences_b,
            m,
            mheight,
            mwidth,
            self_join,
            exclusion,
        )
    if (
        profile in {"1nn", "1nn_value", "1nn_bidirectional"}
        and use_metal_1nn
        and float32_sources
        and join_fits_tile
    ) or sum_recurrence or matrix_recurrence:
        recurrence_a = _prepare_metal_recurrence(series_a, m)
        recurrence_b = (
            recurrence_a
            if self_join
            else _prepare_metal_recurrence(series_b, m)
        )
        if (
            recurrence_a is not None
            and recurrence_b is not None
            and _metal_recurrence_is_safe(recurrence_a, recurrence_b, m)
        ):
            if profile in {"1nn", "1nn_value"}:
                if profile == "1nn_value":
                    return _best_match_values(
                        recurrence_a,
                        recurrence_b,
                        m,
                        pearson,
                        self_join,
                        exclusion,
                        True,
                    )
                return _best_match_profile(
                    recurrence_a,
                    recurrence_b,
                    m,
                    pearson,
                    self_join,
                    exclusion,
                    True,
                )
            if profile == "1nn_bidirectional":
                return _bidirectional_best_match_profile(
                    recurrence_a,
                    recurrence_b,
                    m,
                    pearson,
                    exclusion,
                    True,
                )
            if profile == "sum":
                return _sum_threshold_profile(
                    recurrence_a,
                    recurrence_b,
                    m,
                    threshold,
                    self_join,
                    exclusion,
                    True,
                )
            return _matrix_summary(
                recurrence_a,
                recurrence_b,
                m,
                pearson,
                threshold,
                mheight,
                mwidth,
                self_join,
                exclusion,
                use_metal_kernel=True,
            )

    # Portable reducers normalize only overlapping tile segments. The full
    # (subsequences, m) tensor and full pairwise distance matrix are never
    # materialized by this path.
    tile_rows, tile_columns = _portable_tile_shape(
        subsequences_b,
        subsequences_a,
        m,
        compute_dtype,
        effective_max_tile_size,
        portable_row_cap,
    )
    prepared_a = _prepare_tiled_series(series_a, m)
    prepared_b = (
        prepared_a if self_join else _prepare_tiled_series(series_b, m)
    )

    if profile == "1nn_value":
        return _best_match_values(
            prepared_a,
            prepared_b,
            m,
            pearson,
            self_join,
            exclusion,
            False,
            tile_rows,
            tile_columns,
        )
    if profile == "1nn":
        return _best_match_profile(
            prepared_a,
            prepared_b,
            m,
            pearson,
            self_join,
            exclusion,
            False,
            tile_rows,
            tile_columns,
        )
    if profile == "1nn_bidirectional":
        return _bidirectional_best_match_profile(
            prepared_a,
            prepared_b,
            m,
            pearson,
            exclusion,
            False,
            tile_rows,
            tile_columns,
        )
    if profile == "sum":
        return _sum_threshold_profile(
            prepared_a,
            prepared_b,
            m,
            threshold,
            self_join,
            exclusion,
            False,
            tile_rows,
            tile_columns,
        )
    if profile == "matrix":
        return _matrix_summary(
            prepared_a,
            prepared_b,
            m,
            pearson,
            threshold,
            mheight,
            mwidth,
            self_join,
            exclusion,
            tile_rows=tile_rows,
            tile_columns=tile_columns,
        )
    if profile == "knn":
        if k is None or k <= 0:
            raise ValueError("k must be greater than 0")
        return _knn_profile(
            prepared_a,
            prepared_b,
            m,
            k,
            threshold,
            pearson,
            exclusion,
            tile_rows,
            tile_columns,
        )
    raise ValueError(f"Unknown profile type: {profile}")


def _resolve_execution_stream(params: dict[str, Any]) -> Any | None:
    execution_stream = _select_execution_stream(params["gpus"], params["threads"])
    if params["precision"] == "single":
        return execution_stream

    if execution_stream is not None and execution_stream.device == mx.gpu:
        raise ValueError(
            "Metal does not support float64; use precision='single' with "
            "gpus=[0], or select CPU execution for double/ultra precision"
        )
    return mx.default_stream(mx.cpu)


def _sequence_length_for_tuning(values: Any) -> int | None:
    shape = getattr(values, "shape", None)
    if shape is not None:
        try:
            if len(shape) == 1:
                return int(shape[0])
        except (TypeError, ValueError, OverflowError):
            return None
    try:
        return len(values)
    except (TypeError, OverflowError):
        return None


def _source_dtype_class_for_tuning(a: Any, b: Any | None) -> str:
    def classify(values: Any) -> str:
        if isinstance(values, mx.array):
            if values.dtype == mx.float32:
                return "float32"
            if values.dtype == mx.float64:
                return "float64"
            return "other"
        try:
            dtype = np.asarray(values).dtype
        except (TypeError, ValueError):
            return "other"
        if dtype == np.dtype(np.float32):
            return "float32"
        if dtype == np.dtype(np.float64):
            return "float64"
        return "other"

    class_a = classify(a)
    class_b = class_a if b is None else classify(b)
    return class_a if class_a == class_b else "other"


def _implicit_tuning_strategy(
    params: dict[str, Any], args: tuple[Any, ...], kwargs: dict[str, Any]
):
    if params["gpus"] is not None or params["threads"] > 0:
        return None
    try:
        a, b, raw_m = args[:3]
        m = operator.index(raw_m)
    except (TypeError, ValueError):
        return None
    length_a = _sequence_length_for_tuning(a)
    length_b = length_a if b is None else _sequence_length_for_tuning(b)
    if length_a is None or length_b is None:
        return None
    n_a = length_a - m + 1
    n_b = length_b - m + 1
    if min(n_a, n_b, m) <= 0:
        return None

    family = {
        "1nn": "1nn_index",
        "1nn_value": "1nn_value",
        "1nn_bidirectional": "bidirectional_ab",
        "sum": "sum_thresh",
        "matrix": "matrix_summary",
        "knn": "knn",
    }.get(kwargs.get("profile"))
    if family is None:
        return None

    from ._autotune_cache import (
        STRATEGY_BY_NAME,
        load_records,
        lookup_record,
        make_workload_key,
    )

    key_arguments = {
        "self_join": b is None,
        "aligned": b is not None
        and not params.get("allow_trivial_match", True),
        "dtype_class": _source_dtype_class_for_tuning(a, b),
        "max_tile_size": params["max_tile_size"],
        "k": kwargs.get("k"),
        "matrix_shape": (
            (kwargs.get("mheight"), kwargs.get("mwidth"))
            if family == "matrix_summary"
            else None
        ),
    }
    key = make_workload_key(
        family,
        params["precision"],
        "auto",
        n_a,
        n_b,
        m,
        **key_arguments,
    )
    record = lookup_record(key)
    if record is not None:
        return _ResolvedTuningStrategy(STRATEGY_BY_NAME[record.candidate])
    if family != "sum_thresh":
        return None

    # Density sampling is bounded to the sampled windows and is performed only
    # when this environment actually has a record for the otherwise-identical
    # SUM workload. Cache misses retain the untuned path without input work.
    comparable_fields = tuple(
        name
        for name in type(key).__dataclass_fields__
        if name != "profile_bucket"
    )
    signature = tuple(getattr(key, name) for name in comparable_fields)
    if not any(
        tuple(getattr(candidate.key, name) for name in comparable_fields)
        == signature
        for candidate in load_records()
    ):
        return None
    try:
        if b is None and n_a <= self_join_exclusion(m):
            threshold_density = 0.0
        else:
            threshold_density = _estimate_metal_sum_density(
                a,
                a if b is None else b,
                m,
                float(kwargs.get("threshold", 0.0)),
                b is None,
            )
    except (TypeError, ValueError, OverflowError, MemoryError):
        return None
    if threshold_density is None:
        return None
    key = make_workload_key(
        family,
        params["precision"],
        "auto",
        n_a,
        n_b,
        m,
        threshold_density=threshold_density,
        **key_arguments,
    )
    record = lookup_record(key)
    if record is None:
        return None
    return _ResolvedTuningStrategy(
        STRATEGY_BY_NAME[record.candidate], threshold_density
    )


def _run_profile_with_resources(params: dict[str, Any], *args: Any, **kwargs: Any):
    tuning_strategy = _implicit_tuning_strategy(params, args, kwargs)
    if tuning_strategy is None:
        execution_stream = _resolve_execution_stream(params)
    elif tuning_strategy.route == "cpu":
        execution_stream = mx.default_stream(mx.cpu)
    else:
        execution_stream = (
            mx.default_stream(mx.gpu)
            if params["precision"] == "single" and mx.metal.is_available()
            else _resolve_execution_stream(params)
        )
    execution_device = (
        mx.default_device()
        if execution_stream is None
        else execution_stream.device
    )
    custom_metal = tuning_strategy is None or tuning_strategy.route.startswith(
        "metal_"
    )
    use_metal_1nn = (
        params["precision"] == "single"
        and execution_device == mx.gpu
        and mx.metal.is_available()
        and custom_metal
    )
    portable_row_cap = BLOCK_ROWS
    if tuning_strategy is not None:
        portable_row_cap = dict(tuning_strategy.parameters).get(
            "portable_row_cap", BLOCK_ROWS
        )
    sum_density = (
        None
        if tuning_strategy is None
        else getattr(tuning_strategy, "sum_density", None)
    )
    stream_context = (
        nullcontext()
        if execution_stream is None
        else mx.stream(execution_stream)
    )
    with stream_context:
        return _run_profile(
            *args,
            precision=params["precision"],
            max_tile_size=params["max_tile_size"],
            use_metal_1nn=use_metal_1nn,
            use_metal_matrix=use_metal_1nn,
            use_metal_sum=use_metal_1nn,
            portable_row_cap=portable_row_cap,
            sum_density=sum_density,
            **kwargs,
        )


_AUTOTUNE_SUM_THRESHOLDS: dict[tuple[Any, ...], tuple[float, float]] = {}


def _autotune_sum_threshold(
    workload: Any, a: np.ndarray, b: np.ndarray | None
) -> tuple[float, float]:
    target = workload.threshold_density
    if target is None:
        raise ValueError("SUM autotune workloads require threshold density")
    cache_key = (
        workload.name,
        workload.precision,
        workload.n_a,
        workload.n_b,
        workload.m,
        workload.self_join,
        float(target),
    )
    cached = _AUTOTUNE_SUM_THRESHOLDS.get(cache_key)
    if cached is not None:
        return cached

    values_b = a if b is None else b
    lower = -1.0
    upper = 1.0
    selected_density = _estimate_metal_sum_density(
        a, values_b, workload.m, upper, workload.self_join
    )
    if selected_density is None:
        raise RuntimeError("SUM density sampling is unavailable")
    for _ in range(16):
        midpoint = (lower + upper) / 2.0
        density = _estimate_metal_sum_density(
            a, values_b, workload.m, midpoint, workload.self_join
        )
        if density is None:
            raise RuntimeError("SUM density sampling is unavailable")
        if density > target:
            lower = midpoint
        else:
            upper = midpoint
            selected_density = density
    result = (upper, selected_density)
    if len(_AUTOTUNE_SUM_THRESHOLDS) >= 64:
        _AUTOTUNE_SUM_THRESHOLDS.pop(next(iter(_AUTOTUNE_SUM_THRESHOLDS)))
    _AUTOTUNE_SUM_THRESHOLDS[cache_key] = result
    return result


def _autotune_execute_candidate(workload: Any, strategy: Any) -> Any:
    """Execute one explicit, deterministic autotune candidate."""

    dtype = np.float32 if workload.precision == "single" else np.float64
    length_a = workload.n_a + workload.m - 1
    length_b = workload.n_b + workload.m - 1
    seed = sum((index + 1) * ord(char) for index, char in enumerate(workload.name))
    rng = np.random.default_rng(seed % (2**32))
    a = rng.standard_normal(length_a).astype(dtype)
    a += np.sin(np.arange(length_a, dtype=dtype) / dtype(17.0))
    b = None
    if not workload.self_join:
        b = rng.standard_normal(length_b).astype(dtype)
        b += np.cos(np.arange(length_b, dtype=dtype) / dtype(19.0))

    route = strategy.route
    if route == "cpu":
        execution_stream = mx.default_stream(mx.cpu)
    else:
        if workload.precision != "single" or not mx.metal.is_available():
            raise RuntimeError("Metal candidate is not eligible for this workload")
        execution_stream = mx.default_stream(mx.gpu)
    custom_metal = route.startswith("metal_")
    row_cap = dict(strategy.parameters).get("portable_row_cap", BLOCK_ROWS)

    profile = {
        "1nn_index": "1nn",
        "1nn_value": "1nn_value",
        "sum_thresh": "sum",
        "matrix_summary": "matrix",
        "knn": "knn",
        "bidirectional_ab": "1nn_bidirectional",
    }[workload.profile]
    threshold = 0.0
    sum_density = None
    if workload.profile == "sum_thresh":
        threshold, sum_density = _autotune_sum_threshold(workload, a, b)
    elif workload.profile == "matrix_summary":
        threshold = -1.0

    if route == "metal_sum" and not _metal_sum_workload_is_worthwhile(
        workload.n_a,
        workload.n_b,
        workload.m,
        0.0 if sum_density is None else sum_density,
        workload.self_join,
    ):
        raise RuntimeError("sparse Metal SUM is ineligible for this density")

    run_kwargs: dict[str, Any] = {
        "pearson": True,
        "precision": workload.precision,
        "threshold": threshold,
        "allow_trivial_match": not workload.aligned,
        "profile": profile,
        "max_tile_size": workload.max_tile_size,
        "use_metal_1nn": custom_metal,
        "use_metal_matrix": custom_metal,
        "use_metal_sum": custom_metal,
        "portable_row_cap": row_cap,
        "sum_density": sum_density,
    }
    if workload.profile == "matrix_summary":
        if workload.matrix_shape is None:
            raise ValueError("matrix autotune workloads require a shape")
        run_kwargs["mheight"], run_kwargs["mwidth"] = workload.matrix_shape
    if workload.profile == "knn":
        run_kwargs["k"] = workload.k

    with mx.stream(execution_stream):
        return _run_profile(a, b, workload.m, **run_kwargs)


def selfjoin(a: Any, m: int, **kwargs: Any) -> tuple[np.ndarray, np.ndarray]:
    params = _parse_common_kwargs(kwargs)
    return _run_profile_with_resources(
        params,
        a,
        None,
        m,
        pearson=params["pearson"],
        profile="1nn",
    )


def abjoin(a: Any, b: Any, m: int, **kwargs: Any) -> tuple[np.ndarray, np.ndarray]:
    params = _parse_common_kwargs(kwargs, is_ab_join=True)
    return _run_profile_with_resources(
        params,
        a,
        b,
        m,
        pearson=params["pearson"],
        allow_trivial_match=params["allow_trivial_match"],
        profile="1nn",
    )


def abjoin_bidirectional(
    a: Any,
    b: Any,
    m: int,
    **kwargs: Any,
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Compute indexed AB-join profiles for both distance-matrix axes.

    The first pair matches ``abjoin(a, b, m)``. The second pair matches
    ``abjoin(b, a, m)`` and corresponds to SCAMP's ``keep_rows`` output.
    This native extension is intentionally not part of the strict ``pyscamp``
    compatibility namespace.
    """

    params = _parse_common_kwargs(kwargs, is_ab_join=True)
    return _run_profile_with_resources(
        params,
        a,
        b,
        m,
        pearson=params["pearson"],
        allow_trivial_match=params["allow_trivial_match"],
        profile="1nn_bidirectional",
    )


def selfjoin_1nn(a: Any, m: int, **kwargs: Any) -> np.ndarray:
    """Compute SCAMP's index-free 1NN profile for a self-join."""

    params = _parse_common_kwargs(kwargs)
    return _run_profile_with_resources(
        params,
        a,
        None,
        m,
        pearson=params["pearson"],
        profile="1nn_value",
    )


def abjoin_1nn(a: Any, b: Any, m: int, **kwargs: Any) -> np.ndarray:
    """Compute SCAMP's index-free 1NN profile for an AB-join."""

    params = _parse_common_kwargs(kwargs, is_ab_join=True)
    return _run_profile_with_resources(
        params,
        a,
        b,
        m,
        pearson=params["pearson"],
        allow_trivial_match=params["allow_trivial_match"],
        profile="1nn_value",
    )


def selfjoin_sum(a: Any, m: int, **kwargs: Any) -> np.ndarray:
    params = _parse_common_kwargs(kwargs, allow_threshold=True)
    return _run_profile_with_resources(
        params,
        a,
        None,
        m,
        pearson=True,
        threshold=params["threshold"],
        profile="sum",
    )


def abjoin_sum(a: Any, b: Any, m: int, **kwargs: Any) -> np.ndarray:
    params = _parse_common_kwargs(
        kwargs, allow_threshold=True, is_ab_join=True
    )
    return _run_profile_with_resources(
        params,
        a,
        b,
        m,
        pearson=True,
        threshold=params["threshold"],
        allow_trivial_match=params["allow_trivial_match"],
        profile="sum",
    )


def selfjoin_matrix(a: Any, m: int, **kwargs: Any) -> np.ndarray:
    params = _parse_common_kwargs(kwargs, allow_threshold=True, allow_matrix=True)
    return _run_profile_with_resources(
        params,
        a,
        None,
        m,
        pearson=params["pearson"],
        threshold=params["threshold"],
        mheight=params["mheight"],
        mwidth=params["mwidth"],
        profile="matrix",
    )


def abjoin_matrix(a: Any, b: Any, m: int, **kwargs: Any) -> np.ndarray:
    params = _parse_common_kwargs(
        kwargs,
        allow_threshold=True,
        allow_matrix=True,
        is_ab_join=True,
    )
    return _run_profile_with_resources(
        params,
        a,
        b,
        m,
        pearson=params["pearson"],
        threshold=params["threshold"],
        mheight=params["mheight"],
        mwidth=params["mwidth"],
        allow_trivial_match=params["allow_trivial_match"],
        profile="matrix",
    )


def selfjoin_knn(a: Any, m: int, k: int, **kwargs: Any) -> list[tuple[int, int, float]]:
    k = _positive_knn_k(k)
    params = _parse_common_kwargs(kwargs, allow_threshold=True)
    return _run_profile_with_resources(
        params,
        a,
        None,
        m,
        pearson=params["pearson"],
        threshold=params["threshold"],
        profile="knn",
        k=k,
    )


def abjoin_knn(a: Any, b: Any, m: int, k: int, **kwargs: Any) -> list[tuple[int, int, float]]:
    k = _positive_knn_k(k)
    params = _parse_common_kwargs(
        kwargs, allow_threshold=True, is_ab_join=True
    )
    return _run_profile_with_resources(
        params,
        a,
        b,
        m,
        pearson=params["pearson"],
        threshold=params["threshold"],
        allow_trivial_match=params["allow_trivial_match"],
        profile="knn",
        k=k,
    )
