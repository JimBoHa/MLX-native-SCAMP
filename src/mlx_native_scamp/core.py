from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import numpy as np

SENTINEL = -2.0
FLATNESS_EPSILON = 1e-13
VALID_PRECISIONS = {"single", "mixed", "double", "ultra"}
BLOCK_ROWS = 256
MIB = 1024 * 1024
DEFAULT_UNIFIED_MEMORY_BYTES = 2 * 1024 * MIB
MIN_SIMILARITY_TILE_BUDGET_BYTES = 8 * MIB
MAX_SIMILARITY_TILE_BUDGET_BYTES = 64 * MIB
SIMILARITY_TILE_WORKING_SET_DIVISOR = 16
SIMILARITY_TILE_TEMPORARY_FACTOR = 8
MAX_IN_FLIGHT_SIMILARITY_TILES = 2


@dataclass(slots=True)
class PreparedSeries:
    windows: Any
    valid: Any
    subsequences: int


def _schedule_reducer_state(*state: Any) -> None:
    """Schedule compact state so MLX can release its similarity block."""
    mx.async_eval(*state)


@dataclass(slots=True)
class ReducerScheduler:
    """Apply backpressure while materializing compact reducer state."""

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


@dataclass(slots=True)
class SimilarityTile:
    row_start: int
    row_end: int
    col_start: int
    col_end: int
    row_indices: Any
    col_indices: Any
    values: Any


def _device_working_set_bytes() -> int:
    """Return Apple's recommended unified-memory working set when available."""
    getters = []
    if hasattr(mx, "device_info"):
        getters.append(mx.device_info)
    metal = getattr(mx, "metal", None)
    if metal is not None and hasattr(metal, "device_info"):
        getters.append(metal.device_info)

    for getter in getters:
        try:
            info = getter()
        except Exception:
            continue
        if not isinstance(info, dict):
            continue
        for key in ("max_recommended_working_set_size", "memory_size"):
            value = info.get(key)
            try:
                size = int(value)
            except (TypeError, ValueError):
                continue
            if size > 0:
                return size
    return DEFAULT_UNIFIED_MEMORY_BYTES


def _prepared_series_bytes(prepared: PreparedSeries) -> int:
    itemsize = int(prepared.windows.dtype.size)
    window_values = prepared.subsequences * int(prepared.windows.shape[1])
    return window_values * itemsize + prepared.subsequences


def _automatic_max_tile_size(
    prepared_a: PreparedSeries,
    prepared_b: PreparedSeries,
    m: int,
) -> int:
    """Choose a bounded SCAMP-style time-series tile from unified memory."""
    working_set = _device_working_set_bytes()
    prepared_bytes = _prepared_series_bytes(prepared_a) + _prepared_series_bytes(
        prepared_b
    )
    available = max(0, working_set - prepared_bytes)
    tile_budget = available // SIMILARITY_TILE_WORKING_SET_DIVISOR
    tile_budget = max(
        MIN_SIMILARITY_TILE_BUDGET_BYTES,
        min(MAX_SIMILARITY_TILE_BUDGET_BYTES, tile_budget),
    )

    itemsize = max(
        int(prepared_a.windows.dtype.size), int(prepared_b.windows.dtype.size)
    )
    bytes_per_column = BLOCK_ROWS * (
        itemsize * SIMILARITY_TILE_TEMPORARY_FACTOR + 1
    )
    tile_subsequences = max(1, tile_budget // bytes_per_column)

    minimum_time_series_tile = max(1024, 2 * m)
    minimum_subsequences = minimum_time_series_tile - m + 1
    tile_subsequences = max(tile_subsequences, minimum_subsequences)
    return tile_subsequences + m - 1


def _tile_subsequence_count(max_tile_size: int, m: int) -> int:
    """Translate SCAMP's time-series tile length to correlation dimensions."""
    return max_tile_size - m + 1


def gpu_supported() -> bool:
    try:
        return mx.default_device() == mx.gpu
    except Exception:
        return False


def _ensure_1d_array(values: Any, name: str) -> Any:
    if isinstance(values, mx.array):
        array = values
        if array.dtype != mx.float32:
            array = array.astype(mx.float32)
    else:
        array = mx.array(values, dtype=mx.float32)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1D array")
    return array


def _parse_common_kwargs(kwargs: dict[str, Any], allow_matrix: bool = False, allow_threshold: bool = False) -> dict[str, Any]:
    valid_keys = {"verbose", "precision", "pearson", "gpus", "threads", "max_tile_size"}
    if allow_threshold:
        valid_keys.add("threshold")
    if allow_matrix:
        valid_keys.update({"mheight", "mwidth"})

    unknown = set(kwargs) - valid_keys
    if unknown:
        raise ValueError(f"Invalid keyword argument specified unknown argument: {sorted(unknown)[0]}")

    precision = kwargs.get("precision", "double")
    if precision not in VALID_PRECISIONS:
        raise ValueError("Invalid precision type specified: valid options are single, mixed, double, ultra")

    threshold = float(kwargs.get("threshold", 0.0))
    if allow_threshold and (threshold < -1.0 or threshold > 1.0):
        raise ValueError("Invalid threshold specified: value must be between -1 and 1")

    threads = int(kwargs.get("threads", 0))
    if threads < 0:
        raise ValueError("Invalid number of cpu worker threads specified, must be greater than or equal to 0.")

    max_tile_size = None
    if "max_tile_size" in kwargs:
        try:
            max_tile_size = operator.index(kwargs["max_tile_size"])
        except TypeError as exc:
            raise ValueError("max_tile_size must be an integer") from exc
        if max_tile_size < 1024:
            raise ValueError("max_tile_size must be at least 1024")

    params = {
        "pearson": bool(kwargs.get("pearson", False)),
        "precision": precision,
        "threshold": threshold,
        "verbose": bool(kwargs.get("verbose", False)),
        "threads": threads,
        "gpus": kwargs.get("gpus", None),
        "max_tile_size": max_tile_size,
    }
    if allow_matrix:
        params["mheight"] = int(kwargs.get("mheight", 50))
        params["mwidth"] = int(kwargs.get("mwidth", 50))
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
    means = mx.mean(windows, axis=1, keepdims=True)
    centered = windows - means
    norms_sq = mx.sum(centered * centered, axis=1)
    valid = _sliding_valid_mask(finite, m) & (norms_sq > FLATNESS_EPSILON)
    inv_norm = mx.where(valid, 1.0 / mx.sqrt(mx.maximum(norms_sq, FLATNESS_EPSILON)), 0.0)
    normalized = centered * inv_norm[:, None]
    return PreparedSeries(windows=normalized, valid=valid, subsequences=int(normalized.shape[0]))


def _topk_desc_axis0(values: Any, k: int) -> Any:
    rows = int(values.shape[0])
    k = max(1, min(int(k), rows))
    order = mx.argsort(values, axis=0)
    positions = mx.arange(rows - 1, rows - k - 1, -1)
    return mx.take(order, positions, axis=0)


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


def _column_ranges(
    subsequences: int, tile_subsequences: int
) -> list[tuple[int, int]]:
    return [
        (start, min(subsequences, start + tile_subsequences))
        for start in range(0, subsequences, tile_subsequences)
    ]


def _iterate_blocks(
    prepared_a: PreparedSeries,
    prepared_b: PreparedSeries,
    m: int,
    self_join: bool,
    block_rows: int,
    tile_subsequences: int,
):
    n_cols = prepared_a.subsequences
    exclusion = m // 4
    row_tile_size = min(block_rows, tile_subsequences)
    col_ranges = _column_ranges(n_cols, tile_subsequences)
    for row_start in range(0, prepared_b.subsequences, row_tile_size):
        row_end = min(prepared_b.subsequences, row_start + row_tile_size)
        row_indices = mx.arange(row_start, row_end, dtype=mx.int32)
        block_b = mx.take(prepared_b.windows, row_indices, axis=0)
        row_valid = mx.take(prepared_b.valid, row_indices, axis=0)
        for col_start, col_end in col_ranges:
            col_indices = mx.arange(col_start, col_end, dtype=mx.int32)
            block_a = prepared_a.windows[col_start:col_end]
            col_valid = prepared_a.valid[col_start:col_end]
            block = block_b @ block_a.T
            valid_mask = row_valid[:, None] & col_valid[None, :]
            sentinel_block = mx.full(block.shape, SENTINEL, dtype=block.dtype)
            block = mx.where(valid_mask, block, sentinel_block)
            if self_join and exclusion > 0:
                diag_mask = mx.abs(row_indices[:, None] - col_indices[None, :]) < exclusion
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


def _best_match_profile(
    prepared_a: PreparedSeries,
    prepared_b: PreparedSeries,
    m: int,
    pearson: bool,
    self_join: bool,
    tile_subsequences: int,
) -> tuple[np.ndarray, np.ndarray]:
    col_ranges = _column_ranges(prepared_a.subsequences, tile_subsequences)
    best_corr = {
        start: mx.full(
            (end - start,), SENTINEL, dtype=prepared_a.windows.dtype
        )
        for start, end in col_ranges
    }
    best_idx = {start: mx.full((end - start,), -1, dtype=mx.int32) for start, end in col_ranges}
    scheduler = ReducerScheduler()
    for tile in _iterate_blocks(
        prepared_a,
        prepared_b,
        m,
        self_join,
        block_rows=BLOCK_ROWS,
        tile_subsequences=tile_subsequences,
    ):
        block_best_corr = mx.max(tile.values, axis=0)
        block_best_idx = mx.argmax(tile.values, axis=0) + tile.row_start
        current_corr = best_corr[tile.col_start]
        current_idx = best_idx[tile.col_start]
        update = block_best_corr > current_corr
        best_corr[tile.col_start] = mx.where(update, block_best_corr, current_corr)
        best_idx[tile.col_start] = mx.where(update, block_best_idx, current_idx)
        scheduler.schedule(best_corr[tile.col_start], best_idx[tile.col_start])
    scheduler.finish()
    corr_np = np.concatenate(
        [np.asarray(best_corr[start]).astype(np.float32) for start, _ in col_ranges]
    )
    idx_np = np.concatenate(
        [np.asarray(best_idx[start]).astype(np.int32) for start, _ in col_ranges]
    )
    idx_np[corr_np < -1.0] = -1
    return _convert_profile_output(corr_np, m, pearson), idx_np


def _sum_threshold_profile(
    prepared_a: PreparedSeries,
    prepared_b: PreparedSeries,
    m: int,
    threshold: float,
    self_join: bool,
    tile_subsequences: int,
) -> np.ndarray:
    col_ranges = _column_ranges(prepared_a.subsequences, tile_subsequences)
    accum = {
        start: mx.zeros((end - start,), dtype=prepared_a.windows.dtype)
        for start, end in col_ranges
    }
    scheduler = ReducerScheduler()
    for tile in _iterate_blocks(
        prepared_a,
        prepared_b,
        m,
        self_join,
        block_rows=BLOCK_ROWS,
        tile_subsequences=tile_subsequences,
    ):
        filtered = mx.where(
            tile.values > threshold, tile.values, mx.zeros_like(tile.values)
        )
        accum[tile.col_start] = accum[tile.col_start] + mx.sum(filtered, axis=0)
        scheduler.schedule(accum[tile.col_start])
    scheduler.finish()
    return np.concatenate(
        [np.asarray(accum[start], dtype=np.float64) for start, _ in col_ranges]
    )


def _matrix_summary(
    prepared_a: PreparedSeries,
    prepared_b: PreparedSeries,
    m: int,
    pearson: bool,
    threshold: float,
    rows: int,
    cols: int,
    self_join: bool,
    tile_subsequences: int,
) -> np.ndarray:
    row_edges = np.ceil(np.arange(rows + 1) * prepared_b.subsequences / rows).astype(int)
    col_edges = np.ceil(np.arange(cols + 1) * prepared_a.subsequences / cols).astype(int)
    col_ranges = [(int(col_edges[c]), int(col_edges[c + 1])) for c in range(cols)]
    summary = mx.full(
        (rows, cols), SENTINEL, dtype=prepared_a.windows.dtype
    )
    scheduler = ReducerScheduler()

    for tile in _iterate_blocks(
        prepared_a,
        prepared_b,
        m,
        self_join,
        block_rows=BLOCK_ROWS,
        tile_subsequences=tile_subsequences,
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
                    mx.max(row_slice[:, cs - tile.col_start : ce - tile.col_start])
                )
            block_rows_summary.append(mx.stack(block_cols_summary, axis=0))
        summary = mx.maximum(summary, mx.stack(block_rows_summary, axis=0))
        scheduler.schedule(summary)

    scheduler.finish()
    summary = np.asarray(summary, dtype=np.float32)
    summary[summary < -1.0] = np.nan
    if threshold is not None:
        summary[summary < threshold] = np.nan
    if pearson:
        return summary.astype(np.float32)
    out = np.sqrt(np.maximum(2.0 * m * (1.0 - summary), 0.0)).astype(np.float32)
    out[np.isnan(summary)] = np.nan
    return out


def _knn_profile(
    prepared_a: PreparedSeries,
    prepared_b: PreparedSeries,
    m: int,
    k: int,
    threshold: float,
    pearson: bool,
    self_join: bool,
    tile_subsequences: int,
) -> list[tuple[int, int, float]]:
    col_ranges = _column_ranges(prepared_a.subsequences, tile_subsequences)
    best_corr = {
        start: mx.full(
            (k, end - start), SENTINEL, dtype=prepared_a.windows.dtype
        )
        for start, end in col_ranges
    }
    best_idx = {
        start: mx.full((k, end - start), -1, dtype=mx.int32) for start, end in col_ranges
    }
    scheduler = ReducerScheduler()

    for tile in _iterate_blocks(
        prepared_a,
        prepared_b,
        m,
        self_join,
        block_rows=BLOCK_ROWS,
        tile_subsequences=tile_subsequences,
    ):
        local_order = _topk_desc_axis0(tile.values, min(k, int(tile.values.shape[0])))
        local_corr = mx.take_along_axis(tile.values, local_order, axis=0)
        local_idx = local_order + tile.row_start
        merged_corr = mx.concatenate([best_corr[tile.col_start], local_corr], axis=0)
        merged_idx = mx.concatenate([best_idx[tile.col_start], local_idx], axis=0)
        merged_order = _topk_desc_axis0(merged_corr, k)
        best_corr[tile.col_start] = mx.take_along_axis(merged_corr, merged_order, axis=0)
        best_idx[tile.col_start] = mx.take_along_axis(merged_idx, merged_order, axis=0)
        scheduler.schedule(best_corr[tile.col_start], best_idx[tile.col_start])

    scheduler.finish()
    results: list[tuple[int, int, float]] = []
    for col_start, col_end in col_ranges:
        corr_np = np.asarray(best_corr[col_start]).astype(np.float32)
        idx_np = np.asarray(best_idx[col_start]).astype(np.int32)
        for local_col in range(col_end - col_start):
            seen_rows: set[int] = set()
            for rank in range(k):
                corr = corr_np[rank, local_col]
                row = int(idx_np[rank, local_col])
                if corr < threshold or corr < -1.0 or row < 0 or row in seen_rows:
                    continue
                seen_rows.add(row)
                results.append(
                    (col_start + local_col, row, _convert_match_value(corr, m, pearson))
                )
    return results


def _run_profile(
    a: Any,
    b: Any | None,
    m: int,
    *,
    pearson: bool,
    threshold: float = 0.0,
    mheight: int = 50,
    mwidth: int = 50,
    profile: str,
    k: int | None = None,
    max_tile_size: int | None = None,
):
    series_a = _ensure_1d_array(a, "a")
    if m <= 0:
        raise ValueError("m must be greater than 0")
    if int(series_a.shape[0]) < m:
        raise ValueError("m must be less than or equal to len(a)")

    has_b = b is not None
    series_b = _ensure_1d_array(b, "b") if has_b else series_a
    if int(series_b.shape[0]) < m:
        raise ValueError("m must be less than or equal to len(b)")
    if max_tile_size is not None and max_tile_size < 2 * m:
        raise ValueError("max_tile_size must be at least twice m")

    prepared_a = _prepare_series(series_a, m)
    prepared_b = _prepare_series(series_b, m)
    self_join = not has_b
    resolved_max_tile_size = (
        max_tile_size
        if max_tile_size is not None
        else _automatic_max_tile_size(prepared_a, prepared_b, m)
    )
    tile_subsequences = _tile_subsequence_count(resolved_max_tile_size, m)

    if profile == "1nn":
        return _best_match_profile(
            prepared_a, prepared_b, m, pearson, self_join, tile_subsequences
        )
    if profile == "sum":
        return _sum_threshold_profile(
            prepared_a, prepared_b, m, threshold, self_join, tile_subsequences
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
            tile_subsequences,
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
            self_join,
            tile_subsequences,
        )
    raise ValueError(f"Unknown profile type: {profile}")


def selfjoin(a: Any, m: int, **kwargs: Any) -> tuple[np.ndarray, np.ndarray]:
    params = _parse_common_kwargs(kwargs)
    return _run_profile(
        a,
        None,
        m,
        pearson=params["pearson"],
        profile="1nn",
        max_tile_size=params["max_tile_size"],
    )


def abjoin(a: Any, b: Any, m: int, **kwargs: Any) -> tuple[np.ndarray, np.ndarray]:
    params = _parse_common_kwargs(kwargs)
    return _run_profile(
        a,
        b,
        m,
        pearson=params["pearson"],
        profile="1nn",
        max_tile_size=params["max_tile_size"],
    )


def selfjoin_sum(a: Any, m: int, **kwargs: Any) -> np.ndarray:
    params = _parse_common_kwargs(kwargs, allow_threshold=True)
    return _run_profile(
        a,
        None,
        m,
        pearson=True,
        threshold=params["threshold"],
        profile="sum",
        max_tile_size=params["max_tile_size"],
    )


def abjoin_sum(a: Any, b: Any, m: int, **kwargs: Any) -> np.ndarray:
    params = _parse_common_kwargs(kwargs, allow_threshold=True)
    return _run_profile(
        a,
        b,
        m,
        pearson=True,
        threshold=params["threshold"],
        profile="sum",
        max_tile_size=params["max_tile_size"],
    )


def selfjoin_matrix(a: Any, m: int, **kwargs: Any) -> np.ndarray:
    params = _parse_common_kwargs(kwargs, allow_threshold=True, allow_matrix=True)
    return _run_profile(
        a,
        None,
        m,
        pearson=params["pearson"],
        threshold=params["threshold"],
        mheight=params["mheight"],
        mwidth=params["mwidth"],
        profile="matrix",
        max_tile_size=params["max_tile_size"],
    )


def abjoin_matrix(a: Any, b: Any, m: int, **kwargs: Any) -> np.ndarray:
    params = _parse_common_kwargs(kwargs, allow_threshold=True, allow_matrix=True)
    return _run_profile(
        a,
        b,
        m,
        pearson=params["pearson"],
        threshold=params["threshold"],
        mheight=params["mheight"],
        mwidth=params["mwidth"],
        profile="matrix",
        max_tile_size=params["max_tile_size"],
    )


def selfjoin_knn(a: Any, m: int, k: int, **kwargs: Any) -> list[tuple[int, int, float]]:
    params = _parse_common_kwargs(kwargs, allow_threshold=True)
    return _run_profile(
        a,
        None,
        m,
        pearson=params["pearson"],
        threshold=params["threshold"],
        profile="knn",
        k=k,
        max_tile_size=params["max_tile_size"],
    )


def abjoin_knn(a: Any, b: Any, m: int, k: int, **kwargs: Any) -> list[tuple[int, int, float]]:
    params = _parse_common_kwargs(kwargs, allow_threshold=True)
    return _run_profile(
        a,
        b,
        m,
        pearson=params["pearson"],
        threshold=params["threshold"],
        profile="knn",
        k=k,
        max_tile_size=params["max_tile_size"],
    )
