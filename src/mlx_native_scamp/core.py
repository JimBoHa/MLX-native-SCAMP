from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import numpy as np

from ._autotune import (
    DEFAULT_BLOCK_ROWS,
    DEFAULT_THREADGROUP_WIDTH,
    TuningConfig,
    autotune,
    load_tuning,
)

SENTINEL = -2.0
FLATNESS_EPSILON = 1e-13
VALID_PRECISIONS = {"single", "double", "ultra"}
BLOCK_ROWS = DEFAULT_BLOCK_ROWS


@dataclass(slots=True)
class PreparedSeries:
    windows: Any
    valid: Any
    subsequences: int
    recurrence_clean: Any | None = None
    recurrence_means: Any | None = None
    recurrence_inv_norm: Any | None = None
    recurrence_df: Any | None = None
    recurrence_dg: Any | None = None


def gpu_supported() -> bool:
    try:
        return mx.default_device() == mx.gpu
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
        array = mx.array(values, dtype=dtype)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1D array")
    return array


def _parse_common_kwargs(kwargs: dict[str, Any], allow_matrix: bool = False, allow_threshold: bool = False) -> dict[str, Any]:
    valid_keys = {"verbose", "precision", "pearson", "gpus", "threads"}
    if allow_threshold:
        valid_keys.add("threshold")
    if allow_matrix:
        valid_keys.update({"mheight", "mwidth"})

    unknown = set(kwargs) - valid_keys
    if unknown:
        raise ValueError(f"Invalid keyword argument specified unknown argument: {sorted(unknown)[0]}")

    precision = kwargs.get("precision", "double")
    if precision not in VALID_PRECISIONS:
        raise ValueError("Invalid precision type specified: valid options are single, double, ultra")

    threshold = float(kwargs.get("threshold", 0.0))
    if allow_threshold and (threshold < -1.0 or threshold > 1.0):
        raise ValueError("Invalid threshold specified: value must be between -1 and 1")

    threads = int(kwargs.get("threads", 0))
    if threads < 0:
        raise ValueError("Invalid number of cpu worker threads specified, must be greater than or equal to 0.")

    params = {
        "pearson": bool(kwargs.get("pearson", False)),
        "precision": precision,
        "threshold": threshold,
        "verbose": bool(kwargs.get("verbose", False)),
        "threads": threads,
        "gpus": kwargs.get("gpus", None),
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


def _iterate_blocks(prepared_a: PreparedSeries, prepared_b: PreparedSeries, m: int, self_join: bool, block_rows: int):
    n_cols = prepared_a.subsequences
    exclusion = m // 4
    col_indices = mx.arange(n_cols, dtype=mx.int32)
    for row_start in range(0, prepared_b.subsequences, block_rows):
        row_end = min(prepared_b.subsequences, row_start + block_rows)
        row_indices = mx.arange(row_start, row_end, dtype=mx.int32)
        block_b = mx.take(prepared_b.windows, row_indices, axis=0)
        row_valid = mx.take(prepared_b.valid, row_indices, axis=0)
        block = block_b @ prepared_a.windows.T
        valid_mask = row_valid[:, None] & prepared_a.valid[None, :]
        sentinel_block = mx.full(block.shape, SENTINEL, dtype=block.dtype)
        block = mx.where(valid_mask, block, sentinel_block)
        if self_join and exclusion > 0:
            diag_mask = mx.abs(row_indices[:, None] - col_indices[None, :]) < exclusion
            block = mx.where(diag_mask, sentinel_block, block)
        yield row_start, row_end, row_indices, block


def _best_match_profile(
    prepared_a: PreparedSeries,
    prepared_b: PreparedSeries,
    m: int,
    pearson: bool,
    self_join: bool,
    use_metal_kernel: bool,
    block_rows: int,
    metal_threadgroup_width: int,
) -> tuple[np.ndarray, np.ndarray]:
    if use_metal_kernel:
        from ._metal_1nn import best_match

        corr_np, idx_np = best_match(
            prepared_a,
            prepared_b,
            m,
            self_join,
            threadgroup_width=metal_threadgroup_width,
        )
        return _convert_profile_output(corr_np, m, pearson), idx_np

    best_corr = mx.full(
        (prepared_a.subsequences,), SENTINEL, dtype=prepared_a.windows.dtype
    )
    best_idx = mx.full((prepared_a.subsequences,), -1, dtype=mx.int32)
    for row_start, _, _, block in _iterate_blocks(
        prepared_a, prepared_b, m, self_join, block_rows=block_rows
    ):
        block_best_corr = mx.max(block, axis=0)
        block_best_idx = mx.argmax(block, axis=0) + row_start
        update = block_best_corr > best_corr
        best_corr = mx.where(update, block_best_corr, best_corr)
        best_idx = mx.where(update, block_best_idx, best_idx)
    corr_np = np.asarray(best_corr).astype(np.float32)
    idx_np = np.asarray(best_idx).astype(np.int32)
    idx_np[corr_np < -1.0] = -1
    return _convert_profile_output(corr_np, m, pearson), idx_np


def _sum_threshold_profile(
    prepared_a: PreparedSeries,
    prepared_b: PreparedSeries,
    m: int,
    threshold: float,
    self_join: bool,
    block_rows: int,
) -> np.ndarray:
    accum = mx.zeros((prepared_a.subsequences,), dtype=prepared_a.windows.dtype)
    for _, _, _, block in _iterate_blocks(
        prepared_a, prepared_b, m, self_join, block_rows=block_rows
    ):
        filtered = mx.where(block > threshold, block, mx.zeros_like(block))
        accum = accum + mx.sum(filtered, axis=0)
    return np.asarray(accum, dtype=np.float64)


def _matrix_summary(
    prepared_a: PreparedSeries,
    prepared_b: PreparedSeries,
    m: int,
    pearson: bool,
    threshold: float,
    rows: int,
    cols: int,
    self_join: bool,
    block_rows: int,
) -> np.ndarray:
    row_edges = np.ceil(np.arange(rows + 1) * prepared_b.subsequences / rows).astype(int)
    col_edges = np.ceil(np.arange(cols + 1) * prepared_a.subsequences / cols).astype(int)
    col_ranges = [(int(col_edges[c]), int(col_edges[c + 1])) for c in range(cols)]
    col_indices = mx.arange(prepared_a.subsequences, dtype=mx.int32)
    summary = mx.full((rows, cols), SENTINEL, dtype=prepared_a.windows.dtype)

    for row_start, row_end, row_indices, block in _iterate_blocks(
        prepared_a, prepared_b, m, self_join, block_rows=block_rows
    ):
        if self_join:
            upper_mask = row_indices[:, None] <= col_indices[None, :]
            block = mx.where(upper_mask, block, mx.full(block.shape, SENTINEL, dtype=block.dtype))
        block_rows_summary: list[Any] = []
        for r in range(rows):
            rs = int(max(row_start, row_edges[r]))
            re = int(min(row_end, row_edges[r + 1]))
            if rs >= re:
                block_rows_summary.append(mx.full((cols,), SENTINEL, dtype=summary.dtype))
                continue
            row_slice = block[rs - row_start : re - row_start]
            block_cols_summary: list[Any] = []
            for c in range(cols):
                cs, ce = col_ranges[c]
                if cs >= ce:
                    block_cols_summary.append(mx.array(SENTINEL, dtype=summary.dtype))
                    continue
                block_cols_summary.append(mx.max(row_slice[:, cs:ce]))
            block_rows_summary.append(mx.stack(block_cols_summary, axis=0))
        summary = mx.maximum(summary, mx.stack(block_rows_summary, axis=0))

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
    block_rows: int,
) -> list[tuple[int, int, float]]:
    n_cols = prepared_a.subsequences
    best_corr = mx.full((k, n_cols), SENTINEL, dtype=prepared_a.windows.dtype)
    best_idx = mx.full((k, n_cols), -1, dtype=mx.int32)

    for row_start, _, _, block in _iterate_blocks(
        prepared_a, prepared_b, m, self_join, block_rows=block_rows
    ):
        local_order = _topk_desc_axis0(block, min(k, int(block.shape[0])))
        local_corr = mx.take_along_axis(block, local_order, axis=0)
        local_idx = local_order + row_start
        merged_corr = mx.concatenate([best_corr, local_corr], axis=0)
        merged_idx = mx.concatenate([best_idx, local_idx], axis=0)
        merged_order = _topk_desc_axis0(merged_corr, k)
        best_corr = mx.take_along_axis(merged_corr, merged_order, axis=0)
        best_idx = mx.take_along_axis(merged_idx, merged_order, axis=0)

    corr_np = np.asarray(best_corr).astype(np.float32)
    idx_np = np.asarray(best_idx).astype(np.int32)
    results: list[tuple[int, int, float]] = []
    for col in range(n_cols):
        seen_rows: set[int] = set()
        for rank in range(k):
            corr = corr_np[rank, col]
            row = int(idx_np[rank, col])
            if corr < threshold or corr < -1.0 or row < 0 or row in seen_rows:
                continue
            seen_rows.add(row)
            results.append((col, row, _convert_match_value(corr, m, pearson)))
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
    profile: str,
    k: int | None = None,
    use_metal_1nn: bool = False,
    block_rows: int = BLOCK_ROWS,
    metal_threadgroup_width: int = DEFAULT_THREADGROUP_WIDTH,
):
    has_b = b is not None
    float32_sources = _is_float32_input(a) and (
        not has_b or _is_float32_input(b)
    )
    compute_dtype = mx.float32 if precision == "single" else mx.float64
    # Metal does not provide native float64. The execution resolver places
    # double and ultra on MLX CPU; single can stay on the selected Metal GPU.
    # Upstream's ultra mode changes its sliding recurrence, while this direct
    # normalized-window implementation uses the same float64 path for both.
    series_a = _ensure_1d_array(a, "a", compute_dtype)
    if m <= 0:
        raise ValueError("m must be greater than 0")
    if int(series_a.shape[0]) < m:
        raise ValueError("m must be less than or equal to len(a)")

    series_b = _ensure_1d_array(b, "b", compute_dtype) if has_b else series_a
    if int(series_b.shape[0]) < m:
        raise ValueError("m must be less than or equal to len(b)")

    prepared_a = _prepare_series(series_a, m)
    prepared_b = _prepare_series(series_b, m)
    self_join = not has_b

    if profile == "1nn":
        use_metal_1nn = (
            use_metal_1nn
            and float32_sources
            and _metal_recurrence_is_safe(prepared_a, prepared_b, m)
        )
        return _best_match_profile(
            prepared_a,
            prepared_b,
            m,
            pearson,
            self_join,
            use_metal_1nn,
            block_rows,
            metal_threadgroup_width,
        )
    if profile == "sum":
        return _sum_threshold_profile(
            prepared_a, prepared_b, m, threshold, self_join, block_rows
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
            block_rows,
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
            block_rows,
        )
    raise ValueError(f"Unknown profile type: {profile}")


def _resolve_execution_stream(
    params: dict[str, Any],
    tuning: TuningConfig | None = None,
    profile: str = "1nn",
) -> Any | None:
    execution_stream = _select_execution_stream(params["gpus"], params["threads"])
    if params["precision"] == "single":
        if (
            tuning is not None
            and params["gpus"] is None
            and params["threads"] == 0
        ):
            preferred_backend = (
                tuning.preferred_1nn_backend
                if profile == "1nn"
                else tuning.preferred_portable_backend
            )
            if preferred_backend == "metal" and mx.metal.is_available():
                return mx.default_stream(mx.gpu)
            return mx.default_stream(mx.cpu)
        return execution_stream

    if execution_stream is not None and execution_stream.device == mx.gpu:
        raise ValueError(
            "Metal does not support float64; use precision='single' with "
            "gpus=[0], or select CPU execution for double/ultra precision"
        )
    return mx.default_stream(mx.cpu)


def _run_profile_with_resources(params: dict[str, Any], *args: Any, **kwargs: Any):
    tuning = load_tuning()
    execution_stream = _resolve_execution_stream(
        params, tuning, profile=str(kwargs.get("profile", "1nn"))
    )
    execution_device = (
        mx.default_device()
        if execution_stream is None
        else execution_stream.device
    )
    use_metal_1nn = (
        params["precision"] == "single"
        and execution_device == mx.gpu
        and mx.metal.is_available()
    )
    if tuning is None:
        block_rows = BLOCK_ROWS
        metal_threadgroup_width = DEFAULT_THREADGROUP_WIDTH
    elif execution_device == mx.gpu:
        block_rows = tuning.metal_block_rows
        metal_threadgroup_width = tuning.metal_threadgroup_width
    else:
        block_rows = tuning.cpu_block_rows
        metal_threadgroup_width = DEFAULT_THREADGROUP_WIDTH
    stream_context = (
        nullcontext()
        if execution_stream is None
        else mx.stream(execution_stream)
    )
    with stream_context:
        return _run_profile(
            *args,
            precision=params["precision"],
            use_metal_1nn=use_metal_1nn,
            block_rows=block_rows,
            metal_threadgroup_width=metal_threadgroup_width,
            **kwargs,
        )


def _autotune_benchmark_candidate(
    workload: str,
    device: str,
    block_rows: int,
    threadgroup_width: int,
    series: np.ndarray,
    window: int,
) -> tuple[np.ndarray, np.ndarray]:
    if workload not in {"portable_1nn", "metal_1nn"}:
        raise ValueError(f"Unknown autotune workload: {workload}")
    if device not in {"cpu", "metal"}:
        raise ValueError(f"Unknown autotune device: {device}")
    if workload == "metal_1nn" and device != "metal":
        raise ValueError("The Metal diagonal workload requires device='metal'")

    execution_device = mx.cpu if device == "cpu" else mx.gpu
    with mx.stream(execution_device):
        return _run_profile(
            series,
            None,
            window,
            pearson=True,
            precision="single",
            profile="1nn",
            use_metal_1nn=workload == "metal_1nn",
            block_rows=block_rows,
            metal_threadgroup_width=threadgroup_width,
        )


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
    params = _parse_common_kwargs(kwargs)
    return _run_profile_with_resources(
        params,
        a,
        b,
        m,
        pearson=params["pearson"],
        profile="1nn",
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
    params = _parse_common_kwargs(kwargs, allow_threshold=True)
    return _run_profile_with_resources(
        params,
        a,
        b,
        m,
        pearson=True,
        threshold=params["threshold"],
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
    params = _parse_common_kwargs(kwargs, allow_threshold=True, allow_matrix=True)
    return _run_profile_with_resources(
        params,
        a,
        b,
        m,
        pearson=params["pearson"],
        threshold=params["threshold"],
        mheight=params["mheight"],
        mwidth=params["mwidth"],
        profile="matrix",
    )


def selfjoin_knn(a: Any, m: int, k: int, **kwargs: Any) -> list[tuple[int, int, float]]:
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
    params = _parse_common_kwargs(kwargs, allow_threshold=True)
    return _run_profile_with_resources(
        params,
        a,
        b,
        m,
        pearson=params["pearson"],
        threshold=params["threshold"],
        profile="knn",
        k=k,
    )
