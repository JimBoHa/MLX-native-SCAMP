"""MLX execution primitives shared by local and remote SCAMP workers."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import numpy as np

from mlx_native_scamp.core import SENTINEL, _prepare_series


MAX_PROFILE_INDEX = int(np.iinfo(np.int64).max)

# MLX's stream context changes process-global default state. Overlapping
# contexts from gRPC threads can restore that state out of order, leaving a
# CPU worker's device selected for unrelated callers after the request ends.
_MLX_STREAM_LOCK = threading.RLock()


def estimate_1nn_tile_working_set_bytes(
    row_count: int, column_count: int, window: int
) -> int:
    """Conservatively estimate peak bytes for the current dense MLX tile."""

    if row_count <= 0 or column_count <= 0 or window <= 0:
        raise ValueError("tile dimensions and window must be positive")
    matrix_cells = row_count * column_count
    window_cells = (row_count + column_count) * window
    # The lazy graph may retain the GEMM result, validity/sentinel selections,
    # and normalized-window intermediates until reduction is evaluated.
    return matrix_cells * 20 + window_cells * 16 + (row_count + column_count) * 32


@dataclass(frozen=True, slots=True)
class TileExecution:
    """Partial profiles produced by one rectangular similarity tile."""

    column_values: np.ndarray
    column_indices: np.ndarray
    row_values: np.ndarray
    row_indices: np.ndarray


def execute_1nn_tile(
    series_a: np.ndarray,
    series_b: np.ndarray | None,
    window: int,
    *,
    row_start: int,
    row_stop: int,
    column_start: int,
    column_stop: int,
    exclusion_zone: int,
    compute_rows: bool,
    compute_columns: bool,
    device: Any,
    series_a_offset: int = 0,
    series_b_offset: int = 0,
) -> TileExecution:
    """Execute a rectangular 1NN correlation tile on one MLX device.

    Series A is the column dimension. Series B is the row dimension; omitting
    it performs a self-join. Returned indices are global subsequence indices so
    a coordinator can merge independently scheduled tiles directly.
    """

    if window <= 0:
        raise ValueError("window must be greater than zero")
    if not compute_rows and not compute_columns:
        raise ValueError("at least one tile profile direction must be requested")
    if exclusion_zone < 0:
        raise ValueError("exclusion_zone cannot be negative")
    if series_a_offset < 0 or series_b_offset < 0:
        raise ValueError("series payload offsets cannot be negative")
    if row_stop > MAX_PROFILE_INDEX or column_stop > MAX_PROFILE_INDEX:
        raise ValueError("global profile bounds exceed the signed 64-bit index range")

    input_a = np.asarray(series_a)
    input_b = input_a if series_b is None else np.asarray(series_b)
    if input_a.ndim != 1 or input_b.ndim != 1:
        raise ValueError("tile input series must be one-dimensional")
    if input_a.size < window or input_b.size < window:
        raise ValueError("window cannot exceed either input series length")

    resolved_b_offset = series_a_offset if series_b is None else series_b_offset
    column_local_start = column_start - series_a_offset
    column_local_stop = column_stop - series_a_offset
    row_local_start = row_start - resolved_b_offset
    row_local_stop = row_stop - resolved_b_offset
    columns = input_a.size - window + 1
    rows = input_b.size - window + 1
    if not 0 <= column_local_start < column_local_stop <= columns:
        raise ValueError("column tile bounds are outside the supplied A payload")
    if not 0 <= row_local_start < row_local_stop <= rows:
        raise ValueError("row tile bounds are outside the supplied B payload")

    # Transfer and normalize only the raw samples needed by this tile, keeping
    # the worker's Metal working set proportional to tile size.
    segment_a = input_a[column_local_start : column_local_stop + window - 1]
    segment_b = input_b[row_local_start : row_local_stop + window - 1]
    same_self_tile = (
        series_b is None and row_start == column_start and row_stop == column_stop
    )

    with _MLX_STREAM_LOCK:
        with mx.stream(device):
            mlx_a = mx.array(segment_a, dtype=mx.float32)
            prepared_a = _prepare_series(mlx_a, window)
            if same_self_tile:
                prepared_b = prepared_a
            else:
                mlx_b = mx.array(segment_b, dtype=mx.float32)
                prepared_b = _prepare_series(mlx_b, window)

            tile_a = prepared_a.windows
            tile_b = prepared_b.windows
            block = tile_b @ tile_a.T
            valid = prepared_b.valid[:, None] & prepared_a.valid[None, :]
            block = mx.where(
                valid,
                mx.clip(block, -1.0, 1.0),
                mx.full(block.shape, SENTINEL, dtype=mx.float32),
            )
            if exclusion_zone:
                separated = (
                    row_start >= column_stop + exclusion_zone - 1
                    or column_start >= row_stop + exclusion_zone - 1
                )
                if not separated:
                    base_delta = row_start - column_start
                    int32 = np.iinfo(np.int32)
                    row_count = row_stop - row_start
                    column_count = column_stop - column_start
                    expression_min = base_delta - (column_count - 1)
                    expression_max = base_delta + (row_count - 1)
                    if (
                        int32.min < expression_min
                        and expression_max <= int32.max
                        and exclusion_zone <= int32.max
                    ):
                        row_indices = mx.arange(row_count, dtype=mx.int32)
                        column_indices = mx.arange(column_count, dtype=mx.int32)
                        trivial = (
                            mx.abs(
                                row_indices[:, None]
                                - column_indices[None, :]
                                + int(base_delta)
                            )
                            < exclusion_zone
                        )
                    else:
                        # Metal's fast int32 path cannot represent this
                        # boundary. Build the rare large-offset mask with safe
                        # host int64 arithmetic, then transfer only the mask.
                        column_indices = np.arange(column_count, dtype=np.int64)
                        host_mask = np.empty((row_count, column_count), dtype=np.bool_)
                        for row_index in range(row_count):
                            differences = (
                                np.int64(base_delta + row_index) - column_indices
                            )
                            host_mask[row_index] = np.abs(differences) < exclusion_zone
                        trivial = mx.array(host_mask)
                    block = mx.where(
                        trivial,
                        mx.full(block.shape, SENTINEL, dtype=mx.float32),
                        block,
                    )

            if compute_columns:
                column_values_mx = mx.max(block, axis=0)
                column_indices_mx = mx.argmax(block, axis=0).astype(mx.int32)
            else:
                column_values_mx = mx.array([], dtype=mx.float32)
                column_indices_mx = mx.array([], dtype=mx.int64)

            if compute_rows:
                row_values_mx = mx.max(block, axis=1)
                row_indices_mx = mx.argmax(block, axis=1).astype(mx.int32)
            else:
                row_values_mx = mx.array([], dtype=mx.float32)
                row_indices_mx = mx.array([], dtype=mx.int64)

            mx.eval(
                column_values_mx,
                column_indices_mx,
                row_values_mx,
                row_indices_mx,
            )

    column_values = np.asarray(column_values_mx, dtype=np.float32)
    column_indices = np.asarray(column_indices_mx, dtype=np.int64)
    column_indices += row_start
    column_indices[column_values < -1.0] = -1
    row_values = np.asarray(row_values_mx, dtype=np.float32)
    row_indices = np.asarray(row_indices_mx, dtype=np.int64)
    row_indices += column_start
    row_indices[row_values < -1.0] = -1
    return TileExecution(column_values, column_indices, row_values, row_indices)
