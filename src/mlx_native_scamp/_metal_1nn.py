from __future__ import annotations

from typing import Any

import mlx.core as mx
import numpy as np

_INPUT_NAMES = [
    "clean_a",
    "clean_b",
    "means_a",
    "means_b",
    "inv_norm_a",
    "inv_norm_b",
    "df_a",
    "df_b",
    "dg_a",
    "dg_b",
    "config",
]

_PROFILE_SOURCE = r"""
    uint diag_slot = thread_position_in_grid.x;
    uint n_a = means_a_shape[0];
    uint n_b = means_b_shape[0];
    uint m = config[0];
    uint exclusion = config[1];
    bool self_join = config[2] != 0;
    bool apply_exclusion = config[3] != 0;

    int diagonal;
    if (self_join) {
        diagonal = int(exclusion + diag_slot);
    } else {
        diagonal = int(diag_slot) - int(n_a - 1);
    }

    uint col = diagonal < 0 ? uint(-diagonal) : 0;
    uint row = diagonal > 0 ? uint(diagonal) : 0;
    if (self_join) {
        col = uint(diagonal);
        row = 0;
    }
    uint diagonal_length = metal::min(n_a - col, n_b - row);

    float covariance = 0.0f;
    for (uint k = 0; k < m; ++k) {
        covariance += (clean_a[col + k] - means_a[col]) *
                      (clean_b[row + k] - means_b[row]);
    }

    for (uint step = 0; step < diagonal_length; ++step, ++col, ++row) {
        bool excluded = apply_exclusion &&
            metal::abs(int(col) - int(row)) < int(exclusion);
        float norm_product = inv_norm_a[col] * inv_norm_b[row];
        if (!excluded && norm_product > 0.0f) {
            float corr = covariance * norm_product;
            if (metal::isfinite(corr)) {
                corr = metal::fmin(1.0f, metal::fmax(-1.0f, corr));
                uint bits = as_type<uint>(corr);
                uint key = (bits & 0x80000000u) ? ~bits :
                                                     (bits ^ 0x80000000u);
                atomic_fetch_max_explicit(
                    &best[col], key, memory_order_relaxed);
                if (self_join) {
                    atomic_fetch_max_explicit(
                        &best[row], key, memory_order_relaxed);
                }
            }
        }
        if (step + 1 < diagonal_length) {
            covariance += df_a[col] * dg_b[row] +
                          dg_a[col] * df_b[row];
        }
    }
"""

_INDEX_SOURCE = r"""
    uint diag_slot = thread_position_in_grid.x;
    uint n_a = means_a_shape[0];
    uint n_b = means_b_shape[0];
    uint m = config[0];
    uint exclusion = config[1];
    bool self_join = config[2] != 0;
    bool apply_exclusion = config[3] != 0;

    int diagonal;
    if (self_join) {
        diagonal = int(exclusion + diag_slot);
    } else {
        diagonal = int(diag_slot) - int(n_a - 1);
    }

    uint col = diagonal < 0 ? uint(-diagonal) : 0;
    uint row = diagonal > 0 ? uint(diagonal) : 0;
    if (self_join) {
        col = uint(diagonal);
        row = 0;
    }
    uint diagonal_length = metal::min(n_a - col, n_b - row);

    float covariance = 0.0f;
    for (uint k = 0; k < m; ++k) {
        covariance += (clean_a[col + k] - means_a[col]) *
                      (clean_b[row + k] - means_b[row]);
    }

    for (uint step = 0; step < diagonal_length; ++step, ++col, ++row) {
        bool excluded = apply_exclusion &&
            metal::abs(int(col) - int(row)) < int(exclusion);
        float norm_product = inv_norm_a[col] * inv_norm_b[row];
        if (!excluded && norm_product > 0.0f) {
            float corr = covariance * norm_product;
            if (metal::isfinite(corr)) {
                corr = metal::fmin(1.0f, metal::fmax(-1.0f, corr));
                uint bits = as_type<uint>(corr);
                uint key = (bits & 0x80000000u) ? ~bits :
                                                     (bits ^ 0x80000000u);
                if (best[col] == key) {
                    atomic_fetch_min_explicit(
                        &index[col], row, memory_order_relaxed);
                }
                if (self_join && best[row] == key) {
                    atomic_fetch_min_explicit(
                        &index[row], col, memory_order_relaxed);
                }
            }
        }
        if (step + 1 < diagonal_length) {
            covariance += df_a[col] * dg_b[row] +
                          dg_a[col] * df_b[row];
        }
    }
"""

_BIDIRECTIONAL_PROFILE_SOURCE = r"""
    uint diag_slot = thread_position_in_grid.x;
    uint n_a = means_a_shape[0];
    uint n_b = means_b_shape[0];
    uint m = config[0];
    uint exclusion = config[1];
    bool apply_exclusion = config[3] != 0;

    int diagonal = int(diag_slot) - int(n_a - 1);
    uint col = diagonal < 0 ? uint(-diagonal) : 0;
    uint row = diagonal > 0 ? uint(diagonal) : 0;
    uint diagonal_length = metal::min(n_a - col, n_b - row);

    float covariance = 0.0f;
    for (uint k = 0; k < m; ++k) {
        covariance += (clean_a[col + k] - means_a[col]) *
                      (clean_b[row + k] - means_b[row]);
    }

    for (uint step = 0; step < diagonal_length; ++step, ++col, ++row) {
        bool excluded = apply_exclusion &&
            metal::abs(int(col) - int(row)) < int(exclusion);
        float norm_product = inv_norm_a[col] * inv_norm_b[row];
        if (!excluded && norm_product > 0.0f) {
            float corr = covariance * norm_product;
            if (metal::isfinite(corr)) {
                corr = metal::fmin(1.0f, metal::fmax(-1.0f, corr));
                uint bits = as_type<uint>(corr);
                uint key = (bits & 0x80000000u) ? ~bits :
                                                     (bits ^ 0x80000000u);
                atomic_fetch_max_explicit(
                    &best_a[col], key, memory_order_relaxed);
                atomic_fetch_max_explicit(
                    &best_b[row], key, memory_order_relaxed);
            }
        }
        if (step + 1 < diagonal_length) {
            covariance += df_a[col] * dg_b[row] +
                          dg_a[col] * df_b[row];
        }
    }
"""

_BIDIRECTIONAL_INDEX_SOURCE = r"""
    uint diag_slot = thread_position_in_grid.x;
    uint n_a = means_a_shape[0];
    uint n_b = means_b_shape[0];
    uint m = config[0];
    uint exclusion = config[1];
    bool apply_exclusion = config[3] != 0;

    int diagonal = int(diag_slot) - int(n_a - 1);
    uint col = diagonal < 0 ? uint(-diagonal) : 0;
    uint row = diagonal > 0 ? uint(diagonal) : 0;
    uint diagonal_length = metal::min(n_a - col, n_b - row);

    float covariance = 0.0f;
    for (uint k = 0; k < m; ++k) {
        covariance += (clean_a[col + k] - means_a[col]) *
                      (clean_b[row + k] - means_b[row]);
    }

    for (uint step = 0; step < diagonal_length; ++step, ++col, ++row) {
        bool excluded = apply_exclusion &&
            metal::abs(int(col) - int(row)) < int(exclusion);
        float norm_product = inv_norm_a[col] * inv_norm_b[row];
        if (!excluded && norm_product > 0.0f) {
            float corr = covariance * norm_product;
            if (metal::isfinite(corr)) {
                corr = metal::fmin(1.0f, metal::fmax(-1.0f, corr));
                uint bits = as_type<uint>(corr);
                uint key = (bits & 0x80000000u) ? ~bits :
                                                     (bits ^ 0x80000000u);
                if (best_a[col] == key) {
                    atomic_fetch_min_explicit(
                        &index_a[col], row, memory_order_relaxed);
                }
                if (best_b[row] == key) {
                    atomic_fetch_min_explicit(
                        &index_b[row], col, memory_order_relaxed);
                }
            }
        }
        if (step + 1 < diagonal_length) {
            covariance += df_a[col] * dg_b[row] +
                          dg_a[col] * df_b[row];
        }
    }
"""


_PROFILE_KERNEL = mx.fast.metal_kernel(
    name="scamp_1nn_diagonal_profile",
    input_names=_INPUT_NAMES,
    output_names=["best"],
    source=_PROFILE_SOURCE,
    atomic_outputs=True,
)

_INDEX_KERNEL = mx.fast.metal_kernel(
    name="scamp_1nn_diagonal_index",
    input_names=[*_INPUT_NAMES, "best"],
    output_names=["index"],
    source=_INDEX_SOURCE,
    atomic_outputs=True,
)

_BIDIRECTIONAL_PROFILE_KERNEL = mx.fast.metal_kernel(
    name="scamp_1nn_bidirectional_profile",
    input_names=_INPUT_NAMES,
    output_names=["best_a", "best_b"],
    source=_BIDIRECTIONAL_PROFILE_SOURCE,
    atomic_outputs=True,
)

_BIDIRECTIONAL_INDEX_KERNEL = mx.fast.metal_kernel(
    name="scamp_1nn_bidirectional_index",
    input_names=[*_INPUT_NAMES, "best_a", "best_b"],
    output_names=["index_a", "index_b"],
    source=_BIDIRECTIONAL_INDEX_SOURCE,
    atomic_outputs=True,
)


def _ordered_key(value: float) -> int:
    bits = np.asarray(value, dtype=np.float32).view(np.uint32).item()
    if bits & 0x80000000:
        return int(~bits & 0xFFFFFFFF)
    return int(bits ^ 0x80000000)


def _decode_ordered_keys(keys: np.ndarray) -> np.ndarray:
    sign = np.uint32(0x80000000)
    bits = np.where(keys & sign, keys ^ sign, ~keys).astype(np.uint32)
    return bits.view(np.float32)


def _profile_state(
    prepared_a: Any,
    prepared_b: Any,
    m: int,
    self_join: bool,
    exclusion: int,
    *,
    keep_rows: bool = False,
) -> tuple[Any, list[Any], int, int, int]:
    """Launch the correlation pass and return its reusable device state."""

    n_a = prepared_a.subsequences
    n_b = prepared_b.subsequences
    diagonal_count = n_a - exclusion if self_join else n_a + n_b - 1
    if diagonal_count <= 0:
        return None, [], n_a, 0, 1

    config = mx.array(
        [m, exclusion, int(self_join), int(exclusion > 0)],
        dtype=mx.uint32,
    )
    inputs = [
        prepared_a.recurrence_clean,
        prepared_b.recurrence_clean,
        prepared_a.recurrence_means,
        prepared_b.recurrence_means,
        prepared_a.recurrence_inv_norm,
        prepared_b.recurrence_inv_norm,
        prepared_a.recurrence_df,
        prepared_b.recurrence_df,
        prepared_a.recurrence_dg,
        prepared_b.recurrence_dg,
        config,
    ]
    threadgroup_width = min(256, diagonal_count)
    kernel = _BIDIRECTIONAL_PROFILE_KERNEL if keep_rows else _PROFILE_KERNEL
    output_shapes = [(n_a,), (n_b,)] if keep_rows else [(n_a,)]
    best_state = kernel(
        inputs=inputs,
        output_shapes=output_shapes,
        output_dtypes=[mx.uint32] * len(output_shapes),
        grid=(diagonal_count, 1, 1),
        threadgroup=(threadgroup_width, 1, 1),
        init_value=_ordered_key(-2.0),
        stream=mx.gpu,
    )
    best = best_state if keep_rows else best_state[0]
    return best, inputs, n_a, diagonal_count, threadgroup_width


def best_profile(
    prepared_a: Any,
    prepared_b: Any,
    m: int,
    self_join: bool,
    exclusion: int,
) -> np.ndarray:
    """Compute an index-free float32 1NN profile on Metal."""

    best, _, n_a, _, _ = _profile_state(
        prepared_a, prepared_b, m, self_join, exclusion
    )
    if best is None:
        return np.full((n_a,), -2.0, dtype=np.float32)
    return _decode_ordered_keys(np.asarray(best, dtype=np.uint32)).copy()


def best_match(
    prepared_a: Any,
    prepared_b: Any,
    m: int,
    self_join: bool,
    exclusion: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute an indexed float32 1NN profile on Metal."""

    best, inputs, n_a, diagonal_count, threadgroup_width = _profile_state(
        prepared_a, prepared_b, m, self_join, exclusion
    )
    if best is None:
        return (
            np.full((n_a,), -2.0, dtype=np.float32),
            np.full((n_a,), -1, dtype=np.int32),
        )
    index = _INDEX_KERNEL(
        inputs=[*inputs, best],
        output_shapes=[(n_a,)],
        output_dtypes=[mx.uint32],
        grid=(diagonal_count, 1, 1),
        threadgroup=(threadgroup_width, 1, 1),
        init_value=0xFFFFFFFF,
        stream=mx.gpu,
    )[0]

    best_np = np.asarray(best, dtype=np.uint32)
    index_np = np.asarray(index, dtype=np.uint32)
    corr = _decode_ordered_keys(best_np).copy()
    idx = index_np.view(np.int32).copy()
    idx[corr < -1.0] = -1
    return corr, idx


def bidirectional_best_match(
    prepared_a: Any,
    prepared_b: Any,
    m: int,
    exclusion: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute both indexed axes of one float32 AB-join on Metal."""

    n_b = prepared_b.subsequences
    best, inputs, n_a, diagonal_count, threadgroup_width = _profile_state(
        prepared_a,
        prepared_b,
        m,
        False,
        exclusion,
        keep_rows=True,
    )
    if best is None:
        return (
            np.full((n_a,), -2.0, dtype=np.float32),
            np.full((n_a,), -1, dtype=np.int32),
            np.full((n_b,), -2.0, dtype=np.float32),
            np.full((n_b,), -1, dtype=np.int32),
        )

    best_a, best_b = best
    index_a, index_b = _BIDIRECTIONAL_INDEX_KERNEL(
        inputs=[*inputs, best_a, best_b],
        output_shapes=[(n_a,), (n_b,)],
        output_dtypes=[mx.uint32, mx.uint32],
        grid=(diagonal_count, 1, 1),
        threadgroup=(threadgroup_width, 1, 1),
        init_value=0xFFFFFFFF,
        stream=mx.gpu,
    )

    corr_a = _decode_ordered_keys(
        np.asarray(best_a, dtype=np.uint32)
    ).copy()
    corr_b = _decode_ordered_keys(
        np.asarray(best_b, dtype=np.uint32)
    ).copy()
    idx_a = np.asarray(index_a, dtype=np.uint32).view(np.int32).copy()
    idx_b = np.asarray(index_b, dtype=np.uint32).view(np.int32).copy()
    idx_a[corr_a < -1.0] = -1
    idx_b[corr_b < -1.0] = -1
    return corr_a, idx_a, corr_b, idx_b
