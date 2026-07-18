from __future__ import annotations

from typing import Any

import mlx.core as mx
import numpy as np

from ._metal_1nn import _decode_ordered_keys, _ordered_key


_SOURCE = r"""
    uint diag_slot = thread_position_in_grid.x;
    uint n_a = means_a_shape[0];
    uint n_b = means_b_shape[0];
    uint m = config[0];
    uint exclusion = config[1];
    bool self_join = config[2] != 0;
    uint matrix_rows = config[3];
    uint matrix_cols = config[4];

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
        float norm_product = inv_norm_a[col] * inv_norm_b[row];
        if (norm_product > 0.0f) {
            float corr = covariance * norm_product;
            if (metal::isfinite(corr)) {
                corr = metal::fmin(1.0f, metal::fmax(-1.0f, corr));
                uint row_bin = metal::min(
                    matrix_rows - 1, (row * matrix_rows) / n_b);
                uint col_bin = metal::min(
                    matrix_cols - 1, (col * matrix_cols) / n_a);
                uint cell = row_bin * matrix_cols + col_bin;
                uint bits = as_type<uint>(corr);
                uint key = (bits & 0x80000000u) ? ~bits :
                                                     (bits ^ 0x80000000u);
                atomic_fetch_max_explicit(
                    &summary[cell], key, memory_order_relaxed);
            }
        }
        if (step + 1 < diagonal_length) {
            covariance += df_a[col] * dg_b[row] +
                          dg_a[col] * df_b[row];
        }
    }
"""


_KERNEL = mx.fast.metal_kernel(
    name="scamp_matrix_summary_diagonal",
    input_names=[
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
    ],
    output_names=["summary"],
    source=_SOURCE,
    atomic_outputs=True,
)


def matrix_summary(
    prepared_a: Any,
    prepared_b: Any,
    m: int,
    rows: int,
    cols: int,
    self_join: bool,
) -> np.ndarray:
    """Compute a pooled correlation matrix with SCAMP's recurrence."""

    n_a = prepared_a.subsequences
    n_b = prepared_b.subsequences
    exclusion = (m + 3) // 4 if self_join else 0
    diagonal_count = n_a - exclusion if self_join else n_a + n_b - 1
    if diagonal_count <= 0:
        return np.full((rows, cols), -2.0, dtype=np.float32)

    config = mx.array(
        [m, exclusion, int(self_join), rows, cols],
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
    keys = _KERNEL(
        inputs=inputs,
        output_shapes=[(rows * cols,)],
        output_dtypes=[mx.uint32],
        grid=(diagonal_count, 1, 1),
        threadgroup=(threadgroup_width, 1, 1),
        init_value=_ordered_key(-2.0),
        stream=mx.gpu,
    )[0]
    values = _decode_ordered_keys(np.asarray(keys, dtype=np.uint32)).copy()
    return values.reshape(rows, cols)
