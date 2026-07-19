from __future__ import annotations

from typing import Any

import mlx.core as mx
import numpy as np

from ._exclusion import self_join_exclusion


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
    "threshold",
    "config",
]
DIAGONALS_PER_PARTIAL = 2048
RECURRENCE_CHECKPOINT = 64

_SOURCE = r"""
    uint diag_slot = config[3] + thread_position_in_grid.x;
    uint n_a = means_a_shape[0];
    uint n_b = means_b_shape[0];
    uint m = config[0];
    uint exclusion = config[1];
    bool self_join = config[2] != 0;

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

    uint steps_until_checkpoint = config[4];
    for (uint step = 0; step < diagonal_length; ++step, ++col, ++row) {
        float norm_product = inv_norm_a[col] * inv_norm_b[row];
        if (norm_product > 0.0f) {
            float corr = covariance * norm_product;
            if (metal::isfinite(corr)) {
                corr = metal::fmin(1.0f, metal::fmax(-1.0f, corr));
                if (corr > threshold[0]) {
                    uint old_bits = atomic_load_explicit(
                        &sums[col], memory_order_relaxed);
                    while (true) {
                        float new_value = as_type<float>(old_bits) + corr;
                        uint expected = old_bits;
                        if (atomic_compare_exchange_weak_explicit(
                                &sums[col], &expected, as_type<uint>(new_value),
                                memory_order_relaxed, memory_order_relaxed)) {
                            break;
                        }
                        old_bits = expected;
                    }
                    if (self_join) {
                        old_bits = atomic_load_explicit(
                            &sums[row], memory_order_relaxed);
                        while (true) {
                            float new_value = as_type<float>(old_bits) + corr;
                            uint expected = old_bits;
                            if (atomic_compare_exchange_weak_explicit(
                                    &sums[row], &expected,
                                    as_type<uint>(new_value),
                                    memory_order_relaxed,
                                    memory_order_relaxed)) {
                                break;
                            }
                            old_bits = expected;
                        }
                    }
                }
            }
        }
        if (step + 1 < diagonal_length) {
            --steps_until_checkpoint;
            if (steps_until_checkpoint == 0) {
                covariance = 0.0f;
                for (uint k = 0; k < m; ++k) {
                    covariance +=
                        (clean_a[col + 1 + k] - means_a[col + 1]) *
                        (clean_b[row + 1 + k] - means_b[row + 1]);
                }
                steps_until_checkpoint = config[4];
            } else {
                covariance += df_a[col] * dg_b[row] +
                              dg_a[col] * df_b[row];
            }
        }
    }
"""


_SUM_KERNEL = mx.fast.metal_kernel(
    name="scamp_sum_threshold_diagonal",
    input_names=_INPUT_NAMES,
    output_names=["sums"],
    source=_SOURCE,
    atomic_outputs=True,
)


def sum_threshold(
    prepared_a: Any,
    prepared_b: Any,
    m: int,
    threshold: float,
    self_join: bool,
) -> np.ndarray:
    """Compute a single-precision SUM_THRESH profile by diagonal recurrence."""

    n_a = prepared_a.subsequences
    n_b = prepared_b.subsequences
    exclusion = self_join_exclusion(m) if self_join else 0
    diagonal_count = n_a - exclusion if self_join else n_a + n_b - 1
    if diagonal_count <= 0:
        return np.zeros((n_a,), dtype=np.float64)

    threshold_array = mx.array([threshold], dtype=mx.float32)
    fixed_inputs = [
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
        threshold_array,
    ]
    result = np.zeros((n_a,), dtype=np.float64)
    for diagonal_start in range(0, diagonal_count, DIAGONALS_PER_PARTIAL):
        batch_count = min(DIAGONALS_PER_PARTIAL, diagonal_count - diagonal_start)
        config = mx.array(
            [
                m,
                exclusion,
                int(self_join),
                diagonal_start,
                RECURRENCE_CHECKPOINT,
            ],
            dtype=mx.uint32,
        )
        threadgroup_width = min(256, batch_count)
        sums = _SUM_KERNEL(
            inputs=[*fixed_inputs, config],
            output_shapes=[(n_a,)],
            output_dtypes=[mx.uint32],
            grid=(batch_count, 1, 1),
            threadgroup=(threadgroup_width, 1, 1),
            init_value=0,
            stream=mx.gpu,
        )[0]
        sum_bits = np.asarray(sums, dtype=np.uint32)
        result += sum_bits.view(np.float32)
    return result
