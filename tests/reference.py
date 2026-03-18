import math
import numpy as np


def moving_mean(a, w):
    out = np.zeros((len(a) - w + 1,), dtype=np.float64)
    for i in range(len(out)):
        out[i] = np.mean(a[i : i + w], dtype=np.float64)
    return out


def sum_of_squared_differences(a, means, w):
    out = np.zeros((len(a) - w + 1,), dtype=np.float64)
    for i in range(len(out)):
        vals = a[i : i + w] - means[i]
        out[i] = np.sum(vals * vals, dtype=np.float64)
    return out


def get_precomputes(T, m, nanvalues):
    n = len(T) - m + 1
    means = moving_mean(T, m)
    norms = sum_of_squared_differences(T, means, m)
    for i in range(n):
        if nanvalues[i] or norms[i] <= 1e-13:
            norms[i] = np.nan
        else:
            norms[i] = 1.0 / math.sqrt(norms[i])
    return means, norms


def convert_non_finite_to_zero(T, m):
    clean = np.zeros((len(T),), dtype=np.float64)
    nanvals = np.zeros((len(T) - m + 1,), dtype=bool)
    steps_since_last_nan = m
    for i in range(len(T)):
        if np.isfinite(T[i]):
            clean[i] = T[i]
        else:
            steps_since_last_nan = 0
            clean[i] = 0.0
        if i >= m - 1:
            nanvals[i - m + 1] = steps_since_last_nan < m
        steps_since_last_nan += 1
    return clean, nanvals


def distance_matrix(a, b, m):
    has_b = b is not None
    if not has_b:
        b = a
    a, nan_a = convert_non_finite_to_zero(np.asarray(a, dtype=np.float64), m)
    b, nan_b = convert_non_finite_to_zero(np.asarray(b, dtype=np.float64), m)
    mua, siga = get_precomputes(a, m, nan_a)
    mub, sigb = get_precomputes(b, m, nan_b)

    na = len(a) - m + 1
    nb = len(b) - m + 1
    out = np.ones((nb, na), dtype=np.float64) * -2.0
    minlag = m // 4 if not has_b else 0

    for row in range(nb):
        y = b[row : row + m]
        for col in range(na):
            if not has_b and abs(row - col) < minlag:
                continue
            if np.isnan(siga[col]) or np.isnan(sigb[row]):
                continue
            x = a[col : col + m]
            corr = np.dot(x - mua[col], y - mub[row]) * siga[col] * sigb[row]
            out[row, col] = corr
    return out


def reduce_1nn_index(dm):
    corr = np.amax(dm, axis=0)
    idxs = np.argmax(dm, axis=0)
    idxs[corr == -2] = -1
    corr[corr == -2] = np.nan
    return corr.astype(np.float32), idxs.astype(np.int32)


def reduce_sum_thresh(dm, thresh):
    dm2 = np.copy(dm)
    dm2[dm2 <= thresh] = 0
    return np.sum(dm2, dtype=np.float64, axis=0)


def reduce_matrix(dm_orig, rows, cols, self_join):
    dm = np.copy(dm_orig)
    if self_join:
        for col in range(dm.shape[1]):
            if col + 1 >= dm.shape[0]:
                break
            dm[col + 1 :, col] = np.nan
    reduced_rows = dm.shape[0] / rows
    reduced_cols = dm.shape[1] / cols
    out = np.ones((rows, cols), dtype=np.float32) * -2
    for r in range(rows):
        st_r = math.ceil(r * reduced_rows)
        ed_r = min(dm.shape[0], math.ceil((r + 1) * reduced_rows))
        for c in range(cols):
            st_c = math.ceil(c * reduced_cols)
            ed_c = min(dm.shape[1], math.ceil((c + 1) * reduced_cols))
            out[r, c] = np.nanmax(dm[st_r:ed_r, st_c:ed_c])
    out[out == -2] = np.nan
    return out
