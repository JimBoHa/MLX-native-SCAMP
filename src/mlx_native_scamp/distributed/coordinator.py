"""Coordinator reducers for independently executed profile tiles."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from ._version import API_VERSION
from .codec import decode_array
from .proto import scamp_worker_v1_pb2 as messages


def merge_1nn_slices(
    results: Iterable[messages.ProfileTileResult],
    profile_length: int,
    *,
    direction: str = "column",
) -> tuple[np.ndarray, np.ndarray]:
    """Merge partial 1NN tile profiles using their global offsets.

    Equal correlations use the lower non-negative global index, making the
    reduction deterministic regardless of worker completion order.
    """

    if profile_length < 0:
        raise ValueError("profile_length cannot be negative")
    if direction not in {"column", "row"}:
        raise ValueError("direction must be 'column' or 'row'")

    best_values = np.full(profile_length, -2.0, dtype=np.float32)
    best_indices = np.full(profile_length, -1, dtype=np.int64)
    for result in results:
        if result.api_version != API_VERSION:
            raise ValueError(
                f"cannot merge distributed API version {result.api_version} "
                f"with version {API_VERSION}"
            )
        profile = result.column_profile if direction == "column" else result.row_profile
        values = decode_array(profile.values)
        indices = decode_array(profile.indices)
        if values.dtype != np.dtype("<f4") or indices.dtype != np.dtype("<i8"):
            raise ValueError(
                "1NN profile slices must contain float32 values and int64 indices"
            )
        if values.size != indices.size:
            raise ValueError("profile value and index slices have different lengths")
        valid_correlations = np.isfinite(values) & (values >= -1.0) & (values <= 1.0)
        sentinels = values == -2.0
        if not np.all(valid_correlations | sentinels):
            raise ValueError("1NN profile values must be finite correlations or -2")
        if np.any(sentinels & (indices != -1)) or np.any(
            valid_correlations & (indices < 0)
        ):
            raise ValueError("1NN profile values and indices are inconsistent")
        start = int(profile.offset)
        stop = start + values.size
        if start < 0 or stop > profile_length:
            raise ValueError("profile slice is outside the destination profile")

        current_values = best_values[start:stop]
        current_indices = best_indices[start:stop]
        better = values > current_values
        tied_lower_index = (
            (values == current_values)
            & (indices >= 0)
            & ((current_indices < 0) | (indices < current_indices))
        )
        update = better | tied_lower_index
        best_values[start:stop] = np.where(update, values, current_values)
        best_indices[start:stop] = np.where(update, indices, current_indices)

    return best_values, best_indices


__all__ = ["merge_1nn_slices"]
