"""SCAMP-compatible matrix-profile functions implemented with Apple MLX.

This namespace mirrors the public Python API from upstream SCAMP. Import
``mlx_native_scamp`` for native-only index-free, bidirectional, and detailed
autotune controls, or ``mlx_native_scamp.distributed`` for the optional gRPC
runtime.
"""

from mlx_native_scamp import (
    __version__,
    abjoin,
    abjoin_knn,
    abjoin_matrix,
    abjoin_sum,
    autotune,
    gpu_supported,
    selfjoin,
    selfjoin_knn,
    selfjoin_matrix,
    selfjoin_sum,
)

__all__ = [
    "__version__",
    "abjoin",
    "abjoin_knn",
    "abjoin_matrix",
    "abjoin_sum",
    "autotune",
    "gpu_supported",
    "selfjoin",
    "selfjoin_knn",
    "selfjoin_matrix",
    "selfjoin_sum",
]
