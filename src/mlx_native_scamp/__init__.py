from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _distribution_version

from .core import (
    abjoin,
    abjoin_knn,
    abjoin_matrix,
    abjoin_sum,
    gpu_supported,
    selfjoin,
    selfjoin_knn,
    selfjoin_matrix,
    selfjoin_sum,
)

try:
    __version__ = _distribution_version("mlx-native-scamp")
except _PackageNotFoundError:
    __version__ = "dev"

__all__ = [
    "__version__",
    "abjoin",
    "abjoin_knn",
    "abjoin_matrix",
    "abjoin_sum",
    "gpu_supported",
    "selfjoin",
    "selfjoin_knn",
    "selfjoin_matrix",
    "selfjoin_sum",
]
