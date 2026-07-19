from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _distribution_version

from .core import (
    abjoin,
    abjoin_1nn,
    abjoin_knn,
    abjoin_matrix,
    abjoin_sum,
    gpu_supported,
    selfjoin,
    selfjoin_1nn,
    selfjoin_knn,
    selfjoin_matrix,
    selfjoin_sum,
)


def _resolve_version() -> str:
    try:
        return _distribution_version("mlx-native-scamp")
    except _PackageNotFoundError:
        return "dev"


__version__ = _resolve_version()

__all__ = [
    "__version__",
    "abjoin",
    "abjoin_1nn",
    "abjoin_knn",
    "abjoin_matrix",
    "abjoin_sum",
    "gpu_supported",
    "selfjoin",
    "selfjoin_1nn",
    "selfjoin_knn",
    "selfjoin_matrix",
    "selfjoin_sum",
]
