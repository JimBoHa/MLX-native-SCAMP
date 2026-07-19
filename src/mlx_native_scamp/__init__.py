from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _distribution_version

from ._autotune import (
    AutotunePlan,
    AutotuneWorkload,
    CandidateMeasurement,
    StrategyDescription,
    autotune,
    autotune_plan,
    run_autotune,
    strategy_descriptions,
)
from ._autotune_cache import cache_status as autotune_status
from ._autotune_cache import reset_cache as reset_autotune
from .core import (
    abjoin,
    abjoin_1nn,
    abjoin_bidirectional,
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
    "abjoin_bidirectional",
    "abjoin_knn",
    "abjoin_matrix",
    "abjoin_sum",
    "AutotunePlan",
    "AutotuneWorkload",
    "CandidateMeasurement",
    "StrategyDescription",
    "autotune",
    "autotune_plan",
    "autotune_status",
    "gpu_supported",
    "reset_autotune",
    "run_autotune",
    "selfjoin",
    "selfjoin_1nn",
    "selfjoin_knn",
    "selfjoin_matrix",
    "selfjoin_sum",
    "strategy_descriptions",
]
