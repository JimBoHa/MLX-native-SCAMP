"""Optional distributed coordinator and MLX worker support.

Install ``mlx-native-scamp[distributed]`` before accessing these names. The
main ``pyscamp`` API does not import gRPC or Protobuf.
"""

from __future__ import annotations

from importlib import import_module


_RUNTIME_EXPORTS = {
    "API_VERSION",
    "DEFAULT_MAX_RETRIES",
    "DEFAULT_MAX_MESSAGE_BYTES",
    "DEFAULT_MAX_TILE_WORKING_SET_BYTES",
    "DEFAULT_MAX_TILE_BYTES",
    "Distributed1NNResult",
    "DistributedCoordinator",
    "JobProgress",
    "ProfileTile",
    "ProfileTilePlan",
    "WorkerClient",
    "WorkerPool",
    "WorkerServer",
    "WorkerSnapshot",
    "make_tile_request",
    "messages",
    "estimate_tile_working_set_bytes",
    "merge_1nn_slices",
    "plan_1nn_tiles",
}

_COORDINATOR_EXPORTS = {
    "DEFAULT_MAX_RETRIES",
    "DEFAULT_MAX_TILE_BYTES",
    "Distributed1NNResult",
    "DistributedCoordinator",
    "JobProgress",
    "ProfileTile",
    "ProfileTilePlan",
    "estimate_tile_working_set_bytes",
    "merge_1nn_slices",
    "plan_1nn_tiles",
}


def __getattr__(name: str):
    if name not in _RUNTIME_EXPORTS:
        raise AttributeError(name)
    module = ".coordinator" if name in _COORDINATOR_EXPORTS else ".runtime"
    value = getattr(import_module(module, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | _RUNTIME_EXPORTS)


__all__ = sorted(_RUNTIME_EXPORTS)
