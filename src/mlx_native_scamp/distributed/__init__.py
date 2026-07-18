"""Optional distributed coordinator and MLX worker support.

Install ``mlx-native-scamp[distributed]`` before accessing these names. The
main ``pyscamp`` API does not import gRPC or Protobuf.
"""

from __future__ import annotations

from importlib import import_module


_RUNTIME_EXPORTS = {
    "API_VERSION",
    "DEFAULT_MAX_MESSAGE_BYTES",
    "WorkerClient",
    "WorkerPool",
    "WorkerServer",
    "WorkerSnapshot",
    "make_tile_request",
    "messages",
    "merge_1nn_slices",
}


def __getattr__(name: str):
    if name not in _RUNTIME_EXPORTS:
        raise AttributeError(name)
    module = ".coordinator" if name == "merge_1nn_slices" else ".runtime"
    value = getattr(import_module(module, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | _RUNTIME_EXPORTS)


__all__ = sorted(_RUNTIME_EXPORTS)
