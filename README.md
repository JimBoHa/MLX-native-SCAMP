# MLX-native-SCAMP

`MLX-native-SCAMP` is an Apple Silicon port of SCAMP's Python-facing API built on top of Apple MLX.

The package provides an MLX-native `pyscamp`-compatible import surface for the full upstream Python API:

- `gpu_supported`
- `autotune`
- `selfjoin`
- `abjoin`
- `selfjoin_sum`
- `abjoin_sum`
- `selfjoin_matrix`
- `abjoin_matrix`
- `selfjoin_knn`
- `abjoin_knn`

The implementation is pure Python plus MLX and is meant to be called by other apps without requiring CUDA.
All ten upstream `pyscamp` callables are implemented in the local MLX engine rather than delegated back to CUDA SCAMP.

## Install

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .
```

## Usage

```python
import numpy as np
import pyscamp as mp

series = np.random.random(4096).astype(np.float32)
profile, index = mp.selfjoin(series, 128, pearson=True, precision="single")
```

## Apple/MLX autotuning

Run the autotuner once to measure this Mac's CPU and Metal paths and persist
the best safe launch choices:

```python
import pyscamp

pyscamp.autotune()
```

The signature and return value match upstream SCAMP:
`autotune(devices=None, cache_path="") -> int`. MLX exposes one Metal GPU, so
the default and `devices=[]` tune device `0`; an explicit list may contain only
device `0`. Each sweep verifies candidate results against the portable CPU path
before timing and selects:

- MLX CPU versus the custom Metal diagonal kernel for default
  single-precision 1NN joins, plus a separate CPU-versus-Metal choice for
  profiles using the portable reducer path;
- portable CPU and Metal reducer block sizes from `64`, `128`, `256`, and
  `512` rows; and
- a Metal diagonal-kernel threadgroup width from `32`, `64`, `128`, and `256`.

The default deterministic workload has 4,096 samples, one warmup, and three
timed trials per candidate. `SCAMP_AUTOTUNE_INPUT_LENGTH`,
`SCAMP_AUTOTUNE_WARMUP_RUNS`, and `MLX_SCAMP_AUTOTUNE_TRIALS` can adjust these
values for shorter exploratory or larger production-oriented tuning runs.

Like upstream, the cache path resolves through `SCAMP_AUTOTUNE_CACHE`, then
`XDG_CACHE_HOME/scamp/autotune.txt`, then `~/.cache/scamp/autotune.txt`.
Writes are atomic and records are keyed by the Apple GPU identity, unified
memory size, machine architecture, MLX version, and MLX kernel revision.
CUDA SCAMP and this port intentionally use different cache formats; choose a
separate `cache_path` or `SCAMP_AUTOTUNE_CACHE` value if both installations
share a home directory. A custom `cache_path` is reused by normal joins only
when `SCAMP_AUTOTUNE_CACHE` points to it.

## Notes

- The compute-heavy matrix profile kernels are MLX-native on Apple Silicon.
- Single-precision `selfjoin` and `abjoin` use a custom Metal kernel that
  follows SCAMP's rolling-covariance diagonal algorithm. Other profiles and
  execution modes use the portable MLX implementation.
- The diagonal kernel is selected only for native float32 inputs whose raw
  rolling covariance is safe in float32. Non-float32 and extreme-magnitude
  inputs retain the normalized-window path so precision and overflow fixes can
  be applied there without being bypassed.
- The 1NN kernel keeps profile output state linear in the series length, but it
  currently walks each diagonal in one Metal dispatch. Checkpointed/tiled
  dispatch—and integration with the separate `max_tile_size` work—remains a
  follow-up for exceptionally long joins.
- Inputs can be NumPy arrays, Python sequences, or MLX arrays.
- `gpus=[]` or a positive `threads` value selects MLX CPU execution; `gpus=[0]`
  selects the Metal GPU. With neither, the current MLX default device is
  preserved until an autotune record selects the faster single-precision
  backend for this Mac.
- Explicit `gpus` and `threads` selectors override the autotuned backend. The
  cached block or threadgroup choice still optimizes the explicitly selected
  device.
- MLX manages its own CPU scheduler, so `threads` selects CPU execution but cannot
  enforce an exact worker count.
- Multi-GPU requests and GPU IDs other than `0` are unsupported and rejected.
- Concurrent CPU+GPU workers are not exposed by MLX, so supplying both selectors
  is rejected rather than silently ignoring one or claiming SCAMP's
  heterogeneous-worker behavior.
- `precision="single"` computes in float32 on the selected MLX device (normally Metal).
- The default `precision="double"` and `precision="ultra"` share a float64
  MLX CPU implementation because Metal does not support float64. Upstream
  SCAMP gives `ultra` a separate recurrence; this port computes normalized
  window dot products directly, so the two modes currently have identical
  numerical behavior.
- Explicit `gpus=[0]` requests with `double` or `ultra` are rejected instead of
  silently moving float64 work to CPU.
- The implementation currently targets dense 1D numeric arrays.
