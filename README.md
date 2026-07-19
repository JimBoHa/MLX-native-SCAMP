# MLX-native-SCAMP

`MLX-native-SCAMP` is an Apple Silicon port of SCAMP's Python-facing API built on top of Apple MLX.

The package provides an MLX-native `pyscamp`-compatible import surface for the full upstream Python API:

- `gpu_supported`
- `selfjoin`
- `abjoin`
- `selfjoin_sum`
- `abjoin_sum`
- `selfjoin_matrix`
- `abjoin_matrix`
- `selfjoin_knn`
- `abjoin_knn`

The implementation is pure Python plus MLX and is meant to be called by other apps without requiring CUDA.
All nine upstream `pyscamp` callables are implemented in the local MLX engine rather than delegated back to CUDA SCAMP.

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

## Notes

- The compute-heavy matrix profile kernels are MLX-native on Apple Silicon.
- Single-precision `selfjoin`, `abjoin`, `selfjoin_sum`, and `abjoin_sum` use
  custom Metal kernels that follow SCAMP's rolling-covariance diagonal
  algorithm. Other profiles and execution modes use the portable MLX
  implementation.
- Eligible native float32 joins are translated by a finite per-series origin
  after quantization, then prepared with compensated float64 CPU statistics.
  Only five linear-sized float32 recurrence arrays are sent to Metal; an
  `(subsequences, window)` normalized-window matrix is not constructed.
- The recurrence path is selected only when its conservative float32 bound is
  safe. Non-float32, unstable-precompute, and extreme-range inputs retain the
  existing portable normalized-window path.
- The 1NN kernel keeps profile output state linear in the series length, but it
  currently walks each diagonal in one Metal dispatch. Checkpointed/tiled
  dispatch—and integration with the separate `max_tile_size` work—remains a
  follow-up for exceptionally long joins.
- For nonnegative thresholds, sufficiently large SUM workloads use a sparse
  Metal reducer. It refreshes covariance directly every 64 diagonal steps,
  bounds float32 atomic accumulation to 2,048 diagonals at a time, and merges
  partial profiles in float64 on CPU. A bounded correlation sample estimates
  atomic-update density; density, window size, pair count, join shape, and join
  type select conservative benchmarked crossovers. Smaller, short-window,
  dense, highly rectangular, and negative-threshold joins retain the portable
  reducer because atomics or short diagonals can be slower than MLX matrix
  multiplication. Correlations within float32
  roundoff of a strict threshold can fall on either side; use double precision
  on CPU when that boundary must be stable. Double and ultra precision remain
  entirely on MLX CPU.
- Inputs can be NumPy arrays, Python sequences, or MLX arrays.
- `gpus=[]` or a positive `threads` value selects MLX CPU execution; `gpus=[0]`
  selects the Metal GPU. With neither, the current MLX default device is
  preserved.
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
