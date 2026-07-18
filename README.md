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
profile, index = mp.selfjoin(series, 128, pearson=True)
```

## Notes

- The compute-heavy matrix profile kernels are MLX-native on Apple Silicon.
- Inputs can be NumPy arrays, Python sequences, or MLX arrays.
- `gpus=[]` or a positive `threads` value selects MLX CPU execution; `gpus=[0]`
  selects the Metal GPU and takes precedence if both are supplied. With neither,
  the current MLX default device is preserved.
- MLX manages its own CPU scheduler, so `threads` selects CPU execution but cannot
  enforce an exact worker count.
- Multi-GPU requests and GPU IDs other than `0` are unsupported and rejected.
- Concurrent CPU+GPU workers are not exposed by MLX; when both selectors are
  supplied, this port deliberately uses Metal rather than claiming SCAMP's
  heterogeneous-worker behavior.
- Other compatibility kwargs such as `precision` are accepted, but CUDA-specific
  behavior is not reproduced.
- The implementation currently targets dense 1D numeric arrays.
