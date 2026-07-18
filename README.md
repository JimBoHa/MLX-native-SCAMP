# MLX-native-SCAMP

`MLX-native-SCAMP` is an Apple Silicon port of SCAMP's Python-facing API built on top of Apple MLX.

The package provides an MLX-native `pyscamp`-compatible import surface for the
full SCAMP 4.0.3 Python API:

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
All nine SCAMP 4.0.3 `pyscamp` callables are implemented in the local MLX
engine rather than delegated back to CUDA SCAMP.

## Compatibility target

The compatibility baseline is the nine-callable Python API in the official
[SCAMP 4.0.3 release](https://github.com/zpzim/SCAMP/releases/tag/v4.0.3).
Additions found only on the current upstream `master` branch are tracked as
separate compatibility work and are not part of this baseline unless explicitly
documented here.

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
- Compatibility kwargs like `gpus`, `threads`, and `precision` are accepted for API compatibility, but CUDA-specific behavior is not reproduced.
- The implementation currently targets dense 1D numeric arrays.
