# MLX-native-SCAMP

`MLX-native-SCAMP` is an Apple Silicon port of SCAMP's Python-facing API built on top of Apple MLX.

The package provides a `pyscamp`-compatible import surface for the most common callable API:

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

- The compute-heavy matrix profile kernels are MLX-native.
- Compatibility kwargs like `gpus`, `threads`, and `precision` are accepted for API compatibility, but CUDA-specific behavior is not reproduced.
- The implementation currently targets dense 1D numeric arrays.
