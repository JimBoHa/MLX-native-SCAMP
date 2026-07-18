# MLX-native-SCAMP

`MLX-native-SCAMP` is an Apple Silicon port of the Python-facing API from
[zpzim/SCAMP](https://github.com/zpzim/SCAMP), built on Apple MLX.

The project targets the `pyscamp` Python surface. It does not currently port
SCAMP's C++ CLI, distributed client/server runtime, or CUDA kernel-management
tools. Computation that maps to Apple hardware is implemented with MLX and
Metal semantics instead of emulating NVIDIA devices.

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

The implementation uses Python and NumPy-compatible I/O around an MLX compute
engine and is meant to be called by other apps without requiring CUDA.
All nine SCAMP 4.0.3 `pyscamp` callables are implemented in the local MLX
engine rather than delegated back to CUDA SCAMP.

## Compatibility target

The compatibility baseline is the nine-callable Python API in the official
[SCAMP 4.0.3 release](https://github.com/zpzim/SCAMP/releases/tag/v4.0.3).
Compatible additions from current upstream SCAMP are tracked as separate work
until they are merged and documented here.

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

## Upstream and citation

This port builds on the SCAMP algorithm and public interface maintained by
[Zachary Zimmerman and the SCAMP contributors](https://github.com/zpzim/SCAMP).
Users should retain the upstream attribution and cite the SCAMP work when
appropriate:

Zimmerman, Zachary, et al. “Matrix Profile XIV: Scaling Time Series Motif
Discovery with GPUs to Break a Quintillion Pairwise Comparisons a Day and
Beyond.” *Proceedings of the ACM Symposium on Cloud Computing*, 2019.
