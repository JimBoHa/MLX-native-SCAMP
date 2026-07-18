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

## Distributed MLX worker (experimental)

The first macOS-native distributed slice provides a versioned gRPC worker,
capability and health discovery, and remote execution of rectangular 1NN
profile tiles. Install the optional transport dependencies and start a
loopback worker with:

```bash
python -m pip install '.[distributed]'
python -m mlx_native_scamp.distributed --backend auto
```

The protocol uses compact float32 byte buffers for the single-precision MLX
path instead of upstream's repeated-double payloads. See
[Distributed MLX runtime](docs/distributed.md) for coordinator usage, protocol
details, and the remaining upstream distributed features. Non-loopback binds
require the explicit `--allow-insecure-remote` opt-in while TLS/authentication
remain follow-up work.

## Notes

- The compute-heavy matrix profile kernels are MLX-native on Apple Silicon.
- Inputs can be NumPy arrays, Python sequences, or MLX arrays.
- Compatibility kwargs like `gpus`, `threads`, and `precision` are accepted for API compatibility, but CUDA-specific behavior is not reproduced.
- The implementation currently targets dense 1D numeric arrays.
