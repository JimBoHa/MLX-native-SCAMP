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

### Command line

The package installs both `scamp` and `mlx-scamp` entry points. They accept the
upstream SCAMP command-line names while running the local MLX engine:

```bash
scamp --window=128 --input_a_file_name=series.txt
scamp --window=128 --input_a_file_name=a.txt --input_b_file_name=b.txt \
  --profile_type=ALL_NEIGHBORS --max_matches_per_column=5 \
  --threshold=0.5 --single_precision
```

Inputs use SCAMP's whitespace-delimited ASCII format. `-` may be used as one
input to read from stdin or as one output to write to stdout. Progress messages
from `--print_debug_info` go to stderr so stdout remains machine-readable.
Active input/output paths are checked for canonical, hard-link, case-only, and
Unicode-normalization aliases before computation; file outputs replace their
targets atomically only after a complete write.

All upstream profile modes are available: `1NN_INDEX`, `1NN`, `SUM_THRESH`,
`ALL_NEIGHBORS`, and `MATRIX_SUMMARY`. For an AB-join, `--keep_rows` computes
the reverse profile with the same MLX engine and writes the configured B
outputs. Apple Silicon exposes one Metal device, ID `0`; `--no_gpu` or a
positive `--num_cpu_workers` selects the MLX CPU path, while `--gpus=0` selects
Metal. MLX does not provide CUDA-style heterogeneous CPU/GPU or multi-GPU
execution. `--max_tile_size` is forwarded to the MLX tiler when explicitly
set; otherwise the engine chooses an Apple unified-memory-aware tile size.
The `single` precision path can use Metal; `double` (the upstream default) and
`ultra` use MLX's CPU backend because Apple Metal does not expose float64.

Current upstream SCAMP's `--autotune` and `--list_variants` flags are also
recognized. `--autotune` uses the separately packaged MLX tuner when it is
installed; `--list_variants` reports MLX execution strategies instead of CUDA
kernel geometry.

Distributed global offsets and self-join left/right `--keep_rows` profiles are
rejected with capability errors until their corresponding MLX APIs are
available. This avoids silently producing a profile with different semantics.

## Notes

- The compute-heavy matrix profile kernels are MLX-native on Apple Silicon.
- Inputs can be NumPy arrays, Python sequences, or MLX arrays.
- Compatibility kwargs like `gpus`, `threads`, and `precision` are accepted for API compatibility, but CUDA-specific behavior is not reproduced.
- The implementation currently targets dense 1D numeric arrays.
