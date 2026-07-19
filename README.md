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
The native `mlx_native_scamp` namespace additionally exposes
`abjoin_bidirectional`, which computes both AB-join axes in one traversal for
SCAMP `keep_rows` consumers without changing the strict `pyscamp` surface.

## Native C++ library and CLI

The repository also contains an initial native C++ API compatible with
SCAMP's `SCAMPArgs`, `Profile`, and `do_SCAMP` concepts. It links directly to
MLX's C++ library and executes indexed or index-free 1NN self/AB joins with a
custom Metal diagonal-recurrence kernel; it does not launch or embed Python.

Top-level CMake builds also produce `mlx-scamp-native`, a native executable
linked to that library. Its distinct name avoids a case-insensitive APFS name
collision with the Python `scamp` entry point. The current CLI accepts the
implemented single-precision `1NN_INDEX` and `1NN` capabilities:

```bash
mlx-scamp-native \
  --window=128 \
  --input_a_file_name=series.txt \
  --single_precision \
  --output_a_file_name=profile.txt \
  --output_a_index_file_name=index.txt
```

It supports self joins, one- or two-sided AB joins (`--keep_rows`), Pearson or
default z-normalized Euclidean output, and aligned distributed offsets. Input
is whitespace-delimited text. Unsupported upstream reducers and execution
modes fail before input is read or output is touched; output sets are fully
staged, fsynced, and then committed with rollback protection.

Use `--profile_type=1NN` when match indexes are not needed. That path omits the
second Metal index-selection pass and writes only `--output_a_file_name` (plus
`--output_b_file_name` with `--keep_rows`); index-output flags are inactive.

See [`cpp/README.md`](cpp/README.md) for build instructions, current coverage,
and the remaining C++ parity work.

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

# Upstream-compatible upper bound on each time-series tile. Values must be
# at least 1024 samples and at least twice the subsequence window.
profile, index = mp.selfjoin(series, 128, max_tile_size=4096)
```

## Distributed MLX worker (experimental)

The macOS-native distributed runtime provides versioned gRPC workers,
capability and health discovery, and complete coordinator-side 1NN-index
self-joins and AB-joins. Install the optional transport dependencies and start
a loopback worker with:

```bash
python -m pip install '.[distributed]'
python -m mlx_native_scamp.distributed --backend auto
```

The coordinator automatically decomposes a join into memory-bounded tiles,
uses upper-triangle symmetry for self-joins, retries transient failures, and
assembles deterministic global profiles. The protocol uses compact float32
byte buffers for the single-precision MLX path instead of upstream's
repeated-double payloads. See
[Distributed MLX runtime](docs/distributed.md) for coordinator usage, protocol
details, and the remaining upstream distributed features. Non-loopback binds
require the explicit `--allow-insecure-remote` opt-in while TLS/authentication
remain follow-up work.

## Notes

- The compute-heavy matrix profile kernels are MLX-native on Apple Silicon.
- Single-precision `selfjoin`, `abjoin`, `selfjoin_sum`, and `abjoin_sum` use
  custom Metal kernels that follow SCAMP's rolling-covariance diagonal
  algorithm. Other profiles and execution modes use the portable MLX
  implementation.
- Eligible native float32 joins are translated by a finite per-series origin
  after quantization, then prepared with stable float64 CPU statistics. The
  rolling recurrence is vectorized in bounded NumPy blocks with high-accuracy
  checkpoints; cancellation-sensitive blocks automatically rerun the
  compensated scalar path. Only five linear-sized float32 recurrence arrays
  are sent to Metal; an `(subsequences, window)` normalized-window matrix is
  not constructed.
- The recurrence path is selected only when its conservative float32 bound is
  safe. Non-float32, unstable-precompute, and extreme-range inputs retain the
  existing portable normalized-window path.
- Portable reducers apply `max_tile_size` on both axes using upstream's
  time-series-sample units (`max_tile_size - window + 1` subsequences per
  axis). They normalize only the overlapping row and column segments for one
  tile, materialize reducer state with bounded MLX backpressure, and never
  construct the full normalized-window or pairwise-similarity matrix.
  Explicit values retain upstream's 1024-sample and `2 * window` minimums.
- KNN results order equal-correlation matches by the smallest global row index,
  keeping the selected rows deterministic when tile geometry changes.
- With no explicit limit, the upstream CPU/Metal default (128K/512K) is the
  enforced upper ceiling while an 8–64 MiB transient target is selected from
  Apple's recommended unified-memory working set. Windows larger than half
  that resource default require an explicit larger `max_tile_size`, matching
  upstream validation. The byte target is advisory because MLX controls
  allocator internals; the row/column dimensional ceilings are enforced.
  Linear input, recurrence, reducer, and output storage is outside this
  transient tile target.
- `mlx_native_scamp.selfjoin_1nn` and `abjoin_1nn` expose SCAMP's index-free
  `1NN` profile for native integrations. They use the same bounded tile
  ceilings as indexed 1NN while omitting portable index reduction and Metal's
  second index-selection pass. The strict `pyscamp` compatibility namespace
  remains limited to the upstream Python exports.
- The 1NN kernel keeps profile output state linear in the series length, but it
  currently walks each diagonal in one Metal dispatch. A `max_tile_size`
  ceiling that cannot contain the join selects the bounded portable path;
  checkpointed multi-dispatch execution in the custom Metal kernel remains a
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
