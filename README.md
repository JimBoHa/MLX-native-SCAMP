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
- `autotune`

The implementation is pure Python plus MLX and is meant to be called by other apps without requiring CUDA.
All ten upstream `pyscamp` callables are implemented locally rather than
delegated back to CUDA SCAMP. The native `mlx_native_scamp` namespace also
exposes index-free 1NN, one-pass bidirectional AB joins, and typed autotune
planning/status controls without expanding the strict `pyscamp` surface.

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

# Report the route that actually ran after cache selection and safety fallbacks.
profile, index = mp.selfjoin(series, 128, verbose=True)
```

## Apple/MLX autotuning

Autotuning is explicit: importing the package and running an untuned join do
not launch benchmarks. The upstream-compatible entry point runs a bounded
quick plan covering self-joins and AB-joins for all five upstream profile
families in single and double precision:

```python
import pyscamp

pyscamp.autotune()
```

Native integrations can inspect the plan, opt into the larger asymmetric plan
with aligned AB and bidirectional coverage, inspect cache status, or reset only
the MLX records:

```python
import mlx_native_scamp as mlx_scamp

plan = mlx_scamp.autotune_plan("quick")
mlx_scamp.run_autotune(mode="full")
print(mlx_scamp.autotune_status())
mlx_scamp.reset_autotune()
```

Results are stored in a versioned JSON sidecar next to SCAMP's cache (for
example, `autotune.txt.mlx.json`). The upstream `SCAMP_AUTOTUNE_V1` file is
never read as MLX JSON or overwritten. Records include the profile family,
precision, self/AB shape, work and window buckets, aligned mode, profile
knobs, tile regime, Apple hardware, macOS, MLX version, and candidate-manifest
revision. Malformed or stale records are ignored by joins; explicit tuning
refuses to overwrite an unreadable or newer-format sidecar.

Only joins with implicit resources consult the cache. Explicit `gpus`,
`threads`, and `max_tile_size` choices retain authority, and cached candidates
are rechecked by the existing precision, float32 recurrence, density, matrix,
and memory-ceiling gates. A cache miss preserves the normal MLX default route.

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
- Single-precision matrix summaries whose inputs fit the active
  `max_tile_size` ceiling also use the diagonal recurrence on Metal, atomically
  reducing directly into the requested pooled matrix instead of materializing
  similarity blocks. Oversized, CPU, and higher-precision summaries retain the
  bounded portable implementation.
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
- Cached autotune winners can choose MLX CPU, portable MLX Metal, or an
  eligible custom Metal reducer for the matching workload bucket. Portable
  row choices remain clamped by the adaptive 8–64 MiB scheduler and the
  explicit/default `max_tile_size` ceiling.
- `gpus=[]` or a positive `threads` value selects MLX CPU execution; `gpus=[0]`
  selects the Metal GPU. With neither and no matching cached winner, the
  current MLX default device is preserved.
- `verbose=True` emits one resolved start event and one synchronized completion
  event (or error event), correlated by operation ID. It reports the actual CPU,
  portable Metal, or custom Metal implementation, compute dtype, work shape,
  effective tile ceiling, and portable tile geometry after every fallback gate.
  The default path performs no reporting, formatting, timing, or extra
  synchronization.
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
