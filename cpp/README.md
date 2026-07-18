# Native C++ API

This directory is the first native C++ parity slice for SCAMP on Apple
Silicon. The `MLXNativeSCAMP::scamp` CMake target links directly to the C++
headers and `libmlx` shipped in Apple's MLX wheel. The resulting library does
not start a Python process or embed a Python interpreter.

## Current coverage

The public headers preserve the upstream namespace and core concepts:

- `SCAMP::SCAMPArgs`, `SCAMPProfileType`, and `SCAMPPrecisionType`
- `SCAMP::Profile` and upstream's packed `mp_entry` representation
- `SCAMP::do_SCAMP(args)` and the explicit resource overload
- `SCAMP::num_available_gpus()` (the retained name reports MLX Metal devices)

Both the new `<scamp/...>` include layout and upstream-compatible
`<common/scamp_args.h>`, `<common/profile.h>`, and
`<common/scamp_interface.h>` forwarding headers are installed.

The implemented operation is `PROFILE_TYPE_1NN_INDEX` with
`PRECISION_SINGLE` on Metal device 0. It covers:

- self joins with SCAMP's `ceil(window / 4)` exclusion zone;
- one-sided AB joins producing `profile_a`;
- two-sided AB joins using `computing_rows` and `keep_rows_separate` to also
  produce `profile_b`;
- aligned AB exclusion, including distributed row/column offsets; and
- non-finite and flat subsequences, which produce correlation `-2` and index
  `-1` when no valid match exists.

The compute path uses SCAMP's diagonal rolling-covariance recurrence in two
custom Metal passes. The first atomically reduces correlations, and the second
selects the smallest exact index for ties. Compensated CPU statistics and each
diagonal's initial covariance are computed in double precision before the
single-precision Metal recurrence, preserving variation in high-offset input.
Device/output storage is linear in the input lengths; no dense distance matrix
or normalized window matrix is materialized.

## Build and test

MLX 0.23.2 or newer is required. Its Python wheel includes the C++ development
package under `site-packages/mlx` (`include/`, `lib/libmlx.dylib`, and
`share/cmake/MLX`). Configure with that package explicitly:

```bash
MLX_CMAKE_DIR="$(.venv/bin/python -c \
  'import mlx.core, pathlib; print(pathlib.Path(mlx.core.__file__).parent / "share/cmake/MLX")')"

cmake -S . -B build/cpp \
  -DCMAKE_BUILD_TYPE=Release \
  -DMLX_DIR="$MLX_CMAKE_DIR"
cmake --build build/cpp --parallel
ctest --test-dir build/cpp --output-on-failure
./build/cpp/mlx_scamp_cpp_example
```

The default target is static. Add `-DBUILD_SHARED_LIBS=ON` to produce the
native arm64 `libmlx_scamp.dylib` instead.

When MLX is already visible to CMake through `MLX_DIR`, `MLX_ROOT`, or
`CMAKE_PREFIX_PATH`, no Python executable is involved even during configure.
Otherwise CMake can use an installed Python interpreter solely to locate the
MLX development package. Runtime execution remains entirely native C++/MLX.
Set `MLX_SCAMP_PYTHON_EXECUTABLE` if the desired interpreter is not the active
virtual environment or the first `python3` on `PATH`.

Consumers can use the build-tree target directly:

```cmake
add_subdirectory(path/to/MLX-native-SCAMP)
target_link_libraries(my_app PRIVATE MLXNativeSCAMP::scamp)
```

or install the exported `MLXNativeSCAMPConfig.cmake` package with
`cmake --install`.

## Explicit expansion gaps

Unsupported requests throw `SCAMPException`; they are not routed through a
slower implementation or silently ignored. The remaining C++ API work is:

- `PROFILE_TYPE_1NN`, `SUM_THRESH`, `APPROX_ALL_NEIGHBORS`, and
  `MATRIX_SUMMARY` reducers;
- `PRECISION_DOUBLE` and `PRECISION_ULTRA` CPU paths (Metal has no float64);
- native CPU workers and heterogeneous CPU/Metal scheduling;
- bounded multi-dispatch tiling for very long joins (`max_tile_size` is
  currently validated for argument compatibility, while this kernel retains
  linear storage and executes one join-wide dispatch);
- integration with the macOS CLI and distributed worker runtime; and
- multiple accelerators or CUDA device management, which do not map to current
  Apple Silicon hardware.
