# Distributed MLX runtime

This experimental runtime is the first independently usable part of SCAMP's
distributed client/server architecture ported to macOS and MLX. It is not yet
a replacement for upstream's complete job server.

Upstream uses a central gRPC job server and workers that repeatedly pull tiles.
Version 1 deliberately exposes each MLX worker as a gRPC service and lets the
coordinator push bounded tiles. This makes worker health and Apple hardware
capabilities discoverable before scheduling and removes the always-running
Kubernetes-oriented job server from local Mac deployments.

## What is implemented

- A versioned Protobuf/gRPC service. Every request declares API version 1 and
  incompatible peers fail with `FAILED_PRECONDITION`.
- Apple Silicon and MLX capability discovery, including Metal/CPU backend,
  MLX version, unified memory, recommended working set, and supported profile
  and precision modes.
- Worker health and execution counters.
- A real rectangular 1NN-index tile operation. A is the column dimension and B
  is the row dimension, matching upstream SCAMP's distributed tile convention.
  Results contain global indices and both row and column partial profiles, so
  they can be merged without rewriting tile-local indices.
- A coordinator-side `WorkerPool` that discovers workers, checks health, and
  dispatches individual tiles round-robin with failover to another serving
  worker.
- A deterministic 1NN slice reducer that combines tiles by global offset and
  produces the same result regardless of worker completion order.
- Compact little-endian float32 payloads for the current single-precision path.
  These halve input bandwidth relative to upstream's repeated `double` series
  fields, include only the samples needed by each tile, and decode directly
  into NumPy/MLX-compatible buffers.

The worker defaults to one concurrent tile. This avoids multiple large Metal
allocations competing for the same unified-memory working set. gRPC handling
still uses multiple threads, so health and discovery remain responsive during
compute.

## Run a worker

Install the optional dependencies:

```bash
python -m pip install '.[distributed]'
```

Start a worker using Metal automatically on Apple Silicon:

```bash
python -m mlx_native_scamp.distributed --host 127.0.0.1 --port 30078
```

`--backend cpu` explicitly selects the MLX CPU device. `--backend metal`
requires Metal instead of falling back. The default loopback bind is deliberate:
version 1 uses the same insecure gRPC transport as upstream SCAMP. Do not expose
it to an untrusted network. TLS and authentication remain future work.

## Discover and execute

```python
import numpy as np

from mlx_native_scamp.distributed import (
    WorkerPool,
    make_tile_request,
    merge_1nn_slices,
)
from mlx_native_scamp.distributed.codec import decode_array

a = np.random.default_rng(7).random(4096, dtype=np.float32)
request = make_tile_request(
    a,
    window=128,
    row_start=0,
    row_stop=1024,
    column_start=1024,
    column_stop=2048,
    compute_rows=True,
)

with WorkerPool(["127.0.0.1:30078"]) as workers:
    snapshot = workers.discover()[0]
    print(snapshot.capabilities.device_name, snapshot.health.message)
    result = workers.execute_tile(request)

column_correlations = decode_array(result.column_profile.values)
column_indices = decode_array(result.column_profile.indices)
```

When a column range is split across multiple row tiles, merge the returned
partial profiles with `merge_1nn_slices(results, profile_length)`. The reducer
uses each result's global offset and deterministically resolves equal values by
the lower global match index.

For self-joins, `make_tile_request` supplies SCAMP's `ceil(window / 4)`
exclusion zone by default. Passing a B series creates an AB-join and defaults
the exclusion zone to zero. Bounds are subsequence bounds, not raw time-series
sample bounds.

## Upstream relationship and remaining work

This slice covers the useful core of upstream workers' `ExecuteWork`: receive a
tile, execute it on the available accelerator, and return mergeable partial
profiles. Capability discovery and health are additions needed for heterogeneous
Mac workers; upstream workers advertise only an estimated throughput number.

The following upstream distributed behavior is not claimed yet:

- job submission, status polling, and final-result retrieval;
- automatic decomposition of complete joins into tiles;
- global profile combining, retry queues, failure requeue, progress, and ETA;
- SUM, matrix-summary, KNN, multidimensional, and double-precision tile reducers;
- scheduling weighted by measured worker throughput;
- streaming or content-addressed input distribution to avoid resending samples
  shared by adjacent tiles;
- TLS/authentication and deployment tooling for remote Mac fleets.

The versioned tile messages and global result offsets are intended to support
those additions without breaking this worker API.
