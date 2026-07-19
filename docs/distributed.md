# Distributed MLX runtime

This experimental runtime is an independently usable part of SCAMP's
distributed client/server architecture ported to macOS and MLX. It runs
complete 1NN-index self-joins and AB-joins across a pool of Macs. It is not yet
a replacement for upstream's persistent, multi-profile job server.

Upstream uses a central gRPC job server and workers that repeatedly pull tiles.
Version 1 exposes each MLX worker as a gRPC service and lets an in-process
coordinator push bounded tiles. This makes worker health and Apple hardware
capabilities discoverable before scheduling and removes the always-running,
Kubernetes-oriented job server from local Mac deployments.

## What is implemented

- A versioned Protobuf/gRPC service. Every request declares API version 1 and
  incompatible peers fail with `FAILED_PRECONDITION`.
- Apple Silicon and MLX capability discovery, including Metal/CPU backend,
  MLX version, unified memory, recommended working set, and supported profile
  and precision modes. Workers also advertise and enforce a conservative
  per-tile working-set limit before constructing a dense similarity matrix.
- Worker health and execution counters.
- A real rectangular 1NN-index tile operation. A is the column dimension and B
  is the row dimension, matching upstream SCAMP's distributed convention.
  Results contain global indices and both row and column partial profiles, so
  they can be merged without rewriting tile-local indices.
- A `WorkerPool` that tolerates unreachable peers, checks health, and
  dispatches tiles round-robin with failover to another serving worker.
- A `DistributedCoordinator` that decomposes complete self-joins and AB-joins,
  keeps only a bounded number of requests in flight, merges results as they
  arrive, and returns global profiles in the same Euclidean or Pearson
  representation as upstream SCAMP.
- Upper-triangle self-join scheduling. Off-diagonal tiles compute both profile
  directions and represent their transposes, while diagonal blocks are
  computed once. A `g` by `g` grid therefore needs `g * (g + 1) / 2` tiles
  instead of `g * g`.
- Optional AB row profiles corresponding to upstream's `keep_rows` behavior.
- Per-tile retry with bounded exponential backoff after all live workers fail,
  plus monotonic progress snapshots containing completed tiles, retry count,
  elapsed time, and ETA.
- Deterministic merging. Equal correlations select the lower non-negative
  global index regardless of worker completion order. Correlations are clamped
  to Pearson's valid range before argmax to stabilize perfect short-window
  matches across tile shapes.
- Compact little-endian float32 payloads for the current single-precision path.
  These halve input bandwidth relative to upstream's repeated `double` series
  fields, include only the samples needed by each tile, and decode directly
  into NumPy/MLX-compatible buffers.

The worker defaults to one concurrent tile. This avoids multiple large Metal
allocations competing for the same unified-memory working set. gRPC handling
still uses multiple threads, so health and discovery remain responsive during
compute. The coordinator defaults to one in-flight tile per serving worker.

Automatic planning uses the smallest message and working-set limits advertised
by the serving workers, capped by a conservative 256 MiB coordinator budget.
The estimate includes the dense correlation matrix, masks, and
normalized-window intermediates. Set
`max_tile_bytes` on `DistributedCoordinator` to lower that limit, or pass an
explicit `tile_size`; unsafe explicit sizes fail before work is dispatched.

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
it to an untrusted network. A non-loopback bind is rejected unless
`--allow-insecure-remote` is supplied explicitly. TLS and authentication remain
future work.

The worker derives its default tile limit from one quarter of Apple's reported
recommended working set (bounded between 64 MiB and 4 GiB), or uses a 1 GiB
fallback when MLX cannot report memory. Override it with
`--max-tile-working-set-bytes`; requests over the limit fail with
`RESOURCE_EXHAUSTED` before MLX allocation.

Workers on other Macs must bind a reachable interface with explicit opt-in,
for example `--host 0.0.0.0 --allow-insecure-remote`. Use that only on a
trusted, firewalled private network until TLS and authentication are
implemented.

## Run a complete join

Start one or more workers, then give their targets to a pool:

```python
import numpy as np

from mlx_native_scamp.distributed import DistributedCoordinator, WorkerPool

rng = np.random.default_rng(7)
a = rng.random(100_000, dtype=np.float32)
b = rng.random(80_000, dtype=np.float32)

events = []
with WorkerPool(["mac-studio.local:30078", "macbook.local:30078"]) as workers:
    coordinator = DistributedCoordinator(workers)
    self_result = coordinator.selfjoin(
        a,
        128,
        pearson=True,
        progress=events.append,
    )
    ab_result = coordinator.abjoin(
        a,
        b,
        128,
        keep_rows=True,
    )

self_correlations = self_result.column_values
self_global_indices = self_result.column_indices
ab_distances = ab_result.column_values
b_distances = ab_result.row_values
print(events[-1].fraction, events[-1].eta_seconds)
```

Like `pyscamp`, output values are Euclidean distances by default; pass
`pearson=True` for correlations. Global indices remain signed 64-bit values so
large distributed jobs are not limited to 32-bit offsets. Invalid or flat
windows produce `NaN` and index `-1`.

For a normal AB-join, only the A/column profile is computed. `keep_rows=True`
also computes the B/row profile in the same tile traversal. A regular AB-join
has no exclusion zone; self-joins use SCAMP's `ceil(window / 4)` zone. Advanced
AB workloads whose subsequence coordinates share the same origin may pass a
nonzero `exclusion_zone` explicitly.

Transient RPC failures are first failed over within `WorkerPool`, then retried
up to three times by the coordinator. Protocol, authentication, permission,
and invalid-request errors fail immediately. A retry reuses the request ID, so
workers and logs can associate attempts with the same tile.

## Execute one tile directly

The versioned tile API remains available for custom schedulers:

```python
import numpy as np

from mlx_native_scamp.distributed import WorkerPool, make_tile_request
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

When managing tiles manually, combine partial profiles with
`merge_1nn_slices(results, profile_length)`. The reducer uses each result's
global offset and resolves exactly equal values by the lower global match
index. Bounds are subsequence bounds, not raw time-series sample bounds.

## Upstream relationship and remaining work

This runtime covers the useful core of upstream workers' `ExecuteWork` and the
complete in-process lifecycle for 1NN-index jobs: plan tiles, execute on the
available accelerators, retry and fail over, combine global profiles, and
report progress. Capability discovery and health are additions needed for
heterogeneous Mac workers; upstream workers advertise only an estimated
throughput number.

The following upstream distributed behavior is not claimed yet:

- persistent job submission, status polling, cancellation, and later result
  retrieval independent of the coordinating process;
- SUM, matrix-summary, KNN, multidimensional, and double-precision tile
  reducers;
- scheduling weighted by measured worker throughput;
- streaming or content-addressed input distribution to avoid resending samples
  shared by adjacent tiles;
- TLS, authentication, and deployment tooling for remote Mac fleets.

Version 1's single-precision request builder converts raw samples to float32.
This matches its advertised compute mode but can erase small variations riding
on very large offsets before normalization. Integrating the safe double-input
preprocessing used by the local high-offset path remains required before this
transport can claim that edge-case parity with upstream SCAMP.

The versioned tile messages and global result offsets support those additions
without breaking this worker API.
