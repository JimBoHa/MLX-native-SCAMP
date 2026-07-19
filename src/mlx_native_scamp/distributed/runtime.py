"""Versioned gRPC coordinator and Apple/MLX worker runtime."""

from __future__ import annotations

import os
import ipaddress
import platform
import socket
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import Iterable

try:
    import grpc
except ModuleNotFoundError as error:  # pragma: no cover - exercised without the extra
    raise ImportError(
        "Distributed SCAMP requires the optional dependencies; "
        "install mlx-native-scamp[distributed]."
    ) from error

import mlx.core as mx
import numpy as np

from ._version import API_VERSION
from .codec import decode_array, encode_array, is_empty_payload
from .execution import (
    _MLX_STREAM_LOCK,
    estimate_1nn_tile_working_set_bytes,
    execute_1nn_tile,
)
from .proto import scamp_worker_v1_pb2 as messages
from .proto import scamp_worker_v1_pb2_grpc as services


DEFAULT_MAX_MESSAGE_BYTES = 256 * 1024 * 1024
DEFAULT_MAX_TILE_WORKING_SET_BYTES = 1024 * 1024 * 1024
_TRANSIENT_RPC_CODES = {
    grpc.StatusCode.ABORTED,
    grpc.StatusCode.DEADLINE_EXCEEDED,
    grpc.StatusCode.RESOURCE_EXHAUSTED,
    grpc.StatusCode.UNAVAILABLE,
}


def _is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine().lower() in {
        "arm64",
        "aarch64",
    }


def _mlx_version() -> str:
    try:
        return version("mlx")
    except PackageNotFoundError:  # pragma: no cover - MLX is a required dependency
        return "unknown"


def _physical_memory_bytes() -> int:
    try:
        return int(os.sysconf("SC_PHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, OSError, TypeError, ValueError):  # pragma: no cover
        return 0


def _is_loopback_host(host: str) -> bool:
    if host.casefold() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host.strip("[]")).is_loopback
    except ValueError:
        return False


def _default_tile_working_set_bytes() -> int:
    try:
        device_info = mx.device_info()
    except Exception:  # pragma: no cover - old MLX fallback
        device_info = {}
    available = int(device_info.get("max_recommended_working_set_size", 0))
    if not available:
        available = int(device_info.get("memory_size", 0)) or _physical_memory_bytes()
    if not available:
        return DEFAULT_MAX_TILE_WORKING_SET_BYTES
    return max(64 * 1024 * 1024, min(available // 4, 4 * 1024 * 1024 * 1024))


def _select_device(backend: str):
    requested = backend.lower()
    if requested not in {"auto", "metal", "cpu"}:
        raise ValueError("backend must be one of: auto, metal, cpu")
    use_metal = requested == "metal" or (requested == "auto" and _is_apple_silicon())
    device = mx.Device(mx.gpu if use_metal else mx.cpu, 0)
    try:
        with _MLX_STREAM_LOCK:
            with mx.stream(device):
                probe = mx.array([1.0], dtype=mx.float32) + 1.0
                mx.eval(probe)
    except Exception as error:
        if requested == "auto" and use_metal:
            device = mx.Device(mx.cpu, 0)
            use_metal = False
        else:
            raise RuntimeError(f"MLX {requested} backend is unavailable") from error
    return (
        device,
        messages.DEVICE_BACKEND_METAL if use_metal else messages.DEVICE_BACKEND_CPU,
    )


@dataclass(frozen=True, slots=True)
class WorkerSnapshot:
    """Coordinator view of one discovered worker."""

    target: str
    capabilities: messages.WorkerCapabilities
    health: messages.WorkerHealth


class _WorkerService(services.ScampWorkerServicer):
    def __init__(
        self,
        *,
        worker_id: str,
        backend: str,
        max_message_bytes: int,
        max_concurrent_tiles: int,
        max_tile_working_set_bytes: int | None,
    ) -> None:
        self.worker_id = worker_id
        self.device, self.backend = _select_device(backend)
        self.max_message_bytes = max_message_bytes
        self.max_tile_working_set_bytes = (
            _default_tile_working_set_bytes()
            if max_tile_working_set_bytes is None
            else max_tile_working_set_bytes
        )
        self._started_ns = time.monotonic_ns()
        self._state = messages.WORKER_STATE_NOT_SERVING
        self._active = 0
        self._completed = 0
        self._failed = 0
        self._counter_lock = threading.Lock()
        self._tile_slots = threading.BoundedSemaphore(max_concurrent_tiles)

    def set_serving(self, serving: bool) -> None:
        self._state = (
            messages.WORKER_STATE_SERVING
            if serving
            else messages.WORKER_STATE_NOT_SERVING
        )

    def _check_version(self, api_version: int, context) -> None:
        if api_version != API_VERSION:
            context.abort(
                grpc.StatusCode.FAILED_PRECONDITION,
                f"unsupported distributed API version {api_version}; worker supports {API_VERSION}",
            )

    def _capabilities(self) -> messages.WorkerCapabilities:
        try:
            device_info = mx.device_info()
        except Exception:  # pragma: no cover - old MLX fallback
            device_info = {}
        apple_silicon = _is_apple_silicon()
        unified_memory_bytes = int(device_info.get("memory_size", 0))
        if apple_silicon and not unified_memory_bytes:
            unified_memory_bytes = _physical_memory_bytes()
        return messages.WorkerCapabilities(
            api_version=API_VERSION,
            worker_id=self.worker_id,
            hostname=socket.gethostname(),
            operating_system=platform.platform(),
            machine=platform.machine(),
            apple_silicon=apple_silicon,
            mlx_version=_mlx_version(),
            backend=self.backend,
            device_name=str(
                device_info.get(
                    "device_name",
                    "Apple Silicon Metal" if apple_silicon else self.device,
                )
            ),
            unified_memory_bytes=unified_memory_bytes if apple_silicon else 0,
            recommended_working_set_bytes=int(
                device_info.get("max_recommended_working_set_size", 0)
            ),
            logical_cpu_count=os.cpu_count() or 1,
            profile_kinds=[messages.PROFILE_KIND_1NN_INDEX],
            precisions=[messages.PRECISION_SINGLE],
            max_message_bytes=self.max_message_bytes,
            max_tile_working_set_bytes=self.max_tile_working_set_bytes,
        )

    def _health(self) -> messages.WorkerHealth:
        with self._counter_lock:
            active = self._active
            completed = self._completed
            failed = self._failed
        return messages.WorkerHealth(
            api_version=API_VERSION,
            worker_id=self.worker_id,
            state=self._state,
            uptime_millis=(time.monotonic_ns() - self._started_ns) // 1_000_000,
            active_requests=active,
            completed_requests=completed,
            failed_requests=failed,
            message="ready"
            if self._state == messages.WORKER_STATE_SERVING
            else "stopping",
        )

    def GetCapabilities(self, request, context):
        self._check_version(request.api_version, context)
        return self._capabilities()

    def CheckHealth(self, request, context):
        self._check_version(request.api_version, context)
        return self._health()

    def ExecuteProfileTile(self, request, context):
        self._check_version(request.api_version, context)
        if self._state != messages.WORKER_STATE_SERVING:
            context.abort(grpc.StatusCode.UNAVAILABLE, "worker is not serving")
        if request.profile_kind != messages.PROFILE_KIND_1NN_INDEX:
            context.abort(
                grpc.StatusCode.UNIMPLEMENTED, "worker only supports 1NN index tiles"
            )
        if request.precision != messages.PRECISION_SINGLE:
            context.abort(
                grpc.StatusCode.UNIMPLEMENTED, "worker only supports single precision"
            )
        rows = int(request.row_stop) - int(request.row_start)
        columns = int(request.column_stop) - int(request.column_start)
        try:
            estimated_bytes = estimate_1nn_tile_working_set_bytes(
                rows, columns, int(request.window)
            )
        except ValueError as error:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(error))
        if estimated_bytes > self.max_tile_working_set_bytes:
            context.abort(
                grpc.StatusCode.RESOURCE_EXHAUSTED,
                f"estimated tile working set {estimated_bytes} bytes exceeds "
                f"worker limit {self.max_tile_working_set_bytes} bytes",
            )
        if not self._tile_slots.acquire(blocking=False):
            context.abort(
                grpc.StatusCode.RESOURCE_EXHAUSTED,
                "worker has no free tile execution slot",
            )

        with self._counter_lock:
            self._active += 1
        started_ns = time.perf_counter_ns()
        succeeded = False
        try:
            series_a = decode_array(request.series_a)
            if series_a.dtype not in (np.dtype("<f4"), np.dtype("<f8")):
                raise ValueError("series_a must use float32 or float64")
            if request.self_join:
                series_b = (
                    None
                    if is_empty_payload(request.series_b)
                    else decode_array(request.series_b)
                )
                if series_b is not None and series_b.dtype not in (
                    np.dtype("<f4"),
                    np.dtype("<f8"),
                ):
                    raise ValueError("series_b must use float32 or float64")
            else:
                if is_empty_payload(request.series_b):
                    raise ValueError("series_b is required for an AB-join tile")
                series_b = decode_array(request.series_b)
                if series_b.dtype not in (np.dtype("<f4"), np.dtype("<f8")):
                    raise ValueError("series_b must use float32 or float64")

            output = execute_1nn_tile(
                series_a,
                series_b,
                int(request.window),
                row_start=int(request.row_start),
                row_stop=int(request.row_stop),
                column_start=int(request.column_start),
                column_stop=int(request.column_stop),
                exclusion_zone=int(request.exclusion_zone),
                compute_rows=request.compute_rows,
                compute_columns=request.compute_columns,
                device=self.device,
                series_a_offset=int(request.series_a_offset),
                series_b_offset=int(request.series_b_offset),
            )
            elapsed_ns = time.perf_counter_ns() - started_ns
            result = messages.ProfileTileResult(
                api_version=API_VERSION,
                request_id=request.request_id,
                worker_id=self.worker_id,
                column_profile=messages.ProfileSlice(
                    offset=request.column_start,
                    values=encode_array(output.column_values),
                    indices=encode_array(output.column_indices),
                ),
                row_profile=messages.ProfileSlice(
                    offset=request.row_start,
                    values=encode_array(output.row_values),
                    indices=encode_array(output.row_indices),
                ),
                execution_nanos=elapsed_ns,
            )
            succeeded = True
            return result
        except ValueError as error:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(error))
        except MemoryError:
            context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED, "tile allocation failed")
        except Exception as error:
            context.abort(
                grpc.StatusCode.INTERNAL,
                f"tile execution failed: {type(error).__name__}",
            )
        finally:
            with self._counter_lock:
                self._active -= 1
                if succeeded:
                    self._completed += 1
                else:
                    self._failed += 1
            self._tile_slots.release()


class WorkerServer:
    """Lifecycle wrapper for one MLX gRPC worker.

    The default loopback bind is suitable for one Mac. Remote deployments are
    intentionally opt-in because this first protocol version uses insecure
    gRPC, matching upstream SCAMP's transport.
    """

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        backend: str = "auto",
        worker_id: str | None = None,
        rpc_threads: int = 4,
        max_concurrent_tiles: int = 1,
        max_message_bytes: int = DEFAULT_MAX_MESSAGE_BYTES,
        max_tile_working_set_bytes: int | None = None,
        allow_insecure_remote: bool = False,
    ) -> None:
        if rpc_threads <= 0 or max_concurrent_tiles <= 0:
            raise ValueError("worker thread and concurrency counts must be positive")
        if max_message_bytes <= 0:
            raise ValueError("max_message_bytes must be positive")
        if max_tile_working_set_bytes is not None and max_tile_working_set_bytes <= 0:
            raise ValueError("max_tile_working_set_bytes must be positive")
        if not _is_loopback_host(host) and not allow_insecure_remote:
            raise ValueError(
                "non-loopback workers require allow_insecure_remote=True until "
                "TLS/authentication support is configured"
            )
        self.host = host
        self.requested_port = port
        self.max_message_bytes = max_message_bytes
        resolved_id = worker_id or (
            f"{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex[:8]}"
        )
        self.service = _WorkerService(
            worker_id=resolved_id,
            backend=backend,
            max_message_bytes=max_message_bytes,
            max_concurrent_tiles=max_concurrent_tiles,
            max_tile_working_set_bytes=max_tile_working_set_bytes,
        )
        options = (
            ("grpc.max_receive_message_length", max_message_bytes),
            ("grpc.max_send_message_length", max_message_bytes),
        )
        self._server = grpc.server(
            ThreadPoolExecutor(max_workers=rpc_threads), options=options
        )
        services.add_ScampWorkerServicer_to_server(self.service, self._server)
        self.port = self._server.add_insecure_port(f"{host}:{port}")
        if not self.port:
            raise RuntimeError(f"could not bind distributed worker to {host}:{port}")
        self.target = f"{host}:{self.port}"
        self._started = False
        self._stopped = False

    def start(self) -> "WorkerServer":
        if self._stopped:
            raise RuntimeError("a stopped gRPC worker cannot be restarted")
        if not self._started:
            self._server.start()
            self.service.set_serving(True)
            self._started = True
        return self

    def stop(self, grace: float = 5.0) -> None:
        if self._started:
            self.service.set_serving(False)
            self._server.stop(grace).wait(timeout=grace + 1.0)
            self._started = False
            self._stopped = True

    def wait(self) -> None:
        self._server.wait_for_termination()

    def __enter__(self) -> "WorkerServer":
        return self.start()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.stop()


class WorkerClient:
    """Coordinator-side client for one version-1 worker."""

    def __init__(
        self,
        target: str,
        *,
        timeout: float = 10.0,
        max_message_bytes: int = DEFAULT_MAX_MESSAGE_BYTES,
    ) -> None:
        self.target = target
        self.timeout = timeout
        options = (
            ("grpc.max_receive_message_length", max_message_bytes),
            ("grpc.max_send_message_length", max_message_bytes),
        )
        self.channel = grpc.insecure_channel(target, options=options)
        self.stub = services.ScampWorkerStub(self.channel)

    def wait_ready(self, timeout: float | None = None) -> None:
        grpc.channel_ready_future(self.channel).result(timeout=timeout or self.timeout)

    def capabilities(self) -> messages.WorkerCapabilities:
        return self.stub.GetCapabilities(
            messages.VersionRequest(api_version=API_VERSION),
            timeout=self.timeout,
            wait_for_ready=True,
        )

    def health(self) -> messages.WorkerHealth:
        return self.stub.CheckHealth(
            messages.VersionRequest(api_version=API_VERSION),
            timeout=self.timeout,
            wait_for_ready=True,
        )

    def execute_tile(
        self, request: messages.ProfileTileRequest
    ) -> messages.ProfileTileResult:
        result = self.stub.ExecuteProfileTile(
            request,
            timeout=self.timeout,
            wait_for_ready=True,
        )
        if result.api_version != API_VERSION:
            raise ValueError(
                f"worker returned distributed API version {result.api_version}; "
                f"expected {API_VERSION}"
            )
        if result.request_id != request.request_id:
            raise ValueError("worker response request_id does not match the request")
        return result

    def close(self) -> None:
        self.channel.close()

    def __enter__(self) -> "WorkerClient":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


class WorkerPool:
    """Small coordinator that discovers workers and schedules single tiles."""

    def __init__(self, targets: Iterable[str], *, timeout: float = 10.0) -> None:
        self._clients = tuple(
            WorkerClient(target, timeout=timeout) for target in targets
        )
        if not self._clients:
            raise ValueError("at least one worker target is required")
        self._next_worker = 0
        self._lock = threading.Lock()
        self._preferred_targets: frozenset[str] | None = None

    def discover(self) -> tuple[WorkerSnapshot, ...]:
        def inspect(client: WorkerClient) -> WorkerSnapshot:
            return WorkerSnapshot(client.target, client.capabilities(), client.health())

        with ThreadPoolExecutor(max_workers=len(self._clients)) as executor:
            return tuple(executor.map(inspect, self._clients))

    def discover_serving(self) -> tuple[WorkerSnapshot, ...]:
        """Return healthy serving workers while tolerating unreachable peers.

        A complete job can continue when one configured Mac is offline. If no
        worker can be inspected, the last gRPC error is preserved so callers
        retain its status and diagnostic details.
        """

        def inspect(client: WorkerClient):
            try:
                snapshot = WorkerSnapshot(
                    client.target, client.capabilities(), client.health()
                )
                return snapshot, None
            except grpc.RpcError as error:
                return None, error

        with ThreadPoolExecutor(max_workers=len(self._clients)) as executor:
            inspected = tuple(executor.map(inspect, self._clients))
        serving = tuple(
            snapshot
            for snapshot, _ in inspected
            if snapshot is not None
            and snapshot.health.state == messages.WORKER_STATE_SERVING
        )
        if serving:
            with self._lock:
                self._preferred_targets = frozenset(
                    snapshot.target for snapshot in serving
                )
            return serving
        errors = tuple(error for _, error in inspected if error is not None)
        if errors:
            raise errors[-1]
        raise RuntimeError("no serving distributed workers are available")

    def execute_tile(
        self, request: messages.ProfileTileRequest
    ) -> messages.ProfileTileResult:
        with self._lock:
            eligible = tuple(
                client
                for client in self._clients
                if self._preferred_targets is None
                or client.target in self._preferred_targets
            )
            if not eligible:
                eligible = self._clients
            start = self._next_worker % len(eligible)
            candidates = eligible[start:] + eligible[:start]
            self._next_worker = (start + 1) % len(eligible)
        last_error: grpc.RpcError | None = None
        for client in candidates:
            try:
                return client.execute_tile(request)
            except grpc.RpcError as error:
                if error.code() not in _TRANSIENT_RPC_CODES:
                    raise
                last_error = error
        if last_error is not None:
            raise last_error
        raise RuntimeError("no serving distributed workers are available")

    def close(self) -> None:
        for client in self._clients:
            client.close()

    @property
    def worker_count(self) -> int:
        """Number of configured worker endpoints."""

        return len(self._clients)

    def __enter__(self) -> "WorkerPool":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def make_tile_request(
    series_a,
    series_b=None,
    *,
    window: int,
    row_start: int = 0,
    row_stop: int | None = None,
    column_start: int = 0,
    column_stop: int | None = None,
    exclusion_zone: int | None = None,
    compute_rows: bool = False,
    compute_columns: bool = True,
    request_id: str | None = None,
) -> messages.ProfileTileRequest:
    """Build a compact single-precision 1NN tile request."""

    a = np.asarray(series_a, dtype=np.float32)
    b = None if series_b is None else np.asarray(series_b, dtype=np.float32)
    if a.ndim != 1 or (b is not None and b.ndim != 1):
        raise ValueError("distributed input series must be one-dimensional")
    if window <= 0 or a.size < window or (b is not None and b.size < window):
        raise ValueError("window must be positive and fit both input series")
    column_count = a.size - window + 1
    row_count = (a if b is None else b).size - window + 1
    resolved_row_stop = row_count if row_stop is None else row_stop
    resolved_column_stop = column_count if column_stop is None else column_stop
    if not 0 <= column_start < resolved_column_stop <= column_count:
        raise ValueError("column tile bounds are outside the A subsequences")
    if not 0 <= row_start < resolved_row_stop <= row_count:
        raise ValueError("row tile bounds are outside the B subsequences")
    resolved_exclusion = (
        (window + 3) // 4
        if exclusion_zone is None and b is None
        else exclusion_zone or 0
    )
    payload_a = a[column_start : resolved_column_stop + window - 1]
    row_source = a if b is None else b
    reuse_a_for_rows = (
        b is None
        and row_start == column_start
        and resolved_row_stop == resolved_column_stop
    )
    payload_b = (
        None
        if reuse_a_for_rows
        else row_source[row_start : resolved_row_stop + window - 1]
    )
    return messages.ProfileTileRequest(
        api_version=API_VERSION,
        request_id=request_id or uuid.uuid4().hex,
        series_a=encode_array(payload_a),
        series_b=(
            messages.ArrayPayload() if payload_b is None else encode_array(payload_b)
        ),
        self_join=b is None,
        window=window,
        row_start=row_start,
        row_stop=resolved_row_stop,
        column_start=column_start,
        column_stop=resolved_column_stop,
        exclusion_zone=resolved_exclusion,
        compute_rows=compute_rows,
        compute_columns=compute_columns,
        profile_kind=messages.PROFILE_KIND_1NN_INDEX,
        precision=messages.PRECISION_SINGLE,
        series_a_offset=column_start,
        series_b_offset=(column_start if payload_b is None else row_start),
    )


__all__ = [
    "API_VERSION",
    "DEFAULT_MAX_MESSAGE_BYTES",
    "WorkerClient",
    "WorkerPool",
    "WorkerServer",
    "WorkerSnapshot",
    "make_tile_request",
    "messages",
]
