"""Bounded coordinator for complete distributed 1NN-index joins."""

from __future__ import annotations

import math
import time
import uuid
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from numbers import Integral
from typing import Any

import grpc
import numpy as np

from ._version import API_VERSION
from .codec import decode_array
from .execution import estimate_1nn_tile_working_set_bytes
from .proto import scamp_worker_v1_pb2 as messages
from .runtime import (
    DEFAULT_MAX_MESSAGE_BYTES,
    _TRANSIENT_RPC_CODES,
    WorkerPool,
    WorkerSnapshot,
    make_tile_request,
)


DEFAULT_MAX_TILE_BYTES = 256 * 1024 * 1024
DEFAULT_MAX_RETRIES = 3


@dataclass(frozen=True, slots=True)
class ProfileTile:
    """One rectangular tile in global subsequence coordinates."""

    ordinal: int
    row_start: int
    row_stop: int
    column_start: int
    column_stop: int
    compute_rows: bool
    compute_columns: bool = True

    @property
    def rows(self) -> int:
        return self.row_stop - self.row_start

    @property
    def columns(self) -> int:
        return self.column_stop - self.column_start


@dataclass(frozen=True, slots=True)
class ProfileTilePlan:
    """Lazy, deterministic decomposition of one complete profile job."""

    row_subsequences: int
    column_subsequences: int
    window: int
    tile_size: int
    exclusion_zone: int
    self_join: bool
    keep_rows: bool
    total_tiles: int
    estimated_peak_tile_bytes: int
    max_tile_bytes: int
    max_message_bytes: int

    def __iter__(self) -> Iterator[ProfileTile]:
        ordinal = 0
        for row_start in range(0, self.row_subsequences, self.tile_size):
            row_stop = min(self.row_subsequences, row_start + self.tile_size)
            first_column = row_start if self.self_join else 0
            for column_start in range(
                first_column, self.column_subsequences, self.tile_size
            ):
                column_stop = min(
                    self.column_subsequences, column_start + self.tile_size
                )
                diagonal = (
                    self.self_join
                    and row_start == column_start
                    and row_stop == column_stop
                )
                yield ProfileTile(
                    ordinal=ordinal,
                    row_start=row_start,
                    row_stop=row_stop,
                    column_start=column_start,
                    column_stop=column_stop,
                    # An off-diagonal self tile represents its transpose too.
                    # A full diagonal block already contributes every column.
                    compute_rows=(self.keep_rows or self.self_join) and not diagonal,
                )
                ordinal += 1


@dataclass(frozen=True, slots=True)
class JobProgress:
    """Monotonic snapshot emitted after each completed tile."""

    completed_tiles: int
    total_tiles: int
    retry_attempts: int
    elapsed_seconds: float
    eta_seconds: float | None
    last_worker_id: str | None

    @property
    def fraction(self) -> float:
        return self.completed_tiles / self.total_tiles


@dataclass(frozen=True, slots=True)
class Distributed1NNResult:
    """Complete profile plus execution metadata.

    ``row_values`` and ``row_indices`` are populated only for an AB-join with
    ``keep_rows=True``. A self-join has one symmetric global profile.
    """

    column_values: np.ndarray
    column_indices: np.ndarray
    row_values: np.ndarray | None
    row_indices: np.ndarray | None
    plan: ProfileTilePlan
    progress: JobProgress


ProgressCallback = Callable[[JobProgress], None]


def estimate_tile_working_set_bytes(rows: int, columns: int, window: int) -> int:
    """Conservatively estimate a worker's peak dense-tile working set.

    The estimate accounts for the correlation matrix and masks as well as
    normalized window intermediates for both axes. It intentionally errs high
    because Metal allocations share the Mac's unified memory with the system.
    """

    return estimate_1nn_tile_working_set_bytes(rows, columns, window)


def _estimate_message_bytes(
    rows: int, columns: int, window: int, *, compute_rows: bool
) -> int:
    # Two input segments are the worst case. Protobuf scalar/tag overhead is
    # tiny but a fixed allowance keeps the planner conservative.
    request_bytes = 4096 + 4 * (rows + columns + 2 * (window - 1))
    response_elements = columns + (rows if compute_rows else 0)
    response_bytes = 4096 + 12 * response_elements
    return max(request_bytes, response_bytes)


def _maximum_safe_tile_size(
    *,
    rows: int,
    columns: int,
    window: int,
    max_tile_bytes: int,
    max_message_bytes: int,
    compute_rows: bool,
) -> int:
    high = max(rows, columns)
    low = 1
    best = 0
    while low <= high:
        candidate = (low + high) // 2
        tile_rows = min(rows, candidate)
        tile_columns = min(columns, candidate)
        fits = (
            estimate_tile_working_set_bytes(tile_rows, tile_columns, window)
            <= max_tile_bytes
            and _estimate_message_bytes(
                tile_rows,
                tile_columns,
                window,
                compute_rows=compute_rows,
            )
            <= max_message_bytes
        )
        if fits:
            best = candidate
            low = candidate + 1
        else:
            high = candidate - 1
    return min(best, max(rows, columns))


def plan_1nn_tiles(
    column_subsequences: int,
    row_subsequences: int,
    window: int,
    *,
    self_join: bool,
    keep_rows: bool = False,
    tile_size: int | None = None,
    exclusion_zone: int | None = None,
    max_tile_bytes: int = DEFAULT_MAX_TILE_BYTES,
    max_message_bytes: int = DEFAULT_MAX_MESSAGE_BYTES,
) -> ProfileTilePlan:
    """Plan a complete job without materializing its tile list."""

    for value, name in (
        (column_subsequences, "column_subsequences"),
        (row_subsequences, "row_subsequences"),
        (window, "window"),
        (max_tile_bytes, "max_tile_bytes"),
        (max_message_bytes, "max_message_bytes"),
    ):
        if not isinstance(value, Integral) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if window < 3:
        raise ValueError("window must be at least 3")
    if self_join and row_subsequences != column_subsequences:
        raise ValueError("a self-join must have equal row and column dimensions")
    resolved_exclusion = (
        (window + 3) // 4
        if exclusion_zone is None and self_join
        else 0
        if exclusion_zone is None
        else exclusion_zone
    )
    if not isinstance(resolved_exclusion, Integral) or resolved_exclusion < 0:
        raise ValueError("exclusion_zone must be a non-negative integer")

    maximum = _maximum_safe_tile_size(
        rows=row_subsequences,
        columns=column_subsequences,
        window=window,
        max_tile_bytes=max_tile_bytes,
        max_message_bytes=max_message_bytes,
        compute_rows=keep_rows or self_join,
    )
    if maximum <= 0:
        raise ValueError(
            "even a one-subsequence tile exceeds the message or working-set limit"
        )
    if tile_size is None:
        resolved_tile_size = maximum
    else:
        if not isinstance(tile_size, Integral) or tile_size <= 0:
            raise ValueError("tile_size must be a positive integer")
        requested_tile_size = min(
            int(tile_size), max(row_subsequences, column_subsequences)
        )
        if requested_tile_size > maximum:
            raise ValueError(
                f"tile_size {tile_size} exceeds the safe limit {maximum} for "
                "this window and resource budget"
            )
        resolved_tile_size = requested_tile_size

    tile_rows = math.ceil(row_subsequences / resolved_tile_size)
    tile_columns = math.ceil(column_subsequences / resolved_tile_size)
    total_tiles = (
        tile_rows * (tile_rows + 1) // 2 if self_join else tile_rows * tile_columns
    )
    peak_rows = min(row_subsequences, resolved_tile_size)
    peak_columns = min(column_subsequences, resolved_tile_size)
    return ProfileTilePlan(
        row_subsequences=int(row_subsequences),
        column_subsequences=int(column_subsequences),
        window=int(window),
        tile_size=resolved_tile_size,
        exclusion_zone=int(resolved_exclusion),
        self_join=self_join,
        keep_rows=keep_rows,
        total_tiles=total_tiles,
        estimated_peak_tile_bytes=estimate_tile_working_set_bytes(
            peak_rows, peak_columns, window
        ),
        max_tile_bytes=int(max_tile_bytes),
        max_message_bytes=int(max_message_bytes),
    )


def _apply_profile_slice(
    values: np.ndarray,
    indices: np.ndarray,
    profile: messages.ProfileSlice,
    *,
    expected_start: int,
    expected_length: int,
    match_start: int | None = None,
    match_stop: int | None = None,
) -> None:
    incoming_values = decode_array(profile.values)
    incoming_indices = decode_array(profile.indices)
    if incoming_values.dtype != np.dtype("<f4") or incoming_indices.dtype != np.dtype(
        "<i8"
    ):
        raise ValueError("1NN profile slices require float32 values and int64 indices")
    if int(profile.offset) != expected_start:
        raise ValueError("worker returned a profile slice at the wrong global offset")
    if (
        incoming_values.size != expected_length
        or incoming_indices.size != expected_length
    ):
        raise ValueError("worker returned a profile slice with the wrong length")
    stop = expected_start + expected_length
    if expected_start < 0 or stop > values.size or indices.size != values.size:
        raise ValueError("profile slice is outside the destination profile")
    valid_correlations = (
        np.isfinite(incoming_values)
        & (incoming_values >= -1.0)
        & (incoming_values <= 1.0)
    )
    invalid = incoming_values == -2.0
    if not np.all(valid_correlations | invalid):
        raise ValueError("worker returned values other than correlations or -2")
    if np.any(incoming_indices[invalid] != -1) or np.any(
        incoming_indices[~invalid] < 0
    ):
        raise ValueError("worker returned inconsistent profile indices")
    if match_start is not None or match_stop is not None:
        if match_start is None or match_stop is None or match_start >= match_stop:
            raise ValueError("match bounds must form a non-empty interval")
        valid_indices = incoming_indices[~invalid]
        if np.any((valid_indices < match_start) | (valid_indices >= match_stop)):
            raise ValueError("worker returned an index outside its assigned tile")

    current_values = values[expected_start:stop]
    current_indices = indices[expected_start:stop]
    better = incoming_values > current_values
    tied_lower_index = (
        (incoming_values == current_values)
        & (incoming_indices >= 0)
        & ((current_indices < 0) | (incoming_indices < current_indices))
    )
    update = better | tied_lower_index
    values[expected_start:stop] = np.where(update, incoming_values, current_values)
    indices[expected_start:stop] = np.where(update, incoming_indices, current_indices)


def merge_1nn_slices(
    results: Iterable[messages.ProfileTileResult],
    profile_length: int,
    *,
    direction: str = "column",
) -> tuple[np.ndarray, np.ndarray]:
    """Merge partial 1NN profiles with deterministic lower-index ties."""

    if profile_length < 0:
        raise ValueError("profile_length cannot be negative")
    if direction not in {"column", "row"}:
        raise ValueError("direction must be 'column' or 'row'")

    best_values = np.full(profile_length, -2.0, dtype=np.float32)
    best_indices = np.full(profile_length, -1, dtype=np.int64)
    for result in results:
        if result.api_version != API_VERSION:
            raise ValueError(
                f"cannot merge distributed API version {result.api_version} "
                f"with version {API_VERSION}"
            )
        profile = result.column_profile if direction == "column" else result.row_profile
        incoming_values = decode_array(profile.values)
        _apply_profile_slice(
            best_values,
            best_indices,
            profile,
            expected_start=int(profile.offset),
            expected_length=incoming_values.size,
        )
    return best_values, best_indices


def _validate_worker_capabilities(
    snapshots: tuple[WorkerSnapshot, ...],
) -> tuple[int, int]:
    message_limits: list[int] = []
    working_set_limits: list[int] = []
    for snapshot in snapshots:
        capabilities = snapshot.capabilities
        if capabilities.api_version != API_VERSION:
            raise ValueError(
                f"worker {capabilities.worker_id} uses distributed API "
                f"{capabilities.api_version}, expected {API_VERSION}"
            )
        if messages.PROFILE_KIND_1NN_INDEX not in capabilities.profile_kinds:
            raise ValueError(
                f"worker {capabilities.worker_id} does not support 1NN-index"
            )
        if messages.PRECISION_SINGLE not in capabilities.precisions:
            raise ValueError(
                f"worker {capabilities.worker_id} does not support single precision"
            )
        if capabilities.max_message_bytes <= 0:
            raise ValueError(
                f"worker {capabilities.worker_id} reported an invalid message limit"
            )
        if capabilities.max_tile_working_set_bytes <= 0:
            raise ValueError(
                f"worker {capabilities.worker_id} reported an invalid tile limit"
            )
        message_limits.append(int(capabilities.max_message_bytes))
        working_set_limits.append(int(capabilities.max_tile_working_set_bytes))
    return min(message_limits), min(working_set_limits)


def _make_progress(
    *,
    completed: int,
    total: int,
    retries: int,
    started: float,
    last_worker_id: str | None,
) -> JobProgress:
    elapsed = time.perf_counter() - started
    eta = None if completed == 0 else elapsed * (total - completed) / completed
    return JobProgress(
        completed_tiles=completed,
        total_tiles=total,
        retry_attempts=retries,
        elapsed_seconds=elapsed,
        eta_seconds=eta,
        last_worker_id=last_worker_id,
    )


def _execute_with_retries(
    pool: WorkerPool,
    request: messages.ProfileTileRequest,
    *,
    eligible_targets: frozenset[str],
    max_retries: int,
    retry_backoff: float,
) -> tuple[messages.ProfileTileResult, int]:
    retries = 0
    while True:
        try:
            return (
                pool.execute_tile(request, eligible_targets=eligible_targets),
                retries,
            )
        except grpc.RpcError as error:
            if error.code() not in _TRANSIENT_RPC_CODES or retries >= max_retries:
                raise
            delay = retry_backoff * (2**retries)
            retries += 1
            # Refresh the preferred set only after every currently selected
            # worker failed. This admits a recovered Mac without adding a
            # health round trip to successful tile dispatches.
            try:
                pool.discover_serving()
            except (grpc.RpcError, RuntimeError):
                pass
            if delay > 0:
                time.sleep(min(delay, 1.0))


def _convert_correlations(values: np.ndarray, window: int, pearson: bool) -> np.ndarray:
    valid = values >= -1.0
    if pearson:
        output = values.astype(np.float32, copy=True)
    else:
        output = np.sqrt(np.maximum(2.0 * window * (1.0 - values), 0.0)).astype(
            np.float32
        )
    output[~valid] = np.nan
    return output


class DistributedCoordinator:
    """Execute complete 1NN-index jobs across a reusable worker pool."""

    def __init__(
        self,
        pool: WorkerPool,
        *,
        max_in_flight: int | None = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_backoff: float = 0.05,
        max_tile_bytes: int = DEFAULT_MAX_TILE_BYTES,
    ) -> None:
        if max_in_flight is not None and (
            not isinstance(max_in_flight, Integral) or max_in_flight <= 0
        ):
            raise ValueError("max_in_flight must be a positive integer")
        if not isinstance(max_retries, Integral) or max_retries < 0:
            raise ValueError("max_retries must be a non-negative integer")
        if retry_backoff < 0:
            raise ValueError("retry_backoff cannot be negative")
        if not isinstance(max_tile_bytes, Integral) or max_tile_bytes <= 0:
            raise ValueError("max_tile_bytes must be a positive integer")
        self.pool = pool
        self.max_in_flight = None if max_in_flight is None else int(max_in_flight)
        self.max_retries = int(max_retries)
        self.retry_backoff = float(retry_backoff)
        self.max_tile_bytes = int(max_tile_bytes)

    def selfjoin(
        self,
        series: Any,
        window: int,
        *,
        tile_size: int | None = None,
        exclusion_zone: int | None = None,
        pearson: bool = False,
        progress: ProgressCallback | None = None,
    ) -> Distributed1NNResult:
        """Compute one complete symmetric 1NN-index profile."""

        return self._run(
            series,
            None,
            window,
            tile_size=tile_size,
            exclusion_zone=exclusion_zone,
            keep_rows=False,
            pearson=pearson,
            progress=progress,
        )

    def abjoin(
        self,
        series_a: Any,
        series_b: Any,
        window: int,
        *,
        tile_size: int | None = None,
        exclusion_zone: int = 0,
        keep_rows: bool = False,
        pearson: bool = False,
        progress: ProgressCallback | None = None,
    ) -> Distributed1NNResult:
        """Compute a complete AB 1NN-index profile and optional row profile."""

        return self._run(
            series_a,
            series_b,
            window,
            tile_size=tile_size,
            exclusion_zone=exclusion_zone,
            keep_rows=keep_rows,
            pearson=pearson,
            progress=progress,
        )

    def _run(
        self,
        series_a: Any,
        series_b: Any | None,
        window: int,
        *,
        tile_size: int | None,
        exclusion_zone: int | None,
        keep_rows: bool,
        pearson: bool,
        progress: ProgressCallback | None,
    ) -> Distributed1NNResult:
        if not isinstance(window, Integral) or window < 3:
            raise ValueError("window must be an integer at least 3")
        a = np.asarray(series_a, dtype=np.float32)
        b = None if series_b is None else np.asarray(series_b, dtype=np.float32)
        if a.ndim != 1 or (b is not None and b.ndim != 1):
            raise ValueError("distributed input series must be one-dimensional")
        if a.size < window or (b is not None and b.size < window):
            raise ValueError("window must fit both input series")

        snapshots = self.pool.discover_serving()
        worker_message_bytes, worker_tile_bytes = _validate_worker_capabilities(
            snapshots
        )
        max_message_bytes = min(
            worker_message_bytes,
            int(
                getattr(
                    self.pool, "max_message_bytes", DEFAULT_MAX_MESSAGE_BYTES
                )
            ),
        )
        eligible_targets = frozenset(snapshot.target for snapshot in snapshots)
        columns = int(a.size - window + 1)
        rows = columns if b is None else int(b.size - window + 1)
        plan = plan_1nn_tiles(
            columns,
            rows,
            int(window),
            self_join=b is None,
            keep_rows=keep_rows,
            tile_size=tile_size,
            exclusion_zone=exclusion_zone,
            max_tile_bytes=min(self.max_tile_bytes, worker_tile_bytes),
            max_message_bytes=max_message_bytes,
        )

        column_values = np.full(columns, -2.0, dtype=np.float32)
        column_indices = np.full(columns, -1, dtype=np.int64)
        row_values = (
            np.full(rows, -2.0, dtype=np.float32)
            if b is not None and keep_rows
            else None
        )
        row_indices = (
            np.full(rows, -1, dtype=np.int64) if b is not None and keep_rows else None
        )
        job_id = uuid.uuid4().hex
        started = time.perf_counter()
        completed = 0
        retries = 0
        latest_progress = _make_progress(
            completed=0,
            total=plan.total_tiles,
            retries=0,
            started=started,
            last_worker_id=None,
        )
        if progress is not None:
            progress(latest_progress)

        def submit_tile(
            executor: ThreadPoolExecutor, tile: ProfileTile
        ) -> Future[tuple[messages.ProfileTileResult, int]]:
            request = make_tile_request(
                a,
                b,
                window=int(window),
                row_start=tile.row_start,
                row_stop=tile.row_stop,
                column_start=tile.column_start,
                column_stop=tile.column_stop,
                exclusion_zone=plan.exclusion_zone,
                compute_rows=tile.compute_rows,
                compute_columns=tile.compute_columns,
                request_id=f"{job_id}:{tile.ordinal}",
            )
            expected_response_bytes = _estimate_message_bytes(
                tile.rows,
                tile.columns,
                int(window),
                compute_rows=tile.compute_rows,
            )
            if request.ByteSize() > plan.max_message_bytes:
                raise ValueError(
                    "planned tile request exceeds the worker message limit"
                )
            if expected_response_bytes > plan.max_message_bytes:
                raise ValueError(
                    "planned tile response exceeds the worker message limit"
                )
            future = executor.submit(
                _execute_with_retries,
                self.pool,
                request,
                eligible_targets=eligible_targets,
                max_retries=self.max_retries,
                retry_backoff=self.retry_backoff,
            )
            return future

        in_flight_limit = self.max_in_flight or len(snapshots)
        tiles = iter(plan)
        pending: dict[Future[tuple[messages.ProfileTileResult, int]], ProfileTile] = {}
        executor = ThreadPoolExecutor(
            max_workers=in_flight_limit, thread_name_prefix="scamp-coordinator"
        )
        try:
            for _ in range(in_flight_limit):
                tile = next(tiles, None)
                if tile is None:
                    break
                pending[submit_tile(executor, tile)] = tile

            while pending:
                done, _ = wait(tuple(pending), return_when=FIRST_COMPLETED)
                for future in sorted(done, key=lambda item: pending[item].ordinal):
                    tile = pending.pop(future)
                    result, tile_retries = future.result()
                    expected_request_id = f"{job_id}:{tile.ordinal}"
                    if result.api_version != API_VERSION:
                        raise ValueError("worker returned an incompatible API version")
                    if result.request_id != expected_request_id:
                        raise ValueError(
                            "worker returned a result for the wrong request"
                        )
                    _apply_profile_slice(
                        column_values,
                        column_indices,
                        result.column_profile,
                        expected_start=tile.column_start,
                        expected_length=tile.columns,
                        match_start=tile.row_start,
                        match_stop=tile.row_stop,
                    )
                    if tile.compute_rows:
                        target_values = column_values if b is None else row_values
                        target_indices = column_indices if b is None else row_indices
                        if target_values is None or target_indices is None:
                            raise RuntimeError("row profile storage was not allocated")
                        _apply_profile_slice(
                            target_values,
                            target_indices,
                            result.row_profile,
                            expected_start=tile.row_start,
                            expected_length=tile.rows,
                            match_start=tile.column_start,
                            match_stop=tile.column_stop,
                        )
                    completed += 1
                    retries += tile_retries
                    latest_progress = _make_progress(
                        completed=completed,
                        total=plan.total_tiles,
                        retries=retries,
                        started=started,
                        last_worker_id=result.worker_id,
                    )
                    if progress is not None:
                        progress(latest_progress)

                    tile = next(tiles, None)
                    if tile is not None:
                        pending[submit_tile(executor, tile)] = tile
        finally:
            for future in pending:
                future.cancel()
            executor.shutdown(wait=True, cancel_futures=True)

        return Distributed1NNResult(
            column_values=_convert_correlations(column_values, int(window), pearson),
            column_indices=column_indices,
            row_values=(
                None
                if row_values is None
                else _convert_correlations(row_values, int(window), pearson)
            ),
            row_indices=row_indices,
            plan=plan,
            progress=latest_progress,
        )


__all__ = [
    "DEFAULT_MAX_RETRIES",
    "DEFAULT_MAX_TILE_BYTES",
    "Distributed1NNResult",
    "DistributedCoordinator",
    "JobProgress",
    "ProfileTile",
    "ProfileTilePlan",
    "estimate_tile_working_set_bytes",
    "merge_1nn_slices",
    "plan_1nn_tiles",
]
