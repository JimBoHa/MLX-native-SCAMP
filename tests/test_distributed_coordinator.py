from __future__ import annotations

import socket
import unittest

import grpc
import mlx.core as mx
import numpy as np

from mlx_native_scamp.distributed import (
    DistributedCoordinator,
    WorkerPool,
    WorkerServer,
    estimate_tile_working_set_bytes,
    make_tile_request,
    plan_1nn_tiles,
)


def _reference_matrix(
    series_a: np.ndarray,
    series_b: np.ndarray | None,
    window: int,
    exclusion_zone: int = 0,
) -> np.ndarray:
    a = np.asarray(series_a, dtype=np.float64)
    b = a if series_b is None else np.asarray(series_b, dtype=np.float64)

    def prepare(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        windows = np.lib.stride_tricks.sliding_window_view(values, window)
        finite = np.isfinite(windows).all(axis=1)
        clean = np.where(np.isfinite(windows), windows, 0.0)
        centered = clean - clean.mean(axis=1, keepdims=True)
        norms = np.linalg.norm(centered, axis=1)
        valid = finite & (norms * norms > 1e-13)
        normalized = np.zeros_like(centered)
        normalized[valid] = centered[valid] / norms[valid, None]
        return normalized, valid

    prepared_a, valid_a = prepare(a)
    prepared_b, valid_b = prepare(b)
    matrix = prepared_b @ prepared_a.T
    matrix = np.clip(matrix, -1.0, 1.0)
    matrix[~(valid_b[:, None] & valid_a[None, :])] = -2.0
    if series_b is None and exclusion_zone:
        rows = np.arange(matrix.shape[0])
        columns = np.arange(matrix.shape[1])
        matrix[np.abs(rows[:, None] - columns[None, :]) < exclusion_zone] = -2.0
    return matrix


def _reference_profile(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = matrix.max(axis=0).astype(np.float32)
    indices = matrix.argmax(axis=0).astype(np.int64)
    indices[values < -1.0] = -1
    values[values < -1.0] = np.nan
    return values, indices


class _TransientError(grpc.RpcError):
    def code(self):
        return grpc.StatusCode.UNAVAILABLE


class _FailOncePool:
    def __init__(self, pool: WorkerPool) -> None:
        self.pool = pool
        self.calls = 0
        self.request_ids = []

    def discover_serving(self):
        return self.pool.discover_serving()

    @property
    def max_message_bytes(self):
        return self.pool.max_message_bytes

    def execute_tile(self, request, *, eligible_targets=None):
        self.calls += 1
        self.request_ids.append(request.request_id)
        if self.calls == 1:
            raise _TransientError()
        return self.pool.execute_tile(
            request, eligible_targets=eligible_targets
        )


class _RefreshAddsWorkerPool:
    def __init__(self, pool: WorkerPool, initial_worker_id: str) -> None:
        self.pool = pool
        self.initial_worker_id = initial_worker_id
        self.discovery_calls = 0
        self.execution_calls = 0
        self.eligible_targets = []

    @property
    def max_message_bytes(self):
        return self.pool.max_message_bytes

    def discover_serving(self):
        self.discovery_calls += 1
        if self.discovery_calls == 1:
            return tuple(
                snapshot
                for snapshot in self.pool.discover()
                if snapshot.capabilities.worker_id == self.initial_worker_id
            )
        return self.pool.discover_serving()

    def execute_tile(self, request, *, eligible_targets=None):
        self.execution_calls += 1
        self.eligible_targets.append(eligible_targets)
        if self.execution_calls == 1:
            raise _TransientError()
        return self.pool.execute_tile(
            request, eligible_targets=eligible_targets
        )


class DistributedTilePlanningTests(unittest.TestCase):
    def test_window_must_match_upstream_scamp_minimum(self):
        with self.assertRaisesRegex(ValueError, "at least 3"):
            plan_1nn_tiles(8, 8, 2, self_join=True)

    def test_self_join_uses_only_upper_triangle_and_transpose_profiles(self):
        plan = plan_1nn_tiles(
            10,
            10,
            7,
            self_join=True,
            tile_size=4,
            max_message_bytes=1024 * 1024,
        )
        tiles = tuple(plan)
        self.assertEqual(6, plan.total_tiles)
        self.assertEqual(6, len(tiles))
        self.assertEqual(2, plan.exclusion_zone)
        self.assertTrue(all(tile.row_start <= tile.column_start for tile in tiles))
        for tile in tiles:
            if tile.row_start == tile.column_start:
                self.assertFalse(tile.compute_rows)
            else:
                self.assertTrue(tile.compute_rows)

    def test_ab_join_covers_the_full_rectangle_and_optional_rows(self):
        plan = plan_1nn_tiles(
            9,
            5,
            4,
            self_join=False,
            keep_rows=True,
            tile_size=4,
            max_message_bytes=1024 * 1024,
        )
        tiles = tuple(plan)
        self.assertEqual(6, len(tiles))
        self.assertTrue(all(tile.compute_rows for tile in tiles))
        covered = {
            (row, column)
            for tile in tiles
            for row in range(tile.row_start, tile.row_stop)
            for column in range(tile.column_start, tile.column_stop)
        }
        self.assertEqual(
            {(row, column) for row in range(5) for column in range(9)}, covered
        )

    def test_automatic_tile_size_honors_memory_and_message_limits(self):
        memory_limit = 2 * 1024 * 1024
        message_limit = 64 * 1024
        plan = plan_1nn_tiles(
            1000,
            700,
            64,
            self_join=False,
            keep_rows=True,
            max_tile_bytes=memory_limit,
            max_message_bytes=message_limit,
        )
        self.assertLess(plan.tile_size, 700)
        self.assertLessEqual(plan.estimated_peak_tile_bytes, memory_limit)
        self.assertLessEqual(
            estimate_tile_working_set_bytes(plan.tile_size, plan.tile_size, 64),
            memory_limit,
        )

    def test_impossible_resource_budget_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "even a one-subsequence tile"):
            plan_1nn_tiles(
                3,
                3,
                8,
                self_join=True,
                max_tile_bytes=1,
                max_message_bytes=1024 * 1024,
            )

    def test_every_planned_request_fits_the_advertised_message_limit(self):
        rng = np.random.default_rng(51)
        series_a = rng.normal(size=1000).astype(np.float32)
        series_b = rng.normal(size=700).astype(np.float32)
        window = 64
        message_limit = 64 * 1024
        plan = plan_1nn_tiles(
            series_a.size - window + 1,
            series_b.size - window + 1,
            window,
            self_join=False,
            keep_rows=True,
            max_message_bytes=message_limit,
        )
        for tile in plan:
            request = make_tile_request(
                series_a,
                series_b,
                window=window,
                row_start=tile.row_start,
                row_stop=tile.row_stop,
                column_start=tile.column_start,
                column_stop=tile.column_stop,
                compute_rows=tile.compute_rows,
            )
            self.assertLessEqual(request.ByteSize(), message_limit)


class DistributedCoordinatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = WorkerServer(
            backend="cpu",
            worker_id="coordinator-worker",
            max_concurrent_tiles=2,
        ).start()
        cls.pool = WorkerPool([cls.server.target], timeout=15.0)
        cls.coordinator = DistributedCoordinator(
            cls.pool,
            max_in_flight=2,
            retry_backoff=0.0,
        )

    @classmethod
    def tearDownClass(cls):
        cls.pool.close()
        cls.server.stop(grace=0.0)

    def test_randomized_self_joins_match_full_reference(self):
        rng = np.random.default_rng(90210)
        for case in range(18):
            length = int(rng.integers(20, 75))
            window = int(rng.integers(3, min(14, length - 1)))
            series = rng.normal(size=length).astype(np.float32)
            subsequences = length - window + 1
            tile_size = int(rng.integers(1, subsequences + 4))
            result = self.coordinator.selfjoin(
                series,
                window,
                tile_size=tile_size,
                pearson=True,
            )

            exclusion = (window + 3) // 4
            matrix = _reference_matrix(series, None, window, exclusion)
            expected_values, expected_indices = _reference_profile(matrix)
            with self.subTest(case=case, window=window, tile_size=tile_size):
                np.testing.assert_allclose(
                    result.column_values,
                    expected_values,
                    rtol=3e-5,
                    atol=3e-5,
                    equal_nan=True,
                )
                np.testing.assert_array_equal(result.column_indices, expected_indices)

    def test_randomized_ab_joins_and_row_profiles_match_reference(self):
        rng = np.random.default_rng(19106)
        for case in range(18):
            length_a = int(rng.integers(18, 70))
            length_b = int(rng.integers(18, 70))
            window = int(rng.integers(3, min(12, length_a, length_b)))
            series_a = rng.normal(size=length_a).astype(np.float32)
            series_b = rng.normal(size=length_b).astype(np.float32)
            tile_size = int(rng.integers(1, max(length_a, length_b)))
            result = self.coordinator.abjoin(
                series_a,
                series_b,
                window,
                tile_size=tile_size,
                keep_rows=True,
                pearson=True,
            )

            matrix = _reference_matrix(series_a, series_b, window)
            expected_values, expected_indices = _reference_profile(matrix)
            expected_row_values, expected_row_indices = _reference_profile(matrix.T)
            with self.subTest(case=case, window=window, tile_size=tile_size):
                np.testing.assert_allclose(
                    result.column_values,
                    expected_values,
                    rtol=3e-5,
                    atol=3e-5,
                )
                np.testing.assert_array_equal(result.column_indices, expected_indices)
                np.testing.assert_allclose(
                    result.row_values,
                    expected_row_values,
                    rtol=3e-5,
                    atol=3e-5,
                )
                np.testing.assert_array_equal(result.row_indices, expected_row_indices)

    def test_default_output_uses_upstream_euclidean_conversion(self):
        rng = np.random.default_rng(8)
        series_a = rng.normal(size=41).astype(np.float32)
        series_b = rng.normal(size=47).astype(np.float32)
        window = 9
        result = self.coordinator.abjoin(series_a, series_b, window, tile_size=7)
        correlations, expected_indices = _reference_profile(
            _reference_matrix(series_a, series_b, window)
        )
        expected_values = np.sqrt(
            np.maximum(2.0 * window * (1.0 - correlations), 0.0)
        ).astype(np.float32)
        np.testing.assert_allclose(result.column_values, expected_values, atol=3e-5)
        np.testing.assert_array_equal(result.column_indices, expected_indices)
        self.assertIsNone(result.row_values)
        self.assertIsNone(result.row_indices)

    def test_flat_and_nonfinite_windows_keep_invalid_scamp_output(self):
        series = np.ones(32, dtype=np.float32)
        series[7] = np.nan
        result = self.coordinator.selfjoin(series, 5, tile_size=3, pearson=True)
        np.testing.assert_array_equal(
            np.full(result.column_values.shape, np.nan, dtype=np.float32),
            result.column_values,
        )
        np.testing.assert_array_equal(
            np.full(result.column_indices.shape, -1, dtype=np.int64),
            result.column_indices,
        )

    def test_equal_correlations_choose_the_lower_global_index(self):
        pattern = np.array([0.0, 1.0, 0.0, -1.0], dtype=np.float32)
        series = np.tile(pattern, 14)
        window = 4
        result = self.coordinator.selfjoin(series, window, tile_size=5, pearson=True)
        expected_values, expected_indices = _reference_profile(
            _reference_matrix(series, None, window, (window + 3) // 4)
        )
        np.testing.assert_allclose(result.column_values, expected_values, atol=3e-5)
        np.testing.assert_array_equal(result.column_indices, expected_indices)

    def test_progress_is_monotonic_and_finishes_with_an_eta_of_zero(self):
        events = []
        series = np.random.default_rng(2).normal(size=48).astype(np.float32)
        result = self.coordinator.selfjoin(
            series,
            6,
            tile_size=8,
            pearson=True,
            progress=events.append,
        )
        self.assertEqual(0, events[0].completed_tiles)
        self.assertEqual(
            list(range(result.plan.total_tiles + 1)),
            [event.completed_tiles for event in events],
        )
        self.assertTrue(
            all(
                left.fraction <= right.fraction
                for left, right in zip(events, events[1:])
            )
        )
        self.assertEqual(1.0, events[-1].fraction)
        self.assertEqual(0.0, events[-1].eta_seconds)
        self.assertEqual("coordinator-worker", events[-1].last_worker_id)

    def test_transient_failure_is_retried_with_the_same_job(self):
        flaky_pool = _FailOncePool(self.pool)
        coordinator = DistributedCoordinator(
            flaky_pool,
            max_in_flight=1,
            max_retries=3,
            retry_backoff=0.0,
        )
        series = np.random.default_rng(4).normal(size=35).astype(np.float32)
        result = coordinator.selfjoin(series, 5, tile_size=100, pearson=True)
        self.assertEqual(1, result.progress.retry_attempts)
        self.assertEqual(2, flaky_pool.calls)
        self.assertEqual(1, len(set(flaky_pool.request_ids)))

    def test_concurrent_cpu_tiles_restore_the_callers_mlx_device(self):
        original_device = mx.default_device()
        series = np.random.default_rng(31).normal(size=50).astype(np.float32)
        self.coordinator.selfjoin(series, 5, tile_size=4, pearson=True)
        self.assertEqual(original_device, mx.default_device())

    def test_complete_job_schedules_work_on_each_serving_worker(self):
        with WorkerServer(
            backend="cpu", worker_id="second-worker", max_concurrent_tiles=1
        ) as second_server:
            with WorkerPool(
                [self.server.target, second_server.target], timeout=15.0
            ) as pool:
                before = {
                    snapshot.capabilities.worker_id: snapshot.health.completed_requests
                    for snapshot in pool.discover()
                }
                coordinator = DistributedCoordinator(
                    pool, max_in_flight=2, retry_backoff=0.0
                )
                series = np.random.default_rng(73).normal(size=72).astype(np.float32)
                result = coordinator.selfjoin(series, 7, tile_size=8, pearson=True)
                after = {
                    snapshot.capabilities.worker_id: snapshot.health.completed_requests
                    for snapshot in pool.discover()
                }
        self.assertEqual(result.plan.total_tiles, result.progress.completed_tiles)
        self.assertGreater(after["coordinator-worker"], before["coordinator-worker"])
        self.assertGreater(after["second-worker"], before["second-worker"])

    def test_retry_refresh_does_not_admit_an_unvalidated_worker(self):
        with WorkerServer(
            backend="cpu", worker_id="newly-recovered-worker"
        ) as second_server:
            with WorkerPool(
                [self.server.target, second_server.target], timeout=15.0
            ) as pool:
                wrapped = _RefreshAddsWorkerPool(pool, "coordinator-worker")
                coordinator = DistributedCoordinator(
                    wrapped,
                    max_in_flight=1,
                    max_retries=2,
                    retry_backoff=0.0,
                )
                before = pool.discover()[1].health.completed_requests
                series = np.random.default_rng(79).normal(size=36).astype(
                    np.float32
                )
                result = coordinator.selfjoin(
                    series, 5, tile_size=100, pearson=True
                )
                after = pool.discover()[1].health.completed_requests

        self.assertEqual(1, result.progress.retry_attempts)
        self.assertGreaterEqual(wrapped.discovery_calls, 2)
        self.assertEqual(before, after)
        self.assertTrue(
            all(
                targets == wrapped.eligible_targets[0]
                for targets in wrapped.eligible_targets
            )
        )
        self.assertEqual(1, len(wrapped.eligible_targets[0]))

    def test_planner_honors_the_smallest_worker_working_set_limit(self):
        worker_limit = 4096
        with WorkerServer(
            backend="cpu",
            worker_id="small-memory-worker",
            max_tile_working_set_bytes=worker_limit,
        ) as server:
            with WorkerPool([server.target], timeout=15.0) as pool:
                coordinator = DistributedCoordinator(pool, retry_backoff=0.0)
                series = np.random.default_rng(101).normal(size=52).astype(np.float32)
                result = coordinator.selfjoin(series, 5, pearson=True)
                self.assertEqual(worker_limit, result.plan.max_tile_bytes)
                self.assertLessEqual(
                    result.plan.estimated_peak_tile_bytes, worker_limit
                )
                completed = pool.discover()[0].health.completed_requests
                with self.assertRaisesRegex(ValueError, "safe limit"):
                    coordinator.selfjoin(series, 5, tile_size=48, pearson=True)
                self.assertEqual(
                    completed, pool.discover()[0].health.completed_requests
                )

    def test_planner_honors_the_client_channel_message_limit(self):
        client_limit = 8 * 1024
        with WorkerServer(
            backend="cpu",
            worker_id="large-message-worker",
            max_message_bytes=1024 * 1024,
        ) as server:
            with WorkerPool(
                [server.target],
                timeout=15.0,
                max_message_bytes=client_limit,
            ) as pool:
                coordinator = DistributedCoordinator(pool, retry_backoff=0.0)
                series = np.random.default_rng(103).normal(size=220).astype(
                    np.float32
                )
                result = coordinator.selfjoin(series, 16, pearson=True)
                self.assertEqual(client_limit, result.plan.max_message_bytes)
                self.assertLess(result.plan.tile_size, series.size - 16 + 1)

    def test_complete_job_rejects_a_window_below_three(self):
        with self.assertRaisesRegex(ValueError, "at least 3"):
            self.coordinator.selfjoin(np.arange(8, dtype=np.float32), 2)

    def test_worker_pool_discovers_and_fails_over_from_an_offline_peer(self):
        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        unused_port = listener.getsockname()[1]
        listener.close()
        with WorkerPool(
            [f"127.0.0.1:{unused_port}", self.server.target], timeout=0.2
        ) as pool:
            snapshots = pool.discover_serving()
            self.assertEqual(
                ["coordinator-worker"],
                [snapshot.capabilities.worker_id for snapshot in snapshots],
            )
            series = np.arange(24, dtype=np.float32)
            result = pool.execute_tile(make_tile_request(series, window=4))
            self.assertEqual("coordinator-worker", result.worker_id)


if __name__ == "__main__":
    unittest.main()
