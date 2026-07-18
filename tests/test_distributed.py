from __future__ import annotations

import platform
import unittest
from unittest import mock

import grpc
import numpy as np

from mlx_native_scamp.distributed import (
    API_VERSION,
    WorkerClient,
    WorkerPool,
    WorkerServer,
    make_tile_request,
    merge_1nn_slices,
    messages,
)
from mlx_native_scamp.distributed.codec import decode_array, encode_array
from mlx_native_scamp.distributed.execution import execute_1nn_tile
from mlx_native_scamp.distributed import runtime


def _normalized_windows(values: np.ndarray, window: int) -> np.ndarray:
    windows = np.lib.stride_tricks.sliding_window_view(
        values.astype(np.float64), window
    )
    centered = windows - windows.mean(axis=1, keepdims=True)
    return centered / np.linalg.norm(centered, axis=1, keepdims=True)


class DistributedCodecTests(unittest.TestCase):
    def test_compact_array_round_trip(self):
        values = np.linspace(-1, 1, 17, dtype=np.float32)
        payload = encode_array(values)
        self.assertEqual(messages.ARRAY_DTYPE_FLOAT32, payload.dtype)
        self.assertEqual(values.size * 4, len(payload.data))
        np.testing.assert_array_equal(values, decode_array(payload))

    def test_corrupt_array_length_is_rejected(self):
        payload = messages.ArrayPayload(
            dtype=messages.ARRAY_DTYPE_FLOAT32,
            length=3,
            data=b"not-four-byte-values",
        )
        with self.assertRaisesRegex(ValueError, "byte length"):
            decode_array(payload)

    def test_combiner_rejects_an_incompatible_result_version(self):
        result = messages.ProfileTileResult(api_version=API_VERSION + 1)
        with self.assertRaisesRegex(ValueError, "cannot merge distributed API"):
            merge_1nn_slices([result], 0)


class DistributedWorkerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = WorkerServer(backend="auto", worker_id="test-mlx-worker").start()
        cls.client = WorkerClient(cls.server.target, timeout=15.0)
        cls.client.wait_ready()

    @classmethod
    def tearDownClass(cls):
        cls.client.close()
        cls.server.stop(grace=0.0)

    def setUp(self):
        rng = np.random.default_rng(41)
        self.a = rng.normal(size=96).astype(np.float32)
        self.b = rng.normal(size=104).astype(np.float32)
        self.window = 12

    def test_capability_and_health_discovery(self):
        capabilities = self.client.capabilities()
        self.assertEqual(API_VERSION, capabilities.api_version)
        self.assertEqual("test-mlx-worker", capabilities.worker_id)
        self.assertTrue(capabilities.mlx_version)
        self.assertTrue(capabilities.device_name)
        self.assertGreater(capabilities.max_tile_working_set_bytes, 0)
        self.assertIn(messages.PROFILE_KIND_1NN_INDEX, capabilities.profile_kinds)
        self.assertEqual([messages.PRECISION_SINGLE], list(capabilities.precisions))
        if platform.system() == "Darwin" and platform.machine() == "arm64":
            self.assertTrue(capabilities.apple_silicon)
            self.assertEqual(messages.DEVICE_BACKEND_METAL, capabilities.backend)
            self.assertGreater(capabilities.unified_memory_bytes, 0)

        health = self.client.health()
        self.assertEqual(messages.WORKER_STATE_SERVING, health.state)
        self.assertEqual("test-mlx-worker", health.worker_id)

    def test_cpu_backend_can_be_selected_explicitly(self):
        with WorkerServer(backend="cpu", worker_id="cpu-worker") as server:
            with WorkerClient(server.target) as client:
                capabilities = client.capabilities()
                self.assertEqual(messages.DEVICE_BACKEND_CPU, capabilities.backend)

    def test_rectangular_ab_tile_matches_numpy_and_returns_global_indices(self):
        row_start, row_stop = 5, 37
        column_start, column_stop = 7, 45
        request = make_tile_request(
            self.a,
            self.b,
            window=self.window,
            row_start=row_start,
            row_stop=row_stop,
            column_start=column_start,
            column_stop=column_stop,
            compute_rows=True,
            request_id="rectangular-test",
        )
        self.assertEqual(column_start, request.series_a_offset)
        self.assertEqual(row_start, request.series_b_offset)
        self.assertEqual(
            (column_stop - column_start + self.window - 1) * 4,
            len(request.series_a.data),
        )
        self.assertEqual(
            (row_stop - row_start + self.window - 1) * 4,
            len(request.series_b.data),
        )
        result = self.client.execute_tile(request)

        expected = (
            _normalized_windows(self.b, self.window)[row_start:row_stop]
            @ _normalized_windows(self.a, self.window)[column_start:column_stop].T
        )
        np.testing.assert_allclose(
            expected.max(axis=0),
            decode_array(result.column_profile.values),
            rtol=2e-5,
            atol=2e-5,
        )
        np.testing.assert_array_equal(
            expected.argmax(axis=0) + row_start,
            decode_array(result.column_profile.indices),
        )
        np.testing.assert_allclose(
            expected.max(axis=1),
            decode_array(result.row_profile.values),
            rtol=2e-5,
            atol=2e-5,
        )
        np.testing.assert_array_equal(
            expected.argmax(axis=1) + column_start,
            decode_array(result.row_profile.indices),
        )
        self.assertEqual("rectangular-test", result.request_id)
        self.assertEqual(column_start, result.column_profile.offset)
        self.assertEqual(row_start, result.row_profile.offset)
        self.assertGreater(result.execution_nanos, 0)

    def test_self_join_uses_upstream_exclusion_and_worker_pool(self):
        request = make_tile_request(
            self.a,
            window=self.window,
            compute_rows=True,
            request_id="self-test",
        )
        self.assertEqual((self.window + 3) // 4, request.exclusion_zone)
        with WorkerPool([self.server.target], timeout=15.0) as pool:
            snapshot = pool.discover()[0]
            self.assertEqual("test-mlx-worker", snapshot.capabilities.worker_id)
            result = pool.execute_tile(request)

        expected = _normalized_windows(self.a, self.window)
        matrix = expected @ expected.T
        exclusion = request.exclusion_zone
        positions = np.arange(matrix.shape[0])
        matrix[np.abs(positions[:, None] - positions[None, :]) < exclusion] = -2.0
        np.testing.assert_allclose(
            matrix.max(axis=0),
            decode_array(result.column_profile.values),
            rtol=2e-5,
            atol=2e-5,
        )
        np.testing.assert_array_equal(
            matrix.argmax(axis=0), decode_array(result.column_profile.indices)
        )

    def test_row_tiles_merge_into_complete_column_profile(self):
        rows = self.b.size - self.window + 1
        columns = self.a.size - self.window + 1
        split = rows // 2
        requests = (
            make_tile_request(
                self.a,
                self.b,
                window=self.window,
                row_start=start,
                row_stop=stop,
            )
            for start, stop in ((0, split), (split, rows))
        )
        results = [self.client.execute_tile(request) for request in requests]
        values, indices = merge_1nn_slices(reversed(results), columns)

        matrix = (
            _normalized_windows(self.b, self.window)
            @ _normalized_windows(self.a, self.window).T
        )
        np.testing.assert_allclose(matrix.max(axis=0), values, rtol=2e-5, atol=2e-5)
        np.testing.assert_array_equal(matrix.argmax(axis=0), indices)

    def test_invalid_bounds_return_invalid_argument_and_update_health(self):
        before = self.client.health().failed_requests
        request = make_tile_request(self.a, self.b, window=self.window)
        request.row_stop = 99999
        with self.assertRaises(grpc.RpcError) as caught:
            self.client.execute_tile(request)
        self.assertEqual(grpc.StatusCode.INVALID_ARGUMENT, caught.exception.code())
        self.assertEqual(before + 1, self.client.health().failed_requests)

    def test_flat_windows_keep_scamp_invalid_sentinel(self):
        flat = np.ones_like(self.a)
        result = self.client.execute_tile(
            make_tile_request(flat, self.b, window=self.window)
        )
        values = decode_array(result.column_profile.values)
        indices = decode_array(result.column_profile.indices)
        np.testing.assert_array_equal(np.full_like(values, -2.0), values)
        np.testing.assert_array_equal(np.full_like(indices, -1), indices)

    def test_incompatible_protocol_version_fails_explicitly(self):
        with self.assertRaises(grpc.RpcError) as caught:
            self.client.stub.GetCapabilities(
                messages.VersionRequest(api_version=API_VERSION + 1), timeout=5.0
            )
        self.assertEqual(grpc.StatusCode.FAILED_PRECONDITION, caught.exception.code())
        self.assertIn("supports 1", caught.exception.details())

    def test_worker_rejects_a_tile_above_its_working_set_before_execution(self):
        with WorkerServer(
            backend="cpu",
            worker_id="bounded-worker",
            max_tile_working_set_bytes=1024,
        ) as server:
            with WorkerClient(server.target) as client:
                request = make_tile_request(self.a, self.b, window=self.window)
                with self.assertRaises(grpc.RpcError) as caught:
                    client.execute_tile(request)
        self.assertEqual(grpc.StatusCode.RESOURCE_EXHAUSTED, caught.exception.code())
        self.assertIn("working set", caught.exception.details())

    def test_uint32_aliasing_cannot_create_a_false_exclusion(self):
        values = np.arange(12, dtype=np.float32)
        output = execute_1nn_tile(
            values,
            values,
            4,
            row_start=2**32,
            row_stop=2**32 + 9,
            column_start=0,
            column_stop=9,
            exclusion_zone=1,
            compute_rows=False,
            compute_columns=True,
            device=self.server.service.device,
            series_a_offset=0,
            series_b_offset=2**32,
        )
        np.testing.assert_allclose(output.column_values, np.ones(9), atol=2e-6)
        self.assertTrue(np.all(output.column_indices >= 2**32))

    def test_client_rejects_mismatched_response_identity(self):
        request = make_tile_request(self.a, self.b, window=self.window)
        bad_result = messages.ProfileTileResult(
            api_version=API_VERSION,
            request_id="a-different-request",
        )
        with mock.patch.object(
            self.client.stub, "ExecuteProfileTile", return_value=bad_result
        ):
            with self.assertRaisesRegex(ValueError, "request_id"):
                self.client.execute_tile(request)


class DistributedValidationTests(unittest.TestCase):
    def test_reducer_rejects_inconsistent_or_non_correlation_slices(self):
        cases = (
            (np.array([np.nan], dtype=np.float32), np.array([0], dtype=np.int64)),
            (np.array([1.1], dtype=np.float32), np.array([0], dtype=np.int64)),
            (np.array([0.9], dtype=np.float32), np.array([-1], dtype=np.int64)),
            (np.array([-2.0], dtype=np.float32), np.array([3], dtype=np.int64)),
        )
        for values, indices in cases:
            with self.subTest(values=values, indices=indices):
                result = messages.ProfileTileResult(
                    api_version=API_VERSION,
                    column_profile=messages.ProfileSlice(
                        values=encode_array(values), indices=encode_array(indices)
                    ),
                )
                with self.assertRaises(ValueError):
                    merge_1nn_slices([result], 1)

    def test_non_loopback_worker_requires_explicit_insecure_opt_in(self):
        with self.assertRaisesRegex(ValueError, "allow_insecure_remote"):
            WorkerServer(host="0.0.0.0", backend="cpu")

    def test_worker_pool_does_not_retry_permanent_rpc_errors(self):
        class PermanentError(grpc.RpcError):
            def code(self):
                return grpc.StatusCode.INVALID_ARGUMENT

        first = mock.Mock()
        first.health.return_value = messages.WorkerHealth(
            state=messages.WORKER_STATE_SERVING
        )
        first.execute_tile.side_effect = PermanentError()
        second = mock.Mock()
        second.health.return_value = messages.WorkerHealth(
            state=messages.WORKER_STATE_SERVING
        )
        with mock.patch.object(runtime, "WorkerClient", side_effect=[first, second]):
            pool = runtime.WorkerPool(["first", "second"])
        with self.assertRaises(PermanentError):
            pool.execute_tile(messages.ProfileTileRequest())
        second.execute_tile.assert_not_called()


if __name__ == "__main__":
    unittest.main()
