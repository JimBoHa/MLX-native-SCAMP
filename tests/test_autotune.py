import inspect
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import mock

import mlx.core as mx
import numpy as np

import mlx_native_scamp
import mlx_native_scamp._autotune as mlx_autotune
import mlx_native_scamp.core as scamp_core
import pyscamp


def _config(**overrides):
    values = {
        "preferred_1nn_backend": "metal",
        "preferred_portable_backend": "cpu",
        "cpu_block_rows": 128,
        "metal_block_rows": 512,
        "metal_threadgroup_width": 64,
        "input_length": 4096,
        "timings_ns": {"candidate": 100},
    }
    values.update(overrides)
    return mlx_autotune.TuningConfig(**values)


class AutotuneTests(unittest.TestCase):
    def test_signature_and_exports_match_upstream(self):
        self.assertIs(pyscamp.autotune, mlx_native_scamp.autotune)
        self.assertIn("autotune", pyscamp.__all__)
        self.assertIn("autotune", mlx_native_scamp.__all__)
        signature = inspect.signature(pyscamp.autotune)
        self.assertEqual(["devices", "cache_path"], list(signature.parameters))
        self.assertIsNone(signature.parameters["devices"].default)
        self.assertEqual("", signature.parameters["cache_path"].default)

    def test_device_and_cache_arguments_are_validated(self):
        for devices in (0, "0", [0, 1.5]):
            with self.subTest(devices=devices):
                with self.assertRaisesRegex(TypeError, "devices"):
                    pyscamp.autotune(devices=devices)

        for devices in ([1], [-1], [0, 0, 0, 1]):
            with self.subTest(devices=devices):
                with self.assertRaisesRegex(ValueError, "GPU device ID"):
                    pyscamp.autotune(devices=devices)

        with self.assertRaisesRegex(TypeError, "cache_path"):
            pyscamp.autotune(cache_path=None)

    def test_no_metal_device_matches_upstream_no_gpu_error(self):
        with mock.patch.object(mx.metal, "is_available", return_value=False):
            with self.assertRaisesRegex(ValueError, "No Metal device"):
                pyscamp.autotune()

    def test_cache_path_resolution_matches_upstream_order(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            environment = {
                "SCAMP_AUTOTUNE_CACHE": f"{temp_dir}/explicit.txt",
                "XDG_CACHE_HOME": f"{temp_dir}/xdg",
            }
            with mock.patch.dict(os.environ, environment, clear=False):
                self.assertEqual(
                    Path(environment["SCAMP_AUTOTUNE_CACHE"]),
                    mlx_autotune._default_cache_path(),
                )
            with mock.patch.dict(
                os.environ,
                {"XDG_CACHE_HOME": f"{temp_dir}/xdg"},
                clear=True,
            ):
                self.assertEqual(
                    Path(temp_dir) / "xdg" / "scamp" / "autotune.txt",
                    mlx_autotune._default_cache_path(),
                )

    def test_sweep_selects_independent_launch_winners(self):
        result = (np.zeros(4, dtype=np.float32), np.zeros(4, dtype=np.int32))

        def benchmark(*_args):
            return result

        def deterministic_time(_benchmark, _reference, **kwargs):
            if kwargs["workload"] == "metal_1nn":
                return {32: 50, 64: 5, 128: 20, 256: 30}[
                    kwargs["threadgroup_width"]
                ]
            if kwargs["device"] == "cpu":
                return {64: 40, 128: 10, 256: 30, 512: 20}[
                    kwargs["block_rows"]
                ]
            return {64: 60, 128: 45, 256: 35, 512: 25}[
                kwargs["block_rows"]
            ]

        with mock.patch.object(
            mlx_autotune,
            "_benchmark_trials",
            side_effect=deterministic_time,
        ):
            config = mlx_autotune._run_sweep(
                benchmark, input_length=512, warmups=0, trials=1
            )

        self.assertEqual("metal", config.preferred_1nn_backend)
        self.assertEqual("cpu", config.preferred_portable_backend)
        self.assertEqual(128, config.cpu_block_rows)
        self.assertEqual(512, config.metal_block_rows)
        self.assertEqual(64, config.metal_threadgroup_width)

    def test_candidate_is_rejected_before_a_bad_result_can_be_cached(self):
        reference = (
            np.array([0.5], dtype=np.float32),
            np.array([2], dtype=np.int32),
        )
        candidate = (
            np.array([0.5], dtype=np.float32),
            np.array([3], dtype=np.int32),
        )
        with self.assertRaisesRegex(RuntimeError, "correctness check"):
            mlx_autotune._assert_equivalent(candidate, reference, "bad")

    def test_autotune_atomically_writes_and_reloads_current_device(self):
        config = _config()
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir) / "nested" / "autotune.txt"
            with mock.patch.object(
                mx.metal, "is_available", return_value=True
            ), mock.patch.object(
                mlx_autotune, "_run_sweep", return_value=config
            ), mock.patch.object(
                mlx_autotune, "_device_key", return_value="test-device"
            ), redirect_stdout(StringIO()):
                tuned = pyscamp.autotune([], str(cache_path))
                loaded = mlx_autotune.load_tuning(str(cache_path))

            self.assertEqual(1, tuned)
            self.assertEqual(config, loaded)
            self.assertEqual(0o600, cache_path.stat().st_mode & 0o777)
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            self.assertEqual(mlx_autotune.CACHE_FORMAT, payload["format"])
            self.assertEqual(config.cpu_block_rows, payload["records"]["test-device"]["cpu_block_rows"])
            self.assertFalse(list(cache_path.parent.glob("*.tmp")))

    def test_runtime_cache_changes_real_stream_and_launch_parameters(self):
        config = _config(
            preferred_1nn_backend="cpu", preferred_portable_backend="metal"
        )
        observed = []

        def record_run(*_args, **kwargs):
            observed.append((mx.default_device(), kwargs))
            return (
                np.zeros(5, dtype=np.float32),
                np.zeros(5, dtype=np.int32),
            )

        series = np.arange(8, dtype=np.float32)
        with mock.patch.object(
            scamp_core, "load_tuning", return_value=config
        ), mock.patch.object(scamp_core, "_run_profile", side_effect=record_run):
            pyscamp.selfjoin(series, 4, precision="single")
            pyscamp.selfjoin(series, 4, precision="single", gpus=[0])
            pyscamp.selfjoin_sum(series, 4, precision="single")

        self.assertEqual(mx.cpu, observed[0][0])
        self.assertEqual(128, observed[0][1]["block_rows"])
        self.assertEqual(mx.gpu, observed[1][0])
        self.assertEqual(512, observed[1][1]["block_rows"])
        self.assertEqual(64, observed[1][1]["metal_threadgroup_width"])
        self.assertEqual(mx.gpu, observed[2][0])
        self.assertEqual(512, observed[2][1]["block_rows"])

    def test_foreign_or_malformed_cache_is_never_applied(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir) / "autotune.txt"
            cache_path.write_text("SCAMP_AUTOTUNE_V1\n", encoding="utf-8")
            self.assertIsNone(mlx_autotune.load_tuning(str(cache_path)))
            with self.assertRaisesRegex(ValueError, "Unable to read|not in"):
                mlx_autotune._save_tuning(_config(), cache_path)

    def test_runtime_cache_file_is_loaded_only_once_per_process(self):
        config = _config()
        payload = {
            "format": mlx_autotune.CACHE_FORMAT,
            "records": {
                "cached-device": {
                    "preferred_1nn_backend": config.preferred_1nn_backend,
                    "preferred_portable_backend": config.preferred_portable_backend,
                    "cpu_block_rows": config.cpu_block_rows,
                    "metal_block_rows": config.metal_block_rows,
                    "metal_threadgroup_width": config.metal_threadgroup_width,
                    "input_length": config.input_length,
                    "timings_ns": config.timings_ns,
                }
            },
        }
        mlx_autotune._load_tuning_once.cache_clear()
        try:
            with mock.patch.object(
                mlx_autotune, "_read_payload", return_value=payload
            ) as read_payload, mock.patch.object(
                mlx_autotune, "_device_key", return_value="cached-device"
            ):
                first = mlx_autotune.load_tuning("cache.txt")
                second = mlx_autotune.load_tuning("cache.txt")
        finally:
            mlx_autotune._load_tuning_once.cache_clear()
        self.assertEqual(config, first)
        self.assertIs(first, second)
        read_payload.assert_called_once()

    def test_device_key_invalidates_software_and_kernel_changes(self):
        mlx_autotune._device_key.cache_clear()
        try:
            with mock.patch.object(
                mlx_autotune,
                "_device_information",
                return_value={
                    "device_name": "Apple Test",
                    "architecture": "applegpu_test",
                    "memory_size": 16,
                },
            ), mock.patch.object(
                mlx_autotune, "_mlx_version", return_value="1.2.3"
            ):
                key = mlx_autotune._device_key()
        finally:
            mlx_autotune._device_key.cache_clear()
        self.assertIn("Apple Test", key)
        self.assertIn("applegpu_test", key)
        self.assertIn("mlx-1.2.3", key)
        self.assertIn(mlx_autotune.KERNEL_REVISION, key)

    def test_environment_controls_are_bounded_and_mockable(self):
        with mock.patch.object(mx.metal, "is_available", return_value=True):
            for name, value, message in (
                ("SCAMP_AUTOTUNE_INPUT_LENGTH", "not-an-int", "integer"),
                ("SCAMP_AUTOTUNE_INPUT_LENGTH", "255", "256"),
                ("SCAMP_AUTOTUNE_WARMUP_RUNS", "-1", "0"),
                ("MLX_SCAMP_AUTOTUNE_TRIALS", "0", "1"),
            ):
                with self.subTest(name=name, value=value):
                    with mock.patch.dict(os.environ, {name: value}, clear=False):
                        with self.assertRaisesRegex(ValueError, message):
                            pyscamp.autotune(cache_path="unused")

    def test_list_variants_describes_every_swept_choice(self):
        descriptions = mlx_autotune.variant_descriptions()
        self.assertEqual(
            len(mlx_autotune.BLOCK_ROW_CANDIDATES)
            + len(mlx_autotune.THREADGROUP_CANDIDATES),
            len(descriptions),
        )
        self.assertTrue(any("block_rows=512" in row for row in descriptions))
        self.assertTrue(any("threadgroup_width=256" in row for row in descriptions))


if __name__ == "__main__":
    unittest.main()
