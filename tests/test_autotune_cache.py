import json
import os
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from unittest import mock

from mlx_native_scamp import _autotune_cache as cache


def _key(
    profile="1nn_index",
    *,
    n_a=1024,
    n_b=1024,
    m=64,
    self_join=True,
    route="auto",
    aligned=False,
    max_tile_size=None,
    threshold_density=None,
    k=None,
    matrix_shape=None,
):
    return cache.make_workload_key(
        profile,
        "single",
        route,
        n_a,
        n_b,
        m,
        self_join=self_join,
        aligned=aligned,
        dtype_class="float32",
        max_tile_size=max_tile_size,
        threshold_density=threshold_density,
        k=k,
        matrix_shape=matrix_shape,
    )


def _record(key, candidate, created_ns=1):
    return cache.new_record(
        key,
        candidate,
        duration_ns=100,
        trials=3,
        created_ns=created_ns,
    )


class AutotuneCacheTests(unittest.TestCase):
    def setUp(self):
        cache._sidecar_exists_once.cache_clear()
        cache._load_records_once.cache_clear()

    def tearDown(self):
        cache._sidecar_exists_once.cache_clear()
        cache._load_records_once.cache_clear()

    def test_sidecar_never_reuses_upstream_cache_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            upstream = Path(temp_dir) / "autotune.txt"
            upstream.write_text("SCAMP_AUTOTUNE_V1\n", encoding="utf-8")

            record = _record(
                _key(), "1nn_index:cpu:rows-64"
            )
            written = cache.save_record(record, str(upstream))

            self.assertEqual(
                upstream.with_name("autotune.txt.mlx.json"), written
            )
            self.assertEqual(
                "SCAMP_AUTOTUNE_V1\n",
                upstream.read_text(encoding="utf-8"),
            )
            self.assertEqual(0o600, written.stat().st_mode & 0o777)
            payload = json.loads(written.read_text(encoding="utf-8"))
            self.assertEqual(cache.CACHE_FORMAT, payload["format"])

    def test_default_path_order_and_sidecar_suffix(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            override = str(Path(temp_dir) / "explicit.txt")
            with mock.patch.dict(
                os.environ,
                {
                    "SCAMP_AUTOTUNE_CACHE": override,
                    "XDG_CACHE_HOME": str(Path(temp_dir) / "xdg"),
                },
                clear=False,
            ):
                self.assertEqual(
                    Path(f"{override}.mlx.json"), cache.sidecar_path()
                )

            with mock.patch.dict(
                os.environ,
                {"SCAMP_AUTOTUNE_CACHE": "", "XDG_CACHE_HOME": ""},
                clear=False,
            ), mock.patch.object(Path, "home", return_value=Path(temp_dir)):
                self.assertEqual(
                    Path(temp_dir) / ".cache/scamp/autotune.txt.mlx.json",
                    cache.sidecar_path(),
                )

            with mock.patch.dict(
                os.environ,
                {"SCAMP_AUTOTUNE_CACHE": "", "XDG_CACHE_HOME": "relative"},
                clear=False,
            ), mock.patch.object(Path, "home", return_value=Path(temp_dir)):
                self.assertEqual(
                    Path(temp_dir) / ".cache/scamp/autotune.txt.mlx.json",
                    cache.sidecar_path(),
                )

    def test_workload_keys_separate_sizes_shapes_and_profile_knobs(self):
        baseline = _key()
        self.assertNotEqual(baseline, _key(n_a=4096, n_b=4096))
        self.assertNotEqual(baseline, _key(m=256))
        self.assertNotEqual(
            _key(self_join=False, n_a=2048, n_b=256),
            _key(self_join=False, n_a=256, n_b=2048),
        )
        self.assertNotEqual(
            _key(self_join=False, aligned=False),
            _key(self_join=False, aligned=True),
        )
        self.assertNotEqual(baseline, _key(max_tile_size=1024))
        self.assertNotEqual(
            _key("sum_thresh", threshold_density=0.01),
            _key("sum_thresh", threshold_density=0.5),
        )
        self.assertNotEqual(_key("knn", k=3), _key("knn", k=17))
        self.assertNotEqual(
            _key("matrix_summary", matrix_shape=(8, 32)),
            _key("matrix_summary", matrix_shape=(32, 32)),
        )

    def test_invalid_workload_parameters_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "profile"):
            _key("unknown")
        with self.assertRaisesRegex(ValueError, "positive k"):
            _key("knn")
        with self.assertRaisesRegex(ValueError, "between zero and one"):
            _key("sum_thresh", threshold_density=2.0)
        with self.assertRaisesRegex(ValueError, "positive shape"):
            _key("matrix_summary", matrix_shape=(0, 2))

    def test_lazy_lookup_reads_once_and_ignores_stale_rows(self):
        current = _record(_key(), "1nn_index:cpu:rows-64")
        stale = replace(current, manifest_id="0" * 64, created_ns=2)
        malformed = {"candidate": "missing-fields"}
        payload = cache._empty_payload()
        payload["records"] = {
            cache.record_id(current): cache._record_to_dict(current),
            "stale": cache._record_to_dict(stale),
            "malformed": malformed,
        }

        with mock.patch.object(
            cache, "_sidecar_exists_once", return_value=True
        ), mock.patch.object(
            cache, "_read_payload", return_value=payload
        ) as read_payload:
            first = cache.lookup_record(current.key, "cache.txt")
            second = cache.lookup_record(current.key, "cache.txt")

        self.assertEqual(current, first)
        self.assertIs(first, second)
        read_payload.assert_called_once()

    def test_missing_sidecar_skips_hardware_fingerprint_and_caches_stat(self):
        with tempfile.TemporaryDirectory() as temp_dir, mock.patch.object(
            cache,
            "environment_id",
            side_effect=AssertionError("hardware fingerprinted"),
        ):
            upstream = str(Path(temp_dir) / "autotune.txt")
            self.assertEqual((), cache.load_records(upstream))
            self.assertEqual((), cache.load_records(upstream))

        info = cache._sidecar_exists_once.cache_info()
        self.assertEqual(1, info.misses)
        self.assertEqual(1, info.hits)

    def test_save_invalidates_a_cached_missing_sidecar(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            upstream = str(Path(temp_dir) / "autotune.txt")
            self.assertEqual((), cache.load_records(upstream))
            record = _record(_key(), "1nn_index:cpu:rows-64")

            cache.save_record(record, upstream)

            self.assertEqual(record, cache.lookup_record(record.key, upstream))

    def test_malformed_and_oversized_files_are_tolerated(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            upstream = str(Path(temp_dir) / "autotune.txt")
            sidecar = cache.sidecar_path(upstream)
            sidecar.write_text("not json", encoding="utf-8")
            self.assertEqual((), cache.load_records(upstream))

            cache._load_records_once.cache_clear()
            sidecar.write_bytes(b"x" * 32)
            with mock.patch.object(cache, "MAX_CACHE_BYTES", 16):
                self.assertEqual((), cache.load_records(upstream))

    def test_save_preserves_malformed_or_future_sidecar(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            upstream = str(Path(temp_dir) / "autotune.txt")
            sidecar = cache.sidecar_path(upstream)
            record = _record(_key(), "1nn_index:cpu:rows-64")
            for contents in (
                "not json",
                json.dumps(
                    {
                        "format": "MLX_SCAMP_AUTOTUNE_V3",
                        "schema": 3,
                        "records": {},
                        "environments": {},
                    }
                ),
            ):
                with self.subTest(contents=contents[:16]):
                    sidecar.write_text(contents, encoding="utf-8")
                    with self.assertRaisesRegex(
                        ValueError, "reset.*different cache_path"
                    ):
                        cache.save_record(record, upstream)
                    self.assertEqual(contents, sidecar.read_text(encoding="utf-8"))

    def test_same_key_keeps_newest_record(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            upstream = str(Path(temp_dir) / "autotune.txt")
            key = _key()
            newest = _record(key, "1nn_index:cpu:rows-256", 2)
            cache.save_record(
                _record(key, "1nn_index:cpu:rows-64", 1), upstream
            )
            cache.save_record(newest, upstream)

            self.assertEqual(newest, cache.lookup_record(key, upstream))

    def test_saved_replacement_wins_when_clock_moves_backward(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            upstream = str(Path(temp_dir) / "autotune.txt")
            key = _key()
            cache.save_record(
                _record(key, "1nn_index:cpu:rows-64", 100), upstream
            )
            replacement = _record(
                key, "1nn_index:cpu:rows-256", 99
            )
            cache.save_record(replacement, upstream)

            self.assertEqual(replacement, cache.lookup_record(key, upstream))

    def test_save_sanitizes_unknown_near_limit_payload_fields(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            upstream = str(Path(temp_dir) / "autotune.txt")
            sidecar = cache.sidecar_path(upstream)
            payload = cache._empty_payload()
            payload["junk"] = "x" * (cache.MAX_CACHE_BYTES - 512)
            sidecar.write_text(json.dumps(payload), encoding="utf-8")

            record = _record(_key(), "1nn_index:cpu:rows-64")
            cache.save_record(record, upstream)

            stored = json.loads(sidecar.read_text(encoding="utf-8"))
            self.assertNotIn("junk", stored)
            self.assertEqual(
                {"format", "schema", "environments", "records"},
                set(stored),
            )
            self.assertIsNotNone(cache.lookup_record(record.key, upstream))

    def test_save_caps_environments_and_their_records(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            upstream = str(Path(temp_dir) / "autotune.txt")
            sidecar = cache.sidecar_path(upstream)
            payload = cache._empty_payload()
            base = _record(_key(), "1nn_index:cpu:rows-64")
            for index in range(cache.MAX_ENVIRONMENTS + 8):
                environment = f"{index + 1:064x}"
                item = replace(
                    base,
                    key=replace(base.key, work_bucket=f"2^{index + 10}"),
                    environment_id=environment,
                    created_ns=index + 1,
                )
                payload["records"][cache.record_id(item)] = cache._record_to_dict(item)
                payload["environments"][environment] = {"junk": "ignored"}
            sidecar.write_text(json.dumps(payload), encoding="utf-8")

            cache.save_record(base, upstream)
            stored = json.loads(sidecar.read_text(encoding="utf-8"))
            environment_ids = set(stored["environments"])

            self.assertLessEqual(len(environment_ids), cache.MAX_ENVIRONMENTS)
            self.assertIn(cache.environment_id(), environment_ids)
            self.assertTrue(
                all(
                    row["environment_id"] in environment_ids
                    for row in stored["records"].values()
                )
            )

    def test_concurrent_writers_preserve_independent_records(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            upstream = str(Path(temp_dir) / "autotune.txt")
            records = [
                _record(
                    _key(n_a=2 ** (index + 8), n_b=2 ** (index + 8)),
                    "1nn_index:cpu:rows-64",
                    index + 1,
                )
                for index in range(8)
            ]
            with ThreadPoolExecutor(max_workers=8) as pool:
                list(pool.map(lambda item: cache.save_record(item, upstream), records))

            loaded = cache.load_records(upstream)
            self.assertEqual(8, len(loaded))
            self.assertEqual(
                {cache.record_id(item) for item in records},
                {cache.record_id(item) for item in loaded},
            )

    def test_record_validation_rejects_route_and_precision_mismatches(self):
        cpu_key = _key(route="cpu")
        with self.assertRaisesRegex(ValueError, "invalid"):
            cache.new_record(
                cpu_key,
                "1nn_index:metal-diagonal",
                duration_ns=1,
                trials=1,
            )

        double_key = replace(_key(route="metal"), precision="double")
        value = cache._record_to_dict(
            cache.TuningRecord(
                double_key,
                "1nn_index:portable_metal:rows-64",
                1,
                1,
                cache.environment_id(),
                cache.candidate_manifest_id(),
                1,
            )
        )
        self.assertIsNone(cache._record_from_dict(value))

    def test_environment_fallback_uses_no_subprocess(self):
        cache.environment_fingerprint.cache_clear()
        cache.environment_id.cache_clear()
        try:
            with mock.patch.object(
                cache, "_device_information", return_value={}
            ), mock.patch.object(
                cache.os, "sysconf", side_effect=OSError
            ), mock.patch.object(
                cache,
                "_hardware_identity",
                return_value={"hardware_model": "Mac14,6"},
            ), mock.patch(
                "subprocess.check_output",
                side_effect=AssertionError("environment probe spawned"),
            ):
                fingerprint = cache.environment_fingerprint()
        finally:
            cache.environment_fingerprint.cache_clear()
            cache.environment_id.cache_clear()

        self.assertEqual({}, fingerprint["device"])
        self.assertEqual("Mac14,6", fingerprint["hardware"]["hardware_model"])
        self.assertIsNone(fingerprint["memory_bytes"])
        self.assertNotIn("subprocess", cache.__dict__)

    def test_fallback_hardware_model_changes_environment_id(self):
        def identifier(model):
            cache.environment_fingerprint.cache_clear()
            cache.environment_id.cache_clear()
            with mock.patch.object(
                cache, "_device_information", return_value={}
            ), mock.patch.object(
                cache,
                "_hardware_identity",
                return_value={"hardware_model": model},
            ):
                return cache.environment_id()

        try:
            first = identifier("Mac14,6")
            second = identifier("Mac15,3")
        finally:
            cache.environment_fingerprint.cache_clear()
            cache.environment_id.cache_clear()

        self.assertNotEqual(first, second)

    def test_huge_json_integer_and_deep_nesting_are_tolerated(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            upstream = str(Path(temp_dir) / "autotune.txt")
            sidecar = cache.sidecar_path(upstream)
            sidecar.write_text(
                '{"format":"MLX_SCAMP_AUTOTUNE_V2","schema":2,'
                f'"records":{{"bad":{("9" * 5000)}}},'
                '"environments":{}}',
                encoding="utf-8",
            )
            self.assertEqual((), cache.load_records(upstream))

            cache._load_records_once.cache_clear()
            sidecar.write_text("[" * 1200 + "]" * 1200, encoding="utf-8")
            self.assertEqual((), cache.load_records(upstream))

    def test_reset_removes_only_the_mlx_sidecar(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            upstream = Path(temp_dir) / "autotune.txt"
            upstream.write_text("SCAMP_AUTOTUNE_V1\n", encoding="utf-8")
            cache.save_record(
                _record(_key(), "1nn_index:cpu:rows-64"), str(upstream)
            )

            self.assertTrue(cache.reset_cache(str(upstream)))
            self.assertFalse(cache.sidecar_path(str(upstream)).exists())
            self.assertTrue(upstream.exists())
            self.assertFalse(cache.reset_cache(str(upstream)))

    def test_candidate_manifest_is_stable_and_complete(self):
        manifest = cache.candidate_manifest_id()
        self.assertEqual(64, len(manifest))
        self.assertEqual(cache.PROFILE_FAMILIES, {row.profile for row in cache.STRATEGIES})
        self.assertEqual(len(cache.STRATEGIES), len(cache.STRATEGY_BY_NAME))
        for strategy in cache.STRATEGIES:
            self.assertEqual(strategy, cache.STRATEGY_BY_NAME[strategy.name])
            self.assertEqual(
                dict(strategy.parameters),
                cache.new_record(
                    _key(
                        strategy.profile,
                        self_join=strategy.profile != "bidirectional_ab",
                        threshold_density=(
                            0.1 if strategy.profile == "sum_thresh" else None
                        ),
                        k=3 if strategy.profile == "knn" else None,
                        matrix_shape=(4, 4)
                        if strategy.profile == "matrix_summary"
                        else None,
                    ),
                    strategy.name,
                    1,
                    1,
                ).parameters,
            )


if __name__ == "__main__":
    unittest.main()
