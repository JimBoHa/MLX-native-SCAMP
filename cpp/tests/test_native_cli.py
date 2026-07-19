#!/usr/bin/env python3
"""End-to-end safety and compatibility tests for mlx-scamp-native."""

from __future__ import annotations

import math
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import unittest


CLI = Path(sys.argv[1]).resolve()
# Keep unittest from treating the executable path as a test selector.
sys.argv[1:] = []


def correlation(left: list[float], right: list[float]) -> float | None:
    if not all(math.isfinite(value) for value in left + right):
        return None
    mean_left = sum(left) / len(left)
    mean_right = sum(right) / len(right)
    centered_left = [value - mean_left for value in left]
    centered_right = [value - mean_right for value in right]
    norm_left = sum(value * value for value in centered_left)
    norm_right = sum(value * value for value in centered_right)
    if norm_left <= 1e-13 or norm_right <= 1e-13:
        return None
    covariance = sum(a * b for a, b in zip(centered_left, centered_right))
    return max(-1.0, min(1.0, covariance / math.sqrt(norm_left * norm_right)))


def reference(
    a: list[float],
    b: list[float],
    window: int,
    *,
    columns: bool,
    self_join: bool = False,
    aligned: bool = False,
    global_col: int = 0,
    global_row: int = 0,
) -> tuple[list[float], list[int]]:
    columns_count = len(a) - window + 1
    rows_count = len(b) - window + 1
    output_count = columns_count if columns else rows_count
    values = [math.nan] * output_count
    indexes = [-1] * output_count
    exclusion = (window + 3) // 4
    for col in range(columns_count):
        for row in range(rows_count):
            if (self_join or aligned) and abs(
                col + global_col - row - global_row
            ) < exclusion:
                continue
            corr = correlation(a[col : col + window], b[row : row + window])
            if corr is None:
                continue
            target = col if columns else row
            match = row if columns else col
            if (
                indexes[target] == -1
                or corr > values[target] + 1e-6
                or (abs(corr - values[target]) <= 1e-6 and match < indexes[target])
            ):
                values[target] = corr
                indexes[target] = match
    return values, indexes


def read_values(path: Path) -> list[float]:
    return [float(line) for line in path.read_text(encoding="utf-8").splitlines()]


def read_indexes(path: Path) -> list[int]:
    return [int(line) for line in path.read_text(encoding="utf-8").splitlines()]


def assert_values_close(
    testcase: unittest.TestCase,
    actual: list[float],
    expected: list[float],
    tolerance: float = 2e-4,
) -> None:
    testcase.assertEqual(len(actual), len(expected))
    for position, (got, wanted) in enumerate(zip(actual, expected)):
        if math.isnan(wanted):
            testcase.assertTrue(math.isnan(got), f"position {position}: {got}")
        else:
            testcase.assertAlmostEqual(
                got, wanted, delta=tolerance, msg=f"position {position}"
            )


class NativeCliTest(unittest.TestCase):
    def run_cli(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [str(CLI), *map(str, arguments)],
            check=False,
            text=True,
            capture_output=True,
        )

    def write_series(self, path: Path, values: list[float], *, trailing=True) -> None:
        text = " \t\n".join(str(value) for value in values)
        if trailing:
            text += "\n"
        path.write_text(text, encoding="utf-8")

    def job_flags(self, root: Path, input_a: Path) -> list[str]:
        return [
            "--window=4",
            f"--input_a_file_name={input_a}",
            "--single_precision",
            f"--output_a_file_name={root / 'values'}",
            f"--output_a_index_file_name={root / 'indexes'}",
        ]

    def test_help_and_variant_listing_need_no_job_arguments(self) -> None:
        for flag in ("--help", "-h", "--helpshort", "--helpfull"):
            result = self.run_cli(flag)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("--input_a_file_name", result.stdout)
        result = self.run_cli("--list_variants")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            result.stdout.strip(),
            "v0 backend=mlx-metal strategy=two-pass-diagonal-recurrence "
            "profile=1NN_INDEX precision=single device=0",
        )

    def test_self_join_supports_gflags_forms_and_pearson(self) -> None:
        series = [
            0.2,
            1.3,
            -0.7,
            2.1,
            0.5,
            -1.4,
            0.9,
            1.8,
            -0.2,
            0.4,
            1.1,
            -0.8,
            0.6,
            1.5,
            -1.1,
            0.3,
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_a = root / "a.txt"
            self.write_series(input_a, series, trailing=False)
            result = self.run_cli(
                "-window",
                "5",
                "--input_a_file_name",
                str(input_a),
                "--single_precision=true",
                "--output_pearson",
                "--noprint_debug_info",
                "--gpus",
                "0",
                "--output_a_file_name",
                str(root / "values"),
                f"--output_a_index_file_name={root / 'indexes'}",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout, "")
            self.assertIn("computing on MLX Metal device 0", result.stderr)
            expected_values, expected_indexes = reference(
                series, series, 5, columns=True, self_join=True
            )
            assert_values_close(self, read_values(root / "values"), expected_values)
            self.assertEqual(read_indexes(root / "indexes"), expected_indexes)
            self.assertFalse(list(root.glob(".mlx-scamp-*")))

    def test_default_output_is_euclidean(self) -> None:
        series = [0.2, 1.3, -0.7, 2.1, 0.5, -1.4, 0.9, 1.8, -0.2, 0.4, 1.1]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_a = root / "a"
            self.write_series(input_a, series)
            result = self.run_cli(*self.job_flags(root, input_a))
            self.assertEqual(result.returncode, 0, result.stderr)
            correlations, indexes = reference(
                series, series, 4, columns=True, self_join=True
            )
            distances = [
                math.nan if math.isnan(value) else math.sqrt(max(8 * (1 - value), 0))
                for value in correlations
            ]
            assert_values_close(self, read_values(root / "values"), distances, 5e-4)
            self.assertEqual(read_indexes(root / "indexes"), indexes)

    def test_ab_join_writes_four_outputs_with_aligned_offsets(self) -> None:
        a = [0.1, 1.0, -0.3, 0.8, 1.7, -1.2, 0.4, 1.1, -0.6, 0.2, 1.5]
        b = [-0.8, 0.5, 1.4, -0.1, 0.9, -1.5, 0.3, 1.8, -0.4, 0.6, 1.2, -0.7]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_a, input_b = root / "a", root / "b"
            self.write_series(input_a, a)
            self.write_series(input_b, b)
            paths = [root / name for name in ("av", "ai", "bv", "bi")]
            result = self.run_cli(
                "--window=4",
                f"--input_a_file_name={input_a}",
                f"--input_b_file_name={input_b}",
                "--single_precision",
                "--keep_rows=true",
                "--output_pearson",
                "--aligned",
                "--global_col=3",
                "--global_row=1",
                f"--output_a_file_name={paths[0]}",
                f"--output_a_index_file_name={paths[1]}",
                f"--output_b_file_name={paths[2]}",
                f"--output_b_index_file_name={paths[3]}",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            expected_a, expected_ai = reference(
                a, b, 4, columns=True, aligned=True, global_col=3, global_row=1
            )
            expected_b, expected_bi = reference(
                a, b, 4, columns=False, aligned=True, global_col=3, global_row=1
            )
            assert_values_close(self, read_values(paths[0]), expected_a)
            self.assertEqual(read_indexes(paths[1]), expected_ai)
            assert_values_close(self, read_values(paths[2]), expected_b)
            self.assertEqual(read_indexes(paths[3]), expected_bi)

    def test_nonfinite_and_flat_windows_emit_nan_and_minus_one(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_a = root / "invalid"
            self.write_series(input_a, [math.nan, math.inf, 2.0, 2.0, 2.0, 2.0])
            result = self.run_cli(
                *self.job_flags(root, input_a), "--output_pearson=false"
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(all(math.isnan(value) for value in read_values(root / "values")))
            self.assertTrue(all(value == -1 for value in read_indexes(root / "indexes")))

    def test_unsupported_requests_fail_before_input_or_output_io(self) -> None:
        unsupported = [
            ("--profile_type=SUM_THRESH", "1NN_INDEX only"),
            ("--profile_type=1NN", "1NN_INDEX only"),
            ("--profile_type=ALL_NEIGHBORS", "1NN_INDEX only"),
            ("--profile_type=MATRIX_SUMMARY", "1NN_INDEX only"),
            ("--double_precision", "double and ultra"),
            ("--ultra_precision", "double and ultra"),
            ("--no_gpu", "CPU path"),
            ("--num_cpu_workers=2", "must be 0"),
            ("--autotune", "fixed strategy"),
            ("--max_tile_size=512000", "does not yet tile"),
            ("--threshold=0.5", "reducer not implemented"),
            ("--reduced_height=20", "reducer not implemented"),
            ("--reduced_width=20", "reducer not implemented"),
            ("--max_matches_per_column=3", "reducer not implemented"),
            ("--gpus=1", "empty or 0"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            values, indexes = root / "values", root / "indexes"
            values.write_text("keep-values", encoding="utf-8")
            indexes.write_text("keep-indexes", encoding="utf-8")
            common = [
                "--window=4",
                f"--input_a_file_name={root / 'missing-input'}",
                "--single_precision",
                f"--output_a_file_name={values}",
                f"--output_a_index_file_name={indexes}",
            ]
            for flag, message in unsupported:
                with self.subTest(flag=flag):
                    result = self.run_cli(*common, flag)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn(message, result.stderr)
                    self.assertEqual(values.read_text(encoding="utf-8"), "keep-values")
                    self.assertEqual(indexes.read_text(encoding="utf-8"), "keep-indexes")

    def test_parser_and_join_validation_are_strict(self) -> None:
        cases = [
            (["--window=999999999999999999999"], "out of range"),
            (["--window=4x"], "invalid integer"),
            (["--single_precision=maybe"], "invalid boolean"),
            (["--unknown_flag=1"], "unknown flag"),
            (["positional"], "unexpected positional"),
            (["--window=4", "--single_precision"], "primary input filename"),
            (
                [
                    "--window=4",
                    "--single_precision",
                    "--input_a_file_name=missing",
                    "--global_row=0",
                ],
                "must be supplied together",
            ),
        ]
        for arguments, message in cases:
            with self.subTest(arguments=arguments):
                result = self.run_cli(*arguments)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(message, result.stderr)

    def test_malformed_input_and_compute_failure_preserve_existing_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_a = root / "a"
            values, indexes = root / "values", root / "indexes"
            values.write_text("old-values", encoding="utf-8")
            indexes.write_text("old-indexes", encoding="utf-8")
            input_a.write_text("0 1 nope 2", encoding="utf-8")
            result = self.run_cli(*self.job_flags(root, input_a))
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("could not parse token 3", result.stderr)
            self.assertEqual(values.read_text(encoding="utf-8"), "old-values")
            self.assertEqual(indexes.read_text(encoding="utf-8"), "old-indexes")

            input_a.write_text("0 1 2", encoding="utf-8")
            result = self.run_cli(*self.job_flags(root, input_a))
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("at least the window size", result.stderr)
            self.assertEqual(values.read_text(encoding="utf-8"), "old-values")
            self.assertEqual(indexes.read_text(encoding="utf-8"), "old-indexes")

    def test_staging_failure_does_not_partially_replace_outputs(self) -> None:
        series = [0.2, 1.3, -0.7, 2.1, 0.5, -1.4, 0.9, 1.8]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_a = root / "a"
            self.write_series(input_a, series)
            values = root / "values"
            values.write_text("old-values", encoding="utf-8")
            readonly = root / "readonly"
            readonly.mkdir()
            readonly.chmod(0o555)
            try:
                result = self.run_cli(
                    "--window=4",
                    f"--input_a_file_name={input_a}",
                    "--single_precision",
                    f"--output_a_file_name={values}",
                    f"--output_a_index_file_name={readonly / 'indexes'}",
                )
            finally:
                readonly.chmod(0o755)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("temporary output", result.stderr)
            self.assertEqual(values.read_text(encoding="utf-8"), "old-values")
            self.assertFalse(list(root.glob(".mlx-scamp-*")))

    def test_commit_failure_restores_every_existing_output(self) -> None:
        if not hasattr(os, "chflags") or not hasattr(stat, "UF_IMMUTABLE"):
            self.skipTest("immutable file flags are unavailable")
        series = [0.2, 1.3, -0.7, 2.1, 0.5, -1.4, 0.9, 1.8]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_a = root / "a"
            self.write_series(input_a, series)
            values, indexes = root / "values", root / "indexes"
            values.write_text("old-values", encoding="utf-8")
            indexes.write_text("old-indexes", encoding="utf-8")
            os.chflags(indexes, stat.UF_IMMUTABLE)
            try:
                result = self.run_cli(*self.job_flags(root, input_a))
            finally:
                os.chflags(indexes, 0)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("stage existing output", result.stderr)
            self.assertEqual(values.read_text(encoding="utf-8"), "old-values")
            self.assertEqual(indexes.read_text(encoding="utf-8"), "old-indexes")
            self.assertFalse(list(root.glob(".mlx-scamp-*")))

    def test_input_output_aliases_are_rejected_conservatively(self) -> None:
        series = [0.2, 1.3, -0.7, 2.1, 0.5, -1.4, 0.9, 1.8]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_a = root / "Input-Series"
            original = " ".join(str(value) for value in series)
            input_a.write_text(original, encoding="utf-8")
            candidates: list[tuple[str, Path, str]] = [
                ("canonical", input_a, "aliases input")
            ]

            symlink = root / "symlink-output"
            symlink.symlink_to(input_a)
            candidates.append(("symlink", symlink, "output symlinks"))

            hardlink = root / "hardlink-output"
            os.link(input_a, hardlink)
            candidates.append(("hardlink", hardlink, "aliases input"))
            candidates.append(("casefold", root / "input-series", "aliases input"))

            unicode_input = root / "caf\u00e9"
            unicode_input.write_text(original, encoding="utf-8")
            candidates.append(("unicode", root / "cafe\u0301", "aliases input"))

            for label, output, message in candidates:
                with self.subTest(alias=label):
                    selected_input = unicode_input if label == "unicode" else input_a
                    result = self.run_cli(
                        "--window=4",
                        f"--input_a_file_name={selected_input}",
                        "--single_precision",
                        f"--output_a_file_name={output}",
                        f"--output_a_index_file_name={root / (label + '-index')}",
                    )
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn(message, result.stderr)
                    self.assertEqual(selected_input.read_text(encoding="utf-8"), original)

    def test_unrelated_output_symlink_and_target_are_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_a = root / "a"
            self.write_series(input_a, [0.2, 1.3, -0.7, 2.1, 0.5, -1.4])
            target = root / "unrelated-target"
            target.write_text("old-target", encoding="utf-8")
            output = root / "output-link"
            output.symlink_to(target)
            result = self.run_cli(
                "--window=3",
                f"--input_a_file_name={input_a}",
                "--single_precision",
                f"--output_a_file_name={output}",
                f"--output_a_index_file_name={root / 'indexes'}",
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("output symlinks", result.stderr)
            self.assertTrue(output.is_symlink())
            self.assertEqual(target.read_text(encoding="utf-8"), "old-target")

    def test_output_aliases_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_a = root / "a"
            self.write_series(input_a, [0.2, 1.3, -0.7, 2.1, 0.5, -1.4])
            same = root / "same"
            result = self.run_cli(
                "--window=3",
                f"--input_a_file_name={input_a}",
                "--single_precision",
                f"--output_a_file_name={same}",
                f"--output_a_index_file_name={same}",
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("alias each other", result.stderr)
            self.assertFalse(same.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
