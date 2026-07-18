from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from mlx_native_scamp import cli


class CLIParityTests(unittest.TestCase):
    def parse(self, *arguments: str):
        return cli.build_parser().parse_args(arguments)

    def test_list_variants_does_not_require_job_arguments(self):
        stdout = io.StringIO()
        with mock.patch.object(cli, "_variant_descriptions", return_value=["v0 device=metal"]):
            status = cli.run(self.parse("--list_variants", "--gpus=not-parsed"), stdout=stdout)
        self.assertEqual(0, status)
        self.assertEqual("v0 device=metal\n", stdout.getvalue())

    def test_autotune_lazily_calls_public_api_without_input(self):
        args = self.parse("--autotune", "--gpus=0")
        with mock.patch.object(cli, "_run_autotune") as autotune:
            self.assertEqual(0, cli.run(args))
        autotune.assert_called_once_with([0])

    def test_autotune_failures_become_clean_cli_errors(self):
        fake_module = mock.Mock()
        fake_module.autotune.side_effect = RuntimeError("benchmark failed")
        with mock.patch.object(cli.importlib, "import_module", return_value=fake_module):
            with self.assertRaisesRegex(cli.CLIError, "Autotune failed: benchmark failed"):
                cli._run_autotune(None)

    def test_reads_stdin_and_writes_upstream_1nn_files(self):
        with tempfile.TemporaryDirectory() as directory:
            profile_path = Path(directory, "profile.txt")
            index_path = Path(directory, "index.txt")
            args = self.parse(
                "--window=3",
                "--input_a_file_name=-",
                f"--output_a_file_name={profile_path}",
                f"--output_a_index_file_name={index_path}",
                "--output_pearson",
            )
            series = "0\n1\n0\n2\n0\n1\n0\n"
            self.assertEqual(0, cli.run(args, stdin=io.StringIO(series)))

            expected_profile, expected_index = cli.selfjoin(
                np.fromstring(series, sep="\n"), 3, pearson=True
            )
            np.testing.assert_allclose(
                np.loadtxt(profile_path), expected_profile, equal_nan=True, rtol=1e-7, atol=1e-7
            )
            np.testing.assert_array_equal(np.loadtxt(index_path, dtype=int), expected_index)

    def test_1nn_profile_omits_index_output(self):
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory, "input.txt")
            profile_path = Path(directory, "profile.txt")
            index_path = Path(directory, "must-not-exist.txt")
            input_path.write_text("0\n1\n0\n2\n0\n1\n0\n", encoding="utf-8")
            args = self.parse(
                "--window=3",
                f"--input_a_file_name={input_path}",
                "--profile_type=1NN",
                f"--output_a_file_name={profile_path}",
                f"--output_a_index_file_name={index_path}",
            )
            self.assertEqual(0, cli.run(args))
            self.assertTrue(profile_path.exists())
            self.assertFalse(index_path.exists())

    def test_keep_rows_runs_reverse_abjoin_and_writes_b_outputs(self):
        args = self.parse(
            "--window=3",
            "--input_a_file_name=a.txt",
            "--input_b_file_name=b.txt",
            "--keep_rows",
        )
        a = np.arange(6, dtype=np.float64)
        b = np.arange(8, dtype=np.float64)
        with (
            mock.patch.object(cli, "_read_series", side_effect=[a, b]),
            mock.patch.object(cli, "abjoin", side_effect=[("a-profile", "a-index"), ("b-profile", "b-index")]) as join,
            mock.patch.object(cli, "_write_result") as write,
        ):
            self.assertEqual(0, cli.run(args))
        self.assertIs(join.call_args_list[0].args[0], a)
        self.assertIs(join.call_args_list[0].args[1], b)
        self.assertIs(join.call_args_list[1].args[0], b)
        self.assertIs(join.call_args_list[1].args[1], a)
        self.assertFalse(write.call_args_list[0].kwargs["row_profile"])
        self.assertTrue(write.call_args_list[1].kwargs["row_profile"])

    def test_all_profile_modes_dispatch_to_upstream_python_surface(self):
        a = np.arange(8, dtype=np.float64)
        cases = {
            "1NN_INDEX": "selfjoin",
            "1NN": "selfjoin",
            "SUM_THRESH": "selfjoin_sum",
            "ALL_NEIGHBORS": "selfjoin_knn",
            "MATRIX_SUMMARY": "selfjoin_matrix",
        }
        for profile_type, function_name in cases.items():
            with self.subTest(profile_type=profile_type):
                args = self.parse(
                    "--window=3",
                    "--input_a_file_name=input.txt",
                    f"--profile_type={profile_type}",
                    "--reduced_height=2",
                    "--reduced_width=2",
                )
                result = ([], []) if profile_type == "1NN_INDEX" else []
                with mock.patch.object(cli, function_name, return_value=result) as function:
                    cli._compute_one(args, a, None, cli._common_kwargs(args, None))
                function.assert_called_once()

    def test_output_is_deterministic_and_matches_upstream_text_shapes(self):
        stdout = io.StringIO()
        cli._write_lines("-", cli._match_lines([(1, 4, 0.5), (0, 3, 0.4), (0, 2, 0.8)], True), stdout)
        self.assertEqual("0 2 0.8\n0 3 0.4\n1 4 0.5\n", stdout.getvalue())

        stdout = io.StringIO()
        cli._write_lines("-", cli._matrix_lines(np.array([[0.1, 0.2], [0.3, np.nan]])), stdout)
        self.assertEqual("0.1 0.2\n0.3 nan\n", stdout.getvalue())

    def test_gflags_style_boolean_values_are_accepted(self):
        args = self.parse("--output_pearson=false", "--single_precision=true")
        self.assertFalse(args.output_pearson)
        self.assertTrue(args.single_precision)

    def test_advanced_engine_flags_are_forwarded(self):
        args = self.parse(
            "--single_precision",
            "--gpus=0",
            "--max_tile_size=4096",
            "--aligned",
            "--input_b_file_name=b.txt",
        )
        self.assertEqual(
            {
                "pearson": False,
                "precision": "single",
                "threads": 0,
                "verbose": False,
                "gpus": [0],
                "max_tile_size": 4096,
                "allow_trivial_match": False,
            },
            cli._common_kwargs(args, cli._parse_gpu_ids(args.gpus)),
        )

    def test_empty_gpu_list_retains_cli_auto_device_semantics(self):
        self.assertIsNone(cli._parse_gpu_ids(""))

    def test_main_returns_nonzero_status_for_validation_errors(self):
        stderr = io.StringIO()
        with mock.patch("sys.stderr", stderr), self.assertRaises(SystemExit) as error:
            cli.main(["--window=2", "--input_a_file_name=a.txt"])
        self.assertEqual(1, error.exception.code)
        self.assertIn("Subsequence length must be at least 3", stderr.getvalue())

    def test_invalid_jobs_fail_before_computation(self):
        cases = [
            self.parse("--window=2", "--input_a_file_name=a.txt"),
            self.parse("--window=3", "--input_a_file_name=-", "--input_b_file_name=-"),
            self.parse("--window=3", "--input_a_file_name=a.txt", "--keep_rows"),
            self.parse("--window=3", "--input_a_file_name=a.txt", "--gpus=1"),
            self.parse("--window=3", "--input_a_file_name=a.txt", "--max_tile_size=1000"),
        ]
        for args in cases:
            with self.subTest(args=args):
                with self.assertRaises(cli.CLIError):
                    cli.run(args)

    def test_matrix_dimensions_are_validated_after_reading_inputs(self):
        args = self.parse(
            "--window=3",
            "--input_a_file_name=input.txt",
            "--profile_type=MATRIX_SUMMARY",
            "--reduced_height=7",
            "--reduced_width=7",
        )
        with mock.patch.object(cli, "_read_series", return_value=np.arange(8, dtype=float)):
            with self.assertRaisesRegex(cli.CLIError, "distance matrix dimensions"):
                cli.run(args)


if __name__ == "__main__":
    unittest.main()
