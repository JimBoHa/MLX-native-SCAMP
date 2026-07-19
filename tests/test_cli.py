from __future__ import annotations

import io
import os
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from mlx_native_scamp import cli


class _BrokenStdout(io.StringIO):
    def write(self, value: str) -> int:
        raise OSError("simulated stdout failure")


class _ShortStdout(io.StringIO):
    def write(self, value: str) -> int:
        accepted = value[: max(1, len(value) // 2)]
        super().write(accepted)
        return len(accepted)


class CLIParserTests(unittest.TestCase):
    def parse(self, *arguments: str) -> object:
        return cli.build_parser().parse_args(arguments)

    def test_accepts_gflags_long_single_dash_and_boolean_spellings(self):
        args = self.parse(
            "-window=3",
            "-input_a_file_name",
            "-",
            "--output_pearson=true",
            "-single_precision",
            "false",
            "--nosingle_precision",
            "--print_debug_info",
        )
        self.assertEqual(3, args.window)
        self.assertEqual("-", args.input_a_file_name)
        self.assertTrue(args.output_pearson)
        self.assertFalse(args.single_precision)
        self.assertTrue(args.print_debug_info)

    def test_boolean_negation_for_no_gpu_is_nono_gpu(self):
        args = self.parse("--no_gpu", "--nono_gpu")
        self.assertFalse(args.no_gpu)

    def test_rejects_values_on_negated_boolean(self):
        with self.assertRaises(SystemExit):
            self.parse("--nooutput_pearson=true")

    def test_integer_parser_is_ascii_decimal_and_range_checked(self):
        invalid = (
            ("--window=3.0",),
            ("--window=0x10",),
            ("--window=1_024",),
            ("--window=٣",),
            (f"--window={2**31}",),
            (f"--global_row={2**63}",),
        )
        for arguments in invalid:
            with self.subTest(arguments=arguments), self.assertRaises(SystemExit):
                self.parse(*arguments)
        self.assertEqual(3, self.parse("--window=+3").window)

    def test_help_aliases_exit_successfully(self):
        for option in ("-h", "--help", "-helpshort", "--helpfull"):
            with self.subTest(option=option), self.assertRaises(SystemExit) as error:
                self.parse(option)
            self.assertEqual(0, error.exception.code)

    def test_list_variants_short_circuits_job_and_gpu_validation(self):
        stdout = io.StringIO()
        args = self.parse("--list_variants", "--gpus=not-an-integer")
        with mock.patch.object(
            cli, "_variant_descriptions", return_value=["v0 backend=cpu"]
        ):
            self.assertEqual(0, cli.run(args, stdout=stdout))
        self.assertEqual("v0 backend=cpu\n", stdout.getvalue())

    def test_autotune_short_circuits_input_validation_and_uses_public_api(self):
        args = self.parse("--autotune", "--gpus=0")
        with mock.patch.object(cli, "autotune", return_value=20) as tune:
            self.assertEqual(0, cli.run(args))
        tune.assert_called_once_with(devices=[0])

    def test_variant_listing_is_stable_and_one_to_one(self):
        descriptions = cli.strategy_descriptions()
        lines = cli._variant_descriptions()
        self.assertEqual(len(descriptions), len(lines))
        for index, (description, line) in enumerate(zip(descriptions, lines)):
            strategy = description.strategy
            expected = (
                f"v{index} backend={description.backend} "
                f"strategy={strategy.name} profile={strategy.profile} "
                f"route={strategy.route}"
            )
            self.assertTrue(line.startswith(expected), line)
            for key, value in strategy.parameters:
                self.assertIn(f" {key}={value}", line)

    def test_validation_rejects_unsupported_combinations_before_io(self):
        cases = (
            ("--window=2", "--input_a_file_name=a"),
            ("--window=3", "--input_a_file_name=-", "--input_b_file_name=-"),
            ("--window=3", "--input_a_file_name=a", "--keep_rows"),
            ("--window=3", "--input_a_file_name=a", "--aligned"),
            (
                "--window=3",
                "--input_a_file_name=a",
                "--input_b_file_name=b",
                "--profile_type=MATRIX_SUMMARY",
                "--keep_rows",
            ),
            (
                "--window=3",
                "--input_a_file_name=a",
                "--profile_type=SUM_THRESH",
                "--threshold=nan",
            ),
            ("--window=3", "--input_a_file_name=a", "--global_row=0"),
            ("--window=3", "--input_a_file_name=a", "--max_tile_size=1000"),
            (
                "--window=600",
                "--input_a_file_name=a",
                "--max_tile_size=1024",
            ),
            (
                "--window=3",
                "--input_a_file_name=a",
                "--output_a_file_name=",
            ),
        )
        for arguments in cases:
            with self.subTest(arguments=arguments):
                args = self.parse(*arguments)
                with (
                    mock.patch.object(cli, "_capture_plan") as capture,
                    self.assertRaises(cli.CLIError),
                ):
                    cli.run(args)
                capture.assert_not_called()

    def test_gpu_and_precision_capabilities_fail_before_io(self):
        cases = (
            ("--gpus=1", "--single_precision"),
            ("--gpus=0", "--double_precision"),
            ("--gpus=0", "--single_precision", "--no_gpu"),
            (
                "--gpus=0",
                "--single_precision",
                "--num_cpu_workers=1",
            ),
        )
        for extra in cases:
            with self.subTest(extra=extra):
                args = self.parse(
                    "--window=3", "--input_a_file_name=missing", *extra
                )
                with (
                    mock.patch.object(cli, "_capture_plan") as capture,
                    self.assertRaises(cli.CLIError),
                ):
                    cli.run(args)
                capture.assert_not_called()

        args = self.parse(
            "--window=3",
            "--input_a_file_name=missing",
            "--gpus=0",
            "--single_precision",
        )
        with (
            mock.patch.object(cli, "gpu_supported", return_value=False),
            mock.patch.object(cli, "_capture_plan") as capture,
            self.assertRaisesRegex(cli.CLIError, "unavailable"),
        ):
            cli.run(args)
        capture.assert_not_called()

    def test_profile_specific_unused_knobs_are_ignored(self):
        args = self.parse(
            "--window=3",
            "--input_a_file_name=missing",
            "--profile_type=1NN",
            "--threshold=2",
            "--max_matches_per_column=0",
            "--reduced_height=0",
            "--reduced_width=0",
        )
        cli._validate_job(args, None)

    def test_main_reports_clean_validation_error(self):
        stderr = io.StringIO()
        with mock.patch("sys.stderr", stderr), self.assertRaises(SystemExit) as error:
            cli.main(["--window=2", "--input_a_file_name=a"])
        self.assertEqual(1, error.exception.code)
        self.assertIn("Subsequence length must be at least 3", stderr.getvalue())


class CLIDispatchTests(unittest.TestCase):
    def parse(self, *arguments: str) -> object:
        return cli.build_parser().parse_args(
            ("--window=3", "--input_a_file_name=a", *arguments)
        )

    def test_self_profiles_use_each_native_surface(self):
        series = np.arange(8, dtype=np.float64)
        cases = (
            ("1NN_INDEX", "selfjoin", (np.array([]), np.array([]))),
            ("1NN", "selfjoin_1nn", np.array([])),
            ("SUM_THRESH", "selfjoin_sum", np.array([])),
            ("ALL_NEIGHBORS", "selfjoin_knn", []),
            ("MATRIX_SUMMARY", "selfjoin_matrix", np.empty((2, 2))),
        )
        for profile, function_name, result in cases:
            with self.subTest(profile=profile):
                extra = [f"--profile_type={profile}"]
                if profile == "MATRIX_SUMMARY":
                    extra.extend(("--reduced_height=2", "--reduced_width=2"))
                args = self.parse(*extra)
                with mock.patch.object(
                    cli, function_name, return_value=result
                ) as function:
                    forward, reverse = cli._compute(
                        args, series, None, cli._common_kwargs(args, None)
                    )
                function.assert_called_once()
                self.assertIs(forward, result)
                self.assertIsNone(reverse)

    def test_ab_profiles_use_forward_native_surfaces(self):
        a = np.arange(8, dtype=np.float64)
        b = np.arange(9, dtype=np.float64)
        cases = (
            ("1NN_INDEX", "abjoin", (np.array([]), np.array([]))),
            ("1NN", "abjoin_1nn", np.array([])),
            ("SUM_THRESH", "abjoin_sum", np.array([])),
            ("ALL_NEIGHBORS", "abjoin_knn", []),
            ("MATRIX_SUMMARY", "abjoin_matrix", np.empty((2, 2))),
        )
        for profile, function_name, result in cases:
            with self.subTest(profile=profile):
                extra = [
                    f"--profile_type={profile}",
                    "--input_b_file_name=b",
                ]
                if profile == "MATRIX_SUMMARY":
                    extra.extend(("--reduced_height=2", "--reduced_width=2"))
                args = self.parse(*extra)
                with mock.patch.object(
                    cli, function_name, return_value=result
                ) as function:
                    forward, reverse = cli._compute(
                        args, a, b, cli._common_kwargs(args, None)
                    )
                function.assert_called_once()
                self.assertIs(forward, result)
                self.assertIsNone(reverse)

    def test_indexed_keep_rows_uses_one_bidirectional_join(self):
        a = np.arange(8, dtype=np.float64)
        b = np.arange(9, dtype=np.float64)
        expected = (("av", "ai"), ("bv", "bi"))
        args = self.parse("--input_b_file_name=b", "--keep_rows")
        with mock.patch.object(
            cli, "abjoin_bidirectional", return_value=expected
        ) as join:
            self.assertEqual(
                expected,
                cli._compute(args, a, b, cli._common_kwargs(args, None)),
            )
        join.assert_called_once()

    def test_index_free_keep_rows_uses_one_join_and_discards_indexes(self):
        a = np.arange(8, dtype=np.float64)
        b = np.arange(9, dtype=np.float64)
        expected = (("av", "ai"), ("bv", "bi"))
        args = self.parse(
            "--profile_type=1NN", "--input_b_file_name=b", "--keep_rows"
        )
        with mock.patch.object(
            cli, "abjoin_bidirectional", return_value=expected
        ) as join:
            self.assertEqual(
                ("av", "bv"),
                cli._compute(args, a, b, cli._common_kwargs(args, None)),
            )
        join.assert_called_once()

    def test_sum_and_knn_keep_rows_run_reverse_native_profile(self):
        a = np.arange(8, dtype=np.float64)
        b = np.arange(9, dtype=np.float64)
        for profile, function_name in (
            ("SUM_THRESH", "abjoin_sum"),
            ("ALL_NEIGHBORS", "abjoin_knn"),
        ):
            with self.subTest(profile=profile):
                args = self.parse(
                    f"--profile_type={profile}",
                    "--input_b_file_name=b",
                    "--keep_rows",
                )
                with mock.patch.object(
                    cli, function_name, side_effect=("forward", "reverse")
                ) as join:
                    result = cli._compute(
                        args, a, b, cli._common_kwargs(args, None)
                    )
                self.assertEqual(("forward", "reverse"), result)
                self.assertIs(join.call_args_list[0].args[0], a)
                self.assertIs(join.call_args_list[0].args[1], b)
                self.assertIs(join.call_args_list[1].args[0], b)
                self.assertIs(join.call_args_list[1].args[1], a)

    def test_common_resource_flags_are_forwarded_exactly(self):
        args = self.parse(
            "--input_b_file_name=b",
            "--single_precision",
            "--gpus=0",
            "--max_tile_size=4096",
            "--aligned",
            "--print_debug_info",
        )
        self.assertEqual(
            {
                "pearson": False,
                "precision": "single",
                "threads": 0,
                "verbose": True,
                "gpus": [0],
                "max_tile_size": 4096,
                "allow_trivial_match": False,
            },
            cli._common_kwargs(args, cli._parse_gpu_ids(args.gpus)),
        )


class CLIOutputTests(unittest.TestCase):
    def test_text_shapes_and_deterministic_knn_order(self):
        self.assertEqual(
            ["0.1 0.2", "0.3 nan"],
            list(cli._matrix_lines(np.array([[0.1, 0.2], [0.3, np.nan]]))),
        )
        pearson_matches = [(0, 2, 0.8), (0, 3, 0.4), (1, 4, 0.5)]
        self.assertEqual(
            ["0 2 0.8", "0 3 0.4", "1 4 0.5"],
            list(cli._match_lines(pearson_matches, True)),
        )
        distance_matches = [(0, 3, 0.4), (0, 2, 0.8), (1, 4, 0.5)]
        self.assertEqual(
            ["0 3 0.4", "0 2 0.8", "1 4 0.5"],
            list(cli._match_lines(distance_matches, False)),
        )

    def test_debug_route_is_stderr_and_stdout_is_pure_data(self):
        args = cli.build_parser().parse_args(
            (
                "--window=3",
                "--input_a_file_name=-",
                "--output_a_file_name=-",
                "--profile_type=1NN",
                "--print_debug_info",
                "--no_gpu",
            )
        )
        stdout = io.StringIO()
        stderr = io.StringIO()

        def compute(*unused_args: object, **unused_kwargs: object) -> tuple[object, None]:
            print("mlx-scamp op=1 selfjoin/1nn_value start: backend=cpu")
            print("mlx-scamp op=1 selfjoin/1nn_value complete: backend=cpu")
            return np.array([0.25, np.nan]), None

        with mock.patch.object(cli, "_compute", side_effect=compute):
            self.assertEqual(
                0,
                cli.run(
                    args,
                    stdin=io.StringIO("0 1 2 3"),
                    stdout=stdout,
                    stderr=stderr,
                ),
            )
        self.assertEqual("0.25\nnan\n", stdout.getvalue())
        self.assertIn("backend=cpu", stderr.getvalue())
        self.assertNotIn("backend=cpu", stdout.getvalue())

    def test_single_precision_input_is_parsed_directly_as_float32(self):
        args = cli.build_parser().parse_args(
            (
                "--window=3",
                "--input_a_file_name=-",
                "--output_a_file_name=-",
                "--profile_type=1NN",
                "--single_precision",
                "--no_gpu",
            )
        )
        with mock.patch.object(
            cli, "_compute", return_value=(np.array([]), None)
        ) as compute:
            cli.run(args, stdin=io.StringIO("0 1 2 3"), stdout=io.StringIO())
        self.assertEqual(np.dtype(np.float32), compute.call_args.args[1].dtype)

    def test_real_cpu_cli_job_runs_from_parser_through_serialization(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "input"
            profile_path = root / "profile"
            index_path = root / "index"
            series = np.array([0.0, 1.0, 0.0, 2.0, 0.0, 1.0, 0.0])
            np.savetxt(input_path, series)
            args = cli.build_parser().parse_args(
                (
                    "--window=3",
                    f"--input_a_file_name={input_path}",
                    f"--output_a_file_name={profile_path}",
                    f"--output_a_index_file_name={index_path}",
                    "--output_pearson",
                    "--no_gpu",
                )
            )
            self.assertEqual(0, cli.run(args))
            expected_profile, expected_index = cli.selfjoin(
                series, 3, pearson=True, gpus=[]
            )
            np.testing.assert_allclose(
                np.loadtxt(profile_path),
                expected_profile,
                equal_nan=True,
                rtol=1e-7,
                atol=1e-7,
            )
            np.testing.assert_array_equal(
                np.loadtxt(index_path, dtype=np.int64), expected_index
            )


class CLIPathSafetyTests(unittest.TestCase):
    def parse_job(
        self,
        directory: str,
        *arguments: str,
    ) -> tuple[object, Path, Path, Path]:
        root = Path(directory)
        input_path = root / "input"
        profile = root / "profile"
        index = root / "index"
        if not input_path.exists():
            input_path.write_text("0 1 2 3 4\n", encoding="utf-8")
        args = cli.build_parser().parse_args(
            (
                "--window=3",
                f"--input_a_file_name={input_path}",
                f"--output_a_file_name={profile}",
                f"--output_a_index_file_name={index}",
                "--no_gpu",
                *arguments,
            )
        )
        return args, input_path, profile, index

    def test_relative_case_nfkc_and_hardlink_output_aliases_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            args, _, profile, _ = self.parse_job(directory)
            cases = (
                str(profile.parent / "." / profile.name),
                str(profile.parent / "PROFILE"),
                str(profile.parent / "ｐｒｏｆｉｌｅ"),
            )
            for alias in cases:
                with self.subTest(alias=alias):
                    args.output_a_index_file_name = alias
                    with self.assertRaisesRegex(cli.CLIError, "distinct"):
                        cli._capture_plan(args)

            profile.write_text("old", encoding="utf-8")
            hardlink = Path(directory, "hardlink")
            os.link(profile, hardlink)
            args.output_a_index_file_name = str(hardlink)
            with self.assertRaisesRegex(cli.CLIError, "distinct"):
                cli._capture_plan(args)

    def test_input_output_aliases_include_hardlinks_and_nfkc(self):
        with tempfile.TemporaryDirectory() as directory:
            args, input_path, _, index = self.parse_job(directory)
            args.output_a_file_name = str(input_path)
            with self.assertRaisesRegex(cli.CLIError, "aliases an input"):
                cli._capture_plan(args)

            hardlink = Path(directory, "input-hardlink")
            os.link(input_path, hardlink)
            args.output_a_file_name = str(hardlink)
            args.output_a_index_file_name = str(index)
            with self.assertRaisesRegex(cli.CLIError, "aliases an input"):
                cli._capture_plan(args)

    def test_output_symlinks_dangling_symlinks_and_nonregulars_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            args, _, profile, _ = self.parse_job(directory)
            unrelated = Path(directory, "unrelated")
            unrelated.write_text("safe", encoding="utf-8")
            profile.symlink_to(unrelated)
            with self.assertRaisesRegex(cli.CLIError, "symbolic"):
                cli._capture_plan(args)
            profile.unlink()
            profile.symlink_to(Path(directory, "missing"))
            with self.assertRaisesRegex(cli.CLIError, "symbolic"):
                cli._capture_plan(args)
            profile.unlink()
            os.mkfifo(profile)
            with self.assertRaisesRegex(cli.CLIError, "not a regular"):
                cli._capture_plan(args)

    def test_missing_output_parent_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            args, _, _, _ = self.parse_job(directory)
            args.output_a_file_name = str(Path(directory, "missing", "profile"))
            with self.assertRaisesRegex(cli.CLIError, "parent"):
                cli._capture_plan(args)

    def test_input_symlink_to_regular_file_is_allowed(self):
        with tempfile.TemporaryDirectory() as directory:
            args, input_path, profile, index = self.parse_job(directory)
            alias = Path(directory, "input-link")
            alias.symlink_to(input_path)
            args.input_a_file_name = str(alias)
            with mock.patch.object(
                cli,
                "_compute",
                return_value=((np.array([0.5]), np.array([1])), None),
            ):
                self.assertEqual(0, cli.run(args))
            self.assertEqual("0.5\n", profile.read_text(encoding="utf-8"))
            self.assertEqual("1\n", index.read_text(encoding="utf-8"))


class CLITransactionTests(unittest.TestCase):
    def make_job(
        self,
        directory: str,
        *,
        index_stdout: bool = False,
    ) -> tuple[object, Path, Path, Path]:
        root = Path(directory)
        input_path = root / "input"
        profile = root / "profile"
        index = root / "index"
        input_path.write_text("0 1 2 3 4\n", encoding="utf-8")
        args = cli.build_parser().parse_args(
            (
                "--window=3",
                f"--input_a_file_name={input_path}",
                f"--output_a_file_name={profile}",
                "--output_a_index_file_name=-"
                if index_stdout
                else f"--output_a_index_file_name={index}",
                "--no_gpu",
            )
        )
        return args, input_path, profile, index

    @staticmethod
    def fake_result() -> tuple[tuple[np.ndarray, np.ndarray], None]:
        return (np.array([0.25, 0.5]), np.array([2, 3])), None

    @staticmethod
    def hidden_files(directory: str) -> list[Path]:
        return [path for path in Path(directory).iterdir() if path.name.startswith(".")]

    def test_all_outputs_are_staged_before_any_target_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            args, _, profile, index = self.make_job(directory)
            profile.write_text("old-profile\n", encoding="utf-8")
            index.write_text("old-index\n", encoding="utf-8")

            def fail(point: str, position: int) -> None:
                if point == "stage_write" and position == 1:
                    raise OSError("simulated second-stage failure")

            with (
                mock.patch.object(cli, "_compute", return_value=self.fake_result()),
                mock.patch.object(cli, "_FAULT_HOOK", fail),
                self.assertRaisesRegex(cli.CLIError, "second-stage failure"),
            ):
                cli.run(args)
            self.assertEqual("old-profile\n", profile.read_text(encoding="utf-8"))
            self.assertEqual("old-index\n", index.read_text(encoding="utf-8"))
            self.assertEqual([], self.hidden_files(directory))

    def test_install_failure_rolls_back_old_and_new_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            args, _, profile, index = self.make_job(directory)
            profile.write_text("old-profile\n", encoding="utf-8")

            def fail(point: str, position: int) -> None:
                if point == "after_install" and position == 1:
                    raise OSError("simulated install failure")

            with (
                mock.patch.object(cli, "_compute", return_value=self.fake_result()),
                mock.patch.object(cli, "_FAULT_HOOK", fail),
                self.assertRaisesRegex(cli.CLIError, "install failure"),
            ):
                cli.run(args)
            self.assertEqual("old-profile\n", profile.read_text(encoding="utf-8"))
            self.assertFalse(index.exists())
            self.assertEqual([], self.hidden_files(directory))

    def test_stdout_failure_rolls_back_file_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            args, _, profile, _ = self.make_job(directory, index_stdout=True)
            profile.write_text("old-profile\n", encoding="utf-8")
            with (
                mock.patch.object(cli, "_compute", return_value=self.fake_result()),
                self.assertRaisesRegex(cli.CLIError, "stdout failure"),
            ):
                cli.run(args, stdout=_BrokenStdout())
            self.assertEqual("old-profile\n", profile.read_text(encoding="utf-8"))
            self.assertEqual([], self.hidden_files(directory))

    def test_short_stdout_writes_are_retried_without_truncation(self):
        with tempfile.TemporaryDirectory() as directory:
            args, _, _, _ = self.make_job(directory, index_stdout=True)
            stdout = _ShortStdout()
            with mock.patch.object(
                cli, "_compute", return_value=self.fake_result()
            ):
                cli.run(args, stdout=stdout)
            self.assertEqual("2\n3\n", stdout.getvalue())

    def test_interrupt_after_backup_rename_recovers_original_by_inode(self):
        with tempfile.TemporaryDirectory() as directory:
            args, _, profile, index = self.make_job(directory)
            profile.write_text("old-profile\n", encoding="utf-8")
            index.write_text("old-index\n", encoding="utf-8")
            real_rename = cli._rename_exclusive
            interrupted = False

            def rename(source: Path, destination: Path) -> None:
                nonlocal interrupted
                real_rename(source, destination)
                if not interrupted and str(destination).endswith(".backup"):
                    interrupted = True
                    raise KeyboardInterrupt

            with (
                mock.patch.object(cli, "_compute", return_value=self.fake_result()),
                mock.patch.object(cli, "_rename_exclusive", side_effect=rename),
                self.assertRaises(KeyboardInterrupt),
            ):
                cli.run(args)
            self.assertEqual("old-profile\n", profile.read_text(encoding="utf-8"))
            self.assertEqual("old-index\n", index.read_text(encoding="utf-8"))
            self.assertEqual([], self.hidden_files(directory))

    def test_target_swap_at_backup_checkpoint_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            args, _, profile, index = self.make_job(directory)
            profile.write_text("old-profile\n", encoding="utf-8")
            index.write_text("old-index\n", encoding="utf-8")
            saved = Path(directory, "saved-original")

            def swap(point: str, position: int) -> None:
                if point == "backup_reserved" and position == 0:
                    os.replace(profile, saved)
                    profile.write_text("attacker\n", encoding="utf-8")

            with (
                mock.patch.object(cli, "_compute", return_value=self.fake_result()),
                mock.patch.object(cli, "_FAULT_HOOK", swap),
                self.assertRaisesRegex(cli.CLIError, "changed during execution"),
            ):
                cli.run(args)
            self.assertEqual("attacker\n", profile.read_text(encoding="utf-8"))
            self.assertEqual("old-profile\n", saved.read_text(encoding="utf-8"))
            self.assertEqual("old-index\n", index.read_text(encoding="utf-8"))

    def test_replaced_staged_file_is_neither_installed_nor_deleted(self):
        with tempfile.TemporaryDirectory() as directory:
            args, _, profile, index = self.make_job(directory)
            profile.write_text("old-profile\n", encoding="utf-8")
            index.write_text("old-index\n", encoding="utf-8")
            attacker = Path(directory, "attacker")
            attacker.write_text("attacker-payload\n", encoding="utf-8")

            def swap(point: str, position: int) -> None:
                if point == "before_install" and position == 0:
                    temporary = next(Path(directory).glob(".profile.*.tmp"))
                    os.replace(attacker, temporary)

            with (
                mock.patch.object(cli, "_compute", return_value=self.fake_result()),
                mock.patch.object(cli, "_FAULT_HOOK", swap),
                self.assertRaisesRegex(cli.CLIError, "staged output changed"),
            ):
                cli.run(args)
            self.assertEqual("old-profile\n", profile.read_text(encoding="utf-8"))
            self.assertEqual("old-index\n", index.read_text(encoding="utf-8"))
            preserved = list(Path(directory).glob(".profile.*.tmp"))
            self.assertEqual(1, len(preserved))
            self.assertEqual(
                "attacker-payload\n", preserved[0].read_text(encoding="utf-8")
            )

    def test_replaced_mode_probe_is_not_deleted(self):
        with tempfile.TemporaryDirectory() as directory:
            args, _, profile, index = self.make_job(directory)
            victim = Path(directory, "victim")
            victim.write_text("do-not-delete\n", encoding="utf-8")

            def swap(point: str, position: int) -> None:
                if point == "mode_probe" and position == 0:
                    probe = next(Path(directory).glob(".profile.*.mode-probe"))
                    os.replace(victim, probe)

            with (
                mock.patch.object(cli, "_compute", return_value=self.fake_result()),
                mock.patch.object(cli, "_FAULT_HOOK", swap),
                self.assertRaisesRegex(cli.CLIError, "refusing to remove"),
            ):
                cli.run(args)
            preserved = list(Path(directory).glob(".profile.*.mode-probe"))
            self.assertEqual(1, len(preserved))
            self.assertEqual(
                "do-not-delete\n", preserved[0].read_text(encoding="utf-8")
            )
            self.assertFalse(profile.exists())
            self.assertFalse(index.exists())

    def test_replaced_backup_is_never_restored_over_target(self):
        with tempfile.TemporaryDirectory() as directory:
            args, _, profile, index = self.make_job(directory)
            profile.write_text("old-profile\n", encoding="utf-8")
            index.write_text("old-index\n", encoding="utf-8")
            saved_original = Path(directory, "saved-original")

            def swap(point: str, position: int) -> None:
                if point == "after_backup" and position == 0:
                    backup = next(Path(directory).glob(".profile.*.backup"))
                    os.replace(backup, saved_original)
                    backup.write_text("attacker\n", encoding="utf-8")
                if point == "before_install" and position == 0:
                    raise OSError("force rollback")

            with (
                mock.patch.object(cli, "_compute", return_value=self.fake_result()),
                mock.patch.object(cli, "_FAULT_HOOK", swap),
                self.assertRaisesRegex(cli.CLIError, "backup was replaced"),
            ):
                cli.run(args)
            self.assertFalse(profile.exists())
            self.assertEqual(
                "old-profile\n", saved_original.read_text(encoding="utf-8")
            )
            attacker_backups = list(Path(directory).glob(".profile.*.backup"))
            self.assertEqual(1, len(attacker_backups))
            self.assertEqual(
                "attacker\n", attacker_backups[0].read_text(encoding="utf-8")
            )
            self.assertEqual("old-index\n", index.read_text(encoding="utf-8"))

    def test_rollback_never_replaces_a_new_destination(self):
        with tempfile.TemporaryDirectory() as directory:
            args, _, profile, index = self.make_job(directory)
            profile.write_text("old-profile\n", encoding="utf-8")
            index.write_text("old-index\n", encoding="utf-8")
            victim = Path(directory, "victim")
            victim.write_text("do-not-delete\n", encoding="utf-8")
            canonical_profile = profile.resolve(strict=False)
            real_rename = cli._rename_exclusive

            def rename(source: Path, destination: Path) -> None:
                if (
                    source.name.endswith(".backup")
                    and destination == canonical_profile
                ):
                    os.replace(victim, profile)
                real_rename(source, destination)

            def fail(point: str, position: int) -> None:
                if point == "after_install" and position == 0:
                    raise OSError("force rollback")

            with (
                mock.patch.object(cli, "_compute", return_value=self.fake_result()),
                mock.patch.object(cli, "_FAULT_HOOK", fail),
                mock.patch.object(cli, "_rename_exclusive", side_effect=rename),
                self.assertRaisesRegex(cli.CLIError, "File exists"),
            ):
                cli.run(args)
            self.assertEqual(
                "do-not-delete\n", profile.read_text(encoding="utf-8")
            )
            backups = list(Path(directory).glob(".profile.*.backup"))
            self.assertEqual(1, len(backups))
            self.assertEqual(
                "old-profile\n", backups[0].read_text(encoding="utf-8")
            )
            self.assertEqual("old-index\n", index.read_text(encoding="utf-8"))

    def test_new_target_swapped_during_rollback_is_not_deleted(self):
        with tempfile.TemporaryDirectory() as directory:
            args, _, profile, _ = self.make_job(directory)
            victim = Path(directory, "victim")
            victim.write_text("do-not-delete\n", encoding="utf-8")
            saved_stage = Path(directory, "saved-stage")
            canonical_profile = profile.resolve(strict=False)
            real_current_file_id = cli._current_file_id
            profile_reads = 0

            def current_file_id(path: Path) -> object:
                nonlocal profile_reads
                result = real_current_file_id(path)
                if path == canonical_profile and result is not None:
                    profile_reads += 1
                    if profile_reads == 2:
                        os.replace(profile, saved_stage)
                        os.replace(victim, profile)
                return result

            def fail(point: str, position: int) -> None:
                if point == "after_install" and position == 0:
                    raise OSError("force rollback")

            with (
                mock.patch.object(cli, "_compute", return_value=self.fake_result()),
                mock.patch.object(cli, "_FAULT_HOOK", fail),
                mock.patch.object(
                    cli, "_current_file_id", side_effect=current_file_id
                ),
                self.assertRaisesRegex(cli.CLIError, "installed output"),
            ):
                cli.run(args)
            self.assertEqual(
                "do-not-delete\n", profile.read_text(encoding="utf-8")
            )
            self.assertEqual("0.25\n0.5\n", saved_stage.read_text(encoding="utf-8"))

    def test_interrupt_after_staging_removes_owned_temps(self):
        with tempfile.TemporaryDirectory() as directory:
            args, _, profile, index = self.make_job(directory)

            def fail(point: str, position: int) -> None:
                if point == "after_staging":
                    raise KeyboardInterrupt

            with (
                mock.patch.object(cli, "_compute", return_value=self.fake_result()),
                mock.patch.object(cli, "_FAULT_HOOK", fail),
                self.assertRaises(KeyboardInterrupt),
            ):
                cli.run(args)
            self.assertFalse(profile.exists())
            self.assertFalse(index.exists())
            self.assertEqual([], self.hidden_files(directory))

    def test_existing_target_mode_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            args, _, profile, index = self.make_job(directory)
            profile.write_text("old\n", encoding="utf-8")
            index.write_text("old\n", encoding="utf-8")
            profile.chmod(0o640)
            index.chmod(0o400)
            with mock.patch.object(
                cli, "_compute", return_value=self.fake_result()
            ):
                cli.run(args)
            self.assertEqual(0o640, stat.S_IMODE(profile.stat().st_mode))
            self.assertEqual(0o400, stat.S_IMODE(index.stat().st_mode))

    @unittest.skipUnless(sys.platform == "darwin", "macOS metadata API")
    def test_existing_target_extended_attributes_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            args, _, profile, index = self.make_job(directory)
            profile.write_text("old\n", encoding="utf-8")
            index.write_text("old\n", encoding="utf-8")
            subprocess.run(
                ["xattr", "-w", "test.cli.marker", "profile-value", profile],
                check=True,
            )
            subprocess.run(
                ["xattr", "-w", "test.cli.marker", "index-value", index],
                check=True,
            )
            with mock.patch.object(
                cli, "_compute", return_value=self.fake_result()
            ):
                cli.run(args)
            self.assertEqual(
                "profile-value",
                subprocess.check_output(
                    ["xattr", "-p", "test.cli.marker", profile], text=True
                ).strip(),
            )
            self.assertEqual(
                "index-value",
                subprocess.check_output(
                    ["xattr", "-p", "test.cli.marker", index], text=True
                ).strip(),
            )

    def test_new_target_mode_honors_umask_without_mutating_it(self):
        script = """
import os
import pathlib
from unittest import mock
import numpy as np
from mlx_native_scamp import cli
root = pathlib.Path(os.environ['CLI_MODE_ROOT'])
(root / 'input').write_text('0 1 2 3 4\\n', encoding='utf-8')
args = cli.build_parser().parse_args([
    '--window=3', f'--input_a_file_name={root / "input"}',
    f'--output_a_file_name={root / "profile"}',
    f'--output_a_index_file_name={root / "index"}', '--no_gpu'])
before = os.umask(int(os.environ['CLI_TEST_UMASK'], 8))
os.umask(before)
os.umask(int(os.environ['CLI_TEST_UMASK'], 8))
with mock.patch.object(cli, '_compute', return_value=((np.array([0.5]), np.array([1])), None)):
    cli.run(args)
current = os.umask(before)
os.umask(current)
print(oct((root / 'profile').stat().st_mode & 0o777), oct(current))
"""
        for mask, expected in (("022", "0o644 0o22"), ("077", "0o600 0o77")):
            with self.subTest(mask=mask), tempfile.TemporaryDirectory() as directory:
                environment = os.environ.copy()
                environment["CLI_MODE_ROOT"] = directory
                environment["CLI_TEST_UMASK"] = mask
                environment["PYTHONPATH"] = os.pathsep.join(
                    filter(
                        None,
                        (
                            str(Path(__file__).parents[1] / "src"),
                            environment.get("PYTHONPATH"),
                        ),
                    )
                )
                result = subprocess.run(
                    [sys.executable, "-c", script],
                    check=True,
                    capture_output=True,
                    text=True,
                    env=environment,
                )
                self.assertEqual(expected, result.stdout.strip())

    def test_existing_output_mutated_after_compute_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            args, _, profile, index = self.make_job(directory)
            profile.write_text("old-profile\n", encoding="utf-8")
            index.write_text("old-index\n", encoding="utf-8")

            def mutate(point: str, position: int) -> None:
                if point == "after_compute":
                    profile.write_text("attacker\n", encoding="utf-8")

            with (
                mock.patch.object(cli, "_compute", return_value=self.fake_result()),
                mock.patch.object(cli, "_FAULT_HOOK", mutate),
                self.assertRaisesRegex(cli.CLIError, "changed during execution"),
            ):
                cli.run(args)
            self.assertEqual("attacker\n", profile.read_text(encoding="utf-8"))
            self.assertEqual("old-index\n", index.read_text(encoding="utf-8"))

    def test_absent_output_created_after_staging_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            args, _, profile, _ = self.make_job(directory)

            def mutate(point: str, position: int) -> None:
                if point == "after_staging" and not profile.exists():
                    profile.write_text("attacker\n", encoding="utf-8")

            with (
                mock.patch.object(cli, "_compute", return_value=self.fake_result()),
                mock.patch.object(cli, "_FAULT_HOOK", mutate),
                self.assertRaisesRegex(cli.CLIError, "changed during execution"),
            ):
                cli.run(args)
            self.assertEqual("attacker\n", profile.read_text(encoding="utf-8"))
            self.assertEqual([], self.hidden_files(directory))

    def test_input_mutated_during_compute_aborts_before_output_commit(self):
        with tempfile.TemporaryDirectory() as directory:
            args, input_path, profile, index = self.make_job(directory)
            profile.write_text("old-profile\n", encoding="utf-8")
            index.write_text("old-index\n", encoding="utf-8")

            def mutate(point: str, position: int) -> None:
                if point == "after_compute":
                    input_path.write_text("9 9 9 9 9\n", encoding="utf-8")

            with (
                mock.patch.object(cli, "_compute", return_value=self.fake_result()),
                mock.patch.object(cli, "_FAULT_HOOK", mutate),
                self.assertRaisesRegex(cli.CLIError, "input changed"),
            ):
                cli.run(args)
            self.assertEqual("old-profile\n", profile.read_text(encoding="utf-8"))
            self.assertEqual("old-index\n", index.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
