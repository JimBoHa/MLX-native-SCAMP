from __future__ import annotations

import argparse
import importlib
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, TextIO

import numpy as np

from . import (
    abjoin,
    abjoin_knn,
    abjoin_matrix,
    abjoin_sum,
    gpu_supported,
    selfjoin,
    selfjoin_knn,
    selfjoin_matrix,
    selfjoin_sum,
)


PROFILE_TYPES = ("1NN_INDEX", "1NN", "SUM_THRESH", "ALL_NEIGHBORS", "MATRIX_SUMMARY")
OUTPUT_BATCH_LINES = 4096


class CLIError(Exception):
    """An expected command-line input or capability error."""


def _parse_bool(value: str) -> bool:
    normalized = value.casefold()
    if normalized in {"1", "true", "t", "yes", "y"}:
        return True
    if normalized in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean value, got {value!r}")


def _boolean_flag(parser: argparse.ArgumentParser, name: str, help_text: str) -> None:
    parser.add_argument(
        f"--{name}",
        nargs="?",
        const=True,
        default=False,
        type=_parse_bool,
        metavar="BOOL",
        help=help_text,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scamp",
        allow_abbrev=False,
        description="Compute SCAMP matrix profiles with Apple MLX.",
    )
    parser.add_argument("--window", type=int, default=-1, help="subsequence length (at least 3)")
    parser.add_argument("--input_a_file_name", default="", help="primary input file, or '-' for stdin")
    parser.add_argument("--input_b_file_name", default="", help="secondary AB-join input file, or '-' for stdin")
    parser.add_argument(
        "--output_a_file_name",
        default="mp_columns_out",
        help="primary profile output, or '-' for stdout",
    )
    parser.add_argument(
        "--output_a_index_file_name",
        default="mp_columns_out_index",
        help="primary index output, or '-' for stdout",
    )
    parser.add_argument(
        "--output_b_file_name",
        default="mp_rows_out",
        help="row-wise profile output, or '-' for stdout",
    )
    parser.add_argument(
        "--output_b_index_file_name",
        default="mp_rows_out_index",
        help="row-wise index output, or '-' for stdout",
    )
    parser.add_argument("--profile_type", default="1NN_INDEX", choices=PROFILE_TYPES)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--max_matches_per_column", type=int, default=5)
    parser.add_argument("--reduced_height", type=int, default=50)
    parser.add_argument("--reduced_width", type=int, default=50)
    parser.add_argument("--num_cpu_workers", type=int, default=0)
    parser.add_argument("--max_tile_size", type=int, default=None)
    parser.add_argument("--gpus", default=None, help="MLX Metal device IDs; Apple Silicon exposes only device 0")
    parser.add_argument("--global_row", type=int, default=-1)
    parser.add_argument("--global_col", type=int, default=-1)

    _boolean_flag(parser, "output_pearson", "write Pearson correlation instead of Euclidean distance")
    _boolean_flag(parser, "print_debug_info", "write progress information to stderr")
    _boolean_flag(parser, "no_gpu", "run on the MLX CPU backend")
    _boolean_flag(parser, "ultra_precision", "use the ultra precision path")
    _boolean_flag(parser, "double_precision", "use the double precision path")
    _boolean_flag(parser, "single_precision", "use the single precision path")
    _boolean_flag(parser, "keep_rows", "also compute the row-wise AB-join profile")
    _boolean_flag(parser, "aligned", "apply an exclusion zone to an aligned AB-join")
    _boolean_flag(parser, "autotune", "tune and cache MLX execution strategies, then exit")
    _boolean_flag(parser, "list_variants", "list available MLX execution strategies, then exit")
    return parser


def _parse_gpu_ids(value: str | None) -> list[int] | None:
    if value is None:
        return None
    tokens = value.replace(",", " ").split()
    if not tokens:
        # This matches the upstream CLI: an empty --gpus string means
        # automatic device selection. CPU-only execution is expressed by
        # --no_gpu, which maps to the pyscamp gpus=[] convention below.
        return None
    try:
        devices = [int(token) for token in tokens]
    except ValueError as exc:
        raise CLIError("--gpus must contain comma- or space-separated integer device IDs") from exc
    if len(devices) != len(set(devices)):
        raise CLIError("--gpus contains a duplicate device ID")
    unsupported = [device for device in devices if device != 0]
    if unsupported:
        raise CLIError(
            f"MLX/Metal exposes only GPU device 0; unsupported device ID {unsupported[0]}"
        )
    return devices


def _precision(args: argparse.Namespace) -> str:
    selected = [
        name
        for name, enabled in (
            ("ultra", args.ultra_precision),
            ("double", args.double_precision),
            ("single", args.single_precision),
        )
        if enabled
    ]
    if len(selected) > 1:
        raise CLIError("only one precision flag can be enabled at a time")
    return selected[0] if selected else "double"


def _validate_job(args: argparse.Namespace, devices: list[int] | None) -> None:
    if args.window < 3:
        raise CLIError(
            "Subsequence length must be at least 3; use --window=<window_size>"
        )
    if not args.input_a_file_name:
        raise CLIError("primary input filename must be specified using --input_a_file_name")
    if args.input_a_file_name == "-" and args.input_b_file_name == "-":
        raise CLIError("only one input can read from stdin")
    if not -1.0 <= args.threshold <= 1.0:
        raise CLIError("threshold must be between -1 and 1")
    if args.max_matches_per_column <= 0:
        raise CLIError("max_matches_per_column must be greater than 0")
    if args.reduced_height <= 0 or args.reduced_width <= 0:
        raise CLIError("reduced matrix dimensions must be greater than 0")
    if args.num_cpu_workers < 0:
        raise CLIError("num_cpu_workers must be greater than or equal to 0")
    if args.no_gpu and devices not in (None, []):
        raise CLIError("--no_gpu cannot be combined with a non-empty --gpus list")
    if devices and args.num_cpu_workers > 0:
        raise CLIError("concurrent CPU and Metal execution is not supported by MLX")
    if args.aligned and not args.input_b_file_name:
        raise CLIError("--aligned is only valid for AB-joins")
    if args.keep_rows and not args.input_b_file_name:
        raise CLIError(
            "self-join --keep_rows requires left/right profile support that is not yet available"
        )
    if args.keep_rows and args.profile_type == "MATRIX_SUMMARY":
        raise CLIError("--keep_rows is not defined for MATRIX_SUMMARY profiles")
    if args.global_row != -1 or args.global_col != -1:
        raise CLIError(
            "--global_row and --global_col require the distributed partition API"
        )

    tile_size = args.max_tile_size
    if tile_size is not None:
        if tile_size < 1024:
            raise CLIError("max tile size must be at least 1024")
        if tile_size // 2 < args.window:
            raise CLIError("max tile size must be at least twice the window size")


def _read_series(name: str, stdin: TextIO) -> np.ndarray:
    if name == "-":
        stream = stdin
        label = "stdin"
        close = False
    else:
        label = name
        try:
            stream = Path(name).open("r", encoding="utf-8")
        except OSError as exc:
            raise CLIError(f"unable to open {name!r} for reading: {exc.strerror or exc}") from exc
        close = True

    def values() -> Iterable[float]:
        for line_number, line in enumerate(stream, start=1):
            for token in line.split():
                try:
                    yield float(token)
                except ValueError as exc:
                    raise CLIError(
                        f"could not parse value {token!r} on line {line_number} of {label}"
                    ) from exc

    try:
        # fromiter avoids a second, Python-object copy of large input series.
        return np.fromiter(values(), dtype=np.float64)
    finally:
        if close:
            stream.close()


def _common_kwargs(args: argparse.Namespace, devices: list[int] | None) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "pearson": args.output_pearson,
        "precision": _precision(args),
        "threads": args.num_cpu_workers,
        "verbose": args.print_debug_info,
    }
    if args.no_gpu:
        kwargs["gpus"] = []
    elif devices is not None:
        kwargs["gpus"] = devices
    if args.max_tile_size is not None:
        kwargs["max_tile_size"] = args.max_tile_size
    if args.aligned:
        kwargs["allow_trivial_match"] = False
    return kwargs


def _compute_one(
    args: argparse.Namespace,
    series_a: np.ndarray,
    series_b: np.ndarray | None,
    kwargs: dict[str, Any],
) -> Any:
    is_ab = series_b is not None
    if args.profile_type in {"1NN_INDEX", "1NN"}:
        if is_ab:
            return abjoin(series_a, series_b, args.window, **kwargs)
        return selfjoin(series_a, args.window, **kwargs)
    if args.profile_type == "SUM_THRESH":
        kwargs["threshold"] = args.threshold
        if is_ab:
            return abjoin_sum(series_a, series_b, args.window, **kwargs)
        return selfjoin_sum(series_a, args.window, **kwargs)
    if args.profile_type == "ALL_NEIGHBORS":
        kwargs["threshold"] = args.threshold
        if is_ab:
            return abjoin_knn(
                series_a,
                series_b,
                args.window,
                args.max_matches_per_column,
                **kwargs,
            )
        return selfjoin_knn(
            series_a,
            args.window,
            args.max_matches_per_column,
            **kwargs,
        )
    if args.profile_type == "MATRIX_SUMMARY":
        kwargs.update(
            threshold=args.threshold,
            mheight=args.reduced_height,
            mwidth=args.reduced_width,
        )
        if is_ab:
            return abjoin_matrix(series_a, series_b, args.window, **kwargs)
        return selfjoin_matrix(series_a, args.window, **kwargs)
    raise CLIError(f"unsupported profile type {args.profile_type!r}")


def _format_number(value: Any) -> str:
    return format(float(value), ".10g")


def _profile_lines(profile: Iterable[Any]) -> Iterable[str]:
    for value in profile:
        yield _format_number(value)


def _index_lines(index: Iterable[Any]) -> Iterable[str]:
    for value in index:
        yield str(int(value))


def _matrix_lines(matrix: Any) -> Iterable[str]:
    for row in np.asarray(matrix):
        yield " ".join(_format_number(value) for value in row)


def _match_lines(matches: Iterable[Sequence[Any]], pearson: bool) -> Iterable[str]:
    normalized = [(int(col), int(row), float(value)) for col, row, value in matches]
    if pearson:
        normalized.sort(key=lambda match: (match[0], -match[2], match[1]))
    else:
        normalized.sort(key=lambda match: (match[0], match[2], match[1]))
    for col, row, value in normalized:
        yield f"{col} {row} {_format_number(value)}"


def _active_outputs(args: argparse.Namespace) -> list[str]:
    outputs = [args.output_a_file_name]
    if args.profile_type == "1NN_INDEX":
        outputs.append(args.output_a_index_file_name)
    if args.keep_rows:
        outputs.append(args.output_b_file_name)
        if args.profile_type == "1NN_INDEX":
            outputs.append(args.output_b_index_file_name)
    return outputs


def _validate_outputs(args: argparse.Namespace) -> None:
    outputs = _active_outputs(args)
    if any(not output for output in outputs):
        raise CLIError("active output filenames cannot be empty")
    if outputs.count("-") > 1:
        raise CLIError("at most one active output can be written to stdout")
    file_outputs = [output for output in outputs if output != "-"]
    if len(file_outputs) != len(set(file_outputs)):
        raise CLIError("active output filenames must be distinct")


def _write_lines(name: str, lines: Iterable[str], stdout: TextIO) -> None:
    if name == "-":
        stream = stdout
        close = False
    else:
        try:
            stream = Path(name).open("w", encoding="utf-8", newline="\n")
        except OSError as exc:
            raise CLIError(f"unable to open {name!r} for writing: {exc.strerror or exc}") from exc
        close = True
    try:
        batch: list[str] = []
        for line in lines:
            batch.append(line)
            if len(batch) == OUTPUT_BATCH_LINES:
                stream.write("\n".join(batch) + "\n")
                batch.clear()
        if batch:
            stream.write("\n".join(batch) + "\n")
        stream.flush()
    except OSError as exc:
        raise CLIError(f"unable to write {name!r}: {exc.strerror or exc}") from exc
    finally:
        if close:
            stream.close()


def _write_result(
    args: argparse.Namespace,
    result: Any,
    *,
    row_profile: bool,
    stdout: TextIO,
) -> None:
    profile_name = args.output_b_file_name if row_profile else args.output_a_file_name
    index_name = args.output_b_index_file_name if row_profile else args.output_a_index_file_name
    if args.profile_type in {"1NN_INDEX", "1NN"}:
        profile, index = result
        _write_lines(profile_name, _profile_lines(profile), stdout)
        if args.profile_type == "1NN_INDEX":
            _write_lines(index_name, _index_lines(index), stdout)
    elif args.profile_type == "MATRIX_SUMMARY":
        _write_lines(profile_name, _matrix_lines(result), stdout)
    elif args.profile_type == "ALL_NEIGHBORS":
        _write_lines(profile_name, _match_lines(result, args.output_pearson), stdout)
    else:
        _write_lines(profile_name, _profile_lines(result), stdout)


def _fallback_variant_descriptions() -> list[str]:
    descriptions = []
    if gpu_supported():
        descriptions.append("v0 device=metal precision=single family=mlx-runtime")
    descriptions.append(
        f"v{len(descriptions)} device=cpu precision=single,double,ultra family=mlx-runtime"
    )
    return descriptions


def _variant_descriptions() -> list[str]:
    try:
        module = importlib.import_module("mlx_native_scamp._autotune")
        describe = getattr(module, "variant_descriptions")
    except (ImportError, AttributeError):
        return _fallback_variant_descriptions()
    return [str(description) for description in describe()]


def _run_autotune(devices: list[int] | None) -> None:
    try:
        pyscamp = importlib.import_module("pyscamp")
        autotune = getattr(pyscamp, "autotune")
    except (ImportError, AttributeError) as exc:
        raise CLIError(
            "--autotune requires the optional MLX autotuning implementation"
        ) from exc
    try:
        autotune(devices=devices)
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        raise CLIError(f"Autotune failed: {exc}") from exc


def run(
    args: argparse.Namespace,
    *,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    stdin = sys.stdin if stdin is None else stdin
    stdout = sys.stdout if stdout is None else stdout
    stderr = sys.stderr if stderr is None else stderr

    if args.list_variants:
        for description in _variant_descriptions():
            stdout.write(description + "\n")
        return 0
    devices = _parse_gpu_ids(args.gpus)
    if args.autotune:
        if args.no_gpu:
            raise CLIError("--autotune cannot be combined with --no_gpu")
        _run_autotune(devices)
        return 0

    _validate_job(args, devices)
    _validate_outputs(args)
    if args.print_debug_info:
        stderr.write("Starting SCAMP\n")

    series_a = _read_series(args.input_a_file_name, stdin)
    series_b = _read_series(args.input_b_file_name, stdin) if args.input_b_file_name else None
    if len(series_a) < args.window or (series_b is not None and len(series_b) < args.window):
        raise CLIError("window size must be smaller than or equal to the time-series length")
    if args.profile_type == "MATRIX_SUMMARY":
        columns = len(series_a) - args.window + 1
        rows = (len(series_b) if series_b is not None else len(series_a)) - args.window + 1
        if args.reduced_width > columns or args.reduced_height > rows:
            raise CLIError(
                "reduced matrix dimensions must not exceed the distance matrix dimensions"
            )

    kwargs = _common_kwargs(args, devices)
    try:
        result_a = _compute_one(args, series_a, series_b, kwargs.copy())
        result_b = None
        if args.keep_rows:
            result_b = _compute_one(args, series_b, series_a, kwargs.copy())
    except (TypeError, ValueError, RuntimeError) as exc:
        raise CLIError(str(exc)) from exc

    if args.print_debug_info:
        stderr.write("Now writing result to files\n")
    _write_result(args, result_a, row_profile=False, stdout=stdout)
    if result_b is not None:
        _write_result(args, result_b, row_profile=True, stdout=stdout)
    if args.print_debug_info:
        stderr.write("Done\n")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run(args)
    except CLIError as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
