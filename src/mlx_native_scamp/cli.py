from __future__ import annotations

import argparse
import contextlib
import errno
import math
import os
import re
import secrets
import stat
import sys
import tempfile
import unicodedata
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

import numpy as np

from . import (
    abjoin,
    abjoin_1nn,
    abjoin_bidirectional,
    abjoin_knn,
    abjoin_matrix,
    abjoin_sum,
    autotune,
    gpu_supported,
    selfjoin,
    selfjoin_1nn,
    selfjoin_knn,
    selfjoin_matrix,
    selfjoin_sum,
    strategy_descriptions,
)


PROFILE_TYPES = (
    "1NN_INDEX",
    "1NN",
    "SUM_THRESH",
    "ALL_NEIGHBORS",
    "MATRIX_SUMMARY",
)
OUTPUT_BATCH_LINES = 4096
INT32_MIN = -(2**31)
INT32_MAX = 2**31 - 1
INT64_MIN = -(2**63)
INT64_MAX = 2**63 - 1


class CLIError(Exception):
    """An expected command-line input, capability, or filesystem error."""


FaultHook = Callable[[str, int], None]
_FAULT_HOOK: FaultHook | None = None
_DECIMAL_INTEGER = re.compile(r"[+-]?[0-9]+\Z")
_PATH_EXCEPTIONS = (OSError, RuntimeError, ValueError, UnicodeError)


def _fault(point: str, index: int = 0) -> None:
    """Private deterministic checkpoint used by transaction/TOCTOU tests."""

    if _FAULT_HOOK is not None:
        _FAULT_HOOK(point, index)


def _parse_bool(value: str) -> bool:
    normalized = value.casefold()
    if normalized in {"1", "true", "t", "yes", "y"}:
        return True
    if normalized in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(
        f"expected a boolean value, got {value!r}"
    )


def _decimal_integer(value: str, *, minimum: int, maximum: int) -> int:
    if _DECIMAL_INTEGER.fullmatch(value) is None:
        raise argparse.ArgumentTypeError(
            f"expected a base-10 integer, got {value!r}"
        )
    result = int(value, 10)
    if result < minimum or result > maximum:
        raise argparse.ArgumentTypeError(
            f"integer must be between {minimum} and {maximum}"
        )
    return result


def _int32(value: str) -> int:
    return _decimal_integer(value, minimum=INT32_MIN, maximum=INT32_MAX)


def _int64(value: str) -> int:
    return _decimal_integer(value, minimum=INT64_MIN, maximum=INT64_MAX)


def _option(name: str) -> tuple[str, str]:
    return f"--{name}", f"-{name}"


def _boolean_flag(
    parser: argparse.ArgumentParser,
    name: str,
    help_text: str,
) -> None:
    parser.add_argument(
        *_option(name),
        dest=name,
        nargs="?",
        const=True,
        default=False,
        type=_parse_bool,
        metavar="BOOL",
        help=help_text,
    )
    parser.add_argument(
        *_option(f"no{name}"),
        dest=name,
        action="store_false",
        help=argparse.SUPPRESS,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scamp",
        allow_abbrev=False,
        description="Compute SCAMP matrix profiles natively with Apple MLX.",
    )
    parser.add_argument(
        "--helpshort",
        "-helpshort",
        "--helpfull",
        "-helpfull",
        action="help",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        *_option("window"),
        type=_int32,
        default=-1,
        help="subsequence length (at least 3)",
    )
    parser.add_argument(
        *_option("input_a_file_name"),
        default="",
        help="primary input file, or '-' for stdin",
    )
    parser.add_argument(
        *_option("input_b_file_name"),
        default="",
        help="secondary AB-join input file, or '-' for stdin",
    )
    parser.add_argument(
        *_option("output_a_file_name"),
        default="mp_columns_out",
        help="primary profile output, or '-' for stdout",
    )
    parser.add_argument(
        *_option("output_a_index_file_name"),
        default="mp_columns_out_index",
        help="primary index output, or '-' for stdout",
    )
    parser.add_argument(
        *_option("output_b_file_name"),
        default="mp_rows_out",
        help="row profile output, or '-' for stdout",
    )
    parser.add_argument(
        *_option("output_b_index_file_name"),
        default="mp_rows_out_index",
        help="row index output, or '-' for stdout",
    )
    parser.add_argument(
        *_option("profile_type"),
        default="1NN_INDEX",
        choices=PROFILE_TYPES,
    )
    parser.add_argument(*_option("threshold"), type=float, default=0.0)
    parser.add_argument(
        *_option("max_matches_per_column"),
        type=_int64,
        default=5,
    )
    parser.add_argument(
        *_option("reduced_height"), type=_int32, default=50
    )
    parser.add_argument(
        *_option("reduced_width"), type=_int32, default=50
    )
    parser.add_argument(
        *_option("num_cpu_workers"), type=_int32, default=0
    )
    parser.add_argument(
        *_option("max_tile_size"), type=_int32, default=None
    )
    parser.add_argument(
        *_option("gpus"),
        default="",
        help="comma- or space-separated MLX Metal device IDs",
    )
    parser.add_argument(*_option("global_row"), type=_int64, default=-1)
    parser.add_argument(*_option("global_col"), type=_int64, default=-1)

    _boolean_flag(
        parser,
        "output_pearson",
        "write Pearson correlation instead of Euclidean distance",
    )
    _boolean_flag(
        parser, "print_debug_info", "write resolved execution details to stderr"
    )
    _boolean_flag(parser, "no_gpu", "select the MLX CPU backend")
    _boolean_flag(parser, "ultra_precision", "use ultra precision")
    _boolean_flag(parser, "double_precision", "use double precision")
    _boolean_flag(parser, "single_precision", "use single precision")
    _boolean_flag(
        parser, "keep_rows", "also compute the row-wise AB profile"
    )
    _boolean_flag(
        parser, "aligned", "apply the aligned AB-join exclusion zone"
    )
    _boolean_flag(
        parser, "autotune", "run the bounded MLX autotuner and exit"
    )
    _boolean_flag(
        parser, "list_variants", "list versioned MLX strategies and exit"
    )
    return parser


def _parse_gpu_ids(value: str) -> list[int] | None:
    tokens = value.replace(",", " ").split()
    if not tokens:
        return None
    devices: list[int] = []
    for token in tokens:
        try:
            device = _decimal_integer(token, minimum=0, maximum=INT32_MAX)
        except argparse.ArgumentTypeError as exc:
            raise CLIError(
                "--gpus must contain comma- or space-separated integer device IDs"
            ) from exc
        devices.append(device)
    if len(devices) != len(set(devices)):
        raise CLIError("--gpus contains a duplicate device ID")
    unsupported = next((device for device in devices if device != 0), None)
    if unsupported is not None:
        raise CLIError(
            "MLX/Metal exposes only GPU device 0; "
            f"unsupported device ID {unsupported}"
        )
    return devices


def _precision(args: argparse.Namespace) -> str:
    selected = [
        precision
        for precision, enabled in (
            ("ultra", args.ultra_precision),
            ("double", args.double_precision),
            ("single", args.single_precision),
        )
        if enabled
    ]
    if len(selected) > 1:
        raise CLIError("only one precision flag can be enabled at a time")
    return selected[0] if selected else "double"


def _active_outputs(args: argparse.Namespace) -> list[str]:
    outputs = [args.output_a_file_name]
    if args.profile_type == "1NN_INDEX":
        outputs.append(args.output_a_index_file_name)
    if args.keep_rows:
        outputs.append(args.output_b_file_name)
        if args.profile_type == "1NN_INDEX":
            outputs.append(args.output_b_index_file_name)
    return outputs


def _validate_job(
    args: argparse.Namespace,
    devices: list[int] | None,
) -> None:
    if args.window < 3:
        raise CLIError(
            "Subsequence length must be at least 3; use --window=<window_size>"
        )
    if not args.input_a_file_name:
        raise CLIError(
            "primary input filename must be specified using --input_a_file_name"
        )
    if args.input_a_file_name == "-" and args.input_b_file_name == "-":
        raise CLIError("only one input can read from stdin")
    if args.profile_type in {
        "SUM_THRESH",
        "ALL_NEIGHBORS",
        "MATRIX_SUMMARY",
    } and (
        not math.isfinite(args.threshold)
        or not -1.0 <= args.threshold <= 1.0
    ):
        raise CLIError("threshold must be finite and between -1 and 1")
    if (
        args.profile_type == "ALL_NEIGHBORS"
        and args.max_matches_per_column <= 0
    ):
        raise CLIError("max_matches_per_column must be greater than 0")
    if args.profile_type == "MATRIX_SUMMARY" and (
        args.reduced_height <= 0 or args.reduced_width <= 0
    ):
        raise CLIError("reduced matrix dimensions must be greater than 0")
    if args.num_cpu_workers < 0:
        raise CLIError("num_cpu_workers must be greater than or equal to 0")
    if args.global_row != -1 or args.global_col != -1:
        raise CLIError(
            "--global_row and --global_col require the distributed partition API"
        )
    if args.aligned and not args.input_b_file_name:
        raise CLIError("--aligned is only valid for AB-joins")
    if args.keep_rows and not args.input_b_file_name:
        raise CLIError("--keep_rows is only valid for AB-joins")
    if args.keep_rows and args.profile_type == "MATRIX_SUMMARY":
        raise CLIError("--keep_rows is not defined for MATRIX_SUMMARY profiles")

    precision = _precision(args)
    if args.no_gpu and devices:
        raise CLIError("--no_gpu cannot be combined with a non-empty --gpus list")
    if devices and args.num_cpu_workers > 0:
        raise CLIError("concurrent CPU and Metal execution is not supported by MLX")
    if devices and precision != "single":
        raise CLIError(
            "Metal supports only single precision; use --single_precision "
            "or select CPU execution"
        )
    if devices and not gpu_supported():
        raise CLIError("MLX Metal GPU device 0 is unavailable on this system")

    if args.max_tile_size is not None:
        if args.max_tile_size < 1024:
            raise CLIError("max tile size must be at least 1024")
        if args.max_tile_size < 2 * args.window:
            raise CLIError("max tile size must be at least twice the window size")

    outputs = _active_outputs(args)
    if any(not output for output in outputs):
        raise CLIError("active output filenames cannot be empty")
    if outputs.count("-") > 1:
        raise CLIError("at most one active output can be written to stdout")


@dataclass(frozen=True, slots=True)
class _FileId:
    device: int
    inode: int


@dataclass(frozen=True, slots=True)
class _NodeIdentity:
    file_id: _FileId
    kind: int
    permissions: int


@dataclass(frozen=True, slots=True)
class _Fingerprint:
    file_id: _FileId
    kind: int
    permissions: int
    links: int
    size: int
    mtime_ns: int
    ctime_ns: int
    flags: int | None


@dataclass(frozen=True, slots=True)
class _ParentSnapshot:
    raw: Path
    canonical: Path
    raw_node: _NodeIdentity
    canonical_node: _NodeIdentity


@dataclass(frozen=True, slots=True)
class _PathSnapshot:
    raw: Path
    canonical: Path
    folded: str
    parent: _ParentSnapshot
    target: _Fingerprint | None
    raw_node: _Fingerprint | None
    is_input: bool


@dataclass(frozen=True, slots=True)
class _SafetyPlan:
    inputs: tuple[_PathSnapshot, ...]
    outputs: tuple[_PathSnapshot, ...]


def _file_id(value: os.stat_result) -> _FileId:
    return _FileId(int(value.st_dev), int(value.st_ino))


def _node_identity(value: os.stat_result) -> _NodeIdentity:
    return _NodeIdentity(
        _file_id(value),
        stat.S_IFMT(value.st_mode),
        stat.S_IMODE(value.st_mode),
    )


def _fingerprint(value: os.stat_result) -> _Fingerprint:
    flags = getattr(value, "st_flags", None)
    return _Fingerprint(
        _file_id(value),
        stat.S_IFMT(value.st_mode),
        stat.S_IMODE(value.st_mode),
        int(value.st_nlink),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
        None if flags is None else int(flags),
    )


def _path_key(path: Path) -> str:
    text = unicodedata.normalize("NFKC", os.fspath(path))
    return unicodedata.normalize("NFKC", text.casefold())


def _absolute(path: str | Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _lstat_optional(path: Path) -> os.stat_result | None:
    try:
        return path.lstat()
    except OSError as exc:
        if exc.errno in {errno.ENOENT, errno.ENOTDIR}:
            return None
        raise


def _capture_parent(raw: Path) -> _ParentSnapshot:
    raw_parent = raw.parent
    try:
        raw_stat = raw_parent.lstat()
        canonical = raw_parent.resolve(strict=True)
        canonical_stat = canonical.stat()
    except _PATH_EXCEPTIONS as exc:
        raise CLIError(
            f"unable to inspect parent directory {os.fspath(raw_parent)!r}: {exc}"
        ) from exc
    if not stat.S_ISDIR(canonical_stat.st_mode):
        raise CLIError(
            f"output/input parent is not a directory: {os.fspath(raw_parent)!r}"
        )
    return _ParentSnapshot(
        raw_parent,
        canonical,
        _node_identity(raw_stat),
        _node_identity(canonical_stat),
    )


def _capture_input(name: str) -> _PathSnapshot:
    raw = _absolute(name)
    parent = _capture_parent(raw)
    try:
        raw_stat = raw.lstat()
        canonical = raw.resolve(strict=True)
        target_stat = raw.stat()
    except _PATH_EXCEPTIONS as exc:
        raise CLIError(f"unable to inspect input {name!r}: {exc}") from exc
    if not stat.S_ISREG(target_stat.st_mode):
        raise CLIError(f"input is not a regular file: {name!r}")
    return _PathSnapshot(
        raw,
        canonical,
        _path_key(canonical),
        parent,
        _fingerprint(target_stat),
        _fingerprint(raw_stat),
        True,
    )


def _capture_output(name: str) -> _PathSnapshot:
    raw = _absolute(name)
    parent = _capture_parent(raw)
    try:
        raw_stat = _lstat_optional(raw)
    except _PATH_EXCEPTIONS as exc:
        raise CLIError(f"unable to inspect output {name!r}: {exc}") from exc
    if raw_stat is not None and stat.S_ISLNK(raw_stat.st_mode):
        raise CLIError(f"output paths cannot be symbolic links: {name!r}")
    if raw_stat is not None and not stat.S_ISREG(raw_stat.st_mode):
        raise CLIError(f"output is not a regular file: {name!r}")
    canonical = parent.canonical / raw.name
    if raw_stat is not None:
        try:
            resolved = raw.resolve(strict=True)
        except _PATH_EXCEPTIONS as exc:
            raise CLIError(f"unable to resolve output {name!r}: {exc}") from exc
        if resolved != canonical:
            raise CLIError(f"output path changed while it was inspected: {name!r}")
    target = None if raw_stat is None else _fingerprint(raw_stat)
    return _PathSnapshot(
        raw,
        canonical,
        _path_key(canonical),
        parent,
        target,
        target,
        False,
    )


def _check_aliases(
    inputs: Sequence[_PathSnapshot],
    outputs: Sequence[_PathSnapshot],
) -> None:
    output_keys = [output.folded for output in outputs]
    if len(output_keys) != len(set(output_keys)):
        raise CLIError("active output filenames must be distinct")
    existing_output_ids = [
        output.target.file_id for output in outputs if output.target is not None
    ]
    if len(existing_output_ids) != len(set(existing_output_ids)):
        raise CLIError("active output filenames must be distinct")

    input_keys = {input_path.folded for input_path in inputs}
    input_ids = {
        input_path.target.file_id
        for input_path in inputs
        if input_path.target is not None
    }
    for output in outputs:
        if output.folded in input_keys or (
            output.target is not None and output.target.file_id in input_ids
        ):
            raise CLIError("an output filename aliases an input file")


def _capture_plan(args: argparse.Namespace) -> _SafetyPlan:
    inputs = tuple(
        _capture_input(name)
        for name in (args.input_a_file_name, args.input_b_file_name)
        if name and name != "-"
    )
    outputs = tuple(
        _capture_output(name)
        for name in _active_outputs(args)
        if name != "-"
    )
    _check_aliases(inputs, outputs)
    return _SafetyPlan(inputs, outputs)


def _verify_parent(parent: _ParentSnapshot) -> None:
    try:
        raw_stat = parent.raw.lstat()
        canonical = parent.raw.resolve(strict=True)
        canonical_stat = canonical.stat()
    except _PATH_EXCEPTIONS as exc:
        raise CLIError(
            f"parent directory changed during execution: {parent.raw}: {exc}"
        ) from exc
    if (
        _node_identity(raw_stat) != parent.raw_node
        or canonical != parent.canonical
        or _node_identity(canonical_stat) != parent.canonical_node
    ):
        raise CLIError(
            f"parent directory changed during execution: {parent.raw}"
        )


def _verify_input(snapshot: _PathSnapshot) -> None:
    _verify_parent(snapshot.parent)
    try:
        raw_stat = snapshot.raw.lstat()
        canonical = snapshot.raw.resolve(strict=True)
        target_stat = snapshot.raw.stat()
    except _PATH_EXCEPTIONS as exc:
        raise CLIError(f"input changed during execution: {snapshot.raw}: {exc}") from exc
    if (
        canonical != snapshot.canonical
        or _fingerprint(raw_stat) != snapshot.raw_node
        or _fingerprint(target_stat) != snapshot.target
    ):
        raise CLIError(f"input changed during execution: {snapshot.raw}")


def _verify_output(snapshot: _PathSnapshot) -> None:
    _verify_parent(snapshot.parent)
    try:
        current = _lstat_optional(snapshot.raw)
    except _PATH_EXCEPTIONS as exc:
        raise CLIError(
            f"output changed during execution: {snapshot.raw}: {exc}"
        ) from exc
    if snapshot.target is None:
        if current is not None:
            raise CLIError(f"output changed during execution: {snapshot.raw}")
    else:
        if current is None or stat.S_ISLNK(current.st_mode):
            raise CLIError(f"output changed during execution: {snapshot.raw}")
        try:
            canonical = snapshot.raw.resolve(strict=True)
        except _PATH_EXCEPTIONS as exc:
            raise CLIError(
                f"output changed during execution: {snapshot.raw}: {exc}"
            ) from exc
        if canonical != snapshot.canonical or _fingerprint(current) != snapshot.target:
            raise CLIError(f"output changed during execution: {snapshot.raw}")


def _verify_plan(plan: _SafetyPlan) -> None:
    for snapshot in plan.inputs:
        _verify_input(snapshot)
    for snapshot in plan.outputs:
        _verify_output(snapshot)
    _check_aliases(plan.inputs, plan.outputs)


def _read_stream(
    stream: TextIO,
    label: str,
    dtype: type[np.float32] | type[np.float64],
) -> np.ndarray:
    def values() -> Iterator[float]:
        for line_number, line in enumerate(stream, start=1):
            for token in line.split():
                try:
                    yield float(token)
                except ValueError as exc:
                    raise CLIError(
                        f"could not parse value {token!r} on line "
                        f"{line_number} of {label}"
                    ) from exc

    try:
        return np.fromiter(values(), dtype=dtype)
    except OverflowError as exc:
        raise CLIError(f"numeric value in {label} is outside the selected precision") from exc
    except UnicodeError as exc:
        raise CLIError(f"input is not valid UTF-8: {label}") from exc
    except OSError as exc:
        raise CLIError(f"unable to read input {label!r}: {exc}") from exc


def _read_file(
    snapshot: _PathSnapshot,
    dtype: type[np.float32] | type[np.float64],
    index: int,
) -> np.ndarray:
    flags = os.O_RDONLY
    flags |= getattr(os, "O_CLOEXEC", 0)
    _fault("before_input_open", index)
    try:
        descriptor = os.open(snapshot.raw, flags)
    except OSError as exc:
        raise CLIError(f"unable to open input {snapshot.raw!r}: {exc}") from exc
    try:
        if _fingerprint(os.fstat(descriptor)) != snapshot.target:
            raise CLIError(f"input changed while it was opened: {snapshot.raw}")
        with os.fdopen(
            descriptor,
            "r",
            encoding="utf-8",
            errors="strict",
            newline=None,
        ) as stream:
            descriptor = -1
            result = _read_stream(stream, os.fspath(snapshot.raw), dtype)
    except UnicodeError as exc:
        raise CLIError(f"input is not valid UTF-8: {snapshot.raw}") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    _fault("after_input_read", index)
    _verify_input(snapshot)
    return result


def _read_inputs(
    args: argparse.Namespace,
    plan: _SafetyPlan,
    stdin: TextIO,
) -> tuple[np.ndarray, np.ndarray | None]:
    dtype = np.float32 if _precision(args) == "single" else np.float64
    snapshots = iter(plan.inputs)
    results: list[np.ndarray | None] = []
    for index, name in enumerate(
        (args.input_a_file_name, args.input_b_file_name)
    ):
        if not name:
            results.append(None)
        elif name == "-":
            _fault("before_input_read", index)
            results.append(_read_stream(stdin, "stdin", dtype))
            _fault("after_input_read", index)
        else:
            results.append(_read_file(next(snapshots), dtype, index))
    return results[0], results[1]


def _validate_series(
    args: argparse.Namespace,
    series_a: np.ndarray,
    series_b: np.ndarray | None,
) -> None:
    if len(series_a) < args.window or (
        series_b is not None and len(series_b) < args.window
    ):
        raise CLIError(
            "window size must be smaller than or equal to the time-series length"
        )
    if args.profile_type == "MATRIX_SUMMARY":
        columns = len(series_a) - args.window + 1
        row_length = len(series_a) if series_b is None else len(series_b)
        rows = row_length - args.window + 1
        if args.reduced_width > columns or args.reduced_height > rows:
            raise CLIError(
                "reduced matrix dimensions must not exceed the distance "
                "matrix dimensions"
            )


def _common_kwargs(
    args: argparse.Namespace,
    devices: list[int] | None,
) -> dict[str, Any]:
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


def _compute(
    args: argparse.Namespace,
    series_a: np.ndarray,
    series_b: np.ndarray | None,
    kwargs: dict[str, Any],
) -> tuple[Any, Any | None]:
    is_ab = series_b is not None
    if args.profile_type == "1NN_INDEX":
        if is_ab and args.keep_rows:
            return abjoin_bidirectional(
                series_a, series_b, args.window, **kwargs
            )
        if is_ab:
            return abjoin(series_a, series_b, args.window, **kwargs), None
        return selfjoin(series_a, args.window, **kwargs), None

    if args.profile_type == "1NN":
        if is_ab and args.keep_rows:
            forward, reverse = abjoin_bidirectional(
                series_a, series_b, args.window, **kwargs
            )
            return forward[0], reverse[0]
        if is_ab:
            return abjoin_1nn(series_a, series_b, args.window, **kwargs), None
        return selfjoin_1nn(series_a, args.window, **kwargs), None

    profile_kwargs = dict(kwargs)
    profile_kwargs["threshold"] = args.threshold
    if args.profile_type == "SUM_THRESH":
        if is_ab:
            forward = abjoin_sum(
                series_a, series_b, args.window, **profile_kwargs
            )
            reverse = (
                abjoin_sum(series_b, series_a, args.window, **profile_kwargs)
                if args.keep_rows
                else None
            )
            return forward, reverse
        return selfjoin_sum(series_a, args.window, **profile_kwargs), None

    if args.profile_type == "ALL_NEIGHBORS":
        k = args.max_matches_per_column
        if is_ab:
            forward = abjoin_knn(
                series_a, series_b, args.window, k, **profile_kwargs
            )
            reverse = (
                abjoin_knn(
                    series_b, series_a, args.window, k, **profile_kwargs
                )
                if args.keep_rows
                else None
            )
            return forward, reverse
        return selfjoin_knn(
            series_a, args.window, k, **profile_kwargs
        ), None

    if args.profile_type == "MATRIX_SUMMARY":
        profile_kwargs.update(
            mheight=args.reduced_height,
            mwidth=args.reduced_width,
        )
        if is_ab:
            return abjoin_matrix(
                series_a, series_b, args.window, **profile_kwargs
            ), None
        return selfjoin_matrix(
            series_a, args.window, **profile_kwargs
        ), None
    raise CLIError(f"unsupported profile type {args.profile_type!r}")


def _format_number(value: Any) -> str:
    return format(float(value), ".10g")


def _profile_lines(profile: Iterable[Any]) -> Iterator[str]:
    for value in profile:
        yield _format_number(value)


def _index_lines(indices: Iterable[Any]) -> Iterator[str]:
    for value in indices:
        yield str(int(value))


def _matrix_lines(matrix: Any) -> Iterator[str]:
    for row in np.asarray(matrix):
        yield " ".join(_format_number(value) for value in row)


def _match_lines(
    matches: Iterable[Sequence[Any]], pearson: bool
) -> Iterator[str]:
    # The native KNN reducer already emits columns in ascending order and each
    # column nearest-first (with row as the deterministic tie-breaker).
    # Preserve that order without duplicating an O(N*K) result in Python.
    del pearson
    for column, row, value in matches:
        yield f"{column} {row} {_format_number(value)}"


@dataclass(frozen=True, slots=True)
class _OutputSpec:
    name: str
    lines: Iterable[str]


def _result_outputs(
    args: argparse.Namespace,
    forward: Any,
    reverse: Any | None,
) -> list[_OutputSpec]:
    if args.profile_type == "1NN_INDEX":
        forward_values, forward_indices = forward
        specs = [
            _OutputSpec(
                args.output_a_file_name, _profile_lines(forward_values)
            ),
            _OutputSpec(
                args.output_a_index_file_name, _index_lines(forward_indices)
            ),
        ]
        if reverse is not None:
            reverse_values, reverse_indices = reverse
            specs.extend(
                (
                    _OutputSpec(
                        args.output_b_file_name,
                        _profile_lines(reverse_values),
                    ),
                    _OutputSpec(
                        args.output_b_index_file_name,
                        _index_lines(reverse_indices),
                    ),
                )
            )
        return specs

    if args.profile_type == "MATRIX_SUMMARY":
        return [_OutputSpec(args.output_a_file_name, _matrix_lines(forward))]
    if args.profile_type == "ALL_NEIGHBORS":
        line_builder = lambda result: _match_lines(  # noqa: E731
            result, args.output_pearson
        )
    else:
        line_builder = _profile_lines
    specs = [_OutputSpec(args.output_a_file_name, line_builder(forward))]
    if reverse is not None:
        specs.append(
            _OutputSpec(args.output_b_file_name, line_builder(reverse))
        )
    return specs


def _write_lines(stream: TextIO, lines: Iterable[str]) -> None:
    batch: list[str] = []
    for line in lines:
        batch.append(line)
        if len(batch) == OUTPUT_BATCH_LINES:
            _write_all(stream, "\n".join(batch) + "\n")
            batch.clear()
    if batch:
        _write_all(stream, "\n".join(batch) + "\n")


def _write_all(stream: TextIO, value: str) -> None:
    offset = 0
    while offset < len(value):
        written = stream.write(value[offset:])
        if not isinstance(written, int) or written <= 0:
            raise OSError("output stream made no progress")
        if written > len(value) - offset:
            raise OSError("output stream reported an invalid write length")
        offset += written


def _probe_output_mode(parent: Path, target_name: str, index: int) -> int:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_CLOEXEC", 0)
    for _ in range(100):
        probe = parent / (
            f".{target_name}.{secrets.token_hex(12)}.mode-probe"
        )
        try:
            descriptor = os.open(probe, flags, 0o666)
        except FileExistsError:
            continue
        probe_id = _file_id(os.fstat(descriptor))
        try:
            _fault("mode_probe", index)
            return stat.S_IMODE(os.fstat(descriptor).st_mode)
        finally:
            os.close(descriptor)
            _safe_unlink_owned(probe, probe_id)
    raise CLIError(f"unable to allocate a mode probe in {parent}")


def _copy_macos_metadata(source: Path, destination: Path) -> None:
    """Copy stat, ACL, and xattr metadata without copying file data."""

    if sys.platform != "darwin":
        return
    import ctypes

    copyfile = ctypes.CDLL(None, use_errno=True).copyfile
    copyfile.argtypes = (
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_void_p,
        ctypes.c_uint32,
    )
    copyfile.restype = ctypes.c_int
    # COPYFILE_METADATA = COPYFILE_ACL | COPYFILE_STAT | COPYFILE_XATTR.
    if copyfile(os.fsencode(source), os.fsencode(destination), None, 0x7) != 0:
        error_number = ctypes.get_errno()
        raise OSError(
            error_number,
            os.strerror(error_number),
            os.fspath(source),
        )


def _rename_exclusive(source: Path, destination: Path) -> None:
    """Rename without replacing an occupied destination on macOS."""

    if sys.platform == "darwin":
        import ctypes

        renameatx_np = ctypes.CDLL(None, use_errno=True).renameatx_np
        renameatx_np.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        renameatx_np.restype = ctypes.c_int
        # AT_FDCWD=-2 and RENAME_EXCL=0x4.
        if (
            renameatx_np(
                -2,
                os.fsencode(source),
                -2,
                os.fsencode(destination),
                0x4,
            )
            != 0
        ):
            error_number = ctypes.get_errno()
            raise OSError(
                error_number,
                os.strerror(error_number),
                os.fspath(destination),
            )
        return
    # The project targets macOS. This fallback keeps source testing usable on
    # other platforms, but they do not provide renameatx_np(RENAME_EXCL).
    if _lstat_optional(destination) is not None:
        raise FileExistsError(
            errno.EEXIST, os.strerror(errno.EEXIST), destination
        )
    os.rename(source, destination)


@dataclass(slots=True)
class _StagedFile:
    snapshot: _PathSnapshot
    temporary: Path
    staged: _Fingerprint
    backup: Path | None = None
    backup_has_original: bool = False
    installed: bool = False

    @property
    def staged_id(self) -> _FileId:
        return self.staged.file_id


def _safe_unlink_owned(path: Path, expected: _FileId) -> None:
    current = _current_file_id(path)
    if current is None:
        return
    if current != expected:
        raise OSError(f"refusing to remove replaced path {path}")
    path.unlink()


def _stage_file(
    snapshot: _PathSnapshot,
    lines: Iterable[str],
    index: int,
) -> _StagedFile:
    parent = snapshot.canonical.parent
    desired_mode = (
        snapshot.target.permissions
        if snapshot.target is not None
        else _probe_output_mode(parent, snapshot.canonical.name, index)
    )
    _fault("before_stage", index)
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            dir=parent,
            prefix=f".{snapshot.canonical.name}.",
            suffix=".tmp",
        )
    except OSError as exc:
        raise CLIError(f"unable to stage output {snapshot.raw!r}: {exc}") from exc
    temporary = Path(temporary_name)
    created_id = _file_id(os.fstat(descriptor))
    staged_fingerprint: _Fingerprint | None = None
    try:
        if snapshot.target is not None:
            _verify_output(snapshot)
            _copy_macos_metadata(snapshot.canonical, temporary)
            _verify_output(snapshot)
        with os.fdopen(
            descriptor,
            "w",
            encoding="utf-8",
            errors="strict",
            newline="\n",
        ) as stream:
            descriptor = -1
            _fault("stage_write", index)
            _write_lines(stream, lines)
            _fault("stage_flush", index)
            stream.flush()
            os.fchmod(stream.fileno(), desired_mode)
            _fault("stage_fsync", index)
            os.fsync(stream.fileno())
            staged_fingerprint = _fingerprint(os.fstat(stream.fileno()))
        _fault("after_stage", index)
    except BaseException as exc:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            _safe_unlink_owned(temporary, created_id)
        except BaseException as cleanup_error:
            if hasattr(exc, "add_note"):
                exc.add_note(
                    f"unable to remove incomplete stage {temporary}: "
                    f"{cleanup_error}"
                )
        raise
    assert staged_fingerprint is not None
    return _StagedFile(snapshot, temporary, staged_fingerprint)


def _stage_stdout(lines: Iterable[str], index: int) -> TextIO:
    spool = tempfile.SpooledTemporaryFile(
        max_size=8 * 1024 * 1024,
        mode="w+",
        encoding="utf-8",
        errors="strict",
        newline="\n",
    )
    try:
        _fault("stdout_stage", index)
        _write_lines(spool, lines)
        spool.flush()
        spool.seek(0)
    except BaseException:
        spool.close()
        raise
    return spool


def _stage_outputs(
    plan: _SafetyPlan,
    specs: Sequence[_OutputSpec],
) -> tuple[list[_StagedFile], TextIO | None]:
    staged: list[_StagedFile] = []
    stdout_spool: TextIO | None = None
    snapshots = iter(plan.outputs)
    try:
        for index, spec in enumerate(specs):
            if spec.name == "-":
                stdout_spool = _stage_stdout(spec.lines, index)
            else:
                snapshot = next(snapshots)
                if _absolute(spec.name) != snapshot.raw:
                    raise RuntimeError("output plan/spec order mismatch")
                staged.append(_stage_file(snapshot, spec.lines, index))
    except BaseException as exc:
        cleanup_failures: list[str] = []
        if stdout_spool is not None:
            try:
                stdout_spool.close()
            except BaseException as cleanup_error:
                cleanup_failures.append(f"stdout spool: {cleanup_error}")
        for item in staged:
            try:
                _safe_unlink_owned(item.temporary, item.staged_id)
            except BaseException as cleanup_error:
                cleanup_failures.append(
                    f"{item.temporary}: {cleanup_error}"
                )
        if cleanup_failures and hasattr(exc, "add_note"):
            exc.add_note(
                "staging cleanup problems: " + "; ".join(cleanup_failures)
            )
        raise
    return staged, stdout_spool


def _directory_fsync(parents: Iterable[Path]) -> None:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_DIRECTORY", 0)
    for parent in sorted(set(parents), key=os.fspath):
        descriptor = os.open(parent, flags)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _reserve_backup(item: _StagedFile, index: int) -> None:
    del index
    parent = item.snapshot.canonical.parent
    for _ in range(100):
        candidate = parent / (
            f".{item.snapshot.canonical.name}."
            f"{secrets.token_hex(12)}.backup"
        )
        if _lstat_optional(candidate) is None:
            item.backup = candidate
            return
    raise CLIError(f"unable to reserve a backup name in {parent}")


def _current_file_id(path: Path) -> _FileId | None:
    value = _lstat_optional(path)
    return None if value is None else _file_id(value)


def _rollback(
    staged: Sequence[_StagedFile],
    parents: Sequence[Path],
) -> list[str]:
    failures: list[str] = []
    for index, item in reversed(list(enumerate(staged))):
        target = item.snapshot.canonical
        try:
            target_id = _current_file_id(target)
            backup_id = (
                None
                if item.backup is None
                else _current_file_id(item.backup)
            )
            item.installed = target_id == item.staged_id
            original_id = (
                None
                if item.snapshot.target is None
                else item.snapshot.target.file_id
            )
            item.backup_has_original = (
                original_id is not None and backup_id == original_id
            )
            if item.installed:
                if _current_file_id(target) != item.staged_id:
                    raise OSError(
                        "installed output was replaced; preserving recovery files"
                    )
                if _lstat_optional(item.temporary) is not None:
                    raise OSError(
                        "stage recovery path was occupied; preserving recovery files"
                    )
                _rename_exclusive(target, item.temporary)
                if _current_file_id(item.temporary) != item.staged_id:
                    if _lstat_optional(target) is None:
                        _rename_exclusive(item.temporary, target)
                    raise OSError(
                        "installed output changed while quarantining it; "
                        "preserving recovery files"
                    )
                item.installed = False
                target_id = None
            if original_id is not None and target_id != original_id:
                if (
                    item.backup is None
                    or _current_file_id(item.backup) != original_id
                ):
                    raise OSError(
                        "original backup was replaced; preserving recovery files"
                    )
                if _lstat_optional(target) is not None:
                    raise OSError(
                        "output path was occupied; preserving original backup"
                    )
                _rename_exclusive(item.backup, target)
                if _current_file_id(target) != original_id:
                    if (
                        _lstat_optional(item.backup) is None
                        and _lstat_optional(target) is not None
                    ):
                        _rename_exclusive(target, item.backup)
                    raise OSError(
                        "original backup changed while restoring it; "
                        "preserving recovery files"
                    )
                item.backup_has_original = False
            _fault("rollback", index)
        except BaseException as exc:
            failures.append(f"{target}: {exc}")
    for item in staged:
        try:
            _safe_unlink_owned(item.temporary, item.staged_id)
        except BaseException as exc:
            failures.append(f"cleanup {item.temporary}: {exc}")
    try:
        _directory_fsync(parents)
    except BaseException as exc:
        failures.append(f"directory fsync: {exc}")
    return failures


def _copy_stdout(spool: TextIO, stdout: TextIO) -> None:
    while True:
        block = spool.read(64 * 1024)
        if not block:
            break
        _write_all(stdout, block)
    stdout.flush()


def _commit_outputs(
    plan: _SafetyPlan,
    staged: Sequence[_StagedFile],
    stdout_spool: TextIO | None,
    stdout: TextIO,
    stderr: TextIO,
) -> None:
    parents = [item.snapshot.canonical.parent for item in staged]
    try:
        _fault("before_commit", 0)
        _verify_plan(plan)
        for index, item in enumerate(staged):
            if item.snapshot.target is None:
                continue
            _fault("before_backup", index)
            _verify_output(item.snapshot)
            _reserve_backup(item, index)
            _fault("backup_reserved", index)
            _verify_output(item.snapshot)
            if (
                item.backup is None
                or _lstat_optional(item.backup) is not None
            ):
                raise CLIError(
                    f"backup staging path changed during commit: {item.backup}"
                )
            _rename_exclusive(item.snapshot.canonical, item.backup)
            assert item.snapshot.target is not None
            if _current_file_id(item.backup) != item.snapshot.target.file_id:
                raise CLIError(
                    f"output changed while it was backed up: "
                    f"{item.snapshot.canonical}"
                )
            item.backup_has_original = True
            _fault("after_backup", index)

        for index, item in enumerate(staged):
            target = item.snapshot.canonical
            _fault("before_install", index)
            if _lstat_optional(target) is not None:
                raise CLIError(f"output changed during commit: {target}")
            staged_stat = _lstat_optional(item.temporary)
            if (
                staged_stat is None
                or stat.S_ISLNK(staged_stat.st_mode)
                or _fingerprint(staged_stat) != item.staged
            ):
                raise CLIError(
                    f"staged output changed during commit: {item.temporary}"
                )
            _rename_exclusive(item.temporary, target)
            if _current_file_id(target) != item.staged_id:
                raise CLIError(
                    f"staged output changed while it was installed: {target}"
                )
            item.installed = True
            _fault("after_install", index)
        _fault("commit_fsync", 0)
        _directory_fsync(parents)

        if stdout_spool is not None:
            _fault("before_stdout", 0)
            _copy_stdout(stdout_spool, stdout)
            _fault("after_stdout", 0)
    except BaseException as exc:
        failures = _rollback(staged, parents)
        failure_text = ""
        if failures:
            failure_text = "; rollback problems: " + "; ".join(failures)
        if isinstance(exc, CLIError):
            if failure_text:
                raise CLIError(f"{exc}{failure_text}") from exc
            raise
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            if failure_text and hasattr(exc, "add_note"):
                exc.add_note(failure_text.removeprefix("; "))
            raise
        if isinstance(exc, OSError):
            raise CLIError(
                f"unable to commit completed outputs: {exc}{failure_text}"
            ) from exc
        if failure_text and hasattr(exc, "add_note"):
            exc.add_note(failure_text.removeprefix("; "))
        raise
    finally:
        if stdout_spool is not None:
            stdout_spool.close()

    cleanup_failures: list[str] = []
    for index, item in enumerate(staged):
        if item.backup_has_original and item.backup is not None:
            try:
                _fault("backup_cleanup", index)
                assert item.snapshot.target is not None
                _safe_unlink_owned(
                    item.backup, item.snapshot.target.file_id
                )
                item.backup_has_original = False
            except BaseException as exc:
                cleanup_failures.append(f"{item.backup}: {exc}")
    try:
        _fault("cleanup_fsync", 0)
        _directory_fsync(parents)
    except BaseException as exc:
        cleanup_failures.append(f"directory fsync: {exc}")
    if cleanup_failures:
        stderr.write(
            "Warning: outputs were committed, but recovery cleanup failed: "
            + "; ".join(cleanup_failures)
            + "\n"
        )


def _variant_descriptions() -> list[str]:
    lines: list[str] = []
    for index, description in enumerate(strategy_descriptions()):
        strategy = description.strategy
        fields = [
            f"v{index}",
            f"backend={description.backend}",
            f"strategy={strategy.name}",
            f"profile={strategy.profile}",
            f"route={strategy.route}",
        ]
        fields.extend(f"{key}={value}" for key, value in strategy.parameters)
        lines.append(" ".join(fields))
    return lines


def _run_autotune(devices: list[int] | None) -> None:
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
        try:
            for description in _variant_descriptions():
                _write_all(stdout, description + "\n")
            stdout.flush()
        except OSError as exc:
            raise CLIError(f"unable to write strategy listing: {exc}") from exc
        return 0

    devices = _parse_gpu_ids(args.gpus)
    if args.autotune:
        if args.no_gpu:
            raise CLIError("--autotune cannot be combined with --no_gpu")
        _run_autotune(devices)
        return 0

    _validate_job(args, devices)
    plan = _capture_plan(args)
    _fault("after_preflight", 0)
    series_a, series_b = _read_inputs(args, plan, stdin)
    _verify_plan(plan)
    _validate_series(args, series_a, series_b)

    if args.print_debug_info:
        stderr.write("Starting SCAMP\n")
    kwargs = _common_kwargs(args, devices)
    try:
        output_context = (
            contextlib.redirect_stdout(stderr)
            if args.print_debug_info
            else contextlib.nullcontext()
        )
        with output_context:
            forward, reverse = _compute(args, series_a, series_b, kwargs)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise CLIError(str(exc)) from exc

    _fault("after_compute", 0)
    _verify_plan(plan)
    if args.print_debug_info:
        stderr.write("Staging completed results\n")
    specs = _result_outputs(args, forward, reverse)
    try:
        staged, stdout_spool = _stage_outputs(plan, specs)
    except OSError as exc:
        raise CLIError(f"unable to stage completed outputs: {exc}") from exc
    try:
        _fault("after_staging", 0)
        _verify_plan(plan)
    except BaseException as exc:
        cleanup_failures: list[str] = []
        for item in staged:
            try:
                _safe_unlink_owned(item.temporary, item.staged_id)
            except BaseException as cleanup_error:
                cleanup_failures.append(
                    f"{item.temporary}: {cleanup_error}"
                )
        if stdout_spool is not None:
            try:
                stdout_spool.close()
            except BaseException as cleanup_error:
                cleanup_failures.append(f"stdout spool: {cleanup_error}")
        if cleanup_failures and hasattr(exc, "add_note"):
            exc.add_note(
                "staging cleanup problems: " + "; ".join(cleanup_failures)
            )
        raise
    _commit_outputs(plan, staged, stdout_spool, stdout, stderr)
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
