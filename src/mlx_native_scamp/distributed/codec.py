"""Compact NumPy encodings used by the distributed worker protocol."""

from __future__ import annotations

from typing import Any

import numpy as np

from .proto import scamp_worker_v1_pb2 as messages


_PROTO_TO_DTYPE = {
    messages.ARRAY_DTYPE_FLOAT32: np.dtype("<f4"),
    messages.ARRAY_DTYPE_FLOAT64: np.dtype("<f8"),
    messages.ARRAY_DTYPE_INT64: np.dtype("<i8"),
}
_DTYPE_TO_PROTO = {
    np.dtype("float32"): messages.ARRAY_DTYPE_FLOAT32,
    np.dtype("float64"): messages.ARRAY_DTYPE_FLOAT64,
    np.dtype("int64"): messages.ARRAY_DTYPE_INT64,
}


def encode_array(
    values: Any, *, dtype: np.dtype[Any] | type[Any] | None = None
) -> messages.ArrayPayload:
    """Encode a one-dimensional array as contiguous little-endian bytes."""

    array = np.asarray(values, dtype=dtype)
    if array.ndim != 1:
        raise ValueError("distributed array payloads must be one-dimensional")
    native_dtype = np.dtype(array.dtype).newbyteorder("=")
    try:
        proto_dtype = _DTYPE_TO_PROTO[native_dtype]
    except KeyError as error:
        raise ValueError(
            f"unsupported distributed array dtype: {array.dtype}"
        ) from error
    wire_dtype = _PROTO_TO_DTYPE[proto_dtype]
    wire_array = np.ascontiguousarray(array, dtype=wire_dtype)
    return messages.ArrayPayload(
        dtype=proto_dtype,
        length=wire_array.size,
        data=wire_array.tobytes(order="C"),
    )


def decode_array(payload: messages.ArrayPayload, *, copy: bool = False) -> np.ndarray:
    """Validate and decode a protocol array payload."""

    try:
        dtype = _PROTO_TO_DTYPE[payload.dtype]
    except KeyError as error:
        raise ValueError(
            "array payload has an unsupported or unspecified dtype"
        ) from error
    expected_bytes = int(payload.length) * dtype.itemsize
    if expected_bytes != len(payload.data):
        raise ValueError(
            "array payload byte length does not match its declared dtype and length"
        )
    result = np.frombuffer(payload.data, dtype=dtype, count=int(payload.length))
    return result.copy() if copy else result


def is_empty_payload(payload: messages.ArrayPayload) -> bool:
    """Return whether an optional array payload was omitted."""

    return (
        payload.dtype == messages.ARRAY_DTYPE_UNSPECIFIED
        and not payload.data
        and payload.length == 0
    )
