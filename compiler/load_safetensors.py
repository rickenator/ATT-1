#!/usr/bin/env python3
"""
ATT-1 safetensors tensor reader (Milestone 47 / M67 BF16 coercion).

Reads tensor payload bytes from a .safetensors file after the M46 metadata
scanner validates the header.  Uses Python standard library only (struct
module); no numpy or external dependencies.

Supported source dtypes (M47 + M67):
    F32  — decoded directly as IEEE 754 float32.
    BF16 — coerced to F32 by zero-extending the low 16 mantissa bits.
           BF16 is the top 16 bits of F32; no information is manufactured.
    F16  — coerced to F32 via IEEE 754 half-precision conversion.
           Handles subnormals, infinities, and NaN correctly.

Returned TensorData.dtype always reflects the *source* dtype from the
safetensors header ("BF16", "F16", "F32").  Callers see F32 float values
regardless of source dtype.  Use TensorData.coerced=True to detect cases
where coercion was applied.

Usage:
    # load and display one tensor
    python3 compiler/load_safetensors.py model.safetensors \\
        --tensor model.embed_tokens.weight

    # assert dtype and shape while loading
    python3 compiler/load_safetensors.py model.safetensors \\
        --tensor model.embed_tokens.weight \\
        --expected-dtype F32 --expected-shape 16,8

    # validate all F32 tensors are finite (no NaN/Inf)
    python3 compiler/load_safetensors.py model.safetensors --check-values

    # load all readable tensors and show per-tensor stats (min/max/mean)
    python3 compiler/load_safetensors.py model.safetensors --summary

Exit codes:
    0  success
    1  scan or load error
    2  check-values: found NaN or Inf values
"""

import argparse
import math
import os
import struct
import sys
from collections import namedtuple

# ---------------------------------------------------------------------------
# Bootstrap: ensure scan_safetensors is importable when run from any cwd
# ---------------------------------------------------------------------------

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scan_safetensors import ScanError, scan_safetensors  # noqa: E402

# ---------------------------------------------------------------------------
# Public types and exceptions
# ---------------------------------------------------------------------------

TensorData = namedtuple("TensorData", ["name", "dtype", "shape", "values", "nbytes", "coerced"])
"""Immutable tensor payload returned by load_tensor().

    name     str           — tensor name as found in the safetensors header
    dtype    str           — source dtype string from safetensors, e.g. "BF16"
    shape    list[int]     — tensor dimensions
    values   tuple[float]  — flat decoded values, always F32 floats (row-major)
    nbytes   int           — raw byte count read from file (source size)
    coerced  bool          — True if source dtype was BF16 or F16 (coerced to F32)
"""

# Dtypes that load_tensor() can decode (F32 native; BF16/F16 coerced to F32).
_READABLE_DTYPES = frozenset({"F32", "BF16", "F16"})

# struct format character for native dtypes (little-endian unpack).
_STRUCT_FMT = {"F32": "f"}


# ---------------------------------------------------------------------------
# BF16 / F16 → F32 coercion helpers (Python standard library only)
# ---------------------------------------------------------------------------

def _coerce_bf16(raw: bytes) -> tuple:
    """Decode BF16 payload bytes to a tuple of Python floats.

    BF16 is identical to the top 16 bits of IEEE 754 float32.  Coercion:
    zero-extend each 2-byte little-endian BF16 to 4 bytes and reinterpret
    as float32.  This preserves sign, exponent, subnormals, inf, and NaN.
    No information is introduced or discarded.

    raw must have even length.
    """
    n      = len(raw) // 2
    result = []
    for i in range(n):
        b0, b1 = raw[2 * i], raw[2 * i + 1]
        # LE layout: b0 = bits[7:0], b1 = bits[15:8]
        # F32 LE: [bits7:0, bits15:8, bits23:16, bits31:24]
        # BF16 occupies the high half of F32, so prepend two zero bytes:
        f32_bytes = bytes([0, 0, b0, b1])
        result.append(struct.unpack("<f", f32_bytes)[0])
    return tuple(result)


def _coerce_f16(raw: bytes) -> tuple:
    """Decode F16 payload bytes to a tuple of Python floats.

    Implements IEEE 754 half-precision (5-bit exponent, 10-bit mantissa,
    bias 15).  Handles subnormals, zeroes, infinities, and NaN correctly.

    raw must have even length.
    """
    n      = len(raw) // 2
    result = []
    for i in range(n):
        h     = struct.unpack_from("<H", raw, 2 * i)[0]
        sign  = -1.0 if (h >> 15) else 1.0
        exp5  = (h >> 10) & 0x1F
        mant  = h & 0x3FF
        if exp5 == 0:
            # Zero or subnormal
            f = sign * (mant / 1024.0) * (2.0 ** -14)
        elif exp5 == 31:
            # Infinity or NaN
            f = float("nan") if mant != 0 else sign * float("inf")
        else:
            f = sign * (1.0 + mant / 1024.0) * (2.0 ** (exp5 - 15))
        result.append(f)
    return tuple(result)


class LoadError(Exception):
    """Fatal error during tensor payload read or validation."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _product(shape):
    r = 1
    for d in shape:
        r *= d
    return r


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_tensor(path: str, name: str,
                expected_dtype: str = None,
                expected_shape=None) -> TensorData:
    """Load and decode one tensor payload from a .safetensors file.

    Parameters:
        path            Path to .safetensors file.
        name            Tensor name as it appears in the JSON header.
        expected_dtype  If given, raise LoadError if actual dtype differs.
        expected_shape  If given (list or tuple), raise LoadError on mismatch.

    Returns:
        TensorData namedtuple: name, dtype, shape, values (tuple of float),
        nbytes.

    Raises:
        ScanError  — file cannot be parsed; propagated from scan_safetensors.
        LoadError  — tensor not found; dtype unsupported or mismatched;
                     shape mismatched; or payload read shorter than expected.
    """
    scan  = scan_safetensors(path)
    index = {t["name"]: t for t in scan["tensors"]}

    if name not in index:
        raise LoadError(f"tensor not found: {name!r}")

    td    = index[name]
    dtype = td["dtype"]

    # Caller-supplied dtype assertion.
    # When expected_dtype="F32" and the source is a coercible dtype (BF16/F16),
    # do not raise: the returned values are F32 regardless of source encoding.
    _coercible = frozenset({"BF16", "F16"})
    _dtype_ok  = (
        expected_dtype is None
        or dtype == expected_dtype
        or (expected_dtype == "F32" and dtype in _coercible)
    )
    if not _dtype_ok:
        raise LoadError(
            f"dtype mismatch for {name!r}: "
            f"expected {expected_dtype!r}, got {dtype!r}"
        )

    # Readability gate — F32 native; BF16/F16 coerced to F32 (M67).
    if dtype not in _READABLE_DTYPES:
        raise LoadError(
            f"unsupported dtype {dtype!r} for tensor {name!r} "
            f"(supported: {sorted(_READABLE_DTYPES)})"
        )

    # Caller-supplied shape assertion.
    if expected_shape is not None and list(expected_shape) != td["shape"]:
        raise LoadError(
            f"shape mismatch for {name!r}: "
            f"expected {list(expected_shape)}, got {td['shape']}"
        )

    begin, end   = td["data_offsets"]
    nbytes       = end - begin
    file_offset  = scan["data_offset"] + begin

    with open(path, "rb") as fh:
        fh.seek(file_offset)
        raw = fh.read(nbytes)

    if len(raw) != nbytes:
        raise LoadError(
            f"truncated payload for {name!r}: "
            f"expected {nbytes} bytes at file offset {file_offset}, "
            f"got {len(raw)}"
        )

    # Decode / coerce to F32 floats.
    if dtype == "F32":
        n      = nbytes // 4
        values = struct.unpack(f"<{n}f", raw)
        coerced = False
    elif dtype == "BF16":
        values  = _coerce_bf16(raw)
        coerced = True
    elif dtype == "F16":
        values  = _coerce_f16(raw)
        coerced = True
    else:
        # Should be unreachable given the _READABLE_DTYPES gate above.
        raise LoadError(f"internal: unhandled dtype {dtype!r}")  # pragma: no cover

    return TensorData(
        name=name,
        dtype=dtype,
        shape=td["shape"],
        values=values,
        nbytes=nbytes,
        coerced=coerced,
    )


def load_all(path: str):
    """Load all F32/BF16/F16 tensors from a .safetensors file.

    BF16 and F16 tensors are coerced to F32 values automatically (M67).

    Returns:
        (loaded, skipped, errors) where:
            loaded   list[TensorData] — successfully decoded tensors
            skipped  list[str]        — names of tensors with unsupported dtype
            errors   list[str]        — error messages for any read failures
    """
    scan    = scan_safetensors(path)
    loaded  = []
    skipped = []
    errors  = []

    for t in scan["tensors"]:
        if t["dtype"] not in _READABLE_DTYPES:
            skipped.append(t["name"])
            continue
        try:
            td = load_tensor(path, t["name"])
            loaded.append(td)
        except (LoadError, ScanError) as exc:
            errors.append(f"{t['name']}: {exc}")

    return loaded, skipped, errors


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def format_tensor_report(td: TensorData, max_values: int = 8) -> str:
    """Render a human-readable report for one loaded tensor."""
    shape_str  = "[" + ",".join(str(d) for d in td.shape) + "]"
    n          = len(td.values)
    show       = td.values[:max_values]
    vals_str   = " ".join(f"{v:.6g}" for v in show)
    dtype_disp = f"{td.dtype}->F32" if td.coerced else td.dtype
    if n > max_values:
        vals_str += f" ... ({n - max_values} more)"
    lines = [
        f"tensor: {td.name}",
        f"dtype: {dtype_disp}",
        f"shape: {shape_str}",
        f"nbytes: {td.nbytes}",
        f"elements: {n}",
        f"values[0:{min(n, max_values)}]: {vals_str}",
        "load: ok",
    ]
    return "\n".join(lines)


def format_summary(loaded, skipped, errors) -> str:
    """Render a summary report from load_all() results."""
    lines = [
        f"loaded_count={len(loaded)}",
        f"skipped_count={len(skipped)}",
        f"load_errors={len(errors)}",
    ]
    if loaded:
        lines.append("")
        for td in loaded:
            shape_str = "[" + ",".join(str(d) for d in td.shape) + "]"
            n  = len(td.values)
            lo = min(td.values) if n else 0.0
            hi = max(td.values) if n else 0.0
            mn = sum(td.values) / n if n else 0.0
            lines.append(
                f"  {td.name:<64} {td.dtype:<6} {shape_str:<18} "
                f"elements={n:<6} min={lo:.4g}  max={hi:.4g}  mean={mn:.4g}"
            )
    if skipped:
        lines.append("")
        for name in skipped:
            lines.append(f"  skipped (unsupported dtype): {name}")
    if errors:
        lines.append("")
        for msg in errors:
            lines.append(f"  load_error: {msg}")
    lines.append("")
    if errors:
        lines.append(f"summary: {len(errors)} error(s)")
    else:
        lines.append("summary: ok")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="ATT-1 safetensors tensor reader (M47)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("path", help="Path to .safetensors file")
    parser.add_argument(
        "--tensor", metavar="NAME", default=None,
        help="Load and display one tensor by name",
    )
    parser.add_argument(
        "--expected-dtype", metavar="DTYPE", default=None,
        help="Assert the tensor has this dtype (used with --tensor)",
    )
    parser.add_argument(
        "--expected-shape", metavar="DIM,...", default=None,
        help="Assert the tensor has this shape, e.g. 16,8 (used with --tensor)",
    )
    parser.add_argument(
        "--summary", action="store_true",
        help="Load all readable tensors and show per-tensor stats",
    )
    parser.add_argument(
        "--check-values", action="store_true",
        help="Load all F32 tensors and verify no NaN or Inf values",
    )
    args = parser.parse_args()

    # Parse --expected-shape
    expected_shape = None
    if args.expected_shape is not None:
        try:
            expected_shape = [int(d) for d in args.expected_shape.split(",")]
        except ValueError:
            print(
                f"error: --expected-shape must be comma-separated integers, "
                f"got {args.expected_shape!r}",
                file=sys.stderr,
            )
            sys.exit(1)

    # --- --tensor mode ---
    if args.tensor is not None:
        try:
            td = load_tensor(args.path, args.tensor,
                             expected_dtype=args.expected_dtype,
                             expected_shape=expected_shape)
        except ScanError as exc:
            print(f"scan error: {exc}", file=sys.stderr)
            sys.exit(1)
        except LoadError as exc:
            print(f"load error: {exc}", file=sys.stderr)
            sys.exit(1)
        print(format_tensor_report(td))
        return

    # --- --check-values mode ---
    if args.check_values:
        try:
            loaded, skipped, errors = load_all(args.path)
        except ScanError as exc:
            print(f"scan error: {exc}", file=sys.stderr)
            sys.exit(1)
        n_nan = sum(1 for td in loaded for v in td.values if math.isnan(v))
        n_inf = sum(1 for td in loaded for v in td.values if math.isinf(v))
        print(f"values_ok={len(loaded)}")
        print(f"values_skipped={len(skipped)}")
        print(f"values_nan={n_nan}")
        print(f"values_inf={n_inf}")
        if errors:
            for msg in errors:
                print(f"  load_error: {msg}")
        print()
        if errors or n_nan or n_inf:
            print(
                f"check_values: {len(errors)} load error(s), "
                f"{n_nan} NaN(s), {n_inf} Inf(s)"
            )
            sys.exit(2)
        print("check_values: ok")
        return

    # --- default / --summary mode ---
    try:
        loaded, skipped, errors = load_all(args.path)
    except ScanError as exc:
        print(f"scan error: {exc}", file=sys.stderr)
        sys.exit(1)
    print(format_summary(loaded, skipped, errors))
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
