#!/usr/bin/env python3
"""
ATT-1 safetensors tensor reader (Milestone 47).

Reads tensor payload bytes from a .safetensors file after the M46 metadata
scanner validates the header.  Uses Python standard library only (struct
module); no numpy or external dependencies.

Supported source dtypes (M47):
    F32  — decoded as a flat tuple of Python floats (little-endian).

Unsupported source dtypes (BF16, F16, I32, etc.) raise LoadError.
Dtype coercion (BF16/F16 → F32) is deferred to M48.

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

TensorData = namedtuple("TensorData", ["name", "dtype", "shape", "values", "nbytes"])
"""Immutable tensor payload returned by load_tensor().

    name    str           — tensor name as found in the safetensors header
    dtype   str           — safetensors dtype string, e.g. "F32"
    shape   list[int]     — tensor dimensions
    values  tuple[float]  — flat decoded values (row-major / C order)
    nbytes  int           — raw byte count read from file
"""

# Dtypes that load_tensor() can decode in this milestone.
_READABLE_DTYPES = frozenset({"F32"})

# struct format character for each readable dtype (little-endian unpack).
_STRUCT_FMT = {"F32": "f"}


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

    # Caller-supplied dtype assertion (checked before readability gate).
    if expected_dtype is not None and dtype != expected_dtype:
        raise LoadError(
            f"dtype mismatch for {name!r}: "
            f"expected {expected_dtype!r}, got {dtype!r}"
        )

    # Readability gate — only F32 supported in M47.
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

    # Decode: F32 = 4-byte little-endian IEEE 754 single.
    fmt    = _STRUCT_FMT[dtype]
    n      = nbytes // 4
    values = struct.unpack(f"<{n}{fmt}", raw)

    return TensorData(
        name=name,
        dtype=dtype,
        shape=td["shape"],
        values=values,
        nbytes=nbytes,
    )


def load_all(path: str):
    """Load all F32 tensors from a .safetensors file.

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
    shape_str = "[" + ",".join(str(d) for d in td.shape) + "]"
    n         = len(td.values)
    show      = td.values[:max_values]
    vals_str  = " ".join(f"{v:.6g}" for v in show)
    if n > max_values:
        vals_str += f" ... ({n - max_values} more)"
    lines = [
        f"tensor: {td.name}",
        f"dtype: {td.dtype}",
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
