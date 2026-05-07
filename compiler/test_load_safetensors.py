#!/usr/bin/env python3
"""
Self-contained tests for compiler/load_safetensors.py (Milestone 47).

Exercises all error paths using in-memory fixtures written to temp files.
Prints "self_test: ok" on success, per-failure lines + "self_test: FAILED"
on any failure.

Usage:
    python3 compiler/test_load_safetensors.py
"""

import json
import os
import struct
import sys
import tempfile

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

from load_safetensors import LoadError, TensorData, load_tensor  # noqa: E402
from scan_safetensors import ScanError                            # noqa: E402

_FIXTURE = os.path.join(_SCRIPT_DIR, "fixtures", "tiny_llama_2l.safetensors")

# ---------------------------------------------------------------------------
# Minimal fixture builder
# ---------------------------------------------------------------------------


def _product(shape):
    r = 1
    for d in shape:
        r *= d
    return r


def _make_safetensors(tensors, truncate_data_to=None):
    """Build a minimal safetensors binary.

    tensors: list of (name, dtype, shape, payload_bytes)
    truncate_data_to: if given, truncate the data region to this many bytes
                      (simulates a truncated/corrupt file).
    """
    offset  = 0
    entries = {}
    all_data = b""
    for name, dtype, shape, payload in tensors:
        nbytes = len(payload)
        entries[name] = {
            "dtype":        dtype,
            "shape":        shape,
            "data_offsets": [offset, offset + nbytes],
        }
        offset   += nbytes
        all_data += payload

    header = json.dumps(
        {"__metadata__": {}, **entries}, separators=(",", ":")
    ).encode("utf-8")
    prefix = struct.pack("<Q", len(header))
    data   = all_data if truncate_data_to is None else all_data[:truncate_data_to]
    return prefix + header + data


def _write_tmp(blob, suffix=".safetensors"):
    fd, path = tempfile.mkstemp(suffix=suffix)
    try:
        os.write(fd, blob)
    finally:
        os.close(fd)
    return path


# ---------------------------------------------------------------------------
# Assertion helpers
# ---------------------------------------------------------------------------

_failures = []


def _ok(label, cond, detail=""):
    if not cond:
        msg = f"FAIL: {label}"
        if detail:
            msg += f" — {detail}"
        _failures.append(msg)


def _raises(label, exc_type, fn):
    """Assert that fn() raises exc_type (or a tuple of types)."""
    try:
        fn()
        _failures.append(f"FAIL: {label} — expected {exc_type} but no exception raised")
    except exc_type:
        pass  # expected
    except Exception as exc:  # noqa: BLE001
        _failures.append(
            f"FAIL: {label} — unexpected {type(exc).__name__}: {exc}"
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def run_tests():
    # ------------------------------------------------------------------
    # 1. Token embedding tensor
    # ------------------------------------------------------------------
    td = load_tensor(_FIXTURE, "model.embed_tokens.weight")
    _ok("embed dtype is F32",     td.dtype  == "F32",    td.dtype)
    _ok("embed shape is [16,8]",  td.shape  == [16, 8],  str(td.shape))
    _ok("embed elements == 128",  len(td.values) == 128, str(len(td.values)))
    _ok("embed returns TensorData", isinstance(td, TensorData))

    # ------------------------------------------------------------------
    # 2. Attention projection tensor
    # ------------------------------------------------------------------
    td = load_tensor(_FIXTURE, "model.layers.0.self_attn.q_proj.weight")
    _ok("q_proj dtype is F32",    td.dtype == "F32")
    _ok("q_proj shape is [8,8]",  td.shape == [8, 8],  str(td.shape))
    _ok("q_proj elements == 64",  len(td.values) == 64)

    # ------------------------------------------------------------------
    # 3. FFN tensor
    # ------------------------------------------------------------------
    td = load_tensor(_FIXTURE, "model.layers.0.mlp.gate_proj.weight")
    _ok("gate_proj dtype is F32",    td.dtype == "F32")
    _ok("gate_proj shape is [16,8]", td.shape == [16, 8], str(td.shape))
    _ok("gate_proj elements == 128", len(td.values) == 128)

    # ------------------------------------------------------------------
    # 4. Final norm and lm_head
    # ------------------------------------------------------------------
    td = load_tensor(_FIXTURE, "model.norm.weight")
    _ok("norm shape is [8]",      td.shape == [8], str(td.shape))
    _ok("norm elements == 8",     len(td.values) == 8)
    td = load_tensor(_FIXTURE, "lm_head.weight")
    _ok("lm_head elements == 128", len(td.values) == 128)

    # ------------------------------------------------------------------
    # 5. expected_dtype mismatch → LoadError
    # ------------------------------------------------------------------
    _raises(
        "expected_dtype mismatch → LoadError",
        LoadError,
        lambda: load_tensor(_FIXTURE, "model.embed_tokens.weight",
                            expected_dtype="BF16"),
    )

    # ------------------------------------------------------------------
    # 6. expected_shape mismatch → LoadError
    # ------------------------------------------------------------------
    _raises(
        "expected_shape mismatch → LoadError",
        LoadError,
        lambda: load_tensor(_FIXTURE, "model.embed_tokens.weight",
                            expected_shape=[8, 16]),
    )

    # ------------------------------------------------------------------
    # 7. BF16 tensor → coerced to F32 (M67; no longer raises LoadError)
    # ------------------------------------------------------------------
    # Encode 4 BF16 values: pack each float as F32 and take top 2 bytes.
    bf16_values = [1.0, -0.5, 0.25, 2.0]
    bf16_payload = b"".join(struct.pack("<f", v)[2:4] for v in bf16_values)
    blob = _make_safetensors([("w", "BF16", [2, 2], bf16_payload)])
    path = _write_tmp(blob)
    try:
        td = load_tensor(path, "w")
        _ok("BF16 tensor coercion: load ok (no LoadError)",  True)
        _ok("BF16 tensor coercion: dtype field is 'BF16'",   td.dtype == "BF16")
        _ok("BF16 tensor coercion: coerced flag is True",    td.coerced)
        _ok("BF16 tensor coercion: 4 elements",              len(td.values) == 4)
        # BF16 truncates mantissa; check round-trip accuracy to within 1%.
        _ok("BF16 tensor coercion: values[0] ≈ 1.0",   abs(td.values[0] - 1.0)    < 0.01)
        _ok("BF16 tensor coercion: values[1] ≈ -0.5",  abs(td.values[1] - (-0.5)) < 0.01)
        _ok("BF16 tensor coercion: values[2] ≈ 0.25",  abs(td.values[2] - 0.25)   < 0.01)
        _ok("BF16 tensor coercion: values[3] ≈ 2.0",   abs(td.values[3] - 2.0)    < 0.01)
    finally:
        os.unlink(path)

    # ------------------------------------------------------------------
    # 8. Tensor name not found → LoadError
    # ------------------------------------------------------------------
    _raises(
        "missing tensor name → LoadError",
        LoadError,
        lambda: load_tensor(_FIXTURE, "does.not.exist"),
    )

    # ------------------------------------------------------------------
    # 9. Truncated file (header claims offsets beyond data region)
    #    → ScanError propagated through load_tensor
    # ------------------------------------------------------------------
    blob = _make_safetensors(
        [("w", "F32", [4], struct.pack("<4f", 1.0, 2.0, 3.0, 4.0))],
        truncate_data_to=8,  # only 8 of the 16 claimed bytes are present
    )
    path = _write_tmp(blob)
    try:
        _raises(
            "truncated file → ScanError",
            ScanError,
            lambda: load_tensor(path, "w"),
        )
    finally:
        os.unlink(path)

    # ------------------------------------------------------------------
    # 10. Known non-zero values decode correctly
    # ------------------------------------------------------------------
    known   = struct.pack("<4f", 1.0, -2.5, 0.125, 1e6)
    blob    = _make_safetensors([("x", "F32", [4], known)])
    path    = _write_tmp(blob)
    try:
        td = load_tensor(path, "x")
        _ok("known values[0] == 1.0",   abs(td.values[0] - 1.0)    < 1e-9)
        _ok("known values[1] == -2.5",  abs(td.values[1] - (-2.5)) < 1e-9)
        _ok("known values[2] == 0.125", abs(td.values[2] - 0.125)  < 1e-9)
        _ok("known values[3] == 1e6",   abs(td.values[3] - 1e6)    < 1.0)
        _ok("known values shape is [4]", td.shape == [4], str(td.shape))
        _ok("known values elements == 4", len(td.values) == 4)
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    run_tests()
    if _failures:
        for msg in _failures:
            print(msg)
        print(f"self_test: FAILED ({len(_failures)} failure(s))")
        sys.exit(1)
    print("self_test: ok")


if __name__ == "__main__":
    main()
