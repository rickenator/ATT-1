#!/usr/bin/env python3
"""
ATT-1 safetensors metadata scanner (Milestone 46).

Parses the JSON header of a .safetensors file and reports tensor inventory
without reading any tensor payload data.

Safetensors binary format:
  [0:8]            header_len  — little-endian uint64 byte length of JSON
  [8:8+header_len] JSON metadata
  [8+header_len:]  tensor data (not read by this scanner)

JSON header structure:
  {
    "__metadata__": { ... },         optional
    "tensor_name":  {
        "dtype":        "F32" | "BF16" | "F16" | "I32" | ...
        "shape":        [dim0, dim1, ...]
        "data_offsets": [begin, end]  relative to start of data region
    },
    ...
  }

Usage:
    # list all tensors
    python3 compiler/scan_safetensors.py model.safetensors

    # suppress per-tensor listing
    python3 compiler/scan_safetensors.py model.safetensors --no-tensors

    # validate required LLaMA keys (n_layers from --n-layers or --config)
    python3 compiler/scan_safetensors.py model.safetensors --check-llama --n-layers 2
    python3 compiler/scan_safetensors.py model.safetensors --check-llama \
        --config path/to/config.json

    # JSON output
    python3 compiler/scan_safetensors.py model.safetensors --json

Exit codes:
    0  scan ok (warnings may be printed to stderr)
    1  fatal scan error (file missing, truncated, bad JSON, offsets OOB)
    2  validation failure (missing required LLaMA tensors)
"""

import argparse
import json
import os
import struct
import sys

# ---------------------------------------------------------------------------
# dtype element sizes (bytes)
# ---------------------------------------------------------------------------

_ELEMENT_SIZES = {
    "F64":  8, "I64":  8,
    "F32":  4, "I32":  4, "U32": 4,
    "BF16": 2, "F16":  2, "I16": 2, "U16": 2,
    "I8":   1, "U8":   1, "BOOL": 1,
}

# ---------------------------------------------------------------------------
# Required tensor keys for a LLaMA-style model
# ---------------------------------------------------------------------------

_LLAMA_PER_LAYER_KEYS = [
    "model.layers.{L}.input_layernorm.weight",
    "model.layers.{L}.self_attn.q_proj.weight",
    "model.layers.{L}.self_attn.k_proj.weight",
    "model.layers.{L}.self_attn.v_proj.weight",
    "model.layers.{L}.self_attn.o_proj.weight",
    "model.layers.{L}.post_attention_layernorm.weight",
    "model.layers.{L}.mlp.gate_proj.weight",
    "model.layers.{L}.mlp.up_proj.weight",
    "model.layers.{L}.mlp.down_proj.weight",
]

_LLAMA_GLOBAL_KEYS = [
    "model.embed_tokens.weight",
    "model.norm.weight",
    "lm_head.weight",
]

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class ScanError(Exception):
    """Fatal error during safetensors header scan."""


def _product(shape):
    r = 1
    for d in shape:
        r *= d
    return r


def scan_safetensors(path: str) -> dict:
    """Parse the JSON metadata header of a .safetensors file.

    Returns a dict with keys:
        file_size     int
        header_len    int
        data_offset   int   (= 8 + header_len)
        data_size     int   (= file_size - data_offset)
        metadata      dict  (__metadata__ entry, or {})
        tensors       list[dict]:
            name            str
            dtype           str
            shape           list[int]
            data_offsets    [int, int]  relative to data region start
            nbytes          int   (data_offsets[1] - data_offsets[0])
            element_size    int   (0 if dtype unrecognised)
            expected_nbytes int   (product(shape) * element_size, or 0)
        errors        list[str]  non-fatal issues discovered during scan

    Raises ScanError on any fatal problem:
        - file not found
        - file < 8 bytes
        - header_len extends beyond file
        - header JSON is not valid UTF-8 or not valid JSON
        - any tensor's data_offsets[1] exceeds data_size
    """
    if not os.path.exists(path):
        raise ScanError(f"file not found: {path}")

    file_size = os.path.getsize(path)
    if file_size < 8:
        raise ScanError(
            f"file too small: {file_size} byte(s) "
            "(minimum 8 bytes required for header length prefix)"
        )

    with open(path, "rb") as fh:
        raw_len = fh.read(8)
        if len(raw_len) < 8:
            raise ScanError("could not read 8-byte header length prefix")

        header_len = struct.unpack("<Q", raw_len)[0]
        if header_len == 0:
            raise ScanError("header_len is zero — not a valid safetensors file")

        if 8 + header_len > file_size:
            raise ScanError(
                f"header_len ({header_len}) extends beyond file size "
                f"({file_size}): file is truncated or corrupt"
            )

        raw_header = fh.read(header_len)

    if len(raw_header) < header_len:
        raise ScanError(
            f"could not read full JSON header: got {len(raw_header)} bytes, "
            f"expected {header_len}"
        )

    try:
        header = json.loads(raw_header.decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise ScanError(f"JSON header is not valid UTF-8: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ScanError(f"JSON header parse error: {exc}") from exc

    if not isinstance(header, dict):
        raise ScanError("JSON header root is not an object")

    data_offset = 8 + header_len
    data_size   = file_size - data_offset
    metadata    = {}
    tensors     = []
    errors      = []

    for name, entry in header.items():
        if name == "__metadata__":
            if isinstance(entry, dict):
                metadata = entry
            continue

        if not isinstance(entry, dict):
            errors.append(f"tensor {name!r}: entry is not an object — skipped")
            continue

        # --- dtype ---
        dtype = entry.get("dtype")
        if dtype is None:
            errors.append(f"tensor {name!r}: missing 'dtype'")
            dtype = "?"
        elif not isinstance(dtype, str):
            errors.append(f"tensor {name!r}: 'dtype' is not a string")
            dtype = "?"

        # --- shape ---
        shape = entry.get("shape")
        if shape is None:
            errors.append(f"tensor {name!r}: missing 'shape'")
            shape = []
        elif not isinstance(shape, list):
            errors.append(f"tensor {name!r}: 'shape' is not a list")
            shape = []

        # --- data_offsets ---
        raw_offsets = entry.get("data_offsets")
        if raw_offsets is None:
            errors.append(f"tensor {name!r}: missing 'data_offsets'")
            begin, end = 0, 0
        elif not isinstance(raw_offsets, list) or len(raw_offsets) != 2:
            errors.append(
                f"tensor {name!r}: 'data_offsets' must be a 2-element list"
            )
            begin, end = 0, 0
        else:
            begin, end = int(raw_offsets[0]), int(raw_offsets[1])

        if begin > end:
            errors.append(
                f"tensor {name!r}: data_offsets[0] ({begin}) > "
                f"data_offsets[1] ({end})"
            )

        # Out-of-bounds offset is fatal — we cannot trust the file layout.
        if end > data_size:
            raise ScanError(
                f"tensor {name!r}: data_offsets[1] ({end}) exceeds "
                f"data region size ({data_size}) — "
                f"file is truncated or offsets are corrupt"
            )

        nbytes       = end - begin
        element_size = _ELEMENT_SIZES.get(dtype, 0)
        expected     = _product(shape) * element_size if element_size else 0

        if element_size and shape and expected != nbytes:
            errors.append(
                f"tensor {name!r}: declared {nbytes} bytes but "
                f"shape {shape} × {dtype} ({element_size} B/elem) "
                f"implies {expected} bytes"
            )

        tensors.append({
            "name":            name,
            "dtype":           dtype,
            "shape":           shape,
            "data_offsets":    [begin, end],
            "nbytes":          nbytes,
            "element_size":    element_size,
            "expected_nbytes": expected,
        })

    # Best-effort overlap check (sort by start offset)
    sorted_t = sorted(tensors, key=lambda t: t["data_offsets"][0])
    for i in range(len(sorted_t) - 1):
        a, b = sorted_t[i], sorted_t[i + 1]
        if a["data_offsets"][1] > b["data_offsets"][0]:
            errors.append(
                f"tensor {a['name']!r} and {b['name']!r}: "
                f"overlapping data regions "
                f"([{a['data_offsets'][0]},{a['data_offsets'][1]}] vs "
                f"[{b['data_offsets'][0]},{b['data_offsets'][1]}])"
            )

    return {
        "file_size":   file_size,
        "header_len":  header_len,
        "data_offset": data_offset,
        "data_size":   data_size,
        "metadata":    metadata,
        "tensors":     tensors,
        "errors":      errors,
    }


def llama_required_keys(n_layers: int) -> list:
    """Return the full list of required tensor keys for an n_layers LLaMA model."""
    keys = list(_LLAMA_GLOBAL_KEYS)
    for layer in range(n_layers):
        for tmpl in _LLAMA_PER_LAYER_KEYS:
            keys.append(tmpl.format(L=layer))
    return keys


def check_llama_tensors(scan: dict, n_layers: int,
                        allow_tied_lm_head: bool = True) -> list:
    """Return list of missing required LLaMA tensor keys.

    If allow_tied_lm_head is True and 'lm_head.weight' is absent but
    'model.embed_tokens.weight' is present, lm_head is considered satisfied
    (tied-weight convention).
    """
    present  = {t["name"] for t in scan["tensors"]}
    required = llama_required_keys(n_layers)
    missing  = []
    for key in required:
        if key not in present:
            if (key == "lm_head.weight" and allow_tied_lm_head
                    and "model.embed_tokens.weight" in present):
                continue  # tied weights — acceptable
            missing.append(key)
    return missing


def format_scan_report(scan: dict, show_tensors: bool = True) -> str:
    """Render a human-readable scan report."""
    lines = []
    lines.append(f"file_size={scan['file_size']}")
    lines.append(f"header_len={scan['header_len']}")
    lines.append(f"data_offset={scan['data_offset']}")
    lines.append(f"data_size={scan['data_size']}")
    lines.append(f"tensor_count={len(scan['tensors'])}")
    lines.append(f"scan_errors={len(scan['errors'])}")

    if show_tensors and scan["tensors"]:
        lines.append("")
        for t in scan["tensors"]:
            shape_str = "[" + ",".join(str(d) for d in t["shape"]) + "]"
            lines.append(
                f"  {t['name']:<64} {t['dtype']:<6} {shape_str:<18} "
                f"data=[{t['data_offsets'][0]},{t['data_offsets'][1]}]"
            )

    if scan["errors"]:
        lines.append("")
        for err in scan["errors"]:
            lines.append(f"  error: {err}")

    lines.append("")
    if scan["errors"]:
        lines.append(f"scan: {len(scan['errors'])} error(s)")
    else:
        lines.append("scan: ok")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="ATT-1 safetensors metadata scanner (M46)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("path", help="Path to .safetensors file")
    parser.add_argument(
        "--check-llama", action="store_true",
        help="Validate that all required LLaMA tensor keys are present",
    )
    parser.add_argument(
        "--n-layers", type=int, default=None, metavar="N",
        help="Number of transformer layers (required with --check-llama unless "
             "--config is given)",
    )
    parser.add_argument(
        "--config", metavar="PATH", default=None,
        help="Path to config.json; used to derive --n-layers if not specified",
    )
    parser.add_argument(
        "--json", action="store_true", dest="output_json",
        help="Write JSON to stdout instead of human-readable text",
    )
    parser.add_argument(
        "--no-tensors", action="store_true",
        help="Suppress the per-tensor listing in text output",
    )
    args = parser.parse_args()

    # Derive n_layers from config.json if --config given and --n-layers not
    n_layers = args.n_layers
    if args.config is not None and n_layers is None:
        try:
            with open(args.config, "r", encoding="utf-8") as fh:
                cfg = json.load(fh)
            n_layers = (
                cfg.get("num_hidden_layers")
                or cfg.get("n_layer")
                or cfg.get("num_layers")
            )
            if n_layers is None:
                print(
                    "warning: could not derive n_layers from config "
                    f"(tried num_hidden_layers, n_layer, num_layers)",
                    file=sys.stderr,
                )
        except OSError as exc:
            print(f"warning: could not open config: {exc}", file=sys.stderr)
        except json.JSONDecodeError as exc:
            print(f"warning: could not parse config JSON: {exc}", file=sys.stderr)

    # --- scan ---
    try:
        scan = scan_safetensors(args.path)
    except ScanError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)

    # --- optional LLaMA validation ---
    llama_missing = []
    if args.check_llama:
        if n_layers is None:
            print(
                "error: --check-llama requires --n-layers N or --config PATH",
                file=sys.stderr,
            )
            sys.exit(1)
        llama_missing = check_llama_tensors(scan, n_layers)

    # --- output ---
    if args.output_json:
        out = dict(scan)
        if args.check_llama:
            out["llama_check"] = {
                "n_layers":      n_layers,
                "missing_count": len(llama_missing),
                "missing":       llama_missing,
            }
        print(json.dumps(out, indent=2))
    else:
        print(format_scan_report(scan, show_tensors=not args.no_tensors))
        if args.check_llama:
            if llama_missing:
                print(f"llama_check: {len(llama_missing)} missing")
                for key in llama_missing:
                    print(f"  missing: {key}")
                print(f"llama_missing: {len(llama_missing)}")
            else:
                print("llama_check: ok")
                print("llama_missing: 0")

    if scan["errors"]:
        sys.exit(1)
    if llama_missing:
        sys.exit(2)


if __name__ == "__main__":
    main()
