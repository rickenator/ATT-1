#!/usr/bin/env python3
"""
ATT-1 LLaMA converter (Milestone 32 — deterministic stub emitter,
                        Milestone 43 — shard metadata plan report).

Validates a LLaMA-style config, resolves architecture fields to the ATT-1
config schema, and emits a deterministic synthetic `.att1` model artifact
compatible with the C model loader, att1-inspect, and att1-bench.

No safetensors or real weight loading is performed; all tensor values are
deterministic synthetic floats derived from the validated config.

Usage:
    # dry run — validate and print plan only
    python3 compiler/convert_llama_to_att1.py --model-dir PATH

    # emit deterministic stub .att1 artifact
    python3 compiler/convert_llama_to_att1.py --model-dir PATH --output OUT.att1

    # short-form aliases
    python3 compiler/convert_llama_to_att1.py --config PATH/config.json --out OUT.att1

    # emit artifact with shard metadata (2 tiles)
    python3 compiler/convert_llama_to_att1.py --config PATH/config.json \\
        --tiles 2 --shard-meta --out OUT.att1

    # print human-readable shard plan report
    python3 compiler/convert_llama_to_att1.py --config PATH/config.json \\
        --tiles 2 --shard-meta --report

    # emit JSON report to file
    python3 compiler/convert_llama_to_att1.py --config PATH/config.json \\
        --tiles 2 --shard-meta --report-json PATH/report.json

    # manual validation sequence after emission
    ./build/att1-inspect OUT.att1
    ./build/att1-bench --model OUT.att1 --prompt hello --tokens 4 \\
        --mode single --backend cpu-f32

Exit codes:
    0  success
    1  config missing, invalid, or failed validation
    2  architecture not supported
"""

import argparse
import json
import math
import os
import struct
import sys

# ---------------------------------------------------------------------------
# .att1 binary format constants (must match C header att1_model.h)
# ---------------------------------------------------------------------------

_MAGIC        = b"ATT1MODL"
_VERSION      = 1
_HEADER_SIZE  = 80
_CONFIG_SIZE  = 36   # 9 × uint32 LE
_DESC_SIZE    = 128
_DTYPE_F32    = 1
_SHARD_REC_SZ = 120  # ATT1_SHARD_META_RECORD_SIZE

# ---------------------------------------------------------------------------
# Supported architectures
# ---------------------------------------------------------------------------

SUPPORTED_ARCH = {"llama", "mistral"}

# ---------------------------------------------------------------------------
# Field aliases: maps ATT-1 name -> list of accepted config.json keys
# ---------------------------------------------------------------------------

FIELD_ALIASES = {
    "vocab_size":   ["vocab_size"],
    "n_layers":     ["n_layers", "num_hidden_layers"],
    "n_heads":      ["n_heads", "num_attention_heads"],
    "d_model":      ["d_model", "hidden_size"],
    "d_ff":         ["d_ff", "intermediate_size"],
    "max_seq_len":  ["max_seq_len", "max_position_embeddings"],
}

# Fields present in rope config (optional)
ROPE_THETA_KEYS = ["rope_theta"]
ROPE_DIM_KEYS   = ["rope_dim", "partial_rotary_factor"]

DEFAULT_ROPE_THETA = 10000.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve(cfg: dict, aliases: list[str], *, required: bool = True):
    """Return the first matching value from cfg, or None."""
    for key in aliases:
        if key in cfg:
            val = cfg[key]
            if isinstance(val, float) and val == math.floor(val):
                val = int(val)
            return val
    if required:
        return None
    return None


def check_arch(cfg: dict) -> str:
    arch = cfg.get("model_type", "").lower()
    if arch not in SUPPORTED_ARCH:
        if arch:
            print(
                f"unsupported: model_type={arch!r} is not in {sorted(SUPPORTED_ARCH)}",
                file=sys.stderr,
            )
        else:
            print(
                "unsupported: config.json has no 'model_type' field",
                file=sys.stderr,
            )
        sys.exit(2)
    return arch


def resolve_att1_config(cfg: dict, rope_theta_override: float | None) -> dict:
    """Map source config fields to ATT-1 config.  Exits on missing required fields."""
    att1 = {}
    missing = []

    for att1_key, aliases in FIELD_ALIASES.items():
        val = _resolve(cfg, aliases)
        if val is None:
            missing.append(f"{att1_key} (looked for: {aliases})")
        else:
            att1[att1_key] = int(val)

    if missing:
        print("error: required fields missing from config.json:", file=sys.stderr)
        for m in missing:
            print(f"  {m}", file=sys.stderr)
        sys.exit(1)

    # --- rope_theta ---
    if rope_theta_override is not None:
        att1["rope_theta"] = rope_theta_override
    else:
        rope_theta = _resolve(cfg, ROPE_THETA_KEYS, required=False)
        att1["rope_theta"] = float(rope_theta) if rope_theta is not None else DEFAULT_ROPE_THETA

    # --- rope_dim ---
    # If the source config has a partial_rotary_factor, derive as fraction of head_dim.
    rope_dim_raw = _resolve(cfg, ROPE_DIM_KEYS, required=False)
    head_dim = att1["d_model"] // att1["n_heads"]
    if rope_dim_raw is None:
        att1["rope_dim"] = head_dim  # full RoPE
    elif isinstance(rope_dim_raw, float) and rope_dim_raw <= 1.0:
        att1["rope_dim"] = max(1, int(math.ceil(head_dim * rope_dim_raw)))
    else:
        att1["rope_dim"] = int(rope_dim_raw)

    # --- derived fields ---
    att1["n_tiles"]      = 1   # default single-tile; may be overridden by --tiles
    att1["shard_count"]  = 0

    return att1


def validate_att1_config(att1: dict) -> list[str]:
    errors = []
    d_model  = att1["d_model"]
    n_heads  = att1["n_heads"]
    rope_dim = att1["rope_dim"]

    if d_model % n_heads != 0:
        errors.append(
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        )
    head_dim = d_model // n_heads
    if rope_dim > head_dim:
        errors.append(
            f"rope_dim ({rope_dim}) must be <= head_dim ({head_dim})"
        )
    if rope_dim % 2 != 0:
        errors.append(f"rope_dim ({rope_dim}) must be even")

    for key in ("vocab_size", "n_layers", "n_heads", "d_model", "d_ff", "max_seq_len"):
        if att1.get(key, 0) <= 0:
            errors.append(f"{key} must be > 0 (got {att1.get(key)!r})")

    return errors


def print_plan(arch: str, att1: dict, output_path: str | None,
               emit_shard_meta: bool = False) -> None:
    print("ATT-1 conversion plan")
    print(f"  source arch   : {arch}")
    print(f"  vocab_size    : {att1['vocab_size']}")
    print(f"  n_layers      : {att1['n_layers']}")
    print(f"  n_heads       : {att1['n_heads']}")
    print(f"  d_model       : {att1['d_model']}")
    print(f"  d_ff          : {att1['d_ff']}")
    print(f"  max_seq_len   : {att1['max_seq_len']}")
    print(f"  rope_dim      : {att1['rope_dim']}")
    print(f"  rope_theta    : {att1['rope_theta']}")
    print(f"  head_dim      : {att1['d_model'] // att1['n_heads']}")
    print(f"  n_tiles       : {att1['n_tiles']}")
    print(f"  shard_meta    : {'yes' if emit_shard_meta else 'no'}")
    n_tensors = 3 + att1["n_layers"] * 9  # tok_embed + 9/layer + output_norm + output_weight
    print(f"  planned tensors: {n_tensors}")
    if output_path:
        print(f"  output path   : {output_path}")
    else:
        print("  output path   : (not specified — dry run)")


# ---------------------------------------------------------------------------
# .att1 binary emission helpers
# ---------------------------------------------------------------------------

def _synthetic_values(tensor_index: int, count: int) -> list[float]:
    """Deterministic per-element floats that are unique per tensor."""
    base = (tensor_index + 1) * 0.01
    return [base + (i * 0.001) for i in range(count)]


def _make_tensor(name: str, shape: list[int], tensor_index: int) -> dict:
    count = 1
    for dim in shape:
        count *= dim
    data = b"".join(
        struct.pack("<f", v) for v in _synthetic_values(tensor_index, count)
    )
    return {"name": name, "shape": shape, "data": data}


def _descriptor(tensor: dict, offset: int) -> bytes:
    name_bytes = tensor["name"].encode("ascii")
    if len(name_bytes) >= 64:
        raise ValueError(f"tensor name too long: {tensor['name']!r}")
    name_padded = name_bytes + b"\x00" * (64 - len(name_bytes))
    shape = list(tensor["shape"]) + [1] * (4 - len(tensor["shape"]))
    return struct.pack(
        "<64sIIQQQQQQII",
        name_padded,
        _DTYPE_F32,
        len(tensor["shape"]),
        shape[0],
        shape[1],
        shape[2],
        shape[3],
        offset,
        len(tensor["data"]),
        0,  # shard_id
        0,  # flags
    )


# ---------------------------------------------------------------------------
# Shard metadata emission helpers (opt-in via --shard-meta)
# ---------------------------------------------------------------------------

def _layer_tile_assignment(n_layers: int, n_tiles: int) -> list[int]:
    """
    Return a list of tile_id[layer] using the same ceiling-division as
    att1_shard_plan_build() in src/shard.c, so the metadata plan produces
    the same layer→tile mapping as the runtime plan.
    """
    assignment: list[int] = []
    next_layer = 0
    for tile in range(n_tiles):
        remaining_layers = n_layers - next_layer
        remaining_tiles  = n_tiles  - tile
        count = (remaining_layers + remaining_tiles - 1) // remaining_tiles
        assignment.extend([tile] * count)
        next_layer += count
    return assignment


def _shard_record_bytes(
    tensor_index: int,
    shape:        list[int],
    tile_id:      int,
    byte_offset:  int,
) -> bytes:
    """Build one 120-byte shard metadata record."""
    ps = list(shape) + [1] * (4 - len(shape))
    return struct.pack(
        "<IIQ4QIIII8IIIIIQ",
        tensor_index,                    # tensor_id
        tile_id,
        byte_offset,
        ps[0], ps[1], ps[2], ps[3],      # shape[4]
        _DTYPE_F32,                       # dtype
        0,                                # quantization (none)
        tile_id,                          # owner_aimu
        0,                                # replication_policy (none)
        0, 0, 0, 0, 0, 0, 0, 0,          # dependency_graph[8]
        0,                                # allowed_ops
        0,                                # routing_requirements
        0,                                # reduction_behavior (none)
        0,                                # _reserved
        0,                                # checksum
    )


def _build_shard_meta_blob(att1: dict, tensors: list[dict]) -> bytes:
    """
    Build the raw shard metadata section bytes.

    Layer tensors ("layers.L.*") are assigned via ceiling-division identical
    to att1_shard_plan_build().  Non-layer tensors map to tile 0.
    """
    n_layers = att1["n_layers"]
    n_tiles  = att1.get("n_tiles", 1)
    layer_assignment = _layer_tile_assignment(n_layers, n_tiles)

    blob        = bytearray()
    byte_offset = 0
    for i, t in enumerate(tensors):
        name   = t["name"]
        shape  = t["shape"]
        nbytes = len(t["data"])
        tile_id = 0
        if name.startswith("layers."):
            rest = name[len("layers."):]
            dot  = rest.find(".")
            if dot > 0:
                try:
                    layer_id = int(rest[:dot])
                    if layer_id < n_layers:
                        tile_id = layer_assignment[layer_id]
                except ValueError:
                    pass
        blob += _shard_record_bytes(i, shape, tile_id, byte_offset)
        byte_offset += nbytes
    return bytes(blob)


# ---------------------------------------------------------------------------
# Shard plan report (Milestone 43)
# ---------------------------------------------------------------------------

def build_shard_plan_report(arch: str,
                             att1: dict,
                             emit_shard_meta: bool,
                             validation_errors: list[str]) -> dict:
    """
    Build a structured report dict describing the converter's shard plan.

    The report is independent of artifact emission; it can be requested
    on a dry run or alongside --output.
    """
    n_layers = att1["n_layers"]
    n_tiles  = att1.get("n_tiles", 1)
    n_tensors = 3 + n_layers * 9   # tok_embed + 9/layer + norm + output

    # --- layer → tile assignment (same algorithm as C runtime) ---
    if emit_shard_meta:
        layer_assignment = _layer_tile_assignment(n_layers, n_tiles)
    else:
        layer_assignment = []

    # --- per-tile layer lists ---
    tiles_layers: dict[int, list[int]] = {t: [] for t in range(n_tiles)}
    for layer_id, tile_id in enumerate(layer_assignment):
        tiles_layers[tile_id].append(layer_id)

    # --- tensor breakdown ---
    tensor_names_plan = tensor_name_plan(att1)

    def _tile_for(name: str) -> int | None:
        """Return the tile_id for a tensor name when shard_meta is active."""
        if not emit_shard_meta:
            return None
        if name.startswith("layers."):
            rest = name[len("layers."):]
            dot  = rest.find(".")
            if dot > 0:
                try:
                    layer_id = int(rest[:dot])
                    if layer_id < n_layers:
                        return layer_assignment[layer_id]
                except ValueError:
                    pass
        return 0   # non-layer tensors → tile 0

    # Build per-tensor records
    tensors_report = []
    for name, shape in tensor_names_plan:
        tile_id = _tile_for(name)
        elem_count = 1
        for s in shape:
            elem_count *= s
        tensors_report.append({
            "name":   name,
            "shape":  shape,
            "dtype":  "f32",
            "quant":  "none",
            "tile_id": tile_id,
            "bytes":  elem_count * 4,
        })

    # --- dtype / quant summary ---
    dtype_f32_count = n_tensors   # all tensors are f32 in stub emitter
    quant_none_count = n_tensors

    # --- per-tile summary ---
    tile_summaries = []
    for tile_id in range(n_tiles):
        owned = [t for t in tensors_report if t["tile_id"] == tile_id]
        layer_ids = tiles_layers[tile_id] if emit_shard_meta else []
        layer_range = (
            f"{layer_ids[0]}-{layer_ids[-1]}"
            if layer_ids
            else "none"
        )
        tile_summaries.append({
            "tile_id":      tile_id,
            "aimu_id":      tile_id,
            "tensor_count": len(owned),
            "layer_ids":    layer_ids,
            "layer_range":  layer_range,
        })

    # --- layer ownership summary ---
    layer_summaries = []
    for layer_id in range(n_layers):
        tile_id = layer_assignment[layer_id] if emit_shard_meta else None
        layer_summaries.append({
            "layer_id": layer_id,
            "tile_id":  tile_id,
        })

    return {
        "schema_version":   1,
        "source_arch":      arch,
        "config": {
            "vocab_size":  att1["vocab_size"],
            "n_layers":    att1["n_layers"],
            "n_heads":     att1["n_heads"],
            "d_model":     att1["d_model"],
            "d_ff":        att1["d_ff"],
            "max_seq_len": att1["max_seq_len"],
            "rope_dim":    att1["rope_dim"],
            "rope_theta":  att1["rope_theta"],
            "n_tiles":     att1["n_tiles"],
        },
        "tensor_count":     n_tensors,
        "shard_meta":       "present" if emit_shard_meta else "absent",
        "tile_count":       n_tiles,
        "aimu_count":       n_tiles,
        "dtype_f32_count":  dtype_f32_count,
        "dtype_q8_count":   0,
        "quant_none_count": quant_none_count,
        "tensors":          tensors_report,
        "tiles":            tile_summaries,
        "layers":           layer_summaries,
        "validation": {
            "status": "ok" if not validation_errors else "failed",
            "errors": validation_errors,
        },
    }


def format_report_text(report: dict) -> str:
    """Render the shard plan report as a human-readable text block."""
    lines: list[str] = []
    cfg = report["config"]

    lines.append("ATT-1 shard metadata plan report")
    lines.append(f"  schema_version  : {report['schema_version']}")
    lines.append(f"  source_arch     : {report['source_arch']}")
    lines.append("")
    lines.append("config:")
    lines.append(f"  vocab_size      : {cfg['vocab_size']}")
    lines.append(f"  n_layers        : {cfg['n_layers']}")
    lines.append(f"  n_heads         : {cfg['n_heads']}")
    lines.append(f"  d_model         : {cfg['d_model']}")
    lines.append(f"  d_ff            : {cfg['d_ff']}")
    lines.append(f"  max_seq_len     : {cfg['max_seq_len']}")
    lines.append(f"  rope_dim        : {cfg['rope_dim']}")
    lines.append(f"  rope_theta      : {cfg['rope_theta']}")
    lines.append(f"  n_tiles         : {cfg['n_tiles']}")
    lines.append("")
    lines.append("shard_meta:")
    lines.append(f"  present         : {report['shard_meta']}")
    lines.append(f"  tensor_count    : {report['tensor_count']}")
    lines.append(f"  tile_count      : {report['tile_count']}")
    lines.append(f"  aimu_count      : {report['aimu_count']}")
    lines.append(f"  dtype_f32       : {report['dtype_f32_count']}")
    lines.append(f"  dtype_q8        : {report['dtype_q8_count']}")
    lines.append(f"  quant_none      : {report['quant_none_count']}")
    lines.append("")
    lines.append("tile ownership:")
    for tile in report["tiles"]:
        lines.append(
            f"  tile[{tile['tile_id']}]  aimu={tile['aimu_id']}"
            f"  tensors={tile['tensor_count']}"
            f"  layers={tile['layer_range']}"
        )
    lines.append("")
    lines.append("layer assignment:")
    for layer in report["layers"]:
        tile_str = str(layer["tile_id"]) if layer["tile_id"] is not None else "n/a"
        lines.append(f"  layer[{layer['layer_id']}] → tile {tile_str}")
    lines.append("")
    lines.append("validation:")
    lines.append(f"  status          : {report['validation']['status']}")
    for err in report["validation"]["errors"]:
        lines.append(f"  error           : {err}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# .att1 binary builder
# ---------------------------------------------------------------------------

def build_att1_bytes(att1: dict, emit_shard_meta: bool = False) -> bytes:
    """
    Build a complete deterministic .att1 binary from a resolved ATT-1 config.

    Tensor names match exactly what the C model_view.c layer_name() and
    att1_model_view_* functions expect.

    When emit_shard_meta is True, a shard metadata section is appended and
    the header shard_metadata_offset/size fields are set accordingly.
    """
    d  = att1["d_model"]
    v  = att1["vocab_size"]
    ff = att1["d_ff"]

    tensors: list[dict] = []
    index = 0

    tensors.append(_make_tensor("tok_embeddings.weight", [v, d], index))
    index += 1

    for layer in range(att1["n_layers"]):
        pfx = f"layers.{layer}"
        tensors.append(_make_tensor(f"{pfx}.attention_norm.weight", [d],    index));     index += 1
        tensors.append(_make_tensor(f"{pfx}.attention.wq.weight",   [d, d], index));     index += 1
        tensors.append(_make_tensor(f"{pfx}.attention.wk.weight",   [d, d], index));     index += 1
        tensors.append(_make_tensor(f"{pfx}.attention.wv.weight",   [d, d], index));     index += 1
        tensors.append(_make_tensor(f"{pfx}.attention.wo.weight",   [d, d], index));     index += 1
        tensors.append(_make_tensor(f"{pfx}.ffn_norm.weight",       [d],    index));     index += 1
        tensors.append(_make_tensor(f"{pfx}.ffn.w_gate.weight",     [d, ff], index));    index += 1
        tensors.append(_make_tensor(f"{pfx}.ffn.w_up.weight",       [d, ff], index));    index += 1
        tensors.append(_make_tensor(f"{pfx}.ffn.w_down.weight",     [ff, d], index));    index += 1

    tensors.append(_make_tensor("output_norm.weight", [d],    index)); index += 1
    tensors.append(_make_tensor("output.weight",      [d, v], index))

    config_offset = _HEADER_SIZE
    desc_offset   = config_offset + _CONFIG_SIZE
    data_offset   = desc_offset + len(tensors) * _DESC_SIZE

    desc_blob = bytearray()
    data_blob = bytearray()
    byte_offset = 0
    for tensor in tensors:
        desc_blob += _descriptor(tensor, byte_offset)
        data_blob += tensor["data"]
        byte_offset += len(tensor["data"])

    shard_blob    = _build_shard_meta_blob(att1, tensors) if emit_shard_meta else b""
    shard_offset  = (data_offset + len(data_blob)) if emit_shard_meta else 0
    shard_size    = len(shard_blob)

    header = struct.pack(
        "<8sIIQQQQQQQQ",
        _MAGIC,
        _VERSION,
        _HEADER_SIZE,
        config_offset,
        _CONFIG_SIZE,
        desc_offset,
        len(tensors),
        data_offset,
        len(data_blob),
        shard_offset,
        shard_size,
    )
    config_blob = struct.pack(
        "<IIIIIIIII",
        att1["vocab_size"],
        att1["n_layers"],
        att1["n_heads"],
        att1["d_model"],
        att1["d_ff"],
        att1["max_seq_len"],
        att1["rope_dim"],
        att1["n_tiles"],
        att1["shard_count"],
    )
    return header + config_blob + bytes(desc_blob) + bytes(data_blob) + shard_blob


def emit_att1(att1: dict, output_path: str, emit_shard_meta: bool = False) -> None:
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    payload = build_att1_bytes(att1, emit_shard_meta=emit_shard_meta)
    with open(output_path, "wb") as f:
        f.write(payload)
    print(f"wrote {len(payload)} bytes → {output_path}")


# ---------------------------------------------------------------------------
# Tensor layout plan (informational)
# ---------------------------------------------------------------------------

def tensor_name_plan(att1: dict) -> list[tuple[str, list[int]]]:
    """Ordered list of (att1_name, shape) matching the C runtime expectations."""
    d  = att1["d_model"]
    v  = att1["vocab_size"]
    ff = att1["d_ff"]
    plan: list[tuple[str, list[int]]] = [("tok_embeddings.weight", [v, d])]
    for layer in range(att1["n_layers"]):
        pfx = f"layers.{layer}"
        plan += [
            (f"{pfx}.attention_norm.weight", [d]),
            (f"{pfx}.attention.wq.weight",   [d, d]),
            (f"{pfx}.attention.wk.weight",   [d, d]),
            (f"{pfx}.attention.wv.weight",   [d, d]),
            (f"{pfx}.attention.wo.weight",   [d, d]),
            (f"{pfx}.ffn_norm.weight",       [d]),
            (f"{pfx}.ffn.w_gate.weight",     [d, ff]),
            (f"{pfx}.ffn.w_up.weight",       [d, ff]),
            (f"{pfx}.ffn.w_down.weight",     [ff, d]),
        ]
    plan += [
        ("output_norm.weight", [d]),
        ("output.weight",      [d, v]),
    ]
    return plan


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a LLaMA-style model directory to ATT-1 format."
    )
    # model source — either a directory (containing config.json) or a file
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--model-dir", metavar="DIR",
                     help="Model directory containing config.json")
    src.add_argument("--config", metavar="FILE",
                     help="Path to config.json directly")

    # output — either --output or --out
    out = parser.add_mutually_exclusive_group()
    out.add_argument("--output", metavar="PATH",
                     help="Output .att1 file path (emit artifact)")
    out.add_argument("--out", metavar="PATH",
                     help="Alias for --output")

    parser.add_argument("--rope-theta", type=float, default=None,
                        help="Override rope_theta (default: from config or 10000.0)")
    parser.add_argument("--tiles", type=int, default=None, metavar="N",
                        help="Number of tiles for shard metadata (default: 1)")
    parser.add_argument("--shard-meta", action="store_true",
                        help="Emit optional shard metadata section")
    parser.add_argument("--report", action="store_true",
                        help="Print human-readable shard plan report to stdout")
    parser.add_argument("--report-json", metavar="PATH", default=None,
                        help="Write JSON shard plan report to PATH")
    parser.add_argument("--show-tensors", action="store_true",
                        help="Print planned tensor names and shapes")
    args = parser.parse_args()

    output_path = args.output or args.out

    # --- resolve config.json path ---
    if args.config:
        config_file = args.config
        if not os.path.isfile(config_file):
            print(f"error: config file not found: {config_file!r}", file=sys.stderr)
            sys.exit(1)
        model_dir = os.path.dirname(os.path.abspath(config_file))
    else:
        if not os.path.isdir(args.model_dir):
            print(f"error: model directory not found: {args.model_dir!r}", file=sys.stderr)
            sys.exit(1)
        model_dir = args.model_dir
        config_file = os.path.join(model_dir, "config.json")
        if not os.path.exists(config_file):
            print(f"error: config.json not found in {model_dir!r}", file=sys.stderr)
            sys.exit(1)

    try:
        with open(config_file, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except json.JSONDecodeError as exc:
        print(f"error: config.json is invalid JSON: {exc}", file=sys.stderr)
        sys.exit(1)

    arch = check_arch(cfg)
    att1 = resolve_att1_config(cfg, args.rope_theta)

    if args.tiles is not None:
        if args.tiles < 1:
            print("error: --tiles must be >= 1", file=sys.stderr)
            sys.exit(1)
        att1["n_tiles"] = args.tiles

    if args.shard_meta and args.tiles is None:
        # --shard-meta without --tiles defaults to 1 tile (still valid)
        pass

    errors = validate_att1_config(att1)
    if errors:
        print("error: ATT-1 config validation failed:", file=sys.stderr)
        for e in errors:
            print(f"  {e}", file=sys.stderr)
        sys.exit(1)

    print_plan(arch, att1, output_path, emit_shard_meta=args.shard_meta)

    if args.show_tensors:
        print()
        print("planned tensor layout:")
        for name, shape in tensor_name_plan(att1):
            shape_str = " \u00d7 ".join(str(s) for s in shape)
            print(f"  {name:<50} {shape_str}")

    # --- shard plan report ---
    if args.report or args.report_json:
        report = build_shard_plan_report(arch, att1, args.shard_meta, errors)
        if args.report:
            print()
            print(format_report_text(report))
        if args.report_json:
            rj_dir = os.path.dirname(args.report_json)
            if rj_dir:
                os.makedirs(rj_dir, exist_ok=True)
            with open(args.report_json, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
                f.write("\n")
            print(f"wrote report \u2192 {args.report_json}")

    if output_path:
        print()
        emit_att1(att1, output_path, emit_shard_meta=args.shard_meta)
        print("note: tensor data is deterministic synthetic values \u2014 not real weights")
        if args.shard_meta:
            n_tensors = 3 + att1["n_layers"] * 9
            print(f"note: shard metadata section: {n_tensors} records \u00d7 {_SHARD_REC_SZ} bytes")
    else:
        print()
        print("note: dry run \u2014 pass --output PATH to emit a .att1 artifact")


if __name__ == "__main__":
    main()
