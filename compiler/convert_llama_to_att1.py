#!/usr/bin/env python3
"""
ATT-1 LLaMA converter skeleton (Milestone 31).

Validates a LLaMA-style model directory layout, resolves architecture fields
to the ATT-1 config, and prints the planned conversion without loading weights.

Usage:
    python3 compiler/convert_llama_to_att1.py --model-dir PATH [--output PATH] [--rope-theta FLOAT]

Exit codes:
    0  plan printed (stub only — no weights converted yet)
    1  config missing or invalid
    2  architecture not supported
"""

import argparse
import json
import math
import os
import sys

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


def load_config(model_dir: str) -> dict:
    path = os.path.join(model_dir, "config.json")
    if not os.path.exists(path):
        print(f"error: config.json not found in {model_dir!r}", file=sys.stderr)
        sys.exit(1)
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as exc:
        print(f"error: config.json is invalid JSON: {exc}", file=sys.stderr)
        sys.exit(1)


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
    att1["n_tiles"]      = 1   # default single-tile; cluster sharding is runtime
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


def print_plan(arch: str, att1: dict, output_path: str | None) -> None:
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
    n_tensors = 2 + att1["n_layers"] * 9  # embed + 9/layer + output_norm + output_weight
    print(f"  planned tensors: {n_tensors}")
    if output_path:
        print(f"  output path   : {output_path}")
    else:
        print("  output path   : (not specified — dry run)")
    print()
    print("status: STUB ONLY — weight loading and .att1 emission not yet implemented")
    print("        To complete conversion, provide safetensors/bin weight files.")


# ---------------------------------------------------------------------------
# Tensor name plan (for documentation / future use)
# ---------------------------------------------------------------------------

def tensor_name_plan(att1: dict) -> list[tuple[str, list[int]]]:
    """Return the ordered list of (att1_name, shape) for the planned model."""
    d  = att1["d_model"]
    v  = att1["vocab_size"]
    ff = att1["d_ff"]
    plan = [("token_embedding", [v, d])]
    for layer in range(att1["n_layers"]):
        prefix = f"L{layer}"
        plan += [
            (f"{prefix}.attention_norm", [d]),
            (f"{prefix}.wq",             [d, d]),
            (f"{prefix}.wk",             [d, d]),
            (f"{prefix}.wv",             [d, d]),
            (f"{prefix}.wo",             [d, d]),
            (f"{prefix}.ffn_norm",       [d]),
            (f"{prefix}.w_gate",         [ff, d]),
            (f"{prefix}.w_up",           [ff, d]),
            (f"{prefix}.w_down",         [d, ff]),
        ]
    plan += [
        ("output_norm",   [d]),
        ("output_weight", [v, d]),
    ]
    return plan


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a LLaMA-style model directory to ATT-1 format (skeleton)."
    )
    parser.add_argument("--model-dir", required=True,
                        help="Path to model directory containing config.json")
    parser.add_argument("--output", default=None,
                        help="Output .att1 file path (optional; dry run if omitted)")
    parser.add_argument("--rope-theta", type=float, default=None,
                        help="Override rope_theta (default: read from config or 10000.0)")
    parser.add_argument("--show-tensors", action="store_true",
                        help="Print planned tensor names and shapes")
    args = parser.parse_args()

    if not os.path.isdir(args.model_dir):
        print(f"error: model directory not found: {args.model_dir!r}", file=sys.stderr)
        sys.exit(1)

    cfg  = load_config(args.model_dir)
    arch = check_arch(cfg)
    att1 = resolve_att1_config(cfg, args.rope_theta)

    errors = validate_att1_config(att1)
    if errors:
        print("error: ATT-1 config validation failed:", file=sys.stderr)
        for e in errors:
            print(f"  {e}", file=sys.stderr)
        sys.exit(1)

    print_plan(arch, att1, args.output)

    if args.show_tensors:
        print()
        print("planned tensor layout:")
        for name, shape in tensor_name_plan(att1):
            shape_str = " × ".join(str(s) for s in shape)
            print(f"  {name:<40} {shape_str}")


if __name__ == "__main__":
    main()
