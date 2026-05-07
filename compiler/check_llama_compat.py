#!/usr/bin/env python3
"""
ATT-1 LLaMA compatibility scanner (Milestone 66).

Inspects a local model directory (config.json + model.safetensors + tokenizer
assets) and reports whether the model is ready for ATT-1 LLaMA conversion.

Reports:
- Architecture support (model_type in SUPPORTED_ARCH)
- Required config field presence and validity
- GQA/MQA detection (num_key_value_heads vs num_attention_heads)
- MoE detection (num_local_experts, etc.)
- BF16/F16 dtype detection in safetensors
- Required tensor names and shape validation
- Tokenizer asset availability and vocab_size cross-check
- Estimated f32 and q8 artifact sizes
- List of required converter changes if model cannot be converted as-is

Usage:
    # scan a local model directory (config.json + model.safetensors in <dir>)
    python3 compiler/check_llama_compat.py --model-dir ~/Models/SmolLM2-135M

    # explicit safetensors path (overrides <model-dir>/model.safetensors)
    python3 compiler/check_llama_compat.py \\
        --model-dir compiler/fixtures/m66_compat_fixture \\
        --safetensors compiler/fixtures/tiny_llama_2l.safetensors

    # skip safetensors check (useful when weights not yet downloaded)
    python3 compiler/check_llama_compat.py --model-dir PATH --no-tensors

    # JSON output
    python3 compiler/check_llama_compat.py --model-dir PATH --json

Exit codes:
    0  compat pass (model is ready for ATT-1 conversion as-is)
    1  fatal error (directory not found, config missing/invalid, parse error)
    2  compat fail (required converter changes needed before conversion)
"""

import argparse
import json
import math
import os
import struct
import sys

# ---------------------------------------------------------------------------
# Bootstrap: ensure companion scripts are importable when run from any cwd.
# ---------------------------------------------------------------------------

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scan_safetensors import (  # noqa: E402
    ScanError,
    check_llama_tensors,
    scan_safetensors,
)
from scan_tokenizer import TokenizerScanError, scan_tokenizer_dir  # noqa: E402

# ---------------------------------------------------------------------------
# Architecture constants (must stay in sync with convert_llama_to_att1.py)
# ---------------------------------------------------------------------------

SUPPORTED_ARCH = {"llama", "mistral"}

FIELD_ALIASES = {
    "vocab_size":  ["vocab_size"],
    "n_layers":    ["n_layers", "num_hidden_layers"],
    "n_heads":     ["n_heads", "num_attention_heads"],
    "d_model":     ["d_model", "hidden_size"],
    "d_ff":        ["d_ff", "intermediate_size"],
    "max_seq_len": ["max_seq_len", "max_position_embeddings"],
}

# Keys that indicate Grouped-Query or Multi-Query Attention.
_N_KV_HEADS_KEYS = ["num_key_value_heads", "n_kv_heads"]

# Keys that indicate a Mixture-of-Experts layer.
_MOE_KEYS = ["num_local_experts", "num_experts", "moe_num_experts"]

# Expected safetensors filename inside model-dir.
_DEFAULT_ST_NAME = "model.safetensors"

# Bytes per parameter for size estimation.
_BYTES_F32  = 4
_BYTES_Q8   = 1

# .att1 binary overhead: magic(8) + version(4) + header_size(4) + config(36) +
# header_padding to 80 bytes, then tensor descriptors (128 bytes each).
_ATT1_HEADER_FIXED = 80
_ATT1_DESC_SIZE    = 128


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve(cfg: dict, aliases: list) -> object:
    """Return the first value from cfg that matches any alias, or None."""
    for key in aliases:
        if key in cfg:
            val = cfg[key]
            if isinstance(val, float) and val == math.floor(val):
                return int(val)
            return val
    return None


def _fmt_bytes(n: int) -> str:
    """Format byte count as human-readable string."""
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    if n < 1024 * 1024 * 1024:
        return f"{n / (1024 * 1024):.1f} MB"
    return f"{n / (1024 * 1024 * 1024):.2f} GB"


def _count_parameters(cfg_resolved: dict) -> int:
    """Estimate total parameter count from resolved ATT-1 config fields.

    Counts: embed_tokens + output (lm_head) + per-layer weights.
    MHA assumed (n_kv_heads = n_heads) for worst-case size estimate.
    """
    V  = cfg_resolved["vocab_size"]
    L  = cfg_resolved["n_layers"]
    H  = cfg_resolved["d_model"]
    FF = cfg_resolved["d_ff"]

    # embed_tokens + lm_head
    params = 2 * V * H
    # per-layer: norm_attn(H) + q(H*H) + k(H*H) + v(H*H) + o(H*H)
    #            + norm_ffn(H) + gate(FF*H) + up(FF*H) + down(H*FF)
    per_layer = 2 * H + 4 * H * H + 2 * H + 3 * FF * H
    params += L * per_layer
    # output_norm
    params += H
    return params


def _att1_size_bytes(n_params: int, n_tensors: int) -> int:
    """Estimate .att1 file size (header + descs + payload)."""
    descs   = n_tensors * _ATT1_DESC_SIZE
    payload = n_params * _BYTES_F32
    return _ATT1_HEADER_FIXED + descs + payload


# ---------------------------------------------------------------------------
# Config scanning
# ---------------------------------------------------------------------------

class CompatError(Exception):
    """Fatal compatibility-scan error (exit 1)."""


def scan_config(config_path: str) -> tuple[dict, dict, list]:
    """Load and parse config.json.

    Returns (raw_cfg, resolved_fields, errors_list).
    errors_list contains non-fatal validation strings.
    On fatal error raises CompatError.
    """
    if not os.path.isfile(config_path):
        raise CompatError(f"config.json not found: {config_path}")

    try:
        with open(config_path, encoding="utf-8") as fh:
            raw = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        raise CompatError(f"config.json parse error: {exc}") from exc

    if not isinstance(raw, dict):
        raise CompatError("config.json root is not a JSON object")

    resolved = {}
    errors   = []

    for att1_key, aliases in FIELD_ALIASES.items():
        val = _resolve(raw, aliases)
        if val is None:
            errors.append(
                f"required field missing: {att1_key} "
                f"(searched: {aliases})"
            )
        else:
            resolved[att1_key] = int(val)

    return raw, resolved, errors


# ---------------------------------------------------------------------------
# Architecture validation
# ---------------------------------------------------------------------------

def check_architecture(raw_cfg: dict) -> tuple[str, list, list]:
    """Check model_type and detect unsupported features.

    Returns (arch_str, required_changes, warnings).
    """
    arch     = raw_cfg.get("model_type", "").lower()
    changes  = []
    warnings = []

    if not arch:
        raise CompatError("config.json is missing 'model_type' field")

    if arch not in SUPPORTED_ARCH:
        raise CompatError(
            f"unsupported model_type={arch!r}; "
            f"supported: {sorted(SUPPORTED_ARCH)}"
        )

    # --- MoE detection ---
    for key in _MOE_KEYS:
        val = raw_cfg.get(key)
        if val is not None and int(val) > 1:
            changes.append(
                f"MoE: {key}={val}; Mixture-of-Experts not supported by ATT-1"
            )

    # --- GQA / MQA detection ---
    n_heads    = _resolve(raw_cfg, ["num_attention_heads", "n_heads"])
    n_kv_heads = _resolve(raw_cfg, _N_KV_HEADS_KEYS)

    gqa_detected = False
    n_kv = None

    if n_kv_heads is not None and n_heads is not None:
        n_kv = int(n_kv_heads)
        n_q  = int(n_heads)
        if n_kv != n_q:
            ratio = n_q // n_kv if n_kv > 0 else "?"
            changes.append(
                f"GQA: num_key_value_heads={n_kv}, num_attention_heads={n_q} "
                f"(ratio {ratio}:1); GQA requires format change (n_kv_heads "
                f"in att1_model_config), converter change, and runtime "
                f"attention change (M68)"
            )
            gqa_detected = True
        else:
            warnings.append(
                f"num_key_value_heads={n_kv} equals num_attention_heads "
                f"({n_q}); treated as full MHA"
            )
    # If n_kv_heads absent → assume MHA (full attention); no warning needed.

    return arch, changes, warnings, gqa_detected, n_kv


# ---------------------------------------------------------------------------
# Safetensors validation
# ---------------------------------------------------------------------------

def scan_tensors(st_path: str, n_layers: int) -> tuple[dict, list, list]:
    """Scan safetensors metadata and validate LLaMA tensor presence/shapes.

    Returns (scan_result, required_changes, warnings).
    On fatal error raises CompatError.
    """
    try:
        scan = scan_safetensors(st_path)
    except ScanError as exc:
        raise CompatError(f"safetensors scan failed: {exc}") from exc

    if scan["errors"]:
        raise CompatError(
            f"safetensors header has {len(scan['errors'])} error(s): "
            + "; ".join(scan["errors"][:3])
        )

    changes  = []
    warnings = []

    # --- LLaMA tensor presence ---
    missing = check_llama_tensors(scan, n_layers)
    if missing:
        changes.append(
            f"missing {len(missing)} required LLaMA tensor(s): "
            + ", ".join(missing[:5])
            + ("…" if len(missing) > 5 else "")
        )

    # --- Dtype detection ---
    dtypes_found = {t["dtype"] for t in scan["tensors"]}
    non_f32 = dtypes_found - {"F32"}
    bf16_present = "BF16" in dtypes_found
    f16_present  = "F16" in dtypes_found

    coercible = {"BF16", "F16"}
    if bf16_present or f16_present:
        # BF16/F16 coercion to F32 is automatic in load_safetensors.py (M67).
        dtype_list = sorted(non_f32 & coercible)
        warnings.append(
            f"source dtype {dtype_list}: automatic BF16/F16\u2192F32 coercion "
            f"applied by load_safetensors.py (M67)"
        )
    truly_unsupported = non_f32 - coercible
    if truly_unsupported:
        # Something other than F32/BF16/F16 — flag as a required change.
        changes.append(
            f"unsupported source dtype(s): {sorted(truly_unsupported)}; "
            f"only F32/BF16/F16 are supported"
        )

    return scan, changes, warnings


# ---------------------------------------------------------------------------
# Tokenizer validation
# ---------------------------------------------------------------------------

def scan_tokenizer(model_dir: str, config_path: str) -> tuple[dict | None, list]:
    """Scan tokenizer assets in model_dir.

    Returns (result_or_None, warnings).  Missing tokenizer is a warning, not
    a fatal error.
    """
    warnings = []
    try:
        result = scan_tokenizer_dir(model_dir, config_path=config_path)
    except TokenizerScanError as exc:
        warnings.append(f"tokenizer scan failed: {exc}")
        return None, warnings

    if result["tokenizer_type"] == "byte":
        warnings.append(
            "no tokenizer assets found in model directory; "
            "byte-level token IDs will be used"
        )
        return result, warnings

    if result.get("vocab_size_match") is False:
        warnings.append(
            f"tokenizer vocab_size ({result.get('vocab_size')}) does not "
            f"match config.json vocab_size ({result.get('config_vocab_size')})"
        )

    return result, warnings


# ---------------------------------------------------------------------------
# Size estimation
# ---------------------------------------------------------------------------

def estimate_sizes(resolved: dict) -> tuple[int, int, int, int]:
    """Estimate f32 and q8 .att1 artifact sizes.

    Returns (n_params, n_tensors, f32_bytes, q8_bytes).
    """
    V  = resolved.get("vocab_size", 0)
    L  = resolved.get("n_layers", 0)
    H  = resolved.get("d_model", 0)
    FF = resolved.get("d_ff", 0)

    # Tensor count: embed(1) + per_layer(9) + output_norm(1) + lm_head(1)
    n_tensors = 1 + L * 9 + 2

    # Parameter count (MHA assumption for worst-case f32 size)
    n_params = _count_parameters(resolved)

    # f32: full precision
    f32_payload = n_params * _BYTES_F32
    f32_total   = _ATT1_HEADER_FIXED + n_tensors * _ATT1_DESC_SIZE + f32_payload

    # q8: projection matrices quantised (8 per layer: q/k/v/o/gate/up/down
    # = 7, but also embed and lm_head stay f32 in current scheme)
    # Projection matrices per layer: q(H*H), k(H*H), v(H*H), o(H*H),
    #                                  gate(FF*H), up(FF*H), down(H*FF) = 7
    proj_params = L * (4 * H * H + 3 * FF * H)
    norm_params = n_params - proj_params
    q8_payload  = proj_params * _BYTES_Q8 + norm_params * _BYTES_F32
    q8_total    = _ATT1_HEADER_FIXED + n_tensors * _ATT1_DESC_SIZE + q8_payload

    return n_params, n_tensors, f32_total, q8_total


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

_COL = 26   # column width for aligned output


def format_compat_report(result: dict) -> str:
    """Render a human-readable compatibility report."""
    lines = []

    def kv(key, val):
        lines.append(f"{key:<{_COL}}{val}")

    lines.append("# ATT-1 LLaMA compatibility scan")
    lines.append("")
    lines.append("# config")
    kv("config_path", result["config_path"])
    kv("model_type", result.get("arch", "?"))

    res = result.get("resolved", {})
    for field in ("vocab_size", "n_layers", "n_heads", "d_model", "d_ff",
                  "max_seq_len"):
        kv(field, res.get(field, "?"))

    n_kv = result.get("n_kv_heads")
    if n_kv is not None:
        kv("n_kv_heads", n_kv)
        kv("gqa_detected", "yes" if result.get("gqa_detected") else "no")
    else:
        kv("n_kv_heads", "absent (MHA assumed)")
        kv("gqa_detected", "no")

    lines.append("")
    lines.append("# safetensors")
    if result.get("st_skipped"):
        kv("safetensors_check", "skipped (--no-tensors)")
    else:
        kv("safetensors_path", result.get("st_path", "?"))
        kv("tensor_count", result.get("tensor_count", "?"))
        kv("source_dtype", result.get("source_dtype", "?"))
        kv("llama_check", result.get("llama_check", "?"))
        missing_n = result.get("llama_missing_count", 0)
        kv("llama_missing", missing_n)

    lines.append("")
    lines.append("# tokenizer")
    tok = result.get("tokenizer")
    if tok is None:
        kv("tokenizer_check", "skipped or failed")
    else:
        kv("tokenizer_type", tok.get("tokenizer_type", "?"))
        kv("vocab_size", tok.get("vocab_size") or "?")
        kv("vocab_size_match",
           "yes" if tok.get("vocab_size_match") else
           "no"  if tok.get("vocab_size_match") is False else
           "n/a")
        kv("bos_id", tok.get("bos_id"))
        kv("eos_id", tok.get("eos_id"))

    lines.append("")
    lines.append("# artifact size estimates")
    kv("n_params", result.get("n_params", "?"))
    kv("n_tensors_est", result.get("n_tensors_est", "?"))
    kv("f32_bytes", result.get("f32_bytes", "?"))
    kv("q8_bytes", result.get("q8_bytes", "?"))
    f32b = result.get("f32_bytes", 0)
    q8b  = result.get("q8_bytes", 0)
    if isinstance(f32b, int) and f32b > 0:
        kv("f32_size", _fmt_bytes(f32b))
        kv("q8_size", _fmt_bytes(q8b))

    lines.append("")
    # Required changes
    changes  = result.get("required_changes", [])
    warnings = result.get("warnings", [])
    errors   = result.get("errors", [])

    if errors:
        lines.append("# validation errors")
        for e in errors:
            lines.append(f"  error: {e}")
        lines.append("")

    if changes:
        lines.append("# required converter changes")
        for c in changes:
            lines.append(f"  required_change: {c}")
        lines.append("")
    else:
        lines.append("# required converter changes")
        kv("required_changes", "none")
        lines.append("")

    if warnings:
        lines.append("# warnings")
        for w in warnings:
            lines.append(f"  warning: {w}")
        lines.append("")

    # Final verdict
    if result.get("fatal"):
        lines.append(f"compat: error ({result['fatal']})")
    elif changes or errors:
        lines.append("compat: fail")
    else:
        lines.append("compat: pass")

    return "\n".join(lines)


def build_json_report(result: dict) -> dict:
    """Return a JSON-serialisable dict of the compat scan result."""
    return {
        "config_path":        result.get("config_path"),
        "arch":               result.get("arch"),
        "resolved":           result.get("resolved", {}),
        "n_kv_heads":         result.get("n_kv_heads"),
        "gqa_detected":       result.get("gqa_detected", False),
        "safetensors": {
            "skipped":        result.get("st_skipped", False),
            "path":           result.get("st_path"),
            "tensor_count":   result.get("tensor_count"),
            "source_dtype":   result.get("source_dtype"),
            "llama_check":    result.get("llama_check"),
            "llama_missing":  result.get("llama_missing_count", 0),
        },
        "tokenizer": {
            "type":              result.get("tokenizer", {}).get("tokenizer_type") if result.get("tokenizer") else None,
            "vocab_size":        result.get("tokenizer", {}).get("vocab_size") if result.get("tokenizer") else None,
            "vocab_size_match":  result.get("tokenizer", {}).get("vocab_size_match") if result.get("tokenizer") else None,
        },
        "estimates": {
            "n_params":     result.get("n_params"),
            "n_tensors":    result.get("n_tensors_est"),
            "f32_bytes":    result.get("f32_bytes"),
            "q8_bytes":     result.get("q8_bytes"),
        },
        "errors":            result.get("errors", []),
        "required_changes":  result.get("required_changes", []),
        "warnings":          result.get("warnings", []),
        "compat":            (
            "error" if result.get("fatal") else
            "fail"  if (result.get("required_changes") or result.get("errors")) else
            "pass"
        ),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="ATT-1 LLaMA compatibility scanner",
    )
    parser.add_argument(
        "--model-dir", metavar="PATH", required=True,
        help="local model directory containing config.json "
             "(and optionally model.safetensors and tokenizer assets)",
    )
    parser.add_argument(
        "--safetensors", metavar="PATH", default=None,
        help="explicit path to .safetensors file "
             "(default: <model-dir>/model.safetensors)",
    )
    parser.add_argument(
        "--no-tensors", action="store_true",
        help="skip safetensors check (useful when weights not downloaded)",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="emit JSON report to stdout instead of human-readable text",
    )
    args = parser.parse_args(argv)

    model_dir = args.model_dir

    result = {
        "config_path":      os.path.join(model_dir, "config.json"),
        "required_changes": [],
        "warnings":         [],
        "errors":           [],
        "fatal":            None,
    }

    # ------------------------------------------------------------------ #
    # 1. Validate model directory                                          #
    # ------------------------------------------------------------------ #
    if not os.path.isdir(model_dir):
        result["fatal"] = f"model directory not found: {model_dir}"
        _emit(result, args.json)
        return 1

    # ------------------------------------------------------------------ #
    # 2. Config scan                                                       #
    # ------------------------------------------------------------------ #
    config_path = os.path.join(model_dir, "config.json")
    result["config_path"] = config_path

    try:
        raw_cfg, resolved, cfg_errors = scan_config(config_path)
    except CompatError as exc:
        result["fatal"] = str(exc)
        _emit(result, args.json)
        return 1

    result["resolved"] = resolved
    result["errors"].extend(cfg_errors)

    # ------------------------------------------------------------------ #
    # 3. Architecture check                                                #
    # ------------------------------------------------------------------ #
    try:
        arch, arch_changes, arch_warnings, gqa_detected, n_kv = \
            check_architecture(raw_cfg)
    except CompatError as exc:
        result["fatal"] = str(exc)
        _emit(result, args.json)
        return 1

    result["arch"]         = arch
    result["gqa_detected"] = gqa_detected
    result["n_kv_heads"]   = n_kv
    result["required_changes"].extend(arch_changes)
    result["warnings"].extend(arch_warnings)

    # ------------------------------------------------------------------ #
    # 4. Safetensors scan                                                  #
    # ------------------------------------------------------------------ #
    if args.no_tensors:
        result["st_skipped"] = True
    else:
        st_path = args.safetensors or os.path.join(model_dir, _DEFAULT_ST_NAME)
        result["st_path"] = st_path

        if not os.path.isfile(st_path):
            result["errors"].append(
                f"safetensors file not found: {st_path}; "
                f"use --no-tensors to skip or --safetensors to specify path"
            )
        else:
            n_layers = resolved.get("n_layers", 0)
            try:
                scan, st_changes, st_warnings = scan_tensors(st_path, n_layers)
            except CompatError as exc:
                result["errors"].append(f"safetensors: {exc}")
            else:
                # Summarise dtypes
                dtypes = sorted({t["dtype"] for t in scan["tensors"]})
                result["tensor_count"]      = len(scan["tensors"])
                result["source_dtype"]      = "/".join(dtypes) if dtypes else "none"
                result["llama_check"]       = "ok" if not st_changes or not any("missing" in c for c in st_changes) else "fail"
                missing_n                   = len(check_llama_tensors(scan, n_layers))
                result["llama_missing_count"] = missing_n
                if missing_n == 0:
                    result["llama_check"] = "ok"
                result["required_changes"].extend(st_changes)
                result["warnings"].extend(st_warnings)

    # ------------------------------------------------------------------ #
    # 5. Tokenizer scan                                                    #
    # ------------------------------------------------------------------ #
    tok_result, tok_warnings = scan_tokenizer(model_dir, config_path)
    result["tokenizer"] = tok_result
    result["warnings"].extend(tok_warnings)

    # ------------------------------------------------------------------ #
    # 6. Size estimates                                                    #
    # ------------------------------------------------------------------ #
    if len(resolved) == len(FIELD_ALIASES):
        n_params, n_tensors, f32b, q8b = estimate_sizes(resolved)
        result["n_params"]      = n_params
        result["n_tensors_est"] = n_tensors
        result["f32_bytes"]     = f32b
        result["q8_bytes"]      = q8b

    # ------------------------------------------------------------------ #
    # 7. Emit report                                                       #
    # ------------------------------------------------------------------ #
    _emit(result, args.json)

    has_fail = bool(result["required_changes"] or result["errors"])
    return 2 if has_fail else 0


def _emit(result: dict, as_json: bool) -> None:
    if as_json:
        print(json.dumps(build_json_report(result), indent=2))
    else:
        print(format_compat_report(result))


if __name__ == "__main__":
    sys.exit(main())
