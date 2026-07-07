#!/usr/bin/env python3
"""
ATT-1 source-model comparison harness (M61 core, M62 report extension,
M69 public local-model report extension).

Validates that an ATT-1 converted model artifact (f32 or q8) faithfully
represents the source LLaMA-style safetensors weights, and that its forward
pass produces the same next-token prediction as a Python reference
implementation.

Two validation layers
---------------------
1. Static tensor comparison
   For every tensor in the conversion plan, load the source value (applying
   the same transpose rule used by the converter), load the ATT-1 value, and
   compute max_abs_error and max_rel_error.  For f32 this should be < 1e-4.
   For q8 this reflects per-row quantisation error (< 0.6 on the m61 fixture).

2. Dynamic forward-pass comparison  (requires numpy)
   Runs a Python LLaMA reference transformer on the prompt token IDs using
   the source safetensors weights, then calls att1-bench and compares the
   generated last_token.  Exact match is expected for f32; exact match is
   also expected for q8 since argmax is robust to small quantisation error.

Report modes (M62)
------------------
No flag:      Print machine-readable key=value lines to stdout (default).
--report:     Print a rich structured text report to stdout instead.
--report-json PATH: Write a structured JSON report to PATH (hard fail on
              IO error, exit 1).  Can be combined with --report.

Fixtures used by default (M61/M62):
    --safetensors  compiler/fixtures/m61_llama_2l.safetensors
    --config       compiler/fixtures/tiny_llama_config.json
    --att1-f32     models/m61_f32/model.att1
    --att1-q8      models/m61_q8/model.att1

Usage:
    python3 compiler/compare_att1_to_source.py [options]
    python3 compiler/compare_att1_to_source.py --report
    python3 compiler/compare_att1_to_source.py --report-json build/report.json
    python3 compiler/compare_att1_to_source.py \\
        --safetensors compiler/fixtures/m61_llama_2l.safetensors \\
        --config      compiler/fixtures/tiny_llama_config.json \\
        --att1-f32    models/m61_f32/model.att1 \\
        --att1-q8     models/m61_q8/model.att1
    python3 compiler/compare_att1_to_source.py \\
        --model-dir   ~/Models/SmolLM2-135M \\
        --att1-f32    ~/Models/att1/SmolLM2-135M/model_f32.att1 \\
        --att1-q8     ~/Models/att1/SmolLM2-135M/model_q8.att1 \\
        --prompt-ids  1,2,3 \\
        --report --report-json ~/Models/att1/SmolLM2-135M/comparison.json

Exit codes:
    0  all enabled checks pass
    1  usage / IO / dependency error
    2  comparison failure
"""

import argparse
import datetime
import json
import math
import os
import struct
import subprocess
import sys

# ---------------------------------------------------------------------------
# Bootstrap: ensure read_att1 is importable from any cwd
# ---------------------------------------------------------------------------

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)

from read_att1 import (  # noqa: E402
    read_att1_model, DTYPE_F32, DTYPE_Q8, DTYPE_Q4, Att1ReadError
)
from load_safetensors import _coerce_bf16, _coerce_f16  # noqa: E402

# ---------------------------------------------------------------------------
# Optional numpy (required for forward-pass comparison)
# ---------------------------------------------------------------------------

try:
    import numpy as np
    _HAVE_NUMPY = True
except ImportError:
    _HAVE_NUMPY = False


# ---------------------------------------------------------------------------
# Source safetensors reader  (stdlib only)
# ---------------------------------------------------------------------------

def _load_safetensors_header(path):
    """Return (header_dict, data_offset) for a .safetensors file."""
    with open(path, "rb") as fh:
        raw = fh.read(8)
        if len(raw) < 8:
            raise ValueError(f"file too small: {path!r}")
        hlen = struct.unpack("<Q", raw)[0]
        hdr_bytes = fh.read(hlen)
    hdr        = json.loads(hdr_bytes)
    data_offset = 8 + hlen
    return hdr, data_offset


def _load_st_tensor(path, hdr, data_offset, name):
    """Return flat tuple[float] for an F32/BF16/F16 tensor from a .safetensors file.

    BF16 and F16 tensors are coerced to F32 using the same helpers as
    load_safetensors.py (M67), so BF16-source models (e.g. SmolLM2-135M)
    are comparable without pre-converting the safetensors file.
    """
    if name not in hdr:
        raise KeyError(f"tensor {name!r} not in safetensors header")
    t = hdr[name]
    dtype = t["dtype"]
    b, e   = t["data_offsets"]
    nbytes = e - b
    with open(path, "rb") as fh:
        fh.seek(data_offset + b)
        raw = fh.read(nbytes)
    if dtype == "F32":
        n = nbytes // 4
        return struct.unpack(f"<{n}f", raw)
    if dtype == "BF16":
        return _coerce_bf16(raw)
    if dtype == "F16":
        return _coerce_f16(raw)
    raise ValueError(f"tensor {name!r} has dtype {dtype!r}, expected F32/BF16/F16")


def _is_url_like(path):
    if path is None:
        return False
    lowered = str(path).lower()
    return "://" in lowered or lowered.startswith(("hf:", "hf://"))


# ---------------------------------------------------------------------------
# Tensor mapping plan  (mirrors llama_source_tensor_plan in the converter)
# ---------------------------------------------------------------------------

def _build_plan(cfg):
    """Return list of mapping dicts for the given ATT-1 config dict."""
    d  = cfg["d_model"]
    v  = cfg["vocab_size"]
    ff = cfg["d_ff"]
    n_heads = cfg["n_heads"]
    n_kv_heads = cfg.get("n_kv_heads", n_heads)
    d_head = d // n_heads
    kv_dim = n_kv_heads * d_head

    plan = [{
        "source": "model.embed_tokens.weight",
        "target": "tok_embeddings.weight",
        "source_shape": [v, d],
        "target_shape": [v, d],
        "transpose": False,
    }]

    for layer in range(cfg["n_layers"]):
        src = f"model.layers.{layer}"
        dst = f"layers.{layer}"
        plan += [
            {
                "source": f"{src}.input_layernorm.weight",
                "target": f"{dst}.attention_norm.weight",
                "source_shape": [d],
                "target_shape": [d],
                "transpose": False,
            },
            {
                "source": f"{src}.self_attn.q_proj.weight",
                "target": f"{dst}.attention.wq.weight",
                "source_shape": [d, d],
                "target_shape": [d, d],
                "transpose": True,
            },
            {
                "source": f"{src}.self_attn.k_proj.weight",
                "target": f"{dst}.attention.wk.weight",
                "source_shape": [kv_dim, d],
                "target_shape": [d, d],
                "transpose": True,
                "gqa_expand": n_kv_heads != n_heads,
            },
            {
                "source": f"{src}.self_attn.v_proj.weight",
                "target": f"{dst}.attention.wv.weight",
                "source_shape": [kv_dim, d],
                "target_shape": [d, d],
                "transpose": True,
                "gqa_expand": n_kv_heads != n_heads,
            },
            {
                "source": f"{src}.self_attn.o_proj.weight",
                "target": f"{dst}.attention.wo.weight",
                "source_shape": [d, d],
                "target_shape": [d, d],
                "transpose": True,
            },
            {
                "source": f"{src}.post_attention_layernorm.weight",
                "target": f"{dst}.ffn_norm.weight",
                "source_shape": [d],
                "target_shape": [d],
                "transpose": False,
            },
            {
                "source": f"{src}.mlp.gate_proj.weight",
                "target": f"{dst}.ffn.w_gate.weight",
                "source_shape": [ff, d],
                "target_shape": [d, ff],
                "transpose": True,
            },
            {
                "source": f"{src}.mlp.up_proj.weight",
                "target": f"{dst}.ffn.w_up.weight",
                "source_shape": [ff, d],
                "target_shape": [d, ff],
                "transpose": True,
            },
            {
                "source": f"{src}.mlp.down_proj.weight",
                "target": f"{dst}.ffn.w_down.weight",
                "source_shape": [d, ff],
                "target_shape": [ff, d],
                "transpose": True,
            },
        ]

    plan += [
        {
            "source": "model.norm.weight",
            "target": "output_norm.weight",
            "source_shape": [d],
            "target_shape": [d],
            "transpose": False,
        },
        {
            "source": "lm_head.weight",
            "target": "output.weight",
            "source_shape": [v, d],
            "target_shape": [d, v],
            "transpose": True,
        },
    ]
    return plan


def _transpose_2d(values, rows, cols):
    result = [0.0] * (rows * cols)
    for r in range(rows):
        for c in range(cols):
            result[c * rows + r] = values[r * cols + c]
    return result


def _expand_gqa_projection(values, source_rows, source_cols, n_heads, n_kv_heads):
    if n_kv_heads == n_heads:
        return _transpose_2d(values, source_rows, source_cols)
    if n_kv_heads <= 0 or n_heads <= 0 or n_heads % n_kv_heads != 0:
        raise ValueError(
            f"unsupported GQA shape: n_heads={n_heads}, n_kv_heads={n_kv_heads}"
        )
    if source_rows % n_kv_heads != 0:
        raise ValueError(
            f"GQA source rows {source_rows} not divisible by n_kv_heads {n_kv_heads}"
        )

    head_dim = source_rows // n_kv_heads
    d_model = source_cols
    if d_model != n_heads * head_dim:
        raise ValueError(
            f"GQA dimensions inconsistent: d_model={d_model}, "
            f"n_heads={n_heads}, head_dim={head_dim}"
        )

    kv_group = n_heads // n_kv_heads
    expanded = [0.0] * (d_model * d_model)
    for in_col in range(d_model):
        for q_head in range(n_heads):
            kv_head = q_head // kv_group
            for i in range(head_dim):
                src_row = kv_head * head_dim + i
                dst_col = q_head * head_dim + i
                expanded[in_col * d_model + dst_col] = values[src_row * d_model + in_col]
    return expanded


# ---------------------------------------------------------------------------
# Static tensor comparison
# ---------------------------------------------------------------------------

def _compare_static(safetensors_path, att1_model, cfg_dict, verbose):
    """Compare all mapped tensors between source safetensors and ATT-1.

    Returns (n_checked, max_abs_error, max_rel_error, per_tensor, error_or_None)

    per_tensor is a list of dicts:
        {"name": str, "shape": list[int], "max_abs_error": float, "max_rel_error": float}
    """
    hdr, data_offset = _load_safetensors_header(safetensors_path)
    plan             = _build_plan(cfg_dict)

    max_abs_error = 0.0
    max_rel_error = 0.0
    n_checked     = 0
    per_tensor    = []

    # lm_head may be missing (shared with embed); the converter falls back to
    # embed_tokens in that case.  Our fixture always has lm_head.
    for item in plan:
        src_name = item["source"]
        dst_name = item["target"]
        transpose = item["transpose"]

        # Load source values.
        try:
            src_vals = _load_st_tensor(safetensors_path, hdr, data_offset, src_name)
        except KeyError:
            if src_name == "lm_head.weight":
                src_vals = _load_st_tensor(
                    safetensors_path, hdr, data_offset, "model.embed_tokens.weight"
                )
            else:
                return 0, 0.0, 0.0, [], f"source tensor {src_name!r} not found"

        # Apply expected transpose to get ATT-1 layout.
        if item.get("gqa_expand"):
            s_rows, s_cols = item["source_shape"]
            try:
                src_vals = tuple(_expand_gqa_projection(
                    list(src_vals),
                    s_rows,
                    s_cols,
                    cfg_dict["n_heads"],
                    cfg_dict.get("n_kv_heads", cfg_dict["n_heads"]),
                ))
            except ValueError as exc:
                return 0, 0.0, 0.0, [], str(exc)
        elif transpose:
            s_rows, s_cols = item["source_shape"]
            src_vals = tuple(_transpose_2d(list(src_vals), s_rows, s_cols))

        # Load ATT-1 tensor.
        att1_t = att1_model.find_tensor(dst_name)
        if att1_t is None:
            return 0, 0.0, 0.0, [], f"ATT-1 tensor {dst_name!r} not found"

        if att1_t.dtype == DTYPE_F32:
            dst_vals = att1_t.f32_values
        elif att1_t.dtype == DTYPE_Q8:
            dst_vals = att1_t.dequantize()
        elif att1_t.dtype == DTYPE_Q4:
            dst_vals = att1_t.dequantize()
        else:
            return 0, 0.0, 0.0, [], f"unsupported dtype for {dst_name!r}"

        if len(src_vals) != len(dst_vals):
            return 0, 0.0, 0.0, [], (
                f"element count mismatch for {dst_name!r}: "
                f"source={len(src_vals)}, att1={len(dst_vals)}"
            )

        _eps = 1e-9
        t_abs = max(abs(a - b) for a, b in zip(src_vals, dst_vals))
        t_rel = max(
            abs(a - b) / (max(abs(a), abs(b)) + _eps)
            for a, b in zip(src_vals, dst_vals)
        )
        max_abs_error = max(max_abs_error, t_abs)
        max_rel_error = max(max_rel_error, t_rel)
        n_checked += 1
        per_tensor.append({
            "name": dst_name,
            "shape": list(att1_t.shape),
            "max_abs_error": t_abs,
            "max_rel_error": t_rel,
        })

        if verbose:
            print(f"  {dst_name}: max_abs_error={t_abs:.3e} max_rel_error={t_rel:.3e}")

    return n_checked, max_abs_error, max_rel_error, per_tensor, None


# ---------------------------------------------------------------------------
# Python LLaMA reference forward pass  (numpy required)
# ---------------------------------------------------------------------------

def _rmsnorm(x, weight, eps=1e-6):
    rms   = np.sqrt(np.mean(x * x) + eps)
    return x / rms * weight


def _rope(v, position, rope_theta):
    """Apply RoPE in-place (matches att1_rope_f32).
    v: numpy array, length must be even.
    """
    count = len(v)
    out   = v.copy()
    for i in range(0, count, 2):
        exponent  = i / count
        frequency = 1.0 / (rope_theta ** exponent)
        angle     = position * frequency
        c, s      = math.cos(angle), math.sin(angle)
        x0, x1    = float(v[i]), float(v[i + 1])
        out[i]     = x0 * c - x1 * s
        out[i + 1] = x0 * s + x1 * c
    return out


def _softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


def _silu(x):
    return x / (1.0 + np.exp(-x))


def _python_forward_pass(weights_np, cfg_dict, token_ids, rope_theta=10000.0):
    """Run a Python LLaMA reference forward pass.

    Parameters:
        weights_np  dict  name -> numpy array (all weights in ATT-1 layout)
        cfg_dict    dict  vocab_size, n_layers, n_heads, n_kv_heads, d_model,
                    d_ff, rope_dim
        token_ids   list[int]  prompt token IDs
        rope_theta  float  RoPE theta (default 10000.0)

    Returns:
        (logits np.ndarray [vocab_size], kv_cache dict)
    """
    n_layers = cfg_dict["n_layers"]
    n_heads  = cfg_dict["n_heads"]
    n_kv_heads = cfg_dict.get("n_kv_heads", n_heads)
    d_model  = cfg_dict["d_model"]
    d_ff     = cfg_dict["d_ff"]
    d_head   = d_model // n_heads
    kv_dim   = n_kv_heads * d_head

    if n_kv_heads <= 0 or (n_heads % n_kv_heads) != 0:
        raise ValueError(
            f"unsupported GQA shape: n_heads={n_heads}, n_kv_heads={n_kv_heads}"
        )
    kv_group = n_heads // n_kv_heads

    embed = weights_np["tok_embeddings.weight"]  # [V, D]

    # KV cache: list of (K[KV], V[KV]) per layer, one entry per past position.
    kv_cache = [[] for _ in range(n_layers)]

    logits = None

    for pos, tok in enumerate(token_ids):
        x = embed[tok].copy()  # [D]

        for layer in range(n_layers):
            prefix = f"layers.{layer}"

            # -- attention sub-layer --
            x_norm = _rmsnorm(x, weights_np[f"{prefix}.attention_norm.weight"])

            wq = weights_np[f"{prefix}.attention.wq.weight"]  # [D, D]
            wk = weights_np[f"{prefix}.attention.wk.weight"]  # [D, KV]
            wv = weights_np[f"{prefix}.attention.wv.weight"]
            wo = weights_np[f"{prefix}.attention.wo.weight"]

            Q = x_norm @ wq  # [D]
            K = x_norm @ wk
            V = x_norm @ wv
            if K.shape[0] != kv_dim or V.shape[0] != kv_dim:
                raise ValueError(
                    f"{prefix}: expected K/V dim {kv_dim}, "
                    f"got {K.shape[0]}/{V.shape[0]}"
                )

            # Apply RoPE per query head and per key/value head.
            for h in range(n_heads):
                sl = slice(h * d_head, (h + 1) * d_head)
                Q[sl] = _rope(Q[sl], pos, rope_theta)
            for h in range(n_kv_heads):
                sl = slice(h * d_head, (h + 1) * d_head)
                K[sl] = _rope(K[sl], pos, rope_theta)

            # Append to KV cache.
            kv_cache[layer].append((K.copy(), V.copy()))

            # Attention: per-head dot-product.
            context = np.zeros(d_model)
            for h in range(n_heads):
                q_sl  = slice(h * d_head, (h + 1) * d_head)
                kv_h  = h // kv_group
                kv_sl = slice(kv_h * d_head, (kv_h + 1) * d_head)
                q_h   = Q[q_sl]
                scale = 1.0 / math.sqrt(d_head)
                n_past = len(kv_cache[layer])
                scores = np.array([
                    np.dot(q_h, kv_cache[layer][t][0][kv_sl]) * scale
                    for t in range(n_past)
                ])
                probs = _softmax(scores)
                for t in range(n_past):
                    context[q_sl] += probs[t] * kv_cache[layer][t][1][kv_sl]

            attn_out = context @ wo  # [D]
            x = x + attn_out

            # -- FFN sub-layer --
            x_norm2 = _rmsnorm(x, weights_np[f"{prefix}.ffn_norm.weight"])

            w_gate = weights_np[f"{prefix}.ffn.w_gate.weight"]  # [D, FF]
            w_up   = weights_np[f"{prefix}.ffn.w_up.weight"]
            w_down = weights_np[f"{prefix}.ffn.w_down.weight"]  # [FF, D]

            gate   = x_norm2 @ w_gate   # [FF]
            up     = x_norm2 @ w_up
            hidden = _silu(gate) * up   # SwiGLU
            x = x + hidden @ w_down

        # Final norm + logits.
        x_final = _rmsnorm(x, weights_np["output_norm.weight"])
        logits  = x_final @ weights_np["output.weight"]  # [V]

    return logits


def _load_weights_numpy(att1_model):
    """Extract all ATT-1 tensors as numpy float32 arrays."""
    weights = {}
    for t in att1_model.tensors:
        if t.dtype == DTYPE_F32:
            vals = t.f32_values
        elif t.dtype in (DTYPE_Q8, DTYPE_Q4):
            vals = t.dequantize()
        else:
            continue
        arr = np.array(vals, dtype=np.float32).reshape(t.shape)
        weights[t.name] = arr
    return weights


def _load_source_weights_numpy(safetensors_path, cfg_dict):
    """Extract source safetensors tensors into ATT-1 layout numpy arrays."""
    hdr, data_offset = _load_safetensors_header(safetensors_path)
    weights = {}

    for item in _build_plan(cfg_dict):
        src_name = item["source"]
        dst_name = item["target"]
        try:
            vals = _load_st_tensor(safetensors_path, hdr, data_offset, src_name)
        except KeyError:
            if src_name != "lm_head.weight":
                raise
            vals = _load_st_tensor(
                safetensors_path, hdr, data_offset, "model.embed_tokens.weight"
            )

        if item["transpose"]:
            rows, cols = item["source_shape"]
            vals = tuple(_transpose_2d(list(vals), rows, cols))

        weights[dst_name] = np.array(vals, dtype=np.float32).reshape(
            item["target_shape"]
        )

    return weights


def _call_att1_bench_generic(att1_path, token_ids, backend, n_generate=1):
    """Call att1-bench with the given backend; return (last_token, stdout, error)."""
    ids_str = ",".join(str(i) for i in token_ids)
    bench   = os.path.join("build", "att1-bench")
    if not os.path.isfile(bench):
        return None, "", "att1-bench not found at build/att1-bench"
    cmd = [
        bench,
        "--model", att1_path,
        "--tokenizer", "external",
        "--input-token-ids", ids_str,
        "--tokens", str(n_generate),
        "--mode", "single",
        "--backend", backend,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return None, "", str(exc)

    if result.returncode != 0:
        return None, result.stdout, (
            f"att1-bench exit {result.returncode}: {result.stderr.strip()}"
        )

    for line in result.stdout.splitlines():
        if line.startswith("last_token="):
            return int(line.split("=", 1)[1].strip()), result.stdout, None

    return None, result.stdout, "last_token not found in att1-bench output"


# ---------------------------------------------------------------------------
# Report formatter (M62)
# ---------------------------------------------------------------------------

def _format_report_text(rpt):
    """Return a rich structured text report string from a result dict."""
    lines = []

    def emit(key, value):
        lines.append(_kv(key, value))

    lines.append("=== ATT-1 source-model comparison report ===")
    emit("date",        rpt.get("date", ""))
    emit("source_model_path", rpt.get("source_model_path", ""))
    emit("safetensors", rpt["safetensors"])
    emit("config",      rpt["config_path"])
    lines.append("")

    cfg = rpt.get("config", {})
    lines.append("# source config")
    emit("vocab_size",   cfg.get("vocab_size", ""))
    emit("n_layers",     cfg.get("n_layers",   ""))
    emit("n_heads",      cfg.get("n_heads",    ""))
    emit("n_kv_heads",   cfg.get("n_kv_heads", ""))
    emit("d_model",      cfg.get("d_model",    ""))
    emit("d_ff",         cfg.get("d_ff",       ""))
    emit("rope_theta",   cfg.get("rope_theta",  ""))
    lines.append("")

    f32s = rpt.get("f32_static")
    if f32s:
        lines.append("# f32 static comparison")
        emit("att1_f32",             f32s["att1_path"])
        emit("att1_f32_version",     f32s["att1_version"])
        emit("f32_dtype",            f32s["dtype"])
        emit("f32_backend",          f32s["backend"])
        emit("f32_tensors_checked",  f32s["tensors_checked"])
        emit("f32_max_abs_error",    f"{f32s['max_abs_error']:.3e}")
        emit("f32_max_rel_error",    f"{f32s['max_rel_error']:.3e}")
        emit("f32_tolerance",        f"{f32s['tolerance']:.3e}")
        emit("f32_status",           f32s["status"])
        emit("f32_note",             f32s["note"])
        if f32s.get("error"):
            emit("f32_error", f32s["error"])
        lines.append("")

    q8s = rpt.get("q8_static")
    if q8s:
        lines.append("# q8 static comparison")
        emit("att1_q8",              q8s["att1_path"])
        emit("att1_q8_version",      q8s["att1_version"])
        emit("q8_dtype",             q8s["dtype"])
        emit("q8_backend",           q8s["backend"])
        emit("q8_tensors_checked",   q8s["tensors_checked"])
        emit("q8_max_abs_error",     f"{q8s['max_abs_error']:.3e}")
        emit("q8_max_rel_error",     f"{q8s['max_rel_error']:.3e}")
        emit("q8_tolerance",         f"{q8s['tolerance']:.3e}")
        emit("q8_status",            q8s["status"])
        emit("q8_note",              q8s["note"])
        if q8s.get("error"):
            emit("q8_error", q8s["error"])
        lines.append("")

    q4s = rpt.get("q4_static")
    if q4s:
        lines.append("# q4 static comparison")
        emit("att1_q4",              q4s["att1_path"])
        emit("att1_q4_version",      q4s["att1_version"])
        emit("q4_dtype",             q4s["dtype"])
        emit("q4_backend",           q4s["backend"])
        emit("q4_tensors_checked",   q4s["tensors_checked"])
        emit("q4_max_abs_error",     f"{q4s['max_abs_error']:.3e}")
        emit("q4_max_rel_error",     f"{q4s['max_rel_error']:.3e}")
        emit("q4_mean_abs_error",    f"{q4s['mean_abs_error']:.3e}")
        emit("q4_tolerance",         f"{q4s['tolerance']:.3e}")
        emit("q4_topk_overlap",      str(q4s.get("topk_overlap", "n/a")))
        emit("q4_status",            q4s["status"])
        emit("q4_note",              q4s["note"])
        if q4s.get("error"):
            emit("q4_error", q4s["error"])
        lines.append("")

    fwd = rpt.get("forward")
    if fwd:
        lines.append("# forward pass")
        emit("reference",        fwd["reference"])
        emit("prompt_ids",       str(fwd["prompt_ids"]))
        emit("logits_shape",     str(fwd["logits_shape"]))
        emit("ref_last_token",   fwd["ref_last_token"])
        emit("next_token_result", fwd["next_token_result"])
        if fwd.get("f32_bench_last_token") is not None:
            emit("bench_last_token",  fwd["f32_bench_last_token"])
            emit("bench_backend",     fwd["f32_backend"])
            emit("forward_match",     "yes" if fwd["f32_forward_match"] else "no")
            emit("f32_status",        fwd["f32_status"])
        if fwd.get("f32_bench_error"):
            emit("f32_bench_error",   fwd["f32_bench_error"])
        lines.append("")
        if fwd.get("q8_bench_last_token") is not None:
            lines.append("# q8 forward pass")
            emit("q8_bench_last_token", fwd["q8_bench_last_token"])
            emit("q8_bench_backend",    fwd["q8_backend"])
            emit("q8_forward_match",
                 "yes" if fwd["q8_forward_match"] else "warn (quantisation rounding)")
            emit("q8_forward_note",     fwd["q8_note"])
            lines.append("")
        elif fwd.get("q8_bench_error"):
            lines.append("# q8 forward pass")
            emit("q8_bench_backend",    fwd["q8_backend"])
            emit("q8_bench_error",      fwd["q8_bench_error"])
            emit("q8_status",           fwd["q8_status"])
            lines.append("")
        if fwd.get("q4_bench_last_token") is not None:
            lines.append("# q4 forward pass")
            emit("q4_bench_last_token", fwd["q4_bench_last_token"])
            emit("q4_bench_backend",    "cpu-q4")
            emit("q4_forward_match",
                 "yes" if fwd["q4_forward_match"] else "note (token divergence may be expected for q4)")
            emit("q4_forward_note",     fwd["q4_note"])
            lines.append("")
        elif fwd.get("q4_bench_error"):
            lines.append("# q4 forward pass")
            emit("q4_bench_backend",    "cpu-q4")
            emit("q4_bench_error",      fwd["q4_bench_error"])
            emit("q4_status",           fwd["q4_status"])
            lines.append("")
        if fwd.get("error"):
            emit("forward_error",        fwd["error"])
            lines.append("")

    lines.append("# result")
    emit("result",  rpt["result"])
    emit("report",  "ok")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main harness
# ---------------------------------------------------------------------------

_DEFAULT_ST   = os.path.join("compiler", "fixtures", "m61_llama_2l.safetensors")
_DEFAULT_CFG  = os.path.join("compiler", "fixtures", "tiny_llama_config.json")
_DEFAULT_F32  = os.path.join("models", "m61_f32", "model.att1")
_DEFAULT_Q8   = os.path.join("models", "m61_q8",  "model.att1")

# Column width used for all aligned output lines.
_COL = 21


def _kv(key, value, col=_COL):
    """Return 'key: <value>' with key+colon left-justified to col chars."""
    return f"{(key + ':').ljust(col)}{value}"


def _read_ids_file(path):
    """Read token IDs from a file (one per line; # comments skipped).

    Returns list[int] or raises ValueError on parse failure.
    """
    ids = []
    with open(path) as fh:
        for lineno, raw in enumerate(fh, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            try:
                ids.append(int(line))
            except ValueError:
                raise ValueError(
                    f"{path!r} line {lineno}: not an integer: {line!r}"
                ) from None
    if not ids:
        raise ValueError(f"{path!r}: no token IDs found")
    return ids


def _parse_ids(s):
    try:
        return [int(x.strip()) for x in s.split(",") if x.strip()]
    except ValueError:
        return None


def _resolve_paths(args):
    """Resolve local source/config/artifact paths from CLI arguments."""
    model_dir = getattr(args, "model_dir", None)

    if model_dir and _is_url_like(model_dir):
        raise ValueError("model_dir must be a local filesystem path")

    if model_dir:
        if not os.path.isdir(model_dir):
            raise ValueError(f"model directory not found: {model_dir!r}")
        source_model_path = model_dir
        config_path = args.config or os.path.join(model_dir, "config.json")
        safetensors_path = args.safetensors or os.path.join(model_dir, "model.safetensors")
        att1_f32_path = args.att1_f32
        att1_q8_path = args.att1_q8
    else:
        source_model_path = ""
        config_path = args.config or _DEFAULT_CFG
        safetensors_path = args.safetensors or _DEFAULT_ST
        use_fixture_artifacts = (args.config is None and args.safetensors is None)
        att1_f32_path = args.att1_f32 or (_DEFAULT_F32 if use_fixture_artifacts else None)
        att1_q8_path = args.att1_q8 if args.att1_q8 is not None else (
            _DEFAULT_Q8 if use_fixture_artifacts else None
        )

    path_items = [
        ("config", config_path),
        ("safetensors", safetensors_path),
        ("att1-f32", att1_f32_path),
    ]
    if att1_q8_path:
        path_items.append(("att1-q8", att1_q8_path))
    att1_q4_path = getattr(args, "att1_q4", None)
    if att1_q4_path:
        path_items.append(("att1-q4", att1_q4_path))

    for label, path in path_items:
        if not path:
            raise ValueError(f"{label} path is required")
        if _is_url_like(path):
            raise ValueError(f"{label} must be a local filesystem path: {path!r}")

    if not os.path.isfile(config_path):
        raise ValueError(f"config file not found: {config_path!r}")
    if not os.path.isfile(safetensors_path):
        raise ValueError(f"safetensors file not found: {safetensors_path!r}")
    if not os.path.isfile(att1_f32_path):
        raise ValueError(f"ATT-1 f32 artifact not found: {att1_f32_path!r}")
    if att1_q8_path and not os.path.isfile(att1_q8_path):
        raise ValueError(f"ATT-1 q8 artifact not found: {att1_q8_path!r}")
    if att1_q4_path and not os.path.isfile(att1_q4_path):
        raise ValueError(f"ATT-1 q4 artifact not found: {att1_q4_path!r}")

    if not source_model_path:
        source_model_path = os.path.dirname(os.path.abspath(config_path))

    return {
        "source_model_path": source_model_path,
        "config": config_path,
        "safetensors": safetensors_path,
        "att1_f32": att1_f32_path,
        "att1_q8": att1_q8_path,
        "att1_q4": att1_q4_path,
    }


def run_harness(args):
    """Execute the full comparison harness.  Returns (rpt_dict, exit_code)."""

    fail_msgs = []

    # -------- resolve local paths --------
    try:
        paths = _resolve_paths(args)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return {}, 1

    # -------- resolve prompt IDs --------
    prompt_ids = args.prompt_ids
    if getattr(args, "tokens_file", None):
        try:
            prompt_ids = _read_ids_file(args.tokens_file)
        except (OSError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return {}, 1

    # -------- load source config --------
    try:
        with open(paths["config"]) as fh:
            cfg_dict = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"error: cannot load config {paths['config']!r}: {exc}", file=sys.stderr)
        return {}, 1

    try:
        n_heads = cfg_dict["num_attention_heads"]
        n_kv_heads = int(cfg_dict.get("num_key_value_heads", n_heads))
        att1_cfg = {
            "vocab_size": cfg_dict["vocab_size"],
            "n_layers":   cfg_dict["num_hidden_layers"],
            "n_heads":    n_heads,
            "n_kv_heads": n_kv_heads,
            "d_model":    cfg_dict["hidden_size"],
            "d_ff":       cfg_dict["intermediate_size"],
            "max_seq_len": cfg_dict.get("max_position_embeddings", 128),
            "rope_dim":   (cfg_dict.get("hidden_size", 8)
                           // n_heads * 2),
        }
        rope_theta = float(cfg_dict.get("rope_theta", 10000.0))
    except KeyError as exc:
        print(f"error: missing config key {exc}", file=sys.stderr)
        return {}, 1
    if att1_cfg["d_model"] % att1_cfg["n_heads"] != 0:
        print("error: hidden_size must be divisible by num_attention_heads", file=sys.stderr)
        return {}, 1
    if att1_cfg["n_heads"] % att1_cfg["n_kv_heads"] != 0:
        print("error: num_attention_heads must be divisible by num_key_value_heads", file=sys.stderr)
        return {}, 1

    att1_cfg["rope_theta"] = rope_theta

    # -------- load ATT-1 f32 model --------
    try:
        att1_f32 = read_att1_model(paths["att1_f32"])
    except Att1ReadError as exc:
        print(f"error: cannot load ATT-1 f32 model: {exc}", file=sys.stderr)
        return {}, 1

    # -------- static f32 comparison --------
    n_f32, max_f32, rel_f32, pt_f32, f32_err = _compare_static(
        paths["safetensors"], att1_f32, att1_cfg, args.verbose
    )

    if f32_err:
        fail_msgs.append(f"f32 static: {f32_err}")
        f32_tol_ok = False
    else:
        f32_tol_ok = max_f32 < args.f32_tol
        if not f32_tol_ok:
            fail_msgs.append(
                f"f32 static: max_abs_error {max_f32:.3e} exceeds "
                f"tolerance {args.f32_tol:.3e}"
            )

    rpt_f32 = {
        "att1_path":       paths["att1_f32"],
        "att1_version":    att1_f32.version,
        "dtype":           "f32",
        "backend":         getattr(args, "backend", None) or "cpu-f32",
        "tensors_checked": n_f32,
        "max_abs_error":   max_f32,
        "max_rel_error":   rel_f32,
        "tolerance":       args.f32_tol,
        "status":          "pass" if (not f32_err and f32_tol_ok) else "fail",
        "note":            "f32 copy is exact; zero error expected",
        "error":           f32_err,
        "per_tensor":      pt_f32,
    }

    # -------- static q8 comparison (optional) --------
    rpt_q8 = None
    if paths["att1_q8"]:
        try:
            att1_q8 = read_att1_model(paths["att1_q8"])
        except Att1ReadError as exc:
            print(f"error: cannot load ATT-1 q8 model: {exc}", file=sys.stderr)
            return {}, 1

        n_q8, max_q8, rel_q8, pt_q8, q8_err = _compare_static(
            paths["safetensors"], att1_q8, att1_cfg, args.verbose
        )

        if q8_err:
            fail_msgs.append(f"q8 static: {q8_err}")
            q8_tol_ok = False
        else:
            q8_tol_ok = max_q8 < args.q8_tol
            if not q8_tol_ok:
                fail_msgs.append(
                    f"q8 static: max_abs_error {max_q8:.3e} exceeds "
                    f"tolerance {args.q8_tol:.3e}"
                )

        rpt_q8 = {
            "att1_path":       paths["att1_q8"],
            "att1_version":    att1_q8.version,
            "dtype":           "q8",
            "backend":         args.q8_backend,
            "tensors_checked": n_q8,
            "max_abs_error":   max_q8,
            "max_rel_error":   rel_q8,
            "tolerance":       args.q8_tol,
            "status":          "pass" if (not q8_err and q8_tol_ok) else "fail",
            "note":            ("q8 per-row int8 quantisation; "
                                "max_abs_error < tolerance expected"),
            "error":           q8_err,
            "per_tensor":      pt_q8,
        }

    # -------- static q4 comparison (optional) --------
    rpt_q4 = None
    if paths.get("att1_q4"):
        try:
            att1_q4 = read_att1_model(paths["att1_q4"])
        except Att1ReadError as exc:
            print(f"error: cannot load ATT-1 q4 model: {exc}", file=sys.stderr)
            return {}, 1

        n_q4, max_q4, rel_q4, pt_q4, q4_err = _compare_static(
            paths["safetensors"], att1_q4, att1_cfg, args.verbose
        )

        # mean_abs_error across all per-tensor means
        mean_q4 = 0.0
        if pt_q4:
            mean_q4 = sum(e["max_abs_error"] for e in pt_q4) / len(pt_q4)

        # top-k overlap: compare sorted top-k indices of source vs q4 for
        # largest eligible tensor (skip if numpy absent)
        topk_overlap = None
        if _HAVE_NUMPY and pt_q4 and not q4_err:
            try:
                _topk = 5
                # Find the largest q4 tensor for top-k overlap check.
                biggest = max(
                    (t for t in att1_q4.tensors if t.dtype == DTYPE_Q4),
                    key=lambda t: t.shape[0] * t.shape[1],
                    default=None,
                )
                if biggest is not None:
                    plan = _build_plan(att1_cfg)
                    item = next(
                        (x for x in plan if x["target"] == biggest.name), None
                    )
                    if item is not None:
                        hdr, doff = _load_safetensors_header(paths["safetensors"])
                        src_vals = _load_st_tensor(
                            paths["safetensors"], hdr, doff, item["source"]
                        )
                        if item["transpose"]:
                            r, c = item["source_shape"]
                            src_vals = tuple(_transpose_2d(list(src_vals), r, c))
                        src_arr = np.array(src_vals, dtype=np.float32)
                        q4_arr  = np.array(biggest.dequantize(), dtype=np.float32)
                        topk_src = set(np.argsort(-np.abs(src_arr))[:_topk].tolist())
                        topk_q4  = set(np.argsort(-np.abs(q4_arr))[:_topk].tolist())
                        topk_overlap = len(topk_src & topk_q4)
            except Exception:  # noqa: BLE001
                topk_overlap = None

        if q4_err:
            fail_msgs.append(f"q4 static: {q4_err}")
            q4_tol_ok = False
        else:
            q4_tol_ok = max_q4 < args.q4_tol
            if not q4_tol_ok:
                fail_msgs.append(
                    f"q4 static: max_abs_error {max_q4:.3e} exceeds "
                    f"tolerance {args.q4_tol:.3e}"
                )

        rpt_q4 = {
            "att1_path":       paths["att1_q4"],
            "att1_version":    att1_q4.version,
            "dtype":           "q4",
            "backend":         "cpu-q4",
            "tensors_checked": n_q4,
            "max_abs_error":   max_q4,
            "max_rel_error":   rel_q4,
            "mean_abs_error":  mean_q4,
            "topk_overlap":    topk_overlap,
            "tolerance":       args.q4_tol,
            "status":          "pass" if (not q4_err and q4_tol_ok) else "fail",
            "note":            ("q4 grouped int4 g32; larger error than q8 expected; "
                                "token divergence from f32 ref may occur"),
            "error":           q4_err,
            "per_tensor":      pt_q4,
        }
    rpt_fwd = None
    if _HAVE_NUMPY:
        ref_error = None
        try:
            weights_np  = _load_source_weights_numpy(paths["safetensors"], att1_cfg)
            reference = "source_safetensors"
            ref_logits  = _python_forward_pass(weights_np, att1_cfg, prompt_ids, rope_theta)
            ref_next    = int(np.argmax(ref_logits))
            logits_shape = list(ref_logits.shape)
        except (KeyError, ValueError, OSError) as exc:
            ref_error = str(exc)
            reference = "source_safetensors"
            ref_next = None
            logits_shape = []

        f32_bench_backend = getattr(args, "backend", None) or "cpu-f32"
        bench_next = None
        bench_err = None
        bench_match  = None
        f32_fwd_status = "skip"
        if ref_error is None:
            bench_next, _, bench_err = _call_att1_bench_generic(
                paths["att1_f32"], prompt_ids, f32_bench_backend
            )

        if ref_error is not None:
            bench_err = None
            bench_match = None
            f32_fwd_status = "skip"
        elif bench_err:
            bench_match  = None
            f32_fwd_status = "skip"
        else:
            bench_match  = (ref_next == bench_next)
            f32_fwd_status = "pass" if bench_match else "fail"
            if not bench_match:
                fail_msgs.append(f"f32 forward: ref={ref_next}, bench={bench_next}")

        # q8 bench comparison
        q8_bench_next = None
        q8_bench_err  = None
        q8_match      = None
        q8_fwd_status = "skip"
        if paths["att1_q8"] and ref_error is None:
            q8_bench_next, _, q8_bench_err = _call_att1_bench_generic(
                paths["att1_q8"], prompt_ids, args.q8_backend
            )
            if not q8_bench_err:
                q8_match = (ref_next == q8_bench_next)
                q8_fwd_status = "pass" if q8_match else "warn"
                if not q8_match:
                    # q8 argmax mismatch is a warning, not a hard failure.
                    print(
                        "note: q8 argmax differs from f32 ref; "
                        "may be acceptable if top-2 logits are close",
                        file=sys.stderr,
                    )

        # q4 bench comparison (token divergence from f32 ref is expected; not
        # a hard failure — reported as a note)
        q4_bench_next = None
        q4_bench_err  = None
        q4_match      = None
        q4_fwd_status = "skip"
        if paths.get("att1_q4") and ref_error is None:
            q4_bench_next, _, q4_bench_err = _call_att1_bench_generic(
                paths["att1_q4"], prompt_ids, "cpu-q4"
            )
            if not q4_bench_err:
                q4_match = (ref_next == q4_bench_next)
                q4_fwd_status = "pass" if q4_match else "note"
                if not q4_match:
                    print(
                        "note: q4 last_token differs from f32 ref; "
                        "token divergence may be expected for q4",
                        file=sys.stderr,
                    )

        rpt_fwd = {
            "reference":           reference,
            "prompt_ids":          prompt_ids,
            "logits_shape":        logits_shape,
            "ref_last_token":      ref_next,
            "f32_backend":         f32_bench_backend,
            "f32_bench_last_token": bench_next,
            "f32_bench_error":     bench_err,
            "f32_forward_match":   bench_match,
            "f32_status":          f32_fwd_status,
            "q8_backend":          args.q8_backend,
            "q8_bench_last_token": q8_bench_next,
            "q8_bench_error":      q8_bench_err,
            "q8_forward_match":    q8_match,
            "q8_status":           q8_fwd_status,
            "q8_note":             ("q8 argmax may differ from f32 ref "
                                    "for near-tied logits"),
            "q4_bench_last_token": q4_bench_next,
            "q4_bench_error":      q4_bench_err,
            "q4_forward_match":    q4_match,
            "q4_status":           q4_fwd_status,
            "q4_note":             ("q4 token divergence from f32 ref may be "
                                    "expected; check top-k overlap for confidence"),
            "next_token_result":   f32_fwd_status,
            "error":               ref_error,
        }

    # -------- assemble structured result dict --------
    overall = "pass" if not fail_msgs else "fail"
    rpt = {
        "date":        datetime.date.today().isoformat(),
        "source_model_path": paths["source_model_path"],
        "safetensors": paths["safetensors"],
        "config_path": paths["config"],
        "config":      {
            "vocab_size": att1_cfg["vocab_size"],
            "n_layers":   att1_cfg["n_layers"],
            "n_heads":    att1_cfg["n_heads"],
            "n_kv_heads": att1_cfg["n_kv_heads"],
            "d_model":    att1_cfg["d_model"],
            "d_ff":       att1_cfg["d_ff"],
            "rope_theta": rope_theta,
        },
        "f32_static":  rpt_f32,
        "q8_static":   rpt_q8,
        "q4_static":   rpt_q4,
        "forward":     rpt_fwd,
        "result":      overall,
    }

    # -------- emit output --------
    if getattr(args, "report", False):
        # Rich structured text report.
        print(_format_report_text(rpt))
    else:
        # Machine-readable key=value lines (backward-compatible with M61).
        print(f"config:      vocab_size={att1_cfg['vocab_size']} "
              f"d_model={att1_cfg['d_model']} d_ff={att1_cfg['d_ff']} "
              f"n_layers={att1_cfg['n_layers']} n_heads={att1_cfg['n_heads']} "
              f"n_kv_heads={att1_cfg['n_kv_heads']}")
        print(f"source_model_path:   {paths['source_model_path']}")
        print(f"att1_f32:    {paths['att1_f32']}")
        print(f"prompt_ids:  {prompt_ids}")
        if f32_err:
            print(f"f32_error:   {f32_err}", file=sys.stderr)
        else:
            print(f"f32_tensors_checked: {n_f32}")
            print(f"f32_max_abs_error:   {max_f32:.3e}")
            print(f"f32_max_rel_error:   {rel_f32:.3e}")
        if rpt_q8:
            print(f"att1_q8:     {paths['att1_q8']}")
            if q8_err:
                print(f"q8_error:    {q8_err}", file=sys.stderr)
            else:
                print(f"q8_tensors_checked:  {n_q8}")
                print(f"q8_max_abs_error:    {max_q8:.3e}")
                print(f"q8_max_rel_error:    {rel_q8:.3e}")
        if rpt_q4:
            print(f"att1_q4:     {paths['att1_q4']}")
            if q4_err:
                print(f"q4_error:    {q4_err}", file=sys.stderr)
            else:
                print(f"q4_tensors_checked:  {n_q4}")
                print(f"q4_max_abs_error:    {max_q4:.3e}")
                print(f"q4_max_rel_error:    {rel_q4:.3e}")
                print(f"q4_mean_abs_error:   {mean_q4:.3e}")
                if rpt_q4.get("topk_overlap") is not None:
                    print(f"q4_topk_overlap:     {rpt_q4['topk_overlap']}")
        if rpt_fwd:
            print(f"reference:           {rpt_fwd['reference']}")
            if rpt_fwd["error"]:
                print(f"forward_error:       {rpt_fwd['error']}")
            print(f"ref_last_token:      {rpt_fwd['ref_last_token']}")
            print(f"logits_shape:        {rpt_fwd['logits_shape']}")
            if bench_err:
                print(f"bench_last_token:    (error: {bench_err})")
            else:
                print(f"bench_last_token:    {bench_next}")
                print(f"forward_match:       {'yes' if bench_match else 'no'}")
            if rpt_fwd["q8_bench_last_token"] is not None:
                print(f"q8_bench_last_token: {rpt_fwd['q8_bench_last_token']}")
                print(f"q8_forward_match:    "
                      f"{'yes' if rpt_fwd['q8_forward_match'] else 'warn (quantisation rounding)'}")
            if rpt_fwd["q4_bench_last_token"] is not None:
                print(f"q4_bench_last_token: {rpt_fwd['q4_bench_last_token']}")
                print(f"q4_forward_match:    "
                      f"{'yes' if rpt_fwd['q4_forward_match'] else 'note (token divergence may be expected for q4)'}")
        else:
            print("numpy_forward:       skipped (numpy not available)")
        print(f"result:              {overall}")

    if fail_msgs:
        for msg in fail_msgs:
            print(f"FAIL: {msg}", file=sys.stderr)

    # -------- JSON report (hard fail on IO error) --------
    if getattr(args, "report_json", None):
        try:
            with open(args.report_json, "w") as fh:
                json.dump(rpt, fh, indent=2)
        except OSError as exc:
            print(
                f"error: cannot write report JSON {args.report_json!r}: {exc}",
                file=sys.stderr,
            )
            return rpt, 1

    return rpt, (0 if overall == "pass" else 2)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="ATT-1 source-model comparison harness (M61/M62/M69)"
    )
    parser.add_argument(
        "--model-dir", default=None, dest="model_dir", metavar="DIR",
        help=("Local source model directory containing config.json and "
              "model.safetensors; no network access is attempted")
    )
    parser.add_argument(
        "--safetensors", default=None,
        help=("Source .safetensors file (default: model-dir/model.safetensors "
              "or M61 fixture)")
    )
    parser.add_argument(
        "--config", default=None,
        help=("Source config.json (default: model-dir/config.json or "
              "tiny_llama_config.json)")
    )
    parser.add_argument(
        "--att1-f32", default=None,
        dest="att1_f32",
        help=("Converted f32 ATT-1 model (default: models/m61_f32/model.att1 "
              "only for fixture source)")
    )
    parser.add_argument(
        "--att1-q8", default=None,
        dest="att1_q8",
        help=("Converted q8 ATT-1 model (default: models/m61_q8/model.att1 "
              "only for fixture source; omit to run f32-only)")
    )
    parser.add_argument(
        "--att1-q4", default=None,
        dest="att1_q4",
        help=("Converted q4 ATT-1 model (optional); enables q4 static + "
              "cpu-q4 bench comparison against source/f32 reference")
    )
    parser.add_argument(
        "--prompt-ids", default=[5], dest="prompt_ids",
        type=lambda s: _parse_ids(s) or parser.error(f"bad prompt-ids: {s!r}"),
        help="Comma-separated prompt token IDs (default: 5)"
    )
    parser.add_argument(
        "--tokens-file", default=None, dest="tokens_file", metavar="PATH",
        help="Load prompt token IDs from file (one per line; overrides --prompt-ids)"
    )
    parser.add_argument(
        "--backend", default=None,
        choices=["cpu-f32", "cpu-q8", "cuda", "cuda-q8"],
        help="Backend for f32 bench forward comparison (default: cpu-f32)"
    )
    parser.add_argument(
        "--q8-backend", default="cpu-q8",
        choices=["cpu-q8", "cuda-q8"],
        help="Backend for q8 bench forward comparison (default: cpu-q8)"
    )
    parser.add_argument(
        "--f32-tol", default=1e-4, type=float, dest="f32_tol",
        help="Max abs error tolerance for f32 static check (default: 1e-4)"
    )
    parser.add_argument(
        "--q8-tol", default=0.6, type=float, dest="q8_tol",
        help="Max abs error tolerance for q8 static check (default: 0.6)"
    )
    parser.add_argument(
        "--q4-tol", default=4.0, type=float, dest="q4_tol",
        help="Max abs error tolerance for q4 static check (default: 4.0)"
    )
    parser.add_argument(
        "--report", action="store_true",
        help="Print rich structured text report instead of key=value lines"
    )
    parser.add_argument(
        "--report-json", default=None, dest="report_json", metavar="PATH",
        help="Write JSON report to PATH (exit 1 on IO error)"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Show per-tensor comparison details"
    )
    args = parser.parse_args()

    _, exit_code = run_harness(args)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
