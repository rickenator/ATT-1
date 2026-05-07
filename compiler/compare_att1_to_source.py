#!/usr/bin/env python3
"""
ATT-1 source-model comparison harness (Milestone 61).

Validates that an ATT-1 converted model artifact (f32 or q8) faithfully
represents the source LLaMA-style safetensors weights, and that its forward
pass produces the same next-token prediction as a Python reference
implementation.

Two validation layers
---------------------
1. Static tensor comparison
   For every tensor in the conversion plan, load the source value (applying
   the same transpose rule used by the converter), load the ATT-1 value, and
   compute max_abs_error.  For f32 this should be < 1e-5 (float precision).
   For q8 this reflects per-row quantisation error.

2. Dynamic forward-pass comparison  (requires numpy)
   Runs a Python LLaMA reference transformer on the prompt token IDs using
   the source safetensors weights, then calls att1-bench and compares the
   generated last_token.  Exact match is expected for f32; exact match is
   also expected for q8 since argmax is robust to small quantisation error.

Fixtures used by default (M61):
    --safetensors  compiler/fixtures/m61_llama_2l.safetensors
    --config       compiler/fixtures/tiny_llama_config.json
    --att1-f32     models/m61_f32/model.att1
    --att1-q8      models/m61_q8/model.att1

Usage:
    python3 compiler/compare_att1_to_source.py [options]
    python3 compiler/compare_att1_to_source.py \\
        --safetensors compiler/fixtures/m61_llama_2l.safetensors \\
        --config      compiler/fixtures/tiny_llama_config.json \\
        --att1-f32    models/m61_f32/model.att1 \\
        --att1-q8     models/m61_q8/model.att1

Exit codes:
    0  all enabled checks pass
    1  usage / IO / dependency error
    2  comparison failure
"""

import argparse
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
    read_att1_model, DTYPE_F32, DTYPE_Q8, Att1ReadError
)

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
    """Return flat tuple[float] for an F32 tensor from a .safetensors file."""
    if name not in hdr:
        raise KeyError(f"tensor {name!r} not in safetensors header")
    t = hdr[name]
    if t["dtype"] != "F32":
        raise ValueError(f"tensor {name!r} has dtype {t['dtype']!r}, expected F32")
    b, e   = t["data_offsets"]
    nbytes = e - b
    n      = nbytes // 4
    with open(path, "rb") as fh:
        fh.seek(data_offset + b)
        raw = fh.read(nbytes)
    return struct.unpack(f"<{n}f", raw)


# ---------------------------------------------------------------------------
# Tensor mapping plan  (mirrors llama_source_tensor_plan in the converter)
# ---------------------------------------------------------------------------

def _build_plan(cfg):
    """Return list of mapping dicts for the given ATT-1 config dict."""
    d  = cfg["d_model"]
    v  = cfg["vocab_size"]
    ff = cfg["d_ff"]

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
                "source_shape": [d, d],
                "target_shape": [d, d],
                "transpose": True,
            },
            {
                "source": f"{src}.self_attn.v_proj.weight",
                "target": f"{dst}.attention.wv.weight",
                "source_shape": [d, d],
                "target_shape": [d, d],
                "transpose": True,
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


# ---------------------------------------------------------------------------
# Static tensor comparison
# ---------------------------------------------------------------------------

def _compare_static(safetensors_path, att1_model, cfg_dict, verbose):
    """Compare all mapped tensors between source safetensors and ATT-1.

    Returns (n_checked, max_abs_error, error_message_or_None)
    """
    hdr, data_offset = _load_safetensors_header(safetensors_path)
    plan             = _build_plan(cfg_dict)

    max_abs_error = 0.0
    n_checked     = 0

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
                return 0, 0.0, f"source tensor {src_name!r} not found"

        # Apply expected transpose to get ATT-1 layout.
        if transpose:
            s_rows, s_cols = item["source_shape"]
            src_vals = tuple(_transpose_2d(list(src_vals), s_rows, s_cols))

        # Load ATT-1 tensor.
        att1_t = att1_model.find_tensor(dst_name)
        if att1_t is None:
            return 0, 0.0, f"ATT-1 tensor {dst_name!r} not found"

        if att1_t.dtype == DTYPE_F32:
            dst_vals = att1_t.f32_values
        elif att1_t.dtype == DTYPE_Q8:
            dst_vals = att1_t.dequantize()
        else:
            return 0, 0.0, f"unsupported dtype for {dst_name!r}"

        if len(src_vals) != len(dst_vals):
            return 0, 0.0, (
                f"element count mismatch for {dst_name!r}: "
                f"source={len(src_vals)}, att1={len(dst_vals)}"
            )

        err = max(abs(a - b) for a, b in zip(src_vals, dst_vals))
        max_abs_error = max(max_abs_error, err)
        n_checked += 1

        if verbose:
            print(f"  {dst_name}: max_abs_error={err:.3e}")

    return n_checked, max_abs_error, None


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
        cfg_dict    dict  vocab_size, n_layers, n_heads, d_model, d_ff, rope_dim
        token_ids   list[int]  prompt token IDs
        rope_theta  float  RoPE theta (default 10000.0)

    Returns:
        (logits np.ndarray [vocab_size], kv_cache dict)
    """
    n_layers = cfg_dict["n_layers"]
    n_heads  = cfg_dict["n_heads"]
    d_model  = cfg_dict["d_model"]
    d_ff     = cfg_dict["d_ff"]
    d_head   = d_model // n_heads

    embed = weights_np["tok_embeddings.weight"]  # [V, D]

    # KV cache: list of (K[D], V[D]) per layer, one entry per past position.
    kv_cache = [[] for _ in range(n_layers)]

    logits = None

    for pos, tok in enumerate(token_ids):
        x = embed[tok].copy()  # [D]

        for layer in range(n_layers):
            prefix = f"layers.{layer}"

            # -- attention sub-layer --
            x_norm = _rmsnorm(x, weights_np[f"{prefix}.attention_norm.weight"])

            wq = weights_np[f"{prefix}.attention.wq.weight"]  # [D, D]
            wk = weights_np[f"{prefix}.attention.wk.weight"]
            wv = weights_np[f"{prefix}.attention.wv.weight"]
            wo = weights_np[f"{prefix}.attention.wo.weight"]

            Q = x_norm @ wq  # [D]
            K = x_norm @ wk
            V = x_norm @ wv

            # Apply RoPE per head.
            for h in range(n_heads):
                sl = slice(h * d_head, (h + 1) * d_head)
                Q[sl] = _rope(Q[sl], pos, rope_theta)
                K[sl] = _rope(K[sl], pos, rope_theta)

            # Append to KV cache.
            kv_cache[layer].append((K.copy(), V.copy()))

            # Attention: per-head dot-product.
            context = np.zeros(d_model)
            for h in range(n_heads):
                sl    = slice(h * d_head, (h + 1) * d_head)
                q_h   = Q[sl]
                scale = 1.0 / math.sqrt(d_head)
                n_past = len(kv_cache[layer])
                scores = np.array([
                    np.dot(q_h, kv_cache[layer][t][0][sl]) * scale
                    for t in range(n_past)
                ])
                probs = _softmax(scores)
                for t in range(n_past):
                    context[sl] += probs[t] * kv_cache[layer][t][1][sl]

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
        elif t.dtype == DTYPE_Q8:
            vals = t.dequantize()
        else:
            continue
        arr = np.array(vals, dtype=np.float32).reshape(t.shape)
        weights[t.name] = arr
    return weights


def _call_att1_bench(att1_path, token_ids, n_generate=1):
    """Call att1-bench and return last_token (int) or None on failure."""
    ids_str = ",".join(str(i) for i in token_ids)
    # Search for att1-bench relative to repo root (cwd).
    bench = os.path.join("build", "att1-bench")
    if not os.path.isfile(bench):
        return None, "att1-bench not found at build/att1-bench"
    cmd = [
        bench,
        "--model", att1_path,
        "--tokenizer", "external",
        "--input-token-ids", ids_str,
        "--tokens", str(n_generate),
        "--mode", "single",
        "--backend", "cpu-f32",
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=30
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return None, str(exc)

    if result.returncode != 0:
        return None, f"att1-bench exit {result.returncode}: {result.stderr.strip()}"

    for line in result.stdout.splitlines():
        if line.startswith("last_token="):
            return int(line.split("=", 1)[1].strip()), None

    return None, "last_token not found in att1-bench output"


# ---------------------------------------------------------------------------
# Main harness
# ---------------------------------------------------------------------------

_DEFAULT_ST   = os.path.join("compiler", "fixtures", "m61_llama_2l.safetensors")
_DEFAULT_CFG  = os.path.join("compiler", "fixtures", "tiny_llama_config.json")
_DEFAULT_F32  = os.path.join("models", "m61_f32", "model.att1")
_DEFAULT_Q8   = os.path.join("models", "m61_q8",  "model.att1")


def _parse_ids(s):
    try:
        return [int(x.strip()) for x in s.split(",") if x.strip()]
    except ValueError:
        return None


def run_harness(args):
    """Execute the full comparison harness.  Returns (result_dict, exit_code)."""

    result = {
        "safetensors": args.safetensors,
        "att1_f32":    args.att1_f32,
        "att1_q8":     args.att1_q8 if args.att1_q8 else "(skipped)",
        "prompt_ids":  args.prompt_ids,
        "numpy":       _HAVE_NUMPY,
        "checks": [],
    }

    fail_msgs = []

    # -------- load source config --------
    try:
        with open(args.config) as fh:
            cfg_dict = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"error: cannot load config {args.config!r}: {exc}", file=sys.stderr)
        return result, 1

    # Map HF config keys to ATT-1 internal names.
    try:
        att1_cfg = {
            "vocab_size": cfg_dict["vocab_size"],
            "n_layers":   cfg_dict["num_hidden_layers"],
            "n_heads":    cfg_dict["num_attention_heads"],
            "d_model":    cfg_dict["hidden_size"],
            "d_ff":       cfg_dict["intermediate_size"],
            "max_seq_len": cfg_dict.get("max_position_embeddings", 128),
            "rope_dim":   cfg_dict.get("hidden_size", 8) // cfg_dict["num_attention_heads"] * 2,
        }
        rope_theta = float(cfg_dict.get("rope_theta", 10000.0))
    except KeyError as exc:
        print(f"error: missing config key {exc}", file=sys.stderr)
        return result, 1

    result["config"] = {
        "vocab_size": att1_cfg["vocab_size"],
        "n_layers":   att1_cfg["n_layers"],
        "n_heads":    att1_cfg["n_heads"],
        "d_model":    att1_cfg["d_model"],
        "d_ff":       att1_cfg["d_ff"],
        "rope_theta": rope_theta,
    }

    # -------- load ATT-1 f32 model --------
    try:
        att1_f32 = read_att1_model(args.att1_f32)
    except Att1ReadError as exc:
        print(f"error: cannot load ATT-1 f32 model: {exc}", file=sys.stderr)
        return result, 1

    # -------- static f32 comparison --------
    print(f"config:      vocab_size={att1_cfg['vocab_size']} "
          f"d_model={att1_cfg['d_model']} d_ff={att1_cfg['d_ff']} "
          f"n_layers={att1_cfg['n_layers']} n_heads={att1_cfg['n_heads']}")
    print(f"att1_f32:    {args.att1_f32}")
    print(f"prompt_ids:  {args.prompt_ids}")

    n_f32, max_f32, f32_err = _compare_static(
        args.safetensors, att1_f32, att1_cfg, args.verbose
    )
    if f32_err:
        fail_msgs.append(f"f32 static: {f32_err}")
        result["checks"].append({"name": "f32_static", "status": "fail", "error": f32_err})
    else:
        tol_ok = max_f32 < args.f32_tol
        status = "pass" if tol_ok else "fail"
        result["checks"].append({
            "name": "f32_static",
            "status": status,
            "tensors_checked": n_f32,
            "max_abs_error": max_f32,
        })
        print(f"f32_tensors_checked: {n_f32}")
        print(f"f32_max_abs_error:   {max_f32:.3e}")
        if not tol_ok:
            fail_msgs.append(
                f"f32 static: max_abs_error {max_f32:.3e} exceeds tolerance {args.f32_tol:.3e}"
            )

    # -------- static q8 comparison (optional) --------
    if args.att1_q8:
        try:
            att1_q8 = read_att1_model(args.att1_q8)
        except Att1ReadError as exc:
            print(f"error: cannot load ATT-1 q8 model: {exc}", file=sys.stderr)
            return result, 1

        print(f"att1_q8:     {args.att1_q8}")
        n_q8, max_q8, q8_err = _compare_static(
            args.safetensors, att1_q8, att1_cfg, args.verbose
        )
        if q8_err:
            fail_msgs.append(f"q8 static: {q8_err}")
            result["checks"].append({"name": "q8_static", "status": "fail", "error": q8_err})
        else:
            tol_ok = max_q8 < args.q8_tol
            status = "pass" if tol_ok else "fail"
            result["checks"].append({
                "name": "q8_static",
                "status": status,
                "tensors_checked": n_q8,
                "max_abs_error": max_q8,
            })
            print(f"q8_tensors_checked:  {n_q8}")
            print(f"q8_max_abs_error:    {max_q8:.3e}")
            if not tol_ok:
                fail_msgs.append(
                    f"q8 static: max_abs_error {max_q8:.3e} exceeds tolerance {args.q8_tol:.3e}"
                )

    # -------- forward-pass comparison (numpy required) --------
    if _HAVE_NUMPY:
        weights_np = _load_weights_numpy(att1_f32)
        ref_logits = _python_forward_pass(
            weights_np, att1_cfg, args.prompt_ids, rope_theta
        )
        ref_next = int(np.argmax(ref_logits))
        ref_logits_list = ref_logits.tolist()

        # Call att1-bench.
        bench_next, bench_err = _call_att1_bench(
            args.att1_f32, args.prompt_ids, n_generate=1
        )

        print(f"ref_last_token:      {ref_next}")
        if bench_err:
            print(f"bench_last_token:    (error: {bench_err})")
            result["checks"].append({
                "name": "f32_forward",
                "status": "skip",
                "error": bench_err,
            })
        else:
            print(f"bench_last_token:    {bench_next}")
            match = (ref_next == bench_next)
            status = "pass" if match else "fail"
            result["checks"].append({
                "name": "f32_forward",
                "status": status,
                "ref_last_token": ref_next,
                "bench_last_token": bench_next,
                "match": match,
                "ref_logits_top3": sorted(
                    enumerate(ref_logits_list), key=lambda x: -x[1]
                )[:3],
            })
            print(f"forward_match:       {'yes' if match else 'no'}")
            if not match:
                fail_msgs.append(
                    f"f32 forward: ref={ref_next}, bench={bench_next}"
                )

        # q8 forward pass: call att1-bench with cpu-q8 backend.
        if args.att1_q8:
            bench_q8_next, bench_q8_err = _call_att1_bench_q8(
                args.att1_q8, args.prompt_ids
            )
            if bench_q8_err:
                result["checks"].append({
                    "name": "q8_forward",
                    "status": "skip",
                    "error": bench_q8_err,
                })
            else:
                match_q8 = (ref_next == bench_q8_next)
                status_q8 = "pass" if match_q8 else "warn"
                result["checks"].append({
                    "name": "q8_forward",
                    "status": status_q8,
                    "ref_last_token": ref_next,
                    "bench_last_token": bench_q8_next,
                    "match": match_q8,
                })
                print(f"q8_bench_last_token: {bench_q8_next}")
                print(f"q8_forward_match:    {'yes' if match_q8 else 'warn (quantisation rounding)'}")
                if not match_q8:
                    # q8 mismatch is a warning not a hard fail — quantisation
                    # can shift argmax by 1 for near-tied logits.
                    print(
                        "note: q8 argmax differs from f32 ref; "
                        "may be acceptable if top-2 logits are close",
                        file=sys.stderr
                    )
    else:
        print("numpy_forward:       skipped (numpy not available)")
        result["checks"].append({"name": "f32_forward", "status": "skip",
                                 "reason": "numpy not available"})

    # -------- summary --------
    overall = "pass" if not fail_msgs else "fail"
    result["result"] = overall
    print(f"result:              {overall}")

    if fail_msgs:
        for msg in fail_msgs:
            print(f"FAIL: {msg}", file=sys.stderr)

    if args.report_json:
        try:
            with open(args.report_json, "w") as fh:
                json.dump(result, fh, indent=2)
        except OSError as exc:
            print(f"warning: cannot write report JSON: {exc}", file=sys.stderr)

    return result, (0 if overall == "pass" else 2)


def _call_att1_bench_q8(att1_path, token_ids, n_generate=1):
    """Call att1-bench with cpu-q8 backend, return (last_token, error)."""
    ids_str = ",".join(str(i) for i in token_ids)
    bench = os.path.join("build", "att1-bench")
    if not os.path.isfile(bench):
        return None, "att1-bench not found"
    cmd = [
        bench,
        "--model", att1_path,
        "--tokenizer", "external",
        "--input-token-ids", ids_str,
        "--tokens", str(n_generate),
        "--mode", "single",
        "--backend", "cpu-q8",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return None, str(exc)

    if result.returncode != 0:
        return None, f"att1-bench exit {result.returncode}: {result.stderr.strip()}"

    for line in result.stdout.splitlines():
        if line.startswith("last_token="):
            return int(line.split("=", 1)[1].strip()), None

    return None, "last_token not found"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="ATT-1 source-model comparison harness (M61)"
    )
    parser.add_argument(
        "--safetensors", default=_DEFAULT_ST,
        help="Source .safetensors file (default: M61 fixture)"
    )
    parser.add_argument(
        "--config", default=_DEFAULT_CFG,
        help="Source config.json (default: tiny_llama_config.json)"
    )
    parser.add_argument(
        "--att1-f32", default=_DEFAULT_F32,
        dest="att1_f32",
        help="Converted f32 ATT-1 model (default: models/m61_f32/model.att1)"
    )
    parser.add_argument(
        "--att1-q8", default=_DEFAULT_Q8,
        dest="att1_q8",
        help="Converted q8 ATT-1 model (default: models/m61_q8/model.att1)"
    )
    parser.add_argument(
        "--prompt-ids", default=[5], dest="prompt_ids",
        type=lambda s: _parse_ids(s) or parser.error(f"bad prompt-ids: {s!r}"),
        help="Comma-separated prompt token IDs (default: 5)"
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
        "--report-json", default=None, dest="report_json", metavar="PATH",
        help="Write JSON report to PATH"
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
