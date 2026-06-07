#!/usr/bin/env python3
"""
ATT-1 public-model q4 conversion and validation (Milestone 83).

Converts a local public LLaMA-style model directory to a q4 .att1 artifact,
inspects it, runs cpu-q4 single and cluster bench paths, and optionally
compares the last generated token against existing f32 and q8 artifacts.

No network access is attempted.  CUDA q4 is not supported.
No public weights or generated artifacts are committed to Git.

Usage:

    python3 compiler/validate_public_q4.py \\
        --model-dir ~/Models/SmolLM2-135M \\
        --out ~/Models/att1/SmolLM2-135M/model_q4.att1 \\
        --tokens-file ~/Models/att1/SmolLM2-135M/prompt.ids \\
        [--att1-f32 ~/Models/att1/SmolLM2-135M/model_f32.att1] \\
        [--att1-q8  ~/Models/att1/SmolLM2-135M/model_q8.att1] \\
        [--bench ./build/att1-bench] \\
        [--report-json PATH]
"""

import argparse
import json
import os
import subprocess
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT   = os.path.dirname(_SCRIPT_DIR)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_url_like(path):
    if path is None:
        return False
    s = str(path).lower()
    return "://" in s or s.startswith(("hf:", "hf://"))


def _require_local_file(label, path):
    if not path:
        raise ValueError(f"{label} path is required")
    if _is_url_like(path):
        raise ValueError(f"{label} must be a local filesystem path: {path!r}")
    if not os.path.isfile(path):
        raise ValueError(f"{label} file not found: {path!r}")


def _require_local_dir(label, path):
    if not path:
        raise ValueError(f"{label} path is required")
    if _is_url_like(path):
        raise ValueError(f"{label} must be a local filesystem path: {path!r}")
    if not os.path.isdir(path):
        raise ValueError(f"{label} directory not found: {path!r}")


def _run(cmd, *, capture=True):
    """Run a command; return (returncode, stdout_text)."""
    result = subprocess.run(
        cmd,
        shell=isinstance(cmd, str),
        capture_output=capture,
        text=True,
    )
    stdout = (result.stdout or "") + (result.stderr or "")
    return result.returncode, stdout


def _parse_kv(text):
    kv = {}
    for line in text.splitlines():
        line = line.strip()
        if "=" in line:
            k, v = line.split("=", 1)
            if k:
                kv[k.strip()] = v.strip()
    return kv


def _fabric_packets_nonzero(kv):
    val = kv.get("fabric_packets_sent", "0")
    try:
        return int(val) > 0
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# Validation steps
# ---------------------------------------------------------------------------

def step_compat_check(model_dir):
    """Run check_llama_compat.py; return (ok, report_text)."""
    script = os.path.join(_SCRIPT_DIR, "check_llama_compat.py")
    rc, out = _run([sys.executable, script, "--model-dir", model_dir])
    ok = rc == 0 and "compat: pass" in out
    return ok, out


def step_convert_q4(model_dir, out_path, *, tiles=1):
    """Convert model directory to q4 .att1 via converter."""
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    script = os.path.join(_SCRIPT_DIR, "convert_llama_to_att1.py")
    cmd = [
        sys.executable, script,
        "--model-dir", model_dir,
        "--weight-format", "q4",
        "--tiles", str(tiles),
        "--out", out_path,
    ]
    rc, out = _run(cmd)
    ok = rc == 0 and os.path.isfile(out_path)
    return ok, out


def step_inspect(bench_dir, att1_path):
    """Run att1-inspect and return (ok, output_text)."""
    inspect = os.path.join(bench_dir, "att1-inspect")
    rc, out = _run([inspect, att1_path])
    ok = rc == 0
    return ok, out


def step_bench(bench_bin, att1_path, tokens_file, *, mode, backend, tiles=1, n_tokens=2):
    """Run att1-bench and return (ok, kv_dict, raw_text)."""
    cmd = [
        bench_bin,
        "--model", att1_path,
        "--tokenizer", "external",
        "--tokens-file", tokens_file,
        "--tokens", str(n_tokens),
        "--mode", mode,
        "--backend", backend,
    ]
    if mode == "cluster":
        cmd += ["--tiles", str(tiles)]
    rc, out = _run(cmd)
    kv = _parse_kv(out)
    ok = rc == 0
    return ok, kv, out


def step_compare_tokens(bench_bin, att1_ref_path, att1_q4_path, tokens_file,
                        backend_ref, *, mode, tiles=1):
    """Run both models and compare last_token output."""
    ok_ref, kv_ref, _  = step_bench(bench_bin, att1_ref_path, tokens_file,
                                      mode=mode, backend=backend_ref, tiles=tiles)
    ok_q4,  kv_q4, _   = step_bench(bench_bin, att1_q4_path, tokens_file,
                                      mode=mode, backend="cpu-q4", tiles=tiles)
    if not ok_ref or not ok_q4:
        return False, None, None, "bench failed"
    tok_ref = kv_ref.get("last_token")
    tok_q4  = kv_q4.get("last_token")
    match = tok_ref == tok_q4
    return True, tok_ref, tok_q4, "match" if match else "diverge"


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def format_report(rows, *, summary):
    lines = ["ATT-1 public-model q4 validation report", ""]
    for label, status, detail in rows:
        tag = "PASS" if status else "FAIL"
        lines.append(f"  [{tag}] {label}")
        if detail:
            for d in detail.splitlines()[:6]:
                lines.append(f"         {d}")
    lines += ["", f"result: {summary}"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Validate public-model q4 conversion (Milestone 83)."
    )
    ap.add_argument("--model-dir", required=True, metavar="DIR",
                    help="Local model directory (config.json + model.safetensors)")
    ap.add_argument("--out", required=True, metavar="PATH",
                    help="Output q4 .att1 artifact path (outside Git)")
    ap.add_argument("--tokens-file", required=True, metavar="PATH",
                    help="One token ID per line; used for all bench calls")
    ap.add_argument("--att1-f32", metavar="PATH", default=None,
                    help="Optional: existing f32 .att1 artifact for token comparison")
    ap.add_argument("--att1-q8", metavar="PATH", default=None,
                    help="Optional: existing q8 .att1 artifact for token comparison")
    ap.add_argument("--tiles", type=int, default=2, metavar="N",
                    help="Tile count for cluster bench (default: 2)")
    ap.add_argument("--bench", metavar="DIR",
                    default=os.path.join(_REPO_ROOT, "build"),
                    help="Directory containing att1-bench / att1-inspect binaries")
    ap.add_argument("--report-json", metavar="PATH", default=None,
                    help="Write JSON report to PATH")
    args = ap.parse_args()

    bench_bin = os.path.join(args.bench, "att1-bench")

    # Validate inputs
    try:
        _require_local_dir("--model-dir", args.model_dir)
        _require_local_file("--tokens-file", args.tokens_file)
        if args.att1_f32:
            _require_local_file("--att1-f32", args.att1_f32)
        if args.att1_q8:
            _require_local_file("--att1-q8", args.att1_q8)
        if not os.path.isfile(bench_bin):
            raise ValueError(f"att1-bench not found: {bench_bin!r} — run 'make' first")
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)

    rows = []
    all_ok = True

    # 1. Compatibility check
    ok, detail = step_compat_check(args.model_dir)
    rows.append(("compat check", ok, detail.strip()))
    if not ok:
        all_ok = False
        print(format_report(rows, summary="fail"))
        sys.exit(1)

    # 2. Convert to q4
    ok, detail = step_convert_q4(args.model_dir, args.out, tiles=args.tiles)
    rows.append(("q4 conversion", ok, detail.strip()))
    if not ok:
        all_ok = False
        print(format_report(rows, summary="fail"))
        sys.exit(1)

    # 3. Inspect
    ok, detail = step_inspect(args.bench, args.out)
    kv_inspect = _parse_kv(detail)
    q4_tensors = sum(1 for l in detail.splitlines() if "dtype_name=q4" in l)
    inspect_detail = (
        f"tensor_count={kv_inspect.get('tensor_count', '?')}"
        f"  q4_tensors={q4_tensors}"
        f"  quant=grouped-q4-g32 present={'yes' if 'grouped-q4-g32' in detail else 'no'}"
    )
    ok = ok and "dtype_name=q4" in detail and "grouped-q4-g32" in detail
    rows.append(("att1-inspect q4 fields", ok, inspect_detail))
    if not ok:
        all_ok = False

    # 4. cpu-q4 single bench
    ok, kv, raw = step_bench(bench_bin, args.out, args.tokens_file,
                              mode="single", backend="cpu-q4")
    single_detail = (
        f"mode={kv.get('mode','?')}  backend={kv.get('backend','?')}"
        f"  last_token={kv.get('last_token','?')}"
        f"  generated_tokens={kv.get('generated_tokens','?')}"
    )
    rows.append(("cpu-q4 single bench", ok, single_detail))
    if not ok:
        all_ok = False

    # 5. cpu-q4 cluster bench
    ok, kv, raw = step_bench(bench_bin, args.out, args.tokens_file,
                              mode="cluster", backend="cpu-q4", tiles=args.tiles)
    packets_ok = _fabric_packets_nonzero(kv)
    cluster_detail = (
        f"mode={kv.get('mode','?')}  backend={kv.get('backend','?')}"
        f"  last_token={kv.get('last_token','?')}"
        f"  fabric_packets_sent={kv.get('fabric_packets_sent','?')}"
    )
    ok = ok and packets_ok
    rows.append(("cpu-q4 cluster bench", ok, cluster_detail))
    if not ok:
        all_ok = False

    # 6. Token comparison: f32 vs q4 (optional)
    if args.att1_f32:
        ok, tok_f32, tok_q4, verdict = step_compare_tokens(
            bench_bin, args.att1_f32, args.out, args.tokens_file,
            backend_ref="cpu-f32", mode="single")
        comp_detail = f"f32_last_token={tok_f32}  q4_last_token={tok_q4}  verdict={verdict}"
        rows.append(("f32 vs q4 token comparison", ok, comp_detail))
        if not ok:
            all_ok = False

    # 7. Token comparison: q8 vs q4 (optional)
    if args.att1_q8:
        ok, tok_q8, tok_q4, verdict = step_compare_tokens(
            bench_bin, args.att1_q8, args.out, args.tokens_file,
            backend_ref="cpu-q8", mode="single")
        comp_detail = f"q8_last_token={tok_q8}  q4_last_token={tok_q4}  verdict={verdict}"
        rows.append(("q8 vs q4 token comparison", ok, comp_detail))
        if not ok:
            all_ok = False

    summary = "pass" if all_ok else "fail"
    print(format_report(rows, summary=summary))

    if args.report_json:
        rpt = {
            "model_dir": args.model_dir,
            "q4_artifact": args.out,
            "steps": [
                {"label": label, "pass": status, "detail": detail}
                for label, status, detail in rows
            ],
            "result": summary,
        }
        rj_dir = os.path.dirname(args.report_json)
        if rj_dir:
            os.makedirs(rj_dir, exist_ok=True)
        with open(args.report_json, "w", encoding="utf-8") as f:
            json.dump(rpt, f, indent=2)
            f.write("\n")
        print(f"wrote report → {args.report_json}")

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
