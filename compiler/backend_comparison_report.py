#!/usr/bin/env python3
"""
ATT-1 backend comparison report (Milestone 92).

Runs att1-bench across f32/q8/q4 × CPU/CUDA × single/cluster combinations
and produces a summary table plus per-row details.

- CUDA rows require --include-cuda; on a CPU-only host they report 'pending'.
- q4 rows that encounter metadata-plan rejection report 'plan_unsupported'.
- No row failure causes the overall result to be 'fail' unless a
  *required* (non-CUDA) row exits unexpectedly.
- JSON report via --report-json.

No network access is attempted.  No public model weights or generated
artifacts are committed to Git.

Usage (checked-in fixtures, CPU-only smoke):

    python3 compiler/backend_comparison_report.py \\
        --att1-f32 models/real_tiny_f32/model.att1 \\
        --att1-q8  models/real_tiny_q8/model.att1 \\
        --att1-q4  models/real_tiny_q4/model.att1 \\
        --tokens-file build/m92_ids.txt \\
        --tokens 1 --tiles 2

Usage (public model, external paths):

    python3 compiler/backend_comparison_report.py \\
        --att1-f32 ~/Models/att1/SmolLM2-135M/model_f32.att1 \\
        --att1-q8  ~/Models/att1/SmolLM2-135M/model_q8.att1 \\
        --att1-q4  ~/Models/att1/SmolLM2-135M/model_q4.att1 \\
        --tokens-file ~/Models/att1/SmolLM2-135M/prompt.ids \\
        --tokens 1 --tiles 2 --include-cuda \\
        --report-json ~/Models/att1/SmolLM2-135M/m92_report.json
"""

import argparse
import json
import os
import subprocess
import sys

# Backend families
_CPU_F32  = "cpu-f32"
_CPU_Q8   = "cpu-q8"
_CPU_Q4   = "cpu-q4"
_CUDA_F32 = "cuda"
_CUDA_Q8  = "cuda-q8"
_CUDA_Q4  = "cuda-q4"

# Backends that are always CUDA-only
_CUDA_BACKENDS = frozenset((_CUDA_F32, _CUDA_Q8, _CUDA_Q4))

# Forbidden silent-fallback backends when running a CUDA backend
_FORBIDDEN_FALLBACKS = frozenset(
    (_CPU_F32, _CPU_Q8, _CPU_Q4, "cuda", "cuda-q8")
)

_Q4_NOTES = (
    "note: q4 logits are expected to match f32 within Q4_TOLERANCE=0.35f",
    "note: generated-token divergence between q4 and f32/q8 is expected",
    "note: q4 metadata shard plan is unsupported (ATT1_ERR_UNSUPPORTED)",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_url_like(path):
    if path is None:
        return False
    s = str(path).lower()
    return "://" in s or s.startswith(("hf:", "hf://"))


def _require_local_path(label, path, *, directory=False):
    if not path:
        raise ValueError(f"{label} path is required")
    if _is_url_like(path):
        raise ValueError(f"{label} must be a local filesystem path: {path!r}")
    exists = os.path.isdir(path) if directory else os.path.isfile(path)
    if not exists:
        kind = "directory" if directory else "file"
        raise ValueError(f"{label} {kind} not found: {path!r}")


def _parse_kv(text):
    out = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or "=" not in line:
            continue
        k, v = line.split("=", 1)
        if k.strip():
            out[k.strip()] = v.strip()
    return out


def _is_cuda_unavailable(stderr):
    lo = stderr.lower()
    return (
        "cuda-q4 backend unavailable" in lo
        or "cuda-q8 backend unavailable" in lo
        or "backend unsupported or unavailable" in lo
        or ("cuda" in lo and "not available" in lo)
        or "error: cuda" in lo
    )


def _is_plan_unsupported(stderr):
    return "metadata shard plan not supported with q4" in stderr


# ---------------------------------------------------------------------------
# Case definition
# ---------------------------------------------------------------------------

def _build_cases(att1_f32, att1_q8, att1_q4, include_cuda):
    """
    Return list of dicts describing each bench invocation.

    Fields: artifact, backend, mode, shard_plan, required
      required=True  → failure causes overall report to fail
      required=False → CUDA-conditional; 'pending' when not --include-cuda,
                       'unavailable' on a CPU-only host even with --include-cuda
    """
    def _case(artifact, backend, mode, shard_plan="n/a", required=True):
        return dict(artifact=artifact, backend=backend, mode=mode,
                    shard_plan=shard_plan, required=required)

    cases = [
        _case(att1_f32, _CPU_F32, "single"),
        _case(att1_f32, _CPU_F32, "cluster", "runtime"),
        _case(att1_q8,  _CPU_Q8,  "single"),
        _case(att1_q8,  _CPU_Q8,  "cluster", "runtime"),
        _case(att1_q4,  _CPU_Q4,  "single"),
        _case(att1_q4,  _CPU_Q4,  "cluster", "runtime"),
    ]

    if include_cuda:
        cuda_cases = [
            _case(att1_f32, _CUDA_F32, "single",  required=False),
            _case(att1_f32, _CUDA_F32, "cluster", "runtime", required=False),
            _case(att1_q8,  _CUDA_Q8,  "single",  required=False),
            _case(att1_q8,  _CUDA_Q8,  "cluster", "runtime", required=False),
            _case(att1_q4,  _CUDA_Q4,  "single",  required=False),
            _case(att1_q4,  _CUDA_Q4,  "cluster", "runtime", required=False),
        ]
        cases.extend(cuda_cases)
    else:
        # Add pending placeholder rows so the table is always complete.
        pending_cases = [
            _case(att1_f32, _CUDA_F32, "single",  required=False),
            _case(att1_f32, _CUDA_F32, "cluster", "runtime", required=False),
            _case(att1_q8,  _CUDA_Q8,  "single",  required=False),
            _case(att1_q8,  _CUDA_Q8,  "cluster", "runtime", required=False),
            _case(att1_q4,  _CUDA_Q4,  "single",  required=False),
            _case(att1_q4,  _CUDA_Q4,  "cluster", "runtime", required=False),
        ]
        cases.extend(pending_cases)

    return cases


# ---------------------------------------------------------------------------
# Running a single bench case
# ---------------------------------------------------------------------------

def _run_case(bench, case, tokens_file, tokens, tiles, pending):
    """Run att1-bench for one case dict; return result dict."""
    artifact  = case["artifact"]
    backend   = case["backend"]
    mode      = case["mode"]
    shard_plan = case["shard_plan"]

    if pending:
        return _make_row(case, status="pending",
                         error="--include-cuda not set; CUDA validation pending")

    cmd = [
        bench,
        "--model", artifact,
        "--tokenizer", "external",
        "--tokens-file", tokens_file,
        "--tokens", str(tokens),
        "--mode", mode,
        "--backend", backend,
    ]
    if mode == "cluster":
        cmd += ["--tiles", str(tiles), "--shard-plan", shard_plan]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return _make_row(case, status="fail", error=str(exc))

    stderr = proc.stderr.strip()
    kv = _parse_kv(proc.stdout)

    if proc.returncode == 0:
        # Silent-fallback check for CUDA backends.
        reported = kv.get("backend", "")
        if backend in _CUDA_BACKENDS and reported and reported in _FORBIDDEN_FALLBACKS:
            return _make_row(case, status="fail",
                             error=f"silent fallback: reported backend={reported!r}",
                             kv=kv)
        status = "pass"
        error = None
    elif _is_plan_unsupported(stderr):
        status = "plan_unsupported"
        error = "metadata shard plan not supported with q4 (ATT1_ERR_UNSUPPORTED)"
    elif backend in _CUDA_BACKENDS and _is_cuda_unavailable(stderr):
        status = "unavailable"
        error = (stderr.splitlines()[0] if stderr else "CUDA backend unavailable")
    else:
        combined = "\n".join(filter(None, [proc.stdout, proc.stderr])).strip()
        status = "fail"
        error = combined or f"att1-bench exited {proc.returncode}"

    return _make_row(case, status=status, error=error, kv=kv)


def _make_row(case, *, status, error, kv=None):
    kv = kv or {}
    return {
        "artifact_path":        case["artifact"],
        "backend":              kv.get("backend", case["backend"]),
        "mode":                 case["mode"],
        "shard_plan":           kv.get("shard_plan",
                                       case["shard_plan"]
                                       if case["mode"] == "cluster" else "n/a"),
        "tokenizer_mode":       kv.get("tokenizer", "external"),
        "prompt_tokens":        kv.get("prompt_tokens"),
        "generated_tokens":     kv.get("generated_tokens"),
        "last_token":           kv.get("last_token"),
        "token_time_us_total":  kv.get("token_time_us_total"),
        "token_time_us_max":    kv.get("token_time_us_max"),
        "logits_bytes":         kv.get("logits_bytes_produced"),
        "kv_appends":           kv.get("kv_appends"),
        "kv_key_reads":         kv.get("kv_key_reads"),
        "fabric_packets_sent":  kv.get("fabric_packets_sent"),
        "status":               status,
        "error":                error,
        "required":             case["required"],
    }


# ---------------------------------------------------------------------------
# Post-run checks
# ---------------------------------------------------------------------------

def _check_fabric_packets(rows):
    """
    Return list of extra failure rows for passing cluster entries with
    fabric_packets_sent=0 on required (non-CUDA) backends.
    """
    extra = []
    for row in rows:
        if row["mode"] != "cluster":
            continue
        if row["status"] != "pass":
            continue
        if not row["required"]:
            continue
        pkts = row.get("fabric_packets_sent")
        if pkts is not None and str(pkts).strip() == "0":
            extra.append({**row,
                          "status": "fail",
                          "error": f"{row['backend']} cluster: fabric_packets_sent=0"})
    return extra


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def _format_row(row):
    parts = [
        f"artifact={row['artifact_path']}",
        f"backend={row['backend']}",
        f"mode={row['mode']}",
        f"shard_plan={row['shard_plan']}",
        f"tokenizer={row['tokenizer_mode']}",
        f"generated_tokens={row['generated_tokens']}",
        f"last_token={row['last_token']}",
        f"token_time_us_total={row['token_time_us_total']}",
        f"token_time_us_max={row['token_time_us_max']}",
    ]
    for key in ("logits_bytes", "kv_appends", "fabric_packets_sent"):
        if row.get(key) is not None:
            parts.append(f"{key}={row[key]}")
    parts.append(f"status={row['status']}")
    if row.get("error"):
        parts.append(f"error={' '.join(str(row['error']).split())}")
    return "run: " + " ".join(parts)


def _summary_table(rows):
    """Return a list of summary lines (one per row, tab-aligned)."""
    lines = []
    header = f"{'backend':<12} {'mode':<8} {'shard_plan':<10} {'status':<18} {'last_token':<12} {'time_us'}"
    lines.append(header)
    lines.append("-" * len(header))
    for row in rows:
        status = row["status"]
        last   = row.get("last_token") or "-"
        time_  = row.get("token_time_us_total") or "-"
        sp     = row.get("shard_plan") or "n/a"
        lines.append(
            f"{row['backend']:<12} {row['mode']:<8} {sp:<10} {status:<18} {str(last):<12} {time_}"
        )
    return lines


# ---------------------------------------------------------------------------
# Main validation logic
# ---------------------------------------------------------------------------

def run_comparison(args):
    """Run all cases and return (report_dict, exit_code)."""
    include_cuda = getattr(args, "include_cuda", False)

    try:
        _require_local_path("f32 ATT-1 artifact", args.att1_f32)
        _require_local_path("q8 ATT-1 artifact", args.att1_q8)
        _require_local_path("q4 ATT-1 artifact", args.att1_q4)
        _require_local_path("token IDs", args.tokens_file)
        _require_local_path("att1-bench", args.bench)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return {}, 1

    cases = _build_cases(args.att1_f32, args.att1_q8, args.att1_q4, include_cuda)

    rows = []
    for case in cases:
        is_cuda = case["backend"] in _CUDA_BACKENDS
        # Pending when CUDA not requested
        pending = is_cuda and not include_cuda
        rows.append(_run_case(args.bench, case, args.tokens_file,
                               args.tokens, args.tiles, pending))

    extra_failures = _check_fabric_packets(rows)

    # Only required rows that status=fail count toward the overall result.
    hard_failures = [
        r for r in rows
        if r["required"] and r["status"] == "fail"
    ] + extra_failures

    overall = "pass" if not hard_failures else "fail"

    report = {
        "att1_f32":     args.att1_f32,
        "att1_q8":      args.att1_q8,
        "att1_q4":      args.att1_q4,
        "tokens_file":  args.tokens_file,
        "tokens":       args.tokens,
        "tiles":        args.tiles,
        "include_cuda": include_cuda,
        "runs":         rows,
        "result":       overall,
    }

    # --- Print report ---
    print("# ATT-1 backend comparison report (M92)")
    print(f"att1_f32: {args.att1_f32}")
    print(f"att1_q8:  {args.att1_q8}")
    print(f"att1_q4:  {args.att1_q4}")
    print(f"tokens_file: {args.tokens_file}")
    print(f"tokens: {args.tokens}  tiles: {args.tiles}")
    print()
    for line in _summary_table(rows):
        print(line)
    print()
    for row in rows:
        print(_format_row(row))
    print()
    for note in _Q4_NOTES:
        print(note)
    if not include_cuda:
        print("note: CUDA rows are pending; re-run with --include-cuda on a CUDA host")
    print(f"result: {overall}")
    print("report: ok")

    if getattr(args, "report_json", None):
        try:
            out_dir = os.path.dirname(args.report_json)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            with open(args.report_json, "w", encoding="utf-8") as fh:
                json.dump(report, fh, indent=2)
                fh.write("\n")
        except OSError as exc:
            print(f"error: cannot write JSON {args.report_json!r}: {exc}",
                  file=sys.stderr)
            return report, 1

    return report, (0 if overall == "pass" else 2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=(
            "ATT-1 backend comparison report (M92). Runs all f32/q8/q4 × "
            "CPU/CUDA × single/cluster bench combinations and summarises the "
            "results. CUDA rows require --include-cuda; without it they are "
            "marked 'pending'."
        )
    )
    ap.add_argument("--att1-f32", required=True, metavar="PATH",
                    dest="att1_f32",
                    help="Local converted f32 .att1 artifact")
    ap.add_argument("--att1-q8", required=True, metavar="PATH",
                    dest="att1_q8",
                    help="Local converted q8 .att1 artifact")
    ap.add_argument("--att1-q4", required=True, metavar="PATH",
                    dest="att1_q4",
                    help="Local converted q4 .att1 artifact")
    ap.add_argument("--tokens-file", required=True, metavar="PATH",
                    dest="tokens_file",
                    help="Local token IDs file for att1-bench --tokens-file")
    ap.add_argument("--tokens", type=int, default=1,
                    help="Generated token count per bench run (default: 1)")
    ap.add_argument("--tiles", type=int, default=2,
                    help="Cluster tile count (default: 2)")
    ap.add_argument("--bench", default=os.path.join("build", "att1-bench"),
                    metavar="PATH",
                    help="Path to att1-bench (default: build/att1-bench)")
    ap.add_argument("--include-cuda", action="store_true",
                    dest="include_cuda",
                    help=("Run CUDA backend rows; on a CPU-only host they "
                          "report 'unavailable' (not failure)"))
    ap.add_argument("--report-json", default=None, metavar="PATH",
                    dest="report_json",
                    help="Optional JSON report path")
    args = ap.parse_args()

    if args.tokens < 1:
        print("error: --tokens must be >= 1", file=sys.stderr)
        sys.exit(1)
    if args.tiles < 1:
        print("error: --tiles must be >= 1", file=sys.stderr)
        sys.exit(1)

    _, exit_code = run_comparison(args)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
