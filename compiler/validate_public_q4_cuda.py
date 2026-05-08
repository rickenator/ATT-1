#!/usr/bin/env python3
"""
ATT-1 public-model q4 CUDA validation report (Milestone 91).

Runs a local converted q4 ATT-1 artifact through all q4 backend paths
(cpu-q4 and cuda-q4, single and cluster, runtime and metadata shard plans),
producing a comprehensive validation report.

- cuda-q4 paths are attempted; on a CPU-only host they report 'unavailable'
  and do not count as failures.
- Metadata shard plan is attempted for cpu-q4 cluster; q4 rejects it with
  ATT1_ERR_UNSUPPORTED and that outcome is reported as 'plan_unsupported',
  not 'fail'.
- Backend name in output is verified: cuda-q4 rows must not silently fall
  back to cpu-q4, cuda-q8, cpu-q8, or f32.
- Fabric packet counters are verified nonzero for all passing cluster rows.

Notes embedded in the report:
  - cuda-q4 logits are expected to match cpu-q4 within Q4_TOLERANCE=0.35f.
  - Generated tokens may diverge between cpu-q4 and cuda-q4 due to
    floating-point order effects; divergence is a note, not a failure.
  - Metadata shard plan is not supported for q4 (ATT1_ERR_UNSUPPORTED);
    status=plan_unsupported is not a failure.

No network access is attempted.  No public weights or generated artifacts are
committed to Git.

Usage (public model, external paths):

    python3 compiler/validate_public_q4_cuda.py \\
        --model-dir ~/Models/SmolLM2-135M \\
        --att1-q4   ~/Models/att1/SmolLM2-135M/model_q4.att1 \\
        --tokens-file ~/Models/att1/SmolLM2-135M/prompt.ids \\
        [--tokens 1] \\
        [--tiles 2] \\
        [--bench ./build/att1-bench] \\
        [--report-json PATH]

Usage (fixture smoke, checked-in artifact):

    python3 compiler/validate_public_q4_cuda.py \\
        --model-dir compiler/fixtures \\
        --att1-q4   models/real_tiny_q4/model.att1 \\
        --tokens-file build/m91_ids.txt \\
        --tokens 1 --tiles 2
"""

import argparse
import json
import os
import subprocess
import sys

_CPU_Q4 = "cpu-q4"
_CUDA_Q4 = "cuda-q4"

# Backends that a cuda-q4 invocation must NOT report (silent fallback check).
_FORBIDDEN_FALLBACKS = frozenset(("cpu-q4", "cpu-q8", "cpu-f32", "cuda", "cuda-q8"))

_Q4_NOTES = (
    "note: cuda-q4 logits are expected to match cpu-q4 within"
    " Q4_TOLERANCE=0.35f",
    "note: generated-token divergence between cpu-q4 and cuda-q4 is expected"
    " due to fp-order effects; divergence is a note, not a failure",
    "note: metadata shard plan is not supported for q4 (ATT1_ERR_UNSUPPORTED);"
    " status=plan_unsupported is not a failure",
)


def _is_url_like(path):
    if path is None:
        return False
    lowered = str(path).lower()
    return "://" in lowered or lowered.startswith(("hf:", "hf://"))


def _require_local_path(label, path, *, directory=False):
    if not path:
        raise ValueError(f"{label} path is required")
    if _is_url_like(path):
        raise ValueError(f"{label} must be a local filesystem path: {path!r}")
    exists = os.path.isdir(path) if directory else os.path.isfile(path)
    if not exists:
        kind = "directory" if directory else "file"
        raise ValueError(f"{label} {kind} not found: {path!r}")


def _parse_kv_stdout(text):
    values = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key:
            values[key.strip()] = value.strip()
    return values


def _is_metadata_plan_unsupported(text):
    """Return True when stderr signals q4 metadata plan rejection."""
    return "metadata shard plan not supported with q4" in text


def _is_cuda_unavailable(text):
    """Return True when stderr signals cuda-q4 is unavailable on this host."""
    lowered = text.lower()
    return (
        "cuda-q4 backend unavailable" in lowered
        or "backend unsupported or unavailable" in lowered
        or ("cuda" in lowered and "not available" in lowered)
    )


def _run_bench(bench, artifact, backend, mode, shard_plan,
               tokens_file, tokens, tiles):
    """Run one att1-bench invocation; return a result dict."""
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
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return _make_row(artifact, backend, mode, shard_plan,
                         status="fail", error=str(exc))

    stderr_text = proc.stderr.strip()
    kv = _parse_kv_stdout(proc.stdout)

    if proc.returncode == 0:
        # Backend-name verification: reject silent fallback.
        reported = kv.get("backend", "")
        if backend == _CUDA_Q4 and reported and reported in _FORBIDDEN_FALLBACKS:
            return _make_row(artifact, backend, mode, shard_plan,
                             status="fail",
                             error=f"silent fallback: reported backend={reported!r}",
                             kv=kv)
        status = "pass"
        error = None
    elif _is_metadata_plan_unsupported(stderr_text):
        status = "plan_unsupported"
        error = "metadata shard plan not supported with q4 (ATT1_ERR_UNSUPPORTED)"
    elif backend == _CUDA_Q4 and _is_cuda_unavailable(stderr_text):
        status = "unavailable"
        error = (stderr_text.splitlines()[0] if stderr_text
                 else "cuda-q4 backend unavailable")
    else:
        combined = "\n".join(filter(None, [proc.stdout, proc.stderr])).strip()
        status = "fail"
        error = combined or f"att1-bench exited {proc.returncode}"

    return _make_row(artifact, backend, mode, shard_plan,
                     status=status, error=error, kv=kv)


def _make_row(artifact, backend, mode, shard_plan,
              *, status, error, kv=None):
    kv = kv or {}
    return {
        "artifact_path":       artifact,
        "backend":             kv.get("backend", backend),
        "mode":                mode,
        "shard_plan":          kv.get("shard_plan",
                                      shard_plan if mode == "cluster" else "n/a"),
        "tokenizer_mode":      "external",
        "generated_tokens":    kv.get("generated_tokens"),
        "last_token":          kv.get("last_token"),
        "token_time_us_total": kv.get("token_time_us_total"),
        "logits_bytes":        kv.get("logits_bytes"),
        "kv_appends":          kv.get("kv_appends"),
        "kv_evictions":        kv.get("kv_evictions"),
        "fabric_packets_sent": kv.get("fabric_packets_sent"),
        "status":              status,
        "error":               error,
    }


def _build_cases(att1_q4):
    """
    Return (artifact, backend, mode, shard_plan) tuples to run.

    Metadata plan is tested for cpu-q4 cluster only; q4 always rejects it
    (ATT1_ERR_UNSUPPORTED) and the result is reported as 'plan_unsupported'.
    cuda-q4 cluster metadata is omitted: the metadata plan rejection fires
    before CUDA availability is checked, mixing two rejection reasons.
    """
    return [
        (att1_q4, _CPU_Q4,  "single",  "n/a"),
        (att1_q4, _CPU_Q4,  "cluster", "runtime"),
        (att1_q4, _CPU_Q4,  "cluster", "metadata"),
        (att1_q4, _CUDA_Q4, "single",  "n/a"),
        (att1_q4, _CUDA_Q4, "cluster", "runtime"),
    ]


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
    ]
    for key in ("logits_bytes", "kv_appends", "kv_evictions",
                "fabric_packets_sent"):
        if row.get(key) is not None:
            parts.append(f"{key}={row[key]}")
    parts.append(f"status={row['status']}")
    if row.get("error"):
        parts.append(f"error={' '.join(str(row['error']).split())}")
    return "run: " + " ".join(parts)


def run_validation(args):
    """Run the full CUDA q4 validation; return (report_dict, exit_code)."""
    try:
        _require_local_path("source model", args.model_dir, directory=True)
        _require_local_path("q4 ATT-1 artifact", args.att1_q4)
        _require_local_path("token IDs", args.tokens_file)
        _require_local_path("att1-bench", args.bench)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return {}, 1

    rows = []
    for artifact, backend, mode, shard_plan in _build_cases(args.att1_q4):
        rows.append(_run_bench(
            args.bench, artifact, backend, mode, shard_plan,
            args.tokens_file, args.tokens, args.tiles,
        ))

    # Hard failures: only status=fail counts.
    # status=unavailable (CUDA absent) and status=plan_unsupported are
    # expected outcomes and do not fail the overall report.
    hard_failures = [r for r in rows if r["status"] == "fail"]

    # Fabric packet check: all passing cluster rows must have nonzero packets.
    for row in rows:
        if row["mode"] == "cluster" and row["status"] == "pass":
            pkts = row.get("fabric_packets_sent")
            if pkts is not None and str(pkts).strip() == "0":
                hard_failures.append({
                    **row,
                    "status": "fail",
                    "error": f"{row['backend']} cluster: fabric_packets_sent=0",
                })

    overall = "pass" if not hard_failures else "fail"
    report = {
        "source_model_path": args.model_dir,
        "att1_q4":           args.att1_q4,
        "tokens_file":       args.tokens_file,
        "tokens":            args.tokens,
        "tiles":             args.tiles,
        "runs":              rows,
        "result":            overall,
    }

    print("# ATT-1 public q4 CUDA validation report (M91)")
    print(f"source_model_path: {args.model_dir}")
    print(f"att1_q4: {args.att1_q4}")
    print(f"tokens_file: {args.tokens_file}")
    print(f"tokens: {args.tokens}")
    print(f"tiles: {args.tiles}")
    for row in rows:
        print(_format_row(row))
    for note in _Q4_NOTES:
        print(note)
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


def main():
    ap = argparse.ArgumentParser(
        description=(
            "ATT-1 public-model q4 CUDA validation report (M91). "
            "Runs cpu-q4 and cuda-q4 single/cluster paths on a local q4 "
            "ATT-1 artifact. cuda-q4 rows report 'unavailable' on a "
            "CPU-only host; metadata plan rows report 'plan_unsupported'; "
            "neither is a failure."
        )
    )
    ap.add_argument("--model-dir", required=True, metavar="DIR",
                    help="Local source model directory (provenance only)")
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

    _, exit_code = run_validation(args)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
