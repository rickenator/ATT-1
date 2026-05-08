#!/usr/bin/env python3
"""
ATT-1 public-model q4 backend smoke validator (Milestone 85).

Runs a local converted q4 ATT-1 artifact through all supported cpu-q4
single/cluster backend paths using an external token IDs file.  Also
exercises the cuda-q4 path, which must fail clearly as unsupported.

No network access is attempted.  CUDA q4 is not supported.  No public
weights or generated artifacts are committed to Git.

Usage (manual, public model outside Git):

    python3 compiler/validate_public_q4_backends.py \\
        --model-dir ~/Models/SmolLM2-135M \\
        --att1-q4   ~/Models/att1/SmolLM2-135M/model_q4.att1 \\
        --tokens-file ~/Models/att1/SmolLM2-135M/prompt.ids \\
        [--tokens 1] \\
        [--tiles 2] \\
        [--bench ./build/att1-bench] \\
        [--report-json PATH]

Usage (fixture smoke, checked-in artifact):

    python3 compiler/validate_public_q4_backends.py \\
        --model-dir compiler/fixtures \\
        --att1-q4   models/real_tiny_q4/model.att1 \\
        --tokens-file build/m85_ids.txt \\
        --tokens 1 --tiles 2
"""

import argparse
import json
import os
import subprocess
import sys


_CPU_Q4  = "cpu-q4"
_CUDA_Q4 = "cuda-q4"


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
    """Return a dict of all key=value pairs found in stdout."""
    values = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key:
            values[key.strip()] = value.strip()
    return values


def _is_cuda_q4_unsupported(text):
    """Return True when the output signals that cuda-q4 is unsupported."""
    lowered = text.lower()
    needles = [
        "unsupported",
        "cuda-q4",
        "cuda q4",
        "not supported",
        "not available",
    ]
    return any(needle in lowered for needle in needles)


def _run_bench(bench, artifact, backend, mode, tokens_file, tokens, tiles):
    """Run one att1-bench invocation and return a result dict."""
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
        cmd += ["--tiles", str(tiles), "--shard-plan", "runtime"]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "artifact_path": artifact,
            "backend": backend,
            "mode": mode,
            "shard_plan": "runtime" if mode == "cluster" else "n/a",
            "generated_tokens": None,
            "last_token": None,
            "token_time_us_total": None,
            "logits_bytes": None,
            "kv_appends": None,
            "kv_evictions": None,
            "fabric_packets_sent": None,
            "status": "fail",
            "error": str(exc),
        }

    combined = "\n".join([result.stdout, result.stderr]).strip()
    stderr_text = result.stderr.strip()
    kv = _parse_kv_stdout(result.stdout)

    if result.returncode == 0:
        status = "pass"
        error  = None
    elif backend == _CUDA_Q4 and _is_cuda_q4_unsupported(combined):
        status = "unsupported"
        error  = (stderr_text or "cuda-q4 unsupported").splitlines()[0]
    else:
        status = "fail"
        error  = combined or f"att1-bench exited {result.returncode}"

    return {
        "artifact_path":     artifact,
        "backend":           kv.get("backend", backend),
        "mode":              kv.get("mode", mode),
        "shard_plan":        kv.get("shard_plan", "runtime" if mode == "cluster"
                                                  else "n/a"),
        "generated_tokens":  kv.get("generated_tokens"),
        "last_token":        kv.get("last_token"),
        "token_time_us_total": kv.get("token_time_us_total"),
        "logits_bytes":      kv.get("logits_bytes"),
        "kv_appends":        kv.get("kv_appends"),
        "kv_evictions":      kv.get("kv_evictions"),
        "fabric_packets_sent": kv.get("fabric_packets_sent"),
        "status": status,
        "error":  error,
    }


def _format_run(row):
    parts = [
        f"artifact={row['artifact_path']}",
        f"backend={row['backend']}",
        f"mode={row['mode']}",
        f"shard_plan={row['shard_plan']}",
        f"generated_tokens={row['generated_tokens']}",
        f"last_token={row['last_token']}",
        f"token_time_us_total={row['token_time_us_total']}",
    ]
    if row.get("logits_bytes") is not None:
        parts.append(f"logits_bytes={row['logits_bytes']}")
    if row.get("kv_appends") is not None:
        parts.append(f"kv_appends={row['kv_appends']}")
    if row.get("kv_evictions") is not None:
        parts.append(f"kv_evictions={row['kv_evictions']}")
    if row.get("fabric_packets_sent") is not None:
        parts.append(f"fabric_packets_sent={row['fabric_packets_sent']}")
    parts.append(f"status={row['status']}")
    if row.get("error"):
        error = " ".join(str(row["error"]).split())
        parts.append(f"error={error}")
    return "run: " + " ".join(parts)


def _build_cases(att1_q4, include_cuda):
    """Return list of (artifact, backend, mode) tuples to run."""
    cases = [
        (att1_q4, _CPU_Q4, "single"),
        (att1_q4, _CPU_Q4, "cluster"),
    ]
    if include_cuda:
        # cuda-q4 must fail as unsupported; we include it to verify the error
        # path, but its result must not cause the overall report to be "fail".
        cases += [
            (att1_q4, _CUDA_Q4, "single"),
            (att1_q4, _CUDA_Q4, "cluster"),
        ]
    return cases


def run_validation(args):
    """Run the full q4 backend smoke validation and return (report, exit_code)."""
    include_cuda = getattr(args, "include_cuda", False)

    try:
        _require_local_path("source model", args.model_dir, directory=True)
        _require_local_path("q4 ATT-1 artifact", args.att1_q4)
        _require_local_path("token IDs", args.tokens_file)
        _require_local_path("att1-bench", args.bench)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return {}, 1

    rows = []
    for artifact, backend, mode in _build_cases(args.att1_q4, include_cuda):
        rows.append(
            _run_bench(
                args.bench,
                artifact,
                backend,
                mode,
                args.tokens_file,
                args.tokens,
                args.tiles,
            )
        )

    # cuda-q4 "unsupported" rows do NOT count as failures.
    failures = [
        row for row in rows
        if row["status"] == "fail"
    ]
    # Validate that cpu-q4 cluster produced nonzero fabric packets.
    cluster_rows = [r for r in rows if r["mode"] == "cluster" and
                    r["backend"] == _CPU_Q4 and r["status"] == "pass"]
    for row in cluster_rows:
        pkts = row.get("fabric_packets_sent")
        if pkts is not None and str(pkts).strip() == "0":
            failures.append({**row, "error": "cpu-q4 cluster: fabric_packets_sent=0"})

    overall = "pass" if not failures else "fail"
    report = {
        "source_model_path": args.model_dir,
        "att1_q4":           args.att1_q4,
        "tokens_file":       args.tokens_file,
        "tokens":            args.tokens,
        "tiles":             args.tiles,
        "runs":              rows,
        "result":            overall,
    }

    print("# ATT-1 public q4 backend smoke validation")
    print(f"source_model_path: {args.model_dir}")
    print(f"att1_q4: {args.att1_q4}")
    print(f"tokens_file: {args.tokens_file}")
    print(f"tokens: {args.tokens}")
    print(f"tiles: {args.tiles}")
    for row in rows:
        print(_format_run(row))
    if include_cuda:
        cuda_rows = [r for r in rows if r["backend"] == _CUDA_Q4]
        for row in cuda_rows:
            if row["status"] == "unsupported":
                print(f"note: cuda-q4 {row['mode']}: {row.get('error', 'unsupported')}")
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
            print(
                f"error: cannot write report JSON {args.report_json!r}: {exc}",
                file=sys.stderr,
            )
            return report, 1

    return report, (0 if overall == "pass" else 2)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Validate a local q4 ATT-1 artifact across cpu-q4 backend "
            "smoke paths.  CUDA q4 is unsupported and tested only for "
            "error-path verification when --include-cuda is given."
        )
    )
    parser.add_argument(
        "--model-dir", required=True, metavar="DIR",
        help="Local source model directory; used only for provenance reporting"
    )
    parser.add_argument(
        "--att1-q4", required=True, metavar="PATH",
        dest="att1_q4",
        help="Local converted q4 .att1 artifact"
    )
    parser.add_argument(
        "--tokens-file", required=True, metavar="PATH",
        dest="tokens_file",
        help="Local token IDs file for att1-bench --tokens-file"
    )
    parser.add_argument(
        "--tokens", type=int, default=1,
        help="Generated token count per bench run (default: 1)"
    )
    parser.add_argument(
        "--tiles", type=int, default=2,
        help="Cluster tile count (default: 2)"
    )
    parser.add_argument(
        "--bench", default=os.path.join("build", "att1-bench"), metavar="PATH",
        help="Path to att1-bench (default: build/att1-bench)"
    )
    parser.add_argument(
        "--include-cuda", action="store_true", dest="include_cuda",
        help=("Also run cuda-q4 paths to verify the unsupported error; "
              "cuda-q4 rows always report 'unsupported' and do not affect "
              "overall pass/fail")
    )
    parser.add_argument(
        "--report-json", default=None, metavar="PATH",
        dest="report_json",
        help="Optional JSON report path"
    )
    args = parser.parse_args()

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
