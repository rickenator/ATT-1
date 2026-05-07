#!/usr/bin/env python3
"""
ATT-1 public-model backend smoke validator (Milestone 70).

Runs local converted f32/q8 ATT-1 artifacts through the supported
single/cluster backend paths using an external token IDs file.  The script is
for manual validation of public models whose source files and generated
artifacts live outside Git.

No network access is attempted.  CUDA paths are optional: when att1-bench
reports a CUDA backend as unsupported or unavailable, the row is reported as
unsupported and does not fail the validation.
"""

import argparse
import json
import os
import subprocess
import sys


_CPU_F32 = "cpu-f32"
_CPU_Q8 = "cpu-q8"
_CUDA_F32 = "cuda"
_CUDA_Q8 = "cuda-q8"


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
        if key and value:
            values[key] = value
    return values


def _is_cuda_unavailable(backend, text):
    if backend not in (_CUDA_F32, _CUDA_Q8):
        return False
    lowered = text.lower()
    needles = [
        "backend unsupported or unavailable",
        "cuda unavailable",
        "cuda not available",
        "unsupported",
    ]
    return any(needle in lowered for needle in needles)


def _run_bench(bench, artifact, backend, mode, tokens_file, tokens, tiles):
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
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "artifact_path": artifact,
            "backend": backend,
            "mode": mode,
            "shard_plan": "runtime",
            "generated_tokens": None,
            "last_token": None,
            "token_time_us_total": None,
            "status": "fail",
            "error": str(exc),
        }

    stderr_text = result.stderr.strip()
    combined = "\n".join([result.stdout, result.stderr]).strip()
    kv = _parse_kv_stdout(result.stdout)

    if result.returncode == 0:
        status = "pass"
        error = None
    elif _is_cuda_unavailable(backend, combined):
        status = "unsupported"
        error = stderr_text or f"{backend} unsupported or unavailable"
    else:
        status = "fail"
        error = combined or f"att1-bench exited {result.returncode}"

    return {
        "artifact_path": artifact,
        "backend": kv.get("backend", backend),
        "mode": kv.get("mode", mode),
        "shard_plan": kv.get("shard_plan", "runtime"),
        "generated_tokens": kv.get("generated_tokens"),
        "last_token": kv.get("last_token"),
        "token_time_us_total": kv.get("token_time_us_total"),
        "status": status,
        "error": error,
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
        f"status={row['status']}",
    ]
    if row.get("error"):
        error = " ".join(str(row["error"]).split())
        parts.append(f"error={error}")
    return "run: " + " ".join(parts)


def _build_cases(att1_f32, att1_q8):
    return [
        (att1_f32, _CPU_F32, "single"),
        (att1_f32, _CPU_F32, "cluster"),
        (att1_f32, _CUDA_F32, "single"),
        (att1_f32, _CUDA_F32, "cluster"),
        (att1_q8, _CPU_Q8, "single"),
        (att1_q8, _CPU_Q8, "cluster"),
        (att1_q8, _CUDA_Q8, "single"),
        (att1_q8, _CUDA_Q8, "cluster"),
    ]


def run_validation(args):
    try:
        _require_local_path("source model", args.model_dir, directory=True)
        _require_local_path("f32 ATT-1 artifact", args.att1_f32)
        _require_local_path("q8 ATT-1 artifact", args.att1_q8)
        _require_local_path("token IDs", args.tokens_file)
        _require_local_path("att1-bench", args.bench)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return {}, 1

    rows = []
    for artifact, backend, mode in _build_cases(args.att1_f32, args.att1_q8):
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

    failures = [row for row in rows if row["status"] == "fail"]
    overall = "pass" if not failures else "fail"
    report = {
        "source_model_path": args.model_dir,
        "tokens_file": args.tokens_file,
        "tokens": args.tokens,
        "tiles": args.tiles,
        "runs": rows,
        "result": overall,
    }

    print("# ATT-1 public backend smoke validation")
    print(f"source_model_path: {args.model_dir}")
    print(f"tokens_file: {args.tokens_file}")
    print(f"tokens: {args.tokens}")
    print(f"tiles: {args.tiles}")
    for row in rows:
        print(_format_run(row))
    print(f"result: {overall}")
    print("report: ok")

    if args.report_json:
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
        description="Validate local public ATT-1 artifacts across backend smoke paths."
    )
    parser.add_argument(
        "--model-dir", required=True, metavar="DIR",
        help="Local source model directory; used only for provenance reporting"
    )
    parser.add_argument(
        "--att1-f32", required=True, metavar="PATH",
        help="Local converted f32 .att1 artifact"
    )
    parser.add_argument(
        "--att1-q8", required=True, metavar="PATH",
        help="Local converted q8 .att1 artifact"
    )
    parser.add_argument(
        "--tokens-file", required=True, metavar="PATH",
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
        "--report-json", default=None, metavar="PATH",
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
