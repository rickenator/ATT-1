#!/usr/bin/env python3
"""
ATT-1 M175 green evidence packet runner.

This wrapper does not download models and does not make hardware claims. It
drives the existing M171-M174 validators against local external artifacts and
checks that the non-runtime M175 evidence files exist:

- host-access decision
- minimum FPGA control-plane scope
- production-like or partner-style trace packet

The output directory receives per-step stdout/stderr, the M171-M174 JSON
reports, and a manifest summarizing whether the M175 green packet is complete.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def _is_url_like(path: str | None) -> bool:
    if path is None:
        return False
    lowered = str(path).lower()
    return "://" in lowered or lowered.startswith(("hf:", "hf://"))


def _require_path(label: str, path: str, *, directory: bool = False) -> None:
    if not path:
        raise ValueError(f"{label} path is required")
    if _is_url_like(path):
        raise ValueError(f"{label} must be a local filesystem path: {path!r}")
    exists = os.path.isdir(path) if directory else os.path.isfile(path)
    if not exists:
        kind = "directory" if directory else "file"
        raise ValueError(f"{label} {kind} not found: {path!r}")


def _file_record(path: str) -> dict[str, Any]:
    st = os.stat(path)
    return {
        "path": path,
        "size_bytes": st.st_size,
        "mtime": int(st.st_mtime),
    }


def _run_step(
    name: str,
    cmd: list[str],
    out_dir: Path,
    timeout_sec: int,
) -> dict[str, Any]:
    stdout_path = out_dir / f"{name}.stdout.txt"
    stderr_path = out_dir / f"{name}.stderr.txt"
    started = time.time()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
        stdout_path.write_text(proc.stdout, encoding="utf-8")
        stderr_path.write_text(proc.stderr, encoding="utf-8")
        return {
            "name": name,
            "command": cmd,
            "returncode": proc.returncode,
            "status": "pass" if proc.returncode == 0 else "fail",
            "elapsed_sec": round(time.time() - started, 3),
            "stdout": str(stdout_path),
            "stderr": str(stderr_path),
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text(str(exc) + "\n", encoding="utf-8")
        return {
            "name": name,
            "command": cmd,
            "returncode": None,
            "status": "fail",
            "elapsed_sec": round(time.time() - started, 3),
            "stdout": str(stdout_path),
            "stderr": str(stderr_path),
            "error": str(exc),
        }


def _append_optional(cmd: list[str], flag: str, value: Any) -> None:
    if value is not None:
        cmd.extend([flag, str(value)])


def _build_commands(args: argparse.Namespace, out_dir: Path) -> list[tuple[str, list[str]]]:
    py = sys.executable or "python3"
    common_tokens = ["--tokens", str(args.tokens)]
    bench = ["--bench", args.bench]

    m171 = [
        py,
        "compiler/validate_m171_two_tile.py",
        "--model-dir", args.model_dir,
        "--att1-f32", args.att1_f32,
        "--att1-q8", args.att1_q8,
        "--tokens-file", args.tokens_file,
        *common_tokens,
        "--tiles", "2",
        *bench,
        "--report-json", str(out_dir / "m171_two_tile.json"),
    ]

    m172 = [
        py,
        "compiler/validate_m172_beachhead.py",
        "--att1-q8", args.att1_q8,
        "--tokens-file", args.tokens_file,
        *common_tokens,
        "--tiles", "2",
        "--runs", str(args.m172_runs),
        "--min-prompt-tokens", str(args.m172_min_prompt_tokens),
        "--max-jitter-pct", str(args.m172_max_jitter_pct),
        "--workload-name", args.workload_name,
        *bench,
        "--timeout-sec", str(args.step_timeout_sec),
        "--report-json", str(out_dir / "m172_beachhead.json"),
    ]
    _append_optional(m172, "--dollars-per-hour", args.dollars_per_hour)
    _append_optional(m172, "--cuda-baseline-us-per-token", args.cuda_baseline_us_per_token)
    _append_optional(m172, "--cuda-dollars-per-hour", args.cuda_dollars_per_hour)

    m173 = [
        py,
        "compiler/validate_m173_capacity.py",
        "--report", args.placement_report,
        "--budgets-mib", args.m173_budgets_mib,
        "--context", str(args.m173_context),
        "--sessions", str(args.m173_sessions),
        "--kv-page-tokens", args.m173_kv_page_tokens,
        "--max-kv-page-kib", str(args.m173_max_kv_page_kib),
        "--report-json", str(out_dir / "m173_capacity.json"),
    ]
    _append_optional(m173, "--require-pass-budget-mib", args.m173_require_pass_budget_mib)

    m174 = [
        py,
        "compiler/validate_m174_activation_precision.py",
        "--route-report", args.route_report,
        "--packet-overhead-bytes", str(args.m174_packet_overhead_bytes),
        "--sample-count", str(args.m174_sample_count),
        "--min-payload-savings-percent", str(args.m174_min_payload_savings_percent),
        "--max-abs-error", str(args.m174_max_abs_error),
        "--max-rms-error", str(args.m174_max_rms_error),
        "--report-json", str(out_dir / "m174_activation_precision.json"),
    ]

    return [
        ("m171_two_tile", m171),
        ("m172_beachhead", m172),
        ("m173_capacity", m173),
        ("m174_activation_precision", m174),
    ]


def run(args: argparse.Namespace) -> int:
    try:
        _require_path("source model", args.model_dir, directory=True)
        _require_path("f32 ATT-1 artifact", args.att1_f32)
        _require_path("q8 ATT-1 artifact", args.att1_q8)
        _require_path("token IDs", args.tokens_file)
        _require_path("placement report", args.placement_report)
        _require_path("route report", args.route_report)
        _require_path("host-access decision", args.host_access_decision)
        _require_path("FPGA control-plane scope", args.fpga_scope)
        _require_path("trace packet", args.trace_packet)
        _require_path("att1-bench", args.bench)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    steps = []
    for name, cmd in _build_commands(args, out_dir):
        step = _run_step(name, cmd, out_dir, args.step_timeout_sec)
        steps.append(step)
        print(f"{name}: {step['status']}")

    evidence = {
        "source_model": {"path": args.model_dir},
        "att1_f32": _file_record(args.att1_f32),
        "att1_q8": _file_record(args.att1_q8),
        "tokens_file": _file_record(args.tokens_file),
        "placement_report": _file_record(args.placement_report),
        "route_report": _file_record(args.route_report),
        "host_access_decision": _file_record(args.host_access_decision),
        "fpga_scope": _file_record(args.fpga_scope),
        "trace_packet": _file_record(args.trace_packet),
    }
    reports = {
        "m171": str(out_dir / "m171_two_tile.json"),
        "m172": str(out_dir / "m172_beachhead.json"),
        "m173": str(out_dir / "m173_capacity.json"),
        "m174": str(out_dir / "m174_activation_precision.json"),
    }
    result = "pass" if all(step["status"] == "pass" for step in steps) else "fail"
    manifest = {
        "m175_green_packet_manifest_version": 1,
        "result": result,
        "out_dir": str(out_dir),
        "workload_name": args.workload_name,
        "evidence": evidence,
        "reports": reports,
        "steps": steps,
        "green_criteria": {
            "real_external_model_packet": result == "pass",
            "host_access_choice": True,
            "minimal_fpga_control_plane_scope": True,
            "trace_packet": True,
            "regression_required_after_recording": True,
        },
    }
    manifest_path = out_dir / "m175_green_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"manifest: {manifest_path}")
    print(f"result: {result}")
    return 0 if result == "pass" else 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the M175 green evidence packet against local artifacts."
    )
    parser.add_argument("--model-dir", required=True, metavar="DIR")
    parser.add_argument("--att1-f32", required=True, metavar="PATH")
    parser.add_argument("--att1-q8", required=True, metavar="PATH")
    parser.add_argument("--tokens-file", required=True, metavar="PATH")
    parser.add_argument("--placement-report", required=True, metavar="PATH")
    parser.add_argument("--route-report", required=True, metavar="PATH")
    parser.add_argument("--host-access-decision", required=True, metavar="PATH")
    parser.add_argument("--fpga-scope", required=True, metavar="PATH")
    parser.add_argument("--trace-packet", required=True, metavar="PATH")
    parser.add_argument("--out-dir", required=True, metavar="DIR")
    parser.add_argument("--bench", default=os.path.join("build", "att1-bench"))
    parser.add_argument("--tokens", type=int, default=4)
    parser.add_argument("--workload-name", default="m175_green_packet")
    parser.add_argument("--step-timeout-sec", type=int, default=300)
    parser.add_argument("--m172-runs", type=int, default=3)
    parser.add_argument("--m172-min-prompt-tokens", type=int, default=128)
    parser.add_argument("--m172-max-jitter-pct", type=float, default=10000.0)
    parser.add_argument("--dollars-per-hour", type=float, default=None)
    parser.add_argument("--cuda-baseline-us-per-token", type=float, default=None)
    parser.add_argument("--cuda-dollars-per-hour", type=float, default=None)
    parser.add_argument("--m173-budgets-mib", default="256,512,1024")
    parser.add_argument("--m173-context", type=int, default=2048)
    parser.add_argument("--m173-sessions", type=int, default=1)
    parser.add_argument("--m173-kv-page-tokens", default="16,32,64,128")
    parser.add_argument("--m173-max-kv-page-kib", type=int, default=256)
    parser.add_argument("--m173-require-pass-budget-mib", type=int, default=None)
    parser.add_argument("--m174-packet-overhead-bytes", type=int, default=64)
    parser.add_argument("--m174-sample-count", type=int, default=4096)
    parser.add_argument("--m174-min-payload-savings-percent", type=float, default=45.0)
    parser.add_argument("--m174-max-abs-error", type=float, default=0.02)
    parser.add_argument("--m174-max-rms-error", type=float, default=0.004)
    args = parser.parse_args()

    if args.tokens < 1:
        print("error: --tokens must be >= 1", file=sys.stderr)
        sys.exit(2)
    if args.step_timeout_sec < 1:
        print("error: --step-timeout-sec must be >= 1", file=sys.stderr)
        sys.exit(2)
    sys.exit(run(args))


if __name__ == "__main__":
    main()
