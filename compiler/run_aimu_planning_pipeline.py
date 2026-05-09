#!/usr/bin/env python3
"""
ATT-1 AIMU integrated planning pipeline (Milestone 119).

Runs the full prototype-planning pipeline end-to-end:

  Stage 1 — Validate placement report          (M99)
  Stage 2 — Run placement advisory             (M101)
  Stage 3 — Map placement → command plan       (M109)
  Stage 4 — Replay command plan (control-plane simulation) (M113)
  Stage 5 — Map command plan → fabric routes   (M117)
  Stage 6 — Validate fabric route report       (M116)
  Stage 7 — Simulate fabric bandwidth/latency  (M118)
  Stage 8 — Emit integrated planning summary

This tool is a planning and control-plane simulation pipeline.  It does NOT
execute inference, change .att1 binary format, access real PCIe/MMIO
registers, or implement a Linux kernel driver.

Exit codes:
  0 — pipeline passed (PASS or WARN; WARN + --strict exits 1)
  1 — pipeline FAIL, or structural/stage error, or WARN + --strict
  2 — parse/input error (malformed JSON, missing file)
  3 — invalid numeric flag

Usage:

    python3 compiler/run_aimu_planning_pipeline.py \\
        --placement-report compiler/fixtures/placement_report_valid.json

    python3 compiler/run_aimu_planning_pipeline.py \\
        --placement-report build/my_placement.json \\
        --model-id llama3 \\
        --session-id session_1 \\
        --target-tokens-per-sec 50 \\
        --fabric-gib-sec 32 \\
        --workdir build/pipeline/ \\
        --report-json build/pipeline/integrated_report.json \\
        --strict
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Repo root and tool paths
# ---------------------------------------------------------------------------

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent

def _tool(name: str) -> str:
    """Return absolute path to a compiler/ Python tool."""
    return str(_HERE / name)


STAGE_TOOLS = {
    "validate_placement":   _tool("validate_tensor_placement_report.py"),
    "advisory":             _tool("propose_tensor_placement.py"),
    "map_to_commands":      _tool("map_placement_to_commands.py"),
    "replay_commands":      _tool("replay_aimu_command_plan.py"),
    "map_to_routes":        _tool("map_commands_to_fabric_routes.py"),
    "validate_routes":      _tool("validate_fabric_routes.py"),
    "simulate_bandwidth":   _tool("simulate_fabric_bandwidth.py"),
}

PIPELINE_VERSION = 1


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class PipelineError(Exception):
    """Unrecoverable pipeline stage error."""


# ---------------------------------------------------------------------------
# Stage runner
# ---------------------------------------------------------------------------

def _run_stage(
    label: str,
    cmd: list[str],
    *,
    capture_output: bool = False,
) -> tuple[int, str, str]:
    """
    Run a subprocess stage and return (returncode, stdout, stderr).
    stdout/stderr are always captured; they are echoed if not capture_output.
    """
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )
    if not capture_output:
        if result.stdout:
            print(result.stdout, end="")
        if result.stderr:
            print(result.stderr, end="", file=sys.stderr)
    return result.returncode, result.stdout, result.stderr


def _load_json(path: str, label: str) -> dict:
    """Load a JSON file produced by a stage; raise PipelineError on failure."""
    try:
        with open(path) as fh:
            return json.load(fh)
    except FileNotFoundError:
        raise PipelineError(f"{label}: output JSON not found at {path}")
    except json.JSONDecodeError as exc:
        raise PipelineError(f"{label}: malformed JSON at {path}: {exc}")


# ---------------------------------------------------------------------------
# Status helpers
# ---------------------------------------------------------------------------

_STATUS_RANK = {"pass": 0, "ok": 0, "warn": 1, "fail": 2, "unknown": 1}


def _worst(*statuses: str) -> str:
    """Return the highest-severity status from a list."""
    statuses = [s.lower() for s in statuses if s]
    if not statuses:
        return "pass"
    return max(statuses, key=lambda s: _STATUS_RANK.get(s, 0))


def _normalise(status: str) -> str:
    """Normalise a stage status to pass/warn/fail."""
    s = status.lower()
    if s in ("pass", "ok"):
        return "pass"
    if s in ("warn",):
        return "warn"
    return "fail"


# ---------------------------------------------------------------------------
# Recommendation builder
# ---------------------------------------------------------------------------

def _recommend(
    advisory_status: str,
    replay_status: str,
    route_val_status: str,
    bw_status: str,
    advisory_next_action: str,
    bw_required: float | None,
    bw_target: float | None,
) -> str:
    parts: list[str] = []

    adv = _normalise(advisory_status)
    bws = bw_status.lower() if bw_status else "unknown"

    if adv == "fail":
        parts.append(f"Placement: {advisory_next_action}")
    elif adv == "warn":
        parts.append("Placement advisory warns; consider remediation before deployment.")

    if _normalise(replay_status) == "fail":
        parts.append("Command replay failed; review unsupported operations in command plan.")

    if _normalise(route_val_status) == "fail":
        parts.append("Fabric route validation failed; fix route report before bandwidth analysis.")

    if bws == "fail":
        if bw_required is not None and bw_target is not None:
            parts.append(
                f"Fabric bandwidth oversubscribed: required {bw_required:.3f} GiB/s "
                f"> target {bw_target:.1f} GiB/s. Increase fabric bandwidth or reduce "
                f"token rate."
            )
        else:
            parts.append("Fabric bandwidth FAIL; increase fabric bandwidth or reduce token rate.")
    elif bws == "warn":
        parts.append("Fabric bandwidth approaching target; consider increasing bandwidth budget.")

    if not parts:
        parts.append("All pipeline stages passed. Placement is ready for prototype evaluation.")

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Integrated report
# ---------------------------------------------------------------------------

def _build_report(
    placement_path: str,
    stage_statuses: dict[str, str],
    stage_details: dict[str, Any],
    overall_status: str,
    recommendation: str,
    args,
) -> dict:
    val_detail = stage_details.get("validate_placement", {})
    adv_detail = stage_details.get("advisory", {})
    cmd_detail = stage_details.get("map_to_commands", {})
    replay_detail = stage_details.get("replay_commands", {})
    route_detail = stage_details.get("map_to_routes", {})
    val_rt_detail = stage_details.get("validate_routes", {})
    bw_detail = stage_details.get("simulate_bandwidth", {})

    bw_agg = bw_detail.get("aggregate", {}) if bw_detail else {}

    return {
        "pipeline_version": PIPELINE_VERSION,
        "placement_report_path": placement_path,
        "placement_validation_status": stage_statuses.get("validate_placement", "unknown"),
        "advisory_status": stage_statuses.get("advisory", "unknown"),
        "command_plan_status": stage_statuses.get("map_to_commands", "unknown"),
        "command_replay_status": stage_statuses.get("replay_commands", "unknown"),
        "fabric_route_map_status": stage_statuses.get("map_to_routes", "unknown"),
        "fabric_route_validation_status": stage_statuses.get("validate_routes", "unknown"),
        "fabric_simulation_status": stage_statuses.get("simulate_bandwidth", "unknown"),
        "tile_count": (
            val_detail.get("tile_count_header")
            or cmd_detail.get("header", {}).get("tile_count")
        ),
        "tensor_count": (
            val_detail.get("tensor_record_count")
            or cmd_detail.get("header", {}).get("tensor_count")
        ),
        "command_count": (
            cmd_detail.get("header", {}).get("command_count")
            or replay_detail.get("command_count")
        ),
        "route_count": (
            route_detail.get("header", {}).get("route_count")
            if route_detail else None
        ),
        "commands_replayed": replay_detail.get("commands_replayed"),
        "completions_seen": replay_detail.get("completions_seen"),
        "failed_commands": replay_detail.get("failed_commands"),
        "unsupported_commands": replay_detail.get("unsupported_commands"),
        "aggregate_required_fabric_gib_sec": bw_agg.get("required_fabric_gib_sec"),
        "fabric_utilization_percent": bw_agg.get("utilization_percent"),
        "capacity_status": adv_detail.get("status", "unknown"),
        "bandwidth_status": (
            bw_agg.get("status", "UNKNOWN").lower()
            if bw_agg else "unknown"
        ),
        "recommended_next_action": recommendation,
        "status": overall_status,
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def _run_pipeline(args) -> int:  # noqa: C901 (complex but linear)
    placement_path = args.placement_report
    strict = args.strict

    if not os.path.exists(placement_path):
        print(f"parse error: placement report not found: {placement_path}", file=sys.stderr)
        return 2

    # Decide working directory for intermediate files
    own_tmpdir = None
    if args.workdir:
        workdir = Path(args.workdir)
        workdir.mkdir(parents=True, exist_ok=True)
    else:
        own_tmpdir = tempfile.mkdtemp(prefix="att1_pipeline_")
        workdir = Path(own_tmpdir)

    try:
        return _pipeline_body(args, placement_path, workdir, strict)
    finally:
        if own_tmpdir:
            # Remove temp files but not the directory tree (may contain user outputs)
            for f in workdir.glob("*.json"):
                try:
                    f.unlink()
                except OSError:
                    pass
            try:
                workdir.rmdir()
            except OSError:
                pass


def _pipeline_body(args, placement_path: str, workdir: Path, strict: bool) -> int:
    py = sys.executable
    stage_statuses: dict[str, str] = {}
    stage_details: dict[str, Any] = {}
    warnings: list[str] = []
    failures: list[str] = []
    overall_status = "pass"

    # ── Intermediate file paths ────────────────────────────────────────
    p_val_json    = str(workdir / "01_placement_validation.json")
    p_adv_json    = str(workdir / "02_advisory.json")
    p_cmd_json    = str(workdir / "03_command_plan.json")
    p_replay_json = str(workdir / "04_replay_report.json")
    p_route_json  = str(workdir / "05_fabric_routes.json")
    p_rtval_json  = str(workdir / "06_route_validation.json")
    p_bw_json     = str(workdir / "07_bw_simulation.json")

    # ── STAGE 1: Validate placement report ────────────────────────────
    print("=== Stage 1: Validate placement report ===")
    rc, _, _ = _run_stage("validate_placement", [
        py, STAGE_TOOLS["validate_placement"],
        "--report", placement_path,
        "--report-json", p_val_json,
    ])
    try:
        val_detail = _load_json(p_val_json, "Stage 1")
        stage_details["validate_placement"] = val_detail
    except PipelineError:
        val_detail = {}

    val_status = _normalise(val_detail.get("status", "fail" if rc != 0 else "pass"))
    stage_statuses["validate_placement"] = val_status
    if val_status == "fail":
        failures.append("Placement validation failed.")
        if strict:
            overall_status = "fail"
            print("pipeline: strict mode — stopping after placement validation failure.",
                  file=sys.stderr)
            return _finish(args, placement_path, stage_statuses, stage_details,
                           "fail", failures, warnings)

    print()

    # ── STAGE 2: Placement advisory ───────────────────────────────────
    print("=== Stage 2: Placement advisory ===")
    rc, _, _ = _run_stage("advisory", [
        py, STAGE_TOOLS["advisory"],
        "--report", placement_path,
        "--report-json", p_adv_json,
    ])
    try:
        adv_detail = _load_json(p_adv_json, "Stage 2")
        stage_details["advisory"] = adv_detail
    except PipelineError:
        adv_detail = {}

    adv_status = _normalise(adv_detail.get("status", "fail" if rc != 0 else "pass"))
    stage_statuses["advisory"] = adv_status
    if adv_status == "fail":
        failures.append(f"Advisory fail: {adv_detail.get('next_action', 'see advisory report')}")
        if strict:
            overall_status = "fail"
            print("pipeline: strict mode — stopping after advisory failure.", file=sys.stderr)
            return _finish(args, placement_path, stage_statuses, stage_details,
                           "fail", failures, warnings)
    elif adv_status == "warn":
        warnings.append("Advisory warn: consider remediation before deployment.")

    print()

    # ── STAGE 3: Map placement to command plan ─────────────────────────
    print("=== Stage 3: Map placement → command plan ===")
    cmd_args = [
        py, STAGE_TOOLS["map_to_commands"],
        "--report", placement_path,
        "--plan-json", p_cmd_json,
    ]
    if args.model_id:
        cmd_args += ["--model-id", args.model_id]
    if args.session_id:
        cmd_args += ["--session-id", args.session_id]
    rc, _, _ = _run_stage("map_to_commands", cmd_args)
    try:
        cmd_detail = _load_json(p_cmd_json, "Stage 3")
        stage_details["map_to_commands"] = cmd_detail
    except PipelineError as exc:
        failures.append(str(exc))
        return _finish(args, placement_path, stage_statuses, stage_details,
                       "fail", failures, warnings)

    cmd_status = _normalise(cmd_detail.get("header", {}).get("status", "fail" if rc != 0 else "ok"))
    stage_statuses["map_to_commands"] = cmd_status
    if cmd_status == "fail":
        failures.append("Command plan mapping failed.")
        return _finish(args, placement_path, stage_statuses, stage_details,
                       "fail", failures, warnings)

    print()

    # ── STAGE 4: Replay command plan ──────────────────────────────────
    print("=== Stage 4: Replay command plan ===")
    replay_cmd = [
        py, STAGE_TOOLS["replay_commands"],
        "--plan", p_cmd_json,
        "--report-json", p_replay_json,
    ]
    rc, _, _ = _run_stage("replay_commands", replay_cmd)
    try:
        replay_detail = _load_json(p_replay_json, "Stage 4")
        stage_details["replay_commands"] = replay_detail
    except PipelineError as exc:
        failures.append(str(exc))
        return _finish(args, placement_path, stage_statuses, stage_details,
                       "fail", failures, warnings)

    replay_status = _normalise(replay_detail.get("status", "fail" if rc != 0 else "pass"))
    stage_statuses["replay_commands"] = replay_status
    if replay_status == "fail":
        failures.append(
            f"Command replay failed: {replay_detail.get('failed_commands', '?')} "
            "failed commands."
        )
        if strict:
            return _finish(args, placement_path, stage_statuses, stage_details,
                           "fail", failures, warnings)
    if replay_detail.get("unsupported_commands", 0):
        warnings.append(
            f"Replay: {replay_detail['unsupported_commands']} unsupported command type(s) skipped."
        )

    print()

    # ── STAGE 5: Map command plan → fabric routes ─────────────────────
    print("=== Stage 5: Map command plan → fabric routes ===")
    route_args = [
        py, STAGE_TOOLS["map_to_routes"],
        "--plan", p_cmd_json,
        "--route-report-json", p_route_json,
    ]
    if args.target_tokens_per_sec is not None:
        route_args += ["--tokens-per-sec", str(args.target_tokens_per_sec)]
    if args.fabric_gib_sec is not None:
        route_args += ["--fabric-gib-sec", str(args.fabric_gib_sec)]
    rc, _, _ = _run_stage("map_to_routes", route_args)
    try:
        route_detail = _load_json(p_route_json, "Stage 5")
        stage_details["map_to_routes"] = route_detail
    except PipelineError as exc:
        failures.append(str(exc))
        return _finish(args, placement_path, stage_statuses, stage_details,
                       "fail", failures, warnings)

    route_map_status = _normalise(route_detail.get("header", {}).get("status", "fail" if rc != 0 else "pass"))
    stage_statuses["map_to_routes"] = route_map_status
    if route_map_status == "fail":
        failures.append("Fabric route mapping failed.")

    print()

    # ── STAGE 6: Validate fabric routes ───────────────────────────────
    print("=== Stage 6: Validate fabric routes ===")
    rc, _, _ = _run_stage("validate_routes", [
        py, STAGE_TOOLS["validate_routes"],
        "--report", p_route_json,
        "--report-json", p_rtval_json,
    ])
    try:
        rtval_detail = _load_json(p_rtval_json, "Stage 6")
        stage_details["validate_routes"] = rtval_detail
    except PipelineError:
        rtval_detail = {}

    rt_val_status = _normalise(rtval_detail.get("status", "fail" if rc != 0 else "pass"))
    stage_statuses["validate_routes"] = rt_val_status
    if rt_val_status == "fail":
        failures.append(
            f"Fabric route validation failed: {rtval_detail.get('total_errors', '?')} error(s)."
        )
        if strict:
            return _finish(args, placement_path, stage_statuses, stage_details,
                           "fail", failures, warnings)

    print()

    # ── STAGE 7: Simulate fabric bandwidth/latency ────────────────────
    print("=== Stage 7: Simulate fabric bandwidth/latency ===")
    bw_args = [
        py, STAGE_TOOLS["simulate_bandwidth"],
        "--route-report", p_route_json,
        "--report-json", p_bw_json,
    ]
    if args.target_tokens_per_sec is not None:
        bw_args += ["--target-tokens-per-sec", str(args.target_tokens_per_sec)]
    if args.fabric_gib_sec is not None:
        bw_args += ["--fabric-gib-sec", str(args.fabric_gib_sec)]
    rc, _, _ = _run_stage("simulate_bandwidth", bw_args)
    try:
        bw_detail = _load_json(p_bw_json, "Stage 7")
        stage_details["simulate_bandwidth"] = bw_detail
    except PipelineError:
        bw_detail = {}

    bw_hdr_status = (bw_detail.get("header", {}).get("status") or
                     ("FAIL" if rc == 1 else "PASS"))
    bw_sim_status = _normalise(bw_hdr_status)
    stage_statuses["simulate_bandwidth"] = bw_sim_status
    if bw_sim_status == "fail":
        failures.append("Fabric bandwidth simulation: FAIL (oversubscribed).")
    elif bw_sim_status == "warn":
        warnings.append("Fabric bandwidth simulation: WARN (approaching target).")

    print()

    # ── Determine overall status ──────────────────────────────────────
    all_statuses = list(stage_statuses.values())
    overall_status = _worst(*all_statuses)
    if overall_status in ("pass", "ok", "unknown"):
        overall_status = "pass" if not warnings else "warn"
    if failures:
        overall_status = "fail"

    bw_agg = bw_detail.get("aggregate", {}) if bw_detail else {}
    bw_req = bw_agg.get("required_fabric_gib_sec")
    bw_tgt = bw_agg.get("fabric_gib_sec_target")

    recommendation = _recommend(
        adv_status,
        replay_status if "replay_commands" in stage_statuses else "pass",
        rt_val_status,
        bw_hdr_status,
        adv_detail.get("next_action", ""),
        bw_req,
        bw_tgt,
    )

    return _finish(args, placement_path, stage_statuses, stage_details,
                   overall_status, failures, warnings, recommendation=recommendation)


# ---------------------------------------------------------------------------
# Finish: print summary and write JSON report
# ---------------------------------------------------------------------------

def _finish(
    args,
    placement_path: str,
    stage_statuses: dict[str, str],
    stage_details: dict[str, Any],
    overall_status: str,
    failures: list[str],
    warnings: list[str],
    recommendation: str = "",
) -> int:

    bw_detail = stage_details.get("simulate_bandwidth", {})
    adv_detail = stage_details.get("advisory", {})
    replay_detail = stage_details.get("replay_commands", {})
    bw_agg = bw_detail.get("aggregate", {}) if bw_detail else {}

    recommendation = recommendation or _recommend(
        stage_statuses.get("advisory", "pass"),
        stage_statuses.get("replay_commands", "pass"),
        stage_statuses.get("validate_routes", "pass"),
        bw_agg.get("status", "UNKNOWN"),
        adv_detail.get("next_action", ""),
        bw_agg.get("required_fabric_gib_sec"),
        bw_agg.get("fabric_gib_sec_target"),
    )

    report = _build_report(
        placement_path, stage_statuses, stage_details, overall_status,
        recommendation, args,
    )

    # Print integrated summary
    print("=" * 72)
    print("ATT-1 AIMU Planning Pipeline Summary (M119)")
    print("  NOTE: Control-plane simulation only — not inference execution,")
    print("  not PCIe hardware access.")
    print(f"  Placement report  : {placement_path}")
    print(f"  Tile count        : {report.get('tile_count', '?')}")
    print(f"  Tensor count      : {report.get('tensor_count', '?')}")
    print(f"  Command count     : {report.get('command_count', '?')}")
    print(f"  Route count       : {report.get('route_count', '?')}")
    print()
    labels = [
        ("Stage 1 placement validation",   "placement_validation_status"),
        ("Stage 2 advisory",               "advisory_status"),
        ("Stage 3 command plan",           "command_plan_status"),
        ("Stage 4 command replay",         "command_replay_status"),
        ("Stage 5 fabric route mapping",   "fabric_route_map_status"),
        ("Stage 6 fabric route validation","fabric_route_validation_status"),
        ("Stage 7 fabric bandwidth sim",   "fabric_simulation_status"),
    ]
    for label, key in labels:
        status = report.get(key, "?")
        print(f"  {label:<36s}: {status}")
    print()
    if report.get("aggregate_required_fabric_gib_sec") is not None:
        print(f"  Required fabric bandwidth : "
              f"{report['aggregate_required_fabric_gib_sec']:.4e} GiB/s")
    if report.get("fabric_utilization_percent") is not None:
        print(f"  Fabric utilization        : "
              f"{report['fabric_utilization_percent']:.2f}%")
    print()
    if warnings:
        print("  Warnings:")
        for w in warnings:
            print(f"    - {w}")
        print()
    if failures:
        print("  Failures:")
        for f in failures:
            print(f"    - {f}")
        print()
    print(f"  Recommendation: {recommendation}")
    print()
    print(f"  Overall status    : {overall_status.upper()}")
    print("=" * 72)

    if args.report_json:
        try:
            with open(args.report_json, "w") as fh:
                json.dump(report, fh, indent=2)
            print(f"JSON report written to: {args.report_json}")
        except OSError as exc:
            print(f"error writing JSON report: {exc}", file=sys.stderr)
            return 1

    if overall_status == "fail":
        return 1
    if overall_status == "warn" and args.strict:
        return 1
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    ap = argparse.ArgumentParser(
        description="ATT-1 AIMU integrated planning pipeline (M119)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--placement-report", required=True, metavar="PATH",
        help="M98/M100 tensor placement report JSON (pipeline input)",
    )
    ap.add_argument(
        "--model-id", default="", metavar="ID",
        help="Model identifier forwarded to command-plan mapper",
    )
    ap.add_argument(
        "--session-id", default="session_0", metavar="ID",
        help="Session identifier forwarded to command-plan mapper",
    )
    ap.add_argument(
        "--target-tokens-per-sec", type=float, default=None, metavar="N",
        help="Inference token rate target; forwarded to route mapper and BW simulator",
    )
    ap.add_argument(
        "--fabric-gib-sec", type=float, default=None, metavar="N",
        help="Fabric bandwidth budget (GiB/s); forwarded to route mapper and BW simulator",
    )
    ap.add_argument(
        "--strict", action="store_true",
        help="Exit nonzero on any WARN or FAIL at any stage",
    )
    ap.add_argument(
        "--workdir", default=None, metavar="PATH",
        help="Directory for intermediate JSON files; created if absent",
    )
    ap.add_argument(
        "--report-json", default=None, metavar="PATH",
        help="Write integrated JSON report to PATH",
    )
    return ap.parse_args()


def _validate_args(args) -> list[str]:
    errors: list[str] = []
    if args.target_tokens_per_sec is not None and args.target_tokens_per_sec <= 0:
        errors.append("--target-tokens-per-sec must be > 0")
    if args.fabric_gib_sec is not None and args.fabric_gib_sec <= 0:
        errors.append("--fabric-gib-sec must be > 0")
    return errors


def main() -> int:
    args = _parse_args()
    arg_errors = _validate_args(args)
    if arg_errors:
        for e in arg_errors:
            print(f"error: {e}", file=sys.stderr)
        return 3
    return _run_pipeline(args)


if __name__ == "__main__":
    sys.exit(main())
