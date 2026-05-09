#!/usr/bin/env python3
"""
ATT-1 integrated execution/replay pipeline (Milestone 132).

Runs the full execution/replay pipeline end-to-end starting from a tensor
execution plan:

  Stage 1 — Validate tensor execution plan          (M128)
  Stage 2 — Map execution plan → command plan       (M129)
  Stage 3 — Replay command plan via MMIO emulator   (M122)
  Stage 4 — Map command plan → fabric routes        (M117)
  Stage 5 — Validate fabric route report            (M116)
  Stage 6 — Replay fabric routes + bandwidth sim    (M123 / M118)
  Stage 7 — Emit integrated regression report

This tool is a control-plane simulation and replay pipeline.  It does NOT
execute inference, change .att1 binary format, access real PCIe/MMIO
registers, or implement a Linux kernel driver.

Exit codes:
  0 — pipeline passed (PASS or WARN; WARN + --strict exits 1)
  1 — pipeline FAIL, or structural/stage error, or WARN + --strict
  2 — parse/input error (malformed JSON, missing execution plan file)
  3 — invalid numeric flag

Usage:

    python3 compiler/run_execution_replay_pipeline.py \\
        --execution-plan compiler/fixtures/exec_plan_valid_tiny.json

    python3 compiler/run_execution_replay_pipeline.py \\
        --execution-plan build/my_exec_plan.json \\
        --tiles 4 \\
        --tile-memory-mib 32 \\
        --kv-memory-mib 8 \\
        --target-tokens-per-sec 50 \\
        --fabric-gib-sec 32 \\
        --workdir build/exec_pipeline/ \\
        --report-json build/exec_pipeline/integrated_report.json \\
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
_AIMU_MMIO_REPLAY_BIN = _REPO / "build" / "att1-aimu-mmio-replay"


def _tool(name: str) -> str:
    """Return absolute path to a compiler/ Python tool."""
    return str(_HERE / name)


STAGE_TOOLS = {
    "validate_exec_plan": _tool("validate_tensor_execution_plan.py"),
    "map_to_commands":    _tool("map_execution_plan_to_commands.py"),
    "mmio_replay":        _tool("replay_command_plan_via_mmio.py"),
    "map_to_routes":      _tool("map_commands_to_fabric_routes.py"),
    "validate_routes":    _tool("validate_fabric_routes.py"),
    "fabric_replay":      _tool("replay_fabric_routes.py"),
}

PIPELINE_VERSION = 132


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
    if s in ("warn", "unknown"):
        return "warn"
    return "fail"


# ---------------------------------------------------------------------------
# Recommendation builder
# ---------------------------------------------------------------------------

def _recommend(
    plan_val_status: str,
    cmd_plan_status: str,
    mmio_status: str,
    route_val_status: str,
    fabric_replay_status: str,
    fabric_bw_status: str,
) -> str:
    parts: list[str] = []

    if _normalise(plan_val_status) == "fail":
        parts.append("Execution plan validation failed; fix plan structure before proceeding.")
    elif _normalise(plan_val_status) == "warn":
        parts.append("Execution plan has warnings; review plan before deployment.")

    if _normalise(cmd_plan_status) == "fail":
        parts.append("Command plan mapping failed; check execution plan command types.")

    if _normalise(mmio_status) == "fail":
        parts.append(
            "MMIO replay failed; ensure build/att1-aimu-mmio-replay is compiled "
            "and command plan is well-formed."
        )
    elif _normalise(mmio_status) == "warn":
        parts.append("MMIO replay warned; review unsupported operations in command plan.")

    if _normalise(route_val_status) == "fail":
        parts.append(
            "Fabric route validation failed; fix route report before bandwidth analysis."
        )

    if _normalise(fabric_replay_status) == "fail":
        parts.append("Fabric route replay failed; review route report for invalid routes.")

    bws = fabric_bw_status.lower() if fabric_bw_status else "unknown"
    if bws == "fail":
        parts.append(
            "Fabric bandwidth oversubscribed; increase fabric bandwidth or reduce token rate."
        )
    elif bws == "warn":
        parts.append("Fabric bandwidth approaching target; consider increasing bandwidth budget.")

    if not parts:
        parts.append(
            "All pipeline stages passed. "
            "Execution plan is ready for prototype evaluation."
        )

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Integrated report
# ---------------------------------------------------------------------------

def _build_report(
    execution_plan_path: str,
    stage_statuses: dict[str, str],
    stage_details: dict[str, Any],
    overall_status: str,
    recommendation: str,
) -> dict:
    plan_val_detail = stage_details.get("validate_exec_plan", {})
    cmd_detail = stage_details.get("map_to_commands", {})
    mmio_detail = stage_details.get("mmio_replay", {})
    route_detail = stage_details.get("map_to_routes", {})
    fab_replay_detail = stage_details.get("fabric_replay", {})

    # tile_count: prefer command plan header; fall back to plan validation
    tile_count = (
        cmd_detail.get("header", {}).get("tile_count")
        or plan_val_detail.get("tile_count")
    )

    # exec_commands_seen: count EXEC_* commands flagged as unsupported by mapper
    exec_commands_seen = (
        cmd_detail.get("summary", {}).get("unsupported_exec_commands")
        if cmd_detail else None
    )

    # route_count from route map header
    route_count = (
        route_detail.get("header", {}).get("route_count")
        if route_detail else None
    )

    # required_fabric_gib_sec from fabric replay
    required_gib_sec = fab_replay_detail.get("required_fabric_gib_sec") if fab_replay_detail else None

    return {
        "pipeline_version": PIPELINE_VERSION,
        "execution_plan_path": execution_plan_path,
        # Stage statuses
        "execution_plan_validation_status": stage_statuses.get("validate_exec_plan", "unknown"),
        "command_plan_status": stage_statuses.get("map_to_commands", "unknown"),
        "mmio_replay_status": stage_statuses.get("mmio_replay", "unknown"),
        "fabric_route_status": stage_statuses.get("fabric_route_combined", "unknown"),
        "fabric_replay_status": stage_statuses.get("fabric_replay", "unknown"),
        "fabric_simulation_status": stage_statuses.get("fabric_simulation", "unknown"),
        # Counts
        "tile_count": tile_count,
        "command_count": (
            cmd_detail.get("header", {}).get("command_count")
            or mmio_detail.get("command_count")
        ),
        "route_count": route_count,
        # MMIO replay fields
        "commands_replayed": mmio_detail.get("commands_replayed") if mmio_detail else None,
        "completions_seen": mmio_detail.get("completions_seen") if mmio_detail else None,
        "exec_commands_seen": exec_commands_seen,
        "failed_commands": mmio_detail.get("failed_commands") if mmio_detail else None,
        "unsupported_commands": mmio_detail.get("unsupported_commands") if mmio_detail else None,
        # Fabric replay/BW fields
        "aggregate_packets_sent": (
            fab_replay_detail.get("aggregate_packets_sent") if fab_replay_detail else None
        ),
        "aggregate_payload_bytes_sent": (
            fab_replay_detail.get("aggregate_payload_bytes_sent") if fab_replay_detail else None
        ),
        "required_fabric_gib_sec": required_gib_sec,
        "fabric_status": (
            fab_replay_detail.get("fabric_status", "UNKNOWN") if fab_replay_detail else "UNKNOWN"
        ),
        # Overall
        "recommended_next_action": recommendation,
        "final_status": overall_status,
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def _run_pipeline(args) -> int:  # noqa: C901
    execution_plan_path = args.execution_plan
    strict = args.strict

    if not os.path.exists(execution_plan_path):
        print(
            f"parse error: execution plan not found: {execution_plan_path}",
            file=sys.stderr,
        )
        return 2

    # Decide working directory for intermediate files
    own_tmpdir = None
    if args.workdir:
        workdir = Path(args.workdir)
        workdir.mkdir(parents=True, exist_ok=True)
    else:
        own_tmpdir = tempfile.mkdtemp(prefix="att1_exec_pipeline_")
        workdir = Path(own_tmpdir)

    try:
        return _pipeline_body(args, execution_plan_path, workdir, strict)
    finally:
        if own_tmpdir:
            for f in workdir.glob("*.json"):
                try:
                    f.unlink()
                except OSError:
                    pass
            try:
                workdir.rmdir()
            except OSError:
                pass


def _pipeline_body(  # noqa: C901
    args, execution_plan_path: str, workdir: Path, strict: bool
) -> int:
    py = sys.executable
    stage_statuses: dict[str, str] = {}
    stage_details: dict[str, Any] = {}
    warnings: list[str] = []
    failures: list[str] = []

    # ── Intermediate file paths ────────────────────────────────────────
    p_plan_val_json    = str(workdir / "01_exec_plan_validation.json")
    p_cmd_json         = str(workdir / "02_command_plan.json")
    p_mmio_json        = str(workdir / "03_mmio_replay.json")
    p_route_json       = str(workdir / "04_fabric_routes.json")
    p_rtval_json       = str(workdir / "05_route_validation.json")
    p_fab_replay_json  = str(workdir / "06_fabric_replay.json")

    # ── STAGE 1: Validate tensor execution plan ────────────────────────
    print("=== Stage 1: Validate tensor execution plan ===")
    stage1_cmd = [
        py, STAGE_TOOLS["validate_exec_plan"],
        "--plan", execution_plan_path,
        "--report-json", p_plan_val_json,
    ]
    if strict:
        stage1_cmd.append("--strict")
    rc, _, _ = _run_stage("validate_exec_plan", stage1_cmd)
    try:
        plan_val_detail = _load_json(p_plan_val_json, "Stage 1")
        stage_details["validate_exec_plan"] = plan_val_detail
    except PipelineError:
        plan_val_detail = {}

    plan_val_status = _normalise(
        plan_val_detail.get("status", "fail" if rc != 0 else "pass")
    )
    stage_statuses["validate_exec_plan"] = plan_val_status
    if plan_val_status == "fail":
        failures.append("Execution plan validation failed.")
        if strict:
            print(
                "pipeline: strict mode — stopping after plan validation failure.",
                file=sys.stderr,
            )
            return _finish(
                args, execution_plan_path, stage_statuses, stage_details,
                "fail", failures, warnings,
            )
    print()

    # ── STAGE 2: Map execution plan → command plan ─────────────────────
    print("=== Stage 2: Map execution plan → command plan ===")
    stage2_cmd = [
        py, STAGE_TOOLS["map_to_commands"],
        "--execution-plan", execution_plan_path,
        "--plan-json", p_cmd_json,
    ]
    if strict:
        stage2_cmd.append("--strict")
    rc, _, _ = _run_stage("map_to_commands", stage2_cmd)
    try:
        cmd_detail = _load_json(p_cmd_json, "Stage 2")
        stage_details["map_to_commands"] = cmd_detail
    except PipelineError as exc:
        failures.append(str(exc))
        stage_statuses["map_to_commands"] = "fail"
        return _finish(
            args, execution_plan_path, stage_statuses, stage_details,
            "fail", failures, warnings,
        )

    cmd_status = _normalise(
        cmd_detail.get("header", {}).get("status", "fail" if rc != 0 else "ok")
    )
    stage_statuses["map_to_commands"] = cmd_status
    if cmd_status == "fail":
        failures.append("Command plan mapping failed.")
        return _finish(
            args, execution_plan_path, stage_statuses, stage_details,
            "fail", failures, warnings,
        )
    if cmd_detail.get("summary", {}).get("unsupported_exec_commands", 0):
        n = cmd_detail["summary"]["unsupported_exec_commands"]
        warnings.append(
            f"Command plan: {n} EXEC_* command(s) mapped with UNSUPPORTED status "
            "(M105 simulator does not execute tensor math)."
        )
    print()

    # ── STAGE 3: Replay command plan via MMIO emulator ─────────────────
    print("=== Stage 3: Replay command plan via MMIO emulator ===")
    # Check binary availability before invoking M122 wrapper
    binary_missing = not _AIMU_MMIO_REPLAY_BIN.exists()
    if binary_missing:
        print(
            f"  NOTE: build/att1-aimu-mmio-replay not found; "
            f"MMIO replay stage skipped (run 'make' to build).",
            file=sys.stderr,
        )
        stage_statuses["mmio_replay"] = "warn"
        warnings.append(
            "MMIO replay skipped: build/att1-aimu-mmio-replay binary not found."
        )
        stage_details["mmio_replay"] = {}
    else:
        stage3_cmd = [
            py, STAGE_TOOLS["mmio_replay"],
            "--plan", p_cmd_json,
            "--report-json", p_mmio_json,
        ]
        if args.tiles is not None:
            stage3_cmd += ["--tiles", str(args.tiles)]
        if args.tile_memory_mib is not None:
            stage3_cmd += ["--tile-memory-mib", str(args.tile_memory_mib)]
        if args.kv_memory_mib is not None:
            stage3_cmd += ["--kv-memory-mib", str(args.kv_memory_mib)]
        if strict:
            stage3_cmd.append("--strict")
        rc, _, _ = _run_stage("mmio_replay", stage3_cmd)
        try:
            mmio_detail = _load_json(p_mmio_json, "Stage 3")
            stage_details["mmio_replay"] = mmio_detail
        except PipelineError:
            mmio_detail = {}

        mmio_status = _normalise(
            mmio_detail.get("status", "fail" if rc != 0 else "pass")
        )
        stage_statuses["mmio_replay"] = mmio_status
        if mmio_status == "fail":
            failures.append(
                f"MMIO replay failed: "
                f"{mmio_detail.get('failed_commands', '?')} failed command(s)."
            )
            if strict:
                return _finish(
                    args, execution_plan_path, stage_statuses, stage_details,
                    "fail", failures, warnings,
                )
        if mmio_detail.get("unsupported_commands", 0):
            n = mmio_detail["unsupported_commands"]
            warnings.append(f"MMIO replay: {n} unsupported command(s) (EXEC_* / KV_*).")
    print()

    # ── STAGE 4: Map command plan → fabric routes ──────────────────────
    print("=== Stage 4: Map command plan → fabric routes ===")
    stage4_cmd = [
        py, STAGE_TOOLS["map_to_routes"],
        "--plan", p_cmd_json,
        "--route-report-json", p_route_json,
    ]
    if args.target_tokens_per_sec is not None:
        stage4_cmd += ["--tokens-per-sec", str(args.target_tokens_per_sec)]
    if args.fabric_gib_sec is not None:
        stage4_cmd += ["--fabric-gib-sec", str(args.fabric_gib_sec)]
    rc, _, _ = _run_stage("map_to_routes", stage4_cmd)
    try:
        route_detail = _load_json(p_route_json, "Stage 4")
        stage_details["map_to_routes"] = route_detail
    except PipelineError as exc:
        failures.append(str(exc))
        stage_statuses["fabric_route_combined"] = "fail"
        return _finish(
            args, execution_plan_path, stage_statuses, stage_details,
            "fail", failures, warnings,
        )

    route_map_status = _normalise(
        route_detail.get("header", {}).get("status", "fail" if rc != 0 else "pass")
    )
    if route_map_status == "fail":
        failures.append("Fabric route mapping failed.")
    print()

    # ── STAGE 5: Validate fabric routes ───────────────────────────────
    print("=== Stage 5: Validate fabric routes ===")
    stage5_cmd = [
        py, STAGE_TOOLS["validate_routes"],
        "--report", p_route_json,
        "--report-json", p_rtval_json,
    ]
    if strict:
        stage5_cmd.append("--strict")
    rc, _, _ = _run_stage("validate_routes", stage5_cmd)
    try:
        rtval_detail = _load_json(p_rtval_json, "Stage 5")
        stage_details["validate_routes"] = rtval_detail
    except PipelineError:
        rtval_detail = {}

    rt_val_status = _normalise(
        rtval_detail.get("status", "fail" if rc != 0 else "pass")
    )
    fabric_route_combined = _worst(route_map_status, rt_val_status)
    stage_statuses["fabric_route_combined"] = fabric_route_combined
    if rt_val_status == "fail":
        failures.append(
            f"Fabric route validation failed: "
            f"{rtval_detail.get('total_errors', '?')} error(s)."
        )
        if strict:
            return _finish(
                args, execution_plan_path, stage_statuses, stage_details,
                "fail", failures, warnings,
            )
    print()

    # ── STAGE 6: Replay fabric routes + bandwidth simulation ───────────
    print("=== Stage 6: Replay fabric routes + bandwidth simulation ===")
    stage6_cmd = [
        py, STAGE_TOOLS["fabric_replay"],
        "--route-report", p_route_json,
        "--report-json", p_fab_replay_json,
    ]
    if args.target_tokens_per_sec is not None:
        stage6_cmd += ["--target-tokens-per-sec", str(args.target_tokens_per_sec)]
    if args.fabric_gib_sec is not None:
        stage6_cmd += ["--fabric-gib-sec", str(args.fabric_gib_sec)]
    if strict:
        stage6_cmd.append("--strict")
    rc, _, _ = _run_stage("fabric_replay", stage6_cmd)
    try:
        fab_replay_detail = _load_json(p_fab_replay_json, "Stage 6")
        stage_details["fabric_replay"] = fab_replay_detail
    except PipelineError:
        fab_replay_detail = {}

    fab_replay_status = _normalise(
        fab_replay_detail.get("status", "fail" if rc != 0 else "pass")
    )
    stage_statuses["fabric_replay"] = fab_replay_status

    # fabric_simulation_status comes from the fabric_status field (BW utilisation)
    raw_fabric_status = fab_replay_detail.get("fabric_status", "UNKNOWN")
    stage_statuses["fabric_simulation"] = _normalise(raw_fabric_status)

    if fab_replay_status == "fail":
        failures.append(
            f"Fabric route replay failed: "
            f"{fab_replay_detail.get('routes_failed', '?')} route(s) failed."
        )
    if _normalise(raw_fabric_status) == "fail":
        failures.append("Fabric bandwidth simulation: FAIL (oversubscribed).")
    elif _normalise(raw_fabric_status) == "warn":
        warnings.append("Fabric bandwidth simulation: WARN (approaching target).")
    print()

    # ── Determine overall status ───────────────────────────────────────
    all_statuses = list(stage_statuses.values())
    overall_status = _worst(*all_statuses)
    if overall_status in ("ok", "unknown"):
        overall_status = "pass"
    if failures:
        overall_status = "fail"
    elif overall_status == "pass" and warnings:
        overall_status = "warn"

    recommendation = _recommend(
        stage_statuses.get("validate_exec_plan", "pass"),
        stage_statuses.get("map_to_commands", "pass"),
        stage_statuses.get("mmio_replay", "pass"),
        stage_statuses.get("fabric_route_combined", "pass"),
        stage_statuses.get("fabric_replay", "pass"),
        raw_fabric_status,
    )

    return _finish(
        args, execution_plan_path, stage_statuses, stage_details,
        overall_status, failures, warnings, recommendation=recommendation,
    )


# ---------------------------------------------------------------------------
# Finish: print summary and write JSON report
# ---------------------------------------------------------------------------

def _finish(
    args,
    execution_plan_path: str,
    stage_statuses: dict[str, str],
    stage_details: dict[str, Any],
    overall_status: str,
    failures: list[str],
    warnings: list[str],
    recommendation: str = "",
) -> int:

    fab_replay_detail = stage_details.get("fabric_replay", {})
    raw_fabric_status = fab_replay_detail.get("fabric_status", "UNKNOWN") if fab_replay_detail else "UNKNOWN"

    recommendation = recommendation or _recommend(
        stage_statuses.get("validate_exec_plan", "pass"),
        stage_statuses.get("map_to_commands", "pass"),
        stage_statuses.get("mmio_replay", "pass"),
        stage_statuses.get("fabric_route_combined", "pass"),
        stage_statuses.get("fabric_replay", "pass"),
        raw_fabric_status,
    )

    report = _build_report(
        execution_plan_path, stage_statuses, stage_details,
        overall_status, recommendation,
    )

    # ── Print integrated summary ───────────────────────────────────────
    print("=" * 72)
    print("ATT-1 Execution/Replay Pipeline Summary (M132)")
    print("  NOTE: Control-plane simulation only — not inference execution,")
    print("  not PCIe hardware access.")
    print(f"  Execution plan    : {execution_plan_path}")
    print(f"  Tile count        : {report.get('tile_count', '?')}")
    print(f"  Command count     : {report.get('command_count', '?')}")
    print(f"  Route count       : {report.get('route_count', '?')}")
    print(f"  EXEC_* commands   : {report.get('exec_commands_seen', '?')}")
    print()
    labels = [
        ("Stage 1 exec plan validation",    "execution_plan_validation_status"),
        ("Stage 2 command plan mapping",    "command_plan_status"),
        ("Stage 3 MMIO replay",             "mmio_replay_status"),
        ("Stage 4+5 fabric routes",         "fabric_route_status"),
        ("Stage 6 fabric replay",           "fabric_replay_status"),
        ("Stage 6 fabric bandwidth sim",    "fabric_simulation_status"),
    ]
    for label, key in labels:
        status = report.get(key, "?")
        print(f"  {label:<36s}: {status}")
    print()
    req_gib = report.get("required_fabric_gib_sec")
    if req_gib is not None:
        print(f"  Required fabric bandwidth : {req_gib:.4e} GiB/s")
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
    print(f"  Final status      : {overall_status.upper()}")
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
        description="ATT-1 integrated execution/replay pipeline (M132)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--execution-plan", required=True, metavar="PATH",
        help="M125/M128 tensor execution plan JSON (pipeline input)",
    )
    ap.add_argument(
        "--tiles", type=int, default=None, metavar="N",
        help="Override tile count forwarded to MMIO emulator (1-16)",
    )
    ap.add_argument(
        "--tile-memory-mib", type=int, default=None, metavar="N",
        help="Tile memory capacity in MiB forwarded to MMIO emulator (1-256)",
    )
    ap.add_argument(
        "--kv-memory-mib", type=int, default=None, metavar="N",
        help="KV cache capacity per tile in MiB forwarded to MMIO emulator",
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
    if args.tiles is not None and not (1 <= args.tiles <= 16):
        errors.append("--tiles must be between 1 and 16")
    if args.tile_memory_mib is not None and not (1 <= args.tile_memory_mib <= 256):
        errors.append("--tile-memory-mib must be between 1 and 256")
    if args.kv_memory_mib is not None and args.kv_memory_mib <= 0:
        errors.append("--kv-memory-mib must be > 0")
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
