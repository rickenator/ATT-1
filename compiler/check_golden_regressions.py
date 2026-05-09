#!/usr/bin/env python3
"""
ATT-1 M133: deterministic golden regression baseline checker.

Runs each ATT-1 planning/replay tool against a small checked-in fixture,
extracts a stable set of semantic fields, and compares against a checked-in
golden JSON file.  Used to detect CLI/output/schema drift without measuring
performance or runtime behaviour.

Golden baselines live in compiler/fixtures/golden/.  They contain only stable
semantic fields (status, counts, command types, etc.).  Volatile fields
(timestamps, absolute paths, elapsed times, host-specific values) are
excluded from the comparison.

Exit codes:
  0 — all checks passed (or golden files updated without error)
  1 — one or more checks failed

Usage:
    # Compare live output against stored golden baselines (default):
    python3 compiler/check_golden_regressions.py
    python3 compiler/check_golden_regressions.py --check

    # Regenerate golden baselines from current tool output:
    python3 compiler/check_golden_regressions.py --update-golden

    # Use an alternate fixtures directory:
    python3 compiler/check_golden_regressions.py --fixture-dir path/to/fixtures

How to update baselines intentionally:
    After a deliberate tool change that alters stable output fields, run with
    --update-golden to record the new expected values, then commit the updated
    golden files alongside the tool change.

Note:
    Checks g05 and g11 invoke compiled C binaries
    (build/att1-aimu-replay and build/att1-aimu-mmio-replay respectively).
    Run 'make' before running this script if those binaries are absent.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
_FIXTURES_DEFAULT = _HERE / "fixtures"
_GOLDEN_SUBDIR = "golden"

_AIMU_REPLAY_BIN = _REPO / "build" / "att1-aimu-replay"
_AIMU_MMIO_REPLAY_BIN = _REPO / "build" / "att1-aimu-mmio-replay"

_PY = sys.executable


# ---------------------------------------------------------------------------
# Check specification
# ---------------------------------------------------------------------------

@dataclass
class _Check:
    id: str
    label: str
    tool: str                                   # path relative to compiler/
    make_args: Callable[[Path, Path], list[str]]  # (fixtures_dir, work_dir) -> args
    output: str                                 # intermediate JSON filename in work_dir
    golden: str                                 # golden filename in golden_dir
    extract: Callable[[dict], dict]             # live JSON → stable fields dict
    skip_if: Callable[[], str | None] = field(
        default_factory=lambda: lambda: None    # return reason string or None
    )


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------

def _pick(d: dict, *keys: str) -> dict:
    """Return a dict of {key: d[key]} for each key present in d."""
    return {k: d[k] for k in keys if k in d}


def _nested(d: dict, path: str) -> Any:
    """Navigate a dotted path like 'header.tile_count' into a nested dict."""
    parts = path.split(".")
    cur = d
    for p in parts:
        if not isinstance(cur, dict) or p not in cur:
            return None
    cur = cur[p]
    return cur


# ---------------------------------------------------------------------------
# Per-check extraction functions
# ---------------------------------------------------------------------------

def _extract_g01(d: dict) -> dict:
    """placement report validation — stable fields."""
    return _pick(d,
                 "status", "total_warnings", "total_errors",
                 "tile_count_header", "tensor_record_count", "tile_record_count")


def _extract_g02(d: dict) -> dict:
    """placement advisory — stable fields."""
    analysis = d.get("analysis", {})
    return {
        "status": d.get("status"),
        "proposal_count": d.get("proposal_count"),
        "analysis": _pick(analysis,
                          "tile_count", "total_model_bytes",
                          "capacity_fail_tiles", "capacity_warn_tiles",
                          "bandwidth_fail_tiles", "bandwidth_warn_tiles"),
    }


def _extract_g03(d: dict) -> dict:
    """placement scenario — stable fields."""
    orig = d.get("original", {})
    rec = d.get("recommendation", {})
    return {
        "original": _pick(orig, "tile_count", "total_model_bytes", "total_kv_bytes"),
        "recommendation": _pick(rec, "tile_count", "capacity_status"),
        "pass_count": d.get("pass_count"),
        "total_count": d.get("total_count"),
    }


def _extract_g04(d: dict) -> dict:
    """command plan from placement — stable fields."""
    hdr = d.get("header", {})
    summ = d.get("summary", {})
    return {
        "header": _pick(hdr, "tile_count", "tensor_count", "command_count", "status"),
        "summary": {
            **_pick(summ,
                    "commands_by_type", "total_tensor_bytes",
                    "f32_tensor_count", "q8_tensor_count", "q4_tensor_count",
                    "capacity_failures_observed", "warnings_observed"),
        },
    }


def _extract_g05(d: dict) -> dict:
    """command replay (M113) — stable fields."""
    return _pick(d,
                 "tile_count", "command_count",
                 "commands_replayed", "completions_seen",
                 "failed_commands", "unsupported_commands",
                 "dma_validations", "doorbell_count", "status")


def _extract_g06(d: dict) -> dict:
    """fabric route report — stable header fields."""
    hdr = d.get("header", {})
    return {
        "header": _pick(hdr,
                        "route_count", "packet_count_estimate",
                        "payload_bytes_estimate", "tile_count",
                        "fabric_policy", "status"),
    }


def _extract_g07(d: dict) -> dict:
    """fabric bandwidth simulation — stable header + aggregate fields."""
    hdr = d.get("header", {})
    agg = d.get("aggregate", {})
    return {
        "header": _pick(hdr,
                        "sim_version", "tile_count", "route_count",
                        "target_tokens_per_sec", "fabric_gib_sec_target", "status"),
        "aggregate": _pick(agg, "total_payload_bytes_per_token", "status"),
    }


def _extract_g08(d: dict) -> dict:
    """fabric route replay — stable fields."""
    return _pick(d,
                 "route_count", "routes_replayed", "routes_failed",
                 "tile_count",
                 "aggregate_packets_sent", "aggregate_packets_received",
                 "aggregate_payload_bytes_sent", "aggregate_payload_bytes_received",
                 "reductions_started", "reductions_completed",
                 "barriers_started", "barriers_completed",
                 "trace_events", "fabric_status", "status")


def _extract_g09(d: dict) -> dict:
    """execution-plan validation — stable fields."""
    return _pick(d,
                 "status", "total_warnings", "total_errors",
                 "phase_count", "command_count",
                 "route_ref_count", "buffer_ref_count")


def _extract_g10(d: dict) -> dict:
    """execution-plan → command mapping — stable fields."""
    hdr = d.get("header", {})
    summ = d.get("summary", {})
    return {
        "header": _pick(hdr, "tile_count", "tensor_count", "command_count", "status"),
        "summary": _pick(summ,
                         "commands_by_type", "total_tensor_bytes",
                         "f32_tensor_count", "q8_tensor_count", "q4_tensor_count",
                         "warnings_observed", "unsupported_exec_commands"),
    }


def _extract_g11(d: dict) -> dict:
    """integrated execution/replay pipeline — stable fields."""
    return _pick(d,
                 "pipeline_version",
                 "execution_plan_validation_status", "command_plan_status",
                 "mmio_replay_status", "fabric_route_status",
                 "fabric_replay_status", "fabric_simulation_status",
                 "tile_count", "command_count", "route_count",
                 "commands_replayed", "completions_seen", "exec_commands_seen",
                 "failed_commands", "unsupported_commands",
                 "aggregate_packets_sent", "aggregate_payload_bytes_sent",
                 "required_fabric_gib_sec", "fabric_status", "final_status")


def _extract_g99(d: dict) -> dict:
    """malformed exec plan validation — expects fail."""
    return _pick(d, "status", "total_errors")


# ---------------------------------------------------------------------------
# Check registry
# The checks chain through a shared workdir:
#   g04 output  → used by g05, g06
#   g06 output  → used by g07, g08
# All other checks are independent of each other.
# ---------------------------------------------------------------------------

_CHECKS: list[_Check] = [
    _Check(
        id="g01",
        label="placement report validation (M100/M99)",
        tool="validate_tensor_placement_report.py",
        make_args=lambda fix, work: [
            "--report", str(fix / "placement_report_valid.json"),
            "--report-json", str(work / "g01.json"),
        ],
        output="g01.json",
        golden="g01_placement_validation.json",
        extract=_extract_g01,
    ),
    _Check(
        id="g02",
        label="placement advisory (M101)",
        tool="propose_tensor_placement.py",
        make_args=lambda fix, work: [
            "--report", str(fix / "placement_report_valid.json"),
            "--report-json", str(work / "g02.json"),
        ],
        output="g02.json",
        golden="g02_placement_advisory.json",
        extract=_extract_g02,
    ),
    _Check(
        id="g03",
        label="placement scenario generation (M102)",
        tool="propose_tensor_scenarios.py",
        make_args=lambda fix, work: [
            "--report", str(fix / "placement_report_valid.json"),
            "--report-json", str(work / "g03.json"),
        ],
        output="g03.json",
        golden="g03_placement_scenario.json",
        extract=_extract_g03,
    ),
    _Check(
        id="g04",
        label="placement → command plan (M109)",
        tool="map_placement_to_commands.py",
        make_args=lambda fix, work: [
            "--report", str(fix / "placement_report_valid.json"),
            "--plan-json", str(work / "g04.json"),
        ],
        output="g04.json",
        golden="g04_command_plan.json",
        extract=_extract_g04,
    ),
    _Check(
        id="g05",
        label="command plan replay (M113)",
        tool="replay_aimu_command_plan.py",
        make_args=lambda fix, work: [
            "--plan", str(work / "g04.json"),
            "--report-json", str(work / "g05.json"),
        ],
        output="g05.json",
        golden="g05_command_replay.json",
        extract=_extract_g05,
        skip_if=lambda: (
            None if _AIMU_REPLAY_BIN.exists()
            else f"build/att1-aimu-replay not found; run 'make' first"
        ),
    ),
    _Check(
        id="g06",
        label="command plan → fabric routes (M117)",
        tool="map_commands_to_fabric_routes.py",
        make_args=lambda fix, work: [
            "--plan", str(work / "g04.json"),
            "--route-report-json", str(work / "g06.json"),
        ],
        output="g06.json",
        golden="g06_fabric_routes.json",
        extract=_extract_g06,
    ),
    _Check(
        id="g07",
        label="fabric bandwidth simulation (M118)",
        tool="simulate_fabric_bandwidth.py",
        make_args=lambda fix, work: [
            "--route-report", str(work / "g06.json"),
            "--target-tokens-per-sec", "50",
            "--fabric-gib-sec", "32",
            "--report-json", str(work / "g07.json"),
        ],
        output="g07.json",
        golden="g07_bw_simulation.json",
        extract=_extract_g07,
    ),
    _Check(
        id="g08",
        label="fabric route replay (M123/M118)",
        tool="replay_fabric_routes.py",
        make_args=lambda fix, work: [
            "--route-report", str(work / "g06.json"),
            "--target-tokens-per-sec", "50",
            "--fabric-gib-sec", "32",
            "--report-json", str(work / "g08.json"),
        ],
        output="g08.json",
        golden="g08_fabric_replay.json",
        extract=_extract_g08,
    ),
    _Check(
        id="g09",
        label="execution-plan validation (M128)",
        tool="validate_tensor_execution_plan.py",
        make_args=lambda fix, work: [
            "--plan", str(fix / "exec_plan_valid_tiny.json"),
            "--report-json", str(work / "g09.json"),
        ],
        output="g09.json",
        golden="g09_exec_plan_validation.json",
        extract=_extract_g09,
    ),
    _Check(
        id="g10",
        label="execution-plan → command mapping (M129)",
        tool="map_execution_plan_to_commands.py",
        make_args=lambda fix, work: [
            "--execution-plan", str(fix / "exec_plan_valid_tiny.json"),
            "--plan-json", str(work / "g10.json"),
        ],
        output="g10.json",
        golden="g10_exec_to_commands.json",
        extract=_extract_g10,
    ),
    _Check(
        id="g11",
        label="integrated execution/replay pipeline (M132)",
        tool="run_execution_replay_pipeline.py",
        make_args=lambda fix, work: [
            "--execution-plan", str(fix / "exec_plan_valid_tiny.json"),
            "--target-tokens-per-sec", "50",
            "--fabric-gib-sec", "32",
            "--workdir", str(work / "g11_workdir"),
            "--report-json", str(work / "g11.json"),
        ],
        output="g11.json",
        golden="g11_exec_replay_pipeline.json",
        extract=_extract_g11,
        skip_if=lambda: (
            None if _AIMU_MMIO_REPLAY_BIN.exists()
            else f"build/att1-aimu-mmio-replay not found; run 'make' first"
        ),
    ),
    # Negative test: malformed exec plan must produce status=fail
    _Check(
        id="g99",
        label="exec-plan validation of malformed fixture (expects fail)",
        tool="validate_tensor_execution_plan.py",
        make_args=lambda fix, work: [
            "--plan", str(fix / "exec_plan_missing_header.json"),
            "--report-json", str(work / "g99.json"),
        ],
        output="g99.json",
        golden="g99_malformed_exec_plan.json",
        extract=_extract_g99,
    ),
]


# ---------------------------------------------------------------------------
# Core comparison logic
# ---------------------------------------------------------------------------

def _deep_equal(a: Any, b: Any) -> bool:
    """Recursively compare two values for equality."""
    if type(a) != type(b):
        # Allow int/float comparison
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return a == b
        return False
    if isinstance(a, dict):
        if set(a.keys()) != set(b.keys()):
            return False
        return all(_deep_equal(a[k], b[k]) for k in a)
    if isinstance(a, list):
        if len(a) != len(b):
            return False
        return all(_deep_equal(x, y) for x, y in zip(a, b))
    return a == b


def _diff_lines(a: Any, b: Any, path: str = "") -> list[str]:
    """Return human-readable diff lines between two values."""
    lines: list[str] = []
    if isinstance(a, dict) and isinstance(b, dict):
        all_keys = sorted(set(a.keys()) | set(b.keys()))
        for k in all_keys:
            sub = f"{path}.{k}" if path else k
            if k not in b:
                lines.append(f"  {sub}: live={a[k]!r}  MISSING in golden")
            elif k not in a:
                lines.append(f"  {sub}: MISSING in live  golden={b[k]!r}")
            else:
                lines.extend(_diff_lines(a[k], b[k], sub))
    elif not _deep_equal(a, b):
        lines.append(f"  {path}: live={a!r}  golden={b!r}")
    return lines


# ---------------------------------------------------------------------------
# Tool runner
# ---------------------------------------------------------------------------

def _run_tool(tool: str, extra_args: list[str]) -> tuple[int, str, str]:
    """Run a compiler/ Python tool and return (returncode, stdout, stderr)."""
    cmd = [_PY, str(_HERE / tool)] + extra_args
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode, result.stdout, result.stderr


def _load_json(path: Path, label: str) -> dict | None:
    """Load a JSON file; return None and print error on failure."""
    try:
        with open(path) as fh:
            return json.load(fh)
    except FileNotFoundError:
        print(f"  ERROR: {label}: file not found: {path}")
        return None
    except json.JSONDecodeError as exc:
        print(f"  ERROR: {label}: malformed JSON at {path}: {exc}")
        return None


# ---------------------------------------------------------------------------
# Single-check runner
# ---------------------------------------------------------------------------

_STATUS_PASS = "PASS"
_STATUS_FAIL = "FAIL"
_STATUS_SKIP = "SKIP"


def _run_check(
    check: _Check,
    fixtures_dir: Path,
    work_dir: Path,
    golden_dir: Path,
    mode: str,
) -> str:
    """
    Run one check in 'check' or 'update' mode.
    Returns _STATUS_PASS, _STATUS_FAIL, or _STATUS_SKIP.
    """
    # ── Skip condition ─────────────────────────────────────────────────
    skip_reason = check.skip_if()
    if skip_reason:
        print(f"  SKIP: {skip_reason}")
        return _STATUS_SKIP

    # ── Build args and run tool ────────────────────────────────────────
    args = check.make_args(fixtures_dir, work_dir)

    # Ensure workdir exists for nested workdirs (e.g. g11)
    for a in args:
        if a.startswith(str(work_dir)) and "workdir" in a.lower():
            Path(a).mkdir(parents=True, exist_ok=True)

    rc, stdout, stderr = _run_tool(check.tool, args)

    # Load tool output
    output_path = work_dir / check.output
    live_raw = _load_json(output_path, f"{check.id} tool output")
    if live_raw is None:
        print(f"  Tool stderr: {stderr[:300]}" if stderr else "  (no stderr)")
        return _STATUS_FAIL

    live_stable = check.extract(live_raw)

    if mode == "update":
        # ── Update golden ──────────────────────────────────────────────
        golden_path = golden_dir / check.golden
        golden_dir.mkdir(parents=True, exist_ok=True)
        try:
            with open(golden_path, "w") as fh:
                json.dump(live_stable, fh, indent=2)
                fh.write("\n")
            print(f"  Updated: {golden_path.relative_to(_REPO)}")
            return _STATUS_PASS
        except OSError as exc:
            print(f"  ERROR: could not write golden file: {exc}")
            return _STATUS_FAIL

    else:
        # ── Check mode ─────────────────────────────────────────────────
        golden_path = golden_dir / check.golden
        golden = _load_json(golden_path, f"{check.id} golden")
        if golden is None:
            print(f"  Run with --update-golden to create: {golden_path}")
            return _STATUS_FAIL

        if _deep_equal(live_stable, golden):
            return _STATUS_PASS
        else:
            diffs = _diff_lines(live_stable, golden)
            print(f"  MISMATCH (live vs golden):")
            for line in diffs:
                print(line)
            return _STATUS_FAIL


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="ATT-1 M133 golden regression checker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    mode_group = ap.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--check", dest="mode", action="store_const", const="check",
        help="Compare live output against golden baselines (default)",
    )
    mode_group.add_argument(
        "--update-golden", dest="mode", action="store_const", const="update",
        help="Regenerate golden baseline files from current tool output",
    )
    ap.add_argument(
        "--fixture-dir", default=None, metavar="PATH",
        help="Directory containing input fixtures (default: compiler/fixtures/)",
    )
    ap.set_defaults(mode="check")
    return ap.parse_args()


def main() -> int:
    args = _parse_args()
    mode = args.mode

    fixtures_dir = Path(args.fixture_dir) if args.fixture_dir else _FIXTURES_DEFAULT
    golden_dir = fixtures_dir / _GOLDEN_SUBDIR

    if not fixtures_dir.is_dir():
        print(f"error: fixture directory not found: {fixtures_dir}", file=sys.stderr)
        return 1

    mode_label = "UPDATE GOLDEN" if mode == "update" else "CHECK"
    print(f"ATT-1 M133 golden regression — {mode_label}")
    print(f"  Fixtures  : {fixtures_dir}")
    print(f"  Golden dir: {golden_dir}")
    print(f"  NOTE: drift detection only — not performance measurement.")
    print("-" * 72)

    n_pass = 0
    n_fail = 0
    n_skip = 0

    with tempfile.TemporaryDirectory(prefix="att1_golden_") as tmpdir:
        work_dir = Path(tmpdir)

        for check in _CHECKS:
            label = f"[{check.id}] {check.label}"
            print(f"{label}")

            result = _run_check(check, fixtures_dir, work_dir, golden_dir, mode)

            if result == _STATUS_PASS:
                n_pass += 1
                print(f"  → PASS")
            elif result == _STATUS_SKIP:
                n_skip += 1
            else:
                n_fail += 1
                print(f"  → FAIL")

    print("-" * 72)
    skip_msg = f"  {n_skip} SKIP" if n_skip else ""
    print(f"M133 golden regressions: {n_pass} PASS  {n_fail} FAIL{skip_msg}")

    if mode == "update" and n_fail == 0:
        print("Golden files updated. Commit the changed files in compiler/fixtures/golden/")

    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
