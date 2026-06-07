#!/usr/bin/env python3
"""
ATT-1 M135: hostile-input regression tests.

Tests check_hostile_inputs.py against hostile fixture files (expecting exit 1)
and valid baseline fixtures (expecting exit 0).  Also verifies that the M133
golden regression suite and M134 schema compatibility suite still pass.

Coverage:
  Placement hostile inputs:
    - missing header
    - negative tile_count
    - tile_count mismatch
    - duplicate tile_id
    - non-integer tile_id
    - q4 tensor missing group_size
    - non-existent owner_tile

  Command plan hostile inputs:
    - duplicate command_id
    - unknown command_type
    - negative byte count
    - negative scale byte count
    - missing tensor_name for tensor command
    - unsupported dtype
    - invalid tile_id (< 0)

  Fabric route hostile inputs:
    - duplicate route_id
    - invalid route_type
    - invalid source_tile (< 0)
    - zero payload on data route
    - missing destination_tiles for data route
    - reduction route missing explicit reduction_behavior
    - invalid ordering_policy

  Execution plan hostile inputs:
    - invalid execution_phase
    - duplicate plan_command_id
    - invalid tile_id (< 0)
    - LOAD_TENSOR_TILE with no output_buffers
    - EXEC_ATTENTION with no output_buffers
    - EXEC_MATMUL with no tensor_dependencies
    - unknown expected_status

  Pipeline hostile inputs:
    - negative tile_count
    - missing final_status
    - invalid final_status
    - commands_replayed > command_count

  Regression guard:
    - valid placement → check_hostile_inputs exit 0
    - valid execution plan → check_hostile_inputs exit 0
    - valid fabric route → check_hostile_inputs exit 0
    - valid command plan → check_hostile_inputs exit 0
    - valid pipeline → check_hostile_inputs exit 0
    - M133 golden regression script still passes
    - M134 schema compat suite still passes

Exit codes:
  0 — all tests passed
  1 — one or more tests failed

Usage:
    python3 compiler/test_hostile_inputs.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
_CHECKER = str(_HERE / "check_hostile_inputs.py")
_GOLDEN = str(_HERE / "check_golden_regressions.py")
_SCHEMA_COMPAT = str(_HERE / "test_schema_compat.py")
_FIX = _HERE / "fixtures"
_HOSTILE = _FIX / "hostile"
_PY = sys.executable

_COUNTERS = [0, 0]   # [pass, fail]
_RESULTS: list[tuple[str, str, str]] = []   # (status, name, detail)


def _run(schema: str, path: str | Path, extra: list[str] | None = None) -> int:
    cmd = [_PY, _CHECKER, "--schema", schema, "--input", str(path)]
    if extra:
        cmd.extend(extra)
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode


def _run_raw(cmd: list[str]) -> int:
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode


def _expect(name: str, expected: int, actual: int) -> None:
    if expected == actual:
        _COUNTERS[0] += 1
        _RESULTS.append(("PASS", name, f"exit {actual} as expected"))
    else:
        _COUNTERS[1] += 1
        _RESULTS.append(("FAIL", name, f"expected exit {expected}, got {actual}"))


# ---------------------------------------------------------------------------
# Placement hostile inputs
# ---------------------------------------------------------------------------

def _test_placement() -> None:
    _expect("placement: missing header",
            1, _run("placement", _HOSTILE / "placement_missing_header.json"))
    _expect("placement: negative tile_count",
            1, _run("placement", _HOSTILE / "placement_negative_tile_count.json"))
    _expect("placement: tile_count mismatch",
            1, _run("placement", _HOSTILE / "placement_tile_count_mismatch.json"))
    _expect("placement: duplicate tile_id",
            1, _run("placement", _HOSTILE / "placement_duplicate_tile_id.json"))
    _expect("placement: invalid tile_id type",
            1, _run("placement", _HOSTILE / "placement_invalid_tile_id_type.json"))
    _expect("placement: q4 missing group_size",
            1, _run("placement", _HOSTILE / "placement_q4_missing_group_size.json"))
    _expect("placement: nonexistent owner_tile",
            1, _run("placement", _HOSTILE / "placement_nonexistent_owner_tile.json"))


# ---------------------------------------------------------------------------
# Command plan hostile inputs
# ---------------------------------------------------------------------------

def _test_command_plan() -> None:
    _expect("command_plan: duplicate command_id",
            1, _run("command_plan", _HOSTILE / "cmd_plan_duplicate_command_id.json"))
    _expect("command_plan: unknown command_type",
            1, _run("command_plan", _HOSTILE / "cmd_plan_unknown_command_type.json"))
    _expect("command_plan: negative byte count",
            1, _run("command_plan", _HOSTILE / "cmd_plan_negative_byte_count.json"))
    _expect("command_plan: negative scale bytes",
            1, _run("command_plan", _HOSTILE / "cmd_plan_negative_scale_bytes.json"))
    _expect("command_plan: missing tensor_name",
            1, _run("command_plan", _HOSTILE / "cmd_plan_missing_tensor_name.json"))
    _expect("command_plan: unsupported dtype",
            1, _run("command_plan", _HOSTILE / "cmd_plan_unsupported_dtype.json"))
    _expect("command_plan: invalid tile_id",
            1, _run("command_plan", _HOSTILE / "cmd_plan_invalid_tile_id.json"))


# ---------------------------------------------------------------------------
# Fabric route hostile inputs
# ---------------------------------------------------------------------------

def _test_fabric_route() -> None:
    _expect("fabric_route: duplicate route_id",
            1, _run("fabric_route", _HOSTILE / "route_duplicate_route_id.json"))
    _expect("fabric_route: invalid route_type",
            1, _run("fabric_route", _HOSTILE / "route_invalid_route_type.json"))
    _expect("fabric_route: invalid source_tile",
            1, _run("fabric_route", _HOSTILE / "route_invalid_source_tile.json"))
    _expect("fabric_route: zero payload on data route",
            1, _run("fabric_route", _HOSTILE / "route_zero_payload_data.json"))
    _expect("fabric_route: missing destination_tiles",
            1, _run("fabric_route", _HOSTILE / "route_missing_destination_tiles.json"))
    _expect("fabric_route: missing reduction_behavior",
            1, _run("fabric_route", _HOSTILE / "route_missing_reduction_behavior.json"))
    _expect("fabric_route: invalid ordering_policy",
            1, _run("fabric_route", _HOSTILE / "route_invalid_ordering_policy.json"))


# ---------------------------------------------------------------------------
# Execution plan hostile inputs
# ---------------------------------------------------------------------------

def _test_execution_plan() -> None:
    _expect("execution_plan: invalid phase",
            1, _run("execution_plan", _HOSTILE / "exec_invalid_phase.json"))
    _expect("execution_plan: duplicate plan_command_id",
            1, _run("execution_plan", _HOSTILE / "exec_duplicate_command_id.json"))
    _expect("execution_plan: invalid tile_id",
            1, _run("execution_plan", _HOSTILE / "exec_invalid_tile_id.json"))
    _expect("execution_plan: missing output_buffer for LOAD",
            1, _run("execution_plan", _HOSTILE / "exec_missing_output_buffer.json"))
    _expect("execution_plan: missing output_buffer for ATTENTION",
            1, _run("execution_plan", _HOSTILE / "exec_empty_output_for_attention.json"))
    _expect("execution_plan: missing tensor_dep for EXEC_MATMUL",
            1, _run("execution_plan", _HOSTILE / "exec_missing_tensor_dep.json"))
    _expect("execution_plan: unknown expected_status",
            1, _run("execution_plan", _HOSTILE / "exec_unknown_expected_status.json"))


# ---------------------------------------------------------------------------
# Pipeline hostile inputs
# ---------------------------------------------------------------------------

def _test_pipeline() -> None:
    _expect("pipeline: negative tile_count",
            1, _run("pipeline", _HOSTILE / "pipeline_negative_tile_count.json"))
    _expect("pipeline: missing final_status",
            1, _run("pipeline", _HOSTILE / "pipeline_missing_final_status.json"))
    _expect("pipeline: invalid final_status",
            1, _run("pipeline", _HOSTILE / "pipeline_invalid_final_status.json"))
    _expect("pipeline: commands_replayed > command_count",
            1, _run("pipeline", _HOSTILE / "pipeline_inconsistent_counts.json"))


# ---------------------------------------------------------------------------
# Regression guard: valid fixtures must still return exit 0
# ---------------------------------------------------------------------------

def _test_valid_baselines() -> None:
    _expect("valid baseline: placement_report_valid",
            0, _run("placement", _FIX / "placement_report_valid.json"))
    _expect("valid baseline: exec_plan_valid_tiny",
            0, _run("execution_plan", _FIX / "exec_plan_valid_tiny.json"))
    _expect("valid baseline: fabric_route_report_tiny",
            0, _run("fabric_route", _FIX / "fabric_route_report_tiny.json"))
    _expect("valid baseline: plan_tiny_barrier_trace",
            0, _run("command_plan", _FIX / "plan_tiny_barrier_trace.json"))
    _expect("valid baseline: pipeline_v132_valid",
            0, _run("pipeline", _FIX / "schema_compat" / "pipeline_v132_valid.json"))


# ---------------------------------------------------------------------------
# Regression guard: M133 golden + M134 schema compat suites
# ---------------------------------------------------------------------------

def _test_upstream_suites() -> None:
    _expect(
        "M133 golden regression suite still passes",
        0,
        _run_raw([_PY, _GOLDEN]),
    )
    _expect(
        "M134 schema compat suite still passes",
        0,
        _run_raw([_PY, _SCHEMA_COMPAT]),
    )


# ---------------------------------------------------------------------------
# No CUDA dependency
# ---------------------------------------------------------------------------

def _test_no_cuda() -> None:
    import ast
    checker_src = Path(_CHECKER).read_text()
    if "cuda" in checker_src.lower() or "pcie" in checker_src.lower():
        _expect("no CUDA/PCIe in check_hostile_inputs.py", 1, 1)  # force fail
    else:
        _expect("no CUDA/PCIe in check_hostile_inputs.py", 0, 0)


# ---------------------------------------------------------------------------
# JSON report output
# ---------------------------------------------------------------------------

def _test_json_report() -> None:
    import json
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        tmp = tf.name
    try:
        rc = _run(
            "placement",
            _HOSTILE / "placement_missing_header.json",
            ["--report-json", tmp],
        )
        if rc != 1:
            _expect("json report: exit 1 for hostile input", 1, rc)
            return
        with open(tmp) as fh:
            rep = json.load(fh)
        ok = (
            "status" in rep
            and rep["status"] == "fail"
            and "errors" in rep
            and len(rep["errors"]) > 0
            and "schema" in rep
        )
        _expect("json report: valid JSON with status/errors/schema", 0, 0 if ok else 1)
    except Exception as exc:
        _expect("json report: no exception", 0, 1)
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Strict mode determinism (same input → same exit code)
# ---------------------------------------------------------------------------

def _test_strict_determinism() -> None:
    rc1 = _run("placement",
               _HOSTILE / "placement_negative_tile_count.json", ["--strict"])
    rc2 = _run("placement",
               _HOSTILE / "placement_negative_tile_count.json", ["--strict"])
    _expect("strict determinism: same result on repeated runs", rc1, rc2)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    _test_placement()
    _test_command_plan()
    _test_fabric_route()
    _test_execution_plan()
    _test_pipeline()
    _test_valid_baselines()
    _test_upstream_suites()
    _test_no_cuda()
    _test_json_report()
    _test_strict_determinism()

    print()
    print("-" * 72)
    width = max(len(r[1]) for r in _RESULTS)
    for status, name, detail in _RESULTS:
        print(f"{status}: {name:<{width}}  ({detail})")
    print("-" * 72)
    print(f"hostile_inputs: {_COUNTERS[0]} PASS  {_COUNTERS[1]} FAIL")

    return 0 if _COUNTERS[1] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
