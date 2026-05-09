#!/usr/bin/env python3
"""
ATT-1 M134: schema and version compatibility regression tests.

Tests the check_schema_compat.py tool against both valid current-version
fixtures (expecting exit 0) and invalid/edge-case fixtures (expecting exit 1).

Coverage:
  - placement report: valid, future version, missing version, missing tiles,
    type mismatch, negative bytes, unknown field (default + strict)
  - command plan: valid, future version, missing commands, non-list commands,
    count mismatch
  - fabric route report: valid, future version, missing routes, count mismatch
  - execution plan: valid, future version, invalid status, type mismatch,
    count mismatch
  - pipeline report: valid, future version, invalid stage status
  - cross-schema: valid pair, tile_count mismatch
  - miscellaneous: JSON report output, no CUDA dependency, strict determinism,
    golden regression still passes

Exit codes:
  0 — all tests passed
  1 — one or more tests failed

Usage:
    python3 compiler/test_schema_compat.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
_CHECKER = str(_HERE / "check_schema_compat.py")
_GOLDEN = str(_HERE / "check_golden_regressions.py")
_FIX = _HERE / "fixtures"
_SC = _FIX / "schema_compat"
_PY = sys.executable

_COUNTERS = [0, 0]   # [pass, fail]
_RESULTS: list[tuple[str, str, str]] = []   # (status, name, detail)


def _run(schema: str, input_path: str | Path, extra: list[str] | None = None) -> int:
    cmd = [_PY, _CHECKER, "--schema", schema, "--input", str(input_path)]
    if extra:
        cmd.extend(extra)
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode


def _run_cross(
    placement: str | Path,
    cmd_plan: str | Path,
    extra: list[str] | None = None,
) -> int:
    cmd = [
        _PY, _CHECKER, "--schema", "cross",
        "--placement", str(placement),
        "--command-plan", str(cmd_plan),
    ]
    if extra:
        cmd.extend(extra)
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode


def _run_raw(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run([_PY, _CHECKER] + args, capture_output=True, text=True)


def _expect(name: str, expected_rc: int, actual_rc: int) -> None:
    if expected_rc == actual_rc:
        _COUNTERS[0] += 1
        _RESULTS.append(("PASS", name, ""))
        print(f"PASS: {name}")
    else:
        _COUNTERS[1] += 1
        _RESULTS.append(("FAIL", name, f"expected rc={expected_rc} got rc={actual_rc}"))
        print(f"FAIL: {name}: expected rc={expected_rc} got rc={actual_rc}")


# ---------------------------------------------------------------------------
# Placement report
# ---------------------------------------------------------------------------

_expect(
    "placement: valid current-version fixture passes",
    0,
    _run("placement", _FIX / "placement_report_valid.json"),
)

_expect(
    "placement: future version rejected (rc=1)",
    1,
    _run("placement", _SC / "placement_future_version.json"),
)

_expect(
    "placement: missing version field rejected (rc=1)",
    1,
    _run("placement", _SC / "placement_missing_version.json"),
)

_expect(
    "placement: missing tiles field rejected (rc=1)",
    1,
    _run("placement", _SC / "placement_missing_tiles.json"),
)

_expect(
    "placement: type mismatch (version as string) rejected (rc=1)",
    1,
    _run("placement", _SC / "placement_type_mismatch.json"),
)

_expect(
    "placement: negative model_bytes rejected (rc=1)",
    1,
    _run("placement", _SC / "placement_negative_bytes.json"),
)

_expect(
    "placement: unknown optional field accepted by default (rc=0)",
    0,
    _run("placement", _SC / "placement_unknown_field.json"),
)

_expect(
    "placement: unknown optional field is warning in strict mode (rc=0 → warn, exit=0)",
    0,
    _run("placement", _SC / "placement_unknown_field.json", ["--strict"]),
)

# ---------------------------------------------------------------------------
# Command plan
# ---------------------------------------------------------------------------

_expect(
    "command_plan: valid current-version fixture passes",
    0,
    _run("command_plan", _FIX / "plan_tiny_barrier_trace.json"),
)

_expect(
    "command_plan: future version rejected (rc=1)",
    1,
    _run("command_plan", _SC / "cmd_plan_future_version.json"),
)

_expect(
    "command_plan: missing commands field rejected (rc=1)",
    1,
    _run("command_plan", _SC / "cmd_plan_missing_commands.json"),
)

_expect(
    "command_plan: non-list commands rejected (rc=1)",
    1,
    _run("command_plan", _SC / "cmd_plan_nonlist_commands.json"),
)

_expect(
    "command_plan: command_count mismatch rejected (rc=1)",
    1,
    _run("command_plan", _SC / "cmd_plan_count_mismatch.json"),
)

# ---------------------------------------------------------------------------
# Fabric route report
# ---------------------------------------------------------------------------

_expect(
    "fabric_route: valid current-version fixture passes",
    0,
    _run("fabric_route", _FIX / "fabric_route_report_tiny.json"),
)

_expect(
    "fabric_route: future version rejected (rc=1)",
    1,
    _run("fabric_route", _SC / "route_future_version.json"),
)

_expect(
    "fabric_route: missing routes field rejected (rc=1)",
    1,
    _run("fabric_route", _SC / "route_missing_routes.json"),
)

_expect(
    "fabric_route: route_count mismatch rejected (rc=1)",
    1,
    _run("fabric_route", _SC / "route_count_mismatch.json"),
)

# ---------------------------------------------------------------------------
# Execution plan
# ---------------------------------------------------------------------------

_expect(
    "execution_plan: valid current-version fixture passes",
    0,
    _run("execution_plan", _FIX / "exec_plan_valid_tiny.json"),
)

_expect(
    "execution_plan: future version rejected (rc=1)",
    1,
    _run("execution_plan", _SC / "exec_future_version.json"),
)

_expect(
    "execution_plan: invalid status enum rejected (rc=1)",
    1,
    _run("execution_plan", _SC / "exec_invalid_status.json"),
)

_expect(
    "execution_plan: type mismatch (command_count as string) rejected (rc=1)",
    1,
    _run("execution_plan", _SC / "exec_type_mismatch.json"),
)

_expect(
    "execution_plan: command_count mismatch rejected (rc=1)",
    1,
    _run("execution_plan", _SC / "exec_count_mismatch.json"),
)

# ---------------------------------------------------------------------------
# Pipeline report
# ---------------------------------------------------------------------------

_expect(
    "pipeline: valid v132 fixture passes",
    0,
    _run("pipeline", _SC / "pipeline_v132_valid.json"),
)

_expect(
    "pipeline: future version rejected (rc=1)",
    1,
    _run("pipeline", _SC / "pipeline_future_version.json"),
)

_expect(
    "pipeline: invalid stage status rejected (rc=1)",
    1,
    _run("pipeline", _SC / "pipeline_invalid_stage_status.json"),
)

# ---------------------------------------------------------------------------
# Cross-schema consistency
# ---------------------------------------------------------------------------

_expect(
    "cross: matching tile_count (placement_valid + plan_tiny_barrier) passes",
    0,
    _run_cross(_FIX / "placement_report_valid.json", _FIX / "plan_tiny_barrier_trace.json"),
)

_expect(
    "cross: mismatched tile_count rejected (rc=1)",
    1,
    _run_cross(
        _FIX / "placement_report_valid.json",
        _SC / "cross_cmd_plan_tile_mismatch.json",
    ),
)

# ---------------------------------------------------------------------------
# Strict mode determinism: run twice, same exit code
# ---------------------------------------------------------------------------

_rc1 = _run("placement", _SC / "placement_unknown_field.json", ["--strict"])
_rc2 = _run("placement", _SC / "placement_unknown_field.json", ["--strict"])
if _rc1 == _rc2:
    _COUNTERS[0] += 1
    _RESULTS.append(("PASS", "strict mode deterministic (same exit code on two runs)", ""))
    print("PASS: strict mode deterministic (same exit code on two runs)")
else:
    _COUNTERS[1] += 1
    _RESULTS.append(("FAIL", "strict mode deterministic", f"rc1={_rc1} rc2={_rc2}"))
    print(f"FAIL: strict mode deterministic: rc1={_rc1} rc2={_rc2}")

# ---------------------------------------------------------------------------
# JSON report output is valid JSON and contains expected keys
# ---------------------------------------------------------------------------

def _test_report_json() -> None:
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        tmp = tf.name
    rc = _run(
        "placement",
        _FIX / "placement_report_valid.json",
        ["--report-json", tmp],
    )
    ok = True
    try:
        with open(tmp) as fh:
            report = json.load(fh)
        for key in ("schema", "status", "error_count", "warning_count", "errors", "warnings"):
            if key not in report:
                ok = False
                print(f"  report JSON missing key: {key!r}")
    except (json.JSONDecodeError, FileNotFoundError) as exc:
        ok = False
        print(f"  report JSON load failed: {exc}")
    finally:
        import os
        try:
            os.unlink(tmp)
        except OSError:
            pass

    if ok and rc == 0:
        _COUNTERS[0] += 1
        _RESULTS.append(("PASS", "JSON report output is valid and contains expected keys", ""))
        print("PASS: JSON report output is valid and contains expected keys")
    else:
        _COUNTERS[1] += 1
        _RESULTS.append(("FAIL", "JSON report output", f"rc={rc} ok={ok}"))
        print(f"FAIL: JSON report output: rc={rc} ok={ok}")


_test_report_json()

# ---------------------------------------------------------------------------
# No CUDA dependency
# ---------------------------------------------------------------------------

def _test_no_cuda() -> None:
    with open(_CHECKER) as fh:
        src = fh.read()
    cuda_refs = [
        tok for tok in ("cuda", "CUDA", "nvcc", "torch", "cupy")
        if tok in src
    ]
    if not cuda_refs:
        _COUNTERS[0] += 1
        _RESULTS.append(("PASS", "no_cuda: check_schema_compat.py has no CUDA dependency", ""))
        print("PASS: no_cuda: check_schema_compat.py has no CUDA dependency")
    else:
        _COUNTERS[1] += 1
        _RESULTS.append(("FAIL", "no_cuda", f"found refs: {cuda_refs}"))
        print(f"FAIL: no_cuda: unexpected references: {cuda_refs}")


_test_no_cuda()

# ---------------------------------------------------------------------------
# Golden regression still passes
# ---------------------------------------------------------------------------

def _test_golden_regression() -> None:
    r = subprocess.run([_PY, _GOLDEN], capture_output=True, text=True)
    if r.returncode == 0:
        _COUNTERS[0] += 1
        _RESULTS.append(("PASS", "golden regression check still passes", ""))
        print("PASS: golden regression check still passes")
    else:
        _COUNTERS[1] += 1
        detail = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[:200]
        _RESULTS.append(("FAIL", "golden regression check", detail))
        print(f"FAIL: golden regression check: rc={r.returncode}: {detail}")


_test_golden_regression()

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

total = _COUNTERS[0] + _COUNTERS[1]
print("-" * 72)
print(f"schema_compat: {_COUNTERS[0]} PASS  {_COUNTERS[1]} FAIL")
sys.exit(0 if _COUNTERS[1] == 0 else 1)
