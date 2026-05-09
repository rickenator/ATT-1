#!/usr/bin/env python3
"""
ATT-1 M132 execution/replay pipeline test suite.

Run with:
    python3 compiler/test_execution_replay_pipeline.py

Tests invoke run_execution_replay_pipeline.py via subprocess using fixtures
from compiler/fixtures/.  The M122 MMIO-replay binary
(build/att1-aimu-mmio-replay) is required for MMIO-stage tests; when absent
those tests emit a graceful skip/warn.

Exit code: 0 if all tests pass, 1 if any fail.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
_PIPELINE = str(_HERE / "run_execution_replay_pipeline.py")
_FIXTURES = _HERE / "fixtures"

_VALID_TINY = str(_FIXTURES / "exec_plan_valid_tiny.json")
_MISSING_HEADER = str(_FIXTURES / "exec_plan_missing_header.json")
_INVALID_TILE = str(_FIXTURES / "exec_plan_invalid_tile.json")

_BINARY = _REPO / "build" / "att1-aimu-mmio-replay"

_PY = sys.executable

# Expected set of top-level report keys
_REQUIRED_REPORT_KEYS = {
    "pipeline_version",
    "execution_plan_path",
    "execution_plan_validation_status",
    "command_plan_status",
    "mmio_replay_status",
    "fabric_route_status",
    "fabric_replay_status",
    "fabric_simulation_status",
    "tile_count",
    "command_count",
    "route_count",
    "commands_replayed",
    "completions_seen",
    "exec_commands_seen",
    "failed_commands",
    "unsupported_commands",
    "aggregate_packets_sent",
    "aggregate_payload_bytes_sent",
    "required_fabric_gib_sec",
    "fabric_status",
    "recommended_next_action",
    "final_status",
}

# ---------------------------------------------------------------------------
# Test framework
# ---------------------------------------------------------------------------

_g_pass = 0
_g_fail = 0


def _run(args: list[str], workdir: str | None = None) -> tuple[int, dict | None, str]:
    """
    Invoke the pipeline with *args*.  Returns (returncode, report_dict, stderr).
    report_dict is None if no --report-json arg or file not written.
    """
    report_path: str | None = None
    for i, a in enumerate(args):
        if a == "--report-json" and i + 1 < len(args):
            report_path = args[i + 1]
            break

    cmd = [_PY, _PIPELINE] + args
    result = subprocess.run(cmd, capture_output=True, text=True)

    report: dict | None = None
    if report_path and os.path.exists(report_path):
        try:
            with open(report_path) as fh:
                report = json.load(fh)
        except (json.JSONDecodeError, OSError):
            pass

    return result.returncode, report, result.stderr


def _pass(name: str) -> None:
    global _g_pass
    _g_pass += 1
    print(f"PASS: M132: {name}")


def _fail(name: str, reason: str = "") -> None:
    global _g_fail
    _g_fail += 1
    msg = f"FAIL: M132: {name}"
    if reason:
        msg += f"  ({reason})"
    print(msg)


def _assert(cond: bool, name: str, reason: str = "") -> bool:
    if cond:
        _pass(name)
    else:
        _fail(name, reason)
    return cond


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_valid_tiny_exits_zero() -> None:
    """Valid tiny fixture exits 0 (pass or warn without --strict)."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        rpath = tf.name
    try:
        rc, report, _ = _run([
            "--execution-plan", _VALID_TINY,
            "--target-tokens-per-sec", "50",
            "--fabric-gib-sec", "32",
            "--report-json", rpath,
        ])
        _assert(rc == 0, "valid_tiny_exits_zero",
                f"expected exit 0, got {rc}")
        if report:
            _assert(
                report.get("final_status") in ("pass", "warn"),
                "valid_tiny_final_status_pass_or_warn",
                f"got {report.get('final_status')!r}",
            )
    finally:
        os.unlink(rpath)


def test_report_contains_all_keys() -> None:
    """Report JSON contains all required top-level keys."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        rpath = tf.name
    try:
        rc, report, _ = _run([
            "--execution-plan", _VALID_TINY,
            "--report-json", rpath,
        ])
        if not _assert(report is not None, "report_file_written",
                       "report JSON not found or unreadable"):
            return
        missing = _REQUIRED_REPORT_KEYS - set(report.keys())
        _assert(
            len(missing) == 0,
            "report_contains_all_required_keys",
            f"missing keys: {sorted(missing)}",
        )
    finally:
        os.unlink(rpath)


def test_report_pipeline_version_is_132() -> None:
    """pipeline_version field equals 132."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        rpath = tf.name
    try:
        _, report, _ = _run([
            "--execution-plan", _VALID_TINY,
            "--report-json", rpath,
        ])
        if report is None:
            _fail("report_pipeline_version_132", "report not written")
            return
        _assert(
            report.get("pipeline_version") == 132,
            "report_pipeline_version_132",
            f"got {report.get('pipeline_version')!r}",
        )
    finally:
        os.unlink(rpath)


def test_report_stage_statuses_present() -> None:
    """All six per-stage status fields are non-null strings."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        rpath = tf.name
    try:
        _, report, _ = _run([
            "--execution-plan", _VALID_TINY,
            "--report-json", rpath,
        ])
        if report is None:
            _fail("report_stage_statuses_present", "report not written")
            return
        status_keys = [
            "execution_plan_validation_status",
            "command_plan_status",
            "mmio_replay_status",
            "fabric_route_status",
            "fabric_replay_status",
            "fabric_simulation_status",
        ]
        bad = [k for k in status_keys if not isinstance(report.get(k), str)]
        _assert(
            len(bad) == 0,
            "report_stage_statuses_present",
            f"missing/non-string: {bad}",
        )
    finally:
        os.unlink(rpath)


def test_report_counts_are_integers() -> None:
    """tile_count, command_count, route_count are positive integers."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        rpath = tf.name
    try:
        _, report, _ = _run([
            "--execution-plan", _VALID_TINY,
            "--report-json", rpath,
        ])
        if report is None:
            _fail("report_counts_are_integers", "report not written")
            return
        ok = (
            isinstance(report.get("tile_count"), int) and report["tile_count"] > 0
            and isinstance(report.get("command_count"), int) and report["command_count"] > 0
            and isinstance(report.get("route_count"), int) and report["route_count"] >= 0
        )
        _assert(ok, "report_counts_are_integers",
                f"tile={report.get('tile_count')} cmd={report.get('command_count')} "
                f"route={report.get('route_count')}")
    finally:
        os.unlink(rpath)


def test_malformed_plan_fails() -> None:
    """Malformed execution plan (missing header) causes exit != 0 and final_status=fail."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        rpath = tf.name
    try:
        rc, report, _ = _run([
            "--execution-plan", _MISSING_HEADER,
            "--report-json", rpath,
        ])
        _assert(rc != 0, "malformed_plan_exits_nonzero",
                f"expected nonzero exit, got {rc}")
        if report:
            _assert(
                report.get("final_status") == "fail",
                "malformed_plan_final_status_fail",
                f"got {report.get('final_status')!r}",
            )
    finally:
        if os.path.exists(rpath):
            os.unlink(rpath)


def test_invalid_tile_plan_fails() -> None:
    """Execution plan with invalid tile_id causes exit != 0."""
    if not os.path.exists(_INVALID_TILE):
        _fail("invalid_tile_plan_fails", "fixture not found: exec_plan_invalid_tile.json")
        return
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        rpath = tf.name
    try:
        rc, report, _ = _run([
            "--execution-plan", _INVALID_TILE,
            "--report-json", rpath,
        ])
        _assert(rc != 0, "invalid_tile_plan_exits_nonzero",
                f"expected nonzero exit, got {rc}")
    finally:
        if os.path.exists(rpath):
            os.unlink(rpath)


def test_strict_mode_exits_nonzero_on_warn() -> None:
    """--strict causes exit 1 when pipeline status is warn."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        rpath = tf.name
    try:
        rc_no_strict, report_no_strict, _ = _run([
            "--execution-plan", _VALID_TINY,
            "--report-json", rpath,
        ])
        if report_no_strict is None:
            _fail("strict_mode_exit_nonzero", "report not written in non-strict run")
            return
        final = report_no_strict.get("final_status", "pass")
        if final == "fail":
            _fail("strict_mode_exit_nonzero",
                  "non-strict run is already fail; test expects warn baseline")
            return
        if final == "pass":
            # Nothing to test - pipeline fully passes; skip with informational note
            print(
                f"  SKIP: M132: strict_mode_exit_nonzero "
                f"(pipeline final_status=pass; no warn to trigger strict failure)"
            )
            return
        # final == "warn"
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf2:
            rpath2 = tf2.name
        try:
            rc_strict, _, _ = _run([
                "--execution-plan", _VALID_TINY,
                "--strict",
                "--report-json", rpath2,
            ])
            _assert(rc_strict != 0, "strict_mode_exits_nonzero_on_warn",
                    f"expected nonzero exit with --strict (final=warn), got {rc_strict}")
        finally:
            if os.path.exists(rpath2):
                os.unlink(rpath2)
    finally:
        if os.path.exists(rpath):
            os.unlink(rpath)


def test_repeated_run_deterministic() -> None:
    """Two identical runs produce identical final_status, command_count, route_count."""
    reports = []
    for _ in range(2):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
            rpath = tf.name
        try:
            _, report, _ = _run([
                "--execution-plan", _VALID_TINY,
                "--report-json", rpath,
            ])
            if report:
                reports.append(report)
        finally:
            if os.path.exists(rpath):
                os.unlink(rpath)
    if len(reports) < 2:
        _fail("repeated_run_deterministic", "fewer than two valid reports produced")
        return
    r1, r2 = reports[0], reports[1]
    match = (
        r1.get("final_status") == r2.get("final_status")
        and r1.get("command_count") == r2.get("command_count")
        and r1.get("route_count") == r2.get("route_count")
    )
    _assert(match, "repeated_run_deterministic",
            f"status={r1.get('final_status')!r} vs {r2.get('final_status')!r}, "
            f"commands={r1.get('command_count')} vs {r2.get('command_count')}, "
            f"routes={r1.get('route_count')} vs {r2.get('route_count')}")


def test_no_cuda_import_in_pipeline() -> None:
    """Pipeline script contains no 'cuda' import or dependency."""
    pipeline_src = Path(_PIPELINE).read_text()
    has_cuda = "cuda" in pipeline_src.lower() and "import" in pipeline_src.lower()
    # Check more precisely: any line with both 'import' and 'cuda'
    cuda_lines = [
        ln for ln in pipeline_src.splitlines()
        if "cuda" in ln.lower() and ("import" in ln or "require" in ln)
    ]
    _assert(len(cuda_lines) == 0, "no_cuda_import_in_pipeline",
            f"cuda-related lines: {cuda_lines}")


def test_invalid_tiles_flag_exits_3() -> None:
    """--tiles 99 (out of range) exits with code 3."""
    rc, _, _ = _run(["--execution-plan", _VALID_TINY, "--tiles", "99"])
    _assert(rc == 3, "invalid_tiles_flag_exits_3",
            f"expected exit 3, got {rc}")


def test_mmio_stage_reports_warn_when_binary_absent() -> None:
    """When MMIO binary is absent the pipeline warns but continues (not fail)."""
    if not _BINARY.exists():
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
            rpath = tf.name
        try:
            rc, report, _ = _run([
                "--execution-plan", _VALID_TINY,
                "--report-json", rpath,
            ])
            _assert(rc == 0, "mmio_binary_absent_exits_zero",
                    f"expected exit 0 when binary absent, got {rc}")
            if report:
                _assert(
                    report.get("mmio_replay_status") == "warn",
                    "mmio_binary_absent_status_is_warn",
                    f"got {report.get('mmio_replay_status')!r}",
                )
        finally:
            if os.path.exists(rpath):
                os.unlink(rpath)
    else:
        print(
            "  SKIP: M132: mmio_binary_absent_* "
            "(build/att1-aimu-mmio-replay present; binary-absent path not exercised)"
        )


def test_exec_commands_seen_count() -> None:
    """exec_commands_seen equals the number of EXEC_* commands in the plan."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        rpath = tf.name
    try:
        _, report, _ = _run([
            "--execution-plan", _VALID_TINY,
            "--report-json", rpath,
        ])
        if report is None:
            _fail("exec_commands_seen_count", "report not written")
            return
        # exec_plan_valid_tiny.json has exactly 1 EXEC_MATMUL command
        _assert(
            report.get("exec_commands_seen") == 1,
            "exec_commands_seen_count",
            f"expected 1, got {report.get('exec_commands_seen')!r}",
        )
    finally:
        if os.path.exists(rpath):
            os.unlink(rpath)


def test_workdir_intermediate_files_written() -> None:
    """When --workdir is given, numbered intermediate JSONs are present after run."""
    with tempfile.TemporaryDirectory(prefix="att1_m132_test_") as wd:
        _run([
            "--execution-plan", _VALID_TINY,
            "--workdir", wd,
        ])
        present = sorted(Path(wd).glob("0*.json"))
        _assert(
            len(present) >= 4,
            "workdir_intermediate_files_written",
            f"expected >=4 intermediate JSONs, found {len(present)}: "
            f"{[p.name for p in present]}",
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

_TESTS = [
    test_valid_tiny_exits_zero,
    test_report_contains_all_keys,
    test_report_pipeline_version_is_132,
    test_report_stage_statuses_present,
    test_report_counts_are_integers,
    test_malformed_plan_fails,
    test_invalid_tile_plan_fails,
    test_strict_mode_exits_nonzero_on_warn,
    test_repeated_run_deterministic,
    test_no_cuda_import_in_pipeline,
    test_invalid_tiles_flag_exits_3,
    test_mmio_stage_reports_warn_when_binary_absent,
    test_exec_commands_seen_count,
    test_workdir_intermediate_files_written,
]


def main() -> int:
    print("M132 execution/replay pipeline tests")
    print("-" * 60)
    for test in _TESTS:
        try:
            test()
        except Exception as exc:  # noqa: BLE001
            _fail(test.__name__, f"exception: {exc}")
    print("-" * 60)
    print(f"M132 pipeline tests: {_g_pass} PASS  {_g_fail} FAIL")
    return 0 if _g_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
