#!/usr/bin/env python3
"""
ATT-1 M136: local full-regression runner.

Executes ATT-1 validation layers in a stable order and reports one
consolidated pass/fail summary.  This is test orchestration only;
no new runtime behaviour is added.

Layers executed (CPU-only default):

  Step 1  make clean           — remove all build artefacts
  Step 2  make                 — compile all binaries (C11, no CUDA)
  Step 3  make test            — C unit/integration test suite
  Step 4  golden regressions   — M133 check_golden_regressions.py
  Step 5  schema compat        — M134 test_schema_compat.py
  Step 6  hostile inputs       — M135 test_hostile_inputs.py
  Step 7  pipeline smoke       — M132 run_execution_replay_pipeline.py
  Step 8  cache artifact check — git ls-files | grep …
  Step 9  docs lint/link check — M149 check_docs.py

With --cuda the first three steps become:
  Step 1  make clean
  Step 2  make CUDA=1
  Step 3  make test CUDA=1

Exit codes:
  0 — all steps pass
  1 — one or more steps failed
  2 — runner / tool error

Usage:
    # CPU-only (default)
    python3 compiler/run_full_regression.py

    # CUDA signoff (requires RTX 3090-class host)
    python3 compiler/run_full_regression.py --cuda

    # Write a JSON report
    python3 compiler/run_full_regression.py --report-json regression_report.json

    # Skip the clean + build steps (fast re-run after recent build)
    python3 compiler/run_full_regression.py --no-build

    # Makefile convenience target
    make regression
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Repo layout
# ---------------------------------------------------------------------------

_HERE = Path(__file__).resolve().parent   # compiler/
_REPO = _HERE.parent                      # project root

_COMPILER = _HERE
_GOLDEN = str(_COMPILER / "check_golden_regressions.py")
_SCHEMA_COMPAT = str(_COMPILER / "test_schema_compat.py")
_HOSTILE = str(_COMPILER / "test_hostile_inputs.py")
_PIPELINE = str(_COMPILER / "run_execution_replay_pipeline.py")
_CHECK_DOCS = str(_COMPILER / "check_docs.py")
_EXEC_PLAN = str(_COMPILER / "fixtures" / "exec_plan_valid_tiny.json")

_PY = sys.executable

# ---------------------------------------------------------------------------
# Step result
# ---------------------------------------------------------------------------

@dataclass
class StepResult:
    name: str
    command: str
    status: str = "pending"   # "pass" | "fail" | "skip" | "error"
    rc: int = -1
    elapsed: float = 0.0
    detail: str = ""


@dataclass
class RegressionReport:
    mode: str = "cpu"
    cuda_flag: bool = False
    steps: list[StepResult] = field(default_factory=list)
    overall: str = "pass"
    elapsed: float = 0.0

    def record_failure(self) -> None:
        self.overall = "fail"


# ---------------------------------------------------------------------------
# Runner helpers
# ---------------------------------------------------------------------------

def _run(
    step_name: str,
    cmd: list[str],
    *,
    cwd: str | Path = _REPO,
    env: dict | None = None,
) -> StepResult:
    """Run a subprocess and return a StepResult."""
    cmd_str = " ".join(cmd)
    result = StepResult(name=step_name, command=cmd_str)
    t0 = time.monotonic()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            env=env,
            capture_output=False,   # let output stream to terminal
        )
        result.rc = proc.returncode
        result.status = "pass" if proc.returncode == 0 else "fail"
    except FileNotFoundError as exc:
        result.rc = -1
        result.status = "error"
        result.detail = str(exc)
    except OSError as exc:
        result.rc = -1
        result.status = "error"
        result.detail = str(exc)
    result.elapsed = time.monotonic() - t0
    return result


def _run_quiet(
    step_name: str,
    cmd: list[str],
    *,
    cwd: str | Path = _REPO,
) -> StepResult:
    """Run a subprocess capturing output; only print on failure."""
    cmd_str = " ".join(cmd)
    result = StepResult(name=step_name, command=cmd_str)
    t0 = time.monotonic()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
        )
        result.rc = proc.returncode
        result.status = "pass" if proc.returncode == 0 else "fail"
        if proc.returncode != 0:
            result.detail = (proc.stdout + proc.stderr).strip()
    except FileNotFoundError as exc:
        result.rc = -1
        result.status = "error"
        result.detail = str(exc)
    result.elapsed = time.monotonic() - t0
    return result


# ---------------------------------------------------------------------------
# Individual step builders
# ---------------------------------------------------------------------------

def _step_make_clean() -> list[str]:
    return ["make", "clean"]


def _step_make(cuda: bool) -> list[str]:
    if cuda:
        return ["make", "CUDA=1"]
    return ["make"]


def _step_make_test(cuda: bool) -> list[str]:
    if cuda:
        return ["make", "test", "CUDA=1"]
    return ["make", "test"]


def _step_golden() -> list[str]:
    return [_PY, _GOLDEN]


def _step_schema_compat() -> list[str]:
    return [_PY, _SCHEMA_COMPAT]


def _step_hostile() -> list[str]:
    return [_PY, _HOSTILE]


def _step_pipeline_smoke() -> list[str]:
    """
    M132 pipeline smoke: validate + map + replay the tiny execution plan.
    Runs quietly and only reports the final pipeline status.
    """
    return [_PY, _PIPELINE, "--execution-plan", _EXEC_PLAN]


def _step_docs_check() -> list[str]:
    """M149 documentation lint and link checker."""
    return [_PY, _CHECK_DOCS]


def _check_cache_artifacts() -> StepResult:
    """
    Step 8: verify no __pycache__ / *.pyc / *.pyo files are git-tracked.
    """
    cmd_str = "git ls-files | grep -E '(__pycache__|\\.pyc$|\\.pyo$)'"
    result = StepResult(name="cache artifact check", command=cmd_str)
    t0 = time.monotonic()
    try:
        # Run git ls-files and filter client-side so we don't depend on
        # grep being installed with compatible flags.
        proc = subprocess.run(
            ["git", "ls-files"],
            cwd=str(_REPO),
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            result.status = "error"
            result.detail = proc.stderr.strip()
            result.rc = proc.returncode
        else:
            hits = [
                line for line in proc.stdout.splitlines()
                if "__pycache__" in line or line.endswith(".pyc") or line.endswith(".pyo")
            ]
            if hits:
                result.status = "fail"
                result.detail = "Tracked Python cache artifacts: " + ", ".join(hits)
                result.rc = 1
            else:
                result.status = "pass"
                result.rc = 0
    except OSError as exc:
        result.status = "error"
        result.detail = str(exc)
        result.rc = -1
    result.elapsed = time.monotonic() - t0
    return result


# ---------------------------------------------------------------------------
# Pipeline smoke — special handling because the binary may not exist
# ---------------------------------------------------------------------------

def _run_pipeline_smoke() -> StepResult:
    """
    Run the M132 pipeline with exec_plan_valid_tiny.json.
    If the MMIO replay binary is absent the pipeline exits with status 'warn'
    (exit 0); that is still considered a pass for regression purposes.
    """
    cmd = _step_pipeline_smoke()
    cmd_str = " ".join(cmd)
    result = StepResult(name="pipeline smoke (M132)", command=cmd_str)
    if not Path(_PIPELINE).exists():
        result.status = "skip"
        result.detail = "run_execution_replay_pipeline.py not found"
        result.rc = 0
        return result

    t0 = time.monotonic()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(_REPO),
            capture_output=True,
            text=True,
        )
        result.rc = proc.returncode
        # exit 0 = pass/warn (binary absent is warn), exit 1 = fail
        result.status = "pass" if proc.returncode in (0,) else "fail"
        if proc.returncode != 0:
            result.detail = (proc.stdout + proc.stderr).strip()
    except OSError as exc:
        result.rc = -1
        result.status = "error"
        result.detail = str(exc)
    result.elapsed = time.monotonic() - t0
    return result


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

_STATUS_ICON = {
    "pass":  "PASS",
    "fail":  "FAIL",
    "skip":  "SKIP",
    "error": "ERR ",
    "pending": "... ",
}

def _print_summary(report: RegressionReport) -> None:
    print()
    print("=" * 72)
    print("ATT-1 M136+M149 Full Regression Summary")
    print(f"  Mode   : {'CUDA' if report.cuda_flag else 'CPU-only'}")
    print(f"  Overall: {report.overall.upper()}")
    print(f"  Elapsed: {report.elapsed:.1f}s")
    print("=" * 72)
    name_w = max(len(s.name) for s in report.steps) + 2
    cmd_w  = min(50, max(len(s.command) for s in report.steps))
    print(f"  {'Step':<{name_w}}  {'Status'}  {'Time':>6}  Command")
    print(f"  {'-'*name_w}  {'------'}  {'------'}  {'-'*cmd_w}")
    for step in report.steps:
        icon = _STATUS_ICON.get(step.status, "????")
        cmd_display = step.command
        if len(cmd_display) > cmd_w:
            cmd_display = cmd_display[:cmd_w - 3] + "..."
        print(f"  {step.name:<{name_w}}  {icon}    {step.elapsed:>5.1f}s  {cmd_display}")
        if step.detail:
            for line in step.detail.splitlines()[:5]:
                print(f"  {'':>{name_w}}           | {line}")
    print("=" * 72)
    if report.overall == "pass":
        print("Result: ALL STEPS PASSED")
    else:
        failures = [s for s in report.steps if s.status in ("fail", "error")]
        print(f"Result: {len(failures)} STEP(S) FAILED")
        for s in failures:
            print(f"  - {s.name}")
    print()


# ---------------------------------------------------------------------------
# JSON report
# ---------------------------------------------------------------------------

def _write_json_report(report: RegressionReport, path: str) -> None:
    data = {
        "mode": report.mode,
        "cuda": report.cuda_flag,
        "overall": report.overall,
        "elapsed_seconds": round(report.elapsed, 3),
        "steps": [
            {
                "name": s.name,
                "command": s.command,
                "status": s.status,
                "exit_code": s.rc,
                "elapsed_seconds": round(s.elapsed, 3),
                "detail": s.detail or None,
            }
            for s in report.steps
        ],
    }
    try:
        with open(path, "w") as fh:
            json.dump(data, fh, indent=2)
            fh.write("\n")
        print(f"JSON report written to: {path}")
    except OSError as exc:
        print(f"warning: could not write JSON report: {exc}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="ATT-1 M136 local full-regression runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument(
        "--cuda",
        action="store_true",
        help=(
            "Run CUDA signoff (requires RTX 3090-class host). "
            "Replaces steps 1-3 with: make clean / make CUDA=1 / make test CUDA=1."
        ),
    )
    ap.add_argument(
        "--no-build",
        action="store_true",
        dest="no_build",
        help="Skip make clean / make / make test (fast re-run after a recent build).",
    )
    ap.add_argument(
        "--report-json",
        metavar="FILE",
        dest="report_json",
        help="Write a JSON regression report to this file.",
    )
    return ap.parse_args()


def main() -> int:
    args = _parse_args()
    cuda = args.cuda
    no_build = args.no_build

    report = RegressionReport(
        mode="cuda" if cuda else "cpu",
        cuda_flag=cuda,
    )

    t_start = time.monotonic()

    print()
    print("ATT-1 M136 Full Regression Runner")
    print(f"  Mode: {'CUDA signoff (RTX 3090-class required)' if cuda else 'CPU-only (default)'}")
    print(f"  Repo: {_REPO}")
    print()

    # ------------------------------------------------------------------ #
    # Steps 1-3: build
    # ------------------------------------------------------------------ #
    if no_build:
        print("[skip] Build steps skipped (--no-build)")
    else:
        for label, cmd in [
            ("make clean", _step_make_clean()),
            ("make" + (" CUDA=1" if cuda else ""), _step_make(cuda)),
            ("make test" + (" CUDA=1" if cuda else ""), _step_make_test(cuda)),
        ]:
            print(f"[run ] {label}")
            step = _run(label, cmd)
            report.steps.append(step)
            if step.status != "pass":
                report.record_failure()
                # Build failures are fatal — subsequent steps won't be meaningful
                print(f"       FAILED (rc={step.rc}) — aborting regression run")
                report.elapsed = time.monotonic() - t_start
                _print_summary(report)
                if args.report_json:
                    _write_json_report(report, args.report_json)
                return 1

    # ------------------------------------------------------------------ #
    # Step 4: golden regressions (M133)
    # ------------------------------------------------------------------ #
    print("[run ] golden regressions (M133)")
    step = _run_quiet("golden regressions (M133)", _step_golden())
    report.steps.append(step)
    if step.status != "pass":
        report.record_failure()

    # ------------------------------------------------------------------ #
    # Step 5: schema compatibility (M134)
    # ------------------------------------------------------------------ #
    print("[run ] schema compatibility (M134)")
    step = _run_quiet("schema compatibility (M134)", _step_schema_compat())
    report.steps.append(step)
    if step.status != "pass":
        report.record_failure()

    # ------------------------------------------------------------------ #
    # Step 6: hostile-input regression (M135)
    # ------------------------------------------------------------------ #
    print("[run ] hostile-input regression (M135)")
    step = _run_quiet("hostile-input regression (M135)", _step_hostile())
    report.steps.append(step)
    if step.status != "pass":
        report.record_failure()

    # ------------------------------------------------------------------ #
    # Step 7: pipeline smoke (M132)
    # ------------------------------------------------------------------ #
    print("[run ] pipeline smoke (M132)")
    step = _run_pipeline_smoke()
    report.steps.append(step)
    if step.status not in ("pass", "skip"):
        report.record_failure()

    # ------------------------------------------------------------------ #
    # Step 8: cache artifact check
    # ------------------------------------------------------------------ #
    print("[run ] cache artifact check")
    step = _check_cache_artifacts()
    report.steps.append(step)
    if step.status != "pass":
        report.record_failure()

    # ------------------------------------------------------------------ #
    # Step 9: documentation lint and link checker (M149)
    # ------------------------------------------------------------------ #
    print("[run ] documentation lint and link checker (M149)")
    step = _run_quiet("docs lint/link check (M149)", _step_docs_check())
    report.steps.append(step)
    if step.status != "pass":
        report.record_failure()

    # ------------------------------------------------------------------ #
    # Wrap up
    # ------------------------------------------------------------------ #
    report.elapsed = time.monotonic() - t_start
    _print_summary(report)

    if args.report_json:
        _write_json_report(report, args.report_json)

    return 0 if report.overall == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
