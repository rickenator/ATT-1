#!/usr/bin/env python3
"""Smoke tests for compiler/check_aimu_conformance.py (M161)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
_CHECKER = _HERE / "check_aimu_conformance.py"
_PY = sys.executable

_PASS = 0
_FAIL = 0


def expect(name: str, condition: bool) -> None:
    global _PASS, _FAIL
    if condition:
        _PASS += 1
        print(f"PASS: {name}")
    else:
        _FAIL += 1
        print(f"FAIL: {name}")


def run(extra: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run([_PY, str(_CHECKER)] + extra,
                          cwd=str(_REPO),
                          capture_output=True,
                          text=True)


def main() -> int:
    report_path = _REPO / "build" / "aimu_conformance_report.json"
    if report_path.exists():
        report_path.unlink()

    p = run([])
    expect("checker exits 0 on current repo", p.returncode == 0)
    expect("checker prints PASS", "Result: PASS" in p.stdout)

    p = run(["--report-json", str(report_path)])
    expect("checker --report-json exits 0", p.returncode == 0)
    expect("report-json file created", report_path.exists())

    data = {}
    if report_path.exists():
        data = json.loads(report_path.read_text(encoding="utf-8"))
    expect("report overall == pass", data.get("overall") == "pass")
    expect("report tool key present", data.get("tool") == "check_aimu_conformance.py")
    expect("report checks_run >= 4", int(data.get("checks_run", 0)) >= 4)

    checker_text = _CHECKER.read_text(encoding="utf-8")
    expect("checker has no CUDA import", "import cuda" not in checker_text.lower())

    if report_path.exists():
        report_path.unlink()

    print(f"Summary: {_PASS} passed, {_FAIL} failed")
    return 0 if _FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
