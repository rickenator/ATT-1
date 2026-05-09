#!/usr/bin/env python3
"""
ATT-1 M143: deterministic fuzz/smoke harness for JSON schema validators.

Drives compiler/check_hostile_inputs.py against:

  1. All 27 existing hostile fixtures (compiler/fixtures/hostile/) — each must
     be rejected (non-zero exit).
  2. Valid fixture baselines — each must pass (exit 0).
  3. Inline-generated mutation seeds — each must be rejected (non-zero exit).

This tool does NOT execute inference, access real PCIe/MMIO registers, or
run randomised fuzz loops.  All mutations are deterministic and complete in
well under 10 seconds.

Exit codes:
  0 — all cases behaved as expected
  1 — one or more cases produced an unexpected result

Usage:
    python3 compiler/fuzz_json_schemas.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT   = os.path.dirname(SCRIPT_DIR)
CHECK_HOSTILE = os.path.join(SCRIPT_DIR, "check_hostile_inputs.py")
FIXTURE_DIR = os.path.join(SCRIPT_DIR, "fixtures")
HOSTILE_DIR = os.path.join(FIXTURE_DIR, "hostile")

# ---------------------------------------------------------------------------
# Schema-type inference from filename prefix
# ---------------------------------------------------------------------------

_PREFIX_TO_SCHEMA: dict[str, str] = {
    "placement_": "placement",
    "cmd_plan_":  "command_plan",
    "route_":     "fabric_route",
    "exec_":      "execution_plan",
    "pipeline_":  "pipeline",
}


def _schema_for_fixture(name: str) -> str | None:
    for prefix, schema in _PREFIX_TO_SCHEMA.items():
        if name.startswith(prefix):
            return schema
    return None


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------

def _run_check(schema: str, input_path: str) -> int:
    """Return exit code of check_hostile_inputs.py."""
    result = subprocess.run(
        [sys.executable, CHECK_HOSTILE,
         "--schema", schema, "--input", input_path],
        capture_output=True,
        text=True,
    )
    return result.returncode


def _run_case_file(label: str, schema: str, path: str,
                   expect_pass: bool) -> bool:
    rc = _run_check(schema, path)
    ok = (rc == 0) if expect_pass else (rc != 0)
    if ok:
        verdict = "PASS"
        print(f"PASS: fuzz_json: {label}")
    else:
        verdict = "FAIL"
        want = "exit 0" if expect_pass else "non-zero exit"
        print(
            f"FAIL: fuzz_json: {label} — expected {want}, got exit {rc}",
            file=sys.stderr,
        )
    return ok


def _run_case_json(label: str, schema: str, doc: object,
                   expect_pass: bool) -> bool:
    """Serialise *doc* to a temp file and run the check."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", prefix="att1_fuzz_", delete=False
    ) as f:
        json.dump(doc, f)
        tmp = f.name
    try:
        return _run_case_file(label, schema, tmp, expect_pass)
    finally:
        os.unlink(tmp)


def _run_case_raw(label: str, schema: str, text: str,
                  expect_pass: bool) -> bool:
    """Write *text* verbatim to a temp file and run the check."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", prefix="att1_fuzz_", delete=False
    ) as f:
        f.write(text)
        tmp = f.name
    try:
        return _run_case_file(label, schema, tmp, expect_pass)
    finally:
        os.unlink(tmp)


# ---------------------------------------------------------------------------
# Inline mutation seeds
# ---------------------------------------------------------------------------

# Each entry: (label, schema, doc_or_text, expect_pass)
# doc_or_text is either a Python object (serialised via json.dump) or a str
# written verbatim (for intentionally malformed JSON text).
_MUTATIONS: list[tuple[str, str, object, bool]] = [
    # Truncated / malformed JSON text (parse error → non-zero exit)
    ("mut_truncated_json",
     "placement",
     '{"report_version": 1,',
     False),

    # JSON null at root (not a dict → parse error)
    ("mut_null_document",
     "placement",
     "null",
     False),

    # Integer at root (not a dict → parse error)
    ("mut_integer_root",
     "placement",
     "42",
     False),

    # Empty object — missing all required fields
    ("mut_empty_object",
     "placement",
     {},
     False),

    # tile_count in header mismatches actual tiles list length
    ("mut_tile_count_mismatch",
     "placement",
     {"report_version": 1,
      "header": {"schema_version": 1, "tile_count": 2, "model_name": "x"},
      "tiles": [{"tile_id": 0}]},
     False),

    # Negative tile_count
    ("mut_negative_tile_count",
     "placement",
     {"report_version": 1,
      "header": {"schema_version": 1, "tile_count": -1, "model_name": "x"},
      "tiles": []},
     False),

    # Far-future schema version
    ("mut_future_schema_version",
     "placement",
     {"report_version": 9999,
      "header": {"schema_version": 9999, "tile_count": 0, "model_name": "x"},
      "tiles": []},
     False),

    # Duplicate tile_id values in tiles list
    ("mut_placement_duplicate_tile_id",
     "placement",
     {"report_version": 1,
      "header": {"schema_version": 1, "tile_count": 2, "model_name": "x"},
      "tiles": [{"tile_id": 0}, {"tile_id": 0}]},
     False),

    # execution_plan: missing required header fields
    ("mut_exec_missing_header",
     "execution_plan",
     {"plan_version": 1, "commands": []},
     False),

    # command_plan: negative byte_count
    ("mut_cmd_negative_byte_count",
     "command_plan",
     {"command_plan_version": 1,
      "header": {
          "command_plan_version": 1,
          "model_name": "x",
          "tile_count": 1,
          "command_count": 1,
      },
      "commands": [{
          "command_id": 0,
          "command_type": "LOAD",
          "tile_id": 0,
          "tensor_name": "weight",
          "byte_count": -100,
      }]},
     False),

    # fabric_route: missing reduction_behavior on reduction route
    ("mut_route_missing_reduction_behavior",
     "fabric_route",
     {"route_version": 1,
      "header": {"schema_version": 1, "route_count": 1},
      "routes": [{
          "route_id": 0,
          "route_type": "REDUCTION",
          "source_tile": 0,
          "dest_tile": 1,
          "payload_bytes": 64,
          "ordering_policy": "STRICT",
      }]},
     False),
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    total    = 0
    failures: list[str] = []

    # 1. All hostile fixtures must be rejected (non-zero exit).
    if not os.path.isdir(HOSTILE_DIR):
        print(f"ERROR: hostile fixture dir not found: {HOSTILE_DIR}",
              file=sys.stderr)
        return 1

    for fname in sorted(os.listdir(HOSTILE_DIR)):
        if not fname.endswith(".json"):
            continue
        schema = _schema_for_fixture(fname)
        if schema is None:
            print(f"SKIP: fuzz_json: {fname} (unknown prefix)", file=sys.stderr)
            continue
        label = f"hostile/{fname}"
        path  = os.path.join(HOSTILE_DIR, fname)
        total += 1
        if not _run_case_file(label, schema, path, expect_pass=False):
            failures.append(label)

    # 2. Valid baselines must pass (exit 0).
    baselines = [
        ("placement_report_valid.json", "placement"),
        ("exec_plan_valid_tiny.json",   "execution_plan"),
    ]
    for fname, schema in baselines:
        path = os.path.join(FIXTURE_DIR, fname)
        if not os.path.exists(path):
            print(f"SKIP: fuzz_json: baseline {fname} not found",
                  file=sys.stderr)
            continue
        label = f"baseline/{fname}"
        total += 1
        if not _run_case_file(label, schema, path, expect_pass=True):
            failures.append(label)

    # 3. Inline-generated mutation seeds (must be rejected).
    for entry in _MUTATIONS:
        label, schema, doc_or_text, expect_pass = entry
        total += 1
        if isinstance(doc_or_text, str):
            ok = _run_case_raw(label, schema, doc_or_text, expect_pass)
        else:
            ok = _run_case_json(label, schema, doc_or_text, expect_pass)
        if not ok:
            failures.append(label)

    # Summary
    print()
    n_pass = total - len(failures)
    if failures:
        print(f"fuzz_json: {n_pass}/{total} PASS  {len(failures)} FAIL",
              file=sys.stderr)
        for label in failures:
            print(f"  FAIL: {label}", file=sys.stderr)
        return 1

    print(f"fuzz_json: {total}/{total} PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
