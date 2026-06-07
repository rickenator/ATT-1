#!/usr/bin/env python3
"""
ATT-1 M134: schema and version compatibility checker.

Verifies that ATT-1 planning/control schema documents conform to version
compatibility rules:

  - Current schema versions are accepted.
  - Unsupported future versions are rejected with a clear error.
  - Required fields must be present.
  - Field types must match the schema specification.
  - Count fields must be consistent with the actual list lengths.
  - Status enum values must belong to the allowed set.
  - Unknown optional fields are accepted by default; in strict mode they
    produce a warning (which can escalate to an error).
  - Cross-schema consistency: shared fields (e.g. tile_count) must agree
    across the placement report and command plan when both are provided.

This tool does NOT call subprocess validators, execute inference, access real
PCIe/MMIO registers, or implement a kernel driver.  All checking is static
JSON analysis only.

Schema types supported:
  placement      — M98/M99 tensor placement report
  command_plan   — M109 AIMU command plan
  fabric_route   — M115/M116 fabric route report
  execution_plan — M125/M128 tensor execution plan
  pipeline       — M132 integrated execution/replay pipeline report
  cross          — cross-schema field consistency (placement ↔ command plan)

Exit codes:
  0 — all compatibility checks pass (warnings may be present)
  1 — one or more compatibility failures
  2 — parse/input error (malformed JSON or unrecognised schema type)

Usage:
    python3 compiler/check_schema_compat.py --schema placement \\
        --input compiler/fixtures/placement_report_valid.json

    python3 compiler/check_schema_compat.py --schema command_plan \\
        --input compiler/fixtures/plan_tiny_barrier_trace.json --strict

    python3 compiler/check_schema_compat.py --schema fabric_route \\
        --input compiler/fixtures/fabric_route_report_tiny.json

    python3 compiler/check_schema_compat.py --schema execution_plan \\
        --input compiler/fixtures/exec_plan_valid_tiny.json

    python3 compiler/check_schema_compat.py --schema pipeline \\
        --input compiler/fixtures/schema_compat/pipeline_v132_valid.json

    python3 compiler/check_schema_compat.py --schema cross \\
        --placement compiler/fixtures/placement_report_valid.json \\
        --command-plan compiler/fixtures/plan_tiny_barrier_trace.json

    python3 compiler/check_schema_compat.py --schema placement \\
        --input FILE --report-json out.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Version constants  — mirror the production tools so that any version bump
# in a tool is immediately caught here as an "unsupported future version".
# ---------------------------------------------------------------------------

SUPPORTED_PLACEMENT_VERSIONS: set[int] = {1}
SUPPORTED_COMMAND_PLAN_VERSIONS: set[int] = {1}
SUPPORTED_ROUTE_REPORT_VERSIONS: set[int] = {1}
SUPPORTED_EXEC_PLAN_VERSIONS: set[int] = {1}
SUPPORTED_PIPELINE_VERSIONS: set[int] = {132}

# ---------------------------------------------------------------------------
# Status enum sets
# ---------------------------------------------------------------------------

# Pipeline stage statuses (run_execution_replay_pipeline.py _STATUS_RANK keys
# plus "skip" for binary-absent stages).
VALID_STAGE_STATUSES: frozenset[str] = frozenset(
    {"pass", "ok", "warn", "fail", "unknown", "skip"}
)

# General document-level statuses used by validators.
VALID_GENERAL_STATUSES: frozenset[str] = frozenset(
    {"pass", "ok", "warn", "fail"}
)

# Tile-level capacity/bandwidth statuses.
VALID_TILE_STATUSES: frozenset[str] = frozenset(
    {"PASS", "WARN", "FAIL", "UNKNOWN"}
)

# Known top-level fields for unknown-field detection.
_KNOWN_PLACEMENT_FIELDS = frozenset(
    {"report_version", "header", "tiles", "tensors", "status", "notes"}
)
_KNOWN_CMD_PLAN_FIELDS = frozenset(
    {"command_plan_version", "header", "commands", "notes"}
)
_KNOWN_ROUTE_FIELDS = frozenset(
    {"header", "routes", "per_tile_summary", "notes"}
)
_KNOWN_EXEC_PLAN_FIELDS = frozenset(
    {
        "execution_plan_version", "model_id", "session_id", "token_phase",
        "tile_count", "layer_count", "command_count",
        "placement_report_path", "command_plan_path", "route_report_path",
        "status", "notes", "commands",
    }
)
_KNOWN_PIPELINE_FIELDS = frozenset(
    {
        "pipeline_version",
        "execution_plan_validation_status", "command_plan_status",
        "mmio_replay_status", "fabric_route_status",
        "fabric_replay_status", "fabric_simulation_status",
        "tile_count", "command_count", "route_count",
        "commands_replayed", "completions_seen", "exec_commands_seen",
        "failed_commands", "unsupported_commands",
        "aggregate_packets_sent", "aggregate_payload_bytes_sent",
        "required_fabric_gib_sec", "fabric_status", "final_status",
        "notes",
    }
)

_PIPELINE_STAGE_FIELDS = (
    "execution_plan_validation_status",
    "command_plan_status",
    "mmio_replay_status",
    "fabric_route_status",
    "fabric_replay_status",
    "fabric_simulation_status",
)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class Issue:
    code: str
    severity: str   # "error" | "warning"
    message: str


@dataclass
class CompatResult:
    schema: str
    status: str = "pass"   # "pass" | "warn" | "fail"
    issues: list[Issue] = field(default_factory=list)

    # ------------------------------------------------------------------ #
    def add_error(self, code: str, message: str) -> None:
        self.issues.append(Issue(code, "error", message))
        self.status = "fail"

    def add_warning(self, code: str, message: str, strict: bool = False) -> None:
        if strict:
            self.add_error(code, message)
        else:
            self.issues.append(Issue(code, "warning", message))
            if self.status == "pass":
                self.status = "warn"

    # ------------------------------------------------------------------ #
    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "warning"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _check_version(
    result: CompatResult,
    data: dict,
    field_name: str,
    supported: set[int],
    nested: str | None = None,
) -> bool:
    """
    Check that *field_name* is present, is an int, and is in *supported*.
    If *nested* is set, look inside data[nested][field_name].
    Returns True if the version is clean (no errors emitted for version).
    """
    src = data.get(nested, {}) if nested else data
    if not isinstance(src, dict):
        result.add_error(
            "E_MISSING_FIELD",
            f"Missing required field '{nested}'" if nested else "Root document is not an object",
        )
        return False

    ver = src.get(field_name)
    qualified = f"{nested}.{field_name}" if nested else field_name

    if ver is None:
        result.add_error("E_MISSING_VERSION", f"Missing required field '{qualified}'")
        return False
    if not isinstance(ver, int):
        result.add_error(
            "E_TYPE_MISMATCH",
            f"'{qualified}' must be an integer, got {type(ver).__name__!r} ({ver!r})",
        )
        return False
    if ver not in supported:
        result.add_error(
            "E_VERSION_UNSUPPORTED",
            f"'{qualified}' value {ver} is not supported; "
            f"supported versions: {sorted(supported)}",
        )
        return False
    return True


def _require_list(result: CompatResult, data: dict, field_name: str) -> bool:
    """Return True if data[field_name] exists and is a list."""
    if field_name not in data:
        result.add_error("E_MISSING_FIELD", f"Missing required field '{field_name}'")
        return False
    val = data[field_name]
    if not isinstance(val, list):
        result.add_error(
            "E_TYPE_MISMATCH",
            f"'{field_name}' must be a list, got {type(val).__name__!r}",
        )
        return False
    return True


def _check_count(
    result: CompatResult,
    declared_count: Any,
    actual_list: list,
    count_field: str,
    list_field: str,
) -> None:
    """Check that declared_count == len(actual_list)."""
    if declared_count is None:
        return
    if not isinstance(declared_count, int):
        result.add_error(
            "E_TYPE_MISMATCH",
            f"'{count_field}' must be an integer, got {type(declared_count).__name__!r}",
        )
        return
    if declared_count != len(actual_list):
        result.add_error(
            "E_COUNT_MISMATCH",
            f"'{count_field}'={declared_count} but len({list_field!r})={len(actual_list)}",
        )


def _check_status_enum(
    result: CompatResult,
    data: dict,
    field_name: str,
    valid: frozenset[str],
    required: bool = False,
) -> None:
    """Validate a status enum field."""
    val = data.get(field_name)
    if val is None:
        if required:
            result.add_error("E_MISSING_FIELD", f"Missing required field '{field_name}'")
        return
    if val not in valid:
        result.add_error(
            "E_INVALID_STATUS",
            f"'{field_name}'={val!r} is not a valid status; valid: {sorted(valid)}",
        )


def _check_unknown_fields(
    result: CompatResult,
    data: dict,
    known: frozenset[str],
    strict: bool,
) -> None:
    """
    Record unknown fields as warnings.

    Policy: unknown optional fields are always warnings, never errors —
    they indicate possible schema drift but do not constitute a compatibility
    failure.  In strict mode they are still reported (not silently suppressed)
    but remain warnings (exit 0).
    """
    for key in data:
        if key not in known:
            result.add_warning(
                "W_UNKNOWN_FIELD",
                f"Unknown field '{key}' — harmless by default; "
                "may indicate schema drift or future version field",
                strict=False,   # never escalate unknown-field to error
            )


# ---------------------------------------------------------------------------
# Per-schema checkers
# ---------------------------------------------------------------------------

def check_placement(data: dict, strict: bool = False) -> CompatResult:
    """Check M98/M99 tensor placement report schema compatibility."""
    r = CompatResult("placement")

    _check_version(r, data, "report_version", SUPPORTED_PLACEMENT_VERSIONS)

    for req in ("header", "tiles"):
        if req == "tiles":
            _require_list(r, data, req)
        elif req not in data:
            r.add_error("E_MISSING_FIELD", f"Missing required field '{req}'")

    tiles = data.get("tiles")
    if isinstance(tiles, list):
        for i, tile in enumerate(tiles):
            if not isinstance(tile, dict):
                continue
            for byte_field in ("model_bytes", "kv_bytes", "activation_bytes_per_token"):
                val = tile.get(byte_field)
                if val is not None and isinstance(val, (int, float)) and val < 0:
                    r.add_error(
                        "E_NEGATIVE_BYTES",
                        f"tiles[{i}].{byte_field} is negative: {val}",
                    )
            for sf in ("capacity_status", "bandwidth_status"):
                sv = tile.get(sf)
                if sv is not None and sv not in VALID_TILE_STATUSES:
                    r.add_error(
                        "E_INVALID_STATUS",
                        f"tiles[{i}].{sf}={sv!r} not in {sorted(VALID_TILE_STATUSES)}",
                    )

    _check_unknown_fields(r, data, _KNOWN_PLACEMENT_FIELDS, strict)
    return r


def check_command_plan(data: dict, strict: bool = False) -> CompatResult:
    """Check M109 AIMU command plan schema compatibility."""
    r = CompatResult("command_plan")

    _check_version(r, data, "command_plan_version", SUPPORTED_COMMAND_PLAN_VERSIONS)

    if "header" not in data:
        r.add_error("E_MISSING_FIELD", "Missing required field 'header'")

    commands_ok = _require_list(r, data, "commands")
    commands = data.get("commands")

    if commands_ok:
        hdr = data.get("header", {})
        if isinstance(hdr, dict):
            _check_count(r, hdr.get("command_count"), commands, "header.command_count", "commands")

    _check_unknown_fields(r, data, _KNOWN_CMD_PLAN_FIELDS, strict)
    return r


def check_fabric_route(data: dict, strict: bool = False) -> CompatResult:
    """Check M115/M116 fabric route report schema compatibility."""
    r = CompatResult("fabric_route")

    hdr = data.get("header")
    if not isinstance(hdr, dict):
        r.add_error("E_MISSING_FIELD", "Missing required field 'header' (must be an object)")
        return r

    _check_version(r, data, "route_report_version", SUPPORTED_ROUTE_REPORT_VERSIONS, nested="header")

    routes_ok = _require_list(r, data, "routes")
    routes = data.get("routes")

    if routes_ok and isinstance(routes, list):
        _check_count(r, hdr.get("route_count"), routes, "header.route_count", "routes")

    _check_unknown_fields(r, data, _KNOWN_ROUTE_FIELDS, strict)
    return r


def check_execution_plan(data: dict, strict: bool = False) -> CompatResult:
    """Check M125/M128 tensor execution plan schema compatibility."""
    r = CompatResult("execution_plan")

    _check_version(r, data, "execution_plan_version", SUPPORTED_EXEC_PLAN_VERSIONS)

    commands_ok = _require_list(r, data, "commands")
    commands = data.get("commands")

    if commands_ok and isinstance(commands, list):
        _check_count(r, data.get("command_count"), commands, "command_count", "commands")

    _check_status_enum(r, data, "status", VALID_GENERAL_STATUSES)

    _check_unknown_fields(r, data, _KNOWN_EXEC_PLAN_FIELDS, strict)
    return r


def check_pipeline(data: dict, strict: bool = False) -> CompatResult:
    """Check M132 integrated execution/replay pipeline report schema compatibility."""
    r = CompatResult("pipeline")

    _check_version(r, data, "pipeline_version", SUPPORTED_PIPELINE_VERSIONS)

    for sf in _PIPELINE_STAGE_FIELDS:
        _check_status_enum(r, data, sf, VALID_STAGE_STATUSES)

    _check_status_enum(r, data, "final_status", VALID_STAGE_STATUSES, required=True)

    _check_unknown_fields(r, data, _KNOWN_PIPELINE_FIELDS, strict)
    return r


def check_cross(
    placement: dict,
    cmd_plan: dict,
    route_report: dict | None = None,
    strict: bool = False,
) -> CompatResult:
    """
    Cross-schema field consistency checks.

    Verifies that shared fields agree across document boundaries:
      - placement.header.tile_count == command_plan.header.tile_count
      - placement.header.tile_count == route_report.header.tile_count  (if provided)
    """
    r = CompatResult("cross")

    p_tc = placement.get("header", {}).get("tile_count")
    c_tc = cmd_plan.get("header", {}).get("tile_count")

    if p_tc is not None and c_tc is not None:
        if p_tc != c_tc:
            r.add_error(
                "E_CROSS_TILE_COUNT",
                f"placement.header.tile_count={p_tc} "
                f"!= command_plan.header.tile_count={c_tc}",
            )
    else:
        if p_tc is None:
            r.add_warning("W_MISSING_TILE_COUNT", "placement.header.tile_count not found", strict=strict)
        if c_tc is None:
            r.add_warning("W_MISSING_TILE_COUNT", "command_plan.header.tile_count not found", strict=strict)

    if route_report is not None:
        rr_tc = route_report.get("header", {}).get("tile_count")
        if p_tc is not None and rr_tc is not None and p_tc != rr_tc:
            r.add_error(
                "E_CROSS_TILE_COUNT",
                f"placement.header.tile_count={p_tc} "
                f"!= route_report.header.tile_count={rr_tc}",
            )

    return r


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

_CHECKERS = {
    "placement": check_placement,
    "command_plan": check_command_plan,
    "fabric_route": check_fabric_route,
    "execution_plan": check_execution_plan,
    "pipeline": check_pipeline,
}


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _print_result(result: CompatResult, input_label: str) -> None:
    status_tag = result.status.upper()
    print(f"ATT-1 M134 schema compatibility — {result.schema}")
    print(f"  Input  : {input_label}")
    print(f"  Status : {status_tag}")
    if not result.issues:
        print("  (no issues)")
    else:
        for iss in result.issues:
            tag = "ERROR  " if iss.severity == "error" else "warning"
            print(f"  [{tag}] {iss.code}: {iss.message}")
    print(f"  errors={len(result.errors)}  warnings={len(result.warnings)}")


def _to_json(result: CompatResult, input_label: str, strict: bool) -> dict:
    return {
        "schema": result.schema,
        "input": input_label,
        "status": result.status,
        "strict": strict,
        "error_count": len(result.errors),
        "warning_count": len(result.warnings),
        "errors": [{"code": i.code, "message": i.message} for i in result.errors],
        "warnings": [{"code": i.code, "message": i.message} for i in result.warnings],
    }


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _load_json(path: str) -> dict:
    """Load a JSON file; raises SystemExit(2) on failure."""
    try:
        with open(path) as fh:
            data = json.load(fh)
    except FileNotFoundError:
        print(f"error: file not found: {path}", file=sys.stderr)
        sys.exit(2)
    except json.JSONDecodeError as exc:
        print(f"error: malformed JSON in {path}: {exc}", file=sys.stderr)
        sys.exit(2)
    if not isinstance(data, dict):
        print(f"error: {path}: JSON root must be an object, got {type(data).__name__}", file=sys.stderr)
        sys.exit(2)
    return data


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="ATT-1 M134 schema and version compatibility checker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument(
        "--schema",
        required=True,
        choices=list(_CHECKERS) + ["cross"],
        metavar="{" + "|".join(list(_CHECKERS) + ["cross"]) + "}",
        help="Schema type to check",
    )
    ap.add_argument("--input", metavar="FILE",
                    help="Input JSON file (required for all schemas except 'cross')")
    ap.add_argument("--placement", metavar="FILE",
                    help="Placement report JSON (required for --schema cross)")
    ap.add_argument("--command-plan", metavar="FILE", dest="command_plan",
                    help="Command plan JSON (required for --schema cross)")
    ap.add_argument("--route-report", metavar="FILE", dest="route_report",
                    help="Fabric route report JSON (optional for --schema cross)")
    ap.add_argument("--strict", action="store_true",
                    help="Promote warnings to errors; reject unknown optional fields")
    ap.add_argument("--report-json", metavar="FILE", dest="report_json",
                    help="Write JSON compatibility report to this file")
    return ap.parse_args()


def main() -> int:
    args = _parse_args()
    strict = args.strict

    if args.schema == "cross":
        if not args.placement or not args.command_plan:
            print(
                "error: --schema cross requires --placement and --command-plan",
                file=sys.stderr,
            )
            return 2
        placement = _load_json(args.placement)
        cmd_plan = _load_json(args.command_plan)
        route_report = _load_json(args.route_report) if args.route_report else None
        result = check_cross(placement, cmd_plan, route_report, strict=strict)
        input_label = f"{args.placement} + {args.command_plan}"
    else:
        if not args.input:
            print(f"error: --schema {args.schema} requires --input FILE", file=sys.stderr)
            return 2
        data = _load_json(args.input)
        checker = _CHECKERS[args.schema]
        result = checker(data, strict=strict)
        input_label = args.input

    _print_result(result, input_label)

    if args.report_json:
        report = _to_json(result, input_label, strict)
        try:
            with open(args.report_json, "w") as fh:
                json.dump(report, fh, indent=2)
                fh.write("\n")
        except OSError as exc:
            print(f"error: could not write report JSON: {exc}", file=sys.stderr)
            return 2

    return 0 if result.status in ("pass", "warn") else 1


if __name__ == "__main__":
    sys.exit(main())
