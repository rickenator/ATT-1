#!/usr/bin/env python3
"""
ATT-1 M135: hostile-input checker for planning/control-plane documents.

Extends M134 schema/version compatibility checks (check_schema_compat.py)
with deeper hostile-input validation:

  - Negative counts and impossible dimensions (negative tile_count, etc.)
  - Duplicate identifiers (tile_id, command_id, route_id, plan_command_id)
  - Invalid enum values (command_type, route_type, execution_phase,
    ordering_policy, expected_status, dtype)
  - Missing required references (tensor_name for tensor commands, output
    buffers for LOAD ops, tensor_dependencies for EXEC_MATMUL)
  - Negative byte counts in command plans
  - Q4 tensor missing group_size
  - Non-existent owner tile references
  - Zero payload on data routes
  - Reduction route missing explicit reduction_behavior
  - Pipeline count inconsistency (commands_replayed > command_count)

This tool checks a SINGLE input document (same CLI as check_schema_compat.py).
Use test_hostile_inputs.py to run the full regression suite.

Hostile-input policy (documented in docs/schema_compatibility.md §11):
  - Malformed input must fail clearly with a specific error code.
  - Future/unknown versions must fail clearly.
  - Negative counts and impossible dimensions must fail.
  - Missing required fields must fail.
  - No silent fallback; every hostile input must produce exit code 1.

This tool does NOT call subprocess validators, execute inference, access real
PCIe/MMIO registers, or implement a kernel driver.  All checking is static
JSON analysis only.

Exit codes:
  0 — all hostile-input checks pass (warnings may be present)
  1 — one or more hostile-input failures detected
  2 — parse/input error (malformed JSON or unrecognised schema type)

Usage:
    python3 compiler/check_hostile_inputs.py --schema placement \\
        --input compiler/fixtures/hostile/placement_missing_header.json

    python3 compiler/check_hostile_inputs.py --schema command_plan \\
        --input compiler/fixtures/hostile/cmd_plan_duplicate_command_id.json

    python3 compiler/check_hostile_inputs.py --schema fabric_route \\
        --input compiler/fixtures/hostile/route_invalid_route_type.json

    python3 compiler/check_hostile_inputs.py --schema execution_plan \\
        --input compiler/fixtures/hostile/exec_invalid_phase.json

    python3 compiler/check_hostile_inputs.py --schema pipeline \\
        --input compiler/fixtures/hostile/pipeline_negative_tile_count.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Import base types from M134 checker (same directory).
# Using sys.path so the import works from any working directory.
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))
from check_schema_compat import (   # noqa: E402
    CompatResult,
    Issue,
    _load_json,
    VALID_TILE_STATUSES,
    VALID_GENERAL_STATUSES,
    VALID_STAGE_STATUSES,
    check_placement as _base_check_placement,
    check_command_plan as _base_check_command_plan,
    check_fabric_route as _base_check_fabric_route,
    check_execution_plan as _base_check_execution_plan,
    check_pipeline as _base_check_pipeline,
)

# ---------------------------------------------------------------------------
# Extended allowed-value sets (mirrors production validators)
# ---------------------------------------------------------------------------

# Allowed M109 command types (from validate_tensor_execution_plan.py
# VALID_COMMAND_TYPES + M109 extras seen in fixtures).
_ALLOWED_COMMAND_TYPES: frozenset[str] = frozenset({
    "LOAD_TENSOR_TILE",
    "VALIDATE_TENSOR",
    "EXEC_MATMUL",
    "EXEC_RMSNORM",
    "EXEC_ROPE",
    "EXEC_ATTENTION",
    "EXEC_SWIGLU",
    "EXEC_SOFTMAX",
    "EXEC_RESIDUAL",
    "KV_APPEND",
    "KV_READ",
    "FABRIC_SEND",
    "FABRIC_REDUCE",
    "TILE_BARRIER",
    "TRACE_SNAPSHOT",
    "QUERY_COUNTERS",
    "RESET_TILE",
    "NOP",
})

# Tensor command types that require a non-null tensor_name in command plans.
_TENSOR_CMD_TYPES: frozenset[str] = frozenset({
    "LOAD_TENSOR_TILE",
    "VALIDATE_TENSOR",
    "EXEC_MATMUL",
    "EXEC_RMSNORM",
    "EXEC_ROPE",
    "EXEC_ATTENTION",
    "EXEC_SWIGLU",
    "EXEC_SOFTMAX",
    "EXEC_RESIDUAL",
})

# Allowed dtypes (from validate_tensor_execution_plan.py VALID_DTYPES +
# bf16 seen in fixtures).
_ALLOWED_DTYPES: frozenset[str] = frozenset({"f32", "q8", "q4", "bf16", "i32"})

# Allowed fabric route types (from validate_fabric_routes.py VALID_ROUTE_TYPES).
_ALLOWED_ROUTE_TYPES: frozenset[str] = frozenset({
    "ACTIVATION_SEND",
    "ACTIVATION_BROADCAST",
    "PARTIAL_REDUCE",
    "LOGITS_REDUCE",
    "KV_TRANSFER",
    "TILE_BARRIER",
    "TRACE_EVENT",
    "CONTROL_ACK",
})

# Data routes that must carry a nonzero payload
# (from validate_fabric_routes.py DATA_ROUTE_TYPES).
_DATA_ROUTE_TYPES: frozenset[str] = frozenset({
    "ACTIVATION_SEND",
    "ACTIVATION_BROADCAST",
    "PARTIAL_REDUCE",
    "LOGITS_REDUCE",
    "KV_TRANSFER",
})

# Reduction routes that require an explicit (non-trivial) reduction_behavior
# (from validate_fabric_routes.py REDUCTION_ROUTE_TYPES).
_REDUCTION_ROUTE_TYPES: frozenset[str] = frozenset({
    "PARTIAL_REDUCE",
    "LOGITS_REDUCE",
})

# Explicit (non-trivial) reduction behaviors for reduction routes.
_EXPLICIT_REDUCTION_BEHAVIORS: frozenset[str] = frozenset({
    "sum", "concat", "max", "topk",
})

# Allowed fabric ordering policies (from validate_fabric_routes.py).
_ALLOWED_ORDERING_POLICIES: frozenset[str] = frozenset({
    "ordered", "unordered", "barriered",
})

# Allowed execution phases (from validate_tensor_execution_plan.py
# VALID_EXECUTION_PHASES).
_ALLOWED_EXEC_PHASES: frozenset[str] = frozenset({
    "DEVICE_PROBE",
    "TILE_ENUMERATION",
    "MEMORY_ALLOCATE",
    "LOAD_TENSOR_TILE",
    "VALIDATE_TENSOR_TILE",
    "PREFILL_SETUP",
    "PREFILL_EXECUTION_PLAN",
    "DECODE_STEP_PLAN",
    "KV_APPEND",
    "KV_READ",
    "FABRIC_SEND",
    "FABRIC_REDUCE",
    "TRACE_SNAPSHOT",
    "QUERY_COUNTERS",
    "CLEANUP",
    "RESET",
})

# Execution plan command types that require at least one output_buffer.
_OUTPUT_BUF_REQUIRED: frozenset[str] = frozenset({
    "EXEC_MATMUL", "EXEC_RMSNORM", "EXEC_ROPE", "EXEC_ATTENTION",
    "EXEC_SWIGLU", "EXEC_SOFTMAX", "EXEC_RESIDUAL",
    "LOAD_TENSOR_TILE", "KV_APPEND", "KV_READ", "FABRIC_REDUCE",
})

# Execution plan command types that require at least one tensor_dependency.
_TENSOR_DEP_REQUIRED: frozenset[str] = frozenset({
    "LOAD_TENSOR_TILE", "VALIDATE_TENSOR",
    "EXEC_MATMUL", "EXEC_RMSNORM",
})

# Allowed expected_status values in execution plans
# (from validate_tensor_execution_plan.py VALID_EXPECTED_STATUSES).
_ALLOWED_EXPECTED_STATUSES: frozenset[str] = frozenset({
    "ok", "error", "warn",
    "ATT1_AIMU_ERR_OK",
    "ATT1_AIMU_ERR_UNSUPPORTED_OP",
    "ATT1_AIMU_ERR_INVALID_ARG",
    "ATT1_AIMU_ERR_TIMEOUT",
    "ATT1_AIMU_ERR_BUS_FAULT",
    "ATT1_AIMU_ERR_NO_MEM",
})

# ---------------------------------------------------------------------------
# Extended schema checkers
# ---------------------------------------------------------------------------

def check_placement(data: dict, strict: bool = False) -> CompatResult:
    """
    Extended placement report hostile-input checker.

    Runs all M134 checks first, then adds:
      - E_NEGATIVE_TILE_COUNT: header.tile_count < 0
      - E_TILE_COUNT_MISMATCH: header.tile_count != len(tiles)
      - E_INVALID_TILE_ID: non-integer tile_id in tile records
      - E_DUPLICATE_ID: duplicate tile_id values
      - E_Q4_MISSING_GROUP_SIZE: q4 tensor with null group_size
      - E_NONEXISTENT_OWNER: owner_tile not in the tile_id set
    """
    r = _base_check_placement(data, strict)

    header = data.get("header")
    tiles = data.get("tiles")

    if isinstance(header, dict):
        tc = header.get("tile_count")
        if isinstance(tc, int) and tc < 0:
            r.add_error(
                "E_NEGATIVE_TILE_COUNT",
                f"header.tile_count={tc} is negative — impossible tile count",
            )
        if (
            isinstance(tc, int) and tc >= 0
            and isinstance(tiles, list)
            and tc != len(tiles)
        ):
            r.add_error(
                "E_TILE_COUNT_MISMATCH",
                f"header.tile_count={tc} but len(tiles)={len(tiles)} — count mismatch",
            )

    valid_tile_ids: set[int] = set()
    if isinstance(tiles, list):
        seen: dict[int, int] = {}   # tile_id → first index
        for i, tile in enumerate(tiles):
            if not isinstance(tile, dict):
                continue
            tid = tile.get("tile_id")
            if tid is None:
                continue
            if not isinstance(tid, int):
                r.add_error(
                    "E_INVALID_TILE_ID",
                    f"tiles[{i}].tile_id must be an integer, "
                    f"got {type(tid).__name__!r} ({tid!r})",
                )
            else:
                if tid in seen:
                    r.add_error(
                        "E_DUPLICATE_ID",
                        f"tiles[{i}].tile_id={tid} duplicates "
                        f"tiles[{seen[tid]}].tile_id — duplicate tile identifier",
                    )
                else:
                    seen[tid] = i
                    valid_tile_ids.add(tid)

    tensors = data.get("tensors")
    if isinstance(tensors, list):
        for i, tensor in enumerate(tensors):
            if not isinstance(tensor, dict):
                continue
            quant = tensor.get("quantization")
            if quant == "q4":
                gs = tensor.get("quantization_group_size")
                if gs is None:
                    r.add_error(
                        "E_Q4_MISSING_GROUP_SIZE",
                        f"tensors[{i}] has quantization='q4' but "
                        f"quantization_group_size is null — required for q4",
                    )
            owner = tensor.get("owner_tile")
            if isinstance(owner, int) and valid_tile_ids and owner not in valid_tile_ids:
                r.add_error(
                    "E_NONEXISTENT_OWNER",
                    f"tensors[{i}].owner_tile={owner} references a tile_id "
                    f"not present in tiles — non-existent owner",
                )

    return r


def check_command_plan(data: dict, strict: bool = False) -> CompatResult:
    """
    Extended command plan hostile-input checker.

    Runs all M134 checks first, then adds:
      - E_DUPLICATE_ID: duplicate command_id values
      - E_UNKNOWN_TYPE: command_type not in allowed set
      - E_INVALID_TILE_ID: tile_id < 0
      - E_NEGATIVE_BYTE_COUNT: packed_bytes or total_bytes < 0
      - E_MISSING_TENSOR_NAME: tensor command type with null tensor_name
      - E_UNSUPPORTED_DTYPE: dtype not in allowed set (when not null)
    """
    r = _base_check_command_plan(data, strict)

    commands = data.get("commands")
    if not isinstance(commands, list):
        return r

    seen_ids: dict[Any, int] = {}
    for i, cmd in enumerate(commands):
        if not isinstance(cmd, dict):
            continue
        ctx = f"commands[{i}]"

        # Duplicate command_id
        cid = cmd.get("command_id")
        if cid is not None:
            if cid in seen_ids:
                r.add_error(
                    "E_DUPLICATE_ID",
                    f"{ctx}.command_id={cid} duplicates "
                    f"commands[{seen_ids[cid]}].command_id — duplicate identifier",
                )
            else:
                seen_ids[cid] = i

        # Unknown command_type
        ctype = cmd.get("command_type")
        if ctype is not None and ctype not in _ALLOWED_COMMAND_TYPES:
            r.add_error(
                "E_UNKNOWN_TYPE",
                f"{ctx}.command_type={ctype!r} is not a recognised "
                f"ATT-1 command type",
            )

        # Negative tile_id
        tid = cmd.get("tile_id")
        if isinstance(tid, int) and tid < 0:
            r.add_error(
                "E_INVALID_TILE_ID",
                f"{ctx}.tile_id={tid} is negative — invalid tile identifier",
            )

        # Negative byte counts
        for bf in ("packed_bytes", "total_bytes", "scale_bytes"):
            bval = cmd.get(bf)
            if isinstance(bval, (int, float)) and bval < 0:
                r.add_error(
                    "E_NEGATIVE_BYTE_COUNT",
                    f"{ctx}.{bf}={bval} is negative — impossible byte count",
                )

        # Tensor commands must have a non-null tensor_name
        if ctype in _TENSOR_CMD_TYPES:
            tname = cmd.get("tensor_name")
            if tname is None:
                r.add_error(
                    "E_MISSING_TENSOR_NAME",
                    f"{ctx} command_type={ctype!r} requires tensor_name "
                    f"but tensor_name is null",
                )

        # Unsupported dtype
        dtype = cmd.get("dtype")
        if dtype is not None and dtype not in _ALLOWED_DTYPES:
            r.add_error(
                "E_UNSUPPORTED_DTYPE",
                f"{ctx}.dtype={dtype!r} is not a supported ATT-1 dtype; "
                f"supported: {sorted(_ALLOWED_DTYPES)}",
            )

    return r


def check_fabric_route(data: dict, strict: bool = False) -> CompatResult:
    """
    Extended fabric route report hostile-input checker.

    Runs all M134 checks first, then adds:
      - E_DUPLICATE_ID: duplicate route_id values
      - E_UNKNOWN_TYPE: route_type not in allowed set
      - E_INVALID_TILE_ID: source_tile < 0
      - E_ZERO_PAYLOAD: data route with payload_bytes == 0
      - E_MISSING_REDUCTION: reduction route without explicit reduction_behavior
      - E_INVALID_ORDERING: ordering_policy not in allowed set
    """
    r = _base_check_fabric_route(data, strict)

    routes = data.get("routes")
    if not isinstance(routes, list):
        return r

    seen_ids: dict[Any, int] = {}
    for i, route in enumerate(routes):
        if not isinstance(route, dict):
            continue
        ctx = f"routes[{i}]"

        # Duplicate route_id
        rid = route.get("route_id")
        if rid is not None:
            if rid in seen_ids:
                r.add_error(
                    "E_DUPLICATE_ID",
                    f"{ctx}.route_id={rid} duplicates "
                    f"routes[{seen_ids[rid]}].route_id — duplicate identifier",
                )
            else:
                seen_ids[rid] = i

        rtype = route.get("route_type")

        # Unknown route_type
        if rtype is not None and rtype not in _ALLOWED_ROUTE_TYPES:
            r.add_error(
                "E_UNKNOWN_TYPE",
                f"{ctx}.route_type={rtype!r} is not a recognised "
                f"ATT-1 fabric route type",
            )

        # Negative source_tile
        src = route.get("source_tile")
        if isinstance(src, int) and src < 0:
            r.add_error(
                "E_INVALID_TILE_ID",
                f"{ctx}.source_tile={src} is negative — invalid tile identifier",
            )

        # Data route with zero payload
        pb = route.get("payload_bytes")
        if rtype in _DATA_ROUTE_TYPES and isinstance(pb, (int, float)) and pb == 0:
            r.add_error(
                "E_ZERO_PAYLOAD",
                f"{ctx} route_type={rtype!r} is a data route but "
                f"payload_bytes=0 — data routes must carry nonzero payload",
            )

        # Reduction route without explicit reduction_behavior
        rb = route.get("reduction_behavior")
        if rtype in _REDUCTION_ROUTE_TYPES:
            if rb not in _EXPLICIT_REDUCTION_BEHAVIORS:
                r.add_error(
                    "E_MISSING_REDUCTION",
                    f"{ctx} route_type={rtype!r} requires an explicit "
                    f"reduction_behavior (one of {sorted(_EXPLICIT_REDUCTION_BEHAVIORS)}), "
                    f"got {rb!r}",
                )

        # Invalid ordering_policy
        op = route.get("ordering_policy")
        if op is not None and op not in _ALLOWED_ORDERING_POLICIES:
            r.add_error(
                "E_INVALID_ORDERING",
                f"{ctx}.ordering_policy={op!r} is not a valid ordering policy; "
                f"valid: {sorted(_ALLOWED_ORDERING_POLICIES)}",
            )

    return r


def check_execution_plan(data: dict, strict: bool = False) -> CompatResult:
    """
    Extended execution plan hostile-input checker.

    Runs all M134 checks first, then adds:
      - E_UNKNOWN_TYPE: execution_phase or expected_status not in allowed set
      - E_DUPLICATE_ID: duplicate plan_command_id values
      - E_INVALID_TILE_ID: tile_id < 0
      - E_MISSING_OUTPUT_BUFFER: LOAD_TENSOR_TILE with no output_buffers
      - E_MISSING_TENSOR_DEP: EXEC_MATMUL/EXEC_RMSNORM with no tensor_dependencies
    """
    r = _base_check_execution_plan(data, strict)

    commands = data.get("commands")
    if not isinstance(commands, list):
        return r

    seen_ids: dict[Any, int] = {}
    for i, cmd in enumerate(commands):
        if not isinstance(cmd, dict):
            continue
        ctx = f"commands[{i}]"

        # Duplicate plan_command_id
        cid = cmd.get("plan_command_id")
        if cid is not None:
            if cid in seen_ids:
                r.add_error(
                    "E_DUPLICATE_ID",
                    f"{ctx}.plan_command_id={cid} duplicates "
                    f"commands[{seen_ids[cid]}].plan_command_id — duplicate identifier",
                )
            else:
                seen_ids[cid] = i

        # Unknown execution_phase
        phase = cmd.get("execution_phase")
        if phase is not None and phase not in _ALLOWED_EXEC_PHASES:
            r.add_error(
                "E_UNKNOWN_TYPE",
                f"{ctx}.execution_phase={phase!r} is not a recognised "
                f"ATT-1 execution phase",
            )

        # Negative tile_id
        tid = cmd.get("tile_id")
        if isinstance(tid, int) and tid < 0:
            r.add_error(
                "E_INVALID_TILE_ID",
                f"{ctx}.tile_id={tid} is negative — invalid tile identifier",
            )

        ctype = cmd.get("command_type")

        # LOAD op requires at least one output_buffer
        if ctype in _OUTPUT_BUF_REQUIRED:
            out_bufs = cmd.get("output_buffers")
            if isinstance(out_bufs, list) and len(out_bufs) == 0:
                r.add_error(
                    "E_MISSING_OUTPUT_BUFFER",
                    f"{ctx} command_type={ctype!r} requires at least one "
                    f"output_buffer but output_buffers is empty",
                )

        # EXEC_MATMUL/EXEC_RMSNORM require at least one tensor_dependency
        if ctype in _TENSOR_DEP_REQUIRED:
            deps = cmd.get("tensor_dependencies")
            if isinstance(deps, list) and len(deps) == 0:
                r.add_error(
                    "E_MISSING_TENSOR_DEP",
                    f"{ctx} command_type={ctype!r} requires at least one "
                    f"tensor_dependency but tensor_dependencies is empty",
                )

        # Unknown expected_status
        es = cmd.get("expected_status")
        if es is not None and es not in _ALLOWED_EXPECTED_STATUSES:
            r.add_error(
                "E_UNKNOWN_TYPE",
                f"{ctx}.expected_status={es!r} is not a recognised "
                f"expected status value; valid: {sorted(_ALLOWED_EXPECTED_STATUSES)}",
            )

    return r


def check_pipeline(data: dict, strict: bool = False) -> CompatResult:
    """
    Extended pipeline report hostile-input checker.

    Runs all M134 checks first, then adds:
      - E_NEGATIVE_TILE_COUNT: tile_count < 0
      - E_INCONSISTENT_COUNTS: commands_replayed > command_count
    """
    r = _base_check_pipeline(data, strict)

    tc = data.get("tile_count")
    if isinstance(tc, int) and tc < 0:
        r.add_error(
            "E_NEGATIVE_TILE_COUNT",
            f"tile_count={tc} is negative — impossible tile count",
        )

    cmd_count = data.get("command_count")
    replayed = data.get("commands_replayed")
    if (
        isinstance(cmd_count, int) and cmd_count >= 0
        and isinstance(replayed, int) and replayed > cmd_count
    ):
        r.add_error(
            "E_INCONSISTENT_COUNTS",
            f"commands_replayed={replayed} > command_count={cmd_count} — "
            f"cannot replay more commands than the plan declares",
        )

    return r


# ---------------------------------------------------------------------------
# Dispatch table
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
    print(f"ATT-1 M135 hostile-input check — {result.schema}")
    print(f"  Input  : {input_label}")
    print(f"  Status : {status_tag}")
    if not result.issues:
        print("  (no issues)")
    else:
        for iss in result.issues:
            tag = "ERROR  " if iss.severity == "error" else "warning"
            print(f"  [{tag}] {iss.code}: {iss.message}")
    print(f"  errors={len(result.errors)}  warnings={len(result.warnings)}")


def _to_json_report(result: CompatResult, input_label: str, strict: bool) -> dict:
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
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="ATT-1 M135 hostile-input checker for planning/control-plane documents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument(
        "--schema",
        required=True,
        choices=list(_CHECKERS),
        metavar="{" + "|".join(_CHECKERS) + "}",
        help="Schema type to check",
    )
    ap.add_argument("--input", required=True, metavar="FILE",
                    help="Input JSON file to check")
    ap.add_argument("--strict", action="store_true",
                    help="Promote warnings to errors")
    ap.add_argument("--report-json", metavar="FILE", dest="report_json",
                    help="Write JSON hostile-input report to this file")
    return ap.parse_args()


def main() -> int:
    args = _parse_args()
    strict = args.strict

    data = _load_json(args.input)
    checker = _CHECKERS[args.schema]
    result = checker(data, strict=strict)

    _print_result(result, args.input)

    if args.report_json:
        report = _to_json_report(result, args.input, strict)
        try:
            with open(args.report_json, "w") as fh:
                json.dump(report, fh, indent=2)
                fh.write("\n")
        except OSError as exc:
            print(f"error: could not write report JSON: {exc}", file=sys.stderr)

    return 1 if result.status == "fail" else 0


if __name__ == "__main__":
    sys.exit(main())
