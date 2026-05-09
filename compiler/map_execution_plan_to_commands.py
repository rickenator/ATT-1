#!/usr/bin/env python3
"""
ATT-1 execution-plan-to-command-plan mapper (Milestone 129).

Reads an M125 tensor-level execution-plan JSON (optionally validated by the
M128 validator) and emits an M109-compatible AIMU command-plan JSON suitable
for existing command-plan replay tooling (M113, M122).

This tool does NOT execute inference, change runtime behavior, access real
PCIe/MMIO registers, or implement a kernel driver.  All output is advisory.

Exit codes:
  0 — mapping succeeded (warnings may be present)
  1 — mapping error (invalid or unsupported input)
  2 — parse error (malformed JSON or missing required field)

Usage:
    python3 compiler/map_execution_plan_to_commands.py \\
        --execution-plan compiler/fixtures/exec_plan_valid_tiny.json

    python3 compiler/map_execution_plan_to_commands.py \\
        --execution-plan build/execution_plan.json \\
        --plan-json build/command_plan.json \\
        --model-id my_model \\
        --session-id session_0 \\
        --strict
"""

import argparse
import json
import sys
from collections import defaultdict

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

COMMAND_PLAN_VERSION = 1
SUPPORTED_EXEC_PLAN_VERSIONS: set = {1}

# M103/M105 status strings
STATUS_OK          = "ATT1_AIMU_ERR_OK"
STATUS_UNSUPPORTED = "ATT1_AIMU_ERR_UNSUPPORTED_OP"

# Recognised execution phases from M125 §2 / M128
VALID_EXECUTION_PHASES: set = {
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
}

# Phases that are purely advisory — produce a NOP/QUERY placeholder in the
# command plan rather than a direct M105 command.
ADVISORY_PHASES: set = {
    "DEVICE_PROBE",
    "TILE_ENUMERATION",
    "MEMORY_ALLOCATE",
    "PREFILL_SETUP",
}

# Execution-phase → command-plan command_type mapping.
# EXEC_* types map to their M105 names but are marked STATUS_UNSUPPORTED
# unless the caller opts in via --allow-exec.
EXEC_CMD_TYPES: set = {
    "EXEC_MATMUL",
    "EXEC_RMSNORM",
    "EXEC_ROPE",
    "EXEC_ATTENTION",
    "EXEC_SWIGLU",
    "EXEC_SOFTMAX",
    "EXEC_RESIDUAL",
}

# Direct pass-through command types (M105 recognises them natively)
PASSTHROUGH_CMD_TYPES: set = {
    "LOAD_TENSOR_TILE",
    "VALIDATE_TENSOR",
    "KV_APPEND",
    "KV_READ",
    "FABRIC_SEND",
    "FABRIC_REDUCE",
    "TILE_BARRIER",
    "TRACE_SNAPSHOT",
    "QUERY_COUNTERS",
}

# RESET_TILE is the M105 equivalent; if absent we fall through to NOP
RESET_MAPPING = "RESET_TILE"

# Advisory phases produce a NOP with a descriptive note.
_ADVISORY_NOTE: dict = {
    "DEVICE_PROBE":     "advisory: DEVICE_PROBE has no direct M105 command; use M121 API",
    "TILE_ENUMERATION": "advisory: TILE_ENUMERATION maps to QUERY_COUNTERS for M105",
    "MEMORY_ALLOCATE":  "advisory: MEMORY_ALLOCATE handled by M124 allocator; no M105 command",
    "PREFILL_SETUP":    "advisory: PREFILL_SETUP barrier — replayed as TILE_BARRIER",
}
_ADVISORY_CMD_TYPE: dict = {
    "TILE_ENUMERATION": "QUERY_COUNTERS",
    "PREFILL_SETUP":    "TILE_BARRIER",
}


# ---------------------------------------------------------------------------
# ParseError
# ---------------------------------------------------------------------------

class ParseError(Exception):
    pass


class MappingError(Exception):
    pass


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _load_plan(path: str) -> dict:
    try:
        with open(path, "r") as fh:
            data = json.load(fh)
    except FileNotFoundError:
        raise ParseError(f"execution plan not found: {path}")
    except json.JSONDecodeError as exc:
        raise ParseError(f"malformed JSON in execution plan: {exc}")
    if not isinstance(data, dict):
        raise ParseError("execution plan must be a JSON object at top level")
    return data


# ---------------------------------------------------------------------------
# Header validation (lightweight — full M128 validation is separate)
# ---------------------------------------------------------------------------

def _validate_header(plan: dict) -> tuple:
    """
    Check minimum required fields.  Returns (tile_count, model_id, session_id).
    Raises MappingError on hard failures.
    """
    ver = plan.get("execution_plan_version")
    if ver is None:
        raise MappingError("missing 'execution_plan_version'")
    if ver not in SUPPORTED_EXEC_PLAN_VERSIONS:
        raise MappingError(
            f"unsupported execution_plan_version={ver!r}; "
            f"supported: {sorted(SUPPORTED_EXEC_PLAN_VERSIONS)}"
        )
    if not plan.get("model_id"):
        raise MappingError("missing or empty 'model_id'")
    if not plan.get("session_id"):
        raise MappingError("missing or empty 'session_id'")
    tile_count = plan.get("tile_count")
    if tile_count is None or not isinstance(tile_count, int) or tile_count <= 0:
        raise MappingError(
            f"tile_count must be a positive integer; got {tile_count!r}"
        )
    commands = plan.get("commands")
    if commands is None or not isinstance(commands, list):
        raise MappingError("missing or invalid 'commands' list")
    return tile_count, plan["model_id"], plan["session_id"]


# ---------------------------------------------------------------------------
# Per-command validation
# ---------------------------------------------------------------------------

def _check_command(
    cmd: dict,
    idx: int,
    tile_count: int,
    seen_ids: set,
    fence_map: dict,
    strict: bool,
    warnings: list,
    errors: list,
) -> None:
    """Validate a single execution-plan command record, recording issues."""

    def err(msg: str) -> None:
        errors.append(f"commands[{idx}]: {msg}")

    def warn(msg: str) -> None:
        warnings.append(f"commands[{idx}]: {msg}")

    if not isinstance(cmd, dict):
        err("not a JSON object")
        return

    # plan_command_id
    cid = cmd.get("plan_command_id")
    if cid is None or not isinstance(cid, int):
        err(f"plan_command_id must be an integer; got {cid!r}")
    elif cid in seen_ids:
        err(f"duplicate plan_command_id={cid}")
    else:
        seen_ids.add(cid)

    # tile_id
    tid = cmd.get("tile_id")
    if tid is None or not isinstance(tid, int) or tid < 0:
        err(f"tile_id must be a non-negative integer; got {tid!r}")
    elif tile_count > 0 and tid >= tile_count:
        err(f"tile_id={tid} >= tile_count={tile_count}")

    # execution_phase
    phase = cmd.get("execution_phase")
    if phase is None:
        err("missing 'execution_phase'")
    elif phase not in VALID_EXECUTION_PHASES:
        err(f"unknown execution_phase={phase!r}")

    # command_type
    ct = cmd.get("command_type")
    if ct is None:
        err("missing 'command_type'")
    else:
        # tensor deps required for load/validate/matmul/rmsnorm
        if ct in ("LOAD_TENSOR_TILE", "VALIDATE_TENSOR",
                  "EXEC_MATMUL", "EXEC_RMSNORM"):
            tdeps = cmd.get("tensor_dependencies", [])
            if not isinstance(tdeps, list) or len(tdeps) == 0:
                err(f"command_type={ct} requires at least one tensor_dependency")

    # dependency_fence_id must reference a prior command's fence_id
    dep_fid = cmd.get("dependency_fence_id")
    if dep_fid is not None and dep_fid != 0:
        if not isinstance(dep_fid, int):
            err(f"dependency_fence_id must be an integer; got {dep_fid!r}")
        elif dep_fid not in fence_map:
            err(
                f"dependency_fence_id={dep_fid} does not match any "
                f"fence_id seen so far (dependency on future command?)"
            )

    # Record this command's fence_id for subsequent dependency checks
    fid = cmd.get("fence_id")
    if isinstance(fid, int) and fid != 0:
        fence_map[fid] = idx

    # Input buffers required for DMA/load style
    ct = cmd.get("command_type", "")
    ibufs = cmd.get("input_buffers", [])
    if ct == "FABRIC_SEND" and (not isinstance(ibufs, list) or len(ibufs) == 0):
        if strict:
            err("FABRIC_SEND requires at least one input_buffer in --strict mode")
        else:
            warn("FABRIC_SEND has no input_buffers")


# ---------------------------------------------------------------------------
# Command record builders
# ---------------------------------------------------------------------------

def _null_cmd(
    cmd_id: int,
    ct: str,
    tile_id: int,
    model_id: str,
    session_id: str,
    fence_id: int,
    dep_fence_id: int,
    expected_status: str,
    notes: str,
    tensor_name: str = None,
    tensor_id: int = None,
    dtype: str = None,
    quantization: str = None,
    qgs: int = None,
    byte_count: int = 0,
) -> dict:
    """Build a minimal M109-compatible command record."""
    return {
        "command_id":              cmd_id,
        "command_type":            ct,
        "tile_id":                 tile_id,
        "aimu_id":                 tile_id,
        "session_id":              session_id,
        "model_id":                model_id,
        "tensor_id":               tensor_id,
        "tensor_name":             tensor_name,
        "dtype":                   dtype,
        "quantization":            quantization,
        "quantization_group_size": qgs,
        "packed_bytes":            byte_count,
        "scale_bytes":             0,
        "total_bytes":             byte_count,
        "src_descriptor":          None,
        "dst_descriptor":          None,
        "fence_id":                fence_id,
        "dependency_fence_id":     dep_fence_id,
        "expected_status":         expected_status,
        "checksum":                None,
        "notes":                   notes,
    }


# ---------------------------------------------------------------------------
# Phase-level mapping
# ---------------------------------------------------------------------------

def _map_command(
    src: dict,
    cmd_id: int,
    model_id: str,
    session_id: str,
    strict: bool,
    warnings: list,
) -> dict:
    """
    Convert one M125 execution-plan command record into an M109 command record.

    For EXEC_* types the expected_status is STATUS_UNSUPPORTED unless strict
    mode is off (they are passed through with that status to allow replay tools
    to document unsupported ops).  In strict mode unsupported EXEC_* commands
    cause a MappingError — callers should have validated the plan first.
    """
    phase = src.get("execution_phase", "")
    ct    = src.get("command_type", "")
    tid   = src.get("tile_id", 0)
    fid   = src.get("fence_id", 0)
    dep   = src.get("dependency_fence_id", 0)

    # Resolve tensor metadata (best-effort from tensor_dependencies + buffers)
    tdeps = src.get("tensor_dependencies") or []
    tensor_name = tdeps[0] if tdeps else None
    tensor_id   = src.get("tensor_id")  # usually absent in exec plans

    # dtype from first buffer if available
    ibufs = src.get("input_buffers") or []
    obufs = src.get("output_buffers") or []
    all_bufs = ibufs + obufs
    dtype = None
    byte_count = 0
    quantization = None
    qgs = None
    for buf in all_bufs:
        if isinstance(buf, dict):
            if dtype is None and buf.get("dtype"):
                dtype = buf["dtype"]
                quantization = "none" if dtype in ("f32", "bf16") else dtype
                qgs = buf.get("q4_group_size")
            if isinstance(buf.get("byte_size"), int):
                byte_count += buf["byte_size"]

    # -----------------------------------------------------------------------
    # Advisory phases — no direct M105 command
    # -----------------------------------------------------------------------
    if phase in ADVISORY_PHASES:
        mapped_ct = _ADVISORY_CMD_TYPE.get(phase, "NOP")
        note = _ADVISORY_NOTE.get(phase, f"advisory phase: {phase}")
        return _null_cmd(
            cmd_id, mapped_ct, tid, model_id, session_id,
            fid, dep, STATUS_OK, note,
        )

    # -----------------------------------------------------------------------
    # EXEC_* types — current simulator does not execute tensor math
    # -----------------------------------------------------------------------
    if ct in EXEC_CMD_TYPES:
        if strict:
            raise MappingError(
                f"command_type={ct!r} is an EXEC_* type not supported by the "
                f"current M105 simulator; remove from plan or use non-strict mode "
                f"to emit with expected_status=UNSUPPORTED"
            )
        warnings.append(
            f"command_type={ct!r} mapped with expected_status=UNSUPPORTED "
            f"(M105 simulator does not execute tensor math)"
        )
        note = (
            f"exec-plan phase={phase} op={ct}; "
            f"tensor_deps={tdeps}; "
            f"UNSUPPORTED in current M105 simulator"
        )
        return _null_cmd(
            cmd_id, ct, tid, model_id, session_id,
            fid, dep, STATUS_UNSUPPORTED, note,
            tensor_name=tensor_name,
            dtype=dtype,
            quantization=quantization,
            qgs=qgs,
            byte_count=byte_count,
        )

    # -----------------------------------------------------------------------
    # RESET phase
    # -----------------------------------------------------------------------
    if ct == "RESET" or phase == "RESET":
        return _null_cmd(
            cmd_id, RESET_MAPPING, tid, model_id, session_id,
            fid, dep, STATUS_OK,
            f"tile reset; exec-plan phase={phase}",
        )

    # -----------------------------------------------------------------------
    # CLEANUP phase — emit QUERY_COUNTERS + TRACE_SNAPSHOT pair but only one
    # command per call; the calling loop will handle multi-phase
    # -----------------------------------------------------------------------
    if phase == "CLEANUP" and ct == "QUERY_COUNTERS":
        return _null_cmd(
            cmd_id, "QUERY_COUNTERS", tid, model_id, session_id,
            fid, dep, STATUS_OK,
            "cleanup phase: read counters before teardown",
        )

    # -----------------------------------------------------------------------
    # Pass-through command types (LOAD_TENSOR_TILE, VALIDATE_TENSOR, KV_*,
    # FABRIC_*, TILE_BARRIER, TRACE_SNAPSHOT, QUERY_COUNTERS)
    # -----------------------------------------------------------------------
    if ct in PASSTHROUGH_CMD_TYPES:
        note_parts = [f"exec-plan phase={phase}"]
        if tdeps:
            note_parts.append(f"tensor_deps={tdeps}")
        note = "; ".join(note_parts)

        # Build descriptors for tensor-related commands
        src_desc = None
        dst_desc = None
        if ct == "LOAD_TENSOR_TILE" and tensor_name:
            src_desc = f"host_buf:{tensor_name}"
            dst_desc = f"tile{tid}_local:{tensor_name}"
        elif ct == "VALIDATE_TENSOR" and tensor_name:
            src_desc = f"tile{tid}_local:{tensor_name}"
        elif ct == "TRACE_SNAPSHOT":
            dst_desc = "host_trace_buf:snapshot"
        elif ct == "QUERY_COUNTERS":
            dst_desc = "host_counter_buf:counters"
        elif ct in ("FABRIC_SEND", "FABRIC_REDUCE"):
            dst_tile = src.get("dst_tile")
            if dst_tile is not None:
                dst_desc = f"tile{dst_tile}_fabric_buf:payload"

        rec = _null_cmd(
            cmd_id, ct, tid, model_id, session_id,
            fid, dep, STATUS_OK, note,
            tensor_name=tensor_name,
            tensor_id=tensor_id,
            dtype=dtype,
            quantization=quantization,
            qgs=qgs,
            byte_count=byte_count,
        )
        if src_desc is not None:
            rec["src_descriptor"] = src_desc
        if dst_desc is not None:
            rec["dst_descriptor"] = dst_desc
        return rec

    # -----------------------------------------------------------------------
    # Unrecognised command type — fail clearly
    # -----------------------------------------------------------------------
    raise MappingError(
        f"command_type={ct!r} is not recognised by the M129 mapper; "
        f"check the execution plan was generated by M125"
    )


# ---------------------------------------------------------------------------
# Top-level mapping function
# ---------------------------------------------------------------------------

def map_plan(
    plan: dict,
    override_model_id: str = None,
    override_session_id: str = None,
    strict: bool = False,
) -> tuple:
    """
    Map an M125 execution plan to an M109 command plan.

    Returns (command_plan_dict, warnings_list).
    Raises MappingError on hard failures.
    """
    tile_count, plan_model_id, plan_session_id = _validate_header(plan)

    model_id   = override_model_id   or plan_model_id
    session_id = override_session_id or plan_session_id
    commands_in = plan.get("commands", [])

    warnings: list = []
    errors:   list = []
    seen_ids: set  = set()
    fence_map: dict = {}

    # Validate all commands first
    for i, cmd in enumerate(commands_in):
        _check_command(cmd, i, tile_count, seen_ids, fence_map, strict, warnings, errors)

    if errors:
        raise MappingError(
            f"{len(errors)} validation error(s) in execution plan:\n"
            + "\n".join(f"  {e}" for e in errors)
        )

    # Map each command
    out_commands: list = []
    cmd_id = 1
    for src in commands_in:
        if not isinstance(src, dict):
            continue
        mapped = _map_command(src, cmd_id, model_id, session_id, strict, warnings)
        out_commands.append(mapped)
        cmd_id += 1

    # Build summary counters
    by_type: dict = defaultdict(int)
    by_tile: dict = defaultdict(int)
    for c in out_commands:
        by_type[c["command_type"]] += 1
        by_tile[str(c["tile_id"])] += 1

    command_plan = {
        "command_plan_version": COMMAND_PLAN_VERSION,
        "source_execution_plan": plan.get("placement_report_path"),
        "header": {
            "command_plan_version": COMMAND_PLAN_VERSION,
            "source_execution_plan": None,
            "model_name": model_id,
            "model_id": model_id,
            "session_id": session_id,
            "tile_count": tile_count,
            "tensor_count": 0,
            "command_count": len(out_commands),
            "status": "ok",
        },
        "commands": out_commands,
        "summary": {
            "commands_by_type": dict(by_type),
            "commands_by_tile": dict(by_tile),
            "total_tensor_bytes": sum(
                c.get("total_bytes", 0) or 0 for c in out_commands
            ),
            "f32_tensor_count": sum(
                1 for c in out_commands if c.get("dtype") == "f32"
            ),
            "q8_tensor_count": sum(
                1 for c in out_commands if c.get("dtype") == "q8"
            ),
            "q4_tensor_count": sum(
                1 for c in out_commands if c.get("dtype") == "q4"
            ),
            "warnings_observed": len(warnings),
            "unsupported_exec_commands": sum(
                1 for c in out_commands
                if c.get("expected_status") == STATUS_UNSUPPORTED
            ),
        },
        "warnings": warnings,
    }
    return command_plan, warnings


# ---------------------------------------------------------------------------
# Human-readable report
# ---------------------------------------------------------------------------

def _print_report(
    exec_plan_path: str,
    command_plan: dict,
    warnings: list,
) -> None:
    hdr = command_plan.get("header", {})
    smry = command_plan.get("summary", {})
    print(f"execution-plan-to-command-plan mapper  source={exec_plan_path}")
    print(f"  model_id          : {hdr.get('model_id', '?')}")
    print(f"  session_id        : {hdr.get('session_id', '?')}")
    print(f"  tile_count        : {hdr.get('tile_count', '?')}")
    print(f"  command_count     : {hdr.get('command_count', '?')}")
    print(f"  unsupported_exec  : {smry.get('unsupported_exec_commands', 0)}")
    print(f"  warning_count     : {len(warnings)}")
    if warnings:
        print("warnings:")
        for w in warnings:
            print(f"  [W] {w}")
    cmds_by_type = smry.get("commands_by_type", {})
    if cmds_by_type:
        print("commands_by_type:")
        for ct, n in sorted(cmds_by_type.items()):
            print(f"  {ct}: {n}")
    print("status: PASS")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Map an M125 execution-plan JSON to an M109 command-plan JSON."
    )
    parser.add_argument(
        "--execution-plan", required=True, metavar="PATH",
        help="Path to the M125 execution-plan JSON",
    )
    parser.add_argument(
        "--plan-json", metavar="PATH",
        help="Write the output M109 command-plan JSON to this file",
    )
    parser.add_argument(
        "--model-id", metavar="ID",
        help="Override model_id in the output (default: from execution plan)",
    )
    parser.add_argument(
        "--session-id", metavar="ID",
        help="Override session_id in the output (default: from execution plan)",
    )
    parser.add_argument(
        "--strict", action="store_true",
        help=(
            "Promote warnings to errors; reject EXEC_* commands without "
            "explicit expected_status=UNSUPPORTED override"
        ),
    )
    args = parser.parse_args(argv)

    try:
        plan = _load_plan(args.execution_plan)
    except ParseError as exc:
        print(f"PARSE ERROR: {exc}", file=sys.stderr)
        return 2

    try:
        command_plan, warnings = map_plan(
            plan,
            override_model_id=args.model_id,
            override_session_id=args.session_id,
            strict=args.strict,
        )
    except MappingError as exc:
        print(f"MAPPING ERROR: {exc}", file=sys.stderr)
        return 1

    _print_report(args.execution_plan, command_plan, warnings)

    if args.plan_json:
        with open(args.plan_json, "w") as fh:
            json.dump(command_plan, fh, indent=2)
        print(f"wrote command plan: {args.plan_json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
