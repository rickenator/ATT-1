#!/usr/bin/env python3
"""
ATT-1 placement-report-to-command-plan mapper (Milestone 109).

Reads an M98/M100 tensor placement report JSON and emits a deterministic
simulated AIMU command plan suitable for the M105 command queue / M103
command packet model.  This tool does NOT execute inference, change runtime
behavior, or access real PCIe/MMIO registers.

Exit codes:
  0 — command plan generated successfully (warnings may be present)
  1 — validation error (placement report problems prevent command generation)
  2 — parse error (malformed JSON or missing required top-level fields)

Usage:

    python3 compiler/map_placement_to_commands.py \\
        --report compiler/fixtures/placement_report_valid.json

    python3 compiler/map_placement_to_commands.py \\
        --report build/my_placement.json \\
        --model-id my_model \\
        --session-id session_0 \\
        --plan-json build/my_command_plan.json \\
        --strict

Command plan structure (see docs/tensor_placement_report.md §13):

  For each tile (sorted by tile_id):
    LOAD_TENSOR_TILE   — one per placed tensor on this tile
    VALIDATE_TENSOR    — one per placed tensor on this tile
    TILE_BARRIER       — one per tile (synchronisation fence)
  After all tiles:
    QUERY_COUNTERS     — read command-queue counters
    TRACE_SNAPSHOT     — capture trace snapshot

Command IDs are deterministic sequential integers starting from 1.
"""

import argparse
import json
import os
import sys
from collections import defaultdict

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

COMMAND_PLAN_VERSION = 1

SUPPORTED_REPORT_VERSIONS = {1}

SUPPORTED_DTYPES = {"f32", "q8", "q4"}

VALID_Q4_GROUP_SIZES = {32, 64}

# Command type names matching M103/M105 enumeration
CMD_LOAD_TENSOR_TILE  = "LOAD_TENSOR_TILE"
CMD_VALIDATE_TENSOR   = "VALIDATE_TENSOR"
CMD_TILE_BARRIER      = "TILE_BARRIER"
CMD_QUERY_COUNTERS    = "QUERY_COUNTERS"
CMD_TRACE_SNAPSHOT    = "TRACE_SNAPSHOT"

EXPECTED_STATUS_OK = "ATT1_AIMU_ERR_OK"


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(
        description="ATT-1 placement-report-to-command-plan mapper (M109).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--report", required=True, metavar="PATH",
        help="Path to M98/M100 placement report JSON.",
    )
    p.add_argument(
        "--model-id", default="", metavar="ID",
        help="Model identifier embedded in every command (default: empty).",
    )
    p.add_argument(
        "--session-id", default="session_0", metavar="ID",
        help="Session identifier embedded in every command (default: session_0).",
    )
    p.add_argument(
        "--plan-json", default=None, metavar="PATH",
        help="Write command plan JSON to PATH in addition to stdout text.",
    )
    p.add_argument(
        "--strict", action="store_true",
        help=(
            "Reject the report and exit 1 if any tile has capacity_status=FAIL "
            "or if any tensor has placement_status != placed."
        ),
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Report loading
# ---------------------------------------------------------------------------

def _load_report(path):
    """Load placement report JSON.  Exits with code 2 on parse failure."""
    if not os.path.exists(path):
        print(f"ERROR: report file not found: {path}", file=sys.stderr)
        sys.exit(2)
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as exc:
        print(f"ERROR: malformed JSON in {path}: {exc}", file=sys.stderr)
        sys.exit(2)
    except OSError as exc:
        print(f"ERROR: cannot read {path}: {exc}", file=sys.stderr)
        sys.exit(2)


# ---------------------------------------------------------------------------
# Report structure validation
# ---------------------------------------------------------------------------

def _require_top_level(report, path):
    """Check required top-level fields.  Exits 2 on structural failure."""
    ver = report.get("report_version")
    if ver is None:
        print("ERROR: missing 'report_version'", file=sys.stderr)
        sys.exit(2)
    if ver not in SUPPORTED_REPORT_VERSIONS:
        print(
            f"ERROR: unsupported report_version={ver!r}; "
            f"supported: {sorted(SUPPORTED_REPORT_VERSIONS)}",
            file=sys.stderr,
        )
        sys.exit(2)
    if "header" not in report:
        print("ERROR: missing 'header' section", file=sys.stderr)
        sys.exit(2)
    if "tensors" not in report:
        print("ERROR: missing 'tensors' section", file=sys.stderr)
        sys.exit(2)
    header = report["header"]
    if not isinstance(header, dict):
        print("ERROR: 'header' must be a JSON object", file=sys.stderr)
        sys.exit(2)
    tile_count = header.get("tile_count")
    if tile_count is None:
        print("ERROR: missing header.tile_count", file=sys.stderr)
        sys.exit(2)
    if not isinstance(tile_count, int) or tile_count <= 0:
        print(
            f"ERROR: header.tile_count must be a positive integer; got {tile_count!r}",
            file=sys.stderr,
        )
        sys.exit(2)
    tensors = report["tensors"]
    if not isinstance(tensors, list):
        print("ERROR: 'tensors' must be a JSON array", file=sys.stderr)
        sys.exit(2)
    return header, tensors


# ---------------------------------------------------------------------------
# Per-tensor validation helpers
# ---------------------------------------------------------------------------

def _validate_tensor(t, index, tile_count, strict, warnings, errors):
    """
    Validate a single tensor record.  Appends to errors/warnings.
    Returns the validated tensor dict or None if it should be skipped.
    """
    def _err(msg):
        errors.append(msg)

    def _warn(msg):
        warnings.append(msg)

    if not isinstance(t, dict):
        _err(f"tensors[{index}] is not a JSON object")
        return None

    # Required identity fields
    tensor_id = t.get("tensor_id")
    if tensor_id is None:
        _err(f"tensors[{index}]: missing 'tensor_id'")
        return None

    tensor_name = t.get("tensor_name")
    if tensor_name is None:
        _err(f"tensor_id={tensor_id}: missing 'tensor_name'")
        return None

    label = f"tensor {tensor_id} ({tensor_name!r})"

    # placement_status
    placement_status = t.get("placement_status")
    if placement_status is None:
        _err(f"{label}: missing 'placement_status'")
        return None
    if placement_status != "placed":
        if strict:
            _err(
                f"{label}: placement_status={placement_status!r}; "
                "only 'placed' tensors are accepted in --strict mode"
            )
            return None
        _warn(f"{label}: skipped (placement_status={placement_status!r})")
        return None  # skip non-placed tensor silently

    # owner_tile
    owner_tile = t.get("owner_tile")
    if owner_tile is None:
        _err(f"{label}: missing 'owner_tile'")
        return None
    if not isinstance(owner_tile, int) or owner_tile < 0 or owner_tile >= tile_count:
        _err(
            f"{label}: owner_tile={owner_tile!r} out of range "
            f"[0, {tile_count - 1}]"
        )
        return None

    # dtype
    dtype = t.get("dtype")
    if dtype is None:
        _err(f"{label}: missing 'dtype'")
        return None
    if dtype not in SUPPORTED_DTYPES:
        _err(f"{label}: unsupported dtype={dtype!r}; supported: {sorted(SUPPORTED_DTYPES)}")
        return None

    # q4 group size
    if dtype == "q4":
        qgs = t.get("quantization_group_size")
        if qgs is None:
            _err(
                f"{label}: dtype=q4 requires 'quantization_group_size'; "
                "field is missing or null"
            )
            return None
        if qgs not in VALID_Q4_GROUP_SIZES:
            _err(
                f"{label}: q4 quantization_group_size={qgs} is not supported; "
                f"valid: {sorted(VALID_Q4_GROUP_SIZES)}"
            )
            return None

    return t  # validation passed


# ---------------------------------------------------------------------------
# Command builders
# ---------------------------------------------------------------------------

def _make_load_cmd(cmd_id, t, model_id, session_id):
    """Build a LOAD_TENSOR_TILE command entry."""
    tensor_id    = t["tensor_id"]
    tensor_name  = t.get("tensor_name", "")
    dtype        = t.get("dtype", "f32")
    quant        = t.get("quantization") or "none"
    qgs          = t.get("quantization_group_size")
    packed_bytes = t.get("packed_bytes") or 0
    scale_bytes  = t.get("scale_bytes") or 0
    owner_tile   = t.get("owner_tile", 0)
    aimu_id      = t.get("owner_aimu", owner_tile)
    layer        = t.get("layer")
    category     = t.get("tensor_category", "other")
    checksum     = t.get("checksum", "0x0000000000000000")

    total_bytes = packed_bytes + scale_bytes

    notes_parts = [f"category={category}"]
    if layer is not None:
        notes_parts.append(f"layer={layer}")
    if dtype == "q4" and qgs is not None:
        notes_parts.append(
            f"q4_group_size={qgs} packed_bytes={packed_bytes} scale_bytes={scale_bytes}"
        )
    notes = "; ".join(notes_parts)

    return {
        "command_id":             cmd_id,
        "command_type":           CMD_LOAD_TENSOR_TILE,
        "tile_id":                owner_tile,
        "aimu_id":                aimu_id,
        "session_id":             session_id,
        "model_id":               model_id,
        "tensor_id":              tensor_id,
        "tensor_name":            tensor_name,
        "dtype":                  dtype,
        "quantization":           quant,
        "quantization_group_size": qgs,
        "packed_bytes":           packed_bytes,
        "scale_bytes":            scale_bytes,
        "total_bytes":            total_bytes,
        "src_descriptor":         f"host_buf:tensor_{tensor_id}",
        "dst_descriptor":         f"tile{owner_tile}_local:tensor_{tensor_id}",
        "fence_id":               0,
        "expected_status":        EXPECTED_STATUS_OK,
        "checksum":               checksum,
        "notes":                  notes,
    }


def _make_validate_cmd(cmd_id, t, model_id, session_id, load_cmd_id):
    """Build a VALIDATE_TENSOR command entry that depends on load_cmd_id."""
    tensor_id   = t["tensor_id"]
    tensor_name = t.get("tensor_name", "")
    dtype       = t.get("dtype", "f32")
    quant       = t.get("quantization") or "none"
    qgs         = t.get("quantization_group_size")
    owner_tile  = t.get("owner_tile", 0)
    aimu_id     = t.get("owner_aimu", owner_tile)
    packed_bytes = t.get("packed_bytes") or 0
    scale_bytes  = t.get("scale_bytes") or 0
    total_bytes  = packed_bytes + scale_bytes
    checksum     = t.get("checksum", "0x0000000000000000")

    return {
        "command_id":             cmd_id,
        "command_type":           CMD_VALIDATE_TENSOR,
        "tile_id":                owner_tile,
        "aimu_id":                aimu_id,
        "session_id":             session_id,
        "model_id":               model_id,
        "tensor_id":              tensor_id,
        "tensor_name":            tensor_name,
        "dtype":                  dtype,
        "quantization":           quant,
        "quantization_group_size": qgs,
        "packed_bytes":           packed_bytes,
        "scale_bytes":            scale_bytes,
        "total_bytes":            total_bytes,
        "src_descriptor":         f"tile{owner_tile}_local:tensor_{tensor_id}",
        "dst_descriptor":         None,
        "fence_id":               load_cmd_id,
        "expected_status":        EXPECTED_STATUS_OK,
        "checksum":               checksum,
        "notes":                  f"validate checksum of tensor {tensor_id} on tile {owner_tile}",
    }


def _make_barrier_cmd(cmd_id, tile_id, session_id, model_id, last_validate_id):
    """Build a TILE_BARRIER command entry."""
    return {
        "command_id":             cmd_id,
        "command_type":           CMD_TILE_BARRIER,
        "tile_id":                tile_id,
        "aimu_id":                tile_id,
        "session_id":             session_id,
        "model_id":               model_id,
        "tensor_id":              None,
        "tensor_name":            None,
        "dtype":                  None,
        "quantization":           None,
        "quantization_group_size": None,
        "packed_bytes":           0,
        "scale_bytes":            0,
        "total_bytes":            0,
        "src_descriptor":         None,
        "dst_descriptor":         None,
        "fence_id":               last_validate_id,
        "expected_status":        EXPECTED_STATUS_OK,
        "checksum":               None,
        "notes":                  f"synchronisation barrier: tile {tile_id} tensor load phase complete",
    }


def _make_query_counters_cmd(cmd_id, session_id, model_id, last_barrier_id):
    """Build a QUERY_COUNTERS command entry."""
    return {
        "command_id":             cmd_id,
        "command_type":           CMD_QUERY_COUNTERS,
        "tile_id":                0,
        "aimu_id":                0,
        "session_id":             session_id,
        "model_id":               model_id,
        "tensor_id":              None,
        "tensor_name":            None,
        "dtype":                  None,
        "quantization":           None,
        "quantization_group_size": None,
        "packed_bytes":           0,
        "scale_bytes":            0,
        "total_bytes":            0,
        "src_descriptor":         None,
        "dst_descriptor":         "host_counter_buf:counters",
        "fence_id":               last_barrier_id,
        "expected_status":        EXPECTED_STATUS_OK,
        "checksum":               None,
        "notes":                  "read command-queue counters after tensor load phase",
    }


def _make_trace_snapshot_cmd(cmd_id, session_id, model_id, last_barrier_id):
    """Build a TRACE_SNAPSHOT command entry."""
    return {
        "command_id":             cmd_id,
        "command_type":           CMD_TRACE_SNAPSHOT,
        "tile_id":                0,
        "aimu_id":                0,
        "session_id":             session_id,
        "model_id":               model_id,
        "tensor_id":              None,
        "tensor_name":            None,
        "dtype":                  None,
        "quantization":           None,
        "quantization_group_size": None,
        "packed_bytes":           0,
        "scale_bytes":            0,
        "total_bytes":            0,
        "src_descriptor":         None,
        "dst_descriptor":         "host_trace_buf:snapshot",
        "fence_id":               last_barrier_id,
        "expected_status":        EXPECTED_STATUS_OK,
        "checksum":               None,
        "notes":                  "capture trace snapshot after tensor load phase",
    }


# ---------------------------------------------------------------------------
# Command plan builder
# ---------------------------------------------------------------------------

def _build_commands(header, tensors, model_id, session_id, strict, warnings):
    """
    Build the deterministic command list from validated tensor records.

    Returns list of command dicts.  Exits 1 on validation errors.
    """
    tile_count = header["tile_count"]
    errors = []

    # Validate every tensor; collect placed tensors grouped by tile
    by_tile = defaultdict(list)
    for i, t in enumerate(tensors):
        validated = _validate_tensor(t, i, tile_count, strict, warnings, errors)
        if validated is not None:
            by_tile[validated["owner_tile"]].append(validated)

    if errors:
        for msg in errors:
            print(f"ERROR: {msg}", file=sys.stderr)
        sys.exit(1)

    commands = []
    cmd_id = 1

    last_barrier_id = 0

    for tile_id in sorted(by_tile.keys()):
        tile_tensors = sorted(
            by_tile[tile_id],
            key=lambda x: (x.get("tensor_id", 0), x.get("slice_start") or 0),
        )

        # ---- LOAD_TENSOR_TILE ----
        load_ids = []
        for t in tile_tensors:
            cmd = _make_load_cmd(cmd_id, t, model_id, session_id)
            commands.append(cmd)
            load_ids.append(cmd_id)
            cmd_id += 1

        # ---- VALIDATE_TENSOR ----
        validate_ids = []
        for t, lid in zip(tile_tensors, load_ids):
            cmd = _make_validate_cmd(cmd_id, t, model_id, session_id, lid)
            commands.append(cmd)
            validate_ids.append(cmd_id)
            cmd_id += 1

        # ---- TILE_BARRIER ----
        last_validate = validate_ids[-1] if validate_ids else 0
        cmd = _make_barrier_cmd(cmd_id, tile_id, session_id, model_id, last_validate)
        commands.append(cmd)
        last_barrier_id = cmd_id
        cmd_id += 1

    # ---- QUERY_COUNTERS ----
    commands.append(_make_query_counters_cmd(cmd_id, session_id, model_id, last_barrier_id))
    cmd_id += 1

    # ---- TRACE_SNAPSHOT ----
    commands.append(_make_trace_snapshot_cmd(cmd_id, session_id, model_id, last_barrier_id))

    return commands


# ---------------------------------------------------------------------------
# Summary builder
# ---------------------------------------------------------------------------

def _build_summary(commands, report_tiles):
    """Build the plan summary dict."""
    by_type = defaultdict(int)
    by_tile = defaultdict(int)
    total_tensor_bytes = 0
    f32_count = 0
    q8_count  = 0
    q4_count  = 0
    cap_fails = 0
    warns     = 0

    for cmd in commands:
        by_type[cmd["command_type"]] += 1
        tile_id = cmd.get("tile_id")
        if tile_id is not None:
            by_tile[tile_id] += 1
        if cmd["command_type"] == CMD_LOAD_TENSOR_TILE:
            total_tensor_bytes += cmd.get("total_bytes", 0)
            dtype = cmd.get("dtype")
            if dtype == "f32":
                f32_count += 1
            elif dtype == "q8":
                q8_count += 1
            elif dtype == "q4":
                q4_count += 1

    for tile in (report_tiles or []):
        cs = tile.get("capacity_status", "UNKNOWN")
        if cs == "FAIL":
            cap_fails += 1
        bs = tile.get("bandwidth_status", "UNKNOWN")
        if bs in ("WARN", "FAIL"):
            warns += 1

    return {
        "commands_by_type":     dict(by_type),
        "commands_by_tile":     {str(k): v for k, v in sorted(by_tile.items())},
        "total_tensor_bytes":   total_tensor_bytes,
        "f32_tensor_count":     f32_count,
        "q8_tensor_count":      q8_count,
        "q4_tensor_count":      q4_count,
        "capacity_failures_observed": cap_fails,
        "warnings_observed":    warns,
    }


# ---------------------------------------------------------------------------
# Human-readable renderer
# ---------------------------------------------------------------------------

def _render_plan(plan):
    """Render the command plan to stdout in human-readable form."""
    hdr = plan["header"]
    print("=" * 72)
    print("ATT-1 AIMU Command Plan")
    print("=" * 72)
    print(f"  command_plan_version : {hdr['command_plan_version']}")
    print(f"  source_report_path   : {hdr['source_report_path']}")
    print(f"  model_name           : {hdr['model_name']}")
    print(f"  model_id             : {hdr['model_id']}")
    print(f"  session_id           : {hdr['session_id']}")
    print(f"  tile_count           : {hdr['tile_count']}")
    print(f"  tensor_count         : {hdr['tensor_count']}")
    print(f"  command_count        : {hdr['command_count']}")
    print(f"  status               : {hdr['status']}")

    if plan.get("warnings"):
        print()
        print("WARNINGS:")
        for w in plan["warnings"]:
            print(f"  {w}")

    print()
    print("-" * 72)
    print("COMMANDS")
    print("-" * 72)

    for cmd in plan["commands"]:
        print(f"  [{cmd['command_id']:4d}] {cmd['command_type']:<22s}"
              f"  tile={cmd['tile_id']}  fence={cmd['fence_id']}")
        if cmd.get("tensor_name"):
            qs = (f"  q_group={cmd['quantization_group_size']}"
                  if cmd.get("quantization_group_size") else "")
            print(f"         tensor={cmd['tensor_id']} {cmd['tensor_name']!r}"
                  f"  dtype={cmd['dtype']}{qs}"
                  f"  bytes={cmd['total_bytes']}")
        if cmd.get("notes"):
            print(f"         notes: {cmd['notes']}")

    sm = plan["summary"]
    print()
    print("-" * 72)
    print("SUMMARY")
    print("-" * 72)
    print("  Commands by type:")
    for ctype, cnt in sorted(sm["commands_by_type"].items()):
        print(f"    {ctype:<26s}  {cnt}")
    print("  Commands by tile:")
    for tile_id, cnt in sorted(sm["commands_by_tile"].items()):
        print(f"    tile {tile_id:<3s}  {cnt} commands")
    print(f"  Total tensor bytes scheduled : {sm['total_tensor_bytes']:,}")
    print(f"  f32 tensor commands          : {sm['f32_tensor_count']}")
    print(f"  q8  tensor commands          : {sm['q8_tensor_count']}")
    print(f"  q4  tensor commands          : {sm['q4_tensor_count']}")
    print(f"  Capacity failures observed   : {sm['capacity_failures_observed']}")
    print(f"  Warnings observed            : {sm['warnings_observed']}")
    print("=" * 72)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = _parse_args()

    report = _load_report(args.report)
    header, tensors = _require_top_level(report, args.report)

    # Pull tiles list for capacity checks (optional section)
    report_tiles = report.get("tiles") or []

    # Attach tiles list to header for _build_commands access
    header["_tiles_list"] = report_tiles

    # Capacity check warning in non-strict mode
    warnings = []
    for tile in report_tiles:
        if tile.get("capacity_status") == "FAIL":
            msg = (
                f"tile {tile.get('tile_id', '?')}: capacity_status=FAIL "
                "(model bytes exceed tile memory; consider larger tile or more tiles)"
            )
            if args.strict:
                print(f"ERROR: {msg}", file=sys.stderr)
                sys.exit(1)
            warnings.append(f"WARNING: {msg}")

    model_id   = args.model_id   or header.get("model_name", "") or ""
    session_id = args.session_id or "session_0"

    commands = _build_commands(
        header, tensors, model_id, session_id, args.strict, warnings
    )

    # Count placed tensors (LOAD commands)
    tensor_count = sum(
        1 for cmd in commands if cmd["command_type"] == CMD_LOAD_TENSOR_TILE
    )

    plan = {
        "command_plan_version": COMMAND_PLAN_VERSION,
        "header": {
            "command_plan_version": COMMAND_PLAN_VERSION,
            "source_report_path":  os.path.abspath(args.report),
            "model_name":          header.get("model_name", ""),
            "model_id":            model_id,
            "session_id":          session_id,
            "tile_count":          header["tile_count"],
            "tensor_count":        tensor_count,
            "command_count":       len(commands),
            "status":              "ok",
        },
        "commands": commands,
        "summary":  _build_summary(commands, report_tiles),
        "warnings": warnings,
    }

    _render_plan(plan)

    if args.plan_json:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(args.plan_json)), exist_ok=True)
            with open(args.plan_json, "w", encoding="utf-8") as f:
                # Exclude internal _tiles_list key before serialising
                out = dict(plan)
                json.dump(out, f, indent=2)
                f.write("\n")
        except OSError as exc:
            print(f"ERROR: cannot write plan JSON to {args.plan_json}: {exc}",
                  file=sys.stderr)
            sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
