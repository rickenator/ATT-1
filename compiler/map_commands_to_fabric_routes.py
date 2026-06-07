#!/usr/bin/env python3
"""
ATT-1 command-plan-to-fabric-route mapper (Milestone 117).

Reads an M109 AIMU command plan JSON and emits an M115-compatible fabric route
report JSON.  An optional M100/M98 placement report can be supplied to improve
payload byte estimates using actual tensor sizes.

This tool does NOT execute routes, change inference behavior, access real
PCIe/MMIO registers, or implement a kernel driver.

Exit codes:
  0 — route report generated successfully (warnings may be present)
  1 — mapping failed (structural error in command plan or strict-mode violation)
  2 — parse error (malformed JSON or missing required fields)

Usage:

    python3 compiler/map_commands_to_fabric_routes.py \\
        --plan compiler/fixtures/plan_valid.json

    python3 compiler/map_commands_to_fabric_routes.py \\
        --plan build/my_plan.json \\
        --placement-report build/my_placement.json \\
        --route-report-json build/my_routes.json \\
        --tokens-per-sec 100 \\
        --fabric-gib-sec 32 \\
        --strict

Route generation rules (derived from §11 schema):
  LOAD_TENSOR_TILE   → no AIMU-to-AIMU data route (host-to-tile DMA, not fabric)
  VALIDATE_TENSOR    → CONTROL_ACK route (64-byte control payload, tile→host)
  TILE_BARRIER       → TILE_BARRIER route (all tiles listed as destinations)
  QUERY_COUNTERS     → CONTROL_ACK route (64-byte counter snapshot payload)
  TRACE_SNAPSHOT     → TRACE_EVENT route (64-byte trace snapshot payload)
  FABRIC_SEND        → ACTIVATION_SEND route (payload from total_bytes)
  FABRIC_REDUCE      → PARTIAL_REDUCE route (payload from total_bytes, requires reduction_behavior)
  EXEC_*             → no fabric route (local tile operation)
  KV_*               → no fabric route (local tile KV cache operation)
  NOP/RESET_TILE     → no fabric route
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ROUTE_REPORT_VERSION = 1
COMMAND_PLAN_VERSION = 1

# Command types that produce fabric routes
CMD_TYPES_WITH_ROUTES = frozenset({
    "VALIDATE_TENSOR",
    "TILE_BARRIER",
    "QUERY_COUNTERS",
    "TRACE_SNAPSHOT",
    "FABRIC_SEND",
    "FABRIC_REDUCE",
})

# Command types that are local-only (no fabric route emitted, not an error)
CMD_TYPES_LOCAL_ONLY = frozenset({
    "LOAD_TENSOR_TILE",
    "EXEC_MATMUL",
    "EXEC_SOFTMAX",
    "EXEC_RMSNORM",
    "EXEC_ROPE",
    "EXEC_ATTENTION",
    "EXEC_FFN",
    "KV_APPEND",
    "KV_READ",
    "NOP",
    "RESET_TILE",
})

ALL_KNOWN_CMD_TYPES = CMD_TYPES_WITH_ROUTES | CMD_TYPES_LOCAL_ONLY

# Payload bytes for control/housekeeping routes (64 bytes = one AIMU register
# snapshot or counter block; matches M103 §8 counter snapshot size).
CONTROL_PAYLOAD_BYTES = 64
TRACE_PAYLOAD_BYTES = 64

VALID_FABRIC_POLICIES = {"layer_wise", "token_wise", "mixed"}

# Default payload estimate used when placement data is unavailable (one f32
# tile activation row: 256 elements × 4 bytes).
DEFAULT_ACTIVATION_PAYLOAD_BYTES = 1024


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ATT-1 command-plan-to-fabric-route mapper (M117).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--plan", required=True, metavar="PATH",
        help="Path to M109 AIMU command plan JSON.",
    )
    p.add_argument(
        "--placement-report", metavar="PATH",
        help="Optional: M100 placement report JSON for tensor size information.",
    )
    p.add_argument(
        "--route-report-json", metavar="PATH",
        help="Write generated M115 route report JSON to PATH.",
    )
    p.add_argument(
        "--tokens-per-sec", type=float, metavar="N",
        help="Estimated inference tokens/sec (informational; written to report header).",
    )
    p.add_argument(
        "--fabric-gib-sec", type=float, metavar="N",
        help="Available fabric bandwidth in GiB/sec (informational; written to report header).",
    )
    p.add_argument(
        "--strict", action="store_true",
        help="Fail if any warnings are produced or if plan status is not 'ok'.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# JSON parsing helpers
# ---------------------------------------------------------------------------

class ParseError(Exception):
    """Raised for malformed JSON or missing required fields (exit 2)."""


class MappingError(Exception):
    """Raised for structural errors that prevent route generation (exit 1)."""


def _load_json(path: str) -> dict[str, Any]:
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except FileNotFoundError:
        raise ParseError(f"file not found: {path}")
    except json.JSONDecodeError as exc:
        raise ParseError(f"malformed JSON in {path}: {exc}")
    except OSError as exc:
        raise ParseError(f"cannot read {path}: {exc}")
    if not isinstance(data, dict):
        raise ParseError(f"JSON root must be an object: {path}")
    return data


def _require(d: dict, key: str, context: str) -> Any:
    val = d.get(key)
    if val is None:
        raise ParseError(f"{context}: required field '{key}' is missing or null")
    return val


# ---------------------------------------------------------------------------
# Command plan validation
# ---------------------------------------------------------------------------

def _validate_plan(plan: dict[str, Any], strict: bool) -> list[str]:
    """Validate M109 plan structure. Returns list of warning strings."""
    warnings: list[str] = []

    for key in ("command_plan_version", "header", "commands"):
        if key not in plan:
            raise ParseError(f"command plan missing required top-level key: '{key}'")

    ver = plan.get("command_plan_version")
    if ver != COMMAND_PLAN_VERSION:
        raise ParseError(
            f"command_plan_version={ver!r} is not supported (expected {COMMAND_PLAN_VERSION})"
        )

    header = plan["header"]
    for key in ("tile_count", "command_count"):
        if key not in header:
            raise ParseError(f"command plan header missing required key: '{key}'")

    tile_count = header.get("tile_count")
    if not isinstance(tile_count, int) or tile_count <= 0:
        raise ParseError(f"header.tile_count={tile_count!r} must be a positive integer")

    commands = plan["commands"]
    if not isinstance(commands, list):
        raise ParseError("'commands' must be a JSON array")

    plan_status = header.get("status", "ok")
    if strict and plan_status != "ok":
        raise MappingError(f"--strict: command plan header.status={plan_status!r} is not 'ok'")

    seen_ids: set[int] = set()
    for idx, cmd in enumerate(commands):
        if not isinstance(cmd, dict):
            raise ParseError(f"commands[{idx}] is not an object")
        for key in ("command_id", "command_type", "tile_id", "fence_id"):
            if key not in cmd:
                raise ParseError(f"commands[{idx}]: missing required field '{key}'")
        cid = cmd["command_id"]
        if not isinstance(cid, int):
            raise ParseError(f"commands[{idx}].command_id must be an integer")
        if cid in seen_ids:
            warnings.append(f"commands[{idx}]: duplicate command_id={cid}")
        else:
            seen_ids.add(cid)
        ctype = cmd["command_type"]
        if ctype not in ALL_KNOWN_CMD_TYPES:
            warnings.append(
                f"commands[{idx}]: unknown command_type={ctype!r} — no fabric route will be emitted"
            )
        tid = cmd["tile_id"]
        if not isinstance(tid, int) or not (0 <= tid < tile_count):
            raise MappingError(
                f"commands[{idx}]: tile_id={tid!r} out of range [0, {tile_count})"
            )

    return warnings


# ---------------------------------------------------------------------------
# Placement report helpers
# ---------------------------------------------------------------------------

def _load_tensor_bytes(placement_path: str) -> dict[str | int, int]:
    """
    Parse placement report and return a mapping of tensor_id → total_bytes.
    Returns empty dict on any parse failure (placement is optional).
    """
    try:
        placement = _load_json(placement_path)
    except ParseError:
        return {}

    tensor_map: dict[str | int, int] = {}
    placements = placement.get("placements") or []
    for rec in placements:
        if not isinstance(rec, dict):
            continue
        tid = rec.get("tensor_id")
        tbytes = rec.get("total_bytes") or rec.get("packed_bytes") or 0
        if tid is not None and isinstance(tbytes, int) and tbytes > 0:
            tensor_map[tid] = tbytes
        tname = rec.get("tensor_name")
        if tname is not None and isinstance(tbytes, int) and tbytes > 0:
            tensor_map[tname] = tbytes
    return tensor_map


# ---------------------------------------------------------------------------
# Route generation
# ---------------------------------------------------------------------------

class RouteBuilder:
    """Stateful builder that accumulates route records and tile statistics."""

    def __init__(self, tile_count: int):
        self.tile_count = tile_count
        self._routes: list[dict[str, Any]] = []
        self._next_id = 1
        self._tile_out_packets: dict[int, int] = defaultdict(int)
        self._tile_in_packets: dict[int, int] = defaultdict(int)
        self._tile_out_bytes: dict[int, int] = defaultdict(int)
        self._tile_in_bytes: dict[int, int] = defaultdict(int)
        self._tile_reductions_started: dict[int, int] = defaultdict(int)
        self._tile_reductions_completed: dict[int, int] = defaultdict(int)
        self._tile_barriers_started: dict[int, int] = defaultdict(int)
        self._tile_barriers_completed: dict[int, int] = defaultdict(int)
        self.warnings: list[str] = []

    def _alloc_id(self) -> int:
        rid = self._next_id
        self._next_id += 1
        return rid

    def _add(
        self,
        route_type: str,
        source_tile: int,
        destination_tiles: list[int],
        payload_bytes: int,
        *,
        source_tensor: str | None = None,
        source_command_id: int | None = None,
        destination_tensor: str | None = None,
        destination_command_id: int | None = None,
        payload_type: str = "activation",
        dependency_fence: int = 0,
        reduction_id: int = 0,
        reduction_behavior: str = "none",
        ordering_policy: str = "ordered",
        trace_id: int = 0,
    ) -> None:
        rid = self._alloc_id()
        pce = 1 if payload_bytes > 0 else 0
        route: dict[str, Any] = {
            "route_id": rid,
            "route_type": route_type,
            "source_tile": source_tile,
            "destination_tiles": destination_tiles,
            "source_tensor": source_tensor,
            "source_command_id": source_command_id,
            "destination_tensor": destination_tensor,
            "destination_command_id": destination_command_id,
            "payload_type": payload_type,
            "payload_bytes": payload_bytes,
            "packet_count_estimate": pce,
            "dependency_fence": dependency_fence,
            "reduction_id": reduction_id,
            "reduction_behavior": reduction_behavior,
            "ordering_policy": ordering_policy,
            "trace_id": trace_id,
            "route_status": "ok",
        }
        self._routes.append(route)

        # Update per-tile statistics
        self._tile_out_packets[source_tile] += pce
        self._tile_out_bytes[source_tile] += payload_bytes
        for d in destination_tiles:
            self._tile_in_packets[d] += pce
            self._tile_in_bytes[d] += payload_bytes

    def add_control_ack(
        self,
        source_tile: int,
        source_command_id: int | None,
        fence_id: int,
    ) -> None:
        """Emit a CONTROL_ACK route: tile → all other tiles (host-visible)."""
        dests = [t for t in range(self.tile_count) if t != source_tile]
        if not dests:
            # Single-tile plan: loopback not allowed; emit with host as implicit dest
            # Represent as an empty destination list with a warning.
            self.warnings.append(
                f"CONTROL_ACK for command_id={source_command_id} on single-tile "
                f"plan has no remote destinations; route omitted"
            )
            return
        self._add(
            "CONTROL_ACK",
            source_tile,
            dests,
            CONTROL_PAYLOAD_BYTES,
            source_command_id=source_command_id,
            payload_type="control",
            dependency_fence=fence_id,
            ordering_policy="ordered",
        )

    def add_trace_event(
        self,
        source_tile: int,
        source_command_id: int | None,
        fence_id: int,
        trace_id: int,
    ) -> None:
        """Emit a TRACE_EVENT route: tile → all other tiles."""
        dests = [t for t in range(self.tile_count) if t != source_tile]
        if not dests:
            self.warnings.append(
                f"TRACE_EVENT for command_id={source_command_id} on single-tile "
                f"plan has no remote destinations; route omitted"
            )
            return
        self._add(
            "TRACE_EVENT",
            source_tile,
            dests,
            TRACE_PAYLOAD_BYTES,
            source_command_id=source_command_id,
            payload_type="trace_event",
            dependency_fence=fence_id,
            ordering_policy="ordered",
            trace_id=trace_id,
        )

    def add_tile_barrier(
        self,
        source_tile: int,
        source_command_id: int | None,
        fence_id: int,
    ) -> None:
        """Emit a TILE_BARRIER route: source tile → all tiles (including self)."""
        dests = list(range(self.tile_count))
        self._add(
            "TILE_BARRIER",
            source_tile,
            dests,
            0,
            source_command_id=source_command_id,
            payload_type="barrier_token",
            dependency_fence=fence_id,
            ordering_policy="barriered",
        )
        for t in range(self.tile_count):
            self._tile_barriers_started[t] += 1
            self._tile_barriers_completed[t] += 1

    def add_activation_send(
        self,
        source_tile: int,
        dest_tiles: list[int],
        payload_bytes: int,
        source_command_id: int | None,
        tensor_name: str | None,
        fence_id: int,
    ) -> None:
        self._add(
            "ACTIVATION_SEND",
            source_tile,
            dest_tiles,
            payload_bytes,
            source_tensor=tensor_name,
            source_command_id=source_command_id,
            payload_type="activation",
            dependency_fence=fence_id,
            ordering_policy="ordered",
        )

    def add_partial_reduce(
        self,
        source_tile: int,
        dest_tiles: list[int],
        payload_bytes: int,
        source_command_id: int | None,
        tensor_name: str | None,
        fence_id: int,
        reduction_id: int,
        reduction_behavior: str,
    ) -> None:
        self._tile_reductions_started[source_tile] += 1
        for d in dest_tiles:
            self._tile_reductions_completed[d] += 1
        self._add(
            "PARTIAL_REDUCE",
            source_tile,
            dest_tiles,
            payload_bytes,
            source_tensor=tensor_name,
            source_command_id=source_command_id,
            payload_type="partial_result",
            dependency_fence=fence_id,
            reduction_id=reduction_id,
            reduction_behavior=reduction_behavior,
            ordering_policy="ordered",
        )

    def tile_summaries(self) -> list[dict[str, Any]]:
        summaries = []
        for tid in range(self.tile_count):
            summaries.append({
                "tile_id": tid,
                "outbound_packet_count": self._tile_out_packets[tid],
                "inbound_packet_count": self._tile_in_packets[tid],
                "outbound_payload_bytes": self._tile_out_bytes[tid],
                "inbound_payload_bytes": self._tile_in_bytes[tid],
                "reductions_started": self._tile_reductions_started[tid],
                "reductions_completed": self._tile_reductions_completed[tid],
                "barriers_started": self._tile_barriers_started[tid],
                "barriers_completed": self._tile_barriers_completed[tid],
                "route_failures": 0,
                "estimated_bandwidth_gib_sec": None,
            })
        return summaries

    @property
    def routes(self) -> list[dict[str, Any]]:
        return list(self._routes)

    @property
    def route_count(self) -> int:
        return len(self._routes)

    @property
    def packet_count_estimate(self) -> int:
        return sum(r["packet_count_estimate"] for r in self._routes)

    @property
    def payload_bytes_estimate(self) -> int:
        return sum(r["payload_bytes"] for r in self._routes)


# ---------------------------------------------------------------------------
# Mapping logic
# ---------------------------------------------------------------------------

def _map_plan_to_routes(
    plan: dict[str, Any],
    tensor_bytes: dict[str | int, int],
    args: argparse.Namespace,
) -> RouteBuilder:
    """
    Walk command list and emit fabric route records.

    Returns a populated RouteBuilder.
    """
    header = plan["header"]
    tile_count: int = header["tile_count"]
    commands: list[dict[str, Any]] = plan["commands"]

    builder = RouteBuilder(tile_count)

    # Track per-tile FABRIC_REDUCE groups to assign deterministic reduction IDs.
    # Key: (tile_id, fence_id) → reduction_id counter.
    reduction_counter = 0

    for cmd in commands:
        ctype: str = cmd["command_type"]
        tile_id: int = cmd["tile_id"]
        command_id: int = cmd["command_id"]
        fence_id: int = cmd.get("fence_id", 0)
        tensor_id = cmd.get("tensor_id")
        tensor_name: str | None = cmd.get("tensor_name")

        if ctype in CMD_TYPES_LOCAL_ONLY:
            # No fabric route needed.
            continue

        elif ctype == "VALIDATE_TENSOR":
            builder.add_control_ack(tile_id, command_id, fence_id)

        elif ctype == "TILE_BARRIER":
            builder.add_tile_barrier(tile_id, command_id, fence_id)

        elif ctype == "QUERY_COUNTERS":
            builder.add_control_ack(tile_id, command_id, fence_id)

        elif ctype == "TRACE_SNAPSHOT":
            trace_id = command_id  # use command_id as deterministic trace_id
            builder.add_trace_event(tile_id, command_id, fence_id, trace_id)

        elif ctype == "FABRIC_SEND":
            # Resolve payload bytes: prefer command total_bytes, else placement lookup.
            payload_bytes: int = cmd.get("total_bytes") or 0
            if payload_bytes == 0 and tensor_id is not None:
                payload_bytes = tensor_bytes.get(tensor_id, 0)
            if payload_bytes == 0 and tensor_name is not None:
                payload_bytes = tensor_bytes.get(tensor_name, 0)
            if payload_bytes == 0:
                payload_bytes = DEFAULT_ACTIVATION_PAYLOAD_BYTES
                builder.warnings.append(
                    f"FABRIC_SEND command_id={command_id}: payload_bytes unknown; "
                    f"using default {DEFAULT_ACTIVATION_PAYLOAD_BYTES}"
                )
            # Destination: all other tiles (broadcast-like; realistic mappers
            # would use placement data, but we conservatively route to all others).
            dest_tiles = [t for t in range(tile_count) if t != tile_id]
            if not dest_tiles:
                builder.warnings.append(
                    f"FABRIC_SEND command_id={command_id}: no destination tiles "
                    f"(single-tile plan); route omitted"
                )
                continue
            builder.add_activation_send(
                tile_id, dest_tiles, payload_bytes, command_id, tensor_name, fence_id
            )

        elif ctype == "FABRIC_REDUCE":
            payload_bytes = cmd.get("total_bytes") or 0
            if payload_bytes == 0 and tensor_id is not None:
                payload_bytes = tensor_bytes.get(tensor_id, 0)
            if payload_bytes == 0 and tensor_name is not None:
                payload_bytes = tensor_bytes.get(tensor_name, 0)
            if payload_bytes == 0:
                payload_bytes = DEFAULT_ACTIVATION_PAYLOAD_BYTES
                builder.warnings.append(
                    f"FABRIC_REDUCE command_id={command_id}: payload_bytes unknown; "
                    f"using default {DEFAULT_ACTIVATION_PAYLOAD_BYTES}"
                )
            # reduction_behavior: read from command notes or default to "sum"
            notes: str = cmd.get("notes") or ""
            reduction_behavior = "sum"
            for rb in ("sum", "concat", "max", "topk"):
                if rb in notes:
                    reduction_behavior = rb
                    break
            dest_tiles = [t for t in range(tile_count) if t != tile_id]
            if not dest_tiles:
                builder.warnings.append(
                    f"FABRIC_REDUCE command_id={command_id}: no destination tiles "
                    f"(single-tile plan); route omitted"
                )
                continue
            reduction_counter += 1
            builder.add_partial_reduce(
                tile_id, dest_tiles, payload_bytes, command_id,
                tensor_name, fence_id, reduction_counter, reduction_behavior,
            )

        else:
            # Unknown type (already warned during validation; skip silently here).
            builder.warnings.append(
                f"command_id={command_id}: unrecognized command_type={ctype!r}; route omitted"
            )

    return builder


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------

def _assemble_report(
    plan: dict[str, Any],
    builder: RouteBuilder,
    plan_path: str,
    placement_path: str | None,
    args: argparse.Namespace,
    map_warnings: list[str],
) -> dict[str, Any]:
    header = plan["header"]
    model_name: str = header.get("model_name") or header.get("model_id") or "unknown"
    model_id: str = header.get("model_id") or "unknown"
    session_id: str = header.get("session_id") or "session_0"
    tile_count: int = header["tile_count"]

    report_warnings: list[dict[str, Any]] = []
    for w in map_warnings + builder.warnings:
        report_warnings.append({"severity": "warning", "message": w})

    report_status = "pass"
    if report_warnings and args.strict:
        report_status = "fail"
    elif report_warnings:
        report_status = "warn"

    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    fabric_policy = "layer_wise"

    report: dict[str, Any] = {
        "header": {
            "route_report_version": ROUTE_REPORT_VERSION,
            "source_placement_report": placement_path,
            "source_command_plan": plan_path,
            "model_name": model_name,
            "model_id": model_id,
            "session_id": session_id,
            "tile_count": tile_count,
            "route_count": builder.route_count,
            "packet_count_estimate": builder.packet_count_estimate,
            "payload_bytes_estimate": builder.payload_bytes_estimate,
            "fabric_policy": fabric_policy,
            "status": report_status,
            "report_timestamp": now_iso,
        },
        "routes": builder.routes,
        "tiles": builder.tile_summaries(),
        "warnings": report_warnings,
        "failures": [],
    }
    return report


# ---------------------------------------------------------------------------
# Human-readable output
# ---------------------------------------------------------------------------

def _print_summary(plan_path: str, report: dict[str, Any], map_warnings: list[str]) -> None:
    hdr = report.get("header") or {}
    print("ATT-1 fabric route mapper (M117)")
    print(f"  Plan path         : {plan_path}")
    print(f"  Model             : {hdr.get('model_id', '?')}")
    print(f"  Tile count        : {hdr.get('tile_count', '?')}")
    print(f"  Routes generated  : {hdr.get('route_count', 0)}")
    print(f"  Packets estimated : {hdr.get('packet_count_estimate', 0)}")
    print(f"  Payload bytes     : {hdr.get('payload_bytes_estimate', 0)}")
    print(f"  Warnings          : {len(report.get('warnings') or [])}")
    print(f"  Status            : {hdr.get('status', '?').upper()}")

    if report.get("warnings"):
        print("\nWarnings:")
        for w in report["warnings"]:
            print(f"  {w.get('message', '')}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = _parse_args()

    # Load command plan
    try:
        plan = _load_json(args.plan)
    except ParseError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    # Validate plan structure
    try:
        map_warnings = _validate_plan(plan, args.strict)
    except ParseError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except MappingError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    # Load optional placement report
    tensor_bytes: dict[str | int, int] = {}
    if args.placement_report:
        tensor_bytes = _load_tensor_bytes(args.placement_report)

    # Generate routes
    try:
        builder = _map_plan_to_routes(plan, tensor_bytes, args)
    except MappingError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    # Strict mode: fail if warnings accumulated
    if args.strict and (map_warnings or builder.warnings):
        print(
            f"ERROR: --strict: {len(map_warnings) + len(builder.warnings)} warning(s) produced",
            file=sys.stderr,
        )
        return 1

    # Assemble report
    report = _assemble_report(
        plan, builder, args.plan, args.placement_report, args, map_warnings
    )

    # Print human-readable summary
    _print_summary(args.plan, report, map_warnings)

    # Optionally write JSON
    if args.route_report_json:
        try:
            with open(args.route_report_json, "w", encoding="utf-8") as fh:
                json.dump(report, fh, indent=2)
                fh.write("\n")
            print(f"\nRoute report written to: {args.route_report_json}")
        except OSError as exc:
            print(f"ERROR: cannot write route report: {exc}", file=sys.stderr)
            return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
