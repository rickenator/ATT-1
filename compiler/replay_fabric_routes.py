#!/usr/bin/env python3
"""
ATT-1 AIMU fabric route replay simulator (Milestone 123).

Reads an M115/M116/M117 fabric route report JSON, validates it using
M116 rules, replays routes in deterministic route_id order, accumulates
per-tile counters, and integrates M118 bandwidth/latency estimates.

This tool does NOT execute routes, change inference behavior, access real
PCIe/MMIO registers, or implement a kernel driver.  All estimates are
deterministic model projections derived from route metadata.

Exit codes:
  0 — replay passed (zero errors; warnings may be present)
  1 — replay failed (one or more errors, or --strict with warnings)
  2 — parse error (malformed JSON or missing required field)

Usage:

    python3 compiler/replay_fabric_routes.py \\
        --route-report compiler/fixtures/fabric_route_report_tiny.json

    python3 compiler/replay_fabric_routes.py \\
        --route-report build/routes.json \\
        --target-tokens-per-sec 50 \\
        --fabric-gib-sec 32 \\
        --report-json build/replay_result.json

    python3 compiler/replay_fabric_routes.py \\
        --route-report build/routes.json \\
        --strict

Pipeline position:
  command plan → route report → route replay (M123) → BW/latency estimate (M118)
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Import M116 validator and M118 BW simulator
# ---------------------------------------------------------------------------

_COMPILER_DIR = Path(__file__).parent
sys.path.insert(0, str(_COMPILER_DIR))

from validate_fabric_routes import (  # noqa: E402
    ValidationResult,
    validate_header,
    validate_routes,
    REDUCTION_ROUTE_TYPES,
    BARRIER_ROUTE_TYPES,
    DATA_ROUTE_TYPES,
    EXPLICIT_REDUCTION_BEHAVIORS,
    VALID_ROUTE_STATUSES,
)
from simulate_fabric_bandwidth import (  # noqa: E402
    _simulate_route as bw_simulate_route,
    _simulate_tiles as bw_simulate_tiles,
    _simulate_aggregate as bw_simulate_aggregate,
    AggregateResult,
    ParseError as BwParseError,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPLAY_VERSION: int = 1


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TileReplayCounters:
    tile_id: int
    packets_sent: int = 0
    packets_received: int = 0
    payload_bytes_sent: int = 0
    payload_bytes_received: int = 0
    reductions_started: int = 0
    reductions_completed: int = 0
    barriers_started: int = 0
    barriers_completed: int = 0
    trace_events: int = 0
    route_failures: int = 0


@dataclass
class ReplayResult:
    route_report_path: str
    route_count: int
    routes_replayed: int
    routes_failed: int
    tile_count: int
    tile_counters: list[TileReplayCounters]
    reductions_started: int
    reductions_completed: int
    barriers_started: int
    barriers_completed: int
    trace_events: int
    aggregate_packets_sent: int
    aggregate_packets_received: int
    aggregate_payload_bytes_sent: int
    aggregate_payload_bytes_received: int
    required_fabric_gib_sec: float | None
    fabric_status: str                          # PASS / WARN / FAIL / UNKNOWN
    status: str                                 # pass / warn / fail
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="ATT-1 AIMU fabric route replay simulator (M123)"
    )
    p.add_argument(
        "--route-report",
        metavar="PATH",
        required=True,
        help="Path to an M115/M116/M117 fabric route report JSON file",
    )
    p.add_argument(
        "--target-tokens-per-sec",
        metavar="N",
        type=float,
        default=1.0,
        help="Target decode token rate (tokens/s) for bandwidth estimates "
             "(default: 1.0)",
    )
    p.add_argument(
        "--fabric-gib-sec",
        metavar="N",
        type=float,
        default=None,
        help="Total fabric bandwidth target in GiB/s (optional; omit for UNKNOWN status)",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="Promote warnings to errors and fail on route_status != ok",
    )
    p.add_argument(
        "--report-json",
        metavar="PATH",
        help="Write replay result to a JSON file at this path",
    )
    return p


# ---------------------------------------------------------------------------
# Load report
# ---------------------------------------------------------------------------

def _load_report(path: str) -> dict:
    """Load and JSON-parse a route report file.  Raises SystemExit(2) on error."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError:
        print(f"error: route report not found: {path}", file=sys.stderr)
        sys.exit(2)
    except json.JSONDecodeError as exc:
        print(f"error: malformed JSON in {path}: {exc}", file=sys.stderr)
        sys.exit(2)


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------

def _replay_routes(
    report: dict,
    tile_count: int,
    strict: bool,
    notes: list[str],
) -> tuple[list[dict], list[TileReplayCounters], dict]:
    """Replay all routes in route_id order.

    Returns (replayed_routes, tile_counters, reduction_groups).
    """
    routes_raw: list[dict] = report.get("routes", [])

    # Sort by route_id (deterministic replay order)
    try:
        routes_sorted = sorted(routes_raw, key=lambda r: int(r.get("route_id", 0)))
    except (TypeError, ValueError):
        routes_sorted = routes_raw

    # Per-tile counters keyed by tile_id
    tile_ctrs: dict[int, TileReplayCounters] = {
        tid: TileReplayCounters(tile_id=tid) for tid in range(tile_count)
    }

    # Reduction tracking: reduction_id → list of routes processed
    reduction_groups: dict[int, list[dict]] = defaultdict(list)

    replayed: list[dict] = []

    for route in routes_sorted:
        rid = route.get("route_id", "?")
        rtype = route.get("route_type", "UNKNOWN")
        src = route.get("source_tile", -1)
        dests = route.get("destination_tiles", []) or []
        payload = int(route.get("payload_bytes", 0))
        pkt_est = int(route.get("packet_count_estimate", 0))
        rb = route.get("reduction_behavior", "none")
        red_id = route.get("reduction_id", 0)
        route_status = route.get("route_status", "ok")

        # Effective packet count for counter accumulation
        if rtype == "TILE_BARRIER":
            eff_pkts = 0
        elif pkt_est > 0:
            eff_pkts = pkt_est
        else:
            eff_pkts = 1

        # Route-status failure tracking
        is_failed = route_status not in ("ok", "warn", "skipped")
        if is_failed or (strict and route_status in ("warn", "fail")):
            if src in tile_ctrs:
                tile_ctrs[src].route_failures += 1
            notes.append(
                f"route_id={rid}: route_status={route_status!r}"
                + (" (strict)" if strict else "")
            )

        # Source tile: outbound
        if 0 <= src < tile_count:
            c = tile_ctrs[src]
            c.packets_sent += eff_pkts
            c.payload_bytes_sent += payload

            if rtype in BARRIER_ROUTE_TYPES:
                c.barriers_started += 1
                c.barriers_completed += 1
            elif rtype == "TRACE_EVENT":
                c.trace_events += 1
            elif rtype in REDUCTION_ROUTE_TYPES:
                c.reductions_started += 1

        # Destination tiles: inbound
        for dt in dests:
            if 0 <= dt < tile_count:
                c = tile_ctrs[dt]
                c.packets_received += eff_pkts
                c.payload_bytes_received += payload

                if rtype in BARRIER_ROUTE_TYPES:
                    # barriers_started/completed accounted on source side;
                    # dest tiles receive the barrier token
                    pass

        # Reduction group tracking
        if rtype in REDUCTION_ROUTE_TYPES:
            reduction_groups[red_id].append(route)

        replayed.append(route)

    # Mark reductions completed: when all routes with same reduction_id processed.
    # We count reductions_completed per tile by checking reduction group coverage.
    reduction_completed_ids: set[int] = set()
    for red_id, group in reduction_groups.items():
        # A reduction group is complete when at least one route in the group
        # has an explicit reduction_behavior — i.e., the reduce operation fired.
        has_explicit = any(
            r.get("reduction_behavior") in EXPLICIT_REDUCTION_BEHAVIORS
            for r in group
        )
        if has_explicit:
            reduction_completed_ids.add(red_id)

    # Distribute reductions_completed back to source tiles
    for red_id, group in reduction_groups.items():
        if red_id in reduction_completed_ids:
            for r in group:
                src = r.get("source_tile", -1)
                if 0 <= src < tile_count:
                    tile_ctrs[src].reductions_completed += 1

    return replayed, list(tile_ctrs.values()), dict(reduction_groups)


# ---------------------------------------------------------------------------
# BW simulation args proxy
# ---------------------------------------------------------------------------

class _BwArgs:
    """Minimal args-like object for M118 simulation functions."""

    def __init__(self, target_tokens_per_sec: float, fabric_gib_sec: float | None):
        self.target_tokens_per_sec = target_tokens_per_sec
        self.fabric_gib_sec = fabric_gib_sec
        # Use M118 defaults for packet overhead and latency parameters
        self.base_latency_ns: float = 100.0
        self.per_hop_latency_ns: float = 50.0
        self.packet_overhead_bytes: int = 64


# ---------------------------------------------------------------------------
# Report output helpers
# ---------------------------------------------------------------------------

def _print_text_report(result: ReplayResult) -> None:
    print("ATT-1 fabric route replay simulator (M123)")
    print("  NOTE: Deterministic replay — not hardware execution.")
    print(f"  route_report_path                : {result.route_report_path}")
    print(f"  route_count                      : {result.route_count}")
    print(f"  routes_replayed                  : {result.routes_replayed}")
    print(f"  routes_failed                    : {result.routes_failed}")
    print(f"  tile_count                       : {result.tile_count}")
    print(f"  aggregate_packets_sent           : {result.aggregate_packets_sent}")
    print(f"  aggregate_packets_received       : {result.aggregate_packets_received}")
    print(f"  aggregate_payload_bytes_sent     : {result.aggregate_payload_bytes_sent}")
    print(f"  aggregate_payload_bytes_received : {result.aggregate_payload_bytes_received}")
    print(f"  reductions_started               : {result.reductions_started}")
    print(f"  reductions_completed             : {result.reductions_completed}")
    print(f"  barriers_started                 : {result.barriers_started}")
    print(f"  barriers_completed               : {result.barriers_completed}")
    print(f"  trace_events                     : {result.trace_events}")
    if result.required_fabric_gib_sec is not None:
        print(f"  required_fabric_gib_sec          : {result.required_fabric_gib_sec:.6f}")
    else:
        print(f"  required_fabric_gib_sec          : (unavailable)")
    print(f"  fabric_status                    : {result.fabric_status}")
    print()

    # Per-tile counters
    print("  Per-tile replay counters:")
    for tc in result.tile_counters:
        print(
            f"    tile[{tc.tile_id}]  sent={tc.packets_sent}pkt/{tc.payload_bytes_sent}B"
            f"  recv={tc.packets_received}pkt/{tc.payload_bytes_received}B"
            f"  red={tc.reductions_started}/{tc.reductions_completed}"
            f"  bar={tc.barriers_started}/{tc.barriers_completed}"
            f"  trace={tc.trace_events}  fail={tc.route_failures}"
        )
    print()

    if result.notes:
        print("  Notes:")
        for note in result.notes:
            print(f"    - {note}")
        print()

    print(f"  status                           : {result.status}")


def _build_json_report(result: ReplayResult) -> dict:
    return {
        "replay_version": REPLAY_VERSION,
        "route_report_path": result.route_report_path,
        "route_count": result.route_count,
        "routes_replayed": result.routes_replayed,
        "routes_failed": result.routes_failed,
        "tile_count": result.tile_count,
        "aggregate_packets_sent": result.aggregate_packets_sent,
        "aggregate_packets_received": result.aggregate_packets_received,
        "aggregate_payload_bytes_sent": result.aggregate_payload_bytes_sent,
        "aggregate_payload_bytes_received": result.aggregate_payload_bytes_received,
        "reductions_started": result.reductions_started,
        "reductions_completed": result.reductions_completed,
        "barriers_started": result.barriers_started,
        "barriers_completed": result.barriers_completed,
        "trace_events": result.trace_events,
        "required_fabric_gib_sec": result.required_fabric_gib_sec,
        "fabric_status": result.fabric_status,
        "status": result.status,
        "notes": result.notes,
        "tiles": [
            {
                "tile_id": tc.tile_id,
                "packets_sent": tc.packets_sent,
                "packets_received": tc.packets_received,
                "payload_bytes_sent": tc.payload_bytes_sent,
                "payload_bytes_received": tc.payload_bytes_received,
                "reductions_started": tc.reductions_started,
                "reductions_completed": tc.reductions_completed,
                "barriers_started": tc.barriers_started,
                "barriers_completed": tc.barriers_completed,
                "trace_events": tc.trace_events,
                "route_failures": tc.route_failures,
            }
            for tc in result.tile_counters
        ],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    # ── Step 1: Load report ─────────────────────────────────────────────
    report = _load_report(args.route_report)

    # ── Step 2: Validate using M116 rules ───────────────────────────────
    val_res = ValidationResult(strict=args.strict)
    try:
        tile_count, header_status = validate_header(report, val_res)
        routes_validated = validate_routes(report, tile_count, val_res)
    except Exception as exc:
        print(f"error: validation failed unexpectedly: {exc}", file=sys.stderr)
        sys.exit(2)

    if not val_res.passed:
        print("replay: validation errors:", file=sys.stderr)
        for err in val_res.errors:
            print(f"  [{err['rule']}] {err['message']}", file=sys.stderr)
        sys.exit(1)

    notes: list[str] = []

    if val_res.warnings and not args.strict:
        for w in val_res.warnings:
            notes.append(f"WARNING [{w['rule']}]: {w['message']}")

    # ── Step 3: Replay routes ───────────────────────────────────────────
    replayed_routes, tile_counters, reduction_groups = _replay_routes(
        report, tile_count, args.strict, notes
    )

    # Aggregate counters
    agg_pkts_sent = sum(tc.packets_sent for tc in tile_counters)
    agg_pkts_recv = sum(tc.packets_received for tc in tile_counters)
    agg_bytes_sent = sum(tc.payload_bytes_sent for tc in tile_counters)
    agg_bytes_recv = sum(tc.payload_bytes_received for tc in tile_counters)
    reductions_started = sum(tc.reductions_started for tc in tile_counters)
    reductions_completed = sum(tc.reductions_completed for tc in tile_counters)
    barriers_started = sum(tc.barriers_started for tc in tile_counters)
    barriers_completed = sum(tc.barriers_completed for tc in tile_counters)
    trace_events = sum(tc.trace_events for tc in tile_counters)
    routes_failed = sum(tc.route_failures for tc in tile_counters)

    # ── Step 4: M118 bandwidth/latency integration ──────────────────────
    required_fabric_gib_sec: float | None = None
    fabric_status: str = "UNKNOWN"

    try:
        bw_args = _BwArgs(
            target_tokens_per_sec=args.target_tokens_per_sec,
            fabric_gib_sec=args.fabric_gib_sec,
        )
        route_results = [bw_simulate_route(r, bw_args) for r in replayed_routes]
        tile_results = bw_simulate_tiles(route_results, tile_count, bw_args)
        agg: AggregateResult = bw_simulate_aggregate(route_results, tile_results, bw_args)
        required_fabric_gib_sec = agg.required_fabric_gib_sec
        fabric_status = agg.status
    except (BwParseError, Exception) as exc:
        notes.append(f"BW simulation skipped: {exc}")

    # ── Step 5: Determine overall status ────────────────────────────────
    if routes_failed > 0:
        status = "fail"
    elif val_res.warnings:
        status = "warn"
    else:
        status = "pass"

    # ── Step 6: Build result ─────────────────────────────────────────────
    result = ReplayResult(
        route_report_path=args.route_report,
        route_count=len(replayed_routes),
        routes_replayed=len(replayed_routes),
        routes_failed=routes_failed,
        tile_count=tile_count,
        tile_counters=tile_counters,
        reductions_started=reductions_started,
        reductions_completed=reductions_completed,
        barriers_started=barriers_started,
        barriers_completed=barriers_completed,
        trace_events=trace_events,
        aggregate_packets_sent=agg_pkts_sent,
        aggregate_packets_received=agg_pkts_recv,
        aggregate_payload_bytes_sent=agg_bytes_sent,
        aggregate_payload_bytes_received=agg_bytes_recv,
        required_fabric_gib_sec=required_fabric_gib_sec,
        fabric_status=fabric_status,
        status=status,
        notes=notes,
    )

    # ── Step 7: Output ───────────────────────────────────────────────────
    _print_text_report(result)

    if args.report_json:
        report_dict = _build_json_report(result)
        try:
            with open(args.report_json, "w", encoding="utf-8") as fh:
                json.dump(report_dict, fh, indent=2)
                fh.write("\n")
        except OSError as exc:
            print(f"error: cannot write --report-json {args.report_json}: {exc}",
                  file=sys.stderr)
            sys.exit(1)

    # Exit code
    if status == "fail" or (args.strict and status == "warn"):
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
