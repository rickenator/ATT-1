#!/usr/bin/env python3
"""
ATT-1 AIMU fabric bandwidth and latency simulator (Milestone 118).

Reads an M115/M117 fabric route report JSON and produces per-route, per-tile,
and aggregate fabric bandwidth/latency pressure estimates.

This tool does NOT execute routes, change inference behavior, access real
PCIe/MMIO registers, or implement a kernel driver.  All estimates are
deterministic model projections derived from route metadata, not measured
hardware behavior.

Exit codes:
  0 — simulation complete (PASS or WARN; WARN + --strict exits 1)
  1 — FAIL status, or structural error, or WARN+strict
  2 — parse error (malformed JSON or missing required field)
  3 — invalid numeric flag value

Usage:

    python3 compiler/simulate_fabric_bandwidth.py \\
        --route-report compiler/fixtures/fabric_route_from_plan_tiny.json

    python3 compiler/simulate_fabric_bandwidth.py \\
        --route-report build/routes.json \\
        --target-tokens-per-sec 50 \\
        --fabric-gib-sec 32 \\
        --base-latency-ns 100 \\
        --per-hop-latency-ns 50 \\
        --packet-overhead-bytes 64 \\
        --report-json build/bw_sim.json \\
        --strict

Status rules:
  PASS    — required bandwidth <= 80% of fabric target
  WARN    — required bandwidth <= 100% of fabric target
  FAIL    — required bandwidth > fabric target
  UNKNOWN — no fabric target supplied (--fabric-gib-sec absent)

Latency model (deterministic estimate, not measured hardware latency):
  latency_ns = base_latency_ns + hops * per_hop_latency_ns
  hops = 0 for TILE_BARRIER and CONTROL_ACK (coordination/control routes)
  hops = 1 for all other route types (cross-tile data or trace)
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GIB: int = 1024 ** 3                    # bytes per GiB

DEFAULT_TARGET_TOKENS_PER_SEC: float = 1.0
DEFAULT_BASE_LATENCY_NS: float = 100.0
DEFAULT_PER_HOP_LATENCY_NS: float = 50.0
DEFAULT_PACKET_OVERHEAD_BYTES: int = 64

PASS_THRESHOLD: float = 0.80            # <= 80% utilisation → PASS
WARN_THRESHOLD: float = 1.00            # <= 100% → WARN, > 100% → FAIL

SIM_VERSION: int = 1

# Route types for which hops = 0 (local/control; no data traversal)
ZERO_HOP_ROUTE_TYPES: frozenset[str] = frozenset({
    "TILE_BARRIER",
    "CONTROL_ACK",
})

# Route types that carry real data payload (used for recommendation analysis)
DATA_ROUTE_TYPES: frozenset[str] = frozenset({
    "ACTIVATION_SEND",
    "ACTIVATION_BROADCAST",
    "PARTIAL_REDUCE",
    "LOGITS_REDUCE",
    "KV_TRANSFER",
})


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class ParseError(Exception):
    """Malformed JSON or missing required field."""


class SimulationError(Exception):
    """Structural error preventing simulation."""


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class RouteResult:
    route_id: int
    route_type: str
    source_tile: int
    destination_tiles: list
    effective_payload_bytes: int
    packet_count: int
    packet_overhead_bytes: int
    total_wire_bytes: int
    hops: int
    estimated_bandwidth_gib_sec: float
    estimated_latency_ns: float
    bandwidth_status: str               # PASS / WARN / FAIL / UNKNOWN


@dataclass
class TileResult:
    tile_id: int
    outbound_payload_bytes_per_token: int
    inbound_payload_bytes_per_token: int
    outbound_wire_bytes_per_token: int
    inbound_wire_bytes_per_token: int
    outbound_bandwidth_gib_sec: float
    inbound_bandwidth_gib_sec: float
    bottleneck_status: str              # PASS / WARN / FAIL / UNKNOWN


@dataclass
class AggregateResult:
    total_payload_bytes_per_token: int
    total_wire_bytes_per_token: int
    required_fabric_gib_sec: float
    fabric_gib_sec_target: float | None
    utilization_percent: float | None
    status: str                         # PASS / WARN / FAIL / UNKNOWN
    bottleneck_route_id: int | None
    bottleneck_tile_id: int | None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _required_field(obj: dict, field: str, context: str) -> Any:
    if field not in obj:
        raise ParseError(f"Missing required field '{field}' in {context}")
    return obj[field]


def _bandwidth_status(required_gib_sec: float, target: float | None) -> str:
    if target is None:
        return "UNKNOWN"
    if target <= 0:
        return "FAIL"
    util = required_gib_sec / target
    if util <= PASS_THRESHOLD:
        return "PASS"
    if util <= WARN_THRESHOLD:
        return "WARN"
    return "FAIL"


def _hops_for_route(route_type: str) -> int:
    return 0 if route_type in ZERO_HOP_ROUTE_TYPES else 1


# ---------------------------------------------------------------------------
# Route-level simulation
# ---------------------------------------------------------------------------

def _simulate_route(route: dict, args) -> RouteResult:
    rid = route.get("route_id", 0)
    rt = route.get("route_type", "UNKNOWN")
    src = route.get("source_tile", 0)
    dests = route.get("destination_tiles", [])
    payload = int(route.get("payload_bytes", 0))
    pkt_est = int(route.get("packet_count_estimate", 0))

    # Effective packet count for overhead calculation.
    # TILE_BARRIER routes carry no packets (barrier token only); for all other
    # routes with a zero estimate use 1 so overhead is still accounted for.
    if rt == "TILE_BARRIER":
        eff_pkts = 0
    elif pkt_est > 0:
        eff_pkts = pkt_est
    else:
        eff_pkts = 1

    overhead = args.packet_overhead_bytes * eff_pkts
    total_wire = payload + overhead
    hops = _hops_for_route(rt)

    # Bandwidth: wire bytes transferred per token × token rate
    bw = (total_wire * args.target_tokens_per_sec) / GIB

    # Latency: base + hop penalty (deterministic estimate only)
    latency_ns = args.base_latency_ns + hops * args.per_hop_latency_ns

    status = _bandwidth_status(bw, args.fabric_gib_sec)

    return RouteResult(
        route_id=rid,
        route_type=rt,
        source_tile=src,
        destination_tiles=dests,
        effective_payload_bytes=payload,
        packet_count=eff_pkts,
        packet_overhead_bytes=overhead,
        total_wire_bytes=total_wire,
        hops=hops,
        estimated_bandwidth_gib_sec=bw,
        estimated_latency_ns=latency_ns,
        bandwidth_status=status,
    )


# ---------------------------------------------------------------------------
# Per-tile simulation
# ---------------------------------------------------------------------------

def _simulate_tiles(
    route_results: list[RouteResult],
    tile_count: int,
    args,
) -> list[TileResult]:
    out_payload = [0] * tile_count
    in_payload = [0] * tile_count
    out_wire = [0] * tile_count
    in_wire = [0] * tile_count

    for rr in route_results:
        t = rr.source_tile
        if 0 <= t < tile_count:
            out_payload[t] += rr.effective_payload_bytes
            out_wire[t] += rr.total_wire_bytes
        for dt in rr.destination_tiles:
            if 0 <= dt < tile_count:
                in_payload[dt] += rr.effective_payload_bytes
                in_wire[dt] += rr.total_wire_bytes

    results: list[TileResult] = []
    for tid in range(tile_count):
        out_bw = (out_wire[tid] * args.target_tokens_per_sec) / GIB
        in_bw = (in_wire[tid] * args.target_tokens_per_sec) / GIB
        max_bw = max(out_bw, in_bw)
        status = _bandwidth_status(max_bw, args.fabric_gib_sec)
        results.append(TileResult(
            tile_id=tid,
            outbound_payload_bytes_per_token=out_payload[tid],
            inbound_payload_bytes_per_token=in_payload[tid],
            outbound_wire_bytes_per_token=out_wire[tid],
            inbound_wire_bytes_per_token=in_wire[tid],
            outbound_bandwidth_gib_sec=out_bw,
            inbound_bandwidth_gib_sec=in_bw,
            bottleneck_status=status,
        ))
    return results


# ---------------------------------------------------------------------------
# Aggregate simulation
# ---------------------------------------------------------------------------

def _simulate_aggregate(
    route_results: list[RouteResult],
    tile_results: list[TileResult],
    args,
) -> AggregateResult:
    total_payload = sum(rr.effective_payload_bytes for rr in route_results)
    total_wire = sum(rr.total_wire_bytes for rr in route_results)
    required_bw = (total_wire * args.target_tokens_per_sec) / GIB
    target = args.fabric_gib_sec
    util = (required_bw / target * 100.0) if (target is not None and target > 0) else None
    status = _bandwidth_status(required_bw, target)

    # Bottleneck route: highest individual bandwidth estimate
    bottleneck_route_id = None
    if route_results:
        br = max(route_results, key=lambda r: r.estimated_bandwidth_gib_sec)
        if br.estimated_bandwidth_gib_sec > 0.0:
            bottleneck_route_id = br.route_id

    # Bottleneck tile: highest max(out, in) bandwidth estimate
    bottleneck_tile_id = None
    if tile_results:
        bt = max(
            tile_results,
            key=lambda t: max(t.outbound_bandwidth_gib_sec, t.inbound_bandwidth_gib_sec),
        )
        if max(bt.outbound_bandwidth_gib_sec, bt.inbound_bandwidth_gib_sec) > 0.0:
            bottleneck_tile_id = bt.tile_id

    return AggregateResult(
        total_payload_bytes_per_token=total_payload,
        total_wire_bytes_per_token=total_wire,
        required_fabric_gib_sec=required_bw,
        fabric_gib_sec_target=target,
        utilization_percent=util,
        status=status,
        bottleneck_route_id=bottleneck_route_id,
        bottleneck_tile_id=bottleneck_tile_id,
    )


# ---------------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------------

def _recommendations(
    agg: AggregateResult,
    route_results: list[RouteResult],
) -> list[str]:
    recs: list[str] = []
    status = agg.status
    total_wire = agg.total_wire_bytes_per_token or 1

    if status in ("WARN", "FAIL"):
        recs.append(
            "Increase fabric bandwidth target (--fabric-gib-sec) to reduce utilization."
        )
    if status == "FAIL":
        recs.append(
            "Reduce target token rate (--target-tokens-per-sec) or offload more "
            "computation to local tile operations."
        )

    # Analyse traffic by route type
    type_wire: dict[str, int] = {}
    for rr in route_results:
        type_wire[rr.route_type] = type_wire.get(rr.route_type, 0) + rr.total_wire_bytes

    act_wire = type_wire.get("ACTIVATION_SEND", 0) + type_wire.get("ACTIVATION_BROADCAST", 0)
    logits_wire = type_wire.get("LOGITS_REDUCE", 0)
    kv_wire = type_wire.get("KV_TRANSFER", 0)
    reduce_wire = type_wire.get("PARTIAL_REDUCE", 0) + logits_wire

    if act_wire > 0.5 * total_wire:
        recs.append(
            "Activation routes dominate traffic; reduce cross-tile placement or "
            "split/replicate tensors differently."
        )
        recs.append(
            "Prefer head-wise placement for attention to reduce activation fan-out."
        )

    if logits_wire > 0.3 * total_wire:
        recs.append(
            "Logits reduction routes dominate; consider splitting lm_head/logits "
            "reduction across fewer tiles."
        )

    if kv_wire > 0.3 * total_wire:
        recs.append(
            "KV cache routes dominate; reduce context/session pressure or "
            "co-locate KV shards with their consumers."
        )

    if not recs:
        recs.append("Fabric utilization is within acceptable bounds.")

    return recs


# ---------------------------------------------------------------------------
# Human-readable output
# ---------------------------------------------------------------------------

def _print_report(
    report_path: str,
    report: dict,
    route_results: list[RouteResult],
    tile_results: list[TileResult],
    agg: AggregateResult,
    warnings: list[str],
    args,
) -> None:
    tile_count = report.get("header", {}).get("tile_count", "?")
    print("ATT-1 fabric bandwidth/latency simulator (M118)")
    print("  NOTE: All estimates are deterministic model projections, "
          "not measured hardware behavior.")
    print(f"  Report path   : {report_path}")
    print(f"  Tile count    : {tile_count}")
    print(f"  Route count   : {len(route_results)}")
    print(f"  Tokens/sec    : {args.target_tokens_per_sec}")
    if args.fabric_gib_sec is not None:
        print(f"  Fabric target : {args.fabric_gib_sec} GiB/s")
    else:
        print(f"  Fabric target : (not specified — status will be UNKNOWN)")
    print()

    # Per-route summary
    print("  Route traffic estimates:")
    for rr in route_results:
        dests = ",".join(str(d) for d in rr.destination_tiles)
        print(
            f"    route {rr.route_id:3d}  {rr.route_type:<22s}"
            f"  tile {rr.source_tile}→[{dests}]"
            f"  wire={rr.total_wire_bytes:6d}B"
            f"  bw={rr.estimated_bandwidth_gib_sec:.4e} GiB/s"
            f"  lat={rr.estimated_latency_ns:.0f}ns"
            f"  [{rr.bandwidth_status}]"
        )
    print()

    # Per-tile summary
    print("  Per-tile traffic:")
    for tr in tile_results:
        print(
            f"    tile {tr.tile_id}"
            f"  out={tr.outbound_bandwidth_gib_sec:.4e} GiB/s"
            f"  in={tr.inbound_bandwidth_gib_sec:.4e} GiB/s"
            f"  [{tr.bottleneck_status}]"
        )
    print()

    # Aggregate summary
    print("  Aggregate fabric:")
    print(f"    total payload/token  : {agg.total_payload_bytes_per_token} bytes")
    print(f"    total wire/token     : {agg.total_wire_bytes_per_token} bytes")
    print(f"    required bandwidth   : {agg.required_fabric_gib_sec:.4e} GiB/s")
    if agg.fabric_gib_sec_target is not None:
        print(f"    fabric target        : {agg.fabric_gib_sec_target} GiB/s")
        util_str = f"{agg.utilization_percent:.1f}%" if agg.utilization_percent is not None else "n/a"
        print(f"    utilization          : {util_str}")
    if agg.bottleneck_route_id is not None:
        print(f"    bottleneck route     : route {agg.bottleneck_route_id}")
    if agg.bottleneck_tile_id is not None:
        print(f"    bottleneck tile      : tile {agg.bottleneck_tile_id}")
    print(f"    status               : {agg.status}")
    print()

    # Warnings
    if warnings:
        print("  Warnings:")
        for w in warnings:
            print(f"    - {w}")
        print()

    # Recommendations
    recs = _recommendations(agg, route_results)
    print("  Recommendations:")
    for r in recs:
        print(f"    - {r}")
    print()


# ---------------------------------------------------------------------------
# JSON report assembly
# ---------------------------------------------------------------------------

def _build_json_report(
    report_path: str,
    report: dict,
    route_results: list[RouteResult],
    tile_results: list[TileResult],
    agg: AggregateResult,
    warnings: list[str],
    failures: list[str],
    args,
) -> dict:
    header = report.get("header", {})
    return {
        "header": {
            "sim_version": SIM_VERSION,
            "route_report_version": header.get("route_report_version"),
            "source_route_report": report_path,
            "tile_count": header.get("tile_count"),
            "route_count": len(route_results),
            "target_tokens_per_sec": args.target_tokens_per_sec,
            "fabric_gib_sec_target": args.fabric_gib_sec,
            "base_latency_ns": args.base_latency_ns,
            "per_hop_latency_ns": args.per_hop_latency_ns,
            "packet_overhead_bytes": args.packet_overhead_bytes,
            "status": agg.status,
        },
        "routes": [
            {
                "route_id": rr.route_id,
                "route_type": rr.route_type,
                "source_tile": rr.source_tile,
                "destination_tiles": rr.destination_tiles,
                "effective_payload_bytes": rr.effective_payload_bytes,
                "packet_count": rr.packet_count,
                "packet_overhead_bytes": rr.packet_overhead_bytes,
                "total_wire_bytes": rr.total_wire_bytes,
                "hops": rr.hops,
                "estimated_bandwidth_gib_sec": rr.estimated_bandwidth_gib_sec,
                "estimated_latency_ns": rr.estimated_latency_ns,
                "bandwidth_status": rr.bandwidth_status,
            }
            for rr in route_results
        ],
        "tiles": [
            {
                "tile_id": tr.tile_id,
                "outbound_payload_bytes_per_token": tr.outbound_payload_bytes_per_token,
                "inbound_payload_bytes_per_token": tr.inbound_payload_bytes_per_token,
                "outbound_wire_bytes_per_token": tr.outbound_wire_bytes_per_token,
                "inbound_wire_bytes_per_token": tr.inbound_wire_bytes_per_token,
                "outbound_bandwidth_gib_sec": tr.outbound_bandwidth_gib_sec,
                "inbound_bandwidth_gib_sec": tr.inbound_bandwidth_gib_sec,
                "bottleneck_status": tr.bottleneck_status,
            }
            for tr in tile_results
        ],
        "aggregate": {
            "total_payload_bytes_per_token": agg.total_payload_bytes_per_token,
            "total_wire_bytes_per_token": agg.total_wire_bytes_per_token,
            "required_fabric_gib_sec": agg.required_fabric_gib_sec,
            "fabric_gib_sec_target": agg.fabric_gib_sec_target,
            "utilization_percent": agg.utilization_percent,
            "status": agg.status,
            "bottleneck_route_id": agg.bottleneck_route_id,
            "bottleneck_tile_id": agg.bottleneck_tile_id,
        },
        "warnings": warnings,
        "failures": failures,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    ap = argparse.ArgumentParser(
        description="ATT-1 AIMU fabric bandwidth and latency simulator (M118)",
    )
    ap.add_argument(
        "--route-report", required=True, metavar="PATH",
        help="M115/M117 fabric route report JSON to simulate",
    )
    ap.add_argument(
        "--target-tokens-per-sec", type=float,
        default=DEFAULT_TARGET_TOKENS_PER_SEC, metavar="N",
        help="Inference token rate target in tokens/s (default: 1)",
    )
    ap.add_argument(
        "--fabric-gib-sec", type=float, default=None, metavar="N",
        help="Fabric bandwidth budget in GiB/s; absent → status UNKNOWN",
    )
    ap.add_argument(
        "--base-latency-ns", type=float,
        default=DEFAULT_BASE_LATENCY_NS, metavar="N",
        help="Base per-route latency estimate in ns (default: 100)",
    )
    ap.add_argument(
        "--per-hop-latency-ns", type=float,
        default=DEFAULT_PER_HOP_LATENCY_NS, metavar="N",
        help="Additional latency per fabric hop in ns (default: 50)",
    )
    ap.add_argument(
        "--packet-overhead-bytes", type=int,
        default=DEFAULT_PACKET_OVERHEAD_BYTES, metavar="N",
        help="Per-packet header overhead in bytes (default: 64)",
    )
    ap.add_argument(
        "--report-json", metavar="PATH",
        help="Write JSON simulation report to PATH",
    )
    ap.add_argument(
        "--strict", action="store_true",
        help="Exit nonzero on WARN or FAIL status (default: only FAIL exits nonzero)",
    )
    return ap.parse_args()


def _validate_args(args) -> list[str]:
    errors: list[str] = []
    if args.target_tokens_per_sec <= 0:
        errors.append("--target-tokens-per-sec must be > 0")
    if args.fabric_gib_sec is not None and args.fabric_gib_sec <= 0:
        errors.append("--fabric-gib-sec must be > 0")
    if args.base_latency_ns < 0:
        errors.append("--base-latency-ns must be >= 0")
    if args.per_hop_latency_ns < 0:
        errors.append("--per-hop-latency-ns must be >= 0")
    if args.packet_overhead_bytes < 0:
        errors.append("--packet-overhead-bytes must be >= 0")
    return errors


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = _parse_args()

    # Validate numeric flags first
    arg_errors = _validate_args(args)
    if arg_errors:
        for e in arg_errors:
            print(f"error: {e}", file=sys.stderr)
        return 3

    # Load route report
    try:
        with open(args.route_report) as fh:
            report = json.load(fh)
    except FileNotFoundError:
        print(f"parse error: route report not found: {args.route_report}", file=sys.stderr)
        return 2
    except json.JSONDecodeError as exc:
        print(f"parse error: malformed JSON in {args.route_report}: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"parse error: cannot open {args.route_report}: {exc}", file=sys.stderr)
        return 2

    if not isinstance(report, dict):
        print("parse error: route report must be a JSON object", file=sys.stderr)
        return 2

    # Extract required top-level fields
    try:
        header = _required_field(report, "header", "report root")
        tile_count = _required_field(header, "tile_count", "header")
        routes_raw = _required_field(report, "routes", "report root")
    except ParseError as exc:
        print(f"parse error: {exc}", file=sys.stderr)
        return 2

    if not isinstance(tile_count, int) or tile_count < 1:
        print("parse error: header.tile_count must be a positive integer", file=sys.stderr)
        return 2

    if not isinstance(routes_raw, list):
        print("parse error: 'routes' must be a JSON array", file=sys.stderr)
        return 2

    # Simulate each route
    warnings: list[str] = []
    failures: list[str] = []
    route_results: list[RouteResult] = []

    for i, route in enumerate(routes_raw):
        if not isinstance(route, dict):
            warnings.append(f"Skipping non-object item at routes[{i}]")
            continue
        rr = _simulate_route(route, args)
        route_results.append(rr)

    # Per-tile and aggregate simulation
    tile_results = _simulate_tiles(route_results, tile_count, args)
    agg = _simulate_aggregate(route_results, tile_results, args)

    # Collect per-route and per-tile diagnostic messages
    for rr in route_results:
        if rr.bandwidth_status == "WARN":
            warnings.append(
                f"route {rr.route_id} ({rr.route_type}): "
                f"estimated bandwidth {rr.estimated_bandwidth_gib_sec:.4e} GiB/s "
                f"exceeds 80% of fabric target"
            )
        elif rr.bandwidth_status == "FAIL":
            failures.append(
                f"route {rr.route_id} ({rr.route_type}): "
                f"estimated bandwidth {rr.estimated_bandwidth_gib_sec:.4e} GiB/s "
                f"exceeds fabric target"
            )

    for tr in tile_results:
        max_bw = max(tr.outbound_bandwidth_gib_sec, tr.inbound_bandwidth_gib_sec)
        if tr.bottleneck_status == "WARN":
            warnings.append(
                f"tile {tr.tile_id}: estimated bandwidth {max_bw:.4e} GiB/s "
                f"exceeds 80% of fabric target"
            )
        elif tr.bottleneck_status == "FAIL":
            failures.append(
                f"tile {tr.tile_id}: estimated bandwidth {max_bw:.4e} GiB/s "
                f"exceeds fabric target"
            )

    # Print human-readable report
    _print_report(
        args.route_report, report, route_results, tile_results, agg, warnings, args,
    )

    # Write optional JSON report
    if args.report_json:
        json_report = _build_json_report(
            args.route_report, report, route_results, tile_results,
            agg, warnings, failures, args,
        )
        try:
            with open(args.report_json, "w") as fh:
                json.dump(json_report, fh, indent=2)
            print(f"JSON report written to: {args.report_json}")
        except OSError as exc:
            print(f"error writing JSON report: {exc}", file=sys.stderr)
            return 1

    # Determine exit code
    if agg.status == "FAIL":
        return 1
    if agg.status == "WARN" and args.strict:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
