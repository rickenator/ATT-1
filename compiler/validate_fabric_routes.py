#!/usr/bin/env python3
"""
ATT-1 AIMU fabric route report validator (Milestone 116).

Reads a fabric route report JSON produced according to the M115 schema
(docs/aimu_fabric_routing.md §11) and validates it for structural
correctness, route record integrity, per-tile traffic summary consistency,
reduction/barrier semantics, and dependency ordering.

This tool does NOT execute routes, change inference behavior, access real
PCIe/MMIO registers, or implement a kernel driver.

Exit codes:
  0 — validation passed (zero errors; warnings may be present)
  1 — validation failed (one or more errors detected)
  2 — report could not be parsed (malformed JSON or missing required fields)

Usage:

    python3 compiler/validate_fabric_routes.py \\
        --report compiler/fixtures/fabric_route_report_tiny.json

    python3 compiler/validate_fabric_routes.py \\
        --report build/my_route_report.json \\
        --report-json build/validation_result.json

    python3 compiler/validate_fabric_routes.py \\
        --report build/my_route_report.json \\
        --strict

In strict mode warnings are promoted to errors and a report status of
anything other than "pass" is also treated as an error.
"""

import argparse
import json
import math
import sys
from collections import defaultdict

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUPPORTED_ROUTE_REPORT_VERSIONS = {1}

VALID_ROUTE_TYPES = {
    "ACTIVATION_SEND",
    "ACTIVATION_BROADCAST",
    "PARTIAL_REDUCE",
    "LOGITS_REDUCE",
    "KV_TRANSFER",
    "TILE_BARRIER",
    "TRACE_EVENT",
    "CONTROL_ACK",
}

# Route types that must carry a nonzero payload
DATA_ROUTE_TYPES = {
    "ACTIVATION_SEND",
    "ACTIVATION_BROADCAST",
    "PARTIAL_REDUCE",
    "LOGITS_REDUCE",
    "KV_TRANSFER",
}

# Route types that require an explicit reduction_behavior (not "none")
REDUCTION_ROUTE_TYPES = {
    "PARTIAL_REDUCE",
    "LOGITS_REDUCE",
}

# Route types that must use "barriered" ordering_policy
BARRIER_ROUTE_TYPES = {"TILE_BARRIER"}

VALID_REDUCTION_BEHAVIORS = {"none", "sum", "concat", "max", "topk", "pass_through"}

# Reduction behaviors that are explicit (non-trivial) — required for
# reduction routes.
EXPLICIT_REDUCTION_BEHAVIORS = {"sum", "concat", "max", "topk"}

VALID_ORDERING_POLICIES = {"ordered", "unordered", "barriered"}

VALID_ROUTE_STATUSES = {"ok", "warn", "fail", "skipped"}

VALID_REPORT_STATUSES = {"pass", "warn", "fail"}

# Route types where source == destination tiles are allowed
# (e.g. TILE_BARRIER where the source tile also receives the barrier)
SELF_DEST_ALLOWED_TYPES = {
    "TILE_BARRIER",
    "TRACE_EVENT",
    "CONTROL_ACK",
}


# ---------------------------------------------------------------------------
# Result collector
# ---------------------------------------------------------------------------

class ValidationResult:
    def __init__(self, strict: bool = False):
        self.strict = strict
        self.warnings: list[dict] = []
        self.errors: list[dict] = []

    def warn(self, rule: str, message: str, **kw):
        rec = {"severity": "warning", "rule": rule, "message": message}
        rec.update(kw)
        self.warnings.append(rec)
        if self.strict:
            # Promote to error in strict mode
            self.errors.append(rec)

    def error(self, rule: str, message: str, **kw):
        rec = {"severity": "error", "rule": rule, "message": message}
        rec.update(kw)
        self.errors.append(rec)

    @property
    def passed(self) -> bool:
        return len(self.errors) == 0

    def summary(self, *, checked: dict | None = None) -> dict:
        return {
            "status": "pass" if self.passed else "fail",
            "total_warnings": len(self.warnings),
            "total_errors": len(self.errors),
            "warnings": self.warnings,
            "failures": self.errors,
            "checked": checked or {},
        }


# ---------------------------------------------------------------------------
# Header validation
# ---------------------------------------------------------------------------

def validate_header(report: dict, res: ValidationResult) -> tuple[int, str]:
    """Validate the report header.

    Returns (declared_tile_count, declared_status).
    """
    header = report.get("header")
    if header is None:
        res.error("MISSING_HEADER", "Missing top-level 'header' key")
        return 0, "unknown"

    # route_report_version
    ver = header.get("route_report_version")
    if ver is None:
        res.error("MISSING_VERSION", "header.route_report_version is missing")
    elif not isinstance(ver, int) or ver not in SUPPORTED_ROUTE_REPORT_VERSIONS:
        res.error(
            "UNSUPPORTED_VERSION",
            f"header.route_report_version={ver!r} is not supported "
            f"(supported: {sorted(SUPPORTED_ROUTE_REPORT_VERSIONS)})",
        )

    # tile_count
    tile_count = header.get("tile_count")
    if tile_count is None:
        res.error("MISSING_TILE_COUNT", "header.tile_count is missing")
        tile_count = 0
    elif not isinstance(tile_count, int) or tile_count <= 0:
        res.error(
            "INVALID_TILE_COUNT",
            f"header.tile_count={tile_count!r} must be a positive integer",
        )
        tile_count = 0

    # route_count
    route_count = header.get("route_count")
    if route_count is None:
        res.error("MISSING_ROUTE_COUNT", "header.route_count is missing")
    elif not isinstance(route_count, int) or route_count < 0:
        res.error(
            "INVALID_ROUTE_COUNT",
            f"header.route_count={route_count!r} must be a non-negative integer",
        )

    # packet_count_estimate
    pce = header.get("packet_count_estimate")
    if pce is not None and (not isinstance(pce, int) or pce < 0):
        res.error(
            "INVALID_PACKET_COUNT_ESTIMATE",
            f"header.packet_count_estimate={pce!r} must be non-negative",
        )

    # payload_bytes_estimate
    pbe = header.get("payload_bytes_estimate")
    if pbe is not None and (not isinstance(pbe, int) or pbe < 0):
        res.error(
            "INVALID_PAYLOAD_BYTES_ESTIMATE",
            f"header.payload_bytes_estimate={pbe!r} must be non-negative",
        )

    # status
    status = header.get("status")
    if status is None:
        res.error("MISSING_STATUS", "header.status is missing")
        status = "unknown"
    elif status not in VALID_REPORT_STATUSES:
        res.error(
            "INVALID_STATUS",
            f"header.status={status!r} is not recognized "
            f"(valid: {sorted(VALID_REPORT_STATUSES)})",
        )

    # strict mode: status != pass is an error
    if res.strict and status != "pass":
        res.error(
            "STRICT_STATUS_NOT_PASS",
            f"--strict: header.status={status!r} is not 'pass'",
        )

    return int(tile_count), str(status)


# ---------------------------------------------------------------------------
# Route record validation
# ---------------------------------------------------------------------------

def validate_routes(
    report: dict,
    tile_count: int,
    res: ValidationResult,
) -> list[dict]:
    """Validate all route records.  Returns the route list (may be empty)."""
    routes_raw = report.get("routes")
    if routes_raw is None:
        res.error("MISSING_ROUTES", "Missing top-level 'routes' key")
        return []
    if not isinstance(routes_raw, list):
        res.error("INVALID_ROUTES_TYPE", "'routes' must be a JSON array")
        return []

    seen_ids: set[int] = set()
    valid_routes: list[dict] = []

    for idx, route in enumerate(routes_raw):
        if not isinstance(route, dict):
            res.error("INVALID_ROUTE_ITEM", f"routes[{idx}] is not an object")
            continue

        ctx = f"routes[{idx}]"
        route_ok = True

        # route_id
        rid = route.get("route_id")
        if rid is None:
            res.error("MISSING_ROUTE_ID", f"{ctx}: route_id is missing")
            route_ok = False
        elif not isinstance(rid, int):
            res.error("INVALID_ROUTE_ID", f"{ctx}: route_id must be an integer, got {rid!r}")
            route_ok = False
        elif rid in seen_ids:
            res.error("DUPLICATE_ROUTE_ID", f"{ctx}: route_id={rid} is duplicated")
            route_ok = False
        else:
            seen_ids.add(rid)

        # route_type
        rtype = route.get("route_type")
        if rtype is None:
            res.error("MISSING_ROUTE_TYPE", f"{ctx}: route_type is missing")
            rtype = "__unknown__"
            route_ok = False
        elif rtype not in VALID_ROUTE_TYPES:
            res.error(
                "UNSUPPORTED_PACKET_TYPE",
                f"{ctx}: route_type={rtype!r} is not recognized "
                f"(valid: {sorted(VALID_ROUTE_TYPES)})",
            )
            route_ok = False

        # source_tile
        src = route.get("source_tile")
        if src is None:
            res.error("MISSING_SOURCE_TILE", f"{ctx}: source_tile is missing")
            src = -1
            route_ok = False
        elif not isinstance(src, int) or (tile_count > 0 and not (0 <= src < tile_count)):
            res.error(
                "INVALID_TILE_TARGET",
                f"{ctx}: source_tile={src!r} is out of range [0, {tile_count})",
            )
            route_ok = False

        # destination_tiles
        dests_raw = route.get("destination_tiles")
        # Allow missing or null only for types that explicitly permit no destination
        if dests_raw is None or dests_raw == []:
            if rtype not in {"TRACE_EVENT", "CONTROL_ACK"}:
                res.error(
                    "EMPTY_DESTINATION",
                    f"{ctx}: destination_tiles is missing or empty for route_type={rtype!r}",
                )
                route_ok = False
            dests = []
        else:
            if not isinstance(dests_raw, list):
                res.error("INVALID_DEST_TYPE", f"{ctx}: destination_tiles must be a JSON array")
                dests = []
                route_ok = False
            else:
                dests = dests_raw
                for di, d in enumerate(dests):
                    if not isinstance(d, int) or (tile_count > 0 and not (0 <= d < tile_count)):
                        res.error(
                            "INVALID_TILE_TARGET",
                            f"{ctx}: destination_tiles[{di}]={d!r} is out of range [0, {tile_count})",
                        )
                        route_ok = False
                # source == destination only for self-dest-allowed types
                if rtype not in SELF_DEST_ALLOWED_TYPES:
                    for d in dests:
                        if isinstance(src, int) and d == src:
                            res.error(
                                "SELF_ROUTE",
                                f"{ctx}: source_tile={src} appears in destination_tiles "
                                f"for route_type={rtype!r} (self-routing not permitted for this type)",
                            )
                            route_ok = False

        # payload_bytes
        pbytes = route.get("payload_bytes")
        if pbytes is None:
            res.error("MISSING_PAYLOAD_BYTES", f"{ctx}: payload_bytes is missing")
            route_ok = False
        elif not isinstance(pbytes, int) or pbytes < 0:
            res.error("INVALID_PAYLOAD_BYTES", f"{ctx}: payload_bytes must be a non-negative integer")
            route_ok = False
        elif rtype in DATA_ROUTE_TYPES and pbytes == 0:
            res.error(
                "ZERO_PAYLOAD",
                f"{ctx}: payload_bytes=0 for data route_type={rtype!r}",
            )
            route_ok = False

        # packet_count_estimate
        pce = route.get("packet_count_estimate")
        if pce is not None:
            if not isinstance(pce, int) or pce < 0:
                res.error("INVALID_ROUTE_PACKET_COUNT", f"{ctx}: packet_count_estimate must be non-negative")
                route_ok = False
            elif rtype in DATA_ROUTE_TYPES and pce == 0:
                res.error(
                    "ZERO_PACKET_COUNT",
                    f"{ctx}: packet_count_estimate=0 for data route_type={rtype!r}",
                )
                route_ok = False

        # reduction_behavior
        rb = route.get("reduction_behavior")
        if rb is None:
            res.error("MISSING_REDUCTION_BEHAVIOR", f"{ctx}: reduction_behavior is missing")
            route_ok = False
        elif rb not in VALID_REDUCTION_BEHAVIORS:
            res.error(
                "UNKNOWN_REDUCTION_BEHAVIOR",
                f"{ctx}: reduction_behavior={rb!r} is not recognized "
                f"(valid: {sorted(VALID_REDUCTION_BEHAVIORS)})",
            )
            route_ok = False
        elif rtype in REDUCTION_ROUTE_TYPES and rb not in EXPLICIT_REDUCTION_BEHAVIORS:
            res.error(
                "REDUCTION_NOT_EXPLICIT",
                f"{ctx}: route_type={rtype!r} requires an explicit reduction_behavior "
                f"(got {rb!r}; expected one of {sorted(EXPLICIT_REDUCTION_BEHAVIORS)})",
            )
            route_ok = False

        # reduction_id — required for reduction routes
        r_id = route.get("reduction_id")
        if rtype in REDUCTION_ROUTE_TYPES:
            if r_id is None:
                res.error("MISSING_REDUCTION_ID", f"{ctx}: reduction_id is required for {rtype!r}")
                route_ok = False
            elif not isinstance(r_id, int) or r_id < 0:
                res.error("INVALID_REDUCTION_ID", f"{ctx}: reduction_id must be a non-negative integer")
                route_ok = False

        # ordering_policy
        op = route.get("ordering_policy")
        if op is None:
            res.error("MISSING_ORDERING_POLICY", f"{ctx}: ordering_policy is missing")
            route_ok = False
        elif op not in VALID_ORDERING_POLICIES:
            res.error(
                "UNKNOWN_ORDERING_POLICY",
                f"{ctx}: ordering_policy={op!r} is not recognized "
                f"(valid: {sorted(VALID_ORDERING_POLICIES)})",
            )
            route_ok = False
        elif rtype in BARRIER_ROUTE_TYPES and op != "barriered":
            res.error(
                "BARRIER_ORDERING",
                f"{ctx}: route_type={rtype!r} must have ordering_policy='barriered', got {op!r}",
            )
            route_ok = False

        # route_status
        rs = route.get("route_status")
        if rs is None:
            res.error("MISSING_ROUTE_STATUS", f"{ctx}: route_status is missing")
            route_ok = False
        elif rs not in VALID_ROUTE_STATUSES:
            res.error(
                "UNKNOWN_ROUTE_STATUS",
                f"{ctx}: route_status={rs!r} is not recognized "
                f"(valid: {sorted(VALID_ROUTE_STATUSES)})",
            )
            route_ok = False

        if route_ok:
            valid_routes.append(route)

    return valid_routes


# ---------------------------------------------------------------------------
# Per-tile summary validation
# ---------------------------------------------------------------------------

def validate_tiles(
    report: dict,
    tile_count: int,
    res: ValidationResult,
) -> list[dict]:
    """Validate per-tile summary records.  Returns the tile list."""
    tiles_raw = report.get("tiles")
    if tiles_raw is None:
        res.error("MISSING_TILES", "Missing top-level 'tiles' key")
        return []
    if not isinstance(tiles_raw, list):
        res.error("INVALID_TILES_TYPE", "'tiles' must be a JSON array")
        return []

    seen_ids: set[int] = set()
    valid_tiles: list[dict] = []

    for idx, tile in enumerate(tiles_raw):
        if not isinstance(tile, dict):
            res.error("INVALID_TILE_ITEM", f"tiles[{idx}] is not an object")
            continue

        ctx = f"tiles[{idx}]"
        tile_ok = True

        # tile_id
        tid = tile.get("tile_id")
        if tid is None:
            res.error("MISSING_TILE_ID", f"{ctx}: tile_id is missing")
            tile_ok = False
        elif not isinstance(tid, int) or (tile_count > 0 and not (0 <= tid < tile_count)):
            res.error(
                "INVALID_TILE_ID",
                f"{ctx}: tile_id={tid!r} out of range [0, {tile_count})",
            )
            tile_ok = False
        elif tid in seen_ids:
            res.error("DUPLICATE_TILE_ID", f"{ctx}: tile_id={tid} is duplicated")
            tile_ok = False
        else:
            seen_ids.add(tid)

        non_neg_fields = [
            "outbound_packet_count", "inbound_packet_count",
            "outbound_payload_bytes", "inbound_payload_bytes",
            "reductions_started", "reductions_completed",
            "barriers_started", "barriers_completed",
            "route_failures",
        ]
        for field in non_neg_fields:
            val = tile.get(field)
            if val is None:
                res.error("MISSING_TILE_FIELD", f"{ctx}: {field} is missing")
                tile_ok = False
            elif not isinstance(val, int) or val < 0:
                res.error(
                    "INVALID_TILE_FIELD",
                    f"{ctx}: {field}={val!r} must be a non-negative integer",
                )
                tile_ok = False

        # estimated_bandwidth_gib_sec — optional, must be finite non-negative if present
        bw = tile.get("estimated_bandwidth_gib_sec")
        if bw is not None:
            if not isinstance(bw, (int, float)) or not math.isfinite(bw) or bw < 0:
                res.error(
                    "INVALID_BANDWIDTH",
                    f"{ctx}: estimated_bandwidth_gib_sec={bw!r} must be finite and non-negative",
                )
                tile_ok = False

        if tile_ok:
            valid_tiles.append(tile)

    return valid_tiles


# ---------------------------------------------------------------------------
# Consistency validation
# ---------------------------------------------------------------------------

def validate_consistency(
    report: dict,
    header_tile_count: int,
    routes: list[dict],
    tiles: list[dict],
    res: ValidationResult,
):
    """Cross-check header counts, route totals, per-tile summaries."""
    header = report.get("header") or {}

    # Header route_count vs actual route list length
    declared_route_count = header.get("route_count")
    actual_route_count = len(report.get("routes") or [])
    if isinstance(declared_route_count, int) and declared_route_count != actual_route_count:
        res.error(
            "ROUTE_COUNT_MISMATCH",
            f"header.route_count={declared_route_count} does not match "
            f"actual route list length={actual_route_count}",
        )

    # Per-tile inbound/outbound packet count consistency
    # Build expected per-tile outbound and inbound packet counts from routes
    tile_outbound_packets: dict[int, int] = defaultdict(int)
    tile_inbound_packets: dict[int, int] = defaultdict(int)
    tile_outbound_bytes: dict[int, int] = defaultdict(int)
    tile_inbound_bytes: dict[int, int] = defaultdict(int)

    for route in routes:
        src = route.get("source_tile")
        dests = route.get("destination_tiles") or []
        pce = route.get("packet_count_estimate", 0) or 0
        pbytes = route.get("payload_bytes", 0) or 0
        if isinstance(src, int) and src >= 0:
            tile_outbound_packets[src] += pce
            tile_outbound_bytes[src] += pbytes
        for d in dests:
            if isinstance(d, int) and d >= 0:
                tile_inbound_packets[d] += pce
                tile_inbound_bytes[d] += pbytes

    for tile in tiles:
        tid = tile.get("tile_id")
        if not isinstance(tid, int):
            continue

        reported_out = tile.get("outbound_packet_count", 0) or 0
        expected_out = tile_outbound_packets.get(tid, 0)
        if reported_out != expected_out:
            res.warn(
                "TILE_OUTBOUND_MISMATCH",
                f"tiles[tile_id={tid}]: outbound_packet_count={reported_out} "
                f"but sum from route records={expected_out}",
                tile_id=tid,
            )

        reported_in = tile.get("inbound_packet_count", 0) or 0
        expected_in = tile_inbound_packets.get(tid, 0)
        if reported_in != expected_in:
            res.warn(
                "TILE_INBOUND_MISMATCH",
                f"tiles[tile_id={tid}]: inbound_packet_count={reported_in} "
                f"but sum from route records={expected_in}",
                tile_id=tid,
            )

    # Warnings/failures lists structural check
    for key in ("warnings", "failures"):
        lst = report.get(key)
        if lst is None:
            # Not an error — lists are optional if empty
            continue
        if not isinstance(lst, list):
            res.error(
                "INVALID_DIAG_LIST",
                f"Top-level '{key}' must be a JSON array",
            )


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ATT-1 AIMU fabric route report validator (M116).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--report", required=True, metavar="PATH",
        help="Path to M115 fabric route report JSON.",
    )
    p.add_argument(
        "--report-json", metavar="PATH",
        help="Write JSON validation summary to PATH.",
    )
    p.add_argument(
        "--strict", action="store_true",
        help="Treat warnings as errors; fail if report status is not 'pass'.",
    )
    return p.parse_args()


def _print_summary(report_path: str, header: dict, res: ValidationResult):
    tile_count = header.get("tile_count", 0)
    route_count = len([])  # will be overridden below
    print(f"ATT-1 fabric route validator (M116)")
    print(f"  Report path : {report_path}")
    print(f"  Tile count  : {tile_count}")
    print(f"  Route count : {header.get('route_count', '?')}")
    print(f"  Warnings    : {len(res.warnings)}")
    print(f"  Errors      : {len(res.errors)}")
    print(f"  Status      : {'PASS' if res.passed else 'FAIL'}")

    if res.warnings:
        print("\nWarnings:")
        for w in res.warnings:
            print(f"  [{w['rule']}] {w['message']}")
    if res.errors:
        print("\nErrors:")
        for e in res.errors:
            print(f"  [{e['rule']}] {e['message']}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = _parse_args()

    # Parse JSON
    try:
        with open(args.report, encoding="utf-8") as fh:
            report = json.load(fh)
    except FileNotFoundError:
        print(f"ERROR: file not found: {args.report}", file=sys.stderr)
        return 2
    except json.JSONDecodeError as exc:
        print(f"ERROR: malformed JSON in {args.report}: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"ERROR: cannot read {args.report}: {exc}", file=sys.stderr)
        return 2

    if not isinstance(report, dict):
        print("ERROR: route report JSON root must be an object", file=sys.stderr)
        return 2

    res = ValidationResult(strict=args.strict)

    # Phase 1: header
    tile_count, _status = validate_header(report, res)

    # Phase 2: route records
    valid_routes = validate_routes(report, tile_count, res)

    # Phase 3: per-tile summaries
    valid_tiles = validate_tiles(report, tile_count, res)

    # Phase 4: cross-checks
    validate_consistency(report, tile_count, valid_routes, valid_tiles, res)

    # Output
    header = report.get("header") or {}
    _print_summary(args.report, header, res)

    checked = {
        "header": 1,
        "routes_checked": len(report.get("routes") or []),
        "tiles_checked": len(report.get("tiles") or []),
    }
    summary = res.summary(checked=checked)

    if args.report_json:
        try:
            with open(args.report_json, "w", encoding="utf-8") as fh:
                json.dump(summary, fh, indent=2)
                fh.write("\n")
        except OSError as exc:
            print(f"ERROR: cannot write report JSON {args.report_json}: {exc}", file=sys.stderr)
            return 2

    return 0 if res.passed else 1


if __name__ == "__main__":
    sys.exit(main())
