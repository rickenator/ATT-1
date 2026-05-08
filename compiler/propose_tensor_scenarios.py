#!/usr/bin/env python3
"""
ATT-1 tensor placement scenario proposal generator (Milestone 102).

Reads an M98/M100 placement report JSON and generates candidate
placement/capacity scenarios for AIMU tile planning.  Compares tile SKU
sizes, tile counts, context lengths, and session counts without changing
runtime execution.

Exit codes:
  0 — scenario generation complete; at least one PASS scenario found
  1 — no capacity-PASS scenario found among all candidates
  2 — report could not be parsed (malformed JSON, missing required fields,
      or file not found)

Usage:
    python3 compiler/propose_tensor_scenarios.py \\
        --report build/my_placement.json

    python3 compiler/propose_tensor_scenarios.py \\
        --report build/my_placement.json \\
        --tile-memory-gib 16,32,64,128 \\
        --tile-count 8,12,16 \\
        --context 4096,8192 \\
        --sessions 1,4 \\
        --report-json build/scenarios.json
"""

import argparse
import json
import math
import sys
from typing import Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MIB = 1 << 20
_GIB = 1 << 30

_CAPACITY_WARN_PCT  = 80.0
_CAPACITY_FAIL_PCT  = 100.0

# Default SKU tiers explored when no --tile-memory-gib is given
_DEFAULT_TILE_MEMORY_GIB = [16, 32, 64, 128]

# Maximum rows to display in the human-readable table
_TABLE_MAX_ROWS = 30


# ---------------------------------------------------------------------------
# Scenario data class
# ---------------------------------------------------------------------------

class Scenario:
    """One candidate placement scenario with computed capacity metrics."""

    def __init__(
        self,
        tile_count: int,
        tile_memory_gib: float,
        context_length: int,
        sessions: int,
        fabric_gib_sec: Optional[float],
        model_bytes_per_tile: int,
        kv_bytes_per_tile: int,
        estimated_fabric_gib_sec: Optional[float],
    ) -> None:
        self.tile_count           = tile_count
        self.tile_memory_gib      = tile_memory_gib
        self.context_length       = context_length
        self.sessions             = sessions
        self.fabric_gib_sec       = fabric_gib_sec
        self.model_bytes_per_tile = model_bytes_per_tile
        self.kv_bytes_per_tile    = kv_bytes_per_tile
        self.total_bytes_per_tile = model_bytes_per_tile + kv_bytes_per_tile

        tile_mem_bytes = int(tile_memory_gib * _GIB)
        if tile_mem_bytes > 0:
            self.utilization_percent = (
                self.total_bytes_per_tile / tile_mem_bytes * 100.0
            )
        else:
            self.utilization_percent = 0.0

        if self.utilization_percent > _CAPACITY_FAIL_PCT:
            self.capacity_status = "FAIL"
        elif self.utilization_percent > _CAPACITY_WARN_PCT:
            self.capacity_status = "WARN"
        else:
            self.capacity_status = "PASS"

        self.estimated_fabric_gib_sec = estimated_fabric_gib_sec

        if fabric_gib_sec is not None and estimated_fabric_gib_sec is not None:
            if estimated_fabric_gib_sec > fabric_gib_sec:
                self.fabric_status = "FAIL"
            elif estimated_fabric_gib_sec > fabric_gib_sec * 0.80:
                self.fabric_status = "WARN"
            else:
                self.fabric_status = "PASS"
        else:
            self.fabric_status = "UNKN"

        kv_frac = (
            kv_bytes_per_tile / tile_mem_bytes if tile_mem_bytes > 0 else 0.0
        )
        self.kv_fraction = kv_frac

        # Recommendation score: higher is better
        cap_score = {"PASS": 1000, "WARN": 500, "FAIL": 0}
        fab_score = {"PASS": 100,  "WARN": 50,  "FAIL": 0, "UNKN": 75}
        self.recommendation_score = (
            cap_score.get(self.capacity_status, 0)
            + fab_score.get(self.fabric_status, 0)
            - tile_count * 5           # prefer fewer tiles
            - tile_memory_gib * 2      # prefer smaller memory SKU
            - kv_frac * 50             # prefer lower KV pressure
        )

    def to_dict(self) -> dict:
        return {
            "tile_count":               self.tile_count,
            "tile_memory_gib":          self.tile_memory_gib,
            "context_length":           self.context_length,
            "sessions":                 self.sessions,
            "model_bytes_per_tile":     self.model_bytes_per_tile,
            "kv_bytes_per_tile":        self.kv_bytes_per_tile,
            "total_bytes_per_tile":     self.total_bytes_per_tile,
            "utilization_percent":      round(self.utilization_percent, 1),
            "capacity_status":          self.capacity_status,
            "estimated_fabric_gib_sec": (
                round(self.estimated_fabric_gib_sec, 2)
                if self.estimated_fabric_gib_sec is not None
                else None
            ),
            "fabric_status":            self.fabric_status,
            "kv_fraction":              round(self.kv_fraction, 4),
            "recommendation_score":     round(self.recommendation_score, 1),
        }


# ---------------------------------------------------------------------------
# Report parsing
# ---------------------------------------------------------------------------

def _parse_report(path: str) -> dict:
    """Load and minimally validate the placement report JSON.

    Exits 2 on any parse or I/O error, or if required top-level keys are
    missing.
    """
    try:
        with open(path) as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"error: cannot parse report: {exc}", file=sys.stderr)
        sys.exit(2)

    if not isinstance(data, dict):
        print("error: report root must be a JSON object", file=sys.stderr)
        sys.exit(2)

    for required in ("report_version", "header", "tiles"):
        if required not in data:
            print(
                f"error: missing '{required}' in report", file=sys.stderr
            )
            sys.exit(2)

    return data


# ---------------------------------------------------------------------------
# CLI list parsers
# ---------------------------------------------------------------------------

def _parse_list_int(s: str, name: str) -> list:
    """Parse a comma-separated string of positive ints.

    Exits 2 on error.
    """
    result = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            v = int(part)
            if v <= 0:
                raise ValueError("must be a positive integer")
            result.append(v)
        except ValueError as exc:
            print(
                f"error: invalid {name} value '{part}': {exc}",
                file=sys.stderr,
            )
            sys.exit(2)
    if not result:
        print(f"error: empty {name} list", file=sys.stderr)
        sys.exit(2)
    return result


def _parse_list_float(s: str, name: str) -> list:
    """Parse a comma-separated string of positive floats.

    Exits 2 on error.
    """
    result = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            v = float(part)
            if v <= 0:
                raise ValueError("must be a positive number")
            result.append(v)
        except ValueError as exc:
            print(
                f"error: invalid {name} value '{part}': {exc}",
                file=sys.stderr,
            )
            sys.exit(2)
    if not result:
        print(f"error: empty {name} list", file=sys.stderr)
        sys.exit(2)
    return result


# ---------------------------------------------------------------------------
# Default tile-count heuristic
# ---------------------------------------------------------------------------

def _default_tile_counts(data: dict, total_model_bytes: int) -> list:
    """Generate a compact set of tile counts to explore.

    Includes the original count plus power-of-two breakpoints and the
    minimum tiles required to keep each of the 32 GiB and 64 GiB SKUs
    under 80 % utilisation for the model weights alone.
    """
    header = data["header"]
    tiles  = data.get("tiles", [])
    orig   = int(header.get("tile_count", len(tiles)) or len(tiles) or 1)

    counts: set = {orig}

    # Standard power-of-two tile counts
    for base in (1, 2, 4, 8, 16, 32):
        counts.add(base)

    # Breakeven tile counts for 32 GiB and 64 GiB SKUs
    for gib in (32, 64):
        tile_budget = int(gib * _GIB * 0.80)
        if total_model_bytes > 0 and tile_budget > 0:
            min_tiles = max(1, math.ceil(total_model_bytes / tile_budget))
            counts.add(min_tiles)
            counts.add(min_tiles + 1)

    # Keep a manageable range
    filtered = sorted(c for c in counts if 1 <= c <= 128)
    return filtered[:12]


# ---------------------------------------------------------------------------
# Scenario generation
# ---------------------------------------------------------------------------

def _generate_scenarios(
    data: dict,
    tile_memory_gibs: list,
    tile_counts: list,
    contexts: list,
    sessions_list: list,
    fabric_gibs: list,
    tps: Optional[float],
) -> list:
    """Cross-product all parameter lists and return ranked Scenario objects."""
    header = data["header"]
    tiles  = data.get("tiles", [])

    orig_ctx      = int(header.get("target_context_length", 2048) or 2048)
    orig_sessions = int(header.get("target_sessions", 1) or 1)
    orig_fabric   = header.get("fabric_gib_sec")
    target_tps    = tps or float(header.get("target_tokens_per_sec") or 0)

    total_model_bytes       = sum(t.get("model_bytes", 0) for t in tiles)
    total_kv_bytes_orig     = sum(t.get("kv_bytes",    0) for t in tiles)
    total_fabric_bytes_orig = sum(
        t.get("fabric_payload_bytes_per_token", 0) for t in tiles
    )

    orig_ctx_safe      = max(orig_ctx, 1)
    orig_sessions_safe = max(orig_sessions, 1)

    scenarios = []
    seen: set = set()

    for tile_count in tile_counts:
        for tile_mem_gib in tile_memory_gibs:
            for ctx in contexts:
                for sess in sessions_list:
                    for raw_fab in fabric_gibs:
                        # Resolve fabric target for this scenario
                        if raw_fab is not None:
                            fab_target: Optional[float] = float(raw_fab)
                        elif orig_fabric is not None:
                            fab_target = float(orig_fabric)
                        else:
                            fab_target = None

                        # Deduplicate by scenario key
                        key = (tile_count, tile_mem_gib, ctx, sess, fab_target)
                        if key in seen:
                            continue
                        seen.add(key)

                        # Model bytes per tile: even layer-wise split
                        model_per_tile = math.ceil(
                            total_model_bytes / tile_count
                        )

                        # KV bytes scaled for new context/sessions, then split
                        kv_scaled = math.ceil(
                            total_kv_bytes_orig
                            * (ctx / orig_ctx_safe)
                            * (sess / orig_sessions_safe)
                        )
                        kv_per_tile = math.ceil(kv_scaled / tile_count)

                        # Estimated fabric bandwidth (tokens/sec × bytes/token)
                        est_fabric: Optional[float] = None
                        if target_tps > 0 and total_fabric_bytes_orig > 0:
                            orig_tiles = len(tiles) or 1
                            if orig_tiles > 1:
                                fab_scale = (tile_count - 1) / (orig_tiles - 1)
                            else:
                                fab_scale = float(tile_count)
                            est_fabric = (
                                total_fabric_bytes_orig
                                * fab_scale
                                * target_tps
                                / _GIB
                            )

                        sc = Scenario(
                            tile_count=tile_count,
                            tile_memory_gib=tile_mem_gib,
                            context_length=ctx,
                            sessions=sess,
                            fabric_gib_sec=fab_target,
                            model_bytes_per_tile=model_per_tile,
                            kv_bytes_per_tile=kv_per_tile,
                            estimated_fabric_gib_sec=est_fabric,
                        )
                        scenarios.append(sc)

    scenarios.sort(key=lambda s: -s.recommendation_score)
    return scenarios


# ---------------------------------------------------------------------------
# Human-readable output
# ---------------------------------------------------------------------------

def _render_text(
    data: dict,
    scenarios: list,
    recommended: Optional[Scenario],
) -> str:
    header  = data["header"]
    tiles   = data.get("tiles", [])
    orig_tc = int(header.get("tile_count", len(tiles)) or len(tiles) or 1)
    orig_tm = header.get("tile_memory_mib")
    orig_ctx  = int(header.get("target_context_length", 0) or 0)
    orig_sess = int(header.get("target_sessions", 1) or 1)
    dtype     = header.get("dtype", "?")
    model_name = header.get("model_name", "")

    total_model = sum(t.get("model_bytes", 0) for t in tiles)
    total_kv    = sum(t.get("kv_bytes",    0) for t in tiles)

    lines = []
    lines.append("=" * 72)
    lines.append(
        "ATT-1 Tensor Placement Scenario Comparison (Milestone 102)"
    )
    lines.append("=" * 72)
    lines.append("")
    lines.append("Original report summary:")
    if model_name:
        lines.append(f"  model_name       : {model_name}")
    lines.append(f"  dtype            : {dtype}")
    lines.append(f"  tile_count       : {orig_tc}")
    if orig_tm is not None:
        lines.append(
            f"  tile_memory_gib  : {orig_tm / 1024.0:.0f}"
            f"  ({orig_tm} MiB)"
        )
    lines.append(
        f"  total_model_bytes: {total_model:,}"
        f"  ({total_model / _GIB:.2f} GiB)"
    )
    lines.append(
        f"  total_kv_bytes   : {total_kv:,}"
        f"  ({total_kv / _GIB:.2f} GiB)"
    )
    lines.append(f"  context_length   : {orig_ctx}")
    lines.append(f"  sessions         : {orig_sess}")
    lines.append("")

    # Column header
    col = (
        f"{'tiles':>5}  {'mem_gib':>7}  {'ctx':>6}  {'sess':>4}  "
        f"{'mdl_mib':>8}  {'kv_mib':>7}  {'tot_mib':>8}  "
        f"{'util%':>6}  {'cap':>4}  {'fab':>4}  {'score':>7}"
    )
    lines.append(col)
    lines.append("-" * len(col))

    for sc in scenarios[:_TABLE_MAX_ROWS]:
        rec_mark = "  <-- recommended" if sc is recommended else ""
        row = (
            f"{sc.tile_count:>5}  {sc.tile_memory_gib:>7.0f}  "
            f"{sc.context_length:>6}  {sc.sessions:>4}  "
            f"{sc.model_bytes_per_tile // _MIB:>8}  "
            f"{sc.kv_bytes_per_tile // _MIB:>7}  "
            f"{sc.total_bytes_per_tile // _MIB:>8}  "
            f"{sc.utilization_percent:>6.1f}  {sc.capacity_status:>4}  "
            f"{sc.fabric_status:>4}  {sc.recommendation_score:>7.0f}"
            f"{rec_mark}"
        )
        lines.append(row)

    lines.append("")
    pass_count = sum(1 for s in scenarios if s.capacity_status == "PASS")
    total_count = len(scenarios)
    if pass_count == 0:
        lines.append(
            f"scenarios: 0 PASS capacity scenario(s) out of "
            f"{total_count} candidates"
        )
    else:
        lines.append(
            f"scenarios: {pass_count} PASS capacity scenario(s) "
            f"out of {total_count} candidates"
        )
    lines.append("")

    if recommended is not None:
        lines.append("Recommended scenario:")
        lines.append(f"  tile_count       : {recommended.tile_count}")
        lines.append(
            f"  tile_memory_gib  : {recommended.tile_memory_gib:.0f}"
        )
        lines.append(
            f"  context_length   : {recommended.context_length}"
        )
        lines.append(f"  sessions         : {recommended.sessions}")
        lines.append(
            f"  utilization      : {recommended.utilization_percent:.1f}%"
            f"  {recommended.capacity_status}"
        )
        lines.append(
            f"  kv_fraction      : "
            f"{recommended.kv_fraction * 100:.1f}% of tile memory"
        )
        if recommended.estimated_fabric_gib_sec is not None:
            lines.append(
                f"  est_fabric_gib_s : "
                f"{recommended.estimated_fabric_gib_sec:.2f}"
            )
    else:
        lines.append(
            "No PASS scenario found.  Consider larger tile memory or "
            "fewer context tokens."
        )

    lines.append("")
    lines.append("Notes:")
    lines.append(
        "  16 GiB tile : entry-level SKU; suits small models or q4 inference."
    )
    lines.append(
        "  32 GiB tile : balanced SKU; recommended for mid-range production."
    )
    lines.append(
        "  64 GiB tile : production SKU; accommodates large models and "
        "long contexts."
    )
    lines.append(
        "  More tiles vs larger memory: more tiles increase fabric traffic "
        "linearly."
    )
    lines.append(
        "  KV pressure : context_length x sessions x kv_bytes_per_token "
        "/ tile_memory."
    )
    lines.append(
        "  Fabric      : each additional tile boundary adds one activation "
        "broadcast per layer."
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="ATT-1 tensor placement scenario comparison tool (M102).",
    )
    ap.add_argument(
        "--report",
        required=True,
        help="Path to M98/M100 placement report JSON.",
    )
    ap.add_argument(
        "--tile-memory-gib",
        default=None,
        help=(
            "Comma-separated tile memory sizes in GiB to evaluate "
            "(e.g. 16,32,64,128).  Default: 16,32,64,128."
        ),
    )
    ap.add_argument(
        "--tile-count",
        default=None,
        help=(
            "Comma-separated tile counts to evaluate (e.g. 8,9,12,16).  "
            "Default: derived from report."
        ),
    )
    ap.add_argument(
        "--context",
        default=None,
        help=(
            "Comma-separated context lengths to evaluate.  "
            "Default: value from report."
        ),
    )
    ap.add_argument(
        "--sessions",
        default=None,
        help=(
            "Comma-separated session counts to evaluate.  "
            "Default: value from report."
        ),
    )
    ap.add_argument(
        "--target-tokens-per-sec",
        type=float,
        default=None,
        help="Target decode tokens/sec for fabric bandwidth estimate.",
    )
    ap.add_argument(
        "--fabric-gib-sec",
        default=None,
        help=(
            "Comma-separated fabric bandwidth targets in GiB/sec.  "
            "Default: value from report (if present)."
        ),
    )
    ap.add_argument(
        "--report-json",
        default=None,
        help="Optional path to write machine-readable JSON output.",
    )
    args = ap.parse_args()

    data   = _parse_report(args.report)
    header = data["header"]
    tiles  = data.get("tiles", [])

    total_model_bytes = sum(t.get("model_bytes", 0) for t in tiles)

    # ── Tile memory sizes ───────────────────────────────────────────────
    if args.tile_memory_gib:
        tile_memory_gibs = _parse_list_float(
            args.tile_memory_gib, "--tile-memory-gib"
        )
    else:
        tile_memory_gibs = list(_DEFAULT_TILE_MEMORY_GIB)

    # ── Tile counts ─────────────────────────────────────────────────────
    if args.tile_count:
        tile_counts = _parse_list_int(args.tile_count, "--tile-count")
    else:
        tile_counts = _default_tile_counts(data, total_model_bytes)

    # ── Context lengths ─────────────────────────────────────────────────
    orig_ctx = int(header.get("target_context_length", 2048) or 2048)
    if args.context:
        contexts = _parse_list_int(args.context, "--context")
    else:
        contexts = [orig_ctx]

    # ── Session counts ──────────────────────────────────────────────────
    orig_sessions = int(header.get("target_sessions", 1) or 1)
    if args.sessions:
        sessions_list = _parse_list_int(args.sessions, "--sessions")
    else:
        sessions_list = [orig_sessions]

    # ── Fabric bandwidth targets ────────────────────────────────────────
    if args.fabric_gib_sec:
        fabric_gibs = _parse_list_float(
            args.fabric_gib_sec, "--fabric-gib-sec"
        )
    else:
        orig_fab = header.get("fabric_gib_sec")
        fabric_gibs = [float(orig_fab) if orig_fab is not None else None]

    tps = args.target_tokens_per_sec

    # ── Generate and rank ───────────────────────────────────────────────
    scenarios = _generate_scenarios(
        data,
        tile_memory_gibs,
        tile_counts,
        contexts,
        sessions_list,
        fabric_gibs,
        tps,
    )

    pass_scenarios = [s for s in scenarios if s.capacity_status == "PASS"]
    warn_scenarios = [s for s in scenarios if s.capacity_status == "WARN"]
    recommended: Optional[Scenario] = (
        pass_scenarios[0]
        if pass_scenarios
        else (warn_scenarios[0] if warn_scenarios else None)
    )

    # ── Human-readable output ───────────────────────────────────────────
    print(_render_text(data, scenarios, recommended))

    # ── Optional JSON output ────────────────────────────────────────────
    if args.report_json:
        orig_tc  = int(
            header.get("tile_count", len(tiles)) or len(tiles) or 1
        )
        out = {
            "original": {
                "model_name":        header.get("model_name", ""),
                "dtype":             header.get("dtype", ""),
                "tile_count":        orig_tc,
                "tile_memory_mib":   header.get("tile_memory_mib"),
                "context_length":    orig_ctx,
                "sessions":          orig_sessions,
                "total_model_bytes": total_model_bytes,
                "total_kv_bytes":    sum(
                    t.get("kv_bytes", 0) for t in tiles
                ),
            },
            "scenarios":       [s.to_dict() for s in scenarios],
            "recommendation":  recommended.to_dict() if recommended else None,
            "pass_count":      len(pass_scenarios),
            "total_count":     len(scenarios),
        }
        try:
            with open(args.report_json, "w") as fh:
                json.dump(out, fh, indent=2)
        except OSError as exc:
            print(
                f"error: cannot write report JSON: {exc}", file=sys.stderr
            )
            sys.exit(2)

    # ── Exit code ───────────────────────────────────────────────────────
    if not pass_scenarios:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
