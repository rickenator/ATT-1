#!/usr/bin/env python3
"""
ATT-1 tensor-level placement proposal tool (Milestone 101).

Reads an M98/M100 placement report JSON and produces an advisory remediation
report for any failed or inefficient placements.  Does not modify the report
or the .att1 binary artifact.

Exit codes:
  0 — analysis complete; advisory status is "ok" or "warn"
  1 — analysis complete; advisory status is "fail" (actionable failures found)
  2 — report could not be parsed (malformed JSON, missing required fields,
      or file not found)

Usage:

    python3 compiler/propose_tensor_placement.py \\
        --report build/my_placement.json

    python3 compiler/propose_tensor_placement.py \\
        --report build/my_placement.json \\
        --report-json build/advisory.json

Advisory status levels:

  ok   — no capacity/bandwidth FAIL, no major remediation required.
  warn — one or more tiles WARN but none FAIL; minor remediation suggested.
  fail — one or more tiles FAIL or a critical issue detected; remediation
         required before deployment.
"""

import argparse
import json
import math
import sys

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_KV_PRESSURE_WARN_FRAC = 0.20   # KV bytes > 20% of tile memory → warn
_KV_PRESSURE_FAIL_FRAC = 0.40   # KV bytes > 40% of tile memory → fail
_IMBALANCE_WARN_FRAC   = 0.20   # tile util spread > 20% of mean → warn
_MIB = 1 << 20                   # bytes per MiB


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

class Proposal:
    """One remediation proposal."""

    def __init__(self, trigger: str, action: str, detail: str = "") -> None:
        self.trigger = trigger
        self.action  = action
        self.detail  = detail

    def to_dict(self) -> dict:
        return {"trigger": self.trigger, "action": self.action, "detail": self.detail}

    def __str__(self) -> str:
        suffix = f": {self.detail}" if self.detail else "."
        return f"  [{self.trigger}]  {self.action}{suffix}"


class AdvisoryReport:
    """Accumulated advisory report."""

    _ORDER: dict[str, int] = {"ok": 0, "warn": 1, "fail": 2}

    def __init__(self) -> None:
        self.status: str = "ok"
        self.proposals: list[Proposal] = []
        self.analysis: dict = {}
        self.next_action: str = ""

    def _elevate(self, level: str) -> None:
        if self._ORDER.get(level, 0) > self._ORDER.get(self.status, 0):
            self.status = level

    def add(self, trigger: str, action: str, detail: str = "",
            severity: str = "warn") -> None:
        self.proposals.append(Proposal(trigger, action, detail))
        self._elevate(severity)

    def to_dict(self) -> dict:
        return {
            "status":         self.status,
            "proposal_count": len(self.proposals),
            "next_action":    self.next_action,
            "analysis":       self.analysis,
            "proposals":      [p.to_dict() for p in self.proposals],
        }

    def render_text(self) -> str:
        lines: list[str] = [
            f"advisory: {self.status}",
            f"proposal_count: {len(self.proposals)}",
        ]
        if self.next_action:
            lines.append(f"next_action: {self.next_action}")
        if self.proposals:
            lines.append("")
            lines.append("Proposals:")
            for p in self.proposals:
                lines.append(str(p))
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Report parsing
# ---------------------------------------------------------------------------

def _parse_report(path: str) -> dict:
    """Load and minimally validate the placement report JSON.

    Calls sys.exit(2) on any parse/structure error.
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
            print(f"error: missing '{required}' in report", file=sys.stderr)
            sys.exit(2)

    return data


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def _analyze(data: dict) -> AdvisoryReport:
    rpt     = AdvisoryReport()
    header  = data.get("header", {})
    tiles   = data.get("tiles",   [])
    tensors = data.get("tensors", [])
    ex_warn = data.get("warnings",    [])
    ex_fail = data.get("failures",    [])

    # Header fields ----------------------------------------------------------
    dtype        = header.get("dtype", "f32")
    tile_count   = int(header.get("tile_count", len(tiles)) or 1)
    tile_mem_mib = header.get("tile_memory_mib")        # None when unset
    ctx_len      = int(header.get("target_context_length", 0) or 0)
    sessions     = int(header.get("target_sessions", 1) or 1)
    fabric_gib   = header.get("fabric_gib_sec")

    tile_mem_bytes = int(tile_mem_mib * _MIB) if tile_mem_mib else 0

    # Tile analysis ----------------------------------------------------------
    fail_tiles    = [t for t in tiles if t.get("capacity_status")  == "FAIL"]
    warn_tiles    = [t for t in tiles if t.get("capacity_status")  == "WARN"]
    bw_fail_tiles = [t for t in tiles if t.get("bandwidth_status") == "FAIL"]
    bw_warn_tiles = [t for t in tiles if t.get("bandwidth_status") == "WARN"]

    total_model_bytes = sum(t.get("model_bytes", 0) for t in tiles)
    total_kv_bytes    = sum(t.get("kv_bytes",    0) for t in tiles)

    utils = [
        t.get("memory_utilization_percent", 0.0)
        for t in tiles
        if t.get("capacity_status") != "UNKNOWN"
    ]
    max_util  = max(utils) if utils else 0.0
    min_util  = min(utils) if utils else 0.0
    mean_util = sum(utils) / len(utils) if utils else 0.0

    imbalanced = (
        len(tiles) > 1
        and mean_util > 0
        and (max_util - min_util) / mean_util > _IMBALANCE_WARN_FRAC
    )

    kv_pressure_max = 0.0
    if tile_mem_bytes > 0:
        kv_pressure_max = max(
            (t.get("kv_bytes", 0) / tile_mem_bytes for t in tiles),
            default=0.0,
        )

    max_logits = max(
        (t.get("logits_bytes_per_token",    0) for t in tiles), default=0
    )
    max_act = max(
        (t.get("activation_bytes_per_token", 0) for t in tiles), default=0
    )
    # Flag logits traffic only when bandwidth is already constrained
    logits_heavy = (
        max_act > 0
        and max_logits > 0.10 * max_act
        and (bool(bw_fail_tiles) or bool(bw_warn_tiles))
    )

    # Largest tensors by packed_bytes
    tensor_sizes = sorted(
        [(t.get("packed_bytes", 0), t.get("tensor_name", "?")) for t in tensors],
        reverse=True,
    )[:5]
    oversized = [
        (pb, tn)
        for pb, tn in tensor_sizes
        if tile_mem_bytes > 0 and pb > tile_mem_bytes
    ]

    # Flags from existing warnings/failures ----------------------------------
    all_msgs = [
        str(w.get("message", "")).lower()
        for w in (ex_warn + ex_fail)
        if isinstance(w, dict)
    ]
    has_synthetic  = any("synthetic" in m for m in all_msgs)
    q4_align_issues = any("q4" in m or "alignment" in m for m in all_msgs)

    # Store analysis summary -------------------------------------------------
    rpt.analysis = {
        "total_model_bytes":      total_model_bytes,
        "total_kv_bytes":         total_kv_bytes,
        "tile_count":             tile_count,
        "tile_memory_mib":        tile_mem_mib,
        "capacity_fail_tiles":    len(fail_tiles),
        "capacity_warn_tiles":    len(warn_tiles),
        "bandwidth_fail_tiles":   len(bw_fail_tiles),
        "bandwidth_warn_tiles":   len(bw_warn_tiles),
        "max_utilization_pct":    round(max_util,  1),
        "min_utilization_pct":    round(min_util,  1),
        "mean_utilization_pct":   round(mean_util, 1),
        "imbalanced_tiles":       imbalanced,
        "kv_pressure_max_frac":   round(kv_pressure_max, 4),
        "logits_heavy":           logits_heavy,
        "oversized_tensor_count": len(oversized),
        "has_synthetic_note":     has_synthetic,
        "q4_alignment_issues":    q4_align_issues,
        "dtype":                  dtype,
    }

    # Capacity FAIL proposals ------------------------------------------------
    if fail_tiles:
        if tile_mem_bytes > 0:
            min_tiles = max(1, math.ceil(total_model_bytes / (tile_mem_bytes * 0.8)))
            rpt.add(
                "capacity-fail",
                "Increase tile count",
                f"current {tile_count}, recommended minimum {min_tiles}",
                severity="fail",
            )
            min_mem = math.ceil(total_model_bytes / tile_count / 0.8 / _MIB)
            rpt.add(
                "capacity-fail",
                "Increase tile memory",
                f"current {tile_mem_mib} MiB per tile, "
                f"recommended minimum {min_mem} MiB",
                severity="fail",
            )
        else:
            rpt.add(
                "capacity-fail",
                "Set tile_memory_mib to enable capacity analysis",
                severity="fail",
            )

        # Dtype reduction proposals
        if dtype == "f32":
            rpt.add(
                "capacity-fail",
                "Convert to q8 to reduce model_bytes by approximately 4x",
                severity="warn",
            )
            rpt.add(
                "capacity-fail",
                "Convert to q4 to reduce model_bytes by approximately 8x "
                "(accuracy trade-off)",
                severity="warn",
            )
        elif dtype in ("q8", "bf16"):
            rpt.add(
                "capacity-fail",
                "Convert to q4 to reduce model_bytes by approximately 2x",
                severity="warn",
            )

        # lm_head split
        if any(t.get("logits_bytes_per_token", 0) > 0 for t in tiles):
            rpt.add(
                "capacity-fail",
                "Split lm_head across all tiles to distribute "
                "logit projection memory",
                severity="warn",
            )

        # Embedding split
        emb = [t for t in tensors if t.get("tensor_category") == "embedding"]
        if emb:
            rpt.add(
                "capacity-fail",
                "Split tok_embeddings across tiles to reduce tile 0 memory "
                "pressure",
                severity="warn",
            )

    # Capacity WARN proposals ------------------------------------------------
    elif warn_tiles:
        if tile_mem_bytes > 0:
            worst = max(
                t.get("model_bytes", 0) + t.get("kv_bytes", 0)
                for t in warn_tiles
            )
            min_mem = math.ceil(worst / 0.8 / _MIB)
            rpt.add(
                "capacity-warn",
                "Increase tile memory",
                f"current {tile_mem_mib} MiB per tile, "
                f"recommended {min_mem} MiB to clear warning",
                severity="warn",
            )
        if ctx_len > 512:
            rpt.add(
                "capacity-warn",
                "Reduce target context length",
                f"current {ctx_len}; halving context reduces KV bytes by ~50%",
                severity="warn",
            )
        if sessions > 1:
            rpt.add(
                "capacity-warn",
                "Reduce session count",
                f"current {sessions}; each session multiplies KV bytes",
                severity="warn",
            )

    # KV pressure proposals --------------------------------------------------
    if kv_pressure_max > _KV_PRESSURE_FAIL_FRAC:
        rpt.add(
            "kv-pressure",
            "Excessive KV cache pressure",
            f"{kv_pressure_max * 100:.1f}% of tile memory is KV; "
            "reduce context length or session count",
            severity="fail" if fail_tiles else "warn",
        )
    elif kv_pressure_max > _KV_PRESSURE_WARN_FRAC:
        rpt.add(
            "kv-pressure",
            "High KV cache pressure",
            f"{kv_pressure_max * 100:.1f}% of tile memory is KV",
            severity="warn",
        )

    # Bandwidth proposals ----------------------------------------------------
    if bw_fail_tiles:
        rpt.add(
            "bandwidth-fail",
            "Increase fabric bandwidth or switch to head-wise placement",
            f"{len(bw_fail_tiles)} tile(s) exceed fabric_gib_sec target",
            severity="fail",
        )
    elif bw_warn_tiles:
        rpt.add(
            "bandwidth-warn",
            "Reduce activation traffic or increase fabric bandwidth",
            "head-wise placement keeps QKV traffic tile-local",
            severity="warn",
        )

    # Logits traffic proposals -----------------------------------------------
    if logits_heavy:
        rpt.add(
            "logits-traffic",
            "Split lm_head across tiles to distribute decode-step logit "
            "traffic",
            f"logits_bytes_per_token={max_logits}",
            severity="warn",
        )

    # Load imbalance proposals -----------------------------------------------
    if imbalanced:
        spread = max_util - min_util
        rpt.add(
            "load-imbalance",
            "Redistribute layers to balance tile utilization",
            f"spread {spread:.1f}% (min {min_util:.1f}% / "
            f"max {max_util:.1f}%)",
            severity="warn",
        )

    # Oversized tensor proposals ---------------------------------------------
    for pb, tn in oversized:
        rpt.add(
            "oversized-tensor",
            f"Tensor '{tn}' exceeds tile memory",
            f"{pb / _MIB:.1f} MiB > {tile_mem_mib} MiB; split this tensor",
            severity="fail",
        )

    # Q4 alignment proposals -------------------------------------------------
    if q4_align_issues:
        rpt.add(
            "q4-alignment",
            "Q4 group alignment warnings detected",
            "ensure slice_end is a multiple of group_size (default 32) "
            "or equals source_shape[0]",
            severity="warn",
        )

    # Synthetic model note ---------------------------------------------------
    if has_synthetic:
        rpt.add(
            "model-note",
            "Report contains synthetic/non-executable model notes",
            "validate with a real .att1 artifact before production use",
            severity="warn",
        )

    # Next recommended action ------------------------------------------------
    if rpt.status == "fail":
        if fail_tiles and tile_mem_bytes > 0:
            min_tiles = max(1, math.ceil(total_model_bytes / (tile_mem_bytes * 0.8)))
            min_mem   = math.ceil(total_model_bytes / tile_count / 0.8 / _MIB)
            rpt.next_action = (
                f"Add more tiles (current {tile_count}, minimum {min_tiles}) "
                f"or increase tile_memory_mib to at least {min_mem} MiB."
            )
        elif oversized:
            rpt.next_action = (
                f"Split oversized tensor '{oversized[0][1]}' "
                "or increase tile memory."
            )
        elif bw_fail_tiles:
            rpt.next_action = (
                "Increase fabric_gib_sec target or switch to head-wise "
                "placement."
            )
        else:
            rpt.next_action = "Review failures listed above."
    elif rpt.status == "warn":
        if warn_tiles:
            rpt.next_action = (
                "Increase tile memory or reduce model size (q4 conversion)."
            )
        elif kv_pressure_max > _KV_PRESSURE_WARN_FRAC:
            rpt.next_action = (
                "Reduce context length or session count to relieve KV "
                "pressure."
            )
        elif bw_warn_tiles:
            rpt.next_action = (
                "Increase fabric bandwidth or use head-wise placement."
            )
        elif imbalanced:
            rpt.next_action = "Rebalance layer ranges across tiles."
        else:
            rpt.next_action = "Review warnings listed above."
    else:
        rpt.next_action = (
            "No major remediation required. "
            "Placement fits within capacity and bandwidth targets."
        )

    return rpt


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="ATT-1 tensor placement advisory proposal tool (M101).",
    )
    ap.add_argument(
        "--report", required=True,
        help="Path to M98/M100 placement report JSON.",
    )
    ap.add_argument(
        "--report-json",
        help="Optional path to write advisory JSON output.",
    )
    args = ap.parse_args()

    data = _parse_report(args.report)
    rpt  = _analyze(data)

    print(rpt.render_text())

    if args.report_json:
        try:
            with open(args.report_json, "w") as fh:
                json.dump(rpt.to_dict(), fh, indent=2)
                fh.write("\n")
        except OSError as exc:
            print(f"error: cannot write advisory report: {exc}", file=sys.stderr)
            sys.exit(2)

    sys.exit(1 if rpt.status == "fail" else 0)


if __name__ == "__main__":
    main()
