#!/usr/bin/env python3
"""
ATT-1 M173 placement and capacity validation gate.

Consumes an M98/M100 tensor placement report and evaluates the fixed Phase 2
device-memory envelopes that feed the M175 FPGA gate: 256 MiB, 512 MiB, and
1024 MiB per tile.  It also records the M93 §8.15-2 KV page-size decision by
scoring candidate page-token sizes against page byte size, page count, and
worst-case per-layer/session waste.

Exit codes:
  0 — at least one required budget is feasible and a KV page size is selected
  1 — no required budget is feasible
  2 — malformed input or invalid CLI arguments
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

_MIB = 1 << 20
_DEFAULT_BUDGETS_MIB = [256, 512, 1024]
_DEFAULT_PAGE_TOKENS = [16, 32, 64, 128]
_DEFAULT_CONTEXT = 2048
_DEFAULT_SESSIONS = 1
_CAPACITY_WARN_PCT = 80.0
_F32_BYTES = 4


def _die(message: str) -> None:
    print(f"error: {message}", file=sys.stderr)
    sys.exit(2)


def _load_report(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        _die(f"cannot parse placement report: {exc}")

    if not isinstance(data, dict):
        _die("placement report root must be an object")
    if not isinstance(data.get("header"), dict):
        _die("placement report missing object header")
    if not isinstance(data.get("tiles"), list) or not data["tiles"]:
        _die("placement report must contain a non-empty tiles array")
    return data


def _parse_positive_int_list(raw: str, name: str) -> list[int]:
    out: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            value = int(part)
        except ValueError:
            _die(f"{name} contains non-integer value {part!r}")
        if value <= 0:
            _die(f"{name} values must be positive")
        out.append(value)
    if not out:
        _die(f"{name} must not be empty")
    return out


def _as_positive_int(value: Any, default: int) -> int:
    if isinstance(value, int) and value > 0:
        return value
    return default


def _head_dim(header: dict[str, Any]) -> int:
    explicit = header.get("head_dim")
    if isinstance(explicit, int) and explicit > 0:
        return explicit

    d_model = header.get("d_model")
    n_heads = header.get("n_heads")
    if isinstance(d_model, int) and isinstance(n_heads, int) and n_heads > 0:
        return max(1, d_model // n_heads)
    return 64


def _status(util_pct: float) -> str:
    if util_pct > 100.0:
        return "FAIL"
    if util_pct > _CAPACITY_WARN_PCT:
        return "WARN"
    return "PASS"


def _scale_kv_bytes(
    tile: dict[str, Any],
    orig_context: int,
    orig_sessions: int,
    target_context: int,
    target_sessions: int,
) -> int:
    kv = tile.get("kv_bytes", 0)
    if not isinstance(kv, int) or kv < 0:
        _die("tile kv_bytes values must be non-negative integers")

    scale = (target_context / max(1, orig_context)) * (
        target_sessions / max(1, orig_sessions)
    )
    return int(math.ceil(kv * scale))


def _evaluate_budget(
    report: dict[str, Any],
    budget_mib: int,
    target_context: int,
    target_sessions: int,
) -> dict[str, Any]:
    header = report["header"]
    orig_context = _as_positive_int(
        header.get("target_context_length"), target_context
    )
    orig_sessions = _as_positive_int(header.get("target_sessions"), 1)
    budget_bytes = budget_mib * _MIB

    tile_rows = []
    worst_status = "PASS"
    max_util = 0.0
    total_model = 0
    total_kv = 0
    total_scratch = 0

    for tile in report["tiles"]:
        if not isinstance(tile, dict):
            _die("tiles entries must be objects")

        tile_id = tile.get("tile_id")
        model_bytes = tile.get("model_bytes", 0)
        if not isinstance(tile_id, int):
            _die("tile_id values must be integers")
        if not isinstance(model_bytes, int) or model_bytes < 0:
            _die("tile model_bytes values must be non-negative integers")

        kv_bytes = _scale_kv_bytes(
            tile, orig_context, orig_sessions, target_context, target_sessions
        )
        scratch_bytes = 0
        for field in ("activation_bytes_per_token", "logits_bytes_per_token"):
            value = tile.get(field, 0)
            if not isinstance(value, int) or value < 0:
                _die(f"tile {field} values must be non-negative integers")
            scratch_bytes += value

        total_bytes = model_bytes + kv_bytes + scratch_bytes
        util_pct = (total_bytes / budget_bytes * 100.0) if budget_bytes else 0.0
        cap_status = _status(util_pct)
        if cap_status == "FAIL":
            worst_status = "FAIL"
        elif cap_status == "WARN" and worst_status == "PASS":
            worst_status = "WARN"
        max_util = max(max_util, util_pct)

        total_model += model_bytes
        total_kv += kv_bytes
        total_scratch += scratch_bytes
        tile_rows.append(
            {
                "tile_id": tile_id,
                "model_bytes": model_bytes,
                "kv_bytes": kv_bytes,
                "scratch_bytes": scratch_bytes,
                "total_bytes": total_bytes,
                "utilization_percent": round(util_pct, 3),
                "capacity_status": cap_status,
            }
        )

    return {
        "budget_mib": budget_mib,
        "budget_bytes": budget_bytes,
        "capacity_status": worst_status,
        "max_utilization_percent": round(max_util, 3),
        "total_model_bytes": total_model,
        "total_kv_bytes": total_kv,
        "total_scratch_bytes": total_scratch,
        "tiles": tile_rows,
    }


def _score_kv_pages(
    header: dict[str, Any],
    target_context: int,
    target_sessions: int,
    page_tokens_list: list[int],
    max_page_kib: int,
) -> list[dict[str, Any]]:
    num_layers = _as_positive_int(header.get("n_layers"), 1)
    num_heads = _as_positive_int(header.get("n_kv_heads"), 0)
    if num_heads == 0:
        num_heads = _as_positive_int(header.get("n_heads"), 1)
    head_dim = _head_dim(header)

    rows = []
    for page_tokens in page_tokens_list:
        page_bytes = page_tokens * num_heads * head_dim * 2 * _F32_BYTES
        pages_per_layer_session = math.ceil(target_context / page_tokens)
        total_pages = pages_per_layer_session * num_layers * target_sessions
        worst_case_waste_tokens = max(0, page_tokens - 1)
        worst_case_waste_bytes_per_layer_session = (
            worst_case_waste_tokens * num_heads * head_dim * 2 * _F32_BYTES
        )
        valid = page_bytes <= max_page_kib * 1024

        score = (
            worst_case_waste_bytes_per_layer_session
            + page_bytes
            + pages_per_layer_session * 64
        )
        if not valid:
            score += 1 << 60

        rows.append(
            {
                "page_tokens": page_tokens,
                "page_bytes": page_bytes,
                "pages_per_layer_session": pages_per_layer_session,
                "total_pages": total_pages,
                "worst_case_waste_bytes_per_layer_session":
                    worst_case_waste_bytes_per_layer_session,
                "valid": valid,
                "score": score,
            }
        )

    rows.sort(key=lambda row: (not row["valid"], row["score"], row["page_tokens"]))
    return rows


def _render_text(report: dict[str, Any], out: dict[str, Any]) -> str:
    header = report["header"]
    lines = [
        "# ATT-1 M173 capacity validation",
        f"model_name: {header.get('model_name', '')}",
        f"dtype: {header.get('dtype', '')}",
        f"target_context_length: {out['target_context_length']}",
        f"target_sessions: {out['target_sessions']}",
        "budgets:",
    ]
    for row in out["budgets"]:
        lines.append(
            "  budget_mib={budget_mib} status={capacity_status} "
            "max_utilization_percent={max_utilization_percent}".format(**row)
        )

    decision = out["kv_page_decision"]
    lines.extend(
        [
            "kv_page_decision:",
            f"  selected_page_tokens: {decision['selected_page_tokens']}",
            f"  selected_page_bytes: {decision['selected_page_bytes']}",
            "result: " + out["result"],
        ]
    )
    if out["required_pass_budget_mib"] is not None:
        lines.append(
            f"required_pass_budget_mib: {out['required_pass_budget_mib']}"
        )
    lines.append("report: ok")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Validate M173 placement capacity budgets and KV page size.",
    )
    ap.add_argument("--report", required=True, help="Placement report JSON path.")
    ap.add_argument(
        "--budgets-mib",
        default=",".join(str(v) for v in _DEFAULT_BUDGETS_MIB),
        help="Comma-separated per-tile memory budgets in MiB.",
    )
    ap.add_argument(
        "--context",
        type=int,
        default=_DEFAULT_CONTEXT,
        help="Target context length for scaled KV capacity.",
    )
    ap.add_argument(
        "--sessions",
        type=int,
        default=_DEFAULT_SESSIONS,
        help="Target simultaneous sessions for scaled KV capacity.",
    )
    ap.add_argument(
        "--kv-page-tokens",
        default=",".join(str(v) for v in _DEFAULT_PAGE_TOKENS),
        help="Comma-separated candidate KV page sizes in tokens.",
    )
    ap.add_argument(
        "--max-kv-page-kib",
        type=int,
        default=256,
        help="Maximum acceptable bytes per KV page, in KiB.",
    )
    ap.add_argument(
        "--require-pass-budget-mib",
        type=int,
        default=None,
        help="If supplied, this budget must be PASS for exit 0.",
    )
    ap.add_argument(
        "--report-json",
        default=None,
        help="Optional machine-readable report output path.",
    )
    args = ap.parse_args()

    if args.context <= 0:
        _die("--context must be positive")
    if args.sessions <= 0:
        _die("--sessions must be positive")
    if args.max_kv_page_kib <= 0:
        _die("--max-kv-page-kib must be positive")

    budgets = _parse_positive_int_list(args.budgets_mib, "--budgets-mib")
    page_tokens = _parse_positive_int_list(
        args.kv_page_tokens, "--kv-page-tokens"
    )
    if args.require_pass_budget_mib is not None and (
        args.require_pass_budget_mib not in budgets
    ):
        _die("--require-pass-budget-mib must be one of --budgets-mib")

    report = _load_report(Path(args.report))
    budget_rows = [
        _evaluate_budget(report, budget, args.context, args.sessions)
        for budget in budgets
    ]
    page_rows = _score_kv_pages(
        report["header"],
        args.context,
        args.sessions,
        page_tokens,
        args.max_kv_page_kib,
    )
    selected_page = next((row for row in page_rows if row["valid"]), None)
    if selected_page is None:
        _die("no valid KV page-token candidate under --max-kv-page-kib")

    pass_budgets = [
        row["budget_mib"] for row in budget_rows
        if row["capacity_status"] == "PASS"
    ]
    if args.require_pass_budget_mib is not None:
        result = (
            "pass" if args.require_pass_budget_mib in pass_budgets else "fail"
        )
    else:
        result = "pass" if pass_budgets else "fail"

    out = {
        "m173_capacity_report_version": 1,
        "source_report": args.report,
        "target_context_length": args.context,
        "target_sessions": args.sessions,
        "budgets": budget_rows,
        "pass_budgets_mib": pass_budgets,
        "required_pass_budget_mib": args.require_pass_budget_mib,
        "kv_page_decision": {
            "selected_page_tokens": selected_page["page_tokens"],
            "selected_page_bytes": selected_page["page_bytes"],
            "max_kv_page_kib": args.max_kv_page_kib,
            "rationale": (
                "Select the smallest valid page-score candidate to minimize "
                "worst-case tail waste while keeping page count bounded."
            ),
            "candidates": page_rows,
        },
        "result": result,
    }

    print(_render_text(report, out))
    if args.report_json is not None:
        try:
            with Path(args.report_json).open("w", encoding="utf-8") as fh:
                json.dump(out, fh, indent=2)
                fh.write("\n")
        except OSError as exc:
            _die(f"cannot write report JSON: {exc}")

    sys.exit(0 if result == "pass" else 1)


if __name__ == "__main__":
    main()
