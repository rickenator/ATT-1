#!/usr/bin/env python3
"""
ATT-1 trace diff (Milestone 94).

Reads two att1-bench key=value output files and reports execution, counter,
and numerical differences between backends, shard plans, or modes.

Parses all key=value fields, per-layer counters (layer[N].*), and per-tile
counters (tile[N].*) from standard att1-bench output.

Exit codes:
  0 — both files parsed successfully (differences are reported, not failures)
  1 — one or both files could not be parsed (malformed or missing)

Usage (identical — no differences expected):

    python3 compiler/trace_diff.py build/run_a.txt build/run_a.txt

Usage (cross-backend comparison):

    python3 compiler/trace_diff.py build/f32_single.txt build/q8_single.txt \\
        --report-json build/diff_f32_vs_q8.json

Usage (cross-mode comparison with JSON output):

    python3 compiler/trace_diff.py build/single.txt build/cluster.txt \\
        --report-json build/diff_single_vs_cluster.json
"""

import argparse
import json
import re
import sys

# ---------------------------------------------------------------------------
# Field definitions
# ---------------------------------------------------------------------------

# Fields compared as strings; differences are reported but not scored as errors
_ID_FIELDS = (
    "mode",
    "shard_plan",
    "backend",
    "tokenizer",
)

# Header numeric fields (run configuration)
_HEADER_FIELDS = (
    "prompt_tokens",
    "requested_tokens",
    "benchmark_tokens",
    "generated_tokens",
    "last_token",
)

# Trace counter fields
_COUNTER_FIELDS = (
    "tokens_decoded",
    "token_time_us_total",
    "token_time_us_max",
    "layer_time_us_total",
    "activation_bytes_sent",
    "logits_bytes_produced",
    "fabric_packets_sent",
    "fabric_packets_received",
    "fabric_payload_bytes_sent",
    "fabric_payload_bytes_received",
    "kv_appends",
    "kv_key_reads",
    "kv_value_reads",
    "tile_layer_executions",
)

# Ordered sequence of all scalar fields to compare
_SCALAR_FIELDS = _ID_FIELDS + _HEADER_FIELDS + _COUNTER_FIELDS

# Shard-meta summary fields that may appear in bench output
_SHARD_META_FIELDS = (
    "shard_meta",
    "shard_meta_count",
    "shard_meta_assigned",
    "shard_meta_unassigned",
    "shard_meta_tiles",
    "shard_meta_aimus",
    "shard_meta_dtype_f32",
    "shard_meta_dtype_q8",
)

# Prefill/decode phase split fields (M95)
_PREFILL_DECODE_FIELDS = (
    "decode_tokens",
    "prefill_time_us_total",
    "decode_time_us_total",
    "prefill_kv_appends",
    "decode_kv_appends",
    "prefill_kv_reads",
    "decode_kv_reads",
    "prefill_logits_bytes",
    "decode_logits_bytes",
    "prefill_fabric_packets",
    "decode_fabric_packets",
)

_ALL_SCALAR_FIELDS = _SCALAR_FIELDS + _SHARD_META_FIELDS + _PREFILL_DECODE_FIELDS

# ---------------------------------------------------------------------------
# Patterns for structured output lines
# ---------------------------------------------------------------------------

# layer[N].executions=X time_us=Y kv_appends=Z
_LAYER_RE = re.compile(
    r"^layer\[(\d+)\]\.executions=(\d+)\s+time_us=(\d+)\s+kv_appends=(\d+)$"
)

# tile[N].layers=X activation_bytes_sent=Y logits_bytes=Z
_TILE_RE = re.compile(
    r"^tile\[(\d+)\]\.layers=(\d+)\s+activation_bytes_sent=(\d+)\s+logits_bytes=(\d+)$"
)

# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def _parse(path):
    """
    Parse a bench output file into a structured record.

    Returns a dict with:
      "_raw"    — {key: value_str} for all key=value lines
      "_layers" — {layer_idx: {"executions": int, "time_us": int, "kv_appends": int}}
      "_tiles"  — {tile_idx: {"layers": int, "activation_bytes_sent": int, "logits_bytes": int}}

    Raises ValueError if the file cannot be opened or contains no parseable
    content at all.
    """
    try:
        with open(path) as fh:
            text = fh.read()
    except OSError as exc:
        raise ValueError(f"cannot open {path!r}: {exc}") from exc

    record = {"_raw": {}, "_layers": {}, "_tiles": {}}

    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        m = _LAYER_RE.match(line)
        if m:
            idx = int(m.group(1))
            record["_layers"][idx] = {
                "executions": int(m.group(2)),
                "time_us":    int(m.group(3)),
                "kv_appends": int(m.group(4)),
            }
            continue

        m = _TILE_RE.match(line)
        if m:
            idx = int(m.group(1))
            record["_tiles"][idx] = {
                "layers":                 int(m.group(2)),
                "activation_bytes_sent":  int(m.group(3)),
                "logits_bytes":           int(m.group(4)),
            }
            continue

        if "=" in line:
            key, _, val = line.partition("=")
            record["_raw"][key.strip()] = val.strip()

    has_content = bool(record["_raw"] or record["_layers"] or record["_tiles"])
    if not has_content:
        raise ValueError(f"no parseable key=value fields found in {path!r}")

    return record


# ---------------------------------------------------------------------------
# Diff helpers
# ---------------------------------------------------------------------------

def _val(record, key):
    """Return string value for a scalar field, or None if absent."""
    return record["_raw"].get(key)


def _diff_scalar(key, va, vb, diffs, lines):
    """
    Compare one scalar field.  Append a formatted report line.
    Records entries in *diffs* only when the values actually differ or one
    side is missing.
    """
    if va is None and vb is None:
        return

    col_a = str(va) if va is not None else "(missing)"
    col_b = str(vb) if vb is not None else "(missing)"

    if va is None or vb is None:
        tag = "MISSING"
        diffs.append({"field": key, "a": va, "b": vb, "status": "missing"})
    elif col_a == col_b:
        tag = "same"
    else:
        tag = "DIFF"
        diffs.append({"field": key, "a": va, "b": vb, "status": "diff"})

    lines.append(f"  {key:<40s} {col_a:<20s} {col_b:<20s} {tag}")


def _diff_int(key, va, vb, diffs, lines):
    """Compare an integer sub-field from a layer or tile record."""
    if va is None and vb is None:
        return
    _diff_scalar(key, str(va) if va is not None else None,
                 str(vb) if vb is not None else None, diffs, lines)


# ---------------------------------------------------------------------------
# Main diff logic
# ---------------------------------------------------------------------------

def diff(path_a, path_b):
    """
    Compare two bench output files.

    Returns ``(report_str, diff_list, None)`` on success, or
    ``(None, None, error_str)`` if either file cannot be parsed.

    *diff_list* is a list of dicts:
        {"field": str, "a": str|None, "b": str|None,
         "status": "diff"|"missing"}
    """
    try:
        rec_a = _parse(path_a)
        rec_b = _parse(path_b)
    except ValueError as exc:
        return None, None, str(exc)

    diffs  = []
    lines  = []

    lines.append("# ATT-1 trace diff (M94)")
    lines.append(f"file_a: {path_a}")
    lines.append(f"file_b: {path_b}")
    lines.append("")
    lines.append(f"  {'field':<40s} {'a':<20s} {'b':<20s} status")
    lines.append("  " + "-" * 86)

    # Scalar fields
    for key in _ALL_SCALAR_FIELDS:
        _diff_scalar(key, _val(rec_a, key), _val(rec_b, key), diffs, lines)

    # Per-layer counters
    all_layer_idxs = sorted(
        set(rec_a["_layers"].keys()) | set(rec_b["_layers"].keys())
    )
    for idx in all_layer_idxs:
        la = rec_a["_layers"].get(idx, {})
        lb = rec_b["_layers"].get(idx, {})
        for sub in ("executions", "time_us", "kv_appends"):
            key = f"layer[{idx}].{sub}"
            _diff_int(key, la.get(sub), lb.get(sub), diffs, lines)

    # Per-tile counters
    all_tile_idxs = sorted(
        set(rec_a["_tiles"].keys()) | set(rec_b["_tiles"].keys())
    )
    for idx in all_tile_idxs:
        ta = rec_a["_tiles"].get(idx, {})
        tb = rec_b["_tiles"].get(idx, {})
        for sub in ("layers", "activation_bytes_sent", "logits_bytes"):
            key = f"tile[{idx}].{sub}"
            _diff_int(key, ta.get(sub), tb.get(sub), diffs, lines)

    lines.append("")

    n_diff    = sum(1 for d in diffs if d["status"] == "diff")
    n_missing = sum(1 for d in diffs if d["status"] == "missing")
    lines.append(f"differences: {n_diff}  missing: {n_missing}")
    lines.append("result: pass")
    lines.append("report: ok")

    return "\n".join(lines) + "\n", diffs, None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="trace_diff",
        description="Compare two att1-bench output files.",
    )
    ap.add_argument("file_a", help="first bench output file (A side)")
    ap.add_argument("file_b", help="second bench output file (B side)")
    ap.add_argument(
        "--report-json",
        metavar="PATH",
        help="write JSON diff report to PATH",
    )
    args = ap.parse_args(argv)

    report, diffs, error = diff(args.file_a, args.file_b)
    if error is not None:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)

    print(report, end="")

    if args.report_json:
        n_diff    = sum(1 for d in diffs if d["status"] == "diff")
        n_missing = sum(1 for d in diffs if d["status"] == "missing")
        payload = {
            "file_a":       args.file_a,
            "file_b":       args.file_b,
            "differences":  diffs,
            "n_diff":       n_diff,
            "n_missing":    n_missing,
            "result":       "pass",
        }
        try:
            with open(args.report_json, "w") as fh:
                json.dump(payload, fh, indent=2)
                fh.write("\n")
        except OSError as exc:
            print(f"error: cannot write JSON report: {exc}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
