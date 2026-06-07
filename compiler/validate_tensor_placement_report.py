#!/usr/bin/env python3
"""
ATT-1 tensor-level placement report validator (Milestone 99).

Reads a placement report JSON produced according to the M98 schema
(docs/tensor_placement_report.md) and validates it for structural
correctness, coverage, capacity consistency, and tensor-slice integrity.

Exit codes:
  0 — validation passed (zero errors; warnings may be present)
  1 — validation failed (one or more errors detected)
  2 — report could not be parsed (malformed JSON or missing required fields)

Usage:

    python3 compiler/validate_tensor_placement_report.py \\
        --report compiler/fixtures/placement_report_valid.json

    python3 compiler/validate_tensor_placement_report.py \\
        --report build/my_placement.json \\
        --report-json build/validation_result.json

    python3 compiler/validate_tensor_placement_report.py \\
        --report build/my_placement.json \\
        --strict

In strict mode, warnings are promoted to errors.
"""

import argparse
import json
import sys
from collections import defaultdict

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUPPORTED_REPORT_VERSIONS = {1}

VALID_DTYPES = {"f32", "q8", "q4", "bf16"}

VALID_QUANTIZATION_FAMILIES = {
    "none", "per_row_q8", "per_group_q4", "per_tensor_q8",
}

VALID_TENSOR_CATEGORIES = {
    "embedding", "attention_q", "attention_k", "attention_v", "attention_o",
    "ffn_gate", "ffn_up", "ffn_down", "norm", "lm_head",
    "kv_cache", "activation", "other",
}

VALID_REPLICATION_POLICIES = {"unique", "replicated", "partial"}

VALID_REDUCTION_BEHAVIORS = {"none", "sum", "max", "concat"}

VALID_ROUTING_REQUIREMENTS = {
    "none", "broadcast_activation", "fixed_path",
    "lookup_route", "concat_return",
}

VALID_PLACEMENT_STATUSES = {
    "placed", "unplaced", "partial", "overflow", "invalid",
}

VALID_CAPACITY_STATUSES = {"PASS", "WARN", "FAIL", "UNKNOWN"}

VALID_BANDWIDTH_STATUSES = {"PASS", "WARN", "FAIL", "UNKNOWN"}

# Q4 default group size used for alignment checks when not explicit
_Q4_DEFAULT_GROUP_SIZE = 32


# ---------------------------------------------------------------------------
# Result collector
# ---------------------------------------------------------------------------

class ValidationResult:
    def __init__(self, strict: bool = False):
        self.strict = strict
        self.warnings: list[dict] = []
        self.errors: list[dict] = []

    def warn(self, rule: int, message: str, **kw):
        rec = {"severity": "warning", "rule": rule, "message": message}
        rec.update(kw)
        self.warnings.append(rec)

    def error(self, rule: int, message: str, **kw):
        rec = {"severity": "error", "rule": rule, "message": message}
        rec.update(kw)
        self.errors.append(rec)

    def issue(self, severity: str, rule: int, message: str, **kw):
        """Emit warning or error; in strict mode warnings become errors."""
        if severity == "error" or self.strict:
            self.error(rule, message, **kw)
        else:
            self.warn(rule, message, **kw)

    @property
    def passed(self) -> bool:
        return len(self.errors) == 0

    def summary(self) -> dict:
        return {
            "status": "pass" if self.passed else "fail",
            "total_warnings": len(self.warnings),
            "total_errors": len(self.errors),
            "warnings": self.warnings,
            "failures": self.errors,
        }


# ---------------------------------------------------------------------------
# Header validation
# ---------------------------------------------------------------------------

def validate_header(report: dict, res: ValidationResult) -> int:
    """Validate §1 report header.  Returns declared tile_count or 0."""
    header = report.get("header")
    if header is None:
        res.error(0, "Missing 'header' section")
        return 0

    # report_version
    ver = report.get("report_version")
    if ver is None:
        res.error(0, "Missing 'report_version'")
    elif ver not in SUPPORTED_REPORT_VERSIONS:
        res.error(0, f"Unsupported report_version={ver!r}; "
                     f"supported: {sorted(SUPPORTED_REPORT_VERSIONS)}")

    # model_name
    if "model_name" not in header:
        res.issue("warning", 0, "Missing 'model_name' in header")

    # dtype
    dtype = header.get("dtype")
    if dtype is None:
        res.error(0, "Missing 'dtype' in header")
    elif dtype not in VALID_DTYPES:
        res.error(0, f"Unknown dtype={dtype!r}")

    # quantization_family
    qf = header.get("quantization_family")
    if qf is None:
        res.issue("warning", 0, "Missing 'quantization_family' in header")
    elif qf not in VALID_QUANTIZATION_FAMILIES:
        res.error(0, f"Unknown quantization_family={qf!r}")

    # tile_count
    tile_count = header.get("tile_count")
    if tile_count is None:
        res.error(0, "Missing 'tile_count' in header")
        tile_count = 0
    elif not isinstance(tile_count, int) or tile_count <= 0:
        res.error(0, f"tile_count must be a positive integer; got {tile_count!r}")
        tile_count = 0

    # target_context_length
    ctx = header.get("target_context_length")
    if ctx is not None and (not isinstance(ctx, int) or ctx < 1):
        res.error(0, f"target_context_length must be >= 1; got {ctx!r}")

    # target_sessions
    sess = header.get("target_sessions")
    if sess is not None and (not isinstance(sess, int) or sess < 1):
        res.error(0, f"target_sessions must be >= 1; got {sess!r}")

    # tile_memory_mib (optional, must be positive if present)
    tmm = header.get("tile_memory_mib")
    if tmm is not None and (not isinstance(tmm, (int, float)) or tmm <= 0):
        res.error(0, f"tile_memory_mib must be a positive number; got {tmm!r}")

    # fabric_gib_sec (optional, must be positive if present)
    fgs = header.get("fabric_gib_sec")
    if fgs is not None and (not isinstance(fgs, (int, float)) or fgs <= 0):
        res.error(0, f"fabric_gib_sec must be a positive number; got {fgs!r}")

    return int(tile_count) if isinstance(tile_count, int) and tile_count > 0 else 0


# ---------------------------------------------------------------------------
# Tile summary validation
# ---------------------------------------------------------------------------

def validate_tiles(report: dict, tile_count: int, res: ValidationResult) -> set:
    """Validate §2 tile summary records.  Returns set of valid tile IDs."""
    tiles = report.get("tiles", [])
    if not isinstance(tiles, list):
        res.error(0, "'tiles' must be a JSON array")
        return set()

    valid_ids: set[int] = set()
    seen_ids: set[int] = set()

    for i, tile in enumerate(tiles):
        if not isinstance(tile, dict):
            res.error(0, f"tiles[{i}] is not an object")
            continue

        tid = tile.get("tile_id")
        if tid is None:
            res.error(0, f"tiles[{i}] missing 'tile_id'")
            continue
        if not isinstance(tid, int):
            res.error(0, f"tiles[{i}].tile_id must be an integer; got {tid!r}")
            continue
        if tid in seen_ids:
            res.error(0, f"Duplicate tile_id={tid}")
            continue
        seen_ids.add(tid)

        if tile_count > 0 and tid >= tile_count:
            res.error(4, f"tile_id={tid} out of range for tile_count={tile_count}")
            continue

        valid_ids.add(tid)

        # Non-negative byte counts
        for field in ("model_bytes", "kv_bytes", "activation_bytes_per_token",
                      "logits_bytes_per_token", "fabric_payload_bytes_per_token"):
            val = tile.get(field)
            if val is None:
                res.issue("warning", 0, f"tiles[tile_id={tid}] missing '{field}'")
            elif not isinstance(val, int) or val < 0:
                res.error(0, f"tiles[tile_id={tid}].{field} must be a "
                             f"non-negative integer; got {val!r}")

        # fabric_packets_per_token
        ppt = tile.get("fabric_packets_per_token")
        if ppt is not None and (not isinstance(ppt, int) or ppt < 0):
            res.error(0, f"tiles[tile_id={tid}].fabric_packets_per_token "
                         f"must be non-negative; got {ppt!r}")

        # memory_utilization_percent
        util = tile.get("memory_utilization_percent")
        if util is not None:
            if not isinstance(util, (int, float)) or util < 0:
                res.error(0, f"tiles[tile_id={tid}].memory_utilization_percent "
                             f"must be a non-negative number; got {util!r}")

        # capacity_status
        cap = tile.get("capacity_status")
        if cap not in VALID_CAPACITY_STATUSES:
            res.error(0, f"tiles[tile_id={tid}].capacity_status={cap!r} "
                         f"not in {sorted(VALID_CAPACITY_STATUSES)}")
        else:
            # Consistency: if PASS, utilization must not exceed 100%
            if cap == "PASS" and util is not None and util > 100:
                res.error(0, f"tiles[tile_id={tid}] capacity_status=PASS but "
                             f"memory_utilization_percent={util:.1f}% > 100%")
            # Consistency: if utilization > 100%, status must not be PASS
            if util is not None and util > 100 and cap == "PASS":
                res.error(0, f"tiles[tile_id={tid}] utilization {util:.1f}% "
                             f"exceeds capacity but status=PASS")

        # bandwidth_status
        bw = tile.get("bandwidth_status")
        if bw not in VALID_BANDWIDTH_STATUSES:
            res.error(0, f"tiles[tile_id={tid}].bandwidth_status={bw!r} "
                         f"not in {sorted(VALID_BANDWIDTH_STATUSES)}")

    return valid_ids


# ---------------------------------------------------------------------------
# Tensor placement record validation
# ---------------------------------------------------------------------------

def validate_tensors(report: dict, valid_tile_ids: set,
                     res: ValidationResult, tile_count: int) -> None:
    """Validate §3 tensor placement records and coverage."""
    tensors = report.get("tensors", [])
    if not isinstance(tensors, list):
        res.error(0, "'tensors' must be a JSON array")
        return

    # Group records by tensor_id for coverage checks
    by_id: dict[int, list[dict]] = defaultdict(list)
    # Track unique-policy tensors by id to detect duplicates
    unique_seen: dict[int, bool] = {}

    for i, rec in enumerate(tensors):
        if not isinstance(rec, dict):
            res.error(0, f"tensors[{i}] is not an object")
            continue

        tid = rec.get("tensor_id")
        name = rec.get("tensor_name", f"<tensor {i}>")

        if tid is None and not name:
            res.error(1, f"tensors[{i}] must have 'tensor_id' or 'tensor_name'",
                      tensor_index=i)
        if tid is not None and not isinstance(tid, int):
            res.error(0, f"tensors[{i}].tensor_id must be an integer; got {tid!r}",
                      tensor_index=i)
            tid = None

        # tensor_category
        cat = rec.get("tensor_category")
        if cat is None:
            res.issue("warning", 0, f"tensors[{i}] '{name}' missing 'tensor_category'",
                      tensor_id=tid, tensor_name=name)
        elif cat not in VALID_TENSOR_CATEGORIES:
            res.issue("warning", 0,
                      f"tensors[{i}] '{name}' unrecognised category={cat!r}; "
                      f"use 'other' for custom categories",
                      tensor_id=tid, tensor_name=name)

        # source_shape and placed_shape
        for shape_field in ("source_shape", "placed_shape"):
            shape = rec.get(shape_field)
            if shape is None:
                res.issue("warning", 0,
                          f"tensors[{i}] '{name}' missing '{shape_field}'",
                          tensor_id=tid, tensor_name=name)
            elif not (isinstance(shape, list) and
                      all(isinstance(d, int) and d > 0 for d in shape)):
                res.error(0,
                          f"tensors[{i}] '{name}' {shape_field} must be a list "
                          f"of positive integers; got {shape!r}",
                          tensor_id=tid, tensor_name=name)

        source_shape = rec.get("source_shape")
        rank = len(source_shape) if isinstance(source_shape, list) else 0

        # dtype
        dtype = rec.get("dtype")
        if dtype is None:
            res.issue("warning", 0, f"tensors[{i}] '{name}' missing 'dtype'",
                      tensor_id=tid, tensor_name=name)
        elif dtype not in VALID_DTYPES:
            res.error(5, f"tensors[{i}] '{name}' unknown dtype={dtype!r}",
                      tensor_id=tid, tensor_name=name)

        # quantization metadata
        quant = rec.get("quantization", "none")
        group_size = rec.get("quantization_group_size")
        if quant == "per_group_q4":
            if group_size is None:
                res.error(6,
                          f"tensors[{i}] '{name}' quantization=per_group_q4 "
                          f"but quantization_group_size is absent",
                          tensor_id=tid, tensor_name=name)
            elif not isinstance(group_size, int) or group_size <= 0:
                res.error(6,
                          f"tensors[{i}] '{name}' quantization_group_size "
                          f"must be a positive integer; got {group_size!r}",
                          tensor_id=tid, tensor_name=name)

        # packed_bytes / scale_bytes non-negative
        for bfield in ("packed_bytes", "scale_bytes"):
            bv = rec.get(bfield)
            if bv is not None and (not isinstance(bv, int) or bv < 0):
                res.error(0, f"tensors[{i}] '{name}' {bfield} must be "
                             f"a non-negative integer; got {bv!r}",
                           tensor_id=tid, tensor_name=name)

        # owner_tile
        owner = rec.get("owner_tile")
        if owner is None:
            res.error(0, f"tensors[{i}] '{name}' missing 'owner_tile'",
                      tensor_id=tid, tensor_name=name)
        elif not isinstance(owner, int):
            res.error(0, f"tensors[{i}] '{name}' owner_tile must be integer; "
                         f"got {owner!r}",
                      tensor_id=tid, tensor_name=name)
        elif valid_tile_ids and owner not in valid_tile_ids:
            res.error(4, f"tensors[{i}] '{name}' owner_tile={owner} does not "
                         f"exist in tiles list",
                      tensor_id=tid, tensor_name=name)
        elif tile_count > 0 and isinstance(owner, int) and owner >= tile_count:
            res.error(4, f"tensors[{i}] '{name}' owner_tile={owner} >= "
                         f"tile_count={tile_count}",
                      tensor_id=tid, tensor_name=name)

        # slice fields
        slice_axis = rec.get("slice_axis")
        slice_start = rec.get("slice_start")
        slice_end = rec.get("slice_end")

        if slice_axis is not None:
            if not isinstance(slice_axis, int) or slice_axis < 0:
                res.error(0, f"tensors[{i}] '{name}' slice_axis must be a "
                             f"non-negative integer; got {slice_axis!r}",
                           tensor_id=tid, tensor_name=name)
            elif rank > 0 and slice_axis >= rank:
                res.error(0, f"tensors[{i}] '{name}' slice_axis={slice_axis} "
                             f"out of range for rank={rank}",
                           tensor_id=tid, tensor_name=name)

            # slice_start and slice_end required when slice_axis present
            if slice_start is None or slice_end is None:
                res.error(0, f"tensors[{i}] '{name}' slice_axis is set but "
                             f"slice_start or slice_end is absent",
                           tensor_id=tid, tensor_name=name)
            else:
                if not isinstance(slice_start, int) or slice_start < 0:
                    res.error(0, f"tensors[{i}] '{name}' slice_start must be "
                                 f"a non-negative integer; got {slice_start!r}",
                               tensor_id=tid, tensor_name=name)
                if not isinstance(slice_end, int) or slice_end <= 0:
                    res.error(0, f"tensors[{i}] '{name}' slice_end must be "
                                 f"a positive integer; got {slice_end!r}",
                               tensor_id=tid, tensor_name=name)
                elif isinstance(slice_start, int) and slice_end <= slice_start:
                    res.error(0, f"tensors[{i}] '{name}' slice_end={slice_end} "
                                 f"<= slice_start={slice_start}: empty range",
                               tensor_id=tid, tensor_name=name)
                else:
                    # slice_end must not exceed source_shape[slice_axis]
                    if (isinstance(source_shape, list) and
                            isinstance(slice_axis, int) and
                            0 <= slice_axis < len(source_shape)):
                        dim_size = source_shape[slice_axis]
                        if slice_end > dim_size:
                            res.error(0,
                                      f"tensors[{i}] '{name}' "
                                      f"slice_end={slice_end} > "
                                      f"source_shape[{slice_axis}]={dim_size}",
                                      tensor_id=tid, tensor_name=name)

                    # Q4 group alignment check (rule 6)
                    gs = group_size if group_size else _Q4_DEFAULT_GROUP_SIZE
                    if quant == "per_group_q4" or dtype == "q4":
                        gs_actual = gs if isinstance(gs, int) and gs > 0 \
                                    else _Q4_DEFAULT_GROUP_SIZE
                        if isinstance(slice_start, int) and \
                                slice_start % gs_actual != 0:
                            res.error(6,
                                      f"tensors[{i}] '{name}' slice_start="
                                      f"{slice_start} not aligned to q4 "
                                      f"group_size={gs_actual}",
                                      tensor_id=tid, tensor_name=name)
                        if isinstance(slice_end, int) and \
                                slice_end % gs_actual != 0 and \
                                not (isinstance(source_shape, list) and
                                     isinstance(slice_axis, int) and
                                     0 <= slice_axis < len(source_shape) and
                                     slice_end == source_shape[slice_axis]):
                            res.error(6,
                                      f"tensors[{i}] '{name}' slice_end="
                                      f"{slice_end} not aligned to q4 "
                                      f"group_size={gs_actual} and is not "
                                      f"end of dimension",
                                      tensor_id=tid, tensor_name=name)

        # replication_policy
        rep = rec.get("replication_policy", "unique")
        if rep not in VALID_REPLICATION_POLICIES:
            res.error(0, f"tensors[{i}] '{name}' replication_policy={rep!r} "
                         f"not in {sorted(VALID_REPLICATION_POLICIES)}",
                      tensor_id=tid, tensor_name=name)

        # reduction_behavior
        red = rec.get("reduction_behavior", "none")
        if red not in VALID_REDUCTION_BEHAVIORS:
            res.error(0, f"tensors[{i}] '{name}' reduction_behavior={red!r} "
                         f"not in {sorted(VALID_REDUCTION_BEHAVIORS)}",
                      tensor_id=tid, tensor_name=name)

        # routing_requirements
        rr = rec.get("routing_requirements", "none")
        if rr not in VALID_ROUTING_REQUIREMENTS:
            res.issue("warning", 0,
                      f"tensors[{i}] '{name}' routing_requirements={rr!r} "
                      f"not in recognised set; continuing",
                      tensor_id=tid, tensor_name=name)

        # placement_status
        ps = rec.get("placement_status", "placed")
        if ps not in VALID_PLACEMENT_STATUSES:
            res.error(0, f"tensors[{i}] '{name}' placement_status={ps!r} "
                         f"not in {sorted(VALID_PLACEMENT_STATUSES)}",
                      tensor_id=tid, tensor_name=name)

        # Unique-policy tensor duplicates (rule 2)
        if rep == "unique" and tid is not None:
            if tid in unique_seen:
                res.error(2, f"Duplicate 'unique' placement record for "
                             f"tensor_id={tid} ('{name}')",
                           tensor_id=tid, tensor_name=name)
            else:
                unique_seen[tid] = True

        if tid is not None:
            by_id[tid].append(rec)

    # ---------------------------------------------------------------------------
    # Coverage and overlap checks (per tensor_id)
    # ---------------------------------------------------------------------------
    for tid, records in by_id.items():
        if len(records) <= 1:
            continue

        # All records for a tensor must agree on slice_axis
        axes = {r.get("slice_axis") for r in records}
        rep_policies = {r.get("replication_policy", "unique") for r in records}

        # If all are replicated, skip overlap checks (rule 2 allows)
        if rep_policies == {"replicated"}:
            continue

        if len(axes) > 1:
            res.error(0, f"tensor_id={tid}: inconsistent slice_axis values "
                         f"across records: {axes}")
            continue

        axis = next(iter(axes))
        if axis is None:
            # Multiple whole-tensor records for non-replicated tensor
            policies = [r.get("replication_policy", "unique") for r in records]
            if any(p not in ("replicated",) for p in policies):
                res.error(2, f"tensor_id={tid}: multiple whole-tensor placement "
                             f"records but replication_policy is not 'replicated'")
            continue

        # Check for overlapping slices (rule 2)
        slices = []
        for r in records:
            s, e = r.get("slice_start"), r.get("slice_end")
            if isinstance(s, int) and isinstance(e, int) and e > s:
                slices.append((s, e, r.get("owner_tile")))

        slices.sort()
        for j in range(len(slices) - 1):
            (s0, e0, t0), (s1, e1, t1) = slices[j], slices[j + 1]
            if s1 < e0:
                res.error(2, f"tensor_id={tid}: overlapping slices "
                             f"[{s0},{e0}) on tile {t0} and "
                             f"[{s1},{e1}) on tile {t1}")

        # Check full coverage if tensor is declared as partial split (rule 3)
        source_shape = records[0].get("source_shape")
        if isinstance(source_shape, list) and isinstance(axis, int) \
                and 0 <= axis < len(source_shape):
            total_dim = source_shape[axis]
            covered = sorted({(s, e) for (s, e, _) in slices})
            if covered:
                # Merge ranges and check for gaps
                merged = [list(covered[0])]
                for (s, e) in covered[1:]:
                    if s <= merged[-1][1]:
                        merged[-1][1] = max(merged[-1][1], e)
                    else:
                        merged.append([s, e])
                if len(merged) > 1 or merged[0][0] != 0 or \
                        merged[0][1] != total_dim:
                    # Only warn — partial placement is valid during planning
                    placed_statuses = {r.get("placement_status", "placed")
                                       for r in records}
                    if placed_statuses == {"placed"}:
                        res.issue("warning", 3,
                                  f"tensor_id={tid}: slices do not fully cover "
                                  f"[0,{total_dim}); check for gaps or partial "
                                  f"placement")

        # Check reduction_behavior for split tensors (rule 8)
        reds = {r.get("reduction_behavior", "none") for r in records}
        if reds == {"none"} or reds == {None}:
            names_seen = {r.get("tensor_name", f"tensor_id={tid}")
                          for r in records}
            nm = next(iter(names_seen))
            res.issue("warning", 8,
                      f"tensor_id={tid} ('{nm}'): split across multiple tiles "
                      f"but reduction_behavior is 'none' for all slices; "
                      f"set reduction_behavior if a partial sum is required",
                      tensor_id=tid)


# ---------------------------------------------------------------------------
# Top-level validation entry point
# ---------------------------------------------------------------------------

def validate(report: dict, strict: bool = False) -> ValidationResult:
    res = ValidationResult(strict=strict)

    # 1. Header
    tile_count = validate_header(report, res)

    # 2. Required top-level lists present
    for key in ("tiles", "tensors"):
        if key not in report:
            res.issue("warning", 0, f"Top-level '{key}' list is absent; "
                                    f"no {key} will be checked")

    # 3. Tile records
    valid_tile_ids = validate_tiles(report, tile_count, res)

    # 4. Tensor placement records
    validate_tensors(report, valid_tile_ids, res, tile_count)

    return res


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def print_summary(path: str, report: dict, result: ValidationResult) -> None:
    header = report.get("header", {})
    tile_count = header.get("tile_count", "?")
    tiles = report.get("tiles", [])
    tensors = report.get("tensors", [])
    n_tiles = len(tiles)
    n_tensors = len(tensors)

    print(f"report path         : {path}")
    print(f"tile_count (header) : {tile_count}")
    print(f"tile records        : {n_tiles}")
    print(f"tensor records      : {n_tensors}")
    print(f"warnings            : {len(result.warnings)}")
    print(f"errors              : {len(result.errors)}")

    if result.warnings:
        print()
        print("Warnings:")
        for w in result.warnings:
            print(f"  [W rule={w['rule']}] {w['message']}")

    if result.errors:
        print()
        print("Errors:")
        for e in result.errors:
            print(f"  [E rule={e['rule']}] {e['message']}")

    print()
    if result.passed:
        print("validation: pass")
    else:
        print("validation: fail")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Validate an ATT-1 tensor-level placement report (M98 schema).")
    ap.add_argument("--report", required=True,
                    help="Path to placement report JSON file")
    ap.add_argument("--report-json", metavar="PATH",
                    help="Write JSON validation summary to PATH")
    ap.add_argument("--strict", action="store_true",
                    help="Treat warnings as errors")
    args = ap.parse_args(argv)

    # Parse JSON
    try:
        with open(args.report, encoding="utf-8") as fh:
            report = json.load(fh)
    except FileNotFoundError:
        print(f"error: report file not found: {args.report}", file=sys.stderr)
        sys.exit(2)
    except json.JSONDecodeError as exc:
        print(f"error: malformed JSON in {args.report}: {exc}", file=sys.stderr)
        sys.exit(2)

    if not isinstance(report, dict):
        print(f"error: report root must be a JSON object", file=sys.stderr)
        sys.exit(2)

    # Validate
    result = validate(report, strict=args.strict)

    # Human-readable output
    print_summary(args.report, report, result)

    # Optional JSON output
    if args.report_json:
        summary = result.summary()
        summary["report_path"] = args.report
        summary["tile_count_header"] = report.get("header", {}).get("tile_count")
        summary["tensor_record_count"] = len(report.get("tensors", []))
        summary["tile_record_count"] = len(report.get("tiles", []))
        try:
            with open(args.report_json, "w", encoding="utf-8") as fh:
                json.dump(summary, fh, indent=2)
                fh.write("\n")
        except OSError as exc:
            print(f"error: cannot write JSON output: {exc}", file=sys.stderr)
            sys.exit(2)

    sys.exit(0 if result.passed else 1)


if __name__ == "__main__":
    main()
