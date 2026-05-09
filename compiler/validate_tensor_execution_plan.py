#!/usr/bin/env python3
"""
ATT-1 tensor-level execution-plan validator (Milestone 128).

Reads an M125 execution-plan JSON and validates it for structural correctness,
phase ordering, command integrity, dependency consistency, fabric/reduction
requirements, and buffer field consistency.

This tool does NOT execute inference, change runtime behavior, access real
PCIe/MMIO registers, or implement a kernel driver.  All checking is advisory.

Exit codes:
  0 — validation passed (zero errors; warnings may be present)
  1 — validation failed (one or more errors detected)
  2 — plan could not be parsed (malformed JSON or missing required field)

Usage:
    python3 compiler/validate_tensor_execution_plan.py \\
        --plan compiler/fixtures/exec_plan_valid_tiny.json

    python3 compiler/validate_tensor_execution_plan.py \\
        --plan build/execution_plan.json \\
        --report-json build/validation_result.json

    python3 compiler/validate_tensor_execution_plan.py \\
        --plan build/execution_plan.json \\
        --strict
"""

import argparse
import json
import sys

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUPPORTED_PLAN_VERSIONS: set = {1}

# Valid execution phases (M125 §2 + M128 additions)
VALID_EXECUTION_PHASES: set = {
    "DEVICE_PROBE",
    "TILE_ENUMERATION",
    "MEMORY_ALLOCATE",
    "LOAD_TENSOR_TILE",
    "VALIDATE_TENSOR_TILE",
    "PREFILL_SETUP",
    "PREFILL_EXECUTION_PLAN",
    "DECODE_STEP_PLAN",
    "KV_APPEND",
    "KV_READ",
    "FABRIC_SEND",
    "FABRIC_REDUCE",
    "TRACE_SNAPSHOT",
    "QUERY_COUNTERS",
    "CLEANUP",
    "RESET",
}

# Recognized M105/M125 command types
VALID_COMMAND_TYPES: set = {
    "LOAD_TENSOR_TILE",
    "VALIDATE_TENSOR",
    "EXEC_MATMUL",
    "EXEC_RMSNORM",
    "EXEC_ROPE",
    "EXEC_ATTENTION",
    "EXEC_SWIGLU",
    "EXEC_SOFTMAX",
    "EXEC_RESIDUAL",
    "KV_APPEND",
    "KV_READ",
    "FABRIC_SEND",
    "FABRIC_REDUCE",
    "TILE_BARRIER",
    "TRACE_SNAPSHOT",
    "QUERY_COUNTERS",
}

# Command types that require at least one tensor_dependency
TENSOR_DEP_REQUIRED: set = {
    "LOAD_TENSOR_TILE",
    "VALIDATE_TENSOR",
    "EXEC_MATMUL",
    "EXEC_RMSNORM",
}

# Command types that require at least one input_buffer
INPUT_BUF_REQUIRED: set = {
    "EXEC_MATMUL",
    "EXEC_RMSNORM",
    "EXEC_ROPE",
    "EXEC_ATTENTION",
    "EXEC_SWIGLU",
    "EXEC_SOFTMAX",
    "EXEC_RESIDUAL",
    "VALIDATE_TENSOR",
    "KV_APPEND",
    "KV_READ",
    "FABRIC_SEND",
}

# Command types that require at least one output_buffer
OUTPUT_BUF_REQUIRED: set = {
    "EXEC_MATMUL",
    "EXEC_RMSNORM",
    "EXEC_ROPE",
    "EXEC_ATTENTION",
    "EXEC_SWIGLU",
    "EXEC_SOFTMAX",
    "EXEC_RESIDUAL",
    "LOAD_TENSOR_TILE",
    "KV_APPEND",
    "KV_READ",
    "FABRIC_REDUCE",
}

VALID_DTYPES: set = {"f32", "q8", "q4", "bf16"}

VALID_REGION_TYPES: set = {
    "TENSOR",
    "KV_CACHE",
    "STAGING",
    "DMA_BUFFER",
    "COMMAND_QUEUE",
    "COMPLETION_QUEUE",
    "TRACE_BUFFER",
    "FABRIC_BUFFER",
    "RESERVED",
}

# Accept both short (ok/error/warn) and M105-style status strings
VALID_EXPECTED_STATUSES: set = {
    "ok",
    "error",
    "warn",
    "ATT1_AIMU_ERR_OK",
    "ATT1_AIMU_ERR_UNSUPPORTED_OP",
    "ATT1_AIMU_ERR_INVALID_ARG",
    "ATT1_AIMU_ERR_TIMEOUT",
    "ATT1_AIMU_ERR_BUS_FAULT",
    "ATT1_AIMU_ERR_NO_MEM",
}

VALID_PLAN_STATUSES: set = {"pass", "fail", "warn", "ok", "error"}

VALID_TOKEN_PHASES: set = {"prefill", "decode"}

VALID_REDUCTION_BEHAVIORS: set = {
    "sum", "concat", "max", "pass_through", "topk",
}

# Phase ordering constraints: (must_come_before, must_come_after).
# Enforced only when both phases appear in the plan.
# "First occurrence of A must precede first occurrence of B."
PHASE_ORDER_CONSTRAINTS: list = [
    ("DEVICE_PROBE",             "TILE_ENUMERATION"),
    ("DEVICE_PROBE",             "MEMORY_ALLOCATE"),
    ("TILE_ENUMERATION",         "MEMORY_ALLOCATE"),
    ("MEMORY_ALLOCATE",          "LOAD_TENSOR_TILE"),
    ("LOAD_TENSOR_TILE",         "VALIDATE_TENSOR_TILE"),
    ("VALIDATE_TENSOR_TILE",     "PREFILL_SETUP"),
    ("VALIDATE_TENSOR_TILE",     "PREFILL_EXECUTION_PLAN"),
    ("VALIDATE_TENSOR_TILE",     "DECODE_STEP_PLAN"),
    ("PREFILL_SETUP",            "PREFILL_EXECUTION_PLAN"),
    ("PREFILL_SETUP",            "KV_APPEND"),
    ("PREFILL_SETUP",            "KV_READ"),
    ("PREFILL_EXECUTION_PLAN",   "CLEANUP"),
    ("DECODE_STEP_PLAN",         "CLEANUP"),
    ("PREFILL_EXECUTION_PLAN",   "RESET"),
    ("DECODE_STEP_PLAN",         "RESET"),
    ("CLEANUP",                  "RESET"),
]


# ---------------------------------------------------------------------------
# Result collector
# ---------------------------------------------------------------------------

class ValidationResult:
    def __init__(self, strict: bool = False) -> None:
        self.strict = strict
        self.warnings: list = []
        self.errors: list = []

    def warn(self, rule: int, message: str, **kw) -> None:
        rec = {"severity": "warning", "rule": rule, "message": message}
        rec.update(kw)
        self.warnings.append(rec)

    def error(self, rule: int, message: str, **kw) -> None:
        rec = {"severity": "error", "rule": rule, "message": message}
        rec.update(kw)
        self.errors.append(rec)

    def issue(self, severity: str, rule: int, message: str, **kw) -> None:
        """Emit warning, or error in strict mode."""
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
# Parse helper
# ---------------------------------------------------------------------------

class ParseError(Exception):
    pass


def _load_plan(path: str) -> dict:
    try:
        with open(path, "r") as fh:
            data = json.load(fh)
    except FileNotFoundError:
        raise ParseError(f"execution plan file not found: {path}")
    except json.JSONDecodeError as exc:
        raise ParseError(f"malformed JSON in execution plan: {exc}")
    if not isinstance(data, dict):
        raise ParseError("execution plan must be a JSON object at top level")
    return data


# ---------------------------------------------------------------------------
# §1  Header validation
# ---------------------------------------------------------------------------

def validate_header(plan: dict, res: ValidationResult) -> int:
    """Validate top-level plan header.  Returns tile_count (0 on error)."""

    # execution_plan_version
    ver = plan.get("execution_plan_version")
    if ver is None:
        res.error(1, "Missing 'execution_plan_version'")
    elif ver not in SUPPORTED_PLAN_VERSIONS:
        res.error(1, f"Unsupported execution_plan_version={ver!r}; "
                     f"supported: {sorted(SUPPORTED_PLAN_VERSIONS)}")

    # model_id / session_id
    if not plan.get("model_id"):
        res.error(1, "Missing or empty 'model_id'")
    if not plan.get("session_id"):
        res.error(1, "Missing or empty 'session_id'")

    # token_phase (optional but checked when present)
    tp = plan.get("token_phase")
    if tp is not None and tp not in VALID_TOKEN_PHASES:
        res.error(1, f"Invalid token_phase={tp!r}; "
                     f"expected one of {sorted(VALID_TOKEN_PHASES)}")

    # tile_count
    tile_count = plan.get("tile_count")
    if tile_count is None:
        res.error(1, "Missing 'tile_count'")
        tile_count = 0
    elif not isinstance(tile_count, int) or tile_count <= 0:
        res.error(1, f"tile_count must be a positive integer; got {tile_count!r}")
        tile_count = 0

    # layer_count (optional; checked for sanity when present)
    lc = plan.get("layer_count")
    if lc is not None and (not isinstance(lc, int) or lc < 0):
        res.error(1, f"layer_count must be a non-negative integer; got {lc!r}")

    # command_count vs actual list length
    commands = plan.get("commands")
    if commands is None:
        res.error(1, "Missing 'commands' list")
    elif not isinstance(commands, list):
        res.error(1, "'commands' must be a JSON array")
    else:
        declared = plan.get("command_count")
        if declared is not None and declared != len(commands):
            res.error(1, f"command_count={declared} does not match "
                         f"actual len(commands)={len(commands)}")

    # phases (optional top-level summary; validated as list if present)
    phases = plan.get("phases")
    if phases is not None and not isinstance(phases, list):
        res.error(1, "'phases' must be a JSON array when present")

    # status (optional; warn if unrecognised)
    status = plan.get("status")
    if status is not None and status not in VALID_PLAN_STATUSES:
        res.issue("warning", 1,
                  f"plan status={status!r} not in {sorted(VALID_PLAN_STATUSES)}")

    return int(tile_count) if isinstance(tile_count, int) and tile_count > 0 else 0


# ---------------------------------------------------------------------------
# §2  Buffer descriptor validation helper
# ---------------------------------------------------------------------------

def _validate_buffer_list(
    bufs: object,
    label: str,
    res: ValidationResult,
    rule: int,
) -> None:
    if not isinstance(bufs, list):
        res.error(rule, f"{label} must be a JSON array; got {type(bufs).__name__}")
        return
    for j, buf in enumerate(bufs):
        if not isinstance(buf, dict):
            res.error(rule, f"{label}[{j}] must be an object")
            continue
        rt = buf.get("region_type")
        if rt is None:
            res.error(rule, f"{label}[{j}] missing 'region_type'")
        elif rt not in VALID_REGION_TYPES:
            res.error(rule, f"{label}[{j}] unknown region_type={rt!r}")
        bs = buf.get("byte_size")
        if bs is None:
            res.error(rule, f"{label}[{j}] missing 'byte_size'")
        elif not isinstance(bs, int) or bs <= 0:
            res.error(rule,
                      f"{label}[{j}] byte_size must be a positive integer; "
                      f"got {bs!r}")
        dt = buf.get("dtype")
        if dt is not None and dt not in VALID_DTYPES:
            res.error(rule, f"{label}[{j}] unknown dtype={dt!r}")
        # Q4 metadata: warn (error in strict mode) if absent
        if dt == "q4":
            if buf.get("q4_group_size") is None and buf.get("q4_scale_bytes") is None:
                res.issue(
                    "warning", rule,
                    f"{label}[{j}] dtype=q4 but neither 'q4_group_size' nor "
                    f"'q4_scale_bytes' is present",
                )


# ---------------------------------------------------------------------------
# §3  FABRIC command-specific checks
# ---------------------------------------------------------------------------

def _validate_fabric_send(
    cmd: dict, idx: int, tile_count: int, res: ValidationResult,
) -> None:
    """FABRIC_SEND must reference a destination tile."""
    dst = cmd.get("dst_tile")
    routes = cmd.get("required_routes", [])
    if dst is None and (not isinstance(routes, list) or len(routes) == 0):
        res.error(12, f"commands[{idx}] FABRIC_SEND: missing 'dst_tile' "
                      f"or non-empty 'required_routes'")
    if dst is not None:
        if not isinstance(dst, int) or dst < 0:
            res.error(12, f"commands[{idx}] FABRIC_SEND: dst_tile must be "
                          f"a non-negative integer; got {dst!r}")
        elif tile_count > 0 and dst >= tile_count:
            res.error(12, f"commands[{idx}] FABRIC_SEND: dst_tile={dst} "
                          f">= tile_count={tile_count}")


def _validate_fabric_reduce(
    cmd: dict, idx: int, res: ValidationResult,
) -> None:
    """FABRIC_REDUCE must declare a reduction_behavior."""
    rb = cmd.get("reduction_behavior")
    if rb is None:
        res.error(13, f"commands[{idx}] FABRIC_REDUCE: "
                      f"missing 'reduction_behavior'")
    elif rb not in VALID_REDUCTION_BEHAVIORS:
        res.error(13, f"commands[{idx}] FABRIC_REDUCE: unknown "
                      f"reduction_behavior={rb!r}; "
                      f"expected one of {sorted(VALID_REDUCTION_BEHAVIORS)}")


# ---------------------------------------------------------------------------
# §4  Command record validation
# ---------------------------------------------------------------------------

def validate_commands(
    plan: dict,
    tile_count: int,
    res: ValidationResult,
) -> tuple:
    """
    Validate all command records.

    Returns:
      fence_to_cmd_idx : dict[int, int]  fence_id -> command list index
      cmd_idx_to_phase : dict[int, str]  command list index -> execution_phase
    """
    commands = plan.get("commands")
    if not isinstance(commands, list):
        return {}, {}

    layer_count = plan.get("layer_count")

    seen_ids: set = set()
    fence_to_cmd_idx: dict = {}
    cmd_idx_to_phase: dict = {}

    for i, cmd in enumerate(commands):
        if not isinstance(cmd, dict):
            res.error(2, f"commands[{i}] is not a JSON object")
            continue

        # --- plan_command_id: required, integer, unique ---
        cid = cmd.get("plan_command_id")
        if cid is None:
            res.error(2, f"commands[{i}] missing 'plan_command_id'")
            cid = None
        elif not isinstance(cid, int):
            res.error(2, f"commands[{i}].plan_command_id must be an integer; "
                         f"got {cid!r}")
            cid = None
        elif cid in seen_ids:
            res.error(2, f"commands[{i}] duplicate plan_command_id={cid}")
            cid = None
        if cid is not None:
            seen_ids.add(cid)

        # --- tile_id ---
        tid = cmd.get("tile_id")
        if tid is None:
            res.error(3, f"commands[{i}] missing 'tile_id'")
        elif not isinstance(tid, int) or tid < 0:
            res.error(3, f"commands[{i}].tile_id must be a non-negative "
                         f"integer; got {tid!r}")
        elif tile_count > 0 and tid >= tile_count:
            res.error(3, f"commands[{i}].tile_id={tid} >= tile_count={tile_count}")

        # --- layer_id (optional; -1 means not layer-specific) ---
        lid = cmd.get("layer_id")
        if lid is not None:
            if not isinstance(lid, int) or lid < -1:
                res.error(3, f"commands[{i}].layer_id must be -1 or a "
                             f"non-negative integer; got {lid!r}")
            elif (layer_count is not None and isinstance(layer_count, int)
                  and lid >= 0 and lid >= layer_count):
                res.error(3, f"commands[{i}].layer_id={lid} >= "
                             f"layer_count={layer_count}")

        # --- execution_phase ---
        phase = cmd.get("execution_phase")
        if phase is None:
            res.error(4, f"commands[{i}] missing 'execution_phase'")
        elif phase not in VALID_EXECUTION_PHASES:
            res.error(4, f"commands[{i}] unknown execution_phase={phase!r}")
        else:
            cmd_idx_to_phase[i] = phase

        # --- command_type ---
        ct = cmd.get("command_type")
        if ct is None:
            res.error(5, f"commands[{i}] missing 'command_type'")
        elif ct not in VALID_COMMAND_TYPES:
            res.error(5, f"commands[{i}] unknown command_type={ct!r}")
        else:
            # tensor_dependencies
            tdeps = cmd.get("tensor_dependencies", [])
            if ct in TENSOR_DEP_REQUIRED:
                if not isinstance(tdeps, list) or len(tdeps) == 0:
                    res.error(6, f"commands[{i}] command_type={ct} requires "
                                 f"at least one tensor_dependency")
                elif any(not isinstance(d, str) or not d for d in tdeps):
                    res.error(6, f"commands[{i}] tensor_dependencies contains "
                                 f"a non-string or empty entry")
            # FABRIC_SEND / FABRIC_REDUCE specifics
            if ct == "FABRIC_SEND":
                _validate_fabric_send(cmd, i, tile_count, res)
            if ct == "FABRIC_REDUCE":
                _validate_fabric_reduce(cmd, i, res)

        # --- input_buffers ---
        ibufs = cmd.get("input_buffers", [])
        if ct in INPUT_BUF_REQUIRED and (
            not isinstance(ibufs, list) or len(ibufs) == 0
        ):
            res.error(7, f"commands[{i}] command_type={ct} requires "
                         f"at least one input_buffer")
        if isinstance(ibufs, list) and ibufs:
            _validate_buffer_list(ibufs, f"commands[{i}].input_buffers", res, 7)

        # --- output_buffers ---
        obufs = cmd.get("output_buffers", [])
        if ct in OUTPUT_BUF_REQUIRED and (
            not isinstance(obufs, list) or len(obufs) == 0
        ):
            res.error(7, f"commands[{i}] command_type={ct} requires "
                         f"at least one output_buffer")
        if isinstance(obufs, list) and obufs:
            _validate_buffer_list(obufs, f"commands[{i}].output_buffers", res, 7)

        # --- expected_status ---
        es = cmd.get("expected_status")
        if es is None:
            res.issue("warning", 8, f"commands[{i}] missing 'expected_status'")
        elif es not in VALID_EXPECTED_STATUSES:
            res.error(8, f"commands[{i}] unknown expected_status={es!r}")

        # --- trace_flags ---
        tf = cmd.get("trace_flags")
        if tf is not None and (not isinstance(tf, int) or tf < 0):
            res.error(9, f"commands[{i}].trace_flags must be a non-negative "
                         f"integer; got {tf!r}")

        # --- fence_id ---
        fid = cmd.get("fence_id")
        if fid is None:
            res.issue("warning", 10, f"commands[{i}] missing 'fence_id'")
        elif not isinstance(fid, int) or fid < 0:
            res.error(10, f"commands[{i}].fence_id must be a non-negative "
                          f"integer; got {fid!r}")
        else:
            if fid != 0 and fid in fence_to_cmd_idx:
                res.issue(
                    "warning", 10,
                    f"commands[{i}] fence_id={fid} already used by "
                    f"commands[{fence_to_cmd_idx[fid]}]",
                )
            else:
                fence_to_cmd_idx[fid] = i

        # --- required_routes / required_reductions ---
        rr = cmd.get("required_routes")
        if rr is not None and not isinstance(rr, list):
            res.error(11, f"commands[{i}].required_routes must be a list; "
                          f"got {type(rr).__name__}")
        rrr = cmd.get("required_reductions")
        if rrr is not None and not isinstance(rrr, list):
            res.error(11, f"commands[{i}].required_reductions must be a list; "
                          f"got {type(rrr).__name__}")

    return fence_to_cmd_idx, cmd_idx_to_phase


# ---------------------------------------------------------------------------
# §5  Phase ordering validation
# ---------------------------------------------------------------------------

def validate_phase_ordering(
    cmd_idx_to_phase: dict,
    res: ValidationResult,
) -> None:
    """Check that execution phases appear in a sane order."""
    phase_first: dict = {}
    for idx, phase in cmd_idx_to_phase.items():
        if phase not in phase_first:
            phase_first[phase] = idx

    for before, after in PHASE_ORDER_CONSTRAINTS:
        if before not in phase_first or after not in phase_first:
            continue
        if phase_first[before] > phase_first[after]:
            res.error(
                20,
                f"Phase ordering violation: first '{before}' command "
                f"(index {phase_first[before]}) appears after first "
                f"'{after}' command (index {phase_first[after]})",
            )


# ---------------------------------------------------------------------------
# §6  Dependency graph validation
# ---------------------------------------------------------------------------

def validate_dependencies(
    plan: dict,
    fence_to_cmd_idx: dict,
    res: ValidationResult,
) -> None:
    """Validate dependency_fence_id is forward-only and the graph is acyclic."""
    commands = plan.get("commands")
    if not isinstance(commands, list):
        return

    # Build adjacency: cmd_idx -> set of cmd_idx it depends on
    dep_graph: dict = {i: set() for i in range(len(commands))}

    for i, cmd in enumerate(commands):
        if not isinstance(cmd, dict):
            continue
        dep_fid = cmd.get("dependency_fence_id")
        if dep_fid is None or dep_fid == 0:
            continue
        if not isinstance(dep_fid, int):
            res.error(10, f"commands[{i}].dependency_fence_id must be an "
                          f"integer; got {dep_fid!r}")
            continue

        dep_idx = fence_to_cmd_idx.get(dep_fid)
        if dep_idx is None:
            res.error(10, f"commands[{i}].dependency_fence_id={dep_fid} does "
                          f"not match any fence_id in the plan")
            continue

        # Forward-only: dep must come before this command
        if dep_idx >= i:
            cid = cmd.get("plan_command_id", "?")
            res.error(
                10,
                f"commands[{i}] (plan_command_id={cid}) has "
                f"dependency_fence_id={dep_fid} which belongs to "
                f"commands[{dep_idx}] — a command that has not yet "
                f"appeared (dependency on future command)",
            )
            continue

        dep_graph[i].add(dep_idx)

    # Cycle detection via iterative DFS
    WHITE, GRAY, BLACK = 0, 1, 2
    color = [WHITE] * len(commands)
    cycle_reported: set = set()

    def dfs(start: int) -> None:
        stack = [(start, iter(dep_graph.get(start, set())))]
        path = [start]
        color[start] = GRAY
        while stack:
            node, children = stack[-1]
            try:
                child = next(children)
                if color[child] == GRAY:
                    pair = (min(node, child), max(node, child))
                    if pair not in cycle_reported:
                        cycle_reported.add(pair)
                        res.error(
                            10,
                            f"Cyclic dependency detected: "
                            f"commands[{node}] and commands[{child}] "
                            f"form a cycle",
                        )
                elif color[child] == WHITE:
                    color[child] = GRAY
                    path.append(child)
                    stack.append((child, iter(dep_graph.get(child, set()))))
            except StopIteration:
                color[node] = BLACK
                path.pop()
                stack.pop()

    for node in range(len(commands)):
        if color[node] == WHITE:
            dfs(node)


# ---------------------------------------------------------------------------
# §7  Cross-tile split / reduction validation
# ---------------------------------------------------------------------------

def validate_split_reductions(plan: dict, res: ValidationResult) -> None:
    """
    EXEC_MATMUL commands that declare required_reductions must have at least
    one FABRIC_REDUCE command present in the plan.
    """
    commands = plan.get("commands")
    if not isinstance(commands, list):
        return

    has_reduce = any(
        isinstance(cmd, dict) and cmd.get("command_type") == "FABRIC_REDUCE"
        for cmd in commands
    )

    for i, cmd in enumerate(commands):
        if not isinstance(cmd, dict):
            continue
        if cmd.get("command_type") != "EXEC_MATMUL":
            continue
        rr = cmd.get("required_reductions", [])
        if isinstance(rr, list) and len(rr) > 0 and not has_reduce:
            res.error(
                13,
                f"commands[{i}] EXEC_MATMUL has required_reductions "
                f"but no FABRIC_REDUCE command found in the plan",
            )


# ---------------------------------------------------------------------------
# §8  Human-readable report
# ---------------------------------------------------------------------------

def _print_report(
    plan_path: str,
    plan: dict,
    res: ValidationResult,
    phase_set: set,
    route_count: int,
    buf_count: int,
    cmd_count: int,
) -> None:
    print(f"execution-plan validator  path={plan_path}")
    print(f"  plan_version      : {plan.get('execution_plan_version', '?')}")
    print(f"  model_id          : {plan.get('model_id', '?')}")
    print(f"  session_id        : {plan.get('session_id', '?')}")
    print(f"  token_phase       : {plan.get('token_phase', '?')}")
    print(f"  tile_count        : {plan.get('tile_count', '?')}")
    print(f"  layer_count       : {plan.get('layer_count', '?')}")
    print(f"  phase_count       : {len(phase_set)}")
    print(f"  command_count     : {cmd_count}")
    print(f"  buffer_ref_count  : {buf_count}")
    print(f"  route_ref_count   : {route_count}")
    print(f"  warning_count     : {len(res.warnings)}")
    print(f"  failure_count     : {len(res.errors)}")
    if res.warnings:
        print("warnings:")
        for w in res.warnings:
            print(f"  [W{w['rule']:02d}] {w['message']}")
    if res.errors:
        print("failures:")
        for e in res.errors:
            print(f"  [E{e['rule']:02d}] {e['message']}")
    print(f"status: {'PASS' if res.passed else 'FAIL'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Validate an ATT-1 M125 tensor-level execution-plan JSON."
    )
    parser.add_argument(
        "--plan", required=True, metavar="PATH",
        help="Path to the execution-plan JSON to validate",
    )
    parser.add_argument(
        "--report-json", metavar="PATH",
        help="Write JSON validation summary to this file",
    )
    parser.add_argument(
        "--strict", action="store_true",
        help="Promote warnings to errors",
    )
    args = parser.parse_args(argv)

    try:
        plan = _load_plan(args.plan)
    except ParseError as exc:
        print(f"PARSE ERROR: {exc}", file=sys.stderr)
        return 2

    res = ValidationResult(strict=args.strict)

    tile_count = validate_header(plan, res)
    fence_to_cmd_idx, cmd_idx_to_phase = validate_commands(plan, tile_count, res)
    validate_phase_ordering(cmd_idx_to_phase, res)
    validate_dependencies(plan, fence_to_cmd_idx, res)
    validate_split_reductions(plan, res)

    # Collect summary counters
    phase_set = set(cmd_idx_to_phase.values())
    commands = plan.get("commands") or []
    cmd_count = len(commands) if isinstance(commands, list) else 0
    route_count = 0
    buf_count = 0
    for cmd in (commands if isinstance(commands, list) else []):
        if isinstance(cmd, dict):
            rr = cmd.get("required_routes")
            if isinstance(rr, list):
                route_count += len(rr)
            buf_count += len(cmd.get("input_buffers") or [])
            buf_count += len(cmd.get("output_buffers") or [])

    _print_report(args.plan, plan, res, phase_set, route_count, buf_count, cmd_count)

    if args.report_json:
        summary = res.summary()
        summary.update({
            "plan_path": args.plan,
            "phase_count": len(phase_set),
            "command_count": cmd_count,
            "route_ref_count": route_count,
            "buffer_ref_count": buf_count,
        })
        with open(args.report_json, "w") as fh:
            json.dump(summary, fh, indent=2)

    return 0 if res.passed else 1


if __name__ == "__main__":
    sys.exit(main())
