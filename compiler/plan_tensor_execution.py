#!/usr/bin/env python3
"""
ATT-1 tensor-level placement execution planner (Milestone 125).

Translates a loaded/validated tensor placement report (M98/M100) into an
advisory per-layer, per-tile EXEC_* command sequence for a single token step
(prefill or decode).  Optionally enriches the plan using an M109 command plan
and an M115/M117 fabric route report.

This tool does NOT execute inference, change runtime behavior, access real
PCIe/MMIO registers, or implement a kernel driver.  All output is advisory.

Exit codes:
  0 — plan generated successfully (warnings may be present)
  1 — planning errors, or strict mode + warnings
  2 — parse error (malformed JSON or missing required field)

Usage:
    python3 compiler/plan_tensor_execution.py \\
        --placement-report compiler/fixtures/placement_report_valid.json

    python3 compiler/plan_tensor_execution.py \\
        --placement-report build/placement.json \\
        --command-plan build/command_plan.json \\
        --route-report build/routes.json \\
        --token-phase decode \\
        --plan-json build/execution_plan.json \\
        --strict
"""

import argparse
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXECUTION_PLAN_VERSION = 1

SUPPORTED_REPORT_VERSIONS   = {1}
SUPPORTED_PLACEMENT_POLICIES = {
    "layer_wise", "tensor_wise", "row_split", "column_split",
    "head_wise", "lm_head_split", "embedding_split",
}

# M105 EXEC command types used in plan generation
CMD_LOAD_TENSOR     = "LOAD_TENSOR_TILE"
CMD_VALIDATE_TENSOR = "VALIDATE_TENSOR"
CMD_EXEC_MATMUL     = "EXEC_MATMUL"
CMD_EXEC_RMSNORM    = "EXEC_RMSNORM"
CMD_EXEC_ROPE       = "EXEC_ROPE"
CMD_EXEC_ATTENTION  = "EXEC_ATTENTION"
CMD_EXEC_SWIGLU     = "EXEC_SWIGLU"
CMD_EXEC_SOFTMAX    = "EXEC_SOFTMAX"
CMD_EXEC_RESIDUAL   = "EXEC_RESIDUAL"
CMD_KV_APPEND       = "KV_APPEND"
CMD_KV_READ         = "KV_READ"
CMD_FABRIC_SEND     = "FABRIC_SEND"
CMD_FABRIC_REDUCE   = "FABRIC_REDUCE"
CMD_TILE_BARRIER    = "TILE_BARRIER"
CMD_TRACE_SNAPSHOT  = "TRACE_SNAPSHOT"
CMD_QUERY_COUNTERS  = "QUERY_COUNTERS"

# Execution phases from M125 §2
PHASE_MEMORY_ALLOCATE   = "MEMORY_ALLOCATE"
PHASE_LOAD_TENSOR       = "LOAD_TENSOR_TILE"
PHASE_VALIDATE_TENSOR   = "VALIDATE_TENSOR_TILE"
PHASE_PREFILL_SETUP     = "PREFILL_SETUP"
PHASE_PREFILL_EXEC      = "PREFILL_EXECUTION_PLAN"
PHASE_DECODE_STEP       = "DECODE_STEP_PLAN"
PHASE_KV_APPEND         = "KV_APPEND"
PHASE_KV_READ           = "KV_READ"
PHASE_FABRIC_SEND       = "FABRIC_SEND"
PHASE_FABRIC_REDUCE     = "FABRIC_REDUCE"
PHASE_TRACE             = "TRACE_SNAPSHOT"
PHASE_QUERY             = "QUERY_COUNTERS"
PHASE_CLEANUP           = "CLEANUP"

# Tensor categories that represent norm weights (can be replicated)
NORM_CATEGORIES = {"norm", "attention_norm", "ffn_norm", "output_norm"}

# Reduction behaviors for cross-tile ops
REDUCTION_SUM    = "sum"
REDUCTION_CONCAT = "concat"
REDUCTION_PASS   = "pass_through"

# ---------------------------------------------------------------------------
# ParseError
# ---------------------------------------------------------------------------

class ParseError(Exception):
    pass

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_json(path: str, label: str) -> dict:
    try:
        with open(path, "r") as fh:
            return json.load(fh)
    except FileNotFoundError:
        raise ParseError(f"{label}: file not found: {path}")
    except json.JSONDecodeError as exc:
        raise ParseError(f"{label}: malformed JSON: {exc}")


def _load_placement_report(path: str) -> dict:
    report = _load_json(path, "placement report")
    version = report.get("report_version")
    if version not in SUPPORTED_REPORT_VERSIONS:
        raise ParseError(
            f"placement report: unsupported report_version={version!r}; "
            f"supported={sorted(SUPPORTED_REPORT_VERSIONS)}"
        )
    if "header" not in report:
        raise ParseError("placement report: missing 'header' key")
    if "tensors" not in report:
        raise ParseError("placement report: missing 'tensors' key")
    return report


def _load_command_plan(path: str) -> dict:
    return _load_json(path, "command plan")


def _load_route_report(path: str) -> dict:
    return _load_json(path, "route report")

# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _validate_placement(report: dict, tile_count: int, notes: list) -> list:
    """Return list of error strings for hard planning failures."""
    errors = []
    tensors = report.get("tensors", [])
    for t in tensors:
        owner = t.get("owner_tile")
        if owner is None:
            errors.append(
                f"tensor '{t.get('tensor_name')}': missing owner_tile"
            )
        elif owner >= tile_count:
            errors.append(
                f"tensor '{t.get('tensor_name')}': owner_tile {owner} "
                f">= tile_count {tile_count}"
            )
        status = t.get("placement_status", "placed")
        if status not in ("placed",):
            notes.append(
                f"WARN: tensor '{t.get('tensor_name')}': "
                f"placement_status={status!r} (expected 'placed')"
            )
        reduction = t.get("reduction_behavior", "none")
        route_req  = t.get("routing_requirements", "none")
        if route_req not in ("none",) and reduction not in (
            "none", "sum", "concat", "max", "topk", "pass_through"
        ):
            errors.append(
                f"tensor '{t.get('tensor_name')}': "
                f"routing_requirements={route_req!r} but "
                f"reduction_behavior={reduction!r} is unrecognised"
            )
    return errors

# ---------------------------------------------------------------------------
# Plan generation
# ---------------------------------------------------------------------------

def _dtype_bytes(dtype: str) -> int:
    return {"f32": 4, "q8": 1, "q4": 1}.get(dtype, 4)


def _make_cmd(
    next_id: list,
    tile_id: int,
    layer_id: int,
    token_phase: str,
    execution_phase: str,
    command_type: str,
    tensor_deps: list,
    input_bufs: list,
    output_bufs: list,
    required_routes: list,
    required_reductions: list,
    fence_id: int,
    dep_fence_id: int,
    expected_status: str = "ok",
    trace_flags: int = 0,
) -> dict:
    cmd_id = next_id[0]
    next_id[0] += 1
    return {
        "plan_command_id": cmd_id,
        "tile_id": tile_id,
        "layer_id": layer_id,
        "token_phase": token_phase,
        "execution_phase": execution_phase,
        "command_type": command_type,
        "tensor_dependencies": tensor_deps,
        "input_buffers": input_bufs,
        "output_buffers": output_bufs,
        "required_routes": required_routes,
        "required_reductions": required_reductions,
        "fence_id": fence_id,
        "dependency_fence_id": dep_fence_id,
        "expected_status": expected_status,
        "trace_flags": trace_flags,
    }


def _buf(region_type: str, byte_size: int, dtype: str) -> dict:
    return {"region_type": region_type, "byte_size": byte_size, "dtype": dtype}


def _generate_load_phase(
    tensors: list, next_id: list, next_fence: list, token_phase: str
) -> list:
    """Generate LOAD_TENSOR_TILE + VALIDATE_TENSOR commands per tile."""
    cmds = []
    # Group tensors by owner tile, sort deterministically
    from collections import defaultdict
    by_tile = defaultdict(list)
    for t in tensors:
        by_tile[t.get("owner_tile", 0)].append(t)

    for tile_id in sorted(by_tile.keys()):
        tile_tensors = sorted(by_tile[tile_id], key=lambda t: t.get("tensor_id", 0))
        for t in tile_tensors:
            f = next_fence[0]
            next_fence[0] += 1
            cmds.append(_make_cmd(
                next_id, tile_id, -1, token_phase, PHASE_LOAD_TENSOR,
                CMD_LOAD_TENSOR,
                tensor_deps=[t.get("tensor_name", "")],
                input_bufs=[],
                output_bufs=[_buf("TENSOR", t.get("packed_bytes", 0),
                                   t.get("dtype", "f32"))],
                required_routes=[], required_reductions=[],
                fence_id=f, dep_fence_id=0,
            ))
        # VALIDATE after all loads for this tile
        for t in tile_tensors:
            f = next_fence[0]
            next_fence[0] += 1
            cmds.append(_make_cmd(
                next_id, tile_id, -1, token_phase, PHASE_VALIDATE_TENSOR,
                CMD_VALIDATE_TENSOR,
                tensor_deps=[t.get("tensor_name", "")],
                input_bufs=[_buf("TENSOR", t.get("packed_bytes", 0),
                                  t.get("dtype", "f32"))],
                output_bufs=[],
                required_routes=[], required_reductions=[],
                fence_id=f, dep_fence_id=0,
            ))
    return cmds


def _generate_prefill_setup(
    tile_count: int, next_id: list, next_fence: list,
    token_phase: str, route_ids_by_type: dict
) -> list:
    """TILE_BARRIER on all tiles, plus TRACE_SNAPSHOT."""
    cmds = []
    barrier_routes = route_ids_by_type.get("TILE_BARRIER", [])
    f = next_fence[0]
    next_fence[0] += 1
    for tile_id in range(tile_count):
        cmds.append(_make_cmd(
            next_id, tile_id, -1, token_phase, PHASE_PREFILL_SETUP,
            CMD_TILE_BARRIER,
            tensor_deps=[],
            input_bufs=[], output_bufs=[],
            required_routes=barrier_routes,
            required_reductions=[],
            fence_id=f, dep_fence_id=0,
        ))
    # TRACE_SNAPSHOT after setup
    f2 = next_fence[0]
    next_fence[0] += 1
    for tile_id in range(tile_count):
        cmds.append(_make_cmd(
            next_id, tile_id, -1, token_phase, PHASE_TRACE,
            CMD_TRACE_SNAPSHOT,
            tensor_deps=[], input_bufs=[], output_bufs=[],
            required_routes=[], required_reductions=[],
            fence_id=f2, dep_fence_id=f, trace_flags=1,
        ))
    return cmds


def _generate_layer_plan(
    layer_id: int,
    header: dict,
    tensors_by_name: dict,
    placement_policy: str,
    token_phase: str,
    next_id: list,
    next_fence: list,
    route_ids_by_type: dict,
    notes: list,
) -> list:
    """Generate the command sequence for one transformer layer."""
    cmds = []
    d_model  = header.get("d_model", 64)
    dtype    = header.get("dtype", "f32")
    db       = _dtype_bytes(dtype)
    act_bytes = d_model * db

    # Determine which tile owns this layer's tensors
    # Simple heuristic: pick the tile that owns the attention norm for this layer
    attn_norm_name = f"layers.{layer_id}.attention_norm.weight"
    attn_norm = tensors_by_name.get(attn_norm_name)
    if attn_norm:
        tile_id = attn_norm.get("owner_tile", 0)
    else:
        # Fall back: distribute layers across tiles round-robin
        tile_id = layer_id % max(header.get("tile_count", 1), 1)

    exec_phase = PHASE_PREFILL_EXEC if token_phase == "prefill" else PHASE_DECODE_STEP
    partial_reduce_routes = route_ids_by_type.get("PARTIAL_REDUCE", [])
    logits_reduce_routes  = route_ids_by_type.get("LOGITS_REDUCE",  [])
    activation_routes     = route_ids_by_type.get("ACTIVATION_SEND", [])

    dep_fence = 0

    # --- Attention sublayer ---
    # 1. attention_norm (RMSNorm)
    f = next_fence[0]; next_fence[0] += 1
    cmds.append(_make_cmd(
        next_id, tile_id, layer_id, token_phase, exec_phase,
        CMD_EXEC_RMSNORM,
        tensor_deps=[f"layers.{layer_id}.attention_norm.weight"],
        input_bufs=[_buf("STAGING", act_bytes, dtype)],
        output_bufs=[_buf("STAGING", act_bytes, dtype)],
        required_routes=[], required_reductions=[],
        fence_id=f, dep_fence_id=dep_fence,
    ))
    dep_fence = f

    # 2. Q matmul
    f = next_fence[0]; next_fence[0] += 1
    cmds.append(_make_cmd(
        next_id, tile_id, layer_id, token_phase, exec_phase,
        CMD_EXEC_MATMUL,
        tensor_deps=[f"layers.{layer_id}.attention.wq.weight"],
        input_bufs=[_buf("STAGING", act_bytes, dtype)],
        output_bufs=[_buf("STAGING", act_bytes, dtype)],
        required_routes=[], required_reductions=[],
        fence_id=f, dep_fence_id=dep_fence,
    ))
    dep_fence = f

    # 3. K matmul
    f = next_fence[0]; next_fence[0] += 1
    cmds.append(_make_cmd(
        next_id, tile_id, layer_id, token_phase, exec_phase,
        CMD_EXEC_MATMUL,
        tensor_deps=[f"layers.{layer_id}.attention.wk.weight"],
        input_bufs=[_buf("STAGING", act_bytes, dtype)],
        output_bufs=[_buf("STAGING", act_bytes, dtype)],
        required_routes=[], required_reductions=[],
        fence_id=f, dep_fence_id=dep_fence,
    ))
    dep_fence = f

    # 4. V matmul
    f = next_fence[0]; next_fence[0] += 1
    cmds.append(_make_cmd(
        next_id, tile_id, layer_id, token_phase, exec_phase,
        CMD_EXEC_MATMUL,
        tensor_deps=[f"layers.{layer_id}.attention.wv.weight"],
        input_bufs=[_buf("STAGING", act_bytes, dtype)],
        output_bufs=[_buf("STAGING", act_bytes, dtype)],
        required_routes=[], required_reductions=[],
        fence_id=f, dep_fence_id=dep_fence,
    ))
    dep_fence = f

    # 5. RoPE Q
    f = next_fence[0]; next_fence[0] += 1
    cmds.append(_make_cmd(
        next_id, tile_id, layer_id, token_phase, exec_phase,
        CMD_EXEC_ROPE,
        tensor_deps=[],
        input_bufs=[_buf("STAGING", act_bytes, dtype)],
        output_bufs=[_buf("STAGING", act_bytes, dtype)],
        required_routes=[], required_reductions=[],
        fence_id=f, dep_fence_id=dep_fence,
    ))
    dep_fence = f

    # 6. RoPE K
    f = next_fence[0]; next_fence[0] += 1
    cmds.append(_make_cmd(
        next_id, tile_id, layer_id, token_phase, exec_phase,
        CMD_EXEC_ROPE,
        tensor_deps=[],
        input_bufs=[_buf("STAGING", act_bytes, dtype)],
        output_bufs=[_buf("STAGING", act_bytes, dtype)],
        required_routes=[], required_reductions=[],
        fence_id=f, dep_fence_id=dep_fence,
    ))
    dep_fence = f

    # 7. KV append
    f = next_fence[0]; next_fence[0] += 1
    cmds.append(_make_cmd(
        next_id, tile_id, layer_id, token_phase, PHASE_KV_APPEND,
        CMD_KV_APPEND,
        tensor_deps=[],
        input_bufs=[_buf("STAGING", act_bytes, dtype)],
        output_bufs=[_buf("KV_CACHE", act_bytes, dtype)],
        required_routes=[], required_reductions=[],
        fence_id=f, dep_fence_id=dep_fence,
    ))
    dep_fence = f

    # 8. KV read
    f = next_fence[0]; next_fence[0] += 1
    cmds.append(_make_cmd(
        next_id, tile_id, layer_id, token_phase, PHASE_KV_READ,
        CMD_KV_READ,
        tensor_deps=[],
        input_bufs=[_buf("KV_CACHE", act_bytes, dtype)],
        output_bufs=[_buf("STAGING", act_bytes, dtype)],
        required_routes=[], required_reductions=[],
        fence_id=f, dep_fence_id=dep_fence,
    ))
    dep_fence = f

    # 9. Attention scores + values
    f = next_fence[0]; next_fence[0] += 1
    # head-wise split → PARTIAL_REDUCE; layer-wise → local
    req_routes = partial_reduce_routes if placement_policy == "head_wise" else []
    req_reductions = [1] if placement_policy == "head_wise" else []
    if placement_policy == "head_wise" and not req_routes:
        notes.append(
            f"WARN layer {layer_id}: head_wise split but no PARTIAL_REDUCE "
            "routes found in route report"
        )
    cmds.append(_make_cmd(
        next_id, tile_id, layer_id, token_phase, exec_phase,
        CMD_EXEC_ATTENTION,
        tensor_deps=[],
        input_bufs=[_buf("STAGING", act_bytes, dtype)],
        output_bufs=[_buf("STAGING", act_bytes, dtype)],
        required_routes=req_routes,
        required_reductions=req_reductions,
        fence_id=f, dep_fence_id=dep_fence,
    ))
    dep_fence = f

    # 10. Output projection
    f = next_fence[0]; next_fence[0] += 1
    cmds.append(_make_cmd(
        next_id, tile_id, layer_id, token_phase, exec_phase,
        CMD_EXEC_MATMUL,
        tensor_deps=[f"layers.{layer_id}.attention.wo.weight"],
        input_bufs=[_buf("STAGING", act_bytes, dtype)],
        output_bufs=[_buf("STAGING", act_bytes, dtype)],
        required_routes=[], required_reductions=[],
        fence_id=f, dep_fence_id=dep_fence,
    ))
    dep_fence = f

    # 11. Residual add (attention)
    f = next_fence[0]; next_fence[0] += 1
    res_routes = activation_routes if placement_policy == "layer_wise" else []
    cmds.append(_make_cmd(
        next_id, tile_id, layer_id, token_phase, exec_phase,
        CMD_EXEC_RESIDUAL,
        tensor_deps=[],
        input_bufs=[_buf("STAGING", act_bytes, dtype)] * 2,
        output_bufs=[_buf("STAGING", act_bytes, dtype)],
        required_routes=res_routes, required_reductions=[],
        fence_id=f, dep_fence_id=dep_fence,
    ))
    dep_fence = f

    # --- FFN sublayer ---
    # 12. ffn_norm (RMSNorm)
    f = next_fence[0]; next_fence[0] += 1
    cmds.append(_make_cmd(
        next_id, tile_id, layer_id, token_phase, exec_phase,
        CMD_EXEC_RMSNORM,
        tensor_deps=[f"layers.{layer_id}.ffn_norm.weight"],
        input_bufs=[_buf("STAGING", act_bytes, dtype)],
        output_bufs=[_buf("STAGING", act_bytes, dtype)],
        required_routes=[], required_reductions=[],
        fence_id=f, dep_fence_id=dep_fence,
    ))
    dep_fence = f

    ffn_hidden = header.get("ffn_hidden", 128)
    ffn_bytes  = ffn_hidden * db

    # 13. Gate projection
    f = next_fence[0]; next_fence[0] += 1
    cmds.append(_make_cmd(
        next_id, tile_id, layer_id, token_phase, exec_phase,
        CMD_EXEC_MATMUL,
        tensor_deps=[f"layers.{layer_id}.feed_forward.w1.weight"],
        input_bufs=[_buf("STAGING", act_bytes, dtype)],
        output_bufs=[_buf("STAGING", ffn_bytes, dtype)],
        required_routes=[], required_reductions=[],
        fence_id=f, dep_fence_id=dep_fence,
    ))
    dep_fence = f

    # 14. Up projection
    f = next_fence[0]; next_fence[0] += 1
    cmds.append(_make_cmd(
        next_id, tile_id, layer_id, token_phase, exec_phase,
        CMD_EXEC_MATMUL,
        tensor_deps=[f"layers.{layer_id}.feed_forward.w3.weight"],
        input_bufs=[_buf("STAGING", act_bytes, dtype)],
        output_bufs=[_buf("STAGING", ffn_bytes, dtype)],
        required_routes=[], required_reductions=[],
        fence_id=f, dep_fence_id=dep_fence,
    ))
    dep_fence = f

    # 15. SwiGLU
    f = next_fence[0]; next_fence[0] += 1
    cmds.append(_make_cmd(
        next_id, tile_id, layer_id, token_phase, exec_phase,
        CMD_EXEC_SWIGLU,
        tensor_deps=[],
        input_bufs=[_buf("STAGING", ffn_bytes, dtype)] * 2,
        output_bufs=[_buf("STAGING", ffn_bytes, dtype)],
        required_routes=[], required_reductions=[],
        fence_id=f, dep_fence_id=dep_fence,
    ))
    dep_fence = f

    # 16. Down projection
    f = next_fence[0]; next_fence[0] += 1
    row_routes = partial_reduce_routes if placement_policy == "row_split" else []
    cmds.append(_make_cmd(
        next_id, tile_id, layer_id, token_phase, exec_phase,
        CMD_EXEC_MATMUL,
        tensor_deps=[f"layers.{layer_id}.feed_forward.w2.weight"],
        input_bufs=[_buf("STAGING", ffn_bytes, dtype)],
        output_bufs=[_buf("STAGING", act_bytes, dtype)],
        required_routes=row_routes, required_reductions=[],
        fence_id=f, dep_fence_id=dep_fence,
    ))
    dep_fence = f

    # 17. Residual add (FFN)
    f = next_fence[0]; next_fence[0] += 1
    cmds.append(_make_cmd(
        next_id, tile_id, layer_id, token_phase, exec_phase,
        CMD_EXEC_RESIDUAL,
        tensor_deps=[],
        input_bufs=[_buf("STAGING", act_bytes, dtype)] * 2,
        output_bufs=[_buf("STAGING", act_bytes, dtype)],
        required_routes=[], required_reductions=[],
        fence_id=f, dep_fence_id=dep_fence,
    ))
    dep_fence = f

    # 18. Layer barrier
    f = next_fence[0]; next_fence[0] += 1
    cmds.append(_make_cmd(
        next_id, tile_id, layer_id, token_phase, exec_phase,
        CMD_TILE_BARRIER,
        tensor_deps=[], input_bufs=[], output_bufs=[],
        required_routes=[], required_reductions=[],
        fence_id=f, dep_fence_id=dep_fence,
    ))

    return cmds


def _generate_lm_head_plan(
    header: dict,
    tensors_by_name: dict,
    placement_policy: str,
    token_phase: str,
    next_id: list,
    next_fence: list,
    route_ids_by_type: dict,
    notes: list,
) -> list:
    """Generate the LM head (output norm + matmul + logits) command sequence."""
    cmds = []
    d_model    = header.get("d_model", 64)
    vocab_size = header.get("vocab_size", 256)
    dtype      = header.get("dtype", "f32")
    db         = _dtype_bytes(dtype)
    act_bytes  = d_model * db
    logit_bytes = vocab_size * db

    # Determine tile that owns the output norm / lm_head
    norm_tensor = tensors_by_name.get("norm.weight") or tensors_by_name.get("output_norm.weight")
    tile_id = norm_tensor.get("owner_tile", 0) if norm_tensor else 0

    exec_phase = PHASE_PREFILL_EXEC if token_phase == "prefill" else PHASE_DECODE_STEP
    logits_routes = route_ids_by_type.get("LOGITS_REDUCE", [])
    dep_fence = 0

    # 1. Output norm
    f = next_fence[0]; next_fence[0] += 1
    cmds.append(_make_cmd(
        next_id, tile_id, -1, token_phase, exec_phase,
        CMD_EXEC_RMSNORM,
        tensor_deps=["norm.weight"],
        input_bufs=[_buf("STAGING", act_bytes, dtype)],
        output_bufs=[_buf("STAGING", act_bytes, dtype)],
        required_routes=[], required_reductions=[],
        fence_id=f, dep_fence_id=dep_fence,
    ))
    dep_fence = f

    # 2. LM head matmul
    f = next_fence[0]; next_fence[0] += 1
    req_routes = logits_routes if placement_policy == "lm_head_split" else []
    cmds.append(_make_cmd(
        next_id, tile_id, -1, token_phase, exec_phase,
        CMD_EXEC_MATMUL,
        tensor_deps=["lm_head.weight"],
        input_bufs=[_buf("STAGING", act_bytes, dtype)],
        output_bufs=[_buf("STAGING", logit_bytes, dtype)],
        required_routes=req_routes, required_reductions=[],
        fence_id=f, dep_fence_id=dep_fence,
    ))
    dep_fence = f

    # 3. Softmax / sampling
    f = next_fence[0]; next_fence[0] += 1
    cmds.append(_make_cmd(
        next_id, tile_id, -1, token_phase, exec_phase,
        CMD_EXEC_SOFTMAX,
        tensor_deps=[],
        input_bufs=[_buf("STAGING", logit_bytes, dtype)],
        output_bufs=[_buf("STAGING", logit_bytes, dtype)],
        required_routes=[], required_reductions=[],
        fence_id=f, dep_fence_id=dep_fence,
        trace_flags=2,
    ))

    return cmds


# ---------------------------------------------------------------------------
# Route index builder
# ---------------------------------------------------------------------------

def _build_route_index(route_report: dict) -> dict:
    """Return {route_type: [route_id, ...]} from a route report."""
    from collections import defaultdict
    index = defaultdict(list)
    for route in route_report.get("routes", []):
        rtype = route.get("route_type")
        rid   = route.get("route_id")
        if rtype and rid is not None:
            index[rtype].append(rid)
    return dict(index)

# ---------------------------------------------------------------------------
# Main planner
# ---------------------------------------------------------------------------

def plan(
    placement_report: dict,
    command_plan: dict | None,
    route_report: dict | None,
    model_id: str,
    session_id: str,
    token_phase: str,
    strict: bool,
) -> dict:
    notes: list[str] = []

    header        = placement_report["header"]
    tile_count    = header.get("tile_count", 1)
    n_layers      = header.get("n_layers", 0)
    placement_pol = header.get("placement_policy", "layer_wise")
    dtype         = header.get("dtype", "f32")

    if placement_pol not in SUPPORTED_PLACEMENT_POLICIES:
        notes.append(
            f"WARN: unknown placement_policy={placement_pol!r}; "
            "treating as layer_wise"
        )

    # Validate tensor placements
    errors = _validate_placement(placement_report, tile_count, notes)

    # Build tensor name index
    tensors_by_name: dict = {}
    for t in placement_report.get("tensors", []):
        name = t.get("tensor_name")
        if name:
            tensors_by_name[name] = t

    # Route index
    route_ids_by_type: dict = {}
    if route_report:
        route_ids_by_type = _build_route_index(route_report)

    # Planner state
    next_id    = [1]
    next_fence = [1]
    commands: list = []

    # 1. Load + validate phase
    commands.extend(_generate_load_phase(
        placement_report.get("tensors", []), next_id, next_fence, token_phase
    ))

    # 2. Prefill setup (barrier + trace)
    commands.extend(_generate_prefill_setup(
        tile_count, next_id, next_fence, token_phase, route_ids_by_type
    ))

    # 3. Per-layer execution plan
    for layer_id in range(n_layers):
        commands.extend(_generate_layer_plan(
            layer_id, header, tensors_by_name, placement_pol,
            token_phase, next_id, next_fence, route_ids_by_type, notes,
        ))
        # Trace every 4 layers
        if (layer_id + 1) % 4 == 0 or (layer_id + 1) == n_layers:
            f = next_fence[0]; next_fence[0] += 1
            for tile_id in range(tile_count):
                commands.append(_make_cmd(
                    next_id, tile_id, layer_id, token_phase, PHASE_TRACE,
                    CMD_TRACE_SNAPSHOT,
                    tensor_deps=[], input_bufs=[], output_bufs=[],
                    required_routes=[], required_reductions=[],
                    fence_id=f, dep_fence_id=0, trace_flags=1,
                ))

    # 4. LM head
    commands.extend(_generate_lm_head_plan(
        header, tensors_by_name, placement_pol,
        token_phase, next_id, next_fence, route_ids_by_type, notes,
    ))

    # 5. Query counters + cleanup
    f_q = next_fence[0]; next_fence[0] += 1
    for tile_id in range(tile_count):
        commands.append(_make_cmd(
            next_id, tile_id, -1, token_phase, PHASE_QUERY,
            CMD_QUERY_COUNTERS,
            tensor_deps=[], input_bufs=[], output_bufs=[],
            required_routes=[], required_reductions=[],
            fence_id=f_q, dep_fence_id=0,
        ))

    # Determine status
    if errors:
        status = "fail"
    elif any(n.startswith("WARN") for n in notes):
        status = "warn" if not strict else "fail"
    else:
        status = "pass"

    for e in errors:
        notes.insert(0, f"ERROR: {e}")

    return {
        "execution_plan_version": EXECUTION_PLAN_VERSION,
        "model_id": model_id,
        "session_id": session_id,
        "token_phase": token_phase,
        "tile_count": tile_count,
        "layer_count": n_layers,
        "command_count": len(commands),
        "placement_report_path": placement_report.get("_path", ""),
        "command_plan_path": command_plan.get("_path", "") if command_plan else None,
        "route_report_path": route_report.get("header", {}).get("source_placement_report")
                             if route_report else None,
        "status": status,
        "notes": notes,
        "commands": commands,
    }

# ---------------------------------------------------------------------------
# Text report
# ---------------------------------------------------------------------------

def _print_report(result: dict) -> None:
    print(f"placement_report_path   : {result['placement_report_path']}")
    print(f"model_id                : {result['model_id']}")
    print(f"session_id              : {result['session_id']}")
    print(f"token_phase             : {result['token_phase']}")
    print(f"tile_count              : {result['tile_count']}")
    print(f"layer_count             : {result['layer_count']}")
    print(f"command_count           : {result['command_count']}")
    print(f"status                  : {result['status']}")
    for note in result.get("notes", []):
        print(f"  note: {note}")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="ATT-1 tensor-level placement execution planner (M125)"
    )
    parser.add_argument("--placement-report", required=True,
                        metavar="PATH", help="M98/M100 placement report JSON")
    parser.add_argument("--command-plan", metavar="PATH",
                        help="M109 command plan JSON (optional)")
    parser.add_argument("--route-report", metavar="PATH",
                        help="M115/M117 fabric route report JSON (optional)")
    parser.add_argument("--model-id", metavar="ID", default=None)
    parser.add_argument("--session-id", metavar="ID", default="session_0")
    parser.add_argument("--token-phase", choices=["prefill", "decode"],
                        default="prefill")
    parser.add_argument("--plan-json", metavar="PATH",
                        help="Write execution plan JSON to path")
    parser.add_argument("--strict", action="store_true",
                        help="Promote warnings to errors")
    args = parser.parse_args()

    try:
        placement_report = _load_placement_report(args.placement_report)
        placement_report["_path"] = args.placement_report

        command_plan = None
        if args.command_plan:
            command_plan = _load_command_plan(args.command_plan)
            command_plan["_path"] = args.command_plan

        route_report = None
        if args.route_report:
            route_report = _load_route_report(args.route_report)

    except ParseError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)

    # Determine model_id
    model_id = args.model_id
    if not model_id:
        model_id = (placement_report.get("header", {}).get("model_name")
                    or "unknown_model")

    result = plan(
        placement_report, command_plan, route_report,
        model_id, args.session_id, args.token_phase, args.strict,
    )

    _print_report(result)

    if args.plan_json:
        try:
            with open(args.plan_json, "w") as fh:
                json.dump(result, fh, indent=2)
                fh.write("\n")
        except OSError as exc:
            print(f"ERROR: could not write plan JSON: {exc}", file=sys.stderr)
            sys.exit(2)

    if result["status"] == "fail":
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
