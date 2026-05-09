# ATT-1 Tensor-Level Placement Execution Plan (Milestone 125)

This document defines the planning model for converting ATT-1 tensor placement
records into per-token, per-layer, per-tile EXEC_* command sequences on
simulated AIMU tiles.

**This is planning and specification only.**  No inference is executed, no
hardware is accessed, no backend behavior is changed, and no tensor math is
performed.  The execution planner is an advisory tool that emits JSON
command-sequence records for review, simulation, and future integration.

---

## 1. Scope and Non-Goals

### 1.1 Scope

The M125 execution planner (`compiler/plan_tensor_execution.py`) consumes:

- An M98/M100 tensor placement report JSON (required).
- An M109 AIMU command plan JSON (optional; enriches tensor-to-command mapping).
- An M115/M117 fabric route report JSON (optional; enriches fabric step planning).

It emits:

- An **advisory execution-plan JSON** describing the per-layer, per-tile
  EXEC_* command sequence for a single token step (prefill or decode).
- A **human-readable summary** to stdout.

### 1.2 Non-Goals

- No actual AIMU execution.
- No runtime scheduling changes.
- No inference behavior changes.
- No backend (CPU/CUDA) changes.
- No `.att1` binary format changes.
- No tokenizer changes.
- No CUDA kernels or runtime changes.
- No real PCIe/MMIO access.
- No Linux kernel driver.
- No patent claim language.

---

## 2. Execution Phases

A complete AIMU tile execution run for one token step proceeds through the
following ordered phases:

| Phase | Command type(s) | Description |
|---|---|---|
| `DEVICE_PROBE` | (M121 API) | Probe BAR0, read DEVICE_ID and REGISTER_MAP_VERSION |
| `TILE_ENUMERATION` | `QUERY_COUNTERS` | Query per-tile capabilities and memory capacity |
| `MEMORY_ALLOCATE` | (M124 allocator) | Reserve tensor, KV, staging, command, trace, and fabric regions |
| `LOAD_TENSOR_TILE` | `LOAD_TENSOR_TILE` | DMA each assigned tensor weight onto its owner tile |
| `VALIDATE_TENSOR_TILE` | `VALIDATE_TENSOR` | Validate loaded tensor checksums |
| `PREFILL_SETUP` | `TILE_BARRIER` | Synchronise all tiles before first token |
| `PREFILL_EXECUTION_PLAN` | `EXEC_*` × N | Per-layer command sequences for prefill pass |
| `DECODE_STEP_PLAN` | `EXEC_*` × N | Per-layer command sequences for each decode step |
| `KV_APPEND` | `KV_APPEND` | Append key/value projections to KV cache after each layer |
| `KV_READ` | `KV_READ` | Read cached keys/values for attention score computation |
| `FABRIC_SEND` | `FABRIC_SEND` | Send cross-tile activation or reduction payloads |
| `FABRIC_REDUCE` | `FABRIC_REDUCE` | Initiate fabric reduction for partial outputs |
| `TRACE_SNAPSHOT` | `TRACE_SNAPSHOT` | Capture per-tile trace snapshot |
| `QUERY_COUNTERS` | `QUERY_COUNTERS` | Read command-queue and performance counters |
| `CLEANUP` / `RESET` | `RESET_TILE` / `NOP` | Release resources and reset tile state |

---

## 3. Per-Layer Command Sequence

For each transformer layer, the execution planner generates command records
in the following order.  Commands whose tile ownership differs from the
default tile may be scheduled on a different tile's command queue.

### 3.1 Attention sublayer

| Step | Operation | EXEC command | Fabric route required |
|---|---|---|---|
| 1 | `attention_norm` (RMSNorm) | `EXEC_RMSNORM` | None (local) |
| 2 | Q matmul | `EXEC_MATMUL` | None (local) |
| 3 | K matmul | `EXEC_MATMUL` | None or KV_TRANSFER if KV owner differs |
| 4 | V matmul | `EXEC_MATMUL` | None or KV_TRANSFER if KV owner differs |
| 5 | RoPE (Q) | `EXEC_ROPE` | None (local) |
| 6 | RoPE (K) | `EXEC_ROPE` | None or KV_TRANSFER |
| 7 | KV append | `KV_APPEND` | KV_TRANSFER if KV owner ≠ layer owner |
| 8 | KV read | `KV_READ` | KV_TRANSFER if KV owner ≠ layer owner |
| 9 | Attention scores + softmax | `EXEC_ATTENTION` | None (local to head-wise tile) |
| 10 | Attention value gather | `EXEC_MATMUL` | PARTIAL_REDUCE if heads split across tiles |
| 11 | Output projection | `EXEC_MATMUL` | PARTIAL_REDUCE or ACTIVATION_SEND |
| 12 | Residual add | `EXEC_RESIDUAL` | ACTIVATION_SEND if residual on different tile |
| 13 | Barrier | `TILE_BARRIER` | TILE_BARRIER (all participating tiles) |

### 3.2 FFN sublayer

| Step | Operation | EXEC command | Fabric route required |
|---|---|---|---|
| 1 | `ffn_norm` (RMSNorm) | `EXEC_RMSNORM` | None (local) |
| 2 | Gate projection | `EXEC_MATMUL` | PARTIAL_REDUCE if row-split |
| 3 | Up projection | `EXEC_MATMUL` | PARTIAL_REDUCE if row-split |
| 4 | SwiGLU | `EXEC_SWIGLU` | None (local) |
| 5 | Down projection | `EXEC_MATMUL` | PARTIAL_REDUCE if column-split |
| 6 | Residual add | `EXEC_RESIDUAL` | ACTIVATION_SEND if residual on different tile |
| 7 | Barrier | `TILE_BARRIER` | TILE_BARRIER (all participating tiles) |

### 3.3 LM head

| Step | Operation | EXEC command | Fabric route required |
|---|---|---|---|
| 1 | Output norm | `EXEC_RMSNORM` | None (local) |
| 2 | LM head matmul | `EXEC_MATMUL` | LOGITS_REDUCE if vocab split |
| 3 | Logits gather | — | LOGITS_REDUCE |
| 4 | Softmax / sampling | `EXEC_SOFTMAX` | None (local to rank-0 tile) |

### 3.4 Trace points

Insert `TRACE_SNAPSHOT` commands at:

- After `PREFILL_SETUP`.
- After every 4 transformer layers (configurable).
- After `DECODE_STEP_PLAN` completes.
- At `CLEANUP`.

---

## 4. Placement-Dependent Execution

The execution command sequence depends on the placement policy recorded in the
M98/M100 placement report `header.placement_policy`.

### 4.1 Layer-wise placement

Each tile owns all tensors for a contiguous range of layers.  No within-layer
cross-tile traffic.  Cross-tile traffic is layer-boundary activation sends.

- After the last layer on tile T, one `FABRIC_SEND` routes the residual
  activation to tile T+1.
- One `TILE_BARRIER` follows.
- Each tile processes its layers independently in sequence.

### 4.2 Tensor-wise placement

Individual tensors may be placed on tiles independently of layer boundaries.
Non-local tensor references require `FABRIC_SEND` before the dependent EXEC.

### 4.3 Row-split matmul

A weight matrix is split along the input dimension (rows) across tiles.

- Each tile computes a partial output from its shard.
- After compute, a `PARTIAL_REDUCE` with `reduction_behavior=sum` collects
  partial results onto a designated accumulation tile.
- One `TILE_BARRIER` follows the reduction.

### 4.4 Column-split matmul

A weight matrix is split along the output dimension (columns) across tiles.

- Each tile computes a slice of the output.
- No reduction required if output slices are distributed (each tile consumes
  its own slice in the next step).
- A `FABRIC_SEND` or `TILE_BARRIER` synchronises the pipeline.

### 4.5 Head-wise attention split

Attention heads are distributed across tiles (one or more heads per tile).

- Each tile computes its assigned head(s) independently.
- After the output projection, each tile holds a partial result.
- A `PARTIAL_REDUCE` with `reduction_behavior=sum` (or `concat` if heads are
  independently addressable) collects the final output.

### 4.6 LM head / vocab split

The lm_head weight matrix rows are split across tiles by vocabulary range.

- Each tile computes logits for its vocab slice.
- A `LOGITS_REDUCE` with `reduction_behavior=concat` assembles the full
  logit vector on a designated tile.
- Sampling (`EXEC_SOFTMAX`) runs on that tile.

### 4.7 Embedding split

Embedding rows (vocabulary → embedding dimension) are distributed by
vocabulary range.

- At token lookup, the tile owning the token's embedding row serves the
  result via `ACTIVATION_SEND`.
- All other tiles await the embedding via `ACTIVATION_BROADCAST` or
  `FABRIC_SEND`.

### 4.8 Replicated norms

Norm weights (RMSNorm scale vectors) may be replicated on all tiles.

- No fabric route required for norm computation.
- Each tile applies its local copy independently.
- Replication consistency is not re-validated per token; it is verified once
  during `VALIDATE_TENSOR_TILE`.

### 4.9 KV ownership

Each tile owns the KV cache for its assigned layers.

- Local KV: `KV_APPEND` and `KV_READ` commands are local; no fabric route.
- Remote KV: `KV_TRANSFER` route sends/receives KV pages between tiles.
- KV_TRANSFER payload bytes are calculated from:
  `2 × seq_len × n_kv_heads_per_tile × head_dim × dtype_bytes`.

---

## 5. Fabric and Reduction Requirements

| Placement style | Route type | Payload source | Reduction behavior | Barrier/fence | Determinism rule |
|---|---|---|---|---|---|
| Layer-wise boundary | `ACTIVATION_SEND` | `activation_bytes_per_token` | `pass_through` | `TILE_BARRIER` after send | Ordered; tile-ID ascending |
| Row-split matmul | `PARTIAL_REDUCE` | `d_model × dtype_bytes` | `sum` | `TILE_BARRIER` after reduction | Source tile-ID ascending, `reduction_id` unique per layer |
| Column-split matmul | `ACTIVATION_SEND` | slice output size | `pass_through` | `TILE_BARRIER` if downstream tile differs | Ordered |
| Head-wise attention | `PARTIAL_REDUCE` | `d_model × dtype_bytes` | `sum` | `TILE_BARRIER` after reduction | Ordered |
| LM head vocab split | `LOGITS_REDUCE` | `vocab_slice × dtype_bytes` | `concat` | `TILE_BARRIER` after concat | Tile-ID ascending |
| Embedding split | `ACTIVATION_BROADCAST` | `d_model × dtype_bytes` | `pass_through` | `TILE_BARRIER` | Source tile holds token row |
| KV transfer | `KV_TRANSFER` | `2 × seq × kv_heads × head_dim × dtype` | `pass_through` | `TILE_BARRIER` | Ordered per-session |

**Tolerance implications:**

- `reduction_behavior=sum` accumulates floating-point values; q8 partial sums
  may introduce abs error ≤ 0.15 per element; q4 ≤ 0.35 per element.
- `reduction_behavior=concat` does not accumulate; tolerance is dtype-native.
- `pass_through` routes incur no additional numerical error.

---

## 6. Memory Allocation Requirements

The M124 tile memory allocator (`att1_aimu_mem`) maps to execution regions as
follows.  All sizes are planning-time estimates; actual allocation is
performed by the M125+ execution planner against M124 metadata.

| Execution use | M124 region type | Size estimate | Alignment |
|---|---|---|---|
| Weight tensor storage | `TENSOR` | `packed_bytes` from placement record | 64 B (DMA) |
| KV cache pages | `KV_CACHE` | `kv_bytes` from tile summary | 64 B (DMA) |
| Activation staging (intra-tile) | `STAGING` | `activation_bytes_per_token` | 16 B |
| DMA descriptor ring | `DMA_BUFFER` | `n_commands × 64 B` | 64 B |
| M105 command ring | `COMMAND_QUEUE` | `ring_depth × sizeof(cmd)` | 16 B |
| Completion ring | `COMPLETION_QUEUE` | `ring_depth × sizeof(compl)` | 16 B |
| M108 trace ring | `TRACE_BUFFER` | `trace_entries × 16 B` | 16 B |
| Fabric send/receive staging | `FABRIC_BUFFER` | `max_payload_bytes` per route | 64 B (DMA) |
| Firmware/reserved areas | `RESERVED` | documented per-tile firmware map | 4 KiB |

When `att1_aimu_mem_range_valid()` is integrated (M125/M126), the DMA
descriptor `device_base` and `byte_length` fields from M107 will be
cross-checked against the tile's allocator to confirm the referenced range
falls within a live `STAGING`, `TENSOR`, or `DMA_BUFFER` allocation.

---

## 7. Execution-Plan Record Schema

The execution plan JSON emitted by `plan_tensor_execution.py` has the
following structure.

### 7.1 Top-level

```json
{
  "execution_plan_version": 1,
  "model_id": "...",
  "session_id": "...",
  "token_phase": "prefill",
  "tile_count": 2,
  "layer_count": 2,
  "command_count": 42,
  "placement_report_path": "...",
  "command_plan_path": null,
  "route_report_path": null,
  "status": "pass",
  "notes": [],
  "commands": [ ... ]
}
```

### 7.2 Command record

```json
{
  "plan_command_id": 1,
  "tile_id": 0,
  "layer_id": 0,
  "token_phase": "prefill",
  "execution_phase": "PREFILL_EXECUTION_PLAN",
  "command_type": "EXEC_MATMUL",
  "tensor_dependencies": ["layers.0.attention.wq.weight"],
  "input_buffers": [
    { "region_type": "STAGING", "byte_size": 256, "dtype": "f32" }
  ],
  "output_buffers": [
    { "region_type": "STAGING", "byte_size": 256, "dtype": "f32" }
  ],
  "required_routes": [],
  "required_reductions": [],
  "fence_id": 1,
  "dependency_fence_id": 0,
  "expected_status": "ok",
  "trace_flags": 0
}
```

### 7.3 Field reference

| Field | Type | Description |
|---|---|---|
| `plan_command_id` | uint | Unique sequential id within this plan (1-based) |
| `tile_id` | uint | Target AIMU tile |
| `layer_id` | int | Transformer layer index; −1 for non-layer ops |
| `token_phase` | str | `prefill` or `decode` |
| `execution_phase` | str | One of the phase names from §2 |
| `command_type` | str | M105 command type name (EXEC_*, KV_*, FABRIC_*, etc.) |
| `tensor_dependencies` | list[str] | Tensor names required as inputs |
| `input_buffers` | list | Simulated buffer descriptors (region_type, byte_size, dtype) |
| `output_buffers` | list | Simulated output buffer descriptors |
| `required_routes` | list[int] | route_id values from route report required before this command |
| `required_reductions` | list[int] | reduction_id values that must complete before this command |
| `fence_id` | uint | Fence issued after this command |
| `dependency_fence_id` | uint | Fence that must be satisfied before execution |
| `expected_status` | str | `ok`, `warn`, or `skipped` |
| `trace_flags` | uint | Bitmask; bit 0 = emit trace event before, bit 1 = after |

---

## 8. Validation and Failure Rules

The planner (`plan_tensor_execution.py`) rejects or warns on the following
conditions.

| Condition | Severity | Rule |
|---|---|---|
| Missing tensor placement (no `owner_tile`) | ERROR | Tensor required by a command has no tile assignment |
| Missing tile ownership (tile_id out of range) | ERROR | `owner_tile` ≥ `tile_count` |
| Missing required route for nonlocal dependency | WARN | Non-local tensor reference with no matching route in route report |
| Unsupported split style | WARN | `placement_policy` not in known split styles |
| Missing reduction behavior | ERROR | Route type is PARTIAL_REDUCE/LOGITS_REDUCE with `reduction_behavior=none` |
| Insufficient tile memory (metadata) | WARN | Sum of `TENSOR` allocations exceeds `capacity_bytes` from tile summary |
| Unsupported dtype/op pairing | WARN | e.g. q4 with EXEC_ATTENTION (deferred to M128 for full validation) |
| Cyclic dependency | ERROR | fence DAG has a cycle |
| Unknown command type | ERROR | `command_type` not in M105 enumeration |
| Placement status not `placed` | WARN | Tensor `placement_status` is `advisory` or `fail` |

---

## 9. Planning Tool

### 9.1 CLI

```
python3 compiler/plan_tensor_execution.py \
    --placement-report PATH \
    [--command-plan PATH] \
    [--route-report PATH] \
    [--model-id ID] \
    [--session-id ID] \
    [--token-phase prefill|decode] \
    [--plan-json PATH] \
    [--strict]
```

| Flag | Default | Description |
|---|---|---|
| `--placement-report PATH` | required | M98/M100 placement report JSON |
| `--command-plan PATH` | (absent) | M109 command plan JSON for command-type mapping |
| `--route-report PATH` | (absent) | M115/M117 fabric route report JSON for route mapping |
| `--model-id ID` | from placement report | Model identifier |
| `--session-id ID` | `session_0` | Session identifier |
| `--token-phase` | `prefill` | `prefill` or `decode` |
| `--plan-json PATH` | (absent) | Write execution plan JSON to path |
| `--strict` | off | Promote warnings to errors |

**Exit codes:** 0 = plan generated (warnings OK), 1 = planning errors or
strict+warn, 2 = parse/arg error.

### 9.2 Pipeline position

```
placement report (M98/M100)
    + command plan (M109)         [optional]
    + fabric route report (M115)  [optional]
        → execution plan (M125)   ← this tool
            → execution-plan validator (M128)
            → execution-plan-to-command-plan mapper (M129)
            → simulated EXEC_* replay (M130)
            → tensor-level placement execution prototype (M131)
```

---

## 10. Fixtures

| Fixture | Description |
|---|---|
| `compiler/fixtures/placement_report_valid.json` | 2-tile, 2-layer, layer-wise placement — primary execution planning input |
| `compiler/fixtures/placement_report_capacity_fail.json` | Placement with capacity violation — triggers memory insufficiency warning |

---

## 11. Non-Goals for M125

- No actual AIMU execution.
- No measured hardware timing — all estimates are derived from placement metadata.
- No fence-reachability analysis beyond linear sequence generation.
- No concat completeness validation (deferred to M128).
- No q4 group-size / packing correctness check (deferred to M128).
- No C, Makefile, binary format, or inference behavior changes.
- No CUDA kernels or runtime scheduler changes.
