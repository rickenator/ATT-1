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

---

## 12. Execution-Plan Validator (Milestone 128)

`compiler/validate_tensor_execution_plan.py` validates execution-plan JSON
files produced by the M125 planner against the schema defined in §7.

**This tool does not execute commands, perform tensor math, or change any
runtime or inference behavior.**

### 12.1 Invocation

```
python3 compiler/validate_tensor_execution_plan.py \
    --plan <path-to-plan.json> \
    [--report-json <output.json>] \
    [--strict]
```

| Flag | Description |
|---|---|
| `--plan PATH` | Path to the M125 execution-plan JSON (required) |
| `--report-json PATH` | Write JSON validation summary to this path |
| `--strict` | Promote warnings to errors |

**Exit codes:** 0 = pass, 1 = validation errors, 2 = parse error.

### 12.2 Validation rules

| Rule | Code | Description |
|---|---|---|
| Header fields | E01 | `execution_plan_version` present and supported; `model_id`, `session_id`, `tile_count > 0` present; `command_count` matches `len(commands)` |
| Unique command IDs | E02 | `plan_command_id` is an integer and unique across all commands |
| Valid tile IDs | E03 | `tile_id` is a non-negative integer and `< tile_count` |
| Known phase | E04 | `execution_phase` is one of the 16 recognised phases |
| Known command type | E05 | `command_type` is one of the 16 recognised M105/M125 types |
| Tensor dependencies | E06 | Commands that require tensors (`EXEC_MATMUL`, `EXEC_RMSNORM`, `LOAD_TENSOR_TILE`, `VALIDATE_TENSOR`) must have at least one non-empty `tensor_dependencies` entry |
| Buffer descriptors | E07 | `region_type` recognised; `byte_size > 0`; `dtype` recognised; q4 metadata warning (error in strict mode) |
| Expected status | E08 | `expected_status` is a recognised status string |
| Trace flags | E09 | `trace_flags` is a non-negative integer when present |
| Dependency ordering | E10 | `dependency_fence_id` references an existing `fence_id` of a command that appears *earlier* in the list; no cycles in the dependency graph |
| Route / reduction lists | E11 | `required_routes` and `required_reductions` are lists when present |
| FABRIC_SEND | E12 | Must specify `dst_tile` or non-empty `required_routes`; `dst_tile < tile_count` |
| FABRIC_REDUCE | E13 | Must include `reduction_behavior` from the recognised set |
| Phase ordering | E20 | First occurrence of each phase respects the partial order (probe before enumeration; allocation before load; load before validate; validate before execution planning; KV ops after setup; cleanup/reset last) |

### 12.3 Fixtures

| Fixture | Expected outcome |
|---|---|
| `compiler/fixtures/exec_plan_valid_tiny.json` | PASS — 1-tile, 1-layer, 6 commands |
| `compiler/fixtures/exec_plan_missing_header.json` | FAIL E01 — missing `execution_plan_version` |
| `compiler/fixtures/exec_plan_duplicate_cmd_id.json` | FAIL E02 — duplicate `plan_command_id=1` |
| `compiler/fixtures/exec_plan_unknown_phase.json` | FAIL E04 — `execution_phase='UNKNOWN_PHASE'` |
| `compiler/fixtures/exec_plan_future_dep.json` | FAIL E10 — dependency on a command not yet seen |
| `compiler/fixtures/exec_plan_cyclic_dep.json` | FAIL E10 — circular fence dependency |
| `compiler/fixtures/exec_plan_missing_tensor.json` | FAIL E06 — `EXEC_MATMUL` with empty `tensor_dependencies` |
| `compiler/fixtures/exec_plan_missing_reduction.json` | FAIL E13 — `FABRIC_REDUCE` without `reduction_behavior` |
| `compiler/fixtures/exec_plan_invalid_tile.json` | FAIL E03 — `tile_id >= tile_count` |
| `compiler/fixtures/exec_plan_q4_bad_meta.json` | PASS with W07 (FAIL with `--strict`) — q4 buffer lacks metadata |

### 12.4 Non-Goals for M128

- Does not execute commands or tensor math.
- Does not access real PCIe/MMIO registers or implement a kernel driver.
- Does not validate byte-level layout of `.att1` model files.
- Does not re-run placement report validation (use `validate_tensor_placement_report.py`).
- Does not check public model weights or generated public `.att1` artifacts.
- No C, Makefile, binary format, or inference behavior changes.
- No CUDA kernels or runtime scheduler changes.

---

## 13. Execution-Plan-to-Command-Plan Mapper (Milestone 129)

`compiler/map_execution_plan_to_commands.py` reads an M125 tensor-level
execution-plan JSON and emits an M109-compatible AIMU command-plan JSON that
can be replayed through existing command-plan tooling (M113, M122).

### 13.1 Invocation

```
python3 compiler/map_execution_plan_to_commands.py \
    --execution-plan PATH \
    [--plan-json PATH] \
    [--model-id ID] \
    [--session-id ID] \
    [--strict]
```

Exit codes: 0 = success (warnings may be present), 1 = mapping error,
2 = parse error (malformed JSON / missing required field).

### 13.2 Mapping rules

| M125 execution_phase / command_type | M109 command_type | expected_status |
|-------------------------------------|-------------------|-----------------|
| `LOAD_TENSOR_TILE` | `LOAD_TENSOR_TILE` | OK |
| `VALIDATE_TENSOR_TILE` → `VALIDATE_TENSOR` | `VALIDATE_TENSOR` | OK |
| `KV_APPEND` | `KV_APPEND` | OK |
| `KV_READ` | `KV_READ` | OK |
| `FABRIC_SEND` | `FABRIC_SEND` | OK |
| `FABRIC_REDUCE` | `FABRIC_REDUCE` | OK |
| `TILE_BARRIER` | `TILE_BARRIER` | OK |
| `TRACE_SNAPSHOT` | `TRACE_SNAPSHOT` | OK |
| `QUERY_COUNTERS` | `QUERY_COUNTERS` | OK |
| `PREFILL_SETUP` (advisory) | `TILE_BARRIER` | OK |
| `TILE_ENUMERATION` (advisory) | `QUERY_COUNTERS` | OK |
| `DEVICE_PROBE` / `MEMORY_ALLOCATE` (advisory) | `NOP` (advisory note) | OK |
| `CLEANUP` + `QUERY_COUNTERS` | `QUERY_COUNTERS` | OK |
| `RESET` | `RESET_TILE` | OK |
| `EXEC_MATMUL`, `EXEC_RMSNORM`, `EXEC_ROPE`, `EXEC_ATTENTION`, `EXEC_SWIGLU`, `EXEC_SOFTMAX`, `EXEC_RESIDUAL` | passthrough name | `ATT1_AIMU_ERR_UNSUPPORTED_OP` (non-strict) / **MappingError** (strict) |

### 13.3 Reference fixture

`compiler/fixtures/exec_plan_mapped_cmd_plan.json` — the M109 command plan
produced from `exec_plan_valid_tiny.json`.  Contains 6 commands:
`LOAD_TENSOR_TILE`, `VALIDATE_TENSOR`, `TILE_BARRIER`, `EXEC_MATMUL`
(unsupported), `KV_APPEND`, `QUERY_COUNTERS`.

### 13.4 Non-Goals for M129

- Does not execute tensor math or run inference.
- Does not access real PCIe/MMIO registers or implement a kernel driver.
- Does not validate `.att1` binary format.
- Does not implement a runtime scheduler or kernel module.
- No C, Makefile, binary format, or inference behavior changes.
- No CUDA kernels or runtime scheduler changes.

---

## 14. Simulated AIMU EXEC Command Replay (Milestone 130)

M130 adds `att1_aimu_exec` — a simulated dispatch layer that replays
`att1_aimu_cmd` sequences without performing any tensor arithmetic.
All dispatch decisions are based solely on the command type, tile capability
bitmask, dtype bitmask, and the `tensor_id` field.

### 14.1 Invocation

```c
#include "att1_aimu_exec.h"

att1_aimu_device    *dev;  /* optional — pass NULL for capability-unchecked replay */
att1_aimu_exec_ctx  *ctx;

att1_aimu_exec_ctx_create(dev, &ctx);

/* replay a command */
att1_aimu_result r = att1_aimu_exec_dispatch(ctx, &cmd);

/* snapshot counters */
att1_aimu_exec_counters cnt;
att1_aimu_exec_ctx_get_counters(ctx, &cnt);

/* reset counters between replays */
att1_aimu_exec_ctx_reset_counters(ctx);

att1_aimu_exec_ctx_destroy(ctx);
```

When `dev` is `NULL` the dispatcher treats the tile as having
`ATT1_AIMU_OP_ALL` and `ATT1_AIMU_DTYPE_ALL` — every op and dtype is
accepted.

### 14.2 Dispatch table

| Command type            | Validation checks                        | Result on success            | Counter incremented            |
|-------------------------|------------------------------------------|------------------------------|--------------------------------|
| `EXEC_MATMUL`           | tile_id range, tensor_id ≠ 0, dtype, op  | `ATT1_AIMU_OK`               | `matmul_count`, byte estimates |
| `EXEC_RMSNORM`          | tile_id range, dtype, op                 | `ATT1_AIMU_OK`               | `rmsnorm_count`, byte estimates|
| `EXEC_ROPE`             | tile_id range, dtype, op                 | `ATT1_AIMU_OK`               | `rope_count`, byte estimates   |
| `EXEC_ATTENTION`        | tile_id range, dtype, op                 | `ATT1_AIMU_OK`               | `attention_count`, byte estimates|
| `EXEC_FFN`              | tile_id range, dtype, op                 | `ATT1_AIMU_OK`               | `ffn_count`, byte estimates    |
| `KV_APPEND`             | tile_id range, op                        | `ATT1_AIMU_OK`               | `kv_append_count`              |
| `KV_READ`               | tile_id range, op                        | `ATT1_AIMU_OK`               | `kv_read_count`                |
| `FABRIC_SEND`           | tile_id range, op                        | `ATT1_AIMU_OK`               | `fabric_send_count`            |
| `FABRIC_REDUCE`         | tile_id range, op                        | `ATT1_AIMU_OK`               | `fabric_reduce_count`          |
| `LOAD_TENSOR_TILE`      | none                                     | `ATT1_AIMU_OK`               | `bytes_written_estimate`       |
| `VALIDATE_TENSOR`       | none                                     | `ATT1_AIMU_OK`               | —                              |
| `NOP`                   | none                                     | `ATT1_AIMU_OK`               | —                              |
| `RESET_TILE`            | none                                     | `ATT1_AIMU_OK`               | —                              |
| `TILE_BARRIER`          | none                                     | `ATT1_AIMU_OK`               | `barrier_count`                |
| `TRACE_SNAPSHOT`        | none                                     | `ATT1_AIMU_OK`               | `trace_snapshot_count`         |
| `QUERY_COUNTERS`        | none                                     | `ATT1_AIMU_OK`               | —                              |
| unknown                 | —                                        | `ATT1_AIMU_ERR_INVALID_COMMAND` | `exec_commands_failed`      |

`exec_commands_seen` is incremented for every call regardless of outcome.
`exec_commands_completed` and per-op counters are incremented only on
`ATT1_AIMU_OK`. `exec_commands_failed` and `exec_unsupported` are
incremented on any error result.

### 14.3 Counter struct (`att1_aimu_exec_counters`)

| Field                   | Meaning                                                   |
|-------------------------|-----------------------------------------------------------|
| `exec_commands_seen`    | Total calls to `att1_aimu_exec_dispatch`                  |
| `exec_commands_completed` | Calls that returned `ATT1_AIMU_OK`                     |
| `exec_commands_failed`  | Calls that returned any error                             |
| `exec_unsupported`      | Calls that returned `UNSUPPORTED_OP` or `UNSUPPORTED_DTYPE` |
| `matmul_count`          | Successful `EXEC_MATMUL` dispatches                       |
| `rmsnorm_count`         | Successful `EXEC_RMSNORM` dispatches                      |
| `rope_count`            | Successful `EXEC_ROPE` dispatches                         |
| `attention_count`       | Successful `EXEC_ATTENTION` dispatches                    |
| `ffn_count`             | Successful `EXEC_FFN` dispatches                          |
| `kv_append_count`       | Successful `KV_APPEND` dispatches                         |
| `kv_read_count`         | Successful `KV_READ` dispatches                           |
| `fabric_send_count`     | Successful `FABRIC_SEND` dispatches                       |
| `fabric_reduce_count`   | Successful `FABRIC_REDUCE` dispatches                     |
| `barrier_count`         | Successful `TILE_BARRIER` dispatches                      |
| `trace_snapshot_count`  | Successful `TRACE_SNAPSHOT` dispatches                    |
| `bytes_read_estimate`   | Sum of `input_buf_bytes` from successful EXEC_* commands  |
| `bytes_written_estimate`| Sum of `output_buf_bytes` from EXEC_* + LOAD_TENSOR_TILE  |

### 14.4 Non-Goals for M130

- Does not execute tensor math, run inference, or touch the CPU/CUDA backends.
- Does not access real PCIe/MMIO registers or implement a kernel driver.
- Does not validate `.att1` binary format.
- Does not implement a runtime scheduler or kernel module.
- No inference, backend, tokenizer, CUDA, binary-format, or `.att1` changes.
- No CUDA kernels or runtime scheduler changes.
