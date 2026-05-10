# AIMU Intrinsics and Operations Reference Manual (M147)

**Scope:** Simulated AIMU command/control operation model — command packets,
EXEC/LOAD/VALIDATE/KV/FABRIC/TRACE semantics, DMA descriptor model,
MMIO/register interactions, result codes, dtype support, tensor placement
implications, and replay tooling.

**Not in scope:** ATT-1 artifact format, C11 runtime API, backend dispatch
table, and inference path. Those are covered in
[docs/ATT1_REFERENCE_MANUAL.md](ATT1_REFERENCE_MANUAL.md).

**Implementation status:** All AIMU operations in this manual are
*control-plane simulations*. No tensor math executes through the AIMU
command path. No real PCIe endpoint, DMA engine, kernel driver, MMIO
mapping, or FPGA hardware exists or is required.

---

## Table of Contents

1. [AIMU Overview](#1-aimu-overview)
2. [Terminology](#2-terminology)
3. [Command Packet Model](#3-command-packet-model)
4. [Command Queue and Completion Semantics](#4-command-queue-and-completion-semantics)
5. [LOAD and VALIDATE Operations](#5-load-and-validate-operations)
6. [EXEC Operations](#6-exec-operations)
7. [KV Operations](#7-kv-operations)
8. [Fabric Operations](#8-fabric-operations)
9. [Trace and Counter Operations](#9-trace-and-counter-operations)
10. [DMA Descriptor Model](#10-dma-descriptor-model)
11. [MMIO and Register Interaction](#11-mmio-and-register-interaction)
12. [Dtype and Quantization Support](#12-dtype-and-quantization-support)
13. [Tensor Placement Implications](#13-tensor-placement-implications)
14. [Operation Status and Error Model](#14-operation-status-and-error-model)
15. [Replay and Pipeline Tools](#15-replay-and-pipeline-tools)
16. [Implemented vs Future](#16-implemented-vs-future)
17. [Non-Goals](#17-non-goals)

---

## 1. AIMU Overview

The AIMU (AI Memory Unit) is the simulated hardware abstraction at the center
of the ATT-1 control-plane pipeline. In the Phase 1 software simulator, each
AIMU tile is a logical compute unit that:

- Owns a local pool of tensor (model-weight) memory.
- Owns a KV-cache memory pool for one or more concurrent inference sessions.
- Receives commands from the host via a ring-buffer command queue.
- Executes (or simulates execution of) those commands in order.
- Returns completions to the host via a completion ring.
- Exchanges activations and reductions with other tiles through a fabric.

### Relationship to ATT-1 artifacts and runtime

```
att1-size --placement-report-json  →  placement_report.json  (M98)
      │
      ↓
validate_tensor_placement_report.py   →  validates schema
propose_tensor_placement.py           →  advisory placement
      │
      ↓
map_placement_to_commands.py          →  command_plan.json    (M109)
      │
      ├──► att1-aimu-replay            (M112/M113 in-process replay)
      └──► att1-aimu-mmio-replay       (M121/M122 MMIO-emulator replay)
      │
      ↓
map_commands_to_fabric_routes.py      →  route_report.json    (M115)
      │
      ↓
map_execution_plan_to_commands.py     →  exec_plan.json       (M125)
      │
      ↓
run_aimu_planning_pipeline.py         →  8-stage integrated report
run_execution_replay_pipeline.py      →  6-stage integrated report
```

Each stage operates on JSON files. No stage loads real model weights, executes
tensor inference, or requires physical hardware.

### Simulator components and milestones

| Module | Milestone | Header | Role |
|--------|-----------|--------|------|
| Command queue simulator | M105 | `att1_aimu_cmdq.h` | Ring-buffer submit/drain |
| Device/tile capability | M106 | `att1_aimu_device.h` | Tile enumeration, dtype/op capability |
| DMA descriptor simulator | M107 | `att1_aimu_dma.h` | Transfer descriptor validation |
| Trace/counter snapshot | M108 | `att1_aimu_trace.h` | Counter aggregation |
| Command-plan mapper | M109 | Python | Placement report → command plan |
| Host harness | M112 | `att1_aimu_host.h` | In-process replay host |
| EXEC replay | M130 | `att1_aimu_exec.h` | Simulated EXEC/KV/FABRIC dispatch |
| MMIO/register-file | M111 | `att1_aimu_mmio.h` | BAR0 register simulator |
| MMIO emulator | M121 | `att1_aimu_userspace.h` | Userspace emulator smoke |
| MMIO command replay | M122 | C tool | MMIO-path command replay |

---

## 2. Terminology

| Term | Definition |
|------|-----------|
| **AIMU** | AI Memory Unit. The simulated hardware tile in ATT-1. |
| **tile** | One logical AIMU instance. Each tile owns disjoint weight memory, KV memory, and a command queue. Indexed 0–(N-1) up to 16 tiles. |
| **tensor tile** | The subset of a model's weight tensors assigned to one tile by the placement report. Not to be confused with the tile itself. |
| **command packet** | A 64-byte fixed-width descriptor submitted to the command ring. Specifies one operation. |
| **command queue** | A shared-memory ring buffer through which the host submits commands to the AIMU. Default depth: 256 slots; maximum: 4096 slots. Power-of-two required. |
| **completion** | A 40-byte record written by the AIMU to the completion ring when a command finishes. Contains result code, latency, fence value, and counters. |
| **fence** | A monotonic integer (`fence_id`, `completion_fence_id`). A command can declare a dependency on a fence value and signal a fence on completion. Enables ordering without polling. |
| **DMA descriptor** | A 64-byte record specifying a data transfer (host↔device or device↔device). Carries address, byte count, dtype, and quant group size. No actual memory transfer occurs in the simulator. |
| **MMIO register** | A 32-bit memory-mapped I/O cell in the simulated BAR0 space (64 KiB). Access via `att1_aimu_mmio_read32` / `att1_aimu_mmio_write32`. |
| **route** | A record in the fabric route report specifying source tile, destination tile(s), route type, payload size, and ordering constraints. |
| **reduction** | A multi-tile operation where each tile contributes a partial result; results are combined at a designated aggregator tile. Reduction type 0 = element-wise sum; type 1 = concatenation. |
| **barrier** | A `TILE_BARRIER` command that holds tile execution until all tiles in the barrier group have reached the same point. |
| **trace snapshot** | A point-in-time capture of all AIMU control-plane counters (cmdq, device, DMA, fabric). Identified by a monotonically incrementing `snapshot_id`. |
| **control-plane simulation** | Simulating the host↔AIMU command exchange: submit, validate, dispatch, completion. Does not execute tensor math. |
| **execution simulation** | Running an EXEC replay: validating command fields, updating counters, returning a result code. Does not run matmul/rmsnorm/etc. |
| **unsupported/future op** | An operation whose command type is recognized but whose capability bit is not set in the device's `supported_ops` bitmask. Returns `ATT1_AIMU_ERR_UNSUPPORTED_OP`. |

---

## 3. Command Packet Model

### 3.1 On-wire layout

The command packet (`att1_aimu_cmd`) is exactly **64 bytes**, naturally
aligned. All fields are little-endian (in-process; no byte-order conversion is
performed by the simulator).

| Offset | Size | Field | Type | Description |
|--------|------|-------|------|-------------|
| 0 | 8 | `input_buf_addr` | `uint64` | Host-physical or AIMU-local input buffer address |
| 8 | 8 | `output_buf_addr` | `uint64` | Host-physical or AIMU-local output buffer address |
| 16 | 4 | `command_id` | `uint32` | Monotonic host-issued command identifier |
| 20 | 4 | `input_buf_bytes` | `uint32` | Input buffer byte count |
| 24 | 4 | `output_buf_bytes` | `uint32` | Output buffer byte count |
| 28 | 4 | `kv_position` | `uint32` | Token sequence position for KV_APPEND, KV_READ, EXEC_ATTENTION |
| 32 | 4 | `op_param_0` | `uint32` | Operation-specific parameter 0 |
| 36 | 4 | `op_param_1` | `uint32` | Operation-specific parameter 1 |
| 40 | 4 | `checksum` | `uint32` | CRC32 over bytes 0–39 (placeholder; validated when `trace_flags` bit 0 set) |
| 44 | 2 | `model_id` | `uint16` | Logical model identifier (0 = default) |
| 46 | 2 | `tensor_id` | `uint16` | Tensor slot on the tile (0 = unset/invalid for EXEC_MATMUL) |
| 48 | 2 | `fence_id` | `uint16` | Wait on this fence value before executing |
| 50 | 2 | `completion_fence_id` | `uint16` | Signal this fence on completion |
| 52 | 2 | `timeout_ms` | `uint16` | Command timeout in milliseconds (0 = no timeout) |
| 54 | 1 | `command_type` | `uint8` | `att1_aimu_cmd_type` enum value |
| 55 | 1 | `tile_id` | `uint8` | Zero-based destination tile index |
| 56 | 1 | `session_id` | `uint8` | Inference session slot |
| 57 | 1 | `dtype` | `uint8` | Dtype: `0`=f32, `1`=q8, `2`=q4 |
| 58 | 1 | `trace_flags` | `uint8` | Bit 0 = enable per-command trace emission |
| 59 | 1 | `priority` | `uint8` | `0` = normal, `1` = high |
| 60 | 1 | `status` | `uint8` | Result code written by AIMU on completion (`att1_aimu_result`) |
| 61–63 | 3 | `_pad` | `uint8[3]` | Explicit pad; must be zero |

### 3.2 Command type encoding

| Hex | Name | Group |
|-----|------|-------|
| `0x00` | `NOP` | Housekeeping |
| `0x01` | `LOAD_TENSOR_TILE` | Tensor management |
| `0x02` | `VALIDATE_TENSOR` | Tensor management |
| `0x10` | `EXEC_MATMUL` | Local execution |
| `0x11` | `EXEC_RMSNORM` | Local execution |
| `0x12` | `EXEC_ROPE` | Local execution |
| `0x13` | `EXEC_ATTENTION` | Local execution |
| `0x14` | `EXEC_FFN` | Local execution |
| `0x20` | `KV_APPEND` | KV cache |
| `0x21` | `KV_READ` | KV cache |
| `0x30` | `FABRIC_SEND` | Fabric |
| `0x31` | `FABRIC_REDUCE` | Fabric |
| `0x40` | `TRACE_SNAPSHOT` | Trace/control |
| `0x41` | `TILE_BARRIER` | Trace/control |
| `0x50` | `RESET_TILE` | Control |
| `0x51` | `QUERY_COUNTERS` | Control |

### 3.3 op_param semantics

| Command | `op_param_0` | `op_param_1` |
|---------|-------------|-------------|
| `EXEC_MATMUL` | Number of output rows | Number of output columns |
| `EXEC_RMSNORM` | Vector length | Unused |
| `EXEC_ROPE` | Vector length | Sequence position (override for `kv_position`) |
| `EXEC_ATTENTION` | Number of heads | Head dimension |
| `EXEC_FFN` | Hidden dimension | Intermediate dimension |
| `FABRIC_SEND` | Payload byte count | Destination tile ID |
| `FABRIC_REDUCE` | Payload byte count | Reduction type (0=sum, 1=concat) |
| `KV_APPEND` | Layer index | Unused |
| `KV_READ` | Layer index | Sequence position count |
| All others | Unused | Unused |

---

## 4. Command Queue and Completion Semantics

### 4.1 Ring-buffer structure

The command queue is a fixed-depth ring buffer. Slots are `att1_aimu_cmd`
packets (64 bytes each). The simulator maintains:

- **head**: consumer (AIMU) read index.
- **tail**: producer (host) write index.
- **capacity**: power-of-two slot count (`ATT1_AIMU_CMDQ_DEFAULT_DEPTH = 256`,
  `ATT1_AIMU_CMDQ_MAX_DEPTH = 4096`).

Full condition: `(tail - head) == capacity`. Empty condition: `tail == head`.

### 4.2 Submit and doorbell sequence

```
1. Host writes command packet to ring[tail % capacity].
2. Host writes ring[tail % capacity].command_id with monotonic value.
3. Host increments tail (or writes CQ_TAIL MMIO register).
4. Host writes CQ_DOORBELL (any value) to notify the AIMU.
5. AIMU advances head, reads the packet, dispatches the command.
6. AIMU writes a completion record to the completion ring.
7. AIMU optionally signals fence_value = completion_fence_id.
```

### 4.3 FIFO ordering

Commands submitted to a single queue are dispatched in FIFO order. There is
no out-of-order execution within one queue. Multi-tile ordering requires
explicit `TILE_BARRIER` commands.

### 4.4 Queue-full behavior

If the queue is full (`ATT1_AIMU_ERR_QUEUE_FULL`):
- The submit call fails immediately.
- The command is NOT enqueued.
- The `queue_full_count` counter is incremented.
- The host is responsible for retry.

### 4.5 Fence semantics

- `fence_id`: if nonzero, the command is not dispatched until the device's
  `fence_value` counter reaches or exceeds this value.
- `completion_fence_id`: if nonzero, the device increments its `fence_value`
  counter when the command completes, signaling any commands waiting on that
  value.
- Fences are monotonically increasing. Deadlock (`ATT1_AIMU_ERR_FENCE_DEADLOCK`)
  is detected if a command waits on a fence value that can never be reached.

### 4.6 Completion descriptor

The completion ring holds `att1_aimu_completion` records:

| Field | Type | Description |
|-------|------|-------------|
| `command_id` | `uint32` | Echoes the originating command's `command_id` |
| `tile_id` | `uint8` | Originating tile |
| `session_id` | `uint8` | Originating session |
| `result_code` | `uint8` | `att1_aimu_result` value |
| `latency_us` | `uint32` | Simulated latency (always 0 in M105/M130) |
| `fence_value` | `uint32` | Fence value signaled by this completion |
| `bytes_read` | `uint32` | Tensor bytes read (EXEC estimate; 0 in simulator) |
| `bytes_written` | `uint32` | Bytes written (KV append / LOAD estimate; 0 in simulator) |
| `packets_sent` | `uint32` | Fabric packets sent by this command (estimate) |
| `packets_received` | `uint32` | Fabric packets received (estimate) |
| `trace_event_count` | `uint32` | Trace records emitted |

### 4.7 Deterministic replay guarantee

When the command queue simulator replays a JSON command plan, it processes
commands in the order listed in the plan file. Result codes and counter
increments are deterministic given fixed input. This enables bit-for-bit
reproducible test assertions. No random ordering or timing variation occurs.

### 4.8 No silent fallback

If a command fails validation, the simulator:
- Returns the appropriate non-OK result code.
- Records the failure in the counter (`exec_commands_failed` or equivalent).
- Does **not** attempt the command with a different dtype, tile, or operation.

---

## 5. LOAD and VALIDATE Operations

### 5.1 LOAD_TENSOR_TILE (`0x01`)

**Purpose:** Transfer a tensor from a host DMA buffer into the tile's local
model-weight memory.

**Required command fields:**

| Field | Requirement |
|-------|-------------|
| `tile_id` | Must be < `device->tile_count` |
| `tensor_id` | Identifies the target tensor slot (nonzero) |
| `model_id` | Logical model identifier |
| `input_buf_addr` | Host-physical address of the source DMA buffer |
| `input_buf_bytes` | Byte count of the transfer |
| `dtype` | `0`=f32, `1`=q8, `2`=q4 |
| `op_param_0` | Logical row dimension (dim0) |
| `op_param_1` | Logical column dimension (dim1) |

**DMA descriptor relationship:** A validated `att1_aimu_dma_desc` with
matching `command_id` and `tensor_id` must be submitted before or alongside
`LOAD_TENSOR_TILE`. The DMA descriptor carries full dtype/quant metadata;
the command packet `dtype` field must agree.

**Memory allocation:** The simulator increments the tile's
`memory_used_bytes` by the reported `input_buf_bytes`. If the incremented
value would exceed `memory_capacity_bytes`, the command fails with
`ATT1_AIMU_ERR_OUT_OF_MEMORY`.

**Quantization metadata:** For q4 tensors, `quant_group_size` in the DMA
descriptor must be `16`, `32`, `64`, or `128` (a valid power-of-two in
`[ATT1_Q4_GROUP_SIZE_MIN, ATT1_Q4_GROUP_SIZE_MAX]`). For q8 and f32, it must
be `0`.

**Success behavior:**
- Returns `ATT1_AIMU_OK`.
- Increments `bytes_written_estimate` by `input_buf_bytes`.
- A completion record is written with `result_code = ATT1_AIMU_OK`.

**Failure behavior:**

| Condition | Result code |
|-----------|-------------|
| `tile_id >= tile_count` | `ATT1_AIMU_ERR_INVALID_COMMAND` |
| `tensor_id == 0` | `ATT1_AIMU_ERR_INVALID_TENSOR` |
| Unrecognized `dtype` | `ATT1_AIMU_ERR_INVALID_DTYPE` |
| Memory would exceed capacity | `ATT1_AIMU_ERR_OUT_OF_MEMORY` |
| DMA alignment failure | `ATT1_AIMU_ERR_ALIGNMENT` |
| DMA range check failure | `ATT1_AIMU_ERR_DMA_FAULT` |

**Counter effects:** `exec_commands_seen++`; on success, `exec_commands_completed++` and `bytes_written_estimate += input_buf_bytes`; on failure, `exec_commands_failed++`.

**Trace effects:** If `trace_flags & 0x01`, a trace event is recorded for this command.

---

### 5.2 VALIDATE_TENSOR (`0x02`)

**Purpose:** Verify that a previously loaded tensor matches its expected
checksum, shape, and dtype — without re-loading it. Used to detect silent
data corruption.

**Required command fields:**

| Field | Requirement |
|-------|-------------|
| `tile_id` | Must be < `device->tile_count` |
| `tensor_id` | Must reference a previously loaded tensor slot |
| `dtype` | Must match the dtype of the loaded tensor |
| `checksum` | Expected CRC32 over the tensor payload (placeholder in M105/M130) |
| `op_param_0` | Expected row dimension |
| `op_param_1` | Expected column dimension |

**Shape validation:** `op_param_0 × op_param_1` bytes (scaled by dtype element
size and q4 group overhead) must equal the stored `input_buf_bytes` of the
loaded tensor. A mismatch returns `ATT1_AIMU_ERR_INVALID_SHAPE`.

**Checksum behavior (current):** In the M105/M130 simulator, checksum
validation is a **placeholder**. The field is read and recorded, but the
simulator does not recompute CRC32 over actual tensor bytes (no tensor bytes
are stored). A future milestone will implement real checksum verification.

**Success behavior:** Returns `ATT1_AIMU_OK`. No byte estimates are updated.

**Failure behavior:**

| Condition | Result code |
|-----------|-------------|
| `tensor_id` not found | `ATT1_AIMU_ERR_INVALID_TENSOR` |
| Shape mismatch | `ATT1_AIMU_ERR_INVALID_SHAPE` |
| Dtype mismatch | `ATT1_AIMU_ERR_INVALID_DTYPE` |
| Checksum mismatch (future) | `ATT1_AIMU_ERR_CHECKSUM` |

---

## 6. EXEC Operations

> **Critical implementation note:** All EXEC operations in M130 are
> *control-plane simulations only*. No tensor buffers are read or written.
> No matmul, normalization, attention, FFN, or activation computation
> executes through the AIMU command path. Real inference is performed by the
> C11 CPU/CUDA runtime via `att1_infer_t` and `att1_cluster_infer_t`.
>
> The EXEC simulator validates command fields, checks dtype and op capability
> against the device model, updates counters, and returns a result code. That
> is the full extent of EXEC simulation.

### 6.1 Validation sequence (all EXEC commands)

```
1. ctx NULL or magic invalid          → ATT1_AIMU_ERR_INVALID_COMMAND
2. cmd NULL                           → ATT1_AIMU_ERR_INVALID_COMMAND
3. tile_id >= device->tile_count      → ATT1_AIMU_ERR_INVALID_COMMAND
4. EXEC_MATMUL: tensor_id == 0        → ATT1_AIMU_ERR_INVALID_TENSOR
5. dtype not in supported_dtypes      → ATT1_AIMU_ERR_UNSUPPORTED_DTYPE
6. op not in supported_ops            → ATT1_AIMU_ERR_UNSUPPORTED_OP
7. (all checks pass)                  → ATT1_AIMU_OK
```

When `device == NULL` (no capability model), steps 3–6 are skipped; all
ops and dtypes are treated as supported.

---

### 6.2 EXEC_MATMUL (`0x10`)

**Purpose:** Tile-local matrix multiply of a loaded weight tensor against an
activation input buffer.

**Required fields:**

| Field | Use |
|-------|-----|
| `tensor_id` | Weight tensor slot (must be nonzero) |
| `dtype` | Dtype of the weight tensor (f32/q8/q4) |
| `input_buf_addr` | Activation input buffer address |
| `input_buf_bytes` | Activation input byte count |
| `output_buf_addr` | Output buffer address |
| `output_buf_bytes` | Output byte count |
| `op_param_0` | Output rows |
| `op_param_1` | Output columns |

**Dtype support:** f32, q8, q4. For q8 and q4, the weight tensor must have
been loaded with matching dtype; the input and output buffers are always f32
activations.

**Placement implications:** Whether this is a row-split, column-split, or
full matmul depends on the placement report. See §13. The command itself does
not encode the split type; that is implicit in the tensor assignment.

**Counters updated on success:** `matmul_count++`, `bytes_read_estimate +=
input_buf_bytes`, `bytes_written_estimate += output_buf_bytes`.

**Simulated behavior:** No math executes. Returns `ATT1_AIMU_OK` after field
validation.

**Expected future behavior:** Loads weights from tile memory, multiplies by
activation input, writes result to output buffer. For q8/q4, dequantizes
per group before the multiply (or uses fused int8/int4 kernel).

---

### 6.3 EXEC_RMSNORM (`0x11`)

**Purpose:** Root Mean Square Layer Normalization applied to an activation
vector using a loaded norm weight tensor.

**Required fields:**

| Field | Use |
|-------|-----|
| `tensor_id` | Norm weight tensor slot (shape `[d_model]`) |
| `dtype` | Must be f32 (norm weights are always f32) |
| `input_buf_addr` | Input activation vector |
| `input_buf_bytes` | `d_model × sizeof(float32)` |
| `output_buf_addr` | Normalized output buffer |
| `output_buf_bytes` | `d_model × sizeof(float32)` |
| `op_param_0` | Vector length (d_model) |

**Dtype support:** f32 only. Norm weight tensors stored as q8 or q4 are not
supported in the current model; q-dtype norm tensors return
`ATT1_AIMU_ERR_UNSUPPORTED_DTYPE`.

**Placement implications:** Norm weights are typically replicated across all
tiles. Each tile applies RMSNorm locally to its activation slice before
dispatching to downstream matmuls.

**Expected future behavior:** Computes $\text{RMSNorm}(x) = x \cdot
\frac{w}{\sqrt{\text{mean}(x^2) + \varepsilon}}$ per the ATT-1 runtime
implementation.

---

### 6.4 EXEC_ROPE (`0x12`)

**Purpose:** Apply Rotary Position Embedding (RoPE) to query or key vectors
in-place.

**Required fields:**

| Field | Use |
|-------|-----|
| `dtype` | f32 |
| `input_buf_addr` | Q or K buffer (modified in-place) |
| `input_buf_bytes` | `n_heads × rope_dim × sizeof(float32)` |
| `output_buf_addr` | Same as `input_buf_addr` (in-place) |
| `op_param_0` | Vector length (number of elements) |
| `op_param_1` | Sequence position (alternative to `kv_position`) |
| `kv_position` | Sequence position (used if `op_param_1 == 0`) |

**Dtype support:** f32 only.

**Expected future behavior:** Applies pair-wise rotation to embedding
dimensions using precomputed sine/cosine tables parameterized by `rope_dim`
and `theta` (from model config). Behavior matches `att1_backend_ops.rope_f32`.

---

### 6.5 EXEC_ATTENTION (`0x13`)

**Purpose:** Scaled dot-product attention over query, key, and value tensors
for one layer on this tile.

**Required fields:**

| Field | Use |
|-------|-----|
| `tensor_id` | References K/V tensors (from KV cache) |
| `dtype` | f32 |
| `input_buf_addr` | Query buffer |
| `input_buf_bytes` | `n_heads × head_dim × sizeof(float32)` |
| `output_buf_addr` | Attention output buffer |
| `output_buf_bytes` | `n_heads × head_dim × sizeof(float32)` |
| `op_param_0` | Number of heads |
| `op_param_1` | Head dimension |
| `kv_position` | Current sequence position (determines KV read range) |

**Dtype support:** f32 activations. KV cache is always f32 in the current
model.

**Placement implications:** In head-split placement, each tile holds a disjoint
subset of heads. The tile executes attention for its assigned heads only.
Post-attention output vectors require a gather step across tiles before the
output projection matmul.

**Expected future behavior:** Computes $\text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V$
with causal masking. Matches `att1_backend_ops.softmax_f32` and
`att1_backend_ops.rope_f32` semantics.

---

### 6.6 EXEC_FFN (`0x14`)

**Purpose:** Feed-forward network (SwiGLU variant) applied to the normed
hidden state.

**Required fields:**

| Field | Use |
|-------|-----|
| `tensor_id` | FFN weight tensor slot (`w_gate`, `w_up`, `w_down` loaded separately) |
| `dtype` | f32, q8, or q4 |
| `input_buf_addr` | Input activation (`d_model` wide) |
| `input_buf_bytes` | `d_model × sizeof(float32)` |
| `output_buf_addr` | FFN output buffer (`d_model` wide) |
| `output_buf_bytes` | `d_model × sizeof(float32)` |
| `op_param_0` | Hidden dimension (d_model) |
| `op_param_1` | Intermediate dimension (d_ff) |

**Dtype support:** f32, q8, q4 for weight tensors. Activations are always f32.

**Placement implications:** In column-split FFN placement, each tile holds a
column slice of `w_gate`, `w_up`, and a row slice of `w_down`. Partial outputs
require a `PARTIAL_REDUCE` fabric step before the next layer's norm.

**Expected future behavior:** Computes
$\text{FFN}(x) = (\sigma(xW_{\text{gate}}) \odot xW_{\text{up}}) W_{\text{down}}$
where $\sigma$ is the SiLU activation. Matches `att1_backend_ops.ffn_swiglu_f32`.

---

## 7. KV Operations

### 7.1 KV_APPEND (`0x20`)

**Purpose:** Append a key and value vector for the current sequence position
to the tile-local KV cache for one layer.

**Required command fields:**

| Field | Requirement |
|-------|-------------|
| `tile_id` | Must be < `tile_count` |
| `session_id` | Identifies the inference session (0-based) |
| `op_param_0` | Layer index |
| `kv_position` | Sequence position to append at |
| `input_buf_addr` | Source buffer containing K and V vectors |
| `input_buf_bytes` | `2 × head_count × head_dim × sizeof(float32)` (one step) |

**KV ownership:** Each tile owns the KV cache for the layers assigned to it by
the shard plan. A tile must not receive `KV_APPEND` for layers it does not own;
that is a placement error, not a hardware error.

**Memory capacity:** The simulator adds `input_buf_bytes` to the tile's
`kv_used_bytes`. If this exceeds `kv_capacity_bytes`, the command fails with
`ATT1_AIMU_ERR_KV_OVERFLOW`.

**Counters:** `kv_append_count++` on success.

**Failure modes:**

| Condition | Result code |
|-----------|-------------|
| Invalid `session_id` | `ATT1_AIMU_ERR_INVALID_SESSION` |
| KV memory full | `ATT1_AIMU_ERR_KV_OVERFLOW` |
| Layer not owned by tile | Not detected by simulator; placement error |

---

### 7.2 KV_READ (`0x21`)

**Purpose:** Read a range of KV history for one layer from the tile-local
KV cache into a buffer for attention computation.

**Required command fields:**

| Field | Requirement |
|-------|-------------|
| `tile_id` | Must be < `tile_count` |
| `session_id` | Identifies the inference session |
| `op_param_0` | Layer index |
| `kv_position` | Number of previously appended positions to read |
| `output_buf_addr` | Destination buffer for K and V history |
| `output_buf_bytes` | `2 × head_count × head_dim × kv_position × sizeof(float32)` |

**Counters:** `kv_read_count++` on success.

**Failure modes:**

| Condition | Result code |
|-----------|-------------|
| `kv_position` exceeds stored history | `ATT1_AIMU_ERR_INVALID_SHAPE` |
| Invalid `session_id` | `ATT1_AIMU_ERR_INVALID_SESSION` |

---

## 8. Fabric Operations

### 8.1 Route types

Fabric traffic is described by route records in the fabric route report
(`map_commands_to_fabric_routes.py` output). Each route specifies a type:

| Code | Name | Topology | Description |
|------|------|----------|-------------|
| `0x01` | `ACTIVATION_SEND` | One-to-one | Send a complete activation vector from one tile to exactly one downstream tile |
| `0x02` | `ACTIVATION_BROADCAST` | One-to-many | Send the same activation vector to all tiles in `dest_tile_mask`; used before split-tensor matmul |
| `0x03` | `PARTIAL_REDUCE` | Many-to-one | Each source tile sends a partial result to the designated aggregator; aggregator accumulates (element-wise sum, `reduce_type=0`) |
| `0x04` | `LOGITS_REDUCE` | Many-to-one | Vocab-split lm_head: each tile sends a contiguous logit slice; aggregator concatenates in `slice_start` order (`reduce_type=1`) |
| `0x05` | `KV_TRANSFER` | One-to-one | Carry a KV page from a source tile to a new owner tile on explicit KV migration only; not issued during normal inference |
| `0x07` | `TRACE_EVENT` | Tile-to-host | Carry a 64-byte trace snapshot record from a tile to the host trace buffer (optional; not required in Phase 2) |
| `0x08` | `CONTROL_ACK` | One-to-one | Acknowledgment from a receiving tile confirming receipt; issued when `completion_fence_id` is set |

---

### 8.2 FABRIC_SEND (`0x30`)

**Purpose:** Send an activation or tensor slice from this tile to one
destination tile.

**Required fields:** `tile_id` (source), `op_param_0` (payload bytes),
`op_param_1` (destination tile ID), `dtype`, `session_id`.

**Ordering:** Activations sent between the same tile pair in the same session
are delivered in order. Cross-session ordering is not guaranteed.

**Payload byte rules:** `op_param_0` must be > 0 and ≤ the fabric's configured
maximum payload bytes. Exceeding the limit returns `ATT1_AIMU_ERR_FABRIC`.

**Counters:** `fabric_send_count++` on success.

**Expected future behavior:** Routes the payload through the AIMU fabric
interconnect to the destination tile's receive buffer, where an `EXEC_MATMUL`
or `EXEC_ATTENTION` command will consume it.

---

### 8.3 FABRIC_REDUCE (`0x31`)

**Purpose:** Send a partial result to the designated aggregator tile, which
accumulates results from all contributing tiles.

**Required fields:** `tile_id` (source), `op_param_0` (payload bytes),
`op_param_1` (`reduce_type`: 0=element-wise sum, 1=concatenate).

**Deterministic reduction order:** The aggregator tile accumulates partial
results in the deterministic order specified by the fabric route plan
(`reduction_id` and `slice_start` fields). Ordering is not timing-dependent.

**Tolerance implications for quantized dtypes:**

- f32 reduction: bit-exact with any reduction order (no quantization).
- q8 partial reduce: dequantization error `< 0.15 max_abs_error`; reduction
  order may affect final accumulated error within this bound.
- q4 partial reduce: dequantization error `< 0.35 max_abs_error`; token-level
  divergence is expected versus f32 reference.
- CUDA reduce: same tolerance bounds as CPU when operating on the same dtype;
  floating-point non-associativity may cause minor variation within tolerance.

**Counters:** `fabric_reduce_count++` on success.

---

### 8.4 TILE_BARRIER (`0x41`)

**Purpose:** Synchronize a set of tiles at a common point. No tile in the
barrier group proceeds past this command until all members have reached it.

**Required fields:** `tile_id`, `session_id`, `fence_id` (barrier group
identifier), `timeout_ms` (0 = no timeout).

**Timeout behavior:** If `timeout_ms > 0` and the barrier is not satisfied
within that period, the waiting tiles return `ATT1_AIMU_ERR_BARRIER_TIMEOUT`.

**Counters:** `barrier_count++` on success.

**Simulator behavior:** In-process barriers are implemented as pthread barrier
or atomic counter semantics within the tile thread pool. All threads reaching
the barrier synchronize before any proceeds.

---

## 9. Trace and Counter Operations

### 9.1 TRACE_SNAPSHOT (`0x40`)

**Purpose:** Capture a point-in-time snapshot of all AIMU control-plane
counters. The result is available via `att1_aimu_trace_snapshot_all()` and
`QUERY_COUNTERS`.

**Trigger methods:**
- Submit a `TRACE_SNAPSHOT` command to the command queue.
- Write `ATT1_MMIO_SNAP_NOW (1 << 0)` to `COUNTER_SNAPSHOT_CONTROL` register.

**Snapshot fields:**

| Sub-struct | Source | Contents |
|------------|--------|---------|
| `meta` | `att1_aimu_trace` | `trace_version` (0x00010000), `snapshot_id` (monotonic), `device_id`, `tile_count`, `event_count`, `dropped_events`, `status` |
| `cmdq` | `att1_aimu_cmdq` | `commands_submitted`, `commands_completed`, `commands_failed`, `queue_full_count`, `unsupported_commands`, `fence_value` |
| `device` | `att1_aimu_device` | `device_resets`, `tile_resets`, `tile_errors` |
| `dma` | `att1_aimu_dma` | `dma_submitted`, `dma_completed`, `dma_failed`, `bytes_host_to_device`, `bytes_device_to_host`, `bytes_device_to_device`, `alignment_failures`, `range_failures`, `unsupported_flags` |
| `fabric` | placeholder | `packets_sent`, `packets_received`, `payload_bytes_sent`, `payload_bytes_received`, `congestion_events` (all 0 in M108; wired in M109+) |

**Snapshot status codes:**

| Value | Constant | Meaning |
|-------|----------|---------|
| 0 | `ATT1_AIMU_TRACE_STATUS_OK` | All sources present; snapshot complete |
| 1 | `ATT1_AIMU_TRACE_STATUS_PARTIAL` | One or more sources NULL; corresponding sub-struct retains previous value |
| 2 | `ATT1_AIMU_TRACE_STATUS_EMPTY` | No snapshot taken yet |

**Deterministic replay:** Snapshot IDs are monotonically incrementing. A
replay run with the same command sequence must produce the same snapshot
values. This invariant is tested by the MMIO regression suite.

**Counters:** `trace_snapshot_count++` on success.

---

### 9.2 QUERY_COUNTERS (`0x51`)

**Purpose:** Read the current counter values from the tile into the host's
completion record or a host-provided output buffer.

**Behavior:** Equivalent to `att1_aimu_exec_ctx_get_counters()`. Returns the
`att1_aimu_exec_counters` snapshot at the time of the command's execution.
Does not reset counters (use `RESET_TILE` or
`att1_aimu_exec_ctx_reset_counters()` to reset).

---

### 9.3 EXEC counter set (`att1_aimu_exec_counters`)

The EXEC context maintains the following counters, all monotonically
non-decreasing within a session:

| Field | Incremented when |
|-------|-----------------|
| `exec_commands_seen` | Every call to `att1_aimu_exec_dispatch` |
| `exec_commands_completed` | Result is `ATT1_AIMU_OK` |
| `exec_commands_failed` | Result is any error code |
| `exec_unsupported` | Result is `ATT1_AIMU_ERR_UNSUPPORTED_OP` |
| `matmul_count` | `EXEC_MATMUL` returns `ATT1_AIMU_OK` |
| `rmsnorm_count` | `EXEC_RMSNORM` returns `ATT1_AIMU_OK` |
| `rope_count` | `EXEC_ROPE` returns `ATT1_AIMU_OK` |
| `attention_count` | `EXEC_ATTENTION` returns `ATT1_AIMU_OK` |
| `ffn_count` | `EXEC_FFN` returns `ATT1_AIMU_OK` |
| `kv_append_count` | `KV_APPEND` returns `ATT1_AIMU_OK` |
| `kv_read_count` | `KV_READ` returns `ATT1_AIMU_OK` |
| `fabric_send_count` | `FABRIC_SEND` returns `ATT1_AIMU_OK` |
| `fabric_reduce_count` | `FABRIC_REDUCE` returns `ATT1_AIMU_OK` |
| `barrier_count` | `TILE_BARRIER` returns `ATT1_AIMU_OK` |
| `trace_snapshot_count` | `TRACE_SNAPSHOT` returns `ATT1_AIMU_OK` |
| `bytes_read_estimate` | `+= input_buf_bytes` for OK `EXEC_*` |
| `bytes_written_estimate` | `+= output_buf_bytes` for OK `EXEC_*` and OK `LOAD_TENSOR_TILE` |

Counters are reset to zero by `att1_aimu_exec_ctx_reset_counters()` or by a
`RESET_TILE` command.

---

## 10. DMA Descriptor Model

### 10.1 Descriptor layout (64 bytes)

| Offset | Size | Field | Description |
|--------|------|-------|-------------|
| 0 | 8 | `host_addr` | Host buffer address (H2D / D2H) |
| 8 | 8 | `device_addr` | AIMU-local address (H2D / D2H) |
| 16 | 8 | `src_device_addr` | AIMU-local source address (D2D only) |
| 24 | 8 | `dst_device_addr` | AIMU-local destination address (D2D only) |
| 32 | 4 | `byte_length` | Transfer size in bytes (1 – 256 MiB) |
| 36 | 4 | `descriptor_id` | Host-assigned descriptor identifier |
| 40 | 4 | `command_id` | Owning M105 command ID |
| 44 | 4 | `tensor_id` | Target tensor slot on the AIMU tile |
| 48 | 4 | `checksum` | CRC32 placeholder (0 = not validated) |
| 52 | 2 | `dim0` | Logical row dimension |
| 54 | 2 | `dim1` | Logical column dimension |
| 56 | 2 | `flags` | `ATT1_AIMU_DMA_FLAG_*` bits |
| 58 | 1 | `dtype` | `0`=f32, `1`=q8, `2`=q4 |
| 59 | 1 | `quant_group_size` | q4: group size (16/32/64/128); q8/f32: must be 0 |
| 60 | 1 | `direction` | `0`=H2D, `1`=D2H, `2`=D2D |
| 61–63 | 3 | `_pad` | Reserved; must be zero |

### 10.2 Direction encoding

| Value | Constant | Description |
|-------|----------|-------------|
| `0` | `ATT1_AIMU_DMA_HOST_TO_DEVICE` | Host buffer → AIMU tile (load) |
| `1` | `ATT1_AIMU_DMA_DEVICE_TO_HOST` | AIMU tile → host buffer (readback) |
| `2` | `ATT1_AIMU_DMA_DEVICE_TO_DEVICE` | AIMU tile → AIMU tile (tile-local migration) |

### 10.3 Descriptor flags

| Bit | Constant | Description |
|-----|----------|-------------|
| `0x0001` | `VALIDATE_CHECKSUM` | AIMU verifies `checksum` on receipt (placeholder in M107) |
| `0x0002` | `GENERATE_CHECKSUM` | AIMU computes checksum for D2H transfers (placeholder in M107) |
| `0x0004` | `LAST_DESCRIPTOR` | Final descriptor in a chain; triggers completion |
| `0x0008` | `SCATTER_GATHER` | Part of a scatter-gather chain (unsupported; returns `ATT1_AIMU_ERR_INVALID_COMMAND`) |

Any bit outside `0x000F` set in `flags` causes a validation failure with
`ATT1_AIMU_ERR_INVALID_COMMAND`.

### 10.4 Alignment and range rules

| Constraint | Value | Error if violated |
|------------|-------|------------------|
| Minimum address alignment | 64 bytes (`ATT1_AIMU_DMA_ALIGN_BYTES`) | `ATT1_AIMU_ERR_ALIGNMENT` |
| Maximum transfer size | 256 MiB (`ATT1_AIMU_DMA_MAX_TRANSFER_BYTES = 0x10000000`) | `ATT1_AIMU_ERR_DMA_FAULT` |
| `byte_length` = 0 | Forbidden | `ATT1_AIMU_ERR_INVALID_COMMAND` |
| Device address out of tile capacity | — | `ATT1_AIMU_ERR_DMA_FAULT` |

### 10.5 q4 payload alignment

For q4 (`dtype = 2`) descriptors, the simulator verifies that `byte_length`
is consistent with the packed nibble layout:

```
expected = rows * cols / 2               (nibble data)
         + rows * (cols / group_size) * 4  (scale floats)
```

A mismatch returns `ATT1_AIMU_ERR_INVALID_SHAPE`.

### 10.6 No real DMA in simulator

The simulator validates descriptor fields and updates byte counters
(`bytes_host_to_device`, etc.). No actual memory copy occurs. No host physical
address is dereferenced. The `host_addr` and `device_addr` fields are treated
as opaque integers for range-check purposes only.

---

## 11. MMIO and Register Interaction

### 11.1 BAR0 overview

The simulated BAR0 is a 64 KiB region (`ATT1_AIMU_MMIO_BAR0_SIZE = 0x10000`)
backed by a heap-allocated `uint32_t[16384]` array. All register accesses use
32-bit aligned reads/writes. 64-bit counter registers are accessed as low/high
32-bit pairs; the low register offset must be 8-byte aligned.

Reserved register offsets return `0xDEADBEEF` on read. Writes to reserved
offsets are silently discarded.

### 11.2 BAR0 region map

| Offset range | Region | Description |
|-------------|--------|-------------|
| `0x0000–0x0FFF` | Global | Device identity, status, control, error, trace control |
| `0x1000–0x1FFF` | CQ | Command queue base, size, head, tail, doorbell, status, fence |
| `0x2000–0x2FFF` | DMA | DMA control, status, error, ring base/size |
| `0x3000–0x3FFF` | Fabric | Fabric status, control, route base, packet counters, error |
| `0x4000–0x4FFF` | Counters | 64-bit hardware counters (all RO) |
| `0x5000–0x5FFF` | Trace | Trace write pointer, read pointer, dropped events |
| `0x8000+N×0x800` | Tile N | Per-tile capability and status registers (N = 0–15) |

### 11.3 Global device registers (`0x0000–0x003F`)

| Offset | Name | Access | Description |
|--------|------|--------|-------------|
| `0x0000` | `DEVICE_ID` | RO | Fixed device identifier |
| `0x0004` | `DEVICE_VERSION` | RO | `major.minor.patch.build` packed |
| `0x0008` | `REGISTER_MAP_VERSION` | RO | `0x00010000` (v1.0) |
| `0x000C` | `FEATURE_FLAGS_LOW` | RO | Low 32 bits of `ATT1_AIMU_FEAT_*` bitmask |
| `0x0010` | `FEATURE_FLAGS_HIGH` | RO | High 32 bits |
| `0x0014` | `TILE_COUNT` | RO | Number of simulated tiles |
| `0x0018` | `COMMAND_QUEUE_COUNT` | RO | Number of command queues (1 per device in M105) |
| `0x001C` | `INTERRUPT_STATUS` | RW1C | Interrupt condition bits |
| `0x0020` | `INTERRUPT_ENABLE` | RW | Interrupt enable mask |
| `0x0024` | `GLOBAL_STATUS` | RO | `DEVICE_READY`, `ANY_TILE_ACTIVE`, `ANY_TILE_ERROR`, `FABRIC_ACTIVE`, `DMA_ACTIVE`, `TRACE_ACTIVE` |
| `0x0028` | `GLOBAL_CONTROL` | WO | `ENABLE_DEVICE`, `DISABLE_DEVICE`, `ENABLE_TRACE`, `DISABLE_TRACE`, `FLUSH_COMPL` |
| `0x002C` | `RESET_CONTROL` | WO | `SOFT_RESET_ALL`, `RESET_COUNTERS`, `RESET_TRACE`, `RESET_FABRIC` |
| `0x0030` | `ERROR_STATUS` | RW1C | Global error condition bits |
| `0x0034` | `ERROR_DETAIL` | RO | Last error detail code |
| `0x0038` | `TRACE_CONTROL` | RW | Trace enable / mode bits |
| `0x003C` | `COUNTER_SNAPSHOT_CONTROL` | RW | Bit 0 = `SNAP_NOW`: triggers `att1_aimu_trace_snapshot_all()` |

### 11.4 Per-tile registers (base `0x8000 + N × 0x800`)

Tile N's register window base: `ATT1_MMIO_BASE_TILE(N) = 0x8000 + N × 0x800`.

| Relative offset | Name | Access | Description |
|----------------|------|--------|-------------|
| `0x000` | `TILE_ID` | RO | Tile index |
| `0x004` | `TILE_STATUS` | RO | `att1_aimu_tile_state` (IDLE/ACTIVE/ERROR/RESET) |
| `0x008` | `TILE_FEATURE_FLAGS` | RO | Per-tile feature flags |
| `0x00C/0x010` | `TILE_MEMORY_CAPACITY_LO/HI` | RO | Total tile SRAM (bytes) |
| `0x014/0x018` | `TILE_MEMORY_USED_LO/HI` | RO | Currently allocated SRAM |
| `0x01C/0x020` | `TILE_KV_CAPACITY_LO/HI` | RO | Total KV memory (bytes) |
| `0x024/0x028` | `TILE_KV_USED_LO/HI` | RO | Currently used KV memory |
| `0x02C` | `TILE_SUPPORTED_DTYPES` | RO | `ATT1_AIMU_DTYPE_*` bitmask |
| `0x030` | `TILE_SUPPORTED_OPS_LOW` | RO | Low 32 bits of `ATT1_AIMU_OP_*` bitmask |
| `0x034` | `TILE_SUPPORTED_OPS_HIGH` | RO | High 32 bits (currently all zero; 9 ops defined) |
| `0x038` | `TILE_FABRIC_LINK_MASK` | RO | Bitmask of tiles this tile has direct links to |
| `0x03C` | `TILE_ERROR_STATUS` | RW1C | Per-tile error bits; cleared by writing 1 |
| `0x040` | `TILE_RESET_CONTROL` | WO | Write any nonzero value to reset this tile |

### 11.5 Command queue registers (`0x1000–0x102C`)

| Offset | Name | Access | Description |
|--------|------|--------|-------------|
| `0x1000/0x1004` | `CQ_BASE_ADDR_LO/HI` | RW | Base address of the command ring buffer |
| `0x1008` | `CQ_SIZE` | RW | Ring capacity in slots (power of two) |
| `0x100C` | `CQ_HEAD` | RO | Consumer (AIMU) head pointer |
| `0x1010` | `CQ_TAIL` | RW | Producer (host) tail pointer |
| `0x1014` | `CQ_DOORBELL` | WO | Write any value to notify AIMU of new commands |
| `0x1018` | `CQ_STATUS` | RO | Queue full/empty/active status bits |
| `0x101C` | `CQ_ERROR` | RW1C | Queue error status |
| `0x1020` | `CQ_FENCE_VALUE` | RO | Current monotonic fence counter value |
| `0x1024/0x1028` | `CQ_COMPLETION_ADDR_LO/HI` | RW | Completion ring base address |
| `0x102C` | `CQ_COMPLETION_SIZE` | RW | Completion ring capacity in slots |

### 11.6 Userspace MMIO emulator role

`att1_aimu_mmio_attach_device()`, `att1_aimu_mmio_attach_cmdq()`, and
`att1_aimu_mmio_attach_trace()` wire the in-process simulator state to the
BAR0 backing array. After calling `att1_aimu_mmio_sync()`, RO register cells
reflect the live device state. Host code reads and writes using
`att1_aimu_mmio_read32()` / `att1_aimu_mmio_write32()`, which enforce access
semantics (RO, RW, RW1C, WO) and alignment requirements.

The mmap-backed BAR0 file used by `att1-aimu-mmio-emulator` and
`att1-aimu-mmio-replay` exposes this same backing array as a file on disk,
allowing the emulator to work without a real MMIO mapping.

---

## 12. Dtype and Quantization Support

### 12.1 Dtype bitmask

Tile capability is reported as a `uint32` bitmask via `TILE_SUPPORTED_DTYPES`:

| Bit | Constant | Dtype |
|-----|----------|-------|
| `1 << 0` | `ATT1_AIMU_DTYPE_F32` | 32-bit IEEE-754 float |
| `1 << 1` | `ATT1_AIMU_DTYPE_Q8` | 8-bit symmetric quantization |
| `1 << 2` | `ATT1_AIMU_DTYPE_Q4` | 4-bit grouped quantization |

`ATT1_AIMU_DTYPE_ALL = 0x07` (all three). The default simulated tile supports
all three.

The `dtype` field in the command packet uses a separate encoding:
`0`=f32, `1`=q8, `2`=q4.

### 12.2 Op bitmask

Tile capability for operations:

| Bit | Constant | Operation |
|-----|----------|-----------|
| `1 << 0` | `ATT1_AIMU_OP_MATMUL` | EXEC_MATMUL |
| `1 << 1` | `ATT1_AIMU_OP_RMSNORM` | EXEC_RMSNORM |
| `1 << 2` | `ATT1_AIMU_OP_ROPE` | EXEC_ROPE |
| `1 << 3` | `ATT1_AIMU_OP_ATTENTION` | EXEC_ATTENTION |
| `1 << 4` | `ATT1_AIMU_OP_FFN` | EXEC_FFN |
| `1 << 5` | `ATT1_AIMU_OP_KV_APPEND` | KV_APPEND |
| `1 << 6` | `ATT1_AIMU_OP_KV_READ` | KV_READ |
| `1 << 7` | `ATT1_AIMU_OP_FABRIC_SEND` | FABRIC_SEND |
| `1 << 8` | `ATT1_AIMU_OP_FABRIC_REDUCE` | FABRIC_REDUCE |

`ATT1_AIMU_OP_ALL = 0x1FF`. The default simulated tile supports all nine.

### 12.3 q4 format summary

The q4 format used by AIMU commands matches the `.att1` artifact q4 wire
format:

| Parameter | Value |
|-----------|-------|
| Group size range | 16–128 (power of two) |
| Default group size | 32 (`ATT1_Q4_GROUP_SIZE_DEFAULT`) |
| Element range | `[-7, 7]` (signed; `-8` excluded) |
| Packing | Two elements per byte; low nibble = even index |
| Scale storage | `float32` per group, immediately after packed data |

For a tensor of shape `[rows, cols]` with group size `G`:
- Packed data: `rows × cols / 2` bytes.
- Scale data: `rows × (cols / G) × 4` bytes.

### 12.4 Unsupported dtype behavior

If a command's `dtype` maps to a dtype bit not set in the tile's
`supported_dtypes` bitmask, the dispatch returns
`ATT1_AIMU_ERR_UNSUPPORTED_DTYPE`. The command is not retried with a
different dtype. There is no silent fallback.

### 12.5 Feature flags

Device-level feature flags (`FEATURE_FLAGS_LOW/HIGH`) expose capabilities:

| Constant | Bit | Feature |
|----------|-----|---------|
| `ATT1_AIMU_FEAT_CMD_RING` | 0 | Command ring buffer |
| `ATT1_AIMU_FEAT_COMP_RING` | 1 | Completion ring buffer |
| `ATT1_AIMU_FEAT_DMA` | 2 | DMA engine |
| `ATT1_AIMU_FEAT_FABRIC` | 3 | Fabric interconnect |
| `ATT1_AIMU_FEAT_TRACE` | 4 | Trace buffer |
| `ATT1_AIMU_FEAT_COUNTERS` | 5 | Hardware performance counters |
| `ATT1_AIMU_FEAT_MSI_X` | 6 | MSI-X interrupt support |
| `ATT1_AIMU_FEAT_MULTI_SESSION` | 7 | Multiple concurrent inference sessions |
| `ATT1_AIMU_FEAT_FENCE` | 8 | Cross-tile fence ordering |
| `ATT1_AIMU_FEAT_KV_MMU` | 9 | KV-cache MMU (paged KV allocation) |
| `ATT1_AIMU_FEAT_PLACEMENT_AWARE` | 10 | Accepts placement report metadata |

Default simulator enables all except `ATT1_AIMU_FEAT_DMA` and
`ATT1_AIMU_FEAT_MSI_X` (not wired in M105–M108).

---

## 13. Tensor Placement Implications

Placement determines which tile executes which command and which tensors are
loaded on which tile. The placement report (output of `att1-size
--placement-report-json`) describes per-tile tensor assignments. The AIMU
command plan (`map_placement_to_commands.py` output) translates that into
concrete LOAD/EXEC/FABRIC commands. This section summarizes how placement
decisions affect the AIMU operation sequence.

### 13.1 Layer-wise placement (default: ATT1_SHARD_PLAN_RUNTIME)

Transformer layers are assigned contiguously across tiles:
- Tile 0 receives layers `0` through `k-1`.
- Tile 1 receives layers `k` through `2k-1`.
- Etc. (ceiling division; last tile may have fewer layers).

Each tile loads the full weight tensors for its assigned layers:
`attention.wq/wk/wv/wo`, `ffn.w_gate/w_up/w_down`, norms.

After each tile completes its layers, it sends the activation to the next
tile via `FABRIC_SEND` (`ACTIVATION_SEND` route type).

### 13.2 Tensor-wise split options

| Split type | Tensors split | EXEC affected | Fabric result |
|------------|--------------|--------------|---------------|
| Full (no split) | None; whole tensor on one tile | `EXEC_MATMUL` runs full matmul | `ACTIVATION_SEND` |
| Row-split | Rows of weight matrix split | Each tile produces a row slice of the output | `PARTIAL_REDUCE` (sum) |
| Column-split | Columns of weight matrix split | Each tile produces a full-width output on a column subset | `ACTIVATION_BROADCAST` before, gather after |
| Head-wise split | Attention heads split across tiles | Each tile's `EXEC_ATTENTION` covers its assigned heads | Post-attention gather via `ACTIVATION_SEND` |
| Embedding split | `tok_embeddings.weight` row-split | Embedding lookup distributed | `PARTIAL_REDUCE` (concat) |
| lm_head/vocab split | `output.weight` column-split | Each tile produces a logit slice | `LOGITS_REDUCE` (concat) |
| Norm replicated | Norms replicated on all tiles | Each tile applies `EXEC_RMSNORM` locally | None; no fabric step |

### 13.3 KV ownership

KV cache is tile-local. Each tile owns the KV cache for the layers assigned
to it. A `KV_APPEND` or `KV_READ` command for layer `L` must be sent to the
tile that owns layer `L` in the shard plan. Sending to the wrong tile is a
placement error; the simulator does not detect this as a hardware error.

### 13.4 Norm replication

RMSNorm weight vectors are small (`d_model` float32 values, typically 4–32 KiB
for small models) and are replicated on every tile that needs them. Each tile
applies `EXEC_RMSNORM` independently. No fabric step is required for norm
application.

---

## 14. Operation Status and Error Model

### 14.1 Result code table

The `att1_aimu_result` enum defines all AIMU operation result codes:

| Hex | Constant | Category | Description |
|-----|----------|----------|-------------|
| `0x00` | `ATT1_AIMU_OK` | Success | Operation completed successfully |
| `0x01` | `ATT1_AIMU_BUSY` | Non-fatal | Resource temporarily unavailable; retry expected |
| `0x02` | `ATT1_AIMU_PENDING` | Non-fatal | Command queued; completion not yet available |
| `0x10` | `ATT1_AIMU_ERR_INVALID_COMMAND` | Input error | Null ctx, null cmd, invalid magic, or unrecognized command type |
| `0x11` | `ATT1_AIMU_ERR_INVALID_TENSOR` | Input error | `tensor_id` not found, zero, or not loaded |
| `0x12` | `ATT1_AIMU_ERR_INVALID_DTYPE` | Input error | `dtype` field is unrecognized or mismatches the loaded tensor |
| `0x13` | `ATT1_AIMU_ERR_INVALID_SHAPE` | Input error | Shape mismatch (dim product, q4 layout, KV position range) |
| `0x14` | `ATT1_AIMU_ERR_INVALID_SESSION` | Input error | `session_id` not found or not active |
| `0x20` | `ATT1_AIMU_ERR_OUT_OF_MEMORY` | Resource | Tile tensor SRAM exhausted |
| `0x21` | `ATT1_AIMU_ERR_QUEUE_FULL` | Resource | Command ring at capacity; host must retry |
| `0x22` | `ATT1_AIMU_ERR_KV_OVERFLOW` | Resource | Tile KV memory exhausted |
| `0x30` | `ATT1_AIMU_ERR_FABRIC` | Fabric | General fabric error (invalid route, payload too large) |
| `0x31` | `ATT1_AIMU_ERR_FABRIC_TIMEOUT` | Fabric | Fabric send/receive timed out |
| `0x32` | `ATT1_AIMU_ERR_BARRIER_TIMEOUT` | Fabric | Barrier did not complete within `timeout_ms` |
| `0x40` | `ATT1_AIMU_ERR_CHECKSUM` | Integrity | CRC32 mismatch on command or DMA payload |
| `0x41` | `ATT1_AIMU_ERR_DMA_FAULT` | Integrity | DMA address out of valid range |
| `0x42` | `ATT1_AIMU_ERR_ALIGNMENT` | Integrity | Address or descriptor not properly aligned |
| `0x50` | `ATT1_AIMU_ERR_TIMEOUT` | Timeout | General operation timeout |
| `0x51` | `ATT1_AIMU_ERR_FENCE_DEADLOCK` | Timeout | Fence value can never be reached |
| `0x60` | `ATT1_AIMU_ERR_UNSUPPORTED_OP` | Capability | Operation not in tile's `supported_ops` bitmask |
| `0x61` | `ATT1_AIMU_ERR_UNSUPPORTED_DTYPE` | Capability | Dtype not in tile's `supported_dtypes` bitmask |
| `0xF0` | `ATT1_AIMU_ERR_INTERNAL` | Internal | Simulator assertion or unexpected state |
| `0xFF` | `ATT1_AIMU_ERR_FATAL` | Fatal | Non-recoverable error; tile must be reset |

### 14.2 Two-error-system boundary

ATT-1 maintains two distinct error systems. Do not mix them:

| System | Header | Type | Values | Used by |
|--------|--------|------|--------|---------|
| `att1_status_t` | `att1_status.h` | Signed int | Negative integers | C11 runtime API |
| `att1_aimu_result` | `att1_aimu_cmdq.h` | Enum / uint8 | Non-negative hex codes | AIMU command plane |

A function returning `att1_status_t` uses `ATT1_OK = 0`, `ATT1_ERR_* < 0`.
A function returning `att1_aimu_result` uses `ATT1_AIMU_OK = 0x00`,
`ATT1_AIMU_ERR_* > 0`.

### 14.3 UNSUPPORTED_OP replay policy

A command for an operation not in the tile's `supported_ops` bitmask returns
`ATT1_AIMU_ERR_UNSUPPORTED_OP`. This is an expected, deterministic failure
mode. In the M130 replay harness, commands with `expected_status =
ATT1_AIMU_ERR_UNSUPPORTED_OP` in the JSON plan are replayed and their result
code is verified against the expectation. A mismatch (e.g., the simulator
unexpectedly returns `OK`) is a test failure.

### 14.4 No silent fallback

No AIMU error code triggers a silent retry with a different:
- dtype (no auto-downcast from q4 to q8 or f32).
- operation (no substitution of a related op).
- tile (no command rerouting on tile error).
- backend (AIMU has no backend concept; CPU/CUDA is a separate runtime layer).

All failures are explicit.

---

## 15. Replay and Pipeline Tools

### 15.1 map_placement_to_commands.py

| Attribute | Detail |
|-----------|--------|
| **Purpose** | Translate an M98 tensor placement report JSON into an M109 AIMU command plan JSON |
| **Input** | `--report PATH` (placement report JSON) |
| **Output** | `--plan-json PATH` (command plan JSON) |
| **Pass behavior** | Exits 0; writes command plan |
| **Fail behavior** | Exits 1 if report fails schema validation; exits 1 if `--strict` and any capacity advisory is FAIL |

---

### 15.2 replay_aimu_command_plan.py

| Attribute | Detail |
|-----------|--------|
| **Purpose** | Python wrapper: submit each command in an M109 JSON plan to `att1-aimu-replay` (M112/M113 in-process harness) |
| **Input** | `--plan PATH` |
| **Output** | `--report-json PATH` (replay report JSON) |
| **Pass behavior** | Exits 0; all commands return expected result codes |
| **Fail behavior** | Exits 1 if any command's `result_code` differs from its `expected_status` in the plan, or if the binary returns non-zero |

---

### 15.3 replay_command_plan_via_mmio.py

| Attribute | Detail |
|-----------|--------|
| **Purpose** | Python wrapper: submit the M109 plan through the M121 userspace MMIO emulator (`att1-aimu-mmio-replay`) |
| **Input** | `--plan PATH`, `--tiles N` |
| **Output** | `--report-json PATH` |
| **Pass behavior** | Exits 0; MMIO emulator returns 0 |
| **Fail behavior** | Exits 1 if emulator returns non-zero or report indicates failures |

---

### 15.4 map_commands_to_fabric_routes.py

| Attribute | Detail |
|-----------|--------|
| **Purpose** | Generate an M115 fabric route report from an M109 command plan |
| **Input** | `--plan PATH` |
| **Output** | `--report-json PATH` (fabric route report JSON) |
| **Pass behavior** | Exits 0; writes route report with per-route records (type, source tile, dest tiles, payload bytes, reduction_id, slice_start/end, fence) |
| **Fail behavior** | Exits 1 on schema error; exits 1 if `--strict` and any validation check fails |

---

### 15.5 replay_fabric_routes.py

| Attribute | Detail |
|-----------|--------|
| **Purpose** | Simulate fabric route replay: process each route record, verify send/receive pairing, check bandwidth estimates |
| **Input** | `--route-report PATH` |
| **Output** | `--report-json PATH` |
| **Pass behavior** | Exits 0; all routes balanced and within bandwidth |
| **Fail behavior** | Exits 1 if any route is unmatched or bandwidth constraint fails |

---

### 15.6 run_aimu_planning_pipeline.py

| Attribute | Detail |
|-----------|--------|
| **Purpose** | 8-stage integrated planning pipeline (M119): placement validation → advisory → scenario → placement→commands → command replay → commands→routes → route validation → bandwidth simulation |
| **Input** | `--placement-report PATH`, `--workdir DIR` |
| **Output** | `--report-json PATH` (M119 integrated report); intermediate JSONs in `--workdir` |
| **Pass behavior** | Exits 0; all 8 stages PASS |
| **Fail behavior** | Exits 1 on any stage failure; `--strict` causes warnings to be treated as failures |

---

### 15.7 validate_tensor_execution_plan.py

| Attribute | Detail |
|-----------|--------|
| **Purpose** | Validate an M125 execution plan JSON against its schema |
| **Input** | `--plan PATH` |
| **Output** | `--report-json PATH` |
| **Pass behavior** | Exits 0; schema valid |
| **Fail behavior** | Exits 1 on schema violation; exits 1 if `--strict` and any WARNING-level issue found |

---

### 15.8 map_execution_plan_to_commands.py

| Attribute | Detail |
|-----------|--------|
| **Purpose** | Translate an M125 execution plan JSON into an M109 command plan JSON |
| **Input** | `--plan PATH` |
| **Output** | `--plan-json PATH` |
| **Pass behavior** | Exits 0 |
| **Fail behavior** | Exits 1 on validation error or unresolvable dependency ordering |

---

### 15.9 run_execution_replay_pipeline.py

| Attribute | Detail |
|-----------|--------|
| **Purpose** | 6-stage integrated execution/replay pipeline (M132): execution plan validation → plan→commands → command replay → commands→routes → route validation → fabric replay |
| **Input** | `--execution-plan PATH`, `--workdir DIR` |
| **Output** | `--report-json PATH` (M132 integrated report) |
| **Pass behavior** | Exits 0; all 6 stages PASS |
| **Fail behavior** | Exits 1 on any stage failure |

---

## 16. Implemented vs Future

### 16.1 Implemented (Phase 1, M105–M143)

| Component | Status | Source |
|-----------|--------|--------|
| Command ring-buffer simulator | Implemented | M105, `att1_aimu_cmdq.h` |
| Device/tile capability discovery | Implemented | M106, `att1_aimu_device.h` |
| DMA descriptor validator | Implemented | M107, `att1_aimu_dma.h` |
| Trace/counter snapshot | Implemented | M108, `att1_aimu_trace.h` |
| Command-plan mapper | Implemented | M109, `map_placement_to_commands.py` |
| In-process command replay harness | Implemented | M112/M113, `att1_aimu_host.h` |
| MMIO/register-file simulator | Implemented | M111, `att1_aimu_mmio.h` |
| Simulated EXEC/KV/FABRIC replay | Implemented | M130, `att1_aimu_exec.h` |
| Userspace MMIO emulator | Implemented | M121, `att1-aimu-mmio-emulator` |
| MMIO command-plan replay | Implemented | M122, `att1-aimu-mmio-replay` |
| Fabric route planner | Implemented | M115, `map_commands_to_fabric_routes.py` |
| Fabric route validator | Implemented | M116, `validate_fabric_routes.py` |
| Fabric bandwidth simulator | Implemented | M117, `simulate_fabric_bandwidth.py` |
| Fabric route replay | Implemented | M118, `replay_fabric_routes.py` |
| Integrated planning pipeline | Implemented | M119, `run_aimu_planning_pipeline.py` |
| Execution plan | Implemented | M125, `plan_tensor_execution.py` |
| Execution plan validator | Implemented | M128, `validate_tensor_execution_plan.py` |
| Execution plan → commands | Implemented | M129, `map_execution_plan_to_commands.py` |
| Integrated execution/replay pipeline | Implemented | M132, `run_execution_replay_pipeline.py` |
| JSON schema fuzz harness | Implemented | M143, `fuzz_json_schemas.py` |

All implemented components operate in-process. No real PCIe transactions,
DMA transfers, or tensor math executes through these paths.

### 16.2 Future/planned

| Component | Target | Notes |
|-----------|--------|-------|
| Real PCIe endpoint (physical BAR0) | Phase 3 | Requires PCIe switch board; not specified |
| VFIO/UIO kernel interface | Phase 3 | Enables userspace DMA from host to real AIMU |
| Linux kernel driver | Phase 3 | PCIe enumeration, MSI-X, IOMMU |
| FPGA BAR0 implementation | Phase 3 | See `docs/fpga_feasibility.md`; feasibility study only |
| Real DMA engine | Phase 3 | Physical descriptor queue, coherence |
| AIMU tensor math hardware | Phase 3 | Hardware matmul, attention, FFN execution |
| Tensor-level placement execution with real math | Phase 2 bridge | Connect EXEC replay to actual C11 math |
| Hardware fabric routing | Phase 3 | Physical interconnect; currently in-process simulation |
| CUDA kernel for AIMU-style tiled matmul | Not planned | CPU/CUDA backends are separate from AIMU command path |
| Distributed inference over real network | Phase 3 | Fabric is currently in-process only |

Phase 3 direction is advisory. No specific ASIC or silicon tape-out is planned
or committed. Hardware feasibility is subject to separate engineering review.

---

## 17. Non-Goals

The following are explicitly out of scope for the AIMU simulator and this
manual:

| Non-goal | Notes |
|----------|-------|
| Patent claim language | Must not appear in any repository file |
| Production ASIC design or silicon tape-out | Not committed; no IP claim |
| Real PCIe endpoint or physical BAR0 mapping | Userspace emulator only |
| Linux kernel driver | No kernel code in this repository |
| FPGA RTL or Verilog synthesis | Feasibility notes only |
| Real DMA engine or IOMMU interaction | All DMA is metadata validation only |
| Real-time hardware fabric or interconnect | In-process ring-buffer only |
| Mobile, Android, Vulkan, or OpenCL targets | Not planned |
| Public model weights in Git | External only; never committed |
| Automatic dtype or backend fallback | All failures are explicit |
| Inference math in AIMU EXEC path | CPU/CUDA runtime handles math |
| Hardware interrupt controller (MSI-X) | Feature flag present; not wired |

---

*End of AIMU Intrinsics and Operations Reference Manual (M147)*
