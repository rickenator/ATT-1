# AIMU/PCIe Command Packet Requirements (Milestone 103)

This document defines the Phase 3 AIMU/PCIe prototype command and control
model using the current ATT-1 simulator, tensor placement reports, q4/q8/f32
backends, shard metadata, and placement scenario tools as reference material.

This is a specification document only.  No C runtime, binary format, or
inference behavior is changed.

---

## 1. Prototype Goal

### 1.1 What the PCIe/AIMU Prototype Must Prove

The ATT-1 Phase 3 PCIe/AIMU prototype must demonstrate the following, at
minimum, before transition to custom silicon:

1. **Tensor tile residency.** A full tensor tile (model weights + KV memory +
   activation scratchpad) can reside in AIMU-local memory with no weight
   movement to the host during inference.

2. **Near-memory execution.** Matmul, RMSNorm, RoPE, FFN/SwiGLU, and attention
   operations can be initiated by a compact command packet rather than by the
   host fetching or writing tensor data.

3. **Activation-only host interface.** The host exchanges only activation
   vectors and control packets with each AIMU tile; no weight data crosses the
   PCIe bus during steady-state inference.

4. **Multi-tile coordination.** Two or more AIMU tiles can cooperate on a
   decode step via the inference fabric without host arbitration in the critical
   path.

5. **Deterministic trace.** For a given input token sequence and placement
   plan, the command packet trace and counter values are identical across runs.

6. **Placement report fidelity.** The per-tile memory footprint observed at
   runtime matches the M100 placement report estimate within a defined
   tolerance (≤ 5 % for model bytes; KV bytes are exact by definition).

7. **No silent fallback.** Any unsupported command type, dtype, or tile
   configuration must return a well-formed error response packet rather than
   silently falling back to an alternative execution path.

### 1.2 Relationship to ATT-1 Software Simulator

The ATT-1 C11 simulator is the canonical behavioral reference for the
hardware prototype.  The mapping is:

| ATT-1 simulator concept | PCIe/AIMU hardware equivalent |
|---|---|
| `att1_shard_t` (one per tile) | AIMU tile local execution context |
| `att1_cluster_infer_t` | Host command dispatcher across tiles |
| `att1_fabric_t` / `sim_fabric_bus.c` | AIMU-to-AIMU fabric interconnect |
| `att1_kv_cache_t` | AIMU-local KV SRAM or HBM bank |
| `att1_trace_t` counter fields | AIMU hardware performance counters |
| `run_command()` in bench tests | PCIe command packet submission |
| `check_placement_*_smoke()` | Placement report→command-plan mapping test |

The software simulator runs on the host CPU and provides a behavioral
oracle for all command types defined in §4.  Hardware validation compares
each command result against the software oracle to within established
tolerances:

| Backend | Max token divergence allowed |
|---|---|
| f32 reference | 0 (exact match) |
| q8 | 0 (same output token) |
| q4 | 0 (same output token within q4 tolerance 0.35) |

### 1.3 Relationship to CUDA Validation

The CUDA backend (M87–M92) serves as a secondary oracle for AIMU compute
validation:

- CUDA q4/q8/f32 matmul, RMSNorm, RoPE, FFN/SwiGLU validated against CPU
  f32 reference (M14–M22, M87–M92).
- The AIMU hardware target must match CUDA numeric outputs for the same
  dtype/op within the same tolerance thresholds.
- The CUDA cluster path (`mode=cluster`) provides the multi-tile data movement
  model that the AIMU fabric must reproduce in hardware.

### 1.4 Relationship to Future Custom Silicon

The PCIe prototype serves as the Phase 2 engineering validation platform.
Phase 3 custom silicon replaces the PCIe card AIMU controller with an
on-chip AIMU fabric while retaining the same:

- Command packet format (§3)
- Command type enumeration (§4)
- Memory model (§6)
- Execution model (§7)
- Counter and trace interface (§8)

The host control plane (§2) transitions from a PCIe BAR-mapped register
interface to a high-speed chip-to-chip link, but the logical command/response
protocol remains identical.

---

## 2. Host/AIMU Control Plane

### 2.1 Device Discovery

On PCIe enumeration:

1. The host scans for AIMU devices by vendor ID / device class.
2. Each physical AIMU card reports a `device_info` register block in BAR0:

| Field | Width | Description |
|---|---|---|
| `magic` | 4 B | `0x41544D55` ("ATMU") |
| `protocol_version` | 2 B | Command protocol version (currently 1) |
| `firmware_version` | 4 B | Major.minor.patch.build packed uint32 |
| `device_id` | 4 B | Board-level unique identifier |
| `aimu_tile_count` | 2 B | Number of AIMU tiles on this card |
| `max_sessions` | 2 B | Maximum simultaneous inference sessions |
| `capabilities` | 8 B | Feature flags (see §2.7) |
| `reserved` | 6 B | Zero |

Total: 32 bytes, 8-byte aligned.

### 2.2 Tile Enumeration

After device discovery, the host reads a tile descriptor array from BAR0 at a
fixed offset.  Each tile descriptor is 64 bytes:

| Field | Width | Description |
|---|---|---|
| `tile_id` | 2 B | Zero-based tile index within the device |
| `tile_status` | 1 B | `0=idle, 1=active, 2=error, 3=reserved` |
| `fabric_port_count` | 1 B | Number of fabric ports on this tile |
| `local_memory_mib` | 2 B | Local SRAM/HBM capacity in MiB |
| `staging_buffer_kib` | 2 B | Host-writable staging buffer size in KiB |
| `command_queue_depth` | 2 B | Maximum outstanding commands per tile |
| `supported_dtypes` | 2 B | Dtype capability flags (see §2.4) |
| `supported_ops` | 4 B | Op capability flags (see §2.5) |
| `firmware_tile_rev` | 4 B | Per-tile firmware revision |
| `reserved` | 44 B | Zero |

### 2.3 Memory Capacity Discovery

The host reads per-tile memory layout from the tile descriptor and a memory
map register:

| Field | Width | Description |
|---|---|---|
| `tensor_memory_base` | 8 B | Physical address of tensor memory region |
| `tensor_memory_bytes` | 8 B | Total tensor memory capacity |
| `kv_memory_base` | 8 B | Physical address of KV cache region |
| `kv_memory_bytes` | 8 B | KV cache capacity |
| `staging_base` | 8 B | Physical address of host-writable staging buffer |
| `staging_bytes` | 4 B | Staging buffer capacity in bytes |
| `command_queue_base` | 8 B | Physical address of command ring buffer |
| `command_queue_bytes` | 4 B | Command ring buffer capacity |
| `trace_memory_base` | 8 B | Physical address of trace/counter memory |
| `trace_memory_bytes` | 4 B | Trace memory capacity |
| `reserved` | 4 B | Zero |

Alignment requirements:
- Tensor memory: 256-byte aligned (accommodates f32, q8, and q4 group alignment)
- KV memory: 64-byte aligned
- Staging buffer: 64-byte aligned
- Command queue: 64-byte aligned (one command is 64 bytes; see §3)
- Trace memory: 8-byte aligned

### 2.4 Supported Dtype Discovery

The `supported_dtypes` field in the tile descriptor is a bitmask:

| Bit | Value | Meaning |
|---|---|---|
| 0 | `DTYPE_F32` | f32 reference inference |
| 1 | `DTYPE_Q8` | per-row int8 (q8) quantized inference |
| 2 | `DTYPE_Q4` | grouped int4 (q4, group_size=32) quantized inference |
| 3–15 | reserved | Must be zero |

All three dtypes (f32, q8, q4) must be supported for full ATT-1 compatibility.
A prototype card may report only `DTYPE_F32 | DTYPE_Q8` during initial
validation and add `DTYPE_Q4` once the q4 dequantize pipeline is verified.

### 2.5 Supported Op Discovery

The `supported_ops` field in the tile descriptor is a bitmask:

| Bit | Op | ATT-1 backend equivalent |
|---|---|---|
| 0 | `OP_MATMUL` | `backend_ops.matmul_f32` / `matmul_q8xf32` / `matmul_q4xf32` |
| 1 | `OP_RMSNORM` | `backend_ops.rmsnorm_f32` |
| 2 | `OP_ROPE` | `backend_ops.rope_f32` |
| 3 | `OP_ATTENTION` | `att1_attention_forward_backend` |
| 4 | `OP_FFN_SWIGLU` | `backend_ops.ffn_swiglu_f32` |
| 5 | `OP_KV_APPEND` | `att1_kv_cache_append` |
| 6 | `OP_KV_READ` | `att1_kv_cache_read` |
| 7 | `OP_FABRIC_SEND` | `att1_fabric_send` |
| 8 | `OP_FABRIC_REDUCE` | partial-result accumulation via fabric |
| 9 | `OP_TRACE_SNAPSHOT` | `att1_trace_snapshot` |
| 10 | `OP_TILE_BARRIER` | `att1_fabric_barrier` |
| 11 | `OP_QUERY_COUNTERS` | `att1_trace_read_counters` |
| 12–31 | reserved | Must be zero |

### 2.6 Firmware/Runtime Version Fields

Version discovery is two-level:

1. **Device-level**: `firmware_version` in `device_info` (§2.1).
2. **Tile-level**: `firmware_tile_rev` in tile descriptor (§2.2).

Protocol compatibility: a host driver must check `protocol_version` before
issuing any command.  If `protocol_version > HOST_KNOWN_VERSION`, the host
must fall back to the lowest common version or refuse the device.

### 2.7 Feature Flags

The `capabilities` bitmask in `device_info`:

| Bit | Flag | Description |
|---|---|---|
| 0 | `CAP_PLACEMENT_REPORT` | Card can consume M100 placement report JSON |
| 1 | `CAP_ASYNC_COMPLETION` | Commands complete asynchronously; completion IRQ or polling |
| 2 | `CAP_FENCE` | Dependency fences supported (see §7.2) |
| 3 | `CAP_PARTIAL_REDUCE` | Hardware partial-logit reduction across tiles |
| 4 | `CAP_Q4_ALIGN32` | Q4 group_size=32 alignment hardware enforced |
| 5 | `CAP_TRACE_PER_TOKEN` | Per-token trace hooks in hardware |
| 6 | `CAP_COUNTER_STALL` | Stall-reason counters (§8.8) |
| 7–63 | reserved | Must be zero |

### 2.8 Error and Status Reporting

Error reporting is per-command (§3.15) and per-tile:

**Tile error register** (4 bytes, polled or IRQ-driven):

| Bits | Field | Description |
|---|---|---|
| 7:0 | `last_error_code` | Most recent error (0 = no error) |
| 15:8 | `error_tile_id` | Tile that raised the error |
| 23:16 | `error_cmd_type` | Command type that caused the error |
| 31:24 | `error_flags` | `bit0=fatal, bit1=recoverable, bit2=trace_available` |

Error codes (8-bit namespace):

| Code | Name | Meaning |
|---|---|---|
| `0x00` | `ERR_NONE` | No error |
| `0x01` | `ERR_UNSUPPORTED_CMD` | Command type not supported on this tile |
| `0x02` | `ERR_UNSUPPORTED_DTYPE` | Dtype not supported for this op |
| `0x03` | `ERR_TENSOR_NOT_LOADED` | Tensor_id not resident in tile memory |
| `0x04` | `ERR_OUT_OF_MEMORY` | Insufficient tile memory for operation |
| `0x05` | `ERR_ALIGNMENT` | Buffer alignment violation |
| `0x06` | `ERR_CHECKSUM` | Tensor checksum mismatch (§6.8) |
| `0x07` | `ERR_TIMEOUT` | Command exceeded timeout |
| `0x08` | `ERR_FENCE_DEADLOCK` | Dependency fence unresolvable |
| `0x09` | `ERR_SESSION_OVERFLOW` | Session ID exceeds `max_sessions` |
| `0x0A` | `ERR_KV_OVERFLOW` | KV cache position exceeds `target_context_length` |
| `0x0B` | `ERR_FABRIC_SEND` | Fabric packet routing failed |
| `0x10`–`0xFE` | reserved | Reserved for future use |
| `0xFF` | `ERR_FATAL` | Tile halted; RESET_TILE required |

---

## 3. Command Packet Format

All commands are submitted to the per-tile command ring buffer as fixed-size
**64-byte** command packets.  The ring buffer is PCIe BAR-mapped and
host-writable.  The AIMU reads command packets in-order within a tile;
cross-tile ordering is enforced via fences (§7.2).

### 3.1 Command Packet Layout

| Offset | Size | Field | Description |
|---|---|---|---|
| 0 | 4 B | `command_id` | Monotonically increasing uint32; wraps at 2^32 |
| 4 | 1 B | `command_type` | See §4 |
| 5 | 1 B | `tile_id` | Target AIMU tile index (0-based) |
| 6 | 1 B | `session_id` | Inference session (0–`max_sessions-1`) |
| 7 | 1 B | `dtype` | `0=f32, 1=q8, 2=q4, 3=reserved` |
| 8 | 2 B | `model_id` | Logical model identifier (set at load time) |
| 10 | 2 B | `tensor_id` | Tensor index within the model (M98 §3 tensor categories) |
| 12 | 8 B | `input_buf_addr` | Host-physical or AIMU-local input buffer address |
| 20 | 4 B | `input_buf_bytes` | Input buffer length in bytes |
| 24 | 8 B | `output_buf_addr` | Host-physical or AIMU-local output buffer address |
| 32 | 4 B | `output_buf_bytes` | Output buffer length in bytes |
| 36 | 4 B | `kv_position` | Current KV cache token position (for KV commands) |
| 40 | 4 B | `op_param_0` | Operation-specific parameter 0 (see §4) |
| 44 | 4 B | `op_param_1` | Operation-specific parameter 1 |
| 48 | 2 B | `fence_id` | Dependency fence to wait on before executing (0=none) |
| 50 | 2 B | `completion_fence_id` | Fence this command signals upon completion (0=none) |
| 52 | 1 B | `trace_flags` | Bitmask; `bit0=trace_on, bit1=counter_snapshot` |
| 53 | 1 B | `priority` | `0=normal, 1=high, 2=preempt` (advisory) |
| 54 | 2 B | `timeout_ms` | Command timeout in milliseconds (0=no timeout) |
| 56 | 4 B | `checksum` | CRC32 of bytes 0–55; computed by host, verified by AIMU |
| 60 | 4 B | `status` | AIMU-written result code; 0=pending, see §2.8 after execution |

Total: 64 bytes.

The host writes bytes 0–59 (including `checksum`).  The AIMU writes `status`
(bytes 60–63) and, if `trace_flags.bit1` is set, a 64-byte trace record to
the tile's trace memory region.

### 3.2 Field Notes

**`command_id`**: The host increments this per-tile, per-session.  The AIMU
echoes `command_id` in the completion record (§7.4) so the host can match
responses to requests.

**`dtype`**: Must match the tensor dtype loaded at `tensor_id`.  If there is
a mismatch, the AIMU returns `ERR_UNSUPPORTED_DTYPE` without executing the op.

**`input_buf_addr` / `output_buf_addr`**: May be:
- A host DMA address (host-to-AIMU or AIMU-to-host transfer)
- An AIMU-local staging buffer address (near-memory path; no PCIe in hot path)
- A fabric routing address for AIMU-to-AIMU traffic

The high 4 bits of the address encode the address space:
`0x0...` = host DMA, `0x1...` = AIMU-local, `0x2...` = fabric.

**`fence_id` / `completion_fence_id`**: See §7.2.  A fence ID of 0 means no
dependency (execute immediately or in-order within the tile's queue).

**`checksum`**: CRC32/ISO-HDLC of the first 56 bytes of the command packet.
The AIMU verifies the checksum before dispatching; a mismatch returns
`ERR_CHECKSUM` without side effects.

---

## 4. Required Command Types

The following command types must be supported by all ATT-1-compatible AIMU
hardware.  The `command_type` byte value is shown in parentheses.

### 4.1 `LOAD_TENSOR_TILE` (0x01)

Transfers a tensor slice from the host staging buffer into AIMU local tensor
memory and registers it under `tensor_id`.

| Field | Meaning |
|---|---|
| `input_buf_addr` | Host DMA source address |
| `input_buf_bytes` | Tensor payload size in bytes |
| `tensor_id` | Tensor slot to register |
| `dtype` | Tensor dtype |
| `op_param_0` | Quantization group size (q4: 32, q8: row stride, f32: 0) |
| `op_param_1` | Tensor shape hint: `dim0[31:16] | dim1[15:0]` |

On completion, the tensor is resident in local memory and available to
`EXEC_*` commands that reference the same `tensor_id`.

### 4.2 `VALIDATE_TENSOR_TILE` (0x02)

Verifies that a loaded tensor's checksum matches the expected value from the
M100 placement report.  Does not modify tensor data.

| Field | Meaning |
|---|---|
| `tensor_id` | Tensor to validate |
| `input_buf_addr` | Host address of 32-byte expected-checksum record |
| `input_buf_bytes` | Must be 32 |

Returns `ERR_CHECKSUM` if the computed checksum does not match.

### 4.3 `EXEC_MATMUL` (0x10)

Execute a matrix-multiply:  `output = A × B`, where A is the input activation
vector and B is the resident weight tensor at `tensor_id`.

| Field | Meaning |
|---|---|
| `input_buf_addr` | Activation input vector (d_model float32 elements) |
| `input_buf_bytes` | `d_model × 4` bytes |
| `output_buf_addr` | Result buffer |
| `output_buf_bytes` | `d_out × 4` bytes |
| `tensor_id` | Resident weight matrix (d_out × d_model) |
| `dtype` | Weight dtype (f32/q8/q4) |
| `op_param_0` | Output dimension d_out |
| `op_param_1` | Accumulate flag: `0=overwrite, 1=add` (for partial-result tiles) |

### 4.4 `EXEC_RMSNORM` (0x11)

Apply RMSNorm in-place or to a result buffer using a resident norm weight
tensor.

| Field | Meaning |
|---|---|
| `input_buf_addr` | Input activation vector |
| `input_buf_bytes` | `d_model × 4` bytes |
| `output_buf_addr` | Output buffer (may alias input for in-place) |
| `tensor_id` | Resident RMSNorm weight vector |
| `op_param_0` | `eps_bits` — IEEE-754 bits of the RMSNorm epsilon (e.g., 1e-6) |

### 4.5 `EXEC_ROPE` (0x12)

Apply Rotary Position Embedding to a Q or K activation buffer.

| Field | Meaning |
|---|---|
| `input_buf_addr` | Q or K activation buffer (modified in-place) |
| `input_buf_bytes` | `n_heads × head_dim × 4` bytes |
| `kv_position` | Current token position (used to compute rotation angles) |
| `op_param_0` | `n_heads[31:16] | head_dim[15:0]` |
| `op_param_1` | `theta_bits` — IEEE-754 bits of the RoPE base theta |

### 4.6 `EXEC_ATTENTION` (0x13)

Execute causal self-attention for one decode step.  Reads KV cache, applies
softmax, returns attention output.

| Field | Meaning |
|---|---|
| `input_buf_addr` | Packed Q/K/V activation buffer |
| `input_buf_bytes` | `3 × n_heads × head_dim × 4` bytes |
| `output_buf_addr` | Attention output buffer |
| `output_buf_bytes` | `n_heads × head_dim × 4` bytes |
| `kv_position` | Current token position |
| `op_param_0` | `n_heads[31:16] | n_kv_heads[15:0]` |
| `op_param_1` | `head_dim` |

GQA (grouped-query attention) is supported when `n_kv_heads < n_heads`.

### 4.7 `EXEC_FFN` (0x14)

Execute FFN/SwiGLU forward pass using resident gate/up/down weight tensors.

| Field | Meaning |
|---|---|
| `input_buf_addr` | Input activation (post-attention, post-norm) |
| `input_buf_bytes` | `d_model × 4` bytes |
| `output_buf_addr` | FFN output |
| `output_buf_bytes` | `d_model × 4` bytes |
| `tensor_id` | First of three consecutive tensor slots (gate=`tensor_id`, up=`tensor_id+1`, down=`tensor_id+2`) |
| `op_param_0` | `ffn_hidden` |

### 4.8 `KV_APPEND` (0x20)

Append a K/V pair to the local KV cache at the given session/position.

| Field | Meaning |
|---|---|
| `input_buf_addr` | K vector followed immediately by V vector |
| `input_buf_bytes` | `2 × n_kv_heads × head_dim × 4` bytes |
| `session_id` | KV session slot |
| `kv_position` | Position to append |
| `op_param_0` | `n_kv_heads[31:16] | head_dim[15:0]` |

Returns `ERR_KV_OVERFLOW` if `kv_position ≥ target_context_length`.

### 4.9 `KV_READ` (0x21)

Read a range of K/V vectors from the KV cache into a result buffer.

| Field | Meaning |
|---|---|
| `output_buf_addr` | Output buffer for K then V vectors |
| `output_buf_bytes` | `position_count × 2 × n_kv_heads × head_dim × 4` bytes |
| `session_id` | KV session slot |
| `kv_position` | Start position |
| `op_param_0` | `position_count` (number of KV pairs to read) |
| `op_param_1` | `n_kv_heads[31:16] | head_dim[15:0]` |

### 4.10 `FABRIC_SEND` (0x30)

Route an activation, partial result, or control payload to one or more peer
AIMU tiles through the inference fabric.

| Field | Meaning |
|---|---|
| `input_buf_addr` | AIMU-local source buffer |
| `input_buf_bytes` | Payload size |
| `output_buf_addr` | Fabric routing address (encodes destination tile set) |
| `op_param_0` | Packet type: `0=activation, 1=partial_logit, 2=barrier, 3=control` |
| `op_param_1` | Destination bitmask (tiles 0–31) |

### 4.11 `FABRIC_REDUCE` (0x31)

Collect partial logit vectors from peer tiles, sum them element-wise, and
write the accumulated result to a local output buffer.  Used for vocab-split
and head-wise-split lm_head across tiles.

| Field | Meaning |
|---|---|
| `output_buf_addr` | AIMU-local accumulation output |
| `output_buf_bytes` | `vocab_size × 4` bytes for logit reduce; `n_heads × head_dim × 4` for head reduce |
| `op_param_0` | `reduce_type: 0=sum, 1=concat` |
| `op_param_1` | Expected participant tile bitmask |

The command blocks until all participating tiles have sent their partial
result or `timeout_ms` expires.

### 4.12 `TRACE_SNAPSHOT` (0x40)

Write a full counter snapshot for this tile to the trace memory region at the
current write pointer.

| Field | Meaning |
|---|---|
| `op_param_0` | Snapshot trigger: `0=on_command, 1=on_prefill_boundary, 2=on_decode_boundary` |
| `output_buf_addr` | Optional host DMA destination for immediate delivery; 0 = write to tile trace memory only |

The 64-byte snapshot format mirrors the ATT-1 `att1_trace_t` structure (see
§8.1).

### 4.13 `TILE_BARRIER` (0x41)

Wait until all tiles in the participant bitmask have reached this fence point.
Used to synchronize multi-tile decode steps at layer boundaries.

| Field | Meaning |
|---|---|
| `op_param_0` | Participant tile bitmask |
| `op_param_1` | Barrier phase tag (0–255; matches across all participating tiles) |
| `timeout_ms` | Barrier timeout (independent of command-level timeout) |

### 4.14 `RESET_TILE` (0x50)

Flush all in-progress commands, clear all tensor registrations, reset the KV
cache, and reset all counters to zero for the specified session or for all
sessions.

| Field | Meaning |
|---|---|
| `session_id` | Session to reset; `0xFF` = reset all sessions and full tile |
| `op_param_0` | `reset_flags`: `bit0=keep_tensors, bit1=keep_counters, bit2=keep_sessions` |

After `RESET_TILE` completes, the tile is in idle state and accepts new
`LOAD_TENSOR_TILE` commands.

### 4.15 `QUERY_COUNTERS` (0x51)

Return the current value of all hardware performance counters for this tile
as a DMA to the host.

| Field | Meaning |
|---|---|
| `output_buf_addr` | Host DMA destination buffer (≥ 256 bytes) |
| `output_buf_bytes` | Must be ≥ 256 |
| `op_param_0` | Counter group bitmask (0 = all groups; see §8) |

---

## 5. Data Movement Model

### 5.1 Host-to-AIMU DMA (Weight Loading)

Model tensor tiles are transferred once at session setup:

1. Host writes the tensor payload to the PCIe staging buffer on the AIMU card.
2. Host issues `LOAD_TENSOR_TILE` with `input_buf_addr = staging buffer address`.
3. AIMU DMA engine copies from staging buffer into local tensor memory.
4. AIMU returns status `0x00` (success) when the tensor is fully resident.

Transfer bandwidth expectation: PCIe Gen4 x16 (~32 GB/s) for initial load.
This path is **not** in the hot path during inference.

### 5.2 AIMU-to-Host DMA (Logit Return)

After the final lm_head projection (or FABRIC_REDUCE across tiles), the
resulting logit vector is DMA'd to a host buffer:

1. AIMU executes `EXEC_MATMUL` for lm_head; output goes to AIMU-local buffer.
2. Host issues `QUERY_COUNTERS` or reads logits via a dedicated output ring
   buffer (implementation choice: register-mapped or DMA descriptor).
3. Host reads logit vector, applies sampler (temperature, top-k, top-p).

The logit vector is `vocab_size × 4` bytes (f32).  For a 32K-vocab model this
is 128 KiB per decode step — the primary host-facing data movement per token.

### 5.3 AIMU-to-AIMU Fabric Movement

Between tiles, the inference fabric carries:

| Payload | Size | Frequency |
|---|---|---|
| Activation vector (Q/K/V/attn out) | `d_model × 4` bytes | Every layer boundary |
| Partial logit (vocab-split) | `vocab_slice × 4` bytes | Final lm_head step only |
| Barrier/sync packet | 8 bytes | Every layer boundary (once per tile pair) |
| Reduction acknowledgment | 8 bytes | After FABRIC_REDUCE completes |

**Activation broadcast:** For layer-wise placement (all layers on one tile per
shard), the host dispatches one activation per layer step via `FABRIC_SEND`.
For head-wise placement, partial attention outputs are summed via
`FABRIC_REDUCE` before crossing the layer boundary.

**No PCIe in the hot path:** During steady-state decode, the host writes only
the next-token ID (4 bytes) and reads only the output logit vector
(`vocab_size × 4` bytes).  All intermediate activations, KV updates, and
inter-tile traffic remain off the PCIe bus.

### 5.4 Tensor Tile Residency

Once loaded via `LOAD_TENSOR_TILE`, weight tensors reside permanently in AIMU
local memory for the duration of the inference session.  The host must not
overwrite a resident tensor while any EXEC command referencing that `tensor_id`
is in-flight.

Eviction is explicit: the host issues `RESET_TILE` with `keep_sessions=1` to
evict all tensors without disrupting the KV cache.

### 5.5 KV Memory Residency

KV cache entries are written by `KV_APPEND` and read by `EXEC_ATTENTION` or
`KV_READ`.  KV data never leaves the AIMU during inference (zero PCIe traffic
for KV updates).  KV state is session-scoped and survives `RESET_TILE`
with `keep_sessions=1`.

### 5.6 Zero-Copy / Near-Memory Assumptions

The AIMU execution model assumes:
- All EXEC commands read weight tensors from AIMU-local tensor memory.
- All EXEC commands read/write activations from AIMU-local staging buffers.
- PCIe DMA is used only for: initial tensor load, logit delivery, and counter
  reads.
- Activation vectors do not round-trip through PCIe between layers.

This is the hardware realization of the ATT-1 simulator's `near-memory
execution` principle (§1 of `docs/aimu_architecture.md`).

---

## 6. Memory Model

### 6.1 Local Tensor Memory

Purpose: stores all weight tensor slices assigned to this tile.

- Allocated statically at boot or at `LOAD_TENSOR_TILE` time (card-specific).
- Capacity: as reported by `tensor_memory_bytes` in §2.3.
- Alignment: 256 bytes (to accommodate q4 group_size=32 alignment rules).
- Reference: M100 placement report `model_bytes` field for per-tile estimates;
  M102 scenario tool for SKU comparison.

### 6.2 Local KV/Session Memory

Purpose: stores K/V cache entries for all active sessions.

- Capacity: as reported by `kv_memory_bytes`.
- Per-session isolation: each session occupies a disjoint KV memory region.
- KV entry size: `n_kv_heads × head_dim × 2 × sizeof(f32)` per token.
- Maximum context enforced: `target_context_length` from the placement report
  header.
- Overflow returns `ERR_KV_OVERFLOW`.

### 6.3 Staging Buffers

Purpose: host-writable DMA landing zone for incoming tensors and activation
inputs.

- Capacity: as reported by `staging_bytes`.
- Alignment: 64 bytes.
- The host may not read from or write to staging buffers while a command
  referencing them is in-flight.

### 6.4 Command Queue Memory

Purpose: host-writable command ring buffer consumed in-order by the AIMU.

- Format: circular buffer of 64-byte command packets (§3).
- Capacity: `command_queue_bytes` / 64 = `command_queue_depth` slots.
- Head/tail pointers maintained by the AIMU; the host advances the write
  pointer by writing new command packets.
- The AIMU signals command completion by writing `status` bytes 60–63 of each
  packet and, optionally, raising a PCIe MSI-X interrupt.

### 6.5 Trace/Counter Memory

Purpose: holds trace records written by `TRACE_SNAPSHOT` and the live counter
registers read by `QUERY_COUNTERS`.

- Capacity: as reported by `trace_memory_bytes`.
- Alignment: 8 bytes.
- Each `TRACE_SNAPSHOT` record is 64 bytes (§8.1).
- The trace region is a circular buffer; the AIMU maintains a write pointer.
  The host reads trace records in order using the last-known write pointer.

### 6.6 Memory Capacity Reporting

The host must query `tensor_memory_bytes` and `kv_memory_bytes` from §2.3
before issuing any load commands.  The M102 scenario tool
(`propose_tensor_scenarios.py`) provides an offline capacity planning
complement that predicts whether a given model/context/session configuration
will fit in a given SKU tier.

### 6.7 Alignment Requirements Summary

| Region | Alignment |
|---|---|
| Tensor memory | 256 bytes |
| Q4 tensor payload | 256 bytes (group_size=32 × max dtype width) |
| Q8 tensor payload | 64 bytes (row stride must be multiple of 64 bytes) |
| F32 tensor payload | 4 bytes (natural; host enforces 64-byte for DMA) |
| KV memory per entry | 64 bytes |
| Staging buffer | 64 bytes |
| Command queue | 64 bytes (one packet = 64 bytes) |
| Trace record | 64 bytes |

### 6.8 Checksum / Validation Hash

Each tensor has an expected CRC32 hash embedded in the M100 placement report
JSON (`"tensor_hash"` field, if present).  The `VALIDATE_TENSOR_TILE` command
verifies this hash after loading.  The checksum covers the raw tensor payload
bytes (post-quantization, as written by the ATT-1 converter).

---

## 7. Execution Model

### 7.1 Command Queue Lifecycle

```
Host writes command packet → advances write pointer
       ↓
AIMU reads packet (verifies checksum)
       ↓
AIMU checks fence dependency (§7.2)
       ↓
AIMU dispatches to op pipeline (EXEC_*, KV_*, FABRIC_*)
       ↓
AIMU writes status=result code to packet bytes 60–63
AIMU writes completion fence if completion_fence_id ≠ 0
AIMU signals MSI-X if CAP_ASYNC_COMPLETION
       ↓
Host reads status / handles interrupt
```

### 7.2 Fences and Barriers

Fences provide ordering guarantees between commands on the same tile or across
tiles:

- **Intra-tile ordering**: commands execute in submission order within one
  tile's queue.  No fence needed for sequential commands on the same tile.
- **Cross-tile dependency**: the host assigns a `completion_fence_id` to the
  producing command and sets the corresponding `fence_id` on the consumer
  command (which may be on a different tile).  The consumer does not execute
  until the fence is signaled.
- **`TILE_BARRIER`**: used for global synchronization at layer boundaries in
  multi-tile decode.  All tiles in the bitmask must reach the same barrier
  phase before any tile proceeds.
- **Fence ID 0**: reserved to mean "no dependency."

Fence IDs are 16-bit and tile-local.  The host allocates them
monotonically per session.

### 7.3 Dependency Ordering

For a standard decode step with two tiles (layer-wise placement):

```
Tile 0:  EXEC_RMSNORM → EXEC_ATTENTION → EXEC_FFN → TILE_BARRIER(phase=N)
Tile 1:  EXEC_RMSNORM → EXEC_ATTENTION → EXEC_FFN → TILE_BARRIER(phase=N)
                                                          ↓
                               Host collects logits after TILE_BARRIER completes
```

For cross-tile activation passing:
```
Tile 0: EXEC_MATMUL → FABRIC_SEND(to=tile_1)
Tile 1: (fence on tile_0 FABRIC_SEND) → EXEC_RMSNORM(uses activation)
```

### 7.4 Completion Records

When `CAP_ASYNC_COMPLETION` is set, the AIMU writes a 16-byte completion
record to a host-mapped completion ring buffer:

| Offset | Size | Field |
|---|---|---|
| 0 | 4 B | `command_id` |
| 4 | 1 B | `tile_id` |
| 5 | 1 B | `session_id` |
| 6 | 2 B | `latency_us` (command latency in microseconds, saturating) |
| 8 | 4 B | `result_code` (see §2.8) |
| 12 | 4 B | `trace_write_ptr` (current trace write pointer after this command) |

### 7.5 Deterministic Trace Points

All trace-flagged commands produce identical counter snapshots for a given
input token sequence and placement plan.  Requirements:
- Partial results in `FABRIC_REDUCE` are accumulated in ascending tile-ID
  order (not arrival order) to guarantee floating-point determinism.
- `TILE_BARRIER` phase tags ensure all tiles observe the same synchronization
  point before counters are snapshotted.
- KV position is determined by `kv_position` in the command packet, not by
  a tile-local counter, to prevent drift.

### 7.6 Error Propagation

- A command that returns any non-zero `result_code` does **not** execute
  subsequent fence-dependent commands on that tile.  Dependents are cancelled
  and return `ERR_FENCE_DEADLOCK`.
- The host must issue `RESET_TILE` after any `ERR_FATAL` before the tile
  accepts new commands.
- Cancelled commands return `ERR_FENCE_DEADLOCK` in their `status` field.
- There is no implicit error recovery or fallback execution path.  All error
  handling is explicit and host-driven.

### 7.7 No Silent Fallback Rule

Consistent with the ATT-1 simulator no-silent-fallback policy:
- If a command type is not supported (`ERR_UNSUPPORTED_CMD`), the AIMU returns
  the error; it does not emulate the operation in a slower mode.
- If a dtype is not supported (`ERR_UNSUPPORTED_DTYPE`), the AIMU returns
  the error; it does not upcast or downcast the operands.
- If the AIMU is in error state, it returns `ERR_FATAL` for all new commands
  until `RESET_TILE` is issued.

---

## 8. Counters and Trace Requirements

### 8.1 Counter Snapshot Format (64 bytes)

The `TRACE_SNAPSHOT` and `QUERY_COUNTERS` commands produce a 64-byte counter
record in the following format.  This mirrors the ATT-1 `att1_trace_t` fields:

| Offset | Size | Field | ATT-1 equivalent |
|---|---|---|---|
| 0 | 8 B | `packets_sent` | `att1_trace_t.fabric_packets_sent` |
| 8 | 8 B | `packets_received` | `att1_trace_t.fabric_packets_received` |
| 16 | 8 B | `payload_bytes_sent` | (new for hardware) |
| 24 | 8 B | `payload_bytes_received` | (new for hardware) |
| 32 | 4 B | `local_op_count` | total EXEC_* commands completed |
| 36 | 4 B | `tensor_bytes_read` | bytes fetched from local tensor memory |
| 40 | 4 B | `tensor_bytes_written` | bytes written to local tensor memory |
| 44 | 4 B | `kv_appends` | `att1_trace_t.kv_appends` |
| 48 | 4 B | `kv_reads` | `att1_trace_t.kv_reads` |
| 52 | 2 B | `queue_depth_max` | peak command queue depth this session |
| 54 | 2 B | `stall_cycles_pct` | stall percentage (0–1000, units 0.1 %) |
| 56 | 4 B | `command_latency_us_max` | maximum single-command latency |
| 60 | 4 B | `snapshot_sequence` | monotonic snapshot counter |

Total: 64 bytes.

### 8.2 Packets Sent / Received

Incremented by `FABRIC_SEND` (sent) and by the fabric receive pipeline
(received).  Matches the ATT-1 simulator `fabric_packets_sent` /
`fabric_packets_received` fields used in bench output:
`fabric_packets_sent=N`.

### 8.3 Payload Bytes

Total bytes of activation, partial-result, and barrier payloads transferred
across the fabric.  Used by the M100 placement report bandwidth estimation
pipeline.

### 8.4 Local Op Count

Incremented once per completed `EXEC_MATMUL`, `EXEC_RMSNORM`, `EXEC_ROPE`,
`EXEC_ATTENTION`, or `EXEC_FFN` command.  Used to validate that every
expected operation was dispatched.

### 8.5 Tensor Bytes Read / Written

Bytes fetched from local tensor memory for each EXEC command.  For a q4
matmul with `d_model=4096, d_out=4096`, tensor bytes read = `4096 × 4096 / 2`
(packed nibbles) + scale bytes.

### 8.6 KV Appends / Reads

Mirrors `att1_trace_t.kv_appends` and `kv_reads`.  Validates that the KV
cache traffic per token matches the placement report's `kv_bytes` estimate.

### 8.7 Command Latency

`command_latency_us_max` records the peak command execution latency observed
since the last `RESET_TILE` or `QUERY_COUNTERS` with reset flag.  Used to
identify slow commands (e.g., tensor loads with cache-cold penalty).

### 8.8 Stall Reason Counters (CAP_COUNTER_STALL)

When `CAP_COUNTER_STALL` is set, the tile tracks stall cycles by category.
These are returned as an extended counter block (additional 64 bytes after
the base §8.1 record):

| Offset | Size | Field |
|---|---|---|
| 0 | 4 B | `stall_fence_cycles` | cycles waiting for a cross-tile fence |
| 4 | 4 B | `stall_dma_cycles` | cycles waiting for PCIe DMA completion |
| 8 | 4 B | `stall_fabric_cycles` | cycles waiting for fabric send/receive |
| 12 | 4 B | `stall_barrier_cycles` | cycles waiting in `TILE_BARRIER` |
| 16 | 4 B | `stall_queue_full_cycles` | cycles with command queue full |
| 20 | 4 B | `fabric_congestion_count` | fabric congestion events detected |
| 24 | 8 B | reserved | Zero |
| 32 | 32 B | `stall_op_cycles[8]` | per-op stall cycles; indexed by op type (0=matmul, 1=rmsnorm, ...) |

### 8.9 Per-Token Execution Trace Hook (CAP_TRACE_PER_TOKEN)

When `CAP_TRACE_PER_TOKEN` is set, the AIMU automatically snapshots the
counter record at the completion of each `EXEC_ATTENTION` command (one per
decode step per tile).  These records mirror the ATT-1 simulator's
prefill/decode split fields:

- `prefill_fabric_packets` — fabric packets during the prefill phase
- `decode_fabric_packets` — fabric packets during the decode phase
- `prefill_kv_appends` / `decode_kv_appends`
- `prefill_time_us_total` / `decode_time_us_total`

This enables the host to compute per-token latency breakdowns without polling.

---

## 9. Relationship to Tensor Placement Reports (M98–M102)

### 9.1 Tile Memory Capacity Planning

The M96 tile capacity estimator and M100 placement report `model_bytes` field
tell the host how much local tensor memory a given model/tile-count/dtype
combination requires.  The host must verify that `tensor_memory_bytes` (§2.3)
is sufficient before issuing `LOAD_TENSOR_TILE` commands:

```
required = placement_report.tiles[tile_id].model_bytes
            + placement_report.tiles[tile_id].kv_bytes
available = tile_descriptor.local_memory_mib × 1024 × 1024
assert available >= required
```

If `available < required`, the host must either choose a larger SKU or use the
M102 scenario tool to find a viable tile-count / context combination.

### 9.2 Tensor Ownership and Routing

Each tensor in the M100 placement report `tensors` array has an `owner_tile`
field.  The host uses this to route `LOAD_TENSOR_TILE` commands:

```
for tensor in placement_report.tensors:
    send LOAD_TENSOR_TILE(
        tile_id   = tensor.owner_tile,
        tensor_id = tensor.tensor_category_id,
        dtype     = placement_report.header.dtype,
        payload   = tensor_data[tensor.tensor_name]
    )
```

### 9.3 Tensor Slices

For tensors with `placement_status = "sliced"` (head-wise or vocab-split),
the slice fields `slice_axis`, `slice_start`, `slice_end` define the byte
range that must be loaded:

- `input_buf_addr` in `LOAD_TENSOR_TILE` points to the start of the slice
  (not the full tensor).
- `input_buf_bytes = (slice_end - slice_start) × element_size`.
- `op_param_1` in `EXEC_MATMUL` carries the slice dimension hint.

### 9.4 Routing Requirements

The M98 `routing_requirement` field in each tensor record maps to fabric
command types:

| `routing_requirement` | Required command(s) |
|---|---|
| `"local"` | No fabric commands needed; all EXEC on same tile |
| `"broadcast"` | `FABRIC_SEND` with all-tile destination bitmask |
| `"reduce"` | `FABRIC_REDUCE` after partial EXEC_MATMUL |
| `"unicast"` | `FABRIC_SEND` to single target tile |
| `"none"` | Tensor not consumed by fabric (e.g., replicated norms) |

### 9.5 Reduction Behavior

The M98 `reduction_behavior` field maps to `FABRIC_REDUCE` parameters:

| `reduction_behavior` | `reduce_type` in FABRIC_REDUCE |
|---|---|
| `"sum"` | `0` (element-wise sum of partial logits/activations) |
| `"concat"` | `1` (concatenate partial head outputs) |
| `"none"` | No `FABRIC_REDUCE` issued |

### 9.6 Bandwidth Estimates

The M100 placement report `bandwidth_status` field and M96 `fabric_gib_sec`
estimate indicate whether the target fabric bandwidth is sufficient.  At
hardware bringup, the host should:

1. Run a `TILE_BARRIER` + `FABRIC_SEND` latency benchmark to measure actual
   fabric bandwidth.
2. Compare against the M100 `bandwidth_status` estimate.
3. If actual BW < estimate, reduce tile count or context length using the
   M102 scenario tool and re-plan.

### 9.7 SKU Comparison (16 / 32 / 64 / 128 GiB Tiles)

The M102 scenario tool output maps directly to AIMU SKU selection:

| M102 `tile_memory_gib` | Prototype target | Notes |
|---|---|---|
| 16 GiB | Entry-level FPGA + external DRAM | Fits small models (q4); gpt-oss-120b FAIL in 16 GiB |
| 32 GiB | Balanced FPGA/PCIe card | Recommended for mid-range production models |
| 64 GiB | Production PCIe card | Large models; long context; q8 comfortable |
| 128 GiB | Future ASIC target | Full 120B+ model in q4; maximum sessions |

---

## 10. Minimal Viable Prototype Options

### Option A: Software PCIe Endpoint Simulator

**Description:** Implement a software AIMU endpoint in a host process that
exposes the same 64-byte command packet interface over shared memory or
`/dev/shm`.  No FPGA or PCIe hardware required.

| Criterion | Assessment |
|---|---|
| Development speed | Fastest; runnable in M105 |
| Hardware fidelity | Low (no real DMA, no real fabric latency) |
| Command format validation | Full — validates packet layout and error handling |
| Cost | ~0 |
| Next step | M105 PCIe command queue simulator |

### Option B: FPGA PCIe Card

**Description:** Implement the AIMU command engine on an FPGA (e.g.,
Xilinx Alveo, Intel Agilex) with an HBM or DDR5 memory controller for
local tensor storage.

| Criterion | Assessment |
|---|---|
| Development speed | 6–12 months to first token |
| Hardware fidelity | High (real PCIe DMA, real fabric latency) |
| Command format validation | Full |
| Memory capacity | 8–32 GiB HBM2e (Alveo U280) |
| Cost | ~$5–15 K per card |

### Option C: PCIe Card with AIMU Controller + External Memory

**Description:** Discrete PCIe card with a custom AIMU controller ASIC
(16 nm or 7 nm), 64–128 GiB HBM3 stack, and a custom NoC fabric.

| Criterion | Assessment |
|---|---|
| Development speed | 18–36 months to tape-out |
| Hardware fidelity | Production-class |
| Command format validation | Full |
| Memory capacity | 64–128 GiB per card |
| Cost | $5–20 M NRE for tape-out |

### Option D: Future ASIC

**Description:** Monolithic AIMU die with on-chip HBM3 stack, integrated
fabric mesh, and GDDR7 or HBM4 in future process nodes.

| Criterion | Assessment |
|---|---|
| Development speed | 36–60 months |
| Hardware fidelity | Maximum |
| Memory capacity | 128+ GiB per die |
| Cost | $20–100 M NRE |

**Recommended prototype path:** Option A (software simulator, M105) first for
command protocol validation, then Option B (FPGA) for hardware bringup and
latency measurement.  Options C and D follow after the FPGA prototype validates
the command model.

---

## 11. Non-Goals

- **No production ASIC design.** This document defines the command protocol,
  not transistor-level microarchitecture.
- **No full PCIe driver implementation.** Driver skeleton is deferred to M106.
- **No MMIO register map implementation.** Register offsets are illustrative;
  final BAR layout is deferred to M104.
- **No patent claim language.** This document contains only engineering
  specifications.
- **No public cloud deployment.** The AIMU is designed as on-premises
  near-memory compute.
- **No mobile, Android, Vulkan, or OpenCL targets.**
- **No change to the ATT-1 C11 runtime, `.att1` binary format, or any existing
  inference behavior.**
- **No DMA descriptor implementation yet.** DMA descriptor format is deferred
  to M107.
- **No physical layer specification.** Electrical/optical interconnect and
  PCIe signal integrity are out of scope.

---

## 12. Future Milestone Split

| Milestone | Title | Scope |
|---|---|---|
| M104 | AIMU register map sketch | Define BAR0 register offsets for device_info, tile_descriptor, memory map, command queue head/tail pointers, completion ring, and MSI-X configuration; documentation only |
| M105 | PCIe command queue simulator | Python or C shim that exposes the §3 command ring buffer over a POSIX shared-memory endpoint; validates command packet checksum, dispatch, and completion record; smoke test: submit LOAD_TENSOR_TILE + EXEC_MATMUL against a tiny model and compare result to att1-bench cpu-f32 |
| M106 | AIMU device discovery simulator | Extend M105 simulator with the §2 discovery interface (device_info, tile_descriptor array, memory_map registers); host-side discovery library stub |
| M107 | DMA descriptor simulator | Define the DMA descriptor format for host-to-AIMU and AIMU-to-host transfers; integrate with M105/M106 simulator; validate tensor load round-trip |
| M108 | Command trace/counter integration | Wire the §8 counter snapshot format into the M105/M106 simulator; export trace records in att1_trace_t-compatible JSON for diff against att1-bench output |
| M109 | Placement-report-to-command-plan mapper | Python tool that reads an M100 placement report and produces an ordered list of LOAD_TENSOR_TILE and EXEC_* commands (one per layer, per tile); validates that the command plan matches the placement report tensor/tile assignments |
| M110 | Minimal PCIe/AIMU prototype design review | Engineering review milestone: reconcile M104–M109 artifacts, resolve open questions from §2 and §6, produce a single-page hardware bringup checklist |

---

## Appendix A: Open Engineering Questions

The following questions are deferred to M104–M110 or hardware bringup:

1. **BAR layout.** What is the exact byte offset of each register region in
   BAR0?  Should command queue and trace memory be in separate BARs?

2. **MSI-X vector allocation.** One MSI-X vector per tile?  Or one per
   completion event type (completion, error, trace-full)?

3. **DMA descriptor format.** Scatter-gather vs. contiguous?  Maximum
   descriptor chain length?

4. **Fabric topology.** Full mesh vs. ring vs. switch-based?  Maximum tile
   count before fabric becomes the bottleneck?

5. **Fence ID namespace.** Tile-local (reused per session) or globally unique
   per device?

6. **Staging buffer lifetime.** Does the AIMU zero-out the staging buffer
   after `LOAD_TENSOR_TILE`?  Or is it the host's responsibility?

7. **q4 group-size flexibility.** Is group_size=32 fixed, or should the
   command packet support group_size=64 and group_size=128 as future dtypes?

8. **Completion ring overflow.** What happens when the host does not drain
   the completion ring fast enough?  Backpressure?  MSI-X throttle?

9. **Multi-model support.** Can a single tile host tensors from two different
   `model_id` values simultaneously?  Or must `RESET_TILE` be issued between
   model switches?

10. **KV cache persistence across sessions.** Can a tile retain KV state for
    a paused session while serving another?  What is the maximum number of
    simultaneously resident KV sessions?
