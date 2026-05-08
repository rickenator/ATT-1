# AIMU/PCIe MMIO Register Map (Milestone 104)

This document defines the prototype MMIO register map for a PCIe-attached
AIMU tile card.  It provides the register-level interface that a host driver
would use to discover, configure, command, monitor, and reset an AIMU device.

This is a specification document only.  No C runtime, PCIe driver, MMIO
simulator, or binary format is implemented by this milestone.

Relationship to prior milestones:
- M93 §8 (`docs/aimu_architecture.md`): prototype requirements overview
- M103 (`docs/aimu_pcie_command_requirements.md`): command packet format,
  command types, data movement model, memory model, execution model, counters
- M104 (this document): MMIO register layout that hosts access to drive the
  M103 command protocol

---

## 1. Register Map Assumptions

### 1.1 Register Width and Encoding

- All registers are **32 bits wide**.  64-bit quantities are represented as
  two consecutive 32-bit registers: `_LOW` (bits 31:0) at the lower address
  and `_HIGH` (bits 31:0, carrying bits 63:32) at the higher address.
- Byte order is **little-endian** throughout.  The least-significant byte of
  each register occupies the lowest byte address in the BAR.
- All registers are **naturally 32-bit aligned** (offset must be a multiple of
  4).  A 64-bit pair is aligned to 8 bytes (offset must be a multiple of 8).
- Multi-register atomic operations: pairs described as `_LOW/HIGH` must be
  written atomically using 64-bit MMIO where the PCIe platform supports it.
  Where only 32-bit MMIO is available, the host must write `_HIGH` before
  `_LOW`, and read `_LOW` before `_HIGH`.

### 1.2 Access Types

Each register carries one of four access type annotations:

| Symbol | Meaning |
|---|---|
| `RO` | Read-only; writes are ignored |
| `RW` | Read-write; value persists until explicitly changed |
| `RW1C` | Read-write-one-to-clear; writing 1 clears the bit; writing 0 has no effect |
| `WO` | Write-only; reads return 0 |

### 1.3 Reserved Bits

- Reserved bits in **RO** registers always read as `0`.
- Reserved bits in **RW** registers must be written as `0`.  Future register
  map versions may assign them new semantics; software that preserves them as
  `0` will be forward-compatible.
- Reserved registers at unassigned offsets read as `0xDEAD_BEEF`; writes are
  discarded.

### 1.4 Versioning

- The register map version is distinct from the firmware version.  Firmware
  updates may increment the firmware version without changing the register map
  version.
- A register map version increment signals that at least one register offset,
  field width, or access type has changed in a non-backward-compatible way.
- The host must read `REGISTER_MAP_VERSION` (§2.3) before accessing any
  other register.  If the version exceeds the highest version the driver
  supports, the driver must fall back to the intersection of features declared
  in `FEATURE_FLAGS_LOW/HIGH` (§2.4), or refuse the device.

### 1.5 Per-Device vs Per-Tile Register Windows

The BAR0 address space is divided into two regions:

```
BAR0 layout
┌────────────────────────────────────────────────────┐
│ 0x0000 – 0x0FFF   Global device registers (§2)     │
│ 0x1000 – 0x1FFF   Command queue registers (§4)     │
│ 0x2000 – 0x2FFF   DMA/buffer descriptor registers (§5) │
│ 0x3000 – 0x3FFF   Fabric/interconnect registers (§6) │
│ 0x4000 – 0x4FFF   Counter registers (§7)           │
│ 0x5000 – 0x5FFF   Trace/debug registers (§8)       │
│ 0x6000 – 0x6FFF   Reserved                         │
│ 0x7000 – 0x7FFF   Reserved                         │
│ 0x8000 – 0xFFFF   Per-tile register windows        │
│                   Tile N window: 0x8000 + N×0x800  │
│                   (max 16 tiles; 0x800 bytes/tile)  │
└────────────────────────────────────────────────────┘
```

Per-tile register windows (§3) start at `0x8000`.  Tile 0 is at `0x8000`,
tile 1 at `0x8800`, tile 2 at `0x9000`, and so on.  The maximum addressable
tile count is 16 per BAR0 window, giving a maximum per-tile window start of
`0xF800`.  Platforms requiring more than 16 tiles per device use multiple PCIe
functions or a BAR extension mechanism (deferred to M110).

---

## 2. Global Device Registers

Base offset: `0x0000`

### 2.1 `DEVICE_ID` — RO, offset 0x0000

| Bits | Field | Description |
|---|---|---|
| 31:16 | `vendor_id` | ATT-1 vendor identifier: `0xA771` |
| 15:0 | `device_class` | `0x0001` = AIMU inference tile; `0x0002` = AIMU fabric switch |

### 2.2 `DEVICE_VERSION` — RO, offset 0x0004

| Bits | Field | Description |
|---|---|---|
| 31:24 | `major` | Major version |
| 23:16 | `minor` | Minor version |
| 15:8 | `patch` | Patch version |
| 7:0 | `build` | Build/stepping identifier |

### 2.3 `REGISTER_MAP_VERSION` — RO, offset 0x0008

| Bits | Field | Description |
|---|---|---|
| 31:16 | `reserved` | Must read 0 |
| 15:8 | `map_major` | Register map major version (currently `0x01`) |
| 7:0 | `map_minor` | Register map minor version (currently `0x00`) |

### 2.4 `FEATURE_FLAGS_LOW` — RO, offset 0x000C

| Bit | Flag | Meaning |
|---|---|---|
| 0 | `CAP_PLACEMENT_REPORT` | Device can consume M100 placement JSON metadata |
| 1 | `CAP_ASYNC_COMPLETION` | Command completions are asynchronous with IRQ/poll |
| 2 | `CAP_FENCE` | Cross-tile dependency fences supported |
| 3 | `CAP_PARTIAL_REDUCE` | Hardware partial-logit reduce across tiles |
| 4 | `CAP_Q4_ALIGN32` | Q4 group_size=32 alignment enforced in hardware |
| 5 | `CAP_TRACE_PER_TOKEN` | Per-token trace hooks in hardware |
| 6 | `CAP_COUNTER_STALL` | Stall-reason counters available |
| 7 | `CAP_DMA_SCATTER_GATHER` | DMA engine supports scatter-gather descriptors |
| 8 | `CAP_MSI_X` | Device supports MSI-X interrupt delivery |
| 9 | `CAP_FABRIC_MESH` | Full-mesh fabric topology available |
| 10 | `CAP_64BIT_MMIO` | Device supports 64-bit MMIO atomic reads/writes |
| 31:11 | reserved | Must read 0 |

### 2.5 `FEATURE_FLAGS_HIGH` — RO, offset 0x0010

Reserved for future capability bits.  Must read 0.

### 2.6 `TILE_COUNT` — RO, offset 0x0014

| Bits | Field | Description |
|---|---|---|
| 31:8 | reserved | Must read 0 |
| 7:0 | `tile_count` | Number of AIMU tiles on this device (1–16) |

### 2.7 `COMMAND_QUEUE_COUNT` — RO, offset 0x0018

| Bits | Field | Description |
|---|---|---|
| 31:8 | reserved | Must read 0 |
| 7:0 | `cq_count` | Number of command queues (one per tile; equals `tile_count`) |

### 2.8 `INTERRUPT_STATUS` — RW1C, offset 0x001C

Each bit corresponds to an interrupt source.  Writing `1` to a bit clears it
after the source condition is resolved.

| Bit | Source |
|---|---|
| 0 | Command completion (any tile) |
| 1 | Error (any tile) |
| 2 | Trace buffer half-full |
| 3 | Trace buffer full (dropped events) |
| 4 | Fabric congestion threshold exceeded |
| 5 | DMA transfer complete |
| 6 | Tile reset complete |
| 7 | Counter overflow (any counter wrapped) |
| 31:8 | reserved |

### 2.9 `INTERRUPT_ENABLE` — RW, offset 0x0020

Bitmask with the same bit assignments as `INTERRUPT_STATUS`.  A bit set to
`1` enables delivery of the corresponding interrupt via MSI-X vector 0.
Default: `0x00000000` (all interrupts disabled; software must enable
explicitly).

### 2.10 `GLOBAL_STATUS` — RO, offset 0x0024

| Bit | Name | Meaning |
|---|---|---|
| 0 | `DEVICE_READY` | Device has completed power-on reset and is ready |
| 1 | `ANY_TILE_ACTIVE` | At least one tile is executing commands |
| 2 | `ANY_TILE_ERROR` | At least one tile is in error state |
| 3 | `FABRIC_ACTIVE` | Fabric interconnect is carrying traffic |
| 4 | `DMA_ACTIVE` | DMA engine has an in-flight transfer |
| 5 | `TRACE_ACTIVE` | Trace capture is enabled on at least one tile |
| 31:6 | reserved | Must read 0 |

### 2.11 `GLOBAL_CONTROL` — RW, offset 0x0028

| Bit | Name | Effect when written `1` |
|---|---|---|
| 0 | `ENABLE_DEVICE` | Take device out of idle and allow command processing |
| 1 | `DISABLE_DEVICE` | Pause all command processing (in-flight commands complete) |
| 2 | `ENABLE_TRACE` | Enable trace capture on all tiles |
| 3 | `DISABLE_TRACE` | Disable trace capture on all tiles |
| 4 | `FLUSH_COMPLETIONS` | Force delivery of all pending completion records |
| 31:5 | reserved | Must be written 0 |

Bits in `GLOBAL_CONTROL` are self-clearing write-one triggers, not latching
state.  The current device state is reflected in `GLOBAL_STATUS`.

### 2.12 `RESET_CONTROL` — RW, offset 0x002C

| Bit | Name | Effect when written `1` |
|---|---|---|
| 0 | `SOFT_RESET_ALL` | Reset all tiles and all queues; re-run device init |
| 1 | `RESET_COUNTERS` | Zero all performance counters across all tiles |
| 2 | `RESET_TRACE` | Clear trace buffer and reset trace write pointer to 0 |
| 3 | `RESET_FABRIC` | Reset fabric link state and congestion counters |
| 31:4 | reserved | Must be written 0 |

`SOFT_RESET_ALL` takes the device through its power-on initialization
sequence.  The host must poll `GLOBAL_STATUS.DEVICE_READY` (§2.10) before
issuing any subsequent commands.

### 2.13 `ERROR_STATUS` — RW1C, offset 0x0030

| Bit | Name | Meaning |
|---|---|---|
| 0 | `PCIE_AER` | PCIe advanced error reported |
| 1 | `FABRIC_LINK_DOWN` | Fabric link failure detected |
| 2 | `DMA_FAULT` | DMA address fault (invalid host address) |
| 3 | `COMMAND_PARITY` | Command packet CRC32 mismatch |
| 4 | `TENSOR_CHECKSUM` | Tensor validation checksum mismatch |
| 5 | `TILE_FAULT` | Per-tile fault; see `TILE_ERROR_STATUS` (§3.13) |
| 6 | `WATCHDOG` | Device-level watchdog expired |
| 31:7 | reserved | Must read 0 |

### 2.14 `ERROR_DETAIL` — RO, offset 0x0034

| Bits | Field | Description |
|---|---|---|
| 31:24 | `error_tile_id` | Tile index that raised the most recent error |
| 23:16 | `error_cmd_type` | Command type byte from the offending command packet |
| 15:8 | `error_session_id` | Session ID from the offending command packet |
| 7:0 | `error_code` | Error code (see §9 for full table) |

`ERROR_DETAIL` is updated on every new error, including when
`ERROR_STATUS` bits are cleared.  It retains the last error detail until the
next error or `RESET_COUNTERS`.

### 2.15 `TRACE_CONTROL` — RW, offset 0x0038

| Bits | Field | Description |
|---|---|---|
| 31:24 | reserved | Must be written 0 |
| 23:16 | `trace_tile_mask` | Bitmask of tiles for which trace capture is enabled |
| 15:8 | reserved | Must be written 0 |
| 7:4 | `trace_level` | `0=off, 1=errors_only, 2=commands, 3=full` |
| 3:0 | `trace_trigger` | `0=always, 1=on_attention, 2=on_layer_boundary, 3=on_decode_step` |

### 2.16 `COUNTER_SNAPSHOT_CONTROL` — RW, offset 0x003C

| Bits | Field | Description |
|---|---|---|
| 31:24 | reserved | Must be written 0 |
| 23:16 | `snapshot_tile_mask` | Bitmask of tiles to include in snapshot |
| 15:1 | reserved | Must be written 0 |
| 0 | `SNAPSHOT_NOW` | Write `1` to trigger an immediate counter snapshot to trace memory; self-clearing |

---

## 3. Per-Tile Register Windows

Base offset per tile N: `0x8000 + N × 0x800`

The fields below are described relative to the per-tile window base.  All
field offsets are the same for every tile.

### 3.1 `TILE_ID` — RO, offset +0x000

| Bits | Field | Description |
|---|---|---|
| 31:8 | reserved | Must read 0 |
| 7:0 | `tile_id` | Physical tile index (0-based; matches the window index N) |

### 3.2 `TILE_STATUS` — RO, offset +0x004

| Bits | Field | Description |
|---|---|---|
| 31:8 | reserved | Must read 0 |
| 7:4 | `session_count` | Number of sessions currently resident on this tile |
| 3:2 | reserved | Must read 0 |
| 1:0 | `state` | `0=idle, 1=active, 2=error, 3=resetting` |

### 3.3 `TILE_FEATURE_FLAGS` — RO, offset +0x008

Per-tile capability flags.  Bit assignments match `FEATURE_FLAGS_LOW` (§2.4)
with the addition of:

| Bit | Flag | Meaning |
|---|---|---|
| 12 | `CAP_LOCAL_RMSNORM` | Tile has dedicated RMSNorm hardware unit |
| 13 | `CAP_LOCAL_ROPE` | Tile has dedicated RoPE rotation unit |
| 14 | `CAP_LOCAL_ATTN` | Tile has dedicated attention softmax + score unit |
| 15 | `CAP_LOCAL_SWIGLU` | Tile has dedicated SwiGLU gate activation unit |

Bits 0–11 mirror the device-level `FEATURE_FLAGS_LOW` bits (§2.4).  A tile
may have a subset of device features if the fabric attached to this tile is
simpler.

### 3.4 `TILE_MEMORY_CAPACITY_LOW` — RO, offset +0x00C
### 3.5 `TILE_MEMORY_CAPACITY_HIGH` — RO, offset +0x010

64-bit pair.  Total local tensor memory capacity in bytes.  Corresponds to
`tensor_memory_bytes` in the M103 memory map (§6.1 of
`docs/aimu_pcie_command_requirements.md`).

### 3.6 `TILE_MEMORY_USED_LOW` — RO, offset +0x014
### 3.7 `TILE_MEMORY_USED_HIGH` — RO, offset +0x018

64-bit pair.  Bytes currently occupied by resident tensors.  Updated after
each `LOAD_TENSOR_TILE` or `RESET_TILE` command completes.

### 3.8 `TILE_KV_CAPACITY_LOW` — RO, offset +0x01C
### 3.9 `TILE_KV_CAPACITY_HIGH` — RO, offset +0x020

64-bit pair.  Total KV cache memory capacity in bytes.

### 3.10 `TILE_KV_USED_LOW` — RO, offset +0x024
### 3.11 `TILE_KV_USED_HIGH` — RO, offset +0x028

64-bit pair.  Bytes currently occupied by KV cache entries across all sessions.

### 3.12 `SUPPORTED_DTYPES` — RO, offset +0x02C

| Bit | Name | Meaning |
|---|---|---|
| 0 | `DTYPE_F32` | f32 inference supported |
| 1 | `DTYPE_Q8` | per-row int8 (q8) quantized inference supported |
| 2 | `DTYPE_Q4` | grouped int4 (q4, group_size=32) supported |
| 31:3 | reserved | Must read 0 |

### 3.13 `SUPPORTED_OPS_LOW` — RO, offset +0x030

| Bit | Op name |
|---|---|
| 0 | `OP_MATMUL` |
| 1 | `OP_RMSNORM` |
| 2 | `OP_ROPE` |
| 3 | `OP_ATTENTION` |
| 4 | `OP_FFN_SWIGLU` |
| 5 | `OP_KV_APPEND` |
| 6 | `OP_KV_READ` |
| 7 | `OP_FABRIC_SEND` |
| 8 | `OP_FABRIC_REDUCE` |
| 9 | `OP_TRACE_SNAPSHOT` |
| 10 | `OP_TILE_BARRIER` |
| 11 | `OP_QUERY_COUNTERS` |
| 31:12 | reserved |

### 3.14 `SUPPORTED_OPS_HIGH` — RO, offset +0x034

Reserved for future op flags.  Must read 0.

### 3.15 `TILE_FABRIC_LINK_MASK` — RO, offset +0x038

| Bits | Field | Description |
|---|---|---|
| 31:16 | reserved | Must read 0 |
| 15:0 | `link_mask` | Bitmask of peer tiles this tile has an active fabric link to |

Bit N is set when there is an active fabric link between this tile and tile N.
A tile does not set its own bit.

### 3.16 `TILE_ERROR_STATUS` — RW1C, offset +0x03C

| Bits | Field | Description |
|---|---|---|
| 31:16 | `last_error_command_id` | `command_id` of the packet that triggered the error |
| 15:8 | `last_error_cmd_type` | Command type byte of the offending packet |
| 7:0 | `last_error_code` | Error code (see §9) |

Writing `1` to bit 0 clears the `last_error_code` field and acknowledges
the error (the tile may remain in error state until `TILE_RESET_CONTROL` is
used if the error was fatal).

### 3.17 `TILE_RESET_CONTROL` — WO, offset +0x040

| Bit | Name | Effect when written `1` |
|---|---|---|
| 0 | `RESET_TILE_SOFT` | Reset this tile's session state and KV cache; retain tensors |
| 1 | `RESET_TILE_FULL` | Full tile reset: clear tensors, sessions, KV, and counters |
| 2 | `RESET_TILE_COUNTERS` | Zero only performance counters for this tile |
| 3 | `RESET_TILE_TRACE` | Clear trace records for this tile |
| 31:4 | reserved | Must be written 0 |

After writing `RESET_TILE_FULL`, the host must poll `TILE_STATUS.state == idle`
before issuing new commands to this tile.

---

## 4. Command Queue Registers

Base offset: `0x1000`

The command queue register block controls the 64-byte command ring buffer
defined in M103 §3.  There is one logical command queue per tile.  For
multi-tile devices, the queue for tile N is addressed using the tile ID field
in the command packet itself; the physical ring buffer is shared or partitioned
as described by `CQ_BASE_ADDR_LOW/HIGH`.

### 4.1 `CQ_BASE_ADDR_LOW` — RW, offset 0x1000
### 4.2 `CQ_BASE_ADDR_HIGH` — RW, offset 0x1004

64-bit pair.  Host-physical base address of the command ring buffer.  Must
be 64-byte aligned.  Written by the host during device initialization.  The
ring buffer must be DMA-accessible from the AIMU device.

### 4.3 `CQ_SIZE` — RW, offset 0x1008

| Bits | Field | Description |
|---|---|---|
| 31:16 | reserved | Must be written 0 |
| 15:0 | `queue_slots` | Number of 64-byte command slots in the ring (power of 2; max 4096) |

### 4.4 `CQ_HEAD` — RO, offset 0x100C

| Bits | Field | Description |
|---|---|---|
| 31:16 | reserved | Must read 0 |
| 15:0 | `head` | Index of the next slot the AIMU will consume (updated by AIMU) |

### 4.5 `CQ_TAIL` — RW, offset 0x1010

| Bits | Field | Description |
|---|---|---|
| 31:16 | reserved | Must be written 0 |
| 15:0 | `tail` | Index of the next slot the host will fill (written by host) |

The ring is empty when `CQ_HEAD == CQ_TAIL`.  The ring is full when
`(CQ_TAIL + 1) % CQ_SIZE == CQ_HEAD`.  The host must not overwrite a slot
that has not yet been consumed (head advanced past it).

### 4.6 `CQ_DOORBELL` — WO, offset 0x1014

| Bits | Field | Description |
|---|---|---|
| 31:16 | reserved | Must be written 0 |
| 15:0 | `new_tail` | Writing a value here notifies the AIMU that commands up to `new_tail` are ready |

Writing `CQ_DOORBELL` is equivalent to writing `CQ_TAIL` and then signaling
the AIMU to begin processing.  On platforms that use polling rather than
interrupt-driven dispatch, the AIMU reads `CQ_TAIL` directly; `CQ_DOORBELL`
still triggers a DMA coherence fence.

### 4.7 `CQ_STATUS` — RO, offset 0x1018

| Bits | Field | Description |
|---|---|---|
| 31:8 | reserved | Must read 0 |
| 7:4 | `queue_state` | `0=idle, 1=processing, 2=stalled_on_fence, 3=error` |
| 3:0 | `last_completed_slot` (low 4 bits) | Low bits of the last consumed slot index (for debug) |

### 4.8 `CQ_ERROR` — RW1C, offset 0x101C

| Bits | Field | Description |
|---|---|---|
| 31:16 | reserved | Must read 0 |
| 15:0 | `error_slot` | Slot index of the most recent command that returned an error |

Writing `1` to bit 0 acknowledges the error and clears `error_slot`.

### 4.9 `CQ_FENCE_VALUE` — RO, offset 0x1020

| Bits | Field | Description |
|---|---|---|
| 31:16 | reserved | Must read 0 |
| 15:0 | `fence_value` | Current completed fence ID; host compares to expected completion fence |

The AIMU increments `CQ_FENCE_VALUE` when it completes a command whose
`completion_fence_id ≠ 0`.  The host can poll this register instead of using
interrupts to determine when a fenced command has completed.

### 4.10 `CQ_COMPLETION_ADDR_LOW` — RW, offset 0x1024
### 4.11 `CQ_COMPLETION_ADDR_HIGH` — RW, offset 0x1028

64-bit pair.  Host-physical base address of the completion ring buffer.  Each
completion record is 16 bytes (see M103 §7.4).  Must be 16-byte aligned.

### 4.12 `CQ_COMPLETION_SIZE` — RW, offset 0x102C

| Bits | Field | Description |
|---|---|---|
| 31:16 | reserved | Must be written 0 |
| 15:0 | `completion_slots` | Number of 16-byte completion slots (power of 2; max 4096) |

---

## 5. DMA / Buffer Descriptor Registers

Base offset: `0x2000`

These registers define the fields of a DMA buffer descriptor.  At the M104
prototype stage, the host submits descriptors inline inside command packets
(M103 §3); the registers below describe the canonical descriptor layout for
future use with a dedicated scatter-gather DMA engine.

### 5.1 DMA Descriptor Layout (64 bytes)

Each DMA descriptor is 64 bytes, naturally aligned:

| Offset | Size | Field | Access | Description |
|---|---|---|---|---|
| 0 | 4 B | `host_addr_low` | RW | Host physical address bits 31:0 |
| 4 | 4 B | `host_addr_high` | RW | Host physical address bits 63:32 |
| 8 | 4 B | `device_addr_low` | RW | AIMU-local address bits 31:0 |
| 12 | 4 B | `device_addr_high` | RW | AIMU-local address bits 63:32 |
| 16 | 4 B | `byte_length` | RW | Transfer length in bytes (max 2^28) |
| 20 | 4 B | `stride_bytes` | RW | Row stride in bytes (0 = contiguous) |
| 24 | 2 B | `dim0` | RW | Logical dimension 0 (rows) |
| 26 | 2 B | `dim1` | RW | Logical dimension 1 (columns) |
| 28 | 1 B | `dtype` | RW | `0=f32, 1=q8, 2=q4` |
| 29 | 1 B | `quant_group_size` | RW | Quantization group size (q4: 32; q8: row-stride; f32: 0) |
| 30 | 2 B | `flags` | RW | See §5.2 |
| 32 | 4 B | `checksum` | RW | CRC32/ISO-HDLC of the payload bytes |
| 36 | 4 B | `tensor_id` | RW | Target tensor ID on the AIMU tile |
| 40 | 4 B | `command_id` | RW | Command ID that owns this descriptor |
| 44 | 4 B | `reserved_0` | RO | Must be 0 |
| 48 | 16 B | `reserved_1[4]` | RO | Must be 0 |

Total: 64 bytes.

### 5.2 DMA Descriptor Flags (`flags` field, §5.1 offset 30)

| Bit | Name | Meaning |
|---|---|---|
| 0 | `HOST_TO_DEVICE` | Transfer direction: host → AIMU |
| 1 | `DEVICE_TO_HOST` | Transfer direction: AIMU → host |
| 2 | `VALIDATE_CHECKSUM` | AIMU verifies `checksum` field on receipt |
| 3 | `GENERATE_CHECKSUM` | AIMU computes and writes checksum on device-to-host |
| 4 | `LAST_DESCRIPTOR` | Final descriptor in a chain; triggers completion interrupt |
| 5 | `SCATTER_GATHER` | Descriptor is part of a chained list |
| 15:6 | reserved | Must be written 0 |

### 5.3 `DMA_CONTROL` — RW, offset 0x2000

| Bit | Name | Effect |
|---|---|---|
| 0 | `DMA_ENABLE` | Enable the DMA engine |
| 1 | `DMA_PAUSE` | Pause after current descriptor completes |
| 2 | `DMA_RESET` | Reset DMA engine state; self-clearing |
| 31:3 | reserved | Must be written 0 |

### 5.4 `DMA_STATUS` — RO, offset 0x2004

| Bits | Field | Description |
|---|---|---|
| 31:16 | reserved | Must read 0 |
| 15:8 | `active_transfers` | Number of in-flight DMA descriptors |
| 7:4 | `dma_state` | `0=idle, 1=active, 2=paused, 3=error` |
| 3:0 | reserved | Must read 0 |

### 5.5 `DMA_ERROR_STATUS` — RW1C, offset 0x2008

| Bit | Name | Meaning |
|---|---|---|
| 0 | `ADDR_FAULT` | Invalid host physical address |
| 1 | `ALIGNMENT_FAULT` | Source or destination not properly aligned |
| 2 | `CHECKSUM_FAIL` | Checksum verification failed on received payload |
| 3 | `OVERFLOW` | Transfer exceeded `byte_length` boundary |
| 31:4 | reserved | Must read 0 |

### 5.6 `DMA_DESCRIPTOR_RING_BASE_LOW` — RW, offset 0x200C
### 5.7 `DMA_DESCRIPTOR_RING_BASE_HIGH` — RW, offset 0x2010

64-bit pair.  Host-physical address of the scatter-gather descriptor ring.
64-byte aligned.

### 5.8 `DMA_DESCRIPTOR_RING_SIZE` — RW, offset 0x2014

| Bits | Field | Description |
|---|---|---|
| 31:16 | reserved | Must be written 0 |
| 15:0 | `ring_slots` | Number of 64-byte descriptor slots (power of 2) |

---

## 6. Fabric / Interconnect Registers

Base offset: `0x3000`

### 6.1 `FABRIC_STATUS` — RO, offset 0x3000

| Bits | Field | Description |
|---|---|---|
| 31:16 | `active_tile_mask` | Bitmask of tiles currently participating in fabric traffic |
| 15:8 | reserved | Must read 0 |
| 7:4 | `fabric_state` | `0=idle, 1=active, 2=congested, 3=error` |
| 3:0 | `link_count` | Number of active fabric links |

### 6.2 `FABRIC_CONTROL` — RW, offset 0x3004

| Bit | Name | Effect |
|---|---|---|
| 0 | `FABRIC_ENABLE` | Enable fabric traffic forwarding |
| 1 | `FABRIC_PAUSE` | Pause new fabric transmissions; in-flight complete |
| 2 | `FABRIC_RESET` | Reset fabric link state; self-clearing |
| 3 | `CONGESTION_BACKPRESSURE` | Enable hardware backpressure on congestion |
| 31:4 | reserved | Must be written 0 |

### 6.3 `FABRIC_ROUTE_TABLE_BASE_LOW` — RW, offset 0x3008
### 6.4 `FABRIC_ROUTE_TABLE_BASE_HIGH` — RW, offset 0x300C

64-bit pair.  AIMU-local address of the routing table used by `FABRIC_SEND`
commands.  Each routing table entry is 8 bytes: `{destination_tile_id: u8,
next_hop_port: u8, reserved: u48}`.  Routing table layout is deferred to M110.

### 6.5 `FABRIC_ROUTE_TABLE_SIZE` — RW, offset 0x3010

| Bits | Field | Description |
|---|---|---|
| 31:8 | reserved | Must be written 0 |
| 7:0 | `table_entries` | Number of routing table entries (one per tile; max 16) |

### 6.6 `FABRIC_PACKET_COUNTER_LOW` — RO, offset 0x3014
### 6.7 `FABRIC_PACKET_COUNTER_HIGH` — RO, offset 0x3018

64-bit packet count.  Total fabric packets transmitted since the last
`RESET_COUNTERS`.  Corresponds to `packets_sent` in the M103 counter snapshot
(§8.2 of `docs/aimu_pcie_command_requirements.md`).

### 6.8 `FABRIC_PAYLOAD_BYTES_LOW` — RO, offset 0x301C
### 6.9 `FABRIC_PAYLOAD_BYTES_HIGH` — RO, offset 0x3020

64-bit byte count.  Total fabric payload bytes transmitted since the last
`RESET_COUNTERS`.  Corresponds to `payload_bytes_sent` in M103 §8.3.

### 6.10 `FABRIC_CONGESTION_COUNTER` — RO, offset 0x3024

| Bits | Field | Description |
|---|---|---|
| 31:0 | `congestion_events` | Number of fabric congestion events detected since `RESET_COUNTERS` |

A congestion event is recorded when a `FABRIC_SEND` or `FABRIC_REDUCE`
command must stall for more than one fabric round-trip latency due to queue
pressure.

### 6.11 `FABRIC_ERROR_STATUS` — RW1C, offset 0x3028

| Bit | Name | Meaning |
|---|---|---|
| 0 | `LINK_DOWN` | A fabric link has gone down |
| 1 | `ROUTE_MISS` | A packet arrived with no matching routing table entry |
| 2 | `PACKET_CORRUPT` | Fabric packet CRC/framing error |
| 3 | `REDUCE_TIMEOUT` | `FABRIC_REDUCE` did not complete within `timeout_ms` |
| 4 | `BARRIER_TIMEOUT` | `TILE_BARRIER` did not complete within `timeout_ms` |
| 31:5 | reserved | Must read 0 |

---

## 7. Counter Registers

Base offset: `0x4000`

All counters are 64-bit (paired `_LOW/HIGH` registers) and are cleared by
`RESET_COUNTERS` (§2.12) or the per-tile `RESET_TILE_COUNTERS` (§3.17).
Counters saturate at `0xFFFF_FFFF_FFFF_FFFF` rather than wrapping.

### 7.1 `CNT_COMMANDS_ISSUED_LOW/HIGH` — RO, offset 0x4000/0x4004

Total command packets written to the command ring by the host.

### 7.2 `CNT_COMMANDS_COMPLETED_LOW/HIGH` — RO, offset 0x4008/0x400C

Total command packets for which the AIMU has written a `status ≠ pending`
result.

### 7.3 `CNT_LOCAL_OPS_LOW/HIGH` — RO, offset 0x4010/0x4014

Total successful `EXEC_*` commands completed (sum of matmul + rmsnorm + rope +
attention + ffn).

### 7.4 `CNT_MATMUL_LOW/HIGH` — RO, offset 0x4018/0x401C

Count of completed `EXEC_MATMUL` (0x10) commands.

### 7.5 `CNT_RMSNORM_LOW/HIGH` — RO, offset 0x4020/0x4024

Count of completed `EXEC_RMSNORM` (0x11) commands.

### 7.6 `CNT_ROPE_LOW/HIGH` — RO, offset 0x4028/0x402C

Count of completed `EXEC_ROPE` (0x12) commands.

### 7.7 `CNT_ATTENTION_LOW/HIGH` — RO, offset 0x4030/0x4034

Count of completed `EXEC_ATTENTION` (0x13) commands.

### 7.8 `CNT_FFN_LOW/HIGH` — RO, offset 0x4038/0x403C

Count of completed `EXEC_FFN` (0x14) commands.

### 7.9 `CNT_KV_APPENDS_LOW/HIGH` — RO, offset 0x4040/0x4044

Count of completed `KV_APPEND` (0x20) commands.  Corresponds to
`att1_trace_t.kv_appends`.

### 7.10 `CNT_KV_READS_LOW/HIGH` — RO, offset 0x4048/0x404C

Count of completed `KV_READ` (0x21) commands.  Corresponds to
`att1_trace_t.kv_reads`.

### 7.11 `CNT_TENSOR_BYTES_READ_LOW/HIGH` — RO, offset 0x4050/0x4054

Bytes fetched from local tensor memory for EXEC operations.

### 7.12 `CNT_TENSOR_BYTES_WRITTEN_LOW/HIGH` — RO, offset 0x4058/0x405C

Bytes written to local tensor memory (KV append, LOAD_TENSOR_TILE).

### 7.13 `CNT_ACTIVATION_BYTES_SENT_LOW/HIGH` — RO, offset 0x4060/0x4064

Bytes of activation vectors sent via `FABRIC_SEND`.

### 7.14 `CNT_ACTIVATION_BYTES_RECEIVED_LOW/HIGH` — RO, offset 0x4068/0x406C

Bytes of activation vectors received from the fabric.

### 7.15 `CNT_LOGITS_BYTES_LOW/HIGH` — RO, offset 0x4070/0x4074

Bytes of logit vectors produced and DMA'd to the host.  For a 32K-vocab
model: `decode_tokens × vocab_size × 4`.

### 7.16 `CNT_FABRIC_PACKETS_SENT_LOW/HIGH` — RO, offset 0x4078/0x407C

Total fabric packets transmitted.  Mirrors `FABRIC_PACKET_COUNTER_LOW/HIGH`
(§6.6); this counter is per-tile (in the tile register window when accessed
from the §4000 region, it aggregates all tiles).

### 7.17 `CNT_FABRIC_PACKETS_RECEIVED_LOW/HIGH` — RO, offset 0x4080/0x4084

Total fabric packets received.

### 7.18 `CNT_STALL_FENCE_LOW/HIGH` — RO, offset 0x4088/0x408C

Cycles spent waiting for a cross-tile fence to be signaled.

### 7.19 `CNT_STALL_DMA_LOW/HIGH` — RO, offset 0x4090/0x4094

Cycles spent waiting for a PCIe DMA transfer to complete.

### 7.20 `CNT_STALL_FABRIC_LOW/HIGH` — RO, offset 0x4098/0x409C

Cycles spent waiting for a fabric send or receive to complete.

### 7.21 `CNT_STALL_BARRIER_LOW/HIGH` — RO, offset 0x40A0/0x40A4

Cycles spent in `TILE_BARRIER` waiting for peer tiles.

### 7.22 `CNT_STALL_QUEUE_FULL_LOW/HIGH` — RO, offset 0x40A8/0x40AC

Cycles the command queue was full (host was stalled by backpressure).

### 7.23 `CNT_ERRORS_TOTAL_LOW/HIGH` — RO, offset 0x40B0/0x40B4

Total error events since `RESET_COUNTERS`.

### 7.24 Counter Summary Table

| Offset (LOW) | Name | ATT-1 equivalent |
|---|---|---|
| 0x4000 | `CNT_COMMANDS_ISSUED` | — |
| 0x4008 | `CNT_COMMANDS_COMPLETED` | — |
| 0x4010 | `CNT_LOCAL_OPS` | (sum of below) |
| 0x4018 | `CNT_MATMUL` | — |
| 0x4020 | `CNT_RMSNORM` | — |
| 0x4028 | `CNT_ROPE` | — |
| 0x4030 | `CNT_ATTENTION` | — |
| 0x4038 | `CNT_FFN` | — |
| 0x4040 | `CNT_KV_APPENDS` | `att1_trace_t.kv_appends` |
| 0x4048 | `CNT_KV_READS` | `att1_trace_t.kv_reads` |
| 0x4050 | `CNT_TENSOR_BYTES_READ` | — |
| 0x4058 | `CNT_TENSOR_BYTES_WRITTEN` | — |
| 0x4060 | `CNT_ACTIVATION_BYTES_SENT` | — |
| 0x4068 | `CNT_ACTIVATION_BYTES_RECEIVED` | — |
| 0x4070 | `CNT_LOGITS_BYTES` | `decode_logits_bytes` (att1-bench) |
| 0x4078 | `CNT_FABRIC_PACKETS_SENT` | `att1_trace_t.fabric_packets_sent` |
| 0x4080 | `CNT_FABRIC_PACKETS_RECEIVED` | `att1_trace_t.fabric_packets_received` |
| 0x4088 | `CNT_STALL_FENCE` | M103 §8.8 `stall_fence_cycles` |
| 0x4090 | `CNT_STALL_DMA` | M103 §8.8 `stall_dma_cycles` |
| 0x4098 | `CNT_STALL_FABRIC` | M103 §8.8 `stall_fabric_cycles` |
| 0x40A0 | `CNT_STALL_BARRIER` | M103 §8.8 `stall_barrier_cycles` |
| 0x40A8 | `CNT_STALL_QUEUE_FULL` | M103 §8.8 `stall_queue_full_cycles` |
| 0x40B0 | `CNT_ERRORS_TOTAL` | — |

---

## 8. Trace / Debug Registers

Base offset: `0x5000`

### 8.1 `TRACE_BUFFER_BASE_LOW` — RW, offset 0x5000
### 8.2 `TRACE_BUFFER_BASE_HIGH` — RW, offset 0x5004

64-bit pair.  Host-physical base address of the trace ring buffer.  Each
trace record is 64 bytes (M103 §8.1 counter snapshot format).  Must be
64-byte aligned.

### 8.3 `TRACE_BUFFER_SIZE` — RW, offset 0x5008

| Bits | Field | Description |
|---|---|---|
| 31:16 | reserved | Must be written 0 |
| 15:0 | `trace_slots` | Number of 64-byte trace records the ring can hold |

### 8.4 `TRACE_WRITE_PTR` — RO, offset 0x500C

| Bits | Field | Description |
|---|---|---|
| 31:16 | reserved | Must read 0 |
| 15:0 | `write_ptr` | Index of the next slot the AIMU will write (wraps at `trace_slots`) |

The host reads `TRACE_WRITE_PTR` to determine how many new records have been
written since the last read.  The difference `(write_ptr - last_read_ptr) %
trace_slots` gives the number of unconsumed records.

### 8.5 `TRACE_FLAGS` — RW, offset 0x5010

| Bits | Field | Description |
|---|---|---|
| 31:24 | reserved | Must be written 0 |
| 23:16 | `trace_tile_mask` | Per-tile capture enable bitmask (mirrors `TRACE_CONTROL.trace_tile_mask`) |
| 15:4 | reserved | Must be written 0 |
| 3:0 | `capture_level` | `0=off, 1=errors, 2=commands, 3=full` |

### 8.6 `TRACE_DROPPED_EVENTS` — RO, offset 0x5014

| Bits | Field | Description |
|---|---|---|
| 31:0 | `dropped_count` | Number of trace records dropped because the ring was full |

The AIMU increments this counter instead of overwriting unconsumed records.
The host should drain the trace ring promptly when `INTERRUPT_STATUS.TRACE_BUFFER_FULL` is set.

### 8.7 `TRACE_SNAPSHOT_CONTROL` — RW, offset 0x5018

| Bits | Field | Description |
|---|---|---|
| 31:24 | reserved | Must be written 0 |
| 23:16 | `snapshot_tile_mask` | Tiles to include in the snapshot (0xFF = all) |
| 15:8 | reserved | Must be written 0 |
| 7:4 | `snapshot_trigger` | `0=manual, 1=on_decode_step, 2=on_layer_boundary, 3=on_kv_append` |
| 3:1 | reserved | Must be written 0 |
| 0 | `SNAPSHOT_NOW` | Write `1` to trigger an immediate snapshot; self-clearing |

---

## 9. Error and Status Code Model

The `last_error_code` field in `TILE_ERROR_STATUS` (§3.16) and `ERROR_DETAIL`
(§2.14), and the `result_code` field in M103 completion records, all use the
8-bit error code namespace defined here.

### 9.1 Error Code Table

| Code | Symbolic name | Meaning |
|---|---|---|
| `0x00` | `STATUS_OK` | No error; command completed successfully |
| `0x01` | `STATUS_BUSY` | Tile is processing; command not yet started |
| `0x02` | `STATUS_PENDING` | Command is queued and awaiting execution |
| `0x10` | `ERR_INVALID_COMMAND` | `command_type` not recognized |
| `0x11` | `ERR_INVALID_TENSOR` | `tensor_id` not resident in tile memory |
| `0x12` | `ERR_INVALID_DTYPE` | `dtype` not supported for this op/tile |
| `0x13` | `ERR_INVALID_SHAPE` | `op_param` shape fields inconsistent |
| `0x14` | `ERR_INVALID_SESSION` | `session_id` out of range or not initialized |
| `0x20` | `ERR_OUT_OF_MEMORY` | Insufficient tile memory for tensor load or KV append |
| `0x21` | `ERR_QUEUE_FULL` | Command queue was full; command dropped |
| `0x22` | `ERR_KV_OVERFLOW` | KV position ≥ `target_context_length` |
| `0x30` | `ERR_FABRIC_ERROR` | Fabric send or reduce failed |
| `0x31` | `ERR_FABRIC_TIMEOUT` | Fabric operation exceeded `timeout_ms` |
| `0x32` | `ERR_BARRIER_TIMEOUT` | `TILE_BARRIER` not completed by all peers within `timeout_ms` |
| `0x40` | `ERR_CHECKSUM_FAIL` | Command packet or tensor payload CRC32 mismatch |
| `0x41` | `ERR_DMA_FAULT` | DMA descriptor referenced an invalid host address |
| `0x42` | `ERR_ALIGNMENT` | Buffer alignment violation |
| `0x50` | `ERR_TIMEOUT` | General command timeout |
| `0x51` | `ERR_FENCE_DEADLOCK` | Dependency fence could not be resolved |
| `0x60` | `ERR_UNSUPPORTED_OP` | Op not available on this tile (see `SUPPORTED_OPS_LOW`) |
| `0x61` | `ERR_UNSUPPORTED_DTYPE` | Dtype not available on this tile (see `SUPPORTED_DTYPES`) |
| `0xF0` | `ERR_INTERNAL` | Tile firmware internal error; issue `RESET_TILE_FULL` |
| `0xFF` | `ERR_FATAL` | Tile halted; only `RESET_TILE_FULL` or `SOFT_RESET_ALL` recovers it |

### 9.2 Error Severity Model

| Range | Severity | Recovery action |
|---|---|---|
| `0x00–0x02` | None / informational | None required |
| `0x10–0x14` | Command error (recoverable) | Correct and resubmit command |
| `0x20–0x22` | Resource error (recoverable) | Free memory / reduce context; resubmit |
| `0x30–0x32` | Fabric error (recoverable) | Re-issue with longer timeout or reduce tile count |
| `0x40–0x42` | Data integrity error (recoverable) | Re-load tensor; resubmit command |
| `0x50–0x51` | Ordering error (recoverable) | Review fence dependency chain |
| `0x60–0x61` | Capability mismatch (fatal per command) | Choose compatible dtype/op; no reset needed |
| `0xF0–0xFF` | Firmware error (fatal per tile) | Issue `RESET_TILE_FULL` or `SOFT_RESET_ALL` |

---

## 10. Host Boot and Probe Sequence

The following sequence describes how a host driver initializes an AIMU device
after PCIe enumeration.

### Step 1: Read Device Identity

```
read DEVICE_ID            // verify vendor_id == 0xA771
read DEVICE_VERSION       // record firmware version
read REGISTER_MAP_VERSION // check map_major == 1; abort if map_major > 1
```

### Step 2: Verify Feature Support

```
read FEATURE_FLAGS_LOW
if not (flags & CAP_ASYNC_COMPLETION):
    use polling mode
if not (flags & CAP_FENCE):
    issue commands in strict submission order only
```

### Step 3: Enumerate Tiles

```
read TILE_COUNT           // N tiles
for tile in 0..N-1:
    base = 0x8000 + tile * 0x800
    read base + TILE_ID
    read base + TILE_STATUS
    read base + TILE_FEATURE_FLAGS
    read base + TILE_MEMORY_CAPACITY_LOW/HIGH
    read base + TILE_KV_CAPACITY_LOW/HIGH
    read base + SUPPORTED_DTYPES
    read base + SUPPORTED_OPS_LOW
```

### Step 4: Allocate Host Buffers

```
allocate command_ring:    CQ_SIZE slots × 64 bytes, 64-byte aligned
allocate completion_ring: CQ_COMPLETION_SIZE slots × 16 bytes, 16-byte aligned
allocate trace_ring:      TRACE_BUFFER_SIZE slots × 64 bytes, 64-byte aligned
```

### Step 5: Configure Command and Completion Queues

```
write CQ_BASE_ADDR_LOW/HIGH  = physical(command_ring)
write CQ_SIZE                = command_ring_slot_count
write CQ_TAIL                = 0
write CQ_COMPLETION_ADDR_LOW/HIGH = physical(completion_ring)
write CQ_COMPLETION_SIZE     = completion_ring_slot_count
```

### Step 6: Configure Trace Buffer

```
write TRACE_BUFFER_BASE_LOW/HIGH = physical(trace_ring)
write TRACE_BUFFER_SIZE          = trace_ring_slot_count
write TRACE_FLAGS.capture_level  = 2  (commands) or 3 (full) as desired
```

### Step 7: Enable Interrupts (Optional)

```
write INTERRUPT_ENABLE = (1 << IRQ_COMPLETION) | (1 << IRQ_ERROR)
// configure MSI-X vector 0 in PCIe config space
```

### Step 8: Enable Device

```
write GLOBAL_CONTROL.ENABLE_DEVICE = 1
poll GLOBAL_STATUS.DEVICE_READY until 1
```

### Step 9: Reset / Initialize Tiles

```
for tile in 0..N-1:
    write (tile_base + TILE_RESET_CONTROL) = RESET_TILE_FULL
    poll (tile_base + TILE_STATUS).state until idle
```

### Step 10: Load Tensor Placement Plan

The host reads the M100 placement report (or M102 scenario output) and
dispatches `LOAD_TENSOR_TILE` command packets for each tensor:

```
for tensor in placement_report.tensors:
    build command_packet:
        command_type = LOAD_TENSOR_TILE (0x01)
        tile_id      = tensor.owner_tile
        tensor_id    = tensor.tensor_category_id
        dtype        = header.dtype
        input_buf_addr = staging_buffer_physical + slice_offset
        input_buf_bytes = tensor.slice_bytes
    write packet to command_ring[CQ_TAIL]
    CQ_TAIL = (CQ_TAIL + 1) % CQ_SIZE
    write CQ_DOORBELL = CQ_TAIL
    wait for completion or poll CQ_HEAD
```

### Step 11: Begin Command Dispatch

After all tensors are loaded (all `LOAD_TENSOR_TILE` commands completed), the
device is ready for inference.  The host submits `EXEC_*`, `KV_APPEND`,
`FABRIC_SEND`, and `TILE_BARRIER` commands per the M103 execution model (§7
of `docs/aimu_pcie_command_requirements.md`).

---

## 11. Relationship to ATT-1 Software Simulator

| ATT-1 simulator / tool concept | Register or mechanism |
|---|---|
| `att1-size --placement-report-json` (M100) | Informs `TILE_MEMORY_CAPACITY_LOW/HIGH` sizing and `CQ_SIZE` selection |
| M102 scenario tool `tile_memory_gib` column | Maps to `TILE_MEMORY_CAPACITY_LOW/HIGH` per SKU |
| `att1_shard_t` initialization | `LOAD_TENSOR_TILE` commands dispatched in Step 10 of §10 |
| `att1_cluster_infer_t` scheduler | Host command producer; writes to `CQ_TAIL` / `CQ_DOORBELL` |
| `att1_fabric_t.packets_sent` | `CNT_FABRIC_PACKETS_SENT_LOW/HIGH` (§7.16) |
| `att1_trace_t.kv_appends` | `CNT_KV_APPENDS_LOW/HIGH` (§7.9) |
| `att1_trace_t.kv_reads` | `CNT_KV_READS_LOW/HIGH` (§7.10) |
| `att1_trace_t.fabric_packets_sent` | `FABRIC_PACKET_COUNTER_LOW/HIGH` (§6.6) |
| `decode_logits_bytes` (att1-bench output) | `CNT_LOGITS_BYTES_LOW/HIGH` (§7.15) |
| `prefill_time_us_total` (att1-bench output) | Derivable from `CNT_ATTENTION` + `TRACE_SNAPSHOT` records |
| `q4` / `q8` / `f32` dtype selection | `SUPPORTED_DTYPES` (§3.12), `dtype` field in command packet |
| `att1-bench` exit code conventions | Mapped from `STATUS_OK (0x00)` or `ERR_*` codes (§9.1) |
| `check_placement_scenarios_smoke()` (M102) | Validates that scenario capacity estimates match `TILE_MEMORY_CAPACITY_LOW/HIGH` |
| `sim_fabric_bus.c` fabric simulator | Behavioral model for `FABRIC_STATUS`, `FABRIC_CONTROL`, `FABRIC_ERROR_STATUS` |

---

## 12. Non-Goals

- No Linux kernel driver implementation.  The register map is a hardware
  specification; driver skeleton is deferred to M106.
- No PCIe endpoint simulation.  No C code models the BAR MMIO interface in
  this milestone.
- No MMIO simulator implementation.  The M105 command queue simulator models
  the command protocol over shared memory, not MMIO registers.
- No production ASIC register finalization.  Offsets and field widths are
  provisional and may change before tape-out.
- No interrupt controller configuration.  PCIe MSI-X vector setup is a
  driver concern deferred to M106.
- No DMA engine implementation.  The scatter-gather descriptor layout (§5) is
  a specification; the DMA engine is deferred to M107.
- No physical layer specification (electrical, optical, PCIe signal integrity).
- No patent claim language.
- No change to the ATT-1 C11 runtime, `.att1` binary format, or any existing
  inference behavior.

---

## 13. Future Milestone Split

| Milestone | Title | Scope |
|---|---|---|
| M105 | PCIe command queue simulator | Python or C shim over POSIX shared memory that models the command ring buffer (§4), processes 64-byte command packets, writes completion records, and validates checksums; smoke test: submit `LOAD_TENSOR_TILE` + `EXEC_MATMUL` against the tiny dummy model and compare result to `att1-bench cpu-f32` |
| M106 | AIMU device discovery simulator | **Complete.** Implemented as an in-process C11 simulator (`include/att1_aimu_device.h`, `src/aimu_device.c`). See §14 for the full struct-to-register mapping. |
| M107 | DMA descriptor simulator | Implement the §5 descriptor layout; validate host-to-AIMU tensor load round-trip; integrate with M105/M106 |
| M108 | Command trace/counter integration | Wire the §7 counter registers and §8 trace buffer into the M105/M106 simulator; export trace records in `att1_trace_t`-compatible JSON for diff against `att1-bench` output |
| M109 | Placement-report-to-command-plan mapper | Python tool that reads an M100 placement report and produces an ordered list of `LOAD_TENSOR_TILE` + `EXEC_*` commands with register field values filled in; validates command plan against placement report tensor/tile assignments |
| M110 | Minimal PCIe/AIMU prototype design review | Engineering review: reconcile M104–M109 artifacts, finalize BAR0 offset assignments, resolve open questions from §14, produce hardware bringup checklist |

---

## 14. M106 In-Process Device Simulator

M106 delivers a pure C11, in-process simulator for the device probe and tile
enumeration registers defined in §2 and §3.  There is **no real PCIe bus, no
MMIO, and no kernel driver** involved.  All register state lives in a
heap-allocated `att1_aimu_device` struct that is created and destroyed by the
host process.

### 14.1 Struct-to-Register Field Mapping

The table below maps each `att1_aimu_device` / `att1_aimu_tile_info` field to
the BAR0 register it simulates.

| C struct field | BAR0 register (§) | Offset |
|---|---|---|
| `att1_aimu_device.register_map_version` | `REGISTER_MAP_VERSION` (§2) | `0x0008` |
| `att1_aimu_device.version` (major/minor/patch/build) | `DEVICE_VERSION` (§2) | `0x0004` |
| `att1_aimu_device.feature_flags` | `FEATURE_FLAGS_LOW` / `FEATURE_FLAGS_HIGH` (§2) | `0x0010` / `0x0014` |
| `att1_aimu_device.global_status` | `GLOBAL_STATUS` (§2) | `0x0038` |
| `att1_aimu_device.global_error` | `ERROR_STATUS` (§9) | `0x0030` |
| `att1_aimu_device.tile_count` | `TILE_COUNT` (§2) | `0x000C` |
| `att1_aimu_tile_info.memory_capacity_bytes` | `TILE_MEMORY_CAPACITY_LOW/HIGH` (§3) | `0x8000 + N×0x800 + 0x0008/0x000C` |
| `att1_aimu_tile_info.kv_capacity_bytes` | `TILE_KV_CAPACITY_LOW/HIGH` (§3) | `0x8000 + N×0x800 + 0x0010/0x0014` |
| `att1_aimu_tile_info.supported_dtypes` | `SUPPORTED_DTYPES` (§3) | `0x8000 + N×0x800 + 0x0020` |
| `att1_aimu_tile_info.supported_ops` | `SUPPORTED_OPS_LOW` (§3) | `0x8000 + N×0x800 + 0x0024` |
| `att1_aimu_tile_info.state` | `TILE_STATUS` (§3) | `0x8000 + N×0x800 + 0x0000` |
| `att1_aimu_tile_info.fabric_link_mask` | `FABRIC_LINK_MASK` (§3) | `0x8000 + N×0x800 + 0x0028` |
| `att1_aimu_tile_info.max_sessions` | `MAX_SESSIONS` (§3) | `0x8000 + N×0x800 + 0x002C` |
| `att1_aimu_tile_info.memory_used_bytes` | `TILE_MEMORY_USED_LOW/HIGH` (§3) | `0x8000 + N×0x800 + 0x0030/0x0034` |
| `att1_aimu_tile_info.kv_used_bytes` | `TILE_KV_USED_LOW/HIGH` (§3) | `0x8000 + N×0x800 + 0x0038/0x003C` |
| `att1_aimu_tile_info.error_code` | `TILE_ERROR_CODE` (§3) | `0x8000 + N×0x800 + 0x0004` |
| `att1_aimu_tile_info.reset_count` | `TILE_RESET_COUNT` (§3) | `0x8000 + N×0x800 + 0x0040` |

### 14.2 API Reference

```c
/* Lifecycle */
att1_status_t att1_aimu_device_create(const att1_aimu_device_config *cfg,
                                      att1_aimu_device **out);
void          att1_aimu_device_destroy(att1_aimu_device *dev);

/* Query */
att1_status_t att1_aimu_device_query_info(const att1_aimu_device *dev,
                                           att1_aimu_device_info *out);
size_t        att1_aimu_device_tile_count(const att1_aimu_device *dev);
att1_status_t att1_aimu_device_query_tile(const att1_aimu_device *dev,
                                           size_t tile_id,
                                           att1_aimu_tile_info *out);
att1_status_t att1_aimu_device_validate_tile_id(const att1_aimu_device *dev,
                                                 size_t tile_id);

/* Capability checks */
int           att1_aimu_device_supports_dtype(const att1_aimu_device *dev,
                                               uint32_t dtype_bit);
int           att1_aimu_device_supports_op(const att1_aimu_device *dev,
                                            uint32_t op_bit);
int           att1_aimu_device_tile_supports_dtype(const att1_aimu_device *dev,
                                                    size_t tile_id,
                                                    uint32_t dtype_bit);
int           att1_aimu_device_tile_supports_op(const att1_aimu_device *dev,
                                                 size_t tile_id,
                                                 uint32_t op_bit);

/* Reset */
att1_status_t att1_aimu_device_reset(att1_aimu_device *dev);
att1_status_t att1_aimu_device_reset_tile(att1_aimu_device *dev, size_t tile_id);

/* Command-queue integration */
att1_status_t att1_aimu_device_attach_cmdq(att1_aimu_device *dev,
                                            att1_aimu_cmdq *cmdq);
att1_status_t att1_aimu_device_snapshot_counters(const att1_aimu_device *dev,
                                                  att1_aimu_cmdq_counters *out);

/* Name helpers */
const char *att1_aimu_dtype_name(uint32_t dtype_bit);
const char *att1_aimu_op_name(uint32_t op_bit);
const char *att1_aimu_feat_name(uint64_t feat_bit);
const char *att1_aimu_tile_state_name(att1_aimu_tile_state state);
```

### 14.3 Using Device Capabilities for Placement Validation

Future placement logic (M98–M102) can query the simulated device to validate a
proposed tensor/session placement before issuing commands:

```c
/* Check that all tiles can handle the required dtype. */
if (!att1_aimu_device_supports_dtype(dev, ATT1_AIMU_DTYPE_Q8)) { ... }

/* Check that a specific tile has enough memory for the weight tensor. */
att1_aimu_tile_info tile;
att1_aimu_device_query_tile(dev, target_tile, &tile);
if (tile.memory_capacity_bytes - tile.memory_used_bytes < required_bytes) { ... }

/* Check that the target tile supports the required ops. */
if (!att1_aimu_device_tile_supports_op(dev, target_tile, ATT1_AIMU_OP_ATTENTION)) { ... }
```

---

## Appendix A: BAR0 Register Map Summary

| Offset range | Region |
|---|---|
| `0x0000–0x003F` | Global device registers (§2) |
| `0x0040–0x0FFF` | Global registers reserved |
| `0x1000–0x102F` | Command queue registers (§4) |
| `0x1030–0x1FFF` | Command queue reserved |
| `0x2000–0x2017` | DMA control registers (§5.3–5.8) |
| `0x2018–0x2FFF` | DMA reserved |
| `0x3000–0x302B` | Fabric/interconnect registers (§6) |
| `0x302C–0x3FFF` | Fabric reserved |
| `0x4000–0x40B7` | Counter registers (§7) |
| `0x40B8–0x4FFF` | Counter reserved |
| `0x5000–0x501B` | Trace/debug registers (§8) |
| `0x501C–0x7FFF` | Trace/debug and misc reserved |
| `0x8000–0x87FF` | Tile 0 register window (§3) |
| `0x8800–0x8FFF` | Tile 1 register window |
| `0x9000–0x97FF` | Tile 2 register window |
| `0x9800–0x9FFF` | Tile 3 register window |
| `0xA000–0xA7FF` | Tile 4 register window |
| `0xA800–0xAFFF` | Tile 5 register window |
| `0xB000–0xB7FF` | Tile 6 register window |
| `0xB800–0xBFFF` | Tile 7 register window |
| `0xC000–0xC7FF` | Tile 8 register window |
| `0xC800–0xCFFF` | Tile 9 register window |
| `0xD000–0xD7FF` | Tile 10 register window |
| `0xD800–0xDFFF` | Tile 11 register window |
| `0xE000–0xE7FF` | Tile 12 register window |
| `0xE800–0xEFFF` | Tile 13 register window |
| `0xF000–0xF7FF` | Tile 14 register window |
| `0xF800–0xFFFF` | Tile 15 register window |

---

## Appendix B: Open Engineering Questions

1. **Multi-function devices.** Should cards with more than 16 tiles expose
   multiple PCIe functions (one per 16-tile group), or use a BAR extension
   beyond 64 KiB?

2. **MSI-X vector allocation.** One vector per tile?  Or one per interrupt
   source class (completion, error, trace)?

3. **Command queue sharing.** Is the command ring buffer per-tile (one ring
   per tile at different offsets) or a single shared ring with per-tile tag
   routing?  The current spec implies a single ring; separate rings may be
   needed for per-tile prioritization.

4. **Fence ID namespace.** Are fence IDs per-tile (reused per session) or
   globally unique per device?

5. **64-bit MMIO atomicity.** Which platforms guarantee 64-bit PCIe MMIO
   writes are atomic?  If atomicity is not guaranteed, are the `_LOW` /
   `_HIGH` pairs safe under concurrent access from multiple host threads?

6. **Counter overflow behavior.** Saturating at `0xFFFF_FFFF_FFFF_FFFF` or
   wrapping?  The current spec says saturate; wrapping counters are simpler
   for hardware but require the host to track rollovers.

7. **Trace ring full policy.** Drop-on-full (current spec) or overwrite oldest?
   Overwrite would ensure the most recent events are always available but
   may lose early prefill trace records.

8. **Q4 group size flexibility.** Is `group_size=32` fixed in `SUPPORTED_DTYPES`,
   or should a separate `Q4_GROUP_SIZE` register report the supported values?

9. **Tile memory map granularity.** Should each tile expose its own DMA
   address window (separate `TILE_MEMORY_BASE` register per tile), or is the
   AIMU-local address space flat across all tiles?

10. **Routing table ownership.** Who populates `FABRIC_ROUTE_TABLE_BASE`?
    The host driver (using M109 placement plan output) or the AIMU firmware
    (using a built-in topology discovery protocol)?
