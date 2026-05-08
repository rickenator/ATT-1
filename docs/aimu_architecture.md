# AIMU Architecture — ATT-1 Tensor Tile Concept

This document describes the AIMU (Application-specific Inference Memory Unit)
concept that motivates the ATT-1 artifact format and C11 runtime prototype.

## AIMU

**AIMU** — pronounced **EYE-mew** — stands for **Application-specific Inference Memory Unit**.

An AIMU is a programmable near-memory tensor-tile execution unit. In the current software prototype, a cluster node or shard simulates an AIMU. In future silicon, an AIMU would own a tensor tile, execute local inference operations, validate tile metadata, and communicate intermediate results through the inference fabric.

In the current ATT-1 software prototype, a cluster node or shard simulates an AIMU. In future silicon, an AIMU would own tensor memory directly, execute local inference operations near that data, validate tile metadata, and exchange only the necessary intermediate results through the inference fabric.

The central idea is simple:

> Move computation to the tensor tile, rather than repeatedly moving tensor data to distant compute units.

This makes the cluster backend more than a distributed runtime experiment. It is a software model for an AIMU-style inference fabric.

---

## 1. Core Concept

Large transformer model inference is dominated by moving weight tensors from
storage into compute units.  The AIMU architecture inverts this relationship:
instead of moving large tensors to compute, compute is placed near memory.

The key ideas are:

- **Tensor partitioning.** Model tensor space is partitioned into tensor tiles.
  Each tile contains a contiguous, aligned region of model weights — typically
  one or more layers, or a shard of a single large projection.

- **AIMU ownership.** Each tensor tile is assigned to one AIMU.  The AIMU owns
  the local tensor memory and is responsible for executing inference operations
  against it.  No external controller reads or writes AIMU-owned tensor memory
  during inference; the AIMU receives compact activations and control signals
  instead.

- **Near-memory execution.** Operations such as matrix-multiply-accumulate,
  RMSNorm, RoPE, FFN/SwiGLU, and KV-cache updates are executed by the AIMU
  locally, without evicting tensor data to a shared bus.

- **Fabric coordination.** A packetized fabric carries:
  - Activation vectors (input to each tile's computation)
  - Logit vectors (output from the final projection tile)
  - Synchronization barriers between dependent tiles
  - Reductions (e.g., summing partial logits across shards)
  - Routing control for multi-tile sequences

- **Runtime scheduling.** The host runtime partitions the model, assigns tiles,
  schedules operations, validates correctness via trace counters, and coordinates
  KV-cache lifecycle.

---

## 2. Why AIMU Matters

### Conventional inference model

Conventional GPU/CPU runtimes treat memory as passive storage.  Each operation
fetches tensor data from DRAM or HBM, performs a computation in a centralized
compute unit, and stores results back.  At large model scale, this results in
repeated, high-bandwidth weight movement across shared buses for every decode
step.

### AIMU model

ATT-1 / AIMU treats tensor memory as **active** and **compute-local**:

| Conventional | AIMU |
|---|---|
| Memory is passive storage | Memory is an owned compute resource |
| Weights pulled to centralized compute | Activations pushed to near-memory compute |
| Shared bus bandwidth dominates | Compact activation vectors dominate traffic |
| One large compute block | Many small tiled compute blocks |
| Memory hierarchy managed by runtime | Memory hierarchy managed per-AIMU |

The goal is **memory-centric inference**: reduce weight movement by routing
compact activations and control signals to stationary tensor tiles, rather than
streaming large weight blocks to centralized execution units at every step.

This is especially beneficial at scale:

- Multi-layer models can pipeline activations across tiles without round-trips
  to a centralized memory pool.
- KV caches are managed locally by each AIMU's session memory unit (KV-MMU),
  avoiding centralized KV traffic.
- Fabric bandwidth is proportional to activation size (small) rather than
  weight size (large).

 ## Why AIMU Is Different from GPU/TPU Acceleration

ATT-1/AIMU is not primarily a bigger compute engine. It is a memory-centric inference architecture.

Conventional accelerators treat memory as passive storage. A GPU or TPU pulls tensor data through a memory hierarchy into execution units, performs compute, then writes results back. That model is powerful, but large-model inference often becomes constrained by memory bandwidth, tensor movement, KV-cache movement, synchronization, and placement policy rather than raw arithmetic alone.

AIMU changes the primitive unit of execution.

Instead of treating tensors as data fetched by a centralized accelerator, ATT-1 partitions model tensor space into tensor tiles. Each tensor tile is assigned to an AIMU — an Application-specific Inference Memory Unit — that owns local tensor memory and performs programmable inference operations near that memory 

---

## 3. Current Prototype Mapping

The ATT-1 C11 simulator approximates the AIMU architecture using standard Linux
processes, threads, and data structures.  The mapping is:

| ATT-1 prototype component | Future AIMU hardware analog |
|---|---|
| Cluster node (`att1_cluster_infer_t`) | AIMU tile unit |
| Model shard (`att1_shard_t`) | AIMU-owned local tensor memory |
| Backend vtable (`att1_backend_ops`) | AIMU programmable local op set |
| KV-MMU (`att1_kv_mmu_t`) | AIMU-managed session memory controller |
| Fabric simulator (`att1_fabric_t`) | Tile-to-tile interconnect |
| Tile runtime threads (`att1_tile_t`) | AIMU execution thread / pipeline stage |
| Trace counters (`att1_trace_t`) | Deterministic proof-of-execution log |
| `.att1` model artifact | AIMU firmware blob and tensor layout descriptor |

Each component is designed so that the same abstraction can be expressed on
real hardware without changing the model artifact format or the public API.

---

## 4. Conceptual Stack

```
┌─────────────────────────────────────────────────────┐
│  ATT-1 artifact format                              │
│  Versioned binary model; config + named tensor      │
│  descriptors; dtype; shard metadata (future)        │
├─────────────────────────────────────────────────────┤
│  AIMU                                               │
│  Tile-local programmable inference unit;            │
│  owns tensor memory; executes backend ops locally   │
├─────────────────────────────────────────────────────┤
│  Fabric                                             │
│  Tile-to-tile packet interconnect; routes           │
│  activations, logits, barriers, reductions          │
├─────────────────────────────────────────────────────┤
│  Runtime                                            │
│  Scheduler, shard planner, validator, executor;     │
│  coordinates KV-cache, token decode loop            │
├─────────────────────────────────────────────────────┤
│  Trace                                              │
│  Deterministic execution log; per-layer counters;   │
│  fabric byte/packet accounting; validation hooks    │
└─────────────────────────────────────────────────────┘
```

The layers are strictly ordered: higher layers depend on lower layers only
through defined interfaces.  The `.att1` format is the contract between the
converter toolchain (Python, under `compiler/`) and the C11 runtime.

---

## 5. Future `.att1` Shard Metadata

The current `.att1` format encodes a flat tensor table with per-tensor
descriptors.  Future revisions will add a shard metadata section to express
AIMU placement and routing requirements directly in the model artifact.

Planned per-shard metadata fields:

| Field | Description |
|---|---|
| `tensor_id` | Unique tensor identifier within the model |
| `tile_id` | Target AIMU tile assignment |
| `offset` | Byte offset within AIMU local memory |
| `shape` | Tensor dimensions (up to 4D) |
| `dtype` | Data type code (f32=1, q8=2, reserved) |
| `quantization` | Per-row scale offset or inline scale table ref |
| `owner_aimu` | Owning AIMU address or broadcast mask |
| `replication_policy` | None / read-replicate / write-broadcast |
| `dependency_graph` | Op dependency edges for pipeline scheduling |
| `allowed_ops` | Bitmask of permitted local operations |
| `routing_requirements` | Fabric path constraints and QoS class |
| `reduction_behavior` | None / sum / max / custom-reduce |
| `checksum` | CRC32 or SHA256 truncated for loader validation |

Until these fields are standardized, shard planning is performed at runtime by
`att1_shard_plan_build()`.  The format version field (`version=1`) will be
incremented when shard metadata becomes mandatory.

---

## 6. Software Prototype vs Future Silicon

### Phase 1: Software prototype (current)

- Each AIMU tile is a Linux thread with an allocated model shard.
- Tensor memory is heap-allocated; no special placement or pinning.
- Backend ops are implemented in C (CPU f32, CPU q8) or dispatched to cuBLAS
  (CUDA backend for live-model numerical validation).
- Fabric is a bounded-queue packet bus running in userspace.
- KV-MMU is a paged in-process cache.

### Phase 2: PCIe prototype (bridge target)

- AIMU tiles become PCIe endpoint memory regions.
- Host runtime enqueues commands to hardware tile command queues.
- Fabric becomes a PCIe switch / NTB fabric.
- KV-MMU maps to on-card SRAM or HBM with host-managed page tables.
- The `.att1` format is used directly as the firmware loading artifact.

### Phase 3: AIMU silicon (target)

- Dedicated ASIC implements the AIMU tile as a fixed-function + programmable
  block: local tensor SRAM, multiply-accumulate array, RoPE/norm unit, and
  fabric transceiver.
- Host runtime API and `.att1` format remain unchanged.
- CUDA backend is retired; the AIMU silicon backend replaces it.

The abstraction is the same at every phase: a tile owns local tensor memory and
exposes a programmable operator interface.  The substrate changes; the model and
runtime contract do not.

---

## 7. Non-Goals

The following are explicitly outside the AIMU / ATT-1 scope:

- **Not a generic ML framework.** ATT-1 targets decoder-only transformer
  inference.  It does not implement training, automatic differentiation,
  dynamic shapes, or operator fusion beyond what the fixed inference path
  requires.

- **Not a CUDA replacement in Phase 1.** The CUDA backend exists for numerical
  validation of math primitives against a known-correct GPU implementation.
  The goal is not to compete with cuDNN or Triton.

- **Not GPT-OSS 120B real inference yet.** The current prototype targets tiny
  synthetic models.  Real large-model inference is a Phase 2+ target after the
  shard metadata, cluster protocol, and hardware bridge are stable.

- **Not mobile, Android, Vulkan, or OpenCL.** The hardware target is a PCIe
  card or discrete AIMU tile, not a mobile SoC.  No mobile or graphics APIs
  will be added to the runtime.

---

## Related Documents

- [model_format.md](model_format.md) — `.att1` binary format specification
- [real_model_conversion.md](real_model_conversion.md) — LLaMA converter plan and tensor naming table
- [cluster_inference.md](cluster_inference.md) — multi-tile cluster inference protocol
- [fabric.md](fabric.md) — packet fabric simulator design
- [kv_mmu.md](kv_mmu.md) — KV-MMU paged session memory design
- [tile_runtime.md](tile_runtime.md) — tile thread lifecycle and command dispatch
- [tracing.md](tracing.md) — trace counter and proof-of-execution design
- [backend.md](backend.md) — backend vtable abstraction and operator set

---

## 8. Phase 2 PCIe/AIMU Prototype Requirements (M93)

This section specifies the engineering requirements for a Phase 2 PCIe/AIMU
prototype.  The ATT-1 software simulator, backend matrix, shard metadata,
q4/q8/f32 inference paths, and scaling reports (M72, M90–M92) are the
reference basis.

---

### 8.1 Prototype Goal

Demonstrate that the ATT-1 cluster inference protocol — as defined by the
software simulator — can execute on real hardware over a PCIe interconnect,
with tensor tiles resident in device-local memory and the host runtime
driving the decode loop through defined control-plane interfaces.

The prototype does **not** need to achieve production throughput.  It needs to
prove that the protocol is hardware-expressible and that the abstractions in
the software simulator correspond to physically realizable hardware primitives.

---

### 8.2 What the Prototype Must Prove

1. **Tile isolation.** A PCIe endpoint can hold a model shard exclusively in
   device-local memory and execute at least one transformer block forward pass
   (matmul, RMSNorm, RoPE, FFN/SwiGLU, softmax, attention) without the host
   reading or writing that tensor memory during inference.

2. **Activation routing.** Compact activation vectors (not weight tensors) can
   be transferred across the PCIe bus between tiles with measured latency and
   bandwidth, matching the fabric packet model used by the ATT-1 simulator.

3. **KV-cache locality.** Session key/value memory can be maintained
   device-locally across multiple decode steps without copying KV data to the
   host between tokens.

4. **Multi-tile decode.** At least two tiles can cooperate on a single decode
   step using the fabric barrier and reduction protocol, producing a final
   logit vector that matches the software simulator output within the
   established tolerances (f32: exact; q8: abs ≤ 0.15; q4: abs ≤ 0.35).

5. **Trace and counter fidelity.** The hardware tile must report the same
   counter categories (fabric packets sent, KV appends, logits bytes produced,
   fabric barrier completions) that the software simulator records, so that
   existing smoke tests can validate hardware output without modification.

6. **Dtype coverage.** At minimum, the f32 reference path and the q8 path must
   be validated end-to-end.  q4 on the prototype is desirable but not blocking.

---

### 8.3 Relationship to the ATT-1 Runtime

The PCIe prototype uses the existing ATT-1 C11 runtime unchanged as the host
control plane.  No new host API is needed for Phase 2:

- `att1_cluster_infer_create()` / `att1_cluster_infer_decode()` drive the
  decode loop.
- `att1_shard_plan_build()` / `att1_shard_plan_from_meta()` assign tensor
  tiles to PCIe endpoint memory regions.
- `att1_fabric_t` packet semantics are preserved; the PCIe bus is the physical
  fabric transport.
- `att1_trace_t` counters are fed from hardware-reported values at each step.
- The `.att1` artifact is used directly as the tensor layout and configuration
  source.  No format change is required for Phase 2.

The backend vtable (`att1_backend_ops`) gains one new concrete implementation:
`backend_pcie.c` (or equivalent), which replaces the CPU and CUDA dispatch
with PCIe DMA commands to the hardware tile.  All other runtime code is
unchanged.

---

### 8.4 Relationship to CUDA Validation

The CUDA backend (M12–M92) established that:

- All math primitives produce numerically correct output at f32, q8, and q4
  precision (verified on RTX 3090, M90–M92).
- The backend vtable abstraction is sufficient to hot-swap the execution
  substrate without changing the inference driver.
- Silent-fallback detection and fabric-packet counters can be used to
  distinguish real hardware dispatch from a CPU fallback.

These properties carry forward to the PCIe prototype:

- The CUDA tolerance baselines (f32: exact; q8 ≤ 0.15; q4 ≤ 0.35) are the
  acceptance tolerances for prototype validation.
- The backend-swap pattern from CUDA is reused for the PCIe backend.
- The backend comparison report (`compiler/backend_comparison_report.py`, M92)
  can be extended with a `pcie` backend column to validate the prototype.

---

### 8.5 AIMU Tile Responsibilities

Each prototype tile must implement the following, locally:

| Responsibility | Required for Phase 2 | Notes |
|---|---|---|
| Tensor memory ownership | yes | tile holds shard; host does not read/write tensor data during inference |
| Matmul (f32) | yes | reference correctness path |
| Matmul (q8×f32) | yes | required for q8 backend validation |
| Matmul (q4×f32) | desirable | blocking for full q4 coverage |
| RMSNorm | yes | pre-attention and pre-FFN normalization |
| RoPE | yes | rotary positional embedding applied to Q and K |
| FFN / SwiGLU | yes | gate+up projection, SiLU activation, down projection |
| Softmax (causal) | yes | attention weight normalization |
| KV-cache append/read | yes | local session memory; one context per session |
| Fabric send/receive | yes | activation and logit routing |
| Barrier participation | yes | cross-tile synchronization per decode step |
| Partial logit reduction | yes | sum partial logits from shards |
| Counter reporting | yes | packets sent, KV appends, logits bytes |

---

### 8.6 Local Tensor Memory Requirements

For a two-tile prototype with a SmolLM2-135M class model (135 M parameters):

| Format | Total weight bytes | Per-tile (2 tiles) |
|--------|-------------------|--------------------|
| f32    | ~540 MB           | ~270 MB            |
| q8     | ~135 MB           | ~68 MB             |
| q4     | ~68 MB            | ~34 MB             |

KV cache per tile (32 layers, 9 heads, head_dim=64, 2048 tokens, f32):
`32 × 9 × 64 × 2048 × 4 bytes × 2 (K+V) ≈ 300 MB`

A Phase 2 PCIe card requires a minimum of **512 MB device-local SRAM or HBM**
to hold the q8 shard and a 2048-token KV cache with margin.  **1 GB** is
recommended for f32 shard + KV cache.  **256 MB** is sufficient for a q4-only
two-tile proof with a short context window (≤512 tokens).

These figures assume no tensor compression beyond what the `.att1` dtype
already encodes.  No additional quantization or sparsity is required for
Phase 2.

---

### 8.7 Host Control Plane Requirements

The host runtime must be able to:

1. **Load** the `.att1` artifact and map tensor descriptors to PCIe BAR address
   ranges (or DMA target addresses) on the device.
2. **Transfer** tensor shards to device-local memory once at model load time.
   Subsequent inference steps must not require re-transfer of weight tensors.
3. **Enqueue** per-decode commands to the tile command queue (activation input,
   decode step index, KV position, tile count, backend selector).
4. **Collect** per-step outputs: final logit vector (device → host), trace
   counter snapshot.
5. **Manage** session state: open session, close session, evict KV pages.
6. **Tolerate** tile errors: read status register and map hardware errors to
   `att1_status_t` codes without undefined behaviour.

The host control plane must not issue fabric packets directly.  Fabric routing
is the tile's responsibility.  The host sends decode commands; the tile drives
the fabric.

---

### 8.8 Fabric / Interconnect Requirements

| Property | Requirement |
|---|---|
| Physical transport | PCIe Gen 3 x4 or better (or NTB / PCIe switch for multi-card) |
| Logical model | Matches `att1_fabric_t`: bounded queues, send/receive, broadcast, barrier |
| Packet payload | Activation vector per tile: `d_model × sizeof(dtype)` bytes (e.g., 512 B for d_model=128, f32) |
| Max in-flight packets | ≥ tile_count × 2 (pipeline two steps without stall) |
| Barrier semantics | All-or-nothing: barrier completes only when all participants have arrived |
| Reduction | Partial logit sum across shards before final sampling |
| Queue-full behaviour | Must return detectable error; must not silently drop packets |
| Counter requirements | packets_sent, bytes_sent, broadcast_packets, barrier_arrivals, queue_full_errors |

For a two-tile software-emulated PCIe endpoint prototype, shared memory or
Unix domain sockets may substitute for physical PCIe, provided the same queue
semantics and counter definitions are preserved.

---

### 8.9 KV-MMU / Session Memory Requirements

Each tile's KV-MMU must implement:

| Capability | Requirement |
|---|---|
| Address dimensions | session_id, layer_id, head_id, position |
| Page granularity | Configurable; 16 or 32 tokens per page sufficient for Phase 2 |
| Append ordering | Sequential per session/layer; out-of-order appends must fail cleanly |
| Duplicate-append rejection | Appending an already-populated position must fail |
| Range-copy | Copy a range of token positions for attention window reads |
| Session lifecycle | Open, append, read, close, evict per session |
| Error mapping | All KV errors must map to `att1_status_t` codes |
| Counter reporting | page_hits, page_misses, page_alloc, appends, reads, errors |

KV memory is device-local.  The host does not read or write KV entries during
inference.  Session eviction may be host-initiated but must complete without
disrupting other active sessions.

---

### 8.10 Required Counters / Trace / Debug Visibility

At minimum, each hardware tile must expose the following counters per decode
step, readable by the host after step completion:

| Counter | Source | Purpose |
|---|---|---|
| `fabric_packets_sent` | fabric unit | detects silent no-op execution |
| `fabric_packets_received` | fabric unit | confirms activation delivery |
| `kv_appends` | KV-MMU | confirms KV cache is advancing |
| `kv_reads` | KV-MMU | confirms attention reads |
| `logits_bytes_produced` | output unit | confirms correct output size |
| `barrier_completions` | fabric unit | confirms cross-tile sync |
| `decode_steps_executed` | tile control | decode loop progress |
| `error_count` | all units | aggregate error indicator |

These counters map 1-to-1 with the fields already validated by the ATT-1
software smoke tests and the backend comparison report.  Hardware that reports
the same counter names can be validated by the existing Python and C test
infrastructure without modification.

Per-layer timing counters (microseconds) are desirable but not required for
Phase 2 prototype acceptance.

---

### 8.11 Minimal Viable Prototype Options

Four options are ordered by implementation cost:

#### Option A: Software-emulated PCIe endpoint (lowest cost)

Replace the `att1_fabric_t` userspace queue with a shared-memory or Unix
socket transport that exposes the same send/receive/barrier/counter API.
Tensor memory is process-local.  Validates the protocol mapping without
any physical hardware.  Suitable as a Phase 2 pre-step before acquiring
FPGA or PCIe card hardware.

**Already partially realised** by the existing `simulator/sim_fabric_bus.c`
and `simulator/sim_tile_thread.c`.

#### Option B: FPGA board (mid cost)

Implement the AIMU tile logic on a PCIe-attached FPGA (e.g., AMD/Xilinx
Alveo U50/U55C or Intel Stratix 10 MX).  The FPGA holds tensor SRAM
(HBM or block RAM), implements the matmul/norm/RoPE ops in RTL or HLS,
and exposes a PCIe BAR-mapped command queue to the host.  The ATT-1 runtime
loads the `.att1` shard, enqueues decode commands, and reads counters.

This is the recommended first physical-hardware target.

#### Option C: PCIe accelerator card (off-the-shelf)

Use an existing PCIe inference card (e.g., Hailo-8, Coral M.2 PCIe,
or a research-grade card with open firmware) as a protocol-compatible
endpoint.  Requires porting the AIMU tile ops to the card's native
programming model and exporting the required counter interface.

Feasibility depends on card-specific memory and programmability constraints.

#### Option D: Custom silicon (long-term target)

Full ASIC implementation of the AIMU tile: local tensor SRAM,
multiply-accumulate array, RoPE/norm unit, fabric transceiver, KV-MMU
controller, and PCIe or custom interconnect.  Host API and `.att1` format
remain identical to Phase 1 and Phase 2.  This is Phase 3.

---

### 8.12 Data Movement Assumptions

| Movement | Direction | When |
|---|---|---|
| Tensor shard (model weights) | host → device | once, at model load |
| Activation vector | host → tile-0 (first token position) | per decode step |
| Activation vector | tile-N → tile-N+1 (intermediate) | per decode step, via fabric |
| Partial logits | tile → tile (reduction) | per decode step, via fabric |
| Final logits | device → host | per decode step |
| KV cache entries | device-local only | never cross PCIe during inference |
| Trace counters | device → host | per decode step (small struct read) |
| Session eviction commands | host → device | on session close or OOM |

Weight tensors never leave device memory after initial load.  The only
per-step host↔device traffic is: one activation vector in, one logit vector
out, one counter snapshot out.

---

### 8.13 Supported Dtypes

| Dtype | Phase 2 status | Tolerance |
|---|---|---|
| f32 | required | exact (reference) |
| q8 (int8 weights × f32 activations) | required | abs ≤ 0.15 per logit |
| q4 (int4 grouped weights × f32 activations) | desirable | abs ≤ 0.35 per logit |

Tolerance values are taken from the CUDA validation results (M90–M92, RTX 3090
verified).  A hardware tile that meets these tolerances for f32 and q8 is
considered Phase 2 complete for dtype coverage.  q4 hardware support may be
deferred to a later milestone without blocking Phase 2 sign-off.

---

### 8.14 Non-Goals

The following are explicitly outside the Phase 2 prototype scope:

- **Full production ASIC.** Phase 2 is a protocol and correctness proof.
  Timing closure, power budgets, and production yield are Phase 3 concerns.
- **Public cloud deployment.** The prototype runs on a local PCIe card or FPGA
  connected to the development machine used for ATT-1 simulation.
- **Mobile / Android / Vulkan / OpenCL.** No mobile or graphics APIs will be
  added.  The hardware target is a PCIe card or discrete AIMU tile.
- **Training or fine-tuning.** ATT-1 is inference-only.  No gradient or
  backward-pass hardware is required.
- **Multi-model switching.** Phase 2 loads one model per prototype session.
  Dynamic model hot-swap is a Phase 3 feature.
- **Patent claims.** This document is an engineering specification.  No claim
  language is used or implied.

---

### 8.15 Open Engineering Questions

1. **Matmul unit.** Should Phase 2 use a systolic array, a dot-product
   accumulator, or an off-the-shelf DSP block?  The answer affects FPGA
   resource utilisation and latency per decode step.

2. **KV-MMU page size.** What page granularity minimises wasted SRAM while
   supporting 2048-token contexts on a 512 MB device?  Initial estimate:
   32 tokens/page for q8 KV.

3. **Activation precision.** The software simulator passes f32 activations
   between tiles.  Should the prototype use f32 or bf16 for inter-tile
   activation packets to reduce fabric bandwidth?  Effect on q8/q4 tolerance
   must be measured before committing.

4. **Barrier implementation.** The `att1_fabric_t` barrier is a single-
   generation all-or-nothing primitive.  Can this be implemented with a
   PCIe atomic compare-and-swap, or is a dedicated barrier register required?

5. **DMA vs MMIO command queue.** Should the host enqueue decode commands via
   MMIO writes to a BAR-mapped command FIFO, or via DMA descriptors?  MMIO is
   simpler for a two-tile prototype; DMA scales better to many tiles.

6. **Error isolation.** If one tile returns a hardware error, should the host
   runtime retry, evict the session, or halt the decode loop?  The current
   C11 runtime propagates the first non-OK status and stops.  Hardware may
   require a retry path.

7. **Counter read timing.** Should counter snapshots be read synchronously
   after each decode step (adds one PCIe read per step) or accumulated on-device
   and read at end-of-session?

---

### 8.16 Next Milestone Proposal

**Milestone 94: PCIe prototype interface layer**

Define the `att1_backend_ops` implementation for a PCIe/hardware tile:
- `backend_pcie_create()` — opens PCIe BAR, validates tile firmware version.
- `backend_pcie_matmul_f32()` — enqueues matmul command, reads result.
- Counter collection stub.
- Software-emulated endpoint (Option A) as the initial target so the
  interface can be validated before physical hardware is available.
- No C runtime change beyond adding `src/backend_pcie.c` and
  `include/att1_backend_pcie.h`.
- Validated by a new test binary `tests/test_backend_pcie.c` using the
  software-emulated endpoint.

---

## 9. Tensor-Level Placement Plan (Milestone 97)

This section covers the AIMU-architecture implications of M97's tensor-level
placement model.  The full placement schema, slicing policies, validation
rules, and future milestone split are specified in
[shard_metadata.md §13](shard_metadata.md).  This section focuses on what
tensor-level placement means for the AIMU fabric protocol, AIMU-local memory
layout, activation routing, and the prototype engineering path.

---

### 9.1 From Layer Ownership to Tensor Slice Ownership

The current AIMU prototype assigns each layer to a tile.  An AIMU tile
executes all ops for its assigned layers: QKV projection, attention, FFN,
RMSNorm.  This maps cleanly onto the software simulation because a tile is a
software thread and there is no physical memory separation.

At silicon scale, a single AIMU tile has a fixed local SRAM budget.  A large
model's projection matrices may not fit in one tile's SRAM.  The tensor-level
placement model allows a projection matrix to be **sliced** across multiple
AIMUs.  Each AIMU receives only the rows or columns it owns, executes a
partial matmul, and returns a partial result vector to the fabric.

The shift in model:

| Dimension | Layer-wise (current) | Tensor-level (M97+) |
|---|---|---|
| Unit of ownership | Layer | Tensor or tensor slice |
| Metadata record | One record per tensor (whole) | One record per tensor slice |
| AIMU local memory | Entire layer's weights | Slice rows/columns only |
| Fabric message | Activation per layer boundary | Activation to every tile holding a slice of the tensor |
| Reduction | Sum at logits only | Sum after every split-tensor matmul |
| KV locality | Follows layer assignment | Follows head assignment (head-wise split) |

---

### 9.2 AIMU Fabric Implications of Each Split Policy

The AIMU fabric (simulated in `simulator/sim_fabric_bus.c`) currently routes
activation vectors in layer-pipeline order.  Tensor-level splits require the
fabric to support the following additional patterns:

**Broadcast to slice owners:**  
When a tensor is row-split across K tiles, the full activation vector must be
broadcast to all K tiles before the partial matmul can begin.  The fabric
packet header must encode the target `tile_id` set (a bitmask) and the
activation payload.  Currently, the fabric routes point-to-point; broadcast
is a logical extension.

**Partial result collection:**  
After each partial matmul, each slice owner sends its partial result vector
back to the reduction aggregator (which may be a designated tile or the host).
The fabric must guarantee all K partial results arrive before the aggregator
performs the sum reduction.

**Head-local KV traffic:**  
With head-wise attention split, QKV ops and KV memory are co-located on the
same tile.  Cross-tile KV traffic is eliminated for the common case.  The
fabric's KV routing path (currently implicit in layer assignment) becomes
explicit in `routing_requirements` = `path_policy=1 (fixed)` on KV placement
records.

**Embedding lookup routing:**  
A vocab-split embedding table requires the fabric to route a token ID lookup
request to the tile holding that token's row.  This is a request-response
pattern, not a streaming activation pattern.  The AIMU command packet for a
lookup request differs from the activation delivery packet; M103 will specify
this.

**Logit concat routing:**  
A vocab-split lm_head requires every tile holding a vocab slice to send its
partial logit vector to the sampler.  The sampler (or a designated aggregation
tile) receives K partial logit vectors and concatenates them in `slice_start`
order.  This is the most fabric-intensive operation per decode step when
vocab-split is used.

---

### 9.3 AIMU Local Memory Layout for Tensor Slices

A tensor slice placement record specifies:

- `byte_offset`: where the slice data begins in AIMU local SRAM.
- `slice_axis`, `slice_start`, `slice_end`: which elements of the full tensor
  this slice covers.
- `dtype`, `quantization`: data format and per-row or per-group scale storage.

The AIMU memory allocator must:

1. Lay out weight slices at their specified `byte_offset` before the first
   decode step (static layout — weights are immutable).
2. Reserve a separate region for the activation scratchpad (partial matmul
   inputs and outputs).
3. Reserve KV cache at the layer-and-head range owned by this tile, per the
   KV placement records.

For q8 row-split weights, scale vectors must be included in the AIMU local
SRAM alongside the quantized weight data.  Scale storage is proportional to
the number of rows owned.

For q4 group-split weights, scale and zero-point vectors are stored per group,
and the slice boundary must align to group size (typically 32 or 64 elements)
as required by the §13.5 validator rule 6.

---

### 9.4 Activation Routing Protocol Changes

The current activation routing protocol (M93) uses fixed layer-pipeline order:

```
host → tile_0 (layers 0…k-1) → tile_1 (layers k…2k-1) → … → host (logits)
```

Tensor-level placement changes the routing graph into a **directed acyclic
graph** (DAG) per decode step.  Each node in the DAG is a tensor-level
operation on a tile.  Edges carry the activation or partial-result vectors.

The DAG topology is determined by the placement records at model-load time.
It does not change between decode steps (for a fixed context length).  This
makes the routing protocol statically schedulable.

Key protocol changes required for tensor-level placement:

1. **Broadcast edges**: The fabric must support sending the same payload to
   multiple tiles simultaneously, or the host must unicast to each slice owner
   sequentially.  Unicast is correct but slower; broadcast is required at scale.

2. **Barrier tokens**: After a split-tensor matmul, all K partial results must
   arrive before the reduction step.  The `att1_fabric_barrier_wait()`
   primitive (M88) must accept a count parameter (`wait for K responses`) rather
   than a fixed single response.

3. **Reduction aggregation point**: The `routing_requirements` field on split
   tensor records must identify the reduction aggregator tile.  This may be a
   dedicated reduction tile, the host, or a round-robin assignment.  For the
   M97 spec, the aggregator is the host for all reductions.

4. **Delivery ordering for concat**: Logit slices from vocab-split lm_head must
   arrive at the sampler in `slice_start` order.  The fabric or the sampler
   must impose this ordering.

---

### 9.5 Trace Determinism Under Tensor-Level Placement

The trace subsystem (`att1_trace_t`, `src/trace.c`) records per-token and
per-step fabric counters.  Tensor-level placement changes the expected counter
values:

| Counter | Layer-wise baseline | Tensor-level (head-wise, K tiles per layer) |
|---|---|---|
| `prefill_fabric_packets` | `n_layers × 1` | `n_layers × K` (broadcast to K tiles per layer) |
| `decode_fabric_packets` | `n_layers × 1` | `n_layers × K × 2` (broadcast + partial collect) |
| `kv_reads` | proportional to context | same (local on owning tile) |
| `logits_bytes_produced` | `vocab_size × 4` | `vocab_size × 4` (same — concat at host) |

Any existing smoke test that hard-codes expected fabric packet counts will
diverge when tensor-level placement is enabled.  The M97 non-goal list
explicitly excludes changing inference behavior or trace values; these changes
will occur when opt-in tensor-level execution is enabled (M102).

**Determinism requirement:** For a given placement plan and a given input
token sequence, the trace counters must be identical across runs.  Partial
results are summed in slice_start order (not arrival order) to ensure the same
floating-point accumulation order.

---

### 9.6 Estimator Integration (AIMU Perspective)

The M96 tile memory and bandwidth estimator (`att1-size`) uses an even-split
heuristic.  From the AIMU perspective, the important refinement is:

- **Memory**: An AIMU tile must fit (a) all weight slices assigned to it,
  (b) activation scratchpad (max activation tensor × 2), (c) KV cache for
  owned heads at max context.  The even-split heuristic may overestimate
  for small-slice tiles and underestimate for tiles that hold large norm
  weights replicated across all tiles.

- **Bandwidth**: The bottleneck is the activation broadcast volume (full
  `d_model` vector broadcast to K tiles) plus the partial-result return
  traffic.  With K=2 head-wise split, fabric traffic doubles vs. layer-wise.
  With vocab-split lm_head, the final logit broadcast becomes the dominant
  traffic source at large `vocab_size`.

A per-tile capacity table (M100) will make these distinctions explicit.

---

### 9.7 Prototype Engineering Path

The M97 tensor-level placement spec prepares the AIMU prototype (Phase 2 PCIe,
M93–M94) for the following next steps:

| Milestone | AIMU prototype impact |
|---|---|
| M98 | Placement report schema; no hardware interface change |
| M99 | Validator Python tool; validates shard_metadata records against §13.5 rules; no hardware change |
| M100 | `att1-size --placement` option; reads a placement JSON; produces per-tile capacity and bandwidth table |
| M101 | Advisory placement proposal tool; generates placement JSON from model config + policy; no inference change |
| M102 | Opt-in CPU execution; validates that CPU inference with a tensor-level plan produces the same output as layer-wise inference within established tolerances; no PCIe hardware change |
| M103 | AIMU command packet spec; defines the binary layout of the per-tile activation delivery, KV position, reduction barrier, and counter read packets sent over PCIe |

Phase 3 silicon will consume the M103 command packet spec directly.  The
Phase 2 PCIe software-emulated endpoint (M94) provides a validation target
for M103 before hardware is available.

---

### 9.8 Non-Goals for M97 (AIMU scope)

- No change to the PCIe BAR-mapped command FIFO protocol (M94).
- No change to `src/backend_pcie.c` or `include/att1_backend_pcie.h`.
- No change to the fabric simulator `simulator/sim_fabric_bus.c`.
- No new AIMU tile instruction set entries.
- No change to the KV-MMU paged-memory protocol (`att1_kv_mmu.h`).
- No power or thermal modeling.
- No physical layer (electrical/optical interconnect) specification.

---

## 10. AIMU/PCIe Command Packet Requirements (M103)

See [`docs/aimu_pcie_command_requirements.md`](aimu_pcie_command_requirements.md)
for the full M103 specification.

### 10.1 Summary

M103 defines the binary command packet format and control protocol for the
PCIe/AIMU prototype.  It extends the M97 tensor-level placement model into
a concrete host↔AIMU communication layer.

Key deliverables:

| M103 section | Deliverable |
|---|---|
| §2 | Host/AIMU control plane: device discovery, tile enumeration, feature flags |
| §3 | 64-byte command packet format (checksum, fence, dtype, op_params) |
| §4 | 15 required command types (LOAD_TENSOR_TILE through QUERY_COUNTERS) |
| §5 | Data movement model: weight-load DMA once, logit-only PCIe in hot path |
| §6 | Memory model: tensor/KV/staging/command/trace regions, alignment rules |
| §7 | Execution model: command lifecycle, fences, barriers, error propagation |
| §8 | Counter/trace format: 64-byte snapshot mirroring `att1_trace_t` fields |
| §9 | Relationship to M98–M102 placement reports |
| §10 | Prototype options A–D (software simulator → FPGA → ASIC) |
| §12 | Future milestone split: M104–M110 |

### 10.2 Command Types (Summary)

| `command_type` | Value | Purpose |
|---|---|---|
| `LOAD_TENSOR_TILE` | 0x01 | DMA tensor slab from host to tile local memory |
| `VALIDATE_TENSOR_TILE` | 0x02 | Verify CRC32 of resident tensor |
| `EXEC_MATMUL` | 0x10 | Local matrix-multiply (weight × activation) |
| `EXEC_RMSNORM` | 0x11 | Local RMSNorm |
| `EXEC_ROPE` | 0x12 | Local Rotary Position Embedding |
| `EXEC_ATTENTION` | 0x13 | Local causal self-attention |
| `EXEC_FFN` | 0x14 | Local FFN/SwiGLU |
| `KV_APPEND` | 0x20 | Append K/V pair to tile KV cache |
| `KV_READ` | 0x21 | Read K/V slice from tile KV cache |
| `FABRIC_SEND` | 0x30 | Route activation to peer tile(s) |
| `FABRIC_REDUCE` | 0x31 | Accumulate partial results across tiles |
| `TRACE_SNAPSHOT` | 0x40 | Write counter record to trace memory |
| `TILE_BARRIER` | 0x41 | Cross-tile synchronization barrier |
| `RESET_TILE` | 0x50 | Clear session state, tensors, counters |
| `QUERY_COUNTERS` | 0x51 | DMA performance counters to host |

### 10.3 PCIe Hot-Path Rule

During steady-state decode, PCIe carries **only**:
1. Next-token ID (4 bytes, host → AIMU staging buffer)
2. Control command packets (64 bytes each, host → command ring)
3. Output logit vector (`vocab_size × 4` bytes, AIMU → host)

Weight matrices, KV updates, and inter-tile activations never cross the
PCIe bus during inference.  This is the hardware realization of the ATT-1
near-memory execution principle.

### 10.4 Relationship to the ATT-1 Simulator

| ATT-1 simulator concept | PCIe/AIMU M103 command |
|---|---|
| `att1_shard_t` initialization | `LOAD_TENSOR_TILE` per shard |
| `att1_transformer_block_forward` | `EXEC_RMSNORM` → `EXEC_ATTENTION` → `EXEC_FFN` per tile |
| `att1_kv_cache_append` | `KV_APPEND` |
| `att1_fabric_send` | `FABRIC_SEND` |
| `att1_fabric_barrier` | `TILE_BARRIER` |
| `att1_trace_snapshot` | `TRACE_SNAPSHOT` |
| `check_placement_*_smoke()` tests | Placement-report-to-command-plan mapper (M109) |
