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
