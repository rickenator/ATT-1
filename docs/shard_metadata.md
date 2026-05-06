# ATT-1 Shard Metadata Design

Milestone 34 defines the future `.att1` shard metadata section.  The current
binary format reserves `shard_metadata_offset` and `shard_metadata_size` in
the header (both zero when absent); this document specifies what will occupy
that section when AIMU tensor-tile ownership is encoded directly in the model
artifact.

No binary format change is made in this milestone.  The spec here is the
authoritative design that future format version 2 will implement.

---

## 1. Purpose and Scope

The current `version=1` format stores a flat tensor table.  The runtime
derives shard assignments at load time via `att1_shard_plan_build()`, which
partitions layers across available tiles using simple round-robin policy.
Dynamic runtime sharding is sufficient for simulation but insufficient for
AIMU hardware deployment, where:

- Each AIMU must know its tensor assignments before the host runtime
  communicates placement.
- Routing requirements and allowed ops must be verifiable on-device without
  host coordination.
- Replication policy must be expressed statically so that multiple AIMUs can
  hold read-replicated tensors without a central placement oracle.
- Dependency edges must be stored so that the fabric can schedule packet
  routing without additional host round-trips.

The shard metadata section solves all of these by embedding AIMU placement,
routing, and op-permission data directly in the `.att1` artifact.

---

## 2. Field Reference

Each shard metadata record describes one tensor as seen from the AIMU fabric.
Fields are little-endian unless noted.

### 2.1 `tensor_id` — `uint32`

Unique identifier for this tensor within the model artifact.  Assigned by the
converter in stable order (same order as the tensor descriptor table).
The C loader validates that `tensor_id` values are contiguous starting at 0
and match the tensor descriptor table length.

### 2.2 `tile_id` — `uint32`

Target AIMU tile assignment.  `0xFFFFFFFF` means unassigned (runtime must
assign before inference starts).  The loader accepts unassigned tensors; the
runtime rejects any tile with unresolved assignments at inference start.

For broadcast/replicated tensors, `tile_id` encodes the primary owner.
Secondary replicas are described via `replication_policy`.

### 2.3 `byte_offset` — `uint64`

Byte offset of this tensor's data within the AIMU's local tensor memory
region.  For `version=1`-compatible files this mirrors the tensor descriptor
`offset` field.  For AIMU silicon, this is the physical address within the
AIMU's local SRAM or HBM bank.

The loader validates: `byte_offset + nbytes <= declared AIMU memory capacity`
when capacity is nonzero in the device descriptor.

### 2.4 `shape[4]` — `uint64[4]`

Tensor dimensions, matching the corresponding tensor descriptor.  Up to 4D;
unused trailing dimensions are 1.  The loader cross-validates shape against
the tensor descriptor and rejects mismatches.

### 2.5 `dtype` — `uint32`

Data type code:

| Code | Meaning                                     |
|------|---------------------------------------------|
| `1`  | float32 (current, required)                 |
| `2`  | q8 — int8 data + per-row float32 scales     |
| `3`  | reserved                                    |

The loader rejects any value not in this table.  dtype `2` in shard metadata
is valid even in `version=1` files if the tensor data section encodes q8
correctly (int8 rows followed by float32 scale table); this allows AIMU
on-device q8 without a runtime re-quantization step.

### 2.6 `quantization` — `uint32`

Quantization metadata selector:

| Code | Meaning                                                    |
|------|------------------------------------------------------------|
| `0`  | None (float32 tensor; no scale table)                      |
| `1`  | Per-row symmetric q8 (scale table follows tensor int8 data)|
| `2`  | Per-channel q8 (future; not yet defined)                   |

When `quantization == 1`, the tensor data region contains:
- `rows × cols` bytes of int8 quantized values (row-major)
- `rows × 4` bytes of float32 per-row scales immediately following

The loader validates that `nbytes == rows*cols + rows*4` when `quantization=1`.

### 2.7 `owner_aimu` — `uint32`

Address of the owning AIMU in the fabric address space.  Format is
implementation-defined at the PCIe/silicon level; for the software prototype,
`owner_aimu` is the zero-based tile index.  `0xFFFFFFFF` is broadcast
(read-replicated).

### 2.8 `replication_policy` — `uint32`

Describes how this tensor is replicated across AIMUs:

| Code | Policy                                                      |
|------|-------------------------------------------------------------|
| `0`  | None — single owner, no replication                         |
| `1`  | Read-replicate — tensor is copied to all tiles at load time |
| `2`  | Write-broadcast — writes are forwarded to all replicas      |
| `3`  | Reserved                                                    |

`replication_policy=1` is the expected value for small shared tensors such as
embedding tables in multi-tile models.

### 2.9 `dependency_graph` — `uint32[8]`

Bitmask array of tensor IDs this tensor depends on.  Bit `i` in element `j`
is set if tensor `(j*32 + i)` must be fully resolved before this tensor's ops
can execute.  A zero array means no dependencies (input tensor).

The fabric scheduler uses dependency edges to sequence activation packet
routing.  The runtime validates that the dependency graph is acyclic at load
time.

For the current software prototype, dependencies are derived dynamically from
layer order; this field is zero-filled in converter output until AIMU silicon
requires static scheduling.

### 2.10 `allowed_ops` — `uint32`

Bitmask of operations permitted on this tensor in AIMU local execution:

| Bit | Operation                        |
|-----|----------------------------------|
| 0   | `matmul_f32`                     |
| 1   | `matmul_q8xf32`                  |
| 2   | `rmsnorm_f32`                    |
| 3   | `ffn_swiglu_f32`                 |
| 4   | `rope_f32`                       |
| 5   | `softmax_f32`                    |
| 6–30| Reserved                         |
| 31  | Allow all (wildcard)             |

The AIMU firmware rejects any activation packet requesting an op not set in
this field.  The software prototype does not enforce this check yet; it is
reserved for Phase 2 device enforcement.

### 2.11 `routing_requirements` — `uint32`

Fabric QoS class and path constraints:

| Bits  | Field            | Description                              |
|-------|------------------|------------------------------------------|
| [1:0] | `qos_class`      | 0=best-effort, 1=latency, 2=throughput   |
| [3:2] | `path_policy`    | 0=any, 1=fixed, 2=round-robin            |
| [7:4] | `priority`       | 0–15, higher = higher priority           |
| [31:8]| Reserved         | Must be zero                             |

Zero is a valid default (best-effort, any path, priority 0).

### 2.12 `reduction_behavior` — `uint32`

Specifies how output values are reduced across partial results from multiple
AIMUs:

| Code | Reduction                              |
|------|----------------------------------------|
| `0`  | None — single tile produces full result|
| `1`  | Sum — partial results are summed       |
| `2`  | Max — element-wise maximum             |
| `3`  | Concatenate — results are concatenated |
| `4`  | Reserved                               |

`reduction_behavior=1` is the expected value for sharded projection matrices
where each tile produces a partial logit vector that must be summed across
tiles.

### 2.13 `checksum` — `uint64`

Truncated CRC-64/ECMA-182 of the tensor's data bytes.  Computed over the raw
content of the tensor data region (including inline scale tables if present).
The loader verifies this on load when `checksum != 0`.  `checksum == 0` means
no verification (allowed for development artifacts).

The converter must compute and embed a nonzero checksum for all production
artifacts.  The current stub emitter may write zero during development.

---

## 3. Record Layout

Each shard metadata record is fixed-size.  Planned wire layout (all
little-endian):

```text
tensor_id              uint32        4 bytes
tile_id                uint32        4 bytes
byte_offset            uint64        8 bytes
shape[4]               uint64[4]    32 bytes
dtype                  uint32        4 bytes
quantization           uint32        4 bytes
owner_aimu             uint32        4 bytes
replication_policy     uint32        4 bytes
dependency_graph[8]    uint32[8]    32 bytes
allowed_ops            uint32        4 bytes
routing_requirements   uint32        4 bytes
reduction_behavior     uint32        4 bytes
_reserved              uint32        4 bytes  (must be zero)
checksum               uint64        8 bytes
────────────────────────────────────────────
Total                               120 bytes per record
```

Records are stored contiguously in the shard metadata section.  Section size
must equal `tensor_count × 120`.  The loader rejects any other size.

---

## 4. Versioning Rules

- Shard metadata is absent in `version=1` files (`shard_metadata_offset=0`,
  `shard_metadata_size=0`).  The loader must not require it.
- `version=2` will be the first version where shard metadata may be present.
  The loader must accept `version=2` files and parse the shard metadata section.
- A `version=2` file with zero `shard_metadata_size` is valid (metadata
  omitted, runtime assigns shards dynamically).
- New fields within the record may be added by expanding the record size, with
  a corresponding `version` bump.  Older loaders must reject records of unknown
  size.
- The `_reserved` field exists to maintain 8-byte alignment of `checksum` and
  must be written as zero.  Future fields may consume it before requiring a
  version bump.

---

## 5. Hostile-Input Validation Rules

The loader must enforce all of the following when shard metadata is present:

1. `shard_metadata_size == tensor_count × 120` — reject any other size.
2. `tensor_id` values are 0, 1, …, `tensor_count − 1` in order — reject any
   gap or duplicate.
3. `tile_id != 0xFFFFFFFF` for all records before inference starts (unassigned
   tensors are rejected at inference time, not load time).
4. `byte_offset + nbytes` does not overflow `uint64`.
5. `shape` matches the corresponding tensor descriptor exactly.
6. `dtype` is a known code (1 or 2); any other value is rejected.
7. `quantization` is a known code (0 or 1); any other value is rejected.
8. `replication_policy` is a known code (0–2); any other value is rejected.
9. `reduction_behavior` is a known code (0–3); any other value is rejected.
10. `_reserved` is zero — reject nonzero values to prevent future field
    aliasing.
11. `dependency_graph` is acyclic — the loader verifies topological sort on the
    dependency graph across all records.
12. `checksum` is verified when nonzero; a mismatch is a hard reject.
13. The shard metadata section must not overlap the tensor descriptor section
    or the tensor data section.

---

## 6. Current Cluster Sharding → Future AIMU Ownership

The current software prototype computes shard assignments at load time:

```
att1_shard_plan_build(model, n_tiles, &plan)
```

This assigns each layer to a tile by round-robin over layer index.  The result
is stored in `att1_shard_plan_t` and used to route activation packets through
the fabric simulator.

The mapping from runtime sharding to future AIMU shard metadata is:

| Runtime (`att1_shard_plan_t`) | Shard metadata field       |
|-------------------------------|----------------------------|
| `plan.tile_for_layer[l]`      | `tile_id`                  |
| Tensor descriptor `offset`    | `byte_offset`              |
| Tensor descriptor `shape`     | `shape`                    |
| Backend vtable name           | `allowed_ops` bitmask      |
| Layer-level sequencing        | `dependency_graph`         |
| Logit reduction in cluster    | `reduction_behavior=1`     |
| Embedding (shared)            | `replication_policy=1`     |

When shard metadata is present in the artifact, `att1_shard_plan_build()` will
be replaced by `att1_shard_plan_from_metadata()` which directly reads
`tile_id` assignments without recomputing them.  Both functions will produce
`att1_shard_plan_t` so that the rest of the runtime is unchanged.

---

## 7. CUDA Validation vs Future PCIe/AIMU Silicon

The CUDA backend (`src/backend_cuda.c`) serves a specific validation role that
differs from the AIMU silicon target:

| Dimension              | CUDA backend                         | Future AIMU silicon              |
|------------------------|--------------------------------------|----------------------------------|
| Purpose                | Numerical validation of math ops     | Production tile execution        |
| Tensor placement       | Managed by CUDA runtime (device mem) | AIMU-owned local SRAM/HBM        |
| Activation routing     | Host-driven via cuBLAS calls         | Fabric packet routing            |
| KV-cache               | Host-allocated, copied per step      | AIMU-managed KV-MMU              |
| Shard metadata         | Not used (host controls placement)   | Required for tile programming    |
| `allowed_ops`          | Not enforced                         | Enforced by AIMU firmware        |
| `routing_requirements` | Not applicable                       | Enforced by fabric arbiter       |
| `checksum`             | Not verified at inference time       | Verified at AIMU load time       |
| Phase                  | Phase 1 validation                   | Phase 3 silicon target           |

The CUDA backend demonstrates that the mathematical primitives behind the
backend vtable are correct.  It does not simulate fabric routing, AIMU
ownership, KV-MMU paging, or shard placement policy.  These are simulated by
the cluster inference path using the CPU backends and the fabric/KV-MMU
simulators.

The PCIe prototype (Phase 2) bridges the two: AIMU tile memory is a PCIe
endpoint memory region, commands flow through host-to-device command queues,
and the shard metadata embedded in the `.att1` artifact drives tile
programming.  The CUDA backend is retired at Phase 2.

---

## 8. Converter Responsibilities

When the converter (`compiler/convert_llama_to_att1.py`) emits shard metadata,
it must:

1. Assign `tensor_id` values in tensor descriptor table order.
2. Default `tile_id` to `0xFFFFFFFF` (unassigned) for single-tile stubs;
   assign explicit tile IDs when a target tile count is provided via
   `--n-tiles`.
3. Set `replication_policy=1` for `tok_embeddings.weight` and
   `output.weight` in multi-tile models (both are read-shared).
4. Set `reduction_behavior=1` for `output.weight` in sharded models (logits
   are summed across tiles).
5. Set `allowed_ops` based on the tensor's role:
   - Norm weights: `rmsnorm_f32` (bit 2)
   - Projection weights: `matmul_f32 | matmul_q8xf32` (bits 0–1)
   - FFN weights: `ffn_swiglu_f32` (bit 3)
   - Embedding: `matmul_f32 | matmul_q8xf32` (bits 0–1)
6. Leave `dependency_graph` zero-filled until static scheduling is required.
7. Compute and embed a nonzero `checksum` for production artifacts; zero is
   acceptable for development stubs.
8. Write `_reserved = 0`.

The stub emitter in Milestone 32 omits shard metadata entirely
(`shard_metadata_offset=0`).  Shard metadata emission will be added when
`version=2` is defined.

---

## Related Documents

- [model_format.md](model_format.md) — current `.att1` binary format (version 1)
- [aimu_architecture.md](aimu_architecture.md) — AIMU concept and prototype mapping
- [cluster_inference.md](cluster_inference.md) — cluster sharding and activation routing
- [real_model_conversion.md](real_model_conversion.md) — converter plan and tensor naming
- [backend.md](backend.md) — backend vtable and operator set
- [fabric.md](fabric.md) — packet fabric simulator
- [kv_mmu.md](kv_mmu.md) — KV-MMU paged session memory
