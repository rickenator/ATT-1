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

## 7. Implementation Status (Milestone 35)

**C parser skeleton is in place** (`src/shard_meta.c`, `include/att1_shard_meta.h`).

The loader (`src/model_loader.c`) now:
- Detects and bounds-checks an optional shard metadata section.
- Verifies the section does not overlap the descriptor table or data section.
- Calls `att1_shard_meta_parse()` when `shard_metadata_size != 0`.
- Populates `att1_model.shard_meta` on success; returns `ATT1_ERR_BAD_FORMAT`
  on any violation.
- Frees parsed records in `att1_model_free()`.

Validation rules implemented:

| Rule | Status |
|------|--------|
| 1. section_size == tensor_count × 120 | ✅ |
| 2. tensor_id sequence 0..N-1 in order | ✅ |
| 3. tile_id unassigned accepted at load time | ✅ |
| 4. byte_offset + nbytes overflow check | deferred |
| 5. shape cross-validation vs tensor descriptor | ✅ |
| 6. dtype in {1, 2} | ✅ |
| 7. quantization in {0, 1} | ✅ |
| 8. replication_policy in {0, 1, 2} | ✅ |
| 9. reduction_behavior in {0, 1, 2, 3} | ✅ |
| 10. _reserved == 0 | ✅ |
| 11. dependency_graph acyclic | deferred |
| 12. checksum verification when nonzero | deferred (CRC-64 TBD) |
| 13. section non-overlap with desc/data | ✅ |

**Not yet implemented**: AIMU placement enforcement, checksum verification,
acyclic graph check, byte_offset capacity validation.  These are deferred to
the milestone that introduces AIMU silicon targets.

Test coverage added in `tests/test_shard_meta.c` (8 test cases):
- No metadata section
- Valid tiny metadata section (1 record)
- Bad bounds (shard_size extends past EOF)
- Truncated records (section_size not a multiple of 120)
- Bad record count (section_size implies wrong number of records)
- Invalid tensor_id order
- Invalid dtype code
- Non-zero _reserved field

---

## 8. Fixture and Tooling (Milestone 36)

**Offline fixture generator**: `compiler/make_shard_meta_fixture.py`

Generates `models/shard_meta/model.att1` — a 14 876-byte deterministic `.att1`
artifact with the same 21-tensor tiny model as `models/dummy/model.att1`, plus
a 2 520-byte shard metadata section (21 × 120 bytes).

Tile assignment scheme in the fixture:
- `n_tiles=1`; all tensors: `tile_id=0`, `owner_aimu=0`
- `replication_policy=none`, `reduction_behavior=none`
- `dtype=f32`, `quantization=none`
- `dependency_graph`: zero-filled (static scheduling not yet required)
- `allowed_ops=0` (enforcement deferred), `checksum=0` (CRC-64 deferred)
- `_reserved=0`

Regenerate with:
```
python3 compiler/make_shard_meta_fixture.py
```

The fixture is **checked into the repository** so `make test` needs no Python.

**`att1-inspect` shard summary** (`tools/att1-inspect.c`):
When shard metadata is present, `att1-inspect` now prints after the tensor
list:
```
shard_meta: 21 records
  shard[0] tile=0 aimu=0 dtype=f32 repl=none reduce=none  tok_embeddings.weight
  shard[1] tile=0 aimu=0 dtype=f32 repl=none reduce=none  layers.0.attention_norm.weight
  ...
```

**C test coverage** (`tests/test_shard_meta_fixture.c`, 4 test cases):
1. Fixture loads with `shard_meta.count == 21`; all records validated
   (tensor_id in order, tile_id=0, dtype=f32, shape cross-checked)
2. `att1-inspect` output contains `"shard_meta: 21 records"` and per-record
   fields
3. No-metadata dummy model still loads with `shard_meta.count == 0`
4. `att1-inspect` on dummy model produces no `"shard_meta:"` line

---

## 9. Reporting and Trace Integration (Milestone 37)

**New API**: `att1_shard_meta_summarize()` (`include/att1_shard_meta.h`, `src/shard_meta.c`)

Populates `att1_shard_meta_summary`:

| Field | Description |
|-------|-------------|
| `count` | Total records (equals tensor count) |
| `assigned` | Records with `tile_id != ATT1_SHARD_TILE_UNASSIGNED` |
| `unassigned` | Records with `tile_id == ATT1_SHARD_TILE_UNASSIGNED` |
| `unique_tiles` | Distinct assigned `tile_id` values |
| `unique_aimus` | Distinct `owner_aimu` values |
| `dtype_f32` | Records with `dtype == ATT1_SHARD_DTYPE_F32` |
| `dtype_q8` | Records with `dtype == ATT1_SHARD_DTYPE_Q8` |

All fields are zero when `meta->count == 0` (no metadata section present).

**`att1-inspect` output** (extended):
When shard metadata is present the summary header now precedes the per-record list:
```
shard_meta: 21 records
shard_meta_tiles=1
shard_meta_aimus=1
shard_meta_assigned=21
shard_meta_unassigned=0
shard_meta_dtype_f32=21
shard_meta_dtype_q8=0
  shard[0] tile=0 aimu=0 dtype=f32 repl=none reduce=none  tok_embeddings.weight
  ...
```

**`att1-bench` output** (extended):
Printed once per run, before mode/backend/tokens lines:
```
shard_meta=present          ← or shard_meta=absent
shard_meta_count=21
shard_meta_assigned=21
shard_meta_unassigned=0
shard_meta_tiles=1
shard_meta_aimus=1
shard_meta_dtype_f32=21
shard_meta_dtype_q8=0
mode=single
...
```

No runtime placement enforcement is added. Inference behavior is unchanged.

**C test coverage** (`tests/test_shard_meta_report.c`, 4 test cases):
1. `att1-bench` on dummy model — output contains `shard_meta=absent`; existing counters still present
2. `att1-bench` on shard fixture — output contains full summary with correct values
3. `att1-inspect` on shard fixture — summary header and per-record detail both present
4. `att1-inspect` on dummy model — no `shard_meta` line in output

---

## 10. Consistency Validation (Milestone 38)

**New API**: `att1_shard_meta_validate()` (`include/att1_shard_meta.h`, `src/shard_meta.c`)

Validates parsed shard metadata against the model configuration and tensor
descriptors.  Returns `ATT1_OK` always (violations are reported, not fatal).
Callers must call `att1_shard_meta_validation_free()` when done.

Populates `att1_shard_meta_validation` with zero or more `att1_shard_meta_violation` records:

| Field | Description |
|-------|-------------|
| `tensor_id` | Index of the offending record |
| `field` | Name of the offending field (e.g. `"tile_id"`, `"dtype"`) |
| `description` | Human-readable detail string |

**Checks performed** (per record):

| Check | Field | Condition |
|-------|-------|-----------|
| tile_id in range | `tile_id` | assigned tile_id >= n_tiles |
| AIMU in range | `owner_aimu` | owner_aimu >= n_tiles |
| dtype consistent | `dtype` | shard dtype doesn't match tensor descriptor dtype |
| byte_offset matches descriptor | `byte_offset` | shard byte_offset != tensor descriptor offset |

Checks already enforced at parse time (`ATT1_ERR_BAD_FORMAT`) are not repeated:
tensor_id sequence, shape cross-validation, enum range validation, `_reserved == 0`.

**`att1-inspect` output** (when violations exist):
```
shard_meta: 1 records
shard_meta_tiles=0
...
shard_meta_violations: 1
  violation[0] tensor_id=0 field=tile_id: tile_id 5 out of range (n_tiles=1)
```
When no violations are found, the `shard_meta_violations:` line is omitted.

**C test coverage** (`tests/test_shard_meta_consistency.c`, 8 test cases):
1. Consistent fixture (21-tensor shard_meta fixture) — 0 violations
2. tile_id >= n_tiles — 1 violation on `tile_id`
3. owner_aimu >= n_tiles — 1 violation on `owner_aimu`
4. shard dtype=Q8 vs tensor dtype=F32 — 1 violation on `dtype`
5. byte_offset != tensor descriptor offset — 1 violation on `byte_offset`
6. Absent metadata (dummy model) — 0 violations
7. `att1-inspect` on violation model — output contains `shard_meta_violations: 1` and `field=tile_id`
8. `att1-inspect` on consistent fixture — no `shard_meta_violations` line

No inference or backend behavior changed.  No placement enforcement added.

---

## 11. Metadata-Driven Shard Plan Proposal (Milestone 39)

**New API**: `att1_meta_plan_build()`, `att1_meta_plan_compare()` (`include/att1_shard.h`, `src/shard.c`)

Derives a *proposed* layer-to-tile shard plan from optional shard metadata and
compares it against the runtime-generated `att1_shard_plan`.  The proposed
plan is **advisory only** — inference always uses the runtime plan.

### `att1_meta_plan` (proposed plan)

| Field | Description |
|-------|-------------|
| `entries` | One `att1_meta_plan_entry` per covered layer, sorted by `layer_id` |
| `count` | Number of layers with at least one assigned metadata record |
| `extra` | Metadata records whose tensor name does not map to any layer (e.g. embeddings) |
| `conflict` | Layers where two or more metadata records disagree on `tile_id` |

Layer identification: a tensor name of the form `"layers.N.anything"` maps to
layer N.  Tensors with `tile_id == ATT1_SHARD_TILE_UNASSIGNED` are skipped.
Absent metadata produces `count=0` with no allocation.

### `att1_meta_plan_diff` (comparison result)

| Field | Description |
|-------|-------------|
| `matching` | Layers where proposed tile == runtime tile |
| `mismatch` | Layers where proposed tile != runtime tile |
| `missing` | Layers present in runtime plan but absent from proposed |
| `extra` | Pass-through from `att1_meta_plan.extra` |
| `conflict` | Pass-through from `att1_meta_plan.conflict` |

### `att1-inspect` output (metadata present)

```
shard_meta_plan_entries=2
shard_meta_plan_extra=3
shard_meta_plan_conflict=0
shard_meta_plan_matching=2
shard_meta_plan_mismatch=0
shard_meta_plan_missing=0
```

Printed after the per-record shard list.  When no metadata is present the
block is omitted entirely.  The runtime plan comparison is skipped if
`att1_shard_plan_build` fails (e.g. unsupported `shard_count` value).

**C test coverage** (`tests/test_shard_meta_plan.c`, 5 test cases):
1. Absent metadata (dummy model) — `count=0`, `extra=0`, `conflict=0`
2. Consistent fixture (21 tensors, 2 layers, tile 0) — `matching=2`, all zeros elsewhere
3. Missing layer (layer 1 tensor is `UNASSIGNED`) — proposed `count=1`, `missing=1`
4. Conflicting tile ownership (two layer-0 tensors with different `tile_id`) — `conflict=1`
5. `att1-inspect` on fixture — output contains `shard_meta_plan_entries=2` and `shard_meta_plan_matching=2`

No inference or backend behavior changed.  No placement enforcement added.

---

## §12  Opt-in Metadata Shard Plan Execution (Milestone 40)

### Goal

Allow cluster inference to use the metadata-derived shard plan when explicitly
requested, while keeping the runtime-generated plan as the default.

### New API

```c
/* include/att1_shard.h */
typedef enum att1_shard_plan_mode {
    ATT1_SHARD_PLAN_RUNTIME  = 0,  /* default: runtime-generated (existing) */
    ATT1_SHARD_PLAN_METADATA = 1   /* opt-in: metadata-derived plan         */
} att1_shard_plan_mode;

att1_status_t att1_shard_plan_from_meta(const att1_meta_plan *proposed,
                                        uint32_t              n_layers,
                                        size_t                tile_count,
                                        att1_shard_plan      *out);
```

`att1_shard_plan_from_meta()` validates that:
- `proposed->count == n_layers` (every layer is covered).
- `proposed->conflict == 0` (no competing tile assignments within a layer).
- All `tile_id` values are `< tile_count`.
- The entries are sorted and contiguous (`layer_id` == entry index).
- Each tile's assigned layers form a single contiguous range.

On success it populates `out` (caller responsible for
`att1_shard_plan_free(out)` when done).

### Config Extension

```c
/* include/att1_cluster_infer.h */
typedef struct att1_cluster_infer_config {
    size_t tile_count;
    size_t fabric_queue_capacity;
    size_t fabric_max_payload_bytes;
    att1_shard_plan_mode shard_plan_mode; /* 0 = runtime (default) */
} att1_cluster_infer_config;
```

Existing struct literals `{N, 4, 0}` remain correct — the new zero field maps
to `ATT1_SHARD_PLAN_RUNTIME`.

### Cluster Inference Behaviour

`att1_cluster_infer_create()` branches on `config->shard_plan_mode`:

| `shard_plan_mode`          | Behaviour |
|----------------------------|-----------|
| `ATT1_SHARD_PLAN_RUNTIME`  | Existing `att1_shard_plan_build()` path (default). |
| `ATT1_SHARD_PLAN_METADATA` | Calls `att1_meta_plan_build()` then `att1_shard_plan_from_meta()`. Returns `ATT1_ERR_INVALID_ARG` if metadata absent, conflicting, incomplete, or assignment non-contiguous. **No silent fallback.** |

### CLI Flag (att1-bench)

```
--shard-plan runtime|metadata   (default: runtime)
```

`att1-bench` prints `shard_plan=runtime` or `shard_plan=metadata` in its
key=value output block (both `--mode single` and `--mode cluster`).

### Invariants

- Models without shard metadata remain valid; `--shard-plan runtime` is always
  available.
- `.att1` binary format is unchanged.
- Backend behaviour is unchanged.
- Fabric packet counters, trace counters, activation bytes, and logits bytes
  are preserved.

### Tests (8)

1. Default cluster create on dummy model succeeds (runtime plan).
2. Metadata plan on shard fixture (n_tiles=1, n_layers=2) succeeds; logits
   match single inference within 1e-6.
3. Metadata absent + `ATT1_SHARD_PLAN_METADATA` → `!= ATT1_OK`.
4. Conflicting tile assignment + `ATT1_SHARD_PLAN_METADATA` → `!= ATT1_OK`.
5. Missing layer in metadata + `ATT1_SHARD_PLAN_METADATA` → `!= ATT1_OK`.
6. No silent fallback: status `!= ATT1_OK` and `infer == NULL` on failure.
7. `att1-bench --mode cluster` output contains `shard_plan=runtime`.
8. `att1-bench --mode cluster --shard-plan metadata` output contains
   `shard_plan=metadata`.

---

## §13  Backend Matrix Validation for Shard Plans (Milestone 41)

### Goal

Extend the backend matrix regression harness to validate both runtime-generated
and metadata-derived shard plans across all supported cluster backends.

### Matrix Extension

The harness (`tests/test_backend_matrix.c`) gains 8 new entries (16 total):

| backend  | mode    | model        | tiles | shard_plan | CUDA required |
|----------|---------|--------------|-------|------------|---------------|
| cpu-f32  | cluster | shard_meta   | 1     | runtime    | no |
| cpu-q8   | cluster | shard_meta   | 1     | runtime    | no |
| cuda     | cluster | shard_meta   | 1     | runtime    | yes |
| cuda-q8  | cluster | shard_meta   | 1     | runtime    | yes |
| cpu-f32  | cluster | shard_meta   | 1     | metadata   | no |
| cpu-q8   | cluster | shard_meta   | 1     | metadata   | no |
| cuda     | cluster | shard_meta   | 1     | metadata   | yes |
| cuda-q8  | cluster | shard_meta   | 1     | metadata   | yes |

The existing 8 entries (single-mode dummy and cluster-mode dummy) are unchanged.

### Consistency Groups

Three consistency groups enforce `last_token` agreement across all passing
entries in each group:

| Group | Description | Entries |
|-------|-------------|---------|
| 0 | single mode, dummy model | 4 (2 skip on CPU-only) |
| 1 | cluster mode, dummy model, runtime plan, 2 tiles | 4 (2 skip) |
| 2 | cluster mode, shard_meta model, runtime + metadata, 1 tile | 8 (4 skip) |

Group 2 cross-validates that runtime and metadata shard plans produce identical
`last_token` on the same model and same tile layout.

### bench smoke additions

`tests/test_bench_smoke.c` now verifies `shard_plan=runtime` appears in both
`--mode single` and `--mode cluster` bench output.

### Invariants

- Default `make` and `make test` remain CUDA-free.
- `make CUDA=1` enables CUDA entries.
- No silent fallback: `--shard-plan metadata` must succeed or the bench exits
  non-zero; the matrix marks such entries as FAIL (not SKIP).
- `.att1` format unchanged.

---

## §14  Converter Stub with Shard Metadata (Milestone 42)

### Goal

Extend the converter stub path to optionally emit a deterministic `.att1`
artifact with a valid shard metadata section, then validate the artifact
through the full C toolchain and backend matrix.

### Converter Changes

`compiler/convert_llama_to_att1.py` gains two new flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--tiles N` | 1 | Set `n_tiles` in the model config and control layer→tile assignment |
| `--shard-meta` | off | Emit a shard metadata section |

When `--shard-meta` is specified, `build_att1_bytes()` appends a shard metadata
section and sets `shard_metadata_offset` / `shard_metadata_size` in the header.

Layer assignment algorithm (`--tiles N`):
- Uses the same ceiling-division as `att1_shard_plan_build()` in `src/shard.c`,
  ensuring the metadata plan produces the same layer→tile mapping as the
  runtime plan.
- Non-layer tensors (`tok_embeddings.weight`, `output_norm.weight`,
  `output.weight`) map to tile 0; they appear as `extra` in
  `att1_meta_plan_build()` and do not affect shard plan validation.

### New Fixture

`models/converted_stub_meta/model.att1` — generated from
`compiler/fixtures/tiny_llama_config.json` with `--tiles 2 --shard-meta`:

| Property | Value |
|----------|-------|
| vocab_size | 256 |
| n_layers | 2 |
| n_heads | 2 |
| d_model | 32 |
| d_ff | 64 |
| max_seq_len | 128 |
| n_tiles | 2 |
| tensor_count | 21 |
| shard_meta records | 21 |
| layer 0 tensors | tile 0 |
| layer 1 tensors | tile 1 |

Regenerate with:
```bash
python3 compiler/convert_llama_to_att1.py \
    --config compiler/fixtures/tiny_llama_config.json \
    --tiles 2 --shard-meta \
    --out models/converted_stub_meta/model.att1
```

The fixture is **checked into the repository** so `make test` needs no Python.

### Validation

```bash
# Inspect: shows config, 21 tensors, shard_meta: 21 records, plan_entries=2
./build/att1-inspect models/converted_stub_meta/model.att1

# Cluster inference — runtime plan (layer assignment computed from config)
./build/att1-bench \
    --model models/converted_stub_meta/model.att1 \
    --prompt hello --tokens 8 \
    --mode cluster --tiles 2 --shard-plan runtime --backend cpu-f32

# Cluster inference — metadata plan (layer assignment read from shard_meta)
./build/att1-bench \
    --model models/converted_stub_meta/model.att1 \
    --prompt hello --tokens 8 \
    --mode cluster --tiles 2 --shard-plan metadata --backend cpu-f32
```

Both bench invocations must exit 0 and produce the same `last_token`.

### Backend Matrix Extension (Group 3)

`tests/test_backend_matrix.c` gains 8 new entries (24 total) in consistency
group 3:

| backend  | mode    | model                | tiles | shard_plan | CUDA required |
|----------|---------|----------------------|-------|------------|---------------|
| cpu-f32  | cluster | converted_stub_meta  | 2     | runtime    | no |
| cpu-q8   | cluster | converted_stub_meta  | 2     | runtime    | no |
| cuda     | cluster | converted_stub_meta  | 2     | runtime    | yes |
| cuda-q8  | cluster | converted_stub_meta  | 2     | runtime    | yes |
| cpu-f32  | cluster | converted_stub_meta  | 2     | metadata   | no |
| cpu-q8   | cluster | converted_stub_meta  | 2     | metadata   | no |
| cuda     | cluster | converted_stub_meta  | 2     | metadata   | yes |
| cuda-q8  | cluster | converted_stub_meta  | 2     | metadata   | yes |

Group 3 cross-validates that runtime and metadata plans produce identical
`last_token` on the 2-tile converted stub.  On a CPU-only build: 12/24 passed,
12 skipped.

### Invariants

- Existing `models/dummy/model.att1` and `models/shard_meta/model.att1`
  fixtures are unchanged.
- Existing `models/converted_stub/model.att1` (no metadata) is unchanged.
- No `.att1` binary format change.
- No backend behavior change.
- `--shard-plan metadata` on the new fixture does not silently fall back to
  runtime.

---

## §15 Converter Shard Metadata Plan Report (Milestone 43)

### Goal

Provide a clear human-readable and/or JSON summary of the shard metadata plan
generated by the converter, without changing the `.att1` binary format or
inference behavior.

### New CLI flags

| Flag | Description |
|------|-------------|
| `--report` | Print human-readable shard plan report to stdout |
| `--report-json PATH` | Write JSON shard plan report to PATH |

Both flags are orthogonal to `--output`, `--tiles`, and `--shard-meta`.

### Report schema (`schema_version=1`)

```json
{
  "schema_version": 1,
  "source_arch": "llama",
  "config": { "vocab_size": ..., "n_layers": ..., ... },
  "tensor_count": 21,
  "shard_meta": "present" | "absent",
  "tile_count": 2,
  "aimu_count": 2,
  "dtype_f32_count": 21,
  "dtype_q8_count": 0,
  "quant_none_count": 21,
  "tensors": [ { "name": "...", "shape": [...], "dtype": "f32",
                  "quant": "none", "tile_id": 0, "bytes": 4096 }, ... ],
  "tiles": [ { "tile_id": 0, "aimu_id": 0, "tensor_count": 12,
                "layer_ids": [0], "layer_range": "0-0" }, ... ],
  "layers": [ { "layer_id": 0, "tile_id": 0 }, ... ],
  "validation": { "status": "ok" | "failed", "errors": [] }
}
```

### Implementation notes

- `build_shard_plan_report()` returns a pure Python dict; no `.att1` I/O.
- `format_report_text()` renders the dict as a human-readable block.
- Both functions live in `compiler/convert_llama_to_att1.py`; no new files.
- `tests/test_bench_smoke.c` adds `check_converter_report()` which shells out
  to `python3` and validates key fields. The check is skipped (not failed) when
  Python is absent, preserving the Python-free `make test` guarantee.
- `schema_version` is 1; increment if the report structure changes.

### Invariants

- No `.att1` binary format change.
- No backend behavior change.
- `make test` remains Python-free (converter report check skips when absent).
- `--report`/`--report-json` are additive; omitting both leaves output unchanged.

---

## Related Documents

- [model_format.md](model_format.md) — current `.att1` binary format (version 1)
---

## §16 Converter Executable Metadata Plan Validation (Milestone 44)

### Goal

Prove that a converter-generated `.att1` stub with shard metadata passes
end-to-end inspection and bench execution with both `--shard-plan runtime` and
`--shard-plan metadata`.

### New artefacts

| Path | Type | Purpose |
|------|------|---------|
| `tests/test_converter_validation.c` | C test | Inspect + bench consistency on checked-in fixture |
| `compiler/validate_converter_flow.sh` | Shell script | Full pipeline including Python generation (dev only) |

### `check_inspect()` — validated fields

```
n_tiles=2
tensor_count=21
shard_meta: 21 records
shard_meta_tiles=2
shard_meta_assigned=21
shard_meta_unassigned=0
shard_meta_dtype_f32=21
shard_meta_dtype_q8=0
tile=0 aimu=0   (at least one record)
tile=1 aimu=1   (at least one record)
```

### `check_bench_consistency()` — validated fields

| Field | runtime | metadata | Required |
|-------|---------|----------|----------|
| `shard_plan` | `runtime` | `metadata` | present + correct label |
| `shard_meta` | `present` | `present` | both |
| `last_token` | 255 | 255 | equal |
| `logits_bytes_produced` | 12288 | 12288 | equal |
| `fabric_packets_sent` | 36 | 36 | equal |

### Invariants

- No `.att1` binary format change.
- No backend behavior change.
- `make test` remains Python-free (test uses checked-in fixture only).
- The validation script is idempotent and self-cleaning.

---

## Related Documents

- [aimu_architecture.md](aimu_architecture.md) — AIMU concept and prototype mapping
- [cluster_inference.md](cluster_inference.md) — cluster sharding and activation routing
- [real_model_conversion.md](real_model_conversion.md) — converter plan and tensor naming
- [backend.md](backend.md) — backend vtable and operator set
- [fabric.md](fabric.md) — packet fabric simulator
- [kv_mmu.md](kv_mmu.md) — KV-MMU paged session memory
