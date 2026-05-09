# AIMU Fabric Routing Requirements (M114)

This document specifies the fabric routing requirements for ATT-1 AIMU tile
interconnect during multi-tile inference.  It covers what data moves across
the fabric, how routes are described, the packet types involved, how
placement-driven topology changes the routing graph, reduction semantics,
observable counters, and validation rules.

**Scope:** Documentation and specification only.  No C runtime, Python tool,
Makefile, `.att1` format, or inference behavior changes.

**Related documents:**

- [fabric.md](fabric.md) — Phase 1 fabric simulator design and API
- [aimu_architecture.md](aimu_architecture.md) — AIMU architecture; §9 covers
  tensor-level placement fabric implications; §12 (this milestone summary)
- [aimu_pcie_command_requirements.md](aimu_pcie_command_requirements.md) —
  M103 command protocol; §4.10 `FABRIC_SEND` / §4.11 `FABRIC_REDUCE`
- [aimu_register_map.md](aimu_register_map.md) — M104 BAR0 register map;
  §6 fabric/interconnect registers
- [tensor_placement_report.md](tensor_placement_report.md) — M98–M113
  placement and command-plan specifications

---

## 1. Fabric Role

The inference fabric carries the lightweight, token-step data that must cross
tile boundaries during a decode iteration.  Everything else stays local.

### 1.1 What Moves Across the Fabric

| Payload class | Description | Frequency |
|---|---|---|
| Activation vector | d_model-wide float32 vector at layer input or output; the primary inter-tile signal | Every layer boundary that spans a tile split |
| Partial logit vector | Partial lm_head result from a vocab-split or head-split tile; summed or concatenated at the aggregator | Once per decode step (lm_head only) |
| Reduction payload | Partial matmul result from a row-split or column-split projection; reduced before crossing to the next op | After each split-tensor EXEC_MATMUL |
| KV transfer payload | Key or value slice moving to a KV-owning tile; only required under KV-redistribute policies | Only on explicit KV migration (not in hot path) |
| Trace and control events | `TILE_BARRIER` sync tokens, `TRACE_SNAPSHOT` triggers, `CONTROL_ACK` response packets | Per layer boundary or on demand |

### 1.2 What Stays Local

| Data class | Reason |
|---|---|
| Resident weight tiles | Loaded once via `LOAD_TENSOR_TILE`; never moved during inference |
| Local KV / session memory | Each tile owns and appends to its own KV cache; no cross-tile KV traffic in the normal case |
| Activation scratchpad | Intermediate activation buffers (post-RMSNorm, post-RoPE, pre-FFN) allocated from tile-local staging |
| Local op output buffers | Results of `EXEC_RMSNORM`, `EXEC_ROPE`, `EXEC_ATTENTION` before the next fabric send |

The fabric carries `d_model × sizeof(f32)` bytes per activation crossing — on
the order of 512 B to 16 KiB depending on model size.  It does not carry weight
data after initial load.

---

## 2. Routing Model

A fabric route describes one directed data flow from a source tile to one or
more destination tiles.  The routing model is fully specified at model-load
time from the placement report; it does not change between decode steps (for a
fixed context length).

### 2.1 Route Descriptor Fields

| Field | Type | Description |
|---|---|---|
| `route_id` | uint16 | Unique per decode-step route identifier; assigned by the host from the command plan |
| `source_tile` | uint8 | Originating tile index (0-based) |
| `dest_tile_mask` | uint32 | Bitmask of destination tile indices (bits 0–31); broadcast = all-ones minus source bit |
| `packet_type` | uint8 | See §3 packet/route types |
| `payload_type` | uint8 | Payload content class: `0=activation`, `1=partial_logit`, `2=reduction`, `3=kv_payload`, `4=trace_event`, `5=control_ack` |
| `payload_bytes` | uint32 | Expected payload size in bytes; used for pre-flight bandwidth check |
| `fence_id` | uint16 | Dependency fence the sending tile waits for before issuing this route (0 = none) |
| `completion_fence_id` | uint16 | Fence signaled by the destination tile upon receive; 0 = no fence |
| `reduction_id` | uint16 | Reduction group tag; all routes sharing a `reduction_id` contribute to the same `FABRIC_REDUCE` accumulation; 0 = no reduction |
| `trace_id` | uint16 | Trace correlation token written to the tile's trace buffer when this route completes; 0 = not traced |

### 2.2 Ordering Requirements

- Routes within the same decode step that share a `fence_id` chain execute in
  declared fence order.
- Routes with the same `reduction_id` must all complete before the reduction
  result is consumed.  The fabric or the destination tile guarantees this via
  the `FABRIC_REDUCE` command's participant bitmask.
- Activation routes across a layer boundary must complete before the
  downstream `EXEC_RMSNORM` or `EXEC_MATMUL` on the receiving tile.

### 2.3 Barrier Requirements

- After the final layer boundary in a decode step, all tiles must reach a
  `TILE_BARRIER` before the host collects the logit vector.
- The `TILE_BARRIER` phase tag must match across all participating tiles.
- No new activation routes are issued for decode step N+1 until the
  step-N `TILE_BARRIER` completes.

---

## 3. Packet and Route Types

The following packet types cover all required fabric traffic patterns.

| Code | Name | Direction | Description |
|---|---|---|---|
| `0x01` | `ACTIVATION_SEND` | one-to-one | Send a complete activation vector from one tile to exactly one downstream tile |
| `0x02` | `ACTIVATION_BROADCAST` | one-to-many | Send the same activation vector to all tiles in `dest_tile_mask`; used before split-tensor matmul |
| `0x03` | `PARTIAL_REDUCE` | many-to-one | Each source tile sends its partial result vector to the designated reduction aggregator; aggregator accumulates with `FABRIC_REDUCE` |
| `0x04` | `LOGITS_REDUCE` | many-to-one | Specialisation of `PARTIAL_REDUCE` for vocab-split lm_head: each tile sends a contiguous slice of the logit vector; aggregator concatenates in `slice_start` order |
| `0x05` | `KV_TRANSFER` | one-to-one | Carry a KV page from a source tile to a new owner tile; only issued on explicit KV migration, not during normal inference |
| `0x06` | `TILE_BARRIER` | all-to-all | Synchronisation token; no payload; all tiles in `dest_tile_mask` must arrive before any proceeds |
| `0x07` | `TRACE_EVENT` | tile-to-host | Optional; carry a 64-byte trace snapshot record across the fabric to the host trace buffer; not required in Phase 2 |
| `0x08` | `CONTROL_ACK` | one-to-one | Acknowledgment packet from a receiving tile to the sender confirming receipt; used when `completion_fence_id` is set |

### 3.1 Packet Type Capabilities by Phase

| Packet type | Phase 1 (software sim) | Phase 2 (PCIe prototype) | Phase 3 (ASIC) |
|---|---|---|---|
| `ACTIVATION_SEND` | yes | required | required |
| `ACTIVATION_BROADCAST` | yes (via `att1_fabric_broadcast`) | required | required |
| `PARTIAL_REDUCE` | yes (via `att1_fabric_broadcast` + host accumulate) | required | required |
| `LOGITS_REDUCE` | yes | required | required |
| `KV_TRANSFER` | deferred | deferred | required |
| `TILE_BARRIER` | yes (via `att1_fabric_barrier`) | required | required |
| `TRACE_EVENT` | via `att1_trace_snapshot` | optional | optional |
| `CONTROL_ACK` | implicit (queue semantics) | optional | optional |

---

## 4. Placement-Driven Routing

The placement report (M98/M100) fully determines the fabric routing graph for
each model.  The routing graph is a directed acyclic graph (DAG) over tile
operations.  Edges in the DAG correspond to fabric routes.

### 4.1 Layer-Wise Placement (Current Baseline)

Each tile owns a contiguous range of transformer layers.  Activation flows in
pipeline order:

```
host → tile_0 → tile_1 → … → tile_K-1 → host (logits)
```

Fabric traffic per decode step: one `ACTIVATION_SEND` per tile boundary, one
`TILE_BARRIER` at end.  KV is tile-local.  No reductions needed until lm_head.

### 4.2 Tensor-Wise Placement (M97+ Extension)

A single tensor is assigned to exactly one tile.  Dependencies between tensors
on different tiles require fabric routes.  The routing graph is determined by
the dependency order in the command plan (M109).

### 4.3 Row-Split Matmul

A projection matrix is split row-wise across K tiles.  Each tile holds a
contiguous block of rows.  Routing requirements:

1. **Broadcast activation**: the full input activation vector (`d_model × f32`)
   is sent from the dispatching tile (or host) to all K slice owners via
   `ACTIVATION_BROADCAST`.
2. **Partial result collection**: each of the K slice owners sends its partial
   output vector to the reduction aggregator tile via `PARTIAL_REDUCE`.
3. **Reduction**: the aggregator performs element-wise sum (`reduce_type=0`).

`payload_bytes` for the broadcast = `d_model × 4`.
`payload_bytes` for each partial result = `(slice_end - slice_start) × 4`.

### 4.4 Column-Split Matmul

A projection matrix is split column-wise across K tiles.  Each tile holds a
contiguous block of columns.  Routing requirements:

1. **Activation slice delivery**: only the columns owned by tile N are applied
   to the corresponding input slice.  The input activation is sliced
   (`slice_start` to `slice_end` in dimension 0) and sent via `ACTIVATION_SEND`
   to each tile.
2. **Result concatenation**: each tile produces its slice of the output vector
   and sends it to the aggregator via `LOGITS_REDUCE` (`reduce_type=1`).

### 4.5 Head-Wise Attention Split

QKV projection, attention, and KV cache are co-located on the same tile per
head group.  Routing requirements:

1. **Activation broadcast**: the token embedding / residual stream is broadcast
   to all head-owning tiles before the QKV projection step.
2. **Attention output collection**: each head-owning tile sends its attention
   output slice to the next-stage aggregator (or to the FFN tile) via
   `PARTIAL_REDUCE` with `reduce_type=1` (concatenate).
3. **KV locality**: no KV traffic crosses the fabric for the normal attention
   step.  KV entries stay on the tile that owns those heads.

### 4.6 lm_head / Vocab Split

The final projection from hidden state to vocabulary logits is split across
K tiles by vocabulary slice.  Routing requirements:

1. **Activation broadcast**: the final hidden state (`d_model × f32`) is
   broadcast to all vocab-slice tiles via `ACTIVATION_BROADCAST`.
2. **Logit slice delivery**: each tile sends its partial logit vector
   (`vocab_slice_size × f32`) to the sampler aggregator tile via `LOGITS_REDUCE`.
3. **Concat ordering**: the aggregator concatenates slices in ascending
   `slice_start` order to reconstruct the full logit vector.  Arrival order
   from the fabric is not assumed to match `slice_start` order; the
   aggregator must buffer and re-order.

### 4.7 Embedding Lookup Routing (tok_embeddings split)

A vocab-split embedding table requires a request-response routing pattern:

1. **Lookup request**: the host (or designated dispatcher tile) issues a
   `CONTROL_ACK`-class packet carrying the token ID to the tile holding that
   token's embedding row.
2. **Response**: the owning tile sends the embedding vector back via
   `ACTIVATION_SEND`.

This is not an activation broadcast; it is a unicast request-response.
The command plan (M109) encodes this as a `FABRIC_SEND` with
`op_param_0 = 0 (activation)` and `dest_tile_mask` set to the single owner.

### 4.8 Replicated Norms

RMSNorm weight tensors marked `replication_policy=read_replicate` are loaded
onto every tile at model load time.  No fabric traffic is required for norm
operations.  `routing_requirement = none`.

### 4.9 KV Ownership and Routing Policy

Under the default placement policy, KV entries for layer L, head H are always
on the tile that owns the QKV tensors for layer L, head H.  No KV routing is
required during the hot path.

The `KV_TRANSFER` packet type (§3, code `0x05`) is reserved for explicit KV
migration (e.g., session handoff between tiles on session eviction).  This is
not required in Phase 2.

---

## 5. Reduction Semantics

### 5.1 Sum Reduction (`reduce_type=0`)

All participating tiles send their partial result vectors to the aggregator.
The aggregator accumulates them element-wise in ascending tile-ID order (not
arrival order) to guarantee deterministic floating-point results.  The
accumulated result is stored in the aggregator's local output buffer.

**Tolerance**: f32 exact (identity order); q8 abs ≤ 0.15 per logit (matches
M88/M92 CUDA baseline); q4 abs ≤ 0.35 per logit.

### 5.2 Concat Reduction (`reduce_type=1`)

All participating tiles send their partial vectors.  The aggregator
concatenates them in ascending `slice_start` order.  The total output size
equals the sum of all partial sizes.

Concat reductions are used for:
- Head-wise attention output reassembly.
- Vocab-split lm_head logit vector reconstruction.

### 5.3 Max / Top-K (Future)

Max reduction and top-K selection are not required for Phase 2.  They are
reserved for future hardware-accelerated sampling pipelines.  `reduce_type=2`
and `reduce_type=3` are reserved.

### 5.4 No-Reduction / Pass-Through

When `reduction_id = 0` and `dest_tile_mask` has exactly one bit set, the
route is a simple unicast pass-through with no reduction.  The receiving tile
does not need to wait for other partial results.

### 5.5 Deterministic Ordering

Reduction order is always ascending tile-ID regardless of network arrival order.
The `FABRIC_REDUCE` command's `op_param_1` (participant bitmask) defines the
complete participant set.  The aggregator buffers partial results until all
participants have sent, then accumulates in tile-ID order.

This rule guarantees bitwise-identical reduction outputs across runs with the
same placement plan and input token, regardless of fabric scheduling
non-determinism.

### 5.6 Tolerance Implications

| Path | Reduction type | Expected max delta per element |
|---|---|---|
| f32 activations, f32 weights | sum (layer-wise) | 0.0 (exact) |
| q8 weights × f32 activations | sum (row-split) | abs ≤ 0.15 |
| q4 weights × f32 activations | sum (row-split) | abs ≤ 0.35 |
| Head-wise concat | concat | 0.0 (no arithmetic, only reorder) |
| Vocab-split logit concat | concat | 0.0 (no arithmetic, only reorder) |

---

## 6. Fabric Counters

Each tile and the fabric interconnect must expose the following counters.
These are observable via `QUERY_COUNTERS` (M103 §4.15) and
`att1_aimu_trace_snapshot_all` (M108).

### 6.1 Per-Tile Packet Counters

| Counter | Description | ATT-1 simulator equivalent |
|---|---|---|
| `packets_sent` | Total fabric packets issued by this tile | `att1_trace_t.fabric_packets_sent` |
| `packets_received` | Total fabric packets received by this tile | `att1_trace_t.fabric_packets_received` |
| `payload_bytes_sent` | Total payload bytes in sent packets | `att1_aimu_trace_fabric_counters.payload_bytes_sent` |
| `payload_bytes_received` | Total payload bytes in received packets | `att1_aimu_trace_fabric_counters.payload_bytes_received` |

### 6.2 Per-Packet-Type Counters

| Counter | Description |
|---|---|
| `packets_by_type[8]` | Per-packet-type packet count (indexed by §3 code) |
| `bytes_by_type[8]` | Per-packet-type payload bytes |

### 6.3 Reduction Counters

| Counter | Description |
|---|---|
| `reductions_started` | Number of `FABRIC_REDUCE` commands submitted |
| `reductions_completed` | Number of `FABRIC_REDUCE` commands that accumulated all participants |
| `reduction_partial_arrivals` | Total partial result packets received across all reductions |

### 6.4 Barrier Counters

| Counter | Description |
|---|---|
| `barriers_started` | `TILE_BARRIER` commands submitted |
| `barriers_completed` | `TILE_BARRIER` commands where all participants arrived |
| `barrier_arrivals` | Individual tile arrivals counted across all barriers |

### 6.5 Congestion and Stall Counters

| Counter | Description | M104 register |
|---|---|---|
| `fabric_congestion_count` | Queue-full or backpressure events on the fabric send path | `FABRIC_CONGESTION_COUNTER` |
| `stall_fabric_cycles` | Tile-local cycles stalled waiting for a fabric send or receive | `stall_fabric_cycles` in §8.8 extended stall block |

### 6.6 Error Counters

| Counter | Description | Error code |
|---|---|---|
| `route_failures` | Packets not delivered due to routing errors | `ERR_FABRIC_SEND (0x0B)` |
| `dropped_packets` | Packets dropped due to destination queue full | (queue-full error) |
| `ordering_violations` | Partial results arriving after a reduction has already been finalised | (diagnostic only) |

### 6.7 Counter Snapshot Integration

Fabric counters are included in the 64-byte `att1_aimu_trace_snapshot`
counter record (M108 §8.1) under the `att1_aimu_trace_fabric_counters`
sub-struct.  The fabric placeholder fields introduced in M108 will be
populated by the M115 fabric route schema implementation.

---

## 7. Validation Rules

The following rules govern a valid fabric routing configuration as produced
by the M109 command plan mapper or a future M116 validator.

| Rule | Category | Description |
|---|---|---|
| F1 | Tile range | `source_tile` and all bits in `dest_tile_mask` must be valid tile indices within `[0, tile_count)` |
| F2 | Route target | `dest_tile_mask` must not be zero (a route must have at least one destination) |
| F3 | Payload size | `payload_bytes` must be nonzero for all non-barrier packet types |
| F4 | Known packet type | `packet_type` must be one of the defined codes in §3 (0x01–0x08) |
| F5 | Reduction explicit | Any route with `reduction_id != 0` must correspond to a `FABRIC_REDUCE` command in the same command plan that references the same `reduction_id` |
| F6 | No missing route | Every non-local tensor dependency (where `routing_requirement != "local"`) must have at least one route in the plan whose `dest_tile_mask` includes the consuming tile |
| F7 | No unresolvable cycle | The routing DAG must be acyclic within a single decode step.  Routes that form a cycle without an explicit barrier are invalid |
| F8 | Barrier ordering | `TILE_BARRIER` routes must appear after all activation and partial-result routes in the same decode step |
| F9 | Fence ordering | If a route has `fence_id != 0`, the producing command for that fence must appear earlier in the command plan |
| F10 | Bandwidth compatibility | The sum of `payload_bytes` across all routes in one decode step must not exceed `fabric_gib_sec × step_interval_sec × 1024³` (from the M100 placement report `bandwidth_status` estimate) |
| F11 | Concat ordering | All `LOGITS_REDUCE` routes with the same `reduction_id` must have non-overlapping `slice_start`/`slice_end` ranges that together cover exactly `[0, vocab_size)` |
| F12 | Reduction completeness | A `FABRIC_REDUCE` command's `op_param_1` participant bitmask must match exactly the set of `source_tile` values for all routes sharing that `reduction_id` |

---

## 8. Relationship to the Existing Simulator

### 8.1 Phase 1 Fabric Simulator (`src/fabric.c`, `simulator/sim_fabric_bus.c`)

The ATT-1 Phase 1 fabric simulator (`att1_fabric_t`) implements:

- Fixed-capacity inbound queues per tile.
- `att1_fabric_send()` — point-to-point unicast.
- `att1_fabric_broadcast()` — excludes sender; preflight checks all destinations.
- `att1_fabric_barrier_wait()` — single-generation all-or-nothing.
- Counters: `packets_sent`, `packets_received`, `broadcast_packets`,
  `payload_bytes_sent`, `queue_full_errors`, `invalid_packets`,
  `empty_receives`, `barrier_arrivals`, `barrier_completions`.

These map to M114 route types as follows:

| Phase 1 API call | M114 route type |
|---|---|
| `att1_fabric_send(src, dst, payload)` | `ACTIVATION_SEND` |
| `att1_fabric_broadcast(src, NULL, payload)` | `ACTIVATION_BROADCAST` |
| `att1_fabric_broadcast` + host accumulate | `PARTIAL_REDUCE` |
| `att1_fabric_barrier_wait()` | `TILE_BARRIER` |

The Phase 1 simulator does not distinguish `PARTIAL_REDUCE` from
`ACTIVATION_BROADCAST` at the API level; both use `att1_fabric_broadcast`.
The M115 fabric route schema will add an explicit route-type field to
distinguish them.

### 8.2 Cluster Inference Fabric Counters

The cluster inference path (`src/cluster_infer.c`) maintains per-tile fabric
counters that are read back via `att1_cluster_infer_get_tile_stats()`.  These
map to the M114 counter set:

| Cluster inference counter | M114 counter |
|---|---|
| `fabric_packets_sent` | `packets_sent` |
| `fabric_packets_received` | `packets_received` |
| `fabric_bytes_sent` | `payload_bytes_sent` |
| `barrier_completions` | `barriers_completed` |

The `packets_by_type` and `bytes_by_type` per-packet-type counters are new
in M114 and will be tracked in M115.

### 8.3 M109 Command Plans and Fabric Routes

The M109 command-plan mapper (`compiler/map_placement_to_commands.py`) emits
`FABRIC_SEND` and `TILE_BARRIER` commands but does not yet emit explicit route
descriptors.  M115 will extend the command plan JSON schema with a `routes`
array using the §2.1 route descriptor fields.

### 8.4 M115/M116 Implementation Path

| Milestone | Deliverable | What it adds |
|---|---|---|
| M115 | Fabric route schema and report | JSON schema for route descriptors; `compiler/map_placement_to_fabric_routes.py` emits a `routes` array; smoke test validates round-trip |
| M116 | Fabric route validator | `compiler/validate_fabric_routes.py` checks F1–F12; exit 0/1/2; smoke test covers all 12 rules |
| M117 | Command-plan-to-fabric-route mapper | Extends M109 command plan output with a `routes` section; driven by M98 `routing_requirement` fields |
| M118 | Fabric bandwidth/latency simulator | Extends Phase 1 `att1_fabric_t` with per-packet-type counters and a per-route bandwidth model; validates rule F10 at runtime |
| M119 | Placement + command + fabric replay integration | Extends M113 replay tool to emit per-route statistics in the JSON report |
| M120 | Phase 3 prototype go/no-go review | Cross-document review: M103 command model, M104 register map, M114 routing spec, M115–M119 tools, gap analysis for ASIC tape-out |

---

## 9. Non-Goals

The following are explicitly outside M114 scope:

- **No hardware fabric implementation.** This document defines the routing
  model; it does not specify RTL, interconnect topology, or signal integrity.
- **No PCIe driver changes.** Fabric routing is tile-to-tile; the host control
  plane is not changed.
- **No runtime scheduling changes.** The ATT-1 C11 runtime, shard planner, and
  cluster inference path are not modified.
- **No new inference kernels.** No new matmul, norm, or attention ops are
  added.
- **No patent claim language.** This document contains only engineering
  specifications.
- **No mobile, Android, Vulkan, or OpenCL targets.**
- **No `.att1` format changes.** The binary model artifact format is unchanged.
- **No CUDA kernel additions.** The CUDA backend is not modified.
- **No new C, Python, or Makefile changes** are made as part of M114.

---

## 10. Future Milestone Split

| Milestone | Title | Scope |
|---|---|---|
| M115 | Fabric route schema and report | Define JSON route descriptor schema (§2.1 fields); `compiler/map_placement_to_fabric_routes.py` reads M109 command plan and emits a `routes` array; `--report-json` option; smoke test: valid plan round-trips, route count matches command count, broadcast route detected |
| M116 | Fabric route validator | `compiler/validate_fabric_routes.py` checks all 12 rules (F1–F12); exit 0=pass, 1=violations, 2=parse error; fixture set covers each rule; smoke test: 12 cases |
| M117 | Command-plan-to-fabric-route mapper | Extend M109 `map_placement_to_commands.py` output with a `routes` section populated from `routing_requirement` and `reduction_behavior` placement fields; F6 check integrated |
| M118 | Fabric bandwidth/latency simulator | Extend Phase 1 `att1_fabric_t` with per-packet-type counters and per-route byte accounting; add rule F10 bandwidth check to `att1_fabric_send`/`att1_fabric_broadcast`; no inference behavior change |
| M119 | Placement + command + fabric replay integration | Extend M113 `tools/att1-aimu-replay.c` to parse and replay `routes` section; emit per-route packet/byte/latency statistics in `--report-json` |
| M120 | Phase 3 prototype go/no-go review | Cross-document design review covering M103–M119; gap analysis for Phase 3 ASIC; recommended silicon architecture changes; non-goals for Phase 2 close-out |
