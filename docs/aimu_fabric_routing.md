# AIMU Fabric Routing Requirements (M114)

This document specifies the fabric routing requirements for ATT-1 AIMU tile
interconnect during multi-tile inference.  It covers what data moves across
the fabric, how routes are described, the packet types involved, how
placement-driven topology changes the routing graph, reduction semantics,
observable counters, and validation rules.

**Scope:** Documentation and specification only.  No C runtime, Python tool,
Makefile, `.att1` format, or inference behavior changes.

**Freeze status: FROZEN v1.0 (Milestone 160).**  See §16 for the frozen-fields
vs. reserved-fields policy, the queue-full/barrier/counter contract, and the
barrier-mechanism decision that Stage 2+ endpoint work must preserve.

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

The f32 reference fabric model carries `d_model × sizeof(f32)` bytes per
activation crossing — on the order of 512 B to 16 KiB depending on model size.
As of M174, the Phase 2 prototype planning path may encode inter-tile
activation payloads as bf16 when the M174 activation-precision report passes;
`payload_bytes` remains the encoded byte count for the selected precision.
The fabric does not carry weight data after initial load.

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

The ATT-1 Phase 1 fabric simulator (`att1_fabric`) implements:

- Fixed-capacity inbound queues per tile.
- `att1_fabric_send()` — point-to-point unicast.
- `att1_fabric_broadcast()` — excludes sender; preflight checks all destinations.
- `att1_fabric_barrier_arrive()` — single-generation all-or-nothing.
- Counters: `packets_sent`, `packets_received`, `broadcast_packets`,
  `payload_bytes_sent`, `payload_bytes_received`, `queue_full_errors`,
  `invalid_packets`, `empty_receives`, `barrier_arrivals`,
  `barrier_completions`.

These map to M114 route types as follows:

| Phase 1 API call | M114 route type |
|---|---|
| `att1_fabric_send(src, dst, payload)` | `ACTIVATION_SEND` |
| `att1_fabric_broadcast(src, NULL, payload)` | `ACTIVATION_BROADCAST` |
| `att1_fabric_broadcast` + host accumulate | `PARTIAL_REDUCE` |
| `att1_fabric_barrier_arrive()` | `TILE_BARRIER` |

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
| M118 | Fabric bandwidth/latency simulator | Extends Phase 1 `att1_fabric_t` with per-packet-type counters and a per-route bandwidth model; validates rule F10 at runtime — **complete** |
| M119 | Placement + command + fabric replay integration | Extends M113 replay tool to emit per-route statistics in the JSON report |
| M120 | Phase 3 prototype go/no-go review | Cross-document review: M103 command model, M104 register map, M114 routing spec, M115–M119 tools, gap analysis for ASIC tape-out |

---

## 11. Fabric Route Report Schema (M115)

This section defines the canonical schema for AIMU fabric route reports.  A
fabric route report is a structured, serializable record that describes the
complete set of inter-tile routes required for a single decode step, derived
from a tensor placement report (M98/M100) and a command plan (M109).

Route reports are **advisory** in M115.  They describe planned fabric traffic
without changing runtime execution.  Future milestones consume this schema:

| Milestone | Role of route report |
|---|---|
| M116 | Validator input — checks route records against validation rules F1–F12 |
| M117 | Command-plan mapper emits route report from `routing_requirement` fields |
| M118 | Bandwidth simulator consumes `payload_bytes` estimates to validate rule F10 |
| M119 | Replay tool produces a route report from observed `FABRIC_SEND`/`FABRIC_REDUCE` completions |

The M115 Python tool `compiler/map_placement_to_fabric_routes.py` (defined in
§11.9) reads a M109 command plan JSON and emits a fabric route report JSON
conforming to this schema.

---

### 11.1 Report Header

The header identifies the source artifacts, model, and planning targets.

#### Key-value text format

```
route_report_version          = 1
source_placement_report       = <path or empty>
source_command_plan           = <path or empty>
model_name                    = <string>
model_id                      = <string>
session_id                    = <string>
tile_count                    = <integer ≥ 1>
route_count                   = <integer ≥ 0>
packet_count_estimate         = <integer ≥ 0>
payload_bytes_estimate        = <integer ≥ 0>
fabric_policy                 = layer_wise | tensor_wise | row_split | col_split | head_wise | vocab_split | mixed
status                        = pass | warn | fail
report_timestamp              = <ISO-8601 UTC string>
```

#### JSON format

```json
{
  "route_report_version": 1,
  "source_placement_report": "<path or null>",
  "source_command_plan": "<path or null>",
  "model_name": "<string>",
  "model_id": "<string>",
  "session_id": "<string>",
  "tile_count": 2,
  "route_count": 3,
  "packet_count_estimate": 3,
  "payload_bytes_estimate": 768,
  "fabric_policy": "layer_wise",
  "status": "pass",
  "report_timestamp": "2026-05-08T00:00:00Z"
}
```

**Field notes:**

- `route_report_version`: Schema version; currently `1`.  Must be incremented
  when any required field is added or renamed.
- `source_placement_report`: Path to the M98/M100 placement report JSON used
  as input; `null` if not available.
- `source_command_plan`: Path to the M109 command plan JSON used as input;
  `null` if not available.
- `fabric_policy`: Describes the dominant placement policy that drives the
  routing graph.  May be `mixed` when multiple policies coexist.
- `packet_count_estimate`: Sum of `packet_count_estimate` across all route
  records; one value per decode step.
- `payload_bytes_estimate`: Sum of `payload_bytes` across all route records.
- `status`: `pass` if no validation failures, `warn` if warnings only,
  `fail` if at least one validation failure.

---

### 11.2 Route Records

Each entry in the `routes` array describes one directed fabric data flow for
the decode step.

#### JSON format (single route record)

```json
{
  "route_id": 1,
  "route_type": "ACTIVATION_SEND",
  "source_tile": 0,
  "destination_tiles": [1],
  "source_tensor": "layers.0.attention.wo.weight",
  "source_command_id": 8,
  "destination_tensor": null,
  "destination_command_id": null,
  "payload_type": "activation",
  "payload_bytes": 256,
  "packet_count_estimate": 1,
  "dependency_fence": 8,
  "reduction_id": 0,
  "reduction_behavior": "none",
  "ordering_policy": "ordered",
  "trace_id": 0,
  "route_status": "ok"
}
```

#### Route record field reference

| Field | Type | Required | Description |
|---|---|---|---|
| `route_id` | uint16 | yes | Unique route identifier within this report; monotonically increasing from 1 |
| `route_type` | string | yes | One of the eight types in §3: `ACTIVATION_SEND`, `ACTIVATION_BROADCAST`, `PARTIAL_REDUCE`, `LOGITS_REDUCE`, `KV_TRANSFER`, `TILE_BARRIER`, `TRACE_EVENT`, `CONTROL_ACK` |
| `source_tile` | int | yes | Originating tile index (0-based) |
| `destination_tiles` | int[] | yes | List of destination tile indices; exactly one element for unicast; multiple for broadcast; empty list is invalid |
| `source_tensor` | string | no | Tensor name in the placement report that triggered this route; `null` for barrier and control routes |
| `source_command_id` | int | no | `command_id` from the M109 command plan of the producing command; `null` if not derived from a command plan |
| `destination_tensor` | string | no | Tensor name on the receiving tile, if applicable |
| `destination_command_id` | int | no | `command_id` of the consuming command on the destination tile |
| `payload_type` | string | yes | Content class: `activation`, `partial_logit`, `reduction`, `kv_payload`, `trace_event`, `control_ack`, `barrier_token` |
| `payload_bytes` | int | yes | Expected payload size in bytes; `0` is valid only for `TILE_BARRIER` and `CONTROL_ACK` types |
| `packet_count_estimate` | int | yes | Number of fabric packets this route generates per decode step; typically 1 for unicast; equal to `len(destination_tiles)` for broadcast |
| `dependency_fence` | int | yes | Fence ID the sending tile must wait for before issuing this route; `0` = no dependency |
| `reduction_id` | int | yes | Reduction group tag; all routes sharing a nonzero `reduction_id` contribute to the same `FABRIC_REDUCE`; `0` = no reduction |
| `reduction_behavior` | string | yes | One of: `none`, `sum`, `concat`, `max`, `topk`, `pass_through` |
| `ordering_policy` | string | yes | One of: `ordered` (fence-ordered relative to other routes), `unordered` (no ordering constraint), `barriered` (ordered via `TILE_BARRIER`) |
| `trace_id` | int | yes | Trace correlation token; `0` = not traced |
| `route_status` | string | yes | Validation result for this route: `ok`, `warn`, or `fail` |

#### `reduction_behavior` values

| Value | Description |
|---|---|
| `none` | No reduction; route is a pass-through or barrier |
| `sum` | Element-wise sum accumulated in tile-ID order (F10-safe, deterministic) |
| `concat` | Concatenate partial vectors in ascending `slice_start` order |
| `max` | Element-wise max (reserved; Phase 3) |
| `topk` | Top-K selection (reserved; Phase 3) |
| `pass_through` | Unicast with no accumulation; destination receives exactly what the source sent |

#### `ordering_policy` values

| Value | Description |
|---|---|
| `ordered` | Route execution is ordered by its `dependency_fence`; the sending tile must not issue this route until the fence is satisfied |
| `unordered` | Route has no ordering constraint relative to other routes in this decode step |
| `barriered` | Route is a `TILE_BARRIER` synchronisation token; no execution proceeds past this point until all participants arrive |

---

### 11.3 Per-Tile Route Summary

The `tiles` array provides one summary record per tile in the placement plan.

#### JSON format (single tile record)

```json
{
  "tile_id": 0,
  "outbound_packet_count": 2,
  "inbound_packet_count": 1,
  "outbound_payload_bytes": 512,
  "inbound_payload_bytes": 256,
  "reductions_started": 0,
  "reductions_completed": 0,
  "barriers_started": 1,
  "barriers_completed": 1,
  "route_failures": 0,
  "estimated_bandwidth_gib_sec": null
}
```

#### Tile summary field reference

| Field | Type | Description |
|---|---|---|
| `tile_id` | int | Tile index (0-based); must match a tile in the placement report |
| `outbound_packet_count` | int | Sum of `packet_count_estimate` for routes where `source_tile == tile_id` |
| `inbound_packet_count` | int | Sum of `packet_count_estimate` for routes where `tile_id` is in `destination_tiles` |
| `outbound_payload_bytes` | int | Sum of `payload_bytes` for outbound routes |
| `inbound_payload_bytes` | int | Sum of `payload_bytes` for inbound routes |
| `reductions_started` | int | Count of routes from this tile with a nonzero `reduction_id` |
| `reductions_completed` | int | Count of `FABRIC_REDUCE`-type routes where this tile is the aggregator |
| `barriers_started` | int | Count of `TILE_BARRIER` routes where this tile participates |
| `barriers_completed` | int | Same as `barriers_started` for the report (live execution may differ) |
| `route_failures` | int | Count of routes assigned `route_status = "fail"` for this tile |
| `estimated_bandwidth_gib_sec` | float or null | Estimated per-tile fabric bandwidth in GiB/s; computed as `outbound_payload_bytes / step_interval_sec / 2^30` when `target_tokens_per_sec` is known; `null` otherwise |

---

### 11.4 Warnings and Failures

The `warnings` and `failures` arrays collect diagnostic messages.  Each entry
follows the same structure:

```json
{
  "code": "INVALID_TILE_TARGET",
  "severity": "fail",
  "route_id": 2,
  "tile_id": 5,
  "message": "destination tile 5 is not in [0, tile_count=2)"
}
```

#### Diagnostic codes

| Code | Severity | Rule | Description |
|---|---|---|---|
| `INVALID_TILE_TARGET` | fail | F1 | `source_tile` or an element of `destination_tiles` is outside `[0, tile_count)` |
| `EMPTY_DESTINATION` | fail | F2 | `destination_tiles` is empty |
| `ZERO_PAYLOAD` | fail | F3 | `payload_bytes` is 0 for a non-barrier, non-control-ack route type |
| `UNKNOWN_PACKET_TYPE` | fail | F4 | `route_type` is not one of the eight defined types |
| `REDUCTION_NOT_EXPLICIT` | fail | F5 | `reduction_id` is nonzero but no `FABRIC_REDUCE`-type route in the plan references the same `reduction_id` |
| `MISSING_ROUTE` | fail | F6 | A tensor with `routing_requirement != "local"` has no route in the report whose `destination_tiles` covers the consuming tile |
| `CYCLIC_DEPENDENCY` | fail | F7 | The routing DAG contains a cycle (detected by DFS over `dependency_fence` chain) |
| `BARRIER_ORDERING` | fail | F8 | A `TILE_BARRIER` route appears before all activation/partial-result routes in the same decode step |
| `FENCE_ORDERING` | fail | F9 | A route's `dependency_fence` references a command that appears later in the command plan |
| `BANDWIDTH_OVERFLOW` | warn | F10 | Sum of `payload_bytes_estimate` exceeds the fabric bandwidth budget derived from `fabric_gib_sec` and `target_tokens_per_sec` |
| `CONCAT_INCOMPLETE` | fail | F11 | `LOGITS_REDUCE` routes sharing a `reduction_id` do not cover exactly `[0, vocab_size)` |
| `REDUCTION_PARTICIPANT_MISMATCH` | fail | F12 | The `FABRIC_REDUCE` command's participant bitmask does not match the set of `source_tile` values for all routes sharing that `reduction_id` |
| `UNKNOWN_COMMAND_REF` | warn | — | `source_command_id` or `destination_command_id` does not match any `command_id` in the source command plan |
| `MISSING_REDUCTION_BEHAVIOR` | warn | — | `reduction_id != 0` but `reduction_behavior == "none"` |
| `UNSUPPORTED_PACKET_TYPE` | warn | — | `route_type` is defined in §3 but marked as `deferred` for the current phase |

---

### 11.5 Human-Readable Report Format

When the route report tool is run without `--report-json`, it emits a
human-readable text summary.  The format consists of five tables.

#### Route summary table

```
Route Summary
=============
  route_report_version : 1
  model_id             : tiny_test_model
  session_id           : session_0
  tile_count           : 2
  route_count          : 3
  packet_count_estimate: 3
  payload_bytes_estimate: 768
  fabric_policy        : layer_wise
  status               : pass
```

#### Per-tile traffic table

```
Per-Tile Traffic (per decode step)
===================================
tile  out_pkts  in_pkts  out_bytes  in_bytes  reductions  barriers  failures
----  --------  -------  ---------  --------  ----------  --------  --------
   0         2        1        512       256           0         1         0
   1         1        2        256       512           0         1         0
```

#### Reduction summary table

```
Reductions
==========
  (none in this report)
```

When reductions are present:

```
Reductions
==========
reduction_id  behavior  participants  aggregator_tile
------------  --------  ------------  ---------------
           1  sum       0,1,2         3
           2  concat    0,1           0
```

#### Barriers and fences table

```
Barriers / Fences
=================
route_id  type          tiles         fence_dep  ordering_policy
--------  ------------  ------------  ---------  ---------------
       3  TILE_BARRIER  0,1           0          barriered
```

#### Warnings / failures table

```
Warnings / Failures
===================
  (none)
```

When issues are present:

```
Warnings / Failures
===================
severity  code                       route_id  message
--------  -------------------------  --------  -----------------------------------------------
fail      INVALID_TILE_TARGET               2  destination tile 5 is not in [0, tile_count=2)
warn      BANDWIDTH_OVERFLOW              all  payload 9.2 GiB/s exceeds budget 8.0 GiB/s
```

---

### 11.6 JSON Report Format

The JSON report has four top-level keys with stable names.

#### Top-level schema

```json
{
  "header": { ... },
  "routes": [ ... ],
  "tiles": [ ... ],
  "warnings": [ ... ],
  "failures": [ ... ]
}
```

| Key | Type | Description |
|---|---|---|
| `header` | object | §11.1 fields |
| `routes` | array | One §11.2 object per route; ordered by `route_id` ascending |
| `tiles` | array | One §11.3 object per tile; ordered by `tile_id` ascending |
| `warnings` | array | §11.4 entries with `severity = "warn"` |
| `failures` | array | §11.4 entries with `severity = "fail"` |

#### Stability rules

- All field names in `header`, `routes`, `tiles`, `warnings`, and `failures`
  are stable across tool versions and must not be renamed without a
  `route_report_version` increment.
- Optional fields (`source_tensor`, `source_command_id`, `destination_tensor`,
  `destination_command_id`, `estimated_bandwidth_gib_sec`) may be `null`.
- Unknown top-level keys must be ignored by consumers (forward compatibility).
- The `route_report_version` field must be checked before parsing any other
  field.  Consumers must fail gracefully on unsupported versions.

---

### 11.7 Relationship to Future Tools

| Tool | Milestone | How it uses the route report |
|---|---|---|
| `compiler/validate_fabric_routes.py` | M116 | Reads `routes` + `tiles` arrays; applies validation rules F1–F12; emits pass/warn/fail per route; exits 0/1/2 |
| Extended `map_placement_to_commands.py` | M117 | Emits a `routes` section from `routing_requirement` and `reduction_behavior` fields; adds M117 route records to the command plan JSON alongside the existing `commands` array |
| Fabric bandwidth/latency simulator | M118 | Reads `payload_bytes` and `packet_count_estimate`; simulates rule F10 bandwidth check; emits per-route latency estimates |
| `att1-aimu-replay` extended | M119 | Replays `routes` section; emits per-route observed `packet_count` and `payload_bytes` from M112 host harness completions; compares against estimates |
| Phase 3 go/no-go review | M120 | Aggregates route reports across all placement configurations to size the ASIC fabric ports and buffer depths |

---

### 11.8 Fixture: `compiler/fixtures/fabric_route_report_tiny.json`

A minimal reference fixture for a 2-tile, 2-layer, f32, layer-wise model.
Contains:

- 2 `ACTIVATION_SEND` routes (tile 0 → tile 1 at layer boundary, tile 1 →
  host sampler).
- 1 `TILE_BARRIER` route (both tiles, at end of decode step).
- 3 route records total; `packet_count_estimate = 3`;
  `payload_bytes_estimate = 512` (2 × 256 B activation + 0 B barrier).
- 0 warnings, 0 failures, status = `pass`.

This fixture is used by the M116 validator smoke test (valid-report-passes
case) and by the M115 tool round-trip smoke test.

See `compiler/fixtures/fabric_route_report_tiny.json` for the full JSON.

---

### 11.9 `compiler/map_placement_to_fabric_routes.py` (M115 Tool Spec)

**Purpose:** Read a M109 command plan JSON (which embeds a placement report
reference) and emit a fabric route report JSON.

**CLI:**

```
map_placement_to_fabric_routes.py --plan PATH [--report-json PATH]
                                              [--strict] [--tile-count N]
```

**Algorithm (per-command scan):**

1. Read `header.tile_count` and `header.fabric_policy` from the command plan.
2. For each `FABRIC_SEND` command in `commands`:
   - Derive `route_type` from `op_param_0`:
     - `op_param_0 = 0` → `ACTIVATION_SEND` (single dest) or
       `ACTIVATION_BROADCAST` (multiple dest via dest bitmask).
     - `op_param_0 = 1` → `PARTIAL_REDUCE` or `LOGITS_REDUCE`.
   - Set `payload_bytes = input_buf_bytes`.
   - Set `dependency_fence = fence_id`.
   - Set `reduction_id = 0`; set `reduction_behavior = none`.
3. For each `FABRIC_REDUCE` command:
   - Emit one `PARTIAL_REDUCE` or `LOGITS_REDUCE` route per participant tile.
   - Set `reduction_id` to a locally-assigned group tag (monotonic from 1).
   - Set `reduction_behavior = sum` or `concat` per `op_param_0`.
4. For each `TILE_BARRIER` command:
   - Emit one `TILE_BARRIER` route covering all participant tiles.
   - Set `payload_bytes = 0`, `payload_type = barrier_token`,
     `reduction_behavior = none`, `ordering_policy = barriered`.
5. Build per-tile summaries by iterating all routes.
6. Apply validation rules F1–F12; populate `warnings` and `failures`.
7. Compute `status`: `pass` if no failures; `warn` if warnings only; `fail`
   if any failures.

**Exit codes:**

- `0`: Report generated; status is `pass` or `warn`.
- `1`: Report generated; status is `fail` (at least one F-rule violation).
- `2`: Input parse error (missing required field, JSON syntax error, invalid
  command plan version).

**In `--strict` mode:** exit 1 if any warning is present (treats warnings as
failures).

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
| M116 | Fabric route validator | `compiler/validate_fabric_routes.py` checks all 12 rules (F1–F12); exit 0=pass, 1=violations, 2=parse error; fixture set covers each rule; smoke test: 12 cases — **complete** |
| M117 | Command-plan-to-fabric-route mapper | Extend M109 `map_placement_to_commands.py` output with a `routes` section populated from `routing_requirement` and `reduction_behavior` placement fields; F6 check integrated — **complete** |
| M118 | Fabric bandwidth/latency simulator | Extend Phase 1 `att1_fabric_t` with per-packet-type counters and per-route byte accounting; add rule F10 bandwidth check to `att1_fabric_send`/`att1_fabric_broadcast`; no inference behavior change — **complete** |
| M119 | Placement + command + fabric replay integration | Extend M113 `tools/att1-aimu-replay.c` to parse and replay `routes` section; emit per-route packet/byte/latency statistics in `--report-json` |
| M120 | Phase 3 prototype go/no-go review | Cross-document design review covering M103–M119; gap analysis for Phase 3 ASIC; recommended silicon architecture changes; non-goals for Phase 2 close-out |

---

## 12. Fabric Route Validator (Milestone 116)

`compiler/validate_fabric_routes.py` implements structural and semantic
validation of M115 fabric route report JSON files.  The validator runs
entirely in Python 3 without external dependencies, performs no route
execution, and does not access real PCIe/MMIO registers.

### 12.1 Validator Overview

| Property | Value |
|---|---|
| Tool path | `compiler/validate_fabric_routes.py` |
| Input schema | M115 fabric route report JSON (§11) |
| Exit 0 | Validation passed (zero errors; warnings may be present) |
| Exit 1 | Validation failed (one or more errors) |
| Exit 2 | Parse error (malformed JSON or missing required fields) |

CLI:

```
python3 compiler/validate_fabric_routes.py \
    --report <PATH> \
    [--report-json <OUTPUT_PATH>] \
    [--strict]
```

`--strict` promotes all warnings to errors and treats
`header.status != "pass"` as an error.

### 12.2 Validation Phases

| Phase | Checks performed |
|---|---|
| Header | `route_report_version` present and supported; `tile_count > 0`; `route_count` non-negative; `packet_count_estimate` non-negative; `payload_bytes_estimate` non-negative; `status` recognized |
| Route records | `route_id` present and unique; `route_type` recognized (8 types); `source_tile` in `[0, tile_count)`; `destination_tiles` non-empty for data/barrier types; no self-routing for send/reduce types; `payload_bytes > 0` for data routes; `packet_count_estimate > 0` for data routes; `reduction_behavior` present and recognized; explicit reduction behavior for `PARTIAL_REDUCE` / `LOGITS_REDUCE`; `reduction_id` present for reduction routes; `ordering_policy` recognized; `TILE_BARRIER` must use `barriered` ordering; `route_status` recognized |
| Per-tile summaries | `tile_id` unique and in `[0, tile_count)`; all counter fields non-negative integers; `estimated_bandwidth_gib_sec` finite and non-negative when present |
| Consistency | `header.route_count` matches actual route list length; per-tile `outbound_packet_count` and `inbound_packet_count` reconciled against route records (warnings on mismatch); `warnings`/`failures` lists are arrays |

### 12.3 Rule Coverage

Rules F1–F12 from §7 are covered as follows:

| Rule | Code | Phase | Error / Warning |
|---|---|---|---|
| F1 | `INVALID_TILE_TARGET` | Route records | Error |
| F2 | `EMPTY_DESTINATION` | Route records | Error |
| F3 | `ZERO_PAYLOAD` | Route records | Error |
| F4 | `UNSUPPORTED_PACKET_TYPE` | Route records | Error |
| F5 | `REDUCTION_NOT_EXPLICIT` | Route records | Error |
| F6 | `ROUTE_COUNT_MISMATCH` | Consistency | Error |
| F7 | (cyclic dependency — static schema only; no DAG check) | — | — |
| F8 | `BARRIER_ORDERING` | Route records | Error |
| F9 | (fence ordering — requires command plan; not checked here) | — | — |
| F10 | (bandwidth — requires timing model; not checked here) | — | — |
| F11 | (concat completeness — requires multi-report context) | — | — |
| F12 | `REDUCTION_NOT_EXPLICIT` / `MISSING_REDUCTION_ID` | Route records | Error |

Rules F7, F9, F10, F11 require runtime or multi-report context and are
deferred to M117–M120.

### 12.4 Diagnostic Codes

| Code | Severity | Meaning |
|---|---|---|
| `MISSING_HEADER` | error | No `header` key |
| `MISSING_VERSION` | error | `route_report_version` absent |
| `UNSUPPORTED_VERSION` | error | Version not in supported set |
| `MISSING_TILE_COUNT` | error | `header.tile_count` absent |
| `INVALID_TILE_COUNT` | error | `tile_count` not a positive integer |
| `MISSING_ROUTE_COUNT` | error | `header.route_count` absent |
| `MISSING_STATUS` | error | `header.status` absent |
| `INVALID_STATUS` | error | `status` not in valid set |
| `STRICT_STATUS_NOT_PASS` | error | `--strict` and `status != pass` |
| `MISSING_ROUTES` | error | No top-level `routes` key |
| `INVALID_ROUTE_ITEM` | error | Route element is not an object |
| `MISSING_ROUTE_ID` | error | `route_id` absent |
| `DUPLICATE_ROUTE_ID` | error | `route_id` seen more than once |
| `MISSING_ROUTE_TYPE` | error | `route_type` absent |
| `UNSUPPORTED_PACKET_TYPE` | error | `route_type` not recognized |
| `INVALID_TILE_TARGET` | error | `source_tile` or destination tile out of range |
| `EMPTY_DESTINATION` | error | `destination_tiles` null/empty for non-trace types |
| `SELF_ROUTE` | error | source equals destination for send/reduce types |
| `ZERO_PAYLOAD` | error | `payload_bytes=0` for a data route |
| `ZERO_PACKET_COUNT` | error | `packet_count_estimate=0` for a data route |
| `MISSING_REDUCTION_BEHAVIOR` | error | `reduction_behavior` absent |
| `UNKNOWN_REDUCTION_BEHAVIOR` | error | Value not in valid set |
| `REDUCTION_NOT_EXPLICIT` | error | Reduction route uses `none`/`pass_through` |
| `MISSING_REDUCTION_ID` | error | `reduction_id` absent for reduction route |
| `MISSING_ORDERING_POLICY` | error | `ordering_policy` absent |
| `UNKNOWN_ORDERING_POLICY` | error | Value not in valid set |
| `BARRIER_ORDERING` | error | `TILE_BARRIER` without `barriered` policy |
| `MISSING_ROUTE_STATUS` | error | `route_status` absent |
| `MISSING_TILES` | error | No top-level `tiles` key |
| `INVALID_TILE_ITEM` | error | Tile element is not an object |
| `MISSING_TILE_ID` | error | `tile_id` absent |
| `INVALID_TILE_ID` | error | `tile_id` out of range or wrong type |
| `DUPLICATE_TILE_ID` | error | `tile_id` seen more than once |
| `MISSING_TILE_FIELD` | error | Required counter field absent |
| `INVALID_TILE_FIELD` | error | Counter field not a non-negative integer |
| `INVALID_BANDWIDTH` | error | `estimated_bandwidth_gib_sec` is negative or non-finite |
| `ROUTE_COUNT_MISMATCH` | error | `header.route_count` != actual list length |
| `TILE_OUTBOUND_MISMATCH` | warning | Per-tile outbound packet count differs from route sum |
| `TILE_INBOUND_MISMATCH` | warning | Per-tile inbound packet count differs from route sum |
| `INVALID_DIAG_LIST` | error | `warnings` or `failures` is not an array |

### 12.5 Fixtures

| Fixture path | Expected result | Triggers |
|---|---|---|
| `compiler/fixtures/fabric_route_report_tiny.json` | pass | — (valid) |
| `compiler/fixtures/fabric_route_invalid_tile.json` | fail | `INVALID_TILE_TARGET` (dest tile 99, tile_count 2) |
| `compiler/fixtures/fabric_route_duplicate_id.json` | fail | `DUPLICATE_ROUTE_ID` (two routes with route_id=1) |
| `compiler/fixtures/fabric_route_zero_payload.json` | fail | `ZERO_PAYLOAD` (ACTIVATION_SEND with payload_bytes=0) |
| `compiler/fixtures/fabric_route_missing_reduction.json` | fail | `REDUCTION_NOT_EXPLICIT` (PARTIAL_REDUCE with reduction_behavior="none") |
| `compiler/fixtures/fabric_route_count_mismatch.json` | fail | `ROUTE_COUNT_MISMATCH` (header says 5, list has 1) |

### 12.6 Non-Goals for M116

- No route execution or fabric simulation.
- No DAG cycle detection (F7) — requires full dependency graph from M117.
- No fence ordering validation (F9) — requires command plan context.
- No bandwidth validation (F10) — requires timing model from M118.
- No concat completeness validation (F11) — requires multi-route context.
- No C, Makefile, binary format, or inference behavior changes.

---

## 13. Command-Plan-to-Fabric-Route Mapper (Milestone 117)

`compiler/map_commands_to_fabric_routes.py` reads an M109 AIMU command plan
JSON and emits an M115-compatible fabric route report JSON.  This tool is a
planner and report generator — it does NOT execute routes, change inference
behavior, access real PCIe/MMIO registers, or implement a kernel driver.

### 13.1 Mapper Overview

| Property | Value |
|---|---|
| Tool path | `compiler/map_commands_to_fabric_routes.py` |
| Primary input | M109 AIMU command plan JSON (`--plan PATH`) |
| Optional input | M100 placement report JSON (`--placement-report PATH`) |
| Output | M115 fabric route report JSON (`--route-report-json PATH`) |
| Exit 0 | Route report generated successfully |
| Exit 1 | Structural error in command plan, or strict-mode violation |
| Exit 2 | Parse error (malformed JSON or missing required fields) |

CLI:

```
python3 compiler/map_commands_to_fabric_routes.py \
    --plan <PATH> \
    [--placement-report <PATH>] \
    [--route-report-json <OUTPUT_PATH>] \
    [--tokens-per-sec N] \
    [--fabric-gib-sec N] \
    [--strict]
```

`--strict` fails if any warnings are produced or if the command plan
`header.status` is not `"ok"`.

The `--tokens-per-sec` and `--fabric-gib-sec` flags are informational and
written into the generated report header for downstream tools.

### 13.2 Route Generation Rules

| Command type | Fabric route emitted | Route type | Notes |
|---|---|---|---|
| `LOAD_TENSOR_TILE` | none | — | Host-to-tile DMA; not a tile-to-tile fabric route |
| `VALIDATE_TENSOR` | yes | `CONTROL_ACK` | 64-byte control payload; source tile → all other tiles |
| `TILE_BARRIER` | yes | `TILE_BARRIER` | 0-byte barrier token; source tile → all tiles (incl. self); `barriered` ordering |
| `QUERY_COUNTERS` | yes | `CONTROL_ACK` | 64-byte counter snapshot payload; source tile → all other tiles |
| `TRACE_SNAPSHOT` | yes | `TRACE_EVENT` | 64-byte trace snapshot payload; source tile → all other tiles |
| `FABRIC_SEND` | yes | `ACTIVATION_SEND` | `total_bytes` from command; placement report improves estimate |
| `FABRIC_REDUCE` | yes | `PARTIAL_REDUCE` | `total_bytes` from command; `reduction_behavior` inferred from notes or defaults to `sum` |
| `EXEC_MATMUL` / `EXEC_*` | none | — | Local tile operation; no fabric transfer |
| `KV_APPEND` / `KV_READ` | none | — | Local tile KV cache; no fabric transfer |
| `NOP` / `RESET_TILE` | none | — | Housekeeping; no fabric transfer |

Route IDs are assigned sequentially starting from 1 in command order,
making the mapping deterministic for the same plan.

Payload bytes for data routes are resolved in priority order:
1. `total_bytes` field of the command record.
2. Tensor size from the placement report (by `tensor_id` or `tensor_name`).
3. Default activation estimate (1024 bytes; a warning is emitted).

### 13.3 Per-Tile Summary Generation

Tile traffic summaries are derived entirely from the emitted route records:

- `outbound_packet_count` — sum of `packet_count_estimate` for routes where
  `source_tile == tile_id`.
- `inbound_packet_count` — sum of `packet_count_estimate` for routes where
  `tile_id` appears in `destination_tiles`.
- `outbound/inbound_payload_bytes` — analogous byte sums.
- `barriers_started` / `barriers_completed` — incremented for each
  `TILE_BARRIER` route that includes the tile.
- `reductions_started` — incremented for the source tile of each
  `PARTIAL_REDUCE` route.
- `reductions_completed` — incremented for each destination tile of each
  `PARTIAL_REDUCE` route.

Because summaries are derived from routes, the generated report always passes
the M116 per-tile consistency checks.

### 13.4 Report Header

The generated report header sets:

| Field | Value |
|---|---|
| `route_report_version` | 1 |
| `source_command_plan` | path supplied via `--plan` |
| `source_placement_report` | path supplied via `--placement-report`, or `null` |
| `fabric_policy` | `"layer_wise"` (default; no inference topology available at this stage) |
| `status` | `"pass"` (no warnings), `"warn"` (warnings present), `"fail"` (strict+warnings) |
| `report_timestamp` | UTC ISO-8601 timestamp of the mapper run |

### 13.5 Fixtures

| Fixture path | Description |
|---|---|
| `compiler/fixtures/plan_tiny_barrier_trace.json` | Minimal M109 command plan: TILE_BARRIER + TRACE_SNAPSHOT + QUERY_COUNTERS, 2 tiles |
| `compiler/fixtures/fabric_route_from_plan_tiny.json` | M115 route report generated from the above plan; passes M116 validator (3 routes: TILE_BARRIER + TRACE_EVENT + CONTROL_ACK) |

### 13.6 Relationship to M116

The mapper's output is designed to pass the M116 validator.  The round-trip:

```
map_commands_to_fabric_routes.py --plan plan.json --route-report-json routes.json
validate_fabric_routes.py --report routes.json  # exit 0
```

is the primary correctness check for M117.  The M116 validator enforces all
structural and semantic rules defined in §11.

### 13.7 Non-Goals for M117

- No route execution or fabric simulation.
- No cyclic-dependency check (deferred to M118 bandwidth sim / M120 review).
- No fence-ordering validation across tiles (requires command-plan context
  beyond the current scope).
- No bandwidth estimation (informational flags only; deferred to M118).
- No C, Makefile, binary format, or inference behavior changes.
- No CUDA kernels or runtime scheduler changes.

---

## 14. Fabric Bandwidth and Latency Simulator (Milestone 118)

`compiler/simulate_fabric_bandwidth.py` reads an M115/M117 fabric route report
JSON and produces per-route, per-tile, and aggregate fabric bandwidth and
latency pressure **estimates**.  This is a planner-level report tool — it does
NOT execute routes, change inference behavior, access real PCIe/MMIO
registers, or implement a kernel driver.

**All estimates are deterministic model projections derived from route
metadata, not measured hardware behavior.**

### 14.1 Overview and CLI

```
python3 compiler/simulate_fabric_bandwidth.py \
    --route-report <PATH> \
    [--target-tokens-per-sec N] \
    [--fabric-gib-sec N] \
    [--base-latency-ns N] \
    [--per-hop-latency-ns N] \
    [--packet-overhead-bytes N] \
    [--report-json <OUTPUT_PATH>] \
    [--strict]
```

| Flag | Default | Description |
|---|---|---|
| `--route-report PATH` | required | M115/M117 route report JSON to simulate |
| `--target-tokens-per-sec N` | 1 | Inference token rate target in tokens/s |
| `--fabric-gib-sec N` | none | Fabric bandwidth budget in GiB/s |
| `--base-latency-ns N` | 100 | Base per-route latency estimate in ns |
| `--per-hop-latency-ns N` | 50 | Additional latency per fabric hop in ns |
| `--packet-overhead-bytes N` | 64 | Per-packet header overhead in bytes |
| `--report-json PATH` | none | Write JSON simulation report to PATH |
| `--strict` | off | Exit nonzero on WARN or FAIL (default: only FAIL exits nonzero) |

Exit codes:

| Code | Meaning |
|---|---|
| 0 | PASS or UNKNOWN (also WARN without `--strict`) |
| 1 | FAIL status, or WARN with `--strict`, or structural error |
| 2 | Parse error (malformed JSON or missing required field) |
| 3 | Invalid numeric flag value |

### 14.2 Simulator Model

The simulator treats every route record as an estimated traffic event for one
token inference step.

**Per-route estimates:**

| Field | Formula |
|---|---|
| `effective_payload_bytes` | `payload_bytes` from route record |
| `packet_count` | `packet_count_estimate` from route (0 for `TILE_BARRIER`) |
| `packet_overhead_bytes` | `packet_overhead_bytes` flag × `packet_count` |
| `total_wire_bytes` | `effective_payload_bytes` + `packet_overhead_bytes` |
| `estimated_bandwidth_gib_sec` | `total_wire_bytes × target_tokens_per_sec / GiB` |
| `estimated_latency_ns` | `base_latency_ns + hops × per_hop_latency_ns` |

**Per-tile estimates** are derived by accumulating route-level wire bytes:

- `outbound_wire_bytes_per_token` — sum of `total_wire_bytes` for routes
  where `source_tile == tile_id`.
- `inbound_wire_bytes_per_token` — sum of `total_wire_bytes` for routes
  where `tile_id` appears in `destination_tiles`.
- Bandwidth = wire bytes per token × target token rate / GiB.
- `bottleneck_status` is based on `max(outbound_bw, inbound_bw)`.

**Aggregate estimates** sum all route wire bytes, then apply the token rate.

### 14.3 Latency Model

```
latency_ns = base_latency_ns + hops × per_hop_latency_ns
```

| Route type | Hops |
|---|---|
| `TILE_BARRIER` | 0 (barrier coordination, no data traversal) |
| `CONTROL_ACK` | 0 (local control acknowledgement) |
| All other types | 1 (cross-tile data or trace transfer) |

This is a deterministic estimate only.  It does not model queuing delay,
contention, NoC arbitration, or measured hardware latency.

### 14.4 Status Rules

| Condition | Status |
|---|---|
| No `--fabric-gib-sec` supplied | `UNKNOWN` |
| `required_bandwidth ≤ 80%` of target | `PASS` |
| `required_bandwidth ≤ 100%` of target | `WARN` |
| `required_bandwidth > 100%` of target | `FAIL` |

Status is computed independently at three levels: per-route, per-tile
(bottleneck direction), and aggregate.  The aggregate status is reported as
the top-level simulation status.

### 14.5 Output

**Human-readable report** (always written to stdout):
- Per-route: `route_id`, route type, tile, destination tiles, wire bytes,
  bandwidth estimate, latency estimate, status.
- Per-tile: outbound/inbound bandwidth estimates, bottleneck status.
- Aggregate: total payload and wire bytes per token, required bandwidth,
  fabric target, utilization %, bottleneck route/tile, overall status.
- Warnings list (per-route or per-tile diagnostics).
- Recommendations (rule-based, see §14.6).

**Optional JSON report** (`--report-json`):
- `header` — sim_version, route_report_version, source_route_report,
  tile_count, route_count, token rate, fabric target, latency parameters,
  packet overhead, overall status.
- `routes` — per-route simulation result array.
- `tiles` — per-tile bandwidth and status array.
- `aggregate` — total bytes, required bandwidth, target, utilization %,
  status, bottleneck route/tile.
- `warnings` — list of warning strings.
- `failures` — list of failure strings.

### 14.6 Recommendations

The simulator generates rule-based recommendations based on aggregate status
and route-type traffic distribution:

| Trigger | Recommendation |
|---|---|
| WARN or FAIL | Increase `--fabric-gib-sec` |
| FAIL | Reduce `--target-tokens-per-sec` |
| Activation routes > 50% of wire bytes | Reduce cross-tile placement; split/replicate tensors; prefer head-wise placement for attention |
| Logits reduction > 30% of wire bytes | Split `lm_head`/logits reduction across fewer tiles |
| KV transfer > 30% of wire bytes | Reduce context/session pressure; co-locate KV shards with consumers |
| None of the above | Report utilization within acceptable bounds |

### 14.7 Fixtures

| Fixture path | Description |
|---|---|
| `compiler/fixtures/fabric_route_from_plan_tiny.json` | Input: M117-generated 3-route report (TILE_BARRIER + TRACE_EVENT + CONTROL_ACK, 2 tiles) |
| `compiler/fixtures/bw_sim_tiny.json` | Output: M118 simulator JSON for the above input at 1 token/s, 32 GiB/s target — status `PASS` |

Round-trip test:

```
python3 compiler/simulate_fabric_bandwidth.py \
    --route-report compiler/fixtures/fabric_route_from_plan_tiny.json \
    --target-tokens-per-sec 1 \
    --fabric-gib-sec 32 \
    --report-json /tmp/bw_check.json
# exit 0, status PASS
```

### 14.8 Non-Goals for M118

- No route execution or fabric simulation.
- No contention, queuing, or NoC arbitration modelling.
- No measured hardware latency — all estimates are deterministic projections.
- No cyclic-dependency check (deferred to M120 review).
- No fence-ordering validation (requires full command-plan DAG).
- No concat completeness validation (requires multi-route aggregation beyond
  current scope).
- No C, Makefile, binary format, or inference behavior changes.
- No CUDA kernels or runtime scheduler changes.

## 15. Fabric Route Replay Simulator (Milestone 123)

### 15.1 Role and Scope

The M123 replay simulator (`compiler/replay_fabric_routes.py`) occupies the
position immediately after the M116 validator and M118 bandwidth estimator in
the fabric toolchain pipeline:

```
command plan (M109/M110)
    → route report (M115/M117)
        → route validation (M116)
        → route replay  (M123)   ← this tool
            → BW/latency estimate (M118)
```

The replay simulator:

- Loads and validates the route report using M116 rules (reuses
  `validate_fabric_routes.py` imports directly).
- Replays all routes in deterministic `route_id` order.
- Accumulates per-tile counters: packets sent/received, payload bytes
  sent/received, reductions started/completed, barriers started/completed,
  trace events, route failures.
- Groups reduction routes by `reduction_id`; marks a group complete when at
  least one route in the group has an explicit `reduction_behavior`
  (`sum`, `concat`, `max`, or `topk`).
- Treats each `TILE_BARRIER` route as one barrier started and one barrier
  completed (deterministic single-source token broadcast).
- Integrates M118 bandwidth/latency estimates by calling
  `simulate_fabric_bandwidth` functions directly (no subprocess).
- Emits a human-readable text report (stdout) and an optional JSON report
  (`--report-json`).

This tool does NOT execute routes, access hardware, change inference
behavior, or alter the `.att1` model format.

### 15.2 CLI

```
python3 compiler/replay_fabric_routes.py \
    --route-report PATH \
    [--target-tokens-per-sec N] \
    [--fabric-gib-sec N] \
    [--strict] \
    [--report-json PATH]
```

| Flag | Default | Description |
|---|---|---|
| `--route-report PATH` | required | M115/M116/M117 fabric route report JSON |
| `--target-tokens-per-sec N` | 1.0 | Token decode rate for BW estimates |
| `--fabric-gib-sec N` | (absent) | Fabric bandwidth target; absent → UNKNOWN |
| `--strict` | off | Promote warnings to errors; fail on any route_status ≠ ok |
| `--report-json PATH` | (absent) | Write replay result JSON to path |

**Exit codes:**

| Code | Meaning |
|---|---|
| 0 | Replay passed (zero errors; warnings may be present) |
| 1 | Replay failed (validation errors, route failures, or `--strict`) |
| 2 | Parse error (malformed JSON or missing required field) |

### 15.3 Per-Tile Counters

For each tile `[0, tile_count)` the simulator tracks:

| Counter | Meaning |
|---|---|
| `packets_sent` | Outbound packet count for routes originating at this tile |
| `packets_received` | Inbound packet count for routes terminating at this tile |
| `payload_bytes_sent` | Total outbound payload bytes |
| `payload_bytes_received` | Total inbound payload bytes |
| `reductions_started` | Number of `PARTIAL_REDUCE` / `LOGITS_REDUCE` routes sourced here |
| `reductions_completed` | Number of reduction groups (by `reduction_id`) completed at this tile |
| `barriers_started` | Number of `TILE_BARRIER` routes sourced here |
| `barriers_completed` | Same as `barriers_started` (deterministic per M115 schema) |
| `trace_events` | Number of `TRACE_EVENT` routes sourced here |
| `route_failures` | Routes with `route_status ∉ {ok, warn, skipped}` originating here |

### 15.4 Replay Report Fields

The text and JSON reports share the same field set:

| Field | Description |
|---|---|
| `route_report_path` | Input file path |
| `route_count` | Total routes in the report |
| `routes_replayed` | Routes successfully processed |
| `routes_failed` | Routes with `route_status=fail` (or strict violations) |
| `tile_count` | Number of tiles from header |
| `aggregate_packets_sent` | Sum of all tile `packets_sent` |
| `aggregate_packets_received` | Sum of all tile `packets_received` |
| `aggregate_payload_bytes_sent` | Sum of all tile `payload_bytes_sent` |
| `aggregate_payload_bytes_received` | Sum of all tile `payload_bytes_received` |
| `reductions_started` | Total across all tiles |
| `reductions_completed` | Total reduction groups completed |
| `barriers_started` | Total across all tiles |
| `barriers_completed` | Total barriers completed |
| `trace_events` | Total trace events across all tiles |
| `required_fabric_gib_sec` | From M118 aggregate estimate |
| `fabric_status` | `PASS` / `WARN` / `FAIL` / `UNKNOWN` from M118 |
| `status` | Overall replay outcome: `pass` / `warn` / `fail` |
| `notes` | List of warning and info strings |
| `tiles` | Per-tile counter objects (JSON only) |

**Status rules:**
- `fail` — `routes_failed > 0` (route failures detected).
- `warn` — no route failures but M116 warnings present.
- `pass` — no errors, no warnings.

### 15.5 Fixtures

| Fixture path | Purpose |
|---|---|
| `compiler/fixtures/fabric_route_report_tiny.json` | 3-route, 2-tile, PASS — primary smoke input |
| `compiler/fixtures/fabric_route_reduction_tiny.json` | 3-route, 2-tile, `PARTIAL_REDUCE` (sum) + `TILE_BARRIER` — reduction counter test |
| `compiler/fixtures/fabric_route_invalid_tile.json` | Destination tile 99 out of range — negative test |
| `compiler/fixtures/fabric_route_zero_payload.json` | `payload_bytes=0` for data route — negative test |
| `compiler/fixtures/fabric_route_missing_reduction.json` | `PARTIAL_REDUCE` with `reduction_behavior=none` — negative test |

### 15.6 Non-Goals for M123

- No route execution or fabric simulation.
- No hardware latency measurements — all estimates are deterministic projections.
- No fence-ordering reachability analysis.
- No cyclic dependency check.
- No concat completeness validation.
- No C, Makefile, binary format, or inference behavior changes.
- No CUDA kernels or runtime scheduler changes.

---

## 16. Freeze Policy (Milestone 160)

This fabric packet, routing-metadata, queue-full, barrier, and counter contract
is declared **frozen at v1.0** as of Milestone 160. This milestone freezes the
already-shipped semantics implemented by `include/att1_fabric.h` and
`src/fabric.c`; it does not introduce a new fabric runtime behavior. Stage 2+
work (the emulated PCIe endpoint, `backend_pcie.c`, and any FPGA/ASIC prototype)
must preserve these semantics, and the M161 conformance suite is expected to
validate against them.

**Frozen fields** — covered by this freeze; changing any of the following after
v1.0 requires a fabric-interface **major** version bump and is a backward-
incompatible change:

- The route-descriptor metadata vocabulary in §2.1: `route_id`, `source_tile`,
  `dest_tile_mask`, `packet_type`, `payload_type`, `payload_bytes`, `fence_id`,
  `completion_fence_id`, `reduction_id`, and `trace_id` keep their current
  names, types, and meanings.
- The §3 route-type code assignments `0x01`–`0x08` keep their current meaning:
  `ACTIVATION_SEND`, `ACTIVATION_BROADCAST`, `PARTIAL_REDUCE`,
  `LOGITS_REDUCE`, `KV_TRANSFER`, `TILE_BARRIER`, `TRACE_EVENT`,
  and `CONTROL_ACK`.
- The in-process C API packet-kind enumeration `att1_packet_type`
  (`include/att1_fabric.h`) keeps its current ordinal values and names:

  | Value | `att1_packet_type` | Meaning in the shipped simulator |
  |---|---|---|
  | `0` | `ATT1_PACKET_ACTIVATION` | Activation-class payload packet |
  | `1` | `ATT1_PACKET_LOGITS` | Logit / partial-result payload packet |
  | `2` | `ATT1_PACKET_KV_PAGE` | KV-page payload packet |
  | `3` | `ATT1_PACKET_CONTROL` | Control / acknowledgment packet |
  | `4` | `ATT1_PACKET_BARRIER` | Barrier token packet |
  | `5` | `ATT1_PACKET_TRACE` | Trace-event packet |

  The six-value C enum and the eight-value §3 route-type table are both frozen
  and intentionally remain distinct: the simulator packet enum is the shipped
  payload-kind API, while the route descriptor fields (`packet_type`,
  `payload_type`, `dest_tile_mask`, `reduction_id`) carry the finer-grained
  routing distinction.
- Queue-full behavior implemented by `att1_fabric_send()` and
  `att1_fabric_broadcast()` (`src/fabric.c`):
  - `att1_fabric_send()` returns `ATT1_ERR_QUEUE_FULL` when the destination
    inbound queue is already at capacity; the packet is not enqueued, existing
    queue order is unchanged, and `queue_full_errors` increments once.
  - `att1_fabric_broadcast()` is preflighted and all-or-nothing: if any target
    tile is invalid, the call returns `ATT1_ERR_INVALID_ARG` and no destination
    receives the packet; if any valid non-source destination queue is full, the
    call returns `ATT1_ERR_QUEUE_FULL` and no destination receives the packet.
  - On success, broadcast fan-out is counted per delivered destination packet:
    `broadcast_packets` counts recipient enqueues, not API-call count.
- Barrier semantics implemented by `att1_fabric_barrier_arrive()`:
  - The barrier is single-generation and all-or-nothing.
  - The first valid arrival fixes the participant set for that generation.
  - Later arrivals must present the exact same participant set; non-participants
    and duplicate arrivals are invalid.
  - `out_complete` is set to `1` only on the arrival that completes the full
    participant set; earlier arrivals observe `0`.
  - Completion resets the barrier state immediately so the same participant set
    or a different one may be reused in the next generation.
- The frozen v1.0 counter-name set that existing tests and future hardware must
  map 1:1:

  | Frozen name | Source / mapping | Frozen meaning |
  |---|---|---|
  | `packets_sent` | `att1_fabric_counters.packets_sent` | Successful packet enqueues |
  | `packets_received` | `att1_fabric_counters.packets_received` | Successful packet dequeues |
  | `broadcast_packets` | `att1_fabric_counters.broadcast_packets` | Successful broadcast recipient enqueues (fan-out count) |
  | `payload_bytes_sent` | `att1_fabric_counters.payload_bytes_sent` | Payload bytes enqueued |
  | `payload_bytes_received` | `att1_fabric_counters.payload_bytes_received` | Payload bytes dequeued |
  | `queue_full_errors` | `att1_fabric_counters.queue_full_errors` | Failed send/broadcast attempts due to full destination queue |
  | `invalid_packets` | `att1_fabric_counters.invalid_packets` | Invalid send/receive/barrier argument events |
  | `empty_receives` | `att1_fabric_counters.empty_receives` | Nonblocking receives that found no packet |
  | `barrier_arrivals` | `att1_fabric_counters.barrier_arrivals` | Individual valid barrier arrivals |
  | `barrier_completions` | `att1_fabric_counters.barrier_completions` | Barrier generations completed |
  | `stall_fabric_cycles` | `CNT_STALL_FABRIC` (`docs/aimu_register_map.md` §7.20) | Cycles stalled on fabric send/receive progress |
  | `stall_barrier_cycles` | `CNT_STALL_BARRIER` (`docs/aimu_register_map.md` §7.21) | Cycles stalled in `TILE_BARRIER` waiting for peers |
  | `stall_queue_full_cycles` | `CNT_STALL_QUEUE_FULL` (`docs/aimu_register_map.md` §7.22) | Cycles stalled by queue-full backpressure |

**Reserved fields / additive extensions — not frozen, safe to extend:**

- Route-type code points outside the current §3 assignments `0x01`–`0x08`.
- `payload_type` values outside the current §2.1 set `0`–`5`.
- Additional optional route-report JSON fields, per-route annotations, or
  diagnostics that do not rename or change any frozen field above.
- Additional counters beyond the frozen name set above (for example the richer
  advisory counters described in §6.2–§6.6 such as `packets_by_type[8]`,
  `bytes_by_type[8]`, `reductions_*`, `route_failures`, or
  `ordering_violations`), provided the frozen v1.0 names and meanings remain
  unchanged.
- The microarchitectural realization of the barrier state machine, except for
  the M93 §8.15-4 decision below that it must not depend on host-visible PCIe
  atomic compare-and-swap support.

**Resolution of M93 §8.15-4 (barrier implementation):**

- v1.0 resolves the open question in `docs/aimu_architecture.md` §8.15-4 in
  favor of a **dedicated AIMU-local barrier register/state-machine path**, not a
  host-visible PCIe atomic compare-and-swap primitive.
- Rationale:
  1. PCIe AtomicOp support is optional and topology-dependent across hosts, root
     complexes, and switches, so making it mandatory would weaken the
     substrate-independent contract that M161 must validate.
  2. The frozen first-arrival-defines-participants rule and the
     completing-arrival-only `out_complete=1` rule require endpoint-local state
     beyond a single shared CAS word.
  3. The decision does **not** change the M158 command packet layout or any
     already-defined §2/§3 route metadata; it only resolves the previously
     reserved implementation choice behind the existing `TILE_BARRIER`
     semantics.

**Resolution of M93 §8.15-3 (activation precision, M174):**

- v1.0 keeps f32 as the reference and diagnostic activation precision, but the
  Phase 2 prototype planning path may use **bf16 inter-tile activation
  payloads** when `compiler/validate_m174_activation_precision.py` passes for
  the route report being evaluated.
- The M174 gate compares f32 and bf16 activation payload sizes from M115/M117
  route metadata and runs a deterministic bf16 round-trip error study over
  representative activation values. The checked-in tiny route fixture shows
  50.0% activation-payload savings and 40.0% wire-byte savings after packet
  overhead, with max absolute round-trip error below the configured q8/q4
  credibility margins.
- This is an additive v1.0-compatible decision: no frozen route field is
  renamed or retyped. `payload_type=activation` keeps its meaning, and
  `payload_bytes` continues to mean the actual encoded payload bytes carried by
  the route. If a future schema adds an explicit per-route precision field, that
  is a minor additive extension provided older consumers can still rely on
  `payload_bytes`.

**Version bump rules** (mirrors M157 §1.6 / M158 §1.5 / M159 §15.7):

- **Patch**: documentation clarification only; no route-field, enum, counter,
  or semantic change.
- **Minor**: additive only — new route-type codes in currently unused space, new
  `payload_type` values in currently unused space, new optional route-report
  fields, new counters beyond the frozen v1.0 name set, or new dedicated
  barrier debug/status registers that preserve all existing semantics.
- **Major**: any rename, removal, or meaning change to a frozen route field,
  route-type code, `att1_packet_type` value, counter name, queue-full rule, or
  barrier semantic listed above.

Implementation status: `include/att1_fabric.h` defines the frozen
`att1_packet_type`, `att1_fabric_packet`, and `att1_fabric_counters` contracts;
`src/fabric.c` implements the send/receive/broadcast/barrier semantics above;
`tests/test_fabric.c` and `tests/test_runtime.c` already exercise the queue-
full, counter, and barrier-completion behavior that this milestone freezes. The
in-process fabric API does not expose a standalone runtime version constant
today; this milestone formalizes the existing behavior as the contractual v1.0
baseline rather than changing it.
