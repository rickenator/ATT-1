# ATT-1 Phase 2 Plan: From Simulator to Hardware-Expressible Prototype (M154)

**Status:** Ratified at Milestone 154 (Phase 2 kickoff).
**Revision:** 1.0

This document is the Phase 2 roadmap scaffolding. As with Phase 1, milestone
scope and ordering will shift as evidence accumulates — that is expected and
normal — but the stages, gates, Definition of Done, and non-goals below are
the governing structure for Phase 2.

---

## 1. Guiding Doctrine

Phase 1 proved the *protocol in software*: placement, command queues,
MMIO/register file, DMA descriptors, fabric routing, replay, schema-validated
planning, and f32/q8/q4 correctness (CPU verified, CUDA implemented pending
signoff). Phase 2 must prove the protocol is **hardware-expressible** — per
M93 §8.1 ([aimu_architecture.md](aimu_architecture.md) §8.1), not production
throughput, but that the simulator's abstractions correspond to physically
realizable primitives.

Phase 2 honors standing decisions from Phase 1 governance:

- **M120 (CONDITIONAL GO)** ([aimu_phase3_go_no_go.md](aimu_phase3_go_no_go.md)):
  proceed via the userspace/emulated path (Option A/B) before any FPGA
  commitment.
- **M126 (DEFER FPGA)** ([fpga_feasibility.md](fpga_feasibility.md)): no RTL
  until interfaces are frozen and gate criteria pass.
- **M153 (Winning Strategy)** (`ROADMAP.md` Milestone 153,
  [aimu_phase3_go_no_go.md](aimu_phase3_go_no_go.md) §12): decision-gated
  hardware progression, interface freezes before hardware commitments,
  userspace product value first, CUDA as benchmark infrastructure, explicit
  kill criteria.

Milestone numbering continues the Phase 1 sequence (Phase 1 ended at M153),
so Phase 2 milestones start at **M154**.

---

## 2. Phase 2 Entry Baseline (frozen at M154)

Recorded at the Phase 2 kickoff commit:

| Check | Result |
|---|---|
| `make clean && make && make test` | 782 PASS, 0 FAIL (all C suites) |
| `make regression` | ALL STEPS PASSED (10 steps, incl. golden baselines, schema, hostile-input, pipeline smoke, docs lint, fuzz smoke/coverage) |
| CUDA status | Implemented; manual RTX 3090 signoff closed at M155 (M138 plan) |
| Schema versions | Placement report, command plan, fabric route, execution plan, and pipeline schemas as validated by M134/M135 tooling |

Git bookkeeping for the phase boundary (maintainer step, requires push
rights): tag `PHASE_1` on the master branch head that carries M153, and
create the `PHASE_2` working branch from it:

```sh
git checkout master && git pull
git tag -a PHASE_1 -m "Phase 1 complete: userspace ATT-1/AIMU simulator, M0-M153"
git push origin PHASE_1
git checkout -b PHASE_2
git push -u origin PHASE_2
```

---

## 3. Stage 0 — Entry, Baseline, and Debt Closure

**M154: Phase 2 kickoff and baseline freeze**

- Tag `PHASE_1` on master; create `PHASE_2` branch (manual step, §2).
- Record the frozen baseline (§2): full C suite green, regression suite
  green, schema versions in force.
- Create `docs/PHASE2_PLAN.md` (this document); add the Phase 2 milestone
  section to `ROADMAP.md`; advance `docs/OPERATION_LOG.md` conventions
  (post-milestone checklist unchanged).
- Declare Phase 2 exit criteria up front (§9, "Phase 2 Definition of Done").

**M155: CUDA signoff closure (carry-over debt)** — *complete.*

- Executed the M138 plan on the RTX 3090 host:
  `make clean && make CUDA=1 && make test CUDA=1` plus milestone-specific
  smokes per [CUDA_VALIDATION_PLAN.md](CUDA_VALIDATION_PLAN.md). See
  [CUDA_SIGNOFF_M155.md](CUDA_SIGNOFF_M155.md).
- CUDA is the frozen benchmark/credibility baseline per M153 — its
  tolerances (f32 exact, q8 ≤ 0.15, q4 ≤ 0.35) are the Phase 2 hardware
  acceptance tolerances (M93 §8.13).

**M156: Phase 1 → Phase 2 gap audit** — *complete;
[PHASE1_TO_PHASE2_GAP_AUDIT.md](PHASE1_TO_PHASE2_GAP_AUDIT.md).*

- Systematic diff of M93 §8 requirements against what M103–M153 actually
  delivered; produce a requirements-traceability table
  (requirement → artifact → test → status: proven / partial / unproven).
- Confirm the five "unproven" items from M120 §1 are the Phase 2 target
  list: no real PCIe endpoint or MMIO hardware; no FPGA prototype; no
  physical tile memory/fabric/DMA/power; no hardware-backed inference; no
  tensor-level placement executed against a real model.

---

## 4. Stage 1 — Interface Freeze (the M153 precondition for hardware)

**M157: Register map freeze (v1.0)** — *complete;
[aimu_register_map.md](aimu_register_map.md) §1.6.*

- Freeze [aimu_register_map.md](aimu_register_map.md) (M104): BAR0 layout,
  doorbell, status, capability, counter registers. Add explicit versioning
  and a frozen-fields vs. reserved-fields policy.
- `REGISTER_MAP_VERSION` = `0x0001_0000` (v1.0) declared frozen; §1.6 defines
  which fields are frozen (all named registers in §2–§9), which remain open
  for additive change (reserved bits/regions), and patch/minor/major version
  bump rules. Appendix B open questions reviewed and confirmed non-blocking
  for the freeze.

**M158: Command packet and completion schema freeze (v1.0)** — *complete;
[aimu_pcie_command_requirements.md](aimu_pcie_command_requirements.md) §1.5.*

- Freeze the M103 command packet format
  ([aimu_pcie_command_requirements.md](aimu_pcie_command_requirements.md))
  and completion record layout, including error/result codes mapped to
  `att1_status_t`.
- §1.5 declares the 64-byte command packet layout (§3), completion record
  layout (§7.4), command type enumeration (§4), and error/result code table
  (§2.8) frozen at v1.0, with frozen-fields vs. reserved-fields policy and
  version-bump rules mirroring M157. `att1_aimu_result_to_status()`
  (`include/att1_aimu_cmdq.h`, `src/aimu_cmdq.c`) implements and freezes the
  `att1_aimu_result` → `att1_status_t` mapping table, exercised by
  `tests/test_aimu_cmdq.c`.

**M159: DMA descriptor and replay schema freeze (v1.0)** — *complete;
[aimu_register_map.md](aimu_register_map.md) §15.7 and
[schema_compatibility.md](schema_compatibility.md) §§1, 9, 12.*

- Freeze the M107 DMA descriptor model and the M113/M122/M123 command-plan
  and fabric-route replay schemas. Extend
  [schema_compatibility.md](schema_compatibility.md) with a compatibility
  contract: what hardware must accept, what it may reject, how versions
  negotiate.

**M160: Fabric packet and barrier semantics freeze (v1.0)** — *complete;
[aimu_fabric_routing.md](aimu_fabric_routing.md) §16.*

- Freeze packet types, routing metadata, queue-full behavior, barrier
  all-or-nothing semantics, and the counter name set (M93 §8.8/§8.10) so
  hardware counters map 1:1 to existing tests.
- Resolve open question M93 §8.15-4 (barrier: PCIe atomic CAS vs. dedicated
  register) at the spec level.

**M161: Conformance test suite for frozen interfaces** — *complete;
[`include/att1_aimu_conformance.h`](../include/att1_aimu_conformance.h),
[`src/aimu_conformance.c`](../src/aimu_conformance.c),
[`tests/test_aimu_conformance.c`](../tests/test_aimu_conformance.c),
[`compiler/check_aimu_conformance.py`](../compiler/check_aimu_conformance.py),
[`compiler/test_aimu_conformance.py`](../compiler/test_aimu_conformance.py).*

- Added a substrate-independent conformance endpoint vtable + in-process
  simulator adapter so the same C harness can validate register semantics,
  command lifecycle, DMA validation, fabric/barrier behavior, counter
  snapshots, and hostile-input rejection against future socket-emulator or
  FPGA backends without test changes.
- Added a Python-side static checker/regression test that cross-check the
  frozen v1.0 docs against the shipped C headers for register-map constants,
  command packet/result-code constants, DMA descriptor schema constants, and
  fabric counter names.

> **Gate 1:** satisfied — all four freezes are ratified and the conformance
> suite passes against the existing in-process simulator.

---

## 5. Stage 2 — Out-of-Process Emulated PCIe Endpoint (M93 Option A / M120 Option B)

This stage moves the tile from "same address space" to "separate process
behind a transport," which is the essential structural step toward a real
device.

**M162: Endpoint process skeleton** — *complete;
[`tools/att1-aimu-endpoint.c`](../tools/att1-aimu-endpoint.c),
[`include/att1_aimu_endpoint_protocol.h`](../include/att1_aimu_endpoint_protocol.h),
[`src/aimu_endpoint_protocol.c`](../src/aimu_endpoint_protocol.c),
[`include/att1_aimu_endpoint_client.h`](../include/att1_aimu_endpoint_client.h),
[`src/aimu_endpoint_client.c`](../src/aimu_endpoint_client.c),
[`tests/test_aimu_endpoint.c`](../tests/test_aimu_endpoint.c).*

- `att1-aimu-endpoint` daemon: a separate process owning tile memory,
  exposing the frozen register file and command queue over shared memory or
  Unix domain sockets with identical queue semantics and counters (M93 §8.8
  explicitly permits this substitution).
- Implemented as a thin Unix-domain-socket transport wrapped around the
  existing M161 in-process conformance simulator (`att1_aimu_conformance_*`),
  so the daemon's register/command/DMA/fabric behavior is byte-identical to
  the same-process path by construction rather than by a separate
  reimplementation.
- A matching socket-backed `att1_aimu_conformance_ops` client
  (`att1_aimu_conformance_socket_connect()`) lets any existing or future
  conformance-style test/tool talk to the daemon exactly like the in-process
  endpoint.
- `tests/test_aimu_endpoint.c` spawns the daemon as a child process,
  connects over the socket, and exercises register RO/RW semantics, the
  NOP command lifecycle with counters, DMA validate/submit with counters,
  and fabric send/receive/barrier with counters — confirming identical
  results to the M161 in-process suite.

**M163: `backend_pcie.c` host backend** — *complete;
[`include/att1_backend.h`](../include/att1_backend.h),
[`src/backend_pcie.c`](../src/backend_pcie.c),
[`tests/test_backend_pcie.c`](../tests/test_backend_pcie.c).*

- New concrete `att1_backend_ops` implementation (per M93 §8.3) that
  dispatches via the endpoint transport instead of CPU/CUDA function calls.
  No other runtime code changes — this validates the backend-swap pattern
  proven by CUDA.
- `att1_backend_pcie_create()` wraps a caller-owned, already-connected
  `att1_aimu_conformance_endpoint` (M161 in-process or M162 socket-backed).
  `alloc`/`free` manage ordinary host memory (mirroring the CUDA backend's
  host-buffer convention); `sync` forwards to `sync_mmio`. Each math op
  submits one frozen v1.0 (M158) command packet
  (`EXEC_MATMUL`/`EXEC_RMSNORM`/`EXEC_ROPE`/`EXEC_FFN`) and round-trips it
  through `cmd_submit`/`cmd_dispatch_one`/`cmd_poll_completion`, mapping the
  completion result code to success/failure via
  `att1_aimu_result_to_status()`. `softmax_f32` is left unimplemented: no
  frozen command type covers plain softmax (only the larger fused
  `EXEC_ATTENTION`).
- Because the M161/M162 command-queue simulator does not yet execute any
  `EXEC_*` command, every math op currently fails with
  `ATT1_ERR_UNSUPPORTED`; this milestone proves the transport plumbing and
  backend-swap contract, not compute correctness — that lands at M166 once
  the endpoint actually executes `EXEC_*` commands.

**M164: One-time shard transfer and tensor residency** — *complete;
[`src/backend_pcie.c`](../src/backend_pcie.c),
[`include/att1_backend.h`](../include/att1_backend.h),
[`tests/test_backend_pcie.c`](../tests/test_backend_pcie.c).*

- `att1_backend_pcie_load_tensor()` implements the M93 §8.12 model-load
  data-movement contract ("Tensor shard (model weights): host → device,
  once, at model load"): it transfers a tensor's bytes from a host address
  to a device address via one or more frozen v1.0 (M159)
  `att1_aimu_dma_desc` submissions issued through the backend's
  `att1_aimu_conformance_endpoint`, chunking any transfer larger than
  `ATT1_AIMU_DMA_MAX_TRANSFER_BYTES` into multiple sequential descriptors
  (the last one flagged `ATT1_AIMU_DMA_FLAG_LAST_DESCRIPTOR`).
- Once a `tensor_id` has been successfully transferred it is recorded as
  resident for the lifetime of the backend in a fixed-capacity
  resident-tensor table (4096 entries). A second call for the same
  `tensor_id` is rejected with `ATT1_ERR_STATE` and does **not** resubmit
  any DMA transfer — this is the enforcement mechanism for "weights are
  never re-read by the host during inference": callers (the model loader,
  in a later milestone) are expected to call this once per tensor at load
  time and never again.
- New `att1_backend_pcie_residency_counters` (tensors resident, transfers
  submitted, descriptors submitted, bytes transferred, duplicate-transfer
  rejections) are exposed via `att1_backend_pcie_get_residency_counters()`;
  `att1_backend_pcie_tensor_is_resident()` lets callers query residency
  directly.
- This milestone reuses the already-frozen (M159) DMA descriptor
  validate/submit contract unchanged; it does not add real device-side
  memory backing or change `EXEC_*` command execution (still deferred to
  M166) — it proves the one-time-transfer/residency policy layer that the
  model loader will use once `backend_pcie` is wired into inference.

**M165: Device-local KV-MMU in the endpoint** — *complete;
[`include/att1_aimu_conformance.h`](../include/att1_aimu_conformance.h),
[`src/aimu_conformance.c`](../src/aimu_conformance.c),
[`include/att1_aimu_endpoint_protocol.h`](../include/att1_aimu_endpoint_protocol.h),
[`src/aimu_endpoint_protocol.c`](../src/aimu_endpoint_protocol.c),
[`src/aimu_endpoint_client.c`](../src/aimu_endpoint_client.c),
[`tools/att1-aimu-endpoint.c`](../tools/att1-aimu-endpoint.c),
[`tests/test_aimu_conformance.c`](../tests/test_aimu_conformance.c),
[`tests/test_aimu_endpoint.c`](../tests/test_aimu_endpoint.c).*

- KV sessions live in the endpoint process (M93 §8.9): append ordering,
  duplicate rejection, range-copy, eviction, per-session lifecycle, full
  counter set. Host never touches KV data between tokens.
- `att1_aimu_conformance_endpoint` (M161) gains
  `kv_create_session`/`kv_destroy_session`/`kv_append`/`kv_read`/
  `kv_copy_range`/`kv_get_counters`, each forwarding to an internal
  `att1_kv_mmu` (M151) instance owned by the endpoint — the already-frozen
  append-ordering/duplicate-rejection/range-copy/eviction/counter semantics
  are reused unchanged rather than reimplemented at the transport layer.
  `att1_aimu_conformance_config` gains a `kv_*` sub-config (sessions, pages,
  layers, heads, head_dim, page_tokens, max_positions) with defaults.
- The M162 socket-backed daemon (`att1-aimu-endpoint`) and client gain the
  same six ops end to end, so KV sessions behave identically whether the
  endpoint is in-process or out-of-process: `kv_append`/`kv_read`/
  `kv_copy_range` carry explicit float-element counts so the wire protocol
  can size key/value transfers without guessing the KV-MMU's configured
  shape (the request/response payload buffer grew from 4096 to 16384 bytes
  to give range-copies room).
- No changes to `src/kv_mmu.c` itself (M151 simulator reused unchanged) or
  to `EXEC_*` command execution (still deferred to M166).

**M166: Single-tile emulated decode, end to end** — *complete;
[`include/att1_aimu_cmdq.h`](../include/att1_aimu_cmdq.h),
[`src/aimu_cmdq.c`](../src/aimu_cmdq.c),
[`src/aimu_conformance.c`](../src/aimu_conformance.c),
[`src/backend_pcie.c`](../src/backend_pcie.c),
[`tests/test_aimu_conformance.c`](../tests/test_aimu_conformance.c),
[`tests/test_backend_pcie.c`](../tests/test_backend_pcie.c).*

- One transformer forward pass + multi-step decode against tiny fixtures
  through `backend_pcie`, matching CPU f32 exactly and q8/q4 within
  tolerance. Existing smoke tests must pass unmodified against
  endpoint-reported counters (M93 §8.2-5).
- `att1_aimu_cmdq` gains an optional real-execution hook
  (`att1_aimu_cmdq_set_exec_hook()`); unset, it preserves the original M105
  "always `ATT1_AIMU_ERR_UNSUPPORTED_OP`" behavior exactly, so every raw
  `att1_aimu_cmdq`-level test is unaffected. The in-process
  `att1_aimu_conformance_endpoint` (M161) installs a hook that maintains a
  resident-tensor registry (populated by `LOAD_TENSOR_TILE`) and executes
  `EXEC_MATMUL`/`EXEC_RMSNORM`/`EXEC_ROPE`/`EXEC_FFN` against real host
  buffers using the same `att1_math.h`/`att1_quant.h` primitives the CPU
  backends call directly.
- `backend_pcie.c`'s math ops auto-register their weight/norm pointer under
  a tensor_id the first time it is used and reuse it thereafter; `EXEC_FFN`
  packs its two input buffers into one scratch buffer since the frozen
  packet has only one input address field; `softmax_f32` is computed
  locally (still no dedicated frozen command type, M103 §4.6).
- `EXEC_ATTENTION`, `KV_APPEND`/`KV_READ`, `FABRIC_SEND`/`FABRIC_REDUCE`,
  and `VALIDATE_TENSOR` remain unsupported at the cmdq level; two-tile
  cluster decode over the M162 socket transport is deferred to M167.

**M167: Two-tile emulated cluster decode** — *complete;
[`include/att1_aimu_cluster_bridge.h`](../include/att1_aimu_cluster_bridge.h),
[`src/aimu_cluster_bridge.c`](../src/aimu_cluster_bridge.c),
[`tests/test_aimu_cluster_decode.c`](../tests/test_aimu_cluster_decode.c).*

- `att1_aimu_cluster_bridge` is a thin host-side relay, built entirely out
  of the already-frozen `att1_aimu_conformance_fabric_send/receive/
  barrier_arrive` calls (M160/M161), that bridges activation/logits
  packets and barrier arrivals between two independent
  `att1_aimu_conformance_endpoint` instances — transport-agnostic, so it
  works unchanged whether both endpoints are in-process (M161) or two
  separate `att1-aimu-endpoint` daemon *processes* connected over Unix
  domain sockets (M162). Each bridged endpoint's fabric is configured with
  `tile_count=2` (local compute tile 0, a bridge-owned "proxy" tile 1); the
  bridge relays proxy-slot packets to the peer's real tile 0 and cascades a
  two-participant `{0,1}` barrier completion on both endpoints only once
  both real tiles have locally arrived.
- `test_two_tile_decode_inprocess` really executes a tiny two-layer
  transformer split one layer per tile across two in-process
  `att1_backend_pcie` (M163/M166) instances: tile 0's layer-0 activation is
  routed to tile 1 over the bridge (M93 §8.2-2), tile 1 runs layer 1 and
  the final RMSNorm and routes the normed hidden state back to tile 0,
  both tiles rendezvous at the bridge barrier (M93 §8.2-4), then each
  computes a row-parallel *partial* logit vector from its own half of the
  output projection's contraction (`d_model`) dimension and combines them
  via one more bridge-routed send — matching a direct cpu-f32 reference
  exactly.
- `test_two_tile_decode_socket` drives the identical protocol over two
  real `att1-aimu-endpoint` daemon *processes*, proving the fabric/barrier
  protocol genuinely crosses OS process boundaries (M93 §8.8) and
  asserting the daemons' own fabric counters (queried back over the
  sockets) reflect real traffic. This variant intentionally reuses the
  cpu-f32 reference values as each tile's "local computation" rather than
  exercising `EXEC_*` math on the socket-backed endpoints: M166's exec
  hook resolves tensor operands via raw host pointers, which are only
  valid within the process that issued them, so forwarding those pointer
  values verbatim to a daemon's different address space and dereferencing
  them there is undefined behavior. Giving the socket-backed endpoint real
  device-local tensor memory (so tensor payloads are transferred and
  dereferenced only within the daemon's own address space) remains a
  documented gap for a future milestone — M167 is scoped to the
  fabric/barrier/reduction protocol (M93 §8.2-2/8.2-4), not cross-process
  compute.
- The backend comparison report's `pcie` column (M92 extension per
  M93 §8.4) remains deferred: it requires wiring a two-process cluster
  decode path into `att1-bench`/`compiler/backend_comparison_report.py`,
  which depends on the device-local tensor memory gap above being closed
  first.

**M168: Fault injection and hostile-endpoint testing**

- Queue-full, malformed completion, out-of-order/duplicate KV appends,
  endpoint crash mid-decode, DMA descriptor rejection — all must map to
  `att1_status_t` without UB (M93 §8.7-6). Extends the Phase 1
  hostile-input tradition to the transport boundary.

**M169: Replay fidelity gate**

- Replay a full planning-pipeline command plan and fabric routes (M119/M132
  artifacts) against the emulated endpoint; byte-level trace/counter
  equivalence with the in-process simulator. This is an explicit M153
  decision-gate criterion ("replay fidelity, deterministic traces").

> **Gate 2:** emulated endpoint passes conformance suite, replay fidelity,
> tolerance targets, and fault injection.

---

## 6. Stage 3 — Performance Model Calibration and Real-Model Scale

**M170: Measured transport characterization**

- Measure latency/bandwidth of the emulated transport per packet class;
  calibrate the M118 fabric bandwidth/latency simulator against measured
  numbers so the model has an empirical anchor before hardware.

**M171: Real small-model end-to-end (SmolLM2-135M class)**

- Convert and run a real ~135M model through the emulated two-tile path
  (the exact sizing case in M93 §8.6), q8 primary, f32 reference. This
  closes M120's "no tensor-level placement executed against a real model"
  gap.

**M172: Beachhead workload definition and baseline metrics (M153 items 1–2)**

- Define the long-context, bandwidth-bound, high-KV-pressure decode
  workload; publish hard success metrics: memory movement, latency
  stability, cost-per-token vs. the CUDA baseline from M155. Deterministic,
  reproducible benchmark harness.

**M173: Placement and capacity validation at Phase 2 memory envelopes**

- Run placement scenarios against the M93 §8.6 device budgets
  (256 MB / 512 MB / 1 GB) and the KV math (≈300 MB for 2048-token f32 KV);
  confirm feasible-placement signals for the prototype configurations.
  Resolve open question M93 §8.15-2 (KV page size) empirically.

**M174: Activation precision study (open question M93 §8.15-3)**

- Measure f32 vs. bf16 inter-tile activation packets: fabric bandwidth
  savings vs. q8/q4 tolerance impact. Decision recorded in the fabric spec
  (as a versioned amendment if adopted).

---

## 7. Stage 4 — FPGA Go/No-Go and (Gated) Physical Prototype

**M175: FPGA gate review (successor to M120/M126)**

- Formal review against the M126 §10 gate criteria: interfaces frozen
  (Stage 1), emulator end-to-end fidelity (Stage 2), calibrated performance
  model and real-model evidence (Stage 3), plus M153 kill-criteria
  assessment.
- Outputs one of: **GO** (proceed to M176+), **HOLD** (iterate emulator), or
  **PIVOT** (per M153: license the planning/control-plane stack).

**M176–M181 (contingent on GO): FPGA control-plane prototype** — scoped per
[fpga_feasibility.md](fpga_feasibility.md) §2, control-plane first, math
later:

- **M176:** Board selection and BOM decision (from
  [phase3_bom_board_options.md](phase3_bom_board_options.md) groundwork);
  procurement.
- **M177:** PCIe enumeration + BAR0 MMIO: device visible on the bus, host
  `mmap`s BAR0, reads capability registers. Userspace access (VFIO or
  vendor XDMA) — no custom kernel driver, preserving the standing non-goal.
- **M178:** Command queue doorbell + completion path on FPGA; conformance
  suite subset passes against real hardware.
- **M179:** DMA descriptor validation and shard transfer to device memory
  (HBM/BRAM).
- **M180:** Counter/trace registers readable by the existing test
  infrastructure unmodified (M93 §8.10's 1:1 counter mapping proven on
  silicon-adjacent hardware).
- **M181:** Fabric route replay acknowledgment on FPGA (no tensor math
  yet) — completing the "protocol is hardware-expressible" proof for the
  control plane.

Math-on-FPGA (matmul unit choice, M93 §8.15-1) is deliberately deferred to a
later decision point within Phase 2 or into Phase 3, informed by M175.

---

## 8. Stage 5 — Productization and Partner Validation (M153 items 4, 7–9)

Runs partially in parallel with Stages 2–4:

**M182: Packaged validation platform**

- The planning/replay/conformance/placement toolchain packaged as the
  standalone "software validation platform" deliverable (M153
  phased-adoption tier 1): install story, versioned schemas, reference
  workflows, docs.

**M183: Design-partner trace program**

- Instrument for ingesting 2–3 design partners' production-like traces with
  explicit pass/fail criteria (M153 item 8); hostile-input validation on
  partner-supplied artifacts.

**M184: Phase 2 close-out review and Phase 3 go/no-go**

- Requirements traceability final pass against M93 §8.2's six proof
  obligations; updated go/no-go successor to M120 covering ASIC direction;
  kill-criteria verdict per M153 item 10; tag `PHASE_2` equivalent
  close-out.

---

## 9. Phase 2 Definition of Done (declared at M154)

Direct restatement of M93 §8.2, plus governance:

1. **Tile isolation proven** — endpoint-owned tensor memory, ≥1 transformer
   block, host never touches weights mid-inference.
2. **Activation routing** across a real transport boundary with measured
   latency/bandwidth.
3. **Device-local KV** across multi-step decode.
4. **Two-tile cooperative decode** with barrier + reduction, within
   tolerances (f32 exact, q8 ≤ 0.15, q4 ≤ 0.35).
5. **Counter fidelity:** existing tests validate hardware/emulator output
   unmodified.
6. **f32 + q8 end-to-end validated** (q4 desirable, non-blocking).
7. **All four interface freezes ratified and honored;** conformance suite
   green on every substrate shipped.
8. **FPGA gate decision made explicitly,** with evidence, whichever way it
   goes.

---

## 10. Non-Goals (unchanged from M93 §8.14 + repo policy)

No production ASIC, no Linux kernel driver, no public-cloud deployment, no
mobile/Vulkan/OpenCL, no training, no multi-model hot-swap, no patent-claim
language. Real large-model (120B-class) inference remains Phase 2+ scaled
work, not a Phase 2 gate.

---

## 11. Key Risks and Expected Milestone Churn

- **Interface freeze too early:** Stage 2/3 findings (esp. bf16 activations,
  KV page size, barrier mechanism) may force versioned amendments —
  expected and handled via the M159 compatibility contract rather than
  silent drift.
- **CUDA signoff slippage (M155):** blocks credible baselines; front-loaded
  deliberately.
- **FPGA gate is a genuine fork:** M176–M181 may be replaced wholesale by a
  HOLD/PIVOT path; the plan treats them as contingent, not committed.
- **Transport emulation fidelity:** shared-memory queues can hide real PCIe
  pathologies (ordering, MMIO latency, DMA coherence); M170 calibration and
  M126's caveats bound how much the emulator can claim.
- **Partner traces (M183):** external dependency; can slide without blocking
  the technical gate at M184.

---

## Related Documents

- [aimu_architecture.md](aimu_architecture.md) — §8 Phase 2 PCIe/AIMU prototype requirements (M93)
- [aimu_phase3_go_no_go.md](aimu_phase3_go_no_go.md) — M120 go/no-go review; §12 M153 winning-strategy execution contract
- [fpga_feasibility.md](fpga_feasibility.md) — M126 FPGA feasibility and gate criteria
- [phase3_bom_board_options.md](phase3_bom_board_options.md) — board/BOM groundwork
- [aimu_register_map.md](aimu_register_map.md) — register map frozen v1.0 at M157 (§1.6)
- [aimu_pcie_command_requirements.md](aimu_pcie_command_requirements.md) — command packet model frozen v1.0 at M158 (§1.5)
- [schema_compatibility.md](schema_compatibility.md) — compatibility contract extended at M159
- [aimu_fabric_routing.md](aimu_fabric_routing.md) — fabric semantics to be frozen at M160
- [CUDA_VALIDATION_PLAN.md](CUDA_VALIDATION_PLAN.md) — M138 signoff plan executed at M155
