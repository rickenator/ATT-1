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

**M157: Register map freeze (v1.0)**

- Freeze [aimu_register_map.md](aimu_register_map.md) (M104): BAR0 layout,
  doorbell, status, capability, counter registers. Add explicit versioning
  and a frozen-fields vs. reserved-fields policy.

**M158: Command packet and completion schema freeze (v1.0)**

- Freeze the M103 command packet format
  ([aimu_pcie_command_requirements.md](aimu_pcie_command_requirements.md))
  and completion record layout, including error/result codes mapped to
  `att1_status_t`.

**M159: DMA descriptor and replay schema freeze (v1.0)**

- Freeze the M107 DMA descriptor model and the M113/M122/M123 command-plan
  and fabric-route replay schemas. Extend
  [schema_compatibility.md](schema_compatibility.md) with a compatibility
  contract: what hardware must accept, what it may reject, how versions
  negotiate.

**M160: Fabric packet and barrier semantics freeze (v1.0)**

- Freeze packet types, routing metadata, queue-full behavior, barrier
  all-or-nothing semantics, and the counter name set (M93 §8.8/§8.10) so
  hardware counters map 1:1 to existing tests.
- Resolve open question M93 §8.15-4 (barrier: PCIe atomic CAS vs. dedicated
  register) at the spec level.

**M161: Conformance test suite for frozen interfaces**

- A substrate-independent conformance harness (C + Python) that any
  endpoint — in-process sim, socket emulator, FPGA — must pass: register
  semantics, command lifecycle, DMA validation, fabric/barrier/counter
  behavior, hostile-input rejection. This is the durable product artifact
  per M153 (validation workflows with no hardware dependency).

> **Gate 1:** no work beyond M161 proceeds until all four freezes are
> ratified and the conformance suite passes against the existing in-process
> simulator.

---

## 5. Stage 2 — Out-of-Process Emulated PCIe Endpoint (M93 Option A / M120 Option B)

This stage moves the tile from "same address space" to "separate process
behind a transport," which is the essential structural step toward a real
device.

**M162: Endpoint process skeleton**

- `att1-aimu-endpoint` daemon: a separate process owning tile memory,
  exposing the frozen register file and command queue over shared memory or
  Unix domain sockets with identical queue semantics and counters (M93 §8.8
  explicitly permits this substitution).

**M163: `backend_pcie.c` host backend**

- New concrete `att1_backend_ops` implementation (per M93 §8.3) that
  dispatches via the endpoint transport instead of CPU/CUDA function calls.
  No other runtime code changes — this validates the backend-swap pattern
  proven by CUDA.

**M164: One-time shard transfer and tensor residency**

- Model load path: `.att1` shard transferred to endpoint-owned memory once;
  enforcement that weights are never re-read by the host during inference
  (M93 §8.12). Residency assertions and counters.

**M165: Device-local KV-MMU in the endpoint**

- KV sessions live in the endpoint process (M93 §8.9): append ordering,
  duplicate rejection, range-copy, eviction, per-session lifecycle, full
  counter set. Host never touches KV data between tokens.

**M166: Single-tile emulated decode, end to end**

- One transformer forward pass + multi-step decode against tiny fixtures
  through `backend_pcie`, matching CPU f32 exactly and q8/q4 within
  tolerance. Existing smoke tests must pass unmodified against
  endpoint-reported counters (M93 §8.2-5).

**M167: Two-tile emulated cluster decode**

- Two endpoint processes, activation routing across the transport, barrier
  and partial-logit reduction (M93 §8.2-4). Backend comparison report gains
  a `pcie` column (M92 extension per M93 §8.4).

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
- [aimu_register_map.md](aimu_register_map.md) — register map to be frozen at M157
- [aimu_pcie_command_requirements.md](aimu_pcie_command_requirements.md) — command packet model to be frozen at M158
- [schema_compatibility.md](schema_compatibility.md) — compatibility contract extended at M159
- [aimu_fabric_routing.md](aimu_fabric_routing.md) — fabric semantics to be frozen at M160
- [CUDA_VALIDATION_PLAN.md](CUDA_VALIDATION_PLAN.md) — M138 signoff plan executed at M155
