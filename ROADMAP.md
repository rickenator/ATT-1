# Roadmap

## Milestone 0: Project Skeleton

- Strict C11 Makefile.
- `build/att1-sim` startup binary.
- Logging module.
- Basic configuration structs.
- Placeholder public headers.
- Smoke test.

## Milestone 1: Core Simulator Objects

- Tile, tensor, memory, and fabric object lifetimes.
- Deterministic simulator clock.
- Structured error handling.

## Milestone 2: Tensor Memory Model

- Tensor descriptors.
- Local model memory accounting.
- Host-to-tile model placement stubs.

## Milestone 3: Tile ISA Sketch

- Minimal programmable tile command format.
- Command decoder.
- Trace-only execution for tensor operations.

## Milestone 4: Packetized Fabric

- Packet headers and routing metadata.
- Tile-to-tile message queues.
- Fabric latency and bandwidth knobs.

## Milestone 5: KV-Cache MMU

- KV page descriptors.
- Logical-to-physical KV mapping.
- Eviction and migration simulation hooks.

## Milestone 6: Runtime Scheduler

- Request admission.
- Token-step scheduling.
- Multi-tile work assignment.

## Milestone 7: Model Loader Prototype

- Simple model manifest format.
- Layer and tensor inventory.
- Placement validation.

## Milestone 8: Quantization Metadata

- Quantization descriptors.
- Per-tensor scale metadata.
- Validation paths without numerical kernels.

## Milestone 9: Sampling and Token Flow

- Sampler interface.
- Token stream lifecycle.
- Deterministic test sampler.

## Milestone 10: Phase-2 Hardware Bridge Plan

- PCIe card runtime boundary.
- Driver command queue model.
- Hardware validation traces.
- Simulator-to-hardware compatibility checklist.

## Milestone 153: Winning Strategy Execution

- Define one beachhead workload: long-context, bandwidth-bound decode with high KV pressure.
- Publish hard success metrics: memory movement, latency stability, and cost-per-token versus strong GPU baselines.
- Freeze command/control interfaces before hardware commitments: register map, command packets, replay schemas.
- Productize userspace value first: deterministic planning, replay, and validation workflows with no hardware dependency.
- Keep hardware progression decision-gated: replay fidelity, deterministic traces, feasible placement scenarios, and fabric-route validation must pass before FPGA.
- Treat CUDA as benchmark infrastructure and credibility gate, not fallback identity.
- Position governance artifacts as product value: placement reports, execution plans, schema compatibility, hostile-input validation.
- Establish 2–3 design partners with production-like traces and explicit pass/fail criteria.
- Package delivery as phased adoption: software validation platform, emulated control-plane integration, optional hardware acceleration.
- Enforce kill criteria: if no durable repeatable advantage after emulator-phase evidence, pivot to licensing the planning/control-plane stack.

---

# Phase 2 Roadmap (M154–M184)

Phase 1 (M0–M153) proved the protocol in software. Phase 2 proves the
protocol is hardware-expressible. Full detail, gates, Definition of Done,
and risks: [docs/PHASE2_PLAN.md](docs/PHASE2_PLAN.md).

## Stage 0: Entry, Baseline, and Debt Closure

- Milestone 154: Phase 2 kickoff and baseline freeze — PHASE_1 tag, PHASE_2 branch, frozen baseline record, `docs/PHASE2_PLAN.md`, Phase 2 Definition of Done declared.
- Milestone 155: CUDA signoff closure (complete) — M138 plan executed on RTX 3090; CUDA tolerances (f32 exact, q8 ≤ 0.15, q4 ≤ 0.35) are now the Phase 2 hardware acceptance tolerances.
- Milestone 156: Phase 1 → Phase 2 gap audit (complete) — requirements-traceability table for M93 §8 vs. M103–M155 deliverables; see [docs/PHASE1_TO_PHASE2_GAP_AUDIT.md](docs/PHASE1_TO_PHASE2_GAP_AUDIT.md).
- Milestone 157: Register map freeze v1.0 (complete) — [docs/aimu_register_map.md](docs/aimu_register_map.md) §1.6 declares `REGISTER_MAP_VERSION` `0x0001_0000` frozen with an explicit frozen-fields vs. reserved-fields policy and version-bump rules; first of four Stage 1 interface freezes toward Gate 1.

## Stage 1: Interface Freeze

- Milestone 157: Register map freeze (v1.0) — BAR0 layout, doorbell, status, capability, counter registers; frozen vs. reserved fields policy.
- Milestone 158: Command packet and completion schema freeze (v1.0) — including `att1_status_t` error/result-code mapping.
- Milestone 159: DMA descriptor and replay schema freeze (v1.0) — compatibility contract in `docs/schema_compatibility.md`.
- Milestone 160: Fabric packet and barrier semantics freeze (v1.0) — packet types, queue-full behavior, barrier semantics, counter name set; resolve barrier mechanism question (M93 §8.15-4).
- Milestone 161: Conformance test suite for frozen interfaces — substrate-independent harness (in-process sim, emulator, FPGA).

Gate 1: all four freezes ratified; conformance suite green on the in-process simulator.

## Stage 2: Out-of-Process Emulated PCIe Endpoint

- Milestone 162: `att1-aimu-endpoint` process skeleton — separate process owning tile memory behind shared-memory/Unix-socket transport.
- Milestone 163: `backend_pcie.c` host backend — new `att1_backend_ops` implementation; no other runtime changes.
- Milestone 164: One-time shard transfer and tensor residency enforcement.
- Milestone 165: Device-local KV-MMU in the endpoint — full session lifecycle and counters.
- Milestone 166: Single-tile emulated decode end to end — f32 exact, q8/q4 within tolerance; existing smoke tests unmodified.
- Milestone 167: Two-tile emulated cluster decode — transport activation routing, barrier, partial-logit reduction; `pcie` backend column in comparison report.
- Milestone 168: Fault injection and hostile-endpoint testing — all failures map to `att1_status_t` without UB.
- Milestone 169: Replay fidelity gate — byte-level trace/counter equivalence with the in-process simulator.

Gate 2: emulated endpoint passes conformance, replay fidelity, tolerances, fault injection.

## Stage 3: Performance Model Calibration and Real-Model Scale

- Milestone 170: Measured transport characterization — calibrate M118 fabric simulator against measured numbers.
- Milestone 171: Real small-model end-to-end (SmolLM2-135M class) through the emulated two-tile path.
- Milestone 172: Beachhead workload definition and baseline metrics — memory movement, latency stability, cost-per-token vs. CUDA baseline.
- Milestone 173: Placement and capacity validation at 256 MB / 512 MB / 1 GB device budgets; resolve KV page size question (M93 §8.15-2).
- Milestone 174: Activation precision study — f32 vs. bf16 inter-tile packets (M93 §8.15-3).

## Stage 4: FPGA Go/No-Go and (Gated) Physical Prototype

- Milestone 175: FPGA gate review — GO / HOLD / PIVOT against M126 §10 criteria and M153 kill criteria.
- Milestone 176: (GO only) Board selection, BOM decision, procurement.
- Milestone 177: (GO only) PCIe enumeration + BAR0 MMIO via userspace access (VFIO/XDMA); no kernel driver.
- Milestone 178: (GO only) Command queue doorbell + completion path on FPGA; conformance subset on real hardware.
- Milestone 179: (GO only) DMA descriptor validation and shard transfer to device memory.
- Milestone 180: (GO only) Counter/trace registers readable by existing test infrastructure unmodified.
- Milestone 181: (GO only) Fabric route replay acknowledgment on FPGA — control-plane hardware-expressibility proof complete.

## Stage 5: Productization and Partner Validation

- Milestone 182: Packaged validation platform — planning/replay/conformance/placement toolchain as a standalone deliverable.
- Milestone 183: Design-partner trace program — 2–3 partners, production-like traces, explicit pass/fail criteria.
- Milestone 184: Phase 2 close-out review and Phase 3 go/no-go — traceability final pass, kill-criteria verdict, PHASE_2 close-out tag.
