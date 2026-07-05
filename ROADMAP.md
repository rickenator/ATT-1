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
- Milestone 158: Command packet and completion schema freeze v1.0 (complete) — [docs/aimu_pcie_command_requirements.md](docs/aimu_pcie_command_requirements.md) §1.5 declares the 64-byte command packet layout, completion record, command type enumeration, and error/result code table frozen v1.0; `att1_aimu_result_to_status()` freezes the `att1_status_t` error/result-code mapping; second of four Stage 1 interface freezes toward Gate 1.
- Milestone 159: DMA descriptor and replay schema freeze v1.0 (complete) — [docs/aimu_register_map.md](docs/aimu_register_map.md) §15.7 freezes the M107 in-process `att1_aimu_dma_desc`/validation/counter contract, and [docs/schema_compatibility.md](docs/schema_compatibility.md) freezes the M113/M122/M123 replay-report schemas plus their v1 compatibility contract; third of four Stage 1 interface freezes toward Gate 1.
- Milestone 160: Fabric packet and barrier semantics freeze v1.0 (complete) — [docs/aimu_fabric_routing.md](docs/aimu_fabric_routing.md) §16 freezes the route-descriptor metadata, route-type codes, shipped `att1_packet_type` / `att1_fabric_counters` contract, queue-full behavior, and single-generation barrier semantics, and resolves M93 §8.15-4 in favor of a dedicated barrier register/state-machine path; fourth of four Stage 1 interface freezes toward Gate 1.
- Milestone 161: Conformance test suite for frozen interfaces (complete) — [include/att1_aimu_conformance.h](include/att1_aimu_conformance.h) / [src/aimu_conformance.c](src/aimu_conformance.c) add a substrate-independent AIMU endpoint vtable and in-process simulator adapter; [tests/test_aimu_conformance.c](tests/test_aimu_conformance.c) exercises register semantics, command lifecycle, DMA validation, and fabric/barrier/counter behavior through the abstract endpoint only; [compiler/check_aimu_conformance.py](compiler/check_aimu_conformance.py) cross-checks the frozen docs against the shipped C headers; Gate 1 satisfied.
- Milestone 162: Endpoint process skeleton (complete) — [tools/att1-aimu-endpoint.c](tools/att1-aimu-endpoint.c) daemon owns tile memory/register file/command queue in a separate process and serves the frozen M161 conformance ops over a Unix domain socket via [include/att1_aimu_endpoint_protocol.h](include/att1_aimu_endpoint_protocol.h)/[src/aimu_endpoint_protocol.c](src/aimu_endpoint_protocol.c); [src/aimu_endpoint_client.c](src/aimu_endpoint_client.c) provides a matching socket-backed conformance client; [tests/test_aimu_endpoint.c](tests/test_aimu_endpoint.c) spawns the daemon and validates identical register/command/DMA/fabric semantics and counters over the transport; first of Stage 2's out-of-process milestones.
- Milestone 163: `backend_pcie.c` host backend (complete) — [src/backend_pcie.c](src/backend_pcie.c) / `att1_backend_pcie_create()` implement `att1_backend_ops` by submitting frozen v1.0 `EXEC_*` command packets through a caller-owned `att1_aimu_conformance_endpoint` (in-process or socket-backed) and mapping completions via `att1_aimu_result_to_status()`; validates the backend-swap pattern proven by CUDA, not compute correctness (the endpoint does not execute `EXEC_*` yet); [tests/test_backend_pcie.c](tests/test_backend_pcie.c) covers alloc/free/sync and the submit/dispatch/poll round trip for every math op.
- Milestone 164: One-time shard transfer and tensor residency (complete) — `att1_backend_pcie_load_tensor()` in [src/backend_pcie.c](src/backend_pcie.c) transfers a tensor shard host→device once via one or more frozen v1.0 (M159) `att1_aimu_dma_desc` submissions (chunked at `ATT1_AIMU_DMA_MAX_TRANSFER_BYTES`), marks the `tensor_id` resident, and rejects a second load for the same `tensor_id` with `ATT1_ERR_STATE` without resubmitting any transfer — the M93 §8.12 "weights never re-read by the host" enforcement; new `att1_backend_pcie_residency_counters` track tensors resident, transfers/descriptors submitted, bytes transferred, and duplicate-load rejections; [tests/test_backend_pcie.c](tests/test_backend_pcie.c) `load_tensor_residency` covers first-load success, duplicate rejection, independent tensors, counters, and invalid-argument handling.
- Milestone 165: Device-local KV-MMU in the endpoint (complete) — [include/att1_aimu_conformance.h](include/att1_aimu_conformance.h) / [src/aimu_conformance.c](src/aimu_conformance.c) add `kv_create_session`/`kv_destroy_session`/`kv_append`/`kv_read`/`kv_copy_range`/`kv_get_counters` to the conformance endpoint, forwarding to an internal `att1_kv_mmu` (M151) instance so append ordering, duplicate rejection, range-copy, eviction, and counters are enforced inside the endpoint rather than the host; the M162 socket transport ([include/att1_aimu_endpoint_protocol.h](include/att1_aimu_endpoint_protocol.h), [src/aimu_endpoint_client.c](src/aimu_endpoint_client.c), [tools/att1-aimu-endpoint.c](tools/att1-aimu-endpoint.c)) gains the same six ops; [tests/test_aimu_conformance.c](tests/test_aimu_conformance.c) and [tests/test_aimu_endpoint.c](tests/test_aimu_endpoint.c) cover session lifecycle, ordering/duplicate rejection, range-copy, and counters both in-process and over the socket.
- Milestone 166: Single-tile emulated decode, end to end (complete) — [include/att1_aimu_cmdq.h](include/att1_aimu_cmdq.h)/[src/aimu_cmdq.c](src/aimu_cmdq.c) add an optional real-execution hook (default off, so every raw `att1_aimu_cmdq` test is unaffected); [src/aimu_conformance.c](src/aimu_conformance.c) installs it on the in-process endpoint, maintaining a resident-tensor registry populated by `LOAD_TENSOR_TILE` and executing `EXEC_MATMUL`/`EXEC_RMSNORM`/`EXEC_ROPE`/`EXEC_FFN` against real host buffers with the same `att1_math.h`/`att1_quant.h` primitives the CPU backends call directly; [src/backend_pcie.c](src/backend_pcie.c) auto-registers weight/norm pointers by identity and implements `softmax_f32` locally; [tests/test_backend_pcie.c](tests/test_backend_pcie.c) proves single-op correctness plus three-step multi-position transformer-block decode matching cpu-f32 exactly and cpu-q8/cpu-q4 within tolerance.


- Milestone 157: Register map freeze (v1.0) — BAR0 layout, doorbell, status, capability, counter registers; frozen vs. reserved fields policy.
- Milestone 158: Command packet and completion schema freeze (v1.0) — including `att1_status_t` error/result-code mapping.
- Milestone 159: DMA descriptor and replay schema freeze (v1.0) — compatibility contract in `docs/schema_compatibility.md`.
- Milestone 160: Fabric packet and barrier semantics freeze (v1.0) — packet types, queue-full behavior, barrier semantics, counter name set; resolve barrier mechanism question (M93 §8.15-4).
- Milestone 161: Conformance test suite for frozen interfaces (complete) — substrate-independent harness (in-process sim, emulator, FPGA).

Gate 1: satisfied — all four freezes ratified; conformance suite green on the in-process simulator.

## Stage 2: Out-of-Process Emulated PCIe Endpoint

- Milestone 162: `att1-aimu-endpoint` process skeleton (complete) — separate process owning tile memory behind a Unix-domain-socket transport, wrapping the M161 in-process conformance simulator so semantics/counters are identical by construction.
- Milestone 163: `backend_pcie.c` host backend (complete) — new `att1_backend_ops` implementation dispatching via the endpoint transport (`cmd_submit`/`cmd_dispatch_one`/`cmd_poll_completion`) instead of CPU/CUDA calls; validates the backend-swap pattern only (endpoint does not execute `EXEC_*` yet).
- Milestone 164: One-time shard transfer and tensor residency (complete) — `att1_backend_pcie_load_tensor()` transfers each tensor shard host→device exactly once via chunked frozen v1.0 DMA descriptors and rejects re-loads of an already-resident `tensor_id`, enforcing the M93 §8.12 "weights never re-read" contract; residency counters expose transfer/rejection accounting.
- Milestone 165: Device-local KV-MMU in the endpoint (complete) — full session lifecycle and counters.
- Milestone 166: Single-tile emulated decode end to end (complete) — the in-process endpoint now really executes `EXEC_MATMUL`/`EXEC_RMSNORM`/`EXEC_ROPE`/`EXEC_FFN` (via an optional, default-off cmdq exec hook); `backend_pcie` matches cpu-f32 exactly and cpu-q8/cpu-q4 within tolerance across a multi-step transformer-block decode; existing smoke tests unmodified.
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
