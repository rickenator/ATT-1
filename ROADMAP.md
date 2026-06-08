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
