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
