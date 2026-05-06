# ATT-1 Design Overview

ATT-1 models a future LLM inference ASIC built from programmable tensor tiles.
The phase-1 simulator is intended to explore resource placement, request flow,
KV-cache behavior, and packet traffic before hardware exists.

## Tensor Tile

Each tile owns local model memory and executes tile commands. The simulator will
eventually model command issue, local memory pressure, tensor descriptors, and
tile-visible runtime state. Milestone 0 only defines headers and configuration
types.

## Local Model Memory

Model weights are expected to be placed close to the compute tile that consumes
them. The simulator will track capacity, placement, and movement costs rather
than implementing transformer math first.

## KV-Cache MMU

The KV-cache MMU maps logical request and layer KV pages onto simulated physical
storage. It will support placement, migration, eviction policy experiments, and
trace visibility.

## Packetized Fabric

Tiles communicate through a packetized fabric. The fabric will carry command,
tensor, KV, and synchronization traffic with configurable latency and bandwidth
parameters.

## Runtime and Compiler Direction

The runtime will schedule inference requests across tiles. The compiler area
will eventually lower model descriptions into tile programs and placement
metadata.

## Phase-2 PCIe Card Direction

Phase 2 is expected to map simulator concepts onto a PCIe-attached accelerator:
host-visible queues, command buffers, memory windows, telemetry, and validation
traces shared between simulator and hardware.
