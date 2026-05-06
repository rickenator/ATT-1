# ATT-1 / Aniviza Tensor Tile

ATT-1 is a phase-1 software simulator for a future LLM inference ASIC
architecture. The target architecture is a programmable tensor tile with local
model memory, a KV-cache MMU, and a packetized fabric between tiles.

This repository currently implements Milestone 0 only: a strict C11 project
skeleton, a startup banner, logging, basic configuration structures, placeholder
public headers, and a smoke test.

Phase 1 focuses on simulation primitives and architecture exploration in plain
C with no external dependencies. Phase 2 points toward a PCIe card direction:
host runtime integration, driver-facing command queues, model placement, and
multi-tile scheduling against hardware-like constraints.

## Build

```sh
make
```

The main simulator binary is written to:

```text
build/att1-sim
```

## Test

```sh
make test
```

## Clean

```sh
make clean
```

## Layout

```text
include/    Public C headers
src/        Simulator entry point and shared implementation
simulator/  Future simulator subsystems
compiler/   Future model and graph lowering tools
tests/      Unit and smoke tests
examples/   Example simulator inputs
tools/      Developer utilities
docs/       Additional notes
```

Transformer math is intentionally not implemented yet.
