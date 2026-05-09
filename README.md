# ATT-1 / Aniviza Tensor Tile

ATT-1 prototypes an AIMU fabric: a tiled tensor artifact format and C11 runtime
where model tensor space is partitioned into memory-owned tiles. Each future
AIMU — Application-specific Inference Memory Unit — owns local tensor memory
and executes programmable inference operations near memory, while the fabric
coordinates routing, synchronization, reductions, and traceable execution.

ATT-1 is a phase-1 software simulator for a future LLM inference ASIC
architecture. The target architecture is a programmable tensor tile with local
model memory, a KV-cache MMU, and a packetized fabric between tiles.

Phase 1 focuses on simulation primitives and architecture exploration in plain
C with no external dependencies. Phase 2 points toward a PCIe card direction:
host runtime integration, driver-facing command queues, model placement, and
multi-tile scheduling against hardware-like constraints.

## Documentation

| Document | Purpose |
|----------|---------|
| [docs/testing.md](docs/testing.md) | Build, test, and CI guide |
| [docs/CUDA_VALIDATION_PLAN.md](docs/CUDA_VALIDATION_PLAN.md) | CUDA signoff policy and self-hosted runner plan |
| [docs/RELEASE_READINESS.md](docs/RELEASE_READINESS.md) | Release and outside-review readiness checklist |
| [docs/OPERATION_LOG.md](docs/OPERATION_LOG.md) | Milestone history and next steps |
| [DESIGN.md](DESIGN.md) | Architecture overview |

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
