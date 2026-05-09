# ATT-1 / Aniviza Tensor Tile

ATT-1 is a phase-1 C11 software simulator for a future LLM inference
architecture built around programmable tensor tiles. Model tensor space is
partitioned across memory-owned tiles; each tile executes quantized inference
operations near local memory while a packetized fabric handles routing,
synchronization, reductions, and traceable execution. The runtime supports
f32, q8, and q4 quantized inference in single-tile and multi-tile cluster
configurations, with CPU and CUDA backends, a KV-cache MMU, a tokenizer, and
a schema-validated planning/control-plane simulator.

## AIMU

An AIMU — Application-specific Inference Memory Unit — is the target tile
architecture: a programmable unit with local model memory, a hardware KV-cache
MMU, and a packetized fabric interface. Phase 1 simulates AIMU command queues,
MMIO register access, DMA transfers, device/host interaction, and fabric
routing entirely in userspace C with no hardware dependency. Phase 2 points
toward a PCIe card form factor: host driver command queues, multi-tile
scheduling, and placement against hardware-like memory constraints.

## Current Status

| Capability | Status |
|------------|--------|
| CPU f32 inference (single-tile, cluster) | Complete |
| CPU q8 quantized inference | Complete |
| CPU q4 quantized inference | Complete |
| CUDA f32/q8/q4 inference | Implemented; manual RTX 3090 signoff pending |
| AIMU command queue / MMIO simulation | Complete |
| AIMU DMA / host / device simulation | Complete |
| Fabric routing simulation | Complete |
| Command and fabric replay (M132) | Complete |
| Tensor placement reports and scenarios | Complete |
| Schema-validated planning outputs | Complete (M134/M135) |
| CPU-only GitHub Actions CI | Complete (M137) |
| CUDA signoff (manual, RTX 3090) | Plan in place (M138); signoff pending |

## Quick Build and Test

```sh
# Build
make clean && make

# C unit and integration tests (781 PASS 0 FAIL)
make test

# Full regression: golden baselines + schema + hostile-input + pipeline smoke
make regression
```

For CUDA builds:

```sh
make clean && make CUDA=1 && make test CUDA=1
```

## Tiny Fixture Demo

The repository includes a deterministic end-to-end demo script that exercises
the binary loader, inference bench, and full planning/control-plane pipeline
using only checked-in tiny/dummy fixtures.  **No CUDA, no model downloads,
no network access required.**

```sh
./tools/demo_tiny_att1.sh
```

Options:

| Flag | Effect |
|------|--------|
| `--skip-build` | Skip `make clean && make`; use existing `build/` |
| `--keep-temp` | Keep temp files in `/tmp/att1-demo-XXXXXX` after exit |
| `--verbose` | Print every command and its full output |

The demo covers: model inspection, capacity planning, cpu-f32 / cpu-q8
single and cluster inference, placement report generation and schema
validation, placement advisory, command-plan mapping, fabric-route mapping
and validation, fabric replay, and the 6-stage integrated execution/replay
pipeline.

Output not representative of real inference performance — the dummy fixture
has d_model=4, 2 layers, and 2 attention heads.

## Documentation

See [docs/INDEX.md](docs/INDEX.md) for the full documentation map.

| Category | Key document |
|----------|-------------|
| Architecture overview | [DESIGN.md](DESIGN.md) |
| Build, test, and CI | [docs/testing.md](docs/testing.md) |
| Model format | [docs/model_format.md](docs/model_format.md) |
| Quantization | [docs/quantization.md](docs/quantization.md) |
| Real model conversion | [docs/real_model_conversion.md](docs/real_model_conversion.md) |
| Tokenizer / pretokenized input | [docs/tokenizer_metadata.md](docs/tokenizer_metadata.md) |
| Tensor placement reports | [docs/tensor_placement_report.md](docs/tensor_placement_report.md) |
| AIMU architecture | [docs/aimu_architecture.md](docs/aimu_architecture.md) |
| AIMU PCIe / control plane | [docs/aimu_pcie_command_requirements.md](docs/aimu_pcie_command_requirements.md) |
| Fabric routing | [docs/aimu_fabric_routing.md](docs/aimu_fabric_routing.md) |
| Tensor execution plan | [docs/tensor_execution_plan.md](docs/tensor_execution_plan.md) |
| Schema compatibility | [docs/schema_compatibility.md](docs/schema_compatibility.md) |
| CUDA signoff policy | [docs/CUDA_VALIDATION_PLAN.md](docs/CUDA_VALIDATION_PLAN.md) |
| Release / review readiness | [docs/RELEASE_READINESS.md](docs/RELEASE_READINESS.md) |
| External reviewer package | [docs/EXTERNAL_REVIEW_PACKAGE.md](docs/EXTERNAL_REVIEW_PACKAGE.md) |
| Milestone history | [docs/OPERATION_LOG.md](docs/OPERATION_LOG.md) |

## Artifact Policy

- Tiny deterministic fixtures in `compiler/fixtures/` and `models/dummy/`
  are tracked in Git.
- Public model weights, generated `.att1` files from real models, and
  external tokenizer/vocab assets must stay outside the repository.
- No `__pycache__` or `.pyc` files are tracked.

## Non-Goals

- No production ASIC design or tape-out.
- No Linux kernel driver.
- No real PCIe endpoint or BAR0 MMIO.
- No FPGA RTL or synthesis.
- No patent claim language of any kind.

## Repository Layout

```
include/     Public C headers
src/         Runtime implementation
simulator/   Fabric and tile simulator subsystems
compiler/    Model tools, regression runners, fixture validators
tests/       Unit, integration, smoke, and regression tests
tools/       Developer utilities (inspect, size, bench)
examples/    Minimal C programs using the runtime API
models/      Tiny fixture models only (no public weights)
docs/        Documentation (see docs/INDEX.md)
```

## Future Manuals

The following reference documents are planned but not yet written:

- **ATT-1 Reference Manual** — complete API, .att1 format, runtime
  semantics, backend selection matrix, KV-cache MMU, quantization pipeline.
- **AIMU Intrinsics and Operations Reference Manual** — AIMU command set,
  MMIO register map, fabric packet format, DMA protocol, execution phases.
