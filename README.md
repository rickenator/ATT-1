[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/rickenator/ATT-1)

# ATT-1 / Aniviza Tensor Tile

ATT-1 is a software prototype for a memory-centric inference architecture.
Instead of treating memory as passive storage behind a GPU or TPU, ATT-1 models
the idea that large neural-network tensor space can be divided into owned tensor
tiles. Each tile carries metadata describing shape, dtype, quantization,
placement, validation, routing, and execution constraints. In today's prototype
those tiles are simulated by C11 runtime components, CPU/CUDA backends, shard
metadata, placement reports, command queues, and fabric-route tools. In the
intended future architecture those same concepts map onto AIMU tiles:
programmable near-memory inference units that own local tensor memory and
participate in a fabric. The core doctrine: **move less data by executing closer
to where the tensor data lives.**

Phase 2 closed at M175 with an initial **HOLD** decision on the FPGA path, then
the M175 green evidence packet passed. The gate is now **GO** for the constrained
M176-M181 control-plane prototype path: board/BOM work may proceed, but tensor
math, custom kernel-driver work, and public model artifacts remain out of
scope. See [docs/M175_GREEN_EVIDENCE.md](docs/M175_GREEN_EVIDENCE.md),
[docs/FPGA_GATE_REVIEW_M175.md](docs/FPGA_GATE_REVIEW_M175.md), and
[docs/PHASE2_CLOSURE.md](docs/PHASE2_CLOSURE.md).

M176 and later hardware work are treated as Phase 3 private work for IP
reasons. Public ATT-1 remains the software/reference baseline; hardware
implementation details belong in the private `ATT-1-HW` repository. See
[docs/PHASE3_PRIVATE_BOUNDARY.md](docs/PHASE3_PRIVATE_BOUNDARY.md).

The runtime supports f32, q8, and q4 quantized inference in single-tile and
multi-tile cluster configurations, with CPU and CUDA backends, a KV-cache MMU,
a tokenizer, and a schema-validated planning/control-plane simulator.

## AIMU

An AIMU — Application-specific Inference Memory Unit — is the target tile
architecture: a programmable unit with local model memory, a hardware KV-cache
MMU, and a packetized fabric interface. Phase 1 simulates AIMU command queues,
MMIO register access, DMA transfers, device/host interaction, and fabric
routing entirely in userspace C with no hardware dependency. Phase 2 points
toward a PCIe card form factor: host driver command queues, multi-tile
scheduling, and placement against hardware-like memory constraints.

## Architecture

ATT-1 separates the model into explicit, validated, executable structure. A
conventional accelerator pipeline often relies on runtime scheduling to decide
where tensors go and how work is split. ATT-1 pushes more of that knowledge
into artifacts, metadata, placement reports, and deterministic traces — enabling
the system to reason about memory capacity, quantization, KV pressure, tile
ownership, fabric traffic, reductions, and command replay before hardware exists.
For example, the current placement tooling can show when a synthetic q4 120B
shape fails on 8×16 GiB tiles while still producing a structurally valid
placement report; that distinction between "valid report" and "feasible
placement" is exactly the kind of engineering signal needed before hardware
design.

The project has matured beyond a simple inference runtime. It now has CPU and CUDA
backends, f32/q8/q4 paths, single and cluster inference, safetensors conversion,
tokenizer/pretokenized input handling, source comparison reports, backend
matrices, tensor placement reports, advisory tools, command-plan mapping,
MMIO/register sketches, command queues, DMA descriptor simulation,
trace/counter integration, and AIMU fabric route planning. The result is an
architecture validation platform: it can test model correctness, estimate memory
and bandwidth pressure, simulate control-plane behavior, and prepare for a
future PCIe/AIMU prototype without prematurely committing to silicon details.

> **In short:** ATT-1 is the executable simulator; AIMU is the future hardware
> concept; the fabric is the movement/reduction layer; placement metadata is the
> bridge between software and silicon. Its advantage is not merely faster math —
> its advantage is making tensor locality, ownership, quantization, validation,
> and routing explicit enough that inference can be planned around memory
> movement instead of being dominated by it.

## Current Status

| Capability | Status |
|------------|--------|
| CPU f32 inference (single-tile, cluster) | Complete |
| CPU q8 quantized inference | Complete |
| CPU q4 quantized inference | Complete |
| CUDA f32/q8/q4 inference | Implemented; CUDA validation complete on RTX 3090 (M155) |
| AIMU command queue / MMIO simulation | Complete |
| AIMU DMA / host / device simulation | Complete |
| Fabric routing simulation | Complete |
| Command and fabric replay (M132) | Complete |
| Tensor placement reports and scenarios | Complete |
| Schema-validated planning outputs | Complete (M134/M135) |
| CPU-only GitHub Actions CI | Complete (M137) |
| CUDA signoff (manual, CUDA-capable host) | Complete on RTX 3090 (M155) |
| Phase 2 hardware gate | M175 green packet passed; M176-M181 control-plane work may begin under constrained scope |

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
| Phase 2 roadmap and closure | [docs/PHASE2_PLAN.md](docs/PHASE2_PLAN.md), [docs/PHASE2_CLOSURE.md](docs/PHASE2_CLOSURE.md) |
| Phase 3 private boundary | [docs/PHASE3_PRIVATE_BOUNDARY.md](docs/PHASE3_PRIVATE_BOUNDARY.md) |
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
| ATT-1 Reference Manual (M146) | [docs/ATT1_REFERENCE_MANUAL.md](docs/ATT1_REFERENCE_MANUAL.md) |
| AIMU Operations Reference (M147) | [docs/AIMU_INTRINSICS_OPERATIONS_REFERENCE.md](docs/AIMU_INTRINSICS_OPERATIONS_REFERENCE.md) |
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
- No tensor-math FPGA RTL or synthesis in the M176-M181 control-plane path.
- No custom Linux kernel driver.
- No public model artifacts or generated real-model `.att1` files in Git.
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

## Reference Manuals

- **ATT-1 Reference Manual** — artifact format, runtime API, backends, inference
  modes, CLI tools, conversion flow, planning pipeline, testing policy, error codes,
  non-goals. See [docs/ATT1_REFERENCE_MANUAL.md](docs/ATT1_REFERENCE_MANUAL.md).

- **AIMU Intrinsics and Operations Reference Manual** — AIMU command packets,
  EXEC/LOAD/VALIDATE/KV/FABRIC/TRACE semantics, DMA descriptor model, MMIO/register
  map, dtype/op bitmasks, tensor placement implications, result codes, replay tools,
  implemented vs future. See [docs/AIMU_INTRINSICS_OPERATIONS_REFERENCE.md](docs/AIMU_INTRINSICS_OPERATIONS_REFERENCE.md).
