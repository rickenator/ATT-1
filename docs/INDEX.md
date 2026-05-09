# ATT-1 Documentation Index (M140)

This index organises all documentation by topic. Every file listed here lives
in `docs/` unless noted otherwise.

---

## Core ATT-1 Runtime

| Document | Description |
|----------|-------------|
| [../DESIGN.md](../DESIGN.md) | Architecture overview: tiled tensor model, AIMU concept, fabric topology, phase-1 scope |
| [inference.md](inference.md) | Single-tile decode path: layer loop, KV-cache integration, output sampling |
| [cluster_inference.md](cluster_inference.md) | Multi-tile cluster inference: layer-sharded decode, fabric synchronisation, reduction |
| [backend.md](backend.md) | Backend abstraction: dispatch table, CPU f32/q8/q4 selection, CUDA opt-in |
| [cuda_backend.md](cuda_backend.md) | CUDA backend skeleton: kernel stubs, build flag `CUDA=1`, device selection |
| [tile_runtime.md](tile_runtime.md) | Tile runtime: pthread-backed simulated tile execution, scheduling, lifecycle |
| [fabric.md](fabric.md) | Fabric simulator: packet routing abstraction, tile-to-tile communication |
| [kv_mmu.md](kv_mmu.md) | Paged KV-cache MMU: page allocator, eviction policy, hardware-shaped design |
| [tracing.md](tracing.md) | Tracing and benchmarking: instrumentation, trace capture, output format |
| [trace_diff.md](trace_diff.md) | Trace diff tool (`compiler/trace_diff.py`): comparing two bench output files |
| [api_ownership_review.md](api_ownership_review.md) | M141 review: header ownership, lifetime, const-correctness, error consistency, ABI notes |

---

## Model Format and Conversion

| Document | Description |
|----------|-------------|
| [model_format.md](model_format.md) | `.att1` binary format: magic, version, section layout, field semantics |
| [real_model_conversion.md](real_model_conversion.md) | Importing an external model into `.att1` format |
| [real_tiny_model_import.md](real_tiny_model_import.md) | Step-by-step tiny model import walkthrough |
| [shard_metadata.md](shard_metadata.md) | Shard metadata section: per-tile memory layout, tensor ownership |

---

## Quantization

| Document | Description |
|----------|-------------|
| [quantization.md](quantization.md) | q8 and q4 quantization pipeline: calibration, packing, group-size conventions |

---

## Tokenizer and Pretokenized Input

| Document | Description |
|----------|-------------|
| [tokenizer_metadata.md](tokenizer_metadata.md) | Tokenizer metadata format, vocabulary encoding, `--input-token-ids` pretokenized path |

---

## Placement, Sizing, and Scenarios

| Document | Description |
|----------|-------------|
| [tensor_placement_report.md](tensor_placement_report.md) | Tensor placement report format: per-tile memory maps, advisory outputs, scenario comparison |

---

## AIMU Architecture

| Document | Description |
|----------|-------------|
| [aimu_architecture.md](aimu_architecture.md) | AIMU tile architecture: command queue, MMIO register map, DMA model, execution phases |
| [aimu_register_map.md](aimu_register_map.md) | AIMU MMIO register map: BAR0 layout, control/status registers, DMA descriptors |
| [aimu_pcie_prototype_review.md](aimu_pcie_prototype_review.md) | AIMU PCIe prototype milestone history and review: M1–M139 entries |
| [aimu_phase3_go_no_go.md](aimu_phase3_go_no_go.md) | Phase 3 go/no-go criteria: readiness gates for hardware prototype direction |
| [fpga_feasibility.md](fpga_feasibility.md) | FPGA feasibility notes: scope, non-goal status, reference only |
| [phase3_bom_board_options.md](phase3_bom_board_options.md) | Phase 3 BOM and board options: reference exploration for future hardware |

---

## AIMU PCIe and Control Plane

| Document | Description |
|----------|-------------|
| [aimu_pcie_command_requirements.md](aimu_pcie_command_requirements.md) | PCIe command plane requirements: host↔AIMU command protocol, queue design |

---

## Fabric Routing

| Document | Description |
|----------|-------------|
| [aimu_fabric_routing.md](aimu_fabric_routing.md) | Fabric routing: packet format, route report schema, route planning |

---

## Tensor Execution Planning

| Document | Description |
|----------|-------------|
| [tensor_execution_plan.md](tensor_execution_plan.md) | Tensor execution plan schema: execution phases, command ordering, replay format |

---

## Schema Compatibility

| Document | Description |
|----------|-------------|
| [schema_compatibility.md](schema_compatibility.md) | Schema versioning policy: supported versions, required fields, unknown-field rules, hostile-input policy |

---

## Testing, CI, and CUDA Signoff

| Document | Description |
|----------|-------------|
| [testing.md](testing.md) | Testing guide: `make test`, `make regression`, CI, CUDA signoff, comparison table |
| [CUDA_VALIDATION_PLAN.md](CUDA_VALIDATION_PLAN.md) | CUDA validation policy, manual signoff commands, paths table, self-hosted runner plan |

---

## Release Readiness

| Document | Description |
|----------|-------------|
| [RELEASE_READINESS.md](RELEASE_READINESS.md) | Release and outside-review checklist: hygiene, build status, artifact policy, security, patent handling |

---

## Milestone History

| Document | Description |
|----------|-------------|
| [OPERATION_LOG.md](OPERATION_LOG.md) | Full milestone log M1–current, known risks, next prompt |

---

## Images and Diagrams

| File | Description |
|------|-------------|
| [ATT1-ARCH.jpg](ATT1-ARCH.jpg) | ATT-1 architecture diagram |
| [concept_board.png](concept_board.png) | Concept board / early design sketch |
