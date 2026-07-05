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
| [PHASE2_PLAN.md](PHASE2_PLAN.md) | Phase 2 roadmap (M154–M184): stages, gates, Definition of Done, frozen entry baseline, risks |
| [PHASE1_TO_PHASE2_GAP_AUDIT.md](PHASE1_TO_PHASE2_GAP_AUDIT.md) | Phase 1 → Phase 2 gap audit (M156): requirements-traceability table for M93 §8 vs. M103–M155 deliverables |
| [aimu_register_map.md](aimu_register_map.md) | AIMU MMIO register map: BAR0 layout, control/status registers, DMA descriptors; **wire/register contract frozen v1.0 at M157** (§1.6), **in-process DMA simulator model frozen v1.0 at M159** (§15.7) |
| → [AIMU_INTRINSICS_OPERATIONS_REFERENCE.md](AIMU_INTRINSICS_OPERATIONS_REFERENCE.md) | See Reference Manuals section: AIMU command packet reference, EXEC/LOAD/KV/FABRIC/TRACE semantics, MMIO interaction, replay tools |
| [aimu_pcie_prototype_review.md](aimu_pcie_prototype_review.md) | AIMU PCIe prototype milestone history and review: M1–M139 entries |
| [aimu_phase3_go_no_go.md](aimu_phase3_go_no_go.md) | Phase 3 go/no-go criteria: readiness gates for hardware prototype direction |
| [comps.md](comps.md) | Architectural comparisons to NVIDIA (Blackwell/Rubin), Apple M4 Neural Engine, and production GPU/TPU/ANE systems (M154 context) |
| [fpga_feasibility.md](fpga_feasibility.md) | FPGA feasibility notes: scope, non-goal status, reference only |
| [phase3_bom_board_options.md](phase3_bom_board_options.md) | Phase 3 BOM and board options: reference exploration for future hardware |

---

## AIMU PCIe and Control Plane

| Document | Description |
|----------|-------------|
| [aimu_pcie_command_requirements.md](aimu_pcie_command_requirements.md) | PCIe command plane requirements: host↔AIMU command protocol, queue design; **frozen v1.0 at M158** (§1.5) |

---

## Fabric Routing

| Document | Description |
|----------|-------------|
| [aimu_fabric_routing.md](aimu_fabric_routing.md) | Fabric routing: packet format, route report schema, route planning; **frozen v1.0 at M160** (§16) |

---

## Tensor Execution Planning

| Document | Description |
|----------|-------------|
| [tensor_execution_plan.md](tensor_execution_plan.md) | Tensor execution plan schema: execution phases, command ordering, replay format |

---

## Schema Compatibility

| Document | Description |
|----------|-------------|
| [schema_compatibility.md](schema_compatibility.md) | Schema versioning policy: supported versions, required fields, unknown-field rules, hostile-input policy; **replay compatibility contract extended at M159** (§§1, 9, 12) |

---

## Testing, CI, and CUDA Signoff

| Document | Description |
|----------|-------------|
| [testing.md](testing.md) | Testing guide: `make test`, `make regression`, CI, CUDA signoff, comparison table |
| [CUDA_VALIDATION_PLAN.md](CUDA_VALIDATION_PLAN.md) | CUDA validation policy, manual signoff commands, paths table, self-hosted runner plan |
| [CUDA_SIGNOFF_M155.md](CUDA_SIGNOFF_M155.md) | M155 CUDA signoff report: RTX 3090 host, CUDA commands, backend matrix, regression summary |

`tests/test_aimu_conformance.c` (M161) is a substrate-independent conformance
suite for the frozen register/command/DMA/fabric interfaces, run through the
abstract endpoint in `include/att1_aimu_conformance.h` /
`src/aimu_conformance.c`; `compiler/check_aimu_conformance.py` cross-checks
the frozen docs against the shipped C headers.

`tools/att1-aimu-endpoint.c` (M162) is an out-of-process endpoint daemon
skeleton that serves the M161 conformance ops over a Unix domain socket
(`include/att1_aimu_endpoint_protocol.h` / `src/aimu_endpoint_protocol.c`);
`src/aimu_endpoint_client.c` provides a matching socket-backed conformance
client, and `tests/test_aimu_endpoint.c` validates identical
register/command/DMA/fabric semantics and counters over the transport.

`src/backend_pcie.c` (M163) is an `att1_backend_ops` implementation
(`att1_backend_pcie_create()`) that dispatches every operation through a
caller-owned `att1_aimu_conformance_endpoint` (in-process or M162
socket-backed) instead of CPU/CUDA calls, validating the backend-swap
pattern proven by CUDA; `tests/test_backend_pcie.c` exercises the
submit/dispatch/poll transport round trip. Compute execution over the
transport is not yet implemented (deferred to M166).

`att1_backend_pcie_load_tensor()` (M164, same file) implements the M93
§8.12 one-time shard-transfer contract: it moves a tensor's bytes
host→device via one or more frozen v1.0 (M159) DMA descriptors, marks the
`tensor_id` resident, and rejects a second load for the same `tensor_id`
with `ATT1_ERR_STATE` without resubmitting any transfer, enforcing that
weights are never re-read by the host once resident;
`att1_backend_pcie_residency_counters` / `att1_backend_pcie_get_residency_counters()`
expose transfer/rejection accounting and `att1_backend_pcie_tensor_is_resident()`
lets callers query residency directly.

---

## Reference Manuals

| Document | Description |
|----------|-------------|
| [ATT1_REFERENCE_MANUAL.md](ATT1_REFERENCE_MANUAL.md) | ATT-1 Reference Manual (M146): artifact format, runtime API, backends, inference modes, CLI tools, conversion flow, planning pipeline, testing policy, error codes, non-goals |
| [AIMU_INTRINSICS_OPERATIONS_REFERENCE.md](AIMU_INTRINSICS_OPERATIONS_REFERENCE.md) | AIMU Intrinsics and Operations Reference Manual (M147): command packets, EXEC/LOAD/VALIDATE/KV/FABRIC/TRACE semantics, DMA descriptor model, MMIO/register map, dtype/op bitmasks, tensor placement implications, result codes, replay tools, implemented vs future |

---

## Release Readiness

| Document | Description |
|----------|-------------|
| [RELEASE_READINESS.md](RELEASE_READINESS.md) | Release and outside-review checklist: hygiene, build status, artifact policy, security, patent handling |
| [RELEASE_CANDIDATE_M150.md](RELEASE_CANDIDATE_M150.md) | M150 release candidate checkpoint: validation baselines, capability summary, known limitations, review decision |
| [EXTERNAL_REVIEW_PACKAGE.md](EXTERNAL_REVIEW_PACKAGE.md) | External reviewer package checklist: included/excluded materials, pre-review validation commands, reviewer quick-start, suggested focus areas, safe sharing notes |

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
