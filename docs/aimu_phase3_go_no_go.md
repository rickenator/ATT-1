# ATT-1 AIMU/PCIe Phase 3 Prototype: Go/No-Go Review (Milestone 120)

**Date:** 2026-05-08
**Revision:** 1.0 (M120)
**Status:** CONDITIONAL GO

---

## 1. Executive Summary

**Recommendation: CONDITIONAL GO — proceed with Phase 3 software/userspace prototype before any
FPGA or hardware commitment.**

The ATT-1 simulator stack through M119 has established a complete, coherent control-plane
simulation: placement estimation, advisory remediation, AIMU command-plan generation, C11
in-process control-plane simulation (command queue, device discovery, DMA, trace, MMIO
register file, host harness), fabric route planning, route validation, fabric bandwidth and
latency simulation, and an integrated end-to-end planning pipeline.

However, several critical areas remain unproven:

- No real PCIe endpoint or MMIO hardware exists.
- No FPGA prototype has been built or evaluated.
- No physical AIMU tile memory, fabric link, DMA engine, or power rail exists.
- No hardware-backed inference has been run.
- No tensor-level placement has been executed against a real model.

The recommended next step is **Option B: userspace MMIO/register-file emulator with
command-plan replay**, followed by fabric route replay simulation.  FPGA prototyping (Option C)
should be deferred until command/control/fabric models are stable and the userspace emulator
demonstrates end-to-end planning fidelity.  Custom silicon (Option D/E) remains premature.

---

## 2. Current Proven Capabilities

The following have been implemented, tested, and committed through M119.

| Capability | Milestone(s) | Artifact |
|---|---|---|
| Versioned `.att1` artifact loading | M6, M32, M35–M40 | `src/model_loader.c`, `att1-inspect` |
| Hostile-input validation (malformed/truncated artifacts) | M6 | `src/model_loader.c` |
| CPU f32 correctness reference | M2, M7 | `backend_cpu_f32.c`, inference pipeline |
| CPU q8 backend | M10, M11, M27 | `backend_cpu_q8.c`, `matmul_q8.c` |
| CPU q4 backend | M81–M85 | `convert_llama_to_att1.py --weight-format q4` |
| CUDA f32 backend | M14–M20 | `backend_cuda.c`, cuBLAS primitives |
| CUDA q8 backend | M23, M28 | `cuda_backend_matmul_q8` |
| CUDA q4 backend | M87–M91 | `cuda_backend_matmul_q4xf32` |
| Single-tile inference | M7, M20 | `att1_infer_decode_token` |
| Cluster inference | M8, M22 | `att1_cluster_infer_t`, `run_cluster` |
| Tokenizer / pretokenized input flow | M7, M95 | `src/tokenizer.c`, `att1-bench` byte/external modes |
| Prefill vs. decode trace split | M95 | `att1-bench` prefill/decode counter output |
| Backend comparison and cross-validation reports | M29, M92 | `tests/test_backend_matrix.c`, `compiler/backend_comparison_report.py` |
| q4 tolerance and trace-diff reporting | M84, M94 | `compare_att1_to_source.py`, `trace_diff.py` |
| Tile memory capacity and bandwidth estimator | M96 | `att1-size`, `[tile_capacity_estimate]` / `[fabric_bandwidth_estimate]` |
| Tensor-level placement report schema and emission | M98, M100 | `docs/tensor_placement_report.md`, `att1-size --placement-report-json` |
| Placement report validation | M99 | `compiler/validate_tensor_placement_report.py` |
| Advisory placement remediation | M101 | `compiler/propose_tensor_placement.py` |
| Placement scenario generation (16/32/64/128 GiB SKUs) | M102 | `compiler/propose_tensor_scenarios.py` |
| AIMU command packet model | M103 | `docs/aimu_pcie_command_requirements.md` |
| AIMU MMIO register map (64 KiB BAR0) | M104 | `docs/aimu_register_map.md` |
| C11 command queue simulator | M105 | `include/att1_aimu_cmdq.h`, `src/aimu_cmdq.c` |
| C11 device discovery simulator | M106 | `include/att1_aimu_device.h`, `src/aimu_device.c` |
| C11 DMA descriptor simulator | M107 | `include/att1_aimu_dma.h`, `src/aimu_dma.c` |
| C11 trace and counter integration | M108 | `include/att1_aimu_trace.h`, `src/aimu_trace.c` |
| Placement-to-command-plan mapper | M109 | `compiler/map_placement_to_commands.py` |
| Minimal viable prototype design review | M110 | `docs/aimu_pcie_prototype_review.md` |
| C11 MMIO/register-file simulator (64 KiB BAR0) | M111 | `include/att1_aimu_mmio.h`, `src/aimu_mmio.c` |
| C11 host control-plane integration harness | M112 | `include/att1_aimu_host.h`, `src/aimu_host.c` |
| AIMU placement-command replay tool | M113 | `tools/att1-aimu-replay.c`, `compiler/replay_aimu_command_plan.py` |
| Fabric routing requirements and schema | M114, M115 | `docs/aimu_fabric_routing.md` §1–§11 |
| Fabric route validator | M116 | `compiler/validate_fabric_routes.py` |
| Command-plan-to-fabric-route mapper | M117 | `compiler/map_commands_to_fabric_routes.py` |
| Fabric bandwidth and latency simulator | M118 | `compiler/simulate_fabric_bandwidth.py` |
| Integrated end-to-end planning pipeline | M119 | `compiler/run_aimu_planning_pipeline.py` |

**Test baseline:** 342 PASS 0 FAIL (CPU-only host). CUDA signoff on a CUDA-capable host per
`docs/CUDA_VALIDATION_PLAN.md`.

---

## 3. Current Unproven Areas

| Gap | Notes |
|---|---|
| No real PCIe endpoint | BAR0 is a flat `uint32_t regs[16384]` array; no hardware BAR |
| No Linux kernel driver | No `/dev/att1`, no `pci_driver`, no BAR mapping via `ioremap` |
| No FPGA prototype | No AXI-Lite or PCIe IP core has been instantiated |
| No real DMA engine | DMA descriptor simulator validates descriptors; no actual data movement |
| No real hardware fabric | Fabric link is a counter model; no physical interconnect |
| No physical AIMU tile memory | Tile memory capacity is an analytic estimate |
| No real power or thermal measurements | All bandwidth/latency numbers are analytic projections |
| No tensor-level placement execution | Placement is advisory and command-plan only; no runtime enforces it |
| No hardware-backed inference | All inference runs on CPU or CUDA host; no AIMU tile runs ops |
| No contention or NoC arbitration model | Fabric bandwidth model assumes linear utilization; no queuing theory |
| No real fence/completion hardware | Fence IDs are software counters in the M112 harness |

---

## 4. Prototype Options

### Option A — Pure software AIMU/PCIe simulator extension (current state)

**What it proves:** Control-plane logic, command sequencing, placement planning, fabric route
planning, bandwidth feasibility.  All of the above is already proven through M119.

**Complexity:** Minimal.  No new hardware required.

**Risk:** Low engineering risk.  Does not prove any hardware path.

**Cost class:** Zero additional hardware cost.

**Recommendation:** Already complete as of M119.  No new work required under Option A alone.
Further Option A work risks over-engineering the software model without hardware validation.

---

### Option B — Userspace MMIO/register-file emulator

**What it proves:** That the M104 BAR0 register map, M105 command queue, M106 device discovery,
M107 DMA descriptors, M108 trace, and M111 MMIO simulator work correctly as a userspace
emulator accessible via a file descriptor or shared-memory interface.  Exercises the full
probe→enumerate→submit→drain→snapshot cycle without a kernel driver.

**Complexity:** Moderate.  Requires a thin shim (mmap'd file or Unix domain socket) that maps
BAR0 offsets to the M111 MMIO simulator.  The C11 simulator already exists; the shim is the
new piece.

**Risk:** Low.  No kernel driver, no PCIe hardware.  All access is userspace.

**Cost class:** Zero additional hardware cost.

**Recommendation:** **RECOMMENDED NEXT STEP.**  Implement M121–M122 as the first Phase 3 target.
This is the lowest-risk path to verifying that the host control-plane software stack is
correct before committing to FPGA or hardware.

---

### Option C — FPGA PCIe endpoint prototype

**What it proves:** That BAR0 register access works over a real PCIe link.  Exercises PCIe
enumeration, BAR mapping, and DMA from a real host driver (kernel or userspace via VFIO).

**Complexity:** High.  Requires FPGA board selection, PCIe IP core instantiation, AXI-Lite
bridge, RTL implementation of at least the M104 register map, and host-side driver or VFIO
code.  RTL complexity alone is several months of engineering.

**Risk:** High.  FPGA supply chain, PCIe IP core licensing, timing closure, and driver
compatibility add schedule risk.

**Cost class:** USD 5,000–30,000 for board + toolchain licenses depending on FPGA vendor.

**Recommendation:** **DEFER until Option B is stable.**  FPGA is the natural next step after
the userspace emulator validates the command/control model.  Do not start FPGA work until
the M104 register map and M105–M108 simulator suite are frozen.

---

### Option D — Custom PCIe board with AIMU controller concept

**What it proves:** That a purpose-built PCIe card can host a minimal AIMU controller and
local tensor memory.  Would prove DMA, BAR access, and tile memory from real hardware.

**Complexity:** Very high.  Requires board bring-up, power delivery, PCIe PHY, local memory
controller, and AIMU firmware.

**Risk:** Very high.  Hardware bugs, silicon errata, and thermal issues add unpredictable schedule.

**Cost class:** USD 50,000–500,000+ for NRE, depending on scope.

**Recommendation:** **NOT RECOMMENDED at this stage.**  The command/control model is not stable
enough to commit to custom board NRE.  Revisit after FPGA prototype (Option C) is complete.

---

### Option E — Custom silicon / ASIC path

**What it proves:** Full AIMU performance, power efficiency, and memory bandwidth at production
scale.

**Complexity:** Extremely high.  Full RTL, verification, DFT, physical design, tape-out, and
bring-up.

**Risk:** Extremely high.  Multi-year timeline, substantial capital.

**Cost class:** USD 1M–50M+ depending on process node and scope.

**Recommendation:** **NOT RECOMMENDED.**  No prototype requirements are stable enough to commit
silicon.  Defer indefinitely until Option C/D are complete and model/placement/command
interfaces are frozen.

---

## 5. Recommended Path

1. **Continue with Option B (userspace MMIO emulator).**  The M111 MMIO simulator and M112
   host harness are the right foundation.  The next step is a thin shim (mmap or Unix socket)
   exposing BAR0 to a test harness that is not an in-process unit test.

2. **Replay command plans against the MMIO emulator (M122).**  Drive the M109 command plan
   through the M112 harness via the userspace interface and verify that the M113 replay tool
   produces the same pass/fail outcomes.

3. **Add fabric route replay simulation (M123).**  Wire the M117 route mapper output into a
   replay loop that drives the M116 validator and M118 bandwidth simulator deterministically.

4. **Freeze the command/control model (M124–M125).**  Once replay against the userspace emulator
   is stable, freeze the M104 register map, M103 command packet format, and M109 command plan
   schema.  Any future changes require a version bump.

5. **Defer FPGA until M124–M125 are complete.**  Publish an FPGA feasibility note (M126) that
   surveys board options, AXI-Lite mapping, and PCIe IP core availability, but do not commit
   FPGA resources until the software model is frozen.

6. **Keep 16/32/64/128 GiB tile SKU modeling.**  The M102 scenario generator and M96 capacity
   estimator already support all four SKUs.  These should remain the canonical planning inputs
   for all subsequent milestones.

---

## 6. Go/No-Go Criteria

The Phase 3 prototype is ready to advance to Option C (FPGA) when all of the following hold:

| Criterion | Gate tool / check | Status |
|---|---|---|
| Placement reports produce feasible scenarios for target models | `propose_tensor_scenarios.py` exits 0; at least one PASS scenario | Proven (M102) |
| Command plan replay passes | `replay_aimu_command_plan.py --plan ... --report-json`; `status=pass`, `failed_commands=0` | Proven (M113) |
| Fabric simulation passes bandwidth targets | `simulate_fabric_bandwidth.py`; `aggregate.status=PASS` or WARN | Proven (M118) |
| Trace/counter outputs deterministic | M108 snapshot: same seed → same counters across runs | Proven (M108) |
| No silent fallback in backend matrix | `test_backend_matrix.c`; zero `ATT1_ERR_OK` for known-bad input | Proven (M29, M41, M90) |
| No hostile-input validation gaps | `validate_tensor_placement_report.py` and `att1-inspect` reject all malformed inputs | Proven (M6, M99) |
| Public artifacts remain out of repo | `git ls-files` contains no public-weight `.att1` files | Enforced (standing rule) |
| CUDA signoff completed when CUDA behavior touched | Manual CUDA validation per `docs/CUDA_VALIDATION_PLAN.md` | Enforced per checklist |
| Userspace MMIO emulator passes end-to-end command cycle | M121 target | **NOT YET MET** |
| Command-plan replay against MMIO emulator passes | M122 target | **NOT YET MET** |
| Fabric route replay simulation passes | M123 target | **NOT YET MET** |
| Command/control interface frozen (version bump policy in place) | M124 target | **NOT YET MET** |

**FPGA go/no-go gate:** All rows must show "Met" before FPGA board commitment.

---

## 7. Hardware Economic Notes

These are engineering-level planning estimates.  They are not cost commitments, market
forecasts, or financial projections.

| SKU | Tile memory | Model fit | KV pressure | Fabric pressure | Notes |
|---|---|---|---|---|---|
| 16 GiB | ~16 GiB per tile | Tight for large models; q4 helps | High per tile if context is long | High: more tiles needed → more cross-tile traffic | Lowest entry cost; most tiles required |
| 32 GiB | ~32 GiB per tile | Balanced; 7B–30B range fits in 1–4 tiles | Moderate | Moderate | Recommended development SKU |
| 64 GiB | ~64 GiB per tile | Better fit for 30B–70B models | Lower per tile | Lower; fewer tiles needed for a given model | Better production large-model fit |
| 128 GiB | ~128 GiB per tile | Long-context or very large models | Lowest per tile | Lowest cross-tile | Premium SKU; high memory cost |

**AIMU ASIC note:** The same AIMU ASIC should ideally support runtime-discoverable memory
capacity across SKUs via the M104 `TILE_MEMORY_CAPACITY_{LOW,HIGH}` register pair.
The M106 device simulator already supports configurable per-tile capacity at object creation.
The frozen register map must carry this field without breaking changes across SKUs.

---

## 8. Risks

| Risk | Severity | Mitigation |
|---|---|---|
| Memory market volatility (DRAM pricing, supply) | Medium | SKU range (16–128 GiB) provides flexibility; avoid hard memory commitment until FPGA prototype |
| PCIe latency hiding (command round-trip cost) | Medium | M119 pipeline identifies latency-sensitive paths; defer PCIe latency model refinement to M121–M122 |
| Fabric routing complexity (topology, congestion, arbitration) | High | Current model is linear; add queuing-theory model in M123; do not commit FPGA until model is stable |
| KV-cache pressure under long-context / multi-session | High | M102 scenario generator models KV bytes per tile; validate with real model sizes before FPGA |
| q4 accuracy and token divergence (workload-specific) | Medium | M76–M91 tolerance tables; revalidate on target model; do not remove CPU f32 reference path |
| Thermal and power unknowns (no real hardware) | High | All estimates are analytic; no thermal model exists; do not commit board NRE until power envelope is estimated |
| Placement model optimistic assumptions (no fragmentation, no DRAM overhead) | Medium | Add 10–20% margin to tile memory estimates in M102 scenarios; fragmentation simulator deferred to M124 |
| Patent-sensitive hardware details must remain local and private | High | Do not describe specific circuit implementations in any public doc or public commit; this review uses only architectural descriptions |

---

## 9. Decision

**CONDITIONAL GO.**

- Phase 3 is authorized to proceed with **Option B (userspace MMIO/register-file emulator)**
  and associated command-plan and fabric replay simulation.
- Phase 3 is **NOT authorized** to commit FPGA board resources until the go/no-go criteria
  in §6 are fully met (specifically M121–M124 targets).
- Custom board NRE (Option D) and ASIC (Option E) remain on hold indefinitely.
- The software simulation path (Option A) is complete and requires no further major investment.

---

## 10. Next Milestone Proposals

| Milestone | Title | Scope |
|---|---|---|
| M121 | Userspace MMIO/register-file emulator workflow | Thin shim exposing M111 MMIO simulator via mmap'd file or Unix domain socket; probe→enumerate→submit→drain→snapshot via file descriptor; no kernel driver; test harness outside unit test suite |
| M122 | Command-plan replay against MMIO emulator | Wire M113 `att1-aimu-replay` and `replay_aimu_command_plan.py` to drive the M121 userspace emulator; verify probe/enumerate/submit/drain/snapshot sequence produces deterministic results; round-trip JSON report |
| M123 | AIMU fabric route replay simulator | Drive M117 route mapper output through a replay loop that invokes M116 validator and M118 bandwidth simulator; add per-hop congestion model; verify deterministic outcomes; report JSON |
| M124 | Tile memory allocator simulator | AIMU-local slab allocator for tensor/KV/staging/trace regions; fragmentation reporting; integrates with M107 DMA descriptor address range checks; freeze memory allocation interface |
| M125 | Tensor-level placement execution plan | Document and simulate how AIMU EXEC_* commands would be dispatched after LOAD/VALIDATE; define the tensor-execution command sequence per layer per tile; advisory/simulation only |
| M126 | FPGA feasibility research notes | Document register-map to AXI-Lite mapping options; survey PCIe IP core availability (Xilinx, Intel/Altera); estimate RTL complexity for M104 register set; resource utilization ballpark; go/no-go criteria for Option C; documentation only |
| M127 | Phase 3 prototype BOM and board options review | Compare development board options for FPGA prototype; supply chain and timeline estimates; identify minimum viable FPGA for M104 register map; documentation only |

---

## 11. Non-Goals

- No patent claim language of any kind.
- No production ASIC design or tape-out.
- No Linux kernel driver (PCIe or otherwise).
- No real PCIe endpoint or BAR0 MMIO hardware implementation.
- No new inference features or model format changes.
- No CUDA kernels or CUDA backend changes.
- No `.att1` binary format changes.
- No public model weights or generated public `.att1` artifacts committed to Git.
- No `__pycache__` or `.pyc` files tracked.
