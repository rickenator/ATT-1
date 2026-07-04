# ATT-1 Phase 1 → Phase 2 Gap Audit (Milestone 156)

**Status:** Complete at Milestone 156.
**Revision:** 1.0

This document is the M156 systematic diff of the Phase 2 PCIe/AIMU prototype
requirements (M93 §8, [aimu_architecture.md](aimu_architecture.md) §8)
against what M103–M155 actually delivered. It produces a
requirements-traceability table (requirement → artifact → test → status) and
confirms that the "unproven" items identified at the M120 go/no-go review
remain the correct and complete Phase 2 hardware target list.

---

## 1. Scope and Method

- **Requirements source:** [aimu_architecture.md](aimu_architecture.md) §8
  ("Phase 2 PCIe/AIMU Prototype Requirements (M93)"), specifically §8.2
  (what the prototype must prove), §8.5 (tile responsibilities), §8.6
  (memory budgets), §8.7 (host control-plane requirements), §8.8
  (fabric/interconnect requirements), §8.9 (KV-MMU requirements), and §8.10
  (required counters).
- **Delivery source:** Milestones M103–M155 as recorded in
  [OPERATION_LOG.md](OPERATION_LOG.md), cross-checked against the M120
  "Current Proven Capabilities" table in
  [aimu_phase3_go_no_go.md](aimu_phase3_go_no_go.md) §2 and the M110 design
  review in [aimu_pcie_prototype_review.md](aimu_pcie_prototype_review.md).
- **Status vocabulary:**
  - **proven (software)** — implemented and covered by an automated C11 or
    Python test that runs in `make test` / `make regression`; validated
    against the in-process simulator or userspace MMIO emulator, not
    physical hardware.
  - **partial** — a software model exists but a stated §8 requirement is
    only partially covered (e.g., an analytic estimate rather than a
    measured value).
  - **unproven** — no artifact exists at any level; this is real,
    physical-hardware work reserved for Phase 2 Stage 2+.

This audit does not change scope, does not add or remove milestones, and
does not modify any runtime, CUDA, or `.att1` format behavior. It is a
documentation/traceability exercise only.

---

## 2. Traceability Table — §8.2 "What the Prototype Must Prove"

| # | Requirement (§8.2) | Artifact | Test | Status |
|---|---|---|---|---|
| 1 | Tile isolation — a PCIe endpoint holds a shard exclusively in device-local memory and executes a full transformer block without host read/write during inference | `att1_aimu_mmio_t` BAR0 model (M111), `att1_aimu_userspace` emulator (M121); tensor residency is a Stage 2 (M164) target | `tests/test_aimu_mmio.c`, `tests/test_aimu_userspace.c` | partial — control-plane isolation modeled; device-local tensor residency enforcement is unproven (Stage 2, M164) |
| 2 | Activation routing — compact activation vectors transferred across PCIe between tiles with measured latency/bandwidth matching the fabric packet model | `att1_fabric_t` (existing simulator), `compiler/simulate_fabric_bandwidth.py` (M118), `compiler/replay_fabric_routes.py` (M123) | `tests/test_bench_smoke.c` (`check_fabric_route_replay_smoke`) | partial — packet model and analytic bandwidth/latency proven; measured (real transport) latency is unproven (Stage 3, M170) |
| 3 | KV-cache locality — session KV memory maintained device-locally across decode steps without host copies between tokens | `att1_kv_mmu` opaque handle (M151), KV-MMU device-local semantics documented in `docs/kv_mmu.md` | `tests/test_kv_mmu.c` | proven (software) for the in-process model; device-local enforcement against a real device is unproven (Stage 2, M165) |
| 4 | Multi-tile decode — ≥2 tiles cooperate via fabric barrier/reduction, producing logits matching the simulator within tolerance (f32 exact, q8 ≤ 0.15, q4 ≤ 0.35) | `att1_cluster_infer_t` / `run_cluster` (existing), tolerances validated on CUDA at M155 | `tests/test_backend_matrix.c`, `compiler/backend_comparison_report.py` (M92) | proven (software/CUDA) for CPU and CUDA backends; two-tile decode against an emulated or physical endpoint is unproven (Stage 2, M166/M167) |
| 5 | Trace and counter fidelity — hardware tile reports the same counter categories the simulator records, so existing smoke tests validate hardware output unmodified | `att1_aimu_trace_t` (M108), unified trace/counter snapshot integrated into replay (M113, M122) | `tests/test_aimu_mmio_regression.c` (`trace_deterministic`) | proven (software) for the in-process and userspace-emulator counters; fidelity against a real or emulated endpoint is unproven until the M169 replay fidelity gate |
| 6 | Dtype coverage — f32 reference path and q8 path validated end-to-end; q4 desirable but not blocking | CPU f32/q8/q4 backends (M2, M7, M10, M11, M27, M81–M85); CUDA f32/q8/q4 backends (M14–M20, M23, M28, M87–M91); RTX 3090 signoff closed at M155 | `tests/test_backend_matrix.c` | proven (software + CUDA hardware) — this is the one §8.2 item proven on real hardware today, because CUDA is a real accelerator; PCIe/AIMU-tile dtype coverage remains Stage 2/3 work |

---

## 3. Traceability Table — §8.5 AIMU Tile Responsibilities

| Responsibility | Required for Phase 2 | Artifact | Status |
|---|---|---|---|
| Tensor memory ownership | yes | none (Stage 2 target: `backend_pcie.c`, M163/M164) | unproven |
| Matmul (f32) | yes | `backend_cpu_f32.c`, `cuda_backend_matmul_f32` | proven (software/CUDA); not on an AIMU tile |
| Matmul (q8×f32) | yes | `matmul_q8.c`, `cuda_backend_matmul_q8` | proven (software/CUDA); not on an AIMU tile |
| Matmul (q4×f32) | desirable | `convert_llama_to_att1.py --weight-format q4`, `cuda_backend_matmul_q4xf32` | proven (software/CUDA); not on an AIMU tile |
| RMSNorm | yes | CPU/CUDA RMSNorm kernels (M15 CUDA RMSNorm) | proven (software/CUDA); not on an AIMU tile |
| RoPE | yes | inference pipeline RoPE application | proven (software/CUDA); not on an AIMU tile |
| FFN / SwiGLU | yes | inference pipeline FFN/SwiGLU | proven (software/CUDA); not on an AIMU tile |
| Softmax (causal) | yes | inference pipeline attention softmax | proven (software/CUDA); not on an AIMU tile |
| KV-cache append/read | yes | `att1_kv_mmu` (M151) | proven (software) for the in-process model only |
| Fabric send/receive | yes | `att1_fabric_t`, `compiler/replay_fabric_routes.py` (M123) | proven (software) for the in-process model only |
| Barrier participation | yes | `att1_fabric_t` barrier semantics, `check_fabric_route_replay_smoke` (M123) | proven (software) for the in-process model only |
| Partial logit reduction | yes | fabric reduction handling in `replay_fabric_routes.py` (M123) | proven (software) for the in-process model only |
| Counter reporting | yes | `att1_aimu_trace_t` (M108), MMIO regression suite (M131) | proven (software) for the in-process/userspace-emulator model only |

None of the §8.5 responsibilities have been executed on an AIMU tile,
because no AIMU tile (physical or emulated-endpoint) exists yet. Every
software/CPU/CUDA math primitive above is proven at the algorithm level;
none has been proven as a tile-resident hardware operation. This is
expected — the emulated endpoint (Stage 2, M162–M169) and any physical
prototype (Stage 4) are precisely where this gap closes.

---

## 4. Traceability Table — §8.6–§8.10 Infrastructure Requirements

| §8 section | Requirement summary | Artifact | Status |
|---|---|---|---|
| §8.6 Local tensor memory | 256 MB–1 GB device-local budget depending on dtype/context | `att1-size` tile capacity estimator (M96), placement scenarios at 16/32/64/128 GiB SKUs (M102) | partial — capacity planning proven analytically; no physical or emulated device memory exists to validate against (Stage 2 M164, Stage 3 M173) |
| §8.7 Host control plane | load/transfer/enqueue/collect/manage/tolerate-errors | `att1_aimu_host` harness (M112), `att1-aimu-replay` / `att1-aimu-mmio-replay` (M113, M122) | proven (software) against the in-process simulator and userspace MMIO emulator; not yet exercised against a real or emulated PCIe endpoint |
| §8.8 Fabric/interconnect | PCIe Gen3 x4+, bounded queues, barrier all-or-nothing, queue-full detection, defined counters | `att1_fabric_t` model, `compiler/simulate_fabric_bandwidth.py` (M118), `compiler/validate_fabric_routes.py` (M116) | partial — logical model and analytic bandwidth proven; physical transport (or Stage 2 shared-memory/socket substitute) unproven |
| §8.9 KV-MMU / session memory | session/layer/head/position addressing, page granularity, ordering, error mapping, counters | `att1_kv_mmu` (M151), `tests/test_kv_mmu.c` | proven (software) for the in-process model; device-local enforcement unproven (Stage 2 M165) |
| §8.10 Required counters | `fabric_packets_sent/received`, `kv_appends`, `kv_reads`, `logits_bytes_produced`, `barrier_completions`, `decode_steps_executed`, `error_count` | `att1_aimu_trace_t` (M108), MMIO regression suite (M131), integrated pipeline regression report (M132) | proven (software) for the in-process/userspace-emulator counters; byte-level fidelity against an emulated or physical endpoint is the M169 gate |

---

## 5. Confirmation of the M120 §1 Unproven Items

The M120 go/no-go review ([aimu_phase3_go_no_go.md](aimu_phase3_go_no_go.md)
§1) identified five critical unproven areas as of M119. This audit confirms
that, despite the substantial software/control-plane work delivered across
M121–M155 (userspace MMIO emulator, command-plan and fabric-route replay,
execution-plan validation, golden/schema/hostile-input regression,
documentation, API refactor, fuzzing expansion, winning-strategy
governance, and CUDA signoff closure), **all five items remain unproven and
remain the correct Phase 2 hardware target list**:

| # | M120 §1 unproven item | Still unproven after M155? | Where it closes in the Phase 2 roadmap |
|---|---|---|---|
| 1 | No real PCIe endpoint or MMIO hardware exists | **Yes** | Stage 2 (M162 `att1-aimu-endpoint` skeleton, M163 `backend_pcie.c`); physical PCIe is Stage 4 (contingent, M175 gate) |
| 2 | No FPGA prototype has been built or evaluated | **Yes** | Stage 4 (M175 gate review; M176–M181 contingent FPGA prototype) |
| 3 | No physical AIMU tile memory, fabric link, DMA engine, or power rail exists | **Yes** | Stage 2 emulates these in-process (M162–M169); physical versions are Stage 4 |
| 4 | No hardware-backed inference has been run | **Yes** | Stage 2 (M166 single-tile emulated decode, M167 two-tile emulated cluster decode); *emulated*, not physical, until Stage 4 |
| 5 | No tensor-level placement has been executed against a real model | **Yes** | Stage 3 (M171 real small-model end-to-end through the emulated two-tile path; M173 placement/capacity validation) |

**Conclusion:** none of the five items have been closed by software-only
work, nor should they have been — closing them is explicitly out of scope
for Stage 0/1 (baseline freeze and interface freeze) by design. The M154
Phase 2 Definition of Done and the M153 decision-gated hardware progression
both anticipated this: interfaces must freeze (Stage 1, M157–M161) before
any endpoint work begins (Stage 2), and no FPGA/physical commitment is made
before the Stage 4 gate review. No scope change to Stage 1 or Stage 2 is
recommended as a result of this audit.

---

## 6. Recommendation

No changes to the Phase 2 roadmap ([PHASE2_PLAN.md](PHASE2_PLAN.md),
`ROADMAP.md`) are required. The audit confirms:

1. The Stage 0 entry baseline (frozen at M154, CUDA closed at M155) is
   accurate and complete.
2. The Stage 1 interface-freeze milestones (M157–M161) are correctly scoped
   to convert the "proven (software)" control-plane artifacts in §2–§4 above
   into versioned, frozen contracts before any endpoint work begins.
3. The Stage 2 emulated-endpoint milestones (M162–M169) are correctly scoped
   to close the "unproven" gaps in §5 above at the emulated (not physical)
   level, consistent with the M120 CONDITIONAL GO recommendation (Option B
   before any FPGA or hardware commitment).
4. No item in this audit justifies accelerating or deferring the Stage 4
   FPGA gate (M175); the go/no-go decision there should be made on evidence
   from Stages 1–3, not from this audit.

---

## 7. Non-Goals

- This audit does not itself close any of the five M120 unproven items.
- This audit does not change any C runtime, CUDA kernel, `.att1` format, or
  hardware interface behavior.
- This audit does not perform a new go/no-go review; that remains the scope
  of the Stage 4 gate review (M175).

---

## Cross-References

- [aimu_architecture.md](aimu_architecture.md) §8 — Phase 2 PCIe/AIMU
  Prototype Requirements (M93), the requirements source for this audit.
- [aimu_phase3_go_no_go.md](aimu_phase3_go_no_go.md) — M120 go/no-go review,
  source of the five unproven items confirmed in §5.
- [aimu_pcie_prototype_review.md](aimu_pcie_prototype_review.md) — M110
  design review and per-milestone history through M139.
- [PHASE2_PLAN.md](PHASE2_PLAN.md) — Phase 2 roadmap (M154–M184), stages,
  gates, and Definition of Done referenced throughout this audit.
- [CUDA_VALIDATION_PLAN.md](CUDA_VALIDATION_PLAN.md) — M138 CUDA signoff
  plan, closed at M155 (§2 and §5, item 6, of this audit).
- [OPERATION_LOG.md](OPERATION_LOG.md) — milestone-by-milestone delivery
  record (M103–M155) used as the delivery source for this audit.
