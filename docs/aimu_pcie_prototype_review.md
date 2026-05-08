# ATT-1 Minimal PCIe/AIMU Prototype Design Review (Milestone 110)

This document is the M110 engineering design review.  It reconciles the
software artifacts produced in M96–M109 and describes the minimal prototype
required to prove the PCIe/AIMU control-plane model before committing to FPGA
or custom-board hardware.

---

## 1. Executive Summary

### 1.1 What the prototype must prove

1. A host can discover AIMU tile count, tile memory capacity, and supported
   dtype/op flags through a well-defined register interface.
2. A placement report produced by `att1-size` can be transformed into a
   deterministic AIMU command plan and submitted to an AIMU command queue.
3. `LOAD_TENSOR_TILE` and `VALIDATE_TENSOR` commands complete with deterministic
   counter/trace output.
4. DMA buffer descriptors are validated against registered host and device
   memory regions before transfer.
5. Invalid placement, command, and DMA inputs fail clearly with structured
   error codes — no silent fallback.
6. The command-plan → command-queue pipeline is testable end-to-end without
   real hardware.

### 1.2 What is already proven in ATT-1 software (M0–M109)

| Capability | Milestone(s) |
|---|---|
| Single-tile and cluster LLM inference (f32/q8/q4) | M7, M27, M79 |
| CUDA backend (f32/q8/q4) | M20, M23, M89 |
| Placement report emission (att1-size) | M100 |
| Placement report validation (M98 schema) | M99 |
| Advisory placement proposals | M101 |
| Tile SKU scenario comparison | M102 |
| 64-byte command packet model | M103 |
| BAR0 register map sketch | M104 |
| In-process command queue simulator (ring buffer, CRC32) | M105 |
| Device/tile discovery simulator | M106 |
| DMA descriptor simulator (64-byte, validation rules) | M107 |
| Unified trace/counter snapshot | M108 |
| Placement-report-to-command-plan mapper | M109 |

### 1.3 What remains unproven

| Gap | Risk level |
|---|---|
| MMIO register-file simulator (actual BAR0 read/write) | Medium |
| PCIe latency and bandwidth modeling | Medium |
| Interrupt model (MSI-X) | Low (simulator can poll) |
| Real DMA engine | High (silicon only) |
| Fabric routing tied to command plan | Medium |
| Full command-queue → inference execution round-trip | High |
| Linux kernel driver | High (not a prototype requirement) |
| FPGA PCIe endpoint | High (hardware cost/risk) |

---

## 2. Prototype Objective

The **minimal prototype** is defined as:

> A host process that can execute the full M109 placement-to-command pipeline
> against an in-process AIMU simulator, produce a deterministic trace/counter
> snapshot, and report pass/fail for every LOAD/VALIDATE/BARRIER/QUERY/TRACE
> command — without requiring real PCIe hardware, a kernel driver, or FPGA.

This prototype is **not** required to execute matrix operations or produce
inference output.  Full hardware inference is a Phase 3 goal, not a Phase 2
proof-of-concept requirement.

### 2.1 Proof criteria

1. **Placement coverage**: every tensor in a valid placement report generates
   exactly one `LOAD_TENSOR_TILE` and one `VALIDATE_TENSOR` command.
2. **Command acceptance**: the M105 command queue accepts all commands in the
   plan without queue-full or checksum errors.
3. **Counter correctness**: after dispatching the plan, `commands_submitted`
   equals the plan's `command_count` and `commands_failed` is zero.
4. **Trace completeness**: the M108 trace snapshot reflects the final command
   queue, device, and DMA counters.
5. **DMA validity**: all buffer descriptors in the plan pass M107 validation
   rules (alignment, range, dtype, q4 group-size).
6. **Strict rejection**: a placement report with a `capacity_status=FAIL` tile
   is rejected in strict mode before any commands are submitted.
7. **Error determinism**: a malformed placement report or invalid command
   always exits with a non-zero status code and a structured error message.

---

## 3. Proposed Prototype Layers

```
┌─────────────────────────────────────────────────────────────┐
│  Layer 7 — Future hardware tile (ASIC / FPGA execution)     │
│  Not a prototype requirement.                               │
├─────────────────────────────────────────────────────────────┤
│  Layer 6 — Future PCIe/MMIO layer (real BAR0 R/W, DMA,     │
│            MSI-X, kernel driver)                            │
│  Not a prototype requirement; stubbed by Layer 5.           │
├─────────────────────────────────────────────────────────────┤
│  Layer 5 — AIMU device/tile simulator (M105–M108)           │
│  att1_aimu_cmdq_t, att1_aimu_device_t, att1_aimu_dma_sim_t, │
│  att1_aimu_trace_t — all in-process, C11, POSIX-safe.       │
├─────────────────────────────────────────────────────────────┤
│  Layer 4 — Host control-plane layer (M109 mapper + M105 CQ) │
│  map_placement_to_commands.py produces the command plan.    │
│  Host submits plan commands to att1_aimu_cmdq.              │
├─────────────────────────────────────────────────────────────┤
│  Layer 3 — Command-plan layer (M109)                        │
│  Deterministic command list derived from placement report.  │
│  JSON plan: header, commands[], summary, warnings.          │
├─────────────────────────────────────────────────────────────┤
│  Layer 2 — Placement/reporting layer (M96–M102)             │
│  att1-size --placement-report-json                          │
│  validate_tensor_placement_report.py                        │
│  propose_tensor_placement.py                                │
│  propose_tensor_scenarios.py                                │
├─────────────────────────────────────────────────────────────┤
│  Layer 1 — ATT-1 artifact/model layer (M0–M95)              │
│  .att1 model file, att1-bench, att1-inspect, tokenizer,     │
│  CPU f32/q8/q4, CUDA f32/q8/q4.                             │
└─────────────────────────────────────────────────────────────┘
```

Each layer is independently testable.  The prototype exercises layers 1–5.
Layers 6–7 are deferred to hardware bringup.

---

## 4. Minimal Viable Prototype Options

### Option A — Pure software AIMU/PCIe simulator (current state)

The M105–M109 stack already delivers this option.  The full pipeline runs
end-to-end in-process with no external dependencies.

| Dimension | Detail |
|---|---|
| Cost/risk | Very low — no hardware procurement |
| Implementation complexity | Complete (M105–M109 already implemented) |
| What it proves | Command-plan → command-queue pipeline; counter/trace correctness; DMA descriptor validation; placement validation; strict/non-strict rejection |
| What it does not prove | PCIe latency; real DMA bandwidth; interrupt-driven completion; hardware tile execution |
| Recommended next step | Integrate layers 1–5 into a single replay harness (M113); add MMIO register-file simulator (M111) |

### Option B — Userspace PCIe/MMIO mock with memory-mapped file

A memory-mapped file emulates the 64 KiB BAR0 register space.  A host thread
writes to offsets following M104; a device thread reads and dispatches.
No kernel driver required; uses `mmap(2)` on a shared file or anonymous
mapping.

| Dimension | Detail |
|---|---|
| Cost/risk | Low — POSIX, no hardware |
| Implementation complexity | Medium — requires register-file state machine and doorbell poll loop |
| What it proves | Register read/write protocol; doorbell write path; completion ring drain; MSI-X-equivalent notification (via eventfd or futex) |
| What it does not prove | Real PCIe latency; TLP framing; hardware tile execution |
| Recommended next step | Implement M111 (MMIO register-file simulator) then M112 (integration test harness) |

### Option C — FPGA PCIe endpoint prototype

A host PCIe card (e.g. Xilinx/AMD Alveo U50 or Intel Agilex F-Series) exposes
a BAR0 register space that a userspace MMIO driver maps.  The FPGA implements
the M104 register layout in RTL.

| Dimension | Detail |
|---|---|
| Cost/risk | High — hardware procurement, RTL development, PCIe IP licensing |
| Implementation complexity | High — requires PCIe IP core, AXI-Lite register bank, RTL simulation |
| What it proves | Real PCIe latency; BAR0 read/write timing; DMA TLP framing; interrupt delivery; actual register-map conformance |
| What it does not prove | Full tensor execution (requires custom compute fabric); production ASIC characteristics |
| Recommended next step | Only pursue after Option A/B are validated and M117 (FPGA feasibility notes) is complete |

### Option D — Custom board / future ASIC path

A custom PCIe board with an embedded processor (e.g. RISC-V soft core or ARM
Cortex-M) implements the M104 register interface and the M105 command dispatch
logic in firmware.  This is the closest to a production ASIC prototype.

| Dimension | Detail |
|---|---|
| Cost/risk | Very high — custom PCB, firmware, supply chain |
| Implementation complexity | Very high |
| What it proves | Full system: PCIe, DMA, register map, command dispatch, tile local memory, compute fabric |
| What it does not prove | Production-scale compute density; power envelope |
| Recommended next step | Defer to Phase 3 go/no-go (M120) |

**Recommendation:** Complete Option A integration (M111–M113), then evaluate
Option B.  Do not begin Option C or D until M120 go/no-go.

---

## 5. Required Host-Visible Capabilities

These capabilities are modelled in M104–M108 and must be exercisable through
the prototype simulator:

| Capability | M104 register(s) | Simulator module |
|---|---|---|
| Device discovery | `DEVICE_ID`, `DEVICE_VERSION`, `REGISTER_MAP_VERSION`, `FEATURE_FLAGS_LOW/HIGH` | M106 `att1_aimu_device_t` |
| Tile enumeration | `TILE_COUNT`; per-tile `TILE_ID`, `TILE_STATUS` | M106 `att1_aimu_device_query_tile` |
| Tile memory capacity | `TILE_MEMORY_CAPACITY_LOW/HIGH`, `TILE_KV_CAPACITY_LOW/HIGH` | M106 |
| Supported dtype discovery | `SUPPORTED_DTYPES` (f32/q8/q4 bits) | M106 `att1_aimu_device_supports_dtype` |
| Supported op discovery | `SUPPORTED_OPS_LOW/HIGH` (matmul/rmsnorm/rope/…) | M106 `att1_aimu_device_supports_op` |
| Command queue setup | `CQ_BASE_ADDR`, `CQ_SIZE`, `CQ_TAIL`, `CQ_DOORBELL` | M105 `att1_aimu_cmdq_t` |
| DMA descriptor validation | `DMA_CONTROL`, descriptor ring registers | M107 `att1_aimu_dma_t` |
| Counter/trace snapshot | Counter §7, trace §8, `COUNTER_SNAPSHOT_CONTROL` | M108 `att1_aimu_trace_t` |
| Error reporting | `ERROR_STATUS`, `ERROR_DETAIL`, `TILE_ERROR_STATUS` | M106 `att1_aimu_device_reset` |
| Reset/control path | `GLOBAL_CONTROL`, `RESET_CONTROL`, `TILE_RESET_CONTROL` | M106 `att1_aimu_device_reset_tile` |

---

## 6. Required AIMU Tile Capabilities

### 6.1 Memory model

| Region | Description |
|---|---|
| Local tensor memory | Weight storage; sized per tile memory SKU (16/32/64/128 GiB) |
| Local KV/session memory | Per-session paged KV cache; sized by `target_context_length × target_sessions` |
| Staging / command ring | Host-written command ring (M105 ring buffer, power-of-two depth) |
| Completion ring | Device-written completion records (same depth as command ring) |
| Trace buffer | Circular event log; size configurable via `TRACE_BUFFER_SIZE` |

### 6.2 Supported dtype flags

| Flag | Meaning |
|---|---|
| `ATT1_AIMU_DTYPE_F32` | 32-bit float; CPU f32 reference baseline |
| `ATT1_AIMU_DTYPE_Q8` | 8-bit quantized (per-row scale); `per_row_q8` quantization family |
| `ATT1_AIMU_DTYPE_Q4` | 4-bit grouped quantization; group sizes 32 or 64; `per_group_q4` family |

### 6.3 Supported op flags

| Flag | ATT-1 equivalent | Notes |
|---|---|---|
| `ATT1_AIMU_OP_MATMUL` | `att1_matmul_f32` / `att1_matmul_q8xf32` / `att1_matmul_q4xf32` | Core projection step |
| `ATT1_AIMU_OP_RMSNORM` | `att1_rmsnorm_f32` | Pre-attention and pre-FFN normalization |
| `ATT1_AIMU_OP_ROPE` | `att1_rope_f32` | Rotary positional encoding |
| `ATT1_AIMU_OP_ATTENTION` | `att1_attention_forward` | Causal softmax attention |
| `ATT1_AIMU_OP_FFN` | `att1_ffn_swiglu_f32` | SwiGLU gated FFN |
| `ATT1_AIMU_OP_KV_APPEND` | `att1_kv_cache_append` | Append KV pair to paged cache |
| `ATT1_AIMU_OP_KV_READ` | `att1_kv_cache_read` | Read KV slice from paged cache |
| `ATT1_AIMU_OP_FABRIC_SEND` | `att1_fabric_enqueue` | Send activation packet to peer tile |
| `ATT1_AIMU_OP_FABRIC_REDUCE` | `att1_fabric_enqueue` (broadcast) | All-reduce partial output across tiles |

### 6.4 Command completion behaviour

- Every submitted command produces exactly one completion record in the
  completion ring.
- `LOAD_TENSOR_TILE` and `VALIDATE_TENSOR` produce `ATT1_AIMU_ERR_OK`
  completions in the current simulator.
- `EXEC_*` commands produce `ATT1_AIMU_ERR_UNSUPPORTED_OP` completions
  (M105 design intent: host can drain the ring even for unsupported ops).
- The completion record carries `command_id`, `result_code`, and `fence_value`.

### 6.5 Counter and trace reporting

Counters are read via `QUERY_COUNTERS` commands (dispatched to host via
`att1_aimu_trace_snapshot_all`) or directly via the M104 counter register
offsets (§7).  The M108 `att1_aimu_trace_snapshot` captures:

- Command-queue counters: `commands_submitted`, `commands_completed`,
  `commands_failed`, `queue_full_count`, `unsupported_commands`, `fence_value`
- Device counters: `device_resets`, `tile_resets`, `tile_errors`
- DMA counters: `dma_submitted`, `dma_completed`, `dma_failed`, byte totals,
  `alignment_failures`, `range_failures`, `unsupported_flags`
- Fabric counters: placeholder (M114+ scope)

---

## 7. Data Movement Assumptions

### 7.1 Allowed PCIe traffic (cold path only)

| Operation | Direction | Notes |
|---|---|---|
| Tensor loading at model setup | Host → AIMU tile | One-time; DMA descriptor per tensor; M107 validation |
| Counter/trace snapshot | AIMU → Host | After tensor-load phase; before inference begins |
| Logit output per token | AIMU → Host | Only the lm_head tile outputs logits |
| Error / completion ring drain | AIMU → Host | Completion records; ~64 bytes per command |
| Command ring write (doorbell) | Host → AIMU | Host writes tail pointer; AIMU polls or receives MSI-X |

### 7.2 PCIe must NOT be in the hot path

The following must be AIMU-local during a decode step:

| Operation | Reason |
|---|---|
| Attention matmul (Q×K, scores×V) | Requires KV cache which must be tile-local |
| RMSNorm, RoPE, FFN (SwiGLU) | Low arithmetic intensity; PCIe latency would dominate |
| KV cache append/read | Paged KV memory is tile-local |
| Activation routing between tiles | Uses AIMU fabric (not PCIe); MTU ≈ 512 bytes |

### 7.3 Fabric traffic model

Activation tensors (d_model × 4 bytes for f32) are broadcast across tiles for
head-local attention or collected for partial-output reduction.  Fabric traffic
per decode step:

```
fabric_bytes ≈ activation_bytes_per_token × (tile_count − 1) × 2
```

The M96 bandwidth estimator models this as:

```
estimated_fabric_gib_sec = (fabric_payload_bytes_per_token × target_tokens_per_sec)
                            / (1024³)
```

This is a linear approximation; actual traffic depends on topology (ring,
mesh, or crossbar) and batching.

### 7.4 Tensor residency

Weights are loaded once per model session and remain AIMU-tile-local.  The
M109 command plan makes this explicit: every `LOAD_TENSOR_TILE` command
specifies `src_descriptor = host_buf:tensor_N` and
`dst_descriptor = tileN_local:tensor_N`.  No weight is transferred during
decode steps.

---

## 8. Register / Command Model Review

### 8.1 Command packet model (M103)

The M103 64-byte command packet carries 14 named fields:

```
command_id (u32) | command_type (u8) | tile_id (u8) | session_id (u16) |
dtype (u8)       | model_id (u32)   | tensor_id (u32)                   |
input_buf (u64)  | output_buf (u64) | kv_position (u32)                 |
op_params (u64)  | fence_ids (u32)  | trace_flags (u8) | checksum (u32) |
status (u8)      | _pad[N]
```

The M105 `att1_aimu_cmd` struct implements this layout exactly (64 bytes,
no implicit padding, CRC32 on submit).

### 8.2 Register map (M104)

BAR0 is a 64 KiB MMIO region divided into named offset ranges (§11.1 of
`docs/aimu_architecture.md`).  The simulator implements the logical
counterparts of each register in the M105–M108 C structs.

### 8.3 Command queue (M105)

- Power-of-two depth ring buffer (default 256 slots, max 4096).
- Host writes `CQ_TAIL`; AIMU polls head.
- CRC32 verified on every dispatch.
- Completion ring: same depth; AIMU writes; host drains.
- `commands_submitted` / `commands_completed` / `commands_failed` counters.
- Unsupported ops (`EXEC_*`) increment `unsupported_commands` and
  `commands_completed` (not `commands_failed`) — host can always drain.

### 8.4 DMA descriptor rules (M107)

- Descriptor: 64 bytes, 64-byte aligned.
- `byte_length` ∈ [1, 2²⁸]; Q4: `byte_length % (group_size / 2) == 0`.
- Host and device addresses: 64-byte aligned.
- Direction: H2D = 0, D2H = 1, D2D = 2.
- Dtype: F32 = 0, Q8 = 1, Q4 = 2; group sizes 32 or 64 for Q4.
- Validation: `att1_aimu_dma_validate` checks rules without counter increment.

### 8.5 Trace/counter snapshot (M108)

`att1_aimu_trace_snapshot_all` aggregates all three simulators in one call,
increments `snapshot_id`, and sets `status = OK` (all sources present) or
`PARTIAL` (some NULL).  The snapshot is self-contained and does not mutate
source counters.

### 8.6 Identified gaps

| Gap | Impact | Proposed resolution |
|---|---|---|
| MMIO register-file simulator | Medium: cannot test BAR0 R/W directly | M111 |
| Linux kernel driver | Low (prototype uses userspace) | Not a prototype requirement |
| MSI-X interrupt model | Low (can poll completion ring) | M112 optional eventfd |
| Real DMA engine | High (silicon only) | DMA simulator sufficient for prototype |
| Fabric routing tied to command plan | Medium: fabric counters are placeholders | M114 |
| EXEC_* command execution | High for inference | Deferred to hardware execution layer |

---

## 9. Placement Pipeline Review

The full placement pipeline (M96–M109) is:

```
att1-size [--config / --preset / --model]
  └─ --placement-report-json  →  placement_report.json   [M100]
        │
        ├─ validate_tensor_placement_report.py            [M99]
        │    └─ exits 0/1/2; structured error output
        │
        ├─ propose_tensor_placement.py                    [M101]
        │    └─ advisory remediation: dtype / tile-count proposals
        │
        ├─ propose_tensor_scenarios.py                    [M102]
        │    └─ cross-product SKU comparison table
        │
        └─ map_placement_to_commands.py                   [M109]
             └─ --plan-json  →  command_plan.json
                   │
                   └─ submitted to att1_aimu_cmdq  [M105]
                         │
                         ├─ att1_aimu_device  [M106]
                         ├─ att1_aimu_dma     [M107]
                         └─ att1_aimu_trace   [M108]
```

### 9.1 Tile economics supported by the pipeline

| Tile memory SKU | Max model parameters (q4) | `att1-size` preset | Typical tile count |
|---|---|---|---|
| 16 GiB | ~32 B params | `gpt-oss-30b-shape` (q4) | 2 |
| 32 GiB | ~65 B params | `gpt-oss-65b-shape` (q4) | 2–4 |
| 64 GiB | ~120 B params | `gpt-oss-120b-shape` (q4) | 2–8 |
| 128 GiB | ~240 B params | (custom config) | 2–8 |

Numbers are approximate for layer-wise placement with 80 % utilization target.
The `propose_tensor_scenarios.py` output shows the exact PASS/WARN/FAIL status
for each combination.

### 9.2 q4/q8/f32 storage impact on placement

| dtype | Storage vs f32 | Notes |
|---|---|---|
| f32 | 1× | Reference; CPU and CUDA |
| q8 | 0.25× (weight) + scale | Per-row scale adds ~1 byte/row |
| q4 | 0.125× (packed) + scale | Per-group scale; group sizes 32 or 64 |

The placement report captures `packed_bytes` and `scale_bytes` per tensor.
The command plan propagates these to every `LOAD_TENSOR_TILE` command so the
DMA descriptor can compute the correct `byte_length`.

### 9.3 Capacity feasibility loop

A tile is `capacity_status=FAIL` when:

```
model_bytes + kv_bytes > tile_memory_bytes
```

The placement-to-command mapper rejects FAIL capacity in `--strict` mode and
emits a `WARNING:` in non-strict mode.  The advisory tool proposes dtype
conversion or tile-count increase as remediation.

---

## 10. Minimal Success Criteria

The prototype is considered **passing** when all of the following hold:

| Criterion | How to verify |
|---|---|
| A valid placement report produces a command plan | `map_placement_to_commands.py --report ... --plan-json ...`; exit 0 |
| Command plan submits to simulated command queue | `att1_aimu_cmdq_submit` for each command; `commands_failed == 0` |
| LOAD/VALIDATE commands complete deterministically | `commands_completed == command_count`; `fence_value` monotone |
| Device/tile capabilities are discoverable | `att1_aimu_device_query_info` and `att1_aimu_device_query_tile` return expected flags |
| DMA descriptors pass validation | `att1_aimu_dma_validate` returns `ATT1_OK` for all plan tensors |
| Trace/counter snapshot captures command activity | `att1_aimu_trace_snapshot_all`; `snapshot.cmdq.commands_submitted > 0` |
| Invalid placement inputs fail clearly | Malformed JSON → exit 2; bad tile_id → exit 1; strict capacity FAIL → exit 1 |
| Invalid DMA inputs fail clearly | `att1_aimu_dma_submit` returns `ATT1_ERR_INVALID_ARG`; `alignment_failures` or `range_failures` incremented |
| No silent fallback | `commands_failed == 0`; no `ATT1_ERR_OK` returned for a known-bad input |
| No public artifacts committed | `git ls-files` contains no `.att1` from public model weights |

---

## 11. Risks

| Risk | Severity | Mitigation |
|---|---|---|
| Capacity model too optimistic (DRAM overhead, fragmentation) | Medium | Add 10–20 % margin to tile memory estimates; re-run scenario tool |
| Fabric traffic model too simple (ignores topology, batching) | Medium | Replace linear estimate with topology-aware model in M114 |
| Tensor-level placement not yet executable (advisory only) | High | Defer until EXEC_* commands are wired to runtime (post-M120) |
| Command simulator not yet connected to real inference | High | By design: prototype does not require inference execution |
| DMA simulator does not model real PCIe latency | Medium | PCIe latency model deferred to M116 |
| q4/q8 tolerance and token divergence workload-specific | Medium | Covered per M76–M91 tolerance tables; re-validate on target model |
| Register map may change after FPGA/PCIe constraints | Low | Register map versioned via `REGISTER_MAP_VERSION = 0x00010000` |
| Patent-sensitive hardware details should remain local/private | High | Do not describe specific circuit implementations in public docs |

---

## 12. Recommended Next Milestones

| Milestone | Title | Scope |
|---|---|---|
| M111 | MMIO register-file simulator | In-process 64 KiB BAR0 register array; R/W accessors for all M104 register offsets; read-only enforcement for RO registers; doorbell write path; integration with M105–M108 simulators |
| M112 | Command queue + device + DMA integration test harness | Single C11 test that runs the full pipeline: device probe → DMA register → submit command plan → snapshot → assert counters; no Python required |
| M113 | Placement-command replay tool | `compiler/replay_command_plan.py` or `tools/att1-replay.c`; reads M109 JSON plan; drives M105/M106/M107/M108 simulators; emits pass/fail and snapshot JSON |
| M114 | AIMU fabric routing requirements | Document and simulate fabric route-table updates per placement policy; connect fabric counters to placement policy; topology-aware bandwidth estimate |
| M115 | AIMU tile memory allocator simulator | AIMU-local slab allocator for tensor, KV, staging, and trace regions; fragmentation reporting; integrates with DMA descriptor address range checks |
| M116 | PCIe latency/bandwidth model | Analytical model for PCIe Gen4/Gen5 latency per command and per DMA transfer; integrate with placement scenario tool bandwidth estimate |
| M117 | FPGA feasibility notes | Document register-map to AXI-Lite mapping; PCIe IP core options; RTL complexity estimate; resource utilization ballpark; go/no-go criteria for Option C |
| M118 | Prototype bill-of-materials / board options review | Compare development board options (Alveo U50, Agilex F-Series, custom PCIe card); supply chain, cost, and timeline estimates for Option C/D |
| M119 | Hardware/software boundary review | Define the exact API surface between the host driver, MMIO register file, DMA engine, command dispatcher, and tile execution fabric; version and freeze the boundary |
| M120 | Phase 3 prototype go/no-go review | Final decision: Option A (software only), B (userspace MMIO mock), C (FPGA), or D (custom board); gate on M111–M119 completion |

---

## 13. Non-Goals

- No production ASIC design or tape-out.
- No patent claim language of any kind.
- No Linux kernel driver (PCIe or otherwise).
- No real PCIe endpoint or BAR0 MMIO implementation.
- No public cloud deployment or container packaging.
- No mobile, Android, Vulkan, or OpenCL target.
- No new inference features or model format changes.
- No public model weights or generated public `.att1` artifacts committed to Git.
- No `__pycache__` or `.pyc` files tracked.

---

## Appendix A: M104–M109 Artifact Summary

| Milestone | Key artifact(s) | Status |
|---|---|---|
| M104 | `docs/aimu_register_map.md` | Complete |
| M105 | `include/att1_aimu_cmdq.h`, `src/aimu_cmdq.c` | Complete |
| M106 | `include/att1_aimu_device.h`, `src/aimu_device.c` | Complete |
| M107 | `include/att1_aimu_dma.h`, `src/aimu_dma.c` | Complete |
| M108 | `include/att1_aimu_trace.h`, `src/aimu_trace.c` | Complete |
| M109 | `compiler/map_placement_to_commands.py` | Complete |
| M110 | `docs/aimu_pcie_prototype_review.md` | This document |

---

## Appendix B: Open Engineering Questions (carried from M103/M104)

The following questions are still open and should be resolved before committing
to Option C or D hardware:

1. **BAR layout.** Should the command queue ring and trace buffer be in BAR0
   or a separate BAR?  Separate BARs allow the trace buffer to be a much larger
   PCIe MMIO window without affecting the register access latency of BAR0.

2. **MSI-X vector allocation.** Recommend one MSI-X vector per tile completion
   event type (completion, error, trace-full).  Total: `tile_count × 3` vectors
   for a 16-tile device = 48 vectors (within PCIe MSI-X limit of 2048).

3. **DMA descriptor format.** The M107 64-byte descriptor is contiguous only.
   Scatter-gather support (linked-list descriptors) is deferred to M115.

4. **Fence ID namespace.** The current M105 `fence_value` is a single monotone
   u64.  Cross-tile fences (e.g., tile 0 FABRIC_REDUCE completes before tile 1
   KV_APPEND begins) require a per-tile fence namespace or a global barrier
   instruction.  Resolved by M114.

5. **64-bit register atomicity.** The M104 64-bit counter registers are split
   into LOW/HIGH 32-bit halves.  On real hardware, a shadow-register read
   protocol is required to prevent torn reads.  M111 should implement a
   consistent read sequence (read LOW → read HIGH → re-read LOW; retry if
   changed).

6. **Counter overflow.** The 64-bit counters support 2⁶⁴ events before
   wrapping.  At 1 billion commands per second, overflow takes approximately
   585 years.  No overflow handling is required.

7. **Trace buffer full policy.** The M104 `TRACE_DROPPED_EVENTS` register
   counts dropped events when the trace buffer is full.  The recommended policy
   is drop-oldest (ring overwrite) with `TRACE_DROPPED_EVENTS` incremented.

8. **Q4 group size on hardware.** Hardware tile implementations may support
   only group size 32 or only group size 64 for area/power reasons.  The
   `SUPPORTED_DTYPES` register should carry a per-group-size bit to allow
   runtime capability query.  M119 should freeze this field.

9. **Tile memory granularity.** Local tensor memory must be allocated in
   page-aligned chunks.  The M107 DMA descriptor requires 64-byte alignment
   for all addresses.  The M115 tile memory allocator should use a minimum
   allocation granularity of 64 bytes.

10. **Routing table ownership.** The M104 `FABRIC_ROUTE_TABLE_BASE` register
    points to a host-written routing table.  Who writes the initial table — the
    host driver at boot, the placement report tool, or the command plan mapper?
    M114 should define the routing table schema and the M109 mapper should
    optionally emit route-table update commands.
