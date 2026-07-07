# ATT-1 FPGA Gate Review (M175)

**Decision:** HOLD

**Date:** 2026-07-07

M175 is the formal successor to the M120/M126 hardware gate. It reviews the
Phase 2 evidence accumulated through M174 against the M126 FPGA criteria,
the M153 winning-strategy rules, and the Phase 2 Definition of Done declared
at M154.

The decision is **HOLD**: do not start M176 board selection, BOM commitment,
procurement, or FPGA RTL yet. The emulator/control-plane work is strong enough
to keep the hardware path alive, but not strong enough to authorize spend and
schedule drag on a physical prototype. The missing evidence is practical:
real external-artifact beachhead runs, explicit host-access selection, and a
small packet of partner/production-like traces. This is not a PIVOT because
the protocol evidence is improving and the userspace validation platform has
standalone value.

---

## 1. Inputs Reviewed

| Input | Status |
|---|---|
| Stage 1 interface freezes (M157-M161) | Complete; register map, command/completion packet, DMA/replay schemas, and fabric/barrier semantics are frozen v1.0 and conformance-checked. |
| Stage 2 emulated endpoint (M162-M169) | Complete; out-of-process endpoint, pcie backend, device-local KV-MMU, two-tile decode, hostile-endpoint testing, and replay fidelity gate all pass. |
| Stage 3 calibration/scale evidence (M170-M174) | Partial; transport characterization, validation harnesses, capacity gate, and activation-precision gate exist and pass on checked-in fixtures. |
| M126 FPGA feasibility criteria | Mostly advanced from "not yet" to software-complete, but host path and real workload evidence remain incomplete. |
| M153 kill criteria | Not triggered; no evidence yet of durable repeatable advantage, but also no evidence that the planning/control-plane path should be abandoned. |

---

## 2. Gate Criteria

| Criterion | Evidence | Result |
|---|---|---|
| Userspace MMIO/emulated endpoint stable and regression-tested | `make regression` passes; endpoint, MMIO, conformance, hostile-input, replay, and transport suites run in CPU-only mode. | PASS |
| Command/control interfaces frozen before hardware | M157-M160 freeze docs plus M161 conformance checks. | PASS |
| Replay fidelity across in-process and socket endpoint | M169 `att1-aimu-replay-fidelity` gate. | PASS |
| Fault injection and hostile endpoint behavior | M168 suite covers queue full, malformed completions, DMA rejection, endpoint crash, and KV ordering faults. | PASS |
| Two-tile cooperative decode through emulated transport | M167 plus socket-backed execution fix; `aimu_cluster_decode` passes. | PASS |
| Real small-model validation path | M171 harness exists and CI covers tiny fixtures; real external artifacts remain local/opt-in and are not committed as gate evidence. | PARTIAL |
| Beachhead workload metrics | M172 defines latency, memory movement, KV pressure, fabric packets, and optional CUDA/cost comparison; checked-in evidence is tiny-fixture only. | PARTIAL |
| Placement/capacity envelopes | M173 evaluates 256/512/1024 MiB budgets and selects 16-token KV pages; checked-in evidence is tiny-fixture only. | PARTIAL |
| Activation precision decision | M174 selects bf16 activation payloads for Phase 2 planning when the gate passes; f32 remains reference/diagnostic precision. | PASS |
| Host hardware access path selected | Standing direction is userspace access via VFIO or vendor XDMA with no custom kernel driver, but a concrete board/toolchain path is not selected. | PARTIAL |
| M153 design-partner / production-like traces | Not yet available in repo evidence. | FAIL |

---

## 3. Decision

**HOLD.**

Do not proceed to M176 board selection or procurement yet. The correct next
move is to tighten the software evidence packet until the hardware decision
can be made without leaning on hope:

1. Run the M171/M172/M173/M174 gates on one real local SmolLM2-135M-class
   artifact and a long-context token file. Keep model weights and generated
   public `.att1` artifacts out of Git, but record the report outputs.
2. Select the exact Phase 2 host-access path: VFIO, vendor XDMA userspace
   flow, or LitePCIe userspace bridge. No custom kernel driver.
3. Define the minimum control-plane FPGA target in one page: BAR0 read/write,
   command doorbell, completion polling, DMA descriptor validation, counter
   snapshot, and fabric-route acknowledgment only.
4. Add at least one production-like or partner-style trace packet with explicit
   pass/fail criteria, even if synthetic/anonymized at first.

---

## 4. Why Not GO

GO would authorize M176-M181, which means hardware research turns into board
selection, procurement, and implementation work. That is premature because
the most important Stage 3 evidence is still represented by deterministic
fixtures and harnesses, not by a recorded external-model beachhead run.

The protocol is hardware-expressible in the emulator. The next question is
whether the chosen workload has enough repeatable advantage to justify a
physical control-plane prototype. M175 does not yet have that answer.

---

## 5. Why Not PIVOT

PIVOT would abandon the hardware path and focus only on licensing or packaging
the planning/control-plane stack. That is also premature. The Stage 1 and
Stage 2 work has materially reduced interface risk, and Stage 3 now has
decision tools for transport, beachhead metrics, capacity, KV page sizing, and
activation precision.

The correct move is to keep the hardware path alive but gated.

---

## 6. Evidence Commands

The M175 review is validated by the same CPU-only regression packet used for
recent milestones:

```sh
python3 compiler/check_docs.py
git diff --check
make test
make regression
```

The real-workload evidence packet required to reopen the gate should include:

```sh
python3 compiler/validate_m171_two_tile.py ...
python3 compiler/validate_m172_beachhead.py ...
python3 compiler/validate_m173_capacity.py ...
python3 compiler/validate_m174_activation_precision.py ...
```

Use local paths only. Do not commit public model weights or generated public
`.att1` artifacts.

---

## 7. Exit Criteria To Reopen

The gate can reopen when all of the following are true:

| Reopen item | Required evidence |
|---|---|
| Real external-model packet | M171/M172/M173/M174 JSON/text reports from a local external artifact and long-context token file. |
| Host-access choice | One selected path: VFIO, vendor XDMA userspace, or LitePCIe userspace bridge, with board/toolchain implications. |
| Minimal FPGA control-plane scope | Explicit list of registers, queues, counters, and replay paths for M176-M181; tensor math remains out of scope. |
| Trace packet | At least one production-like or partner-style trace with pass/fail criteria. |
| Regression stays green | `make test`, `make regression`, and docs lint pass after the evidence packet is recorded. |

Until then, M176-M181 remain contingent and should not start.
