# ATT-1 FPGA Gate Review (M175)

**Decision:** GO for constrained M176-M181 control-plane work.

**Date:** 2026-07-07

M175 is the formal successor to the M120/M126 hardware gate. It reviews the
Phase 2 evidence accumulated through M174 against the M126 FPGA criteria,
the M153 winning-strategy rules, and the Phase 2 Definition of Done declared
at M154.

The original M175 review decision was **HOLD**: do not start M176 board
selection, BOM commitment, procurement, or FPGA RTL until a practical evidence
packet exists. That evidence packet now exists and passes, so the current
decision is **GO** for constrained M176-M181 control-plane work. This does not
authorize tensor math, a custom Linux kernel driver, or public model artifacts
in Git.

The green path is part of M175 itself: when the reopen criteria below are met,
M175 flips from HOLD to GO and M176 may begin without renumbering the roadmap.

---

## 1. Inputs Reviewed

| Input | Status |
|---|---|
| Stage 1 interface freezes (M157-M161) | Complete; register map, command/completion packet, DMA/replay schemas, and fabric/barrier semantics are frozen v1.0 and conformance-checked. |
| Stage 2 emulated endpoint (M162-M169) | Complete; out-of-process endpoint, pcie backend, device-local KV-MMU, two-tile decode, hostile-endpoint testing, and replay fidelity gate all pass. |
| Stage 3 calibration/scale evidence (M170-M174) | Complete for the M175 gate; transport characterization, real local SmolLM2-135M-class reports, capacity gate, and activation-precision gate pass. |
| M126 FPGA feasibility criteria | Advanced enough for a constrained control-plane prototype; tensor math remains deferred. |
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
| Real small-model validation path | M171 green-packet report passed on local SmolLM2-135M f32/q8 `.att1` artifacts outside Git. | PASS |
| Beachhead workload metrics | M172 green-packet report passed on the local q8 two-tile beachhead workload. | PASS |
| Placement/capacity envelopes | M173 green-packet report passed; 16-token KV pages remain the selected planning point. | PASS |
| Activation precision decision | M174 selects bf16 activation payloads for Phase 2 planning when the gate passes; f32 remains reference/diagnostic precision. | PASS |
| Host hardware access path selected | Vendor XDMA userspace bridge selected first, with VFIO fallback; no custom kernel driver. | PASS |
| M153 design-partner / production-like traces | One production-like trace packet with explicit pass/fail criteria is recorded as a local M175 input. | PASS |

---

## 3. Decision

**GO for constrained control-plane work.**

M176 may begin with board/BOM selection and host-access planning around vendor
XDMA userspace first, VFIO fallback. M176-M181 remain limited to BAR0
read/write, command doorbell, completion polling, DMA descriptor validation,
counter/trace readback, and fabric-route acknowledgment. Tensor math and custom
kernel-driver work stay deferred.

---

## 4. Why This Is Now GO

The missing M175 evidence is now present: a local external SmolLM2-135M-class
artifact pair, long-context token input, M171-M174 reports, host-access
decision, minimum FPGA scope, and trace packet all passed through
`compiler/run_m175_green_packet.py`.

The protocol is still not a full hardware-inference proof. The GO decision only
authorizes the smallest physical control-plane prototype that can test PCIe
enumeration, BAR0/MMIO, doorbell/completion, DMA descriptor validation,
counters/traces, and route acknowledgment against the existing software stack.

---

## 5. Why Not PIVOT

PIVOT would abandon the hardware path and focus only on licensing or packaging
the planning/control-plane stack. That is also premature. The Stage 1 and
Stage 2 work has materially reduced interface risk, and Stage 3 now has
decision tools for transport, beachhead metrics, capacity, KV page sizing, and
activation precision.

The correct move is to keep the hardware path narrow and evidence-gated.

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

The real-workload evidence packet that reopened the gate is:

```sh
python3 compiler/run_m175_green_packet.py \
  --model-dir ~/Models/SmolLM2-135M \
  --att1-f32 ~/Models/att1/SmolLM2-135M/model_f32.att1 \
  --att1-q8 ~/Models/att1/SmolLM2-135M/model_q8.att1 \
  --out-dir ~/Models/att1/SmolLM2-135M/m175_green_packet \
  ...
```

Use local paths only. Do not commit public model weights or generated public
`.att1` artifacts.

---

## 7. Green Criteria To Reopen

The gate can reopen to **GO** when all of the following are true:

| Reopen item | Required evidence |
|---|---|
| Real external-model packet | M171/M172/M173/M174 JSON/text reports from a local external artifact and long-context token file, preferably tied together by `compiler/run_m175_green_packet.py`. |
| Host-access choice | One selected path: VFIO, vendor XDMA userspace, or LitePCIe userspace bridge, with board/toolchain implications. |
| Minimal FPGA control-plane scope | Explicit list of registers, queues, counters, and replay paths for M176-M181; tensor math remains out of scope. |
| Trace packet | At least one production-like or partner-style trace with pass/fail criteria. |
| Regression stays green | `make test`, `make regression`, and docs lint pass after the evidence packet is recorded. |

These are true. M175 is no longer HOLD and M176-M181 may start under the
constrained control-plane scope.
