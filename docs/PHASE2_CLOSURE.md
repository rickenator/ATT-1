# ATT-1 Phase 2 Closure

**Status:** Closed at M175 with a HOLD decision.

**Date:** 2026-07-07

Phase 2 is closed as a software evidence phase, not as a hardware launch.
M154-M175 completed the baseline freeze, CUDA signoff, Phase 1 gap audit,
interface freezes, out-of-process endpoint path, pcie-style backend, endpoint
KV-MMU, single-tile and two-tile emulated decode, hostile-endpoint testing,
replay fidelity, transport characterization, real-artifact validation harnesses,
beachhead metrics, placement/capacity validation, activation precision study,
and the M175 FPGA gate review.

The closure decision is **HOLD**. M176-M181 board selection, procurement, FPGA
RTL, BAR0-on-hardware, DMA-on-hardware, trace registers on hardware, and fabric
route acknowledgment on FPGA are not started by this closure. They remain
contingent on reopening the M175 gate.

## Closure Result

| Area | Result |
|---|---|
| Stage 0 baseline/debt closure | Complete through M156 |
| Stage 1 interface freezes | Complete through M161; frozen v1.0 contracts documented |
| Stage 2 emulated endpoint | Complete through M169; endpoint, pcie backend, KV-MMU, two-tile decode, hostile tests, and replay fidelity are covered |
| Stage 3 evidence tools | Complete through M174; harnesses and fixtures exist, but real external-model evidence remains local/opt-in |
| Stage 4 hardware gate | M175 complete; decision is HOLD |
| Stage 5 product/partner work | Deferred; design-partner traces are a reopen criterion, not completed Phase 2 evidence |

## Reopen Criteria

The hardware path can reopen when the evidence packet from
[FPGA_GATE_REVIEW_M175.md](FPGA_GATE_REVIEW_M175.md) is present:

1. M171/M172/M173/M174 reports from one local external SmolLM2-135M-class
   artifact and a long-context token file.
2. One selected host-access path: VFIO, vendor XDMA userspace, or LitePCIe
   userspace bridge.
3. A one-page minimum FPGA control-plane scope.
4. At least one production-like or partner-style trace packet with explicit
   pass/fail criteria.
5. `make test`, `make regression`, and docs lint still pass after the evidence
   packet is recorded.

Until those items exist, the correct next work is evidence closure around M175,
not M176 hardware execution.
