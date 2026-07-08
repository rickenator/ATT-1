# ATT-1 Phase 3 Private Boundary

**Status:** Active boundary for M176 and later hardware work.

M176 starts the Phase 3 hardware-private track. The public ATT-1 repository
remains the software validation baseline: simulator, `.att1` artifact format,
test harnesses, public architecture notes, and non-sensitive milestone records.

Phase 3 hardware work belongs in the private `ATT-1-HW` repository. That
includes board/BOM selection, vendor toolchain notes, procurement details,
private traces, FPGA experiments, register-bridge implementation details,
timing/resource reports, and any implementation notes that could affect IP
position.

## Public ATT-1 Allows

- Public simulator/runtime changes.
- Public validation harnesses and schema-compatible tooling.
- Non-sensitive architecture summaries.
- References to public ATT-1 commit hashes used as a baseline.
- General statement that Phase 3 hardware work continues privately.

## Private ATT-1-HW Owns

- M176-M181 board, host-access, procurement, and vendor-flow decisions.
- FPGA, PCIe, BAR0/MMIO, DMA, trace, and route-ack implementation details.
- Partner-style, production-like, or private trace packets.
- Cost, schedule, sourcing, and toolchain-risk notes.
- Any patent-sensitive or disclosure-sensitive analysis.

## Sync Rule

Move information from public ATT-1 to private ATT-1-HW freely when needed.
Move information from private ATT-1-HW back to public ATT-1 only after an
explicit disclosure review. Public updates should be sanitized summaries, not
raw hardware notes.
