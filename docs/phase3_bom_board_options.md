# ATT-1 Phase 3 Prototype BOM and Board Options Review (Milestone 127)

This document is research and planning documentation only.  No hardware is
selected, no purchase is authorized, and no RTL is implemented by this
milestone.  All content is advisory.  Prices and availability change; any
figures given are approximate and should be independently verified before
planning a purchase.

---

## 1. Executive Summary

**Phase 3 hardware is not ready for purchase or commitment.**

The purpose of this document is to define BOM categories, board option
classes, decision criteria, and the gate conditions that must be satisfied
before hardware acquisition is authorized.

The ATT-1 software control-plane interfaces (register map, command packet
schema, command queue model, fabric route schema, execution plan schema) are
still evolving.  Purchasing an FPGA board before these interfaces are frozen
would risk purchasing hardware that cannot implement the final interface, or
board re-selection after a register map change.

**Recommended near-term path:** Continue the no-hardware userspace emulator
path (M121–M125).  Define a board shortlist only after M121–M131 software
milestones stabilize interfaces.  Revisit hardware authorization at the M132
FPGA board shortlist update and M133 hardware purchase go/no-go checkpoint.

---

## 2. Prototype BOM Categories

A Phase 3 ATT-1 FPGA prototype would require the following categories of
equipment and materials.  No specific part numbers or quantities are committed.

### 2.1 FPGA/PCIe development board

- The primary platform for implementing the M104 register map, M105 command
  queue, M107 DMA descriptor model, and optionally M108 trace/counter block.
- Must provide: PCIe Gen2+ hard endpoint IP (preferred), BRAM/URAM sufficient
  for command and completion queues, JTAG programming header.
- Optional: on-board DDR3/DDR4/HBM for tensor weight storage in later phases.

### 2.2 Host workstation / PCIe slot

- An x86_64 Linux host with an available PCIe x4 or x8 mechanical slot.
- Slot must be electrically PCIe Gen2 or later.
- IOMMU support required for future VFIO DMA path (VT-d on Intel, AMD-Vi on AMD).
- RAM: ≥16 GiB recommended for Vivado/Quartus synthesis flows.
- Disk: ≥100 GiB free for toolchain installation and synthesis artifacts.

### 2.3 Power and cooling

- Most FPGA dev boards draw 10–25 W from the PCIe slot (75 W edge limit).
- Boards with on-board DDR or HBM may require external PCIe power connector
  (6-pin or 8-pin auxiliary) or dedicated ATX supply.
- Standard desktop tower ATX supply is adequate for prototype; server chassis
  provides better stability.

### 2.4 Cabling and adapters

- PCIe x1 to x4 riser cable if full-length slot is not available (verify
  signal integrity; some risers introduce PCIe link instability).
- USB-A or USB-C JTAG programming cable (vendor-specific; typically bundled
  with dev board or available separately).
- Ethernet cable for remote access to host during debug sessions.

### 2.5 Debug and programming tools

- **JTAG programmer**: required for bitstream loading and in-circuit debugging.
  Many dev boards include a USB-based JTAG interface on-board.
- **Logic analyzer** (optional): useful for debugging AXI-Lite or BRAM
  interface timing; can be deferred to later prototype phase.
- **UART/serial console**: many FPGA boards expose a UART for debug print
  output from soft-core or IP debug blocks.

### 2.6 Storage for model artifacts

- The `.att1` model binary for smoke testing DMA descriptor paths.
- Local NVMe or SSD recommended; HDD is sufficient for prototype.
- No large model files committed to Git; model artifacts stored outside
  version control.

### 2.7 Software and toolchain licensing

- **Xilinx/AMD Vivado**: free WebPACK edition covers Artix-7 and most
  UltraScale devices; paid enterprise edition required for UltraScale+ HBM.
- **Intel/Altera Quartus**: free Lite edition covers Cyclone V and some Arria;
  paid Pro edition required for Stratix 10 and Agilex.
- **LitePCIe (open-source)**: no licensing cost; Python/Migen build chain;
  community supported.
- **Host Linux tools**: all required host tools (VFIO, UIO, `lspci`, `devmem2`,
  `pcimem`) are open-source.

### 2.8 Spare parts and risk buffer

- Budget at least one spare FPGA board unit if the prototype is on the critical
  path; FPGA boards can be damaged by MMIO address errors that assert SLVERR
  unexpectedly, or by incorrect PCIe strapping.
- Spare programming cable recommended.

---

## 3. Board Option Classes

Six option classes are defined.  No specific board SKU is selected within any
class.

### 3.1 No-hardware userspace emulator path (current path)

| Attribute | Value |
|---|---|
| What it proves | All control-plane logic: register map semantics, command queue FSM, doorbell, completion, DMA descriptor validation, fabric route counting, execution plan generation |
| What it does not prove | Real PCIe enumeration, actual DMA transfer, hardware timing, interrupt latency |
| Complexity | Low |
| Risk | Very low |
| Cost class | Zero (no hardware) |
| M104 BAR0 register map | Yes (via mmap-backed file) |
| Command/completion queue | Yes (M105 + M122 replay) |
| DMA descriptor testing | Descriptor-only (no actual DMA) |
| Fabric route replay | Yes (M123) |
| Tensor math hardware | No |

**Recommended** until gate criteria in §9 are met.

---

### 3.2 Low-cost PCIe FPGA dev board

Target FPGA families: Xilinx Artix-7, Intel Cyclone V, AMD Spartan-7.

| Attribute | Value |
|---|---|
| What it proves | Real PCIe enumeration, BAR0 MMIO access, AXI-Lite register block, doorbell write, descriptor validator FSM, counter/trace registers |
| What it does not prove | Sustained DMA throughput (limited BRAM), tensor math, HBM-class memory bandwidth |
| Complexity | Moderate |
| Risk | Moderate (timing closure on Artix-7 PCIe Gen2 can be challenging) |
| Cost class | Low (USD 100–500 approximate, board only) |
| M104 BAR0 register map | Yes |
| Command/completion queue | Yes (with BRAM; limited depth) |
| DMA descriptor testing | Descriptor-only or basic DMA engine |
| Fabric route replay | Route counter registers only |
| Tensor math hardware | No (insufficient DSP cascade depth) |

Suitable for: validating PCIe enumeration and BAR0 register access.
Not suitable for: tensor math, sustained DMA, or multi-tile fabric.

---

### 3.3 Midrange FPGA PCIe dev board

Target FPGA families: Xilinx Kintex-7, Xilinx UltraScale, Intel Arria 10 GX.

| Attribute | Value |
|---|---|
| What it proves | Everything in §3.2 plus: larger command/completion ring capacity, deeper trace BRAM, basic XDMA scatter-gather DMA, more comfortable timing closure margin |
| What it does not prove | HBM-class memory bandwidth, full tensor inference throughput |
| Complexity | Moderate |
| Risk | Low–Moderate |
| Cost class | Midrange (USD 500–3000 approximate, board only) |
| M104 BAR0 register map | Yes |
| Command/completion queue | Yes (comfortable depth) |
| DMA descriptor testing | Full XDMA scatter-gather DMA feasible |
| Fabric route replay | Route counter registers + basic DMA payload path |
| Tensor math hardware | Feasible (sufficient DSP slices for prototype q8 matmul) |

Suitable for: DMA descriptor testing, basic q8 matmul prototype, command-plan
replay with actual DMA.
Not suitable for: HBM bandwidth targets, multi-tile interconnect.

---

### 3.4 High-end FPGA dev board with DDR/HBM

Target FPGA families: Xilinx UltraScale+ HBM, AMD Versal, Intel Stratix 10 MX/GX.

| Attribute | Value |
|---|---|
| What it proves | Everything in §3.3 plus: sustained tensor weight DMA, HBM bandwidth for large models, more fabric routing resources |
| What it does not prove | Production-grade inference throughput, full multi-tile fabric switching at scale |
| Complexity | High |
| Risk | High (toolchain effort, timing closure, power design, cost) |
| Cost class | High (USD 3000–15000+ approximate, board only) |
| M104 BAR0 register map | Yes |
| Command/completion queue | Yes |
| DMA descriptor testing | Full DMA with HBM target |
| Fabric route replay | Yes, with simulated payload DMA |
| Tensor math hardware | Yes (sufficient resources for q4/q8 matmul prototype) |

Suitable for: full prototype with tensor math, once interfaces are frozen.
Not suitable for: early prototype phase; defer until gate criteria met.

---

### 3.5 Embedded SoC FPGA board

Target devices: Xilinx Zynq-7000, Zynq UltraScale+ MPSoC, Intel SoC FPGA.

| Attribute | Value |
|---|---|
| What it proves | AXI-Lite register block accessible from embedded ARM core; command queue FSM; descriptor validator; soft-PCIe or PCIe via external bridge |
| What it does not prove | Real PCIe enumeration on a standard x86 host (Zynq has PCIe, but typically as root complex or endpoint depending on board) |
| Complexity | Moderate–High (dual ARM+FPGA toolchain) |
| Risk | Moderate (PCIe endpoint support varies by board) |
| Cost class | Low–Midrange (USD 200–1500 approximate, board only) |
| M104 BAR0 register map | Yes (via AXI-Lite from ARM) |
| Command/completion queue | Yes |
| DMA descriptor testing | Yes (ARM DMA controller or PL DMA) |
| Fabric route replay | Yes |
| Tensor math hardware | Feasible for prototype scale |

Suitable for: standalone embedded prototype without an x86 host; Linux ARM
soft-driver path; alternative to PCIe dev board if PCIe slot not available.

---

### 3.6 Custom PCIe carrier board

A custom PCB with selected FPGA, DDR/HBM, and PCIe connectors, designed to
spec for ATT-1 AIMU requirements.

| Attribute | Value |
|---|---|
| What it proves | Everything the device is designed to prove; matches ATT-1 spec exactly |
| What it does not prove | Anything until completed (multi-month NRE) |
| Complexity | Very high |
| Risk | Very high (PCB design, fabrication, bring-up, firmware stack) |
| Cost class | Very high (USD 50K–500K+ NRE estimate; highly uncertain) |
| All capabilities | Yes, if designed correctly |

Not suitable for current phase.  Deferred indefinitely until commercial
viability is demonstrated via dev board prototypes.

---

## 4. Minimum Viable FPGA Capabilities

Any board considered for the ATT-1 PCIe prototype must provide:

| Capability | Requirement |
|---|---|
| PCIe endpoint | Hard PCIe Gen2 ×1 or ×4 endpoint IP block (soft PCIe accepted with caveats) |
| BAR0 MMIO region | ≥64 KiB BAR0 accessible via AXI-Lite slave |
| Register bus | AXI-Lite (preferred) or Avalon-MM |
| BRAM | ≥4 Mb for command/completion/trace queues |
| URAM | Not required for minimum viable; beneficial for queue depth |
| External DDR/HBM | Not required for minimum viable; required for DMA weight load phase |
| JTAG/debug access | Required (on-board USB-JTAG preferred) |
| Linux host compatibility | PCIe endpoint must appear on Linux x86_64 PCI bus |
| Vendor toolchain | Vivado WebPACK, Quartus Lite, or equivalent free tier available |
| Power | PCIe slot power (75 W edge) sufficient; no external power required for minimum viable |

---

## 5. Host Requirements

### 5.1 CPU and OS

- Architecture: x86_64.
- OS: Linux (kernel ≥5.10 LTS recommended for VFIO improvements).
- RAM: ≥16 GiB for synthesis; ≥8 GiB for host driver + emulator testing.
- No macOS or Windows requirement; all ATT-1 tooling targets Linux.

### 5.2 PCIe slot

- Minimum: PCIe x4 mechanical slot, electrically Gen2 or later.
- Preferred: PCIe x8 or x16 physical slot (accepts x4 card with bandwidth headroom).
- IOMMU must be enabled in UEFI/BIOS for VFIO path (VT-d / AMD-Vi).

### 5.3 Power and cooling

- Standard ATX tower power supply with PCIe auxiliary connector is sufficient
  for low-cost and midrange boards.
- Server chassis with redundant PSU provides better stability for extended
  test runs.

### 5.4 Userspace tool requirements

- `lspci`, `setpci`, `pcimem` or `devmem2` for initial BAR0 probing.
- Python 3.10+ for ATT-1 compiler and emulator scripts.
- GCC ≥ 12 or Clang ≥ 15 with `-std=c11` for ATT-1 C runtime.

### 5.5 Future driver path

- First pass: UIO thin driver for BAR0 mmap to userspace.
- Preferred: VFIO for DMA-safe userspace access.
- Long-term: custom `pci_driver` for interrupt-driven completion and power
  management; defer until interfaces are frozen.

### 5.6 3090-class server compatibility note

If the development host is a server-class machine with an NVIDIA 3090 or
similar GPU (already present for CUDA opt-in use), the FPGA PCIe board can
occupy a second slot.  Verify that the platform supports multiple PCIe devices
with IOMMU group isolation; some consumer platforms do not provide separate
IOMMU groups per slot.

---

## 6. Cost and Risk Classes

| Path | Cost class | Engineering risk | Capability ceiling |
|---|---|---|---|
| No-hardware userspace emulator | Zero | Very low | Control-plane logic only; no real PCIe |
| Low-cost FPGA dev board | Low (~USD 100–500) | Moderate | BAR0 + command queue + descriptor validation |
| Midrange FPGA dev board | Midrange (~USD 500–3000) | Low–Moderate | + DMA + basic tensor math |
| High-end FPGA/HBM board | High (~USD 3000–15000+) | High | + sustained tensor inference prototype |
| Custom carrier board | Very high (USD 50K–500K+ NRE) | Very high | Full AIMU spec |

All cost figures are approximate and subject to market variation.  Verify with
vendors or distributors before planning.

---

## 7. Decision Matrix

Rows are board option classes.  Columns are evaluation criteria.
Rating: ✓ = yes / ✗ = no / ~ = partial/conditional.

| Criterion | Userspace emulator | Low-cost FPGA | Midrange FPGA | High-end FPGA | Custom board |
|---|---|---|---|---|---|
| Proves BAR0 register map | ✓ (software) | ✓ | ✓ | ✓ | ✓ |
| Proves command queue | ✓ (software) | ✓ | ✓ | ✓ | ✓ |
| Proves completion queue | ✓ (software) | ✓ | ✓ | ✓ | ✓ |
| Proves DMA descriptor flow | ~ (descriptor-only) | ~ | ✓ | ✓ | ✓ |
| Proves real PCIe enumeration | ✗ | ✓ | ✓ | ✓ | ✓ |
| Supports future DMA engine | ✗ | ~ | ✓ | ✓ | ✓ |
| Supports fabric counter model | ✓ (software) | ~ | ✓ | ✓ | ✓ |
| Supports tensor math hardware | ✗ | ✗ | ~ | ✓ | ✓ |
| Toolchain difficulty | Very low | Moderate | Moderate | High | Very high |
| Schedule risk | Very low | Moderate | Low–Moderate | High | Very high |
| Cost risk | Zero | Low | Moderate | High | Very high |

For the current phase, the **userspace emulator** scores best on risk, cost,
and toolchain difficulty while covering all control-plane validation needs.

---

## 8. Recommended Near-Term Path

| Action | Rationale |
|---|---|
| Continue no-hardware userspace emulator path | Covers all current validation needs; zero cost; low risk |
| Do not purchase FPGA board yet | Interfaces not frozen; purchase now risks wrong board selection |
| Prepare board shortlist only after M121–M131 stabilize | Shortlist is research artifact, not a purchase order |
| Revisit hardware after execution-plan validator (M128) and replay tooling (M130) mature | Ensures the control-plane model is stable before committing to silicon |
| Draft shortlist at M132 FPGA board shortlist update | Defined milestone for re-evaluation |
| Gate hardware purchase at M133 | Defined checkpoint with explicit go/no-go |

---

## 9. Hardware Purchase Gate

Hardware must not be purchased until **all** of the following conditions are
satisfied:

| Gate condition | Status |
|---|---|
| M121 userspace MMIO emulator stable and regression-tested | Complete |
| M122 command-plan replay through emulator stable | Complete |
| M123 fabric route replay stable | Complete |
| M124 tile memory allocator interface stable | Complete |
| M125 tensor execution plan stable | Complete |
| Register map versioning formally frozen | **Not yet** |
| Command packet schema formally frozen | **Not yet** |
| Fabric route schema formally frozen | **Not yet** |
| Userspace emulator regression suite passing (M131) | **Not yet** |
| Desired prototype proof narrowed (BAR0-only / DMA / tensor math) | **Not yet** |
| Board shortlist reviewed and approved (M132) | **Not yet** |
| Hardware purchase go/no-go checkpoint passed (M133) | **Not yet** |

Current status: **5 of 12 conditions met.**

---

## 10. Risks

| Risk | Severity | Notes |
|---|---|---|
| Wrong board selection | Medium | Selecting a board before interfaces are frozen may require re-selection |
| Toolchain lock-in | Medium | Vivado/Quartus tool versions must be pinned; toolchain updates can break timing closure |
| PCIe IP friction | Medium–High | PCIe hard IP configuration is complex; incorrect lane/width settings prevent enumeration |
| Insufficient BRAM/DDR | Medium | Low-cost boards may lack BRAM for full command queue depth or trace log |
| Host compatibility issues | Low–Medium | Some desktop platforms do not support IOMMU group isolation per slot |
| Expensive idle hardware | Medium | FPGA board purchased before software is ready sits unused; opportunity cost |
| Scope creep into tensor math too early | High | Tensor math hardware requires much larger FPGA and much more engineering time; should be deferred |
| Memory SKU assumptions not validated | Medium | M120 modeled 16/32/64/128 GiB tile SKUs; real FPGA BRAM is much smaller; assumptions need re-baseline |
| Supply chain / availability | Low–Medium | FPGA dev board availability varies; lead times can extend 8–16 weeks |
| Register map churn before freeze | High | Any BAR0 offset or access-type change after FPGA RTL is written requires re-spin |
| Command schema churn | High | M103 packet format changes require command FSM re-design |
| Fabric model instability | Medium | M115/M117 route schema still evolving; fabric block must be re-designed if schema changes |

---

## 11. Future Milestone Proposals

| Milestone | Proposed title | Notes |
|---|---|---|
| M128 | Execution-plan validator | Validates M125 execution-plan JSON against placement and route reports |
| M129 | Execution-plan-to-command-plan mapper | Converts M125 advisory records into M109 command-plan JSON |
| M130 | Simulated AIMU EXEC_* replay, no tensor math | Replays M129 command plans through M121 emulator; counts, validates, reports |
| M131 | Userspace MMIO emulator regression suite | Full regression suite for M121–M125 paths |
| M132 | FPGA board shortlist update | Narrow board classes to 2–3 candidates when gate criteria approach completion |
| M133 | Hardware purchase go/no-go checkpoint | Formal authorization to purchase first FPGA dev board |

---

## 12. Non-Goals for M127

- No final board selection or purchase recommendation.
- No RTL of any kind.
- No Linux kernel driver.
- No real PCIe endpoint or real MMIO access.
- No FPGA synthesis or place-and-route.
- No C, Makefile, binary format, or inference behavior changes.
- No CUDA kernels or runtime changes.
- No tokenizer changes.
- No `.att1` format changes.
- No patent claim language.
- No tracked `__pycache__` or `*.pyc` files.
