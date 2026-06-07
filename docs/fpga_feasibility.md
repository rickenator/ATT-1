# ATT-1 FPGA Prototype Feasibility Notes (Milestone 126)

This document is a research and feasibility review only.  No RTL is
implemented, no board is selected or purchased, no kernel driver is written,
and no real PCIe endpoint is instantiated by this milestone.  All content is
advisory.

---

## 1. Executive Summary

**Recommendation: DEFER FPGA BUILD.**

The ATT-1 software/control-plane model is not yet stable enough to justify
committing FPGA engineering time.  The register map (M104), command packet
schema (M103), command queue simulator (M105), DMA descriptor model (M107),
MMIO emulator (M121), command-plan replay (M122), fabric route replay (M123),
tile memory allocator (M124), and tensor execution plan (M125) are all still
individually evolving.  None are formally frozen.

An FPGA prototype built now would need to be re-spun when any of these
interfaces change.  FPGA re-spin cycles are expensive in engineering time
and risk timing closure regressions.

**Proceed with:** software/userspace emulator path (Option B from M120).

**Revisit FPGA after:** the gate criteria in §10 are met.

---

## 2. FPGA Prototype Purpose

An FPGA prototype of the ATT-1 AIMU PCIe endpoint would prove the following
capabilities that cannot be validated in software simulation alone:

| Capability | Notes |
|---|---|
| PCIe enumeration | Device appears on the host PCI bus with correct vendor/device IDs |
| BAR0 MMIO access | Host can `mmap` BAR0 and read/write 32-bit registers |
| AXI-Lite register block | M104 register map implemented in FPGA fabric |
| Command queue doorbell | Host writes doorbell, FPGA processes one command from ring |
| Command completion | FPGA writes completion record, host polls or receives IRQ |
| DMA descriptor validation | FPGA reads descriptor ring, validates address/length, accepts or rejects |
| Tile capability registers | Host reads `TILE_COUNT`, `TILE_MEMORY_CAPACITY` from FPGA |
| Counter/trace registers | Host reads command counters and snapshot registers |
| Simple fabric route replay | Optional: FPGA acknowledges a fabric route without executing tensor math |

**What the FPGA prototype would NOT prove at this stage:**

- Actual tensor inference or AIMU matrix math.
- q8/q4 matmul hardware correctness.
- Multi-tile fabric switching.
- Full inference throughput.
- Production-grade timing or power.

---

## 3. Minimum Viable FPGA Design

The minimum viable AIMU FPGA prototype requires the following blocks:

### 3.1 PCIe endpoint

- PCIe Gen2 ×1 or ×4 endpoint IP.
- BAR0 mapped to AXI-Lite slave (64 KiB, matching M104 layout).
- Optional: BAR1 for future DMA target.
- Vendor PCIe hard IP (preferred) or soft PCIe (higher risk).

### 3.2 AXI-Lite register block (BAR0 slave)

- Implements M104 register map at correct BAR0 offsets.
- Enforces RO / RW / RW1C / WO semantics.
- Reserved offset reads return `0xDEAD_BEEF`.
- Reset behavior per §2.5 of M104.
- Register map version register returns `ATT1_AIMU_REGISTER_MAP_VERSION`.

### 3.3 Command queue RAM

- Small dual-port BRAM (e.g., 256 × 64-byte entries = 16 KiB).
- Host writes command records via BAR0 DMA target or MMIO window.
- `CMDQ_HEAD_PTR` / `CMDQ_TAIL_PTR` / `CMDQ_DOORBELL_OFFSET` registers.
- Simple FSM: reads next command on doorbell strobe, increments head pointer.

### 3.4 Completion queue RAM

- Matching BRAM for completion records.
- FSM writes completion record after each processed command.
- `CMPQ_HEAD_PTR` / `CMPQ_TAIL_PTR` registers.

### 3.5 Tile capability registers

- Static registers: `TILE_COUNT`, `TILE_MEMORY_CAPACITY_LOW/HIGH`.
- Values set at synthesis time or via configuration register.

### 3.6 Descriptor validator state machine

- Reads DMA descriptor fields (device_base, host_addr, byte_length, flags).
- Validates descriptor against tile memory capacity.
- Does not perform actual DMA; validates descriptor-only.
- Returns ATT1_OK or ATT1_ERR_RANGE in completion record.

### 3.7 Counter/trace block

- Command counter register increments on each processed command.
- Error counter register increments on each failed descriptor.
- Snapshot register captures counters on host write to `COUNTER_SNAPSHOT_TRIGGER`.
- Optionally: circular trace BRAM for command-type event log.

### 3.8 Interrupt (optional / deferred)

- MSI or MSI-X capable via PCIe endpoint IP.
- Deferred: first prototype can use host polling of `CMPQ_TAIL_PTR`.

### 3.9 DMA engine (deferred)

- Full DMA engine (host→device data movement) deferred to a future milestone.
- First prototype validates descriptors only; no actual DMA transfer.

---

## 4. Register Map to AXI-Lite Mapping

### 4.1 BAR0 offset layout

The M104 register map divides the 64 KiB BAR0 into regions:

| BAR0 region | Offset range | AXI-Lite slave | Contents |
|---|---|---|---|
| Global device | `0x0000–0x0FFF` | `reg_global` | DEVICE_ID, version, feature flags, tile count, tile memory |
| Command queue | `0x1000–0x1FFF` | `reg_cmdq` | Ring pointers, doorbell, command count |
| DMA/descriptor | `0x2000–0x2FFF` | `reg_dma` | Descriptor base address, length, flags, status |
| Fabric/interconnect | `0x3000–0x3FFF` | `reg_fabric` | Route counters, fabric policy |
| Counter | `0x4000–0x4FFF` | `reg_counter` | Command, error, completion, fabric counters |
| Trace/debug | `0x5000–0x5FFF` | `reg_trace` | Trace head/tail, snapshot trigger, snapshot buffer |
| Reserved | `0x6000–0xFFFF` | (reads `0xDEAD_BEEF`) | — |

### 4.2 Register semantics

| Access type | AXI-Lite read behavior | AXI-Lite write behavior |
|---|---|---|
| RO | Returns stored value | Write ignored; SLVERR may be asserted |
| RW | Returns stored value | Stores new value |
| RW1C | Returns stored value | Bit-clear: `reg <= reg & ~wdata` |
| WO | Returns 0 | Stores or triggers side-effect |

### 4.3 64-bit register pairs

M104 defines 64-bit quantities as `_LOW` / `_HIGH` pairs.  The AXI-Lite
slave must implement both halves as independent 32-bit registers.

Write order on hosts without 64-bit MMIO: write `_HIGH` first, then `_LOW`.
Read order: read `_LOW` first, then `_HIGH`.

The FPGA implementation should latch `_HIGH` on `_LOW` write to present a
coherent 64-bit view if the quantity changes asynchronously (e.g., byte
counters).

### 4.4 Doorbell write behavior

Writing any value to `CMDQ_DOORBELL_OFFSET` (WO, BAR0 offset `0x1010`):

1. Asserts a single-cycle strobe to the command-queue FSM.
2. FSM reads the next command from the ring at `CMDQ_HEAD_PTR`.
3. FSM increments `CMDQ_HEAD_PTR` (modulo ring depth).
4. FSM processes the command (validate, execute stub, write completion).
5. FSM increments `CMDQ_TAIL_PTR` on completion.

No AXI-Lite read of the doorbell register is meaningful; it returns 0.

### 4.5 Counter snapshot behavior

Writing any value to `COUNTER_SNAPSHOT_TRIGGER` (WO, BAR0 `0x4000`):

1. Latches all live counter values into shadow registers.
2. Host reads shadow registers via `COUNTER_SNAPSHOT_*` offsets.
3. Shadow values are stable until the next snapshot trigger.

### 4.6 Reset behavior

PCIe hot reset or write to `DEVICE_SOFT_RESET` (WO, BAR0 `0x0020`):

1. All RW registers return to reset values.
2. `CMDQ_HEAD_PTR` and `CMDQ_TAIL_PTR` reset to 0.
3. Command and error counters reset to 0.
4. Trace ring pointers reset to 0.
5. DEVICE_ID, REGISTER_MAP_VERSION, and feature flags are preserved (RO).

---

## 5. PCIe IP Options

This section surveys generic vendor-neutral considerations.  No specific
pricing is claimed; pricing changes frequently and should be verified directly
with vendors or distributors at purchase time.

### 5.1 Xilinx/AMD PCIe endpoint IP

- **Vivado PCIe IP core** (7-Series, UltraScale, UltraScale+): hard PCIe
  block on supported devices.  Exposes AXI4/AXI-Lite user interface.
- **Supported generations**: PCIe Gen1–Gen3 (device dependent), Gen4 on
  Versal.
- **AXI-Lite bridge**: included or available as a separate IP block.
- **DMA**: Xilinx DMA/Bridge subsystem (XDMA) IP available under Vivado;
  implements PCIe DMA with scatter-gather.
- **Host driver**: XDMA kernel driver available open-source from Xilinx/AMD
  GitHub; UIO-mode operation also supported.
- **Toolchain**: Vivado (full featured, free edition for most devices, paid
  edition for UltraScale+ HBM and some Versal).

### 5.2 Intel/Altera PCIe endpoint IP

- **Intel Quartus PCIe IP** (Stratix, Arria, Cyclone): PCIe hard IP on
  supported device families.  Exposes Avalon-MM or AXI interface.
- **AXI bridge**: available as IP or via Avalon-AXI bridges.
- **DMA**: Intel FPGA DMA IP available in Quartus IP catalog.
- **Host driver**: DFL (Device Feature List) framework in mainline Linux
  (5.4+) supports Intel FPGA devices for data center use cases.
- **Toolchain**: Quartus Prime Lite (free for Cyclone and some Arria devices),
  Pro edition for Stratix.

### 5.3 Open-source options

- **LitePCIe**: pure-Python/Migen PCIe endpoint + DMA stack.
  - Target device: Xilinx 7-Series and UltraScale (soft PHY or via hard IP
    wrapper).
  - Suitable for academic/research prototypes.
  - Limited to PCIe Gen2 in most configurations.
  - Host-side: custom kernel driver or VFIO; examples available.
  - Risk: limited timing closure support vs. vendor hard IP; community
    maintained.
- **OpenCores PCIe**: various maturity levels; generally not recommended for
  PCIe Gen2+ due to timing complexity.

### 5.4 AXI-Lite bridge

All three paths above support an AXI-Lite slave bridge for register block
access.  AXI-Lite is preferred for the M104 register block because:
- Simple request/response (no burst).
- Naturally maps to 32-bit MMIO reads/writes.
- No out-of-order transaction support needed for register access.

### 5.5 DMA IP

Full DMA support (host→device tensor weight loading) is deferred from the
minimum viable FPGA design.  When added:
- Xilinx: XDMA subsystem supports scatter-gather DMA.
- Intel: Intel FPGA DMA and PCIe AVMM IP.
- LitePCIe: includes a basic DMA engine.

First prototype uses descriptor-only validation (§3.6); no actual DMA
transfer occurs.

### 5.6 Host access model

| Method | Kernel mode | Notes |
|---|---|---|
| `/dev/mem` + `mmap` | No (root) | Debugging only; not secure |
| UIO (Userspace I/O) | Yes (thin driver) | Simple BAR mmap to userspace |
| VFIO | Yes (thin driver) | IOMMU-safe; supports DMA from userspace |
| Custom `pci_driver` | Yes (full driver) | Full control; required for MSI/MSI-X |
| `libpciaccess` | No (root) | Config space access only; not MMIO |

For the ATT-1 prototype, VFIO is the preferred future path because:
- Provides IOMMU protection for DMA operations.
- Allows DMA mapping from userspace.
- Matches the M121 userspace emulator philosophy.
- Defers kernel driver complexity.

---

## 6. Host Software Options

The ATT-1 host software stack currently operates entirely in userspace via the
M121 mmap-backed BAR0 emulator.  When transitioning to a real FPGA:

### 6.1 Current path: userspace mmap emulator (M121)

- `att1_aimu_userspace_open()` mmap-backs a regular file to emulate BAR0.
- No kernel driver, no real PCI device.
- Advantages: zero hardware dependency; CI-safe; cross-host portable.
- Limitation: cannot validate actual PCIe enumeration or DMA.

### 6.2 UIO-style userspace BAR mapping

- Thin Linux kernel UIO driver exposes BAR0 as a character device.
- Userspace calls `mmap()` on the UIO device file to access BAR0.
- Advantages: simple; no full kernel driver required.
- Disadvantages: no DMA IOMMU protection; unsafe for production DMA.
- Migration from M121: replace the mmap-backed file with the UIO mmap target;
  all `att1_aimu_userspace_*` API calls remain unchanged.

### 6.3 VFIO

- Linux VFIO framework allows userspace processes to directly access PCIe
  device BARs and program DMA via IOMMU.
- Advantages: IOMMU protection; suitable for DMA; mainline kernel support.
- Disadvantages: more complex setup (IOMMU group isolation, VFIO bind).
- Preferred for the first real-hardware path once DMA is needed.

### 6.4 Custom kernel `pci_driver`

- Full Linux `pci_driver` with `probe()`/`remove()` lifecycle.
- Necessary for: MSI/MSI-X interrupt registration; fine-grained DMA mapping;
  power management; hot-plug.
- Deferred: adds significant kernel engineering burden; not needed until
  interrupt-driven completions or production DMA are required.

### 6.5 Why kernel driver is deferred

- VFIO covers the first real-hardware DMA use case without a custom driver.
- M121 emulator already validates all control-plane logic in userspace.
- Kernel driver churn tracks register map changes; deferring avoids
  double-maintenance until interfaces are frozen.

---

## 7. Prototype Board Considerations

This section discusses board categories only.  No specific board is selected
or purchased.

### 7.1 PCIe generation and lanes

- **Minimum**: PCIe Gen1 ×1 (250 MB/s unidirectional) — sufficient for
  register access and descriptor-only validation.
- **Preferred**: PCIe Gen2 ×4 (2 GB/s) — sufficient for practical DMA weight
  loading during early prototype phases.
- **Future**: PCIe Gen3 ×8 or ×16 for production throughput targets.

### 7.2 FPGA fabric resources

| Resource | Minimum FPGA requirement |
|---|---|
| BRAM | ≥4 Mb (command/completion queue RAM + trace RAM) |
| URAM | Not required for minimum viable design |
| DSP slices | Not required for register-only design; needed for future matmul |
| LUT/FF | ~20K–50K LUTs estimated for PCIe IP + register block + FSMs |
| DDR/HBM | Not required for minimum viable; needed for full tensor weight storage |
| PCIe hard block | Required (soft PCIe Gen2+ is high risk for timing closure) |

### 7.3 Toolchain friction

- Xilinx/AMD Vivado: mature, well-documented, free tier covers most 7-Series
  and UltraScale devices; recommended starting point.
- Intel/Altera Quartus: mature; free tier covers Cyclone and some Arria.
- Open-source (LitePCIe): viable for research; expect more toolchain effort.
- All require Linux host for Makefile-driven flows; Vivado and Quartus both
  have Linux-native versions.

### 7.4 Linux host compatibility

- PCIe endpoint card requires a Linux host with an available PCIe x4 or x8
  physical slot.
- IOMMU must be enabled in BIOS/UEFI for VFIO DMA mapping.
- Kernel version ≥5.10 recommended (LTS; includes mainline VFIO improvements).
- No special CPU architecture requirement (x86-64 standard).

### 7.5 Physical slot and power

- Half-height or full-height PCIe card form factor.
- 75W slot power budget (PCIe card edge) is sufficient for most mid-range
  FPGA boards; boards with DDR or high-density BRAM may require PCIe aux power
  connector or ATX power supply.
- Cooling: passive or low-speed active fan on most FPGA dev boards; adequate
  for prototype power levels.

### 7.6 Cost class

Cost information is approximate and subject to change; verify with vendors or
distributors before planning:

| Class | FPGA family examples | Approximate cost range |
|---|---|---|
| Low cost | Xilinx Artix-7, Intel Cyclone V | USD 100–500 (board) |
| Mid range | Xilinx Kintex-7/UltraScale, Intel Arria 10 | USD 500–3000 (board) |
| High end | Xilinx UltraScale+ HBM, Intel Stratix 10 | USD 3000–15000+ (board) |

For the minimum viable ATT-1 prototype (register block + command queue +
descriptor validator), a low-cost or mid-range board is sufficient.  High-end
boards are not needed until tensor matmul hardware (q8/q4) is targeted.

---

## 8. Resource Estimate Categories

Qualitative resource estimates for each ATT-1 FPGA subsystem.  These are
not synthesis reports; they are planning-level assessments.

| Subsystem | FPGA resource cost | Notes |
|---|---|---|
| BAR0 register block (AXI-Lite slave) | **Low** | ~500–2000 LUTs; fits any target device |
| Command queue RAM | **Low** | 16–64 KiB BRAM (1–4 RAMB36 on 7-Series) |
| Completion queue RAM | **Low** | Same order as command queue |
| Descriptor validator FSM | **Low / Moderate** | ~200–500 LUTs; no external memory needed |
| Counter/trace block | **Low** | ~300–800 LUTs + small BRAM for trace log |
| PCIe endpoint IP | **Fixed (hard IP)** | Vendor-provided; no LUT cost for hard block |
| DMA engine (scatter-gather) | **Moderate / High** | ~5K–15K LUTs; requires DDR/URAM for descriptor rings |
| Fabric routing logic | **Moderate / High** | Depends on route count and arbitration policy |
| Tensor math engine (f32/q8) | **High** | DSP-intensive; requires DSP cascade blocks |
| q4/q8 matmul hardware | **High** | Custom datapath; requires careful pipelining |
| Full AIMU tile (all of above) | **Very High** | Multi-month engineering; high-end device required |

---

## 9. Risks

| Risk | Severity | Notes |
|---|---|---|
| PCIe IP complexity | Medium | Vendor hard IP reduces risk; soft PCIe is high risk |
| Timing closure | Medium–High | Hard for complex designs on smaller devices |
| Driver/debug time | Medium | VFIO simplifies; UIO is simpler still for first pass |
| DMA correctness | High | Scatter-gather DMA is subtle; address translation errors silent |
| Host platform compatibility | Low–Medium | PCIe enumeration issues on some desktop boards; server platforms more reliable |
| Board cost and availability | Low–Medium | Supply-chain variability; verify before committing |
| FPGA resource limits on small devices | Medium | Command queue + PCIe IP may fill Artix-7 with little room for future expansion |
| Fabric model instability | High | M123 fabric route schema is not frozen; FPGA fabric block would need re-spin |
| Register map churn | High | M104 register map not frozen; any offset/access-type change requires FPGA re-spin |
| Command schema churn | High | M103 command packet format not frozen |
| Hardware effort distraction | High | FPGA engineering diverts attention from completing M126–M131 software milestones |
| Interrupt design complexity | Low (deferred) | MSI-X requires kernel driver changes; polling avoids this initially |

---

## 10. Recommended FPGA Gate Criteria

FPGA RTL development must not begin until **all** of the following criteria
are met:

| Gate criterion | Status | Required milestone |
|---|---|---|
| Userspace MMIO emulator stable and regression-tested | Complete | M121 |
| Command-plan replay through emulator stable | Complete | M122 |
| Fabric route replay stable | Complete | M123 |
| Tile memory allocator interface stable | Complete | M124 |
| Tensor execution plan stable | Complete | M125 |
| Register map versioning formally frozen | **Not yet** | Requires explicit freeze decision |
| Command packet schema formally frozen | **Not yet** | M103 schema still evolving |
| Fabric route schema formally frozen | **Not yet** | M115/M117 schema still evolving |
| Minimal host software path selected (UIO vs. VFIO) | **Not yet** | Requires explicit selection |
| Execution plan (M125) regression suite passing | **Not yet** | M131 (planned) |
| FPGA go/no-go review completed | **Not yet** | M132 (planned) |

Current gate status: **2 of 11 criteria met.**  FPGA build is not authorized.

---

## 11. Recommendation

| Decision | Rationale |
|---|---|
| **NOT YET: Do not start FPGA RTL** | Too many upstream interfaces still evolving |
| **Continue software/userspace emulator path** | M121–M125 provide correct behavior validation without hardware cost |
| **Prepare FPGA feasibility notes only** | This document (M126) is the correct deliverable |
| **Revisit after interfaces are frozen** | M132 FPGA go/no-go update is the planned re-evaluation point |
| **Next software milestones: M127–M131** | Complete the control-plane and execution pipeline before committing to hardware |

The FPGA path remains available and technically feasible once interfaces
stabilize.  The minimum viable FPGA design (§3) is well-scoped for a
mid-range Xilinx or Intel board and does not require tensor math hardware.
However, committing to hardware before the register map, command schema, and
fabric route schema are frozen would result in repeated re-spins.

---

## 12. Future Milestone Proposals

| Milestone | Proposed title | Notes |
|---|---|---|
| M127 | Phase 3 prototype BOM and board options review | Formal board option analysis when gate criteria approach completion |
| M128 | Execution-plan validator | Validates M125 execution-plan JSON against placement report and route report |
| M129 | Execution-plan-to-command-plan mapper | Converts M125 advisory records into M109 command-plan format |
| M130 | Simulated AIMU EXEC_* replay, no tensor math | Replays M129 command plans through M121 emulator; counts, validates, reports |
| M131 | Userspace MMIO emulator regression suite | Full regression suite for M121 + M122 + M123 + M124 paths |
| M132 | FPGA go/no-go update | Re-evaluate FPGA gate criteria after M127–M131 complete |

---

## 13. Non-Goals for M126

- No RTL of any kind.
- No board selection or purchase commitment.
- No kernel driver.
- No real PCIe endpoint or real MMIO access.
- No FPGA synthesis or place-and-route.
- No C, Makefile, binary format, or inference behavior changes.
- No CUDA kernels or runtime changes.
- No tokenizer changes.
- No `.att1` format changes.
- No patent claim language.
- No tracked `__pycache__` or `*.pyc` files.
