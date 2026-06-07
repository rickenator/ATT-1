/*
 * att1_aimu_mmio.h  —  AIMU MMIO/register-file simulator (M111)
 *
 * In-process simulator for the 64 KiB BAR0 MMIO register space defined in
 * M104 (docs/aimu_register_map.md).  Host code can read and write simulated
 * device, tile, command-queue, DMA, fabric, trace, and counter registers
 * without requiring real PCIe hardware, an MMIO mapping, or a kernel driver.
 *
 * This is NOT a real PCIe endpoint, MMIO accessor, DMA engine, interrupt
 * controller, or kernel driver.  All state lives in a single heap-allocated
 * 64 KiB backing array plus a few pointer fields.
 *
 * Register semantics (M104 §1.2)
 * --------------------------------
 *   RO   — read-only; write returns ATT1_ERR_UNSUPPORTED
 *   RW   — read-write; value persists
 *   RW1C — read-write-one-to-clear; write 1 clears bits; write 0 is no-op
 *   WO   — write-only; read returns 0; write performs a side-effect
 *
 * Reserved offsets return 0xDEADBEEF on read; writes are silently discarded.
 *
 * Alignment requirement
 * ----------------------
 * Every register offset must be a multiple of 4.  64-bit helpers require
 * the LOW register offset to be a multiple of 8.  Unaligned accesses return
 * ATT1_ERR_INVALID_ARG.
 *
 * Integration with other simulator modules
 * -----------------------------------------
 *   att1_aimu_mmio_attach_device() — copies M106 device/tile register values
 *       into the RO register cells and wires TILE_RESET_CONTROL side-effects
 *       to att1_aimu_device_reset_tile().
 *
 *   att1_aimu_mmio_attach_cmdq()   — wires CQ_HEAD/TAIL/STATUS from the M105
 *       ring-buffer state; CQ_DOORBELL writes increment a doorbell counter.
 *
 *   att1_aimu_mmio_attach_trace()  — wires COUNTER_SNAPSHOT_CONTROL writes
 *       to att1_aimu_trace_snapshot_all().
 *
 * Call att1_aimu_mmio_sync() after attaching sources (and after any source
 * mutation) to refresh all RO register cells from the live simulator state.
 *
 * No ATT-1 inference, backend, tokenizer, CUDA, or binary format behaviour is
 * changed by this module.
 */

#ifndef ATT1_AIMU_MMIO_H
#define ATT1_AIMU_MMIO_H

#include "att1_status.h"
#include "att1_aimu_device.h"   /* att1_aimu_device, att1_aimu_tile_info */
#include "att1_aimu_cmdq.h"     /* att1_aimu_cmdq                        */
#include "att1_aimu_trace.h"    /* att1_aimu_trace                       */

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

/* =========================================================================
 * Compile-time constants
 * ====================================================================== */

/** Magic written into every att1_aimu_mmio to detect use-after-free. */
#define ATT1_AIMU_MMIO_MAGIC    UINT32_C(0xA771BB10)

/**
 * BAR0 size: 64 KiB.
 * Matches M104 §1.5 address map (0x0000–0xFFFF).
 */
#define ATT1_AIMU_MMIO_BAR0_SIZE    UINT32_C(0x10000)

/** Number of 32-bit register slots in the backing array. */
#define ATT1_AIMU_MMIO_NREGS        (ATT1_AIMU_MMIO_BAR0_SIZE / 4u)

/**
 * Value returned on reads to reserved/unassigned offsets.
 * Mirrors M104 §1.3: "reserved registers read as 0xDEAD_BEEF".
 */
#define ATT1_AIMU_MMIO_RESERVED_RD  UINT32_C(0xDEADBEEF)

/* =========================================================================
 * BAR0 region base offsets  (M104 §1.5)
 * ====================================================================== */

#define ATT1_MMIO_BASE_GLOBAL   UINT32_C(0x0000)  /**< Global device registers */
#define ATT1_MMIO_BASE_CQ       UINT32_C(0x1000)  /**< Command queue registers */
#define ATT1_MMIO_BASE_DMA      UINT32_C(0x2000)  /**< DMA registers           */
#define ATT1_MMIO_BASE_FABRIC   UINT32_C(0x3000)  /**< Fabric registers        */
#define ATT1_MMIO_BASE_COUNTER  UINT32_C(0x4000)  /**< Counter registers       */
#define ATT1_MMIO_BASE_TRACE    UINT32_C(0x5000)  /**< Trace/debug registers   */
#define ATT1_MMIO_BASE_TILE(N)  (UINT32_C(0x8000) + (uint32_t)(N) * UINT32_C(0x800))

/* =========================================================================
 * Global device register offsets  (M104 §2)
 * ====================================================================== */

#define ATT1_MMIO_DEVICE_ID                 UINT32_C(0x0000)  /**< RO  */
#define ATT1_MMIO_DEVICE_VERSION            UINT32_C(0x0004)  /**< RO  */
#define ATT1_MMIO_REGISTER_MAP_VERSION      UINT32_C(0x0008)  /**< RO  */
#define ATT1_MMIO_FEATURE_FLAGS_LOW         UINT32_C(0x000C)  /**< RO  */
#define ATT1_MMIO_FEATURE_FLAGS_HIGH        UINT32_C(0x0010)  /**< RO  */
#define ATT1_MMIO_TILE_COUNT                UINT32_C(0x0014)  /**< RO  */
#define ATT1_MMIO_COMMAND_QUEUE_COUNT       UINT32_C(0x0018)  /**< RO  */
#define ATT1_MMIO_INTERRUPT_STATUS          UINT32_C(0x001C)  /**< RW1C */
#define ATT1_MMIO_INTERRUPT_ENABLE          UINT32_C(0x0020)  /**< RW  */
#define ATT1_MMIO_GLOBAL_STATUS             UINT32_C(0x0024)  /**< RO  */
#define ATT1_MMIO_GLOBAL_CONTROL            UINT32_C(0x0028)  /**< WO  (self-clearing) */
#define ATT1_MMIO_RESET_CONTROL             UINT32_C(0x002C)  /**< WO  (self-clearing) */
#define ATT1_MMIO_ERROR_STATUS              UINT32_C(0x0030)  /**< RW1C */
#define ATT1_MMIO_ERROR_DETAIL              UINT32_C(0x0034)  /**< RO  */
#define ATT1_MMIO_TRACE_CONTROL             UINT32_C(0x0038)  /**< RW  */
#define ATT1_MMIO_COUNTER_SNAPSHOT_CONTROL  UINT32_C(0x003C)  /**< RW  */

/* =========================================================================
 * GLOBAL_STATUS bit definitions  (M104 §2.10)
 * ====================================================================== */

#define ATT1_MMIO_GSTAT_DEVICE_READY    (UINT32_C(1) << 0)
#define ATT1_MMIO_GSTAT_ANY_TILE_ACTIVE (UINT32_C(1) << 1)
#define ATT1_MMIO_GSTAT_ANY_TILE_ERROR  (UINT32_C(1) << 2)
#define ATT1_MMIO_GSTAT_FABRIC_ACTIVE   (UINT32_C(1) << 3)
#define ATT1_MMIO_GSTAT_DMA_ACTIVE      (UINT32_C(1) << 4)
#define ATT1_MMIO_GSTAT_TRACE_ACTIVE    (UINT32_C(1) << 5)

/* GLOBAL_CONTROL bit definitions (M104 §2.11) */
#define ATT1_MMIO_GCTRL_ENABLE_DEVICE   (UINT32_C(1) << 0)
#define ATT1_MMIO_GCTRL_DISABLE_DEVICE  (UINT32_C(1) << 1)
#define ATT1_MMIO_GCTRL_ENABLE_TRACE    (UINT32_C(1) << 2)
#define ATT1_MMIO_GCTRL_DISABLE_TRACE   (UINT32_C(1) << 3)
#define ATT1_MMIO_GCTRL_FLUSH_COMPL     (UINT32_C(1) << 4)

/* RESET_CONTROL bit definitions (M104 §2.12) */
#define ATT1_MMIO_RSTCTL_SOFT_RESET_ALL (UINT32_C(1) << 0)
#define ATT1_MMIO_RSTCTL_RESET_COUNTERS (UINT32_C(1) << 1)
#define ATT1_MMIO_RSTCTL_RESET_TRACE    (UINT32_C(1) << 2)
#define ATT1_MMIO_RSTCTL_RESET_FABRIC   (UINT32_C(1) << 3)

/* COUNTER_SNAPSHOT_CONTROL bit definitions (M104 §2.16) */
#define ATT1_MMIO_SNAP_NOW              (UINT32_C(1) << 0)

/* =========================================================================
 * Per-tile register offsets  (M104 §3 — relative to tile window base)
 * ====================================================================== */

#define ATT1_MMIO_TILE_ID                   UINT32_C(0x000)  /**< RO  */
#define ATT1_MMIO_TILE_STATUS               UINT32_C(0x004)  /**< RO  */
#define ATT1_MMIO_TILE_FEATURE_FLAGS        UINT32_C(0x008)  /**< RO  */
#define ATT1_MMIO_TILE_MEMORY_CAPACITY_LOW  UINT32_C(0x00C)  /**< RO  */
#define ATT1_MMIO_TILE_MEMORY_CAPACITY_HIGH UINT32_C(0x010)  /**< RO  */
#define ATT1_MMIO_TILE_MEMORY_USED_LOW      UINT32_C(0x014)  /**< RO  */
#define ATT1_MMIO_TILE_MEMORY_USED_HIGH     UINT32_C(0x018)  /**< RO  */
#define ATT1_MMIO_TILE_KV_CAPACITY_LOW      UINT32_C(0x01C)  /**< RO  */
#define ATT1_MMIO_TILE_KV_CAPACITY_HIGH     UINT32_C(0x020)  /**< RO  */
#define ATT1_MMIO_TILE_KV_USED_LOW          UINT32_C(0x024)  /**< RO  */
#define ATT1_MMIO_TILE_KV_USED_HIGH         UINT32_C(0x028)  /**< RO  */
#define ATT1_MMIO_TILE_SUPPORTED_DTYPES     UINT32_C(0x02C)  /**< RO  */
#define ATT1_MMIO_TILE_SUPPORTED_OPS_LOW    UINT32_C(0x030)  /**< RO  */
#define ATT1_MMIO_TILE_SUPPORTED_OPS_HIGH   UINT32_C(0x034)  /**< RO  */
#define ATT1_MMIO_TILE_FABRIC_LINK_MASK     UINT32_C(0x038)  /**< RO  */
#define ATT1_MMIO_TILE_ERROR_STATUS         UINT32_C(0x03C)  /**< RW1C */
#define ATT1_MMIO_TILE_RESET_CONTROL        UINT32_C(0x040)  /**< WO  */

/* Convenience: absolute offset for tile N, field F */
#define ATT1_MMIO_TILE_REG(N, F) \
    (ATT1_MMIO_BASE_TILE(N) + (uint32_t)(F))

/* =========================================================================
 * Command queue register offsets  (M104 §4)
 * ====================================================================== */

#define ATT1_MMIO_CQ_BASE_ADDR_LOW          UINT32_C(0x1000)  /**< RW  */
#define ATT1_MMIO_CQ_BASE_ADDR_HIGH         UINT32_C(0x1004)  /**< RW  */
#define ATT1_MMIO_CQ_SIZE                   UINT32_C(0x1008)  /**< RW  */
#define ATT1_MMIO_CQ_HEAD                   UINT32_C(0x100C)  /**< RO  */
#define ATT1_MMIO_CQ_TAIL                   UINT32_C(0x1010)  /**< RW  */
#define ATT1_MMIO_CQ_DOORBELL               UINT32_C(0x1014)  /**< WO  */
#define ATT1_MMIO_CQ_STATUS                 UINT32_C(0x1018)  /**< RO  */
#define ATT1_MMIO_CQ_ERROR                  UINT32_C(0x101C)  /**< RW1C */
#define ATT1_MMIO_CQ_FENCE_VALUE            UINT32_C(0x1020)  /**< RO  */
#define ATT1_MMIO_CQ_COMPLETION_ADDR_LOW    UINT32_C(0x1024)  /**< RW  */
#define ATT1_MMIO_CQ_COMPLETION_ADDR_HIGH   UINT32_C(0x1028)  /**< RW  */
#define ATT1_MMIO_CQ_COMPLETION_SIZE        UINT32_C(0x102C)  /**< RW  */

/* =========================================================================
 * DMA register offsets  (M104 §5)
 * ====================================================================== */

#define ATT1_MMIO_DMA_CONTROL               UINT32_C(0x2000)  /**< RW  */
#define ATT1_MMIO_DMA_STATUS                UINT32_C(0x2004)  /**< RO  */
#define ATT1_MMIO_DMA_ERROR_STATUS          UINT32_C(0x2008)  /**< RW1C */
#define ATT1_MMIO_DMA_RING_BASE_LOW         UINT32_C(0x200C)  /**< RW  */
#define ATT1_MMIO_DMA_RING_BASE_HIGH        UINT32_C(0x2010)  /**< RW  */
#define ATT1_MMIO_DMA_RING_SIZE             UINT32_C(0x2014)  /**< RW  */

/* =========================================================================
 * Fabric register offsets  (M104 §6)
 * ====================================================================== */

#define ATT1_MMIO_FABRIC_STATUS             UINT32_C(0x3000)  /**< RO  */
#define ATT1_MMIO_FABRIC_CONTROL            UINT32_C(0x3004)  /**< RW  */
#define ATT1_MMIO_FABRIC_ROUTE_BASE_LOW     UINT32_C(0x3008)  /**< RW  */
#define ATT1_MMIO_FABRIC_ROUTE_BASE_HIGH    UINT32_C(0x300C)  /**< RW  */
#define ATT1_MMIO_FABRIC_ROUTE_SIZE         UINT32_C(0x3010)  /**< RW  */
#define ATT1_MMIO_FABRIC_PKT_CTR_LOW        UINT32_C(0x3014)  /**< RO  */
#define ATT1_MMIO_FABRIC_PKT_CTR_HIGH       UINT32_C(0x3018)  /**< RO  */
#define ATT1_MMIO_FABRIC_PAYLOAD_BYTES_LOW  UINT32_C(0x301C)  /**< RO  */
#define ATT1_MMIO_FABRIC_PAYLOAD_BYTES_HIGH UINT32_C(0x3020)  /**< RO  */
#define ATT1_MMIO_FABRIC_CONGESTION_CTR     UINT32_C(0x3024)  /**< RO  */
#define ATT1_MMIO_FABRIC_ERROR_STATUS       UINT32_C(0x3028)  /**< RW1C */

/* =========================================================================
 * Counter register offsets  (M104 §7 — 64-bit pairs; all RO)
 *
 * Each 64-bit counter is two consecutive 32-bit registers: _LO then _HI.
 * Base: 0x4000.  Stride: 8 bytes per counter.
 * ====================================================================== */

#define ATT1_MMIO_CNT_BASE                  UINT32_C(0x4000)

#define ATT1_MMIO_CNT_CMD_ISSUED_LO         UINT32_C(0x4000)  /**< RO */
#define ATT1_MMIO_CNT_CMD_ISSUED_HI         UINT32_C(0x4004)  /**< RO */
#define ATT1_MMIO_CNT_CMD_COMPLETED_LO      UINT32_C(0x4008)  /**< RO */
#define ATT1_MMIO_CNT_CMD_COMPLETED_HI      UINT32_C(0x400C)  /**< RO */
#define ATT1_MMIO_CNT_LOCAL_OPS_LO          UINT32_C(0x4010)  /**< RO */
#define ATT1_MMIO_CNT_LOCAL_OPS_HI          UINT32_C(0x4014)  /**< RO */
#define ATT1_MMIO_CNT_MATMUL_LO             UINT32_C(0x4018)  /**< RO */
#define ATT1_MMIO_CNT_MATMUL_HI             UINT32_C(0x401C)  /**< RO */
#define ATT1_MMIO_CNT_RMSNORM_LO            UINT32_C(0x4020)  /**< RO */
#define ATT1_MMIO_CNT_RMSNORM_HI            UINT32_C(0x4024)  /**< RO */
#define ATT1_MMIO_CNT_ROPE_LO               UINT32_C(0x4028)  /**< RO */
#define ATT1_MMIO_CNT_ROPE_HI               UINT32_C(0x402C)  /**< RO */
#define ATT1_MMIO_CNT_ATTENTION_LO          UINT32_C(0x4030)  /**< RO */
#define ATT1_MMIO_CNT_ATTENTION_HI          UINT32_C(0x4034)  /**< RO */
#define ATT1_MMIO_CNT_FFN_LO                UINT32_C(0x4038)  /**< RO */
#define ATT1_MMIO_CNT_FFN_HI                UINT32_C(0x403C)  /**< RO */
#define ATT1_MMIO_CNT_KV_APPENDS_LO         UINT32_C(0x4040)  /**< RO */
#define ATT1_MMIO_CNT_KV_APPENDS_HI         UINT32_C(0x4044)  /**< RO */
#define ATT1_MMIO_CNT_KV_READS_LO           UINT32_C(0x4048)  /**< RO */
#define ATT1_MMIO_CNT_KV_READS_HI           UINT32_C(0x404C)  /**< RO */
#define ATT1_MMIO_CNT_TENSOR_BYTES_RD_LO    UINT32_C(0x4050)  /**< RO */
#define ATT1_MMIO_CNT_TENSOR_BYTES_RD_HI    UINT32_C(0x4054)  /**< RO */
#define ATT1_MMIO_CNT_TENSOR_BYTES_WR_LO    UINT32_C(0x4058)  /**< RO */
#define ATT1_MMIO_CNT_TENSOR_BYTES_WR_HI    UINT32_C(0x405C)  /**< RO */
#define ATT1_MMIO_CNT_ACT_BYTES_SENT_LO     UINT32_C(0x4060)  /**< RO */
#define ATT1_MMIO_CNT_ACT_BYTES_SENT_HI     UINT32_C(0x4064)  /**< RO */
#define ATT1_MMIO_CNT_ACT_BYTES_RECV_LO     UINT32_C(0x4068)  /**< RO */
#define ATT1_MMIO_CNT_ACT_BYTES_RECV_HI     UINT32_C(0x406C)  /**< RO */
#define ATT1_MMIO_CNT_LOGITS_BYTES_LO       UINT32_C(0x4070)  /**< RO */
#define ATT1_MMIO_CNT_LOGITS_BYTES_HI       UINT32_C(0x4074)  /**< RO */
#define ATT1_MMIO_CNT_FABRIC_PKT_SENT_LO    UINT32_C(0x4078)  /**< RO */
#define ATT1_MMIO_CNT_FABRIC_PKT_SENT_HI    UINT32_C(0x407C)  /**< RO */
#define ATT1_MMIO_CNT_FABRIC_PKT_RECV_LO    UINT32_C(0x4080)  /**< RO */
#define ATT1_MMIO_CNT_FABRIC_PKT_RECV_HI    UINT32_C(0x4084)  /**< RO */
#define ATT1_MMIO_CNT_STALL_FENCE_LO        UINT32_C(0x4088)  /**< RO */
#define ATT1_MMIO_CNT_STALL_FENCE_HI        UINT32_C(0x408C)  /**< RO */
#define ATT1_MMIO_CNT_STALL_DMA_LO          UINT32_C(0x4090)  /**< RO */
#define ATT1_MMIO_CNT_STALL_DMA_HI          UINT32_C(0x4094)  /**< RO */
#define ATT1_MMIO_CNT_STALL_FABRIC_LO       UINT32_C(0x4098)  /**< RO */
#define ATT1_MMIO_CNT_STALL_FABRIC_HI       UINT32_C(0x409C)  /**< RO */
#define ATT1_MMIO_CNT_STALL_BARRIER_LO      UINT32_C(0x40A0)  /**< RO */
#define ATT1_MMIO_CNT_STALL_BARRIER_HI      UINT32_C(0x40A4)  /**< RO */
#define ATT1_MMIO_CNT_STALL_QUEUE_FULL_LO   UINT32_C(0x40A8)  /**< RO */
#define ATT1_MMIO_CNT_STALL_QUEUE_FULL_HI   UINT32_C(0x40AC)  /**< RO */
#define ATT1_MMIO_CNT_ERRORS_TOTAL_LO       UINT32_C(0x40B0)  /**< RO */
#define ATT1_MMIO_CNT_ERRORS_TOTAL_HI       UINT32_C(0x40B4)  /**< RO */

/* End of counter region (exclusive): 0x40B8 */
#define ATT1_MMIO_CNT_END                   UINT32_C(0x40B8)

/* =========================================================================
 * Trace/debug register offsets  (M104 §8)
 * ====================================================================== */

#define ATT1_MMIO_TRACE_BUFFER_BASE_LOW     UINT32_C(0x5000)  /**< RW  */
#define ATT1_MMIO_TRACE_BUFFER_BASE_HIGH    UINT32_C(0x5004)  /**< RW  */
#define ATT1_MMIO_TRACE_BUFFER_SIZE         UINT32_C(0x5008)  /**< RW  */
#define ATT1_MMIO_TRACE_WRITE_PTR           UINT32_C(0x500C)  /**< RO  */
#define ATT1_MMIO_TRACE_FLAGS               UINT32_C(0x5010)  /**< RW  */
#define ATT1_MMIO_TRACE_DROPPED_EVENTS      UINT32_C(0x5014)  /**< RO  */
#define ATT1_MMIO_TRACE_SNAPSHOT_CONTROL    UINT32_C(0x5018)  /**< RW  */

/* =========================================================================
 * DEVICE_ID field helpers  (M104 §2.1)
 * ====================================================================== */

/** ATT-1 vendor identifier in the DEVICE_ID register (bits 31:16). */
#define ATT1_MMIO_VENDOR_ID             UINT32_C(0xA771)

/** AIMU inference tile device class (bits 15:0). */
#define ATT1_MMIO_DEVICE_CLASS_TILE     UINT32_C(0x0001)

/** Default DEVICE_ID register value. */
#define ATT1_MMIO_DEVICE_ID_DEFAULT \
    ((ATT1_MMIO_VENDOR_ID << 16u) | ATT1_MMIO_DEVICE_CLASS_TILE)

/* =========================================================================
 * Simulator object
 * ====================================================================== */

/**
 * att1_aimu_mmio
 *
 * Opaque handle for the MMIO/register-file simulator.  Allocate with
 * att1_aimu_mmio_create(); free with att1_aimu_mmio_destroy().
 */
typedef struct att1_aimu_mmio {
    uint32_t            magic;

    /**
     * Flat backing store for all BAR0 registers.
     * Indexed as regs[offset / 4].
     */
    uint32_t            regs[ATT1_AIMU_MMIO_NREGS];

    /**
     * Optional attached M106 device/tile simulator.
     * NULL if not attached.  The MMIO simulator does NOT own this pointer.
     */
    att1_aimu_device   *device;

    /**
     * Optional attached M105 command-queue simulator.
     * NULL if not attached.  The MMIO simulator does NOT own this pointer.
     */
    att1_aimu_cmdq     *cmdq;

    /**
     * Optional attached M108 trace/counter snapshot source.
     * NULL if not attached.  The MMIO simulator does NOT own this pointer.
     */
    att1_aimu_trace    *trace;

    /**
     * Doorbell write counter.
     * Incremented every time the host writes ATT1_MMIO_CQ_DOORBELL.
     * Not an MMIO register; used in tests to confirm the doorbell path.
     */
    uint32_t            doorbell_write_count;

    /**
     * Counter snapshot trigger count.
     * Incremented every time the host writes COUNTER_SNAPSHOT_CONTROL with
     * SNAP_NOW set.  Used in tests to confirm the snapshot path.
     */
    uint32_t            snapshot_trigger_count;
} att1_aimu_mmio;

/* =========================================================================
 * Lifecycle
 * ====================================================================== */

/**
 * att1_aimu_mmio_create
 *
 * Allocate and initialise the MMIO/register-file simulator.
 *
 * The backing register file is cleared to 0.  Reserved offsets are NOT
 * pre-populated with 0xDEADBEEF in the backing store; the read path returns
 * that sentinel dynamically for any offset that has no assigned register.
 *
 * The DEVICE_ID, REGISTER_MAP_VERSION, and GLOBAL_STATUS.DEVICE_READY
 * registers are pre-populated with their architectural defaults.
 *
 * @param out  Output pointer.  Must not be NULL.
 * @return ATT1_OK on success.
 * @return ATT1_ERR_INVALID_ARG if out is NULL.
 * @return ATT1_ERR_OOM if allocation fails.
 */
att1_status_t att1_aimu_mmio_create(att1_aimu_mmio **out);

/**
 * att1_aimu_mmio_destroy
 *
 * Free all resources held by the simulator.  The attached device, cmdq, and
 * trace sources are NOT freed; callers manage their lifetimes.
 * Safe to call with NULL.
 */
void att1_aimu_mmio_destroy(att1_aimu_mmio *m);

/* =========================================================================
 * Register access
 * ====================================================================== */

/**
 * att1_aimu_mmio_read32
 *
 * Read a 32-bit register at byte offset @offset.
 *
 * Rules:
 *   - @offset must be a multiple of 4.
 *   - @offset must be < ATT1_AIMU_MMIO_BAR0_SIZE.
 *   - WO registers return 0.
 *   - Reserved/unassigned offsets return ATT1_AIMU_MMIO_RESERVED_RD.
 *
 * @param m       MMIO simulator.  Must not be NULL.
 * @param offset  Byte offset within BAR0.
 * @param out     Destination.  Must not be NULL.
 * @return ATT1_OK on success.
 * @return ATT1_ERR_INVALID_ARG on NULL args, misaligned, or out-of-range offset.
 */
att1_status_t att1_aimu_mmio_read32(att1_aimu_mmio *m,
                                     uint32_t        offset,
                                     uint32_t       *out);

/**
 * att1_aimu_mmio_write32
 *
 * Write a 32-bit value to the register at byte offset @offset.
 *
 * Rules:
 *   - @offset must be a multiple of 4.
 *   - @offset must be < ATT1_AIMU_MMIO_BAR0_SIZE.
 *   - RO registers reject writes (return ATT1_ERR_UNSUPPORTED).
 *   - RW1C registers: written 1-bits clear; 0-bits are ignored.
 *   - WO registers: value is consumed for side-effects only; not stored for read.
 *   - Reserved/unassigned offsets: write is silently discarded (return ATT1_OK).
 *
 * Side-effects for special WO registers (when an appropriate simulator is
 * attached via att1_aimu_mmio_attach_*):
 *
 *   ATT1_MMIO_CQ_DOORBELL:
 *     Increments m->doorbell_write_count.  If m->cmdq is attached, the
 *     new_tail field (bits 15:0) is written to the cmdq tail register cell.
 *
 *   ATT1_MMIO_RESET_CONTROL:
 *     SOFT_RESET_ALL (bit 0): calls att1_aimu_device_reset() on m->device
 *       (if attached) and re-syncs all device/tile RO registers.
 *     RESET_COUNTERS (bit 1): zeroes all counter register cells (0x4000–
 *       0x40B7).  If m->trace is attached, calls att1_aimu_trace_reset().
 *
 *   ATT1_MMIO_TILE_RESET_CONTROL (per tile N):
 *     Calls att1_aimu_device_reset_tile(m->device, N) and re-syncs tile N's
 *     RO registers.
 *
 *   ATT1_MMIO_COUNTER_SNAPSHOT_CONTROL (SNAP_NOW bit):
 *     If m->trace is attached, calls att1_aimu_trace_snapshot_all() and
 *     copies the resulting cmdq counters into the 0x4000 counter registers.
 *     Increments m->snapshot_trigger_count.  The SNAP_NOW bit is self-clearing
 *     (not stored in the backing register).
 *
 * @param m       MMIO simulator.  Must not be NULL.
 * @param offset  Byte offset within BAR0.
 * @param value   Value to write.
 * @return ATT1_OK on success (including discarded writes to reserved offsets).
 * @return ATT1_ERR_INVALID_ARG on NULL m, misaligned, or out-of-range offset.
 * @return ATT1_ERR_UNSUPPORTED if the register is RO.
 */
att1_status_t att1_aimu_mmio_write32(att1_aimu_mmio *m,
                                      uint32_t        offset,
                                      uint32_t        value);

/**
 * att1_aimu_mmio_read64
 *
 * Read a 64-bit value from a LOW/HIGH register pair.
 *
 * @offset must be the LOW register's offset and must be 8-byte aligned.
 * Internally calls read32(offset) for LOW and read32(offset+4) for HIGH,
 * then assembles (HIGH << 32) | LOW.
 *
 * @return ATT1_OK on success.
 * @return ATT1_ERR_INVALID_ARG on NULL args, misaligned (not % 8), or
 *         out-of-range offsets.
 */
att1_status_t att1_aimu_mmio_read64(att1_aimu_mmio *m,
                                     uint32_t        offset,
                                     uint64_t       *out);

/**
 * att1_aimu_mmio_write64
 *
 * Write a 64-bit value to a LOW/HIGH register pair.
 *
 * @offset must be the LOW register's offset and must be 8-byte aligned.
 * Internally calls write32(offset, lo) then write32(offset+4, hi).
 *
 * @return ATT1_OK on success.
 * @return ATT1_ERR_INVALID_ARG on NULL m, misaligned, or out-of-range offsets.
 * @return ATT1_ERR_UNSUPPORTED if either register is RO.
 */
att1_status_t att1_aimu_mmio_write64(att1_aimu_mmio *m,
                                      uint32_t        offset,
                                      uint64_t        value);

/* =========================================================================
 * Simulator attachment
 * ====================================================================== */

/**
 * att1_aimu_mmio_attach_device
 *
 * Attach an M106 device simulator as the source for device-level and per-tile
 * RO register values.  After attaching, call att1_aimu_mmio_sync() to
 * populate the register file from the device's current state.
 *
 * The MMIO simulator does NOT own the device pointer.
 *
 * @return ATT1_ERR_INVALID_ARG if m is NULL.
 */
att1_status_t att1_aimu_mmio_attach_device(att1_aimu_mmio  *m,
                                            att1_aimu_device *dev);

/**
 * att1_aimu_mmio_attach_cmdq
 *
 * Attach an M105 command-queue simulator.  After attaching and syncing,
 * CQ_HEAD, CQ_TAIL, CQ_STATUS, and CQ_FENCE_VALUE are populated from the
 * queue's live state.  CQ_DOORBELL writes update the tail.
 *
 * @return ATT1_ERR_INVALID_ARG if m is NULL.
 */
att1_status_t att1_aimu_mmio_attach_cmdq(att1_aimu_mmio *m,
                                          att1_aimu_cmdq *cq);

/**
 * att1_aimu_mmio_attach_trace
 *
 * Attach an M108 trace/counter source.  After attaching and syncing,
 * counter registers are populated from the most recent trace snapshot.
 * COUNTER_SNAPSHOT_CONTROL SNAP_NOW writes trigger a new snapshot and
 * refresh the counter registers.
 *
 * @return ATT1_ERR_INVALID_ARG if m is NULL.
 */
att1_status_t att1_aimu_mmio_attach_trace(att1_aimu_mmio *m,
                                           att1_aimu_trace *tr);

/* =========================================================================
 * Synchronisation
 * ====================================================================== */

/**
 * att1_aimu_mmio_sync
 *
 * Copy the current state of all attached simulators into the RO register
 * backing cells.  Call this:
 *   - after calling any attach function
 *   - after any external mutation of attached simulator state
 *
 * If no simulator is attached, only the architectural defaults are
 * written (DEVICE_ID, REGISTER_MAP_VERSION, GLOBAL_STATUS.DEVICE_READY).
 *
 * @return ATT1_ERR_INVALID_ARG if m is NULL.
 */
att1_status_t att1_aimu_mmio_sync(att1_aimu_mmio *m);

/* =========================================================================
 * Reset
 * ====================================================================== */

/**
 * att1_aimu_mmio_reset
 *
 * Clear the entire register backing store and re-apply architectural defaults.
 * Does NOT call reset on attached simulators.  Does NOT detach attached sources.
 * After reset, call att1_aimu_mmio_sync() to re-populate from attached sources.
 *
 * @return ATT1_ERR_INVALID_ARG if m is NULL.
 */
att1_status_t att1_aimu_mmio_reset(att1_aimu_mmio *m);

/* =========================================================================
 * Helper / diagnostics
 * ====================================================================== */

/**
 * att1_aimu_mmio_render
 *
 * Print a human-readable register dump to @fp.  Lists all defined registers
 * by name, offset, and current value.  Reserved offsets are omitted.
 *
 * @return ATT1_ERR_INVALID_ARG if m or fp is NULL.
 */
att1_status_t att1_aimu_mmio_render(const att1_aimu_mmio *m, FILE *fp);

/**
 * att1_aimu_mmio_err_name
 *
 * Return a static string describing the att1_status_t code.
 * Returns "ATT1_UNKNOWN" for unrecognised values.  Never returns NULL.
 */
const char *att1_aimu_mmio_err_name(att1_status_t s);

#ifdef __cplusplus
}
#endif

#endif /* ATT1_AIMU_MMIO_H */
