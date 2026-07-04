/*
 * att1_aimu_dma.h  —  AIMU DMA descriptor simulator (M107)
 *
 * In-process simulator for the DMA buffer descriptor model defined in M104
 * (docs/aimu_register_map.md §5).  It validates 64-byte transfer descriptors,
 * tracks per-direction byte counters, and enforces address alignment and range
 * constraints — without accessing real host physical memory or MMIO registers.
 *
 * This is NOT a Linux kernel DMA driver, a PCIe DMA engine, or an MMIO
 * accessor.  All state lives in heap-allocated C structs.
 *
 * Relationship to other M10x modules
 * ------------------------------------
 *   M105  att1_aimu_cmdq  — command ring buffer; LOAD_TENSOR_TILE commands
 *                           carry an embedded descriptor_id that refers to a
 *                           validated att1_aimu_dma_desc.
 *   M106  att1_aimu_device — device/tile capability discovery; tile
 *                            memory_capacity_bytes bounds device-side range
 *                            checks performed here.
 *   M107  att1_aimu_dma   — THIS MODULE.  Validates descriptors and tracks
 *                            transfer counters; does not copy real memory.
 *
 * Freeze status: FROZEN v1.0 (Milestone 159).
 * The in-process DMA contract is now frozen for: the 64-byte
 * att1_aimu_dma_desc layout, att1_aimu_dma_direction values, dtype constants,
 * ATT1_AIMU_DMA_FLAG_* bits plus ATT1_AIMU_DMA_FLAG_VALID_MASK, the
 * validation rules documented below, and the att1_aimu_dma_counters field
 * names.  See docs/aimu_register_map.md §15.7 and
 * docs/schema_compatibility.md §12.
 */

#ifndef ATT1_AIMU_DMA_H
#define ATT1_AIMU_DMA_H

#include "att1_status.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* =========================================================================
 * Compile-time constants
 * ====================================================================== */

/** Magic written into every att1_aimu_dma to detect use-after-free. */
#define ATT1_AIMU_DMA_MAGIC              UINT32_C(0xA771D4A7)

/**
 * Required address alignment for all DMA transfers, in bytes.
 * Mirrors M104 §5.1: "Each DMA descriptor is 64 bytes, naturally aligned."
 */
#define ATT1_AIMU_DMA_ALIGN_BYTES        UINT32_C(64)

/**
 * Maximum payload size for a single DMA transfer (2^28 = 256 MiB).
 * Mirrors M104 §5.1 field `byte_length` constraint.
 */
#define ATT1_AIMU_DMA_MAX_TRANSFER_BYTES UINT32_C(0x10000000)

/** Maximum number of host memory regions that can be registered. */
#define ATT1_AIMU_DMA_MAX_HOST_REGIONS   8u

/** Maximum number of device memory regions that can be registered. */
#define ATT1_AIMU_DMA_MAX_DEVICE_REGIONS 8u

/* =========================================================================
 * Descriptor direction
 * ====================================================================== */

typedef enum att1_aimu_dma_direction {
    ATT1_AIMU_DMA_HOST_TO_DEVICE    = 0, /**< host buffer → AIMU tile         */
    ATT1_AIMU_DMA_DEVICE_TO_HOST    = 1, /**< AIMU tile → host buffer         */
    ATT1_AIMU_DMA_DEVICE_TO_DEVICE  = 2  /**< AIMU tile → AIMU tile (D2D)     */
} att1_aimu_dma_direction;

/* =========================================================================
 * Descriptor dtype  (matches M105 att1_aimu_cmd.dtype encoding)
 * ====================================================================== */

#define ATT1_AIMU_DMA_DTYPE_F32   UINT8_C(0)  /**< 32-bit IEEE-754 float        */
#define ATT1_AIMU_DMA_DTYPE_Q8    UINT8_C(1)  /**< 8-bit symmetric quantised    */
#define ATT1_AIMU_DMA_DTYPE_Q4    UINT8_C(2)  /**< 4-bit packed quantised       */

/* =========================================================================
 * Descriptor flag bits  (subset of M104 §5.2; direction bits are in the
 * `direction` field rather than in `flags` to make D2D unambiguous)
 * ====================================================================== */

/** AIMU verifies the `checksum` field on receipt (placeholder in M107). */
#define ATT1_AIMU_DMA_FLAG_VALIDATE_CHECKSUM  UINT16_C(0x0001)

/** AIMU computes and writes checksum for device-to-host transfers (placeholder). */
#define ATT1_AIMU_DMA_FLAG_GENERATE_CHECKSUM  UINT16_C(0x0002)

/** Final descriptor in a chain; triggers completion notification. */
#define ATT1_AIMU_DMA_FLAG_LAST_DESCRIPTOR    UINT16_C(0x0004)

/** Descriptor is part of a scatter-gather chain (unsupported in M107). */
#define ATT1_AIMU_DMA_FLAG_SCATTER_GATHER     UINT16_C(0x0008)

/** Mask of all valid flag bits.  Any other bit set → validation failure. */
#define ATT1_AIMU_DMA_FLAG_VALID_MASK         UINT16_C(0x000F)

/* =========================================================================
 * DMA descriptor  — exactly 64 bytes
 *
 * This is the in-process simulator representation.  It extends the M104 §5.1
 * 64-byte on-wire format with separate src_device_addr / dst_device_addr
 * fields to support device-to-device transfers.  For H2D or D2H transfers,
 * host_addr and device_addr are used; the D2D fields are ignored.
 * ====================================================================== */

typedef struct att1_aimu_dma_desc {
    /* --- 8-byte fields (offset 0–31) ---------------------------------- */
    uint64_t host_addr;        /**< host buffer address      (H2D / D2H)     */
    uint64_t device_addr;      /**< AIMU-local address        (H2D / D2H)    */
    uint64_t src_device_addr;  /**< AIMU-local source address (D2D only)     */
    uint64_t dst_device_addr;  /**< AIMU-local dest address   (D2D only)     */
    /* --- 4-byte fields (offset 32–51) --------------------------------- */
    uint32_t byte_length;      /**< transfer size in bytes (1 – 2^28)        */
    uint32_t descriptor_id;    /**< host-assigned descriptor identifier       */
    uint32_t command_id;       /**< owning M105 command ID                    */
    uint32_t tensor_id;        /**< target tensor slot on the AIMU tile       */
    uint32_t checksum;         /**< CRC32 placeholder (0 = not validated)     */
    /* --- 2-byte fields (offset 52–57) --------------------------------- */
    uint16_t dim0;             /**< logical row dimension                     */
    uint16_t dim1;             /**< logical column dimension                  */
    uint16_t flags;            /**< ATT1_AIMU_DMA_FLAG_* bits                 */
    /* --- 1-byte fields (offset 58–63) --------------------------------- */
    uint8_t  dtype;            /**< ATT1_AIMU_DMA_DTYPE_* value               */
    uint8_t  quant_group_size; /**< Q4: 32 or 64; Q8/F32: must be 0          */
    uint8_t  direction;        /**< att1_aimu_dma_direction value              */
    uint8_t  _pad[3];          /**< reserved; must be zero                    */
} att1_aimu_dma_desc;

/* Verify the descriptor is exactly 64 bytes. */
typedef char att1_aimu_dma_desc_size_check[
    (sizeof(att1_aimu_dma_desc) == 64u) ? 1 : -1];

/* =========================================================================
 * Per-simulator counters
 * ====================================================================== */

typedef struct att1_aimu_dma_counters {
    uint64_t dma_submitted;          /**< total descriptors submitted          */
    uint64_t dma_completed;          /**< descriptors that passed validation   */
    uint64_t dma_failed;             /**< descriptors that failed validation   */
    uint64_t bytes_host_to_device;   /**< payload bytes across H2D transfers  */
    uint64_t bytes_device_to_host;   /**< payload bytes across D2H transfers  */
    uint64_t bytes_device_to_device; /**< payload bytes across D2D transfers  */
    uint64_t alignment_failures;     /**< rejected: bad address alignment      */
    uint64_t range_failures;         /**< rejected: out-of-range / overflow    */
    uint64_t unsupported_flags;      /**< rejected: unknown flag bits          */
} att1_aimu_dma_counters;

/* =========================================================================
 * Registered memory region
 * ====================================================================== */

typedef struct att1_aimu_dma_region {
    uint64_t base; /**< region start address (64-byte aligned)                */
    uint64_t size; /**< region size in bytes                                   */
} att1_aimu_dma_region;

/* =========================================================================
 * Simulator object
 * ====================================================================== */

typedef struct att1_aimu_dma {
    uint32_t               magic;

    /* Registered host memory regions used for range checking. */
    att1_aimu_dma_region   host_regions[ATT1_AIMU_DMA_MAX_HOST_REGIONS];
    size_t                 host_region_count;

    /* Registered device memory regions used for range checking. */
    att1_aimu_dma_region   device_regions[ATT1_AIMU_DMA_MAX_DEVICE_REGIONS];
    size_t                 device_region_count;

    att1_aimu_dma_counters counters;

    uint32_t               next_descriptor_id; /**< auto-assigned ID seed     */
} att1_aimu_dma;

/* =========================================================================
 * Lifecycle
 * ====================================================================== */

/**
 * att1_aimu_dma_create
 *
 * Allocate and initialise a DMA descriptor simulator.
 *
 * Returns ATT1_OK and sets *out on success.
 * Returns ATT1_ERR_INVALID_ARG if out is NULL.
 * Returns ATT1_ERR_OOM if allocation fails.
 */
att1_status_t att1_aimu_dma_create(att1_aimu_dma **out);

/**
 * att1_aimu_dma_destroy
 *
 * Free the simulator.  Poisons the magic field.  NULL is safe.
 */
void att1_aimu_dma_destroy(att1_aimu_dma *sim);

/* =========================================================================
 * Region registration
 * ====================================================================== */

/**
 * att1_aimu_dma_register_host_region
 *
 * Register a host memory region [base, base+size) that is reachable by the
 * DMA engine.  H2D and D2H descriptors whose host_addr falls outside all
 * registered regions are rejected.  If no host regions are registered, the
 * host range check is skipped (permissive mode).
 *
 * base must be 64-byte aligned; size must be > 0.
 *
 * Returns ATT1_ERR_STATE if ATT1_AIMU_DMA_MAX_HOST_REGIONS is exceeded.
 */
att1_status_t att1_aimu_dma_register_host_region(att1_aimu_dma *sim,
                                                   uint64_t       base,
                                                   uint64_t       size);

/**
 * att1_aimu_dma_register_device_region
 *
 * Register an AIMU device memory region.  Validated the same way as host
 * regions, against device_addr (H2D/D2H) or src/dst_device_addr (D2D).
 */
att1_status_t att1_aimu_dma_register_device_region(att1_aimu_dma *sim,
                                                     uint64_t       base,
                                                     uint64_t       size);

/* =========================================================================
 * Descriptor operations
 * ====================================================================== */

/**
 * att1_aimu_dma_validate
 *
 * Validate a DMA descriptor against the registered regions and alignment
 * rules.  Does NOT update any counters.
 *
 * Checks (in order):
 *   1. direction must be a valid att1_aimu_dma_direction value
 *   2. byte_length must be in [1, ATT1_AIMU_DMA_MAX_TRANSFER_BYTES]
 *   3. flags must contain no bits outside ATT1_AIMU_DMA_FLAG_VALID_MASK
 *   4. dtype must be F32, Q8, or Q4
 *   5. if Q4: quant_group_size must be 32 or 64; byte_length must be a
 *      multiple of (quant_group_size / 2)
 *   6. addresses must be ATT1_AIMU_DMA_ALIGN_BYTES-aligned
 *   7. address + byte_length must not overflow uint64_t
 *   8. if host/device regions are registered: transfer range must fit
 *   9. D2D: src and dst ranges must not overlap
 *
 * Returns ATT1_OK if valid.
 * Returns ATT1_ERR_INVALID_ARG if sim or desc is NULL, or any check fails.
 */
att1_status_t att1_aimu_dma_validate(const att1_aimu_dma      *sim,
                                      const att1_aimu_dma_desc *desc);

/**
 * att1_aimu_dma_submit
 *
 * Validate a descriptor and simulate its submission.  Always increments
 * dma_submitted.  On success, increments dma_completed and the appropriate
 * bytes_* counter.  On failure, increments dma_failed and the specific
 * failure counter (alignment_failures, range_failures, or unsupported_flags).
 *
 * Returns ATT1_OK on success.
 * Returns ATT1_ERR_INVALID_ARG if sim or desc is NULL, or validation fails.
 */
att1_status_t att1_aimu_dma_submit(att1_aimu_dma            *sim,
                                    const att1_aimu_dma_desc *desc);

/* =========================================================================
 * Counters
 * ====================================================================== */

/**
 * att1_aimu_dma_get_counters
 *
 * Copy current counters into *out.
 * Returns ATT1_ERR_INVALID_ARG if sim or out is NULL.
 */
att1_status_t att1_aimu_dma_get_counters(const att1_aimu_dma    *sim,
                                          att1_aimu_dma_counters *out);

/**
 * att1_aimu_dma_reset_counters
 *
 * Zero all counters.
 * Returns ATT1_ERR_INVALID_ARG if sim is NULL.
 */
att1_status_t att1_aimu_dma_reset_counters(att1_aimu_dma *sim);

#ifdef __cplusplus
}
#endif

#endif /* ATT1_AIMU_DMA_H */
