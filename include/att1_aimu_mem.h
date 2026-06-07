/*
 * att1_aimu_mem.h  —  AIMU tile memory allocator simulator (M124)
 *
 * Models AIMU-local tile memory allocation and accounting.  Each tile has a
 * configurable simulated address space (capacity_bytes); the allocator returns
 * simulated address ranges, tracks region types and flags, and reports
 * fragmentation estimates — without allocating large backing buffers.
 *
 * Memory guardrail
 * ----------------
 * capacity_bytes is metadata only.  A tile configured with 64 GiB of capacity
 * does NOT cause a 64 GiB malloc.  Allocation records are heap-allocated C
 * structs whose size is O(allocation_count), independent of capacity.
 *
 * Address space
 * -------------
 * Simulated addresses start at ATT1_AIMU_MEM_BASE (0x0000_0000_0000_0000) and
 * run to capacity_bytes-1.  All addresses are 64-bit to support multi-GiB tile
 * capacities without overflow.
 *
 * Alignment policy
 * ----------------
 * Minimum alignment is ATT1_AIMU_MEM_MIN_ALIGN (16 bytes).  Callers may
 * request larger power-of-two alignments up to
 * ATT1_AIMU_MEM_MAX_ALIGN (64 KiB).  Non-power-of-two alignments are
 * rejected with ATT1_ERR_INVALID_ARG.
 *
 * Region types
 * ------------
 * See att1_aimu_mem_region_type.  Unknown region type values are rejected.
 *
 * Allocation flags
 * ----------------
 * See ATT1_AIMU_MEM_FLAG_* constants.  Unknown flag bits are rejected.
 *
 * Relation to other modules
 * -------------------------
 * M125/M126 will integrate DMA range validation so that M107 DMA descriptor
 * device_base addresses can be cross-checked against the tile allocator.
 * This module is standalone; it does not change M107 behavior.
 *
 * This is NOT a real memory allocator, PCIe DMA engine, or kernel driver.
 * No inference, CUDA, tokenizer, or .att1 binary format behavior is changed.
 */

#ifndef ATT1_AIMU_MEM_H
#define ATT1_AIMU_MEM_H

#include "att1_status.h"

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

/* =========================================================================
 * Constants
 * ====================================================================== */

/** Magic written into every att1_aimu_mem to detect use-after-free. */
#define ATT1_AIMU_MEM_MAGIC             UINT32_C(0xA771A124)

/** Base simulated address for the tile address space. */
#define ATT1_AIMU_MEM_BASE              UINT64_C(0x0000000000000000)

/** Minimum allocation alignment in bytes (must be power of two). */
#define ATT1_AIMU_MEM_MIN_ALIGN         UINT64_C(16)

/** Maximum allocation alignment in bytes (must be power of two). */
#define ATT1_AIMU_MEM_MAX_ALIGN         UINT64_C(65536)   /* 64 KiB */

/** Maximum number of live allocations per tile allocator. */
#define ATT1_AIMU_MEM_MAX_ALLOCS        4096u

/** Maximum name/tag length including NUL terminator. */
#define ATT1_AIMU_MEM_NAME_MAX          64u

/** Sentinel value returned as allocation id on failure. */
#define ATT1_AIMU_MEM_INVALID_ID        UINT32_MAX

/* =========================================================================
 * Region types
 * ====================================================================== */

typedef enum att1_aimu_mem_region_type {
    ATT1_AIMU_MEM_REGION_TENSOR          = 0,  /**< model weight / activation   */
    ATT1_AIMU_MEM_REGION_KV_CACHE        = 1,  /**< KV-cache pages              */
    ATT1_AIMU_MEM_REGION_STAGING         = 2,  /**< DMA staging area            */
    ATT1_AIMU_MEM_REGION_DMA_BUFFER      = 3,  /**< DMA descriptor ring         */
    ATT1_AIMU_MEM_REGION_COMMAND_QUEUE   = 4,  /**< M105 command ring           */
    ATT1_AIMU_MEM_REGION_COMPLETION_QUEUE= 5,  /**< completion ring             */
    ATT1_AIMU_MEM_REGION_TRACE_BUFFER    = 6,  /**< M108 trace ring             */
    ATT1_AIMU_MEM_REGION_FABRIC_BUFFER   = 7,  /**< fabric send/receive buffer  */
    ATT1_AIMU_MEM_REGION_RESERVED        = 8,  /**< reserved/firmware area      */
    ATT1_AIMU_MEM_REGION_COUNT           = 9   /**< sentinel — not a valid type */
} att1_aimu_mem_region_type;

/* =========================================================================
 * Allocation flags
 * ====================================================================== */

/** No special flags. */
#define ATT1_AIMU_MEM_FLAG_NONE         UINT32_C(0x00000000)

/** Region is read-only (tile may not write). */
#define ATT1_AIMU_MEM_FLAG_READ_ONLY    UINT32_C(0x00000001)

/** Region is write-only (tile may not read). */
#define ATT1_AIMU_MEM_FLAG_WRITE_ONLY   UINT32_C(0x00000002)

/** Region is pinned and must not be moved by a future compaction pass. */
#define ATT1_AIMU_MEM_FLAG_PINNED       UINT32_C(0x00000004)

/** Region is a DMA target (address must meet DMA alignment requirements). */
#define ATT1_AIMU_MEM_FLAG_DMA_TARGET   UINT32_C(0x00000008)

/** All known flag bits — unknown bits are rejected. */
#define ATT1_AIMU_MEM_FLAG_ALL_KNOWN    UINT32_C(0x0000000F)

/* =========================================================================
 * dtype constants (mirrors M107 DMA dtype encoding)
 * ====================================================================== */

#define ATT1_AIMU_MEM_DTYPE_NONE  UINT8_C(0xFF) /**< no dtype / non-tensor     */
#define ATT1_AIMU_MEM_DTYPE_F32   UINT8_C(0)    /**< 32-bit IEEE-754 float     */
#define ATT1_AIMU_MEM_DTYPE_Q8    UINT8_C(1)    /**< 8-bit symmetric quantised */
#define ATT1_AIMU_MEM_DTYPE_Q4    UINT8_C(2)    /**< 4-bit packed quantised    */

/* =========================================================================
 * Allocation record
 * ====================================================================== */

/** A single simulated allocation within a tile's address space. */
typedef struct att1_aimu_mem_alloc {
    uint32_t  alloc_id;     /**< unique id within this allocator (1-based)  */
    uint32_t  tile_id;      /**< owning tile                                */

    att1_aimu_mem_region_type region_type; /**< semantic region type         */
    uint8_t   dtype;        /**< data type (ATT1_AIMU_MEM_DTYPE_*)          */
    uint32_t  flags;        /**< ATT1_AIMU_MEM_FLAG_* bits                  */

    uint64_t  base_address; /**< simulated byte offset from ATT1_AIMU_MEM_BASE */
    uint64_t  byte_size;    /**< allocation size in bytes                   */
    uint64_t  alignment;    /**< alignment used for this allocation         */

    char      name[ATT1_AIMU_MEM_NAME_MAX]; /**< optional debug tag         */

    int       live;         /**< non-zero if allocation is active           */
} att1_aimu_mem_alloc;

/* =========================================================================
 * Fragmentation summary
 * ====================================================================== */

typedef struct att1_aimu_mem_frag {
    uint64_t capacity_bytes;     /**< total tile capacity                   */
    uint64_t used_bytes;         /**< sum of live allocation byte_size      */
    uint64_t free_bytes;         /**< capacity_bytes - used_bytes           */
    uint64_t largest_free_block; /**< largest contiguous free span          */
    uint32_t allocation_count;   /**< number of live allocations            */
    uint32_t fragmentation_pct;  /**< 0-100 estimate; 0 = no fragmentation  */
} att1_aimu_mem_frag;

/* =========================================================================
 * Allocator
 * ====================================================================== */

/**
 * att1_aimu_mem — simulated per-tile memory allocator.
 *
 * The allocator owns an array of att1_aimu_mem_alloc records and manages a
 * simple free-list using the existing allocation array (first-fit linear
 * scan).  No backing buffer of capacity_bytes is ever allocated.
 */
typedef struct att1_aimu_mem {
    uint32_t  magic;         /**< ATT1_AIMU_MEM_MAGIC                       */
    uint32_t  tile_id;       /**< owning tile                               */
    uint64_t  capacity_bytes;/**< total simulated address-space size        */
    uint32_t  next_id;       /**< next allocation id to issue               */

    att1_aimu_mem_alloc allocs[ATT1_AIMU_MEM_MAX_ALLOCS]; /**< allocation table */
    uint32_t  alloc_count;   /**< total slots used (live + freed)           */
} att1_aimu_mem;

/* =========================================================================
 * Lifecycle
 * ====================================================================== */

/**
 * att1_aimu_mem_create — initialise a tile memory allocator.
 *
 * @param tile_id        owning tile index.
 * @param capacity_bytes simulated address-space size; metadata only,
 *                       no buffer of this size is allocated.
 * @param out            receives the allocated att1_aimu_mem on success.
 * @return ATT1_OK, ATT1_ERR_INVALID_ARG, or ATT1_ERR_OOM.
 */
att1_status_t att1_aimu_mem_create(uint32_t tile_id,
                                   uint64_t capacity_bytes,
                                   att1_aimu_mem **out);

/**
 * att1_aimu_mem_destroy — free the allocator and all internal records.
 *
 * @param m  allocator to destroy; NULL is silently ignored.
 */
void att1_aimu_mem_destroy(att1_aimu_mem *m);

/* =========================================================================
 * Allocation
 * ====================================================================== */

/**
 * att1_aimu_mem_alloc_range — allocate a region within the tile's address
 * space.
 *
 * Uses first-fit placement.  Returns a simulated base_address in *addr_out
 * and the new allocation id in *id_out (both optional).
 *
 * @param m           allocator.
 * @param region_type semantic type (must be < ATT1_AIMU_MEM_REGION_COUNT).
 * @param byte_size   must be > 0.
 * @param alignment   power of two in [ATT1_AIMU_MEM_MIN_ALIGN,
 *                    ATT1_AIMU_MEM_MAX_ALIGN]; 0 → use MIN_ALIGN.
 * @param dtype       ATT1_AIMU_MEM_DTYPE_* or ATT1_AIMU_MEM_DTYPE_NONE.
 * @param flags       ATT1_AIMU_MEM_FLAG_* bits; unknown bits → error.
 * @param name        optional debug tag (NULL → empty string).
 * @param addr_out    receives simulated base address; may be NULL.
 * @param id_out      receives allocation id; may be NULL.
 * @return ATT1_OK, ATT1_ERR_INVALID_ARG, ATT1_ERR_OOM (capacity full),
 *         ATT1_ERR_STATE (alloc table full).
 */
att1_status_t att1_aimu_mem_alloc_range(att1_aimu_mem             *m,
                                        att1_aimu_mem_region_type  region_type,
                                        uint64_t                   byte_size,
                                        uint64_t                   alignment,
                                        uint8_t                    dtype,
                                        uint32_t                   flags,
                                        const char                *name,
                                        uint64_t                  *addr_out,
                                        uint32_t                  *id_out);

/**
 * att1_aimu_mem_free — release an allocation by id.
 *
 * @return ATT1_OK, ATT1_ERR_NOT_FOUND (unknown id), ATT1_ERR_STATE
 *         (already freed / double free).
 */
att1_status_t att1_aimu_mem_free(att1_aimu_mem *m, uint32_t alloc_id);

/* =========================================================================
 * Queries
 * ====================================================================== */

/**
 * att1_aimu_mem_query_by_id — look up an allocation record by id.
 *
 * @param out receives a const pointer into the internal table on success.
 * @return ATT1_OK or ATT1_ERR_NOT_FOUND.
 */
att1_status_t att1_aimu_mem_query_by_id(const att1_aimu_mem       *m,
                                        uint32_t                   alloc_id,
                                        const att1_aimu_mem_alloc **out);

/**
 * att1_aimu_mem_query_by_address — find the live allocation that contains
 * the given simulated address.
 *
 * @return ATT1_OK or ATT1_ERR_NOT_FOUND.
 */
att1_status_t att1_aimu_mem_query_by_address(const att1_aimu_mem       *m,
                                             uint64_t                   addr,
                                             const att1_aimu_mem_alloc **out);

/**
 * att1_aimu_mem_range_valid — return non-zero if [addr, addr+size) is
 * entirely within a single live allocation, zero otherwise.
 *
 * Useful for validating DMA descriptor device_base / byte_length pairs
 * against the tile allocator (integration deferred to M125/M126).
 */
int att1_aimu_mem_range_valid(const att1_aimu_mem *m,
                              uint64_t             addr,
                              uint64_t             size);

/* =========================================================================
 * Accounting
 * ====================================================================== */

/**
 * att1_aimu_mem_get_frag — compute used/free bytes and a fragmentation
 * estimate.
 *
 * Fragmentation percent is estimated as:
 *   (free_bytes - largest_free_block) * 100 / free_bytes
 * clamped to [0, 100].  Returns 0 when free_bytes == 0.
 *
 * @return ATT1_OK or ATT1_ERR_INVALID_ARG.
 */
att1_status_t att1_aimu_mem_get_frag(const att1_aimu_mem *m,
                                     att1_aimu_mem_frag  *out);

/* =========================================================================
 * Allocator reset
 * ====================================================================== */

/**
 * att1_aimu_mem_reset — free all allocations, reset the id counter.
 *
 * @return ATT1_OK or ATT1_ERR_INVALID_ARG.
 */
att1_status_t att1_aimu_mem_reset(att1_aimu_mem *m);

/* =========================================================================
 * Debug render
 * ====================================================================== */

/**
 * att1_aimu_mem_render — print a human-readable summary to fp.
 *
 * @return ATT1_OK or ATT1_ERR_INVALID_ARG.
 */
att1_status_t att1_aimu_mem_render(const att1_aimu_mem *m, FILE *fp);

#ifdef __cplusplus
}
#endif

#endif /* ATT1_AIMU_MEM_H */
