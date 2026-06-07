/*
 * att1_aimu_trace.h  —  AIMU unified trace/counter snapshot (M108)
 *
 * Aggregates control-plane counters from the AIMU simulator components
 * introduced in M105–M107 into a single deterministic snapshot.
 *
 *   M105  att1_aimu_cmdq   — command submission / dispatch counters
 *   M106  att1_aimu_device — device / tile reset and error counters
 *   M107  att1_aimu_dma    — DMA descriptor submission counters
 *
 * Fabric counters are included as zero-value placeholders; a future
 * milestone will populate them from the fabric simulator.
 *
 * This is NOT a real PCIe trace buffer, MMIO accessor, or kernel driver.
 * All state is in-process heap-allocated C structs.
 *
 * No ATT-1 inference, backend, tokenizer, or CUDA behaviour is changed.
 *
 * Register mapping (M104)
 * -----------------------
 *   meta    → §8  TRACE_* registers; §2 DEVICE_ID / TILE_COUNT
 *   cmdq    → §4  CQ_* registers; §7 CNT_COMMANDS_* counters
 *   device  → §2  GLOBAL_STATUS; §3 TILE_STATUS / TILE_RESET_CONTROL
 *   dma     → §5  DMA_STATUS / DMA_CONTROL; §7 DMA counters
 *   fabric  → §6  FABRIC_* registers (placeholder; wired in M109+)
 */

#ifndef ATT1_AIMU_TRACE_H
#define ATT1_AIMU_TRACE_H

#include "att1_status.h"
#include "att1_aimu_cmdq.h"     /* att1_aimu_cmdq */
#include "att1_aimu_device.h"   /* att1_aimu_device */
#include "att1_aimu_dma.h"      /* att1_aimu_dma */

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

/* -------------------------------------------------------------------------
 * Constants
 * ---------------------------------------------------------------------- */

/** Magic written into every att1_aimu_trace to detect use-after-free. */
#define ATT1_AIMU_TRACE_MAGIC    UINT32_C(0xA771FACE)

/** Snapshot format version produced by this module (v1.0). */
#define ATT1_AIMU_TRACE_VERSION  UINT32_C(0x00010000)

/* -------------------------------------------------------------------------
 * Snapshot status codes   (meta.status)
 * ---------------------------------------------------------------------- */

/** All registered sources were present; snapshot is complete. */
#define ATT1_AIMU_TRACE_STATUS_OK       UINT32_C(0)

/**
 * One or more sources were NULL in att1_aimu_trace_snapshot_all(); the
 * corresponding snapshot sub-struct retains its previous value.
 */
#define ATT1_AIMU_TRACE_STATUS_PARTIAL  UINT32_C(1)

/** No snapshot has been taken yet (initial state after create/reset). */
#define ATT1_AIMU_TRACE_STATUS_EMPTY    UINT32_C(2)

/* -------------------------------------------------------------------------
 * Snapshot sub-structs
 * ---------------------------------------------------------------------- */

/** Command-queue counters captured from att1_aimu_cmdq (M105). */
typedef struct att1_aimu_trace_cmdq_counters {
    uint64_t commands_submitted;   /**< M104 §7 CNT_COMMANDS_SUBMITTED  */
    uint64_t commands_completed;   /**< M104 §7 CNT_COMMANDS_COMPLETED  */
    uint64_t commands_failed;      /**< M104 §7 (derived)               */
    uint64_t queue_full_count;     /**< M104 §4 CQ_STATUS queue-full    */
    uint64_t unsupported_commands; /**< M104 §9 ERR_UNSUPPORTED_OP      */
    uint64_t fence_value;          /**< M104 §4 CQ_FENCE_VALUE          */
} att1_aimu_trace_cmdq_counters;

/** Device / tile counters captured from att1_aimu_device (M106). */
typedef struct att1_aimu_trace_device_counters {
    uint64_t device_resets;  /**< dev->reset_count; M104 §2 RESET_CONTROL */
    uint64_t tile_resets;    /**< sum of tiles[i].reset_count; M104 §3 TILE_RESET_CONTROL */
    uint64_t tile_errors;    /**< count of tiles with state == TILE_ERROR; M104 §3 TILE_STATUS */
} att1_aimu_trace_device_counters;

/** DMA counters captured from att1_aimu_dma (M107). */
typedef struct att1_aimu_trace_dma_counters {
    uint64_t dma_submitted;            /**< M104 §5 DMA_STATUS            */
    uint64_t dma_completed;            /**< M104 §5 (derived)             */
    uint64_t dma_failed;               /**< M104 §5 DMA_ERROR_STATUS      */
    uint64_t bytes_host_to_device;     /**< M104 §7 (derived H2D)         */
    uint64_t bytes_device_to_host;     /**< M104 §7 (derived D2H)         */
    uint64_t bytes_device_to_device;   /**< M104 §7 (derived D2D)         */
    uint64_t alignment_failures;       /**< M104 §9 ERR_ALIGNMENT         */
    uint64_t range_failures;           /**< M104 §9 ERR_DMA_FAULT         */
    uint64_t unsupported_flags;        /**< M104 §9 ERR_INVALID_COMMAND   */
} att1_aimu_trace_dma_counters;

/**
 * Fabric / interconnect placeholder counters.
 *
 * Maps to M104 §6 FABRIC_* registers.  All fields are zero in M108; a
 * future milestone will wire these to the fabric simulator.
 */
typedef struct att1_aimu_trace_snapshot_fabric {
    uint64_t packets_sent;             /**< M104 §6 FABRIC_PACKET_COUNTER_LO/HI   */
    uint64_t packets_received;         /**< M104 §6 (derived)                      */
    uint64_t payload_bytes_sent;       /**< M104 §6 FABRIC_PAYLOAD_BYTES_LO/HI    */
    uint64_t payload_bytes_received;   /**< M104 §6 (derived)                      */
    uint64_t congestion_events;        /**< M104 §6 FABRIC_CONGESTION_COUNTER      */
} att1_aimu_trace_fabric_counters;

/** Trace metadata fields. */
typedef struct att1_aimu_trace_meta {
    uint32_t trace_version;   /**< ATT1_AIMU_TRACE_VERSION; M104 §8 TRACE_CONTROL */
    uint32_t snapshot_id;     /**< monotonic; incremented by snapshot_all          */
    uint32_t device_id;       /**< identifier of the source device (0 = default)   */
    uint32_t tile_count;      /**< captured from att1_aimu_device; else 0          */
    uint64_t event_count;     /**< placeholder; M104 §8 TRACE_WRITE_PTR            */
    uint64_t dropped_events;  /**< placeholder; M104 §8 TRACE_DROPPED_EVENTS       */
    uint32_t status;          /**< ATT1_AIMU_TRACE_STATUS_* value                  */
    uint32_t _pad;            /**< reserved                                        */
} att1_aimu_trace_meta;

/* -------------------------------------------------------------------------
 * Unified snapshot struct
 * ---------------------------------------------------------------------- */

/**
 * att1_aimu_trace_snapshot — a point-in-time capture of all AIMU
 * control-plane counters from the M105–M107 simulators plus fabric
 * placeholders.
 */
typedef struct att1_aimu_trace_snapshot {
    att1_aimu_trace_meta             meta;
    att1_aimu_trace_cmdq_counters    cmdq;
    att1_aimu_trace_device_counters  device;
    att1_aimu_trace_dma_counters     dma;
    att1_aimu_trace_fabric_counters  fabric;
} att1_aimu_trace_snapshot;

/* -------------------------------------------------------------------------
 * Trace object
 * ---------------------------------------------------------------------- */

typedef struct att1_aimu_trace {
    uint32_t                  magic;
    uint32_t                  _pad;
    att1_aimu_trace_snapshot  snapshot;
} att1_aimu_trace;

/* -------------------------------------------------------------------------
 * Lifecycle
 * ---------------------------------------------------------------------- */

/**
 * att1_aimu_trace_create
 *
 * Allocate and initialise an AIMU trace object.  The initial snapshot has
 * all counters at zero and status ATT1_AIMU_TRACE_STATUS_EMPTY.
 *
 * Returns ATT1_OK and writes *out on success.
 * Returns ATT1_ERR_INVALID_ARG if out is NULL.
 * Returns ATT1_ERR_OOM if allocation fails.
 */
att1_status_t att1_aimu_trace_create(att1_aimu_trace **out);

/**
 * att1_aimu_trace_destroy
 *
 * Free all resources.  Poisons the magic field.  NULL is safe.
 */
void att1_aimu_trace_destroy(att1_aimu_trace *t);

/* -------------------------------------------------------------------------
 * Individual counter snapshots
 *
 * Each function copies the relevant counters from the source simulator into
 * the trace's internal snapshot sub-struct.  It does NOT increment
 * snapshot_id and does NOT mutate any counters in the source simulator.
 * ---------------------------------------------------------------------- */

/**
 * att1_aimu_trace_snapshot_cmdq
 *
 * Copy att1_aimu_cmdq counters into the trace snapshot's cmdq sub-struct.
 *
 * Returns ATT1_ERR_INVALID_ARG if t or q is NULL.
 */
att1_status_t att1_aimu_trace_snapshot_cmdq(att1_aimu_trace      *t,
                                              const att1_aimu_cmdq *q);

/**
 * att1_aimu_trace_snapshot_device
 *
 * Derive device/tile counters from dev and write them into the trace
 * snapshot's device sub-struct:
 *   device_resets = dev->reset_count
 *   tile_resets   = sum of dev->tiles[i].reset_count for i in [0, tile_count)
 *   tile_errors   = count of tiles with state == ATT1_AIMU_TILE_ERROR
 *
 * Also updates meta.tile_count from dev->tile_count.
 *
 * Returns ATT1_ERR_INVALID_ARG if t or dev is NULL.
 */
att1_status_t att1_aimu_trace_snapshot_device(att1_aimu_trace        *t,
                                               const att1_aimu_device *dev);

/**
 * att1_aimu_trace_snapshot_dma
 *
 * Copy att1_aimu_dma counters into the trace snapshot's dma sub-struct.
 *
 * Returns ATT1_ERR_INVALID_ARG if t or sim is NULL.
 */
att1_status_t att1_aimu_trace_snapshot_dma(att1_aimu_trace     *t,
                                            const att1_aimu_dma *sim);

/* -------------------------------------------------------------------------
 * Combined snapshot
 * ---------------------------------------------------------------------- */

/**
 * att1_aimu_trace_snapshot_all
 *
 * Snapshot all provided sources in one call:
 *   - If q   is non-NULL: update snapshot.cmdq
 *   - If dev is non-NULL: update snapshot.device and meta.tile_count
 *   - If dma is non-NULL: update snapshot.dma
 *
 * Fabric counters remain at zero (placeholder for M109+).
 * Always increments meta.snapshot_id.
 * Sets meta.status to ATT1_AIMU_TRACE_STATUS_OK if all three sources are
 * non-NULL, ATT1_AIMU_TRACE_STATUS_PARTIAL otherwise.
 *
 * Returns ATT1_ERR_INVALID_ARG if t is NULL.
 * NULL q/dev/dma are not errors; those sub-structs are simply skipped.
 */
att1_status_t att1_aimu_trace_snapshot_all(att1_aimu_trace        *t,
                                            const att1_aimu_cmdq   *q,
                                            const att1_aimu_device *dev,
                                            const att1_aimu_dma    *dma);

/* -------------------------------------------------------------------------
 * Accessors
 * ---------------------------------------------------------------------- */

/**
 * att1_aimu_trace_get_snapshot
 *
 * Copy the current trace snapshot into *out.
 *
 * Returns ATT1_ERR_INVALID_ARG if t or out is NULL.
 */
att1_status_t att1_aimu_trace_get_snapshot(const att1_aimu_trace    *t,
                                            att1_aimu_trace_snapshot *out);

/* -------------------------------------------------------------------------
 * Reset
 * ---------------------------------------------------------------------- */

/**
 * att1_aimu_trace_reset
 *
 * Clear the trace-local snapshot (all counters to zero, status to EMPTY,
 * snapshot_id to 0).  Does NOT modify any attached source simulators.
 *
 * Returns ATT1_ERR_INVALID_ARG if t is NULL.
 */
att1_status_t att1_aimu_trace_reset(att1_aimu_trace *t);

/* -------------------------------------------------------------------------
 * Rendering
 * ---------------------------------------------------------------------- */

/**
 * att1_aimu_trace_render
 *
 * Write a human-readable summary of snap to f.  Output is plain text with
 * one labelled line per counter group.  Includes all four sections (cmdq,
 * device, dma, fabric) and the metadata header.
 *
 * Does NOT mutate snap.
 *
 * Returns ATT1_ERR_INVALID_ARG if snap or f is NULL.
 * Returns ATT1_OK on success.
 */
att1_status_t att1_aimu_trace_render(const att1_aimu_trace_snapshot *snap,
                                      FILE                           *f);

/* -------------------------------------------------------------------------
 * Name helpers
 * ---------------------------------------------------------------------- */

/**
 * att1_aimu_trace_status_name
 *
 * Return a stable ASCII name for a trace status value.
 *   ATT1_AIMU_TRACE_STATUS_OK      → "OK"
 *   ATT1_AIMU_TRACE_STATUS_PARTIAL → "PARTIAL"
 *   ATT1_AIMU_TRACE_STATUS_EMPTY   → "EMPTY"
 * Returns "UNKNOWN" for unrecognised values.
 */
const char *att1_aimu_trace_status_name(uint32_t status);

#ifdef __cplusplus
}
#endif

#endif /* ATT1_AIMU_TRACE_H */
