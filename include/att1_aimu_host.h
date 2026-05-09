/*
 * att1_aimu_host.h  —  AIMU control-plane integration harness (M112)
 *
 * Connects the M105 command-queue simulator, M106 device discovery
 * simulator, M107 DMA descriptor simulator, M108 trace/counter snapshot,
 * and M111 MMIO/register-file simulator into one deterministic
 * host-to-AIMU control-plane flow.
 *
 * This is NOT a real PCIe driver or MMIO accessor.  Every interaction is
 * an in-process C11 function call.  The harness models the full sequence:
 *
 *   1.  Host probes device  (MMIO read DEVICE_ID / REGISTER_MAP_VERSION /
 *       TILE_COUNT; device_query_info; per-tile query_tile)
 *   2.  Host sets up command queue (create cmdq; attach to MMIO)
 *   3.  Host validates a DMA descriptor
 *   4.  Host submits LOAD_TENSOR_TILE command
 *   5.  Host rings CQ doorbell through MMIO
 *   6.  Simulated AIMU consumes and dispatches commands
 *   7.  Host reads completion records (FIFO order)
 *   8.  Host snapshots trace/counters
 *
 * No tensor bytes move; no inference executes; no CUDA kernels run.
 *
 * Lifetime rules
 * --------------
 * att1_aimu_host owns its att1_aimu_device, att1_aimu_cmdq,
 * att1_aimu_dma, att1_aimu_trace, and att1_aimu_mmio.  All are created
 * by att1_aimu_host_create and destroyed by att1_aimu_host_destroy.
 * Callers MUST NOT free the sub-objects independently.
 */

#ifndef ATT1_AIMU_HOST_H
#define ATT1_AIMU_HOST_H

#include "att1_status.h"
#include "att1_aimu_cmdq.h"
#include "att1_aimu_device.h"
#include "att1_aimu_dma.h"
#include "att1_aimu_trace.h"
#include "att1_aimu_mmio.h"

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

/* =========================================================================
 * Constants
 * ====================================================================== */

/** Magic cookie placed in att1_aimu_host to detect use-after-free. */
#define ATT1_AIMU_HOST_MAGIC    UINT32_C(0xA771A057)

/* =========================================================================
 * Configuration
 * ====================================================================== */

/**
 * att1_aimu_host_config
 *
 * Parameters forwarded to the sub-simulators created inside
 * att1_aimu_host_create.  Pass NULL to att1_aimu_host_create for defaults.
 */
typedef struct att1_aimu_host_config {
    /** Number of simulated AIMU tiles (1–ATT1_AIMU_DEVICE_MAX_TILES; 0→4). */
    size_t   tile_count;

    /** Per-tile model-weight memory capacity in bytes (0→1 GiB). */
    uint64_t tile_memory_bytes;

    /** Per-tile KV-cache memory capacity in bytes (0→256 MiB). */
    uint64_t tile_kv_bytes;

    /** ATT1_AIMU_DTYPE_* bitmask (0→ATT1_AIMU_DTYPE_ALL). */
    uint32_t supported_dtypes;

    /** ATT1_AIMU_OP_* bitmask (0→ATT1_AIMU_OP_ALL). */
    uint32_t supported_ops;

    /** Command ring depth (0→ATT1_AIMU_CMDQ_DEFAULT_DEPTH). */
    size_t   cmd_ring_depth;
} att1_aimu_host_config;

/* =========================================================================
 * Probe result  (returned by att1_aimu_host_probe_device)
 * ====================================================================== */

typedef struct att1_aimu_host_probe_result {
    uint32_t device_id;             /**< MMIO DEVICE_ID register            */
    uint32_t register_map_version;  /**< MMIO REGISTER_MAP_VERSION register */
    uint32_t global_status;         /**< MMIO GLOBAL_STATUS register        */
    size_t   tile_count;            /**< MMIO TILE_COUNT register           */
} att1_aimu_host_probe_result;

/* =========================================================================
 * Tile enumeration result  (one per tile)
 * ====================================================================== */

typedef struct att1_aimu_host_tile_info {
    uint8_t  tile_id;
    uint64_t memory_capacity_bytes;
    uint64_t kv_capacity_bytes;
    uint32_t supported_dtypes;
    uint32_t supported_ops;
    uint8_t  state;     /**< att1_aimu_tile_state value  */
} att1_aimu_host_tile_info;

/* =========================================================================
 * Summary  (returned by att1_aimu_host_get_summary)
 * ====================================================================== */

typedef struct att1_aimu_host_summary {
    /* Device identity */
    uint32_t device_id;
    uint32_t register_map_version;
    size_t   tile_count;

    /* Command-queue counters */
    uint64_t commands_submitted;
    uint64_t commands_completed;
    uint64_t commands_failed;

    /* DMA counters */
    uint64_t dma_submitted;
    uint64_t dma_completed;
    uint64_t dma_failed;

    /* MMIO interaction counters */
    uint32_t doorbell_count;        /**< times CQ_DOORBELL was written     */
    uint32_t snapshot_trigger_count;/**< times SNAP_NOW was triggered      */

    /* Fence tracking */
    uint64_t fence_value;           /**< last completed fence value        */

    /* Trace */
    uint64_t trace_event_count;     /**< meta.event_count from last snap   */
    uint32_t trace_snapshot_id;     /**< meta.snapshot_id from last snap   */
    uint32_t trace_status;          /**< ATT1_AIMU_TRACE_STATUS_*          */

    /**
     * Overall harness status:
     *   ATT1_AIMU_OK       — all operations succeeded
     *   ATT1_AIMU_PENDING  — not all commands have completed
     *   Any error          — first fatal error encountered
     */
    att1_aimu_result status;
} att1_aimu_host_summary;

/* =========================================================================
 * Host context object
 * ====================================================================== */

typedef struct att1_aimu_host {
    uint32_t             magic;

    /* Owned sub-simulators */
    att1_aimu_device    *device;
    att1_aimu_cmdq      *cmdq;
    att1_aimu_dma       *dma;
    att1_aimu_trace     *trace;
    att1_aimu_mmio      *mmio;

    /* Cached probe results (set by att1_aimu_host_probe_device) */
    att1_aimu_host_probe_result probe;

    /** Last fatal error, if any. */
    att1_aimu_result     last_error;

    /** TRUE if att1_aimu_host_probe_device has been called successfully. */
    int                  probed;

    /** TRUE if att1_aimu_host_setup_cmdq has been called successfully. */
    int                  cmdq_ready;
} att1_aimu_host;

/* =========================================================================
 * Lifecycle
 * ====================================================================== */

/**
 * att1_aimu_host_create
 *
 * Allocate and initialise the integration harness together with all
 * sub-simulators.  Pass NULL for config to accept defaults.
 *
 * Returns ATT1_OK and writes *out on success.
 * Returns ATT1_ERR_INVALID_ARG if out is NULL.
 * Returns ATT1_ERR_OOM if any sub-allocation fails.
 */
att1_status_t att1_aimu_host_create(const att1_aimu_host_config *config,
                                     att1_aimu_host             **out);

/**
 * att1_aimu_host_destroy
 *
 * Free all owned sub-simulators and the host context itself.
 * Safe to call with NULL.
 */
void att1_aimu_host_destroy(att1_aimu_host *h);

/* =========================================================================
 * Control-plane flow
 * ====================================================================== */

/**
 * att1_aimu_host_probe_device
 *
 * Step 1–4 of the integration flow:
 *   - Read DEVICE_ID, REGISTER_MAP_VERSION, GLOBAL_STATUS, TILE_COUNT from
 *     MMIO.
 *   - Synchronise the MMIO register file from attached simulators.
 *   - Store results in h->probe.
 *   - Return ATT1_ERR_STATE if GLOBAL_STATUS.DEVICE_READY is not set.
 *
 * Returns ATT1_ERR_INVALID_ARG if h is NULL.
 */
att1_status_t att1_aimu_host_probe_device(att1_aimu_host             *h,
                                           att1_aimu_host_probe_result *out);

/**
 * att1_aimu_host_enumerate_tiles
 *
 * Read per-tile registers for all tiles discovered during probe.
 * Writes up to *count entries into infos[]; updates *count with the actual
 * number written (== h->probe.tile_count clamped to the input *count).
 *
 * Returns ATT1_ERR_STATE if probe has not been called yet.
 * Returns ATT1_ERR_INVALID_ARG if h or infos or count is NULL.
 */
att1_status_t att1_aimu_host_enumerate_tiles(
        att1_aimu_host           *h,
        att1_aimu_host_tile_info *infos,
        size_t                   *count);

/**
 * att1_aimu_host_setup_cmdq
 *
 * Step 5 of the integration flow:
 *   - Attach the owned command queue to MMIO.
 *   - Attach the owned device to the command queue.
 *   - Synchronise the MMIO register file.
 *   - Set h->cmdq_ready = 1.
 *
 * Returns ATT1_ERR_STATE if probe has not been called yet.
 * Returns ATT1_ERR_INVALID_ARG if h is NULL.
 */
att1_status_t att1_aimu_host_setup_cmdq(att1_aimu_host *h);

/**
 * att1_aimu_host_validate_dma
 *
 * Step 6 of the integration flow:
 *   - Validate a DMA descriptor against the owned DMA simulator.
 *   - Does NOT submit (use att1_aimu_host_submit_cmd for the associated
 *     LOAD_TENSOR_TILE command).
 *
 * Returns ATT1_ERR_INVALID_ARG if h or desc is NULL.
 * Forwards the att1_aimu_dma_validate return code on failure.
 */
att1_status_t att1_aimu_host_validate_dma(att1_aimu_host           *h,
                                           const att1_aimu_dma_desc *desc);

/**
 * att1_aimu_host_submit_cmd
 *
 * Steps 7 of the integration flow:
 *   - Submit a command to the owned command queue.
 *   - Returns ATT1_ERR_STATE if cmdq is not ready.
 *   - Returns ATT1_ERR_INVALID_ARG if h or cmd is NULL.
 *   - Forwards att1_aimu_cmdq_submit errors.
 */
att1_status_t att1_aimu_host_submit_cmd(att1_aimu_host *h,
                                         att1_aimu_cmd  *cmd);

/**
 * att1_aimu_host_ring_doorbell
 *
 * Step 8 of the integration flow:
 *   - Write ATT1_MMIO_CQ_DOORBELL via MMIO with the current queue tail.
 *   - Returns ATT1_ERR_STATE if cmdq is not ready.
 *   - Returns ATT1_ERR_INVALID_ARG if h is NULL.
 */
att1_status_t att1_aimu_host_ring_doorbell(att1_aimu_host *h);

/**
 * att1_aimu_host_process_one
 *
 * Step 9 of the integration flow (simulated AIMU side):
 *   - Dispatch one command from the queue.
 *   - Returns ATT1_ERR_QUEUE_EMPTY if no commands are pending.
 *   - Returns ATT1_ERR_INVALID_ARG if h is NULL.
 */
att1_status_t att1_aimu_host_process_one(att1_aimu_host *h);

/**
 * att1_aimu_host_drain
 *
 * Dispatch all pending commands.
 * Equivalent to att1_aimu_cmdq_dispatch_all on the owned queue.
 *
 * Returns ATT1_ERR_INVALID_ARG if h is NULL.
 */
att1_status_t att1_aimu_host_drain(att1_aimu_host *h);

/**
 * att1_aimu_host_read_completion
 *
 * Read and consume the next completion from the owned queue.
 *
 * Returns ATT1_ERR_QUEUE_EMPTY if no completions are available.
 * Returns ATT1_ERR_INVALID_ARG if h or out is NULL.
 */
att1_status_t att1_aimu_host_read_completion(att1_aimu_host       *h,
                                              att1_aimu_completion *out);

/**
 * att1_aimu_host_snapshot_counters
 *
 * Step 12 of the integration flow:
 *   - Trigger att1_aimu_trace_snapshot_all on the owned trace object.
 *   - Synchronise MMIO register file from all sources.
 *
 * Returns ATT1_ERR_INVALID_ARG if h is NULL.
 */
att1_status_t att1_aimu_host_snapshot_counters(att1_aimu_host *h);

/**
 * att1_aimu_host_get_summary
 *
 * Step 13 of the integration flow:
 *   - Aggregate counters from MMIO, command queue, DMA, and trace into
 *     a single flat summary struct.
 *
 * Returns ATT1_ERR_INVALID_ARG if h or out is NULL.
 */
att1_status_t att1_aimu_host_get_summary(att1_aimu_host         *h,
                                          att1_aimu_host_summary *out);

/**
 * att1_aimu_host_reset
 *
 * Reset all sub-simulators and clear the MMIO register file.
 * Clears h->probed and h->cmdq_ready.
 *
 * Returns ATT1_ERR_INVALID_ARG if h is NULL.
 */
att1_status_t att1_aimu_host_reset(att1_aimu_host *h);

/**
 * att1_aimu_host_render
 *
 * Write a human-readable summary of the harness state to fp.
 * Calls att1_aimu_mmio_render on the owned MMIO object.
 *
 * Returns ATT1_ERR_INVALID_ARG if h or fp is NULL.
 */
att1_status_t att1_aimu_host_render(const att1_aimu_host *h, FILE *fp);

#ifdef __cplusplus
}
#endif

#endif /* ATT1_AIMU_HOST_H */
