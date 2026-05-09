/*
 * att1_aimu_exec.h  —  Simulated AIMU EXEC command replay (Milestone 130)
 *
 * Provides a deterministic, control-plane-only simulator for the EXEC_*,
 * KV_*, FABRIC_*, and housekeeping AIMU command types.  All dispatch is
 * performed in-process without tensor buffers, inference math, PCIe/MMIO
 * access, CUDA kernels, or kernel drivers.
 *
 * Typical usage:
 *
 *   att1_aimu_device     *dev = ...;   // capability reference
 *   att1_aimu_exec_ctx   *ctx = NULL;
 *   att1_aimu_exec_ctx_create(dev, &ctx);
 *
 *   att1_aimu_cmd cmd = { ... };       // command built from M129 plan
 *   att1_aimu_result r = att1_aimu_exec_dispatch(ctx, &cmd);
 *
 *   att1_aimu_exec_counters cntrs;
 *   att1_aimu_exec_ctx_get_counters(ctx, &cntrs);
 *
 *   att1_aimu_exec_ctx_destroy(ctx);   // does NOT destroy dev
 *
 * The att1_aimu_exec_ctx does NOT own the att1_aimu_device; the device must
 * outlive the context.
 *
 * No C, Makefile, binary format, backend, tokenizer, or inference behaviour
 * is altered by this module.
 */

#ifndef ATT1_AIMU_EXEC_H
#define ATT1_AIMU_EXEC_H

#include "att1_status.h"
#include "att1_aimu_cmdq.h"
#include "att1_aimu_device.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* =========================================================================
 * Magic cookie
 * ====================================================================== */

#define ATT1_AIMU_EXEC_CTX_MAGIC  UINT32_C(0xA771E5EC)

/* =========================================================================
 * Counters
 * ====================================================================== */

/**
 * att1_aimu_exec_counters
 *
 * Aggregate per-replay-session counters updated by att1_aimu_exec_dispatch.
 * All fields are monotonically non-decreasing within a session; reset with
 * att1_aimu_exec_ctx_reset_counters.
 */
typedef struct att1_aimu_exec_counters {
    /** Total commands seen (successful or failed). */
    uint64_t    exec_commands_seen;
    /** Commands that completed with ATT1_AIMU_OK. */
    uint64_t    exec_commands_completed;
    /** Commands that completed with any error code. */
    uint64_t    exec_commands_failed;
    /** Commands that completed with ATT1_AIMU_ERR_UNSUPPORTED_OP. */
    uint64_t    exec_unsupported;

    /* Per-op completion counts (incremented only on success) */
    uint64_t    matmul_count;
    uint64_t    rmsnorm_count;
    uint64_t    rope_count;
    uint64_t    attention_count;
    uint64_t    ffn_count;
    uint64_t    kv_append_count;
    uint64_t    kv_read_count;
    uint64_t    fabric_send_count;
    uint64_t    fabric_reduce_count;
    uint64_t    barrier_count;
    uint64_t    trace_snapshot_count;

    /** Estimated tensor bytes read (sum of input_buf_bytes for OK EXEC_*). */
    uint64_t    bytes_read_estimate;
    /** Estimated bytes written (sum of output_buf_bytes for OK EXEC_* plus
     *  output_buf_bytes for OK LOAD_TENSOR_TILE). */
    uint64_t    bytes_written_estimate;
} att1_aimu_exec_counters;

/* =========================================================================
 * Context object
 * ====================================================================== */

/**
 * att1_aimu_exec_ctx
 *
 * Replay context that holds a (non-owning) reference to the device
 * capability model and the running counter state.
 */
typedef struct att1_aimu_exec_ctx {
    uint32_t                    magic;
    /** Non-owning reference to the device (may be NULL; all ops allowed). */
    const att1_aimu_device     *device;
    att1_aimu_exec_counters     counters;
} att1_aimu_exec_ctx;

/* =========================================================================
 * Lifecycle
 * ====================================================================== */

/**
 * att1_aimu_exec_ctx_create
 *
 * Allocate and initialise a simulated-exec context.  The device pointer is
 * stored as a reference; pass NULL to disable capability checking (all ops
 * and dtypes treated as supported).
 *
 * Returns ATT1_OK and writes *out on success.
 * Returns ATT1_ERR_INVALID_ARG if out is NULL.
 * Returns ATT1_ERR_OOM if allocation fails.
 */
att1_status_t att1_aimu_exec_ctx_create(const att1_aimu_device *device,
                                         att1_aimu_exec_ctx    **out);

/**
 * att1_aimu_exec_ctx_destroy
 *
 * Free the context.  Does NOT destroy the referenced device.
 * Safe to call with NULL.
 */
void att1_aimu_exec_ctx_destroy(att1_aimu_exec_ctx *ctx);

/**
 * att1_aimu_exec_ctx_reset_counters
 *
 * Zero all counters without destroying the context.
 *
 * Returns ATT1_ERR_INVALID_ARG if ctx is NULL or has an invalid magic.
 */
att1_status_t att1_aimu_exec_ctx_reset_counters(att1_aimu_exec_ctx *ctx);

/**
 * att1_aimu_exec_ctx_get_counters
 *
 * Copy the current counter snapshot into *out.
 *
 * Returns ATT1_ERR_INVALID_ARG if ctx or out is NULL or magic is invalid.
 */
att1_status_t att1_aimu_exec_ctx_get_counters(const att1_aimu_exec_ctx *ctx,
                                               att1_aimu_exec_counters  *out);

/* =========================================================================
 * Dispatch
 * ====================================================================== */

/**
 * att1_aimu_exec_dispatch
 *
 * Simulate execution of one M103 command descriptor against the device
 * capability model.  No tensor buffers are read/written; no inference
 * executes; no PCIe/MMIO access occurs.
 *
 * Validation order for EXEC_* commands:
 *   1. If ctx is NULL or magic invalid     → ATT1_AIMU_ERR_INVALID_COMMAND
 *   2. If cmd is NULL                      → ATT1_AIMU_ERR_INVALID_COMMAND
 *   3. If tile_id >= device->tile_count    → ATT1_AIMU_ERR_INVALID_COMMAND
 *   4. For EXEC_MATMUL: tensor_id == 0    → ATT1_AIMU_ERR_INVALID_TENSOR
 *   5. If dtype not in supported_dtypes    → ATT1_AIMU_ERR_UNSUPPORTED_DTYPE
 *   6. If op not in supported_ops          → ATT1_AIMU_ERR_UNSUPPORTED_OP
 *   7. Otherwise                           → ATT1_AIMU_OK
 *
 * Counter updates:
 *   exec_commands_seen        — always incremented.
 *   exec_commands_completed   — incremented on ATT1_AIMU_OK.
 *   exec_commands_failed      — incremented on any error.
 *   exec_unsupported          — incremented on ATT1_AIMU_ERR_UNSUPPORTED_OP.
 *   Per-op counter            — incremented on ATT1_AIMU_OK only.
 *   bytes_read_estimate       — += input_buf_bytes  on ATT1_AIMU_OK EXEC_*.
 *   bytes_written_estimate    — += output_buf_bytes on ATT1_AIMU_OK EXEC_*.
 *
 * Returns ATT1_AIMU_ERR_INVALID_COMMAND if ctx or cmd is NULL / magic bad.
 */
att1_aimu_result att1_aimu_exec_dispatch(att1_aimu_exec_ctx  *ctx,
                                          const att1_aimu_cmd *cmd);

#ifdef __cplusplus
}
#endif

#endif /* ATT1_AIMU_EXEC_H */
