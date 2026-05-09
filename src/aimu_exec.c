/*
 * aimu_exec.c  —  Simulated AIMU EXEC command replay (Milestone 130)
 *
 * Implements deterministic, control-plane-only simulation of EXEC_*,
 * KV_*, FABRIC_*, and housekeeping command types.  No tensor buffers
 * are read or written; no inference math executes; no PCIe/MMIO access
 * occurs; no CUDA kernels run.
 */

#define _POSIX_C_SOURCE 200112L

#include "att1_aimu_exec.h"

#include <stdlib.h>
#include <string.h>

/* =========================================================================
 * Internal helpers
 * ====================================================================== */

/**
 * Map a cmd->dtype byte (0=f32, 1=q8, 2=q4) to an ATT1_AIMU_DTYPE_* bit.
 * Returns 0 for unknown values (disables dtype check).
 */
static uint32_t dtype_to_bit(uint8_t dtype_byte)
{
    if (dtype_byte <= 2u) {
        return UINT32_C(1) << dtype_byte;   /* F32=bit0, Q8=bit1, Q4=bit2 */
    }
    return 0u;
}

/* =========================================================================
 * Lifecycle
 * ====================================================================== */

att1_status_t
att1_aimu_exec_ctx_create(const att1_aimu_device *device,
                           att1_aimu_exec_ctx    **out)
{
    if (!out) {
        return ATT1_ERR_INVALID_ARG;
    }

    att1_aimu_exec_ctx *ctx = calloc(1, sizeof(*ctx));
    if (!ctx) {
        return ATT1_ERR_OOM;
    }

    ctx->magic  = ATT1_AIMU_EXEC_CTX_MAGIC;
    ctx->device = device;
    /* counters are zero-initialised by calloc */

    *out = ctx;
    return ATT1_OK;
}

void
att1_aimu_exec_ctx_destroy(att1_aimu_exec_ctx *ctx)
{
    if (!ctx) {
        return;
    }
    ctx->magic = 0u;
    free(ctx);
}

att1_status_t
att1_aimu_exec_ctx_reset_counters(att1_aimu_exec_ctx *ctx)
{
    if (!ctx || ctx->magic != ATT1_AIMU_EXEC_CTX_MAGIC) {
        return ATT1_ERR_INVALID_ARG;
    }
    memset(&ctx->counters, 0, sizeof(ctx->counters));
    return ATT1_OK;
}

att1_status_t
att1_aimu_exec_ctx_get_counters(const att1_aimu_exec_ctx *ctx,
                                 att1_aimu_exec_counters  *out)
{
    if (!ctx || ctx->magic != ATT1_AIMU_EXEC_CTX_MAGIC || !out) {
        return ATT1_ERR_INVALID_ARG;
    }
    *out = ctx->counters;
    return ATT1_OK;
}

/* =========================================================================
 * Dispatch
 * ====================================================================== */

att1_aimu_result
att1_aimu_exec_dispatch(att1_aimu_exec_ctx  *ctx,
                         const att1_aimu_cmd *cmd)
{
    /* ------------------------------------------------------------------ */
    /* Guard: invalid context or command                                   */
    /* ------------------------------------------------------------------ */
    if (!ctx || ctx->magic != ATT1_AIMU_EXEC_CTX_MAGIC || !cmd) {
        return ATT1_AIMU_ERR_INVALID_COMMAND;
    }

    att1_aimu_exec_counters *cnt = &ctx->counters;
    cnt->exec_commands_seen++;

    /* ------------------------------------------------------------------ */
    /* Tile range check                                                     */
    /* ------------------------------------------------------------------ */
    if (ctx->device &&
        (size_t)cmd->tile_id >= ctx->device->tile_count) {
        cnt->exec_commands_failed++;
        return ATT1_AIMU_ERR_INVALID_COMMAND;
    }

    /* ------------------------------------------------------------------ */
    /* Resolve tile capabilities (default: all ops/dtypes when no device)  */
    /* ------------------------------------------------------------------ */
    uint32_t tile_ops   = ATT1_AIMU_OP_ALL;
    uint32_t tile_dtype = ATT1_AIMU_DTYPE_ALL;

    if (ctx->device &&
        (size_t)cmd->tile_id < ctx->device->tile_count) {
        tile_ops   = ctx->device->tiles[cmd->tile_id].supported_ops;
        tile_dtype = ctx->device->tiles[cmd->tile_id].supported_dtypes;
    }

    uint32_t cmd_dtype_bit = dtype_to_bit(cmd->dtype);

    att1_aimu_cmd_type ct = (att1_aimu_cmd_type)cmd->command_type;

    /* ------------------------------------------------------------------ */
    /* EXEC_* helper: dtype check then op check then success               */
    /* ------------------------------------------------------------------ */
#define EXEC_OP_CHECK(op_bit, counter_field) \
    do { \
        if (cmd_dtype_bit && !(tile_dtype & cmd_dtype_bit)) { \
            cnt->exec_commands_failed++; \
            return ATT1_AIMU_ERR_UNSUPPORTED_DTYPE; \
        } \
        if (!((tile_ops) & (op_bit))) { \
            cnt->exec_unsupported++; \
            cnt->exec_commands_failed++; \
            return ATT1_AIMU_ERR_UNSUPPORTED_OP; \
        } \
        cnt->counter_field++; \
        cnt->exec_commands_completed++; \
        cnt->bytes_read_estimate    += cmd->input_buf_bytes; \
        cnt->bytes_written_estimate += cmd->output_buf_bytes; \
        return ATT1_AIMU_OK; \
    } while (0)

    switch (ct) {

    /* ---- Tensor management: pass through as OK, no EXEC counter ---- */

    case ATT1_AIMU_CMD_LOAD_TENSOR_TILE:
        cnt->exec_commands_completed++;
        cnt->bytes_written_estimate += cmd->output_buf_bytes;
        return ATT1_AIMU_OK;

    case ATT1_AIMU_CMD_VALIDATE_TENSOR:
        cnt->exec_commands_completed++;
        return ATT1_AIMU_OK;

    /* ---- EXEC_MATMUL: requires non-zero tensor_id ---- */

    case ATT1_AIMU_CMD_EXEC_MATMUL:
        if (cmd->tensor_id == 0u) {
            cnt->exec_commands_failed++;
            return ATT1_AIMU_ERR_INVALID_TENSOR;
        }
        EXEC_OP_CHECK(ATT1_AIMU_OP_MATMUL, matmul_count);

    /* ---- Other EXEC_* ops ---- */

    case ATT1_AIMU_CMD_EXEC_RMSNORM:
        EXEC_OP_CHECK(ATT1_AIMU_OP_RMSNORM, rmsnorm_count);

    case ATT1_AIMU_CMD_EXEC_ROPE:
        EXEC_OP_CHECK(ATT1_AIMU_OP_ROPE, rope_count);

    case ATT1_AIMU_CMD_EXEC_ATTENTION:
        EXEC_OP_CHECK(ATT1_AIMU_OP_ATTENTION, attention_count);

    case ATT1_AIMU_CMD_EXEC_FFN:
        EXEC_OP_CHECK(ATT1_AIMU_OP_FFN, ffn_count);

    /* ---- KV cache ---- */

    case ATT1_AIMU_CMD_KV_APPEND:
        if (!(tile_ops & ATT1_AIMU_OP_KV_APPEND)) {
            cnt->exec_unsupported++;
            cnt->exec_commands_failed++;
            return ATT1_AIMU_ERR_UNSUPPORTED_OP;
        }
        cnt->kv_append_count++;
        cnt->exec_commands_completed++;
        cnt->bytes_written_estimate += cmd->output_buf_bytes;
        return ATT1_AIMU_OK;

    case ATT1_AIMU_CMD_KV_READ:
        if (!(tile_ops & ATT1_AIMU_OP_KV_READ)) {
            cnt->exec_unsupported++;
            cnt->exec_commands_failed++;
            return ATT1_AIMU_ERR_UNSUPPORTED_OP;
        }
        cnt->kv_read_count++;
        cnt->exec_commands_completed++;
        cnt->bytes_read_estimate += cmd->input_buf_bytes;
        return ATT1_AIMU_OK;

    /* ---- Fabric ---- */

    case ATT1_AIMU_CMD_FABRIC_SEND:
        if (!(tile_ops & ATT1_AIMU_OP_FABRIC_SEND)) {
            cnt->exec_unsupported++;
            cnt->exec_commands_failed++;
            return ATT1_AIMU_ERR_UNSUPPORTED_OP;
        }
        cnt->fabric_send_count++;
        cnt->exec_commands_completed++;
        return ATT1_AIMU_OK;

    case ATT1_AIMU_CMD_FABRIC_REDUCE:
        if (!(tile_ops & ATT1_AIMU_OP_FABRIC_REDUCE)) {
            cnt->exec_unsupported++;
            cnt->exec_commands_failed++;
            return ATT1_AIMU_ERR_UNSUPPORTED_OP;
        }
        cnt->fabric_reduce_count++;
        cnt->exec_commands_completed++;
        return ATT1_AIMU_OK;

    /* ---- Housekeeping (always succeed) ---- */

    case ATT1_AIMU_CMD_TILE_BARRIER:
        cnt->barrier_count++;
        cnt->exec_commands_completed++;
        return ATT1_AIMU_OK;

    case ATT1_AIMU_CMD_TRACE_SNAPSHOT:
        cnt->trace_snapshot_count++;
        cnt->exec_commands_completed++;
        return ATT1_AIMU_OK;

    case ATT1_AIMU_CMD_QUERY_COUNTERS:
        cnt->exec_commands_completed++;
        return ATT1_AIMU_OK;

    case ATT1_AIMU_CMD_NOP:
        cnt->exec_commands_completed++;
        return ATT1_AIMU_OK;

    case ATT1_AIMU_CMD_RESET_TILE:
        cnt->exec_commands_completed++;
        return ATT1_AIMU_OK;

    /* ---- Unknown command type ---- */

    default:
        cnt->exec_commands_failed++;
        return ATT1_AIMU_ERR_INVALID_COMMAND;
    }

#undef EXEC_OP_CHECK
}
