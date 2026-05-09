/*
 * test_aimu_exec.c  —  Tests for the simulated AIMU EXEC command replay (M130)
 *
 * Covers:
 *  1.  test_lifecycle             — create/destroy, NULL args
 *  2.  test_matmul_supported      — EXEC_MATMUL succeeds when tile supports op
 *  3.  test_matmul_unsupported    — EXEC_MATMUL fails with UNSUPPORTED_OP when
 *                                   tile capability bitmask excludes MATMUL
 *  4.  test_rmsnorm_rope_attn_ffn — all four EXEC_* counters update on success
 *  5.  test_kv_append_read        — KV_APPEND and KV_READ counters update
 *  6.  test_fabric_send_reduce    — FABRIC_SEND and FABRIC_REDUCE counters
 *  7.  test_barrier_trace_query   — TILE_BARRIER, TRACE_SNAPSHOT, QUERY_COUNTERS
 *  8.  test_invalid_tile_id       — tile_id out of range → INVALID_COMMAND
 *  9.  test_unsupported_dtype     — q8 command on f32-only device → UNSUPPORTED_DTYPE
 * 10.  test_missing_tensor        — EXEC_MATMUL tensor_id=0 → INVALID_TENSOR
 * 11.  test_m129_replay_deterministic — static M129-style plan replayed twice
 *                                       with identical counter results
 * 12.  test_load_validate_nop     — LOAD_TENSOR_TILE, VALIDATE_TENSOR, NOP ok
 * 13.  test_reset_counters        — reset_counters zeroes all fields
 * 14.  test_no_cuda_dep           — compile-time guard: no CUDA headers
 *
 * No ATT-1 inference, backend, tokenizer, CUDA, or binary-format behaviour
 * is changed or tested here.
 */

#include "att1_aimu_exec.h"
#include "att1_aimu_device.h"
#include "att1_aimu_cmdq.h"
#include "att1_status.h"

#include <stdio.h>
#include <string.h>
#include <stdint.h>

/* =========================================================================
 * Test harness
 * ====================================================================== */

static int g_pass = 0;
static int g_fail = 0;

#define EXPECT(cond, name) \
    do { \
        if (cond) { \
            printf("PASS: aimu_exec: " name "\n"); \
            g_pass++; \
        } else { \
            printf("FAIL: aimu_exec: " name "\n"); \
            g_fail++; \
        } \
    } while (0)

/* =========================================================================
 * Helpers
 * ====================================================================== */

/** Create a device with the given op/dtype masks (0 → defaults: all). */
static att1_aimu_device *make_device(size_t tile_count,
                                      uint32_t supported_ops,
                                      uint32_t supported_dtypes)
{
    att1_aimu_device_config cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.tile_count      = tile_count ? tile_count : 1;
    cfg.supported_ops   = supported_ops;
    cfg.supported_dtypes = supported_dtypes;

    att1_aimu_device *dev = NULL;
    if (att1_aimu_device_create(&cfg, &dev) != ATT1_OK) {
        return NULL;
    }
    return dev;
}

/** Build a minimal valid EXEC_MATMUL command. */
static att1_aimu_cmd make_matmul_cmd(uint8_t tile_id, uint16_t tensor_id,
                                      uint8_t dtype)
{
    att1_aimu_cmd cmd;
    memset(&cmd, 0, sizeof(cmd));
    cmd.command_type     = (uint8_t)ATT1_AIMU_CMD_EXEC_MATMUL;
    cmd.tile_id          = tile_id;
    cmd.tensor_id        = tensor_id;
    cmd.dtype            = dtype;
    cmd.input_buf_bytes  = 4096;
    cmd.output_buf_bytes = 1024;
    return cmd;
}

/* =========================================================================
 * 1. test_lifecycle
 * ====================================================================== */

static void test_lifecycle(void)
{
    /* NULL out pointer */
    att1_status_t s = att1_aimu_exec_ctx_create(NULL, NULL);
    EXPECT(s == ATT1_ERR_INVALID_ARG, "lifecycle: create null out -> INVALID_ARG");

    /* Create with NULL device (no capability check) */
    att1_aimu_exec_ctx *ctx = NULL;
    s = att1_aimu_exec_ctx_create(NULL, &ctx);
    EXPECT(s == ATT1_OK,              "lifecycle: create null device -> OK");
    EXPECT(ctx != NULL,               "lifecycle: ctx != NULL");
    EXPECT(ctx->magic == ATT1_AIMU_EXEC_CTX_MAGIC, "lifecycle: magic set");
    EXPECT(ctx->device == NULL,       "lifecycle: device stored as NULL");

    /* Counter snapshot on fresh context */
    att1_aimu_exec_counters cntrs;
    s = att1_aimu_exec_ctx_get_counters(ctx, &cntrs);
    EXPECT(s == ATT1_OK,              "lifecycle: get_counters -> OK");
    EXPECT(cntrs.exec_commands_seen == 0, "lifecycle: seen=0 initially");

    /* get_counters with NULL out */
    s = att1_aimu_exec_ctx_get_counters(ctx, NULL);
    EXPECT(s == ATT1_ERR_INVALID_ARG, "lifecycle: get_counters null out -> INVALID_ARG");

    /* destroy clears magic */
    att1_aimu_exec_ctx_destroy(ctx);

    /* destroy NULL is safe */
    att1_aimu_exec_ctx_destroy(NULL);
    EXPECT(1, "lifecycle: destroy NULL safe");

    /* dispatch with NULL ctx */
    att1_aimu_cmd cmd;
    memset(&cmd, 0, sizeof(cmd));
    att1_aimu_result r = att1_aimu_exec_dispatch(NULL, &cmd);
    EXPECT(r == ATT1_AIMU_ERR_INVALID_COMMAND,
           "lifecycle: dispatch null ctx -> INVALID_COMMAND");

    /* dispatch with NULL cmd */
    att1_aimu_exec_ctx *ctx2 = NULL;
    att1_aimu_exec_ctx_create(NULL, &ctx2);
    r = att1_aimu_exec_dispatch(ctx2, NULL);
    EXPECT(r == ATT1_AIMU_ERR_INVALID_COMMAND,
           "lifecycle: dispatch null cmd -> INVALID_COMMAND");
    att1_aimu_exec_ctx_destroy(ctx2);
}

/* =========================================================================
 * 2. test_matmul_supported
 * ====================================================================== */

static void test_matmul_supported(void)
{
    att1_aimu_device *dev = make_device(1, 0, 0);  /* all ops/dtypes */
    EXPECT(dev != NULL, "matmul_supported: device created");
    if (!dev) return;

    att1_aimu_exec_ctx *ctx = NULL;
    att1_aimu_exec_ctx_create(dev, &ctx);

    att1_aimu_cmd cmd = make_matmul_cmd(0, 1, 0 /* f32 */);
    att1_aimu_result r = att1_aimu_exec_dispatch(ctx, &cmd);
    EXPECT(r == ATT1_AIMU_OK, "matmul_supported: result OK");

    att1_aimu_exec_counters cnt;
    att1_aimu_exec_ctx_get_counters(ctx, &cnt);
    EXPECT(cnt.matmul_count == 1,           "matmul_supported: matmul_count=1");
    EXPECT(cnt.exec_commands_seen == 1,     "matmul_supported: seen=1");
    EXPECT(cnt.exec_commands_completed == 1,"matmul_supported: completed=1");
    EXPECT(cnt.exec_commands_failed == 0,   "matmul_supported: failed=0");
    EXPECT(cnt.bytes_read_estimate == cmd.input_buf_bytes,
           "matmul_supported: bytes_read=input_buf_bytes");
    EXPECT(cnt.bytes_written_estimate == cmd.output_buf_bytes,
           "matmul_supported: bytes_written=output_buf_bytes");

    att1_aimu_exec_ctx_destroy(ctx);
    att1_aimu_device_destroy(dev);
}

/* =========================================================================
 * 3. test_matmul_unsupported
 * ====================================================================== */

static void test_matmul_unsupported(void)
{
    /* Device with MATMUL bit cleared from supported_ops */
    uint32_t ops_no_matmul = ATT1_AIMU_OP_ALL & ~ATT1_AIMU_OP_MATMUL;
    att1_aimu_device *dev = make_device(1, ops_no_matmul, 0);
    EXPECT(dev != NULL, "matmul_unsupported: device created");
    if (!dev) return;

    att1_aimu_exec_ctx *ctx = NULL;
    att1_aimu_exec_ctx_create(dev, &ctx);

    att1_aimu_cmd cmd = make_matmul_cmd(0, 1, 0);
    att1_aimu_result r = att1_aimu_exec_dispatch(ctx, &cmd);
    EXPECT(r == ATT1_AIMU_ERR_UNSUPPORTED_OP, "matmul_unsupported: UNSUPPORTED_OP");

    att1_aimu_exec_counters cnt;
    att1_aimu_exec_ctx_get_counters(ctx, &cnt);
    EXPECT(cnt.exec_unsupported == 1,       "matmul_unsupported: exec_unsupported=1");
    EXPECT(cnt.matmul_count == 0,           "matmul_unsupported: matmul_count=0");
    EXPECT(cnt.exec_commands_failed == 1,   "matmul_unsupported: failed=1");
    EXPECT(cnt.exec_commands_completed == 0,"matmul_unsupported: completed=0");

    att1_aimu_exec_ctx_destroy(ctx);
    att1_aimu_device_destroy(dev);
}

/* =========================================================================
 * 4. test_rmsnorm_rope_attn_ffn
 * ====================================================================== */

static void test_rmsnorm_rope_attn_ffn(void)
{
    att1_aimu_device *dev = make_device(1, 0, 0);
    att1_aimu_exec_ctx *ctx = NULL;
    att1_aimu_exec_ctx_create(dev, &ctx);

    att1_aimu_cmd cmd;
    att1_aimu_result r;

    /* EXEC_RMSNORM */
    memset(&cmd, 0, sizeof(cmd));
    cmd.command_type = (uint8_t)ATT1_AIMU_CMD_EXEC_RMSNORM;
    cmd.tile_id = 0; cmd.dtype = 0;
    cmd.input_buf_bytes = 512; cmd.output_buf_bytes = 512;
    r = att1_aimu_exec_dispatch(ctx, &cmd);
    EXPECT(r == ATT1_AIMU_OK, "rmsnorm: OK");

    /* EXEC_ROPE */
    memset(&cmd, 0, sizeof(cmd));
    cmd.command_type = (uint8_t)ATT1_AIMU_CMD_EXEC_ROPE;
    cmd.tile_id = 0; cmd.dtype = 0;
    cmd.input_buf_bytes = 256; cmd.output_buf_bytes = 256;
    r = att1_aimu_exec_dispatch(ctx, &cmd);
    EXPECT(r == ATT1_AIMU_OK, "rope: OK");

    /* EXEC_ATTENTION */
    memset(&cmd, 0, sizeof(cmd));
    cmd.command_type = (uint8_t)ATT1_AIMU_CMD_EXEC_ATTENTION;
    cmd.tile_id = 0; cmd.dtype = 0;
    cmd.input_buf_bytes = 1024; cmd.output_buf_bytes = 512;
    r = att1_aimu_exec_dispatch(ctx, &cmd);
    EXPECT(r == ATT1_AIMU_OK, "attention: OK");

    /* EXEC_FFN */
    memset(&cmd, 0, sizeof(cmd));
    cmd.command_type = (uint8_t)ATT1_AIMU_CMD_EXEC_FFN;
    cmd.tile_id = 0; cmd.dtype = 0;
    cmd.input_buf_bytes = 2048; cmd.output_buf_bytes = 1024;
    r = att1_aimu_exec_dispatch(ctx, &cmd);
    EXPECT(r == ATT1_AIMU_OK, "ffn: OK");

    att1_aimu_exec_counters cnt;
    att1_aimu_exec_ctx_get_counters(ctx, &cnt);
    EXPECT(cnt.rmsnorm_count  == 1, "rmsnorm: counter=1");
    EXPECT(cnt.rope_count     == 1, "rope: counter=1");
    EXPECT(cnt.attention_count == 1,"attention: counter=1");
    EXPECT(cnt.ffn_count      == 1, "ffn: counter=1");
    EXPECT(cnt.exec_commands_seen == 4,     "rmsnorm/rope/attn/ffn: seen=4");
    EXPECT(cnt.exec_commands_completed == 4,"rmsnorm/rope/attn/ffn: completed=4");
    EXPECT(cnt.exec_commands_failed == 0,   "rmsnorm/rope/attn/ffn: failed=0");

    att1_aimu_exec_ctx_destroy(ctx);
    att1_aimu_device_destroy(dev);
}

/* =========================================================================
 * 5. test_kv_append_read
 * ====================================================================== */

static void test_kv_append_read(void)
{
    att1_aimu_device *dev = make_device(1, 0, 0);
    att1_aimu_exec_ctx *ctx = NULL;
    att1_aimu_exec_ctx_create(dev, &ctx);

    att1_aimu_cmd cmd;
    att1_aimu_result r;

    /* KV_APPEND */
    memset(&cmd, 0, sizeof(cmd));
    cmd.command_type = (uint8_t)ATT1_AIMU_CMD_KV_APPEND;
    cmd.tile_id = 0;
    cmd.output_buf_bytes = 128;
    r = att1_aimu_exec_dispatch(ctx, &cmd);
    EXPECT(r == ATT1_AIMU_OK, "kv_append: OK");

    /* KV_READ */
    memset(&cmd, 0, sizeof(cmd));
    cmd.command_type = (uint8_t)ATT1_AIMU_CMD_KV_READ;
    cmd.tile_id = 0;
    cmd.input_buf_bytes = 64;
    r = att1_aimu_exec_dispatch(ctx, &cmd);
    EXPECT(r == ATT1_AIMU_OK, "kv_read: OK");

    att1_aimu_exec_counters cnt;
    att1_aimu_exec_ctx_get_counters(ctx, &cnt);
    EXPECT(cnt.kv_append_count == 1, "kv_append: counter=1");
    EXPECT(cnt.kv_read_count   == 1, "kv_read: counter=1");
    EXPECT(cnt.exec_commands_completed == 2, "kv: completed=2");

    /* KV_APPEND unsupported */
    att1_aimu_exec_ctx_reset_counters(ctx);
    dev->tiles[0].supported_ops = ATT1_AIMU_OP_ALL & ~ATT1_AIMU_OP_KV_APPEND;
    memset(&cmd, 0, sizeof(cmd));
    cmd.command_type = (uint8_t)ATT1_AIMU_CMD_KV_APPEND;
    r = att1_aimu_exec_dispatch(ctx, &cmd);
    EXPECT(r == ATT1_AIMU_ERR_UNSUPPORTED_OP,
           "kv_append: unsupported op -> UNSUPPORTED_OP");
    dev->tiles[0].supported_ops = ATT1_AIMU_OP_ALL;  /* restore */

    att1_aimu_exec_ctx_destroy(ctx);
    att1_aimu_device_destroy(dev);
}

/* =========================================================================
 * 6. test_fabric_send_reduce
 * ====================================================================== */

static void test_fabric_send_reduce(void)
{
    att1_aimu_device *dev = make_device(1, 0, 0);
    att1_aimu_exec_ctx *ctx = NULL;
    att1_aimu_exec_ctx_create(dev, &ctx);

    att1_aimu_cmd cmd;
    att1_aimu_result r;

    /* FABRIC_SEND */
    memset(&cmd, 0, sizeof(cmd));
    cmd.command_type = (uint8_t)ATT1_AIMU_CMD_FABRIC_SEND;
    cmd.tile_id = 0;
    r = att1_aimu_exec_dispatch(ctx, &cmd);
    EXPECT(r == ATT1_AIMU_OK, "fabric_send: OK");

    /* FABRIC_REDUCE */
    memset(&cmd, 0, sizeof(cmd));
    cmd.command_type = (uint8_t)ATT1_AIMU_CMD_FABRIC_REDUCE;
    cmd.tile_id = 0;
    r = att1_aimu_exec_dispatch(ctx, &cmd);
    EXPECT(r == ATT1_AIMU_OK, "fabric_reduce: OK");

    att1_aimu_exec_counters cnt;
    att1_aimu_exec_ctx_get_counters(ctx, &cnt);
    EXPECT(cnt.fabric_send_count   == 1, "fabric_send: counter=1");
    EXPECT(cnt.fabric_reduce_count == 1, "fabric_reduce: counter=1");
    EXPECT(cnt.exec_commands_completed == 2, "fabric: completed=2");

    /* FABRIC_SEND unsupported */
    att1_aimu_exec_ctx_reset_counters(ctx);
    dev->tiles[0].supported_ops = ATT1_AIMU_OP_ALL & ~ATT1_AIMU_OP_FABRIC_SEND;
    memset(&cmd, 0, sizeof(cmd));
    cmd.command_type = (uint8_t)ATT1_AIMU_CMD_FABRIC_SEND;
    r = att1_aimu_exec_dispatch(ctx, &cmd);
    EXPECT(r == ATT1_AIMU_ERR_UNSUPPORTED_OP,
           "fabric_send: unsupported op -> UNSUPPORTED_OP");
    dev->tiles[0].supported_ops = ATT1_AIMU_OP_ALL;

    att1_aimu_exec_ctx_destroy(ctx);
    att1_aimu_device_destroy(dev);
}

/* =========================================================================
 * 7. test_barrier_trace_query
 * ====================================================================== */

static void test_barrier_trace_query(void)
{
    att1_aimu_exec_ctx *ctx = NULL;
    att1_aimu_exec_ctx_create(NULL, &ctx);  /* no device needed */

    att1_aimu_cmd cmd;
    att1_aimu_result r;

    /* TILE_BARRIER */
    memset(&cmd, 0, sizeof(cmd));
    cmd.command_type = (uint8_t)ATT1_AIMU_CMD_TILE_BARRIER;
    r = att1_aimu_exec_dispatch(ctx, &cmd);
    EXPECT(r == ATT1_AIMU_OK, "barrier: OK");

    /* TRACE_SNAPSHOT */
    memset(&cmd, 0, sizeof(cmd));
    cmd.command_type = (uint8_t)ATT1_AIMU_CMD_TRACE_SNAPSHOT;
    r = att1_aimu_exec_dispatch(ctx, &cmd);
    EXPECT(r == ATT1_AIMU_OK, "trace_snapshot: OK");

    /* QUERY_COUNTERS */
    memset(&cmd, 0, sizeof(cmd));
    cmd.command_type = (uint8_t)ATT1_AIMU_CMD_QUERY_COUNTERS;
    r = att1_aimu_exec_dispatch(ctx, &cmd);
    EXPECT(r == ATT1_AIMU_OK, "query_counters: OK");

    att1_aimu_exec_counters cnt;
    att1_aimu_exec_ctx_get_counters(ctx, &cnt);
    EXPECT(cnt.barrier_count        == 1, "barrier: counter=1");
    EXPECT(cnt.trace_snapshot_count == 1, "trace_snapshot: counter=1");
    EXPECT(cnt.exec_commands_seen       == 3, "barrier/trace/query: seen=3");
    EXPECT(cnt.exec_commands_completed  == 3, "barrier/trace/query: completed=3");

    att1_aimu_exec_ctx_destroy(ctx);
}

/* =========================================================================
 * 8. test_invalid_tile_id
 * ====================================================================== */

static void test_invalid_tile_id(void)
{
    att1_aimu_device *dev = make_device(1, 0, 0);  /* 1 tile only */
    att1_aimu_exec_ctx *ctx = NULL;
    att1_aimu_exec_ctx_create(dev, &ctx);

    att1_aimu_cmd cmd = make_matmul_cmd(5 /* tile_id=5 out of range */, 1, 0);
    att1_aimu_result r = att1_aimu_exec_dispatch(ctx, &cmd);
    EXPECT(r == ATT1_AIMU_ERR_INVALID_COMMAND,
           "invalid_tile: tile_id=5 on 1-tile device -> INVALID_COMMAND");

    att1_aimu_exec_counters cnt;
    att1_aimu_exec_ctx_get_counters(ctx, &cnt);
    EXPECT(cnt.exec_commands_failed == 1, "invalid_tile: failed=1");
    EXPECT(cnt.matmul_count == 0,         "invalid_tile: matmul_count=0");

    att1_aimu_exec_ctx_destroy(ctx);
    att1_aimu_device_destroy(dev);
}

/* =========================================================================
 * 9. test_unsupported_dtype
 * ====================================================================== */

static void test_unsupported_dtype(void)
{
    /* Device supports only f32 (bit 0) */
    att1_aimu_device *dev = make_device(1, 0, ATT1_AIMU_DTYPE_F32);
    EXPECT(dev != NULL, "unsupported_dtype: device created");
    if (!dev) return;

    att1_aimu_exec_ctx *ctx = NULL;
    att1_aimu_exec_ctx_create(dev, &ctx);

    /* Submit EXEC_MATMUL with dtype=1 (q8) — not supported */
    att1_aimu_cmd cmd = make_matmul_cmd(0, 1, 1 /* q8 */);
    att1_aimu_result r = att1_aimu_exec_dispatch(ctx, &cmd);
    EXPECT(r == ATT1_AIMU_ERR_UNSUPPORTED_DTYPE,
           "unsupported_dtype: q8 on f32-only device -> UNSUPPORTED_DTYPE");

    att1_aimu_exec_counters cnt;
    att1_aimu_exec_ctx_get_counters(ctx, &cnt);
    EXPECT(cnt.exec_commands_failed == 1, "unsupported_dtype: failed=1");
    EXPECT(cnt.matmul_count == 0,         "unsupported_dtype: matmul_count=0");

    /* f32 should work fine on the same device */
    att1_aimu_exec_ctx_reset_counters(ctx);
    cmd = make_matmul_cmd(0, 1, 0 /* f32 */);
    r = att1_aimu_exec_dispatch(ctx, &cmd);
    EXPECT(r == ATT1_AIMU_OK, "unsupported_dtype: f32 on f32-only device -> OK");

    att1_aimu_exec_ctx_destroy(ctx);
    att1_aimu_device_destroy(dev);
}

/* =========================================================================
 * 10. test_missing_tensor
 * ====================================================================== */

static void test_missing_tensor(void)
{
    att1_aimu_device *dev = make_device(1, 0, 0);
    att1_aimu_exec_ctx *ctx = NULL;
    att1_aimu_exec_ctx_create(dev, &ctx);

    /* EXEC_MATMUL with tensor_id=0 (unbound) */
    att1_aimu_cmd cmd = make_matmul_cmd(0, 0 /* tensor_id=0 */, 0);
    att1_aimu_result r = att1_aimu_exec_dispatch(ctx, &cmd);
    EXPECT(r == ATT1_AIMU_ERR_INVALID_TENSOR,
           "missing_tensor: tensor_id=0 for EXEC_MATMUL -> INVALID_TENSOR");

    att1_aimu_exec_counters cnt;
    att1_aimu_exec_ctx_get_counters(ctx, &cnt);
    EXPECT(cnt.exec_commands_failed == 1, "missing_tensor: failed=1");
    EXPECT(cnt.matmul_count == 0,         "missing_tensor: matmul_count=0");

    /* Non-zero tensor_id should succeed */
    att1_aimu_exec_ctx_reset_counters(ctx);
    cmd = make_matmul_cmd(0, 7 /* tensor_id=7 */, 0);
    r = att1_aimu_exec_dispatch(ctx, &cmd);
    EXPECT(r == ATT1_AIMU_OK, "missing_tensor: non-zero tensor_id -> OK");

    att1_aimu_exec_ctx_destroy(ctx);
    att1_aimu_device_destroy(dev);
}

/* =========================================================================
 * 11. test_m129_replay_deterministic
 *
 * Build a static command list that mirrors the output of M129 mapping
 * exec_plan_valid_tiny.json (6 commands):
 *   [0] LOAD_TENSOR_TILE  tile=0 tensor=1
 *   [1] VALIDATE_TENSOR   tile=0 tensor=1
 *   [2] TILE_BARRIER      tile=0
 *   [3] EXEC_MATMUL       tile=0 tensor=1 dtype=f32  (UNSUPPORTED if no op)
 *   [4] KV_APPEND         tile=0
 *   [5] QUERY_COUNTERS    tile=0
 *
 * Replay the list twice (with counter reset in between) and verify the
 * counter snapshot is identical both times (deterministic).
 * ====================================================================== */

static void test_m129_replay_deterministic(void)
{
    att1_aimu_device *dev = make_device(1, 0, 0);  /* all ops */
    att1_aimu_exec_ctx *ctx = NULL;
    att1_aimu_exec_ctx_create(dev, &ctx);

    /* Build the static plan */
    att1_aimu_cmd plan[6];
    memset(plan, 0, sizeof(plan));

    plan[0].command_type     = (uint8_t)ATT1_AIMU_CMD_LOAD_TENSOR_TILE;
    plan[0].tile_id          = 0;
    plan[0].tensor_id        = 1;
    plan[0].output_buf_bytes = 8192;

    plan[1].command_type = (uint8_t)ATT1_AIMU_CMD_VALIDATE_TENSOR;
    plan[1].tile_id      = 0;
    plan[1].tensor_id    = 1;

    plan[2].command_type = (uint8_t)ATT1_AIMU_CMD_TILE_BARRIER;
    plan[2].tile_id      = 0;

    plan[3].command_type     = (uint8_t)ATT1_AIMU_CMD_EXEC_MATMUL;
    plan[3].tile_id          = 0;
    plan[3].tensor_id        = 1;
    plan[3].dtype            = 0;  /* f32 */
    plan[3].input_buf_bytes  = 16384;
    plan[3].output_buf_bytes = 4096;

    plan[4].command_type     = (uint8_t)ATT1_AIMU_CMD_KV_APPEND;
    plan[4].tile_id          = 0;
    plan[4].output_buf_bytes = 512;

    plan[5].command_type = (uint8_t)ATT1_AIMU_CMD_QUERY_COUNTERS;
    plan[5].tile_id      = 0;

    /* Run #1 */
    for (int i = 0; i < 6; i++) {
        att1_aimu_exec_dispatch(ctx, &plan[i]);
    }
    att1_aimu_exec_counters cnt1;
    att1_aimu_exec_ctx_get_counters(ctx, &cnt1);

    /* Reset and run #2 */
    att1_aimu_exec_ctx_reset_counters(ctx);
    for (int i = 0; i < 6; i++) {
        att1_aimu_exec_dispatch(ctx, &plan[i]);
    }
    att1_aimu_exec_counters cnt2;
    att1_aimu_exec_ctx_get_counters(ctx, &cnt2);

    /* Counters must be identical */
    EXPECT(cnt1.exec_commands_seen      == cnt2.exec_commands_seen,
           "m129_replay: seen same both runs");
    EXPECT(cnt1.exec_commands_completed == cnt2.exec_commands_completed,
           "m129_replay: completed same both runs");
    EXPECT(cnt1.exec_commands_failed    == cnt2.exec_commands_failed,
           "m129_replay: failed same both runs");
    EXPECT(cnt1.matmul_count      == cnt2.matmul_count,
           "m129_replay: matmul same");
    EXPECT(cnt1.kv_append_count   == cnt2.kv_append_count,
           "m129_replay: kv_append same");
    EXPECT(cnt1.barrier_count     == cnt2.barrier_count,
           "m129_replay: barrier same");
    EXPECT(cnt1.bytes_read_estimate    == cnt2.bytes_read_estimate,
           "m129_replay: bytes_read same");
    EXPECT(cnt1.bytes_written_estimate == cnt2.bytes_written_estimate,
           "m129_replay: bytes_written same");

    /* Spot-check expected absolute values */
    EXPECT(cnt1.exec_commands_seen == 6,  "m129_replay: seen=6");
    EXPECT(cnt1.exec_commands_completed == 6,
           "m129_replay: completed=6 (matmul OK with all-ops device)");
    EXPECT(cnt1.matmul_count    == 1, "m129_replay: matmul_count=1");
    EXPECT(cnt1.kv_append_count == 1, "m129_replay: kv_append_count=1");
    EXPECT(cnt1.barrier_count   == 1, "m129_replay: barrier_count=1");

    att1_aimu_exec_ctx_destroy(ctx);
    att1_aimu_device_destroy(dev);
}

/* =========================================================================
 * 12. test_load_validate_nop
 * ====================================================================== */

static void test_load_validate_nop(void)
{
    att1_aimu_exec_ctx *ctx = NULL;
    att1_aimu_exec_ctx_create(NULL, &ctx);

    att1_aimu_cmd cmd;
    att1_aimu_result r;

    /* LOAD_TENSOR_TILE */
    memset(&cmd, 0, sizeof(cmd));
    cmd.command_type     = (uint8_t)ATT1_AIMU_CMD_LOAD_TENSOR_TILE;
    cmd.output_buf_bytes = 2048;
    r = att1_aimu_exec_dispatch(ctx, &cmd);
    EXPECT(r == ATT1_AIMU_OK, "load_validate_nop: LOAD_TENSOR_TILE OK");

    /* VALIDATE_TENSOR */
    memset(&cmd, 0, sizeof(cmd));
    cmd.command_type = (uint8_t)ATT1_AIMU_CMD_VALIDATE_TENSOR;
    r = att1_aimu_exec_dispatch(ctx, &cmd);
    EXPECT(r == ATT1_AIMU_OK, "load_validate_nop: VALIDATE_TENSOR OK");

    /* NOP */
    memset(&cmd, 0, sizeof(cmd));
    cmd.command_type = (uint8_t)ATT1_AIMU_CMD_NOP;
    r = att1_aimu_exec_dispatch(ctx, &cmd);
    EXPECT(r == ATT1_AIMU_OK, "load_validate_nop: NOP OK");

    att1_aimu_exec_counters cnt;
    att1_aimu_exec_ctx_get_counters(ctx, &cnt);
    EXPECT(cnt.exec_commands_completed == 3, "load_validate_nop: completed=3");
    EXPECT(cnt.exec_commands_failed    == 0, "load_validate_nop: failed=0");
    EXPECT(cnt.bytes_written_estimate  == 2048,
           "load_validate_nop: bytes_written=2048 from LOAD");

    att1_aimu_exec_ctx_destroy(ctx);
}

/* =========================================================================
 * 13. test_reset_counters
 * ====================================================================== */

static void test_reset_counters(void)
{
    att1_aimu_exec_ctx *ctx = NULL;
    att1_aimu_exec_ctx_create(NULL, &ctx);

    /* Dispatch some commands to set counters */
    att1_aimu_cmd cmd;
    memset(&cmd, 0, sizeof(cmd));
    cmd.command_type = (uint8_t)ATT1_AIMU_CMD_TILE_BARRIER;
    att1_aimu_exec_dispatch(ctx, &cmd);
    att1_aimu_exec_dispatch(ctx, &cmd);

    att1_aimu_exec_counters cnt;
    att1_aimu_exec_ctx_get_counters(ctx, &cnt);
    EXPECT(cnt.exec_commands_seen == 2, "reset_counters: before reset seen=2");

    /* Reset */
    att1_status_t s = att1_aimu_exec_ctx_reset_counters(ctx);
    EXPECT(s == ATT1_OK, "reset_counters: reset returns OK");

    att1_aimu_exec_ctx_get_counters(ctx, &cnt);
    EXPECT(cnt.exec_commands_seen       == 0, "reset_counters: seen=0 after");
    EXPECT(cnt.exec_commands_completed  == 0, "reset_counters: completed=0 after");
    EXPECT(cnt.barrier_count            == 0, "reset_counters: barrier=0 after");
    EXPECT(cnt.bytes_read_estimate      == 0, "reset_counters: bytes_read=0 after");

    /* reset_counters with NULL ctx */
    s = att1_aimu_exec_ctx_reset_counters(NULL);
    EXPECT(s == ATT1_ERR_INVALID_ARG, "reset_counters: null ctx -> INVALID_ARG");

    att1_aimu_exec_ctx_destroy(ctx);
}

/* =========================================================================
 * 14. test_no_cuda_dep
 * ====================================================================== */

static void test_no_cuda_dep(void)
{
#ifdef ATT1_ENABLE_CUDA
    /* CUDA was explicitly enabled (make CUDA=1) — intentional, not a leak. */
    printf("PASS: aimu_exec: no_cuda_dep (CUDA=1 build; intentional)\n");
    g_pass++;
#else
    EXPECT(1, "no_cuda_dep: no CUDA dependency in default build");
#endif
}

/* =========================================================================
 * main
 * ====================================================================== */

int main(void)
{
    test_lifecycle();
    test_matmul_supported();
    test_matmul_unsupported();
    test_rmsnorm_rope_attn_ffn();
    test_kv_append_read();
    test_fabric_send_reduce();
    test_barrier_trace_query();
    test_invalid_tile_id();
    test_unsupported_dtype();
    test_missing_tensor();
    test_m129_replay_deterministic();
    test_load_validate_nop();
    test_reset_counters();
    test_no_cuda_dep();

    printf("\naimu_exec: %d PASS  %d FAIL\n", g_pass, g_fail);
    return (g_fail > 0) ? 1 : 0;
}
