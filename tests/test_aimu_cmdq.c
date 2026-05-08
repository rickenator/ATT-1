/*
 * test_aimu_cmdq.c  —  Unit tests for the AIMU command-queue simulator (M105)
 */

#include "att1_aimu_cmdq.h"
#include "att1_status.h"

#include <stdio.h>
#include <string.h>

/* -------------------------------------------------------------------------
 * Test helpers
 * ---------------------------------------------------------------------- */

#define PASS(name) do { printf("PASS: aimu_cmdq: %s\n", (name)); } while (0)
#define FAIL(name) do { printf("FAIL: aimu_cmdq: %s\n", (name)); return 1; } while (0)

#define REQUIRE(cond, name) \
    do { if (!(cond)) { FAIL(name); } } while (0)

/* Build a minimal command with zeroed fields and a given type / tile. */
static void make_cmd(att1_aimu_cmd     *cmd,
                     att1_aimu_cmd_type ctype,
                     uint8_t            tile_id)
{
    memset(cmd, 0, sizeof(*cmd));
    cmd->command_type = (uint8_t)ctype;
    cmd->tile_id      = tile_id;
}

/* Submit, dispatch all, poll one completion, return result_code. */
static att1_aimu_result submit_and_run(att1_aimu_cmdq    *q,
                                        att1_aimu_cmd     *cmd,
                                        att1_status_t     *submit_st)
{
    att1_aimu_completion comp;
    att1_status_t        st;

    st = att1_aimu_cmdq_submit(q, cmd);
    if (submit_st) {
        *submit_st = st;
    }
    if (st != ATT1_OK) {
        return (att1_aimu_result)0xFF; /* ERR_FATAL sentinel */
    }

    att1_aimu_cmdq_dispatch_all(q);

    memset(&comp, 0, sizeof(comp));
    st = att1_aimu_cmdq_poll_completion(q, &comp);
    if (st != ATT1_OK) {
        return (att1_aimu_result)0xFF;
    }
    return (att1_aimu_result)comp.result_code;
}

/* -------------------------------------------------------------------------
 * Individual test functions
 * ---------------------------------------------------------------------- */

/* create / destroy lifecycle */
static int test_create_destroy(void)
{
    att1_aimu_cmdq *q = NULL;
    att1_status_t   st;

    st = att1_aimu_cmdq_create(NULL, &q);
    REQUIRE(st == ATT1_OK, "create_destroy: create returns OK");
    REQUIRE(q != NULL,     "create_destroy: out pointer non-null");
    REQUIRE(q->magic == ATT1_AIMU_CMDQ_MAGIC, "create_destroy: magic set");
    REQUIRE(q->config.tile_count == 1u,        "create_destroy: default tile_count 1");

    att1_aimu_cmdq_destroy(q);
    att1_aimu_cmdq_destroy(NULL); /* must not crash */
    PASS("create_destroy");
    return 0;
}

/* create with explicit config */
static int test_create_with_config(void)
{
    att1_aimu_cmdq_config cfg;
    att1_aimu_cmdq       *q = NULL;
    att1_status_t         st;

    cfg.tile_count      = 4u;
    cfg.cmd_ring_depth  = 64u;
    cfg.comp_ring_depth = 32u;

    st = att1_aimu_cmdq_create(&cfg, &q);
    REQUIRE(st == ATT1_OK,           "create_config: returns OK");
    REQUIRE(q->config.tile_count == 4u,     "create_config: tile_count 4");
    REQUIRE(q->config.cmd_ring_depth == 64u, "create_config: cmd_ring_depth 64");

    att1_aimu_cmdq_destroy(q);
    PASS("create_with_config");
    return 0;
}

/* create with invalid config */
static int test_create_invalid(void)
{
    att1_aimu_cmdq_config cfg;
    att1_aimu_cmdq       *q = NULL;
    att1_status_t         st;

    /* tile_count = 0 */
    cfg.tile_count = 0u; cfg.cmd_ring_depth = 64u; cfg.comp_ring_depth = 64u;
    st = att1_aimu_cmdq_create(&cfg, &q);
    REQUIRE(st == ATT1_ERR_INVALID_ARG, "create_invalid: tile_count 0 rejected");

    /* tile_count > max */
    cfg.tile_count = ATT1_AIMU_CMDQ_MAX_TILES + 1u;
    st = att1_aimu_cmdq_create(&cfg, &q);
    REQUIRE(st == ATT1_ERR_INVALID_ARG, "create_invalid: tile_count too large");

    /* non-power-of-two cmd ring */
    cfg.tile_count = 1u; cfg.cmd_ring_depth = 7u;
    st = att1_aimu_cmdq_create(&cfg, &q);
    REQUIRE(st == ATT1_ERR_INVALID_ARG, "create_invalid: non-pow2 cmd_ring_depth");

    /* null out pointer */
    st = att1_aimu_cmdq_create(NULL, NULL);
    REQUIRE(st == ATT1_ERR_INVALID_ARG, "create_invalid: null out rejected");

    PASS("create_invalid");
    return 0;
}

/* submit NOP and complete it */
static int test_nop(void)
{
    att1_aimu_cmdq       *q = NULL;
    att1_aimu_cmd         cmd;
    att1_aimu_completion  comp;
    att1_status_t         st;

    att1_aimu_cmdq_create(NULL, &q);

    make_cmd(&cmd, ATT1_AIMU_CMD_NOP, 0u);
    st = att1_aimu_cmdq_submit(q, &cmd);
    REQUIRE(st == ATT1_OK, "nop: submit returns OK");
    REQUIRE(att1_aimu_cmdq_pending(q) == 1u, "nop: pending == 1");

    st = att1_aimu_cmdq_dispatch_one(q);
    REQUIRE(st == ATT1_OK, "nop: dispatch_one returns OK");
    REQUIRE(att1_aimu_cmdq_pending(q) == 0u, "nop: pending == 0 after dispatch");
    REQUIRE(att1_aimu_cmdq_completions_available(q) == 1u, "nop: 1 completion");

    memset(&comp, 0, sizeof(comp));
    st = att1_aimu_cmdq_poll_completion(q, &comp);
    REQUIRE(st == ATT1_OK,                "nop: poll returns OK");
    REQUIRE(comp.result_code == ATT1_AIMU_OK, "nop: result OK");
    REQUIRE(comp.command_id == 1u,        "nop: command_id == 1");
    REQUIRE(att1_aimu_cmdq_completions_available(q) == 0u, "nop: comp ring empty");

    att1_aimu_cmdq_destroy(q);
    PASS("nop");
    return 0;
}

/* FIFO ordering: submit A, B, C; completions come back in order */
static int test_fifo_ordering(void)
{
    att1_aimu_cmdq       *q = NULL;
    att1_aimu_cmd         cmd;
    att1_aimu_completion  comp;
    att1_status_t         st;

    att1_aimu_cmdq_create(NULL, &q);

    make_cmd(&cmd, ATT1_AIMU_CMD_NOP, 0u); att1_aimu_cmdq_submit(q, &cmd);
    make_cmd(&cmd, ATT1_AIMU_CMD_RESET_TILE, 0u); att1_aimu_cmdq_submit(q, &cmd);
    make_cmd(&cmd, ATT1_AIMU_CMD_QUERY_COUNTERS, 0u); att1_aimu_cmdq_submit(q, &cmd);

    REQUIRE(att1_aimu_cmdq_pending(q) == 3u, "fifo: 3 pending");
    att1_aimu_cmdq_dispatch_all(q);
    REQUIRE(att1_aimu_cmdq_completions_available(q) == 3u, "fifo: 3 completions");

    st = att1_aimu_cmdq_poll_completion(q, &comp);
    REQUIRE(st == ATT1_OK && comp.command_id == 1u, "fifo: first command_id == 1");

    st = att1_aimu_cmdq_poll_completion(q, &comp);
    REQUIRE(st == ATT1_OK && comp.command_id == 2u, "fifo: second command_id == 2");

    st = att1_aimu_cmdq_poll_completion(q, &comp);
    REQUIRE(st == ATT1_OK && comp.command_id == 3u, "fifo: third command_id == 3");

    att1_aimu_cmdq_destroy(q);
    PASS("fifo_ordering");
    return 0;
}

/* queue full: fill ring and verify the correct error */
static int test_queue_full(void)
{
    att1_aimu_cmdq_config  cfg;
    att1_aimu_cmdq        *q = NULL;
    att1_aimu_cmd          cmd;
    att1_status_t          st;

    /* depth=4: ring holds 3 live commands (one slot reserved for head==tail) */
    cfg.tile_count = 1u; cfg.cmd_ring_depth = 4u; cfg.comp_ring_depth = 256u;
    att1_aimu_cmdq_create(&cfg, &q);

    make_cmd(&cmd, ATT1_AIMU_CMD_NOP, 0u);
    REQUIRE(att1_aimu_cmdq_submit(q, &cmd) == ATT1_OK,           "queue_full: submit 1 OK");
    REQUIRE(att1_aimu_cmdq_submit(q, &cmd) == ATT1_OK,           "queue_full: submit 2 OK");
    REQUIRE(att1_aimu_cmdq_submit(q, &cmd) == ATT1_OK,           "queue_full: submit 3 OK");
    st = att1_aimu_cmdq_submit(q, &cmd);
    REQUIRE(st == ATT1_ERR_QUEUE_FULL,                            "queue_full: 4th submit fails");

    att1_aimu_cmdq_destroy(q);
    PASS("queue_full");
    return 0;
}

/* empty queue: dispatch and poll on empty ring */
static int test_empty_queue(void)
{
    att1_aimu_cmdq       *q = NULL;
    att1_aimu_completion  comp;
    att1_status_t         st;

    att1_aimu_cmdq_create(NULL, &q);

    st = att1_aimu_cmdq_dispatch_one(q);
    REQUIRE(st == ATT1_ERR_QUEUE_EMPTY, "empty: dispatch_one on empty");

    st = att1_aimu_cmdq_poll_completion(q, &comp);
    REQUIRE(st == ATT1_ERR_QUEUE_EMPTY, "empty: poll on empty");

    att1_aimu_cmdq_destroy(q);
    PASS("empty_queue");
    return 0;
}

/* invalid tile ID is rejected at submit time */
static int test_invalid_tile_id(void)
{
    att1_aimu_cmdq *q = NULL;
    att1_aimu_cmd   cmd;
    att1_status_t   st;

    /* Create with 2 tiles; tile 2 is out of range. */
    att1_aimu_cmdq_config cfg = { 2u, 64u, 64u };
    att1_aimu_cmdq_create(&cfg, &q);

    make_cmd(&cmd, ATT1_AIMU_CMD_NOP, 2u);
    st = att1_aimu_cmdq_submit(q, &cmd);
    REQUIRE(st == ATT1_ERR_INVALID_ARG, "invalid_tile: submit rejected");

    att1_aimu_cmdq_destroy(q);
    PASS("invalid_tile_id");
    return 0;
}

/* unsupported EXEC_MATMUL returns ERR_UNSUPPORTED_OP in the completion */
static int test_exec_matmul_unsupported(void)
{
    att1_aimu_cmdq       *q = NULL;
    att1_aimu_cmd         cmd;
    att1_aimu_completion  comp;
    att1_aimu_cmdq_counters ctrs;

    att1_aimu_cmdq_create(NULL, &q);

    make_cmd(&cmd, ATT1_AIMU_CMD_EXEC_MATMUL, 0u);
    att1_aimu_cmdq_submit(q, &cmd);
    att1_aimu_cmdq_dispatch_all(q);
    att1_aimu_cmdq_poll_completion(q, &comp);

    REQUIRE(comp.result_code == ATT1_AIMU_ERR_UNSUPPORTED_OP,
            "exec_matmul_unsupported: result ERR_UNSUPPORTED_OP");

    att1_aimu_cmdq_get_counters(q, &ctrs);
    REQUIRE(ctrs.unsupported_commands == 1u,
            "exec_matmul_unsupported: unsupported_commands == 1");

    att1_aimu_cmdq_destroy(q);
    PASS("exec_matmul_unsupported");
    return 0;
}

/* all EXEC_* commands produce a completion (no silent drop) */
static int test_exec_commands_produce_completion(void)
{
    att1_aimu_cmdq       *q = NULL;
    att1_aimu_cmd         cmd;
    att1_aimu_completion  comp;

    att1_aimu_cmd_type exec_types[] = {
        ATT1_AIMU_CMD_EXEC_MATMUL,
        ATT1_AIMU_CMD_EXEC_RMSNORM,
        ATT1_AIMU_CMD_EXEC_ROPE,
        ATT1_AIMU_CMD_EXEC_ATTENTION,
        ATT1_AIMU_CMD_EXEC_FFN,
        ATT1_AIMU_CMD_KV_APPEND,
        ATT1_AIMU_CMD_KV_READ,
        ATT1_AIMU_CMD_FABRIC_SEND,
        ATT1_AIMU_CMD_FABRIC_REDUCE,
        ATT1_AIMU_CMD_LOAD_TENSOR_TILE,
        ATT1_AIMU_CMD_VALIDATE_TENSOR
    };
    size_t n = sizeof(exec_types) / sizeof(exec_types[0]);

    att1_aimu_cmdq_create(NULL, &q);

    for (size_t i = 0u; i < n; i++) {
        make_cmd(&cmd, exec_types[i], 0u);
        att1_aimu_cmdq_submit(q, &cmd);
    }
    att1_aimu_cmdq_dispatch_all(q);

    REQUIRE(att1_aimu_cmdq_completions_available(q) == n,
            "exec_produce_completion: completion count matches submit count");

    for (size_t i = 0u; i < n; i++) {
        att1_aimu_cmdq_poll_completion(q, &comp);
        REQUIRE(comp.result_code == ATT1_AIMU_ERR_UNSUPPORTED_OP,
                "exec_produce_completion: result ERR_UNSUPPORTED_OP");
    }

    att1_aimu_cmdq_destroy(q);
    PASS("exec_commands_produce_completion");
    return 0;
}

/* RESET_TILE increments the reset counter */
static int test_reset_tile_counter(void)
{
    att1_aimu_cmdq          *q = NULL;
    att1_aimu_cmd            cmd;
    att1_aimu_cmdq_counters  ctrs;

    att1_aimu_cmdq_create(NULL, &q);

    make_cmd(&cmd, ATT1_AIMU_CMD_RESET_TILE, 0u);
    att1_aimu_cmdq_submit(q, &cmd);
    make_cmd(&cmd, ATT1_AIMU_CMD_RESET_TILE, 0u);
    att1_aimu_cmdq_submit(q, &cmd);
    att1_aimu_cmdq_dispatch_all(q);

    att1_aimu_cmdq_get_counters(q, &ctrs);
    REQUIRE(ctrs.resets == 2u,                  "reset_counter: resets == 2");
    REQUIRE(ctrs.commands_completed == 2u,       "reset_counter: completed == 2");

    att1_aimu_cmdq_destroy(q);
    PASS("reset_tile_counter");
    return 0;
}

/* QUERY_COUNTERS completes with OK */
static int test_query_counters(void)
{
    att1_aimu_cmdq       *q = NULL;
    att1_aimu_cmd         cmd;
    att1_aimu_result      r;

    att1_aimu_cmdq_create(NULL, &q);
    make_cmd(&cmd, ATT1_AIMU_CMD_QUERY_COUNTERS, 0u);
    r = submit_and_run(q, &cmd, NULL);
    REQUIRE(r == ATT1_AIMU_OK, "query_counters: result OK");

    att1_aimu_cmdq_destroy(q);
    PASS("query_counters");
    return 0;
}

/* fence values are monotonically increasing */
static int test_fence_monotonic(void)
{
    att1_aimu_cmdq       *q = NULL;
    att1_aimu_cmd         cmd;
    att1_aimu_completion  comp;
    uint32_t              prev_fence = 0u;
    int                   seen_nonzero = 0;

    att1_aimu_cmdq_create(NULL, &q);

    for (int i = 0; i < 8; i++) {
        make_cmd(&cmd, ATT1_AIMU_CMD_NOP, 0u);
        cmd.completion_fence_id = (uint16_t)(i + 1);
        att1_aimu_cmdq_submit(q, &cmd);
    }
    att1_aimu_cmdq_dispatch_all(q);

    while (att1_aimu_cmdq_poll_completion(q, &comp) == ATT1_OK) {
        if (comp.fence_value != 0u) {
            REQUIRE(comp.fence_value > prev_fence,
                    "fence_monotonic: fence_value is strictly increasing");
            prev_fence = comp.fence_value;
            seen_nonzero = 1;
        }
    }
    REQUIRE(seen_nonzero, "fence_monotonic: at least one non-zero fence");

    att1_aimu_cmdq_destroy(q);
    PASS("fence_monotonic");
    return 0;
}

/* completion ring full: dispatcher returns ATT1_ERR_QUEUE_FULL */
static int test_completion_ring_full(void)
{
    att1_aimu_cmdq_config  cfg;
    att1_aimu_cmdq        *q = NULL;
    att1_aimu_cmd          cmd;
    att1_status_t          st;

    /* comp_ring_depth=2: only 1 live completion slot */
    cfg.tile_count = 1u; cfg.cmd_ring_depth = 256u; cfg.comp_ring_depth = 2u;
    att1_aimu_cmdq_create(&cfg, &q);

    /* submit 3 NOPs — after one dispatch the comp ring is full */
    make_cmd(&cmd, ATT1_AIMU_CMD_NOP, 0u);
    att1_aimu_cmdq_submit(q, &cmd);
    att1_aimu_cmdq_submit(q, &cmd);
    att1_aimu_cmdq_submit(q, &cmd);

    att1_aimu_cmdq_dispatch_one(q); /* fills the one comp slot */
    st = att1_aimu_cmdq_dispatch_one(q);
    REQUIRE(st == ATT1_ERR_QUEUE_FULL, "compq_full: dispatch returns QUEUE_FULL");

    att1_aimu_cmdq_destroy(q);
    PASS("completion_ring_full");
    return 0;
}

/* dispatch_all returns OK when ring is empty */
static int test_dispatch_all_empty(void)
{
    att1_aimu_cmdq *q = NULL;
    att1_status_t   st;

    att1_aimu_cmdq_create(NULL, &q);
    st = att1_aimu_cmdq_dispatch_all(q);
    REQUIRE(st == ATT1_OK, "dispatch_all_empty: returns OK on empty ring");

    att1_aimu_cmdq_destroy(q);
    PASS("dispatch_all_empty");
    return 0;
}

/* command_type name helper returns stable strings */
static int test_cmd_type_names(void)
{
    REQUIRE(strcmp(att1_aimu_cmd_type_name(ATT1_AIMU_CMD_NOP),
                   "NOP") == 0,
            "name_nop");
    REQUIRE(strcmp(att1_aimu_cmd_type_name(ATT1_AIMU_CMD_EXEC_MATMUL),
                   "EXEC_MATMUL") == 0,
            "name_exec_matmul");
    REQUIRE(strcmp(att1_aimu_cmd_type_name(ATT1_AIMU_CMD_LOAD_TENSOR_TILE),
                   "LOAD_TENSOR_TILE") == 0,
            "name_load_tensor_tile");
    REQUIRE(strcmp(att1_aimu_cmd_type_name(ATT1_AIMU_CMD_QUERY_COUNTERS),
                   "QUERY_COUNTERS") == 0,
            "name_query_counters");
    REQUIRE(strcmp(att1_aimu_cmd_type_name((att1_aimu_cmd_type)0xEE),
                   "UNKNOWN") == 0,
            "name_unknown");
    PASS("cmd_type_names");
    return 0;
}

/* result code name helper */
static int test_result_names(void)
{
    REQUIRE(strcmp(att1_aimu_result_name(ATT1_AIMU_OK),               "OK") == 0,
            "result_name_ok");
    REQUIRE(strcmp(att1_aimu_result_name(ATT1_AIMU_ERR_UNSUPPORTED_OP),
                   "ERR_UNSUPPORTED_OP") == 0,
            "result_name_unsupported_op");
    REQUIRE(strcmp(att1_aimu_result_name(ATT1_AIMU_ERR_FATAL),         "ERR_FATAL") == 0,
            "result_name_fatal");
    REQUIRE(strcmp(att1_aimu_result_name((att1_aimu_result)0x77),      "UNKNOWN") == 0,
            "result_name_unknown");
    PASS("result_names");
    return 0;
}

/* get_counters tracks all submitted / completed / failed */
static int test_counter_tracking(void)
{
    att1_aimu_cmdq          *q = NULL;
    att1_aimu_cmd            cmd;
    att1_aimu_cmdq_counters  ctrs;

    att1_aimu_cmdq_create(NULL, &q);

    /* 2 NOPs (OK) + 1 EXEC_MATMUL (unsupported) */
    make_cmd(&cmd, ATT1_AIMU_CMD_NOP, 0u);
    att1_aimu_cmdq_submit(q, &cmd);
    att1_aimu_cmdq_submit(q, &cmd);
    make_cmd(&cmd, ATT1_AIMU_CMD_EXEC_MATMUL, 0u);
    att1_aimu_cmdq_submit(q, &cmd);

    att1_aimu_cmdq_dispatch_all(q);
    att1_aimu_cmdq_get_counters(q, &ctrs);

    REQUIRE(ctrs.commands_submitted == 3u,   "counters: submitted == 3");
    REQUIRE(ctrs.commands_completed == 3u,   "counters: completed == 3");
    REQUIRE(ctrs.commands_failed    == 0u,   "counters: failed == 0");
    REQUIRE(ctrs.unsupported_commands == 1u, "counters: unsupported == 1");
    REQUIRE(ctrs.queue_full_count   == 0u,   "counters: queue_full == 0");

    att1_aimu_cmdq_destroy(q);
    PASS("counter_tracking");
    return 0;
}

/* queue_full_count increments when ring is full */
static int test_queue_full_counter(void)
{
    att1_aimu_cmdq_config    cfg;
    att1_aimu_cmdq          *q = NULL;
    att1_aimu_cmd            cmd;
    att1_aimu_cmdq_counters  ctrs;

    cfg.tile_count = 1u; cfg.cmd_ring_depth = 4u; cfg.comp_ring_depth = 64u;
    att1_aimu_cmdq_create(&cfg, &q);

    make_cmd(&cmd, ATT1_AIMU_CMD_NOP, 0u);
    att1_aimu_cmdq_submit(q, &cmd);
    att1_aimu_cmdq_submit(q, &cmd);
    att1_aimu_cmdq_submit(q, &cmd);
    /* 4th and 5th should fail */
    att1_aimu_cmdq_submit(q, &cmd);
    att1_aimu_cmdq_submit(q, &cmd);

    att1_aimu_cmdq_get_counters(q, &ctrs);
    REQUIRE(ctrs.queue_full_count == 2u, "queue_full_counter: count == 2");

    att1_aimu_cmdq_destroy(q);
    PASS("queue_full_counter");
    return 0;
}

/* command_id is monotonically assigned starting at 1 */
static int test_command_id_monotonic(void)
{
    att1_aimu_cmdq       *q = NULL;
    att1_aimu_cmd         cmd;
    att1_aimu_completion  comp;

    att1_aimu_cmdq_create(NULL, &q);

    for (int i = 0; i < 5; i++) {
        make_cmd(&cmd, ATT1_AIMU_CMD_NOP, 0u);
        att1_aimu_cmdq_submit(q, &cmd);
    }
    att1_aimu_cmdq_dispatch_all(q);

    for (uint32_t expected = 1u; expected <= 5u; expected++) {
        att1_aimu_cmdq_poll_completion(q, &comp);
        REQUIRE(comp.command_id == expected, "cmd_id_monotonic: expected ID");
    }

    att1_aimu_cmdq_destroy(q);
    PASS("command_id_monotonic");
    return 0;
}

/* null argument safety */
static int test_null_safety(void)
{
    att1_aimu_cmdq      *q = NULL;
    att1_aimu_cmd        cmd;
    att1_aimu_completion comp;
    att1_aimu_cmdq_counters ctrs;

    memset(&cmd,  0, sizeof(cmd));
    memset(&comp, 0, sizeof(comp));

    REQUIRE(att1_aimu_cmdq_submit(NULL, &cmd)             == ATT1_ERR_INVALID_ARG,
            "null: submit NULL q");
    REQUIRE(att1_aimu_cmdq_submit(q, NULL)                == ATT1_ERR_INVALID_ARG,
            "null: submit NULL cmd");
    REQUIRE(att1_aimu_cmdq_dispatch_one(NULL)             == ATT1_ERR_INVALID_ARG,
            "null: dispatch_one NULL");
    REQUIRE(att1_aimu_cmdq_dispatch_all(NULL)             == ATT1_ERR_INVALID_ARG,
            "null: dispatch_all NULL");
    REQUIRE(att1_aimu_cmdq_poll_completion(NULL, &comp)   == ATT1_ERR_INVALID_ARG,
            "null: poll NULL q");
    REQUIRE(att1_aimu_cmdq_poll_completion(q, NULL)       == ATT1_ERR_INVALID_ARG,
            "null: poll NULL out");
    REQUIRE(att1_aimu_cmdq_get_counters(NULL, &ctrs)      == ATT1_ERR_INVALID_ARG,
            "null: counters NULL q");
    REQUIRE(att1_aimu_cmdq_get_counters(q, NULL)          == ATT1_ERR_INVALID_ARG,
            "null: counters NULL out");
    REQUIRE(att1_aimu_cmdq_pending(NULL)                  == 0u,
            "null: pending NULL");
    REQUIRE(att1_aimu_cmdq_completions_available(NULL)    == 0u,
            "null: completions_available NULL");

    PASS("null_safety");
    return 0;
}

/* descriptor size is exactly 64 bytes */
static int test_descriptor_size(void)
{
    REQUIRE(sizeof(att1_aimu_cmd) == 64u,
            "descriptor_size: sizeof(att1_aimu_cmd) == 64");
    PASS("descriptor_size");
    return 0;
}

/* -------------------------------------------------------------------------
 * main
 * ---------------------------------------------------------------------- */

int main(void)
{
    int rc = 0;

    rc |= test_create_destroy();
    rc |= test_create_with_config();
    rc |= test_create_invalid();
    rc |= test_nop();
    rc |= test_fifo_ordering();
    rc |= test_queue_full();
    rc |= test_empty_queue();
    rc |= test_invalid_tile_id();
    rc |= test_exec_matmul_unsupported();
    rc |= test_exec_commands_produce_completion();
    rc |= test_reset_tile_counter();
    rc |= test_query_counters();
    rc |= test_fence_monotonic();
    rc |= test_completion_ring_full();
    rc |= test_dispatch_all_empty();
    rc |= test_cmd_type_names();
    rc |= test_result_names();
    rc |= test_counter_tracking();
    rc |= test_queue_full_counter();
    rc |= test_command_id_monotonic();
    rc |= test_null_safety();
    rc |= test_descriptor_size();

    return rc;
}
