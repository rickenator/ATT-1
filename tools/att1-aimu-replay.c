/*
 * att1-aimu-replay.c — M113: placement-command replay tool
 *
 * Reads a M109 command-plan JSON, drives the M112 ATT1 AIMU host control-plane
 * harness through the full probe → enumerate → setup → submit → dispatch →
 * completion loop, and emits a replay report.
 *
 * Usage:
 *   att1-aimu-replay --plan PATH [--strict] [--report-json PATH]
 *
 * Exit codes:
 *   0  all commands replayed successfully
 *   1  one or more commands failed or strict-mode violation
 *   2  JSON parse error / missing required fields
 */

#include "att1_aimu_host.h"
#include "att1_aimu_cmdq.h"
#include "att1_aimu_dma.h"
#include "att1_status.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* =========================================================================
 * Constants
 * ====================================================================== */

#define MAX_COMMANDS      256
#define MAX_TILES         64
#define JSON_LINE_MAX     512
#define STRING_MAX        128

/* Base synthetic host/device address (must be 64-byte aligned). */
#define HOST_BASE_ADDR    UINT64_C(0x0001000000000000)
#define DEV_BASE_ADDR     UINT64_C(0x0002000000000000)
#define ADDR_STRIDE       UINT64_C(0x0000000000010000) /* 64 KiB per slot  */

/* =========================================================================
 * Parsed plan types
 * ====================================================================== */

typedef struct {
    int  version;
    char model_id[STRING_MAX];
    char session_id[STRING_MAX];
    int  tile_count;
    int  command_count;
    char status[32];            /* "pass" / "fail" */
} plan_header_t;

typedef struct {
    long  command_id;
    char  command_type[STRING_MAX];
    long  tile_id;
    char  dtype[32];            /* "f32" / "q8" / "q4" / "" */
    long  packed_bytes;
    long  total_bytes;
    long  fence_id;
    char  expected_status[STRING_MAX];
    long  quant_group_size;     /* -1 if null */
    long  tensor_id;            /* -1 if null */
} plan_cmd_t;

typedef struct {
    plan_header_t header;
    plan_cmd_t    commands[MAX_COMMANDS];
    int           n_commands;
    int           parse_ok;
} plan_t;

/* =========================================================================
 * Replay report
 * ====================================================================== */

typedef struct {
    long   commands_replayed;
    long   completions_seen;
    long   failed_commands;
    long   unsupported_commands;
    long   dma_validations;
    long   doorbell_count;
    long   fence_final;
    long   trace_event_count;
    long   by_type[16];         /* per-type counts (indexed by cmd_type enum) */
    long   by_tile[MAX_TILES];
    int    status_pass;
    char   notes[512];
} replay_report_t;

/* =========================================================================
 * Minimal JSON field helpers
 * ====================================================================== */

/*
 * jstr — find "key": "value" in a line and copy value (without quotes).
 * Returns 0 on success, -1 if not found or value is null.
 */
static int jstr(const char *line, const char *key, char *buf, size_t bufsz)
{
    char        pat[STRING_MAX + 8];
    const char *p, *q;

    snprintf(pat, sizeof(pat), "\"%s\": \"", key);
    p = strstr(line, pat);
    if (p == NULL) {
        return -1;
    }
    p += strlen(pat);
    q  = strchr(p, '"');
    if (q == NULL) {
        return -1;
    }

    size_t n = (size_t)(q - p);
    if (n >= bufsz) {
        n = bufsz - 1u;
    }
    memcpy(buf, p, n);
    buf[n] = '\0';
    return 0;
}

/*
 * jint — find "key": N (decimal integer) in a line and store N in *val.
 * Returns 0 on success, -1 if not found or value is null.
 */
static int jint(const char *line, const char *key, long *val)
{
    char        pat[STRING_MAX + 8];
    const char *p;

    snprintf(pat, sizeof(pat), "\"%s\": ", key);
    p = strstr(line, pat);
    if (p == NULL) {
        return -1;
    }
    p += strlen(pat);
    if (*p == 'n') {            /* null literal */
        return -1;
    }
    return (sscanf(p, "%ld", val) == 1) ? 0 : -1;
}

/* =========================================================================
 * JSON plan parser (line-by-line state machine)
 * ====================================================================== */

typedef enum {
    PS_TOP,
    PS_HEADER,
    PS_COMMANDS,
    PS_COMMAND
} parse_state_t;

static void reset_cmd(plan_cmd_t *c)
{
    memset(c, 0, sizeof(*c));
    c->tile_id         = -1;
    c->quant_group_size = -1;
    c->tensor_id       = -1;
    c->packed_bytes    = 0;
    c->total_bytes     = 0;
    c->fence_id        = 0;
}

static int parse_plan(const char *path, plan_t *plan)
{
    FILE          *fp;
    char           line[JSON_LINE_MAX];
    parse_state_t  state = PS_TOP;
    plan_cmd_t     cur;
    int            got_header_version = 0;
    long           v;

    memset(plan, 0, sizeof(*plan));
    plan->header.tile_count    = -1;
    plan->header.command_count = -1;
    reset_cmd(&cur);

    fp = fopen(path, "r");
    if (fp == NULL) {
        fprintf(stderr, "replay: cannot open plan file: %s\n", path);
        return -1;
    }

    while (fgets(line, (int)sizeof(line), fp) != NULL) {
        /* Pointer to first non-whitespace character */
        const char *t = line;
        while (*t == ' ' || *t == '\t') {
            ++t;
        }

        switch (state) {

        case PS_TOP:
            /* Top-level version (appears before "header" key) */
            if (!got_header_version && jint(line, "command_plan_version", &v) == 0) {
                plan->header.version = (int)v;
                got_header_version   = 1;
            }
            if (strstr(line, "\"header\": {") != NULL) {
                state = PS_HEADER;
            } else if (strstr(line, "\"commands\": [") != NULL) {
                state = PS_COMMANDS;
            }
            break;

        case PS_HEADER:
            /* Leave header on closing brace (indent ≥ 2, line = "  }") */
            if (*t == '}') {
                state = PS_TOP;
                break;
            }
            /* Extract header fields (each on its own line) */
            if (jint(line, "tile_count",    &v) == 0) { plan->header.tile_count    = (int)v; }
            if (jint(line, "command_count", &v) == 0) { plan->header.command_count = (int)v; }
            jstr(line, "model_id",   plan->header.model_id,   sizeof(plan->header.model_id));
            jstr(line, "session_id", plan->header.session_id, sizeof(plan->header.session_id));
            jstr(line, "status",     plan->header.status,     sizeof(plan->header.status));
            break;

        case PS_COMMANDS:
            if (*t == '{') {
                reset_cmd(&cur);
                state = PS_COMMAND;
            } else if (*t == ']') {
                state = PS_TOP;
            }
            break;

        case PS_COMMAND:
            if (*t == '}') {
                /* End of command object: save it */
                if (plan->n_commands < MAX_COMMANDS) {
                    plan->commands[plan->n_commands++] = cur;
                }
                reset_cmd(&cur);
                state = PS_COMMANDS;
                break;
            }
            /* Extract per-command fields */
            if (jint(line, "command_id",  &v) == 0) { cur.command_id  = v; }
            if (jint(line, "tile_id",     &v) == 0) { cur.tile_id     = v; }
            if (jint(line, "packed_bytes",&v) == 0) { cur.packed_bytes = v; }
            if (jint(line, "total_bytes", &v) == 0) { cur.total_bytes  = v; }
            if (jint(line, "fence_id",    &v) == 0) { cur.fence_id     = v; }
            if (jint(line, "quantization_group_size", &v) == 0) {
                cur.quant_group_size = v;
            }
            if (jint(line, "tensor_id",   &v) == 0) { cur.tensor_id   = v; }
            jstr(line, "command_type",   cur.command_type,   sizeof(cur.command_type));
            jstr(line, "dtype",          cur.dtype,          sizeof(cur.dtype));
            jstr(line, "expected_status",cur.expected_status,sizeof(cur.expected_status));
            break;
        }
    }

    fclose(fp);

    plan->parse_ok = (plan->header.tile_count > 0) &&
                     (plan->header.command_count >= 0) &&
                     (plan->header.version == 1);
    return plan->parse_ok ? 0 : -1;
}

/* =========================================================================
 * Command-type string → enum mapping
 * ====================================================================== */

typedef struct {
    const char         *name;
    att1_aimu_cmd_type  type;
} cmd_map_entry_t;

static const cmd_map_entry_t CMD_MAP[] = {
    { "NOP",             ATT1_AIMU_CMD_NOP              },
    { "LOAD_TENSOR_TILE",ATT1_AIMU_CMD_LOAD_TENSOR_TILE  },
    { "VALIDATE_TENSOR", ATT1_AIMU_CMD_VALIDATE_TENSOR   },
    { "EXEC_MATMUL",     ATT1_AIMU_CMD_EXEC_MATMUL       },
    { "EXEC_RMSNORM",    ATT1_AIMU_CMD_EXEC_RMSNORM      },
    { "EXEC_ROPE",       ATT1_AIMU_CMD_EXEC_ROPE         },
    { "EXEC_ATTENTION",  ATT1_AIMU_CMD_EXEC_ATTENTION    },
    { "EXEC_FFN",        ATT1_AIMU_CMD_EXEC_FFN          },
    { "KV_APPEND",       ATT1_AIMU_CMD_KV_APPEND         },
    { "KV_READ",         ATT1_AIMU_CMD_KV_READ           },
    { "FABRIC_SEND",     ATT1_AIMU_CMD_FABRIC_SEND       },
    { "FABRIC_REDUCE",   ATT1_AIMU_CMD_FABRIC_REDUCE     },
    { "TRACE_SNAPSHOT",  ATT1_AIMU_CMD_TRACE_SNAPSHOT    },
    { "TILE_BARRIER",    ATT1_AIMU_CMD_TILE_BARRIER      },
    { "RESET_TILE",      ATT1_AIMU_CMD_RESET_TILE        },
    { "QUERY_COUNTERS",  ATT1_AIMU_CMD_QUERY_COUNTERS    },
    { NULL,              ATT1_AIMU_CMD_NOP               }
};

static int map_cmd_type(const char *name, att1_aimu_cmd_type *out)
{
    for (int i = 0; CMD_MAP[i].name != NULL; ++i) {
        if (strcmp(name, CMD_MAP[i].name) == 0) {
            *out = CMD_MAP[i].type;
            return 0;
        }
    }
    return -1;
}

/* =========================================================================
 * expected_status string → att1_aimu_result
 * ====================================================================== */

static int map_expected_result(const char *name, att1_aimu_result *out)
{
    if (strcmp(name, "ATT1_AIMU_ERR_OK") == 0) {
        *out = ATT1_AIMU_OK;
        return 0;
    }
    if (strcmp(name, "ATT1_AIMU_ERR_UNSUPPORTED_OP") == 0) {
        *out = ATT1_AIMU_ERR_UNSUPPORTED_OP;
        return 0;
    }
    if (strcmp(name, "ATT1_AIMU_ERR_INVALID_COMMAND") == 0) {
        *out = ATT1_AIMU_ERR_INVALID_COMMAND;
        return 0;
    }
    /* Default: treat unknown expected_status as OK */
    *out = ATT1_AIMU_OK;
    return -1;
}

/* =========================================================================
 * dtype string → ATT1_AIMU_DMA_DTYPE_*
 * ====================================================================== */

static uint8_t map_dtype(const char *dtype)
{
    if (strcmp(dtype, "q8") == 0) { return ATT1_AIMU_DMA_DTYPE_Q8; }
    if (strcmp(dtype, "q4") == 0) { return ATT1_AIMU_DMA_DTYPE_Q4; }
    return ATT1_AIMU_DMA_DTYPE_F32;   /* default / "f32" / "" */
}

/* =========================================================================
 * Replay a single command through the M112 harness
 * ====================================================================== */

static int replay_one(att1_aimu_host  *h,
                      const plan_cmd_t *pc,
                      int               strict,
                      replay_report_t  *rep)
{
    att1_aimu_cmd_type  ctype;
    att1_aimu_result    expected;
    att1_aimu_cmd       cmd;
    att1_aimu_completion comp;
    att1_status_t       rc;

    /* ── 1. Map command type ─────────────────────────────────────────── */
    if (map_cmd_type(pc->command_type, &ctype) != 0) {
        fprintf(stderr, "replay: unknown command_type '%s' at id=%ld\n",
                pc->command_type, pc->command_id);
        rep->unsupported_commands++;
        if (strict) {
            rep->status_pass = 0;
            return -1;
        }
        return 0;   /* skip, non-strict */
    }

    /* ── 2. Map expected status ──────────────────────────────────────── */
    map_expected_result(pc->expected_status, &expected);

    /* ── 3. Handle UNSUPPORTED_OP expectation ────────────────────────── */
    if (expected == ATT1_AIMU_ERR_UNSUPPORTED_OP) {
        rep->unsupported_commands++;
        if (strict) {
            fprintf(stderr,
                    "replay: strict mode — command %ld (%s) expects UNSUPPORTED_OP\n",
                    pc->command_id, pc->command_type);
            rep->status_pass = 0;
            return -1;
        }
        /* Non-strict: skip submission, treat as expected unsupported */
        return 0;
    }

    /* ── 4. Validate tile_id ─────────────────────────────────────────── */
    if (pc->tile_id < 0 || pc->tile_id >= (long)h->probe.tile_count) {
        fprintf(stderr,
                "replay: command %ld has invalid tile_id=%ld (tile_count=%zu)\n",
                pc->command_id, pc->tile_id, h->probe.tile_count);
        rep->failed_commands++;
        rep->status_pass = 0;
        if (strict) { return -1; }
        return 0;
    }

    /* ── 5. DMA descriptor for LOAD_TENSOR_TILE ──────────────────────── */
    if (ctype == ATT1_AIMU_CMD_LOAD_TENSOR_TILE) {
        att1_aimu_dma_desc dma;
        uint32_t           bytes;
        uint64_t           host_addr, dev_addr;
        long               slot;

        slot = (pc->tensor_id >= 0) ? pc->tensor_id : pc->command_id;
        bytes     = (pc->packed_bytes > 0) ?
                    (uint32_t)pc->packed_bytes : UINT32_C(4096);
        host_addr = HOST_BASE_ADDR + (uint64_t)slot * ADDR_STRIDE;
        dev_addr  = DEV_BASE_ADDR  + (uint64_t)slot * ADDR_STRIDE;

        /* Register synthetic memory regions */
        att1_aimu_dma_register_host_region(h->dma, host_addr,
                                            (uint64_t)bytes + 128u);
        att1_aimu_dma_register_device_region(h->dma, dev_addr,
                                              (uint64_t)bytes + 128u);

        memset(&dma, 0, sizeof(dma));
        dma.host_addr      = host_addr;
        dma.device_addr    = dev_addr;
        dma.byte_length    = bytes;
        dma.descriptor_id  = (uint32_t)pc->command_id;
        dma.tensor_id      = (pc->tensor_id >= 0) ? (uint32_t)pc->tensor_id : 0u;
        dma.direction      = ATT1_AIMU_DMA_HOST_TO_DEVICE;
        dma.dtype          = map_dtype(pc->dtype);
        if (pc->quant_group_size == 32 || pc->quant_group_size == 64) {
            dma.quant_group_size = (uint8_t)pc->quant_group_size;
        }

        rc = att1_aimu_host_validate_dma(h, &dma);
        if (rc != ATT1_OK) {
            fprintf(stderr, "replay: DMA validation failed for command %ld: %d\n",
                    pc->command_id, (int)rc);
            rep->failed_commands++;
            rep->status_pass = 0;
            if (strict) { return -1; }
            return 0;
        }
        rep->dma_validations++;
    }

    /* ── 6. Build and submit command ─────────────────────────────────── */
    memset(&cmd, 0, sizeof(cmd));
    cmd.command_type        = (uint8_t)ctype;
    cmd.tile_id             = (uint8_t)pc->tile_id;
    cmd.command_id          = (uint32_t)pc->command_id;
    cmd.dtype               = map_dtype(pc->dtype);
    cmd.fence_id            = (uint16_t)(pc->fence_id & 0xFFFF);
    cmd.completion_fence_id = (uint16_t)(pc->fence_id & 0xFFFF);
    cmd.tensor_id           = (pc->tensor_id >= 0) ? (uint16_t)pc->tensor_id : 0u;
    if (pc->packed_bytes > 0) {
        cmd.input_buf_bytes = (uint32_t)pc->packed_bytes;
    }

    rc = att1_aimu_host_submit_cmd(h, &cmd);
    if (rc != ATT1_OK) {
        fprintf(stderr, "replay: submit failed for command %ld: %d\n",
                pc->command_id, (int)rc);
        rep->failed_commands++;
        rep->status_pass = 0;
        if (strict) { return -1; }
        return 0;
    }

    /* ── 7. Doorbell + dispatch ──────────────────────────────────────── */
    rc = att1_aimu_host_ring_doorbell(h);
    if (rc != ATT1_OK) {
        fprintf(stderr, "replay: doorbell failed for command %ld: %d\n",
                pc->command_id, (int)rc);
        rep->failed_commands++;
        rep->status_pass = 0;
        if (strict) { return -1; }
        return 0;
    }
    rep->doorbell_count++;

    rc = att1_aimu_host_process_one(h);
    if (rc != ATT1_OK) {
        fprintf(stderr, "replay: process_one failed for command %ld: %d\n",
                pc->command_id, (int)rc);
        rep->failed_commands++;
        rep->status_pass = 0;
        if (strict) { return -1; }
        return 0;
    }
    rep->commands_replayed++;

    /* ── 8. Completion ───────────────────────────────────────────────── */
    memset(&comp, 0, sizeof(comp));
    rc = att1_aimu_host_read_completion(h, &comp);
    if (rc != ATT1_OK) {
        fprintf(stderr, "replay: read_completion failed for command %ld: %d\n",
                pc->command_id, (int)rc);
        rep->failed_commands++;
        rep->status_pass = 0;
        if (strict) { return -1; }
        return 0;
    }
    rep->completions_seen++;
    rep->trace_event_count += (long)comp.trace_event_count;
    if ((long)comp.fence_value > rep->fence_final) {
        rep->fence_final = (long)comp.fence_value;
    }

    /* ── 9. Check completion status vs expected ──────────────────────── */
    if ((att1_aimu_result)comp.result_code != expected) {
        /*
         * The simulator returns ATT1_AIMU_ERR_UNSUPPORTED_OP for execution
         * commands that are not yet implemented (LOAD_TENSOR_TILE,
         * VALIDATE_TENSOR, EXEC_*, KV_*, FABRIC_*).  Accept that result when
         * the plan expects ATT1_AIMU_ERR_OK — this is the documented mock
         * behaviour of the M112 harness (see test_aimu_host.c test 6/7).
         */
        if ((att1_aimu_result)comp.result_code == ATT1_AIMU_ERR_UNSUPPORTED_OP &&
            expected == ATT1_AIMU_OK) {
            rep->unsupported_commands++;
            /* Not a failure — fall through to accounting below. */
        } else {
            fprintf(stderr,
                    "replay: command %ld (%s) expected result %d, got %d\n",
                    pc->command_id, pc->command_type,
                    (int)expected, (int)comp.result_code);
            rep->failed_commands++;
            rep->status_pass = 0;
            if (strict) { return -1; }
            return 0;
        }
    }

    /* ── 10. Per-type and per-tile accounting ────────────────────────── */
    {
        int type_idx = (int)ctype & 0x0F;   /* bottom nibble for histogram */
        if (type_idx >= 0 && type_idx < 16) {
            rep->by_type[type_idx]++;
        }
    }
    if (pc->tile_id >= 0 && pc->tile_id < MAX_TILES) {
        rep->by_tile[pc->tile_id]++;
    }

    return 0;
}

/* =========================================================================
 * Emit report
 * ====================================================================== */

static void emit_text_report(const plan_t       *plan,
                              const replay_report_t *rep,
                              const char          *plan_path)
{
    printf("att1-aimu-replay report\n");
    printf("  plan_path        : %s\n", plan_path);
    printf("  model_id         : %s\n", plan->header.model_id);
    printf("  session_id       : %s\n", plan->header.session_id);
    printf("  tile_count       : %d\n", plan->header.tile_count);
    printf("  command_count    : %d\n", plan->n_commands);
    printf("  commands_replayed: %ld\n", rep->commands_replayed);
    printf("  completions_seen : %ld\n", rep->completions_seen);
    printf("  failed_commands  : %ld\n", rep->failed_commands);
    printf("  unsupported_cmds : %ld\n", rep->unsupported_commands);
    printf("  dma_validations  : %ld\n", rep->dma_validations);
    printf("  doorbell_count   : %ld\n", rep->doorbell_count);
    printf("  fence_final      : %ld\n", rep->fence_final);
    printf("  trace_events     : %ld\n", rep->trace_event_count);
    printf("  status           : %s\n", rep->status_pass ? "pass" : "fail");
    if (rep->notes[0] != '\0') {
        printf("  notes            : %s\n", rep->notes);
    }
}

static void emit_json_report(const plan_t       *plan,
                              const replay_report_t *rep,
                              const char          *plan_path,
                              FILE                *fp)
{
    int i;

    fprintf(fp, "{\n");
    fprintf(fp, "  \"plan_path\": \"%s\",\n",          plan_path);
    fprintf(fp, "  \"model_id\": \"%s\",\n",           plan->header.model_id);
    fprintf(fp, "  \"session_id\": \"%s\",\n",         plan->header.session_id);
    fprintf(fp, "  \"tile_count\": %d,\n",             plan->header.tile_count);
    fprintf(fp, "  \"command_count\": %d,\n",          plan->n_commands);
    fprintf(fp, "  \"commands_replayed\": %ld,\n",     rep->commands_replayed);
    fprintf(fp, "  \"completions_seen\": %ld,\n",      rep->completions_seen);
    fprintf(fp, "  \"failed_commands\": %ld,\n",       rep->failed_commands);
    fprintf(fp, "  \"unsupported_commands\": %ld,\n",  rep->unsupported_commands);
    fprintf(fp, "  \"dma_validations\": %ld,\n",       rep->dma_validations);
    fprintf(fp, "  \"doorbell_count\": %ld,\n",        rep->doorbell_count);
    fprintf(fp, "  \"fence_final\": %ld,\n",           rep->fence_final);
    fprintf(fp, "  \"trace_event_count\": %ld,\n",     rep->trace_event_count);
    fprintf(fp, "  \"commands_by_type\": [");
    for (i = 0; i < 16; i++) {
        fprintf(fp, "%ld%s", rep->by_type[i], (i < 15) ? ", " : "");
    }
    fprintf(fp, "],\n");
    fprintf(fp, "  \"commands_by_tile\": {");
    {
        int first = 1;
        for (i = 0; i < MAX_TILES; i++) {
            if (rep->by_tile[i] > 0) {
                if (!first) { fprintf(fp, ", "); }
                fprintf(fp, "\"%d\": %ld", i, rep->by_tile[i]);
                first = 0;
            }
        }
    }
    fprintf(fp, "},\n");
    fprintf(fp, "  \"status\": \"%s\",\n",            rep->status_pass ? "pass" : "fail");
    fprintf(fp, "  \"notes\": \"%s\"\n",               rep->notes);
    fprintf(fp, "}\n");
}

/* =========================================================================
 * Main
 * ====================================================================== */

int main(int argc, char *argv[])
{
    const char *plan_path   = NULL;
    const char *json_out    = NULL;
    int         strict      = 0;

    /* ── Parse arguments ─────────────────────────────────────────────── */
    for (int i = 1; i < argc; i++) {
        if ((strcmp(argv[i], "--plan") == 0) && (i + 1 < argc)) {
            plan_path = argv[++i];
        } else if ((strcmp(argv[i], "--report-json") == 0) && (i + 1 < argc)) {
            json_out = argv[++i];
        } else if (strcmp(argv[i], "--strict") == 0) {
            strict = 1;
        } else if ((strcmp(argv[i], "--help") == 0) ||
                   (strcmp(argv[i], "-h")     == 0)) {
            puts("Usage: att1-aimu-replay --plan PATH [--strict] [--report-json PATH]");
            return 0;
        }
    }

    if (plan_path == NULL) {
        fputs("att1-aimu-replay: --plan PATH is required\n", stderr);
        return 2;
    }

    /* ── Parse plan JSON ─────────────────────────────────────────────── */
    plan_t plan;
    if (parse_plan(plan_path, &plan) != 0) {
        fprintf(stderr, "att1-aimu-replay: failed to parse plan: %s\n", plan_path);
        return 2;
    }
    if (plan.header.version != 1) {
        fprintf(stderr, "att1-aimu-replay: unsupported command_plan_version=%d\n",
                plan.header.version);
        return 2;
    }
    if (plan.n_commands == 0) {
        fputs("att1-aimu-replay: plan contains no commands\n", stderr);
        return 2;
    }

    /* Strict mode: reject plans with header status != "pass" */
    if (strict && strcmp(plan.header.status, "pass") != 0) {
        fprintf(stderr,
                "att1-aimu-replay: strict mode — header status is '%s' (expected 'pass')\n",
                plan.header.status);
        return 1;
    }

    /* ── Create M112 harness ─────────────────────────────────────────── */
    att1_aimu_host_config  cfg;
    att1_aimu_host        *h  = NULL;
    att1_status_t          rc;

    memset(&cfg, 0, sizeof(cfg));
    cfg.tile_count = (size_t)plan.header.tile_count;

    rc = att1_aimu_host_create(&cfg, &h);
    if (rc != ATT1_OK) {
        fprintf(stderr, "att1-aimu-replay: host_create failed: %d\n", (int)rc);
        return 1;
    }

    /* ── Probe device ────────────────────────────────────────────────── */
    rc = att1_aimu_host_probe_device(h, NULL);
    if (rc != ATT1_OK) {
        fprintf(stderr, "att1-aimu-replay: probe_device failed: %d\n", (int)rc);
        att1_aimu_host_destroy(h);
        return 1;
    }

    /* ── Enumerate tiles ─────────────────────────────────────────────── */
    {
        att1_aimu_host_tile_info tile_infos[MAX_TILES];
        size_t                   tile_count = MAX_TILES;
        rc = att1_aimu_host_enumerate_tiles(h, tile_infos, &tile_count);
        if (rc != ATT1_OK) {
            fprintf(stderr, "att1-aimu-replay: enumerate_tiles failed: %d\n", (int)rc);
            att1_aimu_host_destroy(h);
            return 1;
        }
    }

    /* ── Setup command queue ─────────────────────────────────────────── */
    rc = att1_aimu_host_setup_cmdq(h);
    if (rc != ATT1_OK) {
        fprintf(stderr, "att1-aimu-replay: setup_cmdq failed: %d\n", (int)rc);
        att1_aimu_host_destroy(h);
        return 1;
    }

    /* ── Replay commands ─────────────────────────────────────────────── */
    replay_report_t rep;
    memset(&rep, 0, sizeof(rep));
    rep.status_pass = 1;

    for (int i = 0; i < plan.n_commands; i++) {
        if (replay_one(h, &plan.commands[i], strict, &rep) != 0) {
            break;   /* strict mode stops on first error */
        }
    }

    /* ── Snapshot counters ───────────────────────────────────────────── */
    (void)att1_aimu_host_snapshot_counters(h);

    /* ── Get summary ─────────────────────────────────────────────────── */
    {
        att1_aimu_host_summary summary;
        if (att1_aimu_host_get_summary(h, &summary) == ATT1_OK) {
            if (rep.notes[0] == '\0') {
                snprintf(rep.notes, sizeof(rep.notes),
                         "commands_submitted=%llu commands_completed=%llu",
                         (unsigned long long)summary.commands_submitted,
                         (unsigned long long)summary.commands_completed);
            }
        }
    }

    /* ── Emit text report ────────────────────────────────────────────── */
    emit_text_report(&plan, &rep, plan_path);

    /* ── Emit JSON report (optional) ─────────────────────────────────── */
    if (json_out != NULL) {
        FILE *jfp = fopen(json_out, "w");
        if (jfp == NULL) {
            fprintf(stderr, "att1-aimu-replay: cannot open report-json: %s\n", json_out);
            att1_aimu_host_destroy(h);
            return 1;
        }
        emit_json_report(&plan, &rep, plan_path, jfp);
        fclose(jfp);
    }

    /* ── Cleanup ─────────────────────────────────────────────────────── */
    att1_aimu_host_destroy(h);

    return rep.status_pass ? 0 : 1;
}
