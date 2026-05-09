/*
 * att1-aimu-mmio-replay.c  —  M122: command-plan replay via M121 userspace MMIO emulator
 *
 * Reads a M109 command-plan JSON (same format accepted by M113 att1-aimu-replay)
 * and drives the plan through the M121 att1_aimu_userspace interface instead of
 * directly through the M112 att1_aimu_host harness.
 *
 * This exercises the full emulator code path:
 *
 *   open M121 emulator → probe → enumerate tiles → setup_cmdq
 *   → (for each command) validate_dma / submit_cmd → ring_doorbell → drain
 *   → snapshot_counters → get_summary → close
 *
 * The BAR0 register file is optionally backed by an mmap'd file so external
 * tools can observe register state after replay.
 *
 * This is a userspace emulator replay, NOT real PCIe/MMIO hardware.
 * Tile memory capacity is register metadata only; no huge buffers are
 * allocated.  DMA payloads remain descriptor-only (no real tensor data).
 *
 * Usage:
 *   att1-aimu-mmio-replay --plan PATH
 *       [--bar0-file PATH]
 *       [--tiles N]
 *       [--tile-memory-mib N]
 *       [--kv-memory-mib N]
 *       [--strict]
 *       [--report-json PATH]
 *       [--verbose]
 *
 * Exit codes:
 *   0  all commands replayed successfully (status=pass)
 *   1  one or more commands failed, strict-mode violation, or emulator error
 *   2  JSON parse error / missing required fields / bad arguments
 */

#define _POSIX_C_SOURCE 200112L   /* for unlink() in cleanup */

#include "att1_aimu_userspace.h"
#include "att1_aimu_mmio.h"
#include "att1_aimu_cmdq.h"
#include "att1_aimu_dma.h"
#include "att1_status.h"

#include <stdint.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* =========================================================================
 * Shared constants (mirrors att1-aimu-replay.c)
 * ====================================================================== */

#define MAX_COMMANDS      256
#define MAX_TILES          16   /* M121 limit */
#define JSON_LINE_MAX     512
#define STRING_MAX        128

/* Synthetic 64-byte-aligned host/device base addresses for mock DMA. */
#define HOST_BASE_ADDR    UINT64_C(0x0001000000000000)
#define DEV_BASE_ADDR     UINT64_C(0x0002000000000000)
#define ADDR_STRIDE       UINT64_C(0x0000000000010000)   /* 64 KiB / slot */

/* Mock DMA byte count used when the plan provides no payload size. */
#define MOCK_DMA_BYTES    UINT32_C(64)

/* =========================================================================
 * Plan types  (identical layout to att1-aimu-replay.c)
 * ====================================================================== */

typedef struct {
    int  version;
    char model_id[STRING_MAX];
    char session_id[STRING_MAX];
    int  tile_count;
    int  command_count;
    char status[32];
} plan_header_t;

typedef struct {
    long  command_id;
    char  command_type[STRING_MAX];
    long  tile_id;
    char  dtype[32];
    long  packed_bytes;
    long  total_bytes;
    long  fence_id;
    char  expected_status[STRING_MAX];
    long  quant_group_size;
    long  tensor_id;
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
    long   by_type[16];
    long   by_tile[MAX_TILES];
    int    status_pass;
    char   notes[512];
    /* MMIO-specific */
    uint32_t mmio_doorbell_count;
    uint32_t mmio_device_id;
    uint32_t mmio_reg_map_version;
    uint32_t mmio_tile_count;
} mmio_replay_report_t;

/* =========================================================================
 * Minimal JSON field helpers  (shared with att1-aimu-replay.c logic)
 * ====================================================================== */

static int jstr(const char *line, const char *key, char *buf, size_t bufsz)
{
    char        pat[STRING_MAX + 8];
    const char *p, *q;
    size_t      n;

    snprintf(pat, sizeof(pat), "\"%s\": \"", key);
    p = strstr(line, pat);
    if (!p) return -1;
    p += strlen(pat);
    q  = strchr(p, '"');
    if (!q) return -1;
    n = (size_t)(q - p);
    if (n >= bufsz) n = bufsz - 1u;
    memcpy(buf, p, n);
    buf[n] = '\0';
    return 0;
}

static int jint(const char *line, const char *key, long *val)
{
    char        pat[STRING_MAX + 8];
    const char *p;

    snprintf(pat, sizeof(pat), "\"%s\": ", key);
    p = strstr(line, pat);
    if (!p) return -1;
    p += strlen(pat);
    if (*p == 'n') return -1;   /* null */
    return (sscanf(p, "%ld", val) == 1) ? 0 : -1;
}

/* =========================================================================
 * Plan parser  (line-by-line state machine, same as att1-aimu-replay.c)
 * ====================================================================== */

typedef enum { PS_TOP, PS_HEADER, PS_COMMANDS, PS_COMMAND } parse_state_t;

static void reset_cmd(plan_cmd_t *c)
{
    memset(c, 0, sizeof(*c));
    c->tile_id          = -1;
    c->quant_group_size = -1;
    c->tensor_id        = -1;
}

static int parse_plan(const char *path, plan_t *plan)
{
    FILE          *fp;
    char           line[JSON_LINE_MAX];
    parse_state_t  state = PS_TOP;
    plan_cmd_t     cur;
    int            got_version = 0;
    long           v;

    memset(plan, 0, sizeof(*plan));
    plan->header.tile_count    = -1;
    plan->header.command_count = -1;
    reset_cmd(&cur);

    fp = fopen(path, "r");
    if (!fp) {
        fprintf(stderr, "mmio-replay: cannot open plan: %s\n", path);
        return -1;
    }

    while (fgets(line, (int)sizeof(line), fp)) {
        const char *t = line;
        while (*t == ' ' || *t == '\t') ++t;

        switch (state) {
        case PS_TOP:
            if (!got_version && jint(line, "command_plan_version", &v) == 0) {
                plan->header.version = (int)v;
                got_version = 1;
            }
            if (strstr(line, "\"header\": {"))   { state = PS_HEADER; }
            if (strstr(line, "\"commands\": [")) { state = PS_COMMANDS; }
            break;

        case PS_HEADER:
            if (*t == '}') { state = PS_TOP; break; }
            if (jint(line, "tile_count",    &v) == 0) plan->header.tile_count    = (int)v;
            if (jint(line, "command_count", &v) == 0) plan->header.command_count = (int)v;
            jstr(line, "model_id",   plan->header.model_id,   sizeof(plan->header.model_id));
            jstr(line, "session_id", plan->header.session_id, sizeof(plan->header.session_id));
            jstr(line, "status",     plan->header.status,     sizeof(plan->header.status));
            break;

        case PS_COMMANDS:
            if (*t == '{') { reset_cmd(&cur); state = PS_COMMAND; }
            if (*t == ']') { state = PS_TOP; }
            break;

        case PS_COMMAND:
            if (*t == '}') {
                if (plan->n_commands < MAX_COMMANDS)
                    plan->commands[plan->n_commands++] = cur;
                reset_cmd(&cur);
                state = PS_COMMANDS;
                break;
            }
            if (jint(line, "command_id",  &v) == 0) cur.command_id  = v;
            if (jint(line, "tile_id",     &v) == 0) cur.tile_id     = v;
            if (jint(line, "packed_bytes",&v) == 0) cur.packed_bytes = v;
            if (jint(line, "total_bytes", &v) == 0) cur.total_bytes  = v;
            if (jint(line, "fence_id",    &v) == 0) cur.fence_id     = v;
            if (jint(line, "quantization_group_size", &v) == 0) cur.quant_group_size = v;
            if (jint(line, "tensor_id",   &v) == 0) cur.tensor_id   = v;
            jstr(line, "command_type",    cur.command_type,    sizeof(cur.command_type));
            jstr(line, "dtype",           cur.dtype,           sizeof(cur.dtype));
            jstr(line, "expected_status", cur.expected_status, sizeof(cur.expected_status));
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
 * Command-type string → enum  (same table as att1-aimu-replay.c)
 * ====================================================================== */

typedef struct { const char *name; att1_aimu_cmd_type type; } cmd_map_t;

static const cmd_map_t CMD_MAP[] = {
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
    { NULL,              ATT1_AIMU_CMD_NOP               },
};

static int map_cmd_type(const char *name, att1_aimu_cmd_type *out)
{
    for (int i = 0; CMD_MAP[i].name; ++i) {
        if (strcmp(name, CMD_MAP[i].name) == 0) { *out = CMD_MAP[i].type; return 0; }
    }
    return -1;
}

static int map_expected_result(const char *name, att1_aimu_result *out)
{
    if (strcmp(name, "ATT1_AIMU_ERR_OK") == 0)              { *out = ATT1_AIMU_OK; return 0; }
    if (strcmp(name, "ATT1_AIMU_ERR_UNSUPPORTED_OP") == 0)  { *out = ATT1_AIMU_ERR_UNSUPPORTED_OP; return 0; }
    if (strcmp(name, "ATT1_AIMU_ERR_INVALID_COMMAND") == 0) { *out = ATT1_AIMU_ERR_INVALID_COMMAND; return 0; }
    *out = ATT1_AIMU_OK;
    return -1;
}

static uint8_t map_dtype(const char *dtype)
{
    if (strcmp(dtype, "q8") == 0) return ATT1_AIMU_DMA_DTYPE_Q8;
    if (strcmp(dtype, "q4") == 0) return ATT1_AIMU_DMA_DTYPE_Q4;
    return ATT1_AIMU_DMA_DTYPE_F32;
}

/* =========================================================================
 * Replay one command through the M121 userspace emulator
 * ====================================================================== */

static int mmio_replay_one(att1_aimu_userspace   *u,
                            const plan_cmd_t      *pc,
                            int                    strict,
                            int                    verbose,
                            mmio_replay_report_t  *rep)
{
    att1_aimu_cmd_type  ctype;
    att1_aimu_result    expected;
    att1_aimu_cmd       cmd;
    att1_aimu_completion comp;
    att1_status_t       rc;

    /* 1. Map command type */
    if (map_cmd_type(pc->command_type, &ctype) != 0) {
        fprintf(stderr, "mmio-replay: unknown command_type '%s' at id=%ld\n",
                pc->command_type, pc->command_id);
        rep->unsupported_commands++;
        if (strict) { rep->status_pass = 0; return -1; }
        return 0;
    }

    /* 2. Map expected result */
    map_expected_result(pc->expected_status, &expected);

    /* 3. Commands expected to produce UNSUPPORTED_OP: count, skip submission in strict */
    if (expected == ATT1_AIMU_ERR_UNSUPPORTED_OP) {
        rep->unsupported_commands++;
        if (strict) {
            fprintf(stderr,
                    "mmio-replay: strict — command %ld (%s) expects UNSUPPORTED_OP\n",
                    pc->command_id, pc->command_type);
            rep->status_pass = 0;
            return -1;
        }
        /* Non-strict: submit anyway to exercise emulator path */
    }

    /* 4. Validate tile_id */
    if (pc->tile_id < 0 || pc->tile_id >= (long)u->host->probe.tile_count) {
        fprintf(stderr,
                "mmio-replay: command %ld invalid tile_id=%ld (tile_count=%zu)\n",
                pc->command_id, pc->tile_id, u->host->probe.tile_count);
        rep->failed_commands++;
        rep->status_pass = 0;
        if (strict) return -1;
        return 0;
    }

    /* 5. DMA descriptor for LOAD_TENSOR_TILE (descriptor-only, no real buffer) */
    if (ctype == ATT1_AIMU_CMD_LOAD_TENSOR_TILE) {
        long      slot     = (pc->tensor_id >= 0) ? pc->tensor_id : pc->command_id;
        uint32_t  bytes    = (pc->packed_bytes > 0) ? (uint32_t)pc->packed_bytes : MOCK_DMA_BYTES;
        uint64_t  host_a   = HOST_BASE_ADDR + (uint64_t)slot * ADDR_STRIDE;
        uint64_t  dev_a    = DEV_BASE_ADDR  + (uint64_t)slot * ADDR_STRIDE;

        att1_aimu_dma_register_host_region(u->host->dma, host_a, (uint64_t)bytes + 128u);
        att1_aimu_dma_register_device_region(u->host->dma, dev_a, (uint64_t)bytes + 128u);

        att1_aimu_dma_desc dma;
        memset(&dma, 0, sizeof(dma));
        dma.host_addr      = host_a;
        dma.device_addr    = dev_a;
        dma.byte_length    = bytes;
        dma.descriptor_id  = (uint32_t)pc->command_id;
        dma.tensor_id      = (pc->tensor_id >= 0) ? (uint32_t)pc->tensor_id : 0u;
        dma.direction      = ATT1_AIMU_DMA_HOST_TO_DEVICE;
        dma.dtype          = map_dtype(pc->dtype);
        if (pc->quant_group_size == 32 || pc->quant_group_size == 64)
            dma.quant_group_size = (uint8_t)pc->quant_group_size;

        rc = att1_aimu_userspace_validate_dma(u, &dma);
        if (rc != ATT1_OK) {
            fprintf(stderr, "mmio-replay: DMA validation failed cmd=%ld rc=%d\n",
                    pc->command_id, (int)rc);
            rep->failed_commands++;
            rep->status_pass = 0;
            if (strict) return -1;
            return 0;
        }
        rep->dma_validations++;
    }

    /* 6. Build and submit command through emulator */
    memset(&cmd, 0, sizeof(cmd));
    cmd.command_type        = (uint8_t)ctype;
    cmd.tile_id             = (uint8_t)pc->tile_id;
    cmd.command_id          = (uint32_t)pc->command_id;
    cmd.dtype               = map_dtype(pc->dtype);
    cmd.fence_id            = (uint16_t)(pc->fence_id & 0xFFFF);
    cmd.completion_fence_id = (uint16_t)(pc->fence_id & 0xFFFF);
    cmd.tensor_id           = (pc->tensor_id >= 0) ? (uint16_t)pc->tensor_id : 0u;
    if (pc->packed_bytes > 0)
        cmd.input_buf_bytes = (uint32_t)pc->packed_bytes;

    rc = att1_aimu_userspace_submit_cmd(u, &cmd);
    if (rc != ATT1_OK) {
        fprintf(stderr, "mmio-replay: submit failed cmd=%ld rc=%d\n",
                pc->command_id, (int)rc);
        rep->failed_commands++;
        rep->status_pass = 0;
        if (strict) return -1;
        return 0;
    }

    if (verbose) {
        printf("  submitted cmd_id=%u type=%s tile=%d\n",
               cmd.command_id, pc->command_type, cmd.tile_id);
    }

    /* 7. Ring doorbell and process through emulator */
    rc = att1_aimu_userspace_ring_doorbell(u);
    if (rc != ATT1_OK) {
        fprintf(stderr, "mmio-replay: ring_doorbell failed cmd=%ld rc=%d\n",
                pc->command_id, (int)rc);
        rep->failed_commands++;
        rep->status_pass = 0;
        if (strict) return -1;
        return 0;
    }
    rep->doorbell_count++;

    /* Process one command: delegate through host process_one */
    rc = att1_aimu_host_process_one(u->host);
    if (rc != ATT1_OK) {
        fprintf(stderr, "mmio-replay: process_one failed cmd=%ld rc=%d\n",
                pc->command_id, (int)rc);
        rep->failed_commands++;
        rep->status_pass = 0;
        if (strict) return -1;
        return 0;
    }

    /* Flush BAR0 after dispatch */
    att1_aimu_userspace_flush_bar0(u);
    rep->commands_replayed++;

    /* 8. Read completion */
    memset(&comp, 0, sizeof(comp));
    rc = att1_aimu_host_read_completion(u->host, &comp);
    if (rc != ATT1_OK) {
        fprintf(stderr, "mmio-replay: read_completion failed cmd=%ld rc=%d\n",
                pc->command_id, (int)rc);
        rep->failed_commands++;
        rep->status_pass = 0;
        if (strict) return -1;
        return 0;
    }
    rep->completions_seen++;
    rep->trace_event_count += (long)comp.trace_event_count;
    if ((long)comp.fence_value > rep->fence_final)
        rep->fence_final = (long)comp.fence_value;

    /* 9. Check completion result vs expected */
    if ((att1_aimu_result)comp.result_code != expected) {
        /* Accept UNSUPPORTED_OP when plan expects OK (documented M112 mock behaviour) */
        if ((att1_aimu_result)comp.result_code == ATT1_AIMU_ERR_UNSUPPORTED_OP &&
            expected == ATT1_AIMU_OK) {
            rep->unsupported_commands++;
        } else {
            fprintf(stderr,
                    "mmio-replay: cmd %ld (%s) expected result %d got %d\n",
                    pc->command_id, pc->command_type,
                    (int)expected, (int)comp.result_code);
            rep->failed_commands++;
            rep->status_pass = 0;
            if (strict) return -1;
            return 0;
        }
    }

    /* 10. Per-type / per-tile accounting */
    {
        int tidx = (int)ctype & 0x0F;
        if (tidx >= 0 && tidx < 16) rep->by_type[tidx]++;
    }
    if (pc->tile_id >= 0 && pc->tile_id < MAX_TILES)
        rep->by_tile[pc->tile_id]++;

    return 0;
}

/* =========================================================================
 * Report emitters
 * ====================================================================== */

static void emit_text_report(const plan_t            *plan,
                              const mmio_replay_report_t *rep,
                              const char               *plan_path,
                              const char               *bar0_path)
{
    printf("att1-aimu-mmio-replay report (M122 — userspace emulator replay)\n");
    printf("  emulator         : userspace MMIO (M121); NOT real PCIe/hardware\n");
    printf("  plan_path        : %s\n", plan_path);
    if (bar0_path && bar0_path[0])
        printf("  bar0_file        : %s\n", bar0_path);
    printf("  model_id         : %s\n", plan->header.model_id);
    printf("  session_id       : %s\n", plan->header.session_id);
    printf("  tile_count       : %d\n", plan->header.tile_count);
    printf("  command_count    : %d\n", plan->n_commands);
    printf("  commands_replayed: %ld\n", rep->commands_replayed);
    printf("  completions_seen : %ld\n", rep->completions_seen);
    printf("  failed_commands  : %ld\n", rep->failed_commands);
    printf("  unsupported_cmds : %ld\n", rep->unsupported_commands);
    printf("  dma_validations  : %ld\n", rep->dma_validations);
    printf("  doorbell_count   : %ld (emulator)\n", rep->doorbell_count);
    printf("  mmio_doorbell    : %u (BAR0 register)\n", rep->mmio_doorbell_count);
    printf("  fence_final      : %ld\n", rep->fence_final);
    printf("  trace_events     : %ld\n", rep->trace_event_count);
    printf("  device_id        : 0x%08X\n", rep->mmio_device_id);
    printf("  reg_map_version  : 0x%08X\n", rep->mmio_reg_map_version);
    printf("  mmio_tile_count  : %u\n", rep->mmio_tile_count);
    printf("  status           : %s\n", rep->status_pass ? "pass" : "fail");
    if (rep->notes[0])
        printf("  notes            : %s\n", rep->notes);
}

static void emit_json_report(const plan_t            *plan,
                              const mmio_replay_report_t *rep,
                              const char               *plan_path,
                              const char               *bar0_path,
                              FILE                     *fp)
{
    fprintf(fp, "{\n");
    fprintf(fp, "  \"emulator\": \"att1-aimu-mmio-replay\",\n");
    fprintf(fp, "  \"milestone\": \"M122\",\n");
    fprintf(fp, "  \"plan_path\": \"%s\",\n",              plan_path);
    if (bar0_path && bar0_path[0])
        fprintf(fp, "  \"bar0_file\": \"%s\",\n",           bar0_path);
    else
        fprintf(fp, "  \"bar0_file\": null,\n");
    fprintf(fp, "  \"model_id\": \"%s\",\n",               plan->header.model_id);
    fprintf(fp, "  \"session_id\": \"%s\",\n",             plan->header.session_id);
    fprintf(fp, "  \"tile_count\": %d,\n",                 plan->header.tile_count);
    fprintf(fp, "  \"command_count\": %d,\n",              plan->n_commands);
    fprintf(fp, "  \"commands_replayed\": %ld,\n",         rep->commands_replayed);
    fprintf(fp, "  \"completions_seen\": %ld,\n",          rep->completions_seen);
    fprintf(fp, "  \"failed_commands\": %ld,\n",           rep->failed_commands);
    fprintf(fp, "  \"unsupported_commands\": %ld,\n",      rep->unsupported_commands);
    fprintf(fp, "  \"dma_validations\": %ld,\n",           rep->dma_validations);
    fprintf(fp, "  \"doorbell_count\": %ld,\n",            rep->doorbell_count);
    fprintf(fp, "  \"mmio_doorbell_count\": %u,\n",        rep->mmio_doorbell_count);
    fprintf(fp, "  \"fence_final\": %ld,\n",               rep->fence_final);
    fprintf(fp, "  \"trace_event_count\": %ld,\n",         rep->trace_event_count);
    fprintf(fp, "  \"device_id\": \"0x%08X\",\n",          rep->mmio_device_id);
    fprintf(fp, "  \"register_map_version\": \"0x%08X\",\n",rep->mmio_reg_map_version);
    fprintf(fp, "  \"mmio_tile_count\": %u,\n",            rep->mmio_tile_count);
    fprintf(fp, "  \"commands_by_type\": [");
    for (int i = 0; i < 16; i++)
        fprintf(fp, "%ld%s", rep->by_type[i], i < 15 ? ", " : "");
    fprintf(fp, "],\n");
    fprintf(fp, "  \"commands_by_tile\": {");
    {
        int first = 1;
        for (int i = 0; i < MAX_TILES; i++) {
            if (rep->by_tile[i] > 0) {
                if (!first) fprintf(fp, ", ");
                fprintf(fp, "\"%d\": %ld", i, rep->by_tile[i]);
                first = 0;
            }
        }
    }
    fprintf(fp, "},\n");
    fprintf(fp, "  \"status\": \"%s\",\n", rep->status_pass ? "pass" : "fail");
    fprintf(fp, "  \"notes\": \"%s\"\n",   rep->notes);
    fprintf(fp, "}\n");
}

/* =========================================================================
 * main
 * ====================================================================== */

int main(int argc, char *argv[])
{
    const char *plan_path     = NULL;
    const char *bar0_path     = NULL;
    const char *json_out      = NULL;
    int         strict        = 0;
    int         verbose       = 0;
    long        tiles         = 0;   /* 0 → use plan tile_count */
    long        tile_mem_mib  = 32;
    long        kv_mem_mib    = 8;

    /* ── Argument parsing ────────────────────────────────────────────── */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--plan") == 0 && i + 1 < argc) {
            plan_path = argv[++i];
        } else if (strcmp(argv[i], "--bar0-file") == 0 && i + 1 < argc) {
            bar0_path = argv[++i];
        } else if (strcmp(argv[i], "--tiles") == 0 && i + 1 < argc) {
            tiles = strtol(argv[++i], NULL, 10);
            if (tiles <= 0 || tiles > 16) {
                fprintf(stderr, "--tiles must be 1-16\n");
                return 2;
            }
        } else if (strcmp(argv[i], "--tile-memory-mib") == 0 && i + 1 < argc) {
            tile_mem_mib = strtol(argv[++i], NULL, 10);
            if (tile_mem_mib <= 0 || tile_mem_mib > 256) {
                fprintf(stderr, "--tile-memory-mib must be 1-256\n");
                return 2;
            }
        } else if (strcmp(argv[i], "--kv-memory-mib") == 0 && i + 1 < argc) {
            kv_mem_mib = strtol(argv[++i], NULL, 10);
            if (kv_mem_mib <= 0) {
                fprintf(stderr, "--kv-memory-mib must be positive\n");
                return 2;
            }
        } else if (strcmp(argv[i], "--strict") == 0) {
            strict = 1;
        } else if (strcmp(argv[i], "--report-json") == 0 && i + 1 < argc) {
            json_out = argv[++i];
        } else if (strcmp(argv[i], "--verbose") == 0) {
            verbose = 1;
        } else if (strcmp(argv[i], "--help") == 0 ||
                   strcmp(argv[i], "-h") == 0) {
            puts("Usage: att1-aimu-mmio-replay --plan PATH\n"
                 "    [--bar0-file PATH] [--tiles N] [--tile-memory-mib N]\n"
                 "    [--kv-memory-mib N] [--strict] [--report-json PATH] [--verbose]");
            return 0;
        } else {
            fprintf(stderr, "mmio-replay: unknown option: %s\n", argv[i]);
            return 2;
        }
    }

    if (!plan_path) {
        fputs("mmio-replay: --plan PATH is required\n", stderr);
        return 2;
    }

    /* ── Parse plan ──────────────────────────────────────────────────── */
    plan_t plan;
    if (parse_plan(plan_path, &plan) != 0) {
        fprintf(stderr, "mmio-replay: parse failed: %s\n", plan_path);
        return 2;
    }
    if (plan.header.version != 1) {
        fprintf(stderr, "mmio-replay: unsupported command_plan_version=%d\n",
                plan.header.version);
        return 2;
    }
    if (plan.n_commands == 0) {
        fputs("mmio-replay: plan contains no commands\n", stderr);
        return 2;
    }

    /* Strict: reject plans whose header status != "pass"/"ok" */
    if (strict && strcmp(plan.header.status, "pass") != 0 &&
                  strcmp(plan.header.status, "ok") != 0) {
        fprintf(stderr,
                "mmio-replay: strict — header status='%s' (expected pass/ok)\n",
                plan.header.status);
        return 1;
    }

    /* Determine tile count: command line overrides plan header */
    if (tiles == 0)
        tiles = (long)plan.header.tile_count;
    if (tiles <= 0 || tiles > 16) {
        fprintf(stderr, "mmio-replay: invalid tile_count=%ld\n", tiles);
        return 2;
    }

    if (verbose) {
        printf("mmio-replay: plan=%s tiles=%ld tile_mem=%ldMiB kv=%ldMiB\n",
               plan_path, tiles, tile_mem_mib, kv_mem_mib);
    }

    /* ── Open M121 userspace emulator ────────────────────────────────── */
    att1_aimu_userspace_config cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.tile_count        = (size_t)tiles;
    cfg.tile_memory_bytes = (uint64_t)tile_mem_mib * 1024u * 1024u;
    cfg.tile_kv_bytes     = (uint64_t)kv_mem_mib   * 1024u * 1024u;

    att1_aimu_userspace *u = NULL;
    att1_status_t rc = att1_aimu_userspace_open(&cfg, bar0_path, &u);
    if (rc != ATT1_OK) {
        fprintf(stderr, "mmio-replay: emulator open failed rc=%d\n", (int)rc);
        return 1;
    }

    /* ── Probe device ────────────────────────────────────────────────── */
    att1_aimu_host_probe_result probe;
    rc = att1_aimu_userspace_probe(u, &probe);
    if (rc != ATT1_OK) {
        fprintf(stderr, "mmio-replay: probe failed rc=%d\n", (int)rc);
        att1_aimu_userspace_close(u);
        return 1;
    }
    if (verbose)
        printf("mmio-replay: probed device_id=0x%08X tile_count=%zu\n",
               probe.device_id, probe.tile_count);

    /* ── Enumerate tiles ─────────────────────────────────────────────── */
    att1_aimu_host_tile_info tile_infos[16];
    size_t tile_count = 16;
    rc = att1_aimu_userspace_enumerate_tiles(u, tile_infos, &tile_count);
    if (rc != ATT1_OK) {
        fprintf(stderr, "mmio-replay: enumerate_tiles failed rc=%d\n", (int)rc);
        att1_aimu_userspace_close(u);
        return 1;
    }

    /* ── Verify plan tile count vs emulator tile count ───────────────── */
    if ((size_t)plan.header.tile_count > tile_count) {
        fprintf(stderr,
                "mmio-replay: plan tile_count=%d > emulator tile_count=%zu\n",
                plan.header.tile_count, tile_count);
        att1_aimu_userspace_close(u);
        return 1;
    }

    /* ── Verify register map version ─────────────────────────────────── */
    {
        uint32_t rmv = 0;
        (void)att1_aimu_userspace_read32(u, ATT1_MMIO_REGISTER_MAP_VERSION, &rmv);
        if (rmv != ATT1_AIMU_REGISTER_MAP_VERSION) {
            fprintf(stderr,
                    "mmio-replay: register_map_version mismatch: got 0x%08X want 0x%08X\n",
                    rmv, ATT1_AIMU_REGISTER_MAP_VERSION);
            att1_aimu_userspace_close(u);
            return 1;
        }
        if (verbose) printf("mmio-replay: register_map_version=0x%08X OK\n", rmv);
    }

    /* ── Setup command queue ─────────────────────────────────────────── */
    rc = att1_aimu_userspace_setup_cmdq(u);
    if (rc != ATT1_OK) {
        fprintf(stderr, "mmio-replay: setup_cmdq failed rc=%d\n", (int)rc);
        att1_aimu_userspace_close(u);
        return 1;
    }

    /* ── Replay loop ─────────────────────────────────────────────────── */
    mmio_replay_report_t rep;
    memset(&rep, 0, sizeof(rep));
    rep.status_pass = 1;

    for (int i = 0; i < plan.n_commands; i++) {
        if (mmio_replay_one(u, &plan.commands[i], strict, verbose, &rep) != 0)
            break;
    }

    /* ── Final drain ─────────────────────────────────────────────────── */
    (void)att1_aimu_userspace_drain(u);

    /* ── Snapshot counters ───────────────────────────────────────────── */
    (void)att1_aimu_userspace_snapshot(u);

    /* ── Read back MMIO register values for report ───────────────────── */
    (void)att1_aimu_userspace_read32(u, ATT1_MMIO_DEVICE_ID, &rep.mmio_device_id);
    (void)att1_aimu_userspace_read32(u, ATT1_MMIO_REGISTER_MAP_VERSION,
                                     &rep.mmio_reg_map_version);
    {
        uint32_t tc32 = 0;
        (void)att1_aimu_userspace_read32(u, ATT1_MMIO_TILE_COUNT, &tc32);
        rep.mmio_tile_count = tc32;
    }

    /* ── Get host summary for notes ──────────────────────────────────── */
    {
        att1_aimu_host_summary sum;
        memset(&sum, 0, sizeof(sum));
        if (att1_aimu_userspace_get_summary(u, &sum) == ATT1_OK) {
            rep.mmio_doorbell_count = sum.doorbell_count;
            snprintf(rep.notes, sizeof(rep.notes),
                     "commands_submitted=%" PRIu64 " commands_completed=%" PRIu64
                     " mmio_doorbell=%u",
                     sum.commands_submitted, sum.commands_completed,
                     sum.doorbell_count);
        }
    }

    /* ── Emit text report ────────────────────────────────────────────── */
    emit_text_report(&plan, &rep, plan_path, bar0_path);

    /* ── Emit JSON report (optional) ─────────────────────────────────── */
    if (json_out) {
        FILE *jfp = fopen(json_out, "w");
        if (!jfp) {
            fprintf(stderr, "mmio-replay: cannot open report-json: %s\n", json_out);
        } else {
            emit_json_report(&plan, &rep, plan_path, bar0_path, jfp);
            fclose(jfp);
        }
    }

    /* ── Cleanup ─────────────────────────────────────────────────────── */
    att1_aimu_userspace_close(u);

    return rep.status_pass ? 0 : 1;
}
