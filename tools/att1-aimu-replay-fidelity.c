#define _POSIX_C_SOURCE 200112L

#include "att1_aimu_cmdq.h"
#include "att1_aimu_conformance.h"
#include "att1_aimu_endpoint_client.h"
#include "att1_aimu_endpoint_protocol.h"
#include "att1_fabric.h"
#include "att1_status.h"

#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#define MAX_COMMANDS 512
#define MAX_ROUTES 512
#define MAX_TILES 16
#define MAX_DEST_TILES 16
#define STRING_MAX 128
#define JSON_LINE_MAX 1024
#define REPORT_VERSION 1
#define SYNTH_FLOATS 16u
#define SYNTH_FFN_FLOATS 8u
#define SYNTH_LOAD_MAX_BYTES 4096u

#define PASS(name) do { printf("PASS: aimu_replay_fidelity: %s\n", (name)); } while (0)
#define FAIL(name) do { printf("FAIL: aimu_replay_fidelity: %s\n", (name)); } while (0)

typedef struct {
    int version;
    char model_id[STRING_MAX];
    char session_id[STRING_MAX];
    int tile_count;
    int command_count;
    char status[32];
} plan_header_t;

typedef struct {
    long command_id;
    char command_type[STRING_MAX];
    long tile_id;
    char tensor_name[STRING_MAX];
    char dst_descriptor[STRING_MAX];
    char dtype[32];
    long tensor_id;
    long packed_bytes;
    long total_bytes;
    long fence_id;
    long dependency_fence_id;
    long quant_group_size;
    char expected_status[STRING_MAX];
} plan_cmd_t;

typedef struct {
    plan_header_t header;
    plan_cmd_t commands[MAX_COMMANDS];
    int n_commands;
    int parse_ok;
} plan_t;

typedef struct {
    long route_id;
    char route_type[STRING_MAX];
    long source_tile;
    long destination_tiles[MAX_DEST_TILES];
    int destination_count;
    long payload_bytes;
    long dependency_fence;
    long reduction_id;
    char reduction_behavior[STRING_MAX];
} route_t;

typedef struct {
    int version;
    int tile_count;
    int route_count;
    route_t routes[MAX_ROUTES];
    int n_routes;
    int parse_ok;
} route_report_t;

typedef struct {
    int used;
    uint16_t tensor_id;
    uint8_t tile_id;
    char tensor_name[STRING_MAX];
    uint8_t dtype;
    uint32_t bytes;
} loaded_tensor_t;

typedef struct {
    int created;
    uint64_t session_id;
    size_t next_append_position;
} kv_session_state_t;

typedef struct {
    long commands_replayed;
    long routes_replayed;
    long completions_compared;
    long payload_compares;
    long counter_compares;
    long trace_compares;
    long mismatches;
    int status_pass;
    char notes[512];
} fidelity_report_t;

typedef struct {
    att1_aimu_conformance_endpoint *inproc;
    att1_aimu_conformance_endpoint *socket;
    pid_t daemon_pid;
    char socket_path[256];
    loaded_tensor_t tensors[MAX_COMMANDS];
    size_t tensor_count;
    kv_session_state_t kv;
    uint32_t next_tensor_id;
    uint32_t participants[MAX_TILES];
    size_t participant_count;
    att1_aimu_conformance_config cfg;
} replay_ctx_t;

typedef enum {
    PS_TOP,
    PS_HEADER,
    PS_COMMANDS,
    PS_COMMAND,
    RS_TOP,
    RS_HEADER,
    RS_ROUTES,
    RS_ROUTE
} parse_state_t;

static int jstr(const char *line, const char *key, char *buf, size_t bufsz)
{
    char pat[STRING_MAX + 8];
    const char *p;
    const char *q;

    snprintf(pat, sizeof(pat), "\"%s\": \"", key);
    p = strstr(line, pat);
    if (p == NULL) {
        return -1;
    }
    p += strlen(pat);
    q = strchr(p, '\"');
    if (q == NULL) {
        return -1;
    }
    if ((size_t)(q - p) >= bufsz) {
        q = p + bufsz - 1u;
    }
    memcpy(buf, p, (size_t)(q - p));
    buf[q - p] = '\0';
    return 0;
}

static int jint(const char *line, const char *key, long *val)
{
    char pat[STRING_MAX + 8];
    const char *p;

    snprintf(pat, sizeof(pat), "\"%s\": ", key);
    p = strstr(line, pat);
    if (p == NULL) {
        return -1;
    }
    p += strlen(pat);
    if (*p == 'n') {
        return -1;
    }
    return (sscanf(p, "%ld", val) == 1) ? 0 : -1;
}

static void reset_plan_cmd(plan_cmd_t *cmd)
{
    memset(cmd, 0, sizeof(*cmd));
    cmd->tile_id = -1;
    cmd->tensor_id = -1;
    cmd->packed_bytes = 0;
    cmd->total_bytes = 0;
    cmd->fence_id = 0;
    cmd->dependency_fence_id = 0;
    cmd->quant_group_size = -1;
}

static void reset_route(route_t *route)
{
    memset(route, 0, sizeof(*route));
    route->source_tile = -1;
    route->payload_bytes = 0;
    route->dependency_fence = 0;
    route->reduction_id = 0;
}

static int parse_int_array_line(const char *line,
                                const char *key,
                                long *values,
                                int max_values,
                                int *out_count)
{
    char pat[STRING_MAX + 8];
    const char *p;
    int count = 0;

    snprintf(pat, sizeof(pat), "\"%s\": [", key);
    p = strstr(line, pat);
    if (p == NULL) {
        return -1;
    }
    p += strlen(pat);
    while ((*p != '\0') && (*p != ']')) {
        long v;
        while ((*p == ' ') || (*p == ',')) {
            ++p;
        }
        if (*p == ']') {
            break;
        }
        if ((sscanf(p, "%ld", &v) != 1) || (count >= max_values)) {
            return -1;
        }
        values[count++] = v;
        while ((*p != '\0') && (*p != ',') && (*p != ']')) {
            ++p;
        }
    }
    *out_count = count;
    return 0;
}

static int parse_plan(const char *path, plan_t *plan)
{
    FILE *fp;
    char line[JSON_LINE_MAX];
    parse_state_t state = PS_TOP;
    plan_cmd_t cur;
    long v;

    memset(plan, 0, sizeof(*plan));
    plan->header.tile_count = -1;
    plan->header.command_count = -1;
    reset_plan_cmd(&cur);

    fp = fopen(path, "r");
    if (fp == NULL) {
        fprintf(stderr, "replay-fidelity: cannot open plan file: %s\n", path);
        return -1;
    }

    while (fgets(line, (int)sizeof(line), fp) != NULL) {
        const char *t = line;
        while ((*t == ' ') || (*t == '\t')) {
            ++t;
        }
        switch (state) {
        case PS_TOP:
            if (jint(line, "command_plan_version", &v) == 0) {
                plan->header.version = (int)v;
            }
            if (strstr(line, "\"header\": {") != NULL) {
                state = PS_HEADER;
            } else if (strstr(line, "\"commands\": [") != NULL) {
                state = PS_COMMANDS;
            }
            break;
        case PS_HEADER:
            if (*t == '}') {
                state = PS_TOP;
                break;
            }
            if (jint(line, "tile_count", &v) == 0) {
                plan->header.tile_count = (int)v;
            }
            if (jint(line, "command_count", &v) == 0) {
                plan->header.command_count = (int)v;
            }
            jstr(line, "model_id", plan->header.model_id, sizeof(plan->header.model_id));
            jstr(line, "session_id", plan->header.session_id, sizeof(plan->header.session_id));
            jstr(line, "status", plan->header.status, sizeof(plan->header.status));
            break;
        case PS_COMMANDS:
            if (*t == '{') {
                reset_plan_cmd(&cur);
                state = PS_COMMAND;
            } else if (*t == ']') {
                state = PS_TOP;
            }
            break;
        case PS_COMMAND:
            if (*t == '}') {
                if (plan->n_commands < MAX_COMMANDS) {
                    plan->commands[plan->n_commands++] = cur;
                }
                reset_plan_cmd(&cur);
                state = PS_COMMANDS;
                break;
            }
            if (jint(line, "command_id", &v) == 0) { cur.command_id = v; }
            if (jint(line, "tile_id", &v) == 0) { cur.tile_id = v; }
            if (jint(line, "tensor_id", &v) == 0) { cur.tensor_id = v; }
            if (jint(line, "packed_bytes", &v) == 0) { cur.packed_bytes = v; }
            if (jint(line, "total_bytes", &v) == 0) { cur.total_bytes = v; }
            if (jint(line, "fence_id", &v) == 0) { cur.fence_id = v; }
            if (jint(line, "dependency_fence_id", &v) == 0) { cur.dependency_fence_id = v; }
            if (jint(line, "quantization_group_size", &v) == 0) { cur.quant_group_size = v; }
            jstr(line, "command_type", cur.command_type, sizeof(cur.command_type));
            jstr(line, "tensor_name", cur.tensor_name, sizeof(cur.tensor_name));
            jstr(line, "dst_descriptor", cur.dst_descriptor, sizeof(cur.dst_descriptor));
            jstr(line, "dtype", cur.dtype, sizeof(cur.dtype));
            jstr(line, "expected_status", cur.expected_status, sizeof(cur.expected_status));
            break;
        default:
            break;
        }
    }

    fclose(fp);
    plan->parse_ok = (plan->header.version == 1) &&
                     (plan->header.tile_count > 0) &&
                     (plan->header.tile_count <= MAX_TILES) &&
                     (plan->header.command_count >= 0);
    return plan->parse_ok ? 0 : -1;
}

static int parse_routes(const char *path, route_report_t *report)
{
    FILE *fp;
    char line[JSON_LINE_MAX];
    parse_state_t state = RS_TOP;
    route_t cur;
    long v;

    memset(report, 0, sizeof(*report));
    report->tile_count = -1;
    reset_route(&cur);

    fp = fopen(path, "r");
    if (fp == NULL) {
        fprintf(stderr, "replay-fidelity: cannot open route file: %s\n", path);
        return -1;
    }

    while (fgets(line, (int)sizeof(line), fp) != NULL) {
        const char *t = line;
        while ((*t == ' ') || (*t == '\t')) {
            ++t;
        }
        switch (state) {
        case RS_TOP:
            if (jint(line, "route_report_version", &v) == 0) {
                report->version = (int)v;
            }
            if (strstr(line, "\"header\": {") != NULL) {
                state = RS_HEADER;
            } else if (strstr(line, "\"routes\": [") != NULL) {
                state = RS_ROUTES;
            }
            break;
        case RS_HEADER:
            if (*t == '}') {
                state = RS_TOP;
                break;
            }
            if (jint(line, "route_report_version", &v) == 0) {
                report->version = (int)v;
            }
            if (jint(line, "tile_count", &v) == 0) {
                report->tile_count = (int)v;
            }
            if (jint(line, "route_count", &v) == 0) {
                report->route_count = (int)v;
            }
            break;
        case RS_ROUTES:
            if (*t == '{') {
                reset_route(&cur);
                state = RS_ROUTE;
            } else if (*t == ']') {
                state = RS_TOP;
            }
            break;
        case RS_ROUTE:
            if (*t == '}') {
                if (report->n_routes < MAX_ROUTES) {
                    report->routes[report->n_routes++] = cur;
                }
                reset_route(&cur);
                state = RS_ROUTES;
                break;
            }
            if (jint(line, "route_id", &v) == 0) { cur.route_id = v; }
            if (jint(line, "source_tile", &v) == 0) { cur.source_tile = v; }
            if (jint(line, "payload_bytes", &v) == 0) { cur.payload_bytes = v; }
            if (jint(line, "dependency_fence", &v) == 0) { cur.dependency_fence = v; }
            if (jint(line, "reduction_id", &v) == 0) { cur.reduction_id = v; }
            jstr(line, "route_type", cur.route_type, sizeof(cur.route_type));
            jstr(line, "reduction_behavior", cur.reduction_behavior, sizeof(cur.reduction_behavior));
            (void)parse_int_array_line(line,
                                       "destination_tiles",
                                       cur.destination_tiles,
                                       MAX_DEST_TILES,
                                       &cur.destination_count);
            break;
        default:
            break;
        }
    }

    fclose(fp);
    report->parse_ok = (report->version == 1) &&
                       (report->tile_count > 0) &&
                       (report->tile_count <= MAX_TILES);
    return report->parse_ok ? 0 : -1;
}

static uint8_t map_dtype(const char *dtype)
{
    if (strcmp(dtype, "q8") == 0) {
        return ATT1_AIMU_DMA_DTYPE_Q8;
    }
    if (strcmp(dtype, "q4") == 0) {
        return ATT1_AIMU_DMA_DTYPE_Q4;
    }
    return ATT1_AIMU_DMA_DTYPE_F32;
}

static int map_cmd_type(const char *name, att1_aimu_cmd_type *out)
{
    if (strcmp(name, "NOP") == 0) { *out = ATT1_AIMU_CMD_NOP; return 0; }
    if (strcmp(name, "LOAD_TENSOR_TILE") == 0) { *out = ATT1_AIMU_CMD_LOAD_TENSOR_TILE; return 0; }
    if (strcmp(name, "VALIDATE_TENSOR") == 0) { *out = ATT1_AIMU_CMD_VALIDATE_TENSOR; return 0; }
    if (strcmp(name, "EXEC_MATMUL") == 0) { *out = ATT1_AIMU_CMD_EXEC_MATMUL; return 0; }
    if (strcmp(name, "EXEC_RMSNORM") == 0) { *out = ATT1_AIMU_CMD_EXEC_RMSNORM; return 0; }
    if (strcmp(name, "EXEC_ROPE") == 0) { *out = ATT1_AIMU_CMD_EXEC_ROPE; return 0; }
    if (strcmp(name, "EXEC_ATTENTION") == 0) { *out = ATT1_AIMU_CMD_EXEC_ATTENTION; return 0; }
    if (strcmp(name, "EXEC_FFN") == 0) { *out = ATT1_AIMU_CMD_EXEC_FFN; return 0; }
    if (strcmp(name, "KV_APPEND") == 0) { *out = ATT1_AIMU_CMD_KV_APPEND; return 0; }
    if (strcmp(name, "KV_READ") == 0) { *out = ATT1_AIMU_CMD_KV_READ; return 0; }
    if (strcmp(name, "FABRIC_SEND") == 0) { *out = ATT1_AIMU_CMD_FABRIC_SEND; return 0; }
    if (strcmp(name, "FABRIC_REDUCE") == 0) { *out = ATT1_AIMU_CMD_FABRIC_REDUCE; return 0; }
    if (strcmp(name, "TRACE_SNAPSHOT") == 0) { *out = ATT1_AIMU_CMD_TRACE_SNAPSHOT; return 0; }
    if (strcmp(name, "TILE_BARRIER") == 0) { *out = ATT1_AIMU_CMD_TILE_BARRIER; return 0; }
    if (strcmp(name, "RESET_TILE") == 0) { *out = ATT1_AIMU_CMD_RESET_TILE; return 0; }
    if (strcmp(name, "QUERY_COUNTERS") == 0) { *out = ATT1_AIMU_CMD_QUERY_COUNTERS; return 0; }
    return -1;
}

static int map_expected_result(const char *name, att1_aimu_result *out)
{
    struct entry {
        const char *name;
        att1_aimu_result result;
    } entries[] = {
        { "ATT1_AIMU_ERR_OK", ATT1_AIMU_OK },
        { "ATT1_AIMU_OK", ATT1_AIMU_OK },
        { "ATT1_AIMU_ERR_UNSUPPORTED_OP", ATT1_AIMU_ERR_UNSUPPORTED_OP },
        { "ATT1_AIMU_ERR_INVALID_COMMAND", ATT1_AIMU_ERR_INVALID_COMMAND },
        { NULL, ATT1_AIMU_OK }
    };
    int i;

    for (i = 0; entries[i].name != NULL; i++) {
        if (strcmp(name, entries[i].name) == 0) {
            *out = entries[i].result;
            return 0;
        }
    }
    *out = ATT1_AIMU_OK;
    return -1;
}

static uint64_t fnv1a_seed(uint32_t a, uint32_t b)
{
    uint64_t seed = UINT64_C(1469598103934665603);
    seed ^= a;
    seed *= UINT64_C(1099511628211);
    seed ^= b;
    seed *= UINT64_C(1099511628211);
    return seed;
}

static uint32_t prng_next(uint64_t *state)
{
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    return (uint32_t)(*state & UINT32_C(0xFFFFFFFF));
}

static void fill_bytes(unsigned char *buf, size_t bytes, uint32_t a, uint32_t b)
{
    size_t i;
    uint64_t state = fnv1a_seed(a, b);

    for (i = 0u; i < bytes; i++) {
        buf[i] = (unsigned char)(prng_next(&state) & 0xFFu);
    }
}

static void fill_floats(float *buf, size_t count, uint32_t a, uint32_t b)
{
    size_t i;
    uint64_t state = fnv1a_seed(a, b);

    for (i = 0u; i < count; i++) {
        uint32_t v = prng_next(&state) % 1000u;
        buf[i] = ((float)v / 257.0f) - 1.5f;
    }
}

static uint64_t parse_session_id(const char *session_id)
{
    const char *p = session_id;
    while ((*p != '\0') && ((*p < '0') || (*p > '9'))) {
        ++p;
    }
    return (*p == '\0') ? 0u : (uint64_t)strtoull(p, NULL, 10);
}

static loaded_tensor_t *find_tensor(replay_ctx_t *ctx, const plan_cmd_t *pc)
{
    size_t i;
    for (i = 0u; i < ctx->tensor_count; i++) {
        loaded_tensor_t *t = &ctx->tensors[i];
        if (!t->used) {
            continue;
        }
        if ((pc->tensor_id >= 0) && (t->tensor_id == (uint16_t)pc->tensor_id)) {
            return t;
        }
        if ((pc->tensor_name[0] != '\0') && (strcmp(t->tensor_name, pc->tensor_name) == 0)) {
            return t;
        }
    }
    return NULL;
}

static loaded_tensor_t *remember_tensor(replay_ctx_t *ctx, const plan_cmd_t *pc, uint32_t bytes)
{
    loaded_tensor_t *t = find_tensor(ctx, pc);
    if (t != NULL) {
        t->bytes = bytes;
        t->dtype = map_dtype(pc->dtype);
        t->tile_id = (uint8_t)pc->tile_id;
        return t;
    }
    if (ctx->tensor_count >= MAX_COMMANDS) {
        return NULL;
    }
    t = &ctx->tensors[ctx->tensor_count++];
    memset(t, 0, sizeof(*t));
    t->used = 1;
    t->tensor_id = (pc->tensor_id >= 0) ? (uint16_t)pc->tensor_id : (uint16_t)ctx->next_tensor_id++;
    t->tile_id = (uint8_t)pc->tile_id;
    t->dtype = map_dtype(pc->dtype);
    t->bytes = bytes;
    if (pc->tensor_name[0] != '\0') {
        strncpy(t->tensor_name, pc->tensor_name, sizeof(t->tensor_name) - 1u);
    }
    return t;
}

static size_t clamp_load_bytes(const plan_cmd_t *pc)
{
    size_t bytes = 0u;
    if (pc->packed_bytes > 0) {
        bytes = (size_t)pc->packed_bytes;
    } else if (pc->total_bytes > 0) {
        bytes = (size_t)pc->total_bytes;
    }
    if (bytes == 0u) {
        bytes = SYNTH_FLOATS * sizeof(float);
    }
    if (bytes > SYNTH_LOAD_MAX_BYTES) {
        bytes = SYNTH_LOAD_MAX_BYTES;
    }
    if (map_dtype(pc->dtype) == ATT1_AIMU_DMA_DTYPE_F32) {
        bytes &= ~((size_t)sizeof(float) - 1u);
        if (bytes == 0u) {
            bytes = sizeof(float);
        }
    }
    return bytes;
}

static void encode_dims(uint32_t element_count, uint16_t *dim0, uint16_t *dim1)
{
    uint32_t d1 = (element_count > 65535u) ? 65535u : element_count;
    uint32_t d0 = (element_count + d1 - 1u) / d1;
    if (d0 == 0u) {
        d0 = 1u;
    }
    if (d0 > 65535u) {
        d0 = 65535u;
    }
    *dim0 = (uint16_t)d0;
    *dim1 = (uint16_t)d1;
}

static int compare_bytes(const void *lhs,
                         const void *rhs,
                         size_t bytes,
                         const char *label,
                         fidelity_report_t *report)
{
    if (memcmp(lhs, rhs, bytes) != 0) {
        snprintf(report->notes, sizeof(report->notes), "%s mismatch", label);
        report->mismatches++;
        report->status_pass = 0;
        return -1;
    }
    return 0;
}

static int compare_completion(const att1_aimu_completion *a,
                              const att1_aimu_completion *b,
                              fidelity_report_t *report,
                              const plan_cmd_t *pc)
{
    if (compare_bytes(a, b, sizeof(*a), "completion", report) != 0) {
        fprintf(stderr,
                "replay-fidelity: completion mismatch for command %ld (%s)\n",
                pc->command_id,
                pc->command_type);
        return -1;
    }
    report->completions_compared++;
    return 0;
}

static int compare_fabric_receive(const att1_fabric_packet *a_pkt,
                                  const unsigned char *a_payload,
                                  size_t a_bytes,
                                  const att1_fabric_packet *b_pkt,
                                  const unsigned char *b_payload,
                                  size_t b_bytes,
                                  fidelity_report_t *report,
                                  const char *label)
{
    if ((a_bytes != b_bytes) ||
        (memcmp(a_pkt, b_pkt, sizeof(*a_pkt)) != 0) ||
        (memcmp(a_payload, b_payload, a_bytes) != 0)) {
        snprintf(report->notes, sizeof(report->notes), "%s mismatch", label);
        report->mismatches++;
        report->status_pass = 0;
        return -1;
    }
    report->payload_compares++;
    return 0;
}

static pid_t spawn_daemon(const char *socket_path, size_t tile_count, size_t cmd_ring, size_t comp_ring)
{
    pid_t pid = fork();
    char tiles_arg[32];
    char cmd_arg[32];
    char comp_arg[32];

    if (pid < 0) {
        return -1;
    }
    if (pid == 0) {
        snprintf(tiles_arg, sizeof(tiles_arg), "%zu", tile_count);
        snprintf(cmd_arg, sizeof(cmd_arg), "%zu", cmd_ring);
        snprintf(comp_arg, sizeof(comp_arg), "%zu", comp_ring);
        execl("./build/att1-aimu-endpoint",
              "att1-aimu-endpoint",
              "--socket",
              socket_path,
              "--tiles",
              tiles_arg,
              "--cmd-ring-depth",
              cmd_arg,
              "--comp-ring-depth",
              comp_arg,
              "--once",
              (char *)NULL);
        _exit(127);
    }
    return pid;
}

static int connect_with_retry(const char *socket_path, att1_aimu_conformance_endpoint **out)
{
    int attempt;
    for (attempt = 0; attempt < 100; ++attempt) {
        if (att1_aimu_conformance_socket_connect(socket_path, out) == ATT1_OK) {
            return 0;
        }
        {
            struct timespec ts = {0, 10 * 1000 * 1000L};
            nanosleep(&ts, NULL);
        }
    }
    return -1;
}

static void replay_ctx_cleanup(replay_ctx_t *ctx)
{
    int status;

    if (ctx->inproc != NULL) {
        att1_aimu_conformance_endpoint_destroy(ctx->inproc);
        ctx->inproc = NULL;
    }
    if (ctx->socket != NULL) {
        att1_aimu_conformance_endpoint_destroy(ctx->socket);
        ctx->socket = NULL;
    }
    if (ctx->daemon_pid > 0) {
        if (waitpid(ctx->daemon_pid, &status, WNOHANG) == 0) {
            kill(ctx->daemon_pid, SIGTERM);
            waitpid(ctx->daemon_pid, &status, 0);
        }
        ctx->daemon_pid = 0;
    }
    if (ctx->socket_path[0] != '\0') {
        unlink(ctx->socket_path);
        ctx->socket_path[0] = '\0';
    }
}

static int replay_ctx_init(replay_ctx_t *ctx, const plan_t *plan)
{
    size_t i;

    memset(ctx, 0, sizeof(*ctx));
    att1_aimu_conformance_default_config(&ctx->cfg);
    ctx->cfg.tile_count = (size_t)plan->header.tile_count;
    ctx->participant_count = (size_t)plan->header.tile_count;
    for (i = 0u; i < ctx->participant_count; i++) {
        ctx->participants[i] = (uint32_t)i;
    }
    ctx->next_tensor_id = 1u;
    ctx->kv.session_id = parse_session_id(plan->header.session_id);
    snprintf(ctx->socket_path,
             sizeof(ctx->socket_path),
             "build/att1-aimu-replay-fidelity-%ld.sock",
             (long)getpid());
    unlink(ctx->socket_path);
    if (att1_aimu_conformance_inproc_create(&ctx->cfg, &ctx->inproc) != ATT1_OK) {
        return -1;
    }
    ctx->daemon_pid = spawn_daemon(ctx->socket_path,
                                   ctx->cfg.tile_count,
                                   ctx->cfg.cmd_ring_depth,
                                   ctx->cfg.comp_ring_depth);
    if (ctx->daemon_pid <= 0) {
        replay_ctx_cleanup(ctx);
        return -1;
    }
    if (connect_with_retry(ctx->socket_path, &ctx->socket) != 0) {
        replay_ctx_cleanup(ctx);
        return -1;
    }
    return 0;
}

static int submit_and_compare(replay_ctx_t *ctx,
                              att1_aimu_cmd *cmd,
                              att1_aimu_result expected,
                              fidelity_report_t *report,
                              const plan_cmd_t *pc)
{
    att1_status_t st;
    att1_aimu_completion inproc_comp;
    att1_aimu_completion socket_comp;

    st = att1_aimu_conformance_cmd_submit(ctx->inproc, cmd);
    if (st != ATT1_OK) {
        return -1;
    }
    st = att1_aimu_conformance_cmd_submit(ctx->socket, cmd);
    if (st != ATT1_OK) {
        return -1;
    }
    st = att1_aimu_conformance_cmd_dispatch_all(ctx->inproc);
    if (st != ATT1_OK) {
        return -1;
    }
    st = att1_aimu_conformance_cmd_dispatch_all(ctx->socket);
    if (st != ATT1_OK) {
        return -1;
    }
    memset(&inproc_comp, 0, sizeof(inproc_comp));
    memset(&socket_comp, 0, sizeof(socket_comp));
    st = att1_aimu_conformance_cmd_poll_completion(ctx->inproc, &inproc_comp);
    if (st != ATT1_OK) {
        return -1;
    }
    st = att1_aimu_conformance_cmd_poll_completion(ctx->socket, &socket_comp);
    if (st != ATT1_OK) {
        return -1;
    }
    if (compare_completion(&inproc_comp, &socket_comp, report, pc) != 0) {
        return -1;
    }
    if ((att1_aimu_result)inproc_comp.result_code != expected) {
        fprintf(stderr,
                "replay-fidelity: command %ld (%s) expected %d got %d\n",
                pc->command_id,
                pc->command_type,
                (int)expected,
                (int)inproc_comp.result_code);
        return -1;
    }
    report->commands_replayed++;
    return 0;
}

static int ensure_kv_session(replay_ctx_t *ctx)
{
    att1_status_t st;
    if (ctx->kv.created) {
        return 0;
    }
    st = att1_aimu_conformance_kv_create_session(ctx->inproc, ctx->kv.session_id);
    if (st != ATT1_OK) {
        return -1;
    }
    st = att1_aimu_conformance_kv_create_session(ctx->socket, ctx->kv.session_id);
    if (st != ATT1_OK) {
        return -1;
    }
    ctx->kv.created = 1;
    return 0;
}

static int handle_load_tensor(replay_ctx_t *ctx,
                              const plan_cmd_t *pc,
                              att1_aimu_result expected,
                              fidelity_report_t *report)
{
    loaded_tensor_t *tensor;
    att1_aimu_cmd cmd;
    unsigned char *host_buf;
    size_t bytes;
    uint16_t dim0;
    uint16_t dim1;
    uint32_t elem_count;

    bytes = clamp_load_bytes(pc);
    tensor = remember_tensor(ctx, pc, (uint32_t)bytes);
    if (tensor == NULL) {
        return -1;
    }
    host_buf = (unsigned char *)malloc(bytes);
    if (host_buf == NULL) {
        return -1;
    }
    fill_bytes(host_buf, bytes, (uint32_t)pc->command_id, tensor->tensor_id);
    memset(&cmd, 0, sizeof(cmd));
    cmd.command_type = ATT1_AIMU_CMD_LOAD_TENSOR_TILE;
    cmd.tile_id = (uint8_t)pc->tile_id;
    cmd.session_id = (uint8_t)ctx->kv.session_id;
    cmd.dtype = tensor->dtype;
    cmd.tensor_id = tensor->tensor_id;
    cmd.input_buf_addr = (uint64_t)(uintptr_t)host_buf;
    cmd.input_buf_bytes = (uint32_t)bytes;
    elem_count = (tensor->dtype == ATT1_AIMU_DMA_DTYPE_F32) ?
                 (uint32_t)(bytes / sizeof(float)) : (uint32_t)bytes;
    encode_dims(elem_count, &dim0, &dim1);
    cmd.op_param_1 = ((uint32_t)dim0 << 16) | (uint32_t)dim1;
    if (submit_and_compare(ctx, &cmd, expected, report, pc) != 0) {
        free(host_buf);
        return -1;
    }
    free(host_buf);
    return 0;
}

static int handle_validate_tensor(replay_ctx_t *ctx,
                                  const plan_cmd_t *pc,
                                  att1_aimu_result expected)
{
    loaded_tensor_t *tensor = find_tensor(ctx, pc);
    if ((expected != ATT1_AIMU_OK) || (tensor == NULL)) {
        return -1;
    }
    if (att1_aimu_conformance_sync_mmio(ctx->inproc) != ATT1_OK) {
        return -1;
    }
    if (att1_aimu_conformance_sync_mmio(ctx->socket) != ATT1_OK) {
        return -1;
    }
    return 0;
}

static int handle_exec(replay_ctx_t *ctx,
                       const plan_cmd_t *pc,
                       att1_aimu_cmd_type ctype,
                       att1_aimu_result expected,
                       fidelity_report_t *report)
{
    att1_aimu_cmd cmd;
    loaded_tensor_t *tensor = find_tensor(ctx, pc);
    float *input;
    float *output;
    size_t input_floats = SYNTH_FLOATS;
    size_t output_floats = SYNTH_FLOATS;
    size_t count;
    union { uint32_t u; float f; } bits;

    if (ctype == ATT1_AIMU_CMD_EXEC_FFN) {
        input_floats = SYNTH_FFN_FLOATS * 2u;
        output_floats = SYNTH_FFN_FLOATS;
    } else if (ctype == ATT1_AIMU_CMD_EXEC_MATMUL) {
        input_floats = SYNTH_FLOATS;
        output_floats = 1u;
    }
    input = (float *)malloc(input_floats * sizeof(float));
    output = (float *)calloc(output_floats, sizeof(float));
    if ((input == NULL) || (output == NULL)) {
        free(input);
        free(output);
        return -1;
    }
    fill_floats(input, input_floats, (uint32_t)pc->command_id, (uint32_t)ctype);
    memset(&cmd, 0, sizeof(cmd));
    cmd.command_type = (uint8_t)ctype;
    cmd.tile_id = (uint8_t)pc->tile_id;
    cmd.session_id = (uint8_t)ctx->kv.session_id;
    cmd.dtype = map_dtype(pc->dtype);
    if ((tensor != NULL) && (ctype != ATT1_AIMU_CMD_EXEC_ROPE) && (ctype != ATT1_AIMU_CMD_EXEC_FFN)) {
        cmd.tensor_id = tensor->tensor_id;
    }
    cmd.input_buf_addr = (uint64_t)(uintptr_t)input;
    cmd.input_buf_bytes = (uint32_t)(input_floats * sizeof(float));
    cmd.output_buf_addr = (uint64_t)(uintptr_t)output;
    cmd.output_buf_bytes = (uint32_t)(output_floats * sizeof(float));

    switch (ctype) {
    case ATT1_AIMU_CMD_EXEC_MATMUL:
        cmd.op_param_0 = 1u;
        break;
    case ATT1_AIMU_CMD_EXEC_RMSNORM:
        bits.f = 1.0e-5f;
        cmd.op_param_0 = bits.u;
        break;
    case ATT1_AIMU_CMD_EXEC_ROPE:
        bits.f = 10000.0f;
        cmd.output_buf_addr = cmd.input_buf_addr;
        cmd.output_buf_bytes = cmd.input_buf_bytes;
        cmd.op_param_1 = bits.u;
        cmd.kv_position = 1u;
        break;
    case ATT1_AIMU_CMD_EXEC_FFN:
        count = input_floats / 2u;
        cmd.op_param_0 = (uint32_t)count;
        break;
    default:
        break;
    }

    if (submit_and_compare(ctx, &cmd, expected, report, pc) != 0) {
        free(input);
        free(output);
        return -1;
    }
    free(input);
    free(output);
    return 0;
}

static int handle_query_counters(replay_ctx_t *ctx,
                                 att1_aimu_result expected,
                                 fidelity_report_t *report)
{
    att1_aimu_cmdq_counters a;
    att1_aimu_cmdq_counters b;
    att1_aimu_dma_counters da;
    att1_aimu_dma_counters db;
    att1_fabric_counters fa;
    att1_fabric_counters fb;
    att1_kv_mmu_counters ka;
    att1_kv_mmu_counters kb;

    if (expected != ATT1_AIMU_OK) {
        return -1;
    }
    if ((att1_aimu_conformance_cmd_get_counters(ctx->inproc, &a) != ATT1_OK) ||
        (att1_aimu_conformance_cmd_get_counters(ctx->socket, &b) != ATT1_OK) ||
        (att1_aimu_conformance_dma_get_counters(ctx->inproc, &da) != ATT1_OK) ||
        (att1_aimu_conformance_dma_get_counters(ctx->socket, &db) != ATT1_OK) ||
        (att1_aimu_conformance_fabric_get_counters(ctx->inproc, &fa) != ATT1_OK) ||
        (att1_aimu_conformance_fabric_get_counters(ctx->socket, &fb) != ATT1_OK) ||
        (att1_aimu_conformance_kv_get_counters(ctx->inproc, &ka) != ATT1_OK) ||
        (att1_aimu_conformance_kv_get_counters(ctx->socket, &kb) != ATT1_OK)) {
        return -1;
    }
    if ((compare_bytes(&a, &b, sizeof(a), "cmd counters", report) != 0) ||
        (compare_bytes(&da, &db, sizeof(da), "dma counters", report) != 0) ||
        (compare_bytes(&fa, &fb, sizeof(fa), "fabric counters", report) != 0) ||
        (compare_bytes(&ka, &kb, sizeof(ka), "kv counters", report) != 0)) {
        return -1;
    }
    report->counter_compares += 4;
    return 0;
}

static int handle_trace_snapshot(replay_ctx_t *ctx,
                                 att1_aimu_result expected,
                                 fidelity_report_t *report)
{
    att1_aimu_trace_snapshot a;
    att1_aimu_trace_snapshot b;

    if (expected != ATT1_AIMU_OK) {
        return -1;
    }
    if ((att1_aimu_conformance_trace_get_snapshot(ctx->inproc, &a) != ATT1_OK) ||
        (att1_aimu_conformance_trace_get_snapshot(ctx->socket, &b) != ATT1_OK)) {
        return -1;
    }
    if (compare_bytes(&a, &b, sizeof(a), "trace snapshot", report) != 0) {
        return -1;
    }
    report->trace_compares++;
    return 0;
}

static int handle_tile_barrier(replay_ctx_t *ctx, att1_aimu_result expected)
{
    int a_complete = 0;
    int b_complete = 0;
    size_t i;

    if (expected != ATT1_AIMU_OK) {
        return -1;
    }
    for (i = 0u; i < ctx->participant_count; i++) {
        if ((att1_aimu_conformance_fabric_barrier_arrive(ctx->inproc,
                                                         ctx->participants[i],
                                                         ctx->participants,
                                                         ctx->participant_count,
                                                         &a_complete) != ATT1_OK) ||
            (att1_aimu_conformance_fabric_barrier_arrive(ctx->socket,
                                                         ctx->participants[i],
                                                         ctx->participants,
                                                         ctx->participant_count,
                                                         &b_complete) != ATT1_OK)) {
            return -1;
        }
        if (a_complete != b_complete) {
            return -1;
        }
    }
    return 0;
}

static int parse_dst_tile(const char *dst_descriptor, uint32_t *out)
{
    const char *p = strstr(dst_descriptor, "tile");
    if (p == NULL) {
        return -1;
    }
    p += 4;
    if ((*p < '0') || (*p > '9')) {
        return -1;
    }
    *out = (uint32_t)strtoul(p, NULL, 10);
    return 0;
}

static int handle_fabric_command(replay_ctx_t *ctx,
                                 const plan_cmd_t *pc,
                                 att1_aimu_cmd_type ctype,
                                 att1_aimu_result expected,
                                 fidelity_report_t *report)
{
    unsigned char payload[128];
    unsigned char a_out[128];
    unsigned char b_out[128];
    size_t a_bytes = 0u;
    size_t b_bytes = 0u;
    att1_fabric_packet a_pkt;
    att1_fabric_packet b_pkt;
    uint32_t dst_tile = 0u;
    size_t payload_bytes = sizeof(payload);
    att1_status_t st;
    uint64_t tag = (((uint64_t)(uint32_t)pc->command_id) << 32) | (uint32_t)ctype;

    if (expected != ATT1_AIMU_OK) {
        return -1;
    }
    if (parse_dst_tile(pc->dst_descriptor, &dst_tile) != 0) {
        dst_tile = (uint32_t)(((pc->tile_id + 1) < (long)ctx->cfg.tile_count) ? (pc->tile_id + 1) : 0);
    }
    fill_bytes(payload, payload_bytes, (uint32_t)pc->command_id, dst_tile);
    st = att1_aimu_conformance_fabric_send(ctx->inproc,
                                           (uint32_t)pc->tile_id,
                                           dst_tile,
                                           (ctype == ATT1_AIMU_CMD_FABRIC_REDUCE) ? ATT1_PACKET_LOGITS
                                                                                  : ATT1_PACKET_ACTIVATION,
                                           payload,
                                           payload_bytes,
                                           tag);
    if (st != ATT1_OK) {
        return -1;
    }
    st = att1_aimu_conformance_fabric_send(ctx->socket,
                                           (uint32_t)pc->tile_id,
                                           dst_tile,
                                           (ctype == ATT1_AIMU_CMD_FABRIC_REDUCE) ? ATT1_PACKET_LOGITS
                                                                                  : ATT1_PACKET_ACTIVATION,
                                           payload,
                                           payload_bytes,
                                           tag);
    if (st != ATT1_OK) {
        return -1;
    }
    memset(&a_pkt, 0, sizeof(a_pkt));
    memset(&b_pkt, 0, sizeof(b_pkt));
    st = att1_aimu_conformance_fabric_receive(ctx->inproc,
                                              dst_tile,
                                              &a_pkt,
                                              a_out,
                                              sizeof(a_out),
                                              &a_bytes);
    if (st != ATT1_OK) {
        return -1;
    }
    st = att1_aimu_conformance_fabric_receive(ctx->socket,
                                              dst_tile,
                                              &b_pkt,
                                              b_out,
                                              sizeof(b_out),
                                              &b_bytes);
    if (st != ATT1_OK) {
        return -1;
    }
    return compare_fabric_receive(&a_pkt, a_out, a_bytes,
                                  &b_pkt, b_out, b_bytes,
                                  report, "fabric command payload");
}

static int handle_kv_append(replay_ctx_t *ctx,
                            att1_aimu_result expected,
                            fidelity_report_t *report)
{
    float a[64];
    float v[64];
    att1_kv_mmu_counters ca;
    att1_kv_mmu_counters cb;

    if ((expected != ATT1_AIMU_OK) || (ensure_kv_session(ctx) != 0)) {
        return -1;
    }
    fill_floats(a, 64u, (uint32_t)ctx->kv.next_append_position, 0xA11u);
    fill_floats(v, 64u, (uint32_t)ctx->kv.next_append_position, 0xB22u);
    if ((att1_aimu_conformance_kv_append(ctx->inproc,
                                         ctx->kv.session_id,
                                         0u,
                                         ctx->kv.next_append_position,
                                         a,
                                         64u,
                                         v,
                                         64u) != ATT1_OK) ||
        (att1_aimu_conformance_kv_append(ctx->socket,
                                         ctx->kv.session_id,
                                         0u,
                                         ctx->kv.next_append_position,
                                         a,
                                         64u,
                                         v,
                                         64u) != ATT1_OK)) {
        return -1;
    }
    ctx->kv.next_append_position++;
    if ((att1_aimu_conformance_kv_get_counters(ctx->inproc, &ca) != ATT1_OK) ||
        (att1_aimu_conformance_kv_get_counters(ctx->socket, &cb) != ATT1_OK)) {
        return -1;
    }
    if (compare_bytes(&ca, &cb, sizeof(ca), "kv append counters", report) != 0) {
        return -1;
    }
    report->counter_compares++;
    return 0;
}

static int handle_kv_read(replay_ctx_t *ctx,
                          att1_aimu_result expected,
                          fidelity_report_t *report)
{
    float a_key[16];
    float a_val[16];
    float b_key[16];
    float b_val[16];

    if ((expected != ATT1_AIMU_OK) || (ensure_kv_session(ctx) != 0)) {
        return -1;
    }
    if (ctx->kv.next_append_position == 0u) {
        if (handle_kv_append(ctx, ATT1_AIMU_OK, report) != 0) {
            return -1;
        }
    }
    memset(a_key, 0, sizeof(a_key));
    memset(a_val, 0, sizeof(a_val));
    memset(b_key, 0, sizeof(b_key));
    memset(b_val, 0, sizeof(b_val));
    if ((att1_aimu_conformance_kv_read(ctx->inproc,
                                       ctx->kv.session_id,
                                       0u,
                                       0u,
                                       0u,
                                       a_key,
                                       16u,
                                       a_val,
                                       16u) != ATT1_OK) ||
        (att1_aimu_conformance_kv_read(ctx->socket,
                                       ctx->kv.session_id,
                                       0u,
                                       0u,
                                       0u,
                                       b_key,
                                       16u,
                                       b_val,
                                       16u) != ATT1_OK)) {
        return -1;
    }
    if ((compare_bytes(a_key, b_key, sizeof(a_key), "kv key", report) != 0) ||
        (compare_bytes(a_val, b_val, sizeof(a_val), "kv value", report) != 0)) {
        return -1;
    }
    report->payload_compares += 2;
    return 0;
}

static int replay_plan_command(replay_ctx_t *ctx,
                               const plan_cmd_t *pc,
                               int strict,
                               fidelity_report_t *report)
{
    att1_aimu_cmd_type ctype;
    att1_aimu_result expected;

    if (map_cmd_type(pc->command_type, &ctype) != 0) {
        return strict ? -1 : 0;
    }
    (void)map_expected_result(pc->expected_status, &expected);

    switch (ctype) {
    case ATT1_AIMU_CMD_LOAD_TENSOR_TILE:
        return handle_load_tensor(ctx, pc, expected, report);
    case ATT1_AIMU_CMD_VALIDATE_TENSOR:
        if (handle_validate_tensor(ctx, pc, expected) != 0) {
            return -1;
        }
        report->commands_replayed++;
        return 0;
    case ATT1_AIMU_CMD_EXEC_MATMUL:
    case ATT1_AIMU_CMD_EXEC_RMSNORM:
    case ATT1_AIMU_CMD_EXEC_ROPE:
    case ATT1_AIMU_CMD_EXEC_FFN:
        return handle_exec(ctx, pc, ctype, expected, report);
    case ATT1_AIMU_CMD_EXEC_ATTENTION:
    case ATT1_AIMU_CMD_NOP:
    case ATT1_AIMU_CMD_RESET_TILE: {
        att1_aimu_cmd cmd;
        memset(&cmd, 0, sizeof(cmd));
        cmd.command_type = (uint8_t)ctype;
        cmd.tile_id = (uint8_t)pc->tile_id;
        cmd.session_id = (uint8_t)ctx->kv.session_id;
        return submit_and_compare(ctx, &cmd, expected, report, pc);
    }
    case ATT1_AIMU_CMD_QUERY_COUNTERS:
        if (handle_query_counters(ctx, expected, report) != 0) {
            return -1;
        }
        report->commands_replayed++;
        return 0;
    case ATT1_AIMU_CMD_TRACE_SNAPSHOT:
        if (handle_trace_snapshot(ctx, expected, report) != 0) {
            return -1;
        }
        report->commands_replayed++;
        return 0;
    case ATT1_AIMU_CMD_TILE_BARRIER:
        if (handle_tile_barrier(ctx, expected) != 0) {
            return -1;
        }
        report->commands_replayed++;
        return 0;
    case ATT1_AIMU_CMD_FABRIC_SEND:
    case ATT1_AIMU_CMD_FABRIC_REDUCE:
        if (handle_fabric_command(ctx, pc, ctype, expected, report) != 0) {
            return -1;
        }
        report->commands_replayed++;
        return 0;
    case ATT1_AIMU_CMD_KV_APPEND:
        if (handle_kv_append(ctx, expected, report) != 0) {
            return -1;
        }
        report->commands_replayed++;
        return 0;
    case ATT1_AIMU_CMD_KV_READ:
        if (handle_kv_read(ctx, expected, report) != 0) {
            return -1;
        }
        report->commands_replayed++;
        return 0;
    default:
        return -1;
    }
}

static att1_packet_type route_packet_type(const char *route_type)
{
    if (strstr(route_type, "ACTIVATION") != NULL) {
        return ATT1_PACKET_ACTIVATION;
    }
    if ((strstr(route_type, "LOGIT") != NULL) || (strstr(route_type, "REDUCE") != NULL)) {
        return ATT1_PACKET_LOGITS;
    }
    if (strstr(route_type, "KV") != NULL) {
        return ATT1_PACKET_KV_PAGE;
    }
    if (strstr(route_type, "TRACE") != NULL) {
        return ATT1_PACKET_TRACE;
    }
    if (strstr(route_type, "BARRIER") != NULL) {
        return ATT1_PACKET_BARRIER;
    }
    return ATT1_PACKET_CONTROL;
}

static int replay_route(replay_ctx_t *ctx,
                        const route_t *route,
                        int strict,
                        fidelity_report_t *report)
{
    unsigned char *payload;
    size_t payload_bytes;
    att1_packet_type pkt_type;
    uint64_t tag;
    att1_status_t st;
    int i;

    if (route->destination_count <= 0) {
        return strict ? -1 : 0;
    }
    if (strcmp(route->route_type, "TILE_BARRIER") == 0) {
        int a_complete = 0;
        int b_complete = 0;
        for (i = 0; i < route->destination_count; i++) {
            st = att1_aimu_conformance_fabric_barrier_arrive(ctx->inproc,
                                                             (uint32_t)route->destination_tiles[i],
                                                             ctx->participants,
                                                             ctx->participant_count,
                                                             &a_complete);
            if (st != ATT1_OK) {
                return -1;
            }
            st = att1_aimu_conformance_fabric_barrier_arrive(ctx->socket,
                                                             (uint32_t)route->destination_tiles[i],
                                                             ctx->participants,
                                                             ctx->participant_count,
                                                             &b_complete);
            if (st != ATT1_OK) {
                return -1;
            }
            if (a_complete != b_complete) {
                return -1;
            }
        }
        report->routes_replayed++;
        return 0;
    }

    payload_bytes = (route->payload_bytes <= 0) ? 1u : (size_t)route->payload_bytes;
    if (payload_bytes > ATT1_AIMU_ENDPOINT_MAX_PAYLOAD) {
        payload_bytes = ATT1_AIMU_ENDPOINT_MAX_PAYLOAD;
    }
    payload = (unsigned char *)malloc(payload_bytes);
    if (payload == NULL) {
        return -1;
    }
    fill_bytes(payload, payload_bytes, (uint32_t)route->route_id, (uint32_t)route->source_tile);
    pkt_type = route_packet_type(route->route_type);
    tag = (((uint64_t)(uint32_t)route->route_id) << 32) | (uint32_t)route->reduction_id;

    if (route->destination_count > 1) {
        uint32_t group[MAX_DEST_TILES];
        att1_fabric_packet a_pkt;
        att1_fabric_packet b_pkt;
        unsigned char a_out[ATT1_AIMU_ENDPOINT_MAX_PAYLOAD];
        unsigned char b_out[ATT1_AIMU_ENDPOINT_MAX_PAYLOAD];
        size_t a_bytes;
        size_t b_bytes;
        for (i = 0; i < route->destination_count; i++) {
            group[i] = (uint32_t)route->destination_tiles[i];
        }
        st = att1_aimu_conformance_fabric_broadcast(ctx->inproc,
                                                    (uint32_t)route->source_tile,
                                                    group,
                                                    (size_t)route->destination_count,
                                                    pkt_type,
                                                    payload,
                                                    payload_bytes,
                                                    tag);
        if (st != ATT1_OK) {
            free(payload);
            return -1;
        }
        st = att1_aimu_conformance_fabric_broadcast(ctx->socket,
                                                    (uint32_t)route->source_tile,
                                                    group,
                                                    (size_t)route->destination_count,
                                                    pkt_type,
                                                    payload,
                                                    payload_bytes,
                                                    tag);
        if (st != ATT1_OK) {
            free(payload);
            return -1;
        }
        for (i = 0; i < route->destination_count; i++) {
            memset(&a_pkt, 0, sizeof(a_pkt));
            memset(&b_pkt, 0, sizeof(b_pkt));
            a_bytes = 0u;
            b_bytes = 0u;
            st = att1_aimu_conformance_fabric_receive(ctx->inproc,
                                                      group[i],
                                                      &a_pkt,
                                                      a_out,
                                                      sizeof(a_out),
                                                      &a_bytes);
            if (st != ATT1_OK) {
                free(payload);
                return -1;
            }
            st = att1_aimu_conformance_fabric_receive(ctx->socket,
                                                      group[i],
                                                      &b_pkt,
                                                      b_out,
                                                      sizeof(b_out),
                                                      &b_bytes);
            if (st != ATT1_OK) {
                free(payload);
                return -1;
            }
            if (compare_fabric_receive(&a_pkt, a_out, a_bytes,
                                       &b_pkt, b_out, b_bytes,
                                       report, "route broadcast") != 0) {
                free(payload);
                return -1;
            }
        }
    } else {
        att1_fabric_packet a_pkt;
        att1_fabric_packet b_pkt;
        unsigned char a_out[ATT1_AIMU_ENDPOINT_MAX_PAYLOAD];
        unsigned char b_out[ATT1_AIMU_ENDPOINT_MAX_PAYLOAD];
        size_t a_bytes = 0u;
        size_t b_bytes = 0u;
        uint32_t dst = (uint32_t)route->destination_tiles[0];
        st = att1_aimu_conformance_fabric_send(ctx->inproc,
                                               (uint32_t)route->source_tile,
                                               dst,
                                               pkt_type,
                                               payload,
                                               payload_bytes,
                                               tag);
        if (st != ATT1_OK) {
            free(payload);
            return -1;
        }
        st = att1_aimu_conformance_fabric_send(ctx->socket,
                                               (uint32_t)route->source_tile,
                                               dst,
                                               pkt_type,
                                               payload,
                                               payload_bytes,
                                               tag);
        if (st != ATT1_OK) {
            free(payload);
            return -1;
        }
        memset(&a_pkt, 0, sizeof(a_pkt));
        memset(&b_pkt, 0, sizeof(b_pkt));
        st = att1_aimu_conformance_fabric_receive(ctx->inproc,
                                                  dst,
                                                  &a_pkt,
                                                  a_out,
                                                  sizeof(a_out),
                                                  &a_bytes);
        if (st != ATT1_OK) {
            free(payload);
            return -1;
        }
        st = att1_aimu_conformance_fabric_receive(ctx->socket,
                                                  dst,
                                                  &b_pkt,
                                                  b_out,
                                                  sizeof(b_out),
                                                  &b_bytes);
        if (st != ATT1_OK) {
            free(payload);
            return -1;
        }
        if (compare_fabric_receive(&a_pkt, a_out, a_bytes,
                                   &b_pkt, b_out, b_bytes,
                                   report, "route send") != 0) {
            free(payload);
            return -1;
        }
    }

    free(payload);
    report->routes_replayed++;
    return 0;
}

static int final_compare(replay_ctx_t *ctx, fidelity_report_t *report)
{
    att1_aimu_cmdq_counters cmd_a;
    att1_aimu_cmdq_counters cmd_b;
    att1_aimu_dma_counters dma_a;
    att1_aimu_dma_counters dma_b;
    att1_fabric_counters fab_a;
    att1_fabric_counters fab_b;
    att1_kv_mmu_counters kv_a;
    att1_kv_mmu_counters kv_b;
    att1_aimu_trace_snapshot trace_a;
    att1_aimu_trace_snapshot trace_b;

    if ((att1_aimu_conformance_cmd_get_counters(ctx->inproc, &cmd_a) != ATT1_OK) ||
        (att1_aimu_conformance_cmd_get_counters(ctx->socket, &cmd_b) != ATT1_OK) ||
        (att1_aimu_conformance_dma_get_counters(ctx->inproc, &dma_a) != ATT1_OK) ||
        (att1_aimu_conformance_dma_get_counters(ctx->socket, &dma_b) != ATT1_OK) ||
        (att1_aimu_conformance_fabric_get_counters(ctx->inproc, &fab_a) != ATT1_OK) ||
        (att1_aimu_conformance_fabric_get_counters(ctx->socket, &fab_b) != ATT1_OK) ||
        (att1_aimu_conformance_kv_get_counters(ctx->inproc, &kv_a) != ATT1_OK) ||
        (att1_aimu_conformance_kv_get_counters(ctx->socket, &kv_b) != ATT1_OK) ||
        (att1_aimu_conformance_trace_get_snapshot(ctx->inproc, &trace_a) != ATT1_OK) ||
        (att1_aimu_conformance_trace_get_snapshot(ctx->socket, &trace_b) != ATT1_OK)) {
        return -1;
    }
    if ((compare_bytes(&cmd_a, &cmd_b, sizeof(cmd_a), "final cmd counters", report) != 0) ||
        (compare_bytes(&dma_a, &dma_b, sizeof(dma_a), "final dma counters", report) != 0) ||
        (compare_bytes(&fab_a, &fab_b, sizeof(fab_a), "final fabric counters", report) != 0) ||
        (compare_bytes(&kv_a, &kv_b, sizeof(kv_a), "final kv counters", report) != 0) ||
        (compare_bytes(&trace_a, &trace_b, sizeof(trace_a), "final trace snapshot", report) != 0)) {
        return -1;
    }
    report->counter_compares += 4;
    report->trace_compares++;
    return 0;
}

static void emit_text_report(const char *plan_path,
                             const char *route_path,
                             const plan_t *plan,
                             const route_report_t *routes,
                             const fidelity_report_t *report)
{
    printf("att1-aimu-replay-fidelity report\n");
    printf("  plan_path           : %s\n", plan_path);
    printf("  routes_path         : %s\n", (route_path != NULL) ? route_path : "(none)");
    printf("  model_id            : %s\n", plan->header.model_id);
    printf("  session_id          : %s\n", plan->header.session_id);
    printf("  tile_count          : %d\n", plan->header.tile_count);
    printf("  command_count       : %d\n", plan->n_commands);
    printf("  route_count         : %d\n", (routes != NULL) ? routes->n_routes : 0);
    printf("  commands_replayed   : %ld\n", report->commands_replayed);
    printf("  routes_replayed     : %ld\n", report->routes_replayed);
    printf("  completions_compared: %ld\n", report->completions_compared);
    printf("  payload_compares    : %ld\n", report->payload_compares);
    printf("  counter_compares    : %ld\n", report->counter_compares);
    printf("  trace_compares      : %ld\n", report->trace_compares);
    printf("  mismatches          : %ld\n", report->mismatches);
    printf("  status              : %s\n", report->status_pass ? "pass" : "fail");
    if (report->notes[0] != '\0') {
        printf("  notes               : %s\n", report->notes);
    }
}

static void emit_json_report(FILE *fp,
                             const char *plan_path,
                             const char *route_path,
                             const plan_t *plan,
                             const route_report_t *routes,
                             const fidelity_report_t *report)
{
    fprintf(fp, "{\n");
    fprintf(fp, "  \"replay_fidelity_report_version\": %d,\n", REPORT_VERSION);
    fprintf(fp, "  \"plan_path\": \"%s\",\n", plan_path);
    if (route_path != NULL) {
        fprintf(fp, "  \"routes_path\": \"%s\",\n", route_path);
    } else {
        fprintf(fp, "  \"routes_path\": null,\n");
    }
    fprintf(fp, "  \"model_id\": \"%s\",\n", plan->header.model_id);
    fprintf(fp, "  \"session_id\": \"%s\",\n", plan->header.session_id);
    fprintf(fp, "  \"tile_count\": %d,\n", plan->header.tile_count);
    fprintf(fp, "  \"command_count\": %d,\n", plan->n_commands);
    fprintf(fp, "  \"route_count\": %d,\n", (routes != NULL) ? routes->n_routes : 0);
    fprintf(fp, "  \"commands_replayed\": %ld,\n", report->commands_replayed);
    fprintf(fp, "  \"routes_replayed\": %ld,\n", report->routes_replayed);
    fprintf(fp, "  \"completions_compared\": %ld,\n", report->completions_compared);
    fprintf(fp, "  \"payload_compares\": %ld,\n", report->payload_compares);
    fprintf(fp, "  \"counter_compares\": %ld,\n", report->counter_compares);
    fprintf(fp, "  \"trace_compares\": %ld,\n", report->trace_compares);
    fprintf(fp, "  \"mismatches\": %ld,\n", report->mismatches);
    fprintf(fp, "  \"status\": \"%s\",\n", report->status_pass ? "pass" : "fail");
    fprintf(fp, "  \"notes\": \"%s\"\n", report->notes);
    fprintf(fp, "}\n");
}

int main(int argc, char *argv[])
{
    const char *plan_path = NULL;
    const char *route_path = NULL;
    const char *report_json_path = NULL;
    int strict = 0;
    int i;
    plan_t plan;
    route_report_t routes;
    route_report_t *routes_ptr = NULL;
    replay_ctx_t ctx;
    fidelity_report_t report;
    int exit_code = 0;

    memset(&report, 0, sizeof(report));
    memset(&ctx, 0, sizeof(ctx));
    report.status_pass = 1;

    for (i = 1; i < argc; i++) {
        if ((strcmp(argv[i], "--plan") == 0) && (i + 1 < argc)) {
            plan_path = argv[++i];
        } else if ((strcmp(argv[i], "--routes") == 0) && (i + 1 < argc)) {
            route_path = argv[++i];
        } else if ((strcmp(argv[i], "--report-json") == 0) && (i + 1 < argc)) {
            report_json_path = argv[++i];
        } else if (strcmp(argv[i], "--strict") == 0) {
            strict = 1;
        } else if (strcmp(argv[i], "--help") == 0) {
            printf("Usage: att1-aimu-replay-fidelity --plan PATH [--routes PATH] [--report-json PATH] [--strict]\n");
            return 0;
        } else {
            fprintf(stderr, "replay-fidelity: unknown argument: %s\n", argv[i]);
            return 2;
        }
    }

    if (plan_path == NULL) {
        fprintf(stderr, "replay-fidelity: --plan PATH is required\n");
        return 2;
    }
    if (parse_plan(plan_path, &plan) != 0) {
        return 2;
    }
    if (route_path != NULL) {
        if (parse_routes(route_path, &routes) != 0) {
            return 2;
        }
        routes_ptr = &routes;
    }
    if (replay_ctx_init(&ctx, &plan) != 0) {
        fprintf(stderr, "replay-fidelity: failed to create endpoints\n");
        return 1;
    }

    for (i = 0; i < plan.n_commands; i++) {
        if (replay_plan_command(&ctx, &plan.commands[i], strict, &report) != 0) {
            report.status_pass = 0;
            exit_code = 1;
            break;
        }
    }
    if ((exit_code == 0) && (routes_ptr != NULL)) {
        for (i = 0; i < routes_ptr->n_routes; i++) {
            if (replay_route(&ctx, &routes_ptr->routes[i], strict, &report) != 0) {
                report.status_pass = 0;
                exit_code = 1;
                break;
            }
        }
    }
    if ((exit_code == 0) && (final_compare(&ctx, &report) != 0)) {
        report.status_pass = 0;
        exit_code = 1;
    }

    emit_text_report(plan_path, route_path, &plan, routes_ptr, &report);
    if (report_json_path != NULL) {
        FILE *fp = fopen(report_json_path, "w");
        if (fp == NULL) {
            fprintf(stderr, "replay-fidelity: cannot write report JSON: %s\n", report_json_path);
            exit_code = 1;
        } else {
            emit_json_report(fp, plan_path, route_path, &plan, routes_ptr, &report);
            fclose(fp);
        }
    }

    if (exit_code == 0) {
        PASS("replay_fidelity_gate");
    } else {
        FAIL("replay_fidelity_gate");
    }

    replay_ctx_cleanup(&ctx);
    return exit_code;
}
