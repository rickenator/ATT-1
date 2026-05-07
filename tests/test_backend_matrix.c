/* Backend matrix regression harness (M29).
 *
 * Exercises all 8 backend×mode combinations for the dummy .att1 model via
 * att1-bench.  For each entry it validates:
 *   - att1-bench exits zero
 *   - output reports the expected mode= and backend= labels
 *   - tokens_decoded, logits_bytes_produced are nonzero
 *   - fabric_packets_sent is nonzero in cluster mode
 *
 * After individual checks, all passing single-mode entries must produce the
 * same last_token, and all passing cluster-mode entries must produce the same
 * last_token (cross-backend consistency for the dummy model).
 *
 * CUDA entries are skipped when CUDA is unavailable.
 */
#include "att1_backend.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MATRIX_MODEL      "models/dummy/model.att1"
#define MATRIX_META_MODEL "models/shard_meta/model.att1"
#define MATRIX_STUB_MODEL "models/converted_stub_meta/model.att1"
#define MATRIX_Q4_MODEL   "models/q4_tiny/model.att1"
#define MATRIX_PROMPT     "hello"
#define MATRIX_TOKENS     "8"
#define MATRIX_TILES      "2"
#define MATRIX_META_TILES "1"
#define MATRIX_STUB_TILES "2"

typedef struct {
    const char *backend;
    const char *mode;              /* "single" or "cluster" */
    const char *model;             /* model file path */
    const char *tiles;             /* NULL for single mode */
    const char *shard_plan;        /* NULL=single, "runtime", "metadata" */
    int         is_cluster;
    int         requires_cuda;
    int         consistency_group; /* 0=single, 1=cluster-dummy, 2=cluster-meta, 3=cluster-stub */
} matrix_entry_t;

typedef struct {
    int      status;               /* 0=pass, 1=skip, -1=fail */
    uint32_t last_token;
    uint64_t tokens_decoded;
    uint64_t token_time_us_total;
    uint64_t logits_bytes_produced;
    uint64_t fabric_packets_sent;  /* cluster only */
} matrix_result_t;

static const matrix_entry_t k_matrix[] = {
    /* single mode — dummy model */
    { "cpu-f32", "single",  MATRIX_MODEL,      NULL,              NULL,       0, 0, 0 },
    { "cpu-q8",  "single",  MATRIX_MODEL,      NULL,              NULL,       0, 0, 0 },
    { "cuda",    "single",  MATRIX_MODEL,      NULL,              NULL,       0, 1, 0 },
    { "cuda-q8", "single",  MATRIX_MODEL,      NULL,              NULL,       0, 1, 0 },
    /* cluster mode — dummy model, runtime plan, 2 tiles */
    { "cpu-f32", "cluster", MATRIX_MODEL,      MATRIX_TILES,      "runtime",  1, 0, 1 },
    { "cpu-q8",  "cluster", MATRIX_MODEL,      MATRIX_TILES,      "runtime",  1, 0, 1 },
    { "cuda",    "cluster", MATRIX_MODEL,      MATRIX_TILES,      "runtime",  1, 1, 1 },
    { "cuda-q8", "cluster", MATRIX_MODEL,      MATRIX_TILES,      "runtime",  1, 1, 1 },
    /* cluster mode — shard_meta fixture, runtime plan, 1 tile */
    { "cpu-f32", "cluster", MATRIX_META_MODEL, MATRIX_META_TILES, "runtime",  1, 0, 2 },
    { "cpu-q8",  "cluster", MATRIX_META_MODEL, MATRIX_META_TILES, "runtime",  1, 0, 2 },
    { "cuda",    "cluster", MATRIX_META_MODEL, MATRIX_META_TILES, "runtime",  1, 1, 2 },
    { "cuda-q8", "cluster", MATRIX_META_MODEL, MATRIX_META_TILES, "runtime",  1, 1, 2 },
    /* cluster mode — shard_meta fixture, metadata plan, 1 tile */
    { "cpu-f32", "cluster", MATRIX_META_MODEL, MATRIX_META_TILES, "metadata", 1, 0, 2 },
    { "cpu-q8",  "cluster", MATRIX_META_MODEL, MATRIX_META_TILES, "metadata", 1, 0, 2 },
    { "cuda",    "cluster", MATRIX_META_MODEL, MATRIX_META_TILES, "metadata", 1, 1, 2 },
    { "cuda-q8", "cluster", MATRIX_META_MODEL, MATRIX_META_TILES, "metadata", 1, 1, 2 },
    /* cluster mode — converted stub with shard metadata, runtime plan, 2 tiles */
    { "cpu-f32", "cluster", MATRIX_STUB_MODEL, MATRIX_STUB_TILES, "runtime",  1, 0, 3 },
    { "cpu-q8",  "cluster", MATRIX_STUB_MODEL, MATRIX_STUB_TILES, "runtime",  1, 0, 3 },
    { "cuda",    "cluster", MATRIX_STUB_MODEL, MATRIX_STUB_TILES, "runtime",  1, 1, 3 },
    { "cuda-q8", "cluster", MATRIX_STUB_MODEL, MATRIX_STUB_TILES, "runtime",  1, 1, 3 },
    /* cluster mode — converted stub with shard metadata, metadata plan, 2 tiles */
    { "cpu-f32", "cluster", MATRIX_STUB_MODEL, MATRIX_STUB_TILES, "metadata", 1, 0, 3 },
    { "cpu-q8",  "cluster", MATRIX_STUB_MODEL, MATRIX_STUB_TILES, "metadata", 1, 0, 3 },
    { "cuda",    "cluster", MATRIX_STUB_MODEL, MATRIX_STUB_TILES, "metadata", 1, 1, 3 },
    { "cuda-q8", "cluster", MATRIX_STUB_MODEL, MATRIX_STUB_TILES, "metadata", 1, 1, 3 },
    /* cpu-q4 single and cluster — q4_tiny fixture (group 4) */
    { "cpu-q4",  "single",  MATRIX_Q4_MODEL,   NULL,              NULL,       0, 0, 4 },
    { "cpu-q4",  "cluster", MATRIX_Q4_MODEL,   MATRIX_TILES,      "runtime",  1, 0, 4 },
};

#define MATRIX_COUNT (sizeof(k_matrix) / sizeof(k_matrix[0]))

static int matrix_read_file(const char *path, char *buf, size_t cap)
{
    FILE *fp = NULL;
    size_t n = 0u;

    if ((path == NULL) || (buf == NULL) || (cap == 0u)) {
        return -1;
    }

    fp = fopen(path, "rb");
    if (fp == NULL) {
        return -1;
    }

    n = fread(buf, 1u, cap - 1u, fp);
    if (ferror(fp) != 0) {
        fclose(fp);
        return -1;
    }

    buf[n] = '\0';
    fclose(fp);
    return 0;
}

static int matrix_parse_u32(const char *text, const char *key, uint32_t *out)
{
    const char *p = NULL;
    char *end = NULL;
    unsigned long v = 0u;

    if ((text == NULL) || (key == NULL) || (out == NULL)) {
        return -1;
    }

    p = strstr(text, key);
    if (p == NULL) {
        return -1;
    }

    p += strlen(key);
    if (*p != '=') {
        return -1;
    }
    p++;

    v = strtoul(p, &end, 10);
    if (end == p) {
        return -1;
    }

    *out = (uint32_t)v;
    return 0;
}

static int matrix_parse_u64(const char *text, const char *key, uint64_t *out)
{
    const char *p = NULL;
    char *end = NULL;
    unsigned long long v = 0u;

    if ((text == NULL) || (key == NULL) || (out == NULL)) {
        return -1;
    }

    p = strstr(text, key);
    if (p == NULL) {
        return -1;
    }

    p += strlen(key);
    if (*p != '=') {
        return -1;
    }
    p++;

    v = strtoull(p, &end, 10);
    if (end == p) {
        return -1;
    }

    *out = (uint64_t)v;
    return 0;
}

static void run_entry(size_t idx,
                      const matrix_entry_t *e,
                      matrix_result_t *r)
{
    char cmd[640];
    char outpath[64];
    char output[4096];
    char expected_mode[32];
    char expected_backend[32];
    const char *expected_shard_plan;
    int rc = 0;

    r->status               = -1;
    r->last_token           = 0u;
    r->tokens_decoded       = 0u;
    r->token_time_us_total  = 0u;
    r->logits_bytes_produced = 0u;
    r->fabric_packets_sent  = 0u;

    if (e->requires_cuda && !att1_backend_cuda_available()) {
        r->status = 1; /* skip */
        return;
    }

    snprintf(outpath, sizeof(outpath), "build/matrix_%zu.txt", idx);

    if (e->is_cluster) {
        snprintf(cmd, sizeof(cmd),
                 "./build/att1-bench"
                 " --model %s"
                 " --prompt " MATRIX_PROMPT
                 " --tokens " MATRIX_TOKENS
                 " --mode cluster --tiles %s"
                 " --backend %s"
                 " --shard-plan %s"
                 " > %s 2>&1",
                 e->model, e->tiles, e->backend, e->shard_plan, outpath);
    } else {
        snprintf(cmd, sizeof(cmd),
                 "./build/att1-bench"
                 " --model %s"
                 " --prompt " MATRIX_PROMPT
                 " --tokens " MATRIX_TOKENS
                 " --mode single"
                 " --backend %s"
                 " > %s 2>&1",
                 e->model, e->backend, outpath);
    }

    rc = system(cmd);
    if (rc != 0) {
        return; /* status stays -1 */
    }

    if (matrix_read_file(outpath, output, sizeof(output)) != 0) {
        return;
    }

    snprintf(expected_mode,    sizeof(expected_mode),    "mode=%s",    e->mode);
    snprintf(expected_backend, sizeof(expected_backend), "backend=%s", e->backend);

    if ((strstr(output, expected_mode)    == NULL) ||
        (strstr(output, expected_backend) == NULL)) {
        return;
    }

    /* Verify shard_plan= label: cluster passes explicit plan, single is always runtime. */
    expected_shard_plan = e->is_cluster ? e->shard_plan : "runtime";
    {
        char expected_plan[32];
        snprintf(expected_plan, sizeof(expected_plan), "shard_plan=%s", expected_shard_plan);
        if (strstr(output, expected_plan) == NULL) {
            return;
        }
    }

    if ((matrix_parse_u32(output, "last_token",            &r->last_token)           != 0) ||
        (matrix_parse_u64(output, "tokens_decoded",         &r->tokens_decoded)        != 0) ||
        (matrix_parse_u64(output, "token_time_us_total",    &r->token_time_us_total)   != 0) ||
        (matrix_parse_u64(output, "logits_bytes_produced",  &r->logits_bytes_produced) != 0)) {
        return;
    }

    if ((r->tokens_decoded == 0u) || (r->logits_bytes_produced == 0u)) {
        return;
    }

    if (e->is_cluster) {
        if ((matrix_parse_u64(output, "fabric_packets_sent",
                              &r->fabric_packets_sent) != 0) ||
            (r->fabric_packets_sent == 0u)) {
            return;
        }
    }

    r->status = 0; /* pass */
}

static void print_result(const matrix_entry_t *e, const matrix_result_t *r)
{
    const char *plan = (e->shard_plan != NULL) ? e->shard_plan : "n/a";
    const char *tag  = (r->status == 0) ? "PASS" :
                       (r->status == 1) ? "SKIP" : "FAIL";

    if (r->status == 1) {
        fprintf(stderr,
                "backend_matrix: %-8s %-7s %-8s  %s  (CUDA unavailable)\n",
                e->backend, e->mode, plan, tag);
        return;
    }

    if (r->status != 0) {
        fprintf(stderr,
                "backend_matrix: %-8s %-7s %-8s  %s\n",
                e->backend, e->mode, plan, tag);
        return;
    }

    if (e->is_cluster) {
        fprintf(stderr,
                "backend_matrix: %-8s %-7s %-8s  %s"
                "  last_token=%-5u tokens=%-3llu"
                " time_us=%-9llu fabric_pkts=%-4llu logits_bytes=%llu\n",
                e->backend, e->mode, plan, tag,
                r->last_token,
                (unsigned long long)r->tokens_decoded,
                (unsigned long long)r->token_time_us_total,
                (unsigned long long)r->fabric_packets_sent,
                (unsigned long long)r->logits_bytes_produced);
    } else {
        fprintf(stderr,
                "backend_matrix: %-8s %-7s %-8s  %s"
                "  last_token=%-5u tokens=%-3llu"
                " time_us=%-9llu logits_bytes=%llu\n",
                e->backend, e->mode, plan, tag,
                r->last_token,
                (unsigned long long)r->tokens_decoded,
                (unsigned long long)r->token_time_us_total,
                (unsigned long long)r->logits_bytes_produced);
    }
}

int main(void)
{
    matrix_result_t results[MATRIX_COUNT];
    size_t   i = 0u;
    size_t   passed = 0u;
    size_t   skipped = 0u;
    size_t   failed = 0u;
    /* One reference last_token per consistency group (0=single, 1=cluster-dummy,
     * 2=cluster-shard_meta, 3=cluster-converted_stub_meta, 4=q4-q4_tiny).
     * Group 2 spans both runtime and metadata plans on the 1-tile shard_meta fixture.
     * Group 3 spans both runtime and metadata plans on the 2-tile converted stub.
     * Group 4 spans cpu-q4 single and cpu-q4 cluster on the q4_tiny fixture. */
    uint32_t refs[5]     = {0u, 0u, 0u, 0u, 0u};
    int      refs_set[5] = {0, 0, 0, 0, 0};
    int      token_mismatch = 0;

    for (i = 0u; i < MATRIX_COUNT; i++) {
        run_entry(i, &k_matrix[i], &results[i]);
        print_result(&k_matrix[i], &results[i]);

        if (results[i].status == 0) {
            passed++;
        } else if (results[i].status == 1) {
            skipped++;
        } else {
            failed++;
        }
    }

    /* Cross-backend/cross-plan consistency: within each group, all passing
     * entries must produce the same last_token. */
    for (i = 0u; i < MATRIX_COUNT; i++) {
        int g = k_matrix[i].consistency_group;

        if (results[i].status != 0) {
            continue;
        }

        if (!refs_set[g]) {
            refs[g]     = results[i].last_token;
            refs_set[g] = 1;
        } else if (results[i].last_token != refs[g]) {
            fprintf(stderr,
                    "backend_matrix: token mismatch (group %d)"
                    " %s/%s/%s got %u expected %u\n",
                    g,
                    k_matrix[i].backend,
                    k_matrix[i].mode,
                    k_matrix[i].shard_plan ? k_matrix[i].shard_plan : "n/a",
                    results[i].last_token,
                    refs[g]);
            token_mismatch = 1;
        }
    }

    fprintf(stderr,
            "backend_matrix: %zu/%zu passed, %zu skipped, %zu failed\n",
            passed, MATRIX_COUNT, skipped, failed);

    if ((failed != 0u) || (token_mismatch != 0)) {
        fputs("backend_matrix test failed\n", stderr);
        return 1;
    }

    puts("backend_matrix test passed");
    return 0;
}
