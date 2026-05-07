/*
 * test_shard_meta_exec.c — Milestone 40: opt-in metadata shard plan execution
 *
 * Tests for att1_shard_plan_from_meta() and the ATT1_SHARD_PLAN_METADATA
 * opt-in path in att1_cluster_infer_create().
 *
 * All tests that need a model file write it to OUT_DIR to avoid polluting
 * the checked-in models directory.
 */

#include "att1_cluster_infer.h"
#include "att1_infer.h"
#include "att1_model.h"
#include "att1_shard.h"
#include "att1_shard_meta.h"
#include "att1_status.h"

#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#define OUT_DIR "build/shard_meta_exec_test"

/* ── little-endian write helpers ──────────────────────────────────────────── */

static void write_u32le(unsigned char *p, uint32_t v)
{
    p[0] = (unsigned char)(v & 0xFFu);
    p[1] = (unsigned char)((v >> 8u) & 0xFFu);
    p[2] = (unsigned char)((v >> 16u) & 0xFFu);
    p[3] = (unsigned char)((v >> 24u) & 0xFFu);
}

static void write_u64le(unsigned char *p, uint64_t v)
{
    p[0] = (unsigned char)(v & 0xFFu);
    p[1] = (unsigned char)((v >> 8u) & 0xFFu);
    p[2] = (unsigned char)((v >> 16u) & 0xFFu);
    p[3] = (unsigned char)((v >> 24u) & 0xFFu);
    p[4] = (unsigned char)((v >> 32u) & 0xFFu);
    p[5] = (unsigned char)((v >> 40u) & 0xFFu);
    p[6] = (unsigned char)((v >> 48u) & 0xFFu);
    p[7] = (unsigned char)((v >> 56u) & 0xFFu);
}

static void write_f32le(unsigned char *p, float v)
{
    uint32_t bits = 0u;
    memcpy(&bits, &v, sizeof(bits));
    write_u32le(p, bits);
}

/* ── file I/O helper ──────────────────────────────────────────────────────── */

static int write_file(const char *path,
                      const unsigned char *data,
                      size_t size)
{
    FILE *fp = fopen(path, "wb");

    if (fp == NULL) {
        return -1;
    }
    if ((size > 0u) && (fwrite(data, 1u, size, fp) != size)) {
        fclose(fp);
        return -1;
    }
    fclose(fp);
    return 0;
}

/* ── multi-tensor model builder (reused from test_shard_meta_plan.c) ──────── */

#define EM_HEADER_SIZE  80u
#define EM_CONFIG_SIZE  36u
#define EM_DESC_SIZE   128u
#define EM_SHARD_SIZE  120u
#define EM_TENSOR_BYTES 16u   /* 4 × float32 */

static uint32_t em_desc_off(uint32_t n)  { (void)n; return EM_HEADER_SIZE + EM_CONFIG_SIZE; }
static uint32_t em_data_off(uint32_t n)  { return em_desc_off(n) + n * EM_DESC_SIZE; }
static uint32_t em_shard_off(uint32_t n) { return em_data_off(n) + n * EM_TENSOR_BYTES; }
static uint32_t em_file_size(uint32_t n) { return em_shard_off(n) + n * EM_SHARD_SIZE; }

typedef struct em_tensor_spec {
    const char *name;
    uint32_t    tile_id;
} em_tensor_spec;

static void em_write_header(unsigned char *buf,
                            uint32_t n_tensors,
                            uint32_t n_tiles,
                            uint32_t n_layers,
                            int with_shard)
{
    uint64_t shard_off  = with_shard ? em_shard_off(n_tensors) : 0u;
    uint64_t shard_size = with_shard ? (uint64_t)n_tensors * EM_SHARD_SIZE : 0u;

    memcpy(&buf[0], ATT1_MODEL_MAGIC, ATT1_MODEL_MAGIC_SIZE);
    write_u32le(&buf[8],  ATT1_MODEL_VERSION);
    write_u32le(&buf[12], ATT1_MODEL_HEADER_SIZE);
    write_u64le(&buf[16], EM_HEADER_SIZE);
    write_u64le(&buf[24], EM_CONFIG_SIZE);
    write_u64le(&buf[32], em_desc_off(n_tensors));
    write_u64le(&buf[40], n_tensors);
    write_u64le(&buf[48], em_data_off(n_tensors));
    write_u64le(&buf[56], (uint64_t)n_tensors * EM_TENSOR_BYTES);
    write_u64le(&buf[64], shard_off);
    write_u64le(&buf[72], shard_size);
    (void)n_tiles;
    (void)n_layers;
}

static void em_write_config(unsigned char *buf,
                            uint32_t n_tiles,
                            uint32_t n_layers)
{
    /* vocab=4, n_layers=param, n_heads=1, d_model=4, d_ff=4,
       max_seq_len=8, rope_dim=2, n_tiles=param, shard_count=0 */
    write_u32le(&buf[0],  4u);
    write_u32le(&buf[4],  n_layers);
    write_u32le(&buf[8],  1u);
    write_u32le(&buf[12], 4u);
    write_u32le(&buf[16], 4u);
    write_u32le(&buf[20], 8u);
    write_u32le(&buf[24], 2u);
    write_u32le(&buf[28], n_tiles);
    write_u32le(&buf[32], 0u);
}

static void em_write_desc(unsigned char *buf,
                          uint32_t tensor_idx,
                          const char *name)
{
    uint64_t offset = (uint64_t)tensor_idx * EM_TENSOR_BYTES;

    memset(buf, 0, EM_DESC_SIZE);
    (void)snprintf((char *)buf, ATT1_MODEL_NAME_SIZE, "%s", name);
    write_u32le(&buf[64], (uint32_t)ATT1_MODEL_DTYPE_F32);
    write_u32le(&buf[68], 1u);
    write_u64le(&buf[72], 4u);   /* shape[0] */
    write_u64le(&buf[80], 1u);
    write_u64le(&buf[88], 1u);
    write_u64le(&buf[96], 1u);
    write_u64le(&buf[104], offset);
    write_u64le(&buf[112], EM_TENSOR_BYTES);
}

static void em_write_data(unsigned char *buf, uint32_t tensor_idx)
{
    uint32_t i;
    for (i = 0u; i < 4u; i++) {
        write_f32le(&buf[i * 4u],
                    (float)(tensor_idx + 1u) * 0.01f + (float)i * 0.001f);
    }
}

static void em_write_shard(unsigned char *buf,
                           uint32_t tensor_idx,
                           uint32_t tile_id)
{
    uint64_t byte_offset = (uint64_t)tensor_idx * EM_TENSOR_BYTES;
    uint32_t owner_aimu  = (tile_id == ATT1_SHARD_TILE_UNASSIGNED) ? 0u : tile_id;
    uint32_t j;

    memset(buf, 0, EM_SHARD_SIZE);
    write_u32le(&buf[0],  tensor_idx);
    write_u32le(&buf[4],  tile_id);
    write_u64le(&buf[8],  byte_offset);
    write_u64le(&buf[16], 4u);  /* shape[0] */
    write_u64le(&buf[24], 1u);
    write_u64le(&buf[32], 1u);
    write_u64le(&buf[40], 1u);
    write_u32le(&buf[48], ATT1_SHARD_DTYPE_F32);
    write_u32le(&buf[52], ATT1_SHARD_QUANT_NONE);
    write_u32le(&buf[56], owner_aimu);
    write_u32le(&buf[60], ATT1_SHARD_REPL_NONE);
    for (j = 0u; j < 8u; j++) {
        write_u32le(&buf[64u + j * 4u], 0u);
    }
    write_u32le(&buf[96],  0u);
    write_u32le(&buf[100], 0u);
    write_u32le(&buf[104], ATT1_SHARD_REDUCE_NONE);
    write_u32le(&buf[108], 0u);
    write_u64le(&buf[112], 0u);
}

static unsigned char *em_build(uint32_t n_tiles,
                               uint32_t n_layers,
                               const em_tensor_spec *specs,
                               uint32_t n_tensors,
                               int with_shard)
{
    uint32_t       file_size = with_shard ? em_file_size(n_tensors)
                                          : em_shard_off(n_tensors);
    unsigned char *buf = calloc(file_size, 1u);
    uint32_t       i;

    if (buf == NULL) {
        return NULL;
    }

    em_write_header(buf, n_tensors, n_tiles, n_layers, with_shard);
    em_write_config(&buf[EM_HEADER_SIZE], n_tiles, n_layers);

    for (i = 0u; i < n_tensors; i++) {
        em_write_desc(&buf[em_desc_off(n_tensors) + i * EM_DESC_SIZE],
                      i, specs[i].name);
        em_write_data(&buf[em_data_off(n_tensors) + i * EM_TENSOR_BYTES], i);
        if (with_shard) {
            em_write_shard(&buf[em_shard_off(n_tensors) + i * EM_SHARD_SIZE],
                           i, specs[i].tile_id);
        }
    }

    return buf;
}

static int em_load(const char *path,
                   uint32_t n_tiles,
                   uint32_t n_layers,
                   const em_tensor_spec *specs,
                   uint32_t n_tensors,
                   int with_shard,
                   att1_model *model)
{
    uint32_t file_size = with_shard ? em_file_size(n_tensors)
                                    : em_shard_off(n_tensors);
    unsigned char *buf = em_build(n_tiles, n_layers, specs, n_tensors,
                                  with_shard);

    if (buf == NULL) {
        return -1;
    }
    if (write_file(path, buf, file_size) != 0) {
        free(buf);
        return -1;
    }
    free(buf);
    return (att1_model_load(path, model) == ATT1_OK) ? 0 : -1;
}

/* ── helpers ──────────────────────────────────────────────────────────────── */

static int logits_close(const float *a, const float *b,
                        size_t count, float tol)
{
    size_t i;
    for (i = 0u; i < count; i++) {
        float diff = a[i] - b[i];
        if (diff < -tol || diff > tol) {
            return 0;
        }
    }
    return 1;
}

static int buffer_contains(const char *buf, const char *needle)
{
    return strstr(buf, needle) != NULL;
}

/* ── test 1: default cluster mode (runtime plan) ──────────────────────────── */

/*
 * att1_cluster_infer_create with the zero-initialised config (shard_plan_mode=0)
 * must succeed on the baseline dummy model (no metadata).
 */
static int test_default_uses_runtime(void)
{
    att1_model                model;
    att1_cluster_infer_config config;
    att1_cluster_infer_t     *infer = NULL;
    int                       ok    = 0;

    if (att1_model_load("models/dummy/model.att1", &model) != ATT1_OK) {
        fputs("test_default_uses_runtime: model load failed\n", stderr);
        return 0;
    }

    memset(&config, 0, sizeof(config));
    config.tile_count = 2u;
    /* shard_plan_mode == 0 == ATT1_SHARD_PLAN_RUNTIME */

    ok = (att1_cluster_infer_create(&model, &config, &infer) == ATT1_OK);

    if (!ok) {
        fputs("test_default_uses_runtime: create failed unexpectedly\n", stderr);
    }

    att1_cluster_infer_destroy(infer);
    att1_model_free(&model);
    return ok;
}

/* ── test 2: metadata plan executes successfully and matches runtime logits ── */

/*
 * Load the shard_meta fixture (n_tiles=1, n_layers=2, 21 tensors all on
 * tile 0). Create cluster with ATT1_SHARD_PLAN_METADATA and tile_count=1.
 * Run one decode token and compare logits with single inference.
 */
static int test_metadata_plan_matches_runtime(void)
{
    att1_model                model;
    att1_cluster_infer_config meta_cfg;
    att1_cluster_infer_t     *meta_cluster  = NULL;
    att1_infer_t             *single        = NULL;
    const float              *meta_logits   = NULL;
    const float              *single_logits = NULL;
    size_t                    meta_count    = 0u;
    size_t                    single_count  = 0u;
    uint32_t                  meta_tok      = 0u;
    uint32_t                  single_tok    = 0u;
    int                       ok            = 0;

    if (att1_model_load("models/shard_meta/model.att1", &model) != ATT1_OK) {
        fputs("test_metadata_plan_matches_runtime: model load failed\n", stderr);
        return 0;
    }

    memset(&meta_cfg, 0, sizeof(meta_cfg));
    meta_cfg.tile_count      = 1u;
    meta_cfg.shard_plan_mode = ATT1_SHARD_PLAN_METADATA;

    if (att1_cluster_infer_create(&model, &meta_cfg, &meta_cluster) != ATT1_OK) {
        fputs("test_metadata_plan_matches_runtime: cluster create failed\n", stderr);
        att1_model_free(&model);
        return 0;
    }

    if (att1_infer_create(&model, &single) != ATT1_OK) {
        fputs("test_metadata_plan_matches_runtime: single create failed\n", stderr);
        att1_cluster_infer_destroy(meta_cluster);
        att1_model_free(&model);
        return 0;
    }

    if ((att1_cluster_infer_decode_token(meta_cluster, 'A', &meta_tok) != ATT1_OK) ||
        (att1_infer_decode_token(single, 'A', &single_tok) != ATT1_OK)) {
        fputs("test_metadata_plan_matches_runtime: decode failed\n", stderr);
        goto cleanup;
    }

    meta_logits   = att1_cluster_infer_logits(meta_cluster, &meta_count);
    single_logits = att1_infer_logits(single, &single_count);

    if ((meta_logits == NULL) || (single_logits == NULL) ||
        (meta_count != model.config.vocab_size) ||
        (single_count != model.config.vocab_size)) {
        fputs("test_metadata_plan_matches_runtime: logit count mismatch\n", stderr);
        goto cleanup;
    }

    if (!logits_close(meta_logits, single_logits,
                      model.config.vocab_size, 0.000001f)) {
        fputs("test_metadata_plan_matches_runtime: logits differ\n", stderr);
        goto cleanup;
    }

    ok = 1;

cleanup:
    att1_infer_destroy(single);
    att1_cluster_infer_destroy(meta_cluster);
    att1_model_free(&model);
    return ok;
}

/* ── test 3: metadata absent + ATT1_SHARD_PLAN_METADATA → fail ────────────── */

static int test_metadata_absent_fails(void)
{
    att1_model                model;
    att1_cluster_infer_config config;
    att1_cluster_infer_t     *infer  = NULL;
    att1_status_t             status;
    int                       ok     = 0;

    if (att1_model_load("models/dummy/model.att1", &model) != ATT1_OK) {
        fputs("test_metadata_absent_fails: model load failed\n", stderr);
        return 0;
    }

    memset(&config, 0, sizeof(config));
    config.tile_count      = 1u;
    config.shard_plan_mode = ATT1_SHARD_PLAN_METADATA;

    status = att1_cluster_infer_create(&model, &config, &infer);
    ok = (status != ATT1_OK);

    if (!ok) {
        fputs("test_metadata_absent_fails: expected failure but got OK\n",
              stderr);
    }

    att1_cluster_infer_destroy(infer);
    att1_model_free(&model);
    return ok;
}

/* ── test 4: conflicting metadata + ATT1_SHARD_PLAN_METADATA → fail ─────── */

/*
 * Build a 2-tensor / 1-layer model where both tensors are in layer 0 but
 * have different tile_id values.  Request metadata plan → must fail.
 */
static int test_metadata_conflict_fails(void)
{
    char path[256];
    att1_model                model;
    att1_cluster_infer_config config;
    att1_cluster_infer_t     *infer  = NULL;
    att1_status_t             status;
    int                       ok     = 0;

    /* n_tiles=2 so tile_id=1 passes M38 validation */
    static const em_tensor_spec specs[] = {
        { "layers.0.wq", 0u },
        { "layers.0.wk", 1u }
    };

    (void)snprintf(path, sizeof(path), OUT_DIR "/conflict.att1");

    if (em_load(path, 2u, 1u, specs, 2u, 1, &model) != 0) {
        fputs("test_metadata_conflict_fails: model load failed\n", stderr);
        return 0;
    }

    memset(&config, 0, sizeof(config));
    config.tile_count      = 2u;
    config.shard_plan_mode = ATT1_SHARD_PLAN_METADATA;

    status = att1_cluster_infer_create(&model, &config, &infer);
    ok = (status != ATT1_OK);

    if (!ok) {
        fputs("test_metadata_conflict_fails: expected failure but got OK\n",
              stderr);
    }

    att1_cluster_infer_destroy(infer);
    att1_model_free(&model);
    return ok;
}

/* ── test 5: incomplete metadata (missing layer) → fail ─────────────────── */

/*
 * 2-layer model where layer 1 tensor has tile_id == UNASSIGNED.
 * Request metadata plan → must fail (proposed.count < n_layers).
 */
static int test_metadata_missing_layer_fails(void)
{
    char path[256];
    att1_model                model;
    att1_cluster_infer_config config;
    att1_cluster_infer_t     *infer  = NULL;
    att1_status_t             status;
    int                       ok     = 0;

    static const em_tensor_spec specs[] = {
        { "layers.0.weight", 0u },
        { "layers.1.weight", ATT1_SHARD_TILE_UNASSIGNED }
    };

    (void)snprintf(path, sizeof(path), OUT_DIR "/missing_layer.att1");

    if (em_load(path, 1u, 2u, specs, 2u, 1, &model) != 0) {
        fputs("test_metadata_missing_layer_fails: model load failed\n", stderr);
        return 0;
    }

    memset(&config, 0, sizeof(config));
    config.tile_count      = 1u;
    config.shard_plan_mode = ATT1_SHARD_PLAN_METADATA;

    status = att1_cluster_infer_create(&model, &config, &infer);
    ok = (status != ATT1_OK);

    if (!ok) {
        fputs("test_metadata_missing_layer_fails: expected failure but got OK\n",
              stderr);
    }

    att1_cluster_infer_destroy(infer);
    att1_model_free(&model);
    return ok;
}

/* ── test 6: no silent fallback when metadata plan fails ─────────────────── */

/*
 * Confirm that when ATT1_SHARD_PLAN_METADATA fails (metadata absent), the
 * returned status is NOT ATT1_OK.  The caller must handle the error; there
 * is no automatic fallback to the runtime plan.
 */
static int test_no_silent_fallback(void)
{
    att1_model                model;
    att1_cluster_infer_config config;
    att1_cluster_infer_t     *infer  = NULL;
    att1_status_t             status;
    int                       ok     = 0;

    /* dummy model has no shard metadata */
    if (att1_model_load("models/dummy/model.att1", &model) != ATT1_OK) {
        fputs("test_no_silent_fallback: model load failed\n", stderr);
        return 0;
    }

    memset(&config, 0, sizeof(config));
    config.tile_count      = 2u;
    config.shard_plan_mode = ATT1_SHARD_PLAN_METADATA;

    status = att1_cluster_infer_create(&model, &config, &infer);

    /* Must fail AND must not have silently returned a working infer object. */
    ok = (status != ATT1_OK) && (infer == NULL);

    if (!ok) {
        fprintf(stderr,
                "test_no_silent_fallback: status=%d infer=%p (both must fail)\n",
                (int)status, (void *)infer);
    }

    att1_cluster_infer_destroy(infer);
    att1_model_free(&model);
    return ok;
}

/* ── test 7: bench reports shard_plan=runtime ─────────────────────────────── */

static int test_bench_reports_runtime(void)
{
    char   out_path[256];
    char   output[4096];
    FILE  *fp  = NULL;
    size_t nr  = 0u;

    (void)snprintf(out_path, sizeof(out_path),
                   OUT_DIR "/bench_runtime.txt");

    {
        char cmd[640];
        (void)snprintf(cmd, sizeof(cmd),
                       "build/att1-bench --model models/dummy/model.att1"
                       " --prompt hi --tokens 1 --mode cluster --tiles 2"
                       " > %s 2>&1",
                       out_path);
        if (system(cmd) != 0) {
            fputs("test_bench_reports_runtime: bench failed\n", stderr);
            return 0;
        }
    }

    fp = fopen(out_path, "rb");
    if (fp == NULL) {
        return 0;
    }
    nr = fread(output, 1u, sizeof(output) - 1u, fp);
    output[nr] = '\0';
    fclose(fp);

    if (!buffer_contains(output, "shard_plan=runtime")) {
        fputs("test_bench_reports_runtime: shard_plan=runtime not found\n",
              stderr);
        fputs(output, stderr);
        return 0;
    }
    return 1;
}

/* ── test 8: bench with --shard-plan metadata reports shard_plan=metadata ─── */

static int test_bench_reports_metadata(void)
{
    char   out_path[256];
    char   output[4096];
    FILE  *fp  = NULL;
    size_t nr  = 0u;

    (void)snprintf(out_path, sizeof(out_path),
                   OUT_DIR "/bench_metadata.txt");

    {
        char cmd[640];
        /* Use the shard fixture with tile_count=1, metadata plan mode. */
        (void)snprintf(cmd, sizeof(cmd),
                       "build/att1-bench"
                       " --model models/shard_meta/model.att1"
                       " --prompt hi --tokens 1 --mode cluster --tiles 1"
                       " --shard-plan metadata"
                       " > %s 2>&1",
                       out_path);
        if (system(cmd) != 0) {
            fputs("test_bench_reports_metadata: bench failed\n", stderr);
            return 0;
        }
    }

    fp = fopen(out_path, "rb");
    if (fp == NULL) {
        return 0;
    }
    nr = fread(output, 1u, sizeof(output) - 1u, fp);
    output[nr] = '\0';
    fclose(fp);

    if (!buffer_contains(output, "shard_plan=metadata")) {
        fputs("test_bench_reports_metadata: shard_plan=metadata not found\n",
              stderr);
        fputs(output, stderr);
        return 0;
    }
    return 1;
}

/* ── main ─────────────────────────────────────────────────────────────────── */

#define RUN(fn) \
    do { \
        if (!(fn())) { \
            fprintf(stderr, "%s failed\n", #fn); \
            return 1; \
        } \
    } while (0)

int main(void)
{
    (void)mkdir(OUT_DIR, 0777);

    RUN(test_default_uses_runtime);
    RUN(test_metadata_plan_matches_runtime);
    RUN(test_metadata_absent_fails);
    RUN(test_metadata_conflict_fails);
    RUN(test_metadata_missing_layer_fails);
    RUN(test_no_silent_fallback);
    RUN(test_bench_reports_runtime);
    RUN(test_bench_reports_metadata);

    puts("shard_meta_exec test passed");
    return 0;
}
