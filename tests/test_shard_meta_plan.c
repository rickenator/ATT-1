/*
 * test_shard_meta_plan.c — Milestone 39: metadata-driven shard plan proposal
 *
 * Tests for att1_meta_plan_build() and att1_meta_plan_compare().
 * The proposed plan is advisory only; inference uses the runtime plan.
 */

#include "att1_model.h"
#include "att1_shard.h"
#include "att1_shard_meta.h"
#include "att1_status.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#define OUT_DIR "build/shard_meta_plan_test"

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

/* ── multi-tensor model builder ───────────────────────────────────────────── */
/*
 * Layout for N tensors:
 *   header  @  0             (80 bytes)
 *   config  @  80            (36 bytes)
 *   descs   @  116           (N × 128 bytes)
 *   data    @  116 + N×128   (N × 16 bytes, each tensor = 4 × float32)
 *   shards  @  116 + N×128 + N×16  (N × 120 bytes)
 *
 * Each tensor descriptor: shape=[4,1,1,1], ndims=1, dtype=f32, nbytes=16
 * tensor[i].offset = i × 16
 */

#define PM_HEADER_SIZE   80u
#define PM_CONFIG_SIZE   36u
#define PM_DESC_SIZE    128u
#define PM_SHARD_SIZE   120u
#define PM_TENSOR_BYTES  16u   /* 4 × float32 */

static uint32_t pm_desc_off(uint32_t n)   { (void)n; return PM_HEADER_SIZE + PM_CONFIG_SIZE; }
static uint32_t pm_data_off(uint32_t n)   { return pm_desc_off(n) + n * PM_DESC_SIZE; }
static uint32_t pm_shard_off(uint32_t n)  { return pm_data_off(n) + n * PM_TENSOR_BYTES; }
static uint32_t pm_file_size(uint32_t n)  { return pm_shard_off(n) + n * PM_SHARD_SIZE; }

/*
 * Tensor spec: name + proposed tile_id for the shard record.
 * Use ATT1_SHARD_TILE_UNASSIGNED to mark a tensor as unassigned.
 */
typedef struct pm_tensor_spec {
    const char *name;
    uint32_t    tile_id;
} pm_tensor_spec;

static void pm_write_header(unsigned char *buf,
                            uint32_t n_tensors,
                            uint32_t n_tiles,
                            uint32_t n_layers)
{
    uint64_t shard_off  = pm_shard_off(n_tensors);
    uint64_t shard_size = (uint64_t)n_tensors * PM_SHARD_SIZE;

    memcpy(&buf[0], ATT1_MODEL_MAGIC, ATT1_MODEL_MAGIC_SIZE);
    write_u32le(&buf[8],  ATT1_MODEL_VERSION);
    write_u32le(&buf[12], ATT1_MODEL_HEADER_SIZE);
    write_u64le(&buf[16], PM_HEADER_SIZE);         /* config_offset */
    write_u64le(&buf[24], PM_CONFIG_SIZE);         /* config_size */
    write_u64le(&buf[32], pm_desc_off(n_tensors)); /* desc_offset */
    write_u64le(&buf[40], n_tensors);              /* tensor_count */
    write_u64le(&buf[48], pm_data_off(n_tensors)); /* data_offset */
    write_u64le(&buf[56], (uint64_t)n_tensors * PM_TENSOR_BYTES); /* data_size */
    write_u64le(&buf[64], shard_off);
    write_u64le(&buf[72], shard_size);
    (void)n_tiles;   /* stored in config */
    (void)n_layers;  /* stored in config */
}

static void pm_write_config(unsigned char *buf,
                            uint32_t n_tiles,
                            uint32_t n_layers)
{
    /* vocab=4, n_layers=param, n_heads=1, d_model=4, d_ff=4,
       max_seq_len=4, rope_dim=2, n_tiles=param, shard_count=0 */
    write_u32le(&buf[0],  4u);
    write_u32le(&buf[4],  n_layers);
    write_u32le(&buf[8],  1u);
    write_u32le(&buf[12], 4u);
    write_u32le(&buf[16], 4u);
    write_u32le(&buf[20], 4u);
    write_u32le(&buf[24], 2u);
    write_u32le(&buf[28], n_tiles);
    write_u32le(&buf[32], 0u);
}

static void pm_write_desc(unsigned char *buf,
                          uint32_t tensor_idx,
                          const char *name)
{
    /* tensor[i].offset = i × 16 in data section */
    uint64_t offset = (uint64_t)tensor_idx * PM_TENSOR_BYTES;

    memset(buf, 0, PM_DESC_SIZE);
    (void)snprintf((char *)buf, ATT1_MODEL_NAME_SIZE, "%s", name);
    write_u32le(&buf[64], (uint32_t)ATT1_MODEL_DTYPE_F32);
    write_u32le(&buf[68], 1u);               /* ndims */
    write_u64le(&buf[72], 4u);               /* shape[0] */
    write_u64le(&buf[80], 1u);               /* shape[1] */
    write_u64le(&buf[88], 1u);               /* shape[2] */
    write_u64le(&buf[96], 1u);               /* shape[3] */
    write_u64le(&buf[104], offset);
    write_u64le(&buf[112], PM_TENSOR_BYTES);
}

static void pm_write_data(unsigned char *buf, uint32_t tensor_idx)
{
    uint32_t i;

    for (i = 0u; i < 4u; i++) {
        write_f32le(&buf[i * 4u],
                    (float)(tensor_idx + 1u) * 0.01f + (float)i * 0.001f);
    }
}

static void pm_write_shard(unsigned char *buf,
                           uint32_t tensor_idx,
                           uint32_t tile_id)
{
    uint64_t byte_offset = (uint64_t)tensor_idx * PM_TENSOR_BYTES;
    uint32_t owner_aimu  = (tile_id == ATT1_SHARD_TILE_UNASSIGNED) ? 0u : tile_id;
    uint32_t j;

    memset(buf, 0, PM_SHARD_SIZE);
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
    write_u32le(&buf[108], 0u);   /* _reserved */
    write_u64le(&buf[112], 0u);   /* checksum */
}

/* Build and return a buffer for a model with n_tensors tensors. */
static unsigned char *pm_build(uint32_t n_tiles,
                               uint32_t n_layers,
                               const pm_tensor_spec *specs,
                               uint32_t n_tensors)
{
    uint32_t       file_size = pm_file_size(n_tensors);
    unsigned char *buf = calloc(file_size, 1u);
    uint32_t       i;

    if (buf == NULL) {
        return NULL;
    }

    pm_write_header(buf, n_tensors, n_tiles, n_layers);
    pm_write_config(&buf[PM_HEADER_SIZE], n_tiles, n_layers);

    for (i = 0u; i < n_tensors; i++) {
        pm_write_desc(&buf[pm_desc_off(n_tensors) + i * PM_DESC_SIZE],
                      i, specs[i].name);
        pm_write_data(&buf[pm_data_off(n_tensors) + i * PM_TENSOR_BYTES], i);
        pm_write_shard(&buf[pm_shard_off(n_tensors) + i * PM_SHARD_SIZE],
                       i, specs[i].tile_id);
    }

    return buf;
}

static int pm_load(const char *path,
                   uint32_t n_tiles,
                   uint32_t n_layers,
                   const pm_tensor_spec *specs,
                   uint32_t n_tensors,
                   att1_model *model)
{
    unsigned char *buf = pm_build(n_tiles, n_layers, specs, n_tensors);

    if (buf == NULL) {
        return -1;
    }
    if (write_file(path, buf, pm_file_size(n_tensors)) != 0) {
        free(buf);
        return -1;
    }
    free(buf);
    return (att1_model_load(path, model) == ATT1_OK) ? 0 : -1;
}

/* ── output capture helper ────────────────────────────────────────────────── */

static int buffer_contains(const char *buf, const char *needle)
{
    return strstr(buf, needle) != NULL;
}

/* ── test cases ───────────────────────────────────────────────────────────── */

/*
 * 1. Absent metadata: proposed plan has count=0, extra=0, conflict=0.
 */
static int test_absent_metadata(void)
{
    att1_model     model;
    att1_meta_plan proposed;
    int            ok = 0;

    if (att1_model_load("models/dummy/model.att1", &model) != ATT1_OK) {
        fputs("test_absent_metadata: model load failed\n", stderr);
        return 0;
    }

    if (att1_meta_plan_build(&model, &proposed) != ATT1_OK) {
        fputs("test_absent_metadata: plan build failed\n", stderr);
        att1_model_free(&model);
        return 0;
    }

    ok = (proposed.count == 0u) &&
         (proposed.extra == 0u) &&
         (proposed.conflict == 0u);

    if (!ok) {
        fprintf(stderr,
                "test_absent_metadata: expected count=0 extra=0 conflict=0,"
                " got count=%u extra=%u conflict=%u\n",
                proposed.count, proposed.extra, proposed.conflict);
    }

    att1_meta_plan_free(&proposed);
    att1_model_free(&model);
    return ok;
}

/*
 * 2. Consistent proposed plan: fixture model with 2 layers, all tensors on
 *    tile 0.  Compare against runtime plan → matching=2, no mismatches.
 */
static int test_consistent_plan(void)
{
    att1_model          model;
    att1_meta_plan      proposed;
    att1_shard_plan     runtime;
    att1_meta_plan_diff diff;
    int                 ok = 0;

    if (att1_model_load("models/shard_meta/model.att1", &model) != ATT1_OK) {
        fputs("test_consistent_plan: model load failed\n", stderr);
        return 0;
    }

    if (att1_meta_plan_build(&model, &proposed) != ATT1_OK) {
        fputs("test_consistent_plan: plan build failed\n", stderr);
        att1_model_free(&model);
        return 0;
    }

    /* Fixture: 2 layers, 21 tensors (18 layer + 3 non-layer). */
    if ((proposed.count != 2u) || (proposed.extra != 3u) ||
        (proposed.conflict != 0u)) {
        fprintf(stderr,
                "test_consistent_plan: expected count=2 extra=3 conflict=0,"
                " got count=%u extra=%u conflict=%u\n",
                proposed.count, proposed.extra, proposed.conflict);
        att1_meta_plan_free(&proposed);
        att1_model_free(&model);
        return 0;
    }

    if (att1_shard_plan_build(&runtime, &model,
                              (size_t)model.config.n_tiles) != ATT1_OK) {
        fputs("test_consistent_plan: runtime plan build failed\n", stderr);
        att1_meta_plan_free(&proposed);
        att1_model_free(&model);
        return 0;
    }

    if (att1_meta_plan_compare(&proposed, &runtime, &diff) != ATT1_OK) {
        fputs("test_consistent_plan: compare failed\n", stderr);
        att1_shard_plan_free(&runtime);
        att1_meta_plan_free(&proposed);
        att1_model_free(&model);
        return 0;
    }

    ok = (diff.matching == 2u) && (diff.mismatch == 0u) &&
         (diff.missing == 0u)  && (diff.extra == 3u) &&
         (diff.conflict == 0u);

    if (!ok) {
        fprintf(stderr,
                "test_consistent_plan: diff matching=%u mismatch=%u"
                " missing=%u extra=%u conflict=%u\n",
                diff.matching, diff.mismatch, diff.missing,
                diff.extra, diff.conflict);
    }

    att1_shard_plan_free(&runtime);
    att1_meta_plan_free(&proposed);
    att1_model_free(&model);
    return ok;
}

/*
 * 3. Missing layer assignment: 2-layer model where layer 1's tensor has
 *    tile_id == ATT1_SHARD_TILE_UNASSIGNED.
 *    Expected: proposed count=1, diff missing=1.
 */
static int test_missing_layer(void)
{
    char path[256];
    att1_model          model;
    att1_meta_plan      proposed;
    att1_shard_plan     runtime;
    att1_meta_plan_diff diff;
    int ok = 0;

    static const pm_tensor_spec specs[] = {
        { "layers.0.weight", 0u },
        { "layers.1.weight", ATT1_SHARD_TILE_UNASSIGNED }
    };

    (void)snprintf(path, sizeof(path), OUT_DIR "/missing_layer.att1");

    if (pm_load(path, 1u, 2u, specs, 2u, &model) != 0) {
        fputs("test_missing_layer: model load failed\n", stderr);
        return 0;
    }

    if (att1_meta_plan_build(&model, &proposed) != ATT1_OK) {
        fputs("test_missing_layer: plan build failed\n", stderr);
        att1_model_free(&model);
        return 0;
    }

    /* Only layer 0 is assigned; extra=0 (both tensors are layer tensors). */
    if ((proposed.count != 1u) || (proposed.extra != 0u) ||
        (proposed.conflict != 0u)) {
        fprintf(stderr,
                "test_missing_layer: expected count=1 extra=0 conflict=0,"
                " got count=%u extra=%u conflict=%u\n",
                proposed.count, proposed.extra, proposed.conflict);
        att1_meta_plan_free(&proposed);
        att1_model_free(&model);
        return 0;
    }

    if (att1_shard_plan_build(&runtime, &model, 1u) != ATT1_OK) {
        fputs("test_missing_layer: runtime plan build failed\n", stderr);
        att1_meta_plan_free(&proposed);
        att1_model_free(&model);
        return 0;
    }

    if (att1_meta_plan_compare(&proposed, &runtime, &diff) != ATT1_OK) {
        fputs("test_missing_layer: compare failed\n", stderr);
        att1_shard_plan_free(&runtime);
        att1_meta_plan_free(&proposed);
        att1_model_free(&model);
        return 0;
    }

    ok = (diff.matching == 1u) && (diff.mismatch == 0u) &&
         (diff.missing == 1u)  && (diff.conflict == 0u);

    if (!ok) {
        fprintf(stderr,
                "test_missing_layer: diff matching=%u mismatch=%u"
                " missing=%u extra=%u conflict=%u\n",
                diff.matching, diff.mismatch, diff.missing,
                diff.extra, diff.conflict);
    }

    att1_shard_plan_free(&runtime);
    att1_meta_plan_free(&proposed);
    att1_model_free(&model);
    return ok;
}

/*
 * 4. Conflicting tile ownership: 1-layer model with 2 tensors in the same
 *    layer but different tile_id values.
 *    Expected: proposed conflict=1.
 */
static int test_tile_conflict(void)
{
    char path[256];
    att1_model     model;
    att1_meta_plan proposed;
    int            ok = 0;

    /* Two tensors both in layer 0 but assigned to different tiles. */
    static const pm_tensor_spec specs[] = {
        { "layers.0.wq", 0u },
        { "layers.0.wk", 1u }
    };

    (void)snprintf(path, sizeof(path), OUT_DIR "/tile_conflict.att1");

    /* n_tiles=2 so tile_id=1 passes M38 validation */
    if (pm_load(path, 2u, 1u, specs, 2u, &model) != 0) {
        fputs("test_tile_conflict: model load failed\n", stderr);
        return 0;
    }

    if (att1_meta_plan_build(&model, &proposed) != ATT1_OK) {
        fputs("test_tile_conflict: plan build failed\n", stderr);
        att1_model_free(&model);
        return 0;
    }

    ok = (proposed.count == 1u) && (proposed.conflict == 1u) &&
         (proposed.extra == 0u);

    if (!ok) {
        fprintf(stderr,
                "test_tile_conflict: expected count=1 conflict=1 extra=0,"
                " got count=%u conflict=%u extra=%u\n",
                proposed.count, proposed.conflict, proposed.extra);
    }

    att1_meta_plan_free(&proposed);
    att1_model_free(&model);
    return ok;
}

/*
 * 5. att1-inspect on the shard fixture includes plan summary lines.
 */
static int test_inspect_plan_output(void)
{
    char   out_path[256];
    char   output[16384];
    FILE  *fp = NULL;
    size_t nread = 0u;

    (void)snprintf(out_path, sizeof(out_path),
                   OUT_DIR "/inspect_plan.txt");

    {
        char cmd[640];

        (void)snprintf(cmd, sizeof(cmd),
                       "build/att1-inspect models/shard_meta/model.att1"
                       " > %s 2>&1",
                       out_path);
        if (system(cmd) != 0) {
            fputs("test_inspect_plan_output: inspect failed\n", stderr);
            return 0;
        }
    }

    fp = fopen(out_path, "rb");
    if (fp == NULL) {
        fputs("test_inspect_plan_output: cannot open output\n", stderr);
        return 0;
    }

    nread = fread(output, 1u, sizeof(output) - 1u, fp);
    output[nread] = '\0';
    fclose(fp);

    if (!buffer_contains(output, "shard_meta_plan_entries=2")) {
        fputs("test_inspect_plan_output: missing shard_meta_plan_entries=2\n",
              stderr);
        return 0;
    }
    if (!buffer_contains(output, "shard_meta_plan_matching=2")) {
        fputs("test_inspect_plan_output: missing shard_meta_plan_matching=2\n",
              stderr);
        return 0;
    }
    if (!buffer_contains(output, "shard_meta_plan_missing=0")) {
        fputs("test_inspect_plan_output: missing shard_meta_plan_missing=0\n",
              stderr);
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

    RUN(test_absent_metadata);
    RUN(test_consistent_plan);
    RUN(test_missing_layer);
    RUN(test_tile_conflict);
    RUN(test_inspect_plan_output);

    puts("shard_meta_plan test passed");
    return 0;
}
