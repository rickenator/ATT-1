#include "att1_model.h"
#include "att1_shard_meta.h"
#include "att1_status.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

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

/* ── file I/O ─────────────────────────────────────────────────────────────── */

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

/* ── minimal 1-tensor model layout ───────────────────────────────────────── */
/*
 * "w.weight"  ndims=1  shape=[4,1,1,1]  dtype=f32  nbytes=16
 *
 * Offsets:
 *   header  @   0  (80 bytes)
 *   config  @  80  (36 bytes)
 *   desc    @ 116  (1 × 128 bytes)
 *   data    @ 244  (16 bytes)
 *   shard   @ 260  (1 × 120 bytes)
 */
#define C_CONFIG_OFF  80u
#define C_DESC_OFF   116u
#define C_DATA_OFF   244u
#define C_DATA_SIZE   16u
#define C_SHARD_OFF  260u
#define C_SHARD_SIZE 120u
#define C_FILE_SIZE  380u   /* header + config + desc + data + shard */

static void c_write_header(unsigned char *buf,
                           uint32_t n_tiles,
                           uint64_t shard_off,
                           uint64_t shard_sz)
{
    memcpy(&buf[0], ATT1_MODEL_MAGIC, ATT1_MODEL_MAGIC_SIZE);
    write_u32le(&buf[8],  ATT1_MODEL_VERSION);
    write_u32le(&buf[12], ATT1_MODEL_HEADER_SIZE);
    write_u64le(&buf[16], C_CONFIG_OFF);
    write_u64le(&buf[24], ATT1_MODEL_CONFIG_SIZE);
    write_u64le(&buf[32], C_DESC_OFF);
    write_u64le(&buf[40], 1u);               /* tensor_count */
    write_u64le(&buf[48], C_DATA_OFF);
    write_u64le(&buf[56], C_DATA_SIZE);
    write_u64le(&buf[64], shard_off);
    write_u64le(&buf[72], shard_sz);
    (void)n_tiles;  /* stored in config, not header */
}

static void c_write_config(unsigned char *buf, uint32_t n_tiles)
{
    /* vocab=4, layers=1, heads=1, d_model=4, d_ff=4,
       max_seq=4, rope_dim=2, n_tiles=<param>, shard_count=0 */
    write_u32le(&buf[0],  4u);
    write_u32le(&buf[4],  1u);
    write_u32le(&buf[8],  1u);
    write_u32le(&buf[12], 4u);
    write_u32le(&buf[16], 4u);
    write_u32le(&buf[20], 4u);
    write_u32le(&buf[24], 2u);
    write_u32le(&buf[28], n_tiles);
    write_u32le(&buf[32], 0u);
}

static void c_write_desc(unsigned char *buf)
{
    memset(buf, 0, ATT1_MODEL_TENSOR_DESC_SIZE);
    (void)snprintf((char *)buf, ATT1_MODEL_NAME_SIZE, "w.weight");
    write_u32le(&buf[64], (uint32_t)ATT1_MODEL_DTYPE_F32);  /* dtype */
    write_u32le(&buf[68], 1u);                              /* ndims */
    write_u64le(&buf[72], 4u);  /* shape[0] */
    write_u64le(&buf[80], 1u);  /* shape[1] */
    write_u64le(&buf[88], 1u);  /* shape[2] */
    write_u64le(&buf[96], 1u);  /* shape[3] */
    write_u64le(&buf[104], 0u); /* offset */
    write_u64le(&buf[112], C_DATA_SIZE); /* nbytes */
}

static void c_write_data(unsigned char *buf)
{
    uint32_t i = 0u;

    for (i = 0u; i < 4u; i++) {
        write_f32le(&buf[i * 4u], 0.01f + ((float)i * 0.001f));
    }
}

/*
 * Write one 120-byte shard record with explicit tile_id, owner_aimu,
 * byte_offset, and dtype (other policy fields are zeroed/valid).
 */
static void c_write_shard(unsigned char *buf,
                          uint32_t tensor_id,
                          uint32_t tile_id,
                          uint32_t owner_aimu,
                          uint64_t byte_offset,
                          uint32_t dtype)
{
    uint32_t j = 0u;

    memset(buf, 0, ATT1_SHARD_META_RECORD_SIZE);
    write_u32le(&buf[0],  tensor_id);
    write_u32le(&buf[4],  tile_id);
    write_u64le(&buf[8],  byte_offset);
    /* shape matches tensor descriptor: [4, 1, 1, 1] */
    write_u64le(&buf[16], 4u);
    write_u64le(&buf[24], 1u);
    write_u64le(&buf[32], 1u);
    write_u64le(&buf[40], 1u);
    write_u32le(&buf[48], dtype);
    write_u32le(&buf[52], ATT1_SHARD_QUANT_NONE);
    write_u32le(&buf[56], owner_aimu);
    write_u32le(&buf[60], ATT1_SHARD_REPL_NONE);
    for (j = 0u; j < 8u; j++) {
        write_u32le(&buf[64u + (j * 4u)], 0u);
    }
    write_u32le(&buf[96],  0u);
    write_u32le(&buf[100], 0u);
    write_u32le(&buf[104], ATT1_SHARD_REDUCE_NONE);
    write_u32le(&buf[108], 0u);  /* _reserved */
    write_u64le(&buf[112], 0u);  /* checksum */
}

/*
 * Allocate and return a C_FILE_SIZE-byte buffer representing a 1-tensor
 * model with the given shard record parameters.
 * Returns NULL on allocation failure.
 */
static unsigned char *build_model(uint32_t n_tiles,
                                  uint32_t tile_id,
                                  uint32_t owner_aimu,
                                  uint64_t byte_offset,
                                  uint32_t dtype)
{
    unsigned char *buf = calloc(C_FILE_SIZE, 1u);

    if (buf == NULL) {
        return NULL;
    }

    c_write_header(&buf[0], n_tiles, C_SHARD_OFF, C_SHARD_SIZE);
    c_write_config(&buf[C_CONFIG_OFF], n_tiles);
    c_write_desc(&buf[C_DESC_OFF]);
    c_write_data(&buf[C_DATA_OFF]);
    c_write_shard(&buf[C_SHARD_OFF],
                  0u,          /* tensor_id */
                  tile_id,
                  owner_aimu,
                  byte_offset,
                  dtype);
    return buf;
}

/* ── convenience: write model file and load it ────────────────────────────── */

static int load_built_model(const char *path,
                            uint32_t n_tiles,
                            uint32_t tile_id,
                            uint32_t owner_aimu,
                            uint64_t byte_offset,
                            uint32_t dtype,
                            att1_model *model)
{
    unsigned char *buf = build_model(n_tiles, tile_id, owner_aimu,
                                     byte_offset, dtype);

    if (buf == NULL) {
        return -1;
    }

    if (write_file(path, buf, C_FILE_SIZE) != 0) {
        free(buf);
        return -1;
    }

    free(buf);
    return (att1_model_load(path, model) == ATT1_OK) ? 0 : -1;
}

/* ── expected-violation helper ────────────────────────────────────────────── */

/*
 * validate_and_check() validates the model's shard metadata and returns 1
 * if exactly expected_count violations are found and the first violation
 * (when expected_count >= 1) has the given tensor_id and field name.
 */
static int validate_and_check(const att1_model *model,
                              uint64_t          expected_count,
                              uint32_t          expected_tensor_id,
                              const char       *expected_field)
{
    att1_shard_meta_validation validation;
    int ok = 0;

    if (att1_shard_meta_validate(&model->shard_meta,
                                 &model->config,
                                 model->tensors,
                                 model->tensor_count,
                                 &validation) != ATT1_OK) {
        fputs("att1_shard_meta_validate returned error\n", stderr);
        return 0;
    }

    if (validation.count != expected_count) {
        fprintf(stderr,
                "expected %" PRIu64 " violations, got %" PRIu64 "\n",
                expected_count, validation.count);
        att1_shard_meta_validation_free(&validation);
        return 0;
    }

    if ((expected_count >= 1u) &&
        (validation.violations[0].tensor_id != expected_tensor_id)) {
        fprintf(stderr,
                "violation[0].tensor_id=%u expected %u\n",
                validation.violations[0].tensor_id,
                expected_tensor_id);
        att1_shard_meta_validation_free(&validation);
        return 0;
    }

    if ((expected_count >= 1u) &&
        (strncmp(validation.violations[0].field,
                 expected_field,
                 ATT1_SHARD_META_VIOLATION_FIELD_SIZE) != 0)) {
        fprintf(stderr,
                "violation[0].field=\"%s\" expected \"%s\"\n",
                validation.violations[0].field,
                expected_field);
        att1_shard_meta_validation_free(&validation);
        return 0;
    }

    ok = 1;
    att1_shard_meta_validation_free(&validation);
    return ok;
}

/* ── test cases ───────────────────────────────────────────────────────────── */

#define OUT_DIR "build/shard_meta_consistency_test"

/*
 * 1. Consistent fixture (21-tensor shard_meta fixture): no violations.
 */
static int test_consistent(void)
{
    att1_model model;
    int ok = 0;

    if (att1_model_load("models/shard_meta/model.att1", &model) != ATT1_OK) {
        fputs("fixture load failed\n", stderr);
        return 0;
    }

    ok = validate_and_check(&model, 0u, 0u, "");
    att1_model_free(&model);
    return ok;
}

/*
 * 2. tile_id >= n_tiles: 1 violation on "tile_id".
 */
static int test_tile_out_of_range(void)
{
    char       path[256];
    att1_model model;
    int        ok = 0;

    (void)snprintf(path, sizeof(path),
                   OUT_DIR "/tile_oob.att1");

    /* n_tiles=1, tile_id=5 → tile_id 5 >= 1 */
    if (load_built_model(path,
                         1u,                   /* n_tiles */
                         5u,                   /* tile_id  */
                         0u,                   /* owner_aimu */
                         0u,                   /* byte_offset */
                         ATT1_SHARD_DTYPE_F32, /* dtype */
                         &model) != 0) {
        fputs("tile_oob model load failed\n", stderr);
        return 0;
    }

    ok = validate_and_check(&model, 1u, 0u, "tile_id");
    att1_model_free(&model);
    return ok;
}

/*
 * 3. owner_aimu >= n_tiles: 1 violation on "owner_aimu".
 */
static int test_aimu_out_of_range(void)
{
    char       path[256];
    att1_model model;
    int        ok = 0;

    (void)snprintf(path, sizeof(path),
                   OUT_DIR "/aimu_oob.att1");

    /* n_tiles=1, owner_aimu=3 → owner_aimu 3 >= 1 */
    if (load_built_model(path,
                         1u,                   /* n_tiles */
                         0u,                   /* tile_id */
                         3u,                   /* owner_aimu */
                         0u,                   /* byte_offset */
                         ATT1_SHARD_DTYPE_F32, /* dtype */
                         &model) != 0) {
        fputs("aimu_oob model load failed\n", stderr);
        return 0;
    }

    ok = validate_and_check(&model, 1u, 0u, "owner_aimu");
    att1_model_free(&model);
    return ok;
}

/*
 * 4. shard dtype=Q8 but tensor dtype=F32: 1 violation on "dtype".
 */
static int test_dtype_mismatch(void)
{
    char       path[256];
    att1_model model;
    int        ok = 0;

    (void)snprintf(path, sizeof(path),
                   OUT_DIR "/dtype_mismatch.att1");

    /* tensor descriptor is f32, shard says q8 */
    if (load_built_model(path,
                         1u,                  /* n_tiles */
                         0u,                  /* tile_id */
                         0u,                  /* owner_aimu */
                         0u,                  /* byte_offset */
                         ATT1_SHARD_DTYPE_Q8, /* dtype — mismatch */
                         &model) != 0) {
        fputs("dtype_mismatch model load failed\n", stderr);
        return 0;
    }

    ok = validate_and_check(&model, 1u, 0u, "dtype");
    att1_model_free(&model);
    return ok;
}

/*
 * 5. byte_offset != tensor descriptor offset: 1 violation on "byte_offset".
 *    (tensor offset = 0; use byte_offset = 9999)
 */
static int test_byte_offset_exceeds(void)
{
    char       path[256];
    att1_model model;
    int        ok = 0;

    (void)snprintf(path, sizeof(path),
                   OUT_DIR "/byte_off_exceeds.att1");

    if (load_built_model(path,
                         1u,                   /* n_tiles */
                         0u,                   /* tile_id */
                         0u,                   /* owner_aimu */
                         9999u,                /* byte_offset > 16 */
                         ATT1_SHARD_DTYPE_F32, /* dtype */
                         &model) != 0) {
        fputs("byte_off_exceeds model load failed\n", stderr);
        return 0;
    }

    ok = validate_and_check(&model, 1u, 0u, "byte_offset");
    att1_model_free(&model);
    return ok;
}

/*
 * 6. No shard metadata (dummy model): validate returns 0 violations.
 */
static int test_absent_metadata(void)
{
    att1_model model;
    int        ok = 0;

    if (att1_model_load("models/dummy/model.att1", &model) != ATT1_OK) {
        fputs("dummy model load failed\n", stderr);
        return 0;
    }

    ok = validate_and_check(&model, 0u, 0u, "");
    att1_model_free(&model);
    return ok;
}

/*
 * 7. att1-inspect on a violation model reports shard_meta_violations.
 */
static int test_inspect_reports_violations(void)
{
    char       model_path[256];
    char       out_path[256];
    char       output[4096];
    unsigned char *buf = NULL;
    FILE *fp = NULL;
    size_t nread = 0u;

    (void)snprintf(model_path, sizeof(model_path),
                   OUT_DIR "/inspect_violation.att1");
    (void)snprintf(out_path, sizeof(out_path),
                   OUT_DIR "/inspect_violation.txt");

    /* tile_id=5, n_tiles=1 → should produce a tile_id violation */
    buf = build_model(1u, 5u, 0u, 0u, ATT1_SHARD_DTYPE_F32);
    if (buf == NULL) {
        return 0;
    }

    if (write_file(model_path, buf, C_FILE_SIZE) != 0) {
        free(buf);
        return 0;
    }
    free(buf);

    {
        char cmd[640];

        (void)snprintf(cmd, sizeof(cmd),
                       "build/att1-inspect %s > %s 2>&1",
                       model_path, out_path);
        /* inspect exits 0 even with violations */
        (void)system(cmd);
    }

    fp = fopen(out_path, "rb");
    if (fp == NULL) {
        return 0;
    }

    nread = fread(output, 1u, sizeof(output) - 1u, fp);
    output[nread] = '\0';
    fclose(fp);

    if (strstr(output, "shard_meta_violations: 1") == NULL ||
        strstr(output, "field=tile_id") == NULL) {
        fputs("inspect did not report tile_id violation\n", stderr);
        fputs(output, stderr);
        return 0;
    }

    return 1;
}

/*
 * 8. att1-inspect on a consistent model reports no shard_meta_violations line.
 */
static int test_inspect_no_violations(void)
{
    char  out_path[256];
    char  output[16384];
    FILE *fp = NULL;
    size_t nread = 0u;

    (void)snprintf(out_path, sizeof(out_path),
                   OUT_DIR "/inspect_consistent.txt");

    {
        char cmd[640];

        (void)snprintf(cmd, sizeof(cmd),
                       "build/att1-inspect models/shard_meta/model.att1"
                       " > %s 2>&1",
                       out_path);
        if (system(cmd) != 0) {
            fputs("inspect on fixture failed\n", stderr);
            return 0;
        }
    }

    fp = fopen(out_path, "rb");
    if (fp == NULL) {
        return 0;
    }

    nread = fread(output, 1u, sizeof(output) - 1u, fp);
    output[nread] = '\0';
    fclose(fp);

    if (strstr(output, "shard_meta_violations") != NULL) {
        fputs("inspect reported violations on consistent fixture\n", stderr);
        fputs(output, stderr);
        return 0;
    }

    if (strstr(output, "shard_meta: 21 records") == NULL) {
        fputs("inspect missing shard_meta summary on fixture\n", stderr);
        return 0;
    }

    return 1;
}

/* ── main ─────────────────────────────────────────────────────────────────── */

int main(void)
{
    (void)mkdir(OUT_DIR, 0777);

#define RUN(name)                                           \
    do {                                                    \
        if (!name()) {                                      \
            fputs(#name " failed\n", stderr);               \
            return 1;                                       \
        }                                                   \
    } while (0)

    RUN(test_consistent);
    RUN(test_tile_out_of_range);
    RUN(test_aimu_out_of_range);
    RUN(test_dtype_mismatch);
    RUN(test_byte_offset_exceeds);
    RUN(test_absent_metadata);
    RUN(test_inspect_reports_violations);
    RUN(test_inspect_no_violations);

#undef RUN

    puts("shard_meta_consistency test passed");
    return 0;
}
