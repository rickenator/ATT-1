#include "att1_model.h"
#include "att1_shard_meta.h"
#include "att1_status.h"

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

/* ── file I/O helpers ─────────────────────────────────────────────────────── */

static int write_file(const char *path, const unsigned char *data, size_t size)
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

/* ── model layout constants ───────────────────────────────────────────────── */

/*
 * Minimal 1-tensor test model:
 *
 *   "w.weight"  ndims=1  shape=[4]  dtype=f32  nbytes=16
 *
 * Offsets:
 *   config  @  80  (36 bytes)
 *   desc    @ 116  (1 × 128 bytes = 128 bytes)
 *   data    @ 244  (16 bytes)
 *   [shard  @ 260  (1 × 120 bytes = 120 bytes)]  — when present
 *
 * File sizes:
 *   without shard metadata:  260 bytes
 *   with    shard metadata:  380 bytes
 */
#define TEST_CONFIG_OFFSET  80u
#define TEST_DESC_OFFSET   116u
#define TEST_DATA_OFFSET   244u
#define TEST_DATA_SIZE      16u   /* 4 f32 elements */
#define TEST_SHARD_OFFSET  260u
#define TEST_SHARD_SIZE    120u   /* 1 record */
#define TEST_FILE_SIZE_NO_META  260u
#define TEST_FILE_SIZE_WITH_META 380u

/*
 * write_header() fills the 80-byte .att1 header.
 * Pass shard_off=0 and shard_sz=0 to omit the shard metadata section.
 */
static void write_header(unsigned char *buf,
                         uint64_t shard_off,
                         uint64_t shard_sz)
{
    memcpy(&buf[0], ATT1_MODEL_MAGIC, ATT1_MODEL_MAGIC_SIZE);
    write_u32le(&buf[8],  ATT1_MODEL_VERSION);
    write_u32le(&buf[12], ATT1_MODEL_HEADER_SIZE);
    write_u64le(&buf[16], TEST_CONFIG_OFFSET);
    write_u64le(&buf[24], ATT1_MODEL_CONFIG_SIZE);
    write_u64le(&buf[32], TEST_DESC_OFFSET);
    write_u64le(&buf[40], 1u);                 /* tensor_count */
    write_u64le(&buf[48], TEST_DATA_OFFSET);
    write_u64le(&buf[56], TEST_DATA_SIZE);
    write_u64le(&buf[64], shard_off);
    write_u64le(&buf[72], shard_sz);
}

static void write_config(unsigned char *buf)
{
    /* vocab=4, layers=1, heads=1, d_model=4, d_ff=4,
       max_seq=4, rope_dim=2, n_tiles=1, shard_count=0 */
    write_u32le(&buf[0],   4u);
    write_u32le(&buf[4],   1u);
    write_u32le(&buf[8],   1u);
    write_u32le(&buf[12],  4u);
    write_u32le(&buf[16],  4u);
    write_u32le(&buf[20],  4u);
    write_u32le(&buf[24],  2u);
    write_u32le(&buf[28],  1u);
    write_u32le(&buf[32],  0u);
}

/*
 * write_tensor_desc() fills the 128-byte tensor descriptor for the single
 * test tensor "w.weight" (ndims=1, shape=[4,1,1,1], dtype=f32, nbytes=16).
 */
static void write_tensor_desc(unsigned char *buf, uint64_t data_rel_off)
{
    uint32_t i = 0u;

    memset(buf, 0, ATT1_MODEL_TENSOR_DESC_SIZE);
    (void)snprintf((char *)buf, ATT1_MODEL_NAME_SIZE, "w.weight");
    write_u32le(&buf[64], ATT1_MODEL_DTYPE_F32);   /* dtype */
    write_u32le(&buf[68], 1u);                     /* ndims */
    /* shape[4]: used dim = 4, padding = 1 */
    write_u64le(&buf[72],  4u);
    write_u64le(&buf[80],  1u);
    write_u64le(&buf[88],  1u);
    write_u64le(&buf[96],  1u);
    write_u64le(&buf[104], data_rel_off);           /* offset (relative to data section) */
    write_u64le(&buf[112], TEST_DATA_SIZE);          /* nbytes */
    for (i = 120u; i < ATT1_MODEL_TENSOR_DESC_SIZE; i++) {
        buf[i] = 0u;
    }
}

/*
 * write_shard_record() fills one 120-byte shard metadata record.
 * All "optional" fields (allowed_ops, routing_requirements, dependency_graph,
 * owner_aimu) are zeroed; checksum is 0 (no verification).
 */
static void write_shard_record(unsigned char *buf,
                                uint32_t tensor_id,
                                uint32_t tile_id,
                                uint32_t dtype,
                                uint32_t quantization,
                                uint32_t repl_policy,
                                uint32_t reduce_behavior,
                                uint32_t reserved_field)
{
    uint32_t j = 0u;

    memset(buf, 0, ATT1_SHARD_META_RECORD_SIZE);
    write_u32le(&buf[0],   tensor_id);
    write_u32le(&buf[4],   tile_id);
    write_u64le(&buf[8],   0u);        /* byte_offset */
    /* shape matches tensor descriptor: [4, 1, 1, 1] */
    write_u64le(&buf[16],  4u);
    write_u64le(&buf[24],  1u);
    write_u64le(&buf[32],  1u);
    write_u64le(&buf[40],  1u);
    write_u32le(&buf[48],  dtype);
    write_u32le(&buf[52],  quantization);
    write_u32le(&buf[56],  0u);        /* owner_aimu */
    write_u32le(&buf[60],  repl_policy);
    for (j = 0u; j < 8u; j++) {       /* dependency_graph[8] */
        write_u32le(&buf[64u + (j * 4u)], 0u);
    }
    write_u32le(&buf[96],  0u);        /* allowed_ops */
    write_u32le(&buf[100], 0u);        /* routing_requirements */
    write_u32le(&buf[104], reduce_behavior);
    write_u32le(&buf[108], reserved_field); /* _reserved */
    write_u64le(&buf[112], 0u);        /* checksum (0 = no verification) */
}

static void write_tensor_data(unsigned char *buf)
{
    uint32_t i = 0u;

    for (i = 0u; i < 4u; i++) {
        write_f32le(&buf[i * 4u], 0.01f + ((float)i * 0.001f));
    }
}

/*
 * build_model_no_meta() allocates and populates a minimal .att1 file with
 * no shard metadata section (shard_offset=0, shard_size=0).
 * Caller must free() the returned pointer.
 */
static unsigned char *build_model_no_meta(void)
{
    unsigned char *buf = calloc(TEST_FILE_SIZE_NO_META, 1u);

    if (buf == NULL) {
        return NULL;
    }

    write_header(&buf[0], 0u, 0u);
    write_config(&buf[TEST_CONFIG_OFFSET]);
    write_tensor_desc(&buf[TEST_DESC_OFFSET], 0u);
    write_tensor_data(&buf[TEST_DATA_OFFSET]);
    return buf;
}

/*
 * build_model_with_meta() allocates and populates a minimal .att1 file
 * with one valid shard metadata record.
 * The caller may patch specific bytes before writing to disk.
 * Caller must free() the returned pointer.
 */
static unsigned char *build_model_with_meta(void)
{
    unsigned char *buf = calloc(TEST_FILE_SIZE_WITH_META, 1u);

    if (buf == NULL) {
        return NULL;
    }

    write_header(&buf[0], TEST_SHARD_OFFSET, TEST_SHARD_SIZE);
    write_config(&buf[TEST_CONFIG_OFFSET]);
    write_tensor_desc(&buf[TEST_DESC_OFFSET], 0u);
    write_tensor_data(&buf[TEST_DATA_OFFSET]);
    write_shard_record(&buf[TEST_SHARD_OFFSET],
                       0u,                        /* tensor_id */
                       0u,                        /* tile_id */
                       ATT1_SHARD_DTYPE_F32,
                       ATT1_SHARD_QUANT_NONE,
                       ATT1_SHARD_REPL_NONE,
                       ATT1_SHARD_REDUCE_NONE,
                       0u);                       /* _reserved = 0 */
    return buf;
}

/* ── test cases ───────────────────────────────────────────────────────────── */

/*
 * 1. Model with no shard metadata loads successfully; shard_meta is empty.
 */
static int test_no_shard_meta(const char *dir)
{
    char path[256];
    unsigned char *buf = NULL;
    att1_model model;
    int ok = 0;

    (void)snprintf(path, sizeof(path), "%s/no_meta.att1", dir);

    buf = build_model_no_meta();
    if (buf == NULL) {
        return 0;
    }

    ok = (write_file(path, buf, TEST_FILE_SIZE_NO_META) == 0);
    free(buf);

    if (!ok) {
        return 0;
    }

    if (att1_model_load(path, &model) != ATT1_OK) {
        return 0;
    }

    ok = (model.shard_meta.count == 0u) && (model.shard_meta.records == NULL);
    att1_model_free(&model);
    return ok;
}

/*
 * 2. Model with one valid shard record loads and parses correctly.
 */
static int test_valid_shard_meta(const char *dir)
{
    char path[256];
    unsigned char *buf = NULL;
    att1_model model;
    int ok = 0;

    (void)snprintf(path, sizeof(path), "%s/valid_meta.att1", dir);

    buf = build_model_with_meta();
    if (buf == NULL) {
        return 0;
    }

    ok = (write_file(path, buf, TEST_FILE_SIZE_WITH_META) == 0);
    free(buf);

    if (!ok) {
        return 0;
    }

    if (att1_model_load(path, &model) != ATT1_OK) {
        return 0;
    }

    ok = (model.shard_meta.count == 1u) &&
         (model.shard_meta.records != NULL) &&
         (model.shard_meta.records[0].tensor_id == 0u) &&
         (model.shard_meta.records[0].tile_id == 0u) &&
         (model.shard_meta.records[0].dtype == ATT1_SHARD_DTYPE_F32) &&
         (model.shard_meta.records[0].quantization == ATT1_SHARD_QUANT_NONE) &&
         (model.shard_meta.records[0].replication_policy == ATT1_SHARD_REPL_NONE) &&
         (model.shard_meta.records[0].reduction_behavior == ATT1_SHARD_REDUCE_NONE) &&
         (model.shard_meta.records[0].shape[0] == 4u) &&
         (model.shard_meta.records[0].shape[1] == 1u);
    att1_model_free(&model);
    return ok;
}

/*
 * 3. Shard metadata section offset+size extends past end of file → bad format.
 *    (bad bounds test)
 */
static int test_bad_bounds(const char *dir)
{
    char path[256];
    unsigned char *buf = NULL;
    att1_model model;
    int ok = 0;

    (void)snprintf(path, sizeof(path), "%s/bad_bounds.att1", dir);

    buf = build_model_with_meta();
    if (buf == NULL) {
        return 0;
    }

    /* Point shard_metadata_size well past end of file. */
    write_u64le(&buf[72], TEST_FILE_SIZE_WITH_META + 1u);

    ok = (write_file(path, buf, TEST_FILE_SIZE_WITH_META) == 0);
    free(buf);

    if (!ok) {
        return 0;
    }

    ok = (att1_model_load(path, &model) != ATT1_OK);
    if (!ok) {
        att1_model_free(&model);
    }
    return ok;
}

/*
 * 4. Shard metadata size is not a multiple of the record size → bad format.
 *    (truncated records test)
 */
static int test_truncated_records(const char *dir)
{
    char path[256];
    unsigned char *buf = NULL;
    att1_model model;
    int ok = 0;

    (void)snprintf(path, sizeof(path), "%s/truncated_meta.att1", dir);

    /* Build a model with a slightly larger file so a shortened shard_size
       still passes the file-bounds check but fails the record-multiple check. */
    buf = calloc(TEST_FILE_SIZE_WITH_META + 1u, 1u);
    if (buf == NULL) {
        return 0;
    }

    /* shard_size = 100, which is not a multiple of 120 */
    write_header(&buf[0], TEST_SHARD_OFFSET, 100u);
    write_config(&buf[TEST_CONFIG_OFFSET]);
    write_tensor_desc(&buf[TEST_DESC_OFFSET], 0u);
    write_tensor_data(&buf[TEST_DATA_OFFSET]);

    ok = (write_file(path, buf, TEST_FILE_SIZE_WITH_META + 1u) == 0);
    free(buf);

    if (!ok) {
        return 0;
    }

    ok = (att1_model_load(path, &model) == ATT1_ERR_BAD_FORMAT);
    if (!ok) {
        att1_model_free(&model);
    }
    return ok;
}

/*
 * 5. Shard metadata section contains the wrong number of records relative to
 *    tensor_count (2 records when 1 tensor exists) → bad format.
 *    (bad version / format mismatch test)
 */
static int test_bad_record_count(const char *dir)
{
    char path[256];
    unsigned char *buf = NULL;
    att1_model model;
    size_t file_size = 0u;
    int ok = 0;

    (void)snprintf(path, sizeof(path), "%s/bad_record_count.att1", dir);

    /* Allocate space for 2 records (240 bytes of shard metadata) */
    file_size = TEST_DATA_OFFSET + TEST_DATA_SIZE + (2u * ATT1_SHARD_META_RECORD_SIZE);
    buf = calloc(file_size, 1u);
    if (buf == NULL) {
        return 0;
    }

    write_header(&buf[0],
                 TEST_SHARD_OFFSET,
                 2u * ATT1_SHARD_META_RECORD_SIZE);
    write_config(&buf[TEST_CONFIG_OFFSET]);
    write_tensor_desc(&buf[TEST_DESC_OFFSET], 0u);
    write_tensor_data(&buf[TEST_DATA_OFFSET]);
    /* Write 2 plausible records, but tensor_count == 1 → mismatch */
    write_shard_record(&buf[TEST_SHARD_OFFSET],
                       0u, 0u,
                       ATT1_SHARD_DTYPE_F32,
                       ATT1_SHARD_QUANT_NONE,
                       ATT1_SHARD_REPL_NONE,
                       ATT1_SHARD_REDUCE_NONE,
                       0u);
    write_shard_record(&buf[TEST_SHARD_OFFSET + ATT1_SHARD_META_RECORD_SIZE],
                       1u, 0u,
                       ATT1_SHARD_DTYPE_F32,
                       ATT1_SHARD_QUANT_NONE,
                       ATT1_SHARD_REPL_NONE,
                       ATT1_SHARD_REDUCE_NONE,
                       0u);

    ok = (write_file(path, buf, file_size) == 0);
    free(buf);

    if (!ok) {
        return 0;
    }

    ok = (att1_model_load(path, &model) == ATT1_ERR_BAD_FORMAT);
    if (!ok) {
        att1_model_free(&model);
    }
    return ok;
}

/*
 * 6. tensor_id in the first record is not 0 (must be 0, 1, ...) → bad format.
 *    (invalid tensor reference test)
 */
static int test_invalid_tensor_id(const char *dir)
{
    char path[256];
    unsigned char *buf = NULL;
    att1_model model;
    int ok = 0;

    (void)snprintf(path, sizeof(path), "%s/bad_tensor_id.att1", dir);

    buf = build_model_with_meta();
    if (buf == NULL) {
        return 0;
    }

    /* Patch tensor_id of record 0 from 0 → 1 */
    write_u32le(&buf[TEST_SHARD_OFFSET + 0u], 1u);

    ok = (write_file(path, buf, TEST_FILE_SIZE_WITH_META) == 0);
    free(buf);

    if (!ok) {
        return 0;
    }

    ok = (att1_model_load(path, &model) == ATT1_ERR_BAD_FORMAT);
    if (!ok) {
        att1_model_free(&model);
    }
    return ok;
}

/*
 * 7. dtype field holds an unknown code → bad format.
 *    (invalid field test)
 */
static int test_invalid_dtype(const char *dir)
{
    char path[256];
    unsigned char *buf = NULL;
    att1_model model;
    int ok = 0;

    (void)snprintf(path, sizeof(path), "%s/bad_dtype.att1", dir);

    buf = build_model_with_meta();
    if (buf == NULL) {
        return 0;
    }

    /* Patch dtype @ record offset 48 */
    write_u32le(&buf[TEST_SHARD_OFFSET + 48u], 99u);

    ok = (write_file(path, buf, TEST_FILE_SIZE_WITH_META) == 0);
    free(buf);

    if (!ok) {
        return 0;
    }

    ok = (att1_model_load(path, &model) == ATT1_ERR_BAD_FORMAT);
    if (!ok) {
        att1_model_free(&model);
    }
    return ok;
}

/*
 * 8. _reserved field is nonzero → bad format.
 */
static int test_reserved_nonzero(const char *dir)
{
    char path[256];
    unsigned char *buf = NULL;
    att1_model model;
    int ok = 0;

    (void)snprintf(path, sizeof(path), "%s/bad_reserved.att1", dir);

    buf = build_model_with_meta();
    if (buf == NULL) {
        return 0;
    }

    /* Patch _reserved @ record offset 108 */
    write_u32le(&buf[TEST_SHARD_OFFSET + 108u], 1u);

    ok = (write_file(path, buf, TEST_FILE_SIZE_WITH_META) == 0);
    free(buf);

    if (!ok) {
        return 0;
    }

    ok = (att1_model_load(path, &model) == ATT1_ERR_BAD_FORMAT);
    if (!ok) {
        att1_model_free(&model);
    }
    return ok;
}

/* ── main ─────────────────────────────────────────────────────────────────── */

int main(void)
{
    const char *dir = "build/shard_meta_test";

    (void)mkdir(dir, 0777);

#define RUN(name)                                           \
    do {                                                    \
        if (!name(dir)) {                                   \
            fputs(#name " failed\n", stderr);               \
            return 1;                                       \
        }                                                   \
    } while (0)

    RUN(test_no_shard_meta);
    RUN(test_valid_shard_meta);
    RUN(test_bad_bounds);
    RUN(test_truncated_records);
    RUN(test_bad_record_count);
    RUN(test_invalid_tensor_id);
    RUN(test_invalid_dtype);
    RUN(test_reserved_nonzero);

#undef RUN

    puts("shard_meta test passed");
    return 0;
}
