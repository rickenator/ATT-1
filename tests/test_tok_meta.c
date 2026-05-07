#include "att1_model.h"
#include "att1_tok_meta.h"
#include "att1_status.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ── little-endian write helpers ──────────────────────────────────────────── */

static void write_u32le(unsigned char *p, uint32_t v)
{
    p[0] = (unsigned char)(v & 0xFFu);
    p[1] = (unsigned char)((v >> 8u) & 0xFFu);
    p[2] = (unsigned char)((v >> 16u) & 0xFFu);
    p[3] = (unsigned char)((v >> 24u) & 0xFFu);
}

static void write_i32le(unsigned char *p, int32_t v)
{
    uint32_t u = 0u;
    memcpy(&u, &v, sizeof(u));
    write_u32le(p, u);
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

/* ── checked-in fixture path ─────────────────────────────────────────────── */

#define FIXTURE_PATH  "models/tok_meta/model.att1"
#define V1_MODEL_PATH "models/dummy/model.att1"

/* ── in-memory model builder ─────────────────────────────────────────────── */

/*
 * Minimal v2 model layout (no shard metadata):
 *
 *   header      @   0  ( 96 bytes) — version=2, header_size=96
 *   config      @  96  ( 36 bytes) — vocab_size=16
 *   desc        @ 132  (128 bytes) — tok_embeddings.weight [16,8] f32
 *   data        @ 260  (512 bytes) — 16×8 f32
 *   tok_meta    @ 772  ( 96 bytes) — tokenizer metadata
 *   EOF         @ 868
 */

#define TM_HEADER_SIZE  96u
#define TM_CONFIG_OFF   96u
#define TM_DESC_OFF    132u
#define TM_DATA_OFF    260u
#define TM_DATA_SIZE   512u    /* 16*8*4 */
#define TM_TOKM_OFF    772u
#define TM_TOKM_SIZE    96u
#define TM_FILE_SIZE   868u

#define TM_VOCAB       16u

static void build_v2_header(unsigned char *buf,
                             uint64_t tok_off,
                             uint64_t tok_sz)
{
    memcpy(&buf[0], ATT1_MODEL_MAGIC, ATT1_MODEL_MAGIC_SIZE);
    write_u32le(&buf[8],  ATT1_MODEL_VERSION_2);
    write_u32le(&buf[12], TM_HEADER_SIZE);
    write_u64le(&buf[16], TM_CONFIG_OFF);
    write_u64le(&buf[24], ATT1_MODEL_CONFIG_SIZE);
    write_u64le(&buf[32], TM_DESC_OFF);
    write_u64le(&buf[40], 1u);            /* tensor_count */
    write_u64le(&buf[48], TM_DATA_OFF);
    write_u64le(&buf[56], TM_DATA_SIZE);
    write_u64le(&buf[64], 0u);            /* shard_offset=0 */
    write_u64le(&buf[72], 0u);            /* shard_size=0 */
    write_u64le(&buf[80], tok_off);
    write_u64le(&buf[88], tok_sz);
}

static void build_config(unsigned char *buf, uint32_t vocab_size)
{
    write_u32le(&buf[0],   vocab_size);   /* vocab_size */
    write_u32le(&buf[4],   2u);           /* n_layers */
    write_u32le(&buf[8],   2u);           /* n_heads */
    write_u32le(&buf[12],  8u);           /* d_model */
    write_u32le(&buf[16], 16u);           /* d_ff */
    write_u32le(&buf[20], 128u);          /* max_seq_len */
    write_u32le(&buf[24],  4u);           /* rope_dim */
    write_u32le(&buf[28],  1u);           /* n_tiles */
    write_u32le(&buf[32],  0u);           /* shard_count */
}

static void build_tensor_desc(unsigned char *buf)
{
    uint32_t i = 0u;

    memset(buf, 0, ATT1_MODEL_TENSOR_DESC_SIZE);
    (void)snprintf((char *)buf, ATT1_MODEL_NAME_SIZE, "tok_embeddings.weight");
    write_u32le(&buf[64], ATT1_MODEL_DTYPE_F32);
    write_u32le(&buf[68], 2u);             /* ndims=2 */
    write_u64le(&buf[72], 16u);            /* shape[0] */
    write_u64le(&buf[80],  8u);            /* shape[1] */
    write_u64le(&buf[88],  1u);            /* shape[2] padding */
    write_u64le(&buf[96],  1u);            /* shape[3] padding */
    write_u64le(&buf[104], 0u);            /* offset in data section */
    write_u64le(&buf[112], TM_DATA_SIZE);  /* nbytes */
    for (i = 120u; i < ATT1_MODEL_TENSOR_DESC_SIZE; i++) {
        buf[i] = 0u;
    }
}

static void build_tensor_data(unsigned char *buf)
{
    uint32_t i = 0u;
    const uint32_t n = TM_DATA_SIZE / 4u;

    for (i = 0u; i < n; i++) {
        write_f32le(&buf[i * 4u], (float)i * 0.0001f);
    }
}

/*
 * build_tok_meta_section() fills a 96-byte tokenizer metadata record.
 * All fields are valid by default; callers may overwrite bytes for error cases.
 */
static void build_tok_meta_section(unsigned char *buf,
                                   uint32_t schema_version,
                                   uint32_t tokenizer_type,
                                   uint32_t vocab_size,
                                   int32_t  bos_id,
                                   int32_t  eos_id,
                                   int32_t  pad_id,
                                   int32_t  unk_id,
                                   uint32_t flags)
{
    memset(buf, 0, ATT1_TOK_META_SIZE);
    write_u32le(&buf[0],  schema_version);
    write_u32le(&buf[4],  tokenizer_type);
    write_u32le(&buf[8],  vocab_size);
    write_i32le(&buf[12], bos_id);
    write_i32le(&buf[16], eos_id);
    write_i32le(&buf[20], pad_id);
    write_i32le(&buf[24], unk_id);
    write_u32le(&buf[28], 0u);   /* byte_fallback */
    write_u32le(&buf[32], ATT1_TOK_NORM_NONE);
    write_u32le(&buf[36], ATT1_TOK_PRETOK_BYTE_LEVEL);
    write_u32le(&buf[40], ATT1_TOK_HASH_NONE);
    write_u32le(&buf[44], flags);
    /* asset_hash[32] zeroed */
    write_u64le(&buf[80], 0u);   /* asset_offset */
    write_u64le(&buf[88], 0u);   /* asset_size */
}

/*
 * alloc_v2_model() allocates a TM_FILE_SIZE byte buffer with a valid
 * v2 model.  The tok_meta section starts at TM_TOKM_OFF.
 * Caller must free() the returned pointer.
 */
static unsigned char *alloc_v2_model(void)
{
    unsigned char *buf = calloc(TM_FILE_SIZE, 1u);

    if (buf == NULL) {
        return NULL;
    }

    build_v2_header(buf, TM_TOKM_OFF, TM_TOKM_SIZE);
    build_config(&buf[TM_CONFIG_OFF], TM_VOCAB);
    build_tensor_desc(&buf[TM_DESC_OFF]);
    build_tensor_data(&buf[TM_DATA_OFF]);
    build_tok_meta_section(&buf[TM_TOKM_OFF],
                           ATT1_TOK_META_SCHEMA_VERSION,
                           ATT1_TOK_TYPE_BPE_JSON,
                           TM_VOCAB,
                           1,    /* bos */
                           2,    /* eos */
                           -1,   /* pad absent */
                           0,    /* unk */
                           0u);  /* flags */
    return buf;
}

/* ── model file I/O ─────────────────────────────────────────────────────── */

#define TMP_FILE "build/test_tok_meta_tmp.att1"

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

/* ── test cases ──────────────────────────────────────────────────────────── */

/*
 * Test 1: v1 model has tok_meta.present == 0.
 */
static int test_absent_path(void)
{
    att1_model model;

    if (att1_model_load(V1_MODEL_PATH, &model) != ATT1_OK) {
        fputs("absent_path: v1 load failed\n", stderr);
        return 0;
    }

    if (model.tok_meta.present != 0) {
        fputs("absent_path: expected tok_meta.present==0\n", stderr);
        att1_model_free(&model);
        return 0;
    }

    att1_model_free(&model);
    return 1;
}

/*
 * Test 2: valid tokenizer metadata in checked-in fixture.
 */
static int test_valid_tok_meta(void)
{
    att1_model model;

    if (att1_model_load(FIXTURE_PATH, &model) != ATT1_OK) {
        fputs("valid_tok_meta: load failed\n", stderr);
        return 0;
    }

    if (model.tok_meta.present != 1) {
        fputs("valid_tok_meta: expected present==1\n", stderr);
        att1_model_free(&model);
        return 0;
    }

    if (model.tok_meta.tokenizer_type != ATT1_TOK_TYPE_BPE_JSON) {
        fputs("valid_tok_meta: wrong tokenizer_type\n", stderr);
        att1_model_free(&model);
        return 0;
    }

    if (model.tok_meta.vocab_size != 16u) {
        fputs("valid_tok_meta: wrong vocab_size\n", stderr);
        att1_model_free(&model);
        return 0;
    }

    if ((model.tok_meta.bos_token_id != 1) ||
        (model.tok_meta.eos_token_id != 2) ||
        (model.tok_meta.pad_token_id != ATT1_TOK_ID_ABSENT) ||
        (model.tok_meta.unk_token_id != 0)) {
        fputs("valid_tok_meta: wrong special token IDs\n", stderr);
        att1_model_free(&model);
        return 0;
    }

    if (model.tok_meta.normalization_policy != ATT1_TOK_NORM_NONE) {
        fputs("valid_tok_meta: wrong norm policy\n", stderr);
        att1_model_free(&model);
        return 0;
    }

    if (model.tok_meta.pretokenizer_policy != ATT1_TOK_PRETOK_BYTE_LEVEL) {
        fputs("valid_tok_meta: wrong pretok policy\n", stderr);
        att1_model_free(&model);
        return 0;
    }

    att1_model_free(&model);
    return 1;
}

/*
 * Test 3: bad schema_version in tokenizer metadata section.
 */
static int test_bad_version(void)
{
    unsigned char *buf = alloc_v2_model();
    att1_model model;
    att1_status_t rc = ATT1_OK;

    if (buf == NULL) {
        return 0;
    }

    /* Corrupt schema_version to 0. */
    write_u32le(&buf[TM_TOKM_OFF + 0u], 0u);

    if (write_file(TMP_FILE, buf, TM_FILE_SIZE) != 0) {
        free(buf);
        return 0;
    }
    free(buf);

    rc = att1_model_load(TMP_FILE, &model);
    if (rc == ATT1_OK) {
        fputs("bad_version: expected load to fail\n", stderr);
        att1_model_free(&model);
        return 0;
    }

    return 1;
}

/*
 * Test 4: bad tokenizer_type in tokenizer metadata section.
 */
static int test_bad_tok_type(void)
{
    unsigned char *buf = alloc_v2_model();
    att1_model model;
    att1_status_t rc = ATT1_OK;

    if (buf == NULL) {
        return 0;
    }

    /* Set tokenizer_type to 99 (unknown/invalid). */
    write_u32le(&buf[TM_TOKM_OFF + 4u], 99u);

    if (write_file(TMP_FILE, buf, TM_FILE_SIZE) != 0) {
        free(buf);
        return 0;
    }
    free(buf);

    rc = att1_model_load(TMP_FILE, &model);
    if (rc == ATT1_OK) {
        fputs("bad_tok_type: expected load to fail\n", stderr);
        att1_model_free(&model);
        return 0;
    }

    return 1;
}

/*
 * Test 5: tok_meta.vocab_size != model config vocab_size.
 */
static int test_vocab_mismatch(void)
{
    unsigned char *buf = alloc_v2_model();
    att1_model model;
    att1_status_t rc = ATT1_OK;

    if (buf == NULL) {
        return 0;
    }

    /* Set tok_meta vocab_size to 32 while model config stays at 16. */
    write_u32le(&buf[TM_TOKM_OFF + 8u], 32u);

    if (write_file(TMP_FILE, buf, TM_FILE_SIZE) != 0) {
        free(buf);
        return 0;
    }
    free(buf);

    rc = att1_model_load(TMP_FILE, &model);
    if (rc == ATT1_OK) {
        fputs("vocab_mismatch: expected load to fail\n", stderr);
        att1_model_free(&model);
        return 0;
    }

    return 1;
}

/*
 * Test 6: tokenizer metadata section size != ATT1_TOK_META_SIZE (truncated).
 */
static int test_truncated_section(void)
{
    /* Build a model where tok_meta_size points past EOF or is wrong size. */
    unsigned char *buf = calloc(TM_FILE_SIZE, 1u);
    att1_model model;
    att1_status_t rc = ATT1_OK;

    if (buf == NULL) {
        return 0;
    }

    /* Announce tok_meta as 50 bytes (not ATT1_TOK_META_SIZE=96). */
    build_v2_header(buf, TM_TOKM_OFF, 50u);
    build_config(&buf[TM_CONFIG_OFF], TM_VOCAB);
    build_tensor_desc(&buf[TM_DESC_OFF]);
    build_tensor_data(&buf[TM_DATA_OFF]);

    if (write_file(TMP_FILE, buf, TM_FILE_SIZE) != 0) {
        free(buf);
        return 0;
    }
    free(buf);

    rc = att1_model_load(TMP_FILE, &model);
    if (rc == ATT1_OK) {
        fputs("truncated_section: expected load to fail\n", stderr);
        att1_model_free(&model);
        return 0;
    }

    return 1;
}

/*
 * Test 7: nonzero reserved flags field rejected.
 */
static int test_bad_flags(void)
{
    unsigned char *buf = alloc_v2_model();
    att1_model model;
    att1_status_t rc = ATT1_OK;

    if (buf == NULL) {
        return 0;
    }

    /* Set flags to 1 (reserved, must be 0). */
    write_u32le(&buf[TM_TOKM_OFF + 44u], 1u);

    if (write_file(TMP_FILE, buf, TM_FILE_SIZE) != 0) {
        free(buf);
        return 0;
    }
    free(buf);

    rc = att1_model_load(TMP_FILE, &model);
    if (rc == ATT1_OK) {
        fputs("bad_flags: expected load to fail\n", stderr);
        att1_model_free(&model);
        return 0;
    }

    return 1;
}

/*
 * Test 8: v1 model still loads normally (no tok_meta regression).
 */
static int test_v1_compat(void)
{
    att1_model model;

    if (att1_model_load(V1_MODEL_PATH, &model) != ATT1_OK) {
        fputs("v1_compat: v1 load failed\n", stderr);
        return 0;
    }

    /* v1 model must have tok_meta absent and inference-critical fields intact. */
    if ((model.tok_meta.present != 0) ||
        (model.config.vocab_size == 0u) ||
        (model.tensor_count == 0u)) {
        fputs("v1_compat: unexpected model state\n", stderr);
        att1_model_free(&model);
        return 0;
    }

    att1_model_free(&model);
    return 1;
}

/* ── main ─────────────────────────────────────────────────────────────────── */

int main(void)
{
    int pass = 1;

    if (!test_absent_path()) {
        fputs("FAIL: absent_path\n", stderr);
        pass = 0;
    }

    if (!test_valid_tok_meta()) {
        fputs("FAIL: valid_tok_meta\n", stderr);
        pass = 0;
    }

    if (!test_bad_version()) {
        fputs("FAIL: bad_version\n", stderr);
        pass = 0;
    }

    if (!test_bad_tok_type()) {
        fputs("FAIL: bad_tok_type\n", stderr);
        pass = 0;
    }

    if (!test_vocab_mismatch()) {
        fputs("FAIL: vocab_mismatch\n", stderr);
        pass = 0;
    }

    if (!test_truncated_section()) {
        fputs("FAIL: truncated_section\n", stderr);
        pass = 0;
    }

    if (!test_bad_flags()) {
        fputs("FAIL: bad_flags\n", stderr);
        pass = 0;
    }

    if (!test_v1_compat()) {
        fputs("FAIL: v1_compat\n", stderr);
        pass = 0;
    }

    if (!pass) {
        fputs("tok_meta test failed\n", stderr);
        return 1;
    }

    puts("tok_meta test passed");
    return 0;
}
