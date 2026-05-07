#include "att1_tok_meta.h"
#include "att1_status.h"

#include <stdio.h>
#include <string.h>

/* ── helpers ─────────────────────────────────────────────────────────────── */

static att1_tok_meta make_valid_meta(uint32_t vocab_size)
{
    att1_tok_meta m;

    memset(&m, 0, sizeof(m));
    m.present              = 1;
    m.schema_version       = ATT1_TOK_META_SCHEMA_VERSION;
    m.tokenizer_type       = ATT1_TOK_TYPE_BPE_JSON;
    m.vocab_size           = vocab_size;
    m.bos_token_id         = 1;
    m.eos_token_id         = 2;
    m.pad_token_id         = ATT1_TOK_ID_ABSENT;
    m.unk_token_id         = 0;
    m.byte_fallback        = 0u;
    m.normalization_policy = ATT1_TOK_NORM_NONE;
    m.pretokenizer_policy  = ATT1_TOK_PRETOK_BYTE_LEVEL;
    m.asset_hash_kind      = ATT1_TOK_HASH_NONE;
    m.flags                = 0u;
    m.asset_offset         = 0u;
    m.asset_size           = 0u;
    return m;
}

/* ── test cases ──────────────────────────────────────────────────────────── */

/*
 * Test 1: valid metadata returns ATT1_OK.
 */
static int test_valid(void)
{
    att1_tok_meta m = make_valid_meta(16u);
    const att1_status_t rc = att1_tok_meta_check_runtime(&m, 16u);

    if (rc != ATT1_OK) {
        fprintf(stderr, "valid: expected ATT1_OK, got %d\n", (int)rc);
        return 0;
    }
    return 1;
}

/*
 * Test 2: NULL meta pointer returns error.
 */
static int test_null_meta(void)
{
    const att1_status_t rc = att1_tok_meta_check_runtime(NULL, 16u);

    if (rc == ATT1_OK) {
        fputs("null_meta: expected error, got ATT1_OK\n", stderr);
        return 0;
    }
    return 1;
}

/*
 * Test 3: meta->present == 0 returns error.
 */
static int test_absent_meta(void)
{
    att1_tok_meta m = make_valid_meta(16u);
    att1_status_t rc = ATT1_OK;

    m.present = 0;
    rc = att1_tok_meta_check_runtime(&m, 16u);
    if (rc == ATT1_OK) {
        fputs("absent_meta: expected error, got ATT1_OK\n", stderr);
        return 0;
    }
    return 1;
}

/*
 * Test 4: schema_version != 1 returns ATT1_ERR_BAD_FORMAT.
 */
static int test_bad_schema_version(void)
{
    att1_tok_meta m = make_valid_meta(16u);
    att1_status_t rc = ATT1_OK;

    m.schema_version = 0u;
    rc = att1_tok_meta_check_runtime(&m, 16u);
    if (rc != ATT1_ERR_BAD_FORMAT) {
        fprintf(stderr, "bad_schema_version: expected BAD_FORMAT, got %d\n",
                (int)rc);
        return 0;
    }
    return 1;
}

/*
 * Test 5: tokenizer_type == ATT1_TOK_TYPE_UNKNOWN returns ATT1_ERR_UNSUPPORTED.
 */
static int test_unknown_type(void)
{
    att1_tok_meta m = make_valid_meta(16u);
    att1_status_t rc = ATT1_OK;

    m.tokenizer_type = ATT1_TOK_TYPE_UNKNOWN;
    rc = att1_tok_meta_check_runtime(&m, 16u);
    if (rc != ATT1_ERR_UNSUPPORTED) {
        fprintf(stderr, "unknown_type: expected UNSUPPORTED, got %d\n",
                (int)rc);
        return 0;
    }
    return 1;
}

/*
 * Test 6: out-of-range tokenizer_type returns ATT1_ERR_BAD_FORMAT.
 */
static int test_bad_type(void)
{
    att1_tok_meta m = make_valid_meta(16u);
    att1_status_t rc = ATT1_OK;

    m.tokenizer_type = 99u;
    rc = att1_tok_meta_check_runtime(&m, 16u);
    if (rc != ATT1_ERR_BAD_FORMAT) {
        fprintf(stderr, "bad_type: expected BAD_FORMAT, got %d\n", (int)rc);
        return 0;
    }
    return 1;
}

/*
 * Test 7: vocab_size == 0 returns ATT1_ERR_BAD_FORMAT.
 */
static int test_zero_vocab(void)
{
    att1_tok_meta m = make_valid_meta(16u);
    att1_status_t rc = ATT1_OK;

    m.vocab_size = 0u;
    rc = att1_tok_meta_check_runtime(&m, 0u);
    if (rc != ATT1_ERR_BAD_FORMAT) {
        fprintf(stderr, "zero_vocab: expected BAD_FORMAT, got %d\n", (int)rc);
        return 0;
    }
    return 1;
}

/*
 * Test 8: vocab_size mismatch vs model_vocab returns ATT1_ERR_BAD_FORMAT.
 */
static int test_vocab_mismatch(void)
{
    att1_tok_meta m = make_valid_meta(16u);
    att1_status_t rc = ATT1_OK;

    /* meta says 16 but model says 32 */
    rc = att1_tok_meta_check_runtime(&m, 32u);
    if (rc != ATT1_ERR_BAD_FORMAT) {
        fprintf(stderr, "vocab_mismatch: expected BAD_FORMAT, got %d\n",
                (int)rc);
        return 0;
    }
    return 1;
}

/*
 * Test 9: bos_token_id out of range [0, vocab_size) returns ATT1_ERR_BAD_FORMAT.
 */
static int test_bos_out_of_range(void)
{
    att1_tok_meta m = make_valid_meta(16u);
    att1_status_t rc = ATT1_OK;

    m.bos_token_id = 16; /* must be < vocab_size=16 */
    rc = att1_tok_meta_check_runtime(&m, 16u);
    if (rc != ATT1_ERR_BAD_FORMAT) {
        fprintf(stderr, "bos_out_of_range: expected BAD_FORMAT, got %d\n",
                (int)rc);
        return 0;
    }
    return 1;
}

/*
 * Test 10: unk_token_id out of range returns ATT1_ERR_BAD_FORMAT.
 */
static int test_unk_out_of_range(void)
{
    att1_tok_meta m = make_valid_meta(16u);
    att1_status_t rc = ATT1_OK;

    m.unk_token_id = 100; /* > vocab_size */
    rc = att1_tok_meta_check_runtime(&m, 16u);
    if (rc != ATT1_ERR_BAD_FORMAT) {
        fprintf(stderr, "unk_out_of_range: expected BAD_FORMAT, got %d\n",
                (int)rc);
        return 0;
    }
    return 1;
}

/*
 * Test 11: special token ID == -1 (absent) is valid.
 */
static int test_absent_ids_ok(void)
{
    att1_tok_meta m = make_valid_meta(16u);
    att1_status_t rc = ATT1_OK;

    m.bos_token_id = ATT1_TOK_ID_ABSENT;
    m.eos_token_id = ATT1_TOK_ID_ABSENT;
    m.pad_token_id = ATT1_TOK_ID_ABSENT;
    m.unk_token_id = ATT1_TOK_ID_ABSENT;
    rc = att1_tok_meta_check_runtime(&m, 16u);
    if (rc != ATT1_OK) {
        fprintf(stderr, "absent_ids_ok: expected ATT1_OK, got %d\n", (int)rc);
        return 0;
    }
    return 1;
}

/*
 * Test 12: byte_fallback > 1 returns ATT1_ERR_BAD_FORMAT.
 */
static int test_bad_byte_fallback(void)
{
    att1_tok_meta m = make_valid_meta(16u);
    att1_status_t rc = ATT1_OK;

    m.byte_fallback = 2u;
    rc = att1_tok_meta_check_runtime(&m, 16u);
    if (rc != ATT1_ERR_BAD_FORMAT) {
        fprintf(stderr, "bad_byte_fallback: expected BAD_FORMAT, got %d\n",
                (int)rc);
        return 0;
    }
    return 1;
}

/*
 * Test 13: nonzero flags returns ATT1_ERR_BAD_FORMAT.
 */
static int test_nonzero_flags(void)
{
    att1_tok_meta m = make_valid_meta(16u);
    att1_status_t rc = ATT1_OK;

    m.flags = 1u;
    rc = att1_tok_meta_check_runtime(&m, 16u);
    if (rc != ATT1_ERR_BAD_FORMAT) {
        fprintf(stderr, "nonzero_flags: expected BAD_FORMAT, got %d\n",
                (int)rc);
        return 0;
    }
    return 1;
}

/*
 * Test 14: sentencepiece type is supported (not UNKNOWN) but not implemented.
 *          ATT1_OK is returned by check_runtime; the caller decides not-impl.
 */
static int test_sentpiece_type_ok(void)
{
    att1_tok_meta m = make_valid_meta(16u);
    att1_status_t rc = ATT1_OK;

    m.tokenizer_type = ATT1_TOK_TYPE_SENTPIECE;
    rc = att1_tok_meta_check_runtime(&m, 16u);
    if (rc != ATT1_OK) {
        fprintf(stderr, "sentpiece_type_ok: expected ATT1_OK, got %d\n",
                (int)rc);
        return 0;
    }
    return 1;
}

/*
 * Test 15: asset_offset nonzero but asset_size zero → ATT1_ERR_BAD_FORMAT.
 */
static int test_asset_half_set(void)
{
    att1_tok_meta m = make_valid_meta(16u);
    att1_status_t rc = ATT1_OK;

    m.asset_offset = 1000u;
    m.asset_size   = 0u;
    rc = att1_tok_meta_check_runtime(&m, 16u);
    if (rc != ATT1_ERR_BAD_FORMAT) {
        fprintf(stderr, "asset_half_set: expected BAD_FORMAT, got %d\n",
                (int)rc);
        return 0;
    }
    return 1;
}

/* ── main ─────────────────────────────────────────────────────────────────── */

int main(void)
{
    int pass = 1;

#define RUN(name) \
    if (!test_##name()) { fputs("FAIL: " #name "\n", stderr); pass = 0; }

    RUN(valid)
    RUN(null_meta)
    RUN(absent_meta)
    RUN(bad_schema_version)
    RUN(unknown_type)
    RUN(bad_type)
    RUN(zero_vocab)
    RUN(vocab_mismatch)
    RUN(bos_out_of_range)
    RUN(unk_out_of_range)
    RUN(absent_ids_ok)
    RUN(bad_byte_fallback)
    RUN(nonzero_flags)
    RUN(sentpiece_type_ok)
    RUN(asset_half_set)

#undef RUN

    if (!pass) {
        fputs("tok_meta_select test failed\n", stderr);
        return 1;
    }

    puts("tok_meta_select test passed");
    return 0;
}
