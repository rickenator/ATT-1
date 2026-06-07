/*
 * test_quant_q4.c  –  M74: q4 format/schema validation tests
 *
 * Covers:
 *   - Loader accepts valid q4 tensor descriptors (all legal group sizes).
 *   - Loader rejects invalid q4 group sizes (not a power-of-two in [16,128]).
 *   - Loader rejects q4 tensors where cols is not divisible by group_size.
 *   - Loader rejects wrong nbytes for a q4 tensor.
 *   - Loader rejects q4 tensors with reserved flag bits set.
 *   - att1_model_view_validate_decoder returns ATT1_ERR_UNSUPPORTED for any
 *     model containing a q4 tensor (no silent fallback to q8/f32 inference).
 *   - att1-inspect prints "dtype_name=q4" and "quant=grouped-q4-g<N>" for q4
 *     tensors and exits 0 (inspect is read-only; no inference attempted).
 *   - Existing f32/q8 models are unaffected.
 */

#include "att1_model.h"
#include "att1_model_view.h"
#include "att1_quant.h"
#include "att1_status.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

/* ── little-endian helpers ───────────────────────────────────────────────── */

static void write_u32le(unsigned char *p, uint32_t v)
{
    p[0] = (unsigned char)(v & 0xffu);
    p[1] = (unsigned char)((v >> 8u)  & 0xffu);
    p[2] = (unsigned char)((v >> 16u) & 0xffu);
    p[3] = (unsigned char)((v >> 24u) & 0xffu);
}

static void write_u64le(unsigned char *p, uint64_t v)
{
    p[0] = (unsigned char)(v         & 0xffu);
    p[1] = (unsigned char)((v >> 8u)  & 0xffu);
    p[2] = (unsigned char)((v >> 16u) & 0xffu);
    p[3] = (unsigned char)((v >> 24u) & 0xffu);
    p[4] = (unsigned char)((v >> 32u) & 0xffu);
    p[5] = (unsigned char)((v >> 40u) & 0xffu);
    p[6] = (unsigned char)((v >> 48u) & 0xffu);
    p[7] = (unsigned char)((v >> 56u) & 0xffu);
}

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

static int read_file_alloc(const char *path,
                           unsigned char **out_data,
                           size_t *out_size)
{
    FILE *fp = fopen(path, "rb");
    long sz = 0;
    unsigned char *buf = NULL;

    if (fp == NULL) {
        return -1;
    }
    if ((fseek(fp, 0L, SEEK_END) != 0) ||
        ((sz = ftell(fp)) < 0) ||
        (fseek(fp, 0L, SEEK_SET) != 0)) {
        fclose(fp);
        return -1;
    }
    buf = malloc((size_t)sz);
    if ((buf == NULL) && (sz != 0)) {
        fclose(fp);
        return -1;
    }
    if ((sz > 0) && (fread(buf, 1u, (size_t)sz, fp) != (size_t)sz)) {
        free(buf);
        fclose(fp);
        return -1;
    }
    fclose(fp);
    *out_data = buf;
    *out_size = (size_t)sz;
    return 0;
}

static int buf_contains(const unsigned char *data, size_t size, const char *needle)
{
    const size_t nlen = strlen(needle);
    size_t i = 0u;

    if ((nlen == 0u) || (nlen > size)) {
        return nlen == 0u;
    }
    for (i = 0u; i <= size - nlen; i++) {
        if (memcmp(&data[i], needle, nlen) == 0) {
            return 1;
        }
    }
    return 0;
}

/* ── minimal .att1 builder for q4 tests ─────────────────────────────────── */

/*
 * Build a minimal ATT-1 v1 model with a single q4 tensor.
 *
 *   rows       – tensor rows (output dim)
 *   cols       – tensor cols (input dim; must be even and divisible by group_size
 *                when group_size != 0)
 *   group_size – value written to flags[7:0]; 0 = default (32)
 *   flags_extra – extra bits ORed into the flags field (for reserved-bits tests)
 *   nbytes_override – if non-zero, overrides the computed nbytes
 *
 * Returns allocated buffer in *out_data / *out_size; caller must free().
 */
static int build_q4_model(uint64_t rows,
                          uint64_t cols,
                          uint32_t group_size,
                          uint32_t flags_extra,
                          uint64_t nbytes_override,
                          unsigned char **out_data,
                          size_t *out_size)
{
    /* Layout constants */
    const uint64_t config_offset = ATT1_MODEL_HEADER_SIZE;               /* 80 */
    const uint64_t desc_offset   = config_offset + ATT1_MODEL_CONFIG_SIZE; /* 116 */
    const uint64_t data_offset   = desc_offset + ATT1_MODEL_TENSOR_DESC_SIZE; /* 244 */

    /* Compute correct q4 payload size */
    const uint32_t gs = (group_size == 0u) ? ATT1_Q4_GROUP_SIZE_DEFAULT : group_size;
    const uint64_t packed_bytes = rows * cols / 2u;
    const uint64_t n_groups     = rows * (cols / (uint64_t)gs);
    const uint64_t scale_bytes  = n_groups * 4u; /* float32 */
    const uint64_t correct_nbytes = packed_bytes + scale_bytes;
    const uint64_t actual_nbytes  = (nbytes_override != 0u)
                                    ? nbytes_override : correct_nbytes;

    const uint64_t data_size = actual_nbytes;
    const size_t   file_size = (size_t)(data_offset + data_size);

    unsigned char *file = calloc(file_size, 1u);
    uint32_t flags = (group_size & ATT1_Q4_FLAGS_GROUP_MASK) | flags_extra;

    if (file == NULL) {
        return -1;
    }

    /* Header */
    memcpy(&file[0], ATT1_MODEL_MAGIC, ATT1_MODEL_MAGIC_SIZE);
    write_u32le(&file[8],  ATT1_MODEL_VERSION);
    write_u32le(&file[12], ATT1_MODEL_HEADER_SIZE);
    write_u64le(&file[16], config_offset);
    write_u64le(&file[24], ATT1_MODEL_CONFIG_SIZE);
    write_u64le(&file[32], desc_offset);
    write_u64le(&file[40], 1u);          /* tensor_count */
    write_u64le(&file[48], data_offset);
    write_u64le(&file[56], data_size);
    write_u64le(&file[64], 0u);          /* shard_offset */
    write_u64le(&file[72], 0u);          /* shard_size */

    /* Config (dummy values) */
    write_u32le(&file[config_offset +  0u], 256u); /* vocab_size */
    write_u32le(&file[config_offset +  4u],   2u); /* n_layers */
    write_u32le(&file[config_offset +  8u],   2u); /* n_heads */
    write_u32le(&file[config_offset + 12u],   4u); /* d_model */
    write_u32le(&file[config_offset + 16u],   8u); /* d_ff */
    write_u32le(&file[config_offset + 20u],   8u); /* max_seq_len */
    write_u32le(&file[config_offset + 24u],   2u); /* rope_dim */
    write_u32le(&file[config_offset + 28u],   1u); /* n_tiles */
    write_u32le(&file[config_offset + 32u],   0u); /* shard_count */

    /* Tensor descriptor (128 bytes) */
    {
        unsigned char *desc = &file[desc_offset];

        memset(desc, 0, ATT1_MODEL_TENSOR_DESC_SIZE);
        (void)snprintf((char *)desc, ATT1_MODEL_NAME_SIZE, "output.weight");
        write_u32le(&desc[64], ATT1_MODEL_DTYPE_Q4); /* dtype */
        write_u32le(&desc[68], 2u);                  /* ndims */
        write_u64le(&desc[72], rows);
        write_u64le(&desc[80], cols);
        write_u64le(&desc[88], 1u);
        write_u64le(&desc[96], 1u);
        write_u64le(&desc[104], 0u);              /* offset within data section */
        write_u64le(&desc[112], actual_nbytes);
        write_u32le(&desc[120], 0u);              /* shard_id */
        write_u32le(&desc[124], flags);
    }

    /* Data: all zeros (valid packed nibbles; not tested for values here) */

    *out_data = file;
    *out_size = file_size;
    return 0;
}

static int write_q4_model(const char *path,
                          uint64_t rows, uint64_t cols,
                          uint32_t group_size, uint32_t flags_extra,
                          uint64_t nbytes_override)
{
    unsigned char *data = NULL;
    size_t size = 0u;
    int rc = -1;

    if (build_q4_model(rows, cols, group_size, flags_extra,
                       nbytes_override, &data, &size) != 0) {
        return -1;
    }
    rc = write_file(path, data, size);
    free(data);
    return rc;
}

/* ── test functions ──────────────────────────────────────────────────────── */

/*
 * Load a valid q4 model and verify load succeeds and dtype is reported as Q4.
 */
static int test_q4_load_valid(const char *dir)
{
    const char *path = "build/test_q4/q4_valid_g32.att1";
    att1_model model;

    (void)dir;

    /* rows=4, cols=64, group_size=32 → nbytes = 4*64/2 + 4*(64/32)*4 = 128+32 = 160 */
    if (write_q4_model(path, 4u, 64u, 32u, 0u, 0u) != 0) {
        fputs("q4_load_valid: fixture write failed\n", stderr);
        return -1;
    }

    if (att1_model_load(path, &model) != ATT1_OK) {
        fputs("q4_load_valid: load failed\n", stderr);
        return -1;
    }

    if (model.tensors[0].dtype != ATT1_MODEL_DTYPE_Q4) {
        fputs("q4_load_valid: dtype not Q4\n", stderr);
        att1_model_free(&model);
        return -1;
    }

    att1_model_free(&model);
    return 0;
}

/*
 * All valid group sizes (0/default, 16, 32, 64, 128) load successfully.
 * Uses cols=128 which is divisible by all valid group sizes.
 */
static int test_q4_valid_group_sizes(void)
{
    /* group_size -> expected nbytes for rows=4, cols=128 */
    const struct { uint32_t gs; uint64_t expected_nbytes; } cases[] = {
        {  0u, 256u + 64u  }, /* default=32: packed=256, scales=4*(128/32)*4=64 */
        { 16u, 256u + 128u }, /* gs=16: scales=4*(128/16)*4=128 */
        { 32u, 256u + 64u  }, /* gs=32: scales=4*(128/32)*4=64  */
        { 64u, 256u + 32u  }, /* gs=64: scales=4*(128/64)*4=32  */
        {128u, 256u + 16u  }  /* gs=128:scales=4*(128/128)*4=16 */
    };
    const size_t ncases = sizeof(cases) / sizeof(cases[0]);
    size_t k = 0u;
    char path[128];
    att1_model model;

    for (k = 0u; k < ncases; k++) {
        (void)snprintf(path, sizeof(path),
                       "build/test_q4/q4_gs%u.att1", cases[k].gs);

        if (write_q4_model(path, 4u, 128u, cases[k].gs, 0u, 0u) != 0) {
            fprintf(stderr, "q4_valid_group_sizes: write failed gs=%u\n",
                    cases[k].gs);
            return -1;
        }

        if (att1_model_load(path, &model) != ATT1_OK) {
            fprintf(stderr, "q4_valid_group_sizes: load failed gs=%u\n",
                    cases[k].gs);
            return -1;
        }

        if (model.tensors[0].nbytes != cases[k].expected_nbytes) {
            fprintf(stderr,
                    "q4_valid_group_sizes: nbytes mismatch gs=%u "
                    "expected=%" PRIu64 " got=%" PRIu64 "\n",
                    cases[k].gs,
                    cases[k].expected_nbytes,
                    model.tensors[0].nbytes);
            att1_model_free(&model);
            return -1;
        }

        att1_model_free(&model);
    }
    return 0;
}

/*
 * Invalid group sizes (not a power-of-two in [16,128]) must be rejected.
 */
static int test_q4_bad_group_size(void)
{
    const uint32_t bad_sizes[] = { 1u, 3u, 7u, 15u, 48u, 100u, 129u, 200u, 255u };
    const size_t   ncases      = sizeof(bad_sizes) / sizeof(bad_sizes[0]);
    size_t         k           = 0u;
    char           path[128];
    att1_model     model;

    for (k = 0u; k < ncases; k++) {
        (void)snprintf(path, sizeof(path),
                       "build/test_q4/q4_bad_gs%u.att1", bad_sizes[k]);

        /* Provide a plausible but incorrect nbytes so the file is structurally
         * complete; the group_size validation must fire first. */
        if (write_q4_model(path, 4u, 64u, bad_sizes[k], 0u, 160u) != 0) {
            fprintf(stderr, "q4_bad_group_size: write failed gs=%u\n",
                    bad_sizes[k]);
            return -1;
        }

        if (att1_model_load(path, &model) == ATT1_OK) {
            fprintf(stderr,
                    "q4_bad_group_size: load succeeded unexpectedly gs=%u\n",
                    bad_sizes[k]);
            att1_model_free(&model);
            return -1;
        }
    }
    return 0;
}

/*
 * cols not divisible by group_size must be rejected.
 */
static int test_q4_bad_cols_alignment(void)
{
    const char *path = "build/test_q4/q4_bad_cols_align.att1";
    att1_model  model;

    /* cols=48, group_size=32 → 48 % 32 = 16 ≠ 0 → should fail */
    /* Provide nbytes for a theoretically valid 4×48 g32 model so the file
     * looks structurally correct but alignment check still fires. */
    if (write_q4_model(path, 4u, 48u, 32u, 0u, 160u) != 0) {
        fputs("q4_bad_cols_alignment: fixture write failed\n", stderr);
        return -1;
    }

    if (att1_model_load(path, &model) == ATT1_OK) {
        fputs("q4_bad_cols_alignment: load succeeded unexpectedly\n", stderr);
        att1_model_free(&model);
        return -1;
    }
    return 0;
}

/*
 * Correct group_size and alignment but wrong nbytes must be rejected.
 */
static int test_q4_bad_nbytes(void)
{
    const char *path = "build/test_q4/q4_bad_nbytes.att1";
    att1_model  model;

    /* rows=4, cols=64, g32 → correct nbytes=160; write 200 instead */
    if (write_q4_model(path, 4u, 64u, 32u, 0u, 200u) != 0) {
        fputs("q4_bad_nbytes: fixture write failed\n", stderr);
        return -1;
    }

    if (att1_model_load(path, &model) == ATT1_OK) {
        fputs("q4_bad_nbytes: load succeeded unexpectedly\n", stderr);
        att1_model_free(&model);
        return -1;
    }
    return 0;
}

/*
 * Reserved flag bits ([31:8]) must be zero for q4 tensors.
 */
static int test_q4_flags_reserved_bits(void)
{
    const char *path = "build/test_q4/q4_bad_flags.att1";
    att1_model  model;

    /* group_size=32 + reserved bit 8 set */
    if (write_q4_model(path, 4u, 64u, 32u, 0x100u, 0u) != 0) {
        fputs("q4_flags_reserved_bits: fixture write failed\n", stderr);
        return -1;
    }

    if (att1_model_load(path, &model) == ATT1_OK) {
        fputs("q4_flags_reserved_bits: load succeeded unexpectedly\n", stderr);
        att1_model_free(&model);
        return -1;
    }
    return 0;
}

/*
 * validate_decoder must return ATT1_ERR_UNSUPPORTED for any model that
 * contains a q4 tensor. No silent fallback to q8 or f32 inference.
 */
static int test_q4_inference_rejected(void)
{
    const char *path = "build/test_q4/q4_infer_reject.att1";
    att1_model  model;
    att1_status_t rc = ATT1_OK;

    if (write_q4_model(path, 4u, 64u, 32u, 0u, 0u) != 0) {
        fputs("q4_inference_rejected: fixture write failed\n", stderr);
        return -1;
    }

    if (att1_model_load(path, &model) != ATT1_OK) {
        fputs("q4_inference_rejected: load failed (should succeed for inspection)\n",
              stderr);
        return -1;
    }

    rc = att1_model_view_validate_decoder(&model);
    if (rc != ATT1_ERR_UNSUPPORTED) {
        fprintf(stderr,
                "q4_inference_rejected: validate_decoder returned %d "
                "(expected ATT1_ERR_UNSUPPORTED=%d)\n",
                (int)rc, (int)ATT1_ERR_UNSUPPORTED);
        att1_model_free(&model);
        return -1;
    }

    att1_model_free(&model);
    return 0;
}

/*
 * att1-inspect must print "dtype_name=q4" and "quant=grouped-q4-g32" for a
 * q4 tensor and exit 0 (inspection is read-only; no inference attempted).
 */
static int test_q4_inspect_output(void)
{
    const char *model_path  = "build/test_q4/q4_inspect.att1";
    const char *output_path = "build/test_q4/q4_inspect_out.txt";
    char cmd[512];
    unsigned char *data = NULL;
    size_t size = 0u;
    int rc = 0;

    if (write_q4_model(model_path, 4u, 64u, 32u, 0u, 0u) != 0) {
        fputs("q4_inspect_output: fixture write failed\n", stderr);
        return -1;
    }

    (void)snprintf(cmd, sizeof(cmd),
                   "build/att1-inspect %s > %s 2>&1",
                   model_path, output_path);
    if (system(cmd) != 0) {
        fputs("q4_inspect_output: att1-inspect exited nonzero\n", stderr);
        return -1;
    }

    if (read_file_alloc(output_path, &data, &size) != 0) {
        fputs("q4_inspect_output: cannot read inspect output\n", stderr);
        return -1;
    }

    if (!buf_contains(data, size, "dtype_name=q4")) {
        fputs("q4_inspect_output: missing dtype_name=q4\n", stderr);
        rc = -1;
    }
    if (!buf_contains(data, size, "quant=grouped-q4-g32")) {
        fputs("q4_inspect_output: missing quant=grouped-q4-g32\n", stderr);
        rc = -1;
    }
    if (!buf_contains(data, size, "q4_groups=8")) {
        fputs("q4_inspect_output: missing q4_groups=8\n", stderr);
        rc = -1;
    }
    if (!buf_contains(data, size, "q4_packed_bytes=128")) {
        fputs("q4_inspect_output: missing q4_packed_bytes=128\n", stderr);
        rc = -1;
    }
    if (!buf_contains(data, size, "q4_scale_bytes=32")) {
        fputs("q4_inspect_output: missing q4_scale_bytes=32\n", stderr);
        rc = -1;
    }

    free(data);
    return rc;
}

/*
 * Existing f32 and q8 models must still load unchanged (regression guard).
 */
static int test_existing_dtypes_unaffected(void)
{
    att1_model model;

    /* f32 model */
    if (att1_model_load("models/dummy/model.att1", &model) != ATT1_OK) {
        fputs("existing_dtypes_unaffected: f32 model load failed\n", stderr);
        return -1;
    }
    att1_model_free(&model);

    /* q8 model */
    if (att1_model_load("models/real_tiny_q8/model.att1", &model) != ATT1_OK) {
        fputs("existing_dtypes_unaffected: q8 model load failed\n", stderr);
        return -1;
    }
    att1_model_free(&model);

    return 0;
}

/* ── main ───────────────────────────────────────────────────────────────── */

int main(void)
{
    const char *dir = "build/test_q4";

    (void)mkdir(dir, 0777);

    if (test_q4_load_valid(dir) != 0) {
        fputs("FAIL: q4_load_valid\n", stderr);
        return 1;
    }

    if (test_q4_valid_group_sizes() != 0) {
        fputs("FAIL: q4_valid_group_sizes\n", stderr);
        return 1;
    }

    if (test_q4_bad_group_size() != 0) {
        fputs("FAIL: q4_bad_group_size\n", stderr);
        return 1;
    }

    if (test_q4_bad_cols_alignment() != 0) {
        fputs("FAIL: q4_bad_cols_alignment\n", stderr);
        return 1;
    }

    if (test_q4_bad_nbytes() != 0) {
        fputs("FAIL: q4_bad_nbytes\n", stderr);
        return 1;
    }

    if (test_q4_flags_reserved_bits() != 0) {
        fputs("FAIL: q4_flags_reserved_bits\n", stderr);
        return 1;
    }

    if (test_q4_inference_rejected() != 0) {
        fputs("FAIL: q4_inference_rejected\n", stderr);
        return 1;
    }

    if (test_q4_inspect_output() != 0) {
        fputs("FAIL: q4_inspect_output\n", stderr);
        return 1;
    }

    if (test_existing_dtypes_unaffected() != 0) {
        fputs("FAIL: existing_dtypes_unaffected\n", stderr);
        return 1;
    }

    puts("quant_q4 test passed");
    return 0;
}
