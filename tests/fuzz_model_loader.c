/*
 * ATT-1 M143: deterministic fuzz/smoke harness for the binary model loader.
 *
 * Drives att1_model_load() against malformed .att1 blobs constructed
 * in-memory and written to temporary files in /tmp.  Each test case either
 * expects the loader to reject the input (non-zero status) or to accept it
 * (zero status).
 *
 * This is a local hardening check only; it is not part of make test.
 * Run via:  make fuzz-loader
 *
 * Exit codes:
 *   0 — all cases behaved as expected
 *   1 — at least one case failed (loader accepted malformed input or
 *       rejected valid input)
 */

#define _POSIX_C_SOURCE 200809L

#include "att1_model.h"
#include "att1_status.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>

/* ---- byte helpers ---- */

static void w32le(unsigned char *p, uint32_t v)
{
    p[0] = (unsigned char)(v         & 0xFFu);
    p[1] = (unsigned char)((v >>  8) & 0xFFu);
    p[2] = (unsigned char)((v >> 16) & 0xFFu);
    p[3] = (unsigned char)((v >> 24) & 0xFFu);
}

static void w64le(unsigned char *p, uint64_t v)
{
    p[0] = (unsigned char)(v         & 0xFFu);
    p[1] = (unsigned char)((v >>  8) & 0xFFu);
    p[2] = (unsigned char)((v >> 16) & 0xFFu);
    p[3] = (unsigned char)((v >> 24) & 0xFFu);
    p[4] = (unsigned char)((v >> 32) & 0xFFu);
    p[5] = (unsigned char)((v >> 40) & 0xFFu);
    p[6] = (unsigned char)((v >> 48) & 0xFFu);
    p[7] = (unsigned char)((v >> 56) & 0xFFu);
}

/* ---- build minimal valid V1 .att1 in-memory ---- */
/*
 * Layout: Header (80 bytes) + Config (36 bytes) = 116 bytes total.
 * Zero tensors; config fields are all zero (valid for loader checks).
 */
#define VALID_BASE_SIZE \
    ((size_t)(ATT1_MODEL_HEADER_SIZE + ATT1_MODEL_CONFIG_SIZE))

static void build_valid_base(unsigned char *b)
{
    size_t desc_and_data_off = ATT1_MODEL_HEADER_SIZE + ATT1_MODEL_CONFIG_SIZE;

    memset(b, 0, VALID_BASE_SIZE);
    memcpy(b, ATT1_MODEL_MAGIC, ATT1_MODEL_MAGIC_SIZE);
    w32le(b +  8, ATT1_MODEL_VERSION);        /* version = 1 */
    w32le(b + 12, ATT1_MODEL_HEADER_SIZE);    /* header_size = 80 */
    w64le(b + 16, ATT1_MODEL_HEADER_SIZE);    /* config_offset = 80 */
    w64le(b + 24, ATT1_MODEL_CONFIG_SIZE);    /* config_size = 36 */
    w64le(b + 32, (uint64_t)desc_and_data_off); /* desc_offset (past config) */
    w64le(b + 40, 0u);                        /* tensor_count = 0 */
    w64le(b + 48, (uint64_t)desc_and_data_off); /* data_offset */
    w64le(b + 56, 0u);                        /* data_size = 0 */
    w64le(b + 64, 0u);                        /* shard_offset = 0 */
    w64le(b + 72, 0u);                        /* shard_size = 0 */
    /* config section at offset 80: all zeros */
}

/* ---- write bytes to a new /tmp temp file ---- */

static int write_tmp(const unsigned char *data, size_t size, char *out_path)
{
    int fd = -1;
    ssize_t nw = 0;

    memcpy(out_path, "/tmp/att1_fuzz_XXXXXX", 22u); /* includes NUL */
    fd = mkstemp(out_path);
    if (fd < 0) {
        return -1;
    }
    if (size > 0u) {
        nw = write(fd, data, size);
        close(fd);
        if (nw != (ssize_t)size) {
            (void)unlink(out_path);
            return -1;
        }
    } else {
        close(fd);
    }
    return 0;
}

/* ---- test runners ---- */

/*
 * Run one in-memory case.
 * expected_ok: 1 = loader must succeed (status 0), 0 = must reject.
 * Returns 1 on test PASS, 0 on test FAIL.
 */
static int run_case(const char *label,
                    const unsigned char *data, size_t size,
                    int expected_ok)
{
    char path[32];
    att1_model model;
    att1_status_t st;

    if (write_tmp(data, size, path) != 0) {
        fprintf(stderr, "FAIL: fuzz_loader: %s: could not write temp file\n",
                label);
        return 0;
    }

    memset(&model, 0, sizeof(model));
    st = att1_model_load(path, &model);
    (void)unlink(path);

    if (expected_ok) {
        if (st != 0) {
            att1_model_free(&model);
            fprintf(stderr,
                    "FAIL: fuzz_loader: %s: expected success, got status %d\n",
                    label, (int)st);
            return 0;
        }
        att1_model_free(&model);
        printf("PASS: fuzz_loader: %s\n", label);
    } else {
        if (st == 0) {
            att1_model_free(&model);
            fprintf(stderr,
                    "FAIL: fuzz_loader: %s: loader accepted malformed input\n",
                    label);
            return 0;
        }
        att1_model_free(&model);
        printf("PASS: fuzz_loader: %s (rejected status=%d)\n",
               label, (int)st);
    }
    return 1;
}

/* Run one case against an existing file path. */
static int run_case_path(const char *label, const char *path, int expected_ok)
{
    att1_model model;
    att1_status_t st;

    memset(&model, 0, sizeof(model));
    st = att1_model_load(path, &model);

    if (expected_ok) {
        if (st != 0) {
            att1_model_free(&model);
            fprintf(stderr,
                    "FAIL: fuzz_loader: %s: expected success, got status %d\n",
                    label, (int)st);
            return 0;
        }
        att1_model_free(&model);
        printf("PASS: fuzz_loader: %s\n", label);
    } else {
        if (st == 0) {
            att1_model_free(&model);
            fprintf(stderr,
                    "FAIL: fuzz_loader: %s: loader accepted malformed input\n",
                    label);
            return 0;
        }
        att1_model_free(&model);
        printf("PASS: fuzz_loader: %s (rejected status=%d)\n",
               label, (int)st);
    }
    return 1;
}

/* ---- main ---- */

int main(void)
{
    int pass  = 0;
    int total = 0;

    unsigned char buf[VALID_BASE_SIZE];
    unsigned char zeros[VALID_BASE_SIZE];

#define RUN(label, data, size, ok) \
    do { total++; pass += run_case((label), (data), (size), (ok)); } while (0)

#define RUNP(label, path, ok) \
    do { total++; pass += run_case_path((label), (path), (ok)); } while (0)

    /* ---- VALID INPUTS (must be accepted) ---- */

    build_valid_base(buf);
    RUN("valid_minimal_v1", buf, VALID_BASE_SIZE, 1);

    RUNP("valid_dummy_model", "models/dummy/model.att1", 1);

    /* ---- HOSTILE: file is completely empty ---- */
    {
        unsigned char empty[1] = {0u};
        RUN("bad_empty_file", empty, 0u, 0);
    }

    /* ---- HOSTILE: 7 bytes — below magic threshold ---- */
    {
        static const unsigned char short7[7] =
            { 'A', 'T', 'T', '1', 'M', 'O', 'D' };
        RUN("bad_truncated_7bytes", short7, sizeof(short7), 0);
    }

    /* ---- HOSTILE: truncated at 40 bytes — header incomplete ---- */
    build_valid_base(buf);
    RUN("bad_truncated_40bytes", buf, 40u, 0);

    /* ---- HOSTILE: bad magic (first byte changed) ---- */
    build_valid_base(buf);
    buf[0] = 'X';
    RUN("bad_magic_byte0", buf, VALID_BASE_SIZE, 0);

    /* ---- HOSTILE: last magic byte wrong ---- */
    build_valid_base(buf);
    buf[7] = 0u;
    RUN("bad_magic_byte7", buf, VALID_BASE_SIZE, 0);

    /* ---- HOSTILE: version = 0 ---- */
    build_valid_base(buf);
    w32le(buf + 8, 0u);
    RUN("bad_version_0", buf, VALID_BASE_SIZE, 0);

    /* ---- HOSTILE: future version = 99 ---- */
    build_valid_base(buf);
    w32le(buf + 8, 99u);
    RUN("bad_version_99", buf, VALID_BASE_SIZE, 0);

    /* ---- HOSTILE: version = 1 but header_size != 80 ---- */
    build_valid_base(buf);
    w32le(buf + 12, 99u);
    RUN("bad_header_size_v1_mismatch", buf, VALID_BASE_SIZE, 0);

    /* ---- HOSTILE: version = 1 but header_size = 0 ---- */
    build_valid_base(buf);
    w32le(buf + 12, 0u);
    RUN("bad_header_size_zero", buf, VALID_BASE_SIZE, 0);

    /* ---- HOSTILE: config_size = 0 (must be 36) ---- */
    build_valid_base(buf);
    w64le(buf + 24, 0u);
    RUN("bad_config_size_zero", buf, VALID_BASE_SIZE, 0);

    /* ---- HOSTILE: config_size too large (would exceed ATT1_MODEL_CONFIG_SIZE) ---- */
    build_valid_base(buf);
    w64le(buf + 24, 999u);
    RUN("bad_config_size_too_large", buf, VALID_BASE_SIZE, 0);

    /* ---- HOSTILE: config_offset = UINT64_MAX ---- */
    build_valid_base(buf);
    w64le(buf + 16, UINT64_MAX);
    RUN("bad_config_offset_overflow", buf, VALID_BASE_SIZE, 0);

    /* ---- HOSTILE: config_offset points beyond EOF ---- */
    build_valid_base(buf);
    w64le(buf + 16, 4096u);
    RUN("bad_config_offset_past_eof", buf, VALID_BASE_SIZE, 0);

    /* ---- HOSTILE: data_size extends beyond EOF while data_offset is valid ---- */
    build_valid_base(buf);
    w64le(buf + 56, 1u);
    RUN("bad_data_size_past_eof", buf, VALID_BASE_SIZE, 0);

    /* ---- HOSTILE: data_offset past end of file ---- */
    build_valid_base(buf);
    w64le(buf + 48, UINT64_MAX);
    w64le(buf + 56, 1u);
    RUN("bad_data_offset_overflow", buf, VALID_BASE_SIZE, 0);

    /* ---- HOSTILE: tensor_count = UINT64_MAX (overflow in multiplication) ---- */
    build_valid_base(buf);
    w64le(buf + 40, UINT64_MAX);
    RUN("bad_tensor_count_max", buf, VALID_BASE_SIZE, 0);

    /* ---- HOSTILE: tensor_count requires descriptors that are absent ---- */
    build_valid_base(buf);
    w64le(buf + 40, 1u);
    RUN("bad_tensor_count_without_descriptor", buf, VALID_BASE_SIZE, 0);

    /* ---- HOSTILE: shard_size non-zero but shard_offset = 0 ---- */
    build_valid_base(buf);
    w64le(buf + 64, 0u);  /* shard_offset */
    w64le(buf + 72, 1u);  /* shard_size (inconsistent) */
    RUN("bad_shard_size_without_offset", buf, VALID_BASE_SIZE, 0);

    /* ---- HOSTILE: shard_offset overflows while shard_size is non-zero ---- */
    build_valid_base(buf);
    w64le(buf + 64, UINT64_MAX);
    w64le(buf + 72, 1u);
    RUN("bad_shard_offset_overflow", buf, VALID_BASE_SIZE, 0);

    /* ---- HOSTILE: all-zeros file (no valid magic) ---- */
    memset(zeros, 0, sizeof(zeros));
    RUN("bad_all_zeros", zeros, sizeof(zeros), 0);

    /* ---- summary ---- */

    printf("fuzz_loader: %d/%d PASS", pass, total);
    if (pass < total) {
        printf("  %d FAIL", total - pass);
        putchar('\n');
        return 1;
    }
    putchar('\n');
    return 0;
}
