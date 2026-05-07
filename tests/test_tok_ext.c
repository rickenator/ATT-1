#include "att1_tok_ext.h"
#include "att1_status.h"

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* ── helpers ─────────────────────────────────────────────────────────────── */

/*
 * Write the given token IDs to a plain-text file, one per line.
 * Returns 0 on success, -1 on failure.
 */
static int write_ids_file(const char *path, const uint32_t *ids, size_t count)
{
    FILE *fp = fopen(path, "w");
    size_t i = 0u;

    if (fp == NULL) {
        return -1;
    }
    for (i = 0u; i < count; i++) {
        if (fprintf(fp, "%u\n", ids[i]) < 0) {
            fclose(fp);
            return -1;
        }
    }
    fclose(fp);
    return 0;
}

/*
 * Expected parser failures print public diagnostics to stderr.  Suppress only
 * those calls so a passing test run does not look like it found CUDA errors.
 */
static int silence_stderr(int *saved_fd)
{
    int null_fd = -1;

    fflush(stderr);
    *saved_fd = dup(fileno(stderr));
    if (*saved_fd < 0) {
        return -1;
    }

    null_fd = open("/dev/null", O_WRONLY);
    if (null_fd < 0) {
        close(*saved_fd);
        *saved_fd = -1;
        return -1;
    }

    if (dup2(null_fd, fileno(stderr)) < 0) {
        close(null_fd);
        close(*saved_fd);
        *saved_fd = -1;
        return -1;
    }

    close(null_fd);
    return 0;
}

static void restore_stderr(int saved_fd)
{
    if (saved_fd >= 0) {
        fflush(stderr);
        dup2(saved_fd, fileno(stderr));
        close(saved_fd);
    }
}

static att1_status_t parse_ids_str_quiet(const char  *str,
                                         uint32_t     vocab_size,
                                         uint32_t   **out_ids,
                                         size_t      *out_count)
{
    int saved_fd = -1;
    att1_status_t rc = ATT1_OK;

    (void)silence_stderr(&saved_fd);
    rc = att1_tok_ext_parse_ids_str(str, vocab_size, out_ids, out_count);
    restore_stderr(saved_fd);
    return rc;
}

static att1_status_t parse_ids_file_quiet(const char  *path,
                                          uint32_t     vocab_size,
                                          uint32_t   **out_ids,
                                          size_t      *out_count)
{
    int saved_fd = -1;
    att1_status_t rc = ATT1_OK;

    (void)silence_stderr(&saved_fd);
    rc = att1_tok_ext_parse_ids_file(path, vocab_size, out_ids, out_count);
    restore_stderr(saved_fd);
    return rc;
}

/* ── test cases ──────────────────────────────────────────────────────────── */

/*
 * Test 1: valid comma-separated string is parsed correctly.
 */
static int test_parse_valid(void)
{
    uint32_t *ids = NULL;
    size_t count = 0u;
    const att1_status_t rc =
        att1_tok_ext_parse_ids_str("1,2,3", 16u, &ids, &count);

    if (rc != ATT1_OK) {
        fprintf(stderr, "parse_valid: expected OK, got %d\n", (int)rc);
        return 0;
    }
    if (count != 3u) {
        fprintf(stderr, "parse_valid: expected count=3, got %zu\n", count);
        free(ids);
        return 0;
    }
    if ((ids[0] != 1u) || (ids[1] != 2u) || (ids[2] != 3u)) {
        fprintf(stderr, "parse_valid: unexpected ID values\n");
        free(ids);
        return 0;
    }
    free(ids);
    return 1;
}

/*
 * Test 2: single token ID "0" is valid.
 */
static int test_parse_single_zero(void)
{
    uint32_t *ids = NULL;
    size_t count = 0u;
    const att1_status_t rc =
        att1_tok_ext_parse_ids_str("0", 16u, &ids, &count);

    if ((rc != ATT1_OK) || (count != 1u) || (ids[0] != 0u)) {
        fprintf(stderr, "parse_single_zero: unexpected result rc=%d count=%zu\n",
                (int)rc, count);
        free(ids);
        return 0;
    }
    free(ids);
    return 1;
}

/*
 * Test 3: maximum valid token ID (vocab_size - 1) is accepted.
 */
static int test_parse_max_id(void)
{
    uint32_t *ids = NULL;
    size_t count = 0u;
    const att1_status_t rc =
        att1_tok_ext_parse_ids_str("15", 16u, &ids, &count);

    if ((rc != ATT1_OK) || (count != 1u) || (ids[0] != 15u)) {
        fprintf(stderr, "parse_max_id: unexpected result rc=%d\n", (int)rc);
        free(ids);
        return 0;
    }
    free(ids);
    return 1;
}

/*
 * Test 4: empty string returns ATT1_ERR_INVALID_ARG.
 */
static int test_parse_empty_string(void)
{
    uint32_t *ids = NULL;
    size_t count = 0u;
    const att1_status_t rc =
        parse_ids_str_quiet("", 16u, &ids, &count);

    if (rc == ATT1_OK) {
        fputs("parse_empty_string: expected error, got ATT1_OK\n", stderr);
        free(ids);
        return 0;
    }
    return 1;
}

/*
 * Test 5: NULL string returns error.
 */
static int test_parse_null_string(void)
{
    uint32_t *ids = NULL;
    size_t count = 0u;
    const att1_status_t rc =
        parse_ids_str_quiet(NULL, 16u, &ids, &count);

    if (rc == ATT1_OK) {
        fputs("parse_null_string: expected error, got ATT1_OK\n", stderr);
        free(ids);
        return 0;
    }
    return 1;
}

/*
 * Test 6: alphabetic token returns ATT1_ERR_BAD_FORMAT.
 */
static int test_parse_malformed_alpha(void)
{
    uint32_t *ids = NULL;
    size_t count = 0u;
    const att1_status_t rc =
        parse_ids_str_quiet("1,foo,3", 16u, &ids, &count);

    if (rc != ATT1_ERR_BAD_FORMAT) {
        fprintf(stderr,
                "parse_malformed_alpha: expected BAD_FORMAT, got %d\n",
                (int)rc);
        free(ids);
        return 0;
    }
    return 1;
}

/*
 * Test 7: token ID equal to vocab_size is out of range → ATT1_ERR_BAD_FORMAT.
 */
static int test_parse_out_of_range(void)
{
    uint32_t *ids = NULL;
    size_t count = 0u;
    /* vocab_size=16; token ID 16 is out of range */
    const att1_status_t rc =
        parse_ids_str_quiet("1,16,3", 16u, &ids, &count);

    if (rc != ATT1_ERR_BAD_FORMAT) {
        fprintf(stderr, "parse_out_of_range: expected BAD_FORMAT, got %d\n",
                (int)rc);
        free(ids);
        return 0;
    }
    return 1;
}

/*
 * Test 8: leading minus sign is rejected → ATT1_ERR_BAD_FORMAT.
 */
static int test_parse_negative(void)
{
    uint32_t *ids = NULL;
    size_t count = 0u;
    const att1_status_t rc =
        parse_ids_str_quiet("-1", 16u, &ids, &count);

    if (rc != ATT1_ERR_BAD_FORMAT) {
        fprintf(stderr, "parse_negative: expected BAD_FORMAT, got %d\n",
                (int)rc);
        free(ids);
        return 0;
    }
    return 1;
}

/*
 * Test 9: empty segment (double comma) is rejected → ATT1_ERR_BAD_FORMAT.
 */
static int test_parse_empty_segment(void)
{
    uint32_t *ids = NULL;
    size_t count = 0u;
    const att1_status_t rc =
        parse_ids_str_quiet("1,,3", 16u, &ids, &count);

    if (rc != ATT1_ERR_BAD_FORMAT) {
        fprintf(stderr, "parse_empty_segment: expected BAD_FORMAT, got %d\n",
                (int)rc);
        free(ids);
        return 0;
    }
    return 1;
}

/*
 * Test 10: trailing comma is rejected → ATT1_ERR_BAD_FORMAT.
 */
static int test_parse_trailing_comma(void)
{
    uint32_t *ids = NULL;
    size_t count = 0u;
    const att1_status_t rc =
        parse_ids_str_quiet("1,2,", 16u, &ids, &count);

    if (rc != ATT1_ERR_BAD_FORMAT) {
        fprintf(stderr, "parse_trailing_comma: expected BAD_FORMAT, got %d\n",
                (int)rc);
        free(ids);
        return 0;
    }
    return 1;
}

/*
 * Test 11: NULL out_ids argument returns error.
 */
static int test_parse_null_out(void)
{
    size_t count = 0u;
    const att1_status_t rc =
        parse_ids_str_quiet("1,2,3", 16u, NULL, &count);

    if (rc == ATT1_OK) {
        fputs("parse_null_out: expected error, got ATT1_OK\n", stderr);
        return 0;
    }
    return 1;
}

/*
 * Test 12: file-based parsing of valid IDs succeeds.
 */
static int test_file_valid(void)
{
    const uint32_t expected[] = { 1u, 2u, 3u };
    const char *path = "build/tok_ext_test_valid.txt";
    uint32_t *ids = NULL;
    size_t count = 0u;
    att1_status_t rc = ATT1_OK;

    if (write_ids_file(path, expected, 3u) != 0) {
        fputs("file_valid: could not write test file\n", stderr);
        return 0;
    }

    rc = att1_tok_ext_parse_ids_file(path, 16u, &ids, &count);
    if (rc != ATT1_OK) {
        fprintf(stderr, "file_valid: expected OK, got %d\n", (int)rc);
        free(ids);
        return 0;
    }
    if ((count != 3u) ||
        (ids[0] != 1u) || (ids[1] != 2u) || (ids[2] != 3u)) {
        fputs("file_valid: unexpected values\n", stderr);
        free(ids);
        return 0;
    }
    free(ids);
    return 1;
}

/*
 * Test 13: file with blank lines and comment lines is handled correctly.
 */
static int test_file_comments(void)
{
    const char *path = "build/tok_ext_test_comments.txt";
    uint32_t *ids = NULL;
    size_t count = 0u;
    FILE *fp = NULL;
    att1_status_t rc = ATT1_OK;

    fp = fopen(path, "w");
    if (fp == NULL) {
        fputs("file_comments: could not create test file\n", stderr);
        return 0;
    }
    fputs("# header comment\n", fp);
    fputs("\n", fp);
    fputs("5\n", fp);
    fputs("# another comment\n", fp);
    fputs("10\n", fp);
    fclose(fp);

    rc = att1_tok_ext_parse_ids_file(path, 16u, &ids, &count);
    if (rc != ATT1_OK) {
        fprintf(stderr, "file_comments: expected OK, got %d\n", (int)rc);
        free(ids);
        return 0;
    }
    if ((count != 2u) || (ids[0] != 5u) || (ids[1] != 10u)) {
        fputs("file_comments: unexpected values\n", stderr);
        free(ids);
        return 0;
    }
    free(ids);
    return 1;
}

/*
 * Test 14: non-existent file returns ATT1_ERR_NOT_FOUND.
 */
static int test_file_not_found(void)
{
    uint32_t *ids = NULL;
    size_t count = 0u;
    const att1_status_t rc =
        parse_ids_file_quiet("build/tok_ext_nonexistent_xyz.txt",
                             16u, &ids, &count);

    if (rc != ATT1_ERR_NOT_FOUND) {
        fprintf(stderr, "file_not_found: expected NOT_FOUND, got %d\n",
                (int)rc);
        free(ids);
        return 0;
    }
    return 1;
}

/*
 * Test 15: file with an out-of-range ID returns ATT1_ERR_BAD_FORMAT.
 */
static int test_file_out_of_range(void)
{
    const char *path = "build/tok_ext_test_oor.txt";
    uint32_t *ids = NULL;
    size_t count = 0u;
    FILE *fp = NULL;
    att1_status_t rc = ATT1_OK;

    fp = fopen(path, "w");
    if (fp == NULL) {
        fputs("file_out_of_range: could not create test file\n", stderr);
        return 0;
    }
    fputs("1\n16\n3\n", fp); /* 16 >= vocab_size=16 */
    fclose(fp);

    rc = parse_ids_file_quiet(path, 16u, &ids, &count);
    if (rc != ATT1_ERR_BAD_FORMAT) {
        fprintf(stderr, "file_out_of_range: expected BAD_FORMAT, got %d\n",
                (int)rc);
        free(ids);
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

    RUN(parse_valid)
    RUN(parse_single_zero)
    RUN(parse_max_id)
    RUN(parse_empty_string)
    RUN(parse_null_string)
    RUN(parse_malformed_alpha)
    RUN(parse_out_of_range)
    RUN(parse_negative)
    RUN(parse_empty_segment)
    RUN(parse_trailing_comma)
    RUN(parse_null_out)
    RUN(file_valid)
    RUN(file_comments)
    RUN(file_not_found)
    RUN(file_out_of_range)

#undef RUN

    if (!pass) {
        fputs("tok_ext test failed\n", stderr);
        return 1;
    }

    puts("tok_ext test passed");
    return 0;
}
