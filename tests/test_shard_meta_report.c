#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

/*
 * M37: shard metadata reporting and trace integration tests.
 *
 * Verifies that att1-bench and att1-inspect emit the correct shard metadata
 * summary lines for both a metadata-present fixture and the baseline
 * no-metadata dummy model.
 */

#define OUT_DIR "build/shard_meta_report_test"
#define FIXTURE "models/shard_meta/model.att1"
#define DUMMY   "models/dummy/model.att1"

/* ---------------------------------------------------------------------------
 * Helpers
 * ------------------------------------------------------------------------- */

static int read_file_text(const char *path, char *buf, size_t capacity)
{
    FILE *fp = fopen(path, "rb");
    size_t nread = 0u;

    if (fp == NULL) {
        return -1;
    }

    nread = fread(buf, 1u, capacity - 1u, fp);
    if (ferror(fp) != 0) {
        fclose(fp);
        return -1;
    }

    buf[nread] = '\0';
    fclose(fp);
    return 0;
}

/* ---------------------------------------------------------------------------
 * Test 1: bench on dummy model prints shard_meta=absent
 * ------------------------------------------------------------------------- */

static int test_bench_absent(void)
{
    char output[4096];

    if (system("build/att1-bench --model " DUMMY
               " --prompt hi --tokens 1 --mode single"
               " > " OUT_DIR "/bench_absent.txt 2>&1") != 0) {
        fputs("bench on dummy model failed\n", stderr);
        return 0;
    }

    if (read_file_text(OUT_DIR "/bench_absent.txt",
                       output, sizeof(output)) != 0) {
        fputs("read bench_absent output failed\n", stderr);
        return 0;
    }

    if (strstr(output, "shard_meta=absent") == NULL) {
        fputs("bench_absent: shard_meta=absent not found\n", stderr);
        fputs(output, stderr);
        return 0;
    }

    /* Existing counters must still be present. */
    if (strstr(output, "mode=single") == NULL ||
        strstr(output, "tokens_decoded=") == NULL) {
        fputs("bench_absent: expected counters missing\n", stderr);
        return 0;
    }

    return 1;
}

/* ---------------------------------------------------------------------------
 * Test 2: bench on fixture model prints full shard metadata summary
 * ------------------------------------------------------------------------- */

static int test_bench_present(void)
{
    char output[4096];

    if (system("build/att1-bench --model " FIXTURE
               " --prompt hi --tokens 1 --mode single"
               " > " OUT_DIR "/bench_present.txt 2>&1") != 0) {
        fputs("bench on fixture model failed\n", stderr);
        return 0;
    }

    if (read_file_text(OUT_DIR "/bench_present.txt",
                       output, sizeof(output)) != 0) {
        fputs("read bench_present output failed\n", stderr);
        return 0;
    }

    if (strstr(output, "shard_meta=present") == NULL ||
        strstr(output, "shard_meta_count=21") == NULL ||
        strstr(output, "shard_meta_assigned=21") == NULL ||
        strstr(output, "shard_meta_unassigned=0") == NULL ||
        strstr(output, "shard_meta_tiles=1") == NULL ||
        strstr(output, "shard_meta_aimus=1") == NULL ||
        strstr(output, "shard_meta_dtype_f32=21") == NULL ||
        strstr(output, "shard_meta_dtype_q8=0") == NULL) {
        fputs("bench_present: expected shard_meta summary missing\n", stderr);
        fputs(output, stderr);
        return 0;
    }

    /* Existing counters must still be present. */
    if (strstr(output, "mode=single") == NULL ||
        strstr(output, "tokens_decoded=") == NULL) {
        fputs("bench_present: expected counters missing\n", stderr);
        return 0;
    }

    return 1;
}

/* ---------------------------------------------------------------------------
 * Test 3: inspect on fixture model prints shard metadata summary header
 * ------------------------------------------------------------------------- */

static int test_inspect_summary(void)
{
    char output[16384];

    if (system("build/att1-inspect " FIXTURE
               " > " OUT_DIR "/inspect_summary.txt 2>&1") != 0) {
        fputs("inspect on fixture failed\n", stderr);
        return 0;
    }

    if (read_file_text(OUT_DIR "/inspect_summary.txt",
                       output, sizeof(output)) != 0) {
        fputs("read inspect_summary output failed\n", stderr);
        return 0;
    }

    if (strstr(output, "shard_meta: 21 records") == NULL ||
        strstr(output, "shard_meta_tiles=1") == NULL ||
        strstr(output, "shard_meta_aimus=1") == NULL ||
        strstr(output, "shard_meta_assigned=21") == NULL ||
        strstr(output, "shard_meta_unassigned=0") == NULL ||
        strstr(output, "shard_meta_dtype_f32=21") == NULL ||
        strstr(output, "shard_meta_dtype_q8=0") == NULL) {
        fputs("inspect_summary: expected summary lines missing\n", stderr);
        fputs(output, stderr);
        return 0;
    }

    /* Per-record detail must still be present. */
    if (strstr(output, "shard[0]") == NULL ||
        strstr(output, "tok_embeddings.weight") == NULL ||
        strstr(output, "output.weight") == NULL) {
        fputs("inspect_summary: per-record detail missing\n", stderr);
        return 0;
    }

    return 1;
}

/* ---------------------------------------------------------------------------
 * Test 4: inspect on dummy model has no shard_meta summary lines
 * ------------------------------------------------------------------------- */

static int test_inspect_absent(void)
{
    char output[8192];

    if (system("build/att1-inspect " DUMMY
               " > " OUT_DIR "/inspect_absent.txt 2>&1") != 0) {
        fputs("inspect on dummy failed\n", stderr);
        return 0;
    }

    if (read_file_text(OUT_DIR "/inspect_absent.txt",
                       output, sizeof(output)) != 0) {
        fputs("read inspect_absent output failed\n", stderr);
        return 0;
    }

    if (strstr(output, "shard_meta") != NULL) {
        fputs("inspect_absent: unexpected shard_meta output\n", stderr);
        fputs(output, stderr);
        return 0;
    }

    if (strstr(output, "tensor_count=21") == NULL) {
        fputs("inspect_absent: tensor_count line missing\n", stderr);
        return 0;
    }

    return 1;
}

/* ---------------------------------------------------------------------------
 * main
 * ------------------------------------------------------------------------- */

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

    RUN(test_bench_absent);
    RUN(test_bench_present);
    RUN(test_inspect_summary);
    RUN(test_inspect_absent);

#undef RUN

    puts("shard_meta_report test passed");
    return 0;
}
