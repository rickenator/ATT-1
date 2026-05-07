#include "att1_model.h"
#include "att1_shard_meta.h"
#include "att1_status.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

/*
 * Path to the checked-in fixture, relative to the project root (where
 * `make test` runs the binaries from).
 */
#define FIXTURE_PATH "models/shard_meta/model.att1"

/* Expected values derived from compiler/make_shard_meta_fixture.py. */
#define EXPECTED_TENSOR_COUNT   21u
#define EXPECTED_SHARD_COUNT    21u

/* ---------------------------------------------------------------------------
 * Helpers
 * ------------------------------------------------------------------------- */

static int read_file(const char *path, unsigned char **out_data, size_t *out_size)
{
    FILE *fp = fopen(path, "rb");
    long size = 0;
    unsigned char *data = NULL;

    if (fp == NULL) {
        return -1;
    }

    if ((fseek(fp, 0L, SEEK_END) != 0) || ((size = ftell(fp)) < 0) ||
        (fseek(fp, 0L, SEEK_SET) != 0)) {
        fclose(fp);
        return -1;
    }

    data = malloc((size_t)size);
    if ((data == NULL) && (size != 0)) {
        fclose(fp);
        return -1;
    }

    if ((size > 0) && (fread(data, 1u, (size_t)size, fp) != (size_t)size)) {
        free(data);
        fclose(fp);
        return -1;
    }

    fclose(fp);
    *out_data = data;
    *out_size = (size_t)size;
    return 0;
}

static int buffer_contains(const unsigned char *data, size_t size,
                           const char *needle)
{
    const size_t needle_len = strlen(needle);
    size_t i = 0u;

    if (needle_len == 0u) {
        return 1;
    }

    if (needle_len > size) {
        return 0;
    }

    for (i = 0u; i <= size - needle_len; i++) {
        if (memcmp(&data[i], needle, needle_len) == 0) {
            return 1;
        }
    }

    return 0;
}

/* ---------------------------------------------------------------------------
 * Test 1: Fixture loads with shard metadata populated
 * ------------------------------------------------------------------------- */

static int test_fixture_loads(void)
{
    att1_model model;
    uint64_t i = 0u;

    if (att1_model_load(FIXTURE_PATH, &model) != ATT1_OK) {
        fputs("fixture load failed\n", stderr);
        return 0;
    }

    /* Config must match the tiny model definition. */
    if ((model.config.vocab_size != 256u) ||
        (model.config.n_layers   != 2u) ||
        (model.config.n_heads    != 2u) ||
        (model.config.d_model    != 4u) ||
        (model.config.n_tiles    != 1u)) {
        fputs("fixture config check failed\n", stderr);
        att1_model_free(&model);
        return 0;
    }

    /* Tensor count */
    if (model.tensor_count != EXPECTED_TENSOR_COUNT) {
        fprintf(stderr, "fixture tensor_count check failed: got %" PRIu64 "\n",
                model.tensor_count);
        att1_model_free(&model);
        return 0;
    }

    /* Shard metadata must be populated. */
    if (model.shard_meta.count != EXPECTED_SHARD_COUNT ||
        model.shard_meta.records == NULL) {
        fprintf(stderr, "fixture shard_meta.count check failed: got %" PRIu64 "\n",
                model.shard_meta.count);
        att1_model_free(&model);
        return 0;
    }

    /* All records: tensor_id in order, tile_id=0, dtype=f32. */
    for (i = 0u; i < model.shard_meta.count; i++) {
        const att1_shard_meta_record *rec = &model.shard_meta.records[i];

        if (rec->tensor_id != (uint32_t)i) {
            fprintf(stderr,
                    "shard record %"PRIu64": tensor_id=%u expected %"PRIu64"\n",
                    i, rec->tensor_id, i);
            att1_model_free(&model);
            return 0;
        }

        if (rec->tile_id != 0u) {
            fprintf(stderr,
                    "shard record %"PRIu64": tile_id=%u expected 0\n",
                    i, rec->tile_id);
            att1_model_free(&model);
            return 0;
        }

        if (rec->dtype != ATT1_SHARD_DTYPE_F32) {
            fprintf(stderr,
                    "shard record %"PRIu64": dtype=%u expected %u\n",
                    i, rec->dtype, ATT1_SHARD_DTYPE_F32);
            att1_model_free(&model);
            return 0;
        }

        if (rec->replication_policy != ATT1_SHARD_REPL_NONE) {
            fprintf(stderr,
                    "shard record %"PRIu64": repl=%u expected %u\n",
                    i, rec->replication_policy, ATT1_SHARD_REPL_NONE);
            att1_model_free(&model);
            return 0;
        }

        if (rec->reduction_behavior != ATT1_SHARD_REDUCE_NONE) {
            fprintf(stderr,
                    "shard record %"PRIu64": reduce=%u expected %u\n",
                    i, rec->reduction_behavior, ATT1_SHARD_REDUCE_NONE);
            att1_model_free(&model);
            return 0;
        }

        /* Shape must match the tensor descriptor. */
        {
            uint32_t d = 0u;

            for (d = 0u; d < ATT1_MODEL_MAX_DIMS; d++) {
                if (rec->shape[d] != model.tensors[i].shape[d]) {
                    fprintf(stderr,
                            "shard record %"PRIu64": shape[%u]=%"PRIu64
                            " expected %"PRIu64"\n",
                            i, d, rec->shape[d], model.tensors[i].shape[d]);
                    att1_model_free(&model);
                    return 0;
                }
            }
        }
    }

    att1_model_free(&model);
    return 1;
}

/* ---------------------------------------------------------------------------
 * Test 2: att1-inspect prints shard metadata summary
 * ------------------------------------------------------------------------- */

static int test_inspect_output(void)
{
    unsigned char *data = NULL;
    size_t size = 0u;
    int ok = 0;

    if (system("build/att1-inspect models/shard_meta/model.att1"
               " > build/shard_meta_fixture_test/inspect.txt 2>&1") != 0) {
        fputs("att1-inspect exited non-zero on fixture\n", stderr);
        return 0;
    }

    if (read_file("build/shard_meta_fixture_test/inspect.txt",
                  &data, &size) != 0) {
        fputs("inspect output read failed\n", stderr);
        return 0;
    }

    ok = buffer_contains(data, size, "shard_meta: 21 records") &&
         buffer_contains(data, size, "shard[0]") &&
         buffer_contains(data, size, "tile=0") &&
         buffer_contains(data, size, "dtype=f32") &&
         buffer_contains(data, size, "repl=none") &&
         buffer_contains(data, size, "tok_embeddings.weight") &&
         buffer_contains(data, size, "output.weight");

    if (!ok) {
        fputs("inspect output content check failed\n", stderr);
        /* Print what we got for debugging. */
        if (size > 0u) {
            (void)fwrite(data, 1u, size, stderr);
        }
    }

    free(data);
    return ok;
}

/* ---------------------------------------------------------------------------
 * Test 3: No-metadata dummy model still loads without shard_meta
 * ------------------------------------------------------------------------- */

static int test_no_meta_model_unchanged(void)
{
    att1_model model;
    int ok = 0;

    if (att1_model_load("models/dummy/model.att1", &model) != ATT1_OK) {
        fputs("dummy model load failed\n", stderr);
        return 0;
    }

    ok = (model.shard_meta.count == 0u) && (model.shard_meta.records == NULL);

    if (!ok) {
        fputs("dummy model shard_meta not empty\n", stderr);
    }

    att1_model_free(&model);
    return ok;
}

/* ---------------------------------------------------------------------------
 * Test 4: att1-inspect works correctly on the no-metadata dummy model
 * ------------------------------------------------------------------------- */

static int test_inspect_no_meta(void)
{
    unsigned char *data = NULL;
    size_t size = 0u;
    int ok = 0;

    if (system("build/att1-inspect models/dummy/model.att1"
               " > build/shard_meta_fixture_test/inspect_no_meta.txt 2>&1") != 0) {
        fputs("att1-inspect exited non-zero on dummy model\n", stderr);
        return 0;
    }

    if (read_file("build/shard_meta_fixture_test/inspect_no_meta.txt",
                  &data, &size) != 0) {
        fputs("inspect no-meta output read failed\n", stderr);
        return 0;
    }

    /* Must have tensor output but no shard_meta line. */
    ok = buffer_contains(data, size, "tensor_count=21") &&
         !buffer_contains(data, size, "shard_meta:");

    if (!ok) {
        fputs("inspect no-meta output content check failed\n", stderr);
        if (size > 0u) {
            (void)fwrite(data, 1u, size, stderr);
        }
    }

    free(data);
    return ok;
}

/* ---------------------------------------------------------------------------
 * main
 * ------------------------------------------------------------------------- */

int main(void)
{
    const char *dir = "build/shard_meta_fixture_test";

    (void)mkdir(dir, 0777);

#define RUN(name)                                           \
    do {                                                    \
        if (!name()) {                                      \
            fputs(#name " failed\n", stderr);               \
            return 1;                                       \
        }                                                   \
    } while (0)

    RUN(test_fixture_loads);
    RUN(test_inspect_output);
    RUN(test_no_meta_model_unchanged);
    RUN(test_inspect_no_meta);

#undef RUN

    puts("shard_meta_fixture test passed");
    return 0;
}
