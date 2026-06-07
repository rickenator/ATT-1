#include "att1_backend.h"
#include "att1_infer.h"
#include "att1_model.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define Q4_FIXTURE_PATH "models/q4_tiny/model.att1"
#define DUMMY_MODEL_PATH "models/dummy/model.att1"

#define PASS(name) do { printf("PASS: %s\n", (name)); } while (0)
#define FAIL(name, msg) \
    do { printf("FAIL: %s -- %s\n", (name), (msg)); return 1; } while (0)

/* ── tests ──────────────────────────────────────────────────────────────── */

static int test_q4_create_and_decode(void)
{
    const char *name = "q4_create_and_decode";
    att1_model    model;
    att1_infer_t *infer  = NULL;
    att1_status_t status;
    uint32_t token = 0u;

    if (att1_model_load(Q4_FIXTURE_PATH, &model) != ATT1_OK) { FAIL(name, "failed to load q4 fixture"); }

    status = att1_infer_create_q4(&model, &infer);
    if (status != ATT1_OK) { att1_model_free(&model); FAIL(name, "create_q4 returned error"); }

    status = att1_infer_decode_token(infer, 0u, &token);
    att1_infer_destroy(infer);
    att1_model_free(&model);
    if (status != ATT1_OK) { FAIL(name, "decode_token returned error"); }

    PASS(name);
    return 0;
}

static int test_q4_backend_is_cpu_q4(void)
{
    const char *name = "q4_backend_is_cpu_q4";
    att1_backend *backend = NULL;
    att1_status_t status;

    /* Verify that att1_backend_cpu_q4_create produces a backend named "cpu-q4". */
    status = att1_backend_cpu_q4_create(&backend);
    if (status != ATT1_OK || backend == NULL || backend->ops == NULL || backend->ops->name == NULL) {
        FAIL(name, "could not create cpu-q4 backend");
    }
    if (strcmp(backend->ops->name, "cpu-q4") != 0) {
        att1_backend_destroy(backend);
        FAIL(name, "backend name is not cpu-q4");
    }
    att1_backend_destroy(backend);
    PASS(name);
    return 0;
}

static int test_q4_logit_count(void)
{
    const char *name = "q4_logit_count";
    att1_model    model;
    att1_infer_t *infer  = NULL;
    att1_status_t status;
    uint32_t token = 0u;
    size_t count = 0u;
    const float *logits = NULL;

    if (att1_model_load(Q4_FIXTURE_PATH, &model) != ATT1_OK) { FAIL(name, "failed to load q4 fixture"); }

    status = att1_infer_create_q4(&model, &infer);
    if (status != ATT1_OK) { att1_model_free(&model); FAIL(name, "create_q4 failed"); }

    status = att1_infer_decode_token(infer, 0u, &token);
    if (status != ATT1_OK) { att1_infer_destroy(infer); att1_model_free(&model); FAIL(name, "decode_token failed"); }

    logits = att1_infer_logits(infer, &count);
    att1_infer_destroy(infer);
    att1_model_free(&model);
    if (logits == NULL || count != 256u) { FAIL(name, "unexpected logit count (expected 256)"); }

    PASS(name);
    return 0;
}

static int test_q4_create_on_f32_model_fails(void)
{
    const char *name = "q4_create_on_f32_model_fails";
    att1_model    model;
    att1_infer_t *infer  = NULL;
    att1_status_t status;

    if (att1_model_load(DUMMY_MODEL_PATH, &model) != ATT1_OK) { FAIL(name, "failed to load dummy model"); }

    status = att1_infer_create_q4(&model, &infer);
    att1_model_free(&model);
    if (status == ATT1_OK) {
        att1_infer_destroy(infer);
        FAIL(name, "expected error but got ATT1_OK");
    }
    if (infer != NULL) {
        att1_infer_destroy(infer);
        FAIL(name, "out_infer should be NULL on failure");
    }

    PASS(name);
    return 0;
}

static int test_q4_fixture_rejected_by_create(void)
{
    const char *name = "q4_fixture_rejected_by_create";
    att1_model    model;
    att1_infer_t *infer  = NULL;
    att1_status_t status;

    if (att1_model_load(Q4_FIXTURE_PATH, &model) != ATT1_OK) { FAIL(name, "failed to load q4 fixture"); }

    /* att1_infer_create (f32/q8 path) should reject q4 model */
    status = att1_infer_create(&model, &infer);
    att1_model_free(&model);
    if (status == ATT1_OK) {
        att1_infer_destroy(infer);
        FAIL(name, "expected error but got ATT1_OK");
    }

    PASS(name);
    return 0;
}

static int test_q4_position_advances(void)
{
    const char *name = "q4_position_advances";
    att1_model    model;
    att1_infer_t *infer  = NULL;
    att1_status_t status;
    uint32_t token = 0u;
    int step = 0;
    size_t pos = 0u;

    if (att1_model_load(Q4_FIXTURE_PATH, &model) != ATT1_OK) { FAIL(name, "failed to load q4 fixture"); }

    status = att1_infer_create_q4(&model, &infer);
    if (status != ATT1_OK) { att1_model_free(&model); FAIL(name, "create_q4 failed"); }

    for (step = 0; step < 4; step++) {
        status = att1_infer_decode_token(infer, (uint32_t)step, &token);
        if (status != ATT1_OK) {
            att1_infer_destroy(infer); att1_model_free(&model);
            FAIL(name, "decode_token failed mid-loop");
        }
    }

    if (att1_infer_position(infer, &pos) != ATT1_OK || pos != 4u) {
        att1_infer_destroy(infer); att1_model_free(&model);
        FAIL(name, "position should be 4 after 4 decode steps");
    }

    att1_infer_destroy(infer);
    att1_model_free(&model);
    PASS(name);
    return 0;
}

static int test_f32_path_unchanged(void)
{
    const char *name = "f32_path_unchanged";
    att1_model    model;
    att1_infer_t *infer  = NULL;
    att1_status_t status;
    uint32_t token = 0u;

    if (att1_model_load(DUMMY_MODEL_PATH, &model) != ATT1_OK) { FAIL(name, "failed to load dummy model"); }

    status = att1_infer_create(&model, &infer);
    if (status != ATT1_OK) { att1_model_free(&model); FAIL(name, "create failed"); }

    status = att1_infer_decode_token(infer, 0u, &token);
    att1_infer_destroy(infer);
    att1_model_free(&model);
    if (status != ATT1_OK) { FAIL(name, "decode_token failed"); }

    PASS(name);
    return 0;
}

/* ── main ───────────────────────────────────────────────────────────────── */

int main(void)
{
    int failures = 0;

    failures += test_q4_create_and_decode();
    failures += test_q4_backend_is_cpu_q4();
    failures += test_q4_logit_count();
    failures += test_q4_create_on_f32_model_fails();
    failures += test_q4_fixture_rejected_by_create();
    failures += test_q4_position_advances();
    failures += test_f32_path_unchanged();

    if (failures == 0) {
        printf("All q4 infer tests passed.\n");
    } else {
        printf("%d q4 infer test(s) FAILED.\n", failures);
    }
    return failures == 0 ? 0 : 1;
}
