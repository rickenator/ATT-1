#include "att1_backend.h"
#include "att1_cluster_infer.h"
#include "att1_model.h"

#include <stdio.h>
#include <string.h>

#define Q4_FIXTURE_PATH  "models/q4_tiny/model.att1"
#define DUMMY_MODEL_PATH "models/dummy/model.att1"

#define PASS(name) do { printf("PASS: %s\n", (name)); } while (0)
#define FAIL(name, msg) \
    do { printf("FAIL: %s -- %s\n", (name), (msg)); return 1; } while (0)

/* ── helpers ─────────────────────────────────────────────────────────────── */

static att1_cluster_infer_config default_config(void)
{
    att1_cluster_infer_config cfg;
    cfg.tile_count = 2u;
    cfg.fabric_queue_capacity = 4u;
    cfg.fabric_max_payload_bytes = 0u; /* auto */
    cfg.shard_plan_mode = ATT1_SHARD_PLAN_RUNTIME;
    return cfg;
}

/* ── tests ───────────────────────────────────────────────────────────────── */

static int test_q4_cluster_create_and_decode(void)
{
    const char *name = "q4_cluster_create_and_decode";
    att1_model model;
    att1_cluster_infer_t *cluster = NULL;
    att1_cluster_infer_config cfg = default_config();
    uint32_t token = 0u;
    att1_status_t status;

    if (att1_model_load(Q4_FIXTURE_PATH, &model) != ATT1_OK) {
        FAIL(name, "failed to load q4 fixture");
    }

    status = att1_cluster_infer_create_q4(&model, &cfg, &cluster);
    if (status != ATT1_OK) {
        att1_model_free(&model);
        FAIL(name, "att1_cluster_infer_create_q4 returned error");
    }

    status = att1_cluster_infer_decode_token(cluster, 0u, &token);
    att1_cluster_infer_destroy(cluster);
    att1_model_free(&model);
    if (status != ATT1_OK) {
        FAIL(name, "decode_token returned error");
    }

    PASS(name);
    return 0;
}

static int test_q4_cluster_backend_name(void)
{
    const char *name = "q4_cluster_backend_name";
    att1_backend *backend = NULL;
    att1_status_t status;

    status = att1_backend_cpu_q4_create(&backend);
    if ((status != ATT1_OK) || (backend == NULL) || (backend->ops == NULL) ||
        (backend->ops->name == NULL)) {
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

static int test_q4_cluster_logit_count(void)
{
    const char *name = "q4_cluster_logit_count";
    att1_model model;
    att1_cluster_infer_t *cluster = NULL;
    att1_cluster_infer_config cfg = default_config();
    uint32_t token = 0u;
    size_t count = 0u;
    const float *logits = NULL;
    att1_status_t status;

    if (att1_model_load(Q4_FIXTURE_PATH, &model) != ATT1_OK) {
        FAIL(name, "failed to load q4 fixture");
    }

    status = att1_cluster_infer_create_q4(&model, &cfg, &cluster);
    if (status != ATT1_OK) {
        att1_model_free(&model);
        FAIL(name, "create_q4 failed");
    }

    status = att1_cluster_infer_decode_token(cluster, 0u, &token);
    if (status != ATT1_OK) {
        att1_cluster_infer_destroy(cluster);
        att1_model_free(&model);
        FAIL(name, "decode_token failed");
    }

    logits = att1_cluster_infer_logits(cluster, &count);
    att1_cluster_infer_destroy(cluster);
    att1_model_free(&model);

    if ((logits == NULL) || (count != 256u)) {
        FAIL(name, "unexpected logit count (expected 256)");
    }

    PASS(name);
    return 0;
}

static int test_q4_cluster_f32_model_rejected(void)
{
    const char *name = "q4_cluster_f32_model_rejected";
    att1_model model;
    att1_cluster_infer_t *cluster = NULL;
    att1_cluster_infer_config cfg = default_config();
    att1_status_t status;

    if (att1_model_load(DUMMY_MODEL_PATH, &model) != ATT1_OK) {
        FAIL(name, "failed to load dummy model");
    }

    status = att1_cluster_infer_create_q4(&model, &cfg, &cluster);
    att1_model_free(&model);
    if (status == ATT1_OK) {
        att1_cluster_infer_destroy(cluster);
        FAIL(name, "expected error but got ATT1_OK");
    }

    PASS(name);
    return 0;
}

static int test_q4_cluster_f32_path_unchanged(void)
{
    const char *name = "q4_cluster_f32_path_unchanged";
    att1_model model;
    att1_cluster_infer_t *cluster = NULL;
    att1_cluster_infer_config cfg = default_config();
    uint32_t token = 0u;
    att1_status_t status;

    if (att1_model_load(DUMMY_MODEL_PATH, &model) != ATT1_OK) {
        FAIL(name, "failed to load dummy model");
    }

    status = att1_cluster_infer_create(&model, &cfg, &cluster);
    if (status != ATT1_OK) {
        att1_model_free(&model);
        FAIL(name, "f32 cluster create failed");
    }

    status = att1_cluster_infer_decode_token(cluster, 0u, &token);
    att1_cluster_infer_destroy(cluster);
    att1_model_free(&model);
    if (status != ATT1_OK) {
        FAIL(name, "f32 cluster decode failed");
    }

    PASS(name);
    return 0;
}

static int test_q4_cluster_position_advances(void)
{
    const char *name = "q4_cluster_position_advances";
    att1_model model;
    att1_cluster_infer_t *cluster = NULL;
    att1_cluster_infer_config cfg = default_config();
    uint32_t token = 0u;
    size_t pos = 0u;
    att1_status_t status;

    if (att1_model_load(Q4_FIXTURE_PATH, &model) != ATT1_OK) {
        FAIL(name, "failed to load q4 fixture");
    }

    status = att1_cluster_infer_create_q4(&model, &cfg, &cluster);
    if (status != ATT1_OK) {
        att1_model_free(&model);
        FAIL(name, "create_q4 failed");
    }

    if (att1_cluster_infer_decode_token(cluster, 0u, &token) != ATT1_OK) {
        att1_cluster_infer_destroy(cluster);
        att1_model_free(&model);
        FAIL(name, "first decode failed");
    }

    if (att1_cluster_infer_position(cluster, &pos) != ATT1_OK) {
        att1_cluster_infer_destroy(cluster);
        att1_model_free(&model);
        FAIL(name, "cluster_infer_position failed");
    }

    att1_cluster_infer_destroy(cluster);
    att1_model_free(&model);

    if (pos != 1u) {
        FAIL(name, "position did not advance to 1");
    }

    PASS(name);
    return 0;
}

static int test_q4_cluster_fabric_packets_nonzero(void)
{
    const char *name = "q4_cluster_fabric_packets_nonzero";
    att1_model model;
    att1_cluster_infer_t *cluster = NULL;
    att1_cluster_infer_config cfg = default_config();
    att1_cluster_tile_counters counters;
    uint32_t token = 0u;
    att1_status_t status;

    if (att1_model_load(Q4_FIXTURE_PATH, &model) != ATT1_OK) {
        FAIL(name, "failed to load q4 fixture");
    }

    status = att1_cluster_infer_create_q4(&model, &cfg, &cluster);
    if (status != ATT1_OK) {
        att1_model_free(&model);
        FAIL(name, "create_q4 failed");
    }

    if (att1_cluster_infer_decode_token(cluster, 0u, &token) != ATT1_OK) {
        att1_cluster_infer_destroy(cluster);
        att1_model_free(&model);
        FAIL(name, "decode_token failed");
    }

    if (att1_cluster_infer_get_tile_counters(cluster, 0u, &counters) != ATT1_OK) {
        att1_cluster_infer_destroy(cluster);
        att1_model_free(&model);
        FAIL(name, "get_tile_counters failed");
    }

    att1_cluster_infer_destroy(cluster);
    att1_model_free(&model);

    if (counters.activations_received == 0u) {
        FAIL(name, "tile 0 activations_received is zero after decode");
    }

    PASS(name);
    return 0;
}

static int test_q4_cluster_metadata_plan_rejected(void)
{
    const char *name = "q4_cluster_metadata_plan_rejected";
    att1_model model;
    att1_cluster_infer_t *cluster = NULL;
    att1_cluster_infer_config cfg = default_config();
    att1_status_t status;

    cfg.shard_plan_mode = ATT1_SHARD_PLAN_METADATA;

    if (att1_model_load(Q4_FIXTURE_PATH, &model) != ATT1_OK) {
        FAIL(name, "failed to load q4 fixture");
    }

    status = att1_cluster_infer_create_q4(&model, &cfg, &cluster);
    att1_model_free(&model);
    if (status == ATT1_OK) {
        att1_cluster_infer_destroy(cluster);
        FAIL(name, "expected error for metadata plan but got ATT1_OK");
    }

    PASS(name);
    return 0;
}

/* ── main ────────────────────────────────────────────────────────────────── */

int main(void)
{
    int failures = 0;

    failures += test_q4_cluster_create_and_decode();
    failures += test_q4_cluster_backend_name();
    failures += test_q4_cluster_logit_count();
    failures += test_q4_cluster_f32_model_rejected();
    failures += test_q4_cluster_f32_path_unchanged();
    failures += test_q4_cluster_position_advances();
    failures += test_q4_cluster_fabric_packets_nonzero();
    failures += test_q4_cluster_metadata_plan_rejected();

    return (failures > 0) ? 1 : 0;
}
