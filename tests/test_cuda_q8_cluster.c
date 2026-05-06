/* CUDA q8 cluster inference API tests (M28) */
#include "att1_backend.h"
#include "att1_cluster_infer.h"
#include "att1_model.h"
#include "att1_trace.h"

#include <stdio.h>
#include <string.h>

#define MODEL_PATH "models/dummy/model.att1"

/*
 * Create one cpu-q8 cluster and one cuda-q8 cluster for the same model.
 * Both use 2 tiles.  Returns 0 on success, -1 on any error.
 * Caller owns both infer objects on success.
 */
static int create_cluster_pair(const att1_model *model,
                               att1_cluster_infer_t **out_cpuq8,
                               att1_cluster_infer_t **out_cudaq8)
{
    const att1_cluster_infer_config config = {2u, 4u, 0u};
    att1_cluster_infer_t *cpuq8 = NULL;
    att1_cluster_infer_t *cudaq8 = NULL;
    att1_backend *cpuq8_backend = NULL;
    att1_backend *cudaq8_backend = NULL;

    if ((out_cpuq8 == NULL) || (out_cudaq8 == NULL)) {
        return -1;
    }
    *out_cpuq8 = NULL;
    *out_cudaq8 = NULL;

    if ((att1_cluster_infer_create(model, &config, &cpuq8) != ATT1_OK) ||
        (att1_cluster_infer_create(model, &config, &cudaq8) != ATT1_OK)) {
        att1_cluster_infer_destroy(cpuq8);
        att1_cluster_infer_destroy(cudaq8);
        return -1;
    }

    if (att1_backend_cpu_q8_create(&cpuq8_backend) != ATT1_OK) {
        att1_cluster_infer_destroy(cpuq8);
        att1_cluster_infer_destroy(cudaq8);
        return -1;
    }
    if (att1_cluster_infer_set_backend(cpuq8, cpuq8_backend) != ATT1_OK) {
        att1_backend_destroy(cpuq8_backend);
        att1_cluster_infer_destroy(cpuq8);
        att1_cluster_infer_destroy(cudaq8);
        return -1;
    }
    cpuq8_backend = NULL;

    if (att1_backend_cuda_q8_create(&cudaq8_backend) != ATT1_OK) {
        att1_cluster_infer_destroy(cpuq8);
        att1_cluster_infer_destroy(cudaq8);
        return -1;
    }
    if ((cudaq8_backend->ops == NULL) ||
        (cudaq8_backend->ops->name == NULL) ||
        (strcmp(cudaq8_backend->ops->name, "cuda-q8") != 0)) {
        att1_backend_destroy(cudaq8_backend);
        att1_cluster_infer_destroy(cpuq8);
        att1_cluster_infer_destroy(cudaq8);
        return -1;
    }
    if (att1_cluster_infer_set_backend(cudaq8, cudaq8_backend) != ATT1_OK) {
        att1_backend_destroy(cudaq8_backend);
        att1_cluster_infer_destroy(cpuq8);
        att1_cluster_infer_destroy(cudaq8);
        return -1;
    }
    cudaq8_backend = NULL;

    *out_cpuq8 = cpuq8;
    *out_cudaq8 = cudaq8;
    return 0;
}

/*
 * Test: cpu-q8 and cuda-q8 cluster decode the same first token for the
 * same input.
 */
static int test_next_token_equivalence(const att1_model *model)
{
    att1_cluster_infer_t *cpuq8 = NULL;
    att1_cluster_infer_t *cudaq8 = NULL;
    uint32_t cpuq8_next = 0u;
    uint32_t cudaq8_next = 0u;

    if (create_cluster_pair(model, &cpuq8, &cudaq8) != 0) {
        fputs("FAIL: create_cluster_pair for token equivalence\n", stderr);
        return -1;
    }

    if ((att1_cluster_infer_decode_token(cpuq8, (uint32_t)'h', &cpuq8_next) !=
         ATT1_OK) ||
        (att1_cluster_infer_decode_token(cudaq8, (uint32_t)'h', &cudaq8_next) !=
         ATT1_OK)) {
        fputs("FAIL: decode_token error in token equivalence\n", stderr);
        att1_cluster_infer_destroy(cpuq8);
        att1_cluster_infer_destroy(cudaq8);
        return -1;
    }

    if (cpuq8_next != cudaq8_next) {
        fprintf(stderr,
                "FAIL: token mismatch: cpu-q8=%u cuda-q8=%u\n",
                cpuq8_next,
                cudaq8_next);
        att1_cluster_infer_destroy(cpuq8);
        att1_cluster_infer_destroy(cudaq8);
        return -1;
    }

    att1_cluster_infer_destroy(cpuq8);
    att1_cluster_infer_destroy(cudaq8);
    fputs("PASS: cuda-q8 cluster next-token equivalence\n", stderr);
    return 0;
}

/*
 * Test: 2-token and 4-token sequences match between cpu-q8 cluster and
 * cuda-q8 cluster.
 */
static int test_sequence_equivalence(const att1_model *model)
{
    att1_cluster_infer_t *cpuq8 = NULL;
    att1_cluster_infer_t *cudaq8 = NULL;
    const unsigned char prompt[] = "hi";
    uint32_t cpuq8_tokens[4] = {0u};
    uint32_t cudaq8_tokens[4] = {0u};
    size_t cpuq8_count = 0u;
    size_t cudaq8_count = 0u;

    /* 2-token sequence */
    if (create_cluster_pair(model, &cpuq8, &cudaq8) != 0) {
        fputs("FAIL: create_cluster_pair for 2-token sequence\n", stderr);
        return -1;
    }

    if ((att1_cluster_infer_generate(cpuq8,
                                     prompt,
                                     sizeof(prompt) - 1u,
                                     2u,
                                     cpuq8_tokens,
                                     4u,
                                     &cpuq8_count) != ATT1_OK) ||
        (att1_cluster_infer_generate(cudaq8,
                                     prompt,
                                     sizeof(prompt) - 1u,
                                     2u,
                                     cudaq8_tokens,
                                     4u,
                                     &cudaq8_count) != ATT1_OK) ||
        (cpuq8_count != 2u) ||
        (cudaq8_count != 2u) ||
        (memcmp(cpuq8_tokens, cudaq8_tokens,
                2u * sizeof(cpuq8_tokens[0])) != 0)) {
        fputs("FAIL: 2-token sequence mismatch between cpu-q8 and cuda-q8 cluster\n",
              stderr);
        att1_cluster_infer_destroy(cpuq8);
        att1_cluster_infer_destroy(cudaq8);
        return -1;
    }

    att1_cluster_infer_destroy(cpuq8);
    att1_cluster_infer_destroy(cudaq8);

    /* 4-token sequence */
    if (create_cluster_pair(model, &cpuq8, &cudaq8) != 0) {
        fputs("FAIL: create_cluster_pair for 4-token sequence\n", stderr);
        return -1;
    }

    if ((att1_cluster_infer_generate(cpuq8,
                                     prompt,
                                     sizeof(prompt) - 1u,
                                     4u,
                                     cpuq8_tokens,
                                     4u,
                                     &cpuq8_count) != ATT1_OK) ||
        (att1_cluster_infer_generate(cudaq8,
                                     prompt,
                                     sizeof(prompt) - 1u,
                                     4u,
                                     cudaq8_tokens,
                                     4u,
                                     &cudaq8_count) != ATT1_OK) ||
        (cpuq8_count != 4u) ||
        (cudaq8_count != 4u) ||
        (memcmp(cpuq8_tokens, cudaq8_tokens, sizeof(cpuq8_tokens)) != 0)) {
        fputs("FAIL: 4-token sequence mismatch between cpu-q8 and cuda-q8 cluster\n",
              stderr);
        att1_cluster_infer_destroy(cpuq8);
        att1_cluster_infer_destroy(cudaq8);
        return -1;
    }

    att1_cluster_infer_destroy(cpuq8);
    att1_cluster_infer_destroy(cudaq8);
    fputs("PASS: cuda-q8 cluster sequence equivalence\n", stderr);
    return 0;
}

/*
 * Test: fabric packet counts and byte counters are nonzero and match
 * between cpu-q8 cluster and cuda-q8 cluster.
 */
static int test_trace_and_counter_equivalence(const att1_model *model)
{
    att1_cluster_infer_t *cpuq8 = NULL;
    att1_cluster_infer_t *cudaq8 = NULL;
    att1_trace_t *cpuq8_trace = NULL;
    att1_trace_t *cudaq8_trace = NULL;
    att1_trace_counters cpuq8_counters;
    att1_trace_counters cudaq8_counters;
    uint32_t cpuq8_next = 0u;
    uint32_t cudaq8_next = 0u;
    const unsigned char prompt[] = "he";
    size_t i = 0u;

    if ((create_cluster_pair(model, &cpuq8, &cudaq8) != 0) ||
        (att1_trace_create(model->config.n_layers, 2u, &cpuq8_trace) !=
         ATT1_OK) ||
        (att1_trace_create(model->config.n_layers, 2u, &cudaq8_trace) !=
         ATT1_OK) ||
        (att1_cluster_infer_set_trace(cpuq8, cpuq8_trace) != ATT1_OK) ||
        (att1_cluster_infer_set_trace(cudaq8, cudaq8_trace) != ATT1_OK)) {
        fputs("FAIL: setup in trace/counter equivalence\n", stderr);
        att1_cluster_infer_destroy(cpuq8);
        att1_cluster_infer_destroy(cudaq8);
        att1_trace_destroy(cpuq8_trace);
        att1_trace_destroy(cudaq8_trace);
        return -1;
    }

    for (i = 0u; i < (sizeof(prompt) - 1u); i++) {
        if ((att1_cluster_infer_decode_token(cpuq8,
                                             (uint32_t)prompt[i],
                                             &cpuq8_next) != ATT1_OK) ||
            (att1_cluster_infer_decode_token(cudaq8,
                                             (uint32_t)prompt[i],
                                             &cudaq8_next) != ATT1_OK) ||
            (cpuq8_next != cudaq8_next)) {
            fputs("FAIL: decode mismatch in trace/counter equivalence\n", stderr);
            att1_cluster_infer_destroy(cpuq8);
            att1_cluster_infer_destroy(cudaq8);
            att1_trace_destroy(cpuq8_trace);
            att1_trace_destroy(cudaq8_trace);
            return -1;
        }
    }

    if ((att1_trace_snapshot(cpuq8_trace, &cpuq8_counters) != ATT1_OK) ||
        (att1_trace_snapshot(cudaq8_trace, &cudaq8_counters) != ATT1_OK)) {
        fputs("FAIL: trace snapshot in trace/counter equivalence\n", stderr);
        att1_cluster_infer_destroy(cpuq8);
        att1_cluster_infer_destroy(cudaq8);
        att1_trace_destroy(cpuq8_trace);
        att1_trace_destroy(cudaq8_trace);
        return -1;
    }

    if ((cudaq8_counters.fabric_packets_sent == 0u) ||
        (cudaq8_counters.fabric_packets_received == 0u) ||
        (cudaq8_counters.activation_bytes_sent == 0u) ||
        (cudaq8_counters.logits_bytes_produced == 0u) ||
        (cudaq8_counters.kv_appends == 0u)) {
        fputs("FAIL: cuda-q8 cluster counters are zero\n", stderr);
        att1_cluster_infer_destroy(cpuq8);
        att1_cluster_infer_destroy(cudaq8);
        att1_trace_destroy(cpuq8_trace);
        att1_trace_destroy(cudaq8_trace);
        return -1;
    }

    if ((cudaq8_counters.fabric_packets_sent !=
         cpuq8_counters.fabric_packets_sent) ||
        (cudaq8_counters.fabric_packets_received !=
         cpuq8_counters.fabric_packets_received) ||
        (cudaq8_counters.activation_bytes_sent !=
         cpuq8_counters.activation_bytes_sent) ||
        (cudaq8_counters.logits_bytes_produced !=
         cpuq8_counters.logits_bytes_produced)) {
        fputs("FAIL: cuda-q8 cluster counters differ from cpu-q8 cluster\n",
              stderr);
        att1_cluster_infer_destroy(cpuq8);
        att1_cluster_infer_destroy(cudaq8);
        att1_trace_destroy(cpuq8_trace);
        att1_trace_destroy(cudaq8_trace);
        return -1;
    }

    att1_cluster_infer_destroy(cpuq8);
    att1_cluster_infer_destroy(cudaq8);
    att1_trace_destroy(cpuq8_trace);
    att1_trace_destroy(cudaq8_trace);
    fputs("PASS: cuda-q8 cluster trace and counter equivalence\n", stderr);
    return 0;
}

/*
 * Test: cuda-q8 backend must report its own name and must not silently
 * use a different backend (no silent fallback to cpu-q8, cpu-f32, or cuda).
 */
static int test_no_silent_fallback(const att1_model *model)
{
    const att1_cluster_infer_config config = {2u, 4u, 0u};
    att1_cluster_infer_t *infer = NULL;
    att1_backend *backend = NULL;
    uint32_t next = 0u;

    if ((att1_cluster_infer_create(model, &config, &infer) != ATT1_OK) ||
        (att1_backend_cuda_q8_create(&backend) != ATT1_OK)) {
        att1_cluster_infer_destroy(infer);
        att1_backend_destroy(backend);
        fputs("FAIL: create in no-fallback test\n", stderr);
        return -1;
    }

    /* Verify the backend name before setting it */
    if ((backend->ops == NULL) ||
        (backend->ops->name == NULL) ||
        (strcmp(backend->ops->name, "cuda-q8") != 0)) {
        fputs("FAIL: cuda-q8 backend name mismatch\n", stderr);
        att1_cluster_infer_destroy(infer);
        att1_backend_destroy(backend);
        return -1;
    }

    if (att1_cluster_infer_set_backend(infer, backend) != ATT1_OK) {
        fputs("FAIL: set_backend in no-fallback test\n", stderr);
        att1_cluster_infer_destroy(infer);
        att1_backend_destroy(backend);
        return -1;
    }
    backend = NULL;

    if (att1_cluster_infer_decode_token(infer, (uint32_t)'h', &next) !=
        ATT1_OK) {
        fputs("FAIL: decode_token in no-fallback test\n", stderr);
        att1_cluster_infer_destroy(infer);
        return -1;
    }

    att1_cluster_infer_destroy(infer);
    fputs("PASS: cuda-q8 cluster no-silent-fallback\n", stderr);
    return 0;
}

int main(void)
{
    att1_model model;
    int failures = 0;

    if (!att1_backend_cuda_available()) {
        fputs("SKIP: CUDA not available — all cuda_q8_cluster tests skipped\n",
              stderr);
        return 0;
    }

    if (att1_model_load(MODEL_PATH, &model) != ATT1_OK) {
        fputs("cuda_q8_cluster model load failed\n", stderr);
        return 1;
    }

    if (test_next_token_equivalence(&model) != 0) {
        failures++;
    }
    if (test_sequence_equivalence(&model) != 0) {
        failures++;
    }
    if (test_trace_and_counter_equivalence(&model) != 0) {
        failures++;
    }
    if (test_no_silent_fallback(&model) != 0) {
        failures++;
    }

    att1_model_free(&model);

    if (failures != 0) {
        fputs("cuda_q8_cluster test failed\n", stderr);
        return 1;
    }

    puts("cuda_q8_cluster test passed");
    return 0;
}
