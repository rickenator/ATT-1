#include "att1_backend.h"
#include "att1_cluster_infer.h"
#include "att1_model.h"
#include "att1_trace.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define MODEL_PATH "models/dummy/model.att1"
#define Q8_LOGIT_TOLERANCE 0.15f

static int create_cluster_with_backend(const att1_model *model,
                                       const char *backend_name,
                                       att1_cluster_infer_t **out_infer,
                                       att1_trace_t **out_trace)
{
    const att1_cluster_infer_config config = {2u, 4u, 0u};
    att1_cluster_infer_t *infer = NULL;
    att1_trace_t *trace = NULL;
    att1_backend *backend = NULL;

    if ((model == NULL) || (backend_name == NULL) ||
        (out_infer == NULL) || (out_trace == NULL)) {
        return -1;
    }
    *out_infer = NULL;
    *out_trace = NULL;

    if ((att1_cluster_infer_create(model, &config, &infer) != ATT1_OK) ||
        (att1_trace_create(model->config.n_layers, config.tile_count, &trace) != ATT1_OK)) {
        att1_cluster_infer_destroy(infer);
        att1_trace_destroy(trace);
        return -1;
    }

    if (strcmp(backend_name, "cpu-f32") == 0) {
        if (att1_backend_cpu_f32_create(&backend) != ATT1_OK) {
            att1_cluster_infer_destroy(infer);
            att1_trace_destroy(trace);
            return -1;
        }
    } else if (strcmp(backend_name, "cpu-q8") == 0) {
        if (att1_backend_cpu_q8_create(&backend) != ATT1_OK) {
            att1_cluster_infer_destroy(infer);
            att1_trace_destroy(trace);
            return -1;
        }
        if ((backend->ops == NULL) || (backend->ops->name == NULL) ||
            (strcmp(backend->ops->name, "cpu-q8") != 0)) {
            att1_backend_destroy(backend);
            att1_cluster_infer_destroy(infer);
            att1_trace_destroy(trace);
            return -1;
        }
    } else {
        att1_cluster_infer_destroy(infer);
        att1_trace_destroy(trace);
        return -1;
    }

    if ((att1_cluster_infer_set_backend(infer, backend) != ATT1_OK) ||
        (att1_cluster_infer_set_trace(infer, trace) != ATT1_OK)) {
        att1_backend_destroy(backend);
        att1_cluster_infer_destroy(infer);
        att1_trace_destroy(trace);
        return -1;
    }

    *out_infer = infer;
    *out_trace = trace;
    return 0;
}

static int test_cluster_logits_tolerance_and_counters(const att1_model *model)
{
    const unsigned char prompt[] = "hello";
    att1_cluster_infer_t *f32 = NULL;
    att1_cluster_infer_t *q8 = NULL;
    att1_trace_t *f32_trace = NULL;
    att1_trace_t *q8_trace = NULL;
    const float *f32_logits = NULL;
    const float *q8_logits = NULL;
    size_t f32_count = 0u;
    size_t q8_count = 0u;
    uint32_t f32_next = 0u;
    uint32_t q8_next = 0u;
    att1_trace_counters f32_counters;
    att1_trace_counters q8_counters;
    float max_abs_diff = 0.0f;
    size_t i = 0u;

    if ((create_cluster_with_backend(model, "cpu-f32", &f32, &f32_trace) != 0) ||
        (create_cluster_with_backend(model, "cpu-q8", &q8, &q8_trace) != 0)) {
        att1_cluster_infer_destroy(f32);
        att1_cluster_infer_destroy(q8);
        att1_trace_destroy(f32_trace);
        att1_trace_destroy(q8_trace);
        return -1;
    }

    for (i = 0u; i < (sizeof(prompt) - 1u); i++) {
        if ((att1_cluster_infer_decode_token(f32,
                                             (uint32_t)prompt[i],
                                             &f32_next) != ATT1_OK) ||
            (att1_cluster_infer_decode_token(q8,
                                             (uint32_t)prompt[i],
                                             &q8_next) != ATT1_OK)) {
            att1_cluster_infer_destroy(f32);
            att1_cluster_infer_destroy(q8);
            att1_trace_destroy(f32_trace);
            att1_trace_destroy(q8_trace);
            return -1;
        }
    }

    f32_logits = att1_cluster_infer_logits(f32, &f32_count);
    q8_logits = att1_cluster_infer_logits(q8, &q8_count);
    if ((f32_logits == NULL) || (q8_logits == NULL) ||
        (f32_count != q8_count) || (f32_count != model->config.vocab_size)) {
        att1_cluster_infer_destroy(f32);
        att1_cluster_infer_destroy(q8);
        att1_trace_destroy(f32_trace);
        att1_trace_destroy(q8_trace);
        return -1;
    }

    for (i = 0u; i < f32_count; i++) {
        const float abs_diff = fabsf(f32_logits[i] - q8_logits[i]);
        if (abs_diff > max_abs_diff) {
            max_abs_diff = abs_diff;
        }
    }

    if (max_abs_diff > Q8_LOGIT_TOLERANCE) {
        fprintf(stderr,
                "cluster q8 logits exceed tolerance: max_abs_diff=%f tol=%f\n",
                max_abs_diff,
                Q8_LOGIT_TOLERANCE);
        att1_cluster_infer_destroy(f32);
        att1_cluster_infer_destroy(q8);
        att1_trace_destroy(f32_trace);
        att1_trace_destroy(q8_trace);
        return -1;
    }

    if ((att1_trace_snapshot(f32_trace, &f32_counters) != ATT1_OK) ||
        (att1_trace_snapshot(q8_trace, &q8_counters) != ATT1_OK)) {
        att1_cluster_infer_destroy(f32);
        att1_cluster_infer_destroy(q8);
        att1_trace_destroy(f32_trace);
        att1_trace_destroy(q8_trace);
        return -1;
    }

    if ((q8_counters.fabric_packets_sent == 0u) ||
        (q8_counters.fabric_packets_received == 0u) ||
        (q8_counters.activation_bytes_sent == 0u) ||
        (q8_counters.logits_bytes_produced == 0u) ||
        (q8_counters.kv_appends == 0u) ||
        (q8_counters.fabric_packets_sent != f32_counters.fabric_packets_sent) ||
        (q8_counters.fabric_packets_received != f32_counters.fabric_packets_received) ||
        (q8_counters.activation_bytes_sent != f32_counters.activation_bytes_sent) ||
        (q8_counters.logits_bytes_produced != f32_counters.logits_bytes_produced)) {
        fputs("cluster q8 trace/fabric counters are invalid\n", stderr);
        att1_cluster_infer_destroy(f32);
        att1_cluster_infer_destroy(q8);
        att1_trace_destroy(f32_trace);
        att1_trace_destroy(q8_trace);
        return -1;
    }

    att1_cluster_infer_destroy(f32);
    att1_cluster_infer_destroy(q8);
    att1_trace_destroy(f32_trace);
    att1_trace_destroy(q8_trace);
    return 0;
}

static int test_cluster_generated_token_determinism(const att1_model *model)
{
    const unsigned char prompt[] = "he";
    att1_cluster_infer_t *f32 = NULL;
    att1_cluster_infer_t *q8 = NULL;
    att1_trace_t *f32_trace = NULL;
    att1_trace_t *q8_trace = NULL;
    uint32_t f32_tokens[4] = {0u};
    uint32_t q8_tokens[4] = {0u};
    size_t f32_count = 0u;
    size_t q8_count = 0u;

    if ((create_cluster_with_backend(model, "cpu-f32", &f32, &f32_trace) != 0) ||
        (create_cluster_with_backend(model, "cpu-q8", &q8, &q8_trace) != 0)) {
        att1_cluster_infer_destroy(f32);
        att1_cluster_infer_destroy(q8);
        att1_trace_destroy(f32_trace);
        att1_trace_destroy(q8_trace);
        return -1;
    }

    if ((att1_cluster_infer_generate(f32,
                                     prompt,
                                     sizeof(prompt) - 1u,
                                     4u,
                                     f32_tokens,
                                     4u,
                                     &f32_count) != ATT1_OK) ||
        (att1_cluster_infer_generate(q8,
                                     prompt,
                                     sizeof(prompt) - 1u,
                                     4u,
                                     q8_tokens,
                                     4u,
                                     &q8_count) != ATT1_OK) ||
        (f32_count != 4u) ||
        (q8_count != 4u) ||
        (memcmp(f32_tokens, q8_tokens, sizeof(f32_tokens)) != 0)) {
        fputs("cluster generated token mismatch for dummy model\n", stderr);
        att1_cluster_infer_destroy(f32);
        att1_cluster_infer_destroy(q8);
        att1_trace_destroy(f32_trace);
        att1_trace_destroy(q8_trace);
        return -1;
    }

    att1_cluster_infer_destroy(f32);
    att1_cluster_infer_destroy(q8);
    att1_trace_destroy(f32_trace);
    att1_trace_destroy(q8_trace);
    return 0;
}

int main(void)
{
    att1_model model;

    if (att1_model_load(MODEL_PATH, &model) != ATT1_OK) {
        fputs("q8_cluster model load failed\n", stderr);
        return 1;
    }

    if ((test_cluster_logits_tolerance_and_counters(&model) != 0) ||
        (test_cluster_generated_token_determinism(&model) != 0)) {
        att1_model_free(&model);
        fputs("q8_cluster test failed\n", stderr);
        return 1;
    }

    att1_model_free(&model);
    puts("q8_cluster test passed");
    return 0;
}
