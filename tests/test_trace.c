#include "att1_cluster_infer.h"
#include "att1_infer.h"
#include "att1_model.h"
#include "att1_trace.h"

#include <stdio.h>
#include <string.h>

#define MODEL_PATH "models/dummy/model.att1"

static int check_trace_disabled(const att1_model *model)
{
    att1_infer_t *infer = NULL;
    uint32_t token = 0u;

    if (att1_infer_create(model, &infer) != ATT1_OK) {
        return -1;
    }

    if ((att1_infer_decode_token(infer, (uint32_t)'A', &token) != ATT1_OK) ||
        (token >= model->config.vocab_size)) {
        att1_infer_destroy(infer);
        return -1;
    }

    att1_infer_destroy(infer);
    return 0;
}

static int check_single_trace(const att1_model *model)
{
    att1_infer_t *infer = NULL;
    att1_trace_t *trace = NULL;
    att1_trace_counters counters;
    uint32_t token = 0u;

    if ((att1_trace_create(model->config.n_layers, 1u, &trace) != ATT1_OK) ||
        (att1_infer_create(model, &infer) != ATT1_OK) ||
        (att1_infer_set_trace(infer, trace) != ATT1_OK)) {
        att1_infer_destroy(infer);
        att1_trace_destroy(trace);
        return -1;
    }

    if ((att1_infer_decode_token(infer, (uint32_t)'A', &token) != ATT1_OK) ||
        (att1_infer_decode_token(infer, token, &token) != ATT1_OK)) {
        att1_infer_destroy(infer);
        att1_trace_destroy(trace);
        return -1;
    }

    if (att1_trace_snapshot(trace, &counters) != ATT1_OK) {
        att1_infer_destroy(infer);
        att1_trace_destroy(trace);
        return -1;
    }

    if ((counters.tokens_decoded != 2u) ||
        (counters.kv_appends != (uint64_t)model->config.n_layers * 2u) ||
        (counters.kv_key_reads == 0u) ||
        (counters.kv_value_reads == 0u) ||
        (counters.logits_bytes_produced !=
         (uint64_t)model->config.vocab_size * sizeof(float) * 2u)) {
        att1_infer_destroy(infer);
        att1_trace_destroy(trace);
        fputs("single trace counters failed\n", stderr);
        return -1;
    }

    att1_infer_destroy(infer);
    att1_trace_destroy(trace);
    return 0;
}

static int check_cluster_trace(const att1_model *model)
{
    const att1_cluster_infer_config config = {2u, 4u, 0u, ATT1_SHARD_PLAN_RUNTIME};
    att1_cluster_infer_t *infer = NULL;
    att1_trace_t *trace = NULL;
    att1_trace_counters counters;
    uint32_t token = 0u;

    if ((att1_trace_create(model->config.n_layers, config.tile_count, &trace) !=
         ATT1_OK) ||
        (att1_cluster_infer_create(model, &config, &infer) != ATT1_OK) ||
        (att1_cluster_infer_set_trace(infer, trace) != ATT1_OK)) {
        att1_cluster_infer_destroy(infer);
        att1_trace_destroy(trace);
        return -1;
    }

    if ((att1_cluster_infer_decode_token(infer, (uint32_t)'A', &token) !=
         ATT1_OK) ||
        (att1_cluster_infer_decode_token(infer, token, &token) != ATT1_OK)) {
        att1_cluster_infer_destroy(infer);
        att1_trace_destroy(trace);
        return -1;
    }

    if (att1_trace_snapshot(trace, &counters) != ATT1_OK) {
        att1_cluster_infer_destroy(infer);
        att1_trace_destroy(trace);
        return -1;
    }

    if ((counters.tokens_decoded != 2u) ||
        (counters.fabric_packets_sent == 0u) ||
        (counters.fabric_packets_received == 0u) ||
        (counters.fabric_payload_bytes_sent == 0u) ||
        (counters.fabric_payload_bytes_received == 0u) ||
        (counters.activation_bytes_sent == 0u) ||
        (counters.kv_appends != (uint64_t)model->config.n_layers * 2u)) {
        att1_cluster_infer_destroy(infer);
        att1_trace_destroy(trace);
        fputs("cluster trace counters failed\n", stderr);
        return -1;
    }

    att1_cluster_infer_destroy(infer);
    att1_trace_destroy(trace);
    return 0;
}

static int check_behavior_unchanged(const att1_model *model)
{
    att1_infer_t *plain = NULL;
    att1_infer_t *traced = NULL;
    att1_trace_t *trace = NULL;
    const float *plain_logits = NULL;
    const float *traced_logits = NULL;
    size_t plain_count = 0u;
    size_t traced_count = 0u;
    uint32_t plain_token = 0u;
    uint32_t traced_token = 0u;
    int ok = 0;

    if ((att1_infer_create(model, &plain) != ATT1_OK) ||
        (att1_infer_create(model, &traced) != ATT1_OK) ||
        (att1_trace_create(model->config.n_layers, 1u, &trace) != ATT1_OK) ||
        (att1_infer_set_trace(traced, trace) != ATT1_OK)) {
        goto done;
    }

    if ((att1_infer_decode_token(plain, (uint32_t)'A', &plain_token) !=
         ATT1_OK) ||
        (att1_infer_decode_token(traced, (uint32_t)'A', &traced_token) !=
         ATT1_OK)) {
        goto done;
    }

    plain_logits = att1_infer_logits(plain, &plain_count);
    traced_logits = att1_infer_logits(traced, &traced_count);
    ok = (plain_token == traced_token) &&
         (plain_count == traced_count) &&
         (plain_logits != NULL) &&
         (traced_logits != NULL) &&
         (memcmp(plain_logits,
                 traced_logits,
                 plain_count * sizeof(float)) == 0);

done:
    att1_infer_destroy(plain);
    att1_infer_destroy(traced);
    att1_trace_destroy(trace);
    return ok ? 0 : -1;
}

int main(void)
{
    att1_model model;

    if (att1_model_load(MODEL_PATH, &model) != ATT1_OK) {
        fputs("failed to load dummy model\n", stderr);
        return 1;
    }

    if ((check_trace_disabled(&model) != 0) ||
        (check_single_trace(&model) != 0) ||
        (check_cluster_trace(&model) != 0) ||
        (check_behavior_unchanged(&model) != 0)) {
        att1_model_free(&model);
        return 1;
    }

    att1_model_free(&model);
    puts("trace test passed");
    return 0;
}
