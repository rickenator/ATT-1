#include "att1_cluster_infer.h"
#include "att1_infer.h"
#include "att1_model.h"
#include "att1_trace.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void usage(const char *argv0)
{
    fprintf(stderr,
            "usage: %s --model PATH --prompt TEXT --tokens N --mode single|cluster\n",
            argv0);
}

static int parse_size(const char *text, size_t *out)
{
    char *end = NULL;
    unsigned long value = 0u;

    if ((text == NULL) || (out == NULL)) {
        return -1;
    }

    value = strtoul(text, &end, 10);
    if ((end == text) || (*end != '\0')) {
        return -1;
    }

    *out = (size_t)value;
    return 0;
}

static void print_counters(const att1_trace_t *trace)
{
    att1_trace_counters counters;
    size_t i = 0u;

    if (att1_trace_snapshot(trace, &counters) != ATT1_OK) {
        return;
    }

    printf("tokens_decoded=%llu\n",
           (unsigned long long)counters.tokens_decoded);
    printf("token_time_us_total=%llu\n",
           (unsigned long long)counters.token_time_us_total);
    printf("token_time_us_max=%llu\n",
           (unsigned long long)counters.token_time_us_max);
    printf("layer_time_us_total=%llu\n",
           (unsigned long long)counters.layer_time_us_total);
    printf("activation_bytes_sent=%llu\n",
           (unsigned long long)counters.activation_bytes_sent);
    printf("logits_bytes_produced=%llu\n",
           (unsigned long long)counters.logits_bytes_produced);
    printf("fabric_packets_sent=%llu\n",
           (unsigned long long)counters.fabric_packets_sent);
    printf("fabric_packets_received=%llu\n",
           (unsigned long long)counters.fabric_packets_received);
    printf("fabric_payload_bytes_sent=%llu\n",
           (unsigned long long)counters.fabric_payload_bytes_sent);
    printf("fabric_payload_bytes_received=%llu\n",
           (unsigned long long)counters.fabric_payload_bytes_received);
    printf("kv_appends=%llu\n",
           (unsigned long long)counters.kv_appends);
    printf("kv_key_reads=%llu\n",
           (unsigned long long)counters.kv_key_reads);
    printf("kv_value_reads=%llu\n",
           (unsigned long long)counters.kv_value_reads);
    printf("tile_layer_executions=%llu\n",
           (unsigned long long)counters.tile_layer_executions);

    for (i = 0u; i < att1_trace_layer_count(trace); i++) {
        att1_trace_layer layer;
        if (att1_trace_layer_snapshot(trace, i, &layer) == ATT1_OK) {
            printf("layer[%zu].executions=%llu time_us=%llu kv_appends=%llu\n",
                   i,
                   (unsigned long long)layer.executions,
                   (unsigned long long)layer.time_us_total,
                   (unsigned long long)layer.kv_appends);
        }
    }

    for (i = 0u; i < att1_trace_tile_count(trace); i++) {
        att1_trace_tile tile;
        if (att1_trace_tile_snapshot(trace, i, &tile) == ATT1_OK) {
            printf("tile[%zu].layers=%llu activation_bytes_sent=%llu logits_bytes=%llu\n",
                   i,
                   (unsigned long long)tile.layer_executions,
                   (unsigned long long)tile.activation_bytes_sent,
                   (unsigned long long)tile.logits_bytes_produced);
        }
    }
}

static int run_single(const att1_model *model,
                      const unsigned char *prompt,
                      size_t prompt_bytes,
                      size_t max_tokens)
{
    att1_infer_t *infer = NULL;
    att1_trace_t *trace = NULL;
    uint32_t *tokens = NULL;
    size_t out_count = 0u;
    size_t capacity = max_tokens == 0u ? 1u : max_tokens;
    int rc = 1;

    tokens = calloc(capacity, sizeof(*tokens));
    if (tokens == NULL) {
        return 1;
    }

    if ((att1_trace_create(model->config.n_layers, 1u, &trace) != ATT1_OK) ||
        (att1_infer_create(model, &infer) != ATT1_OK) ||
        (att1_infer_set_trace(infer, trace) != ATT1_OK) ||
        (att1_infer_generate(infer,
                             prompt,
                             prompt_bytes,
                             max_tokens,
                             tokens,
                             capacity,
                             &out_count) != ATT1_OK)) {
        goto cleanup;
    }

    printf("mode=single\n");
    printf("generated_tokens=%zu\n", out_count);
    if (out_count > 0u) {
        printf("last_token=%u\n", tokens[out_count - 1u]);
    }
    print_counters(trace);
    rc = 0;

cleanup:
    att1_infer_destroy(infer);
    att1_trace_destroy(trace);
    free(tokens);
    return rc;
}

static int run_cluster(const att1_model *model,
                       const unsigned char *prompt,
                       size_t prompt_bytes,
                       size_t max_tokens)
{
    att1_cluster_infer_config config;
    att1_cluster_infer_t *infer = NULL;
    att1_trace_t *trace = NULL;
    uint32_t *tokens = NULL;
    size_t out_count = 0u;
    size_t capacity = max_tokens == 0u ? 1u : max_tokens;
    size_t tile_count = model->config.n_tiles > 1u ? model->config.n_tiles : 2u;
    int rc = 1;

    tokens = calloc(capacity, sizeof(*tokens));
    if (tokens == NULL) {
        return 1;
    }

    config.tile_count = tile_count;
    config.fabric_queue_capacity = 4u;
    config.fabric_max_payload_bytes = 0u;

    if ((att1_trace_create(model->config.n_layers, tile_count, &trace) != ATT1_OK) ||
        (att1_cluster_infer_create(model, &config, &infer) != ATT1_OK) ||
        (att1_cluster_infer_set_trace(infer, trace) != ATT1_OK) ||
        (att1_cluster_infer_generate(infer,
                                     prompt,
                                     prompt_bytes,
                                     max_tokens,
                                     tokens,
                                     capacity,
                                     &out_count) != ATT1_OK)) {
        goto cleanup;
    }

    printf("mode=cluster\n");
    printf("tiles=%zu\n", tile_count);
    printf("generated_tokens=%zu\n", out_count);
    if (out_count > 0u) {
        printf("last_token=%u\n", tokens[out_count - 1u]);
    }
    print_counters(trace);
    rc = 0;

cleanup:
    att1_cluster_infer_destroy(infer);
    att1_trace_destroy(trace);
    free(tokens);
    return rc;
}

int main(int argc, char **argv)
{
    const char *model_path = NULL;
    const char *prompt = NULL;
    const char *mode = NULL;
    size_t max_tokens = 0u;
    att1_model model;
    int i = 0;
    int rc = 1;

    for (i = 1; i < argc; i++) {
        if ((strcmp(argv[i], "--model") == 0) && ((i + 1) < argc)) {
            model_path = argv[++i];
        } else if ((strcmp(argv[i], "--prompt") == 0) && ((i + 1) < argc)) {
            prompt = argv[++i];
        } else if ((strcmp(argv[i], "--tokens") == 0) && ((i + 1) < argc)) {
            if (parse_size(argv[++i], &max_tokens) != 0) {
                usage(argv[0]);
                return 1;
            }
        } else if ((strcmp(argv[i], "--mode") == 0) && ((i + 1) < argc)) {
            mode = argv[++i];
        } else {
            usage(argv[0]);
            return 1;
        }
    }

    if ((model_path == NULL) || (prompt == NULL) || (mode == NULL) ||
        (prompt[0] == '\0')) {
        usage(argv[0]);
        return 1;
    }

    if (att1_model_load(model_path, &model) != ATT1_OK) {
        fprintf(stderr, "failed to load model: %s\n", model_path);
        return 1;
    }

    if (strcmp(mode, "single") == 0) {
        rc = run_single(&model,
                        (const unsigned char *)prompt,
                        strlen(prompt),
                        max_tokens);
    } else if (strcmp(mode, "cluster") == 0) {
        rc = run_cluster(&model,
                         (const unsigned char *)prompt,
                         strlen(prompt),
                         max_tokens);
    } else {
        usage(argv[0]);
        rc = 1;
    }

    att1_model_free(&model);
    return rc;
}
