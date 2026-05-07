#include "att1_backend.h"
#include "att1_cluster_infer.h"
#include "att1_model.h"
#include "att1_trace.h"

#include <stdio.h>
#include <string.h>

#define MODEL_PATH "models/dummy/model.att1"

static int create_cluster_pair(const att1_model *model,
                               att1_cluster_infer_t **out_cpu,
                               att1_cluster_infer_t **out_cuda)
{
    const att1_cluster_infer_config config = {2u, 4u, 0u, ATT1_SHARD_PLAN_RUNTIME};
    att1_cluster_infer_t *cpu = NULL;
    att1_cluster_infer_t *cuda = NULL;
    att1_backend *cpu_backend = NULL;
    att1_backend *cuda_backend = NULL;

    if ((out_cpu == NULL) || (out_cuda == NULL)) {
        return -1;
    }
    *out_cpu = NULL;
    *out_cuda = NULL;

    if ((att1_cluster_infer_create(model, &config, &cpu) != ATT1_OK) ||
        (att1_cluster_infer_create(model, &config, &cuda) != ATT1_OK)) {
        att1_cluster_infer_destroy(cpu);
        att1_cluster_infer_destroy(cuda);
        return -1;
    }

    if ((att1_backend_cpu_f32_create(&cpu_backend) != ATT1_OK) ||
        (att1_cluster_infer_set_backend(cpu, cpu_backend) != ATT1_OK)) {
        att1_backend_destroy(cpu_backend);
        att1_cluster_infer_destroy(cpu);
        att1_cluster_infer_destroy(cuda);
        return -1;
    }
    cpu_backend = NULL;

    if (att1_backend_cuda_create(&cuda_backend) != ATT1_OK) {
        att1_cluster_infer_destroy(cpu);
        att1_cluster_infer_destroy(cuda);
        return -1;
    }

    if ((cuda_backend->ops == NULL) ||
        (cuda_backend->ops->name == NULL) ||
        (strcmp(cuda_backend->ops->name, "cuda") != 0)) {
        att1_backend_destroy(cuda_backend);
        att1_cluster_infer_destroy(cpu);
        att1_cluster_infer_destroy(cuda);
        return -1;
    }

    if (att1_cluster_infer_set_backend(cuda, cuda_backend) != ATT1_OK) {
        att1_backend_destroy(cuda_backend);
        att1_cluster_infer_destroy(cpu);
        att1_cluster_infer_destroy(cuda);
        return -1;
    }
    cuda_backend = NULL;

    *out_cpu = cpu;
    *out_cuda = cuda;
    return 0;
}

static int test_next_token_equivalence(const att1_model *model)
{
    att1_cluster_infer_t *cpu = NULL;
    att1_cluster_infer_t *cuda = NULL;
    uint32_t cpu_next = 0u;
    uint32_t cuda_next = 0u;

    if (create_cluster_pair(model, &cpu, &cuda) != 0) {
        return -1;
    }

    if ((att1_cluster_infer_decode_token(cpu, (uint32_t)'h', &cpu_next) !=
         ATT1_OK) ||
        (att1_cluster_infer_decode_token(cuda, (uint32_t)'h', &cuda_next) !=
         ATT1_OK) ||
        (cpu_next != cuda_next)) {
        att1_cluster_infer_destroy(cpu);
        att1_cluster_infer_destroy(cuda);
        return -1;
    }

    att1_cluster_infer_destroy(cpu);
    att1_cluster_infer_destroy(cuda);
    return 0;
}

static int test_sequence_equivalence_2_to_4_tokens(const att1_model *model)
{
    att1_cluster_infer_t *cpu = NULL;
    att1_cluster_infer_t *cuda = NULL;
    const unsigned char prompt[] = "hi";
    uint32_t cpu_tokens[4] = {0u};
    uint32_t cuda_tokens[4] = {0u};
    size_t cpu_count = 0u;
    size_t cuda_count = 0u;

    if (create_cluster_pair(model, &cpu, &cuda) != 0) {
        return -1;
    }

    if ((att1_cluster_infer_generate(cpu,
                                     prompt,
                                     sizeof(prompt) - 1u,
                                     2u,
                                     cpu_tokens,
                                     4u,
                                     &cpu_count) != ATT1_OK) ||
        (att1_cluster_infer_generate(cuda,
                                     prompt,
                                     sizeof(prompt) - 1u,
                                     2u,
                                     cuda_tokens,
                                     4u,
                                     &cuda_count) != ATT1_OK) ||
        (cpu_count != 2u) ||
        (cuda_count != 2u) ||
        (memcmp(cpu_tokens, cuda_tokens, 2u * sizeof(cpu_tokens[0])) != 0)) {
        att1_cluster_infer_destroy(cpu);
        att1_cluster_infer_destroy(cuda);
        return -1;
    }

    att1_cluster_infer_destroy(cpu);
    att1_cluster_infer_destroy(cuda);

    if (create_cluster_pair(model, &cpu, &cuda) != 0) {
        return -1;
    }

    if ((att1_cluster_infer_generate(cpu,
                                     prompt,
                                     sizeof(prompt) - 1u,
                                     4u,
                                     cpu_tokens,
                                     4u,
                                     &cpu_count) != ATT1_OK) ||
        (att1_cluster_infer_generate(cuda,
                                     prompt,
                                     sizeof(prompt) - 1u,
                                     4u,
                                     cuda_tokens,
                                     4u,
                                     &cuda_count) != ATT1_OK) ||
        (cpu_count != 4u) ||
        (cuda_count != 4u) ||
        (memcmp(cpu_tokens, cuda_tokens, sizeof(cpu_tokens)) != 0)) {
        att1_cluster_infer_destroy(cpu);
        att1_cluster_infer_destroy(cuda);
        return -1;
    }

    att1_cluster_infer_destroy(cpu);
    att1_cluster_infer_destroy(cuda);
    return 0;
}

static int test_trace_and_counter_equivalence(const att1_model *model)
{
    att1_cluster_infer_t *cpu = NULL;
    att1_cluster_infer_t *cuda = NULL;
    att1_trace_t *cpu_trace = NULL;
    att1_trace_t *cuda_trace = NULL;
    att1_trace_counters cpu_counters;
    att1_trace_counters cuda_counters;
    uint32_t cpu_next = 0u;
    uint32_t cuda_next = 0u;
    const unsigned char prompt[] = "he";
    size_t i = 0u;

    if ((create_cluster_pair(model, &cpu, &cuda) != 0) ||
        (att1_trace_create(model->config.n_layers, 2u, &cpu_trace) !=
         ATT1_OK) ||
        (att1_trace_create(model->config.n_layers, 2u, &cuda_trace) !=
         ATT1_OK) ||
        (att1_cluster_infer_set_trace(cpu, cpu_trace) != ATT1_OK) ||
        (att1_cluster_infer_set_trace(cuda, cuda_trace) != ATT1_OK)) {
        att1_cluster_infer_destroy(cpu);
        att1_cluster_infer_destroy(cuda);
        att1_trace_destroy(cpu_trace);
        att1_trace_destroy(cuda_trace);
        return -1;
    }

    for (i = 0u; i < (sizeof(prompt) - 1u); i++) {
        if ((att1_cluster_infer_decode_token(cpu,
                                             (uint32_t)prompt[i],
                                             &cpu_next) != ATT1_OK) ||
            (att1_cluster_infer_decode_token(cuda,
                                             (uint32_t)prompt[i],
                                             &cuda_next) != ATT1_OK) ||
            (cpu_next != cuda_next)) {
            att1_cluster_infer_destroy(cpu);
            att1_cluster_infer_destroy(cuda);
            att1_trace_destroy(cpu_trace);
            att1_trace_destroy(cuda_trace);
            return -1;
        }
    }

    if ((att1_trace_snapshot(cpu_trace, &cpu_counters) != ATT1_OK) ||
        (att1_trace_snapshot(cuda_trace, &cuda_counters) != ATT1_OK)) {
        att1_cluster_infer_destroy(cpu);
        att1_cluster_infer_destroy(cuda);
        att1_trace_destroy(cpu_trace);
        att1_trace_destroy(cuda_trace);
        return -1;
    }

    if ((cpu_counters.fabric_packets_sent == 0u) ||
        (cpu_counters.fabric_packets_received == 0u) ||
        (cuda_counters.fabric_packets_sent == 0u) ||
        (cuda_counters.fabric_packets_received == 0u) ||
        (cpu_counters.fabric_packets_sent != cuda_counters.fabric_packets_sent) ||
        (cpu_counters.fabric_packets_received !=
         cuda_counters.fabric_packets_received) ||
        (cpu_counters.fabric_payload_bytes_sent !=
         cuda_counters.fabric_payload_bytes_sent) ||
        (cpu_counters.fabric_payload_bytes_received !=
         cuda_counters.fabric_payload_bytes_received) ||
        (cpu_counters.activation_bytes_sent !=
         cuda_counters.activation_bytes_sent) ||
        (cpu_counters.logits_bytes_produced !=
         cuda_counters.logits_bytes_produced) ||
        (cpu_counters.tile_layer_executions !=
         cuda_counters.tile_layer_executions)) {
        att1_cluster_infer_destroy(cpu);
        att1_cluster_infer_destroy(cuda);
        att1_trace_destroy(cpu_trace);
        att1_trace_destroy(cuda_trace);
        return -1;
    }

    for (i = 0u; i < 2u; i++) {
        att1_trace_tile cpu_tile;
        att1_trace_tile cuda_tile;

        if ((att1_trace_tile_snapshot(cpu_trace, i, &cpu_tile) != ATT1_OK) ||
            (att1_trace_tile_snapshot(cuda_trace, i, &cuda_tile) != ATT1_OK) ||
            (cpu_tile.layer_executions != cuda_tile.layer_executions) ||
            (cpu_tile.activation_bytes_sent !=
             cuda_tile.activation_bytes_sent) ||
            (cpu_tile.logits_bytes_produced != cuda_tile.logits_bytes_produced)) {
            att1_cluster_infer_destroy(cpu);
            att1_cluster_infer_destroy(cuda);
            att1_trace_destroy(cpu_trace);
            att1_trace_destroy(cuda_trace);
            return -1;
        }
    }

    att1_cluster_infer_destroy(cpu);
    att1_cluster_infer_destroy(cuda);
    att1_trace_destroy(cpu_trace);
    att1_trace_destroy(cuda_trace);
    return 0;
}

static int test_no_silent_cpu_fallback(const att1_model *model)
{
    const att1_cluster_infer_config config = {2u, 4u, 0u, ATT1_SHARD_PLAN_RUNTIME};
    att1_cluster_infer_t *cuda = NULL;
    att1_backend *cuda_backend = NULL;
    uint32_t next = 0u;

    if (att1_cluster_infer_create(model, &config, &cuda) != ATT1_OK) {
        return -1;
    }

    if (att1_backend_cuda_create(&cuda_backend) != ATT1_OK) {
        att1_cluster_infer_destroy(cuda);
        return -1;
    }

    if ((cuda_backend->ops == NULL) ||
        (cuda_backend->ops->name == NULL) ||
        (strcmp(cuda_backend->ops->name, "cuda") != 0) ||
        (att1_cluster_infer_set_backend(cuda, cuda_backend) != ATT1_OK)) {
        att1_backend_destroy(cuda_backend);
        att1_cluster_infer_destroy(cuda);
        return -1;
    }
    cuda_backend = NULL;

    if (att1_cluster_infer_decode_token(cuda, (uint32_t)'z', &next) !=
        ATT1_OK) {
        att1_cluster_infer_destroy(cuda);
        return -1;
    }

    att1_cluster_infer_destroy(cuda);
    return 0;
}

int main(void)
{
    att1_model model;

    if (att1_backend_cuda_available() == 0) {
        att1_backend *cuda_backend = NULL;
        if (att1_backend_cuda_create(&cuda_backend) != ATT1_ERR_UNSUPPORTED) {
            att1_backend_destroy(cuda_backend);
            fputs("cuda cluster unavailable-path check failed\n", stderr);
            return 1;
        }

        puts("cuda_cluster test skipped (CUDA unavailable)");
        return 0;
    }

    if (att1_model_load(MODEL_PATH, &model) != ATT1_OK) {
        fputs("cuda_cluster model load failed\n", stderr);
        return 1;
    }

    if ((test_next_token_equivalence(&model) != 0) ||
        (test_sequence_equivalence_2_to_4_tokens(&model) != 0) ||
        (test_trace_and_counter_equivalence(&model) != 0) ||
        (test_no_silent_cpu_fallback(&model) != 0)) {
        att1_model_free(&model);
        fputs("cuda_cluster test failed\n", stderr);
        return 1;
    }

    att1_model_free(&model);
    puts("cuda_cluster test passed");
    return 0;
}
