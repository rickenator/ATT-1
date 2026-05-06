#include "att1_cluster_infer.h"
#include "att1_infer.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#define MODEL_PATH "models/dummy/model.att1"

static int logits_close(const float *lhs,
                        const float *rhs,
                        size_t count,
                        float tolerance)
{
    size_t i = 0u;

    for (i = 0u; i < count; i++) {
        if (fabsf(lhs[i] - rhs[i]) > tolerance) {
            return 0;
        }
    }

    return 1;
}

static int check_single_cluster_equivalence(const att1_model *model)
{
    const unsigned char prompt[2] = {'A', 'T'};
    const att1_cluster_infer_config config = {3u, 4u, 0u};
    att1_infer single;
    att1_cluster_infer cluster;
    uint32_t single_token = 0u;
    uint32_t cluster_token = 0u;
    att1_cluster_tile_counters counters;
    size_t tile = 0u;
    size_t i = 0u;

    if ((att1_infer_init(&single, model) != 0) ||
        (att1_cluster_infer_init(&cluster, model, &config) != 0)) {
        fputs("failed to init single or cluster inference\n", stderr);
        return -1;
    }

    for (i = 0u; i < sizeof(prompt); i++) {
        if ((att1_infer_decode_token(&single,
                                     (uint32_t)prompt[i],
                                     &single_token) != 0) ||
            (att1_cluster_infer_decode_token(&cluster,
                                             (uint32_t)prompt[i],
                                             &cluster_token) != 0)) {
            att1_infer_free(&single);
            att1_cluster_infer_free(&cluster);
            fputs("single or cluster prompt decode failed\n", stderr);
            return -1;
        }
    }

    if ((single_token != cluster_token) ||
        !logits_close(single.logits,
                      cluster.logits,
                      model->config.vocab_size,
                      0.000001f)) {
        att1_infer_free(&single);
        att1_cluster_infer_free(&cluster);
        fputs("single and cluster outputs differ\n", stderr);
        return -1;
    }

    if (att1_cluster_infer_get_tile_counters(&cluster, 0u, &counters) != 0) {
        att1_infer_free(&single);
        att1_cluster_infer_free(&cluster);
        return -1;
    }
    if (counters.activations_received != sizeof(prompt)) {
        att1_infer_free(&single);
        att1_cluster_infer_free(&cluster);
        fputs("tile 0 did not receive prompt activation first\n", stderr);
        return -1;
    }

    for (tile = 0u; tile < cluster.shard_plan.tile_count; tile++) {
        const uint64_t expected_layers =
            (uint64_t)(cluster.shard_plan.tiles[tile].layer_end -
                       cluster.shard_plan.tiles[tile].layer_start) *
            (uint64_t)sizeof(prompt);

        if (att1_cluster_infer_get_tile_counters(&cluster,
                                                 (uint32_t)tile,
                                                 &counters) != 0) {
            att1_infer_free(&single);
            att1_cluster_infer_free(&cluster);
            return -1;
        }

        if (counters.layers_run != expected_layers) {
            att1_infer_free(&single);
            att1_cluster_infer_free(&cluster);
            fputs("tile ran layers outside its assignment\n", stderr);
            return -1;
        }
    }

    if (att1_cluster_infer_get_tile_counters(&cluster, 2u, &counters) != 0) {
        att1_infer_free(&single);
        att1_cluster_infer_free(&cluster);
        return -1;
    }
    if (counters.logits_sent != sizeof(prompt)) {
        att1_infer_free(&single);
        att1_cluster_infer_free(&cluster);
        fputs("last tile did not produce final logits\n", stderr);
        return -1;
    }

    att1_infer_free(&single);
    att1_cluster_infer_free(&cluster);
    return 0;
}

static int check_activation_packet_copy(void)
{
    att1_fabric fabric;
    att1_fabric_bus_config config = {2u, 2u, 32u};
    att1_fabric_packet packet;
    float sent[2] = {1.0f, 2.0f};
    float received[2] = {0.0f, 0.0f};
    size_t received_bytes = 0u;

    if (att1_fabric_create(&fabric, &config) != ATT1_OK) {
        return -1;
    }

    if (att1_fabric_send(&fabric,
                         0u,
                         1u,
                         ATT1_PACKET_ACTIVATION,
                         sent,
                         sizeof(sent),
                         7u) != ATT1_OK) {
        att1_fabric_destroy(&fabric);
        return -1;
    }

    sent[0] = 99.0f;
    sent[1] = 100.0f;
    if ((att1_fabric_receive(&fabric,
                             1u,
                             &packet,
                             received,
                             sizeof(received),
                             &received_bytes) != ATT1_OK) ||
        (packet.type != ATT1_PACKET_ACTIVATION) ||
        (received_bytes != sizeof(received)) ||
        (packet.payload_bytes != sizeof(received)) ||
        (received[0] != 1.0f) ||
        (received[1] != 2.0f)) {
        att1_fabric_destroy(&fabric);
        fputs("activation packet copy behavior failed\n", stderr);
        return -1;
    }

    att1_fabric_destroy(&fabric);
    return 0;
}

static int check_error_paths(const att1_model *model)
{
    const att1_cluster_infer_config invalid_config = {0u, 4u, 0u};
    const att1_cluster_infer_config config = {2u, 4u, 0u};
    const att1_cluster_infer_config full_config = {2u, 1u, 0u};
    att1_cluster_infer cluster;
    uint32_t token = 0u;

    if (att1_cluster_infer_init(&cluster, model, &invalid_config) == 0) {
        att1_cluster_infer_free(&cluster);
        fputs("invalid tile count should fail\n", stderr);
        return -1;
    }

    if (att1_cluster_infer_init(&cluster, model, &config) != 0) {
        fputs("cluster init failed for missing-shard test\n", stderr);
        return -1;
    }
    cluster.shard_plan.tiles[0].layer_end = 0u;
    if (att1_cluster_infer_decode_token(&cluster, (uint32_t)'A', &token) == 0) {
        att1_cluster_infer_free(&cluster);
        fputs("missing layer shard should fail\n", stderr);
        return -1;
    }
    att1_cluster_infer_free(&cluster);

    if (att1_cluster_infer_init(&cluster, model, &full_config) != 0) {
        fputs("cluster init failed for queue-full test\n", stderr);
        return -1;
    }
    if (att1_fabric_send(&cluster.fabric,
                         0u,
                         1u,
                         ATT1_PACKET_CONTROL,
                         NULL,
                         0u,
                         99u) != ATT1_OK) {
        att1_cluster_infer_free(&cluster);
        fputs("failed to prefill fabric queue\n", stderr);
        return -1;
    }
    if (att1_cluster_infer_decode_token(&cluster, (uint32_t)'A', &token) == 0) {
        att1_cluster_infer_free(&cluster);
        fputs("queue-full activation send should fail\n", stderr);
        return -1;
    }
    att1_cluster_infer_free(&cluster);

    return 0;
}

static int check_determinism(const att1_model *model)
{
    const att1_cluster_infer_config config = {2u, 4u, 0u};
    att1_cluster_infer lhs;
    att1_cluster_infer rhs;
    uint32_t lhs_token = 0u;
    uint32_t rhs_token = 0u;

    if ((att1_cluster_infer_init(&lhs, model, &config) != 0) ||
        (att1_cluster_infer_init(&rhs, model, &config) != 0)) {
        return -1;
    }

    if ((att1_cluster_infer_decode_token(&lhs, (uint32_t)'T', &lhs_token) != 0) ||
        (att1_cluster_infer_decode_token(&rhs, (uint32_t)'T', &rhs_token) != 0) ||
        (lhs_token != rhs_token) ||
        (memcmp(lhs.logits,
                rhs.logits,
                model->config.vocab_size * sizeof(float)) != 0)) {
        att1_cluster_infer_free(&lhs);
        att1_cluster_infer_free(&rhs);
        fputs("cluster determinism check failed\n", stderr);
        return -1;
    }

    att1_cluster_infer_free(&lhs);
    att1_cluster_infer_free(&rhs);
    return 0;
}

int main(void)
{
    att1_model model;

    if (att1_model_load(MODEL_PATH, &model) != 0) {
        fputs("failed to load dummy model\n", stderr);
        return 1;
    }

    if ((check_single_cluster_equivalence(&model) != 0) ||
        (check_activation_packet_copy() != 0) ||
        (check_error_paths(&model) != 0) ||
        (check_determinism(&model) != 0)) {
        att1_model_free(&model);
        return 1;
    }

    att1_model_free(&model);
    puts("cluster_infer test passed");
    return 0;
}
