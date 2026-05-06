#include "att1_cluster_infer.h"
#include "att1_fabric.h"
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
    att1_infer_t *single = NULL;
    att1_cluster_infer_t *cluster = NULL;
    const float *single_logits = NULL;
    const float *cluster_logits = NULL;
    size_t single_logits_count = 0u;
    size_t cluster_logits_count = 0u;
    uint32_t single_token = 0u;
    uint32_t cluster_token = 0u;
    att1_cluster_tile_counters counters;
    size_t tile = 0u;
    size_t i = 0u;

    if ((att1_infer_create(model, &single) != ATT1_OK) ||
        (att1_cluster_infer_create(model, &config, &cluster) != ATT1_OK)) {
        att1_infer_destroy(single);
        att1_cluster_infer_destroy(cluster);
        fputs("failed to init single or cluster inference\n", stderr);
        return -1;
    }

    for (i = 0u; i < sizeof(prompt); i++) {
        if ((att1_infer_decode_token(single,
                                     (uint32_t)prompt[i],
                                     &single_token) != ATT1_OK) ||
            (att1_cluster_infer_decode_token(cluster,
                                             (uint32_t)prompt[i],
                                             &cluster_token) != ATT1_OK)) {
            att1_infer_destroy(single);
            att1_cluster_infer_destroy(cluster);
            fputs("single or cluster prompt decode failed\n", stderr);
            return -1;
        }
    }

    single_logits = att1_infer_logits(single, &single_logits_count);
    cluster_logits = att1_cluster_infer_logits(cluster, &cluster_logits_count);
    if ((single_token != cluster_token) ||
        (single_logits == NULL) ||
        (cluster_logits == NULL) ||
        (single_logits_count != model->config.vocab_size) ||
        (cluster_logits_count != model->config.vocab_size) ||
        !logits_close(single_logits,
                      cluster_logits,
                      model->config.vocab_size,
                      0.000001f)) {
        att1_infer_destroy(single);
        att1_cluster_infer_destroy(cluster);
        fputs("single and cluster outputs differ\n", stderr);
        return -1;
    }

    if (att1_cluster_infer_get_tile_counters(cluster, 0u, &counters) != ATT1_OK) {
        att1_infer_destroy(single);
        att1_cluster_infer_destroy(cluster);
        return -1;
    }
    if (counters.activations_received != sizeof(prompt)) {
        att1_infer_destroy(single);
        att1_cluster_infer_destroy(cluster);
        fputs("tile 0 did not receive prompt activation first\n", stderr);
        return -1;
    }

    for (tile = 0u; tile < config.tile_count; tile++) {
        att1_layer_shard shard;
        uint64_t expected_layers = 0u;

        if (att1_cluster_infer_get_tile_shard(cluster,
                                              (uint32_t)tile,
                                              &shard) != ATT1_OK) {
            att1_infer_destroy(single);
            att1_cluster_infer_destroy(cluster);
            return -1;
        }
        expected_layers = (uint64_t)(shard.layer_end - shard.layer_start) *
            (uint64_t)sizeof(prompt);

        if (att1_cluster_infer_get_tile_counters(cluster,
                                                 (uint32_t)tile,
                                                 &counters) != ATT1_OK) {
            att1_infer_destroy(single);
            att1_cluster_infer_destroy(cluster);
            return -1;
        }

        if (counters.layers_run != expected_layers) {
            att1_infer_destroy(single);
            att1_cluster_infer_destroy(cluster);
            fputs("tile ran layers outside its assignment\n", stderr);
            return -1;
        }
    }

    if (att1_cluster_infer_get_tile_counters(cluster, 2u, &counters) != ATT1_OK) {
        att1_infer_destroy(single);
        att1_cluster_infer_destroy(cluster);
        return -1;
    }
    if (counters.logits_sent != sizeof(prompt)) {
        att1_infer_destroy(single);
        att1_cluster_infer_destroy(cluster);
        fputs("last tile did not produce final logits\n", stderr);
        return -1;
    }

    att1_infer_destroy(single);
    att1_cluster_infer_destroy(cluster);
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
    const att1_cluster_infer_config small_payload_config = {2u, 4u, 4u};
    att1_cluster_infer_t *cluster = NULL;

    if (att1_cluster_infer_create(model,
                                  &invalid_config,
                                  &cluster) != ATT1_ERR_INVALID_ARG) {
        att1_cluster_infer_destroy(cluster);
        fputs("invalid tile count should fail\n", stderr);
        return -1;
    }

    if (att1_cluster_infer_create(model,
                                  &small_payload_config,
                                  &cluster) != ATT1_ERR_INVALID_ARG) {
        att1_cluster_infer_destroy(cluster);
        fputs("too-small fabric payload should fail\n", stderr);
        return -1;
    }

    return 0;
}

static int check_determinism(const att1_model *model)
{
    const att1_cluster_infer_config config = {2u, 4u, 0u};
    att1_cluster_infer_t *lhs = NULL;
    att1_cluster_infer_t *rhs = NULL;
    const float *lhs_logits = NULL;
    const float *rhs_logits = NULL;
    size_t lhs_logits_count = 0u;
    size_t rhs_logits_count = 0u;
    uint32_t lhs_token = 0u;
    uint32_t rhs_token = 0u;

    if ((att1_cluster_infer_create(model, &config, &lhs) != ATT1_OK) ||
        (att1_cluster_infer_create(model, &config, &rhs) != ATT1_OK)) {
        att1_cluster_infer_destroy(lhs);
        att1_cluster_infer_destroy(rhs);
        return -1;
    }

    lhs_logits = att1_cluster_infer_logits(lhs, &lhs_logits_count);
    rhs_logits = att1_cluster_infer_logits(rhs, &rhs_logits_count);
    if ((att1_cluster_infer_decode_token(lhs, (uint32_t)'T', &lhs_token) != ATT1_OK) ||
        (att1_cluster_infer_decode_token(rhs, (uint32_t)'T', &rhs_token) != ATT1_OK) ||
        (lhs_token != rhs_token) ||
        (lhs_logits == NULL) ||
        (rhs_logits == NULL) ||
        (lhs_logits_count != model->config.vocab_size) ||
        (rhs_logits_count != model->config.vocab_size) ||
        (memcmp(lhs_logits,
                rhs_logits,
                model->config.vocab_size * sizeof(float)) != 0)) {
        att1_cluster_infer_destroy(lhs);
        att1_cluster_infer_destroy(rhs);
        fputs("cluster determinism check failed\n", stderr);
        return -1;
    }

    att1_cluster_infer_destroy(lhs);
    att1_cluster_infer_destroy(rhs);
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
