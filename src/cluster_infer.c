#include "att1_cluster_infer.h"

#include "att1_backend.h"
#include "att1_fabric.h"
#include "att1_kv_cache.h"
#include "att1_model_view.h"
#include "att1_sampler.h"
#include "att1_transformer_block.h"

#include <stdlib.h>
#include <string.h>

struct att1_cluster_infer {
    const att1_model *model;
    att1_shard_plan shard_plan;
    att1_fabric fabric;
    uint32_t host_tile_id;
    att1_kv_cache *layer_kv;
    float *hidden;
    float *next_hidden;
    float *norm;
    float *logits;
    att1_backend *backend;
    att1_trace_t *trace;
    att1_cluster_tile_counters *tile_counters;
    size_t position;
};

static int cluster_plan_complete(const att1_cluster_infer_t *infer)
{
    unsigned char *seen = NULL;
    size_t tile = 0u;
    size_t layer = 0u;
    int ok = 0;

    if ((infer == NULL) ||
        (infer->shard_plan.tiles == NULL) ||
        (infer->shard_plan.layer_to_tile == NULL) ||
        (infer->shard_plan.tile_count == 0u) ||
        (infer->shard_plan.layer_count != infer->model->config.n_layers)) {
        return 0;
    }

    seen = calloc(infer->shard_plan.layer_count, sizeof(*seen));
    if (seen == NULL) {
        return 0;
    }

    for (tile = 0u; tile < infer->shard_plan.tile_count; tile++) {
        const att1_layer_shard *shard = &infer->shard_plan.tiles[tile];

        if ((shard->tile_id != tile) ||
            (shard->layer_start > shard->layer_end) ||
            (shard->layer_end > infer->shard_plan.layer_count)) {
            goto done;
        }

        for (layer = shard->layer_start; layer < shard->layer_end; layer++) {
            if ((seen[layer] != 0u) ||
                (infer->shard_plan.layer_to_tile[layer] != tile)) {
                goto done;
            }
            seen[layer] = 1u;
        }
    }

    for (layer = 0u; layer < infer->shard_plan.layer_count; layer++) {
        if (seen[layer] == 0u) {
            goto done;
        }
    }

    ok = 1;

done:
    free(seen);
    return ok;
}

static att1_status_t receive_payload(att1_cluster_infer_t *infer,
                                     uint32_t tile_id,
                                     att1_packet_type type,
                                     float *payload,
                                     size_t payload_bytes)
{
    att1_fabric_packet packet;
    size_t received = 0u;
    att1_status_t status = ATT1_OK;

    status = (att1_status_t)att1_fabric_receive(&infer->fabric,
                                                tile_id,
                                                &packet,
                                                payload,
                                                payload_bytes,
                                                &received);
    if (status != ATT1_OK) {
        return status;
    }

    if ((packet.type != type) || (received != payload_bytes)) {
        return ATT1_ERR_BAD_FORMAT;
    }

    return ATT1_OK;
}

static void cluster_release_members(att1_cluster_infer_t *infer)
{
    uint32_t layer = 0u;

    if (infer == NULL) {
        return;
    }

    if ((infer->model != NULL) && (infer->layer_kv != NULL)) {
        for (layer = 0u; layer < infer->model->config.n_layers; layer++) {
            att1_kv_cache_free(&infer->layer_kv[layer]);
        }
    }

    att1_fabric_destroy(&infer->fabric);
    att1_shard_plan_free(&infer->shard_plan);
    free(infer->layer_kv);
    free(infer->hidden);
    free(infer->next_hidden);
    free(infer->norm);
    free(infer->logits);
    att1_backend_destroy(infer->backend);
    free(infer->tile_counters);
    memset(infer, 0, sizeof(*infer));
}

att1_status_t att1_cluster_infer_create(
    const att1_model *model,
    const att1_cluster_infer_config *config,
    att1_cluster_infer_t **out_infer)
{
    att1_cluster_infer_t *infer = NULL;
    att1_fabric_bus_config fabric_config;
    size_t tile_count = 0u;
    size_t queue_capacity = 0u;
    size_t max_payload_bytes = 0u;
    size_t required_payload_bytes = 0u;
    uint32_t layer = 0u;
    size_t tile = 0u;
    const size_t head_dim = (model != NULL && model->config.n_heads != 0u) ?
        (model->config.d_model / model->config.n_heads) : 0u;
    att1_status_t status = ATT1_OK;

    if ((out_infer == NULL) || (config == NULL)) {
        return ATT1_ERR_INVALID_ARG;
    }
    *out_infer = NULL;

    status = att1_model_view_validate_decoder(model);
    if (status != ATT1_OK) {
        return status;
    }

    if ((config->tile_count == 0u) ||
        (config->tile_count > (size_t)(UINT32_MAX - 1u))) {
        return ATT1_ERR_INVALID_ARG;
    }

    tile_count = config->tile_count;
    queue_capacity = config->fabric_queue_capacity != 0u ?
        config->fabric_queue_capacity : 4u;
    required_payload_bytes = model->config.vocab_size > model->config.d_model ?
        ((size_t)model->config.vocab_size * sizeof(float)) :
        ((size_t)model->config.d_model * sizeof(float));
    max_payload_bytes = config->fabric_max_payload_bytes != 0u ?
        config->fabric_max_payload_bytes : required_payload_bytes;
    if (max_payload_bytes < required_payload_bytes) {
        return ATT1_ERR_INVALID_ARG;
    }

    infer = calloc(1u, sizeof(*infer));
    if (infer == NULL) {
        return ATT1_ERR_OOM;
    }

    infer->model = model;
    infer->host_tile_id = (uint32_t)tile_count;

    status = att1_shard_plan_build(&infer->shard_plan, model, tile_count);
    if (status != ATT1_OK) {
        cluster_release_members(infer);
        free(infer);
        return status;
    }

    fabric_config.tile_count = tile_count + 1u;
    fabric_config.queue_capacity = queue_capacity;
    fabric_config.max_payload_bytes = max_payload_bytes;
    status = (att1_status_t)att1_fabric_create(&infer->fabric, &fabric_config);
    if (status != ATT1_OK) {
        cluster_release_members(infer);
        free(infer);
        return status;
    }

    infer->layer_kv = calloc(model->config.n_layers, sizeof(*infer->layer_kv));
    infer->hidden = calloc(model->config.d_model, sizeof(float));
    infer->next_hidden = calloc(model->config.d_model, sizeof(float));
    infer->norm = calloc(model->config.d_model, sizeof(float));
    infer->logits = calloc(model->config.vocab_size, sizeof(float));
    infer->tile_counters = calloc(tile_count, sizeof(*infer->tile_counters));
    status = att1_backend_default_create(&infer->backend);
    if ((infer->layer_kv == NULL) ||
        (infer->hidden == NULL) ||
        (infer->next_hidden == NULL) ||
        (infer->norm == NULL) ||
        (infer->logits == NULL) ||
        (infer->tile_counters == NULL) ||
        (status != ATT1_OK)) {
        cluster_release_members(infer);
        free(infer);
        return status == ATT1_OK ? ATT1_ERR_OOM : status;
    }

    for (layer = 0u; layer < model->config.n_layers; layer++) {
        if (att1_kv_cache_init(&infer->layer_kv[layer],
                               model->config.max_seq_len,
                               model->config.n_heads,
                               head_dim) != 0) {
            cluster_release_members(infer);
            free(infer);
            return ATT1_ERR_OOM;
        }
    }

    for (tile = 0u; tile < tile_count; tile++) {
        const att1_layer_shard *shard = &infer->shard_plan.tiles[tile];
        infer->tile_counters[tile].layer_start = shard->layer_start;
        infer->tile_counters[tile].layer_end = shard->layer_end;
    }

    *out_infer = infer;
    return ATT1_OK;
}

void att1_cluster_infer_destroy(att1_cluster_infer_t *infer)
{
    if (infer == NULL) {
        return;
    }

    cluster_release_members(infer);
    free(infer);
}

att1_status_t att1_cluster_infer_decode_token(att1_cluster_infer_t *infer,
                                              uint32_t token_id,
                                              uint32_t *out_token)
{
    const att1_model *model = NULL;
    const float *embedding = NULL;
    const float *output_norm = NULL;
    const float *output_weight = NULL;
    att1_transformer_block_config block_config;
    size_t d_model_bytes = 0u;
    size_t logits_bytes = 0u;
    size_t tile = 0u;
    size_t i = 0u;
    att1_status_t status = ATT1_OK;
    uint64_t token_start_us = 0u;

    if ((infer == NULL) || (infer->model == NULL) || (out_token == NULL)) {
        return ATT1_ERR_INVALID_ARG;
    }

    model = infer->model;
    if ((token_id >= model->config.vocab_size) ||
        (infer->position >= model->config.max_seq_len)) {
        return ATT1_ERR_INVALID_ARG;
    }

    if (!cluster_plan_complete(infer)) {
        return ATT1_ERR_STATE;
    }

    if (infer->trace != NULL) {
        token_start_us = att1_trace_now_us();
    }

    status = att1_model_view_token_embedding(model, &embedding);
    if (status != ATT1_OK) {
        return status;
    }
    status = att1_model_view_output_norm(model, &output_norm);
    if (status != ATT1_OK) {
        return status;
    }
    status = att1_model_view_output_weight(model, &output_weight);
    if (status != ATT1_OK) {
        return status;
    }

    d_model_bytes = (size_t)model->config.d_model * sizeof(float);
    logits_bytes = (size_t)model->config.vocab_size * sizeof(float);

    for (i = 0u; i < model->config.d_model; i++) {
        infer->hidden[i] = embedding[((size_t)token_id * model->config.d_model) + i];
    }

    status = (att1_status_t)att1_fabric_send(&infer->fabric,
                                             infer->host_tile_id,
                                             0u,
                                             ATT1_PACKET_ACTIVATION,
                                             infer->hidden,
                                             d_model_bytes,
                                             infer->position);
    if (status != ATT1_OK) {
        return status;
    }
    att1_trace_record_activation_send(infer->trace, 0u, d_model_bytes);
    att1_trace_record_fabric_send(infer->trace, d_model_bytes);

    block_config.model_dim = model->config.d_model;
    block_config.num_heads = model->config.n_heads;
    block_config.head_dim = model->config.d_model / model->config.n_heads;
    block_config.ffn_dim = model->config.d_ff;
    block_config.rms_epsilon = 0.000001f;
    block_config.rope_theta = 10000.0f;

    for (tile = 0u; tile < infer->shard_plan.tile_count; tile++) {
        const att1_layer_shard *shard = &infer->shard_plan.tiles[tile];
        uint32_t layer = 0u;

        status = receive_payload(infer,
                                 (uint32_t)tile,
                                 ATT1_PACKET_ACTIVATION,
                                 infer->hidden,
                                 d_model_bytes);
        if (status != ATT1_OK) {
            return status;
        }
        att1_trace_record_fabric_receive(infer->trace, d_model_bytes);
        infer->tile_counters[tile].activations_received++;

        for (layer = shard->layer_start; layer < shard->layer_end; layer++) {
            att1_transformer_block_weights weights;
            uint64_t layer_start_us = 0u;
            uint64_t layer_us = 0u;
            const uint64_t kv_reads = (uint64_t)(infer->position + 1u) *
                (uint64_t)model->config.n_heads;

            status = att1_model_view_load_layer_weights(model, layer, &weights);
            if (status != ATT1_OK) {
                return status;
            }

            if (infer->trace != NULL) {
                layer_start_us = att1_trace_now_us();
            }

            if (att1_transformer_block_forward_backend(infer->next_hidden,
                                                       &infer->layer_kv[layer],
                                                       infer->hidden,
                                                       &weights,
                                                       &block_config,
                                                       infer->position,
                                                       infer->backend) != 0) {
                return ATT1_ERR_STATE;
            }

            if (infer->trace != NULL) {
                layer_us = att1_trace_now_us() - layer_start_us;
                att1_trace_record_layer(infer->trace,
                                        layer,
                                        layer_us,
                                        1u,
                                        kv_reads,
                                        kv_reads);
            }

            memcpy(infer->hidden, infer->next_hidden, d_model_bytes);
            infer->tile_counters[tile].layers_run++;
            att1_trace_record_tile_layers(infer->trace, tile, 1u);
        }

        if ((tile + 1u) < infer->shard_plan.tile_count) {
            status = (att1_status_t)att1_fabric_send(&infer->fabric,
                                                     (uint32_t)tile,
                                                     (uint32_t)(tile + 1u),
                                                     ATT1_PACKET_ACTIVATION,
                                                     infer->hidden,
                                                     d_model_bytes,
                                                     infer->position);
            if (status != ATT1_OK) {
                return status;
            }
            att1_trace_record_activation_send(infer->trace,
                                             tile + 1u,
                                             d_model_bytes);
            att1_trace_record_fabric_send(infer->trace, d_model_bytes);
            infer->tile_counters[tile].activations_sent++;
        } else {
            if (infer->backend->ops->rmsnorm_f32(infer->backend,
                                                 infer->norm,
                                                 infer->hidden,
                                                 output_norm,
                                                 model->config.d_model,
                                                 0.000001f) != 0) {
                return ATT1_ERR_STATE;
            }

            if (infer->backend->ops->matmul_f32(infer->backend,
                                                infer->logits,
                                                infer->norm,
                                                output_weight,
                                                1u,
                                                model->config.vocab_size,
                                                model->config.d_model) != 0) {
                return ATT1_ERR_STATE;
            }

            if ((infer->backend->ops->sync != NULL) &&
                (infer->backend->ops->sync(infer->backend) != 0)) {
                return ATT1_ERR_STATE;
            }

            status = (att1_status_t)att1_fabric_send(&infer->fabric,
                                                     (uint32_t)tile,
                                                     infer->host_tile_id,
                                                     ATT1_PACKET_LOGITS,
                                                     infer->logits,
                                                     logits_bytes,
                                                     infer->position);
            if (status != ATT1_OK) {
                return status;
            }
            att1_trace_record_logits(infer->trace, tile, logits_bytes);
            att1_trace_record_fabric_send(infer->trace, logits_bytes);
            infer->tile_counters[tile].logits_sent++;
        }
    }

    status = receive_payload(infer,
                             infer->host_tile_id,
                             ATT1_PACKET_LOGITS,
                             infer->logits,
                             logits_bytes);
    if (status != ATT1_OK) {
        return status;
    }
    att1_trace_record_fabric_receive(infer->trace, logits_bytes);

    if (att1_sampler_greedy_f32(infer->logits,
                                model->config.vocab_size,
                                out_token) != 0) {
        return ATT1_ERR_STATE;
    }

    if (infer->trace != NULL) {
        att1_trace_record_token(infer->trace,
                                att1_trace_now_us() - token_start_us);
    }

    infer->position++;
    return ATT1_OK;
}

att1_status_t att1_cluster_infer_generate(att1_cluster_infer_t *infer,
                                          const unsigned char *prompt,
                                          size_t prompt_bytes,
                                          size_t generated_token_count,
                                          uint32_t *out_tokens,
                                          size_t out_token_capacity,
                                          size_t *out_token_count)
{
    uint32_t token = 0u;
    size_t i = 0u;
    size_t produced = 0u;
    att1_status_t status = ATT1_OK;

    if ((infer == NULL) || (prompt == NULL) || (out_tokens == NULL) ||
        (out_token_count == NULL)) {
        return ATT1_ERR_INVALID_ARG;
    }

    if ((prompt_bytes == 0u) ||
        (out_token_capacity < generated_token_count)) {
        return ATT1_ERR_INVALID_ARG;
    }

    for (i = 0u; i < prompt_bytes; i++) {
        uint32_t next = 0u;
        status = att1_cluster_infer_decode_token(infer,
                                                 (uint32_t)prompt[i],
                                                 &next);
        if (status != ATT1_OK) {
            return status;
        }
        token = next;
    }

    for (produced = 0u; produced < generated_token_count; produced++) {
        uint32_t next = 0u;
        out_tokens[produced] = token;
        if ((produced + 1u) < generated_token_count) {
            status = att1_cluster_infer_decode_token(infer, token, &next);
            if (status != ATT1_OK) {
                return status;
            }
            token = next;
        }
    }

    *out_token_count = generated_token_count;
    return ATT1_OK;
}

const float *att1_cluster_infer_logits(const att1_cluster_infer_t *infer,
                                       size_t *out_count)
{
    if ((infer == NULL) || (infer->model == NULL) || (infer->logits == NULL)) {
        if (out_count != NULL) {
            *out_count = 0u;
        }
        return NULL;
    }

    if (out_count != NULL) {
        *out_count = infer->model->config.vocab_size;
    }
    return infer->logits;
}

att1_status_t att1_cluster_infer_position(const att1_cluster_infer_t *infer,
                                          size_t *out_position)
{
    if ((infer == NULL) || (out_position == NULL)) {
        return ATT1_ERR_INVALID_ARG;
    }

    *out_position = infer->position;
    return ATT1_OK;
}

att1_status_t att1_cluster_infer_get_tile_counters(
    const att1_cluster_infer_t *infer,
    uint32_t tile_id,
    att1_cluster_tile_counters *out_counters)
{
    if ((infer == NULL) || (out_counters == NULL) ||
        (infer->tile_counters == NULL) ||
        ((size_t)tile_id >= infer->shard_plan.tile_count)) {
        return ATT1_ERR_INVALID_ARG;
    }

    *out_counters = infer->tile_counters[tile_id];
    return ATT1_OK;
}

att1_status_t att1_cluster_infer_get_tile_shard(
    const att1_cluster_infer_t *infer,
    uint32_t tile_id,
    att1_layer_shard *out_shard)
{
    const att1_layer_shard *shard = NULL;

    if ((infer == NULL) || (out_shard == NULL)) {
        return ATT1_ERR_INVALID_ARG;
    }

    shard = att1_shard_plan_tile(&infer->shard_plan, tile_id);
    if (shard == NULL) {
        return ATT1_ERR_INVALID_ARG;
    }

    *out_shard = *shard;
    return ATT1_OK;
}

att1_status_t att1_cluster_infer_set_trace(att1_cluster_infer_t *infer,
                                           att1_trace_t *trace)
{
    if (infer == NULL) {
        return ATT1_ERR_INVALID_ARG;
    }

    infer->trace = trace;
    return ATT1_OK;
}
