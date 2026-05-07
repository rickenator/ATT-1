#include "att1_cluster_infer.h"

#include "att1_backend.h"
#include "att1_fabric.h"
#include "att1_kv_cache.h"
#include "att1_model_view.h"
#include "att1_sampler.h"
#include "att1_transformer_block.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

typedef struct att1_cluster_q8_layer {
    const float *attention_norm;
    const float *ffn_norm;
    att1_q8_matrix wq;
    att1_q8_matrix wk;
    att1_q8_matrix wv;
    att1_q8_matrix wo;
    att1_q8_matrix w_gate;
    att1_q8_matrix w_up;
    att1_q8_matrix w_down;
} att1_cluster_q8_layer;

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
    att1_cluster_q8_layer *q8_layers;
    const float *q8_output_norm;
    att1_q8_matrix q8_output_weight;
    int q8_owned;
    int q8_ready;
    att1_trace_t *trace;
    att1_cluster_tile_counters *tile_counters;
    size_t position;
};

static int cluster_backend_is_q8(const att1_backend *backend)
{
    return (backend != NULL) &&
           (backend->ops != NULL) &&
           (backend->ops->name != NULL) &&
           ((strcmp(backend->ops->name, "cpu-q8") == 0) ||
            (strcmp(backend->ops->name, "cuda-q8") == 0));
}

static int cluster_backend_supports_q8(const att1_backend *backend)
{
    return cluster_backend_is_q8(backend) &&
           (backend->ops->alloc != NULL) &&
           (backend->ops->free != NULL) &&
           (backend->ops->matmul_q8xf32 != NULL) &&
           (backend->ops->rmsnorm_f32 != NULL) &&
           (backend->ops->softmax_f32 != NULL) &&
           (backend->ops->rope_f32 != NULL) &&
           (backend->ops->ffn_swiglu_f32 != NULL);
}

static void cluster_release_q8_layer(att1_cluster_q8_layer *layer, int owned)
{
    if (layer == NULL) {
        return;
    }

    if (owned) {
        att1_q8_matrix_free(&layer->wq);
        att1_q8_matrix_free(&layer->wk);
        att1_q8_matrix_free(&layer->wv);
        att1_q8_matrix_free(&layer->wo);
        att1_q8_matrix_free(&layer->w_gate);
        att1_q8_matrix_free(&layer->w_up);
        att1_q8_matrix_free(&layer->w_down);
    }
    memset(layer, 0, sizeof(*layer));
}

static void cluster_release_q8(att1_cluster_infer_t *infer)
{
    uint32_t layer = 0u;

    if (infer == NULL) {
        return;
    }

    if ((infer->model != NULL) && (infer->q8_layers != NULL)) {
        for (layer = 0u; layer < infer->model->config.n_layers; layer++) {
            cluster_release_q8_layer(&infer->q8_layers[layer], infer->q8_owned);
        }
    }

    free(infer->q8_layers);
    if (infer->q8_owned) {
        att1_q8_matrix_free(&infer->q8_output_weight);
    }
    infer->q8_layers = NULL;
    infer->q8_output_norm = NULL;
    infer->q8_owned = 0;
    infer->q8_ready = 0;
}

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
    cluster_release_q8(infer);
    free(infer->tile_counters);
    memset(infer, 0, sizeof(*infer));
}

static int cluster_quantize_transposed(att1_q8_matrix *matrix,
                                       const float *weights,
                                       size_t input_count,
                                       size_t output_count)
{
    float *transposed = NULL;
    size_t input = 0u;
    size_t output = 0u;
    int rc = -1;

    if ((matrix == NULL) || (weights == NULL) ||
        (input_count == 0u) || (output_count == 0u) ||
        (input_count > ((size_t)-1) / output_count)) {
        return -1;
    }

    transposed = malloc(input_count * output_count * sizeof(*transposed));
    if (transposed == NULL) {
        return -1;
    }

    for (input = 0u; input < input_count; input++) {
        for (output = 0u; output < output_count; output++) {
            transposed[(output * input_count) + input] =
                weights[(input * output_count) + output];
        }
    }

    rc = att1_quantize_q8_per_row(matrix,
                                  transposed,
                                  output_count,
                                  input_count);
    free(transposed);
    return rc;
}

static int cluster_model_has_file_q8(const att1_model *model)
{
    const att1_model_tensor *tensor = att1_model_find_tensor(model, "output.weight");

    return (tensor != NULL) && (tensor->dtype == ATT1_MODEL_DTYPE_Q8);
}

static att1_status_t cluster_layer_q8_name(char *out,
                                           size_t out_size,
                                           uint32_t layer,
                                           const char *suffix)
{
    const int written = snprintf(out, out_size, "layers.%u.%s", layer, suffix);

    if ((written < 0) || ((size_t)written >= out_size)) {
        return ATT1_ERR_INVALID_ARG;
    }

    return ATT1_OK;
}

static att1_status_t cluster_load_file_q8_layer(const att1_model *model,
                                                uint32_t layer,
                                                att1_cluster_q8_layer *q8)
{
    char name[ATT1_MODEL_NAME_SIZE];
    const uint64_t d_model = model->config.d_model;
    const uint64_t d_ff = model->config.d_ff;
    att1_status_t status = ATT1_OK;

    status = cluster_layer_q8_name(name, sizeof(name), layer, "attention_norm.weight");
    if (status != ATT1_OK) { return status; }
    status = att1_model_view_tensor_f32(model, name, 1u, d_model, 1u, &q8->attention_norm);
    if (status != ATT1_OK) { return status; }

#define LOAD_Q8(member, suffix, rows, cols) \
    do { \
        status = cluster_layer_q8_name(name, sizeof(name), layer, suffix); \
        if (status != ATT1_OK) { return status; } \
        status = att1_model_view_tensor_q8(model, name, rows, cols, &q8->member); \
        if (status != ATT1_OK) { return status; } \
    } while (0)

    LOAD_Q8(wq,     "attention.wq.weight", d_model, d_model);
    LOAD_Q8(wk,     "attention.wk.weight", d_model, d_model);
    LOAD_Q8(wv,     "attention.wv.weight", d_model, d_model);
    LOAD_Q8(wo,     "attention.wo.weight", d_model, d_model);

    status = cluster_layer_q8_name(name, sizeof(name), layer, "ffn_norm.weight");
    if (status != ATT1_OK) { return status; }
    status = att1_model_view_tensor_f32(model, name, 1u, d_model, 1u, &q8->ffn_norm);
    if (status != ATT1_OK) { return status; }

    LOAD_Q8(w_gate, "ffn.w_gate.weight",   d_ff, d_model);
    LOAD_Q8(w_up,   "ffn.w_up.weight",     d_ff, d_model);
    LOAD_Q8(w_down, "ffn.w_down.weight",   d_model, d_ff);

#undef LOAD_Q8

    return ATT1_OK;
}

static att1_status_t cluster_prepare_file_q8(att1_cluster_infer_t *infer)
{
    const att1_model *model = infer->model;
    uint32_t layer = 0u;
    att1_status_t status = ATT1_OK;

    infer->q8_layers = calloc(model->config.n_layers, sizeof(*infer->q8_layers));
    if (infer->q8_layers == NULL) {
        return ATT1_ERR_OOM;
    }

    for (layer = 0u; layer < model->config.n_layers; layer++) {
        status = cluster_load_file_q8_layer(model, layer, &infer->q8_layers[layer]);
        if (status != ATT1_OK) {
            cluster_release_q8(infer);
            return status;
        }
    }

    status = att1_model_view_output_norm(model, &infer->q8_output_norm);
    if (status != ATT1_OK) {
        cluster_release_q8(infer);
        return status;
    }

    status = att1_model_view_tensor_q8(model,
                                       "output.weight",
                                       model->config.vocab_size,
                                       model->config.d_model,
                                       &infer->q8_output_weight);
    if (status != ATT1_OK) {
        cluster_release_q8(infer);
        return status;
    }

    infer->q8_owned = 0;
    infer->q8_ready = 1;
    return ATT1_OK;
}

static att1_status_t cluster_prepare_runtime_q8(att1_cluster_infer_t *infer)
{
    const att1_model *model = infer->model;
    const float *output_weight = NULL;
    uint32_t layer = 0u;
    att1_status_t status = ATT1_OK;

    infer->q8_layers = calloc(model->config.n_layers, sizeof(*infer->q8_layers));
    if (infer->q8_layers == NULL) {
        return ATT1_ERR_OOM;
    }
    infer->q8_owned = 1;

    for (layer = 0u; layer < model->config.n_layers; layer++) {
        att1_transformer_block_weights weights;
        att1_cluster_q8_layer *q8 = &infer->q8_layers[layer];

        status = att1_model_view_load_layer_weights(model, layer, &weights);
        if (status != ATT1_OK) {
            cluster_release_q8(infer);
            return status;
        }

        q8->attention_norm = weights.attention_norm;
        q8->ffn_norm = weights.ffn_norm;

        if ((cluster_quantize_transposed(&q8->wq, weights.wq,
                                         model->config.d_model,
                                         model->config.d_model) != 0) ||
            (cluster_quantize_transposed(&q8->wk, weights.wk,
                                         model->config.d_model,
                                         model->config.d_model) != 0) ||
            (cluster_quantize_transposed(&q8->wv, weights.wv,
                                         model->config.d_model,
                                         model->config.d_model) != 0) ||
            (cluster_quantize_transposed(&q8->wo, weights.wo,
                                         model->config.d_model,
                                         model->config.d_model) != 0) ||
            (cluster_quantize_transposed(&q8->w_gate, weights.w_gate,
                                         model->config.d_model,
                                         model->config.d_ff) != 0) ||
            (cluster_quantize_transposed(&q8->w_up, weights.w_up,
                                         model->config.d_model,
                                         model->config.d_ff) != 0) ||
            (cluster_quantize_transposed(&q8->w_down, weights.w_down,
                                         model->config.d_ff,
                                         model->config.d_model) != 0)) {
            cluster_release_q8(infer);
            return ATT1_ERR_OOM;
        }
    }

    status = att1_model_view_output_norm(model, &infer->q8_output_norm);
    if (status != ATT1_OK) {
        cluster_release_q8(infer);
        return status;
    }

    status = att1_model_view_output_weight(model, &output_weight);
    if (status != ATT1_OK) {
        cluster_release_q8(infer);
        return status;
    }

    if (cluster_quantize_transposed(&infer->q8_output_weight,
                                    output_weight,
                                    model->config.d_model,
                                    model->config.vocab_size) != 0) {
        cluster_release_q8(infer);
        return ATT1_ERR_OOM;
    }

    infer->q8_ready = 1;
    return ATT1_OK;
}

static att1_status_t cluster_prepare_q8(att1_cluster_infer_t *infer)
{
    if ((infer == NULL) || (infer->model == NULL)) {
        return ATT1_ERR_INVALID_ARG;
    }
    if (infer->q8_ready) {
        return ATT1_OK;
    }

    cluster_release_q8(infer);
    if (cluster_model_has_file_q8(infer->model)) {
        return cluster_prepare_file_q8(infer);
    }

    return cluster_prepare_runtime_q8(infer);
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

    if (config->shard_plan_mode == ATT1_SHARD_PLAN_METADATA) {
        att1_meta_plan proposed;
        att1_status_t meta_status;

        if (model->shard_meta.count == 0u) {
            /* Metadata explicitly requested but absent. */
            cluster_release_members(infer);
            free(infer);
            return ATT1_ERR_INVALID_ARG;
        }

        meta_status = att1_meta_plan_build(model, &proposed);
        if (meta_status != ATT1_OK) {
            cluster_release_members(infer);
            free(infer);
            return meta_status;
        }

        meta_status = att1_shard_plan_from_meta(&proposed,
                                                model->config.n_layers,
                                                tile_count,
                                                &infer->shard_plan);
        att1_meta_plan_free(&proposed);

        if (meta_status != ATT1_OK) {
            cluster_release_members(infer);
            free(infer);
            return meta_status;
        }
    } else {
        status = att1_shard_plan_build(&infer->shard_plan, model, tile_count);
        if (status != ATT1_OK) {
            cluster_release_members(infer);
            free(infer);
            return status;
        }
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
    int use_q8 = 0;
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
    use_q8 = cluster_backend_is_q8(infer->backend);
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
    if (use_q8) {
        if (!infer->q8_ready || !cluster_backend_supports_q8(infer->backend)) {
            return ATT1_ERR_STATE;
        }
        output_norm = infer->q8_output_norm;
    } else {
        status = att1_model_view_output_weight(model, &output_weight);
        if (status != ATT1_OK) {
            return status;
        }
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
            uint64_t layer_start_us = 0u;
            uint64_t layer_us = 0u;
            const uint64_t kv_reads = (uint64_t)(infer->position + 1u) *
                (uint64_t)model->config.n_heads;

            if (infer->trace != NULL) {
                layer_start_us = att1_trace_now_us();
            }

            if (use_q8) {
                att1_transformer_block_q8_weights weights;
                const att1_cluster_q8_layer *q8 = &infer->q8_layers[layer];

                weights.attention_norm = q8->attention_norm;
                weights.ffn_norm = q8->ffn_norm;
                weights.wq = &q8->wq;
                weights.wk = &q8->wk;
                weights.wv = &q8->wv;
                weights.wo = &q8->wo;
                weights.w_gate = &q8->w_gate;
                weights.w_up = &q8->w_up;
                weights.w_down = &q8->w_down;

                if (att1_transformer_block_forward_backend_q8(
                        infer->next_hidden,
                        &infer->layer_kv[layer],
                        infer->hidden,
                        &weights,
                        &block_config,
                        infer->position,
                        infer->backend) != 0) {
                    return ATT1_ERR_STATE;
                }
            } else {
                att1_transformer_block_weights weights;

                status = att1_model_view_load_layer_weights(model, layer, &weights);
                if (status != ATT1_OK) {
                    return status;
                }

                if (att1_transformer_block_forward_backend(
                        infer->next_hidden,
                        &infer->layer_kv[layer],
                        infer->hidden,
                        &weights,
                        &block_config,
                        infer->position,
                        infer->backend) != 0) {
                    return ATT1_ERR_STATE;
                }
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

            if (use_q8) {
                if (infer->backend->ops->matmul_q8xf32(infer->backend,
                                                       infer->logits,
                                                       infer->norm,
                                                       1u,
                                                       model->config.d_model,
                                                       &infer->q8_output_weight) != 0) {
                    return ATT1_ERR_STATE;
                }
            } else {
                if (infer->backend->ops->matmul_f32(infer->backend,
                                                    infer->logits,
                                                    infer->norm,
                                                    output_weight,
                                                    1u,
                                                    model->config.vocab_size,
                                                    model->config.d_model) != 0) {
                    return ATT1_ERR_STATE;
                }
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

att1_status_t att1_cluster_infer_set_backend(att1_cluster_infer_t *infer,
                                             att1_backend *backend)
{
    att1_status_t status = ATT1_OK;

    if ((infer == NULL) || (backend == NULL) || (backend->ops == NULL)) {
        return ATT1_ERR_INVALID_ARG;
    }

    if (cluster_backend_is_q8(backend)) {
        if (!cluster_backend_supports_q8(backend)) {
            return ATT1_ERR_UNSUPPORTED;
        }
        status = cluster_prepare_q8(infer);
        if (status != ATT1_OK) {
            return status;
        }
    }

    att1_backend_destroy(infer->backend);
    infer->backend = backend;
    return ATT1_OK;
}
