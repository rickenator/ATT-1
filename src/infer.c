#include "att1_infer.h"

#include "att1_backend.h"
#include "att1_kv_cache.h"
#include "att1_model_view.h"
#include "att1_sampler.h"
#include "att1_transformer_block.h"

#include <stdlib.h>
#include <string.h>

struct att1_infer {
    const att1_model *model;
    att1_kv_cache *layer_kv;
    float *hidden;
    float *next_hidden;
    float *norm;
    float *logits;
    att1_backend *backend;
    att1_trace_t *trace;
    size_t position;
};

static void infer_release_members(att1_infer_t *infer)
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

    free(infer->layer_kv);
    free(infer->hidden);
    free(infer->next_hidden);
    free(infer->norm);
    free(infer->logits);
    att1_backend_destroy(infer->backend);
    memset(infer, 0, sizeof(*infer));
}

att1_status_t att1_infer_create(const att1_model *model,
                                att1_infer_t **out_infer)
{
    att1_infer_t *infer = NULL;
    const size_t d_model = model != NULL ? model->config.d_model : 0u;
    const size_t vocab_size = model != NULL ? model->config.vocab_size : 0u;
    const size_t head_dim = (model != NULL && model->config.n_heads != 0u) ?
        (model->config.d_model / model->config.n_heads) : 0u;
    att1_status_t status = ATT1_OK;
    uint32_t layer = 0u;

    if (out_infer == NULL) {
        return ATT1_ERR_INVALID_ARG;
    }
    *out_infer = NULL;

    status = att1_model_view_validate_decoder(model);
    if (status != ATT1_OK) {
        return status;
    }

    infer = calloc(1u, sizeof(*infer));
    if (infer == NULL) {
        return ATT1_ERR_OOM;
    }

    infer->layer_kv = calloc(model->config.n_layers, sizeof(*infer->layer_kv));
    infer->hidden = calloc(d_model, sizeof(float));
    infer->next_hidden = calloc(d_model, sizeof(float));
    infer->norm = calloc(d_model, sizeof(float));
    infer->logits = calloc(vocab_size, sizeof(float));
    status = att1_backend_default_create(&infer->backend);
    if ((infer->layer_kv == NULL) ||
        (infer->hidden == NULL) ||
        (infer->next_hidden == NULL) ||
        (infer->norm == NULL) ||
        (infer->logits == NULL) ||
        (status != ATT1_OK)) {
        infer_release_members(infer);
        free(infer);
        return status == ATT1_OK ? ATT1_ERR_OOM : status;
    }

    infer->model = model;
    for (layer = 0u; layer < model->config.n_layers; layer++) {
        if (att1_kv_cache_init(&infer->layer_kv[layer],
                               model->config.max_seq_len,
                               model->config.n_heads,
                               head_dim) != 0) {
            infer_release_members(infer);
            free(infer);
            return ATT1_ERR_OOM;
        }
    }

    *out_infer = infer;
    return ATT1_OK;
}

void att1_infer_destroy(att1_infer_t *infer)
{
    if (infer == NULL) {
        return;
    }

    infer_release_members(infer);
    free(infer);
}

att1_status_t att1_infer_decode_token(att1_infer_t *infer,
                                      uint32_t token_id,
                                      uint32_t *out_token)
{
    const att1_model *model = NULL;
    const float *embedding = NULL;
    const float *output_norm = NULL;
    const float *output_weight = NULL;
    att1_transformer_block_config block_config;
    att1_status_t status = ATT1_OK;
    uint32_t layer = 0u;
    size_t i = 0u;
    uint64_t token_start_us = 0u;

    if ((infer == NULL) || (infer->model == NULL) || (out_token == NULL)) {
        return ATT1_ERR_INVALID_ARG;
    }

    model = infer->model;
    if ((token_id >= model->config.vocab_size) ||
        (infer->position >= model->config.max_seq_len)) {
        return ATT1_ERR_INVALID_ARG;
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

    for (i = 0u; i < model->config.d_model; i++) {
        infer->hidden[i] = embedding[((size_t)token_id * model->config.d_model) + i];
    }

    block_config.model_dim = model->config.d_model;
    block_config.num_heads = model->config.n_heads;
    block_config.head_dim = model->config.d_model / model->config.n_heads;
    block_config.ffn_dim = model->config.d_ff;
    block_config.rms_epsilon = 0.000001f;
    block_config.rope_theta = 10000.0f;

    for (layer = 0u; layer < model->config.n_layers; layer++) {
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

        memcpy(infer->hidden,
               infer->next_hidden,
               model->config.d_model * sizeof(float));
    }

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

    if (att1_sampler_greedy_f32(infer->logits,
                                model->config.vocab_size,
                                out_token) != 0) {
        return ATT1_ERR_STATE;
    }

    if (infer->trace != NULL) {
        att1_trace_record_logits(infer->trace,
                                 0u,
                                 (uint64_t)model->config.vocab_size *
                                 sizeof(float));
        att1_trace_record_token(infer->trace,
                                att1_trace_now_us() - token_start_us);
    }

    infer->position++;
    return ATT1_OK;
}

att1_status_t att1_infer_generate(att1_infer_t *infer,
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
        status = att1_infer_decode_token(infer, (uint32_t)prompt[i], &next);
        if (status != ATT1_OK) {
            return status;
        }
        token = next;
    }

    for (produced = 0u; produced < generated_token_count; produced++) {
        uint32_t next = 0u;
        out_tokens[produced] = token;
        if ((produced + 1u) < generated_token_count) {
            status = att1_infer_decode_token(infer, token, &next);
            if (status != ATT1_OK) {
                return status;
            }
            token = next;
        }
    }

    *out_token_count = generated_token_count;
    return ATT1_OK;
}

const float *att1_infer_logits(const att1_infer_t *infer,
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

att1_status_t att1_infer_position(const att1_infer_t *infer,
                                  size_t *out_position)
{
    if ((infer == NULL) || (out_position == NULL)) {
        return ATT1_ERR_INVALID_ARG;
    }

    *out_position = infer->position;
    return ATT1_OK;
}

att1_status_t att1_infer_layer_kv_length(const att1_infer_t *infer,
                                         uint32_t layer_id,
                                         size_t *out_length)
{
    if ((infer == NULL) || (infer->model == NULL) ||
        (infer->layer_kv == NULL) || (out_length == NULL) ||
        (layer_id >= infer->model->config.n_layers)) {
        return ATT1_ERR_INVALID_ARG;
    }

    *out_length = infer->layer_kv[layer_id].length;
    return ATT1_OK;
}

att1_status_t att1_infer_set_trace(att1_infer_t *infer,
                                   att1_trace_t *trace)
{
    if (infer == NULL) {
        return ATT1_ERR_INVALID_ARG;
    }

    infer->trace = trace;
    return ATT1_OK;
}

att1_status_t att1_infer_set_backend(att1_infer_t *infer,
                                     att1_backend *backend)
{
    if ((infer == NULL) || (backend == NULL) || (backend->ops == NULL)) {
        return ATT1_ERR_INVALID_ARG;
    }

    att1_backend_destroy(infer->backend);
    infer->backend = backend;
    return ATT1_OK;
}
