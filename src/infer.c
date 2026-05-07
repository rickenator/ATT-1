#include "att1_infer.h"

#include "att1_backend.h"
#include "att1_kv_cache.h"
#include "att1_model_view.h"
#include "att1_quant.h"
#include "att1_sampler.h"
#include "att1_transformer_block.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

typedef struct att1_infer_q8_layer {
    const float *attention_norm;
    const float *ffn_norm;
    att1_q8_matrix wq;
    att1_q8_matrix wk;
    att1_q8_matrix wv;
    att1_q8_matrix wo;
    att1_q8_matrix w_gate;
    att1_q8_matrix w_up;
    att1_q8_matrix w_down;
} att1_infer_q8_layer;

struct att1_infer {
    const att1_model *model;
    att1_kv_cache *layer_kv;
    float *hidden;
    float *next_hidden;
    float *norm;
    float *logits;
    att1_backend *backend;
    att1_infer_q8_layer *q8_layers;
    const float *q8_output_norm;
    att1_q8_matrix q8_output_weight;
    int q8_owned;
    int q8_ready;
    att1_trace_t *trace;
    size_t position;
};

static void infer_release_q8_layer(att1_infer_q8_layer *layer, int owned)
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

static void infer_release_q8(att1_infer_t *infer)
{
    uint32_t layer = 0u;

    if (infer == NULL) {
        return;
    }

    if ((infer->model != NULL) && (infer->q8_layers != NULL)) {
        for (layer = 0u; layer < infer->model->config.n_layers; layer++) {
            infer_release_q8_layer(&infer->q8_layers[layer], infer->q8_owned);
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

    infer_release_q8(infer);
    free(infer->layer_kv);
    free(infer->hidden);
    free(infer->next_hidden);
    free(infer->norm);
    free(infer->logits);
    att1_backend_destroy(infer->backend);
    memset(infer, 0, sizeof(*infer));
}

static int infer_backend_is_q8(const att1_backend *backend)
{
    return (backend != NULL) &&
           (backend->ops != NULL) &&
           (backend->ops->name != NULL) &&
           ((strcmp(backend->ops->name, "cpu-q8") == 0) ||
            (strcmp(backend->ops->name, "cuda-q8") == 0));
}

static int infer_backend_supports_q8(const att1_backend *backend)
{
    return infer_backend_is_q8(backend) &&
           (backend->ops->alloc != NULL) &&
           (backend->ops->free != NULL) &&
           (backend->ops->matmul_q8xf32 != NULL) &&
           (backend->ops->rmsnorm_f32 != NULL) &&
           (backend->ops->softmax_f32 != NULL) &&
           (backend->ops->rope_f32 != NULL) &&
           (backend->ops->ffn_swiglu_f32 != NULL);
}

static int infer_quantize_transposed(att1_q8_matrix *matrix,
                                     const float *weights,
                                     size_t input_count,
                                     size_t output_count)
{
    float *transposed = NULL;
    size_t input = 0u;
    size_t output = 0u;
    int rc = -1;

    if ((matrix == NULL) || (weights == NULL) ||
        (input_count == 0u) || (output_count == 0u)) {
        return -1;
    }

    if (input_count > ((size_t)-1) / output_count) {
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

static int infer_model_has_file_q8(const att1_model *model)
{
    const att1_model_tensor *tensor = att1_model_find_tensor(model, "output.weight");

    return (tensor != NULL) && (tensor->dtype == ATT1_MODEL_DTYPE_Q8);
}

static att1_status_t infer_layer_q8_name(char *out,
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

static att1_status_t infer_load_file_q8_layer(const att1_model *model,
                                              uint32_t layer,
                                              att1_infer_q8_layer *q8)
{
    char name[ATT1_MODEL_NAME_SIZE];
    const uint64_t d_model = model->config.d_model;
    const uint64_t d_ff = model->config.d_ff;
    att1_status_t status = ATT1_OK;

    status = infer_layer_q8_name(name, sizeof(name), layer, "attention_norm.weight");
    if (status != ATT1_OK) { return status; }
    status = att1_model_view_tensor_f32(model, name, 1u, d_model, 1u, &q8->attention_norm);
    if (status != ATT1_OK) { return status; }

#define LOAD_Q8(member, suffix, rows, cols) \
    do { \
        status = infer_layer_q8_name(name, sizeof(name), layer, suffix); \
        if (status != ATT1_OK) { return status; } \
        status = att1_model_view_tensor_q8(model, name, rows, cols, &q8->member); \
        if (status != ATT1_OK) { return status; } \
    } while (0)

    LOAD_Q8(wq,     "attention.wq.weight", d_model, d_model);
    LOAD_Q8(wk,     "attention.wk.weight", d_model, d_model);
    LOAD_Q8(wv,     "attention.wv.weight", d_model, d_model);
    LOAD_Q8(wo,     "attention.wo.weight", d_model, d_model);

    status = infer_layer_q8_name(name, sizeof(name), layer, "ffn_norm.weight");
    if (status != ATT1_OK) { return status; }
    status = att1_model_view_tensor_f32(model, name, 1u, d_model, 1u, &q8->ffn_norm);
    if (status != ATT1_OK) { return status; }

    LOAD_Q8(w_gate, "ffn.w_gate.weight",   d_ff, d_model);
    LOAD_Q8(w_up,   "ffn.w_up.weight",     d_ff, d_model);
    LOAD_Q8(w_down, "ffn.w_down.weight",   d_model, d_ff);

#undef LOAD_Q8

    return ATT1_OK;
}

static att1_status_t infer_prepare_file_q8(att1_infer_t *infer)
{
    const att1_model *model = infer->model;
    uint32_t layer = 0u;
    att1_status_t status = ATT1_OK;

    infer->q8_layers = calloc(model->config.n_layers, sizeof(*infer->q8_layers));
    if (infer->q8_layers == NULL) {
        return ATT1_ERR_OOM;
    }

    for (layer = 0u; layer < model->config.n_layers; layer++) {
        status = infer_load_file_q8_layer(model, layer, &infer->q8_layers[layer]);
        if (status != ATT1_OK) {
            infer_release_q8(infer);
            return status;
        }
    }

    status = att1_model_view_output_norm(model, &infer->q8_output_norm);
    if (status != ATT1_OK) {
        infer_release_q8(infer);
        return status;
    }

    status = att1_model_view_tensor_q8(model,
                                       "output.weight",
                                       model->config.vocab_size,
                                       model->config.d_model,
                                       &infer->q8_output_weight);
    if (status != ATT1_OK) {
        infer_release_q8(infer);
        return status;
    }

    infer->q8_owned = 0;
    infer->q8_ready = 1;
    return ATT1_OK;
}

static att1_status_t infer_prepare_q8(att1_infer_t *infer)
{
    const att1_model *model = NULL;
    const float *output_weight = NULL;
    uint32_t layer = 0u;
    att1_status_t status = ATT1_OK;

    if ((infer == NULL) || (infer->model == NULL)) {
        return ATT1_ERR_INVALID_ARG;
    }
    if (infer->q8_ready) {
        return ATT1_OK;
    }

    model = infer->model;
    infer_release_q8(infer);

    if (infer_model_has_file_q8(model)) {
        return infer_prepare_file_q8(infer);
    }

    infer->q8_layers = calloc(model->config.n_layers,
                              sizeof(*infer->q8_layers));
    if (infer->q8_layers == NULL) {
        return ATT1_ERR_OOM;
    }
    infer->q8_owned = 1;

    for (layer = 0u; layer < model->config.n_layers; layer++) {
        att1_transformer_block_weights weights;
        att1_infer_q8_layer *q8 = &infer->q8_layers[layer];

        status = att1_model_view_load_layer_weights(model, layer, &weights);
        if (status != ATT1_OK) {
            infer_release_q8(infer);
            return status;
        }

        q8->attention_norm = weights.attention_norm;
        q8->ffn_norm = weights.ffn_norm;

        if ((infer_quantize_transposed(&q8->wq,
                                       weights.wq,
                                       model->config.d_model,
                                       model->config.d_model) != 0) ||
            (infer_quantize_transposed(&q8->wk,
                                       weights.wk,
                                       model->config.d_model,
                                       model->config.d_model) != 0) ||
            (infer_quantize_transposed(&q8->wv,
                                       weights.wv,
                                       model->config.d_model,
                                       model->config.d_model) != 0) ||
            (infer_quantize_transposed(&q8->wo,
                                       weights.wo,
                                       model->config.d_model,
                                       model->config.d_model) != 0) ||
            (infer_quantize_transposed(&q8->w_gate,
                                       weights.w_gate,
                                       model->config.d_model,
                                       model->config.d_ff) != 0) ||
            (infer_quantize_transposed(&q8->w_up,
                                       weights.w_up,
                                       model->config.d_model,
                                       model->config.d_ff) != 0) ||
            (infer_quantize_transposed(&q8->w_down,
                                       weights.w_down,
                                       model->config.d_ff,
                                       model->config.d_model) != 0)) {
            infer_release_q8(infer);
            return ATT1_ERR_OOM;
        }
    }

    status = att1_model_view_output_norm(model, &infer->q8_output_norm);
    if (status != ATT1_OK) {
        infer_release_q8(infer);
        return status;
    }

    status = att1_model_view_output_weight(model, &output_weight);
    if (status != ATT1_OK) {
        infer_release_q8(infer);
        return status;
    }

    if (infer_quantize_transposed(&infer->q8_output_weight,
                                  output_weight,
                                  model->config.d_model,
                                  model->config.vocab_size) != 0) {
        infer_release_q8(infer);
        return ATT1_ERR_OOM;
    }

    infer->q8_ready = 1;
    return ATT1_OK;
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
    int use_q8 = 0;
    att1_status_t status = ATT1_OK;
    uint32_t layer = 0u;
    size_t i = 0u;
    uint64_t token_start_us = 0u;

    if ((infer == NULL) || (infer->model == NULL) || (out_token == NULL)) {
        return ATT1_ERR_INVALID_ARG;
    }

    use_q8 = infer_backend_is_q8(infer->backend);
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
    if (use_q8) {
        if (!infer->q8_ready || !infer_backend_supports_q8(infer->backend)) {
            return ATT1_ERR_STATE;
        }
        output_norm = infer->q8_output_norm;
    } else {
        status = att1_model_view_output_weight(model, &output_weight);
        if (status != ATT1_OK) {
            return status;
        }
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
        uint64_t layer_start_us = 0u;
        uint64_t layer_us = 0u;
        const uint64_t kv_reads = (uint64_t)(infer->position + 1u) *
            (uint64_t)model->config.n_heads;

        if (infer->trace != NULL) {
            layer_start_us = att1_trace_now_us();
        }

        if (use_q8) {
            att1_transformer_block_q8_weights weights;
            const att1_infer_q8_layer *q8 = &infer->q8_layers[layer];

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

            status = att1_model_view_load_layer_weights(model,
                                                        layer,
                                                        &weights);
            if (status != ATT1_OK) {
                return status;
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
    att1_status_t status = ATT1_OK;

    if ((infer == NULL) || (backend == NULL) || (backend->ops == NULL)) {
        return ATT1_ERR_INVALID_ARG;
    }

    if (infer_backend_is_q8(backend)) {
        if (!infer_backend_supports_q8(backend)) {
            return ATT1_ERR_UNSUPPORTED;
        }
        status = infer_prepare_q8(infer);
        if (status != ATT1_OK) {
            return status;
        }
    }

    att1_backend_destroy(infer->backend);
    infer->backend = backend;
    return ATT1_OK;
}
