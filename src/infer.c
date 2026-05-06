#include "att1_infer.h"

#include "att1_math.h"
#include "att1_sampler.h"
#include "att1_tokenizer.h"
#include "att1_transformer_block.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int tensor_is_f32_shape(const att1_model_tensor *tensor,
                               uint32_t ndims,
                               uint64_t dim0,
                               uint64_t dim1)
{
    if (tensor == NULL) {
        return 0;
    }

    if ((tensor->dtype != ATT1_MODEL_DTYPE_F32) || (tensor->ndims != ndims)) {
        return 0;
    }

    if ((tensor->shape[0] != dim0) || ((ndims > 1u) && (tensor->shape[1] != dim1))) {
        return 0;
    }

    return 1;
}

static const float *tensor_data(const att1_model *model,
                                const char *name,
                                uint32_t ndims,
                                uint64_t dim0,
                                uint64_t dim1)
{
    const att1_model_tensor *tensor = att1_model_find_tensor(model, name);

    if (!tensor_is_f32_shape(tensor, ndims, dim0, dim1)) {
        return NULL;
    }

    return (const float *)tensor->data;
}

static int layer_name(char *out,
                      size_t out_size,
                      uint32_t layer,
                      const char *suffix)
{
    const int written = snprintf(out, out_size, "layers.%u.%s", layer, suffix);

    if ((written < 0) || ((size_t)written >= out_size)) {
        return -1;
    }

    return 0;
}

static int load_layer_weights(const att1_model *model,
                              uint32_t layer,
                              att1_transformer_block_weights *weights)
{
    char name[ATT1_MODEL_NAME_SIZE];
    const uint64_t d_model = model->config.d_model;
    const uint64_t d_ff = model->config.d_ff;

    if (layer_name(name, sizeof(name), layer, "attention_norm.weight") != 0) {
        return -1;
    }
    weights->attention_norm = tensor_data(model, name, 1u, d_model, 1u);

    if (layer_name(name, sizeof(name), layer, "attention.wq.weight") != 0) {
        return -1;
    }
    weights->wq = tensor_data(model, name, 2u, d_model, d_model);

    if (layer_name(name, sizeof(name), layer, "attention.wk.weight") != 0) {
        return -1;
    }
    weights->wk = tensor_data(model, name, 2u, d_model, d_model);

    if (layer_name(name, sizeof(name), layer, "attention.wv.weight") != 0) {
        return -1;
    }
    weights->wv = tensor_data(model, name, 2u, d_model, d_model);

    if (layer_name(name, sizeof(name), layer, "attention.wo.weight") != 0) {
        return -1;
    }
    weights->wo = tensor_data(model, name, 2u, d_model, d_model);

    if (layer_name(name, sizeof(name), layer, "ffn_norm.weight") != 0) {
        return -1;
    }
    weights->ffn_norm = tensor_data(model, name, 1u, d_model, 1u);

    if (layer_name(name, sizeof(name), layer, "ffn.w_gate.weight") != 0) {
        return -1;
    }
    weights->w_gate = tensor_data(model, name, 2u, d_model, d_ff);

    if (layer_name(name, sizeof(name), layer, "ffn.w_up.weight") != 0) {
        return -1;
    }
    weights->w_up = tensor_data(model, name, 2u, d_model, d_ff);

    if (layer_name(name, sizeof(name), layer, "ffn.w_down.weight") != 0) {
        return -1;
    }
    weights->w_down = tensor_data(model, name, 2u, d_ff, d_model);

    if ((weights->attention_norm == NULL) ||
        (weights->ffn_norm == NULL) ||
        (weights->wq == NULL) ||
        (weights->wk == NULL) ||
        (weights->wv == NULL) ||
        (weights->wo == NULL) ||
        (weights->w_gate == NULL) ||
        (weights->w_up == NULL) ||
        (weights->w_down == NULL)) {
        return -1;
    }

    return 0;
}

int att1_infer_init(att1_infer *infer, const att1_model *model)
{
    uint32_t layer = 0u;
    const size_t d_model = model != NULL ? model->config.d_model : 0u;
    const size_t vocab_size = model != NULL ? model->config.vocab_size : 0u;
    const size_t head_dim = (model != NULL && model->config.n_heads != 0u) ?
        (model->config.d_model / model->config.n_heads) : 0u;

    if ((infer == NULL) || (model == NULL)) {
        return -1;
    }

    memset(infer, 0, sizeof(*infer));

    if ((model->config.vocab_size != 256u) ||
        (model->config.n_layers == 0u) ||
        (model->config.n_heads == 0u) ||
        (model->config.d_model == 0u) ||
        (model->config.d_ff == 0u) ||
        (model->config.max_seq_len == 0u) ||
        ((model->config.d_model % model->config.n_heads) != 0u) ||
        ((head_dim % 2u) != 0u)) {
        return -1;
    }

    if (tensor_data(model, "tok_embeddings.weight", 2u, vocab_size, d_model) == NULL ||
        tensor_data(model, "output_norm.weight", 1u, d_model, 1u) == NULL ||
        tensor_data(model, "output.weight", 2u, d_model, vocab_size) == NULL) {
        return -1;
    }

    infer->layer_kv = calloc(model->config.n_layers, sizeof(*infer->layer_kv));
    infer->hidden = calloc(d_model, sizeof(float));
    infer->next_hidden = calloc(d_model, sizeof(float));
    infer->norm = calloc(d_model, sizeof(float));
    infer->logits = calloc(vocab_size, sizeof(float));

    if ((infer->layer_kv == NULL) ||
        (infer->hidden == NULL) ||
        (infer->next_hidden == NULL) ||
        (infer->norm == NULL) ||
        (infer->logits == NULL)) {
        att1_infer_free(infer);
        return -1;
    }

    infer->model = model;
    for (layer = 0u; layer < model->config.n_layers; layer++) {
        if (att1_kv_cache_init(&infer->layer_kv[layer],
                               model->config.max_seq_len,
                               model->config.n_heads,
                               head_dim) != 0) {
            att1_infer_free(infer);
            return -1;
        }
    }

    return 0;
}

void att1_infer_free(att1_infer *infer)
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
    memset(infer, 0, sizeof(*infer));
}

int att1_infer_decode_token(att1_infer *infer,
                            uint32_t token_id,
                            uint32_t *out_token)
{
    const att1_model *model = NULL;
    const float *embedding = NULL;
    const float *output_norm = NULL;
    const float *output_weight = NULL;
    att1_transformer_block_config block_config;
    uint32_t layer = 0u;
    size_t i = 0u;

    if ((infer == NULL) || (infer->model == NULL) || (out_token == NULL)) {
        return -1;
    }

    model = infer->model;
    if ((token_id >= model->config.vocab_size) ||
        (infer->position >= model->config.max_seq_len)) {
        return -1;
    }

    embedding = tensor_data(model,
                            "tok_embeddings.weight",
                            2u,
                            model->config.vocab_size,
                            model->config.d_model);
    output_norm = tensor_data(model,
                              "output_norm.weight",
                              1u,
                              model->config.d_model,
                              1u);
    output_weight = tensor_data(model,
                                "output.weight",
                                2u,
                                model->config.d_model,
                                model->config.vocab_size);
    if ((embedding == NULL) || (output_norm == NULL) || (output_weight == NULL)) {
        return -1;
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

        if (load_layer_weights(model, layer, &weights) != 0) {
            return -1;
        }

        if (att1_transformer_block_forward_f32(infer->next_hidden,
                                               &infer->layer_kv[layer],
                                               infer->hidden,
                                               &weights,
                                               &block_config,
                                               infer->position) != 0) {
            return -1;
        }

        memcpy(infer->hidden,
               infer->next_hidden,
               model->config.d_model * sizeof(float));
    }

    if (att1_rmsnorm_f32(infer->norm,
                         infer->hidden,
                         output_norm,
                         model->config.d_model,
                         0.000001f) != 0) {
        return -1;
    }

    if (att1_matmul_f32(infer->logits,
                        infer->norm,
                        output_weight,
                        1u,
                        model->config.vocab_size,
                        model->config.d_model) != 0) {
        return -1;
    }

    if (att1_sampler_greedy_f32(infer->logits,
                                model->config.vocab_size,
                                out_token) != 0) {
        return -1;
    }

    infer->position++;
    return 0;
}

int att1_infer_generate(att1_infer *infer,
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

    if ((infer == NULL) || (prompt == NULL) || (out_tokens == NULL) ||
        (out_token_count == NULL)) {
        return -1;
    }

    if ((prompt_bytes == 0u) ||
        (out_token_capacity < generated_token_count)) {
        return -1;
    }

    for (i = 0u; i < prompt_bytes; i++) {
        uint32_t next = 0u;
        if (att1_infer_decode_token(infer, (uint32_t)prompt[i], &next) != 0) {
            return -1;
        }
        token = next;
    }

    for (produced = 0u; produced < generated_token_count; produced++) {
        uint32_t next = 0u;
        out_tokens[produced] = token;
        if ((produced + 1u) < generated_token_count) {
            if (att1_infer_decode_token(infer, token, &next) != 0) {
                return -1;
            }
            token = next;
        }
    }

    *out_token_count = generated_token_count;
    return 0;
}
