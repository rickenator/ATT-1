#include "att1_model_view.h"

#include <stdio.h>
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

    if ((tensor->shape[0] != dim0) ||
        ((ndims > 1u) && (tensor->shape[1] != dim1))) {
        return 0;
    }

    return 1;
}

static att1_status_t layer_name(char *out,
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

att1_status_t att1_model_view_tensor_f32(const att1_model *model,
                                         const char *name,
                                         uint32_t ndims,
                                         uint64_t dim0,
                                         uint64_t dim1,
                                         const float **out_data)
{
    const att1_model_tensor *tensor = NULL;

    if ((model == NULL) || (name == NULL) || (out_data == NULL) ||
        (ndims == 0u) || (ndims > ATT1_MODEL_MAX_DIMS)) {
        return ATT1_ERR_INVALID_ARG;
    }

    *out_data = NULL;
    tensor = att1_model_find_tensor(model, name);
    if (tensor == NULL) {
        return ATT1_ERR_NOT_FOUND;
    }

    if (!tensor_is_f32_shape(tensor, ndims, dim0, dim1)) {
        return ATT1_ERR_SHAPE;
    }

    *out_data = (const float *)tensor->data;
    return ATT1_OK;
}

att1_status_t att1_model_view_token_embedding(const att1_model *model,
                                              const float **out_data)
{
    if (model == NULL) {
        return ATT1_ERR_INVALID_ARG;
    }

    return att1_model_view_tensor_f32(model,
                                      "tok_embeddings.weight",
                                      2u,
                                      model->config.vocab_size,
                                      model->config.d_model,
                                      out_data);
}

att1_status_t att1_model_view_output_norm(const att1_model *model,
                                          const float **out_data)
{
    if (model == NULL) {
        return ATT1_ERR_INVALID_ARG;
    }

    return att1_model_view_tensor_f32(model,
                                      "output_norm.weight",
                                      1u,
                                      model->config.d_model,
                                      1u,
                                      out_data);
}

att1_status_t att1_model_view_output_weight(const att1_model *model,
                                            const float **out_data)
{
    if (model == NULL) {
        return ATT1_ERR_INVALID_ARG;
    }

    return att1_model_view_tensor_f32(model,
                                      "output.weight",
                                      2u,
                                      model->config.d_model,
                                      model->config.vocab_size,
                                      out_data);
}

att1_status_t att1_model_view_validate_decoder(const att1_model *model)
{
    const float *data = NULL;
    const size_t head_dim = (model != NULL && model->config.n_heads != 0u) ?
        (model->config.d_model / model->config.n_heads) : 0u;
    att1_status_t status = ATT1_OK;

    if (model == NULL) {
        return ATT1_ERR_INVALID_ARG;
    }

    if ((model->config.vocab_size != 256u) ||
        (model->config.n_layers == 0u) ||
        (model->config.n_heads == 0u) ||
        (model->config.d_model == 0u) ||
        (model->config.d_ff == 0u) ||
        (model->config.max_seq_len == 0u) ||
        ((model->config.d_model % model->config.n_heads) != 0u) ||
        ((head_dim % 2u) != 0u)) {
        return ATT1_ERR_INVALID_ARG;
    }

    status = att1_model_view_token_embedding(model, &data);
    if (status != ATT1_OK) {
        return status;
    }

    status = att1_model_view_output_norm(model, &data);
    if (status != ATT1_OK) {
        return status;
    }

    return att1_model_view_output_weight(model, &data);
}

att1_status_t att1_model_view_load_layer_weights(
    const att1_model *model,
    uint32_t layer,
    att1_transformer_block_weights *out_weights)
{
    char name[ATT1_MODEL_NAME_SIZE];
    const uint64_t d_model = model != NULL ? model->config.d_model : 0u;
    const uint64_t d_ff = model != NULL ? model->config.d_ff : 0u;
    att1_status_t status = ATT1_OK;

    if ((model == NULL) || (out_weights == NULL) ||
        (layer >= model->config.n_layers)) {
        return ATT1_ERR_INVALID_ARG;
    }

    memset(out_weights, 0, sizeof(*out_weights));

#define LOAD_LAYER_TENSOR(member, suffix, ndims, dim0, dim1) \
    do { \
        status = layer_name(name, sizeof(name), layer, suffix); \
        if (status != ATT1_OK) { \
            return status; \
        } \
        status = att1_model_view_tensor_f32(model, name, ndims, dim0, dim1, \
                                            &out_weights->member); \
        if (status != ATT1_OK) { \
            return status; \
        } \
    } while (0)

    LOAD_LAYER_TENSOR(attention_norm, "attention_norm.weight", 1u, d_model, 1u);
    LOAD_LAYER_TENSOR(wq, "attention.wq.weight", 2u, d_model, d_model);
    LOAD_LAYER_TENSOR(wk, "attention.wk.weight", 2u, d_model, d_model);
    LOAD_LAYER_TENSOR(wv, "attention.wv.weight", 2u, d_model, d_model);
    LOAD_LAYER_TENSOR(wo, "attention.wo.weight", 2u, d_model, d_model);
    LOAD_LAYER_TENSOR(ffn_norm, "ffn_norm.weight", 1u, d_model, 1u);
    LOAD_LAYER_TENSOR(w_gate, "ffn.w_gate.weight", 2u, d_model, d_ff);
    LOAD_LAYER_TENSOR(w_up, "ffn.w_up.weight", 2u, d_model, d_ff);
    LOAD_LAYER_TENSOR(w_down, "ffn.w_down.weight", 2u, d_ff, d_model);

#undef LOAD_LAYER_TENSOR

    return ATT1_OK;
}
