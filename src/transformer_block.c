#include "att1_transformer_block.h"

#include "att1_attention.h"

static int att1_block_config_valid(const att1_transformer_block_config *config,
                                   const att1_kv_cache *cache)
{
    if ((config == NULL) || (cache == NULL)) {
        return 0;
    }

    if ((config->model_dim == 0u) ||
        (config->num_heads == 0u) ||
        (config->head_dim == 0u) ||
        (config->ffn_dim == 0u) ||
        (config->rms_epsilon <= 0.0f) ||
        (config->rope_theta <= 0.0f)) {
        return 0;
    }

    if ((config->num_heads * config->head_dim) != config->model_dim) {
        return 0;
    }

    if ((cache->num_heads != config->num_heads) ||
        (cache->head_dim != config->head_dim)) {
        return 0;
    }

    return 1;
}

static int att1_block_weights_valid(
    const att1_transformer_block_weights *weights)
{
    return (weights != NULL) &&
           (weights->attention_norm != NULL) &&
           (weights->ffn_norm != NULL) &&
           (weights->wq != NULL) &&
           (weights->wk != NULL) &&
           (weights->wv != NULL) &&
           (weights->wo != NULL) &&
           (weights->w_gate != NULL) &&
           (weights->w_up != NULL) &&
           (weights->w_down != NULL);
}

static void att1_add_f32(float *dst,
                         const float *lhs,
                         const float *rhs,
                         size_t count)
{
    size_t i = 0u;

    for (i = 0u; i < count; i++) {
        dst[i] = lhs[i] + rhs[i];
    }
}

int att1_transformer_block_forward_f32(
    float *output,
    att1_kv_cache *cache,
    const float *input,
    const att1_transformer_block_weights *weights,
    const att1_transformer_block_config *config,
    size_t position)
{
    att1_backend *backend = NULL;
    int rc = -1;

    if (att1_backend_default_create(&backend) != ATT1_OK) {
        return -1;
    }

    rc = att1_transformer_block_forward_backend(output,
                                                cache,
                                                input,
                                                weights,
                                                config,
                                                position,
                                                backend);
    att1_backend_destroy(backend);
    return rc;
}

int att1_transformer_block_forward_backend(
    float *output,
    att1_kv_cache *cache,
    const float *input,
    const att1_transformer_block_weights *weights,
    const att1_transformer_block_config *config,
    size_t position,
    att1_backend *backend)
{
    att1_attention_config attention_config;
    att1_attention_weights attention_weights;
    float *attention_input = NULL;
    float *attention_output = NULL;
    float *attention_residual = NULL;
    float *ffn_input = NULL;
    float *ffn_gate = NULL;
    float *ffn_up = NULL;
    float *ffn_hidden = NULL;
    float *ffn_output = NULL;
    int rc = -1;

    if ((output == NULL) || (input == NULL)) {
        return -1;
    }

    if ((backend == NULL) || (backend->ops == NULL) ||
        (backend->ops->alloc == NULL) || (backend->ops->free == NULL) ||
        (backend->ops->matmul_f32 == NULL) ||
        (backend->ops->rmsnorm_f32 == NULL) ||
        (backend->ops->ffn_swiglu_f32 == NULL)) {
        return -1;
    }

    if (!att1_block_weights_valid(weights) ||
        !att1_block_config_valid(config, cache)) {
        return -1;
    }

    attention_input = backend->ops->alloc(backend, config->model_dim * sizeof(float));
    attention_output = backend->ops->alloc(backend, config->model_dim * sizeof(float));
    attention_residual = backend->ops->alloc(backend, config->model_dim * sizeof(float));
    ffn_input = backend->ops->alloc(backend, config->model_dim * sizeof(float));
    ffn_gate = backend->ops->alloc(backend, config->ffn_dim * sizeof(float));
    ffn_up = backend->ops->alloc(backend, config->ffn_dim * sizeof(float));
    ffn_hidden = backend->ops->alloc(backend, config->ffn_dim * sizeof(float));
    ffn_output = backend->ops->alloc(backend, config->model_dim * sizeof(float));

    if ((attention_input == NULL) ||
        (attention_output == NULL) ||
        (attention_residual == NULL) ||
        (ffn_input == NULL) ||
        (ffn_gate == NULL) ||
        (ffn_up == NULL) ||
        (ffn_hidden == NULL) ||
        (ffn_output == NULL)) {
        goto cleanup;
    }

    if (backend->ops->rmsnorm_f32(backend,
                                  attention_input,
                                  input,
                                  weights->attention_norm,
                                  config->model_dim,
                                  config->rms_epsilon) != 0) {
        goto cleanup;
    }

    attention_config.model_dim = config->model_dim;
    attention_config.num_heads = config->num_heads;
    attention_config.head_dim = config->head_dim;
    attention_config.rope_theta = config->rope_theta;

    attention_weights.wq = weights->wq;
    attention_weights.wk = weights->wk;
    attention_weights.wv = weights->wv;
    attention_weights.wo = weights->wo;

    if (att1_attention_forward_backend(attention_output,
                                       cache,
                                       attention_input,
                                       &attention_weights,
                                       &attention_config,
                                       position,
                                       backend) != 0) {
        goto cleanup;
    }

    att1_add_f32(attention_residual,
                 input,
                 attention_output,
                 config->model_dim);

    if (backend->ops->rmsnorm_f32(backend,
                                  ffn_input,
                                  attention_residual,
                                  weights->ffn_norm,
                                  config->model_dim,
                                  config->rms_epsilon) != 0) {
        goto cleanup;
    }

    if (backend->ops->matmul_f32(backend,
                                 ffn_gate,
                                 ffn_input,
                                 weights->w_gate,
                                 1u,
                                 config->ffn_dim,
                                 config->model_dim) != 0) {
        goto cleanup;
    }

    if (backend->ops->matmul_f32(backend,
                                 ffn_up,
                                 ffn_input,
                                 weights->w_up,
                                 1u,
                                 config->ffn_dim,
                                 config->model_dim) != 0) {
        goto cleanup;
    }

    if (backend->ops->ffn_swiglu_f32(backend,
                                     ffn_hidden,
                                     ffn_gate,
                                     ffn_up,
                                     config->ffn_dim) != 0) {
        goto cleanup;
    }

    if (backend->ops->matmul_f32(backend,
                                 ffn_output,
                                 ffn_hidden,
                                 weights->w_down,
                                 1u,
                                 config->model_dim,
                                 config->ffn_dim) != 0) {
        goto cleanup;
    }

    att1_add_f32(output, attention_residual, ffn_output, config->model_dim);
    rc = 0;

cleanup:
    backend->ops->free(backend, attention_input);
    backend->ops->free(backend, attention_output);
    backend->ops->free(backend, attention_residual);
    backend->ops->free(backend, ffn_input);
    backend->ops->free(backend, ffn_gate);
    backend->ops->free(backend, ffn_up);
    backend->ops->free(backend, ffn_hidden);
    backend->ops->free(backend, ffn_output);
    return rc;
}
