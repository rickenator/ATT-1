#include "att1_attention.h"

#include <math.h>
#include <string.h>

static int att1_attention_config_valid(const att1_attention_config *config,
                                       const att1_kv_cache *cache)
{
    if ((config == NULL) || (cache == NULL)) {
        return 0;
    }

    if ((config->model_dim == 0u) ||
        (config->num_heads == 0u) ||
        (config->head_dim == 0u) ||
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

static int att1_attention_weights_valid(const att1_attention_weights *weights)
{
    return (weights != NULL) &&
           (weights->wq != NULL) &&
           (weights->wk != NULL) &&
           (weights->wv != NULL) &&
           (weights->wo != NULL);
}

static int att1_attention_q8_weights_valid(
    const att1_attention_q8_weights *weights)
{
    return (weights != NULL) &&
           (weights->wq != NULL) &&
           (weights->wk != NULL) &&
           (weights->wv != NULL) &&
           (weights->wo != NULL);
}

static float att1_dot_f32(const float *lhs, const float *rhs, size_t count)
{
    float sum = 0.0f;
    size_t i = 0u;

    for (i = 0u; i < count; i++) {
        sum += lhs[i] * rhs[i];
    }

    return sum;
}

static void att1_zero_f32(float *values, size_t count)
{
    size_t i = 0u;

    for (i = 0u; i < count; i++) {
        values[i] = 0.0f;
    }
}

static int att1_attention_apply_rope(float *query,
                                     float *key,
                                     const att1_attention_config *config,
                                     size_t position,
                                     att1_backend *backend)
{
    size_t head = 0u;

    if ((config->head_dim % 2u) != 0u) {
        return -1;
    }

    for (head = 0u; head < config->num_heads; head++) {
        float *q_head = &query[head * config->head_dim];
        float *k_head = &key[head * config->head_dim];

        if (backend->ops->rope_f32(backend,
                                   q_head,
                                   config->head_dim,
                                   position,
                                   config->rope_theta) != 0) {
            return -1;
        }

        if (backend->ops->rope_f32(backend,
                                   k_head,
                                   config->head_dim,
                                   position,
                                   config->rope_theta) != 0) {
            return -1;
        }
    }

    return 0;
}

int att1_attention_forward_f32(float *output,
                               att1_kv_cache *cache,
                               const float *input,
                               const att1_attention_weights *weights,
                               const att1_attention_config *config,
                               size_t position)
{
    att1_backend *backend = NULL;
    int rc = -1;

    if (att1_backend_default_create(&backend) != ATT1_OK) {
        return -1;
    }

    rc = att1_attention_forward_backend(output,
                                        cache,
                                        input,
                                        weights,
                                        config,
                                        position,
                                        backend);
    att1_backend_destroy(backend);
    return rc;
}

int att1_attention_forward_backend(float *output,
                                   att1_kv_cache *cache,
                                   const float *input,
                                   const att1_attention_weights *weights,
                                   const att1_attention_config *config,
                                   size_t position,
                                   att1_backend *backend)
{
    float *query = NULL;
    float *key = NULL;
    float *value = NULL;
    float *context = NULL;
    float *scores = NULL;
    float inv_sqrt_head_dim = 0.0f;
    size_t head = 0u;
    size_t pos = 0u;
    size_t dim = 0u;
    int rc = -1;

    if ((output == NULL) || (input == NULL)) {
        return -1;
    }

    if ((backend == NULL) || (backend->ops == NULL) ||
        (backend->ops->alloc == NULL) || (backend->ops->free == NULL) ||
        (backend->ops->matmul_f32 == NULL) ||
        (backend->ops->softmax_f32 == NULL) ||
        (backend->ops->rope_f32 == NULL)) {
        return -1;
    }

    if (!att1_attention_weights_valid(weights) ||
        !att1_attention_config_valid(config, cache)) {
        return -1;
    }

    if (position != cache->length) {
        return -1;
    }

    query = backend->ops->alloc(backend, config->model_dim * sizeof(float));
    key = backend->ops->alloc(backend, config->model_dim * sizeof(float));
    value = backend->ops->alloc(backend, config->model_dim * sizeof(float));
    context = backend->ops->alloc(backend, config->model_dim * sizeof(float));
    scores = backend->ops->alloc(backend, (cache->length + 1u) * sizeof(float));
    if ((query == NULL) ||
        (key == NULL) ||
        (value == NULL) ||
        (context == NULL) ||
        (scores == NULL)) {
        goto cleanup;
    }

    if (backend->ops->matmul_f32(backend,
                                 query,
                                 input,
                                 weights->wq,
                                 1u,
                                 config->model_dim,
                                 config->model_dim) != 0) {
        goto cleanup;
    }

    if (backend->ops->matmul_f32(backend,
                                 key,
                                 input,
                                 weights->wk,
                                 1u,
                                 config->model_dim,
                                 config->model_dim) != 0) {
        goto cleanup;
    }

    if (backend->ops->matmul_f32(backend,
                                 value,
                                 input,
                                 weights->wv,
                                 1u,
                                 config->model_dim,
                                 config->model_dim) != 0) {
        goto cleanup;
    }

    if (att1_attention_apply_rope(query, key, config, position, backend) != 0) {
        goto cleanup;
    }

    if (att1_kv_cache_append(cache, key, value) != 0) {
        goto cleanup;
    }

    att1_zero_f32(context, config->model_dim);
    inv_sqrt_head_dim = 1.0f / sqrtf((float)config->head_dim);

    for (head = 0u; head < config->num_heads; head++) {
        const float *q_head = &query[head * config->head_dim];
        float *context_head = &context[head * config->head_dim];

        for (pos = 0u; pos < cache->length; pos++) {
            const float *k_head = att1_kv_cache_key(cache, pos, head);

            if (k_head == NULL) {
                goto cleanup;
            }

            scores[pos] = att1_dot_f32(q_head, k_head, config->head_dim) *
                          inv_sqrt_head_dim;
        }

        if (backend->ops->softmax_f32(backend, scores, cache->length) != 0) {
            goto cleanup;
        }

        for (pos = 0u; pos < cache->length; pos++) {
            const float *v_head = att1_kv_cache_value(cache, pos, head);

            if (v_head == NULL) {
                goto cleanup;
            }

            for (dim = 0u; dim < config->head_dim; dim++) {
                context_head[dim] += scores[pos] * v_head[dim];
            }
        }
    }

    if (backend->ops->matmul_f32(backend,
                                 output,
                                 context,
                                 weights->wo,
                                 1u,
                                 config->model_dim,
                                 config->model_dim) != 0) {
        goto cleanup;
    }

    rc = 0;

cleanup:
    backend->ops->free(backend, query);
    backend->ops->free(backend, key);
    backend->ops->free(backend, value);
    backend->ops->free(backend, context);
    backend->ops->free(backend, scores);
    return rc;
}

int att1_attention_forward_backend_q8(
    float *output,
    att1_kv_cache *cache,
    const float *input,
    const att1_attention_q8_weights *weights,
    const att1_attention_config *config,
    size_t position,
    att1_backend *backend)
{
    float *query = NULL;
    float *key = NULL;
    float *value = NULL;
    float *context = NULL;
    float *scores = NULL;
    float inv_sqrt_head_dim = 0.0f;
    size_t head = 0u;
    size_t pos = 0u;
    size_t dim = 0u;
    int rc = -1;

    if ((output == NULL) || (input == NULL)) {
        return -1;
    }

    if ((backend == NULL) || (backend->ops == NULL) ||
        (backend->ops->alloc == NULL) || (backend->ops->free == NULL) ||
        (backend->ops->matmul_q8xf32 == NULL) ||
        (backend->ops->softmax_f32 == NULL) ||
        (backend->ops->rope_f32 == NULL)) {
        return -1;
    }

    if (!att1_attention_q8_weights_valid(weights) ||
        !att1_attention_config_valid(config, cache)) {
        return -1;
    }

    if (position != cache->length) {
        return -1;
    }

    query = backend->ops->alloc(backend, config->model_dim * sizeof(float));
    key = backend->ops->alloc(backend, config->model_dim * sizeof(float));
    value = backend->ops->alloc(backend, config->model_dim * sizeof(float));
    context = backend->ops->alloc(backend, config->model_dim * sizeof(float));
    scores = backend->ops->alloc(backend, (cache->length + 1u) * sizeof(float));
    if ((query == NULL) ||
        (key == NULL) ||
        (value == NULL) ||
        (context == NULL) ||
        (scores == NULL)) {
        goto cleanup;
    }

    if (backend->ops->matmul_q8xf32(backend,
                                    query,
                                    input,
                                    1u,
                                    config->model_dim,
                                    weights->wq) != 0) {
        goto cleanup;
    }

    if (backend->ops->matmul_q8xf32(backend,
                                    key,
                                    input,
                                    1u,
                                    config->model_dim,
                                    weights->wk) != 0) {
        goto cleanup;
    }

    if (backend->ops->matmul_q8xf32(backend,
                                    value,
                                    input,
                                    1u,
                                    config->model_dim,
                                    weights->wv) != 0) {
        goto cleanup;
    }

    if (att1_attention_apply_rope(query, key, config, position, backend) != 0) {
        goto cleanup;
    }

    if (att1_kv_cache_append(cache, key, value) != 0) {
        goto cleanup;
    }

    att1_zero_f32(context, config->model_dim);
    inv_sqrt_head_dim = 1.0f / sqrtf((float)config->head_dim);

    for (head = 0u; head < config->num_heads; head++) {
        const float *q_head = &query[head * config->head_dim];
        float *context_head = &context[head * config->head_dim];

        for (pos = 0u; pos < cache->length; pos++) {
            const float *k_head = att1_kv_cache_key(cache, pos, head);

            if (k_head == NULL) {
                goto cleanup;
            }

            scores[pos] = att1_dot_f32(q_head, k_head, config->head_dim) *
                          inv_sqrt_head_dim;
        }

        if (backend->ops->softmax_f32(backend, scores, cache->length) != 0) {
            goto cleanup;
        }

        for (pos = 0u; pos < cache->length; pos++) {
            const float *v_head = att1_kv_cache_value(cache, pos, head);

            if (v_head == NULL) {
                goto cleanup;
            }

            for (dim = 0u; dim < config->head_dim; dim++) {
                context_head[dim] += scores[pos] * v_head[dim];
            }
        }
    }

    if (backend->ops->matmul_q8xf32(backend,
                                    output,
                                    context,
                                    1u,
                                    config->model_dim,
                                    weights->wo) != 0) {
        goto cleanup;
    }

    rc = 0;

cleanup:
    backend->ops->free(backend, query);
    backend->ops->free(backend, key);
    backend->ops->free(backend, value);
    backend->ops->free(backend, context);
    backend->ops->free(backend, scores);
    return rc;
}

int att1_attention_forward_backend_q4(
    float *output,
    att1_kv_cache *cache,
    const float *input,
    const att1_attention_q4_weights *weights,
    const att1_attention_config *config,
    size_t position,
    att1_backend *backend)
{
    float *query = NULL;
    float *key = NULL;
    float *value = NULL;
    float *context = NULL;
    float *scores = NULL;
    float inv_sqrt_head_dim = 0.0f;
    size_t head = 0u;
    size_t pos = 0u;
    size_t dim = 0u;
    int rc = -1;

    if ((output == NULL) || (input == NULL)) {
        return -1;
    }

    if ((backend == NULL) || (backend->ops == NULL) ||
        (backend->ops->alloc == NULL) || (backend->ops->free == NULL) ||
        (backend->ops->softmax_f32 == NULL) ||
        (backend->ops->rope_f32 == NULL)) {
        return -1;
    }

    if ((weights == NULL) ||
        (weights->wq == NULL) || (weights->wk == NULL) ||
        (weights->wv == NULL) || (weights->wo == NULL)) {
        return -1;
    }

    if (!att1_attention_config_valid(config, cache)) {
        return -1;
    }

    if (position != cache->length) {
        return -1;
    }

    query   = backend->ops->alloc(backend, config->model_dim * sizeof(float));
    key     = backend->ops->alloc(backend, config->model_dim * sizeof(float));
    value   = backend->ops->alloc(backend, config->model_dim * sizeof(float));
    context = backend->ops->alloc(backend, config->model_dim * sizeof(float));
    scores  = backend->ops->alloc(backend, (cache->length + 1u) * sizeof(float));
    if ((query == NULL) || (key == NULL) || (value == NULL) ||
        (context == NULL) || (scores == NULL)) {
        goto cleanup;
    }

    if (att1_matmul_q4xf32(query, input, 1u, config->model_dim, weights->wq) != 0) {
        goto cleanup;
    }
    if (att1_matmul_q4xf32(key,   input, 1u, config->model_dim, weights->wk) != 0) {
        goto cleanup;
    }
    if (att1_matmul_q4xf32(value, input, 1u, config->model_dim, weights->wv) != 0) {
        goto cleanup;
    }

    if (att1_attention_apply_rope(query, key, config, position, backend) != 0) {
        goto cleanup;
    }

    if (att1_kv_cache_append(cache, key, value) != 0) {
        goto cleanup;
    }

    att1_zero_f32(context, config->model_dim);
    inv_sqrt_head_dim = 1.0f / sqrtf((float)config->head_dim);

    for (head = 0u; head < config->num_heads; head++) {
        const float *q_head = &query[head * config->head_dim];
        float *context_head = &context[head * config->head_dim];

        for (pos = 0u; pos < cache->length; pos++) {
            const float *k_head = att1_kv_cache_key(cache, pos, head);

            if (k_head == NULL) {
                goto cleanup;
            }

            scores[pos] = att1_dot_f32(q_head, k_head, config->head_dim) *
                          inv_sqrt_head_dim;
        }

        if (backend->ops->softmax_f32(backend, scores, cache->length) != 0) {
            goto cleanup;
        }

        for (pos = 0u; pos < cache->length; pos++) {
            const float *v_head = att1_kv_cache_value(cache, pos, head);

            if (v_head == NULL) {
                goto cleanup;
            }

            for (dim = 0u; dim < config->head_dim; dim++) {
                context_head[dim] += scores[pos] * v_head[dim];
            }
        }
    }

    if (att1_matmul_q4xf32(output, context, 1u, config->model_dim, weights->wo) != 0) {
        goto cleanup;
    }

    rc = 0;

cleanup:
    backend->ops->free(backend, query);
    backend->ops->free(backend, key);
    backend->ops->free(backend, value);
    backend->ops->free(backend, context);
    backend->ops->free(backend, scores);
    return rc;
}
