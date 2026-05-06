#include "att1_attention.h"

#include "att1_math.h"

#include <math.h>
#include <stdlib.h>
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
                                     size_t position)
{
    size_t head = 0u;

    if ((config->head_dim % 2u) != 0u) {
        return -1;
    }

    for (head = 0u; head < config->num_heads; head++) {
        float *q_head = &query[head * config->head_dim];
        float *k_head = &key[head * config->head_dim];

        if (att1_rope_f32(q_head,
                          config->head_dim,
                          position,
                          config->rope_theta) != 0) {
            return -1;
        }

        if (att1_rope_f32(k_head,
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

    if (!att1_attention_weights_valid(weights) ||
        !att1_attention_config_valid(config, cache)) {
        return -1;
    }

    if (position != cache->length) {
        return -1;
    }

    query = malloc(config->model_dim * sizeof(float));
    key = malloc(config->model_dim * sizeof(float));
    value = malloc(config->model_dim * sizeof(float));
    context = malloc(config->model_dim * sizeof(float));
    scores = malloc((cache->length + 1u) * sizeof(float));
    if ((query == NULL) ||
        (key == NULL) ||
        (value == NULL) ||
        (context == NULL) ||
        (scores == NULL)) {
        goto cleanup;
    }

    if (att1_matmul_f32(query,
                        input,
                        weights->wq,
                        1u,
                        config->model_dim,
                        config->model_dim) != 0) {
        goto cleanup;
    }

    if (att1_matmul_f32(key,
                        input,
                        weights->wk,
                        1u,
                        config->model_dim,
                        config->model_dim) != 0) {
        goto cleanup;
    }

    if (att1_matmul_f32(value,
                        input,
                        weights->wv,
                        1u,
                        config->model_dim,
                        config->model_dim) != 0) {
        goto cleanup;
    }

    if (att1_attention_apply_rope(query, key, config, position) != 0) {
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

        if (att1_softmax_f32(scores, cache->length) != 0) {
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

    if (att1_matmul_f32(output,
                        context,
                        weights->wo,
                        1u,
                        config->model_dim,
                        config->model_dim) != 0) {
        goto cleanup;
    }

    rc = 0;

cleanup:
    free(query);
    free(key);
    free(value);
    free(context);
    free(scores);
    return rc;
}
