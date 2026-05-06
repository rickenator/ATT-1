#include "att1_kv_cache.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static int att1_mul_size(size_t lhs, size_t rhs, size_t *out)
{
    if ((lhs != 0u) && (rhs > (SIZE_MAX / lhs))) {
        return -1;
    }

    *out = lhs * rhs;
    return 0;
}

static float *att1_kv_cache_slot(float *base,
                                 size_t head_dim,
                                 size_t num_heads,
                                 size_t position,
                                 size_t head)
{
    return &base[((position * num_heads) + head) * head_dim];
}

int att1_kv_cache_init(att1_kv_cache *cache,
                       size_t max_positions,
                       size_t num_heads,
                       size_t head_dim)
{
    size_t vectors = 0u;
    size_t elements = 0u;

    if (cache == NULL) {
        return -1;
    }

    memset(cache, 0, sizeof(*cache));

    if ((max_positions == 0u) || (num_heads == 0u) || (head_dim == 0u)) {
        return -1;
    }

    if (att1_mul_size(max_positions, num_heads, &vectors) != 0) {
        return -1;
    }

    if (att1_mul_size(vectors, head_dim, &elements) != 0) {
        return -1;
    }

    cache->keys = calloc(elements, sizeof(float));
    cache->values = calloc(elements, sizeof(float));
    if ((cache->keys == NULL) || (cache->values == NULL)) {
        att1_kv_cache_free(cache);
        return -1;
    }

    cache->max_positions = max_positions;
    cache->num_heads = num_heads;
    cache->head_dim = head_dim;
    return 0;
}

void att1_kv_cache_free(att1_kv_cache *cache)
{
    if (cache == NULL) {
        return;
    }

    free(cache->keys);
    free(cache->values);
    memset(cache, 0, sizeof(*cache));
}

int att1_kv_cache_append(att1_kv_cache *cache,
                         const float *key,
                         const float *value)
{
    size_t elements = 0u;

    if ((cache == NULL) || (key == NULL) || (value == NULL)) {
        return -1;
    }

    if ((cache->keys == NULL) || (cache->values == NULL)) {
        return -1;
    }

    if (cache->length >= cache->max_positions) {
        return -1;
    }

    if (att1_mul_size(cache->num_heads, cache->head_dim, &elements) != 0) {
        return -1;
    }

    memcpy(att1_kv_cache_slot(cache->keys,
                              cache->head_dim,
                              cache->num_heads,
                              cache->length,
                              0u),
           key,
           elements * sizeof(float));
    memcpy(att1_kv_cache_slot(cache->values,
                              cache->head_dim,
                              cache->num_heads,
                              cache->length,
                              0u),
           value,
           elements * sizeof(float));
    cache->length++;
    return 0;
}

const float *att1_kv_cache_key(const att1_kv_cache *cache,
                               size_t position,
                               size_t head)
{
    if ((cache == NULL) || (cache->keys == NULL)) {
        return NULL;
    }

    if ((position >= cache->length) || (head >= cache->num_heads)) {
        return NULL;
    }

    return att1_kv_cache_slot(cache->keys,
                              cache->head_dim,
                              cache->num_heads,
                              position,
                              head);
}

const float *att1_kv_cache_value(const att1_kv_cache *cache,
                                 size_t position,
                                 size_t head)
{
    if ((cache == NULL) || (cache->values == NULL)) {
        return NULL;
    }

    if ((position >= cache->length) || (head >= cache->num_heads)) {
        return NULL;
    }

    return att1_kv_cache_slot(cache->values,
                              cache->head_dim,
                              cache->num_heads,
                              position,
                              head);
}

int att1_kv_cache_copy_range(const att1_kv_cache *cache,
                             size_t start_position,
                             size_t position_count,
                             size_t head,
                             float *out_keys,
                             float *out_values)
{
    size_t pos = 0u;

    if ((cache == NULL) || ((out_keys == NULL) && (out_values == NULL))) {
        return -1;
    }

    if ((position_count == 0u) || (head >= cache->num_heads)) {
        return -1;
    }

    if (start_position > cache->length) {
        return -1;
    }

    if (position_count > (cache->length - start_position)) {
        return -1;
    }

    for (pos = 0u; pos < position_count; pos++) {
        const size_t cache_position = start_position + pos;
        const float *key = att1_kv_cache_key(cache, cache_position, head);
        const float *value = att1_kv_cache_value(cache, cache_position, head);

        if ((key == NULL) || (value == NULL)) {
            return -1;
        }

        if (out_keys != NULL) {
            memcpy(&out_keys[pos * cache->head_dim],
                   key,
                   cache->head_dim * sizeof(float));
        }

        if (out_values != NULL) {
            memcpy(&out_values[pos * cache->head_dim],
                   value,
                   cache->head_dim * sizeof(float));
        }
    }

    return 0;
}
