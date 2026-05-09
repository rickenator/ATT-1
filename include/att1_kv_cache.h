#ifndef ATT1_KV_CACHE_H
#define ATT1_KV_CACHE_H

#include <stddef.h>

typedef struct att1_kv_cache {
    size_t max_positions;
    size_t num_heads;
    size_t head_dim;
    size_t length;
    float *keys;   /* owned; freed by att1_kv_cache_free */
    float *values; /* owned; freed by att1_kv_cache_free */
    /* must not be shallow-copied: owns keys and values */
} att1_kv_cache;

/*
 * Allocate a zero-initialized local KV cache for Milestone 2 decode.
 *
 * This is a simple contiguous cache, not the future paged KV-MMU. Storage is
 * laid out as [position][head][head_dim] for both keys and values.
 */
int att1_kv_cache_init(att1_kv_cache *cache,
                       size_t max_positions,
                       size_t num_heads,
                       size_t head_dim);

/*
 * Release local cache storage and clear metadata.
 *
 * Passing NULL is allowed.
 */
void att1_kv_cache_free(att1_kv_cache *cache);

/*
 * Append one decoded token's key and value vectors.
 *
 * key and value must contain num_heads * head_dim float32 values. Appends are
 * sequential and increase cache->length by one.
 */
int att1_kv_cache_append(att1_kv_cache *cache,
                         const float *key,
                         const float *value);

/*
 * Return a pointer to a cached key or value vector for one position/head.
 *
 * NULL is returned if indices are outside the populated local cache.
 * The returned pointer borrows into cache storage; it is valid only while
 * the cache lives and position < cache->length.
 */
const float *att1_kv_cache_key(const att1_kv_cache *cache,
                               size_t position,
                               size_t head);
const float *att1_kv_cache_value(const att1_kv_cache *cache,
                                 size_t position,
                                 size_t head);

/*
 * Copy a contiguous position range for one head into caller-owned buffers.
 *
 * start_position and position_count select positions in decode order. out_keys
 * and out_values must each have position_count * head_dim elements. Either
 * output pointer may be NULL to skip copying that stream. Invalid ranges,
 * heads, or empty ranges return -1.
 */
int att1_kv_cache_copy_range(const att1_kv_cache *cache,
                             size_t start_position,
                             size_t position_count,
                             size_t head,
                             float *out_keys,
                             float *out_values);

#endif
