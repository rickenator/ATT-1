#include "att1_kv_cache.h"

#include <math.h>
#include <stdio.h>

static int near_f32(float lhs, float rhs)
{
    return fabsf(lhs - rhs) < 0.00001f;
}

int main(void)
{
    att1_kv_cache cache;
    const float key0[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    const float value0[4] = {101.0f, 102.0f, 103.0f, 104.0f};
    const float key1[4] = {5.0f, 6.0f, 7.0f, 8.0f};
    const float value1[4] = {105.0f, 106.0f, 107.0f, 108.0f};
    const float key2[4] = {9.0f, 10.0f, 11.0f, 12.0f};
    const float value2[4] = {109.0f, 110.0f, 111.0f, 112.0f};
    const float expected_key_range[6] = {
        3.0f, 4.0f,
        7.0f, 8.0f,
        11.0f, 12.0f
    };
    const float expected_value_range[6] = {
        103.0f, 104.0f,
        107.0f, 108.0f,
        111.0f, 112.0f
    };
    float key_range[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float value_range[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    const float *slot = NULL;
    size_t i = 0u;

    if (att1_kv_cache_init(&cache, 3u, 2u, 2u) != 0) {
        fputs("kv cache init failed\n", stderr);
        return 1;
    }

    if (att1_kv_cache_append(&cache, key0, value0) != 0) {
        fputs("kv cache first append failed\n", stderr);
        att1_kv_cache_free(&cache);
        return 1;
    }

    if (cache.length != 1u) {
        fputs("kv cache length check failed\n", stderr);
        att1_kv_cache_free(&cache);
        return 1;
    }

    slot = att1_kv_cache_key(&cache, 0u, 1u);
    if ((slot == NULL) || !near_f32(slot[0], 3.0f) || !near_f32(slot[1], 4.0f)) {
        fputs("kv cache key slot check failed\n", stderr);
        att1_kv_cache_free(&cache);
        return 1;
    }

    slot = att1_kv_cache_value(&cache, 0u, 1u);
    if ((slot == NULL) || !near_f32(slot[0], 103.0f) || !near_f32(slot[1], 104.0f)) {
        fputs("kv cache value slot check failed\n", stderr);
        att1_kv_cache_free(&cache);
        return 1;
    }

    if (att1_kv_cache_append(&cache, key1, value1) != 0) {
        fputs("kv cache second append failed\n", stderr);
        att1_kv_cache_free(&cache);
        return 1;
    }

    if (att1_kv_cache_append(&cache, key2, value2) != 0) {
        fputs("kv cache third append failed\n", stderr);
        att1_kv_cache_free(&cache);
        return 1;
    }

    if (cache.length != 3u) {
        fputs("kv cache final length check failed\n", stderr);
        att1_kv_cache_free(&cache);
        return 1;
    }

    slot = att1_kv_cache_key(&cache, 2u, 0u);
    if ((slot == NULL) || !near_f32(slot[0], 9.0f) || !near_f32(slot[1], 10.0f)) {
        fputs("kv cache position 2 key check failed\n", stderr);
        att1_kv_cache_free(&cache);
        return 1;
    }

    if (att1_kv_cache_copy_range(&cache,
                                 0u,
                                 3u,
                                 1u,
                                 key_range,
                                 value_range) != 0) {
        fputs("kv cache range copy failed\n", stderr);
        att1_kv_cache_free(&cache);
        return 1;
    }

    for (i = 0u; i < 6u; i++) {
        if (!near_f32(key_range[i], expected_key_range[i]) ||
            !near_f32(value_range[i], expected_value_range[i])) {
            fputs("kv cache range order check failed\n", stderr);
            att1_kv_cache_free(&cache);
            return 1;
        }
    }

    if ((att1_kv_cache_key(&cache, 3u, 0u) != NULL) ||
        (att1_kv_cache_value(&cache, 0u, 2u) != NULL)) {
        fputs("kv cache invalid pointer access check failed\n", stderr);
        att1_kv_cache_free(&cache);
        return 1;
    }

    if ((att1_kv_cache_copy_range(&cache, 2u, 2u, 0u, key_range, NULL) == 0) ||
        (att1_kv_cache_copy_range(&cache, 0u, 1u, 2u, key_range, NULL) == 0) ||
        (att1_kv_cache_copy_range(&cache, 0u, 0u, 0u, key_range, NULL) == 0)) {
        fputs("kv cache invalid range check failed\n", stderr);
        att1_kv_cache_free(&cache);
        return 1;
    }

    if (att1_kv_cache_append(&cache, key1, value1) == 0) {
        fputs("kv cache overflow check failed\n", stderr);
        att1_kv_cache_free(&cache);
        return 1;
    }

    att1_kv_cache_free(&cache);
    if ((cache.keys != NULL) || (cache.values != NULL) || (cache.length != 0u)) {
        fputs("kv cache free check failed\n", stderr);
        return 1;
    }

    puts("kv_cache test passed");
    return 0;
}
