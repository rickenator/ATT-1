#include "att1_attention.h"

#include <math.h>
#include <stdio.h>

static int near_f32(float lhs, float rhs)
{
    return fabsf(lhs - rhs) < 0.00001f;
}

int main(void)
{
    const float zero2[4] = {
        0.0f, 0.0f,
        0.0f, 0.0f
    };
    const float identity2[4] = {
        1.0f, 0.0f,
        0.0f, 1.0f
    };
    const att1_attention_config config = {
        .model_dim = 2u,
        .num_heads = 1u,
        .head_dim = 2u,
        .rope_theta = 1.0f
    };
    const att1_attention_weights weights = {
        .wq = zero2,
        .wk = identity2,
        .wv = identity2,
        .wo = identity2
    };
    const float input0[2] = {10.0f, 0.0f};
    const float input1[2] = {0.0f, 20.0f};
    const float input2[2] = {30.0f, 60.0f};
    float output[2] = {0.0f, 0.0f};
    att1_kv_cache cache;

    if (att1_kv_cache_init(&cache, 3u, 1u, 2u) != 0) {
        fputs("attention cache init failed\n", stderr);
        return 1;
    }

    if (att1_attention_forward_f32(output,
                                   &cache,
                                   input0,
                                   &weights,
                                   &config,
                                   0u) != 0) {
        fputs("attention first step failed\n", stderr);
        att1_kv_cache_free(&cache);
        return 1;
    }

    /* Position 0 can only see token 0. */
    if (!near_f32(output[0], 10.0f) || !near_f32(output[1], 0.0f)) {
        fputs("attention position 0 causal range check failed\n", stderr);
        att1_kv_cache_free(&cache);
        return 1;
    }

    if (att1_attention_forward_f32(output,
                                   &cache,
                                   input1,
                                   &weights,
                                   &config,
                                   1u) != 0) {
        fputs("attention second step failed\n", stderr);
        att1_kv_cache_free(&cache);
        return 1;
    }

    /* With zero Q scores, position 1 uniformly averages tokens 0..1. */
    if (!near_f32(output[0], 5.0f) || !near_f32(output[1], 10.0f)) {
        fputs("attention position 1 causal range check failed\n", stderr);
        att1_kv_cache_free(&cache);
        return 1;
    }

    if (att1_attention_forward_f32(output,
                                   &cache,
                                   input2,
                                   &weights,
                                   &config,
                                   2u) != 0) {
        fputs("attention third step failed\n", stderr);
        att1_kv_cache_free(&cache);
        return 1;
    }

    /* Position N attends exactly over 0..N; here N=2. */
    if (!near_f32(output[0], 40.0f / 3.0f) ||
        !near_f32(output[1], 80.0f / 3.0f)) {
        fputs("attention position N causal range check failed\n", stderr);
        att1_kv_cache_free(&cache);
        return 1;
    }

    if (cache.length != 3u) {
        fputs("attention cache final length check failed\n", stderr);
        att1_kv_cache_free(&cache);
        return 1;
    }

    output[0] = -99.0f;
    output[1] = -99.0f;
    if (att1_attention_forward_f32(output,
                                   &cache,
                                   input1,
                                   &weights,
                                   &config,
                                   1u) == 0) {
        fputs("attention future-cache rejection failed\n", stderr);
        att1_kv_cache_free(&cache);
        return 1;
    }

    if (!near_f32(output[0], -99.0f) || !near_f32(output[1], -99.0f)) {
        fputs("attention rejected decode modified output\n", stderr);
        att1_kv_cache_free(&cache);
        return 1;
    }

    att1_kv_cache_free(&cache);
    puts("attention test passed");
    return 0;
}
