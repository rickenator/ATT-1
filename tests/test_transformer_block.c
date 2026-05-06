#include "att1_transformer_block.h"

#include <math.h>
#include <stdio.h>

static int near_f32(float lhs, float rhs)
{
    return fabsf(lhs - rhs) < 0.00001f;
}

static void rmsnorm2(const float *src,
                     const float *weight,
                     float epsilon,
                     float *dst)
{
    const float mean_square = ((src[0] * src[0]) + (src[1] * src[1])) * 0.5f;
    const float scale = 1.0f / sqrtf(mean_square + epsilon);

    dst[0] = src[0] * scale * weight[0];
    dst[1] = src[1] * scale * weight[1];
}

static float silu(float value)
{
    return value / (1.0f + expf(-value));
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
    const float norm_weight[2] = {1.0f, 1.0f};
    const float input[2] = {1.0f, 2.0f};
    const att1_transformer_block_config config = {
        .model_dim = 2u,
        .num_heads = 1u,
        .head_dim = 2u,
        .ffn_dim = 2u,
        .rms_epsilon = 0.000001f,
        .rope_theta = 1.0f
    };
    const att1_transformer_block_weights weights = {
        .attention_norm = norm_weight,
        .ffn_norm = norm_weight,
        .wq = identity2,
        .wk = identity2,
        .wv = identity2,
        .wo = identity2,
        .w_gate = identity2,
        .w_up = identity2,
        .w_down = identity2
    };
    const att1_transformer_block_weights zero_weights = {
        .attention_norm = norm_weight,
        .ffn_norm = norm_weight,
        .wq = zero2,
        .wk = zero2,
        .wv = zero2,
        .wo = zero2,
        .w_gate = zero2,
        .w_up = zero2,
        .w_down = zero2
    };
    float output[2] = {0.0f, 0.0f};
    float output_repeat[2] = {0.0f, 0.0f};
    float residual_output[2] = {0.0f, 0.0f};
    float attention_input[2] = {0.0f, 0.0f};
    float attention_residual[2] = {0.0f, 0.0f};
    float ffn_input[2] = {0.0f, 0.0f};
    float expected[2] = {0.0f, 0.0f};
    att1_kv_cache cache;
    att1_kv_cache repeat_cache;
    att1_kv_cache residual_cache;

    if (att1_kv_cache_init(&cache, 1u, 1u, 2u) != 0) {
        fputs("transformer cache init failed\n", stderr);
        return 1;
    }

    if (att1_transformer_block_forward_f32(output,
                                           &cache,
                                           input,
                                           &weights,
                                           &config,
                                           0u) != 0) {
        fputs("transformer block forward failed\n", stderr);
        att1_kv_cache_free(&cache);
        return 1;
    }

    if (att1_kv_cache_init(&repeat_cache, 1u, 1u, 2u) != 0) {
        fputs("transformer repeat cache init failed\n", stderr);
        att1_kv_cache_free(&cache);
        return 1;
    }

    if (att1_transformer_block_forward_f32(output_repeat,
                                           &repeat_cache,
                                           input,
                                           &weights,
                                           &config,
                                           0u) != 0) {
        fputs("transformer repeat forward failed\n", stderr);
        att1_kv_cache_free(&repeat_cache);
        att1_kv_cache_free(&cache);
        return 1;
    }

    if (!near_f32(output[0], output_repeat[0]) ||
        !near_f32(output[1], output_repeat[1])) {
        fputs("transformer determinism check failed\n", stderr);
        att1_kv_cache_free(&repeat_cache);
        att1_kv_cache_free(&cache);
        return 1;
    }

    att1_kv_cache_free(&repeat_cache);

    rmsnorm2(input, norm_weight, config.rms_epsilon, attention_input);
    attention_residual[0] = input[0] + attention_input[0];
    attention_residual[1] = input[1] + attention_input[1];
    rmsnorm2(attention_residual, norm_weight, config.rms_epsilon, ffn_input);
    expected[0] = attention_residual[0] + (silu(ffn_input[0]) * ffn_input[0]);
    expected[1] = attention_residual[1] + (silu(ffn_input[1]) * ffn_input[1]);

    if (!near_f32(output[0], expected[0]) ||
        !near_f32(output[1], expected[1])) {
        fputs("transformer block output check failed\n", stderr);
        att1_kv_cache_free(&cache);
        return 1;
    }

    if (cache.length != 1u) {
        fputs("transformer block cache length check failed\n", stderr);
        att1_kv_cache_free(&cache);
        return 1;
    }

    if (att1_kv_cache_init(&residual_cache, 1u, 1u, 2u) != 0) {
        fputs("transformer residual cache init failed\n", stderr);
        att1_kv_cache_free(&cache);
        return 1;
    }

    if (att1_transformer_block_forward_f32(residual_output,
                                           &residual_cache,
                                           input,
                                           &zero_weights,
                                           &config,
                                           0u) != 0) {
        fputs("transformer residual forward failed\n", stderr);
        att1_kv_cache_free(&residual_cache);
        att1_kv_cache_free(&cache);
        return 1;
    }

    if (!near_f32(residual_output[0], input[0]) ||
        !near_f32(residual_output[1], input[1])) {
        fputs("transformer zero-weight residual check failed\n", stderr);
        att1_kv_cache_free(&residual_cache);
        att1_kv_cache_free(&cache);
        return 1;
    }

    att1_kv_cache_free(&residual_cache);
    att1_kv_cache_free(&cache);
    puts("transformer_block test passed");
    return 0;
}
