#include "att1_attention.h"
#include "att1_backend.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

static int near_f32(float lhs, float rhs)
{
    return fabsf(lhs - rhs) < 0.00001f;
}

static int near_cuda_f32(float lhs, float rhs)
{
    return fabsf(lhs - rhs) < 0.001f;
}

/*
 * Test 1: Position 0 can only attend to token 0 (causal mask).
 */
static int test_position_0_causal(void)
{
    att1_backend *backend = NULL;
    att1_backend *cpu_backend = NULL;
    att1_kv_cache cache;
    att1_kv_cache cpu_cache;
    float output_cuda[2] = {0.0f, 0.0f};
    float output_cpu[2] = {0.0f, 0.0f};

    const float zero2[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const float identity2[4] = {1.0f, 0.0f, 0.0f, 1.0f};
    const att1_attention_config config = {
        .model_dim = 2u, .num_heads = 1u, .head_dim = 2u, .rope_theta = 1.0f};
    const att1_attention_weights weights = {
        .wq = zero2, .wk = identity2, .wv = identity2, .wo = identity2};
    const float input0[2] = {10.0f, 0.0f};

    if (att1_backend_cuda_available() == 0) {
        return 0; /* skip */
    }

    if (att1_backend_cuda_create(&backend) != ATT1_OK) {
        return -1;
    }
    if (att1_backend_cpu_f32_create(&cpu_backend) != ATT1_OK) {
        att1_backend_destroy(backend);
        return -1;
    }

    if (att1_kv_cache_init(&cache, 3u, 1u, 2u) != 0) {
        att1_backend_destroy(backend);
        att1_backend_destroy(cpu_backend);
        return -1;
    }
    if (att1_kv_cache_init(&cpu_cache, 3u, 1u, 2u) != 0) {
        att1_kv_cache_free(&cache);
        att1_backend_destroy(backend);
        att1_backend_destroy(cpu_backend);
        return -1;
    }

    if (att1_attention_forward_backend(output_cuda, &cache, input0, &weights,
                                       &config, 0u, backend) != 0) {
        att1_kv_cache_free(&cache);
        att1_kv_cache_free(&cpu_cache);
        att1_backend_destroy(backend);
        att1_backend_destroy(cpu_backend);
        return -1;
    }

    if (att1_attention_forward_backend(output_cpu, &cpu_cache, input0, &weights,
                                       &config, 0u, cpu_backend) != 0) {
        att1_kv_cache_free(&cache);
        att1_kv_cache_free(&cpu_cache);
        att1_backend_destroy(backend);
        att1_backend_destroy(cpu_backend);
        return -1;
    }

    /* Position 0 attends only to token 0. */
    if (!near_f32(output_cpu[0], 10.0f) || !near_f32(output_cpu[1], 0.0f)) {
        att1_kv_cache_free(&cache);
        att1_kv_cache_free(&cpu_cache);
        att1_backend_destroy(backend);
        att1_backend_destroy(cpu_backend);
        return -1;
    }

    /* CUDA output within tolerance of CPU. */
    if (!near_cuda_f32(output_cuda[0], output_cpu[0]) ||
        !near_cuda_f32(output_cuda[1], output_cpu[1])) {
        att1_kv_cache_free(&cache);
        att1_kv_cache_free(&cpu_cache);
        att1_backend_destroy(backend);
        att1_backend_destroy(cpu_backend);
        return -1;
    }

    att1_kv_cache_free(&cache);
    att1_kv_cache_free(&cpu_cache);
    att1_backend_destroy(backend);
    att1_backend_destroy(cpu_backend);
    return 1; /* pass */
}

/*
 * Test 2: Position N attends only to 0..N (causal mask).
 */
static int test_position_n_causal(void)
{
    att1_backend *backend = NULL;
    att1_backend *cpu_backend = NULL;
    att1_kv_cache cache;
    att1_kv_cache cpu_cache;
    float output_cuda[2] = {0.0f, 0.0f};
    float output_cpu[2] = {0.0f, 0.0f};

    const float zero2[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const float identity2[4] = {1.0f, 0.0f, 0.0f, 1.0f};
    const att1_attention_config config = {
        .model_dim = 2u, .num_heads = 1u, .head_dim = 2u, .rope_theta = 1.0f};
    const att1_attention_weights weights = {
        .wq = zero2, .wk = identity2, .wv = identity2, .wo = identity2};
    const float input0[2] = {10.0f, 0.0f};
    const float input1[2] = {0.0f, 20.0f};
    const float input2[2] = {30.0f, 60.0f};

    if (att1_backend_cuda_available() == 0) {
        return 0; /* skip */
    }

    if (att1_backend_cuda_create(&backend) != ATT1_OK) {
        return -1;
    }
    if (att1_backend_cpu_f32_create(&cpu_backend) != ATT1_OK) {
        att1_backend_destroy(backend);
        return -1;
    }

    if (att1_kv_cache_init(&cache, 3u, 1u, 2u) != 0) {
        att1_backend_destroy(backend);
        att1_backend_destroy(cpu_backend);
        return -1;
    }
    if (att1_kv_cache_init(&cpu_cache, 3u, 1u, 2u) != 0) {
        att1_kv_cache_free(&cache);
        att1_backend_destroy(backend);
        att1_backend_destroy(cpu_backend);
        return -1;
    }

    /* Populate cache for positions 0 and 1. */
    if (att1_attention_forward_backend(output_cuda, &cache, input0, &weights,
                                       &config, 0u, backend) != 0) {
        att1_kv_cache_free(&cache);
        att1_kv_cache_free(&cpu_cache);
        att1_backend_destroy(backend);
        att1_backend_destroy(cpu_backend);
        return -1;
    }
    if (att1_attention_forward_backend(output_cpu, &cpu_cache, input0, &weights,
                                       &config, 0u, cpu_backend) != 0) {
        att1_kv_cache_free(&cache);
        att1_kv_cache_free(&cpu_cache);
        att1_backend_destroy(backend);
        att1_backend_destroy(cpu_backend);
        return -1;
    }

    if (att1_attention_forward_backend(output_cuda, &cache, input1, &weights,
                                       &config, 1u, backend) != 0) {
        att1_kv_cache_free(&cache);
        att1_kv_cache_free(&cpu_cache);
        att1_backend_destroy(backend);
        att1_backend_destroy(cpu_backend);
        return -1;
    }
    if (att1_attention_forward_backend(output_cpu, &cpu_cache, input1, &weights,
                                       &config, 1u, cpu_backend) != 0) {
        att1_kv_cache_free(&cache);
        att1_kv_cache_free(&cpu_cache);
        att1_backend_destroy(backend);
        att1_backend_destroy(cpu_backend);
        return -1;
    }

    /* Position 2 attends to 0..2. */
    if (att1_attention_forward_backend(output_cuda, &cache, input2, &weights,
                                       &config, 2u, backend) != 0) {
        att1_kv_cache_free(&cache);
        att1_kv_cache_free(&cpu_cache);
        att1_backend_destroy(backend);
        att1_backend_destroy(cpu_backend);
        return -1;
    }
    if (att1_attention_forward_backend(output_cpu, &cpu_cache, input2, &weights,
                                       &config, 2u, cpu_backend) != 0) {
        att1_kv_cache_free(&cache);
        att1_kv_cache_free(&cpu_cache);
        att1_backend_destroy(backend);
        att1_backend_destroy(cpu_backend);
        return -1;
    }

    /* CPU reference: position 2 averages 0..2. */
    if (!near_f32(output_cpu[0], 40.0f / 3.0f) ||
        !near_f32(output_cpu[1], 80.0f / 3.0f)) {
        att1_kv_cache_free(&cache);
        att1_kv_cache_free(&cpu_cache);
        att1_backend_destroy(backend);
        att1_backend_destroy(cpu_backend);
        return -1;
    }

    /* CUDA output within tolerance of CPU. */
    if (!near_cuda_f32(output_cuda[0], output_cpu[0]) ||
        !near_cuda_f32(output_cuda[1], output_cpu[1])) {
        att1_kv_cache_free(&cache);
        att1_kv_cache_free(&cpu_cache);
        att1_backend_destroy(backend);
        att1_backend_destroy(cpu_backend);
        return -1;
    }

    att1_kv_cache_free(&cache);
    att1_kv_cache_free(&cpu_cache);
    att1_backend_destroy(backend);
    att1_backend_destroy(cpu_backend);
    return 1; /* pass */
}

/*
 * Test 3: Future KV with huge value does not affect output (causal mask enforced).
 */
static int test_future_kv_no_affect(void)
{
    att1_backend *backend = NULL;
    att1_kv_cache cache;
    float output_pos0_first[2] = {0.0f, 0.0f};
    float output_pos1[2] = {0.0f, 0.0f};
    float output_pos0_second[2] = {0.0f, 0.0f};

    const float zero2[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const float identity2[4] = {1.0f, 0.0f, 0.0f, 1.0f};
    const att1_attention_config config = {
        .model_dim = 2u, .num_heads = 1u, .head_dim = 2u, .rope_theta = 1.0f};
    const att1_attention_weights weights = {
        .wq = zero2, .wk = identity2, .wv = identity2, .wo = identity2};
    const float input0[2] = {10.0f, 0.0f};
    const float input1[2] = {0.0f, 20.0f};

    if (att1_backend_cuda_available() == 0) {
        return 0; /* skip */
    }

    if (att1_backend_cuda_create(&backend) != ATT1_OK) {
        return -1;
    }

    if (att1_kv_cache_init(&cache, 10u, 1u, 2u) != 0) {
        att1_backend_destroy(backend);
        return -1;
    }

    /* Get output at position 0. */
    if (att1_attention_forward_backend(output_pos0_first, &cache, input0, &weights,
                                       &config, 0u, backend) != 0) {
        att1_kv_cache_free(&cache);
        att1_backend_destroy(backend);
        return -1;
    }

    /* Append input1 at position 1 (normal flow). */
    if (att1_attention_forward_backend(output_pos1, &cache, input1, &weights,
                                       &config, 1u, backend) != 0) {
        att1_kv_cache_free(&cache);
        att1_backend_destroy(backend);
        return -1;
    }

    /* Reset cache and re-run position 0 (should get same output). */
    att1_kv_cache_free(&cache);
    if (att1_kv_cache_init(&cache, 10u, 1u, 2u) != 0) {
        att1_backend_destroy(backend);
        return -1;
    }

    if (att1_attention_forward_backend(output_pos0_second, &cache, input0, &weights,
                                       &config, 0u, backend) != 0) {
        att1_kv_cache_free(&cache);
        att1_backend_destroy(backend);
        return -1;
    }

    /* Position 0 output should be identical regardless of future tokens. */
    if (!near_cuda_f32(output_pos0_first[0], output_pos0_second[0]) ||
        !near_cuda_f32(output_pos0_first[1], output_pos0_second[1])) {
        att1_kv_cache_free(&cache);
        att1_backend_destroy(backend);
        return -1;
    }

    att1_kv_cache_free(&cache);
    att1_backend_destroy(backend);
    return 1; /* pass */
}

/*
 * Test 4: Softmax numerical stability with large attention scores.
 */
static int test_softmax_stability(void)
{
    att1_backend *backend = NULL;
    att1_backend *cpu_backend = NULL;
    att1_kv_cache cache;
    att1_kv_cache cpu_cache;
    float output_cuda[8] = {0.0f};
    float output_cpu[8] = {0.0f};

    /* 4 heads, 2 dims per head, large score values. */
    const float wq[64] = {0.0f};
    const float wk[64] = {0.0f};
    const float wv[64] = {0.0f};
    const float wo[64] = {0.0f};
    const att1_attention_config config = {
        .model_dim = 8u, .num_heads = 4u, .head_dim = 2u, .rope_theta = 1.0f};
    const att1_attention_weights weights = {
        .wq = wq, .wk = wk, .wv = wv, .wo = wo};
    const float input[8] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};

    if (att1_backend_cuda_available() == 0) {
        return 0; /* skip */
    }

    if (att1_backend_cuda_create(&backend) != ATT1_OK) {
        return -1;
    }
    if (att1_backend_cpu_f32_create(&cpu_backend) != ATT1_OK) {
        att1_backend_destroy(backend);
        return -1;
    }

    if (att1_kv_cache_init(&cache, 5u, 4u, 2u) != 0) {
        att1_backend_destroy(backend);
        att1_backend_destroy(cpu_backend);
        return -1;
    }
    if (att1_kv_cache_init(&cpu_cache, 5u, 4u, 2u) != 0) {
        att1_kv_cache_free(&cache);
        att1_backend_destroy(backend);
        att1_backend_destroy(cpu_backend);
        return -1;
    }

    /* Multiple positions to accumulate scores. */
    for (int i = 0; i < 3; i++) {
        if (att1_attention_forward_backend(output_cuda, &cache, input, &weights,
                                           &config, (size_t)i, backend) != 0) {
            att1_kv_cache_free(&cache);
            att1_kv_cache_free(&cpu_cache);
            att1_backend_destroy(backend);
            att1_backend_destroy(cpu_backend);
            return -1;
        }
        if (att1_attention_forward_backend(output_cpu, &cpu_cache, input,
                                           &weights, &config, (size_t)i,
                                           cpu_backend) != 0) {
            att1_kv_cache_free(&cache);
            att1_kv_cache_free(&cpu_cache);
            att1_backend_destroy(backend);
            att1_backend_destroy(cpu_backend);
            return -1;
        }
    }

    /* Outputs should be close (numerically stable softmax). */
    for (int i = 0; i < 8; i++) {
        if (!near_cuda_f32(output_cuda[i], output_cpu[i])) {
            att1_kv_cache_free(&cache);
            att1_kv_cache_free(&cpu_cache);
            att1_backend_destroy(backend);
            att1_backend_destroy(cpu_backend);
            return -1;
        }
    }

    att1_kv_cache_free(&cache);
    att1_kv_cache_free(&cpu_cache);
    att1_backend_destroy(backend);
    att1_backend_destroy(cpu_backend);
    return 1; /* pass */
}

/*
 * Test 5: Multi-head deterministic CPU vs CUDA comparison.
 */
static int test_multihead_deterministic(void)
{
    att1_backend *backend = NULL;
    att1_backend *cpu_backend = NULL;
    att1_kv_cache cache;
    att1_kv_cache cpu_cache;
    float output_cuda[4] = {0.0f};
    float output_cpu[4] = {0.0f};

    /* 2 heads, 2 dims per head. */
    const float wq[16] = {0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f,
                          0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f};
    const float wk[16] = {0.3f, 0.3f, 0.3f, 0.3f, 0.3f, 0.3f, 0.3f, 0.3f,
                          0.3f, 0.3f, 0.3f, 0.3f, 0.3f, 0.3f, 0.3f, 0.3f};
    const float wv[16] = {0.7f, 0.7f, 0.7f, 0.7f, 0.7f, 0.7f, 0.7f, 0.7f,
                          0.7f, 0.7f, 0.7f, 0.7f, 0.7f, 0.7f, 0.7f, 0.7f};
    const float wo[16] = {2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f,
                          2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f};
    const att1_attention_config config = {
        .model_dim = 4u, .num_heads = 2u, .head_dim = 2u, .rope_theta = 2.0f};
    const att1_attention_weights weights = {
        .wq = wq, .wk = wk, .wv = wv, .wo = wo};
    const float input[4] = {1.1f, 2.2f, 3.3f, 4.4f};

    if (att1_backend_cuda_available() == 0) {
        return 0; /* skip */
    }

    if (att1_backend_cuda_create(&backend) != ATT1_OK) {
        return -1;
    }
    if (att1_backend_cpu_f32_create(&cpu_backend) != ATT1_OK) {
        att1_backend_destroy(backend);
        return -1;
    }

    if (att1_kv_cache_init(&cache, 2u, 2u, 2u) != 0) {
        att1_backend_destroy(backend);
        att1_backend_destroy(cpu_backend);
        return -1;
    }
    if (att1_kv_cache_init(&cpu_cache, 2u, 2u, 2u) != 0) {
        att1_kv_cache_free(&cache);
        att1_backend_destroy(backend);
        att1_backend_destroy(cpu_backend);
        return -1;
    }

    if (att1_attention_forward_backend(output_cuda, &cache, input, &weights,
                                       &config, 0u, backend) != 0) {
        att1_kv_cache_free(&cache);
        att1_kv_cache_free(&cpu_cache);
        att1_backend_destroy(backend);
        att1_backend_destroy(cpu_backend);
        return -1;
    }
    if (att1_attention_forward_backend(output_cpu, &cpu_cache, input, &weights,
                                       &config, 0u, cpu_backend) != 0) {
        att1_kv_cache_free(&cache);
        att1_kv_cache_free(&cpu_cache);
        att1_backend_destroy(backend);
        att1_backend_destroy(cpu_backend);
        return -1;
    }

    /* CUDA output within tolerance of CPU. */
    for (int i = 0; i < 4; i++) {
        if (!near_cuda_f32(output_cuda[i], output_cpu[i])) {
            att1_kv_cache_free(&cache);
            att1_kv_cache_free(&cpu_cache);
            att1_backend_destroy(backend);
            att1_backend_destroy(cpu_backend);
            return -1;
        }
    }

    att1_kv_cache_free(&cache);
    att1_kv_cache_free(&cpu_cache);
    att1_backend_destroy(backend);
    att1_backend_destroy(cpu_backend);
    return 1; /* pass */
}

/*
 * Test 6: Empty/invalid KV range fails cleanly.
 */
static int test_invalid_kv_range(void)
{
    att1_backend *backend = NULL;
    att1_kv_cache cache;
    float output[2] = {0.0f, 0.0f};

    const float zero2[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const float identity2[4] = {1.0f, 0.0f, 0.0f, 1.0f};
    const att1_attention_config config = {
        .model_dim = 2u, .num_heads = 1u, .head_dim = 2u, .rope_theta = 1.0f};
    const att1_attention_weights weights = {
        .wq = zero2, .wk = identity2, .wv = identity2, .wo = identity2};
    const float input[2] = {10.0f, 0.0f};

    if (att1_backend_cuda_available() == 0) {
        return 0; /* skip */
    }

    if (att1_backend_cuda_create(&backend) != ATT1_OK) {
        return -1;
    }

    if (att1_kv_cache_init(&cache, 5u, 1u, 2u) != 0) {
        att1_backend_destroy(backend);
        return -1;
    }

    /* Mismatched position should fail. */
    if (att1_attention_forward_backend(output, &cache, input, &weights, &config,
                                       5u, backend) == 0) {
        att1_kv_cache_free(&cache);
        att1_backend_destroy(backend);
        return -1; /* should have failed */
    }

    att1_kv_cache_free(&cache);
    att1_backend_destroy(backend);
    return 1; /* pass */
}

/*
 * Test 7: CUDA backend selected without silent CPU fallback.
 */
static int test_no_silent_fallback(void)
{
    att1_backend *backend = NULL;
    att1_kv_cache cache;
    float output[2] = {0.0f, 0.0f};

    const float zero2[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const float identity2[4] = {1.0f, 0.0f, 0.0f, 1.0f};
    const att1_attention_config config = {
        .model_dim = 2u, .num_heads = 1u, .head_dim = 2u, .rope_theta = 1.0f};
    const att1_attention_weights weights = {
        .wq = zero2, .wk = identity2, .wv = identity2, .wo = identity2};
    const float input[2] = {10.0f, 0.0f};

    if (att1_backend_cuda_available() == 0) {
        return 0; /* skip */
    }

    if (att1_backend_cuda_create(&backend) != ATT1_OK) {
        return -1;
    }

    if (att1_kv_cache_init(&cache, 1u, 1u, 2u) != 0) {
        att1_backend_destroy(backend);
        return -1;
    }

    /* Verify backend name is cuda. */
    if (backend->ops == NULL || backend->ops->name == NULL ||
        strcmp(backend->ops->name, "cuda") != 0) {
        att1_kv_cache_free(&cache);
        att1_backend_destroy(backend);
        return -1;
    }

    /* Attention should work and use CUDA backend operations. */
    if (att1_attention_forward_backend(output, &cache, input, &weights, &config,
                                       0u, backend) != 0) {
        att1_kv_cache_free(&cache);
        att1_backend_destroy(backend);
        return -1;
    }

    att1_kv_cache_free(&cache);
    att1_backend_destroy(backend);
    return 1; /* pass */
}

int main(void)
{
    int passed = 0;
    int failed = 0;

    if (att1_backend_cuda_available() == 0) {
        puts("cuda_attention test skipped (CUDA unavailable)");
        return 0;
    }

    int test1 = test_position_0_causal();
    if (test1 < 0) {
        fputs("cuda_attention test_position_0_causal failed\n", stderr);
        failed++;
    } else if (test1 > 0) {
        passed++;
    }

    int test2 = test_position_n_causal();
    if (test2 < 0) {
        fputs("cuda_attention test_position_n_causal failed\n", stderr);
        failed++;
    } else if (test2 > 0) {
        passed++;
    }

    int test3 = test_future_kv_no_affect();
    if (test3 < 0) {
        fputs("cuda_attention test_future_kv_no_affect failed\n", stderr);
        failed++;
    } else if (test3 > 0) {
        passed++;
    }

    int test4 = test_softmax_stability();
    if (test4 < 0) {
        fputs("cuda_attention test_softmax_stability failed\n", stderr);
        failed++;
    } else if (test4 > 0) {
        passed++;
    }

    int test5 = test_multihead_deterministic();
    if (test5 < 0) {
        fputs("cuda_attention test_multihead_deterministic failed\n", stderr);
        failed++;
    } else if (test5 > 0) {
        passed++;
    }

    int test6 = test_invalid_kv_range();
    if (test6 < 0) {
        fputs("cuda_attention test_invalid_kv_range failed\n", stderr);
        failed++;
    } else if (test6 > 0) {
        passed++;
    }

    int test7 = test_no_silent_fallback();
    if (test7 < 0) {
        fputs("cuda_attention test_no_silent_fallback failed\n", stderr);
        failed++;
    } else if (test7 > 0) {
        passed++;
    }

    if (failed > 0) {
        fprintf(stderr, "cuda_attention: %d/%d tests passed\n", passed,
                passed + failed);
        return 1;
    }

    puts("cuda_attention test passed");
    return 0;
}
