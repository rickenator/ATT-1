#include "att1_backend.h"
#include "att1_kv_cache.h"
#include "att1_transformer_block.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

static int near_cuda_f32(float lhs, float rhs)
{
    return fabsf(lhs - rhs) < 0.001f;
}

static int run_block_pair_step(float *out_cuda,
                               float *out_cpu,
                               att1_kv_cache *cache_cuda,
                               att1_kv_cache *cache_cpu,
                               const float *input,
                               const att1_transformer_block_weights *weights,
                               const att1_transformer_block_config *config,
                               size_t position,
                               att1_backend *cuda_backend,
                               att1_backend *cpu_backend)
{
    if (att1_transformer_block_forward_backend(out_cuda,
                                               cache_cuda,
                                               input,
                                               weights,
                                               config,
                                               position,
                                               cuda_backend) != 0) {
        return -1;
    }

    if (att1_transformer_block_forward_backend(out_cpu,
                                               cache_cpu,
                                               input,
                                               weights,
                                               config,
                                               position,
                                               cpu_backend) != 0) {
        return -1;
    }

    return 0;
}

/* Required: Zero attention/FFN weights preserve residual behavior. */
static int test_zero_weights_residual_behavior(void)
{
    att1_backend *cuda_backend = NULL;
    att1_kv_cache cache;
    float output[4] = {0.0f};
    const float input[4] = {1.0f, -2.0f, 3.0f, -4.0f};
    const float norm_weight[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    const float zero44[16] = {0.0f};
    const att1_transformer_block_config config = {
        .model_dim = 4u,
        .num_heads = 2u,
        .head_dim = 2u,
        .ffn_dim = 4u,
        .rms_epsilon = 0.000001f,
        .rope_theta = 10000.0f
    };
    const att1_transformer_block_weights weights = {
        .attention_norm = norm_weight,
        .ffn_norm = norm_weight,
        .wq = zero44,
        .wk = zero44,
        .wv = zero44,
        .wo = zero44,
        .w_gate = zero44,
        .w_up = zero44,
        .w_down = zero44
    };

    if (att1_backend_cuda_available() == 0) {
        return 0;
    }

    if ((att1_backend_cuda_create(&cuda_backend) != ATT1_OK) ||
        (att1_kv_cache_init(&cache, 2u, 2u, 2u) != 0)) {
        att1_backend_destroy(cuda_backend);
        return -1;
    }

    if (att1_transformer_block_forward_backend(output,
                                               &cache,
                                               input,
                                               &weights,
                                               &config,
                                               0u,
                                               cuda_backend) != 0) {
        att1_kv_cache_free(&cache);
        att1_backend_destroy(cuda_backend);
        return -1;
    }

    if (!near_cuda_f32(output[0], input[0]) ||
        !near_cuda_f32(output[1], input[1]) ||
        !near_cuda_f32(output[2], input[2]) ||
        !near_cuda_f32(output[3], input[3])) {
        att1_kv_cache_free(&cache);
        att1_backend_destroy(cuda_backend);
        return -1;
    }

    att1_kv_cache_free(&cache);
    att1_backend_destroy(cuda_backend);
    return 1;
}

/* Required: CPU vs CUDA block output comparison (tiny deterministic). */
static int test_cpu_vs_cuda_tiny(void)
{
    att1_backend *cuda_backend = NULL;
    att1_backend *cpu_backend = NULL;
    att1_kv_cache cache_cuda;
    att1_kv_cache cache_cpu;
    float out_cuda[2] = {0.0f};
    float out_cpu[2] = {0.0f};
    const float input[2] = {1.5f, -0.5f};
    const float norm_weight[2] = {1.0f, 1.0f};
    const float identity2[4] = {
        1.0f, 0.0f,
        0.0f, 1.0f
    };
    const att1_transformer_block_config config = {
        .model_dim = 2u,
        .num_heads = 1u,
        .head_dim = 2u,
        .ffn_dim = 2u,
        .rms_epsilon = 0.000001f,
        .rope_theta = 10000.0f
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

    if (att1_backend_cuda_available() == 0) {
        return 0;
    }

    if ((att1_backend_cuda_create(&cuda_backend) != ATT1_OK) ||
        (att1_backend_cpu_f32_create(&cpu_backend) != ATT1_OK) ||
        (att1_kv_cache_init(&cache_cuda, 2u, 1u, 2u) != 0) ||
        (att1_kv_cache_init(&cache_cpu, 2u, 1u, 2u) != 0)) {
        att1_kv_cache_free(&cache_cuda);
        att1_kv_cache_free(&cache_cpu);
        att1_backend_destroy(cuda_backend);
        att1_backend_destroy(cpu_backend);
        return -1;
    }

    if (run_block_pair_step(out_cuda,
                            out_cpu,
                            &cache_cuda,
                            &cache_cpu,
                            input,
                            &weights,
                            &config,
                            0u,
                            cuda_backend,
                            cpu_backend) != 0) {
        att1_kv_cache_free(&cache_cuda);
        att1_kv_cache_free(&cache_cpu);
        att1_backend_destroy(cuda_backend);
        att1_backend_destroy(cpu_backend);
        return -1;
    }

    if (!near_cuda_f32(out_cuda[0], out_cpu[0]) ||
        !near_cuda_f32(out_cuda[1], out_cpu[1])) {
        att1_kv_cache_free(&cache_cuda);
        att1_kv_cache_free(&cache_cpu);
        att1_backend_destroy(cuda_backend);
        att1_backend_destroy(cpu_backend);
        return -1;
    }

    att1_kv_cache_free(&cache_cuda);
    att1_kv_cache_free(&cache_cpu);
    att1_backend_destroy(cuda_backend);
    att1_backend_destroy(cpu_backend);
    return 1;
}

/* Required: KV cache position updates match CPU. */
static int test_kv_position_updates_match_cpu(void)
{
    att1_backend *cuda_backend = NULL;
    att1_backend *cpu_backend = NULL;
    att1_kv_cache cache_cuda;
    att1_kv_cache cache_cpu;
    float out_cuda[4] = {0.0f};
    float out_cpu[4] = {0.0f};
    const float input0[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    const float input1[4] = {4.0f, 3.0f, 2.0f, 1.0f};
    const float norm_weight[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    const float identity4[16] = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f
    };
    const att1_transformer_block_config config = {
        .model_dim = 4u,
        .num_heads = 2u,
        .head_dim = 2u,
        .ffn_dim = 4u,
        .rms_epsilon = 0.000001f,
        .rope_theta = 10000.0f
    };
    const att1_transformer_block_weights weights = {
        .attention_norm = norm_weight,
        .ffn_norm = norm_weight,
        .wq = identity4,
        .wk = identity4,
        .wv = identity4,
        .wo = identity4,
        .w_gate = identity4,
        .w_up = identity4,
        .w_down = identity4
    };

    if (att1_backend_cuda_available() == 0) {
        return 0;
    }

    if ((att1_backend_cuda_create(&cuda_backend) != ATT1_OK) ||
        (att1_backend_cpu_f32_create(&cpu_backend) != ATT1_OK) ||
        (att1_kv_cache_init(&cache_cuda, 4u, 2u, 2u) != 0) ||
        (att1_kv_cache_init(&cache_cpu, 4u, 2u, 2u) != 0)) {
        att1_kv_cache_free(&cache_cuda);
        att1_kv_cache_free(&cache_cpu);
        att1_backend_destroy(cuda_backend);
        att1_backend_destroy(cpu_backend);
        return -1;
    }

    if (run_block_pair_step(out_cuda,
                            out_cpu,
                            &cache_cuda,
                            &cache_cpu,
                            input0,
                            &weights,
                            &config,
                            0u,
                            cuda_backend,
                            cpu_backend) != 0) {
        att1_kv_cache_free(&cache_cuda);
        att1_kv_cache_free(&cache_cpu);
        att1_backend_destroy(cuda_backend);
        att1_backend_destroy(cpu_backend);
        return -1;
    }

    if ((cache_cuda.length != cache_cpu.length) || (cache_cuda.length != 1u)) {
        att1_kv_cache_free(&cache_cuda);
        att1_kv_cache_free(&cache_cpu);
        att1_backend_destroy(cuda_backend);
        att1_backend_destroy(cpu_backend);
        return -1;
    }

    if (run_block_pair_step(out_cuda,
                            out_cpu,
                            &cache_cuda,
                            &cache_cpu,
                            input1,
                            &weights,
                            &config,
                            1u,
                            cuda_backend,
                            cpu_backend) != 0) {
        att1_kv_cache_free(&cache_cuda);
        att1_kv_cache_free(&cache_cpu);
        att1_backend_destroy(cuda_backend);
        att1_backend_destroy(cpu_backend);
        return -1;
    }

    if ((cache_cuda.length != cache_cpu.length) || (cache_cuda.length != 2u)) {
        att1_kv_cache_free(&cache_cuda);
        att1_kv_cache_free(&cache_cpu);
        att1_backend_destroy(cuda_backend);
        att1_backend_destroy(cpu_backend);
        return -1;
    }

    att1_kv_cache_free(&cache_cuda);
    att1_kv_cache_free(&cache_cpu);
    att1_backend_destroy(cuda_backend);
    att1_backend_destroy(cpu_backend);
    return 1;
}

/* Required: Multi-head deterministic case (medium deterministic). */
static int test_multi_head_deterministic_medium(void)
{
    att1_backend *cuda_backend = NULL;
    att1_backend *cpu_backend = NULL;
    att1_kv_cache cache_cuda;
    att1_kv_cache cache_cpu;
    float out_cuda[8] = {0.0f};
    float out_cpu[8] = {0.0f};
    float input0[8] = {0.0f};
    float input1[8] = {0.0f};
    float norm_weight[8] = {0.0f};
    float wq[64] = {0.0f};
    float wk[64] = {0.0f};
    float wv[64] = {0.0f};
    float wo[64] = {0.0f};
    float w_gate[64] = {0.0f};
    float w_up[64] = {0.0f};
    float w_down[64] = {0.0f};
    size_t i = 0u;
    const att1_transformer_block_config config = {
        .model_dim = 8u,
        .num_heads = 4u,
        .head_dim = 2u,
        .ffn_dim = 8u,
        .rms_epsilon = 0.000001f,
        .rope_theta = 10000.0f
    };
    att1_transformer_block_weights weights;

    if (att1_backend_cuda_available() == 0) {
        return 0;
    }

    for (i = 0u; i < 8u; i++) {
        input0[i] = (float)(i + 1u) * 0.25f;
        input1[i] = (float)(8u - i) * 0.30f;
        norm_weight[i] = 1.0f;
    }
    for (i = 0u; i < 64u; i++) {
        const float base = (float)((i % 7u) + 1u) * 0.01f;
        wq[i] = base;
        wk[i] = base * 1.1f;
        wv[i] = base * 0.9f;
        wo[i] = base * 1.2f;
        w_gate[i] = base * 0.8f;
        w_up[i] = base * 1.3f;
        w_down[i] = base * 0.7f;
    }

    weights.attention_norm = norm_weight;
    weights.ffn_norm = norm_weight;
    weights.wq = wq;
    weights.wk = wk;
    weights.wv = wv;
    weights.wo = wo;
    weights.w_gate = w_gate;
    weights.w_up = w_up;
    weights.w_down = w_down;

    if ((att1_backend_cuda_create(&cuda_backend) != ATT1_OK) ||
        (att1_backend_cpu_f32_create(&cpu_backend) != ATT1_OK) ||
        (att1_kv_cache_init(&cache_cuda, 4u, 4u, 2u) != 0) ||
        (att1_kv_cache_init(&cache_cpu, 4u, 4u, 2u) != 0)) {
        att1_kv_cache_free(&cache_cuda);
        att1_kv_cache_free(&cache_cpu);
        att1_backend_destroy(cuda_backend);
        att1_backend_destroy(cpu_backend);
        return -1;
    }

    if (run_block_pair_step(out_cuda,
                            out_cpu,
                            &cache_cuda,
                            &cache_cpu,
                            input0,
                            &weights,
                            &config,
                            0u,
                            cuda_backend,
                            cpu_backend) != 0) {
        att1_kv_cache_free(&cache_cuda);
        att1_kv_cache_free(&cache_cpu);
        att1_backend_destroy(cuda_backend);
        att1_backend_destroy(cpu_backend);
        return -1;
    }

    for (i = 0u; i < 8u; i++) {
        if (!near_cuda_f32(out_cuda[i], out_cpu[i])) {
            att1_kv_cache_free(&cache_cuda);
            att1_kv_cache_free(&cache_cpu);
            att1_backend_destroy(cuda_backend);
            att1_backend_destroy(cpu_backend);
            return -1;
        }
    }

    if (run_block_pair_step(out_cuda,
                            out_cpu,
                            &cache_cuda,
                            &cache_cpu,
                            input1,
                            &weights,
                            &config,
                            1u,
                            cuda_backend,
                            cpu_backend) != 0) {
        att1_kv_cache_free(&cache_cuda);
        att1_kv_cache_free(&cache_cpu);
        att1_backend_destroy(cuda_backend);
        att1_backend_destroy(cpu_backend);
        return -1;
    }

    for (i = 0u; i < 8u; i++) {
        if (!near_cuda_f32(out_cuda[i], out_cpu[i])) {
            att1_kv_cache_free(&cache_cuda);
            att1_kv_cache_free(&cache_cpu);
            att1_backend_destroy(cuda_backend);
            att1_backend_destroy(cpu_backend);
            return -1;
        }
    }

    att1_kv_cache_free(&cache_cuda);
    att1_kv_cache_free(&cache_cpu);
    att1_backend_destroy(cuda_backend);
    att1_backend_destroy(cpu_backend);
    return 1;
}

/* Required: CUDA-selected test must not silently use CPU fallback. */
static int test_no_silent_fallback(void)
{
    att1_backend *cuda_backend = NULL;
    att1_kv_cache cache;
    float output[2] = {0.0f};
    const float input[2] = {1.0f, 2.0f};
    const float norm_weight[2] = {1.0f, 1.0f};
    const float identity2[4] = {
        1.0f, 0.0f,
        0.0f, 1.0f
    };
    const att1_transformer_block_config config = {
        .model_dim = 2u,
        .num_heads = 1u,
        .head_dim = 2u,
        .ffn_dim = 2u,
        .rms_epsilon = 0.000001f,
        .rope_theta = 10000.0f
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

    if (att1_backend_cuda_available() == 0) {
        return 0;
    }

    if ((att1_backend_cuda_create(&cuda_backend) != ATT1_OK) ||
        (cuda_backend == NULL) || (cuda_backend->ops == NULL) ||
        (cuda_backend->ops->name == NULL) ||
        (strcmp(cuda_backend->ops->name, "cuda") != 0) ||
        (att1_kv_cache_init(&cache, 2u, 1u, 2u) != 0)) {
        att1_kv_cache_free(&cache);
        att1_backend_destroy(cuda_backend);
        return -1;
    }

    if (att1_transformer_block_forward_backend(output,
                                               &cache,
                                               input,
                                               &weights,
                                               &config,
                                               0u,
                                               cuda_backend) != 0) {
        att1_kv_cache_free(&cache);
        att1_backend_destroy(cuda_backend);
        return -1;
    }

    att1_kv_cache_free(&cache);
    att1_backend_destroy(cuda_backend);
    return 1;
}

int main(void)
{
    int passed = 0;
    int failed = 0;
    int rc = 0;

    if (att1_backend_cuda_available() == 0) {
        puts("cuda_transformer_block test skipped (CUDA unavailable)");
        return 0;
    }

    rc = test_zero_weights_residual_behavior();
    if (rc < 0) {
        fputs("cuda_transformer_block test_zero_weights_residual_behavior failed\n", stderr);
        failed++;
    } else if (rc > 0) {
        passed++;
    }

    rc = test_cpu_vs_cuda_tiny();
    if (rc < 0) {
        fputs("cuda_transformer_block test_cpu_vs_cuda_tiny failed\n", stderr);
        failed++;
    } else if (rc > 0) {
        passed++;
    }

    rc = test_kv_position_updates_match_cpu();
    if (rc < 0) {
        fputs("cuda_transformer_block test_kv_position_updates_match_cpu failed\n", stderr);
        failed++;
    } else if (rc > 0) {
        passed++;
    }

    rc = test_multi_head_deterministic_medium();
    if (rc < 0) {
        fputs("cuda_transformer_block test_multi_head_deterministic_medium failed\n", stderr);
        failed++;
    } else if (rc > 0) {
        passed++;
    }

    rc = test_no_silent_fallback();
    if (rc < 0) {
        fputs("cuda_transformer_block test_no_silent_fallback failed\n", stderr);
        failed++;
    } else if (rc > 0) {
        passed++;
    }

    if (failed > 0) {
        fprintf(stderr,
                "cuda_transformer_block: %d/%d tests passed\n",
                passed,
                passed + failed);
        return 1;
    }

    puts("cuda_transformer_block test passed");
    return 0;
}
