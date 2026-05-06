#include "att1_backend.h"
#include "att1_attention.h"
#include "att1_math.h"

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

static int check_cpu_f32_backend(void)
{
    const float lhs[6] = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f
    };
    const float rhs[6] = {
        7.0f, 8.0f,
        9.0f, 10.0f,
        11.0f, 12.0f
    };
    const float norm_src[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    const float norm_weight[4] = {1.0f, 0.5f, 2.0f, 1.5f};
    const float swiglu_gate[3] = {-1.0f, 0.0f, 2.0f};
    const float swiglu_value[3] = {3.0f, 4.0f, -2.0f};
    float backend_out[6] = {0.0f};
    float reference_out[6] = {0.0f};
    float backend_norm[4] = {0.0f};
    float reference_norm[4] = {0.0f};
    float backend_softmax[3] = {1.0f, 2.0f, 3.0f};
    float reference_softmax[3] = {1.0f, 2.0f, 3.0f};
    float backend_rope[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float reference_rope[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float backend_swiglu[3] = {0.0f};
    float reference_swiglu[3] = {0.0f};
    att1_backend *backend = NULL;
    void *scratch = NULL;
    size_t i = 0u;

    if (att1_backend_cpu_f32_create(&backend) != ATT1_OK) {
        fputs("cpu f32 backend creation failed\n", stderr);
        return -1;
    }

    if ((backend->ops == NULL) ||
        (backend->ops->name == NULL) ||
        (strcmp(backend->ops->name, "cpu-f32") != 0)) {
        fputs("cpu f32 backend metadata check failed\n", stderr);
        att1_backend_destroy(backend);
        return -1;
    }

    scratch = backend->ops->alloc(backend, 16u);
    if (scratch == NULL) {
        fputs("cpu f32 backend alloc hook failed\n", stderr);
        att1_backend_destroy(backend);
        return -1;
    }
    backend->ops->free(backend, scratch);

    if ((backend->ops->sync == NULL) ||
        (backend->ops->sync(backend) != 0)) {
        fputs("cpu f32 backend sync hook failed\n", stderr);
        att1_backend_destroy(backend);
        return -1;
    }

    if ((backend->ops->matmul_f32(backend,
                                  backend_out,
                                  lhs,
                                  rhs,
                                  2u,
                                  2u,
                                  3u) != 0) ||
        (att1_matmul_f32(reference_out, lhs, rhs, 2u, 2u, 3u) != 0)) {
        fputs("cpu f32 backend matmul call failed\n", stderr);
        att1_backend_destroy(backend);
        return -1;
    }

    if ((backend->ops->rmsnorm_f32(backend,
                                   backend_norm,
                                   norm_src,
                                   norm_weight,
                                   4u,
                                   0.000001f) != 0) ||
        (att1_rmsnorm_f32(reference_norm,
                          norm_src,
                          norm_weight,
                          4u,
                          0.000001f) != 0)) {
        fputs("cpu f32 backend rmsnorm call failed\n", stderr);
        att1_backend_destroy(backend);
        return -1;
    }

    if ((backend->ops->softmax_f32(backend, backend_softmax, 3u) != 0) ||
        (att1_softmax_f32(reference_softmax, 3u) != 0)) {
        fputs("cpu f32 backend softmax call failed\n", stderr);
        att1_backend_destroy(backend);
        return -1;
    }

    if ((backend->ops->rope_f32(backend, backend_rope, 4u, 1u, 10000.0f) != 0) ||
        (att1_rope_f32(reference_rope, 4u, 1u, 10000.0f) != 0)) {
        fputs("cpu f32 backend rope call failed\n", stderr);
        att1_backend_destroy(backend);
        return -1;
    }

    if ((backend->ops->ffn_swiglu_f32(backend,
                                      backend_swiglu,
                                      swiglu_gate,
                                      swiglu_value,
                                      3u) != 0) ||
        (att1_swiglu_f32(reference_swiglu, swiglu_gate, swiglu_value, 3u) != 0)) {
        fputs("cpu f32 backend swiglu call failed\n", stderr);
        att1_backend_destroy(backend);
        return -1;
    }

    for (i = 0u; i < 6u; i++) {
        if (!near_f32(backend_out[i], reference_out[i])) {
            fputs("cpu f32 backend matmul value mismatch\n", stderr);
            att1_backend_destroy(backend);
            return -1;
        }
    }

    for (i = 0u; i < 4u; i++) {
        if (!near_f32(backend_norm[i], reference_norm[i]) ||
            !near_f32(backend_rope[i], reference_rope[i])) {
            fputs("cpu f32 backend norm/rope value mismatch\n", stderr);
            att1_backend_destroy(backend);
            return -1;
        }
    }

    for (i = 0u; i < 3u; i++) {
        if (!near_f32(backend_softmax[i], reference_softmax[i]) ||
            !near_f32(backend_swiglu[i], reference_swiglu[i])) {
            fputs("cpu f32 backend softmax/swiglu value mismatch\n", stderr);
            att1_backend_destroy(backend);
            return -1;
        }
    }

    att1_backend_destroy(backend);
    return 0;
}

static int check_cpu_q8_backend(void)
{
    const float lhs[3] = {1.0f, -2.0f, 0.5f};
    const float weights[6] = {
        0.75f, -1.25f, 0.50f,
        -0.50f, 0.25f, 1.50f
    };
    float backend_out[2] = {0.0f, 0.0f};
    float reference_out[2] = {0.0f, 0.0f};
    att1_backend *backend = NULL;
    att1_q8_matrix q8;
    size_t i = 0u;

    memset(&q8, 0, sizeof(q8));
    if ((att1_backend_cpu_q8_create(&backend) != ATT1_OK) ||
        (att1_quantize_q8_per_row(&q8, weights, 2u, 3u) != 0)) {
        fputs("cpu q8 backend setup failed\n", stderr);
        att1_backend_destroy(backend);
        return -1;
    }

    if ((backend->ops == NULL) ||
        (backend->ops->matmul_q8xf32 == NULL) ||
        (strcmp(backend->ops->name, "cpu-q8") != 0)) {
        fputs("cpu q8 backend metadata check failed\n", stderr);
        att1_q8_matrix_free(&q8);
        att1_backend_destroy(backend);
        return -1;
    }

    if ((backend->ops->matmul_q8xf32(backend,
                                     backend_out,
                                     lhs,
                                     1u,
                                     3u,
                                     &q8) != 0) ||
        (att1_matmul_q8xf32(reference_out, lhs, 1u, 3u, &q8) != 0)) {
        fputs("cpu q8 backend q8 matmul call failed\n", stderr);
        att1_q8_matrix_free(&q8);
        att1_backend_destroy(backend);
        return -1;
    }

    for (i = 0u; i < 2u; i++) {
        if (!near_f32(backend_out[i], reference_out[i])) {
            fputs("cpu q8 backend q8 matmul value mismatch\n", stderr);
            att1_q8_matrix_free(&q8);
            att1_backend_destroy(backend);
            return -1;
        }
    }

    att1_q8_matrix_free(&q8);
    att1_backend_destroy(backend);
    return 0;
}

static int check_cuda_backend_skeleton(void)
{
    att1_backend *backend = NULL;
    att1_status_t status = ATT1_OK;
    const float norm_src[4] = {1.0f, -2.0f, 3.0f, -4.0f};
    const float norm_weight[4] = {1.0f, 0.5f, 1.5f, 2.0f};
    const float swiglu_gate[4] = {-2.0f, -0.5f, 1.0f, 3.0f};
    const float swiglu_value[4] = {1.5f, -2.0f, 0.5f, 4.0f};
    float backend_rope[6] = {1.0f, 0.0f, 0.0f, 1.0f, 2.0f, 3.0f};
    float reference_rope[6] = {1.0f, 0.0f, 0.0f, 1.0f, 2.0f, 3.0f};
    float backend_norm[4] = {0.0f};
    float reference_norm[4] = {0.0f};
    float backend_swiglu[4] = {0.0f};
    float reference_swiglu[4] = {0.0f};
    float value = 1.0f;
    size_t i = 0u;

    status = att1_backend_cuda_create(&backend);
    if (!att1_backend_cuda_available()) {
        if ((status != ATT1_ERR_UNSUPPORTED) || (backend != NULL)) {
            fputs("cuda unavailable path should report unsupported\n", stderr);
            att1_backend_destroy(backend);
            return -1;
        }

        if (att1_backend_cuda_copy_host_to_device(NULL,
                                                  &value,
                                                  &value,
                                                  sizeof(value)) !=
            ATT1_ERR_INVALID_ARG) {
            fputs("cuda copy null backend should fail invalid-arg\n", stderr);
            return -1;
        }

        return 0;
    }

    if ((status != ATT1_OK) || (backend == NULL) ||
        (backend->ops == NULL) ||
        (strcmp(backend->ops->name, "cuda") != 0) ||
        (backend->ops->alloc == NULL) ||
        (backend->ops->free == NULL) ||
        (backend->ops->sync == NULL)) {
        fputs("cuda available path should create lifecycle backend\n", stderr);
        att1_backend_destroy(backend);
        return -1;
    }

    /* Milestone 14: matmul_f32 is now implemented via cuBLAS.
       1x1x1: dst = lhs * rhs = 1.0 * 1.0 = 1.0 */
    if (backend->ops->matmul_f32(backend,
                                 &value,
                                 &value,
                                 &value,
                                 1u,
                                 1u,
                                 1u) != 0) {
        fputs("cuda matmul_f32 1x1x1 should succeed\n", stderr);
        att1_backend_destroy(backend);
        return -1;
    }
    if (!near_f32(value, 1.0f)) {
        fputs("cuda matmul_f32 1x1x1 result mismatch\n", stderr);
        att1_backend_destroy(backend);
        return -1;
    }

    if ((backend->ops->rmsnorm_f32 == NULL) ||
        (backend->ops->rmsnorm_f32(backend,
                                   backend_norm,
                                   norm_src,
                                   norm_weight,
                                   4u,
                                   0.000001f) != 0) ||
        (att1_rmsnorm_f32(reference_norm,
                          norm_src,
                          norm_weight,
                          4u,
                          0.000001f) != 0)) {
        fputs("cuda rmsnorm_f32 should succeed\n", stderr);
        att1_backend_destroy(backend);
        return -1;
    }

    for (i = 0u; i < 4u; i++) {
        if (!near_cuda_f32(backend_norm[i], reference_norm[i])) {
            fputs("cuda rmsnorm_f32 result mismatch\n", stderr);
            att1_backend_destroy(backend);
            return -1;
        }
    }

    if ((backend->ops->ffn_swiglu_f32 == NULL) ||
        (backend->ops->ffn_swiglu_f32(backend,
                                      backend_swiglu,
                                      swiglu_gate,
                                      swiglu_value,
                                      4u) != 0) ||
        (att1_swiglu_f32(reference_swiglu,
                         swiglu_gate,
                         swiglu_value,
                         4u) != 0)) {
        fputs("cuda ffn_swiglu_f32 should succeed\n", stderr);
        att1_backend_destroy(backend);
        return -1;
    }

    for (i = 0u; i < 4u; i++) {
        if (!near_cuda_f32(backend_swiglu[i], reference_swiglu[i])) {
            fputs("cuda ffn_swiglu_f32 result mismatch\n", stderr);
            att1_backend_destroy(backend);
            return -1;
        }
    }

    if ((backend->ops->rope_f32 == NULL) ||
        (backend->ops->rope_f32(backend,
                                backend_rope,
                                6u,
                                1u,
                                1.0f) != 0) ||
        (att1_rope_f32(reference_rope,
                       6u,
                       1u,
                       1.0f) != 0)) {
        fputs("cuda rope_f32 should succeed\n", stderr);
        att1_backend_destroy(backend);
        return -1;
    }

    for (i = 0u; i < 6u; i++) {
        if (!near_cuda_f32(backend_rope[i], reference_rope[i])) {
            fputs("cuda rope_f32 result mismatch\n", stderr);
            att1_backend_destroy(backend);
            return -1;
        }
    }

    /* Milestone 18: Verify CUDA attention works with all components. */
    {
        att1_kv_cache cache;
        att1_attention_config attention_config = {
            .model_dim = 2u,
            .num_heads = 1u,
            .head_dim = 2u,
            .rope_theta = 1.0f
        };
        const float zero2[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        const float identity2[4] = {1.0f, 0.0f, 0.0f, 1.0f};
        att1_attention_weights attention_weights = {
            .wq = zero2, .wk = identity2, .wv = identity2, .wo = identity2
        };
        const float attention_input[2] = {1.0f, 2.0f};
        float attention_output[2] = {0.0f};

        if (att1_kv_cache_init(&cache, 3u, 1u, 2u) != 0) {
            fputs("cuda attention cache init failed\n", stderr);
            att1_backend_destroy(backend);
            return -1;
        }

        if (att1_attention_forward_backend(attention_output,
                                           &cache,
                                           attention_input,
                                           &attention_weights,
                                           &attention_config,
                                           0u,
                                           backend) != 0) {
            fputs("cuda attention_forward_backend should succeed\n", stderr);
            att1_kv_cache_free(&cache);
            att1_backend_destroy(backend);
            return -1;
        }

        att1_kv_cache_free(&cache);
    }

    att1_backend_destroy(backend);
    return 0;
}

int main(void)
{
    if (att1_backend_default_create(NULL) != ATT1_ERR_INVALID_ARG) {
        fputs("backend null create should fail\n", stderr);
        return 1;
    }
    att1_backend_destroy(NULL);

    if ((check_cpu_f32_backend() != 0) ||
        (check_cpu_q8_backend() != 0) ||
        (check_cuda_backend_skeleton() != 0)) {
        fputs("backend test failed\n", stderr);
        return 1;
    }

    puts("backend test passed");
    return 0;
}
