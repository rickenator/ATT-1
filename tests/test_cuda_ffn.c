/*
 * test_cuda_ffn.c — Milestone 16: CUDA FFN/SwiGLU validation
 *
 * This test composes the existing backend API the same way the transformer
 * block FFN path does: gate matmul, up matmul, SwiGLU, then down matmul.
 * CPU f32 remains the correctness reference. Non-CUDA builds and machines
 * without a usable CUDA runtime skip CUDA execution-path checks and verify the
 * unsupported path instead.
 */

#include "att1_backend.h"
#include "att1_math.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define FFN_TOL 1e-3f

static int near_f32(float lhs, float rhs)
{
    return fabsf(lhs - rhs) < FFN_TOL;
}

static int cpu_ffn_forward(float *dst,
                           const float *input,
                           const float *w_gate,
                           const float *w_up,
                           const float *w_down,
                           size_t model_dim,
                           size_t ffn_dim)
{
    float *gate = NULL;
    float *up = NULL;
    float *hidden = NULL;
    int rc = -1;

    if ((dst == NULL) || (input == NULL) || (w_gate == NULL) ||
        (w_up == NULL) || (w_down == NULL)) {
        return -1;
    }
    if ((model_dim == 0u) || (ffn_dim == 0u)) {
        return -1;
    }

    gate = malloc(ffn_dim * sizeof(float));
    up = malloc(ffn_dim * sizeof(float));
    hidden = malloc(ffn_dim * sizeof(float));
    if ((gate == NULL) || (up == NULL) || (hidden == NULL)) {
        goto cleanup;
    }

    if ((att1_matmul_f32(gate, input, w_gate, 1u, ffn_dim, model_dim) != 0) ||
        (att1_matmul_f32(up, input, w_up, 1u, ffn_dim, model_dim) != 0) ||
        (att1_swiglu_f32(hidden, gate, up, ffn_dim) != 0) ||
        (att1_matmul_f32(dst, hidden, w_down, 1u, model_dim, ffn_dim) != 0)) {
        goto cleanup;
    }

    rc = 0;

cleanup:
    free(gate);
    free(up);
    free(hidden);
    return rc;
}

static int cuda_ffn_forward(att1_backend *backend,
                            float *dst,
                            const float *input,
                            const float *w_gate,
                            const float *w_up,
                            const float *w_down,
                            size_t model_dim,
                            size_t ffn_dim)
{
    float *gate = NULL;
    float *up = NULL;
    float *hidden = NULL;
    int rc = -1;

    if ((backend == NULL) || (backend->ops == NULL) ||
        (backend->ops->matmul_f32 == NULL) ||
        (backend->ops->ffn_swiglu_f32 == NULL) ||
        (dst == NULL) || (input == NULL) || (w_gate == NULL) ||
        (w_up == NULL) || (w_down == NULL)) {
        return -1;
    }
    if ((model_dim == 0u) || (ffn_dim == 0u)) {
        return -1;
    }

    gate = malloc(ffn_dim * sizeof(float));
    up = malloc(ffn_dim * sizeof(float));
    hidden = malloc(ffn_dim * sizeof(float));
    if ((gate == NULL) || (up == NULL) || (hidden == NULL)) {
        goto cleanup;
    }

    if ((backend->ops->matmul_f32(backend,
                                  gate,
                                  input,
                                  w_gate,
                                  1u,
                                  ffn_dim,
                                  model_dim) != 0) ||
        (backend->ops->matmul_f32(backend,
                                  up,
                                  input,
                                  w_up,
                                  1u,
                                  ffn_dim,
                                  model_dim) != 0) ||
        (backend->ops->ffn_swiglu_f32(backend,
                                      hidden,
                                      gate,
                                      up,
                                      ffn_dim) != 0) ||
        (backend->ops->matmul_f32(backend,
                                  dst,
                                  hidden,
                                  w_down,
                                  1u,
                                  model_dim,
                                  ffn_dim) != 0)) {
        goto cleanup;
    }

    rc = 0;

cleanup:
    free(gate);
    free(up);
    free(hidden);
    return rc;
}

static int check_tiny_hand_checkable_ffn(void)
{
    const float input[2] = {1.0f, -2.0f};
    const float w_gate[8] = {
        1.0f, 0.0f, -1.0f, 2.0f,
        0.5f, -0.5f, 1.5f, -1.0f
    };
    const float w_up[8] = {
        0.5f, 1.0f, 0.0f, -1.0f,
        -1.0f, 2.0f, 1.0f, 0.5f
    };
    const float w_down[8] = {
        1.0f, -1.0f,
        0.5f, 0.0f,
        -0.5f, 2.0f,
        1.5f, -1.5f
    };
    float cpu_out[2] = {0.0f};
    float cuda_out[2] = {0.0f};
    att1_backend *backend = NULL;
    size_t i = 0u;

    if (cpu_ffn_forward(cpu_out,
                        input,
                        w_gate,
                        w_up,
                        w_down,
                        2u,
                        4u) != 0) {
        fputs("tiny ffn: cpu reference failed\n", stderr);
        return -1;
    }

    if (!att1_backend_cuda_available()) {
        return 0;
    }

    if (att1_backend_cuda_create(&backend) != ATT1_OK) {
        fputs("tiny ffn: cuda backend creation failed\n", stderr);
        return -1;
    }

    if (cuda_ffn_forward(backend,
                         cuda_out,
                         input,
                         w_gate,
                         w_up,
                         w_down,
                         2u,
                         4u) != 0) {
        fputs("tiny ffn: cuda path failed\n", stderr);
        att1_backend_destroy(backend);
        return -1;
    }

    for (i = 0u; i < 2u; i++) {
        if (!near_f32(cuda_out[i], cpu_out[i])) {
            fprintf(stderr,
                    "tiny ffn: element %zu cuda=%.6f cpu=%.6f\n",
                    i,
                    (double)cuda_out[i],
                    (double)cpu_out[i]);
            att1_backend_destroy(backend);
            return -1;
        }
    }

    att1_backend_destroy(backend);
    return 0;
}

static int check_medium_deterministic_ffn(void)
{
    float input[4];
    float w_gate[32];
    float w_up[32];
    float w_down[32];
    float cpu_out[4] = {0.0f};
    float cuda_out[4] = {0.0f};
    att1_backend *backend = NULL;
    size_t i = 0u;

    for (i = 0u; i < 4u; i++) {
        input[i] = ((float)((int)i - 1) * 0.75f) + 0.25f;
    }
    for (i = 0u; i < 32u; i++) {
        w_gate[i] = ((float)((int)(i % 9u) - 4) * 0.125f);
        w_up[i] = ((float)((i % 7u) + 1u) * 0.2f) - 0.6f;
        w_down[i] = ((float)((int)(i % 11u) - 5) * 0.1f);
    }

    if (cpu_ffn_forward(cpu_out,
                        input,
                        w_gate,
                        w_up,
                        w_down,
                        4u,
                        8u) != 0) {
        fputs("medium ffn: cpu reference failed\n", stderr);
        return -1;
    }

    if (!att1_backend_cuda_available()) {
        return 0;
    }

    if (att1_backend_cuda_create(&backend) != ATT1_OK) {
        fputs("medium ffn: cuda backend creation failed\n", stderr);
        return -1;
    }

    if (cuda_ffn_forward(backend,
                         cuda_out,
                         input,
                         w_gate,
                         w_up,
                         w_down,
                         4u,
                         8u) != 0) {
        fputs("medium ffn: cuda path failed\n", stderr);
        att1_backend_destroy(backend);
        return -1;
    }

    for (i = 0u; i < 4u; i++) {
        if (!near_f32(cuda_out[i], cpu_out[i])) {
            fprintf(stderr,
                    "medium ffn: element %zu cuda=%.6f cpu=%.6f\n",
                    i,
                    (double)cuda_out[i],
                    (double)cpu_out[i]);
            att1_backend_destroy(backend);
            return -1;
        }
    }

    att1_backend_destroy(backend);
    return 0;
}

static int check_zero_weights(void)
{
    const float input[4] = {1.0f, -2.0f, 3.0f, -4.0f};
    const float w_gate[32] = {0.0f};
    const float w_up[32] = {0.0f};
    const float w_down[32] = {0.0f};
    float cpu_out[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    float cuda_out[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    att1_backend *backend = NULL;
    size_t i = 0u;

    if (cpu_ffn_forward(cpu_out,
                        input,
                        w_gate,
                        w_up,
                        w_down,
                        4u,
                        8u) != 0) {
        fputs("zero weights: cpu reference failed\n", stderr);
        return -1;
    }
    for (i = 0u; i < 4u; i++) {
        if (!near_f32(cpu_out[i], 0.0f)) {
            fputs("zero weights: cpu output should be zero\n", stderr);
            return -1;
        }
    }

    if (!att1_backend_cuda_available()) {
        return 0;
    }

    if (att1_backend_cuda_create(&backend) != ATT1_OK) {
        fputs("zero weights: cuda backend creation failed\n", stderr);
        return -1;
    }

    if (cuda_ffn_forward(backend,
                         cuda_out,
                         input,
                         w_gate,
                         w_up,
                         w_down,
                         4u,
                         8u) != 0) {
        fputs("zero weights: cuda path failed\n", stderr);
        att1_backend_destroy(backend);
        return -1;
    }

    for (i = 0u; i < 4u; i++) {
        if (!near_f32(cuda_out[i], 0.0f)) {
            fputs("zero weights: cuda output should be zero\n", stderr);
            att1_backend_destroy(backend);
            return -1;
        }
    }

    att1_backend_destroy(backend);
    return 0;
}

static int check_activation_values(void)
{
    const float gate[5] = {-3.0f, -1.0f, 0.0f, 2.0f, 4.0f};
    const float value[5] = {1.0f, -2.0f, 3.0f, -4.0f, 5.0f};
    float cpu_out[5] = {0.0f};
    float cuda_out[5] = {0.0f};
    att1_backend *backend = NULL;
    size_t i = 0u;

    if (att1_swiglu_f32(cpu_out, gate, value, 5u) != 0) {
        fputs("activation values: cpu swiglu failed\n", stderr);
        return -1;
    }

    if (!att1_backend_cuda_available()) {
        return 0;
    }

    if (att1_backend_cuda_create(&backend) != ATT1_OK) {
        fputs("activation values: cuda backend creation failed\n", stderr);
        return -1;
    }

    if (backend->ops->ffn_swiglu_f32(backend,
                                     cuda_out,
                                     gate,
                                     value,
                                     5u) != 0) {
        fputs("activation values: cuda swiglu failed\n", stderr);
        att1_backend_destroy(backend);
        return -1;
    }

    for (i = 0u; i < 5u; i++) {
        if (!near_f32(cuda_out[i], cpu_out[i])) {
            fprintf(stderr,
                    "activation values: element %zu cuda=%.6f cpu=%.6f\n",
                    i,
                    (double)cuda_out[i],
                    (double)cpu_out[i]);
            att1_backend_destroy(backend);
            return -1;
        }
    }

    att1_backend_destroy(backend);
    return 0;
}

static int check_shape_failure(void)
{
    float dst[4] = {0.0f};
    const float input[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    const float weights[16] = {0.0f};
    att1_backend *backend = NULL;

    if (cpu_ffn_forward(dst,
                        input,
                        weights,
                        weights,
                        weights,
                        0u,
                        4u) == 0) {
        fputs("shape failure: cpu zero model_dim should fail\n", stderr);
        return -1;
    }
    if (att1_swiglu_f32(dst, input, input, 0u) == 0) {
        fputs("shape failure: cpu zero count swiglu should fail\n", stderr);
        return -1;
    }

    if (!att1_backend_cuda_available()) {
        return 0;
    }

    if (att1_backend_cuda_create(&backend) != ATT1_OK) {
        fputs("shape failure: cuda backend creation failed\n", stderr);
        return -1;
    }

    if (cuda_ffn_forward(backend,
                         dst,
                         input,
                         weights,
                         weights,
                         weights,
                         4u,
                         0u) == 0) {
        fputs("shape failure: cuda zero ffn_dim should fail\n", stderr);
        att1_backend_destroy(backend);
        return -1;
    }
    if (backend->ops->ffn_swiglu_f32(backend,
                                     dst,
                                     input,
                                     input,
                                     0u) == 0) {
        fputs("shape failure: cuda zero count swiglu should fail\n", stderr);
        att1_backend_destroy(backend);
        return -1;
    }

    att1_backend_destroy(backend);
    return 0;
}

static int check_cuda_unavailable(void)
{
    att1_backend *backend = NULL;

    if (att1_backend_cuda_available()) {
        return 0;
    }

    if (att1_backend_cuda_create(&backend) != ATT1_ERR_UNSUPPORTED) {
        fputs("unavailable ffn: expected unsupported from cuda create\n", stderr);
        att1_backend_destroy(backend);
        return -1;
    }
    if (backend != NULL) {
        fputs("unavailable ffn: backend should remain NULL\n", stderr);
        att1_backend_destroy(backend);
        return -1;
    }

    return 0;
}

static int check_no_silent_fallback(void)
{
    const float input[2] = {0.25f, -0.75f};
    const float w_gate[8] = {
        0.5f, -1.0f, 1.5f, 0.25f,
        -0.5f, 0.75f, -1.25f, 1.0f
    };
    const float w_up[8] = {
        1.0f, 0.5f, -0.5f, 1.5f,
        -1.0f, 0.25f, 0.75f, -0.25f
    };
    const float w_down[8] = {
        1.0f, 0.5f,
        -0.5f, 1.5f,
        0.75f, -1.25f,
        0.25f, 0.0f
    };
    float cpu_out[2] = {0.0f};
    float cuda_out[2] = {0.0f};
    att1_backend *backend = NULL;
    size_t i = 0u;

    if (!att1_backend_cuda_available()) {
        return 0;
    }

    if (att1_backend_cuda_create(&backend) != ATT1_OK) {
        fputs("fallback ffn: cuda backend creation failed\n", stderr);
        return -1;
    }

    if ((backend->ops == NULL) ||
        (backend->ops->name == NULL) ||
        (strcmp(backend->ops->name, "cuda") != 0)) {
        fputs("fallback ffn: expected backend name \"cuda\"\n", stderr);
        att1_backend_destroy(backend);
        return -1;
    }

    if ((cpu_ffn_forward(cpu_out,
                         input,
                         w_gate,
                         w_up,
                         w_down,
                         2u,
                         4u) != 0) ||
        (cuda_ffn_forward(backend,
                          cuda_out,
                          input,
                          w_gate,
                          w_up,
                          w_down,
                          2u,
                          4u) != 0)) {
        fputs("fallback ffn: reference or cuda path failed\n", stderr);
        att1_backend_destroy(backend);
        return -1;
    }

    for (i = 0u; i < 2u; i++) {
        if (!near_f32(cuda_out[i], cpu_out[i])) {
            fprintf(stderr,
                    "fallback ffn: element %zu cuda=%.6f cpu=%.6f\n",
                    i,
                    (double)cuda_out[i],
                    (double)cpu_out[i]);
            att1_backend_destroy(backend);
            return -1;
        }
    }

    att1_backend_destroy(backend);
    return 0;
}

int main(void)
{
    if ((check_tiny_hand_checkable_ffn() != 0) ||
        (check_medium_deterministic_ffn() != 0) ||
        (check_zero_weights() != 0) ||
        (check_activation_values() != 0) ||
        (check_shape_failure() != 0) ||
        (check_cuda_unavailable() != 0) ||
        (check_no_silent_fallback() != 0)) {
        fputs("cuda_ffn test failed\n", stderr);
        return 1;
    }

    puts("cuda_ffn test passed");
    return 0;
}