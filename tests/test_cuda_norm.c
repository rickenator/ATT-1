/*
 * test_cuda_norm.c — Milestone 15: CUDA RMSNorm validation
 *
 * The CUDA RMSNorm path is compared against the CPU f32 reference with a
 * tolerance of 1e-3. Non-CUDA builds and machines without a usable CUDA
 * runtime skip execution-path checks and verify unsupported behavior instead.
 */

#include "att1_backend.h"
#include "att1_math.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#define RMSNORM_TOL 1e-3f

static int near_f32(float lhs, float rhs)
{
    return fabsf(lhs - rhs) < RMSNORM_TOL;
}

static int check_tiny_rmsnorm(void)
{
    const float src[4] = {1.0f, -2.0f, 3.0f, -4.0f};
    const float weight[4] = {1.0f, 0.5f, 1.5f, 2.0f};
    float cpu_out[4] = {0.0f};
    float cuda_out[4] = {0.0f};
    att1_backend *backend = NULL;
    size_t i = 0u;

    if (att1_rmsnorm_f32(cpu_out, src, weight, 4u, 0.000001f) != 0) {
        fputs("tiny rmsnorm: cpu reference failed\n", stderr);
        return -1;
    }

    if (!att1_backend_cuda_available()) {
        return 0;
    }

    if (att1_backend_cuda_create(&backend) != ATT1_OK) {
        fputs("tiny rmsnorm: cuda backend creation failed\n", stderr);
        return -1;
    }

    if (backend->ops->rmsnorm_f32(backend,
                                  cuda_out,
                                  src,
                                  weight,
                                  4u,
                                  0.000001f) != 0) {
        fputs("tiny rmsnorm: cuda rmsnorm_f32 failed\n", stderr);
        att1_backend_destroy(backend);
        return -1;
    }

    for (i = 0u; i < 4u; i++) {
        if (!near_f32(cuda_out[i], cpu_out[i])) {
            fprintf(stderr,
                    "tiny rmsnorm: element %zu cuda=%.6f cpu=%.6f\n",
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

static int check_deterministic_rmsnorm(void)
{
    float src[32];
    float weight[32];
    float cpu_out[32];
    float cuda_out[32];
    att1_backend *backend = NULL;
    size_t i = 0u;

    for (i = 0u; i < 32u; i++) {
        src[i] = ((float)((int)(i % 7u) - 3) * 0.75f) + ((float)i * 0.03125f);
        weight[i] = 0.5f + ((float)((i % 5u) + 1u) * 0.2f);
    }

    if (att1_rmsnorm_f32(cpu_out, src, weight, 32u, 0.00001f) != 0) {
        fputs("deterministic rmsnorm: cpu reference failed\n", stderr);
        return -1;
    }

    if (!att1_backend_cuda_available()) {
        return 0;
    }

    if (att1_backend_cuda_create(&backend) != ATT1_OK) {
        fputs("deterministic rmsnorm: cuda backend creation failed\n", stderr);
        return -1;
    }

    if (backend->ops->rmsnorm_f32(backend,
                                  cuda_out,
                                  src,
                                  weight,
                                  32u,
                                  0.00001f) != 0) {
        fputs("deterministic rmsnorm: cuda rmsnorm_f32 failed\n", stderr);
        att1_backend_destroy(backend);
        return -1;
    }

    for (i = 0u; i < 32u; i++) {
        if (!near_f32(cuda_out[i], cpu_out[i])) {
            fprintf(stderr,
                    "deterministic rmsnorm: element %zu cuda=%.6f cpu=%.6f\n",
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

static int check_shape_handling(void)
{
    float dst[4] = {0.0f};
    const float src[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    const float weight[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    att1_backend *backend = NULL;

    if (att1_rmsnorm_f32(NULL, src, weight, 4u, 0.000001f) == 0) {
        fputs("shape rmsnorm: cpu null dst should fail\n", stderr);
        return -1;
    }
    if (att1_rmsnorm_f32(dst, NULL, weight, 4u, 0.000001f) == 0) {
        fputs("shape rmsnorm: cpu null src should fail\n", stderr);
        return -1;
    }
    if (att1_rmsnorm_f32(dst, src, NULL, 4u, 0.000001f) == 0) {
        fputs("shape rmsnorm: cpu null weight should fail\n", stderr);
        return -1;
    }
    if (att1_rmsnorm_f32(dst, src, weight, 0u, 0.000001f) == 0) {
        fputs("shape rmsnorm: cpu zero count should fail\n", stderr);
        return -1;
    }
    if (att1_rmsnorm_f32(dst, src, weight, 4u, 0.0f) == 0) {
        fputs("shape rmsnorm: cpu nonpositive epsilon should fail\n", stderr);
        return -1;
    }

    if (!att1_backend_cuda_available()) {
        return 0;
    }

    if (att1_backend_cuda_create(&backend) != ATT1_OK) {
        fputs("shape rmsnorm: cuda backend creation failed\n", stderr);
        return -1;
    }

    if (backend->ops->rmsnorm_f32(backend,
                                  NULL,
                                  src,
                                  weight,
                                  4u,
                                  0.000001f) == 0) {
        fputs("shape rmsnorm: cuda null dst should fail\n", stderr);
        att1_backend_destroy(backend);
        return -1;
    }
    if (backend->ops->rmsnorm_f32(backend,
                                  dst,
                                  src,
                                  weight,
                                  0u,
                                  0.000001f) == 0) {
        fputs("shape rmsnorm: cuda zero count should fail\n", stderr);
        att1_backend_destroy(backend);
        return -1;
    }
    if (backend->ops->rmsnorm_f32(backend,
                                  dst,
                                  src,
                                  weight,
                                  4u,
                                  0.0f) == 0) {
        fputs("shape rmsnorm: cuda nonpositive epsilon should fail\n", stderr);
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
        fputs("unavailable rmsnorm: expected unsupported from cuda create\n",
              stderr);
        att1_backend_destroy(backend);
        return -1;
    }
    if (backend != NULL) {
        fputs("unavailable rmsnorm: backend should remain NULL\n", stderr);
        att1_backend_destroy(backend);
        return -1;
    }

    return 0;
}

static int check_no_silent_fallback(void)
{
    const float src[4] = {0.5f, -1.5f, 2.5f, -3.5f};
    const float weight[4] = {1.0f, 1.1f, 1.2f, 1.3f};
    float cpu_out[4] = {0.0f};
    float cuda_out[4] = {0.0f};
    att1_backend *backend = NULL;
    size_t i = 0u;

    if (!att1_backend_cuda_available()) {
        return 0;
    }

    if (att1_backend_cuda_create(&backend) != ATT1_OK) {
        fputs("fallback rmsnorm: cuda backend creation failed\n", stderr);
        return -1;
    }

    if ((backend->ops == NULL) ||
        (backend->ops->name == NULL) ||
        (strcmp(backend->ops->name, "cuda") != 0)) {
        fputs("fallback rmsnorm: expected backend name \"cuda\"\n", stderr);
        att1_backend_destroy(backend);
        return -1;
    }

    if ((att1_rmsnorm_f32(cpu_out, src, weight, 4u, 0.000001f) != 0) ||
        (backend->ops->rmsnorm_f32(backend,
                                   cuda_out,
                                   src,
                                   weight,
                                   4u,
                                   0.000001f) != 0)) {
        fputs("fallback rmsnorm: reference or cuda path failed\n", stderr);
        att1_backend_destroy(backend);
        return -1;
    }

    for (i = 0u; i < 4u; i++) {
        if (!near_f32(cuda_out[i], cpu_out[i])) {
            fprintf(stderr,
                    "fallback rmsnorm: element %zu cuda=%.6f cpu=%.6f\n",
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
    if ((check_tiny_rmsnorm() != 0) ||
        (check_deterministic_rmsnorm() != 0) ||
        (check_shape_handling() != 0) ||
        (check_cuda_unavailable() != 0) ||
        (check_no_silent_fallback() != 0)) {
        fputs("cuda_norm test failed\n", stderr);
        return 1;
    }

    puts("cuda_norm test passed");
    return 0;
}