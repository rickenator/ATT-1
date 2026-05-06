/*
 * test_cuda_rope.c — Milestone 17: CUDA RoPE validation
 *
 * Compares CUDA backend RoPE behavior against CPU f32 RoPE with deterministic
 * vectors and positions. Non-CUDA builds or unavailable runtime paths skip CUDA
 * execution checks and verify unsupported behavior via backend creation.
 */

#include "att1_backend.h"
#include "att1_math.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#define ROPE_TOL 1e-3f

static int near_f32(float lhs, float rhs)
{
    return fabsf(lhs - rhs) < ROPE_TOL;
}

static int check_pos0_unchanged(void)
{
    const float original[6] = {1.0f, 0.5f, -2.0f, 3.0f, 4.0f, -1.0f};
    float cpu_values[6] = {0.0f};
    float cuda_values[6] = {0.0f};
    att1_backend *backend = NULL;
    size_t i = 0u;

    memcpy(cpu_values, original, sizeof(cpu_values));
    if (att1_rope_f32(cpu_values, 6u, 0u, 10000.0f) != 0) {
        fputs("pos0: cpu rope failed\n", stderr);
        return -1;
    }
    for (i = 0u; i < 6u; i++) {
        if (!near_f32(cpu_values[i], original[i])) {
            fputs("pos0: cpu values changed unexpectedly\n", stderr);
            return -1;
        }
    }

    if (!att1_backend_cuda_available()) {
        return 0;
    }

    memcpy(cuda_values, original, sizeof(cuda_values));
    if (att1_backend_cuda_create(&backend) != ATT1_OK) {
        fputs("pos0: cuda backend creation failed\n", stderr);
        return -1;
    }
    if (backend->ops->rope_f32(backend, cuda_values, 6u, 0u, 10000.0f) != 0) {
        fputs("pos0: cuda rope failed\n", stderr);
        att1_backend_destroy(backend);
        return -1;
    }

    for (i = 0u; i < 6u; i++) {
        if (!near_f32(cuda_values[i], original[i])) {
            fputs("pos0: cuda values changed unexpectedly\n", stderr);
            att1_backend_destroy(backend);
            return -1;
        }
    }

    att1_backend_destroy(backend);
    return 0;
}

static int check_rotation_pairs(void)
{
    float values[6] = {1.0f, 0.0f, 0.0f, 1.0f, 2.0f, 3.0f};
    const float c = cosf(1.0f);
    const float s = sinf(1.0f);

    if (att1_rope_f32(values, 6u, 1u, 1.0f) != 0) {
        fputs("pairs: cpu rope failed\n", stderr);
        return -1;
    }

    if (!near_f32(values[0], c) ||
        !near_f32(values[1], s) ||
        !near_f32(values[2], -s) ||
        !near_f32(values[3], c) ||
        !near_f32(values[4], (2.0f * c) - (3.0f * s)) ||
        !near_f32(values[5], (2.0f * s) + (3.0f * c))) {
        fputs("pairs: rotation pair layout mismatch\n", stderr);
        return -1;
    }

    return 0;
}

static int check_odd_dimension_failure(void)
{
    float values[5] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    att1_backend *backend = NULL;

    if (att1_rope_f32(values, 5u, 1u, 10000.0f) == 0) {
        fputs("odd-dim: cpu rope should fail\n", stderr);
        return -1;
    }

    if (!att1_backend_cuda_available()) {
        return 0;
    }

    if (att1_backend_cuda_create(&backend) != ATT1_OK) {
        fputs("odd-dim: cuda backend creation failed\n", stderr);
        return -1;
    }
    if (backend->ops->rope_f32(backend, values, 5u, 1u, 10000.0f) == 0) {
        fputs("odd-dim: cuda rope should fail\n", stderr);
        att1_backend_destroy(backend);
        return -1;
    }

    att1_backend_destroy(backend);
    return 0;
}

static int check_multi_head_multi_position(void)
{
    /* 3 heads, head_dim=6 -> 18 values total */
    float base[18];
    float cpu_values[18];
    float cuda_values[18];
    att1_backend *backend = NULL;
    size_t pos = 0u;
    size_t i = 0u;

    for (i = 0u; i < 18u; i++) {
        base[i] = ((float)((int)(i % 7u) - 3) * 0.5f) + ((float)i * 0.03125f);
    }

    if (!att1_backend_cuda_available()) {
        return 0;
    }

    if (att1_backend_cuda_create(&backend) != ATT1_OK) {
        fputs("multi-head: cuda backend creation failed\n", stderr);
        return -1;
    }

    for (pos = 0u; pos < 4u; pos++) {
        memcpy(cpu_values, base, sizeof(cpu_values));
        memcpy(cuda_values, base, sizeof(cuda_values));

        /* Apply RoPE per head exactly as attention code would. */
        for (i = 0u; i < 3u; i++) {
            float *cpu_head = &cpu_values[i * 6u];
            float *cuda_head = &cuda_values[i * 6u];

            if (att1_rope_f32(cpu_head, 6u, pos, 10000.0f) != 0) {
                fputs("multi-head: cpu rope failed\n", stderr);
                att1_backend_destroy(backend);
                return -1;
            }
            if (backend->ops->rope_f32(backend,
                                       cuda_head,
                                       6u,
                                       pos,
                                       10000.0f) != 0) {
                fputs("multi-head: cuda rope failed\n", stderr);
                att1_backend_destroy(backend);
                return -1;
            }
        }

        for (i = 0u; i < 18u; i++) {
            if (!near_f32(cuda_values[i], cpu_values[i])) {
                fprintf(stderr,
                        "multi-head: pos=%zu idx=%zu cuda=%.6f cpu=%.6f\n",
                        pos,
                        i,
                        (double)cuda_values[i],
                        (double)cpu_values[i]);
                att1_backend_destroy(backend);
                return -1;
            }
        }
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
        fputs("unavail: expected unsupported from cuda create\n", stderr);
        att1_backend_destroy(backend);
        return -1;
    }
    if (backend != NULL) {
        fputs("unavail: backend should be NULL\n", stderr);
        att1_backend_destroy(backend);
        return -1;
    }

    return 0;
}

static int check_no_silent_fallback(void)
{
    float cpu_values[6] = {0.5f, -1.0f, 1.5f, 2.0f, -2.5f, 3.0f};
    float cuda_values[6] = {0.5f, -1.0f, 1.5f, 2.0f, -2.5f, 3.0f};
    att1_backend *backend = NULL;
    size_t i = 0u;

    if (!att1_backend_cuda_available()) {
        return 0;
    }

    if (att1_backend_cuda_create(&backend) != ATT1_OK) {
        fputs("fallback: cuda backend creation failed\n", stderr);
        return -1;
    }
    if ((backend->ops == NULL) || (backend->ops->name == NULL) ||
        (strcmp(backend->ops->name, "cuda") != 0)) {
        fputs("fallback: expected backend name \"cuda\"\n", stderr);
        att1_backend_destroy(backend);
        return -1;
    }

    if ((att1_rope_f32(cpu_values, 6u, 2u, 10000.0f) != 0) ||
        (backend->ops->rope_f32(backend, cuda_values, 6u, 2u, 10000.0f) != 0)) {
        fputs("fallback: cpu or cuda rope failed\n", stderr);
        att1_backend_destroy(backend);
        return -1;
    }

    for (i = 0u; i < 6u; i++) {
        if (!near_f32(cuda_values[i], cpu_values[i])) {
            fprintf(stderr,
                    "fallback: idx=%zu cuda=%.6f cpu=%.6f\n",
                    i,
                    (double)cuda_values[i],
                    (double)cpu_values[i]);
            att1_backend_destroy(backend);
            return -1;
        }
    }

    att1_backend_destroy(backend);
    return 0;
}

int main(void)
{
    if ((check_pos0_unchanged() != 0) ||
        (check_rotation_pairs() != 0) ||
        (check_odd_dimension_failure() != 0) ||
        (check_multi_head_multi_position() != 0) ||
        (check_cuda_unavailable() != 0) ||
        (check_no_silent_fallback() != 0)) {
        fputs("cuda_rope test failed\n", stderr);
        return 1;
    }

    puts("cuda_rope test passed");
    return 0;
}