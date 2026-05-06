/*
 * test_cuda_matmul.c — Milestone 14: CUDA f32 matmul validation
 *
 * Tests:
 *   1. Tiny known 2×3 × 3×1 case with hand-checkable expected values.
 *   2. Larger deterministic 4×8 × 8×4 case compared to CPU f32 reference.
 *   3. Shape handling: NULL/zero arguments fail cleanly on both CPU and CUDA.
 *   4. CUDA unavailable: att1_backend_cuda_create returns ATT1_ERR_UNSUPPORTED.
 *   5. No silent fallback: CUDA backend name is "cuda", not "cpu-f32".
 *
 * Non-CUDA builds and machines without a GPU skip CUDA execution tests and
 * exercise the unavailable-path behavior instead.  All tests return 0 on
 * success; the binary exits 1 on any failure.
 */

#include "att1_backend.h"
#include "att1_math.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

/* Tolerance for comparing CUDA and CPU f32 results.  Single-precision
   arithmetic differences accumulate slowly for small matrices; 1e-3 is
   conservative enough for the matrix sizes used here. */
#define MATMUL_TOL 1e-3f

static int near_f32(float a, float b)
{
    return fabsf(a - b) < MATMUL_TOL;
}

/* ------------------------------------------------------------------ */
/* Test 1: tiny known 2×3 × 3×1 = 2×1                                 */
/* ------------------------------------------------------------------ */

static int check_tiny_known(void)
{
    /* lhs (2 rows, 3 inner) */
    const float lhs[6] = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f
    };
    /* rhs (3 inner, 1 col) */
    const float rhs[3] = {7.0f, 8.0f, 9.0f};
    /* expected: [1*7+2*8+3*9, 4*7+5*8+6*9] = [50, 122] */
    const float expected[2] = {50.0f, 122.0f};
    float cuda_out[2] = {0.0f, 0.0f};
    att1_backend *backend = NULL;
    size_t i = 0u;

    if (!att1_backend_cuda_available()) {
        return 0;
    }

    if (att1_backend_cuda_create(&backend) != ATT1_OK) {
        fputs("tiny: cuda backend creation failed\n", stderr);
        return -1;
    }

    if (backend->ops->matmul_f32(backend,
                                 cuda_out, lhs, rhs,
                                 2u, 1u, 3u) != 0) {
        fputs("tiny: cuda matmul_f32 returned failure\n", stderr);
        att1_backend_destroy(backend);
        return -1;
    }

    for (i = 0u; i < 2u; i++) {
        if (!near_f32(cuda_out[i], expected[i])) {
            fprintf(stderr,
                    "tiny: element %zu: cuda=%.6f expected=%.6f\n",
                    i, (double)cuda_out[i], (double)expected[i]);
            att1_backend_destroy(backend);
            return -1;
        }
    }

    att1_backend_destroy(backend);
    return 0;
}

/* ------------------------------------------------------------------ */
/* Test 2: larger deterministic 4×8 × 8×4 = 4×4                       */
/* ------------------------------------------------------------------ */

static int check_larger_deterministic(void)
{
    /* rows=4, inner=8, cols=4 */
    const size_t M = 4u;
    const size_t K = 8u;
    const size_t N = 4u;
    float lhs[32];  /* M*K */
    float rhs[32];  /* K*N */
    float cpu_out[16];
    float cuda_out[16];
    att1_backend *backend = NULL;
    size_t i = 0u;
    size_t k = 0u;

    /* Deterministic fill: lhs[i][k] = (i*K + k + 1) * 0.25f */
    for (i = 0u; i < M; i++) {
        for (k = 0u; k < K; k++) {
            lhs[i * K + k] = (float)(i * K + k + 1u) * 0.25f;
        }
    }
    /* rhs[k][j] = (k*N + j + 1) * 0.25f */
    for (k = 0u; k < K; k++) {
        size_t j = 0u;
        for (j = 0u; j < N; j++) {
            rhs[k * N + j] = (float)(k * N + j + 1u) * 0.25f;
        }
    }

    /* CPU reference */
    if (att1_matmul_f32(cpu_out, lhs, rhs, M, N, K) != 0) {
        fputs("larger: cpu matmul_f32 failed\n", stderr);
        return -1;
    }

    if (!att1_backend_cuda_available()) {
        return 0;
    }

    if (att1_backend_cuda_create(&backend) != ATT1_OK) {
        fputs("larger: cuda backend creation failed\n", stderr);
        return -1;
    }

    if (backend->ops->matmul_f32(backend,
                                 cuda_out, lhs, rhs,
                                 M, N, K) != 0) {
        fputs("larger: cuda matmul_f32 returned failure\n", stderr);
        att1_backend_destroy(backend);
        return -1;
    }

    for (i = 0u; i < M * N; i++) {
        if (!near_f32(cuda_out[i], cpu_out[i])) {
            fprintf(stderr,
                    "larger: element %zu: cuda=%.6f cpu=%.6f\n",
                    i, (double)cuda_out[i], (double)cpu_out[i]);
            att1_backend_destroy(backend);
            return -1;
        }
    }

    att1_backend_destroy(backend);
    return 0;
}

/* ------------------------------------------------------------------ */
/* Test 3: shape handling — NULL and zero-dimension arguments          */
/* ------------------------------------------------------------------ */

static int check_shape_handling(void)
{
    float dst[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const float lhs[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    const float rhs[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    att1_backend *backend = NULL;

    /* CPU reference: NULL and zero checks */
    if (att1_matmul_f32(NULL, lhs, rhs, 2u, 2u, 2u) == 0) {
        fputs("shape: cpu null dst should fail\n", stderr);
        return -1;
    }
    if (att1_matmul_f32(dst, NULL, rhs, 2u, 2u, 2u) == 0) {
        fputs("shape: cpu null lhs should fail\n", stderr);
        return -1;
    }
    if (att1_matmul_f32(dst, lhs, NULL, 2u, 2u, 2u) == 0) {
        fputs("shape: cpu null rhs should fail\n", stderr);
        return -1;
    }
    if (att1_matmul_f32(dst, lhs, rhs, 0u, 2u, 2u) == 0) {
        fputs("shape: cpu zero rows should fail\n", stderr);
        return -1;
    }
    if (att1_matmul_f32(dst, lhs, rhs, 2u, 0u, 2u) == 0) {
        fputs("shape: cpu zero cols should fail\n", stderr);
        return -1;
    }
    if (att1_matmul_f32(dst, lhs, rhs, 2u, 2u, 0u) == 0) {
        fputs("shape: cpu zero inner should fail\n", stderr);
        return -1;
    }

    if (!att1_backend_cuda_available()) {
        return 0;
    }

    if (att1_backend_cuda_create(&backend) != ATT1_OK) {
        fputs("shape: cuda backend creation failed\n", stderr);
        return -1;
    }

    /* CUDA backend: same invalid-arg contract */
    if (backend->ops->matmul_f32(backend,
                                 NULL, lhs, rhs, 2u, 2u, 2u) == 0) {
        fputs("shape: cuda null dst should fail\n", stderr);
        att1_backend_destroy(backend);
        return -1;
    }
    if (backend->ops->matmul_f32(backend,
                                 dst, NULL, rhs, 2u, 2u, 2u) == 0) {
        fputs("shape: cuda null lhs should fail\n", stderr);
        att1_backend_destroy(backend);
        return -1;
    }
    if (backend->ops->matmul_f32(backend,
                                 dst, lhs, rhs, 0u, 2u, 2u) == 0) {
        fputs("shape: cuda zero rows should fail\n", stderr);
        att1_backend_destroy(backend);
        return -1;
    }
    if (backend->ops->matmul_f32(backend,
                                 dst, lhs, rhs, 2u, 2u, 0u) == 0) {
        fputs("shape: cuda zero inner should fail\n", stderr);
        att1_backend_destroy(backend);
        return -1;
    }

    att1_backend_destroy(backend);
    return 0;
}

/* ------------------------------------------------------------------ */
/* Test 4: CUDA unavailable — create returns ATT1_ERR_UNSUPPORTED      */
/* ------------------------------------------------------------------ */

static int check_cuda_unavailable(void)
{
    att1_backend *backend = NULL;
    att1_status_t status = ATT1_OK;

    if (att1_backend_cuda_available()) {
        /* CUDA is present; unavailable-path does not apply. */
        return 0;
    }

    status = att1_backend_cuda_create(&backend);
    if (status != ATT1_ERR_UNSUPPORTED) {
        fputs("unavail: expected ATT1_ERR_UNSUPPORTED from cuda create\n",
              stderr);
        att1_backend_destroy(backend);
        return -1;
    }
    if (backend != NULL) {
        fputs("unavail: backend pointer must be NULL on unsupported\n", stderr);
        att1_backend_destroy(backend);
        return -1;
    }

    return 0;
}

/* ------------------------------------------------------------------ */
/* Test 5: no silent fallback — backend name must be "cuda"            */
/* ------------------------------------------------------------------ */

static int check_no_silent_fallback(void)
{
    /* 2×2 identity × 2×2 test matrix */
    const float lhs[4] = {1.0f, 0.0f, 0.0f, 1.0f};
    const float rhs[4] = {3.0f, 4.0f, 5.0f, 6.0f};
    float cuda_out[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float cpu_out[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    att1_backend *cuda_backend = NULL;
    size_t i = 0u;

    if (!att1_backend_cuda_available()) {
        return 0;
    }

    if (att1_backend_cuda_create(&cuda_backend) != ATT1_OK) {
        fputs("fallback: cuda backend creation failed\n", stderr);
        return -1;
    }

    /* Verify backend identity — must be "cuda", not "cpu-f32" */
    if ((cuda_backend->ops == NULL) ||
        (strcmp(cuda_backend->ops->name, "cuda") != 0)) {
        fputs("fallback: expected backend name \"cuda\"\n", stderr);
        att1_backend_destroy(cuda_backend);
        return -1;
    }

    /* Run matmul through the CUDA ops vtable */
    if (cuda_backend->ops->matmul_f32(cuda_backend,
                                      cuda_out, lhs, rhs,
                                      2u, 2u, 2u) != 0) {
        fputs("fallback: cuda matmul_f32 returned failure\n", stderr);
        att1_backend_destroy(cuda_backend);
        return -1;
    }

    /* CPU reference for comparison */
    if (att1_matmul_f32(cpu_out, lhs, rhs, 2u, 2u, 2u) != 0) {
        fputs("fallback: cpu matmul_f32 failed\n", stderr);
        att1_backend_destroy(cuda_backend);
        return -1;
    }

    for (i = 0u; i < 4u; i++) {
        if (!near_f32(cuda_out[i], cpu_out[i])) {
            fprintf(stderr,
                    "fallback: element %zu: cuda=%.6f cpu=%.6f\n",
                    i, (double)cuda_out[i], (double)cpu_out[i]);
            att1_backend_destroy(cuda_backend);
            return -1;
        }
    }

    att1_backend_destroy(cuda_backend);
    return 0;
}

/* ------------------------------------------------------------------ */
/* main                                                                */
/* ------------------------------------------------------------------ */

int main(void)
{
    if ((check_tiny_known() != 0) ||
        (check_larger_deterministic() != 0) ||
        (check_shape_handling() != 0) ||
        (check_cuda_unavailable() != 0) ||
        (check_no_silent_fallback() != 0)) {
        fputs("cuda_matmul test failed\n", stderr);
        return 1;
    }

    puts("cuda_matmul test passed");
    return 0;
}
