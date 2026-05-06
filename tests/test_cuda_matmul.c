/*
 * test_cuda_matmul.c — Milestone 14: CUDA f32 matmul validation
 *
 * Tests:
 *   1. Tiny known 2×3 × 3×1 case with hand-checkable expected values.
 *   2. Larger deterministic 4×8 × 8×4 case compared to CPU f32 reference.
 *   3. Shape handling: NULL/zero arguments fail cleanly on both CPU and CUDA.
 *   4. CUDA unavailable: att1_backend_cuda_create returns ATT1_ERR_UNSUPPORTED.
 *   5. No silent fallback: CUDA backend name is "cuda", not "cpu-f32".
 *   6. CUDA q8xf32 matmul matches CPU q8xf32 and CPU f32 references.
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
#define Q8_CUDA_TOL 1e-3f
#define Q8_F32_TOL 0.035f

static int near_f32(float a, float b)
{
    return fabsf(a - b) < MATMUL_TOL;
}

static int near_tol(float a, float b, float tolerance)
{
    return fabsf(a - b) <= tolerance;
}

static void transpose_out_in_to_in_out(float *dst,
                                       const float *src,
                                       size_t rows,
                                       size_t cols)
{
    size_t row = 0u;
    size_t col = 0u;

    for (row = 0u; row < rows; row++) {
        for (col = 0u; col < cols; col++) {
            dst[(col * rows) + row] = src[(row * cols) + col];
        }
    }
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

static int check_cuda_q8_tiny_known(void)
{
    const float lhs[2] = {3.0f, 4.0f};
    const float expected[2] = {-5.0f, 3.5f};
    float cpu_out[2] = {0.0f, 0.0f};
    float cuda_out[2] = {0.0f, 0.0f};
    att1_q8_matrix weights;
    att1_backend *backend = NULL;
    size_t i = 0u;

    memset(&weights, 0, sizeof(weights));
    if (att1_q8_matrix_alloc(&weights, 2u, 2u) != 0) {
        return -1;
    }

    weights.scales[0] = 0.5f;
    weights.values[0] = 2;
    weights.values[1] = -4;
    weights.scales[1] = 0.25f;
    weights.values[2] = -6;
    weights.values[3] = 8;

    if (att1_matmul_q8xf32(cpu_out, lhs, 1u, 2u, &weights) != 0) {
        fputs("q8 tiny: cpu q8 matmul failed\n", stderr);
        att1_q8_matrix_free(&weights);
        return -1;
    }

    if (!att1_backend_cuda_available()) {
        att1_q8_matrix_free(&weights);
        return 0;
    }

    if (att1_backend_cuda_create(&backend) != ATT1_OK) {
        fputs("q8 tiny: cuda backend creation failed\n", stderr);
        att1_q8_matrix_free(&weights);
        return -1;
    }

    if (backend->ops->matmul_q8xf32(backend,
                                    cuda_out,
                                    lhs,
                                    1u,
                                    2u,
                                    &weights) != 0) {
        fputs("q8 tiny: cuda q8 matmul failed\n", stderr);
        att1_backend_destroy(backend);
        att1_q8_matrix_free(&weights);
        return -1;
    }

    for (i = 0u; i < 2u; i++) {
        if (!near_tol(cpu_out[i], expected[i], 0.000001f) ||
            !near_tol(cuda_out[i], cpu_out[i], Q8_CUDA_TOL)) {
            fprintf(stderr,
                    "q8 tiny: element %zu cuda=%.6f cpu=%.6f expected=%.6f\n",
                    i,
                    (double)cuda_out[i],
                    (double)cpu_out[i],
                    (double)expected[i]);
            att1_backend_destroy(backend);
            att1_q8_matrix_free(&weights);
            return -1;
        }
    }

    att1_backend_destroy(backend);
    att1_q8_matrix_free(&weights);
    return 0;
}

static int check_cuda_q8_medium_deterministic(void)
{
    const size_t M = 3u;
    const size_t K = 8u;
    const size_t N = 5u;
    float lhs[24];
    float weights_out_in[40];
    float weights_in_out[40];
    float cpu_q8_out[15];
    float cuda_q8_out[15];
    float cpu_f32_out[15];
    att1_q8_matrix q8;
    att1_backend *backend = NULL;
    size_t i = 0u;

    memset(&q8, 0, sizeof(q8));
    for (i = 0u; i < M * K; i++) {
        const int centered = (int)(i % 7u) - 3;
        lhs[i] = (float)centered * 0.375f + (float)(i / K) * 0.125f;
    }
    for (i = 0u; i < N * K; i++) {
        const int centered = (int)((i * 5u + 3u) % 11u) - 5;
        weights_out_in[i] = (float)centered * 0.1875f;
    }
    transpose_out_in_to_in_out(weights_in_out, weights_out_in, N, K);

    if (att1_quantize_q8_per_row(&q8, weights_out_in, N, K) != 0) {
        return -1;
    }

    if ((att1_matmul_q8xf32(cpu_q8_out, lhs, M, K, &q8) != 0) ||
        (att1_matmul_f32(cpu_f32_out, lhs, weights_in_out, M, N, K) != 0)) {
        fputs("q8 medium: cpu references failed\n", stderr);
        att1_q8_matrix_free(&q8);
        return -1;
    }

    if (!att1_backend_cuda_available()) {
        att1_q8_matrix_free(&q8);
        return 0;
    }

    if (att1_backend_cuda_create(&backend) != ATT1_OK) {
        fputs("q8 medium: cuda backend creation failed\n", stderr);
        att1_q8_matrix_free(&q8);
        return -1;
    }

    if (backend->ops->matmul_q8xf32(backend,
                                    cuda_q8_out,
                                    lhs,
                                    M,
                                    K,
                                    &q8) != 0) {
        fputs("q8 medium: cuda q8 matmul failed\n", stderr);
        att1_backend_destroy(backend);
        att1_q8_matrix_free(&q8);
        return -1;
    }

    for (i = 0u; i < M * N; i++) {
        if (!near_tol(cuda_q8_out[i], cpu_q8_out[i], Q8_CUDA_TOL) ||
            !near_tol(cuda_q8_out[i], cpu_f32_out[i], Q8_F32_TOL)) {
            fprintf(stderr,
                    "q8 medium: element %zu cuda=%.6f cpu_q8=%.6f cpu_f32=%.6f\n",
                    i,
                    (double)cuda_q8_out[i],
                    (double)cpu_q8_out[i],
                    (double)cpu_f32_out[i]);
            att1_backend_destroy(backend);
            att1_q8_matrix_free(&q8);
            return -1;
        }
    }

    att1_backend_destroy(backend);
    att1_q8_matrix_free(&q8);
    return 0;
}

static int check_cuda_q8_zero_scale_row(void)
{
    const float lhs[6] = {
        1.0f, -2.0f, 3.0f,
        -4.0f, 5.0f, -6.0f
    };
    float cpu_out[6] = {0.0f};
    float cuda_out[6] = {0.0f};
    att1_q8_matrix weights;
    att1_backend *backend = NULL;
    size_t i = 0u;

    memset(&weights, 0, sizeof(weights));
    if (att1_q8_matrix_alloc(&weights, 3u, 3u) != 0) {
        return -1;
    }

    weights.scales[0] = 0.5f;
    weights.values[0] = 2;
    weights.values[1] = -4;
    weights.values[2] = 6;
    weights.scales[1] = 0.0f;
    weights.values[3] = 127;
    weights.values[4] = -127;
    weights.values[5] = 64;
    weights.scales[2] = 0.25f;
    weights.values[6] = -8;
    weights.values[7] = 4;
    weights.values[8] = 12;

    if (att1_matmul_q8xf32(cpu_out, lhs, 2u, 3u, &weights) != 0) {
        fputs("q8 zero-scale: cpu q8 matmul failed\n", stderr);
        att1_q8_matrix_free(&weights);
        return -1;
    }

    if ((cpu_out[1] != 0.0f) || (cpu_out[4] != 0.0f)) {
        fputs("q8 zero-scale: CPU zero-scale row was not zero\n", stderr);
        att1_q8_matrix_free(&weights);
        return -1;
    }

    if (!att1_backend_cuda_available()) {
        att1_q8_matrix_free(&weights);
        return 0;
    }

    if (att1_backend_cuda_create(&backend) != ATT1_OK) {
        fputs("q8 zero-scale: cuda backend creation failed\n", stderr);
        att1_q8_matrix_free(&weights);
        return -1;
    }

    if (backend->ops->matmul_q8xf32(backend,
                                    cuda_out,
                                    lhs,
                                    2u,
                                    3u,
                                    &weights) != 0) {
        fputs("q8 zero-scale: cuda q8 matmul failed\n", stderr);
        att1_backend_destroy(backend);
        att1_q8_matrix_free(&weights);
        return -1;
    }

    for (i = 0u; i < 6u; i++) {
        if (!near_tol(cuda_out[i], cpu_out[i], Q8_CUDA_TOL)) {
            fprintf(stderr,
                    "q8 zero-scale: element %zu cuda=%.6f cpu=%.6f\n",
                    i,
                    (double)cuda_out[i],
                    (double)cpu_out[i]);
            att1_backend_destroy(backend);
            att1_q8_matrix_free(&weights);
            return -1;
        }
    }

    att1_backend_destroy(backend);
    att1_q8_matrix_free(&weights);
    return 0;
}

static int check_cuda_q8_saturation_edges(void)
{
    const float lhs[6] = {
        1.0f, -1.0f, 0.5f,
        -0.75f, 2.0f, -1.5f
    };
    float cpu_out[4] = {0.0f};
    float cuda_out[4] = {0.0f};
    att1_q8_matrix weights;
    att1_backend *backend = NULL;
    size_t i = 0u;

    memset(&weights, 0, sizeof(weights));
    if (att1_q8_matrix_alloc(&weights, 2u, 3u) != 0) {
        return -1;
    }

    weights.scales[0] = 0.125f;
    weights.values[0] = 127;
    weights.values[1] = -127;
    weights.values[2] = 0;
    weights.scales[1] = 0.25f;
    weights.values[3] = -127;
    weights.values[4] = 127;
    weights.values[5] = 127;

    if (att1_matmul_q8xf32(cpu_out, lhs, 2u, 3u, &weights) != 0) {
        fputs("q8 saturation: cpu q8 matmul failed\n", stderr);
        att1_q8_matrix_free(&weights);
        return -1;
    }

    if (!att1_backend_cuda_available()) {
        att1_q8_matrix_free(&weights);
        return 0;
    }

    if (att1_backend_cuda_create(&backend) != ATT1_OK) {
        fputs("q8 saturation: cuda backend creation failed\n", stderr);
        att1_q8_matrix_free(&weights);
        return -1;
    }

    if (backend->ops->matmul_q8xf32(backend,
                                    cuda_out,
                                    lhs,
                                    2u,
                                    3u,
                                    &weights) != 0) {
        fputs("q8 saturation: cuda q8 matmul failed\n", stderr);
        att1_backend_destroy(backend);
        att1_q8_matrix_free(&weights);
        return -1;
    }

    for (i = 0u; i < 4u; i++) {
        if (!near_tol(cuda_out[i], cpu_out[i], Q8_CUDA_TOL)) {
            fprintf(stderr,
                    "q8 saturation: element %zu cuda=%.6f cpu=%.6f\n",
                    i,
                    (double)cuda_out[i],
                    (double)cpu_out[i]);
            att1_backend_destroy(backend);
            att1_q8_matrix_free(&weights);
            return -1;
        }
    }

    att1_backend_destroy(backend);
    att1_q8_matrix_free(&weights);
    return 0;
}

static int check_cuda_q8_no_silent_fallback(void)
{
    const float lhs[2] = {1.0f, -1.0f};
    float cuda_out[1] = {0.0f};
    att1_q8_matrix weights;
    att1_backend *backend = NULL;

    memset(&weights, 0, sizeof(weights));
    if (att1_q8_matrix_alloc(&weights, 1u, 2u) != 0) {
        return -1;
    }
    weights.scales[0] = 0.5f;
    weights.values[0] = 4;
    weights.values[1] = -2;

    if (!att1_backend_cuda_available()) {
        att1_q8_matrix_free(&weights);
        return 0;
    }

    if (att1_backend_cuda_create(&backend) != ATT1_OK) {
        fputs("q8 fallback: cuda backend creation failed\n", stderr);
        att1_q8_matrix_free(&weights);
        return -1;
    }

    if ((backend->ops == NULL) ||
        (backend->ops->name == NULL) ||
        (strcmp(backend->ops->name, "cuda") != 0)) {
        fputs("q8 fallback: expected backend name \"cuda\"\n", stderr);
        att1_backend_destroy(backend);
        att1_q8_matrix_free(&weights);
        return -1;
    }

    if (backend->ops->matmul_q8xf32(backend,
                                    cuda_out,
                                    lhs,
                                    1u,
                                    2u,
                                    &weights) != 0) {
        fputs("q8 fallback: cuda q8 matmul failed\n", stderr);
        att1_backend_destroy(backend);
        att1_q8_matrix_free(&weights);
        return -1;
    }

    if (!near_tol(cuda_out[0], 3.0f, Q8_CUDA_TOL)) {
        fprintf(stderr,
                "q8 fallback: cuda=%.6f expected=3.000000\n",
                (double)cuda_out[0]);
        att1_backend_destroy(backend);
        att1_q8_matrix_free(&weights);
        return -1;
    }

    att1_backend_destroy(backend);
    att1_q8_matrix_free(&weights);
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
        (check_no_silent_fallback() != 0) ||
        (check_cuda_q8_tiny_known() != 0) ||
        (check_cuda_q8_medium_deterministic() != 0) ||
        (check_cuda_q8_zero_scale_row() != 0) ||
        (check_cuda_q8_saturation_edges() != 0) ||
        (check_cuda_q8_no_silent_fallback() != 0)) {
        fputs("cuda_matmul test failed\n", stderr);
        return 1;
    }

    puts("cuda_matmul test passed");
    return 0;
}
