/*
 * test_cuda_matmul_q4.c — Milestone 87: CUDA q4xf32 matmul prototype
 *
 * Tests:
 *   1. Tiny hand-checkable q4 matrix/vector case.
 *   2. Deterministic medium q4 matrix/vector case: CUDA q4 vs CPU q4.
 *   3. Zero weight matrix produces zero output.
 *   4. CUDA q4 output matches CPU q4xf32 within documented tolerance.
 *   5. Invalid dimension arguments fail clearly.
 *   6. Invalid q4 metadata / group size fails clearly.
 *   7. Backend identity: cuda-q4 backend name must be "cuda-q4".
 *   8. Non-CUDA build: att1_backend_cuda_q4_create returns ATT1_ERR_UNSUPPORTED.
 *
 * Tolerance notes:
 *   - CUDA q4 vs CPU q4: 1e-4f (same dequant algorithm, only f32 rounding).
 *   - CPU q4 vs CPU f32: Q4_TOLERANCE = 0.35f (quantisation loss).
 *
 * Non-CUDA builds and machines without a GPU skip CUDA execution tests and
 * verify the unavailable-path instead.  All tests return 0 on success; the
 * binary exits 1 on any failure.
 */

#include "att1_backend.h"
#include "att1_quant.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

/* Tolerance for CUDA q4 vs CPU q4 — only floating-point rounding differs. */
#define Q4_CUDA_TOL 1e-4f
/* Tolerance for q4 vs reference f32 — quantisation loss. */
#define Q4_F32_TOL  0.35f

static int near_tol(float a, float b, float tol)
{
    return fabsf(a - b) <= tol;
}

/* ------------------------------------------------------------------ */
/* Test 1: tiny hand-checkable case (1×16, 2 output rows)              */
/* ------------------------------------------------------------------ */

/*
 * weights (2 rows × 16 cols, group_size=16):
 *   row 0: all 0.5  → quantised max_abs=0.5, scale=0.5/7
 *          dequantised ≈ 0.5 per element
 *   row 1: alternating +1 / -1 → scale=1/7
 *          dequantised ≈ ±1 per element (alternating)
 * lhs (1×16): all 1.0
 *   dst[0] = sum(1.0 * 0.5) × 16 ≈ 8.0
 *   dst[1] = sum(1.0 * alternating ±1) ≈ 0.0
 */
static int check_tiny_hand(void)
{
    float weights_f32[2 * 16];
    float lhs[16];
    float cpu_out[2] = {0.0f, 0.0f};
    float cuda_out[2] = {0.0f, 0.0f};
    att1_q4_matrix q4;
    att1_backend *backend = NULL;
    size_t i = 0u;

    for (i = 0u; i < 16u; i++) { weights_f32[i] = 0.5f; }
    for (i = 0u; i < 16u; i++) {
        weights_f32[16u + i] = (i % 2u == 0u) ? 1.0f : -1.0f;
    }
    for (i = 0u; i < 16u; i++) { lhs[i] = 1.0f; }

    memset(&q4, 0, sizeof(q4));
    if (att1_quantize_q4_per_group(&q4, weights_f32, 2u, 16u, 16u) != 0) {
        fputs("tiny: quantize failed\n", stderr);
        return -1;
    }

    /* CPU reference */
    if (att1_matmul_q4xf32(cpu_out, lhs, 1u, 16u, &q4) != 0) {
        fputs("tiny: cpu q4 matmul failed\n", stderr);
        att1_q4_matrix_free(&q4);
        return -1;
    }

    if (!near_tol(cpu_out[0], 8.0f, Q4_F32_TOL) ||
        !near_tol(cpu_out[1], 0.0f, Q4_F32_TOL)) {
        fprintf(stderr,
                "tiny: cpu_out[0]=%.6f (want ~8.0) cpu_out[1]=%.6f (want ~0.0)\n",
                (double)cpu_out[0], (double)cpu_out[1]);
        att1_q4_matrix_free(&q4);
        return -1;
    }

    if (!att1_backend_cuda_available()) {
        att1_q4_matrix_free(&q4);
        return 0;
    }

    if (att1_backend_cuda_q4_create(&backend) != ATT1_OK) {
        fputs("tiny: cuda-q4 backend creation failed\n", stderr);
        att1_q4_matrix_free(&q4);
        return -1;
    }

    if (backend->ops->matmul_q4xf32(backend,
                                    cuda_out,
                                    lhs,
                                    1u,
                                    16u,
                                    &q4) != 0) {
        fputs("tiny: cuda q4 matmul failed\n", stderr);
        att1_backend_destroy(backend);
        att1_q4_matrix_free(&q4);
        return -1;
    }

    for (i = 0u; i < 2u; i++) {
        if (!near_tol(cuda_out[i], cpu_out[i], Q4_CUDA_TOL)) {
            fprintf(stderr,
                    "tiny: element %zu cuda=%.6f cpu=%.6f\n",
                    i, (double)cuda_out[i], (double)cpu_out[i]);
            att1_backend_destroy(backend);
            att1_q4_matrix_free(&q4);
            return -1;
        }
    }

    att1_backend_destroy(backend);
    att1_q4_matrix_free(&q4);
    return 0;
}

/* ------------------------------------------------------------------ */
/* Test 2: deterministic medium case — 3×8 weight, 2-row lhs           */
/* ------------------------------------------------------------------ */

static int check_medium_deterministic(void)
{
    const size_t M = 2u;   /* lhs_rows */
    const size_t K = 32u;  /* lhs_cols / weight cols */
    const size_t N = 4u;   /* weight rows (output dim) */
    float lhs[64];         /* M * K */
    float weights_f32[128]; /* N * K */
    float cpu_out[8];       /* M * N */
    float cuda_out[8];
    att1_q4_matrix q4;
    att1_backend *backend = NULL;
    size_t i = 0u;

    memset(&q4, 0, sizeof(q4));

    for (i = 0u; i < M * K; i++) {
        const int v = (int)(i % 9u) - 4;
        lhs[i] = (float)v * 0.25f;
    }
    for (i = 0u; i < N * K; i++) {
        const int v = (int)((i * 3u + 1u) % 11u) - 5;
        weights_f32[i] = (float)v * 0.1875f;
    }

    if (att1_quantize_q4_per_group(&q4, weights_f32, N, K, 16u) != 0) {
        fputs("medium: quantize failed\n", stderr);
        return -1;
    }

    if (att1_matmul_q4xf32(cpu_out, lhs, M, K, &q4) != 0) {
        fputs("medium: cpu q4 matmul failed\n", stderr);
        att1_q4_matrix_free(&q4);
        return -1;
    }

    if (!att1_backend_cuda_available()) {
        att1_q4_matrix_free(&q4);
        return 0;
    }

    if (att1_backend_cuda_q4_create(&backend) != ATT1_OK) {
        fputs("medium: cuda-q4 backend creation failed\n", stderr);
        att1_q4_matrix_free(&q4);
        return -1;
    }

    if (backend->ops->matmul_q4xf32(backend,
                                    cuda_out,
                                    lhs,
                                    M,
                                    K,
                                    &q4) != 0) {
        fputs("medium: cuda q4 matmul failed\n", stderr);
        att1_backend_destroy(backend);
        att1_q4_matrix_free(&q4);
        return -1;
    }

    for (i = 0u; i < M * N; i++) {
        if (!near_tol(cuda_out[i], cpu_out[i], Q4_CUDA_TOL)) {
            fprintf(stderr,
                    "medium: element %zu cuda=%.6f cpu=%.6f\n",
                    i, (double)cuda_out[i], (double)cpu_out[i]);
            att1_backend_destroy(backend);
            att1_q4_matrix_free(&q4);
            return -1;
        }
    }

    att1_backend_destroy(backend);
    att1_q4_matrix_free(&q4);
    return 0;
}

/* ------------------------------------------------------------------ */
/* Test 3: zero weight matrix produces zero output                     */
/* ------------------------------------------------------------------ */

static int check_zero_weights(void)
{
    const size_t M = 2u;
    const size_t K = 32u;
    const size_t N = 3u;
    float lhs[64];
    float weights_f32[96]; /* N * K */
    float cuda_out[6];
    att1_q4_matrix q4;
    att1_backend *backend = NULL;
    size_t i = 0u;

    memset(&q4, 0, sizeof(q4));
    memset(weights_f32, 0, sizeof(weights_f32));
    for (i = 0u; i < M * K; i++) { lhs[i] = (float)(i + 1u) * 0.5f; }
    memset(cuda_out, 0, sizeof(cuda_out));

    if (att1_quantize_q4_per_group(&q4, weights_f32, N, K, 16u) != 0) {
        fputs("zero: quantize failed\n", stderr);
        return -1;
    }

    if (!att1_backend_cuda_available()) {
        att1_q4_matrix_free(&q4);
        return 0;
    }

    if (att1_backend_cuda_q4_create(&backend) != ATT1_OK) {
        fputs("zero: cuda-q4 backend creation failed\n", stderr);
        att1_q4_matrix_free(&q4);
        return -1;
    }

    if (backend->ops->matmul_q4xf32(backend,
                                    cuda_out,
                                    lhs,
                                    M,
                                    K,
                                    &q4) != 0) {
        fputs("zero: cuda q4 matmul failed\n", stderr);
        att1_backend_destroy(backend);
        att1_q4_matrix_free(&q4);
        return -1;
    }

    for (i = 0u; i < M * N; i++) {
        if (fabsf(cuda_out[i]) > Q4_CUDA_TOL) {
            fprintf(stderr,
                    "zero: element %zu = %.6f (expected ~0)\n",
                    i, (double)cuda_out[i]);
            att1_backend_destroy(backend);
            att1_q4_matrix_free(&q4);
            return -1;
        }
    }

    att1_backend_destroy(backend);
    att1_q4_matrix_free(&q4);
    return 0;
}

/* ------------------------------------------------------------------ */
/* Test 4: invalid dimension arguments fail clearly                    */
/* ------------------------------------------------------------------ */

static int check_invalid_dims(void)
{
    float dst[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const float lhs[32] = {0.0f};
    att1_q4_matrix q4;
    att1_backend *backend = NULL;

    memset(&q4, 0, sizeof(q4));
    if (att1_quantize_q4_per_group(&q4, lhs, 2u, 16u, 16u) != 0) {
        fputs("invalid_dims: setup quantize failed\n", stderr);
        return -1;
    }

    if (!att1_backend_cuda_available()) {
        att1_q4_matrix_free(&q4);
        return 0;
    }

    if (att1_backend_cuda_q4_create(&backend) != ATT1_OK) {
        fputs("invalid_dims: backend creation failed\n", stderr);
        att1_q4_matrix_free(&q4);
        return -1;
    }

    /* null dst */
    if (backend->ops->matmul_q4xf32(backend, NULL, lhs, 1u, 16u, &q4) == 0) {
        fputs("invalid_dims: null dst should fail\n", stderr);
        att1_backend_destroy(backend);
        att1_q4_matrix_free(&q4);
        return -1;
    }
    /* null lhs */
    if (backend->ops->matmul_q4xf32(backend, dst, NULL, 1u, 16u, &q4) == 0) {
        fputs("invalid_dims: null lhs should fail\n", stderr);
        att1_backend_destroy(backend);
        att1_q4_matrix_free(&q4);
        return -1;
    }
    /* null weights */
    if (backend->ops->matmul_q4xf32(backend, dst, lhs, 1u, 16u, NULL) == 0) {
        fputs("invalid_dims: null weights should fail\n", stderr);
        att1_backend_destroy(backend);
        att1_q4_matrix_free(&q4);
        return -1;
    }
    /* zero lhs_rows */
    if (backend->ops->matmul_q4xf32(backend, dst, lhs, 0u, 16u, &q4) == 0) {
        fputs("invalid_dims: zero lhs_rows should fail\n", stderr);
        att1_backend_destroy(backend);
        att1_q4_matrix_free(&q4);
        return -1;
    }
    /* zero lhs_cols */
    if (backend->ops->matmul_q4xf32(backend, dst, lhs, 1u, 0u, &q4) == 0) {
        fputs("invalid_dims: zero lhs_cols should fail\n", stderr);
        att1_backend_destroy(backend);
        att1_q4_matrix_free(&q4);
        return -1;
    }
    /* lhs_cols != weights->cols */
    if (backend->ops->matmul_q4xf32(backend, dst, lhs, 1u, 32u, &q4) == 0) {
        fputs("invalid_dims: mismatched cols should fail\n", stderr);
        att1_backend_destroy(backend);
        att1_q4_matrix_free(&q4);
        return -1;
    }

    att1_backend_destroy(backend);
    att1_q4_matrix_free(&q4);
    return 0;
}

/* ------------------------------------------------------------------ */
/* Test 5: invalid q4 metadata fails clearly                           */
/* ------------------------------------------------------------------ */

static int check_invalid_q4_metadata(void)
{
    float dst[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const float lhs[16] = {0.0f};
    att1_q4_matrix q4;
    att1_backend *backend = NULL;

    if (!att1_backend_cuda_available()) {
        return 0;
    }

    if (att1_backend_cuda_q4_create(&backend) != ATT1_OK) {
        fputs("invalid_q4: backend creation failed\n", stderr);
        return -1;
    }

    /* Zeroed q4 struct (invalid: zero rows/cols/packed) */
    memset(&q4, 0, sizeof(q4));
    if (backend->ops->matmul_q4xf32(backend, dst, lhs, 1u, 16u, &q4) == 0) {
        fputs("invalid_q4: zero q4 struct should fail\n", stderr);
        att1_backend_destroy(backend);
        return -1;
    }

    att1_backend_destroy(backend);
    return 0;
}

/* ------------------------------------------------------------------ */
/* Test 6: backend identity — name must be "cuda-q4"                   */
/* ------------------------------------------------------------------ */

static int check_backend_identity(void)
{
    att1_backend *backend = NULL;

    if (!att1_backend_cuda_available()) {
        return 0;
    }

    if (att1_backend_cuda_q4_create(&backend) != ATT1_OK) {
        fputs("identity: backend creation failed\n", stderr);
        return -1;
    }

    if ((backend->ops == NULL) ||
        (backend->ops->name == NULL) ||
        (strcmp(backend->ops->name, "cuda-q4") != 0)) {
        fprintf(stderr,
                "identity: expected name \"cuda-q4\", got \"%s\"\n",
                (backend->ops != NULL) ? backend->ops->name : "(null)");
        att1_backend_destroy(backend);
        return -1;
    }

    /* M88: inference ops must be populated — cuda-q4 is a full inference backend */
    if ((backend->ops->rmsnorm_f32 == NULL) ||
        (backend->ops->softmax_f32 == NULL) ||
        (backend->ops->rope_f32 == NULL) ||
        (backend->ops->ffn_swiglu_f32 == NULL)) {
        fputs("identity: inference ops should be non-NULL on cuda-q4 (M88)\n", stderr);
        att1_backend_destroy(backend);
        return -1;
    }

    att1_backend_destroy(backend);
    return 0;
}

/* ------------------------------------------------------------------ */
/* Test 7: non-CUDA build reports unsupported cleanly                  */
/* ------------------------------------------------------------------ */

static int check_cuda_unavailable(void)
{
    att1_backend *backend = NULL;
    att1_status_t status = ATT1_OK;

    if (att1_backend_cuda_available()) {
        return 0;
    }

    status = att1_backend_cuda_q4_create(&backend);
    if (status != ATT1_ERR_UNSUPPORTED) {
        fputs("unavail: expected ATT1_ERR_UNSUPPORTED from cuda-q4 create\n",
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
/* Test 8: CUDA q4 matches CPU q4 across a larger batch               */
/* ------------------------------------------------------------------ */

static int check_large_batch(void)
{
    const size_t M = 4u;
    const size_t K = 64u;
    const size_t N = 8u;
    float lhs[256];  /* M * K */
    float weights_f32[512]; /* N * K */
    float cpu_out[32]; /* M * N */
    float cuda_out[32];
    att1_q4_matrix q4;
    att1_backend *backend = NULL;
    size_t i = 0u;

    memset(&q4, 0, sizeof(q4));

    for (i = 0u; i < M * K; i++) {
        lhs[i] = ((float)(int)(i % 13u) - 6.0f) * 0.1f;
    }
    for (i = 0u; i < N * K; i++) {
        const int v = (int)((i * 7u + 2u) % 15u) - 7;
        weights_f32[i] = (float)v * 0.125f;
    }

    if (att1_quantize_q4_per_group(&q4, weights_f32, N, K, 32u) != 0) {
        fputs("large_batch: quantize failed\n", stderr);
        return -1;
    }

    if (att1_matmul_q4xf32(cpu_out, lhs, M, K, &q4) != 0) {
        fputs("large_batch: cpu q4 matmul failed\n", stderr);
        att1_q4_matrix_free(&q4);
        return -1;
    }

    if (!att1_backend_cuda_available()) {
        att1_q4_matrix_free(&q4);
        return 0;
    }

    if (att1_backend_cuda_q4_create(&backend) != ATT1_OK) {
        fputs("large_batch: cuda-q4 backend creation failed\n", stderr);
        att1_q4_matrix_free(&q4);
        return -1;
    }

    if (backend->ops->matmul_q4xf32(backend,
                                    cuda_out,
                                    lhs,
                                    M,
                                    K,
                                    &q4) != 0) {
        fputs("large_batch: cuda q4 matmul failed\n", stderr);
        att1_backend_destroy(backend);
        att1_q4_matrix_free(&q4);
        return -1;
    }

    for (i = 0u; i < M * N; i++) {
        if (!near_tol(cuda_out[i], cpu_out[i], Q4_CUDA_TOL)) {
            fprintf(stderr,
                    "large_batch: element %zu cuda=%.6f cpu=%.6f\n",
                    i, (double)cuda_out[i], (double)cpu_out[i]);
            att1_backend_destroy(backend);
            att1_q4_matrix_free(&q4);
            return -1;
        }
    }

    att1_backend_destroy(backend);
    att1_q4_matrix_free(&q4);
    return 0;
}

/* ------------------------------------------------------------------ */
/* main                                                                */
/* ------------------------------------------------------------------ */

int main(void)
{
    if ((check_tiny_hand() != 0) ||
        (check_medium_deterministic() != 0) ||
        (check_zero_weights() != 0) ||
        (check_invalid_dims() != 0) ||
        (check_invalid_q4_metadata() != 0) ||
        (check_backend_identity() != 0) ||
        (check_cuda_unavailable() != 0) ||
        (check_large_batch() != 0)) {
        fputs("cuda_matmul_q4 test failed\n", stderr);
        return 1;
    }

    puts("cuda_matmul_q4 test passed");
    return 0;
}
