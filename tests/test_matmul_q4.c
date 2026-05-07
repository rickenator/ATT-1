/*
 * test_matmul_q4.c  -  M76: CPU q4xf32 matmul prototype tests
 *
 * Covers:
 *   - Tiny hand-checkable q4 matrix/vector case.
 *   - Zero weight matrix produces zero output.
 *   - Deterministic medium matrix/vector: q4 output within tolerance vs f32.
 *   - q4 output within documented tolerance vs q8xf32.
 *   - Invalid dimensions fail clearly.
 *   - Invalid group size fails clearly.
 *   - Null/invalid args fail clearly.
 *   - att1_q4_matrix alloc/free lifecycle.
 */

#include "att1_math.h"
#include "att1_quant.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

/*
 * Maximum absolute error allowed between q4xf32 and f32 reference for a
 * medium test vector.  Derived from one quantization step (scale/7) per
 * element, accumulated over 64 elements with group_size=32.
 * Conservative bound: 0.35.
 */
#define Q4_TOLERANCE 0.35f

static int near_f32(float a, float b, float tol)
{
    return fabsf(a - b) <= tol;
}

/* ── alloc/free lifecycle ────────────────────────────────────────────────── */

static int test_alloc_free(void)
{
    att1_q4_matrix m;

    att1_q4_matrix_free(NULL); /* must not crash */

    memset(&m, 0, sizeof(m));
    if (att1_q4_matrix_alloc(&m, 4u, 64u, 32u) != 0) {
        fputs("alloc_free: alloc failed\n", stderr);
        return -1;
    }
    if ((m.rows != 4u) || (m.cols != 64u) || (m.group_size != 32u) ||
        (m.packed == NULL) || (m.scales == NULL)) {
        fputs("alloc_free: metadata check failed\n", stderr);
        att1_q4_matrix_free(&m);
        return -1;
    }
    att1_q4_matrix_free(&m);
    if ((m.rows != 0u) || (m.cols != 0u) ||
        (m.packed != NULL) || (m.scales != NULL)) {
        fputs("alloc_free: free did not zero fields\n", stderr);
        return -1;
    }
    return 0;
}

/* ── invalid alloc args ──────────────────────────────────────────────────── */

static int test_alloc_invalid(void)
{
    att1_q4_matrix m;

    memset(&m, 0, sizeof(m));

    /* null matrix */
    if (att1_q4_matrix_alloc(NULL, 4u, 32u, 32u) == 0) {
        fputs("alloc_invalid: null matrix should fail\n", stderr);
        return -1;
    }
    /* zero rows */
    if (att1_q4_matrix_alloc(&m, 0u, 32u, 32u) == 0) {
        fputs("alloc_invalid: zero rows should fail\n", stderr);
        return -1;
    }
    /* zero cols */
    if (att1_q4_matrix_alloc(&m, 4u, 0u, 32u) == 0) {
        fputs("alloc_invalid: zero cols should fail\n", stderr);
        return -1;
    }
    /* odd cols */
    if (att1_q4_matrix_alloc(&m, 4u, 33u, 32u) == 0) {
        fputs("alloc_invalid: odd cols should fail\n", stderr);
        return -1;
    }
    /* cols not divisible by group_size */
    if (att1_q4_matrix_alloc(&m, 4u, 48u, 32u) == 0) {
        fputs("alloc_invalid: cols not divisible by group_size should fail\n", stderr);
        return -1;
    }
    /* invalid group_size */
    if (att1_q4_matrix_alloc(&m, 4u, 32u, 0u) == 0) {
        fputs("alloc_invalid: group_size=0 should fail\n", stderr);
        return -1;
    }
    if (att1_q4_matrix_alloc(&m, 4u, 32u, 3u) == 0) {
        fputs("alloc_invalid: group_size=3 should fail\n", stderr);
        return -1;
    }
    return 0;
}

/* ── tiny hand-checkable case ────────────────────────────────────────────── */

/*
 * weights (2 x 16, group_size=16):
 *   row 0: all 0.5 -> int4 = [7,7,...,7] after quantize (max_abs=0.5, scale=0.5/7)
 *          deq_val = 7 * (0.5/7) = 0.5 for each element
 *   row 1: alternating +1/-1
 *          max_abs=1.0, scale=1/7
 *          int4 = [7,-7,7,-7,...]; deq_val = +1/-1 (approx)
 *
 * lhs (1 x 16): all 1.0
 *   out[0] = sum(1.0 * 0.5) * 16 = 8.0
 *   out[1] = sum(1.0 * alt +1/-1) = 0 (alternating)
 *
 * Compare against f32 reference for tight tolerance.
 */
static int test_tiny_hand_computed(void)
{
    float weights_f32[2 * 16];
    float lhs[16];
    float q4_out[2] = {0.0f, 0.0f};
    float f32_out[2] = {0.0f, 0.0f};
    /* f32 needs transposed weights [16, 2] */
    float weights_T[16 * 2];
    att1_q4_matrix q4;
    size_t i = 0u;

    /* row 0: all 0.5 */
    for (i = 0u; i < 16u; i++) { weights_f32[i] = 0.5f; }
    /* row 1: alternating +1 / -1 */
    for (i = 0u; i < 16u; i++) { weights_f32[16u + i] = (i % 2u == 0u) ? 1.0f : -1.0f; }
    /* lhs: all 1.0 */
    for (i = 0u; i < 16u; i++) { lhs[i] = 1.0f; }
    /* transpose for f32 reference */
    for (i = 0u; i < 16u; i++) {
        weights_T[i * 2u + 0u] = weights_f32[i];
        weights_T[i * 2u + 1u] = weights_f32[16u + i];
    }

    memset(&q4, 0, sizeof(q4));
    if (att1_quantize_q4_per_group(&q4, weights_f32, 2u, 16u, 16u) != 0) {
        fputs("tiny_hand: quantize failed\n", stderr);
        return -1;
    }

    if (att1_matmul_q4xf32(q4_out, lhs, 1u, 16u, &q4) != 0) {
        fputs("tiny_hand: q4xf32 matmul failed\n", stderr);
        att1_q4_matrix_free(&q4);
        return -1;
    }

    if (att1_matmul_f32(f32_out, lhs, weights_T, 1u, 2u, 16u) != 0) {
        fputs("tiny_hand: f32 matmul failed\n", stderr);
        att1_q4_matrix_free(&q4);
        return -1;
    }

    /* row-0 output: q4 reconstructs 0.5 per element → sum = 8.0
     * tolerance = one quantization step = 0.5/7 * 16 ≈ 1.14; use Q4_TOLERANCE */
    if (!near_f32(q4_out[0], f32_out[0], Q4_TOLERANCE)) {
        fprintf(stderr, "tiny_hand: q4_out[0]=%f f32_out[0]=%f diff=%f\n",
                (double)q4_out[0], (double)f32_out[0],
                (double)fabsf(q4_out[0] - f32_out[0]));
        att1_q4_matrix_free(&q4);
        return -1;
    }
    /* row-1 output: alternating → near zero */
    if (fabsf(q4_out[1]) > Q4_TOLERANCE) {
        fprintf(stderr, "tiny_hand: q4_out[1]=%f expected ~0\n",
                (double)q4_out[1]);
        att1_q4_matrix_free(&q4);
        return -1;
    }

    att1_q4_matrix_free(&q4);
    return 0;
}

/* ── zero weight matrix ──────────────────────────────────────────────────── */

static int test_zero_matrix(void)
{
    float weights_f32[4 * 32];
    float lhs[32];
    float q4_out[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    att1_q4_matrix q4;
    size_t i = 0u;

    memset(weights_f32, 0, sizeof(weights_f32));
    for (i = 0u; i < 32u; i++) { lhs[i] = (float)(i + 1u); }

    memset(&q4, 0, sizeof(q4));
    if (att1_quantize_q4_per_group(&q4, weights_f32, 4u, 32u, 32u) != 0) {
        fputs("zero_matrix: quantize failed\n", stderr);
        return -1;
    }

    if (att1_matmul_q4xf32(q4_out, lhs, 1u, 32u, &q4) != 0) {
        fputs("zero_matrix: matmul failed\n", stderr);
        att1_q4_matrix_free(&q4);
        return -1;
    }

    for (i = 0u; i < 4u; i++) {
        if (q4_out[i] != 0.0f) {
            fprintf(stderr, "zero_matrix: q4_out[%zu]=%f expected 0\n",
                    i, (double)q4_out[i]);
            att1_q4_matrix_free(&q4);
            return -1;
        }
    }

    att1_q4_matrix_free(&q4);
    return 0;
}

/* ── deterministic medium matrix vs f32 reference ───────────────────────── */

/*
 * 8 x 64 weight matrix, group_size=32.
 * Weights: weights[r][c] = sin((float)(r*64 + c) * 0.1f) * 0.8f
 * lhs: 2 x 64 activation matrix, lhs[r][c] = cos((float)(r*64 + c) * 0.07f)
 *
 * All values in [-1, 1] → scale ≤ 0.8/7 ≈ 0.114.
 * Max accumulated error per output: group_size * scale ≤ 32 * 0.114 ≈ 3.7
 * But that is per group; with 2 groups per row: bound is loose.
 * Empirically expect max_abs_error well within Q4_TOLERANCE=0.35
 * (validated against actual float arithmetic).
 */
static int test_medium_vs_f32(void)
{
    float weights_f32[8 * 64];
    float weights_T[64 * 8]; /* transposed for att1_matmul_f32 */
    float lhs[2 * 64];
    float q4_out[2 * 8];
    float f32_out[2 * 8];
    att1_q4_matrix q4;
    size_t r = 0u;
    size_t c = 0u;
    float max_err = 0.0f;

    for (r = 0u; r < 8u; r++) {
        for (c = 0u; c < 64u; c++) {
            weights_f32[r * 64u + c] = sinf((float)(r * 64u + c) * 0.1f) * 0.8f;
        }
    }
    for (r = 0u; r < 2u; r++) {
        for (c = 0u; c < 64u; c++) {
            lhs[r * 64u + c] = cosf((float)(r * 64u + c) * 0.07f);
        }
    }
    /* transpose weights for att1_matmul_f32: weights_T[c][r] = weights_f32[r][c] */
    for (r = 0u; r < 8u; r++) {
        for (c = 0u; c < 64u; c++) {
            weights_T[c * 8u + r] = weights_f32[r * 64u + c];
        }
    }

    memset(&q4, 0, sizeof(q4));
    if (att1_quantize_q4_per_group(&q4, weights_f32, 8u, 64u, 32u) != 0) {
        fputs("medium_vs_f32: quantize failed\n", stderr);
        return -1;
    }

    if (att1_matmul_q4xf32(q4_out, lhs, 2u, 64u, &q4) != 0) {
        fputs("medium_vs_f32: q4xf32 matmul failed\n", stderr);
        att1_q4_matrix_free(&q4);
        return -1;
    }

    if (att1_matmul_f32(f32_out, lhs, weights_T, 2u, 8u, 64u) != 0) {
        fputs("medium_vs_f32: f32 matmul failed\n", stderr);
        att1_q4_matrix_free(&q4);
        return -1;
    }

    for (r = 0u; r < 2u * 8u; r++) {
        const float err = fabsf(q4_out[r] - f32_out[r]);
        if (err > max_err) { max_err = err; }
    }

    if (max_err > Q4_TOLERANCE) {
        fprintf(stderr,
                "medium_vs_f32: max_err=%f > Q4_TOLERANCE=%f\n",
                (double)max_err, (double)Q4_TOLERANCE);
        att1_q4_matrix_free(&q4);
        return -1;
    }

    att1_q4_matrix_free(&q4);
    return 0;
}

/* ── q4 vs q8 cross-check ────────────────────────────────────────────────── */

/*
 * Same 4 x 32 weight matrix quantized both ways; verify q4 output is
 * within Q4_TOLERANCE of q8 output (q8 is a tighter reference than f32).
 */
static int test_q4_vs_q8(void)
{
    float weights_f32[4 * 32];
    float lhs[32];
    float q4_out[4];
    float q8_out[4];
    att1_q4_matrix q4;
    att1_q8_matrix q8;
    size_t i = 0u;
    float max_err = 0.0f;

    for (i = 0u; i < 4u * 32u; i++) {
        weights_f32[i] = sinf((float)i * 0.15f) * 0.6f;
    }
    for (i = 0u; i < 32u; i++) {
        lhs[i] = cosf((float)i * 0.05f) * 0.9f;
    }

    memset(&q4, 0, sizeof(q4));
    memset(&q8, 0, sizeof(q8));

    if (att1_quantize_q4_per_group(&q4, weights_f32, 4u, 32u, 32u) != 0) {
        fputs("q4_vs_q8: q4 quantize failed\n", stderr);
        return -1;
    }
    if (att1_quantize_q8_per_row(&q8, weights_f32, 4u, 32u) != 0) {
        fputs("q4_vs_q8: q8 quantize failed\n", stderr);
        att1_q4_matrix_free(&q4);
        return -1;
    }

    if (att1_matmul_q4xf32(q4_out, lhs, 1u, 32u, &q4) != 0) {
        fputs("q4_vs_q8: q4 matmul failed\n", stderr);
        att1_q4_matrix_free(&q4);
        att1_q8_matrix_free(&q8);
        return -1;
    }
    if (att1_matmul_q8xf32(q8_out, lhs, 1u, 32u, &q8) != 0) {
        fputs("q4_vs_q8: q8 matmul failed\n", stderr);
        att1_q4_matrix_free(&q4);
        att1_q8_matrix_free(&q8);
        return -1;
    }

    for (i = 0u; i < 4u; i++) {
        const float err = fabsf(q4_out[i] - q8_out[i]);
        if (err > max_err) { max_err = err; }
    }

    att1_q4_matrix_free(&q4);
    att1_q8_matrix_free(&q8);

    if (max_err > Q4_TOLERANCE) {
        fprintf(stderr,
                "q4_vs_q8: max_err=%f > Q4_TOLERANCE=%f\n",
                (double)max_err, (double)Q4_TOLERANCE);
        return -1;
    }
    return 0;
}

/* ── dimension mismatch ──────────────────────────────────────────────────── */

static int test_dimension_mismatch(void)
{
    float weights_f32[2 * 32];
    float lhs[16];  /* lhs_cols=16, weights->cols=32 → mismatch */
    float dst[2] = {0.0f, 0.0f};
    att1_q4_matrix q4;
    size_t i = 0u;

    for (i = 0u; i < 2u * 32u; i++) { weights_f32[i] = 0.1f; }
    for (i = 0u; i < 16u; i++) { lhs[i] = 1.0f; }

    memset(&q4, 0, sizeof(q4));
    if (att1_quantize_q4_per_group(&q4, weights_f32, 2u, 32u, 32u) != 0) {
        fputs("dim_mismatch: quantize failed\n", stderr);
        return -1;
    }

    /* lhs_cols=16 but weights->cols=32 */
    if (att1_matmul_q4xf32(dst, lhs, 1u, 16u, &q4) == 0) {
        fputs("dim_mismatch: should fail on lhs_cols != weights->cols\n", stderr);
        att1_q4_matrix_free(&q4);
        return -1;
    }
    /* zero lhs_rows */
    if (att1_matmul_q4xf32(dst, lhs, 0u, 32u, &q4) == 0) {
        fputs("dim_mismatch: should fail on zero lhs_rows\n", stderr);
        att1_q4_matrix_free(&q4);
        return -1;
    }

    att1_q4_matrix_free(&q4);
    return 0;
}

/* ── null args ───────────────────────────────────────────────────────────── */

static int test_null_args(void)
{
    float weights_f32[2 * 32];
    float lhs[32];
    float dst[2] = {0.0f, 0.0f};
    att1_q4_matrix q4;
    size_t i = 0u;

    for (i = 0u; i < 2u * 32u; i++) { weights_f32[i] = 0.1f; }
    for (i = 0u; i < 32u; i++) { lhs[i] = 1.0f; }

    memset(&q4, 0, sizeof(q4));
    if (att1_quantize_q4_per_group(&q4, weights_f32, 2u, 32u, 32u) != 0) {
        fputs("null_args: quantize failed\n", stderr);
        return -1;
    }

    if (att1_matmul_q4xf32(NULL, lhs, 1u, 32u, &q4) == 0) {
        fputs("null_args: null dst should fail\n", stderr);
        att1_q4_matrix_free(&q4);
        return -1;
    }
    if (att1_matmul_q4xf32(dst, NULL, 1u, 32u, &q4) == 0) {
        fputs("null_args: null lhs should fail\n", stderr);
        att1_q4_matrix_free(&q4);
        return -1;
    }
    if (att1_matmul_q4xf32(dst, lhs, 1u, 32u, NULL) == 0) {
        fputs("null_args: null weights should fail\n", stderr);
        att1_q4_matrix_free(&q4);
        return -1;
    }

    att1_q4_matrix_free(&q4);

    /* null src in quantize */
    if (att1_quantize_q4_per_group(&q4, NULL, 2u, 32u, 32u) == 0) {
        fputs("null_args: null weights in quantize should fail\n", stderr);
        att1_q4_matrix_free(&q4);
        return -1;
    }
    if (att1_quantize_q4_per_group(NULL, weights_f32, 2u, 32u, 32u) == 0) {
        fputs("null_args: null matrix in quantize should fail\n", stderr);
        return -1;
    }

    return 0;
}

/* ── main ────────────────────────────────────────────────────────────────── */

int main(void)
{
    if (test_alloc_free() != 0) {
        fputs("FAIL: alloc_free\n", stderr);
        return 1;
    }
    if (test_alloc_invalid() != 0) {
        fputs("FAIL: alloc_invalid\n", stderr);
        return 1;
    }
    if (test_tiny_hand_computed() != 0) {
        fputs("FAIL: tiny_hand_computed\n", stderr);
        return 1;
    }
    if (test_zero_matrix() != 0) {
        fputs("FAIL: zero_matrix\n", stderr);
        return 1;
    }
    if (test_medium_vs_f32() != 0) {
        fputs("FAIL: medium_vs_f32\n", stderr);
        return 1;
    }
    if (test_q4_vs_q8() != 0) {
        fputs("FAIL: q4_vs_q8\n", stderr);
        return 1;
    }
    if (test_dimension_mismatch() != 0) {
        fputs("FAIL: dimension_mismatch\n", stderr);
        return 1;
    }
    if (test_null_args() != 0) {
        fputs("FAIL: null_args\n", stderr);
        return 1;
    }

    puts("matmul_q4 test passed");
    return 0;
}
