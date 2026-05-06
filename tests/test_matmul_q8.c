#include "att1_math.h"
#include "att1_quant.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#define Q8_TOLERANCE 0.035f

static int near_f32(float lhs, float rhs, float tolerance)
{
    return fabsf(lhs - rhs) <= tolerance;
}

static int check_tiny_hand_computed(void)
{
    const float lhs[2] = {3.0f, 4.0f};
    const float expected[2] = {-5.0f, 3.5f};
    float actual[2] = {0.0f, 0.0f};
    att1_q8_matrix weights;

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

    if (att1_matmul_q8xf32(actual, lhs, 1u, 2u, &weights) != 0) {
        fputs("q8 tiny matmul failed\n", stderr);
        att1_q8_matrix_free(&weights);
        return -1;
    }

    if (!near_f32(actual[0], expected[0], 0.000001f) ||
        !near_f32(actual[1], expected[1], 0.000001f)) {
        fputs("q8 tiny matmul value check failed\n", stderr);
        att1_q8_matrix_free(&weights);
        return -1;
    }

    att1_q8_matrix_free(&weights);
    return 0;
}

static int check_against_f32_reference(void)
{
    const float lhs[6] = {
        0.5f, -1.0f, 2.0f,
        1.5f, 0.25f, -0.75f
    };
    const float weights_out_in[6] = {
        0.75f, -0.50f, 1.25f,
        -1.00f, 0.25f, 0.50f
    };
    const float weights_in_out[6] = {
        0.75f, -1.00f,
        -0.50f, 0.25f,
        1.25f, 0.50f
    };
    float f32_out[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float q8_out[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    att1_q8_matrix q8;
    size_t i = 0u;

    memset(&q8, 0, sizeof(q8));
    if (att1_quantize_q8_per_row(&q8, weights_out_in, 2u, 3u) != 0) {
        return -1;
    }

    if ((att1_matmul_f32(f32_out, lhs, weights_in_out, 2u, 2u, 3u) != 0) ||
        (att1_matmul_q8xf32(q8_out, lhs, 2u, 3u, &q8) != 0)) {
        fputs("q8/f32 reference matmul failed\n", stderr);
        att1_q8_matrix_free(&q8);
        return -1;
    }

    for (i = 0u; i < 4u; i++) {
        if (!near_f32(q8_out[i], f32_out[i], Q8_TOLERANCE)) {
            fputs("q8/f32 tolerance check failed\n", stderr);
            att1_q8_matrix_free(&q8);
            return -1;
        }
    }

    att1_q8_matrix_free(&q8);
    return 0;
}

static int check_zero_matrix(void)
{
    const float lhs[4] = {1.0f, -2.0f, 3.0f, -4.0f};
    const float weights[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float actual[6] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    att1_q8_matrix q8;
    size_t i = 0u;

    memset(&q8, 0, sizeof(q8));
    if (att1_quantize_q8_per_row(&q8, weights, 3u, 2u) != 0) {
        return -1;
    }

    if (att1_matmul_q8xf32(actual, lhs, 2u, 2u, &q8) != 0) {
        fputs("q8 zero matmul failed\n", stderr);
        att1_q8_matrix_free(&q8);
        return -1;
    }

    for (i = 0u; i < 6u; i++) {
        if (actual[i] != 0.0f) {
            fputs("q8 zero matmul value check failed\n", stderr);
            att1_q8_matrix_free(&q8);
            return -1;
        }
    }

    att1_q8_matrix_free(&q8);
    return 0;
}

static int check_shape_mismatch(void)
{
    const float lhs[2] = {1.0f, 2.0f};
    float dst[1] = {0.0f};
    att1_q8_matrix q8;

    memset(&q8, 0, sizeof(q8));
    if (att1_q8_matrix_alloc(&q8, 1u, 3u) != 0) {
        return -1;
    }

    if (att1_matmul_q8xf32(dst, lhs, 1u, 2u, &q8) == 0) {
        fputs("q8 shape mismatch should fail\n", stderr);
        att1_q8_matrix_free(&q8);
        return -1;
    }

    if ((att1_matmul_q8xf32(NULL, lhs, 1u, 2u, &q8) == 0) ||
        (att1_matmul_q8xf32(dst, NULL, 1u, 2u, &q8) == 0) ||
        (att1_matmul_q8xf32(dst, lhs, 0u, 2u, &q8) == 0)) {
        fputs("q8 invalid argument should fail\n", stderr);
        att1_q8_matrix_free(&q8);
        return -1;
    }

    att1_q8_matrix_free(&q8);
    return 0;
}

int main(void)
{
    if ((check_tiny_hand_computed() != 0) ||
        (check_against_f32_reference() != 0) ||
        (check_zero_matrix() != 0) ||
        (check_shape_mismatch() != 0)) {
        fputs("matmul_q8 test failed\n", stderr);
        return 1;
    }

    puts("matmul_q8 test passed");
    return 0;
}
