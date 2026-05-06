#include "att1_quant.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

static int near_f32(float lhs, float rhs)
{
    return fabsf(lhs - rhs) < 0.000001f;
}

static int check_quantization_values(void)
{
    const float weights[12] = {
        -1.0f, 0.0f, 1.0f,
        -2.0f, 0.0f, 1.0f,
        0.0f, 0.0f, 0.0f,
        -1270.0f, 640.0f, 1270.0f
    };
    att1_q8_matrix matrix;
    size_t i = 0u;

    memset(&matrix, 0, sizeof(matrix));
    if (att1_quantize_q8_per_row(&matrix, weights, 4u, 3u) != 0) {
        fputs("q8 quantization failed\n", stderr);
        return -1;
    }

    if (!near_f32(matrix.scales[0], 1.0f / 127.0f) ||
        !near_f32(matrix.scales[1], 2.0f / 127.0f) ||
        !near_f32(matrix.scales[2], 1.0f) ||
        !near_f32(matrix.scales[3], 10.0f)) {
        fputs("q8 per-row scale check failed\n", stderr);
        att1_q8_matrix_free(&matrix);
        return -1;
    }

    if ((matrix.values[0] != -127) ||
        (matrix.values[1] != 0) ||
        (matrix.values[2] != 127) ||
        (matrix.values[3] != -127) ||
        (matrix.values[4] != 0) ||
        (matrix.values[5] != 64) ||
        (matrix.values[6] != 0) ||
        (matrix.values[7] != 0) ||
        (matrix.values[8] != 0) ||
        (matrix.values[9] != -127) ||
        (matrix.values[10] != 64) ||
        (matrix.values[11] != 127)) {
        fputs("q8 rounded value check failed\n", stderr);
        att1_q8_matrix_free(&matrix);
        return -1;
    }

    for (i = 0u; i < 12u; i++) {
        const int qvalue = (int)matrix.values[i];
        if ((qvalue < -127) || (qvalue > 127)) {
            fputs("q8 saturation range check failed\n", stderr);
            att1_q8_matrix_free(&matrix);
            return -1;
        }
    }

    att1_q8_matrix_free(&matrix);
    return 0;
}

static int check_ownership(void)
{
    att1_q8_matrix matrix;

    att1_q8_matrix_free(NULL);

    memset(&matrix, 0, sizeof(matrix));
    if (att1_q8_matrix_alloc(&matrix, 2u, 3u) != 0) {
        fputs("q8 allocation failed\n", stderr);
        return -1;
    }

    if ((matrix.rows != 2u) || (matrix.cols != 3u) ||
        (matrix.values == NULL) || (matrix.scales == NULL)) {
        fputs("q8 allocation metadata check failed\n", stderr);
        att1_q8_matrix_free(&matrix);
        return -1;
    }

    att1_q8_matrix_free(&matrix);
    if ((matrix.rows != 0u) || (matrix.cols != 0u) ||
        (matrix.values != NULL) || (matrix.scales != NULL)) {
        fputs("q8 free clear check failed\n", stderr);
        return -1;
    }

    return 0;
}

static int check_invalid_inputs(void)
{
    const float weights[1] = {1.0f};
    const float bad_weights[1] = {NAN};
    att1_q8_matrix matrix;

    memset(&matrix, 0, sizeof(matrix));
    if (att1_q8_matrix_alloc(&matrix, 0u, 1u) == 0) {
        fputs("q8 zero-row allocation should fail\n", stderr);
        att1_q8_matrix_free(&matrix);
        return -1;
    }

    if (att1_quantize_q8_per_row(NULL, weights, 1u, 1u) == 0) {
        fputs("q8 null matrix quantization should fail\n", stderr);
        return -1;
    }

    memset(&matrix, 0, sizeof(matrix));
    if (att1_quantize_q8_per_row(&matrix, bad_weights, 1u, 1u) == 0) {
        fputs("q8 nonfinite quantization should fail\n", stderr);
        att1_q8_matrix_free(&matrix);
        return -1;
    }

    return 0;
}

int main(void)
{
    if ((check_quantization_values() != 0) ||
        (check_ownership() != 0) ||
        (check_invalid_inputs() != 0)) {
        fputs("quant test failed\n", stderr);
        return 1;
    }

    puts("quant test passed");
    return 0;
}
