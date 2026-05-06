#include "att1_quant.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

static int att1_mul_size(size_t lhs, size_t rhs, size_t *out)
{
    if (out == NULL) {
        return -1;
    }

    if ((lhs != 0u) && (rhs > (((size_t)-1) / lhs))) {
        return -1;
    }

    *out = lhs * rhs;
    return 0;
}

int att1_q8_matrix_alloc(att1_q8_matrix *matrix,
                         size_t rows,
                         size_t cols)
{
    size_t value_count = 0u;

    if ((matrix == NULL) || (rows == 0u) || (cols == 0u)) {
        return -1;
    }

    memset(matrix, 0, sizeof(*matrix));
    if (att1_mul_size(rows, cols, &value_count) != 0) {
        return -1;
    }

    matrix->values = calloc(value_count, sizeof(*matrix->values));
    matrix->scales = calloc(rows, sizeof(*matrix->scales));
    if ((matrix->values == NULL) || (matrix->scales == NULL)) {
        att1_q8_matrix_free(matrix);
        return -1;
    }

    matrix->rows = rows;
    matrix->cols = cols;
    return 0;
}

void att1_q8_matrix_free(att1_q8_matrix *matrix)
{
    if (matrix == NULL) {
        return;
    }

    free(matrix->values);
    free(matrix->scales);
    memset(matrix, 0, sizeof(*matrix));
}

int att1_quantize_q8_per_row(att1_q8_matrix *matrix,
                             const float *weights,
                             size_t rows,
                             size_t cols)
{
    size_t row = 0u;
    size_t col = 0u;

    if ((matrix == NULL) || (weights == NULL) ||
        (rows == 0u) || (cols == 0u)) {
        return -1;
    }

    if (att1_q8_matrix_alloc(matrix, rows, cols) != 0) {
        return -1;
    }

    for (row = 0u; row < rows; row++) {
        float max_abs = 0.0f;
        float scale = 1.0f;

        for (col = 0u; col < cols; col++) {
            const float value = weights[(row * cols) + col];
            const float abs_value = fabsf(value);

            if (!isfinite(value)) {
                att1_q8_matrix_free(matrix);
                return -1;
            }

            if (abs_value > max_abs) {
                max_abs = abs_value;
            }
        }

        if (max_abs > 0.0f) {
            scale = max_abs / 127.0f;
        }
        matrix->scales[row] = scale;

        for (col = 0u; col < cols; col++) {
            const float scaled = weights[(row * cols) + col] / scale;
            long quantized = lroundf(scaled);

            if (quantized > 127) {
                quantized = 127;
            } else if (quantized < -127) {
                quantized = -127;
            }

            matrix->values[(row * cols) + col] = (int8_t)quantized;
        }
    }

    return 0;
}
