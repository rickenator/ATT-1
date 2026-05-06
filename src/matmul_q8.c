#include "att1_quant.h"

int att1_matmul_q8xf32(float *dst,
                       const float *lhs,
                       size_t lhs_rows,
                       size_t lhs_cols,
                       const att1_q8_matrix *weights)
{
    size_t row = 0u;
    size_t out = 0u;
    size_t col = 0u;

    if ((dst == NULL) || (lhs == NULL) || (weights == NULL) ||
        (weights->values == NULL) || (weights->scales == NULL)) {
        return -1;
    }

    if ((lhs_rows == 0u) || (lhs_cols == 0u) ||
        (weights->rows == 0u) || (weights->cols == 0u) ||
        (lhs_cols != weights->cols)) {
        return -1;
    }

    for (row = 0u; row < lhs_rows; row++) {
        for (out = 0u; out < weights->rows; out++) {
            const float scale = weights->scales[out];
            float sum = 0.0f;

            for (col = 0u; col < lhs_cols; col++) {
                const int qvalue = weights->values[(out * weights->cols) + col];
                sum += lhs[(row * lhs_cols) + col] *
                       ((float)qvalue * scale);
            }

            dst[(row * weights->rows) + out] = sum;
        }
    }

    return 0;
}
