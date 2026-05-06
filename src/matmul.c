#include "att1_math.h"

int att1_matmul_f32(float *dst,
                    const float *lhs,
                    const float *rhs,
                    size_t rows,
                    size_t cols,
                    size_t inner)
{
    size_t row = 0u;
    size_t col = 0u;
    size_t k = 0u;

    if ((dst == NULL) || (lhs == NULL) || (rhs == NULL)) {
        return -1;
    }

    if ((rows == 0u) || (cols == 0u) || (inner == 0u)) {
        return -1;
    }

    for (row = 0u; row < rows; row++) {
        for (col = 0u; col < cols; col++) {
            float sum = 0.0f;

            for (k = 0u; k < inner; k++) {
                sum += lhs[(row * inner) + k] * rhs[(k * cols) + col];
            }

            dst[(row * cols) + col] = sum;
        }
    }

    return 0;
}
