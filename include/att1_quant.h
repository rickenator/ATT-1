#ifndef ATT1_QUANT_H
#define ATT1_QUANT_H

#include <stddef.h>
#include <stdint.h>

typedef enum att1_quant_format {
    ATT1_QUANT_NONE = 0,
    ATT1_QUANT_INT8,
    ATT1_QUANT_INT4
} att1_quant_format;

typedef struct att1_quant_desc {
    att1_quant_format format;
    uint32_t group_size;
} att1_quant_desc;

typedef struct att1_q8_matrix {
    size_t rows;
    size_t cols;
    int8_t *values;
    float *scales;
} att1_q8_matrix;

/*
 * Allocate a row-major int8 weight matrix with one float32 scale per row.
 *
 * Passing zero rows or columns is invalid. On success, matrix must be released
 * with att1_q8_matrix_free.
 */
int att1_q8_matrix_alloc(att1_q8_matrix *matrix,
                         size_t rows,
                         size_t cols);

/*
 * Release owned q8 matrix storage and clear metadata.
 *
 * Passing NULL is allowed.
 */
void att1_q8_matrix_free(att1_q8_matrix *matrix);

/*
 * Quantize row-major float32 weights into a q8 matrix using independent
 * symmetric per-row scales. The q8 matrix stores rows as output rows and
 * columns as input dimensions.
 */
int att1_quantize_q8_per_row(att1_q8_matrix *matrix,
                             const float *weights,
                             size_t rows,
                             size_t cols);

/*
 * Compute row-major q8xf32 matrix multiplication:
 * dst[lhs_rows, weights->rows] = lhs[lhs_rows, weights->cols] *
 * dequant(weights[weights->rows, weights->cols])^T.
 *
 * Activations stay float32. Weights are dequantized on the fly as
 * (float)qvalue * row_scale.
 */
int att1_matmul_q8xf32(float *dst,
                       const float *lhs,
                       size_t lhs_rows,
                       size_t lhs_cols,
                       const att1_q8_matrix *weights);

#endif
