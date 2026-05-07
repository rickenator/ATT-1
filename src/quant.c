#include "att1_quant.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ── internal helpers ─────────────────────────────────────────────────────── */

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

/* ── q4 helpers (M75) ─────────────────────────────────────────────────────── */

static int q4_group_size_valid(uint32_t group_size)
{
    return (group_size >= ATT1_Q4_GROUP_SIZE_MIN) &&
           (group_size <= ATT1_Q4_GROUP_SIZE_MAX) &&
           ((group_size & (group_size - 1u)) == 0u);
}

int att1_q4_group_scale(const float *src,
                        uint32_t     group_size,
                        float       *out_scale)
{
    uint32_t i = 0u;
    float max_abs = 0.0f;

    if ((src == NULL) || (out_scale == NULL) || !q4_group_size_valid(group_size)) {
        return -1;
    }

    for (i = 0u; i < group_size; i++) {
        const float a = fabsf(src[i]);
        if (!isfinite(src[i])) { return -1; }
        if (a > max_abs) { max_abs = a; }
    }

    *out_scale = (max_abs > 0.0f) ? (max_abs / 7.0f) : 1.0f;
    return 0;
}

int att1_q4_pack_group(const int8_t *src_int4,
                       uint32_t      group_size,
                       uint8_t      *dst_packed)
{
    uint32_t i = 0u;

    if ((src_int4 == NULL) || (dst_packed == NULL) ||
        !q4_group_size_valid(group_size)) {
        return -1;
    }

    for (i = 0u; i < group_size; i += 2u) {
        int8_t lo = src_int4[i];
        int8_t hi = src_int4[i + 1u];

        /* Clamp to [-7, 7] */
        if (lo >  7) { lo =  7; }
        if (lo < -7) { lo = -7; }
        if (hi >  7) { hi =  7; }
        if (hi < -7) { hi = -7; }

        /* Mask to 4 bits (signed nibble in two's-complement) */
        dst_packed[i / 2u] = (uint8_t)(((uint8_t)lo & 0x0Fu) |
                                        (((uint8_t)hi & 0x0Fu) << 4u));
    }
    return 0;
}

int att1_q4_unpack_group(const uint8_t *src_packed,
                         uint32_t       group_size,
                         int8_t        *dst_int4)
{
    uint32_t i = 0u;

    if ((src_packed == NULL) || (dst_int4 == NULL) ||
        !q4_group_size_valid(group_size)) {
        return -1;
    }

    for (i = 0u; i < group_size; i += 2u) {
        const uint8_t byte = src_packed[i / 2u];
        const uint8_t lo_nibble = byte & 0x0Fu;
        const uint8_t hi_nibble = (byte >> 4u) & 0x0Fu;

        /* Sign-extend 4-bit to 8-bit: if bit 3 is set, value is negative */
        dst_int4[i]      = (lo_nibble & 0x08u) ? (int8_t)((int8_t)lo_nibble | (int8_t)0xF0) : (int8_t)lo_nibble;
        dst_int4[i + 1u] = (hi_nibble & 0x08u) ? (int8_t)((int8_t)hi_nibble | (int8_t)0xF0) : (int8_t)hi_nibble;
    }
    return 0;
}

int att1_q4_quantize_group(const float *src,
                           uint32_t     group_size,
                           uint8_t     *dst_packed,
                           float       *out_scale)
{
    float scale = 1.0f;
    uint32_t i = 0u;
    int8_t int4_buf[ATT1_Q4_GROUP_SIZE_MAX];

    if ((src == NULL) || (dst_packed == NULL) || (out_scale == NULL) ||
        !q4_group_size_valid(group_size)) {
        return -1;
    }

    if (att1_q4_group_scale(src, group_size, &scale) != 0) {
        return -1;
    }

    for (i = 0u; i < group_size; i++) {
        long q = lroundf(src[i] / scale);
        if (q >  7) { q =  7; }
        if (q < -7) { q = -7; }
        int4_buf[i] = (int8_t)q;
    }

    *out_scale = scale;
    return att1_q4_pack_group(int4_buf, group_size, dst_packed);
}

int att1_q4_dequantize_group(const uint8_t *src_packed,
                             uint32_t       group_size,
                             float          scale,
                             float         *dst)
{
    int8_t int4_buf[ATT1_Q4_GROUP_SIZE_MAX];
    uint32_t i = 0u;

    if ((src_packed == NULL) || (dst == NULL) ||
        !q4_group_size_valid(group_size)) {
        return -1;
    }

    if (att1_q4_unpack_group(src_packed, group_size, int4_buf) != 0) {
        return -1;
    }

    for (i = 0u; i < group_size; i++) {
        dst[i] = (float)int4_buf[i] * scale;
    }
    return 0;
}

/* ── q4 matrix alloc/free/quantize/matmul (M76) ──────────────────────────── */

int att1_q4_matrix_alloc(att1_q4_matrix *matrix,
                         size_t          rows,
                         size_t          cols,
                         uint32_t        group_size)
{
    size_t n_groups = 0u;
    size_t packed_bytes = 0u;

    if ((matrix == NULL) || (rows == 0u) || (cols == 0u) ||
        !q4_group_size_valid(group_size) ||
        ((cols & 1u) != 0u) ||
        ((cols % (size_t)group_size) != 0u)) {
        return -1;
    }

    memset(matrix, 0, sizeof(*matrix));

    /* packed = rows * cols / 2 bytes */
    if (att1_mul_size(rows, cols / 2u, &packed_bytes) != 0) { return -1; }

    /* scales = rows * (cols / group_size) float32 values */
    if (att1_mul_size(cols / (size_t)group_size, rows, &n_groups) != 0) { return -1; }

    matrix->packed = calloc(packed_bytes, sizeof(*matrix->packed));
    matrix->scales = calloc(n_groups, sizeof(*matrix->scales));
    if ((matrix->packed == NULL) || (matrix->scales == NULL)) {
        att1_q4_matrix_free(matrix);
        return -1;
    }

    matrix->rows       = rows;
    matrix->cols       = cols;
    matrix->group_size = group_size;
    return 0;
}

void att1_q4_matrix_free(att1_q4_matrix *matrix)
{
    if (matrix == NULL) { return; }
    free(matrix->packed);
    free(matrix->scales);
    memset(matrix, 0, sizeof(*matrix));
}

int att1_quantize_q4_per_group(att1_q4_matrix *matrix,
                               const float    *weights,
                               size_t          rows,
                               size_t          cols,
                               uint32_t        group_size)
{
    size_t row = 0u;
    size_t g   = 0u;
    size_t n_groups_per_row = 0u;

    if ((matrix == NULL) || (weights == NULL) ||
        (rows == 0u) || (cols == 0u) ||
        !q4_group_size_valid(group_size) ||
        ((cols & 1u) != 0u) ||
        ((cols % (size_t)group_size) != 0u)) {
        return -1;
    }

    if (att1_q4_matrix_alloc(matrix, rows, cols, group_size) != 0) {
        return -1;
    }

    n_groups_per_row = cols / (size_t)group_size;

    for (row = 0u; row < rows; row++) {
        for (g = 0u; g < n_groups_per_row; g++) {
            const size_t src_off    = row * cols + g * (size_t)group_size;
            const size_t packed_off = row * (cols / 2u) + g * ((size_t)group_size / 2u);
            const size_t scale_idx  = row * n_groups_per_row + g;
            float scale = 1.0f;

            if (att1_q4_quantize_group(&weights[src_off], group_size,
                                      &matrix->packed[packed_off],
                                      &scale) != 0) {
                att1_q4_matrix_free(matrix);
                return -1;
            }
            matrix->scales[scale_idx] = scale;
        }
    }
    return 0;
}

int att1_matmul_q4xf32(float                *dst,
                       const float          *lhs,
                       size_t                lhs_rows,
                       size_t                lhs_cols,
                       const att1_q4_matrix *weights)
{
    size_t   lhs_row     = 0u;
    size_t   weight_row  = 0u;
    size_t   g           = 0u;
    size_t   j           = 0u;
    size_t   n_groups_per_row = 0u;
    int8_t   int4_buf[ATT1_Q4_GROUP_SIZE_MAX];

    if ((dst == NULL) || (lhs == NULL) || (weights == NULL) ||
        (weights->packed == NULL) || (weights->scales == NULL)) {
        return -1;
    }

    if ((lhs_rows == 0u) || (lhs_cols == 0u) ||
        (weights->rows == 0u) || (weights->cols == 0u) ||
        (lhs_cols != weights->cols)) {
        return -1;
    }

    n_groups_per_row = weights->cols / (size_t)weights->group_size;

    for (lhs_row = 0u; lhs_row < lhs_rows; lhs_row++) {
        for (weight_row = 0u; weight_row < weights->rows; weight_row++) {
            float sum = 0.0f;

            for (g = 0u; g < n_groups_per_row; g++) {
                const size_t packed_off = weight_row * (weights->cols / 2u)
                                        + g * ((size_t)weights->group_size / 2u);
                const size_t act_off    = lhs_row * lhs_cols
                                        + g * (size_t)weights->group_size;
                const size_t scale_idx  = weight_row * n_groups_per_row + g;
                const float  scale      = weights->scales[scale_idx];

                if (att1_q4_unpack_group(&weights->packed[packed_off],
                                         weights->group_size,
                                         int4_buf) != 0) {
                    return -1;
                }

                for (j = 0u; j < (size_t)weights->group_size; j++) {
                    sum += lhs[act_off + j] * ((float)int4_buf[j] * scale);
                }
            }

            dst[lhs_row * weights->rows + weight_row] = sum;
        }
    }
    return 0;
}
