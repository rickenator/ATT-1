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

/*
 * q4 wire-format constants (M74).
 *
 * Group size is encoded in the low byte of the tensor descriptor flags field
 * (bits [7:0]).  A value of 0 means "use the default group size" (32).
 * Valid non-zero values are powers of two in [ATT1_Q4_GROUP_SIZE_MIN,
 * ATT1_Q4_GROUP_SIZE_MAX].  Bits [31:8] of flags must be zero for q4 tensors.
 *
 * Payload layout for a q4 tensor with shape [rows, cols]:
 *   uint8  packed[rows * cols / 2]           -- low-nibble-first int4 pairs
 *   float32 scales[rows * (cols / group_size)] -- one f32 scale per group
 * Total: rows*cols/2 + rows*(cols/group_size)*4 bytes.
 *
 * Values are signed two's-complement int4, range [-7, 7].  -8 is excluded
 * to maintain a symmetric range around zero.  Low nibble holds the even
 * element, high nibble holds the odd element.
 */
#define ATT1_Q4_GROUP_SIZE_DEFAULT 32u
#define ATT1_Q4_GROUP_SIZE_MIN     16u
#define ATT1_Q4_GROUP_SIZE_MAX     128u
#define ATT1_Q4_FLAGS_GROUP_MASK   0xFFu

/*
 * q4 primitives (M75).
 *
 * All functions operate on a single group of `group_size` float32 values.
 * `group_size` must be a power of two in [ATT1_Q4_GROUP_SIZE_MIN,
 * ATT1_Q4_GROUP_SIZE_MAX], or ATT1_Q4_GROUP_SIZE_DEFAULT (32).
 *
 * Nibble packing convention (matches M74 wire format):
 *   packed byte [i/2]: low nibble = element[i], high nibble = element[i+1]
 *   i is always even.
 *
 * Signed int4 range: [-7, 7]. -8 is excluded (symmetric around zero).
 */

/*
 * Compute the per-group scale for a row segment [src, src+group_size).
 * scale = max(|src[i]|) / 7.0, or 1.0 if the row is all-zero.
 * Returns -1 on null/bad args, 0 on success.
 */
int att1_q4_group_scale(const float *src,
                        uint32_t     group_size,
                        float       *out_scale);

/*
 * Pack `group_size` int4 values (in [-7,7], clamped before packing) from
 * `src_int4` into `group_size/2` packed bytes in `dst_packed`.
 * Low nibble = even index, high nibble = odd index.
 * Returns -1 on null/bad args, 0 on success.
 */
int att1_q4_pack_group(const int8_t *src_int4,
                       uint32_t      group_size,
                       uint8_t      *dst_packed);

/*
 * Unpack `group_size/2` packed bytes from `src_packed` into `group_size`
 * int4 values in `dst_int4`. Sign-extends 4-bit values to int8.
 * Returns -1 on null/bad args, 0 on success.
 */
int att1_q4_unpack_group(const uint8_t *src_packed,
                         uint32_t       group_size,
                         int8_t        *dst_int4);

/*
 * Quantize `group_size` float32 values from `src` into packed int4 bytes in
 * `dst_packed` and store the computed scale in `*out_scale`.
 * Returns -1 on null/bad args or non-finite input, 0 on success.
 */
int att1_q4_quantize_group(const float *src,
                           uint32_t     group_size,
                           uint8_t     *dst_packed,
                           float       *out_scale);

/*
 * Dequantize `group_size/2` packed bytes from `src_packed` using `scale`
 * into `group_size` float32 values in `dst`.
 * dst[i] = int4_value[i] * scale.
 * Returns -1 on null/bad args, 0 on success.
 */
int att1_q4_dequantize_group(const uint8_t *src_packed,
                             uint32_t       group_size,
                             float          scale,
                             float         *dst);

typedef struct att1_q8_matrix {
    size_t rows;
    size_t cols;
    int8_t *values;
    float *scales;
} att1_q8_matrix;

/*
 * q4 weight matrix (M76).
 *
 * Row-major grouped int4 weights.  Each row is divided into groups of
 * `group_size` elements; each group has one float32 scale.
 *
 *   packed : rows * cols / 2  bytes  (low-nibble-first, M74 wire format)
 *   scales : rows * (cols / group_size)  float32 values
 *
 * `group_size` must be a valid q4 group size (power-of-two in
 * [ATT1_Q4_GROUP_SIZE_MIN, ATT1_Q4_GROUP_SIZE_MAX]).
 *
 * Use att1_q4_matrix_alloc / att1_q4_matrix_free to manage storage.
 * Use att1_quantize_q4_per_group to fill from a float32 weight array.
 */
typedef struct att1_q4_matrix {
    size_t   rows;
    size_t   cols;
    uint32_t group_size;
    uint8_t *packed;  /* rows * cols / 2 bytes */
    float   *scales;  /* rows * (cols / group_size) float32 values */
} att1_q4_matrix;

/*
 * Allocate storage for a q4 weight matrix.
 * rows, cols must be non-zero; cols must be even and divisible by group_size.
 * group_size must be a valid q4 group size.
 * Returns 0 on success; -1 on invalid args or allocation failure.
 * On success, release with att1_q4_matrix_free().
 */
int att1_q4_matrix_alloc(att1_q4_matrix *matrix,
                         size_t          rows,
                         size_t          cols,
                         uint32_t        group_size);

/*
 * Release owned q4 matrix storage and zero all fields.
 * Passing NULL is allowed.
 */
void att1_q4_matrix_free(att1_q4_matrix *matrix);

/*
 * Quantize row-major float32 weights into a q4 matrix using per-group
 * symmetric scales.  Allocates `matrix` internally; caller must free with
 * att1_q4_matrix_free().
 * Returns 0 on success; -1 on invalid args, non-finite input, or alloc failure.
 */
int att1_quantize_q4_per_group(att1_q4_matrix *matrix,
                               const float    *weights,
                               size_t          rows,
                               size_t          cols,
                               uint32_t        group_size);

/*
 * Compute row-major q4xf32 matrix multiplication (dequantize-then-multiply):
 *   dst[lhs_rows, weights->rows] = lhs[lhs_rows, weights->cols]
 *                                  * dequant(weights)^T
 *
 * Activations (lhs) remain float32.  Weights are dequantized on the fly:
 *   element = int4_nibble * group_scale.
 *
 * lhs_cols must equal weights->cols.
 * Returns 0 on success; -1 on invalid args or dimension mismatch.
 */
int att1_matmul_q4xf32(float                *dst,
                       const float          *lhs,
                       size_t                lhs_rows,
                       size_t                lhs_cols,
                       const att1_q4_matrix *weights);

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
