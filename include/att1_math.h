#ifndef ATT1_MATH_H
#define ATT1_MATH_H

#include <stddef.h>

/*
 * Compute row-major float32 matrix multiplication:
 * dst[rows, cols] = lhs[rows, inner] * rhs[inner, cols].
 *
 * All pointers must be non-NULL and dimensions must be nonzero.
 */
int att1_matmul_f32(float *dst,
                    const float *lhs,
                    const float *rhs,
                    size_t rows,
                    size_t cols,
                    size_t inner);

/*
 * Compute RMSNorm over count float32 values:
 * dst[i] = src[i] * weight[i] / sqrt(mean(src^2) + epsilon).
 *
 * All pointers must be non-NULL, count must be nonzero, and epsilon must be
 * positive.
 */
int att1_rmsnorm_f32(float *dst,
                     const float *src,
                     const float *weight,
                     size_t count,
                     float epsilon);

/*
 * Compute an in-place numerically stabilized softmax over count values.
 *
 * values must be non-NULL and count must be nonzero.
 */
int att1_softmax_f32(float *values, size_t count);

/*
 * Apply in-place rotary position embedding to even/odd float32 pairs.
 *
 * count must be nonzero and even. theta must be positive. The frequency for
 * pair i is 1 / pow(theta, i / count), matching the conventional RoPE pair
 * layout for a single vector.
 */
int att1_rope_f32(float *values,
                  size_t count,
                  size_t position,
                  float theta);

/*
 * Compute SiLU(x) = x / (1 + exp(-x)).
 */
float att1_silu_f32(float value);

/*
 * Compute SwiGLU helper output:
 * dst[i] = SiLU(gate[i]) * value[i].
 *
 * All pointers must be non-NULL and count must be nonzero.
 */
int att1_swiglu_f32(float *dst,
                    const float *gate,
                    const float *value,
                    size_t count);

#endif
