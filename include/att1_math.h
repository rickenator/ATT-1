#ifndef ATT1_MATH_H
#define ATT1_MATH_H

#include <stddef.h>

int att1_matmul_f32(float *dst,
                    const float *lhs,
                    const float *rhs,
                    size_t rows,
                    size_t cols,
                    size_t inner);

int att1_rmsnorm_f32(float *dst,
                     const float *src,
                     const float *weight,
                     size_t count,
                     float epsilon);

int att1_softmax_f32(float *values, size_t count);

int att1_rope_f32(float *values,
                  size_t count,
                  size_t position,
                  float theta);

float att1_silu_f32(float value);

int att1_swiglu_f32(float *dst,
                    const float *gate,
                    const float *value,
                    size_t count);

#endif
