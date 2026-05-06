#include "att1_math.h"

#include <math.h>

int att1_rmsnorm_f32(float *dst,
                     const float *src,
                     const float *weight,
                     size_t count,
                     float epsilon)
{
    float sum_squares = 0.0f;
    float scale = 0.0f;
    size_t i = 0u;

    if ((dst == NULL) || (src == NULL) || (weight == NULL)) {
        return -1;
    }

    if ((count == 0u) || (epsilon <= 0.0f)) {
        return -1;
    }

    for (i = 0u; i < count; i++) {
        sum_squares += src[i] * src[i];
    }

    scale = 1.0f / sqrtf((sum_squares / (float)count) + epsilon);

    for (i = 0u; i < count; i++) {
        dst[i] = src[i] * scale * weight[i];
    }

    return 0;
}

int att1_softmax_f32(float *values, size_t count)
{
    float max_value = 0.0f;
    float sum = 0.0f;
    size_t i = 0u;

    if (values == NULL) {
        return -1;
    }

    if (count == 0u) {
        return -1;
    }

    max_value = values[0];
    for (i = 1u; i < count; i++) {
        if (values[i] > max_value) {
            max_value = values[i];
        }
    }

    for (i = 0u; i < count; i++) {
        values[i] = expf(values[i] - max_value);
        sum += values[i];
    }

    if (sum == 0.0f) {
        return -1;
    }

    for (i = 0u; i < count; i++) {
        values[i] /= sum;
    }

    return 0;
}
