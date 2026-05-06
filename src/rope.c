#include "att1_math.h"

#include <math.h>

int att1_rope_f32(float *values,
                  size_t count,
                  size_t position,
                  float theta)
{
    size_t i = 0u;

    if (values == NULL) {
        return -1;
    }

    if ((count == 0u) || ((count % 2u) != 0u) || (theta <= 0.0f)) {
        return -1;
    }

    for (i = 0u; i < count; i += 2u) {
        const float exponent = (float)i / (float)count;
        const float frequency = 1.0f / powf(theta, exponent);
        const float angle = (float)position * frequency;
        const float c = cosf(angle);
        const float s = sinf(angle);
        const float x0 = values[i];
        const float x1 = values[i + 1u];

        values[i] = (x0 * c) - (x1 * s);
        values[i + 1u] = (x0 * s) + (x1 * c);
    }

    return 0;
}
