#include "att1_math.h"

#include <math.h>

float att1_silu_f32(float value)
{
    return value / (1.0f + expf(-value));
}

int att1_swiglu_f32(float *dst,
                    const float *gate,
                    const float *value,
                    size_t count)
{
    size_t i = 0u;

    if ((dst == NULL) || (gate == NULL) || (value == NULL)) {
        return -1;
    }

    if (count == 0u) {
        return -1;
    }

    for (i = 0u; i < count; i++) {
        dst[i] = att1_silu_f32(gate[i]) * value[i];
    }

    return 0;
}
