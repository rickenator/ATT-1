#include "att1_math.h"

#include <math.h>
#include <stdio.h>

static int near_f32(float lhs, float rhs)
{
    return fabsf(lhs - rhs) < 0.00001f;
}

int main(void)
{
    const float values[4] = {-2.0f, -1.0f, 0.0f, 2.0f};
    size_t i = 0u;

    for (i = 0u; i < 4u; i++) {
        const float expected = values[i] / (1.0f + expf(-values[i]));

        if (!near_f32(att1_silu_f32(values[i]), expected)) {
            fputs("silu value check failed\n", stderr);
            return 1;
        }
    }

    puts("silu test passed");
    return 0;
}
