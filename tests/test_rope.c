#include "att1_math.h"

#include <math.h>
#include <stdio.h>

static int near_f32(float lhs, float rhs)
{
    return fabsf(lhs - rhs) < 0.00001f;
}

int main(void)
{
    float unchanged[6] = {1.0f, 0.0f, 0.0f, 1.0f, 2.0f, 3.0f};
    float values[6] = {1.0f, 0.0f, 0.0f, 1.0f, 2.0f, 3.0f};
    const float c = cosf(1.0f);
    const float s = sinf(1.0f);
    size_t i = 0u;

    if (att1_rope_f32(unchanged, 6u, 0u, 1.0f) != 0) {
        fputs("rope position 0 call failed\n", stderr);
        return 1;
    }

    for (i = 0u; i < 6u; i++) {
        const float expected[6] = {1.0f, 0.0f, 0.0f, 1.0f, 2.0f, 3.0f};

        if (!near_f32(unchanged[i], expected[i])) {
            fputs("rope position 0 check failed\n", stderr);
            return 1;
        }
    }

    if (att1_rope_f32(values, 6u, 1u, 1.0f) != 0) {
        fputs("rope position 1 call failed\n", stderr);
        return 1;
    }

    /* RoPE rotates independent pairs: [0,1], [2,3], [4,5]. */
    if (!near_f32(values[0], c) ||
        !near_f32(values[1], s) ||
        !near_f32(values[2], -s) ||
        !near_f32(values[3], c) ||
        !near_f32(values[4], (2.0f * c) - (3.0f * s)) ||
        !near_f32(values[5], (2.0f * s) + (3.0f * c))) {
        fputs("rope value check failed\n", stderr);
        return 1;
    }

    if (att1_rope_f32(values, 5u, 1u, 1.0f) == 0) {
        fputs("rope odd dimension rejection failed\n", stderr);
        return 1;
    }

    puts("rope test passed");
    return 0;
}
