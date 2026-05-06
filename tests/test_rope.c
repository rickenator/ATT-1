#include "att1_math.h"

#include <math.h>
#include <stdio.h>

static int near_f32(float lhs, float rhs)
{
    return fabsf(lhs - rhs) < 0.00001f;
}

int main(void)
{
    float values[4] = {1.0f, 0.0f, 0.0f, 1.0f};
    const float c = cosf(1.0f);
    const float s = sinf(1.0f);

    if (att1_rope_f32(values, 4u, 1u, 1.0f) != 0) {
        fputs("rope call failed\n", stderr);
        return 1;
    }

    if (!near_f32(values[0], c) ||
        !near_f32(values[1], s) ||
        !near_f32(values[2], -s) ||
        !near_f32(values[3], c)) {
        fputs("rope value check failed\n", stderr);
        return 1;
    }

    puts("rope test passed");
    return 0;
}
