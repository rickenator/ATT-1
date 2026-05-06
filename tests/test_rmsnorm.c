#include "att1_math.h"

#include <math.h>
#include <stdio.h>

static int near_f32(float lhs, float rhs)
{
    return fabsf(lhs - rhs) < 0.00001f;
}

int main(void)
{
    const float src[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    const float weight[4] = {1.0f, 0.5f, 2.0f, 1.0f};
    const float scale = 1.0f / sqrtf(7.5f + 0.000001f);
    const float expected[4] = {
        1.0f * scale,
        2.0f * scale * 0.5f,
        3.0f * scale * 2.0f,
        4.0f * scale
    };
    float actual[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    size_t i = 0u;

    if (att1_rmsnorm_f32(actual, src, weight, 4u, 0.000001f) != 0) {
        fputs("rmsnorm call failed\n", stderr);
        return 1;
    }

    for (i = 0u; i < 4u; i++) {
        if (!near_f32(actual[i], expected[i])) {
            fputs("rmsnorm value check failed\n", stderr);
            return 1;
        }
    }

    puts("rmsnorm test passed");
    return 0;
}
