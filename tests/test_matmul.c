#include "att1_math.h"

#include <math.h>
#include <stdio.h>

static int near_f32(float lhs, float rhs)
{
    return fabsf(lhs - rhs) < 0.00001f;
}

int main(void)
{
    const float lhs[6] = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f
    };
    const float rhs[6] = {
        7.0f, 8.0f,
        9.0f, 10.0f,
        11.0f, 12.0f
    };
    const float expected[4] = {
        58.0f, 64.0f,
        139.0f, 154.0f
    };
    float actual[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    size_t i = 0u;

    if (att1_matmul_f32(actual, lhs, rhs, 2u, 2u, 3u) != 0) {
        fputs("matmul call failed\n", stderr);
        return 1;
    }

    for (i = 0u; i < 4u; i++) {
        if (!near_f32(actual[i], expected[i])) {
            fputs("matmul value check failed\n", stderr);
            return 1;
        }
    }

    puts("matmul test passed");
    return 0;
}
