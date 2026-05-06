#include "att1_math.h"

#include <math.h>
#include <stdio.h>

static int near_f32(float lhs, float rhs)
{
    return fabsf(lhs - rhs) < 0.00001f;
}

int main(void)
{
    const float gate[3] = {-1.0f, 0.0f, 2.0f};
    const float value[3] = {2.0f, 3.0f, -4.0f};
    float actual[3] = {0.0f, 0.0f, 0.0f};
    size_t i = 0u;

    if (att1_swiglu_f32(actual, gate, value, 3u) != 0) {
        fputs("swiglu call failed\n", stderr);
        return 1;
    }

    for (i = 0u; i < 3u; i++) {
        const float expected = (gate[i] / (1.0f + expf(-gate[i]))) * value[i];

        if (!near_f32(actual[i], expected)) {
            fputs("swiglu value check failed\n", stderr);
            return 1;
        }
    }

    puts("swiglu test passed");
    return 0;
}
