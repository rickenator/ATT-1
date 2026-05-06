#include "att1_math.h"

#include <math.h>
#include <stdio.h>

static int near_f32(float lhs, float rhs)
{
    return fabsf(lhs - rhs) < 0.00001f;
}

int main(void)
{
    float values[3] = {1.0f, 2.0f, 3.0f};
    float large_values[3] = {1000.0f, 1001.0f, 1002.0f};
    float equal_values[4] = {7.0f, 7.0f, 7.0f, 7.0f};
    const float denom = expf(-2.0f) + expf(-1.0f) + 1.0f;
    const float expected[3] = {
        expf(-2.0f) / denom,
        expf(-1.0f) / denom,
        1.0f / denom
    };
    float sum = 0.0f;
    size_t i = 0u;

    if (att1_softmax_f32(values, 3u) != 0) {
        fputs("softmax call failed\n", stderr);
        return 1;
    }

    for (i = 0u; i < 3u; i++) {
        sum += values[i];
        if (!near_f32(values[i], expected[i])) {
            fputs("softmax value check failed\n", stderr);
            return 1;
        }
    }

    if (!near_f32(sum, 1.0f)) {
        fputs("softmax sum check failed\n", stderr);
        return 1;
    }

    if (att1_softmax_f32(large_values, 3u) != 0) {
        fputs("softmax large logits call failed\n", stderr);
        return 1;
    }

    for (i = 0u; i < 3u; i++) {
        if (!isfinite(large_values[i]) || !near_f32(large_values[i], expected[i])) {
            fputs("softmax large logits stability check failed\n", stderr);
            return 1;
        }
    }

    if (att1_softmax_f32(equal_values, 4u) != 0) {
        fputs("softmax equal logits call failed\n", stderr);
        return 1;
    }

    for (i = 0u; i < 4u; i++) {
        if (!near_f32(equal_values[i], 0.25f)) {
            fputs("softmax equal logits check failed\n", stderr);
            return 1;
        }
    }

    puts("softmax test passed");
    return 0;
}
