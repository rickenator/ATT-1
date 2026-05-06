#include "att1_math.h"

#include <math.h>
#include <stdio.h>

static int near_f32(float lhs, float rhs)
{
    return fabsf(lhs - rhs) < 0.00001f;
}

int main(void)
{
    /* General matmul: lhs is [rows][inner], rhs is [inner][cols]. */
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
    /*
     * Projection convention used by attention/FFN:
     * vector[1][in_dim] * weight[in_dim][out_dim] -> output[1][out_dim].
     */
    const float vector[3] = {2.0f, 3.0f, 4.0f};
    const float weight_in_out[6] = {
        1.0f, 5.0f,
        2.0f, 6.0f,
        3.0f, 7.0f
    };
    const float expected_projection[2] = {
        20.0f,
        56.0f
    };
    float actual[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float projection[2] = {0.0f, 0.0f};
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

    if (att1_matmul_f32(projection, vector, weight_in_out, 1u, 2u, 3u) != 0) {
        fputs("matmul projection call failed\n", stderr);
        return 1;
    }

    for (i = 0u; i < 2u; i++) {
        if (!near_f32(projection[i], expected_projection[i])) {
            fputs("matmul projection convention check failed\n", stderr);
            return 1;
        }
    }

    puts("matmul test passed");
    return 0;
}
