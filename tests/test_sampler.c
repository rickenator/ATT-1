#include "att1_sampler.h"

#include <math.h>
#include <stdio.h>

int main(void)
{
    const float logits[4] = {0.0f, 1.0f, 5.0f, 2.0f};
    const float ties[4] = {3.0f, 7.0f, 7.0f, 1.0f};
    const float bad[3] = {0.0f, NAN, 1.0f};
    uint32_t token = 0u;

    if ((att1_sampler_greedy_f32(logits, 4u, &token) != 0) || (token != 2u)) {
        fputs("sampler argmax check failed\n", stderr);
        return 1;
    }

    if ((att1_sampler_greedy_f32(ties, 4u, &token) != 0) || (token != 1u)) {
        fputs("sampler tie check failed\n", stderr);
        return 1;
    }

    if (att1_sampler_greedy_f32(bad, 3u, &token) == 0) {
        fputs("sampler NaN rejection failed\n", stderr);
        return 1;
    }

    puts("sampler test passed");
    return 0;
}
