#include "att1_sampler.h"

#include <math.h>

int att1_sampler_greedy_f32(const float *logits,
                            size_t count,
                            uint32_t *out_token)
{
    size_t i = 0u;
    size_t best = 0u;

    if ((logits == NULL) || (out_token == NULL) || (count == 0u)) {
        return -1;
    }

    for (i = 0u; i < count; i++) {
        if (isnan(logits[i])) {
            return -1;
        }
    }

    for (i = 1u; i < count; i++) {
        if (logits[i] > logits[best]) {
            best = i;
        }
    }

    if (best > (size_t)UINT32_MAX) {
        return -1;
    }

    *out_token = (uint32_t)best;
    return 0;
}
