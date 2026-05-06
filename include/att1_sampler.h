#ifndef ATT1_SAMPLER_H
#define ATT1_SAMPLER_H

#include <stddef.h>
#include <stdint.h>

typedef struct att1_sampler_config {
    uint32_t top_k;
    float temperature;
} att1_sampler_config;

/*
 * Greedy argmax sampler over float32 logits.
 *
 * Ties resolve to the lowest token id. NaN logits are rejected. logits must be
 * non-NULL, count must be nonzero, and out_token must be non-NULL.
 */
int att1_sampler_greedy_f32(const float *logits,
                            size_t count,
                            uint32_t *out_token);

#endif
