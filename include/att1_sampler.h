#ifndef ATT1_SAMPLER_H
#define ATT1_SAMPLER_H

#include <stdint.h>

typedef struct att1_sampler_config {
    uint32_t top_k;
    float temperature;
} att1_sampler_config;

#endif
