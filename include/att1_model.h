#ifndef ATT1_MODEL_H
#define ATT1_MODEL_H

#include <stddef.h>
#include <stdint.h>

typedef struct att1_model_info {
    const char *name;
    uint32_t layer_count;
    size_t parameter_bytes;
} att1_model_info;

#endif
