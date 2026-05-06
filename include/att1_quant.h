#ifndef ATT1_QUANT_H
#define ATT1_QUANT_H

#include <stdint.h>

typedef enum att1_quant_format {
    ATT1_QUANT_NONE = 0,
    ATT1_QUANT_INT8,
    ATT1_QUANT_INT4
} att1_quant_format;

typedef struct att1_quant_desc {
    att1_quant_format format;
    uint32_t group_size;
} att1_quant_desc;

#endif
