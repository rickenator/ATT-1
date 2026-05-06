#ifndef ATT1_TENSOR_H
#define ATT1_TENSOR_H

#include <stddef.h>
#include <stdint.h>

typedef enum att1_tensor_dtype {
    ATT1_DTYPE_UNKNOWN = 0,
    ATT1_DTYPE_F16,
    ATT1_DTYPE_BF16,
    ATT1_DTYPE_Q8,
    ATT1_DTYPE_Q4
} att1_tensor_dtype;

typedef struct att1_tensor_desc {
    uint64_t id;
    att1_tensor_dtype dtype;
    uint32_t rank;
    size_t bytes;
} att1_tensor_desc;

#endif
