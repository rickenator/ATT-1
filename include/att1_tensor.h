#ifndef ATT1_TENSOR_H
#define ATT1_TENSOR_H

#include <stddef.h>
#include <stdint.h>

#define ATT1_TENSOR_MAX_RANK 4

typedef enum att1_tensor_dtype {
    ATT1_DTYPE_UNKNOWN = 0,
    ATT1_DTYPE_F32,
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

typedef struct att1_tensor {
    att1_tensor_desc desc;
    size_t shape[ATT1_TENSOR_MAX_RANK];
    size_t element_count;
    float *data;
} att1_tensor;

/*
 * Allocate a zero-initialized owned float32 tensor.
 *
 * rank must be in [1, ATT1_TENSOR_MAX_RANK]. Every shape dimension must be
 * nonzero. On success, tensor owns data and must be released with
 * att1_tensor_free. On failure, tensor is left cleared and -1 is returned.
 */
int att1_tensor_alloc_f32(att1_tensor *tensor,
                          uint32_t rank,
                          const size_t *shape);

/*
 * Release tensor storage and clear all tensor metadata.
 *
 * Passing NULL is allowed.
 */
void att1_tensor_free(att1_tensor *tensor);

#endif
