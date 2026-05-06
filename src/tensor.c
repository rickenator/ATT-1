#include "att1_tensor.h"

#include <stdlib.h>
#include <string.h>

static int att1_mul_size(size_t lhs, size_t rhs, size_t *out)
{
    if ((lhs != 0u) && (rhs > (SIZE_MAX / lhs))) {
        return -1;
    }

    *out = lhs * rhs;
    return 0;
}

int att1_tensor_alloc_f32(att1_tensor *tensor,
                          uint32_t rank,
                          const size_t *shape)
{
    size_t elements = 1u;
    size_t bytes = 0u;
    uint32_t i = 0u;

    if ((tensor == NULL) || (shape == NULL)) {
        return -1;
    }

    if ((rank == 0u) || (rank > ATT1_TENSOR_MAX_RANK)) {
        return -1;
    }

    memset(tensor, 0, sizeof(*tensor));

    for (i = 0u; i < rank; i++) {
        if (shape[i] == 0u) {
            return -1;
        }

        if (att1_mul_size(elements, shape[i], &elements) != 0) {
            return -1;
        }

        tensor->shape[i] = shape[i];
    }

    if (att1_mul_size(elements, sizeof(float), &bytes) != 0) {
        return -1;
    }

    tensor->data = calloc(elements, sizeof(float));
    if (tensor->data == NULL) {
        memset(tensor, 0, sizeof(*tensor));
        return -1;
    }

    tensor->desc.dtype = ATT1_DTYPE_F32;
    tensor->desc.rank = rank;
    tensor->desc.bytes = bytes;
    tensor->element_count = elements;
    return 0;
}

void att1_tensor_free(att1_tensor *tensor)
{
    if (tensor == NULL) {
        return;
    }

    free(tensor->data);
    memset(tensor, 0, sizeof(*tensor));
}
