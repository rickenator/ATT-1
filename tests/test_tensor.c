#include "att1_tensor.h"

#include <stdio.h>

int main(void)
{
    const size_t shape[2] = {2u, 3u};
    att1_tensor tensor;

    if (att1_tensor_alloc_f32(&tensor, 2u, shape) != 0) {
        fputs("tensor allocation failed\n", stderr);
        return 1;
    }

    if (tensor.desc.dtype != ATT1_DTYPE_F32) {
        fputs("tensor dtype check failed\n", stderr);
        att1_tensor_free(&tensor);
        return 1;
    }

    if ((tensor.desc.rank != 2u) ||
        (tensor.shape[0] != 2u) ||
        (tensor.shape[1] != 3u) ||
        (tensor.element_count != 6u) ||
        (tensor.desc.bytes != (6u * sizeof(float)))) {
        fputs("tensor metadata check failed\n", stderr);
        att1_tensor_free(&tensor);
        return 1;
    }

    for (size_t i = 0u; i < tensor.element_count; i++) {
        if (tensor.data[i] != 0.0f) {
            fputs("tensor zero init check failed\n", stderr);
            att1_tensor_free(&tensor);
            return 1;
        }
    }

    tensor.data[5] = 42.0f;
    att1_tensor_free(&tensor);

    if ((tensor.data != NULL) || (tensor.element_count != 0u)) {
        fputs("tensor free check failed\n", stderr);
        return 1;
    }

    puts("tensor test passed");
    return 0;
}
