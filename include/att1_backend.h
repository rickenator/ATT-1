#ifndef ATT1_BACKEND_H
#define ATT1_BACKEND_H

#include "att1_quant.h"
#include "att1_status.h"

#include <stddef.h>

typedef struct att1_backend att1_backend;

typedef struct att1_backend_ops {
    const char *name;

    void *(*alloc)(att1_backend *backend, size_t bytes);
    void (*free)(att1_backend *backend, void *ptr);
    int (*sync)(att1_backend *backend);

    int (*matmul_f32)(att1_backend *backend,
                      float *dst,
                      const float *lhs,
                      const float *rhs,
                      size_t rows,
                      size_t cols,
                      size_t inner);

    int (*matmul_q8xf32)(att1_backend *backend,
                         float *dst,
                         const float *lhs,
                         size_t lhs_rows,
                         size_t lhs_cols,
                         const att1_q8_matrix *weights);

    int (*rmsnorm_f32)(att1_backend *backend,
                       float *dst,
                       const float *src,
                       const float *weight,
                       size_t count,
                       float epsilon);

    int (*softmax_f32)(att1_backend *backend,
                       float *values,
                       size_t count);

    int (*rope_f32)(att1_backend *backend,
                    float *values,
                    size_t count,
                    size_t position,
                    float theta);

    int (*ffn_swiglu_f32)(att1_backend *backend,
                          float *dst,
                          const float *gate,
                          const float *value,
                          size_t count);
} att1_backend_ops;

struct att1_backend {
    const att1_backend_ops *ops;
    void *user_data;
};

att1_status_t att1_backend_cpu_f32_create(att1_backend **out_backend);
att1_status_t att1_backend_cpu_q8_create(att1_backend **out_backend);
int att1_backend_cuda_available(void);
att1_status_t att1_backend_cuda_create(att1_backend **out_backend);
att1_status_t att1_backend_cuda_q8_create(att1_backend **out_backend);
att1_status_t att1_backend_cuda_copy_host_to_device(att1_backend *backend,
                                                    void *device_dst,
                                                    const void *host_src,
                                                    size_t bytes);
att1_status_t att1_backend_cuda_copy_device_to_host(att1_backend *backend,
                                                    void *host_dst,
                                                    const void *device_src,
                                                    size_t bytes);
att1_status_t att1_backend_default_create(att1_backend **out_backend);
void att1_backend_destroy(att1_backend *backend);

#endif
