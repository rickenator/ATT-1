#include "att1_backend.h"

#include "att1_math.h"

#include <stdlib.h>

static void *cpu_q8_alloc(att1_backend *backend, size_t bytes)
{
    (void)backend;
    return malloc(bytes);
}

static void cpu_q8_free(att1_backend *backend, void *ptr)
{
    (void)backend;
    free(ptr);
}

static int cpu_q8_sync(att1_backend *backend)
{
    (void)backend;
    return 0;
}

static int cpu_q8_matmul_f32(att1_backend *backend,
                             float *dst,
                             const float *lhs,
                             const float *rhs,
                             size_t rows,
                             size_t cols,
                             size_t inner)
{
    (void)backend;
    return att1_matmul_f32(dst, lhs, rhs, rows, cols, inner);
}

static int cpu_q8_matmul_q8xf32(att1_backend *backend,
                                float *dst,
                                const float *lhs,
                                size_t lhs_rows,
                                size_t lhs_cols,
                                const att1_q8_matrix *weights)
{
    (void)backend;
    return att1_matmul_q8xf32(dst, lhs, lhs_rows, lhs_cols, weights);
}

static int cpu_q8_rmsnorm_f32(att1_backend *backend,
                              float *dst,
                              const float *src,
                              const float *weight,
                              size_t count,
                              float epsilon)
{
    (void)backend;
    return att1_rmsnorm_f32(dst, src, weight, count, epsilon);
}

static int cpu_q8_softmax_f32(att1_backend *backend,
                              float *values,
                              size_t count)
{
    (void)backend;
    return att1_softmax_f32(values, count);
}

static int cpu_q8_rope_f32(att1_backend *backend,
                           float *values,
                           size_t count,
                           size_t position,
                           float theta)
{
    (void)backend;
    return att1_rope_f32(values, count, position, theta);
}

static int cpu_q8_ffn_swiglu_f32(att1_backend *backend,
                                 float *dst,
                                 const float *gate,
                                 const float *value,
                                 size_t count)
{
    (void)backend;
    return att1_swiglu_f32(dst, gate, value, count);
}

static const att1_backend_ops cpu_q8_ops = {
    "cpu-q8",
    cpu_q8_alloc,
    cpu_q8_free,
    cpu_q8_sync,
    cpu_q8_matmul_f32,
    cpu_q8_matmul_q8xf32,
    cpu_q8_rmsnorm_f32,
    cpu_q8_softmax_f32,
    cpu_q8_rope_f32,
    cpu_q8_ffn_swiglu_f32
};

att1_status_t att1_backend_cpu_q8_create(att1_backend **out_backend)
{
    att1_backend *backend = NULL;

    if (out_backend == NULL) {
        return ATT1_ERR_INVALID_ARG;
    }
    *out_backend = NULL;

    backend = calloc(1u, sizeof(*backend));
    if (backend == NULL) {
        return ATT1_ERR_OOM;
    }

    backend->ops = &cpu_q8_ops;
    *out_backend = backend;
    return ATT1_OK;
}
