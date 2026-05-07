/*
 * backend_cpu_q4.c  -  M78: cpu-q4 backend
 *
 * Provides CPU memory management plus the standard f32 ops (rmsnorm, softmax,
 * rope, swiglu).  Q4 weight matmuls are called directly via att1_matmul_q4xf32
 * in the inference path rather than through the ops table; matmul_f32 is
 * included for completeness and matmul_q8xf32 is left NULL to prevent silent
 * q8 fallback.
 */

#include "att1_backend.h"

#include "att1_math.h"

#include <stdlib.h>

static void *cpu_q4_alloc(att1_backend *backend, size_t bytes)
{
    (void)backend;
    return malloc(bytes);
}

static void cpu_q4_free(att1_backend *backend, void *ptr)
{
    (void)backend;
    free(ptr);
}

static int cpu_q4_sync(att1_backend *backend)
{
    (void)backend;
    return 0;
}

static int cpu_q4_matmul_f32(att1_backend *backend,
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

static int cpu_q4_rmsnorm_f32(att1_backend *backend,
                               float *dst,
                               const float *src,
                               const float *weight,
                               size_t count,
                               float epsilon)
{
    (void)backend;
    return att1_rmsnorm_f32(dst, src, weight, count, epsilon);
}

static int cpu_q4_softmax_f32(att1_backend *backend,
                               float *values,
                               size_t count)
{
    (void)backend;
    return att1_softmax_f32(values, count);
}

static int cpu_q4_rope_f32(att1_backend *backend,
                            float *values,
                            size_t count,
                            size_t position,
                            float theta)
{
    (void)backend;
    return att1_rope_f32(values, count, position, theta);
}

static int cpu_q4_ffn_swiglu_f32(att1_backend *backend,
                                  float *dst,
                                  const float *gate,
                                  const float *value,
                                  size_t count)
{
    (void)backend;
    return att1_swiglu_f32(dst, gate, value, count);
}

static const att1_backend_ops cpu_q4_ops = {
    "cpu-q4",
    cpu_q4_alloc,
    cpu_q4_free,
    cpu_q4_sync,
    cpu_q4_matmul_f32,
    NULL,                   /* matmul_q8xf32: not supported; NULL prevents silent q8 fallback */
    cpu_q4_rmsnorm_f32,
    cpu_q4_softmax_f32,
    cpu_q4_rope_f32,
    cpu_q4_ffn_swiglu_f32
};

att1_status_t att1_backend_cpu_q4_create(att1_backend **out_backend)
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

    backend->ops = &cpu_q4_ops;
    *out_backend = backend;
    return ATT1_OK;
}
