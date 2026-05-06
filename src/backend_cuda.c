#include "att1_backend.h"

#include <stdlib.h>

#ifdef ATT1_ENABLE_CUDA
#include <cuda_runtime_api.h>
#endif

typedef struct att1_cuda_backend_data {
    int device_id;
} att1_cuda_backend_data;

#ifdef ATT1_ENABLE_CUDA
static void *cuda_backend_alloc(att1_backend *backend, size_t bytes)
{
    void *ptr = NULL;

    if ((backend == NULL) || (bytes == 0u)) {
        return NULL;
    }

    if (cudaMalloc(&ptr, bytes) != cudaSuccess) {
        return NULL;
    }

    return ptr;
}

static void cuda_backend_free(att1_backend *backend, void *ptr)
{
    (void)backend;
    if (ptr != NULL) {
        (void)cudaFree(ptr);
    }
}

static int cuda_backend_sync(att1_backend *backend)
{
    (void)backend;
    return cudaDeviceSynchronize() == cudaSuccess ? 0 : -1;
}
#else
static void *cuda_backend_alloc(att1_backend *backend, size_t bytes)
{
    (void)backend;
    (void)bytes;
    return NULL;
}

static void cuda_backend_free(att1_backend *backend, void *ptr)
{
    (void)backend;
    (void)ptr;
}

static int cuda_backend_sync(att1_backend *backend)
{
    (void)backend;
    return -1;
}
#endif

static int cuda_backend_matmul_f32(att1_backend *backend,
                                   float *dst,
                                   const float *lhs,
                                   const float *rhs,
                                   size_t rows,
                                   size_t cols,
                                   size_t inner)
{
    (void)backend;
    (void)dst;
    (void)lhs;
    (void)rhs;
    (void)rows;
    (void)cols;
    (void)inner;
    return -1;
}

static int cuda_backend_matmul_q8xf32(att1_backend *backend,
                                      float *dst,
                                      const float *lhs,
                                      size_t lhs_rows,
                                      size_t lhs_cols,
                                      const att1_q8_matrix *weights)
{
    (void)backend;
    (void)dst;
    (void)lhs;
    (void)lhs_rows;
    (void)lhs_cols;
    (void)weights;
    return -1;
}

static int cuda_backend_rmsnorm_f32(att1_backend *backend,
                                    float *dst,
                                    const float *src,
                                    const float *weight,
                                    size_t count,
                                    float epsilon)
{
    (void)backend;
    (void)dst;
    (void)src;
    (void)weight;
    (void)count;
    (void)epsilon;
    return -1;
}

static int cuda_backend_softmax_f32(att1_backend *backend,
                                    float *values,
                                    size_t count)
{
    (void)backend;
    (void)values;
    (void)count;
    return -1;
}

static int cuda_backend_rope_f32(att1_backend *backend,
                                 float *values,
                                 size_t count,
                                 size_t position,
                                 float theta)
{
    (void)backend;
    (void)values;
    (void)count;
    (void)position;
    (void)theta;
    return -1;
}

static int cuda_backend_ffn_swiglu_f32(att1_backend *backend,
                                       float *dst,
                                       const float *gate,
                                       const float *value,
                                       size_t count)
{
    (void)backend;
    (void)dst;
    (void)gate;
    (void)value;
    (void)count;
    return -1;
}

static const att1_backend_ops cuda_backend_ops = {
    "cuda",
    cuda_backend_alloc,
    cuda_backend_free,
    cuda_backend_sync,
    cuda_backend_matmul_f32,
    cuda_backend_matmul_q8xf32,
    cuda_backend_rmsnorm_f32,
    cuda_backend_softmax_f32,
    cuda_backend_rope_f32,
    cuda_backend_ffn_swiglu_f32
};

int att1_backend_cuda_available(void)
{
#ifdef ATT1_ENABLE_CUDA
    int count = 0;

    if (cudaGetDeviceCount(&count) != cudaSuccess) {
        return 0;
    }

    return count > 0 ? 1 : 0;
#else
    return 0;
#endif
}

att1_status_t att1_backend_cuda_create(att1_backend **out_backend)
{
    att1_backend *backend = NULL;
    att1_cuda_backend_data *data = NULL;

    if (out_backend == NULL) {
        return ATT1_ERR_INVALID_ARG;
    }
    *out_backend = NULL;

#ifdef ATT1_ENABLE_CUDA
    if (!att1_backend_cuda_available()) {
        return ATT1_ERR_UNSUPPORTED;
    }
#else
    return ATT1_ERR_UNSUPPORTED;
#endif

    backend = calloc(1u, sizeof(*backend));
    data = calloc(1u, sizeof(*data));
    if ((backend == NULL) || (data == NULL)) {
        free(data);
        free(backend);
        return ATT1_ERR_OOM;
    }

    data->device_id = 0;
    backend->ops = &cuda_backend_ops;
    backend->user_data = data;
    *out_backend = backend;
    return ATT1_OK;
}

att1_status_t att1_backend_cuda_copy_host_to_device(att1_backend *backend,
                                                    void *device_dst,
                                                    const void *host_src,
                                                    size_t bytes)
{
    if ((backend == NULL) || (device_dst == NULL) || (host_src == NULL) ||
        (bytes == 0u)) {
        return ATT1_ERR_INVALID_ARG;
    }

#ifdef ATT1_ENABLE_CUDA
    if (backend->ops != &cuda_backend_ops) {
        return ATT1_ERR_INVALID_ARG;
    }

    return cudaMemcpy(device_dst, host_src, bytes, cudaMemcpyHostToDevice) ==
        cudaSuccess ? ATT1_OK : ATT1_ERR_STATE;
#else
    (void)backend;
    (void)device_dst;
    (void)host_src;
    (void)bytes;
    return ATT1_ERR_UNSUPPORTED;
#endif
}

att1_status_t att1_backend_cuda_copy_device_to_host(att1_backend *backend,
                                                    void *host_dst,
                                                    const void *device_src,
                                                    size_t bytes)
{
    if ((backend == NULL) || (host_dst == NULL) || (device_src == NULL) ||
        (bytes == 0u)) {
        return ATT1_ERR_INVALID_ARG;
    }

#ifdef ATT1_ENABLE_CUDA
    if (backend->ops != &cuda_backend_ops) {
        return ATT1_ERR_INVALID_ARG;
    }

    return cudaMemcpy(host_dst, device_src, bytes, cudaMemcpyDeviceToHost) ==
        cudaSuccess ? ATT1_OK : ATT1_ERR_STATE;
#else
    (void)backend;
    (void)host_dst;
    (void)device_src;
    (void)bytes;
    return ATT1_ERR_UNSUPPORTED;
#endif
}
