#include "att1_backend.h"

#include <stdlib.h>

#ifdef ATT1_ENABLE_CUDA
#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <limits.h>
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
#ifdef ATT1_ENABLE_CUDA
    att1_cuda_backend_data *data = (att1_cuda_backend_data *)backend->user_data;
    cublasHandle_t handle;
    float *d_lhs = NULL;
    float *d_rhs = NULL;
    float *d_dst = NULL;
    const float alpha = 1.0f;
    const float beta = 0.0f;
    int ok = 0;
    int handle_valid = 0;

    if ((dst == NULL) || (lhs == NULL) || (rhs == NULL)) {
        return -1;
    }
    if ((rows == 0u) || (cols == 0u) || (inner == 0u)) {
        return -1;
    }
    if ((rows > (size_t)INT_MAX) || (cols > (size_t)INT_MAX) ||
        (inner > (size_t)INT_MAX)) {
        return -1;
    }

    if (cudaSetDevice(data->device_id) != cudaSuccess) {
        return -1;
    }

    if (cudaMalloc((void **)&d_lhs,
                   rows * inner * sizeof(float)) != cudaSuccess) {
        goto cleanup;
    }
    if (cudaMalloc((void **)&d_rhs,
                   inner * cols * sizeof(float)) != cudaSuccess) {
        goto cleanup;
    }
    if (cudaMalloc((void **)&d_dst,
                   rows * cols * sizeof(float)) != cudaSuccess) {
        goto cleanup;
    }
    if (cudaMemcpy(d_lhs, lhs,
                   rows * inner * sizeof(float),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
        goto cleanup;
    }
    if (cudaMemcpy(d_rhs, rhs,
                   inner * cols * sizeof(float),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
        goto cleanup;
    }

    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
        goto cleanup;
    }
    handle_valid = 1;

    /*
     * Row-major C[M,N] = A[M,K] * B[K,N] via the cuBLAS column-major trick:
     *   C^T = B^T * A^T
     * Pass rhs as the first cuBLAS argument (read as N×K column-major)
     * and lhs as the second (read as K×M column-major).  cuBLAS produces
     * C^T in column-major (N×M), which is the row-major result C[M,N].
     *   m = N = cols,  n = M = rows,  k = K = inner
     *   lda = N = cols,  ldb = K = inner,  ldc = N = cols
     */
    if (cublasSgemm(handle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    (int)cols, (int)rows, (int)inner,
                    &alpha,
                    d_rhs, (int)cols,
                    d_lhs, (int)inner,
                    &beta,
                    d_dst, (int)cols) != CUBLAS_STATUS_SUCCESS) {
        goto cleanup;
    }

    if (cudaMemcpy(dst, d_dst,
                   rows * cols * sizeof(float),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        goto cleanup;
    }
    ok = 1;

cleanup:
    if (handle_valid) {
        (void)cublasDestroy(handle);
    }
    (void)cudaFree(d_lhs);
    (void)cudaFree(d_rhs);
    (void)cudaFree(d_dst);
    return ok ? 0 : -1;
#else
    (void)backend;
    (void)dst;
    (void)lhs;
    (void)rhs;
    (void)rows;
    (void)cols;
    (void)inner;
    return -1;
#endif
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
