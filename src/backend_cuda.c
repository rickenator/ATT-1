#include "att1_backend.h"

#include <math.h>
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
static int cuda_backend_vector_mul_f32(cublasHandle_t handle,
                                       float *dst,
                                       const float *lhs,
                                       const float *rhs,
                                       size_t count)
{
    if ((dst == NULL) || (lhs == NULL) || (rhs == NULL)) {
        return -1;
    }
    if ((count == 0u) || (count > (size_t)INT_MAX)) {
        return -1;
    }

    if (cublasSdgmm(handle,
                    CUBLAS_SIDE_LEFT,
                    (int)count,
                    1,
                    lhs,
                    (int)count,
                    rhs,
                    1,
                    dst,
                    (int)count) != CUBLAS_STATUS_SUCCESS) {
        return -1;
    }

    return 0;
}
#endif

#ifdef ATT1_ENABLE_CUDA
static void *cuda_backend_alloc(att1_backend *backend, size_t bytes)
{
    void *ptr = NULL;

    if ((backend == NULL) || (bytes == 0u)) {
        return NULL;
    }

    /* Allocate host memory for intermediate buffers (scores, query, key, value, context).
     * Individual CUDA operations (matmul, rope, softmax) manage device memory internally.
     * This ensures attention operations can read/write buffers from the host. */
    ptr = malloc(bytes);
    return ptr;
}

static void cuda_backend_free(att1_backend *backend, void *ptr)
{
    (void)backend;
    if (ptr != NULL) {
        free(ptr);
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
#ifdef ATT1_ENABLE_CUDA
    float *dequant_rhs = NULL;
    size_t row = 0u;
    size_t col = 0u;
    int status = -1;

    if ((backend == NULL) || (backend->user_data == NULL) ||
        (dst == NULL) || (lhs == NULL) || (weights == NULL) ||
        (weights->values == NULL) || (weights->scales == NULL)) {
        return -1;
    }
    if ((lhs_rows == 0u) || (lhs_cols == 0u) ||
        (weights->rows == 0u) || (weights->cols == 0u) ||
        (lhs_cols != weights->cols)) {
        return -1;
    }
    if (weights->cols > ((size_t)-1) / weights->rows) {
        return -1;
    }

    dequant_rhs = malloc(weights->cols * weights->rows * sizeof(*dequant_rhs));
    if (dequant_rhs == NULL) {
        return -1;
    }

    for (row = 0u; row < weights->rows; row++) {
        const float scale = weights->scales[row];

        for (col = 0u; col < weights->cols; col++) {
            const int qvalue = weights->values[(row * weights->cols) + col];
            dequant_rhs[(col * weights->rows) + row] =
                (float)qvalue * scale;
        }
    }

    status = cuda_backend_matmul_f32(backend,
                                     dst,
                                     lhs,
                                     dequant_rhs,
                                     lhs_rows,
                                     weights->rows,
                                     lhs_cols);

    free(dequant_rhs);
    return status;
#else
    (void)backend;
    (void)dst;
    (void)lhs;
    (void)lhs_rows;
    (void)lhs_cols;
    (void)weights;
    return -1;
#endif
}

static int cuda_backend_matmul_q4xf32(att1_backend *backend,
                                      float *dst,
                                      const float *lhs,
                                      size_t lhs_rows,
                                      size_t lhs_cols,
                                      const att1_q4_matrix *weights)
{
#ifdef ATT1_ENABLE_CUDA
    float *dequant_rhs = NULL;
    size_t row = 0u;
    size_t grp = 0u;
    size_t col = 0u;
    size_t groups_per_row = 0u;
    int status = -1;

    if ((backend == NULL) || (backend->user_data == NULL) ||
        (dst == NULL) || (lhs == NULL) || (weights == NULL) ||
        (weights->packed == NULL) || (weights->scales == NULL)) {
        return -1;
    }
    if ((lhs_rows == 0u) || (lhs_cols == 0u) ||
        (weights->rows == 0u) || (weights->cols == 0u) ||
        (lhs_cols != weights->cols) || (weights->group_size == 0u) ||
        ((weights->cols % (size_t)weights->group_size) != 0u)) {
        return -1;
    }
    if (weights->cols > ((size_t)-1) / weights->rows) {
        return -1;
    }

    groups_per_row = weights->cols / (size_t)weights->group_size;

    dequant_rhs = malloc(weights->cols * weights->rows * sizeof(*dequant_rhs));
    if (dequant_rhs == NULL) {
        return -1;
    }

    /* Dequantize q4 weights into transposed column-major layout for
     * cublasSgemm, mirroring the q8 approach. Each group of group_size
     * packed nibbles is expanded to float32 values and stored at
     * dequant_rhs[col * weights->rows + row]. */
    for (row = 0u; row < weights->rows; row++) {
        const float *row_scales = &weights->scales[row * groups_per_row];
        const uint8_t *row_packed =
            &weights->packed[(row * weights->cols) / 2u];

        for (grp = 0u; grp < groups_per_row; grp++) {
            const float scale = row_scales[grp];
            const uint8_t *grp_packed =
                row_packed + (grp * (size_t)weights->group_size) / 2u;

            for (col = 0u; col < (size_t)weights->group_size; col += 2u) {
                const size_t abs_col = grp * (size_t)weights->group_size + col;
                const uint8_t byte = grp_packed[col / 2u];
                int8_t v0 = (int8_t)(byte & 0x0Fu);
                int8_t v1 = (int8_t)((byte >> 4u) & 0x0Fu);

                /* Sign-extend 4-bit signed values [-7, 7] */
                if (v0 > 7) { v0 = (int8_t)(v0 - 16); }
                if (v1 > 7) { v1 = (int8_t)(v1 - 16); }

                /* Transposed: column-major layout for cublasSgemm */
                dequant_rhs[abs_col * weights->rows + row] =
                    (float)v0 * scale;
                dequant_rhs[(abs_col + 1u) * weights->rows + row] =
                    (float)v1 * scale;
            }
        }
    }

    status = cuda_backend_matmul_f32(backend,
                                     dst,
                                     lhs,
                                     dequant_rhs,
                                     lhs_rows,
                                     weights->rows,
                                     lhs_cols);

    free(dequant_rhs);
    return status;
#else
    (void)backend;
    (void)dst;
    (void)lhs;
    (void)lhs_rows;
    (void)lhs_cols;
    (void)weights;
    return -1;
#endif
}

static int cuda_backend_rmsnorm_f32(att1_backend *backend,
                                    float *dst,
                                    const float *src,
                                    const float *weight,
                                    size_t count,
                                    float epsilon)
{
#ifdef ATT1_ENABLE_CUDA
    att1_cuda_backend_data *data = NULL;
    cublasHandle_t handle;
    float *d_src = NULL;
    float *d_weight = NULL;
    float *d_dst = NULL;
    float sum_squares = 0.0f;
    float scale = 0.0f;
    int ok = 0;
    int handle_valid = 0;

    if ((backend == NULL) || (backend->user_data == NULL) ||
        (dst == NULL) || (src == NULL) || (weight == NULL)) {
        return -1;
    }
    if ((count == 0u) || (epsilon <= 0.0f) ||
        (count > (size_t)INT_MAX)) {
        return -1;
    }

    data = (att1_cuda_backend_data *)backend->user_data;
    if (cudaSetDevice(data->device_id) != cudaSuccess) {
        return -1;
    }

    if (cudaMalloc((void **)&d_src, count * sizeof(float)) != cudaSuccess) {
        goto cleanup;
    }
    if (cudaMalloc((void **)&d_weight, count * sizeof(float)) != cudaSuccess) {
        goto cleanup;
    }
    if (cudaMalloc((void **)&d_dst, count * sizeof(float)) != cudaSuccess) {
        goto cleanup;
    }

    if (cudaMemcpy(d_src,
                   src,
                   count * sizeof(float),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
        goto cleanup;
    }
    if (cudaMemcpy(d_weight,
                   weight,
                   count * sizeof(float),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
        goto cleanup;
    }

    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
        goto cleanup;
    }
    handle_valid = 1;

    if (cublasSdot(handle,
                   (int)count,
                   d_src,
                   1,
                   d_src,
                   1,
                   &sum_squares) != CUBLAS_STATUS_SUCCESS) {
        goto cleanup;
    }

    scale = 1.0f / sqrtf((sum_squares / (float)count) + epsilon);
    if (cuda_backend_vector_mul_f32(handle,
                                    d_dst,
                                    d_src,
                                    d_weight,
                                    count) != 0) {
        goto cleanup;
    }
    if (cublasSscal(handle,
                    (int)count,
                    &scale,
                    d_dst,
                    1) != CUBLAS_STATUS_SUCCESS) {
        goto cleanup;
    }

    if (cudaMemcpy(dst,
                   d_dst,
                   count * sizeof(float),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        goto cleanup;
    }
    ok = 1;

cleanup:
    if (handle_valid) {
        (void)cublasDestroy(handle);
    }
    (void)cudaFree(d_src);
    (void)cudaFree(d_weight);
    (void)cudaFree(d_dst);
    return ok ? 0 : -1;
#else
    (void)backend;
    (void)dst;
    (void)src;
    (void)weight;
    (void)count;
    (void)epsilon;
    return -1;
#endif
}

static int cuda_backend_softmax_f32(att1_backend *backend,
                                    float *values,
                                    size_t count)
{
    float max_val = -INFINITY;
    float sum_exp = 0.0f;
    size_t i = 0u;

    if ((backend == NULL) || (values == NULL)) {
        return -1;
    }
    if (count == 0u) {
        return -1;
    }

    /* Find maximum for numerical stability. */
    for (i = 0u; i < count; i++) {
        if (values[i] > max_val) {
            max_val = values[i];
        }
    }

    /* Compute exp(x - max) and sum. */
    for (i = 0u; i < count; i++) {
        values[i] = expf(values[i] - max_val);
        sum_exp += values[i];
    }

    /* Normalize by sum. */
    if (sum_exp <= 0.0f) {
        return -1;
    }
    for (i = 0u; i < count; i++) {
        values[i] /= sum_exp;
    }

    return 0;
}

static int cuda_backend_rope_f32(att1_backend *backend,
                                 float *values,
                                 size_t count,
                                 size_t position,
                                 float theta)
{
#ifdef ATT1_ENABLE_CUDA
    att1_cuda_backend_data *data = NULL;
    cublasHandle_t handle;
    float *d_values = NULL;
    int ok = 0;
    int handle_valid = 0;
    size_t i = 0u;

    if ((backend == NULL) || (backend->user_data == NULL) ||
        (values == NULL)) {
        return -1;
    }
    if ((count == 0u) || ((count % 2u) != 0u) ||
        (theta <= 0.0f) || (count > (size_t)INT_MAX)) {
        return -1;
    }

    data = (att1_cuda_backend_data *)backend->user_data;
    if (cudaSetDevice(data->device_id) != cudaSuccess) {
        return -1;
    }

    if (cudaMalloc((void **)&d_values, count * sizeof(float)) != cudaSuccess) {
        goto cleanup;
    }
    if (cudaMemcpy(d_values,
                   values,
                   count * sizeof(float),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
        goto cleanup;
    }

    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
        goto cleanup;
    }
    handle_valid = 1;

    for (i = 0u; i < count; i += 2u) {
        const float exponent = (float)i / (float)count;
        const float frequency = 1.0f / powf(theta, exponent);
        const float angle = (float)position * frequency;
        const float c = cosf(angle);
        /* cublasSrot uses [x'; y'] = [c s; -s c] [x; y].
           Use -sin(angle) to match RoPE's [c -s; s c]. */
        const float s = -sinf(angle);

        if (cublasSrot(handle,
                       1,
                       d_values + i,
                       1,
                       d_values + i + 1u,
                       1,
                       &c,
                       &s) != CUBLAS_STATUS_SUCCESS) {
            goto cleanup;
        }
    }

    if (cudaMemcpy(values,
                   d_values,
                   count * sizeof(float),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        goto cleanup;
    }
    ok = 1;

cleanup:
    if (handle_valid) {
        (void)cublasDestroy(handle);
    }
    (void)cudaFree(d_values);
    return ok ? 0 : -1;
#else
    (void)backend;
    (void)values;
    (void)count;
    (void)position;
    (void)theta;
    return -1;
#endif
}

static int cuda_backend_ffn_swiglu_f32(att1_backend *backend,
                                       float *dst,
                                       const float *gate,
                                       const float *value,
                                       size_t count)
{
#ifdef ATT1_ENABLE_CUDA
    att1_cuda_backend_data *data = NULL;
    cublasHandle_t handle;
    float *d_gate = NULL;
    float *d_value = NULL;
    float *d_dst = NULL;
    float *host_silu = NULL;
    int ok = 0;
    int handle_valid = 0;
    size_t i = 0u;

    if ((backend == NULL) || (backend->user_data == NULL) ||
        (dst == NULL) || (gate == NULL) || (value == NULL)) {
        return -1;
    }
    if ((count == 0u) || (count > (size_t)INT_MAX)) {
        return -1;
    }

    host_silu = malloc(count * sizeof(float));
    if (host_silu == NULL) {
        return -1;
    }

    for (i = 0u; i < count; i++) {
        host_silu[i] = gate[i] / (1.0f + expf(-gate[i]));
    }

    data = (att1_cuda_backend_data *)backend->user_data;
    if (cudaSetDevice(data->device_id) != cudaSuccess) {
        goto cleanup;
    }

    if (cudaMalloc((void **)&d_gate, count * sizeof(float)) != cudaSuccess) {
        goto cleanup;
    }
    if (cudaMalloc((void **)&d_value, count * sizeof(float)) != cudaSuccess) {
        goto cleanup;
    }
    if (cudaMalloc((void **)&d_dst, count * sizeof(float)) != cudaSuccess) {
        goto cleanup;
    }
    if (cudaMemcpy(d_gate,
                   host_silu,
                   count * sizeof(float),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
        goto cleanup;
    }
    if (cudaMemcpy(d_value,
                   value,
                   count * sizeof(float),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
        goto cleanup;
    }

    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
        goto cleanup;
    }
    handle_valid = 1;

    if (cuda_backend_vector_mul_f32(handle,
                                    d_dst,
                                    d_gate,
                                    d_value,
                                    count) != 0) {
        goto cleanup;
    }

    if (cudaMemcpy(dst,
                   d_dst,
                   count * sizeof(float),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        goto cleanup;
    }
    ok = 1;

cleanup:
    if (handle_valid) {
        (void)cublasDestroy(handle);
    }
    (void)cudaFree(d_gate);
    (void)cudaFree(d_value);
    (void)cudaFree(d_dst);
    free(host_silu);
    return ok ? 0 : -1;
#else
    (void)backend;
    (void)dst;
    (void)gate;
    (void)value;
    (void)count;
    return -1;
#endif
}

static const att1_backend_ops cuda_backend_ops = {
    "cuda",
    cuda_backend_alloc,
    cuda_backend_free,
    cuda_backend_sync,
    cuda_backend_matmul_f32,
    cuda_backend_matmul_q8xf32,
    NULL, /* matmul_q4xf32: use cuda-q4 backend */
    cuda_backend_rmsnorm_f32,
    cuda_backend_softmax_f32,
    cuda_backend_rope_f32,
    cuda_backend_ffn_swiglu_f32
};

static const att1_backend_ops cuda_q8_backend_ops = {
    "cuda-q8",
    cuda_backend_alloc,
    cuda_backend_free,
    cuda_backend_sync,
    cuda_backend_matmul_f32,
    cuda_backend_matmul_q8xf32,
    NULL, /* matmul_q4xf32: not supported on cuda-q8 */
    cuda_backend_rmsnorm_f32,
    cuda_backend_softmax_f32,
    cuda_backend_rope_f32,
    cuda_backend_ffn_swiglu_f32
};
/* M88: cuda-q4 backend — full single-tile inference with CUDA q4 matmul. */
static const att1_backend_ops cuda_q4_backend_ops = {
    "cuda-q4",
    cuda_backend_alloc,
    cuda_backend_free,
    cuda_backend_sync,
    cuda_backend_matmul_f32,
    NULL, /* matmul_q8xf32: not supported on cuda-q4 */
    cuda_backend_matmul_q4xf32,
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

static att1_status_t cuda_backend_create_with_ops(
    att1_backend **out_backend,
    const att1_backend_ops *ops)
{
    att1_backend *backend = NULL;
    att1_cuda_backend_data *data = NULL;

    if ((out_backend == NULL) || (ops == NULL)) {
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
    backend->ops = ops;
    backend->user_data = data;
    *out_backend = backend;
    return ATT1_OK;
}

att1_status_t att1_backend_cuda_create(att1_backend **out_backend)
{
    return cuda_backend_create_with_ops(out_backend, &cuda_backend_ops);
}

att1_status_t att1_backend_cuda_q8_create(att1_backend **out_backend)
{
    return cuda_backend_create_with_ops(out_backend, &cuda_q8_backend_ops);
}

att1_status_t att1_backend_cuda_q4_create(att1_backend **out_backend)
{
    return cuda_backend_create_with_ops(out_backend, &cuda_q4_backend_ops);
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
    if ((backend->ops != &cuda_backend_ops) &&
        (backend->ops != &cuda_q8_backend_ops) &&
        (backend->ops != &cuda_q4_backend_ops)) {
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
    if ((backend->ops != &cuda_backend_ops) &&
        (backend->ops != &cuda_q8_backend_ops) &&
        (backend->ops != &cuda_q4_backend_ops)) {
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
