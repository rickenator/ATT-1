#ifndef ATT1_BACKEND_H
#define ATT1_BACKEND_H

#include "att1_quant.h"
#include "att1_status.h"

#include <stddef.h>
#include <stdint.h>

typedef struct att1_backend att1_backend;

/* Opaque forward declaration; full definition lives in
 * att1_aimu_conformance.h. Only a pointer is needed here so
 * att1_backend.h does not pull in the AIMU conformance/command-queue
 * header graph for consumers that never use the pcie backend. */
typedef struct att1_aimu_conformance_endpoint att1_aimu_conformance_endpoint;

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

    int (*matmul_q4xf32)(att1_backend *backend,
                         float *dst,
                         const float *lhs,
                         size_t lhs_rows,
                         size_t lhs_cols,
                         const att1_q4_matrix *weights);

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
att1_status_t att1_backend_cpu_q4_create(att1_backend **out_backend);
int att1_backend_cuda_available(void);
att1_status_t att1_backend_cuda_create(att1_backend **out_backend);
att1_status_t att1_backend_cuda_q8_create(att1_backend **out_backend);
att1_status_t att1_backend_cuda_q4_create(att1_backend **out_backend);
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

/*
 * att1_backend_pcie_create  (M163)
 *
 * Creates an att1_backend_ops implementation that dispatches every
 * operation through an att1_aimu_conformance_endpoint (M161/M162) instead
 * of calling CPU/CUDA functions directly, exercising the M93 §8.3
 * backend-swap contract over the out-of-process transport.
 *
 * `endpoint` must already be connected/created (in-process via
 * att1_aimu_conformance_inproc_create(), or out-of-process via
 * att1_aimu_conformance_socket_connect()) and remains owned by the caller:
 * att1_backend_destroy() on the returned backend never destroys it.
 *
 * `tile_id` selects which simulated/emulated AIMU tile every submitted
 * command targets.
 *
 * alloc/free manage ordinary host memory for staging operands and results
 * (mirroring the CUDA backend's host-side buffer convention); no tile-local
 * tensor residency or DMA transfer is implemented yet (M164/M165).
 *
 * Every math operation (matmul_f32, matmul_q8xf32, matmul_q4xf32,
 * rmsnorm_f32, rope_f32, ffn_swiglu_f32) is issued as one frozen v1.0
 * (M158) command packet, dispatched, and polled to completion; the
 * completion result code is mapped via att1_aimu_result_to_status() to
 * decide success/failure. The in-process/out-of-process command-queue
 * simulator does not yet execute EXEC_* commands (M161/M162 dispatch
 * returns ATT1_AIMU_ERR_UNSUPPORTED_OP for all of them by design), so
 * these calls currently fail with ATT1_ERR_UNSUPPORTED once real command
 * execution lands in a later milestone (M166); this milestone validates
 * only the submit/dispatch/poll transport plumbing and the backend-swap
 * pattern, not compute correctness. softmax_f32 has no dedicated frozen
 * command type (only EXEC_ATTENTION exists, which is a larger fused op),
 * so it is left unimplemented (NULL) on this backend.
 */
att1_status_t att1_backend_pcie_create(att1_aimu_conformance_endpoint *endpoint,
                                       uint32_t tile_id,
                                       att1_backend **out_backend);

#endif
