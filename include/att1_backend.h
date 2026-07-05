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

/*
 * M164: One-time shard transfer and tensor residency
 *
 * `att1_backend_pcie_load_tensor()` implements the M93 §8.12 model-load
 * data-movement contract ("Tensor shard (model weights): host -> device,
 * once, at model load"): it transfers `byte_length` bytes for a given
 * `tensor_id` from `host_addr` to `device_addr` (addresses in the DMA
 * simulator's configured host/device address spaces — see
 * att1_aimu_conformance_config's dma_host_.../dma_device_... fields, not raw
 * host pointers) via one or more frozen v1.0 (M159) att1_aimu_dma_desc
 * transfers issued through the backend's att1_aimu_conformance_endpoint.
 * Transfers larger than ATT1_AIMU_DMA_MAX_TRANSFER_BYTES are split into
 * multiple sequential descriptors, the last one flagged
 * ATT1_AIMU_DMA_FLAG_LAST_DESCRIPTOR.
 *
 * Once a tensor_id has been successfully loaded it is marked resident for
 * the lifetime of the backend. A second call for the same tensor_id is
 * rejected with ATT1_ERR_STATE and does not resubmit any DMA transfer —
 * this is the enforcement mechanism for "weights are never re-read by the
 * host during inference" (M93 §8.12): callers (e.g. the model loader) are
 * expected to call this once per tensor at load time and never again.
 *
 * Returns ATT1_ERR_INVALID_ARG for a NULL backend/non-pcie backend or a
 * zero byte_length. Returns ATT1_ERR_STATE if tensor_id is already
 * resident, or if the backend's resident-tensor table is full. Returns
 * whatever att1_status_t the underlying DMA submission failed with
 * otherwise.
 */
att1_status_t att1_backend_pcie_load_tensor(att1_backend *backend,
                                            uint32_t tensor_id,
                                            uint64_t host_addr,
                                            uint64_t device_addr,
                                            uint64_t byte_length,
                                            uint8_t dtype,
                                            uint8_t quant_group_size);

/*
 * att1_backend_pcie_tensor_is_resident
 *
 * Returns 1 if tensor_id has already been successfully transferred via
 * att1_backend_pcie_load_tensor() on this backend, 0 otherwise (including
 * when backend is NULL or not a pcie backend).
 */
int att1_backend_pcie_tensor_is_resident(att1_backend *backend, uint32_t tensor_id);

typedef struct att1_backend_pcie_residency_counters {
    uint64_t tensors_resident;             /**< distinct tensors currently resident  */
    uint64_t transfers_submitted;          /**< successful load_tensor() calls       */
    uint64_t descriptors_submitted;        /**< individual DMA descriptors issued    */
    uint64_t bytes_transferred;            /**< total payload bytes transferred H2D  */
    uint64_t duplicate_transfer_rejections; /**< re-load attempts rejected           */
} att1_backend_pcie_residency_counters;

/*
 * att1_backend_pcie_get_residency_counters
 *
 * Copies the backend's M164 residency counters into *out.
 *
 * Returns ATT1_ERR_INVALID_ARG if backend is NULL/not-pcie or out is NULL.
 */
att1_status_t att1_backend_pcie_get_residency_counters(
        att1_backend *backend,
        att1_backend_pcie_residency_counters *out);

#endif
