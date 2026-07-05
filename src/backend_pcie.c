/*
 * backend_pcie.c  —  M163 `backend_pcie.c` host backend; M166 real EXEC_*
 * execution.
 *
 * A concrete att1_backend_ops implementation (M93 §8.3) that dispatches
 * every operation through an att1_aimu_conformance_endpoint (M161/M162)
 * instead of calling CPU/CUDA functions directly. This validates the
 * backend-swap pattern already proven by backend_cuda.c: the rest of the
 * runtime (attention.c, transformer_block.c, infer.c, ...) is unaware of
 * which concrete backend it is talking to.
 *
 * Every math operation builds one frozen v1.0 (M158) command packet,
 * submits it, dispatches it, and polls for the matching completion; the
 * completion result code is mapped to att1_status_t via
 * att1_aimu_result_to_status(). As of M166 the in-process endpoint's
 * command-queue simulator actually executes EXEC_MATMUL/EXEC_RMSNORM/
 * EXEC_ROPE/EXEC_FFN against the real operand buffers (see the exec hook
 * installed in src/aimu_conformance.c), so these calls now produce results
 * that match the CPU backends exactly (f32/q8/q4 all reuse the same
 * att1_math.h / att1_quant.h primitives the CPU backends call directly).
 *
 * `EXEC_MATMUL`/`EXEC_RMSNORM` reference their weight tensor by `tensor_id`
 * rather than passing a third buffer pointer in the (frozen, two-address)
 * command packet, matching the real hardware residency model (M93 §8.12):
 * this backend auto-registers a weight pointer the first time it is used
 * (submitting one `LOAD_TENSOR_TILE` command) and reuses that `tensor_id`
 * on every subsequent call with the same pointer, since the generic
 * att1_backend_ops interface has no explicit "load once" call for
 * rmsnorm/matmul weights (unlike att1_backend_pcie_load_tensor(), which is
 * an explicit, separate API for model-load shard transfers, M164).
 * `EXEC_FFN`'s swiglu combine step needs two input buffers (gate, value)
 * and has no resident weight, so this backend packs them contiguously into
 * one scratch buffer and passes a single input address instead.
 * `softmax_f32` still has no dedicated frozen command type (M103 §4.6: only
 * the larger fused `EXEC_ATTENTION` covers softmax), so it is computed
 * locally on the host rather than round-tripped through the endpoint.
 */

#include "att1_backend.h"

#include "att1_aimu_cmdq.h"
#include "att1_aimu_conformance.h"
#include "att1_aimu_dma.h"
#include "att1_math.h"
#include "att1_quant.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* M164: fixed-capacity resident-tensor table. A fixed array (rather than a
 * separate heap allocation) is used because att1_backend_destroy() is a
 * generic helper shared by every backend that only frees backend->user_data
 * and backend itself; embedding the table avoids a second allocation that
 * would otherwise need backend-specific cleanup. 4096 entries comfortably
 * covers every weight tensor of any tiny/small fixture model used today. */
#define ATT1_BACKEND_PCIE_MAX_RESIDENT_TENSORS 4096u

/* M166: fixed-capacity auto weight-tensor registry (see file header). */
#define ATT1_BACKEND_PCIE_MAX_AUTO_TENSORS 512u

typedef struct att1_pcie_resident_tensor {
    uint32_t tensor_id;
    uint64_t device_addr;
    uint64_t byte_length;
} att1_pcie_resident_tensor;

typedef struct att1_pcie_auto_tensor {
    const void *ptr;
    uint8_t     dtype;
    uint16_t    tensor_id;
} att1_pcie_auto_tensor;

typedef struct att1_pcie_backend_data {
    att1_aimu_conformance_endpoint *endpoint; /* not owned */
    uint32_t                        tile_id;

    /* M164: one-time shard transfer / tensor residency tracking. */
    att1_pcie_resident_tensor resident[ATT1_BACKEND_PCIE_MAX_RESIDENT_TENSORS];
    size_t                    resident_count;
    uint32_t                  next_descriptor_id;
    att1_backend_pcie_residency_counters residency_counters;

    /* M166: auto weight-tensor registry keyed by host pointer identity. */
    att1_pcie_auto_tensor auto_tensors[ATT1_BACKEND_PCIE_MAX_AUTO_TENSORS];
    size_t                auto_tensor_count;
    uint16_t              next_auto_tensor_id;
} att1_pcie_backend_data;

static att1_pcie_backend_data *pcie_data(att1_backend *backend)
{
    if ((backend == NULL) || (backend->user_data == NULL)) {
        return NULL;
    }
    return (att1_pcie_backend_data *)backend->user_data;
}

/*
 * pcie_submit_exec
 *
 * Submits a fully-populated command targeting the backend's tile,
 * dispatches it, polls its completion, and maps the completion's result
 * code to att1_status_t. Returns 0 for ATT1_OK, -1 otherwise.
 */
static int pcie_submit_exec(att1_backend *backend, att1_aimu_cmd *cmd)
{
    att1_pcie_backend_data *data = pcie_data(backend);
    att1_aimu_completion completion;

    if ((data == NULL) || (cmd == NULL)) {
        return -1;
    }

    cmd->tile_id = (uint8_t)data->tile_id;

    if (att1_aimu_conformance_cmd_submit(data->endpoint, cmd) != ATT1_OK) {
        return -1;
    }
    if (att1_aimu_conformance_cmd_dispatch_one(data->endpoint) != ATT1_OK) {
        return -1;
    }
    memset(&completion, 0, sizeof(completion));
    if (att1_aimu_conformance_cmd_poll_completion(data->endpoint, &completion) != ATT1_OK) {
        return -1;
    }

    return (att1_aimu_result_to_status((att1_aimu_result)completion.result_code) == ATT1_OK)
                   ? 0
                   : -1;
}

/*
 * pcie_register_weight
 *
 * Looks up `ptr` (a weight/norm-vector pointer already resident in host
 * memory) in this backend's auto weight-tensor registry by pointer+dtype
 * identity; if not yet present, submits one LOAD_TENSOR_TILE command to
 * register it under a freshly assigned tensor_id and records the mapping.
 * Returns 0 and writes *out_tensor_id on success, -1 otherwise.
 */
static int pcie_register_weight(att1_backend *backend,
                                const void *ptr,
                                uint8_t dtype,
                                uint16_t dim0,
                                uint16_t dim1,
                                uint16_t *out_tensor_id)
{
    att1_pcie_backend_data *data = pcie_data(backend);
    att1_aimu_cmd cmd;
    size_t i;

    if ((data == NULL) || (ptr == NULL) || (out_tensor_id == NULL)) {
        return -1;
    }

    for (i = 0u; i < data->auto_tensor_count; i++) {
        if ((data->auto_tensors[i].ptr == ptr) && (data->auto_tensors[i].dtype == dtype)) {
            *out_tensor_id = data->auto_tensors[i].tensor_id;
            return 0;
        }
    }

    if ((data->auto_tensor_count >= ATT1_BACKEND_PCIE_MAX_AUTO_TENSORS) ||
        (data->next_auto_tensor_id == 0u)) {
        return -1;
    }

    memset(&cmd, 0, sizeof(cmd));
    cmd.command_type = (uint8_t)ATT1_AIMU_CMD_LOAD_TENSOR_TILE;
    cmd.dtype = dtype;
    cmd.tensor_id = data->next_auto_tensor_id;
    cmd.input_buf_addr = (uint64_t)(uintptr_t)ptr;
    cmd.input_buf_bytes = 1u; /* informational only; not read by the exec hook */
    cmd.op_param_1 = ((uint32_t)dim0 << 16) | (uint32_t)dim1;

    if (pcie_submit_exec(backend, &cmd) != 0) {
        return -1;
    }

    data->auto_tensors[data->auto_tensor_count].ptr = ptr;
    data->auto_tensors[data->auto_tensor_count].dtype = dtype;
    data->auto_tensors[data->auto_tensor_count].tensor_id = data->next_auto_tensor_id;
    *out_tensor_id = data->next_auto_tensor_id;
    data->auto_tensor_count++;
    data->next_auto_tensor_id--;
    return 0;
}

static void *pcie_alloc(att1_backend *backend, size_t bytes)
{
    (void)backend;
    if (bytes == 0u) {
        return NULL;
    }
    return malloc(bytes);
}

static void pcie_free(att1_backend *backend, void *ptr)
{
    (void)backend;
    free(ptr);
}

static int pcie_sync(att1_backend *backend)
{
    att1_pcie_backend_data *data = pcie_data(backend);

    if (data == NULL) {
        return -1;
    }
    return (att1_aimu_conformance_sync_mmio(data->endpoint) == ATT1_OK) ? 0 : -1;
}

static int pcie_matmul_f32(att1_backend *backend,
                           float *dst,
                           const float *lhs,
                           const float *rhs,
                           size_t rows,
                           size_t cols,
                           size_t inner)
{
    att1_aimu_cmd cmd;
    uint16_t tensor_id;

    if (rows != 1u) {
        /* This codebase only ever calls matmul_f32 with a single activation
         * row (one token at a time); EXEC_MATMUL's frozen packet models a
         * single activation vector, matching that usage exactly. */
        return -1;
    }
    if (pcie_register_weight(backend, rhs, ATT1_AIMU_DMA_DTYPE_F32,
                             (uint16_t)inner, (uint16_t)cols, &tensor_id) != 0) {
        return -1;
    }

    memset(&cmd, 0, sizeof(cmd));
    cmd.command_type = (uint8_t)ATT1_AIMU_CMD_EXEC_MATMUL;
    cmd.dtype = ATT1_AIMU_DMA_DTYPE_F32;
    cmd.tensor_id = tensor_id;
    cmd.input_buf_addr = (uint64_t)(uintptr_t)lhs;
    cmd.input_buf_bytes = (uint32_t)(inner * sizeof(float));
    cmd.output_buf_addr = (uint64_t)(uintptr_t)dst;
    cmd.output_buf_bytes = (uint32_t)(cols * sizeof(float));
    cmd.op_param_0 = (uint32_t)cols;

    return pcie_submit_exec(backend, &cmd);
}

static int pcie_matmul_q8xf32(att1_backend *backend,
                              float *dst,
                              const float *lhs,
                              size_t lhs_rows,
                              size_t lhs_cols,
                              const att1_q8_matrix *weights)
{
    att1_aimu_cmd cmd;
    uint16_t tensor_id;

    if ((lhs_rows != 1u) || (weights == NULL)) {
        return -1;
    }
    if (pcie_register_weight(backend, weights, ATT1_AIMU_DMA_DTYPE_Q8,
                             (uint16_t)weights->rows, (uint16_t)weights->cols,
                             &tensor_id) != 0) {
        return -1;
    }

    memset(&cmd, 0, sizeof(cmd));
    cmd.command_type = (uint8_t)ATT1_AIMU_CMD_EXEC_MATMUL;
    cmd.dtype = ATT1_AIMU_DMA_DTYPE_Q8;
    cmd.tensor_id = tensor_id;
    cmd.input_buf_addr = (uint64_t)(uintptr_t)lhs;
    cmd.input_buf_bytes = (uint32_t)(lhs_cols * sizeof(float));
    cmd.output_buf_addr = (uint64_t)(uintptr_t)dst;
    cmd.output_buf_bytes = (uint32_t)(weights->rows * sizeof(float));
    cmd.op_param_0 = (uint32_t)weights->rows;

    return pcie_submit_exec(backend, &cmd);
}

static int pcie_matmul_q4xf32(att1_backend *backend,
                              float *dst,
                              const float *lhs,
                              size_t lhs_rows,
                              size_t lhs_cols,
                              const att1_q4_matrix *weights)
{
    att1_aimu_cmd cmd;
    uint16_t tensor_id;

    if ((lhs_rows != 1u) || (weights == NULL)) {
        return -1;
    }
    if (pcie_register_weight(backend, weights, ATT1_AIMU_DMA_DTYPE_Q4,
                             (uint16_t)weights->rows, (uint16_t)weights->cols,
                             &tensor_id) != 0) {
        return -1;
    }

    memset(&cmd, 0, sizeof(cmd));
    cmd.command_type = (uint8_t)ATT1_AIMU_CMD_EXEC_MATMUL;
    cmd.dtype = ATT1_AIMU_DMA_DTYPE_Q4;
    cmd.tensor_id = tensor_id;
    cmd.input_buf_addr = (uint64_t)(uintptr_t)lhs;
    cmd.input_buf_bytes = (uint32_t)(lhs_cols * sizeof(float));
    cmd.output_buf_addr = (uint64_t)(uintptr_t)dst;
    cmd.output_buf_bytes = (uint32_t)(weights->rows * sizeof(float));
    cmd.op_param_0 = (uint32_t)weights->rows;

    return pcie_submit_exec(backend, &cmd);
}

static int pcie_rmsnorm_f32(att1_backend *backend,
                            float *dst,
                            const float *src,
                            const float *weight,
                            size_t count,
                            float epsilon)
{
    att1_aimu_cmd cmd;
    uint16_t tensor_id;
    union { uint32_t u; float f; } eps_bits;

    if (pcie_register_weight(backend, weight, ATT1_AIMU_DMA_DTYPE_F32,
                             1u, (uint16_t)count, &tensor_id) != 0) {
        return -1;
    }

    memset(&cmd, 0, sizeof(cmd));
    cmd.command_type = (uint8_t)ATT1_AIMU_CMD_EXEC_RMSNORM;
    cmd.dtype = ATT1_AIMU_DMA_DTYPE_F32;
    cmd.tensor_id = tensor_id;
    cmd.input_buf_addr = (uint64_t)(uintptr_t)src;
    cmd.input_buf_bytes = (uint32_t)(count * sizeof(float));
    cmd.output_buf_addr = (uint64_t)(uintptr_t)dst;
    cmd.output_buf_bytes = (uint32_t)(count * sizeof(float));
    eps_bits.f = epsilon;
    cmd.op_param_0 = eps_bits.u;

    return pcie_submit_exec(backend, &cmd);
}

static int pcie_softmax_f32(att1_backend *backend,
                            float *values,
                            size_t count)
{
    /* No frozen command type covers plain softmax (M103 §4.6: only the
     * larger fused EXEC_ATTENTION does), so it is computed locally rather
     * than round-tripped through the endpoint. */
    (void)backend;
    return att1_softmax_f32(values, count);
}

static int pcie_rope_f32(att1_backend *backend,
                         float *values,
                         size_t count,
                         size_t position,
                         float theta)
{
    att1_aimu_cmd cmd;
    union { uint32_t u; float f; } theta_bits;

    memset(&cmd, 0, sizeof(cmd));
    cmd.command_type = (uint8_t)ATT1_AIMU_CMD_EXEC_ROPE;
    cmd.dtype = ATT1_AIMU_DMA_DTYPE_F32;
    cmd.input_buf_addr = (uint64_t)(uintptr_t)values;
    cmd.input_buf_bytes = (uint32_t)(count * sizeof(float));
    cmd.output_buf_addr = (uint64_t)(uintptr_t)values;
    cmd.output_buf_bytes = (uint32_t)(count * sizeof(float));
    cmd.kv_position = (uint32_t)position;
    theta_bits.f = theta;
    cmd.op_param_1 = theta_bits.u;

    return pcie_submit_exec(backend, &cmd);
}

static int pcie_ffn_swiglu_f32(att1_backend *backend,
                               float *dst,
                               const float *gate,
                               const float *value,
                               size_t count)
{
    att1_aimu_cmd cmd;
    float *packed;
    int rc;

    if (count == 0u) {
        return -1;
    }

    /* EXEC_FFN's frozen packet has only one input address field, but this
     * op needs two input buffers (gate, value); pack them contiguously
     * into one scratch buffer for the round trip. */
    packed = (float *)malloc(count * 2u * sizeof(float));
    if (packed == NULL) {
        return -1;
    }
    memcpy(packed, gate, count * sizeof(float));
    memcpy(packed + count, value, count * sizeof(float));

    memset(&cmd, 0, sizeof(cmd));
    cmd.command_type = (uint8_t)ATT1_AIMU_CMD_EXEC_FFN;
    cmd.dtype = ATT1_AIMU_DMA_DTYPE_F32;
    cmd.input_buf_addr = (uint64_t)(uintptr_t)packed;
    cmd.input_buf_bytes = (uint32_t)(count * 2u * sizeof(float));
    cmd.output_buf_addr = (uint64_t)(uintptr_t)dst;
    cmd.output_buf_bytes = (uint32_t)(count * sizeof(float));
    cmd.op_param_0 = (uint32_t)count;

    rc = pcie_submit_exec(backend, &cmd);
    free(packed);
    return rc;
}

static const att1_backend_ops pcie_ops = {
    "pcie",
    pcie_alloc,
    pcie_free,
    pcie_sync,
    pcie_matmul_f32,
    pcie_matmul_q8xf32,
    pcie_matmul_q4xf32,
    pcie_rmsnorm_f32,
    pcie_softmax_f32,
    pcie_rope_f32,
    pcie_ffn_swiglu_f32
};

/* M164: resident-tensor table lookup. Returns NULL if not found. */
static att1_pcie_resident_tensor *resident_find(att1_pcie_backend_data *data,
                                                uint32_t tensor_id)
{
    size_t i;

    for (i = 0u; i < data->resident_count; i++) {
        if (data->resident[i].tensor_id == tensor_id) {
            return &data->resident[i];
        }
    }
    return NULL;
}

att1_status_t att1_backend_pcie_load_tensor(att1_backend *backend,
                                            uint32_t tensor_id,
                                            uint64_t host_addr,
                                            uint64_t device_addr,
                                            uint64_t byte_length,
                                            uint8_t dtype,
                                            uint8_t quant_group_size)
{
    att1_pcie_backend_data *data = pcie_data(backend);
    uint64_t offset;

    if ((data == NULL) || (byte_length == 0u)) {
        return ATT1_ERR_INVALID_ARG;
    }

    if (resident_find(data, tensor_id) != NULL) {
        data->residency_counters.duplicate_transfer_rejections++;
        return ATT1_ERR_STATE;
    }

    if (data->resident_count >= ATT1_BACKEND_PCIE_MAX_RESIDENT_TENSORS) {
        return ATT1_ERR_STATE;
    }

    offset = 0u;
    while (offset < byte_length) {
        uint64_t remaining = byte_length - offset;
        uint64_t chunk = (remaining > (uint64_t)ATT1_AIMU_DMA_MAX_TRANSFER_BYTES)
                                 ? (uint64_t)ATT1_AIMU_DMA_MAX_TRANSFER_BYTES
                                 : remaining;
        att1_aimu_dma_desc desc;
        att1_status_t st;

        memset(&desc, 0, sizeof(desc));
        desc.host_addr = host_addr + offset;
        desc.device_addr = device_addr + offset;
        desc.byte_length = (uint32_t)chunk;
        desc.descriptor_id = data->next_descriptor_id++;
        desc.tensor_id = tensor_id;
        desc.dim0 = 0u;
        desc.dim1 = 0u;
        desc.dtype = dtype;
        desc.quant_group_size = quant_group_size;
        desc.direction = (uint8_t)ATT1_AIMU_DMA_HOST_TO_DEVICE;
        desc.flags = ((offset + chunk) >= byte_length)
                             ? ATT1_AIMU_DMA_FLAG_LAST_DESCRIPTOR
                             : 0u;

        st = att1_aimu_conformance_dma_submit(data->endpoint, &desc);
        if (st != ATT1_OK) {
            return st;
        }

        data->residency_counters.descriptors_submitted++;
        data->residency_counters.bytes_transferred += chunk;
        offset += chunk;
    }

    data->resident[data->resident_count].tensor_id = tensor_id;
    data->resident[data->resident_count].device_addr = device_addr;
    data->resident[data->resident_count].byte_length = byte_length;
    data->resident_count++;

    data->residency_counters.transfers_submitted++;
    data->residency_counters.tensors_resident = (uint64_t)data->resident_count;

    return ATT1_OK;
}

int att1_backend_pcie_tensor_is_resident(att1_backend *backend, uint32_t tensor_id)
{
    att1_pcie_backend_data *data = pcie_data(backend);

    if (data == NULL) {
        return 0;
    }
    return (resident_find(data, tensor_id) != NULL) ? 1 : 0;
}

att1_status_t att1_backend_pcie_get_residency_counters(
        att1_backend *backend,
        att1_backend_pcie_residency_counters *out)
{
    att1_pcie_backend_data *data = pcie_data(backend);

    if ((data == NULL) || (out == NULL)) {
        return ATT1_ERR_INVALID_ARG;
    }
    *out = data->residency_counters;
    return ATT1_OK;
}

att1_status_t att1_backend_pcie_create(att1_aimu_conformance_endpoint *endpoint,
                                       uint32_t tile_id,
                                       att1_backend **out_backend)
{
    att1_backend *backend = NULL;
    att1_pcie_backend_data *data = NULL;

    if ((out_backend == NULL) || (endpoint == NULL)) {
        return ATT1_ERR_INVALID_ARG;
    }
    *out_backend = NULL;

    backend = calloc(1u, sizeof(*backend));
    if (backend == NULL) {
        return ATT1_ERR_OOM;
    }

    data = calloc(1u, sizeof(*data));
    if (data == NULL) {
        free(backend);
        return ATT1_ERR_OOM;
    }
    data->endpoint = endpoint;
    data->tile_id = tile_id;
    data->next_auto_tensor_id = UINT16_MAX;

    backend->ops = &pcie_ops;
    backend->user_data = data;
    *out_backend = backend;
    return ATT1_OK;
}
