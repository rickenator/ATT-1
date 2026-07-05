/*
 * backend_pcie.c  —  M163 `backend_pcie.c` host backend.
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
 * att1_aimu_result_to_status(). Today the M161/M162 command-queue
 * simulator returns ATT1_AIMU_ERR_UNSUPPORTED_OP for every EXEC_* command
 * by design (see src/aimu_cmdq.c), so these calls currently fail with
 * ATT1_ERR_UNSUPPORTED once mapped — that is expected until a later
 * milestone teaches the endpoint to actually execute EXEC_* commands
 * (M166 "single-tile emulated decode"). This milestone proves the
 * transport plumbing (submit -> dispatch -> poll -> result mapping), not
 * compute correctness.
 */

#include "att1_backend.h"

#include "att1_aimu_cmdq.h"
#include "att1_aimu_conformance.h"
#include "att1_aimu_dma.h"

#include <stdlib.h>
#include <string.h>

/* M164: fixed-capacity resident-tensor table. A fixed array (rather than a
 * separate heap allocation) is used because att1_backend_destroy() is a
 * generic helper shared by every backend that only frees backend->user_data
 * and backend itself; embedding the table avoids a second allocation that
 * would otherwise need backend-specific cleanup. 4096 entries comfortably
 * covers every weight tensor of any tiny/small fixture model used today. */
#define ATT1_BACKEND_PCIE_MAX_RESIDENT_TENSORS 4096u

typedef struct att1_pcie_resident_tensor {
    uint32_t tensor_id;
    uint64_t device_addr;
    uint64_t byte_length;
} att1_pcie_resident_tensor;

typedef struct att1_pcie_backend_data {
    att1_aimu_conformance_endpoint *endpoint; /* not owned */
    uint32_t                        tile_id;

    /* M164: one-time shard transfer / tensor residency tracking. */
    att1_pcie_resident_tensor resident[ATT1_BACKEND_PCIE_MAX_RESIDENT_TENSORS];
    size_t                    resident_count;
    uint32_t                  next_descriptor_id;
    att1_backend_pcie_residency_counters residency_counters;
} att1_pcie_backend_data;

static att1_pcie_backend_data *pcie_data(att1_backend *backend)
{
    if ((backend == NULL) || (backend->user_data == NULL)) {
        return NULL;
    }
    return (att1_pcie_backend_data *)backend->user_data;
}

/*
 * pcie_exec_command
 *
 * Submits a single command of `type` targeting the backend's tile,
 * dispatches it, polls its completion, and maps the completion's result
 * code to att1_status_t. Returns 0 for ATT1_OK, -1 otherwise.
 */
static int pcie_exec_command(att1_backend *backend, att1_aimu_cmd_type type)
{
    att1_pcie_backend_data *data = pcie_data(backend);
    att1_aimu_cmd cmd;
    att1_aimu_completion completion;

    if (data == NULL) {
        return -1;
    }

    memset(&cmd, 0, sizeof(cmd));
    cmd.command_type = (uint8_t)type;
    cmd.tile_id = (uint8_t)data->tile_id;

    if (att1_aimu_conformance_cmd_submit(data->endpoint, &cmd) != ATT1_OK) {
        return -1;
    }
    if (att1_aimu_conformance_cmd_dispatch_one(data->endpoint) != ATT1_OK) {
        return -1;
    }
    memset(&completion, 0, sizeof(completion));
    if (att1_aimu_conformance_cmd_poll_completion(data->endpoint, &completion) != ATT1_OK) {
        return -1;
    }

    if (att1_aimu_result_to_status((att1_aimu_result)completion.result_code) != ATT1_OK) {
        return -1;
    }
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
    (void)dst;
    (void)lhs;
    (void)rhs;
    (void)rows;
    (void)cols;
    (void)inner;
    return pcie_exec_command(backend, ATT1_AIMU_CMD_EXEC_MATMUL);
}

static int pcie_matmul_q8xf32(att1_backend *backend,
                              float *dst,
                              const float *lhs,
                              size_t lhs_rows,
                              size_t lhs_cols,
                              const att1_q8_matrix *weights)
{
    (void)dst;
    (void)lhs;
    (void)lhs_rows;
    (void)lhs_cols;
    (void)weights;
    return pcie_exec_command(backend, ATT1_AIMU_CMD_EXEC_MATMUL);
}

static int pcie_matmul_q4xf32(att1_backend *backend,
                              float *dst,
                              const float *lhs,
                              size_t lhs_rows,
                              size_t lhs_cols,
                              const att1_q4_matrix *weights)
{
    (void)dst;
    (void)lhs;
    (void)lhs_rows;
    (void)lhs_cols;
    (void)weights;
    return pcie_exec_command(backend, ATT1_AIMU_CMD_EXEC_MATMUL);
}

static int pcie_rmsnorm_f32(att1_backend *backend,
                            float *dst,
                            const float *src,
                            const float *weight,
                            size_t count,
                            float epsilon)
{
    (void)dst;
    (void)src;
    (void)weight;
    (void)count;
    (void)epsilon;
    return pcie_exec_command(backend, ATT1_AIMU_CMD_EXEC_RMSNORM);
}

static int pcie_rope_f32(att1_backend *backend,
                         float *values,
                         size_t count,
                         size_t position,
                         float theta)
{
    (void)values;
    (void)count;
    (void)position;
    (void)theta;
    return pcie_exec_command(backend, ATT1_AIMU_CMD_EXEC_ROPE);
}

static int pcie_ffn_swiglu_f32(att1_backend *backend,
                               float *dst,
                               const float *gate,
                               const float *value,
                               size_t count)
{
    (void)dst;
    (void)gate;
    (void)value;
    (void)count;
    return pcie_exec_command(backend, ATT1_AIMU_CMD_EXEC_FFN);
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
    NULL, /* softmax_f32: no dedicated frozen command type yet */
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

    backend->ops = &pcie_ops;
    backend->user_data = data;
    *out_backend = backend;
    return ATT1_OK;
}
