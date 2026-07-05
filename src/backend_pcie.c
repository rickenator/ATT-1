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

#include <stdlib.h>
#include <string.h>

typedef struct att1_pcie_backend_data {
    att1_aimu_conformance_endpoint *endpoint; /* not owned */
    uint32_t                        tile_id;
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
