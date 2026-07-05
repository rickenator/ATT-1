/*
 * aimu_endpoint_client.c  —  M162 socket-backed conformance client.
 *
 * Implements att1_aimu_conformance_ops (M161) over a connected Unix domain
 * socket to an `att1-aimu-endpoint` daemon process. Every call is a single
 * blocking request/response round trip using the fixed-size messages in
 * att1_aimu_endpoint_protocol.h; the daemon owns the real tile memory,
 * register file, command queue, DMA simulator, and fabric bus.
 */

#define _POSIX_C_SOURCE 200112L

#include "att1_aimu_cmdq.h"
#include "att1_aimu_dma.h"
#include "att1_aimu_endpoint_client.h"
#include "att1_aimu_endpoint_protocol.h"

#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

/*
 * Cross-process EXEC_* tensor-math execution (real fix for the M167 known
 * limitation, see the matching comment in tools/att1-aimu-endpoint.c):
 * `client_cmd_submit()` below copies the real operand bytes named by
 * `cmd->input_buf_addr` into the request's `payload` field (instead of
 * forwarding the raw, client-process-only pointer value) for the command
 * types whose exec hook dereferences those addresses (LOAD_TENSOR_TILE
 * f32, EXEC_MATMUL, EXEC_RMSNORM, EXEC_ROPE, EXEC_FFN); the daemon copies
 * that payload into memory it owns and rewrites the command before
 * executing it (see dispatch_cmd_submit()). Because the M158 frozen
 * command packet's `output_buf_addr` is likewise only valid in the client
 * process, and dispatch (real execution) happens asynchronously relative
 * to submit, this client remembers where each in-flight command's result
 * belongs (keyed by the `command_id` the daemon assigns during submit) and
 * copies the daemon's result payload — attached to the matching
 * `CMD_POLL_COMPLETION` response — back into that local buffer once the
 * command actually completes.
 */
#define ATT1_AIMU_ENDPOINT_CLIENT_MAX_PENDING_XFERS 8u

typedef struct endpoint_client_pending_xfer {
    int      used;
    uint32_t command_id;
    void    *output_ptr;   /* client-local destination buffer */
    uint32_t output_bytes;
} endpoint_client_pending_xfer;

typedef struct endpoint_client_ctx {
    int fd;
    endpoint_client_pending_xfer pending[ATT1_AIMU_ENDPOINT_CLIENT_MAX_PENDING_XFERS];
} endpoint_client_ctx;

static int client_cmd_needs_buffer_xfer(uint8_t command_type)
{
    switch ((att1_aimu_cmd_type)command_type) {
    case ATT1_AIMU_CMD_EXEC_MATMUL:
    case ATT1_AIMU_CMD_EXEC_RMSNORM:
    case ATT1_AIMU_CMD_EXEC_ROPE:
    case ATT1_AIMU_CMD_EXEC_FFN:
        return 1;
    default:
        return 0;
    }
}

static void pending_xfer_set(endpoint_client_pending_xfer *entry,
                             uint32_t command_id,
                             void *output_ptr,
                             uint32_t output_bytes)
{
    entry->used = 1;
    entry->command_id = command_id;
    entry->output_ptr = output_ptr;
    entry->output_bytes = output_bytes;
}

static void pending_xfer_register(endpoint_client_ctx *cc,
                                  uint32_t command_id,
                                  void *output_ptr,
                                  uint32_t output_bytes)
{
    size_t i;
    for (i = 0u; i < ATT1_AIMU_ENDPOINT_CLIENT_MAX_PENDING_XFERS; i++) {
        if (!cc->pending[i].used) {
            pending_xfer_set(&cc->pending[i], command_id, output_ptr, output_bytes);
            return;
        }
    }
    /* Table full (shouldn't happen: the only current caller,
     * src/backend_pcie.c, keeps exactly one command in flight at a time);
     * reclaim the oldest slot rather than silently dropping the result. */
    pending_xfer_set(&cc->pending[0], command_id, output_ptr, output_bytes);
}

static att1_status_t endpoint_roundtrip(endpoint_client_ctx *cc,
                                        att1_aimu_endpoint_request *req,
                                        att1_aimu_endpoint_response *resp)
{
    att1_status_t st = att1_aimu_endpoint_send_request(cc->fd, req);
    if (st != ATT1_OK) {
        return ATT1_ERR_IO;
    }
    st = att1_aimu_endpoint_recv_response(cc->fd, resp);
    if (st != ATT1_OK) {
        return ATT1_ERR_IO;
    }
    return (att1_status_t)resp->status;
}

static void client_destroy(void *ctx)
{
    endpoint_client_ctx *cc = (endpoint_client_ctx *)ctx;
    if (cc == NULL) {
        return;
    }
    if (cc->fd >= 0) {
        att1_aimu_endpoint_request req;
        att1_aimu_endpoint_response resp;
        memset(&req, 0, sizeof(req));
        req.op = ATT1_AIMU_ENDPOINT_OP_SHUTDOWN;
        /* Best-effort notice; ignore failures during teardown. */
        (void)att1_aimu_endpoint_send_request(cc->fd, &req);
        (void)att1_aimu_endpoint_recv_response(cc->fd, &resp);
        close(cc->fd);
    }
    free(cc);
}

static att1_status_t client_sync_mmio(void *ctx)
{
    endpoint_client_ctx *cc = (endpoint_client_ctx *)ctx;
    att1_aimu_endpoint_request req;
    att1_aimu_endpoint_response resp;
    memset(&req, 0, sizeof(req));
    req.op = ATT1_AIMU_ENDPOINT_OP_SYNC_MMIO;
    return endpoint_roundtrip(cc, &req, &resp);
}

static att1_status_t client_snapshot_counters(void *ctx)
{
    endpoint_client_ctx *cc = (endpoint_client_ctx *)ctx;
    att1_aimu_endpoint_request req;
    att1_aimu_endpoint_response resp;
    memset(&req, 0, sizeof(req));
    req.op = ATT1_AIMU_ENDPOINT_OP_SNAPSHOT_COUNTERS;
    return endpoint_roundtrip(cc, &req, &resp);
}

static att1_status_t client_mmio_read32(void *ctx, uint32_t offset, uint32_t *out)
{
    endpoint_client_ctx *cc = (endpoint_client_ctx *)ctx;
    att1_aimu_endpoint_request req;
    att1_aimu_endpoint_response resp;
    att1_status_t st;
    memset(&req, 0, sizeof(req));
    req.op = ATT1_AIMU_ENDPOINT_OP_MMIO_READ32;
    req.offset = offset;
    st = endpoint_roundtrip(cc, &req, &resp);
    if (st == ATT1_OK && out != NULL) {
        *out = resp.value32;
    }
    return st;
}

static att1_status_t client_mmio_write32(void *ctx, uint32_t offset, uint32_t value)
{
    endpoint_client_ctx *cc = (endpoint_client_ctx *)ctx;
    att1_aimu_endpoint_request req;
    att1_aimu_endpoint_response resp;
    memset(&req, 0, sizeof(req));
    req.op = ATT1_AIMU_ENDPOINT_OP_MMIO_WRITE32;
    req.offset = offset;
    req.value32 = value;
    return endpoint_roundtrip(cc, &req, &resp);
}

static att1_status_t client_mmio_read64(void *ctx, uint32_t offset, uint64_t *out)
{
    endpoint_client_ctx *cc = (endpoint_client_ctx *)ctx;
    att1_aimu_endpoint_request req;
    att1_aimu_endpoint_response resp;
    att1_status_t st;
    memset(&req, 0, sizeof(req));
    req.op = ATT1_AIMU_ENDPOINT_OP_MMIO_READ64;
    req.offset = offset;
    st = endpoint_roundtrip(cc, &req, &resp);
    if (st == ATT1_OK && out != NULL) {
        *out = resp.value64;
    }
    return st;
}

static att1_status_t client_mmio_write64(void *ctx, uint32_t offset, uint64_t value)
{
    endpoint_client_ctx *cc = (endpoint_client_ctx *)ctx;
    att1_aimu_endpoint_request req;
    att1_aimu_endpoint_response resp;
    memset(&req, 0, sizeof(req));
    req.op = ATT1_AIMU_ENDPOINT_OP_MMIO_WRITE64;
    req.offset = offset;
    req.value64 = value;
    return endpoint_roundtrip(cc, &req, &resp);
}

static att1_status_t client_cmd_submit(void *ctx, att1_aimu_cmd *cmd)
{
    endpoint_client_ctx *cc = (endpoint_client_ctx *)ctx;
    att1_aimu_endpoint_request req;
    att1_aimu_endpoint_response resp;
    att1_status_t st;
    void *output_ptr = NULL;
    uint32_t output_bytes = 0u;
    if (cmd == NULL) {
        return ATT1_ERR_INVALID_ARG;
    }
    memset(&req, 0, sizeof(req));
    req.op = ATT1_AIMU_ENDPOINT_OP_CMD_SUBMIT;
    req.cmd = *cmd;

    /* See the file header comment: forward the real operand bytes instead
     * of the (client-process-only) raw pointer for command types whose
     * exec hook dereferences them. */
    if ((att1_aimu_cmd_type)cmd->command_type == ATT1_AIMU_CMD_LOAD_TENSOR_TILE) {
        uint32_t dim0 = cmd->op_param_1 >> 16;
        uint32_t dim1 = cmd->op_param_1 & 0xFFFFu;
        uint32_t bytes;

        if (cmd->dtype != ATT1_AIMU_DMA_DTYPE_F32) {
            /* q8/q4 weight tensors are referenced by a struct pointer with
             * its own nested owned buffers (values/scales or
             * packed/scales, plus a group_size the frozen packet has
             * nowhere to carry for q4); safely reconstructing that in the
             * daemon's address space needs wire-protocol changes beyond
             * this fix, so refuse cleanly instead of forwarding a
             * cross-process struct pointer (the prior undefined
             * behavior). */
            return ATT1_ERR_UNSUPPORTED;
        }
        bytes = dim0 * dim1 * 4u;
        if ((bytes == 0u) || (cmd->input_buf_addr == 0u) ||
            (bytes > ATT1_AIMU_ENDPOINT_MAX_PAYLOAD)) {
            return ATT1_ERR_INVALID_ARG;
        }
        memcpy(req.payload, (const void *)(uintptr_t)cmd->input_buf_addr, bytes);
        req.payload_bytes = bytes;
    } else if (client_cmd_needs_buffer_xfer(cmd->command_type)) {
        uint32_t bytes = cmd->input_buf_bytes;
        if ((bytes == 0u) || (cmd->input_buf_addr == 0u) ||
            (bytes > ATT1_AIMU_ENDPOINT_MAX_PAYLOAD)) {
            return ATT1_ERR_INVALID_ARG;
        }
        memcpy(req.payload, (const void *)(uintptr_t)cmd->input_buf_addr, bytes);
        req.payload_bytes = bytes;
        if (cmd->output_buf_bytes > 0u) {
            output_ptr = (void *)(uintptr_t)cmd->output_buf_addr;
            output_bytes = cmd->output_buf_bytes;
        }
    }

    st = endpoint_roundtrip(cc, &req, &resp);
    if (st == ATT1_OK) {
        *cmd = resp.cmd;
        if (output_ptr != NULL) {
            pending_xfer_register(cc, cmd->command_id, output_ptr, output_bytes);
        }
    }
    return st;
}

static att1_status_t client_cmd_dispatch_one(void *ctx)
{
    endpoint_client_ctx *cc = (endpoint_client_ctx *)ctx;
    att1_aimu_endpoint_request req;
    att1_aimu_endpoint_response resp;
    memset(&req, 0, sizeof(req));
    req.op = ATT1_AIMU_ENDPOINT_OP_CMD_DISPATCH_ONE;
    return endpoint_roundtrip(cc, &req, &resp);
}

static att1_status_t client_cmd_dispatch_all(void *ctx)
{
    endpoint_client_ctx *cc = (endpoint_client_ctx *)ctx;
    att1_aimu_endpoint_request req;
    att1_aimu_endpoint_response resp;
    memset(&req, 0, sizeof(req));
    req.op = ATT1_AIMU_ENDPOINT_OP_CMD_DISPATCH_ALL;
    return endpoint_roundtrip(cc, &req, &resp);
}

static att1_status_t client_cmd_poll_completion(void *ctx, att1_aimu_completion *out)
{
    endpoint_client_ctx *cc = (endpoint_client_ctx *)ctx;
    att1_aimu_endpoint_request req;
    att1_aimu_endpoint_response resp;
    att1_status_t st;
    memset(&req, 0, sizeof(req));
    req.op = ATT1_AIMU_ENDPOINT_OP_CMD_POLL_COMPLETION;
    st = endpoint_roundtrip(cc, &req, &resp);
    if (st == ATT1_OK) {
        size_t i;
        if (out != NULL) {
            *out = resp.completion;
        }
        for (i = 0u; i < ATT1_AIMU_ENDPOINT_CLIENT_MAX_PENDING_XFERS; i++) {
            endpoint_client_pending_xfer *entry = &cc->pending[i];
            if (entry->used && entry->command_id == resp.completion.command_id) {
                if (resp.payload_bytes == entry->output_bytes) {
                    memcpy(entry->output_ptr, resp.payload, resp.payload_bytes);
                } else {
                    /* Protocol violation: the daemon computed output_bytes
                     * from the same cmd->output_buf_bytes this client sent
                     * at submit time, so these must match exactly; fail
                     * loudly instead of silently dropping/truncating the
                     * result. */
                    st = ATT1_ERR_IO;
                }
                memset(entry, 0, sizeof(*entry));
                break;
            }
        }
    }
    return st;
}

static att1_status_t client_cmd_get_counters(void *ctx, att1_aimu_cmdq_counters *out)
{
    endpoint_client_ctx *cc = (endpoint_client_ctx *)ctx;
    att1_aimu_endpoint_request req;
    att1_aimu_endpoint_response resp;
    att1_status_t st;
    memset(&req, 0, sizeof(req));
    req.op = ATT1_AIMU_ENDPOINT_OP_CMD_GET_COUNTERS;
    st = endpoint_roundtrip(cc, &req, &resp);
    if (st == ATT1_OK && out != NULL) {
        *out = resp.cmd_counters;
    }
    return st;
}

static att1_status_t client_dma_validate(void *ctx, const att1_aimu_dma_desc *desc)
{
    endpoint_client_ctx *cc = (endpoint_client_ctx *)ctx;
    att1_aimu_endpoint_request req;
    att1_aimu_endpoint_response resp;
    if (desc == NULL) {
        return ATT1_ERR_INVALID_ARG;
    }
    memset(&req, 0, sizeof(req));
    req.op = ATT1_AIMU_ENDPOINT_OP_DMA_VALIDATE;
    req.dma_desc = *desc;
    return endpoint_roundtrip(cc, &req, &resp);
}

static att1_status_t client_dma_submit(void *ctx, const att1_aimu_dma_desc *desc)
{
    endpoint_client_ctx *cc = (endpoint_client_ctx *)ctx;
    att1_aimu_endpoint_request req;
    att1_aimu_endpoint_response resp;
    if (desc == NULL) {
        return ATT1_ERR_INVALID_ARG;
    }
    memset(&req, 0, sizeof(req));
    req.op = ATT1_AIMU_ENDPOINT_OP_DMA_SUBMIT;
    req.dma_desc = *desc;
    return endpoint_roundtrip(cc, &req, &resp);
}

static att1_status_t client_dma_get_counters(void *ctx, att1_aimu_dma_counters *out)
{
    endpoint_client_ctx *cc = (endpoint_client_ctx *)ctx;
    att1_aimu_endpoint_request req;
    att1_aimu_endpoint_response resp;
    att1_status_t st;
    memset(&req, 0, sizeof(req));
    req.op = ATT1_AIMU_ENDPOINT_OP_DMA_GET_COUNTERS;
    st = endpoint_roundtrip(cc, &req, &resp);
    if (st == ATT1_OK && out != NULL) {
        *out = resp.dma_counters;
    }
    return st;
}

static att1_status_t client_fabric_send(void *ctx,
                                        uint32_t source_tile,
                                        uint32_t target_tile,
                                        att1_packet_type type,
                                        const void *payload,
                                        size_t payload_bytes,
                                        uint64_t tag)
{
    endpoint_client_ctx *cc = (endpoint_client_ctx *)ctx;
    att1_aimu_endpoint_request req;
    att1_aimu_endpoint_response resp;
    if (payload_bytes > ATT1_AIMU_ENDPOINT_MAX_PAYLOAD) {
        return ATT1_ERR_INVALID_ARG;
    }
    memset(&req, 0, sizeof(req));
    req.op = ATT1_AIMU_ENDPOINT_OP_FABRIC_SEND;
    req.source_tile = source_tile;
    req.target_tile = target_tile;
    req.packet_type = type;
    req.value64 = tag;
    req.payload_bytes = (uint32_t)payload_bytes;
    if (payload_bytes > 0u && payload != NULL) {
        memcpy(req.payload, payload, payload_bytes);
    }
    return endpoint_roundtrip(cc, &req, &resp);
}

static att1_status_t client_fabric_broadcast(void *ctx,
                                             uint32_t source_tile,
                                             const uint32_t *group_tiles,
                                             size_t group_count,
                                             att1_packet_type type,
                                             const void *payload,
                                             size_t payload_bytes,
                                             uint64_t tag)
{
    endpoint_client_ctx *cc = (endpoint_client_ctx *)ctx;
    att1_aimu_endpoint_request req;
    att1_aimu_endpoint_response resp;
    size_t i;
    if (payload_bytes > ATT1_AIMU_ENDPOINT_MAX_PAYLOAD ||
        group_count > ATT1_AIMU_ENDPOINT_MAX_GROUP_TILES) {
        return ATT1_ERR_INVALID_ARG;
    }
    memset(&req, 0, sizeof(req));
    req.op = ATT1_AIMU_ENDPOINT_OP_FABRIC_BROADCAST;
    req.source_tile = source_tile;
    req.packet_type = type;
    req.value64 = tag;
    req.payload_bytes = (uint32_t)payload_bytes;
    req.group_count = (uint32_t)group_count;
    for (i = 0; i < group_count; ++i) {
        req.group_tiles[i] = group_tiles[i];
    }
    if (payload_bytes > 0u && payload != NULL) {
        memcpy(req.payload, payload, payload_bytes);
    }
    return endpoint_roundtrip(cc, &req, &resp);
}

static att1_status_t client_fabric_receive(void *ctx,
                                           uint32_t tile_id,
                                           att1_fabric_packet *out_packet,
                                           void *out_payload,
                                           size_t out_payload_capacity,
                                           size_t *out_payload_bytes)
{
    endpoint_client_ctx *cc = (endpoint_client_ctx *)ctx;
    att1_aimu_endpoint_request req;
    att1_aimu_endpoint_response resp;
    att1_status_t st;
    memset(&req, 0, sizeof(req));
    req.op = ATT1_AIMU_ENDPOINT_OP_FABRIC_RECEIVE;
    req.tile_id = tile_id;
    req.payload_capacity = (uint32_t)(out_payload_capacity > ATT1_AIMU_ENDPOINT_MAX_PAYLOAD
                                               ? ATT1_AIMU_ENDPOINT_MAX_PAYLOAD
                                               : out_payload_capacity);
    st = endpoint_roundtrip(cc, &req, &resp);
    if (st == ATT1_OK) {
        /* A malformed/hostile daemon response could claim a payload_bytes
         * value larger than the fixed resp.payload array itself
         * (ATT1_AIMU_ENDPOINT_MAX_PAYLOAD) or larger than the caller's own
         * buffer capacity; clamp to both so a hostile response can never
         * cause a read past resp.payload, and so *out_payload_bytes never
         * reports more bytes than were actually copied into out_payload
         * (M168 hostile-endpoint hardening). */
        uint32_t payload_bytes = resp.payload_bytes;
        if (payload_bytes > ATT1_AIMU_ENDPOINT_MAX_PAYLOAD) {
            payload_bytes = ATT1_AIMU_ENDPOINT_MAX_PAYLOAD;
        }
        if (payload_bytes > out_payload_capacity) {
            payload_bytes = (uint32_t)out_payload_capacity;
        }
        if (out_packet != NULL) {
            *out_packet = resp.fabric_packet;
        }
        if (out_payload_bytes != NULL) {
            *out_payload_bytes = payload_bytes;
        }
        if (out_payload != NULL && payload_bytes > 0u) {
            memcpy(out_payload, resp.payload, payload_bytes);
        }
    }
    return st;
}

static att1_status_t client_fabric_barrier_arrive(void *ctx,
                                                  uint32_t tile_id,
                                                  const uint32_t *participants,
                                                  size_t participant_count,
                                                  int *out_complete)
{
    endpoint_client_ctx *cc = (endpoint_client_ctx *)ctx;
    att1_aimu_endpoint_request req;
    att1_aimu_endpoint_response resp;
    att1_status_t st;
    size_t i;
    if (participant_count > ATT1_AIMU_ENDPOINT_MAX_GROUP_TILES) {
        return ATT1_ERR_INVALID_ARG;
    }
    memset(&req, 0, sizeof(req));
    req.op = ATT1_AIMU_ENDPOINT_OP_FABRIC_BARRIER_ARRIVE;
    req.tile_id = tile_id;
    req.participant_count = (uint32_t)participant_count;
    for (i = 0; i < participant_count; ++i) {
        req.participants[i] = participants[i];
    }
    st = endpoint_roundtrip(cc, &req, &resp);
    if (st == ATT1_OK && out_complete != NULL) {
        *out_complete = resp.out_complete;
    }
    return st;
}

static att1_status_t client_fabric_get_counters(void *ctx, att1_fabric_counters *out)
{
    endpoint_client_ctx *cc = (endpoint_client_ctx *)ctx;
    att1_aimu_endpoint_request req;
    att1_aimu_endpoint_response resp;
    att1_status_t st;
    memset(&req, 0, sizeof(req));
    req.op = ATT1_AIMU_ENDPOINT_OP_FABRIC_GET_COUNTERS;
    st = endpoint_roundtrip(cc, &req, &resp);
    if (st == ATT1_OK && out != NULL) {
        *out = resp.fabric_counters;
    }
    return st;
}

static att1_status_t client_trace_get_snapshot(void *ctx, att1_aimu_trace_snapshot *out)
{
    endpoint_client_ctx *cc = (endpoint_client_ctx *)ctx;
    att1_aimu_endpoint_request req;
    att1_aimu_endpoint_response resp;
    att1_status_t st;
    memset(&req, 0, sizeof(req));
    req.op = ATT1_AIMU_ENDPOINT_OP_TRACE_GET_SNAPSHOT;
    st = endpoint_roundtrip(cc, &req, &resp);
    if (st == ATT1_OK && out != NULL) {
        *out = resp.trace_snapshot;
    }
    return st;
}

static att1_status_t client_kv_create_session(void *ctx, uint64_t session_id)
{
    endpoint_client_ctx *cc = (endpoint_client_ctx *)ctx;
    att1_aimu_endpoint_request req;
    att1_aimu_endpoint_response resp;
    memset(&req, 0, sizeof(req));
    req.op = ATT1_AIMU_ENDPOINT_OP_KV_CREATE_SESSION;
    req.kv_session_id = session_id;
    return endpoint_roundtrip(cc, &req, &resp);
}

static att1_status_t client_kv_destroy_session(void *ctx, uint64_t session_id)
{
    endpoint_client_ctx *cc = (endpoint_client_ctx *)ctx;
    att1_aimu_endpoint_request req;
    att1_aimu_endpoint_response resp;
    memset(&req, 0, sizeof(req));
    req.op = ATT1_AIMU_ENDPOINT_OP_KV_DESTROY_SESSION;
    req.kv_session_id = session_id;
    return endpoint_roundtrip(cc, &req, &resp);
}

static att1_status_t client_kv_stream_bytes_fit(size_t bytes)
{
    if (bytes > (ATT1_AIMU_ENDPOINT_MAX_PAYLOAD / 2u)) {
        return ATT1_ERR_INVALID_ARG;
    }
    return ATT1_OK;
}

static att1_status_t client_kv_append(void *ctx,
                                      uint64_t session_id,
                                      size_t layer_id,
                                      size_t position,
                                      const float *key,
                                      size_t key_count,
                                      const float *value,
                                      size_t value_count)
{
    endpoint_client_ctx *cc = (endpoint_client_ctx *)ctx;
    att1_aimu_endpoint_request req;
    att1_aimu_endpoint_response resp;
    size_t key_bytes = key_count * sizeof(float);
    size_t value_bytes = value_count * sizeof(float);
    const size_t half_payload = ATT1_AIMU_ENDPOINT_MAX_PAYLOAD / 2u;

    if (client_kv_stream_bytes_fit(key_bytes) != ATT1_OK ||
        client_kv_stream_bytes_fit(value_bytes) != ATT1_OK) {
        return ATT1_ERR_INVALID_ARG;
    }

    memset(&req, 0, sizeof(req));
    req.op = ATT1_AIMU_ENDPOINT_OP_KV_APPEND;
    req.kv_session_id = session_id;
    req.kv_layer_id = (uint32_t)layer_id;
    req.kv_position = (uint32_t)position;
    req.kv_key_bytes = (uint32_t)key_bytes;
    req.kv_value_bytes = (uint32_t)value_bytes;
    if (key != NULL && key_bytes > 0u) {
        memcpy(req.payload, key, key_bytes);
    }
    if (value != NULL && value_bytes > 0u) {
        memcpy(req.payload + half_payload, value, value_bytes);
    }
    return endpoint_roundtrip(cc, &req, &resp);
}

static att1_status_t client_kv_read(void *ctx,
                                    uint64_t session_id,
                                    size_t layer_id,
                                    size_t head_id,
                                    size_t position,
                                    float *out_key,
                                    size_t key_count,
                                    float *out_value,
                                    size_t value_count)
{
    endpoint_client_ctx *cc = (endpoint_client_ctx *)ctx;
    att1_aimu_endpoint_request req;
    att1_aimu_endpoint_response resp;
    att1_status_t st;
    const size_t half_payload = ATT1_AIMU_ENDPOINT_MAX_PAYLOAD / 2u;

    if (client_kv_stream_bytes_fit(key_count * sizeof(float)) != ATT1_OK ||
        client_kv_stream_bytes_fit(value_count * sizeof(float)) != ATT1_OK) {
        return ATT1_ERR_INVALID_ARG;
    }

    memset(&req, 0, sizeof(req));
    req.op = ATT1_AIMU_ENDPOINT_OP_KV_READ;
    req.kv_session_id = session_id;
    req.kv_layer_id = (uint32_t)layer_id;
    req.kv_head_id = (uint32_t)head_id;
    req.kv_position = (uint32_t)position;
    req.kv_key_bytes = (uint32_t)(key_count * sizeof(float));
    req.kv_value_bytes = (uint32_t)(value_count * sizeof(float));
    st = endpoint_roundtrip(cc, &req, &resp);
    if (st == ATT1_OK) {
        /* Clamp against both the caller's own buffer capacity and the
         * wire payload's key/value half-capacity: a malformed or hostile
         * daemon response could otherwise claim kv_key_bytes/kv_value_bytes
         * larger than half_payload, which would read past the matching
         * half of resp.payload (an out-of-bounds read within this local
         * struct) once the caller's buffer is large enough not to trip the
         * caller-capacity clamp alone (M168 hostile-endpoint hardening). */
        size_t out_key_bytes = resp.kv_key_bytes;
        size_t out_value_bytes = resp.kv_value_bytes;
        if (out_key_bytes > (key_count * sizeof(float))) {
            out_key_bytes = key_count * sizeof(float);
        }
        if (out_key_bytes > half_payload) {
            out_key_bytes = half_payload;
        }
        if (out_value_bytes > (value_count * sizeof(float))) {
            out_value_bytes = value_count * sizeof(float);
        }
        if (out_value_bytes > half_payload) {
            out_value_bytes = half_payload;
        }
        if (out_key != NULL && out_key_bytes > 0u) {
            memcpy(out_key, resp.payload, out_key_bytes);
        }
        if (out_value != NULL && out_value_bytes > 0u) {
            memcpy(out_value, resp.payload + half_payload, out_value_bytes);
        }
    }
    return st;
}

static att1_status_t client_kv_copy_range(void *ctx,
                                          uint64_t session_id,
                                          size_t layer_id,
                                          size_t head_id,
                                          size_t start_position,
                                          size_t position_count,
                                          float *out_keys,
                                          size_t keys_count,
                                          float *out_values,
                                          size_t values_count)
{
    endpoint_client_ctx *cc = (endpoint_client_ctx *)ctx;
    att1_aimu_endpoint_request req;
    att1_aimu_endpoint_response resp;
    att1_status_t st;
    const size_t half_payload = ATT1_AIMU_ENDPOINT_MAX_PAYLOAD / 2u;

    if (client_kv_stream_bytes_fit(keys_count * sizeof(float)) != ATT1_OK ||
        client_kv_stream_bytes_fit(values_count * sizeof(float)) != ATT1_OK) {
        return ATT1_ERR_INVALID_ARG;
    }

    memset(&req, 0, sizeof(req));
    req.op = ATT1_AIMU_ENDPOINT_OP_KV_COPY_RANGE;
    req.kv_session_id = session_id;
    req.kv_layer_id = (uint32_t)layer_id;
    req.kv_head_id = (uint32_t)head_id;
    req.kv_start_position = (uint32_t)start_position;
    req.kv_position_count = (uint32_t)position_count;
    req.kv_key_bytes = (uint32_t)(keys_count * sizeof(float));
    req.kv_value_bytes = (uint32_t)(values_count * sizeof(float));
    st = endpoint_roundtrip(cc, &req, &resp);
    if (st == ATT1_OK) {
        /* See client_kv_read() above: clamp against half_payload too, not
         * just the caller's own buffer capacity (M168 hostile-endpoint
         * hardening). */
        size_t out_key_bytes = resp.kv_key_bytes;
        size_t out_value_bytes = resp.kv_value_bytes;
        if (out_key_bytes > (keys_count * sizeof(float))) {
            out_key_bytes = keys_count * sizeof(float);
        }
        if (out_key_bytes > half_payload) {
            out_key_bytes = half_payload;
        }
        if (out_value_bytes > (values_count * sizeof(float))) {
            out_value_bytes = values_count * sizeof(float);
        }
        if (out_value_bytes > half_payload) {
            out_value_bytes = half_payload;
        }
        if (out_keys != NULL && out_key_bytes > 0u) {
            memcpy(out_keys, resp.payload, out_key_bytes);
        }
        if (out_values != NULL && out_value_bytes > 0u) {
            memcpy(out_values, resp.payload + half_payload, out_value_bytes);
        }
    }
    return st;
}

static att1_status_t client_kv_get_counters(void *ctx, att1_kv_mmu_counters *out)
{
    endpoint_client_ctx *cc = (endpoint_client_ctx *)ctx;
    att1_aimu_endpoint_request req;
    att1_aimu_endpoint_response resp;
    att1_status_t st;
    memset(&req, 0, sizeof(req));
    req.op = ATT1_AIMU_ENDPOINT_OP_KV_GET_COUNTERS;
    st = endpoint_roundtrip(cc, &req, &resp);
    if (st == ATT1_OK && out != NULL) {
        *out = resp.kv_counters;
    }
    return st;
}

static const att1_aimu_conformance_ops g_endpoint_client_ops = {
    .name = "att1-aimu-endpoint-socket-client",
    .destroy = client_destroy,
    .sync_mmio = client_sync_mmio,
    .snapshot_counters = client_snapshot_counters,
    .mmio_read32 = client_mmio_read32,
    .mmio_write32 = client_mmio_write32,
    .mmio_read64 = client_mmio_read64,
    .mmio_write64 = client_mmio_write64,
    .cmd_submit = client_cmd_submit,
    .cmd_dispatch_one = client_cmd_dispatch_one,
    .cmd_dispatch_all = client_cmd_dispatch_all,
    .cmd_poll_completion = client_cmd_poll_completion,
    .cmd_get_counters = client_cmd_get_counters,
    .dma_validate = client_dma_validate,
    .dma_submit = client_dma_submit,
    .dma_get_counters = client_dma_get_counters,
    .fabric_send = client_fabric_send,
    .fabric_broadcast = client_fabric_broadcast,
    .fabric_receive = client_fabric_receive,
    .fabric_barrier_arrive = client_fabric_barrier_arrive,
    .fabric_get_counters = client_fabric_get_counters,
    .trace_get_snapshot = client_trace_get_snapshot,
    .kv_create_session = client_kv_create_session,
    .kv_destroy_session = client_kv_destroy_session,
    .kv_append = client_kv_append,
    .kv_read = client_kv_read,
    .kv_copy_range = client_kv_copy_range,
    .kv_get_counters = client_kv_get_counters,
};

att1_status_t att1_aimu_conformance_socket_connect(
        const char *socket_path,
        att1_aimu_conformance_endpoint **out_endpoint)
{
    struct sockaddr_un addr;
    endpoint_client_ctx *cc;
    att1_aimu_conformance_endpoint *endpoint;
    int fd;

    if (socket_path == NULL || out_endpoint == NULL) {
        return ATT1_ERR_INVALID_ARG;
    }
    *out_endpoint = NULL;

    if (strlen(socket_path) >= sizeof(addr.sun_path)) {
        return ATT1_ERR_INVALID_ARG;
    }

    fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) {
        return ATT1_ERR_IO;
    }

    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, socket_path, sizeof(addr.sun_path) - 1u);

    if (connect(fd, (struct sockaddr *)&addr, sizeof(addr)) != 0) {
        close(fd);
        return ATT1_ERR_IO;
    }

    cc = (endpoint_client_ctx *)malloc(sizeof(*cc));
    if (cc == NULL) {
        close(fd);
        return ATT1_ERR_OOM;
    }
    memset(cc, 0, sizeof(*cc));
    cc->fd = fd;

    endpoint = (att1_aimu_conformance_endpoint *)malloc(sizeof(*endpoint));
    if (endpoint == NULL) {
        close(fd);
        free(cc);
        return ATT1_ERR_OOM;
    }
    endpoint->ops = &g_endpoint_client_ops;
    endpoint->ctx = cc;

    *out_endpoint = endpoint;
    return ATT1_OK;
}
