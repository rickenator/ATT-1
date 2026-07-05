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

#include "att1_aimu_endpoint_client.h"
#include "att1_aimu_endpoint_protocol.h"

#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

typedef struct endpoint_client_ctx {
    int fd;
} endpoint_client_ctx;

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
    if (cmd == NULL) {
        return ATT1_ERR_INVALID_ARG;
    }
    memset(&req, 0, sizeof(req));
    req.op = ATT1_AIMU_ENDPOINT_OP_CMD_SUBMIT;
    req.cmd = *cmd;
    st = endpoint_roundtrip(cc, &req, &resp);
    if (st == ATT1_OK) {
        *cmd = resp.cmd;
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
    if (st == ATT1_OK && out != NULL) {
        *out = resp.completion;
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
        if (out_packet != NULL) {
            *out_packet = resp.fabric_packet;
        }
        if (out_payload_bytes != NULL) {
            *out_payload_bytes = resp.payload_bytes;
        }
        if (out_payload != NULL && resp.payload_bytes > 0u &&
            out_payload_capacity >= resp.payload_bytes) {
            memcpy(out_payload, resp.payload, resp.payload_bytes);
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

    if (key_bytes > half_payload || value_bytes > half_payload) {
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

    if ((key_count * sizeof(float)) > half_payload ||
        (value_count * sizeof(float)) > half_payload) {
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
        size_t out_key_bytes = resp.kv_key_bytes;
        size_t out_value_bytes = resp.kv_value_bytes;
        if (out_key_bytes > (key_count * sizeof(float))) {
            out_key_bytes = key_count * sizeof(float);
        }
        if (out_value_bytes > (value_count * sizeof(float))) {
            out_value_bytes = value_count * sizeof(float);
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

    if ((keys_count * sizeof(float)) > half_payload ||
        (values_count * sizeof(float)) > half_payload) {
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
        size_t out_key_bytes = resp.kv_key_bytes;
        size_t out_value_bytes = resp.kv_value_bytes;
        if (out_key_bytes > (keys_count * sizeof(float))) {
            out_key_bytes = keys_count * sizeof(float);
        }
        if (out_value_bytes > (values_count * sizeof(float))) {
            out_value_bytes = values_count * sizeof(float);
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
