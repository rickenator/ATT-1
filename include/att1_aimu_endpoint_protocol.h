#ifndef ATT1_AIMU_ENDPOINT_PROTOCOL_H
#define ATT1_AIMU_ENDPOINT_PROTOCOL_H

/*
 * att1_aimu_endpoint_protocol.h  —  M162 endpoint process wire protocol.
 *
 * Fixed-size, same-architecture request/response messages exchanged between
 * an `att1-aimu-endpoint` daemon (owning tile memory, register file, and
 * command queue in a separate process) and an in-process client that
 * implements the `att1_aimu_conformance_ops` interface (M161) over the
 * transport. This is a userspace simulator protocol only: no real PCIe
 * transaction layer, no cross-architecture byte-order handling, and no
 * kernel driver involvement (M93 §8.8 explicitly permits this substitution
 * for the frozen register/queue semantics).
 *
 * The struct layouts below intentionally embed the frozen M161 conformance
 * ABI types directly (att1_aimu_cmd, att1_aimu_dma_desc, att1_fabric_packet,
 * counters, trace snapshot) so the daemon and client always agree on the
 * same field semantics as the in-process simulator.
 */

#include "att1_aimu_cmdq.h"
#include "att1_aimu_dma.h"
#include "att1_aimu_trace.h"
#include "att1_fabric.h"
#include "att1_kv_mmu.h"
#include "att1_status.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ATT1_AIMU_ENDPOINT_MAX_PAYLOAD 16384u
#define ATT1_AIMU_ENDPOINT_MAX_GROUP_TILES 16u

typedef enum att1_aimu_endpoint_op {
    ATT1_AIMU_ENDPOINT_OP_SYNC_MMIO = 1,
    ATT1_AIMU_ENDPOINT_OP_SNAPSHOT_COUNTERS,
    ATT1_AIMU_ENDPOINT_OP_MMIO_READ32,
    ATT1_AIMU_ENDPOINT_OP_MMIO_WRITE32,
    ATT1_AIMU_ENDPOINT_OP_MMIO_READ64,
    ATT1_AIMU_ENDPOINT_OP_MMIO_WRITE64,
    ATT1_AIMU_ENDPOINT_OP_CMD_SUBMIT,
    ATT1_AIMU_ENDPOINT_OP_CMD_DISPATCH_ONE,
    ATT1_AIMU_ENDPOINT_OP_CMD_DISPATCH_ALL,
    ATT1_AIMU_ENDPOINT_OP_CMD_POLL_COMPLETION,
    ATT1_AIMU_ENDPOINT_OP_CMD_GET_COUNTERS,
    ATT1_AIMU_ENDPOINT_OP_DMA_VALIDATE,
    ATT1_AIMU_ENDPOINT_OP_DMA_SUBMIT,
    ATT1_AIMU_ENDPOINT_OP_DMA_GET_COUNTERS,
    ATT1_AIMU_ENDPOINT_OP_FABRIC_SEND,
    ATT1_AIMU_ENDPOINT_OP_FABRIC_BROADCAST,
    ATT1_AIMU_ENDPOINT_OP_FABRIC_RECEIVE,
    ATT1_AIMU_ENDPOINT_OP_FABRIC_BARRIER_ARRIVE,
    ATT1_AIMU_ENDPOINT_OP_FABRIC_GET_COUNTERS,
    ATT1_AIMU_ENDPOINT_OP_TRACE_GET_SNAPSHOT,
    ATT1_AIMU_ENDPOINT_OP_KV_CREATE_SESSION,
    ATT1_AIMU_ENDPOINT_OP_KV_DESTROY_SESSION,
    ATT1_AIMU_ENDPOINT_OP_KV_APPEND,
    ATT1_AIMU_ENDPOINT_OP_KV_READ,
    ATT1_AIMU_ENDPOINT_OP_KV_COPY_RANGE,
    ATT1_AIMU_ENDPOINT_OP_KV_GET_COUNTERS,
    ATT1_AIMU_ENDPOINT_OP_SHUTDOWN
} att1_aimu_endpoint_op;

typedef struct att1_aimu_endpoint_request {
    uint32_t op;                 /* att1_aimu_endpoint_op */
    uint32_t offset;             /* MMIO offset / barrier tile_id */
    uint32_t value32;            /* MMIO write32 value */
    uint64_t value64;            /* MMIO write64 value / fabric tag */
    att1_aimu_cmd cmd;           /* CMD_SUBMIT payload */
    att1_aimu_dma_desc dma_desc; /* DMA_VALIDATE / DMA_SUBMIT payload */
    att1_packet_type packet_type;
    uint32_t source_tile;
    uint32_t target_tile;
    uint32_t tile_id;
    uint32_t group_count;
    uint32_t group_tiles[ATT1_AIMU_ENDPOINT_MAX_GROUP_TILES];
    uint32_t participant_count;
    uint32_t participants[ATT1_AIMU_ENDPOINT_MAX_GROUP_TILES];
    uint32_t payload_bytes;
    uint32_t payload_capacity;
    unsigned char payload[ATT1_AIMU_ENDPOINT_MAX_PAYLOAD];
    uint64_t kv_session_id;
    uint32_t kv_layer_id;
    uint32_t kv_head_id;
    uint32_t kv_position;
    uint32_t kv_start_position;
    uint32_t kv_position_count;
    uint32_t kv_key_bytes;
    uint32_t kv_value_bytes;
} att1_aimu_endpoint_request;

typedef struct att1_aimu_endpoint_response {
    int32_t status;   /* att1_status_t */
    uint32_t value32;
    uint64_t value64;
    int32_t out_complete;
    att1_aimu_cmd cmd;    /* CMD_SUBMIT: mutated command (id/status/checksum) */
    att1_aimu_completion completion;
    att1_aimu_cmdq_counters cmd_counters;
    att1_aimu_dma_counters dma_counters;
    att1_fabric_counters fabric_counters;
    att1_fabric_packet fabric_packet;
    att1_aimu_trace_snapshot trace_snapshot;
    att1_kv_mmu_counters kv_counters;
    uint32_t kv_key_bytes;
    uint32_t kv_value_bytes;
    uint32_t payload_bytes;
    unsigned char payload[ATT1_AIMU_ENDPOINT_MAX_PAYLOAD];
} att1_aimu_endpoint_response;

/*
 * Blocking, restart-on-EINTR helpers for exchanging fixed-size messages over
 * a connected stream socket (Unix domain socket for the M162 skeleton).
 * Return ATT1_OK on a full transfer, ATT1_ERR_IO on short read/write or
 * connection loss.
 */
att1_status_t att1_aimu_endpoint_send_request(int fd, const att1_aimu_endpoint_request *req);
att1_status_t att1_aimu_endpoint_recv_request(int fd, att1_aimu_endpoint_request *req);
att1_status_t att1_aimu_endpoint_send_response(int fd, const att1_aimu_endpoint_response *resp);
att1_status_t att1_aimu_endpoint_recv_response(int fd, att1_aimu_endpoint_response *resp);

#ifdef __cplusplus
}
#endif

#endif
