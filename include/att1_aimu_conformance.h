#ifndef ATT1_AIMU_CONFORMANCE_H
#define ATT1_AIMU_CONFORMANCE_H

#include "att1_aimu_cmdq.h"
#include "att1_aimu_dma.h"
#include "att1_aimu_mmio.h"
#include "att1_aimu_trace.h"
#include "att1_fabric.h"
#include "att1_status.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct att1_aimu_conformance_endpoint att1_aimu_conformance_endpoint;

typedef struct att1_aimu_conformance_config {
    size_t tile_count;
    uint64_t tile_memory_bytes;
    uint64_t tile_kv_bytes;
    size_t cmd_ring_depth;
    size_t comp_ring_depth;
    size_t fabric_queue_capacity;
    size_t fabric_max_payload_bytes;
    uint64_t dma_host_base;
    uint64_t dma_host_size;
    uint64_t dma_device_base;
    uint64_t dma_device_size;
} att1_aimu_conformance_config;

typedef struct att1_aimu_conformance_ops {
    const char *name;
    void (*destroy)(void *ctx);
    att1_status_t (*sync_mmio)(void *ctx);
    att1_status_t (*snapshot_counters)(void *ctx);
    att1_status_t (*mmio_read32)(void *ctx, uint32_t offset, uint32_t *out);
    att1_status_t (*mmio_write32)(void *ctx, uint32_t offset, uint32_t value);
    att1_status_t (*mmio_read64)(void *ctx, uint32_t offset, uint64_t *out);
    att1_status_t (*mmio_write64)(void *ctx, uint32_t offset, uint64_t value);
    att1_status_t (*cmd_submit)(void *ctx, att1_aimu_cmd *cmd);
    att1_status_t (*cmd_dispatch_one)(void *ctx);
    att1_status_t (*cmd_dispatch_all)(void *ctx);
    att1_status_t (*cmd_poll_completion)(void *ctx, att1_aimu_completion *out);
    att1_status_t (*cmd_get_counters)(void *ctx, att1_aimu_cmdq_counters *out);
    att1_status_t (*dma_validate)(void *ctx, const att1_aimu_dma_desc *desc);
    att1_status_t (*dma_submit)(void *ctx, const att1_aimu_dma_desc *desc);
    att1_status_t (*dma_get_counters)(void *ctx, att1_aimu_dma_counters *out);
    att1_status_t (*fabric_send)(void *ctx,
                                 uint32_t source_tile,
                                 uint32_t target_tile,
                                 att1_packet_type type,
                                 const void *payload,
                                 size_t payload_bytes,
                                 uint64_t tag);
    att1_status_t (*fabric_broadcast)(void *ctx,
                                      uint32_t source_tile,
                                      const uint32_t *group_tiles,
                                      size_t group_count,
                                      att1_packet_type type,
                                      const void *payload,
                                      size_t payload_bytes,
                                      uint64_t tag);
    att1_status_t (*fabric_receive)(void *ctx,
                                    uint32_t tile_id,
                                    att1_fabric_packet *out_packet,
                                    void *out_payload,
                                    size_t out_payload_capacity,
                                    size_t *out_payload_bytes);
    att1_status_t (*fabric_barrier_arrive)(void *ctx,
                                           uint32_t tile_id,
                                           const uint32_t *participants,
                                           size_t participant_count,
                                           int *out_complete);
    att1_status_t (*fabric_get_counters)(void *ctx, att1_fabric_counters *out);
    att1_status_t (*trace_get_snapshot)(void *ctx, att1_aimu_trace_snapshot *out);
} att1_aimu_conformance_ops;

struct att1_aimu_conformance_endpoint {
    const att1_aimu_conformance_ops *ops;
    void *ctx;
};

void att1_aimu_conformance_default_config(att1_aimu_conformance_config *out);
att1_status_t att1_aimu_conformance_inproc_create(
        const att1_aimu_conformance_config *config,
        att1_aimu_conformance_endpoint **out_endpoint);
void att1_aimu_conformance_endpoint_destroy(att1_aimu_conformance_endpoint *endpoint);

att1_status_t att1_aimu_conformance_sync_mmio(att1_aimu_conformance_endpoint *endpoint);
att1_status_t att1_aimu_conformance_snapshot_counters(att1_aimu_conformance_endpoint *endpoint);
att1_status_t att1_aimu_conformance_mmio_read32(att1_aimu_conformance_endpoint *endpoint,
                                                uint32_t offset,
                                                uint32_t *out);
att1_status_t att1_aimu_conformance_mmio_write32(att1_aimu_conformance_endpoint *endpoint,
                                                 uint32_t offset,
                                                 uint32_t value);
att1_status_t att1_aimu_conformance_mmio_read64(att1_aimu_conformance_endpoint *endpoint,
                                                uint32_t offset,
                                                uint64_t *out);
att1_status_t att1_aimu_conformance_mmio_write64(att1_aimu_conformance_endpoint *endpoint,
                                                 uint32_t offset,
                                                 uint64_t value);
att1_status_t att1_aimu_conformance_cmd_submit(att1_aimu_conformance_endpoint *endpoint,
                                               att1_aimu_cmd *cmd);
att1_status_t att1_aimu_conformance_cmd_dispatch_one(att1_aimu_conformance_endpoint *endpoint);
att1_status_t att1_aimu_conformance_cmd_dispatch_all(att1_aimu_conformance_endpoint *endpoint);
att1_status_t att1_aimu_conformance_cmd_poll_completion(att1_aimu_conformance_endpoint *endpoint,
                                                        att1_aimu_completion *out);
att1_status_t att1_aimu_conformance_cmd_get_counters(att1_aimu_conformance_endpoint *endpoint,
                                                     att1_aimu_cmdq_counters *out);
att1_status_t att1_aimu_conformance_dma_validate(att1_aimu_conformance_endpoint *endpoint,
                                                 const att1_aimu_dma_desc *desc);
att1_status_t att1_aimu_conformance_dma_submit(att1_aimu_conformance_endpoint *endpoint,
                                               const att1_aimu_dma_desc *desc);
att1_status_t att1_aimu_conformance_dma_get_counters(att1_aimu_conformance_endpoint *endpoint,
                                                     att1_aimu_dma_counters *out);
att1_status_t att1_aimu_conformance_fabric_send(att1_aimu_conformance_endpoint *endpoint,
                                                uint32_t source_tile,
                                                uint32_t target_tile,
                                                att1_packet_type type,
                                                const void *payload,
                                                size_t payload_bytes,
                                                uint64_t tag);
att1_status_t att1_aimu_conformance_fabric_broadcast(att1_aimu_conformance_endpoint *endpoint,
                                                     uint32_t source_tile,
                                                     const uint32_t *group_tiles,
                                                     size_t group_count,
                                                     att1_packet_type type,
                                                     const void *payload,
                                                     size_t payload_bytes,
                                                     uint64_t tag);
att1_status_t att1_aimu_conformance_fabric_receive(att1_aimu_conformance_endpoint *endpoint,
                                                   uint32_t tile_id,
                                                   att1_fabric_packet *out_packet,
                                                   void *out_payload,
                                                   size_t out_payload_capacity,
                                                   size_t *out_payload_bytes);
att1_status_t att1_aimu_conformance_fabric_barrier_arrive(att1_aimu_conformance_endpoint *endpoint,
                                                          uint32_t tile_id,
                                                          const uint32_t *participants,
                                                          size_t participant_count,
                                                          int *out_complete);
att1_status_t att1_aimu_conformance_fabric_get_counters(att1_aimu_conformance_endpoint *endpoint,
                                                        att1_fabric_counters *out);
att1_status_t att1_aimu_conformance_trace_get_snapshot(att1_aimu_conformance_endpoint *endpoint,
                                                       att1_aimu_trace_snapshot *out);

#ifdef __cplusplus
}
#endif

#endif
