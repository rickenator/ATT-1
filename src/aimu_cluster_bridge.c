/*
 * aimu_cluster_bridge.c  —  M167 two-endpoint activation/barrier relay.
 *
 * See include/att1_aimu_cluster_bridge.h for the design rationale. This is
 * a thin host-side relay built entirely out of the already-frozen
 * att1_aimu_conformance_endpoint fabric ops (M160/M161), so it is
 * transport-agnostic: it works identically whether tile_a/tile_b are two
 * in-process endpoints (M161) or two att1-aimu-endpoint daemon processes
 * connected over separate Unix domain sockets (M162).
 */

#include "att1_aimu_cluster_bridge.h"

#include <string.h>

/* Generous local scratch capacity for one relayed packet's payload. Must be
 * >= the configured fabric_max_payload_bytes on both bridged endpoints. */
#define ATT1_CLUSTER_BRIDGE_SCRATCH_BYTES 65536u

static att1_status_t relay_one(att1_aimu_conformance_endpoint *from,
                               att1_aimu_conformance_endpoint *to,
                               uint32_t global_source_tile,
                               uint64_t *out_relayed)
{
    unsigned char scratch[ATT1_CLUSTER_BRIDGE_SCRATCH_BYTES];
    att1_fabric_packet packet;
    size_t payload_bytes = 0u;
    att1_status_t status;

    memset(&packet, 0, sizeof(packet));
    status = att1_aimu_conformance_fabric_receive(from,
                                                  ATT1_CLUSTER_BRIDGE_PROXY_TILE,
                                                  &packet,
                                                  scratch,
                                                  sizeof(scratch),
                                                  &payload_bytes);
    if (status == ATT1_ERR_QUEUE_EMPTY) {
        return ATT1_OK;
    }
    if (status != ATT1_OK) {
        return status;
    }

    status = att1_aimu_conformance_fabric_send(to,
                                               global_source_tile,
                                               ATT1_CLUSTER_BRIDGE_LOCAL_TILE,
                                               packet.type,
                                               scratch,
                                               payload_bytes,
                                               packet.tag);
    if (status != ATT1_OK) {
        return status;
    }

    if (out_relayed != NULL) {
        (*out_relayed)++;
    }
    return ATT1_OK;
}

att1_status_t att1_aimu_cluster_bridge_init(att1_aimu_cluster_bridge *bridge,
                                            att1_aimu_conformance_endpoint *tile_a,
                                            att1_aimu_conformance_endpoint *tile_b)
{
    if ((bridge == NULL) || (tile_a == NULL) || (tile_b == NULL)) {
        return ATT1_ERR_INVALID_ARG;
    }

    memset(bridge, 0, sizeof(*bridge));
    bridge->tile_a = tile_a;
    bridge->tile_b = tile_b;
    return ATT1_OK;
}

att1_status_t att1_aimu_cluster_bridge_pump(att1_aimu_cluster_bridge *bridge)
{
    att1_status_t status;

    if (bridge == NULL) {
        return ATT1_ERR_INVALID_ARG;
    }

    /* Anything tile_a queued for the peer (proxy slot 1 on A) moves to
     * tile_b's real compute tile (0), tagged as originating from the
     * global "tile A" identity (0), and vice versa. */
    status = relay_one(bridge->tile_a, bridge->tile_b,
                       ATT1_CLUSTER_BRIDGE_LOCAL_TILE, &bridge->packets_relayed);
    if (status != ATT1_OK) {
        return status;
    }

    status = relay_one(bridge->tile_b, bridge->tile_a,
                       ATT1_CLUSTER_BRIDGE_LOCAL_TILE, &bridge->packets_relayed);
    if (status != ATT1_OK) {
        return status;
    }

    return ATT1_OK;
}

att1_status_t att1_aimu_cluster_bridge_send(att1_aimu_cluster_bridge *bridge,
                                            int from_a,
                                            att1_packet_type type,
                                            const void *payload,
                                            size_t payload_bytes,
                                            uint64_t tag)
{
    att1_aimu_conformance_endpoint *src = NULL;
    att1_status_t status;

    if (bridge == NULL) {
        return ATT1_ERR_INVALID_ARG;
    }
    src = from_a ? bridge->tile_a : bridge->tile_b;

    /* Enqueue on the sender's own proxy slot (1); the peer is the only
     * other endpoint that will ever see it via att1_aimu_cluster_bridge_pump(). */
    status = att1_aimu_conformance_fabric_send(src,
                                               ATT1_CLUSTER_BRIDGE_LOCAL_TILE,
                                               ATT1_CLUSTER_BRIDGE_PROXY_TILE,
                                               type,
                                               payload,
                                               payload_bytes,
                                               tag);
    if (status != ATT1_OK) {
        return status;
    }

    return att1_aimu_cluster_bridge_pump(bridge);
}

static const uint32_t g_bridge_barrier_participants[2] = {
    ATT1_CLUSTER_BRIDGE_LOCAL_TILE,
    ATT1_CLUSTER_BRIDGE_PROXY_TILE
};

att1_status_t att1_aimu_cluster_bridge_barrier(att1_aimu_cluster_bridge *bridge,
                                               int is_a,
                                               int *out_complete)
{
    att1_aimu_conformance_endpoint *self_ep = NULL;
    int local_complete = 0;
    att1_status_t status;

    if (bridge == NULL) {
        return ATT1_ERR_INVALID_ARG;
    }
    if (out_complete != NULL) {
        *out_complete = 0;
    }

    self_ep = is_a ? bridge->tile_a : bridge->tile_b;

    status = att1_aimu_conformance_fabric_barrier_arrive(self_ep,
                                                         ATT1_CLUSTER_BRIDGE_LOCAL_TILE,
                                                         g_bridge_barrier_participants, 2u,
                                                         &local_complete);
    if (status != ATT1_OK) {
        return status;
    }

    if (is_a) {
        bridge->a_arrived = 1;
    } else {
        bridge->b_arrived = 1;
    }

    if (!(bridge->a_arrived && bridge->b_arrived)) {
        /* Peer has not arrived yet; nothing more to do this call. */
        return ATT1_OK;
    }

    /* Both sides have arrived: complete each endpoint's local barrier by
     * arriving on its proxy slot on the peer's behalf. */
    status = att1_aimu_conformance_fabric_barrier_arrive(bridge->tile_a,
                                                         ATT1_CLUSTER_BRIDGE_PROXY_TILE,
                                                         g_bridge_barrier_participants, 2u,
                                                         &local_complete);
    if (status != ATT1_OK) {
        return status;
    }

    status = att1_aimu_conformance_fabric_barrier_arrive(bridge->tile_b,
                                                         ATT1_CLUSTER_BRIDGE_PROXY_TILE,
                                                         g_bridge_barrier_participants, 2u,
                                                         &local_complete);
    if (status != ATT1_OK) {
        return status;
    }

    bridge->a_arrived = 0;
    bridge->b_arrived = 0;
    bridge->barriers_completed++;
    if (out_complete != NULL) {
        *out_complete = 1;
    }
    return ATT1_OK;
}
