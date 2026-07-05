#ifndef ATT1_AIMU_CLUSTER_BRIDGE_H
#define ATT1_AIMU_CLUSTER_BRIDGE_H

/*
 * att1_aimu_cluster_bridge.h  —  Two-endpoint activation/barrier relay
 * (M167 "Two-tile emulated cluster decode").
 *
 * Each `att1_aimu_conformance_endpoint` (M161 in-process, M162 socket-backed)
 * owns its own private `att1_fabric` bus (M4/M114): fabric packets sent on
 * one endpoint are never visible to a different endpoint instance, even
 * when both represent tiles of the same logical decode step. This mirrors
 * real hardware: two separate AIMU/PCIe tiles do not share a queue, they
 * are bridged by *something* that moves packets across the transport
 * (M93 §8.8).
 *
 * `att1_aimu_cluster_bridge` is that bridge. It owns no tensor or KV state;
 * it is a thin host-side relay that treats each endpoint's local fabric as
 * a two-tile bus: local tile 0 is the endpoint's real compute tile, and
 * local tile 1 is a "remote proxy" slot reserved for the bridge's own use.
 * `att1_aimu_cluster_bridge_pump_activation()` copies at most one pending
 * packet addressed to the proxy slot on one endpoint into the peer
 * endpoint's real tile inbox (tile 0), so tile-to-tile activation routing
 * works identically whether both endpoints are in-process objects or two
 * separate `att1-aimu-endpoint` daemon processes connected over a Unix
 * domain socket (M162).
 *
 * Barrier arrivals are relayed the same way: a local arrival at tile 0
 * is mirrored as an arrival at the peer's proxy slot (tile 1) once both
 * sides have locally arrived, so `att1_fabric_barrier_arrive()`'s
 * two-participant {0,1} barrier on *each* endpoint completes only after
 * both physical tiles have reached the rendezvous point.
 *
 * This is a coordination convenience, not a new transport: it is built
 * entirely out of the already-frozen
 * `att1_aimu_conformance_fabric_send/receive/broadcast/barrier_arrive`
 * calls (M160/M161), so it works unchanged over the M162 socket transport.
 */

#include "att1_aimu_conformance.h"
#include "att1_fabric.h"
#include "att1_status.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Local tile-id convention used by the bridge on every endpoint. */
#define ATT1_CLUSTER_BRIDGE_LOCAL_TILE   0u
#define ATT1_CLUSTER_BRIDGE_PROXY_TILE   1u
/* Minimum fabric tile_count each bridged endpoint must be configured with. */
#define ATT1_CLUSTER_BRIDGE_TILE_COUNT   2u

typedef struct att1_aimu_cluster_bridge {
    att1_aimu_conformance_endpoint *tile_a;
    att1_aimu_conformance_endpoint *tile_b;

    /* Barrier rendezvous state: set once the local tile on that side has
     * arrived but the peer has not yet been observed as arrived. */
    int a_arrived;
    int b_arrived;

    uint64_t packets_relayed;
    uint64_t barriers_completed;
} att1_aimu_cluster_bridge;

/*
 * Bind a bridge to two already-connected endpoints (in-process or
 * socket-backed). Both endpoints must have been created with a fabric
 * tile_count >= ATT1_CLUSTER_BRIDGE_TILE_COUNT (2). Ownership of the
 * endpoints is not transferred: the caller destroys them independently.
 */
att1_status_t att1_aimu_cluster_bridge_init(att1_aimu_cluster_bridge *bridge,
                                            att1_aimu_conformance_endpoint *tile_a,
                                            att1_aimu_conformance_endpoint *tile_b);

/*
 * Relay one activation/logits packet, if any is pending in either side's
 * proxy inbox (local tile 1), to the peer's real tile (local tile 0).
 *
 * Returns ATT1_OK whether or not a packet was relayed (this is a
 * nonblocking, best-effort pump, matching att1_fabric_receive()'s
 * nonblocking contract); ATT1_ERR_QUEUE_FULL if the destination inbox was
 * full when a pending packet was found and any other status is a hard
 * failure from one of the underlying endpoint calls.
 */
att1_status_t att1_aimu_cluster_bridge_pump(att1_aimu_cluster_bridge *bridge);

/*
 * Send one packet from tile_a's or tile_b's local compute tile (0) to the
 * peer's local compute tile (0), relaying it via the peer's proxy slot (1)
 * and pumping it across. `from_a` selects the direction: nonzero sends
 * tile_a -> tile_b, zero sends tile_b -> tile_a.
 */
att1_status_t att1_aimu_cluster_bridge_send(att1_aimu_cluster_bridge *bridge,
                                            int from_a,
                                            att1_packet_type type,
                                            const void *payload,
                                            size_t payload_bytes,
                                            uint64_t tag);

/*
 * Two-tile rendezvous barrier (M93 §8.2-4). Each side calls this once it
 * has reached its local barrier point (`is_a` selects which side is
 * arriving). This is nonblocking: the caller is expected to be a single
 * driver thread that calls it once per side per decode step (the same
 * pattern the two-tile decode tests use), so there is no other thread to
 * wait on. The call always registers the calling side's local arrival on
 * its own endpoint's fabric first; once *both* sides have arrived, the
 * bridge completes the rendezvous by arriving on each endpoint's proxy
 * slot (1) on the peer's behalf, which unblocks both endpoints' local
 * {0,1} barrier and increments `barriers_completed`. `*out_complete` is
 * set to 1 only on the call that observes both sides arrived.
 */
att1_status_t att1_aimu_cluster_bridge_barrier(att1_aimu_cluster_bridge *bridge,
                                               int is_a,
                                               int *out_complete);

#ifdef __cplusplus
}
#endif

#endif
