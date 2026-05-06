#ifndef ATT1_FABRIC_H
#define ATT1_FABRIC_H

#include <stddef.h>
#include <stdint.h>

typedef enum att1_status {
    ATT1_OK = 0,
    ATT1_ERR_INVALID = -1,
    ATT1_ERR_QUEUE_EMPTY = -2,
    ATT1_ERR_QUEUE_FULL = -3,
    ATT1_ERR_TIMEOUT = -4,
    ATT1_ERR_NO_MEMORY = -5
} att1_status;

typedef enum att1_packet_type {
    ATT1_PACKET_ACTIVATION = 0,
    ATT1_PACKET_LOGITS,
    ATT1_PACKET_KV_PAGE,
    ATT1_PACKET_CONTROL,
    ATT1_PACKET_BARRIER,
    ATT1_PACKET_TRACE
} att1_packet_type;

typedef struct att1_fabric_packet {
    att1_packet_type type;
    uint32_t source_tile;
    uint32_t target_tile;
    uint32_t payload_bytes;
    uint64_t tag;
} att1_fabric_packet;

typedef struct att1_fabric_bus_config {
    size_t tile_count;
    size_t queue_capacity;
    size_t max_payload_bytes;
} att1_fabric_bus_config;

typedef struct att1_fabric_counters {
    uint64_t packets_sent;
    uint64_t packets_received;
    uint64_t broadcast_packets;
    uint64_t payload_bytes_sent;
    uint64_t payload_bytes_received;
    uint64_t queue_full_errors;
    uint64_t invalid_packets;
    uint64_t empty_receives;
    uint64_t barrier_arrivals;
    uint64_t barrier_completions;
} att1_fabric_counters;

typedef struct att1_fabric {
    att1_fabric_bus_config config;
    att1_fabric_counters counters;
    void *queues;
    unsigned char *payload_storage;
    unsigned char *barrier_expected;
    unsigned char *barrier_arrived;
    size_t barrier_expected_count;
    int barrier_active;
} att1_fabric;

/*
 * Create a fixed-capacity packet fabric.
 *
 * Each tile owns one inbound queue. Sending copies the packet header and
 * payload into the destination queue; the caller keeps ownership of the
 * original payload memory.
 */
int att1_fabric_create(att1_fabric *fabric,
                       const att1_fabric_bus_config *config);

/*
 * Release all queue and barrier storage.
 *
 * Passing NULL is allowed.
 */
void att1_fabric_destroy(att1_fabric *fabric);

/*
 * Send one packet to a destination tile.
 *
 * payload may be NULL only when payload_bytes is zero. Queue full returns
 * ATT1_ERR_QUEUE_FULL and the packet is not enqueued.
 */
int att1_fabric_send(att1_fabric *fabric,
                     uint32_t source_tile,
                     uint32_t target_tile,
                     att1_packet_type type,
                     const void *payload,
                     size_t payload_bytes,
                     uint64_t tag);

/*
 * Receive one packet from a tile's inbound queue.
 *
 * This is nonblocking. Empty queues return ATT1_ERR_QUEUE_EMPTY. The payload is
 * copied into caller-owned storage and the queue entry is removed only after
 * output capacity has been validated.
 */
int att1_fabric_receive(att1_fabric *fabric,
                        uint32_t tile_id,
                        att1_fabric_packet *out_packet,
                        void *out_payload,
                        size_t out_payload_capacity,
                        size_t *out_payload_bytes);

/*
 * Poll for a packet up to timeout_ticks attempts.
 *
 * This simulator has no pthread runtime; timeout_ticks is a deterministic poll
 * budget. ATT1_ERR_TIMEOUT is returned if no packet is available.
 */
int att1_fabric_receive_timeout(att1_fabric *fabric,
                                uint32_t tile_id,
                                att1_fabric_packet *out_packet,
                                void *out_payload,
                                size_t out_payload_capacity,
                                size_t *out_payload_bytes,
                                size_t timeout_ticks);

/*
 * Broadcast to all tiles or a caller-specified group.
 *
 * Broadcast excludes the source tile. If group_tiles is NULL, the destination
 * group is all fabric tiles except source_tile. The send is preflighted: if any
 * destination is invalid or full, no destination receives the packet.
 */
int att1_fabric_broadcast(att1_fabric *fabric,
                          uint32_t source_tile,
                          const uint32_t *group_tiles,
                          size_t group_count,
                          att1_packet_type type,
                          const void *payload,
                          size_t payload_bytes,
                          uint64_t tag);

/*
 * Arrive at a simple single-generation barrier.
 *
 * The first arriving tile defines the participant set. Later arrivals must pass
 * the same set. out_complete is set to 1 only for the arrival that completes
 * the barrier; the barrier then resets for reuse.
 */
int att1_fabric_barrier_arrive(att1_fabric *fabric,
                               uint32_t tile_id,
                               const uint32_t *participants,
                               size_t participant_count,
                               int *out_complete);

void att1_fabric_get_counters(const att1_fabric *fabric,
                              att1_fabric_counters *out_counters);
void att1_fabric_reset_counters(att1_fabric *fabric);

int att1_sim_fabric_bus_create(att1_fabric *fabric,
                               size_t tile_count,
                               size_t queue_capacity,
                               size_t max_payload_bytes);

#endif
