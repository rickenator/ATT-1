#include "att1_fabric.h"

#include <stdlib.h>
#include <string.h>

typedef struct att1_fabric_queue {
    size_t head;
    size_t count;
    att1_fabric_packet *packets;
} att1_fabric_queue;

static int att1_mul_size(size_t lhs, size_t rhs, size_t *out)
{
    if ((lhs != 0u) && (rhs > (SIZE_MAX / lhs))) {
        return -1;
    }

    *out = lhs * rhs;
    return 0;
}

static int att1_fabric_config_valid(const att1_fabric_bus_config *config)
{
    size_t slots = 0u;
    size_t payload_bytes = 0u;

    if (config == NULL) {
        return 0;
    }

    if ((config->tile_count == 0u) ||
        (config->queue_capacity == 0u) ||
        (config->max_payload_bytes == 0u)) {
        return 0;
    }

    if (config->tile_count > (size_t)UINT32_MAX) {
        return 0;
    }

    if (config->max_payload_bytes > (size_t)UINT32_MAX) {
        return 0;
    }

    if (att1_mul_size(config->tile_count, config->queue_capacity, &slots) != 0) {
        return 0;
    }

    if (att1_mul_size(slots, config->max_payload_bytes, &payload_bytes) != 0) {
        return 0;
    }

    return payload_bytes > 0u;
}

static att1_fabric_queue *att1_fabric_queues(att1_fabric *fabric)
{
    return (att1_fabric_queue *)fabric->queues;
}

static const att1_fabric_queue *att1_fabric_queues_const(
    const att1_fabric *fabric)
{
    return (const att1_fabric_queue *)fabric->queues;
}

static unsigned char *att1_fabric_slot_payload(att1_fabric *fabric,
                                               size_t tile_id,
                                               size_t slot)
{
    const size_t flat_slot = (tile_id * fabric->config.queue_capacity) + slot;

    return &fabric->payload_storage[flat_slot * fabric->config.max_payload_bytes];
}

static int att1_fabric_valid_tile(const att1_fabric *fabric, uint32_t tile_id)
{
    return (fabric != NULL) && ((size_t)tile_id < fabric->config.tile_count);
}

static void att1_fabric_count_invalid(att1_fabric *fabric)
{
    if (fabric != NULL) {
        fabric->counters.invalid_packets++;
    }
}

static int att1_fabric_payload_valid(att1_fabric *fabric,
                                     const void *payload,
                                     size_t payload_bytes)
{
    if (payload_bytes > fabric->config.max_payload_bytes) {
        att1_fabric_count_invalid(fabric);
        return 0;
    }

    if ((payload_bytes > 0u) && (payload == NULL)) {
        att1_fabric_count_invalid(fabric);
        return 0;
    }

    return 1;
}

static int att1_fabric_queue_has_room(const att1_fabric *fabric,
                                      uint32_t tile_id)
{
    const att1_fabric_queue *queues = att1_fabric_queues_const(fabric);

    return queues[tile_id].count < fabric->config.queue_capacity;
}

static int att1_fabric_enqueue(att1_fabric *fabric,
                               uint32_t source_tile,
                               uint32_t target_tile,
                               att1_packet_type type,
                               const void *payload,
                               size_t payload_bytes,
                               uint64_t tag)
{
    att1_fabric_queue *queues = att1_fabric_queues(fabric);
    att1_fabric_queue *queue = &queues[target_tile];
    const size_t slot = (queue->head + queue->count) %
                        fabric->config.queue_capacity;
    unsigned char *slot_payload = att1_fabric_slot_payload(fabric,
                                                          target_tile,
                                                          slot);

    queue->packets[slot].type = type;
    queue->packets[slot].source_tile = source_tile;
    queue->packets[slot].target_tile = target_tile;
    queue->packets[slot].payload_bytes = (uint32_t)payload_bytes;
    queue->packets[slot].tag = tag;

    if (payload_bytes > 0u) {
        memcpy(slot_payload, payload, payload_bytes);
    }

    queue->count++;
    fabric->counters.packets_sent++;
    fabric->counters.payload_bytes_sent += payload_bytes;
    return ATT1_OK;
}

int att1_fabric_create(att1_fabric *fabric,
                       const att1_fabric_bus_config *config)
{
    att1_fabric_queue *queues = NULL;
    size_t slots = 0u;
    size_t payload_bytes = 0u;
    size_t tile = 0u;

    if ((fabric == NULL) || !att1_fabric_config_valid(config)) {
        return ATT1_ERR_INVALID;
    }

    memset(fabric, 0, sizeof(*fabric));
    fabric->config = *config;

    queues = calloc(config->tile_count, sizeof(*queues));
    fabric->barrier_expected = calloc(config->tile_count, sizeof(unsigned char));
    fabric->barrier_arrived = calloc(config->tile_count, sizeof(unsigned char));
    if ((queues == NULL) ||
        (fabric->barrier_expected == NULL) ||
        (fabric->barrier_arrived == NULL)) {
        fabric->queues = queues;
        att1_fabric_destroy(fabric);
        return ATT1_ERR_NO_MEMORY;
    }

    if (att1_mul_size(config->tile_count, config->queue_capacity, &slots) != 0) {
        fabric->queues = queues;
        att1_fabric_destroy(fabric);
        return ATT1_ERR_INVALID;
    }

    if (att1_mul_size(slots, config->max_payload_bytes, &payload_bytes) != 0) {
        fabric->queues = queues;
        att1_fabric_destroy(fabric);
        return ATT1_ERR_INVALID;
    }

    fabric->payload_storage = calloc(payload_bytes, sizeof(unsigned char));
    if (fabric->payload_storage == NULL) {
        fabric->queues = queues;
        att1_fabric_destroy(fabric);
        return ATT1_ERR_NO_MEMORY;
    }

    for (tile = 0u; tile < config->tile_count; tile++) {
        queues[tile].packets = calloc(config->queue_capacity,
                                      sizeof(*queues[tile].packets));
        if (queues[tile].packets == NULL) {
            fabric->queues = queues;
            att1_fabric_destroy(fabric);
            return ATT1_ERR_NO_MEMORY;
        }
    }

    fabric->queues = queues;
    return ATT1_OK;
}

void att1_fabric_destroy(att1_fabric *fabric)
{
    att1_fabric_queue *queues = NULL;
    size_t tile = 0u;

    if (fabric == NULL) {
        return;
    }

    queues = att1_fabric_queues(fabric);
    if (queues != NULL) {
        for (tile = 0u; tile < fabric->config.tile_count; tile++) {
            free(queues[tile].packets);
        }
    }

    free(queues);
    free(fabric->payload_storage);
    free(fabric->barrier_expected);
    free(fabric->barrier_arrived);
    memset(fabric, 0, sizeof(*fabric));
}

int att1_fabric_send(att1_fabric *fabric,
                     uint32_t source_tile,
                     uint32_t target_tile,
                     att1_packet_type type,
                     const void *payload,
                     size_t payload_bytes,
                     uint64_t tag)
{
    if ((fabric == NULL) || (fabric->queues == NULL)) {
        return ATT1_ERR_INVALID;
    }

    if (!att1_fabric_valid_tile(fabric, source_tile) ||
        !att1_fabric_valid_tile(fabric, target_tile)) {
        att1_fabric_count_invalid(fabric);
        return ATT1_ERR_INVALID;
    }

    if (!att1_fabric_payload_valid(fabric, payload, payload_bytes)) {
        return ATT1_ERR_INVALID;
    }

    if (!att1_fabric_queue_has_room(fabric, target_tile)) {
        fabric->counters.queue_full_errors++;
        return ATT1_ERR_QUEUE_FULL;
    }

    return att1_fabric_enqueue(fabric,
                               source_tile,
                               target_tile,
                               type,
                               payload,
                               payload_bytes,
                               tag);
}

int att1_fabric_receive(att1_fabric *fabric,
                        uint32_t tile_id,
                        att1_fabric_packet *out_packet,
                        void *out_payload,
                        size_t out_payload_capacity,
                        size_t *out_payload_bytes)
{
    att1_fabric_queue *queues = NULL;
    att1_fabric_queue *queue = NULL;
    att1_fabric_packet *packet = NULL;
    unsigned char *slot_payload = NULL;

    if ((fabric == NULL) || (fabric->queues == NULL) || (out_packet == NULL)) {
        return ATT1_ERR_INVALID;
    }

    if (!att1_fabric_valid_tile(fabric, tile_id)) {
        att1_fabric_count_invalid(fabric);
        return ATT1_ERR_INVALID;
    }

    queues = att1_fabric_queues(fabric);
    queue = &queues[tile_id];
    if (queue->count == 0u) {
        fabric->counters.empty_receives++;
        return ATT1_ERR_QUEUE_EMPTY;
    }

    packet = &queue->packets[queue->head];
    if ((packet->payload_bytes > out_payload_capacity) ||
        ((packet->payload_bytes > 0u) && (out_payload == NULL))) {
        att1_fabric_count_invalid(fabric);
        return ATT1_ERR_INVALID;
    }

    slot_payload = att1_fabric_slot_payload(fabric, tile_id, queue->head);
    *out_packet = *packet;
    if (packet->payload_bytes > 0u) {
        memcpy(out_payload, slot_payload, packet->payload_bytes);
    }

    if (out_payload_bytes != NULL) {
        *out_payload_bytes = packet->payload_bytes;
    }

    queue->head = (queue->head + 1u) % fabric->config.queue_capacity;
    queue->count--;
    fabric->counters.packets_received++;
    fabric->counters.payload_bytes_received += packet->payload_bytes;
    return ATT1_OK;
}

int att1_fabric_receive_timeout(att1_fabric *fabric,
                                uint32_t tile_id,
                                att1_fabric_packet *out_packet,
                                void *out_payload,
                                size_t out_payload_capacity,
                                size_t *out_payload_bytes,
                                size_t timeout_ticks)
{
    size_t tick = 0u;

    for (tick = 0u; tick <= timeout_ticks; tick++) {
        const int rc = att1_fabric_receive(fabric,
                                           tile_id,
                                           out_packet,
                                           out_payload,
                                           out_payload_capacity,
                                           out_payload_bytes);

        if (rc != ATT1_ERR_QUEUE_EMPTY) {
            return rc;
        }
    }

    return ATT1_ERR_TIMEOUT;
}

int att1_fabric_broadcast(att1_fabric *fabric,
                          uint32_t source_tile,
                          const uint32_t *group_tiles,
                          size_t group_count,
                          att1_packet_type type,
                          const void *payload,
                          size_t payload_bytes,
                          uint64_t tag)
{
    size_t tile = 0u;
    size_t sent = 0u;

    if ((fabric == NULL) || (fabric->queues == NULL)) {
        return ATT1_ERR_INVALID;
    }

    if (!att1_fabric_valid_tile(fabric, source_tile)) {
        att1_fabric_count_invalid(fabric);
        return ATT1_ERR_INVALID;
    }

    if (!att1_fabric_payload_valid(fabric, payload, payload_bytes)) {
        return ATT1_ERR_INVALID;
    }

    if ((group_tiles == NULL) && (group_count != 0u)) {
        att1_fabric_count_invalid(fabric);
        return ATT1_ERR_INVALID;
    }

    if (group_tiles != NULL) {
        for (tile = 0u; tile < group_count; tile++) {
            const uint32_t target = group_tiles[tile];

            if (!att1_fabric_valid_tile(fabric, target)) {
                att1_fabric_count_invalid(fabric);
                return ATT1_ERR_INVALID;
            }

            if ((target != source_tile) && !att1_fabric_queue_has_room(fabric, target)) {
                fabric->counters.queue_full_errors++;
                return ATT1_ERR_QUEUE_FULL;
            }
        }

        for (tile = 0u; tile < group_count; tile++) {
            const uint32_t target = group_tiles[tile];

            if (target != source_tile) {
                (void)att1_fabric_enqueue(fabric,
                                          source_tile,
                                          target,
                                          type,
                                          payload,
                                          payload_bytes,
                                          tag);
                sent++;
            }
        }
    } else {
        for (tile = 0u; tile < fabric->config.tile_count; tile++) {
            if ((tile != (size_t)source_tile) &&
                !att1_fabric_queue_has_room(fabric, (uint32_t)tile)) {
                fabric->counters.queue_full_errors++;
                return ATT1_ERR_QUEUE_FULL;
            }
        }

        for (tile = 0u; tile < fabric->config.tile_count; tile++) {
            if (tile != (size_t)source_tile) {
                (void)att1_fabric_enqueue(fabric,
                                          source_tile,
                                          (uint32_t)tile,
                                          type,
                                          payload,
                                          payload_bytes,
                                          tag);
                sent++;
            }
        }
    }

    fabric->counters.broadcast_packets += sent;
    return ATT1_OK;
}

static int att1_fabric_participant_set_valid(att1_fabric *fabric,
                                             const uint32_t *participants,
                                             size_t participant_count)
{
    size_t i = 0u;

    if ((participants == NULL) || (participant_count == 0u)) {
        return 0;
    }

    if (participant_count > fabric->config.tile_count) {
        return 0;
    }

    memset(fabric->barrier_expected, 0, fabric->config.tile_count);
    for (i = 0u; i < participant_count; i++) {
        const uint32_t tile = participants[i];

        if (!att1_fabric_valid_tile(fabric, tile)) {
            return 0;
        }

        if (fabric->barrier_expected[tile] != 0u) {
            return 0;
        }

        fabric->barrier_expected[tile] = 1u;
    }

    return 1;
}

static int att1_fabric_participant_set_matches(att1_fabric *fabric,
                                               const uint32_t *participants,
                                               size_t participant_count)
{
    unsigned char *saved = NULL;
    size_t tile = 0u;
    int matches = 1;

    saved = malloc(fabric->config.tile_count);
    if (saved == NULL) {
        return 0;
    }

    memcpy(saved, fabric->barrier_expected, fabric->config.tile_count);
    if (!att1_fabric_participant_set_valid(fabric, participants, participant_count)) {
        matches = 0;
    } else {
        for (tile = 0u; tile < fabric->config.tile_count; tile++) {
            if (fabric->barrier_expected[tile] != saved[tile]) {
                matches = 0;
                break;
            }
        }
    }

    memcpy(fabric->barrier_expected, saved, fabric->config.tile_count);
    free(saved);
    return matches;
}

int att1_fabric_barrier_arrive(att1_fabric *fabric,
                               uint32_t tile_id,
                               const uint32_t *participants,
                               size_t participant_count,
                               int *out_complete)
{
    size_t tile = 0u;
    size_t arrived_count = 0u;

    if (out_complete != NULL) {
        *out_complete = 0;
    }

    if ((fabric == NULL) || (fabric->barrier_expected == NULL) ||
        (fabric->barrier_arrived == NULL)) {
        return ATT1_ERR_INVALID;
    }

    if (!att1_fabric_valid_tile(fabric, tile_id)) {
        att1_fabric_count_invalid(fabric);
        return ATT1_ERR_INVALID;
    }

    if (fabric->barrier_active == 0) {
        if (!att1_fabric_participant_set_valid(fabric,
                                               participants,
                                               participant_count)) {
            att1_fabric_count_invalid(fabric);
            return ATT1_ERR_INVALID;
        }

        memset(fabric->barrier_arrived, 0, fabric->config.tile_count);
        fabric->barrier_expected_count = participant_count;
        fabric->barrier_active = 1;
    } else if (!att1_fabric_participant_set_matches(fabric,
                                                    participants,
                                                    participant_count)) {
        att1_fabric_count_invalid(fabric);
        return ATT1_ERR_INVALID;
    }

    if (fabric->barrier_expected[tile_id] == 0u) {
        att1_fabric_count_invalid(fabric);
        return ATT1_ERR_INVALID;
    }

    if (fabric->barrier_arrived[tile_id] != 0u) {
        att1_fabric_count_invalid(fabric);
        return ATT1_ERR_INVALID;
    }

    fabric->barrier_arrived[tile_id] = 1u;
    fabric->counters.barrier_arrivals++;

    for (tile = 0u; tile < fabric->config.tile_count; tile++) {
        if (fabric->barrier_arrived[tile] != 0u) {
            arrived_count++;
        }
    }

    if (arrived_count == fabric->barrier_expected_count) {
        fabric->counters.barrier_completions++;
        fabric->barrier_active = 0;
        fabric->barrier_expected_count = 0u;
        memset(fabric->barrier_expected, 0, fabric->config.tile_count);
        memset(fabric->barrier_arrived, 0, fabric->config.tile_count);
        if (out_complete != NULL) {
            *out_complete = 1;
        }
    }

    return ATT1_OK;
}

void att1_fabric_get_counters(const att1_fabric *fabric,
                              att1_fabric_counters *out_counters)
{
    if ((fabric == NULL) || (out_counters == NULL)) {
        return;
    }

    *out_counters = fabric->counters;
}

void att1_fabric_reset_counters(att1_fabric *fabric)
{
    if (fabric == NULL) {
        return;
    }

    memset(&fabric->counters, 0, sizeof(fabric->counters));
}
