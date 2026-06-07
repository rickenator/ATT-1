#include "att1_fabric.h"

#include <stdio.h>
#include <string.h>

static int check_bytes(const unsigned char *actual,
                       const unsigned char *expected,
                       size_t count)
{
    return memcmp(actual, expected, count) == 0;
}

static int recv_packet(att1_fabric *fabric,
                       uint32_t tile,
                       att1_packet_type expected_type,
                       uint32_t expected_source,
                       uint64_t expected_tag,
                       const unsigned char *expected_payload,
                       size_t expected_bytes)
{
    unsigned char payload[16] = {0u};
    att1_fabric_packet packet;
    size_t payload_bytes = 0u;

    if (att1_fabric_receive(fabric,
                            tile,
                            &packet,
                            payload,
                            sizeof(payload),
                            &payload_bytes) != ATT1_OK) {
        return 0;
    }

    if ((packet.type != expected_type) ||
        (packet.source_tile != expected_source) ||
        (packet.target_tile != tile) ||
        (packet.tag != expected_tag) ||
        (payload_bytes != expected_bytes) ||
        (packet.payload_bytes != expected_bytes)) {
        return 0;
    }

    return check_bytes(payload, expected_payload, expected_bytes);
}

int main(void)
{
    const unsigned char payload_a[3] = {1u, 2u, 3u};
    const unsigned char payload_b[3] = {4u, 5u, 6u};
    const unsigned char payload_c[3] = {7u, 8u, 9u};
    const uint32_t group[3] = {0u, 2u, 3u};
    const uint32_t barrier_group[3] = {0u, 1u, 2u};
    unsigned char mutable_payload[3] = {10u, 11u, 12u};
    unsigned char payload_out[16] = {0u};
    att1_fabric_packet packet;
    att1_fabric_counters counters;
    size_t payload_bytes = 0u;
    int complete = 1;
    att1_fabric fabric;

    if (att1_sim_fabric_bus_create(&fabric, 4u, 3u, 16u) != ATT1_OK) {
        fputs("fabric create failed\n", stderr);
        return 1;
    }

    if ((att1_fabric_send(&fabric,
                          0u,
                          1u,
                          ATT1_PACKET_CONTROL,
                          payload_a,
                          sizeof(payload_a),
                          11u) != ATT1_OK) ||
        (att1_fabric_send(&fabric,
                          2u,
                          1u,
                          ATT1_PACKET_LOGITS,
                          payload_b,
                          sizeof(payload_b),
                          12u) != ATT1_OK)) {
        fputs("fabric send ordering setup failed\n", stderr);
        att1_fabric_destroy(&fabric);
        return 1;
    }

    if (!recv_packet(&fabric,
                     1u,
                     ATT1_PACKET_CONTROL,
                     0u,
                     11u,
                     payload_a,
                     sizeof(payload_a)) ||
        !recv_packet(&fabric,
                     1u,
                     ATT1_PACKET_LOGITS,
                     2u,
                     12u,
                     payload_b,
                     sizeof(payload_b))) {
        fputs("fabric send/receive ordering failed\n", stderr);
        att1_fabric_destroy(&fabric);
        return 1;
    }

    if (att1_fabric_send(&fabric,
                         0u,
                         9u,
                         ATT1_PACKET_CONTROL,
                         payload_a,
                         sizeof(payload_a),
                         13u) != ATT1_ERR_INVALID_ARG) {
        fputs("fabric invalid tile send check failed\n", stderr);
        att1_fabric_destroy(&fabric);
        return 1;
    }

    if (att1_fabric_receive(&fabric,
                            1u,
                            &packet,
                            payload_out,
                            sizeof(payload_out),
                            &payload_bytes) != ATT1_ERR_QUEUE_EMPTY) {
        fputs("fabric empty receive check failed\n", stderr);
        att1_fabric_destroy(&fabric);
        return 1;
    }

    if (att1_fabric_receive_timeout(&fabric,
                                    1u,
                                    &packet,
                                    payload_out,
                                    sizeof(payload_out),
                                    &payload_bytes,
                                    2u) != ATT1_ERR_TIMEOUT) {
        fputs("fabric timeout receive check failed\n", stderr);
        att1_fabric_destroy(&fabric);
        return 1;
    }

    if (att1_fabric_send(&fabric,
                         0u,
                         1u,
                         ATT1_PACKET_ACTIVATION,
                         mutable_payload,
                         sizeof(mutable_payload),
                         20u) != ATT1_OK) {
        fputs("fabric payload copy send failed\n", stderr);
        att1_fabric_destroy(&fabric);
        return 1;
    }

    mutable_payload[0] = 99u;
    if (!recv_packet(&fabric,
                     1u,
                     ATT1_PACKET_ACTIVATION,
                     0u,
                     20u,
                     (const unsigned char[]){10u, 11u, 12u},
                     3u)) {
        fputs("fabric payload copy ownership failed\n", stderr);
        att1_fabric_destroy(&fabric);
        return 1;
    }

    if (att1_fabric_broadcast(&fabric,
                              0u,
                              NULL,
                              0u,
                              ATT1_PACKET_TRACE,
                              payload_c,
                              sizeof(payload_c),
                              30u) != ATT1_OK) {
        fputs("fabric all-tile broadcast failed\n", stderr);
        att1_fabric_destroy(&fabric);
        return 1;
    }

    if (att1_fabric_receive(&fabric,
                            0u,
                            &packet,
                            payload_out,
                            sizeof(payload_out),
                            &payload_bytes) != ATT1_ERR_QUEUE_EMPTY) {
        fputs("fabric broadcast sender exclusion failed\n", stderr);
        att1_fabric_destroy(&fabric);
        return 1;
    }

    if (!recv_packet(&fabric,
                     1u,
                     ATT1_PACKET_TRACE,
                     0u,
                     30u,
                     payload_c,
                     sizeof(payload_c)) ||
        !recv_packet(&fabric,
                     2u,
                     ATT1_PACKET_TRACE,
                     0u,
                     30u,
                     payload_c,
                     sizeof(payload_c)) ||
        !recv_packet(&fabric,
                     3u,
                     ATT1_PACKET_TRACE,
                     0u,
                     30u,
                     payload_c,
                     sizeof(payload_c))) {
        fputs("fabric all-tile broadcast receive failed\n", stderr);
        att1_fabric_destroy(&fabric);
        return 1;
    }

    if (att1_fabric_broadcast(&fabric,
                              2u,
                              group,
                              3u,
                              ATT1_PACKET_KV_PAGE,
                              payload_a,
                              sizeof(payload_a),
                              31u) != ATT1_OK) {
        fputs("fabric group broadcast failed\n", stderr);
        att1_fabric_destroy(&fabric);
        return 1;
    }

    if (!recv_packet(&fabric,
                     0u,
                     ATT1_PACKET_KV_PAGE,
                     2u,
                     31u,
                     payload_a,
                     sizeof(payload_a)) ||
        !recv_packet(&fabric,
                     3u,
                     ATT1_PACKET_KV_PAGE,
                     2u,
                     31u,
                     payload_a,
                     sizeof(payload_a))) {
        fputs("fabric group broadcast receive failed\n", stderr);
        att1_fabric_destroy(&fabric);
        return 1;
    }

    if (att1_fabric_receive(&fabric,
                            2u,
                            &packet,
                            payload_out,
                            sizeof(payload_out),
                            &payload_bytes) != ATT1_ERR_QUEUE_EMPTY) {
        fputs("fabric group broadcast source exclusion failed\n", stderr);
        att1_fabric_destroy(&fabric);
        return 1;
    }

    if ((att1_fabric_send(&fabric, 0u, 1u, ATT1_PACKET_CONTROL, payload_a, 3u, 40u) != ATT1_OK) ||
        (att1_fabric_send(&fabric, 0u, 1u, ATT1_PACKET_CONTROL, payload_a, 3u, 41u) != ATT1_OK) ||
        (att1_fabric_send(&fabric, 0u, 1u, ATT1_PACKET_CONTROL, payload_a, 3u, 42u) != ATT1_OK) ||
        (att1_fabric_send(&fabric, 0u, 1u, ATT1_PACKET_CONTROL, payload_a, 3u, 43u) != ATT1_ERR_QUEUE_FULL)) {
        fputs("fabric queue capacity check failed\n", stderr);
        att1_fabric_destroy(&fabric);
        return 1;
    }

    if (!recv_packet(&fabric, 1u, ATT1_PACKET_CONTROL, 0u, 40u, payload_a, 3u) ||
        !recv_packet(&fabric, 1u, ATT1_PACKET_CONTROL, 0u, 41u, payload_a, 3u) ||
        !recv_packet(&fabric, 1u, ATT1_PACKET_CONTROL, 0u, 42u, payload_a, 3u)) {
        fputs("fabric queue full preserved packet order failed\n", stderr);
        att1_fabric_destroy(&fabric);
        return 1;
    }

    complete = 1;
    if ((att1_fabric_barrier_arrive(&fabric,
                                    0u,
                                    barrier_group,
                                    3u,
                                    &complete) != ATT1_OK) ||
        (complete != 0)) {
        fputs("fabric barrier first arrival failed\n", stderr);
        att1_fabric_destroy(&fabric);
        return 1;
    }

    if ((att1_fabric_barrier_arrive(&fabric,
                                    1u,
                                    barrier_group,
                                    3u,
                                    &complete) != ATT1_OK) ||
        (complete != 0)) {
        fputs("fabric barrier second arrival failed\n", stderr);
        att1_fabric_destroy(&fabric);
        return 1;
    }

    if ((att1_fabric_barrier_arrive(&fabric,
                                    2u,
                                    barrier_group,
                                    3u,
                                    &complete) != ATT1_OK) ||
        (complete != 1)) {
        fputs("fabric barrier completion failed\n", stderr);
        att1_fabric_destroy(&fabric);
        return 1;
    }

    att1_fabric_get_counters(&fabric, &counters);
    if ((counters.packets_sent != 11u) ||
        (counters.packets_received != 11u) ||
        (counters.broadcast_packets != 5u) ||
        (counters.queue_full_errors != 1u) ||
        (counters.invalid_packets != 1u) ||
        (counters.empty_receives < 3u) ||
        (counters.barrier_arrivals != 3u) ||
        (counters.barrier_completions != 1u) ||
        (counters.payload_bytes_sent != 33u) ||
        (counters.payload_bytes_received != 33u)) {
        fputs("fabric counter check failed\n", stderr);
        att1_fabric_destroy(&fabric);
        return 1;
    }

    att1_fabric_destroy(&fabric);
    puts("fabric test passed");
    return 0;
}
