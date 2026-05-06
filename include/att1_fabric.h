#ifndef ATT1_FABRIC_H
#define ATT1_FABRIC_H

#include <stdint.h>

typedef enum att1_packet_kind {
    ATT1_PACKET_COMMAND = 0,
    ATT1_PACKET_TENSOR,
    ATT1_PACKET_KV,
    ATT1_PACKET_SYNC
} att1_packet_kind;

typedef struct att1_packet_header {
    att1_packet_kind kind;
    uint32_t source_tile;
    uint32_t target_tile;
    uint32_t payload_bytes;
} att1_packet_header;

#endif
