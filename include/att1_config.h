#ifndef ATT1_CONFIG_H
#define ATT1_CONFIG_H

#include <stddef.h>
#include <stdint.h>

typedef struct att1_tile_config {
    uint32_t tile_id;
    uint32_t compute_lanes;
    size_t local_model_memory_bytes;
    size_t local_scratch_bytes;
} att1_tile_config;

typedef struct att1_fabric_config {
    uint32_t tile_count;
    uint32_t max_packet_bytes;
    uint32_t link_latency_cycles;
    uint64_t link_bandwidth_bytes_per_sec;
} att1_fabric_config;

typedef struct att1_kv_config {
    size_t page_bytes;
    size_t max_pages;
    uint32_t address_bits;
} att1_kv_config;

typedef struct att1_sim_config {
    att1_tile_config tile;
    att1_fabric_config fabric;
    att1_kv_config kv;
    uint64_t random_seed;
} att1_sim_config;

att1_sim_config att1_default_config(void);

#endif
