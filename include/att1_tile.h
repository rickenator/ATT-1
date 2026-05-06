#ifndef ATT1_TILE_H
#define ATT1_TILE_H

#include "att1_config.h"

#include <stdint.h>

typedef struct att1_tile_counters {
    uint64_t commands_processed;
    uint64_t load_model_shard_commands;
    uint64_t run_layer_range_commands;
    uint64_t send_activation_commands;
    uint64_t recv_activation_commands;
    uint64_t barrier_commands;
    uint64_t shutdown_commands;
    uint64_t fabric_send_ops;
    uint64_t fabric_recv_ops;
    uint64_t errors;
} att1_tile_counters;

typedef struct att1_tile_state {
    uint32_t tile_id;
    int running;
    uint64_t model_shard_id;
    uint32_t last_layer_start;
    uint32_t last_layer_end;
    att1_tile_counters counters;
} att1_tile_state;

typedef struct att1_tile {
    att1_tile_config config;
    att1_tile_state state;
} att1_tile;

void att1_tile_state_init(att1_tile_state *state, uint32_t tile_id);

#endif
