#ifndef ATT1_RUNTIME_H
#define ATT1_RUNTIME_H

#include "att1_fabric.h"
#include "att1_tile.h"

#include <stddef.h>
#include <stdint.h>

typedef enum att1_request_state {
    ATT1_REQUEST_PENDING = 0,
    ATT1_REQUEST_RUNNING,
    ATT1_REQUEST_DONE,
    ATT1_REQUEST_FAILED
} att1_request_state;

typedef struct att1_request {
    uint64_t id;
    att1_request_state state;
} att1_request;

#define ATT1_RUNTIME_MAX_COMMAND_PAYLOAD 64u

typedef enum att1_runtime_command_type {
    ATT1_RUNTIME_CMD_LOAD_MODEL_SHARD = 0,
    ATT1_RUNTIME_CMD_RUN_LAYER_RANGE,
    ATT1_RUNTIME_CMD_SEND_ACTIVATION,
    ATT1_RUNTIME_CMD_RECV_ACTIVATION,
    ATT1_RUNTIME_CMD_SYNC_BARRIER,
    ATT1_RUNTIME_CMD_SHUTDOWN
} att1_runtime_command_type;

typedef struct att1_runtime_command {
    att1_runtime_command_type type;
    uint32_t peer_tile;
    uint32_t layer_start;
    uint32_t layer_end;
    uint64_t arg0;
    uint64_t tag;
    size_t payload_bytes;
    unsigned char payload[ATT1_RUNTIME_MAX_COMMAND_PAYLOAD];
} att1_runtime_command;

typedef struct att1_runtime_config {
    size_t tile_count;
    size_t command_queue_capacity;
    size_t fabric_queue_capacity;
    size_t fabric_max_payload_bytes;
} att1_runtime_config;

typedef struct att1_runtime {
    att1_runtime_config config;
    att1_fabric fabric;
    att1_tile_state *tiles;
    void *impl;
} att1_runtime;

/*
 * Create/destroy a simulated tile runtime.
 *
 * Creation allocates tile state, command queues, and an internal fabric bus.
 * Worker threads are not started until att1_runtime_start.
 */
int att1_runtime_create(att1_runtime *runtime,
                        const att1_runtime_config *config);
void att1_runtime_destroy(att1_runtime *runtime);

/*
 * Start/stop worker threads.
 *
 * Commands sent before start fail with ATT1_ERR_INVALID_ARG. Starting an already
 * running runtime returns ATT1_ERR_ALREADY_STARTED. Stop is idempotent and
 * succeeds as a no-op when the runtime is already stopped. A runtime may be
 * started again after stop.
 */
int att1_runtime_start(att1_runtime *runtime);
int att1_runtime_stop(att1_runtime *runtime);

/*
 * Copy a command into one tile's command queue.
 *
 * The command payload is copied into queue storage. Queue full returns
 * ATT1_ERR_QUEUE_FULL.
 */
int att1_runtime_send_command(att1_runtime *runtime,
                              uint32_t tile_id,
                              const att1_runtime_command *command);

int att1_runtime_get_tile_state(att1_runtime *runtime,
                                uint32_t tile_id,
                                att1_tile_state *out_state);

void att1_runtime_get_fabric_counters(att1_runtime *runtime,
                                      att1_fabric_counters *out_counters);

#endif
