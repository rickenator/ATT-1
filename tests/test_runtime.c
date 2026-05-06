#define _POSIX_C_SOURCE 199309L

#include "att1_runtime.h"

#include <stdio.h>
#include <string.h>
#include <time.h>

static void tiny_sleep(void)
{
    const struct timespec ts = {
        .tv_sec = 0,
        .tv_nsec = 1000000L
    };

    (void)nanosleep(&ts, NULL);
}

static int wait_for_commands(att1_runtime *runtime,
                             uint32_t tile_id,
                             uint64_t expected)
{
    size_t tries = 0u;

    for (tries = 0u; tries < 200u; tries++) {
        att1_tile_state state;

        if (att1_runtime_get_tile_state(runtime, tile_id, &state) != ATT1_OK) {
            return 0;
        }

        if (state.counters.commands_processed >= expected) {
            return 1;
        }

        tiny_sleep();
    }

    return 0;
}

static int wait_for_running(att1_runtime *runtime,
                            uint32_t tile_id,
                            int running)
{
    size_t tries = 0u;

    for (tries = 0u; tries < 200u; tries++) {
        att1_tile_state state;

        if (att1_runtime_get_tile_state(runtime, tile_id, &state) != ATT1_OK) {
            return 0;
        }

        if (state.running == running) {
            return 1;
        }

        tiny_sleep();
    }

    return 0;
}

static int test_lifecycle_and_dispatch(void)
{
    const att1_runtime_config config = {
        .tile_count = 3u,
        .command_queue_capacity = 4u,
        .fabric_queue_capacity = 4u,
        .fabric_max_payload_bytes = ATT1_RUNTIME_MAX_COMMAND_PAYLOAD
    };
    att1_runtime_command command;
    att1_tile_state tile0;
    att1_tile_state tile1;
    att1_runtime runtime;

    if (att1_runtime_create(&runtime, &config) != ATT1_OK) {
        fputs("runtime create failed\n", stderr);
        return 0;
    }

    memset(&command, 0, sizeof(command));
    command.type = ATT1_RUNTIME_CMD_RUN_LAYER_RANGE;
    command.layer_start = 2u;
    command.layer_end = 5u;

    if (att1_runtime_send_command(&runtime, 0u, &command) != ATT1_ERR_INVALID) {
        fputs("runtime pre-start command policy failed\n", stderr);
        att1_runtime_destroy(&runtime);
        return 0;
    }

    if (att1_runtime_start(&runtime) != ATT1_OK) {
        fputs("runtime start failed\n", stderr);
        att1_runtime_destroy(&runtime);
        return 0;
    }

    if (att1_runtime_start(&runtime) != ATT1_ERR_ALREADY_STARTED) {
        fputs("runtime double-start policy failed\n", stderr);
        att1_runtime_destroy(&runtime);
        return 0;
    }

    if (att1_runtime_send_command(&runtime, 0u, &command) != ATT1_OK) {
        fputs("runtime dispatch send failed\n", stderr);
        att1_runtime_destroy(&runtime);
        return 0;
    }

    if (!wait_for_commands(&runtime, 0u, 1u)) {
        fputs("runtime tile 0 did not process command\n", stderr);
        att1_runtime_destroy(&runtime);
        return 0;
    }

    if ((att1_runtime_get_tile_state(&runtime, 0u, &tile0) != ATT1_OK) ||
        (att1_runtime_get_tile_state(&runtime, 1u, &tile1) != ATT1_OK)) {
        fputs("runtime tile state read failed\n", stderr);
        att1_runtime_destroy(&runtime);
        return 0;
    }

    if ((tile0.counters.commands_processed != 1u) ||
        (tile0.counters.run_layer_range_commands != 1u) ||
        (tile0.last_layer_start != 2u) ||
        (tile0.last_layer_end != 5u) ||
        (tile1.counters.commands_processed != 0u)) {
        fputs("runtime dispatch counters failed\n", stderr);
        att1_runtime_destroy(&runtime);
        return 0;
    }

    if (att1_runtime_send_command(&runtime, 9u, &command) != ATT1_ERR_INVALID) {
        fputs("runtime invalid tile command failed\n", stderr);
        att1_runtime_destroy(&runtime);
        return 0;
    }

    if ((att1_runtime_stop(&runtime) != ATT1_OK) ||
        (att1_runtime_stop(&runtime) != ATT1_OK)) {
        fputs("runtime stop idempotence failed\n", stderr);
        att1_runtime_destroy(&runtime);
        return 0;
    }

    if (att1_runtime_start(&runtime) != ATT1_OK) {
        fputs("runtime restart failed\n", stderr);
        att1_runtime_destroy(&runtime);
        return 0;
    }

    if (att1_runtime_stop(&runtime) != ATT1_OK) {
        fputs("runtime second stop failed\n", stderr);
        att1_runtime_destroy(&runtime);
        return 0;
    }

    att1_runtime_destroy(&runtime);
    return 1;
}

static int test_shutdown_and_fabric(void)
{
    const att1_runtime_config config = {
        .tile_count = 2u,
        .command_queue_capacity = 4u,
        .fabric_queue_capacity = 4u,
        .fabric_max_payload_bytes = ATT1_RUNTIME_MAX_COMMAND_PAYLOAD
    };
    att1_runtime_command command;
    att1_tile_state tile0;
    att1_tile_state tile1;
    att1_fabric_counters fabric_counters;
    att1_runtime runtime;

    if (att1_runtime_create(&runtime, &config) != ATT1_OK) {
        fputs("runtime fabric create failed\n", stderr);
        return 0;
    }

    if (att1_runtime_start(&runtime) != ATT1_OK) {
        fputs("runtime fabric start failed\n", stderr);
        att1_runtime_destroy(&runtime);
        return 0;
    }

    memset(&command, 0, sizeof(command));
    command.type = ATT1_RUNTIME_CMD_SEND_ACTIVATION;
    command.peer_tile = 1u;
    command.payload_bytes = 3u;
    command.payload[0] = 1u;
    command.payload[1] = 2u;
    command.payload[2] = 3u;

    if (att1_runtime_send_command(&runtime, 0u, &command) != ATT1_OK) {
        fputs("runtime send activation command failed\n", stderr);
        att1_runtime_destroy(&runtime);
        return 0;
    }

    if (!wait_for_commands(&runtime, 0u, 1u)) {
        fputs("runtime send activation not processed\n", stderr);
        att1_runtime_destroy(&runtime);
        return 0;
    }

    memset(&command, 0, sizeof(command));
    command.type = ATT1_RUNTIME_CMD_RECV_ACTIVATION;
    if (att1_runtime_send_command(&runtime, 1u, &command) != ATT1_OK) {
        fputs("runtime recv activation command failed\n", stderr);
        att1_runtime_destroy(&runtime);
        return 0;
    }

    if (!wait_for_commands(&runtime, 1u, 1u)) {
        fputs("runtime fabric commands not processed\n", stderr);
        att1_runtime_destroy(&runtime);
        return 0;
    }

    att1_runtime_get_fabric_counters(&runtime, &fabric_counters);
    if ((fabric_counters.packets_sent != 1u) ||
        (fabric_counters.packets_received != 1u)) {
        fputs("runtime fabric counter integration failed\n", stderr);
        att1_runtime_destroy(&runtime);
        return 0;
    }

    if ((att1_runtime_get_tile_state(&runtime, 0u, &tile0) != ATT1_OK) ||
        (att1_runtime_get_tile_state(&runtime, 1u, &tile1) != ATT1_OK) ||
        (tile0.counters.fabric_send_ops != 1u) ||
        (tile1.counters.fabric_recv_ops != 1u)) {
        fputs("runtime per-tile fabric counters failed\n", stderr);
        att1_runtime_destroy(&runtime);
        return 0;
    }

    memset(&command, 0, sizeof(command));
    command.type = ATT1_RUNTIME_CMD_SHUTDOWN;
    if ((att1_runtime_send_command(&runtime, 0u, &command) != ATT1_OK) ||
        (att1_runtime_send_command(&runtime, 1u, &command) != ATT1_OK)) {
        fputs("runtime shutdown command send failed\n", stderr);
        att1_runtime_destroy(&runtime);
        return 0;
    }

    if (!wait_for_running(&runtime, 0u, 0) ||
        !wait_for_running(&runtime, 1u, 0)) {
        fputs("runtime shutdown command did not stop workers\n", stderr);
        att1_runtime_destroy(&runtime);
        return 0;
    }

    if ((att1_runtime_get_tile_state(&runtime, 0u, &tile0) != ATT1_OK) ||
        (att1_runtime_get_tile_state(&runtime, 1u, &tile1) != ATT1_OK) ||
        (tile0.counters.shutdown_commands != 1u) ||
        (tile1.counters.shutdown_commands != 1u)) {
        fputs("runtime shutdown command counters failed\n", stderr);
        att1_runtime_destroy(&runtime);
        return 0;
    }

    if (att1_runtime_stop(&runtime) != ATT1_OK) {
        fputs("runtime stop after shutdown failed\n", stderr);
        att1_runtime_destroy(&runtime);
        return 0;
    }

    att1_runtime_destroy(&runtime);
    return 1;
}

static int test_barrier_and_queue_capacity(void)
{
    const att1_runtime_config config = {
        .tile_count = 2u,
        .command_queue_capacity = 1u,
        .fabric_queue_capacity = 2u,
        .fabric_max_payload_bytes = ATT1_RUNTIME_MAX_COMMAND_PAYLOAD
    };
    att1_runtime_command command;
    att1_fabric_counters fabric_counters;
    att1_runtime runtime;

    if (att1_runtime_create(&runtime, &config) != ATT1_OK) {
        fputs("runtime barrier create failed\n", stderr);
        return 0;
    }

    if (att1_runtime_start(&runtime) != ATT1_OK) {
        fputs("runtime barrier start failed\n", stderr);
        att1_runtime_destroy(&runtime);
        return 0;
    }

    memset(&command, 0, sizeof(command));
    command.type = ATT1_RUNTIME_CMD_SYNC_BARRIER;
    if (att1_runtime_send_command(&runtime, 0u, &command) != ATT1_OK) {
        fputs("runtime missing barrier send failed\n", stderr);
        att1_runtime_destroy(&runtime);
        return 0;
    }

    if (!wait_for_commands(&runtime, 0u, 1u)) {
        fputs("runtime missing barrier did not arrive\n", stderr);
        att1_runtime_destroy(&runtime);
        return 0;
    }

    att1_runtime_get_fabric_counters(&runtime, &fabric_counters);
    if (fabric_counters.barrier_completions != 0u) {
        fputs("runtime missing participant falsely completed barrier\n", stderr);
        att1_runtime_destroy(&runtime);
        return 0;
    }

    memset(&command, 0, sizeof(command));
    command.type = ATT1_RUNTIME_CMD_RUN_LAYER_RANGE;
    if ((att1_runtime_send_command(&runtime, 0u, &command) != ATT1_OK) ||
        (att1_runtime_send_command(&runtime, 0u, &command) != ATT1_ERR_QUEUE_FULL)) {
        fputs("runtime command queue capacity failed\n", stderr);
        att1_runtime_destroy(&runtime);
        return 0;
    }

    memset(&command, 0, sizeof(command));
    command.type = ATT1_RUNTIME_CMD_SYNC_BARRIER;
    if (att1_runtime_send_command(&runtime, 1u, &command) != ATT1_OK) {
        fputs("runtime completing barrier send failed\n", stderr);
        att1_runtime_destroy(&runtime);
        return 0;
    }

    if (!wait_for_commands(&runtime, 1u, 1u) ||
        !wait_for_commands(&runtime, 0u, 2u)) {
        fputs("runtime barrier completion did not resume tiles\n", stderr);
        att1_runtime_destroy(&runtime);
        return 0;
    }

    att1_runtime_get_fabric_counters(&runtime, &fabric_counters);
    if ((fabric_counters.barrier_arrivals != 2u) ||
        (fabric_counters.barrier_completions != 1u)) {
        fputs("runtime barrier counters failed\n", stderr);
        att1_runtime_destroy(&runtime);
        return 0;
    }

    att1_runtime_destroy(&runtime);
    return 1;
}

int main(void)
{
    if (!test_lifecycle_and_dispatch()) {
        return 1;
    }

    if (!test_shutdown_and_fabric()) {
        return 1;
    }

    if (!test_barrier_and_queue_capacity()) {
        return 1;
    }

    puts("runtime test passed");
    return 0;
}
