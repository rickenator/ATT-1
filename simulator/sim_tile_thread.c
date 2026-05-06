#include "../src/runtime_internal.h"

#include <stdlib.h>

static int att1_runtime_dequeue(att1_runtime_impl *impl,
                                uint32_t tile_id,
                                att1_runtime_command *out_command)
{
    att1_runtime *runtime = impl->runtime;
    att1_runtime_queue *queue = &impl->queues[tile_id];

    pthread_mutex_lock(&queue->mutex);
    while ((queue->count == 0u) && (impl->stop_requested == 0)) {
        pthread_cond_wait(&queue->cond, &queue->mutex);
    }

    if ((queue->count == 0u) && (impl->stop_requested != 0)) {
        pthread_mutex_unlock(&queue->mutex);
        return 0;
    }

    *out_command = queue->commands[queue->head];
    queue->head = (queue->head + 1u) % runtime->config.command_queue_capacity;
    queue->count--;
    pthread_mutex_unlock(&queue->mutex);
    return 1;
}

static void att1_runtime_apply_model_shard(att1_runtime *runtime,
                                           uint32_t tile_id,
                                           uint64_t model_shard_id)
{
    att1_runtime_impl *impl = (att1_runtime_impl *)runtime->impl;

    pthread_mutex_lock(&impl->state_mutexes[tile_id]);
    runtime->tiles[tile_id].model_shard_id = model_shard_id;
    pthread_mutex_unlock(&impl->state_mutexes[tile_id]);
}

static void att1_runtime_apply_layer_range(att1_runtime *runtime,
                                           uint32_t tile_id,
                                           uint32_t layer_start,
                                           uint32_t layer_end)
{
    att1_runtime_impl *impl = (att1_runtime_impl *)runtime->impl;

    pthread_mutex_lock(&impl->state_mutexes[tile_id]);
    runtime->tiles[tile_id].last_layer_start = layer_start;
    runtime->tiles[tile_id].last_layer_end = layer_end;
    pthread_mutex_unlock(&impl->state_mutexes[tile_id]);
}

static void att1_runtime_handle_send_activation(att1_runtime *runtime,
                                                uint32_t tile_id,
                                                const att1_runtime_command *command)
{
    att1_runtime_impl *impl = (att1_runtime_impl *)runtime->impl;
    const int rc_payload = (command->payload_bytes <=
                            runtime->config.fabric_max_payload_bytes);
    int rc = ATT1_ERR_INVALID;

    if (rc_payload != 0) {
        pthread_mutex_lock(&impl->fabric_mutex);
        rc = att1_fabric_send(&runtime->fabric,
                              tile_id,
                              command->peer_tile,
                              ATT1_PACKET_ACTIVATION,
                              command->payload,
                              command->payload_bytes,
                              command->tag);
        pthread_mutex_unlock(&impl->fabric_mutex);
    }

    if (rc == ATT1_OK) {
        att1_runtime_count_fabric_send(runtime, tile_id);
    } else {
        att1_runtime_count_error(runtime, tile_id);
    }
}

static void att1_runtime_handle_recv_activation(att1_runtime *runtime,
                                                uint32_t tile_id)
{
    att1_runtime_impl *impl = (att1_runtime_impl *)runtime->impl;
    unsigned char payload[ATT1_RUNTIME_MAX_COMMAND_PAYLOAD];
    att1_fabric_packet packet;
    size_t payload_bytes = 0u;
    int rc = ATT1_ERR_INVALID;

    pthread_mutex_lock(&impl->fabric_mutex);
    rc = att1_fabric_receive(&runtime->fabric,
                             tile_id,
                             &packet,
                             payload,
                             sizeof(payload),
                             &payload_bytes);
    pthread_mutex_unlock(&impl->fabric_mutex);

    if (rc == ATT1_OK) {
        att1_runtime_count_fabric_recv(runtime, tile_id);
    } else {
        att1_runtime_count_error(runtime, tile_id);
    }
}

static int att1_runtime_process_command(att1_runtime *runtime,
                                        uint32_t tile_id,
                                        const att1_runtime_command *command)
{
    att1_runtime_count_command(runtime, tile_id, command->type);

    switch (command->type) {
    case ATT1_RUNTIME_CMD_LOAD_MODEL_SHARD:
        att1_runtime_apply_model_shard(runtime, tile_id, command->arg0);
        return 1;
    case ATT1_RUNTIME_CMD_RUN_LAYER_RANGE:
        att1_runtime_apply_layer_range(runtime,
                                       tile_id,
                                       command->layer_start,
                                       command->layer_end);
        return 1;
    case ATT1_RUNTIME_CMD_SEND_ACTIVATION:
        att1_runtime_handle_send_activation(runtime, tile_id, command);
        return 1;
    case ATT1_RUNTIME_CMD_RECV_ACTIVATION:
        att1_runtime_handle_recv_activation(runtime, tile_id);
        return 1;
    case ATT1_RUNTIME_CMD_SYNC_BARRIER:
        (void)att1_runtime_execute_barrier(runtime, tile_id);
        return 1;
    case ATT1_RUNTIME_CMD_SHUTDOWN:
        return 0;
    default:
        att1_runtime_count_error(runtime, tile_id);
        return 1;
    }
}

void *att1_sim_tile_thread_main(void *arg)
{
    att1_tile_thread_arg *thread_arg = (att1_tile_thread_arg *)arg;
    att1_runtime *runtime = thread_arg->runtime;
    att1_runtime_impl *impl = (att1_runtime_impl *)runtime->impl;
    const uint32_t tile_id = thread_arg->tile_id;
    int keep_running = 1;

    free(thread_arg);
    att1_runtime_mark_tile_running(runtime, tile_id, 1);

    while (keep_running != 0) {
        att1_runtime_command command;

        if (!att1_runtime_dequeue(impl, tile_id, &command)) {
            break;
        }

        keep_running = att1_runtime_process_command(runtime, tile_id, &command);
    }

    att1_runtime_mark_tile_running(runtime, tile_id, 0);
    return NULL;
}
