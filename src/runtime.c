#include "runtime_internal.h"

#include <stdlib.h>
#include <string.h>

static int att1_runtime_config_valid(const att1_runtime_config *config)
{
    if (config == NULL) {
        return 0;
    }

    if ((config->tile_count == 0u) ||
        (config->command_queue_capacity == 0u) ||
        (config->fabric_queue_capacity == 0u) ||
        (config->fabric_max_payload_bytes == 0u)) {
        return 0;
    }

    if (config->tile_count > (size_t)UINT32_MAX) {
        return 0;
    }

    return 1;
}

static att1_runtime_impl *att1_runtime_impl_get(att1_runtime *runtime)
{
    if (runtime == NULL) {
        return NULL;
    }

    return (att1_runtime_impl *)runtime->impl;
}

static int att1_runtime_valid_tile(const att1_runtime *runtime, uint32_t tile_id)
{
    return (runtime != NULL) && ((size_t)tile_id < runtime->config.tile_count);
}

static void att1_runtime_queue_clear(att1_runtime_queue *queue)
{
    queue->head = 0u;
    queue->count = 0u;
}

static void att1_runtime_queue_destroy(att1_runtime_queue *queue)
{
    if (queue == NULL) {
        return;
    }

    pthread_cond_destroy(&queue->cond);
    pthread_mutex_destroy(&queue->mutex);
    free(queue->commands);
    memset(queue, 0, sizeof(*queue));
}

static int att1_runtime_queue_init(att1_runtime_queue *queue, size_t capacity)
{
    memset(queue, 0, sizeof(*queue));

    queue->commands = calloc(capacity, sizeof(*queue->commands));
    if (queue->commands == NULL) {
        return ATT1_ERR_OOM;
    }

    if (pthread_mutex_init(&queue->mutex, NULL) != 0) {
        free(queue->commands);
        memset(queue, 0, sizeof(*queue));
        return ATT1_ERR_INVALID_ARG;
    }

    if (pthread_cond_init(&queue->cond, NULL) != 0) {
        pthread_mutex_destroy(&queue->mutex);
        free(queue->commands);
        memset(queue, 0, sizeof(*queue));
        return ATT1_ERR_INVALID_ARG;
    }

    return ATT1_OK;
}

void att1_runtime_mark_tile_running(att1_runtime *runtime,
                                    uint32_t tile_id,
                                    int running)
{
    att1_runtime_impl *impl = att1_runtime_impl_get(runtime);

    if ((impl == NULL) || !att1_runtime_valid_tile(runtime, tile_id)) {
        return;
    }

    pthread_mutex_lock(&impl->state_mutexes[tile_id]);
    runtime->tiles[tile_id].running = running;
    pthread_mutex_unlock(&impl->state_mutexes[tile_id]);
}

void att1_runtime_count_command(att1_runtime *runtime,
                                uint32_t tile_id,
                                att1_runtime_command_type type)
{
    att1_runtime_impl *impl = att1_runtime_impl_get(runtime);
    att1_tile_counters *counters = NULL;

    if ((impl == NULL) || !att1_runtime_valid_tile(runtime, tile_id)) {
        return;
    }

    pthread_mutex_lock(&impl->state_mutexes[tile_id]);
    counters = &runtime->tiles[tile_id].counters;
    counters->commands_processed++;

    switch (type) {
    case ATT1_RUNTIME_CMD_LOAD_MODEL_SHARD:
        counters->load_model_shard_commands++;
        break;
    case ATT1_RUNTIME_CMD_RUN_LAYER_RANGE:
        counters->run_layer_range_commands++;
        break;
    case ATT1_RUNTIME_CMD_SEND_ACTIVATION:
        counters->send_activation_commands++;
        break;
    case ATT1_RUNTIME_CMD_RECV_ACTIVATION:
        counters->recv_activation_commands++;
        break;
    case ATT1_RUNTIME_CMD_SYNC_BARRIER:
        counters->barrier_commands++;
        break;
    case ATT1_RUNTIME_CMD_SHUTDOWN:
        counters->shutdown_commands++;
        break;
    default:
        counters->errors++;
        break;
    }

    pthread_mutex_unlock(&impl->state_mutexes[tile_id]);
}

void att1_runtime_count_error(att1_runtime *runtime, uint32_t tile_id)
{
    att1_runtime_impl *impl = att1_runtime_impl_get(runtime);

    if ((impl == NULL) || !att1_runtime_valid_tile(runtime, tile_id)) {
        return;
    }

    pthread_mutex_lock(&impl->state_mutexes[tile_id]);
    runtime->tiles[tile_id].counters.errors++;
    pthread_mutex_unlock(&impl->state_mutexes[tile_id]);
}

void att1_runtime_count_fabric_send(att1_runtime *runtime, uint32_t tile_id)
{
    att1_runtime_impl *impl = att1_runtime_impl_get(runtime);

    if ((impl == NULL) || !att1_runtime_valid_tile(runtime, tile_id)) {
        return;
    }

    pthread_mutex_lock(&impl->state_mutexes[tile_id]);
    runtime->tiles[tile_id].counters.fabric_send_ops++;
    pthread_mutex_unlock(&impl->state_mutexes[tile_id]);
}

void att1_runtime_count_fabric_recv(att1_runtime *runtime, uint32_t tile_id)
{
    att1_runtime_impl *impl = att1_runtime_impl_get(runtime);

    if ((impl == NULL) || !att1_runtime_valid_tile(runtime, tile_id)) {
        return;
    }

    pthread_mutex_lock(&impl->state_mutexes[tile_id]);
    runtime->tiles[tile_id].counters.fabric_recv_ops++;
    pthread_mutex_unlock(&impl->state_mutexes[tile_id]);
}

static void att1_runtime_reset_barrier(att1_runtime_impl *impl)
{
    impl->barrier_active = 0;
    impl->barrier_expected = 0u;
    impl->barrier_arrived = 0u;
    impl->barrier_generation++;
    pthread_cond_broadcast(&impl->barrier_cond);
}

int att1_runtime_execute_barrier(att1_runtime *runtime, uint32_t tile_id)
{
    att1_runtime_impl *impl = att1_runtime_impl_get(runtime);
    int complete = 0;
    uint32_t fabric_group_stack[1] = {0u};
    uint32_t *fabric_group = NULL;
    size_t i = 0u;
    uint64_t generation = 0u;

    if ((impl == NULL) || !att1_runtime_valid_tile(runtime, tile_id)) {
        return ATT1_ERR_INVALID_ARG;
    }

    fabric_group = malloc(runtime->config.tile_count * sizeof(*fabric_group));
    if (fabric_group == NULL) {
        fabric_group = (uint32_t *)fabric_group_stack;
        if (runtime->config.tile_count != 1u) {
            att1_runtime_count_error(runtime, tile_id);
            return ATT1_ERR_OOM;
        }
    }

    for (i = 0u; i < runtime->config.tile_count; i++) {
        fabric_group[i] = (uint32_t)i;
    }

    pthread_mutex_lock(&impl->fabric_mutex);
    (void)att1_fabric_barrier_arrive(&runtime->fabric,
                                     tile_id,
                                     fabric_group,
                                     runtime->config.tile_count,
                                     &complete);
    pthread_mutex_unlock(&impl->fabric_mutex);

    if (fabric_group != fabric_group_stack) {
        free(fabric_group);
    }

    pthread_mutex_lock(&impl->barrier_mutex);
    if (impl->barrier_active == 0) {
        impl->barrier_active = 1;
        impl->barrier_expected = runtime->config.tile_count;
        impl->barrier_arrived = 0u;
    }

    generation = impl->barrier_generation;
    impl->barrier_arrived++;

    if (impl->barrier_arrived == impl->barrier_expected) {
        att1_runtime_reset_barrier(impl);
        pthread_mutex_unlock(&impl->barrier_mutex);
        return ATT1_OK;
    }

    while ((impl->stop_requested == 0) &&
           (impl->barrier_active != 0) &&
           (impl->barrier_generation == generation)) {
        pthread_cond_wait(&impl->barrier_cond, &impl->barrier_mutex);
    }

    if (impl->stop_requested != 0) {
        pthread_mutex_unlock(&impl->barrier_mutex);
        att1_runtime_count_error(runtime, tile_id);
        return ATT1_ERR_INVALID_ARG;
    }

    pthread_mutex_unlock(&impl->barrier_mutex);
    return ATT1_OK;
}

int att1_runtime_create(att1_runtime *runtime,
                        const att1_runtime_config *config)
{
    att1_runtime_impl *impl = NULL;
    att1_fabric_bus_config fabric_config;
    size_t i = 0u;

    if ((runtime == NULL) || !att1_runtime_config_valid(config)) {
        return ATT1_ERR_INVALID_ARG;
    }

    memset(runtime, 0, sizeof(*runtime));
    runtime->config = *config;

    impl = calloc(1u, sizeof(*impl));
    runtime->tiles = calloc(config->tile_count, sizeof(*runtime->tiles));
    if ((impl == NULL) || (runtime->tiles == NULL)) {
        free(impl);
        free(runtime->tiles);
        memset(runtime, 0, sizeof(*runtime));
        return ATT1_ERR_OOM;
    }

    runtime->impl = impl;
    impl->runtime = runtime;
    impl->threads = calloc(config->tile_count, sizeof(*impl->threads));
    impl->queues = calloc(config->tile_count, sizeof(*impl->queues));
    impl->state_mutexes = calloc(config->tile_count, sizeof(*impl->state_mutexes));
    if ((impl->threads == NULL) ||
        (impl->queues == NULL) ||
        (impl->state_mutexes == NULL)) {
        att1_runtime_destroy(runtime);
        return ATT1_ERR_OOM;
    }

    if ((pthread_mutex_init(&impl->fabric_mutex, NULL) != 0) ||
        (pthread_mutex_init(&impl->barrier_mutex, NULL) != 0) ||
        (pthread_cond_init(&impl->barrier_cond, NULL) != 0)) {
        att1_runtime_destroy(runtime);
        return ATT1_ERR_INVALID_ARG;
    }

    for (i = 0u; i < config->tile_count; i++) {
        att1_tile_state_init(&runtime->tiles[i], (uint32_t)i);
        if (pthread_mutex_init(&impl->state_mutexes[i], NULL) != 0) {
            att1_runtime_destroy(runtime);
            return ATT1_ERR_INVALID_ARG;
        }

        if (att1_runtime_queue_init(&impl->queues[i],
                                    config->command_queue_capacity) != ATT1_OK) {
            att1_runtime_destroy(runtime);
            return ATT1_ERR_OOM;
        }
    }

    fabric_config.tile_count = config->tile_count;
    fabric_config.queue_capacity = config->fabric_queue_capacity;
    fabric_config.max_payload_bytes = config->fabric_max_payload_bytes;
    if (att1_fabric_create(&runtime->fabric, &fabric_config) != ATT1_OK) {
        att1_runtime_destroy(runtime);
        return ATT1_ERR_OOM;
    }

    return ATT1_OK;
}

void att1_runtime_destroy(att1_runtime *runtime)
{
    att1_runtime_impl *impl = att1_runtime_impl_get(runtime);
    size_t i = 0u;

    if (runtime == NULL) {
        return;
    }

    (void)att1_runtime_stop(runtime);

    if (impl != NULL) {
        if (impl->queues != NULL) {
            for (i = 0u; i < runtime->config.tile_count; i++) {
                att1_runtime_queue_destroy(&impl->queues[i]);
            }
        }

        if (impl->state_mutexes != NULL) {
            for (i = 0u; i < runtime->config.tile_count; i++) {
                pthread_mutex_destroy(&impl->state_mutexes[i]);
            }
        }

        pthread_cond_destroy(&impl->barrier_cond);
        pthread_mutex_destroy(&impl->barrier_mutex);
        pthread_mutex_destroy(&impl->fabric_mutex);
        free(impl->threads);
        free(impl->queues);
        free(impl->state_mutexes);
        free(impl);
    }

    att1_fabric_destroy(&runtime->fabric);
    free(runtime->tiles);
    memset(runtime, 0, sizeof(*runtime));
}

int att1_runtime_start(att1_runtime *runtime)
{
    att1_runtime_impl *impl = att1_runtime_impl_get(runtime);
    size_t i = 0u;

    if (impl == NULL) {
        return ATT1_ERR_INVALID_ARG;
    }

    if (impl->threads_started != 0) {
        return ATT1_ERR_ALREADY_STARTED;
    }

    impl->stop_requested = 0;
    impl->barrier_active = 0;
    impl->barrier_expected = 0u;
    impl->barrier_arrived = 0u;
    impl->barrier_generation++;

    for (i = 0u; i < runtime->config.tile_count; i++) {
        pthread_mutex_lock(&impl->queues[i].mutex);
        att1_runtime_queue_clear(&impl->queues[i]);
        pthread_mutex_unlock(&impl->queues[i].mutex);
        att1_runtime_mark_tile_running(runtime, (uint32_t)i, 1);
    }

    for (i = 0u; i < runtime->config.tile_count; i++) {
        att1_tile_thread_arg *arg = malloc(sizeof(*arg));
        if (arg == NULL) {
            (void)att1_runtime_stop(runtime);
            return ATT1_ERR_OOM;
        }

        arg->runtime = runtime;
        arg->tile_id = (uint32_t)i;
        if (pthread_create(&impl->threads[i],
                           NULL,
                           att1_sim_tile_thread_main,
                           arg) != 0) {
            free(arg);
            (void)att1_runtime_stop(runtime);
            return ATT1_ERR_INVALID_ARG;
        }
    }

    impl->threads_started = 1;
    return ATT1_OK;
}

int att1_runtime_stop(att1_runtime *runtime)
{
    att1_runtime_impl *impl = att1_runtime_impl_get(runtime);
    size_t i = 0u;

    if (impl == NULL) {
        return ATT1_OK;
    }

    if (impl->threads_started == 0) {
        return ATT1_OK;
    }

    impl->stop_requested = 1;

    for (i = 0u; i < runtime->config.tile_count; i++) {
        pthread_mutex_lock(&impl->queues[i].mutex);
        pthread_cond_broadcast(&impl->queues[i].cond);
        pthread_mutex_unlock(&impl->queues[i].mutex);
    }

    pthread_mutex_lock(&impl->barrier_mutex);
    pthread_cond_broadcast(&impl->barrier_cond);
    pthread_mutex_unlock(&impl->barrier_mutex);

    for (i = 0u; i < runtime->config.tile_count; i++) {
        pthread_join(impl->threads[i], NULL);
        att1_runtime_mark_tile_running(runtime, (uint32_t)i, 0);
    }

    impl->threads_started = 0;
    impl->stop_requested = 0;
    return ATT1_OK;
}

int att1_runtime_send_command(att1_runtime *runtime,
                              uint32_t tile_id,
                              const att1_runtime_command *command)
{
    att1_runtime_impl *impl = att1_runtime_impl_get(runtime);
    att1_runtime_queue *queue = NULL;
    size_t slot = 0u;

    if ((impl == NULL) || (command == NULL)) {
        return ATT1_ERR_INVALID_ARG;
    }

    if (impl->threads_started == 0) {
        return ATT1_ERR_INVALID_ARG;
    }

    if (!att1_runtime_valid_tile(runtime, tile_id)) {
        return ATT1_ERR_INVALID_ARG;
    }

    if (command->payload_bytes > ATT1_RUNTIME_MAX_COMMAND_PAYLOAD) {
        return ATT1_ERR_INVALID_ARG;
    }

    queue = &impl->queues[tile_id];
    pthread_mutex_lock(&queue->mutex);
    if (queue->count == runtime->config.command_queue_capacity) {
        pthread_mutex_unlock(&queue->mutex);
        return ATT1_ERR_QUEUE_FULL;
    }

    slot = (queue->head + queue->count) % runtime->config.command_queue_capacity;
    queue->commands[slot] = *command;
    queue->count++;
    pthread_cond_signal(&queue->cond);
    pthread_mutex_unlock(&queue->mutex);
    return ATT1_OK;
}

int att1_runtime_get_tile_state(att1_runtime *runtime,
                                uint32_t tile_id,
                                att1_tile_state *out_state)
{
    att1_runtime_impl *impl = att1_runtime_impl_get(runtime);

    if ((impl == NULL) || (out_state == NULL) ||
        !att1_runtime_valid_tile(runtime, tile_id)) {
        return ATT1_ERR_INVALID_ARG;
    }

    pthread_mutex_lock(&impl->state_mutexes[tile_id]);
    *out_state = runtime->tiles[tile_id];
    pthread_mutex_unlock(&impl->state_mutexes[tile_id]);
    return ATT1_OK;
}

void att1_runtime_get_fabric_counters(att1_runtime *runtime,
                                      att1_fabric_counters *out_counters)
{
    att1_runtime_impl *impl = att1_runtime_impl_get(runtime);

    if ((impl == NULL) || (out_counters == NULL)) {
        return;
    }

    pthread_mutex_lock(&impl->fabric_mutex);
    att1_fabric_get_counters(&runtime->fabric, out_counters);
    pthread_mutex_unlock(&impl->fabric_mutex);
}
