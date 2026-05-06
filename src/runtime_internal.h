#ifndef ATT1_RUNTIME_INTERNAL_H
#define ATT1_RUNTIME_INTERNAL_H

#include "att1_runtime.h"

#include <pthread.h>

typedef struct att1_runtime_queue {
    pthread_mutex_t mutex;
    pthread_cond_t cond;
    att1_runtime_command *commands;
    size_t head;
    size_t count;
} att1_runtime_queue;

typedef struct att1_runtime_impl {
    att1_runtime *runtime;
    pthread_t *threads;
    att1_runtime_queue *queues;
    pthread_mutex_t *state_mutexes;
    pthread_mutex_t fabric_mutex;
    pthread_mutex_t barrier_mutex;
    pthread_cond_t barrier_cond;
    size_t barrier_expected;
    size_t barrier_arrived;
    uint64_t barrier_generation;
    int barrier_active;
    int stop_requested;
    int threads_started;
} att1_runtime_impl;

typedef struct att1_tile_thread_arg {
    att1_runtime *runtime;
    uint32_t tile_id;
} att1_tile_thread_arg;

void *att1_sim_tile_thread_main(void *arg);

void att1_runtime_mark_tile_running(att1_runtime *runtime,
                                    uint32_t tile_id,
                                    int running);
void att1_runtime_count_command(att1_runtime *runtime,
                                uint32_t tile_id,
                                att1_runtime_command_type type);
void att1_runtime_count_error(att1_runtime *runtime, uint32_t tile_id);
void att1_runtime_count_fabric_send(att1_runtime *runtime, uint32_t tile_id);
void att1_runtime_count_fabric_recv(att1_runtime *runtime, uint32_t tile_id);
int att1_runtime_execute_barrier(att1_runtime *runtime, uint32_t tile_id);

#endif
