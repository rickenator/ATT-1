#include "att1_trace.h"

#include <stdlib.h>
#include <string.h>
#include <time.h>

struct att1_trace {
    size_t layer_count;
    size_t tile_count;
    size_t token_count;
    size_t token_capacity;
    uint64_t *token_time_us;
    att1_trace_counters counters;
    att1_trace_layer *layers;
    att1_trace_tile *tiles;
};

static int grow_tokens(att1_trace_t *trace)
{
    size_t next_capacity = 0u;
    uint64_t *next = NULL;

    if (trace->token_count < trace->token_capacity) {
        return 0;
    }

    next_capacity = trace->token_capacity == 0u ?
        8u : trace->token_capacity * 2u;
    if (next_capacity < trace->token_capacity) {
        return -1;
    }

    next = realloc(trace->token_time_us, next_capacity * sizeof(*next));
    if (next == NULL) {
        return -1;
    }

    trace->token_time_us = next;
    trace->token_capacity = next_capacity;
    return 0;
}

att1_status_t att1_trace_create(size_t layer_count,
                                size_t tile_count,
                                att1_trace_t **out_trace)
{
    att1_trace_t *trace = NULL;

    if ((out_trace == NULL) || (layer_count == 0u)) {
        return ATT1_ERR_INVALID_ARG;
    }

    *out_trace = NULL;
    trace = calloc(1u, sizeof(*trace));
    if (trace == NULL) {
        return ATT1_ERR_OOM;
    }

    trace->layers = calloc(layer_count, sizeof(*trace->layers));
    if (trace->layers == NULL) {
        att1_trace_destroy(trace);
        return ATT1_ERR_OOM;
    }

    if (tile_count != 0u) {
        trace->tiles = calloc(tile_count, sizeof(*trace->tiles));
        if (trace->tiles == NULL) {
            att1_trace_destroy(trace);
            return ATT1_ERR_OOM;
        }
    }

    trace->layer_count = layer_count;
    trace->tile_count = tile_count;
    *out_trace = trace;
    return ATT1_OK;
}

void att1_trace_destroy(att1_trace_t *trace)
{
    if (trace == NULL) {
        return;
    }

    free(trace->token_time_us);
    free(trace->layers);
    free(trace->tiles);
    free(trace);
}

void att1_trace_reset(att1_trace_t *trace)
{
    if (trace == NULL) {
        return;
    }

    trace->token_count = 0u;
    memset(&trace->counters, 0, sizeof(trace->counters));
    memset(trace->layers, 0, trace->layer_count * sizeof(*trace->layers));
    if (trace->tiles != NULL) {
        memset(trace->tiles, 0, trace->tile_count * sizeof(*trace->tiles));
    }
}

att1_status_t att1_trace_snapshot(const att1_trace_t *trace,
                                  att1_trace_counters *out_counters)
{
    if ((trace == NULL) || (out_counters == NULL)) {
        return ATT1_ERR_INVALID_ARG;
    }

    *out_counters = trace->counters;
    return ATT1_OK;
}

att1_status_t att1_trace_layer_snapshot(const att1_trace_t *trace,
                                        size_t layer_id,
                                        att1_trace_layer *out_layer)
{
    if ((trace == NULL) || (out_layer == NULL) ||
        (layer_id >= trace->layer_count)) {
        return ATT1_ERR_INVALID_ARG;
    }

    *out_layer = trace->layers[layer_id];
    return ATT1_OK;
}

att1_status_t att1_trace_tile_snapshot(const att1_trace_t *trace,
                                       size_t tile_id,
                                       att1_trace_tile *out_tile)
{
    if ((trace == NULL) || (out_tile == NULL) ||
        (tile_id >= trace->tile_count) || (trace->tiles == NULL)) {
        return ATT1_ERR_INVALID_ARG;
    }

    *out_tile = trace->tiles[tile_id];
    return ATT1_OK;
}

att1_status_t att1_trace_token_time_us(const att1_trace_t *trace,
                                       size_t token_index,
                                       uint64_t *out_time_us)
{
    if ((trace == NULL) || (out_time_us == NULL) ||
        (token_index >= trace->token_count)) {
        return ATT1_ERR_INVALID_ARG;
    }

    *out_time_us = trace->token_time_us[token_index];
    return ATT1_OK;
}

size_t att1_trace_layer_count(const att1_trace_t *trace)
{
    return trace != NULL ? trace->layer_count : 0u;
}

size_t att1_trace_tile_count(const att1_trace_t *trace)
{
    return trace != NULL ? trace->tile_count : 0u;
}

size_t att1_trace_token_count(const att1_trace_t *trace)
{
    return trace != NULL ? trace->token_count : 0u;
}

uint64_t att1_trace_now_us(void)
{
    const clock_t now = clock();

    if (now == (clock_t)-1) {
        return 0u;
    }

    return ((uint64_t)now * 1000000u) / (uint64_t)CLOCKS_PER_SEC;
}

void att1_trace_record_token(att1_trace_t *trace, uint64_t time_us)
{
    if (trace == NULL) {
        return;
    }

    if (grow_tokens(trace) == 0) {
        trace->token_time_us[trace->token_count] = time_us;
        trace->token_count++;
    }

    trace->counters.tokens_decoded++;
    trace->counters.token_time_us_total += time_us;
    if (time_us > trace->counters.token_time_us_max) {
        trace->counters.token_time_us_max = time_us;
    }
}

void att1_trace_record_layer(att1_trace_t *trace,
                             size_t layer_id,
                             uint64_t time_us,
                             uint64_t kv_appends,
                             uint64_t kv_key_reads,
                             uint64_t kv_value_reads)
{
    att1_trace_layer *layer = NULL;

    if ((trace == NULL) || (layer_id >= trace->layer_count)) {
        return;
    }

    layer = &trace->layers[layer_id];
    layer->executions++;
    layer->time_us_total += time_us;
    layer->kv_appends += kv_appends;
    layer->kv_key_reads += kv_key_reads;
    layer->kv_value_reads += kv_value_reads;

    trace->counters.layer_time_us_total += time_us;
    trace->counters.kv_appends += kv_appends;
    trace->counters.kv_key_reads += kv_key_reads;
    trace->counters.kv_value_reads += kv_value_reads;
}

void att1_trace_record_activation_send(att1_trace_t *trace,
                                       size_t tile_id,
                                       uint64_t bytes)
{
    if (trace == NULL) {
        return;
    }

    trace->counters.activation_bytes_sent += bytes;
    if ((trace->tiles != NULL) && (tile_id < trace->tile_count)) {
        trace->tiles[tile_id].activation_bytes_sent += bytes;
    }
}

void att1_trace_record_logits(att1_trace_t *trace,
                              size_t tile_id,
                              uint64_t bytes)
{
    if (trace == NULL) {
        return;
    }

    trace->counters.logits_bytes_produced += bytes;
    if ((trace->tiles != NULL) && (tile_id < trace->tile_count)) {
        trace->tiles[tile_id].logits_bytes_produced += bytes;
    }
}

void att1_trace_record_fabric_send(att1_trace_t *trace,
                                   uint64_t payload_bytes)
{
    if (trace == NULL) {
        return;
    }

    trace->counters.fabric_packets_sent++;
    trace->counters.fabric_payload_bytes_sent += payload_bytes;
}

void att1_trace_record_fabric_receive(att1_trace_t *trace,
                                      uint64_t payload_bytes)
{
    if (trace == NULL) {
        return;
    }

    trace->counters.fabric_packets_received++;
    trace->counters.fabric_payload_bytes_received += payload_bytes;
}

void att1_trace_record_tile_layers(att1_trace_t *trace,
                                   size_t tile_id,
                                   uint64_t layer_count)
{
    if (trace == NULL) {
        return;
    }

    trace->counters.tile_layer_executions += layer_count;
    if ((trace->tiles != NULL) && (tile_id < trace->tile_count)) {
        trace->tiles[tile_id].layer_executions += layer_count;
    }
}
