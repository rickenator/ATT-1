#ifndef ATT1_TRACE_H
#define ATT1_TRACE_H

#include "att1_status.h"

#include <stddef.h>
#include <stdint.h>

typedef struct att1_trace att1_trace_t;

typedef struct att1_trace_counters {
    uint64_t tokens_decoded;
    uint64_t token_time_us_total;
    uint64_t token_time_us_max;
    uint64_t layer_time_us_total;
    uint64_t activation_bytes_sent;
    uint64_t logits_bytes_produced;
    uint64_t fabric_packets_sent;
    uint64_t fabric_packets_received;
    uint64_t fabric_payload_bytes_sent;
    uint64_t fabric_payload_bytes_received;
    uint64_t kv_appends;
    uint64_t kv_key_reads;
    uint64_t kv_value_reads;
    uint64_t tile_layer_executions;
} att1_trace_counters;

typedef struct att1_trace_layer {
    uint64_t executions;
    uint64_t time_us_total;
    uint64_t kv_appends;
    uint64_t kv_key_reads;
    uint64_t kv_value_reads;
} att1_trace_layer;

typedef struct att1_trace_tile {
    uint64_t layer_executions;
    uint64_t activation_bytes_sent;
    uint64_t logits_bytes_produced;
} att1_trace_tile;

att1_status_t att1_trace_create(size_t layer_count,
                                size_t tile_count,
                                att1_trace_t **out_trace);

void att1_trace_destroy(att1_trace_t *trace);
void att1_trace_reset(att1_trace_t *trace);

att1_status_t att1_trace_snapshot(const att1_trace_t *trace,
                                  att1_trace_counters *out_counters);

att1_status_t att1_trace_layer_snapshot(const att1_trace_t *trace,
                                        size_t layer_id,
                                        att1_trace_layer *out_layer);

att1_status_t att1_trace_tile_snapshot(const att1_trace_t *trace,
                                       size_t tile_id,
                                       att1_trace_tile *out_tile);

att1_status_t att1_trace_token_time_us(const att1_trace_t *trace,
                                       size_t token_index,
                                       uint64_t *out_time_us);

size_t att1_trace_layer_count(const att1_trace_t *trace);
size_t att1_trace_tile_count(const att1_trace_t *trace);
size_t att1_trace_token_count(const att1_trace_t *trace);

uint64_t att1_trace_now_us(void);

void att1_trace_record_token(att1_trace_t *trace, uint64_t time_us);
void att1_trace_record_layer(att1_trace_t *trace,
                             size_t layer_id,
                             uint64_t time_us,
                             uint64_t kv_appends,
                             uint64_t kv_key_reads,
                             uint64_t kv_value_reads);
void att1_trace_record_activation_send(att1_trace_t *trace,
                                       size_t tile_id,
                                       uint64_t bytes);
void att1_trace_record_logits(att1_trace_t *trace,
                              size_t tile_id,
                              uint64_t bytes);
void att1_trace_record_fabric_send(att1_trace_t *trace,
                                   uint64_t payload_bytes);
void att1_trace_record_fabric_receive(att1_trace_t *trace,
                                      uint64_t payload_bytes);
void att1_trace_record_tile_layers(att1_trace_t *trace,
                                   size_t tile_id,
                                   uint64_t layer_count);

#endif
