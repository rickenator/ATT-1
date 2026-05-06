#ifndef ATT1_CLUSTER_INFER_H
#define ATT1_CLUSTER_INFER_H

#include "att1_backend.h"
#include "att1_model.h"
#include "att1_shard.h"
#include "att1_status.h"
#include "att1_trace.h"

#include <stddef.h>
#include <stdint.h>

typedef struct att1_cluster_infer att1_cluster_infer_t;

typedef struct att1_cluster_infer_config {
    size_t tile_count;
    size_t fabric_queue_capacity;
    size_t fabric_max_payload_bytes;
} att1_cluster_infer_config;

typedef struct att1_cluster_tile_counters {
    uint64_t activations_received;
    uint64_t activations_sent;
    uint64_t layers_run;
    uint64_t logits_sent;
    uint32_t layer_start;
    uint32_t layer_end;
} att1_cluster_tile_counters;

/*
 * Create/destroy a synchronous multi-tile layer-sharded inference context.
 *
 * Compute tiles are numbered 0..tile_count-1. The host is modeled as fabric
 * endpoint tile_count and receives final LOGITS packets. The opaque context
 * owns all buffers, local KV caches, shard plan storage, and internal fabric.
 */
att1_status_t att1_cluster_infer_create(
    const att1_model *model,
    const att1_cluster_infer_config *config,
    att1_cluster_infer_t **out_infer);

void att1_cluster_infer_destroy(att1_cluster_infer_t *infer);

att1_status_t att1_cluster_infer_decode_token(att1_cluster_infer_t *infer,
                                              uint32_t token_id,
                                              uint32_t *out_token);

att1_status_t att1_cluster_infer_generate(att1_cluster_infer_t *infer,
                                          const unsigned char *prompt,
                                          size_t prompt_bytes,
                                          size_t generated_token_count,
                                          uint32_t *out_tokens,
                                          size_t out_token_capacity,
                                          size_t *out_token_count);

const float *att1_cluster_infer_logits(const att1_cluster_infer_t *infer,
                                       size_t *out_count);

att1_status_t att1_cluster_infer_position(const att1_cluster_infer_t *infer,
                                          size_t *out_position);

att1_status_t att1_cluster_infer_get_tile_counters(
    const att1_cluster_infer_t *infer,
    uint32_t tile_id,
    att1_cluster_tile_counters *out_counters);

att1_status_t att1_cluster_infer_get_tile_shard(
    const att1_cluster_infer_t *infer,
    uint32_t tile_id,
    att1_layer_shard *out_shard);

att1_status_t att1_cluster_infer_set_trace(att1_cluster_infer_t *infer,
                                           att1_trace_t *trace);

/*
 * Replace the inference backend. On success, infer takes ownership of backend.
 * Passing NULL is invalid.
 */
att1_status_t att1_cluster_infer_set_backend(att1_cluster_infer_t *infer,
                                             att1_backend *backend);

#endif
